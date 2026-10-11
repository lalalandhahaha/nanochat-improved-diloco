"""CPU tests for paired worker-local DiLoCo training checkpoints."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from nanochat import checkpoint_manager
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint, save_rank_model
from nanochat.diloco import DiLoCoWrapper


class TestDiLoCoCheckpoints(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        logger_patch = patch.object(checkpoint_manager.logger, 'info')
        logger_patch.start()
        self.addCleanup(logger_patch.stop)

    def make_worker(self, rank, model_data=None, optimizer_data=None):
        model = torch.nn.Linear(1, 1, bias=False).double()
        model.register_buffer('worker_buffer', torch.tensor([rank]))
        with torch.no_grad():
            model.weight.fill_(1.0)
        if model_data is not None:
            # Match base_train: load the model before constructing the optimizer.
            model.load_state_dict(model_data, assign=True)
        inner = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.6)
        optimizer = DiLoCoWrapper([inner], model, H=4, outer_lr=0.75, outer_momentum=0.9)
        if optimizer_data is not None:
            optimizer.load_state_dict(optimizer_data)
        return model, optimizer

    def synchronize(self, workers):
        pseudo_grads = [opt.params_offloaded[0] - model.weight.detach() for model, opt in workers]
        mean = torch.stack(pseudo_grads).mean(dim=0)

        def average(tensor, op):
            self.assertEqual(op, torch.distributed.ReduceOp.AVG)
            tensor.copy_(mean)

        # Simulate only the collective: both workers use the original outer update.
        with patch.object(torch.distributed, 'is_initialized', return_value=True), \
                patch.object(torch.distributed, 'all_reduce', side_effect=average) as reduce:
            for _, optimizer in workers:
                optimizer._outer_step()
        self.assertEqual(reduce.call_count, len(workers))

    def advance(self, workers, steps):
        for _ in range(steps):
            sync_due = []
            for rank, (model, optimizer) in enumerate(workers):
                # Different deterministic objectives for each worker.
                model.weight.grad = model.weight.detach() - 2.0 * (rank + 1)
                # Defer the collective until both simulated workers finish locally.
                with patch.object(optimizer, '_outer_step') as deferred:
                    optimizer.step()
                sync_due.append(deferred.called)
                optimizer.zero_grad()
            self.assertTrue(all(value == sync_due[0] for value in sync_due))
            if sync_due[0]:
                self.synchronize(workers)

    def assert_state_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.assert_state_equal(value, reference)
        else:
            self.assertEqual(actual, expected)

    def metadata(self, rank, optimizer):
        return {
            'step': optimizer.inner_step_count,
            'distributed_world_size': 2,
            'diloco_sync_before_checkpoint': optimizer.last_sync_step == optimizer.inner_step_count,
            'dataloader_state_dict': {'pq_idx': 3 + rank, 'rg_idx': 10 + rank, 'epoch': 1},
            'loop_state': {'smooth_train_loss': 0.5 + rank},
        }

    def save_workers(self, workers, rank_local=True, legacy=False):
        # A non-master worker can finish writing before rank 0.
        for rank in reversed(range(len(workers))):
            model, optimizer = workers[rank]
            state = copy.deepcopy(optimizer.state_dict())
            if legacy:
                state.pop('last_sync_step')
            save_checkpoint(
                str(self.path), optimizer.inner_step_count, model.state_dict(), state,
                self.metadata(rank, optimizer), rank=rank, rank_local=rank_local,
            )

    def restore_workers(self, step):
        workers = []
        for rank in range(2):
            model_data, optimizer_data, metadata = load_checkpoint(
                str(self.path), step, 'cpu', load_optimizer=True, rank=rank, world_size=2,
            )
            workers.append(self.make_worker(rank, model_data, optimizer_data))
            self.assertEqual(metadata['checkpoint_rank'], rank)
            self.assertEqual(metadata['dataloader_state_dict']['rg_idx'], 10 + rank)
            self.assertEqual(metadata['loop_state']['smooth_train_loss'], 0.5 + rank)
        return workers

    def test_mid_round_restore_matches_uninterrupted_training(self):
        workers = [self.make_worker(rank) for rank in range(2)]
        self.advance(workers, 6)  # Includes nonzero inner and outer momentum.
        self.assertNotEqual(workers[0][0].weight.item(), workers[1][0].weight.item())
        before = [(copy.deepcopy(m.state_dict()), copy.deepcopy(o.state_dict())) for m, o in workers]
        self.save_workers(workers)
        restored = self.restore_workers(6)
        for (model, optimizer), (saved_model, saved_optimizer) in zip(workers, before):
            self.assert_state_equal(model.state_dict(), saved_model)
            self.assert_state_equal(optimizer.state_dict(), saved_optimizer)
        for step in range(11):
            for original, resumed in zip(workers, restored):
                self.assert_state_equal(resumed[0].state_dict(), original[0].state_dict())
                self.assert_state_equal(resumed[1].state_dict(), original[1].state_dict())
            if step < 10:
                self.advance(workers, 1)
                self.advance(restored, 1)

    def test_final_partial_round_checkpoint_is_already_synced_on_restore(self):
        workers = [self.make_worker(rank) for rank in range(2)]
        self.advance(workers, 3)
        self.synchronize(workers)
        self.save_workers(workers)
        restored = self.restore_workers(3)
        for (model, optimizer), (restored_model, restored_optimizer) in zip(workers, restored):
            with patch.object(restored_optimizer, '_outer_step') as update:
                restored_optimizer.sync_if_pending()
            update.assert_not_called()
            self.assert_state_equal(restored_model.state_dict(), model.state_dict())
            self.assert_state_equal(restored_optimizer.state_dict(), optimizer.state_dict())

    def test_inference_uses_common_rank_zero_model(self):
        workers = [self.make_worker(rank) for rank in range(2)]
        self.advance(workers, 2)
        self.save_workers(workers)
        model_data, optimizer_data, _ = load_checkpoint(str(self.path), 2, 'cpu', rank=1)
        self.assert_state_equal(model_data, workers[0][0].state_dict())
        self.assertIsNone(optimizer_data)

    def test_missing_rank_files_never_fall_back_to_shared_model(self):
        workers = [self.make_worker(rank) for rank in range(2)]
        self.advance(workers, 2)
        self.save_workers(workers)
        for filename in ('model_000002_rank1.pt', 'optim_000002_rank1.pt', 'meta_000002_rank1.json'):
            with self.subTest(filename=filename):
                path = self.path / filename
                backup = path.with_suffix('.backup')
                path.rename(backup)
                try:
                    with self.assertRaisesRegex(FileNotFoundError, 'Incomplete rank-local checkpoint'):
                        load_checkpoint(str(self.path), 2, 'cpu', load_optimizer=True, rank=1, world_size=2)
                finally:
                    backup.rename(path)

    def test_changed_world_size_is_rejected(self):
        workers = [self.make_worker(rank) for rank in range(2)]
        self.advance(workers, 2)
        self.save_workers(workers)
        with self.assertRaisesRegex(ValueError, 'world size'):
            load_checkpoint(str(self.path), 2, 'cpu', load_optimizer=True, rank=0, world_size=4)

    def test_mismatched_rank_metadata_is_rejected(self):
        workers = [self.make_worker(rank) for rank in range(2)]
        self.advance(workers, 2)
        self.save_workers(workers)
        path = self.path / 'meta_000002_rank1.json'
        metadata = json.loads(path.read_text(encoding='utf-8'))
        metadata['checkpoint_rank'] = 0
        path.write_text(json.dumps(metadata), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Mismatched rank-local'):
            load_checkpoint(str(self.path), 2, 'cpu', load_optimizer=True, rank=1, world_size=2)

    def test_regular_shared_checkpoint_still_loads_shared_model_and_rank_optimizer(self):
        for rank in range(2):
            save_checkpoint(
                str(self.path), 4, {'weight': torch.tensor([rank + 1.0])},
                {'state': {}, 'param_groups': [], 'worker': rank}, {'step': 4}, rank=rank,
            )
        # Old diagnostic rank files must not silently override the shared model.
        save_rank_model(str(self.path), 4, {'weight': torch.tensor([99.0])}, rank=1)
        model_data, optimizer_data, _ = load_checkpoint(str(self.path), 4, 'cpu', load_optimizer=True, rank=1)
        self.assertEqual(model_data['weight'].item(), 1.0)
        self.assertEqual(optimizer_data['worker'], 1)

    def test_legacy_explicitly_synced_checkpoint_ignores_presync_rank_files(self):
        workers = [self.make_worker(rank) for rank in range(2)]
        self.advance(workers, 3)
        for rank, (model, _) in enumerate(workers):
            save_rank_model(str(self.path), 3, model.state_dict(), rank=rank)
        self.synchronize(workers)
        self.save_workers(workers, rank_local=False, legacy=True)
        for rank in range(2):
            model_data, optimizer_data, _ = load_checkpoint(
                str(self.path), 3, 'cpu', load_optimizer=True, rank=rank, world_size=2,
            )
            self.assertEqual(model_data['weight'].item(), workers[0][0].weight.item())
            self.assertEqual(optimizer_data['last_sync_step'], 3)
            _, restored_optimizer = self.make_worker(rank, model_data, optimizer_data)
            with patch.object(restored_optimizer, '_outer_step') as update:
                restored_optimizer.sync_if_pending()
            update.assert_not_called()

    def test_legacy_pending_checkpoints_are_rejected(self):
        for legacy in (False, True):
            with self.subTest(missing_sync_marker=legacy):
                workers = [self.make_worker(rank) for rank in range(2)]
                self.advance(workers, 6)
                self.save_workers(workers, rank_local=False, legacy=legacy)
                for rank in range(2):
                    with self.assertRaisesRegex(ValueError, 'cannot be reconstructed'):
                        load_checkpoint(str(self.path), 6, 'cpu', load_optimizer=True, rank=rank, world_size=2)


if __name__ == '__main__':
    unittest.main()

"""CPU regression tests for DiLoCo synchronization and checkpoint state."""

import copy
import io
import unittest
from unittest.mock import Mock

import torch

from nanochat.diloco import DiLoCoWrapper


class TestDiLoCoSynchronization(unittest.TestCase):
    def make_wrapper(self, H=4):
        model = torch.nn.Linear(1, 1, bias=False).double()
        with torch.no_grad():
            model.weight.fill_(1.0)
        callback = Mock()
        optimizer = DiLoCoWrapper(
            [torch.optim.SGD(model.parameters(), lr=0.1)],
            model,
            H=H,
            outer_lr=0.75,
            outer_momentum=0.9,
            pre_sync_callback=callback,
        )
        return model, optimizer, callback

    def advance(self, model, optimizer, steps):
        for _ in range(steps):
            model.weight.grad = torch.ones_like(model.weight)
            optimizer.step()
            optimizer.zero_grad()

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

    def assert_sync_is_noop(self, model, optimizer, callback):
        model_before = copy.deepcopy(model.state_dict())
        state_before = copy.deepcopy(optimizer.state_dict())
        calls_before = callback.call_count
        optimizer.sync_if_pending()
        self.assert_state_equal(model.state_dict(), model_before)
        # Includes outer momentum, the offloaded snapshot and both step counters.
        self.assert_state_equal(optimizer.state_dict(), state_before)
        self.assertEqual(callback.call_count, calls_before)

    def roundtrip(self, model, optimizer):
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location="cpu", weights_only=True)
        # Verify that the checkpoint's H, not the constructor's H, is restored.
        restored_model, restored_optimizer, callback = self.make_wrapper(H=7)
        restored_model.load_state_dict(saved["model"])
        restored_optimizer.load_state_dict(saved["optimizer"])
        self.assertEqual(restored_optimizer.H, optimizer.H)
        return restored_model, restored_optimizer, callback

    def test_sync_before_training_is_noop(self):
        model, optimizer, callback = self.make_wrapper()
        self.assert_sync_is_noop(model, optimizer, callback)

    def test_repeated_forced_sync_preserves_weights_and_momentum(self):
        model, optimizer, callback = self.make_wrapper()
        self.advance(model, optimizer, 3)
        callback.assert_not_called()
        optimizer.sync_if_pending()
        # Pseudo-gradient = 0.3; first Nesterov update = 0.75 * 1.9 * 0.3.
        self.assertAlmostEqual(model.weight.item(), 0.5725)
        self.assertEqual(callback.call_count, 1)
        self.assert_sync_is_noop(model, optimizer, callback)
        self.assert_sync_is_noop(model, optimizer, callback)

    def test_normal_H_step_schedule_and_nesterov_updates_are_preserved(self):
        model, optimizer, callback = self.make_wrapper()
        for expected_weight, expected_calls in ((0.43, 1), (-0.383, 2)):
            self.advance(model, optimizer, 3)
            self.assertEqual(callback.call_count, expected_calls - 1)
            self.advance(model, optimizer, 1)
            # Each full round has pseudo-gradient 0.4; momentum persists.
            self.assertAlmostEqual(model.weight.item(), expected_weight)
            self.assertEqual(callback.call_count, expected_calls)
            self.assert_sync_is_noop(model, optimizer, callback)

    def test_training_after_forced_sync_can_create_new_pending_updates(self):
        model, optimizer, callback = self.make_wrapper(H=5)
        self.advance(model, optimizer, 3)
        optimizer.sync_if_pending()
        self.advance(model, optimizer, 1)
        self.assertEqual(callback.call_count, 1)
        optimizer.sync_if_pending()
        # Second pseudo-gradient = 0.1; momentum = 0.9 * 0.3 + 0.1.
        self.assertAlmostEqual(model.weight.item(), 0.24775)
        self.assertEqual(callback.call_count, 2)
        self.assert_sync_is_noop(model, optimizer, callback)
        self.advance(model, optimizer, 1)
        self.assertEqual(callback.call_count, 3)  # Original step-5 boundary.

    def test_synced_partial_round_checkpoint_does_not_update_again(self):
        model, optimizer, _ = self.make_wrapper()
        self.advance(model, optimizer, 3)
        optimizer.sync_if_pending()
        restored_model, restored_optimizer, callback = self.roundtrip(model, optimizer)
        self.assert_sync_is_noop(restored_model, restored_optimizer, callback)
        self.advance(model, optimizer, 1)
        self.advance(restored_model, restored_optimizer, 1)
        self.assert_state_equal(restored_model.state_dict(), model.state_dict())
        self.assert_state_equal(restored_optimizer.state_dict(), optimizer.state_dict())

    def test_pending_checkpoint_still_syncs_once_after_restore(self):
        model, optimizer, _ = self.make_wrapper()
        self.advance(model, optimizer, 3)
        restored_model, restored_optimizer, callback = self.roundtrip(model, optimizer)
        optimizer.sync_if_pending()
        restored_optimizer.sync_if_pending()
        self.assertEqual(callback.call_count, 1)
        self.assert_state_equal(restored_model.state_dict(), model.state_dict())
        self.assert_state_equal(restored_optimizer.state_dict(), optimizer.state_dict())
        self.assert_sync_is_noop(restored_model, restored_optimizer, callback)

    def test_legacy_checkpoints_at_H_boundaries_remain_supported(self):
        for step in (0, 4):
            with self.subTest(step=step):
                model, optimizer, _ = self.make_wrapper()
                self.advance(model, optimizer, step)
                legacy = copy.deepcopy(optimizer.state_dict())
                legacy.pop("last_sync_step", None)
                restored_model, restored_optimizer, callback = self.make_wrapper(H=7)
                restored_model.load_state_dict(model.state_dict())
                restored_optimizer.load_state_dict(legacy)
                self.assertEqual(restored_optimizer.H, 4)
                self.assert_sync_is_noop(restored_model, restored_optimizer, callback)

    def test_ambiguous_legacy_checkpoint_is_rejected_before_state_changes(self):
        for already_synced in (False, True):
            with self.subTest(already_synced=already_synced):
                model, optimizer, _ = self.make_wrapper()
                self.advance(model, optimizer, 3)
                if already_synced:
                    optimizer.sync_if_pending()
                legacy = copy.deepcopy(optimizer.state_dict())
                legacy.pop("last_sync_step", None)
                target_model, target_optimizer, _ = self.make_wrapper(H=3)
                before = copy.deepcopy(target_optimizer.state_dict())
                with self.assertRaisesRegex(ValueError, "last_sync_step"):
                    target_optimizer.load_state_dict(legacy)
                self.assert_state_equal(target_optimizer.state_dict(), before)
                self.assertEqual(target_model.weight.item(), 1.0)


if __name__ == "__main__":
    unittest.main()

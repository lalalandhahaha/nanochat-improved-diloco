"""Exercise the actual training script's evaluation branches on CPU.

The script executes training on import, so compile only its validation and
evaluation statements. Expensive dataset/model evaluations are test doubles.
"""

import ast
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from nanochat.common import DummyWandb
from nanochat.diloco import DiLoCoWrapper


def compile_nodes(nodes):
    return compile(ast.Module(body=nodes, type_ignores=[]), 'base_train_eval_branches', 'exec')


def has_attribute(node, name):
    return any(isinstance(child, ast.Attribute) and child.attr == name for child in ast.walk(node))


class TestBaseTrainEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'scripts' / 'base_train.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        validation = next(node for node in tree.body if isinstance(node, ast.If) and any(
            isinstance(child, ast.For) and isinstance(child.iter, ast.Tuple)
            and [value.value for value in child.iter.elts] == ['eval_every', 'core_metric_every']
            for child in ast.walk(node)
        ))
        cls.validation_code = compile_nodes([validation])
        loop = next(node for node in tree.body if isinstance(node, ast.While))
        cls.final_sync_code = compile_nodes([next(node for node in loop.body
            if isinstance(node, ast.If) and has_attribute(node, 'sync_if_pending'))])
        cls.sync_state_code = compile_nodes([next(node for node in loop.body
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'eval_model_synced' for t in node.targets))])
        cls.eval_code = compile_nodes([node for node in loop.body
            if isinstance(node, ast.If) and any(
                isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                and child.func.id in ('evaluate_bpb', 'evaluate_core')
                for child in ast.walk(node)
            )])
        cls.svd_code = compile_nodes([next(node for node in loop.body
            if isinstance(node, ast.If) and has_attribute(node.test, 'eval_svd'))])

    def validate(self, H, eval_every, core_every, use_diloco=True, num_iterations=2000):
        # The restored optimizer H intentionally differs from the CLI H.
        scope = {'args': SimpleNamespace(use_diloco=use_diloco, diloco_H=100,
                                        eval_every=eval_every, core_metric_every=core_every),
                 'optimizer': SimpleNamespace(H=H), 'num_iterations': num_iterations}
        exec(self.validation_code, scope)

    def make_scope(self, step, final=False, master=True, use_diloco=True):
        model = torch.nn.Linear(1, 1, bias=False).double()
        model.transformer = SimpleNamespace(h=[object()])
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = DiLoCoWrapper([torch.optim.SGD(model.parameters(), lr=0.1)], model,
                                 H=4, outer_lr=0.75, outer_momentum=0.9)
        for _ in range(step):
            model.weight.grad = torch.ones_like(model.weight)
            optimizer.step()
            optimizer.zero_grad()
        if not use_diloco:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scope = {
            'args': SimpleNamespace(use_diloco=use_diloco, eval_every=4, core_metric_every=4,
                                    eval_svd=-1, eval_tokens=4, device_batch_size=1,
                                    max_seq_len=1, core_metric_max_per_task=2),
            'step': step, 'last_step': final, 'master_process': master,
            'model': model, 'orig_model': model, 'optimizer': optimizer,
            'build_val_loader': Mock(return_value=[]),
            'disable_fp8': lambda _: nullcontext(), 'token_bytes': None, 'tokenizer': None,
            'device': torch.device('cpu'), 'ddp_world_size': 1,
            'time': time, 'print0': Mock(), 'wandb_run': SimpleNamespace(log=Mock()),
            'flops_so_far': 123, 'total_training_time': 5.0,
            'val_bpb': 9.0, 'min_val_bpb': 9.0, 'val_eval_time': 0.0,
            'weight_analysis_time': 0.0, 'core_eval_time': 0.0, 'results': {},
            'diloco_diff_metrics': {},
        }

        def check_eval_model(eval_model):
            self.assertFalse(eval_model.training)
            if use_diloco:
                self.assertEqual(optimizer.last_sync_step, optimizer.inner_step_count)

        def bpb(eval_model, *args):
            check_eval_model(eval_model)
            return 1.25

        def core(eval_model, *args, **kwargs):
            check_eval_model(eval_model)
            return {'core_metric': 0.75, 'centered_results': {'toy': 0.75}}

        scope['evaluate_bpb'] = Mock(side_effect=bpb)
        scope['evaluate_core'] = Mock(side_effect=core)
        scope['analyze_model_weights'] = Mock(return_value=[{
            'name': 'toy.weight', 'shape': (1, 1), 'max_singular_value': 1.0,
            'min_singular_value': 1.0, 'condition_number': 1.0, 'frobenius_norm': 1.0,
            'spectral_norm': 1.0, 'stable_rank': 1.0, 'effective_rank': 1.0,
        }])
        return scope

    def run_evaluations(self, scope, svd=False):
        exec(self.final_sync_code, scope)
        exec(self.sync_state_code, scope)
        exec(self.eval_code, scope)
        if svd:
            exec(self.svd_code, scope)

    def test_misaligned_bpb_interval_is_rejected_using_restored_H(self):
        with self.assertRaisesRegex(ValueError, r'--eval-every=500.*H=200'):
            self.validate(H=200, eval_every=500, core_every=1000)

    def test_misaligned_core_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r'--core-metric-every=500.*H=200'):
            self.validate(H=200, eval_every=1000, core_every=500)

    def test_aligned_and_disabled_intervals_are_supported(self):
        for intervals in ((1000, 1000), (-1, 1000), (1000, 0), (-1, -1)):
            with self.subTest(intervals=intervals):
                self.validate(H=200, eval_every=intervals[0], core_every=intervals[1])

    def test_final_only_intervals_need_no_H_alignment(self):
        for interval in (214001, 999999):
            with self.subTest(interval=interval):
                self.validate(H=200, eval_every=interval, core_every=interval, num_iterations=214001)

    def test_ddp_does_not_require_H_aligned_intervals(self):
        self.validate(H=200, eval_every=500, core_every=300, use_diloco=False)
        scope = self.make_scope(3, use_diloco=False)
        scope['args'].eval_every = scope['args'].core_metric_every = 3
        self.run_evaluations(scope)
        scope['evaluate_bpb'].assert_called_once()
        scope['evaluate_core'].assert_called_once()

    def test_pending_local_models_are_never_evaluated(self):
        scope = self.make_scope(2)
        scope['args'].eval_every = scope['args'].core_metric_every = 2
        # Defense in depth: even bypassing interval validation cannot evaluate drifted weights.
        before = scope['model'].weight.detach().clone()
        with patch.object(scope['optimizer'], '_outer_step') as outer:
            self.run_evaluations(scope)
        outer.assert_not_called()
        scope['evaluate_bpb'].assert_not_called()
        scope['evaluate_core'].assert_not_called()
        torch.testing.assert_close(scope['model'].weight, before, rtol=0, atol=0)

    def test_synchronized_models_are_evaluated_without_extra_outer_updates(self):
        scope = self.make_scope(4)
        before = scope['model'].weight.detach().clone()
        with patch.object(scope['optimizer'], '_outer_step') as outer:
            self.run_evaluations(scope)
        outer.assert_not_called()
        scope['evaluate_bpb'].assert_called_once()
        scope['evaluate_core'].assert_called_once()
        torch.testing.assert_close(scope['model'].weight, before, rtol=0, atol=0)
        self.assertTrue(scope['model'].training)

    def test_partial_final_round_syncs_before_both_evaluations(self):
        scope = self.make_scope(3, final=True)
        self.assertNotEqual(scope['optimizer'].last_sync_step, 3)
        self.run_evaluations(scope)
        self.assertEqual(scope['optimizer'].last_sync_step, 3)
        scope['evaluate_bpb'].assert_called_once()
        scope['evaluate_core'].assert_called_once()

    def test_bpb_is_logged_with_svd_disabled_and_train_mode_restored(self):
        for svd_interval in (-1, 0):
            with self.subTest(svd_interval=svd_interval):
                scope = self.make_scope(4)
                scope['args'].eval_svd = svd_interval
                scope['args'].core_metric_every = -1
                self.run_evaluations(scope, svd=True)
                scope['wandb_run'].log.assert_called_once()
                record = scope['wandb_run'].log.call_args[0][0]
                self.assertEqual(record['val/bpb'], 1.25)
                self.assertEqual(record['step'], 4)
                self.assertGreaterEqual(record['eval/val_bpb_seconds'], 0)
                self.assertEqual(scope['min_val_bpb'], 1.25)
                self.assertTrue(scope['model'].training)
                scope['analyze_model_weights'].assert_not_called()

    def test_only_master_logs_bpb_but_all_workers_evaluate(self):
        scope = self.make_scope(4, master=False)
        scope['args'].core_metric_every = -1
        self.run_evaluations(scope)
        scope['evaluate_bpb'].assert_called_once()
        scope['wandb_run'].log.assert_not_called()
        self.assertTrue(scope['model'].training)

    def test_svd_only_step_does_not_log_stale_bpb(self):
        scope = self.make_scope(6)
        scope['args'].eval_svd = 3
        self.run_evaluations(scope, svd=True)
        scope['evaluate_bpb'].assert_not_called()
        scope['analyze_model_weights'].assert_called_once()
        record = scope['wandb_run'].log.call_args[0][0]
        self.assertNotIn('val/bpb', record)
        self.assertNotIn('eval/val_bpb_seconds', record)
        self.assertIn('weights/toy.weight/spectral_norm', record)

    def test_coincident_svd_and_validation_log_bpb_once(self):
        scope = self.make_scope(4)
        scope['args'].eval_svd = 4
        self.run_evaluations(scope, svd=True)
        records = [call[0][0] for call in scope['wandb_run'].log.call_args_list]
        self.assertEqual(sum('val/bpb' in record for record in records), 1)

    def test_disabled_validation_does_not_log_bpb_even_at_end(self):
        scope = self.make_scope(3, final=True)
        scope['args'].eval_every = scope['args'].core_metric_every = -1
        self.run_evaluations(scope)
        scope['evaluate_bpb'].assert_not_called()
        scope['evaluate_core'].assert_not_called()
        scope['wandb_run'].log.assert_not_called()

    def test_default_dummy_logging_writes_initial_validation_to_jsonl(self):
        scope = self.make_scope(0)
        scope['args'].eval_every = scope['args'].core_metric_every = 500
        with tempfile.TemporaryDirectory() as directory:
            with patch('os.getcwd', return_value=directory):
                scope['wandb_run'] = DummyWandb(name='test_bpb', save_local=True)
            self.run_evaluations(scope)
            records = [json.loads(line) for line in Path(scope['wandb_run'].log_file).read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['step'], 0)
        self.assertEqual(records[0]['val/bpb'], 1.25)
        scope['evaluate_core'].assert_not_called()


if __name__ == '__main__':
    unittest.main()

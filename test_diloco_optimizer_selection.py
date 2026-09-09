"""
Quick test to verify DiLoCo inner/outer optimizer selection logic.
Run with: python test_diloco_optimizer_selection.py
"""
import torch
import torch.nn as nn
from nanochat.diloco import DiLoCoWrapper
from nanochat.isocmerge import ISOCMerge
from nanochat.mergeloco import MergeLoCo

# Mock model
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

print("Creating mock model...")
model = TinyModel()

# Inner optimizer: simple SGD for this test
inner_opt = torch.optim.SGD(model.parameters(), lr=0.01)

print("\n" + "="*80)
print("TEST 1: outer_opt='nesterov'")
print("="*80)
wrapper1 = DiLoCoWrapper(
    inner_optimizers=[inner_opt],
    model=model,
    outer_lr=0.7,
    outer_momentum=0.9,
    H=10,
    outer_opt="nesterov",
)
print(f"outer_optimizer type: {type(wrapper1.outer_optimizer).__name__}")
print(f"outer_is_merge_based: {wrapper1.outer_is_merge_based}")
assert isinstance(wrapper1.outer_optimizer, torch.optim.SGD), "Expected SGD for nesterov"
assert not wrapper1.outer_is_merge_based, "nesterov should not be merge-based"
print("✓ PASS: outer_opt='nesterov' creates SGD")

print("\n" + "="*80)
print("TEST 2: outer_opt='isoc'")
print("="*80)
wrapper2 = DiLoCoWrapper(
    inner_optimizers=[inner_opt],
    model=model,
    outer_lr=0.7,
    outer_momentum=0.9,
    H=10,
    outer_opt="isoc",
    isoc_orthogonalize_object="gradient",
)
print(f"outer_optimizer type: {type(wrapper2.outer_optimizer).__name__}")
print(f"outer_is_merge_based: {wrapper2.outer_is_merge_based}")
assert isinstance(wrapper2.outer_optimizer, ISOCMerge), "Expected ISOCMerge for isoc"
assert wrapper2.outer_is_merge_based, "isoc should be merge-based"
print("✓ PASS: outer_opt='isoc' creates ISOCMerge")

print("\n" + "="*80)
print("TEST 3: outer_opt='ties'")
print("="*80)
wrapper3 = DiLoCoWrapper(
    inner_optimizers=[inner_opt],
    model=model,
    outer_lr=0.7,
    outer_momentum=0.0,
    H=10,
    outer_opt="ties",
    ties_disjoint=True,
    ties_sparsity=0.2,
)
print(f"outer_optimizer type: {type(wrapper3.outer_optimizer).__name__}")
print(f"outer_is_merge_based: {wrapper3.outer_is_merge_based}")
assert isinstance(wrapper3.outer_optimizer, MergeLoCo), "Expected MergeLoCo for ties"
assert wrapper3.outer_is_merge_based, "ties should be merge-based"
print("✓ PASS: outer_opt='ties' creates MergeLoCo")

print("\n" + "="*80)
print("All tests passed ✓")
print("="*80)

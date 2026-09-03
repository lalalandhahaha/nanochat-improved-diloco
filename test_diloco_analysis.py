"""
Quick test script to verify DiLoCo analysis functionality.

Run with:
    python test_diloco_analysis.py
"""

import torch
import torch.distributed as dist
from nanochat.gpt import GPT, GPTConfig
from nanochat.weight_analysis import analyze_model_weights
from nanochat.diloco_analysis import analyze_node_weight_differences
from nanochat.common import DummyWandb

def test_weight_analysis():
    """Test single-node weight analysis."""
    print("=" * 80)
    print("Test 1: Single-node weight analysis")
    print("=" * 80)

    # Create a small model
    config = GPTConfig(
        vocab_size=1000,
        sequence_len=128,
        n_layer=4,
        n_embd=256,
    )
    model = GPT(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Analyze weights
    results = analyze_model_weights(model, layer_indices=[0, 1, 2, 3])

    print(f"\nAnalyzed {len(results)} weight matrices:")
    for name, metrics in results.items():
        print(f"  {name}: s_max={metrics['max_singular_value']:.4f}, "
              f"cond={metrics['condition_number']:.2f}, "
              f"stable_rank={metrics['stable_rank']:.2f}")

    print("\n✓ Weight analysis test passed!")
    return True

def test_diloco_analysis():
    """Test DiLoCo node difference analysis (requires distributed setup)."""
    print("\n" + "=" * 80)
    print("Test 2: DiLoCo node difference analysis")
    print("=" * 80)

    if not dist.is_initialized():
        print("⚠ Distributed not initialized, skipping DiLoCo analysis test")
        print("  Run with: torchrun --nproc_per_node=2 test_diloco_analysis.py")
        return True

    # Create a small model
    config = GPTConfig(
        vocab_size=1000,
        sequence_len=128,
        n_layer=4,
        n_embd=256,
    )
    model = GPT(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Add some noise to weights on different ranks
    rank = dist.get_rank()
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.01 * rank)

    # Analyze node differences
    results = analyze_node_weight_differences(model, layer_indices=[0, 1], device=device)

    if rank == 0 and results:
        print(f"\nAnalyzed node differences for {len(results)} weight matrices:")
        for name, metrics in results.items():
            print(f"  {name}:")
            print(f"    cosine_sim_mean={metrics['cosine_sim_mean']:.6f}")
            print(f"    weight_diff_mean={metrics['weight_diff_mean']:.6f}")
        print("\n✓ DiLoCo analysis test passed!")

    return True

def test_dummy_wandb():
    """Test DummyWandb local logging."""
    print("\n" + "=" * 80)
    print("Test 3: DummyWandb local logging")
    print("=" * 80)

    # Create a DummyWandb instance
    wandb = DummyWandb(project="test", name="test_run", save_local=True)

    # Log some metrics
    for i in range(5):
        wandb.log({
            "step": i,
            "loss": 1.0 / (i + 1),
            "accuracy": i * 0.1,
        })

    wandb.finish()

    print(f"\n✓ DummyWandb test passed! Check {wandb.log_dir}")
    return True

def main():
    print("Testing DiLoCo Analysis Functionality")
    print("=" * 80)

    try:
        # Test 1: Weight analysis
        test_weight_analysis()

        # Test 2: DiLoCo analysis (only if distributed)
        test_diloco_analysis()

        # Test 3: DummyWandb
        test_dummy_wandb()

        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

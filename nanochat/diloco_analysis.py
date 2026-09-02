"""
DiLoCo-specific analysis utilities for comparing weights across nodes.

Analyzes differences between nodes before outer synchronization:
- Singular value similarity
- Weight norm differences
- Weight similarity (cosine similarity)
- Condition number differences
- Stable rank differences
- Effective rank differences
"""

import torch
import torch.distributed as dist
from .weight_analysis import power_iteration, inverse_power_iteration, compute_weight_metrics


def compute_cosine_similarity(w1, w2):
    """
    计算两个权重矩阵的余弦相似度

    Args:
        w1, w2: 权重矩阵 (torch.Tensor)

    Returns:
        float: 余弦相似度 [-1, 1]
    """
    w1_flat = w1.flatten()
    w2_flat = w2.flatten()

    dot_product = torch.dot(w1_flat, w2_flat)
    norm1 = torch.norm(w1_flat)
    norm2 = torch.norm(w2_flat)

    cosine_sim = dot_product / (norm1 * norm2 + 1e-10)
    return cosine_sim.item()


def compute_weight_difference_norm(w1, w2, norm_type='fro'):
    """
    计算两个权重矩阵的差异范数

    Args:
        w1, w2: 权重矩阵 (torch.Tensor)
        norm_type: 范数类型 ('fro' or '2')

    Returns:
        float: 差异范数
    """
    diff = w1 - w2
    if norm_type == 'fro':
        return torch.norm(diff, p='fro').item()
    else:  # spectral norm
        return power_iteration(diff.cpu())


def gather_weights_from_all_nodes(weight, device):
    """
    从所有节点收集权重

    Args:
        weight: 本节点的权重矩阵 (torch.Tensor)
        device: 设备

    Returns:
        list: 所有节点的权重列表 (只在rank 0返回完整列表，其他rank返回None)
    """
    if not dist.is_initialized():
        return [weight]

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 准备接收所有节点的权重（只在rank 0上）
    if rank == 0:
        gathered_weights = [torch.zeros_like(weight) for _ in range(world_size)]
    else:
        gathered_weights = None

    # 收集权重
    dist.gather(weight.contiguous(), gathered_weights if rank == 0 else None, dst=0)

    return gathered_weights if rank == 0 else None


def analyze_node_weight_differences(model, layer_indices=None, device='cuda'):
    """
    分析DiLoCo模式下不同节点的权重差异

    只在rank 0上返回完整分析结果，其他rank参与数据收集但返回空字典

    Args:
        model: GPT模型
        layer_indices: 要分析的层索引列表，None表示所有层
        device: 设备

    Returns:
        dict: 分析结果（只在rank 0上有内容）
    """
    if not dist.is_initialized():
        return {}

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return {}  # 单节点没有差异可分析

    results = {}

    # 收集要分析的权重
    weights_to_analyze = []
    weight_names = []

    # 1. lm_head.weight
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        weights_to_analyze.append(model.lm_head.weight)
        weight_names.append("lm_head.weight")

    # 2. wte.weight
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        weights_to_analyze.append(model.transformer.wte.weight)
        weight_names.append("wte.weight")

    # 3. transformer层
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        num_layers = len(model.transformer.h)
        if layer_indices is None:
            layer_indices = range(num_layers)

        for layer_idx in layer_indices:
            if layer_idx >= num_layers:
                continue

            layer = model.transformer.h[layer_idx]

            # Attention层
            if hasattr(layer, 'attn'):
                if hasattr(layer.attn, 'c_attn') and hasattr(layer.attn.c_attn, 'weight'):
                    weights_to_analyze.append(layer.attn.c_attn.weight)
                    weight_names.append(f'layer_{layer_idx}.attn.c_attn.weight')
                if hasattr(layer.attn, 'c_proj') and hasattr(layer.attn.c_proj, 'weight'):
                    weights_to_analyze.append(layer.attn.c_proj.weight)
                    weight_names.append(f'layer_{layer_idx}.attn.c_proj.weight')

            # MLP层
            if hasattr(layer, 'mlp'):
                if hasattr(layer.mlp, 'c_fc') and hasattr(layer.mlp.c_fc, 'weight'):
                    weights_to_analyze.append(layer.mlp.c_fc.weight)
                    weight_names.append(f'layer_{layer_idx}.mlp.c_fc.weight')
                if hasattr(layer.mlp, 'c_proj') and hasattr(layer.mlp.c_proj, 'weight'):
                    weights_to_analyze.append(layer.mlp.c_proj.weight)
                    weight_names.append(f'layer_{layer_idx}.mlp.c_proj.weight')

    # 对每个权重进行分析
    for weight, name in zip(weights_to_analyze, weight_names):
        # 收集所有节点的权重
        all_weights = gather_weights_from_all_nodes(weight, device)

        if rank == 0:
            # 只在rank 0上进行分析
            layer_results = {
                'name': name,
                'world_size': world_size,
            }

            # 计算每个节点的指标
            node_metrics = []
            for node_id in range(world_size):
                w = all_weights[node_id].cpu()
                metrics = compute_weight_metrics(w, name=f"{name}_node{node_id}", use_full_svd=False)
                node_metrics.append(metrics)

            # 计算节点间的差异统计
            # 1. 最大奇异值的统计
            s_maxs = [m['max_singular_value'] for m in node_metrics]
            layer_results['s_max_mean'] = float(torch.tensor(s_maxs).mean().item())
            layer_results['s_max_std'] = float(torch.tensor(s_maxs).std().item())
            layer_results['s_max_min'] = min(s_maxs)
            layer_results['s_max_max'] = max(s_maxs)

            # 2. 条件数的统计
            cond_nums = [m['condition_number'] for m in node_metrics]
            layer_results['cond_mean'] = float(torch.tensor(cond_nums).mean().item())
            layer_results['cond_std'] = float(torch.tensor(cond_nums).std().item())

            # 3. 稳定秩的统计
            stable_ranks = [m['stable_rank'] for m in node_metrics]
            layer_results['stable_rank_mean'] = float(torch.tensor(stable_ranks).mean().item())
            layer_results['stable_rank_std'] = float(torch.tensor(stable_ranks).std().item())

            # 4. 有效秩的统计
            effective_ranks = [m['effective_rank'] for m in node_metrics]
            layer_results['effective_rank_mean'] = float(torch.tensor(effective_ranks).mean().item())
            layer_results['effective_rank_std'] = float(torch.tensor(effective_ranks).std().item())

            # 5. Frobenius范数的统计
            fro_norms = [m['frobenius_norm'] for m in node_metrics]
            layer_results['fro_norm_mean'] = float(torch.tensor(fro_norms).mean().item())
            layer_results['fro_norm_std'] = float(torch.tensor(fro_norms).std().item())

            # 6. 计算节点间的余弦相似度（两两比较，取平均）
            cosine_sims = []
            for i in range(world_size):
                for j in range(i + 1, world_size):
                    sim = compute_cosine_similarity(all_weights[i].cpu(), all_weights[j].cpu())
                    cosine_sims.append(sim)

            if cosine_sims:
                layer_results['cosine_sim_mean'] = float(torch.tensor(cosine_sims).mean().item())
                layer_results['cosine_sim_std'] = float(torch.tensor(cosine_sims).std().item())
                layer_results['cosine_sim_min'] = min(cosine_sims)

            # 7. 计算节点间的权重差异范数（相对于rank 0）
            weight_diffs = []
            for i in range(1, world_size):
                diff_norm = compute_weight_difference_norm(all_weights[0].cpu(), all_weights[i].cpu(), norm_type='fro')
                weight_diffs.append(diff_norm)

            if weight_diffs:
                layer_results['weight_diff_mean'] = float(torch.tensor(weight_diffs).mean().item())
                layer_results['weight_diff_std'] = float(torch.tensor(weight_diffs).std().item())
                layer_results['weight_diff_max'] = max(weight_diffs)

            results[name] = layer_results

    return results

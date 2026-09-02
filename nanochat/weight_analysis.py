"""
Weight matrix analysis utilities for monitoring training dynamics.

Computes various metrics including:
- Singular values (max, min)
- Norms (Frobenius, spectral)
- Condition number
- Stable rank
- Effective rank
"""

import torch
import numpy as np
from scipy.sparse.linalg import svds


def power_iteration(W, num_iter=100):
    """
    幂迭代求最大奇异值

    Args:
        W: 权重矩阵 (torch.Tensor)
        num_iter: 迭代次数

    Returns:
        float: 最大奇异值
    """
    device = W.device
    dtype = W.dtype

    # 初始化随机向量
    v = torch.randn(W.shape[1], device=device, dtype=dtype)
    v = v / torch.norm(v)

    for _ in range(num_iter):
        # v = W^T W v
        v = W.T @ (W @ v)
        v = v / torch.norm(v)

    # 最大奇异值
    s_max = torch.norm(W @ v)
    return s_max.item()


def inverse_power_iteration(W, num_iter=100, shift=1e-6):
    """
    逆幂迭代求最小奇异值（对于条件数）

    Args:
        W: 权重矩阵 (torch.Tensor)
        num_iter: 迭代次数
        shift: 偏移量，防止奇异

    Returns:
        float: 最小奇异值
    """
    device = W.device
    dtype = W.dtype
    m, n = W.shape

    # 使用 (W^T W + shift*I)^{-1} 的幂迭代
    WtW = W.T @ W
    I = torch.eye(n, device=device, dtype=dtype)

    v = torch.randn(n, device=device, dtype=dtype)
    v = v / torch.norm(v)

    for _ in range(num_iter):
        v = torch.linalg.solve(WtW + shift * I, v)
        v = v / torch.norm(v)

    s_min = torch.norm(W @ v)
    return s_min.item()


def compute_weight_metrics(weight_matrix, name="", use_full_svd=False, k=50):
    """
    计算权重矩阵的各种指标

    Args:
        weight_matrix: 权重矩阵 (torch.Tensor)
        name: 层名称
        use_full_svd: 是否使用全SVD（用于有效秩）
        k: 截断SVD的秩（当use_full_svd=False时）

    Returns:
        dict: 包含所有计算指标的字典
    """
    W = weight_matrix.detach().float().cpu()

    # 1. 权重范数
    frobenius_norm = torch.norm(W, p='fro').item()

    # 2. 最大奇异值（幂迭代）
    s_max = power_iteration(W)
    spectral_norm = s_max

    # 3. 稳定秩 = ||W||_F^2 / ||W||_2^2
    stable_rank = (frobenius_norm ** 2) / (s_max ** 2)

    # 4. 条件数（需要最小奇异值）
    s_min = inverse_power_iteration(W)
    condition_number = s_max / (s_min + 1e-10)

    # 5. 有效秩
    if use_full_svd:
        # 全SVD - 对大矩阵很慢
        if W.shape[0] * W.shape[1] < 10000 * 10000:  # 只对小矩阵做全SVD
            try:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                singular_values = S.numpy()
            except:
                use_full_svd = False
        else:
            use_full_svd = False

    if not use_full_svd:
        # 截断SVD近似
        k_actual = min(k, min(W.shape) - 1)
        try:
            # 使用scipy的稀疏SVD
            U, S, Vt = svds(W.numpy(), k=k_actual)
            singular_values = S[::-1]  # svds返回升序，需要反转
        except:
            singular_values = np.array([s_max])

    # 计算有效秩
    if len(singular_values) > 1:
        # 归一化奇异值
        sv_normalized = singular_values / np.sum(singular_values)
        # 计算熵
        sv_normalized = sv_normalized[sv_normalized > 1e-10]  # 避免log(0)
        entropy = -np.sum(sv_normalized * np.log(sv_normalized + 1e-10))
        effective_rank = np.exp(entropy)
    else:
        effective_rank = 1.0

    results = {
        'name': name,
        'shape': tuple(W.shape),
        'frobenius_norm': frobenius_norm,
        'spectral_norm': spectral_norm,
        'max_singular_value': s_max,
        'min_singular_value': s_min,
        'condition_number': condition_number,
        'stable_rank': stable_rank,
        'effective_rank': effective_rank,
    }

    return results


def analyze_model_weights(model, layer_indices=None):
    """
    分析模型的权重矩阵

    Args:
        model: GPT模型
        layer_indices: 要分析的层索引列表，None表示所有层

    Returns:
        list: 包含所有层分析结果的列表
    """
    results = []

    # 1. 分析 lm_head.weight
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        lm_head_weight = model.lm_head.weight
        results.append(compute_weight_metrics(lm_head_weight, "lm_head.weight", use_full_svd=False))

    # 2. 分析 wte.weight (token embeddings)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        wte_weight = model.transformer.wte.weight
        results.append(compute_weight_metrics(wte_weight, "wte.weight", use_full_svd=False))

    # 3. 分析transformer层
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        num_layers = len(model.transformer.h)
        if layer_indices is None:
            layer_indices = range(num_layers)

        for layer_idx in layer_indices:
            if layer_idx >= num_layers:
                continue

            layer = model.transformer.h[layer_idx]

            # 分析各层权重
            weights_to_analyze = {}

            # Attention层
            if hasattr(layer, 'attn'):
                if hasattr(layer.attn, 'c_attn') and hasattr(layer.attn.c_attn, 'weight'):
                    weights_to_analyze[f'layer_{layer_idx}.attn.c_attn.weight'] = layer.attn.c_attn.weight
                if hasattr(layer.attn, 'c_proj') and hasattr(layer.attn.c_proj, 'weight'):
                    weights_to_analyze[f'layer_{layer_idx}.attn.c_proj.weight'] = layer.attn.c_proj.weight

            # MLP层
            if hasattr(layer, 'mlp'):
                if hasattr(layer.mlp, 'c_fc') and hasattr(layer.mlp.c_fc, 'weight'):
                    weights_to_analyze[f'layer_{layer_idx}.mlp.c_fc.weight'] = layer.mlp.c_fc.weight
                if hasattr(layer.mlp, 'c_proj') and hasattr(layer.mlp.c_proj, 'weight'):
                    weights_to_analyze[f'layer_{layer_idx}.mlp.c_proj.weight'] = layer.mlp.c_proj.weight

            for name, weight in weights_to_analyze.items():
                results.append(compute_weight_metrics(weight, name, use_full_svd=False))

    return results

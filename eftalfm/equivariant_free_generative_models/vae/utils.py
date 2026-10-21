import torch 
from torch import Tensor, BoolTensor, LongTensor
def cardinality_weighted_mean(
    per_sample_values: Tensor,
    cardinalities: Tensor,
    exponent: float,
) -> Tensor:
    """Average per-sample means with sublinear cardinality weighting."""
    exponent = float(exponent)
    if not 0.0 <= exponent <= 1.0:
        raise ValueError(f"cardinality exponent must be in [0, 1]; received {exponent}")
    if per_sample_values.ndim == 0:
        raise ValueError("per_sample_values must have a batch dimension.")

    cardinalities = cardinalities.to(
        device=per_sample_values.device,
        dtype=per_sample_values.dtype,
    ).view(-1)
    if per_sample_values.shape[0] != cardinalities.numel():
        raise ValueError(
            "per_sample_values and cardinalities must have the same batch size; "
            f"received {per_sample_values.shape[0]} and {cardinalities.numel()}."
        )
    if torch.any(cardinalities <= 0):
        raise ValueError("cardinalities must be positive.")

    weights = cardinalities.pow(exponent)
    weight_shape = (weights.shape[0],) + (1,) * (per_sample_values.ndim - 1)
    return (per_sample_values * weights.view(weight_shape)).sum(dim=0) / weights.sum()

def sample_per_molecule_corruption_mask(
    valid_mask: BoolTensor,
    corruption_ratio: float,
    scores: Tensor = None,
) -> tuple[BoolTensor, LongTensor]:
    """Sample exact per-molecule corruption counts without a device sync."""
    if valid_mask.dim() != 2:
        raise ValueError(
            "valid_mask must have shape (batch, n_nodes); "
            f"received {tuple(valid_mask.shape)}"
        )
    if not (0.0 <= corruption_ratio <= 1.0):
        raise ValueError(f"corruption_ratio must be in [0, 1]; received {corruption_ratio}")

    b, max_n_nodes = valid_mask.shape
    device = valid_mask.device
    n_valid_nodes = valid_mask.sum(dim=-1)                                    # (b,)
    m_per_mol = (n_valid_nodes * corruption_ratio).to(torch.long)             # (b,)
    if scores is None:
        scores = torch.rand(b, max_n_nodes, device=device)
    else:
        if scores.shape != valid_mask.shape:
            raise ValueError(
                "scores must match valid_mask shape; "
                f"received {tuple(scores.shape)} vs {tuple(valid_mask.shape)}"
            )
        scores = scores.to(device=device, dtype=torch.float32)

    scores = scores.masked_fill(~valid_mask, -torch.inf)
    ranked_idx = scores.argsort(dim=1, descending=True)
    keep_rank = (
        torch.arange(max_n_nodes, device=device).unsqueeze(0)
        < m_per_mol.unsqueeze(1)
    )
    # ranked_idx is a permutation, so scatter writes every output element.
    corruption_mask = torch.empty_like(valid_mask)
    corruption_mask.scatter_(dim=1, index=ranked_idx, src=keep_rank)
    return corruption_mask, m_per_mol


def select_per_molecule_corruption_indices(
    valid_mask: BoolTensor,
    corruption_ratio: float,
    scores: Tensor = None,
) -> tuple[LongTensor, LongTensor, LongTensor]:
    """Compatibility wrapper returning flattened corruption indices.

    The VAE training hot path uses :func:`sample_per_molecule_corruption_mask`
    directly to avoid the synchronization associated with dynamic-size indices.
    """
    corruption_mask, m_per_mol = sample_per_molecule_corruption_mask(
        valid_mask=valid_mask,
        corruption_ratio=corruption_ratio,
        scores=scores,
    )
    row_idx, perturb_idx = corruption_mask.nonzero(as_tuple=True)
    return row_idx, perturb_idx, m_per_mol
 


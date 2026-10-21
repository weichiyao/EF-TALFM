import numpy as np

def get_kl_weight(
    current,
    total,
    min_beta: float = 0.0,
    max_beta: float = 1.0,
    scaling_factor: float = 5.,
    method: str = "exp_annealing",
    warmup_fraction: float = 0.2,
):
    """Compute KL weight at the current point in training.

    Args:
        current: Current epoch or step (any non-negative number).
        total: Total epochs or steps.  If <= 0 the scheduler degrades to a
            constant ``max_beta`` (safe under step-limited or unknown-horizon
            training where ``total_epochs`` may be -1).
        min_beta / max_beta: Lower / upper bound of the annealed weight.
        scaling_factor: Sharpness for ``exp_annealing`` / ``exp-convex_annealing``.
        method: One of {"none", "linear", "exp_annealing", "exp-convex_annealing"}.
        warmup_fraction: For ``method="linear"``, fraction of training over
            which to ramp from ``min_beta`` to ``max_beta``.  After warmup
            the weight stays at ``max_beta``.

    Returns:
        Scalar KL weight in [min_beta, max_beta].
    """
    if method == "none":
        return max_beta

    # Degrade gracefully when total horizon is unknown / invalid.
    if total is None or total <= 0:
        return max_beta

    progress = max(0.0, min(float(current) / float(total), 1.0))

    if method == "linear":
        ramp = min(progress / max(warmup_fraction, 1e-12), 1.0)
        return float(min_beta + (max_beta - min_beta) * ramp)
    elif method == "exp_annealing":
        return float(max_beta * (1 - np.exp(-progress * scaling_factor)))
    elif method == "exp-convex_annealing":
        x = progress ** scaling_factor
        # Guard log(0) when min_beta == 0
        lo = max(min_beta, 1e-12)
        return float(lo * np.exp((np.log(max_beta) - np.log(lo)) * x))
    else:
        raise ValueError(
            "kl_weight_scheduler must be one of "
            "{'none', 'linear', 'exp_annealing', 'exp-convex_annealing'}; "
            f"received {method!r}."
        )
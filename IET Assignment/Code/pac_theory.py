"""
pac_theory.py
Sample-complexity estimates for the hypothesis spaces used in this
project, using the standard PAC bounds.

1) Finite hypothesis space bound (used as a coarse bound for the
   discretised GA search space):
       m >= (1/epsilon) * ( ln|H| + ln(1/delta) )

2) VC-dimension bound for the MLP (agnostic / realizable settings differ
   by constants; we report the standard realizable-case bound as an
   order-of-magnitude estimate):
       m >= (1/epsilon) * ( 4 log2(2/delta) + 8 * VCdim(H) * log2(13/epsilon) )

   VC-dimension of a feed-forward network with W weights and sigmoidal /
   ReLU units is O(W log W) in general; for a small network we use the
   commonly cited bound VCdim <= O(W log W) with W = number of weights,
   which for ReLU networks with L layers is roughly bounded by
   VCdim = O(W L log W) (Bartlett et al.). We report both the number of
   free parameters and this order-of-magnitude estimate.
"""
import numpy as np


def finite_hypothesis_bound(h_size: int, epsilon: float, delta: float) -> float:
    return (1.0 / epsilon) * (np.log(h_size) + np.log(1.0 / delta))


def vc_dim_estimate_mlp(n_weights: int, n_layers: int = 2) -> float:
    """Order-of-magnitude VC-dimension estimate for a ReLU MLP,
    VC = O(W * L * log(W)), W = #weights, L = #layers."""
    return n_weights * n_layers * np.log2(max(n_weights, 2))


def vc_bound_sample_complexity(vc_dim: float, epsilon: float, delta: float) -> float:
    return (1.0 / epsilon) * (4 * np.log2(2.0 / delta) + 8 * vc_dim * np.log2(13.0 / epsilon))

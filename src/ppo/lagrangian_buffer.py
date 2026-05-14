# =============================================================================
# Lagrangian Rollout Buffer
# =============================================================================
# Subclasses MultiDiscountRolloutBuffer to add cost storage for Lagrangian
# safety constraints. Stores Ct per step; provides mean cost for λc update.
#
# Ct = c1 * 1[|d_eta| < τ_crit] + c2 * 1[collision]
#
# GAE computation is inherited unchanged from MultiDiscountRolloutBuffer.
# =============================================================================

import numpy as np
from src.ppo.multi_discount_buffer import MultiDiscountRolloutBuffer


class LagrangianRolloutBuffer(MultiDiscountRolloutBuffer):
    """
    Extends MultiDiscountRolloutBuffer with a cost buffer for Lagrangian safety.

    Extra behaviour:
        - Allocates self.costs array (buffer_size, n_envs)
        - add_cost(pos, cost) stores Ct at the given buffer position
        - get_mean_cost() returns scalar mean cost over the filled buffer
          for use in the λc update inside LagrangianPPO.train()

    Everything else (dual GAE, reward components, PPO interface) is inherited.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # costs is allocated here; reset() re-allocates to stay in sync
        self.costs = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

    def reset(self):
        """Reset cost buffer alongside parent buffers."""
        self.costs = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        super().reset()

    def add_cost(self, pos: int, cost: np.ndarray) -> None:
        """
        Store constraint cost Ct at the given buffer position.

        Args:
            pos  : current buffer position (int)
            cost : np.array shape (n_envs,) — Ct values for each parallel env
        """
        self.costs[pos] = cost.copy()

    def get_mean_cost(self) -> float:
        """
        Return mean Ct over the entire filled buffer.
        Used by LagrangianPPO.train() to update λc.

        Returns:
            float — scalar mean constraint cost
        """
        return float(self.costs.mean())

#============V03 VARIANT: cancels the crossing-conflict penalty already in v02's
# compute_reward(), since Ct/lambda_c (LagrangianPPO) now owns that signal.
# v02 file is NOT touched — this subclass corrects the scalar after the fact.
#============
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
from alpha_env_v02_discrete import AlphaEnv_v01_Discrete


class AlphaEnv_v01_Discrete_Lagrangian(AlphaEnv_v01_Discrete):
    """
    Wraps AlphaEnv_v01_Discrete.compute_reward() and removes the
    crossing-conflict exponential penalty (|d_eta| < 0.2) that v02 already
    applies — Ct/lambda_c now handle that signal via the Lagrangian
    advantage instead, so leaving both in would double-count it.
    Car-following (ttc_norm) penalty is untouched — Ct doesn't cover it.
    """

    def compute_reward(self, agent_id, fail, goal_reached, current_action=None):
        reward = super().compute_reward(agent_id, fail, goal_reached, current_action)

        if fail or goal_reached:
            return reward

        obs_info = getattr(self, 'last_neighbors_info', [])
        duplicate_penalty = 0.0
        for n in obs_info:
            abs_d_eta = abs(n.get('d_eta', float('inf')))
            if abs_d_eta < 0.2:
                duplicate_penalty += -np.exp(-abs_d_eta * 10.0)

        correction = -duplicate_penalty
        self.last_r_cruise += correction
        return reward + correction

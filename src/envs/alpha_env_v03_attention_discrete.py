#============V03 VARIANT: cancels the duplicate crossing-conflict penalty in v02's
# compute_reward() — Ct/lambda_c (LagrangianPPO) owns that signal instead.
# v02 file (alpha_env_v02_attention_discrete.py) is NOT touched.
#============
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
from alpha_env_v02_attention_discrete import AlphaEnv_v01_AttentionDiscrete


class AlphaEnv_v01_AttentionDiscrete_Lagrangian(AlphaEnv_v01_AttentionDiscrete):
    """
    Wraps AlphaEnv_v01_AttentionDiscrete.compute_reward() (which defines its
    own copy, not inherited) and cancels the crossing-conflict exponential
    penalty (|d_eta| < 0.2) — Ct/lambda_c now own that signal.
    Car-following (ttc_norm) penalty is untouched.
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

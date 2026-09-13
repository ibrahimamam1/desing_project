#============SHIELDED ATTENTION VARIANT (DISCRETE)============
# Applies the same post-decision safety shield as the continuous variant,
# but on top of the discrete-action attention policy.
#
# The shield itself is action-space agnostic: it inspects a proposed
# acceleration and returns a safe one. Only the decoding differs, so the
# checking logic is reused from the continuous variant rather than copied,
# and the two variants cannot drift apart.

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

from alpha_env_v01_attention_discrete import AlphaEnv_v01_AttentionDiscrete
from alpha_env_v01_shielded_attention_continous import AlphaEnv_v01_ShieldedAttention


class AlphaEnv_v01_ShieldedAttentionDiscrete(AlphaEnv_v01_AttentionDiscrete):
    """
    Shielded PPO — Attention + Discrete with the same safety override.

        discrete action -> acceleration -> TTC -> RSS -> right-of-way -> safe action
    """

    # Reuse the tuned constants and the checking logic itself, so a change to
    # the shield applies to both action spaces.
    TTC_THRESHOLD = AlphaEnv_v01_ShieldedAttention.TTC_THRESHOLD
    RSS_MIN_GAP = AlphaEnv_v01_ShieldedAttention.RSS_MIN_GAP
    RSS_REACTION_TIME = AlphaEnv_v01_ShieldedAttention.RSS_REACTION_TIME
    EMERGENCY_DECEL = AlphaEnv_v01_ShieldedAttention.EMERGENCY_DECEL
    TTC_ETA_WINDOW = AlphaEnv_v01_ShieldedAttention.TTC_ETA_WINDOW
    RSS_ETA_WINDOW = AlphaEnv_v01_ShieldedAttention.RSS_ETA_WINDOW
    ROW_ETA_WINDOW = AlphaEnv_v01_ShieldedAttention.ROW_ETA_WINDOW
    ROW_DIST = AlphaEnv_v01_ShieldedAttention.ROW_DIST
    ROW_CROSS = AlphaEnv_v01_ShieldedAttention.ROW_CROSS

    _shield_check = AlphaEnv_v01_ShieldedAttention._shield_check
    _compute_telemetry_stats = AlphaEnv_v01_ShieldedAttention._compute_telemetry_stats

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.shield_stats = {
            'total_steps': 0,
            'ttc_overrides': 0,
            'rss_overrides': 0,
            'row_overrides': 0,
            'total_overrides': 0,
        }

    def _apply_rl_actions(self, rl_action):
        """Decode the discrete action, then pass it through the shield."""
        max_accel = self.env_params.additional_params['max_accel']
        max_decel = self.env_params.additional_params['max_decel']

        try:
            action_idx = int(rl_action)
            action_val = self.ACCEL_BINS[action_idx]
        except (TypeError, ValueError, IndexError):
            action_val = 0.0

        proposed_accel = action_val * max_accel if action_val >= 0 else action_val * max_decel

        self.shield_stats['total_steps'] += 1

        if self.agent_id in self.k.vehicle.get_ids():
            safe_accel, reason = self._shield_check(proposed_accel)
            if reason:
                self.shield_stats['total_overrides'] += 1
                if 'ttc' in reason:
                    self.shield_stats['ttc_overrides'] += 1
                if 'rss' in reason:
                    self.shield_stats['rss_overrides'] += 1
                if 'row' in reason:
                    self.shield_stats['row_overrides'] += 1
            self.k.vehicle.apply_acceleration([self.agent_id], [safe_accel])

    def reset(self, **kwargs):
        self.shield_stats = {
            'total_steps': 0,
            'ttc_overrides': 0,
            'rss_overrides': 0,
            'row_overrides': 0,
            'total_overrides': 0,
        }
        return super().reset(**kwargs)

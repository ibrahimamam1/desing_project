#============SHIELDED ATTENTION VARIANT (CONTINUOUS)============
# Wraps AlphaEnv_v01_Attention with a post-decision safety shield.
# The shield intercepts PPO actions and overrides them when unsafe.

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

from alpha_env_v01_attention_continous import AlphaEnv_v01_Attention


class AlphaEnv_v01_ShieldedAttention(AlphaEnv_v01_Attention):
    """
    Shielded PPO — Attention + Continuous with safety override.

    Shield Architecture (applied in _apply_rl_actions):
        Raw PPO Action → TTC Check → RSS Brake Check → Right-of-Way Check → Safe Action

    Tracks all shield interventions for analysis.
    """

    # ── Shield Tuning Constants ──
    TTC_THRESHOLD = 2.0          # seconds — if TTC < this, shield activates
    RSS_MIN_GAP = 3.0            # meters — minimum safe following distance
    RSS_REACTION_TIME = 0.5      # seconds — assumed reaction time
    EMERGENCY_DECEL = -4.5       # m/s² — maximum emergency braking

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        # ── Shield telemetry ──
        self.shield_stats = {
            'total_steps': 0,
            'ttc_overrides': 0,
            'rss_overrides': 0,
            'row_overrides': 0,      # right-of-way
            'total_overrides': 0,
        }

    def _apply_rl_actions(self, rl_action):
        """
        Intercepts the PPO action, applies safety checks, and potentially
        overrides it with a safer action before sending to SUMO.
        """
        max_accel = self.env_params.additional_params['max_accel']
        max_decel = self.env_params.additional_params['max_decel']

        # 1. Extract raw action
        try:
            action_val = float(rl_action[0]) if isinstance(rl_action, (list, np.ndarray)) else float(rl_action)
        except (TypeError, ValueError):
            action_val = 0.0
        if np.isnan(action_val) or np.isinf(action_val):
            action_val = 0.0

        # Denormalize: [-1,1] → [-max_decel, max_accel]
        if action_val >= 0:
            proposed_accel = action_val * max_accel
        else:
            proposed_accel = action_val * max_decel

        self.shield_stats['total_steps'] += 1

        # 2. Run shield checks (only if agent is alive)
        if self.agent_id in self.k.vehicle.get_ids():
            safe_accel, override_reason = self._shield_check(proposed_accel)

            if override_reason:
                self.shield_stats['total_overrides'] += 1
                if 'ttc' in override_reason:
                    self.shield_stats['ttc_overrides'] += 1
                if 'rss' in override_reason:
                    self.shield_stats['rss_overrides'] += 1
                if 'row' in override_reason:
                    self.shield_stats['row_overrides'] += 1
        else:
            safe_accel = proposed_accel

        # 3. Apply the (possibly overridden) action
        rl_ids = [self.agent_id]
        if self.agent_id in self.k.vehicle.get_ids():
            self.k.vehicle.apply_acceleration(rl_ids, [safe_accel])

    def _shield_check(self, proposed_accel):
        """
        Three-layer safety shield:
          Layer 1 — Dynamic TTC: brake if time-to-collision is dangerously low
          Layer 2 — RSS safe distance: ensure minimum following gap
          Layer 3 — Right-of-way: yield if another vehicle has priority

        Returns:
            (safe_accel, override_reason) — reason is None if no override
        """
        ego_id = self.agent_id
        ego_speed = self.k.vehicle.get_speed(ego_id)
        if ego_speed is None or ego_speed < 0:
            ego_speed = 0.0
        max_speed = self.k.network.max_speed()

        ego_pos = self.k.vehicle.get_2d_position(ego_id)
        if ego_pos is None or ego_pos == -1001:
            return proposed_accel, None

        ego_x, ego_y = ego_pos
        ego_heading = self.k.vehicle.get_heading(ego_id)
        ego_angle_rad = np.radians((-ego_heading) + 90)

        # Use cached neighbor info from last observation
        # Note: on invalid position get_state() may return {} — normalise to []
        raw = getattr(self, 'last_neighbors_info', [])
        neighbors = raw if isinstance(raw, list) else []
        if not neighbors:
            return proposed_accel, None

        override_reason = []
        safe_accel = proposed_accel

        for n in neighbors:
            dist = n.get('distance', self.perception_radius)
            # 'v' in neighbors_info is the RAW speed in m/s (not normalised)
            # 'ego_dist_to_cp' is normalised by perception_radius → denormalise it
            other_speed = n.get('v', 0.0)  # already m/s — do NOT multiply by max_speed
            ego_dist_to_cp = n.get('ego_dist_to_cp', 1.0) * self.perception_radius  # denormalise
            delta_eta = n.get('d_eta', 1.0)  # tanh-normalised, already in [-1, 1]

            # ── Layer 1: Path-Based TTC (works for all geometries incl. left turns) ──
            # Use ego_dist_to_cp (path distance to conflict point) not raw euclidean dist.
            # Only fire if both vehicles will arrive at conflict point at similar times.
            if ego_speed > 0.1 and ego_dist_to_cp > 0.0:
                ego_ttc = ego_dist_to_cp / ego_speed  # seconds until ego reaches conflict point
                # delta_eta is tanh-normalised: |d_eta| < 0.5 ≈ within ~1s of each other
                if ego_ttc < self.TTC_THRESHOLD and abs(delta_eta) < 0.5:
                    # Deceleration needed to stop before conflict point: v²=2as → a=-v²/(2s)
                    safe_dist = max(ego_dist_to_cp - self.RSS_MIN_GAP, 0.5)
                    required_decel = -(ego_speed ** 2) / (2.0 * safe_dist)
                    required_decel = max(required_decel, self.EMERGENCY_DECEL)
                    if safe_accel > required_decel:
                        safe_accel = required_decel
                        override_reason.append('ttc')

            # ── Layer 2: RSS Safe Distance ──
            # d_safe = v * t_react + v² / (2 * a_max_brake)
            d_safe = (ego_speed * self.RSS_REACTION_TIME +
                      ego_speed ** 2 / (2.0 * abs(self.EMERGENCY_DECEL)))
            d_safe += self.RSS_MIN_GAP

            if ego_dist_to_cp < d_safe and ego_dist_to_cp < self.perception_radius * 0.6:
                # We're too close to conflict point — force braking
                # delta_eta is tanh-normalised: |d_eta| < 0.6 ≈ within ~1.5s of simultaneous arrival
                if abs(delta_eta) < 0.6:
                    brake_intensity = max(-2.0, self.EMERGENCY_DECEL * (1.0 - ego_dist_to_cp / d_safe))
                    if safe_accel > brake_intensity:
                        safe_accel = brake_intensity
                        if 'rss' not in override_reason:
                            override_reason.append('rss')

            # ── Layer 3: Right-of-Way (Yield) ──
            # If delta_eta < 0, the other vehicle arrives at conflict point first
            # Ego should yield (brake) if it doesn't have priority
            if abs(delta_eta) < 0.15 and ego_dist_to_cp < 15.0:
                # Near-simultaneous arrival at conflict point
                # Apply right-before-left rule: check relative heading
                other_sin = n.get('sin', 0.0)
                other_cos = n.get('cos', 0.0)

                ego_sin = np.sin(ego_angle_rad)
                ego_cos = np.cos(ego_angle_rad)

                # Cross product: positive means other is to our right (has priority)
                cross = ego_cos * other_sin - ego_sin * other_cos
                if cross > 0.3:  # other is to our right → they have priority
                    yield_decel = max(-1.5, -ego_speed * 0.3)
                    if safe_accel > yield_decel:
                        safe_accel = yield_decel
                        if 'row' not in override_reason:
                            override_reason.append('row')

        reason_str = '+'.join(override_reason) if override_reason else None
        return safe_accel, reason_str

    def compute_reward(self, agent_id, fail, goal_reached, current_action=None):
        """
        Same reward as parent, plus a small bonus for shield compliance
        (agent learns to work with the shield, not against it).
        """
        base_reward = super().compute_reward(agent_id, fail, goal_reached, current_action)

        # Small penalty when shield overrides (teaches agent to be safer naturally)
        if self.shield_stats['total_steps'] > 0:
            recent_override = (self.shield_stats['total_overrides'] /
                               max(self.shield_stats['total_steps'], 1))
            if recent_override > 0.5:
                base_reward -= 0.005  # mild penalty for relying too much on shield

        return base_reward

    def reset(self, **kwargs):
        """Reset shield stats each episode."""
        self.shield_stats = {
            'total_steps': 0,
            'ttc_overrides': 0,
            'rss_overrides': 0,
            'row_overrides': 0,
            'total_overrides': 0,
        }
        return super().reset(**kwargs)

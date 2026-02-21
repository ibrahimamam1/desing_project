# ============ HIGHWAY-ENV INTERSECTION WRAPPER v0.2 ============
#
# Thin Gymnasium wrapper around highway-env's "intersection-v0".
# Keeps the same telemetry dict structure as v0.1 so that
# TrafficCallbacks works unchanged.

import gymnasium as gym
import numpy as np
import highway_env

class AlphaEnvV02(gym.Wrapper):
    """
    Single-agent intersection environment built on highway-env.

    Observation : Flattened kinematics matrix (vehicles_count × features).
    Action      : DiscreteMetaAction (5 actions).
    Reward      : Custom shaping — speed + arrival bonus − collision penalty.
    """

    # ---------- tunables ----------
    COLLISION_REWARD = -10.0
    ARRIVAL_REWARD   =  10.0
    SPEED_REWARD_SCALE = 1.0
    MAX_SPEED = 15.0          # m/s (intersection speeds)

    VEHICLES_COUNT = 6        # ego + 5 neighbours (matches v0.1)
    OBS_FEATURES = [
        "presence", "x", "y", "vx", "vy", "cos_h", "sin_h",
    ]

    def __init__(self, env_config: dict | None = None):
        env_config = env_config or {}

        highway_cfg = {
            # --- observation ---
            "observation": {
                "type": "Kinematics",
                "vehicles_count": self.VEHICLES_COUNT,
                "features": self.OBS_FEATURES,
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": False,       # ego-centric
                "flatten": False,        # we flatten ourselves
                "observe_intentions": False,
            },
            # --- action ---
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False,
            },
            # --- simulation ---
            "duration": 30,              # seconds per episode decision horizon
            "initial_vehicle_count": 10,
            "spawn_probability": 0.6,
            "destination": "o1",
            "collision_reward": 0,       # we compute our own
            "normalize_reward": False,
            # --- rendering ---
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
        }

        # Allow env_config overrides
        render_mode = env_config.pop("render_mode", None)
        highway_cfg.update(env_config)

        if render_mode:
            inner = gym.make("intersection-v1", render_mode=render_mode)
        else:
            inner = gym.make("intersection-v1")

        inner.unwrapped.configure(highway_cfg)
        inner.reset()  # apply the config
        super().__init__(inner)

        # Flatten observation space:  (vehicles_count, features) → (vehicles_count * features,)
        obs_size = self.VEHICLES_COUNT * len(self.OBS_FEATURES)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # Telemetry accumulators (mirrors v0.1)
        self._init_telemetry()

    # ─────────── telemetry ───────────
    def _init_telemetry(self):
        self._tel = {
            "speeds": [],
            "collisions": 0,
            "steps": 0,
        }

    def _compute_telemetry_stats(self):
        avg_speed = float(np.mean(self._tel["speeds"])) if self._tel["speeds"] else 0.0
        return {
            "number_of_collisions": self._tel["collisions"],
            "avg_speed": avg_speed,
        }

    # ─────────── core API ───────────
    def _flatten_obs(self, obs):
        """Flatten (vehicles_count, features) → (obs_size,) and clip to [-1, 1]."""
        flat = np.asarray(obs, dtype=np.float32).flatten()
        # Pad / truncate to expected size
        expected = self.VEHICLES_COUNT * len(self.OBS_FEATURES)
        if flat.shape[0] < expected:
            flat = np.pad(flat, (0, expected - flat.shape[0]))
        elif flat.shape[0] > expected:
            flat = flat[:expected]
        return np.clip(flat, -1.0, 1.0)

    def reset(self, *, seed=None, options=None):
        self._init_telemetry()
        obs, info = self.env.reset(seed=seed, options=options)
        return self._flatten_obs(obs), info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        flat_obs = self._flatten_obs(obs)

        # ---- custom reward ----
        crashed = info.get("crashed", False)
        arrived = info.get("is_success", False)

        if crashed:
            reward = self.COLLISION_REWARD
            self._tel["collisions"] += 1
        elif arrived:
            reward = self.ARRIVAL_REWARD
        else:
            # Speed-based dense reward
            ego = self.env.unwrapped.vehicle
            speed = np.linalg.norm([ego.speed]) if ego is not None else 0.0
            reward = self.SPEED_REWARD_SCALE * (speed / self.MAX_SPEED)

        # ---- telemetry ----
        self._tel["steps"] += 1
        ego = self.env.unwrapped.vehicle
        if ego is not None:
            self._tel["speeds"].append(float(np.linalg.norm([ego.speed])))

        # Attach telemetry on terminal steps (same structure as v0.1)
        if terminated or truncated:
            info["telemetry"] = self._compute_telemetry_stats()

        return flat_obs, reward, terminated, truncated, info

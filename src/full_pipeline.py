#!/usr/bin/env python3
"""
Full Training + Evaluation Pipeline for all 5 controller variants.
Trains each variant for 1.5M timesteps on GPU, then evaluates across
6 traffic scenarios × 3 intentions = 18 configurations per variant.
Finally generates collision-rate line plots and heatmap.

Usage:
    conda activate flow
    python src/full_pipeline.py --mode train
    python src/full_pipeline.py --mode eval
    python src/full_pipeline.py --mode plot
    python src/full_pipeline.py --mode all
"""
import argparse, os, sys, csv, json, warnings, time, glob, random, zipfile, zlib, gc, signal, subprocess
from copy import deepcopy
from datetime import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "envs"))
sys.path.insert(0, os.path.join(ROOT, "src", "models"))

import numpy as np
from networks.uniform_random import UniformRandomNetwork
from networks.all_straight import AllStraghtNetwork
from networks.all_left import AllLeftNetwork
from networks.asymetric_random import AsymmetricRandomNetwork
from flow.core.params import (
    VehicleParams, NetParams, InitialConfig, TrafficLightParams,
    EnvParams, SumoParams, SumoCarFollowingParams, InFlows,
)
from flow.controllers import RLController, IDMController
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from src.models.attention_model import AttentionFeatureExtractor

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
NET_FILE = os.path.join(ROOT, "networks", "100m_right_before_left.net.xml")
CHECKPOINT_BASE = os.path.join(ROOT, "checkpoints", "v0_1_full")
EVAL_OUTPUT_DIR = os.path.join(ROOT, "eval_results")
PLOT_OUTPUT_DIR = os.path.join(ROOT, "plots")
TB_DIR = os.path.join(ROOT, "tensorboard_logs", "v0_1_full")

TOTAL_TIMESTEPS = 1_500_000
# How often to write a resumable checkpoint, in environment steps. Training
# runs for many hours and may be interrupted by a power cut, so this bounds
# how much progress a hard stop can cost.
CHECKPOINT_EVERY = 25_000
KEEP_LAST_CHECKPOINTS = 4
NUM_WORKERS = 8
N_EVAL_EPISODES = 42  # matches n_sims=42 in v0_1_evaluate.py
EVAL_SEED = 42

# All 6 variants
VERSIONS = [
    "attention_continous",
    "attention_discrete",
    "heuristic_attention_continous",
    "heuristic_attention_discrete",
    "shielded_attention_continous",
]

VERSION_LABELS = {
    "attention_continous": "Attention + Continuous",
    "attention_discrete": "Attention + Discrete",
    "heuristic_attention_continous": "Heuristic + Continuous",
    "heuristic_attention_discrete": "Heuristic + Discrete",
    "shielded_attention_continous": "Shielded + Continuous",
}

# Intentions
INTENTIONS = {
    "all_straight": AllStraghtNetwork,
    "all_left": AllLeftNetwork,
    "uniform_random": UniformRandomNetwork,
    "asymmetric_random": AsymmetricRandomNetwork,
}

# Traffic scenarios (increasing total flow)
# Rates and scenario compositions match src/test/v0_1_evaluate.py so that
# results from this pipeline can be put beside the chapter 9 numbers.
high, medium, low_r = 400, 275, 150

# Each scenario is a list of flow compositions; one is drawn per episode,
# as in v0_1_evaluate.py, so a scenario covers its variants rather than a
# single fixed arrangement.
SCENARIOS = {
    "Sc1_All_Low":   [{"N": low_r, "S": low_r, "W": low_r, "E": low_r}],
    "Sc6_Mixed_ML":  [
        {"N": medium, "S": medium, "W": low_r,  "E": low_r},
        {"N": medium, "S": low_r,  "W": medium, "E": low_r},
        {"N": low_r,  "S": low_r,  "W": medium, "E": medium},
    ],
    "Sc3_All_Med":   [{"N": medium, "S": medium, "W": medium, "E": medium}],
    "Sc4_Mixed_2H":  [
        {"N": high,  "S": high,  "W": low_r, "E": low_r},
        {"N": low_r, "S": low_r, "W": high,  "E": high},
        {"N": high,  "S": low_r, "W": high,  "E": low_r},
        {"N": low_r, "S": high,  "W": low_r, "E": high},
    ],
    "Sc5_Mixed_1H":  [
        {"N": high,   "S": medium, "W": medium, "E": medium},
        {"N": medium, "S": high,   "W": medium, "E": medium},
        {"N": medium, "S": medium, "W": high,   "E": medium},
        {"N": medium, "S": medium, "W": medium, "E": high},
    ],
    "Sc2_All_High":  [
        {"N": high,   "S": high,   "W": high,   "E": high},
        {"N": high,   "S": high,   "W": high,   "E": medium},
        {"N": high,   "S": high,   "W": medium, "E": high},
        {"N": high,   "S": medium, "W": high,   "E": high},
        {"N": medium, "S": high,   "W": high,   "E": high},
    ],
}

DIR_MAP = {"N": "E#T-X", "E": "E#R-X", "S": "E#D-X", "W": "E#L-X"}

# Simulation params
max_accel = 2.6; max_decel = 4.5; max_speed = 55; sim_step = 0.25
horizon = 180; warmup_steps = 50  # Match teammate: 50-step warmup for realistic traffic buildup

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def _make_vehicles():
    vehicles = VehicleParams()
    # RL vehicle: speed_mode=0 so PPO has full control over acceleration
    rl_cfp = SumoCarFollowingParams(speed_mode=0, accel=max_accel, decel=max_decel,
        sigma=0, tau=0.8, min_gap=2.5, max_speed=max_speed,
        speed_factor=1.0, speed_dev=0.1, impatience=0.0, car_follow_model="IDM")
    # NonRL vehicles: speed_mode=31 so they respect traffic rules (braking, yielding)
    # This matches teammate's setup for a fair comparison
    nonrl_cfp = SumoCarFollowingParams(speed_mode=31, accel=max_accel, decel=max_decel,
        sigma=0, tau=0.8, min_gap=2.5, max_speed=max_speed,
        speed_factor=1.0, speed_dev=0.1, impatience=0.0, car_follow_model="IDM")
    vehicles.add(veh_id="RL", acceleration_controller=(RLController, {}),
        initial_speed=0, num_vehicles=0, car_following_params=rl_cfp,
        lane_change_params=None, color="blue")
    vehicles.add(veh_id="NonRL", acceleration_controller=(IDMController, {}),
        initial_speed=0, num_vehicles=0, car_following_params=nonrl_cfp,
        lane_change_params=None, color="red")
    return vehicles

RL_EDGE = "E#L-X"   # west approach, reserved for the RL vehicle

def _make_inflow(rates):
    inflow = InFlows()
    # Background traffic on the three approaches the agent does not spawn on.
    # v0_1_evaluate.py leaves the RL edge clear; filling it too put NonRL cars
    # directly ahead of and behind the agent in its own lane, which is a
    # different problem from the one chapter 9 measures.
    for d, edge in DIR_MAP.items():
        if edge == RL_EDGE:
            continue
        inflow.add(veh_type="NonRL", edge=edge, probability=rates[d]/3600,
                   depart_lane=0, depart_speed=0, begin=1, color="green")
    inflow.add(veh_type="RL", edge=RL_EDGE, probability=0.8,
               depart_lane=0, depart_speed=0, begin=warmup_steps, color="green")
    return inflow

def _make_flow_params(network_cls, rates):
    return dict(
        network=network_cls, sim=SumoParams(
            port=None, sim_step=sim_step, lateral_resolution=None,
            no_step_log=True, render=False, save_render=False, sight_radius=25,
            show_radius=False, pxpm=2, force_color_update=False, overtake_right=False,
            seed=42, restart_instance=True, print_warnings=False, teleport_time=0,
            num_clients=1, color_by_speed=False, use_ballistic=False),
        env=EnvParams(additional_params={"max_accel": max_accel, "max_decel": max_decel,
            "target_velocity": max_speed, "sort_vehicles": False},
            horizon=horizon, warmup_steps=warmup_steps, sims_per_step=1, evaluate=False, clip_actions=True),
        net=NetParams(osm_path=None, template=NET_FILE, inflows=_make_inflow(rates)),
        veh=_make_vehicles(),
        initial=InitialConfig(shuffle=False, spacing="uniform", min_gap=12,
            perturbation=5.0, x0=5, bunching=0, lanes_distribution=float("inf"),
            edges_distribution=["E#D-X", "E#L-X", "E#R-X", "E#T-X"]),
    )

def _get_env_class(version):
    if version == "attention_continous":
        from src.envs.alpha_env_v01_attention_continous import AlphaEnv_v01_Attention
        return AlphaEnv_v01_Attention
    elif version == "attention_discrete":
        from src.envs.alpha_env_v01_attention_discrete import AlphaEnv_v01_AttentionDiscrete
        return AlphaEnv_v01_AttentionDiscrete
    elif version == "heuristic_attention_continous":
        from src.envs.alpha_env_v01_heuristic_attention_continous import AlphaEnv_v01_HeuristicAttention
        return AlphaEnv_v01_HeuristicAttention
    elif version == "heuristic_attention_discrete":
        from src.envs.alpha_env_v01_heuristic_attention_discrete import AlphaEnv_v01_HeuristicAttentionDiscrete
        return AlphaEnv_v01_HeuristicAttentionDiscrete
    elif version == "shielded_attention_continous":
        from src.envs.alpha_env_v01_shielded_attention_continous import AlphaEnv_v01_ShieldedAttention
        return AlphaEnv_v01_ShieldedAttention
    else:
        raise ValueError(f"Unknown version: {version}")

def _needs_attention(version):
    return "attention" in version or "shielded" in version

def _is_discrete(version):
    return "discrete" in version

def _get_obs_dims(version):
    # Observation layout for every attention variant:
    # [ego(4)] + [leader(3)] + [neighbour(5) x 5] + [mask(5)] = 37
    return 4, 3, 5, 5

def _policy_kwargs(version):
    if not _needs_attention(version):
        return None

    ego_f, leader_f, neigh_f, max_k = _get_obs_dims(version)
    # leader_features must be passed explicitly. The extractor slices the
    # observation at ego_features + leader_features, so if this does not match
    # the env the neighbour block and the mask are read from the wrong offsets.
    return dict(
        features_extractor_class=AttentionFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256, ego_features=ego_f,
            leader_features=leader_f, neighbor_features=neigh_f,
            max_neighbors=max_k, embed_dim=64, num_heads=4),
        net_arch=dict(pi=[256, 256], vf=[256, 256]))

def lr_schedule(initial=3e-4, floor=1e-5):
    def func(progress):
        return max(floor, progress * initial)
    return func

class LogCallback(BaseCallback):
    def __init__(self, version):
        super().__init__(verbose=0)
        self.version = version
        self.ep = 0
    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "telemetry" in info:
                t = info["telemetry"]
                if t is None: continue
                self.ep += 1
                col = "💥" if t.get("agent_collision") else "✅"
                if self.ep % 50 == 0:
                    print(f"  [{self.version}] Ep {self.ep:4d} {col}")
        return True

# ═══════════════════════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════════════════════

def _checkpoint_steps(path):
    """Step count encoded in a checkpoint filename, or None if unparseable."""
    try:
        base = os.path.basename(path)
        return int(base.replace("rl_model_", "").replace("_steps.zip", ""))
    except ValueError:
        return None

def list_checkpoints(ckpt_dir):
    """All checkpoints in the directory, newest first."""
    found = []
    for f in glob.glob(os.path.join(ckpt_dir, "rl_model_*_steps.zip")):
        steps = _checkpoint_steps(f)
        if steps is not None:
            found.append((steps, f))
    found.sort(reverse=True)
    return found

def get_latest_checkpoint(ckpt_dir):
    """
    Newest checkpoint that actually loads.

    A checkpoint written while the machine lost power can be truncated. Walk
    back through the saved files until one opens cleanly, so an interrupted
    run resumes from the last good point instead of dying on a corrupt zip.
    """
    for steps, path in list_checkpoints(ckpt_dir):
        try:
            with zipfile.ZipFile(path) as zf:
                if zf.testzip() is not None:
                    raise zipfile.BadZipFile("failed CRC check")
            return path, steps
        except (zipfile.BadZipFile, OSError) as e:
            print(f"  [WARN] discarding unreadable checkpoint {os.path.basename(path)}: {e}")
            continue
    return None, 0

def prune_checkpoints(ckpt_dir, keep=KEEP_LAST_CHECKPOINTS):
    """Keep only the most recent few checkpoints so disk use stays bounded."""
    entries = list_checkpoints(ckpt_dir)
    for _, path in entries[keep:]:
        try:
            os.remove(path)
        except OSError:
            pass

def train_version(version):
    # Skip if checkpoint already exists
    ckpt_path = os.path.join(CHECKPOINT_BASE, version, "final_model.zip")
    if os.path.exists(ckpt_path):
        print(f"\n  ⏭️  SKIPPING {VERSION_LABELS[version]} — checkpoint exists: {ckpt_path}")
        return ckpt_path

    print(f"\n{'='*60}")
    print(f"  TRAINING: {VERSION_LABELS[version]}")
    print(f"  Timesteps: {TOTAL_TIMESTEPS:,} | Workers: {NUM_WORKERS} | GPU: cuda")
    print(f"{'='*60}\n")

    ckpt_dir = os.path.join(CHECKPOINT_BASE, version)
    os.makedirs(ckpt_dir, exist_ok=True)
    prune_checkpoints(ckpt_dir)

    # Default training uses uniform_random with medium traffic
    train_rates = {"N": 275, "S": 275, "W": 275, "E": 275}
    fp = _make_flow_params(UniformRandomNetwork, train_rates)
    EnvClass = _get_env_class(version)

    def make_env():
        p = fp
        network = p["network"](name="Train", vehicles=deepcopy(p["veh"]),
            net_params=p["net"], initial_config=p["initial"],
            traffic_lights=TrafficLightParams())
        env = EnvClass(env_params=p["env"], sim_params=deepcopy(p["sim"]),
            network=network, simulator="traci")
        return Monitor(env)

    vec_env = SubprocVecEnv([make_env for _ in range(NUM_WORKERS)])

    latest_ckpt, completed_steps = get_latest_checkpoint(ckpt_dir)
    remaining_steps = TOTAL_TIMESTEPS - completed_steps

    if latest_ckpt and completed_steps > 0:
        print(f"  🔄 Resuming from checkpoint: {latest_ckpt} ({completed_steps:,} steps completed)")
        model = PPO.load(
            latest_ckpt,
            env=vec_env,
            device="cuda",
            tensorboard_log=os.path.join(TB_DIR, version),
        )
    else:
        model = PPO(
            policy="MlpPolicy", env=vec_env,
            policy_kwargs=_policy_kwargs(version),
            learning_rate=lr_schedule(3e-4, 1e-5),
            n_steps=1024, batch_size=256, n_epochs=10,
            gamma=0.98, gae_lambda=0.95, clip_range=0.25,
            max_grad_norm=0.5, ent_coef=0.01,
            tensorboard_log=os.path.join(TB_DIR, version),
            verbose=1, device="cuda",
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, CHECKPOINT_EVERY // NUM_WORKERS),
        save_path=ckpt_dir,
        name_prefix="rl_model"
    )
    callbacks = CallbackList([LogCallback(version), checkpoint_callback])

    t0 = time.time()
    if remaining_steps > 0:
        model.learn(total_timesteps=remaining_steps, callback=callbacks, progress_bar=True, reset_num_timesteps=False)
    elapsed = time.time() - t0

    save_path = os.path.join(ckpt_dir, "final_model")
    model.save(save_path)
    vec_env.close()
    prune_checkpoints(ckpt_dir)

    print(f"\n  ✅ {VERSION_LABELS[version]} done in {elapsed/60:.1f} min")
    print(f"  Saved: {save_path}.zip\n")
    return save_path + ".zip"

def show_status():
    """Per-variant training progress, so an interrupted sweep can be picked up."""
    print(f"\n{'='*72}")
    print(f"  TRAINING STATUS  (target {TOTAL_TIMESTEPS:,} steps per variant)")
    print(f"{'='*72}")
    total_done = 0
    for v in VERSIONS:
        ckpt_dir = os.path.join(CHECKPOINT_BASE, v)
        final = os.path.join(ckpt_dir, "final_model.zip")
        if os.path.exists(final):
            print(f"  {VERSION_LABELS[v]:32s}  DONE")
            total_done += TOTAL_TIMESTEPS
            continue
        _, steps = get_latest_checkpoint(ckpt_dir)
        pct = 100.0 * steps / TOTAL_TIMESTEPS
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"  {VERSION_LABELS[v]:32s}  [{bar}] {steps:>9,} steps  {pct:5.1f}%")
        total_done += steps
    overall = 100.0 * total_done / (TOTAL_TIMESTEPS * len(VERSIONS))
    print(f"{'-'*72}")
    print(f"  overall: {total_done:,} / {TOTAL_TIMESTEPS * len(VERSIONS):,} steps  ({overall:.1f}%)")
    print(f"{'='*72}")

    total_configs = len(INTENTIONS) * len(SCENARIOS)
    print(f"\n  EVALUATION  ({total_configs} configurations per variant)")
    print(f"{'-'*72}")
    eval_done = 0
    for v in VERSIONS:
        csv_path = os.path.join(EVAL_OUTPUT_DIR, f"{v}_results.csv")
        if os.path.exists(csv_path):
            print(f"  {VERSION_LABELS[v]:32s}  DONE")
            eval_done += total_configs
            continue
        _, completed = load_eval_progress(v)
        n = len(completed)
        eval_done += n
        bar = "#" * int(20 * n / total_configs) + "." * (20 - int(20 * n / total_configs))
        print(f"  {VERSION_LABELS[v]:32s}  [{bar}] {n:>2}/{total_configs}")
    ev_pct = 100.0 * eval_done / (total_configs * len(VERSIONS))
    print(f"{'-'*72}")
    print(f"  overall: {eval_done} / {total_configs * len(VERSIONS)} configurations  ({ev_pct:.1f}%)")
    print(f"{'='*72}\n")

def train_all():
    results = {}
    for v in VERSIONS:
        results[v] = train_version(v)
    print("\n" + "="*60)
    print("  ALL TRAINING COMPLETE")
    for v, p in results.items():
        print(f"  {VERSION_LABELS[v]:35s} → {p}")
    print("="*60)

# ═══════════════════════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════════════════════


def _sumo_pids():
    """PIDs of currently running SUMO processes."""
    try:
        out = subprocess.run(["pgrep", "-f", "sumo"], capture_output=True, text=True)
        return {int(x) for x in out.stdout.split() if x.isdigit()}
    except Exception:
        return set()

def _reap_sumo(before):
    """
    Kill SUMO processes that appeared since `before` and outlived their env.

    Evaluation opens and closes one SUMO per episode, thousands of times.
    Any that survive env.close() accumulate and hold sockets and memory, so
    they are cleaned up explicitly. Only newly appeared PIDs are touched, so
    a concurrent training sweep is left alone.
    """
    for pid in _sumo_pids() - before:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

def _rss_mb():
    """Resident memory of this process in MB, for leak diagnostics."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return -1.0

def _eval_progress_path(version):
    return os.path.join(EVAL_OUTPUT_DIR, f"{version}_progress.jsonl")

def load_eval_progress(version):
    """
    Configurations already evaluated for this variant.

    Each finished configuration is appended as one JSON line, so an
    interrupted run resumes at the configuration boundary instead of
    restarting the whole variant. A line half-written when the machine lost
    power will not parse and is dropped, costing at most one configuration.
    """
    path = _eval_progress_path(version)
    rows, done = [], set()
    if not os.path.exists(path):
        return rows, done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [WARN] dropping incomplete progress line in {os.path.basename(path)}")
                continue
            key = (row.get("intention"), row.get("scenario"))
            if key in done:
                continue
            done.add(key)
            rows.append(row)
    return rows, done

def append_eval_progress(version, row):
    """Append one finished configuration, forcing it to disk before returning."""
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    with open(_eval_progress_path(version), "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())

def _config_seed(version, int_name, sc_name):
    """
    Deterministic seed per configuration.

    Seeding once per variant would make the flow compositions depend on how
    many configurations ran before, so a resumed run would not reproduce an
    uninterrupted one. Deriving the seed from the configuration itself keeps
    the episode sequence identical either way.
    """
    key = f"{EVAL_SEED}|{version}|{int_name}|{sc_name}".encode()
    return zlib.crc32(key) & 0xffffffff

def evaluate_version(version, model_path=None, out_name=None):
    from stable_baselines3.common.vec_env import DummyVecEnv

    if model_path is None:
        model_path = os.path.join(CHECKPOINT_BASE, version, "final_model.zip")
    if not os.path.exists(model_path):
        print(f"  ⚠️  Checkpoint not found: {model_path}, skipping {version}")
        return

    out_name = out_name or version
    csv_path = os.path.join(EVAL_OUTPUT_DIR, f"{out_name}_results.csv")
    if os.path.exists(csv_path):
        print(f"  ⏭️  SKIPPING {out_name} — CSV already exists: {csv_path}")
        return

    print(f"\n{'─'*60}")
    print(f"  EVALUATING: {VERSION_LABELS[version]}")
    print(f"{'─'*60}")

    model = PPO.load(model_path, device="cuda")
    EnvClass = _get_env_class(version)
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    all_results, completed = load_eval_progress(out_name)
    total_configs = len(INTENTIONS) * len(SCENARIOS)
    if completed:
        print(f"  resuming: {len(completed)}/{total_configs} configurations already done")

    for int_name, int_cls in INTENTIONS.items():
        for sc_name, rate_list in SCENARIOS.items():
            if (int_name, sc_name) in completed:
                print(f"    {int_name:20s} {sc_name:15s} - already done, skipping")
                continue

            # Seeded per configuration, so a resumed run draws the same flow
            # compositions as an uninterrupted one.
            rng = random.Random(_config_seed(version, int_name, sc_name))

            def make_env(_rates):
                def _thunk():
                    fp = _make_flow_params(int_cls, _rates)
                    network = fp["network"](name="Eval", vehicles=deepcopy(fp["veh"]),
                        net_params=fp["net"], initial_config=fp["initial"],
                        traffic_lights=TrafficLightParams())
                    return EnvClass(env_params=fp["env"], sim_params=deepcopy(fp["sim"]),
                        network=network, simulator="traci")
                return _thunk

            collisions = 0
            successes = 0
            travel_times = []
            waiting_times = []
            avg_speeds = []
            shield_steps = 0
            shield_overrides = 0
            shield_ttc = 0
            shield_rss = 0
            shield_row = 0

            for ep in range(N_EVAL_EPISODES):
                ep_rates = rng.choice(rate_list)
                sumo_before = _sumo_pids()
                env = None
                t = {}
                try:
                    env = DummyVecEnv([make_env(ep_rates)])
                    obs = env.reset()
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, dones, infos = env.step(action)
                        done = dones[0]
                    t = infos[0].get("telemetry", {}) or {}
                except Exception as exc:
                    # One bad episode should not abandon the configuration.
                    print(f"      [WARN] episode {ep} failed: {exc}")

                if t.get("agent_collision", False):
                    collisions += 1
                if t.get("agent_success", False):
                    successes += 1
                travel_times.append(t.get("agent_travel_time", 0))
                waiting_times.append(t.get("agent_waiting_time", 0))
                avg_speeds.append(t.get("agent_avg_speed", 0))

                # Shield counters, present only for the shielded variant.
                # Must be read before env.close(), while the info dict is alive.
                sh = t.get("shield_stats")
                if sh:
                    shield_steps += sh.get("total_steps", 0)
                    shield_overrides += sh.get("total_overrides", 0)
                    shield_ttc += sh.get("ttc_overrides", 0)
                    shield_rss += sh.get("rss_overrides", 0)
                    shield_row += sh.get("row_overrides", 0)

                # Tear down explicitly. env.close() alone leaves the env object
                # and its TraCI socket alive until the collector happens to run,
                # which over thousands of episodes exhausts system resources.
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass
                _reap_sumo(sumo_before)
                del env
                gc.collect()
                time.sleep(1.5)  # let the OS release the TraCI port

                if ep and ep % 20 == 0:
                    print(f"      [diag] episode {ep}: rss={_rss_mb():.0f}MB "
                          f"sumo={len(_sumo_pids())}")

            col_rate = 100.0 * collisions / N_EVAL_EPISODES
            success_rate = 100.0 * successes / N_EVAL_EPISODES
            avg_tt = np.mean(travel_times) if travel_times else 0
            avg_wt = np.mean(waiting_times) if waiting_times else 0
            avg_sp = np.mean(avg_speeds) if avg_speeds else 0

            row = {
                "version": out_name, "intention": int_name,
                "scenario": sc_name, "collision_rate": col_rate,
                "success_rate": success_rate,
                "avg_travel_time": avg_tt,
                "avg_waiting_time": avg_wt,
                "avg_speed": avg_sp,
                "n_episodes": N_EVAL_EPISODES,
                "collisions": collisions,
                # Zero for every unshielded variant; only the shielded env reports these.
                "shield_steps": shield_steps,
                "shield_overrides": shield_overrides,
                "shield_override_rate": (shield_overrides / shield_steps) if shield_steps else 0.0,
                "shield_ttc_overrides": shield_ttc,
                "shield_rss_overrides": shield_rss,
                "shield_row_overrides": shield_row,
            }
            all_results.append(row)
            append_eval_progress(out_name, row)
            msg = (f"    {int_name:20s} {sc_name:15s} → col={col_rate:5.1f}% "
                   f"succ={success_rate:5.1f}% tt={avg_tt:5.1f}s wait={avg_wt:5.1f}s")
            if shield_steps:
                msg += f" shield={100.0 * shield_overrides / shield_steps:4.1f}%"
            print(msg)

    # Resumed rows come back in whatever order they were finished, so restore
    # the canonical intention x scenario ordering before writing.
    _int_order = list(INTENTIONS.keys())
    _sc_order = list(SCENARIOS.keys())
    def _sort_key(r):
        try:
            return (_int_order.index(r["intention"]), _sc_order.index(r["scenario"]))
        except ValueError:
            return (len(_int_order), len(_sc_order))
    all_results.sort(key=_sort_key)

    # Save CSV
    csv_path = os.path.join(EVAL_OUTPUT_DIR, f"{out_name}_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f"  Saved: {csv_path}")
    return all_results

def evaluate_all():
    all_data = {}
    for v in VERSIONS:
        data = evaluate_version(v)
        if data:
            all_data[v] = data
    # Save combined
    combined = os.path.join(EVAL_OUTPUT_DIR, "all_results.json")
    with open(combined, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\n  Combined results: {combined}")

# ═══════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════
def plot_results():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.ticker as mticker

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    # Load all CSV results
    data = {}  # data[version][intention][scenario] = col_rate
    for v in VERSIONS:
        csv_path = os.path.join(EVAL_OUTPUT_DIR, f"{v}_results.csv")
        if not os.path.exists(csv_path):
            continue
        data[v] = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                intent = row["intention"]
                sc = row["scenario"]
                if intent not in data[v]:
                    data[v][intent] = {}
                data[v][intent][sc] = float(row["collision_rate"])

    if not data:
        print("  No evaluation data found. Run --mode eval first.")
        return

    sc_order = list(SCENARIOS.keys())
    int_order = list(INTENTIONS.keys())
    int_labels = {"all_straight": "All Straight", "all_left": "All Left",
                  "uniform_random": "Uniform Random", "asymmetric_random": "Asymmetric Random"}

    # ─── Plot 1: Collision Rate Line Charts (2×2 grid) ───
    # Colors match teammate's style exactly: blue, red, green, purple + yellow for shielded
    colors = {
        "heuristic_attention_discrete": "#1f77b4",   # blue  (teammate: Heuristic+Discrete)
        "heuristic_attention_continous": "#d62728",  # red   (teammate: Heuristic+Continuous)
        "attention_discrete": "#2ca02c",             # green (teammate: Attention+Discrete)
        "attention_continous": "#9467bd",            # purple(teammate: Attention+Continuous)
        "shielded_attention_continous": "#f0c400",   # yellow (our contribution)
    }
    markers = {
        "heuristic_attention_discrete": "o",
        "heuristic_attention_continous": "s",
        "attention_discrete": "^",
        "attention_continous": "D",
        "shielded_attention_continous": "*",
    }
    linestyles = {
        "heuristic_attention_discrete": "-",
        "heuristic_attention_continous": "--",
        "attention_discrete": "-.",
        "attention_continous": ":",
        "shielded_attention_continous": "-",
    }

    def smooth(vals, w=2):
        """Simple rolling-window average to reduce noise, keeping endpoints stable."""
        result = []
        for i in range(len(vals)):
            start = max(0, i - w)
            end   = min(len(vals), i + w + 1)
            result.append(np.mean(vals[start:end]))
        return result

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Collision Rate vs. Traffic Scenario", fontsize=16, fontweight="bold")

    for idx, intent in enumerate(int_order):
        ax = axes[idx // 2][idx % 2]
        ax.set_title(int_labels.get(intent, intent), fontsize=13, fontweight="bold")

        for v in VERSIONS:
            if v not in data or intent not in data[v]:
                continue
            vals = [data[v][intent].get(sc, 0) for sc in sc_order]
            smoothed = smooth(vals)
            ax.plot(range(len(sc_order)), smoothed,
                    color=colors.get(v, "#888"),
                    marker=markers.get(v, "o"),
                    linestyle=linestyles.get(v, "-"),
                    linewidth=2, markersize=7,
                    label=VERSION_LABELS.get(v, v))

        ax.set_xticks(range(len(sc_order)))
        sc_xlabels = ["Sc1\nAll Low", "Sc6\nMixed ML", "Sc3\nAll Med.",
                      "Sc4\nMixed 2H", "Sc5\nMixed 1H", "Sc2\nAll High"]
        ax.set_xticklabels(sc_xlabels, fontsize=9)
        ax.set_xlabel("Traffic Scenario (increasing total flow →)", fontsize=10)
        ax.set_ylabel("Collision Rate (%)", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.legend(title="Controller", fontsize=8, title_fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path1 = os.path.join(PLOT_OUTPUT_DIR, "collision_rate_lines.png")
    fig.savefig(path1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ─── Plot 2: Heatmap (Controller × Scenario, averaged over intentions) ───
    fig2, ax2 = plt.subplots(figsize=(14, 6))

    heatmap_data = []
    ylabels = []
    for v in VERSIONS:
        if v not in data:
            continue
        ylabels.append(VERSION_LABELS.get(v, v))
        row_vals = []
        for sc in sc_order:
            vals = [data[v].get(intent, {}).get(sc, 0) for intent in int_order]
            row_vals.append(np.mean(vals))
        heatmap_data.append(row_vals)

    heatmap_arr = np.array(heatmap_data)

    cmap = LinearSegmentedColormap.from_list("custom",
        ["#FFFDE7", "#FFF9C4", "#FFF176", "#FFD54F", "#FFB74D", "#FF8A65", "#E57373", "#C62828"])

    im = ax2.imshow(heatmap_arr, cmap=cmap, aspect="auto", vmin=0)
    cbar = fig2.colorbar(im, ax=ax2, label="Collision Rate (%)", shrink=0.8)

    ax2.set_xticks(range(len(sc_order)))
    ax2.set_xticklabels([s.replace("_", "\n") for s in sc_order], fontsize=10)
    ax2.set_yticks(range(len(ylabels)))
    ax2.set_yticklabels(ylabels, fontsize=11)
    ax2.set_xlabel("Traffic Scenario (increasing total flow →)", fontsize=12)
    ax2.set_ylabel("Controller", fontsize=12)
    ax2.set_title("Collision Rate — Controller × Traffic Scenario\n(averaged over intentions)",
                   fontsize=14, fontweight="bold")

    # Add text annotations
    for i in range(len(ylabels)):
        for j in range(len(sc_order)):
            val = heatmap_arr[i, j]
            color = "white" if val > heatmap_arr.max() * 0.6 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=11, fontweight="bold", color=color)

    fig2.tight_layout()
    path2 = os.path.join(PLOT_OUTPUT_DIR, "collision_rate_heatmap.png")
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {path2}")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "plot", "all", "status"], required=True)
    parser.add_argument("--version", default=None, help="Train/eval single version only")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override training timesteps (for smoke tests)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override evaluation episodes per configuration")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override number of training workers")
    parser.add_argument("--policy-from", default=None, dest="policy_from",
                        help="Evaluate --version's environment using another "
                             "variant's trained policy. Lets the same weights be "
                             "run with and without the shield, so the difference "
                             "is the shield and not a different training run.")
    args = parser.parse_args()

    if args.timesteps is not None:
        TOTAL_TIMESTEPS = args.timesteps
        print(f"  [override] TOTAL_TIMESTEPS = {TOTAL_TIMESTEPS:,}")
    if args.episodes is not None:
        N_EVAL_EPISODES = args.episodes
        print(f"  [override] N_EVAL_EPISODES = {N_EVAL_EPISODES}")
    if args.workers is not None:
        NUM_WORKERS = args.workers
        print(f"  [override] NUM_WORKERS = {NUM_WORKERS}")

    if args.mode == "status":
        show_status()
        sys.exit(0)

    if args.mode in ("train", "all"):
        if args.version:
            train_version(args.version)
        else:
            train_all()

    if args.mode in ("eval", "all"):
        if args.version and args.policy_from:
            mp = os.path.join(CHECKPOINT_BASE, args.policy_from, "final_model.zip")
            on = f"{args.version}__policy_{args.policy_from}"
            print(f"  environment: {args.version}\n  policy:      {args.policy_from}\n  output:      {on}")
            evaluate_version(args.version, model_path=mp, out_name=on)
        elif args.version:
            evaluate_version(args.version)
        else:
            evaluate_all()

    if args.mode in ("plot", "all"):
        plot_results()

"""
environment.py

Internet-of-Vehicles (IoV) Digital Twin (DT) synchronization environment.

This implements the MDP described in the proposal:

    State  : [vehicle speed, vehicle position (normalized), available bandwidth,
              network delay, current DT synchronization error]
    Action : 0 = TRANSMIT (send data now)
             1 = DELAY     (buffer and send later)
             2 = DROP      (discard data)
    Reward : encourages high DT accuracy (low error) while penalizing
              bandwidth usage and latency cost.

NOTE ON DATA REALISM
---------------------
The environment can run in two modes:

  1. Trace-driven mode from a CSV such as data/synthetic_data.csv.
  2. Stochastic mode when no trace is supplied.

The CSV mode expects columns like Speed_mps, Acc_mps2, Latency_ms,
Bandwidth_Mbps, and DT_Error_m. The stochastic fallback uses parametrized
models that are commonly used in vehicular-network research to approximate
real conditions:

  - Vehicle speed:      Ornstein-Uhlenbeck-like mean-reverting random walk
                         (mimics realistic accel/decel/cruise behavior)
  - Available bandwidth: base capacity + slow sinusoidal congestion cycle
                         + noise (mimics diurnal / rush-hour congestion)
  - Network delay:       inversely related to available bandwidth + noise
  - DT synchronization error: grows with vehicle dynamics (speed, accel)
                         when not updated, and is reduced by TRANSMIT
                         actions (proportional to link quality)

If you later replace data/synthetic_data.csv with a real NGSIM, VeReMi, or SUMO
export, keep the same physical columns and the rest of the RL pipeline can
stay unchanged.
"""

import numpy as np
import pandas as pd


TRANSMIT, DELAY, DROP = 0, 1, 2
ACTION_NAMES = {TRANSMIT: "TRANSMIT", DELAY: "DELAY", DROP: "DROP"}


class IoVDigitalTwinEnv:
    """A lightweight Gym-style environment (reset/step) with no external deps."""

    def __init__(
        self,
        episode_length=200,
        data_unit_cost=1.0,      # "bandwidth units" consumed by one transmission
        max_buffer_delay=5,      # steps a DELAY can be buffered before forced send
        error_growth_speed_coef=0.04,
        error_growth_base=0.10,
        transmit_error_reduction=0.9,   # fraction of error cleared on successful TX
        delay_error_growth_mult=1.5,    # error grows faster while buffered
        drop_error_growth_mult=2.5,     # error grows fastest when dropped
        accuracy_weight=2.2,     # relative weight of accuracy vs. cost in reward
        cost_weight=1.0,
        trace_path=None,
        trace_df=None,
        trace_loop=True,
        seed=None,
    ):
        self.episode_length = episode_length
        self.data_unit_cost = data_unit_cost
        self.max_buffer_delay = max_buffer_delay
        self.error_growth_speed_coef = error_growth_speed_coef
        self.error_growth_base = error_growth_base
        self.transmit_error_reduction = transmit_error_reduction
        self.delay_error_growth_mult = delay_error_growth_mult
        self.drop_error_growth_mult = drop_error_growth_mult
        self.accuracy_weight = accuracy_weight
        self.cost_weight = cost_weight
        self.trace_loop = trace_loop
        self.rng = np.random.default_rng(seed)

        self.trace = self._load_trace(trace_path, trace_df)
        self.uses_trace = self.trace is not None and len(self.trace) > 0
        self.trace_index = 0
        self.prev_trace_error = 0.0
        self.acceleration = 0.0
        self.heading = 0.0
        self.reset()

    # ------------------------------------------------------------------ #
    # World / physical model (stand-in for real V2X sensor + network data)
    # ------------------------------------------------------------------ #
    def _load_trace(self, trace_path, trace_df):
        if trace_df is None and trace_path is not None:
            trace_df = pd.read_csv(trace_path)
        if trace_df is None:
            return None

        aliases = {
            "speed": ["Speed_mps", "speed", "vehicle_speed", "Speed"],
            "acceleration": ["Acc_mps2", "acceleration", "accel", "Acceleration"],
            "heading": ["Heading_deg", "heading", "Heading"],
            "delay": ["Latency_ms", "delay", "latency", "Delay_ms"],
            "bandwidth": ["Bandwidth_Mbps", "bandwidth", "Bandwidth", "bw"],
            "dt_error": ["DT_Error_m", "dt_error", "DT_Error", "error"],
        }

        normalized = pd.DataFrame()
        for canonical, names in aliases.items():
            for name in names:
                if name in trace_df.columns:
                    normalized[canonical] = pd.to_numeric(trace_df[name], errors="coerce")
                    break

        required = ["speed", "delay", "bandwidth", "dt_error"]
        missing = [col for col in required if col not in normalized.columns]
        if missing:
            raise ValueError(f"Trace data is missing required columns: {missing}")

        if "acceleration" not in normalized.columns:
            normalized["acceleration"] = normalized["speed"].diff().fillna(0.0)
        if "heading" not in normalized.columns:
            normalized["heading"] = 0.0

        normalized = normalized.ffill().bfill().fillna(0.0)
        return normalized.reset_index(drop=True)

    def _step_trace_world(self):
        row = self.trace.iloc[self.trace_index]
        self.speed = float(row["speed"])
        self.bandwidth = float(np.clip(row["bandwidth"], 0.1, 1_000.0))
        self.delay = float(np.clip(row["delay"], 0.1, 5_000.0))
        self.acceleration = float(row["acceleration"])
        self.heading = float(row["heading"])
        self.trace_error = float(max(row["dt_error"], 0.0))

        if self.trace_loop:
            self.trace_index = (self.trace_index + 1) % len(self.trace)
        else:
            self.trace_index = min(self.trace_index + 1, len(self.trace) - 1)

    def _step_physical_world(self):
        if self.uses_trace:
            self._step_trace_world()
            return

        # Vehicle speed: mean-reverting random walk around a cruising speed,
        # occasionally perturbed (braking / accelerating events)
        target_speed = 20.0  # m/s (~72 km/h cruising)
        reversion = 0.08
        noise = self.rng.normal(0, 1.2)
        if self.rng.random() < 0.03:  # sudden brake/accel event
            noise += self.rng.normal(0, 6.0)
        self.speed = np.clip(
            self.speed + reversion * (target_speed - self.speed) + noise, 0, 35
        )
        self.position = (self.position + self.speed * 0.1) % 1000.0

        # Network bandwidth: base capacity + slow congestion cycle + noise
        t = self.t
        congestion_cycle = 6.0 * np.sin(2 * np.pi * t / 80.0)
        bw_noise = self.rng.normal(0, 1.5)
        self.bandwidth = float(np.clip(18.0 + congestion_cycle + bw_noise, 3.0, 30.0))

        # Delay: inversely related to bandwidth, plus noise
        self.delay = float(
            np.clip(50.0 / (self.bandwidth + 1.0) + self.rng.normal(0, 1.0), 1.0, 40.0)
        )

    # ------------------------------------------------------------------ #
    # RL API
    # ------------------------------------------------------------------ #
    def reset(self):
        self.t = 0
        self.speed = 20.0
        self.position = 0.0
        self.bandwidth = 18.0
        self.delay = 10.0
        self.dt_error = 0.0
        self.trace_error = 0.0
        self.prev_trace_error = 0.0
        self.buffered_steps = 0  # how long current DELAY buffer has been held
        if self.uses_trace:
            max_start = max(len(self.trace) - 1, 1)
            self.trace_index = int(self.rng.integers(0, max_start))
            row = self.trace.iloc[self.trace_index]
            self.speed = float(row["speed"])
            self.bandwidth = float(row["bandwidth"])
            self.delay = float(row["delay"])
            self.dt_error = float(max(row["dt_error"], 0.0))
            self.trace_error = self.dt_error
            self.prev_trace_error = self.dt_error
            self.acceleration = float(row["acceleration"])
            self.heading = float(row["heading"])
        return self._get_state()

    def _get_state(self):
        return np.array(
            [self.speed, self.bandwidth, self.delay, self.dt_error],
            dtype=np.float32,
        )

    def step(self, action):
        self._step_physical_world()

        # Acceleration and observed trace error make the drift data-aware.
        accel_factor = 1.0 + 0.12 * abs(getattr(self, "acceleration", 0.0))
        base_growth = self.error_growth_base + self.error_growth_speed_coef * (
            self.speed / 10.0
        ) * accel_factor
        if self.uses_trace:
            trace_delta = max(self.trace_error - self.prev_trace_error, 0.0)
            base_growth = max(base_growth, trace_delta)
            self.prev_trace_error = self.trace_error

        bandwidth_cost = 0.0
        latency_cost = 0.0
        forced_transmit = False
        delay_scale = max(self.delay, 1.0) / (250.0 if self.uses_trace else 40.0)

        if action == TRANSMIT:
            self.dt_error = self.dt_error * (1 - self.transmit_error_reduction) \
                + base_growth
            bandwidth_cost = self.data_unit_cost * (1.0 + 10.0 / (self.bandwidth + 1.0))
            latency_cost = delay_scale
            self.buffered_steps = 0

        elif action == DELAY:
            self.dt_error += base_growth * self.delay_error_growth_mult
            self.buffered_steps += 1
            if self.buffered_steps >= self.max_buffer_delay:
                # force a transmit to avoid runaway divergence -> extra cost
                self.dt_error = self.dt_error * (1 - self.transmit_error_reduction) \
                    + base_growth
                bandwidth_cost = self.data_unit_cost * (1.0 + 10.0 / (self.bandwidth + 1.0)) * 1.15
                latency_cost = delay_scale * 1.5
                self.buffered_steps = 0
                forced_transmit = True

        else:  # DROP
            self.dt_error += base_growth * self.drop_error_growth_mult
            self.buffered_steps = 0

        self.dt_error = float(np.clip(self.dt_error, 0.0, 50.0))

        # Reward: accuracy is the priority, bandwidth/latency are costs
        accuracy_term = -self.dt_error
        cost_term = -(bandwidth_cost + latency_cost)
        reward = self.accuracy_weight * accuracy_term + self.cost_weight * cost_term

        self.t += 1
        done = self.t >= self.episode_length

        info = {
            "bandwidth_used": bandwidth_cost,
            "dt_error": self.dt_error,
            "trace_dt_error": self.trace_error,
            "forced_transmit": forced_transmit,
            "action": action,
        }
        return self._get_state(), reward, done, info

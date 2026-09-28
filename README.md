# RL-Based Adaptive Data Reduction for IoV Digital Twin Synchronization

Code accompanying the proposal *"Integrated Sensing and Communication (ISAC)
with Digital Twins (DT) in a Vehicular Network / IoV"*. It implements the MDP
described in the proposal: state = sensing/vehicle data + communication/network
data + DT error, action = TRANSMIT / DELAY / DROP, reward = accuracy minus
bandwidth and latency cost. The default learner is Double Q-learning, with
standard Q-learning still available for comparison.

## Is this ISAC + RL?

Yes, this is an ISAC-inspired RL implementation. The "sensing" side is modeled
through vehicle/DT state inputs such as speed, acceleration, heading, and DT
error. The "communication" side is modeled through bandwidth and latency. The
RL agent learns when to transmit, delay, or drop updates based on those joint
sensing and communication conditions.

It is not a full physical-layer ISAC simulator. It does not model radar
waveforms, beamforming, channel estimation, OFDM, or RSU scheduling. The most
accurate description is: "ISAC-aware RL decision layer for IoV digital twin
synchronization."

## Data

By default, `src/train.py` uses `data/synthetic_data.csv`, which contains:

- `Speed_mps`, `Acc_mps2`, `Heading_deg` for sensing/vehicle context
- `Latency_ms`, `Bandwidth_Mbps` for communication/network context
- `DT_Error_m` for digital twin synchronization error

If no CSV is provided, `src/environment.py` falls back to a stochastic vehicular
network model with mean-reverting speed, changing bandwidth, delay, and DT
error dynamics.

For a larger experiment, keep the same column meanings and pass a new trace
with `python3 -m src.train --data your_trace.csv`. Useful dataset/simulation
options:

- **NGSIM** (Next Generation Simulation) vehicle trajectory data - real
  highway vehicle speed/position traces: https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm
- **VeReMi** - simulated but widely used V2X/VANET dataset with network-layer
  fields (useful for the bandwidth/delay side): search "VeReMi dataset".
- **SUMO + veins** - run your own vehicular network simulation and export
  speed/position/bandwidth/delay traces to CSV.

## Files

| File | Purpose |
|---|---|
| `src/environment.py` | The IoV Digital Twin MDP (state/action/reward), CSV trace loading, and fallback vehicle + network dynamics |
| `src/q_learning_agent.py` | Tabular Q-learning and Double Q-learning agents |
| `src/baselines.py` | Periodic and threshold-based transmission policies for comparison |
| `src/train.py` | Trains the agent, evaluates vs. baselines, saves charts/CSVs/dashboard to `results/` |
| `data/synthetic_data.csv` | Synthetic ISAC-style sensing + communication trace |
| `results/` | Generated outputs: training curves, comparison charts, comparison CSV, dashboard, saved Q-table |

## Running it

```bash
pip install -r requirements.txt
python3 -m src.train
```

Takes under a minute on a laptop CPU (tabular Q-learning, no GPU needed).

Useful options:

```bash
python3 -m src.train --algorithm double-q
python3 -m src.train --algorithm q-learning
python3 -m src.train --data data/synthetic_data.csv
```

Open `results/dashboard.html` after training for the UI.

## Algorithm

The default algorithm is **Double Q-learning**, which reduces overestimation
bias compared with standard Q-learning. Standard Q-learning is available with
`--algorithm q-learning`. Both methods are dependency-light and keep the
learned policy interpretable through a Q-table.

## Current results (from the run in `results/`)

The RL agent learns to be selective about when to transmit: it
transmits more often than the periodic baseline when the network is cheap
and DT error is climbing, and holds back when transmission is expensive.
In the current reward weighting, this yields:

- **~57% lower average DT synchronization error** than the periodic
  baseline on `data/synthetic_data.csv`, at the cost of using more bandwidth than
  the sparse periodic baseline.
- The `accuracy_weight` vs. `cost_weight` parameters in
  `IoVDigitalTwinEnv.__init__` control exactly where the agent lands on the
  accuracy/bandwidth trade-off (Pareto frontier). Lower `accuracy_weight`
  (e.g. 1.0-1.5) will push the agent toward the low-bandwidth end at some
  accuracy cost. This is worth reporting as a sweep in your paper's
  evaluation section, since that is the trade-off your problem statement
  describes.

Re-run `python3 -m src.train` after changing weights to regenerate `results/`.

## Next steps for your paper

1. Swap in a larger real trace (NGSIM/SUMO/VeReMi) for credible experimental
   results.
2. Sweep `accuracy_weight` to produce a Pareto curve (error vs. bandwidth)
   - this directly answers your "test how the system performs under
   different network and traffic conditions" objective.
3. Add a multi-vehicle version (currently single-vehicle) if your paper
   claims network-level results, since RSU/edge aggregation in your diagram
   implies multiple vehicles competing for shared bandwidth.

## Git Notes

Generated outputs, caches, proposal PDFs, and local Word documents are ignored
through `.gitignore`. The files to commit are the Python source files,
`data/synthetic_data.csv`, `requirements.txt`, `.gitignore`, and this README.

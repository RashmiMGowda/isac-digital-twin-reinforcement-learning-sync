"""
baselines.py

Two traditional (non-RL) baseline policies, as referenced in the proposal's
"Expected Outcome" section (comparing against fixed-rule / periodic
transmission methods).
"""

from .environment import TRANSMIT, DELAY


class PeriodicPolicy:
    """Transmit every `period` steps, delay otherwise."""

    def __init__(self, period=5):
        self.period = period
        self.t = 0

    def select_action(self, state):
        action = TRANSMIT if self.t % self.period == 0 else DELAY
        self.t += 1
        return action

    def reset(self):
        self.t = 0


class ThresholdPolicy:
    """Transmit only when the DT synchronization error exceeds a threshold."""

    def __init__(self, error_threshold=3.0):
        self.error_threshold = error_threshold

    def select_action(self, state):
        _, _, _, dt_error = state
        return TRANSMIT if dt_error >= self.error_threshold else DELAY

    def reset(self):
        pass

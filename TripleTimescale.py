# pip install birdgame river --upgrade

import math
import numpy as np
from collections import deque
from datetime import datetime, timezone

from birdgame import HORIZON
from birdgame.trackers.trackerbase import TrackerBase
from birdgame.stats.fewvar import FEWVar


class TripleTimescale(TrackerBase):
    """
    TripleTimescale Tracker

    Features:
    - Fast estimator (alpha=0.003) for 20-second EWMA
    - Slow estimator (alpha=0.00005) for 20-minute EWMA
    - Medium estimator (alpha=0.0003) for 3-minute EWMA
    - Dual rollover detection
    """

    def __init__(self, horizon=HORIZON):
        super().__init__(horizon)

        self.current_x = None
        self.tick_count = 0
        self.count = 0

        # Estimators
        self.ewa_fast = FEWVar(fading_factor=0.003)
        self.ewa_slow = FEWVar(fading_factor=0.00005)
        self.ewa_medium = FEWVar(fading_factor=0.0003)

        # Tracking
        self.recent_changes = deque(maxlen=100)
        self.recent_abs_changes = deque(maxlen=100)
        self.recent_likelihoods = deque(maxlen=100)

        # Rollover detection
        self.in_rollover = False
        self.time_based_rollover = False
        self.likelihood_based_rollover = False

        # Component weights
        self.fast_weight = 0.70
        self.slow_weight = 0.20
        self.safety_weight = 0.10

        # Baseline statistics
        self.baseline_variance = None
        self.rolling_variance = 0.0

    def _detect_rollover(self, timestamp):
        """Time-based rollover detection"""
        try:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            current_hour = dt.hour
            current_minute = dt.minute

            if current_hour == 20 and current_minute >= 55:
                return True
            elif current_hour == 21 and current_minute <= 35:
                return True
            else:
                return False
        except:
            return False

    def tick(self, payload, performance_metrics=None):
        x = payload['dove_location']
        t = payload['time']

        # Dual rollover detection
        self.time_based_rollover = self._detect_rollover(t)

        self.likelihood_based_rollover = False
        if performance_metrics:
            current_likelihood = performance_metrics.get('recent_likelihood_ewa', None)
            if current_likelihood is not None:
                self.recent_likelihoods.append(current_likelihood)

                if len(self.recent_likelihoods) >= 20:
                    recent_avg = np.mean(list(self.recent_likelihoods)[-20:])
                    if recent_avg > 0 and current_likelihood < recent_avg * 0.5:
                        self.likelihood_based_rollover = True

        self.in_rollover = self.time_based_rollover or self.likelihood_based_rollover

        # Standard tracking
        self.add_to_quarantine(t, x)
        self.current_x = x
        prev_x = self.pop_from_quarantine(t)

        if prev_x is not None:
            x_change = x - prev_x
            abs_change = abs(x_change)

            self.recent_changes.append(x_change)
            self.recent_abs_changes.append(abs_change)

            if len(self.recent_changes) >= 20:
                self.rolling_variance = np.var(self.recent_changes)
                if self.baseline_variance is None and self.count > 500:
                    self.baseline_variance = self.rolling_variance

            # Update estimators
            if self.count > 10:
                fast_var = self.ewa_fast.get()
                fast_threshold = 1.5 * math.sqrt(fast_var)
            else:
                fast_threshold = 0.5

            fast_winsorized = np.clip(x_change, -fast_threshold, fast_threshold)
            self.ewa_fast.update(fast_winsorized)

            if self.count > 100:
                slow_var = self.ewa_slow.get()
                slow_threshold = 3.0 * math.sqrt(slow_var)
            else:
                slow_threshold = 1.5

            slow_winsorized = np.clip(x_change, -slow_threshold, slow_threshold)
            self.ewa_slow.update(slow_winsorized)

            if self.count > 30:
                med_var = self.ewa_medium.get()
                med_threshold = 2.5 * math.sqrt(med_var)
            else:
                med_threshold = 1.0

            med_winsorized = np.clip(x_change, -med_threshold, med_threshold)
            self.ewa_medium.update(med_winsorized)

            # Adaptive weight adjustment
            if self.baseline_variance and self.baseline_variance > 0:
                vol_ratio = self.rolling_variance / self.baseline_variance

                if vol_ratio > 2.0:
                    self.fast_weight = 0.60
                    self.slow_weight = 0.30
                    self.safety_weight = 0.10
                elif vol_ratio < 0.5:
                    self.fast_weight = 0.80
                    self.slow_weight = 0.15
                    self.safety_weight = 0.05
                else:
                    self.fast_weight = 0.70
                    self.slow_weight = 0.20
                    self.safety_weight = 0.10

            self.count += 1

        self.tick_count += 1

    def predict(self):
        if self.tick_count < 30 or self.current_x is None:
            return None

        x_mean = self.current_x

        # Get scales
        try:
            fast_var = self.ewa_fast.get()
            fast_scale = math.sqrt(max(fast_var, 1e-10)) * 0.707
        except:
            fast_scale = 0.07

        try:
            slow_var = self.ewa_slow.get()
            slow_scale = math.sqrt(max(slow_var, 1e-10)) * 0.707
        except:
            slow_scale = 0.15

        try:
            med_var = self.ewa_medium.get()
            med_scale = math.sqrt(max(med_var, 1e-10)) * 0.707
        except:
            med_scale = 0.12

        # Enforce minimums and ordering
        fast_scale = max(fast_scale, 1e-5)
        slow_scale = max(slow_scale, fast_scale * 1.8)
        med_scale = max(med_scale, fast_scale * 1.3)

        # Rollover handling
        if self.in_rollover:
            fast_scale *= 1.5
            slow_scale *= 1.5
            med_scale *= 1.5

            rollover_fast_weight = 0.40
            rollover_slow_weight = 0.35
            rollover_safety_weight = 0.25
        else:
            rollover_fast_weight = self.fast_weight
            rollover_slow_weight = self.slow_weight
            rollover_safety_weight = self.safety_weight

        # Volatility scaling
        if self.baseline_variance and self.baseline_variance > 0 and len(self.recent_changes) >= 20:
            vol_ratio = self.rolling_variance / self.baseline_variance

            if vol_ratio > 2.5:
                fast_scale *= 1.2
                slow_scale *= 1.3
                med_scale *= 1.25
            elif vol_ratio < 0.4:
                fast_scale *= 0.85
                slow_scale *= 0.95
                med_scale *= 0.90

        # Build 3-component mixture
        components = [
            {
                "density": {
                    "type": "builtin",
                    "name": "laplace",
                    "params": {
                        "loc": float(x_mean),
                        "scale": float(fast_scale)
                    }
                },
                "weight": float(rollover_fast_weight)
            },
            {
                "density": {
                    "type": "builtin",
                    "name": "laplace",
                    "params": {
                        "loc": float(x_mean),
                        "scale": float(slow_scale)
                    }
                },
                "weight": float(rollover_slow_weight)
            },
            {
                "density": {
                    "type": "builtin",
                    "name": "laplace",
                    "params": {
                        "loc": float(x_mean),
                        "scale": float(med_scale)
                    }
                },
                "weight": float(rollover_safety_weight)
            }
        ]

        return {"type": "mixture", "components": components}
    
tracker = TripleTimescale()
tracker.test_run(live=False)
"""
timestep.py
===========

Adaptive time-stepping controller and reporting-time schedule enforcement
(README §5.5).

Schedule (defaults, README §12):
    0   - 30   days -> step-size cap 1  day
    30  - 180  days -> step-size cap 5  days
    180 - 1000 days -> step-size cap 10-30 days (achieved by adaptive growth
                        up to the 30-day cap; the "10-30" range in the README
                        is the observed steady-state behaviour of the
                        growth/shrink rule below once past the early
                        transient, not a hand-set floor)

Adaptive rule (README §5.5):
    - Newton converged in <= 4 iterations -> dt *= 1.5 (grow)
    - Newton converged in >= 8 iterations -> dt *= 0.5 (shrink)
    - otherwise                            -> dt unchanged
    then clip to [dt_min, min(dt_max, phase_cap(t))], and finally clip so the
    step lands exactly on the next reporting time or the final time.

Time-step-failure handling (Newton did not converge, README §5.4) is a
*separate* mechanism (:func:`cut_dt`), invoked by ``simulator.py`` to retry
the *same* step starting again from x^n with a smaller dt -- it does not
touch the adaptive-growth state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

__all__ = [
    "TimeStepParams",
    "TimeStepper",
    "phase_dt_cap",
    "clip_to_next_stop",
    "grow_or_shrink",
    "cut_dt",
]


@dataclass
class TimeStepParams:
    """Time-stepping controller settings (README §5.5, §12)."""

    dt_init: float = 1.0
    dt_min: float = 0.01
    dt_max: float = 30.0
    grow_factor: float = 1.5
    shrink_factor: float = 0.5
    iter_grow_threshold: int = 4
    iter_shrink_threshold: int = 8
    max_cuts: int = 5
    phase_boundaries: List[float] = field(default_factory=lambda: [30.0, 180.0, 1000.0])
    phase_dt_caps: List[float] = field(default_factory=lambda: [1.0, 5.0, 30.0])


def phase_dt_cap(t: float, params: TimeStepParams) -> float:
    """
    Return the schedule's step-size cap [days] applicable at time ``t``
    (README §5.5). ``phase_boundaries`` and ``phase_dt_caps`` must have the
    same length; the cap for ``t < phase_boundaries[k]`` is
    ``phase_dt_caps[k]``, and any ``t`` beyond the last boundary uses the
    last cap.
    """
    for boundary, cap in zip(params.phase_boundaries, params.phase_dt_caps):
        if t < boundary - 1.0e-9:
            return cap
    return params.phase_dt_caps[-1]


def clip_to_next_stop(t: float, dt: float, t_end: float, report_times: Sequence[float]) -> float:
    """
    Shrink ``dt`` (never grow it) so that ``t + dt`` lands exactly on the
    next reporting time or on ``t_end``, whichever comes first (README
    §5.5: "always land exactly on reporting times and the final time").
    """
    stops = sorted(set(list(report_times) + [t_end]))
    upcoming = [s for s in stops if s > t + 1.0e-9]
    if not upcoming:
        remaining = t_end - t
        return max(min(dt, remaining), 0.0)
    next_stop = upcoming[0]
    if t + dt > next_stop - 1.0e-9:
        dt = next_stop - t
    return dt


def grow_or_shrink(dt_prev: float, n_iterations: int, params: TimeStepParams) -> float:
    """Apply the README §5.5 iteration-count-based growth/shrink rule."""
    if n_iterations <= params.iter_grow_threshold:
        return dt_prev * params.grow_factor
    elif n_iterations >= params.iter_shrink_threshold:
        return dt_prev * params.shrink_factor
    return dt_prev


def cut_dt(dt: float, params: TimeStepParams) -> float:
    """
    Halve the time step after a failed Newton solve (README §5.4), floored
    at ``dt_min``. The caller is responsible for enforcing the max-retries
    limit (``params.max_cuts``) and for restarting the *same* step from
    x^n with the returned, smaller dt.
    """
    return max(dt * 0.5, params.dt_min)


class TimeStepper:
    """
    Thin stateful convenience wrapper around the pure functions above, used
    by ``simulator.py``'s main time loop.
    """

    def __init__(self, params: TimeStepParams, t_end: float, report_times: Sequence[float]):
        self.params = params
        self.t_end = float(t_end)
        self.report_times = list(report_times)

    def initial_dt(self) -> float:
        """Step size to attempt for the very first time step (t = 0)."""
        dt = self.params.dt_init
        dt = min(dt, phase_dt_cap(0.0, self.params))
        dt = max(dt, self.params.dt_min)
        dt = min(dt, self.params.dt_max)
        dt = clip_to_next_stop(0.0, dt, self.t_end, self.report_times)
        return dt

    def propose_next(self, t_new: float, n_iterations: int, dt_used: float) -> float:
        """
        Step size to attempt for the step starting at ``t_new``, given that
        the just-completed step (of size ``dt_used``) converged in
        ``n_iterations`` Newton iterations.
        """
        dt = grow_or_shrink(dt_used, n_iterations, self.params)
        cap = phase_dt_cap(t_new, self.params)
        dt = min(dt, cap)
        dt = max(dt, self.params.dt_min)
        dt = min(dt, self.params.dt_max)
        dt = clip_to_next_stop(t_new, dt, self.t_end, self.report_times)
        return dt
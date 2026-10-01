"""Live 3D visualization for an in-progress reservoir simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from residual import unpack_state


class LiveSimulationViewer:
    """Interactive 3D saturation view updated after each accepted time step."""

    def __init__(self, model: Any, pause: float = 0.01) -> None:
        try:
            import matplotlib

            backend = matplotlib.get_backend().lower()
            if "agg" in backend or "inline" in backend:
                last_error = None
                for candidate in ("TkAgg", "QtAgg", "WXAgg"):
                    try:
                        matplotlib.use(candidate, force=True)
                        backend = candidate.lower()
                        break
                    except Exception as exc:
                        last_error = exc
                else:
                    raise RuntimeError(
                        "No interactive Matplotlib backend is available. "
                        "Install tkinter/Qt support or run without --live."
                    ) from last_error
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(
                "Live visualization requires a compatible matplotlib installation. "
                "Use the configured btp environment and an interactive GUI backend."
            ) from exc
        if not plt.isinteractive():
            plt.ion()
        self.plt = plt
        self.pause = pause
        self.figure = plt.figure(figsize=(10, 7))
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.axes.set_xlabel("x [ft]")
        self.axes.set_ylabel("y [ft]")
        self.axes.set_zlabel("z [ft]")
        self.axes.set_xlim(0.0, model.grid.nx * model.grid.dx)
        self.axes.set_ylim(0.0, model.grid.ny * model.grid.dy)
        self.axes.set_zlim(0.0, model.grid.nz * model.grid.dz)
        self.scatter = None
        plt.show(block=False)

    def update(self, time_days: float, state: np.ndarray, model: Any, diagnostics: dict | None = None) -> None:
        _, sw, _ = unpack_state(state)
        if self.scatter is None:
            self.scatter = self.axes.scatter(
                model.grid.x_center, model.grid.y_center, model.grid.z_center,
                c=sw, cmap="Blues", vmin=0.0, vmax=1.0, s=18, alpha=0.9,
            )
            self.figure.colorbar(self.scatter, ax=self.axes, label="Water saturation")
        else:
            self.scatter.set_array(sw)
        iterations = diagnostics.get("n_newton_iter", "-") if diagnostics else "-"
        self.axes.set_title(f"3D waterflood: t = {time_days:.1f} days | Newton iterations = {iterations}")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self.plt.pause(self.pause)

    def close(self) -> None:
        self.plt.ioff()
        self.plt.show()

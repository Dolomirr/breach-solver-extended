from dataclasses import dataclass

from .base_config import BaseSolverConfig


@dataclass
class ScipConfig(BaseSolverConfig):
    """
    Configs for ``ScipSolver``.

    :param verbose_output: Control console output (internal logs from PySCIPopt) behavior. Used for debugging and optimization process monitoring.
        Default: ``False``.
    :param absgap: Minimal gat for linear solver, if >= 0.0 allow non-optimal solution. Must be non-negative.
        Default: ``0.0``
    :param time_limit: Limitation on the time (in seconds) for finding solution.
        Default: ``10e+20``
    """

    verbose_output: bool | None = None
    """
    Control console output (internal logs from PySCIPopt) behavior. Used for debugging and optimization process monitoring.
        Default: ``False``.
    """

    absgap: float | None = None
    """
    Minimal gat for linear solver, if >= 0.0 allow non-optimal solution. Must be non-negative.
        Default: ``0.0`` meaning solver with exhaustively look for best possible solution.
    """

    time_limit: float | None = None
    """
    Limitation on the time (in seconds) for finding solution.
        Default: ``10e+20`` meaning solver with exhaustively look for best possible solution.
    """

    def __post_init__(self):
        msg = []

        if self.verbose_output is None:
            self.verbose_output = False
        else:
            try:
                self.verbose_output = bool(self.verbose_output)
            except (TypeError, ValueError):
                msg.append(
                    "output_flag cannot be converted to bool, "
                    f"given {type(self.verbose_output)}: {self.verbose_output!r}",
                )

        if self.absgap is None:
            self.absgap = 0.0
        else:
            try:
                self.absgap = float(self.absgap)
            except (TypeError, ValueError):
                msg.append(
                    f"absgap cannot be converted to float, given {type(self.absgap)}: {self.absgap!r}",
                )
            if self.absgap < 0.0:
                msg.append(f"absgap is invalid, must be non-negative, given: {self.absgap!r}")

        if self.time_limit is None:
            self.time_limit = 1e20
        else:
            try:
                self.time_limit = float(self.time_limit)
            except (TypeError, ValueError):
                msg.append(
                    f"time_limit cannot be converted to float, given {type(self.time_limit)}: {self.time_limit!r}",
                )
            if self.time_limit <= 0.0:
                msg.append(
                    f"time_limit is invalid, must positive, given: {self.time_limit!r}",
                )

        if msg:
            msg = "\n" + "\n".join(msg)
            raise ValueError(msg)

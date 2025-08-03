from time import perf_counter

import numpy as np
from pyscipopt import Model

from core import NoSolution, Solution, SolverResult, Task

from ...solver_abc import OptimizationError, Solver, register_solver
from ...solver_configs import ScipConfig, SolverCode
from .context import TaskContext
from .runner import ModelRunner


@register_solver(SolverCode.SCIP)
class ScipSolver(Solver[ScipConfig]):
    def solve(self, task: Task, config: ScipConfig) -> tuple[SolverResult, float]:
        self.context = TaskContext(task, config)
        self.runner = ModelRunner(self.context)

        self.context.model.hideOutput(not self.context.config.verbose_output)
        
        self.runner.build()
        self.runner.optimize()

        try:
            self.runner.optimize()
        except OptimizationError as e:
            msg = f"SCIP optimization failed:\n    {e}"
            raise OptimizationError(msg) from e

        x_path = self.context.path
        buffer_nums = self.context.buffer_nums
        y_active = self.context.active_daemons
        y_total_points = self.context.total_points
        
        
        if x_path.shape[0] == 0:
            return NoSolution(reason="No valid solution possible for given task"), 0.0

        return Solution(x_path, buffer_nums, y_active, y_total_points), 1.0

        # return NoSolution("Aboba"), 0.

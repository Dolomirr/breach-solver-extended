import logging
from time import perf_counter

from core import NoSolution, Solution, SolverResult, Task, setup_logging

from ...solver_abc import OptimizationError, Solver, register_solver
from ...solvers_configs import ScipConfig, SolverCode
from .context import TaskContext
from .extractor import ResultExtractor
from .runner import ModelRunner

setup_logging()
log = logging.getLogger(__name__)


@register_solver(SolverCode.SCIP)
class ScipSolver(Solver[ScipConfig]):
    """SCIP solver"""

    def solve(self, task: Task, config: ScipConfig| None = None) -> tuple[SolverResult, float]:
        """
        Linear programming solver.

        Exact approach using SCIP solver via `PySCIPOpt` API for constraint-based modeling.
        Find optimal or near-optimal solutions depending on configuration().
        Test seems to indicate that with default values of ``config.absgap`` and ``config.time_limit``, always able to find optimal solution.

        .. seealso ::
            ``ScipConfig``

        :param task: ``Task`` instance.
        :param config: ``ScipConfig`` instance.
        :return: ``tuple: (Solution, execution_time``),
            or if no solution found ``tuple: (NoSolution, -1.0)``.
        """
        if not config:
            config = ScipConfig()
        
        self.context = TaskContext(task, config)
        self.runner = ModelRunner(self.context)
        self.extractor = ResultExtractor(self.context)

        model = self.context.model
        model.hideOutput(not config.verbose_output)
        model.setRealParam("limits/absgap", config.absgap)
        model.setRealParam("limits/time", config.time_limit)

        log.info("Scip Started solving.", extra={'config': config})
        
        start_time = perf_counter()
        try:
            self.runner.build()
        except OptimizationError as e:
            msg = f"SCIP model build failed:\n    {e}"
            log.exception(msg)
            raise OptimizationError(msg) from e


        post_build = perf_counter()
        try:
            self.runner.optimize()
        except OptimizationError as e:
            msg = f"SCIP optimization failed:\n    {e}"
            log.exception(msg)
            raise OptimizationError(msg) from e

        end_time = perf_counter()
        
        build_time = post_build - start_time
        opt_time = end_time - post_build
        if self.context.config.verbose_output:
            print(
                f"SCIP model build time: {build_time:.6f}, optimization time: {opt_time:.6f}",
                flush=True,
            )
        log.info("Scip Finished solving.", extra={'build_time': build_time, 'opt_time': opt_time})

        x_path = self.extractor.path
        buffer_nums = self.extractor.buffer_nums
        y_active = self.extractor.active_daemons
        y_total_points = self.extractor.total_points

        if x_path.shape[0] == 0:
            return NoSolution(reason="Found path is empty"), -1.0

        return Solution(
            x_path,
            buffer_nums,
            y_active,
            y_total_points,
        ), end_time - start_time

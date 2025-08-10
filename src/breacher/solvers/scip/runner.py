from functools import cached_property
from itertools import product
from typing import Self

from pyscipopt import quicksum

from core import HexSymbol

from ...solver_abc import OptimizationError
from .context import TaskContext


class ModelRunner:
    """
    Responsible for building model and running optimization.

    :param context: ``TaskContext`` instance.
    """

    def __init__(self, context: TaskContext) -> None:
        self.context = context
        self._is_builded: bool = False

    def build(self) -> None:
        """
        Builds the SCIP model based on provided ``TaskContext``.
        """
        (
            self._set_step_matrix()  # noqa: SLF001 (false positive on private members)
            ._set_movement_constraints()
            ._set_sequences()
            ._set_daemons_activation()
            ._set_objective()
        )

        self._is_builded = True

    @cached_property
    def _x_matrix_shape(self) -> tuple[int, int, int]:
        """
        3 common used throughout the methods for ranges values, representing the dimensions of x (n, m, buffer_size) matrix.
            - n: amount of rows
            - m: length of each row
            - buffer_size: (or 'step') - amount of available steps and buffer size
        """
        return (
            self.context.n,
            self.context.m,
            int(self.context.buffer_size),
        )

    def _set_step_matrix(self) -> Self:
        ctx = self.context
        x, model = ctx.x, ctx.model
        n, m, step = self._x_matrix_shape

        for i, j, t in product(range(n), range(m), range(step)):
            x[i, j, t] = model.addVar(vtype="B", name=f"x_{i}_{j}_{t}")

        return self

    def _set_movement_constraints(self) -> Self:
        ctx = self.context
        x, model = ctx.x, ctx.model
        n, m, step = self._x_matrix_shape

        # one cell per step
        for t in range(step):
            model.addCons(
                quicksum(x[i, j, t] for i in range(n) for j in range(m)) <= 1,
                name=f"one_cell_per_step_{t}",
            )

        # continuous path
        for t in range(1, step):
            prev_sum = quicksum(x[i, j, t - 1] for i in range(n) for j in range(m))
            curr_sum = quicksum(x[i, j, t] for i in range(n) for j in range(m))
            model.addCons(curr_sum <= prev_sum, name=f"continuous_path_{t}")

        # cell used at max one time
        for i, j in product(range(n), range(m)):
            model.addCons(
                quicksum(x[i, j, t] for t in range(step)) <= 1,
                name=f"cell_once_{i}_{j}",
            )

        # Max steps (probably redundant)
        ctx.used_buffer = quicksum(x[i, j, t] for i in range(n) for j in range(m) for t in range(step))
        model.addCons(ctx.used_buffer <= step, name="max_steps")

        # start in first row
        for i in range(1, n):
            model.addCons(
                quicksum(x[i, j, 0] for j in range(m)) == 0,
                name=f"start_in_first_row_{i}",
            )

        # row/col alteration
        for t in range(1, step):
            for i, j in product(range(n), range(m)):
                if t % 2 == 1:  # column
                    prev_col = quicksum(x[k, j, t - 1] for k in range(n))
                    model.addCons(
                        x[i, j, t] <= prev_col,
                        name=f"move_rule_col_{i}_{j}_step_{t}",
                    )
                else:  # row
                    prev_row = quicksum(x[i, k, t - 1] for k in range(m))
                    model.addCons(
                        x[i, j, t] <= prev_row,
                        name=f"move_rule_row_{i}_{j}_step_{t}",
                    )

        return self

    def _set_sequences(self) -> Self:
        ctx = self.context
        x, model = ctx.x, ctx.model
        z = ctx.z
        n, m, step = self._x_matrix_shape

        ctx.buffer_seq = [
            quicksum(
                ctx.matrix[i][j] * x[i, j, t]
                for i, j in product(range(n), range(m))
            ) for t in range(step)
        ]   # fmt: skip

        # demon sequences starts from buffer
        for i in range(ctx.d_count):
            curr_len = ctx.d_lengths[i]
            valid_p = step - curr_len + 1
            z_i = {}
            for p in range(valid_p):
                z_i[p] = model.addVar(vtype="B", name=f"z_{i}_{p}")
            z.append(z_i)

        big_m = int(max(HexSymbol)) + 1     # larger then any symbol code
        for i in range(ctx.d_count):
            curr_len = ctx.d_lengths[i]
            valid_p = step - curr_len + 1
            for p, s in product(range(valid_p), range(curr_len)):
                t = p + s
                # z[i][p] == 1 --> buffer_seq[t] == demons[i][s]
                model.addCons(
                    ctx.buffer_seq[t] >= ctx.daemons[i][s] - big_m * (1 - z[i][p]),
                    name=f"indicator_lb_{i}_{p}_{s}",
                )
                model.addCons(
                    ctx.buffer_seq[t] <= ctx.daemons[i][s] + big_m * (1 - z[i][p]),
                    name=f"indicator_ub_{i}_{p}_{s}",
                )

        return self

    def _set_daemons_activation(self) -> Self:
        ctx = self.context
        model = ctx.model
        y, z = self.context.y, self.context.z
        _, _, step = self._x_matrix_shape

        for i in range(ctx.d_count):
            y.append(model.addVar(vtype="B", name=f"y_{i}"))

        for i in range(ctx.d_count):
            valid_p = step - ctx.d_lengths[i] + 1
            if valid_p <= 0:
                model.addCons(y[i] == 0, name=f"y_false_{i}")
                continue
            model.addCons(
                y[i] <= quicksum(z[i][p] for p in range(valid_p)),
                name=f"y_upper_{i}",
            )
            for p in range(valid_p):
                model.addCons(y[i] >= z[i][p], name=f"y_lower_{i}_{p}")

        return self

    def _set_objective(self) -> Self:
        ctx = self.context
        y, model = ctx.y, ctx.model
        _, _, step = self._x_matrix_shape

        objective = quicksum(ctx.daemons_costs[i] * y[i] for i in range(ctx.d_count)) + ctx.unused_cell_reward * (
            step - ctx.used_buffer
        )

        model.setObjective(objective, sense="maximize")

        return self

    def optimize(self) -> None:
        """
        Run optimization, call strictly after ``ModelRunner.build()``.
        """
        if not self._is_builded:
            msg = "ModelRunner.build() must be called before ModelRunner.optimize()"
            raise RuntimeError(msg)

        try:
            self.context.model.optimize()
        except Exception as e:
            msg = f"SCIP optimization failed:\n    {e}"
            raise OptimizationError(msg) from e
        else:
            self.context.is_finished = True

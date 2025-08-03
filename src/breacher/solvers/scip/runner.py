from pyscipopt import quicksum

from core import HexSymbol

from ...solver_abc import OptimizationError
from .context import TaskContext


class ModelRunner:
    def __init__(self, context: TaskContext):
        self.context = context

    def build(self) -> None:
        steps = (
            self._set_step_matrix,
            self._set_movement_constraints,
            self._set_sequences,
            self._set_daemons_activation,
            self._set_objective,
        )
        for step in steps:
            step()
        

    def _set_step_matrix(self):
        for i in range(self.context.n):
            for j in range(self.context.n):
                for t in range(self.context.buffer_size):
                    self.context.x[i, j, t] = self.context.model.addVar(vtype="B", name=f"x_{i}_{j}_{t}")

    def _set_movement_constraints(self):
        # one cell per step
        for t in range(self.context.buffer_size):
            self.context.model.addCons(
                quicksum(self.context.x[i, j, t] for i in range(self.context.n) for j in range(self.context.n)) <= 1,
                name=f"one_cell_per_step_{t}",
            )

        # continuous path
        for t in range(1, self.context.buffer_size):
            prev_sum = quicksum(self.context.x[i, j, t - 1] for i in range(self.context.n) for j in range(self.context.n))
            curr_sum = quicksum(self.context.x[i, j, t] for i in range(self.context.n) for j in range(self.context.n))
            self.context.model.addCons(curr_sum <= prev_sum, name=f"continuous_path_{t}")

        # cell used at max one time
        for i in range(self.context.n):
            for j in range(self.context.n):
                self.context.model.addCons(
                    quicksum(self.context.x[i, j, t] for t in range(self.context.buffer_size)) <= 1,
                    name=f"cell_once_{i}_{j}",
                )

        # Max steps (probably redundant)
        self.context.used_buffer = quicksum(
            self.context.x[i, j, t]
            for i in range(self.context.n)
            for j in range(self.context.n)
            for t in range(self.context.buffer_size)
        )
        self.context.model.addCons(self.context.used_buffer <= self.context.buffer_size, name="max_steps")

        # start in first row
        for i in range(1, self.context.n):
            self.context.model.addCons(
                quicksum(self.context.x[i, j, 0] for j in range(self.context.n)) == 0,
                name=f"start_in_first_row_{i}",
            )

        # row/col alteration
        for t in range(1, self.context.buffer_size):
            for i in range(self.context.n):
                for j in range(self.context.n):
                    if t % 2 == 1:  # column
                        prev_col = quicksum(self.context.x[k, j, t - 1] for k in range(self.context.n))
                        self.context.model.addCons(
                            self.context.x[i, j, t] <= prev_col,
                            name=f"move_rule_col_{i}_{j}_step_{t}",
                        )
                    else:  # row
                        prev_row = quicksum(self.context.x[i, k, t - 1] for k in range(self.context.n))
                        self.context.model.addCons(
                            self.context.x[i, j, t] <= prev_row,
                            name=f"move_rule_row_{i}_{j}_step_{t}",
                        )

    def _set_sequences(self):
        self.context.buffer_seq = [
            quicksum(
                self.context.matrix[i][j] * self.context.x[i, j, t]
                for i in range(self.context.n)
                for j in range(self.context.n)
            )
            for t in range(self.context.buffer_size)
        ]

        # demon sequences starts from buffer
        for i in range(self.context.d_count):
            curr_len = self.context.d_lengths[i]
            valid_p = self.context.buffer_size - curr_len + 1
            z_i = {}
            for p in range(valid_p):
                z_i[p] = self.context.model.addVar(vtype="B", name=f"z_{i}_{p}")
            self.context.z.append(z_i)

        big_m = int(max(HexSymbol)) + 1
        for i in range(self.context.d_count):
            curr_len = self.context.d_lengths[i]
            valid_p = self.context.buffer_size - curr_len + 1
            for p in range(valid_p):
                for s in range(curr_len):
                    t = p + s
                    # z[i][p] == 1 --> buffer_seq[t] == demons[i][s]
                    self.context.model.addCons(
                        self.context.buffer_seq[t] >= self.context.daemons[i][s] - big_m * (1 - self.context.z[i][p]),
                        name=f"indicator_lb_{i}_{p}_{s}",
                    )
                    self.context.model.addCons(
                        self.context.buffer_seq[t] <= self.context.daemons[i][s] + big_m * (1 - self.context.z[i][p]),
                        name=f"indicator_ub_{i}_{p}_{s}",
                    )

    def _set_daemons_activation(self):
        self.context.y = [self.context.model.addVar(vtype="B", name=f"y_{i}") for i in range(self.context.d_count)]
        for i in range(self.context.d_count):
            valid_p = self.context.buffer_size - self.context.d_lengths[i] + 1
            if valid_p <= 0:
                self.context.model.addCons(self.context.y[i] == 0, name=f"y_false_{i}")
                continue
            self.context.model.addCons(
                self.context.y[i] <= quicksum(self.context.z[i][p] for p in range(valid_p)),
                name=f"y_upper_{i}",
            )
            for p in range(valid_p):
                self.context.model.addCons(self.context.y[i] >= self.context.z[i][p], name=f"y_lower_{i}_{p}")

    def _set_objective(self):
        objective = quicksum(
            self.context.daemons_costs[i] * self.context.y[i] for i in range(self.context.d_count)
        ) + self.context.unused_cell_reward * (self.context.buffer_size - self.context.used_buffer)

        self.context.model.setObjective(objective, sense="maximize")

    def optimize(self):
        try:
            self.context.model.optimize()
        except Exception as e:
            msg = f"SCIP optimization failed:\n    {e}"
            raise OptimizationError(msg) from e
        else:
            self.context.is_finished = True

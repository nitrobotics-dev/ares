from ares.configs.base import Environment, Robot, Rollout, RolloutSQLModel, Task

ROBOT1 = Robot(name="robot1", type="robot", sensor="camera")
ENV1 = Environment(
    name="environment1", type="table", lighting="high light", simulation=True
)
TASK1 = Task(name="task1", description="task1", success_criteria="task1", success=0.9)
ROLL1 = Rollout(robot=ROBOT1, environment=ENV1, task=TASK1)

ROBOT2 = Robot(name="robot2", type="robot", sensor="camera")
ENV2 = Environment(
    name="environment2", type="gym", lighting="low light", simulation=False
)
TASK2 = Task(name="task2", description="task2", success_criteria="task2", success=0.0)
ROLL2 = Rollout(robot=ROBOT2, environment=ENV2, task=TASK2)

SQL_ROLL1 = RolloutSQLModel(**ROLL1.flatten_fields(""))
SQL_ROLL2 = RolloutSQLModel(**ROLL2.flatten_fields(""))

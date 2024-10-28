from configs import Robot, Environment, Task, Trajectory, TrajectorySQLModel
from sqlmodel import SQLModel, Field, create_engine, Session
from sqlalchemy import Engine, String, Float, Boolean, DateTime, Integer
from sqlmodel import SQLModel, Field, Column
from sqlalchemy.orm import declarative_base, Session

BASE_ROBOT_DB_PATH = "sqlite:///robot_data.db"
TEST_ROBOT_DB_PATH = "sqlite:///test_robot_data.db"


def setup_database(path: str = BASE_ROBOT_DB_PATH) -> Engine:
    engine = create_engine(path)
    SQLModel.metadata.create_all(engine)
    return engine


def add_trajectory(engine: Engine, trajectory: Trajectory):
    trajectory_sql_model = TrajectorySQLModel(**trajectory.flatten_fields(""))
    with Session(engine) as session:
        session.add(trajectory_sql_model)
        session.commit()


if __name__ == "__main__":
    from test_configs import TRAJ1, TRAJ2, SQL_TRAJ1, SQL_TRAJ2

    engine = setup_database(path=TEST_ROBOT_DB_PATH)
    add_trajectory(engine, TRAJ1)
    add_trajectory(engine, TRAJ2)

    sess = Session(engine)
    res = sess.query(TrajectorySQLModel).filter(TrajectorySQLModel.task_success > 0.5)
    breakpoint()

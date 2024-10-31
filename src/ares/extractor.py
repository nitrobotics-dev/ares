import os
from ares.configs.base import Robot, Environment, Task, Trajectory
from ares.configs.open_x_embodiment_configs import OpenXEmbodimentEpisode
import numpy as np
import string
import typing as t


class InformationExtractor:
    def __init__(self):
        pass

    def extract(self, episode: OpenXEmbodimentEpisode) -> Trajectory:
        raise NotImplementedError


class RandomInformationExtractor(InformationExtractor):
    def random_string(self, length_bound: t.Optional[int] = 10) -> str:
        return "".join(
            np.random.choice(
                string.ascii_lowercase.split(),
                size=np.random.randint(1, length_bound + 1),
            )
        )

    def extract(self, episode: OpenXEmbodimentEpisode) -> Trajectory:
        robot = Robot(
            name=self.random_string(),
            sensor=np.random.choice(["camera", "wrist", "gripper"]),
        )
        environment = Environment(
            name=self.random_string(),
            lighting=self.random_string(),
            simulation=np.random.choice([True, False]),
        )
        task = Task(
            name=self.random_string(),
            description=self.random_string(),
            success_criteria=self.random_string(),
            success=np.random.uniform(0, 1),
            language_instruction=episode.steps[0].language_instruction,
        )

        return Trajectory(
            robot=robot, environment=environment, task=task, length=len(episode.steps)
        )

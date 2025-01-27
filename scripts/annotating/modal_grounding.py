"""
Modal wrapper for interacting with GroundingWorker.
"""

from typing import List, Tuple

from modal import enter, method

from ares.models.grounding import GroundingAnnotator
from scripts.annotating.modal_base import BaseModalWrapper, BaseWorker


class GroundingWorker(BaseWorker):
    """Worker class for grounding annotations."""

    @enter()
    def setup(self) -> None:
        pass

    @method()
    async def process(
        self, rollout_id: str, frames: list, label_str: str
    ) -> Tuple[str, list]:
        """
        Process method to annotate a single video.

        Args:
            rollout_id (str): Identifier for the rollout.
            frames (list): List of video frames.
            label_str (str): Label string for annotation.

        Returns:
            Tuple[str, list]: Annotation result.
        """
        worker = GroundingAnnotator(segmenter_id=None)
        result = await worker.annotate_video(rollout_id, frames, label_str)
        return rollout_id, result


class GroundingModalWrapper(BaseModalWrapper):
    """
    Wrapper class to interact with GroundingWorker via Modal.
    """

    def __init__(self, app_name: str = "grounding_app"):
        super().__init__(app_name, worker_cls=GroundingWorker)

    async def annotate_videos(
        self, tasks: List[Tuple[str, list, str]]
    ) -> List[Tuple[str, list]]:
        """
        Submit a batch of annotation tasks to the GroundingWorker.

        Args:
            tasks (List[Tuple[str, list, str]]): List of tuples containing rollout_id, frames, and label_str.

        Returns:
            List[Tuple[str, list]]: List of annotation results.
        """
        return await self.run_batch(tasks)

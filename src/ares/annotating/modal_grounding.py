"""
Modal wrapper for interacting with GroundingWorker.
"""

from modal import enter, method

from ares.annotating.modal_base import BaseModalWrapper, BaseWorker
from ares.models.grounding import GroundingAnnotator


class GroundingWorker(BaseWorker):
    """Worker class for grounding annotations."""

    @enter()
    def setup(self) -> None:
        """Initialize resources needed for grounding."""
        self.worker = GroundingAnnotator(segmenter_id=None)

    @method()
    async def process(
        self, batch: list[tuple[str, list, str]]
    ) -> list[tuple[str, list]]:
        """
        Process method to annotate multiple videos in a batch.

        Args:
            batch: List of tuples containing (rollout_id, frames, label_str)

        Returns:
            list[tuple[str, list]]: List of annotation results
        """
        results = []
        for rollout_id, frames, label_str in batch:
            try:
                result = self.worker.annotate_video(rollout_id, frames, label_str)
                results.append(result)
            except Exception as e:
                print(f"Error processing {rollout_id}: {e}")
                # Return empty annotations for failed items
                results.append((rollout_id, []))
        return results


class GroundingModalWrapper(BaseModalWrapper):
    """
    Wrapper class to interact with GroundingWorker via Modal.
    """

    def __init__(self, app_name: str = "grounding_app"):
        super().__init__(app_name, worker_cls=GroundingWorker)

    async def annotate_videos(
        self, tasks: list[tuple[str, list, str]]
    ) -> list[tuple[str, list]]:
        """
        Submit a batch of annotation tasks to the GroundingWorker.

        Args:
            tasks (list[tuple[str, list, str]]): List of tuples containing rollout_id, frames, and label_str.

        Returns:
            list[tuple[str, list]]: List of annotation results.
        """
        return await self.run_batch(tasks)

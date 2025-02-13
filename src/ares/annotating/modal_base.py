"""
Base Modal infrastructure for running serverless compute tasks. In order to prevent copying over the ARES repository and dependencies,
avoid importing any ARES modules in this file; instead, import the necessary modules in the specific Modal app classes.
"""

import asyncio
import typing as t

from modal import App, Image, enter, method

# Base Modal image with common dependencies
base_image = (
    Image.debian_slim()
    .apt_install("python3-opencv")
    .pip_install("torch", "transformers", "numpy", "opencv-python", "tqdm", "pillow")
)


class BaseWorker:
    """
    Base worker class to be decorated by specific Modal apps.
    """

    @enter()
    def setup(self) -> None:
        """Override in subclass to initialize resources."""
        pass

    @method()
    async def process(self, *args, **kwargs):
        """Override in subclass with task-specific logic."""
        pass


class BaseModalWrapper:
    """
    Base class for Modal task wrappers.
    """

    def __init__(
        self,
        app_name: str,
        worker_cls: t.Type[BaseWorker] = BaseWorker,
        image: Image = base_image,
    ) -> None:
        self.app_name = app_name
        self.app = App(app_name)
        self.WorkerCls = self.app.cls(
            image=image,
            gpu="t4",
            concurrency_limit=10,
            timeout=600,
        )(worker_cls)
        print(f"Modal app {self.app_name} initialized")

    async def run_batch(
        self, items: list[t.Any], batch_size: int = 8
    ) -> list[tuple[str, list[list["Annotation"]]]]:
        """Run batch processing using Modal."""
        tasks = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            tasks.append(self.WorkerCls().process.remote.aio(batch))
        results = await asyncio.gather(*tasks)
        results = [item for batch in results for item in batch]
        return results

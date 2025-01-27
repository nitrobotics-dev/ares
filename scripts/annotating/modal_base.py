"""Base Modal infrastructure for running serverless compute tasks."""

import asyncio

from modal import App, Image, build, enter, method

# Base Modal image with common dependencies
base_image = (
    Image.debian_slim()
    .apt_install("python3-opencv")
    .pip_install("torch", "transformers", "numpy", "opencv-python", "tqdm", "pillow")
)


class BaseWorker:
    """Base worker class to be decorated by specific Modal apps."""

    @enter()
    def setup(self) -> None:
        """Override in subclass to initialize resources."""
        pass

    @method()
    async def process(self, *args, **kwargs):
        """Override in subclass with task-specific logic. Now properly marked as async."""
        pass


class BaseModalWrapper:
    """Base class for Modal task wrappers."""

    def __init__(self, app_name: str, worker_cls=BaseWorker, image=base_image):
        self.app_name = app_name
        self.app = App(app_name)
        self.WorkerCls = self.app.cls(
            image=image,
            gpu="t4",
            concurrency_limit=10,
            timeout=600,
        )(worker_cls)
        print(f"Modal app {self.app_name} initialized")

    async def run_batch(self, items, batch_size=8):
        """Run batch processing using Modal."""
        tasks = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            tasks.append(self.WorkerCls().process.remote.aio(batch))
        results = await asyncio.gather(*tasks)
        # Flatten results from all batches
        return [item for batch in results for item in batch]

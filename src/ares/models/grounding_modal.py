import pickle

import numpy as np
from modal import App, Image, enter, method

from ares.models.grounding import GroundingAnnotator

image = (
    Image.debian_slim()
    .apt_install("python3-opencv")
    .pip_install(
        "torch", "transformers", "numpy", "opencv-python", "tqdm", "numpy", "pillow"
    )
)

app = App("ares-grounding-modal", image=image)


@app.cls(image=image, gpu="any")
class ModalWrapper:
    @enter()
    def setup(self) -> None:
        self.annotator = GroundingAnnotator(segmenter_id=None)

    @method()
    def annotate_video(self, frames, label_str):
        # Convert frames from list of lists back to numpy arrays and ensure uint8 type
        frames = [np.array(f, dtype=np.uint8) for f in frames]
        return self.annotator.annotate_video(frames, label_str)


# test remote modal
@app.local_entrypoint()
def test() -> None:
    # do whatever local test
    import numpy as np

    annotator = ModalWrapper()

    rollout_ids, results = pickle.load(open("label_results.pkl", "rb"))
    for rollout_id, result in zip(rollout_ids[:1], results[:1]):
        frames, frame_indices, label_str = result
        frames = [f.tolist() for f in frames]
        out = annotator.annotate_video.remote(frames, label_str)
        print("got local annotation result", out)
    breakpoint()

import datasets
import os
import logging
from openvla.prismatic.vla.datasets import (
    RLDSDataset,
    EpisodicRLDSDataset,
    RLDSBatchTransform,
)
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from openvla.prismatic.models.backbones.llm.prompting import (
    PurePromptBuilder,
    VicunaV15ChatPromptBuilder,
)
from openvla.prismatic.vla.action_tokenizer import ActionTokenizer


hf_base = "jxu124/OpenX-Embodiment"
ucsd_kitchen_dataset_name = "ucsd_kitchen_dataset_converted_externally_to_rlds"
# cmu_play_fusion_dataset_name = "cmu_play_fusion"
# Load the dataset with the custom session
# ds = datasets.load_dataset(
#     hf_base,
#     ucsd_kitchen_dataset_name,
#     split="train",
#     cache_dir="/workspaces/ares/data",
# )
# breakpoint()


@dataclass
class EpisodicConfig:
    data_root_dir: Path = Path(
        # "datasets/open-x-embodiment"
        f"/workspaces/ares/data"
    )  # Path to Open-X dataset directory
    dataset_name: str = ucsd_kitchen_dataset_name
    image_sizes: tuple[int] = (224, 224)
    vla_path: str = "openvla/openvla-7b"  # Path to OpenVLA model (on HuggingFace Hub)
    shuffle_buffer_size: int = 256  # _000
    image_aug: bool = False
    shuffle: bool = False


cfg = EpisodicConfig()
processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
action_tokenizer = ActionTokenizer(processor.tokenizer)

batch_transform = lambda x: x
vla_dataset = EpisodicRLDSDataset(
    cfg.data_root_dir,
    cfg.dataset_name,
    batch_transform,
    resize_resolution=tuple(cfg.image_sizes),
    shuffle_buffer_size=cfg.shuffle_buffer_size,
    image_aug=cfg.image_aug,
)


for ep in vla_dataset:
    breakpoint()

from tqdm import tqdm

from ares.configs.open_x_embodiment_configs import (
    OpenXEmbodimentEpisode,
    get_dataset_information,
)
from ares.models.extractor import LLMInformationExtractor, RandomInformationExtractor
from ares.models.llm import get_gemini_15_flash

dataset_name = "jaco_play"
data_dir = "/workspaces/ares/data/oxe/"
builder, dataset_dict = build_dataset(dataset_name, data_dir)
extractor = LLMInformationExtractor(get_gemini_15_flash())
ds = dataset_dict["train"]
print(f"working on 'train' out of {list(dataset_dict.keys())}")
dataset_info = get_dataset_information(dataset_name)

print(len(ds))

random_extractor = RandomInformationExtractor()

for i, ep in tqdm(enumerate(ds)):
    episode = OpenXEmbodimentEpisode(**ep)
    rollout = extractor.extract(episode)
    breakpoint()
    # print(rollout)

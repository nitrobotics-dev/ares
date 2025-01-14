# from ares.models.eval import *
# from ares.models.shortcuts import get_claude_3_5_sonnet, get_gpt_o1_mini

# frames = easy_get_frames(
#     dataset_name="pi_demos",
#     task="Laundry fold (shirts)",
#     success_flag="success",
#     fps=10,
# )

# # vlm = get_claude_3_5_sonnet()
# vlm = get_gpt_o1_mini()
dataset_filename = "cmu_play_fusion"
fname = "data/train/episode_212.npy"

from ares.utils.image_utils import load_video_frames, split_video_to_frames

split_video_to_frames()

# frames, frame_indices = load_video_frames(dataset_filename, fname, target_fps=10)


# out = vlm.ask(info=dict(prompt="Describe the image"), images=[frames[0]])
print("hello")
breakpoint()

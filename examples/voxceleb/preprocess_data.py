#%%[markdown]
# # Preprocessing the Video Dataset
# - Only the raget videos are saved
# - The context videos are generated by randomly sampling from them


#%%
%load_ext autoreload
%autoreload 2

from selfmod import NumpyLoader
from piedpiper import *
# jax.config.update("jax_debug_nans", True)

## For reproducibility
seed = 2026

## Dataloader hps
# resolution = (32, 24)
resolution = (128, 128)
proportion_of_shots = 0.1
k_shots = int(np.prod(resolution) * proportion_of_shots)
T, H, W, C = (3**2, resolution[1], resolution[0], 3)

print("==== Main properties of the dataset ====")
print(" Number of context shots:", k_shots)
print(" Number of frames:", T)
print(" Resolution:", resolution)
print("===================================")

shuffle = True

num_workers = 8
video_dataset = 'vox2'
envs_batch_size = num_workers if video_dataset=='vox2' else 1

data_folder="./data/" 



#%%

## List every dir in the data folder

train_dataset = VideoDataset(data_folder, 
                      data_split="train", 
                      num_shots=k_shots, 
                      num_frames=T, 
                      resolution=resolution, 
                      order_pixels=False, 
                      max_envs=-1,
                      dataset=video_dataset,
                      seed=seed)
# print("Total number of environments in the training dataset:", len(train_dataset))
train_dataloader = NumpyLoader(train_dataset, 
                               batch_size=envs_batch_size, 
                               shuffle=shuffle, 
                               num_workers=num_workers,
                               drop_last=False)

# ctx_videos, tgt_videos = next(iter(train_dataloader))
# vt = VisualTester(None)
# vt.visualize_video_frames(tgt_videos[0], resolution)
# vt.visualize_video_frames(ctx_videos[0], resolution)

# print("Loaded context videos shape:", ctx_videos.shape, "Target videos shape:", tgt_videos.shape)


#%%
# Iterate throught he whole dataset and saave the taget sets for faster dataloading later

all_targets = []
for i, (_, tgt_videos) in enumerate(train_dataloader):
    all_targets.append(tgt_videos)

    # print("Loaded target videos shape:", tgt_videos.shape)
    if i >100-1:
        break

all_targets = np.concatenate(all_targets, axis=0)

## Give a name with T, H, W, C, then the number of shots as a proportion of the total number of pixels per frame
np.savez("data/preprocessed/vox2_targets_{}_{}_{}_{}_{}.npz".format(T, H, W, C, k_shots), data=all_targets, T=T, H=H, W=W, C=C, k_shots=k_shots, p_shots=proportion_of_shots)


#%%

## Can we load the data backusing a custom dataset ?
prproc_data = PreprocessedVideoDataset(
                      data_path="data/preprocessed/", 
                      num_shots=k_shots, 
                      num_frames=T, 
                      resolution=resolution, 
                      seed=seed)

train_dataloader_preproc = NumpyLoader(train_dataset, 
                               batch_size=envs_batch_size, 
                               shuffle=shuffle, 
                               num_workers=num_workers,
                               drop_last=False)

ctx_videos, tgt_videos = next(iter(train_dataloader_preproc))
vt = VisualTester(None)
vt.visualize_video_frames(tgt_videos[0], resolution)
vt.visualize_video_frames(ctx_videos[0], resolution)


#%%
all_targets.shape
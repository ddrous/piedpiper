from selfmod import CelebADataset
from ._config import *

from typing import Tuple

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
# from jax_dataloader import Dataset
from PIL import Image
import cv2


class ImageDataset(Dataset):
    def __init__(self, 
                 data_path="./data/", 
                 data_split="train",
                 num_shots=100,
                 resolution=(32, 32),
                 order_pixels=False,
                 max_envs=-1,
                 seed=None):
        """
        A wrapper for both context and target dataloaders.
        """
        self.context_sets = CelebADataset(data_path, 
                                            data_split=data_split,
                                            num_shots=num_shots, 
                                            order_pixels=order_pixels, 
                                            resolution=resolution,
                                            )
        # self.target_sets = self.context_sets
        self.target_sets = CelebADataset(data_path, 
                                        data_split=data_split,
                                        num_shots=np.prod(resolution), 
                                        order_pixels=order_pixels, 
                                        resolution=resolution,
                                        )

        assert self.context_sets.total_envs == self.target_sets.total_envs, "ERROR: The number of environments in the context and target dataloaders must be the same."
        self.total_envs = max_envs if max_envs > 0 else self.context_sets.total_envs

        ## set the seed and randomize the order of the ctx and tgt files
        if seed is not None:
            self.context_sets.files = np.random.RandomState(seed).permutation(self.context_sets.files)
            self.target_sets.files = np.random.RandomState(seed).permutation(self.target_sets.files)


    def __getitem__(self, idx):
        ctx_img = self.context_sets.get_image(self.context_sets.files[idx])
        ctx_normed_coords, ctx_pixel_values = self.context_sets.sample_pixels(ctx_img)

        tgt_img = self.target_sets.get_image(self.target_sets.files[idx])
        tgt_normed_coords, tgt_pixel_values = self.target_sets.sample_pixels(tgt_img)

        return np.concatenate((ctx_normed_coords, ctx_pixel_values), axis=-1), np.concatenate((tgt_normed_coords, tgt_pixel_values), axis=-1)


    def __len__(self):
        return self.total_envs






class VideoDataset(Dataset):
    def __init__(self, 
                 data_path="./data/", 
                 data_split="train",
                 num_shots=100,
                 num_frames=10,
                 resolution=(32, 32),
                 order_pixels=False,
                 max_envs=-1,
                 seed=None):
        """
        A wrapper for both context and target dataloaders.
        """
        self.context_sets = Vox2Dataset(data_path, 
                                            data_split=data_split,
                                            num_shots=num_shots, 
                                            order_pixels=order_pixels, 
                                            num_frames=num_frames,
                                            resolution=resolution,
                                            )
        self.target_sets = Vox2Dataset(data_path, 
                                        data_split=data_split,
                                        num_shots=np.prod(resolution), 
                                        num_frames=num_frames,
                                        order_pixels=order_pixels, 
                                        resolution=resolution,
                                        )

        assert self.context_sets.total_envs == self.target_sets.total_envs, "ERROR: The number of environments in the context and target dataloaders must be the same."
        self.total_envs = max_envs if max_envs > 0 else self.context_sets.total_envs

        ## set the seed and randomize the order of the ctx and tgt files
        if seed is not None:
            self.context_sets.files = np.random.RandomState(seed).permutation(self.context_sets.files)
            self.target_sets.files = np.random.RandomState(seed).permutation(self.target_sets.files)

    def __getitem__(self, idx):
        ctx_vid = self.context_sets.get_video(self.context_sets.files[idx])
        ctx_norm_pixs = self.context_sets.sample_video_pixels(ctx_vid)

        tgt_vid = self.target_sets.get_video(self.target_sets.files[idx])
        tgt_norm_pixs = self.target_sets.sample_video_pixels(tgt_vid)

        return ctx_norm_pixs, tgt_norm_pixs

    def __len__(self):
        return self.total_envs








class Vox2Dataset(Dataset):
    """
    A celeb a dataloader for meta-learning.
    """
    def __init__(self, 
                 data_path="./data/", 
                 data_split="train",
                 num_shots=100,
                 num_frames=10,
                 resolution=(32, 32),
                 order_pixels=False,
                 seed=None):

        if num_shots <= 0:
            raise ValueError("Number of shots must be greater than 0.")
        elif num_shots > resolution[0]*resolution[1]:
            raise ValueError("Number of shots must be less than the total number of pixels per frame.")
        self.nb_shots = num_shots
        self.nb_frames = num_frames

        self.input_dim = 2
        self.output_dim = 3
        self.img_size = (*resolution, self.output_dim)
        # self.img_size = (resolution[1], resolution[0], self.output_dim)
        self.order_pixels = order_pixels

        self.data_path = data_path
        partitions = pd.read_csv(self.data_path+'list_eval_partition.txt', 
                                 header=None, 
                                 sep=r'\s+', 
                                 names=['filename', 'partition'])
        if data_split in ["train"]:
            self.folders = partitions[partitions['partition'] == 0]['filename'].values
        elif data_split in ["val"]:
            self.folders = partitions[partitions['partition'] == 1]['filename'].values
        elif data_split in ["test"]:
            self.folders = partitions[partitions['partition'] == 2]['filename'].values
            # ## To get the translation-equivariance img in front of the test set (incl. Ellen selfie)
            # self.files = partitions[(partitions['partition'] == 2) | (partitions['partition'] == 3)]['filename'].values
            # self.files = np.concatenate((self.files[-1:], self.files[:-1]))
        else:
            raise ValueError(f"Invalid data split provided. Got {data_split}")

        files = []
        for celeb_id in self.folders:
            for event in os.listdir(data_path+f"mp4/{celeb_id}/"):
                if event != ".DS_Store":
                    for mp4 in os.listdir(data_path+f"mp4/{celeb_id}/{event}"):
                        if mp4.endswith(".mp4"):
                            files.append(data_path+f"mp4/{celeb_id}/{event}/{mp4}")
        self.files = files

        if data_split in ["train", "val"]:
            self.adaptation = False
        elif data_split in ["test"]:
            self.adaptation = True
        else:
            raise ValueError(f"Invalid data split provided. Got {data_split}")

        ## A list of MVPs images (or the worst during self-modulation) - Useful for active learning
        # self.mvp_files = self.files

        self.total_envs = len(self.files)
        if self.total_envs == 0:
            raise ValueError("No files found for the split.")

        self.total_pixels = self.img_size[0] * self.img_size[1]     ## Per image !

    def get_video(self, filename) -> torch.Tensor:
        vidcap = cv2.VideoCapture(filename)
        success,image = vidcap.read()
        images = []
        while success:
            success, image = vidcap.read()
            images.append(image)

        if self.nb_frames > len(images):
            raise ValueError(f"Number of frames requested is greater than the number of frames in the video. Got {self.nb_frames} frames, but the video has {len(images)} frames.")
        elif self.nb_frames > len(images)//2:
            images = images[:self.nb_frames]
        else:
            images = images[::len(images)//self.nb_frames]
        
        if len(images) > self.nb_frames:
            images = images[:self.nb_frames]

        ## Convert and resize the images (in fewer numbers)
        video = []
        for img in images:
            # img_norm = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rescaled = Image.fromarray(img_rgb).resize((self.img_size[0], self.img_size[1]), Image.LANCZOS)
            img_rescaled = np.array(img_rescaled) / 255.0
            video.append(img_rescaled)

        return np.stack(video, axis=0)

    def sample_img_pixels(self, img) -> Tuple[np.ndarray, jnp.ndarray]:
        ## img is a video frame with shape (H, W, C), but this method works in the classical (W, H, C) format
        img = img.transpose(1, 0, 2)

        total_pixels = self.img_size[0] * self.img_size[1]

        if self.order_pixels:
            flattened_indices = np.arange(self.nb_shots)
        else:
            flattened_indices = np.random.choice(total_pixels, size=self.nb_shots, replace=False)

        x, y = np.unravel_index(flattened_indices, (self.img_size[0], self.img_size[1]))
        coords = np.vstack((x, y)).T
        normed_coords = (coords / np.array(self.img_size[:2]))

        pixel_values = img[coords[:, 0], coords[:, 1], :]

        return np.concatenate((normed_coords, pixel_values), axis=-1)

    def sample_video_pixels(self, video):
        all_video_coords_pixels = []
        for i in range(video.shape[0]):
            coords_pixels = self.sample_img_pixels(video[i])
            all_video_coords_pixels.append(coords_pixels)
        return np.stack(all_video_coords_pixels, axis=0)

    def __getitem__(self, idx):
        # print(f"Loading video {self.files[idx]}")
        img = self.get_video(self.files[idx])
        coords_pixels = self.sample_video_pixels(img)
        return coords_pixels

    def __len__(self):
        return self.total_envs


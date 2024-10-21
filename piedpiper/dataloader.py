from selmod import DataLoader, CelebADataset

class ImageDataset(DataLoader):
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
        self.target_sets = CelebADataset(data_path, 
                                        data_split=data_split,
                                        num_shots=np.prod(resolution), 
                                        order_pixels=order_pixels, 
                                        resolution=resolution,
                                        )

        assert self.context_sets.total_envs == self.target_sets.total_envs, "ERROR: The number of environments in the context and target dataloaders must be the same."
        self.total_envs = max_envs if max_envs > 0 else self.context_sets.total_envs


    def __getitem__(self, idx):
        ctx_img = self.context_sets.get_image(self.context_sets.files[idx])
        ctx_normed_coords, ctx_pixel_values = self.context_sets.sample_pixels(ctx_img)

        tgt_img = self.target_sets.get_image(self.target_sets.files[idx])
        tgt_normed_coords, tgt_pixel_values = self.target_sets.sample_pixels(tgt_img)

        return torch.concat((ctx_normed_coords, ctx_pixel_values), axis=-1), torch.concat((tgt_normed_coords, tgt_pixel_values), axis=-1)


    def __len__(self):
        return self.total_envs



class VideoDataset(DataLoader):
    """
    A vox celeb dataloader for meta-learning.
    """
    def __init__(self, 
                 data_path="./data/", 
                 data_split="train",
                 num_shots=100,
                 resolution=(32, 32),
                 order_pixels=False,
                 seed=None):

        pass


    def __getitem__(self, idx):
        ctx_vid = self.context_sets.get_video(self.context_sets.files[idx])
        ctx_normed_coords, ctx_pixel_values = self.context_sets.sample_pixels(ctx_vid)

        tgt_vid = self.target_sets.get_video(self.target_sets.files[idx])
        tgt_normed_coords, tgt_pixel_values = self.target_sets.sample_pixels(tgt_vid)

        return torch.concat((ctx_normed_coords, ctx_pixel_values), axis=-1), torch.concat((tgt_normed_coords, tgt_pixel_values), axis=-1)


    def __len__(self):
        return self.total_envs

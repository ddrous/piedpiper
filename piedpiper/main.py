#%%
# from selfmod import CelebADataset, make_image
from selfmod import *

## For reproducibility
seed = 2024

## Dataloader hps
k_shots = 100
resolution = (32, 32)
data_folder="../../Self-Mod/examples/celeb-a/data/"
shuffle = False
num_workers = 24

envs_batch_size = 12
envs_batch_size_val = 12

# os.listdir(data_folder)

#%% 

mother_key = jax.random.PRNGKey(seed)
data_key, model_key, trainer_key, test_key = jax.random.split(mother_key, num=4)

##### Numpy Loader
train_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                            data_split="train",
                                            num_shots=k_shots, 
                                            order_pixels=False, 
                                            resolution=resolution,
                                            # seed=seed
                                            ), 
                              batch_size=envs_batch_size, 
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)
all_shots_train_dataloader = NumpyLoader(CelebADataset(data_folder, 
                                            data_split="train",
                                            num_shots=np.prod(resolution), 
                                            order_pixels=False, 
                                            resolution=resolution,
                                            # seed=seed,
                                            ), 
                              batch_size=envs_batch_size_val, 
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=False)

dat = next(iter(all_shots_train_dataloader))
dat[0].shape, dat[1].shape


## Collect a few xy coodrinate pairs, rgb pxiels tripales, and make a imnage to show

#%% 
img = make_image(dat[0][0], dat[1][0], img_size=(*resolution, 3))

plt.imshow(img)

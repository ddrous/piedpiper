from ._config import *
from selfmod import make_image, interpolate_2D_image

class VisualTester:
    def __init__(self, trainer):
        self.trainer = trainer

    def visualize_losses(self, save_path, log_scale=False, ylim=None):
        losses = self.trainer.train_losses
        if isinstance(losses, list):
            losses = jnp.stack(losses)

        ## Plot the loss curve
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        if ylim is not None:
            ax.plot(np.clip(losses, None, ylim))
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Negative Log Likelihood")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title("Loss curve")

        fig.savefig(save_path+"loss_curve.png")

    def visualize_video_frames(self, video, resolution, title=None):
        nb_frames = video.shape[0]
        fig, axs = plt.subplots(1, nb_frames, figsize=(2*nb_frames, 2))
        for i in range(nb_frames):
            if video.ndim == 3:
                xy, rgb = video[i, :, :2], video[i, :, 2:]
                img = make_image(xy, rgb, img_size=(*resolution, 3))
            else:
                img = video[i]
            axs[i].imshow(img)
            # axs[i].axis("off")
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_title(f"Frame {i}")
        if title is not None:
            fig.suptitle(title, y=1.05)
        plt.show()

    def visualize_images(self, dataloader, plot_ids=None, nb_envs=None, key=None, save_path=None, interp_method="linear"):

        # Visualize an examploe prediction from the latest batch
        if key is None:
            key = jax.random.PRNGKey(time.time_ns())

        ctx_batch, tgt_batch = next(iter(dataloader))
        img_shape = dataloader.dataset.context_sets.img_size

        Xc, Yc = ctx_batch[..., :2], ctx_batch[..., 2:]
        Xt, Yt = tgt_batch[..., :2], tgt_batch[..., 2:]
        mus, sigmas = self.trainer.learner.model(ctx_batch)

        nb_envs_max = ctx_batch.shape[0]
        if plot_ids is None:
            if nb_envs is None:
                nb_envs = 2
            plot_ids = jax.random.randint(key, (nb_envs,), 0, nb_envs_max)
        nb_envs = len(plot_ids) if plot_ids is not None else nb_envs
        nb_envs = min(nb_envs, nb_envs_max)
        print(f"Plotting {nb_envs} environments")

        fig, axs = plt.subplots(nb_envs, 5, figsize=(20, 4*nb_envs))
        for e in range(nb_envs):
        # for e, plt_id in enumerate(plot_ids):
            e_ = plot_ids[e]

            if nb_envs > 1:
                ax1, ax2, ax3, ax4, ax5 = axs[e]
            else:
                ax1, ax2, ax3, ax4, ax5 = axs

            img_true = make_image(Xt[e_], Yt[e_], img_size=img_shape)
            ax1.imshow(img_true)
            if e==0:
                ax1.set_title(f"Target", fontsize=18)

            img_fw = make_image(Xc[e_], Yc[e_], img_size=img_shape)
            ax2.imshow(img_fw)
            if e==0:
                ax2.set_title(f"Context Set", fontsize=18)

            img_pred = mus[e_]
            ax3.imshow(img_pred)
            if e==0:
                ax3.set_title(f"Prediction", fontsize=18)

            img_std = sigmas[e_]
            ax4.imshow(img_std)
            if e==0:
                ax4.set_title(f"Uncertainty", fontsize=18)

            interpolation = interpolate_2D_image(np.asarray(Xc[e_]), np.asarray(Yc[e_]), img_shape, method=interp_method)
            ax5.imshow(interpolation)
            if e==0:
                ax5.set_title(f"{interp_method} Int.", fontsize=18)

            ax5.set_xticks([])
            ax5.set_yticks([])
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax4.set_xticks([])
            ax4.set_yticks([])

        if save_path is not None:
            fig.savefig(save_path)

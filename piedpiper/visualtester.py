from ._config import *
from selfmod import make_image, interpolate_2D_image
import imageio

class VisualTester:
    def __init__(self, trainer):
        self.trainer = trainer

    def visualize_losses(self, save_path, log_scale=False, ylim=None):
        losses = self.trainer.train_losses
        if isinstance(losses, list):
            losses = jnp.stack(losses)

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        if ylim is not None:
            losses = np.clip(losses, None, ylim)

        ax.plot(losses, label="Training", color="royalblue")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Negative Log Likelihood")
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="lower left")

        ## Make a twin axis for validation losses
        val_losses = self.trainer.val_losses
        if len(val_losses) > 0:
            ax2 = ax.twinx()

            len_train = len(losses)
            len_val = len(val_losses)
            val_points = np.linspace(0, len_train, len_val, endpoint=False).astype(int)

            ax2.plot(val_points, val_losses, ".", color="crimson", label="Validation")
            y_label = self.trainer.val_criterion if self.trainer.val_criterion is not None else "Not Specified"
            ax2.set_ylabel(y_label)
            if log_scale:
                ax2.set_yscale("log")
            ax2.legend(loc="upper right")

        ax.set_title("Loss Curves")

        fig.savefig(save_path, bbox_inches='tight')

    def visualize_video_frames(self, video, resolution, title=None, save_path=None):
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

        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')

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
                ax1.set_title(f"Target", fontsize=22)

            img_fw = make_image(Xc[e_], Yc[e_], img_size=img_shape)
            ax2.imshow(img_fw)
            if e==0:
                ax2.set_title(f"Context Set", fontsize=22)

            img_pred = mus[e_]
            ax3.imshow(img_pred)
            if e==0:
                ax3.set_title(f"Prediction", fontsize=22)

            img_std = sigmas[e_]
            ax4.imshow(img_std)
            if e==0:
                ax4.set_title(f"Uncertainty", fontsize=22)

            interpolation = interpolate_2D_image(np.asarray(Xc[e_]), np.asarray(Yc[e_]), img_shape, method=interp_method)
            ax5.imshow(interpolation)
            if e==0:
                ax5.set_title(f"{interp_method.capitalize()} Int.", fontsize=22)

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
            fig.savefig(save_path, bbox_inches='tight')



    def visualize_videos(self, dataloader, nb_envs=None, save_path=None, interp_method="linear", key=None, bootstrap=True, save_video=True, video_prefix="sample"):
        """ Visualize a batch of videos from the dataloader: ctx, tgt, pred, interp """
        if key is None:
            key = jax.random.PRNGKey(time.time_ns())
        img_shape = dataloader.dataset.context_sets.img_size
        nb_frames = dataloader.dataset.context_sets.nb_frames

        batch_size = dataloader.batch_size
        if nb_envs < batch_size:
            ctx_batch, tgt_batch = next(iter(dataloader))
            ctx_batch = ctx_batch[:nb_envs]
            tgt_batch = tgt_batch[:nb_envs]
        else:
            """Concatenate multiple batches"""
            ctx_batch, tgt_batch = [], []
            for _ in range(nb_envs//batch_size):
                ctx, tgt = next(iter(dataloader))
                ctx_batch.append(ctx)
                tgt_batch.append(tgt)
            ctx_batch = np.concatenate(ctx_batch, axis=0)
            tgt_batch = np.concatenate(tgt_batch, axis=0)

        tgt_videos = tgt_batch
        ctx_videos = ctx_batch
        if bootstrap:
            (pred_videos, uq_videos), _ = eqx.filter_vmap(self.trainer.learner.model.bootstrap_predict)(tgt_videos)
        else:
            # (pred_videos, uq_videos), _ = eqx.filter_vmap(self.trainer.learner.model.naive_predict)(ctx_videos)
            (pred_videos, uq_videos), _ = eqx.filter_vmap(self.trainer.learner.model)(ctx_videos)

        def get_img(video, i, return_ints=False):
            if video.ndim == 3:
                xy, rgb = video[i, :, :2], video[i, :, 2:]
                img = make_image(xy, rgb, img_size=img_shape)
            else:
                img = video[i]
            img = np.clip(img, 0., 1.)
            if return_ints:
                img = (img*255).astype(np.uint8)
            return img
    
        nb_vids_per_env = 6
        fig, ax = plt.subplots(nb_vids_per_env*nb_envs, nb_frames+1, figsize=(2*(nb_frames+1), nb_vids_per_env*2*nb_envs))

        for e in range(nb_envs):
            tgt_video = tgt_videos[e]
            ctx_video = ctx_videos[e]
            pred_video = pred_videos[e]
            uq_video = uq_videos[e]

            ## In all columns of row (e*nb_vids_per_env+0, :), and print the environment number centered
            ax[e*nb_vids_per_env, nb_frames//2].text(0.5, 0.5, f"Env {e}", fontsize=36, ha='center', va='top')

            ax[e*nb_vids_per_env, 0].axis('off')
            ax[e*nb_vids_per_env+1, 0].axis('off')
            ax[e*nb_vids_per_env+2, 0].axis('off')
            ax[e*nb_vids_per_env+3, 0].axis('off')
            ax[e*nb_vids_per_env+4, 0].axis('off')
            ax[e*nb_vids_per_env+5, 0].axis('off')
            # if e==0:
            ax[e*nb_vids_per_env+1, 0].text(0.5, 0.5, f"Target\nSet", fontsize=22, ha='center', va='center')
            ax[e*nb_vids_per_env+2, 0].text(0.5, 0.5, f"Context\nSet", fontsize=22, ha='center', va='center')
            ax[e*nb_vids_per_env+3, 0].text(0.5, 0.5, f"Prediction", fontsize=22, ha='center', va='center')
            ax[e*nb_vids_per_env+4, 0].text(0.5, 0.5, f"Uncertainty", fontsize=22, ha='center', va='center')
            ax[e*nb_vids_per_env+5, 0].text(0.5, 0.5, f"{interp_method.capitalize()}\nInterp.", fontsize=22, ha='center', va='center')

            for i in range(1, nb_frames+1):

                ax[e*nb_vids_per_env, i].axis('off')

                ## The taget images at the top row
                img_tgt = get_img(tgt_video, i-1)
                ax[e*nb_vids_per_env+1, i].imshow(img_tgt)
                ax[e*nb_vids_per_env+1, i].set_xticks([])
                ax[e*nb_vids_per_env+1, i].set_yticks([])

                ## The context images in the second row
                img_ctx = get_img(ctx_video, i-1)
                ax[e*nb_vids_per_env+2, i].imshow(img_ctx)
                ax[e*nb_vids_per_env+2, i].set_xticks([])
                ax[e*nb_vids_per_env+2, i].set_yticks([])

                ## The prediction images in the third row
                img_pred = get_img(pred_video, i-1)
                ax[e*nb_vids_per_env+3, i].imshow(img_pred)
                ax[e*nb_vids_per_env+3, i].set_xticks([])
                ax[e*nb_vids_per_env+3, i].set_yticks([])

                ## The uncertainty images in the fourth row
                img_uq = get_img(uq_video, i-1)
                ax[e*nb_vids_per_env+4, i].imshow(img_uq)
                ax[e*nb_vids_per_env+4, i].set_xticks([])
                ax[e*nb_vids_per_env+4, i].set_yticks([])

                ## The interpolated images in the fifth row
                img_interp = interpolate_2D_image(ctx_video[i-1,...,:2], ctx_video[i-1,...,2:], img_shape, method=interp_method)
                ax[e*nb_vids_per_env+5, i].imshow(img_interp)
                ax[e*nb_vids_per_env+5, i].set_xticks([])
                ax[e*nb_vids_per_env+5, i].set_yticks([])

                if True:
                    ax[e*nb_vids_per_env+1, i].set_title(f"Frame {i}")


        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')

        if save_video:
            # Save the video (for the very last environment only)
            save_path_tgt = video_prefix+ "_tgt.mp4"
            with imageio.get_writer(save_path_tgt, mode='I') as writer:
                for i in range(nb_frames):
                    writer.append_data(get_img(tgt_video, i, return_ints=True))

            save_path_ctx = video_prefix + "_ctx.mp4"
            with imageio.get_writer(save_path_ctx, mode='I') as writer:
                for i in range(nb_frames):
                    writer.append_data(get_img(ctx_video, i, return_ints=True))

            save_path_pred = video_prefix + "_pred.mp4"
            with imageio.get_writer(save_path_pred, mode='I') as writer:
                for i in range(nb_frames):
                    writer.append_data(get_img(pred_video, i, return_ints=True))
            
            save_path_uq = video_prefix + "_uq.mp4"
            with imageio.get_writer(save_path_uq, mode='I') as writer:
                for i in range(nb_frames):
                    writer.append_data(get_img(uq_video, i, return_ints=True))

            save_path_interp = video_prefix + "_interp.mp4"
            with imageio.get_writer(save_path_interp, mode='I') as writer:
                for i in range(nb_frames):
                    int_img = interpolate_2D_image(ctx_video[i,...,:2], ctx_video[i,...,2:], img_shape, method=interp_method)*255
                    int_img = int_img.astype(np.uint8)
                    writer.append_data(int_img)

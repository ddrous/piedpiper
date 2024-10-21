

class VisualTester:
    def __init__(self, trainer):
        self.trainer = trainer

    def visualize_images(self, dataloader):

        # Visualize an examploe prediction from the latest batch
        ctx_batch, tgt_batch = next(dataloader)

        Xc, Yc = ctx_batch
        Xt, Yt = tgt_batch
        mus, sigmas = model(Xc, Yc)
        # mus, sigmas = jax.tree_map(lambda x: x.reshape((-1, 1024, C)), (mus, sigmas))

        test_key = jax.random.PRNGKey(time.time_ns())
        # Yt_hat = jax.random.normal(test_key, mus.shape) * sigmas + mus
        Yt_hat = mus

        plt_idx = jax.random.randint(test_key, (1,), 0, envs_batch_size_all)[0]
        # print("Yt_hat shape: ", sigmas[plt_idx])

        img_true = make_image(Xt[plt_idx], Yt[plt_idx], img_size=(*resolution, 3))
        ax1.imshow(img_true)
        ax1.set_title(f"Target")

        img_fw = make_image(Xc[plt_idx], Yc[plt_idx], img_size=(*resolution, 3))
        ax2.imshow(img_fw)
        ax2.set_title(f"Context Set")

        # img_pred = make_image(Xt[plt_idx], Yt_hat[plt_idx], img_size=(*resolution, 3))
        img_pred = mus[plt_idx]
        ax3.imshow(img_pred)
        ax3.set_title(f"Prediction")

        # img_std = make_image(Xt[plt_idx], sigmas[plt_idx], img_size=(*resolution, 3))
        img_std = sigmas[plt_idx]
        ## rescale to 0-1
        # img_std = (img_std - img_std.min()) / (img_std.max() - img_std.min())
        ax4.imshow(img_std)
        ax4.set_title(f"Uncertainty")

        fig.savefig("predictions.png")

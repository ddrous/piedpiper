## Design decisions
- Dataloader return XYRGB_ctx, XYRGB_tgt. of shapes (100, 5), (1024, 5) if num_shots=100
- For videos, the number of frames is the leading dimension: (24, 100, 5), (24, 1024, 5) for instance


## The model has two modes of prediction:
- naive: we provide the context sets for all the frames. Can be used from both training (compression) and testing (decompression) phases.
- bootstrapping: we provide the full video. The frames to use at each step are samples based on the uncertainty of the model. Can only be used during training.


Otherwise, everything is quite similar to other libraries...
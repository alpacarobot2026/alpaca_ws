# Vendored TalkNCE subset

`dlhammer/`, `model/`, `torchvggish/`, `utils/`, `loconet.py`, `loss_multi.py`
and `configs/test.yaml` copied from https://github.com/kaistmm/TalkNCE at
commit `4295996`, plus local modifications made for this project that were
never pushed upstream:

- `loconet.py`: `loadParameters()` skips checkpoint tensors whose shape no
  longer matches the live model (needed after local architecture tweaks to
  `audioEncoder`), instead of failing to load.
- `model/loconet_encoder.py`: `VGGish(..., pretrained=False)` instead of the
  upstream default `pretrained=True`. `loadParameters()` overwrites this
  whole state dict from the TalkNCE checkpoint anyway, so the upstream
  default silently downloaded discarded weights from `torch.hub` on every
  cold start - a needless internet dependency for a robot that may not
  always have connectivity.
- `scripts/ros2_infer_live.py`: face-detector model path and default audio
  device made relocatable (were hardcoded to a specific dev machine and USB
  mic); see the ASD section in `docker/README.md`.

`dlhammer` (MIT-licensed, embedded upstream by the TalkNCE authors rather
than a submodule) is kept whole (136 KB). Only `distributed.py` from `utils/`
is actually imported at inference time; the rest of that directory is
training/eval tooling kept for parity with upstream rather than trimmed.

This tree isn't colcon-installable as-is: `dlhammer`/`torchvggish` do
absolute imports (`import model.xxx`) that only resolve with this directory
itself on `sys.path`, which setuptools packaging doesn't preserve. Run it via
`ros2 run alpaca_interaction talknce_node`, whose entry point
(`../alpaca_interaction/talknce_node.py`) locates this source tree at
runtime and execs `scripts/ros2_infer_live.py` in place, rather than
importing it directly.

Omitted from upstream: training entry points (`train.py`, `builder.py`,
`dataLoader_multiperson.py`, `test_multicard.py`), the AVA dataset,
`venv/`, and all local test videos. The checkpoint (`UniTalk_TalkNCE.model`,
132 MB) isn't vendored either - see [Model weights](../../README.md#model-weights),
it's delivered the same way as the RPEA/prediction checkpoints, via `/models`.

`assets/blaze_face_full_range_sparse.tflite` is Google's MediaPipe
BlazeFace model, used unmodified for face detection ahead of the TalkNCE
active-speaker model.

Upstream license: MIT, see LICENSE.

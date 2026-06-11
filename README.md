# VoteNet (Unofficial TensorFlow Implementation)

An unofficial TensorFlow/[Tensorpack](https://github.com/tensorpack/tensorpack) implementation of [Deep Hough Voting for 3D Object Detection in Point Clouds (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/html/Qi_Deep_Hough_Voting_for_3D_Object_Detection_in_Point_Clouds_ICCV_2019_paper.html) by Qi et al., trained and evaluated on SUN RGB-D.

For the authors' official PyTorch implementation, see [facebookresearch/votenet](https://github.com/facebookresearch/votenet).

## Overview

VoteNet detects 3D objects directly from raw point clouds. A PointNet++ backbone extracts seed points, each seed casts a *vote* toward the object center it (possibly) belongs to (a deep, learned analogue of the Hough transform), votes are clustered into object proposals, and each proposal predicts an oriented 3D bounding box with a class label. Final detections are obtained after 3D non-maximum suppression.

This repository implements that pipeline with:

- `model.py` — voting + proposal network on a PointNet++ backbone.
- `tf_ops/` — custom TensorFlow CUDA ops: farthest-point `sampling`, ball-query `grouping`, `3d_interpolation`, and `3d_nms`.
- `dataset.py` / `sunutils.py` — SUN RGB-D data loading and box utilities.
- `evaluator.py` — per-class AP / mAP evaluation callback.
- `run.py` — training entry point.

## Setup

1. Install TensorFlow 1.x (GPU), Tensorpack, and numpy.
2. Compile the custom TensorFlow ops, as in [PointNet++](https://github.com/charlesq34/pointnet2): run the `*_compile.sh` script inside each `tf_ops/*` folder (adjust the TensorFlow/CUDA paths for your machine).

## Data

Prepare the SUN RGB-D dataset following the instructions of [frustum-pointnets](https://github.com/charlesq34/frustum-pointnets/tree/master/sunrgbd); only the `extract_rgbd_data.m` step is needed.

## Training

Point `MyDataFlow` (in `run.py`) at the root folder of the generated data, then:

```bash
python run.py
```

Training logs and checkpoints are managed by Tensorpack; validation mAP is computed periodically by the `Evaluator` callback.

## Results

Partial reproduction on SUN RGB-D after 75 epochs (per-class AP):

| Class | AP | Class | AP |
| --- | --- | --- | --- |
| bed | 0.371 | chair | 0.054 |
| bookshelf | 0.154 | dresser | 0.035 |
| sofa | 0.122 | toilet | 0.011 |
| night stand | 0.007 | table | 0.003 |
| desk | 0.002 | bathtub | 0.000 |

Overall mAP: **0.076** — well below the paper's reported numbers, so treat this implementation as a starting point for experimentation rather than a faithful reproduction. The official implementation linked above reproduces the paper's results.

## Reference

```bibtex
@inproceedings{qi2019deep,
  title={Deep Hough Voting for 3D Object Detection in Point Clouds},
  author={Qi, Charles R and Litany, Or and He, Kaiming and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

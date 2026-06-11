# VoteNet

<!-- README refined by Cursor -->

Deep Hough Voting for 3D Object Detection in Point Clouds (https://arxiv.org/abs/1904.09664)

## Overview

This repository contains Python, C++, CUDA code from an older research, course, or prototype project. The README has been refreshed to make the repository easier to scan while preserving the original notes below.

## Repository Contents

- `tf_ops/`

## Setup

- This legacy repo does not pin a full environment. Start from the language/toolchain implied by the source files, then install missing packages as reported by the runtime.

## Usage

- inspect the top-level Python entry points: `config.py`, `dataset.py`, `evaluator.py`, `model.py`, `run.py`

## Data and Artifacts

No new large artifact is stored in this repository. If a dataset or checkpoint is required, follow the links and notes in the original section below.

## Status

This is a `Batch B` cleanup pass for a legacy repository. Commands may require dependency/version adjustments on a modern machine.

## License

No explicit license file was found in this checkout; check the original project context before reusing code.

## Original Notes

# VoteNet
This is an unofficial implementation of "Deep Hough Voting for 3D Object Detection in Point Clouds" (https://arxiv.org/abs/1904.09664)

# Training
* Prepare SUN RGB-D dataset following the instructions of https://github.com/charlesq34/frustum-pointnets/tree/master/sunrgbd, note you only need to run `extract_rgbd_data.m`
* Compile custom Tensorflow ops as described in PointNet++
* Pass the root folder of generated data in `MyDataFlow` and run `run.py`.

# TODOs
* ~~Data augmentation~~
* ~~Test/Validation mAP~~
* ~~Train/Validation split~~
* ~~3D NMS~~

# Results (To be updated)
* After 75 epochs, I got the following AP:
  - table:0.002599436474719265
  - bed:0.3710421555617831
  - night_stand:0.0067538466938640174
  - bookshelf:0.15430493020623803
  - chair:0.05431061678741693
  - dresser:0.035260119681139616
  - sofa:0.12197371148148911
  - desk:0.0022836462245173833
  - toilet:0.010896393557409
  - bathtub:0.0

  - mAP:0.075942

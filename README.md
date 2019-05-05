# VoteNet
This is an unofficial implementation of "Deep Hough Voting for 3D Object Detection in Point Clouds" (https://arxiv.org/abs/1904.09664)

# Training
* Prepare SUN RGB-D dataset following the instructions of https://github.com/charlesq34/frustum-pointnets/tree/master/sunrgbd, note you only need to run `extract_rgbd_data.m`
* Pass the root folder of generated data in `MyDataFlow` and run `run.py`.

# TODOs
* Data augmentation
* Test/Validation mAP

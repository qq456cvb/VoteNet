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

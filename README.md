# Basic Python SFM

Run sfm.py with calibration_data.npz (for GoPro, otherwise use camera.npz obtained through calibrate.py) and images folder to obtain point and color pairs for display in a point cloud, such as https://github.com/TarcLO/PointCloud.

To calibrate camera, add pictures of checkerboards to a folder and run calibrate.py with the folder as argument.

orb_slam.py tries to stitch images using ORB. Currently experimental.

GoPro calibration data from https://github.com/EminentCodfish/GoPro-Calibration-Distortion-Removal.

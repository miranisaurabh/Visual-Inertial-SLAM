main.py -->
1) Uses utils.py to load data. 
2) (a) IMU Localization via EKF Prediction
Predicts the IMU Pose
(b) Landmark Mapping via EKF Update
If new features are observed intializes the prior
If previously obvserved features are observed, then updates landmark positions based on predicted and observed feature positions.
(c) Visual-Inertial SLAM
Updates the IMU pose and Landmark positions.
3) The code is well commented for further details

utils.py --> 
1) to read visual features, IMU measurements and calibration parameters
2) Modified the visualize_trajectory_2d() fucntion to display landmars as well using scatter plot
# UCSD ECE 276A Project 2: Particle Filter SLAM
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project focuses on the implementation of simultaneous localization and mapping (SLAM) using an encoder, an IMU odometry, 2-D Li- DAR scans, and RGBD measurements from a differential-drive robot. The odometry and LiDAR measurements are used to localize the robot and build a 2-D occupancy grid map of the environment. The RGBD images are used to assign colors to your 2-D map of the floor.

<p align="center">
  <img src="https://github.com/lintsao/UCSD-ECE-276A-Project-2-Particle-Filter-SLAM/blob/main/gif/test_20.gif" alt="Project Image" width="400">
  <img src="https://github.com/lintsao/UCSD-ECE-276A-Project-2-Particle-Filter-SLAM/blob/main/gif/test_21.gif" alt="Project Image" width="400">
</p>
<p align="center">Here is a visual representation of our project. </p>

## To get started with the motion planning project, follow these steps:

1. Clone this repository:
  ```bash
  git clone https://github.com/lintsao/UCSD-ECE-276A-Project-2-Particle-Filter-SLAM.git
  cd UCSD-ECE-276A-Project-2-Particle-Filter-SLAM
  ```

2. Create a new virtual environment:
  ```bash
  python3 -m venv env
  source env/bin/activate  # For Unix/Linux
  ```

3. Install the required dependencies:
  ```bash
  pip3 install -r requirements.txt
  ```

4. You're ready to use the particle filter slam project!

## Usage

```
cd src
python3 main.py
```

## Source code description:
- **main.py**: Main function.
- **map.py**: Occupancy grid map related class and function.
- **motion.py**: Functions for motion model.
- **observation.py**: Functions for observation model and map correlation.
- **particle.py**: Particle class.
- **transfrom.py**: Transform helper (especially for lidar scan).
- **utils.py**: Functions for file loading, sync data, draw gif etc.
- **test.ipynb**: For testing.

or you could use **test.ipynb** to check the step-by-step implementation.

## Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

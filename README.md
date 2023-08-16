# ECE276A PR2 Winter23 Particle Filter SLAM

## Overview
In this assignment, we implement a particle filter with a differential-drive motion model and scan-grid correlation observation model for simultaneous localization and occupancy-grid mapping.

## Installation
- Install dependencies
```bash
pip install -r requirements.txt
```

## Run code:
```bash
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
- **test.ipynb**: For testing


## Result
Can be found in [report.pdf](./report.pdf)


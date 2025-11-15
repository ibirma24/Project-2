# Phase 2: Relative Pose Estimation

## Overview
This project implements Phase 2 of a camera calibration and pose estimation system. Using the calibrated intrinsic matrix from Phase 1, this phase determines the relative motion (rotation R and translation t) between two camera positions using feature matching and the 8-Point Algorithm.

## Theoretical Background

### The Process Pipeline
1. **Feature Detection & Matching**: Find corresponding points between two images
2. **Fundamental Matrix Estimation**: Use the 8-Point Algorithm with RANSAC
3. **Essential Matrix Computation**: Convert F to E using intrinsic matrix K
4. **Pose Recovery**: Decompose E to extract rotation R and translation t

### Mathematical Relations
- **Fundamental Matrix**: `x₂ᵀ F x₁ = 0` (epipolar constraint)
- **Essential Matrix**: `E = KᵀFK` (metric upgrade from F)
- **Pose Decomposition**: `E = [t]ₓR` (skew-symmetric form)

## Implementation Features

### Feature Detection
- **Primary**: ORB detector (3000 features) for robustness
- **Fallback**: SIFT detector if ORB produces insufficient matches
- **Matching**: BFMatcher for ORB, FLANN for SIFT
- **Filtering**: Lowe's ratio test (0.8 for ORB, 0.7 for SIFT)

### Robust Estimation
- **Algorithm**: 8-Point Algorithm with RANSAC
- **Threshold**: 1.0 pixel reprojection error
- **Confidence**: 99% for RANSAC
- **Minimum Matches**: 15 good matches required

### Validation & Visualization
- **Epipolar Geometry**: Visualizes epipolar lines to validate F matrix
- **Feature Matches**: Shows good matches and inlier matches
- **Essential Matrix**: Validates singular value structure

## Usage

### Command Line Interface
```bash
# Process two specific images
python phase2chosen.py image1.jpg image2.jpg

# Demo mode (uses first two images found)
python phase2chosen.py

# Help
python phase2chosen.py --help
```

### Requirements
- Calibrated camera (Phase 1 completed)
- Two images from different viewpoints
- Sufficient texture/features in scenes
- OpenCV and NumPy installed

## Output Results

### Matrices Computed
- **Fundamental Matrix (F)**: 3x3 matrix encoding epipolar geometry
- **Essential Matrix (E)**: Metric version of F using camera intrinsics
- **Rotation Matrix (R)**: 3x3 orthogonal matrix representing rotation
- **Translation Vector (t)**: 3x1 unit vector (direction only)

### Visualizations Generated
- `good_matches.png`: Feature correspondences after ratio test
- `inlier_matches.png`: RANSAC inliers used for F estimation
- `epipolar_lines.png`: Epipolar lines validating geometry

### Text Output
- Detailed results in `pose_estimation_results.txt`
- Binary data in `pose_estimation_results.pkl`
- Rotation in Euler angles for interpretation
- Translation direction (scale ambiguity noted)

## Key Insights from Test Results

### Successful Estimation
The implementation successfully estimated pose between two chessboard images:
- **Rotation**: -21.50° roll, -18.66° pitch, -168.68° yaw
- **Translation**: Direction (-0.184, 0.025, -0.983)
- **Validation**: Essential matrix has proper singular value structure
- **Quality**: 22% inlier ratio from RANSAC (acceptable for challenging case)

### Technical Notes
1. **Scale Ambiguity**: Only translation direction recoverable from two views
2. **Chessboard Challenge**: Regular patterns have fewer distinctive features
3. **Robustness**: ORB/SIFT fallback ensures successful feature detection
4. **Validation**: Epipolar lines confirm geometric consistency

## Files Structure
```
phase2_results/
├── pose_estimation_results.txt    # Human-readable results
├── pose_estimation_results.pkl    # Binary data for programs
├── good_matches.png               # Initial feature matches
├── inlier_matches.png            # RANSAC inliers
└── epipolar_lines.png            # Epipolar geometry validation
```

## Applications
This relative pose estimation forms the foundation for:
- Visual odometry and SLAM
- Structure from Motion (SfM)
- Augmented Reality tracking
- Robot navigation
- 3D reconstruction

## Future Enhancements
- Bundle adjustment for multi-view refinement
- Loop closure detection for SLAM
- Real-time video processing
- Integration with IMU data
- Dense reconstruction methods
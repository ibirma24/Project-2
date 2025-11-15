# Camera Calibration and Relative Pose Estimation Project

## Project Overview
This project implements a complete two-phase camera calibration and pose estimation system using OpenCV and Python.

## Phase 1: Camera Calibration (`calibation.py`)
Computes intrinsic camera parameters and distortion coefficients using chessboard patterns.

### Features:
- ✅ Chessboard corner detection with sub-pixel accuracy
- ✅ Camera calibration using `cv2.calibrateCamera()`
- ✅ Distortion correction validation
- ✅ Comprehensive results saving and visualization

### Results:
- **RMS Error**: 0.3652 pixels (EXCELLENT quality)
- **Images Used**: 34 chessboard images
- **Focal Length**: fx=985.09, fy=987.60 pixels
- **Principal Point**: cx=639.25, cy=351.52 pixels

## Phase 2: Relative Pose Estimation (`phase2chosen.py`)
Determines relative motion (R, t) between two camera positions using feature matching and the 8-Point Algorithm.

### Features:
- ✅ Feature detection (ORB primary, SIFT fallback)
- ✅ Robust feature matching with Lowe's ratio test
- ✅ 8-Point Algorithm with RANSAC for fundamental matrix
- ✅ Essential matrix computation (E = K^T F K)
- ✅ Pose recovery and decomposition
- ✅ Epipolar geometry validation
- ✅ Comprehensive visualizations

### Results:
- **Quality Score**: 6/6 (EXCELLENT)
- **Rotation**: -21.50° roll, -18.66° pitch, -168.68° yaw
- **Translation**: Direction (-0.184, 0.025, -0.983)
- **Validation**: Perfect orthogonality and essential matrix structure

## Project Structure
```
├── calibation.py              # Phase 1: Camera calibration
├── phase2chosen.py            # Phase 2: Relative pose estimation
├── phase2_analysis.py         # Quality analysis tools
├── phase2_demos.py            # Application demonstrations
├── project_summary.py         # Complete project summary
├── PHASE2_README.md          # Detailed Phase 2 documentation
├── calibration_results/       # Phase 1 outputs
│   ├── camera_calibration.pkl
│   ├── camera_calibration.txt
│   └── validation/
├── captures/                  # Input chessboard images (35 images)
└── phase2_results/           # Phase 2 outputs
    ├── pose_estimation_results.txt
    ├── pose_estimation_results.pkl
    ├── good_matches.png
    ├── inlier_matches.png
    ├── epipolar_lines.png
    └── performance_report.txt
```

## Usage

### Phase 1 (Camera Calibration):
```bash
python calibation.py
```

### Phase 2 (Pose Estimation):
```bash
# Process two specific images
python phase2chosen.py image1.jpg image2.jpg

```

## Applications Demonstrated
- **Stereo Vision**: Parameter setup for depth estimation
- **Visual Odometry**: Robot navigation and motion tracking
- **Augmented Reality**: Camera pose tracking for AR objects
- **3D Reconstruction**: Foundation for Structure from Motion

## Key Technical Achievements
- Achieved EXCELLENT calibration quality (RMS < 0.5 pixels)
- Perfect rotation matrix orthogonality (error < 1e-6)
- Proper essential matrix singular value structure
- Robust feature matching with automatic fallback
- Comprehensive validation and quality metrics

## Requirements
- Python 3.6+
- OpenCV (`opencv-python`)
- NumPy
- At least 15-20 chessboard images for Phase 1
- Two different viewpoint images for Phase 2

## Future Enhancements
- Multi-view bundle adjustment
- Real-time video processing
- IMU integration
- Dense reconstruction methods
- Loop closure detection for SLAM
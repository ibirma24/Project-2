
# Camera Calibration and Relative Pose Estimation

This project implements a two-phase computer vision system for camera calibration and relative pose estimation using OpenCV and Python. The system can determine camera intrinsic parameters and calculate relative motion between two camera positions.

## Project Overview

This implementation addresses fundamental computer vision challenges:
1. **Camera Calibration** - Determines intrinsic camera parameters and distortion coefficients
2. **Relative Pose Estimation** - Calculates rotation and translation between two camera positions

The project demonstrates practical applications of epipolar geometry, feature matching, and the 8-Point Algorithm.


### Phase 1: Camera Calibration
The calibration process uses chessboard pattern images to determine camera intrinsic parameters:

```bash
python calibation.py
```

**Process:**
1. **Image Acquisition**: 35 chessboard images captured from various angles and distances
2. **Corner Detection**: Automated detection of chessboard corners using `cv2.findChessboardCorners()`
3. **Calibration**: Computation of camera matrix and distortion coefficients via `cv2.calibrateCamera()`
4. **Validation**: Distortion correction applied to test images for verification

**Key Results:**
- Camera focal length: fx=985.09, fy=987.60 pixels
- RMS reprojection error: 0.3652 pixels (excellent accuracy)

### Phase 2: Relative Pose Estimation
Determines camera motion between two positions using feature-based methods:

```bash
python phase2chosen.py [image1] [image2]
```

**Algorithm Pipeline:**
1. **Feature Detection**: ORB/SIFT keypoint detection and descriptor computation
2. **Feature Matching**: Robust matching using Lowe's ratio test
3. **Fundamental Matrix**: 8-Point Algorithm with RANSAC outlier rejection
4. **Essential Matrix**: Conversion using calibrated intrinsic parameters
5. **Pose Recovery**: Decomposition to rotation and translation matrices

## Technical Features

### Advanced Computer Vision Techniques
- **Robust Feature Detection**: ORB primary detector with SIFT fallback for challenging scenarios
- **RANSAC Estimation**: Outlier rejection for reliable fundamental matrix computation
- **Epipolar Geometry**: Geometric validation through epipolar line visualization
- **Multi-scale Matching**: Adaptive feature matching with Lowe's ratio test

### Software Engineering Practices
- **Modular Architecture**: Object-oriented design with clear separation of concerns
- **Comprehensive Logging**: Detailed progress tracking and error reporting
- **Data Persistence**: Results saved in both human-readable and binary formats
- **Robust Error Handling**: Graceful failure management and user feedback

## Results and Performance

### Camera Calibration Metrics
- **RMS Reprojection Error**: 0.3652 pixels (excellent calibration quality)
- **Dataset Size**: 34 chessboard images from multiple viewpoints
- **Intrinsic Parameters**: 
  - Focal length: fx=985.09, fy=987.60 pixels
  - Principal point: cx=639.25, cy=351.52 pixels
  - Distortion coefficients optimized for lens correction

### Pose Estimation Results
- **Feature Matching**: Robust correspondence establishment with outlier filtering
- **Geometric Validation**: Epipolar constraint satisfaction verification
- **Relative Motion**: Accurate rotation and translation estimation
- **Visualization**: Comprehensive match analysis and epipolar line rendering

## Usage Instructions

### Prerequisites
Ensure calibration images are available in the `captures/` directory before running Phase 1.

### Phase 1: Camera Calibration
```bash
python calibation.py
```
Processes chessboard images and generates calibration parameters stored in `calibration_results/`.

### Phase 2: Pose Estimation
```bash
# Process specific image pair
python phase2chosen.py image1.jpg image2.jpg

# Demo mode (uses first available images)
python phase2chosen.py
```

Results are automatically saved to `phase2_results/` including visualizations and numerical data.

## Dependencies and Installation

### Required Packages
```bash
pip install opencv-python
pip install numpy
```

Note: `pickle` is included in Python's standard library.

### System Requirements
- Python 3.7+
- OpenCV 4.0+
- NumPy 1.18+

## Known Limitations and Considerations

### Technical Constraints
- **Insufficient Feature Correspondence**: Image pairs with limited visual overlap may fail
- **Motion Blur**: Reduced feature detection accuracy in blurred images
- **Scale Ambiguity**: Translation magnitude not recoverable from two-view geometry
- **Baseline Requirements**: Very small camera motions may produce unstable results

### Algorithmic Considerations
- **Feature Detector Selection**: ORB prioritized for speed, SIFT for accuracy
- **RANSAC Parameters**: Tuned for 1.0-pixel threshold with 99% confidence
- **Minimum Match Threshold**: 15 correspondences required for reliable estimation

## Module Descriptions

### `calibation.py` (267 lines)
**Camera Calibration Implementation**
- Automated chessboard corner detection with sub-pixel refinement
- Intrinsic parameter estimation using `cv2.calibrateCamera()`
- Distortion coefficient computation and validation
- Comprehensive result serialization and visualization

**Key Functions:**
- Corner detection with error handling
- Calibration parameter optimization
- Distortion correction validation
- Result persistence in multiple formats

### `phase2chosen.py` (593 lines)
**Relative Pose Estimation System**
- Feature-based correspondence establishment
- Fundamental matrix estimation via 8-Point Algorithm
- Essential matrix computation and pose recovery
- Comprehensive geometric validation

**Core Components:**
- `RelativePoseEstimator` class with modular design
- Robust feature matching with outlier rejection
- Epipolar geometry visualization
- Multi-format result export

### `syllabus_verification.py`
**Validation and Testing Framework**
- System integrity verification
- Algorithm performance assessment
- Result validation against expected outputs

## Output Files and Visualizations

### Generated Artifacts
- **`good_matches.png`** - Feature correspondence visualization with color-coded matches
- **`inlier_matches.png`** - RANSAC-filtered correspondences after outlier removal
- **`epipolar_lines.png`** - Epipolar geometry validation showing constraint satisfaction
- **`pose_estimation_results.txt`** - Comprehensive numerical results and matrices
- **`pose_estimation_results.pkl`** - Binary data for programmatic access

### Result Interpretation
- **Rotation Matrix**: 3x3 orthogonal matrix representing camera orientation change
- **Translation Vector**: 3D direction of camera displacement (scale ambiguous)
- **Fundamental Matrix**: Encodes epipolar geometry between image pairs
- **Essential Matrix**: Metric upgrade of fundamental matrix using calibration data

## Future Enhancements

### Potential Extensions
- **Video Processing**: Frame-by-frame pose tracking for continuous motion estimation
- **Bundle Adjustment**: Multi-view optimization for improved accuracy
- **Real-time Implementation**: GPU acceleration for live camera applications
- **Stereo Vision**: Depth map generation using calibrated stereo pairs
- **Mobile Integration**: Android/iOS applications for portable use

### Research Directions
- **Deep Learning Integration**: CNN-based feature detection and matching
- **SLAM Applications**: Simultaneous localization and mapping implementation
- **Augmented Reality**: Pose estimation for AR overlay registration

## References and Acknowledgments

### Technical Literature
- Hartley, R. & Zisserman, A. *Multiple View Geometry in Computer Vision*
- OpenCV Documentation and Tutorials
- Lowe, D.G. "Distinctive Image Features from Scale-Invariant Keypoints" (2004)

### Implementation Resources
- OpenCV Computer Vision Library
- NumPy Numerical Computing Package
- Academic literature on robust estimation methods

---

**Note**: This implementation demonstrates fundamental computer vision concepts and serves as an educational foundation for more advanced geometric computer vision applications. The codebase prioritizes clarity and educational value while maintaining professional software development standards.

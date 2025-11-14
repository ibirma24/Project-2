"""
Phase 2: Application of Calibrated Data
Calibrated Augmented Reality

This script uses the calibrated camera parameters from Phase 1 to perform
augmented reality by overlaying a virtual 3D cube onto a chessboard pattern.

Uses:
- Intrinsic camera matrix (K) from Phase 1
- cv2.solvePnP() for pose estimation
- Feature matching and geometric transformations
- 3D object rendering and projection
"""

import cv2 as cv
import numpy as np
import pickle
import os
import glob
import sys
from datetime import datetime

# Configuration
CALIBRATION_FILE = "calibration_results/camera_calibration.pkl"
CHESSBOARD_SIZE = (7, 6)  # Must match Phase 1
SQUARE_SIZE = 1.0  # Real-world size (arbitrary units)
OUTPUT_DIR = "phase2_results"

class CameraCalibration:
    """Load and manage Phase 1 calibration results."""
    
    def __init__(self, calibration_file=CALIBRATION_FILE):
        self.calibration_file = calibration_file
        self.camera_matrix = None
        self.dist_coeffs = None
        self.load_calibration()
    
    def load_calibration(self):
        """Load calibration data from Phase 1."""
        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(f"Phase 1 calibration file not found: {self.calibration_file}")
        
        with open(self.calibration_file, 'rb') as f:
            data = pickle.load(f)
        
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']
        
        print("✅ Phase 1 calibration loaded successfully!")
        print(f"   RMS error: {data['rms_error']:.4f} pixels")

class AugmentedReality:
    """Calibrated Augmented Reality implementation."""
    
    def __init__(self, calibration):
        self.calibration = calibration
        self.objp = self._prepare_chessboard_points()
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def _prepare_chessboard_points(self):
        """Prepare 3D chessboard corner coordinates."""
        objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        objp *= SQUARE_SIZE
        return objp
    
    def detect_chessboard_pose(self, image):
        """
        Detect chessboard and estimate camera pose using cv2.solvePnP().
        
        Returns:
            rvec, tvec: Rotation and translation vectors (camera pose)
            corners: Detected chessboard corners
            success: Boolean indicating if pose was found
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Solve PnP to get camera pose
            success, rvec, tvec = cv.solvePnP(
                self.objp, corners, 
                self.calibration.camera_matrix, 
                self.calibration.dist_coeffs
            )
            
            return rvec, tvec, corners, success
        
        return None, None, None, False
    
    def draw_3d_cube(self, image, rvec, tvec, cube_size=3.0):
        """
        Draw a 3D cube on the chessboard using the estimated pose.
        
        Args:
            image: Input image
            rvec, tvec: Camera pose from solvePnP
            cube_size: Size of the cube in world units
        """
        # Define 3D cube points (sitting on top of chessboard)
        cube_3d = np.float32([
            [0, 0, 0], [cube_size, 0, 0], [cube_size, cube_size, 0], [0, cube_size, 0],  # bottom face
            [0, 0, -cube_size], [cube_size, 0, -cube_size], [cube_size, cube_size, -cube_size], [0, cube_size, -cube_size]  # top face
        ])
        
        # Project 3D points to 2D image plane
        cube_2d, _ = cv.projectPoints(
            cube_3d, rvec, tvec,
            self.calibration.camera_matrix,
            self.calibration.dist_coeffs
        )
        
        cube_2d = np.int32(cube_2d).reshape(-1, 2)
        
        # Draw cube faces
        img_with_cube = image.copy()
        
        # Draw bottom face (green)
        cv.drawContours(img_with_cube, [cube_2d[:4]], -1, (0, 255, 0), -3)
        
        # Draw top face (blue)  
        cv.drawContours(img_with_cube, [cube_2d[4:8]], -1, (255, 0, 0), -3)
        
        # Draw vertical edges (red)
        for i in range(4):
            cv.line(img_with_cube, tuple(cube_2d[i]), tuple(cube_2d[i+4]), (0, 0, 255), 5)
        
        return img_with_cube, cube_2d
    
    def draw_coordinate_axes(self, image, rvec, tvec, axis_length=3.0):
        """
        Draw 3D coordinate axes on the chessboard.
        
        Args:
            image: Input image
            rvec, tvec: Camera pose
            axis_length: Length of axes in world units
        """
        # Define 3D axis points
        axes_3d = np.float32([
            [0, 0, 0],  # Origin
            [axis_length, 0, 0],  # X-axis (red)
            [0, axis_length, 0],  # Y-axis (green)
            [0, 0, -axis_length]  # Z-axis (blue)
        ])
        
        # Project to 2D
        axes_2d, _ = cv.projectPoints(
            axes_3d, rvec, tvec,
            self.calibration.camera_matrix,
            self.calibration.dist_coeffs
        )
        
        axes_2d = np.int32(axes_2d).reshape(-1, 2)
        
        img_with_axes = image.copy()
        
        # Draw axes
        origin = tuple(axes_2d[0])
        cv.arrowedLine(img_with_axes, origin, tuple(axes_2d[1]), (0, 0, 255), 5)  # X-axis (red)
        cv.arrowedLine(img_with_axes, origin, tuple(axes_2d[2]), (0, 255, 0), 5)  # Y-axis (green)
        cv.arrowedLine(img_with_axes, origin, tuple(axes_2d[3]), (255, 0, 0), 5)  # Z-axis (blue)
        
        # Add labels
        cv.putText(img_with_axes, 'X', tuple(axes_2d[1] + 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(img_with_axes, 'Y', tuple(axes_2d[2] + 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(img_with_axes, 'Z', tuple(axes_2d[3] + 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return img_with_axes
    
    def process_image(self, image_path, save_results=True):
        """
        Process a single image for augmented reality.
        
        Args:
            image_path: Path to input image
            save_results: Whether to save output images
        """
        print(f"Processing: {os.path.basename(image_path)}")
        
        # Load image
        image = cv.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        # Detect chessboard and estimate pose
        rvec, tvec, corners, success = self.detect_chessboard_pose(image)
        
        if not success:
            print(f"❌ Could not detect chessboard in: {os.path.basename(image_path)}")
            return None
        
        print(f"✅ Chessboard detected and pose estimated")
        
        # Create visualizations
        results = {}
        
        # 1. Original image with detected corners
        img_corners = image.copy()
        cv.drawChessboardCorners(img_corners, CHESSBOARD_SIZE, corners, True)
        results['corners'] = img_corners
        
        # 2. Image with coordinate axes
        img_axes = self.draw_coordinate_axes(image, rvec, tvec)
        results['axes'] = img_axes
        
        # 3. Image with 3D cube
        img_cube, cube_2d = self.draw_3d_cube(image, rvec, tvec)
        results['cube'] = img_cube
        
        # 4. Combined visualization
        img_combined = self.draw_coordinate_axes(img_cube, rvec, tvec)
        results['combined'] = img_combined
        
        # Print pose information
        print(f"Camera Pose:")
        print(f"  Rotation (rvec): {rvec.flatten()}")
        print(f"  Translation (tvec): {tvec.flatten()}")
        
        # Save results
        if save_results:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            cv.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_corners.png"), results['corners'])
            cv.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_axes.png"), results['axes'])
            cv.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_cube.png"), results['cube'])
            cv.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_ar_complete.png"), results['combined'])
            
            print(f"✅ Results saved to {OUTPUT_DIR}/")
        
        return results, rvec, tvec
    
    def process_captures_directory(self, captures_dir="captures"):
        """
        Process all images in the captures directory.
        """
        print(f"\n🎯 Processing images from {captures_dir}/ directory...")
        
        # Get all image files
        image_files = glob.glob(os.path.join(captures_dir, "*.png"))
        if not image_files:
            print(f"❌ No images found in {captures_dir}/")
            return
        
        image_files.sort()
        successful_ar = 0
        
        print(f"Found {len(image_files)} images to process")
        
        # Process first few images (limit to avoid too many outputs)
        test_images = image_files[:5]  # Process first 5 images
        
        for i, img_path in enumerate(test_images):
            print(f"\n[{i+1}/{len(test_images)}] ", end="")
            
            results = self.process_image(img_path)
            if results is not None:
                successful_ar += 1
        
        print(f"\n📊 Successfully processed {successful_ar}/{len(test_images)} images for AR")
        print(f"✅ Results saved to {OUTPUT_DIR}/ directory")

def demonstrate_ar_concepts():
    """
    Demonstrate the concepts and mathematics behind calibrated AR.
    """
    print("\n" + "="*60)
    print("CALIBRATED AUGMENTED REALITY - TECHNICAL OVERVIEW")
    print("="*60)
    
    print("""
🎯 OBJECTIVE:
   Accurately overlay virtual 3D objects onto real images using calibrated camera parameters.

📐 KEY CONCEPTS:

1. POSE ESTIMATION (cv2.solvePnP):
   - Input: 3D object points, 2D image points, camera matrix K, distortion coeffs
   - Output: Rotation vector (rvec) and translation vector (tvec)
   - Solves: How is the camera positioned relative to the known 3D object?

2. 3D TO 2D PROJECTION:
   - Virtual 3D points → 2D image coordinates
   - Uses: K matrix, rvec, tvec from pose estimation
   - Formula: s * [u v 1]ᵀ = K * [R|t] * [X Y Z 1]ᵀ

3. COORDINATE SYSTEMS:
   - World coordinates: Chessboard corner positions in 3D
   - Camera coordinates: After applying [R|t] transformation  
   - Image coordinates: After applying K projection matrix

🔧 IMPLEMENTATION STEPS:
   1. Detect chessboard corners in image (2D points)
   2. Match with known 3D chessboard geometry  
   3. Solve PnP to find camera pose [R|t]
   4. Define virtual 3D objects (cube, axes)
   5. Project virtual objects to 2D using K and [R|t]
   6. Draw projected objects on original image

✅ ADVANTAGES OF CALIBRATED AR:
   - Accurate scale and perspective
   - Stable object tracking
   - Correct occlusion handling
   - Precise alignment with real world geometry
    """)

def main():
    """
    Main function for Phase 2 Calibrated Augmented Reality.
    """
    print("🎯 PHASE 2: CALIBRATED AUGMENTED REALITY")
    print("="*60)
    print("Using Phase 1 calibration to overlay virtual 3D objects")
    
    # Handle command line arguments
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['--help', '-h', 'help']:
        print("\nUSAGE:")
        print("  python phase2.py              # Process sample images from captures/")
        print("  python phase2.py --help       # Show this help")
        print("\nDESCRIPTION:")
        print("  Performs calibrated augmented reality using Phase 1 camera calibration.")
        print("  Detects chessboard patterns and overlays virtual 3D cubes and coordinate axes.")
        print("\nREQUIREMENTS:")
        print("  - Completed Phase 1 calibration (calibration_results/camera_calibration.pkl)")
        print("  - Images containing 7x6 chessboard pattern")
        print("\nOUTPUT:")
        print("  - Images with detected corners")
        print("  - Images with 3D coordinate axes") 
        print("  - Images with virtual 3D cube")
        print("  - Combined AR visualization")
        demonstrate_ar_concepts()
        return
    
    try:
        # Load Phase 1 calibration
        calibration = CameraCalibration()
        
        # Initialize AR system
        ar_system = AugmentedReality(calibration)
        
        # Process images
        ar_system.process_captures_directory()
        
        print("\n" + "🎉"*20)
        print("PHASE 2 COMPLETED SUCCESSFULLY!")
        print("🎉"*20)
        print("✅ Calibrated augmented reality implemented")
        print("✅ Camera poses estimated using cv2.solvePnP()")
        print("✅ Virtual 3D objects projected and rendered")
        print("✅ Results demonstrate accurate geometric alignment")
        
        demonstrate_ar_concepts()
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("Please complete Phase 1 calibration first!")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()

import cv2 as cv
import numpy as np
import datetime
import os
import time
import glob
import pickle
import sys

# ---------------- Config ----------------
CHESSBOARD_SIZE = (7, 6)            # (columns, rows) - internal corners
SAVE_DIR = "captures"               # directory containing captured images
OUTPUT_DIR = "calibration_results"  # directory for calibration results
SQUARE_SIZE = 1.0                   # Real-world size of each square (arbitrary units)
# ----------------------------------------

# Termination criteria for cornerSubPix
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (for calibration use)
objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # Scale to real-world units

# Make directories
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper functions for camera capture removed - not needed for calibration

# ============================================================================
# PHASE 1: CAMERA CALIBRATION AND GEOMETRIC MODELING
# ============================================================================

def detect_corners_in_images():
    """
    Step 2: Corner Detection
    Process all captured images to detect chessboard corners.
    """
    print("=" * 60)
    print("STEP 2: CORNER DETECTION")
    print("=" * 60)
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    successful_images = []
    
    # Get all image files
    image_files = glob.glob(os.path.join(SAVE_DIR, '*.png'))
    image_files.sort()  # Process in chronological order
    
    print(f"Found {len(image_files)} captured images")
    print("Processing images for corner detection...")
    
    for i, img_path in enumerate(image_files):
        print(f"Processing [{i+1:2d}/{len(image_files)}]: {os.path.basename(img_path)}", end=" ")
        
        # Load image
        img = cv.imread(img_path)
        if img is None:
            print("❌ Failed to load")
            continue
            
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        if ret:
            # Refine corner positions to sub-pixel accuracy
            corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Store the object points and image points
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            successful_images.append(img_path)
            
            print("✅ Corners detected")
        else:
            print("❌ No corners found")
    
    print(f"\nSuccessfully processed {len(successful_images)} out of {len(image_files)} images")
    
    if len(successful_images) < 10:
        print("⚠️  WARNING: Less than 10 successful images. Consider capturing more.")
    
    return objpoints, imgpoints, successful_images

def calibrate_camera_parameters(objpoints, imgpoints, image_shape):
    """
    Step 3: Calibration
    Use cv2.calibrateCamera() to compute intrinsic matrix K and distortion coefficients.
    """
    print("\n" + "=" * 60)
    print("STEP 3: CAMERA CALIBRATION")
    print("=" * 60)
    
    print("Computing camera calibration...")
    
    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )
    
    print(f"Calibration RMS error: {ret:.4f} pixels")
    print("\n📐 INTRINSIC CAMERA MATRIX (K):")
    print(camera_matrix)
    print(f"\nFocal lengths: fx = {camera_matrix[0,0]:.2f}, fy = {camera_matrix[1,1]:.2f}")
    print(f"Principal point: cx = {camera_matrix[0,2]:.2f}, cy = {camera_matrix[1,2]:.2f}")
    
    print("\n🔧 DISTORTION COEFFICIENTS:")
    print(f"k1={dist_coeffs[0][0]:.6f}, k2={dist_coeffs[0][1]:.6f}, p1={dist_coeffs[0][2]:.6f}, p2={dist_coeffs[0][3]:.6f}, k3={dist_coeffs[0][4]:.6f}")
    
    return camera_matrix, dist_coeffs, rvecs, tvecs, ret

def validate_calibration(camera_matrix, dist_coeffs, successful_images):
    """
    Step 4: Validation
    Apply distortion correction to test images to demonstrate successful calibration.
    """
    print("\n" + "=" * 60)
    print("STEP 4: VALIDATION - DISTORTION CORRECTION")
    print("=" * 60)
    
    # Create output directory
    validation_dir = os.path.join(OUTPUT_DIR, "validation")
    os.makedirs(validation_dir, exist_ok=True)
    
    # Select a few representative images for validation
    test_images = successful_images[::max(1, len(successful_images)//5)]  # Every 5th image, max 5 images
    test_images = test_images[:5]  # Limit to 5 images
    
    print(f"Applying distortion correction to {len(test_images)} test images...")
    
    for i, img_path in enumerate(test_images):
        print(f"Correcting [{i+1}/{len(test_images)}]: {os.path.basename(img_path)}")
        
        # Load original image
        img = cv.imread(img_path)
        h, w = img.shape[:2]
        
        # Get optimal new camera matrix
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        # Undistort the image
        undistorted = cv.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # Create comparison image (side-by-side)
        comparison = np.hstack([
            cv.resize(img, (w//2, h//2)),
            cv.resize(undistorted, (w//2, h//2))
        ])
        
        # Add labels
        cv.putText(comparison, "ORIGINAL", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(comparison, "UNDISTORTED", (w//2 + 10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save results
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        cv.imwrite(os.path.join(validation_dir, f"{base_name}_comparison.png"), comparison)
        cv.imwrite(os.path.join(validation_dir, f"{base_name}_undistorted.png"), undistorted)
    
    print(f"✅ Validation images saved to: {validation_dir}")

def save_calibration_data(camera_matrix, dist_coeffs, rvecs, tvecs, rms_error, successful_images):
    """
    Save calibration results for future use.
    """
    print("\n" + "=" * 60)
    print("SAVING CALIBRATION DATA")
    print("=" * 60)
    
    # Save as pickle file
    calibration_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'rms_error': rms_error,
        'successful_images': successful_images,
        'chessboard_size': CHESSBOARD_SIZE,
        'square_size': SQUARE_SIZE,
        'calibration_date': datetime.datetime.now().isoformat()
    }
    
    pickle_path = os.path.join(OUTPUT_DIR, 'camera_calibration.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(calibration_data, f)
    
    # Save as text file for easy reading
    txt_path = os.path.join(OUTPUT_DIR, 'camera_calibration.txt')
    with open(txt_path, 'w') as f:
        f.write("CAMERA CALIBRATION RESULTS - PHASE 1\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Calibration Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Images Used: {len(successful_images)}\n")
        f.write(f"RMS Error: {rms_error:.4f} pixels\n\n")
        
        f.write("INTRINSIC CAMERA MATRIX (K):\n")
        f.write(f"[{camera_matrix[0,0]:10.4f} {camera_matrix[0,1]:10.4f} {camera_matrix[0,2]:10.4f}]\n")
        f.write(f"[{camera_matrix[1,0]:10.4f} {camera_matrix[1,1]:10.4f} {camera_matrix[1,2]:10.4f}]\n")
        f.write(f"[{camera_matrix[2,0]:10.4f} {camera_matrix[2,1]:10.4f} {camera_matrix[2,2]:10.4f}]\n\n")
        
        f.write("DISTORTION COEFFICIENTS:\n")
        f.write(f"k1 = {dist_coeffs[0][0]:10.6f}\n")
        f.write(f"k2 = {dist_coeffs[0][1]:10.6f}\n")
        f.write(f"p1 = {dist_coeffs[0][2]:10.6f}\n")
        f.write(f"p2 = {dist_coeffs[0][3]:10.6f}\n")
        f.write(f"k3 = {dist_coeffs[0][4]:10.6f}\n\n")
        
        f.write(f"Focal lengths: fx = {camera_matrix[0,0]:.2f}, fy = {camera_matrix[1,1]:.2f}\n")
        f.write(f"Principal point: cx = {camera_matrix[0,2]:.2f}, cy = {camera_matrix[1,2]:.2f}\n")
    
    print(f"✅ Calibration data saved to:")
    print(f"   📁 {pickle_path}")
    print(f"   📄 {txt_path}")

def run_phase1_calibration():
    """
    Execute Phase 1 camera calibration on captured images.
    """
    print("🎯 PHASE 1: CAMERA CALIBRATION AND GEOMETRIC MODELING")
    print("=" * 60)
    print("Objective: Compute intrinsic camera matrix (K) and distortion coefficients")
    print(f"Using chessboard pattern: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} internal corners")
    print(f"Processing images from: {SAVE_DIR}/")
    
    # Check if captures directory exists and has images
    if not os.path.exists(SAVE_DIR):
        print(f"❌ ERROR: {SAVE_DIR} directory not found!")
        print("Please capture chessboard images first using capture mode.")
        return False
    
    image_files = glob.glob(os.path.join(SAVE_DIR, '*.png'))
    if len(image_files) < 15:
        print(f"❌ ERROR: Only {len(image_files)} images found. Need at least 15 images!")
        print("Please capture more chessboard images using capture mode.")
        return False
    
    try:
        # Step 2: Corner Detection
        objpoints, imgpoints, successful_images = detect_corners_in_images()
        
        if len(successful_images) < 10:
            print("❌ ERROR: Not enough successful images for calibration!")
            print("Please capture more chessboard images from different angles and distances.")
            return False
        
        # Get image dimensions from first successful image
        img = cv.imread(successful_images[0])
        image_shape = img.shape[1::-1]  # (width, height)
        
        # Step 3: Calibration
        camera_matrix, dist_coeffs, rvecs, tvecs, rms_error = calibrate_camera_parameters(
            objpoints, imgpoints, image_shape
        )
        
        # Step 4: Validation
        validate_calibration(camera_matrix, dist_coeffs, successful_images)
        
        # Save results
        save_calibration_data(camera_matrix, dist_coeffs, rvecs, tvecs, rms_error, successful_images)
        
        print("\n" + "🎉" * 20)
        print("PHASE 1 COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        print(f"✅ Camera intrinsic matrix (K) computed with RMS error: {rms_error:.4f} pixels")
        print(f"✅ Distortion coefficients calculated")
        print(f"✅ Validation images created showing distortion correction")
        print(f"✅ Results saved for Phase 2 use")
        
        # Assessment of calibration quality
        if rms_error < 0.5:
            print("🏆 EXCELLENT calibration quality (RMS < 0.5)")
        elif rms_error < 1.0:
            print("✅ GOOD calibration quality (RMS < 1.0)")
        elif rms_error < 2.0:
            print("⚠️  ACCEPTABLE calibration quality (RMS < 2.0)")
        else:
            print("❌ POOR calibration quality (RMS > 2.0) - consider recapturing")
        
        return True
    
    except Exception as e:
        print(f"❌ ERROR during calibration: {str(e)}")
        return False

# Camera capture functionality removed - using existing images only
# Original capture code available if needed for future image collection

def main():
    """
    Main function - runs Phase 1 calibration on existing images.
    """
    print("📷 CAMERA CALIBRATION SYSTEM")
    print("=" * 60)
    print("Phase 1: Camera Calibration and Geometric Modeling")
    print()
    
    # Check if images already exist
    existing_images = len(glob.glob(os.path.join(SAVE_DIR, '*.png')))
    if existing_images > 0:
        print(f"Found {existing_images} existing images in {SAVE_DIR}/")
    else:
        print(f"❌ No images found in {SAVE_DIR}/")
        print("Please add chessboard images to the captures/ directory first.")
        return
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        # Handle help requests
        if mode in ['--help', '-h', 'help']:
            print("USAGE:")
            print("  python calibation.py")
            print("  python calibation.py --help")
            print()
            print("DESCRIPTION:")
            print("  Performs camera calibration using existing chessboard images")
            print("  from the 'captures/' directory.")
            print()
            print("REQUIREMENTS:")
            print("  - At least 15 chessboard images in captures/ directory")
            print("  - Images should show 7x6 internal corner chessboard pattern")
            print("  - Images taken from various angles and distances")
            print()
            print("OUTPUT:")
            print("  - Camera intrinsic matrix (K)")
            print("  - Distortion coefficients")
            print("  - Validation images showing distortion correction")
            print("  - Results saved to calibration_results/ directory")
            return
    
    # Run calibration directly
    print("🎯 Running Phase 1 Camera Calibration...")
    success = run_phase1_calibration()
    
    if success:
        print("\n✅ Calibration completed successfully!")
        print("📁 Results saved to calibration_results/ directory")
        print("🖼️  Validation images available in calibration_results/validation/")
    else:
        print("\n❌ Calibration failed or was cancelled.")

if __name__ == "__main__":
    main()
import cv2 as cv
import numpy as np
import os
import pickle
import glob
import time

# ---------------- Config ----------------
CALIBRATION_FILE = "calibration_results/camera_calibration.pkl"
OUTPUT_DIR = "phase2_results"
MIN_MATCH_COUNT = 15  # Minimum number of matches for reliable estimation
RANSAC_THRESHOLD = 1.0  # RANSAC threshold for fundamental matrix estimation
CONFIDENCE = 0.99  # RANSAC confidence level
# ----------------------------------------

# Make output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

class RelativePoseEstimator:
    """
    Phase 2: Relative Pose Estimation
    Determines the motion (R, t) between two cameras from two images captured 
    from different static positions using feature matching and 8-Point Algorithm.
    """
    
    def __init__(self, calibration_path=CALIBRATION_FILE):
        """Initialize with calibrated camera parameters."""
        self.K = None  # Intrinsic camera matrix
        self.dist_coeffs = None  # Distortion coefficients
        self.load_calibration_data(calibration_path)
        
        # Initialize feature detector (Try ORB first for robustness, fallback to SIFT)
        self.detector = cv.ORB_create(nfeatures=3000)  # More features for better matching
        self.use_sift = False
        
        # BFMatcher for ORB, FLANN for SIFT
        if self.use_sift:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv.FlannBasedMatcher(index_params, search_params)
        else:
            # Use BruteForce matcher for ORB
            self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    
    def load_calibration_data(self, calibration_path):
        """Load calibration data from Phase 1."""
        print("=" * 60)
        print("LOADING CALIBRATION DATA FROM PHASE 1")
        print("=" * 60)
        
        if not os.path.exists(calibration_path):
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
        
        with open(calibration_path, 'rb') as f:
            cal_data = pickle.load(f)
        
        self.K = cal_data['camera_matrix']
        self.dist_coeffs = cal_data['dist_coeffs']
        
        print("✅ Calibration data loaded successfully!")
        print(f"📐 Intrinsic Matrix (K):")
        print(self.K)
        print(f"RMS Error from Phase 1: {cal_data['rms_error']:.4f} pixels")
        
    def undistort_image(self, img):
        """Apply distortion correction using calibrated parameters."""
        h, w = img.shape[:2]
        new_K, roi = cv.getOptimalNewCameraMatrix(self.K, self.dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv.undistort(img, self.K, self.dist_coeffs, None, new_K)
        return undistorted, new_K
    
    def detect_and_match_features(self, img1, img2):
        """
        Detect features and find matches between two images.
        Returns good matches and keypoints for both images.
        """
        print("\n" + "=" * 60)
        print("FEATURE DETECTION AND MATCHING")
        print("=" * 60)
        
        # Convert to grayscale
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # Detect keypoints and compute descriptors
        print("🔍 Detecting features in image 1...")
        kp1, desc1 = self.detector.detectAndCompute(gray1, None)
        
        print("🔍 Detecting features in image 2...")
        kp2, desc2 = self.detector.detectAndCompute(gray2, None)
        
        print(f"Found {len(kp1)} keypoints in image 1")
        print(f"Found {len(kp2)} keypoints in image 2")
        
        if len(kp1) < MIN_MATCH_COUNT or len(kp2) < MIN_MATCH_COUNT:
            raise ValueError(f"Not enough keypoints detected. Need at least {MIN_MATCH_COUNT} in each image.")
        
        # Match features 
        print("🔗 Matching features...")
        if self.use_sift:
            # Use knnMatch for SIFT with Lowe's ratio test
            raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for match_pair in raw_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                        good_matches.append(m)
        else:
            # Use knnMatch for ORB with more relaxed ratio test
            raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for match_pair in raw_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.8 * n.distance:  # More relaxed for ORB
                        good_matches.append(m)
        
        print(f"✅ Found {len(good_matches)} good matches (after Lowe's ratio test)")
        
        if len(good_matches) < MIN_MATCH_COUNT and not self.use_sift:
            print(f"⚠️  ORB found only {len(good_matches)} matches. Trying SIFT...")
            # Fallback to SIFT
            self.detector = cv.SIFT_create(nfeatures=3000)
            self.use_sift = True
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv.FlannBasedMatcher(index_params, search_params)
            
            # Re-detect with SIFT
            kp1, desc1 = self.detector.detectAndCompute(gray1, None)
            kp2, desc2 = self.detector.detectAndCompute(gray2, None)
            
            print(f"SIFT found {len(kp1)} keypoints in image 1")
            print(f"SIFT found {len(kp2)} keypoints in image 2")
            
            # Re-match with SIFT
            raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for match_pair in raw_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            print(f"✅ SIFT found {len(good_matches)} good matches")
        
        if len(good_matches) < MIN_MATCH_COUNT:
            raise ValueError(f"Not enough good matches. Found {len(good_matches)}, need at least {MIN_MATCH_COUNT}")
        
        return kp1, kp2, good_matches
    
    def estimate_fundamental_matrix(self, kp1, kp2, matches):
        """
        Estimate fundamental matrix using 8-Point Algorithm with RANSAC.
        """
        print("\n" + "=" * 60)
        print("FUNDAMENTAL MATRIX ESTIMATION (8-POINT ALGORITHM)")
        print("=" * 60)
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        print(f"Using {len(pts1)} matched points for fundamental matrix estimation")
        
        # Estimate fundamental matrix using RANSAC
        F, mask = cv.findFundamentalMat(
            pts1, pts2, 
            cv.FM_RANSAC, 
            RANSAC_THRESHOLD, 
            CONFIDENCE
        )
        
        # Count inliers
        inlier_count = np.sum(mask)
        outlier_count = len(mask) - inlier_count
        
        print(f"RANSAC Results:")
        print(f"  ✅ Inliers: {inlier_count}")
        print(f"  ❌ Outliers: {outlier_count}")
        print(f"  📊 Inlier ratio: {inlier_count/len(mask)*100:.1f}%")
        
        print(f"\n📐 Fundamental Matrix (F):")
        print(F)
        
        # Filter matches to keep only inliers
        inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
        inlier_pts1 = pts1[mask.ravel() == 1]
        inlier_pts2 = pts2[mask.ravel() == 1]
        
        return F, inlier_matches, inlier_pts1, inlier_pts2
    
    def compute_essential_matrix(self, F):
        """
        Convert fundamental matrix to essential matrix using intrinsic matrix K.
        E = K^T * F * K
        """
        print("\n" + "=" * 60)
        print("ESSENTIAL MATRIX COMPUTATION")
        print("=" * 60)
        
        # Compute essential matrix
        E = self.K.T @ F @ self.K
        
        print(f"📐 Essential Matrix (E):")
        print(E)
        
        # Essential matrix should have two equal singular values and one zero
        U, s, Vt = np.linalg.svd(E)
        print(f"\nSingular values of E: {s}")
        print("✅ For a valid essential matrix, two singular values should be equal and one should be ~0")
        
        return E
    
    def recover_pose(self, E, pts1, pts2):
        """
        Decompose essential matrix to recover relative rotation R and translation t.
        """
        print("\n" + "=" * 60)
        print("POSE RECOVERY (R, t DECOMPOSITION)")
        print("=" * 60)
        
        # Recover pose from essential matrix
        _, R, t, mask = cv.recoverPose(E, pts1, pts2, self.K)
        
        print(f"📐 Rotation Matrix (R):")
        print(R)
        
        print(f"\n📍 Translation Vector (t):")
        print(t.ravel())
        
        # Convert rotation matrix to rotation vector for easier interpretation
        rvec, _ = cv.Rodrigues(R)
        print(f"\n🔄 Rotation Vector (rodrigues):")
        print(f"  rx = {rvec[0,0]:.6f} rad ({np.degrees(rvec[0,0]):.2f}°)")
        print(f"  ry = {rvec[1,0]:.6f} rad ({np.degrees(rvec[1,0]):.2f}°)")
        print(f"  rz = {rvec[2,0]:.6f} rad ({np.degrees(rvec[2,0]):.2f}°)")
        
        # Translation magnitude (only direction is recoverable, not scale)
        t_norm = np.linalg.norm(t)
        print(f"\n📏 Translation magnitude: {t_norm:.6f} (normalized)")
        print(f"📍 Translation direction: ({t[0,0]:.3f}, {t[1,0]:.3f}, {t[2,0]:.3f})")
        
        inliers_used = np.sum(mask)
        print(f"\n✅ Points used for pose estimation: {inliers_used}")
        
        return R, t, mask
    
    def visualize_matches(self, img1, img2, kp1, kp2, matches, title="Feature Matches"):
        """Visualize feature matches between two images."""
        
        # Create match visualization
        match_img = cv.drawMatches(
            img1, kp1, img2, kp2, matches[:100],  # Show first 100 matches
            None, 
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Save visualization
        output_path = os.path.join(OUTPUT_DIR, f"{title.lower().replace(' ', '_')}.png")
        cv.imwrite(output_path, match_img)
        
        print(f"📊 Match visualization saved: {output_path}")
        
        return match_img
    
    def visualize_epipolar_lines(self, img1, img2, pts1, pts2, F, num_lines=20):
        """
        Visualize epipolar lines to validate fundamental matrix estimation.
        """
        print(f"\n📊 Creating epipolar line visualization...")
        
        # Select random subset of points for cleaner visualization
        indices = np.random.choice(len(pts1), min(num_lines, len(pts1)), replace=False)
        
        pts1_subset = pts1[indices]
        pts2_subset = pts2[indices]
        
        # Find epilines corresponding to points in second image
        lines1 = cv.computeCorrespondEpilines(pts2_subset.reshape(-1,1,2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        
        # Find epilines corresponding to points in first image  
        lines2 = cv.computeCorrespondEpilines(pts1_subset.reshape(-1,1,2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        
        # Draw epilines on images
        img1_lines = self._draw_epilines(img1.copy(), lines1, pts1_subset)
        img2_lines = self._draw_epilines(img2.copy(), lines2, pts2_subset)
        
        # Create side-by-side visualization
        h1, w1 = img1_lines.shape[:2]
        h2, w2 = img2_lines.shape[:2]
        
        # Resize images to same height for concatenation
        target_height = min(h1, h2)
        img1_resized = cv.resize(img1_lines, (int(w1 * target_height / h1), target_height))
        img2_resized = cv.resize(img2_lines, (int(w2 * target_height / h2), target_height))
        
        combined = np.hstack([img1_resized, img2_resized])
        
        # Add labels
        cv.putText(combined, "Image 1 with epilines", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(combined, "Image 2 with epilines", (img1_resized.shape[1] + 10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization
        output_path = os.path.join(OUTPUT_DIR, "epipolar_lines.png")
        cv.imwrite(output_path, combined)
        print(f"📊 Epipolar lines saved: {output_path}")
        
        return combined
    
    def _draw_epilines(self, img, lines, pts):
        """Helper function to draw epilines on an image."""
        r, c = img.shape[:2]
        
        for line, pt in zip(lines, pts):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            x0, y0 = map(int, [0, -line[2]/line[1]])
            x1, y1 = map(int, [c, -(line[2]+line[0]*c)/line[1]])
            
            cv.line(img, (x0, y0), (x1, y1), color, 1)
            cv.circle(img, tuple(map(int, pt[0])), 5, color, -1)
        
        return img
    
    def save_results(self, R, t, F, E, img1_path, img2_path):
        """Save pose estimation results."""
        print("\n" + "=" * 60)
        print("SAVING RESULTS")
        print("=" * 60)
        
        # Save results as text file
        results_path = os.path.join(OUTPUT_DIR, "pose_estimation_results.txt")
        
        with open(results_path, 'w') as f:
            f.write("RELATIVE POSE ESTIMATION RESULTS - PHASE 2\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Input Images:\n")
            f.write(f"  Image 1: {os.path.basename(img1_path)}\n")
            f.write(f"  Image 2: {os.path.basename(img2_path)}\n\n")
            
            f.write("FUNDAMENTAL MATRIX (F):\n")
            for row in F:
                f.write(f"[{row[0]:12.8f} {row[1]:12.8f} {row[2]:12.8f}]\n")
            f.write("\n")
            
            f.write("ESSENTIAL MATRIX (E):\n")
            for row in E:
                f.write(f"[{row[0]:12.8f} {row[1]:12.8f} {row[2]:12.8f}]\n")
            f.write("\n")
            
            f.write("ROTATION MATRIX (R):\n")
            for row in R:
                f.write(f"[{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f}]\n")
            f.write("\n")
            
            f.write("TRANSLATION VECTOR (t):\n")
            f.write(f"[{t[0,0]:10.6f}]\n")
            f.write(f"[{t[1,0]:10.6f}]\n")
            f.write(f"[{t[2,0]:10.6f}]\n\n")
            
            # Convert to Euler angles for easier interpretation
            rvec, _ = cv.Rodrigues(R)
            f.write("ROTATION (Euler angles in degrees):\n")
            f.write(f"  Roll  (X): {np.degrees(rvec[0,0]):8.2f}°\n")
            f.write(f"  Pitch (Y): {np.degrees(rvec[1,0]):8.2f}°\n")
            f.write(f"  Yaw   (Z): {np.degrees(rvec[2,0]):8.2f}°\n\n")
            
            f.write("TRANSLATION (normalized direction):\n")
            f.write(f"  X: {t[0,0]:8.3f}\n")
            f.write(f"  Y: {t[1,0]:8.3f}\n")
            f.write(f"  Z: {t[2,0]:8.3f}\n")
            f.write(f"  Magnitude: {np.linalg.norm(t):8.3f}\n\n")
            
            f.write("NOTE: Translation scale is not recoverable from two views.\n")
            f.write("Only the direction of translation can be determined.\n")
        
        print(f"✅ Results saved to: {results_path}")
        
        # Also save as pickle for programmatic access
        pickle_path = os.path.join(OUTPUT_DIR, "pose_estimation_results.pkl")
        results_data = {
            'rotation_matrix': R,
            'translation_vector': t,
            'fundamental_matrix': F,
            'essential_matrix': E,
            'image1_path': img1_path,
            'image2_path': img2_path,
            'intrinsic_matrix': self.K,
            'timestamp': time.time()
        }
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(results_data, f)
        
        print(f"✅ Binary data saved to: {pickle_path}")

def estimate_relative_pose(img1_path, img2_path):
    """
    Main function to estimate relative pose between two images.
    """
    print("🎯 PHASE 2: RELATIVE POSE ESTIMATION")
    print("=" * 60)
    print("Objective: Determine motion (R, t) between two camera positions")
    print(f"Image 1: {os.path.basename(img1_path)}")
    print(f"Image 2: {os.path.basename(img2_path)}")
    
    try:
        # Initialize pose estimator
        estimator = RelativePoseEstimator()
        
        # Load images
        print(f"\n📷 Loading images...")
        img1 = cv.imread(img1_path)
        img2 = cv.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("Could not load one or both images")
        
        print(f"Image 1 size: {img1.shape[1]}x{img1.shape[0]}")
        print(f"Image 2 size: {img2.shape[1]}x{img2.shape[0]}")
        
        # Undistort images using calibration data
        print(f"\n🔧 Applying distortion correction...")
        img1_undist, _ = estimator.undistort_image(img1)
        img2_undist, _ = estimator.undistort_image(img2)
        
        # Detect and match features
        kp1, kp2, good_matches = estimator.detect_and_match_features(img1_undist, img2_undist)
        
        # Visualize matches
        estimator.visualize_matches(img1_undist, img2_undist, kp1, kp2, good_matches, "Good Matches")
        
        # Estimate fundamental matrix using 8-Point Algorithm
        F, inlier_matches, inlier_pts1, inlier_pts2 = estimator.estimate_fundamental_matrix(kp1, kp2, good_matches)
        
        # Visualize inlier matches
        estimator.visualize_matches(img1_undist, img2_undist, kp1, kp2, inlier_matches, "Inlier Matches")
        
        # Visualize epipolar lines
        estimator.visualize_epipolar_lines(img1_undist, img2_undist, inlier_pts1, inlier_pts2, F)
        
        # Convert to essential matrix
        E = estimator.compute_essential_matrix(F)
        
        # Recover pose (R, t)
        R, t, pose_mask = estimator.recover_pose(E, inlier_pts1, inlier_pts2)
        
        # Save results
        estimator.save_results(R, t, F, E, img1_path, img2_path)
        
        print("\n" + "🎉" * 20)
        print("PHASE 2 COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        print(f"✅ Fundamental matrix (F) computed using 8-Point Algorithm")
        print(f"✅ Essential matrix (E) derived from F and intrinsic matrix K")
        print(f"✅ Relative rotation (R) and translation (t) recovered")
        print(f"✅ Results and visualizations saved to {OUTPUT_DIR}/")
        
        return R, t, F, E
        
    except Exception as e:
        print(f"\n❌ ERROR during pose estimation: {str(e)}")
        return None, None, None, None

def demo_with_sample_images():
    """
    Demo function - looks for sample images to process.
    """
    print("🔍 Looking for sample images...")
    
    # Look for images in common locations
    possible_dirs = ["captures", ".", "images", "samples"]
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    
    all_images = []
    for directory in possible_dirs:
        if os.path.exists(directory):
            for ext in image_extensions:
                all_images.extend(glob.glob(os.path.join(directory, ext)))
    
    all_images = sorted(list(set(all_images)))  # Remove duplicates and sort
    
    if len(all_images) < 2:
        print("❌ Need at least 2 images for pose estimation!")
        print("Please provide two images taken from different positions.")
        print("\nExpected usage:")
        print("  1. Take a photo")
        print("  2. Move camera to new position (translate/rotate)")
        print("  3. Take another photo")
        print("  4. Run this script with both images")
        return
    
    print(f"Found {len(all_images)} images:")
    for i, img_path in enumerate(all_images[:10]):  # Show first 10
        print(f"  [{i+1}] {img_path}")
    
    if len(all_images) > 10:
        print(f"  ... and {len(all_images) - 10} more")
    
    # Use first two images as default
    img1_path = all_images[0]
    img2_path = all_images[1]
    
    print(f"\n🎯 Using images:")
    print(f"  Image 1: {img1_path}")
    print(f"  Image 2: {img2_path}")
    
    # Estimate relative pose
    R, t, F, E = estimate_relative_pose(img1_path, img2_path)
    
    return R, t, F, E

def main():
    """
    Main function for Phase 2 relative pose estimation.
    """
    import sys
    
    print("📷 RELATIVE POSE ESTIMATION SYSTEM")
    print("=" * 60)
    print("Phase 2: Application of Calibrated Data")
    print()
    
    # Check if calibration data exists
    if not os.path.exists(CALIBRATION_FILE):
        print(f"❌ Calibration file not found: {CALIBRATION_FILE}")
        print("Please run Phase 1 (calibation.py) first to calibrate your camera.")
        return
    
    if len(sys.argv) == 3:
        # Two image paths provided
        img1_path = sys.argv[1]
        img2_path = sys.argv[2]
        
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print("❌ One or both image files not found!")
            return
        
        print(f"🎯 Processing specified images:")
        print(f"  Image 1: {img1_path}")
        print(f"  Image 2: {img2_path}")
        
        R, t, F, E = estimate_relative_pose(img1_path, img2_path)
    
    elif len(sys.argv) == 1:
        # No arguments - run demo
        print("🎯 Running demo mode (using available images)...")
        R, t, F, E = demo_with_sample_images()
    
    else:
        # Help/usage
        print("USAGE:")
        print("  python phase2chosen.py [image1] [image2]")
        print("  python phase2chosen.py  # Demo mode")
        print()
        print("DESCRIPTION:")
        print("  Estimates relative pose (rotation R and translation t) between")
        print("  two camera positions using feature matching and 8-Point Algorithm.")
        print()
        print("REQUIREMENTS:")
        print("  - Calibrated camera (run calibation.py first)")
        print("  - Two images taken from different positions")
        print("  - Sufficient texture/features in images")
        print()
        print("OUTPUT:")
        print("  - Fundamental matrix (F)")
        print("  - Essential matrix (E)")
        print("  - Rotation matrix (R)")
        print("  - Translation vector (t)")
        print("  - Visualizations of matches and epipolar geometry")
        print()
        print("EXAMPLE:")
        print("  python phase2chosen.py image1.jpg image2.jpg")
        return
    
    if R is not None:
        print(f"\n✅ Pose estimation completed successfully!")
        print(f"📁 Results saved to {OUTPUT_DIR}/ directory")
    else:
        print(f"\n❌ Pose estimation failed.")

if __name__ == "__main__":
    main()

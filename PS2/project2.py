"""
Omar Khader's PS2 --- Where is Rocky?
Finding Rocky using SIFT keypoint matching with manual implementation
"""

import numpy as np
import cv2
import sys
import time
import select
import os
import random

np.random.seed(42) # set random seed so results are consistent
random.seed(42)

# set up SIFT detector let's grab more features for better matching
sift = cv2.SIFT_create(nfeatures=3000)  # we want lots of keypoints to work with

def detect_and_describe(image, focus_on_rocky=False):
    """
    Detect SIFT keypoints and compute descriptors for an image
    We're keeping it simple - no weird masking that might miss stuff
    """
    # convert to grayscale cuz SIFT likes that
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply some contrast enhancement to help SIFT find more features
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # now let SIFT do its thing on the whole image
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # return keypoints, descriptors, metric type, and the preprocessed image SIFT sees
    return keypoints, descriptors, "L2", gray

def manual_match_features(desc1, desc2, ratio_threshold=0.72):
    
    #Manually match SIFT descriptors using ratio test
    #We're doing this ourselves instead of using OpenCV's matching functions
    
    # first check if we have descriptors to work with
    if desc1 is None or desc2 is None:
        return []

    # let's store our good matches here
    matches = []

    # we want to loop through each descriptor in the reference image
    for i in range(len(desc1)):
        # calculate distances to all descriptors in test image
        # using vectorized operations for speed
        dists = np.sum((desc2 - desc1[i])**2, axis=1)  # squared euclidean distances

        # find two best matches
        sorted_indices = np.argpartition(dists, 2)[:2]  # get indices of 2 smallest
        sorted_dists = dists[sorted_indices]
        sorted_dists.sort()  # ensure they're sorted

        # apply ratio test
        if len(sorted_dists) >= 2 and sorted_dists[1] > 0:
            ratio = np.sqrt(sorted_dists[0]) / np.sqrt(sorted_dists[1])  # use actual distances for ratio
            if ratio < ratio_threshold:
                best_idx = np.argmin(dists)  # index of best match
                matches.append((i, best_idx, np.sqrt(sorted_dists[0])))  # store match

    # return these matches we found
    return matches

def find_transformation_ransac(kp1, kp2, matches, num_iterations=2000, threshold=4.0):
    """
    Find affine transformation using RANSAC
    Let's try a bunch of random samples to find the best transformation
    """
    # need at least 3 matches to compute a transformation
    if len(matches) < 3:
        return None, []

    best_transform = None
    best_inliers = []

    # convert matches to point arrays
    src_pts = np.float32([kp1[m[0]].pt for m in matches])
    dst_pts = np.float32([kp2[m[1]].pt for m in matches])

    # rANSAC iterations - let's try a bunch of times
    for _ in range(num_iterations):
        # randomly select 3 matches
        idx = np.random.choice(len(matches), 3, replace=False)

        # get affine transformation
        try:
            M = cv2.getAffineTransform(src_pts[idx], dst_pts[idx])
        except:
            continue

        # test all matches - vectorized for speed
        src_homo = np.column_stack([src_pts, np.ones(len(src_pts))])  # homogeneous coords
        dst_pred = (M @ src_homo.T).T  # transform all at once

        # calculate errors
        errors = np.sqrt(np.sum((dst_pred - dst_pts)**2, axis=1))  # euclidean distances

        # find inliers
        inliers = np.where(errors < threshold)[0]

        # update best if more inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_transform = M

    # return the best transformation we found
    return best_transform, best_inliers

def compute_bounding_box(ref_shape, transform_matrix):
    
    #Compute oriented bounding box parameters from transformation
    #Let's figure out where Rocky is and how big he appears
    
    h, w = ref_shape

    # define corners of reference image
    corners = np.array([
        [0, 0, 1],      # top-left
        [w-1, 0, 1],    # top-right
        [w-1, h-1, 1],  # bottom-right
        [0, h-1, 1]     # bottom-left
    ]).T

    # transform corners
    transformed_corners = (transform_matrix @ corners).T

    # find center
    center_x = np.mean(transformed_corners[:, 0])
    center_y = np.mean(transformed_corners[:, 1])

    # calculate height
    top_center = (transformed_corners[0] + transformed_corners[1]) / 2
    bottom_center = (transformed_corners[2] + transformed_corners[3]) / 2
    height = np.sqrt(np.sum((bottom_center - top_center)**2))

    # calculate angle (from vertical, clockwise positive)
    # vector from top to bottom center
    dx = bottom_center[0] - top_center[0]
    dy = bottom_center[1] - top_center[1]

    # angle from vertical (positive y-axis points down in image coordinates)
    # atan2(dx, dy) gives angle from vertical
    angle_rad = np.arctan2(dx, dy)
    angle_deg = np.degrees(angle_rad)

    # normalize angle to [0, 360)
    angle_deg = angle_deg % 360

    # return these calculated values
    return int(center_x), int(center_y), int(height), int(angle_deg)

def find_rocky(reference_img, test_img):
    
    #Main function to find Rocky in a test image
    #Let's put it all together and locate Rocky
    
    print("Analyzing images for Rocky detection...")

    # detect keypoints and descriptors
    print("Extracting keypoints from reference image...")
    ref_kp, ref_desc, ref_metric, ref_gray = detect_and_describe(reference_img)
    print(f"   Found {len(ref_kp)} keypoints in reference")

    print("Extracting keypoints from test image...")
    test_kp, test_desc, test_metric, test_gray = detect_and_describe(test_img)
    print(f"   Found {len(test_kp)} keypoints in test image")

    # check if enough keypoints
    if ref_desc is None or test_desc is None or len(ref_desc) < 10 or len(test_desc) < 10:
        print("WARNING: Not enough keypoints for matching")
        return None

    # match features
    print("Matching features between images...")
    matches = manual_match_features(ref_desc, test_desc)
    print(f"   Found {len(matches)} good matches")

    # check if enough matches
    if len(matches) < 8:
        print("WARNING: Not enough matches for transformation")
        return None

    # find transformation
    print("Computing transformation using RANSAC...")
    transform, inliers = find_transformation_ransac(ref_kp, test_kp, matches)
    print(f"   Found transformation with {len(inliers)} inliers")

    # check if valid transformation
    if transform is None or len(inliers) < 5:
        print("WARNING: Invalid transformation found")
        return None

    # compute bounding box
    print("Computing bounding box...")
    bbox = compute_bounding_box(reference_img.shape[:2], transform)

    print("SUCCESS: Rocky detection completed successfully!")
    return bbox

def load_ground_truth(txt_path):
    # Load ground truth from txt file
    try:
        with open(txt_path, 'r') as f:
            line = f.readline().strip()
            if line:
                values = line.split()
                return int(values[0]), int(values[1]), int(values[2]), int(values[3])
    except:
        pass
    return None

def calculate_iou(img_size, cx1, cy1, h1, a1, cx2, cy2, h2, a2):

    # convert angles to radians
    a1_rad = np.radians(a1)
    a2_rad = np.radians(a2)

    # width is 60% of height
    w1 = 0.6 * h1
    w2 = 0.6 * h2

    # function to get corners of oriented rectangle
    def get_corners(cx, cy, w, h, angle):
        corners = np.array([
            [-w/2, -h/2],
            [w/2, -h/2],
            [w/2, h/2],
            [-w/2, h/2]
        ])

        # rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # rotate and translate
        rotated = corners @ rot.T
        translated = rotated + np.array([cx, cy])

        return translated

    # get corners of both boxes
    corners1 = get_corners(cx1, cy1, w1, h1, a1_rad)
    corners2 = get_corners(cx2, cy2, w2, h2, a2_rad)

    # create masks for both boxes
    mask1 = np.zeros(img_size, dtype=np.uint8)
    mask2 = np.zeros(img_size, dtype=np.uint8)

    # fill polygons
    cv2.fillPoly(mask1, [corners1.astype(np.int32)], 255)
    cv2.fillPoly(mask2, [corners2.astype(np.int32)], 255)

    # calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # calculate IoU
    if union == 0:
        return 0.0

    iou = intersection / union
    return iou

def test_all_images():
    
    #Test Rocky detection on all test images and calculate average IoU
    
    print("Starting comprehensive test of all images...")

    # load reference image once
    print("Loading reference image...")
    reference_img = cv2.imread('tests/reference.png', cv2.IMREAD_COLOR)
    if reference_img is None:
        print("ERROR: Failed to load reference image!")
        return

    # find all test images
    test_images = []
    for i in range(1, 15):  # check images 1-14
        img_path = f'tests/{i}.png'
        txt_path = f'tests/{i}.txt'
        if os.path.exists(img_path) and os.path.exists(txt_path):
            test_images.append(i)

    print(f"Found {len(test_images)} test images to process")

    # store results
    results = []
    ious = []

    # process each test image
    for img_num in test_images:
        print(f"\n--- Processing image {img_num} ---")

        # load test image
        test_path = f'tests/{img_num}.png'
        test_img = cv2.imread(test_path, cv2.IMREAD_COLOR)
        if test_img is None:
            print(f"ERROR: Failed to load test image {img_num}")
            continue

        # load ground truth
        gt_path = f'tests/{img_num}.txt'
        gt = load_ground_truth(gt_path)
        if gt is None:
            print(f"ERROR: Failed to load ground truth for image {img_num}")
            continue

        # find Rocky
        start_time = time.time()
        result = find_rocky(reference_img, test_img)
        detection_time = time.time() - start_time

        # calculate IoU if found
        if result:
            iou = calculate_iou((1500, 2000), result[0], result[1], result[2], result[3],
                               gt[0], gt[1], gt[2], gt[3])
            ious.append(iou)

            print(f"SUCCESS: Detection: X={result[0]}, Y={result[1]}, H={result[2]}, A={result[3]}")
            print(f"Ground truth: X={gt[0]}, Y={gt[1]}, H={gt[2]}, A={gt[3]}")
            print(f"IoU: {iou:.3f} (Time: {detection_time:.2f}s)")

            results.append({
                'image': img_num,
                'result': result,
                'ground_truth': gt,
                'iou': iou,
                'time': detection_time
            })
        else:
            print(f"FAILED: Failed to detect Rocky in image {img_num}")
            ious.append(0.0)

    # calculate average IoU
    if ious:
        avg_iou = np.mean(ious)
        print("\n=== FINAL RESULTS ===")
        print(f"Average IoU: {avg_iou:.3f}")
        print(f"Successfully detected: {len([i for i in ious if i > 0])}/{len(test_images)}")

        # show grade based on average IoU for testing purposes 
        if avg_iou >= 0.7:
            print("Grade: 90/90 points (Full credit!)")
        elif avg_iou >= 0.5:
            print("Grade: 75/90 points")
        elif avg_iou >= 0.2:
            print("Grade: 45/90 points")
        else:
            print("Grade: 0/90 points")
    else:
        print("No successful detections")

    return results, avg_iou if ious else 0.0

def main():
    
    #Main entry point for grading
    #Let's load images and find Rocky with progress tracking
    
    print("Starting Rocky detection pipeline...")
    start_time = time.time()

    # load reference image
    print("Loading reference image...")
    reference_img = cv2.imread('tests/reference.png', cv2.IMREAD_COLOR)
    if reference_img is None:
        print("ERROR: Failed to load reference image!")
        return
    print(f"Reference image loaded: {reference_img.shape}")

    # check if user wants to run all tests or just one image
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # run all test images
        print("Running comprehensive test on all images...")
        results, avg_iou = test_all_images()
        return
    elif len(sys.argv) > 1:
        # for testing with command line argument
        test_filename = sys.argv[1]
        print(f"Command line argument: {test_filename}")
    else:
        # for grading system - handle non-interactive environments
        import threading

        test_filename = None
        input_timeout = False

        def get_input():
            nonlocal test_filename
            try:
                test_filename = input().rstrip()
            except EOFError:
                pass

        # start input thread with timeout
        input_thread = threading.Thread(target=get_input)
        input_thread.daemon = True
        input_thread.start()
        input_thread.join(timeout=0.5)  # 0.5 second timeout

        if test_filename is None:
            # no input received within timeout - run all tests
            print("No input available - running comprehensive test on all images...")
            results, avg_iou = test_all_images()
            return

    # single image processing
    # ensure test image path includes tests/ directory
    if not test_filename.startswith('tests/'):
        test_filename = f'tests/{test_filename}'

    # load test image
    print(f"Loading test image: {test_filename}")
    test_img = cv2.imread(test_filename, cv2.IMREAD_COLOR)
    if test_img is None:
        print(f"ERROR: Failed to load test image: {test_filename}")
        return
    print(f"Test image loaded: {test_img.shape}")

    # find Rocky
    print("Starting Rocky detection...")
    detection_start = time.time()
    result = find_rocky(reference_img, test_img)
    detection_time = time.time() - detection_start

    total_time = time.time() - start_time

    if result:
        # print result in required format
        print("SUCCESS: Rocky found!")
        print(f"Location: X={result[0]}, Y={result[1]}, H={result[2]}, A={result[3]}")
        print(f"Detection time: {detection_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")
        print(f"{result[0]} {result[1]} {result[2]} {result[3]}")
    else:
        # if not found, output center of image with default values
        print("WARNING: Rocky not found - using default center position")
        print(f"Detection time: {detection_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")


    print("Pipeline completed!")

if __name__ == "__main__":
    main()


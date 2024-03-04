import cv2
import numpy as np


def detect_and_compute_keypoints(image, detector_type='ORB'):
    """
    Detect keypoints and compute descriptors for an image

    Args:
    - image: Input image
    - detector_type: Type of feature detector to use (default is 'ORB')

    Returns:
    - keypoints: Detected keypoints
    - descriptors: Computed descriptors
    """
    # Select the appropriate feature detector based on the provided type
    if detector_type == 'ORB':
        detector = cv2.ORB_create()
    elif detector_type == 'SIFT':
        detector = cv2.SIFT_create()
    elif detector_type == 'SURF':
        detector = cv2.SURF_create()
    elif detector_type == 'FAST':
        detector = cv2.FastFeatureDetector_create()
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")

    # Detect keypoints and compute descriptors
    keypoints, descriptors = detector.detectAndCompute(image, None)

    return keypoints, descriptors


def match_keypoints(descriptor1, descriptor2, matcher_type='BF', distance_metric='HAMMING', return_keypoints=False):
    """
    Match keypoints between two images using specified matcher type and distance metric.

    Args:
    - descriptor1: Descriptors of keypoints from the first image
    - descriptor2: Descriptors of keypoints from the second image
    - matcher_type: Type of matching algorithm to use ('BF' for Brute-Force, 'FLANN' for FLANN-based matcher)
    - distance_metric: Distance metric to use for matching ('HAMMING' for binary descriptors, 'L2' for floating-point descriptors)
    - return_keypoints: Boolean flag indicating whether to return the matched keypoints

    Returns:
    - matches: List of matched keypoints (if return_keypoints=True)
    - num_matches: Number of matched keypoints (if return_keypoints=False)
    """
    # Validate input matcher_type and distance_metric
    if matcher_type not in ['BF', 'FLANN']:
        raise ValueError("Invalid matcher_type. Use 'BF' for Brute-Force or 'FLANN' for FLANN-based matcher.")

    if distance_metric not in ['HAMMING', 'L2']:
        raise ValueError("Invalid distance_metric. Use 'HAMMING' for binary descriptors or 'L2' for floating-point descriptors.")

    # Select the appropriate matcher object based on the provided type
    if matcher_type == 'BF':
        # Brute-Force matcher is generally suitable for binary and floating-point descriptors
        if distance_metric == 'HAMMING':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:  # FLANN matcher
        if distance_metric == 'HAMMING':
            flann_params = dict(algorithm=0, trees=5)
        else:  # L2 distance
            flann_params = dict(algorithm=1, trees=5)
        matcher = cv2.FlannBasedMatcher(flann_params, {})

    # Perform keypoint matching
    matches = matcher.match(descriptor1, descriptor2)

    if return_keypoints:
        # Sort matches based on their distances
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    else:
        # Return the number of matches
        return len(matches)


def find_matching_keypoints(descriptors_list, calculation_window_size=20, return_keypoints=False):
    """
    Find the number of matching keypoints between pairs of images in the input list
    within (window_size // 2) frames before and after the current frame.

    :param descriptors_list: A list containing descriptors for each image.
    :param calculation_window_size: If window_size > 0, only consider matches between images
                        that are within window_size of each other. 0 matches all images.
    :return: A 2D NumPy array containing the number of matching keypoints between
             pairs of images.
    :param return_keypoints: Boolean flag indicating whether to return the matched
                             keypoints
    """
    assert calculation_window_size >= 0, "Window size must be non-negative"
    num_images = len(descriptors_list)
    matching_keypoints = np.zeros((num_images, num_images), dtype=int)

    if calculation_window_size == 0:
        for i in range(num_images):
            for j in range(i + 1, num_images):
                matching_keypoints[i][j] = match_keypoints(descriptors_list[i], descriptors_list[j], return_keypoints=return_keypoints)
    else:
        for i in range(num_images):
            for j in range(max(0, i - calculation_window_size // 2), min(num_images, i + calculation_window_size // 2 + 1)):
                if i != j:
                    matching_keypoints[i][j] = match_keypoints(descriptors_list[i], descriptors_list[j], return_keypoints=return_keypoints)

    matching_keypoints += matching_keypoints.T
    return matching_keypoints


def maximum_average_with_max_dist(series: list, max_dist: int, ) -> list:
    """
    Given a series of numbers, find the subsequence of length at most max_dist that has the maximum average.
    :param series: List of strictly non-negative numbers
    :param max_dist: Maximum distance between two elements in the subsequence (e.g. index 1 and 3 are 2 apart)
    :return: List of indices of the subsequence with the maximum average
    """

    n = len(series) + 1

    # In best_sum[i], we store the best *best_sum* required to go to this index i.
    best_sum = [0] * n
    best_sum[0] = series[0]  # Mandatory to start from index i.
    # index_table[i] keeps track of the indices selected to get to index i
    index_table = [[0] for _ in range(n)]

    for i in range(1, max_dist + 1):
        current_best_avg = best_sum[i - 1]
        best_sum[i] = best_sum[i - 1]
        index_table[i] = index_table[i - 1]
        for j in range(i - 1, -1, -1):
            new_avg = (best_sum[j] + series[j]) / (len(index_table[j]) + 1)
            if new_avg > current_best_avg:
                current_best_avg = new_avg
                index_table[i] = index_table[j] + [j]
                best_sum[i] = best_sum[j] + series[j]

    for i in range(max_dist + 1, n):
        current_best_avg = -1
        for j in range(i - 1, i - max_dist - 1, -1):
            new_avg = (best_sum[j] + series[j]) / (len(index_table[j]) + 1)
            if new_avg > current_best_avg:
                current_best_avg = new_avg
                index_table[i] = index_table[j] + [j]
                best_sum[i] = best_sum[j] + series[j]

    for i in range(n):
        print(f"Index: {i:>2}, Average: {best_sum[i] / len(index_table[i]):>4.2f}, Indices Accessed: {index_table[i]}")
    return index_table[-1]

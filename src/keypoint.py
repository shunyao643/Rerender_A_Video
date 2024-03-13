import cv2
import numpy as np
import os
import time


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
    _start_time = time.time()
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

    _end_time = time.time()
    print(f"[TIME] (detect_and_compute_keypoints): {_end_time - _start_time:.4f} seconds")

    return keypoints, descriptors


def match_keypoints(descriptor1, descriptor2, matcher_type='BF', distance_metric='HAMMING', return_keypoints=False):
    """
    Match keypoints between two images using specified matcher type and distance metric.

    Args:
    - descriptor1: Descriptors of keypoints from the first image
    - descriptor2: Descriptors of keypoints from the second image
    - matcher_type: Type of matching algorithm to use ('BF' for Brute-Force, 'FLANN' for FLANN-based matcher)
    - distance_metric: Distance metric to use for matching ('HAMMING' for binary descriptors,
    'L2' for floating-point descriptors)
    - return_keypoints: Boolean flag indicating whether to return the matched keypoints

    Returns:
    - matches: List of matched keypoints (if return_keypoints=True)
    - num_matches: Number of matched keypoints (if return_keypoints=False)
    """
    _start_time = time.time()
    # Validate input matcher_type and distance_metric
    if matcher_type not in ['BF', 'FLANN']:
        raise ValueError("Invalid matcher_type. Use 'BF' for Brute-Force or 'FLANN' for FLANN-based matcher.")

    if distance_metric not in ['HAMMING', 'L2']:
        raise ValueError("Invalid distance_metric. Use 'HAMMING' for binary descriptors or 'L2' for floating-point "
                         "descriptors.")

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

    _end_time = time.time()
    print(f"[TIME] (match_keypoints): {_end_time - _start_time:.4f} seconds")

    if return_keypoints:
        # Sort matches based on their distances
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    else:
        # Return the number of matches
        return len(matches)


def find_matching_keypoints(descriptors_list: list, calculation_window_size=20,
                            matcher_type='BF', distance_metric='HAMMING', return_keypoints=False) -> list:
    """
    Find the number of matching keypoints between pairs of images in the input list
    within (window_size // 2) frames before and after the current frame.

    :param distance_metric: Distance metric to use for matching ('HAMMING' for binary descriptors,
    'L2' for floating-point descriptors)
    :param matcher_type: Type of matching algorithm to use ('BF' for Brute-Force, 'FLANN' for FLANN-based matcher)
    :param descriptors_list: A list containing descriptors for each image.
    :param calculation_window_size: If window_size > 0, only consider matches between images
                        that are within window_size of each other. 0 matches all images.
    :return: A 2D NumPy array containing the number of matching keypoints between
             pairs of images.
    :param return_keypoints: Boolean flag indicating whether to return the matched
                             keypoints
    """
    _start_time = time.time()
    assert calculation_window_size >= 0, "Window size must be non-negative"
    num_images = len(descriptors_list)
    matching_keypoints = np.zeros((num_images, num_images), dtype=int)

    if calculation_window_size == 0:
        for i in range(num_images):
            for j in range(i + 1, num_images):
                matching_keypoints[i][j] = match_keypoints(descriptors_list[i], descriptors_list[j],
                                                           matcher_type=matcher_type, distance_metric=distance_metric,
                                                           return_keypoints=return_keypoints)
    else:
        for i in range(num_images):
            for j in range(max(0, i - calculation_window_size // 2),
                           min(num_images, i + calculation_window_size // 2 + 1)):
                if i != j:
                    matching_keypoints[i][j] = match_keypoints(descriptors_list[i], descriptors_list[j],
                                                               matcher_type=matcher_type,
                                                               distance_metric=distance_metric,
                                                               return_keypoints=return_keypoints)

    matching_keypoints += matching_keypoints.T
    _end_time = time.time()
    print(f"[TIME] (find_matching_keypoints): {_end_time - _start_time:.4f} seconds")
    return matching_keypoints


def maximum_per_bucket(descriptors_list: list, bucket_size: int = 10) -> list:
    """
    Extracts frames indices with the most matching keypoints from each bucket
    :param descriptors_list: A list containing descriptors for each image.
    :param bucket_size: Number of frames per bucket
    :return: List of indices of the selected frames
    """
    _start_time = time.time()
    maximum_indices = []
    for i in range(0, len(descriptors_list), bucket_size):
        bucket = descriptors_list[i:i + bucket_size]
        max_index = max(enumerate(bucket), key=lambda x: x[1])[0]
        maximum_indices.append(i + max_index)
    _end_time = time.time()
    print(f"[TIME] (maximum_per_bucket): {_end_time - _start_time:.4f} seconds")
    return maximum_indices


def maximum_average_with_max_dist(series: list, max_dist: int) -> list:
    """
    Given a series of numbers, find the subsequence of length at most max_dist that has the maximum average.
    :param series: List of strictly non-negative numbers
    :param max_dist: Maximum distance between two elements in the subsequence (e.g. index 1 and 3 are 2 apart).
    If max_dist = 0, the entire series is considered: returns only indices of maximum value.
    :return: List of indices of the subsequence with the maximum average
    """
    _start_time = time.time()
    assert len(series) > 0, "Series must be non-empty"
    if max_dist == 0:
        print(f"Warning: Maximum distance is 0. Setting max_dist to {len(series)}.")
        max_value = max(series)
        indices = [index for index, value in enumerate(series) if value == max_value]
        return indices
    if max_dist > len(series):
        print(f"Warning: Maximum distance {max_dist} is greater than the length of the series {len(series)}. "
              f"Setting max_dist to {len(series)}.")
        max_dist = len(series)

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
    _end_time = time.time()
    print(f"[TIME] (maximum_average_with_max_dist): {_end_time - _start_time:.4f} seconds")
    return index_table[-1]


def select_frames_using_keypoints(input_dir: str, max_dist: int, window_size: int,
                                  matcher_type: str = 'BF', distance_metric: str = 'HAMMING',
                                  detector_type: str = 'ORB') -> list:
    """
    Select frames from a directory of images based on the number of matching keypoints.
    :param input_dir: Directory containing images
    :param max_dist: Maximum distance between two frames in the subsequence
    :param window_size: If window_size > 0, only consider matches between images that are within window_size of each
    other. 0 matches all images.
    :param matcher_type: Type of matching algorithm to use ('BF' for Brute-Force, 'FLANN' for
    FLANN-based matcher)
    :param distance_metric: Distance metric to use for matching ('HAMMING' for binary descriptors,
    'L2' for floating-point descriptors)
    :param detector_type: Type of feature detector to use ('ORB', 'SIFT', 'SURF',
    'FAST')
    :return: List of indices of the selected frames
    """
    # Read images from the input directory
    _start_time = time.time()
    image_paths = [f"{input_dir}/{filename}" for filename in sorted(os.listdir(input_dir))]

    # Detect and compute keypoints for each image
    keypoints_list = []
    descriptors_list = []
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = detect_and_compute_keypoints(image, detector_type=detector_type)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    # Find the number of matching keypoints between pairs of images
    matching_keypoints = find_matching_keypoints(descriptors_list, window_size,
                                                 matcher_type=matcher_type, distance_metric=distance_metric)
    _end_time_1 = time.time()
    print(f"[TIME] (select_frames_using_keypoints): {_end_time_1 - _start_time:.4f} seconds")

    # Find the subsequence of frames with the maximum average number of matching keypoints
    selected_frames = maximum_average_with_max_dist(np.sum(matching_keypoints, axis=0), max_dist)

    _end_time_2 = time.time()
    print(f"[TIME] (TOTAL run time - maximum_average_with_max_dist): {_end_time_2 - _start_time:.4f} seconds")

    selected_frames = maximum_per_bucket(np.sum(matching_keypoints, axis=0), bucket_size=max_dist)

    _end_time_3 = time.time()
    print(f"[TIME] (TOTAL run time - maximum_per_bucket           ):"
          f"{(_end_time_3 - _end_time_2) + (_end_time_1 - _start_time):.4f} seconds")

    return selected_frames

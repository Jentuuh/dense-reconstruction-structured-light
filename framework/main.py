import enum
from typing import List

import utils
import numpy as np
import cv2
from codec import Encoder
from PhaseShift2x3Codec import PhaseShift2x3Encoder, PhaseShift2x3Decoder
from GrayCodeCodec import GrayCodeEncoder
import math
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import csv
import sys
from pyrr import Quaternion, Matrix33

np.set_printoptions(threshold=sys.maxsize)

SHOW_INTERMEDIATE_OUTPUTS = False
NEIGHBORHOOD_METRIC = True
GRAYCODE_VERSION = False
TRIANGULATE_POINTS_MANUALLY = False

def load_views(view_paths: List[str]) -> List[np.array]:
    return [utils.load_images(view) for view in view_paths]


def show_graycode_captures():
    views = [
        "../dataset/GrayCodes/graycodes_view0.xml",
        "../dataset/GrayCodes/graycodes_view1.xml"
    ]
    left_view, right_view = load_views(views)
    show_captures(left_view, right_view)


def show_sine_captures():
    views = [
        "../dataset/Sinus/sinus_view0.xml",
        "../dataset/Sinus/sinus_view1.xml"
    ]
    left_view, right_view = load_views(views)
    show_captures(left_view, right_view)


def show_captures(left_view: List[np.array], right_view: List[np.array]):
    window_size = (int(left_view[0].shape[1]/2), int(left_view[0].shape[0]/2))
    left_window_name = "Left view"
    right_window_name = "Right view"

    cv2.namedWindow(left_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(left_window_name,
                     width=window_size[0], height=window_size[1])
    cv2.namedWindow(right_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(right_window_name,
                     width=window_size[0], height=window_size[1])

    for image_left,  image_right in zip(left_view, right_view):
        cv2.imshow(left_window_name, image_left)
        cv2.imshow(right_window_name, image_right)

        key = cv2.waitKey(1000)
        if key == 27:
            break

    cv2.destroyAllWindows()


def show_graycode_patterns():
    encoder = GrayCodeEncoder(1080, 1920, 10)
    show_encoder_patterns(encoder)


def show_sine_patterns():
    encoder = PhaseShift2x3Encoder(1080, 1920)
    show_encoder_patterns(encoder)


def show_encoder_patterns(encoder: Encoder):
    window_name = "Pattern"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(encoder.rows / 2), int(encoder.cols / 2))
    for n in range(encoder.n):
        pattern = encoder.get_encoding_pattern(n)

        dummy = np.copy(pattern)
        pattern = cv2.resize(dummy, (encoder.rows, encoder.cols))

        cv2.imshow(window_name, pattern)
        key = cv2.waitKey(1000)
        if key == "27":
            break

    cv2.destroyAllWindows()


def get_sine_patterns():
    encoder = PhaseShift2x3Encoder(1080, 1920)
    return encoder.patterns, encoder.n

# Calculates the intrinsic matrix of the camera and distortion coefficients 
# that can be used for camera calibration.
def calculate_intrinsic_matrix():
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

    pattern_size = (9, 7)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in world space.
    imgpoints = []  # 2d points in image plane.

    imgshape = (0, 0)

    images = glob.glob('../dataset/Sinus/chess/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgshape = gray.shape[::-1]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # # Draw and display the corners
            # cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey()

    ret, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, imgshape, None, None)
    return K, dist_coeff

# Create mask for image to filter out the occluded areas
def create_mask(image):
    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(image_HSV, np.array(
        [0, 0, 0]), np.array([360, 255, 80]))
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask.astype(np.float32)
    mask = 1.0 - (mask/255)

    return mask

# Generates the phase image given a structured light input image and its mask.
def calc_phase(images, mask):
    # 102
    I1 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
    I3 = cv2.cvtColor(images[2], cv2.COLOR_BGR2GRAY)

    # Phase image calculation according to the structed light patterns paper formulae
    phase_image = np.arctan2((np.sqrt(3.0) * (I1 - I3)),
                             (2.0 * I2 - I1 - I3)) + math.pi

    # Phase shift so the period runs from 0-2pi from the start of the image to the end
    phase_image = np.add(phase_image, math.pi)
    phase_image = np.add(phase_image, math.pi * (4.0 / 3.0))
    phase_image = np.mod(phase_image, math.pi * 2.0)
    return phase_image * mask


# Phase unwrapping algorithm. Uses a low frequency phase image to unwrap 
# a high frequency phase image so its phases become unique.
def unroll_phase(low_freq, high_freq):
    new_shape = low_freq.shape
    total_phase = np.zeros(new_shape)

    for row in range(0, low_freq.shape[0]):
        for col in range(0, low_freq.shape[1]):
            offset = math.floor(
                (low_freq[row][col] * 16) / 2 * math.pi) * 2 * math.pi
            # offset = math.floor(low_freq[row][col] * (16 / (2 * math.pi))) * 2 * math.pi
            total_phase[row][col] = high_freq[row][col] + (offset)

    return total_phase

# Undistorts input image using the intrinsic camera matrix 
# and distortion coefficients of the camera. (Implementation 1)
def undistort_image(input_img, K, distortion_coeff):
    h,  w = input_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        K, distortion_coeff, (w, h), 1, (w, h))

    # Undistort
    dst = cv2.undistort(input_img, K, distortion_coeff, None, newcameramtx)

    # Crop the image (to prevent black borders)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

# Undistorts input image using the intrinsic camera matrix 
# and distortion coefficients of the camera. (Implementation 2)
def undistort_image2(input_img, K, distortion_coeff):
    h,  w = input_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        K, distortion_coeff, (w, h), 1, (w, h))

    # Undistort
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, distortion_coeff, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(input_img, mapx, mapy, cv2.INTER_LINEAR)

    # Crop the image (to prevent black borders)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

# Generates false color visualization of the scene using the 
# vertical and horizontal unwrapped phase images as unique IDs.
def create_fcv_img(hor_unwrapped, ver_unwrapped):
    fcv = np.zeros(
        [hor_unwrapped.shape[0], hor_unwrapped.shape[1], 3], np.float32)
    for row in range(hor_unwrapped.shape[0]):
        for col in range(hor_unwrapped.shape[1]):
            fcv[row][col] = [0, hor_unwrapped[row]
                             [col], ver_unwrapped[row][col]]

    # Normalize for visualization
    fcv /= np.amax(fcv)

    return fcv

# Given an array of structured light images, generate the phase images, 
# perform phase unrolling on the high frequency phase image and generate
# the false color visualization based on the vertical and horizontal unrolled phase.
def generate_fcv(images, mask, cameraview):
    # We only need the grayscale image
    # images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # Calculate high and low frequency phase images for the horizontal and vertical pattern directions
    high_UB = calc_phase(images[8:11], mask)
    low_UB = calc_phase(images[11:14], mask)
    high_LR = calc_phase(images[2:5], mask)
    low_LR = calc_phase(images[5:8], mask)

    if SHOW_INTERMEDIATE_OUTPUTS:
        cv2.imshow("Low frequency horizontal phase img", low_LR/(math.pi))
        cv2.imshow("High frequency horizontal phase img", high_LR/(math.pi))
        cv2.waitKey()

    # Unroll the high frequency phase image (for both the vertical and horizontal direction)
    phase_UB = unroll_phase(low_UB, high_UB)
    phase_LR = unroll_phase(low_LR, high_LR)

    # Plotting unrolled phase (horizontal)
    if SHOW_INTERMEDIATE_OUTPUTS:
        x = np.arange(0, phase_LR.shape[1])
        y = phase_LR[50]
        plt.title("Scan line unwrapped low_freq_hor")
        plt.plot(x, y, color="red")
        plt.show()

    # False color visualization
    fcv = create_fcv_img(phase_LR, phase_UB)
    if SHOW_INTERMEDIATE_OUTPUTS:
        cv2.imshow("False Color Visualization", fcv.astype(np.float32))
        cv2.waitKey()

    if cameraview == "view0":
        cv2.imwrite("fcv0.jpg", fcv.astype(np.float32) * 255)
    elif cameraview == "view1":
        cv2.imwrite("fcv1.jpg", fcv.astype(np.float32) * 255)

# Generates a 2D array of binary codes that should be unique for each pixel, given 
# an array of structured light graycode images.
def generate_graycode_img(images):
    # Convert all images to gray scale
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    
    graycode_img = np.zeros(shape=images[0].shape, dtype=object)
    for i in range(graycode_img.shape[0]):
        for j in range(graycode_img.shape[1]):
            graycode_img[i][j] = ""
    
    # We work with step size 2 because we compare an image and it's inverse each iteration
    for i in range(0, len(images), 2):
        for row in range(len(images[i])):  
            for p in range(len(images[i][row])):
                if images[i][row][p] < images[i+1][row][p]:
                    graycode_img[row][p] += "1"
                else:
                    graycode_img[row][p] += "0"

    return graycode_img


# Given a list of coordinates, finds the median coordinate in that list.
def median_img_coordinate(coord_list):
    sorted_coords = sorted(coord_list)
    median = sorted_coords[math.floor(len(coord_list) / 2)]
    return (median[0], median[1])

# Finds pixels that match in color between 2 images (sine wave implementation)
def find_matches(img1, img2):
    color_map = defaultdict(list)
    points1 = []
    points2 = []

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i][j][0].astype(np.float32) + img1[i][j][1].astype(np.float32) + img1[i][j][2].astype(np.float32) < 10:
                continue
            color_map[str(img1[i][j])].append([j, i])

    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if str(img2[i][j]) in color_map:

                points1.append(median_img_coordinate(color_map[str(img2[i][j])]))
                points2.append((j, i))

    return points1, points2

# Finds pixels that match in color between 2 images (graycode implementation)
def find_matches_graycode(img1, img2, mask1, mask2):

    graycode_map = defaultdict(list)
    points1 = []
    points2 = []
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if mask1[i][j] == 0:
                continue
            graycode_map[img1[i][j]].append([j, i])

    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if mask2[i][j] == 0: 
                continue
            if str(img2[i][j]) in graycode_map:
                points1.append(median_img_coordinate(graycode_map[img2[i][j]]))
                points2.append((j, i))

    return points1, points2

# Helper function that randomly selects 100 matches out of all found matches.
def select_random_match_points(match_points_1, match_points_2):
    points1 = []
    points2 = []

    rand_range = len(match_points_1)

    # A larger amount of points (e.g. 100) makes the results more stable
    for i in range(100):
        index = random.randint(0, rand_range - 1)
        points1.append(match_points_1[index])
        points2.append(match_points_2[index])

    return points1, points2

# Draws 20 matches (randomly chosen out from the list of all matches) 
# onto the input image for testing purposes.
def draw_matches(img1, img2, match_points_1, match_points_2):

    rand_range = len(match_points_1)

    for i in range(20):
        index = random.randint(0, rand_range - 1)
        p1 = match_points_1[index]
        p2 = match_points_2[index]
        img1 = cv2.circle(img1, (p1[1], p1[0]), radius=10, color=((i / 5) * 255, 0, 255), thickness=5)
        img2 = cv2.circle(img2, (p2[1], p2[0]), radius=10, color=((i / 5) * 255, 0, 255), thickness=5)

    cv2.imshow("img_1_matches", img1)
    cv2.imshow("img_2_matches", img2)
    cv2.waitKey()

# Finds the pose of camera 2 (provided that camera 1 is located in the origin, unrotated).
def find_camera_2_pose(match_points_1, match_points_2, intrinsic_matrix):
    E, mask = cv2.findEssentialMat(match_points_1.astype(float), match_points_2.astype(float), intrinsic_matrix, cv2.RANSAC, 0.999, 1.0, None)
    npoints, R, t, mask = cv2.recoverPose(E, match_points_1.astype(float), match_points_2.astype(float), intrinsic_matrix, mask=mask)

    print("E: ", E)
    print("R: ", R)
    print("t: ", t)

    return R, t

# Helper function to generate 4x4 projection matrix.
def calc_projection_matrix_4x4(K, R, t):
    # 3 x 4 rotation-translation pose matrix
    RT = np.c_[R, t]

    # P = K @ RT
    proj = np.dot(K, RT)
    proj = np.append(proj, [[0.0, 0.0, 0.0, 1.0]], axis=0)

    return proj

# Helper function to generate 3x4 projection matrix.
def calc_projection_matrix_3x4(K, R, t):
    # 3 x 4 rotation-translation pose matrix
    RT = np.c_[R, t]

    # P = K @ RT
    proj = np.dot(K, RT)

    return proj

# Helper function to transform the matches coordinates to the expected input 
# for the triangulatePoints function.
def construct_matching_pts_vectors(pts1, pts2):
    list_pts_1 = [[], []]
    list_pts_2 = [[], []]

    for p in pts1:
        list_pts_1[0].append(p[0])
        list_pts_1[1].append(p[1])
    for p in pts2:
        list_pts_2[0].append(p[0])
        list_pts_2[1].append(p[1])

    return np.array([list_pts_1, list_pts_2])

# Helper function that reads in the match coordinates provided by the teaching team.
def read_matches_from_csv(filename):
    input_file = open(filename)
    csvreader = csv.reader(input_file)

    pts1 = []
    pts2 = []
    for row in csvreader:
        x_left = int(row[0])
        y_left = int(row[1].split(';')[0])
        x_right = int(row[1].split(';')[1])
        y_right = int(row[2])

        pts1.append((x_left, y_left))
        pts2.append((x_right, y_right))

    input_file.close()
    return pts1, pts2

# Use the matching pixels from the 2 stereo images to find the corresponding 3D coordinates to these pixels.
# These 3D points represent the point cloud of the scene.
def triangulatePoints(projection_matrices, matching_2d_points, input_img):
    threed_points = cv2.triangulatePoints(projection_matrices[0], projection_matrices[1], np.array(
        matching_2d_points[0], np.float32), np.array(matching_2d_points[1], np.float32))
    threed_points = threed_points.astype(np.float32)
    threed_coords = []
    colors = []
    avg = [0.0, 0.0, 0.0]

    for i in range(len(threed_points[3])):
        if threed_points[3][i] != 0:
            threed_points[0][i] = float(threed_points[0][i]) / float(threed_points[3][i])
            threed_points[1][i] = float(threed_points[1][i]) / float(threed_points[3][i])
            threed_points[2][i] = float(threed_points[2][i]) / float(threed_points[3][i])
            threed_points[3][i] = float(threed_points[3][i]) / float(threed_points[3][i])

        avg[0] += threed_points[0][i]
        avg[1] += threed_points[1][i]
        avg[2] += threed_points[2][i]
        threed_coords.append((threed_points[0][i], threed_points[1][i], threed_points[2][i]))
        # colors.append(input_img[matching_2d_points[0][1][i]][matching_2d_points[0][0][i]])

    avg = [x/len(threed_points[0]) for x in avg]

    if SHOW_INTERMEDIATE_OUTPUTS:
        printTriangulatedPoints(threed_points, colors)
    return threed_coords, np.array(avg)

# Manually calculate triangulated 3D points using SVD
def triangulatePointsManually(img1_pts, img2_pts, proj_mat1, proj_mat2):
    threed_points = []

    for i in range(len(img1_pts)):
        A = np.zeros(shape=(4,4))
        A[0] = img1_pts[i][0] * proj_mat1[2] - proj_mat1[0]
        A[1] = img1_pts[i][1] * proj_mat1[2] - proj_mat1[1]
        A[2] = img2_pts[i][0] * proj_mat2[2] - proj_mat2[0]
        A[3] = img2_pts[i][1] * proj_mat2[2] - proj_mat2[1]

     
        u, s, vh = cv2.SVDecomp(A)
        solution = vh[3]
        solution[0] /= solution[3]
        solution[1] /= solution[3]
        solution[2] /= solution[3]
        solution[3] /= solution[3]

        threed_points.append(solution)
    if SHOW_INTERMEDIATE_OUTPUTS:
        printTriangulatedPoints(threed_points, [])
    return threed_points


# Write 3D point cloud points to a file in format:
# X Y Z
# for each point
def printTriangulatedPoints(threeDpoints, colors):
    f = open("3dpoints.txt", "w")
    for i in range(len(threeDpoints[0])):
        f.write(str(threeDpoints[0][i]) + " ")
        f.write(str(threeDpoints[1][i]) + " ")
        f.write(str(threeDpoints[2][i]) + " \n")
        # f.write(str(colors[i][0]/255.0) + " ")
        # f.write(str(colors[i][1]/255.0) + " ")
        # f.write(str(colors[i][2]/255.0) + " \n")
    f.close()

# Calculates the distance between each 3D point in the point cloud and the camera position.
# Uses this distance information to construct the depth map for that camera.
def construct_depth_map(threed_points, match_pts, cam_pos, img_shape):
    depth_map = np.zeros(img_shape, np.float32)

    for i, p in enumerate(threed_points):
        dist = math.sqrt((p[0] - cam_pos[0])**2 +
                         (p[1] - cam_pos[1])**2 + (p[2] - cam_pos[2])**2)
        depth_map[match_pts[i][1]][match_pts[i][0]] = dist

    return depth_map

# Warps the image to another view (given the pose matrix of that view) using depth map information.
def warp_image(img, depth, mask, Rt, cam_matrix, dist_coeffs):
    warped_img, warped_depth, warped_mask = cv2.rgbd.warpFrame(image=img.astype(np.uint8), depth=depth, mask=mask.astype(np.uint8),
                                                               Rt=Rt, cameraMatrix=cam_matrix, distCoeff=dist_coeffs)
    if SHOW_INTERMEDIATE_OUTPUTS:
        cv2.imshow("warp", warped_img)
        cv2.waitKey()

# Given 2 poses and the amount of in-betweens to generate, generate in-between poses.
def find_intermediate_standpoints(rot1, rot2, cam1_pos, cam2_pos, amount_vp):
    inter_poses = []
    inter_translations = []
    r1_quat = Quaternion.from_matrix(rot1)
    r2_quat = Quaternion.from_matrix(rot2)

    diff = cam2_pos - cam1_pos

    for i in range(amount_vp + 2):
        t = (i) / (amount_vp + 1)
        inter_poses.append(np.array(Matrix33(r1_quat.lerp(r2_quat, t))))
        inter_translations.append(cam1_pos + t * diff)

    return inter_poses, inter_translations


# Deprojects image point to 3D (camera space)
def deproject_2d_point_to_depth_plane(twod_point, depth, focalX, focalY, opt_centerX, opt_centerY):

    x = (twod_point[0] - opt_centerX) * depth / focalX
    y = (twod_point[1] - opt_centerY) * depth / focalY
    z = depth

    return np.matrix([x, y, z, 1])


# Transforms image pixel coordinates of the corners to 3D world space for a certain depth plane
def corners_2d_to_3d(depth, rot_matrix, translate_matrix):
    uv_left_top = [0 * depth, 0 * depth, depth, 1]
    uv_left_bottom = [0 * depth, 799 * depth, depth, 1]
    uv_right_top = [1199 * depth, 0 * depth, depth, 1]
    uv_right_bottom = [1199 * depth, 799 * depth, depth, 1]

    virtual_proj = calc_projection_matrix_4x4(K, rot_matrix, translate_matrix)
    virtual_proj_inv = np.linalg.inv(virtual_proj)

    uv_left_top_proj = virtual_proj_inv @ np.transpose(uv_left_top)
    uv_left_bottom_proj = virtual_proj_inv @ np.transpose(uv_left_bottom)
    uv_right_top_proj = virtual_proj_inv @ np.transpose(uv_right_top)
    uv_right_bottom_proj = virtual_proj_inv @ np.transpose(uv_right_bottom)

    return [uv_left_top_proj, uv_left_bottom_proj, uv_right_top_proj, uv_right_bottom_proj]


# Project 3D corner points of a certain depth plane on an input image corresponding to proj_matrix
# The projection points on the image can then be used to find the homography between the 2 images.
def project_corner_points_to_input_image(threed_corner_points, proj_matrix):
    uvs = []
    for p in threed_corner_points:
        uv = np.matrix(proj_matrix) @ np.transpose(np.matrix(p))

        # Normalize 
        uv[0] = uv[0] / uv[2]
        uv[1] = uv[1] / uv[2]
        uv[2] = uv[2] / uv[2]
        uvs.append(uv)
    return uvs

# Build homography between depth plane of virtual camera and input image plane
def find_depthplane_imageplane_homography(src_points, dest_points):

    H, mask = cv2.findHomography(src_points, dest_points, cv2.RANSAC, 7.0)
    return H


# Applies homography matrix to an input image
def apply_homography(input_img, homography_matrix):
    result = cv2.warpPerspective(input_img, homography_matrix, (1200, 800))
    # cv2.imshow("Homography applied to input image", result.astype(np.uint8))
    # cv2.waitKey()
    return result


# Finds the minimal error for each pixel by comparing the difference images (difference between left and right reprojection)
def find_minimal_error_img(diff_imgs, left_projs, right_projs):
    result_img = np.zeros(diff_imgs[0].shape)
    min_errors = math.inf * np.ones((diff_imgs[0].shape[0], diff_imgs[0].shape[1]))

    kernel = np.matrix([[1,1,1,1,1],[1,1,1,1,1],[1,1,2,1,1],[1,1,1,1,1],[1,1,1,1,1]]).astype(np.float64)
    kernel /= 26

    for i, diff_img in enumerate(diff_imgs):
        if NEIGHBORHOOD_METRIC:
            diff_img = cv2.filter2D(diff_img, -1, kernel)

        for row in range(len(diff_img)):
            for col in range(len(diff_img[row])):
                eucl_norm_squared = (diff_img[row][col][0] ** 2 + diff_img[row][col][1] ** 2 + diff_img[row][col][2] ** 2)
                if eucl_norm_squared == 0:
                    continue
                if eucl_norm_squared < min_errors[row][col]:
                    min_errors[row][col] = eucl_norm_squared
                    result_img[row][col] = (left_projs[i][row][col] + right_projs[i][row][col]) / 2

    cv2.imshow("Minimal error img", result_img.astype(np.uint8))
    cv2.waitKey()


if __name__ == "__main__":
    # Calculate intrinsic matrix and distortion coefficients of the used camera
    K, dist_coeff = calculate_intrinsic_matrix()

    # Intrinsic parameters (derived from intrinsic matrix)
    focal_x = K[0][0]
    focal_y = K[1][1]
    opt_center_x = K[0][2]
    opt_center_y = K[1][2]

    # Load images as type float32 (float32 since we'll be doing a lot of arithmetic and do not want to round to integers each time)
    input_img_camera_left = cv2.imread("../dataset/Sinus/view0/00.jpg")
    input_img_camera_left = input_img_camera_left.astype(np.float32)
    input_img_camera_left = undistort_image(input_img_camera_left, K, dist_coeff)

    input_img_camera_right = cv2.imread("../dataset/Sinus/view1/00.jpg")
    input_img_camera_right = input_img_camera_right.astype(np.float32)
    input_img_camera_right = undistort_image(input_img_camera_right, K, dist_coeff)

    if GRAYCODE_VERSION:
        images_left = load_views(["../dataset/GrayCodes/graycodes_view0.xml"])[0]
        images_right = load_views(["../dataset/GrayCodes/graycodes_view1.xml"])[0]

    else:
        images_left = load_views(["../dataset/Sinus/sinus_view0.xml"])[0]
        images_right = load_views(["../dataset/Sinus/sinus_view1.xml"])[0]

    images_left = [image.astype(np.float32) for image in images_left]
    images_right = [image.astype(np.float32) for image in images_right]

    # We use a mask to filter out the parts of the image in which the pattern projection is occluded
    mask_left = create_mask(images_left[0])
    mask_left = cv2.cvtColor(mask_left, cv2.COLOR_BGR2GRAY)
    mask_right = create_mask(images_right[0])
    mask_right = cv2.cvtColor(mask_right, cv2.COLOR_BGR2GRAY)

    if GRAYCODE_VERSION:
        # Do phase unrolling for both views and generate the FCV
        graycode_img_left = generate_graycode_img(images_left)
        graycode_img_right = generate_graycode_img(images_right)

    else:
        generate_fcv(images_left, mask_left, "view0")
        generate_fcv(images_right, mask_right, "view1")

    # Find matches between the FCV's of both scene views
    fcv_img1 = cv2.imread("fcv0.jpg")
    fcv_img2 = cv2.imread("fcv1.jpg")

    if GRAYCODE_VERSION:
        pts1, pts2 = find_matches_graycode(graycode_img_left, graycode_img_right, mask_left, mask_right)
    else:
        pts1, pts2 = find_matches(fcv_img1, fcv_img2)

    if SHOW_INTERMEDIATE_OUTPUTS:
        # Draws out 20 matches (for testing purposes)
        draw_matches(fcv_img1, fcv_img2, pts1, pts2)

    #  pts1, pts2 = read_matches_from_csv("../dataset/matches.csv")

    # Calculate the position and orientation of camera 2 given the matching pixels (given that camera 1 is in the origin)
    R2, t2 = find_camera_2_pose(np.array(pts1), np.array(pts2), K)

    R1 = np.identity(3).astype(np.float32)
    t1 = np.zeros((3, 1)).astype(np.float32)

    if SHOW_INTERMEDIATE_OUTPUTS:
        print("Camera 1 position:", t1) 
        print("Camera 1 rotation:", R1 )
        print("Camera 2 position:", t2)
        print("Camera 2 rotation:", R2 )


    # Calculate the projection matrix of both cameras to use during triangulation
    proj_1 = calc_projection_matrix_3x4(K, R1, t1)
    proj_2 = calc_projection_matrix_3x4(K, R2, t2)

    # Deproject 2D matches to their corresponding 3D points in space
    if TRIANGULATE_POINTS_MANUALLY:
        triangulatePointsManually(pts1, pts2, proj_1, proj_2)
    else:
        triangulatedPoints, center = triangulatePoints([proj_1, proj_2], construct_matching_pts_vectors(pts1, pts2), images_left[0])

    if SHOW_INTERMEDIATE_OUTPUTS:
        print("Pointcloud center: ", center)

    cam1_pos = np.array([0.0, 0.0, 0.0]).astype(np.float32)
    cam2_pos = np.array([cam1_pos[0] + t2[0][0], cam1_pos[1] + t2[1][0], cam1_pos[2] + t2[2][0]]).astype(np.float32)

    dm_left = construct_depth_map(triangulatedPoints, pts1, cam1_pos, (fcv_img1.shape[0], fcv_img1.shape[1]))
    dm_right = construct_depth_map(triangulatedPoints, pts2, cam2_pos, (fcv_img2.shape[0], fcv_img2.shape[1]))
    cv2.imwrite("depth_left.jpg", dm_left.astype(np.float32) / np.max(dm_left) * 255)
    cv2.imwrite("depth_right.jpg", dm_right.astype(np.float32) / np.max(dm_right) * 255)

    # Find the intermediate poses of the virtual cameras between the 2 stereo input cameras
    inter_rots, inter_pos = find_intermediate_standpoints(R1, R2, cam1_pos, cam2_pos, 2)

    # # Warp frame for left camera
    # src_img = cv2.imread("../dataset/Sinus/view0/00.jpg")
    # Rt = np.c_[inter_rots[0], inter_pos[0]].astype(np.float32)
    # warp_image(src_img, dm_left, mask_left, Rt, K.astype(np.float32), dist_coeff.astype(np.float32))

    # Calculate the distance to the center of the point cloud for each virtual camera
    dists_to_center = []
    for p in inter_pos:
        dists_to_center.append(math.sqrt((center[0] - p[0])**2 + (center[1] - p[1])**2 + (center[2] - p[2])**2))

    # Iterate over all depths, iterate over all virtual cameras, find the corresponding depth plane,
    # deproject the left and right input images onto this plane. Then, for each pixel, find the plane 
    # with the minimal error between the left and right projection. The average of the left and right 
    # projection onto that plane will be the resulting value for the pixel in question.
    depths = np.linspace(center[2] - 0.30, center[2] + 0.15, num=40)

    for i in range(1, len(inter_rots) - 1):
        diff_imgs = []
        left_projs = []
        right_projs = []
        for d in depths:

            # Find the 3D world positions of the corners of depth plane D_i for all of the virtual cameras
            threed_corners = corners_2d_to_3d(d, inter_rots[i], inter_pos[i])

            # Project the 3D world positions of the corners onto the input images and use the projected 2D points as
            # correspondences to build up a homography
            src_uvs = [[0, 0], [0, 799], [1199, 0], [1199, 799]]

            # Left deprojection to depth plane
            uvs_left = project_corner_points_to_input_image(threed_corners, proj_1)
            H_left = find_depthplane_imageplane_homography(np.array(uvs_left), np.array(src_uvs))
            left_projected_on_depth_plane = apply_homography(input_img_camera_left, H_left)
            left_projs.append(left_projected_on_depth_plane)

            # Right deprojection to depth plane
            uvs_right = project_corner_points_to_input_image(threed_corners, proj_2)
            H_right = find_depthplane_imageplane_homography(np.array(uvs_right), np.array(src_uvs))
            right_projected_on_depth_plane = apply_homography(input_img_camera_right, H_right)
            right_projs.append(right_projected_on_depth_plane)

            # Absolute difference between left and right projection
            diff_img = cv2.absdiff(left_projected_on_depth_plane, right_projected_on_depth_plane)
            diff_imgs.append(diff_img)
            cv2.imshow("Abs diff", diff_img.astype(np.uint8))
            cv2.waitKey()

        # For each pixel, find the depth plane with the minimal difference between left and right projections.     
        find_minimal_error_img(diff_imgs, left_projs, right_projs)

import cv2
import numpy as np

class ImageRegistration:
    def __init__(self):
        self.threshold = 1

import os
import imutils

PATH = "/media/ferhatcan/common/Image_Datasets/Flir/UPDATE 8-19-19_ SB Free Dataset-selected/FLIR_ADAS_1_3/train/"
DEBUG = False
# Open the image files.
ir_color = cv2.imread(os.path.join(PATH, "thermal_8_bit/FLIR_00034.jpeg"))  # Image to be aligned.
vis_color = cv2.imread(os.path.join(PATH, "RGB/FLIR_00034.jpg"))    # Reference image.

ir_color = cv2.resize(ir_color, (vis_color.shape[1], vis_color.shape[0]))

# # Convert to grayscale.
# ir = cv2.cvtColor(ir_color, cv2.COLOR_BGR2GRAY)
# vis = cv2.cvtColor(vis_color, cv2.COLOR_BGR2GRAY)
# height, width = vis.shape
#
# # Create ORB detector with 5000 features.
# orb_detector = cv2.AKAZE_create()
#
# # Find keypoints and descriptors.
# # The first arg is the image, second arg is the mask
# #  (which is not reqiured in this case).
# kp1, d1 = orb_detector.detectAndCompute(ir, None)
# kp2, d2 = orb_detector.detectAndCompute(vis, None)
#
# # Match features between the two images.
# # We create a Brute Force matcher with
# # Hamming distance as measurement mode.
# matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# # Match the two sets of descriptors.
# matches = matcher.match(d1, d2)
#
# # Sort matches on the basis of their Hamming distance.
# matches.sort(key=lambda x: x.distance)
#
# # Take the top 90 % matches forward.
# matches = matches[:int(len(matches) * 0.05)]
# no_of_matches = len(matches)
# print(no_of_matches)
#
# # check to see if we should visualize the matched keypoints
# if DEBUG:
#     matchedVis = cv2.drawMatches(ir, kp1, vis_color, kp2,
#         matches[:10], None)
#     matchedVis = imutils.resize(matchedVis, width=1000)
#     cv2.imshow("Matched Keypoints", matchedVis)
#     cv2.waitKey(0)
#
# # Define empty matrices of shape no_of_matches * 2.
# p1 = np.zeros((no_of_matches, 2))
# p2 = np.zeros((no_of_matches, 2))
#
# for i in range(len(matches)):
#     p1[i, :] = kp1[matches[i].queryIdx].pt
#     p2[i, :] = kp2[matches[i].trainIdx].pt
#
# # Find the homography matrix.
# homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
# np.save('homography_flirADAS', homography)
homography = np.load('homography_flirADAS.npy')

height, width, _ = ir_color.shape
# Use this matrix to transform the
# colored image wrt the reference image.
transformed_img = cv2.warpPerspective(ir_color, homography, (width, height))

dst = cv2.addWeighted(transformed_img, 0.7, vis_color, 0.3, 0.0)

dst = dst[200:height-200, 200:width-200]

cv2.imshow("result", transformed_img)
cv2.imshow("IR", imutils.resize(ir_color, width=1000))
cv2.imshow("Blended", imutils.resize(dst, width=1000))
# cv2.imshow("VIS", imutils.resize(vis_color, width=1000))
cv2.waitKey()
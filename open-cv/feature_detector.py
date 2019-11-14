import cv2
import numpy as np

img_url = "../assert/imgs/meinv.jpg"
apple_url = "../assert/imgs/apple.jpg"

kenerl_3x3 = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

kenerl_5x5 = np.array([[-1, -1, -1, -1, -1, ],
                       [-1, -1, 2, -1, -1, ],
                       [-1, 2, 4, 2, -1, ],
                       [-1, 1, 2, 1, -1, ],
                       [-1, -1, -1, -1, -1, ]])

img = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE)

apple = cv2.imread(apple_url, cv2.IMREAD_GRAYSCALE)


def show_image(mat, name="image"):
    cv2.imshow(name, mat)
    cv2.waitKey()
    cv2.destroyWindow(name)


def show_images(mats):
    for index in range(len(mats)):
        cv2.imshow("image-%s" % index, mats[index])
    cv2.waitKey()
    cv2.destroyAllWindows()


# 边角检测
def find_corners():
    dst = cv2.cornerHarris(img, 2, 11, 0.04)
    img[dst > 0.01 * dst.max()] = 255
    show_image(img)


# find_corners()


# 特征转换
def test_sift():
    sift = cv2.xfeatures2d.SIFT_create()
    global img
    keypoints, descriptor = sift.detectAndCompute(img, None)
    img = cv2.drawKeypoints(img, outImage=img, keypoints=keypoints, color=(255, 0, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    show_images([img])


# test_sift()


# 快速特征检测
def test_surf():
    """
    可以使用试探法来让函数达到最优，阈值越高，特征点就越少
    那么可以做一个滑动条，来对图片进行动态阈值采样
    :return:
    """
    sift = cv2.xfeatures2d.SURF_create(8000)
    global img
    keypoints, descriptor = sift.detectAndCompute(img, None)
    img = cv2.drawKeypoints(img, outImage=img, keypoints=keypoints, color=(255, 0, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    show_images([img])


# test_surf()


# 特征匹配
def test_feature_match():
    gray = cv2.imread("gray-meinv.jpg", cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    keypoint1, des1 = orb.detectAndCompute(img, None)
    keypoint2, des2 = orb.detectAndCompute(gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img, keypoint1, gray, keypoint2, matches[:100], img, (255, 0, 0))

    show_images([img3])


# test_feature_match()


# 用KNN来筛选matches
def test_knn_feature_matches():
    gray = cv2.imread("gray-meinv.jpg", cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    keypoint1, des1 = orb.detectAndCompute(img, None)
    keypoint2, des2 = orb.detectAndCompute(gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1, des2, k=2)

    print()

    # matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatchesKnn(img, keypoint1, gray, keypoint2, matches, img)

    # show_images([img3])


test_knn_feature_matches()

import cv2
import numpy as np
from scipy import ndimage

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


# 这么多不同的kernel转变之后的图像样式
def test_scipy_conv():
    # con_img = ndimage.convolve(img, kenerl_3x3)
    con_img = ndimage.convolve(img, kenerl_5x5)
    blurred = cv2.GaussianBlur(con_img, (11, 11), 0)
    cha1 = img - blurred
    cha2 = img - con_img
    cha3 = con_img - blurred
    show_images([con_img, blurred, cha1, cha2, cha3])


# test_scipy_conv()


# 模糊化
def test_blurs():
    base_size = 2
    imgs = []
    for item in range(4, 7):
        size = item * base_size - 1
        imgs.append(cv2.blur(img, (size, size), 0))
    show_images(imgs)


# test_blurs()

# 将图像数据二值化
def test_filter2D():
    filter_img = cv2.filter2D(img, -1, kenerl_5x5)
    show_images([filter_img])


# test_filter2D()


def strokeEdges(src, dst, blurSize=7, edgeKsize=5):
    if blurSize > 3:
        blurredSrc = cv2.medianBlur(src, blurSize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_RGB2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizeInverseAlpha = (1.0 / 255.0) * (255 - graySrc)


def test_canny():
    blur_img = cv2.blur(img, (4, 4))
    canny_img = cv2.Canny(blur_img, 100, 300)
    show_images([blur_img, canny_img])


# test_canny()

def test_threshold():
    ret, thresh = cv2.threshold(apple, 150, 255, 0)
    ss = thresh.copy()
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    cv2.drawContours(image, contours, -1, (0, 0, 255), 6)
    # ret, thresh = cv2.threshold(img, 150, 255, 0)
    show_images([apple, ss, image])


# test_threshold()


def test_contours():
    img = apple
    _, threshold = cv2.threshold(img, 127, 255, 0)
    image, contours, hier = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))

        # 画最小的一个圆

    show_image(image)


# test_contours()


def face_detect():
    face_xml = "haarcascade_frontalface_alt.xml"

    face_cascade = cv2.CascadeClassifier(face_xml)

    global img
    faces = face_cascade.detectMultiScale(img)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

    show_image(img)


# 检测复仇者联盟中不同的英雄，然而效果是相当的差
def detect_heros():
    face_xml = "haarcascade_frontalface_alt.xml"

    img = cv2.imread("../assert/imgs/heros.jpg")

    face_cascade = cv2.CascadeClassifier(face_xml)

    faces = face_cascade.detectMultiScale(img, 1.1, 1)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

    show_image(img)


# face_detect()
# detect_heros()


def detect_face_in_video():
    capture = cv2.VideoCapture(0)
    success, frame = capture.read()

    face_xml = "haarcascade_frontalface_alt.xml"

    face_cascade = cv2.CascadeClassifier(face_xml)

    while success and cv2.waitKey(1) == -1:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.imshow("ss", frame)
        success, frame = capture.read()

    capture.release()


detect_face_in_video()

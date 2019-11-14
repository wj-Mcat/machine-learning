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


import cv2

img_url = "../assert/imgs/meinv.jpg"


def test_video_capture():
    capture = cv2.VideoCapture(0)
    success, frame = capture.read()
    cv2.namedWindow("Image")
    # 当检测到没有键盘按键时，cv2.waitKey(1) 返回值为-1
    while success and cv2.waitKey(1) == -1:
        cv2.imshow("Image", frame)
        success, frame = capture.read()

    cv2.destroyAllWindows()
    capture.release()


# 这里提一个问题，如果在读取图像的循环过程中，没有添加退出(break)代码，图像是不显示出来的
# 到底这是一个什么奇葩的问题，需要后期进行验证

def problem_1():
    capture = cv2.VideoCapture(0)
    success, frame = capture.read()
    cv2.namedWindow("Image")
    while success:
        cv2.imshow("Image", frame)
        success, frame = capture.read()

    cv2.destroyAllWindows()
    capture.release()


# problem_1()


# 测试一个截屏的功能
def test_scrren_capture():
    capture = cv2.VideoCapture(0)
    success, frame = capture.read()
    cv2.namedWindow("Image")
    # 当检测到没有键盘按键时，cv2.waitKey(1) 返回值为-1
    while success:
        cv2.imshow("Image", frame)
        success, frame = capture.read()

        # 添加一些额外的逻辑
        key = cv2.waitKey(1)
        if key != -1:
            print(key)
        # 按下C键
        if key == 99:
            cv2.imwrite("face.jpg", frame)
            break

    cv2.destroyAllWindows()
    capture.release()


# 多个摄像头同步获取数据
def test_multiple_captures():
    capture0 = cv2.VideoCapture(0)
    capture1 = cv2.VideoCapture(1)

    success0 = capture0.grab()
    success1 = capture1.grab()

    while success0 and success1:
        frame0 = capture0.retrieve()
        frame1 = capture1.retrieve()

        cv2.imshow("image-0", frame0)
        cv2.imshow("image-1", frame1)



import cv2
import numpy as np
import sys
import os
from tkinter import *



def get_path_by_relative(path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), path)


def train_person():
    if len(sys.argv) < 3:
        raise EnvironmentError("请输入用户名称")
    person_name = sys.argv[2]
    capture = cv2.VideoCapture(0)
    success, frame = capture.read()
    face_xml = "haarcascade_frontalface_alt.xml"
    face_cascade = cv2.CascadeClassifier(face_xml)
    count = 0
    while success and cv2.waitKey(1) == -1:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 2)
        if len(faces) > 0 and count < 100:
            folder = "data\%s" % person_name

            if not os.path.isdir(folder):
                path = os.path.join(os.path.dirname(os.path.realpath(__file__)), folder)
                os.makedirs(path)
            base_path = "./data/%s/%d.pgm" % (person_name, count)

            if os.path.exists(base_path):
                os.remove(base_path)

            x, y, w, h = faces[0]
            face_img = cv2.resize(gray[x:x + w, y:y + h], (200, 200))
            print(base_path)
            cv2.imwrite(base_path, face_img)
            count += 1
        else:
            break
        cv2.imshow("ss", frame)
        success, frame = capture.read()


    capture.release()


def predict_person():
    def read_image_data():
        x, y = [], []
        for person_name in os.listdir("./data"):
            for _, _, files in os.walk("./data/%s" % person_name):
                for file in files:
                    train_data = cv2.imread("./data/%s/%s" % (person_name, file), cv2.IMREAD_GRAYSCALE)
                    x.append(np.asarray(train_data))
                    y.append(person_name)
        return x, y

    train_x, train_y = read_image_data()
    names = list(set(train_y))
    model = cv2.face.EigenFaceRecognizer_create()
    train_x, train_y = np.asarray(train_x), np.array([names.index(x) for x in train_y])
    model.train(train_x, train_y)

    # https://stackoverflow.com/questions/46288224/opencv-attributeerror-module-cv2-has-no-attribute-face

    #
    #
    capture = cv2.VideoCapture(0)
    success, frame = capture.read()
    face_xml = "haarcascade_frontalface_alt.xml"
    face_cascade = cv2.CascadeClassifier(face_xml)
    count = 0
    while success and cv2.waitKey(1) == -1:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 2)
        if len(faces) > 0 and count < 100:
            x, y, w, h = faces[0]
            face_img = cv2.resize(gray[x:x + w, y:y + h], (200, 200))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            predict_result = model.predict(face_img)
            cv2.putText(frame, "Labels:%s---score:%s" % (names[predict_result[0]], predict_result[1]), (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

        cv2.imshow("ss", frame)
        success, frame = capture.read()

    capture.release()


def remove_all_labels():
    import shutil
    shutil.rmtree("./data")


if __name__ == "__main__" and len(sys.argv) > 1:
    type = sys.argv[1]
    if type == "train" or type == "t":
        train_person()
    elif type == "c" or type == "clear":
        remove_all_labels()
    elif type == "predict" or type == "p":
        predict_person()

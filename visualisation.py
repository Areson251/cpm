from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys 

import comparison


class AlgoProcessing(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CPM")
        self.setFixedSize(QSize(800, 600))
        self.setStyleSheet("background-color: white;")

        self.button_take_photo = QPushButton("ВЕРНУТЬСЯ К\nРЕДАКТИРОВАНИЮ", self)
        self.button_take_photo.setGeometry(50, 520, 200, 60)
        self.button_take_photo.clicked.connect(self.remake_photo)

        self.button_save_photo = QPushButton("ОСТАНОВИТЬ", self)
        self.button_save_photo.setGeometry(540, 530, 200, 40)
        # self.button_save_photo.clicked.connect(self.save_img)

        self.map_img = cv2.imread("photos/1_yandex.png")
        self.map_img = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2RGB) 
        self.map_img = QImage(self.map_img.data, self.map_img.shape[1], self.map_img.shape[0], self.map_img.strides[0], QImage.Format.Format_RGB888)

        self.label1 = QLabel(self)
        self.pixmap1 = QPixmap(self.map_img)
        self.label1.setPixmap(self.pixmap1)
        self.label1.setGeometry(25, 50, int(self.pixmap1.width()*0.3), int(self.pixmap1.height()*0.3))

        self.show()
    
    def start_algo(self, img, map, algo=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
        self.hot_map = comparison.count_difference_with_step(map, gray)
        self.hot_map = np.array(self.hot_map, dtype='uint8')
        width = int(self.hot_map.shape[1] * 5)
        height = int(self.hot_map.shape[0] * 5)
        dim = (width, height)
        print(dim)
        # self.hot_map = cv2.resize(self.hot_map, dim, interpolation = cv2.INTER_LINEAR)
        # cv2.imshow("s", self.hot_map)
        # self.hot_map = cv2.cvtColor(self.hot_map, cv2.COLOR_BGR2RGB) 
        self.hot_map = QImage(self.hot_map.data, self.hot_map.shape[1], self.hot_map.shape[0], self.hot_map.strides[0], QImage.Format.Format_RGB888)

        self.label2 = QLabel(self)
        self.pixmap2 = QPixmap(self.hot_map)
        self.label2.setPixmap(self.pixmap2)
        self.label2.setGeometry(25, 300, self.pixmap2.width(), self.pixmap2.height())
        # cv2.imshow('result', hot_map.astype(np.uint8))

    def remake_photo(self):
        photo_window.show()
        process_window.close() 


class ImagePreprocessing(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CPM")
        self.setFixedSize(QSize(1000, 600))
        self.setStyleSheet("background-color: white;")

        self.sld1 = QSlider(Qt.Orientation.Horizontal, self)
        self.sld1.setGeometry(350, 50, 300, 20)
        self.sld1.valueChanged[int].connect(self.sld1_change_value)

        self.sld2 = QSlider(Qt.Orientation.Horizontal, self)
        self.sld2.setGeometry(350, 100, 300, 20)
        self.sld2.valueChanged[int].connect(self.sld1_change_value)

        self.sld3 = QSlider(Qt.Orientation.Horizontal, self)
        self.sld3.setGeometry(350, 150, 300, 20)
        self.sld3.valueChanged[int].connect(self.sld1_change_value)

        self.load_img()

        self.button_take_photo = QPushButton("ПЕРЕДЕЛАТЬ ФОТО", self)
        self.button_take_photo.setGeometry(125, 520, 200, 40)
        self.button_take_photo.clicked.connect(self.retake_photo)

        self.button_save_photo = QPushButton("СОХРАНИТЬ", self)
        self.button_save_photo.setGeometry(700, 520, 200, 40)
        self.button_save_photo.clicked.connect(self.save_img)

        self.show()

    def sld1_change_value(self, value):
        print(value)

    def load_img(self):
        self.img1 = cv2.imread("photos/2.png")
        self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB) 
        self.image1 = QImage(self.img1.data, self.img1.shape[1], self.img1.shape[0], QImage.Format.Format_RGB888)

        self.img2 = self.img1
        self.image2 = QImage(self.img2.data, self.img2.shape[1], self.img2.shape[0], QImage.Format.Format_RGB888)


        self.arrow = cv2.imread("photos/visualisation/arrow.png")
        self.arrow = cv2.cvtColor(self.arrow, cv2.COLOR_BGR2RGB) 
        self.arrow = QImage(self.arrow.data, self.arrow.shape[1], self.arrow.shape[0], self.arrow.strides[0], QImage.Format.Format_RGB888)

        self.label1 = QLabel(self)
        self.pixmap1 = QPixmap(self.image1)
        self.label1.setPixmap(self.pixmap1)
        self.label1.setGeometry(25, 200, self.pixmap1.width(), self.pixmap1.height())

        self.label2 = QLabel(self)
        self.pixmap2 = QPixmap(self.image2)
        self.label2.setPixmap(self.pixmap2)
        self.label2.setGeometry(575, 200, self.pixmap2.width(), self.pixmap2.height())

        self.label_arrow = QLabel(self)
        self.pixmap_arrow = QPixmap(self.arrow)
        self.label_arrow.setPixmap(self.pixmap_arrow)
        self.label_arrow.setGeometry(440, 275, self.pixmap_arrow.width(), self.pixmap_arrow.height())

    def retake_photo(self):
        main_window.show()
        photo_window.close() 

    def save_img(self):
        process_window.start_algo(self.img2, map_1)
        process_window.show()
        photo_window.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CPM")
        self.setFixedSize(QSize(800, 600))
        self.setStyleSheet("background-color: white;")

        self.label = QLabel(self)
        self.pixmap = QPixmap('photos/2.png')
        self.label.setPixmap(self.pixmap)
        self.label.setGeometry(50, 125, self.pixmap.width(), self.pixmap.height())

        self.button_take_photo = QPushButton("СДЕЛАТЬ СНИМОК", self)
        self.button_take_photo.setGeometry(550, 100, 200, 40)

        self.button_save_photo = QPushButton("ПОСМОТРЕТЬ КАРТУ", self)
        self.button_save_photo.setGeometry(550, 250, 200, 40)

        self.button_change_algo = QPushButton("ВЫБРАТЬ АЛГОРИТМ", self)
        self.button_change_algo.setGeometry(550, 400, 200, 40)

        self.button_take_photo.clicked.connect(self.take_photo)
        self.button_save_photo.clicked.connect(self.the_button_was_clicked)
        self.button_change_algo.clicked.connect(self.the_button_was_clicked)

        self.show()

    def the_button_was_clicked(self):
        print("Clicked!")

    def take_photo(self):
        photo_window.show()
        main_window.close() 


if __name__ == "__main__":
    app = QApplication(sys.argv)
    map_1 = cv2.imread("photos/1_yandex.png",0)

    main_window = MainWindow()
    main_window.show()

    photo_window = ImagePreprocessing()
    photo_window.close()

    process_window = AlgoProcessing()
    process_window.close()

    app.exec()
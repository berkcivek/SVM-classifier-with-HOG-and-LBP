import pickle
import cv2
import imutils
import hog
import lbp
import numpy as np
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMessageBox
from skimage.transform import resize
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from trainWindow import Ui_Form_ModelWin


class Ui_Form(object):
    def yeniac(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_Form_ModelWin()
        self.ui.setupUi(self.window)
        self.window.show()

    def setupUi(self, Form):
        self.path = ""
        self.path_model = ""
        Form.setObjectName("Form")
        Form.setEnabled(True)
        Form.resize(759, 553)
        Form.setStyleSheet("QWidget{\n"
                           "background-color:#082840;\n"
                           "border: 3px solid #1f1f1f;\n"
                           "\n"
                           "}\n"
                           "\n"
                           "")
        self.picture_box = QtWidgets.QLabel(Form)
        self.picture_box.setGeometry(QtCore.QRect(40, 30, 401, 351))
        self.picture_box.setStyleSheet("QLabel{\n"
                                       "border: 3px solid #ffffff;\n"
                                       "border-radius:20px;\n"
                                       "background-color:#1f1f1f;\n"
                                       "}")
        self.picture_box.setText("")
        self.picture_box.setObjectName("picture_box")
        self.frame_2 = QtWidgets.QFrame(Form)
        self.frame_2.setGeometry(QtCore.QRect(460, 30, 251, 351))
        self.frame_2.setStyleSheet("QFrame{\n"
                                   "background-color:#3c3c3e;\n"
                                   "border-radius:10px;\n"
                                   "border-radius:20px;\n"
                                   "border: 3px solid #ffffff;\n"
                                   "\n"
                                   "}")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.train_btn = QtWidgets.QPushButton(self.frame_2)
        self.train_btn.setGeometry(QtCore.QRect(20, 20, 201, 61))
        self.train_btn.clicked.connect(self.yeniac)
        font = QtGui.QFont()
        font.setFamily("Copperplate Gothic Bold")
        font.setPointSize(15)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        font.setKerning(True)
        self.train_btn.setFont(font)
        self.train_btn.setStyleSheet("QPushButton\n"
                                     "{\n"
                                     "background-color:#2AA9C7;\n"
                                     "border-radius:20px;\n"
                                     "border: 3px solid #ffffff;\n"
                                     "\n"
                                     "}\n"
                                     "QPushButton:hover{\n"
                                     "background-color:#b7db35;\n"
                                     "}\n"
                                     "\n"
                                     "\n"
                                     "QPushButton:pressed{\n"
                                     "background-color:rgb(170, 0, 0);\n"
                                     "\n"
                                     "}\n"
                                     "")
        self.train_btn.setIconSize(QtCore.QSize(40, 40))
        self.train_btn.setAutoRepeat(False)
        self.train_btn.setAutoExclusive(False)
        self.train_btn.setObjectName("train_btn")
        self.model_btn = QtWidgets.QPushButton(self.frame_2)
        self.model_btn.setGeometry(QtCore.QRect(20, 90, 201, 61))
        font = QtGui.QFont()
        font.setFamily("Copperplate Gothic Bold")
        font.setPointSize(14)
        self.model_btn.setFont(font)
        self.model_btn.setStyleSheet("QPushButton\n"
                                     "{\n"
                                     "background-color:#2AA9C7;\n"
                                     "border-radius:20px;\n"
                                     "border: 3px solid #ffffff;\n"
                                     "\n"
                                     "}\n"
                                     "QPushButton:hover{\n"
                                     "background-color:#b7db35;\n"
                                     "}\n"
                                     "\n"
                                     "\n"
                                     "QPushButton:pressed{\n"
                                     "background-color:rgb(170, 0, 0);\n"
                                     "\n"
                                     "}\n"
                                     "")
        self.model_btn.setIconSize(QtCore.QSize(40, 40))
        self.model_btn.setObjectName("model_btn")
        self.img_btn = QtWidgets.QPushButton(self.frame_2)
        self.img_btn.setGeometry(QtCore.QRect(20, 160, 201, 61))
        font = QtGui.QFont()
        font.setFamily("Copperplate Gothic Bold")
        font.setPointSize(14)
        self.img_btn.setFont(font)
        self.img_btn.setStyleSheet("QPushButton\n"
                                   "{\n"
                                   "background-color:#2AA9C7;\n"
                                   "border-radius:20px;\n"
                                   "border: 3px solid #ffffff;\n"
                                   "\n"
                                   "}\n"
                                   "QPushButton:hover{\n"
                                   "background-color:#b7db35;\n"
                                   "}\n"
                                   "\n"
                                   "\n"
                                   "QPushButton:pressed{\n"
                                   "background-color:rgb(170, 0, 0);\n"
                                   "\n"
                                   "}\n"
                                   "")
        self.img_btn.setIconSize(QtCore.QSize(40, 40))
        self.img_btn.setObjectName("img_btn")
        self.combo_feature = QtWidgets.QComboBox(self.frame_2)
        self.combo_feature.setGeometry(QtCore.QRect(20, 230, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        self.combo_feature.setFont(font)
        self.combo_feature.setAutoFillBackground(False)
        self.combo_feature.setStyleSheet("QComboBox{\n"
                                         "\n"
                                         "background-color:#1f1f1f;\n"
                                         "border-radius:10px;\n"
                                         "border-radius:20px;\n"
                                         "border: 3px solid #ffffff;\n"
                                         "color:white;\n"
                                         "\n"
                                         "\n"
                                         "}\n"
                                         "QComboBox:hover{\n"
                                         "    border: 2px solid rgb(64, 71, 88);\n"
                                         "}\n"
                                         "QComboBox QAbstractItemView {\n"
                                         "    color: rgb(85, 170, 255);    \n"
                                         "    background-color: rgb(27, 29, 35);\n"
                                         "    padding: 10px;\n"
                                         "    selection-background-color: rgb(39, 44, 54);\n"
                                         "}")
        self.combo_feature.setIconSize(QtCore.QSize(16, 16))
        self.combo_feature.setFrame(True)
        self.combo_feature.setObjectName("combo_feature")
        self.combo_feature.addItem("")
        self.combo_feature.addItem("")
        self.combo_class = QtWidgets.QComboBox(self.frame_2)
        self.combo_class.setGeometry(QtCore.QRect(20, 270, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        self.combo_class.setFont(font)
        self.combo_class.setAutoFillBackground(False)
        self.combo_class.setStyleSheet("QComboBox{\n"
                                       "\n"
                                       "background-color:#1f1f1f;\n"
                                       "border-radius:10px;\n"
                                       "border-radius:20px;\n"
                                       "border: 3px solid #ffffff;\n"
                                       "color:white;\n"
                                       "\n"
                                       "}\n"
                                       "QComboBox:hover{\n"
                                       "    border: 2px solid rgb(64, 71, 88);\n"
                                       "}\n"
                                       "QComboBox QAbstractItemView {\n"
                                       "    color: rgb(85, 170, 255);    \n"
                                       "    background-color: rgb(27, 29, 35);\n"
                                       "    padding: 10px;\n"
                                       "    selection-background-color: rgb(39, 44, 54);\n"
                                       "}")
        self.combo_class.setIconSize(QtCore.QSize(16, 16))
        self.combo_class.setFrame(True)
        self.combo_class.setObjectName("combo_class")
        self.combo_class.addItem("")
        self.combo_class.addItem("")
        self.text_predict = QtWidgets.QLabel(Form)
        self.text_predict.setGeometry(QtCore.QRect(40, 400, 401, 71))
        self.text_predict.setStyleSheet("QLabel{\n"
                                        "background-color:#1f1f1f;\n"
                                        "border-radius:10px;\n"
                                        "border-radius:20px;\n"
                                        "border: 3px solid #ffffff;\n"
                                        "color:white;\n"
                                        "font-size:15px;\n"
                                        "}")
        self.text_predict.setObjectName("text_predict")
        self.predict_btn = QtWidgets.QPushButton(Form)
        self.predict_btn.setGeometry(QtCore.QRect(480, 400, 201, 61))
        font = QtGui.QFont()
        font.setFamily("Copperplate Gothic Bold")
        font.setPointSize(14)
        self.predict_btn.setFont(font)
        self.predict_btn.setStyleSheet("QPushButton\n"
                                       "{\n"
                                       "background-color:#2AA9C7;\n"
                                       "border-radius:20px;\n"
                                       "border: 3px solid #ffffff;\n"
                                       "\n"
                                       "}\n"
                                       "QPushButton:hover{\n"
                                       "background-color:#b7db35;\n"
                                       "}\n"
                                       "\n"
                                       "\n"
                                       "QPushButton:pressed{\n"
                                       "background-color:rgb(170, 0, 0);\n"
                                       "\n"
                                       "}\n"
                                       "")
        self.predict_btn.setIconSize(QtCore.QSize(40, 40))
        self.predict_btn.setObjectName("predict_btn")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "HOG - LBG Goruntu Tahmini"))
        self.train_btn.setText(_translate("Form", "Egitim Yap"))
        self.model_btn.setText(_translate("Form", "Model Sec"))
        self.img_btn.setText(_translate("Form", "Resim Sec"))
        self.combo_feature.setItemText(0, _translate("Form", "HOG"))
        self.combo_feature.setItemText(1, _translate("Form", "LBP"))
        self.combo_class.setItemText(0, _translate("Form", "SVM"))
        self.combo_class.setItemText(1, _translate("Form", "*"))
        self.text_predict.setText(_translate("Form", "Tahmin Bekleniyor:"))
        self.predict_btn.setText(_translate("Form", "Tahmin Et"))

        self.predict_btn.clicked.connect(self.feature_extracture)
        self.img_btn.clicked.connect(self.select_image)
        self.model_btn.clicked.connect(self.select_model)

    def select_image(self):
        self.text_predict.setText("Tahmin Bekleniyor...")
        filename = QFileDialog.getOpenFileName()
        self.path = filename[0]

        if filename[0] != '':
            image = cv2.imread(self.path)
            image = imutils.resize(image, 400, 350)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width, channel = image.shape
            step = channel * width

            qimage = QImage(image.data, width, height,
                            step, QImage.Format_RGB888)

            self.picture_box.setPixmap(QPixmap.fromImage(qimage))

    def select_model(self):
        filename = QFileDialog.getOpenFileName()
        self.path_model = filename[0]

    def feature_extracture(self):

        self.text_predict.setText("Tahmin Bekleniyor...")
        start = time.time()
        # ============================ HOG - BASLANGIC ====================== #
        if self.combo_feature.currentText() == 'HOG':
            try:
                with open(self.path_model, 'rb') as f:
                    svm = pickle.load(f)

                image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
                image = resize(image, (256, 128))
                hog_image = hog.Hog_descriptor(image, cell_size=8, bin_size=1)
                vector, img = hog_image.extract()
                # print(len(vector)) # - DEBUG
                temp = []
                key = 0
                try:
                        for id, i in enumerate(vector):
                                if id == 0:
                                        key = key+1
                                for j in range(4):
                                        temp.append(str(i[j]).replace("[", "").replace("]", ""))
                except:
                        temp.append('0').replace("[", "").replace("]", "")

                data = np.array(temp)
                data = np.transpose(data)
                son = data.reshape(-1, len(data))
                label = svm.predict(son)
                self.text_predict.setText(label[0])
                print("| {:.1f}sn Islem suresi |".format(time.time() - start))
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Model ve Resim Dosyası Seçiniz.")
                msg.setWindowTitle("Uyarı")
                msg.setStandardButtons(QMessageBox.Ok)
                retval = msg.exec_()

            
            
            

            # ============================ HOG - BITIS ====================== #

        elif self.combo_feature.currentText() == 'LBP':
                try:
                        with open(self.path_model, 'rb') as f:
                                svm = pickle.load(f)
                        image = cv2.imread(self.path)
                        image = cv2.resize(image, (400, 400),
                                        interpolation=cv2.INTER_LINEAR)
                        lbp_image, hist_array = lbp.lbp(image)

                        data = np.array(hist_array)
                        data = np.transpose(data)
                        son = data.reshape(-1, len(data))

                        y_pred = svm.predict(son)
                        self.text_predict.setText(y_pred[0])
                        print("| {:.1f}sn Islem suresi |".format(time.time() - start))
                except:
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.Warning)
                        msg.setText("Model ve Resim Dosyası Seçiniz. Model dosyasının uyumlu olduğun emin olunuz!")
                        msg.setWindowTitle("Uyarı")
                        msg.setStandardButtons(QMessageBox.Ok)
                        retval = msg.exec_()

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1071, 620)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setMaximumSize(QtCore.QSize(150, 16777215))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setSpacing(8)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label20 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        font.setKerning(True)
        self.label20.setFont(font)
        self.label20.setObjectName("label20")
        self.verticalLayout.addWidget(self.label20)
        self.mono = QtWidgets.QRadioButton(self.frame)
        self.mono.setObjectName("mono")
        self.verticalLayout.addWidget(self.mono)
        self.stereo = QtWidgets.QRadioButton(self.frame)
        self.stereo.setObjectName("stereo")
        self.verticalLayout.addWidget(self.stereo)
        self.stereo_split = QtWidgets.QRadioButton(self.frame)
        self.stereo_split.setObjectName("stereo_split")
        self.verticalLayout.addWidget(self.stereo_split)
        self.horizontalLayout.addWidget(self.frame)
        self.line = QtWidgets.QFrame(Form)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.frame_2 = QtWidgets.QFrame(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setMaximumSize(QtCore.QSize(150, 16777215))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_2.setContentsMargins(9, -1, -1, 10)
        self.verticalLayout_2.setSpacing(19)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_2.setIndent(-1)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.octave = QtWidgets.QRadioButton(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.octave.sizePolicy().hasHeightForWidth())
        self.octave.setSizePolicy(sizePolicy)
        self.octave.setObjectName("octave")
        self.verticalLayout_2.addWidget(self.octave)
        self.th_octave = QtWidgets.QRadioButton(self.frame_2)
        self.th_octave.setObjectName("th_octave")
        self.verticalLayout_2.addWidget(self.th_octave)
        self.horizontalLayout.addWidget(self.frame_2)
        self.line_2 = QtWidgets.QFrame(Form)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.horizontalLayout.addWidget(self.line_2)
        self.frame_3 = QtWidgets.QFrame(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout.setVerticalSpacing(18)
        self.gridLayout.setObjectName("gridLayout")
        self.label_4 = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.frame_3)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 1, 1, 1)
        self.window = QtWidgets.QLineEdit(self.frame_3)
        self.window.setText("")
        self.window.setObjectName("window")
        self.gridLayout.addWidget(self.window, 3, 1, 1, 1)
        self.schroeder = QtWidgets.QRadioButton(self.frame_3)
        self.schroeder.setObjectName("schroeder")
        self.gridLayout.addWidget(self.schroeder, 1, 0, 1, 1)
        self.mmf = QtWidgets.QRadioButton(self.frame_3)
        self.mmf.setObjectName("mmf")
        self.gridLayout.addWidget(self.mmf, 3, 0, 1, 1)
        self.horizontalLayout.addWidget(self.frame_3)
        self.line_3 = QtWidgets.QFrame(Form)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontalLayout.addWidget(self.line_3)
        self.frame_4 = QtWidgets.QFrame(Form)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.load_ir = QtWidgets.QPushButton(self.frame_4)
        self.load_ir.setObjectName("load_ir")
        self.gridLayout_2.addWidget(self.load_ir, 1, 0, 1, 1)
        self.load_r_ir = QtWidgets.QPushButton(self.frame_4)
        self.load_r_ir.setObjectName("load_r_ir")
        self.gridLayout_2.addWidget(self.load_r_ir, 1, 1, 1, 1)
        self.calculate = QtWidgets.QPushButton(self.frame_4)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.calculate.setFont(font)
        self.calculate.setObjectName("calculate")
        self.gridLayout_2.addWidget(self.calculate, 2, 0, 1, 2)
        self.label_5 = QtWidgets.QLabel(self.frame_4)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 2)
        self.horizontalLayout.addWidget(self.frame_4)
        self.line_4 = QtWidgets.QFrame(Form)
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.horizontalLayout.addWidget(self.line_4)
        self.frame_5 = QtWidgets.QFrame(Form)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_6 = QtWidgets.QLabel(self.frame_5)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_3.addWidget(self.label_6)
        self.L = QtWidgets.QLabel(self.frame_5)
        self.L.setObjectName("L")
        self.verticalLayout_3.addWidget(self.L)
        self.R = QtWidgets.QLabel(self.frame_5)
        self.R.setObjectName("R")
        self.verticalLayout_3.addWidget(self.R)
        self.export_2 = QtWidgets.QPushButton(self.frame_5)
        self.export_2.setObjectName("export_2")
        self.verticalLayout_3.addWidget(self.export_2)
        self.horizontalLayout.addWidget(self.frame_5)
        self.verticalLayout_5.addLayout(self.horizontalLayout)
        self.line_5 = QtWidgets.QFrame(Form)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.verticalLayout_5.addWidget(self.line_5)
        self.label_3 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setWordWrap(False)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_5.addWidget(self.label_3)
        self.line_6 = QtWidgets.QFrame(Form)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.verticalLayout_5.addWidget(self.line_6)
        self.output_graphic = QtWidgets.QVBoxLayout()
        self.output_graphic.setObjectName("output_graphic")
        self.frame_6 = QtWidgets.QFrame(Form)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.output_graphic.addWidget(self.frame_6)
        self.verticalLayout_5.addLayout(self.output_graphic)
        self.tableWidget = QtWidgets.QTableWidget(Form)
        self.tableWidget.setEnabled(True)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.verticalLayout_5.addWidget(self.tableWidget)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Acoustical Parameters"))
        self.label20.setText(_translate("Form", "IR Type"))
        self.mono.setText(_translate("Form", "Mono"))
        self.stereo.setText(_translate("Form", "Stereo"))
        self.stereo_split.setText(_translate("Form", "Stereo (Split)"))
        self.label_2.setText(_translate("Form", "Filtering"))
        self.octave.setText(_translate("Form", "Octave"))
        self.th_octave.setText(_translate("Form", "1/3 octave"))
        self.label_4.setText(_translate("Form", "Smoothing"))
        self.label.setText(_translate("Form", "Win. Size [ms]"))
        self.schroeder.setText(_translate("Form", "Schroeder"))
        self.mmf.setText(_translate("Form", "MMF"))
        self.load_ir.setText(_translate("Form", "Load Mono IR"))
        self.load_r_ir.setText(_translate("Form", "Load R IR"))
        self.calculate.setText(_translate("Form", "Calculate"))
        self.label_5.setText(_translate("Form", "IR Selection"))
        self.label_6.setText(_translate("Form", "IR Files"))
        self.L.setText(_translate("Form", "L:"))
        self.R.setText(_translate("Form", "R:"))
        self.export_2.setText(_translate("Form", "Export CSV"))
        self.label_3.setText(_translate("Form", "RESULTS:"))


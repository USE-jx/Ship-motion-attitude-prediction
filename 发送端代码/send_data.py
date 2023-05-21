"""
发送端使用说明：
    运行程序后。会自动搜索串口毕竟显示，若有多个串口显示，选择需要的串口连接即可
    波特率，停止位，数据位，校验位等都以设置好，无需更改
    然后选择发送数据频率，然后点击”打开串口“按钮，即可连接成功。
    若显示串口连接失败，请检查串口线是否插好
    随后需要选择发送文件（打开方式”r“）和预测值保存文件（打开方式”a+“）
    若显示模型正在加载，则点击加载模型，向TX2端询问模型是否加载成功
    若显示模型加载成功，点击发送按钮，即可发送文件内容（要保证输入框无数据）
    发送前9组数前，未有预测值传输回发送端，因为前9组预测下一组
    随后会显示出预测值，预测花费时间，预测值与真实值图像信息
    文件发送完成后，想要预测新文件数据，需要重新加载模型
"""

import sys
import time
import numpy as np
import PyQt5.QtWidgets as qw
import serial
import pyqtgraph as pg
import threading
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import QObject
from time import sleep
from PyQt5.QtSerialPort import QSerialPort
from PyQt5.QtSerialPort import QSerialPortInfo
from sklearn import metrics
from sklearn.metrics import r2_score
ser = serial.Serial()
#ui界面显示模块
def mse(y, y_hat):
    error = np.sum((y - y_hat) ** 2) / len(y)
    return error


def rmse(y, y_hat):
    error = mse(y, y_hat) ** 0.5
    return error


def mae(y, y_hat):
    error = np.sum(np.abs(y - y_hat)) / len(y)
    return error

def MAPE(testdata, predictdata):
    test_new = []
    predict_new = []
    for k in range(len(testdata)):
        if testdata[k] != 0:
            test_new.append(testdata[k])
            predict_new.append(predictdata[k])
    y_true, y_pred = np.array(test_new), np.array(predict_new)

    n = len(y_pred)
    sum = 0
    for i in range(n):
        sum += np.abs((y_pred[i] - y_true[i]) / y_true[i])
    sum *= 100
    sum = sum / n
    return sum


class Ui_Form(object):
    # 设置主界面区域布局
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1240, 619)
        self.textEdit_read = QtWidgets.QTextEdit(Form)
        self.textEdit_read.setGeometry(QtCore.QRect(240, 30, 441, 321))
        self.textEdit_read.setObjectName("textEdit_read")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 221, 281))
        self.groupBox.setObjectName("groupBox")
        self.comboBox_com = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_com.setGeometry(QtCore.QRect(118, 30, 81, 24))
        self.comboBox_com.setObjectName("comboBox_com")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(30, 30, 61, 21))
        self.label.setObjectName("label")
        self.pushButton_open_kou = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_open_kou.setEnabled(True)
        self.pushButton_open_kou.setGeometry(QtCore.QRect(50, 240, 112, 34))
        self.pushButton_open_kou.setObjectName("pushButton_open_kou")
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(10, 70, 111, 31))
        self.label_8.setObjectName("label_8")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(20, 110, 131, 31))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(20, 170, 131, 31))
        self.label_5.setObjectName("label_5")
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(20, 140, 131, 31))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.groupBox)
        self.label_10.setGeometry(QtCore.QRect(20, 200, 131, 31))
        self.label_10.setObjectName("label_10")
        self.textEdit_pinlv = QtWidgets.QTextEdit(self.groupBox)
        self.textEdit_pinlv.setGeometry(QtCore.QRect(120, 70, 71, 31))
        self.textEdit_pinlv.setObjectName("textEdit_pinlv")
        self.textEdit_write = QtWidgets.QTextEdit(Form)
        self.textEdit_write.setGeometry(QtCore.QRect(240, 390, 16, 141))
        self.textEdit_write.setObjectName("textEdit_write")
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 310, 221, 301))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_push_txt = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_push_txt.setGeometry(QtCore.QRect(40, 200, 112, 34))
        self.pushButton_push_txt.setObjectName("pushButton_push_txt")
        self.pushButton_stop_txt = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_stop_txt.setGeometry(QtCore.QRect(40, 250, 112, 34))
        self.pushButton_stop_txt.setObjectName("pushButton_stop_txt")
        self.pushButton_push_txt_lujing = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_push_txt_lujing.setGeometry(QtCore.QRect(10, 30, 171, 34))
        self.pushButton_push_txt_lujing.setObjectName("pushButton_push_txt_lujing")
        self.pushButton_save_txt = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_save_txt.setGeometry(QtCore.QRect(10, 110, 171, 34))
        self.pushButton_save_txt.setObjectName("pushButton_save_txt")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit.setGeometry(QtCore.QRect(20, 70, 161, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_savetxt = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_savetxt.setGeometry(QtCore.QRect(20, 150, 161, 31))
        self.lineEdit_savetxt.setObjectName("lineEdit_savetxt")
        self.textEdit_photo = QtWidgets.QTextEdit(Form)
        # 创建一个网格布局
        self.main_layout = QtWidgets.QGridLayout()
        # 设置主部件的布局为网格
        self.textEdit_photo.setLayout(self.main_layout)
        # 实例化一个widget部件作为K线图部件
        self.plot_widget = QtWidgets.QWidget()
        # 实例化一个网格布局层
        self.plot_layout = QtWidgets.QGridLayout()
        # 设置K线图部件的布局层
        self.plot_widget.setLayout(self.plot_layout)
        # 实例化一个绘图部件
        self.plot_plt = pg.PlotWidget(background='w')
        # 定义绘图界面图标显示
        self.plot_plt.addLegend()
        self.P1 = self.plot_plt.plot(pen=pg.mkPen(color='r'), name="真实值")
        self.P2 = self.plot_plt.plot(pen=pg.mkPen(color='b'), name="预测值")
        # 显示界面图形网格
        self.plot_plt.showGrid(x=True, y=True)
        # 定义绘图界面坐标轴 X轴
        # xax = self.plot_plt.getAxis('bottom')
        # x = (0,50,100,150,200,250,300,350,400)
        # strs = ['0', '50', '100', '150', '200', '250', '300', '350','400']
        # ticks = [[i, j] for i, j in zip(x, strs)]
        # xax.setTicks([ticks])
        # # 定义绘图界面坐标轴 Y轴
        # xax1 = self.plot_plt.getAxis('left')
        # x1 = (-3,-2,-1,0,1,2,3)
        # strs1 = ['-3.0','-2.0','-1.0','0.0','1.0','2.0','3.0']
        # ticks1 = [[i, j] for i, j in zip(x1, strs1)]
        # xax1.setTicks([ticks1])
        # 添加绘图部件到K线图部件的网格布局层
        self.plot_layout.addWidget(self.plot_plt)
        self.main_layout.addWidget(self.plot_widget, 1, 0, 3, 3)
        # 设置X轴显示范围
        self.plot_plt.setYRange(max=4, min=-4)
        # 设置Y轴显示范围
        self.plot_plt.setXRange(max=1000, min=0)
        # 设置绘图界面在主界面的显示范围
        self.textEdit_photo.setGeometry(QtCore.QRect(700, 30, 561, 491))
        self.textEdit_photo.setObjectName("textEdit_photo")
        self.pushButton_clearin = QtWidgets.QPushButton(Form)
        self.pushButton_clearin.setGeometry(QtCore.QRect(270, 560, 131, 34))
        self.pushButton_clearin.setObjectName("pushButton_clearin")
        self.pushButton_clearout = QtWidgets.QPushButton(Form)
        self.pushButton_clearout.setGeometry(QtCore.QRect(480, 560, 131, 34))
        self.pushButton_clearout.setObjectName("pushButton_clearout")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(730, 560, 131, 18))
        self.label_3.setObjectName("label_3")
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setGeometry(QtCore.QRect(250, 0, 71, 31))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setGeometry(QtCore.QRect(250, 350, 71, 31))
        self.label_7.setObjectName("label_7")
        self.label_11 = QtWidgets.QLabel(Form)
        self.label_11.setGeometry(QtCore.QRect(720, 0, 121, 31))
        self.label_11.setObjectName("label_11")
        self.lineEdit_2 = QtWidgets.QLineEdit(Form)
        self.lineEdit_2.setGeometry(QtCore.QRect(850, 550, 171, 31))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(1050, 550, 112, 34))
        self.pushButton.setObjectName("pushButton")

        self.label_MSE = QtWidgets.QLabel(Form)
        self.label_MSE.setGeometry(QtCore.QRect(270, 400, 61, 31))
        # font = QtGui.QFont()
        # font.setPointSize(14)
        # self.label_MSE.setFont(font)
        self.label_MSE.setObjectName("label_MSE")
        self.label_MAE = QtWidgets.QLabel(Form)
        self.label_MAE.setGeometry(QtCore.QRect(460, 400, 61, 31))
        # font = QtGui.QFont()
        # font.setPointSize(14)
        # self.label_MAE.setFont(font)
        self.label_MAE.setObjectName("label_MAE")
        self.label_RMSE = QtWidgets.QLabel(Form)
        self.label_RMSE.setGeometry(QtCore.QRect(270, 480, 61, 31))
        # font = QtGui.QFont()
        # font.setPointSize(14)
        # self.label_RMSE.setFont(font)
        self.label_RMSE.setObjectName("label_RMSE")
        self.label_R2 = QtWidgets.QLabel(Form)
        self.label_R2.setGeometry(QtCore.QRect(470, 480, 51, 31))
        # font = QtGui.QFont()
        # font.setPointSize(14)
        # self.label_R2.setFont(font)
        self.label_R2.setObjectName("label_R2")
        self.lineEdit_MSE = QtWidgets.QLineEdit(Form)
        self.lineEdit_MSE.setGeometry(QtCore.QRect(340, 400, 111, 31))
        self.lineEdit_MSE.setObjectName("lineEdit_MSE")
        self.lineEdit_4 = QtWidgets.QLineEdit(Form)
        self.lineEdit_4.setGeometry(QtCore.QRect(520, 480, 113, 31))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_RMSE = QtWidgets.QLineEdit(Form)
        self.lineEdit_RMSE.setGeometry(QtCore.QRect(340, 480, 113, 31))
        self.lineEdit_RMSE.setObjectName("lineEdit_RMSE")
        self.lineEdit_MAE = QtWidgets.QLineEdit(Form)
        self.lineEdit_MAE.setGeometry(QtCore.QRect(520, 400, 113, 31))
        self.lineEdit_MAE.setObjectName("lineEdit_MAE")



        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    #按钮界面设计
    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        # 设置主界面窗口名称
        Form.setWindowTitle(_translate("Form", "Form"))
        # 设置主界面按键显示
        self.groupBox.setTitle(_translate("Form", "串口设置"))
        self.label.setText(_translate("Form", "串口号"))
        self.pushButton_open_kou.setText(_translate("Form", "打开串口"))
        self.label_8.setText(_translate("Form", "发送频率(ms)"))
        self.label_4.setText(_translate("Form", "波特率：115200"))
        self.label_5.setText(_translate("Form", "数据位：8"))
        self.label_9.setText(_translate("Form", "停止位：1"))
        self.label_10.setText(_translate("Form", "校验位：None"))
        self.groupBox_2.setTitle(_translate("Form", "发送接收设置"))
        self.pushButton_push_txt.setText(_translate("Form", "发送"))
        self.pushButton_stop_txt.setText(_translate("Form", "停止发送"))
        self.pushButton_push_txt_lujing.setText(_translate("Form", "选择发送文件路径"))
        self.pushButton_save_txt.setText(_translate("Form", "选择保存文件路径"))
        self.pushButton_clearin.setText(_translate("Form", "清空输入窗口"))
        self.pushButton_clearout.setText(_translate("Form", "清空输出窗口"))
        self.label_3.setText(_translate("Form", "模型加载情况："))
        self.label_6.setText(_translate("Form", "数据显示"))
        self.label_7.setText(_translate("Form", "数据发送"))
        self.label_11.setText(_translate("Form", "数据图形显示"))
        self.pushButton.setText(_translate("Form", "加载模型"))

        self.label_MSE.setText(_translate("Form", "MSE:"))
        self.label_MAE.setText(_translate("Form", "MAE:"))
        self.label_RMSE.setText(_translate("Form", "MAPE:"))
        self.label_R2.setText(_translate("Form", "RMSE:"))
#  子线程模块，用来实现串口数据接收和发送
class Qthread_fun(QObject):
    #子线程实现串口数据接收
    signal_start_fun       = pyqtSignal()
    #打开串口操作
    signal_open_kou        = pyqtSignal(object)
    #已打开串口
    signal_open_kou_flage  = pyqtSignal(object)
    #串口读取数据
    signal_readdata        = pyqtSignal(object)
    #串口发送数据
    signal_send_txt        = pyqtSignal(object)

    #初始化
    def __init__(self,parent = None):
        # 0串口未打开，1已打开，2关闭串口
        super(Qthread_fun,self).__init__(parent)
        #主线程IP输出
        print("初始化", threading.current_thread().ident)
        self.state = 0

    #连接串口操作
    def slot_open_kou(self,parameter):
        print("按下串口按钮",parameter)
        if self.state == 0:
            self.Serial.setPortName(parameter['comboBox_com'])#串口号（自动搜索）
            self.Serial.setBaudRate(115200)#波特率
            self.Serial.setStopBits(1)#停止位
            self.Serial.setDataBits(8)#数据位
            self.Serial.setParity(0)#校验位
            if self.Serial.open(QSerialPort.ReadWrite) == True:
                print("打开成功")
                self.state = 1
                self.signal_open_kou_flage.emit(self.state)
            else:
                print("打开失败")
                self.signal_open_kou_flage.emit(0)
        else:
            self.state = 0
            self.Serial.close()
            print("串口关闭")
            self.signal_open_kou_flage.emit(2)

    #串口发送文本
    def slot_send_txt(self, send_data):
        if self.state != 1:  # 串口未开启
            return
        #print("发送数据状态", self.state)
        Byte_data = str.encode(send_data)
        self.Serial.write(Byte_data)
        self.asd = self.Serial.readLine()

    #读取TX2端发串口发送过来的数据，并连接到下一个槽
    def Serial_receive_data(self):
        data = self.Serial.readLine()
        self.signal_readdata.emit(data)

    #子线程
    def qtheard_fun(self):
        sleep(1)
        self.Serial = QSerialPort()
        print("运行线程", threading.current_thread().ident)
        self.Serial.readyRead.connect(self.Serial_receive_data)

#主线程
class MyMainForm(QWidget):
    def __init__(self):
        self.data_list = []
        self.realdata_list = []
        super().__init__()
        self.model_over = False
        self.model_over_temp = False
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.UI_Init()
        # 0表示word ，1表示txt
        self.push_txt_word = 0
        self.send_txt = ''
        self.save_txt = ''
        self.sata1 = 0
        self.set_parameter = {}
        self.set_parameter['pinlv'] = 0
        self.choose_send_txt()
        self.choose_save_txt()
        self.Interface_init()
        self.Serial_Qthread = QThread()
        self.Qthread_fun = Qthread_fun()
        self.Qthread_fun.moveToThread(self.Serial_Qthread)
        self.Serial_Qthread.start()
        self.Qthread_fun.signal_start_fun.connect(self.Qthread_fun.qtheard_fun)
        self.Qthread_fun.signal_start_fun.emit()
        self.Qthread_fun.signal_open_kou.connect(self.Qthread_fun.slot_open_kou)
        self.Qthread_fun.signal_open_kou_flage.connect(self.slot_open_kou_flage)
        self.Qthread_fun.signal_readdata.connect(self.slot_readdata)
        self.Qthread_fun.signal_send_txt.connect(self.Qthread_fun.slot_send_txt)
        self.port_name = []
        self.time_scan = QTimer()
        self.time_scan.timeout.connect(self.TimeOut_Scan)
        # 设置定时器时间间隔为1s
        self.time_scan.start(1000)
        self.time_send = QTimer()
        self.time_send.timeout.connect(self.TimeOut_Send)

    def TimeOut_Scan(self):
        availablePort = QSerialPortInfo.availablePorts()
        new_port = []
        for port in availablePort:
            new_port.append(port.portName())
        if len(self.port_name) != len(new_port):
            self.port_name = new_port
            self.ui.comboBox_com.clear()
            self.ui.comboBox_com.addItems(self.port_name)

    def Interface_init(self):
        self.ui.pushButton_push_txt.clicked.connect(self.pushButton_push_txt)
        self.ui.pushButton_clearin.clicked.connect(self.pushButton_clearin)
        self.ui.pushButton_clearout.clicked.connect(self.pushButton_clearout)
        self.ui.pushButton_stop_txt.clicked.connect(self.pushButton_stop)
        self.ui.pushButton.clicked.connect(self.pushButton_model)

    def UI_Init(self):
        self.ui.pushButton_open_kou.clicked.connect(self.push_open_kou)

    def choose_send_txt(self):
        self.ui.pushButton_push_txt_lujing.clicked.connect(self.choose_send_txt1)

    def choose_save_txt(self):
        self.ui.pushButton_save_txt.clicked.connect(self.choose_save_txt1)

    def choose_send_txt1(self):
        dir = QFileDialog()
        dir.setNameFilter("文本文件(*.txt)")
        # 判断是否选择了文件
        if dir.exec_():
            self.ui.lineEdit.setText(dir.selectedFiles()[0])
        self.send_txt = dir.selectedFiles()[0]
        print(self.send_txt)

    def pushButton_stop(self):
        self.time_send.stop()
        self.realdata_list = []
        self.data_list = []
        self.model_over_temp = False
        self.ui.textEdit_read.insertPlainText('The prediction model is loading...\n')
        self.ui.lineEdit_2.setText('Loading')

    def pushButton_clearin(self):
        print('清除输入界面')
        self.ui.textEdit_write.clear()

    def pushButton_model(self):#加载模型
        if self.model_over_temp == False:
            self.Qthread_fun.signal_send_txt.emit('model\n')
            qw.QMessageBox.warning(self, "提示信息", "正在加载中...")
        else:
            qw.QMessageBox.warning(self, "提示信息", "模型已重新加载！")
            self.Qthread_fun.signal_send_txt.emit('model\n')

    def pushButton_clearout(self):
        print('清除输出界面')
        # self.ui.lineEdit_MSE.setText('Loading complete！')
        self.ui.textEdit_read.clear()

    def choose_save_txt1(self):
        dir1 = QFileDialog()
        dir1.setNameFilter("文本文件(*.txt)")
        if dir1.exec_():
            self.ui.lineEdit_savetxt.setText(dir1.selectedFiles()[0])
        self.save_txt = dir1.selectedFiles()[0]
        print(self.save_txt)
        if self.save_txt != '':
            self.CMD_save = open(self.save_txt, 'a+')

    def push_open_kou(self):
        self.set_parameter['comboBox_com'] = self.ui.comboBox_com.currentText()
        self.set_parameter['pinlv'] = self.ui.textEdit_pinlv.toPlainText()
        self.Qthread_fun.signal_open_kou.emit(self.set_parameter)

    def slot_open_kou_flage(self,sate):
        print("串口打开状态", sate)
        self.sata1 = sate
        if sate == 0:
            qw.QMessageBox.warning(self,"错误信息","串口被占用，打开失败")
        elif sate == 1:
            self.ui.pushButton_open_kou.setStyleSheet("color:red")
            self.ui.pushButton_open_kou.setText("关闭串口")
            qw.QMessageBox.warning(self, "提示信息", "串口已打开")
            if (self.model_over == False) and (self.model_over_temp == False):
                self.ui.textEdit_read.insertPlainText('The prediction model is loading...\n')
            self.time_scan.stop()
            if self.model_over_temp == False:
                self.ui.lineEdit_2.clear()
                self.ui.lineEdit_2.setText('Loading')
        else:
            self.ui.pushButton_open_kou.setStyleSheet("color:black")
            self.ui.pushButton_open_kou.setText("打开串口")
            qw.QMessageBox.warning(self, "提示信息", "串口已关闭")
            self.time_scan.start(1000)


    def slot_readdata(self,data):
        Byte_data = bytes(data)
        print(data)
        Byte_data1 = Byte_data.decode('utf-8', 'ignore')
        if self.model_over_temp == True:
            time_now = str(time.strftime('%H:%M:%S', time.localtime(time.time())))
            # self.ui.textEdit_read.insertPlainText(str(time_now)+'  ')


            if self.push_txt_word == 1:  # 发送文件时画图  ————预测值
                data1 = float((str(data)[2:-3]).split(" ")[1])    #i+1时刻的预测值
                data2 = float((str(data)[2:-3]).split(" ")[0])    #i时刻真实值
                data3 = (str(data)[2:-3]).split(" ")[2]  #time
                self.ui.textEdit_read.insertPlainText(str(time_now)+' pre:'+str(data1)+" need time:"+data3+'\n')
                y_true.append(data2)
                if (len(y_pre) == 0):
                    y_pre.append(data2)
                y_pre.append(data1)
                y_pre1 = np.array(y_pre[0:len(y_pre)-1], dtype='float64')
                y_true1 = np.array(y_true, dtype='float64')
                MSE = round(mse( y_true1,y_pre1),5)
                MAPE1 = round(MAPE(y_true1,y_pre1),5)
                MAE = round(mae(y_true1,y_pre1),5)
                R2 = round(rmse( y_true1,y_pre1),5)
                self.ui.lineEdit_MSE.clear()
                self.ui.lineEdit_RMSE.clear()
                self.ui.lineEdit_MAE.clear()
                self.ui.lineEdit_4.clear()
                self.ui.lineEdit_MSE.setText(str(MSE))
                self.ui.lineEdit_RMSE.setText(str(MAPE1))
                self.ui.lineEdit_MAE.setText(str(MAE))
                self.ui.lineEdit_4.setText(str(R2))


                self.data_list.append(data1)
                self.P2 = self.ui.plot_plt.plot()
                # 在主界面更新预测值绘图
                self.P2.setData(self.data_list, pen=pg.mkPen(color='b',width=3))
            if self.save_txt != '':
                # 保存预测值数值
                self.CMD_save.write(str(data1) + '\n')
        self.model_over = "over" in Byte_data1
        if (self.model_over == True) and (self.model_over_temp == False):   # 模型加载完成
            self.ui.textEdit_read.clear()
            self.ui.textEdit_read.insertPlainText('Forecast model loading completed!\n')
            self.model_over_temp =True
            self.ui.lineEdit_2.clear()
            self.ui.lineEdit_2.setText('Loading complete！')

    def pushButton_push_txt(self):
        print("点击发送按钮")
        print(self.model_over_temp)
        self.set_parameter['pinlv'] = self.ui.textEdit_pinlv.toPlainText()
        if self.sata1 != 1:
            qw.QMessageBox.warning(self, "提示信息", "串口未打开")
            return
        if self.model_over_temp == False:
            qw.QMessageBox.warning(self, "提示信息", "模型未加载")
            return
        send_write =self.ui.textEdit_write.toPlainText()
        # 输入框有数据发送输入框的数据，没有数据发送文件
        if send_write != '':  # 发送输入框
            self.Qthread_fun.signal_send_txt.emit(send_write+'\n')
            self.push_txt_word = 0
        else:   # 发送txt文件
            if self.send_txt != '':  # 发送文件已选取
                if self.set_parameter['pinlv'] == 0:  # 发送频率未选取
                    qw.QMessageBox.warning(self, "提示信息", "请输入发送时间间隔")
                    print("发送时间间隔未选定")
                    return
                else:    # 成功发送文件状态
                    self.ui.plot_plt.clearPlots()
                    self.realdata_list.clear()
                    self.data_list.clear()
                    self.push_txt_word = 1
                    time_send = int(self.set_parameter['pinlv'])
                    self.time_send.start(time_send)
                    self.CMD = open(self.send_txt,encoding='utf-8')
            else:  # 发送文件未选取
                qw.QMessageBox.warning(self, "提示信息", "未选取需要发送的文件")
                print("未选取需要发送的文件")
                return

    def TimeOut_Send(self):
        self.send_data =self.CMD.readline()
        if (self.send_data == ''):  # 判断文件读完
            self.time_send.stop()
            print("文件全部发送完成")
            return
        # 在绘图界面画真实值
        self.realdata_list.append(float(self.send_data))
        if len(self.realdata_list) > 9:
            self.P1= self.ui.plot_plt.plot()
            # 设置在绘图时从真实值列表第10位开始取值
            self.P1.setData(self.realdata_list[9:], pen=pg.mkPen(color='r',width=3))
        self.Qthread_fun.signal_send_txt.emit(self.send_data)


if __name__ == '__main__':
    y_pre = []
    y_true = []
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    myWin.show()
    sys.exit(app.exec_())


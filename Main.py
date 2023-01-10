from GUI import Ui_Form
import AcousticalParameters as AP
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QMessageBox, QTableWidgetItem
import soundfile as sf
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
import matplotlib.pyplot as plt
import pandas as pd


class GUI(QWidget):
    def __init__(self):
        ### Initial widgets state ###
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.mono.setChecked(True)
        self.ui.octave.setChecked(True)
        self.ui.schroeder.setChecked(True)
        self.ui.window.setEnabled(False)
        self.ui.window.setText('20')
        self.ui.label.setEnabled(False)
        self.ui.load_r_ir.setEnabled(False)
        self.ui.calculate.setEnabled(False)
        self.ui.L.setEnabled(False)
        self.ui.R.setEnabled(False)
        self.ui.export_2.setEnabled(False)
        
        
        ### Widgets functions connection ###
        self.ui.mono.clicked.connect(self.MONO)
        self.ui.stereo.clicked.connect(self.STEREO)
        self.ui.mmf.clicked.connect(self.MMF)
        self.ui.schroeder.clicked.connect(self.SCHROEDER)
        self.ui.stereo_split.clicked.connect(self.STEREO_SPLIT)
        self.ui.load_ir.clicked.connect(self.LOAD_IR)
        self.ui.load_r_ir.clicked.connect(self.LOAD_R_IR)
        self.ui.calculate.clicked.connect(self.CALCULATE)
        self.ui.export_2.clicked.connect(self.EXPORT)
        

        ### Graphics initial settings ###
        self.canvas1 = FigureCanvas(plt.Figure())
        self.ui.output_graphic.addWidget(self.canvas1)             
        self.canvas1.figure.subplots()
        self.canvas1.hide()
        
        
    ### Radio buttons functions ###    
    def MONO(self):
        self.ui.load_ir.setText("Load Mono IR")
        self.ui.load_r_ir.setEnabled(False)
        
        
    def STEREO(self):
        self.ui.load_ir.setText("Load Stereo IR")
        self.ui.load_r_ir.setEnabled(False)
    
    
    def STEREO_SPLIT(self):
        self.ui.load_ir.setText("Load L IR")
        self.ui.load_r_ir.setEnabled(True)
    
    
    def MMF(self):
        self.ui.window.setEnabled(True)
        self.ui.label.setEnabled(True)
       
        
    def SCHROEDER(self):
        self.ui.window.setEnabled(False)
        self.ui.label.setEnabled(False)
        
    
    def EXPORT(self):
        # if self.ui.mono.isChecked():
            csv_name = QFileDialog.getSaveFileName(self, "Save results","",filter="CSV Files (*.csv)")
            del self.results['ETC']
            del self.results['smooth']
            df = pd.DataFrame(self.results)
            df.index = self.columns
            df.to_csv(csv_name[0], sep=';')
            
    
    ### Push buttons functions ###
    def LOAD_IR(self):
        self.route = QFileDialog.getOpenFileName(filter='WAV Audio File (*wav)')[0]
        self.audio, self.fs = sf.read(self.route)
        
        ### Type error detection ###        
        if self.ui.mono.isChecked() and self.audio.ndim != 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Type Error")
            msg.setInformativeText('Mono IR must be loaded')
            msg.setWindowTitle("Error")
            msg.exec_()
        
        ### Type error detection ###              
        if self.ui.stereo.isChecked() and self.audio.ndim != 2:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Type Error")
            msg.setInformativeText('Stereo IR must be loaded')
            msg.setWindowTitle("Error")
            msg.exec_()

        ### Type error detection ###         
        if self.ui.stereo_split.isChecked() and self.audio.ndim != 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Type Error")
            msg.setInformativeText('Mono IR must be loaded')
            msg.setWindowTitle("Error")
            msg.exec_()
        
        ### Enable/disable "Calculate & Export" push buttons and write/hide file name ###
        if  self.ui.mono.isChecked() and self.audio.ndim == 1 or self.ui.stereo.isChecked() and self.audio.ndim == 2:
            self.ui.calculate.setEnabled(True)
            self.ui.L.setEnabled(True)
            self.ui.L.setText(self.route.split('/')[-1])

        
        elif  self.ui.stereo_split.isChecked() and self.audio.ndim == 1 :
            self.ui.L.setEnabled(True)
            self.ui.L.setText(self.route.split('/')[-1])
            
               
        else:
            self.ui.calculate.setEnabled(False)
            self.ui.L.setText('L:')
            self.ui.L.setEnabled(False)
            

        if self.ui.mono.isChecked() or self.ui.stereo.isChecked():
            self.ui.R.setText('R:')
            self.ui.R.setEnabled(False)


    def LOAD_R_IR(self):
        
        self.route3 = QFileDialog.getOpenFileName(filter='WAV Audio File (*wav)')[0]
        self.audio3, fs3 = sf.read(self.route3)

        ### Type error detection ###        
        if self.ui.stereo_split.isChecked() and self.audio3.ndim != 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Type Error")
            msg.setInformativeText('Mono IR must be loaded')
            msg.setWindowTitle("Error")
            msg.exec_()
      
        ### Enable/disable "Calculate" push button and write file name ###
        if self.ui.stereo_split.isChecked() and self.audio3.ndim == 1:
            self.ui.R.setEnabled(True)
            self.ui.R.setText(self.route3.split('/')[-1])
        
        if self.ui.stereo_split.isChecked() and self.audio3.ndim == 1 and self.audio.ndim != 0:
            self.ui.calculate.setEnabled(True)
            
        else:
            self.ui.calculate.setEnabled(False)
            self.ui.R.setText('R:')
            self.ui.R.setEnabled(False)
            
            
    def CALCULATE(self):
        
        ### Invalid window size detection ###
        if self.ui.mmf.isChecked() and self.ui.window.text() <= str(0):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Window Error")
            msg.setInformativeText('You must enter a valid window size')
            msg.setWindowTitle("Error")
            msg.exec_()
        
        ### Type error detection ###
        elif self.ui.mono.isChecked() and self.audio.ndim != 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Type Error")
            msg.setInformativeText('Mono IR must be loaded')
            msg.setWindowTitle("Error")
            msg.exec_()
            self.ui.calculate.setEnabled(False)
            self.ui.L.setText('L:')
            self.ui.L.setEnabled(False)
            
        ### Type error detection ###
        elif self.ui.stereo.isChecked() and self.audio.ndim != 2:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Type Error")
            msg.setInformativeText('Stereo IR must be loaded')
            msg.setWindowTitle("Error")
            msg.exec_()
            self.ui.calculate.setEnabled(False)
            self.ui.L.setText('L:')
            self.ui.L.setEnabled(False)
        
        ### Type error detection ###
        elif self.ui.stereo_split.isChecked() and self.audio3.ndim != 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Type Error")
            msg.setInformativeText('Mono IR must be loaded')
            msg.setWindowTitle("Error")
            msg.exec_()
            self.ui.calculate.setEnabled(False)
            self.ui.L.setText('L:')
            self.ui.L.setEnabled(False)
            self.ui.R.setText('R:')
            self.ui.R.setEnabled(False)
            
        ### Type error detection ##
        elif self.ui.stereo_split.isChecked() and self.audio3.ndim != self.audio.ndim:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Type Error")
            msg.setInformativeText('Both IR must be mono')
            msg.setWindowTitle("Error")
            msg.exec_()
            self.ui.calculate.setEnabled(False)
            self.ui.L.setText('L:')
            self.ui.L.setEnabled(False)
            self.ui.R.setText('R:')
            self.ui.R.setEnabled(False)
     
        else:
            self.Process()
   
            
    def Process(self):
        
        ### Filtering detection & table columns configuration ###
        if self.ui.octave.isChecked():
            octave = 0
            self.columns = ['31.5 Hz', '63 Hz', '125 Hz', '250 Hz', '500 Hz', '1000 Hz', 
                       '2000 Hz', '4000 Hz', '8000 Hz', '16000 Hz'] 
            self.ui.tableWidget.setColumnCount(10)
            self.ui.tableWidget.setHorizontalHeaderLabels(self.columns)
        else:
            octave = 1
            self.columns = ("25 Hz", "31.5 Hz", "40 Hz", "50 Hz", "63 Hz", "80 Hz", "100 Hz", 
                       "125 Hz", "160 Hz","200 Hz", "250 Hz", "315 Hz", "400 Hz", " Hz500",
                       "630 Hz", "800 Hz", "1 kHz","1.3 kHz", "1.6 kHz", "2 kHz", "2.5 kHz", 
                       "3.2 kHz", "4 kHz", "5 kHz", "6.3 kHz", "8 kHz", "10 kHz", "12.5 kHz", 
                       "16 kHz", "20 kHz") 
            self.ui.tableWidget.setColumnCount(30)
            self.ui.tableWidget.setHorizontalHeaderLabels(self.columns)
        
        ### Smoothing detection ###   
        if self.ui.schroeder.isChecked():
            schroeder = 0
        else:
            schroeder = 1


        ### Mono parameters proccessing ###
            ### Table filling ###
        if self.ui.mono.isChecked():
            self.ui.tableWidget.setRowCount(7)     
            rows = ('Tt [s]', 'EDTt [s]', 'C50 [dB]','C80 [dB]','EDT [s]','T20 [s]','T30 [s]')
            self.ui.tableWidget.setVerticalHeaderLabels(rows) 
            mmf_window = int(self.ui.window.text())
            self.results = AP.MonoParam(self.audio, self.fs, octave, schroeder, mmf_window)
            j = 0
            c = np.array(['Tt', 'EDTt', 'C50','C80','EDT','T20','T30'])
            for i in range(0, len(c)):
                for reg in self.results[c[i]]:
                    cell = QTableWidgetItem(str(reg))
                    self.ui.tableWidget.setItem(0, j, cell)
                    j += 1
            
            ### Output Graphic ###
            ETC = self.results['ETC']
            smooth = self.results['smooth']
            a = np.arange(0,len(ETC)/self.fs,1/self.fs)
            t = a[:len(ETC)]
            ax = self.canvas1.figure.axes[0]
            ax.cla()
            ax.plot(t, ETC, label='Energy')
            if self.ui.mmf.isChecked():
                ax.plot(t, smooth, label='MMF')
            else:
                ax.plot(t, smooth, label='Schroeder')
            try:
                xlim_max = int(np.where(smooth <= -80)[0][0] * 1.1)
                if xlim_max > t.size:
                    xlim_max = t.size-1
            except:
                xlim_max = t.size-1
            ax.set(xlabel='Time [s]', ylabel='Energy [dB]',
                    xlim=(0, t[xlim_max]), ylim=(-100, max(ETC)))
            ax.legend(loc=1)
            ax.figure.tight_layout(pad=0.1)
            self.canvas1.draw()
            self.canvas1.show()
            self.ui.export_2.setEnabled(True)
            
                    
        ### Stereo parameters proccessing ###
            ### Table filling ###
        if self.ui.stereo.isChecked():
            self.ui.tableWidget.setRowCount(8)     
            rows = ('Tt [s]', 'EDTt [s]', 'C50 [dB]','C80 [dB]','EDT [s]','T20 [s]','T30 [s]', 'IACCe')
            self.ui.tableWidget.setVerticalHeaderLabels(rows)
            mmf_window = int(self.ui.window.text())
            self.results = AP.StereoParam(self.audio, self.fs, octave, schroeder, mmf_window)
            j = 0
            c = np.array(['Tt', 'EDTt', 'C50','C80','EDT','T20','T30', 'IACCe'])
            for i in range(0, len(c)):
                for reg in self.results[c[i]]:
                    cell = QTableWidgetItem(str(reg))
                    self.ui.tableWidget.setItem(0, j, cell)
                    j += 1
                    
            ### Output Graphic ###
            ETC = self.results['ETC'][0]
            smooth = self.results['smooth'][0]
            a = np.arange(0,len(ETC)/self.fs,1/self.fs)
            t = a[:len(ETC)]
            ax = self.canvas1.figure.axes[0]
            ax.cla()
            ax.plot(t, ETC, label='Energy')
            if self.ui.mmf.isChecked():
                ax.plot(t, smooth, label='MMF')
            else:
                ax.plot(t, smooth, label='Schroeder')
            try:
                xlim_max = int(np.where(smooth <= -80)[0][0] * 1.1)
                if xlim_max > t.size:
                    xlim_max = t.size-1
            except:
                xlim_max = t.size-1
            ax.set(xlabel='Time [s]', ylabel='Energy [dB]',
                    xlim=(0, t[xlim_max]), ylim=(-100, max(ETC)))
            ax.legend(loc=1)
            ax.figure.tight_layout(pad=0.1)
            self.canvas1.draw()
            self.canvas1.show()
            self.ui.export_2.setEnabled(True)                    
                    
            
        ### Stereo split parameters proccessing ###    
        if self.ui.stereo_split.isChecked():
            self.ui.tableWidget.setRowCount(8)     
            rows = ('Tt [s]', 'EDTt [s]', 'C50 [dB]','C80 [dB]','EDT [s]','T20 [s]','T30 [s]', 'IACCe')
            self.ui.tableWidget.setVerticalHeaderLabels(rows)
            mmf_window = int(self.ui.window.text())
            self.results = AP.StereoSplitParam(self.audio, self.audio3, self.fs, octave, schroeder, mmf_window)
            j = 0
            c = np.array(['Tt', 'EDTt', 'C50','C80','EDT','T20','T30', 'IACCe'])
            for i in range(0, len(c)):
                for reg in self.results[c[i]]:
                    cell = QTableWidgetItem(str(reg))
                    self.ui.tableWidget.setItem(0, j, cell)
                    j += 1
            
            ### Output Graphic ###
            ETC = self.results['ETC'][0]
            smooth = self.results['smooth'][0]
            a = np.arange(0,len(ETC)/self.fs,1/self.fs)
            t = a[:len(ETC)]
            ax = self.canvas1.figure.axes[0]
            ax.cla()
            ax.plot(t, ETC, label='Energy')
            if self.ui.mmf.isChecked():
                ax.plot(t, smooth, label='MMF')
            else:
                ax.plot(t, smooth, label='Schroeder')
            try:
                xlim_max = int(np.where(smooth <= -80)[0][0] * 1.1)
                if xlim_max > t.size:
                    xlim_max = t.size-1
            except:
                xlim_max = t.size-1
            ax.set(xlabel='Time [s]', ylabel='Energy [dB]',
                    xlim=(0, t[xlim_max]), ylim=(-100, max(ETC)))
            ax.legend(loc=1)
            ax.figure.tight_layout(pad=0.1)
            self.canvas1.draw()
            self.canvas1.show()
            self.ui.export_2.setEnabled(True)       
    

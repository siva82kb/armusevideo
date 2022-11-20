"""Arm Use Video Viewer: Module to display video data from different sources
for the Armu Recorder.

Author: Sivakumar Balasubramanian
Date: 24 June 2022
email: siva82kb@gmail.com
"""
from email.policy import default
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import (QEvent, Qt, pyqtSignal, pyqtSlot, Qt,
                          QThread)
from PyQt5.QtWidgets import (QWidget, QApplication, QLabel)
import sys
import cv2
import enum
import numpy as np
from datetime import datetime as dt
from attrdict import AttrDict
import msgpack
import msgpack_numpy as  msgpacknp
import temporenc
import struct

sys.path.append("scripts")
from support import encode_datetime

from pymf import get_MF_devices

import logging
import logging.config
log = logging.getLogger(__name__)


class WebcamVideoThread(QThread):
    """Class to read webcam data on  thread.
    """
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, index):
        super().__init__()
        self._run_flag = True
        self.index = index
        self._cap = None
        self.imgtime = None
        log.info("Started Webcam video thread.")

    def img_sz(self):
        if self._cap is not None:
            return (self._cap.get(3), self._cap.get(4))
        else:
            return None
    
    def run(self):
        # capture from web cam
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        # self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while self._run_flag:
            ret, cv_img = self._cap.read()
            if ret:
                self.imgtime = dt.now()
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        self._cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        log.info("Stopped Webcam video thread.")
        self.wait()


class ParkourVideoViewer(QWidget):
    """ARMU Video Viewer widget.
    """
    devices = get_MF_devices()
    
    @staticmethod
    def get_device_index(dev):
        for i, _dev in enumerate(ParkourVideoViewer.devices):
            if _dev == dev:
                return i
        return -1
    
    close_event = pyqtSignal()
    keypress_event = pyqtSignal(QEvent)
    
    def __init__(self, dev):
        super().__init__()
        self.setWindowTitle("Armu Video Viewer")
                
        # Display options.
        self.dev = dev
        self.index = ParkourVideoViewer.get_device_index(self.dev)
        
        # Assign the size 
        self.img_w = 3 * 640 // 2
        self.img_h = 3 * 480 // 2
        
        # Set fixed size.
        self.setFixedSize(self.img_w, self.img_h)
        
        # Create controls
        self._create_controls()
        # Disable loggin.
        self._currtaskname = ""
        self._filename = None
        self._fhandle = None
        self._video = None
        
        # create the video capture thread
        self.thread = WebcamVideoThread(self.index)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self._sig_cb_update_image)
        # start the thread
        self.thread.start()
    
    def keyPressEvent(self, event):
        # Toggle recording
        print(event.key())
        if event.key() == 82 or event.key() == 114:
            self.filename = "test"
    
    @property
    def filename(self):
        return self._filename
    
    @filename.setter
    def filename(self, value):
        # Check if the value is changed.
        if self._filename != value:
            # Close the existing file.
            if self._filename is not None:
                self._video.release()
                self._fhandle.close()

            # Check the new value.
            self._filename = value
            
            # Write header details.
            if self._filename is not None:
                # Create file writers
                self._video = cv2.VideoWriter(f'{self._filename}.avi',
                                              cv2.VideoWriter_fourcc('M', 'J',
                                                                     'P', 'G'),
                                              30, (self.img_w, self.img_h))
                self._fhandle = (open(f"{self._filename}.vidt", "wb")
                                 if self._filename is not None
                                 else None)
                log.info(f"Started video logging. {self._filename}.avi, {self._filename}.vidt.")
            else:
                log.info(f"Stopped video logging.")
                

    def closeEvent(self, event):
        self.filename = None
        self.thread.stop()
        event.accept()

    def _create_controls(self):
        # create the label that holds the image
        self.lbl_img = QLabel(self)
        self.lbl_img.resize(self.img_w, self.img_h)
        self.lbl_img.move(0, 0)
        self.lbl_img.setAlignment(Qt.AlignCenter)
        
        # create a text label for message
        self.lbl_message = QLabel(self)
        self.lbl_message.setFont(QtGui.QFont("Arial", 12,
                                               weight=QtGui.QFont.Bold))
        self.lbl_message.setText("")
        self.lbl_message.resize(self.img_w, 20)
        self.lbl_message.move(5, 5)
        self.lbl_message.raise_()
        
        # create a text label current task details
        self.lbl_curr_task = QLabel(self)
        self.lbl_curr_task.setFont(QtGui.QFont("Arial", 14))
        self.lbl_curr_task.setText("")
        self.lbl_curr_task.resize(self.img_w, 20)
        self.lbl_curr_task.move(5, 30)
        self.lbl_curr_task.raise_()

    @pyqtSlot(np.ndarray)
    def _sig_cb_update_image(self, cv_img):
        """Updates the image_label with a new opencv image.
        """
        self.lbl_img.setPixmap(self.convert_cv_qt(cv_img,
                                                  self.img_w,
                                                  self.img_h))
        # Is the video to be logged?
        if self.filename is not None:
            # Write current frame
            self._video.write(cv_img)
            # Write time
            self._fhandle.write(temporenc.packb(dt.now()))
    
    def convert_cv_qt(self, img, img_w, img_h):
        """Convert from an opencv image to QPixmap.
        """
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h,
                                            bytes_per_line,
                                            QtGui.QImage.Format_RGB888)
        p1 = convert_to_Qt_format.scaled(img_w, img_h,
                                         Qt.KeepAspectRatio)
        return QPixmap.fromImage(p1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = ParkourVideoViewer("USB Video Device")
    a.show()
    sys.exit(app.exec_())
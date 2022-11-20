"""Parkour Recorder: Software for recording video for the parkour protocol
used at UQ.

Author: Sivakumar Balasubramanian
Date: 07 Oct 2022
email: siva82kb@gmail.com
"""

import sys
import os
from PyQt5 import (
    QtGui,
    QtWidgets,
)
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import (
    Qt,
    QTimer,
    pyqtSlot, 
)
from PyQt5.QtWidgets import (
    QMessageBox
)
import numpy as np
from PyQt5.QtGui import QPixmap, QFont
import cv2
from attrdict import AttrDict
from datetime import datetime as dt
import enum
import json
import random
import temporenc

from parkour_ui import Ui_ParkourRecorder
from parkour_video_viewer import WebcamVideoThread, ParkourVideoViewer
from pymf import get_MF_devices


class ParkourRecorderStates(enum.Enum):
    Start = 0
    WaitingToStart = 1
    WaitingToRecordTask = 2
    RecordingTask = 3
    WaitingForAllDone = 4
    AllDone = 5


class ParkourRecorder(QtWidgets.QMainWindow, Ui_ParkourRecorder):
    """Main window of the data recorder.
    """

    def __init__(self, *args, **kwargs) -> None:
        """View initializer."""
        super(ParkourRecorder, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # Fix size.
        # self.setFixedSize(self.geometry().width(),
        #                   self.geometry().height())
        self.showMaximized()

        # State callbacks.
        self._state_ui_handlers = {
            ParkourRecorderStates.WaitingToStart: self._handle_waiting_to_start,
            ParkourRecorderStates.WaitingToRecordTask: self._handle_waiting_to_record_task,
            ParkourRecorderStates.RecordingTask: self._handle_recording_task,
            ParkourRecorderStates.WaitingForAllDone: self._handle_waiting_for_all_done,
            ParkourRecorderStates.AllDone: self._handle_all_done,
        }

        # Program state
        self._state : ParkourRecorderStates = ParkourRecorderStates.WaitingToStart

        # Current time
        self.currtime = None
        self.imgdispcount = 0

        # Experiment vdata.
        self._create_expt_vars()

        # Start camera thread.
        self._start_camera()

        # A bunch of timers.
        self._init_timers()

        # Update controls.
        self.update_ui()        

        # Add callbacks
        self.attach_callbacks()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if self._state != value:
            self._state = value
            sys.stdout.write("\n" + f"State set to {self._state}")

    # ############### #
    # Other Functions #
    # ############### #
    def _init_timers(self): 
        self.timer = QTimer()
        self.timer.timeout.connect(self._callback_app_timer)
        self.timer.start(1000)

    def _start_camera(self):
        self.camthread = WebcamVideoThread(self.expt.params['camera index'])
        self.camthread.change_pixmap_signal.connect(self._sig_cb_update_image)
        self.camthread.start()

    def _start_logging(self, task=False):
        # Check if the fname is changed.
        # Close the existing file.
        if self.expt.recording is True:
            self.expt.vidwriter.release()
            self.expt.vidtfhndl.close()
            self.expt.recording = False
            self.expt.basefname = None

        # Check the new fname.
        _ct = int(dt.now().timestamp() * 1e6)
        self.expt.basefname = f"{self.expt.outdir}{os.sep}"
        if task:
            # Task data
            self.expt.basefname += f"{self.expt.params['taskprefix']}_{self.expt.currtaskindx:03d}_{_ct}"
            self.expt.datafiles += ((self.expt.basefname,
                                     self.expt.tasks[self.expt.currtaskindx][0]),)
        else:
            # Non-task data
            self.expt.basefname += f"{self.expt.params['nontaskprefix']}_x_{_ct}"
            self.expt.datafiles += ((self.expt.basefname, ''),)

        # Write header details.
        # Create file writers        
        img_w = int(self.camthread.img_sz()[0])
        img_h = int(self.camthread.img_sz()[1])
        self.expt.vidwriter = cv2.VideoWriter(
            f'{self.expt.basefname}.avi',
            cv2.VideoWriter_fourcc('X','V','I','D'),
            10, (img_w, img_h)
        )
        self.expt.vidtfhndl = (open(f"{self.expt.basefname}.vidt", "wb")
                                if self.expt.basefname is not None
                                else None)
        # Write experiment details file.
        self._save_experiment_details()
        sys.stdout.write(f"\nStarted video logging. {self.expt.basefname.split(os.sep)[-1]}.avi")
        sys.stdout.write(f"\nStarted video time logging. {self.expt.basefname.split(os.sep)[-1]}.vidt.")

    def _stop_logging(self, valid=True):
        # Close the existing file.
        if self.expt.recording is True:
            self.expt.vidwriter.release()
            self.expt.vidtfhndl.close()
            self.expt.basefname = None

    def _save_experiment_details(self):
        # Save all the experiment details.
        # Prepare for experiment.
        with open('expt_params.json') as fh:
            self.expt.params = json.load(fh)
        _details = {
            "subject": self.expt.subject,
            "tasks": self.expt.tasks,
            "datafiles": self.expt.datafiles,
        }
        with open(os.path.join(self.expt.outdir, "experiment_details.json"), "w") as fh:
            json.dump(_details, fh, indent=4)
    
    def update_ui(self):
        """Function to call when the control on the UI need to be updated.
        """
        # Handle state
        self._state_ui_handlers[self._state]()
    
    def attach_callbacks(self):
        """Function to attach callbacks to the different controls.
        """
        self.linedit_subjname.textChanged.connect(self._callback_subjname_changed)
        self.btn_startstop_expt.clicked.connect(self._callback_start_stop_expt)
    
    def _iter_list_tasks(self):
        for i in range(self.listTasks.count()):
            yield self.listTasks.item(i)
    
    # ############################## #
    # Statemachine Handler Functions #
    # ############################## #
    def _handle_all_done(self):
        # Disable everything.
        self.listTasks.clear()
        self.listTasks.setEnabled(True)
        self.linedit_subjname.setEnabled(False)
        self.btn_startstop_expt.setEnabled(False)
        self.lbl_message.setText("All done.")
        self.plain_instructions.clear()
        self.plain_instructions.appendPlainText("All done. You can close the program now.")

    def _handle_waiting_for_all_done(self):
        # Check if this is the start of the experiment.
        self.listTasks.setEnabled(True)
        self.linedit_subjname.setEnabled(False)
        self.btn_startstop_expt.setEnabled(False)
        # Update image label border
        self.lbl_img.setStyleSheet("border: 0px solid red;")
        # Update message details.
        self.lbl_message.setText("All tasks recorded.")
        # Update instructions
        self.plain_instructions.clear()
        # Update list of tasks
        self.listTasks.clearSelection()
        # Color all tasks before the current index to red, 
        # and the ones below the current index to green.
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Light")
        font.setPointSize(12)
        for i, _item in enumerate(self._iter_list_tasks()):
            font.setWeight(QFont.Normal)
            _item.setText(f'{_item.text().split(" [")[0]} [Recorded]')
            _item.setForeground(QtGui.QColor(255, 0, 0, 128))
            _item.setFont(font)
    
    def _handle_recording_task(self):
        self.linedit_subjname.setEnabled(False)
        self.btn_startstop_expt.setEnabled(False)
        # Update instructions
        self.plain_instructions.setStyleSheet(
            """QPlainTextEdit {background-color: #fff;
                               color: #aaa;}"""
        )
        # Update image label border
        self.lbl_img.setStyleSheet("border: 3px solid red;")
        self.listTasks.setCurrentRow(self.expt.currtaskindx)
        # Color all tasks before the current index to red, 
        # and the ones below the current index to green.
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Light")
        font.setPointSize(12)
        for i, _item in enumerate(self._iter_list_tasks()):
            if i < self.expt.currtaskindx:
                # Check if the text has been modified.
                font.setWeight(QFont.Normal)
                _item.setText(f'{_item.text().split(" [")[0]} [Recorded]')
                _item.setForeground(QtGui.QColor(255, 0, 0, 64))
            elif i > self.expt.currtaskindx:
                font.setWeight(QFont.Normal)
                _item.setText(f'{_item.text().split(" [")[0]}')
                _item.setForeground(QtGui.QColor(0, 0, 255, 64))
            else:
                font.setWeight(QFont.Bold)
                _item.setText(f'{_item.text().split(" [")[0]} [Recording]')
                _item.setForeground(QtGui.QColor(0, 0, 0, 255))
                self.lbl_message.setText(f"Recording '{_item.text().split(' [')[0]}'")
            _item.setFont(font)
    
    def _handle_waiting_to_record_task(self):
        # Check if this is the start of the experiment.
        self.listTasks.setEnabled(True)
        self.linedit_subjname.setEnabled(False)
        self.btn_startstop_expt.setEnabled(False)
        # Update instructions
        self.plain_instructions.setStyleSheet(
            """QPlainTextEdit {background-color: #fff;
                               color: #000;}"""
        )
        # Update image label border
        self.lbl_img.setStyleSheet("border: 0px solid red;")
        # Update list of tasks
        self.listTasks.setCurrentRow(self.expt.currtaskindx)
        # Color all tasks before the current index to red, 
        # and the ones below the current index to green.
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Light")
        font.setPointSize(12)
        for i, _item in enumerate(self._iter_list_tasks()):
            if i < self.expt.currtaskindx:
                font.setWeight(QFont.Normal)
                _item.setText(f'{_item.text().split(" [")[0]} [Recorded]')
                _item.setForeground(QtGui.QColor(255, 0, 0, 128))
            elif i > self.expt.currtaskindx:
                font.setWeight(QFont.Normal)
                _item.setText(f'{_item.text().split(" [")[0]}')
                _item.setForeground(QtGui.QColor(0, 0, 255, 128))
            else:
                font.setWeight(QFont.Normal)
                _item.setText(f'{_item.text().split(" [")[0]} [Waiting to Record]')
                self.lbl_message.setText(f"Waiting to Record '{_item.text().split(' [')[0]}'")
                _item.setForeground(QtGui.QColor(0, 0, 0, 255))
            _item.setFont(font)
    
    def _handle_waiting_to_start(self):
        self.listTasks.setEnabled(True)
        self.linedit_subjname.setEnabled(True)
        self.btn_startstop_expt.setEnabled(len(self.linedit_subjname.text()) > 0)
        self.lbl_message.setText("Wait for experiment to start.")

    def _handle_sensors_not_selected(self):
        self.lbl_message.setText("")
        self.lbl_datetime.setText("")
        self.lbl_exptdur.setText("")
        self.listTasks.setEnabled(True)
        self.btn_startstop_expt.setEnabled(len(self.linedit_subjname.text()))

    # ############################ #
    # QT Widget Callback fucntions #
    # ############################ #
    def keyPressEvent(self, event):
        modifiers= QApplication.keyboardModifiers()

        # Do not handle return if the experiment is not started yet.
        if self._state == ParkourRecorderStates.WaitingToStart:
            return
        
        # Check if only the enter key was pressed.
        if (event.key() == 16777220 and modifiers != Qt.ControlModifier):
            # Check if we should start or stop task.
            # Check start is not WaitForAllDone.
            if self.expt.recording:
                self.expt.currtaskstpt = dt.now()
                # Stop recording
                self._stop_logging()
                self.expt.recording = False
                
                # Update task to be recorded.
                if self._prepare_for_the_next_task() is False:
                    # All done.
                    self._state = ParkourRecorderStates.WaitingForAllDone
                else:
                    self._state = ParkourRecorderStates.WaitingToRecordTask        
                
                # Start unlabelled reording.
                self._start_logging(task=False)                
            else:
                if self._state != ParkourRecorderStates.WaitingForAllDone:
                    # Start recording task.
                    self.expt.currtaskstrt = dt.now()
                    self.expt.currtaskstpt = None
                    # Start recording.
                    self._start_logging(task=True)
                    self.expt.recording = True
                    self._state = ParkourRecorderStates.RecordingTask
                else:
                    self._stop_logging()
                    self._state = ParkourRecorderStates.AllDone        
        elif event.key() == 82 and modifiers == Qt.ControlModifier:
            # Redo task. This is responded to only when one is waiting to record a task.
            # We go back to the previous task.
            self._redo_task()
            # Change state.
            self._state = ParkourRecorderStates.WaitingToRecordTask

        self.update_ui()
    
    def _callback_app_timer(self):
        # Update current time.
        self.currtime = dt.now()
        self.lbl_datetime.setText(self.currtime.strftime("%A, %D %T"))
        if self.expt.starttime is not None and self._state != ParkourRecorderStates.AllDone:
            self.expt.exptdur = self.currtime - self.expt.starttime
            self.lbl_exptdur.setText(f"Expt. Duration: {str(self.expt.exptdur)[:7]}")
        # Update task detials if in recording state
        if (self._state == ParkourRecorderStates.WaitingToRecordTask
            or self._state == ParkourRecorderStates.RecordingTask):
            _taskname = self.expt.tasks[self.expt.currtaskindx][0]
            if self.expt.currtaskstrt is None:
                _taskdetails = f"Waiting to Record '{_taskname}'"
            else:
                if self.expt.currtaskstpt is None:
                    _taskdetails = f"Recording '{_taskname}' ({str(self.currtime - self.expt.currtaskstrt)[:7]})"
                else:
                    _taskdetails = f"Recording '{_taskname}' ({str(self.expt.currtaskstpt - self.expt.currtaskstrt)[:7]})"
            self.lbl_message.setText(_taskdetails)
    
    def _callback_subjname_changed(self):
        self.btn_startstop_expt.setEnabled(len(self.linedit_subjname.text()) > 0)
    
    def _callback_start_stop_expt(self):
        # Check if this to start the experiment.
        if "Start" in self.btn_startstop_expt.text():
            # Check from the user that they wanted to start the experiment.
            if self._get_expt_start_confirmation():
                # Start experiment.
                self._start_expt()
                
                # Add tasks to list
                self.listTasks.addItems([_t[0] for _t in self.expt.tasks])
                                
                # Start logging data.
                self._start_logging(task=None)
                
                # Change state to waiting to record task.
                self._state = ParkourRecorderStates.WaitingToRecordTask
                
                # Prepare for the next task
                if self._prepare_for_the_next_task(init=True) is False:
                    # All tasks done.
                    self._state = ParkourRecorderStates.WaitingForAllDone
                
                self.update_ui()

    # ###################### #
    # Video Update Functions #
    # ###################### #
    @pyqtSlot(np.ndarray)
    def _sig_cb_update_image(self, cv_img):
        """Updates the image_label with a new opencv image.
        """
        # Is the video to be logged?
        if self.expt.basefname is not None:
            # Write current frame
            self.expt.vidwriter.write(cv_img)
            # Write time
            self.expt.vidtfhndl.write(temporenc.packb(self.camthread.imgtime))
            print(self.camthread.imgtime)
        
        # Update display
        self.imgdispcount += 1
        self.imgdispcount %= 2
        if self.imgdispcount == 0:
            img_w, img_h = (self.lbl_img.size().width(),
                            self.lbl_img.size().height())
            self.lbl_img.setPixmap(
                self.convert_cv_qt(cv_img, img_w, img_h,
                                   rgb=self.expt.recording)
            )
    
    def convert_cv_qt(self, img, img_w, img_h, rgb=True):
        """Convert from an opencv image to QPixmap.
        """
        if rgb:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(img.shape)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h,
                                                bytes_per_line,
                                                QtGui.QImage.Format_RGB888)
        else:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = rgb_image.shape
            bytes_per_line = w
            convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h,
                                                bytes_per_line,
                                                QtGui.QImage.Format_Grayscale8)
        p1 = convert_to_Qt_format.scaled(img_w, img_h,
                                         Qt.KeepAspectRatio)
        pxmap = QPixmap.fromImage(p1)
        return pxmap.scaledToWidth(self.lbl_img.contentsRect().width(), Qt.SmoothTransformation)
    
    # ########################### #
    # Experiment Related Function #
    # ########################### #
    def _create_expt_vars(self):
        self.expt = AttrDict()
        self.expt.subject = None
        self.expt.outdir = None
        self.expt.basefname = None
        self.expt.vidwriter = None
        self.expt.vidtfhndl = None
        self.expt.starttime = None
        self.expt.exptdur = None
        self.expt.tasks = None
        self.expt.currtaskindx = None
        self.expt.currtaskstrt = None
        self.expt.currtaskstpt = None
        self.expt.datafiles = None
        self.expt.recording = False
        # Prepare for experiment.
        with open('expt_params.json') as fh:
            self.expt.params = json.load(fh)
        
    def _start_expt(self):
        self.expt.starttime = dt.now()
        self.expt.subject = f"{self.linedit_subjname.text()}_{self.expt.starttime.strftime('%y-%m-%d-%H-%M-%S')}"
        self.expt.outdir = os.path.join(self.expt.params["outdir"],
                                        self.expt.params["name"],
                                        self.expt.subject)
        # Cereate directory.
        os.makedirs(self.expt.outdir)
        sys.stdout.write(f"\nCreated user folder {self.expt.outdir}")
        # Experiment start time.
        self.expt.exptdur = None
        _tasks = self.expt.params['tasks'] * self.expt.params['N']
        # random.shuffle(_tasks)
        self.expt.tasks = _tasks
        self.expt.datafiles = []
        self.expt.recording = False

    def _prepare_for_the_next_task(self, init=False):
        # Set current task.
        if self._set_current_task(init) is False:
            return False
        
        # Update instructions.
        self.plain_instructions.clear()
        _currtask = self.expt.tasks[self.expt.currtaskindx]
        self.plain_instructions.appendPlainText(f"Task: {_currtask[0]}")
        self.plain_instructions.appendPlainText(f"")
        self.plain_instructions.appendPlainText(f"Instruction:")
        self.plain_instructions.appendPlainText(_currtask[1][0])
        self.plain_instructions.appendPlainText(f"")
        self.plain_instructions.appendPlainText(f"To do:")
        self.plain_instructions.appendPlainText(_currtask[1][1])
        return True
    
    def _redo_task(self, init=False):
        # Decrement currtaskindex
        if self.expt.currtaskindx > 0:
            self.expt.currtaskindx -= 2
            return self._prepare_for_the_next_task()
    
    def _set_current_task(self, init=False):
        if init:
            self.expt.currtaskindx = 0
        else:
            self.expt.currtaskindx += 1
        # Make sure there are still tasks left.
        if self.expt.currtaskindx >= len(self.expt.tasks):
            return False
        self.expt.currtaskstrt = None
        self.expt.currtaskstpt = None
        self.expt.recording = False
        return True
    
    # #################### #
    # Dialog Box Functions #
    # #################### #   
    def _get_expt_start_confirmation(self):
        _dlg = QMessageBox(self)
        _dlg.setWindowTitle("Ready to start?")
        _dlg.setText("\n".join((f"Subject: {self.linedit_subjname.text()}",
                                "You will not be able to change experiment details after this.",
                                "Simply close the program if you want to stop the experiment.",
                                "Shall I start?")))
        _dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        _dlg.setIcon(QMessageBox.Question)
        return _dlg.exec() == QMessageBox.Yes
        
    def _get_expt_stop_confirmation(self):
        _dlg = QMessageBox(self)
        _dlg.setWindowTitle("Stop experiment?")
        _dlg.setText("Are you sure you want to stop the experiment.")
        _dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        _dlg.setIcon(QMessageBox.Question)
        return _dlg.exec() == QMessageBox.Yes
    
    def _get_close_app_confirmation(self):
        _dlg = QMessageBox(self)
        _dlg.setWindowTitle("Close application?")
        _dlg.setText("\n".join(("Are you sure you want close the application?",
                                "All data will be lost!")))
        _dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        _dlg.setIcon(QMessageBox.Question)
        return _dlg.exec() == QMessageBox.Yes

    def _get_task_save_confirmation(self):
        _dlg = QMessageBox(self)
        _dlg.setWindowTitle("Save task?")
        _dlg.setText("\n".join((f"Are you sure you want to save task",
                                f"'{self.expt.currtaskname}'?")))
        _dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        _dlg.setIcon(QMessageBox.Question)
        return _dlg.exec() == QMessageBox.Yes


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mywin = ParkourRecorder()
    mywin.show()
    sys.exit(app.exec_())
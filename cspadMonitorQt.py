#!/usr/bin/env python
"""
vmiViewer:
A GUI for monitoring Velocity-Map-Imaging (VMI).

Unagi is a Japanese word for 'total state of awareness'
For more explanation:
http://www.youtube.com/watch?v=OJOYdAkIDq0

Copyright (c) Chun Hong Yoon
chun.hong.yoon@desy.de

This file is part of UNAGI.

UNAGI is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

UNAGI is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with UNAGI.  If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import os
import time
import numpy 
import pickle
import copy
import signal
import pyqtgraph
import PyQt4.QtGui
import PyQt4.QtCore
import fnmatch

def get_fnams(directory):
    fnams = []
    for dirname, dirnames, filenames in os.walk('.'):
        for filename in fnmatch.filter(filenames, '*.png'):
            fnam_abs = os.path.join(dirname, filename)
            fnams.append(fnam_abs)
    fnams.sort()
    for fnam in fnams:
        print fnam
    return fnams

class MainFrame(PyQt4.QtGui.QWidget):
    """
    The main frame of the application
    """
    
    def __init__(self):
        super(MainFrame, self).__init__()
        
        # parameters
        self.title = 'OCP Monitor'
        self.zmq_timer = 2000                # milli seconds
        self.integration_depth_counter = 0
        self.cspad_shape = (100, 100)
        self.data_dir = '/home/amorgan/Physics/git-repos/ocp_viewer/20141103/'
        
        dis_temp = {}
        dis_temp['cspad_raw']              = numpy.zeros(self.cspad_shape, numpy.int64)
        dis_temp['cspad_raw_counts']       = []
        dis_temp['cspad_raw_histogram']    = None
        dis_temp['cspad_raw_radial_profile']      = None
        dis_temp['event_id']               = None
        dis_temp['frames']                 = 0
        #
        self.display_data = dis_temp
        
        # user controlled parameters
        self.input_params = {}
        self.input_params['integration_depth'] = 1
        self.input_params['livestream']        = False

        # initialisation of GUI and network functions
        self.initUI()
        self.init_timer()


    def update_display_data(self):
        """ network data --> display data --> call update image (if needed) """
        # live streaming ?
        if self.input_params['livestream'] == False :
            return False
        
        # copy the data from the network data
        #self.local_data = 
        get_fnams(self.data_dir)

        #self.update_image()
        
        return True


    def update_image(self):
        """ display data + user options --> screen """

        self.imageWidget.setImage(self.display_data['cspad_raw'], autoRange = False, autoLevels = False, autoHistogramRange = False)

        self.histWidget.plot(self.display_data['cspad_raw_histogram'][1][1 :-1], self.display_data['cspad_raw_histogram'][0][1 :] + 1)
        
        self.countsWidget.clear()
        self.countsWidget.setTitle('Integrated signal')
        self.countsWidget.plot(self.display_data['cspad_raw_counts'])

        self.radProfWidget.clear()
        self.radProfWidget.plot(self.display_data['cspad_raw_radial_profile'][1], self.display_data['cspad_raw_radial_profile'][0])

        self.display_data['event_id'] = self.local_data['event_id']
        
        return True


    def add_local_dis_hist(self, hist_bin1, hist_bin2):
        bins_old = numpy.array(hist_bin1[1])
        bins_new = numpy.array(hist_bin2[1])
        hist_old = numpy.array(hist_bin1[0])
        hist_new = numpy.array(hist_bin2[0])
        
        # expand the bins to fit both histograms, the bins must have the property: bins[i] = bins[i-1] + 1
        bin_max = max([bins_old[-1], bins_new[-1]])
        bin_min = min([bins_old[0], bins_new[0]])
        bins = numpy.arange(bin_min, bin_max+1, 1)

        # add the hists
        hist = numpy.zeros((bins.shape[0]-1), dtype=numpy.int)
        
        h_args = [numpy.where(bins == bins_old[0])[0][0], numpy.where(bins == bins_old[-2])[0][0] + 1]
        hist[ h_args[0] : h_args[1] ] = hist_old 

        h_args = [numpy.where(bins == bins_new[0])[0][0], numpy.where(bins == bins_new[-2])[0][0] + 1]
        hist[ h_args[0] : h_args[1] ] += hist_new 

        return [hist, bins]
        

    def init_timer(self):
        """ Update the image every milli_secs. """
        self.refresh_timer = PyQt4.QtCore.QTimer()
        self.refresh_timer.timeout.connect(self.update_display_data)
        self.refresh_timer.start(self.zmq_timer)


    def initUI(self):
        """ Set the layout of the GUI window """
        # Input checkers
        self.intregex = PyQt4.QtCore.QRegExp('[0-9]+')
        self.floatregex = PyQt4.QtCore.QRegExp('[0-9\.]+')
        self.ipregex = PyQt4.QtCore.QRegExp('[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+')
        
        self.qtipvalidator = PyQt4.QtGui.QRegExpValidator()
        self.qtipvalidator.setRegExp(self.ipregex)
        self.qtintvalidator = PyQt4.QtGui.QRegExpValidator()
        self.qtintvalidator.setRegExp(self.intregex)

        self.setWindowTitle(self.title)

        # cspad image viewer
        self.imageWidget = pyqtgraph.ImageView(self)
        self.imageWidget.ui.normBtn.hide()
        self.imageWidget.ui.roiBtn.hide()
        self.imageWidget.setLevels(0, 40)

        # cspad histogram viewer
        self.histWidget = pyqtgraph.PlotWidget(self)
        self.histWidget.setTitle('Histogram (single frame)')
        self.histWidget.setLabel('bottom', text = 'adus')

        # clear cspad histogram button
        self.histClear = PyQt4.QtGui.QPushButton("Clear histogram", self)
        self.histClear.clicked.connect(self.clearHistogram)

        # photon counts viewer 
        self.countsWidget = pyqtgraph.PlotWidget(self)
        self.countsWidget.setTitle('Photon counts')
        self.countsWidget.setLabel('bottom', text = 'frame')
        
        # radial profile viewer
        self.radProfWidget = pyqtgraph.PlotWidget(self)
        self.radProfWidget.setTitle('radial profile')
        self.radProfWidget.setLabel('bottom', text = 'pixel radius')

        # Integration depth button
        self.integrate_label = PyQt4.QtGui.QLabel(self)
        self.integrate_label.setText('Integrate images:')
        toolTip = \
                """<b>0</b>: display the running total <br>
                <b>1</b>: display each frame <br>
                <b>n</b>: integrate <b>n</b> frames before display <br>
                The histogram is also integrated. Note that this is 
                not the same as taking the histogram of the sum! 
                """
        self.integrate_label.setToolTip(toolTip)
        self.integrate_lineedit = PyQt4.QtGui.QLineEdit(self)
        self.integrate_lineedit.setValidator(self.qtintvalidator)
        self.integrate_lineedit.setText(str(self.input_params['integration_depth']))
        self.integrate_lineedit.editingFinished.connect(self.update_integration_size)

        # live stream checkbox
        self.livestream_checkbox = PyQt4.QtGui.QCheckBox('Live Stream', self)
        self.livestream_checkbox.stateChanged.connect(self.update_livestream)

        # Add all the stuff to the layout
        hlayouts = []

        # cspad | hist
        #         ----
        #         phot
        #         ----
        #         radial profile
        #  live stream

        # cspad image layout
        hlayouts.append(PyQt4.QtGui.QHBoxLayout())
        Hsplitter = PyQt4.QtGui.QSplitter(PyQt4.QtCore.Qt.Horizontal)
        Hsplitter.addWidget(self.imageWidget)

        # histogram plot layout
        Vsplitter = PyQt4.QtGui.QSplitter(PyQt4.QtCore.Qt.Vertical)
        Vsplitter.addWidget(self.histWidget)

        # integrated signal / photon counts plot layout
        Vsplitter.addWidget(self.countsWidget)
        
        # radial profile plot layout
        Vsplitter.addWidget(self.radProfWidget)
        Hsplitter.addWidget(Vsplitter)

        hlayouts[-1].addWidget(Hsplitter)
        
        # integration depth layout
        hlayouts.append(PyQt4.QtGui.QHBoxLayout())
        hlayouts[-1].addWidget(self.integrate_label)
        hlayouts[-1].addWidget(self.integrate_lineedit)

        # clear histogram button layout
        hlayouts[-1].addWidget(self.histClear)

        # live checkbox layout
        vlayout0 = PyQt4.QtGui.QVBoxLayout()
        vlayout0.addWidget(self.livestream_checkbox)
        hlayouts[-1].addLayout(vlayout0)

        # stack everything vertically 
        vlayout = PyQt4.QtGui.QVBoxLayout()
        for hlayout in hlayouts :
            vlayout.addLayout(hlayout)

        self.setLayout(vlayout)
        self.resize(800,800)
        self.show()


    def update_livestream(self, state):
        if state == PyQt4.QtCore.Qt.Checked:
            self.input_params['livestream'] = True
        else :
            self.input_params['livestream'] = False

    def update_integration_size(self):
        pass

    def clearHistogram(self):
        self.histWidget.clear()

        


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)    # allow Control-C 
    app = PyQt4.QtGui.QApplication(sys.argv)
    ex = MainFrame()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()    

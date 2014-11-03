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
import scipy.misc
import radial_profile as rp
from datetime import datetime
import h5py

def get_fnams(directory, event_id = None):
    """If event_id is not None then get the next event else get the latest
    
    e.g.
    event_id = 0000181
    """
    fnams = []
    for dirname, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.png'):
            fnam_abs = os.path.join(dirname, filename)
            fnams.append(fnam_abs)
    
    if event_id is None :
        fnams.sort()
        fnam_out = fnams[-1]
        new_id = None
    else :
        fnam_out = None
        for i in range(1, 10000):
            #print 'matching:', '*'+str(event_id + i).zfill(7)+'*.png'
            fnam_out = fnmatch.filter(fnams, '*'+str(event_id + i).zfill(7)+'*.png')
            #print fnam_out
            if fnam_out != []:
                new_id = event_id + i
                break
        if fnam_out is None or len(fnam_out) > 1 or fnam_out == [] :
            raise ValueError('Next event not found, with id', '*'+str(event_id + i).zfill(7)+'*.png')
        fnam_out = fnam_out[0]
        
    return fnam_out, new_id



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
        self.cspad_shape = (1024, 1024)
        x, y = numpy.indices(self.cspad_shape)
        x -= self.cspad_shape[0]/2 - 1
        y -= self.cspad_shape[1]/2 - 1
        self.cspad_rads = numpy.sqrt(y**2 + x**2).astype(numpy.int)
        self.roi = numpy.ones(self.cspad_shape, dtype=numpy.bool)
        self.roi = numpy.where(self.roi == False)
        
        dis_temp = {}
        dis_temp['cspad_raw']                   = numpy.zeros(self.cspad_shape, numpy.int64)
        dis_temp['cspad_raw_counts']            = []
        dis_temp['cspad_raw_histogram']         = None
        dis_temp['cspad_raw_bins']              = None
        dis_temp['cspad_raw_radial_values']     = numpy.arange(self.cspad_rads.max()+1)
        dis_temp['cspad_raw_radial_profile']    = numpy.zeros_like(dis_temp['cspad_raw_radial_values'])
        dis_temp['event_id']                    = 0
        dis_temp['frames']                      = 0
        dis_temp['radiusroimin']                = 0
        dis_temp['radiusroimax']                = 0
        #
        self.display_data = dis_temp
        
        # user controlled parameters
        self.input_params = {}
        self.input_params['integration_depth'] = 1
        self.input_params['livestream']        = False
        self.input_params['directory']         = 'Z:/20141103/'

        # initialisation of GUI and network functions
        self.initUI()
        self.init_timer()


    def update_display_data(self):
        """ network data --> display data --> call update image (if needed) """
        
        # copy the data from the network data
        fnam, event_id = get_fnams(self.input_params['directory'], self.display_data['event_id'])
        self.display_data['event_id'] = event_id
	print fnam
        
        # load the image 
        self.display_data['cspad_raw'] = scipy.misc.imread(fnam)

        # apply the ROI
        self.display_data['cspad_raw'][self.roi] = 0

        # radial profile
        rad_prof, raw_counts = rp.radial_profile_integrate(self.display_data['cspad_raw'].astype(numpy.int64), self.cspad_rads, 0, self.cspad_rads.max())
        self.display_data['cspad_raw_radial_profile'] = rad_prof
        self.display_data['cspad_raw_counts'].append(raw_counts)
        #print 'raw counts', raw_counts

        # histogram
        #bins = numpy.arange(self.display_data['cspad_raw'].min(), self.display_data['cspad_raw'].max(), 1)
        hist, bins = numpy.histogram(self.display_data['cspad_raw'], bins=200)
        self.display_data['cspad_raw_histogram'] = hist
        self.display_data['cspad_raw_bins'] = bins
        
        self.update_image()
        
        return True


    def update_image(self):
        """ display data + user options --> screen """

        self.imageWidget.setImage(self.display_data['cspad_raw'], autoRange = False, autoLevels = False, autoHistogramRange = False)

        self.histWidget.plot(self.display_data['cspad_raw_bins'][1 :-1], self.display_data['cspad_raw_histogram'][1 :] + 1)
        
        self.countsWidget.clear()
        self.countsWidget.setTitle('Integrated signal')
        self.countsWidget.plot(self.display_data['cspad_raw_counts'])

        self.radProfWidget.clear()
        self.radProfWidget.plot(self.display_data['cspad_raw_radial_values'], self.display_data['cspad_raw_radial_profile'])

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
        self.refresh_timer.timeout.connect(self.update_data_stream)
        self.refresh_timer.start(self.zmq_timer)

    def update_data_stream(self):
	if self.input_params['livestream'] == True :
            self.update_display_data()


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
        self.qtfloatvalidator = PyQt4.QtGui.QRegExpValidator()
        self.qtfloatvalidator.setRegExp(self.floatregex)

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

        # ROI
        self.radiusroi_label = PyQt4.QtGui.QLabel(self)
        self.radiusroi_label.setText('Radius ROI (Min,Max):')
        self.radiusroimin_lineedit = PyQt4.QtGui.QLineEdit(self)
        self.radiusroimin_lineedit.setValidator(self.qtfloatvalidator)
        self.radiusroimin_lineedit.setText(str(self.display_data['radiusroimin']))
        self.radiusroimin_lineedit.editingFinished.connect(self.update_roi)
        self.radiusroimax_lineedit = PyQt4.QtGui.QLineEdit(self)
        self.radiusroimax_lineedit.setValidator(self.qtfloatvalidator)
        self.radiusroimax_lineedit.setText(str(self.display_data['radiusroimax']))
        self.radiusroimax_lineedit.editingFinished.connect(self.update_roi)
 
        # directory
        self.directory_label = PyQt4.QtGui.QLabel(self)
        self.directory_label.setText('Directory to scan use /\'s:')
        self.directory_lineedit = PyQt4.QtGui.QLineEdit(self)
        self.directory_lineedit.setText(self.input_params['directory'])
        self.directory_lineedit.editingFinished.connect(self.update_directory)

        # live stream checkbox
        self.livestream_checkbox = PyQt4.QtGui.QCheckBox('Live Stream', self)
        self.livestream_checkbox.stateChanged.connect(self.update_livestream)

        # save state
        self.saveStateButton = PyQt4.QtGui.QPushButton("save state", self)
        self.saveStateButton .clicked.connect(self.saveState)

        # load state
        self.loadStateButton = PyQt4.QtGui.QPushButton("load state", self)
        self.loadStateButton.clicked.connect(self.loadState)

        # back button
        self.backButton = PyQt4.QtGui.QPushButton("back one event", self)
        self.backButton .clicked.connect(self.back_event)

        # forward button
        self.forwardButton = PyQt4.QtGui.QPushButton("forward one event", self)
        self.forwardButton.clicked.connect(self.forward_event)

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
        #hlayouts.append(PyQt4.QtGui.QHBoxLayout())
        #hlayouts[-1].addWidget(self.integrate_label)
        #hlayouts[-1].addWidget(self.integrate_lineedit)

        # ROI display layout
        hlayouts.append(PyQt4.QtGui.QHBoxLayout())
        hlayouts[-1].addWidget(self.radiusroi_label)
        hlayouts[-1].addWidget(self.radiusroimin_lineedit)
        hlayouts[-1].addWidget(self.radiusroimax_lineedit)
        hlayouts[-1].addStretch()

        # clear histogram button layout
        hlayouts[-1].addWidget(self.histClear)

        # save and load state button layout
        hlayouts[-1].addWidget(self.saveStateButton)
        hlayouts[-1].addWidget(self.loadStateButton)

        # live checkbox layout
        vlayout0 = PyQt4.QtGui.QVBoxLayout()
        vlayout0.addWidget(self.livestream_checkbox)
        hlayouts[-1].addLayout(vlayout0)

        # directory input layout
        hlayouts.append(PyQt4.QtGui.QHBoxLayout())
        hlayouts[-1].addWidget(self.directory_label)
        hlayouts[-1].addWidget(self.directory_lineedit)

	# back and forwards buttons
        hlayouts[-1].addWidget(self.backButton)
        hlayouts[-1].addWidget(self.forwardButton)

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

    def update_roi(self):
        new_radiusroimin = float(self.radiusroimin_lineedit.text())
        new_radiusroimax = float(self.radiusroimax_lineedit.text())
        if new_radiusroimin != self.display_data['radiusroimin'] or new_radiusroimax != self.display_data['radiusroimax'] :
            self.display_data['radiusroimin'] = new_radiusroimin
            self.display_data['radiusroimax'] = new_radiusroimax
            temp = (self.cspad_rads > new_radiusroimin) * (self.cspad_rads < new_radiusroimax)
            self.roi = numpy.where(temp == False)
            print 'New ROI:', new_radiusroimin, new_radiusroimax
            self.display_data['cspad_raw'][self.roi] = 0
            self.update_image()
        
    def saveState(self):
        """Save all of the class variables to a h5 file with a time"""
        fnam = 'saved_state_OCPviewer_%s.h5'%datetime.now().strftime('%Y-%m-%d-%s')
        print 'writing to file:', fnam
        f = h5py.File(fnam, 'w')
        # loop over key value pairs and write to h5
        dis_h5 = f.create_group('display_data')
        for k in self.display_data.keys() :
            print 'writing ', k, ' of type ', type(self.display_data[k])
            if self.display_data[k] is not None :
                dis_h5.create_dataset(k, data = self.display_data[k])
        f.close()
        print 'done'

    def loadState(self):
        fnam = PyQt4.QtGui.QFileDialog.getOpenFileName(self, 'Open file', './')
        print 'reading from file:', fnam, type(fnam)
        fnam = str(fnam)
        print 'reading from file:', fnam, type(fnam)
        f = h5py.File(fnam, 'r')
        # loop over key value pairs and read from h5
        display_data = f['display_data']
        for k in display_data.keys() :
            print 'reading ', k, ' of type ', type(display_data[k].value)
            self.display_data[k] = display_data[k].value
        f.close()
        
        self.display_data['cspad_raw_counts']       = list(self.display_data['cspad_raw_counts'])

        self.update_image()
        print 'done'

    def update_directory(self):
        self.input_params['directory'] = str(self.directory_lineedit.text())

    def back_event(self):
        self.display_data['event_id'] -= 2
        self.update_display_data()

    def forward_event(self):
        self.update_display_data()

def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)    # allow Control-C 
    app = PyQt4.QtGui.QApplication(sys.argv)
    ex = MainFrame()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()    

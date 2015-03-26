#!/usr/bin/env python

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
import radial_profile as rp
from datetime import datetime
import h5py
import png
import itertools
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def load_png_Valerio(fnam):
    png_reader=png.Reader(fnam)
    data=png_reader.asDirect()[2]
    image_2d = numpy.vstack(itertools.imap(numpy.ushort, data))
    return image_2d


files     = dict()
file_keys = []
class FnamEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        print event.src_path
        if event.is_directory == False :
            if event.src_path[-3 :] == 'png':
                fnam = os.path.basename(event.src_path)
                global files
                global file_keys
                file_keys.append(fnam[:7])
                files[file_keys[-1]] = str(os.path.abspath(event.src_path))


def fill_current_file_list(directory = '/home/amorgan/Physics/git_repos/ocp_viewer/20141103/'):
    fnams = {}
    for dirname, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.png'):
            fnam_abs = os.path.join(dirname, filename)
            fnams[filename[:7]] = fnam_abs
    global files
    global file_keys
    files = fnams
    # 
    print 'loaded file list under selected directory:'
    file_keys = files.keys()
    file_keys.sort()
    for k in file_keys:
        print k, files[k]


def get_fnams(directory, event_index = -1):
    """If event_index is not -1 then get the next event else get the latest
    
    e.g.
    event_index = 0000181
    """
    global files
    global file_keys
    if len(files) == 0 :
        raise ValueError('No files found in:', directory)
    
    if event_index == -1 :
        fnam_out = files[file_keys[-1]]
        # get the event index
        new_id = len(file_keys)
    else :
        if event_index >= 0 and event_index < len(file_keys):
            new_id   = event_index
            fnam_out = files[file_keys[event_index]]
        else :
            print 'end of the line (you are at the latest event)'
            new_id   = event_index
            fnam_out = files[file_keys[-1]]
    
    return fnam_out, new_id


class MainFrame(PyQt4.QtGui.QWidget):
    """
    The main frame of the application
    """
    
    def __init__(self):
        super(MainFrame, self).__init__()
        
        # parameters
        self.title = 'OCP Monitor'
        self.zmq_timer = 1000                # milli seconds
        self.integration_depth_counter = 0
        self.ring_pen = pyqtgraph.mkPen('r', width=2)
        self.display_data = dict()
        
        # set image size dependent parameters
        self.cspad_shape = (1024, 1024)
        x, y = numpy.indices(self.cspad_shape)
        x -= self.cspad_shape[0]/2 - 1
        y -= self.cspad_shape[1]/2 - 1
        self.cspad_rads = numpy.sqrt(y**2 + x**2).astype(numpy.int)
        self.roi = numpy.ones(self.cspad_shape, dtype=numpy.bool)
        self.roi = numpy.where(self.roi == False)
        
        self.display_data['cspad_raw_counts']            = []
        self.display_data['cspad_hit_rate']              = []
        self.display_data['cspad_raw_histogram']         = None
        self.display_data['cspad_raw_bins']              = None
        self.display_data['cspad_raw_radial_values']     = numpy.arange(self.cspad_rads.max()+1)
        self.display_data['cspad_raw_radial_profile']    = numpy.zeros_like(self.display_data['cspad_raw_radial_values'])
        self.display_data['frames']                      = 0
        self.display_data['last_file']                   = None
        
        # user controlled parameters
        self.input_params = dict()
        self.input_params['integration_depth'] = 1
        self.input_params['livestream']        = True
        self.input_params['directory']         = os.path.abspath('Z:\\')
        self.input_params['threshold']         = 0
        self.input_params['radiusroimin']      = 0
        self.input_params['radiusroimax']      = 0
        self.input_params['centrei']           = self.cspad_shape[0]/2 - 1
        self.input_params['centrej']           = self.cspad_shape[1]/2 - 1
        
        # initialisation of GUI and network functions
        # fill the files dict --> global variable files
        fill_current_file_list(self.input_params['directory'])
        # listen for new files and put them in the global dict 
        self.observer = Observer()
        self.observer.schedule(FnamEventHandler(), path=self.input_params['directory'], recursive=True)
        self.observer.start()
        
        self.initUI()
        self.init_timer()


    def update_display_data(self, force_update = True):
        """ network data --> display data --> call update image (if needed) """

        # copy the data from the network data
        if self.input_params['livestream'] == True:
            fnam, event_id = get_fnams(self.input_params['directory'], -1)
        else :
            fnam, event_id = get_fnams(self.input_params['directory'], self.display_data['event_id'])

        if fnam is None:
            return 0

        global file_keys
        self.display_data['event_id'] = event_id

        if self.display_data['last_file'] != fnam:
             self.display_data['last_file'] = fnam
             self.statusbarWidget.showMessage("Displayed image last changed: " + time.strftime("%H:%M:%S") + ' file:' + fnam)
        elif self.display_data['last_file'] == fnam and force_update == False :
            return 0

        # load the image
        #self.display_data['cspad_raw'] = scipy.misc.imread(fnam)
        self.display_data['cspad_raw'] = load_png_Valerio(fnam)

        # If the shape of self.display_data['cspad_raw'] is different than self.cspad_shape then reinitialise
        if self.display_data['cspad_raw'].shape != self.cspad_shape:
            self.cspad_shape = self.display_data['cspad_raw'].shape
            x, y = numpy.indices(self.cspad_shape)
            x -= self.cspad_shape[0]/2 - 1
            y -= self.cspad_shape[1]/2 - 1
            self.cspad_rads = numpy.sqrt(y**2 + x**2).astype(numpy.int)
            self.roi = numpy.ones(self.cspad_shape, dtype=numpy.bool)
            self.roi = numpy.where(self.roi == False)
            self.display_data['cspad_raw_radial_values']     = numpy.arange(self.cspad_rads.max()+1)
            self.display_data['cspad_raw_radial_profile']    = numpy.zeros_like(self.display_data['cspad_raw_radial_values'])
            self.input_params['centrei']           = self.cspad_shape[0]/2 - 1
            self.input_params['centrej']           = self.cspad_shape[1]/2 - 1
            self.centrei_lineedit.setText(str(self.input_params['centrei']))
            self.centrej_lineedit.setText(str(self.input_params['centrej']))


        # subtract background, if set
        if 'background_image' in self.display_data:
            self.display_data['cspad_raw'] -= self.display_data['background_image']

        # roll the axis
        self.display_data['cspad_raw'] = numpy.roll(self.display_data['cspad_raw'], self.input_params['centrei'] - self.cspad_shape[0]/2 - 1, 0)
        self.display_data['cspad_raw'] = numpy.roll(self.display_data['cspad_raw'], self.input_params['centrej'] - self.cspad_shape[1]/2 - 1, 1)

        # radial profile
        rad_prof, raw_counts = rp.radial_profile_integrate(self.display_data['cspad_raw'].astype(numpy.int64), self.cspad_rads, self.input_params['radiusroimin'], self.input_params['radiusroimax'])
        self.display_data['cspad_raw_radial_profile'] = rad_prof
        self.display_data['cspad_raw_counts'].append(raw_counts)
        #print 'raw counts', raw_counts

        # print the hit rate
        depth = 10
        if len(self.display_data['cspad_raw_counts']) >= depth :
            counts = numpy.array(self.display_data['cspad_raw_counts'])
            hit_rate = numpy.sum(counts[-depth :] > self.input_params['threshold']) / float(depth)
            #print 'hit rate: ', hit_rate
        else :
            counts = numpy.array(self.display_data['cspad_raw_counts'])
            hit_rate = numpy.sum(counts > self.input_params['threshold']) / float(counts.size)
            #print 'hit rate: ', hit_rate
        self.display_data['cspad_hit_rate'].append(hit_rate)

        # histogram
        #bins = numpy.arange(self.display_data['cspad_raw'].min(), self.display_data['cspad_raw'].max(), 1)
        hist, bins = numpy.histogram(self.display_data['cspad_raw'], bins=200)
        self.display_data['cspad_raw_histogram'] = hist
        self.display_data['cspad_raw_bins'] = bins

        self.update_image()

        return True


    def update_image(self):
        """ display data + user options --> screen """

        self.imageWidget.setImage(numpy.fliplr(numpy.rot90(self.display_data['cspad_raw'], k=3)), autoRange = False, autoLevels = False, autoHistogramRange = False)
        if self.input_params['radiusroimax'] != 0 :
            self.ring_canvas.setData(x = [self.cspad_shape[0]/2 - 1, self.cspad_shape[0]/2 - 1], y = [self.cspad_shape[1]/2 - 1, self.cspad_shape[1]/2 - 1], symbol = 'o', size = [2*self.input_params['radiusroimin'], 2*self.input_params['radiusroimax']],
                                     brush=(255,255,255,0), pen = self.ring_pen,
                                     pxMode = False)

        self.histWidget.clear()
        self.histWidget.plot(self.display_data['cspad_raw_bins'][1 :-1], self.display_data['cspad_raw_histogram'][1 :] + 1)
        
        self.countsWidget.clear()
        self.countsWidget.setTitle('Integrated signal')
        self.countsWidget.plot(self.display_data['cspad_raw_counts'])
        self.countsWidget.addItem(self.threshold_line)

        self.radProfWidget.clear()
        #self.radProfWidget.plot(self.display_data['cspad_raw_radial_values'], self.display_data['cspad_raw_radial_profile'])
        self.radProfWidget.plot(self.display_data['cspad_hit_rate'])

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
            self.update_display_data(force_update=False)


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

        # status bar
        self.statusbarWidget = PyQt4.QtGui.QStatusBar(self)
        self.statusbarWidget.setFixedHeight(30)
        self.statusbarWidget.setStyleSheet("QStatusBar{padding-left:8px;background:rgba(0,0,0,255);color:white;font-weight:bold;font-size:20px}")
        self.statusbarWidget.showMessage("Initializing...")

        # cspad image viewer
        self.imageWidget = pyqtgraph.ImageView(self)
        self.imageWidget.ui.normBtn.hide()
        self.imageWidget.ui.roiBtn.hide()
        # ROI overlay
        self.ring_canvas = pyqtgraph.ScatterPlotItem()
        self.imageWidget.getView().addItem(self.ring_canvas)
        #self.imageWidget.setLevels(0, 40)

        # cspad histogram viewer
        self.histWidget = pyqtgraph.PlotWidget(self)
        self.histWidget.setTitle('Histogram (single frame)')
        self.histWidget.setLabel('bottom', text = 'adus')

        # clear cspad histogram button
        self.histClear = PyQt4.QtGui.QPushButton("Clear plots", self)
        self.histClear.clicked.connect(self.clearHistogram)

        # photon counts viewer
        self.countsWidget = pyqtgraph.PlotWidget(self)
        self.countsWidget.setTitle('Photon counts')
        self.countsWidget.setLabel('bottom', text = 'frame')
        self.threshold_line = pyqtgraph.InfiniteLine(pos=self.input_params['threshold'], angle=0, movable=True)
        self.countsWidget.addItem(self.threshold_line)
        self.threshold_line.sigPositionChangeFinished.connect(self.update_threshold_graph)

        # radial profile viewer
        self.radProfWidget = pyqtgraph.PlotWidget(self)
        self.radProfWidget.setTitle('hit rate')
        self.radProfWidget.setLabel('bottom', text = 'frame')

        # Integration depth button
        # self.integrate_label = PyQt4.QtGui.QLabel(self)
        # self.integrate_label.setText('Integrate images:')
        # toolTip = \
        #         """<b>0</b>: display the running total <br>
        #         <b>1</b>: display each frame <br>
        #         <b>n</b>: integrate <b>n</b> frames before display <br>
        #         The histogram is also integrated. Note that this is
        #         not the same as taking the histogram of the sum!
        #         """
        # self.integrate_label.setToolTip(toolTip)
        # self.integrate_lineedit = PyQt4.QtGui.QLineEdit(self)
        # self.integrate_lineedit.setValidator(self.qtintvalidator)
        # self.integrate_lineedit.setText(str(self.input_params['integration_depth']))
        # self.integrate_lineedit.editingFinished.connect(self.update_integration_size)

        # ROI
        self.radiusroi_label = PyQt4.QtGui.QLabel(self)
        self.radiusroi_label.setText('Radius ROI (Min,Max):')
        self.radiusroimin_lineedit = PyQt4.QtGui.QLineEdit(self)
        self.radiusroimin_lineedit.setValidator(self.qtfloatvalidator)
        self.radiusroimin_lineedit.setText(str(self.input_params['radiusroimin']))
        self.radiusroimin_lineedit.editingFinished.connect(self.update_roi)
        self.radiusroimax_lineedit = PyQt4.QtGui.QLineEdit(self)
        self.radiusroimax_lineedit.setValidator(self.qtfloatvalidator)
        self.radiusroimax_lineedit.setText(str(self.input_params['radiusroimax']))
        self.radiusroimax_lineedit.editingFinished.connect(self.update_roi)

        # cenre i j
        self.centreij_label = PyQt4.QtGui.QLabel(self)
        self.centreij_label.setText('centre (i,j):')
        self.centrei_lineedit = PyQt4.QtGui.QLineEdit(self)
        self.centrei_lineedit.setValidator(self.qtintvalidator)
        self.centrei_lineedit.setText(str(self.input_params['centrei']))
        self.centrei_lineedit.editingFinished.connect(self.update_centre)
        self.centrej_lineedit = PyQt4.QtGui.QLineEdit(self)
        self.centrej_lineedit.setValidator(self.qtintvalidator)
        self.centrej_lineedit.setText(str(self.input_params['centrej']))
        self.centrej_lineedit.editingFinished.connect(self.update_centre)
 
        # directory
        self.directory_label = PyQt4.QtGui.QLabel(self)
        self.directory_label.setText('Directory to scan use /\'s:')
        self.directory_lineedit = PyQt4.QtGui.QLineEdit(self)
        self.directory_lineedit.setText(self.input_params['directory'])
        self.directory_lineedit.editingFinished.connect(self.update_directory)

        # live stream checkbox
        self.livestream_checkbox = PyQt4.QtGui.QCheckBox('Live Stream', self)
        self.livestream_checkbox.stateChanged.connect(self.update_livestream)
        if self.input_params['livestream'] :
            self.livestream_checkbox.setChecked(True)

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

        # threshold adus
        self.threshold_label = PyQt4.QtGui.QLabel(self)
        self.threshold_label.setText('threshold adus for a hit:')
        self.threshold_lineedit = PyQt4.QtGui.QLineEdit(self)
        self.threshold_lineedit.setText(str(self.input_params['threshold']))
        self.threshold_lineedit.setValidator(self.qtintvalidator)
        self.threshold_lineedit.editingFinished.connect(self.update_threshold)

        # comment
        self.saveComment_label = PyQt4.QtGui.QLabel(self)
        self.saveComment_label.setText('Save comment for current figure:')
        self.saveComment_lineedit = PyQt4.QtGui.QLineEdit(self)
        self.saveComment_lineedit.editingFinished.connect(self.update_saveComment)

        self.backgroundFile_label = PyQt4.QtGui.QLabel(self)
        self.backgroundFile_label.setText('Select background file for substraction:')
        self.backgroundFile_lineedit = PyQt4.QtGui.QLineEdit(self)
        #self.backgroundFile_lineedit.editingFinished.connect(self.update_display_data())
        self.backgroundFile_browse = PyQt4.QtGui.QPushButton(self)
        self.backgroundFile_browse.setText('Browse')
        self.backgroundFile_browse.clicked.connect(self.selectBackground)

        # Add all the stuff to the layout
        hlayouts = []

        # ..statusbar.
        # cspad | hist
        #         ----
        #         phot
        #         ----
        #         radial profile
        #  live stream

        hlayouts.append(PyQt4.QtGui.QHBoxLayout())
        hlayouts[-1].addWidget(self.statusbarWidget)

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

        hlayouts.append(PyQt4.QtGui.QHBoxLayout())

        # centre i j layout
        hlayouts[-1].addWidget(self.centreij_label)
        hlayouts[-1].addWidget(self.centrei_lineedit)
        hlayouts[-1].addWidget(self.centrej_lineedit)

        # threshold layout
        hlayouts[-1].addWidget(self.threshold_label)
        hlayouts[-1].addWidget(self.threshold_lineedit)

        # Save comment
        hlayouts.append(PyQt4.QtGui.QHBoxLayout())	
        hlayouts[-1].addWidget(self.saveComment_label)
        hlayouts[-1].addWidget(self.saveComment_lineedit)

        # Background file
        hlayouts.append(PyQt4.QtGui.QHBoxLayout())
        hlayouts[-1].addWidget(self.backgroundFile_label)
        hlayouts[-1].addWidget(self.backgroundFile_lineedit)
        hlayouts[-1].addWidget(self.backgroundFile_browse)

        # stack everything vertically 
        vlayout = PyQt4.QtGui.QVBoxLayout()
        for hlayout in hlayouts :
            vlayout.addLayout(hlayout)

        self.setLayout(vlayout)
        self.resize(800,800)
        self.show()

    def selectBackground(self):
        self.backgroundFile_lineedit.setText(PyQt4.QtGui.QFileDialog.getOpenFileName())
        print self.backgroundFile_lineedit.text()
        self.display_data['background_image'] = load_png_Valerio(str(self.backgroundFile_lineedit.text()))

    def update_livestream(self, state):
        if state == PyQt4.QtCore.Qt.Checked:
            self.input_params['livestream'] = True
        else :
            self.input_params['livestream'] = False

    def update_integration_size(self):
        pass

    def clearHistogram(self):
        self.histWidget.clear()
        self.countsWidget.clear()
        self.radProfWidget.clear()
        self.display_data['cspad_raw_counts'] = []
        self.display_data['cspad_hit_rate'] = []

    def update_roi(self):
        new_radiusroimin = float(self.radiusroimin_lineedit.text())
        new_radiusroimax = float(self.radiusroimax_lineedit.text())
        if new_radiusroimin == 0 and new_radiusroimax == 0 :
            self.ring_canvas.clear()
        if new_radiusroimin != self.input_params['radiusroimin'] or new_radiusroimax != self.input_params['radiusroimax'] :
            if new_radiusroimin <= new_radiusroimax :
                self.input_params['radiusroimin'] = new_radiusroimin
                self.input_params['radiusroimax'] = new_radiusroimax
                temp = (self.cspad_rads > new_radiusroimin) * (self.cspad_rads < new_radiusroimax)
                self.roi = numpy.where(temp == False)
                print 'New ROI:', new_radiusroimin, new_radiusroimax
                self.update_display_data()
        
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
        inp_h5 = f.create_group('input_params')
        for k in self.input_params.keys() :
            print 'writing ', k, ' of type ', type(self.input_params[k])
            if self.input_params[k] is not None :
                inp_h5.create_dataset(k, data = self.input_params[k])
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
        input_params = f['input_params']
        for k in input_params.keys() :
            print 'reading ', k, ' of type ', type(input_params[k].value)
            self.input_params[k] = input_params[k].value
        f.close()
        
        self.display_data['cspad_raw_counts']       = list(self.display_data['cspad_raw_counts'])

        self.directory_lineedit.setText(self.input_params['directory'])
        self.threshold_lineedit.setText(str(self.input_params['threshold']))
        self.radiusroimin_lineedit.setText(str(self.input_params['radiusroimin']))
        self.radiusroimax_lineedit.setText(str(self.input_params['radiusroimax']))

        self.update_image()
        print 'done'

    def update_directory(self):
        if self.input_params['directory'] != os.path.abspath(str(self.directory_lineedit.text())):
            self.input_params['directory'] = os.path.abspath(str(self.directory_lineedit.text()))
            self.directory_lineedit.setText(self.input_params['directory'])
            
            # fill the files dict --> global variable files
            fill_current_file_list(self.input_params['directory'])

    def back_event(self):
        self.display_data['event_id'] -= 1
        self.update_display_data()

    def forward_event(self):
        self.display_data['event_id'] += 1
        self.update_display_data()

    def update_threshold(self):
        self.input_params['threshold'] = int(self.threshold_lineedit.text())
        self.threshold_line.setValue(self.input_params['threshold'])
        self.update_display_data()

    def update_threshold_graph(self):
        self.input_params['threshold'] = int(self.threshold_line.value())
        self.threshold_lineedit.setText(str(self.input_params['threshold']))
        self.update_display_data()

    def update_centre(self):
        self.input_params['centrei'] = int(self.centrei_lineedit.text())
        self.input_params['centrej'] = int(self.centrej_lineedit.text())
        print 'new centre', self.input_params['centrei'], self.input_params['centrej'] 
        self.update_display_data()

    def update_saveComment(self):
        if str(self.saveComment_lineedit.text()) != '' :
            with open("file_comments.txt", "a") as myfile:
	        myfile.write(os.path.abspath(self.display_data['last_file']) +': ' +str(self.saveComment_lineedit.text()) + '\n' )
                self.statusbarWidget.showMessage("Saved comment for file "+ self.display_data['last_file'])
	        myfile.close()
            self.saveComment_lineedit.setText('')


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)    # allow Control-C
    app = PyQt4.QtGui.QApplication(sys.argv)
    ex = MainFrame()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()    

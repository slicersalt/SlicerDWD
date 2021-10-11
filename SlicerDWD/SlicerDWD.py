import csv
import random
import os.path
import warnings

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import qt

import vtk
from vtk.numpy_interface import dataset_adapter as dsa

import numpy as np

import scipy

import sklearn.metrics

try:
    import dwd
except ModuleNotFoundError:
    slicer.util.pip_install('dwd')
    import dwd


class SlicerDWD(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "DWD Classification"
        self.parent.categories = ["Shape Analysis"]
        self.parent.dependencies = []
        self.parent.contributors = ["David Allemang (Kitware, Inc.)"]
        self.parent.helpText = """
Perform and visualize Distance-Weighted Discrimination on correspondent populations.
"""
        self.parent.acknowledgementText = """
This module depends on the dwd Python package, originally implemented by 
<a href="https://idc9.github.io/">Iain Carmichael</a>. For details on DWD see 
<a href="https://amstat.tandfonline.com/doi/abs/10.1198/016214507000001120">
Marron et al 2007</a>,
<a href="https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssb.12244">
Wang and Zou 2018</a>.
"""


class SlicerDWDWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """Called when the user opens the module the first time and the widget is
        initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

        self.logic = None
        self.classifier = None
        self.trainData = None
        self.testData = None
        self.metrics = None

        self.boldFont = None

    def setup(self):
        """Called when the user opens the module the first time and the widget is
        initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SlicerDWD.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = SlicerDWDLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # self.ui.pathMetrics.currentPathChanged.connect()

        # couple checkboxes with optional parameters
        self.ui.chkSample.stateChanged.connect(self.updateSpnSample)
        self.ui.chkSample.stateChanged.connect(self.updatePathTest)
        self.ui.chkAutoTune.stateChanged.connect(self.updateSpnTuningC)
        self.ui.chkSaveResults.stateChanged.connect(self.updatePathResults)

        # changing these may affect if we can train
        self.ui.pathTrain.currentPathChanged.connect(self.updateBtnTrain)

        # changing these may affect if we can test
        self.ui.chkSaveResults.stateChanged.connect(self.updateBtnTest)
        self.ui.pathResults.currentPathChanged.connect(self.updateBtnTest)
        self.ui.chkSample.stateChanged.connect(self.updateBtnTest)
        self.ui.pathTest.currentPathChanged.connect(self.updateBtnTest)

        # register "real" actions
        self.ui.btnTrain.clicked.connect(self.btnTrainClicked)
        self.ui.btnTest.clicked.connect(self.btnTestClicked)
        self.ui.btnMean.clicked.connect(self.btnMeanClicked)
        self.ui.btnKDE.clicked.connect(self.btnKDEClicked)

        self.boldFont = self.ui.SlicerDWD.font
        self.boldFont.setWeight(qt.QFont.Bold)

    @property
    def trainDataReady(self):
        """Whether inputs are ready to train"""
        fileValid = os.path.isfile(self.ui.pathTrain.currentPath)
        return fileValid

    @property
    def testDataReady(self):
        """Whether inputs are ready to test"""
        sampling = self.ui.chkSample.checked
        fileValid = os.path.isfile(self.ui.pathTest.currentPath)
        return (sampling and self.trainDataReady) or fileValid

    @property
    def testResultsReady(self):
        """Whether outputs are ready to test"""
        saving = self.ui.chkSaveResults.checked
        fileValid = os.path.isfile(self.ui.pathResults.currentPath)
        return not saving or fileValid

    def updateBtnTrain(self):
        self.ui.btnTrain.enabled = self.trainDataReady

    def updateBtnTest(self):
        self.ui.btnTest.enabled = all((
            self.classifier,
            self.testDataReady,
            self.testResultsReady
        ))

    def updateBtnMean(self):
        self.ui.btnMean.enabled = all((
            self.classifier,
            self.testDataReady,
            self.testResultsReady
        ))

    def updateBtnKDE(self):
        self.ui.btnKDE.enabled = all((
            self.classifier,
            self.testDataReady,
            self.testResultsReady
        ))

    def updateBtnCorr(self):
        pass

    def updateComCorr(self):
        pass

    def updateSpnSample(self):
        sampling = self.ui.chkSample.checked
        self.ui.spnSample.enabled = sampling
        self.ui.spnSampleLabel.enabled = sampling

    def updatePathTest(self):
        sampling = self.ui.chkSample.checked
        self.ui.pathTest.enabled = not sampling
        self.ui.pathTestLabel.enabled = not sampling

    def updateSpnTuningC(self):
        autotuning = self.ui.chkAutoTune.checked
        self.ui.spnTuningC.enabled = not autotuning
        self.ui.spnTuningC.enabled = not autotuning

        if self.classifier:
            self.ui.spnTuningC.value = self.classifier.C

    def updatePathResults(self):
        saving = self.ui.chkSaveResults.checked
        self.ui.pathResults.enabled = saving
        self.ui.pathResultsLabel.enabled = saving

    def btnTrainClicked(self):
        """Called when the "Train Classifier" button is clicked."""

        self.classifier = None

        self.trainData = self.logic.buildCases(self.ui.pathTrain.currentPath)
        if self.ui.chkSample.checked:
            part = self.ui.spnSample.value / 100
            self.trainData, self.testData = self.logic.splitCases(self.trainData, part)

        c = 'auto' if self.ui.chkAutoTune.checked else self.ui.spnTuningC.value

        self.classifier = self.logic.train(self.trainData, c)
        self.ui.spnTuningC.value = self.classifier.C

        self.classifierUpdated()

        results = self.logic.compute(self.classifier, self.trainData)
        self.populateStatsTable(self.ui.tblTrainStats, results)

    def classifierUpdated(self):
        """Update any UI that should be enabled or disabled when the classifier is
        created or removed.
        """
        self.updateBtnTest()
        self.updateBtnMean()
        self.updateBtnKDE()
        self.updateBtnCorr()
        self.updateComCorr()

    def btnTestClicked(self):
        """Called when the "Test Classifier" button is clicked."""

        if not self.ui.chkSample.checked:
            self.testData = self.logic.buildCases(self.ui.pathTrain.currentPath)

        results = self.logic.compute(self.classifier, self.testData)
        self.populateStatsTable(self.ui.tblTestStats, results)
        self.populateResultsTable(self.ui.tblTestResults, results)

        # if chkSaveResults, save to pathResults

    def btnMeanClicked(self):
        """Called when the "Compute Projected Mean Shape" button is clicked."""

        if not self.ui.chkSample.checked:
            self.testData = self.logic.buildCases(self.ui.pathTrain.currentPath)

        mean = self.logic.meanShape(self.classifier, self.testData, factor=50.0)

        model = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLModelNode', 'Projected Mean Shape'
        )
        model.CreateDefaultDisplayNodes()

        model.SetAndObservePolyData(mean.VTKObject)
        model.GetDisplayNode().SetVisibility(False)

        module = slicer.modules.shapepopulationviewer
        spv = module.widgetRepresentation()
        slicer.util.selectModule(module)
        spv.loadModel(model)

    def btnKDEClicked(self):
        """Called when the "Show KDE" button is clicked."""

        if not self.ui.chkSample.checked:
            self.testData = self.logic.buildCases(self.ui.pathTrain.currentPath)

        results = self.logic.compute(self.classifier, self.testData)
        kernel_all = scipy.stats.gaussian_kde(results['distance'])
        results['rand'] = np.random.normal(
            size=results['distance'].shape
        ) * kernel_all(results['distance']) * 0.05 + 0.007

        # todo encapsulate kde plot work

        kde_space = np.linspace(results['distance'].min(), results['distance'].max())
        kernels = {
            '{}'.format(t): scipy.stats.gaussian_kde(
                results['distance'][results['actual'] == t]
            ) for t in np.unique(results['actual'])
        }

        results_table = self.logic.table({
            'distance': results['distance'],
            # skip results['actual'] and results['predict'] since they are string
            # arrays, incompatible with vtk.util.numpy_to_vtk
            'rand': results['rand']
        }, 'DWD Results')

        kde_table = self.logic.table({
            'x': kde_space,
            'all': kernel_all(kde_space),
            **{t: kernel(kde_space) for t, kernel in kernels.items()}
        }, 'DWD Results KDE')

        chart = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLPlotChartNode')
        chart.SetTitle('DWD Distance Distribution')
        chart.SetXAxisTitle('Distance')
        chart.SetYAxisTitle('Density')

        scatter = self.logic.scatterPlot(
            chart, results_table, 'distance', 'rand',
            name='Samples'
        )
        scatter.SetLineStyle(scatter.LineStyleNone)
        scatter.SetMarkerStyle(scatter.MarkerStyleCross)
        scatter.SetColor(0, 0, 0)

        full = self.logic.scatterPlot(
            chart, kde_table, 'x', 'all',
            name='All Classes'
        )
        full.SetColor(0, 0, 0)

        colors = [
            (1.0, 0.7, 0.2),
            (0.2, 0.6, 0.8),
        ]

        for key, color in zip(kernels, colors):
            plot = self.logic.scatterPlot(
                chart, kde_table, 'x', key,
                name=key
            )
            plot.SetColor(*color)

        self.logic.show(chart)

        mgr = slicer.app.layoutManager()
        mgr.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpPlotView)

    def populateStatsTable(self, tbl, results):
        clss = np.unique(results['actual'])

        while tbl.rowCount > 2:
            tbl.removeRow(2)

        accuracy = sklearn.metrics.accuracy_score(
            results['actual'], results['predict']
        )

        it = qt.QTableWidgetItem()
        it.setText(format(accuracy, '.2%'))
        tbl.setItem(0, 1, it)

        for i, cls in enumerate(clss):
            precision = sklearn.metrics.precision_score(
                results['actual'], results['predict'],
                pos_label=cls
            )
            recall = sklearn.metrics.recall_score(
                results['actual'], results['predict'],
                pos_label=cls
            )

            tbl.insertRow(2 + i)

            it = tbl.item(0, 0).clone()  # copy "Accuracy" label style
            it.setText('{}'.format(cls))
            tbl.setItem(2 + i, 0, it)

            it = qt.QTableWidgetItem('{:.2%}'.format(precision))
            tbl.setItem(2 + i, 1, it)

            it = qt.QTableWidgetItem('{:.2%}'.format(recall))
            tbl.setItem(2 + i, 2, it)

        tbl.resizeColumnToContents(0)

    def populateResultsTable(self, tbl, results):
        count = len(results['distance'])

        while tbl.rowCount > count:
            tbl.removeRow(0)

        while tbl.rowCount < count:
            tbl.insertRow(0)

        for i in range(count):
            filename = os.path.basename(results['filename'][i])
            actual = str(results['actual'][i])
            predict = str(results['predict'][i])
            distance = '{:.3f}'.format(results['distance'][i])

            tbl.setItem(i, 0, qt.QTableWidgetItem(filename))
            tbl.setItem(i, 1, qt.QTableWidgetItem(actual))
            tbl.setItem(i, 2, qt.QTableWidgetItem(predict))
            tbl.setItem(i, 3, qt.QTableWidgetItem(distance))


    def cleanup(self):
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def onSceneStartClose(self, caller, event):
        """Called just before the scene is closed."""
        # # Parameter node will be reset, do not use it anymore
        # self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """Called just after the scene is closed."""
        # # If this module is shown while the scene is closed then recreate a new
        # # parameter node immediately
        # if self.parent.isEntered:
        #     self.initializeParameterNode()


class SlicerDWDLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual computation done by your module.
    The interface should be such that other python code can import this class and
    make use of the functionality without requiring an instance of the Widget. Uses
    ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer
    /ScriptedLoadableModule.py
    """

    def __init__(self):
        """Called when the logic class is instantiated. Can be used for initializing
        member variables.
        """
        super().__init__(self)

    # region dataset util todo encapsulate

    def buildCases(self, csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            # could do with list() but this enforces row structure
            return [(path, group) for path, group in reader]

    def splitCases(self, cases, test_part):
        random.shuffle(cases)
        idx = int(len(cases) * test_part)
        test, train = cases[:idx], cases[idx:]
        return train, test

    def make_xy(self, cases):
        X = np.array([
            self.read_vtk(path).Points.flatten()
            for path, group in cases
        ])
        y = np.array([group for path, group in cases])

        return X, y

    # endregion

    # region io util todo encapsulate

    def read_vtk(self, path):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        return dsa.WrapDataObject(reader.GetOutput())

    def save_vtk(self, path, pdata: dsa.PolyData):
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(pdata.VTKObject)
        writer.Update()

    def copy_dsa(self, pdata: dsa.PolyData):
        res = dsa.WrapDataObject(vtk.vtkPolyData())
        res.VTKObject.DeepCopy(pdata.VTKObject)
        return res

    # endregion

    def train(self, trainData, c):
        X, y = self.make_xy(trainData)

        classifier = dwd.DWD(c)

        with warnings.catch_warnings(record=True):
            classifier.fit(X, y)

        slicer.util.showStatusMessage('DWD: {}'.format(classifier), 5000)

        return classifier

    def meanShape(self, classifier, cases, factor=1.0):
        base = self.read_vtk(cases[0][0])
        X, _ = self.make_xy(cases)

        shape = base.Points.shape
        d, i = self.direction(classifier)

        mean = np.mean(X, axis=0)
        mean -= d * (i - np.dot(mean, d))
        mean = mean.reshape(shape)

        mean_dir = d * factor
        mean_dir = mean_dir.reshape(shape)

        out = self.copy_dsa(base)
        out.Points = mean
        out.PointData.append(mean_dir, 'Direction')

        return out

    def compute(self, classifier, cases):
        d, i = self.direction(classifier)
        X, y = self.make_xy(cases)

        filename = np.array([row[0] for row in cases])
        actual = y
        predict = classifier.predict(X)
        distance = X.dot(d) - i

        return {
            'filename': filename,
            'actual': actual,
            'predict': predict,
            'distance': distance
        }

    def direction(self, classifier):
        """Return the DWD direction and intercept. The separating hyperplane is of the
        form 'p.d = d.i', where '.' is the dot product. If 'p.d < d.i', then 'p' is label
        0. If 'p.d > d.i', then 'p' is label 1.
        """

        direction = classifier.coef_.reshape(-1)
        intercept = -float(classifier.intercept_)
        return direction, intercept

    # region plot util todo encapsulate

    def table(self, columns, name='Table'):
        tableNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTableNode', name)
        table = tableNode.GetTable()

        for name, data in columns.items():
            print(name, data)
            arr = vtk.util.numpy_support.numpy_to_vtk(data)
            arr.SetName(name)
            table.AddColumn(arr)

        return tableNode

    def scatterPlot(self, chartNode, tableNode, x, y, name='Series'):
        psn = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLPlotSeriesNode', name)
        psn.SetAndObserveTableNodeID(tableNode.GetID())
        psn.SetXColumnName(x)
        psn.SetYColumnName(y)
        psn.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        psn.SetMarkerStyle(psn.MarkerStyleNone)
        psn.SetUniqueColor()

        chartNode.AddAndObservePlotSeriesNodeID(psn.GetID())

        return psn

    def show(self, chartNode):
        plots = slicer.modules.plots.logic()

        mgr = slicer.app.layoutManager()
        mgr.setLayout(plots.GetLayoutWithPlot(mgr.layout))

        widget = mgr.plotWidget(0)
        viewNode = widget.mrmlPlotViewNode()
        viewNode.SetPlotChartNodeID(chartNode.GetID())

    # endregion


class DWDTest(ScriptedLoadableModuleTest):
    """This is the test case for your scripted module. Uses
    ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer /ScriptedLoadableModule.py
    """

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.delayDisplay("Test and testing data not yet implemented.")
        self.setUp()

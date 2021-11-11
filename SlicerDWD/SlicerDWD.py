import csv
import importlib
import os.path
import warnings
from collections import namedtuple

import ctk
import numpy as np
import qt
import sklearn.metrics
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import SlicerDWDUtils as util

DWD_VERSION = '1.0.2'

try:
    import dwd
except ModuleNotFoundError:
    slicer.util.pip_install('dwd=={}'.format(DWD_VERSION))
    import dwd

if dwd.__version__ != DWD_VERSION:
    slicer.util.pip_install('dwd=={}'.format(DWD_VERSION))
    importlib.reload(dwd)

    def request_restart():
        if slicer.util.confirmYesNoDisplay(
            'SlicerDWD has updated Python dependencies. '
            'It is recommended to restart Slicer. Restart now?'
        ):
            slicer.util.restart()
            slicer.util.exit()

    slicer.app.startupCompleted.connect(
        # need to use a singeShot timer so confirmYesNoDisplay has a valid QT context.
        lambda: qt.QTimer.singleShot(0, request_restart)
    )


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

        # Dynamically enable/disable UI elements
        self.ui.chkSample.stateChanged.connect(self.updateUI)
        self.ui.chkAutoTune.stateChanged.connect(self.updateUI)
        self.ui.chkSaveResults.stateChanged.connect(self.updateUI)
        self.ui.pathResults.currentPathChanged.connect(self.updateUI)
        self.ui.pathTrain.currentPathChanged.connect(self.updateUI)
        self.ui.pathTest.currentPathChanged.connect(self.updateUI)
        self.ui.pathMetrics.currentPathChanged.connect(self.updateUI)

        self.ui.pathMetrics.currentPathChanged.connect(self.populateMetricsOptions)

        # register actions
        self.ui.btnTrain.clicked.connect(self.btnTrainClicked)
        self.ui.btnTest.clicked.connect(self.btnTestClicked)
        self.ui.btnMean.clicked.connect(self.btnMeanClicked)
        self.ui.btnKDE.clicked.connect(self.btnKDEClicked)
        self.ui.btnCorr.clicked.connect(self.btnCorrClicked)

        # create bold "header" font for stats tables
        self.boldFont = self.ui.SlicerDWD.font
        self.boldFont.setWeight(qt.QFont.Bold)

        # setup ctkPathLineEdit filters. Ideally this would be done in QML but the
        # properties don't seem to be set up so that's possible.
        self.ui.pathResults.filters |= ctk.ctkPathLineEdit.Writable
        self.ui.pathTrain.nameFilters += ("*.csv",)
        self.ui.pathTest.nameFilters += ("*.csv",)
        self.ui.pathMetrics.nameFilters += ("*.csv",)
        self.ui.pathResults.nameFilters += ("*.csv",)

    @property
    def metricsReady(self):
        """Whether metrics are ready to load"""
        fileValid = os.path.isfile(self.ui.pathMetrics.currentPath)
        return fileValid

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
        dirName = os.path.dirname(self.ui.pathResults.currentPath)
        dirValid = os.path.isdir(dirName)
        return not saving or dirValid

    def updateUI(self):
        sampling = self.ui.chkSample.checked
        self.ui.spnSample.enabled = sampling
        self.ui.spnSampleLabel.enabled = sampling
        self.ui.pathTest.enabled = not sampling
        self.ui.pathTestLabel.enabled = not sampling

        autotuning = self.ui.chkAutoTune.checked
        self.ui.spnTuningC.enabled = not autotuning
        self.ui.spnTuningC.enabled = not autotuning
        if self.classifier:
            self.ui.spnTuningC.value = self.classifier.C

        saving = self.ui.chkSaveResults.checked
        self.ui.pathResults.enabled = saving
        self.ui.pathResultsLabel.enabled = saving

        self.ui.btnTrain.enabled = self.trainDataReady

        self.ui.btnTest.enabled = all(
            (self.classifier, self.testDataReady, self.testResultsReady)
        )

        self.ui.btnMean.enabled = all(
            (self.classifier, self.testDataReady)
        )

        self.ui.btnKDE.enabled = all(
            (self.classifier, self.testDataReady)
        )

        self.ui.btnCorr.enabled = self.ui.comCorr.enabled = all(
            (self.classifier, self.testDataReady, self.metricsReady)
        )

    def btnTrainClicked(self):
        """Called when the "Train Classifier" button is clicked."""

        self.classifier = None

        with open(self.ui.pathTrain.currentPath) as f:
            self.trainData = util.load_cases(f)
        if self.ui.chkSample.checked:
            part = self.ui.spnSample.value / 100
            self.trainData, self.testData = util.split_cases(self.trainData, part)

        c = "auto" if self.ui.chkAutoTune.checked else self.ui.spnTuningC.value

        self.classifier = self.logic.train(self.trainData, c)
        self.ui.spnTuningC.value = self.classifier.C

        self.updateUI()

        results = self.logic.compute(self.classifier, self.trainData)
        self.populateStatsTable(self.ui.tblTrainStats, results)

    def btnTestClicked(self):
        """Called when the "Test Classifier" button is clicked."""

        if not self.ui.chkSample.checked:
            with open(self.ui.pathTrain.currentPath) as f:
                self.testData = util.load_cases(f)

        results = self.logic.compute(self.classifier, self.testData)
        self.populateStatsTable(self.ui.tblTestStats, results)
        self.populateResultsTable(self.ui.tblTestResults, results)

        if self.ui.chkSaveResults.checked:
            self.logic.saveResults(results, self.ui.pathResults.currentPath)

    def btnMeanClicked(self):
        """Called when the "Compute Projected Mean Shape" button is clicked."""

        if not self.ui.chkSample.checked:
            with open(self.ui.pathTrain.currentPath) as f:
                self.testData = util.load_cases(f)

        mean = self.logic.meanShape(self.classifier, self.testData, factor=50.0)

        model = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", "Projected Mean Shape"
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
            with open(self.ui.pathTrain.currentPath) as f:
                self.testData = util.load_cases(f)

        res = self.logic.compute(self.classifier, self.testData)
        kres = util.kde(res.distance, res.actual, rescale=True)

        # to vertically center the samples in the plot
        loc = kres.kernel.max() / 2
        stddev = loc / 20

        table = util.make_table(
            "DWD Results",
            distance=res.distance,
            random=np.random.normal(loc, stddev, size=res.distance.shape),
        )
        kdeTable = util.make_table(
            "KDE Results",
            x=kres.x,
            all=kres.kernel,
            **dict(zip(kres.ulabels, kres.kernels))
        )

        colors = [
            (1.0, 0.7, 0.2),  # orange
            (0.2, 0.6, 0.8),  # blue
            (1.0, 0.4, 0.3),  # red
            (0.2, 1.0, 0.4),  # green
        ]

        allSeries = [
            util.make_series(
                "Samples", table, "distance", "random",
                lineStyle=slicer.vtkMRMLPlotSeriesNode.LineStyleNone,
                markerStyle=slicer.vtkMRMLPlotSeriesNode.MarkerStyleCross,
            ),
            util.make_series("All Classes", kdeTable, "x", "all"),
        ]
        allSeries += [
            util.make_series(label, kdeTable, "x", label, color=color)
            for label, color in zip(kres.ulabels, colors)
        ]

        chart = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
        chart.SetTitle("DWD Distance Distribution")
        chart.SetXAxisTitle("Distance")
        chart.SetYAxisTitle("Density")

        for series in allSeries:
            chart.AddAndObservePlotSeriesNodeID(series.GetID())

        plots = slicer.modules.plots.logic()
        plots.ShowChartInLayout(chart)

    def btnCorrClicked(self):
        if not self.ui.chkSample.checked:
            with open(self.ui.pathTrain.currentPath) as f:
                self.testData = util.load_cases(f)

        result = self.logic.compute(self.classifier, self.testData)
        corr_result = self.logic.corrWith(
            result, self.ui.pathMetrics.currentPath, self.ui.comCorr.currentText
        )

        field = self.ui.comCorr.currentText
        table = util.make_table(
            'Correlation',
            distance=corr_result.distance,
            metric=corr_result.metric,
            fit=corr_result.fit,
        )

        scatter = util.make_series(
            f'Samples', table, 'metric', 'distance',
            lineStyle=slicer.vtkMRMLPlotSeriesNode.LineStyleNone,
            markerStyle=slicer.vtkMRMLPlotSeriesNode.MarkerStyleCross,
        )

        fit = util.make_series(
            f'Linear Fit', table, 'metric', 'fit',
            lineStyle=slicer.vtkMRMLPlotSeriesNode.LineStyleSolid,
            markerStyle=slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone,
        )

        chart = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLPlotChartNode')
        chart.SetTitle(f'Distance vs {field}')
        chart.SetXAxisTitle(f'{field}')
        chart.SetYAxisTitle('Distance')

        chart.AddAndObservePlotSeriesNodeID(scatter.GetID())
        chart.AddAndObservePlotSeriesNodeID(fit.GetID())

        plots = slicer.modules.plots.logic()
        plots.ShowChartInLayout(chart)

    def populateStatsTable(self, tbl, results):
        while tbl.rowCount > 2:
            tbl.removeRow(2)

        stats = self.logic.stats(results)

        it = qt.QTableWidgetItem()
        it.setText(format(stats.accuracy, '.2%'))
        tbl.setItem(0, 1, it)

        for i, label in enumerate(stats.labels):
            tbl.insertRow(2 + i)

            it = tbl.item(0, 0).clone()  # copy "Accuracy" label style
            it.setText("{}".format(label))
            tbl.setItem(2 + i, 0, it)

            it = qt.QTableWidgetItem("{:.2%}".format(stats.precision[i]))
            tbl.setItem(2 + i, 1, it)

            it = qt.QTableWidgetItem("{:.2%}".format(stats.recall[i]))
            tbl.setItem(2 + i, 2, it)

        tbl.resizeColumnToContents(0)

    def populateResultsTable(self, tbl, results):
        count = len(results.distance)

        while tbl.rowCount > count:
            tbl.removeRow(0)

        while tbl.rowCount < count:
            tbl.insertRow(0)

        for i in range(count):
            filename = os.path.basename(results.filename[i])
            actual = str(results.actual[i])
            predict = str(results.predict[i])
            distance = "{:.3f}".format(results.distance[i])

            tbl.setItem(i, 0, qt.QTableWidgetItem(filename))
            tbl.setItem(i, 1, qt.QTableWidgetItem(actual))
            tbl.setItem(i, 2, qt.QTableWidgetItem(predict))
            tbl.setItem(i, 3, qt.QTableWidgetItem(distance))

    def populateMetricsOptions(self):
        if not self.metricsReady:
            pass

        with open(self.ui.pathMetrics.currentPath) as f:
            reader = csv.reader(f)
            fields = next(reader)

        # assume the first field is the case ID/filename
        fields = fields[1:]

        self.ui.comCorr.clear()
        for field in fields:
            self.ui.comCorr.addItem(field)

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


ComputeResult = namedtuple('ComputeResult', ['filename', 'actual', 'predict', 'distance'])
StatsResult = namedtuple('StatsResult', ['accuracy', 'labels', 'precision', 'recall'])
CorrResult = namedtuple('CorrResult', ['coeff', 'filename', 'distance', 'metric', 'fit'])


class SlicerDWDLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual computation done by your module. The
    interface should be such that other python code can import this class and make use
    of the functionality without requiring an instance of the Widget. Uses
    ScriptedLoadableModuleLogic base class, available at:

    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """Called when the logic class is instantiated. Can be used for initializing
        member variables.
        """
        super().__init__(self)

    def train(self, trainData, c):
        X, y = util.make_xy(trainData)

        classifier = dwd.DWD(c)

        with warnings.catch_warnings(record=True):
            classifier.fit(X, y)

        slicer.util.showStatusMessage("DWD: {}".format(classifier), 5000)

        return classifier

    def meanShape(self, classifier, cases, factor=1.0):
        base = util.read_vtk(cases[0][0])
        X, _ = util.make_xy(cases)

        shape = base.Points.shape
        d, i = util.direction(classifier)

        mean = np.mean(X, axis=0)
        mean -= d * (i - np.dot(mean, d))
        mean = mean.reshape(shape)

        mean_dir = d * factor
        mean_dir = mean_dir.reshape(shape)

        out = util.copy_dsa(base)
        out.Points = mean
        out.PointData.append(mean_dir, "Direction")

        return out

    def compute(self, classifier, cases):
        d, i = util.direction(classifier)
        X, y = util.make_xy(cases)

        filename = np.array([path for path, group in cases])
        actual = y
        predict = classifier.predict(X)
        distance = X.dot(d) - i

        return ComputeResult(filename, actual, predict, distance)

    def stats(self, results: ComputeResult):
        labels = np.unique(results.actual)

        accuracy = sklearn.metrics.accuracy_score(results.actual, results.predict)

        precision = [
            sklearn.metrics.precision_score(
                results.actual, results.predict, pos_label=label
            ) for label in labels
        ]

        recall = [
            sklearn.metrics.recall_score(
                results.actual, results.predict, pos_label=label
            ) for label in labels
        ]

        return StatsResult(accuracy, labels, precision, recall)

    def saveResults(self, results, path):
        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Actual', 'Predict', 'Distance'])

            for row in zip(
                results.filename,
                results.actual,
                results.predict,
                results.distance,
            ):
                writer.writerow(row)

    def corrWith(self, results: ComputeResult, metricsPath, field):
        with open(metricsPath) as f:
            reader = csv.reader(f)

            fields = next(reader)
            idx = fields.index(field)

            data = {row[0]: row[idx] for row in reader}

        results.distance.sort()

        # metric data in correspondance with results
        metric = np.array([data[name] for name in results.filename])

        # correlation only works on numeric data
        metric = metric.astype(np.float)

        corr = np.corrcoef(results.distance, metric)

        # linear regression
        x = metric
        y = results.distance
        A = np.vstack([x, np.ones_like(x)]).T

        # print(x.shape)
        # print(y.shape)
        # print(A.shape)

        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        # print((m, c))
        fit = m * x + c

        return CorrResult(corr, results.filename, results.distance, metric, fit)


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

import csv
import warnings

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import vtk
from vtk.numpy_interface import dataset_adapter as dsa

import numpy as np

import scipy

try:
    import dwd
except ModuleNotFoundError:
    slicer.util.pip_install('dwd')
    import dwd


class DWD(ScriptedLoadableModule):
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


class DWDWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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

        self.trainCases = None
        self.testCases = None

    def setup(self):
        """Called when the user opens the module the first time and the widget is
        initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/DWD.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = DWDLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        self.ui.path_train.currentPathChanged.connect(self.trainPathChanged)
        self.ui.chk_autoTune.stateChanged.connect(self.autoTuneStateChanged)
        self.ui.btn_train.clicked.connect(self.trainClicked)

        self.ui.path_test.currentPathChanged.connect(self.testPathChanged)

        self.ui.btn_mean.clicked.connect(self.meanClicked)
        self.ui.btn_kde.clicked.connect(self.kdeClicked)

        self.ui.tbl_stats.insertColumn(0)
        self.ui.tbl_stats.insertRow(0)
        self.ui.tbl_stats.insertRow(0)

        self.ui.tbl_stats.horizontalHeader().visible = False
        self.ui.tbl_stats.setVerticalHeaderLabels(['Accuracy', 'Errors'])

    def autoTuneStateChanged(self, enabled):
        """Called when the "Auto Tune" checkbox is changed."""
        self.ui.spn_tuningC.enabled = not enabled

    @property
    def tuningC(self):
        if self.ui.chk_autoTune.checked:
            return 'auto'
        return self.ui.spn_tuningC.value

    def trainPathChanged(self, path):
        """Called when the training dataset path is changed."""
        try:
            self.trainCases = self.logic.buildCases(path)
            self.ui.btn_train.enabled = bool(self.trainCases)
        except (OSError, ValueError):
            slicer.util.errorDisplay('Failed to load training dataset {}'.format(path))
            self.ui.btn_train.enabled = False

    def trainClicked(self):
        """Called when the "Train Classifier" button is clicked."""

        success = self.logic.train(
            self.trainCases,
            self.tuningC
        )

        self.ui.btn_test.enabled = success

        self.ui.btn_mean.enabled = success
        self.ui.btn_kde.enabled = success
        self.ui.btn_corr.enabled = success

    def testPathChanged(self, path):
        """Called when the testing dataset path is changed."""
        try:
            self.testCases = self.logic.buildCases(path)
            self.ui.btn_test.enabled = bool(self.testCases)
        except (OSError, ValueError):
            slicer.util.errorDisplay('Failed to load testing dataset {}'.format(path))
            self.ui.btn_test.enabled = False

    def meanClicked(self):
        """Called when the 'Compute Mean" button is clicked."""

        mean = self.logic.meanShape(self.trainCases, factor=50.0)

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

    def kdeClicked(self):
        """Called when the "Show KDE" button is clicked."""

        results = self.logic.compute(self.trainCases)
        kernel_all = scipy.stats.gaussian_kde(
            results['distance']
        )
        results['rand'] = np.random.normal(
            size=results['distance'].shape
        ) * kernel_all(results['distance']) * 0.05 + 0.007

        kde_space = np.linspace(results['distance'].min(), results['distance'].max())
        kernels = {
            'Class {}'.format(t): scipy.stats.gaussian_kde(
                results['distance'][results['actual'] == t]
            ) for t in np.unique(results['actual'])
        }

        kde_results = {
            'x': kde_space,
            'all': kernel_all(kde_space),
            **{t: kernel(kde_space) for t, kernel in kernels.items()}
        }

        results_table = self.logic.table(results, 'DWD Results')
        kde_table = self.logic.table(kde_results, 'DWD Results KDE')

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
            name='All Classes KDE'
        )
        full.SetColor(0, 0, 0)

        colors = [
            (1.0, 0.7, 0.2),
            (0.2, 0.6, 0.8),
        ]

        for key, color in zip(kernels, colors):
            plot = self.logic.scatterPlot(
                chart, kde_table, 'x', key,
                name=key + ' KDE'
            )
            plot.SetColor(*color)

        self.logic.show(chart)

        mgr = slicer.app.layoutManager()
        mgr.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpPlotView)

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


class DWDLogic(ScriptedLoadableModuleLogic):
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

        self.classifier = None

    def buildCases(self, csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            # could do with list() but this enforces row structure
            return [(path, group) for path, group in reader]

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

    def make_xy(self, cases):
        X = np.array([
            self.read_vtk(path).Points.flatten()
            for path, group in cases
        ])
        y = np.array([group for path, group in cases])

        return X, y

    def train(self, cases, c='auto'):
        """Train the model."""

        self.classifier = None

        slicer.util.showStatusMessage('DWD: Loading Dataset', 2000)
        X, y = self.make_xy(cases)

        slicer.util.showStatusMessage('DWD: Fitting classifier', 2000)
        self.classifier = dwd.DWD(c)
        with warnings.catch_warnings(record=True):
            self.classifier.fit(X, y)
        slicer.util.showStatusMessage('DWD: {}'.format(self.classifier), 2000)

        return True

    def meanShape(self, cases, factor=1.0):
        base = self.read_vtk(cases[0][0])
        X, _ = self.make_xy(cases)

        shape = base.Points.shape
        d, i = self.direction

        mean = np.mean(X, axis=0)
        mean -= d * (i - np.dot(mean, d))
        mean = mean.reshape(shape)

        mean_dir = d * factor
        mean_dir = mean_dir.reshape(shape)

        out = self.copy_dsa(base)
        out.Points = mean
        out.PointData.append(mean_dir, 'Direction')

        return out

    def compute(self, cases):
        d, i = self.direction
        X, y = self.make_xy(cases)

        actual = y
        predict = self.classifier.predict(X)
        distance = X.dot(d) - i

        return {
            'actual': actual,
            'predict': predict,
            'distance': distance
        }

    @property
    def direction(self):
        """Return the DWD direction and intercept. The separating hyperplane is of the
        form 'p.d = d.i', where '.' is the dot product. If 'p.d < d.i', then 'p' is label
        0. If 'p.d > d.i', then 'p' is label 1.
        """

        direction = self.classifier.coef_.reshape(-1)
        intercept = -float(self.classifier.intercept_)
        return direction, intercept

    def table(self, columns, name='Table'):
        tableNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTableNode', name)
        table = tableNode.GetTable()

        for name, data in columns.items():
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

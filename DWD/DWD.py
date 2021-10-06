import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

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
        self.parent.categories = ["Examples"]
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
        self._updatingGUIFromParameterNode = False

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

        self.ui.chk_autoTune.stateChanged.connect(self.autoTuneStateChanged)

        for _ in range(3):
            self.ui.tbl_stats.insertRow(0)
        for _ in range(2):
            self.ui.tbl_stats.insertColumn(0)

    def autoTuneStateChanged(self, enabled):
        self.ui.spn_tuningC.enabled = not enabled

    def cleanup(self):
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self):
        """Called each time the user opens this module."""
        # # Make sure parameter node exists and observed
        # self.initializeParameterNode()

    def exit(self):
        """Called each time the user opens a different module."""
        # # Do not react to parameter node changes (GUI wlil be updated when the user
        # # enters into the module)
        # self.removeObserver(
        #     self._parameterNode,
        #     vtk.vtkCommand.ModifiedEvent,
        #     self.updateGUIFromParameterNode,
        # )

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

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

    def onApplyButton(self):
        """Run processing when user clicks "Apply" button."""


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

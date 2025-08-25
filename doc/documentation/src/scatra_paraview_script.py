# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# trace generated using paraview version 5.11.2
# import paraview
# paraview.compatibility.major = 5
# paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
# paraview.simple._DisableFirstRenderCameraReset()

# For scalar transport:
resultVariable = "phi_1"
# For thermal simulation:
resultVariable = "temperature"

# get layout
layout1 = GetLayout()
# layout/tab size in pixels
# layout1.SetSize(1954, 1788)

# find source
# scatra1scatrapvd = FindSource('scatra1-scatra.pvd')
simulation_input = GetActiveSource()

# get active view
renderView = GetActiveViewOrCreate("RenderView")

# current camera placement for renderView1
renderView.InteractionMode = "2D"
renderView.CameraPosition = [-1, -10, 2]
renderView.CameraFocalPoint = [0, 0, 0.7]
renderView.CameraViewUp = [0, 0, 1]
# renderView.CameraParallelScale = 0.7071067811865474
# Render()

# view = GetActiveView()
# view.ViewSize = [2000,2000]  # funktioniert nur in bestimmten Kontexten

# create a new 'Warp By Scalar'
relief = WarpByScalar(registrationName="WarpByScalar1", Input=simulation_input)
# show data in view
reliefDisplay = Show(relief, renderView, "UnstructuredGridRepresentation")

# trace defaults for the display properties.
reliefDisplay.Representation = "Surface"

# hide data in view
Hide(simulation_input, renderView)

# Properties modified on warpByScalar1
relief.ScaleFactor = 5.0

# change representation type
reliefDisplay.SetRepresentationType("Surface With Edges")

Render()

# get animation scene
animationScene1 = GetAnimationScene()

animationScene1.GoToLast()

Render()

# set scalar coloring
ColorBy(reliefDisplay, ("POINTS", resultVariable))

# rescale color and/or opacity maps used to include current data range
reliefDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
reliefDisplay.SetScalarBarVisibility(renderView, True)

# update the view to ensure updated data information
renderView.Update()


# get color transfer function/color map for the resultVariable
phi_1LUT = GetColorTransferFunction(resultVariable)

# get color legend/bar for phi_1LUT in view renderView1
phi_1LUTColorBar = GetScalarBar(phi_1LUT, renderView)

# Properties modified on phi_1LUTColorBar
phi_1LUTColorBar.WindowLocation = "Any Location"
phi_1LUTColorBar.WindowLocation = "Any Location"
phi_1LUTColorBar.WindowLocation = "Any Location"
phi_1LUTColorBar.ScalarBarLength = 0.33
phi_1LUTColorBar.Position = [0.8, 0.5]
phi_1LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
phi_1LUTColorBar.TitleFontSize = 20
phi_1LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
phi_1LUTColorBar.LabelFontSize = 20
phi_1LUTColorBar.DrawBackground = 1
phi_1LUTColorBar.BackgroundColor = [1.0, 1.0, 1.0, 1.0]
phi_1LUTColorBar.BackgroundPadding = 20.0

Render()

SaveScreenshot(
    "scatra_resultsurface_" + resultVariable + ".png",
    renderView,
    TransparentBackground=True,
    CompressionLevel=3,
)

# ================================================

plotOverLine1 = PlotOverLine(registrationName="PlotOverLine1", Input=simulation_input)
plotOverLine1.Point1 = [-1.0, -1.0, 0.0]
plotOverLine1.Point2 = [1.0, 1.0, 0.0]

# show data in view
plotOverLine1Display = Show(plotOverLine1, renderView, "GeometryRepresentation")

# trace defaults for the display properties.
plotOverLine1Display.Representation = "Surface"
plotOverLine1Display.ColorArrayName = [None, ""]
plotOverLine1Display.SelectTCoordArray = "None"
plotOverLine1Display.SelectNormalArray = "None"
plotOverLine1Display.SelectTangentArray = "None"
plotOverLine1Display.OSPRayScaleArray = "Owner"
plotOverLine1Display.OSPRayScaleFunction = "PiecewiseFunction"
plotOverLine1Display.SelectOrientationVectors = "None"
plotOverLine1Display.ScaleFactor = 0.2
plotOverLine1Display.SelectScaleArray = "None"
plotOverLine1Display.GlyphType = "Arrow"
plotOverLine1Display.GlyphTableIndexArray = "None"
plotOverLine1Display.GaussianRadius = 0.01
plotOverLine1Display.SetScaleArray = ["POINTS", "Owner"]
plotOverLine1Display.ScaleTransferFunction = "PiecewiseFunction"
plotOverLine1Display.OpacityArray = ["POINTS", "Owner"]
plotOverLine1Display.OpacityTransferFunction = "PiecewiseFunction"
plotOverLine1Display.DataAxesGrid = "GridAxesRepresentation"
plotOverLine1Display.PolarAxes = "PolarAxesRepresentation"
plotOverLine1Display.SelectInputVectors = [None, ""]
plotOverLine1Display.WriteLog = ""

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
plotOverLine1Display.ScaleTransferFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    0.0,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
plotOverLine1Display.OpacityTransferFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    0.0,
    1.0,
    0.5,
    0.0,
]

# Create a new 'Line Chart View'
lineChartView1 = CreateView("XYChartView")

# show data in view
plotOverLine1Display_1 = Show(plotOverLine1, lineChartView1, "XYChartRepresentation")

# trace defaults for the display properties.
plotOverLine1Display_1.UseIndexForXAxis = 0
plotOverLine1Display_1.XArrayName = "Points_X"
plotOverLine1Display_1.SeriesVisibility = ["phi_1"]
plotOverLine1Display_1.SeriesLabel = [
    "Points_X",
    "position",
    "Owner",
    "Owner",
    "phi_1",
    "phi_1",
    "vtkValidPointMask",
    "vtkValidPointMask",
    "Points_X",
    "Points_X",
    "Points_Y",
    "Points_Y",
    "Points_Z",
    "Points_Z",
    "Points_Magnitude",
    "Points_Magnitude",
]
plotOverLine1Display_1.SeriesColor = [
    "arc_length",
    "0",
    "0",
    "0",
    "Owner",
    "0.8899977111467154",
    "0.10000762951094835",
    "0.1100022888532845",
    "phi_1",
    "0.220004577706569",
    "0.4899977111467155",
    "0.7199969481956207",
    "vtkValidPointMask",
    "0.30000762951094834",
    "0.6899977111467155",
    "0.2899977111467155",
    "Points_X",
    "0.6",
    "0.3100022888532845",
    "0.6399938963912413",
    "Points_Y",
    "1",
    "0.5000076295109483",
    "0",
    "Points_Z",
    "0.6500038147554742",
    "0.3400015259021897",
    "0.16000610360875867",
    "Points_Magnitude",
    "0",
    "0",
    "0",
]
plotOverLine1Display_1.SeriesOpacity = [
    "arc_length",
    "1.0",
    "Owner",
    "1.0",
    "phi_1",
    "1.0",
    "vtkValidPointMask",
    "1.0",
    "Points_X",
    "1.0",
    "Points_Y",
    "1.0",
    "Points_Z",
    "1.0",
    "Points_Magnitude",
    "1.0",
]
plotOverLine1Display_1.SeriesPlotCorner = [
    "arc_length",
    "0",
    "Owner",
    "0",
    "phi_1",
    "0",
    "vtkValidPointMask",
    "0",
    "Points_X",
    "0",
    "Points_Y",
    "0",
    "Points_Z",
    "0",
    "Points_Magnitude",
    "0",
]
plotOverLine1Display_1.SeriesLabelPrefix = ""
plotOverLine1Display_1.SeriesLineStyle = [
    "arc_length",
    "1",
    "Owner",
    "1",
    "phi_1",
    "1",
    "vtkValidPointMask",
    "1",
    "Points_X",
    "1",
    "Points_Y",
    "1",
    "Points_Z",
    "1",
    "Points_Magnitude",
    "1",
]
plotOverLine1Display_1.SeriesLineThickness = [
    "arc_length",
    "2",
    "Owner",
    "2",
    "phi_1",
    "2",
    "vtkValidPointMask",
    "2",
    "Points_X",
    "2",
    "Points_Y",
    "2",
    "Points_Z",
    "2",
    "Points_Magnitude",
    "2",
]
plotOverLine1Display_1.SeriesMarkerStyle = [
    "arc_length",
    "0",
    "Owner",
    "0",
    "phi_1",
    "0",
    "vtkValidPointMask",
    "0",
    "Points_X",
    "0",
    "Points_Y",
    "0",
    "Points_Z",
    "0",
    "Points_Magnitude",
    "0",
]
plotOverLine1Display_1.SeriesMarkerSize = [
    "arc_length",
    "4",
    "Owner",
    "4",
    "phi_1",
    "4",
    "vtkValidPointMask",
    "4",
    "Points_X",
    "4",
    "Points_Y",
    "4",
    "Points_Z",
    "4",
    "Points_Magnitude",
    "4",
]
lineChartView1.BottomAxisTitle = "distance"
lineChartView1.LeftAxisTitle = "concentration"
lineChartView1.ShowLegend = 0

# get layout
layout1 = GetLayoutByName("Layout #1")

# add view to a layout so it's visible in UI
AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)

# update the view to ensure updated data information
lineChartView1.Update()

# set scalar coloring
ColorBy(plotOverLine1Display, ("POINTS", "phi_1"))

# set active view
SetActiveView(lineChartView1)

# save data
SaveData(
    "/home/scheider/fourc/source/build/tests/scatra/scatra_2D_surfacesource_zero.csv",
    proxy=simulation_input,
    PointDataArrays=["phi_1"],
    CellDataArrays=["Owner"],
    FieldDataArrays=["TIME"],
)

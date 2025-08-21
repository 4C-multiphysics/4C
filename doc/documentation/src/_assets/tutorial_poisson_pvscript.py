# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# paraview.compatibility.major = 5
# paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *

# get layout
layout1 = GetLayout()

# find source
simulation_input = GetActiveSource()
resultVariable = simulation_input.PointArrays[0]

# get active view
renderView = GetActiveViewOrCreate("RenderView")

# current camera placement for renderView1
renderView.InteractionMode = "2D"
renderView.CameraPosition = [-1, -10, 2]
renderView.CameraFocalPoint = [0, 0, 0.7]
renderView.CameraViewUp = [0, 0, 1]
# Render()

# create a new 'Warp By Scalar'
relief = WarpByScalar(registrationName="WarpByScalar1", Input=simulation_input)
# show data in view
reliefDisplay = Show(relief, renderView, "UnstructuredGridRepresentation")

# reliefDisplay.Representation = 'Surface'

# hide data in view
Hide(simulation_input, renderView)

# Properties modified on the relief:
relief.ScaleFactor = 5.0
reliefDisplay.SetRepresentationType("Surface With Edges")

# draw the last step
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
LUT = GetColorTransferFunction(resultVariable)

# get color legend/bar for phi_1LUT in renderView
LUTColorBar = GetScalarBar(LUT, renderView)

# Properties modified on phi_1LUTColorBar
LUTColorBar.WindowLocation = "Any Location"
LUTColorBar.ScalarBarLength = 0.33
LUTColorBar.Position = [0.8, 0.5]
LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
LUTColorBar.TitleFontSize = 20
LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
LUTColorBar.LabelFontSize = 20
LUTColorBar.DrawBackground = 1
LUTColorBar.BackgroundColor = [1.0, 1.0, 1.0, 1.0]
LUTColorBar.BackgroundPadding = 20.0

Render()

SaveScreenshot(
    "scatra_resultsurface_" + resultVariable + ".png",
    renderView,
    TransparentBackground=True,
)

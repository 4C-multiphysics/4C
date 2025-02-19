.. _3dsolidtutorial:

3D Solid Tutorial with Coreform Cubit\ |reg|\  and *pre_exodus*
=================================================================

Introduction
----------------

This tutorial gives an introduction to the usage of |FOURC| for simulating plastic material behaviour.
It is assumed that |FOURC| including the pre-processing tool `pre_exodus` has been built on your machine according to the instructions and has passed the tests without error messages.
In addition, the pre-processing of the finite element model is built with Coreform Cubit, but the input file can also be generated without this software.

The analyzed structure is a simple round tensile bar,
which is commonly used to identify the material's stress-strain curve by comparing the simulation with the respective experiment.
The geometry, which has a cross sectional diameter of 10 mm and a length of constant diameter of 50 mm is shown in the following figure:

.. figure:: figures/tut_solid-geometry.jpg
   :alt: Problem definition and geometrical setup
   :width: 500px
   :align: center

   Problem definition and geometrical setup


The material under consideration is an Aluminum alloy with a yield strength of 330 MPa and a mild isotropic hardening.
During the test, a force *F* acts on the top of the rod such that it elongates.

For elasto-plastic simulations |FOURC| provides mainly the following seven material models:

::

    MAT_Struct_PlasticLinElast
    MAT_Struct_PlasticNlnLogNeoHooke
    MAT_Struct_ThermoPlasticHyperElast
    MAT_Struct_ThermoPlasticLinElast
    MAT_Struct_Damage
    MAT_Struct_PlasticGTN
    MAT_Struct_DruckerPrager

The following flow chart depicts for which cases these seven models are applicable:

.. figure:: figures/tut_solid_flow_chart.jpg
   :alt: plasticity models flow chart
   :width: 800px
   :align: center

   Capabilities of plasticity models available in 4C


Preprocessing
----------------

Due to the symmetry of this model, we may restrict our analysis to one half in longitudinal direction.
In addition, one can see that the structure has a rotational symmetry.
Since axisymmetric elements are not available in |FOURC|, a 3D structure is modelled with double symmetry in both lateral directions, so that only one eighth of the structure is left.
The final FE model is shown here.


.. figure:: figures/tut_solid-mesh.jpg
   :alt: tutorial_solid mesh
   :width: 500px
   :align: center

   Mesh of one eighth of the tensile bar


Symmetry boundary conditions are applied to the three symmetry planes, denoted with ``XSYMM, YSYMM, ZSYMM`` in the above figure.
The top surface is subject to a Dirichlet boundary condition with linearly increasing displacement in y-direction.

To this end, we model 200 time steps spanning from :math:`t_0 = 0` s to :math:`t_1 = 1` s with a time step of :math:`\Delta t = 0.005` s.
Assuming nonlinear kinematics (large strain theory) and plasticity with isotropic hardening following an exponential curve,
namely :math:`\sigma_Y = \sigma_{Y,0} + (\sigma_{Y,\infty} - \sigma_{Y_0}) \left[ 1 - \exp \left( - k \, \varepsilon_p \right) \right]` with :math:`\sigma_{Y,0}=330, \sigma_{Y,\infty} = 1000, k=5` .

We choose the model ``MAT_Struct_PlasticNlnLogNeoHooke`` to accomplish this task, which is a plasticity model that uses von Mises yield criterion

.. math::

    \Phi = \tilde{\mathbf{\tau}} - \sqrt{\frac{2}{3}} \sigma_Y

with :math:`\tilde{\mathbf{\tau}}` being the deviatoric Kirchhoff stress.
Additionally, a compressible Neo-Hooke elasticity model expressed by the free energy potential

.. math::

    \rho_0 \Psi(\mathbf{B}^e) =  \frac{K}{2} \left[ \frac{1}{2} (J^2 - 1) - \ln J \right] + \frac{1}{2} \mu \left[ \text{tr} \mathbf{\tilde{B}}^e - 3 \right]

Here, :math:`J` is the determinant of the deformation gradient :math:`\mathbf{F}`,
:math:`\mathbf{B}^e = \mathbf{F}^e {\mathbf{F}^e}^T` is the elastic part of the left Cauchy Green tensor, and :math:`K, \mu` are elastic constants.

The geometry, finite element mesh and node sets for the boundary conditions are created using Coreform Cubit\ |reg|\ .
The mesh is rather fine and 27-node brick elements have been used herein.
The corresponding journal file can be run to reproduce this mesh and output a binary exodus mesh information file ``tutorial_solid.e``.
The geometry dimensions (actually mainly the radius, all other dimensions depend on it) and element size can be varied, since they are parametrized in the journal file.

.. literalinclude::  tutorial_solid.jou
   :linenos:

The general definition of the geometry in this cubit journal file, which can be run from the directory ``<source-root>/tests/framework-test/`` includes the complete mesh and four surface node sets.
If you don't have Coreform Cubit, you may use the resulting exodus file directly, which is also located in this directory.

The boundary conditions and loading, the material and all solver details are defined in two other files:
A so-called boundary condition file ``tutorial_solid.bc`` and a header file ``tutorial_solid.head``.
These three files together can be combined to a |FOURC| input file using the command

::

    <build-dir>/pre-exodus --exo=tutorial_solid.e --bc=tutorial_solid.bc --head=tutorial_solid.head

Note that you need to build the ``pre_exodus`` command before.
You'll find the build information for ``pre_exodus`` in the :ref:`setup section <custom_target_specifiers>` or in the :ref:`workflow/preprocessing section <preprocessing>`.

The boundary condition file ``tutorial_solid.bc`` contains the element information and boundary conditions.
The element type is a quadratic (27 node) 3D solid element(Solid HEX27) with nonlinear kinematics, that is, including a large strain formulation.
The boundary conditions include three symmetry conditions (based on the surfaces defined in Cubit) and one loading condition.
Note that only surface conditions are given; the lines that belong to two surfaces automatically get the combined boundary conditions.
The loading condition has a function number (1) included for the y-direction, and also gets a tag: ``TAG monitor_reaction``.
The tag is for an extra output of the total forces, which are printed in a csv file, if the output is requested.

.. literalinclude:: tutorial_solid.bc
   :linenos:




All control parameters for the material, solvers, loading function, output information are contained in ``tutorial_solid.head``.
Particularly, the output of the forces is requested by a particular section, ``IO/MONITOR STRUCTURE DBC``, which controls accuracy and frequency of the output.
This and all other output information is given in section ``IO`` and its subsections.

.. literalinclude::  tutorial_solid.head
   :lines: 17-29
   :lineno-start: 17

You'll find two solver sections in there. In the original header file, which are selected in the section ``STRUCTURAL DYNAMIC`` by the parameter ``LINEAR_SOLVER``.
The value of this parameter refers to the section ``SOLVER x``, in which ``x`` is the number of the selected solver.
In the section ``SOLVER 1``, the direct solver is defined, it does not have any parameters (except an optional name).
The iterative solver in section ``SOLVER 2`` has some parameters, namely an xml file, which contains more parameters.
The given xml file as well as some other examples for different multiphysics problems are given in the folder ``<source-root>/tests/input-files/xml/*/*.xml``.

.. literalinclude::  tutorial_solid.head
   :lines: 30-63
   :lineno-start: 30

If you choose ``LINEAR_SOLVER  2``, you'll get the iterative solver.
Note that you have to copy the xml file to your folder before executing ``pre_exodus`` with the iterative solver setting.

The ``STRUCTURAL DYNAMIC`` section also defines the time integration, particularly the time stepping.
Time step size is given by ``TIMESTEP``, the maximum time is given by ``MAXTIME`` and the maximum number of increments is ``NUMSTEP``.
The end of the simulation is defined either when the maximum time or when the number of increments is reached, whatever comes earlier.
Note that the number of time steps is currently set to 10, in order to keep the simulation time short for the CI/CD tests.
If you want to run the whole simulation including necking of the specimen, you have to change the parameter to at least ``NUMSTEP 200``.

Besides output and solver details, the material and any functions are defined in the header.
For the material, we choose a large strain plasticity model, which is named ``MAT_Struct_PlasticNlnLogNeoHooke`` with parameters, which might represent a high strength Aluminum.
Last but not least, a function is defined in this file, to which the Dirichlet boundary condition refers.

.. literalinclude::  tutorial_solid.head
   :lines: 64-68
   :lineno-start: 64



Simulation
-----------

After the file ``tutorial_solid.dat`` has been created successfully using the ``pre_exodus`` command shown above, the simulation is run on a single processor by the command

::

    <build-dir>/4C tutorial_solid.dat tutorial_solid

Since the solver output is quite lengthy, one might want to redirect it into a file.
For running the simulation with several processes, use ``mpirun -np <n_cpu> <build-dir>/4C [further parameters]``.

Post processing
----------------

The VTK results are collected in the directory ``tutorial_solid-vtk-files/``.
If they shall be opened by Paraview, the data file ``tutorial_solid-structure.pvd`` contains the complete collection of files, so only this file is to be opened, all other files are then read automatically.
Using the filter ``Warp by vector`` with Coloring defined by the scalar ``accumulated_plastic_strain``, one may obtain a contour plot on the deformed geometry:


.. figure:: figures/tut_solid-deformed.jpg
   :alt: Paraview post processing output
   :width: 600px
   :align: center

   Deformed mesh with contours of accumulated plastic strain


The output of the forces needed to apply the Dirichlet boundary condition are given in the directory ``tutorial_solid_monitor_dbc``.
Each boundary condition using the tag ``monitor_reaction`` creates a single ASCII file with suffix ``.csv``,
which for the present case provides the time, the force in y-direction and the initial and current area to which the force is applied.
Since the displacement itself is not given in this file, the displacement is to be calculated from the time.

A force-displacement graph may be created using gnuplot by the lines

::

    set terminal png size 1200,900 font Arial 14
    set output 'tutorial_solid_graph.png'
    set key off
    set xlabel "displacement"; set ylabel "force"; set title "Force-displacement curve"
    plot "tutorial_solid_monitor_dbc/tutorial_solid_10004_monitor_dbc.csv" using ($2*12):(abs($6))  with lines title "force-displacement"


.. figure:: figures/tutorial_solid_graph.png
   :alt: result curves for solid tutorial
   :width: 700px
   :align: center

   Force displacement curve for the tensile bar

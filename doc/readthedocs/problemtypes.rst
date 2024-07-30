.. _problemtypes:

Problem types
==============

.. ToDo:

   Here, we have to describe the different types that can be calculated with |FOURC|. It's not complete yet.

.. _singlefieldproblems:

Single field problems:
----------------------

Ale
~~~~

**General**

Ale stands for *arbitrary Lagrange-Euler*.
Thus, the mesh may have a particular movement, which is defined by the ``ALE_TYPE``.

**Elements and Degrees of freedom**

The element types used are given in :ref:`ALE ELEMENTS<aleelements>`.
The number of degrees of freedom depends on the dimensionality of the structure (2 for 2D structures, 3 for 3D)


ArterialNetwork
~~~~~~~~~~~~~~~

**General**

This problemtype considers the blood flow in networks of arteries, which are modelled as 1D line elements in three-dimensional space.


**Elements and Degrees of freedom**

The element types used are given in :ref:`ARTERY ELEMENTS<arteryelements>` (only 1D line elements are available).
The number of degrees of freedom


Cardiac_Monodomain
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

.. ToDo::

   No information about this special problem type.

This problemtype considers .

**Elements and Degrees of freedom**

The element types used are given in :ref:`TRANSPORT ELEMENTS<transportelements>`.


Fluid
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

This problemtype considers the flow of fluids through a domain.

**Elements and Degrees of freedom**

The element types used are given in :ref:`FLUID ELEMENTS<fluidelements>`.
The number of degrees of freedom depends on the dimension of the problem;
One may have 2 or 3 degrees for the velocity plus one degree for the fluid pressure.


Electrochemistry
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

This problemtype is kind of a scalar transport simulation, which considers xxx as the primary variable,
and xxx is the flow.

**Elements and Degrees of freedom**

The element types used are given in :ref:`TRANSPORT ELEMENTS<transportelements>`, and the domain may be 1D, 2D, or 3D.



Electromagnetics
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

.. ToDo::

   No information about this special problem type.

**Elements and Degrees of freedom**

The element types used are given in :ref:`ELECTROMAGNETIC ELEMENTS<electromagneticelements>`. The number of degrees of freedom


Fluid_Top_Opt
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

.. ToDo::

   No information about this special problem type.

**Elements and Degrees of freedom**

The element types used are given in xxx. The number of degrees of freedom


Fluid_XFEM
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

.. ToDo::

   No information about this special problem type.

**Elements and Degrees of freedom**

The element types used are given in xxx. The number of degrees of freedom


Inverse_Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

.. ToDo::

   No information about this special problem type.

**Elements and Degrees of freedom**

The element types used are given in :ref:`STRUCTURE ELEMENTS<structureelements>`.
The number of degrees of freedom


Level_Set
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

.. ToDo::

   No information about this special problem type.


**Elements and Degrees of freedom**

The element types used are given in xxx. The number of degrees of freedom


Particle
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

This is a completely different problem type, which comes along without elements.
Instead, the method is based on SPH, which means *smoothed particle hydrodynamics*,
and the particles are defined by their coordinates.

**Particles and Degrees of freedom**

The particles are always given as positions in 3D space (without particle numbers), not as elements;
the number of degrees of freedom depends on the value of ``KERNEL_SPACE_DIM``, which may have the value ``KERNEL1D, KERNEL2D, KERNEL3D``.


Polymer_Network
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

The polymer network is, similarly to the Arterial Network, a network of 1D constituents in 3D space, namely here a network of polymer fibers.

**Elements and Degrees of freedom**

The element types are solely BEAM3R elements used, which can be found in :ref:`STRUCTURE ELEMENTS<structureelements>`.



ReducedDimensionalAirWays
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

Similar to other network models, the *Reduced dimensional Airways* simulation considers a network of 1D elements in 3D space,
namely here 1D pipe models of human airways.

**Elements and Degrees of freedom**

The element types used here are 1D elements given in :ref:`REDUCED D AIRWAYS ELEMENTS<reduced d airwayselements>`.


Scalar_Transport
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

This problemtype considers the transport of a scalar variable through a domain.
Such a scalar could be the temperature (which is usually treated by PROBLEMTYP Thermo),
but also any other field quantity.

**Elements and Degrees of freedom**

Since the problemtype is scalar transport,
the value of this scalar is the only active degree of freedom for this type.

The element types used are given in :ref:`TRANSPORT ELEMENTS<transportelements>`.

**Results**

The main output consists of the scalar quantity itself and its flux.

Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

The problemtype *Structure* considers deformations and stresses in solid mechanics.

**Elements and Degrees of freedom**

The element types used are given in :ref:`STRUCTURE ELEMENTS<structureelements>`. The number of degrees of freedom depends on the element type.
In general, all elements have displacements in spatial directions (2 or 3, depending on the dimensionality).
In the case of C1-steady elements like beams and shells, the rotations (again 2 or 3) are added to the degrees of freedom,
so there are up to 6 DoFs.

**Results**

The result variables are stresses and strains within the elements,
and reaction forces at the nodes, where a Dirichlet boundary condition has been applied.
Other internal variables may be calculated as necessary (and desired).

Thermo
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

The problemtype *Thermo* considers heat transfer in arbitrary structures.

**Elements and Degrees of freedom**

The element types used are given in :ref:`THERMO ELEMENTS<thermoelements>`. There is only one degree of freedom, that is the temperature.

**Results**

Internally: Heat flux per area
At Dirichlet boundary conditions: heat flux



.. _multifieldproblems:

Multi field problems:
----------------------

These problems combine a number of single field problems, and are therefore sometimes called *Coupled Problems*.

Biofilm_Fluid_Structure_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Elastohydrodynamic_Lubrication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics: STRUCTURAL | LUBRICATION | ELASTO HYDRO

Fluid_Ale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:  FLUID | ALE | FSI

Fluid_Beam_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics: FSI | FLUID | STRUCTURAL

Fluid_Freesurface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics: FLUID | FSI | ALE


Fluid_Poro_Structure_Interaction_XFEM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics: STRUCTURAL | POROELASTICITY | FSI | FLUID

Fluid_Porous_Structure_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:



Fluid_Porous_Structure_Scalar_Scalar_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Fluid_RedModels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Fluid_Structure_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Fluid_Structure_Interaction_Lung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Fluid_Structure_Interaction_RedModels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Fluid_Structure_Interaction_XFEM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Fluid_XFEM_LevelSet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Gas_Fluid_Structure_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Immersed_FSI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Low_Mach_Number_Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Lubrication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Multiphase_Poroelasticity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Multiphase_Poroelasticity_ScaTra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Multiphase_Porous_Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

NP_Supporting_Procs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Particle_Structure_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Poroelastic_scalar_transport
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Poroelasticity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

RedAirways_Tissue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Scalar_Thermo_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Structure_Ale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Structure_Scalar_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Structure_Scalar_Thermo_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Thermo_Fluid_Structure_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Thermo_Structure_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:


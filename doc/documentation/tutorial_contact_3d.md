# 3D Contact Tutorial

This tutorial gives a short introduction to the setup of a simple 3D contact problem.
The goal of this tutorial is to give an overview of the general workflow in |FOURC| and to show how to create a
working `.dat` input file.
It is neither intended to be an introduction to the theory of contact mechanics, nor demonstrate all possible optional
settings for a given (contact) problem.
It is assumed that |FOURC| has been built on your machine according to the instructions and has passed the tests without
error messages.
For further information on how to build |FOURC| and to test the build, please refer to
the [README.md](https://github.com/4C-multiphysics/4C/blob/main/README.md).

## Overview

1. [Problem description](#problem-description)
1. [Create files](#create-files)
    1. [Define Geometry](#define-geometry)
    1. [Define Boundary Conditions](#define-boundary-conditions)
    1. [Specify Simulation Settings](#specify-simulation-settings)
1. [Create .dat input file](#create-input-file)
1. [Run Simulation in |FOURC|, Post-processing and visualization in Paraview](#run-simulation)

## Problem description

In this tutorial we create two cubes sizes 1x1x1 and 0.5x0.5x0.5,
with a time dependent Dirichlet boundary condition (DBC) imposed on the right outer surface of the smaller cube.
As a result, the smaller cube is pressed horizontally against the larger counterpart, which is clamped on the left outer
surface.
Mortar methodology is applied for the discretization of the contact constraints and the subdomains are coupled with dual
Lagrange multipliers.
The cubes are assumed to consist of two different materials,
represented by a St.Venant-Kirchhoff material model with different values of the Young's modulus and density for each
cube.

```{eval-rst}
.. figure:: figures/tutorial_contact3d.jpg
   :alt: Problem definition and geometrical setup
   :width: 500px
   :align: center

   Geometry of the two bodies in contact.
```

## Create files

To create a `.dat` input file for |FOURC| we need three files:

- `.exo`/`.e` file containing the geometry and mesh
- `.bc` file containing information on the boundary conditions
- `.head` file containing all relevant parameter settings for the simulation

### Define Geometry

Information on the geometry and mesh are passed on to |FOURC| as part of a binary EXODUS file (`.e`).
This file can be generated using the pre-processing software Cubit.
Before we can export an `.exo` file from Cubit (File -> Export -> Files of type: `.e`), we need to specify the geometry
and meshing parameters in Cubit.
This can be done using the GUI or read from a journal file (`.jou`) containing the specific Cubit commands.
A tutorial on how to use the Cubit GUI can be found on
the [coreform webpage](https://coreform.com/products/coreform-cubit/tutorials/) and can be useful to get to know some of
the basic functionalities.
The syntax of a Cubit command to a corresponding Input using the GUI can be seen in the Cubit terminal.

> Remark: After learning the syntax of the basic Cubit commands,
> it may be more convenient to exclusively use journal files for an easy adaptation and reproduction of your geometry
> and/or use in e.g. python scripts for parameter studies etc.

In `tutorial_contact_3d.jou` it can be seen that, apart from the basic geometry definition and meshing,
nodesets and element blocks are assigned to the corresponding nodes/lines/surfaces/volumes.
The nodesets are used later on to assign the boundary conditions, whereas the element blocks are used to assign material
properties.
Additionally, the last line in the given journal file includes a terminal command to export the `.exo`/`.e` file from
cubit.

The journal file can be called in Cubit from the Terminal with the command

```bash
/path/to/cubit/executable -nographics /path/to/tutorial_contact_3d.jou
```

This saves the `.exo` file in the current work directory.

### Define Boundary Conditions

For the definition of the boundary conditions, it is convenient to use the prototype of the `.bc` file,
as it provides syntax examples and a basic structure that automatically inserts the information on number of nodes in
each block element/nodeset/etc.
It also gives a list of all possible conditions that can be defined, as well as the element type names and the
corresponding shapes.
Depending on the type of the problem, undefined/empty/unnecessary lines can be deleted when using it as a template.
Information on how to obtain prototypes for both the `.bc` file and the `.head` file is given in
the [README.md](https://github.com/4C-multiphysics/4C/blob/main/README.md) under the heading "Prepare and Run Simulations".
The terminal command to obtain both prototype files is:

```bash
<buildDir>/pre_exodus --exo=tutorial_contact_3d.e
```

In `tutorial_contact_3d.bc` it can be seen, that the first two blocks of text define the structure elements, which will
later fill the system matrices.
The other text blocks define conditions on parts of the system.
The first line of each block assigns an (optional) name to the element block/nodeset and the shape of the element type
is specified.
The next part has already been filled in by `pre_exodus`.
For our simple geometry and discretization it can easily be checked that the number of elements/nodes in the
corresponding element blocks and nodesets is correct.
Next, the Section of the `.dat` file under which this information will be listed is given.
The structure element properties are listed under `STRUCTURE`, whereas the boundary/contact conditions will be listed
under their corresponding section names.
The parameters for assigning a material model to an element block with `SOLID` elements of shape `HEX8` are:

```bash
description="MAT 1 KINEM nonlinear"
```

where we choose material model 1, set the kinematics to nonlinear and assume no enhanced assumed strain.
Last, the element type/condition specific set of parameters is given and the name of the element type is specified.
Hint: A list of possible options can be found in the `.bc` default file.

Remark: Apart from the separate listing of problem specific conditions, Dirichlet and Neumann BCs (as well as many other
types of conditions) are also listed separately,
depending on the geometric entity that they are imposed on, e.g. points, lines, surfaces or volumes.

The parameters for the definition of a DBC are:

```description="NUMDOF 3 ONOFF 1 0 0 VAL -1.0 0.0 0.0 FUNCT 1 0 0"```

where: NUMDOF = number of degrees of freedom per node, ONOFF = on which of the DOF is the condition imposed (Binary
1/0),
VAL = scaling parameter, FUNCT = Assign a numbered function to a dof (here Function1, defined in .head file).

The master and slave surfaces for the definition of the contact mortar problem can be specified as:

```description="1 Master"```

> Remark: It is common practice to choose the side with the finer discretization as the slave side.

### Specify Simulation Settings

The prototype of the `.head` file contains an extensive list of possible parameter settings for running a simulation in
|FOURC|.
It also gives information on the numerous abbreviations used in the input files.
Hint: Use the search function of your editor to navigate the 13000 lines of text.

As we are setting up a structural contact problem, not all sections and parameters are required (e.g. sections
concerning the simulation of fluids, etc. can be left out).
If they are not specified explicitly in our input file, the parameters take on default values.
Some sections that are required by all problem types include `PROBLEM SIZE`, `PROBLEM TYPE`, `DISCRETIZATION`, `IO` (
Input/Output),
at least one `SOLVER` section (for contact problems we need to specify two solvers:
one for the state, where not contact is active, and one for the state with active contact) and a `MATERIALS` section.
The time integration method `DYNAMICTYPE` (Generalized alpha method), time step size `TIMESTEP` and final time `MAXTIME`
are also specified for structural dynamics.
An important parameter is `RESULTSEVERY`, which specifies how often output is written and thus directly controls the
size of the output file.

The contact specific parameters are given under `--MORTAR COUPLING`. Dual Lagrange multipliers are chosen for the
coupling of the interface.
Either the `BruteForceEleBased` algorithm, or the more efficient `BinaryTree` can be chosen as the contact search
algorithm.
In `tutorial_contact_3d.head` you will also find the definition of the `FUNCTION` that is used for the time dependent
DBC.

## Create input file

The final `.dat` input file for |FOURC| can be created with the `pre_exodus` executable
as described in the [README.md](https://github.com/4C-multiphysics/4C/blob/main/README.md).
To manually create the `.dat` file the full command states:

```bash
<buildDir>/pre_exodus --exo=xxx_tutorial_contact_3d.e --head=tutorial_contact_3d.head --bc=tutorial_contact_3d.bc --dat=tutorial_contact_3d.dat
```

This saves the `.dat` file in the current work directory. The `.dat` file now contains all the information from the
created files,
plus a list of all nodes and structure elements.

> Remark: The framework testing with ctest executes the command above automatically.
> It also requires the prefix `xxx_*.e` so that the file is found during the framework test.

## Run Simulation

Again, following the instructions from the [README.md](https://github.com/4C-multiphysics/4C/blob/main/README.md), the
|FOURC|
executable can be invoked with the `tutorial_contact_3d.dat` input file.
Since we chose the binary output option in our `.head` file (section `IO`),
we also have to run a post processing tool before visualizing the results in Paraview.

Gratitude is given to Christopher Steimer, who created this fascinating framework test.

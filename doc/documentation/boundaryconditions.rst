.. _boundaryconditions:

Boundary Conditions
===================

Various types of boundary conditions can be defined on points (nodes),
lines, surfaces and volumes. These conditions are defined in the general
section format

::

       --DESIGN [POINT|LINE|SURF|VOL] <condition type> CONDITIONS

with subsequent lines defining the region and the specific values of the
boundary conditions. Note that some boundary condition types do not follow this general structure.
For each particular condition, refer to the :ref:`boundary condition reference <prescribedconditionreference>`.

A point conditions must be followed by ``DPOINT <num>``, a line condition by ``DLINE <num>``,
a surface by ``DSURF <num>`` and a volume by ``DVOL <num>``.
The number to be entered in ``num`` is the number of conditions to follow.

Each subsequent line starts with an ``E <set> -``, where ``set`` is the number of the TOPOLOGY set defined in terms of points, lines, surfaces and volumes, respectively.

Note that all boundary conditions are given in terms of node sets.
Boundary conditions may be applied in arbitrary coordinate directions,
which do not need to coincide with the original coordinate system of the geometry.
For this aspect one may define a local (rotated) coordinate system.

.. `locsysconditions`:

Local Coordinate System
----------------------------

Local coordinates may be defined on points, lines, surfaces and volumes.
The coordinate system is given by an axis :math:`\mathbf{n}` and an angle :math:`\alpha` (in rad)
around which the coordinate is rotated **clockwise**.

Since the axis is a unit vector, the angle is given as the length of the vector,
so that the complete rotation can be entered in three values:
:math:`[\alpha \cdot n_x, \, \alpha \cdot n_y, \, \alpha \cdot n_z]`.

The complete definition of a local coordinate system writes:

::

   ------------------------DESIGN [POINT/VOL] LOCSYS CONDITIONS
   DPOINT/DVOL    numtotal   //  numtotal is the number of subsequent lines
   E num - ROTANGLE 0.0 0.0 0.0 FUNCT 0 0 0 USEUPDATEDNODEPOS 0
   // num corresponds to the set_descriptor_id defined in DNODE/DVOL-NODE TOPOLOGY block

for point and volume definitions, and

::

   ------------------------DESIGN [LINE|SURF] LOCSYS CONDITIONS
   DPOINT|DLINE|DSURF    numtotal   //  numtotal is the number of subsequent lines
   E num - ROTANGLE 0.0 0.0 0.0 FUNCT 0 0 0 USEUPDATEDNODEPOS 0  USECONSISTENTNODENORMAL 0
   // num corresponds to the set_descriptor_id defined in DLINE/DSURF-NODE TOPOLOGY block

for lines and surfaces.

The rotation may depend on time and/or space, that is, it can be combined with a function definition,
see the proper definition in the :ref:`functions <functiondefinitions>` section.

In addition, it is possible to calculate a spatial dependence either on the original node coordinates
or on the updated (displaced) node coordinate,
which may be important in a large displacement analysis. This is done by the ``USEUPDATEDNODEPOS`` parameter (=0: original coordinates, =1: updated coordinates).

.. todo::

   The parameter ``USECONSISTENTNODENORMAL`` can (at this time) only be used for ALE and fluid simulation.
   However, there is no test input using this parameter anyway.


.. _`dirichletboundaryconditions`:

Dirichlet Boundary Conditions
-----------------------------

Dirichlet boundary conditions (BC) are defined by specifying the number
of DOFs of the nodes in the respective node set (NUMDOF). A binary
switch that indicates which of these DOFs should be constrained (ONOFF=
0 or 1, with 0=unconstrained and 1=constrained). A list of entries to which
value the respective DOFs are constrained (VAL). If applicable, the
specifier (FUNCT) giving the ID of a function that defines a dependence
of the constraint DOFs on other simulation parameters like, e.g., the
time (see below) and, if applicable, a (TAG) entry. It is noted that,
the number of entries following ONOFF, VAL and FUNCT must be the same as
NUMDOF value. The geometry entity that the Dirichlet boundary condition
applies is specified by ``num`` value. Depending on which block
description of the boundary condition, the type of corresponding
geometry will be selected appropriately, see the comments of the table
below for more information. The value of ``num`` can only be from 1 to
the number of the design descriptor.

In the case that two or more boundary conditions are intersected, the
intersection geometry must be constrained with the constraint
information of both BCs. This feature shall be handled properly by the
pre-processor.

Of course, the applied Dirichlet boundary condition may depend on time and on the position
of the node. This is achieved by a function definition, after the keyword ``FUNCT``.
The number of the function (for each component) can be specified in
order to define a spatial or temporal dependence. The proper definition
of functions is given in the :ref:`functions <functiondefinitions>` section.


The ``TAG`` option allows to monitor the reaction forces at the constraint
nodes by setting it to *monitor_reaction*. With this (TAG) set, the
reaction forces are written to .csv files in a newly created sub
directory of the simulation directory. Note that even even the TAG
parameter can be given for any dirichlet boundary condition, it only
produces results for forces and moments in a structural analysis.

One needs also to set the corresponding IO, such as

::

   -------------------------------------IO/MONITOR STRUCTURE DBC
   INTERVAL_STEPS                   1
   PRECISION_FILE                   16
   PRECISION_SCREEN                 5
   FILE_TYPE                        csv
   WRITE_HEADER                     no

, see :ref:`SECio_monitorstructuredbc`,
and set the right time integration strategy, :ref:`INT_STRATEGY<SECstructuraldynamic>`,
by which the standard one, i.e. Generalized Newmark Alpha, should always work.

::

   --------------------------------------------STRUCTURAL DYNAMIC
   INT_STRATEGY                     Standard

Note that the TAG parameter may only be set for linear elements, not for
quadratic ones (HEX20, TET10)

Below is the valid block definition for various types of Dirichlet
boundary conditions.

::

   ------------------------DESIGN [POINT|LINE|SURF|VOL] DIRICH CONDITIONS
   DPOINT|DLINE|DSURF|DVOL    numtotal   //  numtotal is the number of subsequent lines
   E num - NUMDOF 0  ONOFF 0  VAL 0.0  FUNCT none   TAG none
   // num corresponds to the point_descriptor_id defined in DNODE/DLINE/DSURF/DVOL-NODE TOPOLOGY block
   --------------------DESIGN [POINT|LINE|SURF|VOL] ALE DIRICH CONDITIONS
   DPOINT|DLINE|DSURF|DVOL    numtotal   //  numtotal is the number of subsequent lines
   E num - NUMDOF 0  ONOFF 0  VAL 0.0  FUNCT none   TAG none
   --------------DESIGN [POINT|LINE|SURF|VOL] TRANSPORT DIRICH CONDITIONS
   DPOINT|DLINE|DSURF|DVOL    numtotal   //  numtotal is the number of subsequent lines
   E num - NUMDOF 0  ONOFF 0  VAL 0.0  FUNCT none   TAG none
   -----------------DESIGN [POINT|LINE|SURF|VOL] THERMO DIRICH CONDITIONS
   DPOINT|DLINE|DSURF|DVOL    numtotal   //  numtotal is the number of subsequent lines
   E num - NUMDOF 0  ONOFF 0  VAL 0.0  FUNCT none   TAG none
   -------------------DESIGN [POINT|LINE|SURF|VOL] PORO DIRICH CONDITIONS
   DPOINT|DLINE|DSURF|DVOL    numtotal   //  numtotal is the number of subsequent lines
   E num - NUMDOF 0  ONOFF 0  VAL 0.0  FUNCT none   TAG none
   ---------------DESIGN [POINT|LINE|SURF|VOL] NURBS LS DIRICH CONDITIONS
   DPOINT|DLINE|DSURF|DVOL    numtotal   //  numtotal is the number of subsequent lines
   E num - NUMDOF 0  ONOFF 0  VAL 0.0  FUNCT none   TAG none

See the respective reference sections for
:ref:`mechanical <designpointdirichconditions>`, :ref:`ALE <designpointaledirichconditions>`,
:ref:`TRANSPORT <designpointtransportdirichconditions>`, :ref:`THERMO <designpointthermodirichconditions>`,
:ref:`PORO <designpointporodirichconditions>`, :ref:`NURBS LS <designpointnurbslsdirichconditions>`.

Neumann Boundary Conditions
---------------------------

Neumann boundary conditions are flux conditions. This means that in
contrast to the Dirichlet boundary conditions, they have to be provided
in terms of flux per applied geometry. A POINT NEUMANN condition is, for
example, a concentrated force or heat flux, while a SURF NEUMANN is a
pressure or surface heat flux, accordingly.

::

   ----------------------------DESIGN [POINT|LINE|SURF|VOL] NEUMANN CONDITIONS
   DPOINT|DLINE|DSURF|DVOL     numtotal   //  numtotal is the number of subsequent lines
   //E num - NUMDOF 0  ONOFF 0  VAL 0.0  FUNCT none   TYPE Live
   -------------------------------------------DESIGN POINT MOMENT EB CONDITIONS
   DPOINT|DLINE|DSURF|DVOL     numtotal
   //E num - NUMDOF 0  ONOFF 0  VAL 0.0  FUNCT none   TYPE Live
   -----------------------DESIGN [POINT|LINE|SURF] TRANSPORT NEUMANN CONDITIONS
   DPOINT|DLINE|DSURF          numtotal
   //E num - NUMDOF 0  ONOFF 0  VAL 0.0  FUNCT none   TYPE Live
   ----------------------DESIGN [POINT|LINE|SURF|VOL] THERMO NEUMANN CONDITIONS
   DPOINT|DLINE|DSURF|DVOL     numtotal
   //E num - NUMDOF 0  ONOFF 0  VAL 0.0  FUNCT none   TYPE Live
   ------------------------DESIGN [POINT|LINE|SURF|VOL] PORO NEUMANN CONDITIONS
   DPOINT|DLINE|DSURF|DVOL     numtotal
   //E num - NUMDOF 0  ONOFF 0  VAL 0.0  FUNCT none   TYPE Live

See the respective reference sections for :ref:`mechanical <designpointneumannconditions>`,
:ref:`MOMENT EB <designpointmomentebconditions>`, :ref:`TRANSPORT <designpointtransportneumannconditions>`,
:ref:`THERMO <designpointthermoneumannconditions>`, :ref:`PORO <designpointporoneumannconditions>`.

.. _springdashpotconditions:

Robin (Spring-Dashpot) conditions
----------------------------------

A spring-dashpot condition, also called a Robin boundary condition,
is used to give a surface boundary (and only surface boundaries!)
a stiffness and/or viscosity with respect to its displacement.
For each degree of freedom the stiffness and/or viscosity may be considered or not.
Also, both stiffness and viscosity may depend on a function definition.
The Direction can be given in the global coordinate system or with respect to the surface normal.
The input looks like this:

::

   ------------------------------DESIGN SURF ROBIN SPRING DASHPOT CONDITIONS
   E <surfset> NUMDOF 3 ONOFF 0 0 0 \
   STIFF <X_STIFF> <Y_STIFF> <Z_STIFF> TIMEFUNCTSTIFF 0 0 0 \
   VISCO <X_VISCO> <Y_VISCO> <Z_VISCO> TIMEFUNCTVISCO 0 0 0 \
   DISPLOFFSET 0.0 0.0 0.0 TIMEFUNCTDISPLOFFSET 0 0 0 FUNCTNONLINSTIFF 0 0 0 \
   DIRECTION xyz|refsurfnormal|cursurfnormal COUPLING none

- Commonly the Robin boundary condition couples the nodes at the surface to its original position.
  However, by giving a value to ``DISPLOFFSET``, one may introduce a prestressing of the spring.
  The point in space, to which it the surface nodes are coupled, may also move with time (``TIMEFUNCTDISPLOFFSET``).

- The direction in which the spring and dashpot are acting can be specified by the parameter ``DIRECTION``.
  This is either a global direction (``DIRECTION xyz``)
  or the surface normal (then only the x-axis has to be specified a finite value,
  but all three axis have to be given). The surface normal may then be either the reference one (``refsurfnormal``),
  or the current one (``cursurfnormal``).

- The Robin boundary condition can also couple the surface to another surface
  by specifying a couplingID (``COUPLING <int>``). The coupled surface is then given by a
  ``DESIGN SURF ROBIN SPRING DASHPOT COUPLING CONDITIONS``, see the following input:

::

   ------------------------------DESIGN SURF ROBIN SPRING DASHPOT CONDITIONS
   E <surfset> NUMDOF 3 ONOFF 0 0 0 \
   STIFF <X_STIFF> <Y_STIFF> <Z_STIFF> TIMEFUNCTSTIFF 0 0 0 \
   VISCO <X_VISCO> <Y_VISCO> <Z_VISCO> TIMEFUNCTVISCO 0 0 0 \
   DISPLOFFSET 0.0 0.0 0.0 TIMEFUNCTDISPLOFFSET 0 0 0 FUNCTNONLINSTIFF 0 0 0 \
   DIRECTION xyz|refsurfnormal|cursurfnormal COUPLING <couplingID>
   //
   ------------------------------DESIGN SURF ROBIN SPRING DASHPOT COUPLING CONDITIONS
   DSURF <numtotal>
   E <surfset> - <couplingID>

See also :ref:`designpointrobinspringdashpotconditions` and :ref:`designsurfrobinspringdashpotcouplingconditions`.

Constraint conditions
----------------------

Often, it is useful to prescribe not an absolute value of a nodal displacement or force,
but rather a displacement relative to other displacements, which is commonly called *constraint condition*. |FOURC| has a number of options to define such constraints.

Several nodes coupled for specific degrees of freedom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   ----------------------------------DESIGN POINT COUPLING CONDITIONS
   DPOINT <numtotal>
   E <pointset> - NUMDOF 6 ONOFF 1 1 1 1 1 1

Some applications (typically in structural / solid mechanics) require the coupling of certain DoFs of two or more nodes at the same geometrical position, while certain other DoFs of those
nodes shall remain independent (e.g. joints and hinges in frames).
While it is very easy to couple all(!) DoFs of several nodes at the same geometrical position (by simply merging the nodes into one node), things are more complicated if only certain DoFs are to be coupled.
While it would always be possible to introduce this coupling as a Dirichlet condition / Multipoint
Constraint into the final system of equations, we have decided to implement this at a more fundamental level by changing the assignment of DoFs according to the existing coupling
conditions.
Thus, if a point coupling condition is introduced for a set of nodes, the DoFs to be coupled are identified and the same(!) DoFs are then assigned to all participating nodes,
while the remaining uncoupled DoFs are created and assigned independently for each node. This required some changes in the way nodal DoFs are assigned and handled in |FOURC|.
However, after the initial DoF handling, the nice thing about this approach is that nothing needs to be done anymore at the system matrix level because the coupling is inherently included
in the DoF-maps. If you think of a web-like frame structure with many joints and hinges, this also means that the global system size is drastically reduced as compared to a Dirichlet type
handling of such constraints.

Features:

- new point coupling condition - e.g. for joints / hinges in structural mechanics
- no interference (hopefully) with element or face DoFs
- DofSet class is now able to handle repeated assignment of DoFs to more than one node
- DofSet class is now tracking and storing not only the first DoF ID of a node but all DoF IDs of a node

NOT included so far:

- support for derived DofSet classed that overload AssignDegreesOfFreedom (e.g. MortarDofSet, PotentialDofSet)
- support for special DofSet stuff (e.g. TransparentDofSet, Proxies...)
- support for bandwidth optimization (#define BW_OPT), which is currently however not used anyway

Example input file snippet for a Y-junction of three beam elements with a pin joint ("Momentengelenk")

::

   ------------------------------------DESIGN POINT COUPLING CONDITIONS
   E 1 NUMDOF 6 ONOFF 1 1 1 0 0 0
   -------------------------------------------------DNODE-NODE TOPOLOGY
   NODE 17 DNODE 1
   NODE 42 DNODE 1
   NODE 73 DNODE 1

All nodes of the given pointset have the same displacement in the directions with ONOFF flag 1, that is, here they are coupled for the three translational DoFs, while the rotational DoFs are free; see also :ref:`designpointcouplingconditions`.

.. danger::

   I gave a boundary condition (tried dirichlet and neumann) to a 3D structure in one coordinate direction and coupled the nodes in the other two directions.
   The results seem to be wrong.

Surface coupled to a node in a given direction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   -----------------------------DESIGN SURFACE NORMALDIR MULTIPNT CONSTRAINT 3D
   DSURF <numtotal>
   E <surfset> - <conditionID> <amplitude> <curveID> <inittime> <masternodeID> <n_x> <n_y> <n_z> [disp|x] [abs|rel]
   //
   -----------------------------DESIGN SURFACE NORMALDIR MULTIPNT CONSTRAINT 3D PEN
   DSURF <numtotal>
   E <surfset> - <conditionID> <amplitude> <curveID> <inittime> <penalty> <masternodeID> <n_x> <n_y> <n_z> [disp|x] [abs|rel]


The whole surface is displaced the same amount as a single master node (which is not defined by a DPOINT, but by its ID). The penalty version uses a different algorithm, where one has to provide a penalty parameter.
See also :ref:`designsurfacenormaldirmultipntconstraint3d` and :ref:`designsurfacenormaldirmultipntconstraint3dpen`.

Node displacement relative to a given surface or line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   ---------------------------------------DESIGN SURFACE MULTIPNT CONSTRAINT 3D
   DSURF <numtotal>
   E <surfset> - ConditionID <conditionID> amplitude <amplitude> curve <curveID> activeTime <inittime> planeNodes <node1> <node2> <node3> control [abs|rel]
   //
   ---------------------------------------DESIGN LINE MULTIPNT CONSTRAINT 2D
   DLINE <numtotal>
   E <lineset> - ConditionID <conditionID> amplitude <amplitude> curve <curveID> constrNode1 <node1> constrNode2 <node2> constrNode3 <node3> control [dist|angle] activeTime <inittime>
   //


This is a rather specific constraint, where a plane (or a line, respectively) is defined by three nodes, which are given as index of the ``<surfset>|<lineset>``
(the index is starting from 1 for whatever reasons),
and the other nodes of this set are displaced with respect to this plane/line. See :ref:`designsurfacemultipntconstraint3D` and :ref:`designlinemultipntconstraint2D`

Periodic boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   --------------------------DESIGN SURF PERIODIC BOUNDARY CONDITIONS
   DSURF  <numtotal>
   // definition of slave surface
   E <surfset> - ID 1 MASTER_OR_SLAVE Slave PLANE [xy|xz|yz] LAYER 1 ANGLE 0.0 ABSTREETOL 1e-6
   // definition of master surface
   E <surfset> - ID 1 MASTER_OR_SLAVE Master PLANE [xy|xz|yz] LAYER 1 ANGLE 0.0 ABSTREETOL 1e-6
   //
   --------------------------DESIGN LINE PERIODIC BOUNDARY CONDITIONS
   DLINE  <numtotal>
   // definition of slave line
   E <lineset> - ID 1 MASTER_OR_SLAVE Slave PLANE [xy|xz|yz] LAYER 1 ANGLE 0.0 ABSTREETOL 1e-6
   // definition of master line
   E <lineset> - ID 1 MASTER_OR_SLAVE Master PLANE [xy|xz|yz] LAYER 1 ANGLE 0.0 ABSTREETOL 1e-6


Periodic boundaries are defined as conditions, where nodes on one surface (normal to any of the cartesian coordinate directions, i.e., xy, yz, or xz),
are bound to the respective nodes on the opposite side, see the reference :ref:`designsurfperiodicboundaryconditions`, :ref:`designlineperiodicboundaryconditions`.
They can be defined at one or several sides of the structure.
Nodes at either side must be at equal plane coordinates (within the tolerance given by `ABSTREETOL`), and the normal distance must be equal for each nodal pair.

The definition of `ANGLE` is used for rotational symmetry.
For this case, the master must always be in the defined `PLANE`, and the slave is rotated by the given angle, while the same plane must be given.



Contact conditions
------------------

Contact conditions, which in |FOURC| are set up by the keyword ``MORTAR``
are defined along lines (2D) or surfaces (3D). At least one contact pair
is necessary:

::

   ---------------------------DESIGN LINE|SURF MORTAR CONTACT CONDITIONS 2D|3D
   DLINE              <numtotal>
   //E <num> - InterfaceID <interfaceID> Side <Master|Slave|Selfcontact> Initialization <Inactive|Active> [FrCoeffOrBound 0.0] [AdhesionBound 0.0] [Application <Solidcontact|Beamtosolidcontact|Beamtosolidmeshtying>] [DbcHandling <DoNothing|RemoveDBCSlaveNodes>] [TwoHalfPass 0.0] [RefConfCheckNonSmoothSelfContactSurface 0.0] [ConstitutiveLawID 0]

The parameters
``FrCoeffOrBound, AdhesionBound, Application, DbcHandling, TwoHalfPass, RefConfCheckNonSmoothSelfContactSurface and ConstitutiveLawID``
are optional. You'll find more information about contact in the
:ref:`contact and meshtying <contactandmeshtying>` section.


.. _`sec:bcdefinitionForPre_exodus`:

Definition in a .bc file (for use with ``pre_exodus``)
------------------------------------------------------

In general, boundary conditions are defined in the .bc file. This is
done per previously defined node and side set or element block by the
following general syntax (cf. the default.bc file):

::

   Node Set, named:
   Property Name: INFLOW
   has 45107 Nodes
   '*ns0="CONDITION"'
   sectionname=""
   description=""

for boundary conditions acting on nodes, like displacements or
temperatures, and

::

   Side Set, named: innerSurface
   has 45107 Nodes
   '*ss0="CONDITION"'
   sectionname=""
   description=""

for boundary conditions acting on element surfaces, like surface
pressure or thermal convection, and

::

   Element Block, named:
   of Shape: HEX8
   has 9417816 Elements
   *eb0="ELEMENT"
   sectionname="STRUCTURE"
   description="MAT 1 KINEM nonlinear "

for element types, respectively (the values of the sectionname and description are just examples).

Note that you can specify a condition also on an ElementBlock, just replace 'ELEMENT' with 'CONDITION'
The 'E num' in the dat-file depends on the order of the specification below

These syntax blocks are automatically
created for a given mesh when running the pre_exodus script on the
corresponding .e file and are then collected at the top of the
default.bc file. The node set IDs \*ns and element set IDs \*eb are
automatically matched the IDs in the .e file in this case.

To apply boundary conditions, a string has to be given for
*sectionname=* and *description=*. A collection of all currently
implemented boundary conditions is given in the
:ref:`Prescribed condition reference <prescribedconditionreference>`.
In this, the *sectionname* of a
boundary condition is given first (e.g. DESIGN POINT DIRICH CONDITIONS)
followed by the *E num* entry which (without the E num) has to be put as
the *description* (see above).

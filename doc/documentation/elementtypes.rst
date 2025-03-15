.. _elementtypes:

Element types
-------------

.. _structureelements:

STRUCTURE ELEMENTS
~~~~~~~~~~~~~~~~~~

.. _structurebeam3:

BEAM3
^^^^^

*Shapes:*

- :ref:`LINE2 <line2>` (2 nodes)
- :ref:`LINE3 <line3>` (3 nodes)
- :ref:`LINE4 <line4>` (4 nodes)
- :ref:`LINE5 <line5>` (5 nodes)
- :ref:`LINE6 <line6>` (6 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+

.. _structurebeam3r:

BEAM3R
^^^^^^

*Shapes:*

- :ref:`LINE2 <line2>` (2 nodes)
- :ref:`LINE3 <line3>` (3 nodes)
- :ref:`LINE4 <line4>` (4 nodes)
- :ref:`LINE5 <line5>` (5 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+
| TRIADS       |  6 x number |
+--------------+-------------+
| FAD (opt.)   | None        |
+--------------+-------------+

.. _structurebeam3eb:

BEAM3EB
^^^^^^^

*Shapes:*

- :ref:`LINE2 <line2>` (2 nodes)
- :ref:`LINE3 <line3>` (3 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+

.. _structurebeam3k:

BEAM3K
^^^^^^

*Shapes:*

- :ref:`LINE2 <line2>` (2 nodes)
- :ref:`LINE3 <line3>` (3 nodes)
- :ref:`LINE4 <line4>` (4 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| WK           |  1 x number |
+--------------+-------------+
| ROTVEC       |  1 x number |
+--------------+-------------+
| MAT          |  1 x number |
+--------------+-------------+
| TRIADS       |  6 x number |
+--------------+-------------+
| FAD (opt.)   | None        |
+--------------+-------------+

.. _structurerigidsphere:

RIGIDSPHERE
^^^^^^^^^^^

*Shapes:*

- :ref:`POINT1 <point1>` (1 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| RADIUS       |  1 x number |
+--------------+-------------+
| DENSITY      |  1 x number |
+--------------+-------------+

.. _structureshell7p:

SHELL7P
^^^^^^^

*Shapes:*

- :ref:`QUAD4 <quad4>` (4 nodes)
- :ref:`QUAD8 <quad8>` (8 nodes)
- :ref:`QUAD9 <quad9>` (9 nodes)
- :ref:`TRI3 <tri3>` (3 nodes)
- :ref:`TRI6 <tri6>` (6 nodes)

**Parameters**

+-----------------+-------------+
| Parameter       | Values      |
+=================+=============+
| MAT             |  1 x number |
+-----------------+-------------+
| THICK           |  1 x number |
+-----------------+-------------+
| SDC             |  1 x number |
+-----------------+-------------+
| EAS (opt.)      |  5 x string |
+-----------------+-------------+
| ANS (opt.)      |    none     |
+-----------------+-------------+
| RAD (opt.)      |  3 x number |
+-----------------+-------------+
| AXI (opt.)      |  3 x number |
+-----------------+-------------+
| CIR (opt.)      |  3 x number |
+-----------------+-------------+
| FIBER1 (opt.)   |  3 x number |
+-----------------+-------------+
| FIBER2 (opt.)   |  3 x number |
+-----------------+-------------+
| FIBER3 (opt.)   |  3 x number |
+-----------------+-------------+

.. _structureshell7pscatra:

SHELL7PSCATRA
^^^^^^^^^^^^^

*Shapes:*

- :ref:`QUAD4 <quad4>` (4 nodes)
- :ref:`QUAD8 <quad8>` (8 nodes)
- :ref:`QUAD9 <quad9>` (9 nodes)
- :ref:`TRI3 <tri3>` (3 nodes)
- :ref:`TRI6 <tri6>` (6 nodes)

**Parameters**

+-----------------+-------------+
| Parameter       | Values      |
+=================+=============+
| MAT             |  1 x number |
+-----------------+-------------+
| THICK           |  1 x number |
+-----------------+-------------+
| SDC             |  1 x number |
+-----------------+-------------+
| EAS (opt.)      |  5 x string |
+-----------------+-------------+
| ANS (opt.)      |    none     |
+-----------------+-------------+
| RAD (opt.)      |  3 x number |
+-----------------+-------------+
| AXI (opt.)      |  3 x number |
+-----------------+-------------+
| CIR (opt.)      |  3 x number |
+-----------------+-------------+
| FIBER1 (opt.)   |  3 x number |
+-----------------+-------------+
| FIBER2 (opt.)   |  3 x number |
+-----------------+-------------+
| FIBER3 (opt.)   |  3 x number |
+-----------------+-------------+
| TYPE            |  1 x string |
+-----------------+-------------+

.. _structuresolid:

SOLID
^^^^^

*Shapes:*

- :ref:`HEX8 <hex8>` (8 nodes)
- :ref:`HEX18 <hex18>` (18 nodes)
- :ref:`HEX20 <hex20>` (20 nodes)
- :ref:`HEX27 <hex27>` (27 nodes)
- :ref:`nurbs27 <nurbs27>` (27 nodes)
- :ref:`tet4 <tet4>` (4 nodes)
- :ref:`tet10 <tet10>` (10 nodes)
- :ref:`wedge6 <wedge6>` (6 nodes)
- :ref:`pyramid5 <pyramid5>` (5 nodes)

**Parameters**

+-----------------+-------------+
| Parameter       | Values      |
+=================+=============+
| MAT             |  1 x number |
+-----------------+-------------+
| KINEM           |  1 x string |
+-----------------+-------------+
| TECH            |  1 x string |
+-----------------+-------------+
| PRESTRESS_TECH  |  1 x string |
+-----------------+-------------+
| AXI (opt.)      |  3 x number |
+-----------------+-------------+
| CIR (opt.)      |  3 x number |
+-----------------+-------------+
| FIBER1 (opt.)   |  3 x number |
+-----------------+-------------+
| FIBER2 (opt.)   |  3 x number |
+-----------------+-------------+
| FIBER3 (opt.)   |  3 x number |
+-----------------+-------------+
| RAD (opt.)      |  3 x number |
+-----------------+-------------+


.. _structuresolidscatra:

SOLIDSCATRA
^^^^^^^^^^^

*Shapes:*

- :ref:`HEX8 <hex8>` (8 nodes)
- :ref:`HEX27 <hex27>` (27 nodes)
- :ref:`TET4 <tet4>` (4 nodes)
- :ref:`TET10 <tet10>` (10 nodes)

**Parameters**

+-----------------------+-------------+
| Parameter             | Values      |
+=======================+=============+
| MAT                   |  1 x number |
+-----------------------+-------------+
| KINEM                 |  1 x string |
+-----------------------+-------------+
| PRESTRESS_TECH (opt)  |  1 x string |
+-----------------------+-------------+
| TECH (opt, hex8 only) |  1 x string |
+-----------------------+-------------+
| TYPE                  |  1 x string |
+-----------------------+-------------+
| RAD (opt.)            |  3 x number |
+-----------------------+-------------+
| AXI (opt.)            |  3 x number |
+-----------------------+-------------+
| CIR (opt.)            |  3 x number |
+-----------------------+-------------+
| FIBER1 (opt.)         |  3 x number |
+-----------------------+-------------+
| FIBER2 (opt.)         |  3 x number |
+-----------------------+-------------+
| FIBER3 (opt.)         |  3 x number |
+-----------------------+-------------+

.. _structuretorsion3:

TORSION3
^^^^^^^^

*Shapes:*

- :ref:`LINE3 <line3>` (3 nodes)

**Parameters**

+------------------+-------------+
| Parameter        | Values      |
+==================+=============+
| MAT              |  1 x number |
+------------------+-------------+
| BENDINGPOTENTIAL |  1 x string |
+------------------+-------------+

.. _structuretruss3:

TRUSS3
^^^^^^

*Shapes:*

- :ref:`LINE2 <line2>` (2 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+
| CROSS        |  1 x number |
+--------------+-------------+
| KINEM        |  1 x string |
+--------------+-------------+

.. _structurewall:

WALL
^^^^

*Shapes:*

- :ref:`NURBS4 <nurbs4>` (4 nodes)
- :ref:`NURBS9 <nurbs9>` (9 nodes)
- :ref:`QUAD4 <quad4>` (4 nodes)
- :ref:`QUAD8 <quad8>` (8 nodes)
- :ref:`QUAD9 <quad9>` (9 nodes)
- :ref:`TRI3 <tri3>` (3 nodes)
- :ref:`TRI6 <tri6>` (6 nodes)

**Parameters**

+---------------+-------------+
| Parameter     | Values      |
+===============+=============+
| MAT           |  1 x number |
+---------------+-------------+
| KINEM         |  1 x string |
+---------------+-------------+
| EAS           |  1 x string |
+---------------+-------------+
| THICK         |  1 x number |
+---------------+-------------+
| STRESS_STRAIN |  1 x string |
+---------------+-------------+
| GP            |  2 x number |
+---------------+-------------+

.. _structurewallscatra:

WALLSCATRA
^^^^^^^^^^

*Shapes:*

- :ref:`NURBS4 <nurbs4>` (4 nodes)
- :ref:`NURBS9 <nurbs9>` (9 nodes)
- :ref:`QUAD4 <quad4>` (4 nodes)
- :ref:`QUAD8 <quad8>` (8 nodes)
- :ref:`QUAD9 <quad9>` (9 nodes)
- :ref:`TRI3 <tri3>` (3 nodes)
- :ref:`TRI6 <tri6>` (6 nodes)

**Parameters**

+---------------+-------------+
| Parameter     | Values      |
+===============+=============+
| MAT           |  1 x number |
+---------------+-------------+
| KINEM         |  1 x string |
+---------------+-------------+
| EAS           |  1 x string |
+---------------+-------------+
| THICK         |  1 x number |
+---------------+-------------+
| STRESS_STRAIN |  1 x string |
+---------------+-------------+
| GP            |  2 x number |
+---------------+-------------+
| TYPE          |  1 x string |
+---------------+-------------+

.. _structurewallq4poro:

WALLQ4PORO
^^^^^^^^^^

*Shapes:*

- :ref:`QUAD4 <quad4>` (4 nodes)

**Parameters**

+---------------+-------------+
| Parameter     | Values      |
+===============+=============+
| MAT           |  1 x number |
+---------------+-------------+
| KINEM         |  1 x string |
+---------------+-------------+
| EAS           |  1 x string |
+---------------+-------------+
| THICK         |  1 x number |
+---------------+-------------+
| STRESS_STRAIN |  1 x string |
+---------------+-------------+
| GP            |  2 x number |
+---------------+-------------+

.. _structurewallq4poroscatra:

WALLQ4POROSCATRA
^^^^^^^^^^^^^^^^

*Shapes:*

- :ref:`QUAD4 <quad4>` (4 nodes)

**Parameters**

+---------------+-------------+
| Parameter     | Values      |
+===============+=============+
| MAT           |  1 x number |
+---------------+-------------+
| KINEM         |  1 x string |
+---------------+-------------+
| EAS           |  1 x string |
+---------------+-------------+
| THICK         |  1 x number |
+---------------+-------------+
| STRESS_STRAIN |  1 x string |
+---------------+-------------+
| GP            |  2 x number |
+---------------+-------------+
| TYPE          |  1 x string |
+---------------+-------------+

.. _structurewallq4porop1:

WALLQ4POROP1
^^^^^^^^^^^^

*Shapes:*

- :ref:`QUAD4 <quad4>` (4 nodes)

**Parameters**

+---------------+-------------+
| Parameter     | Values      |
+===============+=============+
| MAT           |  1 x number |
+---------------+-------------+
| KINEM         |  1 x string |
+---------------+-------------+
| EAS           |  1 x string |
+---------------+-------------+
| THICK         |  1 x number |
+---------------+-------------+
| STRESS_STRAIN |  1 x string |
+---------------+-------------+
| GP            |  2 x number |
+---------------+-------------+

.. _structurewallq4porop1scatra:

WALLQ4POROP1SCATRA
^^^^^^^^^^^^^^^^^^

*Shapes:*

- :ref:`QUAD4 <quad4>` (4 nodes)

**Parameters**

+---------------+-------------+
| Parameter     | Values      |
+===============+=============+
| MAT           |  1 x number |
+---------------+-------------+
| KINEM         |  1 x string |
+---------------+-------------+
| EAS           |  1 x string |
+---------------+-------------+
| THICK         |  1 x number |
+---------------+-------------+
| STRESS_STRAIN |  1 x string |
+---------------+-------------+
| GP            |  2 x number |
+---------------+-------------+
| TYPE          |  1 x string |
+---------------+-------------+

.. _structurewallq9poro:

WALLQ9PORO
^^^^^^^^^^

*Shapes:*

- :ref:`QUAD9 <quad9>` (9 nodes)

**Parameters**

+---------------+-------------+
| Parameter     | Values      |
+===============+=============+
| MAT           |  1 x number |
+---------------+-------------+
| KINEM         |  1 x string |
+---------------+-------------+
| EAS           |  1 x string |
+---------------+-------------+
| THICK         |  1 x number |
+---------------+-------------+
| STRESS_STRAIN |  1 x string |
+---------------+-------------+
| GP            |  2 x number |
+---------------+-------------+

.. _fluidelements:

FLUID ELEMENTS
~~~~~~~~~~~~~~

.. _fluidfluid:

FLUID
^^^^^

*Shapes:*

- :ref:`HEX20 <hex20>` (20 nodes)
- :ref:`HEX27 <hex27>` (27 nodes)
- :ref:`HEX8 <hex8>` (8 nodes)
- :ref:`NURBS27 <nurbs27>` (27 nodes)
- :ref:`NURBS4 <nurbs4>` (4 nodes)
- :ref:`NURBS8 <nurbs8>` (8 nodes)
- :ref:`NURBS9 <nurbs9>` (9 nodes)
- :ref:`PYRAMID5 <pyramid5>` (5 nodes)
- :ref:`QUAD4 <quad4>` (4 nodes)
- :ref:`QUAD8 <quad8>` (8 nodes)
- :ref:`QUAD9 <quad9>` (9 nodes)
- :ref:`TET10 <tet10>` (10 nodes)
- :ref:`TET4 <tet4>` (4 nodes)
- :ref:`TRI3 <tri3>` (3 nodes)
- :ref:`TRI6 <tri6>` (6 nodes)
- :ref:`WEDGE15 <wedge15>` (15 nodes)
- :ref:`WEDGE6 <wedge6>` (6 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+
| NA           |  1 x string |
+--------------+-------------+

.. _fluidfluidxw:

FLUIDXW
^^^^^^^

*Shapes:*

- :ref:`HEX8 <hex8>` (8 nodes)
- :ref:`TET4 <tet4>` (4 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+
| NA           |  1 x string |
+--------------+-------------+

.. _fluidfluidhdg:

FLUIDHDG
^^^^^^^^

*Shapes:*

- :ref:`HEX20 <hex20>` (20 nodes)
- :ref:`HEX27 <hex27>` (27 nodes)
- :ref:`HEX8 <hex8>` (8 nodes)
- :ref:`NURBS27 <nurbs27>` (27 nodes)
- :ref:`NURBS4 <nurbs4>` (4 nodes)
- :ref:`NURBS8 <nurbs8>` (8 nodes)
- :ref:`NURBS9 <nurbs9>` (9 nodes)
- :ref:`PYRAMID5 <pyramid5>` (5 nodes)
- :ref:`QUAD4 <quad4>` (4 nodes)
- :ref:`QUAD8 <quad8>` (8 nodes)
- :ref:`QUAD9 <quad9>` (9 nodes)
- :ref:`TET10 <tet10>` (10 nodes)
- :ref:`TET4 <tet4>` (4 nodes)
- :ref:`TRI3 <tri3>` (3 nodes)
- :ref:`TRI6 <tri6>` (6 nodes)
- :ref:`WEDGE15 <wedge15>` (15 nodes)
- :ref:`WEDGE6 <wedge6>` (6 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+
| NA           |  1 x string |
+--------------+-------------+
| DEG          |  1 x number |
+--------------+-------------+
| SPC (opt.)   |  1 x number |
+--------------+-------------+

.. _fluidfluidhdgweakcomp:

FLUIDHDGWEAKCOMP
^^^^^^^^^^^^^^^^

*Shapes:*

- :ref:`HEX20 <hex20>` (20 nodes)
- :ref:`HEX27 <hex27>` (27 nodes)
- :ref:`HEX8 <hex8>` (8 nodes)
- :ref:`NURBS27 <nurbs27>` (27 nodes)
- :ref:`NURBS4 <nurbs4>` (4 nodes)
- :ref:`NURBS8 <nurbs8>` (8 nodes)
- :ref:`NURBS9 <nurbs9>` (9 nodes)
- :ref:`PYRAMID5 <pyramid5>` (5 nodes)
- :ref:`QUAD4 <quad4>` (4 nodes)
- :ref:`QUAD8 <quad8>` (8 nodes)
- :ref:`QUAD9 <quad9>` (9 nodes)
- :ref:`TET10 <tet10>` (10 nodes)
- :ref:`TET4 <tet4>` (4 nodes)
- :ref:`TRI3 <tri3>` (3 nodes)
- :ref:`TRI6 <tri6>` (6 nodes)
- :ref:`WEDGE15 <wedge15>` (15 nodes)
- :ref:`WEDGE6 <wedge6>` (6 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+
| NA           |  1 x string |
+--------------+-------------+
| DEG          |  1 x number |
+--------------+-------------+
| SPC (opt.)   |  1 x number |
+--------------+-------------+

.. _lubricationelements:

LUBRICATION ELEMENTS
~~~~~~~~~~~~~~~~~~~~

.. _lubricationlubrication:

LUBRICATION
^^^^^^^^^^^

*Shapes:*

- :ref:`LINE2 <line2>` (2 nodes)
- :ref:`LINE3 <line3>` (3 nodes)
- :ref:`QUAD4 <quad4>` (4 nodes)
- :ref:`QUAD8 <quad8>` (8 nodes)
- :ref:`QUAD9 <quad9>` (9 nodes)
- :ref:`TRI3 <tri3>` (3 nodes)
- :ref:`TRI6 <tri6>` (6 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+

.. _transportelements:

TRANSPORT ELEMENTS
~~~~~~~~~~~~~~~~~~

.. _transporttransp:

TRANSP
^^^^^^

*Shapes:*

- :ref:`HEX20 <hex20>` (20 nodes)
- :ref:`HEX27 <hex27>` (27 nodes)
- :ref:`HEX8 <hex8>` (8 nodes)
- :ref:`LINE2 <line2>` (2 nodes)
- :ref:`LINE3 <line3>` (3 nodes)
- :ref:`NURBS2 <nurbs2>` (2 nodes)
- :ref:`NURBS27 <nurbs27>` (27 nodes)
- :ref:`NURBS3 <nurbs3>` (3 nodes)
- :ref:`NURBS4 <nurbs4>` (4 nodes)
- :ref:`NURBS8 <nurbs8>` (8 nodes)
- :ref:`NURBS9 <nurbs9>` (9 nodes)
- :ref:`PYRAMID5 <pyramid5>` (5 nodes)
- :ref:`QUAD4 <quad4>` (4 nodes)
- :ref:`QUAD8 <quad8>` (8 nodes)
- :ref:`QUAD9 <quad9>` (9 nodes)
- :ref:`TET10 <tet10>` (10 nodes)
- :ref:`TET4 <tet4>` (4 nodes)
- :ref:`TRI3 <tri3>` (3 nodes)
- :ref:`TRI6 <tri6>` (6 nodes)
- :ref:`WEDGE15 <wedge15>` (15 nodes)
- :ref:`WEDGE6 <wedge6>` (6 nodes)

**Parameters**

+---------------+-------------+
| Parameter     | Values      |
+===============+=============+
| MAT           |  1 x number |
+---------------+-------------+
| TYPE          |  1 x string |
+---------------+-------------+
| FIBER1 (opt.) |  3 x number |
+---------------+-------------+

.. _transport2elements:

TRANSPORT2 ELEMENTS
~~~~~~~~~~~~~~~~~~~

.. _transport2transp:

TRANSP
^^^^^^

*Shapes:*

- :ref:`HEX20 <hex20>` (20 nodes)
- :ref:`HEX27 <hex27>` (27 nodes)
- :ref:`HEX8 <hex8>` (8 nodes)
- :ref:`LINE2 <line2>` (2 nodes)
- :ref:`LINE3 <line3>` (3 nodes)
- :ref:`NURBS2 <nurbs2>` (2 nodes)
- :ref:`NURBS27 <nurbs27>` (27 nodes)
- :ref:`NURBS3 <nurbs3>` (3 nodes)
- :ref:`NURBS4 <nurbs4>` (4 nodes)
- :ref:`NURBS8 <nurbs8>` (8 nodes)
- :ref:`NURBS9 <nurbs9>` (9 nodes)
- :ref:`PYRAMID5 <pyramid5>` (5 nodes)
- :ref:`QUAD4 <quad4>` (4 nodes)
- :ref:`QUAD8 <quad8>` (8 nodes)
- :ref:`QUAD9 <quad9>` (9 nodes)
- :ref:`TET10 <tet10>` (10 nodes)
- :ref:`TET4 <tet4>` (4 nodes)
- :ref:`TRI3 <tri3>` (3 nodes)
- :ref:`TRI6 <tri6>` (6 nodes)
- :ref:`WEDGE15 <wedge15>` (15 nodes)
- :ref:`WEDGE6 <wedge6>` (6 nodes)

**Parameters**

+---------------+-------------+
| Parameter     | Values      |
+===============+=============+
| MAT           |  1 x number |
+---------------+-------------+
| TYPE          |  1 x string |
+---------------+-------------+
| FIBER1 (opt.) |  3 x number |
+---------------+-------------+

.. _aleelements:

ALE ELEMENTS
~~~~~~~~~~~~

.. _aleale2:

ALE2
^^^^

*Shapes:*

- :ref:`QUAD4 <quad4>` (4 nodes)
- :ref:`QUAD8 <quad8>` (8 nodes)
- :ref:`QUAD9 <quad9>` (9 nodes)
- :ref:`TRI3 <tri3>` (3 nodes)
- :ref:`TRI6 <tri6>` (6 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+

.. _aleale3:

ALE3
^^^^

*Shapes:*

- :ref:`HEX20 <hex20>` (20 nodes)
- :ref:`HEX27 <hex27>` (27 nodes)
- :ref:`HEX8 <hex8>` (8 nodes)
- :ref:`PYRAMID5 <pyramid5>` (5 nodes)
- :ref:`TET10 <tet10>` (10 nodes)
- :ref:`TET4 <tet4>` (4 nodes)
- :ref:`WEDGE15 <wedge15>` (15 nodes)
- :ref:`WEDGE6 <wedge6>` (6 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+

.. _thermoelements:

THERMO ELEMENTS
~~~~~~~~~~~~~~~

.. _thermothermo:

THERMO
^^^^^^

*Shapes:*

- :ref:`HEX20 <hex20>` (20 nodes)
- :ref:`HEX27 <hex27>` (27 nodes)
- :ref:`HEX8 <hex8>` (8 nodes)
- :ref:`LINE2 <line2>` (2 nodes)
- :ref:`LINE3 <line3>` (3 nodes)
- :ref:`NURBS27 <nurbs27>` (27 nodes)
- :ref:`NURBS4 <nurbs4>` (4 nodes)
- :ref:`NURBS9 <nurbs9>` (9 nodes)
- :ref:`PYRAMID5 <pyramid5>` (5 nodes)
- :ref:`QUAD4 <quad4>` (4 nodes)
- :ref:`QUAD8 <quad8>` (8 nodes)
- :ref:`QUAD9 <quad9>` (9 nodes)
- :ref:`TET10 <tet10>` (10 nodes)
- :ref:`TET4 <tet4>` (4 nodes)
- :ref:`TRI3 <tri3>` (3 nodes)
- :ref:`TRI6 <tri6>` (6 nodes)
- :ref:`WEDGE15 <wedge15>` (15 nodes)
- :ref:`WEDGE6 <wedge6>` (6 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+

.. _arteryelements:

ARTERY ELEMENTS
~~~~~~~~~~~~~~~

.. _arteryart:

ART
^^^

*Shapes:*

- :ref:`LINE2 <line2>` (2 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+
| GP           |  1 x number |
+--------------+-------------+
| TYPE         |  1 x string |
+--------------+-------------+
| DIAM         |  1 x number |
+--------------+-------------+

.. _reduced d airwayselements:

REDUCED D AIRWAYS ELEMENTS
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _reduced d airwaysred_airway:

RED_AIRWAY
^^^^^^^^^^

*Shapes:*

- :ref:`LINE2 <line2>` (2 nodes)

**Parameters**

+------------------------+-------------+
| Parameter              | Values      |
+========================+=============+
| MAT                    |  1 x number |
+------------------------+-------------+
| ElemSolvingType        |  1 x string |
+------------------------+-------------+
| TYPE                   |  1 x string |
+------------------------+-------------+
| Resistance             |  1 x string |
+------------------------+-------------+
| PowerOfVelocityProfile |  1 x number |
+------------------------+-------------+
| WallElasticity         |  1 x number |
+------------------------+-------------+
| PoissonsRatio          |  1 x number |
+------------------------+-------------+
| ViscousTs              |  1 x number |
+------------------------+-------------+
| ViscousPhaseShift      |  1 x number |
+------------------------+-------------+
| WallThickness          |  1 x number |
+------------------------+-------------+
| Area                   |  1 x number |
+------------------------+-------------+
| Generation             |  1 x number |
+------------------------+-------------+
| AirwayColl (opt.)      |  1 x number |
+------------------------+-------------+
| BranchLength (opt.)    |  1 x number |
+------------------------+-------------+
| Open_Init (opt.)       |  1 x number |
+------------------------+-------------+
| Pcrit_Close (opt.)     |  1 x number |
+------------------------+-------------+
| Pcrit_Open (opt.)      |  1 x number |
+------------------------+-------------+
| S_Close (opt.)         |  1 x number |
+------------------------+-------------+
| S_Open (opt.)          |  1 x number |
+------------------------+-------------+

.. _reduced d airwaysred_acinus:

RED_ACINUS
^^^^^^^^^^

*Shapes:*

- :ref:`LINE2 <line2>` (2 nodes)

**Parameters**

+--------------------+-------------+
| Parameter          | Values      |
+====================+=============+
| MAT                |  1 x number |
+--------------------+-------------+
| TYPE               |  1 x string |
+--------------------+-------------+
| AcinusVolume       |  1 x number |
+--------------------+-------------+
| AlveolarDuctVolume |  1 x number |
+--------------------+-------------+
| Area (opt.)        |  1 x number |
+--------------------+-------------+
| BETA (opt.)        |  1 x number |
+--------------------+-------------+
| E1_0 (opt.)        |  1 x number |
+--------------------+-------------+
| E1_01 (opt.)       |  1 x number |
+--------------------+-------------+
| E1_02 (opt.)       |  1 x number |
+--------------------+-------------+
| E1_EXP (opt.)      |  1 x number |
+--------------------+-------------+
| E1_EXP1 (opt.)     |  1 x number |
+--------------------+-------------+
| E1_EXP2 (opt.)     |  1 x number |
+--------------------+-------------+
| E1_LIN (opt.)      |  1 x number |
+--------------------+-------------+
| E1_LIN1 (opt.)     |  1 x number |
+--------------------+-------------+
| E1_LIN2 (opt.)     |  1 x number |
+--------------------+-------------+
| KAPPA (opt.)       |  1 x number |
+--------------------+-------------+
| TAU (opt.)         |  1 x number |
+--------------------+-------------+
| TAU1 (opt.)        |  1 x number |
+--------------------+-------------+
| TAU2 (opt.)        |  1 x number |
+--------------------+-------------+

.. _reduced d airwaysred_acinar_inter_dep:

RED_ACINAR_INTER_DEP
^^^^^^^^^^^^^^^^^^^^

*Shapes:*

- :ref:`LINE2 <line2>` (2 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+

.. _acousticelements:

ACOUSTIC ELEMENTS
~~~~~~~~~~~~~~~~~

.. _acousticacoustic:

ACOUSTIC
^^^^^^^^

*Shapes:*

- :ref:`HEX8 <hex8>` (8 nodes)
- :ref:`QUAD4 <quad4>` (4 nodes)
- :ref:`QUAD9 <quad9>` (9 nodes)
- :ref:`TET4 <tet4>` (4 nodes)
- :ref:`TRI3 <tri3>` (3 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+
| DEG          |  1 x number |
+--------------+-------------+
| SPC          |  1 x number |
+--------------+-------------+

.. _acousticacousticsol:

ACOUSTICSOL
^^^^^^^^^^^

*Shapes:*

- :ref:`HEX8 <hex8>` (8 nodes)
- :ref:`QUAD4 <quad4>` (4 nodes)
- :ref:`QUAD9 <quad9>` (9 nodes)
- :ref:`TET4 <tet4>` (4 nodes)
- :ref:`TRI3 <tri3>` (3 nodes)

**Parameters**

+--------------+-------------+
| Parameter    | Values      |
+==============+=============+
| MAT          |  1 x number |
+--------------+-------------+
| DEG          |  1 x number |
+--------------+-------------+
| SPC          |  1 x number |
+--------------+-------------+


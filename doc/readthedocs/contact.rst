.. _contactandmeshtying:

Contact and Mesh tying
======================

.. _ contact:

Contact
--------

Contact conditions, which in |FOURC| are set up by the keyword ``MORTAR`` are defined along lines (2D)
or surfaces (3D). Two types of contact can be defined: Master/Slave contact and Selfcontact. In any case, at least one contact pair is necessary:

- 2D:

  ::

     ---------------------------DESIGN LINE MORTAR CONTACT CONDITIONS 2D
     DLINE 2
     E 1 - 1 Master <further parameters>
     E 2 - 1 Slave <further parameters>

for master/slave contact, or

  ::

     ---------------------------DESIGN LINE MORTAR CONTACT CONDITIONS 2D
     DLINE 1
     E 1 - 1 Selfcontact <further parameters>

for self contact, see also :ref:`DESIGN MORTAR CONTACT CONDITIONS 2D<designlinemortarcontactconditions2d>`.


- 3D:

  The 3D case is analogous, for master/slave contact:

  ::

     ---------------------------DESIGN SURF MORTAR CONTACT CONDITIONS 3D
     DSURF 2
     E 1 - 1 Master <further parameters>
     E 2 - 1 Slave <further parameters>

  and for self contact:

  ::

     ---------------------------DESIGN SURF MORTAR CONTACT CONDITIONS 3D
     DSURF 1
     E 1 - 1 Selfcontact <further parameters>


  see see :ref:`DESIGN MORTAR CONTACT CONDITIONS 3D<designsurfmortarcontactconditions3d>`.

The further parameters are:

::

   [Inactive|Active] \
   FrCoeffOrBound 0.0 \
   AdhesionBound 0.0  \
   [Solidcontact | Beamtosolidcontact | Beamtosolidmeshtying]  \
   [DoNothing | RemoveDBCSlaveNodes]  \
   Twohalfpass 0|1  \
   RefConfCheckNonSmoothSelfContactSurface 0|1  \
   ConstitutiveLawID  <num>

Remarks:

- The keyword ``Active`` declarates a surface pair to be in contact initially.
  it is only valid for slave surfaces 
  (but ``Inactive`` must be given for Master surfaces as well if further parameters are given).
  The default is ``Inactive`` anyway, and it is not necessary to denote surfaces as being active,
  since a contact search is conducted in any case.
- While all further parameters are optional, one must not miss any parameter between others; 
  it is only possible to omit parameters at the end. 
- ``AdhesionBound`` declares an adhesive contact condition. 
  The value given subsequently is the tensile strength of the adhesive joint. 
  Note that you have to define the parameter ``ADHESION`` as bounded in :ref:`CONTACT DYNAMIC <SECcontactdynamic>`.`
- the parameters ``TwoHalfPass`` and ``RefConfCheckNonSmoothSelfContactSurface`` 
  do only make sense for self contact. 
  However, they also have to be entered in master/slave contact analyses, 
  if the ``ConstitutiveLawID`` is given, since the position of parameters is crucial.



Contact and symmetry conditions
"""""""""""""""""""""""""""""""

When a contact surface touches a symmetry plane or some other dirichlet boundary condition
(or a contact line touches a line with dirichlet conditions, respectively), 
one has three possibilities to overcome the clashing of two contstrains at the common line/point. 
One can 

#. remove the contact condition,
#. remove the boundary condition,
#. declare a specific condition to allow both conditions.

For the first option, on can use the optional parameter ``RemoveDBCSlaveNodes`` 
in the Slave definition as shown above.

For option two, one can simply define a line dirichlet condition, 
where all dirichlet boundary conditions are removed.

For the third option one can tell |FOURC| that a line belongs to the symmetry plane / dirichlet boundary condition *and* the contact surface.
This is done using the so-called mortar symmetry conditions (Note that the word *symmetry* does not mean that it must be a symmetry condition, it can be any any dirichlet boundary condition, even with non-zero displacement value)::

   --------------------------DESIGN LINE MORTAR SYMMETRY CONDITIONS 3D
   DLINE <n>
   E <num> - ONOFF 0 0 0  
   ----------------------DESIGN POINT MORTAR SYMMETRY CONDITIONS 2D/3D
   DPOINT <n>
   E <num> - ONOFF 0 0 0  

The ONOFF value has to be set to one in the direction of the dirichlet boundary condition, 
and if a contact surface touches two planes with dirchlet conditions, 
the ``DESIGN POINT MORTAR SYMMETRY`` has to be defined as well.

**Reference:** :ref:`DESIGN MORTAR SYMMETRY CONDITIONS<designlinemortarsymmetryconditions3d>`, :ref:`DESIGN MORTAR SYMMETRY CONDITIONS 2D/3D<designpointmortarsymmetryconditions2d/3d>`.


Contact at edges/corners
"""""""""""""""""""""""""

if an edge of a (3D) structure is involved in contact, one may define the edge separately
(in addition to the adjacent contact surfaces, which probably may also come into contact).
For this, the ``MORTAR EDGE CONDITIONS`` are needed, see also :ref:`DESIGN MORTAR EDGE CONDITIONS 3D<designlinemortaredgeconditions3d>`, :ref:`DESIGN MORTAR CORNER CONDITIONS 2D/3D<designpointmortarcornerconditions2d/3d>`
 
.. _meshtying:

Mesh Tying
-----------

Different meshes can be connected with the `MORTAR COUPLING` definition. Two different application cases are envisioned:

- Incompatible meshes of two geometrical regions in one simulation are tied. This may be useful if a very coarse mesh shall be connected to a much finer region.

- In multiphysics simulations, two different meshes can be used for the different physical parts (e.g. temperature and structure, since high temperature gradients may occur in other regions than high highly stressed regions).

::

   --------------------------DESIGN LINE MORTAR COUPLING CONDITIONS 2D
   DLINE                           0
   //E num - 0 Master Inactive 
   --------------------------DESIGN SURF MORTAR COUPLING CONDITIONS 3D
   DSURF                           0
   //E num - 0 Master Inactive 
   --------------------DESIGN LINE MORTAR MULTI-COUPLING CONDITIONS 2D
   DLINE                           0
   //E num - 0 Master Inactive 
   --------------------DESIGN SURF MORTAR MULTI-COUPLING CONDITIONS 3D
   DSURF                           0
   //E num - 0 Master Inactive 

See the reference :ref:`DESIGN MORTAR COUPLING CONDITIONS 3D<designsurfmortarcouplingconditions3d>`, :ref:`DESIGN MORTAR COUPLING CONDITIONS 2D<designlinemortarcouplingconditions2d>`, :ref:`DESIGN MORTAR MULTI-COUPLING CONDITIONS 3D<designsurfmortarmulti-couplingconditions3d>`, :ref:`DESIGN MORTAR MULTI-COUPLING CONDITIONS 2D<designlinemortarmulti-couplingconditions2d>`
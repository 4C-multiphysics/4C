.. _`materialdevelopment`:

Development of material models
-------------------------------

For the definition of a material model, one has to put information at a number of places.
A detailed description of the methods, attributes, files, and directories where new code is needed, is given in the following.
The material model shall get a generic name ``myNewMaterial`` here.

.. note::

   The following sections are focused on the development of solid materials.
   Few information is also given for other material models (fluid, electrochemical, etc.),
   but some methods may be different. There might be separate sections for the different material classes in the future.

Input reader for the material model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main input read is located in ``src/inpar/inpar_validmaterials.cpp``.
Here, the input for the material including all mandatory and optional parameters, is defined.
Parameters that do not appear here, are not recognized.
A description and default values (for optional parameters) can (and should!) be defined here as well.

Definition of the material model as an enumeration item
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main purpose of the file ``src/inpar/inpar_material.H`` is to provide the Enum list ``MaterialType`` of all material laws existing in |FOURC| within the namespace ``INPAT::MAT``.
Thus, for each new material model, one has to add a new item in the enum list.
The name of the enum item starts with ``m_``, followed by the material name in lowercase.
It may be useful to add the physics in front, e.g. ``m_fluid_...`` or ``m_poro_...`` (this is not always the case, particularly not for solid mechanics).
For the example mentioned above the enum item would be

``m_mynewmaterial``

The material models are in general in alphabetical order, so, please, put the  new material model to the correct place.

Definition of the classes needed for the new material
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new file needs to be created: ``/src/mat/mat_<matname>.cpp`` (here: ``mat_mynewmaterial.cpp``, again all lowercase),
including the corresponding header file.

All material related classes are defined in this directory, within the namespace ``MAT``.

**Definition of a material parameter class**

Define the class for Material Parameters in the namespace ``Mat::PAR`` as

::

   class MyNewMaterial : public Core::Mat::PAR::Parameter

Here we need the methods

-	Constructor ``MyNewMaterial(Core::Mat::PAR::Parameter::Data matdata)``
-	``Teuchos::RCP<Core::Mat::Material> CreateMaterial()``

**Definition of the material type class**

Here the material type class is generated within the namespace ``MAT`` as

::

  class MyNewMaterialType : public Core::COMM::ParObjectType

This class is just to define the material itself, together with its respective parameter set.

This class does not even need a constructor, but only  a single private static variable:

::

   Mat::MyNewMaterial Mat::MyNewMaterial::instance_;


and a few public methods:

::

   Core::COMM::ParObject* Mat::MyNewMaterialType::Create(const std::vector<char>& data)
   {
     Mat::MyNewMaterial* mymaterial = new Mat::MyNewMaterial();
     mymaterial->unpack(data);
     return mymaterial;
   }
   std::string Name() const { return "MyNewMaterial"; }

   static MyNewMaterialType& instance() { return instance_; };


**Definition of the material class**

The main work happens in the material class in the ``Mat`` namespace, which is derived from the parent class.
This can be the class ``Material``, which is independent from the underlying physics,
but there are also classes for specific physical representations the material model happen.
The base classes of the material models representing a single physics environment, are given in the following table:

.. list-table::
   :header-rows: 1

   * - Physics
     - base class
   * - Structure (3D solid)
     - ``Mat::So3Material``
   * - Structure (Beams)
     - ``Mat::BeamMaterialTemplated``
   * - Thermo
     - ``Mat::ThermoMaterial``
   * - Fluid
     - ``Mat::Material``
   * - Electrochemistry
     - ``Mat::ElchSingleMat``
   * - Scalar transport
     - ``Mat::Material``

Note that not all existing materials use a physics-specific base class,
some materials are simply derived from ``Mat::Material`` even though a more specialized class exists.

It is probably a good idea to start with a clone of the material law given in the table above, and define a new class, e.g.:

::

   class MyNewMaterial : public So3Material

Here we definitely need a number of methods, which are called from other places,
and which have already a virtual representation in the parent class, e.g., here for solid materials:

-	Constructor
-	UniqueParObject()
-	pack()
-	unpack()
-	setup()   // -> initialize and allocate internal variables
-	update()    // -> update internal variables
-	evaluate()   // calculate stress and constitutive matrix
-	VisNames()  // For the names of variables to be visualized (optional; only if extra variables are to be visualized)
-	VisData()   // For the data of these variables (optional; only if extra variables are to be visualized)

Selection of the material model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The selection of the material model happens in the file ``mat/mat_material.cpp``.
Based on the enum ``Core::Materials::MaterialType``, a switch creates the material and provides the parameters.
Each case within the switch condition has a very similar layout, so for our material it looks like this:

::

    case Core::Materials::m_mynewmaterial:
    {
      if (curmat->Parameter() == nullptr)
        curmat->SetParameter(new Mat::PAR::MyNewMaterial(curmat));
      auto* params = static_cast<Mat::PAR::MyNewMaterial*>(curmat->Parameter());
      return params->CreateMaterial();
    }

Unit test of the material model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One should also write a unit test for the new material routine.
The respective source file should be located in ``/unittests/mat/unit_mynewmaterial.cpp``.
This file must also be included in the ``/unittests/mat/CMakeLists.txt`` file.

Remark on the dimensionality of the material model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All (solid) materials are defined for 3D elements. A reduction of the matrices is not used.
Instead, additional assumptions of the restrictions for plane strain and plane stress are used for the respective 2D elements.
The evaluation is then conducted in 3D. Finally, the stress and stiffness matrices are stored with the reduced component number.

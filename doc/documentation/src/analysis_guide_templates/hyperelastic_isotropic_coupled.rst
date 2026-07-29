.. _hyperelastic-isotropic-coupled:

Coupled isotropic hyperelastic summands
=======================================

These summands define compressible isotropic energies that depend on the full deformation. They
are referenced through the parameter ``MATIDS`` and generally do not require a separate
volumetric summand. The strain-energy formulation of each summand is documented in the Input
Parameter Reference.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Summand
     - Model
   * - :ref:`ELAST_CoupNeoHooke <MATERIALS_ELAST_CoupNeoHooke>`
     - Compressible Neo-Hookean elasticity.
   * - :ref:`ELAST_CoupLogNeoHooke <MATERIALS_ELAST_CoupLogNeoHooke>`
     - Logarithmic Neo-Hookean elasticity.
   * - :ref:`ELAST_CoupLogMixNeoHooke <MATERIALS_ELAST_CoupLogMixNeoHooke>`
     - Mixed logarithmic Neo-Hookean formulation.
   * - :ref:`ELAST_CoupMooneyRivlin <MATERIALS_ELAST_CoupMooneyRivlin>`
     - Compressible Mooney--Rivlin elasticity.
   * - :ref:`ELAST_CoupBlatzKo <MATERIALS_ELAST_CoupBlatzKo>`
     - Blatz--Ko compressible elasticity.
   * - :ref:`ELAST_CoupSimoPister <MATERIALS_ELAST_CoupSimoPister>`
     - Simo--Pister compressible elasticity.
   * - :ref:`ELAST_CoupSVK <MATERIALS_ELAST_CoupSVK>`
     - St. Venant--Kirchhoff energy as a summand.
   * - :ref:`ELAST_CoupExpPol <MATERIALS_ELAST_CoupExpPol>`
     - Exponential-polynomial energy.
   * - :ref:`ELAST_Coup1Pow <MATERIALS_ELAST_Coup1Pow>`
     - One-term power-law energy.
   * - :ref:`ELAST_Coup2Pow <MATERIALS_ELAST_Coup2Pow>`
     - Two-term power-law energy.
   * - :ref:`ELAST_Coup3Pow <MATERIALS_ELAST_Coup3Pow>`
     - Three-term power-law energy.
   * - :ref:`ELAST_Coup13aPow <MATERIALS_ELAST_Coup13aPow>`
     - Specialized coupled polynomial energy.
   * - :ref:`ELAST_CoupVarga <MATERIALS_ELAST_CoupVarga>`
     - Compressible Varga elasticity.

The linked Input Parameter Reference entries define the strain-energy formulation, coefficients,
and parameterization of each energy. A representative ``ELAST_CoupNeoHooke`` composition is shown
on :doc:`hyperelastic_framework`.
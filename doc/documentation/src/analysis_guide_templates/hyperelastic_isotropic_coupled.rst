.. _hyperelastic-isotropic-coupled:

Coupled isotropic hyperelastic summands
=======================================

These summands define compressible isotropic energies that depend on the full deformation. They
are referenced through ``MAT_ElastHyper.MATIDS`` and generally do not require a separate
volumetric summand.

.. list-table::
   :header-rows: 1
   :widths: 28 25 47

   * - Summand
     - Model
     - Implemented potential
   * - :ref:`ELAST_CoupNeoHooke <MATERIALS_ELAST_CoupNeoHooke>`
     - Compressible Neo-Hookean elasticity.
     - :math:`\Psi=c(I_1-3)+\frac{c}{\beta}(I_3^{-\beta}-1)`,
       :math:`c=\frac{E}{4(1+\nu)}`, :math:`\beta=\frac{\nu}{1-2\nu}`.
   * - :ref:`ELAST_CoupLogNeoHooke <MATERIALS_ELAST_CoupLogNeoHooke>`
     - Logarithmic Neo-Hookean elasticity.
     - :math:`\Psi=\frac{\mu}{2}(I_1-3)-\mu\ln J+
       \frac{\lambda}{2}(\ln J)^2`.
   * - :ref:`ELAST_CoupLogMixNeoHooke <MATERIALS_ELAST_CoupLogMixNeoHooke>`
     - Mixed logarithmic Neo-Hookean formulation.
     - :math:`\Psi=\frac{\mu}{2}(I_1-3)-\mu\ln J+
       \frac{\lambda}{2}(J-1)^2`.
   * - :ref:`ELAST_CoupMooneyRivlin <MATERIALS_ELAST_CoupMooneyRivlin>`
     - Compressible Mooney--Rivlin elasticity.
     - :math:`\Psi=c_1(I_1-3)+c_2(I_2-3)-(2c_1+4c_2)\ln J+c_3(J-1)^2`.
   * - :ref:`ELAST_CoupBlatzKo <MATERIALS_ELAST_CoupBlatzKo>`
     - Blatz--Ko compressible elasticity.
     - :math:`\Psi=\frac{\mu}{2}\left\{f\left[I_1-3+
       \frac{I_3^{-\beta}-1}{\beta}\right]+(1-f)\left[
       \frac{I_2}{I_3}-3+\frac{I_3^\beta-1}{\beta}\right]\right\}`,
       :math:`\beta=\frac{\nu}{1-2\nu}`.
   * - :ref:`ELAST_CoupSimoPister <MATERIALS_ELAST_CoupSimoPister>`
     - Simo--Pister compressible elasticity.
     - :math:`\Psi=\frac{\mu}{2}(I_1-3)-\mu\ln J`.
   * - :ref:`ELAST_CoupSVK <MATERIALS_ELAST_CoupSVK>`
     - St. Venant--Kirchhoff energy as a summand.
     - :math:`\Psi=\mu\,\operatorname{tr}(\boldsymbol E^2)+
       \frac{\lambda}{2}(\operatorname{tr}\boldsymbol E)^2`.
   * - :ref:`ELAST_CoupExpPol <MATERIALS_ELAST_CoupExpPol>`
     - Exponential-polynomial energy.
     - :math:`\Psi=a\exp[b(I_1-3)-(2b+c)\ln J+c(J-1)]-a`.
   * - :ref:`ELAST_Coup1Pow <MATERIALS_ELAST_Coup1Pow>`
     - One-term power-law energy.
     - :math:`\Psi=c(I_1-3)^d`.
   * - :ref:`ELAST_Coup2Pow <MATERIALS_ELAST_Coup2Pow>`
     - Two-term power-law energy.
     - :math:`\Psi=c(I_2-3)^d`.
   * - :ref:`ELAST_Coup3Pow <MATERIALS_ELAST_Coup3Pow>`
     - Three-term power-law energy.
     - :math:`\Psi=c(I_3^{1/3}-1)^d`.
   * - :ref:`ELAST_Coup13aPow <MATERIALS_ELAST_Coup13aPow>`
     - Specialized coupled polynomial energy.
     - :math:`\Psi=c(I_1I_3^{-a}-3)^d`.
   * - :ref:`ELAST_CoupVarga <MATERIALS_ELAST_CoupVarga>`
     - Compressible Varga elasticity.
     - :math:`\Psi=(2\mu-\beta)(\lambda_1+\lambda_2+\lambda_3-3)+
       \beta(\lambda_1^{-1}+\lambda_2^{-1}+\lambda_3^{-1}-3)`.

The linked Input Parameter Reference entries define the coefficients and parameterization of each
energy. A representative ``ELAST_CoupNeoHooke`` composition is shown on
:doc:`hyperelastic_framework`.

For ``ELAST_CoupBlatzKo``, the displayed potential is the one differentiated to compute stress and
tangent. Its current strain-energy output routine omits both terms containing :math:`\beta`.
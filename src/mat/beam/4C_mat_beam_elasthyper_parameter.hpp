// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_BEAM_ELASTHYPER_PARAMETER_HPP
#define FOUR_C_MAT_BEAM_ELASTHYPER_PARAMETER_HPP

#include "4C_config.hpp"

#include "4C_material_parameter_base.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  namespace PAR
  {
    // forward declaration
    class Material;


    /** \brief determine shear modulus which is either given directly or via Young's modulus and
     *         Poisson's ratio
     *
     */
    double determine_shear_modulus(const Core::Mat::PAR::Parameter::Data& matdata);

    /** \brief determine default value for interaction radius from area moment of inertia just in
     * case that no value was explicitly specified
     *
     */
    double determine_default_interaction_radius(const Core::Mat::PAR::Parameter::Data& matdata);

    /** \brief determine default value for interaction radius from area moment of inertia just in
     * case that no value was explicitly specified: isotropic case, i.e. only one moment of inertia
     *
     */
    double determine_default_interaction_radius_isotropic(
        const Core::Mat::PAR::Parameter::Data& matdata);

    /*-------------------------------------------------------------------------------------------*/
    /// (generic) constitutive parameters for a beam material based on hyperelastic stored energy
    /// function
    class BeamElastHyperMaterialParameterGeneric : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      BeamElastHyperMaterialParameterGeneric(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name accessors to 'modal' constitutive parameters
      //@{
      virtual double get_axial_rigidity() const = 0;

      virtual double get_shear_rigidity2() const = 0;

      virtual double get_shear_rigidity3() const = 0;


      virtual double get_torsional_rigidity() const = 0;

      virtual double get_bending_rigidity2() const = 0;

      virtual double get_bending_rigidity3() const = 0;


      virtual double get_translational_mass_inertia() const = 0;

      virtual double get_polar_mass_moment_of_inertia() const = 0;

      virtual double get_mass_moment_of_inertia2() const = 0;

      virtual double get_mass_moment_of_inertia3() const = 0;
      //@}

      virtual double get_interaction_radius() const = 0;


      virtual double get_yield_stress_n() const { return -1.0; };

      virtual double get_yield_stress_m() const { return -1.0; };


      virtual double get_hardening_axial_rigidity() const { return -1.0; };

      virtual double get_hardening_shear_rigidity2() const { return -1.0; };

      virtual double get_hardening_shear_rigidity3() const { return -1.0; };


      virtual double get_hardening_momental_rigidity() const { return -1.0; };

      virtual bool get_torsion_plasticity() const { return false; };

      bool uses_fad() { return use_fad_; };

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

     private:
      /// Flag to store if the material is used with a beam material and automatic differentiation
      bool use_fad_;
    };

    /*-------------------------------------------------------------------------------------------*/
    /// constitutive parameters for a Reissner beam formulation (hyperelastic stored energy
    /// function)
    class BeamReissnerElastHyperMaterialParams : public BeamElastHyperMaterialParameterGeneric
    {
     public:
      /// standard constructor
      BeamReissnerElastHyperMaterialParams(const Core::Mat::PAR::Parameter::Data& matdata);


      /// @name derived: accessors to 'modal' constitutive parameters
      //@{
      double get_axial_rigidity() const override { return youngs_modulus_ * cross_section_area_; }

      double get_shear_rigidity2() const override
      {
        return shear_modulus_ * cross_section_area_ * shear_correction_factor_;
      }

      double get_shear_rigidity3() const override
      {
        return shear_modulus_ * cross_section_area_ * shear_correction_factor_;
      }


      double get_torsional_rigidity() const override
      {
        return shear_modulus_ * area_moment_inertia_polar_;
      }

      double get_bending_rigidity2() const override
      {
        return youngs_modulus_ * area_moment_inertia_2_;
      }

      double get_bending_rigidity3() const override
      {
        return youngs_modulus_ * area_moment_inertia_3_;
      }


      double get_translational_mass_inertia() const override
      {
        return density_ * cross_section_area_;
      }


      double get_polar_mass_moment_of_inertia() const override
      {
        return density_ * (area_moment_inertia_2_ + area_moment_inertia_3_);
      }

      double get_mass_moment_of_inertia2() const override
      {
        return density_ * area_moment_inertia_2_;
      }

      double get_mass_moment_of_inertia3() const override
      {
        return density_ * area_moment_inertia_3_;
      }


      double get_interaction_radius() const override
      {
        if (radius_interaction_ == -1.0)
          FOUR_C_THROW(
              "the radius of a beam which is to be used for interactions (contact, potentials, "
              "viscous drag in background fluid ...) has not been specified in the material "
              "definition!");

        return radius_interaction_;
      }
      //@}
     protected:
      double get_cross_section_area() const { return cross_section_area_; }
      double get_shear_correction_factor() const { return shear_correction_factor_; }
      double get_shear_modulus() const { return shear_modulus_; }
      double get_moment_inertia2() const { return area_moment_inertia_2_; }
      double get_moment_inertia3() const { return area_moment_inertia_3_; }
      double get_moment_inertia_polar() const { return area_moment_inertia_polar_; }
      double get_youngs_modulus() const { return youngs_modulus_; }

     private:
      /// @name constitutive parameters
      //@{

      /// Young's modulus
      const double youngs_modulus_;
      /// shear modulus
      const double shear_modulus_;
      /// mass density
      const double density_;

      /// cross-section area
      const double cross_section_area_;
      /// shear correction factor
      const double shear_correction_factor_;

      /// polar/axial area moment of inertia
      const double area_moment_inertia_polar_;
      /// area moment of inertia w.r.t. first principal axis of inertia (i.e. second base vector)
      const double area_moment_inertia_2_;
      /// area moment of inertia w.r.t. second principal axis of inertia (i.e. third base vector)
      const double area_moment_inertia_3_;

      /// radius of a circular cross-section which is ONLY used to evaluate interactions
      //  such as contact, potentials, viscous drag in background fluid ...
      //  it is an optional input parameter and defaults to -1.0 if not required (no interactions at
      //  all) of course, this should be generalized as soon as we allow for other cross-section
      //  shapes in our models for beam-to-X interactions
      const double radius_interaction_;
      //@}
    };

    /*-------------------------------------------------------------------------------------------*/
    /// constitutive parameters for a Reissner beam formulation (hyperelastic stored energy
    /// function),
    // specified individually 'by mode', i.e. axial tension, torsion, bending (2x)
    class BeamReissnerElastHyperMaterialParamsByMode : public BeamElastHyperMaterialParameterGeneric
    {
     public:
      /// standard constructor
      BeamReissnerElastHyperMaterialParamsByMode(const Core::Mat::PAR::Parameter::Data& matdata);


      /// @name derived: accessors to 'modal' constitutive parameters
      //@{
      double get_axial_rigidity() const override { return axial_rigidity_; }

      double get_shear_rigidity2() const override { return shear_rigidity_2_; }

      double get_shear_rigidity3() const override { return shear_rigidity_3_; }


      double get_torsional_rigidity() const override { return torsional_rigidity_; }

      double get_bending_rigidity2() const override { return bending_rigidity_2_; }

      double get_bending_rigidity3() const override { return bending_rigidity_3_; }


      double get_translational_mass_inertia() const override { return translational_mass_inertia_; }


      double get_polar_mass_moment_of_inertia() const override
      {
        return mass_moment_inertia_polar_;
      }

      double get_mass_moment_of_inertia2() const override { return mass_moment_inertia_2_; }

      double get_mass_moment_of_inertia3() const override { return mass_moment_inertia_3_; }

      double get_interaction_radius() const override
      {
        if (radius_interaction_ == -1.0)
          FOUR_C_THROW(
              "the radius of a beam which is to be used for interactions (contact, potentials, "
              "viscous drag in background fluid ...) has not been specified in the material "
              "definition!");

        return radius_interaction_;
      }
      //@}

     private:
      /// @name constitutive parameters
      //@{

      /// axial rigidity
      const double axial_rigidity_;
      /// shear rigidity w.r.t first principal axis of inertia (i.e. second base vector)
      const double shear_rigidity_2_;
      /// shear rigidity w.r.t second principal axis of inertia (i.e. third base vector)
      const double shear_rigidity_3_;

      /// torsional rigidity
      const double torsional_rigidity_;
      /// flexural/bending rigidity w.r.t. first principal axis of inertia (i.e. second base vector)
      const double bending_rigidity_2_;
      /// flexural/bending rigidity w.r.t. second principal axis of inertia (i.e. third base vector)
      const double bending_rigidity_3_;

      /// translational mass inertia: mass density * cross-section area
      const double translational_mass_inertia_;

      /// polar mass moment of inertia, i.e. w.r.t. rotation around beam axis
      const double mass_moment_inertia_polar_;
      /// mass moment of inertia w.r.t. first principal axis of inertia (i.e. second base vector)
      const double mass_moment_inertia_2_;
      /// mass moment of inertia w.r.t. second principal axis of inertia (i.e. third base vector)
      const double mass_moment_inertia_3_;

      /// radius of a circular cross-section which is ONLY used to evaluate interactions
      //  such as contact, potentials, viscous drag in background fluid ...
      //  it is an optional input parameter and defaults to -1.0 if not required (no interactions at
      //  all) of course, this should be generalized as soon as we allow for other cross-section
      //  shapes in our models for beam-to-X interactions
      const double radius_interaction_;
      //@}
    };



    /*-------------------------------------------------------------------------------------------*/
    /// constitutive parameters for a Kirchhoff beam formulation (hyperelastic stored energy
    /// function)
    class BeamKirchhoffElastHyperMaterialParams : public BeamElastHyperMaterialParameterGeneric
    {
     public:
      /// standard constructor
      BeamKirchhoffElastHyperMaterialParams(const Core::Mat::PAR::Parameter::Data& matdata);


      /// @name derived: accessors to 'modal' constitutive parameters
      //@{
      double get_axial_rigidity() const override { return youngs_modulus_ * cross_section_area_; }

      double get_shear_rigidity2() const override { return 0.0; }

      double get_shear_rigidity3() const override { return 0.0; }


      double get_torsional_rigidity() const override
      {
        return shear_modulus_ * area_moment_inertia_polar_;
      }

      double get_bending_rigidity2() const override
      {
        return youngs_modulus_ * area_moment_inertia_2_;
      }

      double get_bending_rigidity3() const override
      {
        return youngs_modulus_ * area_moment_inertia_3_;
      }


      double get_translational_mass_inertia() const override
      {
        return density_ * cross_section_area_;
      }


      double get_polar_mass_moment_of_inertia() const override
      {
        return density_ * (area_moment_inertia_2_ + area_moment_inertia_3_);
      }

      double get_mass_moment_of_inertia2() const override
      {
        return density_ * area_moment_inertia_2_;
      }

      double get_mass_moment_of_inertia3() const override
      {
        return density_ * area_moment_inertia_3_;
      }


      double get_interaction_radius() const override
      {
        if (radius_interaction_ == -1.0)
          FOUR_C_THROW(
              "the radius of a beam which is to be used for interactions (contact, potentials, "
              "viscous drag in background fluid ...) has not been specified in the material "
              "definition!");

        return radius_interaction_;
      }
      //@}

     private:
      /// @name constitutive parameters
      //@{

      /// Young's modulus
      const double youngs_modulus_;
      /// shear modulus
      const double shear_modulus_;
      /// mass density
      const double density_;

      /// cross-section area
      const double cross_section_area_;

      /// polar/axial area moment of inertia
      const double area_moment_inertia_polar_;
      /// area moment of inertia w.r.t. first principal axis of inertia (i.e. second base vector)
      const double area_moment_inertia_2_;
      /// area moment of inertia w.r.t. second principal axis of inertia (i.e. third base vector)
      const double area_moment_inertia_3_;

      /// radius of a circular cross-section which is ONLY used to evaluate interactions
      //  such as contact, potentials, viscous drag in background fluid ...
      //  it is an optional input parameter and defaults to -1.0 if not required (no interactions at
      //  all) of course, this should be generalized as soon as we allow for other cross-section
      //  shapes in our models for beam-to-X interactions
      const double radius_interaction_;
      //@}
    };

    /*-------------------------------------------------------------------------------------------*/
    /// constitutive parameters for a Kirchhoff beam formulation (hyperelastic stored energy
    /// function),
    // specified individually 'by mode', i.e. axial tension, torsion, bending (2x)
    class BeamKirchhoffElastHyperMaterialParamsByMode
        : public BeamElastHyperMaterialParameterGeneric
    {
     public:
      /// standard constructor
      BeamKirchhoffElastHyperMaterialParamsByMode(const Core::Mat::PAR::Parameter::Data& matdata);


      /// @name derived: accessors to 'modal' constitutive parameters
      //@{
      double get_axial_rigidity() const override { return axial_rigidity_; }

      double get_shear_rigidity2() const override { return 0.0; }

      double get_shear_rigidity3() const override { return 0.0; }


      double get_torsional_rigidity() const override { return torsional_rigidity_; }

      double get_bending_rigidity2() const override { return bending_rigidity_2_; }

      double get_bending_rigidity3() const override { return bending_rigidity_3_; }


      double get_translational_mass_inertia() const override { return translational_mass_inertia_; }


      double get_polar_mass_moment_of_inertia() const override
      {
        return mass_moment_inertia_polar_;
      }

      double get_mass_moment_of_inertia2() const override { return mass_moment_inertia_2_; }

      double get_mass_moment_of_inertia3() const override { return mass_moment_inertia_3_; }

      double get_interaction_radius() const override
      {
        if (radius_interaction_ == -1.0)
          FOUR_C_THROW(
              "the radius of a beam which is to be used for interactions (contact, potentials, "
              "viscous drag in background fluid ...) has not been specified in the material "
              "definition!");

        return radius_interaction_;
      }
      //@}

     private:
      /// @name constitutive parameters
      //@{

      /// axial rigidity
      const double axial_rigidity_;

      /// torsional rigidity
      const double torsional_rigidity_;
      /// flexural/bending rigidity w.r.t. first principal axis of inertia (i.e. second base vector)
      const double bending_rigidity_2_;
      /// flexural/bending rigidity w.r.t. second principal axis of inertia (i.e. third base vector)
      const double bending_rigidity_3_;

      /// translational mass inertia: mass density * cross-section area
      const double translational_mass_inertia_;

      /// polar mass moment of inertia, i.e. w.r.t. rotation around beam axis
      const double mass_moment_inertia_polar_;
      /// mass moment of inertia w.r.t. first principal axis of inertia (i.e. second base vector)
      const double mass_moment_inertia_2_;
      /// mass moment of inertia w.r.t. second principal axis of inertia (i.e. third base vector)
      const double mass_moment_inertia_3_;

      /// radius of a circular cross-section which is ONLY used to evaluate interactions
      //  such as contact, potentials, viscous drag in background fluid ...
      //  it is an optional input parameter and defaults to -1.0 if not required (no interactions at
      //  all) of course, this should be generalized as soon as we allow for other cross-section
      //  shapes in our models for beam-to-X interactions
      const double radius_interaction_;
      //@}
    };



    /*-------------------------------------------------------------------------------------------*/
    /// constitutive parameters for a torsion-free, isotropic Kirchhoff beam formulation
    // (hyperelastic stored energy function)
    class BeamKirchhoffTorsionFreeElastHyperMaterialParams
        : public BeamElastHyperMaterialParameterGeneric
    {
     public:
      /// standard constructor
      BeamKirchhoffTorsionFreeElastHyperMaterialParams(
          const Core::Mat::PAR::Parameter::Data& matdata);


      /// @name derived: accessors to 'modal' constitutive parameters
      //@{
      double get_axial_rigidity() const override { return youngs_modulus_ * cross_section_area_; }

      double get_shear_rigidity2() const override { return 0.0; }

      double get_shear_rigidity3() const override { return 0.0; }


      double get_torsional_rigidity() const override { return 0.0; }

      double get_bending_rigidity2() const override
      {
        return youngs_modulus_ * area_moment_inertia_;
      }

      double get_bending_rigidity3() const override
      {
        return youngs_modulus_ * area_moment_inertia_;
      }


      double get_translational_mass_inertia() const override
      {
        return density_ * cross_section_area_;
      }


      double get_polar_mass_moment_of_inertia() const override { return 0.0; }

      double get_mass_moment_of_inertia2() const override { return 0.0; }

      double get_mass_moment_of_inertia3() const override { return 0.0; }


      double get_interaction_radius() const override
      {
        if (radius_interaction_ == -1.0)
          FOUR_C_THROW(
              "the radius of a beam which is to be used for interactions (contact, potentials, "
              "viscous drag in background fluid ...) has not been specified in the material "
              "definition!");

        return radius_interaction_;
      }
      //@}

     private:
      /// @name constitutive parameters
      //@{

      /// Young's modulus
      const double youngs_modulus_;
      /// mass density
      const double density_;

      /// cross-section area
      const double cross_section_area_;
      /// area moment of inertia
      const double area_moment_inertia_;

      /// radius of a circular cross-section which is ONLY used to evaluate interactions
      //  such as contact, potentials, viscous drag in background fluid ...
      //  it is an optional input parameter and defaults to -1.0 if not required (no interactions at
      //  all) of course, this should be generalized as soon as we allow for other cross-section
      //  shapes in our models for beam-to-X interactions
      const double radius_interaction_;
      //@}
    };

    /*-------------------------------------------------------------------------------------------*/
    /// constitutive parameters for a torsion-free, isotropic Kirchhoff beam formulation
    // (hyperelastic stored energy function),
    // specified individually 'by mode', i.e. axial tension and bending
    class BeamKirchhoffTorsionFreeElastHyperMaterialParamsByMode
        : public BeamElastHyperMaterialParameterGeneric
    {
     public:
      /// standard constructor
      BeamKirchhoffTorsionFreeElastHyperMaterialParamsByMode(
          const Core::Mat::PAR::Parameter::Data& matdata);


      /// @name derived: accessors to 'modal' constitutive parameters
      //@{
      double get_axial_rigidity() const override { return axial_rigidity_; }

      double get_shear_rigidity2() const override { return 0.0; }

      double get_shear_rigidity3() const override { return 0.0; }


      double get_torsional_rigidity() const override { return 0.0; }

      double get_bending_rigidity2() const override { return bending_rigidity_; }

      double get_bending_rigidity3() const override { return bending_rigidity_; }


      double get_translational_mass_inertia() const override { return translational_mass_inertia_; }


      double get_polar_mass_moment_of_inertia() const override { return 0.0; }

      double get_mass_moment_of_inertia2() const override { return 0.0; }

      double get_mass_moment_of_inertia3() const override { return 0.0; }


      double get_interaction_radius() const override
      {
        if (radius_interaction_ == -1.0)
          FOUR_C_THROW(
              "the radius of a beam which is to be used for interactions (contact, potentials, "
              "viscous drag in background fluid ...) has not been specified in the material "
              "definition!");

        return radius_interaction_;
      }
      //@}

     private:
      /// @name constitutive parameters
      //@{

      /// axial rigidity
      const double axial_rigidity_;
      /// flexural/bending rigidity
      const double bending_rigidity_;

      /// translational mass inertia: mass density * cross-section area
      const double translational_mass_inertia_;

      /// radius of a circular cross-section which is ONLY used to evaluate interactions
      //  such as contact, potentials, viscous drag in background fluid ...
      //  it is an optional input parameter and defaults to -1.0 if not required (no interactions at
      //  all) of course, this should be generalized as soon as we allow for other cross-section
      //  shapes in our models for beam-to-X interactions
      const double radius_interaction_;
      //@}
    };


  }  // namespace PAR

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

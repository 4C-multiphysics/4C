// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_FLUIDPORO_RELPERMEABILITY_LAW_HPP
#define FOUR_C_MAT_FLUIDPORO_RELPERMEABILITY_LAW_HPP

#include "4C_config.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    //! interface class for generic relative permeability law
    class FluidPoroRelPermeabilityLaw : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      explicit FluidPoroRelPermeabilityLaw(
          const Core::Mat::PAR::Parameter::Data& matdata, bool constrelpermeability)
          : Parameter(matdata), constrelpermeability_(constrelpermeability) {};

      // get relative permeability
      virtual double get_rel_permeability(const double saturation) const = 0;

      // get derivative of relative permeability with respect to the saturation of this phase
      virtual double get_deriv_of_rel_permeability_wrt_saturation(
          const double saturation) const = 0;

      // check for constant permeability
      bool has_constant_rel_permeability() const { return constrelpermeability_; }

      /// factory method
      static Mat::PAR::FluidPoroRelPermeabilityLaw* create_rel_permeability_law(int matID);

     private:
      const bool constrelpermeability_;
    };

    //! class for constant relative permeability law
    class FluidPoroRelPermeabilityLawConstant : public FluidPoroRelPermeabilityLaw
    {
     public:
      /// standard constructor
      explicit FluidPoroRelPermeabilityLawConstant(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; };

      // get permeability
      double get_rel_permeability(const double saturation) const override
      {
        return relpermeability_;
      };

      // get derivative of relative permeability with respect to the saturation of this phase
      double get_deriv_of_rel_permeability_wrt_saturation(const double saturation) const override
      {
        return 0.0;
      };

     private:
      /// @name material parameters
      //@{
      /// permeability (constant in this case)
      const double relpermeability_;
      //@}
    };

    //! class for varying relative permeability
    // relative permeabilty of phase i is calculated via saturation_i^exponent as in
    // G. Sciume, William G. Gray, F. Hussain, M. Ferrari, P. Decuzzi, and B. A. Schrefler.
    // Three phase flow dynamics in tumor growth. Computational Mechanics, 53:465-484, 2014
    class FluidPoroRelPermeabilityLawExponent : public FluidPoroRelPermeabilityLaw
    {
     public:
      /// standard constructor
      explicit FluidPoroRelPermeabilityLawExponent(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; };

      // get permeability
      double get_rel_permeability(const double saturation) const override
      {
        if (saturation > minsat_)
          return std::pow(saturation, exp_);
        else
          return std::pow(minsat_, exp_);
      };

      // get derivative of relative permeability with respect to the saturation of this phase
      double get_deriv_of_rel_permeability_wrt_saturation(const double saturation) const override
      {
        if (saturation > minsat_)
          return exp_ * std::pow(saturation, (exp_ - 1.0));
        else
          return 0.0;
      };

     private:
      /// @name material parameters
      //@{
      /// exponent
      const double exp_;
      /// minimal saturation used for computation -> this variable can be set by the user to avoid
      /// very small values for the relative permeability
      const double minsat_;
      //@}
    };


  }  // namespace PAR
}  // namespace Mat



FOUR_C_NAMESPACE_CLOSE

#endif

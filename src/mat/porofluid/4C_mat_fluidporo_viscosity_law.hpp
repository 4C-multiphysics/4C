// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_FLUIDPORO_VISCOSITY_LAW_HPP
#define FOUR_C_MAT_FLUIDPORO_VISCOSITY_LAW_HPP

#include "4C_config.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    //! interface class for generic viscosity law
    class FluidPoroViscosityLaw : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      explicit FluidPoroViscosityLaw(
          const Core::Mat::PAR::Parameter::Data& matdata, bool constviscosity)
          : Parameter(matdata), constviscosity_(constviscosity) {};

      // get viscosity
      virtual double get_viscosity(const double abspressgrad) const = 0;

      // get derivative of viscosity wrt |grad(p)|
      virtual double get_deriv_of_viscosity_wrt_abs_press_grad(const double abspressgrad) const = 0;

      // check for constant viscosity
      bool has_constant_viscosity() const { return constviscosity_; }

      /// factory method
      static Mat::PAR::FluidPoroViscosityLaw* create_viscosity_law(int matID);

     private:
      const bool constviscosity_;
    };

    //! class for constant viscosity law
    class FluidPoroViscosityLawConstant : public FluidPoroViscosityLaw
    {
     public:
      /// standard constructor
      explicit FluidPoroViscosityLawConstant(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; };

      // get viscosity
      double get_viscosity(const double abspressgrad) const override { return viscosity_; };

      // get derivative of viscosity wrt |grad(p)|  --> 0 in case of const. viscosity
      double get_deriv_of_viscosity_wrt_abs_press_grad(const double abspressgrad) const override
      {
        return 0.0;
      };

     private:
      /// @name material parameters
      //@{
      /// viscosity (constant in this case)
      const double viscosity_;
      //@}
    };

    //! class for viscosity law modelling cell adherence from
    // G. Sciume, William G. Gray, F. Hussain, M. Ferrari, P. Decuzzi, and B. A. Schrefler.
    // Three phase flow dynamics in tumor growth. Computational Mechanics, 53:465-484, 2014.
    // visc = visc0 / ((1 - xi)*(1 - psi / | grad(pressure) |) * heaviside(1 - psi / |
    // grad(pressure) |) + xi)
    class FluidPoroViscosityLawCellAdherence : public FluidPoroViscosityLaw
    {
     public:
      /// standard constructor
      explicit FluidPoroViscosityLawCellAdherence(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; };

      // get viscosity
      double get_viscosity(const double abspressgrad) const override;

      // get derivative of viscosity wrt |grad(p)|
      double get_deriv_of_viscosity_wrt_abs_press_grad(const double abspressgrad) const override;

     private:
      /// @name material parameters
      //@{
      /// viscosity 0
      const double visc0_;
      /// xi
      const double xi_;
      /// psi
      const double psi_;
      //@}
    };


  }  // namespace PAR
}  // namespace Mat



FOUR_C_NAMESPACE_CLOSE

#endif

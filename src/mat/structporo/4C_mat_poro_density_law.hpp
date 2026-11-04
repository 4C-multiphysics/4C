// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_PORO_DENSITY_LAW_HPP
#define FOUR_C_MAT_PORO_DENSITY_LAW_HPP

#include "4C_config.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    //! interface class for generic density law
    class PoroDensityLaw : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      explicit PoroDensityLaw(const Core::Mat::PAR::Parameter::Data& matdata)
          : Parameter(matdata) {};

      //! compute derivative of density w.r.t. pressure
      virtual double compute_cur_density_derivative(
          const double& refdensity,  ///< (i) initial/reference density at gauss point
          const double& press        ///< (i) pressure at gauss point
          ) = 0;

      /// compute current density
      virtual double compute_cur_density(
          const double& refdensity,  ///< (i) initial/reference density at gauss point
          const double& press        ///< (i) pressure at gauss point
          ) = 0;

      /// compute relation of reference density to current density
      virtual double compute_ref_density_to_cur_density(
          const double& press  ///< (i) pressure at gauss point
          ) = 0;

      /// compute derivative of relation of reference density to current density w.r.t. pressure
      virtual double compute_ref_density_to_cur_density_derivative(
          const double& press  ///< (i) pressure at gauss point
          ) = 0;
      /// compute second derivative of relation of reference density to current density w.r.t.
      /// pressure
      virtual double compute_ref_density_to_cur_density_second_derivative(
          const double& press  ///< (i) pressure at gauss point
          ) = 0;

      /// return inverse bulkmodulus (=compressibility)
      virtual double inv_bulkmodulus() const = 0;

      /// factory method
      static Mat::PAR::PoroDensityLaw* create_density_law(int matID);
    };

    //! class for constant density law
    class PoroDensityLawConstant : public PoroDensityLaw
    {
     public:
      /// standard constructor
      explicit PoroDensityLawConstant(const Core::Mat::PAR::Parameter::Data& matdata)
          : PoroDensityLaw(matdata) {};

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; };

      //! compute derivative of density w.r.t. pressure
      double compute_cur_density_derivative(
          const double& refdensity,  ///< (i) initial/reference density at gauss point
          const double& press        ///< (i) pressure at gauss point
          ) override
      {
        return refdensity;
      };

      /// compute current density
      double compute_cur_density(
          const double& refdensity,  ///< (i) initial/reference density at gauss point
          const double& press        ///< (i) pressure at gauss point
          ) override
      {
        return 0.0;
      };

      /// compute relation of reference density to current density
      double compute_ref_density_to_cur_density(
          const double& press  ///< (i) pressure at gauss point
          ) override
      {
        return 1.0;
      };

      /// compute derivative of relation of reference density to current density w.r.t. pressure
      double compute_ref_density_to_cur_density_derivative(
          const double& press  ///< (i) pressure at gauss point
          ) override
      {
        return 0.0;
      };
      /// compute second derivative of relation of reference density to current density w.r.t.
      /// pressure
      double compute_ref_density_to_cur_density_second_derivative(
          const double& press  ///< (i) pressure at gauss point
          ) override
      {
        return 0.0;
      };

      /// return inverse bulkmodulus (=compressibility)
      double inv_bulkmodulus() const override { return 0.0; };
    };

    //! class for exponential density law
    class PoroDensityLawExp : public PoroDensityLaw
    {
     public:
      /// standard constructor
      explicit PoroDensityLawExp(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! compute derivative of density w.r.t. pressure
      double compute_cur_density_derivative(
          const double& refdensity,  ///< (i) initial/reference density at gauss point
          const double& press        ///< (i) pressure at gauss point
          ) override;

      /// compute current density
      double compute_cur_density(
          const double& refdensity,  ///< (i) initial/reference density at gauss point
          const double& press        ///< (i) pressure at gauss point
          ) override;

      /// compute relation of reference density to current density
      double compute_ref_density_to_cur_density(
          const double& press  ///< (i) pressure at gauss point
          ) override;

      /// compute derivative ofrelation of reference density to current density w.r.t. pressure
      double compute_ref_density_to_cur_density_derivative(
          const double& press  ///< (i) pressure at gauss point
          ) override;
      /// compute second derivative of relation of reference density to current density w.r.t.
      /// pressure
      double compute_ref_density_to_cur_density_second_derivative(
          const double& press  ///< (i) pressure at gauss point
          ) override;

      /// return inverse bulkmodulus (=compressibility)
      double inv_bulkmodulus() const override { return 1.0 / bulkmodulus_; };

     private:
      /// @name material parameters
      //@{
      /// bulk modulus
      double bulkmodulus_;
      //@}
    };

  }  // namespace PAR
}  // namespace Mat



FOUR_C_NAMESPACE_CLOSE

#endif

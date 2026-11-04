// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_LUBRICATION_LAW_HPP
#define FOUR_C_MAT_LUBRICATION_LAW_HPP


#include "4C_config.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    //! interface class for generic lubrication law
    class LubricationLaw : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      explicit LubricationLaw(const Core::Mat::PAR::Parameter::Data& matdata);

      /// compute current viscosity and save it
      virtual void compute_viscosity(
          const double& press,  ///< (i) lubrication pressure at gauss point
          double& viscosity     ///< (o) viscosity at gauss point
          ) = 0;


      //! evaluate constitutive relation for viscosity and compute derivatives
      virtual void constitutive_derivatives(
          const double& press,      ///< (i) lubrication pressure at gauss point
          const double& viscosity,  ///< (i) viscosity at gauss point
          double& dviscosity_dp     ///< (o) derivative of viscosity w.r.t. pressure
          ) = 0;
    };

    /*----------------------------------------------------------------------*/
    //! constant lubrication law
    class LubricationLawConstant : public LubricationLaw
    {
     public:
      /// standard constructor
      explicit LubricationLawConstant(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// compute current viscosity and save it
      void compute_viscosity(const double& press,  ///< (i) lubrication pressure at gauss point
          double& viscosity                        ///< (o) viscosity at gauss point
          ) override;

      //! evaluate constitutive relation for viscosity and compute derivatives
      void constitutive_derivatives(
          const double& press,      ///< (i) lubrication pressure at gauss point
          const double& viscosity,  ///< (i) viscosity at gauss point
          double& dviscosity_dp     ///< (o) derivative of potential w.r.t. pressure
          ) override;


      /// constant viscosity
      double viscosity_;

    };  // class PoroLawLinear

    /*-------------------------------------------------------------------*/
    //! Lubrication law regarding Barus viscosity
    class LubricationLawBarus : public LubricationLaw
    {
     public:
      /// standard constructor
      explicit LubricationLawBarus(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;


      /// compute current viscosity and save it
      void compute_viscosity(const double& press,  ///< (i) lubrication pressure at gauss point
          double& viscosity                        ///< (o) viscosity at gauss point
          ) override;

      //! evaluate constitutive relation for viscosity and compute derivatives
      void constitutive_derivatives(
          const double& press,      ///< (i) lubrication pressure at gauss point
          const double& viscosity,  ///< (i) viscosity at gauss point
          double& dviscosity_dp     ///< (o) derivative of potential w.r.t. pressure
          ) override;



      /// material parameters for the calculation of the viscosity
      /// Absolute viscosity value in Barus formulation
      double ABSViscosity_;
      /// Press_vis coefficient value in Barus formulation
      double PreVisCoeff_;
    };

    /*-------------------------------------------------------------------*/
    //! Lubrication law regarding Roeland viscosity
    class LubricationLawRoeland : public LubricationLaw
    {
     public:
      /// standard constructor
      explicit LubricationLawRoeland(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;


      /// compute current viscosity and save it
      void compute_viscosity(const double& press,  ///< (i) lubrication pressure at gauss point
          double& viscosity                        ///< (o) viscosity at gauss point
          ) override;

      //! evaluate constitutive relation for viscosity and compute derivatives
      void constitutive_derivatives(
          const double& press,      ///< (i) lubrication pressure at gauss point
          const double& viscosity,  ///< (i) viscosity at gauss point
          double& dviscosity_dp     ///< (o) derivative of potential w.r.t. pressure
          ) override;



      /// material parameters for the calculation of the viscosity
      /// Absolute viscosity value in Roeland formulation
      double ABSViscosity_;
      /// Press_vis coefficient value in Roeland formulation
      double PreVisCoeff_;

      // constants with units for the calculation of the viscosity
      /// Reference viscosity in Roeland formulation
      double RefVisc_;
      /// Reference pressure in Roeland formulation
      double RefPress_;
      /// The analytical term to corrolate Barus and Roeland formulation
      double z_;
    };


  }  // namespace PAR
}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif

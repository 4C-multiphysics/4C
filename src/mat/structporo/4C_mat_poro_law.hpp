// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_PORO_LAW_HPP
#define FOUR_C_MAT_PORO_LAW_HPP

#include "4C_config.hpp"

#include "4C_material_parameter_base.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    //! interface class for generic porosity law
    class PoroLaw : public Core::Mat::PAR::Parameter
    {
     public:
      //! standard constructor
      explicit PoroLaw(const Core::Mat::PAR::Parameter::Data& matdata);

      //! evaluate constitutive relation for porosity and compute derivatives
      virtual void constitutive_derivatives(
          const Teuchos::ParameterList& params,  //!< (i) parameter list
          const double& press,                   //!< (i) fluid pressure at gauss point
          const double& J,                       //!< (i) Jacobian determinant at gauss point
          const double& porosity,                //!< (i) porosity at gauss point
          const double& refporosity,             //!< (i) porosity at gauss point
          double* dW_dp,                         //!< (o) derivative of potential w.r.t. pressure
          double* dW_dphi,                       //!< (o) derivative of potential w.r.t. porosity
          double* dW_dJ,                         //!< (o) derivative of potential w.r.t. jacobian
          double* dW_dphiref,  //!< (o) derivative of potential w.r.t. reference porosity
          double* W            //!< (o) inner potential
          ) = 0;

      //! compute current porosity and save it
      virtual void compute_porosity(
          const double& refporosity,  //!< (i) initial/reference porosity at gauss point
          const double& press,        //!< (i) pressure at gauss point
          const double& J,            //!< (i) determinant of jacobian at gauss point
          const int& gp,              //!< (i) number of current gauss point
          double& porosity,           //!< (o) porosity at gauss point
          double* dphi_dp,    //!< (o) first derivative of porosity w.r.t. pressure at gauss point
          double* dphi_dJ,    //!< (o) first derivative of porosity w.r.t. jacobian at gauss point
          double* dphi_dJdp,  //!< (o) derivative of porosity w.r.t. pressure and jacobian at gauss
                              //!< point
          double* dphi_dJJ,   //!< (o) second derivative of porosity w.r.t. jacobian at gauss point
          double* dphi_dpp,   //!< (o) second derivative of porosity w.r.t. pressure at gauss point
          double* dphi_dphiref  //!< (o) derivative of porosity w.r.t. reference porosity (only
                                //!< nonzero with reaction)
          ) = 0;

      //! return inverse bulkmodulus (=compressibility)
      virtual double inv_bulk_modulus() const = 0;
    };

    /*----------------------------------------------------------------------*/
    //! linear porosity law
    class PoroLawLinear : public PoroLaw
    {
     public:
      //! standard constructor
      explicit PoroLawLinear(const Core::Mat::PAR::Parameter::Data& matdata);

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;


      //! evaluate constitutive relation for porosity and compute derivatives using reference
      //! porosity
      void constitutive_derivatives(const Teuchos::ParameterList& params,  //!< (i) parameter list
          const double& press,        //!< (i) fluid pressure at gauss point
          const double& J,            //!< (i) Jacobian determinant at gauss point
          const double& porosity,     //!< (i) porosity at gauss point
          const double& refporosity,  //!< (i) porosity at gauss point
          double* dW_dp,              //!< (o) derivative of potential w.r.t. pressure
          double* dW_dphi,            //!< (o) derivative of potential w.r.t. porosity
          double* dW_dJ,              //!< (o) derivative of potential w.r.t. jacobian
          double* dW_dphiref,         //!< (o) derivative of potential w.r.t. reference porosity
          double* W                   //!< (o) inner potential
          ) override;

      //! compute current porosity and save it
      void compute_porosity(
          const double& refporosity,  //!< (i) initial/reference porosity at gauss point
          const double& press,        //!< (i) pressure at gauss point
          const double& J,            //!< (i) determinant of jacobian at gauss point
          const int& gp,              //!< (i) number of current gauss point
          double& porosity,           //!< (o) porosity at gauss point
          double* dphi_dp,    //!< (o) first derivative of porosity w.r.t. pressure at gauss point
          double* dphi_dJ,    //!< (o) first derivative of porosity w.r.t. jacobian at gauss point
          double* dphi_dJdp,  //!< (o) derivative of porosity w.r.t. pressure and jacobian at gauss
                              //!< point
          double* dphi_dJJ,   //!< (o) second derivative of porosity w.r.t. jacobian at gauss point
          double* dphi_dpp,   //!< (o) second derivative of porosity w.r.t. pressure at gauss point
          double* dphi_dphiref  //!< (o) derivative of porosity w.r.t. reference porosity (only
                                //!< nonzero with reaction)
          ) override;

      //! return inverse bulkmodulus (=compressibility)
      double inv_bulk_modulus() const override { return 1.0 / bulk_modulus_; }

     private:
      //! @name material parameters
      //!@{
      //! bulk modulus of skeleton phase
      double bulk_modulus_;
      //!@}
    };

    /*----------------------------------------------------------------------*/
    //! Neo-Hookean like porosity law
    // see   A.-T. Vuong, L. Yoshihara, W.A. Wall :A general approach for modeling interacting flow
    // through porous media under finite deformations, Comput. Methods Appl. Mech. Engrg. 283 (2015)
    // 1240-1259, equation (39)

    class PoroLawNeoHooke : public PoroLaw
    {
     public:
      //! standard constructor
      explicit PoroLawNeoHooke(const Core::Mat::PAR::Parameter::Data& matdata);

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! evaluate constitutive relation for porosity and compute derivatives
      void constitutive_derivatives(const Teuchos::ParameterList& params,  //!< (i) parameter list
          const double& press,        //!< (i) fluid pressure at gauss point
          const double& J,            //!< (i) Jacobian determinant at gauss point
          const double& porosity,     //!< (i) porosity at gauss point
          const double& refporosity,  //!< (i) porosity at gauss point
          double* dW_dp,              //!< (o) derivative of potential w.r.t. pressure
          double* dW_dphi,            //!< (o) derivative of potential w.r.t. porosity
          double* dW_dJ,              //!< (o) derivative of potential w.r.t. jacobian
          double* dW_dphiref,         //!< (o) derivative of potential w.r.t. reference porosity
          double* W                   //!< (o) inner potential
          ) override;

      //! compute current porosity and save it
      void compute_porosity(
          const double& refporosity,  //!< (i) initial/reference porosity at gauss point
          const double& press,        //!< (i) pressure at gauss point
          const double& J,            //!< (i) determinant of jacobian at gauss point
          const int& gp,              //!< (i) number of current gauss point
          double& porosity,           //!< (o) porosity at gauss point
          double* dphi_dp,    //!< (o) first derivative of porosity w.r.t. pressure at gauss point
          double* dphi_dJ,    //!< (o) first derivative of porosity w.r.t. jacobian at gauss point
          double* dphi_dJdp,  //!< (o) derivative of porosity w.r.t. pressure and jacobian at gauss
                              //!< point
          double* dphi_dJJ,   //!< (o) second derivative of porosity w.r.t. jacobian at gauss point
          double* dphi_dpp,   //!< (o) second derivative of porosity w.r.t. pressure at gauss point
          double* dphi_dphiref  //!< (o) derivative of porosity w.r.t. reference porosity (only
                                //!< nonzero with reaction)
          ) override;

      //! return inverse bulkmodulus (=compressibility)
      double inv_bulk_modulus() const override { return 1.0 / bulk_modulus_; }

     private:
      //! @name material parameters
      //!@{
      //! bulk modulus of skeleton phase
      double bulk_modulus_;
      //! penalty parameter for porosity
      double penalty_parameter_;
      //!@}
    };

    /*----------------------------------------------------------------------*/
    //! constant porosity law
    class PoroLawConstant : public PoroLaw
    {
     public:
      //! standard constructor
      explicit PoroLawConstant(const Core::Mat::PAR::Parameter::Data& matdata);

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! evaluate constitutive relation for porosity and compute derivatives
      void constitutive_derivatives(const Teuchos::ParameterList& params,  //!< (i) parameter list
          const double& press,        //!< (i) fluid pressure at gauss point
          const double& J,            //!< (i) Jacobian determinant at gauss point
          const double& porosity,     //!< (i) porosity at gauss point
          const double& refporosity,  //!< (i) porosity at gauss point
          double* dW_dp,              //!< (o) derivative of potential w.r.t. pressure
          double* dW_dphi,            //!< (o) derivative of potential w.r.t. porosity
          double* dW_dJ,              //!< (o) derivative of potential w.r.t. jacobian
          double* dW_dphiref,         //!< (o) derivative of potential w.r.t. reference porosity
          double* W                   //!< (o) inner potential
          ) override;

      //! compute current porosity and save it
      void compute_porosity(
          const double& refporosity,  //!< (i) initial/reference porosity at gauss point
          const double& press,        //!< (i) pressure at gauss point
          const double& J,            //!< (i) determinant of jacobian at gauss point
          const int& gp,              //!< (i) number of current gauss point
          double& porosity,           //!< (o) porosity at gauss point
          double* dphi_dp,    //!< (o) first derivative of porosity w.r.t. pressure at gauss point
          double* dphi_dJ,    //!< (o) first derivative of porosity w.r.t. jacobian at gauss point
          double* dphi_dJdp,  //!< (o) derivative of porosity w.r.t. pressure and jacobian at gauss
                              //!< point
          double* dphi_dJJ,   //!< (o) second derivative of porosity w.r.t. jacobian at gauss point
          double* dphi_dpp,   //!< (o) second derivative of porosity w.r.t. pressure at gauss point
          double* dphi_dphiref  //!< (o) derivative of porosity w.r.t. reference porosity (only
                                //!< nonzero with reaction)
          ) override;

      //! return inverse bulkmodulus (=compressibility)
      double inv_bulk_modulus() const override { return 0.0; }
    };

    /*----------------------------------------------------------------------*/
    //! incompressible skeleton porosity law
    class PoroLawIncompSkeleton : public PoroLaw
    {
     public:
      //! standard constructor
      explicit PoroLawIncompSkeleton(const Core::Mat::PAR::Parameter::Data& matdata);

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! evaluate constitutive relation for porosity and compute derivatives
      void constitutive_derivatives(const Teuchos::ParameterList& params,  //!< (i) parameter list
          const double& press,        //!< (i) fluid pressure at gauss point
          const double& J,            //!< (i) Jacobian determinant at gauss point
          const double& porosity,     //!< (i) porosity at gauss point
          const double& refporosity,  //!< (i) porosity at gauss point
          double* dW_dp,              //!< (o) derivative of potential w.r.t. pressure
          double* dW_dphi,            //!< (o) derivative of potential w.r.t. porosity
          double* dW_dJ,              //!< (o) derivative of potential w.r.t. jacobian
          double* dW_dphiref,         //!< (o) derivative of potential w.r.t. reference porosity
          double* W                   //!< (o) inner potential
          ) override;

      //! compute current porosity and save it
      void compute_porosity(
          const double& refporosity,  //!< (i) initial/reference porosity at gauss point
          const double& press,        //!< (i) pressure at gauss point
          const double& J,            //!< (i) determinant of jacobian at gauss point
          const int& gp,              //!< (i) number of current gauss point
          double& porosity,           //!< (o) porosity at gauss point
          double* dphi_dp,    //!< (o) first derivative of porosity w.r.t. pressure at gauss point
          double* dphi_dJ,    //!< (o) first derivative of porosity w.r.t. jacobian at gauss point
          double* dphi_dJdp,  //!< (o) derivative of porosity w.r.t. pressure and jacobian at gauss
                              //!< point
          double* dphi_dJJ,   //!< (o) second derivative of porosity w.r.t. jacobian at gauss point
          double* dphi_dpp,   //!< (o) second derivative of porosity w.r.t. pressure at gauss point
          double* dphi_dphiref  //!< (o) derivative of porosity w.r.t. reference porosity (only
                                //!< nonzero with reaction)
          ) override;

      //! return inverse bulkmodulus (=compressibility)
      double inv_bulk_modulus() const override { return 0.0; }
    };

    /*----------------------------------------------------------------------*/
    //! linear Biot model for porosity law
    class PoroLawLinBiot : public PoroLaw
    {
     public:
      //! standard constructor
      explicit PoroLawLinBiot(const Core::Mat::PAR::Parameter::Data& matdata);

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! evaluate constitutive relation for porosity and compute derivatives
      void constitutive_derivatives(const Teuchos::ParameterList& params,  //!< (i) parameter list
          const double& press,        //!< (i) fluid pressure at gauss point
          const double& J,            //!< (i) Jacobian determinant at gauss point
          const double& porosity,     //!< (i) porosity at gauss point
          const double& refporosity,  //!< (i) porosity at gauss point
          double* dW_dp,              //!< (o) derivative of potential w.r.t. pressure
          double* dW_dphi,            //!< (o) derivative of potential w.r.t. porosity
          double* dW_dJ,              //!< (o) derivative of potential w.r.t. jacobian
          double* dW_dphiref,         //!< (o) derivative of potential w.r.t. reference porosity
          double* W                   //!< (o) inner potential
          ) override;

      //! compute current porosity and save it
      void compute_porosity(
          const double& refporosity,  //!< (i) initial/reference porosity at gauss point
          const double& press,        //!< (i) pressure at gauss point
          const double& J,            //!< (i) determinant of jacobian at gauss point
          const int& gp,              //!< (i) number of current gauss point
          double& porosity,           //!< (o) porosity at gauss point
          double* dphi_dp,    //!< (o) first derivative of porosity w.r.t. pressure at gauss point
          double* dphi_dJ,    //!< (o) first derivative of porosity w.r.t. jacobian at gauss point
          double* dphi_dJdp,  //!< (o) derivative of porosity w.r.t. pressure and jacobian at gauss
                              //!< point
          double* dphi_dJJ,   //!< (o) second derivative of porosity w.r.t. jacobian at gauss point
          double* dphi_dpp,   //!< (o) second derivative of porosity w.r.t. pressure at gauss point
          double* dphi_dphiref  //!< (o) derivative of porosity w.r.t. reference porosity (only
                                //!< nonzero with reaction)
          ) override;

      //! return inverse bulkmodulus (=compressibility)
      double inv_bulk_modulus() const override { return inv_biot_modulus_; }

     private:
      //! @name material parameters
      //!@{
      //! inverse biot modulus
      double inv_biot_modulus_;
      //! Biot coefficient
      double biot_coeff_;
      //!@}
    };

    class PoroDensityLaw;
    /*----------------------------------------------------------------------*/
    //! porosity law depending on density
    class PoroLawDensityDependent : public PoroLaw
    {
     public:
      //! standard constructor
      explicit PoroLawDensityDependent(const Core::Mat::PAR::Parameter::Data& matdata);

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! evaluate constitutive relation for porosity and compute derivatives
      void constitutive_derivatives(const Teuchos::ParameterList& params,  //!< (i) parameter list
          const double& press,        //!< (i) fluid pressure at gauss point
          const double& J,            //!< (i) Jacobian determinant at gauss point
          const double& porosity,     //!< (i) porosity at gauss point
          const double& refporosity,  //!< (i) porosity at gauss point
          double* dW_dp,              //!< (o) derivative of potential w.r.t. pressure
          double* dW_dphi,            //!< (o) derivative of potential w.r.t. porosity
          double* dW_dJ,              //!< (o) derivative of potential w.r.t. jacobian
          double* dW_dphiref,         //!< (o) derivative of potential w.r.t. reference porosity
          double* W                   //!< (o) inner potential
          ) override;

      //! compute current porosity and save it
      void compute_porosity(
          const double& refporosity,  //!< (i) initial/reference porosity at gauss point
          const double& press,        //!< (i) pressure at gauss point
          const double& J,            //!< (i) determinant of jacobian at gauss point
          const int& gp,              //!< (i) number of current gauss point
          double& porosity,           //!< (o) porosity at gauss point
          double* dphi_dp,    //!< (o) first derivative of porosity w.r.t. pressure at gauss point
          double* dphi_dJ,    //!< (o) first derivative of porosity w.r.t. jacobian at gauss point
          double* dphi_dJdp,  //!< (o) derivative of porosity w.r.t. pressure and jacobian at gauss
                              //!< point
          double* dphi_dJJ,   //!< (o) second derivative of porosity w.r.t. jacobian at gauss point
          double* dphi_dpp,   //!< (o) second derivative of porosity w.r.t. pressure at gauss point
          double* dphi_dphiref  //!< (o) derivative of porosity w.r.t. reference porosity (only
                                //!< nonzero with reaction)
          ) override;

      //! return inverse bulkmodulus (=compressibility)
      double inv_bulk_modulus() const override;

     private:
      //! @name material parameters
      //!@{
      //! density law
      Mat::PAR::PoroDensityLaw* density_law_;
      //!@}
    };
  }  // namespace PAR
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

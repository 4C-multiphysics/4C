// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_CARREAUYASUDA_HPP
#define FOUR_C_MAT_CARREAUYASUDA_HPP



#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters
    class CarreauYasuda : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      CarreauYasuda(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      const double nu_0_;    /* zero-shear viscosity */
      const double nu_inf_;  /* infinite-shear viscosity */
      const double lambda_;  /* characteristic time */
      const double a_param_; /* constant parameter */
      const double b_param_; /* constant parameter */
      const double density_; /* density */

      //@}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class CarreauYasuda

  }  // namespace PAR

  class CarreauYasudaType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "CarreauYasudaType"; }

    static CarreauYasudaType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static CarreauYasudaType instance_;
  };

  /// Nonlinear viscosity according to Carreau-Yasuda
  class CarreauYasuda : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    CarreauYasuda();

    /// construct the material object given material parameters
    explicit CarreauYasuda(Mat::PAR::CarreauYasuda* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */

    int unique_par_object_id() const override
    {
      return CarreauYasudaType::instance().unique_par_object_id();
    }

    /*!
      \brief Pack this class so it can be communicated

      Resizes the vector data and stores all information of a class in it.
      The first information to be stored in data has to be the
      unique parobject id delivered by unique_par_object_id() which will then
      identify the exact class on the receiving processor.

      \param data (in/out): char vector to store class information
    */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
      \brief Unpack data from a char vector into this class

      The vector data contains all information to rebuild the
      exact copy of an instance of a class on a different processor.
      The first entry in data has to be an integer which is the unique
      parobject id defined at the top of this file and delivered by
      unique_par_object_id().

      \param data (in) : vector storing all data to be unpacked into this
      instance.
    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_carreauyasuda;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<CarreauYasuda>(*this);
    }

    /// return material parameters for element calculation
    //@{

    /// return parameter for zero-shear viscosity
    double nu0() const { return params_->nu_0_; }
    /// return parameter for infinite-shear viscosity
    double nu_inf() const { return params_->nu_inf_; }
    /// parameter for characteristic time
    double lambda() const { return params_->lambda_; }
    /// constant parameter
    double a_param() const { return params_->a_param_; }
    /// constant parameter
    double b_param() const { return params_->b_param_; }
    /// density
    double density() const override { return params_->density_; }

    //@}

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    /// my material parameters
    Mat::PAR::CarreauYasuda* params_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MAXWELL_0D_ACINUS_HPP
#define FOUR_C_MAT_MAXWELL_0D_ACINUS_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_red_airways_elem_params.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for Maxwell 0D acinar material
    ///
    class Maxwell0dAcinus : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      Maxwell0dAcinus(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{
      /// first stiffness of the Maxwell model
      const double stiffness1_;
      /// first stiffness of the Maxwell model
      const double stiffness2_;
      /// first viscosity of the Maxwell model
      const double viscosity1_;
      /// first viscosity of the Maxwell model
      const double viscosity2_;
      //@}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class Maxwell_0d_acinus

  }  // namespace PAR

  class Maxwell0dAcinusType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "maxwell_0d_acinusType"; }

    static Maxwell0dAcinusType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static Maxwell0dAcinusType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for Maxwell 0D acinar material
  ///
  /// This object exists (several times) at every element
  class Maxwell0dAcinus : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    Maxwell0dAcinus();

    /// construct the material object given material parameters
    explicit Maxwell0dAcinus(Mat::PAR::Maxwell0dAcinus* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return Maxwell0dAcinusType::instance().unique_par_object_id();
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

    /*!
      \brief
    */
    virtual void setup(const Core::IO::InputParameterContainer& container)
    {
      FOUR_C_THROW(
          "Setup not implemented yet! Check your material type, "
          "maybe you are still using the base class MAT_0D_MAXWELL_ACINUS.");
    }

    /*!
       \brief
     */
    virtual void evaluate(Core::LinAlg::SerialDenseVector& epnp,
        Core::LinAlg::SerialDenseVector& epn, Core::LinAlg::SerialDenseVector& epnm,
        Core::LinAlg::SerialDenseMatrix& sysmat, Core::LinAlg::SerialDenseVector& rhs,
        const Discret::ReducedLung::ElemParams& params, const double NumOfAcini, const double Vo,
        double time, double dt)
    {
      FOUR_C_THROW("Evaluate not implemented yet !");
    }

    //@}

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_0d_maxwell_acinus;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<Maxwell0dAcinus>(*this);
    }

    /// return density
    double density() const override { return -1; }

    /// return first stiffness of the Maxwell model
    double stiffness1() const { return params_->stiffness1_; }

    /// return first stiffness of the Maxwell model
    double stiffness2() const { return params_->stiffness2_; }

    /// return first viscosity of the Maxwell model
    double viscosity1() const { return params_->viscosity1_; }

    /// return first viscosity of the Maxwell model
    double viscosity2() const { return params_->viscosity2_; }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// Return value of class parameter
    virtual double get_params(std::string parametername);

    /// Set value of class parameter
    virtual void set_params(std::string parametername, double new_value);

    /// Return names of visualization data
    virtual void vis_names(std::map<std::string, int>& names) {
      /* do nothing for simple material models */
    };

    /// Return visualization data
    virtual bool vis_data(const std::string& name, std::vector<double>& data, int eleID)
    { /* do nothing for simple material models */
      return false;
    };

   protected:
    /// my material parameters
    Mat::PAR::Maxwell0dAcinus* params_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

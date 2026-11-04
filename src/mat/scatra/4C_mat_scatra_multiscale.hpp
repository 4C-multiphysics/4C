// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SCATRA_MULTISCALE_HPP
#define FOUR_C_MAT_SCATRA_MULTISCALE_HPP

#include "4C_config.hpp"

#include "4C_mat_scatra.hpp"
#include "4C_mat_scatra_micro_macro_coupling.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  // forward declaration
  class ScatraMultiScaleGP;

  namespace PAR
  {
    //! material parameters
    class ScatraMultiScale : public ScatraMat, public ScatraMicroMacroCoupling
    {
     public:
      //! constructor
      ScatraMultiScale(const Core::Mat::PAR::Parameter::Data& matdata);


      //! create material
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! return porosity
      double porosity() const { return porosity_; };

      //! return tortuosity
      double tortuosity() const { return tortuosity_; };

     private:
      //! @name material parameters
      //@{
      //! porosity
      const double porosity_;

      //! tortuosity
      const double tortuosity_;
      //@}
    };  // class Mat::PAR::ScatraMultiScale
  }  // namespace PAR


  /*----------------------------------------------------------------------*/
  class ScatraMultiScaleType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ScatraMultiScaleType"; };

    static ScatraMultiScaleType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ScatraMultiScaleType instance_;
  };


  /*----------------------------------------------------------------------*/
  //! material wrapper
  class ScatraMultiScale : public ScatraMat, public ScatraMicroMacroCoupling
  {
   public:
    //! construct empty material
    ScatraMultiScale();

    //! construct material with specific material parameters
    explicit ScatraMultiScale(Mat::PAR::ScatraMultiScale* params);

    //! @name packing and unpacking
    /*!
      \brief Return unique ParObject id

      Every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return ScatraMultiScaleType::instance().unique_par_object_id();
    };

    /*!
      \brief Pack this class so it can be communicated

      Resizes the vector data and stores all information of a class in it.
      The first information to be stored in data has to be the
      unique ParObject ID delivered by unique_par_object_id() which will then
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

      \param data (in) : vector storing all data to be unpacked into this instance.
    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;
    //@}

    //! return material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_scatra_multiscale;
    };

    //! clone material
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ScatraMultiScale>(*this);
    };

    //! return porosity
    double porosity() const { return params_->porosity(); };

    //! return tortuosity
    double tortuosity() const { return params_->tortuosity(); };

   private:
    //! return material parameters
    const Mat::PAR::ScatraMicroMacroCoupling* params() const override { return params_; };

    //! material parameters
    Mat::PAR::ScatraMultiScale* params_;
  };
}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif

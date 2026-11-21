// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SORET_HPP
#define FOUR_C_MAT_SORET_HPP

#include "4C_config.hpp"

#include "4C_mat_fourier.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    //! parameters for Soret material
    class Soret : public Fourier
    {
     public:
      //! constructor
      Soret(const Core::Mat::PAR::Parameter::Data& matdata);


      //! create instance of Soret material
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! return Soret coefficient
      double soret_coefficient() const { return soretcoefficient_; };

     private:
      //! @name parameters for Soret material
      //@{
      //! Soret coefficient
      const double soretcoefficient_;
      //@}
    };  // class Mat::PAR::Soret
  }  // namespace PAR


  /*----------------------------------------------------------------------*/
  class SoretType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "SoretType"; };

    static SoretType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static SoretType instance_;
  };


  /*----------------------------------------------------------------------*/
  //! wrapper for Soret material
  class Soret : public Fourier
  {
   public:
    //! construct empty Soret material
    Soret();

    //! construct Soret material with specific material parameters
    explicit Soret(Mat::PAR::Soret* params);

    //! @name packing and unpacking
    /*!
      \brief Return unique ParObject id

      Every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return SoretType::instance().unique_par_object_id();
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
      return Core::Materials::m_soret;
    };

    //! clone Soret material
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<Soret>(*this);
    };

    //! return Soret coefficient
    double soret_coefficient() const { return params_->soret_coefficient(); };

   private:
    //! return material parameters
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    //! material parameters
    Mat::PAR::Soret* params_;
  };
}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif

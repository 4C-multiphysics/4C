// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_NEWMAN_MULTISCALE_HPP
#define FOUR_C_MAT_NEWMAN_MULTISCALE_HPP

#include "4C_config.hpp"

#include "4C_mat_newman.hpp"
#include "4C_mat_scatra_micro_macro_coupling.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    //! material parameters
    class NewmanMultiScale : public Newman, public ScatraMicroMacroCoupling
    {
     public:
      //! constructor
      NewmanMultiScale(const Core::Mat::PAR::Parameter::Data& matdata);


      //! create instance of Newman multi-scale material
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! electronic conductivity
      double electronic_cond() const { return electronic_cond_; }

      //! function number to scale electronic conductivity with. The argument for the function is
      //! the concentration
      int conc_dep_scale_func_num() const { return conc_dep_scale_func_num_; }

     private:
      //! @name parameters for Newman multi-scale material
      //@{
      //! electronic conductivity
      const double electronic_cond_;

      //! function number to scale electronic conductivity with. The argument for the function is
      //! the concentration
      const int conc_dep_scale_func_num_;
      //@}
    };  // class Mat::PAR::NewmanMultiScale
  }  // namespace PAR


  /*----------------------------------------------------------------------*/
  class NewmanMultiScaleType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "NewmanMultiScaleType"; };

    static NewmanMultiScaleType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static NewmanMultiScaleType instance_;
  };


  /*----------------------------------------------------------------------*/
  //! wrapper for Newman multi-scale material
  class NewmanMultiScale : public Newman, public ScatraMicroMacroCoupling
  {
   public:
    //! construct empty Newman multi-scale material
    NewmanMultiScale();

    //! construct Newman multi-scale material with specific material parameters
    explicit NewmanMultiScale(Mat::PAR::NewmanMultiScale* params);

    //! @name packing and unpacking
    /*!
      \brief Return unique ParObject id

      Every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return NewmanMultiScaleType::instance().unique_par_object_id();
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
      return Core::Materials::m_newman_multiscale;
    };

    //! clone Newman multi-scale material
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<NewmanMultiScale>(*this);
    };

    //! compute electronic conductivity and scale by function evaluated at @p gp
    double electronic_cond(int gp) const;

   private:
    //! return material parameters
    const Mat::PAR::ScatraMicroMacroCoupling* params() const override { return params_; };

    //! material parameters
    Mat::PAR::NewmanMultiScale* params_;
  };  // wrapper for Newman multi-scale material
}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif

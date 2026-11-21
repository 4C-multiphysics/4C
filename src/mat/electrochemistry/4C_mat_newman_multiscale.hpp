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
      explicit NewmanMultiScale(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! electronic conductivity
      [[nodiscard]] double electronic_cond() const { return electronic_cond_; }

      //! function number to scale electronic conductivity with. The argument for the function is
      //! the concentration
      [[nodiscard]] int conc_dep_scale_func_num() const { return conc_dep_scale_func_num_; }

     private:
      //! @name parameters for Newman multi-scale material
      ///@{
      //! electronic conductivity
      const double electronic_cond_;

      //! function number to scale electronic conductivity with. The argument for the function is
      //! the concentration
      const int conc_dep_scale_func_num_;
      ///@}
    };  // class Mat::PAR::NewmanMultiScale
  }  // namespace PAR


  /*----------------------------------------------------------------------*/
  class NewmanMultiScaleType : public Core::Communication::ParObjectType
  {
   public:
    [[nodiscard]] std::string name() const override { return "NewmanMultiScaleType"; }

    static NewmanMultiScaleType& instance() { return instance_; }

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
    NewmanMultiScale() = default;

    //! construct Newman multi-scale material with specific material parameters
    explicit NewmanMultiScale(Mat::PAR::NewmanMultiScale* params);

    [[nodiscard]] int unique_par_object_id() const override
    {
      return NewmanMultiScaleType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    [[nodiscard]] Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_newman_multiscale;
    }

    [[nodiscard]] std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<NewmanMultiScale>(*this);
    }

    //! compute electronic conductivity and scale by function evaluated at @p gp
    [[nodiscard]] double electronic_cond(int gp) const;

   private:
    [[nodiscard]] const Mat::PAR::ScatraMicroMacroCoupling* params() const override
    {
      return params_;
    }

    //! material parameters
    Mat::PAR::NewmanMultiScale* params_;
  };  // wrapper for Newman multi-scale material
}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif

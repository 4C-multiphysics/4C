// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SCATRA_GROWTH_REMODEL_HPP
#define FOUR_C_MAT_SCATRA_GROWTH_REMODEL_HPP

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
    /// parameters for scalar transport material with growth/remodeling
    class ScatraGrowthRemodelMat : public Core::Mat::PAR::Parameter
    {
     public:
      /// scalar quantity type for which the reaction coefficient gets evaluated
      enum class ScalarQuantity
      {
        growth,
        remodeling
      };

      /// standard constructor
      ScatraGrowthRemodelMat(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// diffusivity
      [[nodiscard]] double diffusivity() const { return diffusivity_; }

      /// ID of structure material (RemodelFiberSsi constituent) that provides reaction coefficient
      [[nodiscard]] int structure_material_id() const { return structure_material_id_; }

      [[nodiscard]] ScalarQuantity scalar_quantity() const { return scalar_quantity_; }

     private:
      double diffusivity_;
      int structure_material_id_;
      ScalarQuantity scalar_quantity_;

    };  // class ScatraGrowthRemodelMat

  }  // namespace PAR

  class ScatraGrowthRemodelMatType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ScatraGrowthRemodelMatType"; }

    static ScatraGrowthRemodelMatType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ScatraGrowthRemodelMatType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// wrapper for scalar transport material with growth/remodeling
  class ScatraGrowthRemodelMat : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    ScatraGrowthRemodelMat();

    /// construct the material object given material parameters
    explicit ScatraGrowthRemodelMat(Mat::PAR::ScatraGrowthRemodelMat* params);

    int unique_par_object_id() const override
    {
      return ScatraGrowthRemodelMatType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_scatra_gr;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ScatraGrowthRemodelMat>(*this);
    }

    /// diffusivity
    [[nodiscard]] double diffusivity() const { return params_->diffusivity(); }

    /// ID of structure material (RemodelFiberSsi constituent) that provides stress
    [[nodiscard]] int structure_material_id() const { return params_->structure_material_id(); }

    /// growth/remodeling scalar quantity type
    [[nodiscard]] Mat::PAR::ScatraGrowthRemodelMat::ScalarQuantity scalar_quantity() const
    {
      return params_->scalar_quantity();
    }

    /// Return material parameter data
    [[nodiscard]] Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    Mat::PAR::ScatraGrowthRemodelMat* params_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif  // FOUR_C_MAT_SCATRA_GROWTH_REMODEL_HPP
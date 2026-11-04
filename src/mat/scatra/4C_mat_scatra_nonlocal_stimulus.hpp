// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SCATRA_NONLOCAL_STIMULUS_HPP
#define FOUR_C_MAT_SCATRA_NONLOCAL_STIMULUS_HPP

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
    /// parameters for the Helmholtz non-local stimulus scalar transport material
    class ScatraNonlocalStimulusMat : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      ScatraNonlocalStimulusMat(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// characteristic length scale squared \f$ \ell_c^2 \f$
      [[nodiscard]] double characteristic_length_sq() const { return characteristic_length_sq_; }

      /// ID of structure material (RemodelFiberSsi constituent) that provides the local stimulus
      [[nodiscard]] int structure_material_id() const { return structure_material_id_; }

     private:
      double characteristic_length_sq_;
      int structure_material_id_;

    };  // class ScatraNonlocalStimulusMat

  }  // namespace PAR

  class ScatraNonlocalStimulusMatType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ScatraNonlocalStimulusMatType"; }

    static ScatraNonlocalStimulusMatType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ScatraNonlocalStimulusMatType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// wrapper for Helmholtz non-local G&R stimulus scalar transport material
  class ScatraNonlocalStimulusMat : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    ScatraNonlocalStimulusMat();

    /// construct the material object given material parameters
    explicit ScatraNonlocalStimulusMat(Mat::PAR::ScatraNonlocalStimulusMat* params);

    int unique_par_object_id() const override
    {
      return ScatraNonlocalStimulusMatType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_scatra_nl_stimulus;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ScatraNonlocalStimulusMat>(*this);
    }

    /// characteristic length squared \ell_c^2
    [[nodiscard]] double characteristic_length_sq() const
    {
      return params_->characteristic_length_sq();
    }

    /// ID of structure material (RemodelFiberSsi constituent) that provides the local stimulus
    [[nodiscard]] int structure_material_id() const { return params_->structure_material_id(); }

    /// Return material parameter data
    [[nodiscard]] Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    Mat::PAR::ScatraNonlocalStimulusMat* params_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif  // FOUR_C_MAT_SCATRA_NONLOCAL_STIMULUS_HPP

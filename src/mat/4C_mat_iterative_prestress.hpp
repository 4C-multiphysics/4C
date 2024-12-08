// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ITERATIVE_PRESTRESS_HPP
#define FOUR_C_MAT_ITERATIVE_PRESTRESS_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    class IterativePrestressMaterial : public Core::Mat::PAR::Parameter
    {
     public:
      explicit IterativePrestressMaterial(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override;

      const int mat_id_;

      const bool is_prestress_active_ = false;
    };
  }  // namespace PAR

  class IterativePrestressMaterialType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "IterativePrestressMaterialType"; }

    static IterativePrestressMaterialType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static IterativePrestressMaterialType instance_;
  };

  class IterativePrestressMaterial : public Mat::So3Material
  {
   public:
    IterativePrestressMaterial() = default;
    explicit IterativePrestressMaterial(PAR::IterativePrestressMaterial* params);

    [[nodiscard]] int unique_par_object_id() const override
    {
      return IterativePrestressMaterialType::instance().unique_par_object_id();
    }

    std::shared_ptr<Material> clone() const override
    {
      return std::make_shared<IterativePrestressMaterial>(*this);
    }

    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      child_material_->valid_kinematics(kinem);
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_iterative_prestress;
    }

    void setup(const int numgp, const Core::IO::InputParameterContainer& container) override;

    [[nodiscard]] double density() const override { return child_material_->density(); }

    void strain_energy(
        const Core::LinAlg::Matrix<6, 1>& glstrain, double& psi, int gp, int eleGID) const override;


    void evaluate(const Core::LinAlg::Matrix<3, 3>* defgrd,
        const Core::LinAlg::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
        Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat, int gp,
        int eleGID) override;

    void post_setup(Teuchos::ParameterList& params, int eleGID) override;

    void update() override;

    void update(Core::LinAlg::Matrix<3, 3> const& defgrd, const int gp,
        Teuchos::ParameterList& params, const int eleGID) override;

    void vis_names(std::map<std::string, int>& names) const override;

    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

   private:
    std::shared_ptr<So3Material> child_material_ = nullptr;

    PAR::IterativePrestressMaterial* params_ = nullptr;

    std::vector<Core::LinAlg::Matrix<3, 3>> prestretch_tensor_ = {};
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

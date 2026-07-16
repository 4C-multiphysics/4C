// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_PLASTICMOHRCOULOMB_HPP
#define FOUR_C_MAT_PLASTICMOHRCOULOMB_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    class PlasticMohrCoulomb : public Core::Mat::PAR::Parameter
    {
     public:
      explicit PlasticMohrCoulomb(const Core::Mat::PAR::Parameter::Data& matdata);

      const double youngs_;
      const double poisson_ratio_;
      const double density_;
      const double cohesion_;
      const double friction_angle_;
      const double dilatancy_angle_;
      const double linear_hardening_;
      const double saturation_hardening_;
      const double hardening_exponent_;
      const double tolerance_;
      const int max_iterations_;

      std::shared_ptr<Core::Mat::Material> create_material() override;
    };
  }  // namespace PAR

  class PlasticMohrCoulombType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "PlasticMohrCoulombType"; }
    static PlasticMohrCoulombType& instance() { return instance_; }
    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static PlasticMohrCoulombType instance_;
  };

  class PlasticMohrCoulomb : public So3Material
  {
   public:
    PlasticMohrCoulomb();
    explicit PlasticMohrCoulomb(Mat::PAR::PlasticMohrCoulomb* params);

    int unique_par_object_id() const override
    {
      return PlasticMohrCoulombType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_plmohrcoulomb;
    }

    void valid_kinematics(Solid::KinemType kinem) override
    {
      if (kinem != Solid::KinemType::nonlinearTotLag)
        FOUR_C_THROW("The Mohr-Coulomb material requires nonlinear total-Lagrangian kinematics.");
    }

    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<PlasticMohrCoulomb>(*this);
    }

    Core::Mat::PAR::Parameter* parameter() const override { return params_; }
    double density() const override { return params_->density_; }

    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;
    void update() override;

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;
    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

   private:
    Mat::PAR::PlasticMohrCoulomb* params_;
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> inv_plastic_rcg_last_;
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> inv_plastic_rcg_current_;
    std::vector<double> accumulated_plastic_strain_last_;
    std::vector<double> accumulated_plastic_strain_current_;
    std::vector<double> accumulated_plastic_volumetric_strain_last_;
    std::vector<double> accumulated_plastic_volumetric_strain_current_;
    std::vector<double> dissipated_energy_last_;
    std::vector<double> dissipated_energy_current_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

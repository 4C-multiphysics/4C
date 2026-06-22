// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_viscoelast_summand.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_viscoelast_coupmyocard.hpp"
#include "4C_mat_viscoelast_fsls.hpp"
#include "4C_mat_viscoelast_generalizedmaxwell.hpp"
#include "4C_mat_viscoelast_isoratedep.hpp"
#include "4C_mat_viscoelast_quasilineargeneralizedmaxwell.hpp"

FOUR_C_NAMESPACE_OPEN

int Mat::ViscoElast::Summand::unique_par_object_id() const { return -1; }

void Mat::ViscoElast::Summand::pack(Core::Communication::PackBuffer& data) const {}

void Mat::ViscoElast::Summand::unpack(Core::Communication::UnpackBuffer& buffer) {}

void Mat::ViscoElast::Summand::pack_summand(Core::Communication::PackBuffer& data) const {}

void Mat::ViscoElast::Summand::unpack_summand(Core::Communication::UnpackBuffer& buffer) {}

void Mat::ViscoElast::Summand::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
}

void Mat::ViscoElast::Summand::update() {}

void Mat::ViscoElast::Summand::specify_formulation(
    bool& isoprinc, bool& isomod, bool& anisoprinc, bool& anisomod, bool& viscogeneral)
{
}

void Mat::ViscoElast::Summand::add_coefficients_visco_principal(
    const Core::LinAlg::Matrix<3, 1>& inv, Core::LinAlg::Matrix<8, 1>& mu,
    Core::LinAlg::Matrix<33, 1>& xi, Core::LinAlg::Matrix<7, 1>& rateinv,
    const Teuchos::ParameterList& params, double dt, int gp, int eleGID)
{
}

void Mat::ViscoElast::Summand::add_coefficients_visco_modified(
    const Core::LinAlg::Matrix<3, 1>& modinv, Core::LinAlg::Matrix<8, 1>& modmu,
    Core::LinAlg::Matrix<33, 1>& modxi, Core::LinAlg::Matrix<7, 1>& modrateinv,
    const Teuchos::ParameterList& params, double dt, int gp, int eleGID)
{
}

void Mat::ViscoElast::Summand::read_material_parameters_visco(
    double& tau, double& beta, double& alpha, std::string& solve)
{
}

void Mat::ViscoElast::Summand::read_material_parameters(
    int& numbranch, const std::vector<int>*& matids, std::string& solve)
{
}

void Mat::ViscoElast::Summand::read_material_parameters(double& tau, int& matid) {}

void Mat::ViscoElast::Summand::specify_visco_formulation(bool& visco_iso_rate,
    bool& visco_generalized_maxwell, bool& visco_quasi_linear_generalized_maxwell, bool& visco_fsls)
{
}

std::shared_ptr<Mat::ViscoElast::Summand> Mat::ViscoElast::Summand::factory(const int matnum)
{
  FOUR_C_ASSERT_ALWAYS(Global::Problem::instance()->materials() != nullptr,
      "List of materials cannot be accessed in the global problem instance.");

  FOUR_C_ASSERT_ALWAYS(Global::Problem::instance()->materials()->num() != 0,
      "List of materials in the global problem instance is empty.");

  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
  auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(matnum);

  switch (curmat->type())
  {
    case Core::Materials::mes_coupmyocard:
    {
      auto* params = dynamic_cast<Mat::ViscoElast::PAR::CoupMyocard*>(curmat);
      return std::make_shared<Mat::ViscoElast::CoupMyocard>(params);
    }
    case Core::Materials::mes_fsls:
    {
      auto* params = dynamic_cast<Mat::ViscoElast::PAR::Fsls*>(curmat);
      return std::make_shared<Mat::ViscoElast::Fsls>(params);
    }
    case Core::Materials::mes_generalizedmaxwell:
    {
      auto* params = dynamic_cast<Mat::ViscoElast::PAR::GeneralizedMaxwell*>(curmat);
      return std::make_shared<Mat::ViscoElast::GeneralizedMaxwell>(params);
    }
    case Core::Materials::mes_quasilineargeneralizedmaxwell:
    {
      auto* params = dynamic_cast<Mat::ViscoElast::PAR::QuasiLinearGeneralizedMaxwell*>(curmat);
      return std::make_shared<Mat::ViscoElast::QuasiLinearGeneralizedMaxwell>(params);
    }
    case Core::Materials::mes_isoratedep:
    {
      auto* params = dynamic_cast<Mat::ViscoElast::PAR::IsoRateDep*>(curmat);
      return std::make_shared<Mat::ViscoElast::IsoRateDep>(params);
    }
    case Core::Materials::mes_viscobranch:
    {
      auto* params = dynamic_cast<Mat::ViscoElast::PAR::ViscoBranch*>(curmat);
      return std::make_shared<Mat::ViscoElast::ViscoBranch>(params);
    }
    default:
      FOUR_C_THROW("cannot deal with type {}", curmat->type());
  }

  return nullptr;
}

FOUR_C_NAMESPACE_CLOSE

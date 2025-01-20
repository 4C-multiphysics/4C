// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_summand.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_elast_anisoactivestress_evolution.hpp"
#include "4C_mat_elast_coup13apow.hpp"
#include "4C_mat_elast_coup1pow.hpp"
#include "4C_mat_elast_coup2pow.hpp"
#include "4C_mat_elast_coup3pow.hpp"
#include "4C_mat_elast_coupanisoexpo.hpp"
#include "4C_mat_elast_coupanisoexposhear.hpp"
#include "4C_mat_elast_coupanisoexpotwocoup.hpp"
#include "4C_mat_elast_coupanisoneohooke.hpp"
#include "4C_mat_elast_coupanisoneohooke_VarProp.hpp"
#include "4C_mat_elast_coupanisopow.hpp"
#include "4C_mat_elast_coupblatzko.hpp"
#include "4C_mat_elast_coupexppol.hpp"
#include "4C_mat_elast_couplogmixneohooke.hpp"
#include "4C_mat_elast_couplogneohooke.hpp"
#include "4C_mat_elast_coupmooneyrivlin.hpp"
#include "4C_mat_elast_coupneohooke.hpp"
#include "4C_mat_elast_coupSaintVenantKirchhoff.hpp"
#include "4C_mat_elast_coupsimopister.hpp"
#include "4C_mat_elast_couptransverselyisotropic.hpp"
#include "4C_mat_elast_coupvarga.hpp"
#include "4C_mat_elast_iso1pow.hpp"
#include "4C_mat_elast_iso2pow.hpp"
#include "4C_mat_elast_isoanisoexpo.hpp"
#include "4C_mat_elast_isoexpopow.hpp"
#include "4C_mat_elast_isomooneyrivlin.hpp"
#include "4C_mat_elast_isomuscle_blemker.hpp"
#include "4C_mat_elast_isoneohooke.hpp"
#include "4C_mat_elast_isoogden.hpp"
#include "4C_mat_elast_isotestmaterial.hpp"
#include "4C_mat_elast_isovarga.hpp"
#include "4C_mat_elast_isoyeoh.hpp"
#include "4C_mat_elast_remodelfiber.hpp"
#include "4C_mat_elast_visco_coupmyocard.hpp"
#include "4C_mat_elast_visco_fract.hpp"
#include "4C_mat_elast_visco_generalizedgenmax.hpp"
#include "4C_mat_elast_visco_genmax.hpp"
#include "4C_mat_elast_visco_isoratedep.hpp"
#include "4C_mat_elast_vologden.hpp"
#include "4C_mat_elast_volpenalty.hpp"
#include "4C_mat_elast_volpow.hpp"
#include "4C_mat_elast_volsussmanbathe.hpp"
#include "4C_mat_par_bundle.hpp"

FOUR_C_NAMESPACE_OPEN

std::shared_ptr<Mat::Elastic::Summand> Mat::Elastic::Summand::factory(int matnum)
{
  // for the sake of safety
  if (Global::Problem::instance()->materials() == nullptr)
    FOUR_C_THROW("List of materials cannot be accessed in the global problem instance.");

  // yet another safety check
  if (Global::Problem::instance()->materials()->num() == 0)
    FOUR_C_THROW("List of materials in the global problem instance is empty.");

  // retrieve problem instance to read from
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
  // retrieve validated input line of material ID in question
  auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(matnum);

  // construct structural tensor strategy for anisotropic materials
  switch (curmat->type())
  {
    case Core::Materials::mes_isoanisoexpo:
    case Core::Materials::mes_isomuscleblemker:
    case Core::Materials::mes_coupanisoexpo:
    case Core::Materials::mes_coupanisoexpoactive:
    case Core::Materials::mes_coupanisoexpotwocoup:
    case Core::Materials::mes_coupanisoneohooke:
    case Core::Materials::mes_coupanisopow:
    case Core::Materials::mes_coupanisoneohooke_varprop:
    {
      break;
    }
    default:
      break;
  }

  switch (curmat->type())
  {
    case Core::Materials::mes_anisoactivestress_evolution:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::AnisoActiveStressEvolution*>(curmat);
      return std::make_shared<AnisoActiveStressEvolution>(params);
    }
    case Core::Materials::mes_coupanisoexpoactive:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupAnisoExpoActive*>(curmat);
      return std::make_shared<CoupAnisoExpoActive>(params);
    }
    case Core::Materials::mes_coupanisoexpo:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupAnisoExpo*>(curmat);
      return std::make_shared<CoupAnisoExpo>(params);
    }
    case Core::Materials::mes_coupanisoexposhear:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupAnisoExpoShear*>(curmat);
      return std::make_shared<CoupAnisoExpoShear>(params);
    }
    case Core::Materials::mes_coupanisoexpotwocoup:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupAnisoExpoTwoCoup*>(curmat);
      return std::make_shared<CoupAnisoExpoTwoCoup>(params);
    }
    case Core::Materials::mes_coupanisoneohooke:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupAnisoNeoHooke*>(curmat);
      return std::make_shared<CoupAnisoNeoHooke>(params);
    }
    case Core::Materials::mes_coupanisoneohooke_varprop:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupAnisoNeoHookeVarProp*>(curmat);
      return std::make_shared<CoupAnisoNeoHookeVarProp>(params);
    }
    case Core::Materials::mes_coupanisopow:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupAnisoPow*>(curmat);
      return std::make_shared<CoupAnisoPow>(params);
    }
    case Core::Materials::mes_coupblatzko:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupBlatzKo*>(curmat);
      return std::make_shared<CoupBlatzKo>(params);
    }
    case Core::Materials::mes_coupexppol:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupExpPol*>(curmat);
      return std::make_shared<CoupExpPol>(params);
    }
    case Core::Materials::mes_couplogneohooke:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupLogNeoHooke*>(curmat);
      return std::make_shared<CoupLogNeoHooke>(params);
    }
    case Core::Materials::mes_couplogmixneohooke:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupLogMixNeoHooke*>(curmat);
      return std::make_shared<CoupLogMixNeoHooke>(params);
    }
    case Core::Materials::mes_coupmooneyrivlin:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupMooneyRivlin*>(curmat);
      return std::make_shared<CoupMooneyRivlin>(params);
    }
    case Core::Materials::mes_coupmyocard:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupMyocard*>(curmat);
      return std::make_shared<CoupMyocard>(params);
    }
    case Core::Materials::mes_coupneohooke:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupNeoHooke*>(curmat);
      return std::make_shared<CoupNeoHooke>(params);
    }
    case Core::Materials::mes_coup1pow:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::Coup1Pow*>(curmat);
      return std::make_shared<Coup1Pow>(params);
    }
    case Core::Materials::mes_coup2pow:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::Coup2Pow*>(curmat);
      return std::make_shared<Coup2Pow>(params);
    }
    case Core::Materials::mes_coup3pow:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::Coup3Pow*>(curmat);
      return std::make_shared<Coup3Pow>(params);
    }
    case Core::Materials::mes_coup13apow:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::Coup13aPow*>(curmat);
      return std::make_shared<Coup13aPow>(params);
    }
    case Core::Materials::mes_coupsimopister:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupSimoPister*>(curmat);
      return std::make_shared<CoupSimoPister>(params);
    }
    case Core::Materials::mes_coupSVK:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupSVK*>(curmat);
      return std::make_shared<CoupSVK>(params);
    }
    case Core::Materials::mes_couptransverselyisotropic:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupTransverselyIsotropic*>(curmat);
      return std::make_shared<CoupTransverselyIsotropic>(params);
    }
    case Core::Materials::mes_coupvarga:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::CoupVarga*>(curmat);
      return std::make_shared<CoupVarga>(params);
    }
    case Core::Materials::mes_fract:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::Fract*>(curmat);
      return std::make_shared<Fract>(params);
    }
    case Core::Materials::mes_genmax:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::GenMax*>(curmat);
      return std::make_shared<GenMax>(params);
    }
    case Core::Materials::mes_generalizedgenmax:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::GeneralizedGenMax*>(curmat);
      return std::make_shared<GeneralizedGenMax>(params);
    }
    case Core::Materials::mes_isoanisoexpo:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::IsoAnisoExpo*>(curmat);
      return std::make_shared<IsoAnisoExpo>(params);
    }
    case Core::Materials::mes_isoexpopow:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::IsoExpoPow*>(curmat);
      return std::make_shared<IsoExpoPow>(params);
    }
    case Core::Materials::mes_isomooneyrivlin:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::IsoMooneyRivlin*>(curmat);
      return std::make_shared<IsoMooneyRivlin>(params);
    }
    case Core::Materials::mes_isomuscleblemker:
    {
      Mat::Elastic::PAR::IsoMuscleBlemker* params =
          static_cast<Mat::Elastic::PAR::IsoMuscleBlemker*>(curmat);
      return std::make_shared<IsoMuscleBlemker>(params);
    }
    case Core::Materials::mes_isoneohooke:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::IsoNeoHooke*>(curmat);
      return std::make_shared<IsoNeoHooke>(params);
    }
    case Core::Materials::mes_isoogden:
    {
      auto* params = static_cast<Mat::Elastic::PAR::IsoOgden*>(curmat);
      return std::make_shared<IsoOgden>(params);
    }
    case Core::Materials::mes_iso1pow:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::Iso1Pow*>(curmat);
      return std::make_shared<Iso1Pow>(params);
    }
    case Core::Materials::mes_iso2pow:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::Iso2Pow*>(curmat);
      return std::make_shared<Iso2Pow>(params);
    }
    case Core::Materials::mes_isoratedep:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::IsoRateDep*>(curmat);
      return std::make_shared<IsoRateDep>(params);
    }
    case Core::Materials::mes_isotestmaterial:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::IsoTestMaterial*>(curmat);
      return std::make_shared<IsoTestMaterial>(params);
    }
    case Core::Materials::mes_isovarga:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::IsoVarga*>(curmat);
      return std::make_shared<IsoVarga>(params);
    }
    case Core::Materials::mes_isoyeoh:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::IsoYeoh*>(curmat);
      return std::make_shared<IsoYeoh>(params);
    }
    case Core::Materials::mes_remodelfiber:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::RemodelFiber*>(curmat);
      return std::make_shared<RemodelFiber>(params);
    }
    case Core::Materials::mes_vologden:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::VolOgden*>(curmat);
      return std::make_shared<VolOgden>(params);
    }
    case Core::Materials::mes_volpenalty:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::VolPenalty*>(curmat);
      return std::make_shared<VolPenalty>(params);
    }
    case Core::Materials::mes_volpow:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::VolPow*>(curmat);
      return std::make_shared<VolPow>(params);
    }
    case Core::Materials::mes_volsussmanbathe:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::VolSussmanBathe*>(curmat);
      return std::make_shared<VolSussmanBathe>(params);
    }
    case Core::Materials::mes_viscobranch:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::ViscoBranch*>(curmat);
      return std::make_shared<ViscoBranch>(params);
    }
    case Core::Materials::mes_viscopart:
    {
      auto* params = dynamic_cast<Mat::Elastic::PAR::ViscoPart*>(curmat);
      return std::make_shared<ViscoPart>(params);
    }
    default:
      FOUR_C_THROW("cannot deal with type %d", curmat->type());
  }
  return nullptr;
}

void Mat::Elastic::Summand::add_shear_mod(bool& haveshearmod, double& shearmod) const
{
  FOUR_C_THROW("Mat::Elastic::Summand::AddShearMod: Add Shear Modulus not implemented - do so!");
}

int Mat::Elastic::Summand::unique_par_object_id() const { return -1; }

void Mat::Elastic::Summand::pack(Core::Communication::PackBuffer& data) const { return; }

void Mat::Elastic::Summand::unpack(Core::Communication::UnpackBuffer& buffer) { return; };


// Function which reads in the given fiber value due to the FIBER1 nomenclature
void Mat::Elastic::Summand::read_fiber(const Core::IO::InputParameterContainer& container,
    const std::string& specifier, Core::LinAlg::Matrix<3, 1>& fiber_vector)
{
  auto fiber1 = container.get<std::vector<double>>(specifier);

  double f1norm = 0.;
  // normalization
  for (int i = 0; i < 3; ++i)
  {
    f1norm += fiber1[i] * fiber1[i];
  }
  f1norm = sqrt(f1norm);

  // fill final fiber vector
  for (int i = 0; i < 3; ++i) fiber_vector(i) = fiber1[i] / f1norm;
}

// Function which reads in the given fiber value due to the CIR-AXI-RAD nomenclature
void Mat::Elastic::Summand::read_rad_axi_cir(
    const Core::IO::InputParameterContainer& container, Core::LinAlg::Matrix<3, 3>& locsys)
{
  // read local (cylindrical) cosy-directions at current element
  // basis is local cosy with third vec e3 = circumferential dir and e2 = axial dir
  Core::LinAlg::Matrix<3, 1> fiber_rad;
  Core::LinAlg::Matrix<3, 1> fiber_axi;
  Core::LinAlg::Matrix<3, 1> fiber_cir;

  read_fiber(container, "RAD", fiber_rad);
  read_fiber(container, "AXI", fiber_axi);
  read_fiber(container, "CIR", fiber_cir);

  for (int i = 0; i < 3; ++i)
  {
    locsys(i, 0) = fiber_rad(i);
    locsys(i, 1) = fiber_axi(i);
    locsys(i, 2) = fiber_cir(i);
  }
}

void Mat::Elastic::Summand::evaluate_first_derivatives_aniso(Core::LinAlg::Matrix<2, 1>& dPI_aniso,
    Core::LinAlg::Matrix<3, 3> const& rcg, int gp, int eleGID)
{
  bool isoprinc, isomod, anisoprinc, anisomod, viscogeneral;
  specify_formulation(isoprinc, isomod, anisoprinc, anisomod, viscogeneral);
  if (anisoprinc or anisomod)
  {
    FOUR_C_THROW(
        "This anisotropic material does not support the first derivative of the free-energy "
        "function with respect to the anisotropic invariants. You need to implement them.");
  }
}

void Mat::Elastic::Summand::evaluate_second_derivatives_aniso(
    Core::LinAlg::Matrix<3, 1>& ddPII_aniso, Core::LinAlg::Matrix<3, 3> const& rcg, int gp,
    int eleGID)
{
  bool isoprinc, isomod, anisoprinc, anisomod, viscogeneral;
  specify_formulation(isoprinc, isomod, anisoprinc, anisomod, viscogeneral);
  if (anisoprinc or anisomod)
  {
    FOUR_C_THROW(
        "This anisotropic material does not support the second derivative of the free-energy "
        "function with respect to the anisotropic invariants. You need to implement them.");
  }
}
FOUR_C_NAMESPACE_CLOSE

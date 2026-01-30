// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_material_factory.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_aaaneohooke.hpp"
#include "4C_mat_beam3r_plasticity.hpp"
#include "4C_mat_beam_elasthyper_parameter.hpp"
#include "4C_mat_carreauyasuda.hpp"
#include "4C_mat_cnst_1d_art.hpp"
#include "4C_mat_constraintmixture.hpp"
#include "4C_mat_crosslinkermat.hpp"
#include "4C_mat_damage.hpp"
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
#include "4C_mat_elasthyper.hpp"
#include "4C_mat_elchmat.hpp"
#include "4C_mat_elchphase.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_mat_fluid_linear_density_viscosity.hpp"
#include "4C_mat_fluid_murnaghantait.hpp"
#include "4C_mat_fluid_weakly_compressible.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_mat_fluidporo_multiphase_reactions.hpp"
#include "4C_mat_fluidporo_multiphase_singlereaction.hpp"
#include "4C_mat_fluidporo_relpermeability_law.hpp"
#include "4C_mat_fluidporo_singlephase.hpp"
#include "4C_mat_fluidporo_singlephaseDof.hpp"
#include "4C_mat_fluidporo_singlephaselaw.hpp"
#include "4C_mat_fluidporo_viscosity_law.hpp"
#include "4C_mat_fourier.hpp"
#include "4C_mat_growthremodel_elasthyper.hpp"
#include "4C_mat_herschelbulkley.hpp"
#include "4C_mat_inelastic_defgrad_factors.hpp"
#include "4C_mat_ion.hpp"
#include "4C_mat_iterative_prestress.hpp"
#include "4C_mat_lin_elast_1D.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_list_chemoreac.hpp"
#include "4C_mat_list_chemotaxis.hpp"
#include "4C_mat_list_reactions.hpp"
#include "4C_mat_lubrication_law.hpp"
#include "4C_mat_lubrication_mat.hpp"
#include "4C_mat_maxwell_0d_acinus.hpp"
#include "4C_mat_maxwell_0d_acinus_DoubleExponential.hpp"
#include "4C_mat_maxwell_0d_acinus_Exponential.hpp"
#include "4C_mat_maxwell_0d_acinus_NeoHookean.hpp"
#include "4C_mat_maxwell_0d_acinus_Ogden.hpp"
#include "4C_mat_membrane_active_strain.hpp"
#include "4C_mat_membrane_elasthyper.hpp"
#include "4C_mat_micromaterial.hpp"
#include "4C_mat_mixture.hpp"
#include "4C_mat_modpowerlaw.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper.hpp"
#include "4C_mat_muscle_combo.hpp"
#include "4C_mat_muscle_giantesio.hpp"
#include "4C_mat_muscle_weickenmeier.hpp"
#include "4C_mat_myocard.hpp"
#include "4C_mat_newman.hpp"
#include "4C_mat_newman_multiscale.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_particle_dem.hpp"
#include "4C_mat_particle_pd.hpp"
#include "4C_mat_particle_sph_boundary.hpp"
#include "4C_mat_particle_sph_fluid.hpp"
#include "4C_mat_particle_wall_dem.hpp"
#include "4C_mat_plastic_VarConstUpdate.hpp"
#include "4C_mat_plasticdruckerprager.hpp"
#include "4C_mat_plasticelasthyper.hpp"
#include "4C_mat_plasticgtn.hpp"
#include "4C_mat_plasticlinelast.hpp"
#include "4C_mat_plasticnlnlogneohooke.hpp"
#include "4C_mat_poro_density_law.hpp"
#include "4C_mat_poro_law.hpp"
#include "4C_mat_robinson.hpp"
#include "4C_mat_scalardepinterp.hpp"
#include "4C_mat_scatra.hpp"
#include "4C_mat_scatra_chemotaxis.hpp"
#include "4C_mat_scatra_multiporo.hpp"
#include "4C_mat_scatra_multiscale.hpp"
#include "4C_mat_scatra_poro_ecm.hpp"
#include "4C_mat_scatra_reaction.hpp"
#include "4C_mat_scl.hpp"
#include "4C_mat_shell_kl.hpp"
#include "4C_mat_soret.hpp"
#include "4C_mat_spring.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_mat_structporo_reaction.hpp"
#include "4C_mat_structporo_reaction_ecm.hpp"
#include "4C_mat_stvenantkirchhoff.hpp"
#include "4C_mat_stvenantkirchhoff_orthotropic.hpp"
#include "4C_mat_superelastic_sma.hpp"
#include "4C_mat_sutherland.hpp"
#include "4C_mat_thermoplastichyperelast.hpp"
#include "4C_mat_thermoplasticlinelast.hpp"
#include "4C_mat_thermostvenantkirchhoff.hpp"
#include "4C_mat_viscoanisotropic.hpp"
#include "4C_mat_viscoelasthyper.hpp"
#include "4C_mat_visconeohooke.hpp"
#include "4C_mat_viscoplastic_no_yield_surface.hpp"
#include "4C_mat_vplast_reform_johnsoncook.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_constituent_elasthyper.hpp"
#include "4C_mixture_constituent_elasthyper_damage.hpp"
#include "4C_mixture_constituent_elasthyper_elastin_membrane.hpp"
#include "4C_mixture_constituent_full_constrained_mixture_fiber.hpp"
#include "4C_mixture_constituent_remodelfiber_expl.hpp"
#include "4C_mixture_constituent_remodelfiber_impl.hpp"
#include "4C_mixture_constituent_remodelfiber_material_exponential.hpp"
#include "4C_mixture_constituent_remodelfiber_material_exponential_active.hpp"
#include "4C_mixture_constituent_solidmaterial.hpp"
#include "4C_mixture_growth_strategy_anisotropic.hpp"
#include "4C_mixture_growth_strategy_isotropic.hpp"
#include "4C_mixture_growth_strategy_stiffness.hpp"
#include "4C_mixture_prestress_strategy_isocyl.hpp"
#include "4C_mixture_prestress_strategy_iterative.hpp"
#include "4C_mixture_prestress_strategy_prescribed.hpp"
#include "4C_mixture_rule_function.hpp"
#include "4C_mixture_rule_growthremodel.hpp"
#include "4C_mixture_rule_simple.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN


namespace
{
  template <typename MaterialParameter>
  std::unique_ptr<Core::Mat::PAR::Parameter> make_parameter_impl(int id,
      Core::Materials::MaterialType type, const Core::IO::InputParameterContainer& input_data)
  {
    static_assert(std::is_base_of_v<Core::Mat::PAR::Parameter, MaterialParameter>);
    return std::make_unique<MaterialParameter>(
        Core::Mat::PAR::Parameter::Data{.id = id, .type = type, .parameters = input_data});
  }
}  // namespace

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::factory(int matnum)
{
  // retrieve problem instance to read from
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();

  // for the sake of safety
  if (Global::Problem::instance(probinst)->materials() == nullptr)
    FOUR_C_THROW("List of materials cannot be accessed in the global problem instance.");
  // yet another safety check
  if (Global::Problem::instance(probinst)->materials()->num() == 0)
    FOUR_C_THROW("List of materials in the global problem instance is empty.");

  // retrieve validated input line of material ID in question
  auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(matnum);
  return curmat->create_material();
}

std::unique_ptr<Core::Mat::PAR::Parameter> Mat::make_parameter(
    int id, Core::Materials::MaterialType type, const Core::IO::InputParameterContainer& input_data)
{
  switch (type)
  {
    case Core::Materials::m_fluid:
    {
      return make_parameter_impl<Mat::PAR::NewtonianFluid>(id, type, input_data);
    }
    case Core::Materials::m_fluid_murnaghantait:
    {
      return make_parameter_impl<Mat::PAR::MurnaghanTaitFluid>(id, type, input_data);
    }
    case Core::Materials::m_fluid_linear_density_viscosity:
    {
      return make_parameter_impl<Mat::PAR::LinearDensityViscosity>(id, type, input_data);
    }
    case Core::Materials::m_fluid_weakly_compressible:
    {
      return make_parameter_impl<Mat::PAR::WeaklyCompressibleFluid>(id, type, input_data);
    }
    case Core::Materials::m_stvenant:
    {
      return make_parameter_impl<Mat::PAR::StVenantKirchhoff>(id, type, input_data);
    }
    case Core::Materials::m_orthostvenant:
    {
      return make_parameter_impl<Mat::PAR::StVenantKirchhoffOrthotropic>(id, type, input_data);
    }
    case Core::Materials::m_thermostvenant:
    {
      return make_parameter_impl<Mat::PAR::ThermoStVenantKirchhoff>(id, type, input_data);
    }
    case Core::Materials::m_thermopllinelast:
    {
      return make_parameter_impl<Mat::PAR::ThermoPlasticLinElast>(id, type, input_data);
    }
    case Core::Materials::m_pldruckprag:
    {
      return make_parameter_impl<Mat::PAR::PlasticDruckerPrager>(id, type, input_data);
    }
    case Core::Materials::m_plgtn:
    {
      return make_parameter_impl<Mat::PAR::PlasticGTN>(id, type, input_data);
    }
    case Core::Materials::m_thermoplhyperelast:
    {
      return make_parameter_impl<Mat::PAR::ThermoPlasticHyperElast>(id, type, input_data);
    }
    case Core::Materials::m_plnlnlogneohooke:
    {
      return make_parameter_impl<Mat::PAR::PlasticNlnLogNeoHooke>(id, type, input_data);
    }
    case Core::Materials::m_pllinelast:
    {
      return make_parameter_impl<Mat::PAR::PlasticLinElast>(id, type, input_data);
    }
    case Core::Materials::m_vp_no_yield_surface:
    {
      return make_parameter_impl<Mat::PAR::ViscoPlasticNoYieldSurface>(id, type, input_data);
    }
    case Core::Materials::m_vp_robinson:
    {
      return make_parameter_impl<Mat::PAR::Robinson>(id, type, input_data);
    }
    case Core::Materials::m_elpldamage:
    {
      return make_parameter_impl<Mat::PAR::Damage>(id, type, input_data);
    }
    case Core::Materials::m_struct_multiscale:
    {
      return make_parameter_impl<Mat::PAR::MicroMaterial>(id, type, input_data);
    }
    case Core::Materials::m_visconeohooke:
    {
      return make_parameter_impl<Mat::PAR::ViscoNeoHooke>(id, type, input_data);
    }
    case Core::Materials::m_viscoanisotropic:
    {
      return make_parameter_impl<Mat::PAR::ViscoAnisotropic>(id, type, input_data);
    }
    case Core::Materials::m_aaaneohooke:
    {
      return make_parameter_impl<Mat::PAR::AAAneohooke>(id, type, input_data);
    }
    case Core::Materials::m_lubrication:
    {
      return make_parameter_impl<Mat::PAR::LubricationMat>(id, type, input_data);
    }
    case Core::Materials::m_lubrication_law_constant:
    {
      return make_parameter_impl<Mat::PAR::LubricationLawConstant>(id, type, input_data);
    }
    case Core::Materials::m_lubrication_law_barus:
    {
      return make_parameter_impl<Mat::PAR::LubricationLawBarus>(id, type, input_data);
    }
    case Core::Materials::m_lubrication_law_roeland:
    {
      return make_parameter_impl<Mat::PAR::LubricationLawRoeland>(id, type, input_data);
    }
    case Core::Materials::m_scatra:
    {
      return make_parameter_impl<Mat::PAR::ScatraMat>(id, type, input_data);
    }
    case Core::Materials::m_scatra_reaction_poroECM:
    {
      return make_parameter_impl<Mat::PAR::ScatraMatPoroECM>(id, type, input_data);
    }
    case Core::Materials::m_scatra_reaction:
    {
      return make_parameter_impl<Mat::PAR::ScatraReactionMat>(id, type, input_data);
    }
    case Core::Materials::m_scatra_in_fluid_porofluid_pressure_based:
    {
      return make_parameter_impl<Mat::PAR::ScatraMatMultiPoroFluid>(id, type, input_data);
    }
    case Core::Materials::m_scatra_in_volfrac_porofluid_pressure_based:
    {
      return make_parameter_impl<Mat::PAR::ScatraMatMultiPoroVolFrac>(id, type, input_data);
    }
    case Core::Materials::m_scatra_in_solid_porofluid_pressure_based:
    {
      return make_parameter_impl<Mat::PAR::ScatraMatMultiPoroSolid>(id, type, input_data);
    }
    case Core::Materials::m_scatra_as_temperature_porofluid_pressure_based:
    {
      return make_parameter_impl<Mat::PAR::ScatraMatMultiPoroTemperature>(id, type, input_data);
    }
    case Core::Materials::m_scatra_multiscale:
    {
      return make_parameter_impl<Mat::PAR::ScatraMultiScale>(id, type, input_data);
    }
    case Core::Materials::m_scatra_chemotaxis:
    {
      return make_parameter_impl<Mat::PAR::ScatraChemotaxisMat>(id, type, input_data);
    }
    case Core::Materials::m_muscle_combo:
    {
      return make_parameter_impl<Mat::PAR::MuscleCombo>(id, type, input_data);
    }
    case Core::Materials::m_muscle_giantesio:
    {
      return make_parameter_impl<Mat::PAR::MuscleGiantesio>(id, type, input_data);
    }
    case Core::Materials::m_muscle_weickenmeier:
    {
      return make_parameter_impl<Mat::PAR::MuscleWeickenmeier>(id, type, input_data);
    }
    case Core::Materials::m_myocard:
    {
      return make_parameter_impl<Mat::PAR::Myocard>(id, type, input_data);
    }
    case Core::Materials::m_sutherland:
    {
      return make_parameter_impl<Mat::PAR::Sutherland>(id, type, input_data);
    }
    case Core::Materials::m_carreauyasuda:
    {
      return make_parameter_impl<Mat::PAR::CarreauYasuda>(id, type, input_data);
    }
    case Core::Materials::m_modpowerlaw:
    {
      return make_parameter_impl<Mat::PAR::ModPowerLaw>(id, type, input_data);
    }
    case Core::Materials::m_herschelbulkley:
    {
      return make_parameter_impl<Mat::PAR::HerschelBulkley>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo:
    {
      return make_parameter_impl<Mat::PAR::FluidPoro>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_multiphase:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroMultiPhase>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_multiphase_reactions:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroMultiPhaseReactions>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_singlereaction:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroSingleReaction>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_singlephase:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroSinglePhase>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_singlevolfrac:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroSingleVolFrac>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_volfracpressure:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroVolFracPressure>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_volfrac_pressure_blood_lung:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroVolFracPressureBloodLung>(id, type, input_data);
    }
    case Core::Materials::m_poro_law_linear:
    {
      return make_parameter_impl<Mat::PAR::PoroLawLinear>(id, type, input_data);
    }
    case Core::Materials::m_poro_law_constant:
    {
      return make_parameter_impl<Mat::PAR::PoroLawConstant>(id, type, input_data);
    }
    case Core::Materials::m_poro_law_logNeoHooke_Penalty:
    {
      return make_parameter_impl<Mat::PAR::PoroLawNeoHooke>(id, type, input_data);
    }
    case Core::Materials::m_poro_law_incompr_skeleton:
    {
      return make_parameter_impl<Mat::PAR::PoroLawIncompSkeleton>(id, type, input_data);
    }
    case Core::Materials::m_poro_law_linear_biot:
    {
      return make_parameter_impl<Mat::PAR::PoroLawLinBiot>(id, type, input_data);
    }
    case Core::Materials::m_poro_law_density_dependent:
    {
      return make_parameter_impl<Mat::PAR::PoroLawDensityDependent>(id, type, input_data);
    }
    case Core::Materials::m_poro_densitylaw_constant:
    {
      return make_parameter_impl<Mat::PAR::PoroDensityLawConstant>(id, type, input_data);
    }
    case Core::Materials::m_poro_densitylaw_exp:
    {
      return make_parameter_impl<Mat::PAR::PoroDensityLawExp>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_phaselaw_linear:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroPhaseLawLinear>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_phaselaw_tangent:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroPhaseLawTangent>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_phaselaw_constraint:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroPhaseLawConstraint>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_phaselaw_byfunction:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroPhaseLawByFunction>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_relpermeabilitylaw_constant:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroRelPermeabilityLawConstant>(
          id, type, input_data);
    }
    case Core::Materials::m_fluidporo_relpermeabilitylaw_exp:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroRelPermeabilityLawExponent>(
          id, type, input_data);
    }
    case Core::Materials::m_fluidporo_viscositylaw_constant:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroViscosityLawConstant>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_viscositylaw_celladh:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroViscosityLawCellAdherence>(
          id, type, input_data);
    }
    case Core::Materials::m_fluidporo_phasedof_diffpressure:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroPhaseDofDiffPressure>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_phasedof_pressure:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroPhaseDofPressure>(id, type, input_data);
    }
    case Core::Materials::m_fluidporo_phasedof_saturation:
    {
      return make_parameter_impl<Mat::PAR::FluidPoroPhaseDofSaturation>(id, type, input_data);
    }
    case Core::Materials::m_matlist:
    {
      return make_parameter_impl<Mat::PAR::MatList>(id, type, input_data);
    }
    case Core::Materials::m_matlist_reactions:
    {
      // Note: We need to do a dynamic_cast here since Chemotaxis, Reaction, and Chemo-reaction are
      // in a diamond inheritance structure
      return make_parameter_impl<Mat::PAR::MatListReactions>(id, type, input_data);
    }
    case Core::Materials::m_matlist_chemotaxis:
    {
      // Note: We need to do a dynamic_cast here since Chemotaxis, Reaction, and Chemo-reaction are
      // in a diamond inheritance structure
      return make_parameter_impl<Mat::PAR::MatListChemotaxis>(id, type, input_data);
    }
    case Core::Materials::m_matlist_chemoreac:
    {
      // Note: We need to do a dynamic_cast here since Chemotaxis, Reaction, and Chemo-reaction are
      // in a diamond inheritance structure
      return make_parameter_impl<Mat::PAR::MatListChemoReac>(id, type, input_data);
    }
    case Core::Materials::m_elchmat:
    {
      return make_parameter_impl<Mat::PAR::ElchMat>(id, type, input_data);
    }
    case Core::Materials::m_elchphase:
    {
      return make_parameter_impl<Mat::PAR::ElchPhase>(id, type, input_data);
    }
    case Core::Materials::m_ion:
    {
      return make_parameter_impl<Mat::PAR::Ion>(id, type, input_data);
    }
    case Core::Materials::m_electrode:
    {
      return make_parameter_impl<Mat::PAR::Electrode>(id, type, input_data);
    }
    case Core::Materials::m_newman:
    {
      return make_parameter_impl<Mat::PAR::Newman>(id, type, input_data);
    }
    case Core::Materials::m_newman_multiscale:
    {
      return make_parameter_impl<Mat::PAR::NewmanMultiScale>(id, type, input_data);
    }
    case Core::Materials::m_scl:
    {
      return make_parameter_impl<Mat::PAR::Scl>(id, type, input_data);
    }
    case Core::Materials::m_elasthyper:
    {
      return make_parameter_impl<Mat::PAR::ElastHyper>(id, type, input_data);
    }
    case Core::Materials::m_viscoelasthyper:
    {
      return make_parameter_impl<Mat::PAR::ViscoElastHyper>(id, type, input_data);
    }
    case Core::Materials::m_plelasthyper:
    {
      return make_parameter_impl<Mat::PAR::PlasticElastHyper>(id, type, input_data);
    }
    case Core::Materials::m_plelasthyperVCU:
    {
      return make_parameter_impl<Mat::PAR::PlasticElastHyperVCU>(id, type, input_data);
    }
    case Core::Materials::m_sc_dep_interp:
    {
      return make_parameter_impl<Mat::PAR::ScalarDepInterp>(id, type, input_data);
    }
    case Core::Materials::m_structporo:
    {
      return make_parameter_impl<Mat::PAR::StructPoro>(id, type, input_data);
    }
    case Core::Materials::m_structpororeaction:
    {
      return make_parameter_impl<Mat::PAR::StructPoroReaction>(id, type, input_data);
    }
    case Core::Materials::m_structpororeactionECM:
    {
      return make_parameter_impl<Mat::PAR::StructPoroReactionECM>(id, type, input_data);
    }
    case Core::Materials::mfi_no_growth:
    {
      return make_parameter_impl<Mat::PAR::InelasticDefgradNoGrowth>(id, type, input_data);
    }
    case Core::Materials::mfi_lin_scalar_aniso:
    {
      return make_parameter_impl<Mat::PAR::InelasticDefgradLinScalarAniso>(id, type, input_data);
    }
    case Core::Materials::mfi_lin_scalar_iso:
    {
      return make_parameter_impl<Mat::PAR::InelasticDefgradLinScalar>(id, type, input_data);
    }
    case Core::Materials::mfi_poly_intercal_frac_aniso:
    {
      return make_parameter_impl<Mat::PAR::InelasticDefgradPolyIntercalFracAniso>(
          id, type, input_data);
    }
    case Core::Materials::mfi_poly_intercal_frac_iso:
    {
      return make_parameter_impl<Mat::PAR::InelasticDefgradPolyIntercalFrac>(id, type, input_data);
    }
    case Core::Materials::mfi_lin_temp_iso:
    {
      return make_parameter_impl<Mat::PAR::InelasticDefgradLinTempIso>(id, type, input_data);
    }
    case Core::Materials::mfi_time_funct:
    {
      return make_parameter_impl<Mat::PAR::InelasticDefgradTimeFunct>(id, type, input_data);
    }
    case Core::Materials::mfi_transv_isotrop_elast_viscoplast:
    {
      return make_parameter_impl<Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast>(
          id, type, input_data);
    }
    case Core::Materials::mvl_reformulated_Johnson_Cook:
    {
      return make_parameter_impl<Mat::Viscoplastic::PAR::ReformulatedJohnsonCook>(
          id, type, input_data);
    }
    case Core::Materials::mix_rule_function:
    {
      return make_parameter_impl<FourC::Mixture::PAR::FunctionMixtureRule>(id, type, input_data);
    }
    case Core::Materials::mix_rule_simple:
    {
      return make_parameter_impl<FourC::Mixture::PAR::SimpleMixtureRule>(id, type, input_data);
    }
    case Core::Materials::mix_rule_growthremodel:
    {
      return make_parameter_impl<FourC::Mixture::PAR::GrowthRemodelMixtureRule>(
          id, type, input_data);
    }
    case Core::Materials::mix_elasthyper:
    {
      return make_parameter_impl<FourC::Mixture::PAR::MixtureConstituentElastHyper>(
          id, type, input_data);
    }
    case Core::Materials::mix_elasthyper_damage:
    {
      return make_parameter_impl<FourC::Mixture::PAR::MixtureConstituentElastHyperDamage>(
          id, type, input_data);
    }
    case Core::Materials::mix_elasthyper_elastin_membrane:
    {
      return make_parameter_impl<FourC::Mixture::PAR::MixtureConstituentElastHyperElastinMembrane>(
          id, type, input_data);
    }
    case Core::Materials::mix_remodelfiber_expl:
    {
      return make_parameter_impl<FourC::Mixture::PAR::MixtureConstituentRemodelFiberExpl>(
          id, type, input_data);
    }
    case Core::Materials::mix_full_constrained_mixture_fiber:
    {
      return make_parameter_impl<
          FourC::Mixture::PAR::MixtureConstituentFullConstrainedMixtureFiber>(id, type, input_data);
    }
    case Core::Materials::mix_remodelfiber_impl:
    {
      return make_parameter_impl<FourC::Mixture::PAR::MixtureConstituentRemodelFiberImpl>(
          id, type, input_data);
    }
    case Core::Materials::mix_solid_material:
    {
      return make_parameter_impl<FourC::Mixture::PAR::MixtureConstituentSolidMaterial>(
          id, type, input_data);
    }
    case Core::Materials::mix_growth_strategy_isotropic:
    {
      return make_parameter_impl<FourC::Mixture::PAR::IsotropicGrowthStrategy>(
          id, type, input_data);
    }
    case Core::Materials::mix_growth_strategy_anisotropic:
    {
      return make_parameter_impl<FourC::Mixture::PAR::AnisotropicGrowthStrategy>(
          id, type, input_data);
    }
    case Core::Materials::mix_growth_strategy_stiffness:
    {
      return make_parameter_impl<FourC::Mixture::PAR::StiffnessGrowthStrategy>(
          id, type, input_data);
    }
    case Core::Materials::mix_prestress_strategy_cylinder:
    {
      return make_parameter_impl<FourC::Mixture::PAR::IsotropicCylinderPrestressStrategy>(
          id, type, input_data);
    }
    case Core::Materials::mix_prestress_strategy_iterative:
    {
      return make_parameter_impl<FourC::Mixture::PAR::IterativePrestressStrategy>(
          id, type, input_data);
    }
    case Core::Materials::mix_prestress_strategy_prescribed:
    {
      return make_parameter_impl<FourC::Mixture::PAR::PrescribedPrestressStrategy>(
          id, type, input_data);
    }
    case Core::Materials::m_iterative_prestress:
    {
      return make_parameter_impl<Mat::PAR::IterativePrestressMaterial>(id, type, input_data);
    }
    case Core::Materials::mix_remodelfiber_material_exponential:
      return make_parameter_impl<FourC::Mixture::PAR::RemodelFiberMaterialExponential<double>>(
          id, type, input_data);
    case Core::Materials::mix_remodelfiber_material_exponential_active:
      return make_parameter_impl<
          FourC::Mixture::PAR::RemodelFiberMaterialExponentialActive<double>>(id, type, input_data);
    case Core::Materials::mes_anisoactivestress_evolution:
    {
      return make_parameter_impl<Mat::Elastic::PAR::AnisoActiveStressEvolution>(
          id, type, input_data);
    }
    case Core::Materials::mes_coupanisoexpoactive:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupAnisoExpoActive>(id, type, input_data);
    }
    case Core::Materials::mes_coupanisoexpo:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupAnisoExpo>(id, type, input_data);
    }
    case Core::Materials::mes_coupanisoexposhear:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupAnisoExpoShear>(id, type, input_data);
    }
    case Core::Materials::mes_coupanisoexpotwocoup:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupAnisoExpoTwoCoup>(id, type, input_data);
    }
    case Core::Materials::mes_coupanisoneohooke:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupAnisoNeoHooke>(id, type, input_data);
    }
    case Core::Materials::mes_coupanisoneohooke_varprop:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupAnisoNeoHookeVarProp>(id, type, input_data);
    }
    case Core::Materials::mes_coupanisopow:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupAnisoPow>(id, type, input_data);
    }
    case Core::Materials::mes_coupblatzko:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupBlatzKo>(id, type, input_data);
    }
    case Core::Materials::mes_coupexppol:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupExpPol>(id, type, input_data);
    }
    case Core::Materials::mes_couplogneohooke:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupLogNeoHooke>(id, type, input_data);
    }
    case Core::Materials::mes_couplogmixneohooke:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupLogMixNeoHooke>(id, type, input_data);
    }
    case Core::Materials::mes_coupmooneyrivlin:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupMooneyRivlin>(id, type, input_data);
    }
    case Core::Materials::mes_coupmyocard:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupMyocard>(id, type, input_data);
    }
    case Core::Materials::mes_coupneohooke:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupNeoHooke>(id, type, input_data);
    }
    case Core::Materials::mes_coup1pow:
    {
      return make_parameter_impl<Mat::Elastic::PAR::Coup1Pow>(id, type, input_data);
    }
    case Core::Materials::mes_coup2pow:
    {
      return make_parameter_impl<Mat::Elastic::PAR::Coup2Pow>(id, type, input_data);
    }
    case Core::Materials::mes_coup3pow:
    {
      return make_parameter_impl<Mat::Elastic::PAR::Coup3Pow>(id, type, input_data);
    }
    case Core::Materials::mes_coup13apow:
    {
      return make_parameter_impl<Mat::Elastic::PAR::Coup13aPow>(id, type, input_data);
    }
    case Core::Materials::mes_coupsimopister:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupSimoPister>(id, type, input_data);
    }
    case Core::Materials::mes_coupSVK:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupSVK>(id, type, input_data);
    }
    case Core::Materials::mes_couptransverselyisotropic:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupTransverselyIsotropic>(
          id, type, input_data);
    }
    case Core::Materials::mes_coupvarga:
    {
      return make_parameter_impl<Mat::Elastic::PAR::CoupVarga>(id, type, input_data);
    }
    case Core::Materials::mes_fract:
    {
      return make_parameter_impl<Mat::Elastic::PAR::Fract>(id, type, input_data);
    }
    case Core::Materials::mes_genmax:
    {
      return make_parameter_impl<Mat::Elastic::PAR::GenMax>(id, type, input_data);
    }
    case Core::Materials::mes_generalizedgenmax:
    {
      return make_parameter_impl<Mat::Elastic::PAR::GeneralizedGenMax>(id, type, input_data);
    }
    case Core::Materials::mes_isoanisoexpo:
    {
      return make_parameter_impl<Mat::Elastic::PAR::IsoAnisoExpo>(id, type, input_data);
    }
    case Core::Materials::mes_isoexpopow:
    {
      return make_parameter_impl<Mat::Elastic::PAR::IsoExpoPow>(id, type, input_data);
    }
    case Core::Materials::mes_isomooneyrivlin:
    {
      return make_parameter_impl<Mat::Elastic::PAR::IsoMooneyRivlin>(id, type, input_data);
    }
    case Core::Materials::mes_isomuscleblemker:
    {
      return make_parameter_impl<Mat::Elastic::PAR::IsoMuscleBlemker>(id, type, input_data);
    }
    case Core::Materials::mes_isoneohooke:
    {
      return make_parameter_impl<Mat::Elastic::PAR::IsoNeoHooke>(id, type, input_data);
    }
    case Core::Materials::mes_isoogden:
    {
      return make_parameter_impl<Mat::Elastic::PAR::IsoOgden>(id, type, input_data);
    }
    case Core::Materials::mes_iso1pow:
    {
      return make_parameter_impl<Mat::Elastic::PAR::Iso1Pow>(id, type, input_data);
    }
    case Core::Materials::mes_iso2pow:
    {
      return make_parameter_impl<Mat::Elastic::PAR::Iso2Pow>(id, type, input_data);
    }
    case Core::Materials::mes_isoratedep:
    {
      return make_parameter_impl<Mat::Elastic::PAR::IsoRateDep>(id, type, input_data);
    }
    case Core::Materials::mes_isotestmaterial:
    {
      return make_parameter_impl<Mat::Elastic::PAR::IsoTestMaterial>(id, type, input_data);
    }
    case Core::Materials::mes_isovarga:
    {
      return make_parameter_impl<Mat::Elastic::PAR::IsoVarga>(id, type, input_data);
    }
    case Core::Materials::mes_isoyeoh:
    {
      return make_parameter_impl<Mat::Elastic::PAR::IsoYeoh>(id, type, input_data);
    }
    case Core::Materials::mes_remodelfiber:
    {
      return make_parameter_impl<Mat::Elastic::PAR::RemodelFiber>(id, type, input_data);
    }
    case Core::Materials::mes_vologden:
    {
      return make_parameter_impl<Mat::Elastic::PAR::VolOgden>(id, type, input_data);
    }
    case Core::Materials::mes_volpenalty:
    {
      return make_parameter_impl<Mat::Elastic::PAR::VolPenalty>(id, type, input_data);
    }
    case Core::Materials::mes_volpow:
    {
      return make_parameter_impl<Mat::Elastic::PAR::VolPow>(id, type, input_data);
    }
    case Core::Materials::mes_volsussmanbathe:
    {
      return make_parameter_impl<Mat::Elastic::PAR::VolSussmanBathe>(id, type, input_data);
    }
    case Core::Materials::mes_viscobranch:
    {
      return make_parameter_impl<Mat::Elastic::PAR::ViscoBranch>(id, type, input_data);
    }
    case Core::Materials::mes_viscopart:
    {
      return make_parameter_impl<Mat::Elastic::PAR::ViscoPart>(id, type, input_data);
    }
    case Core::Materials::mes_structuraltensorstratgy:
    {
      return make_parameter_impl<Mat::Elastic::PAR::StructuralTensorParameter>(
          id, type, input_data);
    }
    case Core::Materials::m_cnst_art:
    {
      return make_parameter_impl<Mat::PAR::Cnst1dArt>(id, type, input_data);
    }
    case Core::Materials::m_0d_maxwell_acinus:
    {
      return make_parameter_impl<Mat::PAR::Maxwell0dAcinus>(id, type, input_data);
    }
    case Core::Materials::m_0d_maxwell_acinus_neohookean:
    {
      return make_parameter_impl<Mat::PAR::Maxwell0dAcinusNeoHookean>(id, type, input_data);
    }
    case Core::Materials::m_0d_maxwell_acinus_exponential:
    {
      return make_parameter_impl<Mat::PAR::Maxwell0dAcinusExponential>(id, type, input_data);
    }
    case Core::Materials::m_0d_maxwell_acinus_doubleexponential:
    {
      return make_parameter_impl<Mat::PAR::Maxwell0dAcinusDoubleExponential>(id, type, input_data);
    }
    case Core::Materials::m_0d_maxwell_acinus_ogden:
    {
      return make_parameter_impl<Mat::PAR::Maxwell0dAcinusOgden>(id, type, input_data);
    }
    case Core::Materials::m_thermo_fourier:
    {
      return make_parameter_impl<Mat::PAR::Fourier>(id, type, input_data);
    }
    case Core::Materials::m_soret:
    {
      return make_parameter_impl<Mat::PAR::Soret>(id, type, input_data);
    }
    case Core::Materials::m_membrane_elasthyper:
    {
      return make_parameter_impl<Mat::PAR::MembraneElastHyper>(id, type, input_data);
    }
    case Core::Materials::m_membrane_activestrain:
    {
      return make_parameter_impl<Mat::PAR::MembraneActiveStrain>(id, type, input_data);
    }
    case Core::Materials::m_growthremodel_elasthyper:
    {
      return make_parameter_impl<Mat::PAR::GrowthRemodelElastHyper>(id, type, input_data);
    }
    case Core::Materials::m_mixture:
    {
      return make_parameter_impl<Mat::PAR::Mixture>(id, type, input_data);
    }
    case Core::Materials::m_multiplicative_split_defgrad_elasthyper:
    {
      return make_parameter_impl<Mat::PAR::MultiplicativeSplitDefgradElastHyper>(
          id, type, input_data);
    }
    case Core::Materials::m_constraintmixture:
    {
      return make_parameter_impl<Mat::PAR::ConstraintMixture>(id, type, input_data);
    }
    case Core::Materials::m_beam_reissner_elast_hyper:
    {
      return make_parameter_impl<Mat::PAR::BeamReissnerElastHyperMaterialParams>(
          id, type, input_data);
    }
    case Core::Materials::m_beam_reissner_elast_plastic:
    {
      return make_parameter_impl<Mat::PAR::BeamReissnerElastPlasticMaterialParams>(
          id, type, input_data);
    }
    case Core::Materials::m_beam_reissner_elast_hyper_bymodes:
    {
      return make_parameter_impl<Mat::PAR::BeamReissnerElastHyperMaterialParamsByMode>(
          id, type, input_data);
    }
    case Core::Materials::m_beam_kirchhoff_elast_hyper:
    {
      return make_parameter_impl<Mat::PAR::BeamKirchhoffElastHyperMaterialParams>(
          id, type, input_data);
    }
    case Core::Materials::m_beam_kirchhoff_elast_hyper_bymodes:
    {
      return make_parameter_impl<Mat::PAR::BeamKirchhoffElastHyperMaterialParamsByMode>(
          id, type, input_data);
    }
    case Core::Materials::m_beam_kirchhoff_torsionfree_elast_hyper:
    {
      return make_parameter_impl<Mat::PAR::BeamKirchhoffTorsionFreeElastHyperMaterialParams>(
          id, type, input_data);
    }
    case Core::Materials::m_beam_kirchhoff_torsionfree_elast_hyper_bymodes:
    {
      return make_parameter_impl<Mat::PAR::BeamKirchhoffTorsionFreeElastHyperMaterialParamsByMode>(
          id, type, input_data);
    }
    case Core::Materials::m_shell_kirchhoff_love:
    {
      return make_parameter_impl<Mat::PAR::KirchhoffLoveShell>(id, type, input_data);
    }
    case Core::Materials::m_crosslinkermat:
    {
      return make_parameter_impl<Mat::PAR::CrosslinkerMat>(id, type, input_data);
    }
    case Core::Materials::m_spring:
    {
      return make_parameter_impl<Mat::PAR::Spring>(id, type, input_data);
    }
    case Core::Materials::m_particle_sph_fluid:
    {
      // note: dynamic_cast needed due diamond inheritance structure
      return make_parameter_impl<Mat::PAR::ParticleMaterialSPHFluid>(id, type, input_data);
    }
    case Core::Materials::m_particle_sph_boundary:
    {
      // note: dynamic_cast needed due diamond inheritance structure
      return make_parameter_impl<Mat::PAR::ParticleMaterialSPHBoundary>(id, type, input_data);
    }
    case Core::Materials::m_particle_dem:
    {
      // note: dynamic_cast needed due diamond inheritance structure
      return make_parameter_impl<Mat::PAR::ParticleMaterialDEM>(id, type, input_data);
    }
    case Core::Materials::m_particle_wall_dem:
    {
      return make_parameter_impl<Mat::PAR::ParticleWallMaterialDEM>(id, type, input_data);
    }
    case Core::Materials::m_particle_pd:
    {
      return make_parameter_impl<Mat::PAR::ParticleMaterialPD>(id, type, input_data);
    }
    case Core::Materials::m_superelast:
    {
      return make_parameter_impl<Mat::PAR::SuperElasticSMA>(id, type, input_data);
    }
    case Core::Materials::m_linelast1D:
    {
      return make_parameter_impl<Mat::PAR::LinElast1D>(id, type, input_data);
    }
    case Core::Materials::m_linelast1D_growth:
    {
      return make_parameter_impl<Mat::PAR::LinElast1DGrowth>(id, type, input_data);
    }
    default:
      FOUR_C_THROW("unknown material type {}", type);
  }
}

FOUR_C_NAMESPACE_CLOSE

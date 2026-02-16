// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_global_legacy_module.hpp"

#include "4C_ale_ale2.hpp"
#include "4C_ale_ale2_nurbs.hpp"
#include "4C_ale_ale3.hpp"
#include "4C_ale_ale3_nurbs.hpp"
#include "4C_art_net_artery.hpp"
#include "4C_beam3_euler_bernoulli.hpp"
#include "4C_beam3_kirchhoff.hpp"
#include "4C_beam3_reissner.hpp"
#include "4C_beaminteraction_crosslinking_link_beam3_reissner_line2_pinjointed.hpp"
#include "4C_beaminteraction_crosslinking_link_beam3_reissner_line2_rigidjointed.hpp"
#include "4C_beaminteraction_crosslinking_link_truss.hpp"
#include "4C_beaminteraction_crosslinking_node.hpp"
#include "4C_bele_bele3.hpp"
#include "4C_bele_vele3.hpp"
#include "4C_binstrategy_meshfree_multibin.hpp"
#include "4C_constraint_element2.hpp"
#include "4C_constraint_element3.hpp"
#include "4C_contact_constitutivelaw_valid_laws.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_node.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_fem_nurbs_discretization_control_point.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_fluid_ele_hdg.hpp"
#include "4C_fluid_ele_hdg_weak_comp.hpp"
#include "4C_fluid_ele_poro.hpp"
#include "4C_fluid_ele_xwall.hpp"
#include "4C_fluid_functions.hpp"
#include "4C_fluid_xfluid_functions.hpp"
#include "4C_fluid_xfluid_functions_combust.hpp"
#include "4C_global_legacy_module_validconditions.hpp"
#include "4C_global_legacy_module_validmaterials.hpp"
#include "4C_global_legacy_module_validparameters.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_lubrication_ele.hpp"
#include "4C_mat_aaaneohooke.hpp"
#include "4C_mat_beam_elasthyper.hpp"
#include "4C_mat_carreauyasuda.hpp"
#include "4C_mat_cnst_1d_art.hpp"
#include "4C_mat_constraintmixture.hpp"
#include "4C_mat_constraintmixture_history.hpp"
#include "4C_mat_crystal_plasticity.hpp"
#include "4C_mat_damage.hpp"
#include "4C_mat_elasthyper.hpp"
#include "4C_mat_elchmat.hpp"
#include "4C_mat_fluid_linear_density_viscosity.hpp"
#include "4C_mat_fluid_murnaghantait.hpp"
#include "4C_mat_fluid_weakly_compressible.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_mat_fluidporo_multiphase_reactions.hpp"
#include "4C_mat_fluidporo_multiphase_singlereaction.hpp"
#include "4C_mat_fluidporo_singlephase.hpp"
#include "4C_mat_fourier.hpp"
#include "4C_mat_growthremodel_elasthyper.hpp"
#include "4C_mat_herschelbulkley.hpp"
#include "4C_mat_ion.hpp"
#include "4C_mat_lin_elast_1D.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_list_chemoreac.hpp"
#include "4C_mat_list_chemotaxis.hpp"
#include "4C_mat_list_reactions.hpp"
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
#include "4C_mat_myocard.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_mat_plastic_VarConstUpdate.hpp"
#include "4C_mat_plasticelasthyper.hpp"
#include "4C_mat_plasticlinelast.hpp"
#include "4C_mat_robinson.hpp"
#include "4C_mat_scalardepinterp.hpp"
#include "4C_mat_scatra.hpp"
#include "4C_mat_scatra_multiporo.hpp"
#include "4C_mat_scatra_poro_ecm.hpp"
#include "4C_mat_shell_kl.hpp"
#include "4C_mat_spring.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_mat_structporo_reaction.hpp"
#include "4C_mat_structporo_reaction_ecm.hpp"
#include "4C_mat_stvenantkirchhoff.hpp"
#include "4C_mat_sutherland.hpp"
#include "4C_mat_thermoplasticlinelast.hpp"
#include "4C_mat_thermostvenantkirchhoff.hpp"
#include "4C_mat_viscoanisotropic.hpp"
#include "4C_mat_viscoelasthyper.hpp"
#include "4C_mat_visconeohooke.hpp"
#include "4C_membrane_eletypes.hpp"
#include "4C_membrane_scatra_eletypes.hpp"
#include "4C_module_registry_callbacks.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_node.hpp"
#include "4C_particle_engine_object.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_function.hpp"
#include "4C_porofluid_pressure_based_ele.hpp"
#include "4C_red_airways_elementbase.hpp"
#include "4C_rigidsphere.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_functions.hpp"
#include "4C_shell7p_ele.hpp"
#include "4C_shell7p_ele_scatra.hpp"
#include "4C_shell_kl_nurbs.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_solid_poro_3D_ele_pressure_based.hpp"
#include "4C_solid_poro_3D_ele_pressure_velocity_based.hpp"
#include "4C_solid_poro_3D_ele_pressure_velocity_based_p1.hpp"
#include "4C_solid_scatra_3D_ele.hpp"
#include "4C_stru_multi_microstatic.hpp"
#include "4C_structure_new_functions.hpp"
#include "4C_structure_new_resulttest.hpp"
#include "4C_thermo_element.hpp"
#include "4C_torsion3.hpp"
#include "4C_truss3.hpp"
#include "4C_truss3_scatra.hpp"
#include "4C_utils_function_library.hpp"
#include "4C_w1.hpp"
#include "4C_w1_nurbs.hpp"
#include "4C_w1_poro_eletypes.hpp"
#include "4C_w1_poro_p1_eletypes.hpp"
#include "4C_w1_poro_p1_scatra_eletypes.hpp"
#include "4C_w1_poro_scatra_eletypes.hpp"
#include "4C_w1_scatra.hpp"

#include <iostream>
#include <string>

FOUR_C_NAMESPACE_OPEN

namespace
{
  void register_par_object_types()
  {
    // Perform a dummy operation for the side-effect of forcing registration.
    std::stringstream s;

    s << Core::Nodes::NodeType::instance().name() << " "
      << Core::FE::Nurbs::ControlPointType::instance().name() << " "
      << CrossLinking::CrosslinkerNodeType::instance().name() << " "
      << Core::FE::MeshFree::MeshfreeMultiBinType::instance().name() << " "
      << Discret::Elements::Beam3rType::instance().name() << " "
      << Discret::Elements::Beam3ebType::instance().name() << " "
      << Discret::Elements::Beam3kType::instance().name() << " "
      << Discret::Elements::RigidsphereType::instance().name() << " "
      << Discret::Elements::Truss3Type::instance().name() << " "
      << Discret::Elements::Truss3ScatraType::instance().name() << " "
      << Discret::Elements::Torsion3Type::instance().name() << " "
      << Discret::Elements::Shell7pType::instance().name() << " "
      << Discret::Elements::Shell7pScatraType::instance().name() << " "
      << Discret::Elements::KirchhoffLoveShellNurbsType::instance().name() << " "
      << Discret::Elements::MembraneTri3Type::instance().name() << " "
      << Discret::Elements::MembraneTri6Type::instance().name() << " "
      << Discret::Elements::MembraneQuad4Type::instance().name() << " "
      << Discret::Elements::MembraneQuad9Type::instance().name() << " "
      << Discret::Elements::MembraneScatraTri3Type::instance().name() << " "
      << Discret::Elements::MembraneScatraTri6Type::instance().name() << " "
      << Discret::Elements::MembraneScatraQuad4Type::instance().name() << " "
      << Discret::Elements::MembraneScatraQuad9Type::instance().name() << " "
      << Discret::Elements::Wall1Type::instance().name() << " "
      << Discret::Elements::WallTri3PoroType::instance().name() << " "
      << Discret::Elements::WallTri3PoroP1Type::instance().name() << " "
      << Discret::Elements::WallQuad4PoroType::instance().name() << " "
      << Discret::Elements::WallQuad4PoroP1Type::instance().name() << " "
      << Discret::Elements::WallQuad9PoroType::instance().name() << " "
      << Discret::Elements::WallQuad9PoroP1Type::instance().name() << " "
      << Discret::Elements::WallNurbs4PoroType::instance().name() << " "
      << Discret::Elements::WallNurbs9PoroType::instance().name() << " "
      << Discret::Elements::Nurbs::Wall1NurbsType::instance().name() << " "
      << Discret::Elements::Wall1ScatraType::instance().name() << " "
      << Discret::Elements::WallQuad4PoroScatraType::instance().name() << " "
      << Discret::Elements::WallQuad4PoroP1ScatraType::instance().name() << " "
      << Discret::Elements::FluidType::instance().name() << " "
      << Discret::Elements::FluidXWallType::instance().name() << " "
      << Discret::Elements::FluidXWallBoundaryType::instance().name() << " "
      << Discret::Elements::FluidPoroEleType::instance().name() << " "
      << Discret::Elements::FluidHDGType::instance().name() << " "
      << Discret::Elements::FluidHDGWeakCompType::instance().name() << " "
      << Discret::Elements::FluidBoundaryType::instance().name() << " "
      << Discret::Elements::FluidPoroBoundaryType::instance().name() << " "
      << Discret::Elements::Ale3Type::instance().name() << " "
      << Discret::Elements::Nurbs::Ale3NurbsType::instance().name() << " "
      << Discret::Elements::Ale2Type::instance().name() << " "
      << Discret::Elements::Nurbs::Ale2NurbsType::instance().name() << " "
      << Discret::Elements::Bele3Type::instance().name() << " "
      << Discret::Elements::Vele3Type::instance().name() << " "
      << Discret::Elements::SolidType::instance().name() << " "
      << Discret::Elements::SolidPoroPressureBasedType::instance().name() << " "
      << Discret::Elements::SolidPoroPressureVelocityBasedType::instance().name() << " "
      << Discret::Elements::SolidPoroPressureVelocityBasedP1Type::instance().name() << " "
      << Discret::Elements::SolidScatraType::instance().name() << " "
      << Discret::Elements::ArteryType::instance().name() << " "
      << Discret::Elements::RedAirwayType::instance().name() << " "
      << Discret::Elements::RedAcinusType::instance().name() << " "
      << Discret::Elements::RedInterAcinarDepType::instance().name() << " "
      << Discret::Elements::ConstraintElement2Type::instance().name() << " "
      << Discret::Elements::ConstraintElement3Type::instance().name() << " "
      << Discret::Elements::LubricationType::instance().name() << " "
      << Discret::Elements::PoroFluidMultiPhaseType::instance().name() << " "
      << Discret::Elements::TransportType::instance().name() << " "
      << Thermo::ElementType::instance().name() << " " << Mat::Cnst1dArtType::instance().name()
      << " " << Mat::AAAneohookeType::instance().name() << " "
      << Mat::CarreauYasudaType::instance().name() << " "
      << Mat::ConstraintMixtureType::instance().name() << " "
      << Mat::ConstraintMixtureHistoryType::instance().name() << " "
      << Mat::CrystalPlasticityType::instance().name() << " "
      << Mat::ElastHyperType::instance().name() << " "
      << Mat::PlasticElastHyperType::instance().name() << " "
      << Mat::PlasticElastHyperVCUType::instance().name() << " "
      << Mat::ViscoElastHyperType::instance().name() << " " << Mat::FluidPoroType::instance().name()
      << " " << Mat::FluidPoroSinglePhaseType::instance().name() << " "
      << Mat::FluidPoroSingleVolFracType::instance().name() << " "
      << Mat::FluidPoroVolFracPressureType::instance().name() << " "
      << Mat::FluidPoroVolFracPressureBloodLungType::instance().name() << " "
      << Mat::FluidPoroSingleReactionType::instance().name() << " "
      << Mat::FluidPoroMultiPhaseType::instance().name() << " "
      << Mat::FluidPoroMultiPhaseReactionsType::instance().name() << " "
      << Mat::FourierType::instance().name() << " "
      << Mat::MembraneElastHyperType::instance().name() << " "
      << Mat::MembraneActiveStrainType::instance().name() << " "
      << Mat::GrowthRemodelElastHyperType::instance().name() << " "
      << Mat::MixtureType::instance().name() << " " << Mat::HerschelBulkleyType::instance().name()
      << " " << Mat::IonType::instance().name() << " "
      << Mat::LinearDensityViscosityType::instance().name() << " "
      << Mat::WeaklyCompressibleFluidType::instance().name() << " "
      << Mat::MatListType::instance().name() << " " << Mat::MatListReactionsType::instance().name()
      << " " << Mat::MatListChemotaxisType::instance().name() << " "
      << Mat::MatListChemoReacType::instance().name() << " " << Mat::ElchMatType::instance().name()
      << " " << Mat::MicroMaterialType::instance().name() << " "
      << Mat::ModPowerLawType::instance().name() << " "
      << Mat::MurnaghanTaitFluidType::instance().name() << " "
      << Mat::MyocardType::instance().name() << Mat::NewtonianFluidType::instance().name() << " "
      << Mat::StructPoroType::instance().name() << " "
      << Mat::StructPoroReactionType::instance().name() << " "
      << Mat::StructPoroReactionECMType::instance().name() << " "
      << Mat::ScalarDepInterpType::instance().name() << " " << Mat::ScatraMatType::instance().name()
      << " " << Mat::ScatraMatPoroECMType::instance().name() << " "
      << Mat::ScatraMatMultiPoroFluidType::instance().name() << " "
      << Mat::ScatraMatMultiPoroVolFracType::instance().name() << " "
      << Mat::ScatraMatMultiPoroSolidType::instance().name() << " "
      << Mat::ScatraMatMultiPoroTemperatureType::instance().name() << " "
      << Mat::StVenantKirchhoffType::instance().name() << " "
      << Mat::LinElast1DType::instance().name() << " "
      << Mat::LinElast1DGrowthType::instance().name() << " "
      << Mat::SutherlandType::instance().name() << " "
      << Mat::ThermoStVenantKirchhoffType::instance().name() << " "
      << Mat::ThermoPlasticLinElastType::instance().name() << " "
      << Mat::ViscoAnisotropicType::instance().name() << " "
      << Mat::ViscoNeoHookeType::instance().name() << " " << Mat::SpringType::instance().name()
      << " " << Mat::BeamElastHyperMaterialType<double>::instance().name() << " "
      << Mat::BeamElastHyperMaterialType<Sacado::Fad::DFad<double>>::instance().name() << " "
      << Discret::Elements::KirchhoffLoveShellNurbsType::instance().name() << " "
      << Mat::PlasticLinElastType::instance().name() << " " << Mat::RobinsonType::instance().name()
      << " " << Mat::DamageType::instance().name() << " "
      << Mat::Maxwell0dAcinusType::instance().name() << " "
      << Mat::Maxwell0dAcinusNeoHookeanType::instance().name() << " "
      << Mat::Maxwell0dAcinusExponentialType::instance().name() << " "
      << Mat::Maxwell0dAcinusDoubleExponentialType::instance().name() << " "
      << Mat::Maxwell0dAcinusOgdenType::instance().name() << " "
      << Mortar::NodeType::instance().name() << " " << Mortar::ElementType::instance().name() << " "
      << CONTACT::NodeType::instance().name() << " " << CONTACT::FriNodeType::instance().name()
      << " " << CONTACT::ElementType::instance().name() << " "
      << BeamInteraction::BeamLinkBeam3rLine2RigidJointedType::instance().name() << " "
      << BeamInteraction::BeamLinkBeam3rLine2PinJointedType::instance().name() << " "
      << BeamInteraction::BeamLinkTrussType::instance().name() << " "
      << Particle::ParticleObjectType::instance().name() << " "
      << MultiScale::MicroStaticParObjectType::instance().name() << " ";
  }

  void attach_function_definitions(Core::Utils::FunctionManager& function_manager)
  {
    add_valid_builtin_functions(function_manager);
    Solid::add_valid_structure_functions(function_manager);
    FLD::add_valid_fluid_functions(function_manager);
    Discret::Utils::add_valid_combust_functions(function_manager);
    Discret::Utils::add_valid_xfluid_functions(function_manager);
    add_valid_library_functions(function_manager);
    PoroPressureBased::add_valid_poro_functions(function_manager);
    ScaTra::add_valid_scatra_functions(function_manager);
  }

  Core::IO::InputSpec valid_result_lines()
  {
    using namespace Core::IO::InputSpecBuilders;

    return one_of({
        group("STRUCTURE",
            {
                one_of({
                    all_of({
                        parameter<std::string>("DIS"),
                        one_of({
                            parameter<int>("NODE"),
                            parameter<int>("LINE"),
                            parameter<int>("SURFACE"),
                            parameter<int>("VOLUME"),
                        }),
                    }),
                    parameter<bool>("SPECIAL"),
                }),
                parameter<Solid::ResultTest::TestOp>(
                    "OP", {.default_value = Solid::ResultTest::TestOp::unknown}),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("FLUID",
            {
                parameter<std::string>("DIS"),
                one_of({
                    parameter<int>("NODE"),
                    parameter<int>("ELEMENT"),
                }),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("XFLUID",
            {
                parameter<std::string>("DIS"),
                parameter<int>("NODE"),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("ALE",
            {
                parameter<std::string>("DIS"),
                parameter<int>("NODE"),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("THERMAL",
            {
                parameter<std::string>("DIS"),
                parameter<int>("NODE"),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("LUBRICATION",
            {
                parameter<std::string>("DIS"),
                parameter<int>("NODE"),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("POROFLUIDMULTIPHASE",
            {
                parameter<std::string>("DIS"),
                one_of({
                    parameter<int>("NODE"),
                    parameter<int>("ELEMENT"),
                    parameter<bool>("SPECIAL"),
                }),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("SCATRA",
            {
                parameter<std::string>("DIS"),
                one_of({
                    parameter<int>("NODE"),
                    parameter<bool>("SPECIAL"),
                }),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("SSI",
            {
                one_of({
                    all_of({
                        parameter<std::string>("DIS"),
                        parameter<int>("NODE"),
                    }),
                    parameter<bool>("SPECIAL"),
                }),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("SSTI",
            {
                parameter<bool>("SPECIAL"),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("STI",
            {
                parameter<bool>("SPECIAL"),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("RED_AIRWAY",
            {
                parameter<std::string>("DIS"),
                one_of({
                    parameter<int>("NODE"),
                    parameter<int>("ELEMENT"),
                }),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("ARTNET",
            {
                parameter<std::string>("DIS"),
                one_of({
                    parameter<int>("NODE"),
                    parameter<int>("ELEMENT"),
                }),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("FSI",
            {
                one_of({
                    parameter<int>("NODE"),
                    parameter<bool>("SPECIAL"),
                }),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("PARTICLE",
            {
                parameter<int>("ID"),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("PARTICLEWALL",
            {
                parameter<std::string>("DIS"),
                one_of({
                    parameter<int>("NODE"),
                    parameter<bool>("SPECIAL"),
                }),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("RIGIDBODY",
            {
                parameter<int>("ID"),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
        group("CARDIOVASCULAR0D",
            {
                parameter<std::string>("DIS"),
                parameter<bool>("SPECIAL"),
                parameter<std::string>("QUANTITY"),
                parameter<double>("VALUE"),
                parameter<double>("TOLERANCE"),
                parameter<std::optional<std::string>>("NAME"),
            }),
    });
  }
}  // namespace

ModuleCallbacks global_legacy_module_callbacks()
{
  ModuleCallbacks callbacks;
  callbacks.RegisterParObjectTypes = register_par_object_types;
  callbacks.AttachFunctionDefinitions = attach_function_definitions;
  callbacks.valid_result_description_lines = valid_result_lines;
  callbacks.materials = Global::valid_materials;
  callbacks.conditions = Global::valid_conditions;
  callbacks.parameters = Global::valid_parameters;

  return callbacks;
}


FOUR_C_NAMESPACE_CLOSE

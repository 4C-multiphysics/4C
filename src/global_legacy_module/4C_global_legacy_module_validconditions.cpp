// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_global_legacy_module_validconditions.hpp"

#include "4C_ale_input.hpp"
#include "4C_art_net_input.hpp"
#include "4C_beaminteraction_potential_input.hpp"
#include "4C_constraint_framework_input.hpp"
#include "4C_ehl_input.hpp"
#include "4C_elch_input.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_beaminteraction.hpp"
#include "4C_inpar_bio.hpp"
#include "4C_inpar_cardiac_monodomain.hpp"
#include "4C_inpar_cardiovascular0d.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_inpar_fpsi.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_inpar_levelset.hpp"
#include "4C_inpar_mortar.hpp"
#include "4C_inpar_s2i.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_inpar_xfem.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_particle_input.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_input.hpp"
#include "4C_red_airways_input.hpp"
#include "4C_ssi_input.hpp"
#include "4C_ssti_input.hpp"
#include "4C_sti_input.hpp"
#include "4C_thermo_input.hpp"

FOUR_C_NAMESPACE_OPEN


namespace
{
  // collect some problem-specific conditions that do not fit in the generic sections
  void set_miscellaneous_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
  {
    using namespace Core::IO::InputSpecBuilders;
    /*--------------------------------------------------------------------*/
    // microscale boundary

    Core::Conditions::ConditionDefinition microscale("MICROSCALE CONDITIONS", "MicroBoundary",
        "Microscale Boundary", Core::Conditions::MicroBoundary, true,
        Core::Conditions::geometry_type_surface);

    condlist.push_back(microscale);

    /*--------------------------------------------------------------------*/
    // stc layer condition

    Core::Conditions::ConditionDefinition stclayer("DESIGN VOL STC LAYER", "STC Layer",
        "Layer for Multilayered STC", Core::Conditions::VolSTCLayer, true,
        Core::Conditions::geometry_type_volume);

    stclayer.add_component(parameter<int>("ConditionID"));

    condlist.push_back(stclayer);
  }
}  // namespace


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<Core::Conditions::ConditionDefinition> Global::valid_conditions()
{
  using namespace Core::IO::InputSpecBuilders;
  std::vector<Core::Conditions::ConditionDefinition> condlist;

  /*--------------------------------------------------------------------*/
  // Neumann conditions
  Core::Conditions::ConditionDefinition pointneumann("DESIGN POINT NEUMANN CONDITIONS",
      "PointNeumann", "Point Neumann", Core::Conditions::PointNeumann, false,
      Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition pointneumanneb("DESIGN POINT MOMENT EB CONDITIONS",
      "PointNeumannEB", "Point Neumann Moment for an Euler-Bernoulli beam",
      Core::Conditions::PointNeumannEB, false, Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition lineneumann("DESIGN LINE NEUMANN CONDITIONS", "LineNeumann",
      "Line Neumann", Core::Conditions::LineNeumann, true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfneumann("DESIGN SURF NEUMANN CONDITIONS",
      "SurfaceNeumann", "Surface Neumann", Core::Conditions::SurfaceNeumann, true,
      Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volneumann("DESIGN VOL NEUMANN CONDITIONS", "VolumeNeumann",
      "Volume Neumann", Core::Conditions::VolumeNeumann, true,
      Core::Conditions::geometry_type_volume);

  // Neumann conditions for transport problems
  Core::Conditions::ConditionDefinition pointtransportneumann(
      "DESIGN POINT TRANSPORT NEUMANN CONDITIONS", "TransportPointNeumann", "Point Neumann",
      Core::Conditions::PointNeumann, false, Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition linetransportneumann(
      "DESIGN LINE TRANSPORT NEUMANN CONDITIONS", "TransportLineNeumann", "Line Neumann",
      Core::Conditions::LineNeumann, true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surftransportneumann(
      "DESIGN SURF TRANSPORT NEUMANN CONDITIONS", "TransportSurfaceNeumann", "Surface Neumann",
      Core::Conditions::SurfaceNeumann, true, Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition voltransportneumann(
      "DESIGN VOL TRANSPORT NEUMANN CONDITIONS", "TransportVolumeNeumann", "Volume Neumann",
      Core::Conditions::VolumeNeumann, true, Core::Conditions::geometry_type_volume);

  // Neumann conditions for thermo problems
  Core::Conditions::ConditionDefinition pointthermoneumann("DESIGN POINT THERMO NEUMANN CONDITIONS",
      "ThermoPointNeumann", "Point Neumann", Core::Conditions::PointNeumann, false,
      Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition linethermoneumann("DESIGN LINE THERMO NEUMANN CONDITIONS",
      "ThermoLineNeumann", "Line Neumann", Core::Conditions::LineNeumann, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfthermoneumann("DESIGN SURF THERMO NEUMANN CONDITIONS",
      "ThermoSurfaceNeumann", "Surface Neumann", Core::Conditions::SurfaceNeumann, true,
      Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volthermoneumann("DESIGN VOL THERMO NEUMANN CONDITIONS",
      "ThermoVolumeNeumann", "Volume Neumann", Core::Conditions::VolumeNeumann, true,
      Core::Conditions::geometry_type_volume);

  // Neumann conditions for poroelasticity problems
  Core::Conditions::ConditionDefinition pointporoneumann("DESIGN POINT PORO NEUMANN CONDITIONS",
      "PoroPointNeumann", "Point Neumann", Core::Conditions::PointNeumann, false,
      Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition lineporoneumann("DESIGN LINE PORO NEUMANN CONDITIONS",
      "PoroLineNeumann", "Line Neumann", Core::Conditions::LineNeumann, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfporoneumann("DESIGN SURF PORO NEUMANN CONDITIONS",
      "PoroSurfaceNeumann", "Surface Neumann", Core::Conditions::SurfaceNeumann, true,
      Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volporoneumann("DESIGN VOL PORO NEUMANN CONDITIONS",
      "PoroVolumeNeumann", "Volume Neumann", Core::Conditions::VolumeNeumann, true,
      Core::Conditions::geometry_type_volume);

  const auto make_neumann_condition = [&condlist](auto& cond)
  {
    cond.add_component(parameter<int>("NUMDOF"));
    cond.add_component(parameter<std::vector<int>>(
        "ONOFF", {.description = "onoff", .size = from_parameter<int>("NUMDOF")}));
    cond.add_component(parameter<std::vector<double>>(
        "VAL", {.description = "values", .size = from_parameter<int>("NUMDOF")}));
    cond.add_component(parameter<std::vector<std::optional<int>>>(
        "FUNCT", {.description = "function ids", .size = from_parameter<int>("NUMDOF")}));
    cond.add_component(deprecated_selection<std::string>("TYPE",
        {"Live", "Dead", "pseudo_orthopressure", "orthopressure", "PressureGrad"},
        {.description = "type", .default_value = "Live"}));

    condlist.emplace_back(cond);
  };

  make_neumann_condition(pointneumann);
  make_neumann_condition(pointneumanneb);
  make_neumann_condition(lineneumann);
  make_neumann_condition(surfneumann);
  make_neumann_condition(volneumann);

  make_neumann_condition(pointtransportneumann);
  make_neumann_condition(linetransportneumann);
  make_neumann_condition(surftransportneumann);
  make_neumann_condition(voltransportneumann);

  make_neumann_condition(pointthermoneumann);
  make_neumann_condition(linethermoneumann);
  make_neumann_condition(surfthermoneumann);
  make_neumann_condition(volthermoneumann);

  make_neumann_condition(pointporoneumann);
  make_neumann_condition(lineporoneumann);
  make_neumann_condition(surfporoneumann);
  make_neumann_condition(volporoneumann);


  /*--------------------------------------------------------------------*/
  // Dirichlet conditions
  Core::Conditions::ConditionDefinition pointdirichlet("DESIGN POINT DIRICH CONDITIONS",
      "Dirichlet", "Point Dirichlet", Core::Conditions::PointDirichlet, false,
      Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition linedirichlet("DESIGN LINE DIRICH CONDITIONS", "Dirichlet",
      "Line Dirichlet", Core::Conditions::LineDirichlet, false,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfdirichlet("DESIGN SURF DIRICH CONDITIONS", "Dirichlet",
      "Surface Dirichlet", Core::Conditions::SurfaceDirichlet, false,
      Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition voldirichlet("DESIGN VOL DIRICH CONDITIONS", "Dirichlet",
      "Volume Dirichlet", Core::Conditions::VolumeDirichlet, false,
      Core::Conditions::geometry_type_volume);

  Core::Conditions::ConditionDefinition pointaledirichlet("DESIGN POINT ALE DIRICH CONDITIONS",
      "ALEDirichlet", "Point Dirichlet", Core::Conditions::PointDirichlet, false,
      Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition linealedirichlet("DESIGN LINE ALE DIRICH CONDITIONS",
      "ALEDirichlet", "Line Dirichlet", Core::Conditions::LineDirichlet, false,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfaledirichlet("DESIGN SURF ALE DIRICH CONDITIONS",
      "ALEDirichlet", "Surface Dirichlet", Core::Conditions::SurfaceDirichlet, false,
      Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volaledirichlet("DESIGN VOL ALE DIRICH CONDITIONS",
      "ALEDirichlet", "Volume Dirichlet", Core::Conditions::VolumeDirichlet, false,
      Core::Conditions::geometry_type_volume);

  // Dirichlet conditions for transport problems
  Core::Conditions::ConditionDefinition pointtransportdirichlet(
      "DESIGN POINT TRANSPORT DIRICH CONDITIONS", "TransportDirichlet", "Point Dirichlet",
      Core::Conditions::PointDirichlet, false, Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition linetransportdirichlet(
      "DESIGN LINE TRANSPORT DIRICH CONDITIONS", "TransportDirichlet", "Line Dirichlet",
      Core::Conditions::LineDirichlet, false, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surftransportdirichlet(
      "DESIGN SURF TRANSPORT DIRICH CONDITIONS", "TransportDirichlet", "Surface Dirichlet",
      Core::Conditions::SurfaceDirichlet, false, Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition voltransportdirichlet(
      "DESIGN VOL TRANSPORT DIRICH CONDITIONS", "TransportDirichlet", "Volume Dirichlet",
      Core::Conditions::VolumeDirichlet, false, Core::Conditions::geometry_type_volume);

  // Dirichlet conditions for thermo problems
  Core::Conditions::ConditionDefinition pointthermodirichlet(
      "DESIGN POINT THERMO DIRICH CONDITIONS", "ThermoDirichlet", "Point Dirichlet",
      Core::Conditions::PointDirichlet, false, Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition linethermodirichlet("DESIGN LINE THERMO DIRICH CONDITIONS",
      "ThermoDirichlet", "Line Dirichlet", Core::Conditions::LineDirichlet, false,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfthermodirichlet("DESIGN SURF THERMO DIRICH CONDITIONS",
      "ThermoDirichlet", "Surface Dirichlet", Core::Conditions::SurfaceDirichlet, false,
      Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volthermodirichlet("DESIGN VOL THERMO DIRICH CONDITIONS",
      "ThermoDirichlet", "Volume Dirichlet", Core::Conditions::VolumeDirichlet, false,
      Core::Conditions::geometry_type_volume);

  // Dirichlet conditions for poroelasticity problems
  Core::Conditions::ConditionDefinition pointporodirichlet("DESIGN POINT PORO DIRICH CONDITIONS",
      "PoroDirichlet", "Point Dirichlet", Core::Conditions::PointDirichlet, false,
      Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition lineporodirichlet("DESIGN LINE PORO DIRICH CONDITIONS",
      "PoroDirichlet", "Line Dirichlet", Core::Conditions::LineDirichlet, false,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfporodirichlet("DESIGN SURF PORO DIRICH CONDITIONS",
      "PoroDirichlet", "Surface Dirichlet", Core::Conditions::SurfaceDirichlet, false,
      Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volporodirichlet("DESIGN VOL PORO DIRICH CONDITIONS",
      "PoroDirichlet", "Volume Dirichlet", Core::Conditions::VolumeDirichlet, false,
      Core::Conditions::geometry_type_volume);

  Core::Conditions::ConditionDefinition pointnurbslsdirichlet(
      "DESIGN POINT NURBS LS DIRICH CONDITIONS", "NurbsLSDirichlet", "Point Dirichlet",
      Core::Conditions::PointDirichlet, true, Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition linenurbslsdirichlet(
      "DESIGN LINE NURBS LS DIRICH CONDITIONS", "NurbsLSDirichlet", "Line Dirichlet",
      Core::Conditions::LineDirichlet, true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfnurbslsdirichlet(
      "DESIGN SURF NURBS LS DIRICH CONDITIONS", "NurbsLSDirichlet", "Surface Dirichlet",
      Core::Conditions::SurfaceDirichlet, true, Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volnurbslsdirichlet("DESIGN VOL NURBS LS DIRICH CONDITIONS",
      "NurbsLSDirichlet", "Volume Dirichlet", Core::Conditions::VolumeDirichlet, true,
      Core::Conditions::geometry_type_volume);

  const auto make_dirichlet_condition = [&condlist](auto& cond)
  {
    cond.add_component(parameter<int>("NUMDOF"));
    cond.add_component(parameter<std::vector<int>>(
        "ONOFF", {.description = "", .size = from_parameter<int>("NUMDOF")}));
    cond.add_component(parameter<std::vector<double>>(
        "VAL", {.description = "", .size = from_parameter<int>("NUMDOF")}));
    cond.add_component(parameter<std::vector<std::optional<int>>>(
        "FUNCT", {.description = "", .size = from_parameter<int>("NUMDOF")}));

    // optional
    cond.add_component(deprecated_selection<std::string>(
        "TAG", {"none", "monitor_reaction"}, {.description = "", .default_value = "none"}));

    condlist.emplace_back(cond);
  };

  make_dirichlet_condition(pointdirichlet);
  make_dirichlet_condition(linedirichlet);
  make_dirichlet_condition(surfdirichlet);
  make_dirichlet_condition(voldirichlet);

  make_dirichlet_condition(pointaledirichlet);
  make_dirichlet_condition(linealedirichlet);
  make_dirichlet_condition(surfaledirichlet);
  make_dirichlet_condition(volaledirichlet);

  make_dirichlet_condition(pointtransportdirichlet);
  make_dirichlet_condition(linetransportdirichlet);
  make_dirichlet_condition(surftransportdirichlet);
  make_dirichlet_condition(voltransportdirichlet);

  make_dirichlet_condition(pointthermodirichlet);
  make_dirichlet_condition(linethermodirichlet);
  make_dirichlet_condition(surfthermodirichlet);
  make_dirichlet_condition(volthermodirichlet);

  make_dirichlet_condition(pointporodirichlet);
  make_dirichlet_condition(lineporodirichlet);
  make_dirichlet_condition(surfporodirichlet);
  make_dirichlet_condition(volporodirichlet);

  make_dirichlet_condition(pointnurbslsdirichlet);
  make_dirichlet_condition(linenurbslsdirichlet);
  make_dirichlet_condition(surfnurbslsdirichlet);
  make_dirichlet_condition(volnurbslsdirichlet);

  /*--------------------------------------------------------------------*/
  // Point coupling (e.g. joints - couple X out of Y nodal DoFs)

  Core::Conditions::ConditionDefinition pointcoupling("DESIGN POINT COUPLING CONDITIONS",
      "PointCoupling", "Point Coupling", Core::Conditions::PointCoupling, false,
      Core::Conditions::geometry_type_point);

  Core::Conditions::ConditionDefinition pointthermocoupling(
      "DESIGN POINT THERMO COUPLING CONDITIONS", "PointThermoCoupling", "Point Coupling",
      Core::Conditions::PointCoupling, false, Core::Conditions::geometry_type_point);

  const auto make_point_coupling_condition = [&condlist](auto& cond)
  {
    cond.add_component(parameter<int>("NUMDOF"));
    cond.add_component(parameter<std::vector<int>>(
        "ONOFF", {.description = "", .size = from_parameter<int>("NUMDOF")}));

    condlist.emplace_back(cond);
  };

  make_point_coupling_condition(pointcoupling);
  make_point_coupling_condition(pointthermocoupling);

  /*--------------------------------------------------------------------*/
  // Initial fields

  // general initial field conditions
  Core::Conditions::ConditionDefinition pointinitfields("DESIGN POINT INITIAL FIELD CONDITIONS",
      "Initfield", "Point Initfield", Core::Conditions::PointInitfield, false,
      Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition lineinitfields("DESIGN LINE INITIAL FIELD CONDITIONS",
      "Initfield", "Line Initfield", Core::Conditions::LineInitfield, false,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfinitfields("DESIGN SURF INITIAL FIELD CONDITIONS",
      "Initfield", "Surface Initfield", Core::Conditions::SurfaceInitfield, false,
      Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volinitfields("DESIGN VOL INITIAL FIELD CONDITIONS",
      "Initfield", "Volume Initfield", Core::Conditions::VolumeInitfield, false,
      Core::Conditions::geometry_type_volume);

  const auto make_init_field_condition = [&condlist](auto& cond)
  {
    cond.add_component(deprecated_selection<std::string>("FIELD",
        {"Undefined", "Velocity", "Pressure", "Temperature", "ScaTra", "Porosity", "PoroMultiFluid",
            "Artery"},
        {.description = "init field"}));

    // for initial vector fields, use the COMPONENT option of our functions
    cond.add_component(parameter<int>("FUNCT"));

    condlist.emplace_back(cond);
  };

  make_init_field_condition(pointinitfields);
  make_init_field_condition(lineinitfields);
  make_init_field_condition(surfinitfields);
  make_init_field_condition(volinitfields);


  /*--------------------------------------------------------------------*/
  // define initial field that can be set on thermo simulations that use the ScaTra
  // discretization e.g. STI, SSTI

  // initial field conditions for temperature on ScaTra discretizations
  Core::Conditions::ConditionDefinition pointthermoinitfields(
      "DESIGN POINT THERMO INITIAL FIELD CONDITIONS", "ThermoInitfield",
      "Set the initial temperature field if the thermo field is solved using a ScaTra "
      "discretization (e.g. STI, SSTI) on points",
      Core::Conditions::PointInitfield, false, Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition linethermoinitfields(
      "DESIGN LINE THERMO INITIAL FIELD CONDITIONS", "ThermoInitfield",
      "Set the initial temperature field if the thermo field is solved using a ScaTra "
      "discretization (e.g. STI, SSTI) on lines",
      Core::Conditions::LineInitfield, false, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfthermoinitfields(
      "DESIGN SURF THERMO INITIAL FIELD CONDITIONS", "ThermoInitfield",
      "Set the initial temperature field if the thermo field is solved using a ScaTra "
      "discretization (e.g. STI, SSTI) on surfaces",
      Core::Conditions::SurfaceInitfield, false, Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volthermoinitfields(
      "DESIGN VOL THERMO INITIAL FIELD CONDITIONS", "ThermoInitfield",
      "Set the initial temperature field if the thermo field is solved using a ScaTra "
      "discretization (e.g. STI, SSTI) on volumes",
      Core::Conditions::VolumeInitfield, false, Core::Conditions::geometry_type_volume);

  const auto make_thermo_init_field_condition = [&condlist](auto& cond)
  {
    cond.add_component(deprecated_selection<std::string>(
        "FIELD", {"Undefined", "ScaTra"}, {.description = "init field"}));
    cond.add_component(parameter<int>("FUNCT"));

    condlist.emplace_back(cond);
  };

  make_thermo_init_field_condition(pointthermoinitfields);
  make_thermo_init_field_condition(linethermoinitfields);
  make_thermo_init_field_condition(surfthermoinitfields);
  make_thermo_init_field_condition(volthermoinitfields);


  /*--------------------------------------------------------------------*/
  // compute domain integrals, i.e., cumulative volumes of 3D domain elements or cumulative
  // surface areas of 2D domain elements

  // definition of surface and volume conditions for domain integral computation
  Core::Conditions::ConditionDefinition domainintegralsurf("DESIGN DOMAIN INTEGRAL SURF CONDITIONS",
      "DomainIntegral", "compute cumulative surface areas of 2D domain elements",
      Core::Conditions::DomainIntegral, true, Core::Conditions::geometry_type_surface);

  Core::Conditions::ConditionDefinition domainintegralvol("DESIGN DOMAIN INTEGRAL VOL CONDITIONS",
      "DomainIntegral", "compute cumulative volumes of 3D domain elements",
      Core::Conditions::DomainIntegral, true, Core::Conditions::geometry_type_volume);

  const auto make_domain_integral_condition = [&condlist](auto& cond)
  {
    cond.add_component(parameter<int>("ConditionID"));

    condlist.emplace_back(cond);
  };

  make_domain_integral_condition(domainintegralsurf);
  make_domain_integral_condition(domainintegralvol);


  /*--------------------------------------------------------------------*/
  // compute boundary integrals, i.e., cumulative surface areas of 2D boundary elements

  Core::Conditions::ConditionDefinition boundaryintegralsurf(
      "DESIGN BOUNDARY INTEGRAL SURF CONDITIONS", "BoundaryIntegral",
      "compute cumulative surface areas of 2D boundary elements",
      Core::Conditions::BoundaryIntegral, true, Core::Conditions::geometry_type_surface);

  // add input file line components to condition definition
  boundaryintegralsurf.add_component(parameter<int>("ConditionID"));

  // insert condition definition into global list of valid condition definitions
  condlist.push_back(boundaryintegralsurf);


  /*--------------------------------------------------------------------*/
  // wear in ALE description

  Core::Conditions::ConditionDefinition linealewear("DESIGN LINE ALE WEAR CONDITIONS 2D", "AleWear",
      "Line Ale Wear", Core::Conditions::AleWear, true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfalewear("DESIGN SURFACE WEAR CONDITIONS 3D", "AleWear",
      "Surface Ale Wear", Core::Conditions::AleWear, true, Core::Conditions::geometry_type_surface);

  condlist.push_back(linealewear);
  condlist.push_back(surfalewear);

  /*--------------------------------------------------------------------*/
  // local coordinate systems

  Core::Conditions::ConditionDefinition pointlocsys("DESIGN POINT LOCSYS CONDITIONS", "Locsys",
      "Point local coordinate system", Core::Conditions::PointLocsys, true,
      Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition linelocsys("DESIGN LINE LOCSYS CONDITIONS", "Locsys",
      "Line local coordinate system", Core::Conditions::LineLocsys, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surflocsys("DESIGN SURF LOCSYS CONDITIONS", "Locsys",
      "Surface local coordinate system", Core::Conditions::SurfaceLocsys, true,
      Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition vollocsys("DESIGN VOL LOCSYS CONDITIONS", "Locsys",
      "Volume local coordinate system", Core::Conditions::VolumeLocsys, true,
      Core::Conditions::geometry_type_volume);

  const auto make_locsys_condition = [&condlist](auto& cond)
  {
    cond.add_component(parameter<std::vector<double>>("ROTANGLE", {.description = "", .size = 3}));
    cond.add_component(
        parameter<std::vector<std::optional<int>>>("FUNCT", {.description = "", .size = 3}));
    cond.add_component(parameter<int>("USEUPDATEDNODEPOS"));

    if (cond.geometry_type() == Core::Conditions::geometry_type_line ||
        cond.geometry_type() == Core::Conditions::geometry_type_surface)
    {
      cond.add_component(parameter<int>("USECONSISTENTNODENORMAL"));
    }

    condlist.emplace_back(cond);
  };

  make_locsys_condition(pointlocsys);
  make_locsys_condition(linelocsys);
  make_locsys_condition(surflocsys);
  make_locsys_condition(vollocsys);

  /*--------------------------------------------------------------------*/
  // periodic boundary

  Core::Conditions::ConditionDefinition lineperiodic("DESIGN LINE PERIODIC BOUNDARY CONDITIONS",
      "LinePeriodic", "Line Periodic", Core::Conditions::LinePeriodic, false,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfperiodic("DESIGN SURF PERIODIC BOUNDARY CONDITIONS",
      "SurfacePeriodic", "Surface Periodic", Core::Conditions::SurfacePeriodic, false,
      Core::Conditions::geometry_type_surface);

  const auto make_periodic_condition = [&condlist](auto& cond)
  {
    cond.add_component(parameter<int>("ID", {.description = "periodic boundary condition id"}));
    cond.add_component(deprecated_selection<std::string>(
        "MASTER_OR_SLAVE", {"Master", "Slave"}, {.description = "master-slave toggle"}));
    cond.add_component(deprecated_selection<std::string>("PLANE", {"xy", "yz", "xz", "xyz"},
        {.description = "degrees of freedom for the pbc plane"}));
    cond.add_component(
        parameter<int>("LAYER", {.description = "layer of periodic boundary condition"}));
    cond.add_component(parameter<double>("ANGLE", {.description = "angle of rotation"}));
    cond.add_component(
        parameter<double>("ABSTREETOL", {.description = "tolerance for nodematching in octree"}));

    condlist.emplace_back(cond);
  };

  make_periodic_condition(lineperiodic);
  make_periodic_condition(surfperiodic);

  /*--------------------------------------------------------------------*/
  // weak Dirichlet conditions
  Core::Conditions::ConditionDefinition lineweakdirichlet("DESIGN LINE WEAK DIRICHLET CONDITIONS",
      "LineWeakDirichlet", "LineWeakDirichlet", Core::Conditions::LineWeakDirichlet, true,
      Core::Conditions::geometry_type_line);

  Core::Conditions::ConditionDefinition surfweakdirichlet(
      "DESIGN SURFACE WEAK DIRICHLET CONDITIONS", "SurfaceWeakDirichlet", "SurfaceWeakDirichlet",
      Core::Conditions::SurfaceWeakDirichlet, true, Core::Conditions::geometry_type_surface);

  const auto make_weak_dirichlet_condition = [&condlist](auto& cond)
  {
    // weak DBCs can be imposed adjoint consistent or adjoint inconsistent
    cond.add_component(deprecated_selection<std::string>("GAMMATYPE",
        {"adjoint-consistent", "diffusive-optimal"}, {.description = "Choice of gamma parameter"}));

    // weak DBCs can be imposed in all directions or only in normal direction
    // (SCATRA: not checked, only in all_directions so far)
    cond.add_component(
        deprecated_selection<std::string>("DIR", {"all_directions", "only_in_normal_direction"},
            {.description = "Directions to apply weak dbc"}));

    // FLUID: penalty parameter either computed dynamically (using Spaldings law of
    // the wall) or by a fixed value; SCATRA: not checked, only constant value so far
    cond.add_component(deprecated_selection<std::string>(
        "PENTYPE", {"constant", "Spalding"}, {.description = "Definition of penalty parameter"}));

    // scaling factor for penalty parameter tauB or
    // stabilization parameter alpha for Nitsche term
    // (SCATRA: if stabilization parameter negative -> mixed-hybrid formulation)
    cond.add_component(parameter<double>("TauBscaling"));

    // linearisation strategies --- the linearisation (i.e. the matrix
    // contribution) of the convective term on the inflow could be
    // suppressed, since the flux is a kink function and including this one
    // might result in even worse convergence behaviour
    // (SCATRA: not checked)
    cond.add_component(deprecated_selection<std::string>(
        "LINEARISATION", {"lin_all", "no_lin_conv_inflow"}, {.description = "Linearisation"}));

    // we provide a vector of 3 values for velocities
    cond.add_component(parameter<std::vector<double>>("VAL", {.description = "values", .size = 3}));

    // and optional spatial functions
    cond.add_component(parameter<std::vector<int>>("FUNCT",
        {.description = "function ids", .default_value = std::vector<int>{0, 0, 0}, .size = 3}));
    // append the condition to the list of all conditions
    condlist.push_back(cond);
  };

  make_weak_dirichlet_condition(lineweakdirichlet);
  make_weak_dirichlet_condition(surfweakdirichlet);

  /*--------------------------------------------------------------------*/
  // boundary for superconvergent patch recovery (SPR)

  Core::Conditions::ConditionDefinition linespr("DESIGN PATCH RECOVERY BOUNDARY LINE CONDITIONS",
      "SPRboundary", "Boundary for SPR", Core::Conditions::SPRboundary, false,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfspr("DESIGN PATCH RECOVERY BOUNDARY SURF CONDITIONS",
      "SPRboundary", "Boundary for SPR", Core::Conditions::SPRboundary, false,
      Core::Conditions::geometry_type_surface);

  condlist.push_back(linespr);
  condlist.push_back(surfspr);

  /*--------------------------------------------------------------------*/
  // volume constraint

  Core::Conditions::ConditionDefinition volumeconstraint("DESIGN SURFACE VOLUME CONSTRAINT 3D",
      "VolumeConstraint_3D", "Surface Volume Constraint", Core::Conditions::VolumeConstraint_3D,
      true, Core::Conditions::geometry_type_surface);

  volumeconstraint.add_component(parameter<int>("ConditionID"));
  volumeconstraint.add_component(
      parameter<std::optional<int>>("curve", {.description = "id of the curve"}));
  volumeconstraint.add_component(parameter<double>("activeTime"));
  volumeconstraint.add_component(deprecated_selection<std::string>("projection",
      {"none", "xy", "yz", "xz"}, {.description = "projection", .default_value = "none"}));


  condlist.push_back(volumeconstraint);

  /*--------------------------------------------------------------------*/
  // volume constraint penalty

  Core::Conditions::ConditionDefinition volumeconstraintpen(
      "DESIGN SURFACE VOLUME CONSTRAINT 3D PEN", "VolumeConstraint_3D_Pen",
      "Surface Volume Constraint Penalty", Core::Conditions::VolumeConstraint_3D_pen, true,
      Core::Conditions::geometry_type_surface);

  volumeconstraintpen.add_component(parameter<int>("ConditionID"));
  volumeconstraintpen.add_component(
      parameter<std::optional<int>>("curve", {.description = "id of the curve"}));
  volumeconstraintpen.add_component(parameter<double>("activeTime"));
  volumeconstraintpen.add_component(parameter<double>("penalty"));
  volumeconstraintpen.add_component(parameter<double>("rho"));
  volumeconstraintpen.add_component(deprecated_selection<std::string>("projection",
      {"none", "xy", "yz", "xz"}, {.description = "projection", .default_value = "none"}));

  condlist.push_back(volumeconstraintpen);

  /*--------------------------------------------------------------------*/
  // area constraint

  Core::Conditions::ConditionDefinition areaconstraint("DESIGN SURFACE AREA CONSTRAINT 3D",
      "AreaConstraint_3D", "Surface Area Constraint", Core::Conditions::AreaConstraint_3D, true,
      Core::Conditions::geometry_type_surface);

  areaconstraint.add_component(parameter<int>("ConditionID"));
  areaconstraint.add_component(
      parameter<std::optional<int>>("curve", {.description = "id of the curve"}));
  areaconstraint.add_component(parameter<double>("activeTime"));

  condlist.push_back(areaconstraint);


  /*--------------------------------------------------------------------*/
  // volume monitor

  Core::Conditions::ConditionDefinition volumemonitor("DESIGN SURFACE VOLUME MONITOR 3D",
      "VolumeMonitor_3D", "Surface Volume Monitor", Core::Conditions::VolumeMonitor_3D, true,
      Core::Conditions::geometry_type_surface);

  volumemonitor.add_component(parameter<int>("ConditionID"));

  condlist.push_back(volumemonitor);

  /*--------------------------------------------------------------------*/
  // area monitor 3D

  Core::Conditions::ConditionDefinition areamonitor("DESIGN SURFACE AREA MONITOR 3D",
      "AreaMonitor_3D", "Surface Area Monitor", Core::Conditions::AreaMonitor_3D, true,
      Core::Conditions::geometry_type_surface);

  areamonitor.add_component(parameter<int>("ConditionID"));
  areamonitor.add_component(deprecated_selection<std::string>("projection",
      {"none", "xy", "yz", "xz"}, {.description = "projection", .default_value = "none"}));

  condlist.push_back(areamonitor);

  /*--------------------------------------------------------------------*/
  // area constraint

  Core::Conditions::ConditionDefinition areaconstraint2D("DESIGN LINE AREA CONSTRAINT 2D",
      "AreaConstraint_2D", "Line Area Constraint", Core::Conditions::AreaConstraint_2D, true,
      Core::Conditions::geometry_type_line);

  areaconstraint2D.add_component(parameter<int>("ConditionID"));
  areaconstraint2D.add_component(parameter<std::optional<int>>("curve", {.description = {}}));
  areaconstraint2D.add_component(parameter<double>("activeTime"));

  condlist.push_back(areaconstraint2D);

  /*--------------------------------------------------------------------*/
  // area monitor 2D

  Core::Conditions::ConditionDefinition areamonitor2D("DESIGN LINE AREA MONITOR 2D",
      "AreaMonitor_2D", "Line Area Monitor", Core::Conditions::AreaMonitor_2D, true,
      Core::Conditions::geometry_type_line);

  areamonitor2D.add_component(parameter<int>("ConditionID"));

  condlist.push_back(areamonitor2D);

  /*--------------------------------------------------------------------*/
  // Multi point constraint in 3D for a node over a plane

  Core::Conditions::ConditionDefinition nodeonplaneconst3D("DESIGN SURFACE MULTIPNT CONSTRAINT 3D",
      "MPC_NodeOnPlane_3D", "Node on Plane Constraint", Core::Conditions::MPC_NodeOnPlane_3D, false,
      Core::Conditions::geometry_type_surface);

  nodeonplaneconst3D.add_component(parameter<int>("ConditionID"));
  nodeonplaneconst3D.add_component(parameter<double>("amplitude"));
  nodeonplaneconst3D.add_component(parameter<std::optional<int>>("curve", {.description = {}}));
  nodeonplaneconst3D.add_component(parameter<double>("activeTime"));
  nodeonplaneconst3D.add_component(parameter<std::vector<int>>(
      "planeNodes", {.description = "ids of the nodes spanning the plane", .size = 3}));
  nodeonplaneconst3D.add_component(deprecated_selection<std::string>("control", {"rel", "abs"},
      {.description = "relative or absolute control", .default_value = "rel"}));

  condlist.push_back(nodeonplaneconst3D);

  /*--------------------------------------------------------------------*/
  // Multi point constraint in 3D, moving all constraint nodes synchronously

  Core::Conditions::ConditionDefinition nodemasterconst3D(
      "DESIGN SURFACE NORMALDIR MULTIPNT CONSTRAINT 3D", "MPC_NormalComponent_3D",
      "Node on Plane Constraint", Core::Conditions::MPC_NormalComponent_3D, false,
      Core::Conditions::geometry_type_surface);

  nodemasterconst3D.add_component(parameter<int>("ConditionID"));
  nodemasterconst3D.add_component(parameter<double>("amplitude"));
  nodemasterconst3D.add_component(parameter<std::optional<int>>("curve", {.description = {}}));
  nodemasterconst3D.add_component(parameter<double>("activeTime"));
  nodemasterconst3D.add_component(parameter<int>("masterNode"));
  nodemasterconst3D.add_component(
      parameter<std::vector<double>>("direction", {.description = "direction", .size = 3}));
  nodemasterconst3D.add_component(deprecated_selection<std::string>(
      "value", {"disp", "x"}, {.description = "value", .default_value = "disp"}));
  nodemasterconst3D.add_component(deprecated_selection<std::string>("control", {"rel", "abs"},
      {.description = "relative or absolute control", .default_value = "rel"}));

  condlist.push_back(nodemasterconst3D);

  /*--------------------------------------------------------------------*/
  // Multi point constraint in 3D, moving all constraint nodes synchronously, penalty based

  Core::Conditions::ConditionDefinition nodemasterconst3Dpen(
      "DESIGN SURFACE NORMALDIR MULTIPNT CONSTRAINT 3D PEN", "MPC_NormalComponent_3D_Pen",
      "Node on Plane Constraint Penalty", Core::Conditions::MPC_NormalComponent_3D_pen, false,
      Core::Conditions::geometry_type_surface);


  nodemasterconst3Dpen.add_component(parameter<int>("ConditionID"));
  nodemasterconst3Dpen.add_component(parameter<double>("amplitude"));
  nodemasterconst3Dpen.add_component(parameter<std::optional<int>>("curve", {.description = {}}));
  nodemasterconst3Dpen.add_component(parameter<double>("activeTime"));
  nodemasterconst3Dpen.add_component(parameter<double>("penalty"));
  nodemasterconst3Dpen.add_component(parameter<int>("masterNode"));
  nodemasterconst3Dpen.add_component(
      parameter<std::vector<int>>("direction", {.description = "direction", .size = 3}));
  nodemasterconst3Dpen.add_component(deprecated_selection<std::string>(
      "value", {"disp", "x"}, {.description = "value", .default_value = "disp"}));
  nodemasterconst3Dpen.add_component(deprecated_selection<std::string>("control", {"rel", "abs"},
      {.description = "relative or absolute control", .default_value = "rel"}));

  condlist.push_back(nodemasterconst3Dpen);
  /*--------------------------------------------------------------------*/
  // Multi point constraint in 2D for a node on a line
  Core::Conditions::ConditionDefinition nodeonlineconst2D("DESIGN LINE MULTIPNT CONSTRAINT 2D",
      "MPC_NodeOnLine_2D", "Node on Line Constraint", Core::Conditions::MPC_NodeOnLine_2D, false,
      Core::Conditions::geometry_type_line);

  nodeonlineconst2D.add_component(parameter<int>("ConditionID"));
  nodeonlineconst2D.add_component(parameter<double>("amplitude"));
  nodeonlineconst2D.add_component(parameter<std::optional<int>>("curve", {.description = {}}));
  nodeonlineconst2D.add_component(parameter<int>("constrNode1"));
  nodeonlineconst2D.add_component(parameter<int>("constrNode2"));
  nodeonlineconst2D.add_component(parameter<int>("constrNode3"));
  nodeonlineconst2D.add_component(deprecated_selection<std::string>("control", {"dist", "angle"},
      {.description = "distance or angle control", .default_value = "dist"}));
  nodeonlineconst2D.add_component(parameter<double>("activeTime"));

  condlist.push_back(nodeonlineconst2D);

  /*--------------------------------------------------------------------*/
  // Krylov Space Projection:
  // ========================
  // specify an unsupported (i.e. rigid body or zero energy) mode orthogonal
  // that is not excited by the body force
  //
  // examples are:
  // fluid:     pure Dirichlet (i.e. velocity) boundary conditions - pressure
  //            level is undetermined
  // scatra:    pure Neumann boundary condition - level of scalar quantity is
  //            undetermined
  // structure: insufficient support of translational or rotational rigid body
  //            modes
  //
  // for fluid and scatra, NUMDOF needs to be the number of dofs per node. the
  // following ONOFF values trigger the fixation of the quantity level (for
  // fluid, only pressure is allowed).
  // for solid NUMDOF is the number of potential rigid body modes (e.g. 6 for
  // 3-D solid, 3 for 2-D solid), where ONOFF triggers first the
  // translational followed by the rotational modes, each in/around x to z

  Core::Conditions::ConditionDefinition linerigidbodymode(
      "DESIGN LINE MODE FOR KRYLOV SPACE PROJECTION", "KrylovSpaceProjection",
      "Line mode for Krylov space projection", Core::Conditions::LineModeKrylovProjection, true,
      Core::Conditions::geometry_type_line);

  Core::Conditions::ConditionDefinition surfrigidbodymode(
      "DESIGN SURF MODE FOR KRYLOV SPACE PROJECTION", "KrylovSpaceProjection",
      "Surface mode for Krylov space projection", Core::Conditions::SurfaceModeKrylovProjection,
      true, Core::Conditions::geometry_type_surface);

  Core::Conditions::ConditionDefinition volrigidbodymode(
      "DESIGN VOL MODE FOR KRYLOV SPACE PROJECTION", "KrylovSpaceProjection",
      "Volume mode for Krylov space projection", Core::Conditions::VolumeModeKrylovProjection, true,
      Core::Conditions::geometry_type_volume);

  const auto make_rigidbody_mode_condition = [&condlist](auto& cond)
  {
    cond.add_component(deprecated_selection<std::string>(
        "DIS", {"fluid", "scatra", "structure"}, {.description = "discretization"}));
    cond.add_component(parameter<int>("NUMMODES"));
    cond.add_component(parameter<std::vector<int>>(
        "ONOFF", {.description = "", .size = from_parameter<int>("NUMMODES")}));
    cond.add_component(deprecated_selection<std::string>("WEIGHTVECDEF",
        {"integration", "pointvalues"}, {.description = "weight vector definition"}));
    cond.add_component(deprecated_selection<std::string>("TYPE", {"projection", "constraint"},
        {.description = "Type of krylov projection enforcement.", .default_value = "projection"}));

    condlist.emplace_back(cond);
  };

  make_rigidbody_mode_condition(linerigidbodymode);
  make_rigidbody_mode_condition(surfrigidbodymode);
  make_rigidbody_mode_condition(volrigidbodymode);


  // Finally, add the problem-specific conditions from the various modules
  Inpar::Mortar::set_valid_conditions(condlist);
  Inpar::S2I::set_valid_conditions(condlist);
  Inpar::ScaTra::set_valid_conditions(condlist);
  STI::set_valid_conditions(condlist);
  ElCh::set_valid_conditions(condlist);
  Inpar::ElectroPhysiology::set_valid_conditions(condlist);
  Inpar::FLUID::set_valid_conditions(condlist);
  ALE::set_valid_conditions(condlist);
  Inpar::FSI::set_valid_conditions(condlist);
  Inpar::FPSI::set_valid_conditions(condlist);
  Inpar::XFEM::set_valid_conditions(condlist);
  Inpar::BioFilm::set_valid_conditions(condlist);
  ArteryNetwork::set_valid_conditions(condlist);
  Airway::set_valid_conditions(condlist);
  Inpar::Cardiovascular0D::set_valid_conditions(condlist);
  Inpar::Solid::set_valid_conditions(condlist);
  Thermo::set_valid_conditions(condlist);
  SSI::set_valid_conditions(condlist);
  SSTI::set_valid_conditions(condlist);
  PARTICLE::set_valid_conditions(condlist);
  Inpar::LevelSet::set_valid_conditions(condlist);
  BeamInteraction::Potential::set_valid_conditions(condlist);
  Constraints::set_valid_conditions(condlist);
  Inpar::BeamInteraction::set_valid_conditions(condlist);
  EHL::set_valid_conditions(condlist);
  PoroPressureBased::set_valid_conditions_porofluid_elast_scatra(condlist);

  // finally some conditions that do not have their own files yet are problem-specific
  set_miscellaneous_conditions(condlist);

  return condlist;
}

FOUR_C_NAMESPACE_CLOSE

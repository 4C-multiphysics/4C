// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_validconditions.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_ale.hpp"
#include "4C_inpar_beaminteraction.hpp"
#include "4C_inpar_beampotential.hpp"
#include "4C_inpar_bio.hpp"
#include "4C_inpar_cardiac_monodomain.hpp"
#include "4C_inpar_cardiovascular0d.hpp"
#include "4C_inpar_ehl.hpp"
#include "4C_inpar_elch.hpp"
#include "4C_inpar_elemag.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_inpar_fpsi.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_inpar_immersed.hpp"
#include "4C_inpar_levelset.hpp"
#include "4C_inpar_mortar.hpp"
#include "4C_inpar_mpc_rve.hpp"
#include "4C_inpar_particle.hpp"
#include "4C_inpar_poromultiphase_scatra.hpp"
#include "4C_inpar_s2i.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_inpar_ssi.hpp"
#include "4C_inpar_ssti.hpp"
#include "4C_inpar_sti.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_inpar_thermo.hpp"
#include "4C_inpar_xfem.hpp"
#include "4C_io_linecomponent.hpp"

FOUR_C_NAMESPACE_OPEN


void Input::print_empty_condition_definitions(std::ostream& stream,
    std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>& condlist)
{
  for (unsigned i = 0; i < condlist.size(); ++i)
  {
    condlist[i]->print(stream);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void print_condition_dat_header()
{
  std::shared_ptr<std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>> condlist =
      Input::valid_conditions();
  Input::print_empty_condition_definitions(std::cout, *condlist);
}


namespace Input
{
  // collect some problem-specific conditions that do not fit in the generic sections
  void set_miscellaneous_conditions(
      std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>& condlist)
  {
    /*--------------------------------------------------------------------*/
    // microscale boundary

    std::shared_ptr<Core::Conditions::ConditionDefinition> microscale =
        std::make_shared<Core::Conditions::ConditionDefinition>("MICROSCALE CONDITIONS",
            "MicroBoundary", "Microscale Boundary", Core::Conditions::MicroBoundary, true,
            Core::Conditions::geometry_type_surface);

    condlist.push_back(microscale);

    /*--------------------------------------------------------------------*/
    // stc layer condition

    std::shared_ptr<Core::Conditions::ConditionDefinition> stclayer =
        std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN VOL STC LAYER", "STC Layer",
            "Layer for Multilayered STC", Core::Conditions::VolSTCLayer, true,
            Core::Conditions::geometry_type_volume);

    add_named_int(stclayer, "ConditionID");

    condlist.push_back(stclayer);
  }
}  // namespace Input


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>>
Input::valid_conditions()
{
  std::shared_ptr<std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>> vc =
      std::make_shared<std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>>();

  std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>& condlist = *vc;

  /*--------------------------------------------------------------------*/
  // Neumann conditions
  std::shared_ptr<Core::Conditions::ConditionDefinition> pointneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN POINT NEUMANN CONDITIONS",
          "PointNeumann", "Point Neumann", Core::Conditions::PointNeumann, false,
          Core::Conditions::geometry_type_point);
  std::shared_ptr<Core::Conditions::ConditionDefinition> pointneumanneb =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN POINT MOMENT EB CONDITIONS",
          "PointNeumannEB", "Point Neumann Moment for an Euler-Bernoulli beam",
          Core::Conditions::PointNeumannEB, false, Core::Conditions::geometry_type_point);
  std::shared_ptr<Core::Conditions::ConditionDefinition> lineneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN LINE NEUMANN CONDITIONS",
          "LineNeumann", "Line Neumann", Core::Conditions::LineNeumann, true,
          Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN SURF NEUMANN CONDITIONS",
          "SurfaceNeumann", "Surface Neumann", Core::Conditions::SurfaceNeumann, true,
          Core::Conditions::geometry_type_surface);
  std::shared_ptr<Core::Conditions::ConditionDefinition> volneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN VOL NEUMANN CONDITIONS",
          "VolumeNeumann", "Volume Neumann", Core::Conditions::VolumeNeumann, true,
          Core::Conditions::geometry_type_volume);

  // Neumann conditions for transport problems
  std::shared_ptr<Core::Conditions::ConditionDefinition> pointtransportneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN POINT TRANSPORT NEUMANN CONDITIONS", "TransportPointNeumann", "Point Neumann",
          Core::Conditions::PointNeumann, false, Core::Conditions::geometry_type_point);
  std::shared_ptr<Core::Conditions::ConditionDefinition> linetransportneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE TRANSPORT NEUMANN CONDITIONS", "TransportLineNeumann", "Line Neumann",
          Core::Conditions::LineNeumann, true, Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surftransportneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF TRANSPORT NEUMANN CONDITIONS", "TransportSurfaceNeumann", "Surface Neumann",
          Core::Conditions::SurfaceNeumann, true, Core::Conditions::geometry_type_surface);
  std::shared_ptr<Core::Conditions::ConditionDefinition> voltransportneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN VOL TRANSPORT NEUMANN CONDITIONS", "TransportVolumeNeumann", "Volume Neumann",
          Core::Conditions::VolumeNeumann, true, Core::Conditions::geometry_type_volume);

  // Neumann conditions for thermo problems
  std::shared_ptr<Core::Conditions::ConditionDefinition> pointthermoneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN POINT THERMO NEUMANN CONDITIONS", "ThermoPointNeumann", "Point Neumann",
          Core::Conditions::PointNeumann, false, Core::Conditions::geometry_type_point);
  std::shared_ptr<Core::Conditions::ConditionDefinition> linethermoneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE THERMO NEUMANN CONDITIONS", "ThermoLineNeumann", "Line Neumann",
          Core::Conditions::LineNeumann, true, Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfthermoneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF THERMO NEUMANN CONDITIONS", "ThermoSurfaceNeumann", "Surface Neumann",
          Core::Conditions::SurfaceNeumann, true, Core::Conditions::geometry_type_surface);
  std::shared_ptr<Core::Conditions::ConditionDefinition> volthermoneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN VOL THERMO NEUMANN CONDITIONS", "ThermoVolumeNeumann", "Volume Neumann",
          Core::Conditions::VolumeNeumann, true, Core::Conditions::geometry_type_volume);

  // Neumann conditions for poroelasticity problems
  std::shared_ptr<Core::Conditions::ConditionDefinition> pointporoneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN POINT PORO NEUMANN CONDITIONS", "PoroPointNeumann", "Point Neumann",
          Core::Conditions::PointNeumann, false, Core::Conditions::geometry_type_point);
  std::shared_ptr<Core::Conditions::ConditionDefinition> lineporoneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN LINE PORO NEUMANN CONDITIONS",
          "PoroLineNeumann", "Line Neumann", Core::Conditions::LineNeumann, true,
          Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfporoneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN SURF PORO NEUMANN CONDITIONS",
          "PoroSurfaceNeumann", "Surface Neumann", Core::Conditions::SurfaceNeumann, true,
          Core::Conditions::geometry_type_surface);
  std::shared_ptr<Core::Conditions::ConditionDefinition> volporoneumann =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN VOL PORO NEUMANN CONDITIONS",
          "PoroVolumeNeumann", "Volume Neumann", Core::Conditions::VolumeNeumann, true,
          Core::Conditions::geometry_type_volume);

  auto all_neumann_conditions = {pointneumann, pointneumanneb, lineneumann, surfneumann, volneumann,
      pointtransportneumann, linetransportneumann, surftransportneumann, voltransportneumann,
      pointthermoneumann, linethermoneumann, surfthermoneumann, volthermoneumann, pointporoneumann,
      lineporoneumann, surfporoneumann, volporoneumann};

  for (const auto& cond : all_neumann_conditions)
  {
    add_named_int(cond, "NUMDOF");
    add_named_int_vector(cond, "ONOFF", "onoff", "NUMDOF");
    add_named_real_vector(cond, "VAL", "values", "NUMDOF");
    add_named_int_vector(cond, "FUNCT", "function ids", "NUMDOF", 0, false);
    add_named_selection_component(cond, "TYPE", "type", "Live",
        Teuchos::tuple<std::string>(
            "Live", "Dead", "pseudo_orthopressure", "orthopressure", "PressureGrad"),
        Teuchos::tuple<std::string>("neum_live", "neum_dead", "neum_pseudo_orthopressure",
            "neum_orthopressure", "neum_pgrad"),
        true);

    condlist.emplace_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // Dirichlet conditions
  std::shared_ptr<Core::Conditions::ConditionDefinition> pointdirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN POINT DIRICH CONDITIONS",
          "Dirichlet", "Point Dirichlet", Core::Conditions::PointDirichlet, false,
          Core::Conditions::geometry_type_point);
  std::shared_ptr<Core::Conditions::ConditionDefinition> linedirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN LINE DIRICH CONDITIONS",
          "Dirichlet", "Line Dirichlet", Core::Conditions::LineDirichlet, false,
          Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfdirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN SURF DIRICH CONDITIONS",
          "Dirichlet", "Surface Dirichlet", Core::Conditions::SurfaceDirichlet, false,
          Core::Conditions::geometry_type_surface);
  std::shared_ptr<Core::Conditions::ConditionDefinition> voldirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN VOL DIRICH CONDITIONS",
          "Dirichlet", "Volume Dirichlet", Core::Conditions::VolumeDirichlet, false,
          Core::Conditions::geometry_type_volume);

  std::shared_ptr<Core::Conditions::ConditionDefinition> pointaledirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN POINT ALE DIRICH CONDITIONS",
          "ALEDirichlet", "Point Dirichlet", Core::Conditions::PointDirichlet, false,
          Core::Conditions::geometry_type_point);
  std::shared_ptr<Core::Conditions::ConditionDefinition> linealedirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN LINE ALE DIRICH CONDITIONS",
          "ALEDirichlet", "Line Dirichlet", Core::Conditions::LineDirichlet, false,
          Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfaledirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN SURF ALE DIRICH CONDITIONS",
          "ALEDirichlet", "Surface Dirichlet", Core::Conditions::SurfaceDirichlet, false,
          Core::Conditions::geometry_type_surface);
  std::shared_ptr<Core::Conditions::ConditionDefinition> volaledirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN VOL ALE DIRICH CONDITIONS",
          "ALEDirichlet", "Volume Dirichlet", Core::Conditions::VolumeDirichlet, false,
          Core::Conditions::geometry_type_volume);

  // Dirichlet conditions for transport problems
  std::shared_ptr<Core::Conditions::ConditionDefinition> pointtransportdirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN POINT TRANSPORT DIRICH CONDITIONS", "TransportDirichlet", "Point Dirichlet",
          Core::Conditions::PointDirichlet, false, Core::Conditions::geometry_type_point);
  std::shared_ptr<Core::Conditions::ConditionDefinition> linetransportdirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE TRANSPORT DIRICH CONDITIONS", "TransportDirichlet", "Line Dirichlet",
          Core::Conditions::LineDirichlet, false, Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surftransportdirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF TRANSPORT DIRICH CONDITIONS", "TransportDirichlet", "Surface Dirichlet",
          Core::Conditions::SurfaceDirichlet, false, Core::Conditions::geometry_type_surface);
  std::shared_ptr<Core::Conditions::ConditionDefinition> voltransportdirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN VOL TRANSPORT DIRICH CONDITIONS", "TransportDirichlet", "Volume Dirichlet",
          Core::Conditions::VolumeDirichlet, false, Core::Conditions::geometry_type_volume);

  // Dirichlet conditions for thermo problems
  std::shared_ptr<Core::Conditions::ConditionDefinition> pointthermodirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN POINT THERMO DIRICH CONDITIONS", "ThermoDirichlet", "Point Dirichlet",
          Core::Conditions::PointDirichlet, false, Core::Conditions::geometry_type_point);
  std::shared_ptr<Core::Conditions::ConditionDefinition> linethermodirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE THERMO DIRICH CONDITIONS", "ThermoDirichlet", "Line Dirichlet",
          Core::Conditions::LineDirichlet, false, Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfthermodirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF THERMO DIRICH CONDITIONS", "ThermoDirichlet", "Surface Dirichlet",
          Core::Conditions::SurfaceDirichlet, false, Core::Conditions::geometry_type_surface);
  std::shared_ptr<Core::Conditions::ConditionDefinition> volthermodirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN VOL THERMO DIRICH CONDITIONS",
          "ThermoDirichlet", "Volume Dirichlet", Core::Conditions::VolumeDirichlet, false,
          Core::Conditions::geometry_type_volume);

  // Dirichlet conditions for poroelasticity problems
  std::shared_ptr<Core::Conditions::ConditionDefinition> pointporodirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN POINT PORO DIRICH CONDITIONS",
          "PoroDirichlet", "Point Dirichlet", Core::Conditions::PointDirichlet, false,
          Core::Conditions::geometry_type_point);
  std::shared_ptr<Core::Conditions::ConditionDefinition> lineporodirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN LINE PORO DIRICH CONDITIONS",
          "PoroDirichlet", "Line Dirichlet", Core::Conditions::LineDirichlet, false,
          Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfporodirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN SURF PORO DIRICH CONDITIONS",
          "PoroDirichlet", "Surface Dirichlet", Core::Conditions::SurfaceDirichlet, false,
          Core::Conditions::geometry_type_surface);
  std::shared_ptr<Core::Conditions::ConditionDefinition> volporodirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN VOL PORO DIRICH CONDITIONS",
          "PoroDirichlet", "Volume Dirichlet", Core::Conditions::VolumeDirichlet, false,
          Core::Conditions::geometry_type_volume);

  std::shared_ptr<Core::Conditions::ConditionDefinition> pointnurbslsdirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN POINT NURBS LS DIRICH CONDITIONS", "NurbsLSDirichlet", "Point Dirichlet",
          Core::Conditions::PointDirichlet, true, Core::Conditions::geometry_type_point);
  std::shared_ptr<Core::Conditions::ConditionDefinition> linenurbslsdirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE NURBS LS DIRICH CONDITIONS", "NurbsLSDirichlet", "Line Dirichlet",
          Core::Conditions::LineDirichlet, true, Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfnurbslsdirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF NURBS LS DIRICH CONDITIONS", "NurbsLSDirichlet", "Surface Dirichlet",
          Core::Conditions::SurfaceDirichlet, true, Core::Conditions::geometry_type_surface);
  std::shared_ptr<Core::Conditions::ConditionDefinition> volnurbslsdirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN VOL NURBS LS DIRICH CONDITIONS", "NurbsLSDirichlet", "Volume Dirichlet",
          Core::Conditions::VolumeDirichlet, true, Core::Conditions::geometry_type_volume);

  auto all_dirichlet_conditions = {pointdirichlet, linedirichlet, surfdirichlet, voldirichlet,
      pointaledirichlet, linealedirichlet, surfaledirichlet, volaledirichlet,
      pointtransportdirichlet, linetransportdirichlet, surftransportdirichlet,
      voltransportdirichlet, pointthermodirichlet, linethermodirichlet, surfthermodirichlet,
      volthermodirichlet, pointporodirichlet, lineporodirichlet, surfporodirichlet,
      volporodirichlet, pointnurbslsdirichlet, linenurbslsdirichlet, surfnurbslsdirichlet,
      volnurbslsdirichlet};

  for (const auto& cond : all_dirichlet_conditions)
  {
    add_named_int(cond, "NUMDOF");
    add_named_int_vector(cond, "ONOFF", "", "NUMDOF");
    add_named_real_vector(cond, "VAL", "", "NUMDOF");
    add_named_int_vector(cond, "FUNCT", "", "NUMDOF", 0, false);

    // optional
    add_named_selection_component(cond, "TAG", "", "none",
        Teuchos::tuple<std::string>("none", "monitor_reaction"),
        Teuchos::tuple<std::string>("none", "monitor_reaction"), true);

    condlist.emplace_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // Point coupling (e.g. joints - couple X out of Y nodal DoFs)

  std::shared_ptr<Core::Conditions::ConditionDefinition> pointcoupling =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN POINT COUPLING CONDITIONS",
          "PointCoupling", "Point Coupling", Core::Conditions::PointCoupling, false,
          Core::Conditions::geometry_type_point);

  std::shared_ptr<Core::Conditions::ConditionDefinition> pointthermocoupling =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN POINT THERMO COUPLING CONDITIONS", "PointThermoCoupling", "Point Coupling",
          Core::Conditions::PointCoupling, false, Core::Conditions::geometry_type_point);

  for (const auto& cond : {pointcoupling, pointthermocoupling})
  {
    add_named_int(cond, "NUMDOF");
    add_named_int_vector(cond, "ONOFF", "", "NUMDOF");

    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // Initial fields

  // general initial field conditions
  std::shared_ptr<Core::Conditions::ConditionDefinition> pointinitfields =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN POINT INITIAL FIELD CONDITIONS", "Initfield", "Point Initfield",
          Core::Conditions::PointInitfield, false, Core::Conditions::geometry_type_point);
  std::shared_ptr<Core::Conditions::ConditionDefinition> lineinitfields =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE INITIAL FIELD CONDITIONS", "Initfield", "Line Initfield",
          Core::Conditions::LineInitfield, false, Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfinitfields =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF INITIAL FIELD CONDITIONS", "Initfield", "Surface Initfield",
          Core::Conditions::SurfaceInitfield, false, Core::Conditions::geometry_type_surface);
  std::shared_ptr<Core::Conditions::ConditionDefinition> volinitfields =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN VOL INITIAL FIELD CONDITIONS",
          "Initfield", "Volume Initfield", Core::Conditions::VolumeInitfield, false,
          Core::Conditions::geometry_type_volume);

  for (const auto& cond : {pointinitfields, lineinitfields, surfinitfields, volinitfields})
  {
    add_named_selection_component(cond, "FIELD", "init field", "Undefined",
        Teuchos::tuple<std::string>("Undefined", "Velocity", "Pressure", "Temperature", "ScaTra",
            "Porosity", "PoroMultiFluid", "Artery"),
        Teuchos::tuple<std::string>("Undefined", "Velocity", "Pressure", "Temperature", "ScaTra",
            "Porosity", "PoroMultiFluid", "Artery"));

    // for initial vector fields, use the COMPONENT option of our functions
    add_named_int(cond, "FUNCT");

    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // define initial field that can be set on thermo simulations that use the ScaTra
  // discretization e.g. STI, SSTI

  // initial field conditions for temperature on ScaTra discretizations
  std::shared_ptr<Core::Conditions::ConditionDefinition> pointthermoinitfields =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN POINT THERMO INITIAL FIELD CONDITIONS", "ThermoInitfield",
          "Set the initial temperature field if the thermo field is solved using a ScaTra "
          "discretization (e.g. STI, SSTI) on points",
          Core::Conditions::PointInitfield, false, Core::Conditions::geometry_type_point);
  std::shared_ptr<Core::Conditions::ConditionDefinition> linethermoinitfields =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE THERMO INITIAL FIELD CONDITIONS", "ThermoInitfield",
          "Set the initial temperature field if the thermo field is solved using a ScaTra "
          "discretization (e.g. STI, SSTI) on lines",
          Core::Conditions::LineInitfield, false, Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfthermoinitfields =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF THERMO INITIAL FIELD CONDITIONS", "ThermoInitfield",
          "Set the initial temperature field if the thermo field is solved using a ScaTra "
          "discretization (e.g. STI, SSTI) on surfaces",
          Core::Conditions::SurfaceInitfield, false, Core::Conditions::geometry_type_surface);
  std::shared_ptr<Core::Conditions::ConditionDefinition> volthermoinitfields =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN VOL THERMO INITIAL FIELD CONDITIONS", "ThermoInitfield",
          "Set the initial temperature field if the thermo field is solved using a ScaTra "
          "discretization (e.g. STI, SSTI) on volumes",
          Core::Conditions::VolumeInitfield, false, Core::Conditions::geometry_type_volume);

  for (const auto& cond :
      {pointthermoinitfields, linethermoinitfields, surfthermoinitfields, volthermoinitfields})
  {
    add_named_selection_component(cond, "FIELD", "init field", "Undefined",
        Teuchos::tuple<std::string>("Undefined", "ScaTra"),
        Teuchos::tuple<std::string>("Undefined", "ScaTra"));
    add_named_int(cond, "FUNCT");

    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // compute domain integrals, i.e., cumulative volumes of 3D domain elements or cumulative
  // surface areas of 2D domain elements

  // definition of surface and volume conditions for domain integral computation
  std::shared_ptr<Core::Conditions::ConditionDefinition> domainintegralsurf =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN DOMAIN INTEGRAL SURF CONDITIONS", "DomainIntegral",
          "compute cumulative surface areas of 2D domain elements",
          Core::Conditions::DomainIntegral, true, Core::Conditions::geometry_type_surface);

  std::shared_ptr<Core::Conditions::ConditionDefinition> domainintegralvol =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN DOMAIN INTEGRAL VOL CONDITIONS", "DomainIntegral",
          "compute cumulative volumes of 3D domain elements", Core::Conditions::DomainIntegral,
          true, Core::Conditions::geometry_type_volume);

  for (const auto& cond : {domainintegralsurf, domainintegralvol})
  {
    // add input file line components to condition definitions
    add_named_int(cond, "ConditionID");

    // insert condition definitions into global list of valid condition definitions
    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // compute boundary integrals, i.e., cumulative surface areas of 2D boundary elements

  std::shared_ptr<Core::Conditions::ConditionDefinition> boundaryintegralsurf =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN BOUNDARY INTEGRAL SURF CONDITIONS", "BoundaryIntegral",
          "compute cumulative surface areas of 2D boundary elements",
          Core::Conditions::BoundaryIntegral, true, Core::Conditions::geometry_type_surface);

  // add input file line components to condition definition
  add_named_int(boundaryintegralsurf, "ConditionID");

  // insert condition definition into global list of valid condition definitions
  condlist.push_back(boundaryintegralsurf);


  /*--------------------------------------------------------------------*/
  // wear in ALE description

  std::shared_ptr<Core::Conditions::ConditionDefinition> linealewear =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN LINE ALE WEAR CONDITIONS 2D",
          "AleWear", "Line Ale Wear", Core::Conditions::AleWear, true,
          Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfalewear =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN SURFACE WEAR CONDITIONS 3D",
          "AleWear", "Surface Ale Wear", Core::Conditions::AleWear, true,
          Core::Conditions::geometry_type_surface);

  condlist.push_back(linealewear);
  condlist.push_back(surfalewear);

  /*--------------------------------------------------------------------*/
  // local coordinate systems

  std::shared_ptr<Core::Conditions::ConditionDefinition> pointlocsys =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN POINT LOCSYS CONDITIONS",
          "Locsys", "Point local coordinate system", Core::Conditions::PointLocsys, true,
          Core::Conditions::geometry_type_point);
  std::shared_ptr<Core::Conditions::ConditionDefinition> linelocsys =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN LINE LOCSYS CONDITIONS",
          "Locsys", "Line local coordinate system", Core::Conditions::LineLocsys, true,
          Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surflocsys =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN SURF LOCSYS CONDITIONS",
          "Locsys", "Surface local coordinate system", Core::Conditions::SurfaceLocsys, true,
          Core::Conditions::geometry_type_surface);
  std::shared_ptr<Core::Conditions::ConditionDefinition> vollocsys =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN VOL LOCSYS CONDITIONS",
          "Locsys", "Volume local coordinate system", Core::Conditions::VolumeLocsys, true,
          Core::Conditions::geometry_type_volume);

  // add components to condition definitions
  for (const auto& cond : {pointlocsys, linelocsys, surflocsys, vollocsys})
  {
    add_named_real_vector(cond, "ROTANGLE", "", 3);
    add_named_int_vector(cond, "FUNCT", "", 3);
    add_named_int(cond, "USEUPDATEDNODEPOS");
  }

  // add node normal system option only for lines and surfaces
  for (const auto& cond : {linelocsys, surflocsys})
  {
    add_named_int(cond, "USECONSISTENTNODENORMAL");
  }

  // add conditions to global list of conditions
  for (const auto& cond : {pointlocsys, linelocsys, surflocsys, vollocsys})
  {
    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // periodic boundary

  std::shared_ptr<Core::Conditions::ConditionDefinition> lineperiodic =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE PERIODIC BOUNDARY CONDITIONS", "LinePeriodic", "Line Periodic",
          Core::Conditions::LinePeriodic, false, Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfperiodic =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF PERIODIC BOUNDARY CONDITIONS", "SurfacePeriodic", "Surface Periodic",
          Core::Conditions::SurfacePeriodic, false, Core::Conditions::geometry_type_surface);

  for (const auto& cond : {lineperiodic, surfperiodic})
  {
    add_named_int(cond, "ID", "periodic boundary condition id", 0, false);
    add_named_selection_component(cond, "MASTER_OR_SLAVE", "master-slave toggle", "Master",
        Teuchos::tuple<std::string>("Master", "Slave"),
        Teuchos::tuple<std::string>("Master", "Slave"));
    add_named_selection_component(cond, "PLANE", "degrees of freedom for the pbc plane", "xy",
        Teuchos::tuple<std::string>("xy", "yx", "yz", "zy", "xz", "zx", "xyz"),
        Teuchos::tuple<std::string>("xy", "xy", "yz", "yz", "xz", "xz", "xyz"));
    add_named_int(cond, "LAYER", "layer of periodic boundary condition", 0, false);
    add_named_real(cond, "ANGLE", "angle of rotation");
    add_named_real(cond, "ABSTREETOL", "tolerance for nodematching in octree");

    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // weak Dirichlet conditions
  std::shared_ptr<Core::Conditions::ConditionDefinition> lineweakdirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE WEAK DIRICHLET CONDITIONS", "LineWeakDirichlet", "LineWeakDirichlet",
          Core::Conditions::LineWeakDirichlet, true, Core::Conditions::geometry_type_line);

  std::shared_ptr<Core::Conditions::ConditionDefinition> surfweakdirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURFACE WEAK DIRICHLET CONDITIONS", "SurfaceWeakDirichlet",
          "SurfaceWeakDirichlet", Core::Conditions::SurfaceWeakDirichlet, true,
          Core::Conditions::geometry_type_surface);

  // attach all components to those condition
  for (const auto& cond : {lineweakdirichlet, surfweakdirichlet})
  {
    // weak DBCs can be imposed adjoint consistent or adjoint inconsistent
    add_named_selection_component(cond, "GAMMATYPE", "Choice of gamma parameter",
        "adjoint-consistent",
        Teuchos::tuple<std::string>("adjoint-consistent", "diffusive-optimal"),
        Teuchos::tuple<std::string>("adjoint-consistent", "diffusive-optimal"));

    // weak DBCs can be imposed in all directions or only in normal direction
    // (SCATRA: not checked, only in all_directions so far)
    add_named_selection_component(cond, "DIR", "Directions to apply weak dbc", "all_directions",
        Teuchos::tuple<std::string>("all_directions", "only_in_normal_direction"),
        Teuchos::tuple<std::string>("all_directions", "only_in_normal_direction"));

    // FLUID: penalty parameter either computed dynamically (using Spaldings law of
    // the wall) or by a fixed value; SCATRA: not checked, only constant value so far
    add_named_selection_component(cond, "PENTYPE", "Definition of penalty parameter", "constant",
        Teuchos::tuple<std::string>("constant", "Spalding"),
        Teuchos::tuple<std::string>("constant", "Spalding"));

    // scaling factor for penalty parameter tauB or
    // stabilization parameter alpha for Nitsche term
    // (SCATRA: if stabilization parameter negative -> mixed-hybrid formulation)
    add_named_real(cond, "TauBscaling");

    // linearisation strategies --- the linearisation (i.e. the matrix
    // contribution) of the convective term on the inflow could be
    // suppressed, since the flux is a kink function and including this one
    // might result in even worse convergence behaviour
    // (SCATRA: not checked)
    add_named_selection_component(cond, "LINEARISATION", "Linearisation", "lin_all",
        Teuchos::tuple<std::string>("lin_all", "no_lin_conv_inflow"),
        Teuchos::tuple<std::string>("lin_all", "no_lin_conv_inflow"));

    // we provide a vector of 3 values for velocities
    add_named_real_vector(cond, "VAL", "values", 3);

    // and optional spatial functions
    add_named_int_vector(cond, "FUNCT", "function ids", 3, 0, true);
    // append the condition to the list of all conditions
    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // boundary for superconvergent patch recovery (SPR)

  std::shared_ptr<Core::Conditions::ConditionDefinition> linespr =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN PATCH RECOVERY BOUNDARY LINE CONDITIONS", "SPRboundary", "Boundary for SPR",
          Core::Conditions::SPRboundary, false, Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfspr =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN PATCH RECOVERY BOUNDARY SURF CONDITIONS", "SPRboundary", "Boundary for SPR",
          Core::Conditions::SPRboundary, false, Core::Conditions::geometry_type_surface);

  condlist.push_back(linespr);
  condlist.push_back(surfspr);

  /*--------------------------------------------------------------------*/
  // volume constraint

  std::shared_ptr<Core::Conditions::ConditionDefinition> volumeconstraint =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN SURFACE VOLUME CONSTRAINT 3D",
          "VolumeConstraint_3D", "Surface Volume Constraint", Core::Conditions::VolumeConstraint_3D,
          true, Core::Conditions::geometry_type_surface);

  add_named_int(volumeconstraint, "ConditionID");
  add_named_int(volumeconstraint, "curve", "id of the curve", 0, false);
  add_named_real(volumeconstraint, "activeTime");
  add_named_selection_component(volumeconstraint, "projection", "projection", "none",
      Teuchos::tuple<std::string>("none", "xy", "yz", "xz"),
      Teuchos::tuple<std::string>("none", "xy", "yz", "xz"), true);


  condlist.push_back(volumeconstraint);

  /*--------------------------------------------------------------------*/
  // volume constraint penalty

  std::shared_ptr<Core::Conditions::ConditionDefinition> volumeconstraintpen =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURFACE VOLUME CONSTRAINT 3D PEN", "VolumeConstraint_3D_Pen",
          "Surface Volume Constraint Penalty", Core::Conditions::VolumeConstraint_3D_pen, true,
          Core::Conditions::geometry_type_surface);

  add_named_int(volumeconstraintpen, "ConditionID");
  add_named_int(volumeconstraintpen, "curve", "id of the curve", 0, false);
  add_named_real(volumeconstraintpen, "activeTime");
  add_named_real(volumeconstraintpen, "penalty");
  add_named_real(volumeconstraintpen, "rho");
  add_named_selection_component(volumeconstraintpen, "projection", "projection", "none",
      Teuchos::tuple<std::string>("none", "xy", "yz", "xz"),
      Teuchos::tuple<std::string>("none", "xy", "yz", "xz"), true);

  condlist.push_back(volumeconstraintpen);

  /*--------------------------------------------------------------------*/
  // area constraint

  std::shared_ptr<Core::Conditions::ConditionDefinition> areaconstraint =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN SURFACE AREA CONSTRAINT 3D",
          "AreaConstraint_3D", "Surface Area Constraint", Core::Conditions::AreaConstraint_3D, true,
          Core::Conditions::geometry_type_surface);

  add_named_int(areaconstraint, "ConditionID");
  add_named_int(areaconstraint, "curve", "id of the curve", 0, false);
  add_named_real(areaconstraint, "activeTime");

  condlist.push_back(areaconstraint);


  /*--------------------------------------------------------------------*/
  // volume monitor

  std::shared_ptr<Core::Conditions::ConditionDefinition> volumemonitor =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN SURFACE VOLUME MONITOR 3D",
          "VolumeMonitor_3D", "Surface Volume Monitor", Core::Conditions::VolumeMonitor_3D, true,
          Core::Conditions::geometry_type_surface);

  add_named_int(volumemonitor, "ConditionID");

  condlist.push_back(volumemonitor);

  /*--------------------------------------------------------------------*/
  // area monitor 3D

  std::shared_ptr<Core::Conditions::ConditionDefinition> areamonitor =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN SURFACE AREA MONITOR 3D",
          "AreaMonitor_3D", "Surface Area Monitor", Core::Conditions::AreaMonitor_3D, true,
          Core::Conditions::geometry_type_surface);

  add_named_int(areamonitor, "ConditionID");
  add_named_selection_component(areamonitor, "projection", "projection", "none",
      Teuchos::tuple<std::string>("none", "xy", "yz", "xz"),
      Teuchos::tuple<std::string>("none", "xy", "yz", "xz"), true);

  condlist.push_back(areamonitor);

  /*--------------------------------------------------------------------*/
  // area constraint

  std::shared_ptr<Core::Conditions::ConditionDefinition> areaconstraint2D =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN LINE AREA CONSTRAINT 2D",
          "AreaConstraint_2D", "Line Area Constraint", Core::Conditions::AreaConstraint_2D, true,
          Core::Conditions::geometry_type_line);

  add_named_int(areaconstraint2D, "ConditionID");
  add_named_int(areaconstraint2D, "curve", {}, 0, false);
  add_named_real(areaconstraint2D, "activeTime");

  condlist.push_back(areaconstraint2D);

  /*--------------------------------------------------------------------*/
  // area monitor 2D

  std::shared_ptr<Core::Conditions::ConditionDefinition> areamonitor2D =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN LINE AREA MONITOR 2D",
          "AreaMonitor_2D", "Line Area Monitor", Core::Conditions::AreaMonitor_2D, true,
          Core::Conditions::geometry_type_line);

  add_named_int(areamonitor2D, "ConditionID");

  condlist.push_back(areamonitor2D);

  /*--------------------------------------------------------------------*/
  // Multi point constraint in 3D for a node over a plane

  std::shared_ptr<Core::Conditions::ConditionDefinition> nodeonplaneconst3D =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURFACE MULTIPNT CONSTRAINT 3D", "MPC_NodeOnPlane_3D", "Node on Plane Constraint",
          Core::Conditions::MPC_NodeOnPlane_3D, false, Core::Conditions::geometry_type_surface);

  add_named_int(nodeonplaneconst3D, "ConditionID");
  add_named_real(nodeonplaneconst3D, "amplitude");
  add_named_int(nodeonplaneconst3D, "curve", {}, 0, false);
  add_named_real(nodeonplaneconst3D, "activeTime");
  add_named_int_vector(nodeonplaneconst3D, "planeNodes", "ids of the nodes spanning the plane", 3);
  add_named_selection_component(nodeonplaneconst3D, "control", "relative or absolute control",
      "rel", Teuchos::tuple<std::string>("rel", "abs"), Teuchos::tuple<std::string>("rel", "abs"),
      true);

  condlist.push_back(nodeonplaneconst3D);

  /*--------------------------------------------------------------------*/
  // Multi point constraint in 3D, moving all constraint nodes synchronously

  std::shared_ptr<Core::Conditions::ConditionDefinition> nodemasterconst3D =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURFACE NORMALDIR MULTIPNT CONSTRAINT 3D", "MPC_NormalComponent_3D",
          "Node on Plane Constraint", Core::Conditions::MPC_NormalComponent_3D, false,
          Core::Conditions::geometry_type_surface);

  add_named_int(nodemasterconst3D, "ConditionID");
  add_named_real(nodemasterconst3D, "amplitude");
  add_named_int(nodemasterconst3D, "curve", {}, 0, false);
  add_named_real(nodemasterconst3D, "activeTime");
  add_named_int(nodemasterconst3D, "masterNode");
  add_named_real_vector(nodemasterconst3D, "direction", "direction", 3);
  add_named_selection_component(nodemasterconst3D, "value", "value", "disp",
      Teuchos::tuple<std::string>("disp", "x"), Teuchos::tuple<std::string>("disp", "x"), true);
  add_named_selection_component(nodemasterconst3D, "control", "relative or absolute control", "rel",
      Teuchos::tuple<std::string>("rel", "abs"), Teuchos::tuple<std::string>("rel", "abs"), true);

  condlist.push_back(nodemasterconst3D);

  /*--------------------------------------------------------------------*/
  // Multi point constraint in 3D, moving all constraint nodes synchronously, penalty based

  std::shared_ptr<Core::Conditions::ConditionDefinition> nodemasterconst3Dpen =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURFACE NORMALDIR MULTIPNT CONSTRAINT 3D PEN", "MPC_NormalComponent_3D_Pen",
          "Node on Plane Constraint Penalty", Core::Conditions::MPC_NormalComponent_3D_pen, false,
          Core::Conditions::geometry_type_surface);


  add_named_int(nodemasterconst3Dpen, "ConditionID");
  add_named_real(nodemasterconst3Dpen, "amplitude");
  add_named_int(nodemasterconst3Dpen, "curve", {}, 0, false);
  add_named_real(nodemasterconst3Dpen, "activeTime");
  add_named_real(nodemasterconst3Dpen, "penalty");
  add_named_int(nodemasterconst3Dpen, "masterNode");
  add_named_int_vector(nodemasterconst3Dpen, "direction", "direction", 3);
  add_named_selection_component(nodemasterconst3Dpen, "value", "value", "disp",
      Teuchos::tuple<std::string>("disp", "x"), Teuchos::tuple<std::string>("disp", "x"), true);
  add_named_selection_component(nodemasterconst3Dpen, "control", "relative or absolute control",
      "rel", Teuchos::tuple<std::string>("rel", "abs"), Teuchos::tuple<std::string>("rel", "abs"),
      true);

  condlist.push_back(nodemasterconst3Dpen);
  /*--------------------------------------------------------------------*/
  // Multi point constraint in 2D for a node on a line
  std::shared_ptr<Core::Conditions::ConditionDefinition> nodeonlineconst2D =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN LINE MULTIPNT CONSTRAINT 2D",
          "MPC_NodeOnLine_2D", "Node on Line Constraint", Core::Conditions::MPC_NodeOnLine_2D,
          false, Core::Conditions::geometry_type_line);

  add_named_int(nodeonlineconst2D, "ConditionID");
  add_named_real(nodeonlineconst2D, "amplitude");
  add_named_int(nodeonlineconst2D, "curve", {}, 0, false);
  add_named_int(nodeonlineconst2D, "constrNode1");
  add_named_int(nodeonlineconst2D, "constrNode2");
  add_named_int(nodeonlineconst2D, "constrNode3");
  ;
  add_named_selection_component(nodeonlineconst2D, "control", "distance or angle control", "dist",
      Teuchos::tuple<std::string>("dist", "angle"), Teuchos::tuple<std::string>("dist", "angle"),
      true);
  add_named_real(nodeonlineconst2D, "activeTime");

  condlist.push_back(nodeonlineconst2D);

  /*--------------------------------------------------------------------*/
  // Krylov Space Projection:
  // ========================
  // specify an unsupported (i.e. rigid body or zero energy) mode orthogonal
  // that is not excited by the body force
  //
  // examples are:
  // fluid:  pure Dirichlet (i.e. velocity) boundary conditions - pressure
  //         level is undetermined
  // scatra: pure Neumann boundary condition - level of scalar quantity is
  //         undetermined
  // solid:  insufficient support of translational or rotational rigid body
  //         modes
  //
  // for fluid and scatra, NUMDOF needs to be the number of dofs per node. the
  // following ONOFF values trigger the fixation of the quantity level (for
  // fluid, only pressure is allowed).
  // for solid NUMDOF is the number of potential rigid body modes (e.g. 6 for
  // 3-D solid, 3 for 2-D solid), where ONOFF triggers first the
  // translational followed by the rotational modes, each in/around x to z

  std::shared_ptr<Core::Conditions::ConditionDefinition> surfrigidbodymode =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF MODE FOR KRYLOV SPACE PROJECTION", "KrylovSpaceProjection",
          "Surface mode for Krylov space projection", Core::Conditions::SurfaceModeKrylovProjection,
          true, Core::Conditions::geometry_type_surface);

  std::shared_ptr<Core::Conditions::ConditionDefinition> volrigidbodymode =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN VOL MODE FOR KRYLOV SPACE PROJECTION", "KrylovSpaceProjection",
          "Volume mode for Krylov space projection", Core::Conditions::VolumeModeKrylovProjection,
          true, Core::Conditions::geometry_type_volume);

  for (const auto& cond : {surfrigidbodymode, volrigidbodymode})
  {
    add_named_selection_component(cond, "DIS", "discretization", "fluid",
        Teuchos::tuple<std::string>("fluid", "scatra", "solid"),
        Teuchos::tuple<std::string>("fluid", "scatra", "solid"));
    add_named_int(cond, "NUMMODES");
    add_named_int_vector(cond, "ONOFF", "", "NUMMODES");
    add_named_selection_component(cond, "WEIGHTVECDEF", "weight vector definition", "integration",
        Teuchos::tuple<std::string>("integration", "pointvalues"),
        Teuchos::tuple<std::string>("integration", "pointvalues"));

    condlist.push_back(cond);
  }


  // Finally, add the problem-specific conditions from the various modules
  Inpar::Mortar::set_valid_conditions(condlist);
  Inpar::S2I::set_valid_conditions(condlist);
  Inpar::ScaTra::set_valid_conditions(condlist);
  Inpar::STI::set_valid_conditions(condlist);
  Inpar::ElCh::set_valid_conditions(condlist);
  Inpar::ElectroPhysiology::set_valid_conditions(condlist);
  Inpar::FLUID::set_valid_conditions(condlist);
  Inpar::ALE::set_valid_conditions(condlist);
  Inpar::FSI::set_valid_conditions(condlist);
  Inpar::FPSI::set_valid_conditions(condlist);
  Inpar::Immersed::set_valid_conditions(condlist);
  Inpar::XFEM::set_valid_conditions(condlist);
  Inpar::BioFilm::set_valid_conditions(condlist);
  Inpar::ArteryNetwork::set_valid_conditions(condlist);
  Inpar::ReducedLung::set_valid_conditions(condlist);
  Inpar::Cardiovascular0D::set_valid_conditions(condlist);
  Inpar::Solid::set_valid_conditions(condlist);
  Inpar::Thermo::set_valid_conditions(condlist);
  Inpar::SSI::set_valid_conditions(condlist);
  Inpar::SSTI::set_valid_conditions(condlist);
  Inpar::PARTICLE::set_valid_conditions(condlist);
  Inpar::LevelSet::set_valid_conditions(condlist);
  Inpar::EleMag::set_valid_conditions(condlist);
  Inpar::BeamPotential::set_valid_conditions(condlist);
  Inpar::RveMpc::set_valid_conditions(condlist);
  Inpar::BeamInteraction::set_valid_conditions(condlist);
  Inpar::EHL::set_valid_conditions(condlist);
  Inpar::PoroMultiPhaseScaTra::set_valid_conditions(condlist);

  // finally some conditions that do not have their own files yet are problem-specific
  set_miscellaneous_conditions(condlist);

  return vc;
}

FOUR_C_NAMESPACE_CLOSE

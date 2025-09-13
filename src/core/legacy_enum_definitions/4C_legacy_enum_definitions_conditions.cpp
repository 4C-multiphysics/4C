// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_legacy_enum_definitions_conditions.hpp"

#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

std::string_view Core::Conditions::to_string(const Core::Conditions::ConditionType condition_type)
{
  switch (condition_type)
  {
    case Core::Conditions::PointDirichlet:
      return "Point Dirichlet boundary condition";
    case Core::Conditions::LineDirichlet:
      return "Line Dirichlet boundary condition";
    case Core::Conditions::SurfaceDirichlet:
      return "Surface Dirichlet boundary condition";
    case Core::Conditions::VolumeDirichlet:
      return "Volume Dirichlet boundary condition";
    case Core::Conditions::PointNeumann:
      return "Point Neumann boundary condition";
    case Core::Conditions::LineNeumann:
      return "Line Neumann boundary condition";
    case Core::Conditions::SurfaceNeumann:
      return "Surface Neumann boundary condition";
    case Core::Conditions::VolumeNeumann:
      return "Volume Neumann boundary condition";
    case Core::Conditions::PointNeumannEB:
      return "Point Neumann boundary condition for Euler-Bernoulli beam";
    case Core::Conditions::PointInitfield:
      return "Point Initfield boundary condition";
    case Core::Conditions::LineInitfield:
      return "Line Initfield boundary condition";
    case Core::Conditions::SurfaceInitfield:
      return "Surface Initfield boundary condition";
    case Core::Conditions::VolumeInitfield:
      return "Volume Initfield boundary condition";
    case Core::Conditions::Mortar:
      return "Mortar coupling boundary condition";
    case Core::Conditions::MortarMulti:
      return "Mortar multi-coupling boundary condition";
    case Core::Conditions::Contact:
      return "Mortar contact boundary condition";
    case Core::Conditions::AleWear:
      return "ALE Wear boundary condition";
    case Core::Conditions::LineMrtrSym:
      return "Line contact symmetry condition";
    case Core::Conditions::PointMrtrSym:
      return "Point contact symmetry condition";
    case Core::Conditions::EdgeMrtr:
      return "Geometrical edge for 2D/3D contact";
    case Core::Conditions::CornerMrtr:
      return "Geometrical corner for 2/3D contact";
    case Core::Conditions::EHLCoupling:
      return "EHL (elastohydrodynamic lubrication) coupling condition";
    case Core::Conditions::PointLocsys:
      return "Point local coordinate system condition";
    case Core::Conditions::LineLocsys:
      return "Line local coordinate system condition";
    case Core::Conditions::SurfaceLocsys:
      return "Surface local coordinate system condition";
    case Core::Conditions::VolumeLocsys:
      return "Volume local coordinate system condition";
    case Core::Conditions::SPRboundary:
      return "Superconvergent Patch Recovery boundary condition";
    case Core::Conditions::FSICoupling:
      return "FSI Coupling condition";
    case Core::Conditions::fpsi_coupling:
      return "FPSI Coupling condition";
    case Core::Conditions::XFEM_Surf_Displacement:
      return "XFEM Surface Displacement condition";
    case Core::Conditions::XFEM_Levelset_Weak_Dirichlet:
      return "XFEM Levelset weak Dirichlet boundary condition";
    case Core::Conditions::XFEM_Levelset_Navier_Slip:
      return "XFEM Levelset Navier Slip boundary condition";
    case Core::Conditions::XFEM_Robin_Dirichlet_Volume:
      return "XFEM Levelset Navier Slip Robin(Dirichlet)-Volume Condition";
    case Core::Conditions::XFEM_Robin_Neumann_Volume:
      return "XFEM Levelset Navier Slip Robin(Neumann)-Volume Condition";
    case Core::Conditions::XFEM_Levelset_Neumann:
      return "XFEM Levelset Neumann boundary condition";
    case Core::Conditions::XFEM_Levelset_Twophase:
      return "XFEM Levelset Twophase coupling condition";
    case Core::Conditions::XFEM_Surf_FSIPart:
      return "XFEM Surface partitioned XFSI boundary condition";
    case Core::Conditions::XFEM_Surf_FSIMono:
      return "XFEM Surface monolithic XFSI coupling condition";
    case Core::Conditions::XFEM_Surf_FPIMono:
      return "XFEM Surface monolithic XPSI coupling condition";
    case Core::Conditions::XFEM_Surf_FluidFluid:
      return "XFEM Surface Fluid-Fluid coupling condition";
    case Core::Conditions::XFEM_Surf_Weak_Dirichlet:
      return "XFEM Surface weak Dirichlet boundary condition";
    case Core::Conditions::XFEM_Surf_Neumann:
      return "XFEM Surface Neumann boundary condition";
    case Core::Conditions::XFEM_Surf_Navier_Slip:
      return "XFEM Surface Navier Slip boundary condition";
    case Core::Conditions::XFEM_Robin_Dirichlet_Surf:
      return "XFEM Mesh Navier Slip Robin(Dirichlet)-Volume Condition";
    case Core::Conditions::XFEM_Robin_Neumann_Surf:
      return "XFEM Mesh Navier Slip Robin(Neumann)-Volume Condition";
    case Core::Conditions::FluidFluidCoupling:
      return "Fluid Fluid Coupling condition";
    case Core::Conditions::ALEFluidCoupling:
      return "ALE Fluid Coupling condition";
    case Core::Conditions::FluidMesh:
      return "Create standalone fluid mesh from condition";
    case Core::Conditions::FluidStressCalc:
      return "Fluid stress calculation";
    case Core::Conditions::LineLIFTDRAG:
      return "Line LIFTDRAG condition";
    case Core::Conditions::SurfLIFTDRAG:
      return "Surf LIFTDRAG condition";
    case Core::Conditions::ALEUPDATECoupling:
      return "Ale update condition";
    case Core::Conditions::MicroBoundary:
      return "Microscale boundary condition";
    case Core::Conditions::VolumeConstraint_3D:
      return "Volume constraint surface boundary condition";
    case Core::Conditions::VolumeConstraint_3D_pen:
      return "Volume constraint surface boundary with penalty";
    case Core::Conditions::AreaConstraint_3D:
      return "Area constraint surface boundary condition";
    case Core::Conditions::AreaConstraint_2D:
      return "Area constraint surface boundary condition";
    case Core::Conditions::VolumeMonitor_3D:
      return "Volume monitor condition";
    case Core::Conditions::AreaMonitor_3D:
      return "Area monitor condition";
    case Core::Conditions::AreaMonitor_2D:
      return "Area monitor condition";
    case Core::Conditions::Cardiovascular0D4ElementWindkessel_Structure:
      return "Surface 0D cardiovascular 4-element windkessel condition";
    case Core::Conditions::Cardiovascular0DArterialProxDist_Structure:
      return "Surface 0D cardiovascular arterial prox dist condition";
    case Core::Conditions::Cardiovascular0DSysPulCirculation_Structure:
      return "Surface 0D cardiovascular sys-pul circulation condition";
    case Core::Conditions::CardiovascularRespiratory0DSysPulPeriphCirculation_Structure:
      return "Surface 0D cardiovascular respiratory sys-pul periph circulation condition";
    case Core::Conditions::Cardiovascular0DStructureCoupling:
      return "Surface 0D cardiovascular-structure coupling condition";
    case Core::Conditions::ImpedanceCond:
      return "Impedance boundary condition";
    case Core::Conditions::Impedance_Calb_Cond:
      return "Impedance calibration boundary condition";
    case Core::Conditions::MPC_NodeOnPlane_3D:
      return "Multipoint constraint on a plane";
    case Core::Conditions::MPC_NodeOnLine_3D:
      return "Multipoint constraint on a line";
    case Core::Conditions::MPC_NodeOnLine_2D:
      return "Multipoint constraint on a line";
    case Core::Conditions::MPC_NormalComponent_3D:
      return "Node on plane constraint";
    case Core::Conditions::MPC_NormalComponent_3D_pen:
      return "Node on plane penalty constraint";
    case Core::Conditions::LJ_Potential_Volume:
      return "Lennard-Jones potential in a volume";
    case Core::Conditions::LJ_Potential_Surface:
      return "Lennard-Jones potential on a surface";
    case Core::Conditions::LJ_Potential_Line:
      return "Lennard-Jones potential on a line";
    case Core::Conditions::VanDerWaals_Potential_Volume:
      return "Van der Waals potential in a volume";
    case Core::Conditions::VanDerWaals_Potential_Surface:
      return "Van der Waals potential on a surface";
    case Core::Conditions::VanDerWaals_Potential_Line:
      return "Van der Waals potential on a line";
    case Core::Conditions::ElectroRepulsion_Potential_Surface:
      return "Electro repulsion potential on a surface";
    case Core::Conditions::ElectroRepulsion_Potential_Line:
      return "Electro repulsion potential on a line";
    case Core::Conditions::LineFlowDepPressure:
      return "line flow-dependent pressure condition";
    case Core::Conditions::SurfaceFlowDepPressure:
      return "surface flow-dependent pressure condition";
    case Core::Conditions::LineSlipSupp:
      return "line slip supplemental curved boundary condition";
    case Core::Conditions::SurfaceSlipSupp:
      return "surface slip supplemental curved boundary condition";
    case Core::Conditions::LineNavierSlip:
      return "line navier-slip boundary condition";
    case Core::Conditions::SurfNavierSlip:
      return "surface navier-slip boundary condition";
    case Core::Conditions::LineWeakDirichlet:
      return "line weak Dirichlet condition";
    case Core::Conditions::SurfaceWeakDirichlet:
      return "surface weak Dirichlet condition";
    case Core::Conditions::LineMixHybDirichlet:
      return "Mixed/hybrid dirichlet line condition";
    case Core::Conditions::SurfaceMixHybDirichlet:
      return "Mixed/hybrid dirichlet surface condition";
    case Core::Conditions::SurfaceConservativeOutflowConsistency:
      return "Surace conservative outflow consistency condition";
    case Core::Conditions::LinePeriodic:
      return "line periodic boundary condition";
    case Core::Conditions::SurfacePeriodic:
      return "surface periodic boundary condition";
    case Core::Conditions::LineRvePeriodic:
      return "Line RVE periodic boundary condition";
    case Core::Conditions::SurfaceRvePeriodic:
      return "Surface RVE periodic boundary condition";
    case Core::Conditions::PointRvePeriodicReference:
      return "Reference point for RVE periodic boundary condition";
    case Core::Conditions::TransferTurbulentInflow:
      return "transfer turbulent inflow";
    case Core::Conditions::TurbulentInflowSection:
      return "turbulent inflow section";
    case Core::Conditions::BlendMaterial:
      return "blend materials";
    case Core::Conditions::FilamentBeamLineCondition:
      return "line condition for polymer networks";
    case Core::Conditions::PointCoupling:
      return "Multi point constraint coupling";
    case Core::Conditions::PenaltyPointCouplingCondition:
      return "condition for beam-to-beam point coupling based on a penalty potential";
    case Core::Conditions::PointLinearCoupledEquation:
      return "Point linear coupling equation";
    case Core::Conditions::RigidspherePotential_PointCharge:
      return "Rigid sphere potential point charge";
    case Core::Conditions::BeamPotential_LineChargeDensity:
      return "Beam potential line charge density";
    case Core::Conditions::BeamToBeamContact:
      return "condition for beam-to-beam contact";
    case Core::Conditions::BeamToSolidVolumeMeshtyingLine:
      return "line condition for beam-to-volume interaction";
    case Core::Conditions::BeamToSolidVolumeMeshtyingVolume:
      return "volume condition for beam-to-volume interaction";
    case Core::Conditions::BeamToSolidSurfaceMeshtyingLine:
      return "line condition for beam-to-surface interaction";
    case Core::Conditions::BeamToSolidSurfaceMeshtyingSurface:
      return "surface condition for beam-to-surface interaction";
    case Core::Conditions::BeamToSolidSurfaceContactLine:
      return "line condition for beam-to-surface contact";
    case Core::Conditions::BeamToSolidSurfaceContactSurface:
      return "surface condition for beam-to-surface contact";
    case Core::Conditions::FlowRateThroughLine_2D:
      return "Monitor flow rate through an line interface";
    case Core::Conditions::FlowRateThroughSurface_3D:
      return "Monitor flow rate through a surface interface";
    case Core::Conditions::FluidNeumannInflow:
      return "Fluid Neumann inflow";
    case Core::Conditions::TransportNeumannInflow:
      return "Scalar transport Neumann inflow";
    case Core::Conditions::ElchBoundaryKinetics:
      return "Electrode kinetics as boundary condition";
    case Core::Conditions::ElchDomainKinetics:
      return "Electrode domain kinetcs";
    case Core::Conditions::ElectrodeSOC:
      return "Electrode state of charge (soc)";
    case Core::Conditions::CCCVCycling:
      return "Constant-current constant-voltage (CCCV) cell cycling";
    case Core::Conditions::CCCVHalfCycle:
      return "Constant-current constant-voltage (CCCV) half-cycle";
    case Core::Conditions::CellVoltage:
      return "Cell voltage condition";
    case Core::Conditions::ArtJunctionCond:
      return "Artery junction boundary condition";
    case Core::Conditions::ArtPrescribedCond:
      return "Artery prescribed boundary condition";
    case Core::Conditions::ArtPorofluidCouplingCondNodebased:
      return "Artery-Porofluid nodebased coupling condition";
    case Core::Conditions::ArtPorofluidCouplingCondNodeToPoint:
      return "Artery-Porofluid node-to-point coupling condition non-conforming";
    case Core::Conditions::PoroMultiphaseScatraOxyPartPressCalcCond:
      return "PoroMultiphaseScatra Oxygen Partial Pressure Calculation condition";
    case Core::Conditions::ArtScatraCouplingCondNodebased:
      return "Artery-Scatra nodebased coupling condition";
    case Core::Conditions::ArtScatraCouplingCondNodeToPoint:
      return "Artery-Scatra node-to-point coupling condition non-conforming";
    case Core::Conditions::ArtRfCond:
      return "Artery reflective boundary condition";
    case Core::Conditions::StructAleCoupling:
      return "Structure - ALE coupling condition";
    case Core::Conditions::StructFluidVolCoupling:
      return "Structure - Fluid volume coupling condition";
    case Core::Conditions::BioGrCoupling:
      return "Biofilm growth coupling condition";
    case Core::Conditions::ArtInOutletCond:
      return "Artery terminal in_outlet condition";
    case Core::Conditions::ArtRedTo3DCouplingCond:
      return "Artery reduced D 3D coupling condition";
    case Core::Conditions::Art3DToRedCouplingCond:
      return "Artery 3D reduced D coupling condition";
    case Core::Conditions::RedAirwayPrescribedCond:
      return "Reduced d airway prescribed boundary condition";
    case Core::Conditions::RedAirwayPrescribedExternalPressure:
      return "Reduced d airway prescribed external pressure boundary condition";
    case Core::Conditions::RedAirwayPrescribedSwitchCond:
      return "Reduced dim airway switch flow pressure condition";
    case Core::Conditions::VolumetricSurfaceFlowCond:
      return "Volumetric Surface Flow Profile";
    case Core::Conditions::VolumetricFlowBorderNodes:
      return "Border Nodes of the volumetric flow Surface";
    case Core::Conditions::VolSTCLayer:
      return "Number of current STC layer";
    case Core::Conditions::ThermoConvections:
      return "ThermoConvections boundary condition";
    case Core::Conditions::TransportThermoConvections:
      return "Transport ThermoConvections boundary condition";
    case Core::Conditions::FSICouplingCenterDisp:
      return "Sliding ALE Center Disp condition";
    case Core::Conditions::RobinSpringDashpot:
      return "Robin Spring Dashpot Condition";
    case Core::Conditions::RobinSpringDashpotCoupling:
      return "Spring Dashpot Coupling Condition";
    case Core::Conditions::TotalTractionCorrectionCond:
      return "Total traction correct condition";
    case Core::Conditions::no_penetration:
      return "No Penetration Condition";
    case Core::Conditions::TotalTractionCorrectionBorderNodes:
      return "Total traction correction border nodes condition";
    case Core::Conditions::PoroCoupling:
      return "porous media coupling condition";
    case Core::Conditions::PoroPartInt:
      return "porous media partial integration condition";
    case Core::Conditions::PoroPresInt:
      return "porous media pressure integration condition";
    case Core::Conditions::NeumannIntegration:
      return "fpsi neumann integration condition";
    case Core::Conditions::DomainIntegral:
      return "condition for domain integral computation";
    case Core::Conditions::BoundaryIntegral:
      return "condition for boundary integral computation";
    case Core::Conditions::ScaTraFluxCalc:
      return "Scalar transport flux calculation boundary condition";
    case Core::Conditions::TotalAndMeanScalar:
      return "Calculation of total and mean scatra values";
    case Core::Conditions::ScaTraCoupling:
      return "scatra coupling condition";
    case Core::Conditions::ParticleWall:
      return "particle wall condition";
    case Core::Conditions::LineModeKrylovProjection:
      return "Line mode for Krylov space projection";
    case Core::Conditions::SurfaceModeKrylovProjection:
      return "Surface mode for Krylov space projection";
    case Core::Conditions::VolumeModeKrylovProjection:
      return "Volume mode for Krylov space projection";
    case Core::Conditions::SurfaceCurrent:
      return "Surface Current Evaluation";
    case Core::Conditions::UncertainSurface:
      return "Uncertain Surface";
    case Core::Conditions::AAASurface:
      return "AAA Surface";
    case Core::Conditions::LsContact:
      return "level-set condition for contact points";
    case Core::Conditions::RedAirwayVolDependentPleuralPressureCond:
      return "Reduced D airways evaluate lungs volume-dependent peural pressure condition";
    case Core::Conditions::RedAirwayEvalLungVolCond:
      return "Reduced D airways evaluate lung volume condition";
    case Core::Conditions::TransportRobin:
      return "Scalar transport Robin boundary condition";
    case Core::Conditions::ThermoRobin:
      return "thermal Robin condition";
    case Core::Conditions::ScatraMultiScaleCoupling:
      return "Scalar transport multi-scale coupling condition";
    case Core::Conditions::ScatraHeteroReactionCondMaster:
      return "Scalar transport reaction coupling condition (Master:";
    case Core::Conditions::ScatraHeteroReactionCondSlave:
      return "Scalar transport reaction coupling condition (Slave:";
    case Core::Conditions::ScatraPartitioning:
      return "Scalar transport partitioning condition for block preconditioning";
    case Core::Conditions::ScatraRelError:
      return "Relative error to analytical solution";
    case Core::Conditions::SSICoupling:
      return "Scalar-Structure coupling condition";
    case Core::Conditions::SSICouplingSolidToScatra:
      return "Scalar-Structure coupling condition from Solid to Scatra";
    case Core::Conditions::SSICouplingScatraToSolid:
      return "Scalar-Structure coupling condition from Scatra to Solid";
    case Core::Conditions::SSIInterfaceContact:
      return "Scalar-structure interface contact condition";
    case Core::Conditions::ssi_interface_meshtying:
      return "Scalar-Structure interaction interface meshtying condition";
    case Core::Conditions::SSIMeshtying3DomainIntersection:
      return "Scalar-Structure interaction interface meshtying condition including 3 domains";
    case Core::Conditions::SSISurfaceManifold:
      return "ScaTra Manifold field in SSI";
    case Core::Conditions::SSISurfaceManifoldKinetics:
      return "Kinetics model for coupling scatra <-> scatra on manifold";
    case Core::Conditions::SSTIInterfaceMeshtying:
      return "Scalar-Structure-Thermo interaction interface meshtying condition";
    case Core::Conditions::SSTIMeshtying3DomainIntersection:
      return "Scalar-Structure-Thermo interaction interface meshtying condition including 3 "
             "domains";
    case Core::Conditions::CellFocalAdhesion:
      return "Scalar transport boundary condition depending on structural surface stress";
    case Core::Conditions::S2IKinetics:
      return "Scatra-scatra interface kinetics";
    case Core::Conditions::S2IKineticsGrowth:
      return "Scatra-scatra kinetics growth";
    case Core::Conditions::S2INoEvaluation:
      return "Scatra-scatra no evaluation";
    case Core::Conditions::S2IMeshtying:
      return "Scatra-scatra interface mesh tying";
    case Core::Conditions::S2ISCLCoupling:
      return "Scatra-scatra with space-charge layer coupling";
    case Core::Conditions::ElementTag:
      return "Tagged elements";
    case Core::Conditions::NodeTag:
      return "Tagged nodes";
    case Core::Conditions::Embedded_Mesh_Solid_Surf_Coupling:
      return "Embedded mesh solid surface coupling";
    case Core::Conditions::Embedded_Mesh_Solid_Volume_Background:
      return "Embedded mesh solid volume background";
    case Core::Conditions::ReinitializationTaylorGalerkin:
      return "Reinitialization Taylor Galerkin condition";
    case Core::Conditions::TaylorGalerkinNeumannInflow:
      return "Taylor Galerkin neumann inflow";
    case Core::Conditions::TaylorGalerkinOutflow:
      return "Taylor Galerkin outflow";
    default:
      FOUR_C_THROW("Unknown condition type for enum constant {}.", condition_type);
  }
}

FOUR_C_NAMESPACE_CLOSE

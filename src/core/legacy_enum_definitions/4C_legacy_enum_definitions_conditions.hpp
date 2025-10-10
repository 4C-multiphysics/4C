// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LEGACY_ENUM_DEFINITIONS_CONDITIONS_HPP
#define FOUR_C_LEGACY_ENUM_DEFINITIONS_CONDITIONS_HPP

#include "4C_config.hpp"

#include <string_view>

FOUR_C_NAMESPACE_OPEN

namespace Core::Conditions
{
  enum ConditionType
  {
    none,
    PointDirichlet,    ///< Dirichlet condition defined on a (set of) point(s)
    LineDirichlet,     ///< Dirichlet condition defined on a (set of) line(s)
    SurfaceDirichlet,  ///< Dirichlet condition defined on a (set of) surface(s)
    VolumeDirichlet,   ///< Dirichlet condition defined on a (set of) volume(s)
    PointNeumann,      ///< Neumann condition defined on a (set of) point(s)
    PointNeumannEB,
    LineNeumann,       ///< Neumann condition defined on a (set of) line(s)
    SurfaceNeumann,    ///< Neumann condition defined on a (set of) surface(s)
    VolumeNeumann,     ///< Neumann condition defined on a (set of) volume(s)
    PointInitfield,    ///< Initial field defined on a (set of) point(s)
    LineInitfield,     ///< Initial field defined on a (set of) line(s)
    SurfaceInitfield,  ///< Initial field defined on a (set of) surface(s)
    VolumeInitfield,   ///< Initial field defined on a (set of) volume(s)
    PointCoupling,
    Mortar,  ///< Surface coupling condition for Mortar coupling (2D and 3D)
    MortarMulti,
    Contact,  ///< Contact between two volumetric bodies
    AleWear,
    LineMrtrSym,
    PointMrtrSym,
    EdgeMrtr,
    CornerMrtr,
    PointLocsys,
    LineLocsys,
    SurfaceLocsys,
    VolumeLocsys,
    LinePeriodic,
    PointLinearCoupledEquation,
    PointRvePeriodicReference,
    LineRvePeriodic,
    SurfacePeriodic,
    SurfaceRvePeriodic,
    TransferTurbulentInflow,
    TurbulentInflowSection,
    BlendMaterial,
    LineFlowDepPressure,
    SurfaceFlowDepPressure,
    LineSlipSupp,
    SurfaceSlipSupp,
    LineNavierSlip,
    SurfNavierSlip,
    LineWeakDirichlet,
    SurfaceWeakDirichlet,
    LineMixHybDirichlet,
    SurfaceMixHybDirichlet,
    SurfaceConservativeOutflowConsistency,
    FSICoupling,  ///< Surface coupling condition for fluid-structure interaction (2D and 3D)
    fpsi_coupling,
    FSICouplingCenterDisp,
    ALEUPDATECoupling,
    MicroBoundary,
    XFEM_Surf_Displacement,
    XFEM_Levelset_Weak_Dirichlet,
    XFEM_Levelset_Neumann,
    XFEM_Levelset_Navier_Slip,
    XFEM_Robin_Dirichlet_Volume,
    XFEM_Robin_Neumann_Volume,
    XFEM_Levelset_Twophase,
    XFEM_Surf_FSIPart,
    XFEM_Surf_FSIMono,
    XFEM_Surf_FPIMono,
    XFEM_Surf_FluidFluid,
    XFEM_Surf_Weak_Dirichlet,
    XFEM_Surf_Neumann,
    XFEM_Surf_Navier_Slip,
    XFEM_Robin_Dirichlet_Surf,
    XFEM_Robin_Neumann_Surf,
    FluidFluidCoupling,
    FluidMesh,
    ALEFluidCoupling,
    FluidStressCalc,
    LineLIFTDRAG,
    SurfLIFTDRAG,
    VolSTCLayer,
    VolumeConstraint_3D,
    VolumeConstraint_3D_pen,
    AreaConstraint_3D,
    AreaConstraint_2D,
    VolumeMonitor_3D,
    AreaMonitor_3D,
    AreaMonitor_2D,
    Cardiovascular0D4ElementWindkessel_Structure,
    Cardiovascular0DArterialProxDist_Structure,
    Cardiovascular0DSysPulCirculation_Structure,
    CardiovascularRespiratory0DSysPulPeriphCirculation_Structure,
    Cardiovascular0DStructureCoupling,
    ImpedanceCond,
    Impedance_Calb_Cond,
    MPC_NodeOnPlane_3D,
    MPC_NodeOnLine_3D,
    MPC_NormalComponent_3D,
    MPC_NormalComponent_3D_pen,
    MPC_NodeOnLine_2D,
    LJ_Potential_Volume,
    LJ_Potential_Surface,
    LJ_Potential_Line,
    VanDerWaals_Potential_Volume,
    VanDerWaals_Potential_Surface,
    VanDerWaals_Potential_Line,
    ElectroRepulsion_Potential_Surface,
    ElectroRepulsion_Potential_Line,
    RigidspherePotential_PointCharge,
    PenaltyPointCouplingCondition,
    FilamentBeamLineCondition,
    BeamPotential_LineChargeDensity,
    BeamToBeamContact,
    BeamToSolidVolumeMeshtyingLine,
    BeamToSolidVolumeMeshtyingVolume,
    BeamToSolidSurfaceMeshtyingLine,
    BeamToSolidSurfaceMeshtyingSurface,
    BeamToSolidSurfaceContactLine,
    BeamToSolidSurfaceContactSurface,
    BeamToSolidEdgeContactBeam,
    BeamToSolidEdgeContactEdge,
    ElchBoundaryKinetics,
    ElchDomainKinetics,
    TransportNeumannInflow,
    FluidNeumannInflow,
    TaylorGalerkinOutflow,
    TaylorGalerkinNeumannInflow,
    ReinitializationTaylorGalerkin,
    FlowRateThroughLine_2D,
    FlowRateThroughSurface_3D,
    LineModeKrylovProjection,
    SurfaceModeKrylovProjection,
    VolumeModeKrylovProjection,
    ArtJunctionCond,
    ArtPrescribedCond,
    ArtPorofluidCouplingCondNodebased,
    ArtScatraCouplingCondNodebased,
    ArtPorofluidCouplingCondNodeToPoint,
    ArtScatraCouplingCondNodeToPoint,
    ArtRfCond,
    StructAleCoupling,
    StructFluidVolCoupling,
    BioGrCoupling,
    ArtInOutletCond,
    ArtRedTo3DCouplingCond,
    Art3DToRedCouplingCond,
    RedAirwayPrescribedCond,
    RedAirwayPrescribedSwitchCond,
    RedAirwayPrescribedExternalPressure,
    VolumetricSurfaceFlowCond,
    VolumetricFlowBorderNodes,
    ThermoConvections,
    TransportThermoConvections,
    S2IKinetics,
    S2IMeshtying,
    S2INoEvaluation,
    S2IKineticsGrowth,
    S2ISCLCoupling,
    ElectrodeSOC,
    CCCVCycling,
    CCCVHalfCycle,
    CellVoltage,
    DomainIntegral,
    BoundaryIntegral,
    ScaTraFluxCalc,
    TotalAndMeanScalar,
    ScaTraCoupling,
    RobinSpringDashpot,
    RobinSpringDashpotCoupling,
    TotalTractionCorrectionCond,
    no_penetration,
    TotalTractionCorrectionBorderNodes,
    PoroCoupling,
    PoroPartInt,
    PoroPresInt,
    PoroMultiphaseScatraOxyPartPressCalcCond,
    SSICoupling,
    EHLCoupling,
    SSICouplingSolidToScatra,
    SSICouplingScatraToSolid,
    SSIInterfaceContact,
    ssi_interface_meshtying,
    SSIMeshtying3DomainIntersection,
    SSISurfaceManifold,
    SSISurfaceManifoldKinetics,
    SSTIInterfaceMeshtying,
    SSTIMeshtying3DomainIntersection,
    ParticleWall,
    NeumannIntegration,
    SurfaceCurrent,
    UncertainSurface,
    AAASurface,
    LsContact,
    RedAirwayVolDependentPleuralPressureCond,
    RedAirwayEvalLungVolCond,
    SPRboundary,
    TransportRobin,
    ThermoRobin,
    ScatraMultiScaleCoupling,
    ScatraRelError,
    CellFocalAdhesion,
    ScatraHeteroReactionCondMaster,
    ScatraHeteroReactionCondSlave,
    ScatraPartitioning,
    ElementTag,
    NodeTag,
    Embedded_Mesh_Solid_Surf_Coupling,
    Embedded_Mesh_Solid_Volume_Background
  };

  std::string_view to_string(ConditionType condition_type);

  /*!
  \brief Type of geometry this conditions lives on

  \warning The order is crucial for and must not be changed!!!
           (used in boundary_conditions_geometry())
  */
  enum GeometryType
  {
    geometry_type_point,
    geometry_type_line,
    geometry_type_surface,
    geometry_type_volume,
    geometry_type_no_geom
  };
}  // namespace Core::Conditions

FOUR_C_NAMESPACE_CLOSE

#endif

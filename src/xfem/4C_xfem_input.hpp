// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_XFEM_INPUT_HPP
#define FOUR_C_XFEM_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Conditions
{
  class ConditionDefinition;
}  // namespace Core::Conditions

namespace XFEM
{
  /// which method is used to enforce the boundary condition
  enum CouplingMethod
  {
    Hybrid_LM_Cauchy_stress,
    Hybrid_LM_viscous_stress,
    Nitsche
  };

  /// how to define the average, coupling with respect to which side?
  enum AveragingStrategy
  {
    Xfluid_Sided,    /// monolithic coupling between fluid and (fluid/structure), average w.r.t
                     /// xfluid side
    Embedded_Sided,  /// monolithic coupling between fluid and (fluid/structure), average w.r.t
                     /// embedded fluid/structure side
    Mean,  /// monolithic coupling between fluid and (fluid/structure), mean average between fluid
           /// and fluid/structure
    Harmonic,  /// monolithic coupling between fluid and fluid, harmonic mean average between
               /// fluid and fluid
    invalid
  };

  /// which type of interface condition is prescribed
  enum InterfaceLaw
  {
    noslip,             /// no slip in tangential direction
    noslip_splitpen,    /// no slip in tangential direction with split normal and tangential
                        /// penalty
    slip,               /// full slip in tangential direction
    navierslip,         /// general navier boundary condition
    navierslip_contact  /// general navier boundary condition to enable continuous transition to
                        /// contact
  };

  /// how is the adjoint consistency term scaled
  enum AdjointScaling
  {
    adj_sym,   /// symmetric adjoint consistency term
    adj_skew,  /// skew-symmetric adjoint consistency term
    adj_none   /// no adjoint consistency term
  };

  /// L2 stress projection on whole fluid element or on partial fluid element
  enum HybridLmL2Proj
  {
    Hybrid_LM_L2_Proj_full,
    Hybrid_LM_L2_Proj_part
  };

  /// estimate of the scaling of the trace inequality for viscous interface stabilization
  /// (Nitsche's method)
  enum ViscStabTraceEstimate
  {
    ViscStab_TraceEstimate_CT_div_by_hk,  /// estimate of the trace-inequality by trace-constant
                                          /// divided by a characteristic element length
    ViscStab_TraceEstimate_eigenvalue     /// estimate of the trace-inequality by solving an
                                          /// eigenvalue problem
  };

  /// how often should the local eigenvalue problem be updated
  enum TraceEstimateEigenvalueUpdate
  {
    Eigenvalue_update_every_iter,      /// every Newton iteration
    Eigenvalue_update_every_timestep,  /// every timestep
    Eigenvalue_update_once             /// once at the beginning of the simulation
  };

  /// definition of characteristic element length in cut elements
  enum ViscStabHk
  {
    ViscStab_hk_vol_equivalent,           /// volume equivalent element diameter
    ViscStab_hk_cut_vol_div_by_cut_surf,  /// physical partial/cut volume divided by physical
                                          /// partial/cut surface measure ( used to estimate the
                                          /// cut-dependent inverse estimate on cut elements, not
                                          /// useful for sliver and/or dotted cut situations)
    ViscStab_hk_ele_vol_div_by_cut_surf,  /// full element volume divided by physical partial/cut
                                          /// surface measure ( used to estimate the cut-dependent
                                          /// inverse estimate on cut elements, however, avoids
                                          /// problems with sliver cuts, not useful for dotted
                                          /// cuts)
    ViscStab_hk_ele_vol_div_by_ele_surf,  /// full element volume divided by surface measure (
                                          /// used for uncut situations, standard weak Dirichlet
                                          /// boundary/coupling conditions)
    ViscStab_hk_ele_vol_div_by_max_ele_surf  /// full element volume divided by maximal element
                                             /// surface measure ( used to estimate the trace
                                             /// inequality for stretched elements in combination
                                             /// with ghost-penalties)
  };

  /// type of scaling for convective/inflow stabilization term
  enum ConvStabScaling
  {
    ConvStabScaling_inflow,      /// interface-normal velocity component in case of inflow, 0 for
                                 /// outflow
    ConvStabScaling_abs_inflow,  /// absolute value of interface-normal velocity component in case
                                 /// of inflow, 0 for outflow
    ConvStabScaling_none
  };

  /// type of scaling for convective/inflow stabilization term (xfluid-fluid applications)
  enum XffConvStabScaling
  {
    XFF_ConvStabScaling_upwinding,  ///
    XFF_ConvStabScaling_only_averaged,
    XFF_ConvStabScaling_none
  };

  /// depending on the flow regime, one can either choose the maximum from the viscous and
  /// convective contributions to the penalty factor or just sum all of them up
  enum MassConservationCombination
  {
    MassConservationCombination_max,  /// take the maximum from the viscous/convective/transient
                                      /// contribution of the penalty term
    MassConservationCombination_sum   /// take the sum of all contributions to the penalty term,
                                      /// independent from the flow regime
  };

  /// apply additional scaling of penalty term to enforce mass conservation for
  /// convection-dominated flow
  enum MassConservationScaling
  {
    MassConservationScaling_full,      /// apply additional penalty scaling
    MassConservationScaling_only_visc  /// use only the viscous scaling
  };

  /// Add interface terms from previous time step (new OST)
  enum InterfaceTermsPreviousState
  {
    PreviousState_only_consistency,  /// evaluate only consistency terms
    PreviousState_full               /// evaluate consistency, adjoint consistency and penalty terms
  };

  /// xfluidfluid-fsi-monolithic approach
  enum MonolithicXffsiApproach
  {
    XFFSI_Full_Newton,
    XFFSI_FixedALE_Interpolation,
    XFFSI_FixedALE_Partitioned
  };

  /// xfluidfluid time integration approach
  enum XFluidFluidTimeInt
  {
    Xff_TimeInt_FullProj,
    Xff_TimeInt_ProjIfMoved,
    Xff_TimeInt_KeepGhostValues,
    Xff_TimeInt_IncompProj
  };

  /// xfluid time integration technique
  enum XFluidTimeIntScheme
  {
    //! std-DOFS:   only copying for std-dofs allowed (no semi-lagrangean algo (SL))
    //! ghost-DOFS: copy dofs or ghost-penalty reconstruction
    Xf_TimeIntScheme_STD_by_Copy_AND_GHOST_by_Copy_or_GP,
    //! std-DOFS:   copying for std-dofs preferred and semi-lagrangean algo (SL) used for large
    //! displacements ghost-DOFS: copy dofs or ghost-penalty reconstruction
    Xf_TimeIntScheme_STD_by_Copy_or_SL_AND_GHOST_by_Copy_or_GP,
    //! std-DOFS:   semi-lagrangean algo (SL) used for large displacements in the whole cut-zone
    //! ghost-DOFS: only ghost-penalty reconstruction
    Xf_TimeIntScheme_STD_by_SL_cut_zone_AND_GHOST_by_GP,
    //! std-DOFS: copy or project from embedded fluid mesh
    //! ghost-DOFS: copy dofs or project from embedded fluid mesh or ghost-penalty reconstruction
    Xf_TimeIntScheme_STD_by_Copy_or_Proj_AND_GHOST_by_Proj_or_Copy_or_GP
  };

  /// xfluid time integration approach for single dofs
  enum XFluidTimeIntMethod
  {
    Xf_TimeInt_STD_by_SL,                 //! Semi-lagrangean algorithm for standard dof
    Xf_TimeInt_STD_by_COPY_from_STD,      //! copy value for std-dof at t^(n+1) from std-dof at t^n
    Xf_TimeInt_STD_by_COPY_from_GHOST,    //! copy value for std-dof at t^(n+1) from ghost-dof at
                                          //! t^n
    Xf_TimeInt_GHOST_by_GP,               //! Ghost-penalty reconstruction for ghost dof
    Xf_TimeInt_GHOST_by_COPY_from_STD,    //! copy value for ghost-dof at t^(n+1) from std-dof at
                                          //! t^n
    Xf_TimeInt_GHOST_by_COPY_from_GHOST,  //! copy value for ghost-dof at t^(n+1) from ghost-dof
                                          //! at t^n
    Xf_TimeInt_by_PROJ_from_DIS,  //! project value for std-/ghost-dof at t^(n+1) from embedded
                                  //! fluid mesh at t^n
    Xf_TimeInt_undefined
  };

  /// type of face
  enum FaceType
  {
    face_type_std,
    face_type_ghost_penalty,
    face_type_ghost,
    face_type_boundary_ghost_penalty,
    face_type_porof
  };


  enum EleCouplingCondType
  {
    CouplingCond_NONE,
    CouplingCond_SURF_FSI_MONO,
    CouplingCond_SURF_FPI_MONO,
    CouplingCond_SURF_FLUIDFLUID,
    CouplingCond_SURF_FSI_PART,
    CouplingCond_SURF_WEAK_DIRICHLET,
    CouplingCond_SURF_NEUMANN,
    CouplingCond_SURF_NAVIER_SLIP,
    CouplingCond_SURF_NAVIER_SLIP_TWOPHASE,
    CouplingCond_EMBEDDEDMESH_BACKGROUND_SOLID_VOL,
    CouplingCond_EMBEDDEDMESH_SOLID_SURF,
    CouplingCond_LEVELSET_WEAK_DIRICHLET,
    CouplingCond_LEVELSET_NEUMANN,
    CouplingCond_LEVELSET_NAVIER_SLIP,
    CouplingCond_LEVELSET_TWOPHASE,
    CouplingCond_LEVELSET_COMBUSTION
  };

  // Type of surface projection for splitting of normal and tangential directions
  enum ProjToSurface
  {
    Proj_normal,
    Proj_smoothed,
    Proj_normal_smoothed_comb,
    Proj_normal_phi
  };

  enum CoupTerm
  {
    // Row Scalings
    F_Con_Row,
    F_Con_n_Row,
    F_Con_t_Row,
    X_Con_Row,
    X_Con_n_Row,
    X_Con_t_Row,
    F_Adj_Row,
    F_Adj_n_Row,
    F_Adj_t_Row,
    XF_Adj_Row,  // specific row scalings for fluid stress slave
    XF_Adj_n_Row,
    XF_Adj_t_Row,
    XS_Adj_Row,  // specific col scalings for structural stress slave
    XS_Adj_n_Row,
    XS_Adj_t_Row,
    F_Pen_Row,
    F_Pen_Row_linF1,
    F_Pen_Row_linF2,
    F_Pen_Row_linF3,
    F_Pen_n_Row,
    F_Pen_t_Row,
    X_Pen_Row,
    X_Pen_n_Row,
    X_Pen_t_Row,
    // Col Scalings
    F_Con_Col,
    F_Con_n_Col,
    F_Con_t_Col,
    XF_Con_Col,  // specific col scalings for fluid stress slave
    XF_Con_n_Col,
    XF_Con_t_Col,
    XS_Con_Col,  // specific col scalings for structural stress slave
    XS_Con_n_Col,
    XS_Con_t_Col,
    F_Adj_Col,
    F_Adj_n_Col,
    F_Adj_t_Col,
    X_Adj_Col,
    X_Adj_n_Col,
    X_Adj_t_Col,
    F_Pen_Col,
    F_Pen_n_Col,
    F_Pen_t_Col,
    X_Pen_Col,
    X_Pen_n_Col,
    X_Pen_t_Col,
    FStr_Adj_Col,
    FStr_Adj_n_Col,
    FStr_Adj_t_Col,
    XStr_Adj_Col,
    XStr_Adj_n_Col,
    XStr_Adj_t_Col,
    // Starting from here are some special Terms (before adding new one think if it's possible to
    // reuse the old terms without duplicating code.)
    F_LB_Rhs,  // Laplace Belatrami Term for traction jump master fluid sided (used only for two
               // phase flow atm)
    X_LB_Rhs,  // ==||== slave fluid sided (used only for two phase flow atm)
    F_TJ_Rhs,  // Traction jump term with given traction jump vector master fluid sided (used only
               // for two phase flow atm)
    X_TJ_Rhs   // Traction jump term with given traction jump vector slave fluid sided (used only
               // for two phase flow atm)
  };

  /// xfem parameters
  std::vector<Core::IO::InputSpec> valid_parameters();

  /// set specific xfem conditions
  void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

}  // namespace XFEM
/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif

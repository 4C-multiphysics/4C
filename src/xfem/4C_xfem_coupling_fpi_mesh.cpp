// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_coupling_fpi_mesh.hpp"

#include "4C_fem_dofset_transparent_independent.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_parameter_xfem.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_xfem_discretization_utils.hpp"
#include "4C_xfem_interface_utils.hpp"
#include "4C_xfem_utils.hpp"
#include "4C_xfem_xfluid_contact_communicator.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

//! constructor
XFEM::MeshCouplingFPI::MeshCouplingFPI(
    std::shared_ptr<Core::FE::Discretization>& bg_dis,  ///< background discretization
    const std::string& cond_name,  ///< name of the condition, by which the derived cutter
                                   ///< discretization is identified
    std::shared_ptr<Core::FE::Discretization>&
        cond_dis,           ///< discretization from which cutter discretization can be derived
    const int coupling_id,  ///< id of composite of coupling conditions
    const double time,      ///< time
    const int step,         ///< time step
    MeshCouplingFPI::CoupledField field  ///< which field is coupled to the fluid
    )
    : MeshVolCoupling(bg_dis, cond_name, cond_dis, coupling_id, time, step,
          (field == MeshCouplingFPI::ps_ps
                  ? "_ps_ps"
                  : (field == MeshCouplingFPI::ps_pf
                            ? "_ps_pf"
                            : (field == MeshCouplingFPI::pf_ps ? "_pf_ps" : "_pf_pf")))),
      coupled_field_(field),
      contact_(false),
      fpsi_contact_hfraction_(0.0),
      fpsi_contact_fullpcfraction_(0.0),
      xf_c_comm_(nullptr)
{
  // TODO: init here, but set in set_condition_specific_parameters
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFPI::set_coupling_name()
{
  // Set couling name dependent on the field type ... to be able to differentiate between the two
  // fpi coupling objects!
  std::stringstream str;
  str << cond_name_
      << (coupled_field_ == MeshCouplingFPI::ps_ps
                 ? "_ps_ps"
                 : (coupled_field_ == MeshCouplingFPI::ps_pf
                           ? "_ps_pf"
                           : (coupled_field_ == MeshCouplingFPI::pf_ps ? "_pf_ps" : "_pf_pf")));

  // replace the set condname by its specification given by the coupling field
  coupl_name_ = str.str();
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFPI::init_state_vectors()
{
  XFEM::MeshCoupling::init_state_vectors();

  const Core::LinAlg::Map* cutterdofrowmap = cutter_dis_->dof_row_map();
  const Core::LinAlg::Map* cutterdofcolmap = cutter_dis_->dof_col_map();

  itrueresidual_ = Core::LinAlg::create_vector(*cutterdofrowmap, true);
  iforcecol_ = Core::LinAlg::create_vector(*cutterdofcolmap, true);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFPI::do_condition_specific_setup()
{
  // We do ghosting just for the first fpi coupling object, which is PS_PS, then this is done an we
  // just reconnect to the parent pointers for every cutter_dis_ Actually this reconnecting step
  // should be done in an setup routine, to guarantee, that it is done late enough (no ghosting
  // afterwards...)
  if (coupled_field_ == MeshCouplingFPI::ps_ps)
    PoroElast::Utils::create_volume_ghosting(*cutter_dis_);
  else
    PoroElast::Utils::reconnect_parent_pointers(*cutter_dis_, *cond_dis_);

  // call base class
  XFEM::MeshCoupling::do_condition_specific_setup();
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFPI::reconnect_parent_pointers()
{
  PoroElast::Utils::reconnect_parent_pointers(*cutter_dis_, *cond_dis_);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFPI::register_side_proc(int sid)
{
  if (contact_ && coupled_field_ == MeshCouplingFPI::ps_ps)
    get_contact_comm()->register_side_proc(sid);
  return;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFPI::complete_state_vectors()
{
  //-------------------------------------------------------------------------------
  // finalize itrueresidual vector

  // need to export the interface forces
  Core::LinAlg::Vector<double> iforce_tmp(itrueresidual_->get_map(), true);
  Core::LinAlg::Export exporter_iforce(iforcecol_->get_map(), iforce_tmp.get_map());
  iforce_tmp.export_to(*iforcecol_, exporter_iforce, Add);

  // scale the interface trueresidual with -1.0 to get the forces acting on structural side (no
  // residual-scaling!)
  itrueresidual_->update(-1.0, iforce_tmp, 0.0);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFPI::setup_configuration_map()
{
  switch (coupled_field_)
  {
    case MeshCouplingFPI::ps_ps:
    {
      // Configuration of Consistency Terms

      if (!sub_tang_)
      {
        configuration_map_[Inpar::XFEM::F_Con_Row] = std::pair<bool, double>(true, 1.0);
        configuration_map_[Inpar::XFEM::F_Con_Col] = std::pair<bool, double>(true, 1.0);
        configuration_map_[Inpar::XFEM::X_Con_Row] = std::pair<bool, double>(true, 1.0);
      }
      else
      {
        configuration_map_[Inpar::XFEM::F_Con_n_Row] = std::pair<bool, double>(true, 1.0);
        configuration_map_[Inpar::XFEM::F_Con_n_Col] = std::pair<bool, double>(true, 1.0);
        configuration_map_[Inpar::XFEM::X_Con_n_Row] = std::pair<bool, double>(true, 1.0);
      }
      // configuration_map_[Inpar::XFEM::X_Con_Row] = std::pair<bool,double>(true,1.0-porosity);
      // Configuration of Adjount Consistency Terms
      configuration_map_[Inpar::XFEM::F_Adj_n_Row] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::F_Adj_n_Col] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::X_Adj_n_Col] = std::pair<bool, double>(true, 1.0);

      if (!sub_tang_)
      {
        configuration_map_[Inpar::XFEM::F_Adj_t_Row] = std::pair<bool, double>(true, 1.0);
        configuration_map_[Inpar::XFEM::F_Adj_t_Col] = std::pair<bool, double>(true, 1.0);
        configuration_map_[Inpar::XFEM::X_Adj_t_Col] = std::pair<bool, double>(true, 1.0);
        configuration_map_[Inpar::XFEM::FStr_Adj_t_Col] =
            std::pair<bool, double>(true, 1.0);  // Here we need alpha BJ finally!!! Todo
      }

      // Configuration of Penalty Terms
      configuration_map_[Inpar::XFEM::F_Pen_n_Row] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::F_Pen_n_Col] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::X_Pen_n_Row] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::X_Pen_n_Col] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::F_Pen_t_Row] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::F_Pen_t_Col] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::X_Pen_t_Row] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::X_Pen_t_Col] = std::pair<bool, double>(true, 1.0);

      if (!sub_tang_)
      {
        configuration_map_[Inpar::XFEM::F_Con_t_Row] =
            std::pair<bool, double>(true, 1.0);  //+sign for penalty!
        configuration_map_[Inpar::XFEM::F_Con_t_Col] = std::pair<bool, double>(true, 1.0);
        configuration_map_[Inpar::XFEM::X_Con_t_Row] =
            std::pair<bool, double>(true, 1.0);  //+sign for penalty!
      }

#ifdef INFLOW_STAB
      configuration_map_[Inpar::XFEM::F_Pen_Col] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::F_Pen_Row] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::F_Pen_Row_linF1] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::F_Pen_Row_linF2] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::F_Pen_Row_linF3] = std::pair<bool, double>(true, 1.0);
#endif
      break;
    }
    case MeshCouplingFPI::ps_pf:
    {
      // Configuration of Adjount Consistency Terms
      configuration_map_[Inpar::XFEM::F_Adj_n_Row] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::X_Adj_n_Col] = std::pair<bool, double>(true, 1.0);

      if (!sub_tang_)
      {
        configuration_map_[Inpar::XFEM::F_Adj_t_Row] = std::pair<bool, double>(true, 1.0);
        if (full_bj_)
          configuration_map_[Inpar::XFEM::X_Adj_t_Col] = std::pair<bool, double>(true, 1.0);
      }

      //    //Configuration of Penalty Terms
      configuration_map_[Inpar::XFEM::F_Pen_n_Row] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::X_Pen_n_Row] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::X_Pen_n_Col] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::F_Pen_t_Row] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::X_Pen_t_Row] = std::pair<bool, double>(true, 1.0);
      if (full_bj_)
        configuration_map_[Inpar::XFEM::X_Pen_t_Col] = std::pair<bool, double>(true, 1.0);
      break;
    }
    case MeshCouplingFPI::pf_ps:
    {
      // Configuration of Consistency Terms
      configuration_map_[Inpar::XFEM::X_Con_n_Row] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::F_Con_n_Col] = std::pair<bool, double>(true, 1.0);
      // Configuration of Penalty Terms
      configuration_map_[Inpar::XFEM::X_Pen_n_Row] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::F_Pen_n_Col] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::X_Pen_n_Col] = std::pair<bool, double>(true, 1.0);
      break;
    }
    case MeshCouplingFPI::pf_pf:
    {
      // Configuration of Penalty Terms
      configuration_map_[Inpar::XFEM::X_Pen_n_Row] = std::pair<bool, double>(true, 1.0);
      configuration_map_[Inpar::XFEM::X_Pen_n_Col] = std::pair<bool, double>(true, 1.0);
      break;
    }
  }
  return;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFPI::update_configuration_map_gp(double& kappa_m,  //< fluid sided weighting
    double& visc_m,          //< master sided dynamic viscosity
    double& visc_s,          //< slave sided dynamic viscosity
    double& density_m,       //< master sided density
    double& visc_stab_tang,  //< viscous tangential NIT Penalty scaling
    double& full_stab, const Core::LinAlg::Matrix<3, 1>& x, const Core::Conditions::Condition* cond,
    Core::Elements::Element* ele,   //< Element
    Core::Elements::Element* bele,  //< Boundary Element
    double* funct,                  //< local shape function for Gauss Point (from fluid element)
    double* derxy,  //< local derivatives of shape function for Gauss Point (from fluid element)
    Core::LinAlg::Matrix<3, 1>& rst_slave,  //< local coord of gp on slave boundary element
    Core::LinAlg::Matrix<3, 1>& normal,     //< normal at gp
    Core::LinAlg::Matrix<3, 1>& vel_m,      //< master velocity at gp
    double* fulltraction                    //< precomputed fsi traction (sigmaF n + gamma relvel)
)
{
  if (!contact_)
  {
    double J = 0;
    double porosity = calc_porosity(bele, rst_slave, J);

    double trperm = calctr_permeability(bele, porosity, J);


    static double sliplength = trperm / (bj_coeff_);

    static double dynvisc = (kappa_m * visc_m + (1.0 - kappa_m) * visc_s);

    static double stabnit = 0.0;
    static double stabadj = 0.0;

    XFEM::Utils::get_navier_slip_stabilization_parameters(
        visc_stab_tang, dynvisc, sliplength, stabnit, stabadj);

    // Overall there are 9 coupling blocks to evaluate for fpi:
    // 1 - ps_ps --> ff,fps,psf,psps
    // 2 - ps_pf --> fpf,pspf
    // 3 - pf_ps --> pff, pfps
    // 4 - pf_pf --> pfpf
    switch (coupled_field_)
    {
      case MeshCouplingFPI::ps_ps:
      {
        configuration_map_[Inpar::XFEM::X_Adj_n_Col].second = 1.0 - porosity;
        if (!sub_tang_)
        {
          configuration_map_[Inpar::XFEM::F_Adj_t_Row].second = stabadj;
          configuration_map_[Inpar::XFEM::X_Adj_t_Col].second = 1.0 - full_bj_ * porosity;
          configuration_map_[Inpar::XFEM::FStr_Adj_t_Col].second = sliplength;
        }
        configuration_map_[Inpar::XFEM::F_Pen_n_Row].second = full_stab;
        configuration_map_[Inpar::XFEM::X_Pen_n_Row].second = full_stab;
        configuration_map_[Inpar::XFEM::X_Pen_n_Col].second = 1.0 - porosity;
        if (!sub_tang_)
        {
          configuration_map_[Inpar::XFEM::F_Pen_t_Row].second = stabnit;
          configuration_map_[Inpar::XFEM::X_Pen_t_Row].second = stabnit;
          configuration_map_[Inpar::XFEM::F_Con_t_Row].second = -stabnit;  //+sign for penalty!
          configuration_map_[Inpar::XFEM::F_Con_t_Col].second = sliplength / dynvisc;
          configuration_map_[Inpar::XFEM::X_Con_t_Row].second = -stabnit;  //+sign for penalty!
        }
        else
        {
          configuration_map_[Inpar::XFEM::F_Pen_t_Row].second = 1. / sliplength;
          configuration_map_[Inpar::XFEM::X_Pen_t_Row].second = 1. / sliplength;

          // does nothing but should just be done in case we don't use the adjoint
          configuration_map_[Inpar::XFEM::F_Adj_t_Col].second = 1.0;
          configuration_map_[Inpar::XFEM::X_Adj_t_Col].second = 1.0 - full_bj_ * porosity;
        }

        configuration_map_[Inpar::XFEM::X_Pen_t_Col].second = 1.0 - full_bj_ * porosity;
        break;
      }
      case MeshCouplingFPI::ps_pf:
      {
        configuration_map_[Inpar::XFEM::X_Adj_n_Col].second = porosity;
        if (!sub_tang_)
        {
          configuration_map_[Inpar::XFEM::F_Adj_t_Row].second = stabadj;
          configuration_map_[Inpar::XFEM::X_Adj_t_Col].second = full_bj_ * porosity;
        }
        configuration_map_[Inpar::XFEM::F_Pen_n_Row].second = full_stab;
        configuration_map_[Inpar::XFEM::X_Pen_n_Row].second = full_stab;
        configuration_map_[Inpar::XFEM::X_Pen_n_Col].second = porosity;
        if (!sub_tang_)
        {
          configuration_map_[Inpar::XFEM::F_Pen_t_Row].second = stabnit;
          configuration_map_[Inpar::XFEM::X_Pen_t_Row].second = stabnit;
        }
        else
        {
          configuration_map_[Inpar::XFEM::F_Pen_t_Row].second = 1. / sliplength;
          configuration_map_[Inpar::XFEM::X_Pen_t_Row].second = 1. / sliplength;

          // does nothing but should just be done in case we don't use the adjoint
          configuration_map_[Inpar::XFEM::X_Adj_t_Col].second = full_bj_ * porosity;
        }
        configuration_map_[Inpar::XFEM::X_Pen_t_Col].second = full_bj_ * porosity;
        break;
      }
      case MeshCouplingFPI::pf_ps:
      {
        configuration_map_[Inpar::XFEM::X_Pen_n_Row].second = full_stab;
        configuration_map_[Inpar::XFEM::X_Pen_n_Col].second = 1.0 - porosity;

        // does nothing but should just be done in case we don't use the adjoint
        configuration_map_[Inpar::XFEM::F_Adj_n_Col].second =
            configuration_map_[Inpar::XFEM::F_Pen_n_Col].second;
        configuration_map_[Inpar::XFEM::X_Adj_n_Col].second =
            configuration_map_[Inpar::XFEM::X_Pen_n_Col].second;
        break;
      }
      case MeshCouplingFPI::pf_pf:
      {
        // Configuration of Penalty Terms
        configuration_map_[Inpar::XFEM::X_Pen_n_Row].second = full_stab;
        configuration_map_[Inpar::XFEM::X_Pen_n_Col].second = porosity;

        // does nothing but should just be done in case we don't use the adjoint
        configuration_map_[Inpar::XFEM::X_Adj_n_Col].second =
            configuration_map_[Inpar::XFEM::X_Pen_n_Col].second;
        break;
      }
    }
  }
  else
    update_configuration_map_gp_contact(kappa_m, visc_m, visc_s, density_m, visc_stab_tang,
        full_stab, x, cond, ele, bele, funct, derxy, rst_slave, normal, vel_m, fulltraction);

  return;
}

/*--------------------------------------------------------------------------*
 * update_configuration_map_gp_contact for XFPSCI
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFPI::update_configuration_map_gp_contact(
    double& kappa_m,         //< fluid sided weighting
    double& visc_m,          //< master sided dynamic viscosity
    double& visc_s,          //< slave sided dynamic viscosity
    double& density_m,       //< master sided density
    double& visc_stab_tang,  //< viscous tangential NIT Penalty scaling
    double& full_stab, const Core::LinAlg::Matrix<3, 1>& x, const Core::Conditions::Condition* cond,
    Core::Elements::Element* ele,   //< Element
    Core::Elements::Element* bele,  //< Boundary Element
    double* funct,                  //< local shape function for Gauss Point (from fluid element)
    double* derxy,  //< local derivatives of shape function for Gauss Point (from fluid element)
    Core::LinAlg::Matrix<3, 1>& rst_slave,  //< local coord of gp on slave boundary element
    Core::LinAlg::Matrix<3, 1>& normal,     //< normal at gp
    Core::LinAlg::Matrix<3, 1>& vel_m,      //< master velocity at gp
    double* fulltraction                    //< precomputed fsi traction (sigmaF n + gamma relvel)
)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  FOUR_C_ASSERT(xf_c_comm_ != nullptr,
      "update_configuration_map_gp_contact but no Xfluid Contact Communicator assigned!");
#endif

  // constant not really meant to be changed
  static const double MAX_sliplength = 1e40;  // large number for slip case
  static const int MAX_h =
      1;  // distance from contact zone at which classical BJ or BJS is prescribed
  static const int MIN_h = 0;  // distance from contact zone at which full-slip is prescribed
  static const double scaling = 1. / (MAX_h - MIN_h);
  static const double refsliplength = 0.1;  // numerical reference sliplength

  Core::LinAlg::Matrix<2, 1> xsi(rst_slave.data(), true);  // 3-->2
  double gap =
      MAX_h * h_scaling_;  // initialize with large value as this should be the default value ...
  bool pure_fsi = true;

  pure_fsi = xf_c_comm_->get_contact_state(bele->id(), "XFEMSurfFPIMono_ps_ps", xsi, *fulltraction,
      gap);  // get gap and if contact is integrated

  double J = 0;
  double porosity = calc_porosity(bele, rst_slave, J);

  double trperm = calctr_permeability(bele, porosity, J);

  double sliplength = trperm / (bj_coeff_);

  if ((gap - MIN_h * h_scaling_) * (MAX_sliplength + scaling) <
      h_scaling_)  // larger than maximal allows sliplength
  {
    sliplength += refsliplength * h_scaling_ * MAX_sliplength;
  }
  else if (gap > MAX_h * h_scaling_)  // BJ or BJS case
  {
    // sliplength += 0.;
  }
  else  // scaling scase
  {
    sliplength += refsliplength * h_scaling_ * (h_scaling_ / (gap - MIN_h * h_scaling_) - scaling);
  }

  if (sliplength < 0.0) FOUR_C_THROW("The slip should not be negative!");

  static double dynvisc = (kappa_m * visc_m + (1.0 - kappa_m) * visc_s);

  static double stabnit = 0.0;
  static double stabadj = 0.0;

  XFEM::Utils::get_navier_slip_stabilization_parameters(
      visc_stab_tang, dynvisc, sliplength, stabnit, stabadj);

  // Overall there are 9 coupling blocks to evaluate for fpi:
  // 1 - ps_ps --> ff,fps,psf,psps
  // 2 - ps_pf --> fpf,pspf
  // 3 - pf_ps --> pff, pfps
  // 4 - pf_pf --> pfpf
  switch (coupled_field_)
  {
    case MeshCouplingFPI::ps_ps:
    {
      configuration_map_[Inpar::XFEM::X_Adj_n_Col].second = 1.0 - porosity;
      configuration_map_[Inpar::XFEM::F_Adj_t_Row].second = stabadj;
      configuration_map_[Inpar::XFEM::X_Adj_t_Col].second = 1.0 - full_bj_ * porosity;
      configuration_map_[Inpar::XFEM::FStr_Adj_t_Col].second = sliplength;

      configuration_map_[Inpar::XFEM::F_Pen_n_Row].second = full_stab;
      configuration_map_[Inpar::XFEM::X_Pen_n_Col].second = 1.0 - porosity;
      configuration_map_[Inpar::XFEM::X_Pen_t_Row].second = stabnit;

      if (pure_fsi)
      {
        configuration_map_[Inpar::XFEM::X_Con_Row] = std::pair<bool, double>(true, 1.0);
        configuration_map_[Inpar::XFEM::X_Pen_n_Row] = std::pair<bool, double>(true, full_stab);
        configuration_map_[Inpar::XFEM::X_Con_t_Row].second = -stabnit;  //+sign for penalty!
      }
      else
      {
        configuration_map_[Inpar::XFEM::X_Con_Row] = std::pair<bool, double>(false, 0.0);
        configuration_map_[Inpar::XFEM::X_Pen_n_Row] = std::pair<bool, double>(false, 0.0);
        if (sliplength > 1e-40)
          configuration_map_[Inpar::XFEM::X_Con_t_Row].second =
              -stabnit + dynvisc / sliplength;  //+sign for penalty! + tangential consistency
        else                                    // avoid to evaluate this term ...
          configuration_map_[Inpar::XFEM::X_Con_t_Row].second = 0;
      }
      configuration_map_[Inpar::XFEM::F_Pen_t_Row].second = stabnit;
      configuration_map_[Inpar::XFEM::F_Con_t_Row].second = -stabnit;  //+sign for penalty!
      configuration_map_[Inpar::XFEM::F_Con_t_Col].second = sliplength / dynvisc;
      configuration_map_[Inpar::XFEM::X_Pen_t_Col].second = 1.0 - full_bj_ * porosity;
      break;
    }
    case MeshCouplingFPI::ps_pf:
    {
      configuration_map_[Inpar::XFEM::X_Adj_n_Col].second = porosity;
      configuration_map_[Inpar::XFEM::F_Adj_t_Row].second = stabadj;
      configuration_map_[Inpar::XFEM::X_Adj_t_Col].second = full_bj_ * porosity;
      configuration_map_[Inpar::XFEM::F_Pen_n_Row].second = full_stab;
      configuration_map_[Inpar::XFEM::X_Pen_n_Col].second = porosity;

      configuration_map_[Inpar::XFEM::F_Pen_t_Row].second = stabnit;
      configuration_map_[Inpar::XFEM::X_Pen_t_Row].second = stabnit;

      configuration_map_[Inpar::XFEM::X_Pen_t_Col].second = full_bj_ * porosity;
      if (pure_fsi)
      {
        configuration_map_[Inpar::XFEM::X_Pen_n_Row] = std::pair<bool, double>(true, full_stab);
      }
      else
      {
        configuration_map_[Inpar::XFEM::X_Pen_n_Row] = std::pair<bool, double>(false, 0.0);
      }
      break;
    }
    case MeshCouplingFPI::pf_ps:
    {
      double ffac = 1;
      if (gap < (1 + get_fpi_pcontact_fullfraction()) * get_fpi_pcontact_exchange_dist() &&
          get_fpi_pcontact_exchange_dist() > 1e-16)
        ffac = gap / (get_fpi_pcontact_exchange_dist()) - get_fpi_pcontact_fullfraction();
      if (ffac < 0) ffac = 0;

      configuration_map_[Inpar::XFEM::X_Con_n_Row].second = ffac;

      configuration_map_[Inpar::XFEM::X_Pen_n_Col].second = 1.0 - porosity;
      configuration_map_[Inpar::XFEM::X_Pen_n_Row].second = full_stab * ffac;

      // does nothing but should just be done in case we don't use the adjoint
      configuration_map_[Inpar::XFEM::F_Adj_n_Col].second =
          configuration_map_[Inpar::XFEM::F_Pen_n_Col].second;
      configuration_map_[Inpar::XFEM::X_Adj_n_Col].second =
          configuration_map_[Inpar::XFEM::X_Pen_n_Col].second;
      break;
    }
    case MeshCouplingFPI::pf_pf:
    {
      double ffac = 1;
      if (gap < (1 + get_fpi_pcontact_fullfraction()) * get_fpi_pcontact_exchange_dist() &&
          get_fpi_pcontact_exchange_dist() > 1e-16)
        ffac = gap / (get_fpi_pcontact_exchange_dist()) - get_fpi_pcontact_fullfraction();
      if (ffac < 0) ffac = 0;

      // Configuration of Penalty Terms
      configuration_map_[Inpar::XFEM::X_Pen_n_Col].second = porosity;
      configuration_map_[Inpar::XFEM::X_Pen_n_Row].second = full_stab * ffac;

      // does nothing but should just be done in case we don't use the adjoint
      configuration_map_[Inpar::XFEM::X_Adj_n_Col].second =
          configuration_map_[Inpar::XFEM::X_Pen_n_Col].second;
      break;
    }
  }
  return;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFPI::zero_state_vectors_fpi()
{
  itrueresidual_->put_scalar(0.0);
  iforcecol_->put_scalar(0.0);
}

// -------------------------------------------------------------------
// Read Restart data for cutter discretization
// -------------------------------------------------------------------
void XFEM::MeshCouplingFPI::read_restart(const int step)
{
  if (myrank_) Core::IO::cout << "read_restart for boundary discretization " << Core::IO::endl;

  //-------- boundary discretization
  Core::IO::DiscretizationReader boundaryreader(
      *cutter_dis_, Global::Problem::instance()->input_control_file(), step);

  const double time = boundaryreader.read_double("time");
  //  const int    step = boundaryreader.ReadInt("step");

  if (myrank_ == 0)
  {
    Core::IO::cout << "time: " << time << Core::IO::endl;
    Core::IO::cout << "step: " << step << Core::IO::endl;
  }

  boundaryreader.read_vector(iveln_, "iveln_res");
  boundaryreader.read_vector(idispn_, "idispn_res");

  // REMARK: ivelnp_ and idispnp_ are set again for the new time step in PrepareSolve()
  boundaryreader.read_vector(ivelnp_, "ivelnp_res");
  boundaryreader.read_vector(idispnp_, "idispnp_res");
  boundaryreader.read_vector(idispnpi_, "idispnpi_res");

  if (not(cutter_dis_->dof_row_map())->same_as(ivelnp_->get_map()))
    FOUR_C_THROW("Global dof numbering in maps does not match");
  if (not(cutter_dis_->dof_row_map())->same_as(iveln_->get_map()))
    FOUR_C_THROW("Global dof numbering in maps does not match");
  if (not(cutter_dis_->dof_row_map())->same_as(idispnp_->get_map()))
    FOUR_C_THROW("Global dof numbering in maps does not match");
  if (not(cutter_dis_->dof_row_map())->same_as(idispn_->get_map()))
    FOUR_C_THROW("Global dof numbering in maps does not match");
  if (not(cutter_dis_->dof_row_map())->same_as(idispnpi_->get_map()))
    FOUR_C_THROW("Global dof numbering in maps does not match");
}


void XFEM::MeshCouplingFPI::output(const int step, const double time, const bool write_restart_data)
{
  // output for interface
  cutter_output_->new_step(step, time);

  cutter_output_->write_vector("ivelnp", ivelnp_);
  cutter_output_->write_vector("idispnp", idispnp_);
  cutter_output_->write_vector("itrueresnp", itrueresidual_);

  cutter_output_->write_element_data(firstoutputofrun_);
  firstoutputofrun_ = false;

  // write restart
  if (write_restart_data)
  {
    cutter_output_->write_vector("iveln_res", iveln_);
    cutter_output_->write_vector("idispn_res", idispn_);
    cutter_output_->write_vector("ivelnp_res", ivelnp_);
    cutter_output_->write_vector("idispnp_res", idispnp_);
    cutter_output_->write_vector("idispnpi_res", idispnpi_);
  }
}

void XFEM::MeshCouplingFPI::set_condition_specific_parameters()
{
  std::vector<const Core::Conditions::Condition*> conditions_XFPI;
  cutter_dis_->get_condition(cond_name_, conditions_XFPI);

  // Create maps for easy extraction at gausspoint level
  auto i = conditions_XFPI.begin();
  for (const auto& cond : conditions_XFPI)
  {
    const bool full_BJ = (cond->parameters().get<std::string>("Variant") == "BJ");
    const bool Sub_tang = (cond->parameters().get<std::string>("Method") == "SUB");
    const bool contact = cond->parameters().get<bool>("Contact");
    if (i != conditions_XFPI.begin())
    {
      if (fabs(bj_coeff_ - cond->parameters().get<double>("BJ_COEFF")) > 1e-16)
        FOUR_C_THROW(
            "XFEM::MeshCouplingFPI::set_condition_specific_parameters: You defined two FPI "
            "conditions, with different BJ_coeff!");

      if (full_bj_ != full_BJ)
        FOUR_C_THROW(
            "XFEM::MeshCouplingFPI::set_condition_specific_parameters: You defined two FPI "
            "conditions, with different BJ Variant!");

      if (sub_tang_ != Sub_tang)
        FOUR_C_THROW(
            "XFEM::MeshCouplingFPI::set_condition_specific_parameters: You defined two FPI "
            "conditions, with different BJ Method!");

      if (contact_ != contact)
        FOUR_C_THROW(
            "XFEM::MeshCouplingFPI::set_condition_specific_parameters: You defined two FPI "
            "conditions, with different contact specification!");
    }

    bj_coeff_ = cond->parameters().get<double>("BJ_COEFF");
    full_bj_ = full_BJ;
    sub_tang_ = Sub_tang;
    contact_ = contact;
    i++;
  }

  if (contact_)  // compute h
  {
    double hmax = 0.0;
    for (int ele = 0; ele < bg_dis_->num_my_row_elements(); ++ele)
    {
      Core::Elements::Element* fluid_ele = bg_dis_->l_row_element(ele);
      if (fluid_ele->shape() == Core::FE::CellType::hex8)
      {
        Core::LinAlg::Matrix<3, 8> xyze(Core::LinAlg::Initialization::zero);
        Core::Geo::fill_initial_position_array(fluid_ele, xyze);
        double vol = XFEM::Utils::eval_element_volume<Core::FE::CellType::hex8>(xyze);
        hmax = std::max(hmax, XFEM::Utils::compute_vol_eq_diameter(vol));
      }
      else
        FOUR_C_THROW("Element type != hex8, add it here!");
    }
    h_scaling_ = Core::Communication::max_all(hmax, bg_dis_->get_comm());
    std::cout << "==| XFEM::MeshCouplingFPI: Computed h_scaling for fluidele is: " << h_scaling_
              << "(Proc: " << Core::Communication::my_mpi_rank(bg_dis_->get_comm())
              << ")! |==" << std::endl;

    fpsi_contact_hfraction_ = (Global::Problem::instance()->x_fluid_dynamic_params())
                                  .sublist("XFPSI MONOLITHIC")
                                  .get<double>("POROCONTACTFPSI_HFRACTION");
    fpsi_contact_fullpcfraction_ = (Global::Problem::instance()->x_fluid_dynamic_params())
                                       .sublist("XFPSI MONOLITHIC")
                                       .get<double>("POROCONTACTFPSI_FULLPCFRACTION");
  }

  if (!full_bj_)
  {
    if (coupled_field_ == XFEM::MeshCouplingFPI::ps_ps)
      std::cout << "==| XFEM::MeshCouplingFPI: Actual FPI Formulation is Beavers Joseph Saffmann"
                << std::flush;
  }
  else
  {
    if (coupled_field_ == XFEM::MeshCouplingFPI::ps_ps)
      std::cout << "==| XFEM::MeshCouplingFPI: Actual FPI Formulation is Beavers Joseph"
                << std::flush;
  }
  if (!sub_tang_)
  {
    if (coupled_field_ == XFEM::MeshCouplingFPI::ps_ps)
      std::cout << " -- by tangential Nitsche formulation! |==" << std::endl;
  }
  else
  {
    if (coupled_field_ == XFEM::MeshCouplingFPI::ps_ps)
      std::cout << " -- by tangential Substitution formulation! |==" << std::endl;
  }

  if (contact_)
  {
    std::cout << "==| XFEM::MeshCouplingFPI: Formulation with contact! |==" << std::endl;
  }

  if (contact_ && sub_tang_)
    FOUR_C_THROW(
        "XFEM::MeshCouplingFPI: Combination Contact with Substitution for BJ/BJS not tested!");
}

//----------------------------------------------------------------------
// lift_drag                                                  chfoe 11/07
//----------------------------------------------------------------------
// calculate lift&drag forces
//
// Lift and drag forces are based upon the right hand side true-residual entities
// of the corresponding nodes. The contribution of the end node of a line is entirely
// added to a present L&D force.
/*----------------------------------------------------------------------*/
void XFEM::MeshCouplingFPI::lift_drag(const int step, const double time) const
{
  // get forces on all procs
  // create interface DOF vectors using the fluid parallel distribution
  std::shared_ptr<const Core::LinAlg::Vector<double>> iforcecol =
      Core::Rebalance::get_col_version_of_row_vector(*cutter_dis_, itrueresidual_);

  if (myrank_ == 0)
  {
    // compute force components
    const int nsd = 3;
    const Core::LinAlg::Map* dofcolmap = cutter_dis_->dof_col_map();
    Core::LinAlg::Matrix<3, 1> c(Core::LinAlg::Initialization::zero);
    for (int inode = 0; inode < cutter_dis_->num_my_col_nodes(); ++inode)
    {
      const Core::Nodes::Node* node = cutter_dis_->l_col_node(inode);
      const std::vector<int> dof = cutter_dis_->dof(node);
      for (int isd = 0; isd < nsd; ++isd)
      {
        // [// minus to get correct sign of lift and drag (force acting on the body) ]
        c(isd) += iforcecol->local_values_as_span()[dofcolmap->lid(dof[isd])];
      }
    }

    // print to file
    std::ostringstream s;
    std::ostringstream header;

    header << std::left << std::setw(10) << "Time" << std::right << std::setw(16) << "F_x"
           << std::right << std::setw(16) << "F_y" << std::right << std::setw(16) << "F_z";
    s << std::left << std::setw(10) << std::scientific << time << std::right << std::setw(16)
      << std::scientific << c(0) << std::right << std::setw(16) << std::scientific << c(1)
      << std::right << std::setw(16) << std::scientific << c(2);

    std::ofstream f;
    const std::string fname = Global::Problem::instance()->output_control_file()->file_name() +
                              ".liftdrag." + cond_name_ + ".txt";
    if (step <= 1)
    {
      f.open(fname.c_str(), std::fstream::trunc);
      f << header.str() << std::endl;
    }
    else
    {
      f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
    }
    f << s.str() << "\n";
    f.close();

    std::cout << header.str() << std::endl << s.str() << std::endl;
  }
}

// ------------------------------------------------------------------------
// Calculate the normalized trace of permeability matrix
//        for J,porosity pair on this FaceElement               ager 12/17
// ------------------------------------------------------------------------
double XFEM::MeshCouplingFPI::calctr_permeability(
    Core::Elements::Element* ele, double& porosity, double& J)
{
  // Calculate normalized trace of permeability matrix
  Core::Elements::FaceElement* fele = dynamic_cast<Core::Elements::FaceElement*>(ele);
  if (!fele) FOUR_C_THROW("Cast to Faceele failed!");
  Core::Elements::Element* coupl_ele = fele->parent_element();
  if (coupl_ele == nullptr) FOUR_C_THROW("No coupl_ele!");
  std::shared_ptr<Mat::FluidPoro> poromat;
  // access second material in structure element
  if (coupl_ele->num_material() > 1)
    poromat = std::dynamic_pointer_cast<Mat::FluidPoro>(coupl_ele->material(1));
  else
    FOUR_C_THROW("no second material defined for element {}", ele->id());

  static Core::LinAlg::Matrix<3, 3> reactiontensor(Core::LinAlg::Initialization::zero);
  poromat->compute_reaction_tensor(reactiontensor, J, porosity);

  return sqrt((1. / reactiontensor(0, 0) + 1. / reactiontensor(1, 1) + 1. / reactiontensor(2, 2)) /
              (poromat->viscosity() * 3));
}

// --------------------------------------------------------------------
// Calculate the Porosity for this FaceElement Gausspoint   ager 12/16
// --------------------------------------------------------------------
double XFEM::MeshCouplingFPI::calc_porosity(
    Core::Elements::Element* ele, Core::LinAlg::Matrix<3, 1>& rst_slave, double& J)
{
  Core::Elements::FaceElement* fele = dynamic_cast<Core::Elements::FaceElement*>(ele);
  if (!fele) FOUR_C_THROW("Cast to Faceele failed!");

  Core::Elements::Element* coupl_ele = fele->parent_element();
  if (coupl_ele == nullptr) FOUR_C_THROW("No coupl_ele!");

  double pres = 0.0;
  J = compute_jacobianand_pressure(ele, rst_slave, pres);

  std::shared_ptr<Mat::StructPoro> poromat;
  // access second material in structure element
  if (coupl_ele->num_material() > 1)
  {
    poromat = std::dynamic_pointer_cast<Mat::StructPoro>(coupl_ele->material(0));
    if (poromat->material_type() != Core::Materials::m_structporo and
        poromat->material_type() != Core::Materials::m_structpororeaction and
        poromat->material_type() != Core::Materials::m_structpororeactionECM)
      FOUR_C_THROW("invalid structure material for poroelasticity");
  }
  else
    FOUR_C_THROW("no second material defined for element {}", ele->id());

  Teuchos::ParameterList params;  // empty parameter list;
  double porosity;
  poromat->compute_porosity(params, pres, J,
      1,  // not used
      porosity, false);
  return porosity;
}

// ---------------------------------------------------------------------------------------
// Compute Jacobian and extract PoroFluidPressure this FaceElement Gausspoint   ager 12/17
// ------------------------------------------------------------------------------------------
double XFEM::MeshCouplingFPI::compute_jacobianand_pressure(
    Core::Elements::Element* ele, Core::LinAlg::Matrix<3, 1>& rst_slave, double& pres)
{
  Core::Elements::FaceElement* fele = dynamic_cast<Core::Elements::FaceElement*>(ele);
  if (!fele) FOUR_C_THROW("Cast to Faceele failed!");

  Core::Elements::Element* coupl_ele = fele->parent_element();

  if (fele->shape() == Core::FE::CellType::quad4)
  {
    pres = 0.0;

    const unsigned int SLAVE_NUMDOF = 3;

    Core::FE::CollectedGaussPoints intpoints =
        Core::FE::CollectedGaussPoints(1);  // reserve just for 1 entry ...
    intpoints.append(rst_slave(0, 0), rst_slave(1, 0), 0.0, 1.0);

    // get coordinates of gauss point w.r.t. local parent coordinate system
    Core::LinAlg::SerialDenseMatrix pqxg(1, SLAVE_NUMDOF);
    Core::LinAlg::Matrix<SLAVE_NUMDOF, SLAVE_NUMDOF> derivtrafo(Core::LinAlg::Initialization::zero);

    Core::FE::boundary_gp_to_parent_gp<SLAVE_NUMDOF>(
        pqxg, derivtrafo, intpoints, coupl_ele->shape(), fele->shape(), fele->face_parent_number());

    Core::LinAlg::Matrix<SLAVE_NUMDOF, 1> pxsi(Core::LinAlg::Initialization::zero);

    // coordinates of the current integration point in parent coordinate system
    for (unsigned int idim = 0; idim < SLAVE_NUMDOF; idim++)
    {
      pxsi(idim) = pqxg(0, idim);
    }
    if (coupl_ele->shape() == Core::FE::CellType::hex8)
    {
      const size_t PARENT_NEN = Core::FE::num_nodes(Core::FE::CellType::hex8);
      Core::LinAlg::Matrix<PARENT_NEN, 1> pfunc_loc(
          Core::LinAlg::Initialization::zero);  // derivatives of parent element shape functions
                                                // in parent element coordinate system
      Core::LinAlg::Matrix<SLAVE_NUMDOF, PARENT_NEN> pderiv_loc(
          Core::LinAlg::Initialization::zero);  // derivatives of parent element shape functions
                                                // in parent element coordinate system

      // evaluate derivatives of parent element shape functions at current integration point in
      // parent coordinate system
      Core::FE::shape_function<Core::FE::CellType::hex8>(pxsi, pfunc_loc);
      Core::FE::shape_function_deriv1<Core::FE::CellType::hex8>(pxsi, pderiv_loc);
      //
      // get Jacobian matrix and determinant w.r.t. spatial configuration
      //
      // |J| = det(xjm) * det(Jmat^-1) = det(xjm) * 1/det(Jmat)
      //
      //    _                     _
      //   |  x_1,1  x_2,1  x_3,1  |           d x_i
      //   |  x_1,2  x_2,2  x_3,2  | = xjm  = --------
      //   |_ x_1,3  x_2,3  x_3,3 _|           d s_j
      //    _                     _
      //   |  X_1,1  X_2,1  X_3,1  |           d X_i
      //   |  X_1,2  X_2,2  X_3,2  | = Jmat = --------
      //   |_ X_1,3  X_2,3  X_3,3 _|           d s_j
      //
      Core::LinAlg::Matrix<SLAVE_NUMDOF, SLAVE_NUMDOF> xjm;
      Core::LinAlg::Matrix<SLAVE_NUMDOF, SLAVE_NUMDOF> Jmat;

      Core::LinAlg::Matrix<SLAVE_NUMDOF, PARENT_NEN> xrefe(
          Core::LinAlg::Initialization::zero);  // material coord. of parent element
      Core::LinAlg::Matrix<SLAVE_NUMDOF, PARENT_NEN> xcurr(
          Core::LinAlg::Initialization::zero);  // current  coord. of parent element

      // update element geometry of parent element
      {
        Core::Nodes::Node** nodes = coupl_ele->nodes();
        for (unsigned int inode = 0; inode < PARENT_NEN; ++inode)
        {
          for (unsigned int idof = 0; idof < SLAVE_NUMDOF; ++idof)
          {
            int lid =
                fulldispnp_->get_map().lid(get_cond_dis()->dof(0, coupl_ele->nodes()[inode], idof));

            const auto& x = nodes[inode]->x();
            xrefe(idof, inode) = x[idof];

            if (lid != -1)
              xcurr(idof, inode) = xrefe(idof, inode) + fulldispnp_->local_values_as_span()[(lid)];
            else
              FOUR_C_THROW("Local ID for dispnp not found (lid = -1)!");
          }
          int lidp = fullpres_->get_map().lid(lm_struct_x_lm_pres_.operator[](
              get_cond_dis()->dof(0, coupl_ele->nodes()[inode], 2)));

          if (lidp != -1)
            pres += fullpres_->local_values_as_span()[(lidp)] * pfunc_loc(inode);
          else
            FOUR_C_THROW("Local ID for pressure not found (lid = -1)!");
        }
      }

      xjm.multiply_nt(pderiv_loc, xcurr);
      Jmat.multiply_nt(pderiv_loc, xrefe);
      double det = xjm.determinant();
      double detJ = Jmat.determinant();
      double J = det / detJ;
      return J;
    }
    else
      FOUR_C_THROW(
          "t_det_deformation_gradient for type {} not yet implemented, just add your element type!",
          (Core::FE::cell_type_to_string(coupl_ele->shape())).c_str());
    return -1.0;
  }
  else
    FOUR_C_THROW(
        "t_det_deformation_gradient for type {} not yet implemented, just add your element type!",
        (Core::FE::cell_type_to_string(fele->shape())).c_str());
  return -1.0;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
bool XFEM::MeshCouplingFPI::initialize_fluid_state(std::shared_ptr<Cut::CutWizard> cutwizard,
    std::shared_ptr<Core::FE::Discretization> fluiddis,
    std::shared_ptr<XFEM::ConditionManager> condition_manager,
    std::shared_ptr<Teuchos::ParameterList> fluidparams)
{
  if (contact_)
    get_contact_comm()->initialize_fluid_state(cutwizard, fluiddis, condition_manager, fluidparams);
  return contact_;
}

FOUR_C_NAMESPACE_CLOSE

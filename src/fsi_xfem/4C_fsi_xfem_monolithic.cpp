// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_xfem_monolithic.hpp"

#include "4C_adapter_ale_fpsi.hpp"
#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_poro_wrapper.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_meshtying_contact_bridge.hpp"
#include "4C_contact_nitsche_strategy_fsi.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fem_discretization_nullspace.hpp"
#include "4C_fluid_xfluid.hpp"
#include "4C_fsi_xfem_XFAcoupling_manager.hpp"
#include "4C_fsi_xfem_XFFcoupling_manager.hpp"
#include "4C_fsi_xfem_XFPcoupling_manager.hpp"
#include "4C_fsi_xfem_XFScoupling_manager.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_poroelast_monolithic.hpp"
#include "4C_utils_parameter_list.hpp"
#include "4C_xfem_condition_manager.hpp"
#include "4C_xfem_xfluid_contact_communicator.hpp"

#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN
/*----------------------------------------------------------------------*/
// constructor
/*----------------------------------------------------------------------*/
FSI::MonolithicXFEM::MonolithicXFEM(MPI_Comm comm, const Teuchos::ParameterList& timeparams,
    const Adapter::FieldWrapper::Fieldtype type)
    : AlgorithmXFEM(comm, timeparams, type),
      fsidyn_(Global::Problem::instance()->fsi_dynamic_params()),
      fsimono_(fsidyn_.sublist("MONOLITHIC SOLVER")),
      xfluidparams_(Global::Problem::instance()->x_fluid_dynamic_params()),
      xfpsimono_(xfluidparams_.sublist("XFPSI MONOLITHIC")),
      solveradapttol_(true),
      solveradaptolbetter_(fsimono_.get<double>("ADAPTIVEDIST")),  // adaptive distance
      merge_fsi_blockmatrix_(false),
      scaling_infnorm_(fsimono_.get<bool>("INFNORMSCALING")),
      log_(nullptr),
      /// tolerance and for linear solver
      tolrhs_(fsimono_.get<double>(
          "BASETOL")),  // absolute tolerance for full residual for adapting the linear solver
      /// iteration counter
      iter_(0),
      iter_outer_(0),
      itermin_(xfpsimono_.get<int>("ITEMIN")),
      itermax_(fsimono_.get<int>("ITEMAX")),
      itermax_outer_(xfpsimono_.get<int>("ITEMAX_OUTER")),
      /// Convergence criterion and convergence tolerances for Newton scheme
      normtypeinc_(Teuchos::getIntegralValue<Inpar::FSI::ConvNorm>(fsimono_, "NORM_INC")),
      normtypefres_(Teuchos::getIntegralValue<Inpar::FSI::ConvNorm>(fsimono_, "NORM_RESF")),
      combincfres_(Teuchos::getIntegralValue<Inpar::FSI::BinaryOp>(fsimono_, "NORMCOMBI_RESFINC")),
      tolinc_(fsimono_.get<double>("CONVTOL")),
      tolfres_(fsimono_.get<double>("CONVTOL")),
      /// set tolerances for nonlinear solver
      /// tolerances for structural displacements
      tol_dis_res_l2_(fsimono_.get<double>("TOL_DIS_RES_L2")),
      tol_dis_res_inf_(fsimono_.get<double>("TOL_DIS_RES_INF")),
      tol_dis_inc_l2_(fsimono_.get<double>("TOL_DIS_INC_L2")),
      tol_dis_inc_inf_(fsimono_.get<double>("TOL_DIS_INC_INF")),
      /// tolerances for fluid pressure
      tol_pre_res_l2_(fsimono_.get<double>("TOL_PRE_RES_L2")),
      tol_pre_res_inf_(fsimono_.get<double>("TOL_PRE_RES_INF")),
      tol_pre_inc_l2_(fsimono_.get<double>("TOL_PRE_INC_L2")),
      tol_pre_inc_inf_(fsimono_.get<double>("TOL_PRE_INC_INF")),
      /// tolerances for fluid velocity
      tol_vel_res_l2_(fsimono_.get<double>("TOL_VEL_RES_L2")),
      tol_vel_res_inf_(fsimono_.get<double>("TOL_VEL_RES_INF")),
      tol_vel_inc_l2_(fsimono_.get<double>("TOL_VEL_INC_L2")),
      tol_vel_inc_inf_(fsimono_.get<double>("TOL_VEL_INC_INF")),
      nd_newton_damping_(xfpsimono_.get<bool>("ND_NEWTON_DAMPING")),
      nd_newton_incmax_damping_(nd_newton_damping_),
      nd_levels_(3),
      nd_reduction_fac_(0.75),
      nd_increase_fac_((1 - (1 - nd_reduction_fac_) * 0.5)),
      nd_normrhs_old_(std::vector<double>(nd_levels_, 1e200)),
      nd_maxscaling_(1.0),
      nd_max_incnorm_(std::vector<double>(5, -1.0)),
      nd_act_scaling_(1.0),
      nd_inc_scaling_(1.0),
      cut_evaluate_mintol_(xfpsimono_.get<double>("CUT_EVALUATE_MINTOL")),
      cut_evaluate_miniter_(xfpsimono_.get<int>("CUT_EVALUATE_MINITER")),
      cut_evaluate_dynamic_(cut_evaluate_mintol_ > 1e-16),
      have_contact_(false),
      xf_c_comm_(nullptr)
{
  if (nd_newton_damping_)
  {
    nd_max_incnorm_[0] = xfpsimono_.get<double>("ND_MAX_DISP_ITERINC");
    nd_max_incnorm_[1] = xfpsimono_.get<double>("ND_MAX_VEL_ITERINC");
    nd_max_incnorm_[2] = xfpsimono_.get<double>("ND_MAX_PRES_ITERINC");
    nd_max_incnorm_[3] = xfpsimono_.get<double>("ND_MAX_PVEL_ITERINC");
    nd_max_incnorm_[4] = xfpsimono_.get<double>("ND_MAX_PPRES_ITERINC");
    if (!(nd_max_incnorm_[0] > 0 || nd_max_incnorm_[1] > 0 || nd_max_incnorm_[2] > 0 ||
            nd_max_incnorm_[3] > 0 || nd_max_incnorm_[4] > 0))
      nd_newton_incmax_damping_ = false;
  }

  // TODO set some of these flags via the input file

  //  const Teuchos::ParameterList& xdyn       = Global::Problem::instance()->XFEMGeneralParams();
  //  const Teuchos::ParameterList& xfluiddyn  = Global::Problem::instance()->XFluidDynamicParams();

  //-------------------------------------------------------------------------
  // write files
  //-------------------------------------------------------------------------

  // write iterations-file
  std::string fileiter = Global::Problem::instance()->output_control_file()->file_name();
  fileiter.append(".iteration");
  log_ = std::make_shared<std::ofstream>(fileiter.c_str());

  // write energy-file
  if (fsidyn_.sublist("MONOLITHIC SOLVER").get<bool>("ENERGYFILE"))
  {
    FOUR_C_THROW("writing energy not supported yet");
    //  TODO
    //    std::string fileiter2 = Global::Problem::instance()->output_control_file()->file_name();
    //    fileiter2.append(".fsienergy");
    //    logenergy_ = Teuchos::rcp(new std::ofstream(fileiter2.c_str()));
  }


  //-------------------------------------------------------------------------
  // time step size adaptivity
  //-------------------------------------------------------------------------
  const bool timeadapton = fsidyn_.sublist("TIMEADAPTIVITY").get<bool>("TIMEADAPTON");

  if (timeadapton)
  {
    FOUR_C_THROW("FSI - TimeIntAdaptivity not supported for XFEM yet");
    // init_tim_int_ada(fsidyn);
  }


  //-------------------------------------------------------------------------
  // Create direct or iterative solver for XFSI system
  //-------------------------------------------------------------------------
  create_linear_solver();


  //-------------------------------------------------------------------------
  // validate parameters for monolithic approach
  //-------------------------------------------------------------------------
  validate_parameters();

  //-------------------------------------------------------------------------
  // Setup Coupling Objects
  //-------------------------------------------------------------------------
  setup_coupling_objects();

  // build ale system matrix in split system
  if (have_ale()) ale_field()->create_system_matrix(ale_field()->interface());


  //-------------------------------------------------------------------------
  // Finish standard fluid_field()->init()!
  // REMARK: We don't want to do this at the beginning, to be able to use std
  // Adapter::Coupling for FA-Coupling
  //-------------------------------------------------------------------------
  const int restart = Global::Problem::instance()->restart();
  if (not restart)
    fluid_field()->create_initial_state();  // otherwise called within the fluid_field-Restart when
                                            // Ale displacements are correct

  // Todo: move that somewhere else
  {
    // set initial field by given function
    // we do this here, since we have direct access to all necessary parameters
    const Teuchos::ParameterList& fdyn = Global::Problem::instance()->fluid_dynamic_params();
    auto initfield = Teuchos::getIntegralValue<Inpar::FLUID::InitialField>(fdyn, "INITIALFIELD");
    if (initfield != Inpar::FLUID::initfield_zero_field)
    {
      int startfuncno = fdyn.get<int>("STARTFUNCNO");
      if (initfield != Inpar::FLUID::initfield_field_by_function and
          initfield != Inpar::FLUID::initfield_disturbed_field_from_function)
      {
        startfuncno = -1;
      }
      fluid_field()->set_initial_flow_field(initfield, startfuncno);
    }
  }
}

/*----------------------------------------------------------------------*
 | setup_coupling_objects                                      ager 06/16 |
 *----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::setup_coupling_objects()
{
  {
    if (structure_poro()->meshtying_contact_bridge() != nullptr)
    {
      if (structure_poro()->meshtying_contact_bridge()->have_contact())
      {
        CONTACT::NitscheStrategy* cs = dynamic_cast<CONTACT::NitscheStrategy*>(
            &structure_poro()->meshtying_contact_bridge()->get_strategy());
        if (!cs)
          FOUR_C_THROW(
              "FSI::MonolithicXFEM: Only Nitsche Contact Strategy for XFSCI/XFPSCI available yet!");
        if (cs->contact_interfaces().size() > 1)
          FOUR_C_THROW("FSI::MonolithicXFEM: Only one contact interface supported!");

        have_contact_ = true;

        // Do contact and xfluid communication stuff
        xf_c_comm_ = std::make_shared<XFEM::XFluidContactComm>(*cs);
        xf_c_comm_->initialize_fluid_state(fluid_field()->get_cut_wizard(),
            fluid_field()->discretization(), fluid_field()->get_condition_manager(),
            fluid_field()->params());

        xf_c_comm_->setup_surf_ele_ptrs(cs->contact_interfaces()[0]->discret());

        for (int i = 0; i < (int)cs->contact_interfaces().size(); ++i)
        {
          cs->contact_interfaces()[i]
              ->interface_params()
              .set<std::shared_ptr<XFEM::XFluidContactComm>>("XFluidContactComm", xf_c_comm_);
        }
      }
    }
  }
  int coup_idx = 0;
  std::vector<int> idx;

  // Just Add coupling object in case there is an FSI Interface
  if (fluid_field()->get_condition_manager()->get_mesh_coupling("XFEMSurfFSIMono") != nullptr)
  {
    idx.push_back(structp_block_);
    idx.push_back(fluid_block_);
    coup_man_[coup_idx] =
        std::make_shared<XFEM::XfsCouplingManager>(fluid_field()->get_condition_manager(),
            structure_poro()->structure_field(), fluid_field(), idx);

    if (have_contact_)
    {
      std::dynamic_pointer_cast<XFEM::MeshCouplingFSI>(
          fluid_field()->get_condition_manager()->get_mesh_coupling("XFEMSurfFSIMono"))
          ->assign_contact_comm(xf_c_comm_);  // assign to mesh coupling object
    }
  }

  if (have_ale())
  {
    ++coup_idx;
    idx.clear();
    idx.push_back(fluid_block_);
    idx.push_back(ale_i_block_);
    idx.push_back(structp_block_);
    coup_man_[coup_idx] = std::make_shared<XFEM::XfaCouplingManager>(
        fluid_field(), ale_field(), idx, structure_poro()->structure_field());
  }

  if (fluid_field()->get_condition_manager()->get_mesh_coupling("XFEMSurfFluidFluid") !=
      nullptr)  // TODO: fluid fluid!!!
  {
    ++coup_idx;
    idx.clear();
    idx.push_back(fluid_block_);
    idx.push_back(fluid_block_);
    coup_man_[coup_idx] = std::make_shared<XFEM::XffCouplingManager>(
        fluid_field()->get_condition_manager(), fluid_field(), fluid_field(), idx);
  }

  if (structure_poro()->is_poro())
  {
    // Just Add coupling object in case there is an FPI Interface
    if (fluid_field()->get_condition_manager()->get_mesh_coupling("XFEMSurfFPIMono_ps_ps") !=
        nullptr)
    {
      ++coup_idx;
      idx.clear();
      idx.push_back(structp_block_);
      idx.push_back(fluid_block_);
      idx.push_back(fluidp_block_);
      coup_man_[coup_idx] =
          std::make_shared<XFEM::XfpCouplingManager>(fluid_field()->get_condition_manager(),
              structure_poro()->poro_field(), fluid_field(), idx);

      if (have_contact_)
      {
        std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(
            fluid_field()->get_condition_manager()->get_mesh_coupling("XFEMSurfFPIMono_ps_ps"))
            ->assign_contact_comm(xf_c_comm_);  // assign to mesh coupling object
        std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(
            fluid_field()->get_condition_manager()->get_mesh_coupling("XFEMSurfFPIMono_pf_ps"))
            ->assign_contact_comm(xf_c_comm_);  // assign to mesh coupling object
        std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(
            fluid_field()->get_condition_manager()->get_mesh_coupling("XFEMSurfFPIMono_ps_pf"))
            ->assign_contact_comm(xf_c_comm_);  // assign to mesh coupling object
        std::dynamic_pointer_cast<XFEM::MeshCouplingFPI>(
            fluid_field()->get_condition_manager()->get_mesh_coupling("XFEMSurfFPIMono_pf_pf"))
            ->assign_contact_comm(xf_c_comm_);  // assign to mesh coupling object
      }
    }
  }


  // ------------------------------------------------------------------
  // set the current interface displacement to the fluid field to be used in the cut
  // ------------------------------------------------------------------
  for (auto coupit = coup_man_.begin(); coupit != coup_man_.end(); ++coupit)
    coupit->second->init_coupling_states();
}

/*----------------------------------------------------------------------*
 | validate the input parameter combinations               schott 07/14 |
 *----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::validate_parameters()
{
  // check for reasonable input parameter combinations!

  // Check for the timestepsize
  if (fabs(fluid_field()->dt() - structure_poro()->structure_field()->dt()) > 1e-16)
  {
    FOUR_C_THROW("validate_parameters(): Timestep of fluid and structure not equal ({} != {})!",
        fluid_field()->dt(), structure_poro()->structure_field()->dt());
  }
  if (have_ale())
  {
    if (fabs(fluid_field()->dt() - ale_field()->dt()) > 1e-16)
    {
      FOUR_C_THROW("validate_parameters(): Timestep of fluid and ale not equal ({} != {})!",
          fluid_field()->dt(), ale_field()->dt());
    }
  }
  if (structure_poro()->is_poro())
  {
    if (fabs(fluid_field()->dt() - structure_poro()->poro_field()->dt()) > 1e-16)
    {
      FOUR_C_THROW("validate_parameters(): Timestep of fluid and poro not equal ({} != {})!",
          fluid_field()->dt(), structure_poro()->poro_field()->dt());
    }
  }

  // TODO
  // REMARK: be aware of using const Dis predictor!
  //         This results in zero disp_incr and u^n+1 = -u^n for second order disp_to_vel interface
  //         conversion
  //                                        and u^n+1 = 0    for first order disp_to_vel interface
  //                                        conversion
}

/*----------------------------------------------------------------------*
 | setup of the monolithic XFSI system,                    schott 08/14 |
 | setup a new combined block row map and a new block matrix            |
 *----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::setup_system()
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicXFEM::SetupSystem()");

  /*----------------------------------------------------------------------
   Create a combined map for Structure/Fluid-DOFs all in one!
   ----------------------------------------------------------------------*/

  create_combined_dof_row_map();


  /*----------------------------------------------------------------------
    Initialise XFSI-systemmatrix_
   ----------------------------------------------------------------------*/
  create_system_matrix();
}

/*----------------------------------------------------------------------*
 | setup of the monolithic XFSI system,                    schott 08/14 |
 | setup a new combined block row map and a new block matrix            |
 *----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::create_system_matrix()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "Create a new global systemmatrix (BlockSparseMatrix)" << std::endl;

  // TODO: check the savegraph option and explicit Dirichlet flag for the matrix!
  // TODO: check if it is okay to use a BlockSparseMatrix without the FE-flag for the
  // fluid-submatrix and the coupling submatrices?
  // TODO: do we add a already communicated (completed) fluid matrix?!
  // TODO: check the number of non-zeros predicted
  /*----------------------------------------------------------------------*/

  if (systemmatrix_.use_count() > 1)
  {
    FOUR_C_THROW(
        "deleting systemmatrix does not work properly, the number of RCPs pointing to it is {}",
        systemmatrix_.use_count());
  }

  // do not want to have two sysmats in memory at the same time
  if (systemmatrix_ != nullptr)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      std::cout << "Delete the global systemmatrix (BlockSparseMatrix)" << std::endl;
    systemmatrix_ = nullptr;
  }

  systemmatrix_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          extractor(), extractor(), 0,
          false,  // explicit dirichlet, do not change the graph and do not create a new matrix when
                  // applying Dirichlet values
          false   // savegraph (used when submatrices will be reset), we create new fluid sysmats
                  // anyway
      );
}


/*----------------------------------------------------------------------*
 * setup composed system matrix from field solvers, complete the global system matrix
 *----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::setup_system_matrix()
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicXFEM::setup_system_matrix");

  // reset the block system matrix
  // note: Zero() is not sufficient for the coupling blocks,
  // as the couplings between fluid and structure can change (structure moves between iterations)
  // while the fluid dofsets remain unchanged
  //  systemmatrix_->reset();

  /*----------------------------------------------------------------------*/
  // extract Jacobian matrices and put them into composite system
  std::shared_ptr<Core::LinAlg::SparseMatrix> f = fluid_field()->system_matrix();

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  // get time integration parameters of structure and fluid time integrators
  // as well as of the FSI time-integration
  // to enable consistent time integration among the fields
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/

  /*----------------------------------------------------------------------*/
  // scaling factors for fluid terms/blocks
  // inverse of the weighting of the quantities w.r.t the new time step
  const double scaling_F = fluid_field()->residual_scaling();  // 1/(theta * dt) = 1/weight^F_np

  /*----------------------------------------------------------------------*/
  // this is the interpolation weight for quantities from last time step
  // alpha_f for genalpha and (1-theta) for OST (weighting of the old time step n for displacements)
  const double stiparam =
      structure_poro()->tim_int_param();  // (1-theta) for OST and alpha_f for Genalpha
  // scale factor for the structure system matrix w.r.t the new time step
  const double scaling_S = 1.0 / (1.0 - stiparam);  // 1/(1-alpha_F) or 1/theta = 1/weight^S_np


  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here.
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/


  /*----------------------------------------------------------------------*/
  // Structure diagonal block (structural system matrix)
  /*----------------------------------------------------------------------*/
  if (!structure_poro()->is_poro())
  {
    // extract Jacobian matrices and put them into composite system
    std::shared_ptr<Core::LinAlg::SparseMatrix> s = structure_poro()->system_matrix();

    // Incomplete structure matrix to be able to deal with slightly defective interface meshes.
    //
    // The additional coupling block C_ss can contain additional non-zero entries,
    // e.g. from DBCs which are already applied to s in the structural evaluate, however, not
    // to the coupling block C_ss yet
    s->un_complete();

    // NOTE: UnComplete creates a new Matrix and a new matrix graph as well which is not allocated
    // with staticprofile Therefore, the savegraph = true option set in structural timint has no
    // effect, as a new graph is created whenever UnComplete is called then, due to memory
    // fragmentation the evaluate time for the structure can vary a lot! UPDATE: actually after the
    // next complete with store the graph

    // scale the structure system matrix
    s->scale(scaling_S);

    // assign the structure sysmat diagonal block
    systemmatrix_->assign(structp_block_, structp_block_, Core::LinAlg::DataAccess::Share, *s);
  }
  else  // we use a block structure for poro
  {
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> ps =
        structure_poro()->block_system_matrix();
    ps->un_complete();
    ps->scale(scaling_S);
    systemmatrix_->assign(
        structp_block_, structp_block_, Core::LinAlg::DataAccess::Share, ps->matrix(0, 0));  // psps
    systemmatrix_->assign(
        fluidp_block_, structp_block_, Core::LinAlg::DataAccess::Share, ps->matrix(1, 0));  // pfps
    systemmatrix_->assign(
        structp_block_, fluidp_block_, Core::LinAlg::DataAccess::Share, ps->matrix(0, 1));  // pspf
    systemmatrix_->assign(
        fluidp_block_, fluidp_block_, Core::LinAlg::DataAccess::Share, ps->matrix(1, 1));  // pfpf
  }

  /*----------------------------------------------------------------------*/
  // Fluid diagonal block
  /*----------------------------------------------------------------------*/

  // scale the fluid diagonal block
  f->scale(scaling_F);  //<  1/(theta_f*dt) = 1/weight(t^f_np)

  // assign the fluid diagonal block
  systemmatrix_->assign(fluid_block_, fluid_block_, Core::LinAlg::DataAccess::Share, *f);

  // Add Coupling Sysmat
  for (auto coupit = coup_man_.begin(); coupit != coup_man_.end(); ++coupit)
    coupit->second->add_coupling_matrix(*systemmatrix_, scaling_F);

  /*----------------------------------------------------------------------*/
  // Complete the global system matrix
  /*----------------------------------------------------------------------*/

  // done. make sure all blocks are filled.
  systemmatrix_->complete();
}


/*----------------------------------------------------------------------*/
// setup composed right hand side from field solvers
/*----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::setup_rhs()
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicXFEM::setup_rhs");

  // We want to add into a zero vector
  rhs_->put_scalar(0.0);

  // contributions of single field residuals
  setup_rhs_residual(*rhs_);

  // Add Coupling RHS
  const double scaling_F = fluid_field()->residual_scaling();
  for (auto coupit = coup_man_.begin(); coupit != coup_man_.end(); ++coupit)
    coupit->second->add_coupling_rhs(rhs_, extractor(), scaling_F);
}

/*----------------------------------------------------------------------*/
// setup RHS contributions based on single field residuals
/*----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::setup_rhs_residual(Core::LinAlg::Vector<double>& f)
{
  /*----------------------------------------------------------------------*/
  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  /*----------------------------------------------------------------------*/

  /*----------------------------------------------------------------------*/
  // scaling factors for fluid terms/blocks
  // inverse of the weighting of the quantities w.r.t the new time step
  const double scaling_F = fluid_field()->residual_scaling();  // 1/(theta * dt) = 1/weight^F_np

  /*----------------------------------------------------------------------*/
  // this is the interpolation weight for quantities from last time step
  // alpha_f for genalpha and (1-theta) for OST (weighting of the old time step n for displacements)
  const double stiparam = structure_poro()
                              ->structure_field()
                              ->tim_int_param();  // (1-theta) for OST and alpha_f for Genalpha

  // scale factor for the structure system matrix w.r.t the new time step
  const double scaling_S = 1.0 / (1.0 - stiparam);  // 1/(1-alpha_F) = 1/weight^S_np


  /*----------------------------------------------------------------------*/
  // get single field residuals
  Core::LinAlg::Vector<double> sv(*structure_poro()->rhs());
  Core::LinAlg::Vector<double> fv(*fluid_field()->rhs());

  // scale the structural rhs
  sv.scale(scaling_S);

  // scale the fluid rhs
  fv.scale(scaling_F);  // scale with fluid_field()->residual_scaling()

  // put the single field residuals together
  combine_field_vectors(f, sv, fv);
}

/*----------------------------------------------------------------------*/
// apply Dirichlet boundary conditions to XFSI system
/*----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::apply_dbc()
{
  // note, the structural evaluate applies DBC already to the structure block,
  // however, the additional C_ss coupling block is added and we have to apply DBC again to the sum
  // no DBC are applied for in the fluid-evaluate for the ff-diagonal block or the fluid coupling
  // blocks therefore we apply BCS to the whole system. Just the internal DBCs via the structural
  // evaluate (prepare_system_for_newton_solve()) are applied twice

  // apply combined Dirichlet to whole XFSI system
  Core::LinAlg::apply_dirichlet_to_system(
      *systemmatrix_, *iterinc_, *rhs_, *zeros_, *combined_dbc_map());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::initial_guess(Core::LinAlg::Vector<double>& ig)
{
  // TODO: what to do with this function???
  //  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicStructureSplit::initial_guess");
  //
  //  setup_vector(*ig,
  //              structure_field()->initial_guess(),
  //              fluid_field().initial_guess(),
  //              ale_field().initial_guess(),
  //              0.0);
}


/*----------------------------------------------------------------------*/
// Create the combined DOF row map for the FSI problem;
// row maps of structure and xfluid to an global FSI DOF row map
/*----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::create_combined_dof_row_map()
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces_mergedporo;

  // Append the structural DOF map
  vecSpaces.push_back(structure_poro()->structure_field()->dof_row_map());
  vecSpaces_mergedporo.push_back(structure_poro()->dof_row_map());

  // Append the background fluid DOF map
  vecSpaces.push_back(fluid_field()->dof_row_map());
  vecSpaces_mergedporo.push_back(fluid_field()->dof_row_map());

  // solid maps empty??
  if (vecSpaces[structp_block_]->num_global_elements() == 0)
    FOUR_C_THROW("No solid equations. Panic.");

  // fluid maps empty??
  if (vecSpaces[fluid_block_]->num_global_elements() == 0)
    FOUR_C_THROW("No fluid equations. Panic.");

  if (structure_poro()->is_poro())
  {
    vecSpaces.push_back(structure_poro()->fluid_field()->dof_row_map());
    std::shared_ptr<const Core::LinAlg::Map> empty_map =
        std::make_shared<Core::LinAlg::Map>(0, 0, get_comm());
    vecSpaces_mergedporo.push_back(empty_map);
    // porofluid maps empty??
    if (vecSpaces[fluidp_block_]->num_global_elements() == 0)
      FOUR_C_THROW("No porofluid equations. Panic.");
  }

  // Append the background fluid DOF map
  if (have_ale())
  {
    vecSpaces.push_back(ale_field()->interface()->other_map());
    vecSpaces_mergedporo.push_back(ale_field()->interface()->other_map());

    // ale maps empty??
    if (vecSpaces[ale_i_block_]->num_global_elements() == 0)
      FOUR_C_THROW("No ale equations. Panic.");
  }

  // The vector is complete, now fill the system's global block row map
  // with the maps previously set together!
  set_dof_row_maps(vecSpaces, vecSpaces_mergedporo);
}



/*----------------------------------------------------------------------*/
// set full monolithic dof row map
/*----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::set_dof_row_maps(
    const std::vector<std::shared_ptr<const Core::LinAlg::Map>>& maps,
    const std::vector<std::shared_ptr<const Core::LinAlg::Map>>& maps_mergedporo)
{
  std::shared_ptr<Core::LinAlg::Map> fullmap = Core::LinAlg::MultiMapExtractor::merge_maps(maps);
  blockrowdofmap_.setup(*fullmap, maps);
  blockrowdofmap_mergedporo_.setup(*fullmap, maps_mergedporo);
}



/*----------------------------------------------------------------------*
 | Put two field vectors together to a monolithic vector
 *----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::combine_field_vectors(
    Core::LinAlg::Vector<double>& v,         ///< composed vector containing all field vectors
    const Core::LinAlg::Vector<double>& sv,  ///< structuralporo DOFs
    const Core::LinAlg::Vector<double>& fv   ///< fluid DOFs
)
{
  extractor_merged_poro().add_vector(sv, structp_block_, v);
  extractor_merged_poro().add_vector(fv, fluid_block_, v);
}



/*----------------------------------------------------------------------
 |                                                        schott 08/14 |
 |   extract the two field vectors from a given composed vector        |
 ----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::extract_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& fx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& ax)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicXFEM::extract_field_vectors");

  /*----------------------------------------------------------------------*/
  // Process structure unknowns
  /*----------------------------------------------------------------------*/
  // Extract whole structure field vector
  sx = extractor_merged_poro().extract_vector(*x, structp_block_);

  /*----------------------------------------------------------------------*/
  // Process fluid unknowns
  /*----------------------------------------------------------------------*/
  // Extract vector of fluid unknowns from x
  fx = extractor_merged_poro().extract_vector(*x, fluid_block_);

  // Extract vector of ale unknowns from x
  if (have_ale()) ax = extractor().extract_vector(*x, ale_i_block_);
}



/*----------------------------------------------------------------------*
 | time loop of the monolithic system                      schott 08/14 |
 *----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::timeloop()
{
  // time loop
  while (not_finished())
  {
    // counter and print header
    // predict solution of both field (call the adapter)
    prepare_time_step();

    // outer iteration loop when active fluid dofsets change
    // calls inner Newton-Raphson iterations within each outer iteration
    solve();

    // calculate stresses, strains, energies
    constexpr bool force_prepare = false;
    prepare_output(force_prepare);

    // update all single field solvers
    update();

    // write output to screen and files
    output();

  }  // not_finished
}  // TimeLoop()


/*----------------------------------------------------------------------*
 | prepare the time step for fluid and structure           schott 08/14 |
 *----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::prepare_time_step()
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicXFEM::prepare_time_step");

  increment_time_and_step();
  print_header();
  //--------------------------------------------
  // Structure prepare_time_step
  //--------------------------------------------
  // * apply structural predictor
  // * apply Dirichlet conditions and
  // * print the residual based on the structural predictor
  structure_poro()->prepare_time_step();

  //--------------------------------------------
  // Fluid prepare_time_step
  //--------------------------------------------
  // * set time integrator
  // * set time parameters
  // * apply fluid predictor (before the cut is performed the first time in the new time step, see
  // first evaluate call)
  // * DBCs will be applied to predicted solution in a PrepareNonlinearSolve-call within the
  // evaluate
  //   after velnp has been mapped to the new interface position. DBCs will be applied again for
  //   each iteration as the CUT is performed for each increment and therefore the DBCs have to be
  //   set again
  fluid_field()->prepare_time_step();

  // predict coupling states (for relaxing ale mesh!) /after Structure->PrepareTimestep
  for (auto coupit = coup_man_.begin(); coupit != coup_man_.end(); ++coupit)
    coupit->second->predict_coupling_states();

  if (have_contact_) xf_c_comm_->prepare_time_step();

  // now we have relaxed ALE mesh -> set in dispnp
  // for safety apply in ale_field again the standard inner DBCs of this timestep

  if (have_ale())  // Apply inner std-Dirichlet boundary conditions on provided state vector and do
                   // locsys
    ale_field()->prepare_time_step();
}


/*----------------------------------------------------------------------*
 | outer iteration loop, restarts inner Newton-Raphson iterations       |
 | when fluid dofsets changes                              schott 08/14 |
 *----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::solve()
{
  // initialize outer loop iteration index which allows for restarts of the Newton scheme in case of
  // changing fluid maps
  iter_outer_ = 1;

  // reset the single step-increments for structural and fluid field
  sx_sum_ = nullptr;
  fx_sum_ = nullptr;
  ax_sum_ = nullptr;


  // We want to make sure, that the outer loop is entered at least once!
  // We exit the outer loop if either the inner Newton loop is converged (checked in Converged()
  // method) OR the maximum number of outer iterations is exceeded!
  while ((iter_outer_ == 1) or (iter_outer_ <= itermax_outer_))
  {
    //--------------------------------------------------------
    // call the inner Newton loop and check for convergence
    //--------------------------------------------------------
    if (newton())  // stop since the main inner Newton loop converged
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        Core::IO::cout
            << "-------------------------------------- Outer loop finished with converged "
               "newton_loop ---------------------------------------"
            << Core::IO::endl;
      }
      break;
    }
    else
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        Core::IO::cout
            << "---------------------------------------- Restart Newton-Raphson - DOF-sets "
               "changed -----------------------------------------"
            << Core::IO::endl;
      }
    }

    iter_outer_++;
  }

  if (iter_outer_ > itermax_outer_)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      Core::IO::cout
          << "-------------------------- Maximum number of restarts reached - Fluid DOF-sets "
             "have changed too often ----------------------"
          << Core::IO::endl;
    }
  }
}


/*----------------------------------------------------------------------*
 | recover Lagrange multiplier (structural forces) needed for rhs in    |
 | next time step and update single fields                 schott 08/14 |
 *----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::update()
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicXFEM::Update");

  const double scaling_F = fluid_field()->residual_scaling();  // 1/(theta * dt) = 1/weight^F_np
  for (auto coupit = coup_man_.begin(); coupit != coup_man_.end(); ++coupit)
    coupit->second->update(scaling_F);

  // update the single fields
  structure_poro()->update();
  fluid_field()->update();
  if (have_ale()) ale_field()->update();
}



/*----------------------------------------------------------------------*
 | write output                                            schott 08/14 |
 *----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::output()
{
  //--------------------------------
  // output for structural field
  //--------------------------------
  structure_poro()->output();

  //--------------------------------
  // output for Lagrange multiplier field (ie forces onto the structure, Robin-type forces
  // consisting of fluid forces and the Nitsche penalty term contribution)
  //--------------------------------
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  const int uprestart = fsidyn.get<int>("RESTARTEVERY");
  const int upres = fsidyn.get<int>("RESULTSEVERY");
  if ((uprestart != 0 && fluid_field()->step() % uprestart == 0) ||
      fluid_field()->step() % upres == 0)  // Fluid decides about restart, write output
  {
    for (auto coupit = coup_man_.begin(); coupit != coup_man_.end(); ++coupit)
      coupit->second->output(*structure_poro()->structure_field()->disc_writer());
  }

  //--------------------------------
  // output for fluid field - writes the whole GMSH output if switched on
  //--------------------------------
  fluid_field()->output();
  fluid_field()->lift_drag();


  //--------------------------------
  if (structure_poro()->get_constraint_manager()->have_monitor())
  {
    structure_poro()->get_constraint_manager()->compute_monitor_values(*structure_poro()->dispnp());
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      structure_poro()->get_constraint_manager()->print_monitor_values();
  }

  if (have_ale()) ale_field()->output();
}

/*----------------------------------------------------------------------*
 | inner iteration loop (Newton-Raphson scheme)            schott 08/14 |
 | return "true" if converged or                                        |
 | "false" if unconverged or in case of changing fluid dof maps         |
 *----------------------------------------------------------------------*/
bool FSI::MonolithicXFEM::newton()
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicXFEM::Newton");


  //--------------------------------------------------------
  // Perform Newton-Raphson iterations as long as the size of the global system does not change
  // between iterations.
  // * Due to the different structural interface positions the fluid dofsets can change:
  //   -> in case that the number of dofsets per node changes or a simple copying of dofs between
  //   std and multiple ghost dofsets
  //      of a node is not possible anymore we have to restart the Newton-scheme
  // * During the Newton iterations the sorting of fluid-dofsets can change. However, for the update
  // of vectors
  //   in a Newton scheme we have to use vectors that do not change ordering of dofs. Therefore
  //   permutation of the increments and other vectors before and after solving the linear systems
  //   can be necessary
  // * Then, the convergence of the Newton scheme is observed based on vectors whose dofsets do not
  // change during this restart
  //   call.
  // * When restarting the Newton, we have to ensure that the step-increments for structure and
  // fluid are initialized
  //   properly, see below.
  // * The fluid and structure evaluate routines expect the full step-increment w.r.t old solution
  // t^n, therefore we have
  //   to sum up the Newton-increments to full step-increment
  // * Applying Dirichlet values to rhs and to system has to be done
  //   for the global system after Evaluate routines for the single fields have been called and have
  //   summed up to the global residual
  // * in xfluid we set Dirichlet values into velnp, therefore the stepinc used in evaluate adds
  // zeros for Dirichlet entries
  //   not to modify the already set Dirichlet values
  //--------------------------------------------------------


  /*----------------------------------------------------------------------*/
  // Initialization
  /*----------------------------------------------------------------------*/
  // Iteration counter
  iter_ = 1;

  permutation_map_.clear();
  permutation_.clear();


  /*----------------------------------------------------------------------*/
  // Create a new solver, important in particular when fluid null space changes
  /*----------------------------------------------------------------------*/
  create_linear_solver();

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  // Newton-Raphson iteration with unchanging, however, permuting dofsets
  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  // We want to make sure, that the loop is entered at least once!
  // We exit the loop if either the convergence criteria are met (checked in
  // Converged() method) OR the maximum number of inner iterations is exceeded!
  while ((iter_ + (iter_outer_ - 1)) <= itermin_ or ((not converged()) and (iter_ <= itermax_)))
  {
    //    std::cout << "Evaluate-Call " << "iter_ " << iter_ << "/" << itermax_ << std::endl;

    /*----------------------------------------------------------------------*/
    // evaluate()- call
    // * calls evaluate methods for single fields
    // * assembles single field rhs and system-matrices and fluid-structure coupling blocks
    // * check if dofsets between last two Newton iterations drastically changed or maybe simply
    // permuted
    /*----------------------------------------------------------------------*/

    if (systemmatrix_.use_count() > 1)
    {
      FOUR_C_THROW(
          "deleting block sparse matrix does not work properly, the number of RCPs pointing to it "
          "is {}",
          systemmatrix_.use_count());
    }

    // reduce counter in the rcp-pointers pointing to underlying matrix objects which
    // actually hold large chunks of memory this ensures that the single field matrices can be
    // really deleted (memory can be freed) before we can create a new state class in fluid's
    // Evaluate NOTE: the blocksparsematrix' sparse matrices hold strong RCP's to the single-fields
    // matrix objects NOTE: fluid's Evaluate will create a new Core::LinAlg::SparseMatrix and
    // coupling matrices anyway
    systemmatrix_ = nullptr;
    // TODO: can we delete the solver here? this is done in solver_->Reset after solving the last
    // system
    //    solver_ = nullptr;

    const bool changed_fluid_dofsets = evaluate();


    //-------------------
    // store fluid step-increment
    //-------------------
    // this step-increment is based on the old solution from t^n (veln) mapped to the interface
    // position of the current Newton iteration (via XFEM-timeintegration) and based on the
    // permutation of dofsets of the current Newton iteration in contrast to the fluid-block of the
    // global x_sum_-step increment (see below) note: current iteration velnp (velnp_ip) and veln
    // have been already mapped to the current interface position
    //       via the Evaluate-call above
    // safe velnp_i+1 - veln in fx_sum_ as start for the Newton as Fluid-Evaluate expects the full
    // step-increment

    fx_sum_ = std::make_shared<Core::LinAlg::Vector<double>>(*fluid_field()->dof_row_map());
    fx_sum_->update(1.0, *fluid_field()->velnp(), -1.0, *fluid_field()->veln(), 0.0);

    //-------------------
    // store ALE step-increment
    //-------------------

    if (have_ale())
    {
      ax_sum_ = std::make_shared<Core::LinAlg::Vector<double>>(*(extractor().map(ale_i_block_)));
      ax_sum_->update(1.0, *ale_field()->interface()->extract_other_vector(*ale_field()->dispnp()),
          -1.0, *ale_field()->interface()->extract_other_vector(*ale_field()->dispn()), 0.0);
    }

    //-------------------
    // store structure step-increment
    //-------------------
    /*
        // save the current structural step-increment (also possible // sx_sum_ = sx;
        sx_sum_ = Teuchos::rcp(new
       Core::LinAlg::Vector<double>(*(Extractor().map(structp_block_)))); if(x_sum_ !=
       nullptr)
        {
          int errsx = sx_sum_->Update(1.0, *Extractor().extract_vector(*x_sum_,structp_block_),
       0.0); if(errsx != 0) FOUR_C_THROW("update not successful");
        }
    */
    /*----------------------------------------------------------------------*/
    // Perform at least one solve after the first Evaluate-call.
    // For further iterations, decide if a Newton-restart is required based on the first computed
    // Newton increment
    /*----------------------------------------------------------------------*/
    if (iter_ > 1)
    {
      if (changed_fluid_dofsets) return false;

      setup_system();  // set new blockdofrowmap and create a new global block sparse matrix
    }
    else if (iter_ == 1)  // the first run
    {
      //-------------------
      // initialize a new system after the first evaluate-run
      //-------------------
      // note: after the first fluid-evaluate run, the size of the system is now known and new
      // global vectors can be created
      //       until a new restart has to be performed, the dofsets are assumed not to change,
      //       however, the std-ghost dofsets of a node can permute as a a priori sorting of sets is
      //       not possible when the interface position changes

      //-------------------
      // setup a new system since the size of the system has changed when NewtonFull is called
      //-------------------
      setup_system();  // set new blockdofrowmap and create a new global block sparse matrix


      //-------------------
      // Create a new Core::LinAlg::Vector<double> with a given row map and initialize it!
      // dof_row_map() contains all DOFS for the monolithic system w.r.t the respective interface
      // position and returns the dof_row_map of the blockrowdofmap_, a MapExtractor object for a
      // DOF map, split into blocks.
      //-------------------

      //-------------------
      // new global vectors based on UNCHANGING/NON-PERMUTING dofsets
      //-------------------

      // Global increment sum vector (= Global step-increment vector)
      // That's the total increment (structure + fluid) w.r.t the old time step t^n.
      // When using a structural predictor or when restarting the Newton, we have to initialize this
      // vector such that it contains the increment w.r.t the old time step t^n and not w.r.t the
      // restarted solution NOTE:
      //   - the structural dofs do not change, x_sum(0) (structure step-increment) is given w.r.t
      //   predictor-solution
      //     from the beginning of the new time-step
      //   - the ALE dofs do also not change, x_sum(2) (ale step-increment) is given w.r.t. old time
      //   step Dispn(), even if a relaxation-predictor is called
      //   - the fluid dofs can permute between Newton iterations and dofsets can completely change
      //   such that a Newton restart
      //     is required. x_sum(1) (fluid step-increment) is given w.r.t the old solution at t^n
      //     (veln) mapped/permuted to the interface position of first evaluate-call after
      //     (re-)starting the Newton scheme (in contrast to fx_sum_!) and so the global dofset of
      //     this vector remains unchanged/non-permuted until the next restart has to be performed.
      //     In order to update this vector and to use it for further evaluate-calls permutations
      //     backward/forward of the fluid block have to be applied
      x_sum_ = Core::LinAlg::create_vector(*dof_row_map(), true);


      //-------------------
      // new global vectors based on UNCHANGING/NON-PERMUTING dofsets before and after the solve,
      // however, PERMUTING dofsets during the solve itself
      //-------------------

      // Global solution vector for linear solve = iteration increment Delta x = x^n+1_i+1 - x^n+1_i
      // based on non-changing, however, permuting dofsets during the Newton.
      // During the solve we use permuting fluid vectors, depending on the std/ghost-dofset order of
      // each iteration. After the solve we directly have to permute the fluid block Newton
      // increment backwards to the reference ordering (see x_sum) to update the step-increment
      // (xsum) which does not change between the Newton iterations as long as no Newton restart is
      // necessary (see linear_solve()) note: during the solve the increment vector can have
      // permuted dofsets,
      //       directly after the linear_solve the vector is permuted backwards to the initial
      //       ordering of creation
      iterinc_ = Core::LinAlg::create_vector(*dof_row_map(), true);

      // Global residual vector, unchanged/permuting dofsets (the same as for the iterinc_ vector)
      // note: for the assembly and during the solve the residual vector can have permuted dofsets,
      //       directly after the linear_solve the vector is permuted backwards to the initial
      //       ordering of creation
      rhs_ = Core::LinAlg::create_vector(*dof_row_map(), true);

      // Global zero vector for DBCs, unchanged/permuting dofsets (the same as for the iterinc_
      // vector) note: this vector is just used during the solve and is NOT permuted backwards as
      // iterinc or rhs
      zeros_ = Core::LinAlg::create_vector(*dof_row_map(), true);
    }
    else
    {
      FOUR_C_THROW("the Newton iteration index is assumed to be >= 0");
    }


    /*----------------------------------------------------------------------*/
    // initialize the structural and fluid part of the global step-increment
    /*----------------------------------------------------------------------*/
    if (iter_ == 1)
    {
      //-------------------
      // initialize the structural part of the global step-increment
      //-------------------
      // it is based on the structural predictor-solution D^(n+1)_(pred,k=0)
      // and has been set at the beginning of the first outer iteration using the structural
      // prepare_time_step-call

      // if not a new time-step, take the step-increment which has been summed up so far
      if (sx_sum_ != nullptr) extractor_merged_poro().add_vector(*sx_sum_, structp_block_, *x_sum_);

      //-------------------
      // initialize the fluid part of the global step-increment
      //-------------------
      // note: during the Newton fx_sum_ and x_sum_(1) can have different dofset orderings.
      //       However, in the first iteration, the two dofsets are equal and permutation is not
      //       necessary at the beginning. After a restart, the interface position w.r.t which we
      //       stored fx_sum_ before the restart and the interface position for the first
      //       fluid-evaluate call are the same (then we can ensure the same dofset sorting). At the
      //       beginning of a new time step fx_sum_ and x_sum are created based on the same
      //       dof_row_map.
      extractor().insert_vector(*fx_sum_, fluid_block_, *x_sum_);

      // if not a new time-step, take the step-increment which has been summed up so far
      if (have_ale() && ax_sum_ != nullptr)
      {
        extractor().add_vector(*ax_sum_, ale_i_block_, *x_sum_);
      }
    }

    /*----------------------------------------------------------------------*/
    // Setup the new linear system and solve it and check convergence
    /*----------------------------------------------------------------------*/

    //-------------------
    // Build the linear system
    // J^{n+1}(x_i) \Delta^{n+1}_{x_i+1}=-r^{n+1}(x_i)
    // i: Newton iteration counter
    // J: Jacobian
    // r: RHS-vector
    //-------------------
    setup_system_matrix();

    if (not systemmatrix_->filled()) FOUR_C_THROW("Unfilled system matrix! Fatal error!");

    // Create the RHS consisting of the field residuals and coupling term residuals
    setup_rhs();


    //-------------------
    // Apply Dirichlet BCs to the whole system
    //-------------------
    apply_dbc();


    //-------------------
    // Solver call
    //-------------------
    linear_solve();


    //-------------------
    // Build residual and incremental norms, count the DOFs!
    //-------------------
    build_convergence_norms();


    //-------------------
    // Give some output
    //-------------------
    print_newton_iter();



    // Increment loop index
    iter_ += 1;

  }  // End of Newton loop!

  // After the loop exit, the iteration counter is 1 higher than the true no. of
  // iterations! Correct that:
  iter_ -= 1;

  // Note:
  // the iteration increment computed at the latest will not be added finally,
  // in doing so, also the lambda-forces correspond to the iteration before.
  // When the residual was small enough and the new iteration increment is also sufficient small
  // this is fine. The lambda-forces and the current iterations of fluid and solid correspond to
  // each other and yield to a global residual which is smaller than the required tolerance


  /*----------------------------------------------------------------------*/
  // print converged/non-converged info
  /*----------------------------------------------------------------------*/
  if (converged())
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      Core::IO::cout << "-------------------------------------------------------Newton Converged ! "
                        "--------------------------------------------------"
                     << Core::IO::endl;
    }
    return true;
  }
  else if (iter_ >= itermax_)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      Core::IO::cout << "----------------------------------------- Newton not converged in ITEMAX "
                        "iterations ! --------------------------------------"
                     << Core::IO::endl;

      if (iter_outer_ < itermax_outer_)  // just in case that another restart will be performed!
      {
        Core::IO::cout
            << "- WARNING: increase the number nonlinear Newton-iterations, the additional "
               "restart does not help but solves the same system twice!!! -"
            << Core::IO::endl;
      }
    }
    return false;
  }

  return false;
}  // NewtonFull()



/*----------------------------------------------------------------------*/
// compute all norms used for convergence check
/*----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::build_convergence_norms()
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicXFEM::build_convergence_norms()");

  // build map extractors for velocity and pressure dofs
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> fluidvelpres;
  fluidvelpres.push_back(fluid_field()->velocity_row_map());
  fluidvelpres.push_back(fluid_field()->pressure_row_map());
  Core::LinAlg::MultiMapExtractor fluidvelpresextract(
      *(fluid_field()->dof_row_map()), fluidvelpres);


  //-------------------------------
  // build residual norms
  //-------------------------------

  // build full residual norms
  rhs_->norm_2(&normrhs_);

  // structural Dofs
  extractor_merged_poro().extract_vector(*rhs_, structp_block_)->norm_2(&normstrrhs_l2_);
  extractor_merged_poro().extract_vector(*rhs_, structp_block_)->norm_inf(&normstrrhs_inf_);

  // fluid velocity Dofs
  fluidvelpresextract.extract_vector(*extractor().extract_vector(*rhs_, fluid_block_), 0)
      ->norm_2(&normflvelrhs_l2_);
  fluidvelpresextract.extract_vector(*extractor().extract_vector(*rhs_, fluid_block_), 0)
      ->norm_inf(&normflvelrhs_inf_);

  // fluid pressure Dofs
  fluidvelpresextract.extract_vector(*extractor().extract_vector(*rhs_, fluid_block_), 1)
      ->norm_2(&normflpresrhs_l2_);
  fluidvelpresextract.extract_vector(*extractor().extract_vector(*rhs_, fluid_block_), 1)
      ->norm_inf(&normflpresrhs_inf_);

  if (structure_poro()->is_poro())
  {
    // porofluid Dofs
    extractor().extract_vector(*rhs_, fluidp_block_)->norm_2(&normpflvelrhs_l2_);
    extractor().extract_vector(*rhs_, fluidp_block_)->norm_inf(&normpflvelrhs_inf_);
  }


  //-------------------------------
  // build solution increment norms
  //-------------------------------

  // build full increment norm
  iterinc_->norm_2(&norminc_);

  // structural Dofs
  extractor_merged_poro().extract_vector(*iterinc_, structp_block_)->norm_2(&normstrinc_l2_);
  extractor_merged_poro().extract_vector(*iterinc_, structp_block_)->norm_inf(&normstrinc_inf_);
  extractor().extract_vector(*iterinc_, structp_block_)->norm_inf(&normstrincdisp_inf_);

  // fluid velocity Dofs
  fluidvelpresextract.extract_vector(*extractor().extract_vector(*iterinc_, fluid_block_), 0)
      ->norm_2(&normflvelinc_l2_);
  fluidvelpresextract.extract_vector(*extractor().extract_vector(*iterinc_, fluid_block_), 0)
      ->norm_inf(&normflvelinc_inf_);

  // fluid pressure Dofs
  fluidvelpresextract.extract_vector(*extractor().extract_vector(*iterinc_, fluid_block_), 1)
      ->norm_2(&normflpresinc_l2_);
  fluidvelpresextract.extract_vector(*extractor().extract_vector(*iterinc_, fluid_block_), 1)
      ->norm_inf(&normflpresinc_inf_);

  if (structure_poro()->is_poro())
  {
    // porofluid Dofs
    extractor().extract_vector(*iterinc_, fluidp_block_)->norm_2(&normpflvelinc_l2_);
    extractor().extract_vector(*iterinc_, fluidp_block_)->norm_inf(&normpflvelinc_inf_);
  }


  //-------------------------------
  // get length of the structural and fluid vector
  //-------------------------------
  ns_ = (*(extractor_merged_poro().extract_vector(*rhs_, structp_block_)))
            .global_length();                                                  // structure
  nf_ = (*(extractor().extract_vector(*rhs_, fluid_block_))).global_length();  // fluid
  nfv_ =
      (*(fluidvelpresextract.extract_vector(*extractor().extract_vector(*rhs_, fluid_block_), 0)))
          .global_length();  // fluid velocity
  nfp_ =
      (*(fluidvelpresextract.extract_vector(*extractor().extract_vector(*rhs_, fluid_block_), 1)))
          .global_length();         // fluid pressure
  nall_ = (*rhs_).global_length();  // all
}



/*----------------------------------------------------------------------*
 * update the global step-increment, evaluate the single fields with
 * x^n+1 with x^n+1 = x_n + stepinc and return if the fluid dofsets
 * between the two last iterations changed and
 * a Newton restart is necessary                            schott 08/14
 *----------------------------------------------------------------------*/
bool FSI::MonolithicXFEM::evaluate()
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicXFEM::Evaluate");


  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // Update the global step-increment from the last Newton-solve and extract the single field
  // step-increments
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------

  // Structure and fluid fields (evaluate-calls) expect the respective step-increment (x^n+1_i+1 -
  // x^n). So we add all of the increments together to build the step increment.
  //
  // The update of the latest step increment with iteration increments:
  // x^n+1_i+1 = x^n+1_i + iterinc with x the current step increment

  // step-increments for single fields:
  // sx contains the current step increment w.r.t. Pred(disp(t^n)) for the structure block
  // fx contains the current step increment w.r.t. t^n from the last Newton restart for the fluid
  // block
  std::shared_ptr<const Core::LinAlg::Vector<double>> sx;
  std::shared_ptr<const Core::LinAlg::Vector<double>> fx;
  std::shared_ptr<const Core::LinAlg::Vector<double>> ax;


  // update the whole step-increment vector
  // note: for iter_ = 1 the global x_sum_ is not available yet, then we take the single
  // step-increments
  if (iter_ > 1)
  {
    // update the step-increment
    x_sum_->update(1.0, *iterinc_, 1.0);

    // extract the single field step-increments from the global step-increment
    extract_field_vectors(x_sum_, sx, fx, ax);
  }
  else
  {
    sx = sx_sum_;                  // take the sx from before the restart
    fx = fx_sum_;                  // take the fx from before the restart
    if (have_ale()) ax = ax_sum_;  // take the ax from before the restart
  }

  sx_sum_ = sx;



  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // Call all fields evaluate method and assemble fields rhs vectors and matrices
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------

  //-------------------
  // structure field
  //-------------------
  {
    Core::Communication::barrier(get_comm());

    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    // Set Field State Section, here we should set the state with the step increments in all fields
    // (atm this is just done for ALE)
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    if (have_ale() && ax != nullptr)  // we should move this into the ALE Field!
    {
      Core::LinAlg::Vector<double> DispnpAle(*ale_field()->dof_row_map());
      DispnpAle.update(1.0, *ale_field()->interface()->insert_other_vector(*ax), 1.0,
          *ale_field()->dispn(), 0.0);  // update ale disp here...
      ale_field()->get_dbc_map_extractor()->insert_other_vector(
          *ale_field()->get_dbc_map_extractor()->extract_other_vector(DispnpAle),
          *ale_field()->write_access_dispnp());  // just update displacements which are not on dbc
                                                 // condition
    }

    // Set new state in StructurePoro
    if (sx == nullptr)
      sx = std::make_shared<Core::LinAlg::Vector<double>>(*structure_poro()->dof_row_map(), true);
    structure_poro()->update_state_incrementally(sx);
    if (have_contact_)
      structure_poro()->meshtying_contact_bridge()->get_strategy().set_state(
          Mortar::state_new_displacement, *structure_poro()->dispnp());
  }

  //--------------------------------------------------------
  // permute the fluid step-inc (ordered w.r.t. restart state) to current dofset-state
  // nothing has to be permuted for the first run as the following first evaluate call will fix the
  // reference dofset for this Newton loop nothing has to be permuted before we call the
  // fluid-evaluate the second time, since the second call we determine if dofsets have permuted the
  // first potentially valid permutation is set and available after the second fluid-evaluate call
  //--------------------------------------------------------

  std::shared_ptr<Core::LinAlg::Vector<double>> fx_permuted = nullptr;

  if (fx != nullptr)
  {
    fx_permuted = std::make_shared<Core::LinAlg::Vector<double>>(*fx);

    permute_fluid_dofs_forward(*fx_permuted);
  }



  // update fluid field increments
  fluid_field()->update_by_increments(fx_permuted);

  // StructurePoro()->structure_field()->writeGmshStructOutputStep();

  // ------------------------------------------------------------------
  // set the current interface displacement to the fluid field to be used in the cut
  // ------------------------------------------------------------------


  // update coupling objects and conditionmanager
  for (auto coupit = coup_man_.begin(); coupit != coup_man_.end(); ++coupit)
    coupit->second->set_coupling_states();

  // update ALE
  if (have_ale()) ale_field()->evaluate();


  //-------------------
  // fluid field
  //-------------------
  // * update ivelnp-vector with the step-increment
  //
  // PrepareSolve();
  // * cut at new interface position
  // * create new state-vectors and systemmatrix and
  // * perform time-integration to obtain a new reference solution of veln at t^n w.r.t. new
  // interface position
  // * perform a pseudo-time-integration to map the current iteration velnp (u^(n+1,i+1)) to new
  // interface position
  // * TODO: possibly perform fluid predictor
  // * update old-rhs and
  // * evaluate Neumann and DBCs
  //
  // Evaluate:
  // * call evaluate routine to assemble fluid rhs and systemmatrix
  //
  {
    Core::Communication::barrier(get_comm());

    // Teuchos::TimeMonitor::zeroOutTimers();

    // fluid field
    Teuchos::Time tf("fluid", true);

    // call the fluid evaluate with the current time-step-increment w.r.t. u^n from the restart:
    // Delta(u,p) = (u,p)^(n+1,i+1) - u^n
    // For the first call of a time-step, call Evaluate with a null-pointer
    // note: call the fluid with the permuted step-increment vector as the fluid-dofsets can permute
    // during the Newton whereas the x_sum_ has to preserve the order of dofs during the Newton

    // Specify if the CUT should be evaluated for this iteration

    if (cut_evaluate_dynamic_)
    {
      if (normstrincdisp_inf_ / std::min(nd_act_scaling_, nd_inc_scaling_) < cut_evaluate_mintol_ &&
          (iter_ > cut_evaluate_miniter_ || iter_outer_ > cut_evaluate_miniter_))
      {
        fluid_field()->set_evaluate_cut(false);

        if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        {
          Core::IO::cout << "==| Do not evaluate CUT for this iteration as disp_inc: "
                         << normstrincdisp_inf_ / std::min(nd_act_scaling_, nd_inc_scaling_)
                         << " < " << cut_evaluate_mintol_ << " |==" << Core::IO::endl;
        }
      }
      else
      {
        fluid_field()->set_evaluate_cut(true);
      }
    }

    fluid_field()->evaluate();

    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      Core::IO::cout << "fluid time : " << tf.totalElapsedTime(true) << Core::IO::endl;

    // Assign the Unphysical Boundary Elements to all procs (only for contact)
    if (have_contact_)
    {
      xf_c_comm_->fill_complete_sele_map();

      // We need these fluid state for the evaluation of contact ...
      fluid_field()->set_state_tim_int();
      if (fluid_field()->get_condition_manager()->get_mesh_coupling("XFEMSurfFSIMono") != nullptr)
        fluid_field()->get_condition_manager()->get_mesh_coupling("XFEMSurfFSIMono")->set_state();
      if (structure_poro()->is_poro() && fluid_field()->get_condition_manager()->get_mesh_coupling(
                                             "XFEMSurfFPIMono_ps_ps") != nullptr)
      {
        fluid_field()
            ->get_condition_manager()
            ->get_mesh_coupling("XFEMSurfFPIMono_ps_ps")
            ->set_state();
        fluid_field()
            ->get_condition_manager()
            ->get_mesh_coupling("XFEMSurfFPIMono_pf_ps")
            ->set_state();
        fluid_field()
            ->get_condition_manager()
            ->get_mesh_coupling("XFEMSurfFPIMono_ps_pf")
            ->set_state();
        fluid_field()
            ->get_condition_manager()
            ->get_mesh_coupling("XFEMSurfFPIMono_pf_pf")
            ->set_state();
      }
    }

    // structural field
    Teuchos::Time ts("structure", true);

    // Evaluate Structure (do not set state again)
    structure_poro()->evaluate(nullptr, iter_ == 1);

    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      Core::IO::cout << "structure time: " << ts.totalElapsedTime(true) << Core::IO::endl;
  }

  //--------------------------------------------------------
  // update permutation cycles how to permute fluid dofs during the Newton
  //--------------------------------------------------------

  // update the permutation map used for permuting fluid-dofs between Dofset after restarting the
  // Newton. Build a vector of cycles how to permute dofs between the reference dofset from
  // restarting the Newton and current dofset note: No permutation for the first call since
  // restarting the Newton - this is the reference dofset
  //       the first potentially valid permutation is set and available after the second
  //       fluid-evaluate call
  if (iter_ > 1)
  {
    update_permutation_map(*fluid_field()->get_permutation_map());
  }

  //-------------------
  // check for changing dofsets compared to the last Newton iteration to decide if the Newton has to
  // get restarted or continued
  //-------------------


  if (fluid_field()->newton_restart_monolithic()) return true;



  return false;  // continue with the setup of the new system and solving the system
}


/*----------------------------------------------------------------------*
 | check convergence of Newton iteration (public)          schott 08/14 |
 *----------------------------------------------------------------------*/
bool FSI::MonolithicXFEM::converged()
{
  // check for single norms (increment, residual)
  bool convinc = false;   // increment converged?
  bool convfres = false;  // residual converged?

  //---------------------------------------------
  // structural and fluid increments
  switch (normtypeinc_)
  {
    case Inpar::FSI::convnorm_abs:
      convinc = norminc_ < tolinc_;
      break;
    case Inpar::FSI::convnorm_rel:
      convinc =
          (((normstrinc_l2_ / ns_) < tol_dis_inc_l2_) and ((normstrinc_inf_) < tol_dis_inc_inf_) and
              ((normflvelinc_l2_ / nfv_) < tol_vel_inc_l2_) and
              ((normflvelinc_inf_) < tol_vel_inc_inf_) and
              ((normflpresinc_l2_ / nfp_) < tol_pre_inc_l2_) and
              ((normflpresinc_inf_) < tol_pre_inc_inf_));
      break;
    case Inpar::FSI::convnorm_mix:
      FOUR_C_THROW("not implemented!");
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual values!");
      break;
  }

  //---------------------------------------------
  // structural and fluid residual forces
  switch (normtypefres_)
  {
    case Inpar::FSI::convnorm_abs:
      convfres = normrhs_ < tolfres_;
      break;
    case Inpar::FSI::convnorm_rel:
      convfres =
          (((normstrrhs_l2_ / ns_) < tol_dis_res_l2_) and ((normstrrhs_inf_) < tol_dis_res_inf_) and
              ((normflvelrhs_l2_ / nfv_) < tol_vel_res_l2_) and
              ((normflvelrhs_inf_) < tol_vel_res_inf_) and
              ((normflpresrhs_l2_ / nfp_) < tol_pre_res_l2_) and
              ((normflpresrhs_inf_) < tol_pre_res_inf_));
      break;
    case Inpar::FSI::convnorm_mix:
      FOUR_C_THROW("not implemented!");
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual forces!");
      break;
  }

  //---------------------------------------------
  // combined increment + residual check?
  bool converged = false;

  if (combincfres_ == Inpar::FSI::bop_and)
  {
    converged = (convinc and convfres);
  }
  else
  {
    FOUR_C_THROW(
        "Just binary operator and for convergence check of Newton increment and residual "
        "supported!");
  }

  return converged;
}  // Converged()


/*----------------------------------------------------------------------
 | update the permutation map between using the recent permutations    |
 | between the last two Newton iterations                 schott 08/14 |
 ----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::update_permutation_map(
    std::map<int, int> permutation_map  /// permutation map between last two Newton iterations, by
                                        /// copy, do not call by reference
)
{
  // TODO: remove the counter and the screen output
  int count_updates = 0;
  int removed_permutations = 0;

  //--------------------------------
  // update the permutation map
  //--------------------------------

  // first, look if one of the already existing permutations have to be updated
  for (auto p = permutation_map_.begin(); p != permutation_map_.end(); p++)
  {
    // check if there is an update in the recent permutation map available
    auto p_recent_it = permutation_map.find(p->second);
    // update available
    if (p_recent_it != permutation_map.end())
    {
      // update the target of the permutation
      p->second = p_recent_it->second;
      // delete the update permutation, this has not to be considered again
      permutation_map.erase(p_recent_it);

      if (p->first == p->second)
      {
        // permutation became obsolete, remove it
        auto tmp_it = p;  // save the current iterator before removing and increase it
        tmp_it++;

        // erase the permutation as it is done
        permutation_map_.erase(p);
        removed_permutations++;

        p = tmp_it;
      }

      count_updates++;
    }
  }

  std::cout << " adapted permutations        " << count_updates << std::endl;
  std::cout << " removed permutations        " << removed_permutations << std::endl;
  std::cout << " new additional permutations " << permutation_map.size() << std::endl;

  // second, we have to add all new additional permutations
  permutation_map_.insert(permutation_map.begin(), permutation_map.end());

  //--------------------------------
  // build new permutation cycles based on the updated permutation map
  //--------------------------------

  // finally build the new permutation cycles used for permuting the dofs forward and backward
  build_fluid_permutation();
}


/*----------------------------------------------------------------------
 | build the new permutation cycles used for permuting the             |
 | fluid dofs forward and backward                        schott 08/14 |
 ----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::build_fluid_permutation()
{
  /// vector of permutation cycles, one cycle consists of a series of global dof ids (gids)
  permutation_.clear();

  // make a copy of the internal permutation_map which we will modify here
  std::map<int, int> permutation_map = permutation_map_;

  // the order in a permutation cycle describes how dofs w.r.t old interface position can be mapped
  // to new interface position by reading the cycle from beginning to the end


  // permutation cycle of ghost dofs (global ghost DOF ids (gids) w.r.t one node: a b c d)
  //      a                              b
  //      b           ---  P --->>       d       -----> (1, 3, 4, 2) (,1) = cycle
  //      c         <<---P^(-1)---       a
  //      d                              c
  //
  // old gids ---> permutation_map ---> new gids ( forward  permutation )
  // new gids <--- permutation_map <--- new gids ( backward permutation )

  // build permutation cycles from the permutation map
  // in the permutation map one-to-one relations between gids are stored (key = old gid, value = new
  // gid)
  for (auto map_it = permutation_map.begin(); map_it != permutation_map.end(); map_it++)
  {
    if (map_it->second == -1) continue;  // mapping already done by another cycle

    // create a new permutation cycle
    std::vector<int> new_cycle;

    int start_gid = map_it->first;

    // initialize the next-iterator for iterating the cycle to be created by the current start
    // iteration of the permutation map
    auto next_it = map_it;

    // create the cycle
    while (next_it != permutation_map.end())
    {
      // add the current stored gid to the cycle
      new_cycle.push_back(next_it->first);
      // the next stored gid in the cycle
      int next_gid = next_it->second;

      // mark the entry as done, do not consider this single permutation again
      next_it->second = -1;

      // we reached the end of the cycle, stop the new cycle here
      if (next_gid == start_gid) break;

      // jump to the next entry
      next_it = permutation_map.find(next_gid);
    }

    if ((int)new_cycle.size() > 2)
    {
      FOUR_C_THROW(
          "this is the first time that we permute more than two ghost dofsets! Check if the "
          "implementation works properly!");
    }

    // new cycle
    permutation_.push_back(new_cycle);
  }
}


/*----------------------------------------------------------------------
 | forward permutation of fluid dofs -                                 |
 | transform vectors (based on dofsets) w.r.t old interface position   |
 | forward to a vector (based on dofsets) w.r.t. new interface position|
 |                                                        schott 08/14 |
 ----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::permute_fluid_dofs_forward(Core::LinAlg::Vector<double>& fx)
{
  //---------------------------------
  // forward permutation of dofsets
  //---------------------------------
  // transform vectors (based on dofsets) w.r.t old interface position forward to a vector (based on
  // dofsets) w.r.t. new interface position

  // loop all permutation cycles
  for (auto i = permutation_.begin(); i != permutation_.end(); i++)
  {
    // permutation cycle of ghost dofs (global ghost DOF ids (gids) w.r.t one node: a b c d)
    //      a                              b
    //      b           ---  P --->>       d       -----> (1, 3, 4, 2) (,1) = cycle
    //      c                              a
    //      d                              c
    //
    // old gids ---> permutation_map ---> new gids ( forward  permutation )

    std::vector<int>& p_cycle = *i;

    double tmp_value = 0.0;

    // first  value -- to -- second position
    // second value -- to -- third  position
    // ...
    // last   value -- to -- first position
    for (auto key = p_cycle.begin(); key != p_cycle.end(); key++)
    {
      if (key + 1 != p_cycle.end())  // standard during the cycle
      {
        tmp_value = fx.local_values_as_span()[fx.get_map().lid(
            *(key + 1))];  // save the value before it will be overwritten
        (fx).get_values()[fx.get_map().lid(*(key + 1))] =
            fx.local_values_as_span()[fx.get_map().lid(
                *(key))];  // set current value to next position
        // std::cout << "copy value from gid " << *(key) << " to " << *(key+1) << std::endl;
      }
      else  // last value in cycle reached
      {
        (fx).get_values()[fx.get_map().lid(*p_cycle.begin())] = tmp_value;
        // std::cout << "copy value from tmp to " << *p_cycle.begin() << std::endl;
      }
    }
  }
}


/*----------------------------------------------------------------------
 | backward permutation of fluid dofs -                                |
 | transform vectors (based on dofsets) w.r.t new interface position   |
 | backward to a vector (based on dofsets) w.r.t. new interface        |
 | position                                               schott 08/14 |
 ----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::permute_fluid_dofs_backward(Core::LinAlg::Vector<double>& fx)
{
  //---------------------------------
  // backward permutation of dofsets
  //---------------------------------
  // transform vectors (based on dofsets) w.r.t new interface position backward to a vector (based
  // on dofsets) w.r.t. old interface position


  // loop all permutation cycles
  for (auto i = permutation_.begin(); i != permutation_.end(); i++)
  {
    // permutation cycle of ghost dofs (global ghost DOF ids (gids) w.r.t one node: a b c d)
    //      a                              b
    //      b                              d       -----> (1, 3, 4, 2) (,1) = cycle
    //      c         <<---P^(-1)---       a
    //      d                              c
    //
    // new gids <--- permutation_map <--- new gids ( backward permutation )

    std::vector<int>& p_cycle = *i;

    double tmp_value = 0.0;

    //  last    value -- to -- (last-1) position
    // (last-1) value -- to -- (last-2) position
    // ...
    //  first   value -- to -- last position
    for (auto key_reverse = p_cycle.end() - 1; (key_reverse + 1) != p_cycle.begin(); key_reverse--)
    {
      if (key_reverse != p_cycle.begin())  // standard during the cycle
      {
        tmp_value = fx.local_values_as_span()[fx.get_map().lid(
            *(key_reverse - 1))];  // save the value before it will be overwritten
        (fx).get_values()[fx.get_map().lid(*(key_reverse - 1))] =
            fx.local_values_as_span()[fx.get_map().lid(
                *(key_reverse))];  // set current value to position before
        // std::cout << "copy value from gid " << *(key_reverse) << " to " << *(key_reverse-1) <<
        // std::endl;
      }
      else
      {
        (fx).get_values()[fx.get_map().lid(*(p_cycle.end() - 1))] = tmp_value;
        // std::cout << "copy value from tmp to " << *(p_cycle.end()-1) << std::endl;
      }
    }
  }
}



/*----------------------------------------------------------------------*
 | create linear solver                            schott/wiesner 10/14 |
 *----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::create_linear_solver()
{
  // get the solver number used for linear XFSI solver
  //  const int linsolvernumber = fsidyn_.get<int>("LINEAR_SOLVER");
  // TODO: get via input file, no LINEAR_SOLVER in FSI-Dynamic so far...
  const int linsolvernumber = 1;
  // check if the XFSI solver has a valid solver number
  if (linsolvernumber == (-1))
  {
    FOUR_C_THROW(
        "no linear solver defined for monolithic XFSI. Please set LINEAR_SOLVER in XFSI DYNAMIC to "
        "a valid number!");
  }

  // get solver parameter list of linear XFSI solver
  const Teuchos::ParameterList& xfsisolverparams =
      Global::Problem::instance()->solver_params(linsolvernumber);

  // safety check if the hard-coded solver number is the XFSI-solver
  if (xfsisolverparams.get<std::string>("NAME") != "XFSI_SOLVER")
    FOUR_C_THROW("check whether solver with number 1 is the XFSI_SOLVER and has this name!");


  const auto solvertype =
      Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(xfsisolverparams, "SOLVER");

  //----------------------------------------------
  // create direct solver for merged block matrix
  //----------------------------------------------
  if (solvertype == Core::LinearSolver::SolverType::umfpack ||
      solvertype == Core::LinearSolver::SolverType::superlu)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      std::cout << "Merged XFSI block matrix is used!\n" << std::endl;

    merge_fsi_blockmatrix_ = true;

    Teuchos::ParameterList solverparams;
    Core::Utils::add_enum_class_to_parameter_list<Core::LinearSolver::SolverType>("SOLVER",
        Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(xfsisolverparams, "SOLVER"),
        solverparams);

    solver_ = std::make_shared<Core::LinAlg::Solver>(solverparams, get_comm(),
        Global::Problem::instance()->solver_params_callback(),
        Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
            Global::Problem::instance()->io_params(), "VERBOSITY"));

    return;
  }

  //----------------------------------------------
  // create iterative solver for XFSI block matrix
  //----------------------------------------------

  if (solvertype != Core::LinearSolver::SolverType::belos)
    FOUR_C_THROW("Iterative solver expected");

  // get parameter list of structural dynamics
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
  // use solver blocks for structure
  // get the solver number used for structural solver
  const int slinsolvernumber = sdyn.get<int>("LINEAR_SOLVER");
  // check if the structural solver has a valid solver number
  if (slinsolvernumber == (-1))
  {
    FOUR_C_THROW(
        "no linear solver defined for structural field. Please set LINEAR_SOLVER in STRUCTURAL "
        "DYNAMIC to a valid number!");
  }

  // get parameter list of fluid dynamics
  const Teuchos::ParameterList& fdyn = Global::Problem::instance()->fluid_dynamic_params();
  // use solver blocks for temperature (thermal field)
  // get the solver number used for thermal solver
  const int flinsolvernumber = fdyn.get<int>("LINEAR_SOLVER");
  // check if the fluid solver has a valid solver number
  if (flinsolvernumber == (-1))
  {
    FOUR_C_THROW(
        "no linear solver defined for fluid field. Please set LINEAR_SOLVER in FLUID DYNAMIC to a "
        "valid number!");
  }

  int alinsolvernumber = -1;
  if (have_ale())
  {
    // get parameter list of ale dynamics
    const Teuchos::ParameterList& adyn = Global::Problem::instance()->ale_dynamic_params();
    alinsolvernumber = adyn.get<int>("LINEAR_SOLVER");
    // check if the ale solver has a valid solver number
    if (alinsolvernumber == (-1))
    {
      FOUR_C_THROW(
          "no linear solver defined for ale field. Please set LINEAR_SOLVER in ALE DYNAMIC to a "
          "valid number!");
    }
  }


  const auto azprectype =
      Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(xfsisolverparams, "AZPREC");

  // plausibility check
  switch (azprectype)
  {
    case Core::LinearSolver::PreconditionerType::block_teko:
      break;
    case Core::LinearSolver::PreconditionerType::multigrid_muelu:
    {
      // no plausibility checks here
      // if you forget to declare an xml file you will get an error message anyway
    }
    break;
    default:
      FOUR_C_THROW(
          "Block Gauss-Seidel BGS2x2 preconditioner expected. Alternatively you can define your "
          "own AMG block preconditioner (using an xml file). This is experimental.");
      break;
  }


  // prepare linear solvers and preconditioners
  switch (azprectype)
  {
    case Core::LinearSolver::PreconditionerType::block_teko:
    {
      // This should be the default case (well-tested and used)
      solver_ = std::make_shared<Core::LinAlg::Solver>(xfsisolverparams,
          // ggfs. explicit Comm von STR wie lungscatra
          get_comm(), Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));

      // use solver blocks for structure and fluid
      const Teuchos::ParameterList& ssolverparams =
          Global::Problem::instance()->solver_params(slinsolvernumber);
      const Teuchos::ParameterList& fsolverparams =
          Global::Problem::instance()->solver_params(flinsolvernumber);

      solver_->put_solver_params_to_sub_params("Inverse1", ssolverparams,
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"),
          get_comm());
      Core::LinearSolver::Parameters::compute_solver_parameters(
          *structure_poro()->discretization(), solver_->params().sublist("Inverse1"));

      solver_->put_solver_params_to_sub_params("Inverse2", fsolverparams,
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"),
          get_comm());
      Core::LinearSolver::Parameters::compute_solver_parameters(
          *fluid_field()->discretization(), solver_->params().sublist("Inverse2"));

      if (structure_poro()->is_poro())
      {
        solver_->put_solver_params_to_sub_params("Inverse3", fsolverparams,
            Global::Problem::instance()->solver_params_callback(),
            Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
                Global::Problem::instance()->io_params(), "VERBOSITY"),
            get_comm());
        Core::LinearSolver::Parameters::compute_solver_parameters(
            *structure_poro()->fluid_field()->discretization(),
            solver_->params().sublist("Inverse3"));
      }
      if (have_ale())
      {
        const Teuchos::ParameterList& asolverparams =
            Global::Problem::instance()->solver_params(alinsolvernumber);
        if (ale_i_block_ == 3)
        {
          solver_->put_solver_params_to_sub_params("Inverse3", asolverparams,
              Global::Problem::instance()->solver_params_callback(),
              Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
                  Global::Problem::instance()->io_params(), "VERBOSITY"),
              get_comm());
          Core::LinearSolver::Parameters::compute_solver_parameters(
              *ale_field()->write_access_discretization(), solver_->params().sublist("Inverse3"));
        }
        else if (ale_i_block_ == 4)
        {
          solver_->put_solver_params_to_sub_params("Inverse4", asolverparams,
              Global::Problem::instance()->solver_params_callback(),
              Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
                  Global::Problem::instance()->io_params(), "VERBOSITY"),
              get_comm());
          Core::LinearSolver::Parameters::compute_solver_parameters(
              *ale_field()->write_access_discretization(), solver_->params().sublist("Inverse4"));
        }
        else
        {
          FOUR_C_THROW("You have more than 4 Fields? --> add another Inverse 5 here!");
        }
      }
      break;
    }
    default:
      FOUR_C_THROW("Block preconditioner expected.");
      break;
  }
}  // create_linear_solver()


/*----------------------------------------------------------------------*
 | solve linear FSI system                                 schott 07/13 |
 *----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::linear_solve()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << " FSI::MonolithicXFEM::linear_solve()" << std::endl;

  // Solve for inc_ = [disi_,tempi_]
  // Solve K_Teffdyn . IncX = -R  ===>  IncX_{n+1} with X=[d,(u,p)]
  // \f$x_{i+1} = x_i + \Delta x_i\f$

  // apply Dirichlet BCs to system of equations
  iterinc_->put_scalar(0.0);  // Useful? depends on solver and more

  // default: use block matrix
  if (merge_fsi_blockmatrix_ == false)
  {
    // adapt solver tolerance
    Core::LinAlg::SolverParams solver_params;
    if (solveradapttol_ and (iter_ > 1))
    {
      solver_params.nonlin_tolerance = tolrhs_;
      solver_params.nonlin_residual = normrhs_;
      solver_params.lin_tol_better = solveradaptolbetter_;
    }

    // Infnormscaling: scale system before solving
    scale_system(*systemmatrix_, *rhs_);

    compute_null_space_if_necessary(
        *fluid_field()->discretization(), solver_->params().sublist("Inverse2"), true);

    // solve the problem, work is done here!
    solver_params.refactor = true;
    solver_params.reset = iter_ == 1;
    solver_->solve(systemmatrix_, iterinc_, rhs_, solver_params);

    // Infnormscaling: unscale system after solving
    unscale_solution(*systemmatrix_, *iterinc_, *rhs_);


    // Adapt solver tolerance
    // TODO: does or how does this work for changing Newton systems
    solver_->reset_tolerance();

  }  // use block matrix
  else  // (merge_fsi_blockmatrix_ == true)
  {
    if (scaling_infnorm_)
      FOUR_C_THROW("infnorm-scaling of FSI-system not supported for direct solver");

    //------------------------------------------
    // merge blockmatrix to SparseMatrix and solve
    std::shared_ptr<Core::LinAlg::SparseMatrix> sparse = systemmatrix_->merge();

    //------------------------------------------
    // standard solver call
    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = iter_ == 1;
    solver_->solve(sparse, iterinc_, rhs_, solver_params);
  }  // MergeBlockMatrix

  apply_newton_damping();

  // TODO: can we do this?!
  // reset the solver (frees the pointer to the Core::LinAlg:: matrix and vectors
  // also!) std::cout << "reset the solver" << std::endl;
  solver_->reset();

  //---------------------------------------------
  // permute the increment and rhs vector back to the reference configuration w.r.t which iterinc_
  // and rhs are defined

  std::shared_ptr<Core::LinAlg::Vector<double>> f_iterinc_permuted =
      extractor().extract_vector(*iterinc_, 1);
  permute_fluid_dofs_backward(*f_iterinc_permuted);
  extractor().insert_vector(*f_iterinc_permuted, 1, *iterinc_);


  std::shared_ptr<Core::LinAlg::Vector<double>> f_rhs_permuted =
      extractor().extract_vector(*rhs_, 1);
  permute_fluid_dofs_backward(*f_rhs_permuted);
  extractor().insert_vector(*f_rhs_permuted, 1, *rhs_);

  //---------------------------------------------

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << " Solved" << std::endl;
  }

}  // linear_solve()


/*----------------------------------------------------------------------*/
// apply infnorm scaling to linear block system            schott 10/14 |
/*----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::scale_system(
    Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& b)
{
  if (scaling_infnorm_)
  {
    if (num_fields_ > 2) FOUR_C_THROW("InfNorm Scaling just implemented for 2x2 Block!");
    // The matrices are modified here. Do we have to change them back later on?

    Core::LinAlg::SparseMatrix& A = mat.matrix(0, 0);
    srowsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A.row_map(), false);
    scolsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A.row_map(), false);
    A.inv_row_sums(*srowsum_);
    A.inv_col_sums(*scolsum_);

    A.left_scale(*srowsum_);
    A.right_scale(*scolsum_);
    mat.matrix(0, 1).left_scale(*srowsum_);
    mat.matrix(1, 0).right_scale(*scolsum_);

    std::shared_ptr<Core::LinAlg::Vector<double>> sx = extractor().extract_vector(b, 0);

    sx->multiply(1.0, *srowsum_, *sx, 0.0);

    extractor().insert_vector(*sx, 0, b);
  }
}



/*----------------------------------------------------------------------*/
// undo infnorm scaling from scaled solution               schott 10/14 |
/*----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::unscale_solution(Core::LinAlg::BlockSparseMatrixBase& mat,
    Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& b)
{
  if (scaling_infnorm_)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> sy = extractor().extract_vector(x, 0);

    sy->multiply(1.0, *scolsum_, *sy, 0.0);

    extractor().insert_vector(*sy, 0, x);

    std::shared_ptr<Core::LinAlg::Vector<double>> sx = extractor().extract_vector(b, 0);

    sx->reciprocal_multiply(1.0, *srowsum_, *sx, 0.0);

    extractor().insert_vector(*sx, 0, b);

    Core::LinAlg::SparseMatrix& A = mat.matrix(0, 0);
    srowsum_->reciprocal(*srowsum_);
    scolsum_->reciprocal(*scolsum_);
    A.left_scale(*srowsum_);
    A.right_scale(*scolsum_);
    mat.matrix(0, 1).left_scale(*srowsum_);
    mat.matrix(1, 0).right_scale(*scolsum_);
  }
}



/*----------------------------------------------------------------------*
 | create combined Dirichlet boundary condition map,                    |
 | map containing the dofs with Dirichlet BC                            |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> FSI::MonolithicXFEM::combined_dbc_map()
{
  std::shared_ptr<const Core::LinAlg::Map> scondmap = structure_poro()->combined_dbc_map();
  const std::shared_ptr<const Core::LinAlg::Map> fcondmap =
      fluid_field()->get_dbc_map_extractor()->cond_map();

  std::shared_ptr<Core::LinAlg::Map> condmap = Core::LinAlg::merge_map(scondmap, fcondmap, false);

  return condmap;
}


/*----------------------------------------------------------------------*/
/*  print Newton-Raphson iteration to screen and error file             */
/*----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::print_newton_iter()
{
  // print to standard out
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    if (iter_ == 1) print_newton_iter_header();

    print_newton_iter_text();
  }
}


/*----------------------------------------------------------------------*/
/* print Newton-Raphson iteration to screen and error file              */
/*----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::print_newton_iter_header()
{
  Core::IO::cout << "CONVTOL: " << tolfres_ << Core::IO::endl;

  Core::IO::cout
      << "===================================================================================="
         "========================================="
      << Core::IO::endl;

  // enter converged state etc
  Core::IO::cout << "|outerit";
  Core::IO::cout << "|  nit  |";

  // different style due relative or absolute error checking
  // displacement
  switch (normtypefres_)
  {
    case Inpar::FSI::convnorm_abs:
      Core::IO::cout << "            "
                     << "abs-res-norm  |";
      break;
    case Inpar::FSI::convnorm_rel:
      Core::IO::cout << "str-rs-l2|"
                     << "flv-rs-l2|"
                     << "flp-rs-l2|";
      Core::IO::cout << "str-rs-li|"
                     << "flv-rs-li|"
                     << "flp-rs-li|";
      break;
    case Inpar::FSI::convnorm_mix:
      FOUR_C_THROW("not implemented");
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypeinc_)
  {
    case Inpar::FSI::convnorm_abs:
      Core::IO::cout << "                  "
                     << "abs-inc-norm";
      break;
    case Inpar::FSI::convnorm_rel:
      Core::IO::cout << "str-in-l2|"
                     << "flv-in-l2|"
                     << "flp-in-l2|";
      Core::IO::cout << "str-in-li|"
                     << "flv-in-li|"
                     << "flp-in-li|";
      break;
    case Inpar::FSI::convnorm_mix:
      FOUR_C_THROW("not implemented");
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  // add solution time
  Core::IO::cout << Core::IO::endl;
  Core::IO::cout
      << "===================================================================================="
         "========================================="
      << Core::IO::endl;
}

/*---------------------------------------------------------------------*/
/*  print Newton-Raphson iteration to screen                           */
/*---------------------------------------------------------------------*/
void FSI::MonolithicXFEM::print_newton_iter_text()
{
  // enter converged state etc
  //   if (myrank_ == 0)
  //   {
  //     if (itnum>0)
  //     {
  //       printf("|  %3d/%3d   | %10.3E[L_2 ]  | %10.3E   | %10.3E   | %10.3E   | %10.3E   |",
  //         itnum,itmax,ittol,vresnorm_,presnorm_,incvelnorm_L2_/velnorm_L2_,
  //         incprenorm_L2_/prenorm_L2_);

  // TODO: komplette Ueberarbeitung von Rel vs Abs notwendig!!! siehe abs vs rel z.B. in Fluid-Code


  Core::IO::cout << " " << iter_outer_ << "/" << itermax_outer_;
  Core::IO::cout << " " << iter_ << "/" << itermax_;

  // different style due relative or absolute error checking
  // displacement
  switch (normtypefres_)
  {
    case Inpar::FSI::convnorm_abs:
      Core::IO::cout << "             " << (normrhs_) << Core::IO::endl;
      break;
    case Inpar::FSI::convnorm_rel:
      Core::IO::cout << "|" << (normstrrhs_l2_ / ns_) << "|" << (normflvelrhs_l2_ / nfv_) << "|"
                     << (normflpresrhs_l2_ / nfp_) << "|" << (normstrrhs_inf_) << "|"
                     << (normflvelrhs_inf_) << "|" << (normflpresrhs_inf_);
      break;
    case Inpar::FSI::convnorm_mix:
      FOUR_C_THROW("not implemented!");
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypeinc_)
  {
    case Inpar::FSI::convnorm_abs:
      Core::IO::cout << "             " << (norminc_) << Core::IO::endl;
      break;
    case Inpar::FSI::convnorm_rel:
      Core::IO::cout << "|" << (normstrinc_l2_ / ns_) << "|" << (normflvelinc_l2_ / nfv_) << "|"
                     << (normflpresinc_l2_ / nfp_) << "|" << (normstrinc_inf_) << "|"
                     << (normflvelinc_inf_) << "|" << (normflpresinc_inf_) << "|" << Core::IO::endl;
      break;
    case Inpar::FSI::convnorm_mix:
      FOUR_C_THROW("not implemented!");
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }
}



/*----------------------------------------------------------------------*/
// read restart data for monolithic XFSI system
/*----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::read_restart(int step)
{
  //--------------------------------
  // read structural field
  structure_poro()->read_restart(step);

  //--------------------------------
  // read ale field
  if (have_ale()) ale_field()->read_restart(step);

  //--------------------------------
  // read fluid field
  // set the current interface displacement to the fluid field to be used in the cut
  // (as we just loaded the displacements in the structure this has to be done again here)
  for (auto coupit = coup_man_.begin(); coupit != coup_man_.end(); ++coupit)
    coupit->second->init_coupling_states();

  // cut to get the correct dofsets (with restart displacements)
  // fluid_field()->CreateInitialState();
  fluid_field()->read_restart(step);



  //--------------------------------
  // setup a new system as dofrowmaps could have been changed!
  setup_system();

  if (structure_poro()->is_poro()) structure_poro()->poro_field()->setup_system();

  //--------------------------------
  // NOTE: do the following after structure_field()->read_restart and after
  // fluid_field()->read_restart as read_mesh can change the discretization and the dofrowmaps!!!

  // read Lagrange multiplier (ie forces onto the structure, Robin-type forces
  // consisting of fluid forces and the Nitsche penalty term contribution)
  Core::IO::DiscretizationReader reader = Core::IO::DiscretizationReader(
      *structure_poro()->discretization(), Global::Problem::instance()->input_control_file(), step);
  for (auto coupit = coup_man_.begin(); coupit != coup_man_.end(); ++coupit)
    coupit->second->read_restart(reader);
  //

  set_time_step(fluid_field()->time(), fluid_field()->step());
}

/*----------------------------------------------------------------------*/
// If activated damp actual Newton increment
/*----------------------------------------------------------------------*/
void FSI::MonolithicXFEM::apply_newton_damping()
{
  if (!nd_newton_damping_) return;

  // 1 // compute damping based on residual comparison
  // get normrhs on all levels
  for (int level = nd_levels_ - 1; level > 0; --level)
    nd_normrhs_old_[level] = nd_normrhs_old_[level - 1];
  nd_normrhs_old_[0] = normrhs_;
  rhs_->norm_2(&normrhs_);
  bool scaleup = false;
  bool scaledown = false;
  if (iter_ == 1 && iter_outer_ == 1)
    nd_act_scaling_ = nd_maxscaling_;
  else if (nd_normrhs_old_[0] < normrhs_ && (iter_ > 1 || iter_outer_ > 1))
    scaledown = true;
  else
  {
    scaleup = true;
    for (int level = 1; level < nd_levels_; ++level)
    {
      if (nd_normrhs_old_[level] < nd_normrhs_old_[level - 1] &&
          (iter_ > level + 1 || iter_outer_ > level + 1))
      {
        if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        {
          std::cout << "==| Skip rescaling level " << level + 1 << " |==" << std::endl;
        }
        scaleup = false;
        break;
      }
    }
  }

  if (scaledown)
    nd_act_scaling_ *= nd_reduction_fac_;
  else if (scaleup && (nd_act_scaling_ < nd_increase_fac_))
    nd_act_scaling_ /= nd_increase_fac_;
  else if (scaleup)
    nd_act_scaling_ = nd_maxscaling_;

  // 2 // compute damping based on maximal increment value
  nd_inc_scaling_ = 1.0;

  if (nd_newton_incmax_damping_)
  {
    std::array<double, 5> incnorm;
    incnorm.fill(-2.0);  // disp, vel, p , porovel, porop
    if (!structure_poro()->is_poro())
    {
      if (nd_max_incnorm_[0] > 0)
        extractor().extract_vector(*iterinc_, structp_block_)->norm_inf(incnorm.data());
    }
    else if (nd_max_incnorm_[0] > 0 || nd_max_incnorm_[3] > 0 || nd_max_incnorm_[4] > 0)
    {
      // build map extractors for velocity and pressure dofs
      std::vector<std::shared_ptr<const Core::LinAlg::Map>> fluidvelpres;
      fluidvelpres.push_back(structure_poro()->fluid_field()->velocity_row_map());
      fluidvelpres.push_back(structure_poro()->fluid_field()->pressure_row_map());
      Core::LinAlg::MultiMapExtractor fluidvelpresextract(
          *(structure_poro()->fluid_field()->dof_row_map()), fluidvelpres);
      extractor().extract_vector(*iterinc_, structp_block_)->norm_inf(incnorm.data());
      fluidvelpresextract.extract_vector(*extractor().extract_vector(*iterinc_, fluidp_block_), 0)
          ->norm_inf(&incnorm[3]);
      fluidvelpresextract.extract_vector(*extractor().extract_vector(*iterinc_, fluidp_block_), 1)
          ->norm_inf(&incnorm[4]);
    }
    if (nd_max_incnorm_[1] > 0 || nd_max_incnorm_[2] > 0)
    {
      // build map extractors for velocity and pressure dofs
      std::vector<std::shared_ptr<const Core::LinAlg::Map>> fluidvelpres;
      fluidvelpres.push_back(fluid_field()->velocity_row_map());
      fluidvelpres.push_back(fluid_field()->pressure_row_map());
      Core::LinAlg::MultiMapExtractor fluidvelpresextract(
          *(fluid_field()->dof_row_map()), fluidvelpres);
      fluidvelpresextract.extract_vector(*extractor().extract_vector(*iterinc_, fluid_block_), 0)
          ->norm_inf(&incnorm[1]);  // fluid velocity Dofs
      fluidvelpresextract.extract_vector(*extractor().extract_vector(*iterinc_, fluid_block_), 1)
          ->norm_inf(&incnorm[2]);  // fluid pressure Dofs
    }
    for (int field = 0; field < 5; ++field)
    {
      if (incnorm[field] > nd_max_incnorm_[field] && nd_max_incnorm_[field] > 0)
        if (nd_max_incnorm_[field] / incnorm[field] < nd_inc_scaling_)
          nd_inc_scaling_ = nd_max_incnorm_[field] / incnorm[field];
    }
  }

  if (nd_act_scaling_ > nd_inc_scaling_)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      std::cout << "==| Incremental Based Damping of Newton Scheme with scaling " << nd_inc_scaling_
                << "! |==" << std::endl;
    }
    iterinc_->scale(nd_inc_scaling_);
  }
  else if (nd_act_scaling_ < 1)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      std::cout << "==| Residual Based Damping of Newton Scheme with scaling " << nd_act_scaling_
                << "! |==" << std::endl;
    }
    iterinc_->scale(nd_act_scaling_);
  }
}

FOUR_C_NAMESPACE_CLOSE

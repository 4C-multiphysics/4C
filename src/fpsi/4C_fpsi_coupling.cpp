// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fpsi_coupling.hpp"

#include "4C_adapter_ale_fpsi.hpp"
#include "4C_adapter_fld_fluid.hpp"
#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_condition_selector.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fpsi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fpsi.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_poroelast_monolithic.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FPSI::FpsiCoupling::FpsiCoupling(std::shared_ptr<PoroElast::Monolithic> poro,
    std::shared_ptr<Adapter::Fluid> fluid, std::shared_ptr<Adapter::AleFpsiWrapper> ale,
    std::shared_ptr<std::map<int, int>> Fluid_PoroFluid_InterfaceMap,
    std::shared_ptr<std::map<int, int>> PoroFluid_Fluid_InterfaceMap)
    : poro_(poro),
      fluid_(fluid),
      ale_(ale),
      fluidvelpres_extractor_(nullptr),
      fluidvel_extractor_(nullptr),
      porofluid_extractor_(nullptr),
      porostruct_extractor_(nullptr),
      poro_extractor_(nullptr),
      fluid_fsifpsi_extractor_(nullptr),
      isfirstcall_(true),
      fluid_poro_fluid_interface_map_(Fluid_PoroFluid_InterfaceMap),  // to be removed later
      poro_fluid_fluid_interface_map_(PoroFluid_Fluid_InterfaceMap),
      conductivity_(0.0)
{
  setup_interface_coupling();
  init_coupling_matrixes_rhs();
  re_init_coupling_matrix_transform();
  return;
}
/*----------------------------------------------------------------------/
| Initialize Coupling Matrixes and Coupling RHS              ager 12/14 |
/----------------------------------------------------------------------*/
void FPSI::FpsiCoupling::init_coupling_matrixes_rhs()
{
  // fluid extractor
  Core::LinAlg::MapExtractor fluidextractor(*fluid_->dof_row_map(), fluid_->dof_row_map(), false);
  // ale extractor
  Core::LinAlg::MapExtractor aleextractor(
      *(ale_->interface()->other_map()), ale_->interface()->other_map(), false);

  c_pp_ = std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>(
      new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *poro_->extractor(), *poro_->extractor(), 81, true, true));
  c_ff_ = std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>(
      new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *fluid_->interface(), *fluid_->interface(), 81, true, true));
  c_pf_ = std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>(
      new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          fluidextractor, *poro_->extractor(), 81, true, true));
  c_fp_ = std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>(
      new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *poro_->extractor(), fluidextractor, 81, true, true));
  c_pa_ = std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>(
      new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          aleextractor, *poro_->extractor(), 81, true, true));
  c_fa_ = std::make_shared<Core::LinAlg::SparseMatrix>(*fluid_->dof_row_map(), 81, true, true);

  c_rhs_s_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *poro_->structure_field()->dof_row_map(), true);
  c_rhs_pf_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*poro_->fluid_field()->dof_row_map(), true);
  c_rhs_f_ = std::make_shared<Core::LinAlg::Vector<double>>(*fluid_->dof_row_map(), true);

  return;
}

/*----------------------------------------------------------------------/
| Setup the Coupling Object                                  ager 12/14 |
/----------------------------------------------------------------------*/
void FPSI::FpsiCoupling::setup_interface_coupling()
{
  const int ndim = Global::Problem::instance()->n_dim();

  std::shared_ptr<Core::FE::Discretization> fluiddis = fluid_field()->discretization();
  std::shared_ptr<Core::FE::Discretization> porofluiddis =
      poro_field()->fluid_field()->discretization();
  std::shared_ptr<Core::FE::Discretization> porostructdis =
      poro_field()->structure_field()->discretization();

  {
    porofluid_extractor_ = std::make_shared<Core::LinAlg::MapExtractor>();
    Core::Conditions::setup_extractor(*porofluiddis, *porofluid_extractor_,
        {Core::Conditions::Selector("fpsi_coupling", 0, ndim + 1)});
  }

  {
    porostruct_extractor_ = std::make_shared<Core::LinAlg::MapExtractor>();
    Core::Conditions::setup_extractor(*porostructdis, *porostruct_extractor_,
        {Core::Conditions::Selector("fpsi_coupling", 0, ndim)});
  }

  {
    fluidvelpres_extractor_ = std::make_shared<Core::LinAlg::MapExtractor>();
    Core::Conditions::setup_extractor(*fluiddis, *fluidvelpres_extractor_,
        {Core::Conditions::Selector("fpsi_coupling", 0, ndim + 1)});
  }

  {
    fluidvel_extractor_ = std::make_shared<Core::LinAlg::MapExtractor>();
    Core::Conditions::setup_extractor(
        *fluiddis, *fluidvel_extractor_, {Core::Conditions::Selector("fpsi_coupling", 0, ndim)});
  }

  {
    fluid_fsifpsi_extractor_ = std::make_shared<FPSI::Utils::MapExtractor>();
    Core::Conditions::setup_extractor(*fluiddis, *fluid_fsifpsi_extractor_,
        {
            Core::Conditions::Selector("FSICoupling", 0, ndim),
            Core::Conditions::Selector("fpsi_coupling", 0, ndim),
        });
  }

  {
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;

    // Split poro_field into:
    //                      --> Structure (inside + FSI-Interface)
    //                      --> Structure FPSI-Interface
    //                      --> PoroFluid (inside + FSI-Interface)
    //                      --> PoroFluid FPSI-Interface

    std::shared_ptr<const Core::LinAlg::Map> s_other_map = Core::LinAlg::merge_map(
        poro_field()->structure_field()->interface()->map(Solid::MapExtractor::cond_other),
        poro_field()->structure_field()->interface()->map(Solid::MapExtractor::cond_fsi));
    vecSpaces.push_back(s_other_map);  // other map
    vecSpaces.push_back(poro_field()->structure_field()->interface()->map(
        Solid::MapExtractor::cond_fpsi));                    // fpsi_coupling
    vecSpaces.push_back(porofluid_extractor_->other_map());  // other map
    vecSpaces.push_back(porofluid_extractor_->cond_map());   // fpsi_coupling

    std::shared_ptr<Core::LinAlg::Map> fullmap =
        Core::LinAlg::MultiMapExtractor::merge_maps(vecSpaces);
    // full Poroelasticity-blockmap
    poro_extractor_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
    poro_extractor_->setup(*fullmap, vecSpaces);
  }

  // porous fluid to fluid
  icoup_pf_f_ = std::make_shared<Coupling::Adapter::Coupling>();
  icoup_pf_f_->setup_condition_coupling(*porofluiddis, porofluid_extractor_->cond_map(), *fluiddis,
      fluidvelpres_extractor_->cond_map(), "fpsi_coupling", ndim + 1, false);

  // porous structure to fluid
  icoup_ps_f_ = std::make_shared<Coupling::Adapter::Coupling>();
  icoup_ps_f_->setup_condition_coupling(*porostructdis, porostruct_extractor_->cond_map(),
      *fluiddis, fluidvel_extractor_->cond_map(), "fpsi_coupling", ndim, false);

  // porous structure to ale
  icoup_ps_a_ = std::make_shared<Coupling::Adapter::Coupling>();
  icoup_ps_a_->setup_condition_coupling(*porostructdis, porostruct_extractor_->cond_map(),
      *ale_field()->discretization(), ale_field()->interface()->fpsi_cond_map(), "fpsi_coupling",
      ndim, false);

  return;
}

/*-----------------------------------------------------------------------/
| Method reinitializes the matrix transformation objects      ager 12/14 |
/-----------------------------------------------------------------------*/
void FPSI::FpsiCoupling::re_init_coupling_matrix_transform()
{
  // create transformation objects for coupling terms
  couplingrowtransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  couplingrowtransform2_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  couplingrowtransform3_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  couplingrowtransform4_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  couplingrowtransform5_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  couplingcoltransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  couplingcoltransform2_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  couplingrowcoltransform_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();
  couplingrowcoltransform2_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();
}

/*-------------------------------------------------------------------------------/
| Evaluate Coupling Matrixes and Coupling RHS    orig. rauch / modif. ager 12/14 |
/-------------------------------------------------------------------------------*/
void FPSI::FpsiCoupling::evaluate_coupling_matrixes_rhs()
{
  // Evaluates all Coupling Matrixes ...
  TEUCHOS_FUNC_TIME_MONITOR("FPSI::fpsi_coupling::evaluate_coupling_matrixes_rhs");

  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fp_porofluid =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *(poro_field()->fluid_field()->dof_row_map()), 81, true, true);
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_pf_porofluid =
      std::make_shared<Core::LinAlg::SparseMatrix>(*(fluid_field()->dof_row_map()), 81, true, true);

  // set all coupling matrixes to zero!!
  c_pp_->zero();
  c_ff_->zero();
  c_pf_->zero();
  c_fp_->zero();
  c_fa_->zero();
  c_pa_->zero();

  c_rhs_s_->put_scalar(0.0);
  c_rhs_pf_->put_scalar(0.0);
  c_rhs_f_->put_scalar(0.0);

  k_pf_porofluid->zero();

  const Coupling::Adapter::Coupling& couppff_fpsi = *icoup_pf_f_;
  const Coupling::Adapter::Coupling& coupsf_fpsi = *icoup_ps_f_;
  const Coupling::Adapter::Coupling& coup_ps_a_fpsi = *icoup_ps_a_;

  Global::Problem* problem = Global::Problem::instance();
  const Teuchos::ParameterList& fpsidynparams = problem->fpsi_dynamic_params();
  auto method = Teuchos::getIntegralValue<Inpar::FPSI::PartitionedCouplingMethod>(
      fpsidynparams, "PARTITIONED");

  if (method != Inpar::FPSI::nocoupling)
  {
    // set general vector values needed by elements

    poro_field()->fluid_field()->discretization()->clear_state();

    poro_field()->fluid_field()->discretization()->set_state(
        0, "dispnp", *poro_field()->fluid_field()->dispnp());

    poro_field()->fluid_field()->discretization()->set_state(
        0, "gridv", *poro_field()->fluid_field()->grid_vel());
    poro_field()->fluid_field()->discretization()->set_state(
        0, "dispn", *poro_field()->fluid_field()->dispn());
    poro_field()->fluid_field()->discretization()->set_state(
        0, "veln", *poro_field()->fluid_field()->veln());
    poro_field()->fluid_field()->discretization()->set_state(
        0, "velaf", *poro_field()->fluid_field()->velnp());
    poro_field()->fluid_field()->discretization()->set_state(
        0, "velnp", *poro_field()->fluid_field()->velnp());

    fluid_field()->discretization()->clear_state();

    fluid_field()->discretization()->set_state(0, "dispnp", *fluid_field()->dispnp());
    fluid_field()->discretization()->set_state(0, "gridv", *fluid_field()->grid_vel());
    fluid_field()->discretization()->set_state(0, "dispn", *fluid_field()->dispn());
    fluid_field()->discretization()->set_state(0, "veln", *fluid_field()->veln());
    fluid_field()->discretization()->set_state(0, "velaf", *fluid_field()->velnp());
    fluid_field()->discretization()->set_state(0, "velnp", *fluid_field()->velnp());

    // create the parameters for the discretization
    Teuchos::ParameterList fparams;

    // action for elements
    fparams.set<FLD::BoundaryAction>("action", FLD::fpsi_coupling);
    fparams.set("timescale", poro_field()->fluid_field()->residual_scaling());

    fparams.set("dt", fpsidynparams.get<double>("TIMESTEP"));
    fparams.set<Inpar::FLUID::PhysicalType>(
        "Physical Type", poro_field()->fluid_field()->physical_type());

    if (method == Inpar::FPSI::monolithic)
    {
      fparams.set<std::string>("fillblock", "Porofluid_Freefluid");
      fparams.set("InterfaceFacingElementMap", fluid_poro_fluid_interface_map_);
      Core::FE::AssembleStrategy fluidstrategy(0,  // fluiddofset for row
          0,                                       // fluiddofset for column
          k_pf_porofluid,                          // coupling matrix with fluid rowmap
          nullptr,                                 // no other matrix or vectors
          nullptr, nullptr, nullptr);

      // what's the current problem type? Is it a fps3i problem?
      Core::ProblemType probtype = Global::Problem::instance()->get_problem_type();

      if (probtype == Core::ProblemType::fps3i)
      {
        if (conductivity_ == 0.0)
        {
          FOUR_C_THROW(
              "In the case of FPS3I, a positive conductivity must be set in DESIGN SCATRA COUPLING "
              "SURF CONDITIONS");
        }
        fparams.set("membrane conductivity", conductivity_);
      }

      fluid_field()->discretization()->evaluate_condition(fparams, fluidstrategy, "fpsi_coupling");
      k_pf_porofluid->complete(*fluid_field()->dof_row_map(), *fluid_field()->dof_row_map());

      {
        TEUCHOS_FUNC_TIME_MONITOR("FPSI::Monolithic::transform");
        (*couplingrowtransform_)(*k_pf_porofluid, 1.0,
            Coupling::Adapter::CouplingSlaveConverter(couppff_fpsi), c_pf_->matrix(1, 0), true);
      }

      fparams.set<std::string>("fillblock", "Porofluid_Structure");
      fparams.set("InterfaceFacingElementMap", fluid_poro_fluid_interface_map_);
      k_pf_porofluid->zero();
      Core::FE::AssembleStrategy fluidstrategy21(0,  // porofluiddofset for row
          0,                                         // porofluiddofset for column
          k_pf_porofluid,                            // coupling matrix with fluid rowmap
          nullptr,                                   // no other matrix or vectors
          nullptr, nullptr, nullptr);

      fluid_field()->discretization()->evaluate_condition(
          fparams, fluidstrategy21, "fpsi_coupling");

      k_pf_porofluid->complete(*fluid_field()->dof_row_map(), *fluid_field()->dof_row_map());
      {
        TEUCHOS_FUNC_TIME_MONITOR("FPSI::Monolithic::transform");
        (*couplingrowcoltransform2_)(*k_pf_porofluid, 1.0,
            Coupling::Adapter::CouplingSlaveConverter(
                couppff_fpsi),  // row converter: important to use slave converter
            Coupling::Adapter::CouplingSlaveConverter(
                coupsf_fpsi),  //  col converter: important to use slave converter
            c_pp_->matrix(1, 0),
            false,  // bool exactmatch = true (default)
            true);
      }

      fparams.set<std::string>("fillblock", "Fluid_Porofluid");
      fparams.set("InterfaceFacingElementMap", poro_fluid_fluid_interface_map_);
      k_fp_porofluid->zero();
      Core::FE::AssembleStrategy porofluidstrategy(0,  // porofluiddofset for row
          0,                                           // porofluiddofset for column
          k_fp_porofluid,                              // porofluid-structure coupling matrix
          nullptr,                                     // no other matrix or vectors
          nullptr, nullptr, nullptr);

      poro_field()->fluid_field()->discretization()->evaluate_condition(
          fparams, porofluidstrategy, "fpsi_coupling");
      k_fp_porofluid->complete(poro_field()->fluid_domain_map(), poro_field()->fluid_range_map());

      {
        TEUCHOS_FUNC_TIME_MONITOR("FPSI::Monolithic::transform");
        (*couplingrowtransform2_)(*k_fp_porofluid, 1.0,
            Coupling::Adapter::CouplingMasterConverter(couppff_fpsi), c_fp_->matrix(0, 1),
            true);  // add
      }

      fparams.set<std::string>("fillblock", "Fluid_Structure");
      fparams.set("InterfaceFacingElementMap", poro_fluid_fluid_interface_map_);

      // move me somewhere else
      std::shared_ptr<Core::LinAlg::SparseMatrix> k_pfs_ =
          std::make_shared<Core::LinAlg::SparseMatrix>(
              *(poro_field()->fluid_field()->discretization()->dof_row_map()), 81, true, true);

      k_pfs_->un_complete();

      Core::FE::AssembleStrategy structurestrategy(0,  // porofluiddofset for row
          1,                                           // structuredofset for column
          k_pfs_,                                      // coupling matrix with porofluid rowmap
          nullptr,                                     // no other matrix or vectors
          nullptr, nullptr, nullptr);

      poro_field()->fluid_field()->discretization()->evaluate_condition(
          fparams, structurestrategy, "fpsi_coupling");

      k_pfs_->complete(poro_field()->structure_domain_map(), poro_field()->fluid_range_map());
      {
        TEUCHOS_FUNC_TIME_MONITOR("FPSI::Monolithic::transform");
        (*couplingrowtransform3_)(*k_pfs_, 1.0,
            Coupling::Adapter::CouplingMasterConverter(couppff_fpsi), c_fp_->matrix(0, 0),
            true);  // add
      }

      ///// Fluid_Structure (fluid part / linearization of tangentials with respect to
      /// displacements)
      k_pf_porofluid->reset();
      k_pf_porofluid->un_complete();

      Core::FE::AssembleStrategy structurestrategy2(0,  // porofluiddofset for row
          0,                                            // fluiddofset for column
          k_pf_porofluid,                               // coupling matrix with porofluid rowmap
          nullptr,                                      // no other matrix or vectors
          nullptr, nullptr, nullptr);

      fparams.set("InterfaceFacingElementMap", fluid_poro_fluid_interface_map_);
      fluid_field()->discretization()->evaluate_condition(
          fparams, structurestrategy2, "fpsi_coupling");

      k_pf_porofluid->complete(*fluid_field()->dof_row_map(), *fluid_field()->dof_row_map());

      {
        TEUCHOS_FUNC_TIME_MONITOR("FPSI::Monolithic::transform");
        (*couplingcoltransform_)(fluid_field()->block_system_matrix()->full_row_map(),
            fluid_field()->block_system_matrix()->full_col_map(), *k_pf_porofluid, 1.0,
            Coupling::Adapter::CouplingSlaveConverter(
                coupsf_fpsi),  // row converter: important to use slave converter
            c_fp_->matrix(0, 0),
            false,  // bool exactmatch = true (default)
            true);
      }

      fparams.set<std::string>("fillblock", "Fluid_Fluid");
      fparams.set("InterfaceFacingElementMap", fluid_poro_fluid_interface_map_);
      k_pf_porofluid->zero();
      k_pf_porofluid->un_complete();

      Core::FE::AssembleStrategy fluidfluidstrategy(0,  // fluiddofset for row
          0,                                            // fluiddofset for column
          c_ff_,                                        // porofluid-structure coupling matrix
          nullptr,                                      // no other matrix or vectors
          nullptr, nullptr, nullptr);

      fluid_field()->discretization()->evaluate_condition(
          fparams, fluidfluidstrategy, "fpsi_coupling");

      fparams.set<std::string>("fillblock", "Structure_Fluid");
      fparams.set("InterfaceFacingElementMap", fluid_poro_fluid_interface_map_);
      k_pf_porofluid->zero();
      k_pf_porofluid->un_complete();

      Core::FE::AssembleStrategy structurefluidstrategy(0,  // fluid dofset for row
          0,                                                // fluid dofset for column
          k_pf_porofluid,                                   // coupling matrix with fluid rowmap
          nullptr,                                          // no other matrix or vectors
          nullptr, nullptr, nullptr);

      fluid_field()->discretization()->evaluate_condition(
          fparams, structurefluidstrategy, "fpsi_coupling");

      k_pf_porofluid->complete(*fluid_field()->dof_row_map(), *fluid_field()->dof_row_map());
      {
        TEUCHOS_FUNC_TIME_MONITOR("FPSI::Monolithic::transform");
        (*couplingrowtransform4_)(
            *std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(k_pf_porofluid), 1.0,
            Coupling::Adapter::CouplingSlaveConverter(
                coupsf_fpsi),  // important to use slave converter
            c_pf_->matrix(0, 0),
            true);  // add
      }

      fparams.set<std::string>("fillblock", "Structure_Structure");
      fparams.set("InterfaceFacingElementMap", fluid_poro_fluid_interface_map_);
      k_pf_porofluid->zero();
      k_pf_porofluid->un_complete();

      Core::FE::AssembleStrategy structurestructurestrategy(0,  // fluid dofset for row
          0,                                                    // fluid dofset for column
          k_pf_porofluid,                                       // coupling matrix with fluid rowmap
          nullptr,                                              // no other matrix or vectors
          nullptr, nullptr, nullptr);

      fluid_field()->discretization()->evaluate_condition(
          fparams, structurestructurestrategy, "fpsi_coupling");
      // condense linearization with respect to the ale mesh motion (interface structural
      // displacements = interface ale displacements)
      k_pf_porofluid->complete(*fluid_field()->dof_row_map(), *fluid_field()->dof_row_map());

      {
        TEUCHOS_FUNC_TIME_MONITOR("FPSI::Monolithic::transform");
        (*couplingrowcoltransform_)(*k_pf_porofluid, 1.0,
            Coupling::Adapter::CouplingSlaveConverter(
                coupsf_fpsi),  // row converter: important to use slave converter
            Coupling::Adapter::CouplingSlaveConverter(
                coupsf_fpsi),  // col converter: important to use slave converter
            c_pp_->matrix(0, 0),
            false,  // bool exactmatch = true (default)
            true);  // add
      }

      // Process inner ale dofs
      fparams.set<std::string>("fillblock", "Structure_Ale");
      fparams.set("InterfaceFacingElementMap", fluid_poro_fluid_interface_map_);

      // temporal matrix
      // todo (initialization should be avoided in every iteration...)
      std::shared_ptr<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>
          temp6 = std::make_shared<
              Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

              *ale_field()->interface(), *fluid_field()->interface(), 81, false,
              false);  // Use fluid_field()->Interface =

      // assemble into fluid row and column dofs -> need to transform rows to structure dofs and
      // cols to ale dofs
      Core::FE::AssembleStrategy structurealestrategy(0,  // fluid dofset for row
          1,                                              // ale dofset for column
          temp6,                                          // coupling matrix with fluid rowmap
          nullptr,                                        // no other matrix or vectors
          nullptr, nullptr, nullptr);

      // evaluate coupling terms
      fluid_field()->discretization()->evaluate_condition(
          fparams, structurealestrategy, "fpsi_coupling");
      temp6->complete();  // for row transform!

      {
        TEUCHOS_FUNC_TIME_MONITOR("FPSI::Monolithic::transform");
        (*couplingrowtransform5_)(temp6->matrix(FLD::Utils::MapExtractor::cond_other,
                                      ALE::Utils::MapExtractor::cond_other),
            1.0,
            Coupling::Adapter::CouplingSlaveConverter(
                coupsf_fpsi),  // important to use slave converter
            c_pa_->matrix(0, 0), false);
      }
    }  // if monolithic
    else
    {
      if (isfirstcall_)
        std::cout << "LINEARIZATION OF FPSI INTERFACE IS TURNED OFF \n"
                     "IF YOU THINK THAT SUCKS. SET 'PARITIONED' to 'monolithic' \n"
                     "IN THE FPSI SECTION OF YOUR INPUT FILE !!! "
                  << std::endl;
      isfirstcall_ = false;
    }

    //////////////////////////////////////
    ///////      __       ___       //////
    ///////      |_|  |_| |__       //////
    ///////      | \  | |  __|      //////
    //////                          //////
    //////////////////////////////////////

    std::shared_ptr<Core::LinAlg::Vector<double>> temprhs = nullptr;
    std::shared_ptr<Core::LinAlg::Vector<double>> temprhs2 = nullptr;

    fparams.set<std::string>("fillblock", "conti");
    fparams.set("InterfaceFacingElementMap", fluid_poro_fluid_interface_map_);
    temprhs = std::make_shared<Core::LinAlg::Vector<double>>(*fluid_field()->dof_row_map(), true);
    temprhs2 = std::make_shared<Core::LinAlg::Vector<double>>(*poro_field()->dof_row_map(), true);
    temprhs->put_scalar(0.0);
    temprhs2->put_scalar(0.0);

    Core::FE::AssembleStrategy rhscontistrategy(0,  // fluid dofset for row
        0,                                          // fluid dofset for column
        nullptr, nullptr,
        temprhs,  // rhs vector
        nullptr, nullptr);

    fluid_field()->discretization()->evaluate_condition(fparams, rhscontistrategy, "fpsi_coupling");

    // extract FPSI part of the fluid field
    temprhs = fluidvelpres_extractor_->extract_cond_vector(*temprhs);

    // replace global fluid interface dofs through porofluid interface dofs
    temprhs = i_fluid_to_porofluid(*temprhs);

    // insert porofluid interface entries into vector with full porofield length (0: inner dofs of
    // structure, 1: interface dofs of structure, 2: inner dofs of porofluid, 3: interface dofs of
    // porofluid )
    porofluid_extractor_->insert_cond_vector(*temprhs, *c_rhs_pf_);

    // add vector with full porofield length to global rhs

    temprhs->put_scalar(0.0);
    temprhs2->put_scalar(0.0);

    fparams.set<std::string>("fillblock", "structure");
    fparams.set("InterfaceFacingElementMap", fluid_poro_fluid_interface_map_);
    temprhs = std::make_shared<Core::LinAlg::Vector<double>>(*fluid_field()->dof_row_map(), true);
    temprhs2 = std::make_shared<Core::LinAlg::Vector<double>>(*poro_field()->dof_row_map(), true);

    Core::FE::AssembleStrategy rhsstructurestrategy(0,  // fluid dofset for row
        0,                                              // fluid dofset for column
        nullptr, nullptr,
        temprhs,  // rhs vector
        nullptr, nullptr);

    fluid_field()->discretization()->evaluate_condition(
        fparams, rhsstructurestrategy, "fpsi_coupling");

    // extract FPSI part of the fluid field
    temprhs = fluidvel_extractor_->extract_cond_vector(*temprhs);  //
    // replace global fluid interface dofs through porofluid interface dofs
    temprhs = i_fluid_to_porostruct(*temprhs);  //
    // insert porofluid interface
    porostruct_extractor_->add_cond_vector(*temprhs, *c_rhs_s_);

    temprhs->put_scalar(0.0);
    temprhs2->put_scalar(0.0);

    fparams.set<std::string>("fillblock", "fluid");
    fparams.set("InterfaceFacingElementMap", poro_fluid_fluid_interface_map_);
    temprhs = std::make_shared<Core::LinAlg::Vector<double>>(
        *poro_field()->fluid_field()->dof_row_map(0), true);
    temprhs2 = std::make_shared<Core::LinAlg::Vector<double>>(*fluid_field()->dof_row_map(0), true);

    Core::FE::AssembleStrategy rhsfluidstrategy(0,  // fluid dofset for row
        0,                                          // fluid dofset for column
        nullptr, nullptr,
        temprhs,  // rhs vector
        nullptr, nullptr);

    poro_field()->fluid_field()->discretization()->evaluate_condition(
        fparams, rhsfluidstrategy, "fpsi_coupling");
    // extract FPSI part of the poro fluid field
    temprhs = porofluid_extractor_->extract_cond_vector(*temprhs);  //

    // replace global fluid interface dofs through porofluid interface dofs
    temprhs = i_porofluid_to_fluid(*temprhs);
    // insert porofluid interface entries into vector with full fluidfield length
    fluidvelpres_extractor_->insert_cond_vector(*temprhs, *temprhs2);
    // add vector with full porofield length to global rhs
    c_rhs_f_->update(1.0, *temprhs2, 0.0);

    temprhs->put_scalar(0.0);
    temprhs2->put_scalar(0.0);

    fparams.set<std::string>("fillblock", "fluidfluid");  // (wot,tangentialfac*uot) part
    fparams.set("InterfaceFacingElementMap", fluid_poro_fluid_interface_map_);
    temprhs = std::make_shared<Core::LinAlg::Vector<double>>(*fluid_field()->dof_row_map(0), true);

    Core::FE::AssembleStrategy rhsfluidfluidstrategy(0,  // fluid dofset for row
        0,                                               // fluid dofset for column
        nullptr, nullptr,
        temprhs,  // rhs vector
        nullptr, nullptr);

    fluid_field()->discretization()->evaluate_condition(
        fparams, rhsfluidfluidstrategy, "fpsi_coupling");

    c_rhs_f_->update(1.0, *temprhs, 1.0);

    temprhs->put_scalar(0.0);
    temprhs2->put_scalar(0.0);

    //////////////////////////////////////
    //////                          //////
    //////   NEUMANN INTEGRATION    //////
    //////                          //////
    //////////////////////////////////////
    if (method == Inpar::FPSI::monolithic or method == Inpar::FPSI::RobinNeumann)
    {
      fparams.set<std::string>("fillblock", "NeumannIntegration");
      fparams.set("InterfaceFacingElementMap", fluid_poro_fluid_interface_map_);

      Core::FE::AssembleStrategy rhsfluidfluidstrategy2(0,  // fluid dofset for row
          0,                                                // fluid dofset for column
          c_ff_,                                            // coupling matrix with fluid rowmap
          nullptr,
          temprhs,  // rhs vector
          nullptr, nullptr);

      fluid_field()->discretization()->evaluate_condition(
          fparams, rhsfluidfluidstrategy2, "NeumannIntegration");

      c_rhs_f_->update(1.0, *temprhs, 1.0);
      temprhs->put_scalar(0.0);

      {
        fparams.set<std::string>("fillblock", "NeumannIntegration_Ale");
        fparams.set("InterfaceFacingElementMap", fluid_poro_fluid_interface_map_);

        std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> tmp_c_fa = std::make_shared<
            Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

            *ale_field()->interface(), *fluid_field()->fpsi_interface(), 81, false, true);

        Core::FE::AssembleStrategy rhsfluidfluidstrategy3(0,  // fluid dofset for row
            1,                                                // ale dofset for column
            tmp_c_fa,                                         // coupling matrix with fluid rowmap
            nullptr, nullptr, nullptr, nullptr);

        fluid_field()->discretization()->evaluate_condition(
            fparams, rhsfluidfluidstrategy3, "NeumannIntegration");
        tmp_c_fa->complete();

        // Add all inner ale parts to c_fa_ directly
        Core::LinAlg::matrix_add(tmp_c_fa->matrix(FLD::Utils::MapExtractor::cond_other,
                                     ALE::Utils::MapExtractor::cond_other),
            false, 1.0, *c_fa_, 0.0);
        Core::LinAlg::matrix_add(tmp_c_fa->matrix(FLD::Utils::MapExtractor::cond_fsi,
                                     ALE::Utils::MapExtractor::cond_other),
            false, 1.0, *c_fa_, 1.0);

        //-->now transform ale fpsi block to structure (is condensed)!!!
        // still in ale domain map!!!
        Core::LinAlg::SparseMatrix tmp_c_fp =
            Core::LinAlg::SparseMatrix(*fluid_field()->dof_row_map(), 81);

        // Add all condensed parts to tmp_c_fa...
        Core::LinAlg::matrix_add(tmp_c_fa->matrix(FLD::Utils::MapExtractor::cond_other,
                                     ALE::Utils::MapExtractor::cond_fpsi),
            false, 1.0, tmp_c_fp, 0.0);
        Core::LinAlg::matrix_add(tmp_c_fa->matrix(FLD::Utils::MapExtractor::cond_fsi,
                                     ALE::Utils::MapExtractor::cond_fpsi),
            false, 1.0, tmp_c_fp, 1.0);
        Core::LinAlg::matrix_add(tmp_c_fa->matrix(FLD::Utils::MapExtractor::cond_other,
                                     ALE::Utils::MapExtractor::cond_fsi),
            false, 1.0, tmp_c_fp, 1.0);
        Core::LinAlg::matrix_add(tmp_c_fa->matrix(FLD::Utils::MapExtractor::cond_fsi,
                                     ALE::Utils::MapExtractor::cond_fsi),
            false, 1.0, tmp_c_fp, 1.0);
        tmp_c_fp.complete(
            *ale_field()->interface()->fpsi_cond_map(), *fluid_field()->dof_row_map());

        // For Ale Condensation ==> AleColumns to StructuralColumns
        (*couplingcoltransform2_)(ale_field()->block_system_matrix()->full_row_map(),
            ale_field()->block_system_matrix()->full_col_map(), tmp_c_fp, 1.0,
            Coupling::Adapter::CouplingSlaveConverter(
                coup_ps_a_fpsi),  // row converter: important to use slave converter
            c_fp_->matrix(0, 0),
            false,  // bool exactmatch = true (default)
            true);  // bool add = false (default)
      }
    }
    else  // only fill rhs
    {
      fparams.set<std::string>("fillblock", "NeumannIntegration");
      fparams.set("InterfaceFacingElementMap", fluid_poro_fluid_interface_map_);
      temprhs->put_scalar(0.0);
      temprhs2->put_scalar(0.0);
      fluid_field()->discretization()->evaluate_condition(
          fparams, rhsfluidfluidstrategy, "NeumannIntegration");

      c_rhs_f_->update(1.0, *temprhs, 1.0);
    }

    ////////////////////////////
    // DONE -> CLEAR STATES
    poro_field()->fluid_field()->discretization()->clear_state();
    fluid_field()->discretization()->clear_state();

  }  // if not nocoupling

  ////////////////////////////
  // DONE -> Complete Coupling Matrixes
  c_ff_->complete();
  c_pp_->complete();
}

/*----------------------------------------------------------------------/
| set hydraulic conductivity
/----------------------------------------------------------------------*/
void FPSI::FpsiCoupling::set_conductivity(double conduct) { conductivity_ = conduct; }

FOUR_C_NAMESPACE_CLOSE

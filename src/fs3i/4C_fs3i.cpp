// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fs3i.hpp"

#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_adapter_structure_scatra_ele.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_condition_selector.hpp"
#include "4C_fluid_implicit_integration.hpp"
#include "4C_fsi_dyn.hpp"
#include "4C_fsi_monolithicfluidsplit.hpp"
#include "4C_fsi_monolithicstructuresplit.hpp"
#include "4C_fsi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fs3i.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_scatra_algorithm.hpp"
#include "4C_scatra_timint_implicit.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FS3I::FS3IBase::FS3IBase()
    : infperm_(Global::Problem::instance()->f_s3_i_dynamic_params().get<bool>("INF_PERM")),
      timemax_(Global::Problem::instance()->f_s3_i_dynamic_params().get<double>("MAXTIME")),
      numstep_(Global::Problem::instance()->f_s3_i_dynamic_params().get<int>("NUMSTEP")),
      dt_(Global::Problem::instance()->f_s3_i_dynamic_params().get<double>("TIMESTEP")),
      time_(0.0),
      step_(0),
      issetup_(false),
      isinit_(false)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::init()
{
  set_is_setup(false);

  scatracoup_ = std::make_shared<Coupling::Adapter::Coupling>();
  scatraglobalex_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
  sbbtransform_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();
  sbitransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  sibtransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  fbitransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();

  set_is_init(true);
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::setup()
{
  check_is_init();

  set_is_setup(true);
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::check_interface_dirichlet_bc()
{
  std::shared_ptr<Core::FE::Discretization> masterdis =
      scatravec_[0]->scatra_field()->discretization();
  std::shared_ptr<Core::FE::Discretization> slavedis =
      scatravec_[1]->scatra_field()->discretization();

  std::shared_ptr<const Core::LinAlg::Map> mastermap = scatracoup_->master_dof_map();
  std::shared_ptr<const Core::LinAlg::Map> permmastermap = scatracoup_->perm_master_dof_map();
  std::shared_ptr<const Core::LinAlg::Map> slavemap = scatracoup_->slave_dof_map();
  std::shared_ptr<const Core::LinAlg::Map> permslavemap = scatracoup_->perm_slave_dof_map();

  const std::shared_ptr<const Core::LinAlg::MapExtractor> masterdirichmapex =
      scatravec_[0]->scatra_field()->dirich_maps();
  const std::shared_ptr<const Core::LinAlg::Map> masterdirichmap = masterdirichmapex->cond_map();

  // filter out master dirichlet dofs associated with the interface
  Core::LinAlg::Vector<double> masterifdirich(*mastermap, true);
  for (int i = 0; i < mastermap->num_my_elements(); ++i)
  {
    int gid = mastermap->gid(i);
    if (masterdirichmap->my_gid(gid))
    {
      (masterifdirich).get_values()[i] = 1.0;
    }
  }
  std::shared_ptr<Core::LinAlg::Vector<double>> test_slaveifdirich =
      scatracoup_->master_to_slave(masterifdirich);

  const std::shared_ptr<const Core::LinAlg::MapExtractor> slavedirichmapex =
      scatravec_[1]->scatra_field()->dirich_maps();
  const std::shared_ptr<const Core::LinAlg::Map> slavedirichmap = slavedirichmapex->cond_map();

  // filter out slave dirichlet dofs associated with the interface
  Core::LinAlg::Vector<double> slaveifdirich(*slavemap, true);
  for (int i = 0; i < slavemap->num_my_elements(); ++i)
  {
    int gid = slavemap->gid(i);
    if (slavedirichmap->my_gid(gid))
    {
      (slaveifdirich).get_values()[i] = 1.0;
    }
  }
  std::shared_ptr<Core::LinAlg::Vector<double>> test_masterifdirich =
      scatracoup_->slave_to_master(slaveifdirich);

  // check if the locations of non-zero entries do not match
  for (int i = 0; i < slavedis->dof_row_map()->num_my_elements(); ++i)
  {
    int gid = slavedis->dof_row_map()->gid(i);
    if (slavemap->my_gid(gid))  // in this case, the current dof is part of the interface
    {
      if (test_slaveifdirich->local_values_as_span()[slavemap->lid(gid)] == 1.0 and
          slaveifdirich.local_values_as_span()[slavemap->lid(gid)] != 1.0)
      {
        FOUR_C_THROW("Dirichlet boundary conditions not matching at the interface");
      }
    }
  }

  for (int i = 0; i < masterdis->dof_row_map()->num_my_elements(); ++i)
  {
    int gid = masterdis->dof_row_map()->gid(i);
    if (mastermap->my_gid(gid))  // in this case, the current dof is part of the interface
    {
      if (test_masterifdirich->local_values_as_span()[mastermap->lid(gid)] == 1.0 and
          masterifdirich.local_values_as_span()[mastermap->lid(gid)] != 1.0)
      {
        FOUR_C_THROW("Dirichlet boundary conditions not matching at the interface");
      }
    }
  }
}

/*----------------------------------------------------------------------*
 | Check FS3I specific inputs                                Thon 11/14 |
 *----------------------------------------------------------------------*/
void FS3I::FS3IBase::check_f_s3_i_inputs()
{
  // Check FS3I dynamic parameters
  Global::Problem* problem = Global::Problem::instance();
  const Teuchos::ParameterList& fs3idyn = problem->f_s3_i_dynamic_params();
  const Teuchos::ParameterList& structdynparams = problem->structural_dynamic_params();
  const Teuchos::ParameterList& scatradynparams = problem->scalar_transport_dynamic_params();
  const Teuchos::ParameterList& fluiddynparams = problem->fluid_dynamic_params();

  // check consistency of time-integration schemes in input file
  // (including parameter theta itself in case of one-step-theta scheme)
  // and rule out unsupported versions of generalized-alpha time-integration
  // scheme (as well as other inappropriate schemes) for fluid subproblem
  auto scatratimealgo = Teuchos::getIntegralValue<Inpar::ScaTra::TimeIntegrationScheme>(
      scatradynparams, "TIMEINTEGR");
  auto fluidtimealgo =
      Teuchos::getIntegralValue<Inpar::FLUID::TimeIntegrationScheme>(fluiddynparams, "TIMEINTEGR");
  auto structtimealgo =
      Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(structdynparams, "DYNAMICTYPE");

  if (fluidtimealgo == Inpar::FLUID::timeint_one_step_theta)
  {
    if (scatratimealgo != Inpar::ScaTra::timeint_one_step_theta or
        structtimealgo != Inpar::Solid::DynamicType::OneStepTheta)
      FOUR_C_THROW(
          "Partitioned FS3I computations should feature consistent time-integration schemes for "
          "the subproblems; in this case, a one-step-theta scheme is intended to be used for the "
          "fluid subproblem, and different schemes are intended to be used for the structure "
          "and/or scalar transport subproblems!");

    if (scatradynparams.get<double>("THETA") != fluiddynparams.get<double>("THETA") or
        scatradynparams.get<double>("THETA") !=
            structdynparams.sublist("ONESTEPTHETA").get<double>("THETA"))
      FOUR_C_THROW(
          "Parameter(s) theta for one-step-theta time-integration scheme defined in one or more of "
          "the individual fields do(es) not match for partitioned FS3I computation.");
  }
  else if (fluidtimealgo == Inpar::FLUID::timeint_afgenalpha)
  {
    if (scatratimealgo != Inpar::ScaTra::timeint_gen_alpha or
        structtimealgo != Inpar::Solid::DynamicType::GenAlpha)
      FOUR_C_THROW(
          "Partitioned FS3I computations should feature consistent time-integration schemes for "
          "the subproblems; in this case, a (alpha_f-based) generalized-alpha scheme is intended "
          "to be used for the fluid subproblem, and different schemes are intended to be used for "
          "the structure and/or scalar transport subproblems!");
  }
  else if (fluidtimealgo == Inpar::FLUID::timeint_npgenalpha)
  {
    FOUR_C_THROW(
        "Partitioned FS3I computations do not support n+1-based generalized-alpha time-integration "
        "schemes for the fluid subproblem!");
  }
  else if (fluidtimealgo == Inpar::FLUID::timeint_bdf2 or
           fluidtimealgo == Inpar::FLUID::timeint_stationary)
  {
    FOUR_C_THROW(
        "Partitioned FS3I computations do not support stationary of BDF2 time-integration schemes "
        "for the fluid subproblem!");
  }

  // check that incremental formulation is used for scalar transport field,
  // according to structure and fluid field
  if (scatravec_[0]->scatra_field()->is_incremental() == false)
    FOUR_C_THROW("Incremental formulation required for partitioned FS3I computations!");

  if (Teuchos::getIntegralValue<Inpar::ScaTra::ConvForm>(fs3idyn, "STRUCTSCAL_CONVFORM") ==
          Inpar::ScaTra::convform_convective and
      Teuchos::getIntegralValue<Inpar::FS3I::VolumeCoupling>(fs3idyn, "STRUCTSCAL_FIELDCOUPLING") ==
          Inpar::FS3I::coupling_match)
  {
    FOUR_C_THROW(
        "Your scalar fields have to be calculated in conservative form, since the velocity field "
        "in the structure is NOT divergence free!");
  }

  auto pstype = Teuchos::getIntegralValue<Inpar::Solid::PreStress>(
      Global::Problem::instance()->structural_dynamic_params(), "PRESTRESS");
  // is structure calculated dynamic when not prestressing?
  if (Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(structdynparams, "DYNAMICTYPE") ==
          Inpar::Solid::DynamicType::Statics and
      pstype != Inpar::Solid::PreStress::mulf)
    FOUR_C_THROW(
        "Since we need a velocity field in the structure domain for the scalar field you need do "
        "calculate the structure dynamically! Exception: when prestressing..");


  // Check DESIGN SCATRA COUPLING SURF CONDITIONS
  std::vector<std::set<int>> condIDs;
  std::set<int> fluidIDs;
  std::set<int> structIDs;
  condIDs.push_back(fluidIDs);
  condIDs.push_back(structIDs);
  std::vector<std::map<int, std::vector<double>*>> PermCoeffs;
  std::map<int, std::vector<double>*> fluidcoeff;
  std::map<int, std::vector<double>*> structcoeff;
  PermCoeffs.push_back(fluidcoeff);
  PermCoeffs.push_back(structcoeff);
  const int numscal = scatravec_[0]->scatra_field()->num_scal();

  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    std::shared_ptr<Core::FE::Discretization> disscatra =
        (scatravec_[i])->scatra_field()->discretization();
    std::vector<const Core::Conditions::Condition*> coupcond;
    disscatra->get_condition("ScaTraCoupling", coupcond);

    for (auto& iter : coupcond)
    {
      int myID = iter->parameters().get<int>("COUPID");
      condIDs[i].insert(myID);

      if (!infperm_)  // get all FS3I interface condition parameters from the input file
      {
        // initialize a large enough vector
        auto* params = new std::vector<double>(7, true);
        params->at(0) = iter->parameters().get<double>("PERMCOEF");
        params->at(1) = iter->parameters().get<double>("CONDUCT");
        params->at(2) = iter->parameters().get<double>("FILTR");
        params->at(3) = (double)iter->parameters().get<bool>("WSSON");
        const auto& mywsscoeffs = iter->parameters().get<std::vector<double>>("WSSCOEFFS");
        params->at(4) = mywsscoeffs.at(0);
        params->at(5) = mywsscoeffs.at(1);
        params->at(6) = (double)(iter->parameters().get<int>("NUMSCAL"));
        const auto& onoffs = iter->parameters().get<std::vector<int>>("ONOFF");
        for (int k = 0; k < numscal; k++)
        {
          params->push_back((double)(onoffs.at(k)));
        }

        if (scatravec_[i]->scatra_field()->num_scal() != params->at(6))
          FOUR_C_THROW(
              "Number of scalars NUMSCAL in ScaTra coupling conditions with COUPID {} does not "
              "equal the number of scalars your scalar field has!",
              myID);

        if ((bool)params->at(3))  // if we have WSS depended interface permeabiliy
        {
          std::vector<const Core::Conditions::Condition*> FSCCond;
          problem->get_dis("fluid")->get_condition("FluidStressCalc", FSCCond);

          if (FSCCond.size() == 0)
            FOUR_C_THROW(
                "If you have a WSS dependent interface permeablity you need at least one FLUID "
                "STRESS CALC CONDITION to specify the region you want to evaluate the WSS. "
                "Typically this region is equal to the SSI interface...");
        }

        PermCoeffs[i].insert(std::pair<int, std::vector<double>*>(myID, params));
      }
    }
  }

  if (condIDs[0].size() != condIDs[1].size())
    FOUR_C_THROW("ScaTra coupling conditions need to be defined on both discretizations");

  if (!infperm_)  // now do the testing
  {
    std::map<int, std::vector<double>*> fluid_PermCoeffs = PermCoeffs[0];
    std::map<int, std::vector<double>*> struct_PermCoeffs = PermCoeffs[1];

    std::vector<int>* onoff_sum = new std::vector<int>(numscal, 0);

    for (std::map<int, std::vector<double>*>::iterator fit = fluid_PermCoeffs.begin();
        fit != fluid_PermCoeffs.end(); ++fit)  // loop over all fluid-scatra COUPIDs
    {
      const int ID = (*fit).first;
      std::vector<double>* fluid_permcoeffs =
          (*fit).second;  // get the pointer to the fluid-scatra params

      std::map<int, std::vector<double>*>::iterator sit = struct_PermCoeffs.find(
          ID);  // get corresponding structure-scatra condition with same COUPID
      std::vector<double>* structure_permcoeffs =
          (*sit).second;  // get the pointer to the structure-scatra params

      // no the actual testing
      if (fluid_permcoeffs->at(0) != structure_permcoeffs->at(0))
        FOUR_C_THROW(
            "Permeability coefficient PERMCOEF of ScaTra couplings with COUPID {} needs to be the "
            "same!",
            ID);
      if (fluid_permcoeffs->at(1) != structure_permcoeffs->at(1))
        FOUR_C_THROW(
            "Hydraulic conductivity coefficient CONDUCT of ScaTra couplings with COUPID {} needs "
            "to be the same!",
            ID);
      if (fluid_permcoeffs->at(2) != structure_permcoeffs->at(2))
        FOUR_C_THROW(
            "Filtration coefficient coefficient FILTR of ScaTra couplings with COUPID {} needs to "
            "be the same!",
            ID);
      if (fluid_permcoeffs->at(2) < 0 or fluid_permcoeffs->at(2) > 1)
        FOUR_C_THROW(
            "The filtration coefficient FILTR of ScaTra couplings with COUPID {} must be in [0;1], "
            "since it is the ratio of average pore size per area!",
            ID);
      if (fluid_permcoeffs->at(3) != structure_permcoeffs->at(3))
        FOUR_C_THROW(
            "WSS onoff flag WSSON of ScaTra couplings with COUPID {} needs to be the same!", ID);
      if (fluid_permcoeffs->at(4) != structure_permcoeffs->at(4))
        FOUR_C_THROW(
            "First WSS coefficient WSSCOEFFS of ScaTra couplings with COUPID {} needs to be the "
            "same!",
            ID);
      if (fluid_permcoeffs->at(5) != structure_permcoeffs->at(5))
        FOUR_C_THROW(
            "Second WSS coefficient WSSCOEFFS of ScaTra couplings with COUPID {} needs to be the "
            "same!",
            ID);
      if (fluid_permcoeffs->at(6) != structure_permcoeffs->at(6))
        FOUR_C_THROW(
            "Number of scalars NUMSCAL of ScaTra couplings with COUPID {} needs to be the same!",
            ID);

      for (int k = 0; k < numscal; k++)
      {
        if (fluid_permcoeffs->at(7 + k) != structure_permcoeffs->at(7 + k))
          FOUR_C_THROW("ONOFF vector of ScaTra couplings with COUPID {} needs to be the same!", ID);

        onoff_sum->at(k) += fluid_permcoeffs->at(7 + k);
      }
    }

    for (int j = 0; j < numscal; j++)
    {
      if (onoff_sum->at(j) > 1)
        FOUR_C_THROW(
            "In the ONOFF vector the {}-th scalar has been switched on multiple times. The ON is "
            "allowed only once per scalar!",
            j);
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::scatra_output()
{
  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra = scatravec_[i];
    scatra->scatra_field()->check_and_write_output_and_restart();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::increment_time_and_step()
{
  step_ += 1;
  time_ += dt_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::update_scatra_fields()
{
  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra = scatravec_[i];
    scatra->scatra_field()->update();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::scatra_evaluate_solve_iter_update()
{
  evaluate_scatra_fields();
  setup_coupled_scatra_system();
  linear_solve_scatra();
  scatra_iter_update();

  // generalized-alpha time integration: compute intermediate values
  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra = scatravec_[i];
    scatra->scatra_field()->compute_intermediate_values();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::evaluate_scatra_fields()
{
  // membrane concentration at the interface needed for simplified membrane equation of Kedem and
  // Katchalsky. NOTE: needs to be set here, since it depends on the scalar interface values on both
  // discretisations changing with each Newton iteration
  set_membrane_concentration();

  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    std::shared_ptr<ScaTra::ScaTraTimIntImpl> scatra = scatravec_[i]->scatra_field();

    // evaluate scatra field
    scatra->prepare_linear_solve();
    // add contributions due to finite interface permeability
    if (!infperm_)
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> coupforce = scatracoupforce_[i];
      std::shared_ptr<Core::LinAlg::SparseMatrix> coupmat = scatracoupmat_[i];

      coupforce->put_scalar(0.0);
      coupmat->zero();

      scatra->surface_permeability(coupmat, coupforce);

      // apply Dirichlet boundary conditions to coupling matrix and vector
      std::shared_ptr<Core::LinAlg::Vector<double>> zeros = scatrazeros_[i];
      const std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmapex = scatra->dirich_maps();
      const std::shared_ptr<const Core::LinAlg::Map> dbcmap = dbcmapex->cond_map();
      coupmat->apply_dirichlet(*dbcmap, false);
      Core::LinAlg::apply_dirichlet_to_system(*coupforce, *zeros, *dbcmap);
    }
  }
}

/*----------------------------------------------------------------------*
 |  Set Membrane concentration in scatra fields              Thon 08/16 |
 *----------------------------------------------------------------------*/
void FS3I::FS3IBase::set_membrane_concentration() const
{
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> MembraneConc;
  extract_membrane_concentration(MembraneConc);

  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra = scatravec_[i];
    scatra->scatra_field()->set_membrane_concentration(MembraneConc[i]);
  }
}

/*----------------------------------------------------------------------*
 |  Extract membrane concentration                           thon 08/16 |
 *----------------------------------------------------------------------*/
void FS3I::FS3IBase::extract_membrane_concentration(
    std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>& MembraneConcentration) const
{
  // ############ Fluid Field ###############
  std::shared_ptr<Core::LinAlg::Vector<double>> MembraneConcentration1 =
      calc_membrane_concentration();
  MembraneConcentration.push_back(MembraneConcentration1);

  // ############ Poro Field ###############
  //  Hint: The mean concentration is not calculated again; we just map the values from the
  //  Fluid-Scatra Field into the Structure-Scatra Field

  // extract interface values
  std::shared_ptr<Core::LinAlg::Vector<double>> interface_phin =
      scatrafieldexvec_[0]->extract_vector(*MembraneConcentration1, 1);

  // insert interface values from Fluid Field into Poro Field;
  std::shared_ptr<Core::LinAlg::Vector<double>> MembraneConcentration2 =
      scatrafieldexvec_[1]->insert_vector(*scatra1_to_scatra2(*interface_phin), 1);
  MembraneConcentration.push_back(MembraneConcentration2);
}

/*----------------------------------------------------------------------*
 |  Calculate membrane concentration                         thon 08/16 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FS3I::FS3IBase::calc_membrane_concentration() const
{
  // Get concentration phi2 in scatrafield2
  // Hint: in the following we talk of phi1 and phi2, but they mean the same concentration just on
  // different scatrafields
  std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra2 = scatravec_[1];
  std::shared_ptr<Core::LinAlg::Vector<double>> scatrafield2_phi2np =
      scatra2->scatra_field()->phinp();

  // extract interface values from phi2 but we are still on scatrafield2
  std::shared_ptr<Core::LinAlg::Vector<double>> interface2_phi2np =
      scatrafieldexvec_[1]->extract_vector(*scatrafield2_phi2np, 1);

  // insert interface values from scatrafield2 into scatrafield1; scatrafield1_phi2n is again of
  // full length, i.e. of size of scatrafield1; all values that do not belong to the interface are
  // zero
  std::shared_ptr<Core::LinAlg::Vector<double>> scatrafield1_phi2np =
      scatrafieldexvec_[0]->insert_vector(*scatra2_to_scatra1(*interface2_phi2np), 1);

  // Get concentration phi1 in scatrafield1
  std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra1 = scatravec_[0];
  std::shared_ptr<Core::LinAlg::Vector<double>> scatrafield1_phi1np =
      scatra1->scatra_field()->phinp();

  // extract interface values from phi1 but we are still on scatrafield1
  std::shared_ptr<Core::LinAlg::Vector<double>> interface1_phi1np =
      scatrafieldexvec_[0]->extract_vector(*scatrafield1_phi1np, 1);

  // insert interface values interface1_phi1n from scatrafield1 into the full scatrafield1 again;
  // this is just to obtain a vector whose entries are zero except for the nodes of the interface
  std::shared_ptr<Core::LinAlg::Vector<double>> temp =
      scatrafieldexvec_[0]->insert_vector(*interface1_phi1np, 1);

  // nodewise calculation of mean concentration in the interface

  for (int i = 0; i < temp->local_length(); i++)
  {
    // here the unweighted average is uses. One could also use a logarithmic average...
    (*temp).get_values()[i] =
        0.5 *
        (temp->local_values_as_span()[i] +
            scatrafield1_phi2np->local_values_as_span()
                [i]);  // log. average:
                       // ((*temp)[i]-(*scatrafield1_phi2n)[i])/log(((*temp)[i])/((*scatrafield1_phi2n)[i]));
                       // linear approach: 0.5*((*temp)[i]+(*scatrafield1_phi2n)[i]);
  }

  // return mean concentration in the interface
  // this vector now belongs to scatrafield1!!!
  return temp;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::setup_coupled_scatra_system()
{
  // set up scatra rhs
  setup_coupled_scatra_rhs();

  // set up scatra system matrix
  setup_coupled_scatra_matrix();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::setup_coupled_scatra_rhs()
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> scatra1 =
      scatravec_[0]->scatra_field()->residual();
  std::shared_ptr<const Core::LinAlg::Vector<double>> scatra2 =
      scatravec_[1]->scatra_field()->residual();
  setup_coupled_scatra_vector(*scatrarhs_, *scatra1, *scatra2);

  // additional contributions in case of finite interface permeability
  if (!infperm_)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> coup1 = scatracoupforce_[0];
    std::shared_ptr<Core::LinAlg::Vector<double>> coup2 = scatracoupforce_[1];

    // contribution of the same field
    scatraglobalex_->add_vector(*coup1, 0, *scatrarhs_, 1.0);
    scatraglobalex_->add_vector(*coup2, 1, *scatrarhs_, 1.0);

    // contribution of the respective other field
    std::shared_ptr<Core::LinAlg::Vector<double>> coup1_boundary =
        scatrafieldexvec_[0]->extract_vector(*coup1, 1);
    std::shared_ptr<Core::LinAlg::Vector<double>> temp =
        scatrafieldexvec_[1]->insert_vector(*scatra1_to_scatra2(*coup1_boundary), 1);
    temp->scale(-1.0);
    scatraglobalex_->add_vector(*temp, 1, *scatrarhs_);

    std::shared_ptr<Core::LinAlg::Vector<double>> coup2_boundary =
        scatrafieldexvec_[1]->extract_vector(*coup2, 1);
    temp = scatrafieldexvec_[0]->insert_vector(*scatra2_to_scatra1(*coup2_boundary), 1);
    temp->scale(-1.0);
    scatraglobalex_->add_vector(*temp, 0, *scatrarhs_);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::setup_coupled_scatra_vector(Core::LinAlg::Vector<double>& globalvec,
    const Core::LinAlg::Vector<double>& vec1, const Core::LinAlg::Vector<double>& vec2)
{
  if (infperm_)
  {
    // concentrations are assumed to be equal at the interface
    // extract the inner (uncoupled) dofs from second field
    std::shared_ptr<Core::LinAlg::Vector<double>> vec2_other =
        scatrafieldexvec_[1]->extract_vector(vec2, 0);

    std::shared_ptr<Core::LinAlg::Vector<double>> vec2_boundary =
        scatrafieldexvec_[1]->extract_vector(vec2, 1);
    std::shared_ptr<Core::LinAlg::Vector<double>> temp =
        scatrafieldexvec_[0]->insert_vector(*scatra2_to_scatra1(*vec2_boundary), 1);
    temp->update(1.0, vec1, 1.0);

    scatraglobalex_->insert_vector(*temp, 0, globalvec);
    scatraglobalex_->insert_vector(*vec2_other, 1, globalvec);
  }
  else
  {
    scatraglobalex_->insert_vector(vec1, 0, globalvec);
    scatraglobalex_->insert_vector(vec2, 1, globalvec);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::setup_coupled_scatra_matrix() const
{
  auto scatra1 = scatravec_[0]->scatra_field()->system_matrix();
  auto scatra2 = scatravec_[1]->scatra_field()->system_matrix();

  if (scatra1 == nullptr) FOUR_C_THROW("expect fluid scatra block matrix");
  if (scatra2 == nullptr) FOUR_C_THROW("expect structure scatra block matrix");

  if (infperm_)
  {
    // Incomplete system matrix to be able to deal with slightly defective interface meshes.
    scatra1->un_complete();

    // structure scatra
    // first split the matrix into 2x2 blocks (boundary vs. inner dofs)
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockscatra2 =
        Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
            *scatra2, *(scatrafieldexvec_[1]), *(scatrafieldexvec_[1]));
    blockscatra2->complete();

    scatrasystemmatrix_->assign(1, 1, Core::LinAlg::DataAccess::Share, blockscatra2->matrix(0, 0));

    (*sibtransform_)(blockscatra2->full_row_map(), blockscatra2->full_col_map(),
        blockscatra2->matrix(0, 1), 1.0, Coupling::Adapter::CouplingSlaveConverter(*scatracoup_),
        scatrasystemmatrix_->matrix(1, 0));
    (*sbitransform_)(blockscatra2->matrix(1, 0), 1.0,
        Coupling::Adapter::CouplingSlaveConverter(*scatracoup_), scatrasystemmatrix_->matrix(0, 1));
    (*sbbtransform_)(blockscatra2->matrix(1, 1), 1.0,
        Coupling::Adapter::CouplingSlaveConverter(*scatracoup_),
        Coupling::Adapter::CouplingSlaveConverter(*scatracoup_), *scatra1, true, true);

    // fluid scatra
    scatrasystemmatrix_->assign(0, 0, Core::LinAlg::DataAccess::Share, *scatra1);
  }
  else
  {
    // conventional contributions
    scatrasystemmatrix_->assign(0, 0, Core::LinAlg::DataAccess::Share, *scatra1);
    scatrasystemmatrix_->assign(1, 1, Core::LinAlg::DataAccess::Share, *scatra2);

    // additional contributions due to interface permeability (-> coupling terms)
    // contribution of the same field
    std::shared_ptr<Core::LinAlg::SparseMatrix> coup1 = scatracoupmat_[0];
    std::shared_ptr<Core::LinAlg::SparseMatrix> coup2 = scatracoupmat_[1];

    Core::LinAlg::matrix_add(*coup1, false, 1.0, scatrasystemmatrix_->matrix(0, 0), 1.0);
    Core::LinAlg::matrix_add(*coup2, false, 1.0, scatrasystemmatrix_->matrix(1, 1), 1.0);

    // contribution of the respective other field
    // first split the matrix into 2x2 blocks (boundary vs. inner dofs)
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> coupblock1 =
        Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
            *coup1, *(scatrafieldexvec_[0]), *(scatrafieldexvec_[0]));
    coupblock1->complete();
    (*fbitransform_)(coupblock1->matrix(1, 1), -1.0,
        Coupling::Adapter::CouplingMasterConverter(*scatracoup_),
        scatrasystemmatrix_->matrix(1, 0));

    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> coupblock2 =
        Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
            *coup2, *(scatrafieldexvec_[1]), *(scatrafieldexvec_[1]));
    coupblock2->complete();
    (*sbitransform_)(coupblock2->matrix(1, 1), -1.0,
        Coupling::Adapter::CouplingSlaveConverter(*scatracoup_), scatrasystemmatrix_->matrix(0, 1));
  }

  scatrasystemmatrix_->complete();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FS3I::FS3IBase::scatra2_to_scatra1(
    const Core::LinAlg::Vector<double>& iv) const
{
  return scatracoup_->slave_to_master(iv);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FS3I::FS3IBase::scatra1_to_scatra2(
    const Core::LinAlg::Vector<double>& iv) const
{
  return scatracoup_->master_to_slave(iv);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::linear_solve_scatra()
{
  scatraincrement_->put_scalar(0.0);

  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  scatrasolver_->solve(scatrasystemmatrix_, scatraincrement_, scatrarhs_, solver_params);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::scatra_iter_update()
{
  // define incremental vectors for fluid- and structure-based scatra
  // fields and extract respective vectors
  std::shared_ptr<const Core::LinAlg::Vector<double>> inc1;
  std::shared_ptr<const Core::LinAlg::Vector<double>> inc2;
  extract_scatra_field_vectors(*scatraincrement_, inc1, inc2);

  // update both fluid- and structure-based solution vectors
  scatravec_[0]->scatra_field()->update_iter(*inc1);
  scatravec_[1]->scatra_field()->update_iter(*inc2);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::extract_scatra_field_vectors(const Core::LinAlg::Vector<double>& globalvec,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& vec1,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& vec2)
{
  if (infperm_)
  {
    // process fluid scatra unknowns
    vec1 = scatraglobalex_->extract_vector(globalvec, 0);

    // process structure scatra unknowns at the boundary
    std::shared_ptr<Core::LinAlg::Vector<double>> vec1_boundary =
        scatrafieldexvec_[0]->extract_vector(*vec1, 1);
    std::shared_ptr<const Core::LinAlg::Vector<double>> vec2_inner =
        scatraglobalex_->extract_vector(globalvec, 1);
    std::shared_ptr<Core::LinAlg::Vector<double>> vec2_boundary =
        scatra1_to_scatra2(*vec1_boundary);

    std::shared_ptr<Core::LinAlg::Vector<double>> vec2_temp =
        scatrafieldexvec_[1]->insert_vector(*vec2_inner, 0);
    scatrafieldexvec_[1]->insert_vector(*vec2_boundary, 1, *vec2_temp);
    vec2 = vec2_temp;
  }
  else
  {
    vec1 = scatraglobalex_->extract_vector(globalvec, 0);
    vec2 = scatraglobalex_->extract_vector(globalvec, 1);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::check_is_setup()
{
  if (not is_setup()) FOUR_C_THROW("setup() was not called.");
};

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::FS3IBase::check_is_init()
{
  if (not is_init()) FOUR_C_THROW("init(...) was not called.");
};

FOUR_C_NAMESPACE_CLOSE

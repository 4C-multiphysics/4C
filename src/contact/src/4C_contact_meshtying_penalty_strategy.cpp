// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_meshtying_penalty_strategy.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_contact_input.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mortar_interface.hpp"
#include "4C_mortar_utils.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | ctor (public)                                              popp 05/09|
 *----------------------------------------------------------------------*/
CONTACT::MtPenaltyStrategy::MtPenaltyStrategy(const Core::LinAlg::Map* dof_row_map,
    const Core::LinAlg::Map* NodeRowMap, Teuchos::ParameterList params,
    std::vector<std::shared_ptr<Mortar::Interface>> interface, const int spatialDim,
    const MPI_Comm& comm, const double alphaf, const int maxdof)
    : MtAbstractStrategy(
          dof_row_map, NodeRowMap, params, interface, spatialDim, comm, alphaf, maxdof)
{
  // initialize constraint norm and initial penalty
  constrnorm_ = 0.0;
  initialpenalty_ = MtPenaltyStrategy::params().get<double>("PENALTYPARAM");
}

/*----------------------------------------------------------------------*
 |  do mortar coupling in reference configuration             popp 12/09|
 *----------------------------------------------------------------------*/
void CONTACT::MtPenaltyStrategy::mortar_coupling(
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& dis)
{
  TEUCHOS_FUNC_TIME_MONITOR("CONTACT::MtPenaltyStrategy::mortar_coupling");

  // print message
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "Performing mortar coupling...............";
    fflush(stdout);
  }

  // time measurement
  Core::Communication::barrier(get_comm());
  const double t_start = Teuchos::Time::wallTime();

  // refer call to parent class
  MtAbstractStrategy::mortar_coupling(dis);

  //----------------------------------------------------------------------
  // CHECK IF WE NEED TRANSFORMATION MATRICES FOR SLAVE DISPLACEMENT DOFS
  //----------------------------------------------------------------------
  // Concretely, we apply the following transformations:
  // D         ---->   D * T^(-1)
  // D^(-1)    ---->   T * D^(-1)
  // These modifications are applied once right here, thus the
  // following code (evaluate_meshtying) remains unchanged.
  //----------------------------------------------------------------------
  if (dualquadslavetrafo())
  {
    // type of LM interpolation for quadratic elements
    auto lagmultquad = Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(params(), "LM_QUAD");

    if (lagmultquad != Inpar::Mortar::lagmult_lin)
    {
      // modify dmatrix_
      std::shared_ptr<Core::LinAlg::SparseMatrix> temp1 =
          Core::LinAlg::matrix_multiply(*dmatrix_, false, *invtrafo_, false, false, false, true);
      dmatrix_ = temp1;
    }
    else
      FOUR_C_THROW(
          "Locally linear LM interpolation is not yet implemented for meshtying with penalty "
          "strategy!");
  }

  // build mortar matrix products
  mtm_ = Core::LinAlg::matrix_multiply(*mmatrix_, true, *mmatrix_, false, false, false, true);
  mtd_ = Core::LinAlg::matrix_multiply(*mmatrix_, true, *dmatrix_, false, false, false, true);
  dtm_ = Core::LinAlg::matrix_multiply(*dmatrix_, true, *mmatrix_, false, false, false, true);
  dtd_ = Core::LinAlg::matrix_multiply(*dmatrix_, true, *dmatrix_, false, false, false, true);

  // transform rows of mortar matrix products to parallel distribution
  // of the global problem (stored in the "p"-version of dof maps)
  if (par_redist())
  {
    mtm_ = Core::LinAlg::matrix_row_transform(*mtm_, *non_redist_gmdofrowmap_);
    mtd_ = Core::LinAlg::matrix_row_transform(*mtd_, *non_redist_gmdofrowmap_);
    dtm_ = Core::LinAlg::matrix_row_transform(*dtm_, *non_redist_gsdofrowmap_);
    dtd_ = Core::LinAlg::matrix_row_transform(*dtd_, *non_redist_gsdofrowmap_);
  }

  // full stiffness matrix
  stiff_ = std::make_shared<Core::LinAlg::SparseMatrix>(*problem_dofs(), 100, false, true);
  double pp = params().get<double>("PENALTYPARAM");

  // add penalty meshtying stiffness terms
  stiff_->add(*mtm_, false, pp, 1.0);
  stiff_->add(*mtd_, false, -pp, 1.0);
  stiff_->add(*dtm_, false, -pp, 1.0);
  stiff_->add(*dtd_, false, pp, 1.0);
  stiff_->complete();

  // time measurement
  Core::Communication::barrier(get_comm());
  const double t_end = Teuchos::Time::wallTime() - t_start;
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "in...." << std::scientific << std::setprecision(6) << t_end << " secs"
              << std::endl;
}

/*----------------------------------------------------------------------*
 |  mesh initialization for rotational invariance              popp 12/09|
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
CONTACT::MtPenaltyStrategy::mesh_initialization()
{
  TEUCHOS_FUNC_TIME_MONITOR("CONTACT::MtPenaltyStrategy::mesh_initialization");

  // get out of here is NTS algorithm is activated
  if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(params(), "ALGORITHM") ==
      Inpar::Mortar::algorithm_nts)
    return nullptr;

  // print message
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "Performing mesh initialization..........." << std::endl;
  }

  // time measurement
  Core::Communication::barrier(get_comm());
  const double t_start = Teuchos::Time::wallTime();

  //**********************************************************************
  // (1) get master positions on global level
  //**********************************************************************
  // fill Xmaster first
  std::shared_ptr<Core::LinAlg::Vector<double>> Xmaster =
      Core::LinAlg::create_vector(*gmdofrowmap_, true);
  assemble_coords("master", true, *Xmaster);

  //**********************************************************************
  // (2) solve for modified slave positions on global level
  //**********************************************************************
  // create linear problem
  std::shared_ptr<Core::LinAlg::Vector<double>> Xslavemod =
      Core::LinAlg::create_vector(*gsdofrowmap_, true);
  std::shared_ptr<Core::LinAlg::Vector<double>> rhs =
      Core::LinAlg::create_vector(*gsdofrowmap_, true);
  mmatrix_->multiply(false, *Xmaster, *rhs);

  // solve with default solver
  Teuchos::ParameterList solvparams;
  Core::Utils::add_enum_class_to_parameter_list<Core::LinearSolver::SolverType>(
      "SOLVER", Core::LinearSolver::SolverType::umfpack, solvparams);
  Core::LinAlg::Solver solver(solvparams, get_comm(), nullptr, Core::IO::Verbositylevel::standard);

  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver.solve(dmatrix_, Xslavemod, rhs, solver_params);

  //**********************************************************************
  // (3) perform mesh initialization node by node
  //**********************************************************************
  // this can be done in the AbstractStrategy now
  MtAbstractStrategy::mesh_initialization(Xslavemod);

  // time measurement
  Core::Communication::barrier(get_comm());
  const double t_end = Teuchos::Time::wallTime() - t_start;
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "in...." << std::scientific << std::setprecision(6) << t_end << " secs"
              << std::endl;
  }

  // return xslavemod for global problem
  return Xslavemod;
}

/*----------------------------------------------------------------------*
 | evaluate meshtying and create linear system                popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::MtPenaltyStrategy::evaluate_meshtying(
    std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
    std::shared_ptr<Core::LinAlg::Vector<double>>& feff,
    std::shared_ptr<Core::LinAlg::Vector<double>> dis)
{
  // since we will modify the graph of kteff by adding additional
  // meshtyong stiffness entries, we have to uncomplete it
  kteff->un_complete();

  /**********************************************************************/
  /* Global setup of kteff, feff (including meshtying)                  */
  /**********************************************************************/
  double pp = params().get<double>("PENALTYPARAM");

  // add penalty meshtying stiffness terms
  kteff->add(*mtm_, false, pp, 1.0);
  kteff->add(*mtd_, false, -pp, 1.0);
  kteff->add(*dtm_, false, -pp, 1.0);
  kteff->add(*dtd_, false, pp, 1.0);

  //***************************************************************************
  // build constraint vector
  //***************************************************************************
  // Since we enforce the meshtying constraint for the displacements u,
  // and not for the configurations x (which would also be possible in theory),
  // we avoid artificial initial stresses (+), but we might not guarantee
  // exact rotational invariance (-). However, since we always apply the
  // so-called mesh initialization procedure, we can then also guarantee
  // exact rotational invariance (+).
  //***************************************************************************
  Core::LinAlg::Vector<double> tempvec1(*gsdofrowmap_);
  Core::LinAlg::Vector<double> tempvec2(*gsdofrowmap_);
  Core::LinAlg::export_to(*dis, tempvec1);
  dmatrix_->multiply(false, tempvec1, tempvec2);
  g_->update(-1.0, tempvec2, 0.0);

  Core::LinAlg::Vector<double> tempvec3(*gmdofrowmap_);
  Core::LinAlg::Vector<double> tempvec4(*gsdofrowmap_);
  Core::LinAlg::export_to(*dis, tempvec3);
  mmatrix_->multiply(false, tempvec3, tempvec4);
  g_->update(1.0, tempvec4, 1.0);

  // update LM vector
  // (in the pure penalty case, zuzawa is zero)
  z_->update(1.0, *zuzawa_, 0.0);
  z_->update(-pp, *g_, 1.0);

  // store updated LM into nodes
  store_nodal_quantities(Mortar::StrategyBase::lmupdate);

  // add penalty meshtying force terms
  Core::LinAlg::Vector<double> fm(*gmdofrowmap_);
  mmatrix_->multiply(true, *z_, fm);
  Core::LinAlg::Vector<double> fmexp(*problem_dofs());
  Core::LinAlg::export_to(fm, fmexp);
  feff->update(1.0, fmexp, 1.0);

  Core::LinAlg::Vector<double> fs(*gsdofrowmap_);
  dmatrix_->multiply(true, *z_, fs);
  Core::LinAlg::Vector<double> fsexp(*problem_dofs());
  Core::LinAlg::export_to(fs, fsexp);
  feff->update(-1.0, fsexp, 1.0);

  // add old contact forces (t_n)
  Core::LinAlg::Vector<double> fsold(*gsdofrowmap_);
  dmatrix_->multiply(true, *zold_, fsold);
  Core::LinAlg::Vector<double> fsoldexp(*problem_dofs());
  Core::LinAlg::export_to(fsold, fsoldexp);
  feff->update(alphaf_, fsoldexp, 1.0);

  Core::LinAlg::Vector<double> fmold(*gmdofrowmap_);
  mmatrix_->multiply(true, *zold_, fmold);
  Core::LinAlg::Vector<double> fmoldexp(*problem_dofs());
  Core::LinAlg::export_to(fmold, fmoldexp);
  feff->update(-alphaf_, fmoldexp, 1.0);
}

/*----------------------------------------------------------------------*
 | initialize Uzawa step 2,3...                                popp 12/09|
 *----------------------------------------------------------------------*/
void CONTACT::MtPenaltyStrategy::initialize_uzawa(
    std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
    std::shared_ptr<Core::LinAlg::Vector<double>>& feff)
{
  // remove penalty meshtying force terms
  Core::LinAlg::Vector<double> fm(*gmdofrowmap_);
  mmatrix_->multiply(true, *z_, fm);
  Core::LinAlg::Vector<double> fmexp(*problem_dofs());
  Core::LinAlg::export_to(fm, fmexp);
  feff->update(-1.0, fmexp, 1.0);

  Core::LinAlg::Vector<double> fs(*gsdofrowmap_);
  dmatrix_->multiply(false, *z_, fs);
  Core::LinAlg::Vector<double> fsexp(*problem_dofs());
  Core::LinAlg::export_to(fs, fsexp);
  feff->update(1.0, fsexp, 1.0);

  // update LM vector
  double pp = params().get<double>("PENALTYPARAM");
  z_->update(1.0, *zuzawa_, 0.0);
  z_->update(-pp, *g_, 1.0);

  // add penalty meshtying force terms
  Core::LinAlg::Vector<double> fmnew(*gmdofrowmap_);
  mmatrix_->multiply(true, *z_, fmnew);
  Core::LinAlg::Vector<double> fmexpnew(*problem_dofs());
  Core::LinAlg::export_to(fmnew, fmexpnew);
  feff->update(1.0, fmexpnew, 1.0);

  Core::LinAlg::Vector<double> fsnew(*gsdofrowmap_);
  dmatrix_->multiply(false, *z_, fsnew);
  Core::LinAlg::Vector<double> fsexpnew(*problem_dofs());
  Core::LinAlg::export_to(fsnew, fsexpnew);
  feff->update(-1.0, fsexpnew, 1.0);
}

/*----------------------------------------------------------------------*
 | reset penalty parameter to initial value                    popp 08/09|
 *----------------------------------------------------------------------*/
void CONTACT::MtPenaltyStrategy::reset_penalty()
{
  // reset penalty parameter in strategy
  params().set<double>("PENALTYPARAM", initial_penalty());


  // reset penalty parameter in all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->interface_params().set<double>("PENALTYPARAM", initial_penalty());
  }
}

/*----------------------------------------------------------------------*
 | modify penalty parameter to initial value                    mhv 03/16|
 *----------------------------------------------------------------------*/
void CONTACT::MtPenaltyStrategy::modify_penalty()
{
  // generate random number between 0.95 and 1.05
  double randnum = ((double)rand() / (double)RAND_MAX) * 0.1 + 0.95;
  double pennew = randnum * initial_penalty();

  // modify penalty parameter in strategy
  params().set<double>("PENALTYPARAM", pennew);

  // modify penalty parameter in all interfaces
  for (int i = 0; i < (int)interface_.size(); ++i)
  {
    interface_[i]->interface_params().set<double>("PENALTYPARAM", pennew);
  }
}

/*----------------------------------------------------------------------*
 | evaluate L2-norm of active constraints                     popp 08/09|
 *----------------------------------------------------------------------*/
void CONTACT::MtPenaltyStrategy::update_constraint_norm(int uzawaiter)
{
  // initialize parameters
  double cnorm = 0.0;
  bool updatepenalty = false;
  double ppcurr = params().get<double>("PENALTYPARAM");

  // compute constraint norm
  g_->norm_2(&cnorm);

  //********************************************************************
  // adaptive update of penalty parameter
  // (only for Uzawa Augmented Lagrange strategy)
  //********************************************************************
  auto soltype = Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(params(), "STRATEGY");

  if (soltype == CONTACT::SolvingStrategy::uzawa)
  {
    // check convergence of cnorm and update penalty parameter
    // only do this for second, third, ... Uzawa iteration
    // cf. Wriggers, Computational Contact Mechanics, 2nd edition (2006), p. 340
    if ((uzawaiter >= 2) && (cnorm > 0.25 * constraint_norm()))
    {
      updatepenalty = true;

      // update penalty parameter in strategy
      params().set<double>("PENALTYPARAM", 10 * ppcurr);

      // update penalty parameter in all interfaces
      for (int i = 0; i < (int)interface_.size(); ++i)
      {
        double ippcurr = interface_[i]->interface_params().get<double>("PENALTYPARAM");
        if (ippcurr != ppcurr) FOUR_C_THROW("Something wrong with penalty parameter");
        interface_[i]->interface_params().set<double>("PENALTYPARAM", 10 * ippcurr);
      }
    }
  }
  //********************************************************************

  // update constraint norm
  constrnorm_ = cnorm;

  // output to screen
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "********************************************\n";
    std::cout << "Constraint Norm: " << cnorm << "\n";
    if (updatepenalty)
      std::cout << "Updated penalty parameter: " << ppcurr << " -> "
                << params().get<double>("PENALTYPARAM") << "\n";
    std::cout << "********************************************\n";
  }
}

/*----------------------------------------------------------------------*
 | store Lagrange multipliers for next Uzawa step             popp 08/09|
 *----------------------------------------------------------------------*/
void CONTACT::MtPenaltyStrategy::update_uzawa_augmented_lagrange()
{
  // store current LM into Uzawa LM
  // (note that this is also done after the last Uzawa step of one
  // time step and thus also gives the guess for the initial
  // Lagrange multiplier lambda_0 of the next time step)
  zuzawa_ = std::make_shared<Core::LinAlg::Vector<double>>(*z_);
  store_nodal_quantities(Mortar::StrategyBase::lmuzawa);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool CONTACT::MtPenaltyStrategy::evaluate_force(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  if (!force_) force_ = std::make_shared<Core::LinAlg::Vector<double>>(*problem_dofs());
  stiff_->multiply(false, *dis, *force_);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool CONTACT::MtPenaltyStrategy::evaluate_stiff(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool CONTACT::MtPenaltyStrategy::evaluate_force_stiff(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dis)
{
  bool successForce = evaluate_force(dis);
  bool successStiff = evaluate_stiff(dis);

  return (successForce && successStiff);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> CONTACT::MtPenaltyStrategy::get_rhs_block_ptr(
    const enum CONTACT::VecBlockType& bt) const
{
  std::shared_ptr<Core::LinAlg::Vector<double>> vec_ptr = nullptr;
  switch (bt)
  {
    case CONTACT::VecBlockType::displ:
    {
      if (!force_) FOUR_C_THROW("force vector not available");
      vec_ptr = force_;
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown Solid::VecBlockType!");
      break;
    }
  }

  return vec_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> CONTACT::MtPenaltyStrategy::get_matrix_block_ptr(
    const enum CONTACT::MatBlockType& bt) const
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> mat_ptr = nullptr;
  switch (bt)
  {
    case CONTACT::MatBlockType::displ_displ:
      if (!stiff_) FOUR_C_THROW("stiffness not available");
      mat_ptr = stiff_;
      break;
    default:
      FOUR_C_THROW("Unknown Solid::MatBlockType!");
      break;
  }
  return mat_ptr;
}

FOUR_C_NAMESPACE_CLOSE

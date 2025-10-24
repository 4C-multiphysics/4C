// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_coupling_adapter_volmortar.hpp"

#include "4C_coupling_volmortar.hpp"
#include "4C_coupling_volmortar_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_rebalance_binning_based.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <utility>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor                                                     farah 10/13|
 *----------------------------------------------------------------------*/
Coupling::Adapter::MortarVolCoupl::MortarVolCoupl()
    : issetup_(false),
      isinit_(false),
      p12_(nullptr),
      p21_(nullptr),
      masterdis_(nullptr),
      slavedis_(nullptr),
      coupleddof12_(nullptr),
      coupleddof21_(nullptr),
      dofsets12_(nullptr),
      dofsets21_(nullptr),
      materialstrategy_(nullptr)
{
  // empty...
}


/*----------------------------------------------------------------------*
 |  init                                                     farah 10/13|
 *----------------------------------------------------------------------*/
void Coupling::Adapter::MortarVolCoupl::init(int spatial_dimension,
    std::shared_ptr<Core::FE::Discretization> dis1,  // masterdis - on Omega_1
    std::shared_ptr<Core::FE::Discretization> dis2,  // slavedis  - on Omega_2
    std::vector<int>* coupleddof12, std::vector<int>* coupleddof21, std::pair<int, int>* dofsets12,
    std::pair<int, int>* dofsets21,
    std::shared_ptr<FourC::Coupling::VolMortar::Utils::DefaultMaterialStrategy> materialstrategy,
    bool createauxdofs)
{
  // Note : We need to make sure that the parallel distribution of discretizations
  //        is the same externally! The best thing is if you do this in your *_dyn.cpp,
  //        i.e., your global control algorithm.

  // reset the setup flag
  issetup_ = false;

  spatial_dimension_ = spatial_dimension;

  // set pointers to discretizations
  masterdis_ = dis1;
  slavedis_ = dis2;

  // set various pointers
  coupleddof12_ = coupleddof12;
  coupleddof21_ = coupleddof21;
  dofsets12_ = dofsets12;
  dofsets21_ = dofsets21;
  materialstrategy_ = materialstrategy;

  if ((dis1->num_dof_sets() == 1) and (dis2->num_dof_sets() == 1) and createauxdofs)
  {
    if (coupleddof12 == nullptr or coupleddof21 == nullptr)
      FOUR_C_THROW("ERROR: No coupling dofs for volmortar algorithm specified!");

    create_aux_dofsets(*dis1, *dis2, coupleddof12, coupleddof21);
  }

  // set flag
  isinit_ = true;
}


/*----------------------------------------------------------------------*
 |  setup                                                    rauch 08/16|
 *----------------------------------------------------------------------*/
void Coupling::Adapter::MortarVolCoupl::setup(
    const Teuchos::ParameterList& params, const Teuchos::ParameterList& cut_params)
{
  check_init();

  // create material strategy
  if (!materialstrategy_)
    materialstrategy_ =
        std::make_shared<FourC::Coupling::VolMortar::Utils::DefaultMaterialStrategy>();

  // create coupling instance
  std::shared_ptr<FourC::Coupling::VolMortar::VolMortarCoupl> coupdis =
      std::make_shared<FourC::Coupling::VolMortar::VolMortarCoupl>(spatial_dimension_, masterdis_,
          slavedis_, params, cut_params, coupleddof12_, coupleddof21_, dofsets12_, dofsets21_,
          materialstrategy_);

  //-----------------------
  // Evaluate volmortar coupling:
  if (Teuchos::getIntegralValue<FourC::Coupling::VolMortar::CouplingType>(params, "COUPLINGTYPE") ==
      FourC::Coupling::VolMortar::couplingtype_volmortar)
    coupdis->evaluate_volmortar();
  //-----------------------
  // consistent interpolation (NO Core::VOLMORTAR)
  else if (Teuchos::getIntegralValue<FourC::Coupling::VolMortar::CouplingType>(
               params, "COUPLINGTYPE") == FourC::Coupling::VolMortar::couplingtype_coninter)
    coupdis->evaluate_consistent_interpolation();
  //-----------------------
  else
    FOUR_C_THROW("ERROR: Chosen coupling not implemented!!!");

  // get the P operators
  p12_ = coupdis->get_p_matrix12();
  p21_ = coupdis->get_p_matrix21();

  /***********************************************************
   * Assign materials                                        *
   ***********************************************************/
  // assign materials from one discretization to the other
  coupdis->assign_materials();

  // validate flag issetup_
  issetup_ = true;
}


/*----------------------------------------------------------------------*
 |  redistribute                                             rauch 08/16|
 *----------------------------------------------------------------------*/
void Coupling::Adapter::MortarVolCoupl::redistribute(const Teuchos::ParameterList& binning_params,
    std::shared_ptr<Core::IO::OutputControl> output_control,
    std::function<const Core::Nodes::Node&(const Core::Nodes::Node& node)> correct_node,
    std::function<std::vector<std::array<double, 3>>(const Core::FE::Discretization&,
        const Core::Elements::Element&, std::shared_ptr<const Core::LinAlg::Vector<double>> disnp)>
        determine_relevant_points)
{
  check_init();

  // create vector of discr.
  std::vector<std::shared_ptr<Core::FE::Discretization>> dis;
  dis.push_back(masterdis_);
  dis.push_back(slavedis_);

  Core::Rebalance::rebalance_discretizations_by_binning(binning_params, output_control, dis,
      std::move(correct_node), std::move(determine_relevant_points), false);
}


/*----------------------------------------------------------------------*
 |  Create Auxiliary dofsets for multiphysics                farah 06/15|
 *----------------------------------------------------------------------*/
void Coupling::Adapter::MortarVolCoupl::create_aux_dofsets(Core::FE::Discretization& dis1,
    Core::FE::Discretization& dis2, std::vector<int>* coupleddof12, std::vector<int>* coupleddof21)
{
  // first call fill_complete for single discretizations.
  // This way the physical dofs are numbered successively
  dis1.fill_complete();
  dis2.fill_complete();

  // build auxiliary dofsets, i.e. pseudo dofs on each discretization
  // add proxy of velocity related degrees of freedom to scatra discretization
  std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux;
  dofsetaux =
      std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(coupleddof21->size(), 0, 0, true);
  if (dis2.add_dof_set(dofsetaux) != 1) FOUR_C_THROW("unexpected dof sets in fluid field");
  dofsetaux =
      std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(coupleddof12->size(), 0, 0, true);
  if (dis1.add_dof_set(dofsetaux) != 1) FOUR_C_THROW("unexpected dof sets in structure field");

  // call assign_degrees_of_freedom also for auxiliary dofsets
  // note: the order of fill_complete() calls determines the gid numbering!
  // 1. dofs 1
  // 2. dofs 2
  // 3. auxiliary dofs 1
  // 4. auxiliary dofs 2
  dis1.fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });
  dis2.fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });
}

/*----------------------------------------------------------------------*
 |  AssignMaterials                                          vuong 09/14|
 *----------------------------------------------------------------------*/
void Coupling::Adapter::MortarVolCoupl::assign_materials(
    std::shared_ptr<Core::FE::Discretization> dis1, std::shared_ptr<Core::FE::Discretization> dis2,
    const Teuchos::ParameterList& volmortar_params, const Teuchos::ParameterList& cut_params,
    std::shared_ptr<FourC::Coupling::VolMortar::Utils::DefaultMaterialStrategy> materialstrategy)
{
  if (materialstrategy == nullptr)
    materialstrategy =
        std::make_shared<FourC::Coupling::VolMortar::Utils::DefaultMaterialStrategy>();
  // create coupling instance
  FourC::Coupling::VolMortar::VolMortarCoupl coupdis(spatial_dimension_, dis1, dis2,
      volmortar_params, cut_params, nullptr, nullptr, nullptr, nullptr, materialstrategy);

  // assign materials from one discretization to the other
  coupdis.assign_materials();
}

/*----------------------------------------------------------------------*
 |  ApplyMapping from Omega_2 --> Omega_1                    farah 01/14|
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Coupling::Adapter::MortarVolCoupl::apply_vector_mapping12(
    const Core::LinAlg::Vector<double>& vec) const
{
  // safety check
  check_setup();
  check_init();

  std::shared_ptr<Core::LinAlg::Vector<double>> mapvec =
      Core::LinAlg::create_vector(p12_->row_map(), true);
  p12_->multiply(false, vec, *mapvec);

  return mapvec;
}

/*----------------------------------------------------------------------*
 |  ApplyMapping from Omega_1 --> Omega_2                    farah 01/14|
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Coupling::Adapter::MortarVolCoupl::apply_vector_mapping21(
    const Core::LinAlg::Vector<double>& vec) const
{
  // safety check
  check_setup();
  check_init();

  std::shared_ptr<Core::LinAlg::Vector<double>> mapvec =
      Core::LinAlg::create_vector(p21_->row_map(), true);
  p21_->multiply(false, vec, *mapvec);

  return mapvec;
}

/*----------------------------------------------------------------------*
 |  ApplyMapping from Omega_2 --> Omega_1                    farah 01/14|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix>
Coupling::Adapter::MortarVolCoupl::apply_matrix_mapping12(
    const Core::LinAlg::SparseMatrix& mat) const
{
  // safety check
  check_setup();
  check_init();

  return Core::LinAlg::matrix_multiply(mat, false, *p12_, false, false, false, true);
}

/*----------------------------------------------------------------------*
 |  ApplyMapping from Omega_1 --> Omega_2                    farah 01/14|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix>
Coupling::Adapter::MortarVolCoupl::apply_matrix_mapping21(
    const Core::LinAlg::SparseMatrix& mat) const
{
  // safety check
  check_setup();
  check_init();

  return Core::LinAlg::matrix_multiply(mat, false, *p21_, false, false, false, true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Coupling::Adapter::MortarVolCoupl::master_to_slave(
    const Core::LinAlg::Vector<double>& mv) const
{
  // safety check
  check_setup();
  check_init();

  // create vector
  std::shared_ptr<Core::LinAlg::Vector<double>> sv =
      Core::LinAlg::create_vector(p21_->row_map(), true);
  // project
  master_to_slave(mv, *sv);

  return sv;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::MortarVolCoupl::master_to_slave(
    const Core::LinAlg::MultiVector<double>& mv, Core::LinAlg::MultiVector<double>& sv) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  FOUR_C_ASSERT(mv.get_map().point_same_as(p21_->domain_map()), "master dof map vector expected");
  FOUR_C_ASSERT(sv.get_map().point_same_as(p21_->row_map()), "slave dof map vector expected");
  FOUR_C_ASSERT(sv.num_vectors() == mv.num_vectors(), "column number mismatch {}!={}",
      sv.num_vectors(), mv.num_vectors());
#endif

  // safety check
  check_setup();
  check_init();

  // slave vector with auxiliary dofmap
  Core::LinAlg::MultiVector<double> sv_aux(p21_->row_map(), sv.num_vectors());

  // project
  int err = p21_->multiply(false, mv, sv_aux);
  if (err != 0) FOUR_C_THROW("ERROR: Matrix multiply returned error code {}", err);

  // copy from auxiliary to physical map (needed for coupling in fluid ale algorithm)
  std::copy(sv_aux.get_values(),
      sv_aux.get_values() + (sv_aux.local_length() * sv_aux.num_vectors()), sv.get_values());

  // in contrast to the Adapter::Coupling class we do not need to export here, as
  // the binning has (or should have) guaranteed the same distribution of master and slave dis
  // on all procs
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>>
Coupling::Adapter::MortarVolCoupl::master_to_slave(
    const Core::LinAlg::MultiVector<double>& mv) const
{
  // safety check
  check_setup();
  check_init();

  // create vector
  std::shared_ptr<Core::LinAlg::MultiVector<double>> sv =
      std::make_shared<Core::LinAlg::MultiVector<double>>(p21_->row_map(), mv.num_vectors());
  // project
  master_to_slave(mv, *sv);

  return sv;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Coupling::Adapter::MortarVolCoupl::slave_to_master(
    const Core::LinAlg::Vector<double>& sv) const
{
  // safety check
  check_setup();
  check_init();

  // create vector
  std::shared_ptr<Core::LinAlg::Vector<double>> mv =
      Core::LinAlg::create_vector(p12_->row_map(), true);
  // project
  slave_to_master(sv, *mv);

  return mv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>>
Coupling::Adapter::MortarVolCoupl::slave_to_master(
    const Core::LinAlg::MultiVector<double>& sv) const
{
  // safety check
  check_setup();
  check_init();

  // create vector
  std::shared_ptr<Core::LinAlg::MultiVector<double>> mv =
      std::make_shared<Core::LinAlg::MultiVector<double>>(p12_->row_map(), sv.num_vectors());
  // project
  slave_to_master(sv, *mv);

  return mv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::MortarVolCoupl::slave_to_master(
    const Core::LinAlg::MultiVector<double>& sv, Core::LinAlg::MultiVector<double>& mv) const
{
  // #ifdef FOUR_C_ENABLE_ASSERTIONS
  FOUR_C_ASSERT(mv.get_map().point_same_as(p12_->row_map()), "master dof map vector expected");
  FOUR_C_ASSERT(sv.get_map().point_same_as(p21_->row_map()), "slave dof map vector expected");
  FOUR_C_ASSERT(sv.num_vectors() == mv.num_vectors(), "column number mismatch {}!={}",
      sv.num_vectors(), mv.num_vectors());
  // #endif

  // safety check
  check_setup();
  check_init();

  // master vector with auxiliary dofmap
  Core::LinAlg::MultiVector<double> mv_aux(p12_->row_map(), mv.num_vectors());

  // project
  int err = p12_->multiply(false, sv, mv_aux);
  if (err != 0) FOUR_C_THROW("ERROR: Matrix multiply returned error code {}", err);

  // copy from auxiliary to physical map (needed for coupling in fluid ale algorithm)
  std::copy(mv_aux.get_values(),
      mv_aux.get_values() + (mv_aux.local_length() * mv_aux.num_vectors()), mv.get_values());

  // in contrast to the Adapter::Coupling class we do not need to export here, as
  // the binning has (or should have) guaranteed the same distribution of master and slave dis
  // on all procs
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> Coupling::Adapter::MortarVolCoupl::master_dof_map() const
{
  // safety check
  check_setup();
  check_init();

  return Core::Utils::shared_ptr_from_ref(p12_->row_map());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> Coupling::Adapter::MortarVolCoupl::slave_dof_map() const
{
  // safety check
  check_setup();

  return Core::Utils::shared_ptr_from_ref(p21_->row_map());
}

FOUR_C_NAMESPACE_CLOSE

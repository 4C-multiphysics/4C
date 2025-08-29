// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_dirichletneumann_volcoupl.hpp"

#include "4C_adapter_ale_fluid.hpp"
#include "4C_adapter_fld_fluid_xfem.hpp"
#include "4C_adapter_fld_fluid_xfsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_volmortar.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_geometry_searchtree.hpp"
#include "4C_fem_geometry_searchtree_service.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mortar_calc_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::DirichletNeumannVolCoupl::DirichletNeumannVolCoupl(MPI_Comm comm)
    : DirichletNeumannDisp(comm), coupsa_(nullptr)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::DirichletNeumannVolCoupl::setup()
{
  FSI::DirichletNeumann::setup();

  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  const Teuchos::ParameterList& fsipart = fsidyn.sublist("PARTITIONED SOLVER");
  set_kinematic_coupling(Teuchos::getIntegralValue<Inpar::FSI::CoupVarPart>(
                             fsipart, "COUPVARIABLE") == Inpar::FSI::CoupVarPart::disp);

  if (!get_kinematic_coupling()) FOUR_C_THROW("Currently only displacement coupling is supported!");

  setup_coupling_struct_ale(fsidyn, get_comm());

  setup_interface_corrector(fsidyn, get_comm());

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::DirichletNeumannVolCoupl::setup_coupling_struct_ale(
    const Teuchos::ParameterList& fsidyn, MPI_Comm comm)
{
  const int ndim = Global::Problem::instance()->n_dim();

  coupsa_ = std::make_shared<Coupling::Adapter::MortarVolCoupl>();

  // do a dynamic cast here
  std::shared_ptr<Adapter::FluidAle> fluidale =
      std::dynamic_pointer_cast<Adapter::FluidAle>(fluid_);

  // projection
  std::vector<int> coupleddof12 = std::vector<int>(ndim, 1);
  std::vector<int> coupleddof21 = std::vector<int>(ndim, 1);

  // define dof sets to be coupled for both projections
  std::pair<int, int> dofsets12(0, 0);
  std::pair<int, int> dofsets21(0, 0);

  // initialize coupling adapter
  coupsa_->init(ndim, structure_field()->discretization(),
      fluidale->ale_field()->write_access_discretization(), &coupleddof12, &coupleddof21,
      &dofsets12, &dofsets21, nullptr, false);

  // setup coupling adapter
  coupsa_->setup(Global::Problem::instance()->volmortar_params(),
      Global::Problem::instance()->cut_general_params());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::DirichletNeumannVolCoupl::setup_interface_corrector(
    const Teuchos::ParameterList& fsidyn, MPI_Comm comm)
{
  icorrector_ = std::make_shared<InterfaceCorrector>();

  icorrector_->setup(std::dynamic_pointer_cast<Adapter::FluidAle>(fluid_));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::DirichletNeumannVolCoupl::fluid_op(
    std::shared_ptr<Core::LinAlg::Vector<double>> idisp, const FillType fillFlag)
{
  FSI::Partitioned::fluid_op(idisp, fillFlag);

  // TODO cant this be done better?
  Core::LinAlg::Vector<double> vdisp(*structure_field()->dispnp());

  if (fillFlag == User)
  {
    // SD relaxation calculation
    return fluid_to_struct(mb_fluid_field()->relaxation_solve(struct_to_fluid(idisp), dt()));
  }
  else
  {
    // normal fluid solve
    // the displacement -> velocity conversion at the interface
    const std::shared_ptr<Core::LinAlg::Vector<double>> ivel = interface_velocity(*idisp);

    // A rather simple hack. We need something better!
    const int itemax = mb_fluid_field()->itemax();
    if (fillFlag == MF_Res and mfresitemax_ > 0) mb_fluid_field()->set_itemax(mfresitemax_ + 1);

    std::shared_ptr<Adapter::FluidAle> fluidale =
        std::dynamic_pointer_cast<Adapter::FluidAle>(mb_fluid_field());

    icorrector_->set_interface_displacements(idisp, structure_fluid_coupling());

    // important difference to dirichletneumann.cpp: vdisp is mapped from structure to ale here
    fluidale->nonlinear_solve_vol_coupl(
        structure_to_ale(vdisp), struct_to_fluid(ivel), icorrector_);

    mb_fluid_field()->set_itemax(itemax);

    return fluid_to_struct(mb_fluid_field()->extract_interface_forces());
  }
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::DirichletNeumannVolCoupl::extract_previous_interface_solution()
{
  iveln_ = fluid_to_struct(mb_fluid_field()->extract_interface_veln());
  idispn_ = structure_field()->extract_interface_dispn();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::DirichletNeumannVolCoupl::structure_to_ale(
    const Core::LinAlg::Vector<double>& iv) const
{
  return coupsa_->master_to_slave(iv);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::DirichletNeumannVolCoupl::ale_to_structure(
    Core::LinAlg::Vector<double>& iv) const
{
  return coupsa_->slave_to_master(iv);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::InterfaceCorrector::setup(std::shared_ptr<Adapter::FluidAle> fluidale)
{
  fluidale_ = fluidale;

  volcorrector_ = std::make_shared<VolCorrector>();
  volcorrector_->setup(Global::Problem::instance()->n_dim(), fluidale);

  return;
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::InterfaceCorrector::set_interface_displacements(
    std::shared_ptr<Core::LinAlg::Vector<double>>& idisp_struct,
    Coupling::Adapter::Coupling& icoupfs)
{
  idisp_ = idisp_struct;
  icoupfs_ = Core::Utils::shared_ptr_from_ref(icoupfs);

  deltadisp_ = nullptr;
  return;
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::InterfaceCorrector::correct_interface_displacements(
    std::shared_ptr<Core::LinAlg::Vector<double>> disp_fluid,
    std::shared_ptr<FLD::Utils::MapExtractor> const& finterface)
{
  if (icoupfs_ == nullptr) FOUR_C_THROW("Coupling adapter not set!");
  if (idisp_ == nullptr) FOUR_C_THROW("Interface displacements not set!");

  // std::cout<<*finterface->FullMap()<<std::endl;
  // std::cout<<*disp_fluid<<std::endl;
  deltadisp_ = Core::LinAlg::create_vector(*finterface->fsi_cond_map(), true);

  Core::LinAlg::export_to(*disp_fluid, *deltadisp_);
  // deltadisp_ = finterface->extract_fsi_cond_vector(disp_fluid);

  // FOUR_C_THROW("stop");

  std::shared_ptr<Core::LinAlg::Vector<double>> idisp_fluid_corrected =
      icoupfs_->master_to_slave(*idisp_);

  deltadisp_->update(1.0, *idisp_fluid_corrected, -1.0);

  Core::LinAlg::export_to(*idisp_fluid_corrected, *disp_fluid);
  // finterface->insert_fsi_cond_vector(idisp_fluid_corrected,disp_fluid);

  volcorrector_->correct_vol_displacements(fluidale_, deltadisp_, disp_fluid, finterface);

  // reset
  idisp_ = nullptr;
  icoupfs_ = nullptr;

  return;
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::VolCorrector::correct_vol_displacements(std::shared_ptr<Adapter::FluidAle> fluidale,
    std::shared_ptr<Core::LinAlg::Vector<double>> deltadisp,
    std::shared_ptr<Core::LinAlg::Vector<double>> disp_fluid,
    std::shared_ptr<FLD::Utils::MapExtractor> const& finterface)
{
  if (Core::Communication::my_mpi_rank(fluidale->ale_field()->discretization()->get_comm()) == 0)
    std::cout << "******************   FSI Volume Correction Step   **********************"
              << std::endl;

  // correction step in parameter space
  if (true) correct_vol_displacements_para_space(fluidale, deltadisp, disp_fluid, finterface);
  // correction step in physical space
  else
    correct_vol_displacements_phys_space(fluidale, deltadisp, disp_fluid, finterface);

  // output
  if (Core::Communication::my_mpi_rank(fluidale->ale_field()->discretization()->get_comm()) == 0)
    std::cout << "******************FSI Volume Correction Step Done***********************"
              << std::endl;


  return;
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::VolCorrector::correct_vol_displacements_para_space(
    std::shared_ptr<Adapter::FluidAle> fluidale,
    std::shared_ptr<Core::LinAlg::Vector<double>> deltadisp,
    std::shared_ptr<Core::LinAlg::Vector<double>> disp_fluid,
    std::shared_ptr<FLD::Utils::MapExtractor> const& finterface)
{
  Core::LinAlg::Vector<double> correction(disp_fluid->get_map(), true);
  Core::LinAlg::Vector<double> DofColMapDummy(
      *fluidale->fluid_field()->discretization()->dof_col_map(), true);
  Core::LinAlg::export_to(*deltadisp, DofColMapDummy);

  const double tol = 1e-5;

  // loop over ale eles
  for (std::map<int, std::vector<int>>::iterator it = fluidalenodemap_.begin();
      it != fluidalenodemap_.end(); ++it)
  {
    Core::Elements::Element* aleele = fluidale->ale_field()->discretization()->g_element(it->first);

    // loop over fluid volume nodes within one ale FSI element
    for (size_t i = 0; i < it->second.size(); ++i)
    {
      int gid = it->second[i];
      Core::Nodes::Node* fluidnode = fluidale->fluid_field()->discretization()->g_node(gid);

      if (fluidnode->owner() !=
          Core::Communication::my_mpi_rank(fluidale->ale_field()->discretization()->get_comm()))
        continue;

      double gpos[3] = {fluidnode->x()[0], fluidnode->x()[1], fluidnode->x()[2]};
      double lpos[3] = {0.0, 0.0, 0.0};
      if (aleele->shape() == Core::FE::CellType::quad4)
        Mortar::Utils::global_to_local<Core::FE::CellType::quad4>(*aleele, gpos, lpos);
      else if (aleele->shape() == Core::FE::CellType::hex8)
        Mortar::Utils::global_to_local<Core::FE::CellType::hex8>(*aleele, gpos, lpos);
      else
        FOUR_C_THROW("ERROR: element type not implemented!");

      if (lpos[0] < -1.0 - tol || lpos[1] < -1.0 - tol || lpos[2] < -1.0 - tol ||
          lpos[0] > 1.0 + tol || lpos[1] > 1.0 + tol || lpos[2] > 1.0 + tol)
        continue;

      double dist = 1.0e12;
      int id = -1;
      for (size_t k = 0; k < fluidalenode_fs_imap_[it->first].size(); ++k)
      {
        int gidfsi = fluidalenode_fs_imap_[it->first][k];
        Core::Nodes::Node* fluidnodeFSI = fluidale->fluid_field()->discretization()->g_node(gidfsi);

        double gposFSI[3] = {fluidnodeFSI->x()[0], fluidnodeFSI->x()[1], fluidnodeFSI->x()[2]};
        double lposFSI[3] = {0.0, 0.0, 0.0};
        if (aleele->shape() == Core::FE::CellType::quad4)
          Mortar::Utils::global_to_local<Core::FE::CellType::quad4>(*aleele, gposFSI, lposFSI);
        else if (aleele->shape() == Core::FE::CellType::hex8)
          Mortar::Utils::global_to_local<Core::FE::CellType::hex8>(*aleele, gposFSI, lposFSI);
        else
          FOUR_C_THROW("ERROR: element type not implemented!");

        if (lposFSI[0] < -1.0 - tol || lposFSI[1] < -1.0 - tol || lposFSI[2] < -1.0 - tol ||
            lposFSI[0] > 1.0 + tol || lposFSI[1] > 1.0 + tol || lposFSI[2] > 1.0 + tol)
          FOUR_C_THROW("ERROR: wrong parameter space coordinates!");

        // valc distance to fsi node
        double vec0 = lposFSI[0] - lpos[0];
        double vec1 = lposFSI[1] - lpos[1];
        double vec2 = lposFSI[2] - lpos[2];
        double actdist = sqrt(vec0 * vec0 + vec1 * vec1 + vec2 * vec2);

        // check length
        if (actdist < dist)
        {
          id = fluidnodeFSI->id();
          dist = actdist;
        }
      }  // end loop

      // safety
      if (id < 0) continue;

      double fac = 0.0;

      if (aleele->shape() == Core::FE::CellType::quad4 or
          aleele->shape() == Core::FE::CellType::hex8)
        fac = 1.0 - 0.5 * dist;
      else
        FOUR_C_THROW("ERROR: element type not implemented!");

      // safety
      if (dist > 2.0) fac = 0.0;

      Core::Nodes::Node* fluidnodeFSI = fluidale->fluid_field()->discretization()->g_node(id);
      std::vector<int> temp = fluidale->fluid_field()->discretization()->dof(fluidnodeFSI);
      std::vector<int> dofsFSI;
      for (int idof = 0; idof < dim_; idof++) dofsFSI.push_back(temp[idof]);

      // extract local values of the global vectors
      std::vector<double> FSIdisp = Core::FE::extract_values(DofColMapDummy, dofsFSI);

      std::vector<int> temp2 = fluidale->fluid_field()->discretization()->dof(fluidnode);
      std::vector<int> dofs;
      for (int idof = 0; idof < dim_; idof++) dofs.push_back(temp2[idof]);

      Core::LinAlg::SerialDenseVector gnode(dim_);
      std::vector<int> lmowner(dim_);
      for (int idof = 0; idof < dim_; idof++)
      {
        gnode(idof) = fac * FSIdisp[idof];
        lmowner[idof] = fluidnode->owner();
      }

      Core::LinAlg::assemble(correction, gnode, dofs, lmowner);
    }  // end fluid volume node loop
  }  // end ale fsi element loop

  // do correction
  disp_fluid->update(1.0, correction, 1.0);

  // calc norm
  double norm = 0.0;
  correction.norm_2(&norm);

  // output
  if (Core::Communication::my_mpi_rank(fluidale->ale_field()->discretization()->get_comm()) == 0)
    std::cout << "Norm of correction (parameter space): " << norm << std::endl;

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::VolCorrector::correct_vol_displacements_phys_space(
    std::shared_ptr<Adapter::FluidAle> fluidale,
    std::shared_ptr<Core::LinAlg::Vector<double>> deltadisp,
    std::shared_ptr<Core::LinAlg::Vector<double>> disp_fluid,
    std::shared_ptr<FLD::Utils::MapExtractor> const& finterface)
{
  Core::LinAlg::Vector<double> correction(disp_fluid->get_map(), true);
  Core::LinAlg::Vector<double> DofColMapDummy(
      *fluidale->fluid_field()->discretization()->dof_col_map(), true);
  Core::LinAlg::export_to(*deltadisp, DofColMapDummy);

  std::map<int, Core::LinAlg::Matrix<9, 2>> CurrentDOPs =
      calc_background_dops(*fluidale->fluid_field()->discretization());

  std::shared_ptr<std::set<int>> FSIaleeles = Core::Conditions::conditioned_element_map(
      *fluidale->ale_field()->discretization(), "FSICoupling");

  // evaluate search
  for (int i = 0; i < fluidale->ale_field()->discretization()->num_my_col_elements(); ++i)
  {
    // 1 map node into bele
    int gid = fluidale->ale_field()->discretization()->element_col_map()->gid(i);
    Core::Elements::Element* aleele = fluidale->ale_field()->discretization()->g_element(gid);

    if (FSIaleeles->find(aleele->id()) == FSIaleeles->end()) continue;
  }

  // do correction
  disp_fluid->update(1.0, correction, 1.0);

  // calc norm
  double norm = 0.0;
  correction.norm_2(&norm);

  // output
  if (Core::Communication::my_mpi_rank(fluidale->ale_field()->discretization()->get_comm()) == 0)
    std::cout << "Norm of correction (physical space): " << norm << std::endl;

  return;
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::VolCorrector::setup(const int dim, std::shared_ptr<Adapter::FluidAle> fluidale)
{
  if (Core::Communication::my_mpi_rank(fluidale->ale_field()->discretization()->get_comm()) == 0)
    std::cout << "******************FSI Volume Correction Setup***********************"
              << std::endl;

  dim_ = dim;
  init_dop_normals();

  // init current positions
  std::map<int, Core::LinAlg::Matrix<3, 1>> currentpositions;

  for (int lid = 0; lid < fluidale->fluid_field()->discretization()->num_my_col_elements(); ++lid)
  {
    Core::Elements::Element* sele = fluidale->fluid_field()->discretization()->l_col_element(lid);

    // calculate slabs for every node on every element
    for (int k = 0; k < sele->num_node(); k++)
    {
      Core::Nodes::Node* node = sele->nodes()[k];
      Core::LinAlg::Matrix<3, 1> currpos;

      currpos(0) = node->x()[0];
      currpos(1) = node->x()[1];
      currpos(2) = node->x()[2];

      currentpositions[node->id()] = currpos;
    }
  }

  // init of 3D search tree
  search_tree_ = std::make_shared<Core::Geo::SearchTree>(5);

  // find the bounding box of the elements and initialize the search tree
  const Core::LinAlg::Matrix<3, 2> rootBox =
      Core::Geo::get_xaab_bof_dis(*fluidale->fluid_field()->discretization(), currentpositions);
  search_tree_->initialize_tree(
      rootBox, *fluidale->fluid_field()->discretization(), Core::Geo::TreeType(Core::Geo::OCTTREE));


  std::map<int, Core::LinAlg::Matrix<9, 2>> CurrentDOPs =
      calc_background_dops(*fluidale->fluid_field()->discretization());

  std::shared_ptr<std::set<int>> FSIaleeles = Core::Conditions::conditioned_element_map(
      *fluidale->ale_field()->discretization(), "FSICoupling");

  // evaluate search
  for (int i = 0; i < fluidale->ale_field()->discretization()->num_my_col_elements(); ++i)
  {
    // 1 map node into bele
    int gid = fluidale->ale_field()->discretization()->element_col_map()->gid(i);
    Core::Elements::Element* aleele = fluidale->ale_field()->discretization()->g_element(gid);

    if (FSIaleeles->find(aleele->id()) == FSIaleeles->end()) continue;

    // get found elements from other discr.
    fluidaleelemap_[gid] = search(*aleele, CurrentDOPs);
  }  // end node loop

  std::shared_ptr<Core::LinAlg::Map> FSIfluidnodes = Core::Conditions::condition_node_col_map(
      *fluidale->fluid_field()->discretization(), "FSICoupling");

  std::set<int> globalnodeids;
  // loop over ale eles
  for (std::map<int, std::vector<int>>::iterator it = fluidaleelemap_.begin();
      it != fluidaleelemap_.end(); ++it)
  {
    Core::Elements::Element* aleele = fluidale->ale_field()->discretization()->g_element(it->first);

    std::vector<int> localnodeids;
    std::vector<int> localnodeidsFSI;

    // loop over fluid eles
    for (size_t i = 0; i < it->second.size(); ++i)
    {
      int gid = it->second[i];
      Core::Elements::Element* fluidele = fluidale->fluid_field()->discretization()->g_element(gid);

      for (int j = 0; j < fluidele->num_node(); ++j)
      {
        const int nodegid = fluidele->node_ids()[j];
        Core::Nodes::Node* fluidnode = fluidele->nodes()[j];

        double gpos[3] = {fluidnode->x()[0], fluidnode->x()[1], fluidnode->x()[2]};
        double lpos[3] = {0.0, 0.0, 0.0};
        if (aleele->shape() == Core::FE::CellType::quad4)
          Mortar::Utils::global_to_local<Core::FE::CellType::quad4>(*aleele, gpos, lpos);
        else if (aleele->shape() == Core::FE::CellType::hex8)
          Mortar::Utils::global_to_local<Core::FE::CellType::hex8>(*aleele, gpos, lpos);
        else
          FOUR_C_THROW("ERROR: element type not implemented!");

        double tol = 1e-5;
        if (lpos[0] < -1.0 - tol || lpos[1] < -1.0 - tol || lpos[2] < -1.0 - tol ||
            lpos[0] > 1.0 + tol || lpos[1] > 1.0 + tol || lpos[2] > 1.0 + tol)
          continue;

        if (FSIfluidnodes->my_gid(nodegid))
        {
          localnodeidsFSI.push_back(nodegid);
          continue;
        }
        if (globalnodeids.find(nodegid) != globalnodeids.end()) continue;

        globalnodeids.insert(nodegid);
        localnodeids.push_back(nodegid);
      }
    }
    fluidalenodemap_[it->first] = localnodeids;
    fluidalenode_fs_imap_[it->first] = localnodeidsFSI;
  }

  std::cout << "ALE elements found: " << fluidaleelemap_.size() << std::endl;

  if (Core::Communication::my_mpi_rank(fluidale->ale_field()->discretization()->get_comm()) == 0)
    std::cout << "******************FSI Volume Correction Setup Done***********************"
              << std::endl;

  return;
}

/*----------------------------------------------------------------------*
 |  Init normals for Dop calculation                         farah 05/16|
 *----------------------------------------------------------------------*/
void FSI::VolCorrector::init_dop_normals()
{
  dopnormals_(0, 0) = 1.0;
  dopnormals_(0, 1) = 0.0;
  dopnormals_(0, 2) = 0.0;

  dopnormals_(1, 0) = 0.0;
  dopnormals_(1, 1) = 1.0;
  dopnormals_(1, 2) = 0.0;

  dopnormals_(2, 0) = 0.0;
  dopnormals_(2, 1) = 0.0;
  dopnormals_(2, 2) = 1.0;

  dopnormals_(3, 0) = 1.0;
  dopnormals_(3, 1) = 1.0;
  dopnormals_(3, 2) = 0.0;

  dopnormals_(4, 0) = 1.0;
  dopnormals_(4, 1) = 0.0;
  dopnormals_(4, 2) = 1.0;

  dopnormals_(5, 0) = 0.0;
  dopnormals_(5, 1) = 1.0;
  dopnormals_(5, 2) = 1.0;

  dopnormals_(6, 0) = 1.0;
  dopnormals_(6, 1) = 0.0;
  dopnormals_(6, 2) = -1.0;

  dopnormals_(7, 0) = 1.0;
  dopnormals_(7, 1) = -1.0;
  dopnormals_(7, 2) = 0.0;

  dopnormals_(8, 0) = 0.0;
  dopnormals_(8, 1) = 1.0;
  dopnormals_(8, 2) = -1.0;

  return;
}


/*----------------------------------------------------------------------*
 |  Calculate Dops for background mesh                       farah 05/16|
 *----------------------------------------------------------------------*/
std::map<int, Core::LinAlg::Matrix<9, 2>> FSI::VolCorrector::calc_background_dops(
    Core::FE::Discretization& searchdis)
{
  std::map<int, Core::LinAlg::Matrix<9, 2>> currentKDOPs;

  for (int lid = 0; lid < searchdis.num_my_col_elements(); ++lid)
  {
    Core::Elements::Element* sele = searchdis.l_col_element(lid);

    currentKDOPs[sele->id()] = calc_dop(*sele);
  }

  return currentKDOPs;
}

/*----------------------------------------------------------------------*
 |  Calculate Dop for one Element                            farah 05/16|
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<9, 2> FSI::VolCorrector::calc_dop(Core::Elements::Element& ele)
{
  Core::LinAlg::Matrix<9, 2> dop;

  // calculate slabs
  for (int j = 0; j < 9; j++)
  {
    // initialize slabs
    dop(j, 0) = 1.0e12;
    dop(j, 1) = -1.0e12;
  }

  // calculate slabs for every node on every element
  for (int k = 0; k < ele.num_node(); k++)
  {
    Core::Nodes::Node* node = ele.nodes()[k];

    // get current node position
    std::array<double, 3> pos = {0.0, 0.0, 0.0};
    for (int j = 0; j < dim_; ++j) pos[j] = node->x()[j];

    // calculate slabs
    for (int j = 0; j < 9; j++)
    {
      //= ax+by+cz=d/sqrt(aa+bb+cc)
      double num =
          dopnormals_(j, 0) * pos[0] + dopnormals_(j, 1) * pos[1] + dopnormals_(j, 2) * pos[2];
      double denom =
          sqrt((dopnormals_(j, 0) * dopnormals_(j, 0)) + (dopnormals_(j, 1) * dopnormals_(j, 1)) +
               (dopnormals_(j, 2) * dopnormals_(j, 2)));
      double dcurrent = num / denom;

      if (dcurrent > dop(j, 1)) dop(j, 1) = dcurrent;
      if (dcurrent < dop(j, 0)) dop(j, 0) = dcurrent;
    }
  }

  return dop;
}


/*----------------------------------------------------------------------*
 |  Perform searching procedure                              farah 05/16|
 *----------------------------------------------------------------------*/
std::vector<int> FSI::VolCorrector::search(
    Core::Elements::Element& ele, std::map<int, Core::LinAlg::Matrix<9, 2>>& currentKDOPs)
{
  // vector of global ids of found elements
  std::vector<int> gids;
  gids.clear();
  std::set<int> gid;
  gid.clear();

  Core::LinAlg::Matrix<9, 2> queryKDOP;

  // calc dop for considered element
  queryKDOP = calc_dop(ele);

  //**********************************************************
  // search for near elements to the background node's coord
  search_tree_->search_collisions(currentKDOPs, queryKDOP, 0, gid);

  for (std::set<int>::iterator iter = gid.begin(); iter != gid.end(); ++iter) gids.push_back(*iter);

  return gids;
}

FOUR_C_NAMESPACE_CLOSE

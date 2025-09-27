// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_coupling_poro_mortar.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_node.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fem_nurbs_discretization_control_point.hpp"
#include "4C_fem_nurbs_discretization_knotvector.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_sparsematrix.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor                                                      ager 10/15|
 *----------------------------------------------------------------------*/
Adapter::CouplingPoroMortar::CouplingPoroMortar(int spatial_dimension,
    Teuchos::ParameterList mortar_coupling_params, Teuchos::ParameterList contact_dynamic_params,
    Core::FE::ShapeFunctionType shape_function_type)
    : CouplingNonLinMortar(
          spatial_dimension, mortar_coupling_params, contact_dynamic_params, shape_function_type),
      firstinit_(false),
      slavetype_(-1),
      mastertype_(-1)
{
  // empty...
}

/*----------------------------------------------------------------------*
 |  Read Mortar Condition                                     ager 10/15|
 *----------------------------------------------------------------------*/
void Adapter::CouplingPoroMortar::read_mortar_condition(
    std::shared_ptr<Core::FE::Discretization> masterdis,
    std::shared_ptr<Core::FE::Discretization> slavedis, std::vector<int> coupleddof,
    const std::string& couplingcond, Teuchos::ParameterList& input,
    std::map<int, Core::Nodes::Node*>& mastergnodes, std::map<int, Core::Nodes::Node*>& slavegnodes,
    std::map<int, std::shared_ptr<Core::Elements::Element>>& masterelements,
    std::map<int, std::shared_ptr<Core::Elements::Element>>& slaveelements)
{
  // Call Base Class
  CouplingNonLinMortar::read_mortar_condition(masterdis, slavedis, coupleddof, couplingcond, input,
      mastergnodes, slavegnodes, masterelements, slaveelements);

  // Set Problem Type to Poro
  switch (Global::Problem::instance()->get_problem_type())
  {
    case Core::ProblemType::poroelast:
      input.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::poroelast);
      break;
    case Core::ProblemType::poroscatra:
      input.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::poroscatra);
      break;
    default:
      FOUR_C_THROW("Invalid poro problem is specified");
      break;
  }

  // porotimefac = 1/(theta*dt) --- required for derivation of structural displacements!
  const Teuchos::ParameterList& stru = Global::Problem::instance()->structural_dynamic_params();
  double porotimefac =
      1 / (stru.sublist("ONESTEPTHETA").get<double>("THETA") * stru.get<double>("TIMESTEP"));
  input.set<double>("porotimefac", porotimefac);
  const Teuchos::ParameterList& porodyn = Global::Problem::instance()->poroelast_dynamic_params();
  input.set<bool>("CONTACT_NO_PENETRATION",
      porodyn.get<bool>("CONTACT_NO_PENETRATION"));  // used in the integrator
  if (!porodyn.get<bool>("CONTACT_NO_PENETRATION"))
    FOUR_C_THROW("Set CONTACT_NO_PENETRATION for Poroelastic meshtying!");
}

/*----------------------------------------------------------------------*
 |  Add Mortar Elements                                        ager 10/15|
 *----------------------------------------------------------------------*/
void Adapter::CouplingPoroMortar::add_mortar_elements(
    std::shared_ptr<Core::FE::Discretization> masterdis,
    std::shared_ptr<Core::FE::Discretization> slavedis, Teuchos::ParameterList& input,
    std::map<int, std::shared_ptr<Core::Elements::Element>>& masterelements,
    std::map<int, std::shared_ptr<Core::Elements::Element>>& slaveelements,
    std::shared_ptr<CONTACT::Interface>& interface, int numcoupleddof)
{
  bool isnurbs = input.get<bool>("NURBS");

  // get problem dimension (2D or 3D) and create (Mortar::Interface)
  const int dim = Global::Problem::instance()->n_dim();

  // feeding master elements to the interface
  std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator elemiter;
  for (elemiter = masterelements.begin(); elemiter != masterelements.end(); ++elemiter)
  {
    std::shared_ptr<Core::Elements::Element> ele = elemiter->second;
    std::shared_ptr<CONTACT::Element> cele = std::make_shared<CONTACT::Element>(
        ele->id(), ele->owner(), ele->shape(), ele->num_node(), ele->node_ids(), false, isnurbs);

    std::shared_ptr<Core::Elements::FaceElement> faceele =
        std::dynamic_pointer_cast<Core::Elements::FaceElement>(ele);
    if (faceele == nullptr) FOUR_C_THROW("Cast to FaceElement failed!");
    cele->phys_type() = Mortar::Element::other;

    std::vector<const Core::Conditions::Condition*> porocondvec;
    masterdis->get_condition("PoroCoupling", porocondvec);
    for (unsigned int i = 0; i < porocondvec.size(); ++i)
    {
      std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator eleitergeometry;
      for (eleitergeometry = porocondvec[i]->geometry().begin();
          eleitergeometry != porocondvec[i]->geometry().end(); ++eleitergeometry)
      {
        if (faceele->parent_element()->id() == eleitergeometry->second->id())
        {
          if (mastertype_ == 0)
            FOUR_C_THROW(
                "struct and poro master elements on the same processor - no mixed interface "
                "supported");
          cele->phys_type() = Mortar::Element::poro;
          mastertype_ = 1;
          break;
        }
      }
    }
    if (cele->phys_type() == Mortar::Element::other)
    {
      if (mastertype_ == 1)
        FOUR_C_THROW(
            "struct and poro master elements on the same processor - no mixed interface supported");
      cele->phys_type() = Mortar::Element::structure;
      mastertype_ = 0;
    }

    cele->set_parent_master_element(faceele->parent_element(), faceele->face_parent_number());

    if (isnurbs)
    {
      std::shared_ptr<Core::FE::Nurbs::NurbsDiscretization> nurbsdis =
          std::dynamic_pointer_cast<Core::FE::Nurbs::NurbsDiscretization>(masterdis);

      std::shared_ptr<Core::FE::Nurbs::Knotvector> knots = (*nurbsdis).get_knot_vector();
      std::vector<Core::LinAlg::SerialDenseVector> parentknots(dim);
      std::vector<Core::LinAlg::SerialDenseVector> mortarknots(dim - 1);

      std::shared_ptr<Core::Elements::FaceElement> faceele =
          std::dynamic_pointer_cast<Core::Elements::FaceElement>(ele);
      double normalfac = 0.0;
      bool zero_size = knots->get_boundary_ele_and_parent_knots(parentknots, mortarknots, normalfac,
          faceele->parent_master_element()->id(), faceele->face_master_number());

      // store nurbs specific data to node
      cele->zero_sized() = zero_size;
      cele->knots() = mortarknots;
      cele->normal_fac() = normalfac;
    }

    interface->add_element(cele);
  }

  // feeding slave elements to the interface
  for (elemiter = slaveelements.begin(); elemiter != slaveelements.end(); ++elemiter)
  {
    std::shared_ptr<Core::Elements::Element> ele = elemiter->second;

    std::shared_ptr<CONTACT::Element> cele = std::make_shared<CONTACT::Element>(
        ele->id(), ele->owner(), ele->shape(), ele->num_node(), ele->node_ids(), true, isnurbs);

    std::shared_ptr<Core::Elements::FaceElement> faceele =
        std::dynamic_pointer_cast<Core::Elements::FaceElement>(ele);
    if (faceele == nullptr) FOUR_C_THROW("Cast to FaceElement failed!");
    cele->phys_type() = Mortar::Element::other;

    std::vector<const Core::Conditions::Condition*> porocondvec;
    masterdis->get_condition("PoroCoupling", porocondvec);

    for (unsigned int i = 0; i < porocondvec.size(); ++i)
    {
      std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator eleitergeometry;
      for (eleitergeometry = porocondvec[i]->geometry().begin();
          eleitergeometry != porocondvec[i]->geometry().end(); ++eleitergeometry)
      {
        if (faceele->parent_element()->id() == eleitergeometry->second->id())
        {
          if (slavetype_ == 0)
            FOUR_C_THROW(
                "struct and poro slave elements on the same processor - no mixed interface "
                "supported");
          cele->phys_type() = Mortar::Element::poro;
          slavetype_ = 1;
          break;
        }
      }
    }
    if (cele->phys_type() == Mortar::Element::other)
    {
      if (slavetype_ == 1)
        FOUR_C_THROW(
            "struct and poro slave elements on the same processor - no mixed interface supported");
      cele->phys_type() = Mortar::Element::structure;
      slavetype_ = 0;
    }
    cele->set_parent_master_element(faceele->parent_element(), faceele->face_parent_number());

    if (isnurbs)
    {
      std::shared_ptr<Core::FE::Nurbs::NurbsDiscretization> nurbsdis =
          std::dynamic_pointer_cast<Core::FE::Nurbs::NurbsDiscretization>(slavedis);

      std::shared_ptr<Core::FE::Nurbs::Knotvector> knots = (*nurbsdis).get_knot_vector();
      std::vector<Core::LinAlg::SerialDenseVector> parentknots(dim);
      std::vector<Core::LinAlg::SerialDenseVector> mortarknots(dim - 1);

      std::shared_ptr<Core::Elements::FaceElement> faceele =
          std::dynamic_pointer_cast<Core::Elements::FaceElement>(ele);
      double normalfac = 0.0;
      bool zero_size = knots->get_boundary_ele_and_parent_knots(parentknots, mortarknots, normalfac,
          faceele->parent_master_element()->id(), faceele->face_master_number());

      // store nurbs specific data to node
      cele->zero_sized() = zero_size;
      cele->knots() = mortarknots;
      cele->normal_fac() = normalfac;
    }

    interface->add_element(cele);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  read mortar condition                                    Ager 02/16 |
 *----------------------------------------------------------------------*/
void Adapter::CouplingPoroMortar::create_strategy(
    std::shared_ptr<Core::FE::Discretization> masterdis,
    std::shared_ptr<Core::FE::Discretization> slavedis, Teuchos::ParameterList& input,
    int numcoupleddof)
{
  // poro lagrange strategy:

  // get problem dimension (2D or 3D) and create (Mortar::Interface)
  const int dim = Global::Problem::instance()->n_dim();

  // bools to decide which side is structural and which side is poroelastic to manage all 4
  // constellations
  // s-s, p-s, s-p, p-p
  bool poromaster = false;
  bool poroslave = false;
  bool structmaster = false;
  bool structslave = false;

  // wait for all processors to determine if they have poro or structural master or slave elements
  Core::Communication::barrier(comm_);
  std::vector<int> slaveTypeList(Core::Communication::num_mpi_ranks(comm_));
  std::vector<int> masterTypeList(Core::Communication::num_mpi_ranks(comm_));
  Core::Communication::gather_all(&slavetype_, slaveTypeList.data(), 1, comm_);
  Core::Communication::gather_all(&mastertype_, masterTypeList.data(), 1, comm_);
  Core::Communication::barrier(comm_);

  for (int i = 0; i < Core::Communication::num_mpi_ranks(comm_); ++i)
  {
    switch (slaveTypeList[i])
    {
      case -1:
        break;
      case 1:
        if (structslave)
          FOUR_C_THROW(
              "struct and poro slave elements on the same adapter - no mixed interface supported");
        // adjust FOUR_C_THROW text, when more than one interface is supported
        poroslave = true;
        break;
      case 0:
        if (poroslave)
          FOUR_C_THROW(
              "struct and poro slave elements on the same adapter - no mixed interface supported");
        structslave = true;
        break;
      default:
        FOUR_C_THROW("this cannot happen");
        break;
    }
  }

  for (int i = 0; i < Core::Communication::num_mpi_ranks(comm_); ++i)
  {
    switch (masterTypeList[i])
    {
      case -1:
        break;
      case 1:
        if (structmaster)
          FOUR_C_THROW(
              "struct and poro master elements on the same adapter - no mixed interface supported");
        // adjust FOUR_C_THROW text, when more than one interface is supported
        poromaster = true;
        break;
      case 0:
        if (poromaster)
          FOUR_C_THROW(
              "struct and poro master elements on the same adapter - no mixed interface supported");
        structmaster = true;
        break;
      default:
        FOUR_C_THROW("this cannot happen");
        break;
    }
  }

  const Teuchos::ParameterList& stru = Global::Problem::instance()->structural_dynamic_params();
  double theta = stru.sublist("ONESTEPTHETA").get<double>("THETA");
  // what if problem is static ? there should be an error for previous line called in a dyna_statics
  // problem and not a value of 0.5 a proper distinction is necessary if poro meshtying is expanded
  // to other time integration strategies

  if (Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(stru, "DYNAMICTYPE") ==
      Inpar::Solid::DynamicType::Statics)
  {
    theta = 1.0;
  }
  std::vector<std::shared_ptr<CONTACT::Interface>> interfaces;
  interfaces.push_back(interface_);
  double alphaf = 1.0 - theta;

  // build the correct data container
  std::shared_ptr<CONTACT::AbstractStrategyDataContainer> data_ptr =
      std::make_shared<CONTACT::AbstractStrategyDataContainer>();
  // create contact poro lagrange strategy for mesh tying
  porolagstrategy_ = std::make_shared<CONTACT::LagrangeStrategyPoro>(data_ptr,
      masterdis->dof_row_map(), masterdis->node_row_map(), input, interfaces, dim, comm_, alphaf,
      numcoupleddof, poroslave, poromaster);

  porolagstrategy_->setup(false, true);
  porolagstrategy_->poro_mt_initialize();

  firstinit_ = true;
}


/*----------------------------------------------------------------------*
 |  complete interface (also print and parallel redist.)      Ager 02/16|
 *----------------------------------------------------------------------*/
void Adapter::CouplingPoroMortar::complete_interface(
    std::shared_ptr<Core::FE::Discretization> masterdis,
    std::shared_ptr<CONTACT::Interface>& interface)
{
  // finalize the contact interface construction
  int maxdof = masterdis->dof_row_map()->max_all_gid();
  interface->fill_complete({}, Global::Problem::instance()->binning_strategy_params(),
      Global::Problem::instance()->output_control_file(),
      Global::Problem::instance()->spatial_approximation_type(), true, maxdof);

  // interface->create_volume_ghosting(*masterdis);

  // store old row maps (before parallel redistribution)
  slavedofrowmap_ = std::make_shared<Core::LinAlg::Map>(*interface->slave_row_dofs());
  masterdofrowmap_ = std::make_shared<Core::LinAlg::Map>(*interface->master_row_dofs());
  slavenoderowmap_ = std::make_shared<Core::LinAlg::Map>(*interface->slave_row_nodes());

  // print parallel distribution
  interface->print_parallel_distribution();

  // store interface
  interface_ = interface;

  // create binary search tree
  interface_->create_search_tree();

  return;
}


/*----------------------------------------------------------------------*
 |  evaluate blockmatrices for poro meshtying             ager 10/15    |
 *----------------------------------------------------------------------*/
void Adapter::CouplingPoroMortar::evaluate_poro_mt(
    std::shared_ptr<Core::LinAlg::Vector<double>> fvel,
    std::shared_ptr<Core::LinAlg::Vector<double>> svel,
    std::shared_ptr<Core::LinAlg::Vector<double>> fpres,
    std::shared_ptr<Core::LinAlg::Vector<double>> sdisp,
    const std::shared_ptr<Core::FE::Discretization> sdis,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& f,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& k_fs,
    std::shared_ptr<Core::LinAlg::Vector<double>>& frhs, Coupling::Adapter::Coupling& coupfs,
    std::shared_ptr<const Core::LinAlg::Map> fdofrowmap)
{
  // safety check
  check_setup();

  // write interface values into poro contact interface data containers for integration in contact
  // integrator
  porolagstrategy_->set_state(Mortar::state_fvelocity, *fvel);
  porolagstrategy_->set_state(Mortar::state_svelocity, *svel);
  porolagstrategy_->set_state(Mortar::state_fpressure, *fpres);
  porolagstrategy_->set_state(Mortar::state_new_displacement, *sdisp);

  // store displacements of parent elements for deformation gradient determinant and its
  // linearization
  porolagstrategy_->set_parent_state(Mortar::StateType::state_new_displacement, *sdisp, *sdis);

  interface_->initialize();
  // in the end of Evaluate coupling condition residuals and linearizations are computed in contact
  // integrator
  interface_->evaluate();

  porolagstrategy_->poro_mt_prepare_fluid_coupling();
  porolagstrategy_->poro_initialize(coupfs, *fdofrowmap, firstinit_);
  if (firstinit_) firstinit_ = false;

  // do system matrix manipulations
  porolagstrategy_->evaluate_poro_no_pen_contact(k_fs, f, frhs);
  return;
}  // Adapter::CouplingNonLinMortar::EvaluatePoroMt()


/*----------------------------------------------------------------------*
 |  update poro meshtying quantities                      ager 10/15    |
 *----------------------------------------------------------------------*/
void Adapter::CouplingPoroMortar::update_poro_mt()
{
  // safety check
  check_setup();

  porolagstrategy_->poro_mt_update();
  return;
}  // Adapter::CouplingNonLinMortar::UpdatePoroMt()

/*----------------------------------------------------------------------*
 |  recover fluid coupling lagrange multiplier            ager 10/15    |
 *----------------------------------------------------------------------*/
void Adapter::CouplingPoroMortar::recover_fluid_lm_poro_mt(
    std::shared_ptr<Core::LinAlg::Vector<double>> disi,
    std::shared_ptr<Core::LinAlg::Vector<double>> veli)
{
  // safety check
  check_setup();

  porolagstrategy_->recover_poro_no_pen(disi, veli);
  return;
}  // Adapter::CouplingNonLinMortar::recover_fluid_lm_poro_mt

FOUR_C_NAMESPACE_CLOSE

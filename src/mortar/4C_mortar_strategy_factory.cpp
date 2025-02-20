// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mortar_strategy_factory.hpp"

#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fem_nurbs_discretization_control_point.hpp"
#include "4C_fem_nurbs_discretization_knotvector.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_interface.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mortar::STRATEGY::Factory::Factory()
    : discret_ptr_(nullptr), isinit_(false), issetup_(false), comm_(MPI_COMM_NULL), dim_(-1)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::STRATEGY::Factory::init(std::shared_ptr<Core::FE::Discretization> dis)
{
  // call setup() after init()
  issetup_ = false;

  discret_ptr_ = dis;

  isinit_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::STRATEGY::Factory::setup(const int dim)
{
  check_init();

  // get a copy of the underlying structural communicator
  comm_ = discret_ptr_->get_comm();

  // get the problem dimension
  dim_ = dim;

  // Note: Since this is an abstract class, the setup flag stays false.

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Mortar::STRATEGY::Factory::check_init_setup() const
{
  if (!is_init() or !is_setup()) FOUR_C_THROW("Call init() and setup() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Mortar::STRATEGY::Factory::check_init() const
{
  if (not is_init()) FOUR_C_THROW("Call init() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::FE::Discretization& Mortar::STRATEGY::Factory::discret()
{
  check_init();
  return *discret_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::FE::Discretization& Mortar::STRATEGY::Factory::discret() const
{
  check_init();
  return *discret_ptr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MPI_Comm Mortar::STRATEGY::Factory::get_comm() const
{
  check_init_setup();
  return comm_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const int& Mortar::STRATEGY::Factory::n_dim() const
{
  if (dim_ == -1)
    FOUR_C_THROW(
        "Call the Solid::MODELEVEALUATOR::setup() routine first to "
        "set the problem dimension variable!");
  return dim_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Mortar::STRATEGY::Factory::check_dimension() const
{
  if (n_dim() != 2 && n_dim() != 3)
    FOUR_C_THROW("Mortar meshtying/contact problems must be 2D or 3D");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::STRATEGY::Factory::prepare_nurbs_element(const Core::FE::Discretization& discret,
    std::shared_ptr<Core::Elements::Element> ele, Mortar::Element& cele) const
{
  const Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
      dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(&(discret));
  if (nurbsdis == nullptr) FOUR_C_THROW("Dynamic cast failed!");

  std::shared_ptr<const Core::FE::Nurbs::Knotvector> knots = nurbsdis->get_knot_vector();
  std::vector<Core::LinAlg::SerialDenseVector> parentknots(n_dim());
  std::vector<Core::LinAlg::SerialDenseVector> mortarknots(n_dim() - 1);

  double normalfac = 0.0;
  std::shared_ptr<Core::Elements::FaceElement> faceele =
      std::dynamic_pointer_cast<Core::Elements::FaceElement>(ele);
  if (!faceele) FOUR_C_THROW("Cast to FaceElement failed!");

  bool zero_size = knots->get_boundary_ele_and_parent_knots(parentknots, mortarknots, normalfac,
      faceele->parent_master_element()->id(), faceele->face_master_number());

  // store nurbs specific data to node
  cele.zero_sized() = zero_size;
  cele.knots() = mortarknots;
  cele.normal_fac() = normalfac;

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::STRATEGY::Factory::prepare_nurbs_node(
    const Core::Nodes::Node* node, Mortar::Node& mnode) const
{
  const Core::FE::Nurbs::ControlPoint* cp =
      dynamic_cast<const Core::FE::Nurbs::ControlPoint*>(node);

  mnode.nurbs_w() = cp->w();

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Mortar::STRATEGY::Factory::build_search_tree(
    const std::vector<std::shared_ptr<Mortar::Interface>>& interfaces) const
{
  for (unsigned i = 0; i < interfaces.size(); ++i) interfaces[i]->create_search_tree();

  return;
}

FOUR_C_NAMESPACE_CLOSE

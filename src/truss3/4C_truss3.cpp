// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_truss3.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_so3_nullspace.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::Elements::Truss3Type Discret::Elements::Truss3Type::instance_;

Discret::Elements::Truss3Type& Discret::Elements::Truss3Type::instance() { return instance_; }

Core::Communication::ParObject* Discret::Elements::Truss3Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::Truss3(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Truss3Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "TRUSS3")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Truss3>(id, owner);
    return ele;
  }
  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Truss3Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Truss3>(id, owner);
  return ele;
}


void Discret::Elements::Truss3Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::Truss3Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_3d_null_space(node, x0);
}

void Discret::Elements::Truss3Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, Input::LineDefinition>& defs = definitions["TRUSS3"];

  defs["LINE2"] = Input::LineDefinition::Builder()
                      .add_int_vector("LINE2", 2)
                      .add_named_int("MAT")
                      .add_named_double("CROSS")
                      .add_named_string("KINEM")
                      .build();
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            cyron 08/08|
 *----------------------------------------------------------------------*/
Discret::Elements::Truss3::Truss3(int id, int owner)
    : Core::Elements::Element(id, owner),
      crosssec_(0.0),
      eint_(0.0),
      lrefe_(0.0),
      gaussrule_(Core::FE::GaussRule1D::line_2point),
      diff_disp_ref_(Core::LinAlg::Matrix<1, 3>(true)),
      interface_ptr_(nullptr),
      isinit_(false),
      jacobimass_(),
      jacobinode_(),
      kintype_(KinematicType::tr3_totlag),
      material_(0),
      x_()
{
}
/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       cyron 08/08|
 *----------------------------------------------------------------------*/
Discret::Elements::Truss3::Truss3(const Discret::Elements::Truss3& old)
    : Core::Elements::Element(old),
      crosssec_(old.crosssec_),
      eint_(old.eint_),
      lrefe_(old.lrefe_),
      gaussrule_(old.gaussrule_),
      diff_disp_ref_(old.diff_disp_ref_),
      interface_ptr_(old.interface_ptr_),
      isinit_(old.isinit_),
      jacobimass_(old.jacobimass_),
      jacobinode_(old.jacobinode_),
      kintype_(old.kintype_),
      material_(old.material_),
      x_(old.x_)

{
}
/*----------------------------------------------------------------------*
 |  Deep copy this instance of Truss3 and return pointer to it (public) |
 |                                                            cyron 08/08|
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Truss3::clone() const
{
  auto* newelement = new Discret::Elements::Truss3(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |(public)                                                   cyron 08/08|
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Truss3::shape() const { return Core::FE::CellType::line2; }


/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           cyron 08/08|
 *----------------------------------------------------------------------*/
void Discret::Elements::Truss3::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Element::pack(data);
  add_to_pack(data, isinit_);
  add_to_pack(data, x_);
  add_to_pack(data, diff_disp_ref_);
  add_to_pack(data, material_);
  add_to_pack(data, lrefe_);
  add_to_pack(data, jacobimass_);
  add_to_pack(data, jacobinode_);
  add_to_pack(data, crosssec_);
  add_to_pack(data, gaussrule_);
  add_to_pack(data, kintype_);
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           cyron 08/08|
 *----------------------------------------------------------------------*/
void Discret::Elements::Truss3::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  std::vector<char> basedata(0);
  extract_from_pack(buffer, basedata);
  Core::Communication::UnpackBuffer base_buffer(basedata);
  Element::unpack(base_buffer);
  extract_from_pack(buffer, isinit_);
  extract_from_pack(buffer, x_);
  extract_from_pack(buffer, diff_disp_ref_);
  extract_from_pack(buffer, material_);
  extract_from_pack(buffer, lrefe_);
  extract_from_pack(buffer, jacobimass_);
  extract_from_pack(buffer, jacobinode_);
  extract_from_pack(buffer, crosssec_);
  extract_from_pack(buffer, gaussrule_);
  // kinematic type
  extract_from_pack(buffer, kintype_);

  FOUR_C_THROW_UNLESS(buffer.at_end(), "Buffer not fully consumed.");
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                              cyron 08/08|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Truss3::lines()
{
  return {Core::Utils::shared_ptr_from_ref(*this)};
}

/*----------------------------------------------------------------------*
 |determine Gauss rule from required type of integration                |
 |                                                   (public)cyron 09/09|
 *----------------------------------------------------------------------*/
Core::FE::GaussRule1D Discret::Elements::Truss3::my_gauss_rule(
    int nnode, IntegrationType integrationtype)
{
  Core::FE::GaussRule1D gaussrule = Core::FE::GaussRule1D::undefined;

  switch (nnode)
  {
    case 2:
    {
      switch (integrationtype)
      {
        case gaussexactintegration:
        {
          gaussrule = Core::FE::GaussRule1D::line_2point;
          break;
        }
        case gaussunderintegration:
        {
          gaussrule = Core::FE::GaussRule1D::line_1point;
          break;
        }
        case lobattointegration:
        {
          gaussrule = Core::FE::GaussRule1D::line_lobatto2point;
          break;
        }
        default:
          FOUR_C_THROW("unknown type of integration");
      }
      break;
    }
    case 3:
    {
      switch (integrationtype)
      {
        case gaussexactintegration:
        {
          gaussrule = Core::FE::GaussRule1D::line_3point;
          break;
        }
        case gaussunderintegration:
        {
          gaussrule = Core::FE::GaussRule1D::line_2point;
          break;
        }
        case lobattointegration:
        {
          gaussrule = Core::FE::GaussRule1D::line_lobatto3point;
          break;
        }
        default:
          FOUR_C_THROW("unknown type of integration");
      }
      break;
    }
    case 4:
    {
      switch (integrationtype)
      {
        case gaussexactintegration:
        {
          gaussrule = Core::FE::GaussRule1D::line_4point;
          break;
        }
        case gaussunderintegration:
        {
          gaussrule = Core::FE::GaussRule1D::line_3point;
          break;
        }
        default:
          FOUR_C_THROW("unknown type of integration");
      }
      break;
    }
    case 5:
    {
      switch (integrationtype)
      {
        case gaussexactintegration:
        {
          gaussrule = Core::FE::GaussRule1D::line_5point;
          break;
        }
        case gaussunderintegration:
        {
          gaussrule = Core::FE::GaussRule1D::line_4point;
          break;
        }
        default:
          FOUR_C_THROW("unknown type of integration");
      }
      break;
    }
    default:
      FOUR_C_THROW("Only Line2, Line3, Line4 and Line5 Elements implemented.");
  }

  return gaussrule;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Truss3::set_up_reference_geometry(const std::vector<double>& xrefe)
{
  if (!isinit_)
  {
    // setting reference coordinates
    for (int i = 0; i < 6; ++i) x_(i) = xrefe[i];

    // length in reference configuration
    lrefe_ = std::sqrt((x_(3) - x_(0)) * (x_(3) - x_(0)) + (x_(4) - x_(1)) * (x_(4) - x_(1)) +
                       (x_(5) - x_(2)) * (x_(5) - x_(2)));

    // set jacobi determinants for integration of mass matrix and at nodes
    jacobimass_.resize(2);
    jacobimass_[0] = lrefe_ / 2.0;
    jacobimass_[1] = lrefe_ / 2.0;
    jacobinode_.resize(2);
    jacobinode_[0] = lrefe_ / 2.0;
    jacobinode_[1] = lrefe_ / 2.0;

    isinit_ = true;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Truss3::scale_reference_length(double scalefac)
{
  // scale length in reference configuration
  x_(3) = x_(0) + (scalefac * (x_(3) - x_(0)));
  x_(4) = x_(1) + (scalefac * (x_(4) - x_(1)));
  x_(5) = x_(2) + (scalefac * (x_(5) - x_(2)));

  lrefe_ *= scalefac;

  // set jacobi determinants for integration of mass matrix and at nodes
  jacobimass_.resize(2);
  jacobimass_[0] = jacobimass_[1] = lrefe_ * 0.5;
  jacobinode_.resize(2);
  jacobinode_[0] = jacobinode_[1] = lrefe_ * 0.5;
}

/*----------------------------------------------------------------------*
 | Get degrees of freedom used by this element               fang 05/16 |
 *----------------------------------------------------------------------*/
// TODO: remove once truss3 element is fixed and no longer expects more dofs (6) than it can
// inherently handle (3)...
void Discret::Elements::Truss3::location_vector(
    const Core::FE::Discretization& dis, Core::Elements::LocationArray& la, bool doDirichlet) const
{
  const int numnode = num_node();
  const Core::Nodes::Node* const* nodes = Truss3::nodes();

  la.clear();

  // we need to look at all DofSets of our discretization
  for (int dofset = 0; dofset < la.size(); ++dofset)
  {
    std::vector<int>& lm = la[dofset].lm_;
    std::vector<int>& lmdirich = la[dofset].lmdirich_;
    std::vector<int>& lmowner = la[dofset].lmowner_;
    std::vector<int>& lmstride = la[dofset].stride_;

    // fill the vector with nodal dofs
    if (nodes)
    {
      for (int i = 0; i < numnode; ++i)
      {
        const Core::Nodes::Node* node = nodes[i];

        const int owner = node->owner();
        std::vector<int> dof;
        dis.dof(dof, node, dofset, 0, this);
        const unsigned size = dof.size();

        if (size) lmstride.push_back(size);
        for (unsigned j = 0; j < size; ++j)
        {
          lmowner.push_back(owner);
          lm.push_back(dof[j]);
        }

        if (doDirichlet)
        {
          const std::vector<int>* flag = nullptr;
          Core::Conditions::Condition* dirich = node->get_condition("Dirichlet");
          if (dirich)
          {
            if (dirich->type() != Core::Conditions::PointDirichlet &&
                dirich->type() != Core::Conditions::LineDirichlet &&
                dirich->type() != Core::Conditions::SurfaceDirichlet &&
                dirich->type() != Core::Conditions::VolumeDirichlet)
              FOUR_C_THROW("condition with name Dirichlet is not of type Dirichlet");
            flag = &dirich->parameters().get<std::vector<int>>("ONOFF");
          }
          for (unsigned j = 0; j < size; ++j)
          {
            if (flag && (*flag)[j])
              lmdirich.push_back(1);
            else
              lmdirich.push_back(0);
          }
        }
      }
    }

    // fill the vector with element dofs
    const int owner = Truss3::owner();
    std::vector<int> dofs = dis.dof(dofset, this);
    if (dofs.size()) lmstride.push_back(dofs.size());
    for (int& dof : dofs)
    {
      lmowner.push_back(owner);
      lm.push_back(dof);
    }

    if (doDirichlet)
    {
      const std::vector<int>* flag = nullptr;
      Core::Conditions::Condition* dirich = get_condition("Dirichlet");
      if (dirich)
      {
        if (dirich->type() != Core::Conditions::PointDirichlet &&
            dirich->type() != Core::Conditions::LineDirichlet &&
            dirich->type() != Core::Conditions::SurfaceDirichlet &&
            dirich->type() != Core::Conditions::VolumeDirichlet)
          FOUR_C_THROW("condition with name Dirichlet is not of type Dirichlet");
        flag = &dirich->parameters().get<std::vector<int>>("ONOFF");
      }
      for (unsigned j = 0; j < dofs.size(); ++j)
      {
        if (flag && (*flag)[j])
          lmdirich.push_back(1);
        else
          lmdirich.push_back(0);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Discret::Elements::Truss3Type::initialize(Core::FE::Discretization& dis)
{
  // reference node positions
  std::vector<double> xrefe;

  // reference nodal tangent positions
  Core::LinAlg::Matrix<3, 1> trefNodeAux(true);
  // resize vectors for the number of coordinates we need to store
  xrefe.resize(3 * 2);

  // setting beam reference director correctly
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    // in case that current element is not a truss3 element there is nothing to do and we go back
    // to the head of the loop
    if (dis.l_col_element(i)->element_type() != *this) continue;

    // if we get so far current element is a truss3 element and  we get a pointer at it
    auto* currele = dynamic_cast<Discret::Elements::Truss3*>(dis.l_col_element(i));
    if (!currele) FOUR_C_THROW("cast to Truss3* failed");

    // getting element's nodal coordinates and treating them as reference configuration
    if (currele->nodes()[0] == nullptr || currele->nodes()[1] == nullptr)
      FOUR_C_THROW("Cannot get nodes in order to compute reference configuration'");
    else
    {
      for (int k = 0; k < 2; k++)  // element has two nodes
        for (int l = 0; l < 3; l++) xrefe[k * 3 + l] = currele->nodes()[k]->x()[l];
    }

    currele->set_up_reference_geometry(xrefe);
  }

  return 0;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Truss3::set_params_interface_ptr(const Teuchos::ParameterList& p)
{
  if (p.isParameter("interface"))
  {
    interface_ptr_ = std::dynamic_pointer_cast<Solid::Elements::ParamsInterface>(
        p.get<std::shared_ptr<Core::Elements::ParamsInterface>>("interface"));
  }
  else
    interface_ptr_ = nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::ParamsInterface> Discret::Elements::Truss3::params_interface_ptr()
{
  return interface_ptr_;
}

FOUR_C_NAMESPACE_CLOSE

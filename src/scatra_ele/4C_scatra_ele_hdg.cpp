// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_hdg.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization_faces.hpp"
#include "4C_fem_discretization_hdg.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_polynomial.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_myocard.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_factory.hpp"
#include "4C_scatra_ele_hdg_boundary_calc.hpp"
#include "4C_scatra_ele_hdg_intfaces_calc.hpp"
#include "4C_scatra_ele_interface.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


// initialize static variable
Discret::Elements::ScaTraHDGType Discret::Elements::ScaTraHDGType::instance_;
Discret::Elements::ScaTraHDGBoundaryType Discret::Elements::ScaTraHDGBoundaryType::instance_;
Discret::Elements::ScaTraHDGIntFaceType Discret::Elements::ScaTraHDGIntFaceType::instance_;


Discret::Elements::ScaTraHDGType& Discret::Elements::ScaTraHDGType::instance() { return instance_; }

Discret::Elements::ScaTraHDGBoundaryType& Discret::Elements::ScaTraHDGBoundaryType::instance()
{
  return instance_;
}

Discret::Elements::ScaTraHDGIntFaceType& Discret::Elements::ScaTraHDGIntFaceType::instance()
{
  return instance_;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::ScaTraHDGType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::ScaTraHDG* object = new Discret::Elements::ScaTraHDG(-1, -1);
  object->unpack(buffer);
  return object;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::ScaTraHDGType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "TRANSPHDG")
  {
    return std::make_shared<Discret::Elements::ScaTraHDG>(id, owner);
  }
  return nullptr;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::ScaTraHDGType::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::ScaTraHDG>(id, owner);
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDGType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 1;  // Only one scalar (so far) is the unknown that is solved for
  dimns = numdf;
  nv = numdf;

  if (Global::Problem::instance(0)->get_problem_type() == Core::ProblemType::elch)
  {
    if (nv > 1)  // only when we have more than 1 dof per node!
    {
      nv -= 1;  // ion concentrations
      np = 1;   // electric potential
    }
  }
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SerialDenseMatrix Discret::Elements::ScaTraHDGType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  Core::LinAlg::SerialDenseMatrix nullspace;
  FOUR_C_THROW("method ComputeNullSpace not implemented right now!");
  return nullspace;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDGType ::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_scatra;
  TransportType::setup_element_definition(definitions_scatra);

  std::map<std::string, Input::LineDefinition>& defs_scatra = definitions_scatra["TRANSP"];

  std::map<std::string, Input::LineDefinition>& defs = definitions["TRANSPHDG"];

  for (const auto& [key, scatra_line_def] : defs_scatra)
  {
    defs[key] = Input::LineDefinition::Builder(scatra_line_def)
                    .add_named_int("DEG")
                    .add_optional_named_int("SPC")
                    .build();
  }
}



/*----------------------------------------------------------------------*
 |  ctor (public)                                         hoermann 09/15|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::ScaTraHDG::ScaTraHDG(int id, int owner)
    : Transport(id, owner),
      diff1_(0.0),
      ndofs_(0),
      onfdofs_(0),
      onfdofs_old_(0),
      degree_(1),
      degree_old_(0),
      completepol_(true),
      padpatele_(true),
      matinit_(false)
{
}



/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                    hoermann 09/15|
 *----------------------------------------------------------------------*/
Discret::Elements::ScaTraHDG::ScaTraHDG(const Discret::Elements::ScaTraHDG& old)
    : Transport(old),
      diff1_(0.0),
      ndofs_(0),
      onfdofs_(0),
      onfdofs_old_(0),
      degree_(old.degree_),
      degree_old_(old.degree_old_),
      completepol_(old.completepol_),
      padpatele_(true),
      matinit_(false)
{
}



/*----------------------------------------------------------------------*
 |  Deep copy this instance of ScaTra and return pointer to it (public) |
 |                                                       hoermann 09/15 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::ScaTraHDG::clone() const
{
  Discret::Elements::ScaTraHDG* newelement = new Discret::Elements::ScaTraHDG(*this);
  return newelement;
}



/*----------------------------------------------------------------------*
 |  Pack data (public)                                   hoermann 09/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDG::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // add base class Element
  Transport::pack(data);

  int degree = degree_;
  add_to_pack(data, degree);
  degree = completepol_;
  add_to_pack(data, degree);
  degree = degree_old_;
  add_to_pack(data, degree);
}



/*----------------------------------------------------------------------*
 |  Unpack data (public)                                 hoermann 09/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDG::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Transport::unpack(buffer);

  int val = 0;
  extract_from_pack(buffer, val);
  FOUR_C_ASSERT(val >= 0 && val < 255, "Degree out of range");
  degree_ = val;
  extract_from_pack(buffer, val);
  completepol_ = val;
  extract_from_pack(buffer, val);
  degree_old_ = val;
}

/*----------------------------------------------------------------------*
 |  pack_material data (public)                           hoermann 12/16 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDG::pack_material(Core::Communication::PackBuffer& data) const
{
  // add material
  if (material() != nullptr)
  {
    // pack only first material
    material()->pack(data);
  }
  else
    FOUR_C_THROW("No material defined to pack!");
}

/*----------------------------------------------------------------------*
 |  UnPackMaterial data (public)                         hoermann 12/16 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDG::unpack_material(Core::Communication::UnpackBuffer& buffer) const
{
  std::shared_ptr<Core::Mat::Material> mat = material();
  if (mat->material_type() == Core::Materials::m_myocard)
  {
    // Note: We need to do a dynamic_cast here
    std::shared_ptr<Mat::Myocard> actmat = std::dynamic_pointer_cast<Mat::Myocard>(mat);
    actmat->unpack_material(buffer);
  }
  else
    FOUR_C_THROW("No material defined to unpack!");
}

/*----------------------------------------------------------------------*
 |  init the element                                     hoermann 12/16 |
 *----------------------------------------------------------------------*/
int Discret::Elements::ScaTraHDG::initialize()
{
  std::shared_ptr<Core::Mat::Material> mat = material();
  // for now, we only need to do something in case of reactions (for the initialization of functions
  // in case of reactions by function)
  if (mat->material_type() == Core::Materials::m_myocard)
  {
    int gp;
    // Note: We need to do a dynamic_cast here
    std::shared_ptr<Mat::Myocard> actmat = std::dynamic_pointer_cast<Mat::Myocard>(mat);
    int deg = 0;
    if (degree_old_ == 1)
      deg = 4 * degree_old_;
    else
      deg = 3 * degree_old_;
    if (this->shape() == Core::FE::CellType::tet4 or this->shape() == Core::FE::CellType::tet10)
    {
      switch (deg)
      {
        case 0:
          gp = 1;
          break;
        case 3:
          gp = 5;
          break;
        case 4:
          gp = 11;
          break;
        case 6:
          gp = 24;
          break;
        case 9:
          gp = 125;
          break;
        case 12:
          gp = 343;
          break;
        case 15:
          gp = 729;
          break;
        default:
          FOUR_C_THROW(
              "Integration rule for TET elements only until polynomial order 5 for TET defined. "
              "You specified a degree of %d ",
              degree_old_);
          gp = 0;
          break;
      }
    }
    else
    {
      std::shared_ptr<Core::FE::GaussPoints> quadrature_(
          Core::FE::create_gauss_points(this->shape(), deg));
      gp = quadrature_->num_points();
    }
    if (actmat->parameter() != nullptr and
        !actmat->myocard_mat())  // in case we are not in post-process mode
    {
      actmat->set_gp(gp);
      actmat->initialize();
    }
  }

  return 0;
}

/*----------------------------------------------------------------------*
 |  Read element from input (public)                     hoermann 09/15 |
 *----------------------------------------------------------------------*/
bool Discret::Elements::ScaTraHDG::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container)
{
  bool success = Transport::read_element(eletype, distype, container);
  degree_ = container.get<int>("DEG");
  degree_old_ = degree_;

  completepol_ = container.get_or<int>("SPC", false);

  return success;
}


/*----------------------------------------------------------------------*
 |  get vector of lines              (public)             hoermann 09/15|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::ScaTraHDG::lines()
{
  return Core::Communication::get_element_lines<ScaTraHDGBoundary, ScaTraHDG>(*this);
}


/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                       hoermann 09/15|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::ScaTraHDG::surfaces()
{
  return Core::Communication::get_element_surfaces<ScaTraHDGBoundary>(*this);
}


/*----------------------------------------------------------------------*
 |  get face element (public)                             hoermann 09/15|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::ScaTraHDG::create_face_element(
    Core::Elements::Element* parent_slave,  //!< parent slave fluid3 element
    int nnode,                              //!< number of surface nodes
    const int* nodeids,                     //!< node ids of surface element
    Core::Nodes::Node** nodes,              //!< nodes of surface element
    const int lsurface_master,              //!< local surface number w.r.t master parent element
    const int lsurface_slave,               //!< local surface number w.r.t slave parent element
    const std::vector<int>& localtrafomap   //! local trafo map
)
{
  // dynamic cast for slave parent element
  Discret::Elements::ScaTraHDG* slave_pele =
      dynamic_cast<Discret::Elements::ScaTraHDG*>(parent_slave);


  // insert both parent elements
  return Core::Communication::element_int_face_factory<ScaTraHDGIntFace, ScaTraHDG>(
      -1,               //!< internal face element id
      -1,               //!< owner of internal face element
      nnode,            //!< number of surface nodes
      nodeids,          //!< node ids of surface element
      nodes,            //!< nodes of surface element
      this,             //!< master parent element
      slave_pele,       //!< slave parent element
      lsurface_master,  //!< local surface number w.r.t master parent element
      lsurface_slave,   //!< local surface number w.r.t slave parent element
      localtrafomap     //!< local trafo map
  );
}


/*---------------------------------------------------------------------*
|  evaluate the element (public)                         hoermann 09/15|
*----------------------------------------------------------------------*/
int Discret::Elements::ScaTraHDG::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // we assume here, that numdofpernode is equal for every node within
  // the discretization and does not change during the computations
  const int numdofpernode = num_dof_per_node(*(nodes()[0]));
  int numscal = numdofpernode;

  // get the action required
  ScaTra::Action act;
  if (params.get<bool>("hdg_action", false))
  {
    switch (Teuchos::getIntegralValue<Core::FE::HDGAction>(params, "action"))
    {
      case Core::FE::HDGAction::project_dirich_field:
        act = ScaTra::Action::project_dirich_field;
        break;
      default:
        FOUR_C_THROW("HDG Action type not supported");
    }
  }
  else
  {
    act = Teuchos::getIntegralValue<ScaTra::Action>(params, "action");
  }

  // get material
  std::shared_ptr<Core::Mat::Material> mat = material();

  // switch between different physical types as used below
  switch (act)
  {
    //-----------------------------------------------------------------------
    // standard implementation enabling time-integration schemes such as
    // one-step-theta, BDF2, and generalized-alpha (n+alpha_F and n+1)
    //-----------------------------------------------------------------------
    case ScaTra::Action::calc_mat_and_rhs:
    {
      return Discret::Elements::ScaTraFactory::provide_impl_hdg(
          shape(), impl_type(), numdofpernode, numscal, discretization.name())
          ->evaluate(this, params, discretization, la, elemat1, elemat2, elevec1, elevec2, elevec3);
    }
    break;

    case ScaTra::Action::interpolate_hdg_to_node:
    case ScaTra::Action::update_interior_variables:
    case ScaTra::Action::project_dirich_field:
    case ScaTra::Action::project_material_field:
    case ScaTra::Action::project_neumann_field:
    case ScaTra::Action::set_initial_field:
    case ScaTra::Action::time_update_material:
    case ScaTra::Action::get_material_internal_state:
    case ScaTra::Action::set_material_internal_state:
    case ScaTra::Action::calc_mat_initial:
    case ScaTra::Action::project_field:
    case ScaTra::Action::calc_padaptivity:
    case ScaTra::Action::calc_error:

    {
      return Discret::Elements::ScaTraFactory::provide_impl_hdg(
          shape(), impl_type(), numdofpernode, numscal, discretization.name())
          ->evaluate_service(
              this, params, discretization, la, elemat1, elemat2, elevec1, elevec2, elevec3);
      break;
    }

    case ScaTra::Action::calc_initial_time_deriv:
    case ScaTra::Action::set_general_scatra_parameter:
    case ScaTra::Action::set_nodeset_parameter:
    case ScaTra::Action::set_time_parameter:
    case ScaTra::Action::set_turbulence_scatra_parameter:
      break;

    default:
      FOUR_C_THROW("Unknown type of action '%i' for ScaTraHDG", act);
      break;
  }  // switch(action)

  return 0;
}  // Discret::Elements::ScaTra::Evaluate


/*----------------------------------------------------------------------*
 |  print this element (public)                           hoermann 09/15|
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDG::print(std::ostream& os) const
{
  os << "ScaTraHDG ";
  Element::print(os);
}



//===========================================================================



std::shared_ptr<Core::Elements::Element> Discret::Elements::ScaTraHDGBoundaryType::create(
    const int id, const int owner)
{
  return nullptr;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                        hoermann 09/15 |
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::ScaTraHDGBoundary::ScaTraHDGBoundary(int id, int owner, int nnode,
    const int* nodeids, Core::Nodes::Node** nodes, Core::Elements::Element* parent,
    const int lsurface)
    : Core::Elements::FaceElement(id, owner)
{
  set_parent_master_element(parent, lsurface);
  set_node_ids(nnode, nodeids);
  build_nodal_pointers(nodes);
  return;
}


/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                   hoermann 09/15 |
 *----------------------------------------------------------------------*/
Discret::Elements::ScaTraHDGBoundary::ScaTraHDGBoundary(
    const Discret::Elements::ScaTraHDGBoundary& old)
    : Core::Elements::FaceElement(old)
{
  return;
}


/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               (public) |
 |                                                       hoermann 09/15 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::ScaTraHDGBoundary::clone() const
{
  Discret::Elements::ScaTraHDGBoundary* newelement =
      new Discret::Elements::ScaTraHDGBoundary(*this);
  return newelement;
}


/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                        hoermann 09/15|
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::ScaTraHDGBoundary::shape() const
{
  return Core::FE::get_shape_of_boundary_element(num_node(), parent_master_element()->shape());
}


/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                        hoermann 09/15|
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDGBoundary::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Element::pack(data);

  // Discretisation type
  // add_to_pack(data,distype_);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                        hoermann 09/15|
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDGBoundary::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);



  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                          hoermann 09/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDGBoundary::print(std::ostream& os) const
{
  os << "ScaTraHDGBoundary ";
  Element::print(os);
  return;
}


/*----------------------------------------------------------------------*
 |  get vector of lines (public)                         hoermann 09/15 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::ScaTraHDGBoundary::lines()
{
  FOUR_C_THROW("Lines of ScaTraHDGBoundary not implemented");
}


/*----------------------------------------------------------------------*
 |  get vector of lines (public)                         hoermann 09/15 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::ScaTraHDGBoundary::surfaces()
{
  FOUR_C_THROW("Surfaces of ScaTraHDGBoundary not implemented");
}


/*----------------------------------------------------------------------*
 |  evaluate the element (public)                        hoermann 09/15 |
 *----------------------------------------------------------------------*/
int Discret::Elements::ScaTraHDGBoundary::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  return 0;
}


/*----------------------------------------------------------------------*
 |  Integrate a surface/line Neumann boundary condition  hoermann 09/15 |
 *----------------------------------------------------------------------*/
int Discret::Elements::ScaTraHDGBoundary::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)

{
  // add Neumann boundary condition to parameter list
  params.set<Core::Conditions::Condition*>("condition", &condition);

  // build location array from location vector
  //(this a little ugly. one could fix this by introducing a evaluate_neumann() method
  // with LocationArray as input in the Core::Elements::Element ...)
  Core::Elements::LocationArray la(1);
  la[0].lm_ = lm;

  Discret::Elements::ScaTraHDGBoundaryImplInterface::impl(this)->evaluate_neumann(
      this, params, discretization, la, *elemat1, elevec1);

  return 0;
}

/*----------------------------------------------------------------------*
 |  Get degrees of freedom used by this element (public) hoermann 09/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDGBoundary::location_vector(const Core::FE::Discretization& dis,
    Core::Elements::LocationArray& la, bool doDirichlet, const std::string& condstring,
    Teuchos::ParameterList& params) const
{
  // we have to do it this way
  parent_master_element()->location_vector(dis, la, false);
  return;
}


/*----------------------------------------------------------------------*
 |  Get degrees of freedom used by this element (public) hoermann 09/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDGBoundary::location_vector(const Core::FE::Discretization& dis,
    std::vector<int>& lm, std::vector<int>& lmowner, std::vector<int>& lmstride) const
{
  // we have to do it this way
  parent_master_element()->location_vector(dis, lm, lmowner, lmstride);
  return;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::ScaTraHDGIntFaceType::create(
    const int id, const int owner)
{
  return nullptr;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                         hoermann 09/15|
 *----------------------------------------------------------------------*/
Discret::Elements::ScaTraHDGIntFace::ScaTraHDGIntFace(int id,  ///< element id
    int owner,                  ///< owner (= owner of parent element with smallest gid)
    int nnode,                  ///< number of nodes
    const int* nodeids,         ///< node ids
    Core::Nodes::Node** nodes,  ///< nodes of surface
    Discret::Elements::ScaTraHDG* parent_master,  ///< master parent element
    Discret::Elements::ScaTraHDG* parent_slave,   ///< slave parent element
    const int lsurface_master,  ///< local surface index with respect to master parent element
    const int lsurface_slave,   ///< local surface index with respect to slave parent element
    const std::vector<int>
        localtrafomap  ///< get the transformation map between the local coordinate systems of the
                       ///< face w.r.t the master parent element's face's coordinate system and the
                       ///< slave element's face's coordinate system
    )
    : Core::Elements::FaceElement(id, owner), degree_(0), degree_old_(0)
{
  set_parent_master_element(parent_master, lsurface_master);
  set_parent_slave_element(parent_slave, lsurface_slave);

  if (parent_slave != nullptr)
  {
    degree_ = std::max(parent_master->degree(), parent_slave->degree());
    degree_old_ = std::max(parent_master->degree_old(), parent_slave->degree_old());
  }
  else
  {
    degree_ = parent_master->degree();
    degree_old_ = parent_master->degree_old();
  }

  set_local_trafo_map(localtrafomap);

  set_node_ids(nnode, nodeids);
  build_nodal_pointers(nodes);
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                    hoermann 09/15|
 *----------------------------------------------------------------------*/
Discret::Elements::ScaTraHDGIntFace::ScaTraHDGIntFace(
    const Discret::Elements::ScaTraHDGIntFace& old)
    : Core::Elements::FaceElement(old), degree_(old.degree_), degree_old_(old.degree_old_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               (public) |
 |                                                        hoermann 09/15|
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::ScaTraHDGIntFace::clone() const
{
  Discret::Elements::ScaTraHDGIntFace* newelement = new Discret::Elements::ScaTraHDGIntFace(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                       hoermann 09/15 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::ScaTraHDGIntFace::shape() const
{
  // could be called for master parent or slave parent element, doesn't matter
  return Core::FE::get_shape_of_boundary_element(num_node(), parent_master_element()->shape());
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                       hoermann 09/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDGIntFace::pack(Core::Communication::PackBuffer& data) const
{
  FOUR_C_THROW("this ScaTraHDGIntFace element does not support communication");
  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                       hoermann 09/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDGIntFace::unpack(Core::Communication::UnpackBuffer& buffer)
{
  FOUR_C_THROW("this ScaTraHDGIntFace element does not support communication");
  return;
}



/*----------------------------------------------------------------------*
 |  create the patch location vector (public)            hoermann 09/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDGIntFace::patch_location_vector(
    Core::FE::Discretization& discretization,  ///< discretization
    std::vector<int>& nds_master,              ///< nodal dofset w.r.t master parent element
    std::vector<int>& nds_slave,               ///< nodal dofset w.r.t slave parent element
    std::vector<int>& patchlm,                 ///< local map for gdof ids for patch of elements
    std::vector<int>& master_lm,               ///< local map for gdof ids for master element
    std::vector<int>& slave_lm,                ///< local map for gdof ids for slave element
    std::vector<int>& face_lm,                 ///< local map for gdof ids for face element
    std::vector<int>& lm_masterToPatch,        ///< local map between lm_master and lm_patch
    std::vector<int>& lm_slaveToPatch,         ///< local map between lm_slave and lm_patch
    std::vector<int>& lm_faceToPatch,          ///< local map between lm_face and lm_patch
    std::vector<int>& lm_masterNodeToPatch,  ///< local map between master nodes and nodes in patch
    std::vector<int>& lm_slaveNodeToPatch    ///< local map between slave nodes and nodes in patch
)
{
  // create one patch location vector containing all dofs of master, slave and
  // *this ScaTraHDGIntFace element only once (no duplicates)

  //-----------------------------------------------------------------------
  const int m_numnode = parent_master_element()->num_node();
  Core::Nodes::Node** m_nodes = parent_master_element()->nodes();

  if (m_numnode != static_cast<int>(nds_master.size()))
  {
    FOUR_C_THROW("wrong number of nodes for master element");
  }

  //-----------------------------------------------------------------------
  const int s_numnode = parent_slave_element()->num_node();
  Core::Nodes::Node** s_nodes = parent_slave_element()->nodes();

  if (s_numnode != static_cast<int>(nds_slave.size()))
  {
    FOUR_C_THROW("wrong number of nodes for slave element");
  }

  //-----------------------------------------------------------------------
  const int f_numnode = num_node();
  Core::Nodes::Node** f_nodes = nodes();

  //-----------------------------------------------------------------------
  // create the patch local map and additional local maps between elements lm and patch lm

  patchlm.clear();

  master_lm.clear();
  slave_lm.clear();
  face_lm.clear();

  lm_masterToPatch.clear();
  lm_slaveToPatch.clear();
  lm_faceToPatch.clear();

  // maps between master/slave nodes and nodes in patch
  lm_masterNodeToPatch.clear();
  lm_slaveNodeToPatch.clear();

  // for each master node, the offset for node's dofs in master_lm
  std::map<int, int> m_node_lm_offset;


  // ---------------------------------------------------
  int dofset = 0;  // assume dofset 0

  int patchnode_count = 0;

  // fill patch lm with master's nodes
  for (int k = 0; k < m_numnode; ++k)
  {
    Core::Nodes::Node* node = m_nodes[k];
    std::vector<int> dof = discretization.dof(dofset, node);

    // get maximum of numdof per node with the help of master and/or slave element (returns 4 in 3D
    // case, does not return dofset's numnode)
    const int size = discretization.num_dof(dofset, node);
    const int offset = size * nds_master[k];

    FOUR_C_ASSERT(
        dof.size() >= static_cast<unsigned>(offset + size), "illegal physical dofs offset");

    // insert a pair of node-Id and current length of master_lm ( to get the start offset for node's
    // dofs)
    m_node_lm_offset.insert(std::pair<int, int>(node->id(), master_lm.size()));

    for (int j = 0; j < size; ++j)
    {
      int actdof = dof[offset + j];

      // current last index will be the index for next push_back operation
      lm_masterToPatch.push_back((patchlm.size()));

      patchlm.push_back(actdof);
      master_lm.push_back(actdof);
    }

    lm_masterNodeToPatch.push_back(patchnode_count);

    patchnode_count++;
  }

  // ---------------------------------------------------
  // fill patch lm with missing slave's nodes and extract slave's lm from patch_lm

  for (int k = 0; k < s_numnode; ++k)
  {
    Core::Nodes::Node* node = s_nodes[k];

    // slave node already contained?
    std::map<int, int>::iterator m_offset;
    m_offset = m_node_lm_offset.find(node->id());

    if (m_offset == m_node_lm_offset.end())  // node not included yet
    {
      std::vector<int> dof = discretization.dof(dofset, node);

      // get maximum of numdof per node with the help of master and/or slave element (returns 4 in
      // 3D case, does not return dofset's numnode)
      const int size = discretization.num_dof(dofset, node);
      const int offset = size * nds_slave[k];

      FOUR_C_ASSERT(
          dof.size() >= static_cast<unsigned>(offset + size), "illegal physical dofs offset");
      for (int j = 0; j < size; ++j)
      {
        int actdof = dof[offset + j];

        lm_slaveToPatch.push_back(patchlm.size());

        patchlm.push_back(actdof);
        slave_lm.push_back(actdof);
      }

      lm_slaveNodeToPatch.push_back(patchnode_count);

      patchnode_count++;
    }
    else  // node is also a master's node
    {
      const int size = discretization.num_dof(dofset, node);

      int offset = m_offset->second;

      for (int j = 0; j < size; ++j)
      {
        int actdof = master_lm[offset + j];

        slave_lm.push_back(actdof);

        // copy from lm_masterToPatch
        lm_slaveToPatch.push_back(lm_masterToPatch[offset + j]);
      }

      if (offset % size != 0)
        FOUR_C_THROW("there was at least one node with not %d dofs per node", size);
      int patchnode_index = offset / size;

      lm_slaveNodeToPatch.push_back(patchnode_index);
      // no patchnode_count++; (node already contained)
    }
  }

  // ---------------------------------------------------
  // extract face's lm from patch_lm
  for (int k = 0; k < f_numnode; ++k)
  {
    Core::Nodes::Node* node = f_nodes[k];

    // face node must be contained
    std::map<int, int>::iterator m_offset;
    m_offset = m_node_lm_offset.find(node->id());

    if (m_offset != m_node_lm_offset.end())  // node not included yet
    {
      const int size = discretization.num_dof(dofset, node);

      int offset = m_offset->second;

      for (int j = 0; j < size; ++j)
      {
        int actdof = master_lm[offset + j];

        face_lm.push_back(actdof);

        // copy from lm_masterToPatch
        lm_faceToPatch.push_back(lm_masterToPatch[offset + j]);
      }
    }
    else
      FOUR_C_THROW("face's nodes not contained in masternodes_offset map");
  }

  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                          hoermann 09/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ScaTraHDGIntFace::print(std::ostream& os) const
{
  os << "ScaTraHDGIntFace ";
  Element::print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                         hoermann 09/15 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::ScaTraHDGIntFace::lines()
{
  FOUR_C_THROW("Lines of ScaTraHDGIntFace not implemented");
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                         hoermann 09/15 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::ScaTraHDGIntFace::surfaces()
{
  FOUR_C_THROW("Surfaces of ScaTraHDGIntFace not implemented");
}

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                        hoermann 09/15 |
 *----------------------------------------------------------------------*/
int Discret::Elements::ScaTraHDGIntFace::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // REMARK: this line ensures that the static
  // Discret::Elements::ScaTraHDGIntFaceImplInterface::Impl is created
  //         this line avoids linker errors
  Discret::Elements::ScaTraHDGIntFaceImplInterface::impl(this);

  FOUR_C_THROW("not available");

  return 0;
}


/*----------------------------------------------------------------------*
 |  Integrate a surface/line Neumann boundary condition  hoermann 09/15 |
 *----------------------------------------------------------------------*/
int Discret::Elements::ScaTraHDGIntFace::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  FOUR_C_THROW("not available");

  return 0;
}

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_geometry_periodic_boundingbox.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_fem_dofset_independent.hpp"
#include "4C_fem_geometry_intersection_math.hpp"
#include "4C_io.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_visualization_parameters.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::Geo::MeshFree::BoundingBox::BoundingBox()
    : isinit_(false),
      issetup_(false),
      boxdiscret_(nullptr),
      disn_row_(nullptr),
      disn_col_(nullptr),
      empty_(true),
      haveperiodicbc_(false),
      havedirichletbc_(false),
      box_(Core::LinAlg::Initialization::zero),
      visualization_output_writer_ptr_(nullptr)
{
  // initialize arrays
  for (int idim = 0; idim < 3; ++idim)
  {
    pbconoff_[idim] = false;
    edgelength_[idim] = 0.0;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::init(const Teuchos::ParameterList& binning_params)
{
  issetup_ = false;

  // get bounding box specified in the input file
  // fixme: like this or by eight nodes of element in discret
  box_.put_scalar(1.0e12);
  std::istringstream xaabbstream(
      Teuchos::getNumericStringParameter(binning_params, "DOMAINBOUNDINGBOX"));
  for (int col = 0; col < 2; ++col)
  {
    for (int row = 0; row < 3; ++row)
    {
      std::string value;
      if (xaabbstream >> value)
      {
        double doublevalue = std::atof(value.c_str());
        box_(row, col) = doublevalue;
      }
      else
        FOUR_C_THROW(
            " Specify six values for bounding box in three dimensional problem."
            " Fix your input file.");
    }
  }

  // set up boundary conditions
  std::istringstream periodicbc(
      Teuchos::getNumericStringParameter(binning_params, "PERIODICONOFF"));

  // loop over all spatial directions
  for (int dim = 0; dim < 3; ++dim)
  {
    std::string val;
    if (periodicbc >> val)
    {
      int intval = std::atoi(val.c_str());
      if (intval)
      {
        // set flag
        pbconoff_[dim] = true;

        // set global flag
        haveperiodicbc_ = true;
      }

      // edge length of box
      edgelength_[dim] = box_(dim, 1) - box_(dim, 0);
    }
    else
    {
      FOUR_C_THROW(
          "Enter three values to specify each direction as periodic or non periodic."
          "Fix your input file ...");
    }
  }

  // todo
  havedirichletbc_ = false;
  empty_ = false;
  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::init(
    Core::LinAlg::Matrix<3, 2> const& box, std::vector<bool> const& pbconoff)
{
  issetup_ = false;

  // get bounding box specified in the input file
  box_ = box;

  // loop over all spatial directions
  for (unsigned int dim = 0; dim < 3; ++dim)
  {
    pbconoff_[dim] = pbconoff[dim];
    if (pbconoff_[dim]) haveperiodicbc_ = true;

    // edge length of box
    edgelength_[dim] = box_(dim, 1) - box_(dim, 0);
  }

  havedirichletbc_ = false;
  empty_ = false;
  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::setup(const Teuchos::ParameterList& io_params,
    std::shared_ptr<Core::FE::Discretization> boundingbox_dis, MPI_Comm comm, const int n_dim,
    const Core::IO::OutputControl& output_control)
{
  throw_if_not_init();

  // initialize bounding box discretization
  setup_bounding_box_discretization(boundingbox_dis, comm, n_dim);

  if (boxdiscret_->has_condition("Dirichlet")) havedirichletbc_ = true;

  // displacement vector in row and col format
  disn_row_ = Core::LinAlg::create_vector(*boxdiscret_->dof_row_map(), true);
  disn_col_ = Core::LinAlg::create_vector(*boxdiscret_->dof_col_map(), true);

  // initialize bounding box runtime output
  if (io_params.sublist("RUNTIME VTK OUTPUT").get<int>("INTERVAL_STEPS") != -1)
    init_runtime_output(io_params, output_control);

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 * (public)                                                                   |
 *----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::setup_bounding_box_discretization(
    std::shared_ptr<Core::FE::Discretization> boundingbox_dis, MPI_Comm comm, const int n_dim)
{
  if (boundingbox_dis != nullptr)
  {
    boxdiscret_ = boundingbox_dis;

    if (boxdiscret_->filled() == false)
      boxdiscret_->fill_complete({
          .assign_degrees_of_freedom = true,
          .init_elements = false,
          .do_boundary_conditions = false,
      });

    // create fully overlapping boundingbox discret
    std::shared_ptr<Core::LinAlg::Map> rednodecolmap =
        Core::LinAlg::allreduce_e_map(*boxdiscret_->node_row_map());
    std::shared_ptr<Core::LinAlg::Map> redelecolmap =
        Core::LinAlg::allreduce_e_map(*boxdiscret_->element_row_map());

    // do the fully overlapping ghosting of the bounding box element to have everything redundant
    boxdiscret_->export_column_nodes(*rednodecolmap);
    boxdiscret_->export_column_elements(*redelecolmap);

    boxdiscret_->fill_complete({
        .assign_degrees_of_freedom = true,
        .init_elements = false,
        .do_boundary_conditions = false,
    });
  }

  if (boundingbox_dis == nullptr or boxdiscret_->num_my_col_elements() == 0)
  {
    if (boundingbox_dis == nullptr)
    {
      boxdiscret_ = std::make_shared<Core::FE::Discretization>("boundingbox", comm, n_dim);
    }
    else
    {
      boxdiscret_ = boundingbox_dis;
    }

    // create nodes
    std::vector<double> cornerpos(3, 0.0);
    int node_ids[8];
    for (int corner_i = 0; corner_i < 8; ++corner_i)
    {
      undeformed_box_corner_point_position(corner_i, cornerpos);
      node_ids[corner_i] = corner_i;

      std::shared_ptr<Core::Nodes::Node> newnode =
          std::make_shared<Core::Nodes::Node>(corner_i, cornerpos, 0);
      boxdiscret_->add_node(newnode->x(), newnode->id(), newnode);
    }

    // assign nodes to element
    std::shared_ptr<Core::Elements::Element> newele =
        Core::Communication::factory("VELE3", "Polynomial", 0, 0);
    newele->set_node_ids(8, node_ids);
    boxdiscret_->add_element(newele);
  }

  // build independent dof set
  std::shared_ptr<Core::DOFSets::IndependentDofSet> independentdofset =
      std::make_shared<Core::DOFSets::IndependentDofSet>(true);
  boxdiscret_->replace_dof_set(independentdofset);
  boxdiscret_->fill_complete();
}

/*----------------------------------------------------------------------------*
 * (public)                                                                   |
 *----------------------------------------------------------------------------*/
bool Core::Geo::MeshFree::BoundingBox::shift_3d(
    Core::LinAlg::Matrix<3, 1>& d, Core::LinAlg::Matrix<3, 1> const X) const
{
  throw_if_not_init();

  bool shifted = false;

  if (not haveperiodicbc_) return shifted;

  // x = X + d
  Core::LinAlg::Matrix<3, 1> x(X);
  x.update(1.0, d, 1.0);

  Core::LinAlg::Matrix<3, 1> x_ud(Core::LinAlg::Initialization::zero);
  transform_from_global_to_undeformed_bounding_box_system(x, x_ud);

  // shift
  for (int dim = 0; dim < 3; ++dim)
    if (shift_1d(dim, x_ud(dim))) shifted = true;

  x.clear();
  d.clear();
  transform_from_undeformed_bounding_box_system_to_global(x_ud, x);

  // d = x - X
  d.update(1.0, x, -1.0, X);

  return shifted;
}

/*----------------------------------------------------------------------------*
 * (public)                                                                   |
 *----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::get_xi_of_intersection_3d(
    Core::LinAlg::Matrix<3, 1> const& x1, Core::LinAlg::Matrix<3, 1> const& x2,
    Core::LinAlg::Matrix<3, 1>& xi) const
{
  throw_if_not_init();
  get_xi_of_intersection_3d(x1, x2, xi, box_);
}
/*----------------------------------------------------------------------------*
 * (public)                                                                   |
 *----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::get_xi_of_intersection_3d(
    Core::LinAlg::Matrix<3, 1> const& x1, Core::LinAlg::Matrix<3, 1> const& x2,
    Core::LinAlg::Matrix<3, 1>& xi, Core::LinAlg::Matrix<3, 2> const& box) const
{
  throw_if_not_init();

  // set default values
  for (unsigned int dim = 0; dim < 3; ++dim) xi(dim) = 2.0;

  Core::LinAlg::Matrix<3, 1> x1_ud(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> x2_ud(Core::LinAlg::Initialization::zero);
  transform_from_global_to_undeformed_bounding_box_system(x1, x1_ud);
  transform_from_global_to_undeformed_bounding_box_system(x2, x2_ud);

  // check which points resides within in which direction
  std::vector<bool> x1_within_in_dir(3, true), x2_within_in_dir(3, true);
  within(box, x1_ud, x1_within_in_dir);
  within(box, x2_ud, x2_within_in_dir);

  for (unsigned int dim = 0; dim < 3; ++dim)
  {
    bool x1_in = x1_within_in_dir[dim];
    bool x2_in = x2_within_in_dir[dim];

    // no intersection in case both are in or both are out
    if ((x1_in or not x2_in) and (x2_in or not x1_in)) continue;

    // x1 out
    if (x2_in)
    {
      if (x1_ud(dim) < box_min(box, dim))
      {
        xi(dim) = (2.0 * box_min(box, dim) - (x2_ud(dim) + x1_ud(dim))) / (x2_ud(dim) - x1_ud(dim));
      }
      else if (x1_ud(dim) > box_max(box, dim))
      {
        xi(dim) = (2.0 * box_max(box, dim) - (x2_ud(dim) + x1_ud(dim))) / (x1_ud(dim) - x2_ud(dim));
      }
      else
      {
        FOUR_C_THROW("Your true/false logic is wrong.");
      }
    }
    else if (x1_in)
    {
      if (x2_ud(dim) < box_min(box, dim))
      {
        xi(dim) = (2.0 * box_min(box, dim) - (x2_ud(dim) + x1_ud(dim))) / (x1_ud(dim) - x2_ud(dim));
      }
      else if (x2_ud(dim) > box_max(box, dim))
      {
        xi(dim) = (2.0 * box_max(box, dim) - (x2_ud(dim) + x1_ud(dim))) / (x2_ud(dim) - x1_ud(dim));
      }
      else
      {
        FOUR_C_THROW("Your true/false logic is wrong.");
      }
    }
    else
    {
      FOUR_C_THROW("Your true/false logic is wrong.");
    }
  }
}

/*----------------------------------------------------------------------------*
 * (public)                                                                   |
 *----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::un_shift_3d(Core::LinAlg::Matrix<3, 1>& d,
    Core::LinAlg::Matrix<3, 1> const& ref, Core::LinAlg::Matrix<3, 1> const X) const
{
  throw_if_not_init();

  if (not haveperiodicbc_) return;

  // x = X + d
  Core::LinAlg::Matrix<3, 1> x(X);
  x.update(1.0, d, 1.0);

  Core::LinAlg::Matrix<3, 1> x_ud(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> ref_ud(Core::LinAlg::Initialization::zero);
  transform_from_global_to_undeformed_bounding_box_system(x, x_ud);
  transform_from_global_to_undeformed_bounding_box_system(ref, ref_ud);

  for (int dim = 0; dim < 3; ++dim) un_shift_1d(dim, x_ud(dim), ref_ud(dim));

  x.clear();
  d.clear();
  transform_from_undeformed_bounding_box_system_to_global(x_ud, x);

  // d = x - X
  d.update(1.0, x, -1.0, X);
}

/*----------------------------------------------------------------------------*
 * (public)                                                                   |
 *----------------------------------------------------------------------------*/
bool Core::Geo::MeshFree::BoundingBox::check_if_shift_between_points(Core::LinAlg::Matrix<3, 1>& d,
    Core::LinAlg::Matrix<3, 1> const& ref, std::vector<bool>& shift_in_dim,
    Core::LinAlg::Matrix<3, 1> const X) const
{
  throw_if_not_init();

  shift_in_dim.clear();
  shift_in_dim.resize(3);

  if (not haveperiodicbc_)
  {
    shift_in_dim = std::vector<bool>(3, false);
    return false;
  }

  // x = X + d
  Core::LinAlg::Matrix<3, 1> x(X);
  x.update(1.0, d, 1.0);

  Core::LinAlg::Matrix<3, 1> x_ud(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> ref_ud(Core::LinAlg::Initialization::zero);
  transform_from_global_to_undeformed_bounding_box_system(x, x_ud);
  transform_from_global_to_undeformed_bounding_box_system(ref, ref_ud);

  for (int dim = 0; dim < 3; ++dim) shift_in_dim[dim] = un_shift_1d(dim, x_ud(dim), ref_ud(dim));

  x.clear();
  d.clear();
  transform_from_undeformed_bounding_box_system_to_global(x_ud, x);

  // d = x - X
  d.update(1.0, x, -1.0, X);

  return (shift_in_dim[0] or shift_in_dim[1] or shift_in_dim[2]);
}

/*----------------------------------------------------------------------------*
 * (private)                                                                   |
 *----------------------------------------------------------------------------*/
bool Core::Geo::MeshFree::BoundingBox::shift_1d(int dim, double& d, double const& X) const
{
  throw_if_not_init();

  bool shifted = false;

  if (not pbconoff_[dim]) return shifted;

  double x = d + X;

  if (x < box_min(dim))
  {
    shifted = true;
    d += edgelength_[dim] * std::ceil(std::abs((x - box_(dim, 0)) / edgelength_[dim]));
  }
  else if (x > box_max(dim))
  {
    shifted = true;
    d -= edgelength_[dim] * std::ceil(std::abs((x - box_(dim, 1)) / edgelength_[dim]));
  }

  return shifted;
}

/*----------------------------------------------------------------------------*
 * (private)                                                                   |
 *----------------------------------------------------------------------------*/
bool Core::Geo::MeshFree::BoundingBox::un_shift_1d(
    int dim, double& d, double const& ref, double const& X) const
{
  throw_if_not_init();

  bool unshifted = false;

  if (not pbconoff_[dim]) return unshifted;

  double x = d + X;

  if (x - ref < -0.5 * edgelength_[dim])
  {
    unshifted = true;
    d += edgelength_[dim];
  }
  else if (x - ref > 0.5 * edgelength_[dim])
  {
    unshifted = true;
    d -= edgelength_[dim];
  }

  return unshifted;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Core::Geo::MeshFree::BoundingBox::in_between(
    double smin, double smax, double omin, double omax) const
{
  double tol = Core::Geo::TOL7;
  return ((omax > smin - tol) and (smax > omin - tol));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::random_pos_within(
    Core::LinAlg::Matrix<3, 1>& randpos, Core::Utils::Random* random) const
{
  throw_if_not_init();

  random->set_rand_range(0.0, 1.0);
  std::vector<double> randuni;
  random->uni(randuni, 3);

  Core::LinAlg::Matrix<3, 1> randpos_ud(Core::LinAlg::Initialization::zero);
  for (int dim = 0; dim < 3; ++dim)
    randpos_ud(dim) = box_min(dim) + (edgelength_[dim] * randuni[dim]);

  transform_from_undeformed_bounding_box_system_to_global(randpos_ud, randpos);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::add_point(const double* x)
{
  FOUR_C_THROW("Check before use.");
  if (empty_)
  {
    empty_ = false;
    box_(0, 0) = box_(0, 1) = x[0];
    box_(1, 0) = box_(1, 1) = x[1];
    box_(2, 0) = box_(2, 1) = x[2];
  }
  else
  {
    box_(0, 0) = std::min(box_(0, 0), x[0]);
    box_(1, 0) = std::min(box_(1, 0), x[1]);
    box_(2, 0) = std::min(box_(2, 0), x[2]);
    box_(0, 1) = std::max(box_(0, 1), x[0]);
    box_(1, 1) = std::max(box_(1, 1), x[1]);
    box_(2, 1) = std::max(box_(2, 1), x[2]);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Core::Geo::MeshFree::BoundingBox::within(BoundingBox const& b) const
{
  FOUR_C_THROW("Check before use.");
  if (empty_) return true;
  return (in_between(box_min(0), box_max(0), b.box_min(0), b.box_max(0)) and
          in_between(box_min(1), box_max(1), b.box_min(1), b.box_max(1)) and
          in_between(box_min(2), box_max(2), b.box_min(2), b.box_max(2)));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Core::Geo::MeshFree::BoundingBox::within(
    const double* x, std::vector<bool>& within_in_dir) const
{
  throw_if_not_init();

  within_in_dir.resize(3);
  within_in_dir[0] = in_between(box_min(0), box_max(0), x[0], x[0]);
  within_in_dir[1] = in_between(box_min(1), box_max(1), x[1], x[1]);
  within_in_dir[2] = in_between(box_min(2), box_max(2), x[2], x[2]);

  return (within_in_dir[0] and within_in_dir[1] and within_in_dir[2]);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Core::Geo::MeshFree::BoundingBox::within(
    Core::LinAlg::Matrix<3, 1> const& x, std::vector<bool>& within_in_dir) const
{
  throw_if_not_init();

  within_in_dir.resize(3);
  within_in_dir[0] = in_between(box_min(0), box_max(0), x(0), x(0));
  within_in_dir[1] = in_between(box_min(1), box_max(1), x(1), x(1));
  within_in_dir[2] = in_between(box_min(2), box_max(2), x(2), x(2));

  return (within_in_dir[0] and within_in_dir[1] and within_in_dir[2]);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Core::Geo::MeshFree::BoundingBox::within(Core::LinAlg::Matrix<3, 2> const& box,
    Core::LinAlg::Matrix<3, 1> const& x, std::vector<bool>& within_in_dir) const
{
  throw_if_not_init();

  within_in_dir.resize(3);
  within_in_dir[0] = in_between(box_min(box, 0), box_max(box, 0), x(0), x(0));
  within_in_dir[1] = in_between(box_min(box, 1), box_max(box, 1), x(1), x(1));
  within_in_dir[2] = in_between(box_min(box, 2), box_max(box, 2), x(2), x(2));

  return (within_in_dir[0] and within_in_dir[1] and within_in_dir[2]);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Core::Geo::MeshFree::BoundingBox::within(const Core::LinAlg::SerialDenseMatrix& xyz) const
{
  FOUR_C_THROW("Check before use.");
  BoundingBox bb;
  int numnode = xyz.numCols();
  for (int i = 0; i < numnode; ++i)
  {
    bb.add_point(&xyz(0, i));
  }
  return within(bb);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::print()
{
  if (empty_)
  {
    std::cout << "  BB: {}\n";
  }
  else
  {
    std::cout << "  BB: {(" << box_(0, 0) << "," << box_(1, 0) << "," << box_(2, 0) << ")-("
              << box_(0, 1) << "," << box_(1, 1) << "," << box_(2, 1) << ")}\n";
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::apply_dirichlet(
    const double timen, const Core::Utils::FunctionManager& function_manager)
{
  throw_if_not_init_or_setup();

  Teuchos::ParameterList p;
  p.set("total time", timen);
  p.set<const Core::Utils::FunctionManager*>("function_manager", &function_manager);

  // disn_ then also holds prescribed new Dirichlet displacements
  boxdiscret_->clear_state();
  boxdiscret_->evaluate_dirichlet(p, disn_row_, nullptr, nullptr, nullptr, nullptr);
  boxdiscret_->clear_state();

  // export to col format
  Core::LinAlg::export_to(*disn_row_, *disn_col_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::init_runtime_output(
    const Teuchos::ParameterList& io_params, const Core::IO::OutputControl& output_control)
{
  // TODO This does not work for restarted simulations as the time is obviously wrong. However, this
  // is called before the restart is read and someone with knowledge on the module has to refactor
  // the code. The only implication is that in restarted simulations the .pvd file does not contain
  // the steps of the simulation that is restarted from
  const double restart_time(0.0);

  visualization_output_writer_ptr_ =
      std::make_shared<Core::IO::DiscretizationVisualizationWriterMesh>(
          boxdiscret_, Core::IO::visualization_parameters_factory(
                           io_params.sublist("RUNTIME VTK OUTPUT"), output_control, restart_time));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::runtime_output_step_state(double timen, int stepn) const
{
  throw_if_not_init_or_setup();

  if (visualization_output_writer_ptr_ == nullptr) return;

  // reset the writer object
  visualization_output_writer_ptr_->reset();
  std::vector<std::optional<std::string>> context(3, "displacement");
  visualization_output_writer_ptr_->append_result_data_vector_with_context(
      *disn_col_, Core::IO::OutputEntity::dof, context);

  // finalize everything and write all required VTU files to filesystem
  visualization_output_writer_ptr_->write_to_disk(timen, stepn);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 1> Core::Geo::MeshFree::BoundingBox::reference_pos_of_corner_point(
    int i) const
{
  // dof gids of node i (note: each proc just has one element and eight nodes,
  // therefore local numbering from 0 to 7 on each proc)
  Core::Nodes::Node* node_i = boxdiscret_->l_col_node(i);

  Core::LinAlg::Matrix<3, 1> x(Core::LinAlg::Initialization::zero);
  for (int dim = 0; dim < 3; ++dim) x(dim) = node_i->x()[dim];

  return x;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 1> Core::Geo::MeshFree::BoundingBox::current_position_of_corner_point(
    int i) const
{
  // dof gids of node i (note: each proc just has one element and eight nodes,
  // therefore local numbering from 0 to 7 on each proc)
  Core::LinAlg::Matrix<3, 1> x(Core::LinAlg::Initialization::zero);
  if (boxdiscret_ != nullptr)
  {
    Core::Nodes::Node* node_i = boxdiscret_->l_col_node(i);
    std::vector<int> dofnode = boxdiscret_->dof(node_i);

    for (int dim = 0; dim < 3; ++dim)
      x(dim) = node_i->x()[dim] +
               disn_col_->local_values_as_span()[disn_col_->get_map().lid(dofnode[dim])];
  }
  else
  {
    x = undeformed_box_corner_point_position(i);
  }

  return x;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::undeformed_box_corner_point_position(
    int i, std::vector<double>& x) const
{
  // to get numbering according to 4C convention of hex eles
  if (i == 2 or i == 6)
    ++i;
  else if (i == 3 or i == 7)
    --i;

  x[0] = ((i & 1) == 1) ? box_max(0) : box_min(0);
  x[1] = ((i & 2) == 2) ? box_max(1) : box_min(1);
  x[2] = ((i & 4) == 4) ? box_max(2) : box_min(2);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 1> Core::Geo::MeshFree::BoundingBox::undeformed_box_corner_point_position(
    int i) const
{
  // to get numbering according to 4C convention of hex eles
  if (i == 2 or i == 6)
    ++i;
  else if (i == 3 or i == 7)
    --i;

  Core::LinAlg::Matrix<3, 1> x(Core::LinAlg::Initialization::zero);
  x(0) = ((i & 1) == 1) ? box_max(0) : box_min(0);
  x(1) = ((i & 2) == 2) ? box_max(1) : box_min(1);
  x(2) = ((i & 4) == 4) ? box_max(2) : box_min(2);

  return x;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::transform_from_undeformed_bounding_box_system_to_global(
    Core::LinAlg::Matrix<3, 1> const& xi, Core::LinAlg::Matrix<3, 1>& x) const
{
  throw_if_not_init();

  // nothing to do in case periodic bounding is not deforming
  if (not havedirichletbc_)
  {
    x = xi;
    return;
  }

  // reset globcoord variable
  x.clear();

  // Evaluate lagrangian shape functions at xi
  Core::LinAlg::Matrix<8, 1> funct;
  lagrange_polynomial_to_map_from_undeformed_bounding_box_system_to_global(
      funct, xi(0), xi(1), xi(2));

  Core::LinAlg::Matrix<3, 8> coord;
  for (unsigned int i = 0; i < 8; ++i)
  {
    for (unsigned int j = 0; j < 3; ++j)
    {
      // use shape function values for interpolation
      coord(j, i) = current_position_of_corner_point(i)(j);
      x(j) += funct(i) * coord(j, i);
    }
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::transform_from_undeformed_bounding_box_system_to_global(
    double const* xi, double* x) const
{
  throw_if_not_init_or_setup();

  Core::Nodes::Node** mynodes = boxdiscret_->l_col_element(0)->nodes();
  if (!mynodes) FOUR_C_THROW("ERROR: local_to_global: Null pointer!");

  // reset globcoord variable
  for (unsigned int dim = 0; dim < 3; ++dim) x[dim] = 0.0;

  // Evaluate lagrangian shape functions at xi
  Core::LinAlg::Matrix<8, 1> funct;
  lagrange_polynomial_to_map_from_undeformed_bounding_box_system_to_global(
      funct, xi[0], xi[1], xi[2]);

  Core::LinAlg::Matrix<3, 8> coord;
  for (unsigned int i = 0; i < 8; ++i)
  {
    for (unsigned int j = 0; j < 3; ++j)
    {
      // use shape function values for interpolation
      coord(j, i) = current_position_of_corner_point(i)(j);
      x[j] += funct(i) * coord(j, i);
    }
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
bool Core::Geo::MeshFree::BoundingBox::transform_from_global_to_undeformed_bounding_box_system(
    Core::LinAlg::Matrix<3, 1> const& x, Core::LinAlg::Matrix<3, 1>& xi) const
{
  throw_if_not_init();

  // nothing to do in case periodic bounding is not deforming
  if (not havedirichletbc_)
  {
    xi = x;
    return true;
  }

  // initialize variables
  int const numnode = 8;
  int const ndim = 3;
  double tol = Core::Geo::TOL12;
  bool converged = false;
  Core::LinAlg::Matrix<numnode, 1> funct;
  Core::LinAlg::Matrix<ndim, numnode> deriv;
  Core::LinAlg::Matrix<ndim, numnode> pbbcurrnodepos;
  Core::LinAlg::Matrix<ndim, ndim> xjm;
  Core::LinAlg::Matrix<ndim, 1> rhs;

  // spatial configuration of this element!
  for (int k = 0; k < numnode; ++k)
    for (int j = 0; j < ndim; ++j) pbbcurrnodepos(j, k) = current_position_of_corner_point(k)(j);

  // first estimation for xi
  for (int dim = 0; dim < ndim; ++dim) xi(dim) = x(dim);

  // solve linearized system R(xi) = x(xi) - x = N(xi) x^ - x != 0
  int numiter = 10;
  int j = 0;
  while (not converged and j < numiter)
  {
    // reset matrices
    xjm.clear();
    deriv.clear();
    rhs.clear();

    // evaluate lagrangian shape functions at xi
    lagrange_polynomial_to_map_from_undeformed_bounding_box_system_to_global(
        funct, xi(0), xi(1), xi(2));
    lagrange_polynomial_to_map_from_undeformed_bounding_box_system_to_global_deriv1(
        deriv, xi(0), xi(1), xi(2));

    // compute dN/dxi x^
    for (int k = 0; k < numnode; ++k)
      for (int p = 0; p < ndim; ++p)
        for (int l = 0; l < ndim; ++l) xjm(p, l) += deriv(l, k) * pbbcurrnodepos(p, k);

    // rhs of (linearized equation) I
    for (int p = 0; p < ndim; ++p) rhs(p) = -x(p);

    // rhs of (linearized equation) I
    for (int k = 0; k < numnode; ++k)
      for (int p = 0; p < ndim; ++p) rhs(p) += funct(k) * pbbcurrnodepos(p, k);

    // compute norm
    double norm = 0.0;
    for (int p = 0; p < ndim; ++p) norm += rhs(p) * rhs(p);

    // check if we are converged
    if (sqrt(norm) < tol)
    {
      converged = true;
      break;
    }

    // safety check
    if (abs(xjm.determinant()) < 1e-15)
    {
      std::cout << "WARNING !!! jacobi determinant singular! In DeformedToUndeformed(...)"
                << std::endl;
      std::cout << "JAC= " << xjm.determinant() << std::endl;
      std::cout << "CONVERGED= " << converged << std::endl;
      FOUR_C_THROW("*** WARNING: jacobi singular ***");
      converged = false;
      break;
    }

    // solve equation
    double xjm_invert = xjm.invert();
    if (abs(xjm_invert) < 1e-15) FOUR_C_THROW("ERROR: Singular Jacobian");

    // compute increment
    Core::LinAlg::Matrix<ndim, 1> deltaxi(Core::LinAlg::Initialization::zero);
    for (int z = 0; z < ndim; ++z)
      for (int p = 0; p < ndim; ++p) deltaxi(z) -= xjm(z, p) * rhs(p);

    // incremental update
    for (int p = 0; p < ndim; ++p) xi(p) += deltaxi(p);

    ++j;
  }
  return converged;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
bool Core::Geo::MeshFree::BoundingBox::transform_from_global_to_undeformed_bounding_box_system(
    double const* x, double* xi) const
{
  throw_if_not_init();
  static Core::LinAlg::Matrix<3, 1> x_m(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1> xi_m(Core::LinAlg::Initialization::zero);
  for (int dim = 0; dim < 3; ++dim) x_m(dim) = x[dim];

  bool converged = transform_from_global_to_undeformed_bounding_box_system(x_m, xi_m);

  for (int dim = 0; dim < 3; ++dim) xi[dim] = xi_m(dim);

  x_m.clear();
  xi_m.clear();
  return converged;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::
    lagrange_polynomial_to_map_from_undeformed_bounding_box_system_to_global(
        Core::LinAlg::Matrix<8, 1>& funct,  ///< to be filled with shape function values
        double r, double s, double t) const
{
  throw_if_not_init();
  // safety check
#ifdef FOUR_C_ENABLE_ASSERTIONS
  for (int dim = 0; dim < 3; ++dim)
    if (abs(edgelength_[dim]) < 1e-12)
      FOUR_C_THROW(" you are about to divide by zero, edgelength not correctly initialized.");
#endif

  const double Q = 1.0 / (edgelength_[0] * edgelength_[1] * edgelength_[2]);


  const double rp = r - box_min(0);
  const double rm = box_max(0) - r;
  const double sp = s - box_min(1);
  const double sm = box_max(1) - s;
  const double tp = t - box_min(2);
  const double tm = box_max(2) - t;

  funct(0) = Q * rm * sm * tm;
  funct(1) = Q * rp * sm * tm;
  funct(2) = Q * rp * sp * tm;
  funct(3) = Q * rm * sp * tm;
  funct(4) = Q * rm * sm * tp;
  funct(5) = Q * rp * sm * tp;
  funct(6) = Q * rp * sp * tp;
  funct(7) = Q * rm * sp * tp;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void Core::Geo::MeshFree::BoundingBox::
    lagrange_polynomial_to_map_from_undeformed_bounding_box_system_to_global_deriv1(
        Core::LinAlg::Matrix<3, 8>& deriv1,  ///< to be filled with shape function derivative values
        double r, double s, double t) const
{
  throw_if_not_init();
  // safety check
#ifdef FOUR_C_ENABLE_ASSERTIONS
  for (int dim = 0; dim < 3; ++dim)
    if (abs(edgelength_[dim]) < 1e-12)
      FOUR_C_THROW(" you are about to divide by zero, edgelength not correctly initialized.");
#endif

  const double Q = 1.0 / (edgelength_[0] * edgelength_[1] * edgelength_[2]);

  const double rp = r - box_min(0);
  const double rm = box_max(0) - r;
  const double sp = s - box_min(1);
  const double sm = box_max(1) - s;
  const double tp = t - box_min(2);
  const double tm = box_max(2) - t;

  deriv1(0, 0) = -Q * sm * tm;
  deriv1(1, 0) = -Q * tm * rm;
  deriv1(2, 0) = -Q * rm * sm;

  deriv1(0, 1) = Q * sm * tm;
  deriv1(1, 1) = -Q * tm * rp;
  deriv1(2, 1) = -Q * rp * sm;

  deriv1(0, 2) = Q * sp * tm;
  deriv1(1, 2) = Q * tm * rp;
  deriv1(2, 2) = -Q * rp * sp;

  deriv1(0, 3) = -Q * sp * tm;
  deriv1(1, 3) = Q * tm * rm;
  deriv1(2, 3) = -Q * rm * sp;

  deriv1(0, 4) = -Q * sm * tp;
  deriv1(1, 4) = -Q * tp * rm;
  deriv1(2, 4) = Q * rm * sm;

  deriv1(0, 5) = Q * sm * tp;
  deriv1(1, 5) = -Q * tp * rp;
  deriv1(2, 5) = Q * rp * sm;

  deriv1(0, 6) = Q * sp * tp;
  deriv1(1, 6) = Q * tp * rp;
  deriv1(2, 6) = Q * rp * sp;

  deriv1(0, 7) = -Q * sp * tp;
  deriv1(1, 7) = Q * tp * rm;
  deriv1(2, 7) = Q * rm * sp;
}

FOUR_C_NAMESPACE_CLOSE

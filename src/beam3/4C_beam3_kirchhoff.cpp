// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beam3_kirchhoff.hpp"

#include "4C_beam3_base.templates.hpp"
#include "4C_beam3_triad_interpolation_local_rotation_vectors.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_geometry_periodic_boundingbox.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_fad.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
Discret::Elements::Beam3kType Discret::Elements::Beam3kType::instance_;

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
Discret::Elements::Beam3kType& Discret::Elements::Beam3kType::instance() { return instance_; }

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::Beam3kType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Beam3k* object = new Discret::Elements::Beam3k(-1, -1);
  object->unpack(buffer);
  return object;
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Beam3kType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "BEAM3K")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Beam3k>(id, owner);
    return ele;
  }
  return nullptr;
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Beam3kType::create(
    const int id, const int owner)
{
  return std::make_shared<Beam3k>(id, owner);
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3kType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  FOUR_C_THROW("method 'nodal_block_information' not implemented for element type beam3k!");
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
Core::LinAlg::SerialDenseMatrix Discret::Elements::Beam3kType::compute_null_space(
    Core::Nodes::Node& node, std::span<const double> x0, const int numdof)
{
  Core::LinAlg::SerialDenseMatrix nullspace;
  FOUR_C_THROW("method ComputeNullSpace not implemented for element type beam3k!");
  return nullspace;
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3kType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["BEAM3K"];

  using namespace Core::IO::InputSpecBuilders;

  const std::vector<std::pair<Core::FE::CellType, int>> cell_types_with_triad_size = {
      {Core::FE::CellType::line2, 6}, {Core::FE::CellType::line3, 9},
      {Core::FE::CellType::line4, 12}};

  for (const auto& [cell_type, triad_size] : cell_types_with_triad_size)
  {
    defs[cell_type] = all_of({
        parameter<Beam3KirchhoffConstraintType>("CONSTRAINT",
            {.description = "Which type of Kirchhoff constraint enforcement should be used."}),
        parameter<Beam3KirchhoffParametrizationType>("PARAMETRIZATION",
            {.description = "Which type of beam parametrization should be used."}),
        parameter<int>("MAT"),
        parameter<std::vector<double>>(
            "TRIADS", {.description = "Nodal triad rotation vectors in reference configuration",
                          .size = triad_size}),
        parameter<bool>(
            "USE_FAD", {.description = "Flag, if automatic differentiation should be used.",
                           .default_value = false}),
    });
  }
}

/*----------------------------------------------------------------------*
 |  Initialize (public)                                      meier 01/16|
 *----------------------------------------------------------------------*/
int Discret::Elements::Beam3kType::initialize(Core::FE::Discretization& dis)
{
  // setting up geometric variables for Beam3k elements
  for (int num = 0; num < dis.num_my_col_elements(); ++num)
  {
    // in case that current element is not a Beam3k element there is nothing to do and we go back
    // to the head of the loop
    if (dis.l_col_element(num)->element_type() != *this) continue;

    // if we get so far current element is a Beam3k element and  we get a pointer at it
    Discret::Elements::Beam3k* currele =
        dynamic_cast<Discret::Elements::Beam3k*>(dis.l_col_element(num));
    if (!currele) FOUR_C_THROW("cast to Beam3k* failed");

    // reference node position
    std::vector<Core::LinAlg::Matrix<3, 1>> xrefe;

    const int nnode = currele->num_node();

    // resize xrefe for the number of nodes to store
    xrefe.resize(nnode);

    // the next section is needed in case of periodic boundary conditions and a shifted
    // configuration (i.e. elements cut by the periodic boundary) in the input file
    Core::Geo::MeshFree::BoundingBox periodic_boundingbox;
    periodic_boundingbox.init(
        Global::Problem::instance()->binning_strategy_params());  // no setup() call needed here

    std::vector<double> disp_shift;
    int numdof = currele->num_dof_per_node(*(currele->nodes()[0]));
    disp_shift.resize(numdof * nnode);
    for (unsigned int i = 0; i < disp_shift.size(); ++i) disp_shift[i] = 0.0;
    if (periodic_boundingbox.have_pbc())
      currele->un_shift_node_position(disp_shift, periodic_boundingbox);

    // getting element's nodal coordinates and treating them as reference configuration
    if (currele->nodes()[0] == nullptr || currele->nodes()[1] == nullptr)
      FOUR_C_THROW("Cannot get nodes in order to compute reference configuration'");
    else
    {
      for (int node = 0; node < nnode; ++node)  // element has k nodes
        for (int dof = 0; dof < 3; ++dof)       // element node has three coordinates x1, x2 and x3
        {
          xrefe[node](dof) = currele->nodes()[node]->x()[dof] + disp_shift[node * numdof + dof];
        }
    }

    // Set up all geometrical (triads, curvatures, jacobians etc.) quantities describing the
    // (initial) reference geometry
    currele->set_up_reference_geometry(xrefe);
  }
  return 0;
}



/*----------------------------------------------------------------------*
 |  ctor (public)                                            meier 05/12|
 *----------------------------------------------------------------------*/
Discret::Elements::Beam3k::Beam3k(int id, int owner)
    : Discret::Elements::Beam3Base(id, owner),
      use_fad_(false),
      isinit_(false),
      t_(0),
      theta0_(0),
      qrefconv_(0),
      qrefnew_(0),
      k0_(0),
      length_(0.0),
      jacobi_(0.0),
      jacobi2_(0.0),
      jacobi_cp_(0.0),
      jacobi2_cp_(0.0),
      rotvec_(false),
      weakkirchhoff_(false),
      eint_(0.0),
      ekin_(0.0),
      qconvmass_(0),
      qnewmass_(0),
      wconvmass_(0),
      wnewmass_(0),
      aconvmass_(0),
      anewmass_(0),
      amodconvmass_(0),
      amodnewmass_(0),
      rttconvmass_(0),
      rttnewmass_(0),
      rttmodconvmass_(0),
      rttmodnewmass_(0),
      rtconvmass_(0),
      rtnewmass_(0),
      rconvmass_(0),
      rnewmass_(0),
      axial_strain_gp_(0),
      twist_gp_(0),
      curvature_2_gp_(0),
      curvature_3_gp_(0),
      axial_force_gp_(0),
      torque_gp_(0),
      bending_moment_2_gp_(0),
      bending_moment_3_gp_(0)
{
  return;
}
/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       meier 05/12|
 *----------------------------------------------------------------------*/
Discret::Elements::Beam3k::Beam3k(const Discret::Elements::Beam3k& old)
    : Discret::Elements::Beam3Base(old),
      use_fad_(old.use_fad_),
      isinit_(old.isinit_),
      t_(old.t_),
      theta0_(old.theta0_),
      qrefconv_(old.qrefconv_),
      qrefnew_(old.qrefnew_),
      k0_(old.k0_),
      length_(old.length_),
      jacobi_(old.jacobi_),
      jacobi2_(old.jacobi2_),
      jacobi_cp_(old.jacobi_cp_),
      jacobi2_cp_(old.jacobi_cp_),
      rotvec_(old.rotvec_),
      weakkirchhoff_(old.weakkirchhoff_),
      eint_(old.eint_),
      ekin_(old.ekin_),
      qconvmass_(old.qconvmass_),
      qnewmass_(old.qnewmass_),
      wconvmass_(old.wconvmass_),
      wnewmass_(old.wnewmass_),
      aconvmass_(old.aconvmass_),
      anewmass_(old.anewmass_),
      amodconvmass_(old.amodconvmass_),
      amodnewmass_(old.amodnewmass_),
      rttconvmass_(old.rttconvmass_),
      rttnewmass_(old.rttnewmass_),
      rttmodconvmass_(old.rttmodconvmass_),
      rttmodnewmass_(old.rttmodnewmass_),
      rtconvmass_(old.rtconvmass_),
      rtnewmass_(old.rtnewmass_),
      rconvmass_(old.rconvmass_),
      rnewmass_(old.rnewmass_),
      axial_strain_gp_(old.axial_strain_gp_),
      twist_gp_(old.twist_gp_),
      curvature_2_gp_(old.curvature_2_gp_),
      curvature_3_gp_(old.curvature_3_gp_),
      axial_force_gp_(old.axial_force_gp_),
      torque_gp_(old.torque_gp_),
      bending_moment_2_gp_(old.bending_moment_2_gp_),
      bending_moment_3_gp_(old.bending_moment_3_gp_)
{
  return;
}
/*--------------------------------------------------------------------------------*
 |  Deep copy this instance of Beam3k and return pointer to it (public) |
 |                                                                    meier 05/12 |
 *--------------------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Beam3k::clone() const
{
  Discret::Elements::Beam3k* newelement = new Discret::Elements::Beam3k(*this);
  return newelement;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                              meier 05/12
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3k::print(std::ostream& os) const { return; }

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                          meier 05/12 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Beam3k::shape() const
{
  int numnodes = num_node();
  switch (numnodes)
  {
    case 2:
      return Core::FE::CellType::line2;
      break;
    case 3:
      return Core::FE::CellType::line3;
      break;
    case 4:
      return Core::FE::CellType::line4;
      break;
    default:
      FOUR_C_THROW("Only Line2, Line3 and Line4 elements are implemented.");
      break;
  }
}


/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           meier 05/12/
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3k::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Beam3Base::pack(data);

  // add all class variables
  add_to_pack(data, use_fad_);
  add_to_pack(data, isinit_);
  add_to_pack(data, Tref_);
  add_to_pack(data, t_);
  add_to_pack(data, theta0_);
  add_to_pack(data, qrefconv_);
  add_to_pack(data, qrefnew_);
  add_to_pack(data, k0_);
  add_to_pack(data, length_);
  add_to_pack(data, jacobi_);
  add_to_pack(data, jacobi2_);
  add_to_pack(data, jacobi_cp_);
  add_to_pack(data, jacobi2_cp_);
  add_to_pack(data, rotvec_);
  add_to_pack(data, weakkirchhoff_);
  add_to_pack(data, eint_);
  add_to_pack(data, ekin_);
  add_to_pack(data, qconvmass_);
  add_to_pack(data, qnewmass_);
  add_to_pack(data, wconvmass_);
  add_to_pack(data, wnewmass_);
  add_to_pack(data, aconvmass_);
  add_to_pack(data, anewmass_);
  add_to_pack(data, amodconvmass_);
  add_to_pack(data, amodnewmass_);
  add_to_pack(data, rttconvmass_);
  add_to_pack(data, rttnewmass_);
  add_to_pack(data, rttmodconvmass_);
  add_to_pack(data, rttmodnewmass_);
  add_to_pack(data, rtconvmass_);
  add_to_pack(data, rtnewmass_);
  add_to_pack(data, rconvmass_);
  add_to_pack(data, rnewmass_);
  add_to_pack(data, axial_strain_gp_);
  add_to_pack(data, twist_gp_);
  add_to_pack(data, curvature_2_gp_);
  add_to_pack(data, curvature_3_gp_);
  add_to_pack(data, axial_force_gp_);
  add_to_pack(data, torque_gp_);
  add_to_pack(data, bending_moment_2_gp_);
  add_to_pack(data, bending_moment_3_gp_);

  return;
}
/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           meier 05/12|
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3k::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Beam3Base::unpack(buffer);

  // extract all class variables of beam3k element
  extract_from_pack(buffer, use_fad_);
  extract_from_pack(buffer, isinit_);
  extract_from_pack(buffer, Tref_);
  extract_from_pack(buffer, t_);
  extract_from_pack(buffer, theta0_);
  extract_from_pack(buffer, qrefconv_);
  extract_from_pack(buffer, qrefnew_);
  extract_from_pack(buffer, k0_);
  extract_from_pack(buffer, length_);
  extract_from_pack(buffer, jacobi_);
  extract_from_pack(buffer, jacobi2_);
  extract_from_pack(buffer, jacobi_cp_);
  extract_from_pack(buffer, jacobi2_cp_);
  extract_from_pack(buffer, rotvec_);
  extract_from_pack(buffer, weakkirchhoff_);
  extract_from_pack(buffer, eint_);
  extract_from_pack(buffer, ekin_);
  extract_from_pack(buffer, qconvmass_);
  extract_from_pack(buffer, qnewmass_);
  extract_from_pack(buffer, wconvmass_);
  extract_from_pack(buffer, wnewmass_);
  extract_from_pack(buffer, aconvmass_);
  extract_from_pack(buffer, anewmass_);
  extract_from_pack(buffer, amodconvmass_);
  extract_from_pack(buffer, amodnewmass_);
  extract_from_pack(buffer, rttconvmass_);
  extract_from_pack(buffer, rttnewmass_);
  extract_from_pack(buffer, rttmodconvmass_);
  extract_from_pack(buffer, rttmodnewmass_);
  extract_from_pack(buffer, rtconvmass_);
  extract_from_pack(buffer, rtnewmass_);
  extract_from_pack(buffer, rconvmass_);
  extract_from_pack(buffer, rnewmass_);
  extract_from_pack(buffer, axial_strain_gp_);
  extract_from_pack(buffer, twist_gp_);
  extract_from_pack(buffer, curvature_2_gp_);
  extract_from_pack(buffer, curvature_3_gp_);
  extract_from_pack(buffer, axial_force_gp_);
  extract_from_pack(buffer, torque_gp_);
  extract_from_pack(buffer, bending_moment_2_gp_);
  extract_from_pack(buffer, bending_moment_3_gp_);


  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                          meier 05/12|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Beam3k::lines()
{
  return {Core::Utils::shared_ptr_from_ref(*this)};
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3k::set_up_initial_rotations(const std::vector<double>& nodal_thetas)
{
  theta0_.resize(BEAM3K_COLLOCATION_POINTS);
  for (int i = 0; i < BEAM3K_COLLOCATION_POINTS; i++)
  {
    for (int j = 0; j < 3; j++) (theta0_[i])(j) = nodal_thetas[3 * i + j];

    // Shift angles by 2PI in case these angles are not in the interval [-PI,PI].
    if (theta0_[i].norm2() > std::numbers::pi)
    {
      Core::LinAlg::Matrix<4, 1> Q(Core::LinAlg::Initialization::zero);
      Core::LargeRotations::angletoquaternion(theta0_[i], Q);
      Core::LargeRotations::quaterniontoangle(Q, theta0_[i]);
    }
  }
}

/*----------------------------------------------------------------------*
 | sets up geometric data from current nodal position as reference
 | position; this method can be used by the register class or when ever
 | a new beam element is generated for which some reference configuration
 | has to be stored; prerequisite for applying this method is that the
 | element nodes are already known (public)                   meier 01/16|
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3k::set_up_reference_geometry(
    const std::vector<Core::LinAlg::Matrix<3, 1>>& xrefe, const bool secondinit)
{
  if (weakkirchhoff_)
    set_up_reference_geometry_wk(xrefe, secondinit);
  else
    set_up_reference_geometry_sk(xrefe, secondinit);
}

/*--------------------------------------------------------------------------------------------*
 |  Set up the reference geometry of the case of a weak Kirchhoff constraint     meier 01/16|
 *--------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3k::set_up_reference_geometry_wk(
    const std::vector<Core::LinAlg::Matrix<3, 1>>& xrefe, const bool secondinit)
{
  /*this method initializes geometric variables of the element; the initialization can usually be
   *applied to elements only once; therefore after the first initialization the flag isinit is set
   *to true and from then on this method does not take any action when called again unless it is
   *called on purpose with the additional parameter secondinit. If this parameter is passed into the
   *method and is true the element is initialized another time with respective xrefe and rotrefe;
   *note: the isinit_ flag is important for avoiding reinitialization upon restart. However, it
   *should be possible to conduct a second initialization in principle (e.g. for periodic boundary
   *conditions*/

  // TODO: It turned out that the approximation of initially curved centerlines via third order
  // Hermite polynomials by simply setting the nodal positions and tangents at the boundary nodes to
  // the corresponding values of the analytical geometry at these nodes yields a worse approximation
  // than that following from a third order Lagrange polynomial interpolation based on nodal
  // positions that coincide with the analytical values at the four element nodes. This worse
  // approximation appears in terms of a larger error in the jacobian, the derivative of the
  // relative angle theta_s and finally of the initial curvature K0_. For strongly curved initial
  // configurations where the error in the initial geometry representation might dominate the
  // discretization error it can be useful to consider strategies which yield in a better Hermite
  // approximation of the initial geometry e.g. by determining the initial nodal values (or the
  // shape function parameter c=c_{opt} instead of c=l) based on optimization strategies.

  if (!isinit_ || secondinit)
  {
    const int nnode = 2;  // number of nodes

    // Calculate the (initial reference triads) = (initial material triads) at the CPs out of the
    // angles theta0_. So far the initial value for the relative angle is set to zero, i.e. material
    // coordinate system and reference system in the reference configuration coincidence.
    std::vector<Core::LinAlg::Matrix<3, 3>> Gref(BEAM3K_COLLOCATION_POINTS);
    for (int node = 0; node < BEAM3K_COLLOCATION_POINTS; node++)
    {
      Gref[node].clear();
      Core::LargeRotations::angletotriad(theta0_[node], Gref[node]);
    }

    Tref_.resize(2);
    t_.resize(2);
    // write initial nodal tangents in extra vector
    for (int i = 0; i < 3; i++)
    {
      (Tref_[0])(i) = (Gref[0])(i, 0);
      (Tref_[1])(i) = (Gref[1])(i, 0);
      (t_[0])(i) = (Gref[0])(i, 0);
      (t_[1])(i) = (Gref[1])(i, 0);
    }

    // Get integration points for exact integration
    Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);

    // Vector holding angle theta of triads
    std::vector<Core::LinAlg::Matrix<3, 1>> theta_cp;
    theta_cp.resize(BEAM3K_COLLOCATION_POINTS);
    Core::LinAlg::Matrix<3, 1> theta(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> theta_s(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 3> triad_mat(
        Core::LinAlg::Initialization::zero);  // material triad at gp

    // Resize vectors for storage of time integration quantities
    resize_class_variables(gausspoints.nquad);

    // assure correct size of strain and stress resultant class variables and fill them
    // with zeros (by definition, the reference configuration is undeformed and stress-free)
    axial_strain_gp_.resize(gausspoints.nquad);
    std::fill(axial_strain_gp_.begin(), axial_strain_gp_.end(), 0.0);
    twist_gp_.resize(gausspoints.nquad);
    std::fill(twist_gp_.begin(), twist_gp_.end(), 0.0);
    curvature_2_gp_.resize(gausspoints.nquad);
    std::fill(curvature_2_gp_.begin(), curvature_2_gp_.end(), 0.0);
    curvature_3_gp_.resize(gausspoints.nquad);
    std::fill(curvature_3_gp_.begin(), curvature_3_gp_.end(), 0.0);

    axial_force_gp_.resize(gausspoints.nquad);
    std::fill(axial_force_gp_.begin(), axial_force_gp_.end(), 0.0);
    torque_gp_.resize(gausspoints.nquad);
    std::fill(torque_gp_.begin(), torque_gp_.end(), 0.0);
    bending_moment_2_gp_.resize(gausspoints.nquad);
    std::fill(bending_moment_2_gp_.begin(), bending_moment_2_gp_.end(), 0.0);
    bending_moment_3_gp_.resize(gausspoints.nquad);
    std::fill(bending_moment_3_gp_.begin(), bending_moment_3_gp_.end(), 0.0);

    // calculate the length of the element via Newton iteration
    Core::LinAlg::Matrix<12, 1> pos_ref_centerline;
    add_ref_values_disp_centerline<2, 2>(pos_ref_centerline);
    length_ = calc_reflength<2, 2>(pos_ref_centerline);

    // Matrices to store the function values of the Lagrange shape functions used to interpolate
    // theta
    Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS> L_i;
    Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS> L_i_xi;

    // Matrices to store the (derivative of) the Hermite shape functions
    Core::LinAlg::Matrix<1, 2 * nnode> N_i;
    Core::LinAlg::Matrix<1, 2 * nnode> N_i_xi;

    // Matrices to store r and dr/dxi
    Core::LinAlg::Matrix<3, 1> r;
    Core::LinAlg::Matrix<3, 1> r_xi;

    // storage index for collocation point
    unsigned int ind = 0;

    // Calculate initial material triads at the collocation points
    for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; node++)
    {
      // colpt=0->xi=-1  colpt=1->xi=0 colpt=2->xi=1
      const double xi = (double)node / (BEAM3K_COLLOCATION_POINTS - 1) * 2 - 1.0;

      // Get values of shape functions
      N_i_xi.clear();
      Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi, length_, Core::FE::CellType::line2);

      // Determine storage position for the node colpt
      ind = Core::LargeRotations::numbering_trafo(node + 1, BEAM3K_COLLOCATION_POINTS);

      // current value of derivatives at GP (derivatives in xi!)
      r_xi.clear();

      for (int i = 0; i < 3; i++)
      {
        r_xi(i) += xrefe[0](i) * N_i_xi(0) + xrefe[1](i) * N_i_xi(2) + Tref_[0](i) * N_i_xi(1) +
                   Tref_[1](i) * N_i_xi(3);
      }

      jacobi_cp_[ind] = r_xi.norm2();

      // rotate (initial reference triad) = (initial material triad) at the interior CPs on
      // tangential line resulting from the Hermite interpolation. This is necessary for initially
      // curved geometries for which the Hermite interpolation does not deliver the exact tangent
      // values of the analytical representation of the initial curve at the CPs.
      if (ind > 1)  // only for internal CPs
      {
        Core::LinAlg::Matrix<3, 3> G_aux(Core::LinAlg::Initialization::zero);
        Core::LargeRotations::calculate_sr_triads<double>(r_xi, Gref[ind], G_aux);
        // rotate also Gref and theta0_ via smallest rotation to get a consistent initial state
        Gref[ind] = G_aux;
        Core::LargeRotations::triadtoquaternion(G_aux, qrefconv_[ind]);
        Core::LargeRotations::quaterniontoangle(qrefconv_[ind], theta0_[ind]);
      }
      else
      {
        Core::LargeRotations::triadtoquaternion(Gref[ind], qrefconv_[ind]);
      }
      qrefnew_[ind] = qrefconv_[ind];
    }

    // SETUP INTERPOLATION via calculation of difference angle
    for (int colpt = 0; colpt < BEAM3K_COLLOCATION_POINTS; colpt++)
    {
      theta_cp[colpt].clear();
      Core::LargeRotations::triadtoangleright(theta_cp[colpt], Gref[REFERENCE_NODE], Gref[colpt]);
    }

    // Loop through all GPs and computation of all relevant values at each gp
    for (int numgp = 0; numgp < gausspoints.nquad; numgp++)
    {
      // Get position xi of GP
      const double xi = gausspoints.qxg[numgp][0];

      // Get values of shape functions
      L_i.clear();
      L_i_xi.clear();
      N_i_xi.clear();
      N_i.clear();
      Core::FE::shape_function_1d(L_i, xi, shape());
      Core::FE::shape_function_1d_deriv1(L_i_xi, xi, shape());
      Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi, length_, Core::FE::CellType::line2);
      Core::FE::shape_function_hermite_1d(N_i, xi, length_, Core::FE::CellType::line2);

      // current value of derivatives at GP (derivatives in xi!)
      r.clear();
      r_xi.clear();
      theta.clear();
      theta_s.clear();

      for (int i = 0; i < 3; i++)
      {
        r(i) += xrefe[0](i) * N_i(0) + xrefe[1](i) * N_i(2) + Tref_[0](i) * N_i(1) +
                Tref_[1](i) * N_i(3);
        r_xi(i) += xrefe[0](i) * N_i_xi(0) + xrefe[1](i) * N_i_xi(2) + Tref_[0](i) * N_i_xi(1) +
                   Tref_[1](i) * N_i_xi(3);
      }

      // calculate jacobi jacobi_=|r'_0|
      jacobi_[numgp] = r_xi.norm2();

      // calculate interpolated angle
      for (int i = 0; i < BEAM3K_COLLOCATION_POINTS; i++)
      {
        for (int j = 0; j < 3; j++)
        {
          theta(j) += L_i(i) * theta_cp[i](j);
          theta_s(j) += L_i_xi(i) * theta_cp[i](j) / jacobi_[numgp];
        }
      }
      computestrain(theta, theta_s, k0_[numgp]);

      Core::LinAlg::Matrix<3, 3> temp_triad(Core::LinAlg::Initialization::zero);
      Core::LargeRotations::angletotriad(theta, temp_triad);
      triad_mat.multiply(Gref[REFERENCE_NODE], temp_triad);
      set_initial_dynamic_class_variables(numgp, triad_mat, r);
    }

    isinit_ = true;
  }
}

/*--------------------------------------------------------------------------------------------*
 |  Set up the reference geometry of the case of a strong Kirchhoff constraint     meier 01/16|
 *--------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3k::set_up_reference_geometry_sk(
    const std::vector<Core::LinAlg::Matrix<3, 1>>& xrefe, const bool secondinit)
{
  /*this method initializes geometric variables of the element; the initialization can usually be
   *applied to elements only once; therefore after the first initialization the flag isinit is set
   *to true and from then on this method does not take any action when called again unless it is
   *called on purpose with the additional parameter secondinit. If this parameter is passed into the
   *method and is true the element is initialized another time with respective xrefe and rotrefe;
   *note: the isinit_ flag is important for avoiding reinitialization upon restart. However, it
   *should be possible to conduct a second initialization in principle (e.g. for periodic boundary
   *conditions*/

  // TODO: It turned out that the approximation of initially curved centerlines via third order
  // Hermite polynomials by simply setting the nodal positions and tangents at the boundary nodes to
  // the corresponding values of the analytical geometry at these nodes yields a worse approximation
  // than that following from a third order Lagrange polynomial interpolation based on nodal
  // positions that coincide with the analytical values at the four element nodes. This worse
  // approximation appears in terms of a larger error in the jacobian, the derivative of the
  // relative angle theta_s and finally of the initial curvature K0_. For strongly curved initial
  // configurations where the error in the initial geometry representation might dominate the
  // discretization error it can be useful to consider strategies which yield in a better Hermite
  // approximation of the initial geometry e.g. by determining the initial nodal values (or the
  // shape function parameter c=c_{opt} instead of c=l) based on optimization strategies.

  if (!isinit_ || secondinit)
  {
    const int nnode = 2;  // number of nodes

    // Calculate the (initial reference triads) = (initial material triads) at the CPs out of the
    // angles theta0_. So far the initial value for the relative angle is set to zero, i.e. material
    // coordinate system and reference system in the reference configuration coincidence.
    std::vector<Core::LinAlg::Matrix<3, 3>> Gref(BEAM3K_COLLOCATION_POINTS);
    for (int node = 0; node < BEAM3K_COLLOCATION_POINTS; node++)
    {
      Gref[node].clear();
      Core::LargeRotations::angletotriad(theta0_[node], Gref[node]);
    }

    Tref_.resize(2);
    t_.resize(2);
    // write initial nodal tangents in extra vector
    for (int i = 0; i < 3; i++)
    {
      (Tref_[0])(i) = (Gref[0])(i, 0);
      (Tref_[1])(i) = (Gref[1])(i, 0);
      (t_[0])(i) = (Gref[0])(i, 0);
      (t_[1])(i) = (Gref[1])(i, 0);
    }

    // Get integration points for exact integration
    Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);

    // Vector holding angle theta of triads
    std::vector<double> phi_cp;
    phi_cp.resize(BEAM3K_COLLOCATION_POINTS);
    double phi(true);
    double phi_s(true);
    Core::LinAlg::Matrix<3, 3> triad_mat(
        Core::LinAlg::Initialization::zero);  // material triad at gp

    // Resize vectors for storage of time integration quantities
    resize_class_variables(gausspoints.nquad);

    // assure correct size of strain and stress resultant class variables and fill them
    // with zeros (by definition, the reference configuration is undeformed and stress-free)
    axial_strain_gp_.resize(gausspoints.nquad);
    std::fill(axial_strain_gp_.begin(), axial_strain_gp_.end(), 0.0);
    twist_gp_.resize(gausspoints.nquad);
    std::fill(twist_gp_.begin(), twist_gp_.end(), 0.0);
    curvature_2_gp_.resize(gausspoints.nquad);
    std::fill(curvature_2_gp_.begin(), curvature_2_gp_.end(), 0.0);
    curvature_3_gp_.resize(gausspoints.nquad);
    std::fill(curvature_3_gp_.begin(), curvature_3_gp_.end(), 0.0);

    axial_force_gp_.resize(gausspoints.nquad);
    std::fill(axial_force_gp_.begin(), axial_force_gp_.end(), 0.0);
    torque_gp_.resize(gausspoints.nquad);
    std::fill(torque_gp_.begin(), torque_gp_.end(), 0.0);
    bending_moment_2_gp_.resize(gausspoints.nquad);
    std::fill(bending_moment_2_gp_.begin(), bending_moment_2_gp_.end(), 0.0);
    bending_moment_3_gp_.resize(gausspoints.nquad);
    std::fill(bending_moment_3_gp_.begin(), bending_moment_3_gp_.end(), 0.0);

    // calculate the length of the element via Newton iteration
    Core::LinAlg::Matrix<12, 1> disp_refe_centerline;
    add_ref_values_disp_centerline<2, 2>(disp_refe_centerline);
    length_ = calc_reflength<2, 2>(disp_refe_centerline);

    // Matrices to store the function values of the Lagrange shape functions used to interpolate
    // theta
    Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS> L_i;
    Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS> L_i_xi;

    // Matrices to store the (derivative of) the Hermite shape functions
    Core::LinAlg::Matrix<1, 2 * nnode> N_i;
    Core::LinAlg::Matrix<1, 2 * nnode> N_i_xi;
    Core::LinAlg::Matrix<1, 2 * nnode> N_i_xixi;

    // Matrices to store r, dr/dxi and d^2r/dxi^2
    Core::LinAlg::Matrix<3, 1> r;
    Core::LinAlg::Matrix<3, 1> r_xi;
    Core::LinAlg::Matrix<3, 1> r_xixi;
    Core::LinAlg::Matrix<3, 1> r_s;
    Core::LinAlg::Matrix<3, 1> r_ss;
    // centerline curvature
    Core::LinAlg::Matrix<3, 1> kappacl;

    // storage index for collocation point
    unsigned int ind = 0;

    // Calculate initial material triads at the collocation points
    for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; node++)
    {
      // colpt=0->xi=-1  colpt=1->xi=0 colpt=2->xi=1
      const double xi = (double)node / (BEAM3K_COLLOCATION_POINTS - 1) * 2 - 1.0;

      // Get values of shape functions
      L_i.clear();
      N_i_xi.clear();
      N_i_xixi.clear();
      Core::FE::shape_function_1d(L_i, xi, shape());
      Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi, length_, Core::FE::CellType::line2);
      Core::FE::shape_function_hermite_1d_deriv2(N_i_xixi, xi, length_, Core::FE::CellType::line2);

      // Determine storage position for the node colpt
      ind = Core::LargeRotations::numbering_trafo(node + 1, BEAM3K_COLLOCATION_POINTS);

      // current value of derivatives at GP (derivatives in xi!)
      r_xi.clear();
      r_xixi.clear();

      for (int i = 0; i < 3; i++)
      {
        r_xi(i) += xrefe[0](i) * N_i_xi(0) + xrefe[1](i) * N_i_xi(2) + Tref_[0](i) * N_i_xi(1) +
                   Tref_[1](i) * N_i_xi(3);
        r_xixi(i) += xrefe[0](i) * N_i_xixi(0) + xrefe[1](i) * N_i_xixi(2) +
                     Tref_[0](i) * N_i_xixi(1) + Tref_[1](i) * N_i_xixi(3);
      }

      // calculate jacobi_=||r'_0|| and jacobi2_=r'_0^T r''_0
      jacobi_cp_[ind] = r_xi.norm2();
      jacobi2_cp_[ind] = r_xi.dot(r_xixi);

      // rotate (initial reference triad) = (initial material triad) at the interior CPs on
      // tangential line resulting from the Hermite interpolation. This is necessary for initially
      // curved geometries for which the Hermite interpolation does not deliver the exact tangent
      // values of the analytical representation of the initial curve at the CPs.
      if (ind > 1)  // only for internal CPs
      {
        Core::LinAlg::Matrix<3, 3> G_aux(Core::LinAlg::Initialization::zero);
        Core::LargeRotations::calculate_sr_triads<double>(r_xi, Gref[ind], G_aux);
        // rotate also Gref and theta0_ via smallest rotation to get a consistent initial state
        Gref[ind] = G_aux;
        Core::LargeRotations::triadtoquaternion(G_aux, qrefconv_[ind]);
        Core::LargeRotations::quaterniontoangle(qrefconv_[ind], theta0_[ind]);
      }
      else
      {
        Core::LargeRotations::triadtoquaternion(Gref[ind], qrefconv_[ind]);
      }
      qrefnew_[ind] = qrefconv_[ind];
    }  //(int node=0;node<BEAM3K_COLLOCATION_POINTS;node++)

    // SETUP INTERPOLATION via calculation of difference angle
    for (int colpt = 0; colpt < BEAM3K_COLLOCATION_POINTS; colpt++)
    {
      Core::LinAlg::Matrix<3, 3> Lambdabarref(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<3, 1> tangentref(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<3, 1> phivec(Core::LinAlg::Initialization::zero);
      for (int i = 0; i < 3; i++)
      {
        tangentref(i) = Gref[colpt](i, 0);
      }
      Core::LargeRotations::calculate_sr_triads<double>(
          tangentref, Gref[REFERENCE_NODE], Lambdabarref);
      Core::LargeRotations::triadtoangleleft(phivec, Lambdabarref, Gref[colpt]);
      phi_cp[colpt] = 0.0;
      for (int i = 0; i < 3; i++)
      {
        phi_cp[colpt] += tangentref(i) * phivec(i);
      }
    }

    // Loop through all GPs and computation of all relevant values at each gp
    for (int numgp = 0; numgp < gausspoints.nquad; numgp++)
    {
      // Get position xi of GP
      const double xi = gausspoints.qxg[numgp][0];

      // Get values of shape functions
      L_i.clear();
      L_i_xi.clear();
      N_i.clear();
      N_i_xi.clear();
      N_i_xixi.clear();

      Core::FE::shape_function_1d(L_i, xi, shape());
      Core::FE::shape_function_1d_deriv1(L_i_xi, xi, shape());
      Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi, length_, Core::FE::CellType::line2);
      Core::FE::shape_function_hermite_1d_deriv2(N_i_xixi, xi, length_, Core::FE::CellType::line2);
      Core::FE::shape_function_hermite_1d(N_i, xi, length_, Core::FE::CellType::line2);

      // current value of derivatives at GP (derivatives in xi!)
      r.clear();
      r_xi.clear();
      r_xixi.clear();
      r_s.clear();
      r_ss.clear();
      kappacl.clear();
      phi = 0.0;
      phi_s = 0.0;

      for (int i = 0; i < 3; i++)
      {
        r(i) += xrefe[0](i) * N_i(0) + xrefe[1](i) * N_i(2) + Tref_[0](i) * N_i(1) +
                Tref_[1](i) * N_i(3);
        r_xi(i) += xrefe[0](i) * N_i_xi(0) + xrefe[1](i) * N_i_xi(2) + Tref_[0](i) * N_i_xi(1) +
                   Tref_[1](i) * N_i_xi(3);
        r_xixi(i) += xrefe[0](i) * N_i_xixi(0) + xrefe[1](i) * N_i_xixi(2) +
                     Tref_[0](i) * N_i_xixi(1) + Tref_[1](i) * N_i_xixi(3);
      }

      // calculate jacobi_=||r'_0|| and jacobi2_=r'_0^T r''_0
      jacobi_[numgp] = r_xi.norm2();
      jacobi2_[ind] = r_xi.dot(r_xixi);

      // calculate interpolated angle
      for (int i = 0; i < BEAM3K_COLLOCATION_POINTS; i++)
      {
        phi += L_i(i) * phi_cp[i];
        phi_s += L_i_xi(i) * phi_cp[i] / jacobi_[numgp];
      }

      // calculate derivatives in s
      r_s = r_xi;
      r_s.scale(1 / jacobi_[numgp]);
      for (int i = 0; i < 3; i++)
      {
        r_ss(i) = r_xixi(i) / pow(jacobi_[numgp], 2.0) -
                  r_xi(i) * jacobi2_[numgp] / pow(jacobi_[numgp], 4.0);
      }

      triad_mat.clear();
      compute_triad_sk(phi, r_s, Gref[REFERENCE_NODE], triad_mat);
      calculate_clcurvature(r_s, r_ss, kappacl);

      computestrain_sk(phi_s, kappacl, Gref[REFERENCE_NODE], triad_mat, k0_[numgp]);

      set_initial_dynamic_class_variables(numgp, triad_mat, r);
    }

    isinit_ = true;
  }  // if(!isinit_)

}  // Discret::Elements::Beam3k::set_up_reference_geometry_sk()

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
double Discret::Elements::Beam3k::get_jacobi_fac_at_xi(const double& xi) const
{
  const int nnode = 2;

  // Matrices to store the the Hermite shape function derivative values
  Core::LinAlg::Matrix<1, 2 * nnode> N_i_xi;
  Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi, length_, Core::FE::CellType::line2);

  // jacobi = ds/dxi = ||r'_0||
  Core::LinAlg::Matrix<3, 1> r_xi;

  for (unsigned int dim = 0; dim < 3; ++dim)
  {
    r_xi(dim) += nodes()[0]->x()[dim] * N_i_xi(0) + nodes()[1]->x()[dim] * N_i_xi(2) +
                 Tref_[0](dim) * N_i_xi(1) + Tref_[1](dim) * N_i_xi(3);
  }

  return r_xi.norm2();
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3k::get_pos_at_xi(
    Core::LinAlg::Matrix<3, 1>& pos, const double& xi, const std::vector<double>& disp) const
{
  /* we expect the (centerline) displacement state vector (positions and tangents at boundary nodes)
   * here; for flexibility, we also accept complete DoF vector of this element (2*7+1 DoFs) and do
   * the rest automatically here NOTE: the latter option is favorable in case of rotvec_==true: in
   * case of ROTVEC=true, we need 3 positions, 3 absolute rotvec DOFs AND the length of tangent
   * vector (7th DoF) */
  Core::LinAlg::Matrix<12, 1> disp_totlag_centerline(Core::LinAlg::Initialization::zero);

  if (disp.size() == 15)
  {
    // in this case, we need to "add" reference values first, because if rotvec_==true,
    // we can extract tangent vectors only from total rotation vectors
    Core::LinAlg::Matrix<15, 1, double> disp_totlag(disp.data());
    add_ref_values_disp<2, double>(disp_totlag);
    this->extract_centerline_dof_values_from_element_state_vector<2, 2, double>(
        disp_totlag, disp_totlag_centerline);
  }
  else if (disp.size() == 12)
  {
    /* in this case, we expect the position and tangent DOF values for both boundary nodes;
     * for rotvec_==true, the tangents are NOT nodal DOFs, so they need to be pre-calculated
     * before calling this method */
    disp_totlag_centerline = Core::LinAlg::Matrix<12, 1>(disp.data());
    add_ref_values_disp_centerline<2, 2, double>(disp_totlag_centerline);
  }
  else
  {
    FOUR_C_THROW(
        "size mismatch: expected either 12 values for disp_centerline or 15 for "
        "full element disp vector and got {}",
        disp.size());
  }

  Beam3Base::get_pos_at_xi<2, 2>(pos, xi, disp_totlag_centerline);
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3k::get_triad_at_xi(
    Core::LinAlg::Matrix<3, 3>& triad, const double& xi, const std::vector<double>& disp) const
{
  if (not weakkirchhoff_)
    FOUR_C_THROW("method GetTriadAtXi is limited to WK so far! extend to SK if needed");

  if (disp.size() != 15)
    FOUR_C_THROW(
        "size mismatch: expected 15 values for element disp vector and got {}", disp.size());


  // Dof vector in total Lagrangian style, i.e. "displacement + reference values"
  Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double> disp_totlag(
      Core::LinAlg::Initialization::zero);

  update_disp_totlag<2, double>(disp, disp_totlag);

  // material triads at collocation points
  std::vector<Core::LinAlg::Matrix<3, 3, double>> triad_mat_cp(BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<3, 3, double>(Core::LinAlg::Initialization::zero));
  Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double> dummy(
      Core::LinAlg::Initialization::zero);
  std::vector<Core::LinAlg::Matrix<4, 1>> Qref_dummy(
      BEAM3K_COLLOCATION_POINTS, Core::LinAlg::Matrix<4, 1>(Core::LinAlg::Initialization::zero));

  // Todo @grill:
  //    this method uses Qrefconv_[node] as reference triads; so we can only call this method if
  //    those were calculated correctly in a preceding time step. (be careful in post-processing!)
  update_nodal_variables<2, double>(disp_totlag, dummy, triad_mat_cp,
      Qref_dummy);  // Todo @grill split/adapt method and avoid dummy variables

  // create object of triad interpolation scheme
  LargeRotations::TriadInterpolationLocalRotationVectors<3, double> triad_interpolation_scheme_ptr;

  // reset scheme with nodal triads
  triad_interpolation_scheme_ptr.reset(triad_mat_cp);

  triad_interpolation_scheme_ptr.get_interpolated_triad_at_xi(triad, xi);
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3k::get_scaled_second_and_third_base_vector_at_xi(const double& xi,
    const std::vector<double>& disp, Core::LinAlg::Matrix<3, 2>& scaledbasevectors) const
{
  // Todo @grill: delete or update this method. kept it for now as we might need it soon
  FOUR_C_THROW(
      "Beam3k: method get_scaled_second_and_third_base_vector_at_xi is deprecated for now because "
      "it "
      "was only valid for the implicit assumption of a rectangular cross-section! If need be, "
      "generalize the definition of cross-section shape and dimensions and adapt this method. "
      "For now, the cross-section shape and dimensions are only required explicitly if beam "
      "interactions (contact, potentials, drag in background fluid) are to be evaluated. In this "
      "case, we only support and hence assume a circular cross-section with a radius which is "
      "either "
      "explicitly specified in the material definition line of the input file or per default "
      "computed "
      "from the area moment of inertia Iyy.");

  //  Core::LinAlg::Matrix<3,3> triad(true);
  //
  //  GetTriadAtXi(triad,xi,disp);
  //
  //  // ToDo careful, this is a hack for rectangular cross-sections ?!?
  //  double Ly=std::pow(pow(12.0*Izz_,3)/(12.0*Iyy_),1.0/8.0);
  //  double Lz=12.0*Izz_/pow(Ly,3);
  //
  //  for(int i=0;i<3;i++)
  //  {
  //    scaledbasevectors(i,0) = 0.5 * Ly * triad(i,1);
  //    scaledbasevectors(i,1) = 0.5 * Lz * triad(i,2);
  //  }
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3k::get_generalized_interpolation_matrix_variations_at_xi(
    Core::LinAlg::SerialDenseMatrix& Ivar, const double& xi, const std::vector<double>& disp) const
{
  if (not weakkirchhoff_) FOUR_C_THROW("method is limited to WK so far! extend to SK if needed");

  if (rotvec_)
    FOUR_C_THROW("method is limited to tangent-based parametrization! extend to rotvec if needed");

  const unsigned int ndim = 3;
  const unsigned int nnodecl = 2;
  const unsigned int vpernode = 2;
  const unsigned int numdof = ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS;

  // safety check
  if ((unsigned int)Ivar.numRows() != 2 * ndim or (unsigned int) Ivar.numCols() != numdof)
    FOUR_C_THROW("size mismatch! expected {}x{} matrix and got {}x{}", 6, numdof, Ivar.numRows(),
        Ivar.numCols());

  Ivar.putScalar(0.0);

  // *******************************************************************************
  // concerning interpolation of variation of CENTERLINE POSITION \vardelta r:
  // *******************************************************************************

  Core::LinAlg::Matrix<ndim, numdof, double> N(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, vpernode * nnodecl, double> N_i(Core::LinAlg::Initialization::zero);


  Core::FE::shape_function_hermite_1d(N_i, xi, length_, Core::FE::CellType::line2);
  assemble_shapefunctions_n(N_i, N);

  // this part is associated with the variation of the centerline position
  // (first three rows of Ivar)
  for (unsigned int irow = 0; irow < N.num_rows(); ++irow)
    for (unsigned int icol = 0; icol < N.num_cols(); ++icol) Ivar(irow, icol) += N(irow, icol);

  // *******************************************************************************
  // concerning interpolation of variation of CENTERLINE ORIENTATION \vardelta \theta:
  // *******************************************************************************

  // define and initialize variables
  // *******************************************************************************
  // position index where CP quantities have to be stored (according to numbering convention)
  unsigned int ind = 0;
  double xi_cp = 0.0;


  Core::LinAlg::Matrix<ndim, numdof, double> N_s(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, vpernode * nnodecl, double> N_i_xi(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<1, numdof, double> L(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i(
      Core::LinAlg::Initialization::zero);


  Core::LinAlg::Matrix<numdof, 1, double> disp_totlag(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdof, 1, double> disp_totlag_centerline(
      Core::LinAlg::Initialization::zero);

  std::vector<Core::LinAlg::Matrix<3, 3, double>> triad_dummy(BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<3, 3, double>(Core::LinAlg::Initialization::zero));
  std::vector<Core::LinAlg::Matrix<4, 1>> Qref_dummy(
      BEAM3K_COLLOCATION_POINTS, Core::LinAlg::Matrix<4, 1>(Core::LinAlg::Initialization::zero));


  Core::LinAlg::Matrix<3, 1, double> r_s(Core::LinAlg::Initialization::zero);  // r' vector
  double abs_r_s = 0.0;                                                        // ||r'||


  std::vector<Core::LinAlg::Matrix<numdof, ndim, double>> v_thetaperp_cp(BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<numdof, ndim, double>(Core::LinAlg::Initialization::zero));
  std::vector<Core::LinAlg::Matrix<numdof, ndim, double>> v_thetapar_cp(BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<numdof, ndim, double>(Core::LinAlg::Initialization::zero));


  // re-interpolated spin vector variation: v_theta_bar
  Core::LinAlg::Matrix<numdof, ndim, double> v_theta_bar(Core::LinAlg::Initialization::zero);


  // set nodal / cp quantities: positions, tangents, triads
  // *******************************************************************************

  // Set current positions and orientations at all nodes:
  update_disp_totlag<nnodecl, double>(disp, disp_totlag);


  update_nodal_variables<nnodecl, double>(disp_totlag, disp_totlag_centerline, triad_dummy,
      Qref_dummy);  // Todo @grill: make this nicer and avoid dummies !


  // compute quantities at collocation points
  // *******************************************************************************
  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    // Determine storage position for this cp
    ind = Core::LargeRotations::numbering_trafo(icp + 1, BEAM3K_COLLOCATION_POINTS);

    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    xi_cp = (double)icp / (double)(BEAM3K_COLLOCATION_POINTS - 1) * 2.0 - 1.0;

    // get value of interpolating function for theta (Lagrange polynomials) at xi_cp
    L_i.clear();
    Core::FE::shape_function_1d(L_i, xi_cp, shape());

    L.clear();
    assemble_shapefunctions_l(L_i, L);


    N_i_xi.clear();
    Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi_cp, length_, Core::FE::CellType::line2);

    N_s.clear();
    assemble_shapefunctions_ns(N_i_xi, jacobi_cp_[ind], N_s);


    // Calculation of r' at xi
    r_s.clear();
    r_s.multiply(N_s, disp_totlag_centerline);

    abs_r_s = Core::FADUtils::norm(r_s);


    calc_v_thetaperp<nnodecl>(v_thetaperp_cp[ind], N_s, r_s, abs_r_s);

    calc_v_thetapartheta<nnodecl>(v_thetapar_cp[ind], L, r_s, abs_r_s);
  }


  // re-interpolation of quantities at xi based on CP values
  // *******************************************************************************

  L_i.clear();
  Core::FE::shape_function_1d(L_i, xi, shape());

  v_theta_bar.clear();
  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    v_theta_bar.update(L_i(icp), v_thetaperp_cp[icp], 1.0);
    v_theta_bar.update(L_i(icp), v_thetapar_cp[icp], 1.0);
  }

  // finally assemble the generalized interpolation matrix for the variations Ivar
  // *******************************************************************************

  // this part is associated with the increment of the centerline orientation
  // (expressed as rotation vector theta)
  // (rows 4-6 of Iinc)
  // note: we need the transposed of v_theta_bar (rows <-> cols)
  for (unsigned int irow = 0; irow < v_theta_bar.num_cols(); ++irow)
    for (unsigned int icol = 0; icol < v_theta_bar.num_rows(); ++icol)
      Ivar(ndim + irow, icol) += v_theta_bar(icol, irow);
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3k::get_stiffmat_resulting_from_generalized_interpolation_matrix_at_xi(
    Core::LinAlg::SerialDenseMatrix& stiffmat, const double& xi, const std::vector<double>& disp,
    const Core::LinAlg::SerialDenseVector& force) const
{
  if (not weakkirchhoff_) FOUR_C_THROW("method is limited to WK so far! extend to SK if needed");

  if (rotvec_)
    FOUR_C_THROW("method is limited to tangent-based parametrization! extend to rotvec if needed");


  const unsigned int ndim = 3;
  const unsigned int nnodecl = 2;
  const unsigned int vpernode = 2;
  const unsigned int numdof = ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS;

  // safety check
  if ((unsigned int)stiffmat.numRows() != numdof or (unsigned int) stiffmat.numCols() != numdof)
    FOUR_C_THROW("size mismatch! expected {}x{} matrix and got {}x{}", numdof, numdof,
        stiffmat.numRows(), stiffmat.numCols());

  stiffmat.putScalar(0.0);

  // create an auxiliary fixed size matrix and set as a view to original data in stiffmat
  Core::LinAlg::Matrix<numdof, numdof, double> stiffmat_fixedsize(stiffmat, true);


  Core::LinAlg::Matrix<ndim, 1, double> moment(&(force(3)));

  Core::LinAlg::Matrix<ndim, ndim, double> S_of_moment(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::computespin<double>(S_of_moment, moment);


  // define and initialize variables
  // *******************************************************************************
  // position index where CP quantities have to be stored (according to numbering convention)
  unsigned int ind = 0;
  double xi_cp = 0.0;


  Core::LinAlg::Matrix<ndim, numdof, double> N_s(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, vpernode * nnodecl, double> N_i_xi(Core::LinAlg::Initialization::zero);


  Core::LinAlg::Matrix<1, numdof, double> L(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i(
      Core::LinAlg::Initialization::zero);


  Core::LinAlg::Matrix<numdof, 1, double> disp_totlag(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdof, 1, double> disp_totlag_centerline(
      Core::LinAlg::Initialization::zero);

  std::vector<Core::LinAlg::Matrix<3, 3, double>> triad_mat_cp(BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<3, 3, double>(Core::LinAlg::Initialization::zero));
  std::vector<Core::LinAlg::Matrix<4, 1>> Qref_dummy(
      BEAM3K_COLLOCATION_POINTS, Core::LinAlg::Matrix<4, 1>(Core::LinAlg::Initialization::zero));


  Core::LinAlg::Matrix<ndim, 1, double> r_s(Core::LinAlg::Initialization::zero);  // r' vector
  double abs_r_s = 0.0;                                                           // ||r'||

  // first base vector at CP
  Core::LinAlg::Matrix<ndim, 1, double> g_1_cp(Core::LinAlg::Initialization::zero);


  std::vector<Core::LinAlg::Matrix<numdof, numdof, double>> lin_v_thetaperp_moment_cp(
      BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<numdof, numdof, double>(Core::LinAlg::Initialization::zero));

  std::vector<Core::LinAlg::Matrix<numdof, numdof, double>> lin_v_thetapar_moment_cp(
      BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<numdof, numdof, double>(Core::LinAlg::Initialization::zero));


  // set nodal / cp quantities: positions, tangents, triads
  // *******************************************************************************

  // Set current positions and orientations at all nodes:
  update_disp_totlag<nnodecl, double>(disp, disp_totlag);


  update_nodal_variables<nnodecl, double>(disp_totlag, disp_totlag_centerline, triad_mat_cp,
      Qref_dummy);  // Todo make this nicer and avoid dummies!


  // compute quantities at collocation points
  // *******************************************************************************
  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    // Determine storage position for this cp
    ind = Core::LargeRotations::numbering_trafo(icp + 1, BEAM3K_COLLOCATION_POINTS);

    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    xi_cp = (double)icp / (double)(BEAM3K_COLLOCATION_POINTS - 1) * 2.0 - 1.0;

    // get value of interpolating function for theta (Lagrange polynomials) at xi_cp
    L_i.clear();
    Core::FE::shape_function_1d(L_i, xi_cp, shape());

    L.clear();
    assemble_shapefunctions_l(L_i, L);


    N_i_xi.clear();
    Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi_cp, length_, Core::FE::CellType::line2);

    N_s.clear();
    assemble_shapefunctions_ns(N_i_xi, jacobi_cp_[ind], N_s);


    // Calculation of first base vector at xi
    r_s.clear();
    r_s.multiply(N_s, disp_totlag_centerline);

    abs_r_s = Core::FADUtils::norm(r_s);

    g_1_cp.clear();
    g_1_cp.update(std::pow(abs_r_s, -1.0), r_s);


    calc_lin_v_thetaperp_moment<nnodecl>(
        lin_v_thetaperp_moment_cp[ind], N_s, g_1_cp, abs_r_s, S_of_moment);

    calc_lin_v_thetapar_moment<nnodecl>(
        lin_v_thetapar_moment_cp[ind], L, N_s, g_1_cp, abs_r_s, moment);
  }


  // re-interpolation of quantities at xi based on CP values
  // *******************************************************************************
  L_i.clear();
  Core::FE::shape_function_1d(L_i, xi, shape());

  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    stiffmat_fixedsize.update(L_i(icp), lin_v_thetaperp_moment_cp[ind], 1.0);
    stiffmat_fixedsize.update(L_i(icp), lin_v_thetapar_moment_cp[ind], 1.0);
  }
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3k::get_generalized_interpolation_matrix_increments_at_xi(
    Core::LinAlg::SerialDenseMatrix& Iinc, const double& xi, const std::vector<double>& disp) const
{
  if (not weakkirchhoff_) FOUR_C_THROW("method is limited to WK so far! extend to SK if needed");

  if (rotvec_)
    FOUR_C_THROW("method is limited to tangent-based parametrization! extend to rotvec if needed");


  const unsigned int ndim = 3;
  const unsigned int nnodecl = 2;
  const unsigned int vpernode = 2;
  const unsigned int numdof = ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS;

  // safety check
  if ((unsigned int)Iinc.numRows() != 2 * ndim or (unsigned int) Iinc.numCols() != numdof)
    FOUR_C_THROW("size mismatch! expected {}x{} matrix and got {}x{}", 6, numdof, Iinc.numRows(),
        Iinc.numCols());

  Iinc.putScalar(0.0);


  // *******************************************************************************
  // concerning interpolation of increment of CENTERLINE POSITION \Delta r:
  // *******************************************************************************

  Core::LinAlg::Matrix<ndim, numdof, double> N(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, vpernode * nnodecl, double> N_i(Core::LinAlg::Initialization::zero);


  Core::FE::shape_function_hermite_1d(N_i, xi, length_, Core::FE::CellType::line2);
  assemble_shapefunctions_n(N_i, N);

  // this part is associated with the increment of the centerline position
  // (first three rows of Iinc)
  for (unsigned int irow = 0; irow < N.num_rows(); ++irow)
    for (unsigned int icol = 0; icol < N.num_cols(); ++icol) Iinc(irow, icol) += N(irow, icol);


  // *******************************************************************************
  // concerning interpolation of increment of CENTERLINE ORIENTATION \Delta \theta:
  // *******************************************************************************

  // define and initialize variables
  // *******************************************************************************
  // position index where CP quantities have to be stored (according to numbering convention)
  unsigned int ind = 0;
  double xi_cp = 0.0;


  Core::LinAlg::Matrix<ndim, numdof, double> N_s(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, vpernode * nnodecl, double> N_i_xi(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<1, numdof, double> L(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i(
      Core::LinAlg::Initialization::zero);


  Core::LinAlg::Matrix<numdof, 1, double> disp_totlag(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdof, 1, double> disp_totlag_centerline(
      Core::LinAlg::Initialization::zero);

  std::vector<Core::LinAlg::Matrix<3, 3, double>> triad_mat_cp(BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<3, 3, double>(Core::LinAlg::Initialization::zero));
  std::vector<Core::LinAlg::Matrix<4, 1>> Qref_dummy(
      BEAM3K_COLLOCATION_POINTS, Core::LinAlg::Matrix<4, 1>(Core::LinAlg::Initialization::zero));


  Core::LinAlg::Matrix<3, 1, double> r_s(Core::LinAlg::Initialization::zero);  // r' vector
  double abs_r_s = 0.0;                                                        // ||r'||

  // first base vector at CP
  Core::LinAlg::Matrix<3, 1, double> g_1_cp(Core::LinAlg::Initialization::zero);
  // first base vector of the triad, from which the new intermediate triad is obtained via
  // smallest rotation (SR); this triad is arbitrary, but we choose the intermediate triad
  // of the last time step; see Dissertation Meier, p.25
  Core::LinAlg::Matrix<3, 1, double> g_1_cp_bar(Core::LinAlg::Initialization::zero);


  Core::LinAlg::Matrix<ndim, numdof, double> lin_theta_perp_cp{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<ndim, numdof, double> lin_theta_par_cp(Core::LinAlg::Initialization::zero);

  // lin_theta_cp = lin_theta_perp_cp + lin_theta_par_cp
  std::vector<Core::LinAlg::Matrix<ndim, numdof, double>> lin_theta_cp(BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<ndim, numdof, double>(Core::LinAlg::Initialization::zero));

  // re-interpolated lin_theta:
  Core::LinAlg::Matrix<ndim, numdof, double> lin_theta_bar(Core::LinAlg::Initialization::zero);


  // set nodal / cp quantities: positions, tangents, triads
  // *******************************************************************************

  // Set current positions and orientations at all nodes:
  update_disp_totlag<nnodecl, double>(disp, disp_totlag);


  update_nodal_variables<nnodecl, double>(disp_totlag, disp_totlag_centerline, triad_mat_cp,
      Qref_dummy);  // Todo @grill: make this nicer and avoid dummies!


  // compute quantities at collocation points
  // *******************************************************************************
  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    // Determine storage position for this cp
    ind = Core::LargeRotations::numbering_trafo(icp + 1, BEAM3K_COLLOCATION_POINTS);

    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    xi_cp = (double)icp / (double)(BEAM3K_COLLOCATION_POINTS - 1) * 2.0 - 1.0;

    // get value of interpolating function for theta (Lagrange polynomials) at xi_cp
    L_i.clear();
    Core::FE::shape_function_1d(L_i, xi_cp, shape());

    L.clear();
    assemble_shapefunctions_l(L_i, L);

    N_i_xi.clear();
    Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi_cp, length_, Core::FE::CellType::line2);

    N_s.clear();
    assemble_shapefunctions_ns(N_i_xi, jacobi_cp_[ind], N_s);

    // Calculation of r' at xi
    r_s.clear();
    r_s.multiply(N_s, disp_totlag_centerline);

    abs_r_s = Core::FADUtils::norm(r_s);


    // lin_thetaperp
    calc_lin_thetaperp<nnodecl>(lin_theta_perp_cp, N_s, r_s, abs_r_s);


    // v_lin_thetapar
    g_1_cp.clear();
    g_1_cp.update(std::pow(abs_r_s, -1.0), r_s);

    Core::LinAlg::Matrix<3, 3, double> triad_ref_conv_cp(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::quaterniontotriad(qrefconv_[ind], triad_ref_conv_cp);

    g_1_cp_bar.clear();
    for (unsigned int idim = 0; idim < ndim; ++idim) g_1_cp_bar(idim) = triad_ref_conv_cp(idim, 0);

    calc_lin_thetapar<nnodecl>(lin_theta_par_cp, L, N_s, g_1_cp, g_1_cp_bar, abs_r_s);

    // lin_theta
    lin_theta_cp[ind].clear();
    lin_theta_cp[ind].update(1.0, lin_theta_par_cp, 1.0, lin_theta_perp_cp);
  }


  // re-interpolation of quantities at xi based on CP values
  // *******************************************************************************
  std::vector<Core::LinAlg::Matrix<3, 3, double>> Itilde(BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<3, 3, double>(Core::LinAlg::Initialization::zero));

  // create object of triad interpolation scheme
  LargeRotations::TriadInterpolationLocalRotationVectors<3, double> triad_interpolation_scheme_ptr;

  // reset scheme with nodal triads
  triad_interpolation_scheme_ptr.reset(triad_mat_cp);

  // compute Itilde matrices required for re-interpolation of CP values of lin_theta
  triad_interpolation_scheme_ptr.get_nodal_generalized_rotation_interpolation_matrices_at_xi(
      Itilde, xi);


  Core::LinAlg::Matrix<3, numdof, double> auxmatrix(Core::LinAlg::Initialization::zero);

  lin_theta_bar.clear();
  for (unsigned int inode = 0; inode < BEAM3K_COLLOCATION_POINTS; ++inode)
  {
    auxmatrix.clear();

    auxmatrix.multiply(Itilde[inode], lin_theta_cp[inode]);

    lin_theta_bar.update(1.0, auxmatrix, 1.0);
  }


  // finally assemble the generalized interpolation matrix for the increments Iinc
  // *******************************************************************************

  // this part is associated with the increment of the centerline orientation
  // (expressed as rotation vector theta)
  // (rows 4-6 of Iinc)
  for (unsigned int irow = 0; irow < lin_theta_bar.num_rows(); ++irow)
    for (unsigned int icol = 0; icol < lin_theta_bar.num_cols(); ++icol)
      Iinc(ndim + irow, icol) += lin_theta_bar(irow, icol);
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3k::extract_centerline_dof_values_from_element_state_vector(
    const std::vector<double>& dofvec, std::vector<double>& dofvec_centerline,
    bool add_reference_values) const
{
  if (dofvec.size() != 15)
    FOUR_C_THROW(
        "size mismatch: expected 15 values for element state vector and got {}", dofvec.size());

  dofvec_centerline.resize(12, 0.0);

  // we use the method for Core::LINALG fixed size matrix and create it as a view on the STL vector
  Core::LinAlg::Matrix<15, 1, double> dofvec_fixedsize(dofvec.data());
  Core::LinAlg::Matrix<12, 1, double> dofvec_centerline_fixedsize(dofvec_centerline.data(), true);

  this->extract_centerline_dof_values_from_element_state_vector<2, 2, double>(
      dofvec_fixedsize, dofvec_centerline_fixedsize, add_reference_values);
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, unsigned int vpernode, typename T>
void Discret::Elements::Beam3k::extract_centerline_dof_values_from_element_state_vector(
    const Core::LinAlg::Matrix<3 * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& dofvec,
    Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, T>& dofvec_centerline,
    bool add_reference_values) const
{
  // nnodecl: number of nodes used for interpolation of centerline
  // vpernode: number of interpolated values per centerline node (1: value (i.e. Lagrange), 2: value
  // + derivative of value (i.e. Hermite))

  const int dofperboundarynode = 3 * vpernode + 1;

  // get current values for position DOFs directly
  for (unsigned int dim = 0; dim < 3; ++dim)
    for (unsigned int node = 0; node < nnodecl; ++node)
    {
      dofvec_centerline(3 * vpernode * node + dim) = dofvec(dofperboundarynode * node + dim);
    }

  // Hermite interpolation: get values of tangent DOFs as well
  if (rotvec_ == false)
  {
    for (unsigned int dim = 0; dim < 3; ++dim)
      for (unsigned int node = 0; node < nnodecl; ++node)
      {
        // in case of rotvec_==false, tangents are equivalent to nodal DOFs
        dofvec_centerline(3 * vpernode * node + 3 + dim) =
            dofvec(dofperboundarynode * node + 3 + dim);
      }
  }
  else
  {
    // values for tangent DOFs must be transformed in case of rotvec_==true
    Core::LinAlg::Matrix<3, 1, T> theta(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 3, T> triad(Core::LinAlg::Initialization::zero);

    for (unsigned int node = 0; node < nnodecl; ++node)
    {
      for (unsigned int dim = 0; dim < 3; ++dim)
      {
        // get values for rotation vector DOFs
        theta(dim) = dofvec(dofperboundarynode * node + 3 + dim);
      }
      // transform to triad
      Core::LargeRotations::angletotriad(theta, triad);

      // direction of tangent is equivalent to first base vector of triad; length of tangent is 7th
      // nodal DOF
      for (unsigned int dim = 0; dim < 3; ++dim)
      {
        dofvec_centerline(3 * vpernode * node + 3 + dim) =
            triad(dim, 0) * dofvec(dofperboundarynode * node + 6);
      }
    }
  }

  if (add_reference_values) add_ref_values_disp_centerline<nnodecl, vpernode, T>(dofvec_centerline);
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void Discret::Elements::Beam3k::add_ref_values_disp(
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& dofvec) const
{
  // For the boundary nodes we have to add the initial values. For the interior nodes we are already
  // done since the initial relative angles are zero: alpha_i(t=0)=0;
  if (rotvec_ == false)
  {
    // Calculate total displacements = positions vectors, tangents and relative angles
    for (unsigned int node = 0; node < 2; ++node)  // loop over boundary nodes
    {
      for (unsigned int ndof = 0; ndof < 7; ++ndof)  // loop over dofs per node
      {
        if (ndof < 3)
        {
          dofvec(7 * node + ndof) += (nodes()[node])->x()[ndof];
        }
        else if (ndof < 6)
        {
          dofvec(7 * node + ndof) += (tref()[node])(ndof - 3);
        }
        else
        {
          // nothing to do here: alpha_i(t=0)=0;
        }
      }
    }
  }
  else
  {
    // Calculate total displacements = positions vectors, tangents and relative angles
    for (unsigned int node = 0; node < 2; ++node)  // loop over boundary nodes
    {
      for (unsigned int ndof = 0; ndof < 7; ++ndof)  // loop over dofs per node
      {
        if (ndof < 3)
        {
          dofvec(7 * node + ndof) += (nodes()[node])->x()[ndof];
        }
        else if (ndof < 6)
        {
          // Nothing to do here, rotations are treated below
        }
        else
        {
          // here we have to add the initial length of the tangents at the boundary nodes,
          // i.e. ||r'_i(t=0)||=1:
          dofvec(7 * node + ndof) += 1.0;
        }
      }  // for (int ndof=0;ndof<7;ndof++)//loop over dofs per node

      Core::LinAlg::Matrix<3, 1> disptheta(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<3, 1> thetanew(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<4, 1> deltaQ(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<4, 1> Qnew(Core::LinAlg::Initialization::zero);
      for (unsigned int i = 0; i < 3; ++i)
      {
        disptheta(i) = Core::FADUtils::cast_to_double(dofvec(7 * node + 3 + i));
      }

      Core::LargeRotations::angletoquaternion(disptheta, deltaQ);

      Core::LinAlg::Matrix<4, 1> Q0;
      Core::LargeRotations::angletoquaternion(theta0()[node], Q0);
      Core::LargeRotations::quaternionproduct(Q0, deltaQ, Qnew);

      // renormalize quaternion to keep its absolute value one even in case of long simulations
      // and intricate calculations
      Qnew.scale(1.0 / Qnew.norm2());

      // Calculate the new nodal angle thetanew \in ]-PI,PI] -> Here, thetanew \in ]-PI,PI] by
      // quaterniontoangle()
      Core::LargeRotations::quaterniontoangle(Qnew, thetanew);

      // Finally set rotation values in disp_totlag
      for (unsigned int i = 0; i < 3; ++i)
      {
        dofvec(7 * node + 3 + i) = thetanew(i);
      }
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void Discret::Elements::Beam3k::update_disp_totlag(const std::vector<double>& disp,
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag) const
{
  disp_totlag.clear();
  for (unsigned int dof = 0; dof < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; ++dof)
    disp_totlag(dof) = disp[dof];

  add_ref_values_disp<nnodecl, T>(disp_totlag);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void Discret::Elements::Beam3k::update_nodal_variables(
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag,
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag_centerline,
    std::vector<Core::LinAlg::Matrix<3, 3, T>>& triad_mat_cp,
    std::vector<Core::LinAlg::Matrix<4, 1>>& Qref_new) const
{
  // Set positions vectors and tangents and triads at boundary nodes
  set_positions_at_boundary_nodes<nnodecl, T>(disp_totlag, disp_totlag_centerline);

  // next, set triads and tangents at boundary nodes
  set_tangents_and_triads_and_reference_triads_at_boundary_nodes<nnodecl, T>(
      disp_totlag, disp_totlag_centerline, triad_mat_cp, Qref_new);

  // finally, set triads at remaining CPs (all except boundary nodes)
  set_triads_and_reference_triads_at_remaining_collocation_points<nnodecl, T>(
      disp_totlag, disp_totlag_centerline, triad_mat_cp, Qref_new);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void Discret::Elements::Beam3k::set_positions_at_boundary_nodes(
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag,
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag_centerline)
    const
{
  for (unsigned int i = 0; i < 3; ++i)
  {
    disp_totlag_centerline(i) = disp_totlag(i);
    disp_totlag_centerline(7 + i) = disp_totlag(7 + i);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void Discret::Elements::Beam3k::set_tangents_and_triads_and_reference_triads_at_boundary_nodes(
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag,
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag_centerline,
    std::vector<Core::LinAlg::Matrix<3, 3, T>>& triad_mat_cp,
    std::vector<Core::LinAlg::Matrix<4, 1>>& Qref_new) const
{
  if (rotvec_ == false)
  {
    Core::LinAlg::Matrix<3, 1, T> tangent(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 3, T> triad_ref(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 3> triad_aux(Core::LinAlg::Initialization::zero);
    // Todo @grill: get rid of auxiliary matrix of different type
    Core::LinAlg::Matrix<3, 3, T> triad_aux2(Core::LinAlg::Initialization::zero);
    T alpha = 0.0;

    for (unsigned int node = 0; node < 2; ++node)
    {
      // tangent
      for (unsigned int i = 0; i < 3; ++i)
      {
        disp_totlag_centerline(7 * node + 3 + i) = disp_totlag(7 * node + 3 + i);
        tangent(i) = disp_totlag(7 * node + 3 + i);
      }

      alpha = disp_totlag(7 * node + 6);

      // calculate new sr triad
      triad_ref.clear();
      triad_aux.clear();
      triad_aux2.clear();
      Core::LargeRotations::quaterniontotriad(qrefconv_[node], triad_aux);
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j) triad_aux2(i, j) = triad_aux(i, j);

      Core::LargeRotations::calculate_sr_triads<T>(tangent, triad_aux2, triad_ref);

      // Store nodal reference triad
      Core::LinAlg::Matrix<4, 1, T> Qref(Core::LinAlg::Initialization::zero);
      Core::LargeRotations::triadtoquaternion(triad_ref, Qref);
      for (unsigned int i = 0; i < 4; ++i)
        Qref_new[node](i) = Core::FADUtils::cast_to_double(Qref(i));

      // calculate material triad
      triad_mat_cp[node].clear();
      Core::LargeRotations::rotate_triad(triad_ref, alpha, triad_mat_cp[node]);
    }
  }
  else
  {
    Core::LinAlg::Matrix<3, 1, T> theta(Core::LinAlg::Initialization::zero);
    for (unsigned int node = 0; node < 2; ++node)
    {
      for (unsigned int i = 0; i < 3; ++i)
      {
        theta(i) = disp_totlag(7 * node + 3 + i);
      }
      triad_mat_cp[node].clear();
      Core::LargeRotations::angletotriad(theta, triad_mat_cp[node]);

      // tangent
      for (unsigned int i = 0; i < 3; ++i)
      {
        disp_totlag_centerline(7 * node + 3 + i) =
            (triad_mat_cp[node])(i, 0) * disp_totlag(7 * node + 6);
      }
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void Discret::Elements::Beam3k::set_triads_and_reference_triads_at_remaining_collocation_points(
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag,
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
        disp_totlag_centerline,
    std::vector<Core::LinAlg::Matrix<3, 3, T>>& triad_mat_cp,
    std::vector<Core::LinAlg::Matrix<4, 1>>& Qref_new) const
{
  Core::LinAlg::Matrix<3, 1, T> tangent(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, T> triad_ref(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> triad_aux(Core::LinAlg::Initialization::zero);
  // Todo @grill: get rid of auxiliary matrix of different type
  Core::LinAlg::Matrix<3, 3, T> triad_aux2(Core::LinAlg::Initialization::zero);
  T alpha = 0.0;

  Core::LinAlg::Matrix<1, 4, T> N_i_xi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, T> N_s(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, T> L_i;
  double xi = 0.0;
  unsigned int ind = 0;

  //********begin: evaluate quantities at collocation points********************************
  // intermediate nodes: all but first (node=0) and last node (node=BEAM3K_COLLOCATION_POINTS)
  for (unsigned int node = 1; node < BEAM3K_COLLOCATION_POINTS - 1; ++node)
  {
    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0 node=2->xi=1
    xi = (double)node / (double)(BEAM3K_COLLOCATION_POINTS - 1) * 2.0 - 1.0;
    N_i_xi.clear();
    Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi, length_, Core::FE::CellType::line2);
    L_i.clear();
    Core::FE::shape_function_1d(L_i, xi, shape());

    // Determine storage position for the node node
    ind = Core::LargeRotations::numbering_trafo(node + 1, BEAM3K_COLLOCATION_POINTS);

    N_s.clear();
    assemble_shapefunctions_ns(N_i_xi, jacobi_cp_[ind], N_s);

    tangent.clear();
    // Calculation of r' at xi
    tangent.multiply(N_s, disp_totlag_centerline);

    alpha = disp_totlag(7 * 2 + ind - 2);

    // calculate new sr triads
    triad_ref.clear();
    triad_aux.clear();
    triad_aux2.clear();
    Core::LargeRotations::quaterniontotriad(qrefconv_[ind], triad_aux);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j) triad_aux2(i, j) = triad_aux(i, j);

    Core::LargeRotations::calculate_sr_triads<T>(tangent, triad_aux2, triad_ref);

    // Store nodal reference triad
    Core::LinAlg::Matrix<4, 1, T> Qref(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::triadtoquaternion(triad_ref, Qref);
    for (unsigned int i = 0; i < 4; ++i) Qref_new[ind](i) = Core::FADUtils::cast_to_double(Qref(i));

    // calculate material triad
    triad_mat_cp[ind].clear();
    Core::LargeRotations::rotate_triad(triad_ref, alpha, triad_mat_cp[ind]);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::set_automatic_differentiation_variables(
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>& disp_totlag) const
{
  for (unsigned int dof = 0; dof < nnodecl * 6 + BEAM3K_COLLOCATION_POINTS; dof++)
  {
    disp_totlag(dof).diff(dof, nnodecl * 6 + BEAM3K_COLLOCATION_POINTS);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, unsigned int vpernode, unsigned int ndim>
void Discret::Elements::Beam3k::calc_velocity(
    const Core::LinAlg::Matrix<ndim * vpernode * nnodecl, 1, double>& velocity_dofvec,
    const Core::LinAlg::Matrix<1, vpernode * nnodecl, double>& N_i,
    Core::LinAlg::Matrix<ndim, 1, double>& velocity,
    const Core::LinAlg::Matrix<ndim, 1, double>& position, int gausspoint_index) const
{
  Discret::Utils::Beam::calc_interpolation<nnodecl, vpernode, ndim, double>(
      velocity_dofvec, N_i, velocity);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, unsigned int vpernode, unsigned int ndim>
void Discret::Elements::Beam3k::calc_velocity(
    const Core::LinAlg::Matrix<ndim * vpernode * nnodecl, 1, double>& velocity_dofvec,
    const Core::LinAlg::Matrix<1, vpernode * nnodecl, double>& N_i,
    Core::LinAlg::Matrix<ndim, 1, FAD>& velocity,
    const Core::LinAlg::Matrix<ndim, 1, FAD>& position, int gausspoint_index)
{
  /* if we use FAD, we need to track the dependency of velocity vector on primary variables, i.e.
   * we must calculate it like this using the class variable rconvmass_ for position vector at GP
   * of last converged state this again is tricky in case of periodic boundary conditions, because
   * the position of this element (nodes) might have been shifted outside; we therefore manually
   * adapt the calculated velocity and compare it with the velocity calculated in time integrator
   * and handed in from outside for safety reasons */
  Core::LinAlg::Matrix<ndim, 1, double> velocity_test;
  Discret::Utils::Beam::calc_interpolation<nnodecl, vpernode, ndim, double>(
      velocity_dofvec, N_i, velocity_test);

  // get time step size
  const double dt = params_interface().get_delta_time();

  Core::LinAlg::Matrix<3, 1> diff(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<ndim, 1, FAD> delta_r_ost(Core::LinAlg::Initialization::zero);
  std::shared_ptr<Core::Geo::MeshFree::BoundingBox> pbb =
      brownian_dyn_params_interface().get_periodic_bounding_box();

  Core::LinAlg::Matrix<3, 1> unshiftedrconvmass_i(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> position_i_double(Core::LinAlg::Initialization::zero);

  for (unsigned int idim = 0; idim < ndim; ++idim)
  {
    unshiftedrconvmass_i(idim) = rconvmass_[gausspoint_index](idim);
    position_i_double(idim) = Core::FADUtils::cast_to_double(position(idim));
  }

  // difference in position of this GP as compared to last time step
  pbb->un_shift_3d(unshiftedrconvmass_i, position_i_double);

  for (unsigned int idim = 0; idim < ndim; ++idim)
  {
    // difference in position of this GP as compared to last time step
    delta_r_ost(idim) = position(idim) - unshiftedrconvmass_i(idim);

    // velocity according to Backward Euler scheme
    velocity(idim) = delta_r_ost(idim) / dt;

    diff(idim) = velocity(idim).val() - velocity_test(idim);

    // set class variable, such that rconvmass_ is available in next time step
    rnewmass_[gausspoint_index](idim) = position(idim).val();
  }

  // safety check
  if (diff.norm_inf() > 1e-11)
  {
    std::cout << "\nrnewmass = " << position;
    std::cout << "\nrconvmass_ = " << rconvmass_[gausspoint_index];
    std::cout << "\nvel = " << velocity;
    std::cout << "\nvel_test = " << velocity_test;
    std::cout << "\nabs(diff) = " << diff.norm2();
    std::cout << "\n*** SERIOUS WARNING: ***";
    std::cout << "\nvelocity vector at GP computed locally in beam3k differs from OneStepTheta "
                 "velocity vector (see values above)!\n\n";
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void Discret::Elements::Beam3k::calc_v_thetaperp(
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, T>& v_thetaperp,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, T>& N_s,
    const Core::LinAlg::Matrix<3, 1, T>& r_s, T abs_r_s) const
{
  v_thetaperp.clear();

  Core::LinAlg::Matrix<3, 3, T> S_of_r_s(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::computespin<T>(S_of_r_s, r_s);

  v_thetaperp.multiply_tn(N_s, S_of_r_s);
  v_thetaperp.scale(-1.0 * std::pow(abs_r_s, -2.0));
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void Discret::Elements::Beam3k::calc_v_thetapartheta(
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, T>& v_thetapartheta,
    const Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, T>& L,
    const Core::LinAlg::Matrix<3, 1, T>& r_s, T abs_r_s) const
{
  v_thetapartheta.clear();

  for (unsigned int idof = 0; idof < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; ++idof)
    for (unsigned int jdim = 0; jdim < 3; ++jdim)
      v_thetapartheta(idof, jdim) = L(idof) * r_s(jdim) / abs_r_s;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calc_lin_thetaperp(
    Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_thetaperp,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const Core::LinAlg::Matrix<3, 1, double>& r_s, double abs_r_s) const
{
  // Todo @grill: maybe re-use method for v_thetaperp here, which is simply the transpose of this
  // term

  lin_thetaperp.clear();

  Core::LinAlg::Matrix<3, 3, double> S_of_r_s(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::computespin<double>(S_of_r_s, r_s);

  lin_thetaperp.multiply(S_of_r_s, N_s);
  lin_thetaperp.scale(std::pow(abs_r_s, -2.0));
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calc_lin_thetapar(
    Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_thetapar,
    const Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& L,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const Core::LinAlg::Matrix<3, 1, double>& g_1,
    const Core::LinAlg::Matrix<3, 1, double>& g_1_bar, double abs_r_s) const
{
  const unsigned int numdof = 6 * nnodecl + BEAM3K_COLLOCATION_POINTS;

  lin_thetapar.clear();

  Core::LinAlg::Matrix<3, 3, double> S_of_g_1(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::computespin<double>(S_of_g_1, g_1);


  // Todo @grill: decide about alternatives, apparently no change in results

  // *********** alternative 1 ************
  Core::LinAlg::Matrix<3, 3, double> g_1dyadicproduct_g_1_bar_T(Core::LinAlg::Initialization::zero);

  for (unsigned int irow = 0; irow < 3; ++irow)
    for (unsigned int icol = 0; icol < 3; ++icol)
      g_1dyadicproduct_g_1_bar_T(irow, icol) = g_1(irow) * g_1_bar(icol);

  Core::LinAlg::Matrix<3, 3, double> g_1dyadicproduct_g_1_bar_T_S_of_g_1(
      Core::LinAlg::Initialization::zero);
  g_1dyadicproduct_g_1_bar_T_S_of_g_1.multiply(g_1dyadicproduct_g_1_bar_T, S_of_g_1);

  lin_thetapar.multiply(g_1dyadicproduct_g_1_bar_T_S_of_g_1, N_s);

  // *********** alternative 2 ************

  //  Core::LinAlg::Matrix<1,3,double> g_1_bar_T_S_of_g_1(true);
  //  g_1_bar_T_S_of_g_1.multiply_tn(g_1_bar, S_of_g_1);
  //
  //
  //  Core::LinAlg::Matrix<1,numdof,double> g_1_bar_T_S_of_g_1_N_s(true);
  //  g_1_bar_T_S_of_g_1_N_s.Multiply(g_1_bar_T_S_of_g_1, N_s);
  //
  //
  //  // dyadic product
  //  for (unsigned int idim=0; idim<3; ++idim)
  //    for (unsigned int jdof=0; jdof<numdof; ++jdof)
  //      lin_thetapar(idim,jdof) += g_1(idim) * g_1_bar_T_S_of_g_1_N_s(jdof) ;

  // *************************************

  lin_thetapar.scale(-1.0 / (1.0 + g_1.dot(g_1_bar)));
  lin_thetapar.scale(1.0 / abs_r_s);


  for (unsigned int idim = 0; idim < 3; ++idim)
    for (unsigned int jdof = 0; jdof < numdof; ++jdof)
      lin_thetapar(idim, jdof) += g_1(idim) * L(jdof);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calc_lin_tangent_tilde(
    Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_tangent_tilde,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const Core::LinAlg::Matrix<3, 1, double>& g_1, double abs_r_s) const
{
  lin_tangent_tilde.clear();

  Core::LinAlg::Matrix<3, 3, double> auxmatrix;

  for (unsigned int idim = 0; idim < 3; ++idim)
    for (unsigned int jdim = 0; jdim < 3; ++jdim)
      auxmatrix(idim, jdim) = 1.0 * (idim == jdim) - 2.0 * g_1(idim) * g_1(jdim);

  auxmatrix.scale(std::pow(abs_r_s, -2.0));

  lin_tangent_tilde.multiply(auxmatrix, N_s);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calc_lin_tangent_tilde_s(
    Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_tangent_tilde_s,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_ss,
    const Core::LinAlg::Matrix<3, 1, double>& g_1, const Core::LinAlg::Matrix<3, 1, double>& g_1_s,
    const Core::LinAlg::Matrix<3, 1, double>& r_s, const Core::LinAlg::Matrix<3, 1, double>& r_ss,
    double abs_r_s) const
{
  lin_tangent_tilde_s.clear();

  Core::LinAlg::Matrix<3, 3, double> auxmatrix;
  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> summand(
      Core::LinAlg::Initialization::zero);

  // first summand
  for (unsigned int idim = 0; idim < 3; ++idim)
    for (unsigned int jdim = 0; jdim < 3; ++jdim)
      auxmatrix(idim, jdim) = 1.0 * (idim == jdim) - 2.0 * g_1(idim) * g_1(jdim);

  auxmatrix.scale(2.0 * r_s.dot(r_ss) * std::pow(abs_r_s, -4.0));

  lin_tangent_tilde_s.multiply(auxmatrix, N_s);

  // second summand
  auxmatrix.clear();

  for (unsigned int idim = 0; idim < 3; ++idim)
    for (unsigned int jdim = 0; jdim < 3; ++jdim)
      auxmatrix(idim, jdim) = g_1_s(idim) * g_1(jdim) + g_1(idim) * g_1_s(jdim);

  auxmatrix.scale(-2.0 * std::pow(abs_r_s, -2.0));

  summand.multiply(auxmatrix, N_s);

  lin_tangent_tilde_s.update(1.0, summand, 1.0);

  // third summand
  auxmatrix.clear();

  for (unsigned int idim = 0; idim < 3; ++idim)
    for (unsigned int jdim = 0; jdim < 3; ++jdim)
      auxmatrix(idim, jdim) = 1.0 * (idim == jdim) - 2.0 * g_1(idim) * g_1(jdim);

  auxmatrix.scale(std::pow(abs_r_s, -2.0));

  summand.clear();

  summand.multiply(auxmatrix, N_ss);

  lin_tangent_tilde_s.update(1.0, summand, 1.0);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calc_lin_g_1(
    Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_g_1,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const Core::LinAlg::Matrix<3, 1, double>& g_1, double abs_r_s) const
{
  lin_g_1.clear();

  Core::LinAlg::Matrix<3, 3, double> auxmatrix;

  for (unsigned int idim = 0; idim < 3; ++idim)
    for (unsigned int jdim = 0; jdim < 3; ++jdim)
      auxmatrix(idim, jdim) = 1.0 * (idim == jdim) - g_1(idim) * g_1(jdim);

  auxmatrix.scale(std::pow(abs_r_s, -1.0));

  lin_g_1.multiply(auxmatrix, N_s);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calc_lin_g_1_s(
    Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_g_1_s,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_ss,
    const Core::LinAlg::Matrix<3, 1, double>& g_1, const Core::LinAlg::Matrix<3, 1, double>& g_1_s,
    const Core::LinAlg::Matrix<3, 1, double>& r_s, const Core::LinAlg::Matrix<3, 1, double>& r_ss,
    double abs_r_s) const
{
  lin_g_1_s.clear();

  Core::LinAlg::Matrix<3, 3, double> auxmatrix;
  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> summand(
      Core::LinAlg::Initialization::zero);

  // first summand
  for (unsigned int idim = 0; idim < 3; ++idim)
    for (unsigned int jdim = 0; jdim < 3; ++jdim)
      auxmatrix(idim, jdim) = 1.0 * (idim == jdim) - g_1(idim) * g_1(jdim);

  auxmatrix.scale(-1.0 * r_s.dot(r_ss) * std::pow(abs_r_s, -3.0));

  lin_g_1_s.multiply(auxmatrix, N_s);

  // second summand
  auxmatrix.clear();

  for (unsigned int idim = 0; idim < 3; ++idim)
    for (unsigned int jdim = 0; jdim < 3; ++jdim)
      auxmatrix(idim, jdim) = g_1_s(idim) * g_1(jdim) + g_1(idim) * g_1_s(jdim);

  auxmatrix.scale(-1.0 * std::pow(abs_r_s, -1.0));

  summand.multiply(auxmatrix, N_s);

  lin_g_1_s.update(1.0, summand, 1.0);

  // third summand
  auxmatrix.clear();

  for (unsigned int idim = 0; idim < 3; ++idim)
    for (unsigned int jdim = 0; jdim < 3; ++jdim)
      auxmatrix(idim, jdim) = 1.0 * (idim == jdim) - g_1(idim) * g_1(jdim);

  auxmatrix.scale(std::pow(abs_r_s, -1.0));

  summand.clear();

  summand.multiply(auxmatrix, N_ss);

  lin_g_1_s.update(1.0, summand, 1.0);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calc_lin_v_epsilon(
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
        6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_v_epsilon,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const Core::LinAlg::Matrix<3, 1, double>& g_1, double abs_r_s) const
{
  lin_v_epsilon.clear();

  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> lin_g_1(
      Core::LinAlg::Initialization::zero);

  calc_lin_g_1<nnodecl>(lin_g_1, N_s, g_1, abs_r_s);

  lin_v_epsilon.multiply_tn(N_s, lin_g_1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calc_lin_moment_resultant(
    Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_moment_resultant,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta_s,
    const Core::LinAlg::Matrix<3, 3, double>& spinmatrix_of_moment,
    const Core::LinAlg::Matrix<3, 3, double>& cm) const
{
  lin_moment_resultant.clear();

  // first summand
  lin_moment_resultant.multiply(spinmatrix_of_moment, lin_theta);
  lin_moment_resultant.scale(-1.0);

  // second summand
  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> auxmatrix(
      Core::LinAlg::Initialization::zero);

  auxmatrix.multiply(cm, lin_theta_s);

  lin_moment_resultant.update(1.0, auxmatrix, 1.0);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calc_lin_moment_inertia(
    Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_moment_inertia,
    const Core::LinAlg::Matrix<3, 3, double>& triad_mat,
    const Core::LinAlg::Matrix<3, 3, double>& triad_mat_conv,
    const Core::LinAlg::Matrix<3, 1, double>& deltatheta,
    const Core::LinAlg::Matrix<3, 1, double>& angular_velocity_material,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta,
    const Core::LinAlg::Matrix<3, 3, double>& spinmatrix_of_moment,
    const Core::LinAlg::Matrix<3, 3, double>& C_rho, double lin_prefactor_acc,
    double lin_prefactor_vel) const
{
  lin_moment_inertia.clear();

  // first summand
  lin_moment_inertia.multiply(spinmatrix_of_moment, lin_theta);

  // second summand
  Core::LinAlg::Matrix<3, 3, double> auxmatrix(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 3, double> spinmatrix(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::computespin(spinmatrix, angular_velocity_material);

  auxmatrix.multiply(spinmatrix, C_rho);

  Core::LinAlg::Matrix<3, 1, double> C_rho_W(Core::LinAlg::Initialization::zero);
  C_rho_W.multiply(C_rho, angular_velocity_material);

  spinmatrix.clear();
  Core::LargeRotations::computespin(spinmatrix, C_rho_W);
  auxmatrix.update(-1.0, spinmatrix, 1.0);
  auxmatrix.scale(lin_prefactor_vel);

  auxmatrix.update(lin_prefactor_acc, C_rho, 1.0);


  Core::LinAlg::Matrix<3, 3, double> Tmat_of_deltatheta = Core::LargeRotations::tmatrix(deltatheta);
  Core::LinAlg::Matrix<3, 3, double> Lambda_conv_Tmat_of_deltatheta(
      Core::LinAlg::Initialization::zero);
  Lambda_conv_Tmat_of_deltatheta.multiply_tn(triad_mat_conv, Tmat_of_deltatheta);

  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> lin_THETA_tilde(
      Core::LinAlg::Initialization::zero);
  lin_THETA_tilde.multiply(Lambda_conv_Tmat_of_deltatheta, lin_theta);

  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> auxmatrix2(
      Core::LinAlg::Initialization::zero);
  auxmatrix2.multiply(auxmatrix, lin_THETA_tilde);


  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> auxmatrix3(
      Core::LinAlg::Initialization::zero);
  auxmatrix3.multiply(triad_mat, auxmatrix2);

  lin_moment_inertia.update(1.0, auxmatrix3, 1.0);

  lin_moment_inertia.scale(-1.0);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calc_lin_moment_viscous(
    Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_moment_viscous,
    const Core::LinAlg::Matrix<3, 3, double>& triad_mat,
    const Core::LinAlg::Matrix<3, 3, double>& triad_mat_conv,
    const Core::LinAlg::Matrix<3, 1, double>& deltatheta,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta,
    const Core::LinAlg::Matrix<3, 3, double>& spinmatrix_of_moment, double gamma_polar,
    double dt) const
{
  lin_moment_viscous.clear();

  // first summand
  lin_moment_viscous.multiply(spinmatrix_of_moment, lin_theta);
  lin_moment_viscous.scale(-1.0);

  // second summand
  Core::LinAlg::Matrix<3, 3, double> auxmatrix(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 3, double> gamma_g1_g1_conv;
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      gamma_g1_g1_conv(i, j) = triad_mat(i, 0) * triad_mat_conv(j, 0) * gamma_polar / dt;

  Core::LinAlg::Matrix<3, 3, double> Tmat_of_deltatheta = Core::LargeRotations::tmatrix(deltatheta);

  auxmatrix.multiply(gamma_g1_g1_conv, Tmat_of_deltatheta);

  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> second_summand(
      Core::LinAlg::Initialization::zero);
  second_summand.multiply(auxmatrix, lin_theta);

  lin_moment_viscous.update(1.0, second_summand, 1.0);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calc_lin_v_thetaperp_moment(
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
        6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_v_thetaperp_moment,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const Core::LinAlg::Matrix<3, 1, double>& g_1, double abs_r_s,
    const Core::LinAlg::Matrix<3, 3, double>& spinmatrix_of_moment) const
{
  lin_v_thetaperp_moment.clear();

  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> lin_tangent_tilde(
      Core::LinAlg::Initialization::zero);

  calc_lin_tangent_tilde<nnodecl>(lin_tangent_tilde, N_s, g_1, abs_r_s);


  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, double> auxmatrix(
      Core::LinAlg::Initialization::zero);

  // Todo @grill: check: is the order of matrix products relevant?
  auxmatrix.multiply_tn(N_s, spinmatrix_of_moment);

  lin_v_thetaperp_moment.multiply(auxmatrix, lin_tangent_tilde);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calc_lin_v_thetaperp_s_moment(
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
        6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_v_thetaperp_s_moment,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_ss,
    const Core::LinAlg::Matrix<3, 1, double>& g_1, const Core::LinAlg::Matrix<3, 1, double>& g_1_s,
    const Core::LinAlg::Matrix<3, 1, double>& r_s, const Core::LinAlg::Matrix<3, 1, double>& r_ss,
    double abs_r_s, const Core::LinAlg::Matrix<3, 3, double>& spinmatrix_of_moment) const
{
  lin_v_thetaperp_s_moment.clear();

  // first summand
  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> lin_tangent_tilde(
      Core::LinAlg::Initialization::zero);

  calc_lin_tangent_tilde<nnodecl>(lin_tangent_tilde, N_s, g_1, abs_r_s);

  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, double> auxmatrix(
      Core::LinAlg::Initialization::zero);

  auxmatrix.multiply_tn(N_ss, spinmatrix_of_moment);

  lin_v_thetaperp_s_moment.multiply(auxmatrix, lin_tangent_tilde);

  // second summand
  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> lin_tangent_tilde_s(
      Core::LinAlg::Initialization::zero);

  calc_lin_tangent_tilde_s<nnodecl>(lin_tangent_tilde_s, N_s, N_ss, g_1, g_1_s, r_s, r_ss, abs_r_s);

  auxmatrix.clear();
  auxmatrix.multiply_tn(N_s, spinmatrix_of_moment);

  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
      6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>
      second_summand(Core::LinAlg::Initialization::zero);

  second_summand.multiply(auxmatrix, lin_tangent_tilde_s);

  lin_v_thetaperp_s_moment.update(1.0, second_summand, 1.0);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calc_lin_v_thetapar_moment(
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
        6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_v_thetapar_moment,
    Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& L,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const Core::LinAlg::Matrix<3, 1, double>& g_1, double abs_r_s,
    const Core::LinAlg::Matrix<3, 1, double>& moment) const
{
  lin_v_thetapar_moment.clear();

  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> lin_g_1(
      Core::LinAlg::Initialization::zero);

  calc_lin_g_1<nnodecl>(lin_g_1, N_s, g_1, abs_r_s);

  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, double> auxmatrix(
      Core::LinAlg::Initialization::zero);

  // Todo @grill: check: is the order of matrix products relevant?
  for (unsigned int idof = 0; idof < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; ++idof)
    for (unsigned int jdim = 0; jdim < 3; ++jdim) auxmatrix(idof, jdim) = L(idof) * moment(jdim);

  lin_v_thetapar_moment.multiply(auxmatrix, lin_g_1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calc_lin_v_thetapar_s_moment(
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
        6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_v_thetapar_s_moment,
    Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& L,
    Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& L_s,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_ss,
    const Core::LinAlg::Matrix<3, 1, double>& g_1, const Core::LinAlg::Matrix<3, 1, double>& g_1_s,
    const Core::LinAlg::Matrix<3, 1, double>& r_s, const Core::LinAlg::Matrix<3, 1, double>& r_ss,
    double abs_r_s, const Core::LinAlg::Matrix<3, 1, double>& moment) const
{
  lin_v_thetapar_s_moment.clear();

  // first summand
  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> lin_g_1(
      Core::LinAlg::Initialization::zero);

  calc_lin_g_1<nnodecl>(lin_g_1, N_s, g_1, abs_r_s);

  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, double> auxmatrix(
      Core::LinAlg::Initialization::zero);

  for (unsigned int idof = 0; idof < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; ++idof)
    for (unsigned int jdim = 0; jdim < 3; ++jdim) auxmatrix(idof, jdim) = L_s(idof) * moment(jdim);

  lin_v_thetapar_s_moment.multiply(auxmatrix, lin_g_1);

  // second summand
  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> lin_g_1_s(
      Core::LinAlg::Initialization::zero);

  calc_lin_g_1_s<nnodecl>(lin_g_1_s, N_s, N_ss, g_1, g_1_s, r_s, r_ss, abs_r_s);

  auxmatrix.clear();

  for (unsigned int idof = 0; idof < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; ++idof)
    for (unsigned int jdim = 0; jdim < 3; ++jdim) auxmatrix(idof, jdim) = L(idof) * moment(jdim);

  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
      6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>
      second_summand(Core::LinAlg::Initialization::zero);

  second_summand.multiply(auxmatrix, lin_g_1_s);

  lin_v_thetapar_s_moment.update(1.0, second_summand, 1.0);
}

// explicit template instantiations
template void Discret::Elements::Beam3k::extract_centerline_dof_values_from_element_state_vector<2,
    2, Sacado::Fad::DFad<double>>(
    const Core::LinAlg::Matrix<12 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&,
    Core::LinAlg::Matrix<12, 1, Sacado::Fad::DFad<double>>&, bool) const;
template void Discret::Elements::Beam3k::extract_centerline_dof_values_from_element_state_vector<2,
    2, double>(const Core::LinAlg::Matrix<12 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    Core::LinAlg::Matrix<12, 1, double>&, bool) const;

template void Discret::Elements::Beam3k::add_ref_values_disp<2, double>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&) const;
template void Discret::Elements::Beam3k::add_ref_values_disp<2, Sacado::Fad::DFad<double>>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&) const;

template void Discret::Elements::Beam3k::update_disp_totlag<2, double>(const std::vector<double>&,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&) const;
template void Discret::Elements::Beam3k::update_disp_totlag<2, Sacado::Fad::DFad<double>>(
    const std::vector<double>&,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&) const;

template void Discret::Elements::Beam3k::update_nodal_variables<2, double>(
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    std::vector<Core::LinAlg::Matrix<3, 3, double>>&,
    std::vector<Core::LinAlg::Matrix<4, 1>>&) const;
template void Discret::Elements::Beam3k::update_nodal_variables<2, Sacado::Fad::DFad<double>>(
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&,
    std::vector<Core::LinAlg::Matrix<3, 3, Sacado::Fad::DFad<double>>>&,
    std::vector<Core::LinAlg::Matrix<4, 1>>&) const;

template void Discret::Elements::Beam3k::set_positions_at_boundary_nodes<2, double>(
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&) const;
template void
Discret::Elements::Beam3k::set_positions_at_boundary_nodes<2, Sacado::Fad::DFad<double>>(
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&) const;

template void
Discret::Elements::Beam3k::set_tangents_and_triads_and_reference_triads_at_boundary_nodes<2,
    double>(const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    std::vector<Core::LinAlg::Matrix<3, 3, double>>&,
    std::vector<Core::LinAlg::Matrix<4, 1>>&) const;
template void
Discret::Elements::Beam3k::set_tangents_and_triads_and_reference_triads_at_boundary_nodes<2,
    Sacado::Fad::DFad<double>>(
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&,
    std::vector<Core::LinAlg::Matrix<3, 3, Sacado::Fad::DFad<double>>>&,
    std::vector<Core::LinAlg::Matrix<4, 1>>&) const;

template void
Discret::Elements::Beam3k::set_triads_and_reference_triads_at_remaining_collocation_points<2,
    double>(const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    std::vector<Core::LinAlg::Matrix<3, 3, double>>&,
    std::vector<Core::LinAlg::Matrix<4, 1>>&) const;
template void
Discret::Elements::Beam3k::set_triads_and_reference_triads_at_remaining_collocation_points<2,
    Sacado::Fad::DFad<double>>(
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&,
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&,
    std::vector<Core::LinAlg::Matrix<3, 3, Sacado::Fad::DFad<double>>>&,
    std::vector<Core::LinAlg::Matrix<4, 1>>&) const;

template void Discret::Elements::Beam3k::set_automatic_differentiation_variables<2>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&) const;

template void Discret::Elements::Beam3k::calc_velocity<2, 2, 3>(
    const Core::LinAlg::Matrix<3 * 2 * 2, 1, double>&,
    const Core::LinAlg::Matrix<1, 2 * 2, double>&, Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, int) const;

template void Discret::Elements::Beam3k::calc_velocity<2, 2, 3>(
    const Core::LinAlg::Matrix<3 * 2 * 2, 1, double>&,
    const Core::LinAlg::Matrix<1, 2 * 2, double>&, Core::LinAlg::Matrix<3, 1, FAD>&,
    const Core::LinAlg::Matrix<3, 1, FAD>&, int);

template void Discret::Elements::Beam3k::calc_v_thetaperp<2, double>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 3, double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, double) const;
template void Discret::Elements::Beam3k::calc_v_thetaperp<2, Sacado::Fad::DFad<double>>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 3, Sacado::Fad::DFad<double>>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&,
    const Core::LinAlg::Matrix<3, 1, Sacado::Fad::DFad<double>>&, Sacado::Fad::DFad<double>) const;

template void Discret::Elements::Beam3k::calc_v_thetapartheta<2, double>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 3, double>&,
    const Core::LinAlg::Matrix<1, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, double) const;
template void Discret::Elements::Beam3k::calc_v_thetapartheta<2, Sacado::Fad::DFad<double>>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 3, Sacado::Fad::DFad<double>>&,
    const Core::LinAlg::Matrix<1, 6 * 2 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&,
    const Core::LinAlg::Matrix<3, 1, Sacado::Fad::DFad<double>>&, Sacado::Fad::DFad<double>) const;

template void Discret::Elements::Beam3k::calc_lin_thetaperp<2>(
    Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, double) const;

template void Discret::Elements::Beam3k::calc_lin_thetapar<2>(
    Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<1, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    double) const;

template void Discret::Elements::Beam3k::calc_lin_tangent_tilde<2>(
    Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, double) const;

template void Discret::Elements::Beam3k::calc_lin_tangent_tilde_s<2>(
    Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    double) const;

template void Discret::Elements::Beam3k::calc_lin_g_1<2>(
    Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, double) const;

template void Discret::Elements::Beam3k::calc_lin_g_1_s<2>(
    Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    double) const;

template void Discret::Elements::Beam3k::calc_lin_v_epsilon<2>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 6 * 2 + BEAM3K_COLLOCATION_POINTS,
        double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, double) const;

template void Discret::Elements::Beam3k::calc_lin_moment_resultant<2>(
    Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&, const Core::LinAlg::Matrix<3, 3, double>&) const;

template void Discret::Elements::Beam3k::calc_lin_moment_inertia<2>(
    Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&, const Core::LinAlg::Matrix<3, 3, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&, const Core::LinAlg::Matrix<3, 3, double>&, double,
    double) const;

template void Discret::Elements::Beam3k::calc_lin_moment_viscous<2>(
    Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&, const Core::LinAlg::Matrix<3, 3, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&, double, double) const;

template void Discret::Elements::Beam3k::calc_lin_v_thetaperp_moment<2>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 6 * 2 + BEAM3K_COLLOCATION_POINTS,
        double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, double,
    const Core::LinAlg::Matrix<3, 3, double>&) const;

template void Discret::Elements::Beam3k::calc_lin_v_thetapar_moment<2>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 6 * 2 + BEAM3K_COLLOCATION_POINTS,
        double>& lin_v_thetapar_moment,
    Core::LinAlg::Matrix<1, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>& L,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const Core::LinAlg::Matrix<3, 1, double>& g_1, double abs_r_s,
    const Core::LinAlg::Matrix<3, 1, double>& moment) const;

template void Discret::Elements::Beam3k::calc_lin_v_thetaperp_s_moment<2>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 6 * 2 + BEAM3K_COLLOCATION_POINTS,
        double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&, double,
    const Core::LinAlg::Matrix<3, 3, double>&) const;

template void Discret::Elements::Beam3k::calc_lin_v_thetapar_s_moment<2>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 6 * 2 + BEAM3K_COLLOCATION_POINTS,
        double>& lin_v_thetapar_s_moment,
    Core::LinAlg::Matrix<1, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>& L,
    Core::LinAlg::Matrix<1, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>& L_s,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>& N_ss,
    const Core::LinAlg::Matrix<3, 1, double>& g_1, const Core::LinAlg::Matrix<3, 1, double>& g_1_s,
    const Core::LinAlg::Matrix<3, 1, double>& r_s, const Core::LinAlg::Matrix<3, 1, double>& r_ss,
    double abs_r_s, const Core::LinAlg::Matrix<3, 1, double>& moment) const;

FOUR_C_NAMESPACE_CLOSE

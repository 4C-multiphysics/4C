// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beam3_euler_bernoulli.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_geometry_periodic_boundingbox.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::Beam3ebType Discret::Elements::Beam3ebType::instance_;

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::Beam3ebType& Discret::Elements::Beam3ebType::instance() { return instance_; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::Beam3ebType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Beam3eb* object = new Discret::Elements::Beam3eb(-1, -1);
  object->unpack(buffer);
  return object;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Beam3ebType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "BEAM3EB")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Beam3eb>(id, owner);
    return ele;
  }
  return nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Beam3ebType::create(
    const int id, const int owner)
{
  return std::make_shared<Beam3eb>(id, owner);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3ebType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 6;  // 3 translations, 3 tangent DOFs per node
  dimns = 5;  // 3 translations + 2 rotations
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SerialDenseMatrix Discret::Elements::Beam3ebType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  if (numdof != 6)
    FOUR_C_THROW(
        "The computation of the euler-bernoulli beam nullspace in three dimensions requires six"
        "DOFs per node, however the current node carries {} DOFs.",
        numdof);

  if (dimnsp != 5)
    FOUR_C_THROW(
        "The computation of the euler-bernoulli beam nullspace in three dimensions requires five"
        " nullspace vectors per node, however the current node carries {} vectors.",
        dimnsp);

  constexpr std::size_t spacedim = 3;

  // getting coordinates of current node
  const auto& x = node.x();

  // getting pointer at current element
  const auto* beam3eb = dynamic_cast<const Discret::Elements::Beam3eb*>(node.elements()[0]);
  if (!beam3eb) FOUR_C_THROW("Cannot cast to Beam3eb");

  // Compute tangent vector with unit length from nodal coordinates.
  // Note: Tangent vector is the same at both nodes due to straight initial configuration.
  Core::LinAlg::Matrix<spacedim, 1> tangent(Core::LinAlg::Initialization::zero);
  {
    const Core::Nodes::Node* firstnode = beam3eb->nodes()[0];
    const Core::Nodes::Node* secondnode = beam3eb->nodes()[1];
    const auto& xfirst = firstnode->x();
    const auto& xsecond = secondnode->x();

    for (std::size_t dim = 0; dim < spacedim; ++dim) tangent(dim) = xsecond[dim] - xfirst[dim];
    tangent.scale(1.0 / tangent.norm2());
  }

  // Form a Cartesian basis
  std::array<Core::LinAlg::Matrix<spacedim, 1>, spacedim> basis;
  Core::LinAlg::Matrix<spacedim, 1> e1(Core::LinAlg::Initialization::zero);
  e1(0) = 1.0;
  Core::LinAlg::Matrix<spacedim, 1> e2(Core::LinAlg::Initialization::zero);
  e2(1) = 1.0;
  Core::LinAlg::Matrix<spacedim, 1> e3(Core::LinAlg::Initialization::zero);
  e3(2) = 1.0;
  basis[0] = e1;
  basis[1] = e2;
  basis[2] = e3;

  // Find basis vector that is the least parallel to the tangent vector
  std::size_t baseVecIndexWithMindDotProduct = 0;
  {
    double dotProduct = tangent.dot(basis[0]);
    double minDotProduct = dotProduct;
    // First basis vector is already done. Start looping at second basis vector.
    for (std::size_t i = 1; i < spacedim; ++i)
    {
      dotProduct = tangent.dot(basis[i]);
      if (dotProduct < minDotProduct)
      {
        minDotProduct = dotProduct;
        baseVecIndexWithMindDotProduct = i;
      }
    }
  }

  // Compute two vectors orthogonal to the tangent vector
  Core::LinAlg::Matrix<spacedim, 1> someVector = basis[baseVecIndexWithMindDotProduct];
  Core::LinAlg::Matrix<spacedim, 1> omegaOne, omegaTwo;
  omegaOne.cross_product(tangent, someVector);
  omegaTwo.cross_product(tangent, omegaOne);

  if (std::abs(omegaOne.dot(tangent)) > 1.0e-12)
    FOUR_C_THROW("omegaOne not orthogonal to tangent vector.");
  if (std::abs(omegaTwo.dot(tangent)) > 1.0e-12)
    FOUR_C_THROW("omegaTwo not orthogonal to tangent vector.");

  Core::LinAlg::Matrix<3, 1> nodeCoords(Core::LinAlg::Initialization::zero);
  for (std::size_t dim = 0; dim < 3; ++dim) nodeCoords(dim) = x[dim] - x0[dim];

  // Compute rotations in displacement DOFs
  Core::LinAlg::Matrix<spacedim, 1> rotOne(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<spacedim, 1> rotTwo(Core::LinAlg::Initialization::zero);
  rotOne.cross_product(omegaOne, nodeCoords);
  rotTwo.cross_product(omegaTwo, nodeCoords);

  // Compute rotations in tangent DOFs
  Core::LinAlg::Matrix<spacedim, 1> rotTangOne(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<spacedim, 1> rotTangTwo(Core::LinAlg::Initialization::zero);
  rotTangOne.cross_product(omegaOne, tangent);
  rotTangTwo.cross_product(omegaTwo, tangent);

  Core::LinAlg::SerialDenseMatrix nullspace(numdof, dimnsp);
  // x-modes
  nullspace(0, 0) = 1.0;
  nullspace(0, 1) = 0.0;
  nullspace(0, 2) = 0.0;
  nullspace(0, 3) = rotOne(0);
  nullspace(0, 4) = rotTwo(0);
  // y-modes
  nullspace(1, 0) = 0.0;
  nullspace(1, 1) = 1.0;
  nullspace(1, 2) = 0.0;
  nullspace(1, 3) = rotOne(1);
  nullspace(1, 4) = rotTwo(1);
  // z-modes
  nullspace(2, 0) = 0.0;
  nullspace(2, 1) = 0.0;
  nullspace(2, 2) = 1.0;
  nullspace(2, 3) = rotOne(2);
  nullspace(2, 4) = rotTwo(2);
  // dx-modes
  nullspace(3, 0) = 0.0;
  nullspace(3, 1) = 0.0;
  nullspace(3, 2) = 0.0;
  nullspace(3, 3) = rotTangOne(0);
  nullspace(3, 4) = rotTangTwo(0);
  // dy-modes
  nullspace(4, 0) = 0.0;
  nullspace(4, 1) = 0.0;
  nullspace(4, 2) = 0.0;
  nullspace(4, 3) = rotTangOne(1);
  nullspace(4, 4) = rotTangTwo(1);
  // dz-modes
  nullspace(5, 0) = 0.0;
  nullspace(5, 1) = 0.0;
  nullspace(5, 2) = 0.0;
  nullspace(5, 3) = rotTangOne(2);
  nullspace(5, 4) = rotTangTwo(2);

  return nullspace;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3ebType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["BEAM3EB"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::line2] = all_of({
      parameter<int>("MAT"),
  });
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Discret::Elements::Beam3ebType::initialize(Core::FE::Discretization& dis)
{
  // setting up geometric variables for beam3eb elements
  for (int num = 0; num < dis.num_my_col_elements(); ++num)
  {
    // in case that current element is not a beam3eb element there is nothing to do and we go back
    // to the head of the loop
    if (dis.l_col_element(num)->element_type() != *this) continue;

    // if we get so far current element is a beam3eb element and  we get a pointer at it
    Discret::Elements::Beam3eb* currele =
        dynamic_cast<Discret::Elements::Beam3eb*>(dis.l_col_element(num));
    if (!currele) FOUR_C_THROW("cast to Beam3eb* failed");

    // reference node position
    std::vector<double> xrefe;

    const int numNnodes = currele->num_node();

    // resize xrefe for the number of coordinates we need to store
    xrefe.resize(3 * numNnodes);

    // the next section is needed in case of periodic boundary conditions and a shifted
    // configuration (i.e. elements cut by the periodic boundary) in the input file
    Core::Geo::MeshFree::BoundingBox periodic_boundingbox;
    periodic_boundingbox.init(
        Global::Problem::instance()->binning_strategy_params());  // no setup() call needed here

    std::vector<double> disp_shift;
    int numdof = currele->num_dof_per_node(*(currele->nodes()[0]));
    disp_shift.resize(numdof * numNnodes);
    for (unsigned int i = 0; i < disp_shift.size(); ++i) disp_shift[i] = 0.0;
    if (periodic_boundingbox.have_pbc())
      currele->un_shift_node_position(disp_shift, periodic_boundingbox);

    // getting element's nodal coordinates and treating them as reference configuration
    if (currele->nodes()[0] == nullptr || currele->nodes()[1] == nullptr)
      FOUR_C_THROW("Cannot get nodes in order to compute reference configuration'");
    else
    {
      constexpr int numDim = 3;
      for (int node = 0; node < numNnodes; ++node)
      {
        for (int dof = 0; dof < numDim; ++dof)
        {
          xrefe[node * 3 + dof] =
              currele->nodes()[node]->x()[dof] + disp_shift[node * numdof + dof];
        }
      }
    }

    currele->set_up_reference_geometry(xrefe);
  }

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::Beam3eb::Beam3eb(int id, int owner)
    : Discret::Elements::Beam3Base(id, owner),
      isinit_(false),
      jacobi_(0.0),
      firstcall_(true),
      ekin_(0.0),
      eint_(0.0),
      l_(Core::LinAlg::Matrix<3, 1>(Core::LinAlg::Initialization::zero)),
      p_(Core::LinAlg::Matrix<3, 1>(Core::LinAlg::Initialization::zero)),
      t0_(Core::LinAlg::Matrix<3, 2>(Core::LinAlg::Initialization::zero)),
      t_(Core::LinAlg::Matrix<3, 2>(Core::LinAlg::Initialization::zero)),
      kappa_max_(0.0),
      epsilon_max_(0.0),
      axial_strain_gp_(0),
      curvature_gp_(0),
      axial_force_gp_(0),
      bending_moment_gp_(0)
{
#if defined(INEXTENSIBLE)
  if (ANSVALUES != 3 or NODALDOFS != 2)
    FOUR_C_THROW(
        "Flag INEXTENSIBLE only possible in combination with ANSVALUES=3 and NODALDOFS=2!");
#endif
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::Beam3eb::Beam3eb(const Discret::Elements::Beam3eb& old)
    : Discret::Elements::Beam3Base(old),
      isinit_(old.isinit_),
      jacobi_(old.jacobi_),
      ekin_(old.ekin_),
      eint_(old.eint_),
      l_(old.l_),
      p_(old.p_),
      t0_(old.t0_),
      t_(old.t_),
      kappa_max_(old.kappa_max_),
      epsilon_max_(old.epsilon_max_),
      axial_strain_gp_(old.axial_strain_gp_),
      curvature_gp_(old.curvature_gp_),
      axial_force_gp_(old.axial_force_gp_),
      bending_moment_gp_(old.bending_moment_gp_)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Beam3eb::clone() const
{
  Discret::Elements::Beam3eb* newelement = new Discret::Elements::Beam3eb(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3eb::print(std::ostream& os) const
{
  os << "beam3eb ";
  Element::print(os);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Beam3eb::shape() const { return Core::FE::CellType::line2; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3eb::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Beam3Base::pack(data);

  // add all class variables
  add_to_pack(data, jacobi_);
  add_to_pack(data, isinit_);
  add_to_pack(data, ekin_);
  add_to_pack(data, eint_);
  add_to_pack(data, Tref_);
  add_to_pack(data, l_);
  add_to_pack(data, p_);
  add_to_pack(data, t0_);
  add_to_pack(data, t_);
  add_to_pack(data, kappa_max_);
  add_to_pack(data, epsilon_max_);
  add_to_pack(data, axial_strain_gp_);
  add_to_pack(data, curvature_gp_);
  add_to_pack(data, axial_force_gp_);
  add_to_pack(data, bending_moment_gp_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3eb::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Beam3Base::unpack(buffer);

  // extract all class variables of beam3 element
  extract_from_pack(buffer, jacobi_);
  extract_from_pack(buffer, isinit_);
  extract_from_pack(buffer, ekin_);
  extract_from_pack(buffer, eint_);
  extract_from_pack(buffer, Tref_);
  extract_from_pack(buffer, l_);
  extract_from_pack(buffer, p_);
  extract_from_pack(buffer, t0_);
  extract_from_pack(buffer, t_);
  extract_from_pack(buffer, kappa_max_);
  extract_from_pack(buffer, epsilon_max_);
  extract_from_pack(buffer, axial_strain_gp_);
  extract_from_pack(buffer, curvature_gp_);
  extract_from_pack(buffer, axial_force_gp_);
  extract_from_pack(buffer, bending_moment_gp_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Beam3eb::lines()
{
  return {Core::Utils::shared_ptr_from_ref(*this)};
}


/*----------------------------------------------------------------------*
 | sets up geometric data from current nodal position as reference
 | position; this method can be used by the register class or when ever
 | a new beam element is generated for which some reference configuration
 | has to be stored; prerequisite for applying this method is that the
 | element nodes are already known
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3eb::set_up_reference_geometry(
    const std::vector<double>& xrefe, const bool secondinit)
{
  /*this method initializes geometric variables of the element; the initialization can usually be
   *applied to elements only once; therefore after the first initialization the flag isinit is set
   *to true and from then on this method does not take any action when called again unless it is
   *called on purpose with the additional parameter secondinit. If this parameter is passed into the
   *method and is true the element is initialized another time with respective xrefe and rotrefe;
   *note: the isinit_ flag is important for avoiding reinitialization upon restart. However, it
   *should be possible to conduct a second initialization in principle (e.g. for periodic boundary
   *conditions*/

  const int nnode = 2;

  if (!isinit_ || secondinit)
  {
    isinit_ = true;

    // Get DiscretizationType
    Core::FE::CellType distype = shape();

    // Get integrationpoints for exact integration
    Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(mygaussruleeb);

    Tref_.resize(gausspoints.nquad);

    // assure correct size of strain and stress resultant class variables and fill them
    // with zeros (by definition, the reference configuration is undeformed and stress-free)
    axial_strain_gp_.resize(gausspoints.nquad);
    std::fill(axial_strain_gp_.begin(), axial_strain_gp_.end(), 0.0);

    curvature_gp_.resize(gausspoints.nquad);
    std::fill(curvature_gp_.begin(), curvature_gp_.end(), 0.0);

    axial_force_gp_.resize(gausspoints.nquad);
    std::fill(axial_force_gp_.begin(), axial_force_gp_.end(), 0.0);

    bending_moment_gp_.resize(gausspoints.nquad);
    std::fill(bending_moment_gp_.begin(), bending_moment_gp_.end(), 0.0);


    // create Matrix for the derivatives of the shapefunctions at the GP
    Core::LinAlg::Matrix<1, nnode> shapefuncderiv;

    // Loop through all GPs and compute jacobi at the GPs
    for (int numgp = 0; numgp < gausspoints.nquad; numgp++)
    {
      // Get position xi of GP
      const double xi = gausspoints.qxg[numgp][0];

      // Get derivatives of shapefunctions at GP --> for simplicity here are Lagrange polynomials
      // instead of Hermite polynomials used to calculate the reference geometry. Since the
      // reference geometry for this beam element must always be a straight line there is no
      // difference between theses to types of interpolation functions.
      Core::FE::shape_function_1d_deriv1(shapefuncderiv, xi, distype);

      Tref_[numgp].clear();

      // calculate vector dxdxi
      for (int node = 0; node < nnode; node++)
      {
        for (int dof = 0; dof < 3; dof++)
        {
          Tref_[numgp](dof) += shapefuncderiv(node) * xrefe[3 * node + dof];
        }  // for(int dof=0; dof<3 ; dof++)
      }  // for(int node=0; node<nnode; node++)

      // Store length factor for every GP
      // note: the length factor jacobi replaces the determinant and refers to the reference
      // configuration by definition
      jacobi_ = Tref_[numgp].norm2();

      Tref_[numgp].scale(1 / jacobi_);
    }

    // compute tangent at each node
    double norm2;

    Tref_.resize(nnode);
#if NODALDOFS == 3
    Kref_.resize(gausspoints.nquad);
#endif

    for (int node = 0; node < nnode; node++)
    {
      Tref_[node].clear();
#if NODALDOFS == 3
      Kref_[node].clear();
#endif
      for (int dof = 0; dof < 3; dof++)
      {
        Tref_[node](dof) = xrefe[3 + dof] - xrefe[dof];
      }
      norm2 = Tref_[node].norm2();
      Tref_[node].scale(1 / norm2);

      for (int i = 0; i < 3; i++) t0_(i, node) = Tref_[node](i);
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
std::vector<Core::LinAlg::Matrix<3, 1>> Discret::Elements::Beam3eb::tref() const { return Tref_; }

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
double Discret::Elements::Beam3eb::jacobi() const { return jacobi_; }

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3eb::get_pos_at_xi(
    Core::LinAlg::Matrix<3, 1>& pos, const double& xi, const std::vector<double>& disp) const
{
  if (disp.size() != 12)
    FOUR_C_THROW(
        "size mismatch: expected 12 values for element displacement vector "
        "and got {}",
        disp.size());

  // add reference positions and tangents => total Lagrangean state vector
  Core::LinAlg::Matrix<12, 1> disp_totlag(Core::LinAlg::Initialization::zero);
  update_disp_totlag<2, 6>(disp, disp_totlag);

  Beam3Base::get_pos_at_xi<2, 2, double>(pos, xi, disp_totlag);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3eb::get_triad_at_xi(
    Core::LinAlg::Matrix<3, 3>& triad, const double& xi, const std::vector<double>& disp) const
{
  if (disp.size() != 12)
    FOUR_C_THROW(
        "size mismatch: expected 12 values for element displacement vector "
        "and got {}",
        disp.size());

  // add reference positions and tangents => total Lagrangean state vector
  Core::LinAlg::Matrix<12, 1> disp_totlag(Core::LinAlg::Initialization::zero);
  update_disp_totlag<2, 6>(disp, disp_totlag);

  triad.clear();

  /* note: this beam formulation (Beam3eb = torsion-free, isotropic Kirchhoff beam)
   *       does not need to track material triads and therefore can not provide it here;
   *       instead, we return the unit tangent vector as first base vector; both are
   *       identical in the case of Kirchhoff beams (shear-free);
   *
   * Todo @grill: what to do with second and third base vector?
   *
   */

  FOUR_C_THROW(
      "\nBeam3eb::GetTriadAtXi(): by definition, this element can not return "
      "a full triad; think about replacing it by GetTangentAtXi or another solution.");
}

FOUR_C_NAMESPACE_CLOSE

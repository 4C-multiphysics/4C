// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_link_beam3_reissner_line2_pinjointed.hpp"

#include "4C_beam3_reissner.hpp"
#include "4C_beaminteraction_link.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>
FOUR_C_NAMESPACE_OPEN


BeamInteraction::BeamLinkBeam3rLine2PinJointedType
    BeamInteraction::BeamLinkBeam3rLine2PinJointedType::instance_;


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::Communication::ParObject* BeamInteraction::BeamLinkBeam3rLine2PinJointedType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  BeamInteraction::BeamLinkBeam3rLine2PinJointed* my_beam3rline2 =
      new BeamInteraction::BeamLinkBeam3rLine2PinJointed();
  my_beam3rline2->unpack(buffer);
  return my_beam3rline2;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::BeamLinkBeam3rLine2PinJointed::BeamLinkBeam3rLine2PinJointed()
    : BeamLinkPinJointed(),
      triad_(Core::LinAlg::Initialization::zero),
      linkele_(nullptr),
      bspotforces_(2, Core::LinAlg::SerialDenseVector(3, true))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
BeamInteraction::BeamLinkBeam3rLine2PinJointed::BeamLinkBeam3rLine2PinJointed(
    const BeamInteraction::BeamLinkBeam3rLine2PinJointed& old)
    : BeamInteraction::BeamLinkPinJointed(old),
      triad_(old.triad_),
      bspotforces_(2, Core::LinAlg::SerialDenseVector(3, true))
{
  if (linkele_ != nullptr)
    linkele_ = std::dynamic_pointer_cast<Discret::Elements::Beam3r>(
        std::shared_ptr<Core::Elements::Element>(old.linkele_->clone()));
  else
    linkele_ = nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<BeamInteraction::BeamLink> BeamInteraction::BeamLinkBeam3rLine2PinJointed::clone()
    const
{
  std::shared_ptr<BeamInteraction::BeamLinkBeam3rLine2PinJointed> newlinker =
      std::make_shared<BeamInteraction::BeamLinkBeam3rLine2PinJointed>(*this);
  return newlinker;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkBeam3rLine2PinJointed::init(int id,
    const std::vector<std::pair<int, int>>& eleids,
    const std::vector<Core::LinAlg::Matrix<3, 1>>& initpos,
    const std::vector<Core::LinAlg::Matrix<3, 3>>& inittriad,
    Inpar::BeamInteraction::CrosslinkerType linkertype, double timelinkwasset)
{
  issetup_ = false;

  BeamLinkPinJointed::init(id, eleids, initpos, inittriad, linkertype, timelinkwasset);

  // *** initialization of the two triads of the connecting element ***
  /* they are determined such that:
   * - the first base vector points in the direction of the distance vector
   *   of the two connection sites; (will be axis of connecting element)
   * - second and third base vector are arbitrarily constructed from cross-product
   *   of first base vector with either first or second base vector of global
   *   coordinate system; this avoids any singularities */
  Core::LinAlg::Matrix<3, 3> linkeletriad(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> distvec(Core::LinAlg::Initialization::zero);

  distvec.update(1.0, get_bind_spot_pos2(), -1.0, get_bind_spot_pos1());

  // feasibility check regarding coinciding connection sites
  if (distvec.norm2() < 1e-12)
  {
    std::cout << "\nBeamLinkPinJointed initialized with ...";
    std::cout << "\ninitbspotpos1 =";
    initpos[0].print(std::cout);
    std::cout << "\ninitbspotpos2 =";
    initpos[1].print(std::cout);

    FOUR_C_THROW(
        "Initialization of BeamLinkPinJointed failed because the two given binding "
        "spot positions are almost identical, i.e. extremely short linker!");
  }

  // FIRST base vector
  distvec.scale(1.0 / distvec.norm2());

  std::copy(distvec.data(), distvec.data() + 3, &linkeletriad(0, 0));

  // SECOND base vector
  // check included angle of desired crosslinker axis (normalized distvec = first
  // base vector) and (1,0,0), i.e. scalar product which in this case simplifies to
  // first component of distvec
  Core::LinAlg::Matrix<3, 1> unit_vector_global_x(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> unit_vector_global_y(Core::LinAlg::Initialization::zero);
  unit_vector_global_x(0) = 1.0;
  unit_vector_global_y(1) = 1.0;

  const double scalarproduct = distvec(0);

  Core::LinAlg::Matrix<3, 1> second_base_vecor_linkerele(Core::LinAlg::Initialization::zero);

  // is included angle smaller than 45 degrees ? then avoid singularity at angle=0 degrees ...
  if (std::abs(scalarproduct) > 0.5 * std::sqrt(2))
  {
    second_base_vecor_linkerele.cross_product(distvec, unit_vector_global_y);
  }
  else
  {
    second_base_vecor_linkerele.cross_product(distvec, unit_vector_global_x);
  }

  // feasibility check
  if (second_base_vecor_linkerele.norm2() < 1e-12)
  {
    std::cout << "\nBeamLinkPinJointed initialized with ...";
    std::cout << "\ninitbspotpos1 =";
    initpos[0].print(std::cout);
    std::cout << "\ninitbspotpos2 =";
    initpos[1].print(std::cout);

    std::cout << "\ndistvec = ";
    distvec.print(std::cout);
    std::cout << "\nsecond_base_vecor_linkerele = ";
    second_base_vecor_linkerele.print(std::cout);

    FOUR_C_THROW(
        "Initialization of BeamLinkPinJointed failed because the second base vector of the"
        "linker element's triad has almost length zero!");
  }
  else
  {
    second_base_vecor_linkerele.scale(1.0 / second_base_vecor_linkerele.norm2());
  }


  // THIRD base vector to complete orthonormal triad
  Core::LinAlg::Matrix<3, 1> third_base_vecor_linkerele(Core::LinAlg::Initialization::zero);
  third_base_vecor_linkerele.cross_product(distvec, second_base_vecor_linkerele);

  // feasibility check
  if (std::abs(third_base_vecor_linkerele.norm2() - 1.0) > 1e-12)
  {
    std::cout << "\nBeamLinkPinJointed initialized with ...";
    std::cout << "\ninitbspotpos1 =";
    initpos[0].print(std::cout);
    std::cout << "\ninitbspotpos2 =";
    initpos[1].print(std::cout);

    std::cout << "\ndistvec = ";
    distvec.print(std::cout);
    std::cout << "\nsecond_base_vecor_linkerele = ";
    second_base_vecor_linkerele.print(std::cout);
    std::cout << "\nthird_base_vecor_linkerele = ";
    third_base_vecor_linkerele.print(std::cout);

    FOUR_C_THROW(
        "Initialization of BeamLinkRigidJointed failed because the third base vector of the"
        "linker element's triad is no unit vector!");
  }


  /* store the initial triads as quaternions in class variables for the subsequent
   * use in setup of reference configuration of the connecting element */
  std::copy(second_base_vecor_linkerele.data(), second_base_vecor_linkerele.data() + 3,
      &linkeletriad(0, 1));
  std::copy(third_base_vecor_linkerele.data(), third_base_vecor_linkerele.data() + 3,
      &linkeletriad(0, 2));

  Core::LargeRotations::triadtoquaternion(linkeletriad, triad_);

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkBeam3rLine2PinJointed::setup(const int matnum)
{
  check_init();

  // call setup of base class first
  BeamLinkPinJointed::setup(matnum);

  /* the idea is to use a beam element as auxiliary object that provides us with a
   * response force (and moment) depending on the position and orientation of the
   * two material cross-sections (binding spots) it is connected to;
   *
   * note: the element instance created in this way can only be used in a limited way
   *       because it is not embedded in a discretization. For example,
   *       Nodes() and other methods are not functional because the
   *       pointers to nodes are not set. Same for reference position of nodes via X() ...
   *
   *       We really only use it as a calculation routine for a sophisticated
   *       (displacement-reaction force) relation here! */
  linkele_ = std::make_shared<Discret::Elements::Beam3r>(-1, 0);

  // set material
  linkele_->set_material(0, Mat::factory(matnum));

  // Todo @grill: safety check for proper material type (done on element anyway, but do it here as
  // well)?!

  linkele_->set_centerline_hermite(false);

  // set dummy node Ids, in order to make NumNodes() method of element return the correct number of
  // nodes
  constexpr std::array nodeids = {-1, -1};
  linkele_->set_node_ids(2, nodeids.data());

  // the triads at the two connection sites are chosen identical initially, so we only use the first
  // one
  Core::LinAlg::Matrix<3, 1> linkelerotvec(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::quaterniontoangle(triad_, linkelerotvec);

  std::vector<double> refpos(6, 0.0);
  std::vector<double> refrotvec(6, 0.0);

  for (unsigned int i = 0; i < 3; ++i)
  {
    refpos[i] = get_bind_spot_pos1()(i);
    refpos[3 + i] = get_bind_spot_pos2()(i);

    refrotvec[i] = linkelerotvec(i);
    refrotvec[3 + i] = linkelerotvec(i);
  }

  linkele_->set_up_reference_geometry<2, 2, 1>(refpos, refrotvec);

  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::BeamLinkBeam3rLine2PinJointed::pack(
    Core::Communication::PackBuffer& data) const
{
  check_init_setup();



  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class
  BeamLinkPinJointed::pack(data);

  // pack linker element
  if (linkele_ != nullptr) linkele_->pack(data);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::BeamLinkBeam3rLine2PinJointed::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class
  BeamLinkPinJointed::unpack(buffer);

  // Unpack data of sub material (these lines are copied from element.cpp)
  std::vector<char> dataele;
  extract_from_pack(buffer, dataele);
  if (dataele.size() > 0)
  {
    Core::Communication::UnpackBuffer dataele_buffer(dataele);
    Core::Communication::ParObject* object =
        Core::Communication::factory(dataele_buffer);  // Unpack is done here
    Discret::Elements::Beam3r* linkele = dynamic_cast<Discret::Elements::Beam3r*>(object);
    if (linkele == nullptr)
      FOUR_C_THROW("failed to unpack Beam3r object within BeamLinkBeam3rLine2PinJointed");
    linkele_ = std::shared_ptr<Discret::Elements::Beam3r>(linkele);
  }
  else
    linkele_ = nullptr;

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BeamInteraction::BeamLinkBeam3rLine2PinJointed::evaluate_force(
    Core::LinAlg::SerialDenseVector& forcevec1, Core::LinAlg::SerialDenseVector& forcevec2)
{
  check_init_setup();

  Core::LinAlg::Matrix<6, 1, double> disp_totlag_centerline;
  std::vector<Core::LinAlg::Matrix<4, 1, double>> Qnode;

  fill_state_variables_for_element_evaluation(disp_totlag_centerline, Qnode);

  Core::LinAlg::SerialDenseVector force(12, true);

  linkele_->calc_internal_and_inertia_forces_and_stiff<2, 2, 1>(
      disp_totlag_centerline, Qnode, nullptr, nullptr, &force, nullptr);

  // Todo maybe we can avoid this copy by setting up 'force' as a view on the
  //      two separate force vectors ?
  std::copy(&force(0), &force(0) + 3, &forcevec1(0));
  std::copy(&force(0) + 6, &force(0) + 9, &forcevec2(0));

  bspotforces_[0] = forcevec1;
  bspotforces_[1] = forcevec2;

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BeamInteraction::BeamLinkBeam3rLine2PinJointed::evaluate_stiff(
    Core::LinAlg::SerialDenseMatrix& stiffmat11, Core::LinAlg::SerialDenseMatrix& stiffmat12,
    Core::LinAlg::SerialDenseMatrix& stiffmat21, Core::LinAlg::SerialDenseMatrix& stiffmat22)
{
  check_init_setup();

  Core::LinAlg::Matrix<6, 1, double> disp_totlag_centerline;
  std::vector<Core::LinAlg::Matrix<4, 1, double>> Qnode;

  fill_state_variables_for_element_evaluation(disp_totlag_centerline, Qnode);

  Core::LinAlg::SerialDenseMatrix stiffmat(12, 12, true);

  linkele_->calc_internal_and_inertia_forces_and_stiff<2, 2, 1>(
      disp_totlag_centerline, Qnode, &stiffmat, nullptr, nullptr, nullptr);

  // Todo the linearization is incomplete yet. fix this or delete related code and
  // resort to truss linker element
  FOUR_C_THROW(
      "we miss stiffness contributions from rotation of the nodal triads here! "
      "implement the transformation matrices that describe the dependency of triad "
      "rotation on current position of binding spots, i.e. nodal positions and "
      "tangents!");

  // Todo can we use std::copy here or even set up 'stiffmat' as a view on the
  //      four individual sub-matrices ?
  for (unsigned int i = 0; i < 3; ++i)
  {
    for (unsigned int j = 0; j < 3; ++j)
    {
      stiffmat11(i, j) = stiffmat(i, j);
      stiffmat12(i, j) = stiffmat(i, 6 + j);
      stiffmat21(i, j) = stiffmat(6 + i, j);
      stiffmat22(i, j) = stiffmat(6 + i, 6 + j);
    }
  }

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BeamInteraction::BeamLinkBeam3rLine2PinJointed::evaluate_force_stiff(
    Core::LinAlg::SerialDenseVector& forcevec1, Core::LinAlg::SerialDenseVector& forcevec2,
    Core::LinAlg::SerialDenseMatrix& stiffmat11, Core::LinAlg::SerialDenseMatrix& stiffmat12,
    Core::LinAlg::SerialDenseMatrix& stiffmat21, Core::LinAlg::SerialDenseMatrix& stiffmat22)
{
  check_init_setup();

  Core::LinAlg::Matrix<6, 1, double> disp_totlag_centerline;
  std::vector<Core::LinAlg::Matrix<4, 1, double>> Qnode;

  fill_state_variables_for_element_evaluation(disp_totlag_centerline, Qnode);

  Core::LinAlg::SerialDenseVector force(12, true);
  Core::LinAlg::SerialDenseMatrix stiffmat(12, 12, true);

  linkele_->calc_internal_and_inertia_forces_and_stiff<2, 2, 1>(
      disp_totlag_centerline, Qnode, &stiffmat, nullptr, &force, nullptr);

  std::copy(&force(0), &force(0) + 3, &forcevec1(0));
  std::copy(&force(0) + 6, &force(0) + 9, &forcevec2(0));

  // Todo the linearization is incomplete yet. fix this or delete related code and
  // resort to truss linker element
  FOUR_C_THROW(
      "we miss stiffness contributions from rotation of the nodal triads here! "
      "implement the transformation matrices that describe the dependency of triad "
      "rotation on current position of binding spots, i.e. nodal positions and "
      "tangents!");

  for (unsigned int i = 0; i < 3; ++i)
  {
    for (unsigned int j = 0; j < 3; ++j)
    {
      stiffmat11(i, j) = stiffmat(i, j);
      stiffmat12(i, j) = stiffmat(i, 6 + j);
      stiffmat21(i, j) = stiffmat(6 + i, j);
      stiffmat22(i, j) = stiffmat(6 + i, 6 + j);
    }
  }

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkBeam3rLine2PinJointed::reset_state(
    std::vector<Core::LinAlg::Matrix<3, 1>>& bspotpos,
    std::vector<Core::LinAlg::Matrix<3, 3>>& bspottriad)
{
  check_init_setup();

  BeamLinkPinJointed::reset_state(bspotpos, bspottriad);

  // *** re-initialization of the two triads of the connecting element ***

  /* the idea is that for this pin jointed linker element the underlying linear
   * Reissner beam element is used as a truss model only; that means, it only
   * reacts with an axial force - no moments and no transverse forces;
   * this shall be achieved by rotating the nodal triads according to the beam
   * axis in every new configuration given here from outside;
   * to keep the Reisner element shear-, bending- and torsion-free, we use the
   * same strategy to determine the nodal triads as for initialization of any
   * linker (see init() ) */



  /* they are determined such that:
   * - the first base vector points in the direction of the distance vector
   *   of the two connection sites; (will be axis of connecting element)
   * - second and third base vector are arbitrarily constructed from cross-product
   *   of first base vector with either first or second base vector of global
   *   coordinate system; this avoids any singularities */
  Core::LinAlg::Matrix<3, 3> linkeletriad(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> distvec(Core::LinAlg::Initialization::zero);

  distvec.update(1.0, get_bind_spot_pos2(), -1.0, get_bind_spot_pos1());

  // feasibility check regarding coinciding connection sites
  if (distvec.norm2() < 1e-12)
  {
    std::cout << "\nBeamLinkRigidJointed initialized with ...";
    std::cout << "\ninitbspotpos1 =";
    get_bind_spot_pos1().print(std::cout);
    std::cout << "\ninitbspotpos2 =";
    get_bind_spot_pos2().print(std::cout);
    std::cout << "\ninitbspottriad1 =";
    get_bind_spot_pos1().print(std::cout);
    std::cout << "\ninitbspottriad2 =";
    get_bind_spot_pos2().print(std::cout);

    FOUR_C_THROW(
        "Initialization of BeamLinkRigidJointed failed because the two given binding "
        "spot positions are almost identical!");
  }

  // first base vector
  distvec.scale(1.0 / distvec.norm2());

  std::copy(distvec.data(), distvec.data() + 3, &linkeletriad(0, 0));


  // second base vector
  // check included angle of desired crosslinker axis (normalized distvec = first
  // base vector) and (1,0,0), i.e. scalar product which in this case simplifies to
  // first component of distvec
  Core::LinAlg::Matrix<3, 1> unit_vector_global_x(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> unit_vector_global_y(Core::LinAlg::Initialization::zero);
  unit_vector_global_x(0) = 1.0;
  unit_vector_global_y(1) = 1.0;

  const double scalarproduct = distvec(0);

  Core::LinAlg::Matrix<3, 1> second_base_vecor_linkerele(Core::LinAlg::Initialization::zero);

  // is included angle smaller than 45 degrees ? then avoid singularity at angle=0 degrees ...
  if (std::abs(scalarproduct) > 0.5 * std::sqrt(2))
  {
    second_base_vecor_linkerele.cross_product(distvec, unit_vector_global_y);
  }
  else
  {
    second_base_vecor_linkerele.cross_product(distvec, unit_vector_global_x);
  }

  // feasibility check
  if (second_base_vecor_linkerele.norm2() < 1e-12)
  {
    std::cout << "\nBeamLinkRigidJointed initialized with ...";
    std::cout << "\ninitbspotpos1 =";
    get_bind_spot_pos1().print(std::cout);
    std::cout << "\ninitbspotpos2 =";
    get_bind_spot_pos2().print(std::cout);
    std::cout << "\ninitbspottriad1 =";
    get_bind_spot_pos1().print(std::cout);
    std::cout << "\ninitbspottriad2 =";
    get_bind_spot_pos2().print(std::cout);

    std::cout << "\ndistvec = ";
    distvec.print(std::cout);
    std::cout << "\nsecond_base_vecor_linkerele = ";
    second_base_vecor_linkerele.print(std::cout);

    FOUR_C_THROW(
        "Initialization of BeamLinkRigidJointed failed because the second base vector of the"
        "linker element's triad has almost length zero!");
  }
  else
  {
    second_base_vecor_linkerele.scale(1.0 / second_base_vecor_linkerele.norm2());
  }


  // third base vector to complete orthonormal triad
  Core::LinAlg::Matrix<3, 1> third_base_vecor_linkerele(Core::LinAlg::Initialization::zero);
  third_base_vecor_linkerele.cross_product(distvec, second_base_vecor_linkerele);

  // feasibility check
  if (std::abs(third_base_vecor_linkerele.norm2() - 1.0) > 1e-12)
  {
    std::cout << "\nBeamLinkRigidJointed initialized with ...";
    std::cout << "\ninitbspotpos1 =";
    get_bind_spot_pos1().print(std::cout);
    std::cout << "\ninitbspotpos2 =";
    get_bind_spot_pos2().print(std::cout);
    std::cout << "\ninitbspottriad1 =";
    get_bind_spot_pos1().print(std::cout);
    std::cout << "\ninitbspottriad2 =";
    get_bind_spot_pos2().print(std::cout);

    std::cout << "\ndistvec = ";
    distvec.print(std::cout);
    std::cout << "\nsecond_base_vecor_linkerele = ";
    second_base_vecor_linkerele.print(std::cout);
    std::cout << "\nthird_base_vecor_linkerele = ";
    third_base_vecor_linkerele.print(std::cout);

    FOUR_C_THROW(
        "Initialization of BeamLinkRigidJointed failed because the third base vector of the"
        "linker element's triad is no unit vector!");
  }

  /* store the initial triads as quaternions in class variables for the subsequent
   * use in setup of reference configuration of the connecting element */
  std::copy(second_base_vecor_linkerele.data(), second_base_vecor_linkerele.data() + 3,
      &linkeletriad(0, 1));
  std::copy(third_base_vecor_linkerele.data(), third_base_vecor_linkerele.data() + 3,
      &linkeletriad(0, 2));

  Core::LargeRotations::triadtoquaternion(linkeletriad, triad_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkBeam3rLine2PinJointed::fill_state_variables_for_element_evaluation(
    Core::LinAlg::Matrix<6, 1, double>& disp_totlag_centerline,
    std::vector<Core::LinAlg::Matrix<4, 1, double>>& Qnode) const
{
  for (unsigned int i = 0; i < 3; ++i)
  {
    disp_totlag_centerline(i) = get_bind_spot_pos1()(i);
    disp_totlag_centerline(3 + i) = get_bind_spot_pos2()(i);
  }

  Qnode.push_back(triad_);
  Qnode.push_back(triad_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkBeam3rLine2PinJointed::get_binding_spot_force(
    int bspotid, Core::LinAlg::SerialDenseVector& bspotforce) const
{
  bspotforce = bspotforces_[bspotid];
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double BeamInteraction::BeamLinkBeam3rLine2PinJointed::get_internal_energy() const
{
  return linkele_->get_internal_energy();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double BeamInteraction::BeamLinkBeam3rLine2PinJointed::get_kinetic_energy() const
{
  return linkele_->get_kinetic_energy();
}

FOUR_C_NAMESPACE_CLOSE

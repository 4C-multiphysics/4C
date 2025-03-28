// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_link_rigidjointed.hpp"

#include "4C_beaminteraction_link.hpp"
#include "4C_beaminteraction_link_beam3_reissner_line2_rigidjointed.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

BeamInteraction::BeamLinkRigidJointedType BeamInteraction::BeamLinkRigidJointedType::instance_;


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::BeamLinkRigidJointed::BeamLinkRigidJointed() {}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
BeamInteraction::BeamLinkRigidJointed::BeamLinkRigidJointed(
    const BeamInteraction::BeamLinkRigidJointed& old)
    : BeamLink(old),
      bspottriad1_(old.bspottriad1_),
      bspottriad2_(old.bspottriad2_),
      lambdarel1_(old.lambdarel1_),
      lambdarel2_(old.lambdarel2_)
{
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkRigidJointed::init(const int id,
    const std::vector<std::pair<int, int>>& eleids,
    const std::vector<Core::LinAlg::Matrix<3, 1>>& initpos,
    const std::vector<Core::LinAlg::Matrix<3, 3>>& inittriad,
    Inpar::BeamInteraction::CrosslinkerType linkertype, double timelinkwasset)
{
  issetup_ = false;

  BeamLink::init(id, eleids, initpos, inittriad, linkertype, timelinkwasset);

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
    std::cout << "\nBeamLinkRigidJointed initialized with ...";
    std::cout << "\ninitbspotpos1 =";
    initpos[0].print(std::cout);
    std::cout << "\ninitbspotpos2 =";
    initpos[1].print(std::cout);
    std::cout << "\ninitbspottriad1 =";
    inittriad[0].print(std::cout);
    std::cout << "\ninitbspottriad2 =";
    inittriad[1].print(std::cout);

    FOUR_C_THROW(
        "Initialization of BeamLinkRigidJointed between element {} and {} failed because the two "
        "given binding "
        "spot positions are almost identical!",
        eleids[0].first, eleids[1].first);
  }

  // first base vector
  distvec.scale(1.0 / distvec.norm2());

  std::copy(distvec.data(), distvec.data() + 3, &linkeletriad(0, 0));


  // second base vector
  // check included angle of desired crosslinker axis (normalized distvec = first
  // base vector) and (1,0,0), i.e. scalar product which in this case simplifies to
  // first component of distvec
  Core::LinAlg::Matrix<3, 1> unit_vector_global_x;
  Core::LinAlg::Matrix<3, 1> unit_vector_global_y;
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
    initpos[0].print(std::cout);
    std::cout << "\ninitbspotpos2 =";
    initpos[1].print(std::cout);
    std::cout << "\ninitbspottriad1 =";
    inittriad[0].print(std::cout);
    std::cout << "\ninitbspottriad2 =";
    inittriad[1].print(std::cout);

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
    initpos[0].print(std::cout);
    std::cout << "\ninitbspotpos2 =";
    initpos[1].print(std::cout);
    std::cout << "\ninitbspottriad1 =";
    inittriad[0].print(std::cout);
    std::cout << "\ninitbspottriad2 =";
    inittriad[1].print(std::cout);

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

  Core::LargeRotations::triadtoquaternion(linkeletriad, bspottriad1_);
  bspottriad2_ = bspottriad1_;

  /* store relative rotation matrix between triads of connecting element and
   * the material triads of the "parent elements"; these remain constant over
   * the entire life of this connection (expressed in material frame!) */
  lambdarel1_.multiply_tn(inittriad[0], linkeletriad);
  lambdarel2_.multiply_tn(inittriad[1], linkeletriad);

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkRigidJointed::setup(const int matnum)
{
  check_init();

  // the flag issetup_ will be set in the derived method!
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::BeamLinkRigidJointed::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  BeamLink::pack(data);

  // bspottriad1_
  add_to_pack(data, bspottriad1_);
  // bspottriad2_
  add_to_pack(data, bspottriad2_);
  // Lambdarel1_
  add_to_pack(data, lambdarel1_);
  // Lambdarel2_
  add_to_pack(data, lambdarel2_);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::BeamLinkRigidJointed::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  BeamLink::unpack(buffer);

  // bspottriad1_
  extract_from_pack(buffer, bspottriad1_);
  // bspottriad2_
  extract_from_pack(buffer, bspottriad2_);
  // Lambdarel1_
  extract_from_pack(buffer, lambdarel1_);
  // Lambdarel2_
  extract_from_pack(buffer, lambdarel2_);



  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkRigidJointed::reset_state(
    std::vector<Core::LinAlg::Matrix<3, 1>>& bspotpos,
    std::vector<Core::LinAlg::Matrix<3, 3>>& bspottriad)
{
  check_init_setup();

  BeamLink::reset_state(bspotpos, bspottriad);

  /* the two orientations, i.e. triads of the linkage element are defined via a
   * constant relative rotation based on the triads at the binding spots of the
   * parent elements.
   * Note: constant rotation in material frame, therefore multiplication from right
   *       side */
  Core::LinAlg::Matrix<3, 3, double> currenttriad(Core::LinAlg::Initialization::zero);
  currenttriad.multiply(bspottriad[0], lambdarel1_);
  Core::LargeRotations::triadtoquaternion<double>(currenttriad, bspottriad1_);

  currenttriad.clear();
  currenttriad.multiply(bspottriad[1], lambdarel2_);
  Core::LargeRotations::triadtoquaternion<double>(currenttriad, bspottriad2_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<BeamInteraction::BeamLinkRigidJointed>
BeamInteraction::BeamLinkRigidJointed::create()
{
  // for now, we always use a 2-noded linear Reissner element
  return std::make_shared<BeamInteraction::BeamLinkBeam3rLine2RigidJointed>();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkRigidJointed::print(std::ostream& out) const
{
  check_init();

  BeamLink::print(out);

  out << "\nbspottriad1_ = ";
  Core::LinAlg::Matrix<3, 3, double> triad;
  Core::LargeRotations::quaterniontotriad(bspottriad1_, triad);
  triad.print(out);
  out << "\nbspottriad2_ = ";
  Core::LargeRotations::quaterniontotriad(bspottriad2_, triad);
  triad.print(out);
  out << "\n";
}

FOUR_C_NAMESPACE_CLOSE

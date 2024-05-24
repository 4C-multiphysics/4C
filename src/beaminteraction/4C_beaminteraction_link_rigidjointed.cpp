/*----------------------------------------------------------------------*/
/*! \file

\brief One beam-to-beam pair (two beam elements) connected by a mechanical link

\level 3

*/
/*----------------------------------------------------------------------*/

#include "4C_beaminteraction_link_rigidjointed.hpp"

#include "4C_beaminteraction_link.hpp"
#include "4C_beaminteraction_link_beam3_reissner_line2_rigidjointed.hpp"
#include "4C_discretization_fem_general_largerotations.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

BEAMINTERACTION::BeamLinkRigidJointedType BEAMINTERACTION::BeamLinkRigidJointedType::instance_;


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BEAMINTERACTION::BeamLinkRigidJointed::BeamLinkRigidJointed()
    : BeamLink(), bspottriad1_(true), bspottriad2_(true), lambdarel1_(true), lambdarel2_(true)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
BEAMINTERACTION::BeamLinkRigidJointed::BeamLinkRigidJointed(
    const BEAMINTERACTION::BeamLinkRigidJointed& old)
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
void BEAMINTERACTION::BeamLinkRigidJointed::Init(const int id,
    const std::vector<std::pair<int, int>>& eleids,
    const std::vector<CORE::LINALG::Matrix<3, 1>>& initpos,
    const std::vector<CORE::LINALG::Matrix<3, 3>>& inittriad,
    INPAR::BEAMINTERACTION::CrosslinkerType linkertype, double timelinkwasset)
{
  issetup_ = false;

  BeamLink::Init(id, eleids, initpos, inittriad, linkertype, timelinkwasset);

  // *** initialization of the two triads of the connecting element ***

  /* they are determined such that:
   * - the first base vector points in the direction of the distance vector
   *   of the two connection sites; (will be axis of connecting element)
   * - second and third base vector are arbitrarily constructed from cross-product
   *   of first base vector with either first or second base vector of global
   *   coordinate system; this avoids any singularities */
  CORE::LINALG::Matrix<3, 3> linkeletriad(true);
  CORE::LINALG::Matrix<3, 1> distvec(true);

  distvec.Update(1.0, GetBindSpotPos2(), -1.0, GetBindSpotPos1());

  // feasibility check regarding coinciding connection sites
  if (distvec.Norm2() < 1e-12)
  {
    std::cout << "\nBeamLinkRigidJointed initialized with ...";
    std::cout << "\ninitbspotpos1 =";
    initpos[0].Print(std::cout);
    std::cout << "\ninitbspotpos2 =";
    initpos[1].Print(std::cout);
    std::cout << "\ninitbspottriad1 =";
    inittriad[0].Print(std::cout);
    std::cout << "\ninitbspottriad2 =";
    inittriad[1].Print(std::cout);

    FOUR_C_THROW(
        "Initialization of BeamLinkRigidJointed between element %i and %i failed because the two "
        "given binding "
        "spot positions are almost identical!",
        eleids[0].first, eleids[1].first);
  }

  // first base vector
  distvec.Scale(1.0 / distvec.Norm2());

  std::copy(distvec.A(), distvec.A() + 3, &linkeletriad(0, 0));


  // second base vector
  // check included angle of desired crosslinker axis (normalized distvec = first
  // base vector) and (1,0,0), i.e. scalar product which in this case simplifies to
  // first component of distvec
  CORE::LINALG::Matrix<3, 1> unit_vector_global_x(true), unit_vector_global_y(true);
  unit_vector_global_x(0) = 1.0;
  unit_vector_global_y(1) = 1.0;

  const double scalarproduct = distvec(0);

  CORE::LINALG::Matrix<3, 1> second_base_vecor_linkerele(true);

  // is included angle smaller than 45 degrees ? then avoid singularity at angle=0 degrees ...
  if (std::abs(scalarproduct) > 0.5 * std::sqrt(2))
  {
    second_base_vecor_linkerele.CrossProduct(distvec, unit_vector_global_y);
  }
  else
  {
    second_base_vecor_linkerele.CrossProduct(distvec, unit_vector_global_x);
  }

  // feasibility check
  if (second_base_vecor_linkerele.Norm2() < 1e-12)
  {
    std::cout << "\nBeamLinkRigidJointed initialized with ...";
    std::cout << "\ninitbspotpos1 =";
    initpos[0].Print(std::cout);
    std::cout << "\ninitbspotpos2 =";
    initpos[1].Print(std::cout);
    std::cout << "\ninitbspottriad1 =";
    inittriad[0].Print(std::cout);
    std::cout << "\ninitbspottriad2 =";
    inittriad[1].Print(std::cout);

    std::cout << "\ndistvec = ";
    distvec.Print(std::cout);
    std::cout << "\nsecond_base_vecor_linkerele = ";
    second_base_vecor_linkerele.Print(std::cout);

    FOUR_C_THROW(
        "Initialization of BeamLinkRigidJointed failed because the second base vector of the"
        "linker element's triad has almost length zero!");
  }
  else
  {
    second_base_vecor_linkerele.Scale(1.0 / second_base_vecor_linkerele.Norm2());
  }


  // third base vector to complete orthonormal triad
  CORE::LINALG::Matrix<3, 1> third_base_vecor_linkerele(true);
  third_base_vecor_linkerele.CrossProduct(distvec, second_base_vecor_linkerele);

  // feasibility check
  if (std::abs(third_base_vecor_linkerele.Norm2() - 1.0) > 1e-12)
  {
    std::cout << "\nBeamLinkRigidJointed initialized with ...";
    std::cout << "\ninitbspotpos1 =";
    initpos[0].Print(std::cout);
    std::cout << "\ninitbspotpos2 =";
    initpos[1].Print(std::cout);
    std::cout << "\ninitbspottriad1 =";
    inittriad[0].Print(std::cout);
    std::cout << "\ninitbspottriad2 =";
    inittriad[1].Print(std::cout);

    std::cout << "\ndistvec = ";
    distvec.Print(std::cout);
    std::cout << "\nsecond_base_vecor_linkerele = ";
    second_base_vecor_linkerele.Print(std::cout);
    std::cout << "\nthird_base_vecor_linkerele = ";
    third_base_vecor_linkerele.Print(std::cout);

    FOUR_C_THROW(
        "Initialization of BeamLinkRigidJointed failed because the third base vector of the"
        "linker element's triad is no unit vector!");
  }


  /* store the initial triads as quaternions in class variables for the subsequent
   * use in setup of reference configuration of the connecting element */
  std::copy(
      second_base_vecor_linkerele.A(), second_base_vecor_linkerele.A() + 3, &linkeletriad(0, 1));
  std::copy(
      third_base_vecor_linkerele.A(), third_base_vecor_linkerele.A() + 3, &linkeletriad(0, 2));

  CORE::LARGEROTATIONS::triadtoquaternion(linkeletriad, bspottriad1_);
  bspottriad2_ = bspottriad1_;

  /* store relative rotation matrix between triads of connecting element and
   * the material triads of the "parent elements"; these remain constant over
   * the entire life of this connection (expressed in material frame!) */
  lambdarel1_.MultiplyTN(inittriad[0], linkeletriad);
  lambdarel2_.MultiplyTN(inittriad[1], linkeletriad);

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::BeamLinkRigidJointed::Setup(const int matnum)
{
  check_init();

  // the flag issetup_ will be set in the derived method!
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BEAMINTERACTION::BeamLinkRigidJointed::Pack(CORE::COMM::PackBuffer& data) const
{
  CORE::COMM::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);
  // add base class Element
  BeamLink::Pack(data);

  // bspottriad1_
  AddtoPack(data, bspottriad1_);
  // bspottriad2_
  AddtoPack(data, bspottriad2_);
  // Lambdarel1_
  AddtoPack(data, lambdarel1_);
  // Lambdarel2_
  AddtoPack(data, lambdarel2_);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BEAMINTERACTION::BeamLinkRigidJointed::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  CORE::COMM::ExtractAndAssertId(position, data, UniqueParObjectId());

  // extract base class Element
  std::vector<char> basedata(0);
  ExtractfromPack(position, data, basedata);
  BeamLink::Unpack(basedata);

  // bspottriad1_
  ExtractfromPack(position, data, bspottriad1_);
  // bspottriad2_
  ExtractfromPack(position, data, bspottriad2_);
  // Lambdarel1_
  ExtractfromPack(position, data, lambdarel1_);
  // Lambdarel2_
  ExtractfromPack(position, data, lambdarel2_);

  if (position != data.size())
    FOUR_C_THROW("Mismatch in size of data %d <-> %d", (int)data.size(), position);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::BeamLinkRigidJointed::ResetState(
    std::vector<CORE::LINALG::Matrix<3, 1>>& bspotpos,
    std::vector<CORE::LINALG::Matrix<3, 3>>& bspottriad)
{
  check_init_setup();

  BeamLink::ResetState(bspotpos, bspottriad);

  /* the two orientations, i.e. triads of the linkage element are defined via a
   * constant relative rotation based on the triads at the binding spots of the
   * parent elements.
   * Note: constant rotation in material frame, therefore multiplication from right
   *       side */
  CORE::LINALG::Matrix<3, 3, double> currenttriad(true);
  currenttriad.Multiply(bspottriad[0], lambdarel1_);
  CORE::LARGEROTATIONS::triadtoquaternion<double>(currenttriad, bspottriad1_);

  currenttriad.Clear();
  currenttriad.Multiply(bspottriad[1], lambdarel2_);
  CORE::LARGEROTATIONS::triadtoquaternion<double>(currenttriad, bspottriad2_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<BEAMINTERACTION::BeamLinkRigidJointed> BEAMINTERACTION::BeamLinkRigidJointed::Create()
{
  // for now, we always use a 2-noded linear Reissner element
  return Teuchos::rcp(new BEAMINTERACTION::BeamLinkBeam3rLine2RigidJointed());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::BeamLinkRigidJointed::Print(std::ostream& out) const
{
  check_init();

  BeamLink::Print(out);

  out << "\nbspottriad1_ = ";
  CORE::LINALG::Matrix<3, 3, double> triad;
  CORE::LARGEROTATIONS::quaterniontotriad(bspottriad1_, triad);
  triad.Print(out);
  out << "\nbspottriad2_ = ";
  CORE::LARGEROTATIONS::quaterniontotriad(bspottriad2_, triad);
  triad.Print(out);
  out << "\n";
}

FOUR_C_NAMESPACE_CLOSE

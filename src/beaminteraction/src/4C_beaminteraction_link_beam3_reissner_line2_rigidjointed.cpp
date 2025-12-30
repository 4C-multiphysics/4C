// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_link_beam3_reissner_line2_rigidjointed.hpp"

#include "4C_beam3_reissner.hpp"
#include "4C_beaminteraction_link.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN



BeamInteraction::BeamLinkBeam3rLine2RigidJointedType
    BeamInteraction::BeamLinkBeam3rLine2RigidJointedType::instance_;


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::Communication::ParObject* BeamInteraction::BeamLinkBeam3rLine2RigidJointedType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  BeamInteraction::BeamLinkBeam3rLine2RigidJointed* my_beam3rline2 =
      new BeamInteraction::BeamLinkBeam3rLine2RigidJointed();
  my_beam3rline2->unpack(buffer);
  return my_beam3rline2;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::BeamLinkBeam3rLine2RigidJointed::BeamLinkBeam3rLine2RigidJointed()
    : BeamLinkRigidJointed(),
      linkele_(nullptr),
      bspotforces_(2, Core::LinAlg::SerialDenseVector(3, true))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
BeamInteraction::BeamLinkBeam3rLine2RigidJointed::BeamLinkBeam3rLine2RigidJointed(
    const BeamInteraction::BeamLinkBeam3rLine2RigidJointed& old)
    : BeamInteraction::BeamLinkRigidJointed(old),
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
std::shared_ptr<BeamInteraction::BeamLink> BeamInteraction::BeamLinkBeam3rLine2RigidJointed::clone()
    const
{
  std::shared_ptr<BeamInteraction::BeamLinkBeam3rLine2RigidJointed> newlinker =
      std::make_shared<BeamInteraction::BeamLinkBeam3rLine2RigidJointed>(*this);
  return newlinker;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkBeam3rLine2RigidJointed::setup(int matnum)
{
  check_init();

  // call setup of base class first
  BeamLinkRigidJointed::setup(matnum);

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
  Core::LargeRotations::quaterniontoangle(get_bind_spot_quaternion1(), linkelerotvec);

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

  //  std::cout << "\nSetup():";
  //  this->print(std::cout);

  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::BeamLinkBeam3rLine2RigidJointed::pack(
    Core::Communication::PackBuffer& data) const
{
  check_init_setup();



  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class
  BeamLinkRigidJointed::pack(data);

  // pack linker element
  if (linkele_ != nullptr)
  {
    add_to_pack(data, true);
    linkele_->pack(data);
  }
  else
    add_to_pack(data, false);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::BeamLinkBeam3rLine2RigidJointed::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class
  BeamLinkRigidJointed::unpack(buffer);

  // Unpack data of sub material (these lines are copied from element.cpp)
  bool dataele_exists;
  Core::Communication::extract_from_pack(buffer, dataele_exists);
  if (dataele_exists)
  {
    Core::Communication::ParObject* object =
        Core::Communication::factory(buffer);  // Unpack is done here
    Discret::Elements::Beam3r* linkele = dynamic_cast<Discret::Elements::Beam3r*>(object);
    if (linkele == nullptr)
      FOUR_C_THROW("failed to unpack Beam3r object within BeamLinkBeam3rLine2RigidJointed");
    linkele_ = std::shared_ptr<Discret::Elements::Beam3r>(linkele);
  }
  else
    linkele_ = nullptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BeamInteraction::BeamLinkBeam3rLine2RigidJointed::evaluate_force(
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
  std::copy(&force(0), &force(0) + 6, &forcevec1(0));
  std::copy(&force(0) + 6, &force(0) + 12, &forcevec2(0));

  bspotforces_[0] = forcevec1;
  bspotforces_[1] = forcevec2;

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BeamInteraction::BeamLinkBeam3rLine2RigidJointed::evaluate_stiff(
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

  // Todo can we use std::copy here or even set up 'stiffmat' as a view on the
  //      four individual sub-matrices ?
  for (unsigned int i = 0; i < 6; ++i)
    for (unsigned int j = 0; j < 6; ++j)
    {
      stiffmat11(i, j) = stiffmat(i, j);
      stiffmat12(i, j) = stiffmat(i, 6 + j);
      stiffmat21(i, j) = stiffmat(6 + i, j);
      stiffmat22(i, j) = stiffmat(6 + i, 6 + j);
    }

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BeamInteraction::BeamLinkBeam3rLine2RigidJointed::evaluate_force_stiff(
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

  std::copy(&force(0), &force(0) + 6, &forcevec1(0));
  std::copy(&force(0) + 6, &force(0) + 12, &forcevec2(0));

  for (unsigned int i = 0; i < 6; ++i)
  {
    for (unsigned int j = 0; j < 6; ++j)
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
void BeamInteraction::BeamLinkBeam3rLine2RigidJointed::fill_state_variables_for_element_evaluation(
    Core::LinAlg::Matrix<6, 1, double>& disp_totlag_centerline,
    std::vector<Core::LinAlg::Matrix<4, 1, double>>& Qnode) const
{
  for (unsigned int i = 0; i < 3; ++i)
  {
    disp_totlag_centerline(i) = get_bind_spot_pos1()(i);
    disp_totlag_centerline(3 + i) = get_bind_spot_pos2()(i);
  }

  Qnode.push_back(get_bind_spot_quaternion1());
  Qnode.push_back(get_bind_spot_quaternion2());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double BeamInteraction::BeamLinkBeam3rLine2RigidJointed::get_internal_energy() const
{
  return linkele_->get_internal_energy();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double BeamInteraction::BeamLinkBeam3rLine2RigidJointed::get_kinetic_energy() const
{
  return linkele_->get_kinetic_energy();
}

FOUR_C_NAMESPACE_CLOSE

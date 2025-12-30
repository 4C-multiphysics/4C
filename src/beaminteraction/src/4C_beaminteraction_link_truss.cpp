// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_link_truss.hpp"

#include "4C_beaminteraction_link.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_truss3.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


BeamInteraction::BeamLinkTrussType BeamInteraction::BeamLinkTrussType::instance_;


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::Communication::ParObject* BeamInteraction::BeamLinkTrussType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  BeamInteraction::BeamLinkTruss* my_truss_linker = new BeamInteraction::BeamLinkTruss();
  my_truss_linker->unpack(buffer);
  return my_truss_linker;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::BeamLinkTruss::BeamLinkTruss()
    : BeamLinkPinJointed(),
      linkele_(nullptr),
      bspotforces_(2, Core::LinAlg::SerialDenseVector(3, true))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
BeamInteraction::BeamLinkTruss::BeamLinkTruss(const BeamInteraction::BeamLinkTruss& old)
    : BeamInteraction::BeamLinkPinJointed(old),
      bspotforces_(2, Core::LinAlg::SerialDenseVector(3, true))
{
  if (linkele_ != nullptr)
    linkele_ = std::dynamic_pointer_cast<Discret::Elements::Truss3>(
        std::shared_ptr<Core::Elements::Element>(old.linkele_->clone()));
  else
    linkele_ = nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<BeamInteraction::BeamLink> BeamInteraction::BeamLinkTruss::clone() const
{
  std::shared_ptr<BeamInteraction::BeamLinkTruss> newlinker =
      std::make_shared<BeamInteraction::BeamLinkTruss>(*this);
  return newlinker;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkTruss::init(int id, const std::vector<std::pair<int, int>>& eleids,
    const std::vector<Core::LinAlg::Matrix<3, 1>>& initpos,
    const std::vector<Core::LinAlg::Matrix<3, 3>>& inittriad,
    Inpar::BeamInteraction::CrosslinkerType linkertype, double timelinkwasset)
{
  issetup_ = false;

  BeamLinkPinJointed::init(id, eleids, initpos, inittriad, linkertype, timelinkwasset);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkTruss::setup(const int matnum)
{
  check_init();

  // call setup of base class first
  BeamLinkPinJointed::setup(matnum);

  /* the idea is to use a truss element as auxiliary object that provides us with a
   * response force depending on the position of the two material points on the
   *  parent elements (i.e. binding spots) it is connected to;
   *
   * note: the element instance created in this way can only be used in a limited way
   *       because it is not embedded in a discretization. For example,
   *       Nodes() and other methods are not functional because the
   *       pointers to nodes are not set. Same for reference position of nodes via X() ...
   *
   *       We really only use it as a calculation routine for a sophisticated
   *       (displacement-reaction force) relation here! */
  linkele_ = std::make_shared<Discret::Elements::Truss3>(-1, 0);

  // set material
  linkele_->set_material(0, Mat::factory(matnum));

  // set cross-section area Fixme hard-coded dummy value for now
  linkele_->set_cross_sec(1.0);

  // set dummy node Ids, in order to make NumNodes() method of element return the correct number of
  // nodes
  int nodeids[] = {-1, -1};
  linkele_->set_node_ids(2, nodeids);

  std::vector<double> refpos(6, 0.0);

  for (unsigned int i = 0; i < 3; ++i)
  {
    refpos[i] = get_bind_spot_pos1()(i);
    refpos[3 + i] = get_bind_spot_pos2()(i);
  }

  linkele_->set_up_reference_geometry(refpos);

  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::BeamLinkTruss::pack(Core::Communication::PackBuffer& data) const
{
  check_init_setup();



  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class
  BeamLinkPinJointed::pack(data);

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
void BeamInteraction::BeamLinkTruss::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class
  BeamLinkPinJointed::unpack(buffer);

  // Unpack data of sub material (these lines are copied from element.cpp)
  bool dataele_exists;
  Core::Communication::extract_from_pack(buffer, dataele_exists);
  if (dataele_exists)
  {
    Core::Communication::ParObject* object = Core::Communication::factory(buffer);
    Discret::Elements::Truss3* linkele = dynamic_cast<Discret::Elements::Truss3*>(object);
    if (linkele == nullptr) FOUR_C_THROW("failed to unpack Truss3 object within BeamLinkTruss");
    linkele_ = std::shared_ptr<Discret::Elements::Truss3>(linkele);
  }
  else
    linkele_ = nullptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BeamInteraction::BeamLinkTruss::evaluate_force(
    Core::LinAlg::SerialDenseVector& forcevec1, Core::LinAlg::SerialDenseVector& forcevec2)
{
  check_init_setup();

  std::map<std::string, std::vector<double>> ele_state;
  get_disp_for_element_evaluation(ele_state);

  Core::LinAlg::SerialDenseVector force(6, true);
  Core::LinAlg::SerialDenseMatrix stiffmat(6, 6, true);

  linkele_->calc_internal_force_stiff_tot_lag(ele_state, force, stiffmat);

  std::copy(&force(0), &force(0) + 3, &forcevec1(0));
  std::copy(&force(0) + 3, &force(0) + 6, &forcevec2(0));

  bspotforces_[0] = forcevec1;
  bspotforces_[1] = forcevec2;

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BeamInteraction::BeamLinkTruss::evaluate_stiff(Core::LinAlg::SerialDenseMatrix& stiffmat11,
    Core::LinAlg::SerialDenseMatrix& stiffmat12, Core::LinAlg::SerialDenseMatrix& stiffmat21,
    Core::LinAlg::SerialDenseMatrix& stiffmat22)
{
  check_init_setup();

  std::map<std::string, std::vector<double>> ele_state;
  get_disp_for_element_evaluation(ele_state);

  Core::LinAlg::SerialDenseVector force(6, true);
  Core::LinAlg::SerialDenseMatrix stiffmat(6, 6, true);

  linkele_->calc_internal_force_stiff_tot_lag(ele_state, force, stiffmat);

  for (unsigned int i = 0; i < 3; ++i)
  {
    for (unsigned int j = 0; j < 3; ++j)
    {
      stiffmat11(i, j) = stiffmat(i, j);
      stiffmat12(i, j) = stiffmat(i, 3 + j);
      stiffmat21(i, j) = stiffmat(3 + i, j);
      stiffmat22(i, j) = stiffmat(3 + i, 3 + j);
    }
  }

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BeamInteraction::BeamLinkTruss::evaluate_force_stiff(
    Core::LinAlg::SerialDenseVector& forcevec1, Core::LinAlg::SerialDenseVector& forcevec2,
    Core::LinAlg::SerialDenseMatrix& stiffmat11, Core::LinAlg::SerialDenseMatrix& stiffmat12,
    Core::LinAlg::SerialDenseMatrix& stiffmat21, Core::LinAlg::SerialDenseMatrix& stiffmat22)
{
  check_init_setup();

  std::map<std::string, std::vector<double>> ele_state;
  get_disp_for_element_evaluation(ele_state);

  Core::LinAlg::SerialDenseVector force(6, true);
  Core::LinAlg::SerialDenseMatrix stiffmat(6, 6, true);

  linkele_->calc_internal_force_stiff_tot_lag(ele_state, force, stiffmat);

  std::copy(&force(0), &force(0) + 3, &forcevec1(0));
  std::copy(&force(0) + 3, &force(0) + 6, &forcevec2(0));

  for (unsigned int i = 0; i < 3; ++i)
  {
    for (unsigned int j = 0; j < 3; ++j)
    {
      stiffmat11(i, j) = stiffmat(i, j);
      stiffmat12(i, j) = stiffmat(i, 3 + j);
      stiffmat21(i, j) = stiffmat(3 + i, j);
      stiffmat22(i, j) = stiffmat(3 + i, 3 + j);
    }
  }

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkTruss::reset_state(std::vector<Core::LinAlg::Matrix<3, 1>>& bspotpos,
    std::vector<Core::LinAlg::Matrix<3, 3>>& bspottriad)
{
  check_init_setup();

  BeamLinkPinJointed::reset_state(bspotpos, bspottriad);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkTruss::fill_state_variables_for_element_evaluation(
    Core::LinAlg::Matrix<6, 1, double>& absolute_nodal_positions) const
{
  for (unsigned int i = 0; i < 3; ++i)
  {
    absolute_nodal_positions(i) = get_bind_spot_pos1()(i);
    absolute_nodal_positions(3 + i) = get_bind_spot_pos2()(i);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkTruss::get_disp_for_element_evaluation(
    std::map<std::string, std::vector<double>>& ele_state) const
{
  const auto ref_position = linkele_->x();
  std::vector<double> disp(6, 0);
  for (unsigned int i = 0; i < 3; ++i)
  {
    disp[i] = get_bind_spot_pos1()(i) - ref_position(i, 0);
    disp[3 + i] = get_bind_spot_pos2()(i) - ref_position(i + 3, 0);
  }

  ele_state.emplace(std::make_pair("disp", disp));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkTruss::scale_linker_reference_length(double scalefac)
{
  linkele_->scale_reference_length(scalefac);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLinkTruss::get_binding_spot_force(
    int bspotid, Core::LinAlg::SerialDenseVector& bspotforce) const
{
  bspotforce = bspotforces_[bspotid];
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double BeamInteraction::BeamLinkTruss::get_current_linker_length() const
{
  Core::LinAlg::Matrix<6, 1> xcurr;

  fill_state_variables_for_element_evaluation(xcurr);

  Core::LinAlg::Matrix<6, 1> curr_nodal_coords;
  curr_nodal_coords(0) = xcurr(0) - xcurr(3);
  curr_nodal_coords(1) = xcurr(1) - xcurr(4);
  curr_nodal_coords(2) = xcurr(2) - xcurr(5);
  curr_nodal_coords(3) = curr_nodal_coords(0);
  curr_nodal_coords(4) = curr_nodal_coords(1);
  curr_nodal_coords(5) = curr_nodal_coords(2);

  return linkele_->curr_length(curr_nodal_coords);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double BeamInteraction::BeamLinkTruss::get_internal_energy() const
{
  return linkele_->get_internal_energy();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double BeamInteraction::BeamLinkTruss::get_kinetic_energy() const { return 0.0; }

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_beam_contact_utils.hpp"

#include "4C_beam3_euler_bernoulli.hpp"
#include "4C_beam3_kirchhoff.hpp"
#include "4C_beam3_reissner.hpp"
#include "4C_contact_element.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_rigidsphere.hpp"
#include "4C_utils_fad.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Check, if current element is a solid contact element      popp 05/16|
 *----------------------------------------------------------------------*/
bool BeamInteraction::solid_contact_element(const Core::Elements::Element& element)
{
  const Core::Elements::ElementType& ele_type = element.element_type();

  if (ele_type == CONTACT::ElementType::instance())
    return true;
  else
    return false;
}

/*----------------------------------------------------------------------*
 |  Check, if two elements share a node -> neighbor elements meier 05/14|
 *----------------------------------------------------------------------*/
bool BeamInteraction::elements_share_node(
    const Core::Elements::Element& element1, const Core::Elements::Element& element2)
{
  bool sharenode = false;

  for (int i = 0; i < element1.num_node(); i++)
  {
    int id = element1.node_ids()[i];

    for (int j = 0; j < element2.num_node(); j++)
    {
      if (id == element2.node_ids()[j]) sharenode = true;
    }
  }

  return sharenode;
}

/*----------------------------------------------------------------------*
 |  Calculate beam radius                                    meier 10/14|
 *----------------------------------------------------------------------*/
double BeamInteraction::calc_ele_radius(const Core::Elements::Element* ele)
{
  double eleradius = 0.0;

  const Discret::Elements::Beam3Base* beamele =
      dynamic_cast<const Discret::Elements::Beam3Base*>(ele);
  const Discret::Elements::Rigidsphere* thissphere =
      dynamic_cast<const Discret::Elements::Rigidsphere*>(ele);

  if (beamele != nullptr)
    eleradius = MANIPULATERADIUS * beamele->get_circular_cross_section_radius_for_interactions();
  else if (thissphere != nullptr)
    eleradius = thissphere->radius();
  else
    FOUR_C_THROW(
        "BeamContact::CalcEleRadius: unknown element type; cannot determine cross-section radius");

  return eleradius;
}

/*----------------------------------------------------------------------*
 |  Test intersection of two parallel cylinders              meier 10/14|
 *----------------------------------------------------------------------*/
bool BeamInteraction::intersect_parallel_cylinders(Core::LinAlg::Matrix<3, 1, double>& r1_a,
    Core::LinAlg::Matrix<3, 1, double>& r1_b, Core::LinAlg::Matrix<3, 1, double>& r2_a,
    Core::LinAlg::Matrix<3, 1, double>& r2_b, double& distancelimit)
{
  double parallellinedist = 0.0;
  double etapoint = 0.0;

  // Check, if node r2_a lies within cylinder 1
  parallellinedist = calc_point_line_dist(r1_a, r1_b, r2_a, etapoint);
  if (parallellinedist < distancelimit and fabs(etapoint) < 1.0 + distancelimit) return true;

  // Check, if node r2_b lies within cylinder 1
  etapoint = 0.0;
  parallellinedist = calc_point_line_dist(r1_a, r1_b, r2_b, etapoint);
  if (parallellinedist < distancelimit and fabs(etapoint) < 1.0 + distancelimit) return true;

  // Check, if node r1_a lies within cylinder 2
  etapoint = 0.0;
  parallellinedist = calc_point_line_dist(r2_a, r2_b, r1_a, etapoint);
  if (parallellinedist < distancelimit and fabs(etapoint) < 1.0 + distancelimit) return true;

  // Check, if node r1_b lies within cylinder 2
  etapoint = 0.0;
  parallellinedist = calc_point_line_dist(r2_a, r2_b, r1_b, etapoint);
  if (parallellinedist < distancelimit and fabs(etapoint) < 1.0 + distancelimit) return true;

  // Else, we have no intersection!!!
  return false;
}

/*-----------------------------------------------------------------------------------*
 |  Test intersection of two non-parallel, arbitrary oriented cylinders   meier 10/14|
 *-----------------------------------------------------------------------------------*/
bool BeamInteraction::intersect_arbitrary_cylinders(Core::LinAlg::Matrix<3, 1, double>& r1_a,
    Core::LinAlg::Matrix<3, 1, double>& r1_b, Core::LinAlg::Matrix<3, 1, double>& r2_a,
    Core::LinAlg::Matrix<3, 1, double>& r2_b, double& distancelimit,
    std::pair<double, double>& closestpoints, bool& etaset)
{
  Core::LinAlg::Matrix<3, 1, double> t1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, double> t2(Core::LinAlg::Initialization::zero);
  double closestnodetolinedist(0.0);
  double closestlinedist(0.0);
  double closestnodaldist(0.0);
  Core::LinAlg::Matrix<3, 1, double> vec1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, double> vec2(Core::LinAlg::Initialization::zero);

  t1 = Core::FADUtils::diff_vector(r1_b, r1_a);
  t2 = Core::FADUtils::diff_vector(r2_b, r2_a);

  vec1 = Core::FADUtils::diff_vector(r1_a, r2_a);
  vec2 = Core::FADUtils::vector_product(t1, t2);
  closestlinedist = Core::FADUtils::norm(Core::FADUtils::scalar_product(vec1, vec2)) /
                    Core::FADUtils::vector_norm<3>(vec2);

  // 1)Check, if a solution for the Closest-Point-Projection of the two lines exists in eta1_seg,
  // eta2_seg \in [-1.0;1.0] (existence of local minimum in the 2D domain eta1_seg, eta2_seg \in
  // [-1.0;1.0])
  if (fabs(closestlinedist) > distancelimit)
  {
    etaset = false;
    return false;
  }
  else
  {
    // Calculate values eta1_seg and eta2_seg of closest point coordinates. The definitions of b_1,
    // b_2, t_1 and t_2 are according to the paper "ON CONTACT BETWEEN THREE-DIMENSIONAL BEAMS
    // UNDERGOING LARGE DEFLECTIONS" of Wriggers and Zavarise (1997)
    Core::LinAlg::Matrix<3, 1, double> b_1(r1_a);
    Core::LinAlg::Matrix<3, 1, double> b_2(r2_a);
    b_1.update(1.0, r1_b, 1.0);
    b_2.update(1.0, r2_b, 1.0);
    double eta1_seg(0.0);
    double eta2_seg(0.0);

    // local variables for element coordinates
    double aux1 = Core::FADUtils::scalar_product(Core::FADUtils::diff_vector(b_1, b_2), t2);
    aux1 = aux1 * Core::FADUtils::scalar_product(t1, t2);
    double aux2 = Core::FADUtils::scalar_product(Core::FADUtils::diff_vector(b_2, b_1), t1);
    aux2 = aux2 * Core::FADUtils::scalar_product(t2, t2);
    eta1_seg = (aux1 + aux2) /
               (Core::FADUtils::scalar_product(t2, t2) * Core::FADUtils::scalar_product(t1, t1) -
                   Core::FADUtils::scalar_product(t2, t1) * Core::FADUtils::scalar_product(t2, t1));

    aux1 = Core::FADUtils::scalar_product(Core::FADUtils::diff_vector(b_2, b_1), t1);
    aux1 = aux1 * Core::FADUtils::scalar_product(t1, t2);
    aux2 = Core::FADUtils::scalar_product(Core::FADUtils::diff_vector(b_1, b_2), t2);
    aux2 = aux2 * Core::FADUtils::scalar_product(t1, t1);
    eta2_seg = (aux1 + aux2) /
               (Core::FADUtils::scalar_product(t2, t2) * Core::FADUtils::scalar_product(t1, t1) -
                   Core::FADUtils::scalar_product(t2, t1) * Core::FADUtils::scalar_product(t2, t1));

    if (fabs(eta1_seg) < 1.0 and fabs(eta2_seg) < 1.0)
    {
      // The closest point are only set, if we have detected an intersection at a valid closest
      // point with eta1_seg, eta2_seg \in [-1.0;1.0]
      closestpoints = std::make_pair(eta1_seg, eta2_seg);
      etaset = true;

      return true;
    }
    // 2)Check, if one of the four pairs of boundary nodes is close (existence of boundary minimum
    // at the four corner points of the domain eta1_seg, eta2_seg \in [-1.0;1.0])
    else
    {
      etaset = false;

      closestnodaldist = BeamInteraction::get_closest_endpoint_dist(r1_a, r1_b, r2_a, r2_b);
      if (fabs(closestnodaldist) < distancelimit)
      {
        return true;
      }
      // 3)Check, if a local minimum exists at one of the four 1D boundaries eta1_seg=-1.0,
      // eta1_seg=1.0, eta2_seg=-1.0 and eta2_seg=1.0 of the domain eta1_seg, eta2_seg \in
      // [-1.0;1.0]
      else
      {
        double etapoint = 0.0;

        closestnodetolinedist = calc_point_line_dist(r1_a, r1_b, r2_a, etapoint);
        if (closestnodetolinedist < distancelimit and fabs(etapoint) < 1.0) return true;

        etapoint = 0.0;
        closestnodetolinedist = calc_point_line_dist(r1_a, r1_b, r2_b, etapoint);
        if (closestnodetolinedist < distancelimit and fabs(etapoint) < 1.0) return true;

        etapoint = 0.0;
        closestnodetolinedist = calc_point_line_dist(r2_a, r2_b, r1_a, etapoint);
        if (closestnodetolinedist < distancelimit and fabs(etapoint) < 1.0) return true;

        etapoint = 0.0;
        closestnodetolinedist = calc_point_line_dist(r2_a, r2_b, r1_b, etapoint);
        if (closestnodetolinedist < distancelimit and fabs(etapoint) < 1.0) return true;

        // No intersection, if we met none of these criteria!!!
        return false;
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Calculate closest distance of a point and a line         meier 10/14|
 *----------------------------------------------------------------------*/
double BeamInteraction::calc_point_line_dist(
    Core::LinAlg::Matrix<3, 1, double>& rline_a,  // at eta=-1.0
    Core::LinAlg::Matrix<3, 1, double>& rline_b,  // at eta=1.0
    Core::LinAlg::Matrix<3, 1, double>& rp, double& eta)
{
  double closestpointlinedist = 0.0;

  Core::LinAlg::Matrix<3, 1, double> tline(Core::LinAlg::Initialization::zero);
  tline = Core::FADUtils::diff_vector(rline_b, rline_a);
  Core::LinAlg::Matrix<3, 1, double> vec1(Core::LinAlg::Initialization::zero);
  vec1 = Core::FADUtils::diff_vector(rline_a, rp);
  closestpointlinedist =
      fabs(Core::FADUtils::vector_norm<3>(Core::FADUtils::vector_product(vec1, tline)) /
           Core::FADUtils::vector_norm<3>(tline));

  vec1.clear();
  vec1.update(-1.0, rline_a, 0.0);
  vec1.update(-1.0, rline_b, 1.0);
  vec1.update(2.0, rp, 1.0);

  eta = Core::FADUtils::scalar_product(tline, vec1) / Core::FADUtils::scalar_product(tline, tline);

  return closestpointlinedist;
}

/*----------------------------------------------------------------------*
 |  Calculate angle enclosed by two vectors a and b          meier 10/14|
 *----------------------------------------------------------------------*/
double BeamInteraction::calc_angle(
    Core::LinAlg::Matrix<3, 1, double> a, Core::LinAlg::Matrix<3, 1, double> b)
{
  if (Core::FADUtils::vector_norm<3>(a) < 1.0e-12 or Core::FADUtils::vector_norm<3>(b) < 1.0e-12)
    FOUR_C_THROW("Can not determine angle for zero vector!");

  double scalarproduct =
      std::fabs(Core::FADUtils::scalar_product(a, b) /
                (Core::FADUtils::vector_norm<3>(a) * Core::FADUtils::vector_norm<3>(b)));
  double angle = 0.0;

  if (scalarproduct < 1.0)
    angle =
        std::acos(scalarproduct);  // returns an angle \in [0;pi/2] since scalarproduct \in [0;1.0]
  else
    angle = 0.0;  // This step is necessary due to round-off errors. However, the derivative
                  // information of the FAD quantity gets lost here!

  // We want an angle \in [0;pi/2] in each case:
  if (angle > std::numbers::pi / 2.0)
    FOUR_C_THROW("Something went wrong here, angle should be in the interval [0;pi/2]!");

  return angle;
}

/*----------------------------------------------------------------------*
 |  Get closest distance between the endpoints of two lines   meier 10/14|
 *----------------------------------------------------------------------*/
template <typename Type>
Type BeamInteraction::get_closest_endpoint_dist(Core::LinAlg::Matrix<3, 1, Type> r1_a,
    Core::LinAlg::Matrix<3, 1, Type> r1_b, Core::LinAlg::Matrix<3, 1, Type> r2_a,
    Core::LinAlg::Matrix<3, 1, Type> r2_b)
{
  Type minnodaldist = 0.0;
  Type nodaldist = 0.0;

  minnodaldist = Core::FADUtils::vector_norm<3>(Core::FADUtils::diff_vector(r1_a, r2_a));

  nodaldist = Core::FADUtils::vector_norm<3>(Core::FADUtils::diff_vector(r1_a, r2_b));
  if (nodaldist < minnodaldist) minnodaldist = nodaldist;

  nodaldist = Core::FADUtils::vector_norm<3>(Core::FADUtils::diff_vector(r1_b, r2_a));
  if (nodaldist < minnodaldist) minnodaldist = nodaldist;

  nodaldist = Core::FADUtils::vector_norm<3>(Core::FADUtils::diff_vector(r1_b, r2_b));
  if (nodaldist < minnodaldist) minnodaldist = nodaldist;

  return minnodaldist;
}

/*----------------------------------------------------------------------------------------*
 |  Determine inpute parameter representing the additive searchbox increment   meier 10/14|
 *----------------------------------------------------------------------------------------*/
double BeamInteraction::determine_searchbox_inc(Teuchos::ParameterList& beamcontactparams)
{
  double searchboxinc = 0.0;

  std::vector<double> extval;
  std::string extrusion_value_in(
      Teuchos::getNumericStringParameter(beamcontactparams, "BEAMS_EXTVAL"));

  Core::IO::ValueParser extrusionvalue_parser(
      extrusion_value_in, {.user_scope_message = "While reading extrusion values: "});

  while (!extrusionvalue_parser.at_end())
  {
    extval.push_back(extrusionvalue_parser.read<double>());
  }

  if ((int)extval.size() > 2)
    FOUR_C_THROW("BEAMS_EXTVAL should contain no more than two values. Check your input file.");
  if (extval.size() == 1)
    searchboxinc = extval.at(0);
  else
    searchboxinc = std::max(extval.at(0), extval.at(1));

  return searchboxinc;
}

FOUR_C_NAMESPACE_CLOSE

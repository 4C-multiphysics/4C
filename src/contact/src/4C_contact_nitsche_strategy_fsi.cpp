// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_nitsche_strategy_fsi.hpp"

#include "4C_contact_element.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_nitsche_integrator_fsi.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_mortar_projector.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

void CONTACT::NitscheStrategyFsi::apply_force_stiff_cmt(
    std::shared_ptr<Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<Core::LinAlg::SparseOperator>& kt,
    std::shared_ptr<Core::LinAlg::Vector<double>>& f, const int step, const int iter,
    bool predictor)
{
  if (predictor) return;
  CONTACT::NitscheStrategy::apply_force_stiff_cmt(dis, kt, f, step, iter, predictor);
}

void CONTACT::NitscheStrategyFsi::set_state(
    const Mortar::StateType& statename, const Core::LinAlg::Vector<double>& vec)
{
  CONTACT::NitscheStrategy::set_state(statename, vec);
  if (statename == Mortar::state_new_displacement)
  {
    do_contact_search();
  }
}

void CONTACT::NitscheStrategyFsi::do_contact_search()
{
  for (auto& interface : interface_)
  {
    interface->initialize();
    interface->evaluate_search_binarytree();
    interface->evaluate_nodal_normals();
    interface->export_nodal_normals();
  }
}

bool CONTACT::NitscheStrategyFsi::check_nitsche_contact_state(CONTACT::Element* cele,
    const Core::LinAlg::Matrix<2, 1>& xsi, const double& full_fsi_traction, double& gap)
{
  return CONTACT::Utils::check_nitsche_contact_state(
      *contact_interfaces()[0], pen_n_, weighting_, cele, xsi, full_fsi_traction, gap);
}

bool CONTACT::Utils::check_nitsche_contact_state(CONTACT::Interface& contactinterface,
    const double& pen_n, CONTACT::NitscheWeighting weighting, CONTACT::Element* cele,
    const Core::LinAlg::Matrix<2, 1>& xsi, const double& full_fsi_traction, double& gap)
{
  // No target elements found
  if (!cele->mo_data().num_search_elements())
  {
    gap = 1.e12;
    return true;
  }
  if (!(cele->shape() == Core::FE::CellType::quad4 || cele->shape() == Core::FE::CellType::quad8 ||
          cele->shape() == Core::FE::CellType::quad9))
    FOUR_C_THROW("This element shape is not yet implemented!");

  // find the corresponding target element
  CONTACT::Element* other_cele = nullptr;
  double target_xi[2] = {0.0, 0.0};
  double projalpha = 0.0;
  static const double tol = 1e-4;
  double near = 0.;
  double max_relevant_gap = gap * 2.;  // safety factor 2
  for (int m = 0; m < cele->mo_data().num_search_elements(); ++m)
  {
    if (other_cele) break;
    auto* test_ele = dynamic_cast<CONTACT::Element*>(
        contactinterface.discret().g_element(cele->mo_data().search_elements()[m]));
    if (!test_ele)
      FOUR_C_THROW("Cannot find element with gid {}", cele->mo_data().search_elements()[m]);

    Mortar::Projector::impl(*cele, *test_ele)
        ->project_gauss_point_3d(*cele, xsi.data(), *test_ele, target_xi, projalpha);
    bool is_inside = false;
    switch (test_ele->shape())
    {
      case Core::FE::CellType::quad4:
      case Core::FE::CellType::quad8:
      case Core::FE::CellType::quad9:
        if (abs(target_xi[0]) < 1. + tol && abs(target_xi[1]) < 1. + tol) is_inside = true;
        break;
      default:
        FOUR_C_THROW("This element shape is not yet implemented ({})!", test_ele->shape());
    }
    if (is_inside) other_cele = test_ele;
    // distance check
    if (other_cele)
    {
      double center[2] = {0., 0.};
      Core::LinAlg::Matrix<3, 1> sc, mc;
      cele->local_to_global(center, sc.data(), 0);
      other_cele->local_to_global(center, mc.data(), 0);
      near = 2. * std::max(cele->max_edge_size(), other_cele->max_edge_size());
      sc.update(-1., mc, 1.);
      if (sc.norm2() > std::max(near, max_relevant_gap)) other_cele = nullptr;
    }
  }
  // orientation check
  if (other_cele)
  {
    double center[2] = {0., 0.};
    Core::LinAlg::Matrix<3, 1> source_normal, target_normal;
    cele->compute_unit_normal_at_xi(center, source_normal.data());
    other_cele->compute_unit_normal_at_xi(center, target_normal.data());
    if (source_normal.dot(target_normal) > 0.) other_cele = nullptr;
  }
  // no target element hit
  if (other_cele == nullptr)
  {
    gap = 1e12;
    return true;
  }

  Core::LinAlg::Matrix<2, 1> target_xi_matrix(target_xi, true);
  double target_x_glob[3];
  double source_x_glob[3];
  cele->local_to_global(xsi.data(), source_x_glob, 0);
  other_cele->local_to_global(target_xi, target_x_glob, 0);
  Core::LinAlg::Matrix<3, 1> target_x(target_x_glob, true);
  Core::LinAlg::Matrix<3, 1> source_x(source_x_glob, true);

  Core::LinAlg::Matrix<3, 1> n(target_x);
  n.update(-1., source_x, 1.);
  Core::LinAlg::Matrix<3, 1> diff(n);
  n.scale(1. / n.norm2());
  gap = diff.dot(n);
  Core::LinAlg::Matrix<3, 1> myN;
  cele->compute_unit_normal_at_xi(xsi.data(), myN.data());
  double dir = n.dot(myN);
  if (dir > 0)
    gap *= 1.;
  else
    gap *= -1.;

  // target element on the other side
  if (gap < -near)
  {
    gap = 1e12;
    return true;
  }


  double w_source = 0.;
  double w_target = 0.;
  double my_pen = pen_n;
  double my_pen_t = 0.0;
  CONTACT::Utils::nitsche_weights_and_scaling(
      *cele, *other_cele, weighting, w_source, w_target, my_pen, my_pen_t);

  Core::LinAlg::Matrix<3, 1> ele_n;
  cele->compute_unit_normal_at_xi(xsi.data(), ele_n.data());

  double stress_plus_penalty =
      w_source * CONTACT::Utils::solid_cauchy_at_xi(cele, xsi, ele_n, ele_n) +
      w_target * CONTACT::Utils::solid_cauchy_at_xi(other_cele, target_xi_matrix, ele_n, ele_n) +
      my_pen * gap;

  if (stress_plus_penalty >= full_fsi_traction)
    return true;  // aka evaluate FSI
  else
    return false;  // aka evaluate contact
}

FOUR_C_NAMESPACE_CLOSE

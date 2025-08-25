// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRY_PAIR_LINE_TO_VOLUME_SEGMENTATION_GEOMETRY_FUNCTIONS_TEST_HPP
#define FOUR_C_GEOMETRY_PAIR_LINE_TO_VOLUME_SEGMENTATION_GEOMETRY_FUNCTIONS_TEST_HPP

#include "4C_beam3_reissner.hpp"
#include "4C_solid_3D_ele.hpp"

namespace
{
  using namespace FourC;

  /**
   * \brief The following code part is generated with beamme. The function defines element
   * coordinates for unit test examples.
   */
  void xtest_line_along_element_surface_geometry(
      std::vector<std::shared_ptr<Core::Elements::Element>>& line_elements,
      std::vector<std::shared_ptr<Core::Elements::Element>>& volume_elements,
      std::vector<Core::LinAlg::Matrix<12, 1, double>>& q_line_elements,
      std::vector<double>& line_ref_lengths,
      std::vector<Core::LinAlg::Matrix<24, 1, double>>& q_volume_elements)
  {
    // Create the elements.
    const int dummy_node_ids[2] = {0, 1};
    line_elements.push_back(std::make_shared<Discret::Elements::Beam3r>(0, 0));
    line_elements.back()->set_node_ids(2, dummy_node_ids);
    volume_elements.push_back(std::make_shared<Discret::Elements::Solid>(1, 0));
    volume_elements.push_back(std::make_shared<Discret::Elements::Solid>(2, 0));

    // Positional and tangent DOFs of the line(s).
    q_line_elements.push_back(Core::LinAlg::Matrix<12, 1, double>());
    q_line_elements.back()(0) = -0.5;
    q_line_elements.back()(1) = 0.0;
    q_line_elements.back()(2) = 0.0;
    q_line_elements.back()(3) = 0.7071067811865477;
    q_line_elements.back()(4) = 0.7071067811865475;
    q_line_elements.back()(5) = 0.0;
    q_line_elements.back()(6) = 0.5;
    q_line_elements.back()(7) = 0.0;
    q_line_elements.back()(8) = 0.0;
    q_line_elements.back()(9) = 0.7071067811865477;
    q_line_elements.back()(10) = 0.7071067811865475;
    q_line_elements.back()(11) = 0.0;
    line_ref_lengths.push_back(1.057009395869620727);

    // Positional DOFs of the solid(s).
    q_volume_elements.push_back(Core::LinAlg::Matrix<24, 1, double>());
    q_volume_elements.back()(0) = -0.5;
    q_volume_elements.back()(1) = -0.5;
    q_volume_elements.back()(2) = 1.0;
    q_volume_elements.back()(3) = -0.5;
    q_volume_elements.back()(4) = -0.5;
    q_volume_elements.back()(5) = 0.0;
    q_volume_elements.back()(6) = -0.5;
    q_volume_elements.back()(7) = 0.5;
    q_volume_elements.back()(8) = 0.0;
    q_volume_elements.back()(9) = -0.5;
    q_volume_elements.back()(10) = 0.5;
    q_volume_elements.back()(11) = 1.0;
    q_volume_elements.back()(12) = 0.5;
    q_volume_elements.back()(13) = -0.5;
    q_volume_elements.back()(14) = 1.0;
    q_volume_elements.back()(15) = 0.5;
    q_volume_elements.back()(16) = -0.5;
    q_volume_elements.back()(17) = 0.0;
    q_volume_elements.back()(18) = 0.5;
    q_volume_elements.back()(19) = 0.5;
    q_volume_elements.back()(20) = 0.0;
    q_volume_elements.back()(21) = 0.5;
    q_volume_elements.back()(22) = 0.5;
    q_volume_elements.back()(23) = 1.0;
    q_volume_elements.push_back(Core::LinAlg::Matrix<24, 1, double>());
    q_volume_elements.back()(0) = -0.5;
    q_volume_elements.back()(1) = -0.5;
    q_volume_elements.back()(2) = 0.0;
    q_volume_elements.back()(3) = -0.5;
    q_volume_elements.back()(4) = -0.5;
    q_volume_elements.back()(5) = -1.0;
    q_volume_elements.back()(6) = -0.5;
    q_volume_elements.back()(7) = 0.5;
    q_volume_elements.back()(8) = -1.0;
    q_volume_elements.back()(9) = -0.5;
    q_volume_elements.back()(10) = 0.5;
    q_volume_elements.back()(11) = 0.0;
    q_volume_elements.back()(12) = 0.5;
    q_volume_elements.back()(13) = -0.5;
    q_volume_elements.back()(14) = 0.0;
    q_volume_elements.back()(15) = 0.5;
    q_volume_elements.back()(16) = -0.5;
    q_volume_elements.back()(17) = -1.0;
    q_volume_elements.back()(18) = 0.5;
    q_volume_elements.back()(19) = 0.5;
    q_volume_elements.back()(20) = -1.0;
    q_volume_elements.back()(21) = 0.5;
    q_volume_elements.back()(22) = 0.5;
    q_volume_elements.back()(23) = 0.0;
  }

  /**
   * \brief Create the geometry for two solid elements and a line between them.
   */
  void xtest_line_in_small_elements_geometry(
      std::vector<std::shared_ptr<Core::Elements::Element>>& line_elements,
      std::vector<std::shared_ptr<Core::Elements::Element>>& volume_elements,
      std::vector<Core::LinAlg::Matrix<12, 1, double>>& q_line_elements,
      std::vector<double>& line_ref_lengths,
      std::vector<Core::LinAlg::Matrix<24, 1, double>>& q_volume_elements)
  {
    // Create the elements.
    const int dummy_node_ids[2] = {0, 1};
    line_elements.push_back(std::make_shared<Discret::Elements::Beam3r>(0, 0));
    line_elements.back()->set_node_ids(2, dummy_node_ids);
    volume_elements.push_back(std::make_shared<Discret::Elements::Solid>(1, 0));
    volume_elements.push_back(std::make_shared<Discret::Elements::Solid>(2, 0));

    // Positional and tangent DOFs of the line(s).
    q_line_elements.push_back(Core::LinAlg::Matrix<12, 1, double>());
    q_line_elements.back()(0) = -0.042544945846900002;
    q_line_elements.back()(1) = 0.0107761505991;
    q_line_elements.back()(2) = -0.079418958671100007;
    q_line_elements.back()(3) = -0.50707031399068514;
    q_line_elements.back()(4) = 0.86012099777576989;
    q_line_elements.back()(5) = 0.055421709235663169;
    q_line_elements.back()(6) = -0.047353734337799998;
    q_line_elements.back()(7) = 0.0189330863154;
    q_line_elements.back()(8) = -0.0788933682941;
    q_line_elements.back()(9) = -0.50707031399068514;
    q_line_elements.back()(10) = 0.86012099777576989;
    q_line_elements.back()(11) = 0.055421709235663169;
    line_ref_lengths.push_back(0.009483474694143378611);

    // Positional DOFs of the solid(s).
    q_volume_elements.push_back(Core::LinAlg::Matrix<24, 1, double>());
    q_volume_elements.back()(0) = -0.059010802172665688;
    q_volume_elements.back()(1) = -4.0342241261323264e-05;
    q_volume_elements.back()(2) = -0.066666666666666652;
    q_volume_elements.back()(3) = -0.056622864988174502;
    q_volume_elements.back()(4) = 0.015639754881976043;
    q_volume_elements.back()(5) = -0.066666666666666652;
    q_volume_elements.back()(6) = -0.040709644689084147;
    q_volume_elements.back()(7) = 0.016677096446779945;
    q_volume_elements.back()(8) = -0.066666666666666666;
    q_volume_elements.back()(9) = -0.041287025540519016;
    q_volume_elements.back()(10) = -0.0036547536928699174;
    q_volume_elements.back()(11) = -0.066666666666666666;
    q_volume_elements.back()(12) = -0.059019184401410199;
    q_volume_elements.back()(13) = -4.8643836701577456e-05;
    q_volume_elements.back()(14) = -0.083333333333333273;
    q_volume_elements.back()(15) = -0.05663155442052388;
    q_volume_elements.back()(16) = 0.015632134107931933;
    q_volume_elements.back()(17) = -0.083333333333333287;
    q_volume_elements.back()(18) = -0.040733363534936125;
    q_volume_elements.back()(19) = 0.016660438882450789;
    q_volume_elements.back()(20) = -0.083333333333333301;
    q_volume_elements.back()(21) = -0.04130006471411609;
    q_volume_elements.back()(22) = -0.0036684247948681049;
    q_volume_elements.back()(23) = -0.083333333333333301;
    q_volume_elements.push_back(Core::LinAlg::Matrix<24, 1, double>());
    q_volume_elements.back()(0) = -0.056622864988174502;
    q_volume_elements.back()(1) = 0.015639754881976043;
    q_volume_elements.back()(2) = -0.066666666666666652;
    q_volume_elements.back()(3) = -0.053802653656544963;
    q_volume_elements.back()(4) = 0.030093324120797998;
    q_volume_elements.back()(5) = -0.066666666666666652;
    q_volume_elements.back()(6) = -0.040779423818941965;
    q_volume_elements.back()(7) = 0.032105257644590407;
    q_volume_elements.back()(8) = -0.066666666666666666;
    q_volume_elements.back()(9) = -0.040709644689084147;
    q_volume_elements.back()(10) = 0.016677096446779945;
    q_volume_elements.back()(11) = -0.066666666666666666;
    q_volume_elements.back()(12) = -0.05663155442052388;
    q_volume_elements.back()(13) = 0.015632134107931933;
    q_volume_elements.back()(14) = -0.083333333333333287;
    q_volume_elements.back()(15) = -0.053807767110452855;
    q_volume_elements.back()(16) = 0.030086120994904451;
    q_volume_elements.back()(17) = -0.083333333333333287;
    q_volume_elements.back()(18) = -0.040791092460926263;
    q_volume_elements.back()(19) = 0.032092609002086644;
    q_volume_elements.back()(20) = -0.083333333333333301;
    q_volume_elements.back()(21) = -0.040733363534936125;
    q_volume_elements.back()(22) = 0.016660438882450789;
    q_volume_elements.back()(23) = -0.083333333333333301;
  }

  /**
   * \brief The following code part is generated with beamme. The function defines element
   * coordinates for unit test examples.
   */
  void xtest_multiple_intersections_hex27_geometry(
      std::vector<std::shared_ptr<Core::Elements::Element>>& line_elements,
      std::vector<std::shared_ptr<Core::Elements::Element>>& volume_elements,
      std::vector<Core::LinAlg::Matrix<12, 1, double>>& q_line_elements,
      std::vector<double>& line_ref_lengths,
      std::vector<Core::LinAlg::Matrix<81, 1, double>>& q_volume_elements)
  {
    // Create the elements.
    const int dummy_node_ids[2] = {0, 1};
    line_elements.push_back(std::make_shared<Discret::Elements::Beam3r>(0, 0));
    line_elements.back()->set_node_ids(2, dummy_node_ids);
    volume_elements.push_back(std::make_shared<Discret::Elements::Solid>(1, 0));

    // Positional and tangent DOFs of the line(s).
    q_line_elements.push_back(Core::LinAlg::Matrix<12, 1, double>());
    q_line_elements.back()(0) = -0.4;
    q_line_elements.back()(1) = -0.5;
    q_line_elements.back()(2) = 0.515;
    q_line_elements.back()(3) = 0.13302222155948917;
    q_line_elements.back()(4) = 0.754406506735489;
    q_line_elements.back()(5) = 0.6427876096865395;
    q_line_elements.back()(6) = -0.3135063030374552;
    q_line_elements.back()(7) = -0.009469868904796452;
    q_line_elements.back()(8) = 0.47142212862617094;
    q_line_elements.back()(9) = 0.13302222155948917;
    q_line_elements.back()(10) = 0.754406506735489;
    q_line_elements.back()(11) = 0.6427876096865395;
    line_ref_lengths.push_back(0.52850469793481058556);

    // Positional DOFs of the solid(s).
    q_volume_elements.push_back(Core::LinAlg::Matrix<81, 1, double>());
    q_volume_elements.back()(0) = -0.45;
    q_volume_elements.back()(1) = -0.425;
    q_volume_elements.back()(2) = 0.6;
    q_volume_elements.back()(3) = -0.5;
    q_volume_elements.back()(4) = -0.5;
    q_volume_elements.back()(5) = -0.5;
    q_volume_elements.back()(6) = -0.5;
    q_volume_elements.back()(7) = 0.5;
    q_volume_elements.back()(8) = -0.5;
    q_volume_elements.back()(9) = -0.5;
    q_volume_elements.back()(10) = 0.5;
    q_volume_elements.back()(11) = 0.5;
    q_volume_elements.back()(12) = 0.5;
    q_volume_elements.back()(13) = -0.5;
    q_volume_elements.back()(14) = 0.5;
    q_volume_elements.back()(15) = 0.5;
    q_volume_elements.back()(16) = -0.5;
    q_volume_elements.back()(17) = -0.5;
    q_volume_elements.back()(18) = 0.5;
    q_volume_elements.back()(19) = 0.5;
    q_volume_elements.back()(20) = -0.5;
    q_volume_elements.back()(21) = 0.5;
    q_volume_elements.back()(22) = 0.5;
    q_volume_elements.back()(23) = 0.5;
    q_volume_elements.back()(24) = -0.5;
    q_volume_elements.back()(25) = -0.5;
    q_volume_elements.back()(26) = 0.0;
    q_volume_elements.back()(27) = -0.5;
    q_volume_elements.back()(28) = 0.0;
    q_volume_elements.back()(29) = -0.5;
    q_volume_elements.back()(30) = -0.5;
    q_volume_elements.back()(31) = 0.5;
    q_volume_elements.back()(32) = 0.0;
    q_volume_elements.back()(33) = -0.4;
    q_volume_elements.back()(34) = -0.1;
    q_volume_elements.back()(35) = 0.45;
    q_volume_elements.back()(36) = 0.0;
    q_volume_elements.back()(37) = -0.5;
    q_volume_elements.back()(38) = 0.5;
    q_volume_elements.back()(39) = 0.0;
    q_volume_elements.back()(40) = -0.5;
    q_volume_elements.back()(41) = -0.5;
    q_volume_elements.back()(42) = 0.0;
    q_volume_elements.back()(43) = 0.5;
    q_volume_elements.back()(44) = -0.5;
    q_volume_elements.back()(45) = 0.0;
    q_volume_elements.back()(46) = 0.5;
    q_volume_elements.back()(47) = 0.5;
    q_volume_elements.back()(48) = 0.5;
    q_volume_elements.back()(49) = -0.5;
    q_volume_elements.back()(50) = 0.0;
    q_volume_elements.back()(51) = 0.5;
    q_volume_elements.back()(52) = 0.0;
    q_volume_elements.back()(53) = -0.5;
    q_volume_elements.back()(54) = 0.5;
    q_volume_elements.back()(55) = 0.5;
    q_volume_elements.back()(56) = 0.0;
    q_volume_elements.back()(57) = 0.5;
    q_volume_elements.back()(58) = 0.0;
    q_volume_elements.back()(59) = 0.5;
    q_volume_elements.back()(60) = -0.5;
    q_volume_elements.back()(61) = 0.0;
    q_volume_elements.back()(62) = 0.0;
    q_volume_elements.back()(63) = 0.0;
    q_volume_elements.back()(64) = -0.5;
    q_volume_elements.back()(65) = 0.0;
    q_volume_elements.back()(66) = 0.0;
    q_volume_elements.back()(67) = 0.0;
    q_volume_elements.back()(68) = -0.5;
    q_volume_elements.back()(69) = 0.0;
    q_volume_elements.back()(70) = 0.5;
    q_volume_elements.back()(71) = 0.0;
    q_volume_elements.back()(72) = 0.0;
    q_volume_elements.back()(73) = 0.0;
    q_volume_elements.back()(74) = 0.5;
    q_volume_elements.back()(75) = 0.5;
    q_volume_elements.back()(76) = 0.0;
    q_volume_elements.back()(77) = 0.0;
    q_volume_elements.back()(78) = 0.0;
    q_volume_elements.back()(79) = 0.0;
    q_volume_elements.back()(80) = 0.0;
  }

  /**
   * \brief The following code part is generated with beamme. The function defines element
   * coordinates for unit test examples.
   */
  void xtest_multiple_intersections_tet10_geometry(
      std::vector<std::shared_ptr<Core::Elements::Element>>& line_elements,
      std::vector<std::shared_ptr<Core::Elements::Element>>& volume_elements,
      std::vector<Core::LinAlg::Matrix<12, 1, double>>& q_line_elements,
      std::vector<double>& line_ref_lengths,
      std::vector<Core::LinAlg::Matrix<30, 1, double>>& q_volume_elements)
  {
    // Create the elements.
    const int dummy_node_ids[2] = {0, 1};
    line_elements.push_back(std::make_shared<Discret::Elements::Beam3r>(0, 0));
    line_elements.back()->set_node_ids(2, dummy_node_ids);
    volume_elements.push_back(std::make_shared<Discret::Elements::Solid>(1, 0));

    // Positional and tangent DOFs of the line(s).
    q_line_elements.push_back(Core::LinAlg::Matrix<12, 1, double>());
    q_line_elements.back()(0) = 0.3984274010784635;
    q_line_elements.back()(1) = 0.112954622498459;
    q_line_elements.back()(2) = 0.4240495331815059;
    q_line_elements.back()(3) = -0.7377092435057686;
    q_line_elements.back()(4) = -0.6531007789277388;
    q_line_elements.back()(5) = -0.17101007166283447;
    q_line_elements.back()(6) = 0.10157259892153653;
    q_line_elements.back()(7) = 0.487045377501541;
    q_line_elements.back()(8) = 0.27595046681849406;
    q_line_elements.back()(9) = -0.9253422132715935;
    q_line_elements.back()(10) = -0.2495541519597498;
    q_line_elements.back()(11) = -0.28541988994686207;
    line_ref_lengths.push_back(0.57148255427415728391);

    // Positional DOFs of the solid(s).
    q_volume_elements.push_back(Core::LinAlg::Matrix<30, 1, double>());
    q_volume_elements.back()(0) = -1.0;
    q_volume_elements.back()(1) = 1.224646799147353e-16;
    q_volume_elements.back()(2) = -0.5;
    q_volume_elements.back()(3) = 0.2;
    q_volume_elements.back()(4) = 0.3;
    q_volume_elements.back()(5) = 0.6;
    q_volume_elements.back()(6) = 0.5;
    q_volume_elements.back()(7) = -0.8660254037844386;
    q_volume_elements.back()(8) = -0.5;
    q_volume_elements.back()(9) = 0.5;
    q_volume_elements.back()(10) = 0.8660254037844386;
    q_volume_elements.back()(11) = -0.5;
    q_volume_elements.back()(12) = -0.5;
    q_volume_elements.back()(13) = 6.123233995736765e-17;
    q_volume_elements.back()(14) = 0.0;
    q_volume_elements.back()(15) = 0.45;
    q_volume_elements.back()(16) = -0.1330127018922193;
    q_volume_elements.back()(17) = 0.1;
    q_volume_elements.back()(18) = -0.25;
    q_volume_elements.back()(19) = -0.43301270189221924;
    q_volume_elements.back()(20) = -0.5;
    q_volume_elements.back()(21) = -0.25;
    q_volume_elements.back()(22) = 0.43301270189221935;
    q_volume_elements.back()(23) = -0.5;
    q_volume_elements.back()(24) = 0.15;
    q_volume_elements.back()(25) = 0.6330127018922194;
    q_volume_elements.back()(26) = 0.05;
    q_volume_elements.back()(27) = 0.5;
    q_volume_elements.back()(28) = 0.0;
    q_volume_elements.back()(29) = -0.5;
  }

  /**
   * \brief The following code creates the geometry for the nurbs test.
   *
   * The solid is a 90degree element of a hollow cylinder.
   */
  void xtest_multiple_intersections_nurbs27_geometry(
      std::vector<std::shared_ptr<Core::Elements::Element>>& line_elements,
      std::vector<std::shared_ptr<Core::Elements::Element>>& volume_elements,
      std::vector<Core::LinAlg::Matrix<12, 1, double>>& q_line_elements,
      std::vector<double>& line_ref_lengths,
      std::vector<Core::LinAlg::Matrix<81, 1, double>>& q_volume_elements,
      std::shared_ptr<Core::FE::Nurbs::NurbsDiscretization> structdis)
  {
    // Create the elements. In this case the volume has to be first, as otherwise the nurbs patches
    // would need a different numbering.
    const int dummy_node_ids[2] = {0, 1};
    line_elements.push_back(std::make_shared<Discret::Elements::Beam3r>(1, 0));
    line_elements.back()->set_node_ids(2, dummy_node_ids);
    volume_elements.push_back(std::make_shared<Discret::Elements::Solid>(0, 0));

    // Positional and tangent DOFs of the line(s).
    Core::LinAlg::Matrix<12, 1, double> q_line(Core::LinAlg::Initialization::zero);
    q_line(0) = -0.05;
    q_line(1) = 0.05;
    q_line(2) = 0.3;
    q_line(3) = 0.5773502691896255;
    q_line(4) = 0.5773502691896258;
    q_line(5) = 0.577350269189626;
    q_line(6) = 0.45;
    q_line(7) = -0.05;
    q_line(8) = 0.1;
    q_line(9) = 0.8017837257372733;
    q_line(10) = -0.5345224838248488;
    q_line(11) = 0.2672612419124244;
    q_line_elements.push_back(q_line);

    line_ref_lengths.push_back(0.61920435714496047108);

    // Positional DOFs of the solid(s).
    Core::LinAlg::Matrix<81, 1, double> q_volume(Core::LinAlg::Initialization::zero);
    q_volume(0) = 0.0;
    q_volume(1) = 0.1;
    q_volume(2) = 0.0;
    q_volume(3) = 0.1;
    q_volume(4) = 0.1;
    q_volume(5) = 0.0;
    q_volume(6) = 0.1;
    q_volume(7) = 0.0;
    q_volume(8) = 0.0;
    q_volume(9) = 0.0;
    q_volume(10) = 0.15;
    q_volume(11) = 0.0;
    q_volume(12) = 0.15;
    q_volume(13) = 0.15;
    q_volume(14) = 0.0;
    q_volume(15) = 0.15;
    q_volume(16) = 0.0;
    q_volume(17) = 0.0;
    q_volume(18) = 0.0;
    q_volume(19) = 0.2;
    q_volume(20) = 0.0;
    q_volume(21) = 0.2;
    q_volume(22) = 0.2;
    q_volume(23) = 0.0;
    q_volume(24) = 0.2;
    q_volume(25) = 0.0;
    q_volume(26) = 0.0;
    q_volume(27) = 0.0;
    q_volume(28) = 0.1;
    q_volume(29) = 0.15;
    q_volume(30) = 0.1;
    q_volume(31) = 0.1;
    q_volume(32) = 0.15;
    q_volume(33) = 0.1;
    q_volume(34) = 0.0;
    q_volume(35) = 0.15;
    q_volume(36) = 0.0;
    q_volume(37) = 0.15;
    q_volume(38) = 0.15;
    q_volume(39) = 0.15;
    q_volume(40) = 0.15;
    q_volume(41) = 0.15;
    q_volume(42) = 0.15;
    q_volume(43) = 0.0;
    q_volume(44) = 0.15;
    q_volume(45) = 0.0;
    q_volume(46) = 0.2;
    q_volume(47) = 0.15;
    q_volume(48) = 0.2;
    q_volume(49) = 0.2;
    q_volume(50) = 0.15;
    q_volume(51) = 0.2;
    q_volume(52) = 0.0;
    q_volume(53) = 0.15;
    q_volume(54) = 0.0;
    q_volume(55) = 0.1;
    q_volume(56) = 0.3;
    q_volume(57) = 0.1;
    q_volume(58) = 0.1;
    q_volume(59) = 0.3;
    q_volume(60) = 0.1;
    q_volume(61) = 0.0;
    q_volume(62) = 0.3;
    q_volume(63) = 0.0;
    q_volume(64) = 0.15;
    q_volume(65) = 0.3;
    q_volume(66) = 0.15;
    q_volume(67) = 0.15;
    q_volume(68) = 0.3;
    q_volume(69) = 0.15;
    q_volume(70) = 0.0;
    q_volume(71) = 0.3;
    q_volume(72) = 0.0;
    q_volume(73) = 0.2;
    q_volume(74) = 0.3;
    q_volume(75) = 0.2;
    q_volume(76) = 0.2;
    q_volume(77) = 0.3;
    q_volume(78) = 0.2;
    q_volume(79) = 0.0;
    q_volume(80) = 0.3;
    q_volume_elements.push_back(q_volume);

    // Set up the needed structure for a nurbs discretization.
    std::shared_ptr<Core::FE::Nurbs::Knotvector> knot_vector =
        std::make_shared<Core::FE::Nurbs::Knotvector>(3, 1);

    // Set the knotvector.
    const std::string knotvectortype = "Interpolated";
    for (unsigned int dir = 0; dir < 3; dir++)
    {
      std::vector<double> directions_knots;
      directions_knots.push_back(0.);
      directions_knots.push_back(0.);
      directions_knots.push_back(0.);
      directions_knots.push_back(1.);
      directions_knots.push_back(1.);
      directions_knots.push_back(1.);
      knot_vector->set_knots(dir, 0, 2, 6, knotvectortype, directions_knots);
    }
    knot_vector->finish_knots(0);
    structdis->set_knot_vector(knot_vector);

    // Set the control points.
    std::vector<double> weights(27);
    weights[0] = 1.0;
    weights[1] = 0.707107;
    weights[2] = 1.0;
    weights[3] = 1.0;
    weights[4] = 0.707107;
    weights[5] = 1.0;
    weights[6] = 1.0;
    weights[7] = 0.707107;
    weights[8] = 1.0;
    weights[9] = 1.0;
    weights[10] = 0.707107;
    weights[11] = 1.0;
    weights[12] = 1.0;
    weights[13] = 0.707107;
    weights[14] = 1.0;
    weights[15] = 1.0;
    weights[16] = 0.707107;
    weights[17] = 1.0;
    weights[18] = 1.0;
    weights[19] = 0.707107;
    weights[20] = 1.0;
    weights[21] = 1.0;
    weights[22] = 0.707107;
    weights[23] = 1.0;
    weights[24] = 1.0;
    weights[25] = 0.707107;
    weights[26] = 1.0;
    int nodes[27];
    std::map<int, std::shared_ptr<Core::Nodes::Node>> nodes_map;
    for (unsigned int i_node = 0; i_node < 27; i_node++)
    {
      nodes[i_node] = (int)i_node;
      std::vector<double> dummycoord = {0., 0., 0., 0., 0., 0.};
      std::shared_ptr<Core::Nodes::Node> new_node =
          std::make_shared<Core::FE::Nurbs::ControlPoint>(i_node, dummycoord, weights[i_node], 0);
      structdis->add_node(new_node);
      nodes_map[i_node] = new_node;
    }

    // Set the nodes in the element.
    volume_elements.back()->set_node_ids(27, nodes);
    volume_elements.back()->build_nodal_pointers(nodes_map);
  }

  /**
   * \brief This example contains a unit cube with its center at the origin and a pre-curved
   line.
   *
   * For more details on the pre-curved line have a look at the script
   * script/unittest_geometry_pair_line_to_volume.wls
   */
  void xtest_create_geometry_single_hex8_with_pre_curved_line(
      std::vector<std::shared_ptr<Core::Elements::Element>>& line_elements,
      std::vector<std::shared_ptr<Core::Elements::Element>>& volume_elements,
      std::vector<Core::LinAlg::Matrix<12, 1, double>>& q_line_elements,
      std::vector<double>& line_ref_lengths,
      std::vector<Core::LinAlg::Matrix<24, 1, double>>& q_volume_elements)
  {
    // Create the elements.
    const std::vector<int> dummy_node_ids{0, 1};
    line_elements.push_back(std::make_shared<Discret::Elements::Beam3r>(0, 0));
    line_elements.back()->set_node_ids(2, dummy_node_ids.data());
    volume_elements.push_back(std::make_shared<Discret::Elements::Solid>(1, 0));

    // Positional and tangent DOFs of the line(s).
    q_line_elements.push_back(Core::LinAlg::Matrix<12, 1, double>());
    q_line_elements.back()(0) = 0.485;
    q_line_elements.back()(1) = 0.0;
    q_line_elements.back()(2) = 0.0;

    q_line_elements.back()(3) = 0.985329278164293;
    q_line_elements.back()(4) = 0.1706640371965724;
    q_line_elements.back()(5) = 0.0;

    q_line_elements.back()(6) = 0.505;
    q_line_elements.back()(7) = 0.5;
    q_line_elements.back()(8) = 0.0;

    q_line_elements.back()(9) = 0.985329278164293;
    q_line_elements.back()(10) = 0.1706640371965724;
    q_line_elements.back()(11) = 0.0;

    // All nodes have 0 rotation (the middle node does not matter in this case)
    line_ref_lengths.push_back(0.58637598747637864616);

    // Positional DOFs of the solid(s).
    q_volume_elements.push_back(Core::LinAlg::Matrix<24, 1, double>());
    q_volume_elements.back()(0) = -0.5;
    q_volume_elements.back()(1) = -0.5;
    q_volume_elements.back()(2) = -0.5;

    q_volume_elements.back()(3) = 0.5;
    q_volume_elements.back()(4) = -0.5;
    q_volume_elements.back()(5) = -0.5;

    q_volume_elements.back()(6) = 0.5;
    q_volume_elements.back()(7) = 0.5;
    q_volume_elements.back()(8) = -0.5;

    q_volume_elements.back()(9) = -0.5;
    q_volume_elements.back()(10) = 0.5;
    q_volume_elements.back()(11) = -0.5;

    q_volume_elements.back()(12) = -0.5;
    q_volume_elements.back()(13) = -0.5;
    q_volume_elements.back()(14) = 0.5;

    q_volume_elements.back()(15) = 0.5;
    q_volume_elements.back()(16) = -0.5;
    q_volume_elements.back()(17) = 0.5;

    q_volume_elements.back()(18) = 0.5;
    q_volume_elements.back()(19) = 0.5;
    q_volume_elements.back()(20) = 0.5;

    q_volume_elements.back()(21) = -0.5;
    q_volume_elements.back()(22) = 0.5;
    q_volume_elements.back()(23) = 0.5;
  }

}  // namespace


#endif

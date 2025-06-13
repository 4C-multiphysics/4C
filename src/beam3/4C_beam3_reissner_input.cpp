// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beam3_reissner.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_legacy_enum_definitions_materials.hpp"
#include "4C_mat_beam_material_generic.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
bool Discret::Elements::Beam3r::read_element(const std::string& eletype, const std::string& distype,
    const Core::IO::InputParameterContainer& container)
{
  /* the triad field is discretized with Lagrange polynomials of order num_node()-1;
   * the centerline is either discretized in the same way or with 3rd order Hermite polynomials;
   * we thus make a difference between nnodetriad and nnodecl;
   * assumptions: nnodecl<=nnodetriad
   * first nodes with local ID 0...nnodecl-1 are used for interpolation of centerline AND triad
   * field*/
  const int nnodetriad = num_node();

  // read number of material model and cross-section specs
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  const auto mat_type = material()->parameter()->type();
  FOUR_C_ASSERT_ALWAYS(mat_type == Core::Materials::m_beam_reissner_elast_hyper ||
                           mat_type == Core::Materials::m_beam_reissner_elast_plastic ||
                           mat_type == Core::Materials::m_beam_reissner_elast_hyper_bymodes,
      "The material parameter definition '{}' is not supported by Beam3r element! "
      "Choose MAT_BeamReissnerElastHyper, MAT_BeamReissnerElastHyper_ByModes or "
      "MAT_BeamReissnerElastPlastic!",
      mat_type);


  centerline_hermite_ = container.get<bool>("HERMITE_CENTERLINE");

  // read whether automatic differentiation via Sacado::Fad package shall be used
  use_fad_ = container.get<bool>("USE_FAD");


  // store nodal triads according to input file
  theta0node_.resize(nnodetriad);

  /* Attention! expression "TRIADS" in input file is misleading.
   * The 3 specified values per node define a rotational pseudovector, which
   * parameterizes the orientation of the triad at this node
   * (relative to the global reference coordinate system)*/
  /* extract rotational pseudovectors at element nodes in reference configuration
   *  and save them as quaternions at each node, respectively*/
  auto nodal_rotvecs = container.get<std::vector<double>>("TRIADS");

  for (int node = 0; node < nnodetriad; node++)
    for (int dim = 0; dim < 3; dim++) theta0node_[node](dim) = nodal_rotvecs[3 * node + dim];

  Core::FE::IntegrationPoints1D gausspoints_force(my_gauss_rule(res_elastic_force));
  Core::FE::IntegrationPoints1D gausspoints_moment(my_gauss_rule(res_elastic_moment));

  get_beam_material().setup(gausspoints_force.num_points(), gausspoints_moment.num_points());

  return true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3r::set_centerline_hermite(const bool yesno)
{
  centerline_hermite_ = yesno;
}

FOUR_C_NAMESPACE_CLOSE

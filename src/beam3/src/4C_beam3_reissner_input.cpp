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
    const Core::IO::InputParameterContainer& container,
    const Core::IO::MeshInput::ElementDataFromCellData& element_data)
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

  // Store the nodal rotation vectors
  std::vector<double> nodal_rotation_vectors(3 * nnodetriad);

  if (container.get_if<std::vector<double>>("TRIADS"))
  {
    nodal_rotation_vectors = container.get<std::vector<double>>("TRIADS");
  }
  else if (container.get_if<std::string>("NODAL_ROTATION_VECTORS"))
  {
    const auto triad_field_name = container.get<std::string>("NODAL_ROTATION_VECTORS");
    switch (nnodetriad)
    {
      case 2:
      {
        // TODO: We currently have to store the triad in a symmetric tensor since that one takes 6
        // components. This should be changed in the future, once we can read plain vectors from
        // the mesh.
        const auto& triad_field_data =
            element_data.get<Core::LinAlg::SymmetricTensor<double, 3, 3>>(triad_field_name);
        // In the input data we define a cell data field with 6 components. For now they internally
        // get converted to a symmetric tensor, the following mapping extracts the correct order as
        // given in the input data.
        nodal_rotation_vectors[0] = triad_field_data(0, 0);
        nodal_rotation_vectors[1] = triad_field_data(1, 1);
        nodal_rotation_vectors[2] = triad_field_data(2, 2);
        nodal_rotation_vectors[3] = triad_field_data(0, 1);
        nodal_rotation_vectors[4] = triad_field_data(1, 2);
        nodal_rotation_vectors[5] = triad_field_data(0, 2);
        break;
      }
      case 3:
      {
        // TODO: We currently have to store the triad in a tensor since that one takes 9
        // components. This should be changed in the future, once we can read plain vectors from
        // the mesh.
        const auto& triad_field_data =
            element_data.get<Core::LinAlg::Tensor<double, 3, 3>>(triad_field_name);
        nodal_rotation_vectors[0] = triad_field_data(0, 0);
        nodal_rotation_vectors[1] = triad_field_data(0, 1);
        nodal_rotation_vectors[2] = triad_field_data(0, 2);
        nodal_rotation_vectors[3] = triad_field_data(1, 0);
        nodal_rotation_vectors[4] = triad_field_data(1, 1);
        nodal_rotation_vectors[5] = triad_field_data(1, 2);
        nodal_rotation_vectors[6] = triad_field_data(2, 0);
        nodal_rotation_vectors[7] = triad_field_data(2, 1);
        nodal_rotation_vectors[8] = triad_field_data(2, 2);
        break;
      }
      default:
      {
        FOUR_C_THROW(
            "Nodal triad definition from mesh is only implemented for up to 2 and 3 nodes per "
            "element, but {} nodes are given for beam3r element.",
            nnodetriad);
      }
    }
  }
  else
  {
    FOUR_C_THROW(
        "No definition for nodal triads provided! Please set either TRIADS or "
        "NODAL_ROTATION_VECTORS for beam3r elements!");
  }

  theta0node_.resize(nnodetriad);
  for (int node = 0; node < nnodetriad; node++)
    for (int dim = 0; dim < 3; dim++)
      theta0node_[node](dim) = nodal_rotation_vectors[3 * node + dim];

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

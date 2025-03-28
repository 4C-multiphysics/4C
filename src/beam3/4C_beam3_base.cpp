// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beam3_base.hpp"

#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_browniandyn_input.hpp"  // enums
#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_geometric_search_bounding_volume.hpp"
#include "4C_fem_geometric_search_params.hpp"
#include "4C_fem_geometry_periodic_boundingbox.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_beam_templated_material_generic.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::Beam3Base::Beam3Base(int id, int owner)
    : Core::Elements::Element(id, owner),
      Tref_(0),
      centerline_hermite_(true),
      filamenttype_(Inpar::BeamInteraction::filetype_none),
      interface_ptr_(nullptr),
      browndyn_interface_ptr_(nullptr)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::Beam3Base::Beam3Base(const Discret::Elements::Beam3Base& old)
    : Core::Elements::Element(old),
      Tref_(old.Tref_),
      centerline_hermite_(old.centerline_hermite_),
      bspotposxi_(old.bspotposxi_),
      filamenttype_(old.filamenttype_)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3Base::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Element::pack(data);

  // bspotposxi_
  add_to_pack(data, bspotposxi_);
  // filamenttype_
  add_to_pack(data, filamenttype_);

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3Base::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);

  // bspotposxi_
  extract_from_pack(buffer, bspotposxi_);
  // filamenttype_
  extract_from_pack(buffer, filamenttype_);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3Base::set_params_interface_ptr(const Teuchos::ParameterList& p)
{
  if (p.isParameter("interface"))
    interface_ptr_ = std::dynamic_pointer_cast<Solid::Elements::ParamsInterface>(
        p.get<std::shared_ptr<Core::Elements::ParamsInterface>>("interface"));
  else
    interface_ptr_ = nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3Base::set_brownian_dyn_params_interface_ptr()
{
  browndyn_interface_ptr_ = interface_ptr_->get_brownian_dyn_param_interface();

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::ParamsInterface>
Discret::Elements::Beam3Base::params_interface_ptr()
{
  return interface_ptr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<BrownianDynamics::ParamsInterface>
Discret::Elements::Beam3Base::brownian_dyn_params_interface_ptr() const
{
  return browndyn_interface_ptr_;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
std::vector<int> Discret::Elements::Beam3Base::get_additive_dof_gids(
    const Core::FE::Discretization& discret, const Core::Nodes::Node& node) const
{
  std::vector<int> dofgids;
  std::vector<int> dofindices;

  // first collect all DoF indices
  this->position_dof_indices(dofindices, node);
  this->tangent_dof_indices(dofindices, node);
  this->rotation_1d_dof_indices(dofindices, node);
  this->tangent_length_dof_indices(dofindices, node);

  // now ask for the GIDs of the DoFs with collected local indices
  dofgids.reserve(dofindices.size());
  for (unsigned int i = 0; i < dofindices.size(); ++i)
    dofgids.push_back(discret.dof(0, &node, dofindices[i]));

  return dofgids;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::vector<int> Discret::Elements::Beam3Base::get_rot_vec_dof_gids(
    const Core::FE::Discretization& discret, const Core::Nodes::Node& node) const
{
  std::vector<int> dofgids;
  std::vector<int> dofindices;

  // first collect all DoF indices
  this->rotation_vec_dof_indices(dofindices, node);

  // now ask for the GIDs of the DoFs with collected local indices
  dofgids.reserve(dofindices.size());
  for (unsigned int i = 0; i < dofindices.size(); ++i)
    dofgids.push_back(discret.dof(0, &node, dofindices[i]));

  return dofgids;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
double Discret::Elements::Beam3Base::get_circular_cross_section_radius_for_interactions() const
{
  return get_beam_material().get_interaction_radius();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3Base::get_ref_pos_at_xi(
    Core::LinAlg::Matrix<3, 1>& refpos, const double& xi) const
{
  const int numclnodes = this->num_centerline_nodes();
  const int numnodalvalues = this->hermite_centerline_interpolation() ? 2 : 1;

  std::vector<double> zerovec;
  zerovec.resize(3 * numnodalvalues * numclnodes);

  this->get_pos_at_xi(refpos, xi, zerovec);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
Mat::BeamMaterial& Discret::Elements::Beam3Base::get_beam_material() const
{
  // get the material law
  std::shared_ptr<Core::Mat::Material> material_ptr = material();

  if (material_ptr->material_type() != Core::Materials::m_beam_elast_hyper_generic)
    FOUR_C_THROW("unknown or improper type of material law! expected beam material law!");

  return *static_cast<Mat::BeamMaterial*>(material_ptr.get());
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/

template <typename T>
Mat::BeamMaterialTemplated<T>& Discret::Elements::Beam3Base::get_templated_beam_material() const
{
  return *std::dynamic_pointer_cast<Mat::BeamMaterialTemplated<T>>(material());
};


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Discret::Elements::Beam3Base::get_constitutive_matrices(
    Core::LinAlg::Matrix<3, 3, T>& CN, Core::LinAlg::Matrix<3, 3, T>& CM) const
{
  get_templated_beam_material<T>().get_constitutive_matrix_of_forces_material_frame(CN);
  get_templated_beam_material<T>().get_constitutive_matrix_of_moments_material_frame(CM);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Discret::Elements::Beam3Base::get_translational_and_rotational_mass_inertia_tensor(
    double& mass_inertia_translational, Core::LinAlg::Matrix<3, 3, T>& J) const
{
  get_translational_mass_inertia_factor(mass_inertia_translational);
  get_beam_material().get_mass_moment_of_inertia_tensor_material_frame(J);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3Base::get_translational_mass_inertia_factor(
    double& mass_inertia_translational) const
{
  mass_inertia_translational = get_beam_material().get_translational_mass_inertia_factor();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Beam3Base::get_damping_coefficients(Core::LinAlg::Matrix<3, 1>& gamma) const
{
  switch (brownian_dyn_params_interface().how_beam_damping_coefficients_are_specified())
  {
    case BrownianDynamics::cylinder_geometry_approx:
    {
      /* These are coefficients for a straight cylindrical rod taken from
       * Howard, p. 107, table 6.2. The order is as follows:
       * (0) damping of translation parallel to axis,
       * (1) damping of translation orthogonal to axis,
       * (2) damping of rotation around its own axis */

      gamma(0) = 2.0 * M_PI * brownian_dyn_params_interface().get_viscosity();
      gamma(1) = 4.0 * M_PI * brownian_dyn_params_interface().get_viscosity();
      gamma(2) = 4.0 * M_PI * brownian_dyn_params_interface().get_viscosity() *
                 get_circular_cross_section_radius_for_interactions() *
                 get_circular_cross_section_radius_for_interactions();

      // huge improvement in convergence of non-linear solver in case of artificial factor 4000
      //      gamma(2) *= 4000.0;

      break;
    }

    case BrownianDynamics::input_file:
    {
      gamma(0) = brownian_dyn_params_interface()
                     .get_beam_damping_coefficient_prefactors_from_input_file()[0] *
                 brownian_dyn_params_interface().get_viscosity();
      gamma(1) = brownian_dyn_params_interface()
                     .get_beam_damping_coefficient_prefactors_from_input_file()[1] *
                 brownian_dyn_params_interface().get_viscosity();
      gamma(2) = brownian_dyn_params_interface()
                     .get_beam_damping_coefficient_prefactors_from_input_file()[2] *
                 brownian_dyn_params_interface().get_viscosity();

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid choice of how damping coefficient values for beams are specified!");

      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <unsigned int ndim, typename T>
void Discret::Elements::Beam3Base::get_background_velocity(
    Teuchos::ParameterList& params,  //!< parameter list
    const Core::LinAlg::Matrix<ndim, 1, T>&
        evaluationpoint,                              //!< point at which background velocity and
                                                      //!< its gradient has to be computed
    Core::LinAlg::Matrix<ndim, 1, T>& velbackground,  //!< velocity of background fluid
    Core::LinAlg::Matrix<ndim, ndim, T>& velbackgroundgrad)
    const  //!< gradient of velocity of background fluid
{
  /*note: this function is not yet a general one, but always assumes a shear flow, where the
   * velocity of the background fluid is always directed in direction
   * params.get<int>("DBCDISPDIR",0) and orthogonal to z-axis. In 3D the velocity increases linearly
   * in z and equals zero for z = 0.
   * In 2D the velocity increases linearly in y and equals zero for y = 0. */

  // default values for background velocity and its gradient
  velbackground.put_scalar(0.0);
  velbackgroundgrad.put_scalar(0.0);
}

/*-----------------------------------------------------------------------------*
 | shifts nodes so that proper evaluation is possible even in case of          |
 | periodic boundary conditions; if two nodes within one element are se-       |
 | parated by a periodic boundary, one of them is shifted such that the final  |
 | distance in R^3 is the same as the initial distance in the periodic         |
 | space; the shift affects computation on element level within that           |
 | iteration step, only (no change in global variables performed)              |
 *-----------------------------------------------------------------------------*/
void Discret::Elements::Beam3Base::un_shift_node_position(
    std::vector<double>& disp, Core::Geo::MeshFree::BoundingBox const& periodic_boundingbox) const
{
  /* get number of degrees of freedom per node; note:
   * the following function assumes the same number of degrees
   * of freedom for each element node*/
  int numdof = num_dof_per_node(*(nodes()[0]));

  // get number of nodes that are used for centerline interpolation
  unsigned int nnodecl = num_centerline_nodes();

  // loop through all nodes except for the first node which remains
  // fixed as reference node
  static Core::LinAlg::Matrix<3, 1> d(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1> ref(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1> X(Core::LinAlg::Initialization::zero);
  for (unsigned int i = 1; i < nnodecl; ++i)
  {
    for (int dim = 0; dim < 3; ++dim)
    {
      d(dim) = disp[numdof * i + dim];
      ref(dim) = nodes()[0]->x()[dim] + disp[numdof * 0 + dim];
      X(dim) = nodes()[i]->x()[dim];
    }

    periodic_boundingbox.un_shift_3d(d, ref, X);

    for (unsigned int dim = 0; dim < 3; ++dim)
    {
      disp[numdof * i + dim] = d(dim);
    }
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void Discret::Elements::Beam3Base::get_directions_of_shifts(std::vector<double>& disp,
    Core::Geo::MeshFree::BoundingBox const& periodic_boundingbox,
    std::vector<bool>& shift_in_dim) const
{
  /* get number of degrees of freedom per node; note:
   * the following function assumes the same number of degrees
   * of freedom for each element node*/
  int numdof = num_dof_per_node(*(nodes()[0]));
  // get number of nodes that are used for centerline interpolation
  unsigned int nnodecl = num_centerline_nodes();

  shift_in_dim.clear();
  shift_in_dim.resize(3);

  // loop through all nodes except for the first node which remains
  // fixed as reference node
  static Core::LinAlg::Matrix<3, 1> d(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1> ref(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1> X(Core::LinAlg::Initialization::zero);
  for (unsigned int i = 1; i < nnodecl; ++i)
  {
    for (int dim = 0; dim < 3; ++dim)
    {
      d(dim) = disp[numdof * i + dim];
      ref(dim) = nodes()[0]->x()[dim] + disp[numdof * 0 + dim];
      X(dim) = nodes()[i]->x()[dim];
    }

    periodic_boundingbox.check_if_shift_between_points(d, ref, shift_in_dim, X);

    for (unsigned int dim = 0; dim < 3; ++dim)
    {
      disp[numdof * i + dim] = d(dim);
    }
  }
}

/*--------------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3Base::get_pos_of_binding_spot(Core::LinAlg::Matrix<3, 1>& pos,
    std::vector<double>& disp, Inpar::BeamInteraction::CrosslinkerType linkertype, int bspotlocn,
    Core::Geo::MeshFree::BoundingBox const& periodic_boundingbox) const
{
  const double xi = bspotposxi_.at(linkertype)[bspotlocn];
  // get position
  get_pos_at_xi(pos, xi, disp);

  // check if pos at xi lies outside the periodic box, if it does, shift it back in
  periodic_boundingbox.shift_3d(pos);
}

/*--------------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3Base::get_triad_of_binding_spot(Core::LinAlg::Matrix<3, 3>& triad,
    std::vector<double>& disp, Inpar::BeamInteraction::CrosslinkerType linkertype,
    int bspotlocn) const
{
  const double xi = bspotposxi_.at(linkertype)[bspotlocn];
  // get position
  get_triad_at_xi(triad, xi, disp);
}

/*--------------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------------*/
Core::GeometricSearch::BoundingVolume Discret::Elements::Beam3Base::get_bounding_volume(
    const Core::FE::Discretization& discret,
    const Core::LinAlg::Vector<double>& result_data_dofbased,
    const Core::GeometricSearch::GeometricSearchParams& params) const
{
  // Get the centerline dof values of the beam.
  std::vector<double> element_posdofvec;
  BeamInteraction::Utils::extract_pos_dof_vec_values(
      discret, this, result_data_dofbased, element_posdofvec);
  Core::GeometricSearch::BoundingVolume bounding_volume;

  Core::LinAlg::Matrix<3, 1, double> point;

  // TODO: replace this with convex hull from bezier curve (small student project?)
  // Add a certain number of points along the beam.
  const unsigned int n_points = 5;
  for (unsigned int i_point = 0; i_point < n_points; ++i_point)
  {
    const double xi = -1.0 + 2.0 / (n_points - 1) * i_point;
    this->get_pos_at_xi(point, xi, element_posdofvec);
    bounding_volume.add_point(point);
  }

  // Add the radius times a safety factor.
  const double safety_factor = params.get_beam_bounding_volume_scaling();
  const double radius = get_circular_cross_section_radius_for_interactions();
  bounding_volume.extend_boundaries(radius * safety_factor);

  return bounding_volume;
}

/*--------------------------------------------------------------------------------------------*
 | explicit template instantiations                                                           |
 *--------------------------------------------------------------------------------------------*/
template void Discret::Elements::Beam3Base::get_constitutive_matrices<double>(
    Core::LinAlg::Matrix<3, 3, double>& CN, Core::LinAlg::Matrix<3, 3, double>& CM) const;
template void Discret::Elements::Beam3Base::get_constitutive_matrices<Sacado::Fad::DFad<double>>(
    Core::LinAlg::Matrix<3, 3, Sacado::Fad::DFad<double>>& CN,
    Core::LinAlg::Matrix<3, 3, Sacado::Fad::DFad<double>>& CM) const;

template void
Discret::Elements::Beam3Base::get_translational_and_rotational_mass_inertia_tensor<double>(
    double&, Core::LinAlg::Matrix<3, 3, double>&) const;
template void Discret::Elements::Beam3Base::get_translational_and_rotational_mass_inertia_tensor<
    Sacado::Fad::DFad<double>>(
    double&, Core::LinAlg::Matrix<3, 3, Sacado::Fad::DFad<double>>&) const;

template void Discret::Elements::Beam3Base::get_background_velocity<3, double>(
    Teuchos::ParameterList&, const Core::LinAlg::Matrix<3, 1, double>&,
    Core::LinAlg::Matrix<3, 1, double>&, Core::LinAlg::Matrix<3, 3, double>&) const;
template void Discret::Elements::Beam3Base::get_background_velocity<3, Sacado::Fad::DFad<double>>(
    Teuchos::ParameterList&, const Core::LinAlg::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    Core::LinAlg::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    Core::LinAlg::Matrix<3, 3, Sacado::Fad::DFad<double>>&) const;

template Mat::BeamMaterialTemplated<double>&
Discret::Elements::Beam3Base::get_templated_beam_material<double>() const;
template Mat::BeamMaterialTemplated<Sacado::Fad::DFad<double>>&
Discret::Elements::Beam3Base::get_templated_beam_material<Sacado::Fad::DFad<double>>() const;

FOUR_C_NAMESPACE_CLOSE

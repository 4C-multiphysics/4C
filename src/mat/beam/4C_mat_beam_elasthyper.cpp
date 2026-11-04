// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_beam_elasthyper.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_beam_elasthyper_parameter.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN



/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
Core::Communication::ParObject* Mat::BeamElastHyperMaterialType<T>::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Core::Mat::Material* matobject = new Mat::BeamElastHyperMaterial<T>();
  matobject->unpack(buffer);
  return matobject;
}



/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
Mat::BeamElastHyperMaterial<T>::BeamElastHyperMaterial() : params_(nullptr)
{
  // empty constructor
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
Mat::BeamElastHyperMaterial<T>::BeamElastHyperMaterial(
    Mat::PAR::BeamElastHyperMaterialParameterGeneric* params)
    : params_(params)
{
  // empty constructor
}



/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamElastHyperMaterial<T>::evaluate_force_contributions_to_stress(
    Core::LinAlg::Matrix<3, 1, T>& stressN, const Core::LinAlg::Matrix<3, 3, T>& CN,
    const Core::LinAlg::Matrix<3, 1, T>& Gamma, const unsigned int gp)
{
  // compute material stresses by multiplying strains with constitutive matrix
  stressN.multiply(CN, Gamma);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamElastHyperMaterial<T>::evaluate_moment_contributions_to_stress(
    Core::LinAlg::Matrix<3, 1, T>& stressM, const Core::LinAlg::Matrix<3, 3, T>& CM,
    const Core::LinAlg::Matrix<3, 1, T>& Cur, const unsigned int gp)
{
  // compute material stresses by multiplying curvature with constitutive matrix
  stressM.multiply(CM, Cur);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamElastHyperMaterial<T>::compute_constitutive_parameter(
    Core::LinAlg::Matrix<3, 3, T>& C_N, Core::LinAlg::Matrix<3, 3, T>& C_M)
{
  // setup constitutive matrices
  Mat::BeamElastHyperMaterial<T>::get_constitutive_matrix_of_forces_material_frame(C_N);
  Mat::BeamElastHyperMaterial<T>::get_constitutive_matrix_of_moments_material_frame(C_M);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamElastHyperMaterial<T>::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamElastHyperMaterial<T>::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;

  if (Global::Problem::instance()->materials() != nullptr)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();

      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);

      /* the idea is that we have a generic type of material (this class), but various
       * possible sets of material parameters to 'feed' these very general constitutive relations */
      if (mat->type() == Core::Materials::m_beam_reissner_elast_hyper or
          mat->type() == Core::Materials::m_beam_reissner_elast_hyper_bymodes or
          mat->type() == Core::Materials::m_beam_kirchhoff_elast_hyper or
          mat->type() == Core::Materials::m_beam_reissner_elast_plastic or
          mat->type() == Core::Materials::m_beam_kirchhoff_elast_hyper_bymodes or
          mat->type() == Core::Materials::m_beam_kirchhoff_torsionfree_elast_hyper or
          mat->type() == Core::Materials::m_beam_kirchhoff_torsionfree_elast_hyper_bymodes)
        params_ = static_cast<Mat::PAR::BeamElastHyperMaterialParameterGeneric*>(mat);
      else
        FOUR_C_THROW("Type of material parameter {} does not fit to type of material law {}",
            mat->type(), material_type());
    }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
Core::Mat::PAR::Parameter* Mat::BeamElastHyperMaterial<T>::parameter() const
{
  return params_;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
const Mat::PAR::BeamElastHyperMaterialParameterGeneric& Mat::BeamElastHyperMaterial<T>::params()
    const
{
  if (params_ == nullptr) FOUR_C_THROW("pointer to parameter class is not set!");

  return *params_;
}


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamElastHyperMaterial<T>::get_constitutive_matrix_of_forces_material_frame(
    Core::LinAlg::Matrix<3, 3, T>& C_N) const
{
  // defining material constitutive matrix CN between Gamma and N
  // according to Jelenic 1999, section 2.4
  C_N.clear();

  C_N(0, 0) = params().get_axial_rigidity();
  C_N(1, 1) = params().get_shear_rigidity2();
  C_N(2, 2) = params().get_shear_rigidity3();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamElastHyperMaterial<T>::get_constitutive_matrix_of_moments_material_frame(
    Core::LinAlg::Matrix<3, 3, T>& C_M) const
{
  // defining material constitutive matrix CM between curvature and moment
  // according to Jelenic 1999, section 2.4
  C_M.clear();

  C_M(0, 0) = params().get_torsional_rigidity();
  C_M(1, 1) = params().get_bending_rigidity2();
  C_M(2, 2) = params().get_bending_rigidity3();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
double Mat::BeamElastHyperMaterial<T>::get_translational_mass_inertia_factor() const
{
  return params().get_translational_mass_inertia();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamElastHyperMaterial<T>::get_mass_moment_of_inertia_tensor_material_frame(
    Core::LinAlg::Matrix<3, 3>& J) const
{
  J.clear();

  J(0, 0) = params().get_polar_mass_moment_of_inertia();
  J(1, 1) = params().get_mass_moment_of_inertia2();
  J(2, 2) = params().get_mass_moment_of_inertia3();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamElastHyperMaterial<T>::get_mass_moment_of_inertia_tensor_material_frame(
    Core::LinAlg::Matrix<3, 3, Sacado::Fad::DFad<double>>& J) const
{
  J.clear();

  J(0, 0) = params().get_polar_mass_moment_of_inertia();
  J(1, 1) = params().get_mass_moment_of_inertia2();
  J(2, 2) = params().get_mass_moment_of_inertia3();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
double Mat::BeamElastHyperMaterial<T>::get_interaction_radius() const
{
  return this->params().get_interaction_radius();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamElastHyperMaterial<T>::get_stiffness_matrix_of_moments(
    Core::LinAlg::Matrix<3, 3, T>& stiffness_matrix, const Core::LinAlg::Matrix<3, 3, T>& C_M,
    const int gp)
{
  stiffness_matrix = C_M;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamElastHyperMaterial<T>::get_stiffness_matrix_of_forces(
    Core::LinAlg::Matrix<3, 3, T>& stiffness_matrix, const Core::LinAlg::Matrix<3, 3, T>& C_N,
    const int gp)
{
  stiffness_matrix = C_N;
}

// explicit template instantiations
template class Mat::BeamElastHyperMaterial<double>;
template class Mat::BeamElastHyperMaterial<Sacado::Fad::DFad<double>>;


template class Mat::BeamElastHyperMaterialType<double>;
template class Mat::BeamElastHyperMaterialType<Sacado::Fad::DFad<double>>;

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_beam3r_plasticity.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_beam_elasthyper_parameter.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_fad.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
Mat::PAR::BeamReissnerElastPlasticMaterialParams::BeamReissnerElastPlasticMaterialParams(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : BeamReissnerElastHyperMaterialParams(matdata),
      yield_stress_n_(matdata.parameters.get<double>("YIELDN")),
      yield_stress_m_(matdata.parameters.get<double>("YIELDM")),
      isohard_modulus_n_(matdata.parameters.get<double>("ISOHARDN")),
      isohard_modulus_m_(matdata.parameters.get<double>("ISOHARDM")),
      torsion_plasticity_(matdata.parameters.get<bool>("TORSIONPLAST"))
{
  if (yield_stress_n_ == -1.0 && yield_stress_m_ == -1.0 && isohard_modulus_n_ == -1.0 &&
      isohard_modulus_m_ == -1.0)
    FOUR_C_THROW("no plasticity material parameter is given; use elastic material instead");

  if (isohard_modulus_n_ <= 0.0) yield_stress_n_ = -1.0;

  if (isohard_modulus_m_ <= 0.0) yield_stress_m_ = -1.0;

  if (torsion_plasticity_ && std::abs(get_youngs_modulus() - 2.0 * get_shear_modulus()) > 1e-9)
  {
    FOUR_C_THROW(
        "Young's modulus must be equal to two times the shear modulus if plasticity for torsional "
        "moments is turned on");
  }

  if (yield_stress_m_ >= 0 && std::abs(get_moment_inertia2() - get_moment_inertia3()) > 1e-9)
    FOUR_C_THROW("area moment of inertia 2 and 3 need to be equal");

  if (torsion_plasticity_ &&
      std::abs(get_moment_inertia_polar() - 2.0 * get_moment_inertia2()) > 1e-9)
  {
    FOUR_C_THROW(
        "polar area moment of inertia needs to be assigned twice the value of area moment of "
        "inertia 2");
  }
}
/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material>
Mat::PAR::BeamReissnerElastPlasticMaterialParams::create_material()
{
  /* all the different parameter sets (Reissner/Kirchhoff/..., 'classic'/'by modes') are used to
   * parameterize the same constitutive relations based on a hyperelastic stored energy function
   * formulated for cross-section resultants which are implemented in BeamElastHyperMaterial */
  std::shared_ptr<Core::Mat::Material> matobject;

  if (uses_fad())
  {
    FOUR_C_THROW(
        "The elastoplastic beam material is not yet implemented to be used with automatic "
        "differentiation!");
  }
  else
    matobject = std::make_shared<Mat::BeamPlasticMaterial<double>>(this);
  return matobject;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
Core::Communication::ParObject* Mat::BeamElastPlasticMaterialType<T>::create(
    Core::Communication::UnpackBuffer& buffer)
{
  // create material from packed data
  Mat::BeamPlasticMaterial<T>* matobject = new Mat::BeamPlasticMaterial<T>();
  matobject->unpack(buffer);
  return matobject;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
Mat::BeamElastPlasticMaterialType<T> Mat::BeamElastPlasticMaterialType<T>::instance_;

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
Mat::BeamPlasticMaterial<T>::BeamPlasticMaterial(
    Mat::PAR::BeamReissnerElastPlasticMaterialParams* params)
    : Mat::BeamElastHyperMaterial<T>(params)
{
  // empty constructor
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/

template <typename T>
void Mat::BeamPlasticMaterial<T>::setup(int numgp_force, int numgp_moment)
{
  c_n_eff_.resize(numgp_force);
  c_m_eff_.resize(numgp_moment);
  gammaplastconv_.resize(numgp_force);
  gammaplastnew_.resize(numgp_force);
  gammaplastaccum_.resize(numgp_force);
  kappaplastconv_.resize(numgp_moment);
  kappaplastnew_.resize(numgp_moment);
  kappaplastaccum_.resize(numgp_moment);
  effyieldstress_n_.resize(numgp_force);
  effyieldstress_m_.resize(numgp_moment);
  delta_kappaplast_.resize(numgp_moment);
  normstress_m_.resize(numgp_moment);
  deltastress_m_.resize(numgp_moment);
  kappaelast_.resize(numgp_moment);
  kappaelastflow_.resize(numgp_moment);
  elastic_curvature_.resize(numgp_moment);
  delta_gammaplast_.resize(numgp_force);
  deltastress_n_.resize(numgp_force);
  stress_n_.resize(numgp_force);

  for (int gp = 0; gp < numgp_force; gp++)
  {
    c_n_eff_[gp] = Core::LinAlg::Matrix<3, 3, T>(Core::LinAlg::Initialization::zero);
    gammaplastconv_[gp] = Core::LinAlg::Matrix<3, 1, T>(Core::LinAlg::Initialization::zero);
    gammaplastnew_[gp] = Core::LinAlg::Matrix<3, 1, T>(Core::LinAlg::Initialization::zero);
    gammaplastaccum_[gp] = 0;
    effyieldstress_n_[gp] = 0;
    delta_kappaplast_[gp] = 0;
    delta_gammaplast_[gp] = Core::LinAlg::Matrix<3, 1, T>(Core::LinAlg::Initialization::zero);
    deltastress_n_[gp] = Core::LinAlg::Matrix<3, 1, T>(Core::LinAlg::Initialization::zero);
    stress_n_[gp] = 0;
  }

  for (int gp = 0; gp < numgp_moment; gp++)
  {
    c_m_eff_[gp] = Core::LinAlg::Matrix<3, 3, T>(Core::LinAlg::Initialization::zero);
    kappaplastconv_[gp] = Core::LinAlg::Matrix<3, 1, T>(Core::LinAlg::Initialization::zero);
    kappaplastnew_[gp] = Core::LinAlg::Matrix<3, 1, T>(Core::LinAlg::Initialization::zero);
    kappaplastaccum_[gp] = 0;
    effyieldstress_m_[gp] = 0;
    normstress_m_[gp] = 0;
    deltastress_m_[gp] = 0;
    kappaelast_[gp] = Core::LinAlg::Matrix<3, 1, T>(Core::LinAlg::Initialization::zero);
    kappaelastflow_[gp] = Core::LinAlg::Matrix<3, 1, T>(Core::LinAlg::Initialization::zero);
    elastic_curvature_[gp] = Core::LinAlg::Matrix<3, 1, T>(Core::LinAlg::Initialization::zero);
  }

  numgp_force_ = numgp_force;
  numgp_moment_ = numgp_moment;
}
/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
// Pack data
template <typename T>
void Mat::BeamPlasticMaterial<T>::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // Pack material id
  int matid = -1;
  if (this->parameter() != nullptr)
    matid = this->params().id();  // in case we are in post-process mode

  add_to_pack(data, matid);

  // Pack all internal variables
  add_to_pack(data, numgp_force_);
  add_to_pack(data, numgp_moment_);
  add_to_pack(data, gammaplastaccum_);
  add_to_pack(data, gammaplastconv_);
  add_to_pack(data, kappaplastaccum_);
  add_to_pack(data, kappaplastconv_);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
// Unpack data
template <typename T>
void Mat::BeamPlasticMaterial<T>::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  int matid;
  extract_from_pack(buffer, matid);

  extract_from_pack(buffer, numgp_force_);
  extract_from_pack(buffer, numgp_moment_);
  this->setup(numgp_force_, numgp_moment_);
  extract_from_pack(buffer, gammaplastaccum_);
  extract_from_pack(buffer, gammaplastconv_);
  extract_from_pack(buffer, kappaplastaccum_);
  extract_from_pack(buffer, kappaplastconv_);

  this->set_parameter(nullptr);

  if (Global::Problem::instance()->materials() != nullptr)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();

      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);


      this->set_parameter(static_cast<Mat::PAR::BeamReissnerElastPlasticMaterialParams*>(mat));
    }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamPlasticMaterial<T>::evaluate_force_contributions_to_stress(
    Core::LinAlg::Matrix<3, 1, T>& stressN, const Core::LinAlg::Matrix<3, 3, T>& CN,
    const Core::LinAlg::Matrix<3, 1, T>& Gamma, const unsigned int gp)
{
  //*************Begin: Plasticity of strains in axial direction

  // If no yielding parameter was given, the material is modeled in a purely elastic manner
  if (this->params().get_yield_stress_n() < 0.0)
  {
    // compute material stresses by multiplying strains with constitutive matrix
    Mat::BeamElastHyperMaterial<T>::evaluate_force_contributions_to_stress(stressN, CN, Gamma, gp);
  }
  else
  {
    // material elastic strain
    Core::LinAlg::Matrix<3, 1, T> Gammaelast(Core::LinAlg::Initialization::zero);

    // compute elastic strain
    for (int i = 0; i < 3; i++)
    {
      Gammaelast(i) = Gamma(i) - gammaplastconv_[gp](i);
    }
    // compute resulting stress
    stressN.multiply(CN, Gammaelast);

    // check if yield stress is surpassed
    if (std::abs(stressN(0)) > effyieldstress_n_[gp])
    {
      // compute plastic strain increment
      deltastress_n_[gp](0) = std::abs(stressN(0)) - effyieldstress_n_[gp];

      delta_gammaplast_[gp](0) =
          ((CN(0, 0) - c_n_eff_[gp](0, 0)) / CN(0, 0) * deltastress_n_[gp](0) / CN(0, 0)) *
          Core::FADUtils::signum(stressN(0));

      gammaplastnew_[gp](0) = gammaplastconv_[gp](0) + delta_gammaplast_[gp](0);

      // update elastic strain and stress
      for (int i = 0; i < 3; i++)
      {
        Gammaelast(i) = Gamma(i) - gammaplastnew_[gp](i);
      }

      stressN.multiply(CN, Gammaelast);
    }
    stress_n_[gp] = std::abs(stressN(0));
  }
  //*************End: Plasticity of strains in axial direction
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamPlasticMaterial<T>::evaluate_moment_contributions_to_stress(
    Core::LinAlg::Matrix<3, 1, T>& stressM, const Core::LinAlg::Matrix<3, 3, T>& CM,
    const Core::LinAlg::Matrix<3, 1, T>& Cur, const unsigned int gp)
{
  //*************Begin: Plasticity of curvatures

  // If no yielding parameter was given, the material is modeled in a purely elastic manner
  if (this->params().get_yield_stress_m() < 0.0)
  {
    // compute material stresses by multiplying curvature with constitutive matrix
    Mat::BeamElastHyperMaterial<T>::evaluate_moment_contributions_to_stress(stressM, CM, Cur, gp);
  }
  else
  {
    //! copy of material curvature K (but first entry is 0 if torsional plasticity is turned off)
    Core::LinAlg::Matrix<3, 1, T> kappa{Core::LinAlg::Initialization::zero};

    // If torsional plasticity is turned on, use full curvature vector for plasticity,
    // else, continue with reduced curvature vector (first entry is zero)
    if (this->params().get_torsion_plasticity())
    {
      kappa(0) = Cur(0);
    }
    kappa(1) = Cur(1);
    kappa(2) = Cur(2);


    // return-mapping algorithm

    // compute elastic curvature
    for (int i = 0; i < 3; i++)
    {
      kappaelast_[gp](i) = kappa(i) - kappaplastconv_[gp](i);
    }

    // compute resulting moments
    stressM.multiply(CM, kappaelast_[gp]);

    // compute norm of moment vector
    normstress_m_[gp] = stressM.norm2();

    // compute fraction that exceeds the current yield moment
    deltastress_m_[gp] = normstress_m_[gp] - effyieldstress_m_[gp];

    // check if yield moment is surpassed
    if (deltastress_m_[gp] > 0.0)
    {
      // compute plastic curvature increment
      delta_kappaplast_[gp] =
          (CM(1, 1) - c_m_eff_[gp](1, 1)) / CM(1, 1) * deltastress_m_[gp] / CM(1, 1);

      // update plastic curvature
      for (int i = 0; i < 3; i++)
      {
        kappaplastnew_[gp](i) = kappaplastconv_[gp](i) + delta_kappaplast_[gp] *
                                                             kappaelast_[gp](i) /
                                                             kappaelast_[gp].norm2();
      }

      // update elastic curvature
      for (int i = 0; i < 3; i++)
      {
        kappaelast_[gp](i) = kappa(i) - kappaplastnew_[gp](i);
      }

      // update moment vector and its norm
      stressM.multiply(CM, kappaelast_[gp]);
      normstress_m_[gp] = stressM.norm2();
    }

    // if torsional plasticity is turned off, the moment needs to be recomputed using the full
    // elastic curvature (kappaelast(0) is zero in this case)
    if (!this->params().get_torsion_plasticity())
    {
      kappaelast_[gp](0) = Cur(0);
      stressM.multiply(CM, kappaelast_[gp]);
    }
  }

  //*************End: Plasticity of curvatures
}
/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamPlasticMaterial<T>::compute_constitutive_parameter(
    Core::LinAlg::Matrix<3, 3, T>& C_N, Core::LinAlg::Matrix<3, 3, T>& C_M)
{
  Mat::BeamElastHyperMaterial<T>::compute_constitutive_parameter(C_N, C_M);

  for (unsigned int gp = 0; gp < numgp_force_; gp++)
  {
    // If plasticity for axial strains is enabled, get hardening constitutive parameters
    if (this->params().get_yield_stress_n() >= 0)
    {
      get_hardening_constitutive_matrix_of_forces_material_frame(c_n_eff_[gp]);
      get_effective_yield_stress_n(effyieldstress_n_[gp], this->params().get_yield_stress_n(),
          C_N(0, 0), c_n_eff_[gp](0, 0), gp);
    }
  }
  for (unsigned int gp = 0; gp < numgp_moment_; gp++)
  {
    // If plasticity for curvatures is enabled, get hardening constitutive parameters
    if (this->params().get_yield_stress_m() >= 0)
    {
      get_hardening_constitutive_matrix_of_moments_material_frame(c_m_eff_[gp]);
      get_effective_yield_stress_m(effyieldstress_m_[gp], this->params().get_yield_stress_m(),
          C_M(1, 1), c_m_eff_[gp](1, 1), gp);
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/

template <typename T>
void Mat::BeamPlasticMaterial<T>::update()
{
  for (unsigned int gp = 0; gp < numgp_force_; gp++)
  {
    gammaplastaccum_[gp] += std::abs(gammaplastconv_[gp](0) - gammaplastnew_[gp](0));
    gammaplastconv_[gp] = gammaplastnew_[gp];
    c_n_eff_[gp].put_scalar(0.0);
    effyieldstress_n_[gp] = 0.0;
    delta_gammaplast_[gp].put_scalar(0.0);
    deltastress_n_[gp].put_scalar(0.0);
  }
  for (unsigned int gp = 0; gp < numgp_moment_; gp++)
  {
    kappaplastaccum_[gp] += std::sqrt((kappaplastconv_[gp](0) - kappaplastnew_[gp](0)) *
                                          (kappaplastconv_[gp](0) - kappaplastnew_[gp](0)) +
                                      (kappaplastconv_[gp](1) - kappaplastnew_[gp](1)) *
                                          (kappaplastconv_[gp](1) - kappaplastnew_[gp](1)) +
                                      (kappaplastconv_[gp](2) - kappaplastnew_[gp](2)) *
                                          (kappaplastconv_[gp](2) - kappaplastnew_[gp](2)));
    kappaplastconv_[gp] = kappaplastnew_[gp];
    c_m_eff_[gp].put_scalar(0.0);
    effyieldstress_m_[gp] = 0.0;
    kappaelast_[gp].put_scalar(0.0);
    kappaelastflow_[gp].put_scalar(0.0);
    elastic_curvature_[gp].put_scalar(0.0);
    delta_kappaplast_[gp] = 0.0;
    normstress_m_[gp] = 0.0;
    deltastress_m_[gp] = 0.0;
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamPlasticMaterial<T>::reset()
{
  gammaplastnew_ = gammaplastconv_;
  kappaplastnew_ = kappaplastconv_;
}
/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamPlasticMaterial<T>::get_constitutive_matrix_of_forces_material_frame(
    Core::LinAlg::Matrix<3, 3, T>& C_N) const
{
  // defining material constitutive matrix CN between Gamma and N
  // according to Jelenic 1999, section 2.4
  C_N.clear();

  C_N(0, 0) = this->params().get_axial_rigidity();
  C_N(1, 1) = this->params().get_shear_rigidity2();
  C_N(2, 2) = this->params().get_shear_rigidity3();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamPlasticMaterial<T>::get_constitutive_matrix_of_moments_material_frame(
    Core::LinAlg::Matrix<3, 3, T>& C_M) const
{
  // defining material constitutive matrix CM between curvature and moment
  // according to Jelenic 1999, section 2.4
  C_M.clear();

  C_M(0, 0) = this->params().get_torsional_rigidity();
  C_M(1, 1) = this->params().get_bending_rigidity2();
  C_M(2, 2) = this->params().get_bending_rigidity3();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
double Mat::BeamPlasticMaterial<T>::get_translational_mass_inertia_factor() const
{
  return this->params().get_translational_mass_inertia();
}
/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
double Mat::BeamPlasticMaterial<T>::get_interaction_radius() const
{
  return this->params().get_interaction_radius();
}


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamPlasticMaterial<T>::get_hardening_constitutive_matrix_of_forces_material_frame(
    Core::LinAlg::Matrix<3, 3, T>& CN_eff) const
{
  CN_eff.clear();

  CN_eff(0, 0) = this->params().get_hardening_axial_rigidity();
  CN_eff(1, 1) = this->params().get_hardening_shear_rigidity2();
  CN_eff(2, 2) = this->params().get_hardening_shear_rigidity3();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamPlasticMaterial<T>::get_hardening_constitutive_matrix_of_moments_material_frame(
    Core::LinAlg::Matrix<3, 3, T>& CM_eff) const
{
  CM_eff.clear();
  if (this->params().get_torsion_plasticity())
    CM_eff(0, 0) = this->params().get_hardening_momental_rigidity();
  else
    CM_eff(0, 0) = this->params().get_torsional_rigidity();

  CM_eff(1, 1) = this->params().get_hardening_momental_rigidity();
  CM_eff(2, 2) = CM_eff(1, 1);
}
/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamPlasticMaterial<T>::get_effective_yield_stress_n(
    T& eff_yieldN, T init_yieldN, T CN_0, T CN_eff_0, const unsigned int gp) const
{
  eff_yieldN = init_yieldN + (CN_0 * CN_eff_0) / (CN_0 - CN_eff_0) * gammaplastaccum_[gp];
}


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamPlasticMaterial<T>::get_effective_yield_stress_m(
    T& eff_yieldM, T init_yieldM, T CM_1, T CM_eff_1, const unsigned int gp) const
{
  eff_yieldM = init_yieldM + (CM_1 * CM_eff_1) / (CM_1 - CM_eff_1) * kappaplastaccum_[gp];
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamPlasticMaterial<T>::get_stiffness_matrix_of_moments(
    Core::LinAlg::Matrix<3, 3, T>& stiffM, const Core::LinAlg::Matrix<3, 3, T>& C_M, const int gp)
{
  /* compute spatial stresses and constitutive matrix from material ones according to Jelenic
   * 1999, page 148, paragraph between (2.22) and (2.23) and Romero 2004, (3.10)*/

  if (this->params().get_yield_stress_m() < 0 || normstress_m_[gp] + 10e-10 < effyieldstress_m_[gp])
    Mat::BeamElastHyperMaterial<T>::get_stiffness_matrix_of_moments(stiffM, C_M, gp);
  else
  {
    // Compute stiffness matrix for plastic regime:

    // norm of kappaelastflow
    T normKappaelastflow;

    // starting index for vector assignments
    int i_start = 0;

    // Set starting index s.t. the first entry of kappaelastflow and e will be 0 if
    // torsion plasticity is turned off
    if (!(this->params().get_torsion_plasticity()))
    {
      i_start = 1;
    }

    // initialize kappaelastflow, which points in the direction of the plastic increment
    for (int i = i_start; i < 3; i++)
    {
      kappaelastflow_[gp](i) = kappaelast_[gp](i);
    }
    normKappaelastflow = kappaelastflow_[gp].norm2();

    // compute e, which is the unit vector in the direction of kappaelastflow
    for (int i = i_start; i < 3; i++)
    {
      elastic_curvature_[gp](i) = kappaelastflow_[gp](i) / normKappaelastflow;
    }

    // compute the stiffness matrix of moments
    for (int i = i_start; i < 3; i++)
    {
      for (int j = i_start; j < 3; j++)
      {
        stiffM(i, j) =
            C_M(1, 1) *
            ((delta_kappaplast_[gp] / normKappaelastflow - (1.0 - c_m_eff_[gp](1, 1) / C_M(1, 1))) *
                elastic_curvature_[gp](i) * elastic_curvature_[gp](j));

        if (i == j)
          stiffM(i, j) =
              stiffM(i, j) + C_M(1, 1) * (1.0 - delta_kappaplast_[gp] / normKappaelastflow);
      }
    }
    // if torsional plasticity is turned off, the first entry of the stiffness matrix is that of
    // cM
    if (i_start == 1)
    {
      stiffM(0, 0) = C_M(1, 1);
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void Mat::BeamPlasticMaterial<T>::get_stiffness_matrix_of_forces(
    Core::LinAlg::Matrix<3, 3, T>& stiffN, const Core::LinAlg::Matrix<3, 3, T>& C_N, const int gp)
{
  if (this->params().get_yield_stress_n() < 0.0 || stress_n_[gp] < effyieldstress_n_[gp])
  {
    stiffN = C_N;
  }
  else
  {
    stiffN = c_n_eff_[gp];
  }
}


// explicit template instantiations
template class Mat::BeamPlasticMaterial<double>;

template class Mat::BeamElastPlasticMaterialType<double>;

FOUR_C_NAMESPACE_CLOSE

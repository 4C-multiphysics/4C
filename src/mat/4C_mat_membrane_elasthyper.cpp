// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_membrane_elasthyper.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_elast_summand.hpp"
#include "4C_mat_membrane_elasthyper_service.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                                          |
 *----------------------------------------------------------------------*/
Mat::PAR::MembraneElastHyper::MembraneElastHyper(const Core::Mat::PAR::Parameter::Data& matdata)
    : Mat::PAR::ElastHyper(matdata)
{
  return;
}  // Mat::PAR::MembraneElastHyper::MembraneElastHyper

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::MembraneElastHyper::create_material()
{
  return std::make_shared<Mat::MembraneElastHyper>(this);
}  // Mat::PAR::MembraneElastHyper::create_material


Mat::MembraneElastHyperType Mat::MembraneElastHyperType::instance_;


Core::Communication::ParObject* Mat::MembraneElastHyperType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::MembraneElastHyper* memelhy = new Mat::MembraneElastHyper();
  memelhy->unpack(buffer);

  return memelhy;
}  // Mat::Membrane_ElastHyperType::Create

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
Mat::MembraneElastHyper::MembraneElastHyper() : Mat::ElastHyper(), fibervecs_(true)
{
  return;
}  // Mat::MembraneElastHyper::MembraneElastHyper()

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
Mat::MembraneElastHyper::MembraneElastHyper(Mat::PAR::MembraneElastHyper* params)
    : Mat::ElastHyper(params), fibervecs_(true)
{
  return;
}  // Mat::MembraneElastHyper::MembraneElastHyper()

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void Mat::MembraneElastHyper::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // add base class Element
  Mat::ElastHyper::pack(data);

  add_to_pack(data, fibervecs_);

  return;
}  // Mat::MembraneElastHyper::pack()

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void Mat::MembraneElastHyper::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Mat::ElastHyper::unpack(buffer);

  extract_from_pack(buffer, fibervecs_);

  return;
}  // Mat::MembraneElastHyper::unpack()

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void Mat::MembraneElastHyper::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // call setup of base class
  Mat::ElastHyper::setup(numgp, fibers, coord_system);

  get_fiber_vecs(fibervecs_);

  return;
}  // Mat::MembraneElastHyper::setup()

/*----------------------------------------------------------------------*
 | hyperelastic stress response plus elasticity tensor                  |
 *----------------------------------------------------------------------*/
void Mat::MembraneElastHyper::evaluate_membrane(const Core::LinAlg::Matrix<3, 3>& defgrd,
    const Core::LinAlg::Matrix<3, 3>& cauchygreen, const Teuchos::ParameterList& params,
    const EvaluationContext& context, const Core::LinAlg::Matrix<3, 3>& Q_trafo,
    Core::LinAlg::Matrix<3, 1>& stress, Core::LinAlg::Matrix<3, 3>& cmat, const int gp,
    const int eleGID)
{
  // blank resulting quantities
  stress.clear();
  cmat.clear();

  // kinematic quantities and identity tensors
  Core::LinAlg::Matrix<3, 1> id2(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> id4sharp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> rcg(Core::LinAlg::Initialization::zero);
  double rcg33;
  Core::LinAlg::Matrix<3, 1> icg(Core::LinAlg::Initialization::zero);
  membrane_elast_hyper_evaluate_kin_quant(cauchygreen, id2, id4sharp, rcg, rcg33, icg);

  // evaluate isotropic 2nd Piola-Kirchhoff stress and constitutive tensor
  Core::LinAlg::Matrix<3, 1> stress_iso(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> cmat_iso(Core::LinAlg::Initialization::zero);
  membrane_elast_hyper_evaluate_isotropic_stress_cmat(stress_iso, cmat_iso, id2, id4sharp, rcg,
      rcg33, icg, gp, eleGID, potsum_, summandProperties_);

  // update 2nd Piola-Kirchhoff stress and constitutive tensor
  stress.update(1.0, stress_iso, 1.0);
  cmat.update(1.0, cmat_iso, 1.0);

  // evaluate anisotropic 2nd Piola-Kirchhoff stress and constitutive tensor
  if (summandProperties_.anisoprinc)
  {
    Core::LinAlg::Matrix<3, 1> stress_aniso(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 3> cmat_aniso(Core::LinAlg::Initialization::zero);
    evaluate_anisotropic_stress_cmat(
        stress_aniso, cmat_aniso, Q_trafo, rcg, rcg33, params, gp, eleGID);

    // update 2nd Piola-Kirchhoff stress and constitutive tensor
    stress.update(1.0, stress_aniso, 1.0);
    cmat.update(1.0, cmat_aniso, 1.0);
  }
  if (summandProperties_.anisomod)
  {
    FOUR_C_THROW("anisomod_ not implemented for membrane elasthyper materials!");
  }

  return;
}  // Mat::MembraneElastHyper::Evaluate

/*----------------------------------------------------------------------*
 | evaluate strain energy function                                      |
 *----------------------------------------------------------------------*/
void Mat::MembraneElastHyper::strain_energy(
    Core::LinAlg::Matrix<3, 3>& cauchygreen, double& psi, const int gp, const int eleGID)
{
  // kinematic quantities and identity tensors
  Core::LinAlg::Matrix<3, 1> id2(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> id4sharp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> rcg(Core::LinAlg::Initialization::zero);
  double rcg33;
  Core::LinAlg::Matrix<3, 1> icg(Core::LinAlg::Initialization::zero);
  membrane_elast_hyper_evaluate_kin_quant(cauchygreen, id2, id4sharp, rcg, rcg33, icg);

  // Green-Lagrange strains matrix E = 0.5 * (Cauchy-Green - Identity)
  // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
  Core::LinAlg::SymmetricTensor<double, 3, 3> glstrain{};
  glstrain(0, 0) = 0.5 * (rcg(0) - 1.0);
  glstrain(1, 1) = 0.5 * (rcg(1) - 1.0);
  glstrain(2, 2) = 0.5 * (rcg33 - 1.0);
  glstrain(1, 0) = 0.5 * rcg(2);

  // principal isotropic invariants
  Core::LinAlg::Matrix<3, 1> prinv_iso(Core::LinAlg::Initialization::zero);
  membrane_elast_hyper_invariants_principal(prinv_iso, rcg, rcg33);

  // loop map of associated potential summands
  for (auto& p : potsum_)
  {
    // note that modified invariants equal the principal invariants as detF=J=1 (incompressibility)
    p->add_strain_energy(psi, prinv_iso, prinv_iso, glstrain, gp, eleGID);
  }
}  // Mat::MembraneElastHyper::StrainEnergy

/*----------------------------------------------------------------------*
 | calculate anisotropic stress and elasticity tensor                   |
 *----------------------------------------------------------------------*/
void Mat::MembraneElastHyper::evaluate_anisotropic_stress_cmat(
    Core::LinAlg::Matrix<3, 1>& stress_aniso, Core::LinAlg::Matrix<3, 3>& cmat_aniso,
    const Core::LinAlg::Matrix<3, 3>& Q_trafo, const Core::LinAlg::Matrix<3, 1>& rcg,
    const double& rcg33, const Teuchos::ParameterList& params, const int gp, int eleGID)
{
  // loop map of associated potential summands
  for (unsigned int p = 0; p < potsum_.size(); ++p)
  {
    // skip for materials without fiber
    if (Core::LinAlg::norm2(fibervecs_[p]) == 0) continue;

    // fibervector in orthonormal frame on membrane surface
    Core::LinAlg::Tensor<double, 3> fibervector{};
    Core::LinAlg::Matrix<3, 1> f_mat = Core::LinAlg::make_matrix_view<3, 1>(fibervector);
    f_mat.multiply_tn(1.0, Q_trafo, Core::LinAlg::make_matrix_view<3, 1>(fibervecs_[p]), 0.0);

    // set new fibervector in anisotropic material
    potsum_[p]->set_fiber_vecs(fibervector);

    // three dimensional right Cauchy-Green
    // REMARK: strain-like 6-Voigt vector
    // NOTE: rcg is a stress-like 3-Voigt vector
    Core::LinAlg::SymmetricTensor<double, 3, 3> rcg_full{};
    rcg_full(0, 0) = rcg(0);
    rcg_full(1, 1) = rcg(1);
    rcg_full(2, 2) = rcg33;
    rcg_full(0, 1) = 2.0 * rcg(2);

    // three dimensional anisotropic stress and constitutive tensor
    Core::LinAlg::SymmetricTensor<double, 3, 3> stress_aniso_full{};
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat_aniso_full{};

    potsum_[p]->add_stress_aniso_principal(
        rcg_full, cmat_aniso_full, stress_aniso_full, params, gp, eleGID);

    // reduced anisotropic stress and constitutive tensor
    Core::LinAlg::Matrix<3, 1> stress_aniso_red(Core::LinAlg::Initialization::zero);
    stress_aniso_red(0) = stress_aniso_full(0, 0);
    stress_aniso_red(1) = stress_aniso_full(1, 1);
    stress_aniso_red(2) = stress_aniso_full(0, 1);

    Core::LinAlg::Matrix<3, 3> cmat_aniso_red(Core::LinAlg::Initialization::zero);
    cmat_aniso_red(0, 0) = cmat_aniso_full(0, 0, 0, 0);
    cmat_aniso_red(0, 1) = cmat_aniso_full(0, 0, 1, 1);
    cmat_aniso_red(0, 2) = cmat_aniso_full(0, 0, 1, 0);
    cmat_aniso_red(1, 0) = cmat_aniso_full(1, 1, 0, 0);
    cmat_aniso_red(1, 1) = cmat_aniso_full(1, 1, 1, 1);
    cmat_aniso_red(1, 2) = cmat_aniso_full(1, 1, 1, 0);
    cmat_aniso_red(2, 0) = cmat_aniso_full(1, 0, 0, 0);
    cmat_aniso_red(2, 1) = cmat_aniso_full(1, 0, 1, 1);
    cmat_aniso_red(2, 2) = cmat_aniso_full(1, 0, 1, 0);

    // anisotropic 2nd Piola Kirchhoff stress
    stress_aniso.update(1.0, stress_aniso_red, 1.0);

    // anisotropic constitutive tensor
    cmat_aniso.update(1.0, cmat_aniso_red, 1.0);
  }
}  // Mat::MembraneElastHyper::evaluate_anisotropic_stress_cmat

FOUR_C_NAMESPACE_CLOSE

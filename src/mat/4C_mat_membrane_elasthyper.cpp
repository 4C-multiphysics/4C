/*----------------------------------------------------------------------*/
/*! \file
\brief hyperelastic toolbox for membranes assuming incompressibility and plane stress

The input line should read
MAT 0 MAT_Membrane_ElastHyper NUMMAT 2 MATIDS 1 2 DENS 0

\level 3


*----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | headers                                               sfuchs 08/2017 |
 *----------------------------------------------------------------------*/
#include "4C_mat_membrane_elasthyper.hpp"

#include "4C_mat_membrane_elasthyper_service.hpp"
#include "4C_matelast_summand.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                           sfuchs 08/2017 |
 *----------------------------------------------------------------------*/
Mat::PAR::MembraneElastHyper::MembraneElastHyper(Teuchos::RCP<Core::Mat::PAR::Material> matdata)
    : Mat::PAR::ElastHyper(matdata)
{
  return;
}  // Mat::PAR::MembraneElastHyper::MembraneElastHyper

/*----------------------------------------------------------------------*
 |                                                       sfuchs 08/2017 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Mat::Material> Mat::PAR::MembraneElastHyper::create_material()
{
  return Teuchos::rcp(new Mat::MembraneElastHyper(this));
}  // Mat::PAR::MembraneElastHyper::create_material


Mat::MembraneElastHyperType Mat::MembraneElastHyperType::instance_;


Core::Communication::ParObject* Mat::MembraneElastHyperType::Create(const std::vector<char>& data)
{
  Mat::MembraneElastHyper* memelhy = new Mat::MembraneElastHyper();
  memelhy->Unpack(data);

  return memelhy;
}  // Mat::Membrane_ElastHyperType::Create

/*----------------------------------------------------------------------*
 |                                                       sfuchs 08/2017 |
 *----------------------------------------------------------------------*/
Mat::MembraneElastHyper::MembraneElastHyper() : Mat::ElastHyper(), fibervecs_(true)
{
  return;
}  // Mat::MembraneElastHyper::MembraneElastHyper()

/*----------------------------------------------------------------------*
 |                                                       sfuchs 08/2017 |
 *----------------------------------------------------------------------*/
Mat::MembraneElastHyper::MembraneElastHyper(Mat::PAR::MembraneElastHyper* params)
    : Mat::ElastHyper(params), fibervecs_(true)
{
  return;
}  // Mat::MembraneElastHyper::MembraneElastHyper()

/*----------------------------------------------------------------------*
 |                                                       sfuchs 08/2017 |
 *----------------------------------------------------------------------*/
void Mat::MembraneElastHyper::Pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);

  // add base class Element
  Mat::ElastHyper::Pack(data);

  AddtoPack(data, fibervecs_);

  return;
}  // Mat::MembraneElastHyper::Pack()

/*----------------------------------------------------------------------*
 |                                                       sfuchs 08/2017 |
 *----------------------------------------------------------------------*/
void Mat::MembraneElastHyper::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  Core::Communication::ExtractAndAssertId(position, data, UniqueParObjectId());

  // extract base class Element
  std::vector<char> basedata(0);
  ExtractfromPack(position, data, basedata);
  Mat::ElastHyper::Unpack(basedata);

  ExtractfromPack(position, data, fibervecs_);

  return;
}  // Mat::MembraneElastHyper::Unpack()

/*----------------------------------------------------------------------*
 |                                                       sfuchs 08/2017 |
 *----------------------------------------------------------------------*/
void Mat::MembraneElastHyper::Setup(int numgp, Input::LineDefinition* linedef)
{
  // call setup of base class
  Mat::ElastHyper::Setup(numgp, linedef);

  GetFiberVecs(fibervecs_);

  return;
}  // Mat::MembraneElastHyper::Setup()

/*----------------------------------------------------------------------*
 | hyperelastic stress response plus elasticity tensor   sfuchs 08/2017 |
 *----------------------------------------------------------------------*/
void Mat::MembraneElastHyper::EvaluateMembrane(const Core::LinAlg::Matrix<3, 3>& defgrd,
    const Core::LinAlg::Matrix<3, 3>& cauchygreen, Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<3, 3>& Q_trafo, Core::LinAlg::Matrix<3, 1>& stress,
    Core::LinAlg::Matrix<3, 3>& cmat, const int gp, const int eleGID)
{
  // blank resulting quantities
  stress.Clear();
  cmat.Clear();

  // kinematic quantities and identity tensors
  Core::LinAlg::Matrix<3, 1> id2(true);
  Core::LinAlg::Matrix<3, 3> id4sharp(true);
  Core::LinAlg::Matrix<3, 1> rcg(true);
  double rcg33;
  Core::LinAlg::Matrix<3, 1> icg(true);
  MembraneElastHyperEvaluateKinQuant(cauchygreen, id2, id4sharp, rcg, rcg33, icg);

  // evaluate isotropic 2nd Piola-Kirchhoff stress and constitutive tensor
  Core::LinAlg::Matrix<3, 1> stress_iso(true);
  Core::LinAlg::Matrix<3, 3> cmat_iso(true);
  MembraneElastHyperEvaluateIsotropicStressCmat(stress_iso, cmat_iso, id2, id4sharp, rcg, rcg33,
      icg, gp, eleGID, potsum_, summandProperties_);

  // update 2nd Piola-Kirchhoff stress and constitutive tensor
  stress.Update(1.0, stress_iso, 1.0);
  cmat.Update(1.0, cmat_iso, 1.0);

  // evaluate anisotropic 2nd Piola-Kirchhoff stress and constitutive tensor
  if (summandProperties_.anisoprinc)
  {
    Core::LinAlg::Matrix<3, 1> stress_aniso(true);
    Core::LinAlg::Matrix<3, 3> cmat_aniso(true);
    evaluate_anisotropic_stress_cmat(
        stress_aniso, cmat_aniso, Q_trafo, rcg, rcg33, params, gp, eleGID);

    // update 2nd Piola-Kirchhoff stress and constitutive tensor
    stress.Update(1.0, stress_aniso, 1.0);
    cmat.Update(1.0, cmat_aniso, 1.0);
  }
  if (summandProperties_.anisomod)
  {
    FOUR_C_THROW("anisomod_ not implemented for membrane elasthyper materials!");
  }

  return;
}  // Mat::MembraneElastHyper::Evaluate

/*----------------------------------------------------------------------*
 | evaluate strain energy function                       sfuchs 08/2017 |
 *----------------------------------------------------------------------*/
void Mat::MembraneElastHyper::StrainEnergy(
    Core::LinAlg::Matrix<3, 3>& cauchygreen, double& psi, const int gp, const int eleGID)
{
  // kinematic quantities and identity tensors
  Core::LinAlg::Matrix<3, 1> id2(true);
  Core::LinAlg::Matrix<3, 3> id4sharp(true);
  Core::LinAlg::Matrix<3, 1> rcg(true);
  double rcg33;
  Core::LinAlg::Matrix<3, 1> icg(true);
  MembraneElastHyperEvaluateKinQuant(cauchygreen, id2, id4sharp, rcg, rcg33, icg);

  // Green-Lagrange strains matrix E = 0.5 * (Cauchy-Green - Identity)
  // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
  Core::LinAlg::Matrix<6, 1> glstrain(true);
  glstrain(0) = 0.5 * (rcg(0) - 1.0);
  glstrain(1) = 0.5 * (rcg(1) - 1.0);
  glstrain(2) = 0.5 * (rcg33 - 1.0);
  glstrain(3) = rcg(2);

  // principal isotropic invariants
  Core::LinAlg::Matrix<3, 1> prinv_iso(true);
  MembraneElastHyperInvariantsPrincipal(prinv_iso, rcg, rcg33);

  // loop map of associated potential summands
  for (auto& p : potsum_)
  {
    // note that modified invariants equal the principal invariants as detF=J=1 (incompressibility)
    p->AddStrainEnergy(psi, prinv_iso, prinv_iso, glstrain, gp, eleGID);
  }
}  // Mat::MembraneElastHyper::StrainEnergy

/*----------------------------------------------------------------------*
 | calculate anisotropic stress and elasticity tensor    sfuchs 08/2017 |
 *----------------------------------------------------------------------*/
void Mat::MembraneElastHyper::evaluate_anisotropic_stress_cmat(
    Core::LinAlg::Matrix<3, 1>& stress_aniso, Core::LinAlg::Matrix<3, 3>& cmat_aniso,
    const Core::LinAlg::Matrix<3, 3>& Q_trafo, const Core::LinAlg::Matrix<3, 1>& rcg,
    const double& rcg33, Teuchos::ParameterList& params, const int gp, int eleGID)
{
  // loop map of associated potential summands
  for (unsigned int p = 0; p < potsum_.size(); ++p)
  {
    // skip for materials without fiber
    if (fibervecs_[p].Norm2() == 0) continue;

    // fibervector in orthonormal frame on membrane surface
    Core::LinAlg::Matrix<3, 1> fibervector(true);
    fibervector.MultiplyTN(1.0, Q_trafo, fibervecs_[p], 0.0);

    // set new fibervector in anisotropic material
    potsum_[p]->SetFiberVecs(fibervector);

    // three dimensional right Cauchy-Green
    // REMARK: strain-like 6-Voigt vector
    // NOTE: rcg is a stress-like 3-Voigt vector
    Core::LinAlg::Matrix<6, 1> rcg_full(true);
    rcg_full(0) = rcg(0);
    rcg_full(1) = rcg(1);
    rcg_full(2) = rcg33;
    rcg_full(3) = 2.0 * rcg(2);

    // three dimensional anisotropic stress and constitutive tensor
    Core::LinAlg::Matrix<6, 1> stress_aniso_full(true);
    Core::LinAlg::Matrix<6, 6> cmat_aniso_full(true);

    potsum_[p]->add_stress_aniso_principal(
        rcg_full, cmat_aniso_full, stress_aniso_full, params, gp, eleGID);

    // reduced anisotropic stress and constitutive tensor
    Core::LinAlg::Matrix<3, 1> stress_aniso_red(true);
    stress_aniso_red(0) = stress_aniso_full(0);
    stress_aniso_red(1) = stress_aniso_full(1);
    stress_aniso_red(2) = stress_aniso_full(3);

    Core::LinAlg::Matrix<3, 3> cmat_aniso_red(true);
    cmat_aniso_red(0, 0) = cmat_aniso_full(0, 0);
    cmat_aniso_red(0, 1) = cmat_aniso_full(0, 1);
    cmat_aniso_red(0, 2) = cmat_aniso_full(0, 3);
    cmat_aniso_red(1, 0) = cmat_aniso_full(1, 0);
    cmat_aniso_red(1, 1) = cmat_aniso_full(1, 1);
    cmat_aniso_red(1, 2) = cmat_aniso_full(1, 3);
    cmat_aniso_red(2, 0) = cmat_aniso_full(3, 0);
    cmat_aniso_red(2, 1) = cmat_aniso_full(3, 1);
    cmat_aniso_red(2, 2) = cmat_aniso_full(3, 3);

    // anisotropic 2nd Piola Kirchhoff stress
    stress_aniso.Update(1.0, stress_aniso_red, 1.0);

    // anisotropic constitutive tensor
    cmat_aniso.Update(1.0, cmat_aniso_red, 1.0);
  }
}  // Mat::MembraneElastHyper::evaluate_anisotropic_stress_cmat

FOUR_C_NAMESPACE_CLOSE

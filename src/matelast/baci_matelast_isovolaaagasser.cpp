/*----------------------------------------------------------------------*/
/*! \file
\brief Implementation of the isochoric contribution of the AAA Gasser material and the corresponding
volumetric contribution.

\level 1
*/
/*----------------------------------------------------------------------*/

#include "baci_matelast_isovolaaagasser.H"

#include "baci_comm_utils.H"
#include "baci_io_linedefinition.H"
#include "baci_lib_discret.H"
#include "baci_lib_globalproblem.H"
#include "baci_mat_par_bundle.H"
#include "baci_mat_par_material.H"
#include "baci_mat_service.H"


MAT::ELASTIC::PAR::IsoVolAAAGasser::IsoVolAAAGasser(const Teuchos::RCP<MAT::PAR::Material>& matdata)
    : Parameter(matdata), isinit_(false)
{
  // new style
  Epetra_Map dummy_map(1, 1, 0, *(DRT::Problem::Instance()->GetCommunicators()->LocalComm()));

  // Epetra_Map dummy_map(1, 1, 0,
  // *(DRT::Problem::Instance()->GetDis("Structure")->ElementColMap()));

  for (int i = first; i <= last; i++)
  {
    matparams_.push_back(Teuchos::rcp(new Epetra_Vector(dummy_map, true)));
    // matparams_.push_back(Teuchos::rcp(new
    // Epetra_Vector(*(DRT::Problem::Instance()->GetDis("structure")->ElementColMap()),true)));
  }
  matparams_.at(clum)->PutScalar(matdata->GetDouble("CLUM"));
  matparams_.at(cmed)->PutScalar(matdata->GetDouble("CMED"));
  matparams_.at(cablum)->PutScalar(matdata->GetDouble("CABLUM"));
  matparams_.at(nue)->PutScalar(matdata->GetDouble("NUE"));
  matparams_.at(beta)->PutScalar(matdata->GetDouble("BETA"));
  matparams_.at(normdist)->PutScalar(-999.0);
  matparams_.at(cele)->PutScalar(-999.0);

  // optional parameters needed for UQ
  matparams_.at(mu_lum)->PutScalar(matdata->GetDouble("MULUM"));
  matparams_.at(mu_med)->PutScalar(matdata->GetDouble("MUMED"));
  matparams_.at(mu_ablum)->PutScalar(matdata->GetDouble("MUABLUM"));
  matparams_.at(sigma_lum)->PutScalar(matdata->GetDouble("SIGMALUM"));
  matparams_.at(sigma_med)->PutScalar(matdata->GetDouble("SIGMAMED"));
  matparams_.at(sigma_ablum)->PutScalar(matdata->GetDouble("SIGMAABLUM"));

  // stochastic parameter that can only be set at runtime during uq analysis
  matparams_.at(xi)->PutScalar(10e12);
}

MAT::ELASTIC::IsoVolAAAGasser::IsoVolAAAGasser(MAT::ELASTIC::PAR::IsoVolAAAGasser* params)
    : params_(params)
{
}

void MAT::ELASTIC::IsoVolAAAGasser::CalcCele(const int eleGID)
{
  // extend parameters to elecolmap_layout
  params_->ExpandParametersToEleColLayout();
  // new style
  double normdist_myele = params_->GetParameter(params_->normdist, eleGID);
  double cele_myele = -999.0;
  if (normdist_myele == -999.0) dserror("Aneurysm mean ilt distance not found.");


  if (0.0 <= normdist_myele and normdist_myele <= 0.5)
  {
    cele_myele = ((normdist_myele - 0.5) / (-0.5)) * params_->GetParameter(params_->clum, eleGID) +
                 (normdist_myele / 0.5) * params_->GetParameter(params_->cmed, eleGID);
    params_->SetParameter(params_->cele, cele_myele, eleGID);
  }
  else if (0.5 < normdist_myele and normdist_myele <= 1.0)
  {
    cele_myele = ((normdist_myele - 1.0) / (-0.5)) * params_->GetParameter(params_->cmed, eleGID) +
                 ((normdist_myele - 0.5) / 0.5) * params_->GetParameter(params_->cablum, eleGID);
    params_->SetParameter(params_->cele, cele_myele, eleGID);
  }
  else
    dserror("Unable to calculate valid stiffness parameter in material AAAGasser");
}

void MAT::ELASTIC::IsoVolAAAGasser::SetupAAA(Teuchos::ParameterList& params, const int eleGID)
{
  // map in GetParameter can now calculate LID, so we do not need it here       05/2017 birzle
  // get element lID incase we have element specific material parameters
  //  int eleID =
  //      DRT::Problem::Instance()->GetDis("structure")->ElementColMap()->LID(
  //          eleGID);

  CalcCele(eleGID);

  // if xi is smaller 10e12 it has been set by uq routine and hence we
  // compute the element stiffness a little different

  if (params_->GetParameter(params_->xi, eleGID) < 10e12)
  {
    double clum = exp(params_->GetParameter(params_->mu_lum, eleGID) +
                      params_->GetParameter(params_->xi, eleGID) *
                          params_->GetParameter(params_->sigma_lum, eleGID));
    double cmed = exp(params_->GetParameter(params_->mu_med, eleGID) +
                      params_->GetParameter(params_->xi, eleGID) *
                          params_->GetParameter(params_->sigma_med, eleGID));
    double cablum = exp(params_->GetParameter(params_->mu_ablum, eleGID) +
                        params_->GetParameter(params_->xi, eleGID) *
                            params_->GetParameter(params_->sigma_ablum, eleGID));

    // set params
    params_->SetParameter(params_->clum, clum, eleGID);
    params_->SetParameter(params_->cmed, cmed, eleGID);
    params_->SetParameter(params_->cablum, cablum, eleGID);


    if (params_->GetParameter(params_->normdist, eleGID) == -999.0)
      dserror("Aneurysm mean ilt distance not found.");

    // recalculate cele_
    CalcCele(eleGID);
  }

  params_->SetInitToTrue();
}

void MAT::ELASTIC::IsoVolAAAGasser::AddStrainEnergy(double& psi,
    const CORE::LINALG::Matrix<3, 1>& prinv, const CORE::LINALG::Matrix<3, 1>& modinv,
    const CORE::LINALG::Matrix<6, 1>& glstrain, const int gp, const int eleGID)
{
  if (params_->IsInit())
  {
    double my_cele = params_->GetParameter(params_->cele, eleGID);
    double nue_myele = params_->GetParameter(params_->nue, eleGID);
    double beta_myele = params_->GetParameter(params_->beta, eleGID);

    // An Ogden type material is chosen for the isochoric part
    // \f$\Psi=c\underset{i=1}{\overset{3}{\sum}}(\lambda_{i}^{4}-1)\f$
    // which is
    // Psi = c*(I_1^2*I_3^{-2/3} -2* I_2*I_3^{-2/3}-3)
    psi += my_cele * (pow(modinv(0), 2.0) - 2.0 * modinv(1) - 3.0);
    // volumetric part
    // contribution is modeled by an Ogden-Simo_Miehe type SEF:
    // \f$\Psi=\frac {\kappa}{\beta^2}(\beta lnJ + J^{-\beta}-1)\f$
    // with kappa= 8*c/(1-2nu)
    // as Gasser paper states that referential stiffness E=24c and
    // K=24c/(3(1-2nu))

    double detF = sqrt(prinv(2));
    psi += (8 * my_cele) / (1.0 - 2.0 * nue_myele) * 1.0 / (pow(beta_myele, 2.0)) *
           (beta_myele * log(detF) + pow(detF, -beta_myele) - 1.0);
  }
  else
    dserror("Material parameters have not been initialized yet!");
}

void MAT::ELASTIC::IsoVolAAAGasser::AddDerivativesModified(CORE::LINALG::Matrix<3, 1>& dPmodI,
    CORE::LINALG::Matrix<6, 1>& ddPmodII, const CORE::LINALG::Matrix<3, 1>& modinv, const int gp,
    const int eleGID)
{
  // map in GetParameter can now calculate LID, so we do not need it here       05/2017 birzle
  // get element lID incase we have element specific material parameters
  //  int eleID =
  //      DRT::Problem::Instance()->GetDis("structure")->ElementColMap()->LID(
  //          eleGID);

  if (params_->IsInit())
  {
    double nue_myele = params_->GetParameter(params_->nue, eleGID);
    double beta_myele = params_->GetParameter(params_->beta, eleGID);
    double my_cele = params_->GetParameter(params_->cele, eleGID);

    dPmodI(0) += 2. * my_cele * modinv(0);
    dPmodI(1) -= 2. * my_cele;
    dPmodI(2) += (8. * my_cele * (1. - std::pow(modinv(2), -beta_myele))) /
                 ((1. - 2. * nue_myele) * beta_myele * modinv(2));

    ddPmodII(0) += 2. * my_cele;
    ddPmodII(2) += (8. * my_cele * (-1. + std::pow(modinv(2), -beta_myele) * (1. + beta_myele))) /
                   ((1. - 2. * nue_myele) * beta_myele * modinv(2) * modinv(2));
  }
  else
    dserror("Material parameters have not been initialized yet!");
}

void MAT::ELASTIC::IsoVolAAAGasser::VisNames(std::map<std::string, int>& names)
{
  std::string temp = "cele";
  names[temp] = 1;  // scalar
}

bool MAT::ELASTIC::IsoVolAAAGasser::VisData(
    const std::string& name, std::vector<double>& data, int numgp, int eleGID)
{
  // map in GetParameter can now calculate LID, so we do not need it here       05/2017 birzle
  // get element lID in case we have element specific material parameters
  //  int eleID =
  //      DRT::Problem::Instance()->GetDis("structure")->ElementColMap()->LID(
  //          eleGID);

  if (name == "cele")
  {
    if ((int)data.size() != 1) dserror("size mismatch");
    data[0] = params_->GetParameter(params_->cele, eleGID);
  }
  else
  {
    return false;
  }
  return true;
}

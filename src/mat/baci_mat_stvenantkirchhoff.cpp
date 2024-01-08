/*----------------------------------------------------------------------*/
/*! \file
\brief
St. Venant-Kirchhoff material

\level 2

*/
/*----------------------------------------------------------------------*/

#include "baci_mat_stvenantkirchhoff.H"

#include "baci_lib_globalproblem.H"
#include "baci_mat_par_bundle.H"

BACI_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
MAT::PAR::StVenantKirchhoff::StVenantKirchhoff(Teuchos::RCP<MAT::PAR::Material> matdata)
    : Parameter(matdata),
      youngs_(matdata->GetDouble("YOUNG")),
      poissonratio_(matdata->GetDouble("NUE")),
      density_(matdata->GetDouble("DENS"))
{
  if (youngs_ <= 0.) dserror("Young's modulus must be greater zero");
  if (poissonratio_ >= 0.5 || poissonratio_ < -1.) dserror("Poisson's ratio must be in [-1;0.5)");
}

Teuchos::RCP<MAT::Material> MAT::PAR::StVenantKirchhoff::CreateMaterial()
{
  return Teuchos::rcp(new MAT::StVenantKirchhoff(this));
}

MAT::StVenantKirchhoffType MAT::StVenantKirchhoffType::instance_;


CORE::COMM::ParObject* MAT::StVenantKirchhoffType::Create(const std::vector<char>& data)
{
  auto* stvenantk = new MAT::StVenantKirchhoff();
  stvenantk->Unpack(data);
  return stvenantk;
}


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
MAT::StVenantKirchhoff::StVenantKirchhoff() : params_(nullptr) {}


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
MAT::StVenantKirchhoff::StVenantKirchhoff(MAT::PAR::StVenantKirchhoff* params) : params_(params) {}


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void MAT::StVenantKirchhoff::Pack(CORE::COMM::PackBuffer& data) const
{
  CORE::COMM::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->Id();  // in case we are in post-process mode
  AddtoPack(data, matid);
}


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void MAT::StVenantKirchhoff::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  CORE::COMM::ExtractAndAssertId(position, data, UniqueParObjectId());

  // matid and recover params_
  int matid;
  ExtractfromPack(position, data, matid);
  params_ = nullptr;
  if (DRT::Problem::Instance()->Materials() != Teuchos::null)
  {
    if (DRT::Problem::Instance()->Materials()->Num() != 0)
    {
      const int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();
      MAT::PAR::Parameter* mat =
          DRT::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
        params_ = static_cast<MAT::PAR::StVenantKirchhoff*>(mat);
      else
        dserror("Type of parameter material %d does not fit to calling type %d", mat->Type(),
            MaterialType());
    }
  }

  if (position != data.size()) dserror("Mismatch in size of data %d <-> %d", data.size(), position);
}

/*----------------------------------------------------------------------*
// computes isotropic elasticity tensor in matrix notion for 3d
 *----------------------------------------------------------------------*/
void MAT::StVenantKirchhoff::SetupCmat(CORE::LINALG::Matrix<6, 6>& cmat)
{
  // get material parameters
  const double Emod = params_->youngs_;      // Young's modulus (modulus of elasticity)
  const double nu = params_->poissonratio_;  // Poisson's ratio (Querdehnzahl)

  FillCmat(cmat, Emod, nu);
}

void MAT::StVenantKirchhoff::FillCmat(
    CORE::LINALG::Matrix<6, 6>& cmat, const double Emod, const double nu)
{
  // isotropic elasticity tensor C in Voigt matrix notation
  //                     [ 1-nu     nu     nu |       0       0       0    ]
  //                     [        1-nu     nu |       0       0       0    ]
  //         E           [               1-nu |       0       0       0    ]
  // C = --------------- [ ~~~~   ~~~~   ~~~~   ~~~~~~~~~~  ~~~~~~  ~~~~~~ ]
  //     (1+nu)*(1-2*nu) [                    | (1-2*nu)/2    0       0    ]
  //                     [                    |         (1-2*nu)/2    0    ]
  //                     [ symmetric          |                 (1-2*nu)/2 ]
  //
  const double mfac = Emod / ((1.0 + nu) * (1.0 - 2.0 * nu));  // factor

  // clear the material tangent
  cmat.Clear();
  // write non-zero components
  cmat(0, 0) = mfac * (1.0 - nu);
  cmat(0, 1) = mfac * nu;
  cmat(0, 2) = mfac * nu;
  cmat(1, 0) = mfac * nu;
  cmat(1, 1) = mfac * (1.0 - nu);
  cmat(1, 2) = mfac * nu;
  cmat(2, 0) = mfac * nu;
  cmat(2, 1) = mfac * nu;
  cmat(2, 2) = mfac * (1.0 - nu);
  // ~~~
  cmat(3, 3) = mfac * 0.5 * (1.0 - 2.0 * nu);
  cmat(4, 4) = mfac * 0.5 * (1.0 - 2.0 * nu);
  cmat(5, 5) = mfac * 0.5 * (1.0 - 2.0 * nu);
}


/*----------------------------------------------------------------------*
//calculates stresses using one of the above method to evaluate the elasticity tensor
 *----------------------------------------------------------------------*/
void MAT::StVenantKirchhoff::Evaluate(const CORE::LINALG::SerialDenseVector* glstrain_e,
    CORE::LINALG::SerialDenseMatrix* cmat_e, CORE::LINALG::SerialDenseVector* stress_e)
{
  // this is temporary as long as the material does not have a
  // Matrix-type interface
  const CORE::LINALG::Matrix<6, 1> glstrain(glstrain_e->values(), true);
  CORE::LINALG::Matrix<6, 6> cmat(cmat_e->values(), true);
  CORE::LINALG::Matrix<6, 1> stress(stress_e->values(), true);

  SetupCmat(cmat);
  // evaluate stresses
  stress.MultiplyNN(cmat, glstrain);  // sigma = C . epsilon
}


/*----------------------------------------------------------------------*
//calculates stresses using one of the above method to evaluate the elasticity tensor
 *----------------------------------------------------------------------*/
void MAT::StVenantKirchhoff::Evaluate(const CORE::LINALG::Matrix<3, 3>* defgrd,
    const CORE::LINALG::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
    CORE::LINALG::Matrix<6, 1>* stress, CORE::LINALG::Matrix<6, 6>* cmat, const int gp,
    const int eleGID)
{
  SetupCmat(*cmat);
  // evaluate stresses
  stress->MultiplyNN(*cmat, *glstrain);  // sigma = C . epsilon
}


/*----------------------------------------------------------------------*
 |  Calculate strain energy                                    gee 10/09|
 *----------------------------------------------------------------------*/
void MAT::StVenantKirchhoff::StrainEnergy(
    const CORE::LINALG::Matrix<6, 1>& glstrain, double& psi, const int gp, const int eleGID)
{
  CORE::LINALG::Matrix<6, 6> cmat(true);
  SetupCmat(cmat);

  CORE::LINALG::Matrix<6, 1> stress(true);
  stress.MultiplyNN(cmat, glstrain);

  for (int k = 0; k < 6; ++k) psi += glstrain(k) * stress(k);
  psi /= 2.0;
}


/*----------------------------------------------------------------------*
 |  Evaluate for GEMM                                           ly 02/13|
 *----------------------------------------------------------------------*/
void MAT::StVenantKirchhoff::EvaluateGEMM(CORE::LINALG::Matrix<MAT::NUM_STRESS_3D, 1>* stress,
    CORE::LINALG::Matrix<MAT::NUM_STRESS_3D, MAT::NUM_STRESS_3D>* cmat,
    const CORE::LINALG::Matrix<MAT::NUM_STRESS_3D, 1>* glstrain_m,
    const CORE::LINALG::Matrix<MAT::NUM_STRESS_3D, 1>* glstrain_new,
    const CORE::LINALG::Matrix<MAT::NUM_STRESS_3D, 1>* glstrain_old,
    const CORE::LINALG::Matrix<3, 3>* rcg_new, const CORE::LINALG::Matrix<3, 3>* rcg_old,
    const int gp, const int eleGID)
{
#ifdef BACI_DEBUG
  if (!stress) dserror("No stress vector supplied");
  if (!cmat) dserror("No material tangent matrix supplied");
  if (!glstrain_m) dserror("No GL strains supplied");
  if (!glstrain_new) dserror("No GL strains supplied");
  if (!glstrain_old) dserror("No GL strains supplied");
#endif

  // strain energy function
  double psi = 0.0;
  double psio = 0.0;

  Teuchos::ParameterList params;
  CORE::LINALG::Matrix<3, 3> defgrd(true);
  Evaluate(&defgrd, glstrain_m, params, stress, cmat, gp, eleGID);
  StrainEnergy(*glstrain_new, psi, gp, eleGID);
  StrainEnergy(*glstrain_old, psio, gp, eleGID);

  //**********************************************************************
  // ALGORITHMIC STRESSES AND CMAT FOR GEMM
  //**********************************************************************
  // tensor M = increment of Cauchy-Green tensor
  CORE::LINALG::Matrix<3, 3> M;
  M.Update(1.0, *rcg_new, -1.0, *rcg_old);
  double Mb = M.Dot(M);

  // second term in algorithmic stress only if Mb > 0
  // see: O. Gonzalez, Exact energy and momentum conserving algorithms for
  // general models in nonlinear elasticity, CMAME, 190(2000), pp. 1763-1783
  if (Mb < 1.0e-12) return;

  // derivative of strain energy function dpsi = 0.5*stressm
  // double contraction dpsi : M
  double dpsiM = 0.5 * (*stress)(0) * M(0, 0) + 0.5 * (*stress)(1) * M(1, 1) +
                 0.5 * (*stress)(2) * M(2, 2) + (*stress)(3) * M(0, 1) + (*stress)(4) * M(1, 2) +
                 (*stress)(5) * M(0, 2);

  // extend stressm to algorithmic stress
  (*stress)(0) += 2 * ((psi - psio - dpsiM) / Mb) * M(0, 0);
  (*stress)(1) += 2 * ((psi - psio - dpsiM) / Mb) * M(1, 1);
  (*stress)(2) += 2 * ((psi - psio - dpsiM) / Mb) * M(2, 2);
  (*stress)(3) += 2 * ((psi - psio - dpsiM) / Mb) * M(0, 1);
  (*stress)(4) += 2 * ((psi - psio - dpsiM) / Mb) * M(1, 2);
  (*stress)(5) += 2 * ((psi - psio - dpsiM) / Mb) * M(0, 2);

  // TODO: extend cmat to algorithmic material tensor
  // -> not yet completely implemented!!!
  // -> using only cmat so far, which is ok but not optimal!!!

  //**********************************************************************
  //**********************************************************************
  //**********************************************************************
}
BACI_NAMESPACE_CLOSE

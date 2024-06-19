/*----------------------------------------------------------------------*/
/*! \file
\brief Implementation of the isochoric contribution of a Mooney-Rivlin-type material

\level 1
*/
/*----------------------------------------------------------------------*/

#include "4C_matelast_isomooneyrivlin.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::IsoMooneyRivlin::IsoMooneyRivlin(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      c1_(matdata.parameters.get<double>("C1")),
      c2_(matdata.parameters.get<double>("C2"))
{
}

Mat::Elastic::IsoMooneyRivlin::IsoMooneyRivlin(Mat::Elastic::PAR::IsoMooneyRivlin* params)
    : params_(params)
{
}

void Mat::Elastic::IsoMooneyRivlin::AddStrainEnergy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::Matrix<6, 1>& glstrain, const int gp, const int eleGID)
{
  const double c1 = params_->c1_;
  const double c2 = params_->c2_;

  // strain energy: Psi = C1 (\overline{I}_{\boldsymbol{C}}-3) + C2
  // (\overline{II}_{\boldsymbol{C}}-3). add to overall strain energy
  psi += c1 * (modinv(0) - 3.) + c2 * (modinv(1) - 3.);
}

void Mat::Elastic::IsoMooneyRivlin::add_derivatives_modified(Core::LinAlg::Matrix<3, 1>& dPmodI,
    Core::LinAlg::Matrix<6, 1>& ddPmodII, const Core::LinAlg::Matrix<3, 1>& modinv, const int gp,
    const int eleGID)
{
  const double c1 = params_->c1_;
  const double c2 = params_->c2_;

  dPmodI(0) += c1;
  dPmodI(1) += c2;
}
FOUR_C_NAMESPACE_CLOSE

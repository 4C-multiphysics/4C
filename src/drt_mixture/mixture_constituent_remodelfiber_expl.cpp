/*----------------------------------------------------------------------*/
/*! \file

\brief Definition of a remodel constituent with explicit update rule

\level 3


*/
/*----------------------------------------------------------------------*/
#include "mixture_constituent_remodelfiber_expl.H"


MIXTURE::PAR::MixtureConstituent_RemodelFiberExpl::MixtureConstituent_RemodelFiberExpl(
    const Teuchos::RCP<MAT::PAR::Material>& matdata, double ref_mass_fraction)
    : MixtureConstituent_RemodelFiber(matdata, ref_mass_fraction)
{
  // do nothing here, everything will be done in the base class
}

Teuchos::RCP<MIXTURE::MixtureConstituent>
MIXTURE::PAR::MixtureConstituent_RemodelFiberExpl::CreateConstituent(int id)
{
  return Teuchos::rcp(new MIXTURE::MixtureConstituent_RemodelFiberExpl(this, id));
}

MIXTURE::MixtureConstituent_RemodelFiberExpl::MixtureConstituent_RemodelFiberExpl(
    MIXTURE::PAR::MixtureConstituent_RemodelFiberExpl* params, int id)
    : MixtureConstituent_RemodelFiber(params, id), params_(params)
{
  // do nothing here, everything will be done in the base class
}

INPAR::MAT::MaterialType MIXTURE::MixtureConstituent_RemodelFiberExpl::MaterialType() const
{
  return INPAR::MAT::mix_remodelfiber_expl;
}

void MIXTURE::MixtureConstituent_RemodelFiberExpl::UpdateElasticPart(const LINALG::Matrix<3, 3>& F,
    const LINALG::Matrix<3, 3>& iFg, Teuchos::ParameterList& params, double dt, const int gp,
    const int eleGID)
{
  MixtureConstituent_RemodelFiber::UpdateElasticPart(F, iFg, params, dt, gp, eleGID);

  // Call explicit update of growth and remodel equations
  if (params_->growth_enabled_)
  {
    UpdateGrowthAndRemodelingExpl(F, iFg, dt, gp, eleGID);
  }
}

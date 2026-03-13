// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_fluidporo_singlephaseDof.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_fluidporo_singlephaselaw.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *  constructor (public)                               vuong 08/16      |
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroPhaseDof::FluidPoroPhaseDof(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata)
{
}

/*----------------------------------------------------------------------*
 *  factory method for phase dof                       vuong 08/16      |
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroPhaseDof* Mat::PAR::FluidPoroPhaseDof::create_phase_dof(int phasedofId)
{
  // retrieve problem instance to read from
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();

  // for the sake of safety
  if (Global::Problem::instance(probinst)->materials() == nullptr)
    FOUR_C_THROW("List of materials cannot be accessed in the global problem instance.");
  // yet another safety check
  if (Global::Problem::instance(probinst)->materials()->num() == 0)
    FOUR_C_THROW("List of materials in the global problem instance is empty.");

  // retrieve validated input line of material ID in question
  auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(phasedofId);

  // phase law
  Mat::PAR::FluidPoroPhaseDof* phasedof = nullptr;

  switch (curmat->type())
  {
    case Core::Materials::m_fluidporo_phasedof_diffpressure:
    {
      phasedof = static_cast<Mat::PAR::FluidPoroPhaseDofDiffPressure*>(curmat);
      break;
    }
    case Core::Materials::m_fluidporo_phasedof_pressure:
    {
      phasedof = static_cast<Mat::PAR::FluidPoroPhaseDofPressure*>(curmat);
      break;
    }
    case Core::Materials::m_fluidporo_phasedof_saturation:
    {
      phasedof = static_cast<Mat::PAR::FluidPoroPhaseDofSaturation*>(curmat);
      break;
    }
    default:
      FOUR_C_THROW("invalid pressure-saturation law for material {}", curmat->type());
      break;
  }

  return phasedof;
}

/************************************************************************/
/************************************************************************/

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroPhaseDofDiffPressure::FluidPoroPhaseDofDiffPressure(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : FluidPoroPhaseDof(matdata),
      diffpresCoeffs_(matdata.parameters.get<std::vector<int>>("PRESCOEFF")),
      phaselawId_(matdata.parameters.get<int>("PHASELAWID"))
{
  phaselaw_ = Mat::PAR::FluidPoroPhaseLaw::create_phase_law(phaselawId_);
}

/*----------------------------------------------------------------------*
 *  Initialize                                               vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::PAR::FluidPoroPhaseDofDiffPressure::initialize()
{
  phaselaw_->initialize();
  return;
}

/*----------------------------------------------------------------------*
 *  return phase law type                                  vuong 08/16 |
 *----------------------------------------------------------------------*/
Core::Materials::MaterialType Mat::PAR::FluidPoroPhaseDofDiffPressure::poro_phase_law_type() const
{
  return phaselaw_->type();
}

/*----------------------------------------------------------------------*
 *  fill the dof matrix with the phase dofs                 vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::PAR::FluidPoroPhaseDofDiffPressure::fill_do_f_matrix(
    Core::LinAlg::SerialDenseMatrix& dofmat, int numphase) const
{
  // safety check
  if ((int)diffpresCoeffs_.size() != dofmat.numCols())
    FOUR_C_THROW(
        "Number of phases given by the poro singlephase material {} "
        "does not match number of DOFs ({} phases and {} DOFs)!",
        phaselaw_->id(), diffpresCoeffs_.size(), dofmat.numCols());

  // fill pressure coefficients into matrix
  for (size_t i = 0; i < diffpresCoeffs_.size(); i++)
  {
    const int val = diffpresCoeffs_[i];
    if (val != 0) dofmat(numphase, i) = val;
  }
}

/*----------------------------------------------------------------------*
 *  Evaluate generalized pressure of a phase                vuong 08/16 |
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofDiffPressure::evaluate_gen_pressure(
    int phasenum, const std::span<const double> state) const
{
  // return the corresponding dof value
  return state[phasenum];
}


/*----------------------------------------------------------------------*
 *   Evaluate saturation of the phase                       vuong 08/16 |
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofDiffPressure::evaluate_saturation(
    int phasenum, const std::span<const double> state, const std::span<const double> pressure) const
{
  // call the phase law
  return phaselaw_->evaluate_saturation(pressure);
}


/*--------------------------------------------------------------------------*
 *  Evaluate derivative of saturation w.r.t. pressure           vuong 08/16 |
 *---------------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofDiffPressure::evaluate_deriv_of_saturation_wrt_pressure(
    int phasenum, int doftoderive, const std::span<const double> pressure) const
{
  // call the phase law
  return phaselaw_->evaluate_deriv_of_saturation_wrt_pressure(doftoderive, pressure);
}

/*--------------------------------------------------------------------------*
 *  Evaluate 2nd derivative of saturation w.r.t. pressure  kremheller 05/17 |
 *---------------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofDiffPressure::evaluate_second_deriv_of_saturation_wrt_pressure(
    int phasenum, int firstdoftoderive, int seconddoftoderive,
    const std::vector<double>& pressure) const
{
  // call the phase law
  return phaselaw_->evaluate_second_deriv_of_saturation_wrt_pressure(
      firstdoftoderive, seconddoftoderive, pressure);
}

/*----------------------------------------------------------------------------------------*
 * Evaluate derivative of degree of freedom with respect to pressure          vuong 08/16 |
 *----------------------------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofDiffPressure::evaluate_deriv_of_dof_wrt_pressure(
    int phasenum, int doftoderive, const std::span<const double> state) const
{
  // derivative is the corresponding coefficient
  return diffpresCoeffs_[doftoderive];
}

/************************************************************************/
/************************************************************************/

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroPhaseDofPressure::FluidPoroPhaseDofPressure(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : FluidPoroPhaseDof(matdata), phaselawId_(matdata.parameters.get<int>("PHASELAWID"))
{
  phaselaw_ = Mat::PAR::FluidPoroPhaseLaw::create_phase_law(phaselawId_);
  return;
}

/*----------------------------------------------------------------------*
 *  Initialize                                               vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::PAR::FluidPoroPhaseDofPressure::initialize()
{
  phaselaw_->initialize();
  return;
}

/*----------------------------------------------------------------------*
 *  return phase law type                                  vuong 08/16 |
 *----------------------------------------------------------------------*/
Core::Materials::MaterialType Mat::PAR::FluidPoroPhaseDofPressure::poro_phase_law_type() const
{
  return phaselaw_->type();
}

/*----------------------------------------------------------------------*
 *  fill the dof matrix with the phase dofs                 vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::PAR::FluidPoroPhaseDofPressure::fill_do_f_matrix(
    Core::LinAlg::SerialDenseMatrix& dofmat, int numphase) const
{
  // just mark the corresponding entry in the matrix
  dofmat(numphase, numphase) = 1.0;
}

/*----------------------------------------------------------------------*
 *  Evaluate generalized pressure of a phase                vuong 08/16 |
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofPressure::evaluate_gen_pressure(
    int phasenum, const std::span<const double> state) const
{
  // return the corresponding dof value
  return state[phasenum];
}


/*----------------------------------------------------------------------*
 *   Evaluate saturation of the phase                       vuong 08/16 |
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofPressure::evaluate_saturation(
    int phasenum, const std::span<const double> state, const std::span<const double> pressure) const
{
  // call the phase law
  return phaselaw_->evaluate_saturation(pressure);
}


/*--------------------------------------------------------------------------*
 *  Evaluate derivative of saturation w.r.t. pressure           vuong 08/16 |
 *---------------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofPressure::evaluate_deriv_of_saturation_wrt_pressure(
    int phasenum, int doftoderive, const std::span<const double> pressure) const
{
  // call the phase law
  return phaselaw_->evaluate_deriv_of_saturation_wrt_pressure(doftoderive, pressure);
}

/*--------------------------------------------------------------------------*
 *  Evaluate 2nd derivative of saturation w.r.t. pressure  kremheller 05/17 |
 *---------------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofPressure::evaluate_second_deriv_of_saturation_wrt_pressure(
    int phasenum, int firstdoftoderive, int seconddoftoderive,
    const std::vector<double>& pressure) const
{
  // call the phase law
  return phaselaw_->evaluate_second_deriv_of_saturation_wrt_pressure(
      firstdoftoderive, seconddoftoderive, pressure);
}

/*----------------------------------------------------------------------------------------*
 * Evaluate derivative of degree of freedom with respect to pressure          vuong 08/16 |
 *----------------------------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofPressure::evaluate_deriv_of_dof_wrt_pressure(
    int phasenum, int doftoderive, const std::span<const double> state) const
{
  double presurederiv = 0.0;

  // respective derivative of w.r.t. is either 0 or 1
  if (phasenum == doftoderive) presurederiv = 1.0;

  return presurederiv;
}

/************************************************************************/
/************************************************************************/

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroPhaseDofSaturation::FluidPoroPhaseDofSaturation(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : FluidPoroPhaseDof(matdata), phaselawId_(matdata.parameters.get<int>("PHASELAWID"))
{
  phaselaw_ = Mat::PAR::FluidPoroPhaseLaw::create_phase_law(phaselawId_);
  return;
}

/*----------------------------------------------------------------------*
 *  Initialize                                               vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::PAR::FluidPoroPhaseDofSaturation::initialize()
{
  phaselaw_->initialize();
  return;
}

/*----------------------------------------------------------------------*
 *  return phase law type                                  vuong 08/16 |
 *----------------------------------------------------------------------*/
Core::Materials::MaterialType Mat::PAR::FluidPoroPhaseDofSaturation::poro_phase_law_type() const
{
  return phaselaw_->type();
}

/*----------------------------------------------------------------------*
 *  fill the dof matrix with the phase dofs                 vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::PAR::FluidPoroPhaseDofSaturation::fill_do_f_matrix(
    Core::LinAlg::SerialDenseMatrix& dofmat, int numphase) const
{
  // get pressure coefficients of phase law
  const std::vector<int>* presIDs = phaselaw_->pres_ids();

  // safety check
  if ((int)presIDs->size() != dofmat.numCols())
    FOUR_C_THROW(
        "Number of phases given by the poro phase law material {} "
        "does not match number of DOFs ({} phases and {} DOFs)!",
        phaselaw_->id(), presIDs->size(), dofmat.numCols());

  // fill pressure coefficients of phase law into matrix
  for (size_t i = 0; i < presIDs->size(); i++)
  {
    const int val = (*presIDs)[i];
    if (val != 0) dofmat(numphase, i) = val;
  }
}

/*----------------------------------------------------------------------*
 *  Evaluate generalized pressure of a phase                vuong 08/16 |
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofSaturation::evaluate_gen_pressure(
    int phasenum, const std::span<const double> state) const
{
  // evaluate the phase law for the generalized (i.e. some differential pressure)
  // the phase law depends on
  return phaselaw_->evaluate_gen_pressure(state[phasenum]);
}


/*----------------------------------------------------------------------*
 *   Evaluate saturation of the phase                       vuong 08/16 |
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofSaturation::evaluate_saturation(
    int phasenum, const std::span<const double> state, const std::span<const double> pressure) const
{
  // get the corresponding dof value
  return state[phasenum];
}


/*--------------------------------------------------------------------------*
 *  Evaluate derivative of saturation w.r.t. pressure           vuong 08/16 |
 *---------------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofSaturation::evaluate_deriv_of_saturation_wrt_pressure(
    int phasenum, int doftoderive, const std::span<const double> pressure) const
{
  // call the phase law
  return phaselaw_->evaluate_deriv_of_saturation_wrt_pressure(doftoderive, pressure);
}

/*--------------------------------------------------------------------------*
 *  Evaluate 2nd derivative of saturation w.r.t. pressure  kremheller 05/17 |
 *---------------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofSaturation::evaluate_second_deriv_of_saturation_wrt_pressure(
    int phasenum, int firstdoftoderive, int seconddoftoderive,
    const std::vector<double>& pressure) const
{
  // call the phase law
  return phaselaw_->evaluate_second_deriv_of_saturation_wrt_pressure(
      firstdoftoderive, seconddoftoderive, pressure);
}

/*----------------------------------------------------------------------------------------*
 * Evaluate derivative of degree of freedom with respect to pressure          vuong 08/16 |
 *----------------------------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseDofSaturation::evaluate_deriv_of_dof_wrt_pressure(
    int phasenum, int doftoderive, const std::span<const double> pressure) const
{
  // call the phase law for the derivative
  return phaselaw_->evaluate_deriv_of_saturation_wrt_pressure(doftoderive, pressure);
}

/************************************************************************/
/************************************************************************/

FOUR_C_NAMESPACE_CLOSE

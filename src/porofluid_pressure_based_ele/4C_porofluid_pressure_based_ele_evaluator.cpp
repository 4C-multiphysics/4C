// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_ele_evaluator.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_porofluid_pressure_based_ele_parameter.hpp"
#include "4C_porofluid_pressure_based_ele_phasemanager.hpp"
#include "4C_porofluid_pressure_based_ele_variablemanager.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
std::shared_ptr<Discret::Elements::PoroFluidEvaluator::EvaluatorInterface<nsd, nen>>
Discret::Elements::PoroFluidEvaluator::EvaluatorInterface<nsd, nen>::create_evaluator(
    const Discret::Elements::PoroFluidMultiPhaseEleParameter& para,
    const PoroPressureBased::Action& action, int numdofpernode, int numfluidphases,
    const PoroFluidManager::PhaseManagerInterface& phasemanager)
{
  // the evaluator
  std::shared_ptr<EvaluatorInterface<nsd, nen>> evaluator = nullptr;

  bool inittimederiv = false;
  if (action == PoroPressureBased::calc_initial_time_deriv) inittimederiv = true;

  // check if fluidphases present
  const bool hasfluidphases = (numfluidphases > 0);

  // check if we also have to evaluate additional volume fraction terms
  const bool hasvolfracs = (numdofpernode - numfluidphases > 0);

  const bool hasvolfrac_blood_lung = std::invoke(
      [&]()
      {
        if (phasemanager.num_vol_frac() > 0 &&
            phasemanager.total_num_dof() ==
                phasemanager.num_fluid_phases() + phasemanager.num_vol_frac())
        {
          return true;
        }
        else if (phasemanager.total_num_dof() ==
                 phasemanager.num_fluid_phases() + 2 * phasemanager.num_vol_frac())
        {
          return false;
        }
        else
          FOUR_C_THROW("unknown action for evaluation class!");
      });

  // determine action
  switch (action)
  {
    // calculate true pressures and saturation
    case PoroPressureBased::calc_initial_time_deriv:
    case PoroPressureBased::calc_mat_and_rhs:
    case PoroPressureBased::calc_fluid_struct_coupl_mat:
    case PoroPressureBased::calc_fluid_scatra_coupl_mat:
    {
      // initialize the evaluator for the multi phase element
      std::shared_ptr<MultiEvaluator<nsd, nen>> evaluator_multiphase =
          std::make_shared<MultiEvaluator<nsd, nen>>();

      if (hasfluidphases)
      {
        // build evaluators for all but last fluid phase
        for (int curphase = 0; curphase < numfluidphases - 1; curphase++)
        {
          // initialize the evaluator for the current phase
          std::shared_ptr<MultiEvaluator<nsd, nen>> evaluator_phase =
              std::make_shared<MultiEvaluator<nsd, nen>>();

          // temporary interfaces
          std::shared_ptr<EvaluatorInterface<nsd, nen>> tmpevaluator = nullptr;
          std::shared_ptr<AssembleInterface> assembler = nullptr;

          // Note: this term cancels because of the formulation w.r.t. the material formulation of
          // the solid add evaluator for the conservative term (w, v \nabla \cdot S ) assembler =
          // Teuchos::rcp(new AssembleStandard(curphase,inittimederiv)); tmpevaluator =
          // Teuchos::rcp(new EvaluatorConv<nsd, nen>(assembler,curphase));
          // evaluator_phase->AddEvaluator(tmpevaluator);

          // add evaluator for the convective conservative term (w, S \nabla \cdot v )
          if (para.is_ale())
          {
            assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
            tmpevaluator = std::make_shared<EvaluatorSatDivVel<nsd, nen>>(assembler, curphase);
            evaluator_phase->add_evaluator(tmpevaluator);
          }
          // add evaluator for Biot stabilization
          if (para.biot_stab())
          {
            assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
            tmpevaluator = std::make_shared<EvaluatorBiotStab<nsd, nen>>(assembler, curphase);
            evaluator_phase->add_evaluator(tmpevaluator);
          }

          // add evaluator for the diffusive term (\nabla w, K \nabla p)
          // the diffusive term is also assembled into the last phase
          assembler = std::make_shared<AssembleAlsoIntoOtherPhase>(
              curphase, numfluidphases - 1, inittimederiv);
          tmpevaluator = std::make_shared<EvaluatorDiff<nsd, nen>>(assembler, curphase);
          evaluator_phase->add_evaluator(tmpevaluator);

          // add evaluator for the reactive term
          if (phasemanager.is_reactive(curphase))
          {
            // the reactive term is also assembled into the last phase
            assembler = std::make_shared<AssembleAlsoIntoOtherPhase>(
                curphase, numfluidphases - 1, inittimederiv);
            tmpevaluator = std::make_shared<EvaluatorReac<nsd, nen>>(assembler, curphase);
            evaluator_phase->add_evaluator(tmpevaluator);
          }

          // add evaluators for the instationary terms
          if (not para.is_stationary())
          {
            // add evaluator for the instationary pressure term
            // the term is also assembled into the last phase
            assembler = std::make_shared<AssembleAlsoIntoOtherPhase>(
                curphase, numfluidphases - 1, inittimederiv);
            tmpevaluator = std::make_shared<EvaluatorMassPressure<nsd, nen>>(assembler, curphase);
            evaluator_phase->add_evaluator(tmpevaluator);

            // add evaluator for the instationary solid pressure term
            assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
            tmpevaluator =
                std::make_shared<EvaluatorMassSolidPressureSat<nsd, nen>>(assembler, curphase);
            evaluator_phase->add_evaluator(tmpevaluator);

            // add evaluator for the instationary saturation term
            assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
            tmpevaluator = std::make_shared<EvaluatorMassSaturation<nsd, nen>>(assembler, curphase);
            evaluator_phase->add_evaluator(tmpevaluator);
          }

          // add evaluators for the additional terms in fluid equations introduced by volume
          // fractions
          if (hasvolfracs)
          {
            if (hasvolfrac_blood_lung)
            {
              // add evaluators for the instationary terms
              if (not para.is_stationary())
              {
                // add evaluator for the instationary solid pressure term
                assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
                tmpevaluator =
                    std::make_shared<EvaluatorVolFracBloodLungAddInstatTermsSat<nsd, nen>>(
                        assembler, curphase);
                evaluator_phase->add_evaluator(tmpevaluator);
              }

              if (para.is_ale())
              {
                assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
                tmpevaluator =
                    std::make_shared<EvaluatorVolFracBloodLungAddDivVelTermSat<nsd, nen>>(
                        assembler, curphase);
                evaluator_phase->add_evaluator(tmpevaluator);
              }
            }
            else
            {
              // add evaluators for the instationary terms
              if (not para.is_stationary())
              {
                // add evaluator for the instationary solid pressure term
                assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
                tmpevaluator = std::make_shared<
                    EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTermsSat<nsd, nen>>(
                    assembler, curphase);
                evaluator_phase->add_evaluator(tmpevaluator);
              }

              if (para.is_ale())
              {
                assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
                tmpevaluator = std::make_shared<
                    EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTermSat<nsd, nen>>(
                    assembler, curphase);
                evaluator_phase->add_evaluator(tmpevaluator);
              }
            }
          }

          // add the evaluator of the phase to the multiphase evaluator
          evaluator_multiphase->add_evaluator(evaluator_phase);
        }

        // build evaluators for the last fluid phase
        {
          const int curphase = numfluidphases - 1;

          // initialize the evaluator for the last phase
          std::shared_ptr<MultiEvaluator<nsd, nen>> evaluator_lastphase =
              std::make_shared<MultiEvaluator<nsd, nen>>();

          // temporary interfaces
          std::shared_ptr<EvaluatorInterface<nsd, nen>> tmpevaluator = nullptr;
          std::shared_ptr<AssembleInterface> assembler = nullptr;

          // add evaluator for the convective conservative term (w, \nabla \cdot v )
          if (para.is_ale())
          {
            assembler = std::make_shared<AssembleStandard>(curphase, false);
            tmpevaluator = std::make_shared<EvaluatorDivVel<nsd, nen>>(assembler, curphase);
            evaluator_lastphase->add_evaluator(tmpevaluator);
          }
          // add evaluator for Biot stabilization
          if (para.biot_stab())
          {
            assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
            tmpevaluator = std::make_shared<EvaluatorBiotStab<nsd, nen>>(assembler, curphase);
            evaluator_lastphase->add_evaluator(tmpevaluator);
          }

          // add evaluator for the diffusive term (\nabla w, K \nabla p)
          assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
          tmpevaluator = std::make_shared<EvaluatorDiff<nsd, nen>>(assembler, curphase);
          evaluator_lastphase->add_evaluator(tmpevaluator);

          // add evaluator for the reactive term
          if (phasemanager.is_reactive(curphase))
          {
            assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
            tmpevaluator = std::make_shared<EvaluatorReac<nsd, nen>>(assembler, curphase);
            evaluator_lastphase->add_evaluator(tmpevaluator);
          }

          // add evaluators for the instationary terms
          if (not para.is_stationary())
          {
            // add evaluator for the instationary pressure term
            assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
            tmpevaluator = std::make_shared<EvaluatorMassPressure<nsd, nen>>(assembler, curphase);
            evaluator_lastphase->add_evaluator(tmpevaluator);

            // add evaluator for the instationary solid pressure term
            assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
            tmpevaluator =
                std::make_shared<EvaluatorMassSolidPressure<nsd, nen>>(assembler, curphase);
            evaluator_lastphase->add_evaluator(tmpevaluator);
          }

          // add evaluators for the additional terms in fluid equations introduced by volume
          // fractions
          if (hasvolfracs)
          {
            if (hasvolfrac_blood_lung)
            {
              // add evaluators for the instationary terms
              if (not para.is_stationary())
              {
                // add evaluator for the instationary solid pressure term
                assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
                tmpevaluator = std::make_shared<EvaluatorVolFracBloodLungAddInstatTerms<nsd, nen>>(
                    assembler, curphase);
                evaluator_lastphase->add_evaluator(tmpevaluator);
              }

              if (para.is_ale())
              {
                assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
                tmpevaluator = std::make_shared<EvaluatorVolFracBloodLungAddDivVelTerm<nsd, nen>>(
                    assembler, curphase);
                evaluator_lastphase->add_evaluator(tmpevaluator);
              }
            }
            else
            {
              // add evaluators for the instationary terms
              if (not para.is_stationary())
              {
                // add evaluator for the instationary solid pressure term
                assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
                tmpevaluator = std::make_shared<
                    EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTerms<nsd, nen>>(
                    assembler, curphase);
                evaluator_lastphase->add_evaluator(tmpevaluator);
              }

              if (para.is_ale())
              {
                assembler = std::make_shared<AssembleStandard>(curphase, inittimederiv);
                tmpevaluator = std::make_shared<
                    EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTerm<nsd, nen>>(
                    assembler, curphase);
                evaluator_lastphase->add_evaluator(tmpevaluator);
              }
            }
          }

          // add the evaluator of the phase to the multiphase evaluator
          evaluator_multiphase->add_evaluator(evaluator_lastphase);
        }
      }

      // evaluate the additional volume fraction terms in the volume fraction equations
      if (hasvolfracs)
      {
        // initialize the evaluator for the volume fractions
        std::shared_ptr<MultiEvaluator<nsd, nen>> evaluator_volfrac =
            std::make_shared<MultiEvaluator<nsd, nen>>();

        // temporary interfaces
        std::shared_ptr<EvaluatorInterface<nsd, nen>> tmpevaluator = nullptr;
        std::shared_ptr<AssembleInterface> assembler = nullptr;

        if (hasvolfrac_blood_lung)
        {
          // ----------------------------------------------------------------- add evaluators for
          // the instationary terms
          if (not para.is_stationary())
          {
            assembler = std::make_shared<AssembleStandard>(-1, inittimederiv);
            tmpevaluator =
                std::make_shared<EvaluatorVolFracBloodLungInstat<nsd, nen>>(assembler, -1);
            evaluator_volfrac->add_evaluator(tmpevaluator);
          }

          // add evaluators for the mesh-divergence term
          if (para.is_ale())
          {
            assembler = std::make_shared<AssembleStandard>(-1, inittimederiv);
            tmpevaluator =
                std::make_shared<EvaluatorVolFracBloodLungDivVel<nsd, nen>>(assembler, -1);
            evaluator_volfrac->add_evaluator(tmpevaluator);
          }
          // 2) volume fraction pressure terms
          // -------------------------------------------------------- diffusive term
          assembler = std::make_shared<AssembleStandard>(-1, inittimederiv);
          tmpevaluator =
              std::make_shared<EvaluatorVolFracBloodLungPressureDiff<nsd, nen>>(assembler, -1);
          evaluator_volfrac->add_evaluator(tmpevaluator);

          // reactive term
          assembler = std::make_shared<AssembleStandard>(-1, inittimederiv);
          tmpevaluator =
              std::make_shared<EvaluatorVolFracBloodLungPressureReac<nsd, nen>>(assembler, -1);
          evaluator_volfrac->add_evaluator(tmpevaluator);
        }
        else
        {
          // 1) volume fraction terms
          // ----------------------------------------------------------------- add evaluators for
          // the instationary terms
          if (not para.is_stationary())
          {
            assembler = std::make_shared<AssembleStandard>(-1, inittimederiv);
            tmpevaluator =
                std::make_shared<EvaluatorVolFracHomogenizedVasculatureTumorInstat<nsd, nen>>(
                    assembler, -1);
            evaluator_volfrac->add_evaluator(tmpevaluator);
          }

          // add evaluators for the mesh-divergence term
          if (para.is_ale())
          {
            assembler = std::make_shared<AssembleStandard>(-1, inittimederiv);
            tmpevaluator =
                std::make_shared<EvaluatorVolFracHomogenizedVasculatureTumorDivVel<nsd, nen>>(
                    assembler, -1);
            evaluator_volfrac->add_evaluator(tmpevaluator);
          }

          // diffusive term
          assembler = std::make_shared<AssembleStandard>(-1, inittimederiv);
          tmpevaluator =
              std::make_shared<EvaluatorVolFracHomogenizedVasculatureTumorDiff<nsd, nen>>(
                  assembler, -1);
          evaluator_volfrac->add_evaluator(tmpevaluator);

          // reactive term
          assembler = std::make_shared<AssembleStandard>(-1, inittimederiv);
          tmpevaluator =
              std::make_shared<EvaluatorVolFracHomogenizedVasculatureTumorReac<nsd, nen>>(
                  assembler, -1);
          evaluator_volfrac->add_evaluator(tmpevaluator);

          // additional flux term
          assembler = std::make_shared<AssembleStandard>(-1, inittimederiv);
          tmpevaluator =
              std::make_shared<EvaluatorVolFracHomogenizedVasculatureTumorAddFlux<nsd, nen>>(
                  assembler, -1);
          evaluator_volfrac->add_evaluator(tmpevaluator);

          // 2) volume fraction pressure terms
          // -------------------------------------------------------- diffusive term
          assembler = std::make_shared<AssembleStandard>(-1, inittimederiv);
          tmpevaluator =
              std::make_shared<EvaluatorVolFracHomogenizedVasculatureTumorPressureDiff<nsd, nen>>(
                  assembler, -1);
          evaluator_volfrac->add_evaluator(tmpevaluator);

          // reactive term
          assembler = std::make_shared<AssembleStandard>(-1, inittimederiv);
          tmpevaluator =
              std::make_shared<EvaluatorVolFracHomogenizedVasculatureTumorPressureReac<nsd, nen>>(
                  assembler, -1);
          evaluator_volfrac->add_evaluator(tmpevaluator);
        }

        // add the evaluator of the volfractions to the multiphase evaluator
        evaluator_multiphase->add_evaluator(evaluator_volfrac);
      }

      evaluator = evaluator_multiphase;
      break;
    }
    case PoroPressureBased::calc_pres_and_sat:
    {
      // initialize the evaluator for the multi phase element
      std::shared_ptr<MultiEvaluator<nsd, nen>> evaluator_multiphase =
          std::make_shared<MultiEvaluator<nsd, nen>>();

      // temporary interfaces
      std::shared_ptr<EvaluatorInterface<nsd, nen>> tmpevaluator = nullptr;

      // initialize temporary assembler
      std::shared_ptr<AssembleInterface> assembler = nullptr;

      // build evaluators for all phases (fluid and volfrac)
      // volfrac does not actually need pressures and saturations --> set to -1 in evaluator
      for (int iphase = 0; iphase < numdofpernode; iphase++)
      {
        assembler = std::make_shared<AssembleStandard>(iphase, false);
        if (hasvolfrac_blood_lung)
        {
          tmpevaluator = std::make_shared<EvaluatorPressureAndSaturationBloodLung<nsd, nen>>(
              assembler, iphase);
        }
        else
        {
          tmpevaluator =
              std::make_shared<EvaluatorPressureAndSaturationHomogenizedVasculatureTumor<nsd, nen>>(
                  assembler, iphase);
        }
        evaluator_multiphase->add_evaluator(tmpevaluator);
      }
      evaluator = evaluator_multiphase;

      break;
    }
    case PoroPressureBased::calc_solidpressure:
    {
      std::shared_ptr<AssembleInterface> assembler = std::make_shared<AssembleStandard>(-1, false);
      evaluator = std::make_shared<EvaluatorSolidPressure<nsd, nen>>(assembler, -1);

      break;
    }
    case PoroPressureBased::calc_porosity:
    {
      std::shared_ptr<AssembleInterface> assembler = std::make_shared<AssembleStandard>(-1, false);
      evaluator = std::make_shared<EvaluatorPorosity<nsd, nen>>(assembler, -1);

      break;
    }
    case PoroPressureBased::calc_determinant_of_deformationgradient:
    {
      std::shared_ptr<AssembleInterface> assembler = std::make_shared<AssembleStandard>(-1, false);
      evaluator =
          std::make_shared<EvaluatorDeterminantOfDeformationgradient<nsd, nen>>(assembler, -1);

      break;
    }
    case PoroPressureBased::calc_volfrac_blood_lung:
    {
      if (hasvolfrac_blood_lung)
      {
        std::shared_ptr<AssembleInterface> assembler =
            std::make_shared<AssembleStandard>(-1, false);
        evaluator = std::make_shared<EvaluatorVolfracBloodLung<nsd, nen>>(assembler, -1);
      }
      else
      {
        FOUR_C_THROW(
            "You have no additional porous network with closing relation <<blood lung>>, so no "
            "output of <<volfrac_blood_lung>> is possible!");
      }

      break;
    }
    case PoroPressureBased::recon_flux_at_nodes:
    {
      // initialize the evaluator for the multi phase element
      std::shared_ptr<MultiEvaluator<nsd, nen>> evaluator_multiphase =
          std::make_shared<MultiEvaluator<nsd, nen>>();

      // temporary interfaces
      std::shared_ptr<EvaluatorInterface<nsd, nen>> tmpevaluator = nullptr;

      // initialize temporary assembler
      std::shared_ptr<AssembleInterface> assembler = nullptr;

      assembler = std::make_shared<AssembleStandard>(-1, false);
      tmpevaluator = std::make_shared<ReconstructFluxLinearization<nsd, nen>>(assembler, -1);
      evaluator_multiphase->add_evaluator(tmpevaluator);

      // build evaluators for all fluid phases
      for (int iphase = 0; iphase < numfluidphases; iphase++)
      {
        assembler = std::make_shared<AssembleStandard>(iphase, false);
        tmpevaluator = std::make_shared<ReconstructFluxRHS<nsd, nen>>(assembler, iphase);
        evaluator_multiphase->add_evaluator(tmpevaluator);
      }
      evaluator = evaluator_multiphase;

      break;
    }
    case PoroPressureBased::calc_phase_velocities:
    {
      std::shared_ptr<MultiEvaluator<nsd, nen>> evaluator_multiphase =
          std::make_shared<MultiEvaluator<nsd, nen>>();

      std::shared_ptr<EvaluatorInterface<nsd, nen>> tmpevaluator = nullptr;
      std::shared_ptr<AssembleInterface> assembler = nullptr;

      if (hasvolfrac_blood_lung)
      {
        // build evaluators for all phases
        for (int iphase = 0; iphase < numdofpernode; iphase++)
        {
          assembler = std::make_shared<AssembleStandard>(iphase, false);
          tmpevaluator = std::make_shared<EvaluatorPhaseVelocitiesBloodLung<nsd, nen>>(
              assembler, iphase, para.is_ale());
          evaluator_multiphase->add_evaluator(tmpevaluator);
        }
      }
      else
      {
        // build evaluators for all phases
        for (int iphase = 0; iphase < numdofpernode; iphase++)
        {
          assembler = std::make_shared<AssembleStandard>(iphase, false);
          tmpevaluator =
              std::make_shared<EvaluatorPhaseVelocitiesHomogenizedVasculatureTumor<nsd, nen>>(
                  assembler, iphase, para.is_ale());
          evaluator_multiphase->add_evaluator(tmpevaluator);
        }
      }

      evaluator = evaluator_multiphase;

      break;
    }
    case PoroPressureBased::calc_valid_dofs:
    {
      std::shared_ptr<AssembleInterface> assembler = std::make_shared<AssembleStandard>(-1, false);
      if (hasvolfrac_blood_lung)
      {
        evaluator =
            std::make_shared<EvaluatorValidVolFracPressuresBloodLung<nsd, nen>>(assembler, -1);
      }
      else
      {
        evaluator =
            std::make_shared<EvaluatorValidVolFracHomogenizedVasculatureTumorPressures<nsd, nen>>(
                assembler, -1);
      }

      break;
    }
    case PoroPressureBased::calc_domain_integrals:
    {
      int numscal = 0;
      if (para.has_scalar()) numscal = phasemanager.num_scal();
      std::shared_ptr<AssembleInterface> assembler = std::make_shared<AssembleStandard>(-1, false);
      evaluator = std::make_shared<EvaluatorDomainIntegrals<nsd, nen>>(
          assembler, -1, para.domain_int_functions(), numscal);
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown action for evaluation class!");
      break;
    }
  }  // switch(action)

  // done
  return evaluator;
}

/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorBase<nsd, nen>::saturation_linearization_fluid(
    Core::LinAlg::SerialDenseMatrix& mymat, const Core::LinAlg::Matrix<nen, 1>& funct,
    const double prefac, const int numdofpernode, const int numfluidphases, const int curphase,
    const int phasetoadd, const PoroFluidManager::PhaseManagerInterface& phasemanager)
{
  for (int vi = 0; vi < nen; ++vi)
  {
    const double v = prefac * funct(vi);
    const int fvi = vi * numdofpernode + phasetoadd;

    for (int ui = 0; ui < nen; ++ui)
    {
      const double vfunct = v * funct(ui);
      for (int idof = 0; idof < numfluidphases; ++idof)
      {
        const int fui = ui * numdofpernode + idof;

        mymat(fvi, fui) += vfunct * phasemanager.saturation_deriv(curphase, idof);
      }
    }
  }
}
/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorBase<nsd, nen>::porosity_linearization_fluid(
    Core::LinAlg::SerialDenseMatrix& mymat, const Core::LinAlg::Matrix<nen, 1>& funct,
    const double prefac, const int numdofpernode, const int phasetoadd,
    const PoroFluidManager::PhaseManagerInterface& phasemanager)
{
  for (int vi = 0; vi < nen; ++vi)
  {
    const double v = prefac * funct(vi);
    const int fvi = vi * numdofpernode + phasetoadd;

    for (int ui = 0; ui < nen; ++ui)
    {
      const double vfunct = v * funct(ui);
      for (int idof = 0; idof < numdofpernode; ++idof)
      {
        const int fui = ui * numdofpernode + idof;

        mymat(fvi, fui) += vfunct * phasemanager.porosity_deriv(idof);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorBase<nsd, nen>::calc_div_vel_od_mesh(
    Core::LinAlg::SerialDenseMatrix& mymat, const Core::LinAlg::Matrix<nen, 1>& funct,
    const Core::LinAlg::Matrix<nsd, nen>& deriv, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nsd>& xjm, const Core::LinAlg::Matrix<nsd, nsd>& gridvelderiv,
    const double timefacfac, const double fac, const double det, const int numdofpernode,
    const int phasetoadd)
{
  // d (div v_s)/d d_n+1 = derxy * 1.0/theta/dt
  // prefactor is fac since timefacfac/theta/dt = fac
  calc_lin_fac_od_mesh(mymat, funct, derxy, fac, numdofpernode, phasetoadd);

  // shapederivatives see fluid_ele_calc_poro.cpp
  if (nsd == 3)
  {
    const double gridvelderiv_0_0 = gridvelderiv(0, 0);
    const double gridvelderiv_0_1 = gridvelderiv(0, 1);
    const double gridvelderiv_0_2 = gridvelderiv(0, 2);
    const double gridvelderiv_1_0 = gridvelderiv(1, 0);
    const double gridvelderiv_1_1 = gridvelderiv(1, 1);
    const double gridvelderiv_1_2 = gridvelderiv(1, 2);
    const double gridvelderiv_2_0 = gridvelderiv(2, 0);
    const double gridvelderiv_2_1 = gridvelderiv(2, 1);
    const double gridvelderiv_2_2 = gridvelderiv(2, 2);

    const double xjm_0_0 = xjm(0, 0);
    const double xjm_0_1 = xjm(0, 1);
    const double xjm_0_2 = xjm(0, 2);
    const double xjm_1_0 = xjm(1, 0);
    const double xjm_1_1 = xjm(1, 1);
    const double xjm_1_2 = xjm(1, 2);
    const double xjm_2_0 = xjm(2, 0);
    const double xjm_2_1 = xjm(2, 1);
    const double xjm_2_2 = xjm(2, 2);

#define derxjm_(r, c, d, i) derxjm_##r##c##d(i)

#define derxjm_001(ui) (deriv(2, ui) * xjm_1_2 - deriv(1, ui) * xjm_2_2)
#define derxjm_002(ui) (deriv(1, ui) * xjm_2_1 - deriv(2, ui) * xjm_1_1)

#define derxjm_100(ui) (deriv(1, ui) * xjm_2_2 - deriv(2, ui) * xjm_1_2)
#define derxjm_102(ui) (deriv(2, ui) * xjm_1_0 - deriv(1, ui) * xjm_2_0)

#define derxjm_200(ui) (deriv(2, ui) * xjm_1_1 - deriv(1, ui) * xjm_2_1)
#define derxjm_201(ui) (deriv(1, ui) * xjm_2_0 - deriv(2, ui) * xjm_1_0)

#define derxjm_011(ui) (deriv(0, ui) * xjm_2_2 - deriv(2, ui) * xjm_0_2)
#define derxjm_012(ui) (deriv(2, ui) * xjm_0_1 - deriv(0, ui) * xjm_2_1)

#define derxjm_110(ui) (deriv(2, ui) * xjm_0_2 - deriv(0, ui) * xjm_2_2)
#define derxjm_112(ui) (deriv(0, ui) * xjm_2_0 - deriv(2, ui) * xjm_0_0)

#define derxjm_210(ui) (deriv(0, ui) * xjm_2_1 - deriv(2, ui) * xjm_0_1)
#define derxjm_211(ui) (deriv(2, ui) * xjm_0_0 - deriv(0, ui) * xjm_2_0)

#define derxjm_021(ui) (deriv(1, ui) * xjm_0_2 - deriv(0, ui) * xjm_1_2)
#define derxjm_022(ui) (deriv(0, ui) * xjm_1_1 - deriv(1, ui) * xjm_0_1)

#define derxjm_120(ui) (deriv(0, ui) * xjm_1_2 - deriv(1, ui) * xjm_0_2)
#define derxjm_122(ui) (deriv(1, ui) * xjm_0_0 - deriv(0, ui) * xjm_1_0)

#define derxjm_220(ui) (deriv(1, ui) * xjm_0_1 - deriv(0, ui) * xjm_1_1)
#define derxjm_221(ui) (deriv(0, ui) * xjm_1_0 - deriv(1, ui) * xjm_0_0)

    for (int ui = 0; ui < nen; ++ui)
    {
      const double v0 =
          gridvelderiv_1_0 * derxjm_(0, 0, 1, ui) + gridvelderiv_1_1 * derxjm_(0, 1, 1, ui) +
          gridvelderiv_1_2 * derxjm_(0, 2, 1, ui) + gridvelderiv_2_0 * derxjm_(0, 0, 2, ui) +
          gridvelderiv_2_1 * derxjm_(0, 1, 2, ui) + gridvelderiv_2_2 * derxjm_(0, 2, 2, ui);

      const double v1 =
          gridvelderiv_0_0 * derxjm_(1, 0, 0, ui) + gridvelderiv_0_1 * derxjm_(1, 1, 0, ui) +
          gridvelderiv_0_2 * derxjm_(1, 2, 0, ui) + gridvelderiv_2_0 * derxjm_(1, 0, 2, ui) +
          gridvelderiv_2_1 * derxjm_(1, 1, 2, ui) + gridvelderiv_2_2 * derxjm_(1, 2, 2, ui);

      const double v2 =
          gridvelderiv_0_0 * derxjm_(2, 0, 0, ui) + gridvelderiv_0_1 * derxjm_(2, 1, 0, ui) +
          gridvelderiv_0_2 * derxjm_(2, 2, 0, ui) + gridvelderiv_1_0 * derxjm_(2, 0, 1, ui) +
          gridvelderiv_1_1 * derxjm_(2, 1, 1, ui) + gridvelderiv_1_2 * derxjm_(2, 2, 1, ui);

      for (int vi = 0; vi < nen; ++vi)
      {
        const int fvi = vi * numdofpernode + phasetoadd;
        const double v = timefacfac / det * funct(vi);

        mymat(fvi, ui * 3 + 0) += v * v0;

        mymat(fvi, ui * 3 + 1) += v * v1;

        mymat(fvi, ui * 3 + 2) += v * v2;
      }
    }
  }
  else if (nsd == 2)
  {
    const double gridvelderiv_0_0 = gridvelderiv(0, 0);
    const double gridvelderiv_0_1 = gridvelderiv(0, 1);
    const double gridvelderiv_1_0 = gridvelderiv(1, 0);
    const double gridvelderiv_1_1 = gridvelderiv(1, 1);

    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi = vi * numdofpernode + phasetoadd;
      const double v = timefacfac / det * funct(vi);
      for (int ui = 0; ui < nen; ++ui)
      {
        mymat(fvi, ui * 2) +=
            v * (-gridvelderiv_1_0 * deriv(1, ui) + gridvelderiv_1_1 * deriv(0, ui));

        mymat(fvi, ui * 2 + 1) +=
            v * (+gridvelderiv_0_0 * deriv(1, ui) - gridvelderiv_0_1 * deriv(0, ui));
      }
    }
  }
  else
    FOUR_C_THROW("shapederivatives not implemented for 1D!");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorBase<nsd, nen>::calc_lin_fac_od_mesh(
    Core::LinAlg::SerialDenseMatrix& mymat, const Core::LinAlg::Matrix<nen, 1>& funct,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const double vrhs, const int numdofpernode,
    const int phasetoadd)
{
  // linearization of mesh motion
  //------------------------------------------------dJ/dd = dJ/dF : dF/dd = J * F^-T . N_{\psi} = J
  //* N_x
  // J denotes the determinant of the Jacobian of the mapping between current and parameter space,
  // i.e. det(dx/ds) in our case: timefacfac = J * dt * theta --> d(timefacfac)/dd = timefacfac *
  // N_x
  //              fac        = J              --> d(fac)/dd        = fac * N_x

  for (int vi = 0; vi < nen; ++vi)
  {
    const int fvi = vi * numdofpernode + phasetoadd;
    const double v = vrhs * funct(vi);

    for (int ui = 0; ui < nen; ++ui)
    {
      for (int idim = 0; idim < nsd; ++idim)
      {
        const int fui = ui * nsd + idim;
        mymat(fvi, fui) += v * derxy(idim, ui);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorBase<nsd, nen>::calc_diff_od_mesh(
    Core::LinAlg::SerialDenseMatrix& mymat, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    const Core::LinAlg::Matrix<nsd, 1>& diffflux, const Core::LinAlg::Matrix<nsd, 1>& refgrad,
    const Core::LinAlg::Matrix<nsd, 1>& grad, const double timefacfac, const double difffac,
    const int numdofpernode, const int phasetoadd)
{
  // linearization of mesh motion
  //------------------------------------------------dJ/dd = dJ/dF : dF/dd = J * F^-T . N_{\psi} = J
  //* N_x
  // J denotes the determinant of the Jacobian of the mapping between current and parameter space,
  // i.e. det(dx/ds) in our case: timefacfac = J * dt * theta --> d(timefacfac)/dd = timefacfac *
  // N_x
  for (int vi = 0; vi < nen; ++vi)
  {
    const int fvi = vi * numdofpernode + phasetoadd;

    // laplacian in weak form
    double laplawf(0.0);
    for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux(j);
    double v = -laplawf * timefacfac;

    for (int ui = 0; ui < nen; ++ui)
    {
      for (int idim = 0; idim < nsd; ++idim)
      {
        const int fui = ui * nsd + idim;
        mymat(fvi, fui) += v * derxy(idim, ui);
      }
    }
  }

  //----------------------------------------------------------------
  // standard Galerkin terms  -- "shapederivatives" diffusive term
  //----------------------------------------------------------------
  // see scatra_ele_calc_OD.cpp

  if (nsd == 3)
  {
    const double xjm_0_0 = xjm(0, 0);
    const double xjm_0_1 = xjm(0, 1);
    const double xjm_0_2 = xjm(0, 2);
    const double xjm_1_0 = xjm(1, 0);
    const double xjm_1_1 = xjm(1, 1);
    const double xjm_1_2 = xjm(1, 2);
    const double xjm_2_0 = xjm(2, 0);
    const double xjm_2_1 = xjm(2, 1);
    const double xjm_2_2 = xjm(2, 2);

    const double grad_0 = grad(0);
    const double grad_1 = grad(1);
    const double grad_2 = grad(2);

    for (unsigned vi = 0; vi < nen; ++vi)
    {
      const double deriv_vi_0 = deriv(0, vi);
      const double deriv_vi_1 = deriv(1, vi);
      const double deriv_vi_2 = deriv(2, vi);

      const int fvi = vi * numdofpernode + phasetoadd;

      for (unsigned ui = 0; ui < nen; ++ui)
      {
        const double v00 =
            +grad_1 * (deriv_vi_0 * (deriv(2, ui) * xjm_1_2 - deriv(1, ui) * xjm_2_2) +
                          deriv_vi_1 * (deriv(0, ui) * xjm_2_2 - deriv(2, ui) * xjm_0_2) +
                          deriv_vi_2 * (deriv(1, ui) * xjm_0_2 - deriv(0, ui) * xjm_1_2)) +
            grad_2 * (deriv_vi_0 * (deriv(1, ui) * xjm_2_1 - deriv(2, ui) * xjm_1_1) +
                         deriv_vi_1 * (deriv(2, ui) * xjm_0_1 - deriv(0, ui) * xjm_2_1) +
                         deriv_vi_2 * (deriv(0, ui) * xjm_1_1 - deriv(1, ui) * xjm_0_1));
        const double v01 =
            +grad_0 * (deriv_vi_0 * (deriv(1, ui) * xjm_2_2 - deriv(2, ui) * xjm_1_2) +
                          deriv_vi_1 * (deriv(2, ui) * xjm_0_2 - deriv(0, ui) * xjm_2_2) +
                          deriv_vi_2 * (deriv(0, ui) * xjm_1_2 - deriv(1, ui) * xjm_0_2)) +
            grad_2 * (deriv_vi_0 * (deriv(2, ui) * xjm_1_0 - deriv(1, ui) * xjm_2_0) +
                         deriv_vi_1 * (deriv(0, ui) * xjm_2_0 - deriv(2, ui) * xjm_0_0) +
                         deriv_vi_2 * (deriv(1, ui) * xjm_0_0 - deriv(0, ui) * xjm_1_0));
        const double v02 =
            +grad_0 * (deriv_vi_0 * (deriv(2, ui) * xjm_1_1 - deriv(1, ui) * xjm_2_1) +
                          deriv_vi_1 * (deriv(0, ui) * xjm_2_1 - deriv(2, ui) * xjm_0_1) +
                          deriv_vi_2 * (deriv(1, ui) * xjm_0_1 - deriv(0, ui) * xjm_1_1)) +
            grad_1 * (deriv_vi_0 * (deriv(1, ui) * xjm_2_0 - deriv(2, ui) * xjm_1_0) +
                         deriv_vi_1 * (deriv(2, ui) * xjm_0_0 - deriv(0, ui) * xjm_2_0) +
                         deriv_vi_2 * (deriv(0, ui) * xjm_1_0 - deriv(1, ui) * xjm_0_0));

        mymat(fvi, ui * nsd + 0) += difffac * v00;
        mymat(fvi, ui * nsd + 1) += difffac * v01;
        mymat(fvi, ui * nsd + 2) += difffac * v02;
      }
    }

    const double refgrad_0 = refgrad(0);
    const double refgrad_1 = refgrad(1);
    const double refgrad_2 = refgrad(2);

    for (unsigned vi = 0; vi < nen; ++vi)
    {
      const double derxy_vi_0 = derxy(0, vi);
      const double derxy_vi_1 = derxy(1, vi);
      const double derxy_vi_2 = derxy(2, vi);

      const int fvi = vi * numdofpernode + phasetoadd;

      for (unsigned ui = 0; ui < nen; ++ui)
      {
        const double v00 =
            +derxy_vi_1 * (refgrad_0 * (deriv(2, ui) * xjm_1_2 - deriv(1, ui) * xjm_2_2) +
                              refgrad_1 * (deriv(0, ui) * xjm_2_2 - deriv(2, ui) * xjm_0_2) +
                              refgrad_2 * (deriv(1, ui) * xjm_0_2 - deriv(0, ui) * xjm_1_2)) +
            derxy_vi_2 * (refgrad_0 * (deriv(1, ui) * xjm_2_1 - deriv(2, ui) * xjm_1_1) +
                             refgrad_1 * (deriv(2, ui) * xjm_0_1 - deriv(0, ui) * xjm_2_1) +
                             refgrad_2 * (deriv(0, ui) * xjm_1_1 - deriv(1, ui) * xjm_0_1));
        const double v01 =
            +derxy_vi_0 * (refgrad_0 * (deriv(1, ui) * xjm_2_2 - deriv(2, ui) * xjm_1_2) +
                              refgrad_1 * (deriv(2, ui) * xjm_0_2 - deriv(0, ui) * xjm_2_2) +
                              refgrad_2 * (deriv(0, ui) * xjm_1_2 - deriv(1, ui) * xjm_0_2)) +
            derxy_vi_2 * (refgrad_0 * (deriv(2, ui) * xjm_1_0 - deriv(1, ui) * xjm_2_0) +
                             refgrad_1 * (deriv(0, ui) * xjm_2_0 - deriv(2, ui) * xjm_0_0) +
                             refgrad_2 * (deriv(1, ui) * xjm_0_0 - deriv(0, ui) * xjm_1_0));
        const double v02 =
            +derxy_vi_0 * (refgrad_0 * (deriv(2, ui) * xjm_1_1 - deriv(1, ui) * xjm_2_1) +
                              refgrad_1 * (deriv(0, ui) * xjm_2_1 - deriv(2, ui) * xjm_0_1) +
                              refgrad_2 * (deriv(1, ui) * xjm_0_1 - deriv(0, ui) * xjm_1_1)) +
            derxy_vi_1 * (refgrad_0 * (deriv(1, ui) * xjm_2_0 - deriv(2, ui) * xjm_1_0) +
                             refgrad_1 * (deriv(2, ui) * xjm_0_0 - deriv(0, ui) * xjm_2_0) +
                             refgrad_2 * (deriv(0, ui) * xjm_1_0 - deriv(1, ui) * xjm_0_0));

        mymat(fvi, ui * nsd + 0) += difffac * v00;
        mymat(fvi, ui * nsd + 1) += difffac * v01;
        mymat(fvi, ui * nsd + 2) += difffac * v02;
      }
    }
  }
  else if (nsd == 2)
  {
    {
      const double grad_0 = grad(0);
      const double grad_1 = grad(1);

      for (int vi = 0; vi < nen; ++vi)
      {
        const double deriv_vi_0 = deriv(0, vi);
        const double deriv_vi_1 = deriv(1, vi);

        const int fvi = vi * numdofpernode + phasetoadd;

        for (int ui = 0; ui < nen; ++ui)
        {
          const double v00 = +grad_1 * (-deriv_vi_0 * deriv(1, ui) + deriv_vi_1 * deriv(0, ui));
          const double v01 = +grad_0 * (deriv_vi_0 * deriv(1, ui) - deriv_vi_1 * deriv(0, ui));

          mymat(fvi, ui * nsd + 0) += difffac * v00;
          mymat(fvi, ui * nsd + 1) += difffac * v01;
        }
      }
    }

    const double refgrad_0 = refgrad(0);
    const double refgrad_1 = refgrad(1);

    for (unsigned vi = 0; vi < nen; ++vi)
    {
      const double derxy_vi_0 = derxy(0, vi);
      const double derxy_vi_1 = derxy(1, vi);

      const int fvi = vi * numdofpernode + phasetoadd;

      for (unsigned ui = 0; ui < nen; ++ui)
      {
        const double v00 = +derxy_vi_1 * (-refgrad_0 * deriv(1, ui) + refgrad_1 * deriv(0, ui));
        const double v01 = +derxy_vi_0 * (refgrad_0 * deriv(1, ui) - refgrad_1 * deriv(0, ui));

        mymat(fvi, ui * nsd + 0) += difffac * v00;
        mymat(fvi, ui * nsd + 1) += difffac * v01;
      }
    }
  }
  else
    FOUR_C_THROW("shapederivatives not implemented for 1D!");
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorConv<nsd, nen>::evaluate_matrix_and_assemble(
    std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();

  // convective term in convective form
  /*
       /                               \
      |                                 |
      | prefac * v * nabla * Dphi  , q  |
      |                                 |
       \                               /
  */
  const double prefac = timefacfac;

  Core::LinAlg::Matrix<nen, 1> conv;
  // convective part in convective form: rho*u_x*N,x+ rho*u_y*N,y
  conv.multiply_tn(derxy, *variablemanager.ConVelnp());

  for (int vi = 0; vi < nen; ++vi)
  {
    const double v = prefac * funct(vi);
    const int fvi = vi * numdofpernode + phasetoadd;

    for (int ui = 0; ui < nen; ++ui)
      for (int idof = 0; idof < numfluidphases; ++idof)
      {
        const int fui = ui * numdofpernode + idof;

        mymat(fvi, fui) += v * conv(ui) * phasemanager.saturation_deriv(curphase, idof);
      }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorConv<nsd, nen>::evaluate_vector_and_assemble(
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  double conv_sat = 0.0;
  for (int idof = 0; idof < numfluidphases; ++idof)
  {
    // convective term
    const double conv_phi = variablemanager.ConVelnp()->Dot((*variablemanager.GradPhinp())[idof]);
    conv_sat += rhsfac * phasemanager.saturation_deriv(curphase, idof) * conv_phi;
  }
  for (int vi = 0; vi < nen; ++vi)
  {
    const int fvi = vi * numdofpernode + phasetoadd;

    myvec[fvi] -= conv_sat * funct(vi);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorConv<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorConv<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDivVel<nsd, nen>::evaluate_matrix_and_assemble(
    std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDivVel<nsd, nen>::evaluate_vector_and_assemble(
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  double vrhs = rhsfac * variablemanager.div_con_velnp();

  for (int vi = 0; vi < nen; ++vi)
  {
    const int fvi = vi * numdofpernode + phasetoadd;

    myvec[fvi] -= vrhs * funct(vi);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDivVel<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  Core::LinAlg::Matrix<nsd, nsd> gridvelderiv(Core::LinAlg::Initialization::zero);
  gridvelderiv.multiply_nt(*(variablemanager.e_con_velnp()), deriv);

  // OD mesh - div vel term
  EvaluatorBase<nsd, nen>::calc_div_vel_od_mesh(mymat, funct, deriv, derxy, xjm, gridvelderiv,
      timefacfac, fac, det, numdofpernode, phasetoadd);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDivVel<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorSatDivVel<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // call base class
  EvaluatorDivVel<nsd, nen>::evaluate_matrix_and_assemble(elemat, funct, derxy, curphase,
      phasetoadd, numdofpernode, phasemanager, variablemanager, timefacfac, fac, inittimederiv);

  // no linearization needed in case of initial time derivative calculation
  if (!inittimederiv)
  {
    const int numfluidphases = phasemanager.num_fluid_phases();
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    const double consfac = timefacfac * variablemanager.div_con_velnp();

    // call base class for saturation linearization
    EvaluatorBase<nsd, nen>::saturation_linearization_fluid(
        mymat, funct, consfac, numdofpernode, numfluidphases, curphase, phasetoadd, phasemanager);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorSatDivVel<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // call base class with scaled factors
  EvaluatorDivVel<nsd, nen>::evaluate_vector_and_assemble(elevec, funct, derxy, xyze, curphase,
      phasetoadd, numdofpernode, phasemanager, variablemanager,
      phasemanager.saturation(curphase) * rhsfac, phasemanager.saturation(curphase) * fac,
      inittimederiv);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorSatDivVel<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // call base class with scaled factors
  EvaluatorDivVel<nsd, nen>::evaluate_matrix_od_struct_and_assemble(elemat, funct, deriv, derxy,
      xjm, curphase, phasetoadd, numdofpernode, phasemanager, variablemanager,
      phasemanager.saturation(curphase) * timefacfac, phasemanager.saturation(curphase) * fac, det);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorSatDivVel<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorBiotStab<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  FOUR_C_THROW("Biot stabilization is still missing");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorBiotStab<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  FOUR_C_THROW("Biot stabilization is still missing");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorBiotStab<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  FOUR_C_THROW("Biot stabilization is still missing");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorBiotStab<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*

 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDiff<nsd, nen>::evaluate_matrix_and_assemble(
    std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // we do not need the matrix if we calculate the initial time derivative
  if (!inittimederiv)
  {
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    const int numfluidphases = phasemanager.num_fluid_phases();

    const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradphi = *variablemanager.grad_phinp();

    // current pressure gradient
    Core::LinAlg::Matrix<nsd, 1> gradpres(Core::LinAlg::Initialization::zero);
    gradpres.clear();

    // compute the pressure gradient from the phi gradients
    for (int idof = 0; idof < numfluidphases; ++idof)
      gradpres.update(phasemanager.pressure_deriv(curphase, idof), gradphi[idof], 1.0);

    double abspressgrad = 0.0;
    for (int i = 0; i < nsd; i++) abspressgrad += gradpres(i) * gradpres(i);
    abspressgrad = sqrt(abspressgrad);

    // permeability tensor
    Core::LinAlg::Matrix<nsd, nsd> permeabilitytensor(Core::LinAlg::Initialization::zero);
    phasemanager.permeability_tensor(curphase, permeabilitytensor);

    Core::LinAlg::Matrix<nsd, nen> diffflux(Core::LinAlg::Initialization::zero);
    diffflux.multiply(permeabilitytensor, derxy);
    diffflux.scale(phasemanager.rel_permeability(curphase) /
                   phasemanager.dyn_viscosity(curphase, abspressgrad));

    // helper variable for linearization
    Core::LinAlg::Matrix<nsd, 1> diffflux_relpermeability(Core::LinAlg::Initialization::zero);

    if (not phasemanager.has_constant_rel_permeability(curphase))
    {
      diffflux_relpermeability.multiply(permeabilitytensor, gradpres);
      diffflux_relpermeability.scale(phasemanager.rel_permeability_deriv(curphase) /
                                     phasemanager.dyn_viscosity(curphase, abspressgrad));
    }
    else
      diffflux_relpermeability.put_scalar(0.0);

    //----------------------------------------------------------------
    // diffusive term and linearization of relative permeability w.r.t. dof
    //----------------------------------------------------------------
    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi = vi * numdofpernode + phasetoadd;
      double laplawf_relpermeability(0.0);

      // helper variable for linearization
      for (int j = 0; j < nsd; j++)
        laplawf_relpermeability += derxy(j, vi) * diffflux_relpermeability(j);

      for (int ui = 0; ui < nen; ++ui)
      {
        double laplawf(0.0);
        for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux(j, ui);

        for (int idof = 0; idof < numfluidphases; ++idof)
        {
          const int fui = ui * numdofpernode + idof;
          mymat(fvi, fui) += timefacfac * (laplawf * phasemanager.pressure_deriv(curphase, idof) +
                                              funct(ui) * laplawf_relpermeability *
                                                  phasemanager.saturation_deriv(curphase, idof));
        }
      }
    }

    //----------------------------------------------------------------
    // linearization of dynamic viscosity w.r.t. dof
    //----------------------------------------------------------------
    if (not phasemanager.has_constant_dyn_viscosity(curphase))
    {
      // derivative of abspressgrad w.r.t. pressure gradient
      Core::LinAlg::Matrix<nsd, 1> dabspressgraddpresgradp(Core::LinAlg::Initialization::zero);
      dabspressgraddpresgradp.put_scalar(0.0);
      // avoid division by zero
      if (abspressgrad > 1.0e-12)
        for (int i = 0; i < nsd; i++) dabspressgraddpresgradp(i) = gradpres(i) / abspressgrad;

      Core::LinAlg::Matrix<nsd, 1> diffflux2(Core::LinAlg::Initialization::zero);
      diffflux2.multiply(permeabilitytensor, gradpres);
      // d (1/visc) / d abspressgrad = -1.0 * visc^(-2) * d visc / d abspressgrad
      diffflux2.scale(-1.0 * phasemanager.rel_permeability(curphase) /
                      phasemanager.dyn_viscosity(curphase, abspressgrad) /
                      phasemanager.dyn_viscosity(curphase, abspressgrad) *
                      phasemanager.dyn_viscosity_deriv(curphase, abspressgrad));

      for (int vi = 0; vi < nen; ++vi)
      {
        const int fvi = vi * numdofpernode + phasetoadd;
        double laplawf = 0.0;
        for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux2(j);
        for (int ui = 0; ui < nen; ++ui)
        {
          double gradpderxy = 0.0;
          for (int j = 0; j < nsd; j++) gradpderxy += derxy(j, ui) * dabspressgraddpresgradp(j);
          for (int idof = 0; idof < numfluidphases; ++idof)
          {
            const int fui = ui * numdofpernode + idof;
            // d abspressgrad / d phi = d abspressgrad / d gradp * d gradp / d phi =
            //                        = d abspressgrad / d gradp * d / d phi( d p / d phi * d phi /
            //                        d x) = = d abspressgrad / d gradp * derxy * d p / d phi
            // Note: FD-Check might fail here due to kink in formulation of cell-adherence model-law
            mymat(fvi, fui) +=
                timefacfac * laplawf * phasemanager.pressure_deriv(curphase, idof) * gradpderxy;
          }
        }
      }
    }
  }  // !inittimederiv
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDiff<nsd, nen>::evaluate_vector_and_assemble(
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  const int numfluidphases = phasemanager.num_fluid_phases();

  const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradphi = *variablemanager.grad_phinp();

  // current pressure gradient
  Core::LinAlg::Matrix<nsd, 1> gradpres(Core::LinAlg::Initialization::zero);
  gradpres.clear();

  // compute the pressure gradient from the phi gradients
  for (int idof = 0; idof < numfluidphases; ++idof)
    gradpres.update(phasemanager.pressure_deriv(curphase, idof), gradphi[idof], 1.0);

  double abspressgrad = 0.0;
  for (int i = 0; i < nsd; i++) abspressgrad += gradpres(i) * gradpres(i);
  abspressgrad = sqrt(abspressgrad);

  // diffusion tensor
  Core::LinAlg::Matrix<nsd, nsd> difftensor(Core::LinAlg::Initialization::zero);
  phasemanager.permeability_tensor(curphase, difftensor);
  difftensor.scale(
      phasemanager.rel_permeability(curphase) / phasemanager.dyn_viscosity(curphase, abspressgrad));

  Core::LinAlg::Matrix<nsd, 1> diffflux(Core::LinAlg::Initialization::zero);
  diffflux.multiply(difftensor, gradpres);

  for (int vi = 0; vi < nen; ++vi)
  {
    const int fvi = vi * numdofpernode + phasetoadd;

    // laplacian in weak form
    double laplawf(0.0);
    for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux(j);
    myvec[fvi] -= rhsfac * laplawf;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDiff<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();

  const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradphi = *variablemanager.grad_phinp();

  // current pressure gradient
  Core::LinAlg::Matrix<nsd, 1> gradpres(Core::LinAlg::Initialization::zero);
  gradpres.clear();

  // compute the pressure gradient from the phi gradients
  for (int idof = 0; idof < numfluidphases; ++idof)
    gradpres.update(phasemanager.pressure_deriv(curphase, idof), gradphi[idof], 1.0);

  double abspressgrad = 0.0;
  for (int i = 0; i < nsd; i++) abspressgrad += gradpres(i) * gradpres(i);
  abspressgrad = sqrt(abspressgrad);

  // diffusion tensor
  Core::LinAlg::Matrix<nsd, nsd> difftensor(Core::LinAlg::Initialization::zero);
  phasemanager.permeability_tensor(curphase, difftensor);
  difftensor.scale(
      phasemanager.rel_permeability(curphase) / phasemanager.dyn_viscosity(curphase, abspressgrad));

  // TODO: anisotropic difftensor and
  //       non-constant viscosity (because of pressure gradient, probably not really necessary)
  Core::LinAlg::Matrix<nsd, 1> diffflux(Core::LinAlg::Initialization::zero);
  diffflux.multiply(difftensor, gradpres);

  // diffusive pre-factor for linearization
  const double v = difftensor(0, 0) * timefacfac / det;

  // gradient of pressure w.r.t. reference coordinates
  Core::LinAlg::Matrix<nsd, 1> refgradpres(Core::LinAlg::Initialization::zero);
  refgradpres.clear();

  // gradient of phi w.r.t. reference coordinates
  std::vector<Core::LinAlg::Matrix<nsd, 1>> refgradphi(
      numfluidphases, Core::LinAlg::Matrix<nsd, 1>(
                          Core::LinAlg::Initialization::zero));  // Core::LinAlg::Matrix<nsd,1>
                                                                 // refgradphi;
  for (int idof = 0; idof < numfluidphases; ++idof) refgradphi[idof].multiply(xjm, gradphi[idof]);

  // compute the pressure gradient from the phi gradients
  for (int idof = 0; idof < numfluidphases; ++idof)
    refgradpres.update(phasemanager.pressure_deriv(curphase, idof), refgradphi[idof], 1.0);

  // OD mesh - diffusive term
  EvaluatorBase<nsd, nen>::calc_diff_od_mesh(mymat, deriv, derxy, xjm, diffflux, refgradpres,
      gradpres, timefacfac, v, numdofpernode, phasetoadd);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDiff<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorReac<nsd, nen>::evaluate_matrix_and_assemble(
    std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // we do not need the matrix if we calculate the initial time derivative
  if (!inittimederiv)
  {
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    // TODO a constant density is assumed here
    double scaledtimefacfac = timefacfac / phasemanager.density(curphase);

    //----------------------------------------------------------------
    // reaction terms
    //----------------------------------------------------------------
    for (int vi = 0; vi < nen; ++vi)
    {
      const double v = scaledtimefacfac * funct(vi);
      const int fvi = vi * numdofpernode + phasetoadd;

      for (int ui = 0; ui < nen; ++ui)
      {
        const double vfunct = v * funct(ui);
        for (int idof = 0; idof < numdofpernode; ++idof)
        {
          const int fui = ui * numdofpernode + idof;

          // rhs ---> -
          mymat(fvi, fui) -= vfunct * phasemanager.reac_deriv(curphase, idof);
        }
      }
    }
  }  // !inittimederiv
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorReac<nsd, nen>::evaluate_vector_and_assemble(
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  if (not phasemanager.is_reactive(curphase)) return;

  // TODO a constant density is assumed here
  double scale = 1.0 / phasemanager.density(curphase);

  double vrhs = scale * rhsfac * phasemanager.reac_term(curphase);

  for (int vi = 0; vi < nen; ++vi)
  {
    const int fvi = vi * numdofpernode + phasetoadd;
    // rhs ---> +
    myvec[fvi] += vrhs * funct(vi);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorReac<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  if (not phasemanager.is_reactive(curphase)) return;

  // TODO a constant density is assumed here
  double scale = 1.0 / phasemanager.density(curphase);
  double vrhs = scale * timefacfac * phasemanager.reac_term(curphase);

  // linearization of porosity (may appear in reaction term)
  //-------------dreac/dd = dreac/dporosity * dporosity/dd = dreac/dporosity * dporosity/dJ * dJ/dd
  //= dreac/dporosity * dporosity/dJ * J * N_x
  // J denotes the determinant of the deformation gradient, i.e. det F = det ( d x / d X ) = det
  // (dx/ds) * ( det(dX/ds) )^-1

  if (phasemanager.porosity_depends_on_struct())
  {
    vrhs += timefacfac * scale * phasemanager.reac_deriv_porosity(curphase) *
            phasemanager.jacobian_def_grad() * phasemanager.porosity_deriv_wrt_jacobian_def_grad();
  }

  // linearization of mesh motion (Jacobian)
  // 1) linearization of fac +
  // 2) possible linearization w.r.t porosity
  // rhs ---> -
  EvaluatorBase<nsd, nen>::calc_lin_fac_od_mesh(
      mymat, funct, derxy, -1.0 * vrhs, numdofpernode, phasetoadd);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorReac<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  if (not phasemanager.is_reactive(curphase)) return;

  const int numscal = phasemanager.num_scal();

  double vrhs = 1.0 / phasemanager.density(curphase) * timefacfac;

  // linearization of reaction term w.r.t scalars
  for (int vi = 0; vi < nen; ++vi)
  {
    const int fvi = vi * numdofpernode + phasetoadd;
    const double v = vrhs * funct(vi);

    for (int ui = 0; ui < nen; ++ui)
    {
      const double vfunct = v * funct(ui);
      for (int iscal = 0; iscal < numscal; ++iscal)
      {
        const int fui = ui * numscal + iscal;
        // rhs ---> -
        mymat(fvi, fui) -= vfunct * phasemanager.reac_deriv_scalar(curphase, iscal);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassPressure<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // in case of an incompressible fluid phase 'curphase' (1/K = 0) the term cancels out
  if (phasemanager.incompressible_fluid_phase(curphase)) return;

  const int numfluidphases = phasemanager.num_fluid_phases();

  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  // saturation
  const double saturation = phasemanager.saturation(curphase);

  // inverse bulk modulus of phase (compressibility)
  const double invbulkmodulus = phasemanager.inv_bulkmodulus(curphase);

  // pre factor
  const double facfacmass = fac * phasemanager.porosity() * saturation * invbulkmodulus;
  //----------------------------------------------------------------
  // standard Galerkin transient term
  //----------------------------------------------------------------
  for (int vi = 0; vi < nen; ++vi)
  {
    const double v = facfacmass * funct(vi);
    const int fvi = vi * numdofpernode + phasetoadd;

    for (int ui = 0; ui < nen; ++ui)
    {
      const double vfunct = v * funct(ui);
      for (int idof = 0; idof < numfluidphases; ++idof)
      {
        const int fui = ui * numdofpernode + idof;

        mymat(fvi, fui) += vfunct * phasemanager.pressure_deriv(curphase, idof);
      }
    }
  }

  // for the initial time derivative calculation we only need the standard Galerkin transient term
  // and no additional linearizations
  if (!inittimederiv)
  {
    //----------------------------------------------------------------
    // linearization of saturation w.r.t. dof
    //----------------------------------------------------------------
    {
      const std::vector<double>& phinp = *variablemanager.phinp();
      double hist = 0.0;
      // if(curphase==phasetoadd) // bug fix??
      hist = (*variablemanager.hist())[phasetoadd];

      double facfacmass2 =
          fac * phasemanager.pressure_deriv(curphase, phasetoadd) * (phinp[phasetoadd] - hist);

      for (int idof = 0; idof < numfluidphases; ++idof)
      {
        if (idof != phasetoadd)
          facfacmass2 += timefacfac * phasemanager.pressure_deriv(curphase, idof) *
                         (*variablemanager.phidtnp())[idof];
      }

      facfacmass2 *= phasemanager.porosity() * invbulkmodulus;

      // call base class for saturation linearization
      EvaluatorBase<nsd, nen>::saturation_linearization_fluid(mymat, funct, facfacmass2,
          numdofpernode, numfluidphases, curphase, phasetoadd, phasemanager);
    }

    //----------------------------------------------------------------
    // linearization of porosity w.r.t. dof
    //----------------------------------------------------------------
    if (phasemanager.porosity_depends_on_fluid())
    {
      const std::vector<double>& phinp = *variablemanager.phinp();
      double hist = 0.0;
      // if(curphase==phasetoadd)  //bugfix??
      hist = (*variablemanager.hist())[phasetoadd];

      double facfacmass =
          fac * phasemanager.pressure_deriv(curphase, phasetoadd) * (phinp[phasetoadd] - hist);

      for (int idof = 0; idof < numfluidphases; ++idof)
      {
        if (idof != phasetoadd)
          facfacmass += timefacfac * phasemanager.pressure_deriv(curphase, idof) *
                        (*variablemanager.phidtnp())[idof];
      }

      facfacmass *= saturation * invbulkmodulus;

      // call base class:
      EvaluatorBase<nsd, nen>::porosity_linearization_fluid(
          mymat, funct, facfacmass, numdofpernode, phasetoadd, phasemanager);
    }
  }  // !inittimederiv
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassPressure<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // in case of an incompressible fluid phase 'curphase' (1/K = 0) the term cancels out
  if (phasemanager.incompressible_fluid_phase(curphase)) return;

  // for the initial time derivative calculation no transient terms enter the rhs
  if (!inittimederiv)
  {
    // get vector to fill
    Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

    double vtrans = get_rhs_trans(
        curphase, phasetoadd, numdofpernode, phasemanager, variablemanager, rhsfac, fac);

    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi = vi * numdofpernode + phasetoadd;

      myvec[fvi] -= vtrans * funct(vi);
    }
  }  // !inittimederiv
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassPressure<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // in case of an incompressible fluid phase 'curphase' (1/K = 0) the term cancels out
  if (phasemanager.incompressible_fluid_phase(curphase)) return;

  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  double vtrans = get_rhs_trans(
      curphase, phasetoadd, numdofpernode, phasemanager, variablemanager, timefacfac, fac);

  // linearization of porosity
  //------------------------------------------------dporosity/dd = dporosity/dJ * dJ/dd =
  // dporosity/dJ * J * N_x
  // J denotes the determinant of the deformation gradient, i.e. det F = det ( d x / d X ) = det
  // (dx/ds) * ( det(dX/ds) )^-1 in our case: vtrans is scaled with porosity in GetRhsTrans -->
  // scale it with 1.0/porosity here

  if (phasemanager.porosity_depends_on_struct())
    vtrans += vtrans * 1.0 / phasemanager.porosity() * phasemanager.jacobian_def_grad() *
              phasemanager.porosity_deriv_wrt_jacobian_def_grad();

  // linearization of mesh motion (Jacobian)
  // 1) linearization of fac +
  // 2) possible linearization w.r.t porosity
  EvaluatorBase<nsd, nen>::calc_lin_fac_od_mesh(
      mymat, funct, derxy, vtrans, numdofpernode, phasetoadd);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassPressure<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
double Discret::Elements::PoroFluidEvaluator::EvaluatorMassPressure<nsd, nen>::get_rhs_trans(
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac)
{
  const int numfluidphases = phasemanager.num_fluid_phases();
  // read data from managers
  double hist = 0.0;
  // if(curphase==phasetoadd) //bugfix??
  hist = (*variablemanager.hist())[phasetoadd];
  // std::cout << "hist = " << hist << std::endl;
  const double porosity = phasemanager.porosity();
  const std::vector<double>& phinp = *variablemanager.phinp();
  const std::vector<double>& phidtnp = *variablemanager.phidtnp();

  // saturation
  const double saturation = phasemanager.saturation(curphase);

  // inverse bulk modulus of phase (compressibility)
  const double invbulkmodulus = phasemanager.inv_bulkmodulus(curphase);

  double vtrans = 0.0;

  // TODO check for Genalpha
  // compute scalar at integration point
  vtrans = fac * phasemanager.pressure_deriv(curphase, phasetoadd) * (phinp[phasetoadd] - hist);
  for (int idof = 0; idof < numfluidphases; ++idof)
    if (idof != phasetoadd)
      vtrans += rhsfac * phasemanager.pressure_deriv(curphase, idof) * phidtnp[idof];

  vtrans *= porosity * saturation * invbulkmodulus;

  return vtrans;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassSolidPressure<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // in case of an incompressible solid (1/K_s = 0) the term cancels out
  if (phasemanager.incompressible_solid()) return;

  const int numfluidphases = phasemanager.num_fluid_phases();

  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  //  get inverse bulkmodulus (=compressiblity)
  // TODO linearization of bulkmodulus
  const double invsolidbulkmodulus = phasemanager.inv_bulkmodulus_solid();

  //----------------------------------------------------------------
  // standard Galerkin transient term
  //----------------------------------------------------------------
  {
    const double facfacmass = fac * (1.0 - phasemanager.porosity()) * invsolidbulkmodulus;
    for (int vi = 0; vi < nen; ++vi)
    {
      const double v = facfacmass * funct(vi);
      const int fvi = vi * numdofpernode + phasetoadd;

      for (int ui = 0; ui < nen; ++ui)
      {
        const double vfunct = v * funct(ui);
        for (int idof = 0; idof < numfluidphases; ++idof)
        {
          const int fui = ui * numdofpernode + idof;

          mymat(fvi, fui) += vfunct * phasemanager.solid_pressure_deriv(idof);
        }
      }
    }
  }

  // for the initial time derivative calculation we only need the standard Galerkin transient term
  // and no additional linearizations
  if (!inittimederiv)
  {
    //----------------------------------------------------------------
    // linearization of solid pressure derivative w.r.t. dof
    //----------------------------------------------------------------
    {
      const std::vector<double>& phinp = *variablemanager.phinp();
      const std::vector<double>& phidtnp = *variablemanager.phidtnp();
      double hist = 0.0;
      // if(curphase==phasetoadd) //bugfix??
      hist = (*variablemanager.hist())[phasetoadd];

      double facfacmass3 = (1.0 - phasemanager.porosity()) * invsolidbulkmodulus;

      std::vector<double> val(numfluidphases, 0.0);

      for (int idof = 0; idof < numfluidphases; ++idof)
      {
        if (idof == phasetoadd)
          for (int jdof = 0; jdof < numfluidphases; ++jdof)
            val[jdof] +=
                fac * phasemanager.solid_pressure_deriv_deriv(idof, jdof) * (phinp[idof] - hist);
        else
          for (int jdof = 0; jdof < numfluidphases; ++jdof)
            val[jdof] +=
                timefacfac * phasemanager.solid_pressure_deriv_deriv(idof, jdof) * (phidtnp[idof]);
      }

      for (int vi = 0; vi < nen; ++vi)
      {
        const double v = facfacmass3 * funct(vi);
        const int fvi = vi * numdofpernode + phasetoadd;

        for (int ui = 0; ui < nen; ++ui)
        {
          const double vfunct = v * funct(ui);
          for (int idof = 0; idof < numfluidphases; ++idof)
          {
            const int fui = ui * numdofpernode + idof;

            mymat(fvi, fui) += vfunct * val[idof];
          }
        }
      }
    }

    //----------------------------------------------------------------
    // linearization of porosity w.r.t. dof
    //----------------------------------------------------------------
    if (phasemanager.porosity_depends_on_fluid())
    {
      const std::vector<double>& phinp = *variablemanager.phinp();
      const std::vector<double>& phidtnp = *variablemanager.phidtnp();
      double hist = 0.0;
      // if(curphase==phasetoadd) //bugfix??
      hist = (*variablemanager.hist())[phasetoadd];

      double facfacmass3 = -1.0 * invsolidbulkmodulus;

      std::vector<double> val(numdofpernode, 0.0);

      for (int idof = 0; idof < numfluidphases; ++idof)
      {
        const double solidpressurederiv = phasemanager.solid_pressure_deriv(idof);
        if (idof == phasetoadd)
          for (int jdof = 0; jdof < numdofpernode; ++jdof)
            val[jdof] +=
                fac * solidpressurederiv * phasemanager.porosity_deriv(jdof) * (phinp[idof] - hist);
        else
          for (int jdof = 0; jdof < numdofpernode; ++jdof)
            val[jdof] += timefacfac * solidpressurederiv * phasemanager.porosity_deriv(jdof) *
                         (phidtnp[idof]);
      }

      for (int vi = 0; vi < nen; ++vi)
      {
        const double v = facfacmass3 * funct(vi);
        const int fvi = vi * numdofpernode + phasetoadd;

        for (int ui = 0; ui < nen; ++ui)
        {
          const double vfunct = v * funct(ui);
          for (int idof = 0; idof < numdofpernode; ++idof)
          {
            const int fui = ui * numdofpernode + idof;

            mymat(fvi, fui) += vfunct * val[idof];
          }
        }
      }
    }
  }  // !inittimederiv
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassSolidPressure<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // in case of an incompressible solid (1/K_s = 0) the term cancels out
  if (phasemanager.incompressible_solid()) return;

  // for the initial time derivative calculation no transient terms enter the rhs
  if (!inittimederiv)
  {
    // get vector to fill
    Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

    double vtrans = get_rhs_trans(
        curphase, phasetoadd, numdofpernode, phasemanager, variablemanager, rhsfac, fac);

    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi = vi * numdofpernode + phasetoadd;

      myvec[fvi] -= vtrans * funct(vi);
    }
  }  // !inittimederiv
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassSolidPressure<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // in case of an incompressible solid (1/K_s = 0) the term cancels out
  if (phasemanager.incompressible_solid()) return;

  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  double vtrans = get_rhs_trans(
      curphase, phasetoadd, numdofpernode, phasemanager, variablemanager, timefacfac, fac);

  // linearization of porosity
  //------------------------------------------------dporosity/dd = dporosity/dJ * dJ/dd =
  // dporosity/dJ * J * N_x
  // J denotes the determinant of the deformation gradient, i.e. det F = det ( d x / d X ) = det
  // (dx/ds) * ( det(dX/ds) )^-1 in our case: vtrans is scaled with (1.0-porosity) in GetRhsTrans
  // --> scale it with 1.0/(1.0-porosity) here

  if (phasemanager.porosity_depends_on_struct())
    vtrans += vtrans * (-1.0) / (1.0 - phasemanager.porosity()) * phasemanager.jacobian_def_grad() *
              phasemanager.porosity_deriv_wrt_jacobian_def_grad();

  // linearization of mesh motion (Jacobian)
  // 1) linearization of fac +
  // 2) possible linearization w.r.t porosity
  EvaluatorBase<nsd, nen>::calc_lin_fac_od_mesh(
      mymat, funct, derxy, vtrans, numdofpernode, phasetoadd);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassSolidPressure<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
double Discret::Elements::PoroFluidEvaluator::EvaluatorMassSolidPressure<nsd, nen>::get_rhs_trans(
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac)
{
  const int numfluidphases = phasemanager.num_fluid_phases();
  // read data from managers
  double hist = 0.0;
  // if(curphase==phasetoadd) //bugfix??
  hist = (*variablemanager.hist())[phasetoadd];
  const double porosity = phasemanager.porosity();
  const std::vector<double>& phinp = *variablemanager.phinp();
  const std::vector<double>& phidtnp = *variablemanager.phidtnp();

  //  get inverse bulkmodulus (=compressiblity)
  const double invsolidbulkmodulus = phasemanager.inv_bulkmodulus_solid();

  // TODO check genalpha
  // compute scalar at integration point
  double vtrans = fac * phasemanager.solid_pressure_deriv(phasetoadd) * (phinp[phasetoadd] - hist);

  for (int idof = 0; idof < numfluidphases; ++idof)
  {
    if (idof != phasetoadd)
    {
      vtrans += rhsfac * phasemanager.solid_pressure_deriv(idof) * phidtnp[idof];
    }
  }

  vtrans *= (1.0 - porosity) * invsolidbulkmodulus;

  return vtrans;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassSolidPressureSat<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // call base class with scaled factors
  EvaluatorMassSolidPressure<nsd, nen>::evaluate_matrix_and_assemble(elemat, funct, derxy, curphase,
      phasetoadd, numdofpernode, phasemanager, variablemanager,
      phasemanager.saturation(curphase) * timefacfac, phasemanager.saturation(curphase) * fac,
      inittimederiv);

  // for the initial time derivative calculation we only need the standard Galerkin transient term
  // and no additional linearizations
  if (!inittimederiv)
  {
    const int numfluidphases = phasemanager.num_fluid_phases();
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    //  get inverse bulkmodulus (=compressiblity)
    // TODO linearization of bulkmodulus
    const double invsolidbulkmodulus = phasemanager.inv_bulkmodulus_solid();

    //----------------------------------------------------------------
    // linearization of saturation w.r.t. dof
    //----------------------------------------------------------------
    {
      const std::vector<double>& phinp = *variablemanager.phinp();
      const std::vector<double>& phidtnp = *variablemanager.phidtnp();
      double hist = 0.0;
      // if(curphase==phasetoadd) //bugfix??
      hist = (*variablemanager.hist())[phasetoadd];

      double facfacmass =
          fac * phasemanager.solid_pressure_deriv(phasetoadd) * (phinp[phasetoadd] - hist);

      for (int idof = 0; idof < numfluidphases; ++idof)
      {
        if (idof != phasetoadd)
          facfacmass += timefacfac * phasemanager.solid_pressure_deriv(idof) * phidtnp[idof];
      }

      facfacmass *= (1.0 - phasemanager.porosity()) * invsolidbulkmodulus;

      // call base class for saturation linearization
      EvaluatorBase<nsd, nen>::saturation_linearization_fluid(mymat, funct, facfacmass,
          numdofpernode, numfluidphases, curphase, phasetoadd, phasemanager);
    }
  }  // !inittimederiv
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassSolidPressureSat<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // call base class with scaled factors
  EvaluatorMassSolidPressure<nsd, nen>::evaluate_vector_and_assemble(elevec, funct, derxy, xyze,
      curphase, phasetoadd, numdofpernode, phasemanager, variablemanager,
      phasemanager.saturation(curphase) * rhsfac, phasemanager.saturation(curphase) * fac,
      inittimederiv);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassSolidPressureSat<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // call base class with scaled factors
  EvaluatorMassSolidPressure<nsd, nen>::evaluate_matrix_od_struct_and_assemble(elemat, funct, deriv,
      derxy, xjm, curphase, phasetoadd, numdofpernode, phasemanager, variablemanager,
      phasemanager.saturation(curphase) * timefacfac, phasemanager.saturation(curphase) * fac, det);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassSolidPressureSat<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassSaturation<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();

  //----------------------------------------------------------------
  // linearization of saturation w.r.t. dof
  //----------------------------------------------------------------
  {
    const double facfacmass = fac * phasemanager.porosity();
    // call base class for saturation linearization
    EvaluatorBase<nsd, nen>::saturation_linearization_fluid(mymat, funct, facfacmass, numdofpernode,
        numfluidphases, curphase, phasetoadd, phasemanager);
  }

  // for the initial time derivative calculation we only need the standard Galerkin transient term
  // and no additional linearizations
  if (!inittimederiv)
  {
    //----------------------------------------------------------------
    // linearization of porosity w.r.t. dof
    //----------------------------------------------------------------
    if (phasemanager.porosity_depends_on_fluid())
    {
      // read data from manager
      double hist = 0.0;
      // if(curphase==phasetoadd) //bugfix??
      hist = (*variablemanager.hist())[phasetoadd];
      const std::vector<double>& phinp = *variablemanager.phinp();
      const std::vector<double>& phidtnp = *variablemanager.phidtnp();

      // TODO genalpha
      // compute scalar at integration point
      double facfacmass =
          fac * phasemanager.saturation_deriv(curphase, phasetoadd) * (phinp[phasetoadd] - hist);

      for (int idof = 0; idof < numfluidphases; ++idof)
        if (phasetoadd != idof)
          facfacmass += timefacfac * phasemanager.saturation_deriv(curphase, idof) * phidtnp[idof];

      // call base class:
      EvaluatorBase<nsd, nen>::porosity_linearization_fluid(
          mymat, funct, facfacmass, numdofpernode, phasetoadd, phasemanager);
    }

    //----------------------------------------------------------------
    // linearization of derivative of saturation w.r.t. dof
    //----------------------------------------------------------------
    {
      // read data from manager
      double hist = 0.0;
      hist = (*variablemanager.hist())[phasetoadd];
      const std::vector<double>& phinp = *variablemanager.phinp();
      const std::vector<double>& phidtnp = *variablemanager.phidtnp();

      /*for (int iphase=0; iphase < numdofpernode; iphase++)
      {
        std::cout << iphase << " =====================================" << std::endl;
        for (int jphase = 0; jphase < numdofpernode; jphase++)
        {
          for (int kphase = 0; kphase < numdofpernode; kphase++)
          {
            std::cout << std::setprecision(8) <<
      phasemanager.saturation_deriv_deriv(iphase,jphase,kphase) << "  ";
          }
          std::cout << "\n";
        }
      }*/

      for (int vi = 0; vi < nen; ++vi)
      {
        const double v = funct(vi);
        const int fvi = vi * numdofpernode + phasetoadd;

        for (int ui = 0; ui < nen; ++ui)
        {
          const double vfunct = v * funct(ui) * phasemanager.porosity();
          for (int idof = 0; idof < numfluidphases; ++idof)
          {
            if (idof == phasetoadd)
            {
              for (int jdof = 0; jdof < numfluidphases; ++jdof)
              {
                const int fui = ui * numdofpernode + jdof;
                mymat(fvi, fui) += fac * vfunct *
                                   phasemanager.saturation_deriv_deriv(curphase, phasetoadd, jdof) *
                                   (phinp[phasetoadd] - hist);
              }
            }
            else
            {
              for (int jdof = 0; jdof < numfluidphases; ++jdof)
              {
                const int fui = ui * numdofpernode + jdof;
                mymat(fvi, fui) += timefacfac * vfunct *
                                   phasemanager.saturation_deriv_deriv(curphase, idof, jdof) *
                                   (phidtnp[idof]);
              }
            }
          }
        }
      }
    }
  }  // !inittimederiv
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassSaturation<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // for the initial time derivative calculation no transient terms enter the rhs
  if (!inittimederiv)
  {
    // get vector to fill
    Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

    double vtrans = get_rhs_trans(
        curphase, phasetoadd, numdofpernode, phasemanager, variablemanager, rhsfac, fac);

    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi = vi * numdofpernode + phasetoadd;

      myvec[fvi] -= vtrans * funct(vi);
    }
  }  // !inittimederiv
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassSaturation<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  double vtrans = get_rhs_trans(
      curphase, phasetoadd, numdofpernode, phasemanager, variablemanager, timefacfac, fac);

  // linearization of porosity
  //------------------------------------------------dporosity/dd = dporosity/dJ * dJ/dd =
  // dporosity/dJ * J * N_x
  // J denotes the determinant of the deformation gradient, i.e. det F = det ( d x / d X ) = det
  // (dx/ds) * ( det(dX/ds) )^-1 in our case: vtrans is scaled with porosity in GetRhsTrans -->
  // scale it with 1.0/porosity here

  if (phasemanager.porosity_depends_on_struct())
    vtrans += vtrans * 1.0 / phasemanager.porosity() * phasemanager.jacobian_def_grad() *
              phasemanager.porosity_deriv_wrt_jacobian_def_grad();

  // linearization of mesh motion (Jacobian)
  // 1) linearization of fac +
  // 2) possible linearization w.r.t porosity
  EvaluatorBase<nsd, nen>::calc_lin_fac_od_mesh(
      mymat, funct, derxy, vtrans, numdofpernode, phasetoadd);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorMassSaturation<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
double Discret::Elements::PoroFluidEvaluator::EvaluatorMassSaturation<nsd, nen>::get_rhs_trans(
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac)
{
  const int numfluidphases = phasemanager.num_fluid_phases();
  // read data from manager
  double hist = 0.0;
  // if(curphase==phasetoadd) //bugfix??
  hist = (*variablemanager.hist())[phasetoadd];

  const double porosity = phasemanager.porosity();
  const std::vector<double>& phinp = *variablemanager.phinp();
  const std::vector<double>& phidtnp = *variablemanager.phidtnp();

  // TODO genalpha
  // compute scalar at integration point
  double vtrans =
      fac * phasemanager.saturation_deriv(curphase, phasetoadd) * (phinp[phasetoadd] - hist);

  for (int idof = 0; idof < numfluidphases; ++idof)
    if (phasetoadd != idof)
      vtrans += rhsfac * phasemanager.saturation_deriv(curphase, idof) * phidtnp[idof];
  // note: for one-step theta: rhsfac*phidtnp = theta*dt*(phinp-phin)/theta/dt+(1-theta)*phidtn
  //                                          = phinp - hist
  vtrans *= porosity;

  return vtrans;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorPressureAndSaturationHomogenizedVasculatureTumor<nsd,
        nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac, bool inittimederiv)
{
  // do nothing
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorPressureAndSaturationHomogenizedVasculatureTumor<nsd,
        nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
        double fac, bool inittimederiv)
{
  // get vectors to be filled
  // pressure
  Core::LinAlg::SerialDenseVector& pressure = *elevec[0];
  // saturation
  Core::LinAlg::SerialDenseVector& saturation = *elevec[1];
  // counter
  Core::LinAlg::SerialDenseVector& counter = *elevec[2];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  // FLUID Phases:
  if (curphase < numfluidphases)
  {
    for (int inode = 0; inode < nen; inode++)
    {
      // save the pressure value
      pressure[inode * numdofpernode + curphase] +=
          fac * funct(inode) * phasemanager.pressure(curphase);
      // save the saturation value
      saturation[inode * numdofpernode + curphase] +=
          fac * funct(inode) * phasemanager.saturation(curphase);
      // mark the evaluated node
      counter[inode * numdofpernode + curphase] += fac * funct(inode);
    }
  }
  // VOLFRAC Phases:
  else if (curphase < numfluidphases + numvolfrac)
  {
    // dummy way: set pressures and saturations to -1
    // TODO: is there a better way to do it ??
    for (int inode = 0; inode < nen; inode++)
    {
      // save the pressure value
      pressure[inode * numdofpernode + curphase] += fac * funct(inode) * (-1.0);
      // save the saturation value
      saturation[inode * numdofpernode + curphase] += fac * funct(inode) * (-1.0);
      // mark the evaluated node
      counter[inode * numdofpernode + curphase] += fac * funct(inode);
    }
  }
  // VOLFRAC PRESSURE Phases:
  else if (curphase < numdofpernode)
  {
    // dummy way: set saturations to -1
    // TODO: is there a better way to do it ??
    for (int inode = 0; inode < nen; inode++)
    {
      // save the pressure value
      pressure[inode * numdofpernode + curphase] +=
          fac * funct(inode) *
          phasemanager.vol_frac_pressure(curphase - numfluidphases - numvolfrac);
      // save the saturation value
      saturation[inode * numdofpernode + curphase] += fac * funct(inode) * (-1.0);
      // mark the evaluated node
      counter[inode * numdofpernode + curphase] += fac * funct(inode);
    }
  }
  else
    FOUR_C_THROW("wrong value for curphase: {}", curphase);
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorPressureAndSaturationHomogenizedVasculatureTumor<nsd,
        nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                         elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
        const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac, double det)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorPressureAndSaturationHomogenizedVasculatureTumor<nsd,
        nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                         elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorPressureAndSaturationBloodLung<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorPressureAndSaturationBloodLung<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get vectors to be filled
  // pressure
  Core::LinAlg::SerialDenseVector& pressure = *elevec[0];
  // saturation
  Core::LinAlg::SerialDenseVector& saturation = *elevec[1];
  // counter
  Core::LinAlg::SerialDenseVector& counter = *elevec[2];

  const int numfluidphases = phasemanager.num_fluid_phases();

  // FLUID Phases:
  if (curphase < numfluidphases)
  {
    for (int inode = 0; inode < nen; inode++)
    {
      // save the pressure value
      pressure[inode * numdofpernode + curphase] +=
          fac * funct(inode) * phasemanager.pressure(curphase);
      // save the saturation value
      saturation[inode * numdofpernode + curphase] +=
          fac * funct(inode) * phasemanager.saturation(curphase);
      // mark the evaluated node
      counter[inode * numdofpernode + curphase] += fac * funct(inode);
    }
  }
  // VOLFRAC PRESSURE Phases:
  else if (curphase < numdofpernode)
  {
    // dummy way: set saturations to -1
    // TODO: is there a better way to do it ??
    for (int inode = 0; inode < nen; inode++)
    {
      // save the pressure value
      pressure[inode * numdofpernode + curphase] +=
          fac * funct(inode) * phasemanager.vol_frac_pressure(curphase - numfluidphases);
      // save the saturation value
      saturation[inode * numdofpernode + curphase] += fac * funct(inode) * (-1.0);
      // mark the evaluated node
      counter[inode * numdofpernode + curphase] += fac * funct(inode);
    }
  }
  else
    FOUR_C_THROW("wrong value for curphase: {}", curphase);
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorPressureAndSaturationBloodLung<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorPressureAndSaturationBloodLung<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorSolidPressure<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // do nothing
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorSolidPressure<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get vectors to be filled
  Core::LinAlg::SerialDenseVector& solidpressure = *elevec[0];
  Core::LinAlg::SerialDenseVector& counter = *elevec[1];

  for (int inode = 0; inode < nen; inode++)
  {
    // save the pressure value
    solidpressure[inode] += fac * funct(inode) * (phasemanager.solid_pressure());
    // mark the evaluated node
    counter[inode] += fac * funct(inode);
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorSolidPressure<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 | evaluate off-diagonal coupling matrix with scatra   kremheller 06/17 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorSolidPressure<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorValidVolFracHomogenizedVasculatureTumorPressures<nsd,
        nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac, bool inittimederiv)
{
  // do nothing
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorValidVolFracHomogenizedVasculatureTumorPressures<nsd,
        nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
        double fac, bool inittimederiv)
{
  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  Core::LinAlg::SerialDenseVector& valid_volfracpress = *elevec[1];
  Core::LinAlg::SerialDenseVector& valid_volfracspec = *elevec[2];

  for (int inode = 0; inode < nen; inode++)
  {
    for (int idof = numfluidphases + numvolfrac; idof < numdofpernode; idof++)
    {
      const int fvi = inode * numdofpernode + idof;

      const bool evaluatevolfracpress =
          variablemanager.element_has_valid_vol_frac_pressure(idof - numfluidphases - numvolfrac);

      const bool evaluatevolfracspec =
          variablemanager.element_has_valid_vol_frac_species(idof - numfluidphases - numvolfrac);

      if (evaluatevolfracpress) valid_volfracpress[fvi] = 1.0;
      if (evaluatevolfracspec) valid_volfracspec[fvi] = 1.0;
    }
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorValidVolFracHomogenizedVasculatureTumorPressures<nsd,
        nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                         elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
        const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac, double det)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorValidVolFracHomogenizedVasculatureTumorPressures<nsd,
        nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                         elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac)
{
  // nothing to do
}



/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorValidVolFracPressuresBloodLung<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorValidVolFracPressuresBloodLung<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  const int numfluidphases = phasemanager.num_fluid_phases();

  Core::LinAlg::SerialDenseVector& valid_volfracpress = *elevec[1];
  Core::LinAlg::SerialDenseVector& valid_volfracspec = *elevec[2];

  for (int inode = 0; inode < nen; inode++)
  {
    for (int idof = numfluidphases; idof < numdofpernode; idof++)
    {
      const int fvi = inode * numdofpernode + idof;

      const bool evaluatevolfracpress =
          variablemanager.element_has_valid_vol_frac_pressure(idof - numfluidphases);

      const bool evaluatevolfracspec =
          variablemanager.element_has_valid_vol_frac_species(idof - numfluidphases);

      if (evaluatevolfracpress) valid_volfracpress[fvi] = 1.0;
      if (evaluatevolfracspec) valid_volfracspec[fvi] = 1.0;
    }
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorValidVolFracPressuresBloodLung<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorValidVolFracPressuresBloodLung<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
}


/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorPorosity<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // do nothing
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorPorosity<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get vectors to be filled
  Core::LinAlg::SerialDenseVector& porosity = *elevec[0];
  Core::LinAlg::SerialDenseVector& counter = *elevec[1];

  for (int inode = 0; inode < nen; inode++)
  {
    // save the porosity value
    porosity[inode] += fac * funct(inode) * phasemanager.porosity();
    // mark the evaluated node
    counter[inode] += fac * funct(inode);
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorPorosity<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorPorosity<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDeterminantOfDeformationgradient<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // do nothing
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDeterminantOfDeformationgradient<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get vectors to be filled
  Core::LinAlg::SerialDenseVector& detdefgrad = *elevec[0];
  Core::LinAlg::SerialDenseVector& counter = *elevec[1];

  for (int inode = 0; inode < nen; inode++)
  {
    // save the gp value
    detdefgrad[inode] += fac * funct(inode) * phasemanager.jacobian_def_grad();
    // mark the evaluated node
    counter[inode] += fac * funct(inode);
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDeterminantOfDeformationgradient<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDeterminantOfDeformationgradient<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolfracBloodLung<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // do nothing
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolfracBloodLung<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get vectors to be filled
  Core::LinAlg::SerialDenseVector& volfrac_blood_lung = *elevec[0];
  Core::LinAlg::SerialDenseVector& counter = *elevec[1];

  for (int inode = 0; inode < nen; inode++)
  {
    // save the gp value
    volfrac_blood_lung[inode] += fac * funct(inode) * phasemanager.vol_frac(0);
    // mark the evaluated node
    counter[inode] += fac * funct(inode);
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolfracBloodLung<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolfracBloodLung<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDomainIntegrals<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // do nothing
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDomainIntegrals<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get vector to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  // get the variables + constants
  std::vector<std::pair<std::string, double>> constants;
  // pressures + saturations + fluiddensities + porosity + volfracs + volfracpressures +
  // volfracdensities + scalars + numdim (x,y and possibly z)
  constants.reserve(3 * numfluidphases + 1 + 3 * numvolfrac + numscal_ + nsd);

  std::vector<std::pair<std::string, double>> variables;
  variables.reserve(0);

  // set pressure, saturation and density values as constants
  for (int k = 0; k < numfluidphases; k++)
  {
    constants.push_back(
        std::pair<std::string, double>("p" + std::to_string(k + 1), phasemanager.pressure(k)));
    constants.push_back(
        std::pair<std::string, double>("S" + std::to_string(k + 1), phasemanager.saturation(k)));
    constants.push_back(
        std::pair<std::string, double>("DENS" + std::to_string(k + 1), phasemanager.density(k)));
  }

  // set porosity value as constant
  constants.push_back(std::pair<std::string, double>("porosity", phasemanager.porosity()));

  // set volfrac, volfrac pressure and volfrac density values as constants
  for (int k = 0; k < numvolfrac; k++)
  {
    constants.push_back(
        std::pair<std::string, double>("VF" + std::to_string(k + 1), phasemanager.vol_frac(k)));
    constants.push_back(std::pair<std::string, double>(
        "VFP" + std::to_string(k + 1), phasemanager.vol_frac_pressure(k)));
    constants.push_back(std::pair<std::string, double>(
        "VFDENS" + std::to_string(k + 1), phasemanager.vol_frac_density(k)));
  }

  // set scalar values as constants
  for (int k = 0; k < numscal_; k++)
  {
    constants.push_back(std::pair<std::string, double>(
        "phi" + std::to_string(k + 1), variablemanager.scalarnp()->at(k)));
  }

  // calculate the coordinates of the gauss point
  std::vector<double> coords(nsd, 0.0);
  for (int idim = 0; idim < nsd; idim++)
  {
    for (int inode = 0; inode < nen; inode++)
    {
      coords[idim] += funct(inode) * xyze(idim, inode);
    }
  }

  // set values as constants in function
  constants.push_back(std::pair<std::string, double>("x", coords[0]));
  if (nsd == 2) constants.push_back(std::pair<std::string, double>("y", coords[1]));
  if (nsd == 3) constants.push_back(std::pair<std::string, double>("z", coords[2]));

  // call the functions and integrate value (multiply with fac)
  for (unsigned int i = 0; i < domainint_funct_.size(); i++)
  {
    // NOLINTNEXTLINE (bugprone-narrowing-conversions)
    myvec[i] += function(domainint_funct_[i]).evaluate(variables, constants, 0) * fac;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
inline const Core::Utils::FunctionOfAnything&
Discret::Elements::PoroFluidEvaluator::EvaluatorDomainIntegrals<nsd, nen>::function(
    int functnum) const
{
  const auto& funct =
      Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfAnything>(functnum);
  if (funct.number_components() != 1)
    FOUR_C_THROW("only one component allowed for domain integral functions");
  return funct;
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDomainIntegrals<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorDomainIntegrals<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::ReconstructFluxLinearization<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // get matrixes to fill
  Core::LinAlg::SerialDenseMatrix& linearization = *elemat[0];
  // Compute element matrix. For L2-projection
  for (int vi = 0; vi < nen; ++vi)
  {
    const double v = fac * funct(vi);
    for (int ui = 0; ui < nen; ++ui)
    {
      linearization(vi, ui) += v * funct(ui);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::ReconstructFluxLinearization<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv) {
  // nothing to do

};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::ReconstructFluxLinearization<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::ReconstructFluxLinearization<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::ReconstructFluxRHS<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // get matrixes to fill
  Core::LinAlg::SerialDenseMatrix& rhs = *elemat[1];

  const int numfluidphases = phasemanager.num_fluid_phases();

  const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradphi = *variablemanager.grad_phinp();

  // current pressure gradient
  Core::LinAlg::Matrix<nsd, 1> gradpres(Core::LinAlg::Initialization::zero);
  gradpres.clear();

  // compute the pressure gradient from the phi gradients
  for (int idof = 0; idof < numfluidphases; ++idof)
    gradpres.update(phasemanager.pressure_deriv(curphase, idof), gradphi[idof], 1.0);

  double abspressgrad = 0.0;
  for (int i = 0; i < nsd; i++) abspressgrad += gradpres(i) * gradpres(i);
  abspressgrad = sqrt(abspressgrad);

  // diffusion tensor
  Core::LinAlg::Matrix<nsd, nsd> difftensor(Core::LinAlg::Initialization::zero);
  phasemanager.permeability_tensor(curphase, difftensor);
  difftensor.scale(
      phasemanager.rel_permeability(curphase) / phasemanager.dyn_viscosity(curphase, abspressgrad));

  // diffusive flux
  Core::LinAlg::Matrix<nsd, 1> diffflux(Core::LinAlg::Initialization::zero);
  diffflux.multiply(-1.0, difftensor, gradpres);

  // Compute element vectors. For L2-Projection
  for (int node_i = 0; node_i < nen; node_i++)
  {
    for (int j = 0; j < nsd; j++)
    {
      rhs(node_i, nsd * curphase + j) += funct(node_i) * fac * diffflux(j);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::ReconstructFluxRHS<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv) {
  // nothing to do

};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::ReconstructFluxRHS<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::ReconstructFluxRHS<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorPhaseVelocitiesHomogenizedVasculatureTumor<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  Core::LinAlg::SerialDenseVector& phase_velocity = *elevec[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradient_phi = *variablemanager.grad_phinp();

  Core::LinAlg::Matrix<nsd, 1> structure_velocity(Core::LinAlg::Initialization::zero);
  if (is_ale_) structure_velocity = *variablemanager.con_velnp();

  // FLUID phases
  if (curphase < numfluidphases)
  {
    const double phase_volume_fraction =
        phasemanager.porosity() * phasemanager.saturation(curphase);

    for (int j = 0; j < nsd; j++)
    {
      if (phase_volume_fraction == 0)
      {
        phase_velocity(nsd * curphase + j) += structure_velocity(j);
      }
      else
      {
        // Compute the pressure gradient from the gradient of the generic primary variable:
        // the generic primary variable psi can be pressure, pressure difference or saturation, and
        // hence we need to employ the chain rule:
        // d p(psi_1, psi_2, psi_3)/dx = sum_i ( (p(psi_1, psi_2, psi_3)/d psi_i) * (d psi_i/dx) )
        Core::LinAlg::Matrix<nsd, 1> pressure_gradient(Core::LinAlg::Initialization::zero);
        pressure_gradient.clear();
        for (int i = 0; i < numfluidphases; ++i)
          pressure_gradient.update(phasemanager.pressure_deriv(curphase, i), gradient_phi[i], 1.0);

        Core::LinAlg::Matrix<nsd, nsd> diffusion_tensor(Core::LinAlg::Initialization::zero);
        phasemanager.permeability_tensor(curphase, diffusion_tensor);
        diffusion_tensor.scale(phasemanager.rel_permeability(curphase) /
                               phasemanager.dyn_viscosity(curphase, pressure_gradient.norm2()));

        Core::LinAlg::Matrix<nsd, 1> diffusive_velocity(Core::LinAlg::Initialization::zero);
        diffusive_velocity.multiply(
            -1.0 / phase_volume_fraction, diffusion_tensor, pressure_gradient);

        phase_velocity(nsd * curphase + j) += diffusive_velocity(j) + structure_velocity(j);
      }
    }
  }
  // VOLFRAC phases
  else if (curphase < numfluidphases + numvolfrac)
  {
    // The VOLFRAC phases only have the volume fraction as primary variable and not the pressure.
    // Hence, no velocity can be computed for these phases.
    // The corresponding velocity is computed in the VOLFRAC_PRESSURE phases (see below).
  }
  // VOLFRAC PRESSURE phases
  else if (curphase < numdofpernode)
  {
    const int i_volfrac_pressure = curphase - numfluidphases - numvolfrac;
    const double phase_volume_fraction = phasemanager.vol_frac(i_volfrac_pressure);

    for (int j = 0; j < nsd; j++)
    {
      if (phase_volume_fraction == 0)
      {
        phase_velocity(nsd * curphase + j) += structure_velocity(j);
      }
      else
      {
        // For the volume fraction, pressure is always the primary variable, and hence the gradient
        // of the primary variable directly is the pressure gradient.
        auto pressure_gradient = gradient_phi[curphase];

        Core::LinAlg::Matrix<nsd, nsd> diffusion_tensor(Core::LinAlg::Initialization::zero);
        phasemanager.permeability_tensor_vol_frac_pressure(i_volfrac_pressure, diffusion_tensor);
        diffusion_tensor.scale(1.0 / phasemanager.dyn_viscosity_vol_frac_pressure(
                                         i_volfrac_pressure, pressure_gradient.norm2()));

        Core::LinAlg::Matrix<nsd, 1> diffusive_velocity(Core::LinAlg::Initialization::zero);
        diffusive_velocity.multiply(
            -1.0 / phase_volume_fraction, diffusion_tensor, pressure_gradient);

        phase_velocity(nsd * curphase + j) += diffusive_velocity(j) + structure_velocity(j);
      }
    }
  }
  else
    FOUR_C_THROW("Invalid phase index for current phase: {}", curphase);
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorPhaseVelocitiesBloodLung<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  Core::LinAlg::SerialDenseVector& phase_velocity = *elevec[0];

  const int numfluidphases = phasemanager.num_fluid_phases();

  const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradient_phi = *variablemanager.grad_phinp();

  Core::LinAlg::Matrix<nsd, 1> structure_velocity(Core::LinAlg::Initialization::zero);
  if (is_ale_) structure_velocity = *variablemanager.con_velnp();

  // FLUID phases
  if (curphase < numfluidphases)
  {
    const double phase_volume_fraction =
        phasemanager.porosity() * phasemanager.saturation(curphase);

    for (int j = 0; j < nsd; j++)
    {
      if (phase_volume_fraction == 0)
      {
        phase_velocity(nsd * curphase + j) += structure_velocity(j);
      }
      else
      {
        // Compute the pressure gradient from the gradient of the generic primary variable:
        // the generic primary variable psi can be pressure, pressure difference or saturation, and
        // hence we need to employ the chain rule:
        // d p(psi_1, psi_2, psi_3)/dx = sum_i ( (p(psi_1, psi_2, psi_3)/d psi_i) * (d psi_i/dx) )
        Core::LinAlg::Matrix<nsd, 1> pressure_gradient(Core::LinAlg::Initialization::zero);
        pressure_gradient.clear();
        for (int i = 0; i < numfluidphases; ++i)
          pressure_gradient.update(phasemanager.pressure_deriv(curphase, i), gradient_phi[i], 1.0);

        Core::LinAlg::Matrix<nsd, nsd> diffusion_tensor(Core::LinAlg::Initialization::zero);
        phasemanager.permeability_tensor(curphase, diffusion_tensor);
        diffusion_tensor.scale(phasemanager.rel_permeability(curphase) /
                               phasemanager.dyn_viscosity(curphase, pressure_gradient.norm2()));

        Core::LinAlg::Matrix<nsd, 1> diffusive_velocity(Core::LinAlg::Initialization::zero);
        diffusive_velocity.multiply(
            -1.0 / phase_volume_fraction, diffusion_tensor, pressure_gradient);

        phase_velocity(nsd * curphase + j) += diffusive_velocity(j) + structure_velocity(j);
      }
    }
  }
  // VOLFRAC PRESSURE phases
  else if (curphase < numdofpernode)
  {
    const int i_volfrac_pressure = curphase - numfluidphases;
    const double phase_volume_fraction = phasemanager.vol_frac(i_volfrac_pressure);

    for (int j = 0; j < nsd; j++)
    {
      if (phase_volume_fraction == 0)
      {
        phase_velocity(nsd * curphase + j) += structure_velocity(j);
      }
      else
      {
        // For the volume fraction, pressure is always the primary variable, and hence the gradient
        // of the primary variable directly is the pressure gradient.
        auto pressure_gradient = gradient_phi[curphase];

        Core::LinAlg::Matrix<nsd, nsd> diffusion_tensor(Core::LinAlg::Initialization::zero);
        phasemanager.permeability_tensor_vol_frac_pressure(i_volfrac_pressure, diffusion_tensor);
        diffusion_tensor.scale(1.0 / phasemanager.dyn_viscosity_vol_frac_pressure_blood_lung(
                                         i_volfrac_pressure, pressure_gradient.norm2()));

        Core::LinAlg::Matrix<nsd, 1> diffusive_velocity(Core::LinAlg::Initialization::zero);
        diffusive_velocity.multiply(
            -1.0 / phase_volume_fraction, diffusion_tensor, pressure_gradient);

        phase_velocity(nsd * curphase + j) += diffusive_velocity(j) + structure_velocity(j);
      }
    }
  }
  else
    FOUR_C_THROW("Invalid phase index for current phase: {}", curphase);
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTerms<nsd,
        nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];
  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  //----------------------------------------------------------------
  // 1) standard Galerkin transient term
  //----------------------------------------------------------------
  {
    const double facfacmass = -fac;
    for (int vi = 0; vi < nen; ++vi)
    {
      const double v = facfacmass * funct(vi);
      const int fvi = vi * numdofpernode + phasetoadd;

      for (int ui = 0; ui < nen; ++ui)
      {
        const double vfunct = v * funct(ui);
        for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ++ivolfrac)
        {
          const int fui = ui * numdofpernode + ivolfrac;

          mymat(fvi, fui) += vfunct;
        }
      }
    }
  }

  //----------------------------------------------------------------
  // 2) - sum_volfrac porosity_volfrac/K_s * d p_s / d t
  //----------------------------------------------------------------
  if (not phasemanager.incompressible_solid())
  {
    const double invsolidbulkmodulus = phasemanager.inv_bulkmodulus_solid();
    const double sumaddvolfrac = phasemanager.sum_add_vol_frac();

    //----------------------------------------------------------------
    // standard Galerkin transient term
    //----------------------------------------------------------------
    {
      const double facfacmass = fac * (-sumaddvolfrac) * invsolidbulkmodulus;
      for (int vi = 0; vi < nen; ++vi)
      {
        const double v = facfacmass * funct(vi);
        const int fvi = vi * numdofpernode + phasetoadd;

        for (int ui = 0; ui < nen; ++ui)
        {
          const double vfunct = v * funct(ui);
          for (int idof = 0; idof < numfluidphases; ++idof)
          {
            const int fui = ui * numdofpernode + idof;

            mymat(fvi, fui) += vfunct * phasemanager.solid_pressure_deriv(idof);
          }
        }
      }
    }
    // for the initial time derivative calculation no additional derivatives are needed
    if (!inittimederiv)
    {
      //----------------------------------------------------------------
      // linearization of solid pressure derivative w.r.t. dof
      //----------------------------------------------------------------
      {
        const std::vector<double>& phinp = *variablemanager.phinp();
        const std::vector<double>& phidtnp = *variablemanager.phidtnp();
        double hist = 0.0;
        // if(curphase==phasetoadd) //bugfix??
        hist = (*variablemanager.hist())[phasetoadd];

        double facfacmass3 = -sumaddvolfrac * invsolidbulkmodulus;

        std::vector<double> val(numfluidphases, 0.0);

        for (int idof = 0; idof < numfluidphases; ++idof)
        {
          if (idof == phasetoadd)
            for (int jdof = 0; jdof < numfluidphases; ++jdof)
              val[jdof] +=
                  fac * phasemanager.solid_pressure_deriv_deriv(idof, jdof) * (phinp[idof] - hist);
          else
            for (int jdof = 0; jdof < numfluidphases; ++jdof)
              val[jdof] += timefacfac * phasemanager.solid_pressure_deriv_deriv(idof, jdof) *
                           (phidtnp[idof]);
        }

        for (int vi = 0; vi < nen; ++vi)
        {
          const double v = facfacmass3 * funct(vi);
          const int fvi = vi * numdofpernode + phasetoadd;

          for (int ui = 0; ui < nen; ++ui)
          {
            const double vfunct = v * funct(ui);
            for (int idof = 0; idof < numfluidphases; ++idof)
            {
              const int fui = ui * numdofpernode + idof;

              mymat(fvi, fui) += vfunct * val[idof];
            }
          }
        }
      }
      //----------------------------------------------------------------
      // linearization of sum_volfrac porosity_volfrac w.r.t. dof
      //----------------------------------------------------------------
      double hist = 0.0;
      // if(curphase==phasetoadd) //bugfix??
      hist = (*variablemanager.hist())[phasetoadd];
      const std::vector<double>& phinp = *variablemanager.phinp();
      const std::vector<double>& phidtnp = *variablemanager.phidtnp();

      // TODO check genalpha
      // compute scalar at integration point
      double vtrans =
          fac * phasemanager.solid_pressure_deriv(phasetoadd) * (phinp[phasetoadd] - hist);

      for (int idof = 0; idof < numfluidphases; ++idof)
      {
        if (idof != phasetoadd)
        {
          vtrans += timefacfac * phasemanager.solid_pressure_deriv(idof) * phidtnp[idof];
        }
      }
      vtrans *= -invsolidbulkmodulus;
      for (int vi = 0; vi < nen; ++vi)
      {
        const double v = vtrans * funct(vi);
        const int fvi = vi * numdofpernode + phasetoadd;

        for (int ui = 0; ui < nen; ++ui)
        {
          const double vfunct = v * funct(ui);
          for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ++ivolfrac)
          {
            const int fui = ui * numdofpernode + ivolfrac;

            mymat(fvi, fui) += vfunct;
          }
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTerms<nsd,
        nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
        double fac, bool inittimederiv)
{
  // get vector to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  // for the initial time derivative calculation no transient terms enter the rhs
  if (!inittimederiv)
  {
    const double vrhs =
        get_rhs(curphase, phasetoadd, numdofpernode, phasemanager, variablemanager, rhsfac, fac);

    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi = vi * numdofpernode + phasetoadd;

      myvec[fvi] -= vrhs * funct(vi);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTerms<nsd,
        nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                         elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
        const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const double vrhs =
      get_rhs(curphase, phasetoadd, numdofpernode, phasemanager, variablemanager, timefacfac, fac);

  // linearization of mesh motion (Jacobian)
  EvaluatorBase<nsd, nen>::calc_lin_fac_od_mesh(
      mymat, funct, derxy, vrhs, numdofpernode, phasetoadd);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTerms<nsd,
        nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                         elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
double Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTerms<nsd, nen>::get_rhs(int curphase,
        int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
        double fac)
{
  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  const std::vector<double>& phidtnp = *variablemanager.phidtnp();

  double vrhs = 0.0;

  // sum^volfrac \frac{\partial phi_volfrac}{\partial t}
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
    vrhs -= rhsfac * phidtnp[ivolfrac];

  // \frac{-\sum^volfrac \phi_volfrac) }{K_s} \frac{\partial p^s}{\partial t}
  if (not phasemanager.incompressible_solid())
  {
    double hist = 0.0;
    // if(curphase==phasetoadd) //bugfix??
    hist = (*variablemanager.hist())[phasetoadd];
    const std::vector<double>& phinp = *variablemanager.phinp();

    //  get inverse bulkmodulus (=compressiblity)
    const double invsolidbulkmodulus = phasemanager.inv_bulkmodulus_solid();

    // TODO check genalpha
    // compute scalar at integration point
    double vtrans =
        fac * phasemanager.solid_pressure_deriv(phasetoadd) * (phinp[phasetoadd] - hist);

    for (int idof = 0; idof < numfluidphases; ++idof)
    {
      if (idof != phasetoadd)
      {
        vtrans += rhsfac * phasemanager.solid_pressure_deriv(idof) * phidtnp[idof];
      }
    }
    vtrans *= -invsolidbulkmodulus;
    const double sumaddvolfrac = phasemanager.sum_add_vol_frac();
    vrhs += vtrans * sumaddvolfrac;
  }

  return vrhs;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddInstatTerms<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const std::vector<double>& phi = *variablemanager.phinp();
  const std::vector<double>& phidt = *variablemanager.phidtnp();
  double scaling_factor = phasemanager.initial_volfrac() *
                          pow(phasemanager.jacobian_def_grad(),
                              phasemanager.volfrac_blood_lung_parameter_deformation_dependence());
  double pA = phi[0];
  double pB = phi[numfluidphases];
  double ratio_pApB = pA / pB;  // for this closing relation: air must be the first phase in
                                // multiphase porespace and blood must be the first phase in
                                // the additional porous network
  double dpAdt = phidt[0];
  double dpBdt = phidt[numfluidphases];

  if (ratio_pApB > 1.0)  // collapse of blood vessels active, pA > pB
  {
    for (int vi = 0; vi < nen; ++vi)
    {
      const double v = -timefacfac * scaling_factor * funct(vi);
      const int fvi = vi * numdofpernode + phasetoadd;

      for (int ui = 0; ui < nen; ++ui)
      {
        const double vfunct = v * funct(ui);

        const int fuiAir = ui * numdofpernode + 0;
        const int fuiBlood = ui * numdofpernode + numfluidphases;

        mymat(fvi, fuiAir) +=
            vfunct *
            (variablemanager.div_con_velnp() *
                    (phasemanager.volfrac_blood_lung_parameter_deformation_dependence() *
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    pow(ratio_pApB,
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 1.0) *
                    (1.0 / pB) +
                (phasemanager.volfrac_blood_lung_parameter_pressure_dependence() *
                    (phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 1.0)) *
                    pow(ratio_pApB,
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 2.0) *
                    pow(pB, -2.0) * dpAdt +
                (phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    pow(ratio_pApB,
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 1.0) *
                    pow(pB, -1.0) * (fac / timefacfac) -
                (phasemanager.volfrac_blood_lung_parameter_pressure_dependence() *
                    (phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 1.0)) *
                    pow(ratio_pApB,
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 1.0) *
                    pow(pB, -2.0) * dpBdt -
                (phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    pow(ratio_pApB,
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 1.0) *
                    pow(pB, -2.0) * dpBdt);

        mymat(fvi, fuiBlood) +=
            vfunct *
            (variablemanager.div_con_velnp() * (-1.0) *
                    (phasemanager.volfrac_blood_lung_parameter_pressure_dependence() *
                        phasemanager.volfrac_blood_lung_parameter_deformation_dependence()) *
                    pow(ratio_pApB,
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    pow(pB, -1.0) +
                (-1.0) *
                    ((phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 1.0) *
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    pow(ratio_pApB,
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 1.0) *
                    pow(pB, -2.0) * dpAdt +
                (-1.0) * (phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    pow(ratio_pApB,
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 1.0) *
                    pow(pB, -2.0) * dpAdt +
                ((phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 1.0) *
                    phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    pow(ratio_pApB,
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    pow(pB, -2.0) * dpBdt +
                (phasemanager.volfrac_blood_lung_parameter_pressure_dependence() * 2.0) *
                    pow(ratio_pApB,
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    pow(pB, -2.0) * dpBdt -
                (phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    pow(ratio_pApB,
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    pow(pB, -1.0) * (fac / timefacfac));
      }
    }
  }
  else
  {
    return;
  }
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddInstatTerms<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)

{
  // first step!
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const std::vector<double>& phi = *variablemanager.phinp();
  const std::vector<double>& phidt = *variablemanager.phidtnp();
  double pA = phi[0];
  double pB = phi[numfluidphases];
  double dpAdt = phidt[0];
  double dpBdt = phidt[numfluidphases];
  double ratio_pApB = phi[0] / phi[numfluidphases];

  Core::LinAlg::Matrix<nsd, nsd> gridvelderiv(Core::LinAlg::Initialization::zero);
  gridvelderiv.multiply_nt(*(variablemanager.e_con_velnp()), deriv);

  double scalingfacor = phasemanager.initial_volfrac() *
                        phasemanager.volfrac_blood_lung_parameter_deformation_dependence() *
                        pow(phasemanager.jacobian_def_grad(),
                            phasemanager.volfrac_blood_lung_parameter_deformation_dependence());

  if (ratio_pApB > 1.0)  // collapse of blood vessels active, pA > pB
  {
    double scalingfacormodified =
        scalingfacor *
        pow(ratio_pApB, phasemanager.volfrac_blood_lung_parameter_pressure_dependence());

    // OD mesh - div vel term
    EvaluatorBase<nsd, nen>::calc_div_vel_od_mesh(mymat, funct, deriv, derxy, xjm, gridvelderiv,
        timefacfac * scalingfacormodified * (-1.0), fac * scalingfacormodified * (-1.0), det,
        numdofpernode, phasetoadd);

    double helper =
        phasemanager.initial_volfrac() *
        pow(phasemanager.jacobian_def_grad(),
            phasemanager.volfrac_blood_lung_parameter_deformation_dependence()) *
        (variablemanager.div_con_velnp() *
                pow(ratio_pApB, phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                (phasemanager.volfrac_blood_lung_parameter_deformation_dependence()) +
            ((phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                pow(ratio_pApB,
                    phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 1.0)) *
                (pow(pB, -1.0) * dpAdt - pA * pow(pB, -2.0) * dpBdt));

    // OD mesh - rest term
    EvaluatorBase<nsd, nen>::calc_lin_fac_od_mesh(
        mymat, funct, derxy, -1.0 * helper, numdofpernode, phasetoadd);

    // derivative of scaling factor w.r.t. mesh motion
    double scalingderiv =
        (-1.0) *
        (phasemanager.initial_volfrac() *
            phasemanager.volfrac_blood_lung_parameter_deformation_dependence() *
            pow(phasemanager.jacobian_def_grad(),
                phasemanager.volfrac_blood_lung_parameter_deformation_dependence()) *
            timefacfac *
            (variablemanager.div_con_velnp() *
                    pow(ratio_pApB,
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    (phasemanager.volfrac_blood_lung_parameter_deformation_dependence()) +
                (phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    pow(ratio_pApB,
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 1.0) *
                    pow(pB, -1.0) * dpAdt +
                (phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    pow(ratio_pApB,
                        phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                    pow(pB, -1.0) * dpBdt));

    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi = vi * numdofpernode + phasetoadd;
      const double v = scalingderiv * funct(vi);

      for (int ui = 0; ui < nen; ++ui)
      {
        for (int idim = 0; idim < nsd; ++idim)
        {
          const int fui = ui * nsd + idim;
          mymat(fvi, fui) += v * derxy(idim, ui);
        }
      }
    }
  }
  else  // no collapse of blood vessels
  {
    // OD mesh - div vel term
    EvaluatorBase<nsd, nen>::calc_div_vel_od_mesh(mymat, funct, deriv, derxy, xjm, gridvelderiv,
        timefacfac * scalingfacor * (-1.0), fac * scalingfacor * (-1.0), det, numdofpernode,
        phasetoadd);

    double scalingfactorderiv =
        (1.0) * phasemanager.initial_volfrac() *
        (phasemanager.volfrac_blood_lung_parameter_deformation_dependence() *
            phasemanager.volfrac_blood_lung_parameter_deformation_dependence()) *
        pow(phasemanager.jacobian_def_grad(),
            phasemanager.volfrac_blood_lung_parameter_deformation_dependence()) *
        variablemanager.div_con_velnp() * timefacfac;

    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi = vi * numdofpernode + phasetoadd;
      const double v = scalingfactorderiv * funct(vi);

      for (int ui = 0; ui < nen; ++ui)
      {
        for (int idim = 0; idim < nsd; ++idim)
        {
          const int fui = ui * nsd + idim;
          mymat(fvi, fui) -= v * derxy(idim, ui);
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddInstatTerms<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}


template <int nsd, int nen>
double
Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddInstatTerms<nsd, nen>::get_rhs(
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac)
{
  // - \frac{\partial volfrac}{\partial t}
  double divvel = variablemanager.div_con_velnp();

  const int numfluidphases = phasemanager.num_fluid_phases();
  const std::vector<double>& phi = *variablemanager.phinp();
  double vrhs = 0.0;
  double scaling_factor = phasemanager.initial_volfrac() *
                          pow(phasemanager.jacobian_def_grad(),
                              phasemanager.volfrac_blood_lung_parameter_deformation_dependence());

  if (phi[0] / phi[numfluidphases] > 1.0)
  {
    vrhs = (-1.0) *
           ((phasemanager.volfrac_blood_lung_parameter_deformation_dependence()) *
                   pow(phi[0] / phi[numfluidphases],
                       phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                   divvel +
               phasemanager.volfrac_blood_lung_parameter_pressure_dependence() *
                   pow(phi[0] / phi[numfluidphases],
                       phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 1.0) *
                   (1.0 / phi[numfluidphases]) * (*variablemanager.phidtnp())[0] +
               (phasemanager.volfrac_blood_lung_parameter_pressure_dependence()) *
                   pow(phi[0] / phi[numfluidphases],
                       phasemanager.volfrac_blood_lung_parameter_pressure_dependence() - 1.0) *
                   phi[0] * pow(phi[numfluidphases], -2.0) *
                   (*variablemanager.phidtnp())[numfluidphases]) *
           rhsfac * scaling_factor;
  }
  else
  {
    vrhs = (-1.0) * phasemanager.volfrac_blood_lung_parameter_deformation_dependence() *
           scaling_factor * divvel * rhsfac;
  }

  if (not phasemanager.incompressible_solid())
  {
    FOUR_C_THROW(
        "CompressibleSolid not yet implemented for deformation dependent closing relation!");
  }

  return vrhs;
}



template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddInstatTerms<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)

{
  // get matrix to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  // for the initial time derivative calculation no transient terms enter the rhs
  if (!inittimederiv)
  {
    const double vrhs =
        get_rhs(curphase, phasetoadd, numdofpernode, phasemanager, variablemanager, rhsfac, fac);

    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi = vi * numdofpernode + phasetoadd;

      myvec[fvi] -= vrhs * funct(vi);
    }
  }
}



/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddInstatTermsSat<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  EvaluatorVolFracBloodLungAddInstatTerms<nsd, nen>::evaluate_matrix_and_assemble(elemat, funct,
      derxy, curphase, phasetoadd, numdofpernode, phasemanager, variablemanager,
      timefacfac * phasemanager.saturation(curphase), fac * phasemanager.saturation(curphase),
      inittimederiv);

  // we do not need additional linearizations if we calculate the initial time derivative
  if (!inittimederiv)
  {
    const int numfluidphases = phasemanager.num_fluid_phases();
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];
    Core::LinAlg::Matrix<nsd, nsd> gridvelderiv;
    //----------------------------------------------------------------
    // linearization of saturation w.r.t. dof
    //----------------------------------------------------------------

    // first: get rhs
    const double vrhs = EvaluatorVolFracBloodLungAddInstatTerms<nsd, nen>::get_rhs(
        curphase, phasetoadd, numdofpernode, phasemanager, variablemanager, timefacfac, fac);

    // call base class for saturation linearization
    EvaluatorBase<nsd, nen>::saturation_linearization_fluid(
        mymat, funct, vrhs, numdofpernode, numfluidphases, curphase, phasetoadd, phasemanager);
  }
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddInstatTermsSat<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)

{
  // call base class with scaled factors
  EvaluatorVolFracBloodLungAddInstatTerms<nsd, nen>::evaluate_matrix_od_struct_and_assemble(elemat,
      funct, deriv, derxy, xjm, curphase, phasetoadd, numdofpernode, phasemanager, variablemanager,
      phasemanager.saturation(curphase) * timefacfac, phasemanager.saturation(curphase) * fac, det);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddInstatTermsSat<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
}


template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddInstatTermsSat<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)

{
  // call base class with scaled factors
  EvaluatorVolFracBloodLungAddInstatTerms<nsd, nen>::evaluate_vector_and_assemble(elevec, funct,
      derxy, xyze, curphase, phasetoadd, numdofpernode, phasemanager, variablemanager,
      phasemanager.saturation(curphase) * rhsfac, phasemanager.saturation(curphase) * fac,
      inittimederiv);
}



/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTerm<nsd,
        nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac, bool inittimederiv)
{
  // we do not need the matrix if we calculate the initial time derivative
  if (!inittimederiv)
  {
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];
    const int numfluidphases = phasemanager.num_fluid_phases();
    const int numvolfrac = phasemanager.num_vol_frac();

    //----------------------------------------------------------------
    // - sum_volfrac porosity_volfrac * div v_s
    //----------------------------------------------------------------
    {
      const double prefac = -timefacfac * variablemanager.div_con_velnp();
      for (int vi = 0; vi < nen; ++vi)
      {
        const double v = prefac * funct(vi);
        const int fvi = vi * numdofpernode + phasetoadd;

        for (int ui = 0; ui < nen; ++ui)
        {
          const double vfunct = v * funct(ui);
          for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ++ivolfrac)
          {
            const int fui = ui * numdofpernode + ivolfrac;

            mymat(fvi, fui) += vfunct;
          }
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTerm<nsd,
        nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
        double fac, bool inittimederiv)
{
  // get vector to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  const double vrhs = -rhsfac * phasemanager.sum_add_vol_frac() * variablemanager.div_con_velnp();

  for (int vi = 0; vi < nen; ++vi)
  {
    const int fvi = vi * numdofpernode + phasetoadd;

    myvec[fvi] -= vrhs * funct(vi);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTerm<nsd,
        nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                         elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
        const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const double sumaddvolfrac = phasemanager.sum_add_vol_frac();

  Core::LinAlg::Matrix<nsd, nsd> gridvelderiv(Core::LinAlg::Initialization::zero);
  gridvelderiv.multiply_nt(*(variablemanager.e_con_velnp()), deriv);

  // OD mesh - div vel term
  EvaluatorBase<nsd, nen>::calc_div_vel_od_mesh(mymat, funct, deriv, derxy, xjm, gridvelderiv,
      timefacfac * sumaddvolfrac * (-1.0), fac * sumaddvolfrac * (-1.0), det, numdofpernode,
      phasetoadd);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTerm<nsd,
        nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                         elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddDivVelTerm<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];
  const int numfluidphases = phasemanager.num_fluid_phases();
  double pA = (*variablemanager.phinp())[0];
  double pB = (*variablemanager.phinp())[numfluidphases];
  double ratio_pApB = pA / pB;
  if (ratio_pApB > 1.0)
  {
    double scalingfacor = phasemanager.initial_volfrac() *
                          pow(phasemanager.jacobian_def_grad(),
                              phasemanager.volfrac_blood_lung_parameter_deformation_dependence());
    const double prefac = -timefacfac * variablemanager.div_con_velnp() * scalingfacor;
    for (int vi = 0; vi < nen; ++vi)
    {
      const double v = prefac * funct(vi);
      const int fvi = vi * numdofpernode + phasetoadd;
      for (int ui = 0; ui < nen; ++ui)
      {
        const double vfunct = v * funct(ui);
        const int fuiAir = ui * numdofpernode + 0;
        const int fuiBlood = ui * numdofpernode + numfluidphases;
        mymat(fvi, fuiAir) +=
            vfunct *
            pow(ratio_pApB,
                phasemanager.volfrac_blood_lung_parameter_deformation_dependence() - 1.0) *
            (phasemanager.volfrac_blood_lung_parameter_deformation_dependence()) * (1.0 / pB);
        mymat(fvi, fuiBlood) +=
            vfunct * (-1.0) * (phasemanager.volfrac_blood_lung_parameter_deformation_dependence()) *
            pow(ratio_pApB,
                phasemanager.volfrac_blood_lung_parameter_deformation_dependence() - 1.0) *
            pA * pow(pB, -2.0);
      }
    }
  }
  else
  {
    return;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddDivVelTerm<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // call base class
  EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTerm<nsd, nen>::evaluate_vector_and_assemble(
      elevec, funct, derxy, xyze, curphase, phasetoadd, numdofpernode, phasemanager,
      variablemanager, rhsfac, fac, inittimederiv);
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddDivVelTerm<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // d(sumvolfrac * divvel)dd = sumvolfrac * ddivveldd + dsumvolfracdd * divvel
  // class Based class for sumvolfrac * ddivveldd
  EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTerm<nsd,
      nen>::evaluate_matrix_od_struct_and_assemble(elemat, funct, deriv, derxy, xjm, curphase,
      phasetoadd, numdofpernode, phasemanager, variablemanager, timefacfac, fac, det);


  // get matrix to fill for dsumvolfracdd * divvel
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  for (int vi = 0; vi < nen; ++vi)
  {
    const int fvi = vi * numdofpernode + phasetoadd;
    const double v = -timefacfac * phasemanager.sum_add_vol_frac() *
                     (phasemanager.volfrac_blood_lung_parameter_deformation_dependence()) *
                     variablemanager.div_con_velnp() * funct(vi);

    for (int ui = 0; ui < nen; ++ui)
    {
      for (int idim = 0; idim < nsd; ++idim)
      {
        const int fui = ui * nsd + idim;
        mymat(fvi, fui) += v * derxy(idim, ui);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddDivVelTerm<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddDivVelTermSat<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // call base class with scaled factors
  EvaluatorVolFracBloodLungAddDivVelTerm<nsd, nen>::evaluate_matrix_and_assemble(elemat, funct,
      derxy, curphase, phasetoadd, numdofpernode, phasemanager, variablemanager,
      timefacfac * phasemanager.saturation(curphase), fac * phasemanager.saturation(curphase),
      inittimederiv);

  // we do not need additional linearizations if we calculate the initial time derivative
  if (!inittimederiv)
  {
    const int numfluidphases = phasemanager.num_fluid_phases();
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    const double vrhs =
        -timefacfac * phasemanager.sum_add_vol_frac() * variablemanager.div_con_velnp();

    // call base class for saturation linearization
    EvaluatorBase<nsd, nen>::saturation_linearization_fluid(
        mymat, funct, vrhs, numdofpernode, numfluidphases, curphase, phasetoadd, phasemanager);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddDivVelTermSat<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // call base class with scaled factors
  EvaluatorVolFracBloodLungAddDivVelTerm<nsd, nen>::evaluate_vector_and_assemble(elevec, funct,
      derxy, xyze, curphase, phasetoadd, numdofpernode, phasemanager, variablemanager,
      phasemanager.saturation(curphase) * rhsfac, phasemanager.saturation(curphase) * fac,
      inittimederiv);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddDivVelTermSat<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // call base class with scaled factors
  EvaluatorVolFracBloodLungAddDivVelTerm<nsd, nen>::evaluate_matrix_od_struct_and_assemble(elemat,
      funct, deriv, derxy, xjm, curphase, phasetoadd, numdofpernode, phasemanager, variablemanager,
      phasemanager.saturation(curphase) * timefacfac, phasemanager.saturation(curphase) * fac, det);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungAddDivVelTermSat<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
}


/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTermsSat<nsd,
        nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac, bool inittimederiv)
{
  // call base class with scaled factors
  EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTerms<nsd, nen>::evaluate_matrix_and_assemble(
      elemat, funct, derxy, curphase, phasetoadd, numdofpernode, phasemanager, variablemanager,
      timefacfac * phasemanager.saturation(curphase), fac * phasemanager.saturation(curphase),
      inittimederiv);

  // we do not need additional linearizations if we calculate the initial time derivative
  if (!inittimederiv)
  {
    const int numfluidphases = phasemanager.num_fluid_phases();
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    //----------------------------------------------------------------
    // linearization of saturation w.r.t. dof
    //----------------------------------------------------------------

    // first: get rhs
    const double vrhs =
        EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTerms<nsd, nen>::get_rhs(
            curphase, phasetoadd, numdofpernode, phasemanager, variablemanager, timefacfac, fac);

    // call base class for saturation linearization
    EvaluatorBase<nsd, nen>::saturation_linearization_fluid(
        mymat, funct, vrhs, numdofpernode, numfluidphases, curphase, phasetoadd, phasemanager);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTermsSat<nsd,
        nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
        double fac, bool inittimederiv)
{
  // call base class with scaled factors
  EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTerms<nsd, nen>::evaluate_vector_and_assemble(
      elevec, funct, derxy, xyze, curphase, phasetoadd, numdofpernode, phasemanager,
      variablemanager, phasemanager.saturation(curphase) * rhsfac,
      phasemanager.saturation(curphase) * fac, inittimederiv);
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTermsSat<nsd,
        nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                         elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
        const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac, double det)
{
  // call base class with scaled factors
  EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTerms<nsd,
      nen>::evaluate_matrix_od_struct_and_assemble(elemat, funct, deriv, derxy, xjm, curphase,
      phasetoadd, numdofpernode, phasemanager, variablemanager,
      phasemanager.saturation(curphase) * timefacfac, phasemanager.saturation(curphase) * fac, det);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddInstatTermsSat<nsd,
        nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                         elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTermSat<nsd,
        nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac, bool inittimederiv)
{
  // call base class with scaled factors
  EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTerm<nsd, nen>::evaluate_matrix_and_assemble(
      elemat, funct, derxy, curphase, phasetoadd, numdofpernode, phasemanager, variablemanager,
      timefacfac * phasemanager.saturation(curphase), fac * phasemanager.saturation(curphase),
      inittimederiv);

  // we do not need additional linearizations if we calculate the initial time derivative
  if (!inittimederiv)
  {
    const int numfluidphases = phasemanager.num_fluid_phases();
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    const double vrhs =
        -timefacfac * phasemanager.sum_add_vol_frac() * variablemanager.div_con_velnp();

    // call base class for saturation linearization
    EvaluatorBase<nsd, nen>::saturation_linearization_fluid(
        mymat, funct, vrhs, numdofpernode, numfluidphases, curphase, phasetoadd, phasemanager);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTermSat<nsd,
        nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
        double fac, bool inittimederiv)
{
  // call base class with scaled factors
  EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTerm<nsd, nen>::evaluate_vector_and_assemble(
      elevec, funct, derxy, xyze, curphase, phasetoadd, numdofpernode, phasemanager,
      variablemanager, phasemanager.saturation(curphase) * rhsfac,
      phasemanager.saturation(curphase) * fac, inittimederiv);
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTermSat<nsd,
        nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                         elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
        const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac, double det)
{
  // call base class with scaled factors
  EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTerm<nsd,
      nen>::evaluate_matrix_od_struct_and_assemble(elemat, funct, deriv, derxy, xjm, curphase,
      phasetoadd, numdofpernode, phasemanager, variablemanager,
      phasemanager.saturation(curphase) * timefacfac, phasemanager.saturation(curphase) * fac, det);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::
    EvaluatorVolFracHomogenizedVasculatureTumorAddDivVelTermSat<nsd,
        nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                         elemat,
        const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
        int curphase, int phasetoadd, int numdofpernode,
        const PoroFluidManager::PhaseManagerInterface& phasemanager,
        const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager,
        double timefacfac, double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorInstat<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  // loop over all volume fractions
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
  {
    const bool evaluatevolfracpress =
        variablemanager.element_has_valid_vol_frac_pressure(ivolfrac - numfluidphases);

    //----------------------------------------------------------------
    // standard Galerkin transient term
    //----------------------------------------------------------------
    for (int vi = 0; vi < nen; ++vi)
    {
      const double v = funct(vi) * fac;
      const int fvi_volfrac = vi * numdofpernode + ivolfrac;
      const int fvi_volfracpress = vi * numdofpernode + numvolfrac + ivolfrac;

      for (int ui = 0; ui < nen; ++ui)
      {
        const int fui = ui * numdofpernode + ivolfrac;

        mymat(fvi_volfrac, fui) += v * funct(ui);
        if (evaluatevolfracpress) mymat(fvi_volfracpress, fui) += v * funct(ui);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorInstat<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // for the initial time derivative calculation no transient terms enter the rhs
  if (!inittimederiv)
  {
    // get vector to fill
    Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

    const int numfluidphases = phasemanager.num_fluid_phases();
    const int numvolfrac = phasemanager.num_vol_frac();

    // loop over all volume fractions
    for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
    {
      const double hist = (*variablemanager.hist())[ivolfrac];
      const double phinp = phasemanager.vol_frac(ivolfrac - numfluidphases);
      const double vtrans = fac * (phinp - hist);

      const bool evaluatevolfracpress =
          variablemanager.element_has_valid_vol_frac_pressure(ivolfrac - numfluidphases);

      for (int vi = 0; vi < nen; ++vi)
      {
        const int fvi_volfrac = vi * numdofpernode + ivolfrac;
        const int fvi_volfracpress = vi * numdofpernode + numvolfrac + ivolfrac;

        myvec[fvi_volfrac] -= vtrans * funct(vi);
        if (evaluatevolfracpress) myvec[fvi_volfracpress] -= vtrans * funct(vi);
      }
    }
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorInstat<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  // loop over all volume fractions
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
  {
    const double hist = (*variablemanager.hist())[ivolfrac];
    const double phinp = phasemanager.vol_frac(ivolfrac - numfluidphases);
    const double vtrans = fac * (phinp - hist);

    const bool evaluatevolfracpress =
        variablemanager.element_has_valid_vol_frac_pressure(ivolfrac - numfluidphases);

    // linearization of mesh motion
    //------------------------------------------------dJ/dd = dJ/dF : dF/dd = J * F^-T . N_{\psi}
    //=
    // J * N_x
    // J denotes the determinant of the Jacobian of the mapping between current and parameter
    // space, i.e. det(dx/ds) in our case: timefacfac = J * dt * theta --> d(timefacfac)/dd =
    // timefacfac * N_x
    //              fac        = J              --> d(fac)/dd        = fac * N_x
    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi_volfrac = vi * numdofpernode + ivolfrac;
      const int fvi_volfracpress = vi * numdofpernode + numvolfrac + ivolfrac;

      const double v = funct(vi) * vtrans;

      for (int ui = 0; ui < nen; ++ui)
      {
        for (int idim = 0; idim < nsd; ++idim)
        {
          const int fui = ui * nsd + idim;

          mymat(fvi_volfrac, fui) += v * derxy(idim, ui);
          if (evaluatevolfracpress) mymat(fvi_volfracpress, fui) += v * derxy(idim, ui);
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorInstat<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}



/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungInstat<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  // loop over all volume fractions
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
  {
    const bool evaluatevolfracpress =
        variablemanager.element_has_valid_vol_frac_pressure(ivolfrac - numfluidphases);

    if (evaluatevolfracpress)
    {
      EvaluatorVolFracBloodLungAddInstatTerms<nsd, nen>::evaluate_matrix_and_assemble(elemat, funct,
          derxy, curphase, ivolfrac, numdofpernode, phasemanager, variablemanager,
          timefacfac * (-1.0), fac * (-1.0), inittimederiv);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungInstat<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // for the initial time derivative calculation no transient terms enter the rhs
  if (!inittimederiv)
  {
    // get vector to fill
    Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

    const int numfluidphases = phasemanager.num_fluid_phases();

    // currently only one volfrac blood ung material possible
    int ivolfrac = numfluidphases;
    // - \frac{\partial volfrac}{\partial t}
    const double vtrans = EvaluatorVolFracBloodLungAddInstatTerms<nsd, nen>::get_rhs(
        curphase, phasetoadd, numdofpernode, phasemanager, variablemanager, rhsfac, fac);

    const bool evaluatevolfracpress =
        variablemanager.element_has_valid_vol_frac_pressure(ivolfrac - numfluidphases);

    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi_volfracpress = vi * numdofpernode + ivolfrac;

      if (evaluatevolfracpress) myvec[fvi_volfracpress] += vtrans * funct(vi);
    }
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungInstat<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  // loop over all volume fractions
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
  {
    const bool evaluatevolfracpress =
        variablemanager.element_has_valid_vol_frac_pressure(ivolfrac - numfluidphases);

    if (evaluatevolfracpress)
    {
      EvaluatorVolFracBloodLungAddInstatTerms<nsd, nen>::evaluate_matrix_od_struct_and_assemble(
          elemat, funct, deriv, derxy, xjm, curphase, ivolfrac, numdofpernode, phasemanager,
          variablemanager, timefacfac * (-1.0), fac * (-1.0), det);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungInstat<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
}


/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorDivVel<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // no linearization needed in case of initial time derivative calculation
  if (!inittimederiv)
  {
    const int numfluidphases = phasemanager.num_fluid_phases();
    const int numvolfrac = phasemanager.num_vol_frac();
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    const double consfac = timefacfac * variablemanager.div_con_velnp();

    // loop over all volume fractions
    for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
    {
      const bool evaluatevolfracpress =
          variablemanager.element_has_valid_vol_frac_pressure(ivolfrac - numfluidphases);

      for (int vi = 0; vi < nen; ++vi)
      {
        const double v = consfac * funct(vi);
        const int fvi_volfrac = vi * numdofpernode + ivolfrac;
        const int fvi_volfracpress = vi * numdofpernode + numvolfrac + ivolfrac;

        for (int ui = 0; ui < nen; ++ui)
        {
          const int fui = ui * numdofpernode + ivolfrac;
          mymat(fvi_volfrac, fui) += v * funct(ui);
          if (evaluatevolfracpress) mymat(fvi_volfracpress, fui) += v * funct(ui);
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorDivVel<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  double vrhs = rhsfac * variablemanager.div_con_velnp();

  // loop over all volume fractions
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
  {
    const double v = vrhs * phasemanager.vol_frac(ivolfrac - numfluidphases);
    const bool evaluatevolfracpress =
        variablemanager.element_has_valid_vol_frac_pressure(ivolfrac - numfluidphases);


    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi_volfrac = vi * numdofpernode + ivolfrac;
      const int fvi_volfracpress = vi * numdofpernode + numvolfrac + ivolfrac;

      myvec[fvi_volfrac] -= v * funct(vi);
      if (evaluatevolfracpress) myvec[fvi_volfracpress] -= v * funct(vi);
    }
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorDivVel<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  // loop over all volume fractions
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
  {
    const bool evaluatevolfracpress =
        variablemanager.element_has_valid_vol_frac_pressure(ivolfrac - numfluidphases);

    const double vrhs = fac * phasemanager.vol_frac(ivolfrac - numfluidphases);
    // d (div v_s)/d d_n+1 = derxy * 1.0/theta/dt * d_n+1
    // prefactor is fac since timefacfac/theta/dt = fac
    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi_volfrac = vi * numdofpernode + ivolfrac;
      const int fvi_volfracpress = vi * numdofpernode + numvolfrac + ivolfrac;
      const double v = vrhs * funct(vi);

      for (int ui = 0; ui < nen; ++ui)
      {
        for (int idim = 0; idim < nsd; ++idim)
        {
          const int fui = ui * nsd + idim;

          mymat(fvi_volfrac, fui) += v * derxy(idim, ui);
          if (evaluatevolfracpress) mymat(fvi_volfracpress, fui) += v * derxy(idim, ui);
        }
      }
    }

    // shapederivatives see fluid_ele_calc_poro.cpp
    Core::LinAlg::Matrix<nsd, nsd> gridvelderiv(Core::LinAlg::Initialization::zero);
    gridvelderiv.multiply_nt(*(variablemanager.e_con_velnp()), deriv);

    if (nsd == 3)
    {
      const double gridvelderiv_0_0 = gridvelderiv(0, 0);
      const double gridvelderiv_0_1 = gridvelderiv(0, 1);
      const double gridvelderiv_0_2 = gridvelderiv(0, 2);
      const double gridvelderiv_1_0 = gridvelderiv(1, 0);
      const double gridvelderiv_1_1 = gridvelderiv(1, 1);
      const double gridvelderiv_1_2 = gridvelderiv(1, 2);
      const double gridvelderiv_2_0 = gridvelderiv(2, 0);
      const double gridvelderiv_2_1 = gridvelderiv(2, 1);
      const double gridvelderiv_2_2 = gridvelderiv(2, 2);

      const double xjm_0_0 = xjm(0, 0);
      const double xjm_0_1 = xjm(0, 1);
      const double xjm_0_2 = xjm(0, 2);
      const double xjm_1_0 = xjm(1, 0);
      const double xjm_1_1 = xjm(1, 1);
      const double xjm_1_2 = xjm(1, 2);
      const double xjm_2_0 = xjm(2, 0);
      const double xjm_2_1 = xjm(2, 1);
      const double xjm_2_2 = xjm(2, 2);

      for (int ui = 0; ui < nen; ++ui)
      {
        const double v0 =
            gridvelderiv_1_0 * derxjm_(0, 0, 1, ui) + gridvelderiv_1_1 * derxjm_(0, 1, 1, ui) +
            gridvelderiv_1_2 * derxjm_(0, 2, 1, ui) + gridvelderiv_2_0 * derxjm_(0, 0, 2, ui) +
            gridvelderiv_2_1 * derxjm_(0, 1, 2, ui) + gridvelderiv_2_2 * derxjm_(0, 2, 2, ui);

        const double v1 =
            gridvelderiv_0_0 * derxjm_(1, 0, 0, ui) + gridvelderiv_0_1 * derxjm_(1, 1, 0, ui) +
            gridvelderiv_0_2 * derxjm_(1, 2, 0, ui) + gridvelderiv_2_0 * derxjm_(1, 0, 2, ui) +
            gridvelderiv_2_1 * derxjm_(1, 1, 2, ui) + gridvelderiv_2_2 * derxjm_(1, 2, 2, ui);

        const double v2 =
            gridvelderiv_0_0 * derxjm_(2, 0, 0, ui) + gridvelderiv_0_1 * derxjm_(2, 1, 0, ui) +
            gridvelderiv_0_2 * derxjm_(2, 2, 0, ui) + gridvelderiv_1_0 * derxjm_(2, 0, 1, ui) +
            gridvelderiv_1_1 * derxjm_(2, 1, 1, ui) + gridvelderiv_1_2 * derxjm_(2, 2, 1, ui);

        for (int vi = 0; vi < nen; ++vi)
        {
          const int fvi_volfrac = vi * numdofpernode + ivolfrac;
          const int fvi_volfracpress = vi * numdofpernode + numvolfrac + ivolfrac;

          const double v =
              timefacfac / det * funct(vi) * phasemanager.vol_frac(ivolfrac - numfluidphases);

          mymat(fvi_volfrac, ui * 3 + 0) += v * v0;
          if (evaluatevolfracpress) mymat(fvi_volfracpress, ui * 3 + 0) += v * v0;

          mymat(fvi_volfrac, ui * 3 + 1) += v * v1;
          if (evaluatevolfracpress) mymat(fvi_volfracpress, ui * 3 + 1) += v * v1;

          mymat(fvi_volfrac, ui * 3 + 2) += v * v2;
          if (evaluatevolfracpress) mymat(fvi_volfracpress, ui * 3 + 2) += v * v2;
        }
      }
    }
    else if (nsd == 2)
    {
      const double gridvelderiv_0_0 = gridvelderiv(0, 0);
      const double gridvelderiv_0_1 = gridvelderiv(0, 1);
      const double gridvelderiv_1_0 = gridvelderiv(1, 0);
      const double gridvelderiv_1_1 = gridvelderiv(1, 1);

      for (int vi = 0; vi < nen; ++vi)
      {
        const int fvi_volfrac = vi * numdofpernode + ivolfrac;
        const int fvi_volfracpress = vi * numdofpernode + numvolfrac + ivolfrac;

        const double v =
            timefacfac / det * funct(vi) * phasemanager.vol_frac(ivolfrac - numfluidphases);

        for (int ui = 0; ui < nen; ++ui)
        {
          mymat(fvi_volfrac, ui * 2) +=
              v * (-gridvelderiv_1_0 * deriv(1, ui) + gridvelderiv_1_1 * deriv(0, ui));
          if (evaluatevolfracpress)
            mymat(fvi_volfracpress, ui * 2) +=
                v * (-gridvelderiv_1_0 * deriv(1, ui) + gridvelderiv_1_1 * deriv(0, ui));

          mymat(fvi_volfrac, ui * 2 + 1) +=
              v * (+gridvelderiv_0_0 * deriv(1, ui) - gridvelderiv_0_1 * deriv(0, ui));
          if (evaluatevolfracpress)
            mymat(fvi_volfracpress, ui * 2 + 1) +=
                v * (+gridvelderiv_0_0 * deriv(1, ui) - gridvelderiv_0_1 * deriv(0, ui));
        }
      }
    }
    else
      FOUR_C_THROW("shapederivatives not implemented for 1D!");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorDivVel<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungDivVel<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  const int numfluidphases = phasemanager.num_fluid_phases();


  // currently only one volfrac blood lung material possible
  int ivolfrac = numfluidphases;
  const bool evaluatevolfracpress =
      variablemanager.element_has_valid_vol_frac_pressure(ivolfrac - numfluidphases);

  if (evaluatevolfracpress)
  {
    EvaluatorVolFracBloodLungAddDivVelTerm<nsd, nen>::evaluate_matrix_and_assemble(elemat, funct,
        derxy, ivolfrac, ivolfrac, numdofpernode, phasemanager, variablemanager, timefacfac, fac,
        inittimederiv);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungDivVel<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  const int numfluidphases = phasemanager.num_fluid_phases();

  double vrhs = rhsfac * variablemanager.div_con_velnp();

  // currently only one volfrac blood lung material possible
  const int ivolfrac = numfluidphases;
  const double v = vrhs * phasemanager.vol_frac(ivolfrac - numfluidphases);
  const bool evaluatevolfracpress =
      variablemanager.element_has_valid_vol_frac_pressure(ivolfrac - numfluidphases);


  for (int vi = 0; vi < nen; ++vi)
  {
    const int fvi_volfracpress = vi * numdofpernode + ivolfrac;

    if (evaluatevolfracpress) myvec[fvi_volfracpress] -= v * funct(vi);
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungDivVel<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();

  // currently only one volfrac blood lung material possible
  int ivolfrac = numfluidphases;

  const bool evaluatevolfracpress =
      variablemanager.element_has_valid_vol_frac_pressure(ivolfrac - numfluidphases);

  if (evaluatevolfracpress)
  {
    // d(sumvolfrac * divvel)dd = sumvolfrac * ddivveldd + dsumvolfracdd * divvel
    // class Based class for sumvolfrac * ddivveldd

    Core::LinAlg::Matrix<nsd, nsd> gridvelderiv(Core::LinAlg::Initialization::zero);
    gridvelderiv.multiply_nt(*(variablemanager.e_con_velnp()), deriv);

    // OD mesh - div vel term
    EvaluatorBase<nsd, nen>::calc_div_vel_od_mesh(mymat, funct, deriv, derxy, xjm, gridvelderiv,
        timefacfac * phasemanager.vol_frac(ivolfrac - numfluidphases) * (-1.0),
        fac * phasemanager.vol_frac(ivolfrac - numfluidphases) * (-1.0), det, numdofpernode,
        ivolfrac);

    // get matrix to fill for dsumvolfracdd * divvel
    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi = vi * numdofpernode + ivolfrac;
      const double v = -timefacfac * phasemanager.vol_frac(ivolfrac - numfluidphases) *
                       (phasemanager.volfrac_blood_lung_parameter_deformation_dependence()) *
                       variablemanager.div_con_velnp() * funct(vi);

      for (int ui = 0; ui < nen; ++ui)
      {
        for (int idim = 0; idim < nsd; ++idim)
        {
          const int fui = ui * nsd + idim;
          mymat(fvi, fui) += v * derxy(idim, ui);
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungDivVel<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorDiff<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // we do not need the matrix if we calculate the initial time derivative
  if (!inittimederiv)
  {
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    const int numfluidphases = phasemanager.num_fluid_phases();
    const int numvolfrac = phasemanager.num_vol_frac();

    // loop over all volume fractions
    for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
    {
      // get difftensor and diffusive flux
      Core::LinAlg::Matrix<nsd, nsd> difftensor(Core::LinAlg::Initialization::zero);
      phasemanager.diff_tensor_vol_frac(ivolfrac - numfluidphases, difftensor);

      Core::LinAlg::Matrix<nsd, nen> diffflux(Core::LinAlg::Initialization::zero);
      diffflux.multiply(difftensor, derxy);

      // diffusive term
      for (int vi = 0; vi < nen; ++vi)
      {
        const int fvi = vi * numdofpernode + ivolfrac;

        for (int ui = 0; ui < nen; ++ui)
        {
          double laplawf(0.0);
          for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux(j, ui);

          const int fui = ui * numdofpernode + ivolfrac;
          mymat(fvi, fui) += timefacfac * laplawf;
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorDiff<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradphi = *variablemanager.grad_phinp();

  // loop over all volume fractions
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
  {
    // diffusion tensor
    Core::LinAlg::Matrix<nsd, nsd> difftensor(Core::LinAlg::Initialization::zero);
    phasemanager.diff_tensor_vol_frac(ivolfrac - numfluidphases, difftensor);

    Core::LinAlg::Matrix<nsd, 1> diffflux(Core::LinAlg::Initialization::zero);
    diffflux.multiply(difftensor, gradphi[ivolfrac]);

    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi = vi * numdofpernode + ivolfrac;

      // laplacian in weak form
      double laplawf(0.0);
      for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux(j);
      myvec[fvi] -= rhsfac * laplawf;
    }
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorDiff<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradphi = *variablemanager.grad_phinp();

  // loop over all volume fractions
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
  {
    // diffusion tensor
    Core::LinAlg::Matrix<nsd, nsd> difftensor(Core::LinAlg::Initialization::zero);
    phasemanager.diff_tensor_vol_frac(ivolfrac - numfluidphases, difftensor);

    Core::LinAlg::Matrix<nsd, 1> diffflux(Core::LinAlg::Initialization::zero);
    diffflux.multiply(difftensor, gradphi[ivolfrac]);

    // TODO: anisotropic difftensor
    const double v = difftensor(0, 0) * timefacfac / det;

    // gradient of phi w.r.t. reference coordinates
    Core::LinAlg::Matrix<nsd, 1> refgradphi(Core::LinAlg::Initialization::zero);
    refgradphi.multiply(xjm, gradphi[ivolfrac]);

    // OD mesh - diffusive term
    EvaluatorBase<nsd, nen>::calc_diff_od_mesh(mymat, deriv, derxy, xjm, diffflux, refgradphi,
        gradphi[ivolfrac], timefacfac, v, numdofpernode, ivolfrac);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorDiff<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorReac<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // we do not need the matrix if we calculate the initial time derivative
  if (!inittimederiv)
  {
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    const int numfluidphases = phasemanager.num_fluid_phases();
    const int numvolfrac = phasemanager.num_vol_frac();

    // loop over all volume fractions
    for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
    {
      if (phasemanager.is_reactive(ivolfrac))
      {
        double scaledtimefacfac =
            timefacfac / phasemanager.vol_frac_density(ivolfrac - numfluidphases);
        //----------------------------------------------------------------
        // reaction terms
        //----------------------------------------------------------------
        for (int vi = 0; vi < nen; ++vi)
        {
          const double v = scaledtimefacfac * funct(vi);
          const int fvi = vi * numdofpernode + ivolfrac;

          for (int ui = 0; ui < nen; ++ui)
          {
            const double vfunct = v * funct(ui);
            for (int idof = 0; idof < numdofpernode; ++idof)
            {
              const int fui = ui * numdofpernode + idof;

              // rhs ---> -
              mymat(fvi, fui) -= vfunct * phasemanager.reac_deriv(ivolfrac, idof);
            }
          }
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorReac<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  // loop over all volume fractions
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
  {
    if (phasemanager.is_reactive(ivolfrac))
    {
      double scale = 1.0 / phasemanager.vol_frac_density(ivolfrac - numfluidphases);

      double vrhs = scale * rhsfac * phasemanager.reac_term(ivolfrac);

      for (int vi = 0; vi < nen; ++vi)
      {
        const int fvi = vi * numdofpernode + ivolfrac;
        // rhs ---> +
        myvec[fvi] += vrhs * funct(vi);
      }
    }
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorReac<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  // loop over all volume fractions
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
  {
    if (phasemanager.is_reactive(ivolfrac))
    {
      // TODO a constant density is assumed here
      double scale = 1.0 / phasemanager.vol_frac_density(ivolfrac - numfluidphases);

      double vrhs = scale * timefacfac * phasemanager.reac_term(ivolfrac);

      // linearization of porosity (may appear in reaction term)
      //-------------dreac/dd = dreac/dporosity * dporosity/dd = dreac/dporosity * dporosity/dJ *
      // dJ/dd = dreac/dporosity * dporosity/dJ * J * N_x
      // J denotes the determinant of the deformation gradient, i.e. det F = det ( d x / d X ) =
      // det (dx/ds) * ( det(dX/ds) )^-1

      if (phasemanager.porosity_depends_on_struct())
      {
        vrhs += timefacfac * scale * phasemanager.reac_deriv_porosity(ivolfrac) *
                phasemanager.jacobian_def_grad() *
                phasemanager.porosity_deriv_wrt_jacobian_def_grad();
      }

      // linearization of mesh motion (Jacobian)
      // 1) linearization of fac +
      // 2) possible linearization w.r.t porosity
      // rhs ---> -
      EvaluatorBase<nsd, nen>::calc_lin_fac_od_mesh(
          mymat, funct, derxy, -1.0 * vrhs, numdofpernode, ivolfrac);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorReac<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();
  const int numscal = phasemanager.num_scal();

  // loop over all volume fractions
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
  {
    if (phasemanager.is_reactive(ivolfrac))
    {
      double vrhs = 1.0 / phasemanager.vol_frac_density(ivolfrac - numfluidphases) * timefacfac;

      // linearization of reaction term w.r.t scalars
      for (int vi = 0; vi < nen; ++vi)
      {
        const int fvi = vi * numdofpernode + ivolfrac;
        const double v = vrhs * funct(vi);

        for (int ui = 0; ui < nen; ++ui)
        {
          const double vfunct = v * funct(ui);
          for (int iscal = 0; iscal < numscal; ++iscal)
          {
            const int fui = ui * numscal + iscal;
            // rhs ---> -
            mymat(fvi, fui) -= vfunct * phasemanager.reac_deriv_scalar(ivolfrac, iscal);
          }
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorAddFlux<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // we do not need the matrix if we calculate the initial time derivative
  if (!inittimederiv)
  {
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    const int numfluidphases = phasemanager.num_fluid_phases();
    const int numvolfrac = phasemanager.num_vol_frac();

    // loop over all volume fractions
    for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
    {
      if (phasemanager.has_add_scalar_dependent_flux(ivolfrac - numfluidphases))
      {
        // only in case of additional flux we have access to numscal
        const int numscal = phasemanager.num_scal();
        const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradscalarnp =
            *variablemanager.grad_scalarnp();

        // loop over scalars
        for (int iscal = 0; iscal < numscal; iscal++)
        {
          if (phasemanager.has_add_scalar_dependent_flux(ivolfrac - numfluidphases, iscal))
          {
            // diffusion tensor and diffusive flux
            Core::LinAlg::Matrix<nsd, nsd> difftensoraddflux(Core::LinAlg::Initialization::zero);
            for (int i = 0; i < nsd; i++)
              difftensoraddflux(i, i) = phasemanager.scalar_diff(ivolfrac - numfluidphases, iscal);

            Core::LinAlg::Matrix<nsd, 1> diffflux(Core::LinAlg::Initialization::zero);
            diffflux.multiply(difftensoraddflux, gradscalarnp[iscal]);

            for (int vi = 0; vi < nen; ++vi)
            {
              const int fvi = vi * numdofpernode + ivolfrac;
              double laplawf = 0.0;
              for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux(j);
              for (int ui = 0; ui < nen; ++ui)
              {
                const double vfunct = timefacfac * funct(ui) * laplawf;
                // derivative w.r.t. fluid phases
                for (int idof = 0; idof < numfluidphases; ++idof)
                {
                  const int fui = ui * numdofpernode + idof;

                  // chemotaxis
                  if (phasemanager.scalar_to_phase(iscal).species_type ==
                      Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid)
                  {
                    if (phasemanager.scalar_to_phase(iscal).phaseID >
                        phasemanager.num_fluid_phases())
                      FOUR_C_THROW("Wrong PhaseID");
                    // 1) saturation deriv
                    // 2) porosity deriv
                    mymat(fvi, fui) +=
                        vfunct * (phasemanager.saturation_deriv(
                                      phasemanager.scalar_to_phase(iscal).phaseID, idof) *
                                         phasemanager.vol_frac(ivolfrac - numfluidphases) *
                                         phasemanager.porosity() +
                                     phasemanager.porosity_deriv(idof) *
                                         phasemanager.saturation(
                                             phasemanager.scalar_to_phase(iscal).phaseID) *
                                         phasemanager.vol_frac(ivolfrac - numfluidphases));
                  }
                  // haptotaxis
                  else if (phasemanager.scalar_to_phase(iscal).species_type ==
                           Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid)
                    // derivative of solid phase volume fraction w.r.t. all fluid phases = 0
                    break;
                  else
                    FOUR_C_THROW(
                        "AddScalarDependentFlux only possible for species in fluid or solid!");
                }
                // derivative w.r.t. volfrac phases
                for (int jvolfrac = numfluidphases; jvolfrac < numdofpernode; ++jvolfrac)
                {
                  const int fui = ui * numdofpernode + jvolfrac;
                  // haptotaxis
                  if (phasemanager.scalar_to_phase(iscal).species_type ==
                      Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid)
                  {
                    // 1) derivative w.r.t. current volume fraction ivolfrac
                    if (ivolfrac == jvolfrac)
                      mymat(fvi, fui) += vfunct * (1.0 - phasemanager.porosity() -
                                                      phasemanager.sum_add_vol_frac());
                    // 2) derivative of solid phase volume fraction w.r.t. all volume fractions =
                    // 0
                  }
                  // chemotaxis
                  else if (phasemanager.scalar_to_phase(iscal).species_type ==
                           Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid)
                  {
                    // 1) derivative w.r.t. current volume fraction ivolfrac
                    if (ivolfrac == jvolfrac)
                      mymat(fvi, fui) +=
                          vfunct *
                          (phasemanager.saturation(phasemanager.scalar_to_phase(iscal).phaseID) *
                              phasemanager.porosity());
                    // 2) porosity deriv w.r.t. all volume fractions
                    mymat(fvi, fui) +=
                        vfunct *
                        (phasemanager.porosity_deriv(jvolfrac) *
                            phasemanager.saturation(phasemanager.scalar_to_phase(iscal).phaseID) *
                            phasemanager.vol_frac(ivolfrac - numfluidphases));
                  }
                  else
                    FOUR_C_THROW(
                        "AddScalarDependentFlux only possible for species in fluid or solid!");
                }
              }
            }
          }
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorAddFlux<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  // loop over all volume fractions
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
  {
    if (phasemanager.has_add_scalar_dependent_flux(ivolfrac - numfluidphases))
    {
      // only in case of additional flux we have access to numscal
      const int numscal = phasemanager.num_scal();
      const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradscalarnp =
          *variablemanager.grad_scalarnp();

      // loop over scalars
      for (int iscal = 0; iscal < numscal; iscal++)
      {
        if (phasemanager.has_add_scalar_dependent_flux(ivolfrac - numfluidphases, iscal))
        {
          // diffusion tensor
          Core::LinAlg::Matrix<nsd, nsd> difftensoraddflux(Core::LinAlg::Initialization::zero);
          for (int i = 0; i < nsd; i++)
            difftensoraddflux(i, i) = phasemanager.scalar_diff(ivolfrac - numfluidphases, iscal);

          // haptotaxis: scale with volfrac * (1 - porosity - sumaddvolfrac)
          if (phasemanager.scalar_to_phase(iscal).species_type ==
              Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid)
          {
            difftensoraddflux.scale(
                phasemanager.vol_frac(ivolfrac - numfluidphases) *
                (1.0 - phasemanager.porosity() - phasemanager.sum_add_vol_frac()));
          }
          // chemotaxis: scale with volfrac * S * porosity
          else if (phasemanager.scalar_to_phase(iscal).species_type ==
                   Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid)
          {
            difftensoraddflux.scale(
                phasemanager.vol_frac(ivolfrac - numfluidphases) * phasemanager.porosity() *
                phasemanager.saturation(phasemanager.scalar_to_phase(iscal).phaseID));
          }
          else
            FOUR_C_THROW("AddScalarDependentFlux only possible for species in fluid or solid!");

          Core::LinAlg::Matrix<nsd, 1> diffflux(Core::LinAlg::Initialization::zero);
          diffflux.multiply(difftensoraddflux, gradscalarnp[iscal]);
          for (int vi = 0; vi < nen; ++vi)
          {
            const int fvi = vi * numdofpernode + ivolfrac;

            // laplacian in weak form
            double laplawf(0.0);
            for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux(j);
            myvec[fvi] -= rhsfac * laplawf;
          }
        }
      }
    }
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorAddFlux<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  // loop over all volume fractions
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
  {
    if (phasemanager.has_add_scalar_dependent_flux(ivolfrac - numfluidphases))
    {
      // only in case of additional flux we have access to numscal
      const int numscal = phasemanager.num_scal();
      const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradscalarnp =
          *variablemanager.grad_scalarnp();

      // loop over all scalars
      for (int iscal = 0; iscal < numscal; iscal++)
      {
        if (phasemanager.has_add_scalar_dependent_flux(ivolfrac - numfluidphases, iscal))
        {
          // diffusion tensor
          Core::LinAlg::Matrix<nsd, nsd> difftensoraddflux(Core::LinAlg::Initialization::zero);
          for (int i = 0; i < nsd; i++)
            difftensoraddflux(i, i) = phasemanager.scalar_diff(ivolfrac - numfluidphases, iscal);

          Core::LinAlg::Matrix<nsd, 1> diffflux(Core::LinAlg::Initialization::zero);
          // haptotaxis: scale with volfrac * (1 - porosity - sumaddvolfrac)
          if (phasemanager.scalar_to_phase(iscal).species_type ==
              Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid)
          {
            diffflux.multiply(phasemanager.vol_frac(ivolfrac - numfluidphases) *
                                  (1 - phasemanager.porosity() - phasemanager.sum_add_vol_frac()),
                difftensoraddflux, gradscalarnp[iscal]);
          }
          // chemotaxis: scale with volfrac * S * porosity
          else if (phasemanager.scalar_to_phase(iscal).species_type ==
                   Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid)
          {
            diffflux.multiply(
                phasemanager.vol_frac(ivolfrac - numfluidphases) * phasemanager.porosity() *
                    phasemanager.saturation(phasemanager.scalar_to_phase(iscal).phaseID),
                difftensoraddflux, gradscalarnp[iscal]);
          }
          else
            FOUR_C_THROW("AddScalarDependentFlux only possible for species in fluid or solid!");

          double v(0.0);
          // TODO: anisotropic difftensor
          // haptotaxis: scale with volfrac * (1 - porosity - sumaddvolfrac)
          if (phasemanager.scalar_to_phase(iscal).species_type ==
              Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid)
          {
            v = difftensoraddflux(0, 0) * timefacfac / det *
                phasemanager.vol_frac(ivolfrac - numfluidphases) *
                (1 - phasemanager.porosity() - phasemanager.sum_add_vol_frac());
          }
          // chemotaxis: scale with volfrac * S * porosity
          else if (phasemanager.scalar_to_phase(iscal).species_type ==
                   Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid)
          {
            v = difftensoraddflux(0, 0) * timefacfac / det *
                phasemanager.vol_frac(ivolfrac - numfluidphases) * phasemanager.porosity() *
                phasemanager.saturation(phasemanager.scalar_to_phase(iscal).phaseID);
          }
          else
            FOUR_C_THROW("AddScalarDependentFlux only possible for species in fluid or solid!");

          // gradient of phi w.r.t. reference coordinates
          Core::LinAlg::Matrix<nsd, 1> refgradscalarnp(Core::LinAlg::Initialization::zero);
          refgradscalarnp.multiply(xjm, gradscalarnp[iscal]);

          // 1)
          // -----------------------------------------------------------------------------------------------------------------------------------------
          // OD mesh - diffusive term
          EvaluatorBase<nsd, nen>::calc_diff_od_mesh(mymat, deriv, derxy, xjm, diffflux,
              refgradscalarnp, gradscalarnp[iscal], timefacfac, v, numdofpernode, ivolfrac);

          // 2)
          // -----------------------------------------------------------------------------------------------------------------------------------------
          // linearization of porosity
          //-------------dreac/dd = dreac/dporosity * dporosity/dd = dreac/dporosity *
          // dporosity/dJ
          //* dJ/dd = dreac/dporosity * dporosity/dJ * J * N_x
          // J denotes the determinant of the deformation gradient, i.e. det F = det ( d x / d X )
          // = det (dx/ds) * ( det(dX/ds) )^-1

          if (phasemanager.porosity_depends_on_struct())
          {
            Core::LinAlg::Matrix<nsd, 1> diffflux2(Core::LinAlg::Initialization::zero);

            // haptotaxis
            if (phasemanager.scalar_to_phase(iscal).species_type ==
                Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid)
            {
              diffflux2.multiply(phasemanager.vol_frac(ivolfrac - numfluidphases) * (-1.0) *
                                     phasemanager.jacobian_def_grad() *
                                     phasemanager.porosity_deriv_wrt_jacobian_def_grad(),
                  difftensoraddflux, gradscalarnp[iscal]);
            }
            // chemotaxis
            else if (phasemanager.scalar_to_phase(iscal).species_type ==
                     Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid)
            {
              diffflux2.multiply(
                  phasemanager.vol_frac(ivolfrac - numfluidphases) *
                      phasemanager.saturation(phasemanager.scalar_to_phase(iscal).phaseID) *
                      phasemanager.jacobian_def_grad() *
                      phasemanager.porosity_deriv_wrt_jacobian_def_grad(),
                  difftensoraddflux, gradscalarnp[iscal]);
            }
            else
              FOUR_C_THROW("AddScalarDependentFlux only possible for species in fluid or solid!");

            for (int vi = 0; vi < nen; ++vi)
            {
              const int fvi = vi * numdofpernode + ivolfrac;
              double laplawf = 0.0;
              for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux2(j);
              const double v = laplawf * timefacfac;

              for (int ui = 0; ui < nen; ++ui)
              {
                for (int idim = 0; idim < nsd; ++idim)
                {
                  const int fui = ui * nsd + idim;
                  mymat(fvi, fui) += v * derxy(idim, ui);
                }
              }
            }
          }
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorAddFlux<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  // loop over all volume fractions
  for (int ivolfrac = numfluidphases; ivolfrac < numfluidphases + numvolfrac; ivolfrac++)
  {
    if (phasemanager.has_add_scalar_dependent_flux(ivolfrac - numfluidphases))
    {
      const int numscal = phasemanager.num_scal();
      // loop over all scalars
      for (int iscal = 0; iscal < numscal; iscal++)
      {
        if (phasemanager.has_add_scalar_dependent_flux(ivolfrac - numfluidphases, iscal))
        {
          // diffusion tensor
          Core::LinAlg::Matrix<nsd, nsd> difftensoraddflux(Core::LinAlg::Initialization::zero);
          for (int i = 0; i < nsd; i++)
            difftensoraddflux(i, i) = phasemanager.scalar_diff(ivolfrac - numfluidphases, iscal);

          // haptotaxis: scale with volfrac * (1 - porosity - sumaddvolfrac)
          if (phasemanager.scalar_to_phase(iscal).species_type ==
              Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid)
            difftensoraddflux.scale(
                phasemanager.vol_frac(ivolfrac - numfluidphases) *
                (1.0 - phasemanager.porosity() - phasemanager.sum_add_vol_frac()));
          // chemotaxis: scale with volfrac * S * porosity
          else if (phasemanager.scalar_to_phase(iscal).species_type ==
                   Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid)
            difftensoraddflux.scale(
                phasemanager.vol_frac(ivolfrac - numfluidphases) * phasemanager.porosity() *
                phasemanager.saturation(phasemanager.scalar_to_phase(iscal).phaseID));
          else
            FOUR_C_THROW("AddScalarDependentFlux only possible for species in fluid or solid!");

          Core::LinAlg::Matrix<nsd, nen> diffflux(Core::LinAlg::Initialization::zero);
          diffflux.multiply(difftensoraddflux, derxy);

          // diffusive term
          for (int vi = 0; vi < nen; ++vi)
          {
            const int fvi = vi * numdofpernode + ivolfrac;

            for (int ui = 0; ui < nen; ++ui)
            {
              double laplawf(0.0);
              for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux(j, ui);

              const int fui = ui * numscal + iscal;
              mymat(fvi, fui) += timefacfac * laplawf;
            }
          }

          // additional linearization of receptor kinetic law
          if (phasemanager.has_receptor_kinetic_law(ivolfrac - numfluidphases, iscal))
          {
            // get scalars
            const std::vector<double> scalars = *variablemanager.scalarnp();
            const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradscalarnp =
                *variablemanager.grad_scalarnp();

            // correctly scale difftensor
            // d diff / d omega = D_0*(-1.0)*w_half/(w_half+w)^2
            //                    value from above * (-1.0)/(w_half+w)
            difftensoraddflux.scale(
                -1.0 /
                (phasemanager.omega_half(ivolfrac - numfluidphases, iscal) + scalars[iscal]));

            Core::LinAlg::Matrix<nsd, 1> diffflux2(Core::LinAlg::Initialization::zero);
            diffflux2.multiply(difftensoraddflux, gradscalarnp[iscal]);

            for (int vi = 0; vi < nen; ++vi)
            {
              const int fvi = vi * numdofpernode + ivolfrac;

              double laplawf = 0.0;
              for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux2(j);

              for (int ui = 0; ui < nen; ++ui)
              {
                const int fui = ui * numscal + iscal;
                mymat(fvi, fui) += timefacfac * funct(ui) * laplawf;
              }
            }
          }
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorPressureDiff<
    nsd, nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // we do not need the matrix if we calculate the initial time derivative
  if (!inittimederiv)
  {
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    const int numfluidphases = phasemanager.num_fluid_phases();
    const int numvolfrac = phasemanager.num_vol_frac();

    // loop over all volume fraction pressures
    for (int ivolfracpress = numfluidphases + numvolfrac; ivolfracpress < numdofpernode;
        ivolfracpress++)
    {
      const bool evaluatevolfracpress = variablemanager.element_has_valid_vol_frac_pressure(
          ivolfracpress - numvolfrac - numfluidphases);

      if (evaluatevolfracpress)
      {
        // get permeability tensor and diffusive flux
        Core::LinAlg::Matrix<nsd, nsd> permeabilitytensorvolfracpress(
            Core::LinAlg::Initialization::zero);
        phasemanager.permeability_tensor_vol_frac_pressure(
            ivolfracpress - numfluidphases - numvolfrac, permeabilitytensorvolfracpress);
        permeabilitytensorvolfracpress.scale(
            1.0 / phasemanager.dyn_viscosity_vol_frac_pressure(
                      ivolfracpress - numfluidphases - numvolfrac, -1.0));  // TODO: change -1.0

        Core::LinAlg::Matrix<nsd, nen> diffflux(Core::LinAlg::Initialization::zero);
        diffflux.multiply(permeabilitytensorvolfracpress, derxy);

        // diffusive term
        for (int vi = 0; vi < nen; ++vi)
        {
          const int fvi = vi * numdofpernode + ivolfracpress;

          for (int ui = 0; ui < nen; ++ui)
          {
            double laplawf(0.0);
            for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux(j, ui);

            const int fui = ui * numdofpernode + ivolfracpress;
            mymat(fvi, fui) += timefacfac * laplawf;
          }
        }

        if (not phasemanager.has_constant_dyn_viscosity_vol_frac_pressure(
                ivolfracpress - numfluidphases - numvolfrac))
          FOUR_C_THROW(
              "only constant dynamic viscosities possible for volume fraction pressures so far");
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorPressureDiff<
    nsd, nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradphi = *variablemanager.grad_phinp();

  // loop over all volume fractions
  for (int ivolfracpress = numfluidphases + numvolfrac; ivolfracpress < numdofpernode;
      ivolfracpress++)
  {
    double absgradphi = 0.0;
    for (int idim = 0; idim < nsd; idim++)
    {
      absgradphi += gradphi[ivolfracpress](idim, 0) * gradphi[ivolfracpress](idim, 0);
    }
    const bool evaluatevolfracpress = variablemanager.element_has_valid_vol_frac_pressure(
        ivolfracpress - numvolfrac - numfluidphases);

    if (evaluatevolfracpress)
    {
      // get permeability tensor
      Core::LinAlg::Matrix<nsd, nsd> permeabilitytensorvolfracpress(
          Core::LinAlg::Initialization::zero);
      phasemanager.permeability_tensor_vol_frac_pressure(
          ivolfracpress - numfluidphases - numvolfrac, permeabilitytensorvolfracpress);
      permeabilitytensorvolfracpress.scale(
          1.0 / phasemanager.dyn_viscosity_vol_frac_pressure(
                    ivolfracpress - numfluidphases - numvolfrac, -1.0));

      Core::LinAlg::Matrix<nsd, 1> diffflux(Core::LinAlg::Initialization::zero);
      diffflux.multiply(permeabilitytensorvolfracpress, gradphi[ivolfracpress]);

      for (int vi = 0; vi < nen; ++vi)
      {
        const int fvi = vi * numdofpernode + ivolfracpress;

        // laplacian in weak form
        double laplawf(0.0);
        for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux(j);
        myvec[fvi] -= rhsfac * laplawf;
      }
    }
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorPressureDiff<
    nsd, nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                          elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradphi = *variablemanager.grad_phinp();

  // loop over all volume fractions
  for (int ivolfracpress = numfluidphases + numvolfrac; ivolfracpress < numdofpernode;
      ivolfracpress++)
  {
    const bool evaluatevolfracpress = variablemanager.element_has_valid_vol_frac_pressure(
        ivolfracpress - numvolfrac - numfluidphases);

    if (evaluatevolfracpress)
    {
      // get permeability tensor
      Core::LinAlg::Matrix<nsd, nsd> permeabilitytensorvolfracpress(
          Core::LinAlg::Initialization::zero);
      phasemanager.permeability_tensor_vol_frac_pressure(
          ivolfracpress - numfluidphases - numvolfrac, permeabilitytensorvolfracpress);
      permeabilitytensorvolfracpress.scale(
          1.0 / phasemanager.dyn_viscosity_vol_frac_pressure(
                    ivolfracpress - numfluidphases - numvolfrac, -1.0));

      Core::LinAlg::Matrix<nsd, 1> diffflux(Core::LinAlg::Initialization::zero);
      diffflux.multiply(permeabilitytensorvolfracpress, gradphi[ivolfracpress]);

      // TODO: anisotropic difftensor
      const double v = permeabilitytensorvolfracpress(0, 0) * timefacfac / det;

      // gradient of phi w.r.t. reference coordinates
      Core::LinAlg::Matrix<nsd, 1> refgradphi(Core::LinAlg::Initialization::zero);
      refgradphi.multiply(xjm, gradphi[ivolfracpress]);

      // OD mesh - diffusive term
      EvaluatorBase<nsd, nen>::calc_diff_od_mesh(mymat, deriv, derxy, xjm, diffflux, refgradphi,
          gradphi[ivolfracpress], timefacfac, v, numdofpernode, ivolfracpress);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorPressureDiff<
    nsd, nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                          elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // nothing to do
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungPressureDiff<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // we do not need the matrix if we calculate the initial time derivative
  if (!inittimederiv)
  {
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    const int numfluidphases = phasemanager.num_fluid_phases();

    // currently only one volfrac blood lung material possible
    int ivolfracpress = numfluidphases;
    const bool evaluatevolfracpress =
        variablemanager.element_has_valid_vol_frac_pressure(ivolfracpress - numfluidphases);

    if (evaluatevolfracpress)
    {
      // get permeability tensor and diffusive flux
      Core::LinAlg::Matrix<nsd, nsd> permeabilitytensorvolfracpress(
          Core::LinAlg::Initialization::zero);
      phasemanager.permeability_tensor_vol_frac_pressure(
          ivolfracpress - numfluidphases, permeabilitytensorvolfracpress);
      permeabilitytensorvolfracpress.scale(
          1.0 / phasemanager.dyn_viscosity_vol_frac_pressure_blood_lung(
                    ivolfracpress - numfluidphases, -1.0));  // TODO: change -1.0

      Core::LinAlg::Matrix<nsd, nen> diffflux(Core::LinAlg::Initialization::zero);
      diffflux.multiply(permeabilitytensorvolfracpress, derxy);

      // diffusive term
      for (int vi = 0; vi < nen; ++vi)
      {
        const int fvi = vi * numdofpernode + ivolfracpress;

        for (int ui = 0; ui < nen; ++ui)
        {
          double laplawf(0.0);
          for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux(j, ui);

          const int fui = ui * numdofpernode + ivolfracpress;
          mymat(fvi, fui) += timefacfac * laplawf;
        }
      }

      if (not phasemanager.has_constant_dyn_viscosity_vol_frac_pressure(
              ivolfracpress - numfluidphases))
        FOUR_C_THROW(
            "only constant dynamic viscosities possible for volume fraction pressures so far");
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungPressureDiff<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  const int numfluidphases = phasemanager.num_fluid_phases();

  const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradphi = *variablemanager.grad_phinp();

  // currently only one volfrac blood lung material possible
  const int ivolfracpress = numfluidphases;
  double absgradphi = 0.0;
  for (int idim = 0; idim < nsd; idim++)
  {
    absgradphi += gradphi[ivolfracpress](idim, 0) * gradphi[ivolfracpress](idim, 0);
  }
  const bool evaluatevolfracpress =
      variablemanager.element_has_valid_vol_frac_pressure(ivolfracpress - numfluidphases);

  if (evaluatevolfracpress)
  {
    // get permeability tensor
    Core::LinAlg::Matrix<nsd, nsd> permeabilitytensorvolfracpress(
        Core::LinAlg::Initialization::zero);
    phasemanager.permeability_tensor_vol_frac_pressure(
        ivolfracpress - numfluidphases, permeabilitytensorvolfracpress);
    permeabilitytensorvolfracpress.scale(
        1.0 / phasemanager.dyn_viscosity_vol_frac_pressure_blood_lung(
                  ivolfracpress - numfluidphases, -1.0));


    Core::LinAlg::Matrix<nsd, 1> diffflux(Core::LinAlg::Initialization::zero);
    diffflux.multiply(permeabilitytensorvolfracpress, gradphi[ivolfracpress]);

    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi = vi * numdofpernode + ivolfracpress;

      // laplacian in weak form
      double laplawf(0.0);
      for (int j = 0; j < nsd; j++) laplawf += derxy(j, vi) * diffflux(j);
      myvec[fvi] -= rhsfac * laplawf;
    }
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungPressureDiff<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();

  const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradphi = *variablemanager.grad_phinp();

  // currently only one volfrac blood lung material possible
  int ivolfracpress = numfluidphases;

  const bool evaluatevolfracpress =
      variablemanager.element_has_valid_vol_frac_pressure(ivolfracpress - numfluidphases);

  if (evaluatevolfracpress)
  {
    // get permeability tensor
    Core::LinAlg::Matrix<nsd, nsd> permeabilitytensorvolfracpress(
        Core::LinAlg::Initialization::zero);
    phasemanager.permeability_tensor_vol_frac_pressure(
        ivolfracpress - numfluidphases, permeabilitytensorvolfracpress);
    permeabilitytensorvolfracpress.scale(
        1.0 / phasemanager.dyn_viscosity_vol_frac_pressure_blood_lung(
                  ivolfracpress - numfluidphases, -1.0));

    Core::LinAlg::Matrix<nsd, 1> diffflux(Core::LinAlg::Initialization::zero);
    diffflux.multiply(permeabilitytensorvolfracpress, gradphi[ivolfracpress]);

    // TODO: anisotropic difftensor
    const double v = permeabilitytensorvolfracpress(0, 0) * timefacfac / det;

    // gradient of phi w.r.t. reference coordinates
    Core::LinAlg::Matrix<nsd, 1> refgradphi(Core::LinAlg::Initialization::zero);
    refgradphi.multiply(xjm, gradphi[ivolfracpress]);

    // OD mesh - diffusive term
    EvaluatorBase<nsd, nen>::calc_diff_od_mesh(mymat, deriv, derxy, xjm, diffflux, refgradphi,
        gradphi[ivolfracpress], timefacfac, v, numdofpernode, ivolfracpress);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungPressureDiff<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
}



/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorPressureReac<
    nsd, nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // we do not need the matrix if we calculate the initial time derivative
  if (!inittimederiv)
  {
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    const int numfluidphases = phasemanager.num_fluid_phases();
    const int numvolfrac = phasemanager.num_vol_frac();

    // loop over all volume fractions
    for (int ivolfracpress = numfluidphases + numvolfrac; ivolfracpress < numdofpernode;
        ivolfracpress++)
    {
      const bool evaluatevolfracpress = variablemanager.element_has_valid_vol_frac_pressure(
          ivolfracpress - numvolfrac - numfluidphases);


      if (phasemanager.is_reactive(ivolfracpress) && evaluatevolfracpress)
      {
        double scaledtimefacfac =
            timefacfac / phasemanager.vol_frac_density(ivolfracpress - numfluidphases - numvolfrac);
        //----------------------------------------------------------------
        // reaction terms
        //----------------------------------------------------------------
        for (int vi = 0; vi < nen; ++vi)
        {
          const double v = scaledtimefacfac * funct(vi);
          const int fvi = vi * numdofpernode + ivolfracpress;

          for (int ui = 0; ui < nen; ++ui)
          {
            const double vfunct = v * funct(ui);
            for (int idof = 0; idof < numdofpernode; ++idof)
            {
              const int fui = ui * numdofpernode + idof;

              // rhs ---> -
              mymat(fvi, fui) -= vfunct * phasemanager.reac_deriv(ivolfracpress, idof);
            }
          }
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorPressureReac<
    nsd, nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  // loop over all volume fractions
  for (int ivolfracpress = numfluidphases + numvolfrac; ivolfracpress < numdofpernode;
      ivolfracpress++)
  {
    const bool evaluatevolfracpress = variablemanager.element_has_valid_vol_frac_pressure(
        ivolfracpress - numvolfrac - numfluidphases);


    if (phasemanager.is_reactive(ivolfracpress) && evaluatevolfracpress)
    {
      double scale =
          1.0 / phasemanager.vol_frac_density(ivolfracpress - numfluidphases - numvolfrac);

      double vrhs = scale * rhsfac * phasemanager.reac_term(ivolfracpress);

      for (int vi = 0; vi < nen; ++vi)
      {
        const int fvi = vi * numdofpernode + ivolfracpress;
        // rhs ---> +
        myvec[fvi] += vrhs * funct(vi);
      }
    }
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorPressureReac<
    nsd, nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                          elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();

  // loop over all volume fractions
  for (int ivolfracpress = numfluidphases + numvolfrac; ivolfracpress < numdofpernode;
      ivolfracpress++)
  {
    const bool evaluatevolfracpress = variablemanager.element_has_valid_vol_frac_pressure(
        ivolfracpress - numvolfrac - numfluidphases);


    if (phasemanager.is_reactive(ivolfracpress) && evaluatevolfracpress)
    {
      // TODO a constant density is assumed here
      double scale =
          1.0 / phasemanager.vol_frac_density(ivolfracpress - numfluidphases - numvolfrac);

      double vrhs = scale * timefacfac * phasemanager.reac_term(ivolfracpress);

      // linearization of porosity (may appear in reaction term)
      //-------------dreac/dd = dreac/dporosity * dporosity/dd = dreac/dporosity * dporosity/dJ *
      // dJ/dd = dreac/dporosity * dporosity/dJ * J * N_x
      // J denotes the determinant of the deformation gradient, i.e. det F = det ( d x / d X ) =
      // det (dx/ds) * ( det(dX/ds) )^-1

      if (phasemanager.porosity_depends_on_struct())
      {
        vrhs += timefacfac * scale * phasemanager.reac_deriv_porosity(ivolfracpress) *
                phasemanager.jacobian_def_grad() *
                phasemanager.porosity_deriv_wrt_jacobian_def_grad();
      }

      // linearization of mesh motion (Jacobian)
      // 1) linearization of fac +
      // 2) possible linearization w.r.t porosity
      // rhs ---> -
      EvaluatorBase<nsd, nen>::calc_lin_fac_od_mesh(
          mymat, funct, derxy, -1.0 * vrhs, numdofpernode, ivolfracpress);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracHomogenizedVasculatureTumorPressureReac<
    nsd, nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                          elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numvolfrac = phasemanager.num_vol_frac();
  const int numscal = phasemanager.num_scal();

  // loop over all volume fractions
  for (int ivolfracpress = numfluidphases + numvolfrac; ivolfracpress < numdofpernode;
      ivolfracpress++)
  {
    const bool evaluatevolfracpress = variablemanager.element_has_valid_vol_frac_pressure(
        ivolfracpress - numvolfrac - numfluidphases);

    if (phasemanager.is_reactive(ivolfracpress) && evaluatevolfracpress)
    {
      double vrhs = 1.0 /
                    phasemanager.vol_frac_density(ivolfracpress - numfluidphases - numvolfrac) *
                    timefacfac;

      // linearization of reaction term w.r.t scalars
      for (int vi = 0; vi < nen; ++vi)
      {
        const int fvi = vi * numdofpernode + ivolfracpress;
        const double v = vrhs * funct(vi);

        for (int ui = 0; ui < nen; ++ui)
        {
          const double vfunct = v * funct(ui);
          for (int iscal = 0; iscal < numscal; ++iscal)
          {
            const int fui = ui * numscal + iscal;
            // rhs ---> -
            mymat(fvi, fui) -= vfunct * phasemanager.reac_deriv_scalar(ivolfracpress, iscal);
          }
        }
      }
    }
  }
}


/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungPressureReac<nsd,
    nen>::evaluate_matrix_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, bool inittimederiv)
{
  // we do not need the matrix if we calculate the initial time derivative
  if (!inittimederiv)
  {
    // get matrix to fill
    Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

    const int numfluidphases = phasemanager.num_fluid_phases();

    // currently only one volfrac blood lung material possible
    int ivolfracpress = numfluidphases;

    const bool evaluatevolfracpress =
        variablemanager.element_has_valid_vol_frac_pressure(ivolfracpress - numfluidphases);


    if (phasemanager.is_reactive(ivolfracpress) && evaluatevolfracpress)
    {
      double scaledtimefacfac =
          timefacfac / phasemanager.vol_frac_density(ivolfracpress - numfluidphases);
      //----------------------------------------------------------------
      // reaction terms
      //----------------------------------------------------------------
      for (int vi = 0; vi < nen; ++vi)
      {
        const double v = scaledtimefacfac * funct(vi);
        const int fvi = vi * numdofpernode + ivolfracpress;

        for (int ui = 0; ui < nen; ++ui)
        {
          const double vfunct = v * funct(ui);
          for (int idof = 0; idof < numdofpernode; ++idof)
          {
            const int fui = ui * numdofpernode + idof;

            // rhs ---> -
            mymat(fvi, fui) -= vfunct * phasemanager.reac_deriv(ivolfracpress, idof);
          }
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungPressureReac<nsd,
    nen>::evaluate_vector_and_assemble(std::vector<Core::LinAlg::SerialDenseVector*>& elevec,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    const Core::LinAlg::Matrix<nsd, nen>& xyze, int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double rhsfac,
    double fac, bool inittimederiv)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseVector& myvec = *elevec[0];

  const int numfluidphases = phasemanager.num_fluid_phases();

  // currently only one volfrac blood lung material possible
  const int ivolfracpress = numfluidphases;

  const bool evaluatevolfracpress =
      variablemanager.element_has_valid_vol_frac_pressure(ivolfracpress - numfluidphases);


  if (phasemanager.is_reactive(ivolfracpress) && evaluatevolfracpress)
  {
    double scale = 1.0 / phasemanager.vol_frac_density(ivolfracpress - numfluidphases);

    double vrhs = scale * rhsfac * phasemanager.reac_term(ivolfracpress);

    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi = vi * numdofpernode + ivolfracpress;
      // rhs ---> +
      myvec[fvi] += vrhs * funct(vi);
    }
  }
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungPressureReac<nsd,
    nen>::evaluate_matrix_od_struct_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& deriv,
    const Core::LinAlg::Matrix<nsd, nen>& derxy, const Core::LinAlg::Matrix<nsd, nsd>& xjm,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac, double det)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();


  // currently only one volfrac blood lung material possible
  int ivolfracpress = numfluidphases;
  const bool evaluatevolfracpress =
      variablemanager.element_has_valid_vol_frac_pressure(ivolfracpress - numfluidphases);


  if (phasemanager.is_reactive(ivolfracpress) && evaluatevolfracpress)
  {
    // TODO a constant density is assumed here
    double scale = 1.0 / phasemanager.vol_frac_density(ivolfracpress - numfluidphases);

    double vrhs = scale * timefacfac * phasemanager.reac_term(ivolfracpress);

    // linearization of porosity (may appear in reaction term)
    //-------------dreac/dd = dreac/dporosity * dporosity/dd = dreac/dporosity * dporosity/dJ *
    // dJ/dd = dreac/dporosity * dporosity/dJ * J * N_x
    // J denotes the determinant of the deformation gradient, i.e. det F = det ( d x / d X ) =
    // det (dx/ds) * ( det(dX/ds) )^-1

    if (phasemanager.porosity_depends_on_struct())
    {
      vrhs += timefacfac * scale * phasemanager.reac_deriv_porosity(ivolfracpress) *
              phasemanager.jacobian_def_grad() *
              phasemanager.porosity_deriv_wrt_jacobian_def_grad();
    }

    // linearization of mesh motion (Jacobian)
    // 1) linearization of fac +
    // 2) possible linearization w.r.t porosity
    // rhs ---> -
    EvaluatorBase<nsd, nen>::calc_lin_fac_od_mesh(
        mymat, funct, derxy, -1.0 * vrhs, numdofpernode, ivolfracpress);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidEvaluator::EvaluatorVolFracBloodLungPressureReac<nsd,
    nen>::evaluate_matrix_od_scatra_and_assemble(std::vector<Core::LinAlg::SerialDenseMatrix*>&
                                                     elemat,
    const Core::LinAlg::Matrix<nen, 1>& funct, const Core::LinAlg::Matrix<nsd, nen>& derxy,
    int curphase, int phasetoadd, int numdofpernode,
    const PoroFluidManager::PhaseManagerInterface& phasemanager,
    const PoroFluidManager::VariableManagerInterface<nsd, nen>& variablemanager, double timefacfac,
    double fac)
{
  // get matrix to fill
  Core::LinAlg::SerialDenseMatrix& mymat = *elemat[0];

  const int numfluidphases = phasemanager.num_fluid_phases();
  const int numscal = phasemanager.num_scal();

  // currently only one volfrac blood lung material possible
  int ivolfracpress = numfluidphases;
  const bool evaluatevolfracpress =
      variablemanager.element_has_valid_vol_frac_pressure(ivolfracpress - numfluidphases);

  if (phasemanager.is_reactive(ivolfracpress) && evaluatevolfracpress)
  {
    double vrhs = 1.0 / phasemanager.vol_frac_density(ivolfracpress - numfluidphases) * timefacfac;

    // linearization of reaction term w.r.t scalars
    for (int vi = 0; vi < nen; ++vi)
    {
      const int fvi = vi * numdofpernode + ivolfracpress;
      const double v = vrhs * funct(vi);

      for (int ui = 0; ui < nen; ++ui)
      {
        const double vfunct = v * funct(ui);
        for (int iscal = 0; iscal < numscal; ++iscal)
        {
          const int fui = ui * numscal + iscal;
          // rhs ---> -
          mymat(fvi, fui) -= vfunct * phasemanager.reac_deriv_scalar(ivolfracpress, iscal);
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// template classes

// 1D elements
// line 2
template class Discret::Elements::PoroFluidEvaluator::EvaluatorInterface<1, 2>;

// line 3
template class Discret::Elements::PoroFluidEvaluator::EvaluatorInterface<1, 3>;

// 2D elements
// tri3
template class Discret::Elements::PoroFluidEvaluator::EvaluatorInterface<2, 3>;
// quad4
template class Discret::Elements::PoroFluidEvaluator::EvaluatorInterface<2, 4>;

// quad9 and nurbs9
template class Discret::Elements::PoroFluidEvaluator::EvaluatorInterface<2, 9>;

// 3D elements
// hex8
template class Discret::Elements::PoroFluidEvaluator::EvaluatorInterface<3, 8>;

// hex27
template class Discret::Elements::PoroFluidEvaluator::EvaluatorInterface<3, 27>;
// tet4
template class Discret::Elements::PoroFluidEvaluator::EvaluatorInterface<3, 4>;
// tet10
template class Discret::Elements::PoroFluidEvaluator::EvaluatorInterface<3, 10>;
// pyramid5
template class Discret::Elements::PoroFluidEvaluator::EvaluatorInterface<3, 5>;

FOUR_C_NAMESPACE_CLOSE

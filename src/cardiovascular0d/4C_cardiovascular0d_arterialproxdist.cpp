// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cardiovascular0d_arterialproxdist.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_utils_function_of_time.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 |  ctor (public)                                              mhv 10/13|
 *----------------------------------------------------------------------*/
Utils::Cardiovascular0DArterialProxDist::Cardiovascular0DArterialProxDist(
    std::shared_ptr<Core::FE::Discretization> discr, const std::string& conditionname,
    std::vector<int>& curID)
    : Cardiovascular0D(discr, conditionname, curID)
{
  // empty
}



/*-----------------------------------------------------------------------*
 |(private)                                                    mhv 03/14 |
 |Evaluate method for a heart valve arterial Cardiovascular0D accounting for   |
 |proximal and distal arterial branches separately (formulation proposed |
 |by Cristobal Bertoglio),                                               |
 |calling element evaluates of a condition and assembing results         |
 |based on this conditions                                               |
 *----------------------------------------------------------------------*/
void Utils::Cardiovascular0DArterialProxDist::evaluate(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat1,
    std::shared_ptr<Core::LinAlg::SparseOperator> sysmat2,
    std::shared_ptr<Core::LinAlg::SparseOperator> sysmat3,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec1,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec2,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec3,
    const std::shared_ptr<Core::LinAlg::Vector<double>> sysvec4,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec5)
{
  if (!actdisc_->filled()) FOUR_C_THROW("fill_complete() was not called");
  if (!actdisc_->have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  params.set("action", "calc_struct_volconstrstiff");

  // get time-integrator dependent values
  double theta = params.get("scale_theta", 1.0);
  double ts_size = params.get("time_step_size", 1.0);
  const double time = params.get("total time", -1.0);
  const int numdof_per_cond = 4;
  std::vector<bool> havegid(numdof_per_cond, false);

  const bool assmat1 = sysmat1 != nullptr;
  const bool assmat2 = sysmat2 != nullptr;
  const bool assmat3 = sysmat3 != nullptr;
  const bool assvec1 = sysvec1 != nullptr;
  const bool assvec2 = sysvec2 != nullptr;
  const bool assvec3 = sysvec3 != nullptr;
  const bool assvec4 = sysvec4 != nullptr;
  const bool assvec5 = sysvec5 != nullptr;

  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (auto* cond : cardiovascular0dcond_)
  {
    // Get ConditionID of current condition if defined and write value in parameterlist
    int condID = cond->parameters().get<int>("id");
    params.set("id", condID);

    double R_arvalve_max = cardiovascular0dcond_[condID]->parameters().get<double>("R_arvalve_max");
    double R_arvalve_min = cardiovascular0dcond_[condID]->parameters().get<double>("R_arvalve_min");
    double R_atvalve_max = cardiovascular0dcond_[condID]->parameters().get<double>("R_atvalve_max");
    double R_atvalve_min = cardiovascular0dcond_[condID]->parameters().get<double>("R_atvalve_min");
    double k_p = cardiovascular0dcond_[condID]->parameters().get<double>("k_p");

    double L_arp = cardiovascular0dcond_[condID]->parameters().get<double>("L_arp");
    double C_arp = cardiovascular0dcond_[condID]->parameters().get<double>("C_arp");
    double R_arp = cardiovascular0dcond_[condID]->parameters().get<double>("R_arp");
    double C_ard = cardiovascular0dcond_[condID]->parameters().get<double>("C_ard");
    double R_ard = cardiovascular0dcond_[condID]->parameters().get<double>("R_ard");

    double p_ref = cardiovascular0dcond_[condID]->parameters().get<double>("p_ref");

    double p_at_fac = cardiovascular0dcond_[condID]->parameters().get<double>("p_at_fac");

    // find out whether we will use a time curve and get the factor
    const auto curvenum =
        cardiovascular0dcond_[condID]->parameters().get<std::optional<int>>("p_at_crv");
    double curvefac_np = 1.0;

    if (curvenum.has_value() && curvenum.value() > 0 && time >= 0)
    {
      // function_by_id takes a zero based index
      curvefac_np = Global::Problem::instance()
                        ->function_by_id<Core::Utils::FunctionOfTime>(curvenum.value())
                        .evaluate(time);
    }

    // Cardiovascular0D stiffness
    Core::LinAlg::SerialDenseMatrix wkstiff(numdof_per_cond, numdof_per_cond);

    // contributions to total residuals r:
    // r_m = df_m              - f_m
    //     = (df_np - df_n)/dt - theta f_np - (1-theta) f_n
    // here we ONLY evaluate df_np, f_np
    std::vector<double> df_np(numdof_per_cond);
    std::vector<double> f_np(numdof_per_cond);

    // end-point values at t_{n+1}
    double p_v_np = 0.;
    double p_arp_np = 0.;
    double q_arp_np = 0.;
    double p_ard_np = 0.;
    // ventricular volume at t_{n+1}
    double V_v_np = 0.;

    double p_at_np = 0.;


    double Rarvlv_np = 0.;
    double Ratvlv_np = 0.;

    double dRarvlvdpv = 0.;
    double dRatvlvdpv = 0.;
    double dRarvlvdparp = 0.;

    if (assvec1 or assvec2 or assvec4 or assvec5)
    {
      // extract values of dof vector at t_{n+1}
      p_v_np = (*sysvec4).local_values_as_span()[numdof_per_cond * condID + 0];
      p_arp_np = (*sysvec4).local_values_as_span()[numdof_per_cond * condID + 1];
      q_arp_np = (*sysvec4).local_values_as_span()[numdof_per_cond * condID + 2];
      p_ard_np = (*sysvec4).local_values_as_span()[numdof_per_cond * condID + 3];

      // ventricular volume at t_{n+1}
      V_v_np = (*sysvec5).local_values_as_span()[numdof_per_cond * condID];

      // atrial pressure at t_{n+1}
      p_at_np = p_at_fac * curvefac_np;

      // nonlinear aortic and mitral valve resistances - at t_{n+1}
      Rarvlv_np = 0.5 * (R_arvalve_max - R_arvalve_min) * (tanh((p_arp_np - p_v_np) / k_p) + 1.) +
                  R_arvalve_min;
      Ratvlv_np = 0.5 * (R_atvalve_max - R_atvalve_min) * (tanh((p_v_np - p_at_np) / k_p) + 1.) +
                  R_atvalve_min;

      // derivatives of valves w.r.t. values at t_{n+1}
      dRarvlvdpv = (R_arvalve_max - R_arvalve_min) *
                   (1. - tanh((p_arp_np - p_v_np) / k_p) * tanh((p_arp_np - p_v_np) / k_p)) /
                   (-2. * k_p);
      dRatvlvdpv = (R_atvalve_max - R_atvalve_min) *
                   (1. - tanh((p_v_np - p_at_np) / k_p) * tanh((p_v_np - p_at_np) / k_p)) /
                   (2. * k_p);
      dRarvlvdparp = (R_arvalve_max - R_arvalve_min) *
                     (1. - tanh((p_arp_np - p_v_np) / k_p) * tanh((p_arp_np - p_v_np) / k_p)) /
                     (2. * k_p);

      df_np[0] = V_v_np;
      df_np[1] = C_arp * p_arp_np;
      df_np[2] = (L_arp / R_arp) * q_arp_np;
      df_np[3] = C_ard * p_ard_np;

      f_np[0] = (p_v_np - p_at_np) / Ratvlv_np + (p_v_np - p_arp_np) / Rarvlv_np;
      f_np[1] = q_arp_np - (p_v_np - p_arp_np) / Rarvlv_np;
      f_np[2] = q_arp_np + (p_ard_np - p_arp_np) / R_arp;
      f_np[3] = (p_ard_np - p_ref) / R_ard - q_arp_np;
    }

    // global and local ID of this bc in the redundant vectors
    const int offsetID = params.get<int>("OffsetID");
    std::vector<int> gindex(numdof_per_cond);
    gindex[0] = numdof_per_cond * condID + offsetID;
    for (int j = 1; j < numdof_per_cond; j++) gindex[j] = gindex[0] + j;

    // elements might need condition
    params.set<const Core::Conditions::Condition*>("condition", cond);

    // assemble of Cardiovascular0D stiffness matrix, scale with time-integrator dependent value
    if (assmat1)
    {
      wkstiff(0, 0) =
          theta * ((p_v_np - p_at_np) * dRatvlvdpv / (-Ratvlv_np * Ratvlv_np) + 1. / Ratvlv_np +
                      (p_v_np - p_arp_np) * dRarvlvdpv / (-Rarvlv_np * Rarvlv_np) + 1. / Rarvlv_np);
      wkstiff(0, 1) =
          theta * ((p_v_np - p_arp_np) * dRarvlvdparp / (-Rarvlv_np * Rarvlv_np) - 1. / Rarvlv_np);
      wkstiff(0, 2) = 0.;
      wkstiff(0, 3) = 0.;

      wkstiff(1, 0) =
          theta * (-(p_v_np - p_arp_np) * dRarvlvdpv / (-Rarvlv_np * Rarvlv_np) - 1. / Rarvlv_np);
      wkstiff(1, 1) = theta * (C_arp / (theta * ts_size) -
                                  (p_v_np - p_arp_np) * dRarvlvdparp / (-Rarvlv_np * Rarvlv_np) +
                                  1. / Rarvlv_np);
      wkstiff(1, 2) = theta * (1.);
      wkstiff(1, 3) = 0.;

      wkstiff(2, 0) = 0.;
      wkstiff(2, 1) = theta * (-1.);
      wkstiff(2, 2) = theta * (L_arp / (R_arp * theta * ts_size) + 1.);
      wkstiff(2, 3) = theta * (1.);

      wkstiff(3, 0) = 0.;
      wkstiff(3, 1) = 0.;
      wkstiff(3, 2) = theta * (-1.);
      wkstiff(3, 3) = theta * (C_ard / (theta * ts_size) + 1. / R_ard);

      sysmat1->un_complete();

      // assemble into cardiovascular0d system matrix - wkstiff contribution
      for (int j = 0; j < numdof_per_cond; j++)
      {
        for (int k = 0; k < numdof_per_cond; k++)
        {
          havegid[k] = sysmat1->row_map().my_gid(gindex[k]);
          if (havegid[k]) sysmat1->assemble(wkstiff(k, j), gindex[k], gindex[j]);
        }
      }
    }

    // rhs part df_np
    if (assvec1)
    {
      for (int j = 0; j < numdof_per_cond; j++)
      {
        sysvec1->sum_into_global_values(1, &df_np[j], &gindex[j]);
      }
    }
    // rhs part f_np
    if (assvec2)
    {
      for (int j = 0; j < numdof_per_cond; j++)
      {
        sysvec2->sum_into_global_values(1, &f_np[j], &gindex[j]);
      }
    }

    // define element matrices and vectors
    Core::LinAlg::SerialDenseMatrix elematrix1;
    Core::LinAlg::SerialDenseMatrix elematrix2;
    Core::LinAlg::SerialDenseVector elevector1;
    Core::LinAlg::SerialDenseVector elevector2;
    Core::LinAlg::SerialDenseVector elevector3;

    const auto& geom = cond->geometry();
    // if (geom.empty()) FOUR_C_THROW("evaluation of condition with empty geometry");
    // no check for empty geometry here since in parallel computations
    // can exist processors which do not own a portion of the elements belonging
    // to the condition geometry
    for (const auto& [id, ele] : geom)
    {
      // get element location vector and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      ele->location_vector(*actdisc_, lm, lmowner, lmstride);

      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      const int eledim = (int)lm.size();

      elematrix2.shape(eledim, eledim);
      elevector2.size(eledim);
      elevector3.size(numdof_per_cond);

      // call the element specific evaluate method
      int err = ele->evaluate(
          params, *actdisc_, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err) FOUR_C_THROW("error while evaluating elements");


      // assembly
      int eid = ele->id();

      if (assmat2)
      {
        // assemble the offdiagonal stiffness block (1,0 block) arising from dR_cardvasc0d/dd
        // -> this matrix is later on transposed when building the whole block matrix
        std::vector<int> colvec(1);
        colvec[0] = gindex[0];
        elevector2.scale(-1. / ts_size);
        sysmat2->assemble(eid, lmstride, elevector2, lm, lmowner, colvec);
      }

      if (assvec3)
      {
        // assemble the current volume of the enclosed surface of the cardiovascular0d condition
        for (int j = 1; j < numdof_per_cond; j++) elevector3[j] = elevector3[0];

        std::vector<int> cardiovascular0dlm;
        std::vector<int> cardiovascular0downer;
        for (int j = 0; j < numdof_per_cond; j++)
        {
          cardiovascular0dlm.push_back(gindex[j]);
          cardiovascular0downer.push_back(ele->owner());
        }
        Core::LinAlg::assemble(*sysvec3, elevector3, cardiovascular0dlm, cardiovascular0downer);
      }
    }
  }

  if (assmat3)
  {
    // offdiagonal stiffness block (0,1 block)
    evaluate_d_struct_dp(params, *sysmat3);
  }

  return;
}  // end of evaluate_condition



/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void Utils::Cardiovascular0DArterialProxDist::initialize(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec1,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec2)
{
  if (!(actdisc_->filled())) FOUR_C_THROW("fill_complete() was not called");
  if (!actdisc_->have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");
  // get the current time
  // const double time = params.get("total time",-1.0);

  params.set("action", "calc_struct_constrvol");

  const int numdof_per_cond = 4;

  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (auto* cond : cardiovascular0dcond_)
  {
    // Get ConditionID of current condition if defined and write value in parameterlist
    int condID = cond->parameters().get<int>("id");
    params.set("id", condID);

    // global and local ID of this bc in the redundant vectors
    const int offsetID = params.get<int>("OffsetID");
    std::vector<int> gindex(numdof_per_cond);
    gindex[0] = numdof_per_cond * condID + offsetID;
    for (int j = 1; j < numdof_per_cond; j++) gindex[j] = gindex[0] + j;

    double p_v_0 = cardiovascular0dcond_[condID]->parameters().get<double>("p_v_0");
    double p_arp_0 = cardiovascular0dcond_[condID]->parameters().get<double>("p_arp_0");
    double q_arp_0 = cardiovascular0dcond_[condID]->parameters().get<double>("y_arp_0");
    double p_ard_0 = cardiovascular0dcond_[condID]->parameters().get<double>("p_ard_0");

    sysvec2->sum_into_global_values(1, &p_v_0, &gindex[0]);
    sysvec2->sum_into_global_values(1, &p_arp_0, &gindex[1]);
    sysvec2->sum_into_global_values(1, &q_arp_0, &gindex[2]);
    sysvec2->sum_into_global_values(1, &p_ard_0, &gindex[3]);


    params.set<const Core::Conditions::Condition*>("condition", cond);

    // define element matrices and vectors
    Core::LinAlg::SerialDenseMatrix elematrix1;
    Core::LinAlg::SerialDenseMatrix elematrix2;
    Core::LinAlg::SerialDenseVector elevector1;
    Core::LinAlg::SerialDenseVector elevector2;
    Core::LinAlg::SerialDenseVector elevector3;

    const auto& geom = cond->geometry();
    // no check for empty geometry here since in parallel computations
    // can exist processors which do not own a portion of the elements belonging
    // to the condition geometry
    for (const auto& [id, ele] : geom)
    {
      // get element location vector and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      ele->location_vector(*actdisc_, lm, lmowner, lmstride);

      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      elevector3.size(numdof_per_cond);

      // call the element specific evaluate method
      int err = ele->evaluate(
          params, *actdisc_, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err) FOUR_C_THROW("error while evaluating elements");

      // assembly

      for (int j = 1; j < numdof_per_cond; j++) elevector3[j] = elevector3[0];

      std::vector<int> cardiovascular0dlm;
      std::vector<int> cardiovascular0downer;
      for (int j = 0; j < numdof_per_cond; j++)
      {
        cardiovascular0dlm.push_back(gindex[j]);
        cardiovascular0downer.push_back(ele->owner());
      }
      Core::LinAlg::assemble(*sysvec1, elevector3, cardiovascular0dlm, cardiovascular0downer);
    }
  }

  if (Core::Communication::my_mpi_rank(actdisc_->get_comm()) == 0)
  {
    std::cout << "===== Welcome to monolithic coupling of 3D structural dynamics to 0D "
                 "cardiovascular flow models ====="
              << std::endl;
    std::cout << "======== Model: Proximal and distal arterial windkessel including "
                 "(pseudo-)smooth valve law =========\n"
              << std::endl;
  }
  return;
}  // end of Initialize Cardiovascular0D

FOUR_C_NAMESPACE_CLOSE

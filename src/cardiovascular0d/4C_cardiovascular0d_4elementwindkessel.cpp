/*----------------------------------------------------------------------*/
/*! \file

\brief Monolithic coupling of 3D structural dynamics and 0D cardiovascular flow models

\level 2

*----------------------------------------------------------------------*/

#include "4C_cardiovascular0d_4elementwindkessel.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_so3_surface.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 |  ctor (public)                                              mhv 10/13|
 *----------------------------------------------------------------------*/
UTILS::Cardiovascular0D4ElementWindkessel::Cardiovascular0D4ElementWindkessel(
    Teuchos::RCP<Core::FE::Discretization> discr, const std::string& conditionname,
    std::vector<int>& curID)
    : Cardiovascular0D(discr, conditionname, curID)
{
  // empty
}



/*-----------------------------------------------------------------------*
 |(private)                                                    mhv 03/14 |
 |Evaluate method for standard 4-element Cardiovascular0D,                     |
 |calling element evaluates of a condition and assembing results         |
 |based on this conditions                                               |
 *----------------------------------------------------------------------*/
void UTILS::Cardiovascular0D4ElementWindkessel::evaluate(Teuchos::ParameterList& params,
    Teuchos::RCP<Core::LinAlg::SparseMatrix> sysmat1,
    Teuchos::RCP<Core::LinAlg::SparseOperator> sysmat2,
    Teuchos::RCP<Core::LinAlg::SparseOperator> sysmat3, Teuchos::RCP<Epetra_Vector> sysvec1,
    Teuchos::RCP<Epetra_Vector> sysvec2, Teuchos::RCP<Epetra_Vector> sysvec3,
    const Teuchos::RCP<Epetra_Vector> sysvec4, Teuchos::RCP<Epetra_Vector> sysvec5)
{
  if (!actdisc_->Filled()) FOUR_C_THROW("fill_complete() was not called");
  if (!actdisc_->HaveDofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  params.set("action", "calc_struct_volconstrstiff");

  // get time-integrator dependent values
  double theta = params.get("scale_theta", 1.0);
  double ts_size = params.get("time_step_size", 1.0);

  const int numdof_per_cond = 3;

  std::vector<bool> havegid(numdof_per_cond);
  for (int j = 0; j < numdof_per_cond; j++)
  {
    havegid[j] = false;
  }

  const bool assmat1 = sysmat1 != Teuchos::null;
  const bool assmat2 = sysmat2 != Teuchos::null;
  const bool assmat3 = sysmat3 != Teuchos::null;
  const bool assvec1 = sysvec1 != Teuchos::null;
  const bool assvec2 = sysvec2 != Teuchos::null;
  const bool assvec3 = sysvec3 != Teuchos::null;
  const bool assvec4 = sysvec4 != Teuchos::null;
  const bool assvec5 = sysvec5 != Teuchos::null;

  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (auto* cond : cardiovascular0dcond_)
  {
    // Get ConditionID of current condition if defined and write value in parameterlist
    int condID = cond->parameters().get<int>("id");
    params.set("id", condID);

    double C = cardiovascular0dcond_[condID]->parameters().get<double>("C");
    double R_p = cardiovascular0dcond_[condID]->parameters().get<double>("R_p");
    double Z_c = cardiovascular0dcond_[condID]->parameters().get<double>("Z_c");
    double L = cardiovascular0dcond_[condID]->parameters().get<double>("L");
    double p_ref = cardiovascular0dcond_[condID]->parameters().get<double>("p_ref");

    // Cardiovascular0D stiffness
    Core::LinAlg::SerialDenseMatrix wkstiff(numdof_per_cond, numdof_per_cond);

    // contributions to total residuals r:
    // r_m = df_m              - f_m
    //     = (df_np - df_n)/dt - theta f_np - (1-theta) f_n
    // here we ONLY evaluate df_np, f_np
    std::vector<double> df_np(numdof_per_cond);
    std::vector<double> f_np(numdof_per_cond);

    // end-point values at t_{n+1}
    double p_np = 0.;
    double q_np = 0.;
    double s_np = 0.;
    // volume at t_{n+1}
    double V_np = 0.;

    if (assvec1 or assvec2 or assvec4 or assvec5)
    {
      // extract values of dof vector at t_{n+1}
      p_np = (*sysvec4)[numdof_per_cond * condID + 0];
      q_np = (*sysvec4)[numdof_per_cond * condID + 1];
      s_np = (*sysvec4)[numdof_per_cond * condID + 2];

      // volume at t_{n+1}
      V_np = (*sysvec5)[numdof_per_cond * condID];

      df_np[0] = C * p_np + L * C * s_np;
      df_np[1] = V_np;
      df_np[2] = q_np;

      f_np[0] = (p_np - p_ref) / R_p + (1. + Z_c / R_p) * q_np + (C * Z_c + L / R_p) * s_np;
      f_np[1] = -q_np;
      f_np[2] = -s_np;
    }

    // global and local ID of this bc in the redundant vectors
    const int offsetID = params.get<int>("OffsetID");
    std::vector<int> gindex(numdof_per_cond);
    gindex[0] = numdof_per_cond * condID + offsetID;
    for (int j = 1; j < numdof_per_cond; j++) gindex[j] = gindex[0] + j;

    // elements might need condition
    params.set<Teuchos::RCP<Core::Conditions::Condition>>("condition", Teuchos::rcp(cond, false));

    // assemble of Cardiovascular0D stiffness matrix, scale with time-integrator dependent value
    if (assmat1)
    {
      wkstiff(0, 0) = C / ts_size + theta / R_p;
      wkstiff(0, 1) = theta * (1. + Z_c / R_p);
      wkstiff(0, 2) = L * C / ts_size + theta * (C * Z_c + L / R_p);

      wkstiff(1, 0) = 0.;
      wkstiff(1, 1) = -theta;
      wkstiff(1, 2) = 0.;

      wkstiff(2, 0) = 0.;
      wkstiff(2, 1) = 1. / ts_size;
      wkstiff(2, 2) = -theta;


      sysmat1->UnComplete();

      // assemble into cardiovascular0d system matrix - wkstiff contribution
      for (int j = 0; j < numdof_per_cond; j++)
      {
        for (int k = 0; k < numdof_per_cond; k++)
        {
          havegid[k] = sysmat1->RowMap().MyGID(gindex[k]);
          if (havegid[k]) sysmat1->Assemble(wkstiff(k, j), gindex[k], gindex[j]);
        }
      }
    }

    // rhs part df_np
    if (assvec1)
    {
      for (int j = 0; j < numdof_per_cond; j++)
      {
        int err = sysvec1->SumIntoGlobalValues(1, &df_np[j], &gindex[j]);
        if (err) FOUR_C_THROW("SumIntoGlobalValues failed!");
      }
    }
    // rhs part f_np
    if (assvec2)
    {
      for (int j = 0; j < numdof_per_cond; j++)
      {
        int err = sysvec2->SumIntoGlobalValues(1, &f_np[j], &gindex[j]);
        if (err) FOUR_C_THROW("SumIntoGlobalValues failed!");
      }
    }

    // define element matrices and vectors
    Core::LinAlg::SerialDenseMatrix elematrix1;
    Core::LinAlg::SerialDenseMatrix elematrix2;
    Core::LinAlg::SerialDenseVector elevector1;
    Core::LinAlg::SerialDenseVector elevector2;
    Core::LinAlg::SerialDenseVector elevector3;

    std::map<int, Teuchos::RCP<Core::Elements::Element>>& geom = cond->Geometry();
    // if (geom.empty()) FOUR_C_THROW("evaluation of condition with empty geometry");
    // no check for empty geometry here since in parallel computations
    // can exist processors which do not own a portion of the elements belonging
    // to the condition geometry
    std::map<int, Teuchos::RCP<Core::Elements::Element>>::iterator curr;
    for (curr = geom.begin(); curr != geom.end(); ++curr)
    {
      // get element location vector and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      curr->second->LocationVector(*actdisc_, lm, lmowner, lmstride);

      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      const int eledim = (int)lm.size();

      elematrix2.shape(eledim, eledim);
      elevector2.size(eledim);
      elevector3.size(numdof_per_cond);

      // call the element specific evaluate method
      int err = curr->second->evaluate(
          params, *actdisc_, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err) FOUR_C_THROW("error while evaluating elements");


      // assembly
      int eid = curr->second->Id();

      if (assmat2)
      {
        // assemble the offdiagonal stiffness block (1,0 block) arising from dR_cardvasc0d/dd
        // -> this matrix is later on transposed when building the whole block matrix
        std::vector<int> colvec(1);
        colvec[0] = gindex[1];
        elevector2.scale(-1. / ts_size);
        sysmat2->Assemble(eid, lmstride, elevector2, lm, lmowner, colvec);
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
          cardiovascular0downer.push_back(curr->second->Owner());
        }
        Core::LinAlg::Assemble(*sysvec3, elevector3, cardiovascular0dlm, cardiovascular0downer);
      }
    }
  }

  if (assmat3)
  {
    // offdiagonal stiffness block (0,1 block)
    EvaluateDStructDp(params, sysmat3);
  }
}  // end of evaluate_condition



/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void UTILS::Cardiovascular0D4ElementWindkessel::Initialize(Teuchos::ParameterList& params,
    Teuchos::RCP<Epetra_Vector> sysvec1, Teuchos::RCP<Epetra_Vector> sysvec2)
{
  if (!(actdisc_->Filled())) FOUR_C_THROW("fill_complete() was not called");
  if (!actdisc_->HaveDofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");
  // get the current time
  // const double time = params.get("total time",-1.0);

  params.set("action", "calc_struct_constrvol");

  const int numdof_per_cond = 3;

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

    double p_0 = cardiovascular0dcond_[condID]->parameters().get<double>("p_0");
    double q_0 = 0.;
    double s_0 = 0.;

    int err1 = sysvec2->SumIntoGlobalValues(1, &p_0, &gindex[0]);
    int err2 = sysvec2->SumIntoGlobalValues(1, &q_0, &gindex[1]);
    int err3 = sysvec2->SumIntoGlobalValues(1, &s_0, &gindex[2]);
    if (err1 or err2 or err3) FOUR_C_THROW("SumIntoGlobalValues failed!");

    params.set<Teuchos::RCP<Core::Conditions::Condition>>("condition", Teuchos::rcp(cond, false));

    // define element matrices and vectors
    Core::LinAlg::SerialDenseMatrix elematrix1;
    Core::LinAlg::SerialDenseMatrix elematrix2;
    Core::LinAlg::SerialDenseVector elevector1;
    Core::LinAlg::SerialDenseVector elevector2;
    Core::LinAlg::SerialDenseVector elevector3;

    std::map<int, Teuchos::RCP<Core::Elements::Element>>& geom = cond->Geometry();
    // no check for empty geometry here since in parallel computations
    // can exist processors which do not own a portion of the elements belonging
    // to the condition geometry
    std::map<int, Teuchos::RCP<Core::Elements::Element>>::iterator curr;
    for (curr = geom.begin(); curr != geom.end(); ++curr)
    {
      // get element location vector and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      curr->second->LocationVector(*actdisc_, lm, lmowner, lmstride);

      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      elevector3.size(numdof_per_cond);

      // call the element specific evaluate method
      int err = curr->second->evaluate(
          params, *actdisc_, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err) FOUR_C_THROW("error while evaluating elements");

      // assembly
      for (int j = 1; j < numdof_per_cond; j++) elevector3[j] = elevector3[0];

      std::vector<int> cardiovascular0dlm;
      std::vector<int> cardiovascular0downer;
      for (int j = 0; j < numdof_per_cond; j++)
      {
        cardiovascular0dlm.push_back(gindex[j]);
        cardiovascular0downer.push_back(curr->second->Owner());
      }
      Core::LinAlg::Assemble(*sysvec1, elevector3, cardiovascular0dlm, cardiovascular0downer);
    }
  }

  if (actdisc_->Comm().MyPID() == 0)
  {
    std::cout << "===== Welcome to monolithic coupling of 3D structural dynamics to 0D "
                 "cardiovascular flow models ====="
              << std::endl;
    std::cout << "=================================== Model: 4-element windkessel "
                 "=====================================\n"
              << std::endl;
  }
}  // end of Initialize Cardiovascular0D

FOUR_C_NAMESPACE_CLOSE

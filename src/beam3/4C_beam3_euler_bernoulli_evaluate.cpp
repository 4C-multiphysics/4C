// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beam3_euler_bernoulli.hpp"
#include "4C_beam3_spatial_discretization_utils.hpp"
#include "4C_browniandyn_input.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN

using FAD = Sacado::Fad::DFad<double>;

/*-----------------------------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------------------------*/
int Discret::Elements::Beam3eb::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  set_params_interface_ptr(params);

  if (is_params_interface()) set_brownian_dyn_params_interface_ptr();

  // start with "none"
  Core::Elements::ActionType act = Core::Elements::none;

  if (is_params_interface())
  {
    act = params_interface().get_action_type();
  }
  else
  {
    // get the action required
    std::string action = params.get<std::string>("action", "calc_none");
    if (action == "calc_none")
      FOUR_C_THROW("No action supplied");
    else if (action == "calc_struct_linstiff")
      act = Core::Elements::struct_calc_linstiff;
    else if (action == "calc_struct_nlnstiff")
      act = Core::Elements::struct_calc_nlnstiff;
    else if (action == "calc_struct_internalforce")
      act = Core::Elements::struct_calc_internalforce;
    else if (action == "calc_struct_linstiffmass")
      act = Core::Elements::struct_calc_linstiffmass;
    else if (action == "calc_struct_nlnstiffmass")
      act = Core::Elements::struct_calc_nlnstiffmass;
    else if (action == "calc_struct_nlnstifflmass")
      act = Core::Elements::struct_calc_nlnstifflmass;  // with lumped mass matrix
    else if (action == "calc_struct_stress")
      act = Core::Elements::struct_calc_stress;
    else if (action == "calc_struct_eleload")
      act = Core::Elements::struct_calc_eleload;
    else if (action == "calc_struct_fsiload")
      act = Core::Elements::struct_calc_fsiload;
    else if (action == "calc_struct_update_istep")
      act = Core::Elements::struct_calc_update_istep;
    else if (action == "calc_struct_reset_istep")
      act = Core::Elements::struct_calc_reset_istep;
    else if (action == "calc_struct_ptcstiff")
      act = Core::Elements::struct_calc_ptcstiff;
    else if (action == "calc_struct_energy")
      act = Core::Elements::struct_calc_energy;
    else
      FOUR_C_THROW("Unknown type of action '{}' for Beam3eb", action);
  }

  std::string test = params.get<std::string>("action", "calc_none");

  switch (act)
  {
    case Core::Elements::struct_calc_ptcstiff:
    {
      evaluate_ptc<2>(params, elemat1);
    }
    break;

    case Core::Elements::struct_calc_linstiff:
    {
      // only nonlinear case implemented!
      FOUR_C_THROW("linear stiffness matrix called, but not implemented");
    }
    break;

    // nonlinear stiffness and mass matrix are calculated even if only nonlinear stiffness matrix is
    // required
    case Core::Elements::struct_calc_nlnstiffmass:
    case Core::Elements::struct_calc_nlnstifflmass:
    case Core::Elements::struct_calc_nlnstiff:
    case Core::Elements::struct_calc_internalforce:
    {
      // need current global displacement and residual forces and get them from discretization
      // making use of the local-to-global map lm one can extract current displacement and residual
      // values for each degree of freedom

      // get element displacements
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);

      // get residual displacements
      std::shared_ptr<const Core::LinAlg::Vector<double>> res =
          discretization.get_state("residual displacement");
      if (res == nullptr) FOUR_C_THROW("Cannot get state vectors 'residual displacement'");
      std::vector<double> myres = Core::FE::extract_values(*res, lm);

      // Only in the dynamic case the velocities are needed.
      // get element velocities only if example is static in nature
      std::shared_ptr<const Core::LinAlg::Vector<double>> vel;
      std::vector<double> myvel(lm.size(), 0.0);
      const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();

      if (Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(sdyn, "DYNAMICTYPE") !=
          Inpar::Solid::DynamicType::Statics)
      {
        vel = discretization.get_state("velocity");
        if (vel == nullptr) FOUR_C_THROW("Cannot get state vectors 'velocity'");
        myvel = Core::FE::extract_values(*vel, lm);
      }

      if (act == Core::Elements::struct_calc_nlnstiffmass)
      {
        calc_internal_and_inertia_forces_and_stiff(
            params, myvel, mydisp, &elemat1, &elemat2, &elevec1);
      }
      else if (act == Core::Elements::struct_calc_nlnstifflmass)
      {
        calc_internal_and_inertia_forces_and_stiff(
            params, myvel, mydisp, &elemat1, &elemat2, &elevec1);
        lumpmass(&elemat2);
      }
      else if (act == Core::Elements::struct_calc_nlnstiff)
      {
        calc_internal_and_inertia_forces_and_stiff(
            params, myvel, mydisp, &elemat1, nullptr, &elevec1);
      }
      else if (act == Core::Elements::struct_calc_internalforce)
      {
        calc_internal_and_inertia_forces_and_stiff(
            params, myvel, mydisp, nullptr, nullptr, &elevec1);
      }
    }
    break;

    case Core::Elements::struct_calc_brownianforce:
    case Core::Elements::struct_calc_brownianstiff:
    {
      // get element displacements
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);

      // get element velocity
      std::shared_ptr<const Core::LinAlg::Vector<double>> vel =
          discretization.get_state("velocity");
      if (vel == nullptr) FOUR_C_THROW("Cannot get state vectors 'velocity'");
      std::vector<double> myvel = Core::FE::extract_values(*vel, lm);

      if (act == Core::Elements::struct_calc_brownianforce)
        calc_brownian_forces_and_stiff<2, 2, 3>(params, myvel, mydisp, nullptr, &elevec1);
      else if (act == Core::Elements::struct_calc_brownianstiff)
        calc_brownian_forces_and_stiff<2, 2, 3>(params, myvel, mydisp, &elemat1, &elevec1);
      else
        FOUR_C_THROW("You shouldn't be here.");

      break;
    }

    case Core::Elements::struct_calc_stress:
      break;
    case Core::Elements::struct_calc_update_istep:
      for (int i = 0; i < 3; i++)
      {
        t0_(i, 0) = t_(i, 0);
        t0_(i, 1) = t_(i, 1);
      }
      break;
    case Core::Elements::struct_calc_reset_istep:
      // not necessary since no class variables are modified in predicting steps
      break;
    case Core::Elements::struct_calc_energy:
    {
      if (is_params_interface())  // new structural time integration
      {
        params_interface().add_contribution_to_energy_type(eint_, Solid::internal_energy);
      }

      break;
    }

    case Core::Elements::struct_calc_predict:
    case Core::Elements::struct_calc_recover:
    case Core::Elements::struct_gauss_point_data_output:
    case Core::Elements::struct_init_gauss_point_data_output:
    {
      // do nothing in this cases
      break;
    }
    default:
      std::cout << "\ncalled element with action type " << action_type_to_string(act);
      FOUR_C_THROW("This action type is not implemented for Beam3eb");
      break;
  }

  return 0;
}

/*-----------------------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------------------*/
int Discret::Elements::Beam3eb::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  set_params_interface_ptr(params);

  // get element displacements
  std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
      discretization.get_state("displacement new");
  if (disp == nullptr)
    FOUR_C_THROW(
        "Cannot get state vector 'displacement new'. "
        "No linearization provided for load (add 'LOADLIN: true' to input file)");
  std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);

#ifndef INEXTENSIBLE
  const int dofpn = 3 * NODALDOFS;
#else
  const int dofpn = 7;
#endif

  const int nnodes = 2;

  // get element velocities only if it's not a static problem, otherwise a dynamics problem
  // (UNCOMMENT IF NEEDED)
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
  if (Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(sdyn, "DYNAMICTYPE") !=
      Inpar::Solid::DynamicType::Statics)
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel = discretization.get_state("velocity");
    if (vel == nullptr) FOUR_C_THROW("Cannot get state vectors 'velocity'");
    std::vector<double> myvel = Core::FE::extract_values(*vel, lm);
  }
  // find out whether we will use a time curve
  double time = -1.0;
  if (this->is_params_interface())
    time = this->params_interface_ptr()->get_total_time();
  else
    time = params.get("total time", -1.0);

  // find out whether we will use a time curve and get the factor
  const auto tmp_funct = condition.parameters().get<std::vector<std::optional<int>>>("FUNCT");

  // get values and switches from the condition
  // "ONOFF" is related to the first 6 flags of a line Neumann condition in the input file;
  // value 1 for flag i says that condition is active for i-th degree of freedom
  const auto onoff = condition.parameters().get<std::vector<int>>("ONOFF");

  // val is related to the 6 "VAL" fields after the "ONOFF" flags of the Neumann condition
  // in the input file; "VAL" gives the values of the force as a multiple of the prescribed load
  // curve
  const auto val = condition.parameters().get<std::vector<double>>("VAL");

#ifndef BEAM3EBDISCRETELINENEUMANN
// funct is related to the 6 "FUNCT" fields after the val field of the Neumann condition
// in the input file; "FUNCT" gives the number of the function defined in the section FUNCT
#endif

  // find out which node is correct
  const std::vector<int>* nodeids = condition.get_nodes();

  // if a point neumann condition needs to be linearized
  if (condition.type() == Core::Conditions::PointNeumannEB)
  {
    // find out local element number --> this is done since the first element of a neumann point
    // condition is used for this function in this case we do not know whether it is the left or the
    // right node.
    int insert = -1;

    if ((*nodeids)[0] == nodes()[0]->id())
      insert = 0;
    else if ((*nodeids)[0] == nodes()[1]->id())
      insert = 1;

    if (insert == -1) FOUR_C_THROW("\nNode could not be found on nodemap!\n");


    // amplitude of load curve at current time called
    std::vector<double> functfac(6, 1.0);

    for (int i = 0; i < 6; ++i)
    {
      if (onoff[i] == 0) continue;
      // number of the load curve related with a specific line Neumann condition called
      if (tmp_funct[i].has_value() && tmp_funct[i].value() > 0)
        functfac[i] = Global::Problem::instance()
                          ->function_by_id<Core::Utils::FunctionOfTime>(tmp_funct[i].value())
                          .evaluate(time);
    }

    // add forces to Res_external according to (5.56). There is a factor (-1) needed, as fext is
    // multiplied by (-1) in 4C
    for (int i = 0; i < 3; i++)
    {
      elevec1(insert * dofpn + i) += onoff[i] * val[i] * functfac[i];
    }

    // matrix for current tangent, moment at node and crossproduct
    Core::LinAlg::Matrix<3, 1> tangent;
    Core::LinAlg::Matrix<3, 1> crossproduct;
    Core::LinAlg::Matrix<3, 1> moment;
    Core::LinAlg::Matrix<3, 3> spinmatrix;

    // clear all matrices
    tangent.clear();
    crossproduct.clear();
    moment.clear();
    spinmatrix.clear();

    // assemble current tangent and moment at node
    for (int dof = 3; dof < 6; dof++)
    {
      // get current tangent at nodes
      tangent(dof - 3) = Tref_[insert](dof - 3) + mydisp[insert * dofpn + dof];
      moment(dof - 3) = onoff[dof] * val[dof] * functfac[dof];
    }

    double abs_tangent = 0.0;

    // Res will be normalized with the length of the current tangent
    abs_tangent = tangent.norm2();

    // computespin = S ( tangent ) using the spinmatrix in namespace largerotations
    Core::LargeRotations::computespin(spinmatrix, tangent);

    // matrixoperation crossproduct = t x m
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) crossproduct(i) += spinmatrix(i, j) * moment(j);

    // add moments to Res_external according to (5.56). There is a factor (-1) needed, as fext is
    // multiplied by (-1) in 4C
    for (int i = 3; i < 6; i++)
    {
#ifndef SIMPLECALC
      elevec1(insert * dofpn + i) -= crossproduct(i - 3) / std::pow(abs_tangent, 2.0);
#else
      elevec1(insert * dofpn + i) -= crossproduct(i - 3) * ScaleFactorLine;
#endif
    }

    // assembly for stiffnessmatrix
    Core::LinAlg::Matrix<3, 3> crossxtangent;

    crossxtangent.clear();

    // perform matrix operation
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) crossxtangent(i, j) = crossproduct(i) * tangent(j);

    spinmatrix.clear();

    // spinmatrix = S ( m )
    Core::LargeRotations::computespin(spinmatrix, moment);

    // add R_external to stiffness matrix
    // all parts have been evaluated at the boundaries which helps simplifying the matrices
    // In contrast to the Neumann part of the residual force here is NOT a factor of (-1) needed, as
    // elemat1 is directly added to the stiffness matrix without sign change.
    if (elemat1 != nullptr)
    {
      for (int i = 3; i < 6; i++)
      {
        for (int j = 3; j < 6; j++)
        {
#ifndef SIMPLECALC
          (*elemat1)(insert* dofpn + i, insert * dofpn + j) -=
              2.0 * crossxtangent(i - 3, j - 3) / std::pow(abs_tangent, 4.0);
          (*elemat1)(insert* dofpn + i, insert * dofpn + j) -=
              spinmatrix(i - 3, j - 3) / std::pow(abs_tangent, 2.0);
#else
          (*elemat1)(insert* dofpn + i, insert * dofpn + j) -= 2.0 * crossxtangent(i - 3, j - 3);
          (*elemat1)(insert* dofpn + i, insert * dofpn + j) -= spinmatrix(i - 3, j - 3);
#endif
        }
      }
    }
  }
  // if a line neumann condition needs to be linearized
  else if (condition.type() == Core::Conditions::LineNeumann)
  {
    // Check if MOMENT line Neumann conditions are applied accidentally and throw error
    for (int dof = 3; dof < 6; ++dof)
    {
      if (tmp_funct[dof].has_value() and tmp_funct[dof].value() > 0)
        FOUR_C_THROW(
            "Line Neumann conditions for distributed moments are not implemented for beam3eb so "
            "far! Only the function first three function flags (i.e. related to forces) can be "
            "set!");
    }

#ifdef SIMPLECALC
    FOUR_C_THROW("SIMPLECALC not implemented for LineNeumann conditions so far!!!");
#endif

#ifndef BEAM3EBDISCRETELINENEUMANN
    // gaussian points
    Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(mygaussruleeb);
#endif

    Core::LinAlg::Matrix<1, NODALDOFS * nnodes> N_i;

#ifndef BEAM3EBDISCRETELINENEUMANN
    // integration loops
    for (int numgp = 0; numgp < gausspoints.nquad; ++numgp)
    {
      // integration points in parameter space and weights
      const double xi = gausspoints.qxg[numgp][0];
      const double wgt = gausspoints.qwgt[numgp];

      // Get DiscretizationType of beam element
      const Core::FE::CellType distype = shape();

      // Clear matrix for shape functions
      N_i.clear();

// evaluation of shape functions at Gauss points
#if (NODALDOFS == 2)
      // Get hermite derivatives N'xi and N''xi (jacobi_*2.0 is length of the element)
      Core::FE::shape_function_hermite_1d(N_i, xi, jacobi_ * 2.0, distype);
// end--------------------------------------------------------
#elif (NODALDOFS == 3)
      // specific-for----------------------------------Frenet Serret
      // Get hermite derivatives N'xi, N''xi and N'''xi
      Core::FE::shape_function_hermite_1D_order5(N_i, xi, jacobi_ * 2.0, distype);
      // end--------------------------------------------------------
#else
      FOUR_C_THROW("Only the values NODALDOFS = 2 and NODALDOFS = 3 are valid!");
#endif

      // position vector at the gauss point at reference configuration needed for function
      // evaluation
      std::vector<double> X_ref(3, 0.0);
      // calculate coordinates of corresponding Gauss point in reference configuration
      for (int node = 0; node < 2; node++)
      {
#if (NODALDOFS == 2)
        for (int dof = 0; dof < 3; dof++)
        {
          X_ref[dof] +=
              nodes()[node]->x()[dof] * N_i(2 * node) + Tref_[node](dof) * N_i(2 * node + 1);
        }
#elif (NODALDOFS == 3)
        for (int dof = 0; dof < 3; dof++)
        {
          X_ref[dof] += Nodes()[node]->X()[dof] * N_i(3 * node) +
                        Tref_[node](dof) * N_i(3 * node + 1) + Kref_[node](dof) * N_i(3 * node + 2);
        }
#else
        FOUR_C_THROW("Only the values NODALDOFS = 2 and NODALDOFS = 3 are valid!");
#endif
      }

      double fac = 0.0;
      fac = wgt * jacobi_;

      // load vector ar
      double ar[6];

      // loop the dofs of a node
      for (int dof = 0; dof < 6; ++dof) ar[dof] = fac * onoff[dof] * val[dof];

      // sum up load components
      double functionfac = 1.0;
      for (int dof = 0; dof < 3; ++dof)
      {
        if (tmp_funct[dof].has_value() && tmp_funct[dof].value() > 0)
        {
          // evaluate function at the position of the current node       --> dof here correct?
          functionfac =
              Global::Problem::instance()
                  ->function_by_id<Core::Utils::FunctionOfSpaceTime>(tmp_funct[dof].value())
                  .evaluate(X_ref, time, dof);
        }
        else
          functionfac = 1.0;
        for (int node = 0; node < 2 * NODALDOFS; ++node)
        {
#ifndef INEXTENSIBLE
          elevec1[node * 3 + dof] += N_i(node) * ar[dof] * functionfac;
#else
          if (node < 2)
            elevec1[node * 3 + dof] += N_i(node) * ar[dof] * functionfac;
          else
            elevec1[node * 3 + dof + 1] += N_i(node) * ar[dof] * functionfac;
#endif
        }
      }
    }  // for (int numgp=0; numgp<intpoints.nquad; ++numgp)
#else
    // hack in order to realize a discrete point force at position xi=0.0
    {
      // integration points in parameter space and weights
      const double xi = BEAM3EBDISCRETELINENEUMANN;

      // Get DiscretizationType of beam element
      const Core::FE::CellType distype = Shape();

      // Clear matrix for shape functions
      N_i.clear();

      // evaluation of shape functions at Gauss points
#if (NODALDOFS == 2)
      // Get hermite derivatives N'xi and N''xi (jacobi_*2.0 is length of the element)
      Core::FE::shape_function_hermite_1D(N_i, xi, jacobi_ * 2.0, distype);
      // end--------------------------------------------------------
#elif (NODALDOFS == 3)
      // specific-for----------------------------------Frenet Serret
      // Get hermite derivatives N'xi, N''xi and N'''xi
      Core::FE::shape_function_hermite_1D_order5(N_i, xi, jacobi_ * 2.0, distype);
      // end--------------------------------------------------------
#else
      FOUR_C_THROW("Only the values NODALDOFS = 2 and NODALDOFS = 3 are valid!");
#endif

      // load vector ar
      double ar[6];

      // loop the dofs of a node
      for (int dof = 0; dof < 6; ++dof) ar[dof] = onoff[dof] * val[dof] * curvefac[dof];

      for (int dof = 0; dof < 3; ++dof)
      {
        if (ar[dof + 3] != 0)
          FOUR_C_THROW("No discrete moment loads in the elements interior implemented so far!");
      }

      // sum up load components
      for (int dof = 0; dof < 3; ++dof)
      {
        for (int node = 0; node < 2 * NODALDOFS; ++node)
        {
#ifndef INEXTENSIBLE
          elevec1[node * 3 + dof] += N_i(node) * ar[dof];
#else
          if (node < 2)
            elevec1[node * 3 + dof] += N_i(node) * ar[dof];
          else
            elevec1[node * 3 + dof + 1] += N_i(node) * ar[dof];
#endif
        }
      }
    }
#endif
  }

  return 0;
}

/*------------------------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3eb::calc_internal_and_inertia_forces_and_stiff(
    Teuchos::ParameterList& params, std::vector<double>& vel, std::vector<double>& disp,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseMatrix* massmatrix,
    Core::LinAlg::SerialDenseVector* force)
{
  // number of nodes fixed for these element
  const int nnode = 2;

  eint_ = 0.0;
  eint_axial_ = 0.0;
  ekin_ = 0.0;
  l_.clear();
  p_.clear();
  kappa_max_ = 0.0;
  epsilon_max_ = 0.0;

  // material constitutive matrices for forces and moments
  // occurring in material law based on a general hyper-elastic stored energy function
  Core::LinAlg::Matrix<3, 3> C_forceresultant, C_momentresultant;
  get_constitutive_matrices(C_forceresultant, C_momentresultant);

  // in this reduced formulation, we only need two of the constitutive factors
  // axial rigidity
  double EA = C_forceresultant(0, 0);
  // bending/flexural rigidity
  double EI = C_momentresultant(1, 1);

#ifdef SIMPLECALC
  {
    // dimensions of freedom per node
    const int dofpn = 3 * NODALDOFS;

    // number of nodes fixed for these element
    const int nnode = 2;

    // matrix for current positions and tangents
    std::vector<double> disp_totlag(nnode * dofpn, 0.0);

    Core::LinAlg::Matrix<3, 1> r_;
    Core::LinAlg::Matrix<3, 1> r_x;
    Core::LinAlg::Matrix<3, 1> r_xx;

    Core::LinAlg::Matrix<3, 1> f1;
    Core::LinAlg::Matrix<3, 1> f2;
    Core::LinAlg::Matrix<3, 1> n1;

    double rxxrxx;
    double rxrx;
    double tension;

    Core::LinAlg::Matrix<dofpn * nnode, dofpn * nnode> NTildex;
    Core::LinAlg::Matrix<dofpn * nnode, dofpn * nnode> NTildexx;

    Core::LinAlg::Matrix<dofpn * nnode, 1> NxTrx;
    Core::LinAlg::Matrix<dofpn * nnode, 1> NxxTrxx;

    Core::LinAlg::Matrix<dofpn * nnode, dofpn * nnode> M2;
    Core::LinAlg::Matrix<dofpn * nnode, dofpn * nnode> NxTrxrxTNx;

    // Matrices for N_i,xi and N_i,xixi. 2*nnode due to hermite shapefunctions
    Core::LinAlg::Matrix<1, NODALDOFS * nnode> N_i;
    Core::LinAlg::Matrix<1, NODALDOFS * nnode> N_i_x;
    Core::LinAlg::Matrix<1, NODALDOFS * nnode> N_i_xx;

    Core::LinAlg::Matrix<3, nnode * dofpn> N;
    Core::LinAlg::Matrix<3, nnode * dofpn> N_x;
    Core::LinAlg::Matrix<3, nnode * dofpn> N_xx;

    // stiffness due to tension and bending
    Core::LinAlg::Matrix<nnode * dofpn, nnode * dofpn> R_tension;
    Core::LinAlg::Matrix<nnode * dofpn, nnode * dofpn> R_bending;

    // internal force due to tension and bending
    Core::LinAlg::Matrix<nnode * dofpn, 1> Res_tension;
    Core::LinAlg::Matrix<nnode * dofpn, 1> Res_bending;

// some matrices necessary for ANS approach
#ifdef ANS_BEAM3EB
#if (NODALDOFS == 3)
    FOUR_C_THROW(
        "ANS_BEAM3EB approach so far only defined for third order Hermitian shape functions, set "
        "NODALDOFS=2!!!");
#endif
    Core::LinAlg::Matrix<1, 3> L_i;
    L_i.clear();
    Core::LinAlg::Matrix<nnode * dofpn, 1> Res_tension_ANS;
    Res_tension_ANS.clear();
    Core::LinAlg::Matrix<nnode * dofpn, nnode * dofpn> R_tension_ANS;
    R_tension_ANS.clear();
    double epsilon_ANS = 0.0;
    Core::LinAlg::Matrix<1, nnode * dofpn> lin_epsilon_ANS;
    lin_epsilon_ANS.clear();
#endif


    // Get integrationpoints for exact integration
    Core::FE::IntegrationPoints1D gausspoints =
        Core::FE::IntegrationPoints1D(Discret::Utils::mygaussruleeb);

    // Get DiscretizationType of beam element
    const Core::FE::CellType distype = Shape();

    // update displacement vector /d in thesis Meier d = [ r1 t1 r2 t2]
    for (int node = 0; node < nnode; node++)
    {
      for (int dof = 0; dof < dofpn; dof++)
      {
        if (dof < 3)
        {
          // position of nodes
          disp_totlag[node * dofpn + dof] =
              (Nodes()[node]->X()[dof] + disp[node * dofpn + dof]) * ScaleFactorColumn;
        }
        else if (dof < 6)
        {
          // tangent at nodes
          disp_totlag[node * dofpn + dof] =
              (Tref_[node](dof - 3) + disp[node * dofpn + dof]) * ScaleFactorColumn;
        }
        else if (dof >= 6)
        {
#if NODALDOFS == 3
          // curvatures at nodes
          disp_totlag[node * dofpn + dof] =
              (Kref_[node](dof - 6) + disp[node * dofpn + dof]) * ScaleFactorColumn;
#endif
        }
      }
    }  // for (int node = 0 ; node < nnode ; node++)

// Calculate epsilon at collocation points
#ifdef ANS_BEAM3EB
    Core::LinAlg::Matrix<3, 1> epsilon_cp;
    epsilon_cp.clear();
    Core::LinAlg::Matrix<3, 3> tangent_cp;
    tangent_cp.clear();
    Core::LinAlg::Matrix<3, NODALDOFS * 6> lin_epsilon_cp;
    lin_epsilon_cp.clear();

    N_i_x.clear();
    Core::FE::shape_function_hermite_1D_deriv1(N_i_x, 0.0, jacobi_ * 2.0, distype);
    for (int i = 0; i < 2 * NODALDOFS; i++)
    {
      N_i_x(i) = N_i_x(i) / jacobi_;
    }

    for (int i = 0; i < 3; i++)
    {
      tangent_cp(i, 0) = disp_totlag[i + 3];
      tangent_cp(i, 1) = disp_totlag[i + 9];

      for (int j = 0; j < 2 * NODALDOFS; j++)
      {
        tangent_cp(i, 2) += N_i_x(j) * disp_totlag[3 * j + i];
      }
    }
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        epsilon_cp(i) += tangent_cp(j, i) * tangent_cp(j, i);
      }
      epsilon_cp(i) = std::pow(epsilon_cp(i), 0.5) - 1.0;
    }

    for (int k = 0; k < 3; k++)
    {
      N_i_x.clear();

      switch (k)
      {
        case 0:
          Core::FE::shape_function_hermite_1D_deriv1(N_i_x, -1.0, jacobi_ * 2.0, distype);
          break;
        case 1:
          Core::FE::shape_function_hermite_1D_deriv1(N_i_x, 1.0, jacobi_ * 2.0, distype);
          break;
        case 2:
          Core::FE::shape_function_hermite_1D_deriv1(N_i_x, 0.0, jacobi_ * 2.0, distype);
          break;
        default:
          FOUR_C_THROW("Index k should only run from 1 to 3 (three collocation points)!");
          break;
      }

      for (int i = 0; i < 2 * NODALDOFS; i++)
      {
        N_i_x(i) = N_i_x(i) / jacobi_;
      }
      // loop over space dimensions
      for (int i = 0; i < 3; i++)
      {  // loop over all shape functions
        for (int j = 0; j < 2 * NODALDOFS; j++)
        {  // loop over CPs
          lin_epsilon_cp(k, 3 * j + i) += tangent_cp(i, k) * N_i_x(j) / (epsilon_cp(k) + 1);
        }
      }
    }
#endif

    // Loop through all GP and calculate their contribution to the internal forcevector and
    // stiffnessmatrix
    for (int numgp = 0; numgp < gausspoints.nquad; numgp++)
    {
      // all matrices and scalars are set to zero again!!!
      // factors for stiffness assembly

      r_.clear();
      r_x.clear();
      r_xx.clear();

      f1.clear();
      f2.clear();
      n1.clear();

      rxxrxx = 0;
      rxrx = 0;
      tension = 0;

      NTildex.clear();
      NTildexx.clear();

      NxTrx.clear();
      NxxTrxx.clear();

      M2.clear();
      NxTrxrxTNx.clear();

      N_i.clear();
      N_i_x.clear();
      N_i_xx.clear();

      N.clear();
      N_x.clear();
      N_xx.clear();

      R_tension.clear();
      R_bending.clear();

      Res_tension.clear();
      Res_bending.clear();

      // Get location and weight of GP in parameter space
      const double xi = gausspoints.qxg[numgp][0];
      const double wgt = gausspoints.qwgt[numgp];

#if (NODALDOFS == 2)
      // Get hermite derivatives N'xi and N''xi (jacobi_*2.0 is length of the element)
      Core::FE::shape_function_hermite_1D(N_i, xi, jacobi_ * 2.0, distype);
      Core::FE::shape_function_hermite_1D_deriv1(N_i_x, xi, jacobi_ * 2.0, distype);
      Core::FE::shape_function_hermite_1D_deriv2(N_i_xx, xi, jacobi_ * 2.0, distype);
      // end--------------------------------------------------------
#elif (NODALDOFS == 3)
      // specific-for----------------------------------Frenet Serret
      // Get hermite derivatives N'xi, N''xi and N'''xi
      Core::FE::shape_function_hermite_1D_order5(N_i, xi, jacobi_ * 2.0, distype);
      Core::FE::shape_function_hermite_1D_order5_deriv1(N_i_x, xi, jacobi_ * 2.0, distype);
      Core::FE::shape_function_hermite_1D_order5_deriv2(N_i_xx, xi, jacobi_ * 2.0, distype);
      // end--------------------------------------------------------
#else
      FOUR_C_THROW("Only the values NODALDOFS = 2 and NODALDOFS = 3 are valid!");
#endif


      // calculate r' and r''
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < 2 * NODALDOFS; j++)
        {
          r_(i, 0) += N_i(j) * disp_totlag[3 * j + i];
          r_x(i, 0) += N_i_x(j) * disp_totlag[3 * j + i];
          r_xx(i, 0) += N_i_xx(j) * disp_totlag[3 * j + i];
        }
      }

      for (int i = 0; i < 3; i++)
      {
        rxxrxx += r_xx(i) * r_xx(i);
        rxrx += r_x(i) * r_x(i);
      }

      tension = 1 / jacobi_ - 1 / std::pow(rxrx, 0.5);

      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 2 * NODALDOFS; ++j)
        {
          N(i, i + 3 * j) += N_i(j);
          N_x(i, i + 3 * j) += N_i_x(j);
          N_xx(i, i + 3 * j) += N_i_xx(j);
          NxTrx(i + 3 * j) += N_i_x(j) * r_x(i);
          NxxTrxx(i + 3 * j) += N_i_xx(j) * r_xx(i);
        }
      }

      NTildex.multiply_tn(N_x, N_x);
      NTildexx.multiply_tn(N_xx, N_xx);

      for (int i = 0; i < nnode * dofpn; i++)
      {
        for (int j = 0; j < nnode * dofpn; j++)
        {
          M2(i, j) += NxxTrxx(i) * NxTrx(j);
          NxTrxrxTNx(i, j) += NxTrx(i) * NxTrx(j);
        }
      }

#ifdef ANS_BEAM3EB
      Core::FE::shape_function_1d(L_i, xi, Core::FE::CellType::line3);
      epsilon_ANS = 0.0;
      lin_epsilon_ANS.clear();
      for (int i = 0; i < ANSVALUES; i++)
      {
        epsilon_ANS += L_i(i) * epsilon_cp(i);
        for (int j = 0; j < nnode * dofpn; j++)
        {
          lin_epsilon_ANS(j) += L_i(i) * lin_epsilon_cp(i, j);
        }
      }

      Res_tension_ANS.clear();
      R_tension_ANS.clear();

      for (int i = 0; i < nnode * dofpn; i++)
      {
        for (int j = 0; j < nnode * dofpn; j++)
        {
          R_tension_ANS(i, j) += NxTrx(i) * lin_epsilon_ANS(j) / jacobi_;
        }
      }
#endif

      // assemble internal stiffness matrix / R = d/(dd) Res in thesis Meier
      if (stiffmatrix != nullptr)
      {
// assemble parts from tension
#ifndef ANS_BEAM3EB
        R_tension = NTildex;
        R_tension.scale(tension);
        R_tension.update(1.0 / std::pow(rxrx, 1.5), NxTrxrxTNx, 1.0);
        R_tension.scale(EA * wgt);
#else
        // attention: in epsilon_ANS and lin_epsilon_ANS the corresponding jacobi factors are
        // already considered, all the other jacobi factors due to differentiation and integration
        // cancel out!!!
        R_tension_ANS.update(epsilon_ANS / jacobi_, NTildex, 1.0);
        R_tension_ANS.scale(EA * wgt);
#endif

        // assemble parts from bending
        R_bending.update(-rxxrxx / std::pow(jacobi_, 2.0), NTildex, 1.0);
        R_bending.update(1.0, NTildexx, 1.0);
        R_bending.update_t(-2.0 / std::pow(jacobi_, 2.0), M2, 1.0);

        R_bending.scale(EI * wgt / std::pow(jacobi_, 3));

        // shifting values from fixed size matrix to flexible size matrix *stiffmatrix
        for (int i = 0; i < dofpn * nnode; i++)
        {
          for (int j = 0; j < dofpn * nnode; j++)
          {
#ifndef ANS_BEAM3EB
            (*stiffmatrix)(i, j) += R_tension(i, j);
#else
            (*stiffmatrix)(i, j) += R_tension_ANS(i, j);
#endif
            (*stiffmatrix)(i, j) += R_bending(i, j);
          }
        }  // for(int i = 0; i < dofpn*nnode; i++)
      }  // if (stiffmatrix != nullptr)

      for (int i = 0; i < 3; i++)
      {
        f1(i) = -r_x(i) * rxxrxx;
        f2(i) = r_xx(i);
        n1(i) = r_x(i) * tension;
      }

      // assemble internal force vector f_internal / Res in thesis Meier
      if (force != nullptr)
      {
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < 2 * NODALDOFS; j++)
          {
            Res_bending(j * 3 + i) +=
                N_i_x(j) * f1(i) / std::pow(jacobi_, 5) + N_i_xx(j) * f2(i) / std::pow(jacobi_, 3);
#ifndef ANS_BEAM3EB
            Res_tension(j * 3 + i) += N_i_x(j) * n1(i);
#endif
          }
        }
#ifdef ANS_BEAM3EB
        Res_tension_ANS.update(EA * wgt * epsilon_ANS / jacobi_, NxTrx, 1.0);
#endif
        Res_bending.scale(EI * wgt);
        Res_tension.scale(EA * wgt);

        // shifting values from fixed size vector to flexible size vector *force
        for (int i = 0; i < dofpn * nnode; i++)
        {
#ifndef ANS_BEAM3EB
          (*force)(i) += Res_tension(i);
#else
          (*force)(i) += Res_tension_ANS(i);
#endif
          (*force)(i) += Res_bending(i);
        }
      }  // if (force != nullptr)

      // assemble massmatrix if requested
      // calculating mass matrix (local version = global version)
      // note: the mass matrix currently implemented is just a dummy and should not yet be used
      if (massmatrix != nullptr)
      {
        for (int i = 0; i < 6 * nnode; i++) (*massmatrix)(i, i) = 1;

      }  // if (massmatrix != nullptr)
    }  // for(int numgp=0; numgp < gausspoints.nquad; numgp++)
  }
#else
  {
    // dimensions of freedom per node
    const int dofpn = 3 * NODALDOFS;

#ifdef ORTHOPRESSURE
    const double time = params.get("total time", -1.0);
    double orthopressureload = 0.0;
    if (time > 1.0) orthopressureload = ORTHOPRESSURE * (time - 1.0) / 0.1;
    if (time > 1.1) orthopressureload = ORTHOPRESSURE;
#endif

    // matrix for current positions and tangents
    Core::LinAlg::Matrix<nnode * dofpn, 1> disp_totlag(Core::LinAlg::Initialization::zero);

#ifdef BEAM3EBAUTOMATICDIFF
    std::vector<FAD> disp_totlag_fad(nnode * dofpn, 0.0);
#endif

#ifdef INEXTENSIBLE
    std::vector<FAD> lm_fad(3, 0.0);
    Core::LinAlg::Matrix<15, 1, FAD> Res_inextensibility(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<15, 15, FAD> R_inextensibility(Core::LinAlg::Initialization::zero);
#endif

    Core::LinAlg::Matrix<3, 1> r_;
    Core::LinAlg::Matrix<3, 1> r_x;
    Core::LinAlg::Matrix<3, 1> r_xx;

    Core::LinAlg::Matrix<3, 1> f1;
    Core::LinAlg::Matrix<3, 1> f2;
    Core::LinAlg::Matrix<3, 1> n1;

    double rxrxx;
    double rxxrxx;
    double rxrx;
    double tension;

#ifdef BEAM3EBAUTOMATICDIFF
    Core::LinAlg::Matrix<3, 1, FAD> rx_fad;
    Core::LinAlg::Matrix<3, 1, FAD> ortho_normal(Core::LinAlg::Initialization::zero);
    FAD rxrx_fad;
#endif

    Core::LinAlg::Matrix<dofpn * nnode, dofpn * nnode> NTilde;
    Core::LinAlg::Matrix<dofpn * nnode, dofpn * nnode> NTildex;
    Core::LinAlg::Matrix<dofpn * nnode, dofpn * nnode> NTildexx;

    Core::LinAlg::Matrix<dofpn * nnode, 1> NxTrx;
    Core::LinAlg::Matrix<dofpn * nnode, 1> NxTrxx;
    Core::LinAlg::Matrix<dofpn * nnode, 1> NxxTrx;
    Core::LinAlg::Matrix<dofpn * nnode, 1> NxxTrxx;

    Core::LinAlg::Matrix<dofpn * nnode, dofpn * nnode> M1;
    Core::LinAlg::Matrix<dofpn * nnode, dofpn * nnode> M2;
    Core::LinAlg::Matrix<dofpn * nnode, dofpn * nnode> M3;
    Core::LinAlg::Matrix<dofpn * nnode, dofpn * nnode> NxTrxrxTNx;

    // Matrices for N_i,xi and N_i,xixi. 2*nnode due to hermite shapefunctions
    Core::LinAlg::Matrix<1, NODALDOFS * nnode> N_i;
    Core::LinAlg::Matrix<1, NODALDOFS * nnode> N_i_x;
    Core::LinAlg::Matrix<1, NODALDOFS * nnode> N_i_xx;

#ifdef BEAM3EBAUTOMATICDIFF
    Core::LinAlg::Matrix<3, nnode * dofpn, FAD> N;
#endif
    Core::LinAlg::Matrix<3, nnode * dofpn> N_x;
    Core::LinAlg::Matrix<3, nnode * dofpn> N_xx;

    // stiffness due to tension and bending
    Core::LinAlg::Matrix<nnode * dofpn, nnode * dofpn> R_tension;
    Core::LinAlg::Matrix<nnode * dofpn, nnode * dofpn> R_bending;
    Core::LinAlg::Matrix<nnode * dofpn, nnode * dofpn> R_orthopressure;

    // internal force due to tension and bending
    Core::LinAlg::Matrix<nnode * dofpn, 1> Res_tension;
    Core::LinAlg::Matrix<nnode * dofpn, 1> Res_bending;
#ifdef BEAM3EBAUTOMATICDIFF
    Core::LinAlg::Matrix<nnode * dofpn, 1, FAD> Res_orthopressure;
#endif

    // some matrices necessary for ANS approach
#ifdef ANS_BEAM3EB
#if (NODALDOFS == 3)
    FOUR_C_THROW(
        "ANS approach so far only defined for third order Hermitian shape functions, set "
        "NODALDOFS=2!!!");
#endif
    Core::LinAlg::Matrix<1, 3> L_i;
    L_i.clear();
    Core::LinAlg::Matrix<nnode * dofpn, 1> Res_tension_ANS;
    Res_tension_ANS.clear();
    Core::LinAlg::Matrix<nnode * dofpn, nnode * dofpn> R_tension_ANS;
    R_tension_ANS.clear();
    double epsilon_ANS = 0.0;
    Core::LinAlg::Matrix<1, nnode * dofpn> lin_epsilon_ANS(Core::LinAlg::Initialization::zero);

#ifdef BEAM3EBAUTOMATICDIFF
    Core::LinAlg::Matrix<1, nnode * dofpn, FAD> lin_epsilon_ANS_fad(
        Core::LinAlg::Initialization::zero);

    Core::LinAlg::Matrix<nnode * dofpn, 1, FAD> Res_tension_ANS_fad;
    Res_tension_ANS_fad.clear();
    Core::LinAlg::Matrix<nnode * dofpn, nnode * dofpn, FAD> R_tension_ANS_fad;
    R_tension_ANS_fad.clear();
    FAD epsilon_ANS_fad = 0.0;
#endif
#endif

    // Get integrationpoints for exact integration
    Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(mygaussruleeb);

    // Get DiscretizationType of beam element
    const Core::FE::CellType distype = shape();

    // unshift node positions, i.e. manipulate element displacement vector
    // as if there where no periodic boundary conditions
    if (brownian_dyn_params_interface_ptr() != nullptr)
      un_shift_node_position(disp, *brownian_dyn_params_interface().get_periodic_bounding_box());

    update_disp_totlag<nnode, dofpn>(disp, disp_totlag);

#ifndef INEXTENSIBLE
#ifdef BEAM3EBAUTOMATICDIFF
    for (int dof = 0; dof < nnode * dofpn; dof++)
    {
      disp_totlag_fad[dof] = disp_totlag(dof);
      disp_totlag_fad[dof].diff(dof, nnode * dofpn);
    }
#endif
#else
    for (int dof = 0; dof < nnode * dofpn; dof++)
    {
      disp_totlag_fad[dof] = disp_totlag(dof);
      disp_totlag_fad[dof].diff(dof, 15);
    }
    lm_fad[0] = disp[6];
    lm_fad[0].diff(12, 15);
    lm_fad[1] = disp[13];
    lm_fad[1].diff(13, 15);
    lm_fad[2] = disp[14];
    lm_fad[2].diff(14, 15);
#endif

    // Set class variables
    for (int i = 0; i < 3; i++)
    {
      t_(i, 0) = disp_totlag(3 + i);
      t_(i, 1) = disp_totlag(9 + i);
    }

    double tangentnorm1 =
        std::sqrt(disp_totlag(3) * disp_totlag(3) + disp_totlag(4) * disp_totlag(4) +
                  disp_totlag(5) * disp_totlag(5));
    double tangentnorm2 =
        std::sqrt(disp_totlag(9) * disp_totlag(9) + disp_totlag(10) * disp_totlag(10) +
                  disp_totlag(11) * disp_totlag(11));

    if (tangentnorm1 < 1.0e-12 or tangentnorm2 < 1.0e-12)
      FOUR_C_THROW("Tangent of norm zero --> deformation to large!!!");

    // Calculate epsilon at collocation points
#ifdef ANS_BEAM3EB
    Core::LinAlg::Matrix<3, 1> epsilon_cp(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 3> tangent_cp(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, NODALDOFS * 6> lin_epsilon_cp(Core::LinAlg::Initialization::zero);

#ifdef BEAM3EBAUTOMATICDIFF
    Core::LinAlg::Matrix<3, 1, FAD> epsilon_cp_fad(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 3, FAD> tangent_cp_fad(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, NODALDOFS * 6, FAD> lin_epsilon_cp_fad(
        Core::LinAlg::Initialization::zero);
#endif

    N_i_x.clear();
    Core::FE::shape_function_hermite_1d_deriv1(N_i_x, 0.0, jacobi_ * 2.0, distype);

    for (int i = 0; i < 2 * NODALDOFS; i++) N_i_x(i) = N_i_x(i) / jacobi_;

    for (int i = 0; i < 3; i++)
    {
      tangent_cp(i, 0) = disp_totlag(i + 3);
      tangent_cp(i, 1) = disp_totlag(i + 9);

      for (int j = 0; j < 2 * NODALDOFS; j++) tangent_cp(i, 2) += N_i_x(j) * disp_totlag(3 * j + i);

#ifdef BEAM3EBAUTOMATICDIFF
      tangent_cp_fad(i, 0) = disp_totlag_fad[i + 3];
      tangent_cp_fad(i, 1) = disp_totlag_fad[i + 9];
      for (int j = 0; j < 2 * NODALDOFS; j++)
      {
        tangent_cp_fad(i, 2) += N_i_x(j) * disp_totlag_fad[3 * j + i];
      }
#endif
    }
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++) epsilon_cp(i) += tangent_cp(j, i) * tangent_cp(j, i);

      epsilon_cp(i) = std::pow(epsilon_cp(i), 0.5) - 1.0;
    }

#ifdef BEAM3EBAUTOMATICDIFF
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        epsilon_cp_fad(i) += tangent_cp_fad(j, i) * tangent_cp_fad(j, i);
      }
      epsilon_cp_fad(i) = std::pow(epsilon_cp_fad(i), 0.5) - 1.0;
    }
#endif

    for (int k = 0; k < 3; k++)
    {
      N_i_x.clear();

      switch (k)
      {
        case 0:
          Core::FE::shape_function_hermite_1d_deriv1(N_i_x, -1.0, jacobi_ * 2.0, distype);
          break;
        case 1:
          Core::FE::shape_function_hermite_1d_deriv1(N_i_x, 1.0, jacobi_ * 2.0, distype);
          break;
        case 2:
          Core::FE::shape_function_hermite_1d_deriv1(N_i_x, 0.0, jacobi_ * 2.0, distype);
          break;
        default:
          FOUR_C_THROW("Index k should only run from 1 to 3 (three collocation points)!");
          break;
      }

      for (int i = 0; i < 2 * NODALDOFS; i++) N_i_x(i) = N_i_x(i) / jacobi_;

      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2 * NODALDOFS; j++)
          lin_epsilon_cp(k, 3 * j + i) += tangent_cp(i, k) * N_i_x(j) / (epsilon_cp(k) + 1);

#ifdef BEAM3EBAUTOMATICDIFF
      // loop over space dimensions
      for (int i = 0; i < 3; i++)
      {  // loop over all shape functions
        for (int j = 0; j < 2 * NODALDOFS; j++)
        {  // loop over CPs
          lin_epsilon_cp_fad(k, 3 * j + i) +=
              tangent_cp_fad(i, k) * N_i_x(j) / (epsilon_cp_fad(k) + 1);
        }
      }
#endif
    }
#endif

#if defined(INEXTENSIBLE)
    for (int i = 0; i < 2; i++)
    {
      for (int j = 0; j < 6; j++)
      {
        for (int k = 0; k < 3; k++)
        {
          Res_inextensibility(j + 7 * i) +=
              INEXTENSIBLE * EA * lm_fad[k] * lin_epsilon_cp_fad(k, j + 6 * i);
        }
      }
    }
    Res_inextensibility(6) += INEXTENSIBLE * EA * epsilon_cp_fad(0);
    Res_inextensibility(13) += INEXTENSIBLE * EA * epsilon_cp_fad(1);
    Res_inextensibility(14) += INEXTENSIBLE * EA * epsilon_cp_fad(2);

    for (int i = 0; i < 15; i++)
    {
      for (int j = 0; j < 6; j++)
      {
        R_inextensibility(i, j) = Res_inextensibility(i).dx(j);
        R_inextensibility(i, 7 + j) = Res_inextensibility(i).dx(6 + j);
      }
      R_inextensibility(i, 6) = Res_inextensibility(i).dx(12);
      R_inextensibility(i, 13) = Res_inextensibility(i).dx(13);
      R_inextensibility(i, 14) = Res_inextensibility(i).dx(14);
    }
#ifdef SWITCHINEXTENSIBLEON
    if (force != nullptr)
    {
      // shifting values from fixed size vector to flexible size vector *force
      for (int i = 0; i < 15; i++)
      {
        (*force)(i) += Res_inextensibility(i).val();
      }
    }  // if (force != nullptr)

    // assemble internal stiffness matrix / R = d/(dd) Res in thesis Meier
    if (stiffmatrix != nullptr)
    {
      for (int i = 0; i < 15; i++)
      {
        for (int j = 0; j < 15; j++)
        {
          (*stiffmatrix)(i, j) += R_inextensibility(i, j).val();
        }
      }  // for(int i = 0; i < dofpn*nnode; i++)
    }  // if (stiffmatrix != nullptr)
#else
    (*stiffmatrix)(6, 6) += 1.0;
    (*stiffmatrix)(13, 13) += 1.0;
    (*stiffmatrix)(14, 14) += 1.0;
#endif
#endif

    // re-assure correct size of strain and stress resultant class variables
    axial_strain_gp_.resize(gausspoints.nquad);
    curvature_gp_.resize(gausspoints.nquad);

    axial_force_gp_.resize(gausspoints.nquad);
    bending_moment_gp_.resize(gausspoints.nquad);

    // Loop through all GP and calculate their contribution to the internal forcevector and
    // stiffnessmatrix
    for (int numgp = 0; numgp < gausspoints.nquad; numgp++)
    {
      // all matrices and scalars are set to zero again!!!
      // factors for stiffness assembly

      r_.clear();
      r_x.clear();
      r_xx.clear();

      f1.clear();
      f2.clear();
      n1.clear();

      rxrxx = 0;
      rxxrxx = 0;
      rxrx = 0;
      tension = 0;

#ifdef BEAM3EBAUTOMATICDIFF
      rx_fad.clear();
      rxrx_fad = 0.0;
      N.clear();
#endif

      NTilde.clear();
      NTildex.clear();
      NTildexx.clear();

      NxTrx.clear();
      NxTrxx.clear();
      NxxTrx.clear();
      NxxTrxx.clear();

      M1.clear();
      M2.clear();
      M3.clear();
      NxTrxrxTNx.clear();

      N_i.clear();
      N_i_x.clear();
      N_i_xx.clear();

      N_x.clear();
      N_xx.clear();

      R_tension.clear();
      R_bending.clear();

      Res_tension.clear();
      Res_bending.clear();

      // Get location and weight of GP in parameter space
      const double xi = gausspoints.qxg[numgp][0];
      const double wgt = gausspoints.qwgt[numgp];

#if (NODALDOFS == 2)
      // Get hermite derivatives N'xi and N''xi (jacobi_*2.0 is length of the element)
      Core::FE::shape_function_hermite_1d(N_i, xi, jacobi_ * 2.0, distype);
      Core::FE::shape_function_hermite_1d_deriv1(N_i_x, xi, jacobi_ * 2.0, distype);
      Core::FE::shape_function_hermite_1d_deriv2(N_i_xx, xi, jacobi_ * 2.0, distype);
      // end--------------------------------------------------------
#elif (NODALDOFS == 3)
      // specific-for----------------------------------Frenet Serret
      // Get hermite derivatives N'xi, N''xi and N'''xi
      Core::FE::shape_function_hermite_1D_order5_deriv1(N_i_x, xi, jacobi_ * 2.0, distype);
      Core::FE::shape_function_hermite_1D_order5_deriv2(N_i_xx, xi, jacobi_ * 2.0, distype);
      // end--------------------------------------------------------
#else
      FOUR_C_THROW("Only the values NODALDOFS = 2 and NODALDOFS = 3 are valid!");
#endif

      // calculate r' and r''
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < nnode * NODALDOFS; j++)
        {
          r_(i, 0) += N_i(j) * disp_totlag(3 * j + i);
          r_x(i, 0) += N_i_x(j) * disp_totlag(3 * j + i);
          r_xx(i, 0) += N_i_xx(j) * disp_totlag(3 * j + i);
        }
      }

#ifdef BEAM3EBAUTOMATICDIFF
      // calculate r' and r''
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < nnode * NODALDOFS; j++)
        {
          rx_fad(i, 0) += N_i_x(j) * disp_totlag_fad[3 * j + i];
        }
      }
#endif

      for (int i = 0; i < 3; i++)
      {
        rxrxx += r_x(i) * r_xx(i);
        rxxrxx += r_xx(i) * r_xx(i);
        rxrx += r_x(i) * r_x(i);
      }

#ifdef BEAM3EBAUTOMATICDIFF
      for (int i = 0; i < 3; i++)
      {
        rxrx_fad += rx_fad(i) * rx_fad(i);
      }
#endif

      tension = 1 / jacobi_ - 1 / std::pow(rxrx, 0.5);

      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < nnode * NODALDOFS; ++j)
        {
          N_x(i, i + 3 * j) += N_i_x(j);
          N_xx(i, i + 3 * j) += N_i_xx(j);
          NxTrx(i + 3 * j) += N_i_x(j) * r_x(i);
          NxTrxx(i + 3 * j) += N_i_x(j) * r_xx(i);
          NxxTrx(i + 3 * j) += N_i_xx(j) * r_x(i);
          NxxTrxx(i + 3 * j) += N_i_xx(j) * r_xx(i);
        }
      }

#ifdef BEAM3EBAUTOMATICDIFF
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < nnode * NODALDOFS; ++j)
        {
          N(i, i + 3 * j) += N_i(j);
        }
      }
#endif

#ifdef ORTHOPRESSURE
      ortho_normal(0) = rx_fad(1, 0);
      ortho_normal(1) = -rx_fad(0, 0);
      ortho_normal(2) = 0.0;
      if (Core::FADUtils::cast_to_double(Core::FADUtils::vector_norm<3>(ortho_normal)) > 1.0e-12)
        ortho_normal.scale(1.0 / (Core::FADUtils::vector_norm<3>(ortho_normal)));

      Res_orthopressure.clear();
      R_orthopressure.clear();
      Res_orthopressure.multiply_tn(N, ortho_normal);
      Res_orthopressure.scale(orthopressureload * wgt * jacobi_);
      for (int i = 0; i < nnode * dofpn; i++)
      {
        for (int j = 0; j < nnode * dofpn; j++)
        {
          R_orthopressure(i, j) = Res_orthopressure(i).dx(j);
        }
      }
#endif

      NTilde.multiply_tn(N_x, N_xx);
      NTildex.multiply_tn(N_x, N_x);
      NTildexx.multiply_tn(N_xx, N_xx);

      for (int i = 0; i < nnode * dofpn; i++)
      {
        for (int j = 0; j < nnode * dofpn; j++)
        {
          M1(i, j) += NxTrx(i) * (NxxTrx(j) + NxTrxx(j));
          M2(i, j) += NxxTrxx(i) * NxTrx(j);
          M3(i, j) += (NxTrxx(i) + NxxTrx(i)) * (NxTrxx(j) + NxxTrx(j));
          NxTrxrxTNx(i, j) += NxTrx(i) * NxTrx(j);
        }
      }

      // calculate quantities necessary for ANS approach
#ifdef ANS_BEAM3EB
      Core::FE::shape_function_1d(L_i, xi, Core::FE::CellType::line3);
      epsilon_ANS = 0.0;
      lin_epsilon_ANS.clear();
      for (int i = 0; i < ANSVALUES; i++)
      {
        epsilon_ANS += L_i(i) * epsilon_cp(i);
        for (int j = 0; j < nnode * dofpn; j++) lin_epsilon_ANS(j) += L_i(i) * lin_epsilon_cp(i, j);
      }

#ifdef BEAM3EBAUTOMATICDIFF
      epsilon_ANS_fad = 0.0;
      lin_epsilon_ANS_fad.clear();
      for (int i = 0; i < ANSVALUES; i++)
      {
        epsilon_ANS_fad += L_i(i) * epsilon_cp_fad(i);
        for (int j = 0; j < nnode * dofpn; j++)
        {
          lin_epsilon_ANS_fad(j) += L_i(i) * lin_epsilon_cp_fad(i, j);
        }
      }

      Res_tension_ANS_fad.clear();
      R_tension_ANS_fad.clear();

#ifndef CONSISTENTANSBEAM3EB
      for (int i = 0; i < nnode * dofpn; i++)
      {
        for (int k = 0; k < 3; k++)
        {
          Res_tension_ANS_fad(i) +=
              N_x(k, i) * rx_fad(k) / std::pow(rxrx_fad, 0.5) * EA * wgt * epsilon_ANS_fad;
        }
      }
#else
      for (int i = 0; i < nnode * dofpn; i++)
      {
        Res_tension_ANS_fad(i) += lin_epsilon_ANS_fad(i) * jacobi_ * EA * wgt * epsilon_ANS_fad;
      }
#endif
      for (int i = 0; i < nnode * dofpn; i++)
      {
        for (int j = 0; j < nnode * dofpn; j++)
        {
          R_tension_ANS_fad(i, j) = Res_tension_ANS_fad(i).dx(j);
        }
      }
#endif
      Res_tension_ANS.clear();
      R_tension_ANS.clear();
#endif

      // assemble internal stiffness matrix / R = d/(dd) Res in thesis Meier
      if (stiffmatrix != nullptr)
      {
        // assemble parts from tension
#ifndef ANS_BEAM3EB
        R_tension = NTildex;
        R_tension.scale(tension);
        R_tension.update(1.0 / std::pow(rxrx, 1.5), NxTrxrxTNx, 1.0);
        R_tension.scale(EA * wgt);
#else
#ifndef CONSISTENTANSBEAM3EB
        // attention: in epsilon_ANS and lin_epsilon_ANS the corresponding jacobi factors are
        // already considered, all the other jacobi factors due to differentiation and integration
        // cancel out!!!
        for (int i = 0; i < nnode * dofpn; i++)
          for (int j = 0; j < nnode * dofpn; j++)
            R_tension_ANS(i, j) += NxTrx(i) * lin_epsilon_ANS(j) / std::pow(rxrx, 0.5);

        R_tension_ANS.update(-epsilon_ANS / std::pow(rxrx, 1.5), NxTrxrxTNx, 1.0);
        R_tension_ANS.update(epsilon_ANS / std::pow(rxrx, 0.5), NTildex, 1.0);
        R_tension_ANS.scale(EA * wgt);
#else
        // since CONSISTENTANSBEAM3EB can so far only be calculated via FAD, R_tension_ANS has to be
        // replaced by R_tension_ANS_fad
        for (int i = 0; i < nnode * dofpn; i++)
        {
          for (int j = 0; j < nnode * dofpn; j++)
          {
            R_tension_ANS(i, j) = R_tension_ANS_fad(i, j).val();
          }
        }
#endif
#endif

        // assemble parts from bending
        R_bending = NTildex;
        R_bending.scale(2.0 * std::pow(rxrxx, 2.0) / std::pow(rxrx, 3.0));
        R_bending.update(-rxxrxx / std::pow(rxrx, 2.0), NTildex, 1.0);
        R_bending.update(-rxrxx / std::pow(rxrx, 2.0), NTilde, 1.0);
        R_bending.update_t(-rxrxx / std::pow(rxrx, 2.0), NTilde, 1.0);
        R_bending.update(1.0 / rxrx, NTildexx, 1.0);
        R_bending.update(-12.0 * std::pow(rxrxx, 2.0) / std::pow(rxrx, 4.0), NxTrxrxTNx, 1.0);
        R_bending.update(4.0 * rxrxx / std::pow(rxrx, 3.0), M1, 1.0);
        R_bending.update_t(4.0 * rxrxx / std::pow(rxrx, 3.0), M1, 1.0);
        R_bending.update(4.0 * rxxrxx / std::pow(rxrx, 3.0), NxTrxrxTNx, 1.0);
        R_bending.update(-2.0 / std::pow(rxrx, 2.0), M2, 1.0);
        R_bending.update_t(-2.0 / std::pow(rxrx, 2.0), M2, 1.0);
        R_bending.update(-1.0 / std::pow(rxrx, 2.0), M3, 1.0);

        R_bending.scale(EI * wgt / jacobi_);

#ifndef INEXTENSIBLE
        // shifting values from fixed size matrix to flexible size matrix *stiffmatrix
        for (int i = 0; i < dofpn * nnode; i++)
        {
          for (int j = 0; j < dofpn * nnode; j++)
          {
#ifndef ANS_BEAM3EB
            (*stiffmatrix)(i, j) += R_tension(i, j);
#else
            (*stiffmatrix)(i, j) += R_tension_ANS(i, j);
#endif
            (*stiffmatrix)(i, j) += R_bending(i, j);
#ifdef ORTHOPRESSURE
            (*stiffmatrix)(i, j) += R_orthopressure(i, j);
#endif
          }
        }  // for(int i = 0; i < dofpn*nnode; i++)
#else
        // shifting values from fixed size matrix to flexible size matrix *stiffmatrix
        int i1 = 0;
        int j1 = 0;
        for (int i = 0; i < 12; i++)
        {
          if (i < 6)
            i1 = i;
          else
            i1 = i + 1;

          for (int j = 0; j < 12; j++)
          {
            if (j < 6)
              j1 = j;
            else
              j1 = j + 1;

            (*stiffmatrix)(i1, j1) += INEXTENSIBLE * R_tension_ANS(i, j);
            (*stiffmatrix)(i1, j1) += R_bending(i, j);
          }
        }  // for(int i = 0; i < dofpn*nnode; i++)
#endif
      }  // if (stiffmatrix != nullptr)

      for (int i = 0; i < 3; i++)
      {
        f1(i) = 2 * r_x(i) * std::pow(rxrxx, 2.0) / std::pow(rxrx, 3.0) -
                (r_x(i) * rxxrxx + r_xx(i) * rxrxx) / std::pow(rxrx, 2.0);
        f2(i) = r_xx(i) / rxrx - r_x(i) * rxrxx / std::pow(rxrx, 2.0);
        n1(i) = r_x(i) * tension;
      }
      // assemble internal force vector f_internal / Res in thesis Meier
      if (force != nullptr)
      {
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < nnode * NODALDOFS; j++)
          {
            Res_bending(j * 3 + i) += N_i_x(j) * f1(i) + N_i_xx(j) * f2(i);
#ifndef ANS_BEAM3EB
            Res_tension(j * 3 + i) += N_i_x(j) * n1(i);
#endif
          }
        }
#ifdef ANS_BEAM3EB
#ifndef CONSISTENTANSBEAM3EB
        // attention: in epsilon_ANS and lin_epsilon_ANS the corresponding jacobi factors are
        // already considered, all the other jacobi factors due to differentiation and integration
        // cancel out!!!
        Res_tension_ANS.update(EA * wgt * epsilon_ANS / std::pow(rxrx, 0.5), NxTrx, 0.0);
#else
        // since CONSISTENTANSBEAM3EB can so far only be calculated via FAD, Rrd_tension_ANS has to
        // be replaced by Rrd_tension_ANS_fad
        for (int i = 0; i < nnode * dofpn; i++)
        {
          Res_tension_ANS(i) = Res_tension_ANS_fad(i).val();
        }
#endif
#endif

        Res_bending.scale(EI * wgt / jacobi_);
        Res_tension.scale(EA * wgt);

#ifndef INEXTENSIBLE
        // shifting values from fixed size vector to flexible size vector *force
        for (int i = 0; i < dofpn * nnode; i++)
        {
#ifndef ANS_BEAM3EB
          (*force)(i) += Res_tension(i);
#else
          (*force)(i) += Res_tension_ANS(i);
#endif
          (*force)(i) += Res_bending(i);
#ifdef ORTHOPRESSURE
          (*force)(i) += Res_orthopressure(i).val();
#endif
        }
#else
        int i1 = 0;
        // shifting values from fixed size vector to flexible size vector *force
        for (int i = 0; i < dofpn * nnode; i++)
        {
          if (i < 6)
            i1 = i;
          else
            i1 = i + 1;

          (*force)(i1) += INEXTENSIBLE * Res_tension_ANS(i);
          (*force)(i1) += Res_bending(i);
        }
#endif
      }  // if (force != nullptr)

#ifdef ANS_BEAM3EB

      double kappa_quad =
          (rxxrxx / rxrx - std::pow(rxrxx, 2) / std::pow(rxrx, 2)) / std::pow(jacobi_, 2);

      if (kappa_quad < 0) kappa_quad = -kappa_quad;

      eint_ += 0.5 * wgt * jacobi_ * EA * std::pow(epsilon_ANS, 2);
      eint_axial_ += 0.5 * wgt * jacobi_ * EA * std::pow(epsilon_ANS, 2);
      eint_ += 0.5 * wgt * jacobi_ * EI * kappa_quad;

      // determine maximal curvature
      if (std::sqrt(kappa_quad) > kappa_max_) kappa_max_ = std::sqrt(kappa_quad);

      double epsilon_norm = std::sqrt(std::pow(epsilon_ANS, 2));

      // determine maximal axial tension
      if (epsilon_norm > epsilon_max_) epsilon_max_ = epsilon_norm;
#endif

      // store strain and stress resultant values in class variables
      axial_strain_gp_[numgp] = epsilon_ANS;
      curvature_gp_[numgp] = std::sqrt(kappa_quad);

      axial_force_gp_[numgp] = EA * axial_strain_gp_[numgp];
      bending_moment_gp_[numgp] = EI * curvature_gp_[numgp];
    }

    // tensor of mass moments of inertia for translational and rotational motion
    double mass_inertia_translational = 0.0;
    get_translational_mass_inertia_factor(mass_inertia_translational);

    std::vector<double> myvel(12, 0.0);

#ifndef INEXTENSIBLE
    for (int i = 0; i < 12; i++) myvel[i] = vel[i];
#else
    for (int i = 0; i < 6; i++)
    {
      myvel[i] = vel[i];
      myvel[i + 6] = vel[i + 7];
    }
#endif

    Core::LinAlg::Matrix<3, nnode * dofpn> N_mass;
    // Loop through all GP and calculate their contribution to the mass matrix
    for (int numgp = 0; numgp < gausspoints.nquad; numgp++)
    {
      Core::LinAlg::Matrix<3, 1> r_t(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<3, 1> r(Core::LinAlg::Initialization::zero);

      N_i.clear();
      N_mass.clear();
      NTilde.clear();

      // Get location and weight of GP in parameter space
      const double xi = gausspoints.qxg[numgp][0];
      const double wgt = gausspoints.qwgt[numgp];

#if (NODALDOFS == 2)
      // Get hermite derivatives N'xi and N''xi (jacobi_*2.0 is length of the element)
      Core::FE::shape_function_hermite_1d(N_i, xi, jacobi_ * 2.0, distype);
      // end--------------------------------------------------------
#elif (NODALDOFS == 3)
      FOUR_C_THROW("massmatrix only implemented for the case NODALDOFS == 2!!!");
#else
      FOUR_C_THROW("Only the values NODALDOFS = 2 and NODALDOFS = 3 are valid!");
#endif

      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < nnode * NODALDOFS; ++j) N_mass(i, i + 3 * j) += N_i(j);

      for (int i = 0; i < 3; i++)
        for (int j = 0; j < nnode * NODALDOFS; j++) r_t(i) += N_i(j) * myvel[3 * j + i];

      // calculate r' and r''
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < nnode * NODALDOFS; j++) r(i, 0) += N_i(j) * disp_totlag(3 * j + i);

      NTilde.multiply_tn(N_mass, N_mass);

      if (massmatrix != nullptr)
      {
#ifndef INEXTENSIBLE
        for (int i = 0; i < 6 * nnode; i++)
          for (int j = 0; j < 6 * nnode; j++)
            (*massmatrix)(i, j) += mass_inertia_translational * wgt * jacobi_ * NTilde(i, j);
#else
        int i1 = 0;
        int j1 = 0;
        for (int i = 0; i < 6 * nnode; i++)
        {
          if (i < 6)
            i1 = i;
          else
            i1 = i + 1;

          for (int j = 0; j < 6 * nnode; j++)
          {
            if (j < 6)
              j1 = j;
            else
              j1 = j + 1;

            (*massmatrix)(i1, j1) += mass_inertia_translational * wgt * jacobi_ * NTilde(i, j);
          }
        }
#endif
      }  // if (massmatrix != nullptr)

      ekin_ += 0.5 * wgt * jacobi_ * mass_inertia_translational * std::pow(r_t.norm2(), 2.0);

      Core::LinAlg::Matrix<3, 1> dL(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<3, 3> S_r(Core::LinAlg::Initialization::zero);
      Core::LargeRotations::computespin(S_r, r);
      dL.multiply(S_r, r_t);
      dL.scale(mass_inertia_translational);
      for (int i = 0; i < 3; i++)
      {
        l_(i) += wgt * jacobi_ * dL(i);
        p_(i) += wgt * jacobi_ * mass_inertia_translational * r_t(i);
      }
    }
  }
#endif

  return;
}

/*-----------------------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------------------*/
template <unsigned int nnode, unsigned int dofpn>
void Discret::Elements::Beam3eb::update_disp_totlag(
    const std::vector<double>& disp, Core::LinAlg::Matrix<dofpn * nnode, 1>& disp_totlag) const
{
#ifndef INEXTENSIBLE
  // update displacement vector /d in thesis Meier d = [ r1 t1 r2 t2]
  for (unsigned int node = 0; node < nnode; node++)
  {
    for (unsigned int dof = 0; dof < dofpn; dof++)
    {
      if (dof < 3)
      {
        // position of nodes
        disp_totlag(node * dofpn + dof) = (nodes()[node]->x()[dof] + disp[node * dofpn + dof]);
      }
      else if (dof < 6)
      {
        // tangent at nodes
        disp_totlag(node * dofpn + dof) = (Tref_[node](dof - 3) + disp[node * dofpn + dof]);
      }
      else if (dof >= 6)
      {
#if NODALDOFS == 3
        // curvatures at nodes
        disp_totlag(node * dofpn + dof) =
            (Kref_[node](dof - 6) + disp[node * dofpn + dof]) * ScaleFactorColumn;
#endif
      }
    }
  }
#else
  // update displacement vector /d in thesis Meier d = [ r1 t1 r2 t2]
  for (int node = 0; node < 2; node++)
  {
    for (int dof = 0; dof < 6; dof++)
    {
      if (dof < 3)
      {
        // position of nodes
        disp_totlag(node * 6 + dof) = (Nodes()[node]->X()[dof] + disp[node * 7 + dof]);
      }
      else if (dof < 6)
      {
        // tangent at nodes
        disp_totlag(node * 6 + dof) = (Tref_[node](dof - 3) + disp[node * 7 + dof]);
      }
    }
  }  // for (int node = 0 ; node < nnode ; node++)
#endif
}

/*-----------------------------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------------------------*/
template <int nnode>
void Discret::Elements::Beam3eb::evaluate_ptc(
    Teuchos::ParameterList& params, Core::LinAlg::SerialDenseMatrix& elemat1)
{
  if (nnode > 2) FOUR_C_THROW("PTC implemented for 2-noded elements only");
  for (int node = 0; node < nnode; node++)
  {
    Core::LinAlg::Matrix<3, 1> t0(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> t(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; i++)
    {
      t0(i) = t0_(i, node);
      t(i) = t_(i, node);
    }
    t0.scale(1.0 / t0.norm2());
    t.scale(1.0 / t.norm2());

    double tTt0 = 0.0;
    for (int i = 0; i < 3; i++) tTt0 += t0(i) * t(i);

#ifdef BEAM3EBROTPTC
    // variant1: PTC for tangential degrees of freedom; the Lobatto integration weight is 0.5 for
    // 2-noded elements
    for (int k = 0; k < 3; k++)
    {
      elemat1(node * 6 + 3 + k, node * 6 + 3 + k) +=
          tTt0 * params.get<double>("crotptc", 0.0) * 0.5 * jacobi_;
    }

    for (int k = 0; k < 3; k++)
    {
      for (int l = 0; l < 3; l++)
      {
        elemat1(node * 6 + 3 + k, node * 6 + 3 + l) +=
            params.get<double>("crotptc", 0.0) * 0.5 * jacobi_ * t(k) * t0(l);
      }
    }
#else
    // variant2: PTC for tangential degrees of freedom; the Lobatto integration weight is 0.5 for
    // 2-noded elements
    for (int k = 0; k < 3; k++)
    {
      elemat1(node * 6 + 3 + k, node * 6 + 3 + k) +=
          tTt0 * params.get<double>("crotptc", 0.0) * 0.5 * jacobi_;
    }
#endif

    // PTC for translational degrees of freedom; the Lobatto integration weight is 0.5 for 2-noded
    // elements
    for (int k = 0; k < 3; k++)
      elemat1(node * 6 + k, node * 6 + k) += params.get<double>("ctransptc", 0.0) * 0.5 * jacobi_;
  }

  return;
}

/*------------------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3eb::lumpmass(Core::LinAlg::SerialDenseMatrix* emass)
{
  std::cout << "\n\nWarning: Massmatrix not implemented yet!";
}

/*-----------------------------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------------------------*/
int Discret::Elements::Beam3eb::how_many_random_numbers_i_need() const
{
  // get Gauss points and weights for evaluation of damping matrix
  Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(mygaussruleeb);
/* at each Gauss point one needs as many random numbers as randomly excited degrees of freedom, i.e.
 * three random numbers for the translational degrees of freedom */
#ifdef BEAM3EBCONSTSTOCHFORCE
  return (3);
#else
  return (3 * gausspoints.nquad);
#endif
}

/*-----------------------------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------------------------*/
template <unsigned int nnode, unsigned int vpernode, int ndim>
void Discret::Elements::Beam3eb::evaluate_translational_damping(
    Teuchos::ParameterList& params,  //!< parameter list
    const Core::LinAlg::Matrix<ndim * vpernode * nnode, 1>& vel,
    const Core::LinAlg::Matrix<ndim * vpernode * nnode, 1>& disp_totlag,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix,  //!< element stiffness matrix
    Core::LinAlg::SerialDenseVector* force)        //!< element internal force vector
{
  // get time step size
  double dt = 1000;
  if (is_params_interface())
    dt = params_interface().get_delta_time();
  else
    dt = params.get<double>("delta time", 1000);

  // get damping coefficients for translational and rotational degrees of freedom (the latter is
  // unused in this element)
  Core::LinAlg::Matrix<ndim, 1> gamma(Core::LinAlg::Initialization::zero);
  get_damping_coefficients(gamma);

  // velocity and gradient of background velocity field
  Core::LinAlg::Matrix<ndim, 1> velbackground(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<ndim, ndim> velbackgroundgrad(Core::LinAlg::Initialization::zero);

  // evaluation point in physical space corresponding to a certain Gauss point in parameter space
  Core::LinAlg::Matrix<ndim, 1> evaluationpoint(Core::LinAlg::Initialization::zero);
  // tangent vector (derivative of beam centerline curve r with respect to arc-length parameter s)
  Core::LinAlg::Matrix<ndim, 1> r_s(Core::LinAlg::Initialization::zero);
  // velocity of beam centerline point relative to background fluid velocity
  Core::LinAlg::Matrix<ndim, 1> vel_rel(Core::LinAlg::Initialization::zero);

  // viscous force vector per unit length at current GP
  Core::LinAlg::Matrix<ndim, 1> f_visc(Core::LinAlg::Initialization::zero);
  // damping matrix
  Core::LinAlg::Matrix<ndim, ndim> damp_mat(Core::LinAlg::Initialization::zero);

  // get Gauss points and weights
  Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(mygaussruleeb);

  // matrix to store individual Hermite shape functions and their derivatives evaluated at a certain
  // Gauss point
  Core::LinAlg::Matrix<1, nnode * vpernode> N_i(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, nnode * vpernode> N_i_xi(Core::LinAlg::Initialization::zero);


  for (int gp = 0; gp < gausspoints.nquad; gp++)
  {
    Discret::Utils::Beam::evaluate_shape_functions_and_derivs_at_xi<nnode, vpernode>(
        gausspoints.qxg[gp][0], N_i, N_i_xi, this->shape(), this->ref_length());

    // compute position vector r of point in physical space corresponding to Gauss point
    calc_r<nnode, vpernode, double>(disp_totlag, N_i, evaluationpoint);

    // compute tangent vector t_{\par}=r' at current Gauss point
    calc_r_s<nnode, vpernode, double>(disp_totlag, N_i_xi, jacobi_, r_s);

    // compute velocity and gradient of background flow field at point r
    get_background_velocity<ndim, double>(
        params, evaluationpoint, velbackground, velbackgroundgrad);

    // compute velocity vector at this Gauss point via same interpolation as for centerline position
    // vector
    Discret::Utils::Beam::calc_interpolation<nnode, vpernode, 3, double>(vel, N_i, vel_rel);
    vel_rel -= velbackground;

    // loop over lines and columns of damping matrix
    for (unsigned int idim = 0; idim < ndim; idim++)
      for (unsigned int jdim = 0; jdim < ndim; jdim++)
        damp_mat(idim, jdim) =
            (idim == jdim) * gamma(1) + (gamma(0) - gamma(1)) * r_s(idim) * r_s(jdim);

    // compute viscous force vector per unit length at current GP
    f_visc.multiply(damp_mat, vel_rel);


    if (force != nullptr)
    {
      // loop over all shape functions
      for (unsigned int i = 0; i < nnode * vpernode; i++)
        // loop over dimensions
        for (unsigned int idim = 0; idim < ndim; idim++)
          (*force)(i* ndim + idim) += N_i(i) * jacobi_ * gausspoints.qwgt[gp] * f_visc(idim);
    }

    if (stiffmatrix != nullptr)
    {
      // compute matrix product of damping matrix and gradient of background velocity
      Core::LinAlg::Matrix<ndim, ndim> dampmatvelbackgroundgrad(Core::LinAlg::Initialization::zero);
      dampmatvelbackgroundgrad.multiply(damp_mat, velbackgroundgrad);

      // loop over all shape functions in row dimension
      for (unsigned int i = 0; i < nnode * vpernode; i++)
        // loop over all shape functions in column dimension
        for (unsigned int j = 0; j < nnode * vpernode; j++)
        {
          for (unsigned int idim = 0; idim < ndim; idim++)
            for (unsigned int jdim = 0; jdim < ndim; jdim++)
            {
              (*stiffmatrix)(i* ndim + idim, j * ndim + jdim) +=
                  gausspoints.qwgt[gp] * N_i(i) * N_i(j) * jacobi_ * damp_mat(idim, jdim) / dt;
              (*stiffmatrix)(i* ndim + idim, j * ndim + jdim) -=
                  gausspoints.qwgt[gp] * N_i(i) * N_i(j) * jacobi_ *
                  dampmatvelbackgroundgrad(idim, jdim);
              (*stiffmatrix)(i* ndim + idim, j * ndim + idim) += gausspoints.qwgt[gp] * N_i(i) *
                                                                 N_i_xi(j) * (gamma(0) - gamma(1)) *
                                                                 r_s(jdim) * vel_rel(jdim);
              (*stiffmatrix)(i* ndim + idim, j * ndim + jdim) += gausspoints.qwgt[gp] * N_i(i) *
                                                                 N_i_xi(j) * (gamma(0) - gamma(1)) *
                                                                 r_s(idim) * vel_rel(jdim);
            }
        }
    }
  }
}

/*-----------------------------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------------------------*/
template <unsigned int nnode, unsigned int vpernode, unsigned int ndim, unsigned int randompergauss>
void Discret::Elements::Beam3eb::evaluate_stochastic_forces(
    Teuchos::ParameterList& params,  //!< parameter list
    const Core::LinAlg::Matrix<ndim * vpernode * nnode, 1>& disp_totlag,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix,  //!< element stiffness matrix
    Core::LinAlg::SerialDenseVector* force)        //!< element internal force vector
{
  // damping coefficients for three translational and one rotatinal degree of freedom
  Core::LinAlg::Matrix<3, 1> gamma(Core::LinAlg::Initialization::zero);
  get_damping_coefficients(gamma);

  /*get pointer at multivector in parameter list linking to random numbers for stochastic
   * forces with zero mean and standard deviation (2*kT / dt)^0.5; note carefully: a space between
   * the two subsequal ">" signs is mandatory for the C++ parser in order to avoid confusion with
   * ">>" for streams*/
  std::shared_ptr<Core::LinAlg::MultiVector<double>> randomforces =
      brownian_dyn_params_interface().get_random_forces();

  // tangent vector (derivative of beam centerline curve r with respect to arc-length parameter s)
  Core::LinAlg::Matrix<ndim, 1> r_s(Core::LinAlg::Initialization::zero);

  // my random number vector at current GP
  Core::LinAlg::Matrix<ndim, 1> randnumvec(Core::LinAlg::Initialization::zero);

  // stochastic force vector per unit length at current GP
  Core::LinAlg::Matrix<ndim, 1> f_stoch(Core::LinAlg::Initialization::zero);


  // get Gauss points and weights for evaluation of damping matrix
  Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(mygaussruleeb);

  // matrix to store hermite shape functions and their derivatives evaluated at a certain Gauss
  // point
  Core::LinAlg::Matrix<1, nnode * vpernode> N_i;
  Core::LinAlg::Matrix<1, nnode * vpernode> N_i_xi;

  for (int gp = 0; gp < gausspoints.nquad; gp++)
  {
    Discret::Utils::Beam::evaluate_shape_functions_and_derivs_at_xi<nnode, vpernode>(
        gausspoints.qxg[gp][0], N_i, N_i_xi, this->shape(), this->ref_length());

    // compute tangent vector t_{\par}=r' at current Gauss point
    calc_r_s<nnode, vpernode, double>(disp_totlag, N_i_xi, jacobi_, r_s);

    // extract random numbers from global vector
    for (unsigned int idim = 0; idim < ndim; idim++)
    {
#ifndef BEAM3EBCONSTSTOCHFORCE
      randnumvec(idim) = (*randomforces)(gp * randompergauss + idim).local_values_as_span()[lid()];
#else
      randnumvec(idim) = (*randomforces)[idim][LID()];
#endif
    }

    // compute stochastic force vector per unit length at current GP
    f_stoch.clear();
    for (unsigned int idim = 0; idim < ndim; idim++)
      for (unsigned int jdim = 0; jdim < ndim; jdim++)
        f_stoch(idim) += (std::sqrt(gamma(1)) * (idim == jdim) +
                             (std::sqrt(gamma(0)) - std::sqrt(gamma(1))) * r_s(idim) * r_s(jdim)) *
                         randnumvec(jdim);


    if (force != nullptr)
    {
      // loop over all shape functions
      for (unsigned int i = 0; i < nnode * vpernode; i++)
        // loop over dimensions
        for (unsigned int idim = 0; idim < ndim; idim++)
          (*force)(i* ndim + idim) -=
              N_i(i) * f_stoch(idim) * std::sqrt(jacobi_ * gausspoints.qwgt[gp]);
    }

    if (stiffmatrix != nullptr)
    {
      // loop over all shape functions in row dimension
      for (unsigned int i = 0; i < nnode * vpernode; i++)
        // loop over all shape functions in column dimension
        for (unsigned int j = 0; j < nnode * vpernode; j++)
        {
          for (unsigned int idim = 0; idim < ndim; idim++)
            for (unsigned int jdim = 0; jdim < ndim; jdim++)
            {
              (*stiffmatrix)(i* ndim + idim, j * ndim + idim) -=
                  N_i(i) * N_i_xi(j) * r_s(jdim) * randnumvec(jdim) *
                  std::sqrt(gausspoints.qwgt[gp] / jacobi_) *
                  (std::sqrt(gamma(0)) - std::sqrt(gamma(1)));
              (*stiffmatrix)(i* ndim + idim, j * ndim + jdim) -=
                  N_i(i) * N_i_xi(j) * r_s(idim) * randnumvec(jdim) *
                  std::sqrt(gausspoints.qwgt[gp] / jacobi_) *
                  (std::sqrt(gamma(0)) - std::sqrt(gamma(1)));
            }
        }
    }
  }
}

/*-----------------------------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------------------------*/
template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
void Discret::Elements::Beam3eb::calc_brownian_forces_and_stiff(Teuchos::ParameterList& params,
    std::vector<double>& vel,                      //!< element velocity vector
    std::vector<double>& disp,                     //!< element displacement vector
    Core::LinAlg::SerialDenseMatrix* stiffmatrix,  //!< element stiffness matrix
    Core::LinAlg::SerialDenseVector* force)        //!< element internal force vector
{
  // unshift node positions, i.e. manipulate element displacement vector
  // as if there where no periodic boundary conditions
  if (brownian_dyn_params_interface_ptr() != nullptr)
    un_shift_node_position(disp, *brownian_dyn_params_interface().get_periodic_bounding_box());

  // update current total position state of element
  Core::LinAlg::Matrix<nnode * vpernode * ndim, 1> disp_totlag(Core::LinAlg::Initialization::zero);
  update_disp_totlag<nnode, vpernode * ndim>(disp, disp_totlag);

  // export current velocity state of element to fixed size matrix
  const Core::LinAlg::Matrix<nnode * vpernode * ndim, 1> vel_fixedsize(vel.data());

  // Evaluation of force vectors and stiffness matrices

  // add stiffness and forces due to translational damping effects
  evaluate_translational_damping<nnode, vpernode, ndim>(
      params, vel_fixedsize, disp_totlag, stiffmatrix, force);

  // add stochastic forces and (if required) resulting stiffness
  evaluate_stochastic_forces<nnode, vpernode, ndim, 3>(params, disp_totlag, stiffmatrix, force);
}

/*----------------------------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------------------------*/
double Discret::Elements::Beam3eb::get_axial_strain(
    double& xi, const Core::LinAlg::Matrix<12, 1>& disp_totlag) const
{
  // Todo implement and call more general method from Beam3Base

  Core::LinAlg::Matrix<3, 1> r_s(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 4> N_i_x(Core::LinAlg::Initialization::zero);
  const Core::FE::CellType distype = shape();
  // First get shape functions
  Core::FE::shape_function_hermite_1d_deriv1(N_i_x, xi, jacobi_ * 2.0, distype);

  for (int i = 0; i < 2 * NODALDOFS; i++) N_i_x(i) = N_i_x(i) / jacobi_;

  Core::LinAlg::Matrix<3, 1> epsilon_cp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> tangent_cp(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++)
  {
    tangent_cp(i, 0) = disp_totlag(i + 3);
    tangent_cp(i, 1) = disp_totlag(i + 9);

    for (int j = 0; j < 2 * NODALDOFS; j++) tangent_cp(i, 2) += N_i_x(j) * disp_totlag(3 * j + i);
  }

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++) epsilon_cp(i) += tangent_cp(j, i) * tangent_cp(j, i);

    epsilon_cp(i) = std::pow(epsilon_cp(i), 0.5) - 1.0;
  }

  Core::LinAlg::Matrix<1, 3> L_i(Core::LinAlg::Initialization::zero);
  Core::FE::shape_function_1d(L_i, xi, Core::FE::CellType::line3);
  double epsilon = 0.0;
  for (int i = 0; i < ANSVALUES; i++) epsilon += L_i(i) * epsilon_cp(i);

  return epsilon;
}

/*----------------------------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------------------------*/
// explicit template instantiations
template void Discret::Elements::Beam3eb::evaluate_ptc<2>(
    Teuchos::ParameterList&, Core::LinAlg::SerialDenseMatrix&);

template void Discret::Elements::Beam3eb::evaluate_translational_damping<2, 2, 3>(
    Teuchos::ParameterList&, const Core::LinAlg::Matrix<12, 1>&, const Core::LinAlg::Matrix<12, 1>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseVector*);

template void Discret::Elements::Beam3eb::evaluate_stochastic_forces<2, 2, 3, 3>(
    Teuchos::ParameterList&, const Core::LinAlg::Matrix<12, 1>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*);

template void Discret::Elements::Beam3eb::calc_brownian_forces_and_stiff<2, 2, 3>(
    Teuchos::ParameterList&, std::vector<double>&, std::vector<double>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseVector*);

template void Discret::Elements::Beam3eb::update_disp_totlag<2, 6>(
    const std::vector<double>&, Core::LinAlg::Matrix<12, 1>&) const;

FOUR_C_NAMESPACE_CLOSE

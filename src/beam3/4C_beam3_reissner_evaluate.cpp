// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beam3_reissner.hpp"
#include "4C_beam3_spatial_discretization_utils.hpp"
#include "4C_beam3_triad_interpolation_local_rotation_vectors.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_beam_elasthyper.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_structure_new_enum_lists.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_fad.hpp"
#include "4C_utils_function.hpp"

#include <iomanip>
#include <iostream>
#include <memory>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------------------*
 |  evaluate the element (public) cyron 01/08|
 *----------------------------------------------------------------------------------------------------------*/
int Discret::Elements::Beam3r::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1,  // nonlinear stiffness matrix
    Core::LinAlg::SerialDenseMatrix& elemat2,  // nonlinear mass matrix
    Core::LinAlg::SerialDenseVector& elevec1,  // nonlinear internal (elastic) forces
    Core::LinAlg::SerialDenseVector& elevec2,  // nonlinear inertia forces
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // Set structure params interface pointer
  set_params_interface_ptr(params);
  // Set brwonian params interface pointer
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
      FOUR_C_THROW("Unknown type of action for Beam3r");
  }

  // nnodetriad: number of nodes used for interpolation of triad field
  const int nnodetriad = num_node();

  switch (act)
  {
    case Core::Elements::struct_calc_ptcstiff:
    {
      switch (nnodetriad)
      {
        case 2:
          evaluate_ptc<2>(params, elemat1);
          break;
        case 3:
          evaluate_ptc<3>(params, elemat1);
          break;
        case 4:
          evaluate_ptc<4>(params, elemat1);
          break;
        case 5:
          evaluate_ptc<5>(params, elemat1);
          break;
        default:
          FOUR_C_THROW("Only Line2, Line3, Line4 and Line5 Elements implemented.");
      }
      break;
    }

    case Core::Elements::struct_calc_linstiff:
    {
      // only nonlinear case implemented!
      FOUR_C_THROW("linear stiffness matrix called, but not implemented");
      break;
    }

    case Core::Elements::struct_calc_energy:
    {
      if (is_params_interface())  // new structural time integration
      {
        params_interface().add_contribution_to_energy_type(eint_, Solid::internal_energy);
        params_interface().add_contribution_to_energy_type(ekin_, Solid::kinetic_energy);
      }
      break;
    }

    case Core::Elements::struct_calc_nlnstiffmass:
    case Core::Elements::struct_calc_nlnstifflmass:
    case Core::Elements::struct_calc_nlnstiff:
    case Core::Elements::struct_calc_internalforce:
    case Core::Elements::struct_calc_internalinertiaforce:
    {
      // need current global displacement and residual forces and get them from discretization
      // making use of the local-to-global map lm one can extract current displacement and residual
      // values for each degree of freedom

      // get element displacements
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);

      if (act == Core::Elements::struct_calc_nlnstiffmass)
      {
        switch (nnodetriad)
        {
          case 2:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<2, 2, 1>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            else
              calc_internal_and_inertia_forces_and_stiff<2, 2, 2>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            break;
          }
          case 3:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<3, 3, 1>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            else
              calc_internal_and_inertia_forces_and_stiff<3, 2, 2>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            break;
          }
          case 4:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<4, 4, 1>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            else
              calc_internal_and_inertia_forces_and_stiff<4, 2, 2>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            break;
          }
          case 5:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<5, 5, 1>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            else
              calc_internal_and_inertia_forces_and_stiff<5, 2, 2>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            break;
          }
        }
      }

      else if (act == Core::Elements::struct_calc_nlnstifflmass)
      {
        // TODO there is a method 'Beam3r::lumpmass'; check generality and functionality and enable
        // action here
        FOUR_C_THROW("Lumped mass matrix not implemented for beam3r elements so far!");
      }

      else if (act == Core::Elements::struct_calc_nlnstiff)
      {
        switch (nnodetriad)
        {
          case 2:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<2, 2, 1>(
                  params, mydisp, &elemat1, nullptr, &elevec1, nullptr);
            else
              calc_internal_and_inertia_forces_and_stiff<2, 2, 2>(
                  params, mydisp, &elemat1, nullptr, &elevec1, nullptr);
            break;
          }
          case 3:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<3, 3, 1>(
                  params, mydisp, &elemat1, nullptr, &elevec1, nullptr);
            else
              calc_internal_and_inertia_forces_and_stiff<3, 2, 2>(
                  params, mydisp, &elemat1, nullptr, &elevec1, nullptr);
            break;
          }
          case 4:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<4, 4, 1>(
                  params, mydisp, &elemat1, nullptr, &elevec1, nullptr);
            else
              calc_internal_and_inertia_forces_and_stiff<4, 2, 2>(
                  params, mydisp, &elemat1, nullptr, &elevec1, nullptr);
            break;
          }
          case 5:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<5, 5, 1>(
                  params, mydisp, &elemat1, nullptr, &elevec1, nullptr);
            else
              calc_internal_and_inertia_forces_and_stiff<5, 2, 2>(
                  params, mydisp, &elemat1, nullptr, &elevec1, nullptr);
            break;
          }
          default:
            FOUR_C_THROW("Only Line2, Line3, Line4, and Line5 Elements implemented.");
        }
      }
      else if (act == Core::Elements::struct_calc_internalforce)
      {
        switch (nnodetriad)
        {
          case 2:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<2, 2, 1>(
                  params, mydisp, nullptr, nullptr, &elevec1, nullptr);
            else
              calc_internal_and_inertia_forces_and_stiff<2, 2, 2>(
                  params, mydisp, nullptr, nullptr, &elevec1, nullptr);
            break;
          }
          case 3:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<3, 3, 1>(
                  params, mydisp, nullptr, nullptr, &elevec1, nullptr);
            else
              calc_internal_and_inertia_forces_and_stiff<3, 2, 2>(
                  params, mydisp, nullptr, nullptr, &elevec1, nullptr);
            break;
          }
          case 4:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<4, 4, 1>(
                  params, mydisp, nullptr, nullptr, &elevec1, nullptr);
            else
              calc_internal_and_inertia_forces_and_stiff<4, 2, 2>(
                  params, mydisp, nullptr, nullptr, &elevec1, nullptr);
            break;
          }
          case 5:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<5, 5, 1>(
                  params, mydisp, nullptr, nullptr, &elevec1, nullptr);
            else
              calc_internal_and_inertia_forces_and_stiff<5, 2, 2>(
                  params, mydisp, nullptr, nullptr, &elevec1, nullptr);
            break;
          }
          default:
            FOUR_C_THROW("Only Line2, Line3, Line4, and Line5 Elements implemented.");
        }
      }

      else if (act == Core::Elements::struct_calc_internalinertiaforce)
      {
        switch (nnodetriad)
        {
          case 2:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<2, 2, 1>(
                  params, mydisp, nullptr, nullptr, &elevec1, &elevec2);
            else
              calc_internal_and_inertia_forces_and_stiff<2, 2, 2>(
                  params, mydisp, nullptr, nullptr, &elevec1, &elevec2);
            break;
          }
          case 3:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<3, 3, 1>(
                  params, mydisp, nullptr, nullptr, &elevec1, &elevec2);
            else
              calc_internal_and_inertia_forces_and_stiff<3, 2, 2>(
                  params, mydisp, nullptr, nullptr, &elevec1, &elevec2);
            break;
          }
          case 4:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<4, 4, 1>(
                  params, mydisp, nullptr, nullptr, &elevec1, &elevec2);
            else
              calc_internal_and_inertia_forces_and_stiff<4, 2, 2>(
                  params, mydisp, nullptr, nullptr, &elevec1, &elevec2);
            break;
          }
          case 5:
          {
            if (!centerline_hermite_)
              calc_internal_and_inertia_forces_and_stiff<5, 5, 1>(
                  params, mydisp, nullptr, nullptr, &elevec1, &elevec2);
            else
              calc_internal_and_inertia_forces_and_stiff<5, 2, 2>(
                  params, mydisp, nullptr, nullptr, &elevec1, &elevec2);
            break;
          }
          default:
            FOUR_C_THROW("Only Line2, Line3, Line4, and Line5 Elements implemented.");
        }
      }

      break;
    }

    case Core::Elements::struct_calc_update_istep:
    {
      /* the action calc_struct_update_istep is called in the very end of a time step when the new
       * dynamic equilibrium has finally been found; this is the point where the variable
       * representing the geometric status of the beam at the end of the time step has to be
       * stored*/
      qconvnode_ = qnewnode_;
      qconv_gp_mass_ = qnew_gp_mass_;
      wconv_gp_mass_ = wnew_gp_mass_;
      aconv_gp_mass_ = anew_gp_mass_;
      amodconv_gp_mass_ = amodnew_gp_mass_;
      rttconv_gp_mass_ = rttnew_gp_mass_;
      rttmodconv_gp_mass_ = rttmodnew_gp_mass_;
      rtconv_gp_mass_ = rtnew_gp_mass_;
      rconv_gp_mass_ = rnew_gp_mass_;
      qconv_gp_dampstoch_ = qnew_gp_dampstoch_;
      get_beam_material().update();
      break;
    }

    case Core::Elements::struct_calc_reset_istep:
    {
      /* the action calc_struct_reset_istep is called by the adaptive time step controller; carries
       * out one test step whose purpose is only figuring out a suitable time step; thus this step
       * may be a very bad one in order to iterated towards the new dynamic equilibrium and the
       * thereby gained new geometric configuration should not be applied as starting point for any
       * further iteration step; as a consequence the thereby generated change of the geometric
       * configuration should be canceled and the configuration should be reset to the value at the
       * beginning of the time step*/
      qnewnode_ = qconvnode_;
      qnew_gp_mass_ = qconv_gp_mass_;
      wnew_gp_mass_ = wconv_gp_mass_;
      anew_gp_mass_ = aconv_gp_mass_;
      amodnew_gp_mass_ = amodconv_gp_mass_;
      rttnew_gp_mass_ = rttconv_gp_mass_;
      rttmodnew_gp_mass_ = rttmodconv_gp_mass_;
      rtnew_gp_mass_ = rtconv_gp_mass_;
      rnew_gp_mass_ = rconv_gp_mass_;
      qnew_gp_dampstoch_ = qconv_gp_dampstoch_;
      get_beam_material().reset();
      break;
    }

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
      {
        switch (nnodetriad)
        {
          case 2:
          {
            if (!centerline_hermite_)
              calc_brownian_forces_and_stiff<2, 2, 1>(params, myvel, mydisp, nullptr, &elevec1);
            else
              calc_brownian_forces_and_stiff<2, 2, 2>(params, myvel, mydisp, nullptr, &elevec1);
            break;
          }
          case 3:
          {
            if (!centerline_hermite_)
              calc_brownian_forces_and_stiff<3, 3, 1>(params, myvel, mydisp, nullptr, &elevec1);
            else
              calc_brownian_forces_and_stiff<3, 2, 2>(params, myvel, mydisp, nullptr, &elevec1);
            break;
          }
          case 4:
          {
            if (!centerline_hermite_)
              calc_brownian_forces_and_stiff<4, 4, 1>(params, myvel, mydisp, nullptr, &elevec1);
            else
              calc_brownian_forces_and_stiff<4, 2, 2>(params, myvel, mydisp, nullptr, &elevec1);
            break;
          }
          case 5:
          {
            if (!centerline_hermite_)
              calc_brownian_forces_and_stiff<5, 5, 1>(params, myvel, mydisp, nullptr, &elevec1);
            else
              calc_brownian_forces_and_stiff<5, 2, 2>(params, myvel, mydisp, nullptr, &elevec1);
            break;
          }
          default:
            FOUR_C_THROW("Only Line2, Line3, Line4, and Line5 Elements implemented.");
        }
      }
      else if (act == Core::Elements::struct_calc_brownianstiff)
      {
        switch (nnodetriad)
        {
          case 2:
          {
            if (!centerline_hermite_)
              calc_brownian_forces_and_stiff<2, 2, 1>(params, myvel, mydisp, &elemat1, &elevec1);
            else
              calc_brownian_forces_and_stiff<2, 2, 2>(params, myvel, mydisp, &elemat1, &elevec1);
            break;
          }
          case 3:
          {
            if (!centerline_hermite_)
              calc_brownian_forces_and_stiff<3, 3, 1>(params, myvel, mydisp, &elemat1, &elevec1);
            else
              calc_brownian_forces_and_stiff<3, 2, 2>(params, myvel, mydisp, &elemat1, &elevec1);
            break;
          }
          case 4:
          {
            if (!centerline_hermite_)
              calc_brownian_forces_and_stiff<4, 4, 1>(params, myvel, mydisp, &elemat1, &elevec1);
            else
              calc_brownian_forces_and_stiff<4, 2, 2>(params, myvel, mydisp, &elemat1, &elevec1);
            break;
          }
          case 5:
          {
            if (!centerline_hermite_)
              calc_brownian_forces_and_stiff<5, 5, 1>(params, myvel, mydisp, &elemat1, &elevec1);
            else
              calc_brownian_forces_and_stiff<5, 2, 2>(params, myvel, mydisp, &elemat1, &elevec1);
            break;
          }
          default:
            FOUR_C_THROW("Only Line2, Line3, Line4, and Line5 Elements implemented.");
        }
      }
      else
        FOUR_C_THROW("You shouldn't be here.");

      break;
    }

    // write stress and strain output
    case Core::Elements::struct_calc_stress:
    {
      break;
    }

    case Core::Elements::struct_calc_recover:
    {
      // do nothing here
      break;
    }

    case Core::Elements::struct_calc_predict:
    {
      // do nothing here
      break;
    }

    // element based PTC scaling
    case Core::Elements::struct_calc_addjacPTC:
    {
      calc_stiff_contributions_ptc(elemat1);
      break;
    }

    case Core::Elements::struct_gauss_point_data_output:
    case Core::Elements::struct_init_gauss_point_data_output:
    {
      // do nothing in this cases
      break;
    }

    default:
      std::cout << "\ncalled element with action type " << action_type_to_string(act);
      FOUR_C_THROW("This action type is not implemented for Beam3r");
      break;
  }
  return 0;
}

/*-----------------------------------------------------------------------------------------------------------*
 |  Integrate a Surface Neumann boundary condition (public) cyron 03/08|
 *----------------------------------------------------------------------------------------------------------*/
int Discret::Elements::Beam3r::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  set_params_interface_ptr(params);

  // find out whether we will use a time curve
  double time = -1.0;

  if (is_params_interface())
    time = params_interface().get_total_time();
  else
    time = params.get<double>("total time", -1.0);

  // nnodetriad: number of nodes used for interpolation of triad field
  const unsigned int nnodetriad = num_node();
  // nnodecl: number of nodes used for interpolation of centerline
  // assumptions: nnodecl<=nnodetriad; centerline nodes have local ID 0...nnodecl-1
  unsigned int nnodecl = nnodetriad;
  if (centerline_hermite_) nnodecl = 2;

  // vpernode: number of interpolated values per node (1: value (i.e. Lagrange), 2: value +
  // derivative of value (i.e. Hermite))
  unsigned int vpernode = 1;
  if (centerline_hermite_) vpernode = 2;

  // number of DOFs per node depending on type of node
  const unsigned int dofperclnode = 3 * vpernode;
  const unsigned int dofpertriadnode = 3;
  const unsigned int dofpercombinode = dofperclnode + dofpertriadnode;

  const Core::FE::CellType distype = this->shape();

  // gaussian points
  const Core::FE::IntegrationPoints1D intpoints(my_gauss_rule(neumann_lineload));

  // declaration of variables in order to store shape functions
  // used for interpolation of triad field
  Core::LinAlg::SerialDenseVector I_i(nnodetriad);
  // used for interpolation of centerline
  Core::LinAlg::SerialDenseVector H_i(vpernode * nnodecl);

  // get values and switches from the condition

  // onoff is related to the first numdf flags of a line Neumann condition in the input file;
  // value 1 for flag i says that condition is active for i-th degree of freedom
  const auto onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  // val is related to the numdf "VAL" fields after the onoff flags of the Neumann condition
  // in the input file; val gives the values of the force as a multiple of the prescribed load curve
  const auto val = condition.parameters().get<std::vector<double>>("VAL");
  // funct is related to the numdf "FUNCT" fields after the val field of the Neumann condition
  // in the input file; funct gives the number of the function defined in the section FUNCT
  const auto functions = condition.parameters().get<std::vector<std::optional<int>>>("FUNCT");

  // integration points in parameter space and weights
  double xi = 0.0;
  double wgt = 0.0;

  // integration loops
  for (int numgp = 0; numgp < intpoints.nquad; ++numgp)
  {
    xi = intpoints.qxg[numgp][0];
    wgt = intpoints.qwgt[numgp];

    // evaluation of shape functions at Gauss points
    Core::FE::shape_function_1d(I_i, xi, distype);
    if (centerline_hermite_)
      Core::FE::shape_function_hermite_1d(H_i, xi, reflength_, Core::FE::CellType::line2);
    else
      Core::FE::shape_function_1d(H_i, xi, distype);

    // position vector at the gauss point at reference configuration needed for function evaluation
    std::vector<double> X_ref(3, 0.0);

    // calculate coordinates of corresponding Gauss point in reference configuration
    for (unsigned int node = 0; node < nnodecl; node++)
    {
      for (unsigned int dim = 0; dim < 3; dim++)
      {
        X_ref[dim] += H_i[vpernode * node] * nodes()[node]->x()[dim];

        if (centerline_hermite_) X_ref[dim] += H_i[vpernode * node + 1] * (Tref_[node])(dim);
      }
    }

    double fac = 0;
    fac = wgt * jacobi_gp_neumannline_[numgp];

    // load vector ar
    double ar[6];

    // loop the relevant dofs of a node
    for (int dof = 0; dof < 6; ++dof) ar[dof] = fac * onoff[dof] * val[dof];
    double functionfac = 1.0;

    // sum up load components
    for (unsigned int dof = 0; dof < 6; ++dof)
    {
      // evaluate function at the position of the current GP if it is not none
      if (functions[dof].has_value() && functions[dof].value() > 0)
        functionfac = Global::Problem::instance()
                          ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functions[dof].value())
                          .evaluate(X_ref, time, dof);
      else
        functionfac = 1.0;

      for (unsigned int node = 0; node < nnodecl; ++node)
      {
        if (dof < 3)
        {
          elevec1[dofpercombinode * node + dof] += H_i[vpernode * node] * ar[dof] * functionfac;

          if (centerline_hermite_)
            elevec1[dofpercombinode * node + 6 + dof] +=
                H_i[vpernode * node + 1] * ar[dof] * functionfac;
        }
        else  // dof<6
          elevec1[dofpercombinode * node + dof] += I_i[node] * ar[dof] * functionfac;
      }

      for (unsigned int node = nnodecl; node < nnodetriad; ++node)
        if (dof > 2 && dof < 6)
          elevec1[dofperclnode * nnodecl + dofpertriadnode * node + dof - 3] +=
              I_i[node] * ar[dof] * functionfac;
    }
  }

  return 0;
}

/*----------------------------------------------------------------------------------------------------------------------*
 |push forward material stress vector and constitutive matrix to their spatial counterparts by
 rotation matrix Lambda   | |according to Romero 2004, eq. (3.10) cyron 04/10|
 *----------------------------------------------------------------------------------------------------------------------*/
template <typename T>
inline void Discret::Elements::Beam3r::pushforward(const Core::LinAlg::Matrix<3, 3, T>& Lambda,
    const Core::LinAlg::Matrix<3, 1, T>& stress_mat, const Core::LinAlg::Matrix<3, 3, T>& C_mat,
    Core::LinAlg::Matrix<3, 1, T>& stress_spatial, Core::LinAlg::Matrix<3, 3, T>& c_spatial) const
{
  // introduce auxiliary variable for pushforward of rotational matrices
  Core::LinAlg::Matrix<3, 3, T> temp;

  // push forward stress vector
  stress_spatial.multiply(Lambda, stress_mat);

  // push forward constitutive matrix according to Jelenic 1999, paragraph following to (2.22) on
  // page 148
  temp.multiply(Lambda, C_mat);
  c_spatial.multiply_nt(temp, Lambda);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff(
    Teuchos::ParameterList& params, std::vector<double>& disp,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseMatrix* massmatrix,
    Core::LinAlg::SerialDenseVector* force, Core::LinAlg::SerialDenseVector* inertia_force)
{
  //************ periodic boundary conditions **********************
  /* unshift node positions, i.e. manipulate element displacement vector
   * as if there where no periodic boundary conditions */
  if (brownian_dyn_params_interface_ptr() != nullptr)
    un_shift_node_position(disp, *brownian_dyn_params_interface().get_periodic_bounding_box());

  /* current nodal DOFs relevant for centerline interpolation in total Lagrangian
   * style, i.e. initial values + displacements */
  Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, double> disp_totlag_centerline(
      Core::LinAlg::Initialization::zero);

  // quaternions of all nodal triads
  std::vector<Core::LinAlg::Matrix<4, 1, double>> Qnode(nnodetriad);

  update_disp_tot_lag_and_nodal_triads<nnodetriad, nnodecl, vpernode, double>(
      disp, disp_totlag_centerline, Qnode);

  calc_internal_and_inertia_forces_and_stiff<nnodetriad, nnodecl, vpernode>(
      disp_totlag_centerline, Qnode, stiffmatrix, massmatrix, force, inertia_force);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff(
    Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, double>& disp_totlag_centerline,
    std::vector<Core::LinAlg::Matrix<4, 1, double>>& Qnode,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseMatrix* massmatrix,
    Core::LinAlg::SerialDenseVector* force, Core::LinAlg::SerialDenseVector* inertia_force)
{
  const unsigned int numdofelement = 3 * vpernode * nnodecl + 3 * nnodetriad;

  if (not use_fad_)
  {
    // internal force vector
    Core::LinAlg::Matrix<numdofelement, 1, double> internal_force(
        Core::LinAlg::Initialization::zero);

    if (force != nullptr)
    {
      internal_force.set_view(&((*force)(0)));
    }

    calc_internal_force_and_stiff<nnodetriad, nnodecl, vpernode, double>(
        disp_totlag_centerline, Qnode, stiffmatrix, internal_force);

    // calculation of inertia forces/moments and mass matrix
    if (massmatrix != nullptr or inertia_force != nullptr)
    {
      calc_inertia_force_and_mass_matrix<nnodetriad, nnodecl, vpernode>(
          disp_totlag_centerline, Qnode, massmatrix, inertia_force);
    }
  }
  else
  {
    // internal force vector
    Core::LinAlg::Matrix<numdofelement, 1, Sacado::Fad::DFad<double>> internal_force(
        Core::LinAlg::Initialization::zero);

    /* current nodal DOFs relevant for centerline interpolation in total Lagrangian
     * style, i.e. initial values + displacements */
    Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, Sacado::Fad::DFad<double>>
        disp_totlag_centerline_FAD;

    for (unsigned int i = 0; i < 3 * vpernode * nnodecl; ++i)
      disp_totlag_centerline_FAD(i) = disp_totlag_centerline(i);

    // quaternions of all nodal triads
    std::vector<Core::LinAlg::Matrix<4, 1, Sacado::Fad::DFad<double>>> Qnode_FAD(nnodetriad);

    for (unsigned int inode = 0; inode < nnodetriad; ++inode)
      for (unsigned int j = 0; j < 4; ++j) Qnode_FAD[inode](j) = Qnode[inode](j);

    set_automatic_differentiation_variables<nnodetriad, nnodecl, vpernode>(
        disp_totlag_centerline_FAD, Qnode_FAD);

    calc_internal_force_and_stiff<nnodetriad, nnodecl, vpernode, Sacado::Fad::DFad<double>>(
        disp_totlag_centerline_FAD, Qnode_FAD, nullptr, internal_force);

    if (force != nullptr)
    {
      for (unsigned int idof = 0; idof < numdofelement; ++idof)
        (*force)(idof) = Core::FADUtils::cast_to_double(internal_force(idof));
    }

    if (stiffmatrix != nullptr)
    {
      calc_stiffmat_automatic_differentiation<nnodetriad, nnodecl, vpernode>(
          *stiffmatrix, Qnode, internal_force);
    }

    // calculation of inertia forces/moments and mass matrix
    if (massmatrix != nullptr or inertia_force != nullptr)
    {
      calc_inertia_force_and_mass_matrix<nnodetriad, nnodecl, vpernode>(
          disp_totlag_centerline, Qnode, massmatrix, inertia_force);
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode, typename T>
void Discret::Elements::Beam3r::calc_internal_force_and_stiff(
    const Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, T>& disp_totlag_centerline,
    const std::vector<Core::LinAlg::Matrix<4, 1, T>>& Qnode,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix,
    Core::LinAlg::Matrix<3 * vpernode * nnodecl + 3 * nnodetriad, 1, T>& internal_force)
{
  // nnodetriad: number of nodes used for interpolation of triad field
  // nnodecl: number of nodes used for interpolation of centerline
  // assumptions: nnodecl<=nnodetriad; centerline nodes have local ID 0...nnodecl-1
  // vpernode: number of interpolated values per centerline node (1: value (i.e. Lagrange), 2: value
  // + derivative of value (i.e. Hermite))

  /********************************** Initialize/resize variables
   ***************************************
   *****************************************************************************************************/

  //********************************** quantities valid for entire element
  //*****************************
  const unsigned int dofperclnode = 3 * vpernode;
  const unsigned int dofpertriadnode = 3;
  const unsigned int dofpercombinode = dofperclnode + dofpertriadnode;

  // clear internal (elastic) energy
  eint_ = 0.0;

  //*************************** physical quantities evaluated at a certain GP
  //***************************

  // derivation of beam centerline with respect to arc-length parameter: r'(x) from (2.12), Jelenic
  // 1999
  Core::LinAlg::Matrix<3, 1, T> r_s;
  // spin matrix related to vector r_s
  Core::LinAlg::Matrix<3, 3, T> r_s_hat;
  // interpolated local relative rotation \Psi^l at a certain Gauss point according to (3.11),
  // Jelenic 1999
  Core::LinAlg::Matrix<3, 1, T> Psi_l;
  /* derivative of interpolated local relative rotation \Psi^l with respect to arc-length parameter
   * at a certain Gauss point according to (3.11), Jelenic 1999*/
  Core::LinAlg::Matrix<3, 1, T> Psi_l_s;
  // triad at GP
  Core::LinAlg::Matrix<3, 3, T> Lambda;

  // 3D vector related to spin matrix \hat{\kappa} from (2.1), Jelenic 1999
  Core::LinAlg::Matrix<3, 1, T> Cur;
  // 3D vector of material axial and shear strains from (2.1), Jelenic 1999
  Core::LinAlg::Matrix<3, 1, T> Gamma;

  // convected stresses N and M and constitutive matrices C_N and C_M according to section 2.4,
  // Jelenic 1999
  Core::LinAlg::Matrix<3, 1, T> stressN;
  Core::LinAlg::Matrix<3, 1, T> stressM;
  Core::LinAlg::Matrix<3, 3, T> CN;
  Core::LinAlg::Matrix<3, 3, T> CM;

  // spatial stresses n and m according to (3.10), Romero 2004 and spatial constitutive matrices c_n
  // and c_m according to page 148, Jelenic 1999
  Core::LinAlg::Matrix<3, 1, T> stressn;
  Core::LinAlg::Matrix<3, 1, T> stressm;
  Core::LinAlg::Matrix<3, 3, T> cn;
  Core::LinAlg::Matrix<3, 3, T> cm;

  //********************************** (generalized) shape functions
  //************************************
  /* Note: index i refers to the i-th shape function (i = 0 ... nnode*vpernode-1)
   * the vectors store individual shape functions, NOT an assembled matrix of shape functions)*/

  /* vector whose numgp-th element is a 1xnnode-matrix with all Lagrange polynomial shape functions
   * evaluated at the numgp-th Gauss point these shape functions are used for the interpolation of
   * the triad field*/
  std::vector<Core::LinAlg::Matrix<1, nnodetriad, double>> I_i;
  // same for the derivatives
  std::vector<Core::LinAlg::Matrix<1, nnodetriad, double>> I_i_xi;

  /* vector whose numgp-th element is a 1x(vpernode*nnode)-matrix with all (Lagrange/Hermite) shape
   * functions evaluated at the numgp-th GP
   * these shape functions are used for the interpolation of the beam centerline*/
  std::vector<Core::LinAlg::Matrix<1, vpernode * nnodecl, double>> H_i;
  // same for the derivatives
  std::vector<Core::LinAlg::Matrix<1, vpernode * nnodecl, double>> H_i_xi;


  /*************************** update/compute quantities valid for entire element
   ***********************
   *****************************************************************************************************/
  // setup constitutive matrices
  get_templated_beam_material<T>().compute_constitutive_parameter(CN, CM);

  // create object of triad interpolation scheme
  std::shared_ptr<LargeRotations::TriadInterpolationLocalRotationVectors<nnodetriad, T>>
      triad_interpolation_scheme_ptr =
          std::make_shared<LargeRotations::TriadInterpolationLocalRotationVectors<nnodetriad, T>>();

  // reset triad interpolation scheme based on nodal quaternions
  triad_interpolation_scheme_ptr->reset(Qnode);

  // matrix containing contributions to the jacobian depending on the material model
  Core::LinAlg::Matrix<3, 3, T> stiffness_contribution(Core::LinAlg::Initialization::zero);

  /******************************* elasticity: compute fint and stiffmatrix
   *****************************
   *****************************************************************************************************/

  //************************* residual and stiffmatrix contributions from forces
  //***********************
  // for these contributions, reduced integration is applied to avoid locking

  // get integration points for elasticity
  Core::FE::IntegrationPoints1D gausspoints_elast_force(my_gauss_rule(res_elastic_force));

  // reuse variables for individual shape functions and resize to new numgp
  I_i.resize(gausspoints_elast_force.nquad);
  H_i_xi.resize(gausspoints_elast_force.nquad);

  // evaluate all shape functions and derivatives with respect to element parameter xi at all
  // specified Gauss points
  Discret::Utils::Beam::evaluate_shape_functions_all_gps<nnodetriad, 1>(
      gausspoints_elast_force, I_i, this->shape());

  Discret::Utils::Beam::evaluate_shape_function_derivs_all_gps<nnodecl, vpernode>(
      gausspoints_elast_force, H_i_xi, this->shape(), this->ref_length());

  // re-assure correct size of strain and stress resultant class variables
  axial_strain_gp_elastf_.resize(gausspoints_elast_force.nquad);
  shear_strain_2_gp_elastf_.resize(gausspoints_elast_force.nquad);
  shear_strain_3_gp_elastf_.resize(gausspoints_elast_force.nquad);

  material_axial_force_gp_elastf_.resize(gausspoints_elast_force.nquad);
  material_shear_force_2_gp_elastf_.resize(gausspoints_elast_force.nquad);
  material_shear_force_3_gp_elastf_.resize(gausspoints_elast_force.nquad);

  spatial_x_force_gp_elastf_.resize(gausspoints_elast_force.nquad);
  spatial_y_force_2_gp_elastf_.resize(gausspoints_elast_force.nquad);
  spatial_z_force_3_gp_elastf_.resize(gausspoints_elast_force.nquad);

  // Loop through all GP and calculate their contribution to the forcevector and stiffnessmatrix
  for (int numgp = 0; numgp < gausspoints_elast_force.nquad; ++numgp)
  {
    // weight of GP in parameter space
    const double wgt = gausspoints_elast_force.qwgt[numgp];

    calc_r_s<nnodecl, vpernode, T>(
        disp_totlag_centerline, H_i_xi[numgp], jacobi_gp_elastf_[numgp], r_s);

    triad_interpolation_scheme_ptr->get_interpolated_triad_at_xi(
        Lambda, gausspoints_elast_force.qxg[numgp][0]);

    // compute spin matrix related to vector rprime for later use
    Core::LargeRotations::computespin<T>(r_s_hat, r_s);

    // compute material strains Gamma and Cur
    compute_gamma<T>(r_s, Lambda, gammaref_gp_[numgp], Gamma);

    get_templated_beam_material<T>().evaluate_force_contributions_to_stress(
        stressN, CN, Gamma, numgp);
    get_templated_beam_material<T>().get_stiffness_matrix_of_forces(
        stiffness_contribution, CN, numgp);

    pushforward<T>(Lambda, stressN, stiffness_contribution, stressn, cn);

    /* computation of internal forces according to Jelenic 1999, eq. (4.3); computation split up
     * with respect to single blocks of matrix in eq. (4.3)*/
    for (unsigned int node = 0; node < nnodecl; ++node)
    {
      /* upper left block
       * note: jacobi factor cancels out because it is defined by ds=(ds/dxi)*dxi
       *       and I^{i'} in Jelenic1999 is derivative with respect to arc-length parameter in
       * reference configuration s which can be computed from I_i_xi by multiplication with the
       * inverse determinant: I^{i'}=I_i_s=I_i_xi*(dxi/ds) */
      for (unsigned int k = 0; k < 3; ++k)
      {
        internal_force(dofpercombinode * node + k) +=
            H_i_xi[numgp](vpernode * node) * stressn(k) * wgt;
        if (centerline_hermite_)
          internal_force(dofpercombinode * node + 6 + k) +=
              H_i_xi[numgp](vpernode * node + 1) * stressn(k) * wgt;
      }

      // lower left block
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          internal_force(dofpercombinode * node + 3 + i) -=
              r_s_hat(i, j) * stressn(j) * I_i[numgp](node) * wgt * jacobi_gp_elastf_[numgp];
    }
    for (unsigned int node = nnodecl; node < nnodetriad;
        ++node)  // this loop is only entered in case of nnodetriad>nnodecl
    {
      // lower left block
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          internal_force(dofperclnode * nnodecl + dofpertriadnode * node + i) -=
              r_s_hat(i, j) * stressn(j) * I_i[numgp](node) * wgt * jacobi_gp_elastf_[numgp];
    }

    if (stiffmatrix != nullptr)
    {
      calc_stiffmat_analytic_force_contributions<nnodetriad, nnodecl, vpernode>(*stiffmatrix,
          stressn, cn, r_s_hat, *triad_interpolation_scheme_ptr, I_i[numgp], H_i_xi[numgp], wgt,
          jacobi_gp_elastf_[numgp]);
    }

    // add elastic energy from forces at this GP
    for (unsigned int dim = 0; dim < 3; ++dim)
    {
      eint_ += 0.5 * Core::FADUtils::cast_to_double(Gamma(dim)) *
               Core::FADUtils::cast_to_double(stressN(dim)) * jacobi_gp_elastf_[numgp] * wgt;
    }

    // store material strain and stress values in class variables
    axial_strain_gp_elastf_[numgp] = Core::FADUtils::cast_to_double(Gamma(0));
    shear_strain_2_gp_elastf_[numgp] = Core::FADUtils::cast_to_double(Gamma(1));
    shear_strain_3_gp_elastf_[numgp] = Core::FADUtils::cast_to_double(Gamma(2));

    material_axial_force_gp_elastf_[numgp] = Core::FADUtils::cast_to_double(stressN(0));
    material_shear_force_2_gp_elastf_[numgp] = Core::FADUtils::cast_to_double(stressN(1));
    material_shear_force_3_gp_elastf_[numgp] = Core::FADUtils::cast_to_double(stressN(2));

    spatial_x_force_gp_elastf_[numgp] = Core::FADUtils::cast_to_double(stressn(0));
    spatial_y_force_2_gp_elastf_[numgp] = Core::FADUtils::cast_to_double(stressn(1));
    spatial_z_force_3_gp_elastf_[numgp] = Core::FADUtils::cast_to_double(stressn(2));
  }


  //************************* residual and stiffmatrix contributions from moments
  //***********************

  // get integration points for elasticity
  Core::FE::IntegrationPoints1D gausspoints_elast_moment(my_gauss_rule(res_elastic_moment));

  // reuse variables for individual shape functions and resize to new numgp
  I_i.resize(gausspoints_elast_moment.nquad);
  I_i_xi.resize(gausspoints_elast_moment.nquad);

  // evaluate all shape functions and derivatives with respect to element parameter xi at all
  // specified Gauss points
  Discret::Utils::Beam::evaluate_shape_functions_and_derivs_all_gps<nnodetriad, 1>(
      gausspoints_elast_moment, I_i, I_i_xi, this->shape());

  // reset norm of maximal bending curvature
  kmax_ = 0.0;

  // assure correct size of strain and stress resultant class variables
  twist_gp_elastm_.resize(gausspoints_elast_moment.nquad);
  curvature_2_gp_elastm_.resize(gausspoints_elast_moment.nquad);
  curvature_3_gp_elastm_.resize(gausspoints_elast_moment.nquad);

  material_torque_gp_elastm_.resize(gausspoints_elast_moment.nquad);
  material_bending_moment_2_gp_elastm_.resize(gausspoints_elast_moment.nquad);
  material_bending_moment_3_gp_elastm_.resize(gausspoints_elast_moment.nquad);

  spatial_x_moment_gp_elastm_.resize(gausspoints_elast_moment.nquad);
  spatial_y_moment_2_gp_elastm_.resize(gausspoints_elast_moment.nquad);
  spatial_z_moment_3_gp_elastm_.resize(gausspoints_elast_moment.nquad);


  // Loop through all GP and calculate their contribution to the forcevector and stiffnessmatrix
  for (int numgp = 0; numgp < gausspoints_elast_moment.nquad; numgp++)
  {
    // weight of GP in parameter space
    const double wgt = gausspoints_elast_moment.qwgt[numgp];

    triad_interpolation_scheme_ptr->get_interpolated_local_rotation_vector(Psi_l, I_i[numgp]);

    triad_interpolation_scheme_ptr->get_interpolated_local_rotation_vector_derivative(
        Psi_l_s, I_i_xi[numgp], jacobi_gp_elastm_[numgp]);

    triad_interpolation_scheme_ptr->get_interpolated_triad(Lambda, Psi_l);

    // compute material curvature Cur
    compute_k<T>(Psi_l, Psi_l_s, kref_gp_[numgp], Cur);

    // determine norm of maximal bending curvature at this GP and store in class variable if needed
    double Kmax =
        std::sqrt(Core::FADUtils::cast_to_double(Cur(1)) * Core::FADUtils::cast_to_double(Cur(1)) +
                  Core::FADUtils::cast_to_double(Cur(2)) * Core::FADUtils::cast_to_double(Cur(2)));
    if (Kmax > kmax_) kmax_ = Kmax;

    get_templated_beam_material<T>().evaluate_moment_contributions_to_stress(
        stressM, CM, Cur, numgp);
    get_templated_beam_material<T>().get_stiffness_matrix_of_moments(
        stiffness_contribution, CM, numgp);

    pushforward<T>(Lambda, stressM, stiffness_contribution, stressm, cm);


    /* computation of internal forces according to Jelenic 1999, eq. (4.3); computation split up
     * with respect to single blocks of matrix in eq. (4.3)*/
    for (unsigned int node = 0; node < nnodecl; ++node)
    {
      // lower right block
      for (unsigned int i = 0; i < 3; ++i)
        internal_force(dofpercombinode * node + 3 + i) += I_i_xi[numgp](node) * stressm(i) * wgt;
    }
    for (unsigned int node = nnodecl; node < nnodetriad;
        ++node)  // this loop is only entered in case of nnodetriad>nnodecl
    {
      // lower right block
      for (unsigned int i = 0; i < 3; ++i)
        internal_force(dofperclnode * nnodecl + dofpertriadnode * node + i) +=
            I_i_xi[numgp](node) * stressm(i) * wgt;
    }


    if (stiffmatrix != nullptr)
    {
      calc_stiffmat_analytic_moment_contributions<nnodetriad, nnodecl, vpernode>(*stiffmatrix,
          stressm, cm, *triad_interpolation_scheme_ptr, Psi_l, Psi_l_s, I_i[numgp], I_i_xi[numgp],
          wgt, jacobi_gp_elastm_[numgp]);
    }

    // add elastic energy from moments at this GP
    for (unsigned int dim = 0; dim < 3; dim++)
    {
      eint_ += 0.5 * Core::FADUtils::cast_to_double(Cur(dim)) *
               Core::FADUtils::cast_to_double(stressM(dim)) * jacobi_gp_elastm_[numgp] * wgt;
    }

    // store material strain and stress values in class variables
    twist_gp_elastm_[numgp] = Core::FADUtils::cast_to_double(Cur(0));
    curvature_2_gp_elastm_[numgp] = Core::FADUtils::cast_to_double(Cur(1));
    curvature_3_gp_elastm_[numgp] = Core::FADUtils::cast_to_double(Cur(2));

    material_torque_gp_elastm_[numgp] = Core::FADUtils::cast_to_double(stressM(0));
    material_bending_moment_2_gp_elastm_[numgp] = Core::FADUtils::cast_to_double(stressM(1));
    material_bending_moment_3_gp_elastm_[numgp] = Core::FADUtils::cast_to_double(stressM(2));

    spatial_x_moment_gp_elastm_[numgp] = Core::FADUtils::cast_to_double(stressm(0));
    spatial_y_moment_2_gp_elastm_[numgp] = Core::FADUtils::cast_to_double(stressm(1));
    spatial_z_moment_3_gp_elastm_[numgp] = Core::FADUtils::cast_to_double(stressm(2));
  }
}

template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
void Discret::Elements::Beam3r::calc_inertia_force_and_mass_matrix(
    const Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, double>& disp_totlag_centerline,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>& Qnode,
    Core::LinAlg::SerialDenseMatrix* massmatrix, Core::LinAlg::SerialDenseVector* inertia_force)
{
  const unsigned int dofperclnode = 3 * vpernode;
  const unsigned int dofpertriadnode = 3;
  const unsigned int dofpercombinode = dofperclnode + dofpertriadnode;

  /* Remark:
   * According to the paper of Jelenic and Crisfield "Geometrically exact 3D beam theory:
   * implementation of a strain-invariant finite element for statics and dynamics", 1999,
   * page 146, a time integration scheme that delivers angular velocities and angular
   * accelerations as needed for the inertia terms of geometrically exact beams has to be
   * based on multiplicative rotation angle increments between two successive time steps.
   * Since 4C does all displacement updates in an additive manner, the global vector of
   * rotational displacements has no physical meaning and, consequently the global velocity
   * and acceleration vectors resulting from the 4C time integration schemes have no
   * physical meaning, too. Therefore, a mass matrix in combination with this global
   * acceleration vector is meaningless from a physical point of view. For these reasons, we
   * have to apply our own time integration scheme at element level. Up to now, the only
   * implemented integration scheme is the gen-alpha Lie group time integration according to
   * [Arnold, Bruels (2007)], [Bruels, Cardona, 2010] and [Bruels, Cardona, Arnold (2012)] in
   * combination with a constdisvelacc predictor. (Christoph Meier, 04.14)*/

  /* Update:
   * we now use a multiplicative update of rotational DOFs on time integrator level. Moreover,
   * a new Lie group GenAlpha has been implemented that consistently updates the discrete
   * TRANSLATIONAL velocity and acceleration vectors according to this element-internal scheme.
   * This would allow us to use the global vel and acc vector at least for translational
   * inertia contributions. Nevertheless, we stick to this completely element-internal temporal
   * discretization of spatially continuous variables (angular velocity and acceleration)
   * because the reverse order of discretization (spatial -> temporal) is much more intricate
   * basically because of the triad interpolation. See also the discussion in Christoph Meier's
   * Dissertation on this topic. (Maximilian Grill, 08/16)*/

  const double dt = params_interface().get_delta_time();
  const double beta = params_interface().get_beam_params_interface_ptr()->get_beta();
  const double gamma = params_interface().get_beam_params_interface_ptr()->get_gamma();
  const double alpha_f = params_interface().get_beam_params_interface_ptr()->get_alphaf();
  const double alpha_m = params_interface().get_beam_params_interface_ptr()->get_alpham();

  const bool materialintegration = true;  // TODO unused? remove or realize coverage by test case
  const double diff_factor_vel = gamma / (beta * dt);
  const double diff_factor_acc = (1.0 - alpha_m) / (beta * dt * dt * (1.0 - alpha_f));

  Core::LinAlg::Matrix<3, 3> Lambdanewmass(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> Lambdaconvmass(Core::LinAlg::Initialization::zero);

  // tensor of mass moments of inertia for translational and rotational motion
  double mass_inertia_translational = 0.0;
  Core::LinAlg::Matrix<3, 3> Jp(Core::LinAlg::Initialization::zero);

  get_translational_and_rotational_mass_inertia_tensor(mass_inertia_translational, Jp);

  //********************************** shape functions ************************************
  /* Note: index i refers to the i-th shape function (i = 0 ... nnode*vpernode-1)
   * the vectors store individual shape functions, NOT an assembled matrix of shape functions)*/

  /* vector whose numgp-th element is a 1xnnode-matrix with all Lagrange polynomial shape functions
   * evaluated at the numgp-th Gauss point these shape functions are used for the interpolation of
   * the triad field*/
  std::vector<Core::LinAlg::Matrix<1, nnodetriad, double>> I_i;

  /* vector whose numgp-th element is a 1x(vpernode*nnode)-matrix with all (Lagrange/Hermite) shape
   * functions evaluated at the numgp-th GP
   * these shape functions are used for the interpolation of the beam centerline*/
  std::vector<Core::LinAlg::Matrix<1, vpernode * nnodecl, double>> H_i;

  // get integration scheme for inertia forces and mass matrix
  Core::FE::IntegrationPoints1D gausspoints_mass(my_gauss_rule(res_inertia));
  // reuse variables for individual shape functions and resize to new numgp
  I_i.resize(gausspoints_mass.nquad);
  H_i.resize(gausspoints_mass.nquad);

  // evaluate all shape functions at all specified Gauss points
  Discret::Utils::Beam::evaluate_shape_functions_all_gps<nnodetriad, 1>(
      gausspoints_mass, I_i, this->shape());
  Discret::Utils::Beam::evaluate_shape_functions_all_gps<nnodecl, vpernode>(
      gausspoints_mass, H_i, this->shape(), this->ref_length());

  // Calculate current centerline position at gauss points (needed for element intern time
  // integration)
  for (int gp = 0; gp < gausspoints_mass.nquad; gp++)  // loop through Gauss points
    calc_r<nnodecl, vpernode, double>(disp_totlag_centerline, H_i[gp], rnew_gp_mass_[gp]);

  // create object of triad interpolation scheme
  std::shared_ptr<LargeRotations::TriadInterpolationLocalRotationVectors<nnodetriad, double>>
      triad_interpolation_scheme_ptr = std::make_shared<
          LargeRotations::TriadInterpolationLocalRotationVectors<nnodetriad, double>>();

  // reset triad interpolation scheme with nodal quaternions
  triad_interpolation_scheme_ptr->reset(Qnode);

  ekin_ = 0.0;
  l_ = 0.0;
  p_ = 0.0;

  // interpolated local relative rotation \Psi^l at a certain Gauss point according to (3.11),
  // Jelenic 1999
  Core::LinAlg::Matrix<3, 1, double> Psi_l(Core::LinAlg::Initialization::zero);

  // vector with nnode elements, who represent the 3x3-matrix-shaped interpolation function
  // \tilde{I}^nnode at a certain Gauss point according to (3.18), Jelenic 1999
  std::vector<Core::LinAlg::Matrix<3, 3, double>> Itilde(nnodetriad);

  for (int gp = 0; gp < gausspoints_mass.nquad; gp++)  // loop through Gauss points
  {
    // weight of GP in parameter space
    const double wgtmass = gausspoints_mass.qwgt[gp];

    Core::LinAlg::Matrix<3, 3> Jp_bar(Jp);
    Jp_bar.scale(diff_factor_acc);

    Core::LinAlg::Matrix<3, 1> dL(Core::LinAlg::Initialization::zero);

    triad_interpolation_scheme_ptr->get_interpolated_local_rotation_vector(Psi_l, I_i[gp]);

    triad_interpolation_scheme_ptr->get_interpolated_quaternion(qnew_gp_mass_[gp], Psi_l);

    triad_interpolation_scheme_ptr->get_nodal_generalized_rotation_interpolation_matrices(
        Itilde, Psi_l, I_i[gp]);

    Lambdanewmass.clear();
    Lambdaconvmass.clear();
    // compute current and old triad at Gauss point
    Core::LargeRotations::quaterniontotriad<double>(qnew_gp_mass_[gp], Lambdanewmass);
    Core::LargeRotations::quaterniontotriad<double>(qconv_gp_mass_[gp], Lambdaconvmass);

    // rotation between last converged position and current position expressed as a quaternion
    Core::LinAlg::Matrix<4, 1> deltaQ(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::quaternionproduct(
        Core::LargeRotations::inversequaternion<double>(qconv_gp_mass_[gp]), qnew_gp_mass_[gp],
        deltaQ);

    // spatial rotation between last converged position and current position expressed as a three
    // element rotation vector
    Core::LinAlg::Matrix<3, 1> deltatheta(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::quaterniontoangle<double>(deltaQ, deltatheta);

    // compute material counterparts of spatial vectors
    Core::LinAlg::Matrix<3, 1> deltaTHETA(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> Wconvmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> Wnewmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> Aconvmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> Anewmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> Amodconvmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> Amodnewmass(Core::LinAlg::Initialization::zero);
    deltaTHETA.multiply_tn(Lambdanewmass, deltatheta);
    Wconvmass.multiply_tn(Lambdaconvmass, wconv_gp_mass_[gp]);
    Aconvmass.multiply_tn(Lambdaconvmass, aconv_gp_mass_[gp]);
    Amodconvmass.multiply_tn(Lambdaconvmass, amodconv_gp_mass_[gp]);

    /* update angular velocities and accelerations according to Newmark time integration scheme in
     * material description (see Jelenic, 1999, p. 146, equations (2.8) and (2.9)).
     * The corresponding equations are adapted according to the gen-alpha Lie group time
     * integration scheme proposed in [Arnold, Bruels (2007)], [Bruels, Cardona, 2010] and
     * [Bruels, Cardona, Arnold (2012)].
     * In the predictor step of the time integration the following formulas automatically
     * deliver a constant displacement (deltatheta=0), consistent velocity and consistent
     * acceleration predictor. This fact has to be reflected in a consistent manner by
     * the choice of the predictor in the input file: */
    if (materialintegration)
    {
      for (unsigned int i = 0; i < 3; i++)
      {
        Anewmass(i) = (1.0 - alpha_m) / (beta * dt * dt * (1.0 - alpha_f)) * deltaTHETA(i) -
                      (1.0 - alpha_m) / (beta * dt * (1.0 - alpha_f)) * Wconvmass(i) -
                      alpha_f / (1.0 - alpha_f) * Aconvmass(i) +
                      (alpha_m / (1.0 - alpha_f) -
                          (0.5 - beta) * (1.0 - alpha_m) / (beta * (1.0 - alpha_f))) *
                          Amodconvmass(i);

        Wnewmass(i) = gamma / (beta * dt) * deltaTHETA(i) + (1 - gamma / beta) * Wconvmass(i) +
                      dt * (1 - gamma / (2 * beta)) * Amodconvmass(i);

        Amodnewmass(i) =
            1.0 / (1.0 - alpha_m) *
            ((1.0 - alpha_f) * Anewmass(i) + alpha_f * Aconvmass(i) - alpha_m * Amodconvmass(i));
      }
      wnew_gp_mass_[gp].multiply(Lambdanewmass, Wnewmass);
      anew_gp_mass_[gp].multiply(Lambdanewmass, Anewmass);
      amodnew_gp_mass_[gp].multiply(Lambdanewmass, Amodnewmass);
    }
    else
    {
      for (unsigned int i = 0; i < 3; i++)
      {
        wnew_gp_mass_[gp](i) = gamma / (beta * dt) * deltatheta(i) +
                               (1 - gamma / beta) * wconv_gp_mass_[gp](i) +
                               dt * (1 - gamma / (2 * beta)) * amodconv_gp_mass_[gp](i);

        anew_gp_mass_[gp](i) =
            (1.0 - alpha_m) / (beta * dt * dt * (1.0 - alpha_f)) * deltatheta(i) -
            (1.0 - alpha_m) / (beta * dt * (1.0 - alpha_f)) * wconv_gp_mass_[gp](i) -
            alpha_f / (1.0 - alpha_f) * aconv_gp_mass_[gp](i) +
            (alpha_m / (1.0 - alpha_f) -
                (0.5 - beta) * (1.0 - alpha_m) / (beta * (1.0 - alpha_f))) *
                amodconv_gp_mass_[gp](i);

        amodnew_gp_mass_[gp](i) =
            1.0 / (1.0 - alpha_m) *
            ((1.0 - alpha_f) * anew_gp_mass_[gp](i) + alpha_f * aconv_gp_mass_[gp](i) -
                alpha_m * amodconv_gp_mass_[gp](i));
      }
      Wnewmass.multiply_tn(Lambdanewmass, wnew_gp_mass_[gp]);
      Anewmass.multiply_tn(Lambdanewmass, anew_gp_mass_[gp]);
      Amodnewmass.multiply_tn(Lambdanewmass, amodnew_gp_mass_[gp]);
    }

    Core::LinAlg::Matrix<3, 1> deltar(Core::LinAlg::Initialization::zero);
    for (unsigned int i = 0; i < 3; i++)
    {
      deltar(i) = rnew_gp_mass_[gp](i) - rconv_gp_mass_[gp](i);
    }
    for (unsigned int i = 0; i < 3; i++)
    {
      rttnew_gp_mass_[gp](i) =
          (1.0 - alpha_m) / (beta * dt * dt * (1.0 - alpha_f)) * deltar(i) -
          (1.0 - alpha_m) / (beta * dt * (1.0 - alpha_f)) * rtconv_gp_mass_[gp](i) -
          alpha_f / (1.0 - alpha_f) * rttconv_gp_mass_[gp](i) +
          (alpha_m / (1.0 - alpha_f) - (0.5 - beta) * (1.0 - alpha_m) / (beta * (1.0 - alpha_f))) *
              rttmodconv_gp_mass_[gp](i);

      rtnew_gp_mass_[gp](i) = gamma / (beta * dt) * deltar(i) +
                              (1 - gamma / beta) * rtconv_gp_mass_[gp](i) +
                              dt * (1 - gamma / (2 * beta)) * rttmodconv_gp_mass_[gp](i);

      rttmodnew_gp_mass_[gp](i) =
          1.0 / (1.0 - alpha_m) *
          ((1.0 - alpha_f) * rttnew_gp_mass_[gp](i) + alpha_f * rttconv_gp_mass_[gp](i) -
              alpha_m * rttmodconv_gp_mass_[gp](i));
    }

    // spin matrix of the material angular velocity, i.e. S(W)
    Core::LinAlg::Matrix<3, 3> SWnewmass(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::computespin<double>(SWnewmass, Wnewmass);
    Core::LinAlg::Matrix<3, 1> Jp_Wnewmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> auxvector1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> Pi_t(Core::LinAlg::Initialization::zero);
    Jp_Wnewmass.multiply(Jp, Wnewmass);
    for (unsigned int i = 0; i < 3; i++)
      for (unsigned int j = 0; j < 3; j++)
        auxvector1(i) += SWnewmass(i, j) * Jp_Wnewmass(j) + Jp(i, j) * Anewmass(j);

    Pi_t.multiply(Lambdanewmass, auxvector1);
    Core::LinAlg::Matrix<3, 1> r_tt(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> r_t(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> r(Core::LinAlg::Initialization::zero);

    r_tt = rttnew_gp_mass_[gp];
    r_t = rtnew_gp_mass_[gp];
    r = rnew_gp_mass_[gp];

    Core::LinAlg::Matrix<3, 3> S_r(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::computespin<double>(S_r, r);
    dL.multiply(S_r, r_t);
    dL.scale(mass_inertia_translational);
    Core::LinAlg::Matrix<3, 1> Lambdanewmass_Jp_Wnewmass(Core::LinAlg::Initialization::zero);
    Lambdanewmass_Jp_Wnewmass.multiply(Lambdanewmass, Jp_Wnewmass);
    dL.update(1.0, Lambdanewmass_Jp_Wnewmass, 1.0);
    for (unsigned int i = 0; i < 3; i++)
    {
      l_(i) += wgtmass * jacobi_gp_mass_[gp] * dL(i);
      p_(i) += wgtmass * jacobi_gp_mass_[gp] * mass_inertia_translational * r_t(i);
    }

    Core::LinAlg::Matrix<3, 3> S_Pit(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::computespin<double>(S_Pit, Pi_t);
    Core::LinAlg::Matrix<3, 3> SJpWnewmass(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::computespin<double>(SJpWnewmass, Jp_Wnewmass);
    Core::LinAlg::Matrix<3, 3> SWnewmass_Jp(Core::LinAlg::Initialization::zero);
    SWnewmass_Jp.multiply(SWnewmass, Jp);
    Jp_bar.update(diff_factor_vel, SWnewmass_Jp, 1.0);
    Jp_bar.update(-diff_factor_vel, SJpWnewmass, 1.0);

    Core::LinAlg::Matrix<3, 3> Tmatrix(Core::LinAlg::Initialization::zero);
    Tmatrix = Core::LargeRotations::tmatrix(deltatheta);

    Core::LinAlg::Matrix<3, 3> Lambdanewmass_Jpbar(Core::LinAlg::Initialization::zero);
    Lambdanewmass_Jpbar.multiply(Lambdanewmass, Jp_bar);
    Core::LinAlg::Matrix<3, 3> LambdaconvmassT_Tmatrix(Core::LinAlg::Initialization::zero);
    LambdaconvmassT_Tmatrix.multiply_tn(Lambdaconvmass, Tmatrix);
    Core::LinAlg::Matrix<3, 3> Lambdanewmass_Jpbar_LambdaconvmassT_Tmatrix(
        Core::LinAlg::Initialization::zero);
    Lambdanewmass_Jpbar_LambdaconvmassT_Tmatrix.multiply(
        Lambdanewmass_Jpbar, LambdaconvmassT_Tmatrix);
    Core::LinAlg::Matrix<3, 3> auxmatrix1(Core::LinAlg::Initialization::zero);
    auxmatrix1.update(-1.0, S_Pit, 1.0);
    auxmatrix1.update(1.0, Lambdanewmass_Jpbar_LambdaconvmassT_Tmatrix, 1.0);

    if (inertia_force != nullptr)
    {
      // inertia forces
      for (unsigned int i = 0; i < 3; i++)
      {
        for (unsigned int node = 0; node < nnodecl; node++)
        {
          // translational contribution
          (*inertia_force)(dofpercombinode* node + i) += jacobi_gp_mass_[gp] * wgtmass *
                                                         mass_inertia_translational *
                                                         H_i[gp](vpernode * node) * r_tt(i);
          if (centerline_hermite_)
            (*inertia_force)(dofpercombinode* node + 6 + i) +=
                jacobi_gp_mass_[gp] * wgtmass * mass_inertia_translational *
                H_i[gp](vpernode * node + 1) * r_tt(i);
          // rotational contribution
          (*inertia_force)(dofpercombinode* node + 3 + i) +=
              jacobi_gp_mass_[gp] * wgtmass * I_i[gp](node) * Pi_t(i);
        }
        for (unsigned int node = nnodecl; node < nnodetriad;
            node++)  // this loop is only entered in case of nnodetriad>nnodecl
        {
          // rotational contribution
          (*inertia_force)(dofperclnode* nnodecl + dofpertriadnode * node + i) +=
              jacobi_gp_mass_[gp] * wgtmass * I_i[gp](node) * Pi_t(i);
        }
      }
    }

    if (massmatrix != nullptr)
    {
      // linearization of inertia forces: massmatrix
      for (unsigned int jnode = 0; jnode < nnodecl; jnode++)
      {
        // translational contribution
        for (unsigned int inode = 0; inode < nnodecl; inode++)
          for (unsigned int k = 0; k < 3; k++)
          {
            (*massmatrix)(dofpercombinode* inode + k, dofpercombinode * jnode + k) +=
                diff_factor_acc * jacobi_gp_mass_[gp] * wgtmass * mass_inertia_translational *
                H_i[gp](vpernode * inode) * H_i[gp](vpernode * jnode);
            if (centerline_hermite_)
            {
              (*massmatrix)(dofpercombinode* inode + 6 + k, dofpercombinode * jnode + 6 + k) +=
                  diff_factor_acc * jacobi_gp_mass_[gp] * wgtmass * mass_inertia_translational *
                  H_i[gp](vpernode * inode + 1) * H_i[gp](vpernode * jnode + 1);
              (*massmatrix)(dofpercombinode* inode + k, dofpercombinode * jnode + 6 + k) +=
                  diff_factor_acc * jacobi_gp_mass_[gp] * wgtmass * mass_inertia_translational *
                  H_i[gp](vpernode * inode) * H_i[gp](vpernode * jnode + 1);
              (*massmatrix)(dofpercombinode* inode + 6 + k, dofpercombinode * jnode + k) +=
                  diff_factor_acc * jacobi_gp_mass_[gp] * wgtmass * mass_inertia_translational *
                  H_i[gp](vpernode * inode + 1) * H_i[gp](vpernode * jnode);
            }
          }

        // rotational contribution
        Core::LinAlg::Matrix<3, 3> auxmatrix2(Core::LinAlg::Initialization::zero);
        auxmatrix2.multiply(auxmatrix1, Itilde[jnode]);
        for (unsigned int inode = 0; inode < nnodecl; inode++)
        {
          for (unsigned int i = 0; i < 3; i++)
            for (unsigned int j = 0; j < 3; j++)
              (*massmatrix)(dofpercombinode* inode + 3 + i, dofpercombinode * jnode + 3 + j) +=
                  jacobi_gp_mass_[gp] * wgtmass * I_i[gp](inode) * auxmatrix2(i, j);
        }
        for (unsigned int inode = nnodecl; inode < nnodetriad;
            inode++)  // this loop is only entered in case of nnodetriad>nnodecl
        {
          for (unsigned int i = 0; i < 3; i++)
            for (unsigned int j = 0; j < 3; j++)
              (*massmatrix)(dofperclnode* nnodecl + dofpertriadnode * inode + i,
                  dofpercombinode * jnode + 3 + j) +=
                  jacobi_gp_mass_[gp] * wgtmass * I_i[gp](inode) * auxmatrix2(i, j);
        }
      }
      for (unsigned int jnode = nnodecl; jnode < nnodetriad;
          ++jnode)  // this loop is only entered in case of nnodetriad>nnodecl
      {
        // rotational contribution
        Core::LinAlg::Matrix<3, 3> auxmatrix2(Core::LinAlg::Initialization::zero);
        auxmatrix2.multiply(auxmatrix1, Itilde[jnode]);
        for (unsigned int inode = 0; inode < nnodecl; inode++)
        {
          for (unsigned int i = 0; i < 3; i++)
            for (unsigned int j = 0; j < 3; j++)
              (*massmatrix)(dofpercombinode* inode + 3 + i,
                  dofperclnode * nnodecl + dofpertriadnode * jnode + j) +=
                  jacobi_gp_mass_[gp] * wgtmass * I_i[gp](inode) * auxmatrix2(i, j);
        }
        for (unsigned int inode = nnodecl; inode < nnodetriad;
            inode++)  // this loop is only entered in case of nnodetriad>nnodecl
        {
          for (unsigned int i = 0; i < 3; i++)
            for (unsigned int j = 0; j < 3; j++)
              (*massmatrix)(dofperclnode* nnodecl + dofpertriadnode * inode + i,
                  dofperclnode * nnodecl + dofpertriadnode * jnode + j) +=
                  jacobi_gp_mass_[gp] * wgtmass * I_i[gp](inode) * auxmatrix2(i, j);
        }
      }
    }

    // Calculation of kinetic energy
    Core::LinAlg::Matrix<1, 1> ekinrot(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<1, 1> ekintrans(Core::LinAlg::Initialization::zero);
    ekinrot.multiply_tn(Wnewmass, Jp_Wnewmass);
    ekintrans.multiply_tn(r_t, r_t);
    ekin_ += 0.5 * (ekinrot.norm2() + mass_inertia_translational * ekintrans.norm2()) *
             jacobi_gp_mass_[gp] * wgtmass;
    ekintorsion_ += 0.5 * Wnewmass(0) * Jp_Wnewmass(0) * jacobi_gp_mass_[gp] * wgtmass;
    ekinbending_ += 0.5 * Wnewmass(1) * Jp_Wnewmass(1) * jacobi_gp_mass_[gp] * wgtmass;
    ekinbending_ += 0.5 * Wnewmass(2) * Jp_Wnewmass(2) * jacobi_gp_mass_[gp] * wgtmass;
    ekintrans_ +=
        0.5 * mass_inertia_translational * ekintrans.norm2() * jacobi_gp_mass_[gp] * wgtmass;

    Jp_Wnewmass.multiply(Jp, Wnewmass);
  }

  // In Lie group GenAlpha algorithm, the mass matrix is multiplied with factor
  // (1.0-alpham_)/(beta_*dt*dt*(1.0-alphaf_)) later. so we apply inverse factor here because the
  // correct prefactors for displacement/velocity/acceleration dependent terms have been applied
  // individually above
  if (massmatrix != nullptr) massmatrix->scale(beta * dt * dt * (1.0 - alpha_f) / (1.0 - alpha_m));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
void Discret::Elements::Beam3r::calc_stiffmat_analytic_force_contributions(
    Core::LinAlg::SerialDenseMatrix& stiffmatrix, const Core::LinAlg::Matrix<3, 1, double>& stressn,
    const Core::LinAlg::Matrix<3, 3, double>& cn, const Core::LinAlg::Matrix<3, 3, double>& r_s_hat,
    const LargeRotations::TriadInterpolationLocalRotationVectors<nnodetriad, double>& triad_intpol,
    const Core::LinAlg::Matrix<1, nnodetriad, double>& I_i,
    const Core::LinAlg::Matrix<1, vpernode * nnodecl, double>& H_i_xi, const double wgt,
    const double jacobifactor) const
{
  const unsigned int dofperclnode = 3 * vpernode;
  const unsigned int dofpertriadnode = 3;
  const unsigned int dofpercombinode = dofperclnode + dofpertriadnode;

  /* computation of stiffness matrix according to Jelenic 1999, eq. (4.7); computation split up with
   * respect to single blocks of matrix in eq. (4.7). note: again, jacobi factor cancels out in
   * terms with I^{i'}=I_i_s=I_i_xi*(dxi/ds) (see comment above) but be careful: Itildeprime and
   * rprime are indeed derivatives with respect to arc-length parameter in reference configuration s
   */

  // vector with nnode elements, who represent the 3x3-matrix-shaped interpolation function
  // \tilde{I}^nnode at a certain Gauss point according to (3.18), Jelenic 1999
  std::vector<Core::LinAlg::Matrix<3, 3, double>> Itilde(nnodetriad);

  Core::LinAlg::Matrix<3, 1, double> Psi_l(Core::LinAlg::Initialization::zero);
  triad_intpol.get_interpolated_local_rotation_vector(Psi_l, I_i);
  triad_intpol.get_nodal_generalized_rotation_interpolation_matrices(Itilde, Psi_l, I_i);


  // auxiliary variables for storing intermediate matrices in computation of entries of stiffness
  // matrix
  Core::LinAlg::Matrix<3, 3, double> auxmatrix1;
  Core::LinAlg::Matrix<3, 3, double> auxmatrix2;
  Core::LinAlg::Matrix<3, 3, double> auxmatrix3;

  for (unsigned int nodei = 0; nodei < nnodecl; nodei++)
  {
    for (unsigned int nodej = 0; nodej < nnodecl; nodej++)
    {
      // upper left block
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
        {
          stiffmatrix(dofpercombinode * nodei + i, dofpercombinode * nodej + j) +=
              H_i_xi(vpernode * nodei) * H_i_xi(vpernode * nodej) * cn(i, j) * wgt / jacobifactor;
          if (centerline_hermite_)
          {
            stiffmatrix(dofpercombinode * nodei + 6 + i, dofpercombinode * nodej + j) +=
                H_i_xi(vpernode * nodei + 1) * H_i_xi(vpernode * nodej) * cn(i, j) * wgt /
                jacobifactor;
            stiffmatrix(dofpercombinode * nodei + i, dofpercombinode * nodej + 6 + j) +=
                H_i_xi(vpernode * nodei) * H_i_xi(vpernode * nodej + 1) * cn(i, j) * wgt /
                jacobifactor;
            stiffmatrix(dofpercombinode * nodei + 6 + i, dofpercombinode * nodej + 6 + j) +=
                H_i_xi(vpernode * nodei + 1) * H_i_xi(vpernode * nodej + 1) * cn(i, j) * wgt /
                jacobifactor;
          }
        }

      // lower left block; note: error in eq. (4.7), Jelenic 1999: the first factor should be I^i
      // instead of I^j
      auxmatrix2.multiply(r_s_hat, cn);
      Core::LargeRotations::computespin(auxmatrix1, stressn);
      auxmatrix1 -= auxmatrix2;
      auxmatrix1.scale(I_i(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
        {
          stiffmatrix(dofpercombinode * nodei + 3 + i, dofpercombinode * nodej + j) +=
              auxmatrix1(i, j) * H_i_xi(vpernode * nodej) * wgt;
          if (centerline_hermite_)
            stiffmatrix(dofpercombinode * nodei + 3 + i, dofpercombinode * nodej + 6 + j) +=
                auxmatrix1(i, j) * H_i_xi(vpernode * nodej + 1) * wgt;
        }

      // upper right block
      auxmatrix2.multiply(cn, r_s_hat);
      Core::LargeRotations::computespin(auxmatrix1, stressn);
      auxmatrix2 -= auxmatrix1;  // auxmatrix2: term in parentheses

      auxmatrix3.multiply(auxmatrix2, Itilde[nodej]);
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
        {
          stiffmatrix(dofpercombinode * nodei + i, dofpercombinode * nodej + 3 + j) +=
              auxmatrix3(i, j) * H_i_xi(vpernode * nodei) * wgt;
          if (centerline_hermite_)
            stiffmatrix(dofpercombinode * nodei + 6 + i, dofpercombinode * nodej + 3 + j) +=
                auxmatrix3(i, j) * H_i_xi(vpernode * nodei + 1) * wgt;
        }

      // lower right block
      // third summand; note: error in eq. (4.7), Jelenic 1999: the first summand in the parentheses
      // should be \hat{\Lambda N} instead of \Lambda N
      auxmatrix1.multiply(
          auxmatrix2, Itilde[nodej]);  // term in parentheses is the same as in upper right block
                                       // but with opposite sign (note '-=' below)

      auxmatrix3.multiply(r_s_hat, auxmatrix1);
      auxmatrix3.scale(I_i(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * nodei + 3 + i, dofpercombinode * nodej + 3 + j) -=
              auxmatrix3(i, j) * jacobifactor * wgt;
    }
    for (unsigned int nodej = nnodecl; nodej < nnodetriad;
        nodej++)  // this loop is only entered in case of nnodetriad>nnodecl
    {
      // upper right block
      auxmatrix2.multiply(cn, r_s_hat);
      Core::LargeRotations::computespin(auxmatrix1, stressn);
      auxmatrix2 -= auxmatrix1;  // auxmatrix2: term in parentheses

      auxmatrix3.multiply(auxmatrix2, Itilde[nodej]);
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
        {
          stiffmatrix(
              dofpercombinode * nodei + i, dofperclnode * nnodecl + dofpertriadnode * nodej + j) +=
              auxmatrix3(i, j) * H_i_xi(vpernode * nodei) * wgt;
          if (centerline_hermite_)
            stiffmatrix(dofpercombinode * nodei + 6 + i,
                dofperclnode * nnodecl + dofpertriadnode * nodej + j) +=
                auxmatrix3(i, j) * H_i_xi(vpernode * nodei + 1) * wgt;
        }

      // lower right block
      // third summand; note: error in eq. (4.7), Jelenic 1999: the first summand in the parentheses
      // should be \hat{\Lambda N} instead of \Lambda N
      auxmatrix1.multiply(
          auxmatrix2, Itilde[nodej]);  // term in parentheses is the same as in upper right block
                                       // but with opposite sign (note '-=' below)

      auxmatrix3.multiply(r_s_hat, auxmatrix1);
      auxmatrix3.scale(I_i(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * nodei + 3 + i,
              dofperclnode * nnodecl + dofpertriadnode * nodej + j) -=
              auxmatrix3(i, j) * jacobifactor * wgt;
    }
  }
  for (unsigned int nodei = nnodecl; nodei < nnodetriad;
      nodei++)  // this loop is only entered in case of nnodetriad>nnodecl
  {
    for (unsigned int nodej = 0; nodej < nnodecl; nodej++)
    {
      // lower left block; note: error in eq. (4.7), Jelenic 1999: the first factor should be I^i
      // instead of I^j
      auxmatrix2.multiply(r_s_hat, cn);
      Core::LargeRotations::computespin(auxmatrix1, stressn);
      auxmatrix1 -= auxmatrix2;
      auxmatrix1.scale(I_i(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
        {
          stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * nodei + i,
              dofpercombinode * nodej + j) += auxmatrix1(i, j) * H_i_xi(vpernode * nodej) * wgt;
          if (centerline_hermite_)
            stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * nodei + i,
                dofpercombinode * nodej + 6 + j) +=
                auxmatrix1(i, j) * H_i_xi(vpernode * nodej + 1) * wgt;
        }

      // lower right block
      // third summand; note: error in eq. (4.7), Jelenic 1999: the first summand in the parentheses
      // should be \hat{\Lambda N} instead of \Lambda N
      auxmatrix2.multiply(cn, r_s_hat);
      Core::LargeRotations::computespin(auxmatrix1, stressn);
      auxmatrix2 -= auxmatrix1;  // auxmatrix2: term in parentheses

      auxmatrix1.multiply(
          auxmatrix2, Itilde[nodej]);  // term in parentheses is the same as in upper right block
                                       // but with opposite sign (note '-=' below)

      auxmatrix3.multiply(r_s_hat, auxmatrix1);
      auxmatrix3.scale(I_i(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * nodei + i,
              dofpercombinode * nodej + 3 + j) -= auxmatrix3(i, j) * jacobifactor * wgt;
    }
    for (unsigned int nodej = nnodecl; nodej < nnodetriad; nodej++)
    {
      // lower right block
      // third summand; note: error in eq. (4.7), Jelenic 1999: the first summand in the parentheses
      // should be \hat{\Lambda N} instead of \Lambda N
      auxmatrix2.multiply(cn, r_s_hat);
      Core::LargeRotations::computespin(auxmatrix1, stressn);
      auxmatrix2 -= auxmatrix1;  // auxmatrix2: term in parentheses

      auxmatrix1.multiply(
          auxmatrix2, Itilde[nodej]);  // term in parentheses is the same as in upper right block
                                       // but with opposite sign (note '-=' below)

      auxmatrix3.multiply(r_s_hat, auxmatrix1);
      auxmatrix3.scale(I_i(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * nodei + i,
              dofperclnode * nnodecl + dofpertriadnode * nodej + j) -=
              auxmatrix3(i, j) * jacobifactor * wgt;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
void Discret::Elements::Beam3r::calc_stiffmat_analytic_moment_contributions(
    Core::LinAlg::SerialDenseMatrix& stiffmatrix, const Core::LinAlg::Matrix<3, 1, double>& stressm,
    const Core::LinAlg::Matrix<3, 3, double>& cm,
    const LargeRotations::TriadInterpolationLocalRotationVectors<nnodetriad, double>& triad_intpol,
    const Core::LinAlg::Matrix<3, 1, double>& Psi_l,
    const Core::LinAlg::Matrix<3, 1, double>& Psi_l_s,
    const Core::LinAlg::Matrix<1, nnodetriad, double>& I_i,
    const Core::LinAlg::Matrix<1, nnodetriad, double>& I_i_xi, const double wgt,
    const double jacobifactor) const
{
  const unsigned int dofperclnode = 3 * vpernode;
  const unsigned int dofpertriadnode = 3;
  const unsigned int dofpercombinode = dofperclnode + dofpertriadnode;

  /* computation of stiffness matrix according to Jelenic 1999, eq. (4.7)*/

  // vector with nnode elements, who represent the 3x3-matrix-shaped interpolation function
  // \tilde{I}^nnode at a certain Gauss point according to (3.18), Jelenic 1999
  std::vector<Core::LinAlg::Matrix<3, 3, double>> Itilde(nnodetriad);

  // vector with nnode elements, who represent the 3x3-matrix-shaped interpolation function
  // \tilde{I'}^nnode at a certain Gauss point according to (3.19), Jelenic 1999
  std::vector<Core::LinAlg::Matrix<3, 3, double>> Itildeprime(nnodetriad);

  triad_intpol.get_nodal_generalized_rotation_interpolation_matrices(Itilde, Psi_l, I_i);

  triad_intpol.get_nodal_generalized_rotation_interpolation_matrices_derivative(
      Itildeprime, Psi_l, Psi_l_s, I_i, I_i_xi, jacobifactor);


  // auxiliary variables for storing intermediate matrices in computation of entries of stiffness
  // matrix
  Core::LinAlg::Matrix<3, 3, double> auxmatrix1;
  Core::LinAlg::Matrix<3, 3, double> auxmatrix2;

  for (unsigned int nodei = 0; nodei < nnodecl; nodei++)
  {
    for (unsigned int nodej = 0; nodej < nnodecl; nodej++)
    {
      // lower right block
      // first summand
      auxmatrix1.multiply(cm, Itildeprime[nodej]);
      auxmatrix1.scale(I_i_xi(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * nodei + 3 + i, dofpercombinode * nodej + 3 + j) +=
              auxmatrix1(i, j) * wgt;

      // second summand
      Core::LargeRotations::computespin(auxmatrix2, stressm);
      auxmatrix1.multiply(auxmatrix2, Itilde[nodej]);
      auxmatrix1.scale(I_i_xi(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * nodei + 3 + i, dofpercombinode * nodej + 3 + j) -=
              auxmatrix1(i, j) * wgt;
    }
    for (unsigned int nodej = nnodecl; nodej < nnodetriad;
        nodej++)  // this loop is only entered in case of nnodetriad>nnodecl
    {
      // lower right block
      // first summand
      auxmatrix1.multiply(cm, Itildeprime[nodej]);
      auxmatrix1.scale(I_i_xi(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * nodei + 3 + i,
              dofperclnode * nnodecl + dofpertriadnode * nodej + j) += auxmatrix1(i, j) * wgt;

      // second summand
      Core::LargeRotations::computespin(auxmatrix2, stressm);
      auxmatrix1.multiply(auxmatrix2, Itilde[nodej]);
      auxmatrix1.scale(I_i_xi(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * nodei + 3 + i,
              dofperclnode * nnodecl + dofpertriadnode * nodej + j) -= auxmatrix1(i, j) * wgt;
    }
  }

  for (unsigned int nodei = nnodecl; nodei < nnodetriad;
      nodei++)  // this loop is only entered in case of nnodetriad>nnodecl
  {
    for (unsigned int nodej = 0; nodej < nnodecl; nodej++)
    {
      // lower right block
      // first summand
      auxmatrix1.multiply(cm, Itildeprime[nodej]);
      auxmatrix1.scale(I_i_xi(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * nodei + i,
              dofpercombinode * nodej + 3 + j) += auxmatrix1(i, j) * wgt;

      // second summand
      Core::LargeRotations::computespin(auxmatrix2, stressm);
      auxmatrix1.multiply(auxmatrix2, Itilde[nodej]);
      auxmatrix1.scale(I_i_xi(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * nodei + i,
              dofpercombinode * nodej + 3 + j) -= auxmatrix1(i, j) * wgt;
    }
    for (unsigned int nodej = nnodecl; nodej < nnodetriad; nodej++)
    {
      // lower right block
      // first summand
      auxmatrix1.multiply(cm, Itildeprime[nodej]);
      auxmatrix1.scale(I_i_xi(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * nodei + i,
              dofperclnode * nnodecl + dofpertriadnode * nodej + j) += auxmatrix1(i, j) * wgt;

      // second summand
      Core::LargeRotations::computespin(auxmatrix2, stressm);
      auxmatrix1.multiply(auxmatrix2, Itilde[nodej]);
      auxmatrix1.scale(I_i_xi(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * nodei + i,
              dofperclnode * nnodecl + dofpertriadnode * nodej + j) -= auxmatrix1(i, j) * wgt;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
void Discret::Elements::Beam3r::calc_stiffmat_automatic_differentiation(
    Core::LinAlg::SerialDenseMatrix& stiffmatrix,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>& Qnode,
    Core::LinAlg::Matrix<3 * vpernode * nnodecl + 3 * nnodetriad, 1, Sacado::Fad::DFad<double>>
        forcevec) const
{
  const unsigned int dofperclnode = 3 * vpernode;
  const unsigned int dofpertriadnode = 3;
  const unsigned int dofpercombinode = dofperclnode + dofpertriadnode;

  // compute stiffness matrix with FAD
  for (unsigned int i = 0; i < dofperclnode * nnodecl + dofpertriadnode * nnodetriad; i++)
  {
    for (unsigned int j = 0; j < dofperclnode * nnodecl + dofpertriadnode * nnodetriad; j++)
    {
      stiffmatrix(i, j) = forcevec(i).dx(j);
    }
  }

  /* we need to transform the stiffmatrix because its entries are derivatives with respect to
   * additive rotational increments we want a stiffmatrix containing derivatives with respect to
   * multiplicative rotational increments therefore apply a trafo matrix to all those 3x3 blocks in
   * stiffmatrix which correspond to derivation with respect to rotational DOFs
   * the trafo matrix is simply the T-Matrix (see Jelenic1999, (2.4)): \Delta_{mult} \vec
   * \theta_{inode} = \mat T(\vec \theta_{inode} * \Delta_{addit} \vec \theta_{inode}*/

  Core::LinAlg::Matrix<3, 3, double> tempmat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> newstiffmat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> Tmat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, double> theta_totlag_j(Core::LinAlg::Initialization::zero);

  for (unsigned int jnode = 0; jnode < nnodecl; jnode++)
  {
    // compute physical total angle theta_totlag
    Core::LargeRotations::quaterniontoangle(Qnode[jnode], theta_totlag_j);

    // compute Tmatrix of theta_totlag_i
    Tmat = Core::LargeRotations::tmatrix(theta_totlag_j);

    for (unsigned int inode = 0; inode < nnodecl; inode++)
    {
      // block1: derivative of nodal positions with respect to theta (rotational DOFs)
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          tempmat(i, j) = stiffmatrix(dofpercombinode * inode + i, dofpercombinode * jnode + 3 + j);

      newstiffmat.clear();
      newstiffmat.multiply_nn(tempmat, Tmat);

      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * inode + i, dofpercombinode * jnode + 3 + j) =
              newstiffmat(i, j);

      // block2: derivative of nodal theta with respect to theta (rotational DOFs)
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          tempmat(i, j) =
              stiffmatrix(dofpercombinode * inode + 3 + i, dofpercombinode * jnode + 3 + j);

      newstiffmat.clear();
      newstiffmat.multiply_nn(tempmat, Tmat);

      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * inode + 3 + i, dofpercombinode * jnode + 3 + j) =
              newstiffmat(i, j);

      // block3: derivative of nodal tangents with respect to theta (rotational DOFs)
      if (centerline_hermite_)
      {
        for (unsigned int i = 0; i < 3; ++i)
          for (unsigned int j = 0; j < 3; ++j)
            tempmat(i, j) =
                stiffmatrix(dofpercombinode * inode + 6 + i, dofpercombinode * jnode + 3 + j);

        newstiffmat.clear();
        newstiffmat.multiply_nn(tempmat, Tmat);

        for (unsigned int i = 0; i < 3; ++i)
          for (unsigned int j = 0; j < 3; ++j)
            stiffmatrix(dofpercombinode * inode + 6 + i, dofpercombinode * jnode + 3 + j) =
                newstiffmat(i, j);
      }
    }
    for (unsigned int inode = nnodecl; inode < nnodetriad;
        inode++)  // this loop is only entered in case of nnodetriad>nnodecl
    {
      // block2: derivative of nodal theta with respect to theta (rotational DOFs)
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          tempmat(i, j) = stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * inode + i,
              dofpercombinode * jnode + 3 + j);

      newstiffmat.clear();
      newstiffmat.multiply_nn(tempmat, Tmat);

      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * inode + i,
              dofpercombinode * jnode + 3 + j) = newstiffmat(i, j);
    }
  }

  for (unsigned int jnode = nnodecl; jnode < nnodetriad;
      jnode++)  // this loop is only entered in case of nnodetriad>nnodecl
  {
    // compute physical total angle theta_totlag
    Core::LargeRotations::quaterniontoangle(Qnode[jnode], theta_totlag_j);

    // compute Tmatrix of theta_totlag_i
    Tmat = Core::LargeRotations::tmatrix(theta_totlag_j);

    for (unsigned int inode = 0; inode < nnodecl; inode++)
    {
      // block1: derivative of nodal positions with respect to theta (rotational DOFs)
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          tempmat(i, j) = stiffmatrix(
              dofpercombinode * inode + i, dofperclnode * nnodecl + dofpertriadnode * jnode + j);

      newstiffmat.clear();
      newstiffmat.multiply_nn(tempmat, Tmat);

      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * inode + i,
              dofperclnode * nnodecl + dofpertriadnode * jnode + j) = newstiffmat(i, j);

      // block2: derivative of nodal theta with respect to theta (rotational DOFs)
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          tempmat(i, j) = stiffmatrix(dofpercombinode * inode + 3 + i,
              dofperclnode * nnodecl + dofpertriadnode * jnode + j);

      newstiffmat.clear();
      newstiffmat.multiply_nn(tempmat, Tmat);

      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * inode + 3 + i,
              dofperclnode * nnodecl + dofpertriadnode * jnode + j) = newstiffmat(i, j);

      // block3: derivative of nodal tangents with respect to theta (rotational DOFs)
      if (centerline_hermite_)
      {
        for (unsigned int i = 0; i < 3; ++i)
          for (unsigned int j = 0; j < 3; ++j)
            tempmat(i, j) = stiffmatrix(dofpercombinode * inode + 6 + i,
                dofperclnode * nnodecl + dofpertriadnode * jnode + j);

        newstiffmat.clear();
        newstiffmat.multiply_nn(tempmat, Tmat);

        for (unsigned int i = 0; i < 3; ++i)
          for (unsigned int j = 0; j < 3; ++j)
            stiffmatrix(dofpercombinode * inode + 6 + i,
                dofperclnode * nnodecl + dofpertriadnode * jnode + j) = newstiffmat(i, j);
      }
    }
    for (unsigned int inode = nnodecl; inode < nnodetriad; inode++)
    {
      // block2: derivative of nodal theta with respect to theta (rotational DOFs)
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          tempmat(i, j) = stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * inode + i,
              dofperclnode * nnodecl + dofpertriadnode * jnode + j);

      newstiffmat.clear();
      newstiffmat.multiply_nn(tempmat, Tmat);

      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * inode + i,
              dofperclnode * nnodecl + dofpertriadnode * jnode + j) = newstiffmat(i, j);
    }
  }
}

/*------------------------------------------------------------------------------------------------------------*
 | calculation of thermal (i.e. stochastic) and damping forces according to Brownian dynamics grill
 06/16|
 *------------------------------------------------------------------------------------------------------------*/
template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
void Discret::Elements::Beam3r::calc_brownian_forces_and_stiff(Teuchos::ParameterList& params,
    std::vector<double>& vel, std::vector<double>& disp,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseVector* force)
{
  // nnodetriad: number of nodes used for interpolation of triad field
  // nnodecl: number of nodes used for interpolation of centerline
  // assumptions: nnodecl<=nnodetriad; centerline nodes have local ID 0...nnodecl-1
  // vpernode: number of interpolated values per centerline node (1: value (i.e. Lagrange), 2: value
  // + derivative of value (i.e. Hermite))

  //********************************* statmech periodic boundary conditions
  //****************************

  // unshift node positions, i.e. manipulate element displacement vector
  // as if there where no periodic boundary conditions
  if (brownian_dyn_params_interface_ptr() != nullptr)
    un_shift_node_position(disp, *brownian_dyn_params_interface().get_periodic_bounding_box());

  /****** update/compute key variables describing displacement and velocity state of this element
   * *****/

  // current nodal DOFs relevant for centerline interpolation in total Lagrangian style, i.e.
  // initial values + displacements
  Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, double> disp_totlag_centerline(
      Core::LinAlg::Initialization::zero);

  // discrete centerline (i.e. translational) velocity vector
  Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, double> vel_centerline(
      Core::LinAlg::Initialization::zero);

  // quaternions of all nodal triads
  std::vector<Core::LinAlg::Matrix<4, 1, double>> Q_i(nnodetriad);

  // update disp_totlag_centerline and nodal triads
  update_disp_tot_lag_and_nodal_triads<nnodetriad, nnodecl, vpernode, double>(
      disp, disp_totlag_centerline, Q_i);

  // update current values of centerline (i.e. translational) velocity
  extract_centerline_dof_values_from_element_state_vector<nnodecl, vpernode, double>(
      vel, vel_centerline);

  /****** compute and assemble force and stiffness contributions from viscous damping and stochastic
   * forces *****/

  // add stiffness and forces (i.e. moments) due to rotational damping effects
  evaluate_rotational_damping<nnodetriad, nnodecl, vpernode, 3>(params, Q_i, stiffmatrix, force);

  if (stiffmatrix != nullptr) stiff_ptc_ = *stiffmatrix;

  // add stiffness and forces due to translational damping effects
  evaluate_translational_damping<nnodecl, vpernode, 3>(
      params, vel_centerline, disp_totlag_centerline, stiffmatrix, force);


  // add stochastic forces and (if required) resulting stiffness
  evaluate_stochastic_forces<nnodecl, vpernode, 3, 3>(
      params, disp_totlag_centerline, stiffmatrix, force);
}

void Discret::Elements::Beam3r::calc_stiff_contributions_ptc(
    Core::LinAlg::SerialDenseMatrix& elemat1)
{
  elemat1 = stiff_ptc_;
}

/*------------------------------------------------------------------------------------------------------------*
 | lump mass matrix             (private)                                                   cyron
 01/08|
 *------------------------------------------------------------------------------------------------------------*/
template <unsigned int nnode>
void Discret::Elements::Beam3r::lumpmass(Core::LinAlg::SerialDenseMatrix* massmatrix)
{
  // lump mass matrix
  if (massmatrix != nullptr)
  {
    // we assume #elemat2 is a square matrix
    for (int c = 0; c < (*massmatrix).numCols(); ++c)  // parse columns
    {
      double d = 0.0;
      for (int r = 0; r < (*massmatrix).numRows(); ++r)  // parse rows
      {
        d += (*massmatrix)(r, c);  // accumulate row entries
        (*massmatrix)(r, c) = 0.0;
      }

      (*massmatrix)(c, c) = d;  // apply sum of row entries on diagonal
    }
  }
}

/*-----------------------------------------------------------------------------------------------------------*
 | Evaluate PTC damping (public) cyron 10/08|
 *----------------------------------------------------------------------------------------------------------*/
template <unsigned int nnode>
void Discret::Elements::Beam3r::evaluate_ptc(
    Teuchos::ParameterList& params, Core::LinAlg::SerialDenseMatrix& elemat1)
{
  // apply PTC rotation damping term using a Lobatto integration rule; implemented for 2 nodes only
  if (nnode > 2 or centerline_hermite_)
    FOUR_C_THROW(
        "PTC was originally implemented for 2-noded Reissner beam element only. Check "
        "functionality for "
        "numnodes>2 and/or Hermite interpolation and extend if needed!");

  for (unsigned int node = 0; node < nnode; node++)
  {
    // computing angle increment from current position in comparison with last converged position
    // for damping
    Core::LinAlg::Matrix<4, 1> deltaQ;
    Core::LargeRotations::quaternionproduct(
        Core::LargeRotations::inversequaternion(qconvnode_[node]), qnewnode_[node], deltaQ);
    Core::LinAlg::Matrix<3, 1> deltatheta;
    Core::LargeRotations::quaterniontoangle(deltaQ, deltatheta);

    // isotropic artificial stiffness
    Core::LinAlg::Matrix<3, 3> artstiff;
    artstiff = Core::LargeRotations::tmatrix(deltatheta);

    // scale artificial damping with crotptc parameter for PTC method
    artstiff.scale(params.get<double>("crotptc", 0.0));

    // each node gets a block diagonal damping term; the Lobatto integration weight is 0.5 for
    // 2-noded elements jacobi determinant is constant and equals 0.5*refelelength for 2-noded
    // elements
    for (int k = 0; k < 3; k++)
      for (int l = 0; l < 3; l++)
        elemat1(node * 6 + 3 + k, node * 6 + 3 + l) += artstiff(k, l) * 0.5 * 0.5 * reflength_;

    // PTC for translational degrees of freedom; the Lobatto integration weight is 0.5 for 2-noded
    // elements
    for (int k = 0; k < 3; k++)
      elemat1(node * 6 + k, node * 6 + k) +=
          params.get<double>("ctransptc", 0.0) * 0.5 * 0.5 * reflength_;
  }

  return;
}

/*-----------------------------------------------------------------------------------------------------------*
 |computes the number of different random numbers required in each time step for generation of
 stochastic    | |forces; (public)           cyron   10/09|
 *----------------------------------------------------------------------------------------------------------*/
int Discret::Elements::Beam3r::how_many_random_numbers_i_need() const
{
  // get Gauss rule for evaluation of stochastic force contributions
  Core::FE::GaussRule1D gaussrule = my_gauss_rule(res_damp_stoch);
  Core::FE::IntegrationPoints1D gausspoints(gaussrule);

  /* at each Gauss point one needs as many random numbers as randomly excited degrees of freedom,
   * i.e. three random numbers for the translational degrees of freedom */
#ifndef BEAM3RCONSTSTOCHFORCE
  return (3 * gausspoints.nquad);
#else
  return (3);
#endif
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode, unsigned int ndim>
void Discret::Elements::Beam3r::evaluate_rotational_damping(
    Teuchos::ParameterList& params,  //!< parameter list
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>& Qnode,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix,  //!< element stiffness matrix
    Core::LinAlg::SerialDenseVector* force)        //!< element internal force vector
{
  const unsigned int dofperclnode = 3 * vpernode;
  const unsigned int dofpertriadnode = 3;
  const unsigned int dofpercombinode = dofperclnode + dofpertriadnode;

  // get time step size
  double dt_inv = 0.0001;
  if (is_params_interface())
    dt_inv = 1.0 / params_interface().get_delta_time();
  else
    dt_inv = 1.0 / params.get<double>("delta time", 1000);

  // get damping coefficients for translational and rotational degrees of freedom
  Core::LinAlg::Matrix<3, 1> gamma(Core::LinAlg::Initialization::zero);
  get_damping_coefficients(gamma);

  // get Gauss points and weights for evaluation of viscous damping contributions
  Core::FE::GaussRule1D gaussrule = my_gauss_rule(res_damp_stoch);
  Core::FE::IntegrationPoints1D gausspoints(gaussrule);

  //*************************** physical quantities evaluated at a certain GP
  //***************************

  // interpolated local relative rotation \Psi^l at a certain Gauss point according to (3.11),
  // Jelenic 1999
  Core::LinAlg::Matrix<3, 1> Psi_l;

  // material triad and corresponding quaternion at a certain Gauss point
  Core::LinAlg::Matrix<3, 3> LambdaGP;
  Core::LinAlg::Matrix<4, 1> QnewGP;

  //********************************** (generalized) shape functions
  //************************************
  /* Note: index i refers to the i-th shape function (i = 0 ... nnodetriad-1)
   * the vectors store individual shape functions, NOT an assembled matrix of shape functions)*/

  /* vector whose numgp-th element is a 1xnnodetriad-matrix with all Lagrange polynomial shape
   * functions evaluated at the numgp-th Gauss point these shape functions are used for the
   * interpolation of the triad field */
  std::vector<Core::LinAlg::Matrix<1, nnodetriad, double>> I_i(gausspoints.nquad);

  // evaluate all shape functions at all specified Gauss points
  Discret::Utils::Beam::evaluate_shape_functions_all_gps<nnodetriad, 1>(
      gausspoints, I_i, this->shape());

  /* vector with nnodetriad elements, who represent the 3x3-matrix-shaped interpolation function
   * \tilde{I}^nnode according to (3.19), Jelenic 1999*/
  std::vector<Core::LinAlg::Matrix<3, 3, double>> Itilde(nnodetriad);


  // create an object of the triad interpolation scheme
  std::shared_ptr<LargeRotations::TriadInterpolationLocalRotationVectors<nnodetriad, double>>
      triad_interpolation_scheme_ptr = std::make_shared<
          LargeRotations::TriadInterpolationLocalRotationVectors<nnodetriad, double>>();

  // reset the scheme with nodal quaternions
  triad_interpolation_scheme_ptr->reset(Qnode);


  for (int gp = 0; gp < gausspoints.nquad; gp++)
  {
    triad_interpolation_scheme_ptr->get_interpolated_local_rotation_vector(Psi_l, I_i[gp]);

    triad_interpolation_scheme_ptr->get_interpolated_quaternion(QnewGP, Psi_l);

    // store in class variable in order to get QconvGPmass_ in subsequent time step
    qnew_gp_dampstoch_[gp] = QnewGP;

    // compute triad at Gauss point
    Core::LargeRotations::quaterniontotriad(QnewGP, LambdaGP);


    // rotation between last converged state and current state expressed as a quaternion

    // ******** alternative 1 *************** Todo @grill
    //    Core::LinAlg::Matrix<4,1> deltaQ;
    //    Core::LargeRotations::quaternionproduct(
    //        Core::LargeRotations::inversequaternion(QconvGPdampstoch_[gp]), QnewGP, deltaQ);

    // ******** alternative 2 ***************

    // get quaternion in converged state at gp and compute corresponding triad
    Core::LinAlg::Matrix<3, 3, double> triad_mat_conv(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<4, 1, double> Qconv(Core::LinAlg::Initialization::zero);
    for (unsigned int i = 0; i < 4; ++i) Qconv(i) = (qconv_gp_dampstoch_[gp])(i);

    Core::LargeRotations::quaterniontotriad(Qconv, triad_mat_conv);

    // compute quaternion of relative rotation from converged to current state
    Core::LinAlg::Matrix<3, 3, double> deltatriad(Core::LinAlg::Initialization::zero);
    deltatriad.multiply_nt(LambdaGP, triad_mat_conv);

    Core::LinAlg::Matrix<4, 1, double> deltaQ(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::triadtoquaternion(deltatriad, deltaQ);

    // **************************************

    // extract rotation vector from quaternion
    Core::LinAlg::Matrix<3, 1> deltatheta;
    Core::LargeRotations::quaterniontoangle(deltaQ, deltatheta);

    // angular velocity at this Gauss point according to backward Euler scheme
    Core::LinAlg::Matrix<3, 1> omega(Core::LinAlg::Initialization::zero);
    omega.update(dt_inv, deltatheta);

    // compute matrix Lambda*[gamma(2) 0 0 \\ 0 0 0 \\ 0 0 0]*Lambda^t = gamma(2) * g_1 \otimes g_1
    // where g_1 is first base vector, i.e. first column of Lambda
    Core::LinAlg::Matrix<3, 3> g1g1gamma;
    for (int k = 0; k < 3; k++)
      for (int j = 0; j < 3; j++) g1g1gamma(k, j) = LambdaGP(k, 0) * LambdaGP(j, 0) * gamma(2);

    // compute vector gamma(2) * g_1 \otimes g_1 * \omega
    Core::LinAlg::Matrix<3, 1> g1g1gammaomega;
    g1g1gammaomega.multiply(g1g1gamma, omega);

    const double jacobifac_gp_weight = jacobi_gp_dampstoch_[gp] * gausspoints.qwgt[gp];

    if (force != nullptr)
    {
      // loop over all nodes
      for (unsigned int inode = 0; inode < nnodecl; inode++)
      {
        // loop over spatial dimensions
        for (unsigned int idim = 0; idim < ndim; idim++)
          (*force)(dofpercombinode* inode + 3 + idim) +=
              g1g1gammaomega(idim) * (I_i[gp])(inode)*jacobifac_gp_weight;
      }
      for (unsigned int inode = nnodecl; inode < nnodetriad; inode++)
      {
        // loop over spatial dimensions
        for (unsigned int idim = 0; idim < ndim; idim++)
          (*force)(dofperclnode* nnodecl + dofpertriadnode * inode + idim) +=
              g1g1gammaomega(idim) * (I_i[gp])(inode)*jacobifac_gp_weight;
      }
    }

    if (stiffmatrix != nullptr)
    {
      triad_interpolation_scheme_ptr->get_nodal_generalized_rotation_interpolation_matrices(
          Itilde, Psi_l, I_i[gp]);

      Core::LinAlg::Matrix<3, 3> g1g1oldgamma;
      for (int k = 0; k < 3; k++)
        for (int j = 0; j < 3; j++)
          g1g1oldgamma(k, j) = LambdaGP(k, 0) * triad_mat_conv(j, 0) * gamma(2);


      // compute matrix gamma(2) * g_1 \otimes g_1 * \omega * Tmat
      Core::LinAlg::Matrix<3, 3> g1g1oldgammaTmat;
      g1g1oldgammaTmat.multiply(g1g1oldgamma, Core::LargeRotations::tmatrix(deltatheta));

      // compute spin matrix S(\omega)
      Core::LinAlg::Matrix<3, 3> Sofomega;
      Core::LargeRotations::computespin(Sofomega, omega);

      // compute matrix gamma(2) * g_1 \otimes g_1 *S(\omega)
      Core::LinAlg::Matrix<3, 3> g1g1gammaSofomega;
      g1g1gammaSofomega.multiply(g1g1gamma, Sofomega);

      // compute spin matrix S(gamma(2) * g_1 \otimes g_1 *\omega)
      Core::LinAlg::Matrix<3, 3> Sofg1g1gammaomega;
      Core::LargeRotations::computespin(Sofg1g1gammaomega, g1g1gammaomega);

      // auxiliary matrices
      Core::LinAlg::Matrix<3, 3> sum(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<3, 3> auxmatrix(Core::LinAlg::Initialization::zero);

      sum += g1g1oldgammaTmat;
      sum.scale(dt_inv);
      sum += g1g1gammaSofomega;
      sum -= Sofg1g1gammaomega;

      // loop over first nnodecl row nodes
      for (unsigned int inode = 0; inode < nnodecl; inode++)
      {
        // loop over first nnodecl column nodes
        for (unsigned int jnode = 0; jnode < nnodecl; jnode++)
        {
          auxmatrix.multiply(sum, Itilde[jnode]);

          // loop over three dimensions in row and column direction
          for (unsigned int idim = 0; idim < ndim; idim++)
            for (unsigned int jdim = 0; jdim < 3; jdim++)
            {
              (*stiffmatrix)(
                  dofpercombinode* inode + 3 + idim, dofpercombinode * jnode + 3 + jdim) +=
                  auxmatrix(idim, jdim) * (I_i[gp])(inode)*jacobifac_gp_weight;
            }
        }
        for (unsigned int jnode = nnodecl; jnode < nnodetriad;
            jnode++)  // this loop is only entered in case of nnodetriad>nnodecl
        {
          auxmatrix.multiply(sum, Itilde[jnode]);

          // loop over three dimensions in row and column direction
          for (unsigned int idim = 0; idim < ndim; idim++)
            for (unsigned int jdim = 0; jdim < 3; jdim++)
            {
              (*stiffmatrix)(dofpercombinode* inode + 3 + idim,
                  dofperclnode * nnodecl + dofpertriadnode * jnode + jdim) +=
                  auxmatrix(idim, jdim) * (I_i[gp])(inode)*jacobifac_gp_weight;
            }
        }
      }
      for (unsigned int inode = nnodecl; inode < nnodetriad;
          inode++)  // this loop is only entered in case of nnodetriad>nnodecl
      {
        // loop over all column nodes
        for (unsigned int jnode = 0; jnode < nnodecl; jnode++)
        {
          auxmatrix.multiply(sum, Itilde[jnode]);

          // loop over three dimensions in row and column direction
          for (unsigned int idim = 0; idim < ndim; idim++)
            for (unsigned int jdim = 0; jdim < 3; jdim++)
            {
              (*stiffmatrix)(dofperclnode* nnodecl + dofpertriadnode * inode + idim,
                  dofpercombinode * jnode + 3 + jdim) +=
                  auxmatrix(idim, jdim) * (I_i[gp])(inode)*jacobifac_gp_weight;
            }
        }
        for (unsigned int jnode = nnodecl; jnode < nnodetriad;
            jnode++)  // this loop is only entered in case of nnodetriad>nnodecl
        {
          auxmatrix.multiply(sum, Itilde[jnode]);

          // loop over three dimensions in row and column direction
          for (unsigned int idim = 0; idim < ndim; idim++)
            for (unsigned int jdim = 0; jdim < 3; jdim++)
            {
              (*stiffmatrix)(dofperclnode* nnodecl + dofpertriadnode * inode + idim,
                  dofperclnode * nnodecl + dofpertriadnode * jnode + jdim) +=
                  auxmatrix(idim, jdim) * (I_i[gp])(inode)*jacobifac_gp_weight;
            }
        }
      }
    }
  }
}

/*-----------------------------------------------------------------------------------------------------------*
 | computes translational damping forces and stiffness (public) cyron   10/09|
 *----------------------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, unsigned int vpernode, unsigned int ndim>
void Discret::Elements::Beam3r::evaluate_translational_damping(Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<ndim * vpernode * nnodecl, 1, double>& vel_centerline,
    const Core::LinAlg::Matrix<ndim * vpernode * nnodecl, 1, double>& disp_totlag_centerline,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseVector* force) const
{
  /* only nodes for centerline interpolation are considered here (= first nnodecl nodes of this
     element); each of these nodes holds 3*vpernode translational DoFs AND 3 rotational DoFs */
  const unsigned int dofpernode = 3 * vpernode + 3;

  // get time step size
  double dt_inv = 0.0001;
  if (is_params_interface())
    dt_inv = 1.0 / params_interface().get_delta_time();
  else
    dt_inv = 1.0 / params.get<double>("delta time", 1000);

  // velocity and gradient of background velocity field
  Core::LinAlg::Matrix<ndim, 1> velbackground;
  Core::LinAlg::Matrix<ndim, ndim> velbackgroundgrad;

  // position of beam centerline point corresponding to a certain Gauss point
  Core::LinAlg::Matrix<ndim, 1> r(Core::LinAlg::Initialization::zero);
  // tangent vector (derivative of beam centerline curve r with respect to arc-length parameter s)
  Core::LinAlg::Matrix<ndim, 1> r_s(Core::LinAlg::Initialization::zero);
  // velocity of beam centerline point relative to background fluid velocity
  Core::LinAlg::Matrix<ndim, 1> vel_rel(Core::LinAlg::Initialization::zero);

  // damping coefficients for translational and rotational degrees of freedom
  Core::LinAlg::Matrix<3, 1> gamma(Core::LinAlg::Initialization::zero);
  get_damping_coefficients(gamma);

  // viscous force vector per unit length at current GP
  Core::LinAlg::Matrix<ndim, 1> f_visc(Core::LinAlg::Initialization::zero);
  // damping matrix
  Core::LinAlg::Matrix<ndim, ndim> damp_mat(Core::LinAlg::Initialization::zero);

  // get Gauss points and weights for evaluation of damping matrix
  Core::FE::GaussRule1D gaussrule = my_gauss_rule(res_damp_stoch);
  Core::FE::IntegrationPoints1D gausspoints(gaussrule);

  /* vector whose numgp-th element is a 1x(vpernode*nnode)-matrix with all (Lagrange/Hermite) shape
   * functions evaluated at the numgp-th GP
   * these shape functions are used for the interpolation of the beam centerline*/
  std::vector<Core::LinAlg::Matrix<1, vpernode * nnodecl, double>> H_i(gausspoints.nquad);
  // same for the derivatives
  std::vector<Core::LinAlg::Matrix<1, vpernode * nnodecl, double>> H_i_xi(gausspoints.nquad);

  // evaluate all shape functions and derivatives with respect to element parameter xi at all
  // specified Gauss points
  Discret::Utils::Beam::evaluate_shape_functions_and_derivs_all_gps<nnodecl, vpernode>(
      gausspoints, H_i, H_i_xi, this->shape(), this->ref_length());

  for (int gp = 0; gp < gausspoints.nquad; gp++)
  {
    // compute position vector r of point in physical space corresponding to Gauss point
    calc_r<nnodecl, vpernode, double>(disp_totlag_centerline, H_i[gp], r);

    // compute tangent vector t_{\par}=r' at current Gauss point
    calc_r_s<nnodecl, vpernode, double>(
        disp_totlag_centerline, H_i_xi[gp], jacobi_gp_dampstoch_[gp], r_s);

    // compute velocity and gradient of background flow field at point r
    get_background_velocity<ndim, double>(params, r, velbackground, velbackgroundgrad);

    // compute velocity vector at this Gauss point via same interpolation as for centerline position
    // vector
    Discret::Utils::Beam::calc_interpolation<nnodecl, vpernode, 3, double>(
        vel_centerline, H_i[gp], vel_rel);
    vel_rel -= velbackground;

    // loop over lines and columns of damping matrix
    for (unsigned int idim = 0; idim < ndim; idim++)
      for (unsigned int jdim = 0; jdim < ndim; jdim++)
        damp_mat(idim, jdim) =
            (idim == jdim) * gamma(1) + (gamma(0) - gamma(1)) * r_s(idim) * r_s(jdim);

    // compute viscous force vector per unit length at current GP
    f_visc.multiply(damp_mat, vel_rel);

    const double jacobifac_gp_weight = jacobi_gp_dampstoch_[gp] * gausspoints.qwgt[gp];

    if (force != nullptr)
    {
      // loop over all nodes used for centerline interpolation
      for (unsigned int inode = 0; inode < nnodecl; inode++)
        // loop over dimensions
        for (unsigned int idim = 0; idim < ndim; idim++)
        {
          (*force)(inode* dofpernode + idim) +=
              H_i[gp](vpernode * inode) * f_visc(idim) * jacobifac_gp_weight;
          if (centerline_hermite_)
            (*force)(inode* dofpernode + 6 + idim) +=
                H_i[gp](vpernode * inode + 1) * f_visc(idim) * jacobifac_gp_weight;
        }
    }

    if (stiffmatrix != nullptr)
    {
      // compute matrix product of damping matrix and gradient of background velocity
      Core::LinAlg::Matrix<ndim, ndim> dampmatvelbackgroundgrad(Core::LinAlg::Initialization::zero);
      dampmatvelbackgroundgrad.multiply(damp_mat, velbackgroundgrad);


      // loop over all nodes used for centerline interpolation
      for (unsigned int inode = 0; inode < nnodecl; inode++)
        // loop over all column nodes used for centerline interpolation
        for (unsigned int jnode = 0; jnode < nnodecl; jnode++)
        {
          for (unsigned int idim = 0; idim < ndim; idim++)
            for (unsigned int jdim = 0; jdim < ndim; jdim++)
            {
              (*stiffmatrix)(inode* dofpernode + idim, jnode * dofpernode + jdim) +=
                  gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) * H_i[gp](vpernode * jnode) *
                  jacobi_gp_dampstoch_[gp] * damp_mat(idim, jdim) * dt_inv;
              (*stiffmatrix)(inode* dofpernode + idim, jnode * dofpernode + jdim) -=
                  gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) * H_i[gp](vpernode * jnode) *
                  jacobi_gp_dampstoch_[gp] * dampmatvelbackgroundgrad(idim, jdim);
              (*stiffmatrix)(inode* dofpernode + idim, jnode * dofpernode + idim) +=
                  gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) * H_i_xi[gp](vpernode * jnode) *
                  (gamma(0) - gamma(1)) * r_s(jdim) * vel_rel(jdim);
              (*stiffmatrix)(inode* dofpernode + idim, jnode * dofpernode + jdim) +=
                  gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) * H_i_xi[gp](vpernode * jnode) *
                  (gamma(0) - gamma(1)) * r_s(idim) * vel_rel(jdim);

              if (centerline_hermite_)
              {
                (*stiffmatrix)(inode* dofpernode + 6 + idim, jnode * dofpernode + jdim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i[gp](vpernode * jnode) * jacobi_gp_dampstoch_[gp] * damp_mat(idim, jdim) *
                    dt_inv;
                (*stiffmatrix)(inode* dofpernode + 6 + idim, jnode * dofpernode + jdim) -=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i[gp](vpernode * jnode) * jacobi_gp_dampstoch_[gp] *
                    dampmatvelbackgroundgrad(idim, jdim);
                (*stiffmatrix)(inode* dofpernode + 6 + idim, jnode * dofpernode + idim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i_xi[gp](vpernode * jnode) * (gamma(0) - gamma(1)) * r_s(jdim) *
                    vel_rel(jdim);
                (*stiffmatrix)(inode* dofpernode + 6 + idim, jnode * dofpernode + jdim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i_xi[gp](vpernode * jnode) * (gamma(0) - gamma(1)) * r_s(idim) *
                    vel_rel(jdim);

                (*stiffmatrix)(inode* dofpernode + idim, jnode * dofpernode + 6 + jdim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) *
                    H_i[gp](vpernode * jnode + 1) * jacobi_gp_dampstoch_[gp] *
                    damp_mat(idim, jdim) * dt_inv;
                (*stiffmatrix)(inode* dofpernode + idim, jnode * dofpernode + 6 + jdim) -=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) *
                    H_i[gp](vpernode * jnode + 1) * jacobi_gp_dampstoch_[gp] *
                    dampmatvelbackgroundgrad(idim, jdim);
                (*stiffmatrix)(inode* dofpernode + idim, jnode * dofpernode + 6 + idim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) *
                    H_i_xi[gp](vpernode * jnode + 1) * (gamma(0) - gamma(1)) * r_s(jdim) *
                    vel_rel(jdim);
                (*stiffmatrix)(inode* dofpernode + idim, jnode * dofpernode + 6 + jdim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) *
                    H_i_xi[gp](vpernode * jnode + 1) * (gamma(0) - gamma(1)) * r_s(idim) *
                    vel_rel(jdim);

                (*stiffmatrix)(inode* dofpernode + 6 + idim, jnode * dofpernode + 6 + jdim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i[gp](vpernode * jnode + 1) * jacobi_gp_dampstoch_[gp] *
                    damp_mat(idim, jdim) * dt_inv;
                (*stiffmatrix)(inode* dofpernode + 6 + idim, jnode * dofpernode + 6 + jdim) -=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i[gp](vpernode * jnode + 1) * jacobi_gp_dampstoch_[gp] *
                    dampmatvelbackgroundgrad(idim, jdim);
                (*stiffmatrix)(inode* dofpernode + 6 + idim, jnode * dofpernode + 6 + idim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i_xi[gp](vpernode * jnode + 1) * (gamma(0) - gamma(1)) * r_s(jdim) *
                    vel_rel(jdim);
                (*stiffmatrix)(inode* dofpernode + 6 + idim, jnode * dofpernode + 6 + jdim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i_xi[gp](vpernode * jnode + 1) * (gamma(0) - gamma(1)) * r_s(idim) *
                    vel_rel(jdim);
              }
            }
        }
    }
  }
}

/*-----------------------------------------------------------------------------------------------------------*
 | computes stochastic forces and resulting stiffness (public) cyron   10/09|
 *----------------------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, unsigned int vpernode, unsigned int ndim,
    unsigned int randompergauss>  // number of nodes, number of dimensions of embedding space,
                                  // number of degrees of freedom per node, number of random numbers
                                  // required per Gauss point
void Discret::Elements::Beam3r::evaluate_stochastic_forces(Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<ndim * vpernode * nnodecl, 1, double>& disp_totlag_centerline,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseVector* force) const
{
  /* only nodes for centerline interpolation are considered here (= first nnodecl nodes of this
     element); each of these nodes holds 3*vpernode translational DoFs AND 3 rotational DoFs */
  const unsigned int dofpernode = 3 * vpernode + 3;

  // damping coefficients for three translational and one rotational degree of freedom
  Core::LinAlg::Matrix<3, 1> sqrt_gamma(Core::LinAlg::Initialization::zero);
  get_damping_coefficients(sqrt_gamma);
  for (unsigned int i = 0; i < 2; ++i) sqrt_gamma(i) = std::sqrt(sqrt_gamma(i));

  /* get pointer at multivector in parameter list linking to random numbers for stochastic
   * forces with zero mean and standard deviation (2*kT / dt)^0.5 */
  std::shared_ptr<Core::LinAlg::MultiVector<double>> randomforces =
      brownian_dyn_params_interface().get_random_forces();

  // my random number vector at current GP
  Core::LinAlg::Matrix<ndim, 1> randnumvec(Core::LinAlg::Initialization::zero);

  // tangent vector (derivative of beam centerline curve r with respect to arc-length parameter s)
  Core::LinAlg::Matrix<ndim, 1> r_s(Core::LinAlg::Initialization::zero);

  // stochastic force vector per unit length at current GP
  Core::LinAlg::Matrix<ndim, 1> f_stoch(Core::LinAlg::Initialization::zero);

  // get Gauss points and weights for evaluation of damping matrix
  Core::FE::GaussRule1D gaussrule = my_gauss_rule(res_damp_stoch);
  Core::FE::IntegrationPoints1D gausspoints(gaussrule);

  /* vector whose numgp-th element is a 1x(vpernode*nnode)-matrix with all (Lagrange/Hermite) shape
   * functions evaluated at the numgp-th GP
   * these shape functions are used for the interpolation of the beam centerline*/
  std::vector<Core::LinAlg::Matrix<1, vpernode * nnodecl, double>> H_i(gausspoints.nquad);
  // same for the derivatives
  std::vector<Core::LinAlg::Matrix<1, vpernode * nnodecl, double>> H_i_xi(gausspoints.nquad);

  // evaluate all shape function derivatives with respect to element parameter xi at all specified
  // Gauss points
  Discret::Utils::Beam::evaluate_shape_functions_and_derivs_all_gps<nnodecl, vpernode>(
      gausspoints, H_i, H_i_xi, this->shape(), this->ref_length());


  for (int gp = 0; gp < gausspoints.nquad; gp++)
  {
    // compute tangent vector t_{\par}=r' at current Gauss point
    calc_r_s<nnodecl, vpernode, double>(
        disp_totlag_centerline, H_i_xi[gp], jacobi_gp_dampstoch_[gp], r_s);

    // extract random numbers from global vector
    for (unsigned int idim = 0; idim < ndim; idim++)
    {
#ifndef BEAM3RCONSTSTOCHFORCE
      randnumvec(idim) = (*randomforces)(gp * randompergauss + idim).local_values_as_span()[lid()];
#else
      randnumvec(idim) = (*randomforces)[idim][LID()];
#endif
    }

    // compute stochastic force vector per unit length at current GP
    f_stoch.clear();
    for (unsigned int idim = 0; idim < ndim; idim++)
      for (unsigned int jdim = 0; jdim < ndim; jdim++)
        f_stoch(idim) += (sqrt_gamma(1) * (idim == jdim) +
                             (sqrt_gamma(0) - sqrt_gamma(1)) * r_s(idim) * r_s(jdim)) *
                         randnumvec(jdim);

    const double sqrt_jacobifac_gp_weight =
        std::sqrt(jacobi_gp_dampstoch_[gp] * gausspoints.qwgt[gp]);

    if (force != nullptr)
    {
      // loop over all nodes
      for (unsigned int inode = 0; inode < nnodecl; inode++)
        // loop dimensions with respect to lines
        for (unsigned int idim = 0; idim < ndim; idim++)
        {
          (*force)(inode* dofpernode + idim) -=
              H_i[gp](vpernode * inode) * f_stoch(idim) * sqrt_jacobifac_gp_weight;
          if (centerline_hermite_)
          {
            (*force)(inode* dofpernode + 6 + idim) -=
                H_i[gp](vpernode * inode + 1) * f_stoch(idim) * sqrt_jacobifac_gp_weight;
          }
        }
    }

    if (stiffmatrix != nullptr)
    {
      // note: division by sqrt of jacobi factor, because H_i_s = H_i_xi / jacobifactor
      const double sqrt_gp_weight_jacobifac_inv =
          std::sqrt(gausspoints.qwgt[gp] / jacobi_gp_dampstoch_[gp]);

      const double prefactor = sqrt_gp_weight_jacobifac_inv * (sqrt_gamma(0) - sqrt_gamma(1));

      // loop over all nodes used for centerline interpolation
      for (unsigned int inode = 0; inode < nnodecl; inode++)
        // loop over all column nodes used for centerline interpolation
        for (unsigned int jnode = 0; jnode < nnodecl; jnode++)
        {
          for (unsigned int idim = 0; idim < ndim; idim++)
            for (unsigned int jdim = 0; jdim < ndim; jdim++)
            {
              (*stiffmatrix)(inode* dofpernode + idim, jnode * dofpernode + idim) -=
                  H_i[gp](vpernode * inode) * H_i_xi[gp](vpernode * jnode) * r_s(jdim) *
                  randnumvec(jdim) * prefactor;
              (*stiffmatrix)(inode* dofpernode + idim, jnode * dofpernode + jdim) -=
                  H_i[gp](vpernode * inode) * H_i_xi[gp](vpernode * jnode) * r_s(idim) *
                  randnumvec(jdim) * prefactor;

              if (centerline_hermite_)
              {
                (*stiffmatrix)(inode* dofpernode + 6 + idim, jnode * dofpernode + idim) -=
                    H_i[gp](vpernode * inode + 1) * H_i_xi[gp](vpernode * jnode) * r_s(jdim) *
                    randnumvec(jdim) * prefactor;
                (*stiffmatrix)(inode* dofpernode + 6 + idim, jnode * dofpernode + jdim) -=
                    H_i[gp](vpernode * inode + 1) * H_i_xi[gp](vpernode * jnode) * r_s(idim) *
                    randnumvec(jdim) * prefactor;

                (*stiffmatrix)(inode* dofpernode + idim, jnode * dofpernode + 6 + idim) -=
                    H_i[gp](vpernode * inode) * H_i_xi[gp](vpernode * jnode + 1) * r_s(jdim) *
                    randnumvec(jdim) * prefactor;
                (*stiffmatrix)(inode* dofpernode + idim, jnode * dofpernode + 6 + jdim) -=
                    H_i[gp](vpernode * inode) * H_i_xi[gp](vpernode * jnode + 1) * r_s(idim) *
                    randnumvec(jdim) * prefactor;

                (*stiffmatrix)(inode* dofpernode + 6 + idim, jnode * dofpernode + 6 + idim) -=
                    H_i[gp](vpernode * inode + 1) * H_i_xi[gp](vpernode * jnode + 1) * r_s(jdim) *
                    randnumvec(jdim) * prefactor;
                (*stiffmatrix)(inode* dofpernode + 6 + idim, jnode * dofpernode + 6 + jdim) -=
                    H_i[gp](vpernode * inode + 1) * H_i_xi[gp](vpernode * jnode + 1) * r_s(idim) *
                    randnumvec(jdim) * prefactor;
              }
            }
        }
    }
  }
}


// explicit template instantiations
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<2, 2, 1>(
    Teuchos::ParameterList&, std::vector<double>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseVector*,
    Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<3, 3, 1>(
    Teuchos::ParameterList&, std::vector<double>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseVector*,
    Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<4, 4, 1>(
    Teuchos::ParameterList&, std::vector<double>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseVector*,
    Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<5, 5, 1>(
    Teuchos::ParameterList&, std::vector<double>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseVector*,
    Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<2, 2, 2>(
    Teuchos::ParameterList&, std::vector<double>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseVector*,
    Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<3, 2, 2>(
    Teuchos::ParameterList&, std::vector<double>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseVector*,
    Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<4, 2, 2>(
    Teuchos::ParameterList&, std::vector<double>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseVector*,
    Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<5, 2, 2>(
    Teuchos::ParameterList&, std::vector<double>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseVector*,
    Core::LinAlg::SerialDenseVector*);

template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<2, 2, 1>(
    Core::LinAlg::Matrix<6, 1, double>&, std::vector<Core::LinAlg::Matrix<4, 1, double>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*, Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<3, 3, 1>(
    Core::LinAlg::Matrix<9, 1, double>&, std::vector<Core::LinAlg::Matrix<4, 1, double>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*, Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<4, 4, 1>(
    Core::LinAlg::Matrix<12, 1, double>&, std::vector<Core::LinAlg::Matrix<4, 1, double>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*, Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<5, 5, 1>(
    Core::LinAlg::Matrix<15, 1, double>&, std::vector<Core::LinAlg::Matrix<4, 1, double>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*, Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<2, 2, 2>(
    Core::LinAlg::Matrix<12, 1, double>&, std::vector<Core::LinAlg::Matrix<4, 1, double>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*, Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<3, 2, 2>(
    Core::LinAlg::Matrix<12, 1, double>&, std::vector<Core::LinAlg::Matrix<4, 1, double>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*, Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<4, 2, 2>(
    Core::LinAlg::Matrix<12, 1, double>&, std::vector<Core::LinAlg::Matrix<4, 1, double>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*, Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_internal_and_inertia_forces_and_stiff<5, 2, 2>(
    Core::LinAlg::Matrix<12, 1, double>&, std::vector<Core::LinAlg::Matrix<4, 1, double>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*, Core::LinAlg::SerialDenseVector*);

template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<2, 2, 1, double>(
    const Core::LinAlg::Matrix<6, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<12, 1, double>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<3, 3, 1, double>(
    const Core::LinAlg::Matrix<9, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<18, 1, double>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<4, 4, 1, double>(
    const Core::LinAlg::Matrix<12, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<24, 1, double>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<5, 5, 1, double>(
    const Core::LinAlg::Matrix<15, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<30, 1, double>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<2, 2, 2, double>(
    const Core::LinAlg::Matrix<12, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<18, 1, double>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<3, 2, 2, double>(
    const Core::LinAlg::Matrix<12, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<21, 1, double>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<4, 2, 2, double>(
    const Core::LinAlg::Matrix<12, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<24, 1, double>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<5, 2, 2, double>(
    const Core::LinAlg::Matrix<12, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<27, 1, double>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<2, 2, 1,
    Sacado::Fad::DFad<double>>(const Core::LinAlg::Matrix<6, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::Matrix<12, 1, Sacado::Fad::DFad<double>>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<3, 3, 1,
    Sacado::Fad::DFad<double>>(const Core::LinAlg::Matrix<9, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::Matrix<18, 1, Sacado::Fad::DFad<double>>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<4, 4, 1,
    Sacado::Fad::DFad<double>>(const Core::LinAlg::Matrix<12, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::Matrix<24, 1, Sacado::Fad::DFad<double>>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<5, 5, 1,
    Sacado::Fad::DFad<double>>(const Core::LinAlg::Matrix<15, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::Matrix<30, 1, Sacado::Fad::DFad<double>>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<2, 2, 2,
    Sacado::Fad::DFad<double>>(const Core::LinAlg::Matrix<12, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::Matrix<18, 1, Sacado::Fad::DFad<double>>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<3, 2, 2,
    Sacado::Fad::DFad<double>>(const Core::LinAlg::Matrix<12, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::Matrix<21, 1, Sacado::Fad::DFad<double>>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<4, 2, 2,
    Sacado::Fad::DFad<double>>(const Core::LinAlg::Matrix<12, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::Matrix<24, 1, Sacado::Fad::DFad<double>>&);
template void Discret::Elements::Beam3r::calc_internal_force_and_stiff<5, 2, 2,
    Sacado::Fad::DFad<double>>(const Core::LinAlg::Matrix<12, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    Core::LinAlg::SerialDenseMatrix*, Core::LinAlg::Matrix<27, 1, Sacado::Fad::DFad<double>>&);

template void Discret::Elements::Beam3r::calc_inertia_force_and_mass_matrix<2, 2, 1>(
    const Core::LinAlg::Matrix<6, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_inertia_force_and_mass_matrix<3, 3, 1>(
    const Core::LinAlg::Matrix<9, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_inertia_force_and_mass_matrix<4, 4, 1>(
    const Core::LinAlg::Matrix<12, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_inertia_force_and_mass_matrix<5, 5, 1>(
    const Core::LinAlg::Matrix<15, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_inertia_force_and_mass_matrix<2, 2, 2>(
    const Core::LinAlg::Matrix<12, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_inertia_force_and_mass_matrix<3, 2, 2>(
    const Core::LinAlg::Matrix<12, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_inertia_force_and_mass_matrix<4, 2, 2>(
    const Core::LinAlg::Matrix<12, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*);
template void Discret::Elements::Beam3r::calc_inertia_force_and_mass_matrix<5, 2, 2>(
    const Core::LinAlg::Matrix<12, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<4, 1, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::SerialDenseVector*);

template void Discret::Elements::Beam3r::calc_stiffmat_analytic_force_contributions<2, 2, 1>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&, const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<2, double>&,
    const Core::LinAlg::Matrix<1, 2, double>&, const Core::LinAlg::Matrix<1, 2, double>&,
    const double, const double) const;
template void Discret::Elements::Beam3r::calc_stiffmat_analytic_force_contributions<3, 3, 1>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&, const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&,
    const Core::LinAlg::Matrix<1, 3, double>&, const Core::LinAlg::Matrix<1, 3, double>&,
    const double, const double) const;
template void Discret::Elements::Beam3r::calc_stiffmat_analytic_force_contributions<4, 4, 1>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&, const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<4, double>&,
    const Core::LinAlg::Matrix<1, 4, double>&, const Core::LinAlg::Matrix<1, 4, double>&,
    const double, const double) const;
template void Discret::Elements::Beam3r::calc_stiffmat_analytic_force_contributions<5, 5, 1>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&, const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<5, double>&,
    const Core::LinAlg::Matrix<1, 5, double>&, const Core::LinAlg::Matrix<1, 5, double>&,
    const double, const double) const;
template void Discret::Elements::Beam3r::calc_stiffmat_analytic_force_contributions<2, 2, 2>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&, const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<2, double>&,
    const Core::LinAlg::Matrix<1, 2, double>&, const Core::LinAlg::Matrix<1, 4, double>&,
    const double, const double) const;
template void Discret::Elements::Beam3r::calc_stiffmat_analytic_force_contributions<3, 2, 2>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&, const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&,
    const Core::LinAlg::Matrix<1, 3, double>&, const Core::LinAlg::Matrix<1, 4, double>&,
    const double, const double) const;
template void Discret::Elements::Beam3r::calc_stiffmat_analytic_force_contributions<4, 2, 2>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&, const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<4, double>&,
    const Core::LinAlg::Matrix<1, 4, double>&, const Core::LinAlg::Matrix<1, 4, double>&,
    const double, const double) const;
template void Discret::Elements::Beam3r::calc_stiffmat_analytic_force_contributions<5, 2, 2>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&, const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<5, double>&,
    const Core::LinAlg::Matrix<1, 5, double>&, const Core::LinAlg::Matrix<1, 4, double>&,
    const double, const double) const;

template void Discret::Elements::Beam3r::calc_stiffmat_analytic_moment_contributions<2, 2, 1>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<2, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<1, 2, double>&, const Core::LinAlg::Matrix<1, 2, double>&,
    const double, const double) const;
template void Discret::Elements::Beam3r::calc_stiffmat_analytic_moment_contributions<3, 3, 1>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<1, 3, double>&, const Core::LinAlg::Matrix<1, 3, double>&,
    const double, const double) const;
template void Discret::Elements::Beam3r::calc_stiffmat_analytic_moment_contributions<4, 4, 1>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<4, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<1, 4, double>&, const Core::LinAlg::Matrix<1, 4, double>&,
    const double, const double) const;
template void Discret::Elements::Beam3r::calc_stiffmat_analytic_moment_contributions<5, 5, 1>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<5, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<1, 5, double>&, const Core::LinAlg::Matrix<1, 5, double>&,
    const double, const double) const;
template void Discret::Elements::Beam3r::calc_stiffmat_analytic_moment_contributions<2, 2, 2>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<2, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<1, 2, double>&, const Core::LinAlg::Matrix<1, 2, double>&,
    const double, const double) const;
template void Discret::Elements::Beam3r::calc_stiffmat_analytic_moment_contributions<3, 2, 2>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<1, 3, double>&, const Core::LinAlg::Matrix<1, 3, double>&,
    const double, const double) const;
template void Discret::Elements::Beam3r::calc_stiffmat_analytic_moment_contributions<4, 2, 2>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<4, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<1, 4, double>&, const Core::LinAlg::Matrix<1, 4, double>&,
    const double, const double) const;
template void Discret::Elements::Beam3r::calc_stiffmat_analytic_moment_contributions<5, 2, 2>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<5, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<1, 5, double>&, const Core::LinAlg::Matrix<1, 5, double>&,
    const double, const double) const;

FOUR_C_NAMESPACE_CLOSE

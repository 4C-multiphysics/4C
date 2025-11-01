// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beam3_kirchhoff.hpp"
#include "4C_beam3_spatial_discretization_utils.hpp"
#include "4C_beam3_triad_interpolation_local_rotation_vectors.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_geometry_periodic_boundingbox.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------------------*
 |  evaluate the element (public) meier 01/16|
 *----------------------------------------------------------------------------------------------------------*/
int Discret::Elements::Beam3k::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1,  // stiffness matrix
    Core::LinAlg::SerialDenseMatrix& elemat2,  // mass matrix
    Core::LinAlg::SerialDenseVector& elevec1,  // internal forces
    Core::LinAlg::SerialDenseVector& elevec2,  // inertia forces
    Core::LinAlg::SerialDenseVector& elevec3)
{
  set_params_interface_ptr(params);

  // Set brownian params interface pointer
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
      FOUR_C_THROW("Unknown type of action for Beam3k");
  }

  switch (act)
  {
    case Core::Elements::struct_calc_ptcstiff:
    {
      FOUR_C_THROW("no ptc implemented for Beam3k element");
      break;
    }

    case Core::Elements::struct_calc_linstiff:
    {
      // only nonlinear case implemented!
      FOUR_C_THROW("linear stiffness matrix called, but not implemented");
      break;
    }

    // nonlinear stiffness and mass matrix are calculated even if only nonlinear stiffness matrix
    // is required
    case Core::Elements::struct_calc_nlnstiffmass:
    case Core::Elements::struct_calc_nlnstifflmass:
    case Core::Elements::struct_calc_nlnstiff:
    case Core::Elements::struct_calc_internalforce:
    case Core::Elements::struct_calc_internalinertiaforce:
    {
      // need current global displacement and residual forces and get them from discretization
      // making use of the local-to-global map lm one can extract current displacement and
      // residual values for each degree of freedom get element displacements
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");

      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);


      if (act == Core::Elements::struct_calc_nlnstiffmass)
      {
        calc_internal_and_inertia_forces_and_stiff(
            params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
      }
      else if (act == Core::Elements::struct_calc_nlnstifflmass)
      {
        FOUR_C_THROW("The action calc_struct_nlnstifflmass is not implemented yet!");
      }
      else if (act == Core::Elements::struct_calc_nlnstiff)
      {
        calc_internal_and_inertia_forces_and_stiff(
            params, mydisp, &elemat1, nullptr, &elevec1, nullptr);
      }
      else if (act == Core::Elements::struct_calc_internalforce)
      {
        calc_internal_and_inertia_forces_and_stiff(
            params, mydisp, nullptr, nullptr, &elevec1, nullptr);
      }
      else if (act == Core::Elements::struct_calc_internalinertiaforce)
      {
        calc_internal_and_inertia_forces_and_stiff(
            params, mydisp, nullptr, nullptr, &elevec1, &elevec2);
      }

      // ATTENTION: In order to perform a brief finite difference check of the nonlinear stiffness
      // matrix the code block "FD-CHECK" from the end of this file has to be copied to this
      // place:
      //***************************************************************************************************************
      // Insert code block here!
      //***************************************************************************************************************

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
        calc_brownian_forces_and_stiff<2, 2, 3>(params, myvel, mydisp, nullptr, &elevec1);
      else if (act == Core::Elements::struct_calc_brownianstiff)
        calc_brownian_forces_and_stiff<2, 2, 3>(params, myvel, mydisp, &elemat1, &elevec1);
      else
        FOUR_C_THROW("You shouldn't be here.");

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

    case Core::Elements::struct_calc_stress:
    {
      break;
    }

    case Core::Elements::struct_calc_update_istep:
    {
      /* the action calc_struct_update_istep is called in the very end of a time step when the new
       * dynamic equilibrium has finally been found; this is the point where the variable
       * representing the geometric
       * status of the beam at the end of the time step has to be stored */
      qconvmass_ = qnewmass_;
      wconvmass_ = wnewmass_;
      aconvmass_ = anewmass_;
      amodconvmass_ = amodnewmass_;
      rttconvmass_ = rttnewmass_;
      rttmodconvmass_ = rttmodnewmass_;
      rtconvmass_ = rtnewmass_;
      rconvmass_ = rnewmass_;

      qrefconv_ = qrefnew_;

      break;
    }

    case Core::Elements::struct_calc_reset_istep:
    {
      /* the action calc_struct_reset_istep is called by the adaptive time step controller;
       * carries out one test step whose purpose is only figuring out a suitable timestep; thus
       * this step may be a very bad one in order to iterated towards the new dynamic equilibrium
       * and the thereby gained new geometric configuration should not be applied as starting
       * point for any further iteration step; as a consequence the thereby generated change of
       * the geometric configuration should be canceled and the configuration should be reset to
       * the value at the beginning of the time step */

      qnewmass_ = qconvmass_;
      wnewmass_ = wconvmass_;
      anewmass_ = aconvmass_;
      amodnewmass_ = amodconvmass_;
      rttnewmass_ = rttconvmass_;
      rttmodnewmass_ = rttmodconvmass_;
      rtnewmass_ = rtconvmass_;
      rnewmass_ = rconvmass_;

      qrefnew_ = qrefconv_;

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

    default:
    {
      std::cout << "\ncalled element with action type " << action_type_to_string(act);
      FOUR_C_THROW("This action type is not implemented for Beam3k");
      break;
    }
  }

  return 0;
}

/*------------------------------------------------------------------------------------------------------------*
 | lump mass matrix             (private) meier 01/16|
 *------------------------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3k::lumpmass(Core::LinAlg::SerialDenseMatrix* emass)
{
  FOUR_C_THROW("Lumped mass matrix not implemented yet!!!");
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void Discret::Elements::Beam3k::calc_internal_and_inertia_forces_and_stiff(
    Teuchos::ParameterList& params, std::vector<double>& disp,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseMatrix* massmatrix,
    Core::LinAlg::SerialDenseVector* force, Core::LinAlg::SerialDenseVector* force_inert)
{
  // number of nodes used for centerline discretization fixed for this element
  const unsigned int nnodecl = 2;
  const unsigned int numdofelement = 2 * 3 * nnodecl + BEAM3K_COLLOCATION_POINTS;


  if (disp.size() != numdofelement)
    FOUR_C_THROW(
        "size mismatch: Number of BEAM3K_COLLOCATION_POINTS does not match number of nodes "
        "defined in the input file!");

  // unshift node positions, i.e. manipulate element displacement vector
  // as if there where no periodic boundary conditions
  if (brownian_dyn_params_interface_ptr() != nullptr)
    un_shift_node_position(disp, *brownian_dyn_params_interface().get_periodic_bounding_box());


  // vector for current nodal DoFs in total Lagrangian style, i.e. displacement + initial values:
  // rotvec_==true:  disp_totlag=[\v{d}_1, \v{theta}_1, t_1, \v{d}_2, \v{theta}_2, t_2, \alpha_3]
  // rotvec_==false: disp_totlag=[\v{d}_1, \v{t}_1, \alpha_1, \v{d}_2, \v{t}_2, \alpha_2, \alpha_3]
  // The Number of collocation points can take on the values 2, 3 and 4. 3 and 4 are interior nodes.
  // This leads e.g. in the case rotvec_==true to the following ordering:
  // if BEAM3K_COLLOCATION_POINTS = 2: [ \v{d}_1 \v{theta}_1 t_1 \v{d}_2 \v{theta}_1 t_2]
  // if BEAM3K_COLLOCATION_POINTS = 3: [ \v{d}_1 \v{theta}_1 t_1 \v{d}_2 \v{theta}_1 t_2 \alpha_3]
  // if BEAM3K_COLLOCATION_POINTS = 4: [ \v{d}_1 \v{theta}_1 t_1 \v{d}_2 \v{theta}_1 t_2 \alpha_3
  // \alpha_4]
  Core::LinAlg::Matrix<numdofelement, 1, double> disp_totlag(Core::LinAlg::Initialization::zero);

  // Set current positions and orientations at all nodes:
  update_disp_totlag<nnodecl, double>(disp, disp_totlag);


  // analytic linearization of internal forces
  if (not use_fad_)
  {
    if (rotvec_ == true)
    {
      FOUR_C_THROW(
          "Beam3k: analytic linearization of internal forces only implemented for "
          "tangent-based variant so far! activate FAD!");
    }

    // internal force vector
    Core::LinAlg::Matrix<numdofelement, 1, double> internal_force(
        Core::LinAlg::Initialization::zero);

    if (force != nullptr)
    {
      // set view on Core::LinAlg::SerialDenseVector to avoid copying of data
      internal_force.set_view(&((*force)(0)));
    }

    // vector containing locally assembled nodal positions and tangents required for centerline:
    // r(s)=N(s)*disp_totlag_centerline, with disp_totlag_centerline=[\v{d}_1, \v{t}_1, 0, \v{d}_2,
    // \v{t}_2, 0, 0]
    Core::LinAlg::Matrix<numdofelement, 1, double> disp_totlag_centerline(
        Core::LinAlg::Initialization::zero);

    // material triads at collocation points
    std::vector<Core::LinAlg::Matrix<3, 3, double>> triad_mat_cp(BEAM3K_COLLOCATION_POINTS,
        Core::LinAlg::Matrix<3, 3, double>(Core::LinAlg::Initialization::zero));

    update_nodal_variables<nnodecl, double>(
        disp_totlag, disp_totlag_centerline, triad_mat_cp, qrefnew_);

    // Store nodal tangents in class variable
    for (unsigned int i = 0; i < 3; ++i)
    {
      t_[0](i) = disp_totlag_centerline(3 + i);
      t_[1](i) = disp_totlag_centerline(10 + i);
    }


    // pre-computed variables that are later re-used in computation of inertia terms

    // interpolated spin vector variation at Gauss points required for inertia forces:
    // v_theta = v_thetaperp + v_thetapar
    std::vector<Core::LinAlg::Matrix<numdofelement, 3, double>> v_theta_gp;

    // multiplicative rotation increment at GP
    // required for analytic linearization and re-used for analytic mass matrix
    std::vector<Core::LinAlg::Matrix<3, numdofelement, double>> lin_theta_gp;

    // Interpolated material triad at Gauss points
    std::vector<Core::LinAlg::Matrix<3, 3, double>> triad_mat_gp;


    if (weakkirchhoff_)
    {
      calculate_internal_forces_and_stiff_wk<nnodecl, double>(params, disp_totlag_centerline,
          triad_mat_cp, stiffmatrix, internal_force, v_theta_gp, lin_theta_gp, triad_mat_gp);
    }
    else
    {
      FOUR_C_THROW(
          "Beam3k: analytic linearization only implemented for variant WK so far! "
          "activate FAD!");
    }

    // *************** INERTIA *******************************************
    if (massmatrix != nullptr or force_inert != nullptr)
    {
      // construct as a view on Core::LinAlg::SerialDenseVector to avoid copying of data
      Core::LinAlg::Matrix<numdofelement, 1, double> inertia_force(*force_inert, true);

      calculate_inertia_forces_and_mass_matrix<nnodecl, double>(params, triad_mat_gp,
          disp_totlag_centerline, v_theta_gp, lin_theta_gp, inertia_force, massmatrix);
    }
  }
  // automatic linearization of internal forces via FAD
  else
  {
    // internal force vector
    Core::LinAlg::Matrix<numdofelement, 1, FAD> internal_force_FAD(
        Core::LinAlg::Initialization::zero);

    // copy pre-computed disp_totlag to a FAD matrix
    Core::LinAlg::Matrix<numdofelement, 1, FAD> disp_totlag_FAD(Core::LinAlg::Initialization::zero);

    for (unsigned int idof = 0; idof < numdofelement; ++idof)
      disp_totlag_FAD(idof) = disp_totlag(idof);

    // vector containing locally assembled nodal positions and tangents required for centerline:
    // r(s)=N(s)*disp_totlag_centerline, with disp_totlag_centerline=[\v{d}_1, \v{t}_1, 0, \v{d}_2,
    // \v{t}_2, 0, 0]
    Core::LinAlg::Matrix<numdofelement, 1, FAD> disp_totlag_centerline_FAD(
        Core::LinAlg::Initialization::zero);

    // material triads at collocation points
    std::vector<Core::LinAlg::Matrix<3, 3, FAD>> triad_mat_cp_FAD(BEAM3K_COLLOCATION_POINTS,
        Core::LinAlg::Matrix<3, 3, FAD>(Core::LinAlg::Initialization::zero));

    // Next, we have to set variables for FAD
    set_automatic_differentiation_variables<nnodecl>(disp_totlag_FAD);

    update_nodal_variables<nnodecl, FAD>(
        disp_totlag_FAD, disp_totlag_centerline_FAD, triad_mat_cp_FAD, qrefnew_);

    // Store nodal tangents in class variable
    for (unsigned int i = 0; i < 3; ++i)
    {
      t_[0](i) = disp_totlag_centerline_FAD(3 + i).val();
      t_[1](i) = disp_totlag_centerline_FAD(10 + i).val();
    }


    // pre-computed variables that are later re-used in computation of inertia terms

    // interpolated spin vector variation at Gauss points required for inertia forces:
    // v_theta = v_thetaperp + v_thetapar
    std::vector<Core::LinAlg::Matrix<numdofelement, 3, FAD>> v_theta_gp_FAD;

    // multiplicative rotation increment at GP
    // required for analytic linearization and re-used for analytic mass matrix
    std::vector<Core::LinAlg::Matrix<3, numdofelement, FAD>> lin_theta_gp_FAD;

    // Interpolated material triad at Gauss points
    std::vector<Core::LinAlg::Matrix<3, 3, FAD>> triad_mat_gp_FAD;


    if (weakkirchhoff_)
    {
      calculate_internal_forces_and_stiff_wk<nnodecl, FAD>(params, disp_totlag_centerline_FAD,
          triad_mat_cp_FAD, nullptr, internal_force_FAD, v_theta_gp_FAD, lin_theta_gp_FAD,
          triad_mat_gp_FAD);
    }
    else
    {
      calculate_internal_forces_and_stiff_sk<nnodecl>(params, disp_totlag_centerline_FAD,
          triad_mat_cp_FAD, nullptr, internal_force_FAD, v_theta_gp_FAD, triad_mat_gp_FAD);
    }

    if (rotvec_ == true)
    {
      apply_rot_vec_trafo<nnodecl, FAD>(disp_totlag_centerline_FAD, internal_force_FAD);
    }


    if (force != nullptr)
    {
      for (unsigned int i = 0; i < numdofelement; ++i)
      {
        (*force)(i) = internal_force_FAD(i).val();
      }
    }


    if (stiffmatrix != nullptr)
    {
      // Calculating stiffness matrix with FAD
      for (unsigned int i = 0; i < numdofelement; ++i)
      {
        for (unsigned int j = 0; j < numdofelement; ++j)
        {
          (*stiffmatrix)(i, j) = internal_force_FAD(i).dx(j);
        }
      }

      if (rotvec_ == true)
        transform_stiff_matrix_multiplicative<nnodecl, double>(stiffmatrix, disp_totlag);
    }


    // *************** INERTIA *******************************************
    if (massmatrix != nullptr or force_inert != nullptr)
    {
      Core::LinAlg::Matrix<numdofelement, 1, FAD> inertia_force_FAD(
          Core::LinAlg::Initialization::zero);

      calculate_inertia_forces_and_mass_matrix<nnodecl, FAD>(params, triad_mat_gp_FAD,
          disp_totlag_centerline_FAD, v_theta_gp_FAD, lin_theta_gp_FAD, inertia_force_FAD, nullptr);

      if (rotvec_ == true)
      {
        apply_rot_vec_trafo<nnodecl, FAD>(disp_totlag_centerline_FAD, inertia_force_FAD);
      }

      if (force_inert != nullptr)
      {
        for (unsigned int i = 0; i < numdofelement; ++i)
          (*force_inert)(i) = inertia_force_FAD(i).val();
      }


      if (massmatrix != nullptr)
      {
        // Calculating stiffness matrix with FAD
        for (unsigned int i = 0; i < numdofelement; ++i)
          for (unsigned int j = 0; j < numdofelement; ++j)
            (*massmatrix)(i, j) = inertia_force_FAD(i).dx(j);

        if (rotvec_ == true)
          transform_stiff_matrix_multiplicative<nnodecl, double>(massmatrix, disp_totlag);
      }
    }
  }



  if (massmatrix != nullptr)
  {
    double dt = 1000.0;
    double beta = -1.0;
    double alpha_f = -1.0;
    double alpha_m = -1.0;

    if (this->is_params_interface())
    {
      dt = params_interface().get_delta_time();
      beta = params_interface().get_beam_params_interface_ptr()->get_beta();
      alpha_f = params_interface().get_beam_params_interface_ptr()->get_alphaf();
      alpha_m = params_interface().get_beam_params_interface_ptr()->get_alpham();
    }
    else
    {
      beta = params.get<double>("rot_beta", 1000);
      alpha_f = params.get<double>("rot_alphaf", 1000);
      alpha_m = params.get<double>("rot_alpham", 1000);
      dt = params.get<double>("delta time", 1000);
    }

    // In Lie group GenAlpha algorithm, the mass matrix is multiplied with factor
    // (1.0-alpham_)/(beta_*dt*dt*(1.0-alphaf_)) later. so we apply inverse factor here because the
    // correct prefactors for linearization of displacement/velocity/acceleration dependent terms
    // have been applied automatically by FAD
    massmatrix->scale(beta * dt * dt * (1.0 - alpha_f) / (1.0 - alpha_m));
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void Discret::Elements::Beam3k::calculate_internal_forces_and_stiff_wk(
    Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
        disp_totlag_centerline,
    const std::vector<Core::LinAlg::Matrix<3, 3, T>>& triad_mat_cp,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix,
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& internal_force,
    std::vector<Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, T>>& v_theta_gp,
    std::vector<Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, T>>& lin_theta_gp,
    std::vector<Core::LinAlg::Matrix<3, 3, T>>& triad_mat_gp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Beam3k::calculate_internal_forces_and_stiff_wk");

  if (BEAM3K_COLLOCATION_POINTS != 2 and BEAM3K_COLLOCATION_POINTS != 3 and
      BEAM3K_COLLOCATION_POINTS != 4)
    FOUR_C_THROW("Only the values 2,3 and 4 are valid for BEAM3K_COLLOCATION_POINTS!!!");

  // number of nodes fixed for this element
  const unsigned int nnode = 2;

  const unsigned int numdofelement = 2 * 3 * nnodecl + BEAM3K_COLLOCATION_POINTS;

  // internal force vector
  Core::LinAlg::Matrix<numdofelement, 1, T> f_int_aux(Core::LinAlg::Initialization::zero);


  // CP values of strains and their variations needed for interpolation
  std::vector<T> epsilon_cp(BEAM3K_COLLOCATION_POINTS);  // axial tension
  std::vector<Core::LinAlg::Matrix<numdofelement, 1, T>> v_epsilon_cp(BEAM3K_COLLOCATION_POINTS);
  std::vector<Core::LinAlg::Matrix<numdofelement, 3, T>> v_thetaperp_cp(BEAM3K_COLLOCATION_POINTS);
  std::vector<Core::LinAlg::Matrix<numdofelement, 3, T>> v_thetapar_cp(BEAM3K_COLLOCATION_POINTS);

  // linearization of strain variations at CPs
  std::vector<Core::LinAlg::Matrix<numdofelement, numdofelement, T>> lin_v_epsilon_cp(
      BEAM3K_COLLOCATION_POINTS);

  // Get integration points for exact integration
  Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);



  // re-interpolated values of strains and their variations evaluated at a specific Gauss point
  T epsilon_bar;
  Core::LinAlg::Matrix<numdofelement, 1, T> v_epsilon_bar(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdofelement, 3, T> v_thetaperp_s_bar(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdofelement, 3, T> v_thetapar_s_bar(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdofelement, 3, T> v_theta_s_bar(
      Core::LinAlg::Initialization::zero);  //=v_thetaperp_s_bar+v_thetapar_s_bar


  // interpolated spin vector variation required for inertia forces: v_theta=v_thetaperp+v_thetapar
  v_theta_gp.resize(gausspoints.nquad);

  // CP values of increments lin_theta_cp = lin_theta_perp_cp + lin_theta_par_cp
  // (required for analytic linearization)
  std::vector<Core::LinAlg::Matrix<3, numdofelement, T>> lin_theta_cp(BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<3, numdofelement, T>(Core::LinAlg::Initialization::zero));

  lin_theta_gp.resize(gausspoints.nquad);

  // Further material and spatial strains and forces to be evaluated at a specific Gauss point
  Core::LinAlg::Matrix<3, 1, T> K;      // material curvature
  Core::LinAlg::Matrix<3, 1, T> Omega;  // material deformation measure Omega:=K-K0
  Core::LinAlg::Matrix<3, 1, T> m;      // spatial moment stress resultant
  Core::LinAlg::Matrix<3, 1, T> M;      // material moment stress resultant
  T f_par;                              // material=spatial axial force component


  // Additional kinematic quantities at a specific Gauss point
  Core::LinAlg::Matrix<3, 1, T> r_s;  // vector to store r'
  T abs_r_s;                          // ||r'||


  // Interpolated material triad and angles evaluated at Gauss point
  triad_mat_gp.resize(gausspoints.nquad);
  Core::LinAlg::Matrix<3, 1, T> theta;    // interpolated angle theta
  Core::LinAlg::Matrix<3, 1, T> theta_s;  // derivative of theta with respect to arc-length s


  // matrices storing the assembled shape functions and s-derivatives
  Core::LinAlg::Matrix<3, numdofelement, T> N_s;
  Core::LinAlg::Matrix<1, numdofelement, T> L;

  // matrices storing individual shape functions, its xi-derivatives and s-derivatives
  Core::LinAlg::Matrix<1, 2 * nnode, double> N_i_xi;
  //  Core::LinAlg::Matrix<1,2*nnode,double> N_i_s;
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i;
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i_xi;
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i_s;

  // material constitutive matrices
  Core::LinAlg::Matrix<3, 3, T> Cn, Cm;
  get_constitutive_matrices(Cn, Cm);


  // parameter coordinate
  double xi_cp = 0.0;
  // position index where CP quantities have to be stored
  unsigned int ind = 0;


  // create object of triad interpolation scheme
  std::shared_ptr<
      LargeRotations::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS, T>>
      triad_interpolation_scheme_ptr(
          new LargeRotations::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS,
              T>());

  // reset scheme with nodal triads
  triad_interpolation_scheme_ptr->reset(triad_mat_cp);


  //********begin: evaluate quantities at collocation points********************************
  for (unsigned int inode = 0; inode < BEAM3K_COLLOCATION_POINTS; inode++)
  {
    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    xi_cp = (double)inode / (double)(BEAM3K_COLLOCATION_POINTS - 1) * 2.0 - 1.0;

    // get value of interpolating function of theta (lagrange polynomials) at xi
    L_i.clear();
    Core::FE::shape_function_1d(L_i, xi_cp, shape());

    N_i_xi.clear();
    Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi_cp, length_, Core::FE::CellType::line2);

    // Determine storage position for the node node
    ind = Core::LargeRotations::numbering_trafo(inode + 1, BEAM3K_COLLOCATION_POINTS);

    N_s.clear();
    assemble_shapefunctions_ns(N_i_xi, jacobi_cp_[ind], N_s);

    r_s.clear();
    // Calculation of r' at xi
    r_s.multiply(N_s, disp_totlag_centerline);

    // calculate epsilon at collocation point
    abs_r_s = Core::FADUtils::norm<T>(r_s);
    epsilon_cp[ind] = abs_r_s - 1.0;

    assemble_shapefunctions_l(L_i, L);


    v_epsilon_cp[ind].clear();
    v_epsilon_cp[ind].multiply_tn(N_s, r_s);
    v_epsilon_cp[ind].scale(1.0 / abs_r_s);

    calc_v_thetaperp<nnodecl>(v_thetaperp_cp[ind], N_s, r_s, abs_r_s);

    calc_v_thetapartheta<nnodecl>(v_thetapar_cp[ind], L, r_s, abs_r_s);


    if (stiffmatrix != nullptr)
    {
      pre_compute_terms_at_cp_for_stiffmat_contributions_analytic_wk<nnodecl>(
          lin_theta_cp[ind], lin_v_epsilon_cp[ind], L, N_s, r_s, abs_r_s, qrefconv_[ind]);
    }
  }
  //********end: evaluate quantities at collocation points********************************


  // Clear energy in the beginning
  eint_ = 0.0;

  // re-assure correct size of strain and stress resultant class variables
  axial_strain_gp_.resize(gausspoints.nquad);
  twist_gp_.resize(gausspoints.nquad);
  curvature_2_gp_.resize(gausspoints.nquad);
  curvature_3_gp_.resize(gausspoints.nquad);

  axial_force_gp_.resize(gausspoints.nquad);
  torque_gp_.resize(gausspoints.nquad);
  bending_moment_2_gp_.resize(gausspoints.nquad);
  bending_moment_3_gp_.resize(gausspoints.nquad);


  //******begin: gauss integration for internal force vector and stiffness matrix*********
  for (int numgp = 0; numgp < gausspoints.nquad; numgp++)
  {
    // Get location and weight of GP in parameter space
    const double xi_gp = gausspoints.qxg[numgp][0];
    const double wgt = gausspoints.qwgt[numgp];

    // Evaluate shape functions
    L_i.clear();
    L_i_xi.clear();
    L_i_s.clear();
    Core::FE::shape_function_1d(L_i, xi_gp, shape());
    Core::FE::shape_function_1d_deriv1(L_i_xi, xi_gp, shape());
    L_i_s.update(1.0 / jacobi_[numgp], L_i_xi);


    // Calculate collocation point interpolations ("v"-vectors and epsilon)
    v_epsilon_bar.clear();
    v_thetaperp_s_bar.clear();
    v_thetapar_s_bar.clear();
    v_theta_s_bar.clear();
    epsilon_bar = 0.0;
    theta.clear();
    theta_s.clear();


    triad_interpolation_scheme_ptr->get_interpolated_local_rotation_vector(theta, L_i);

    triad_interpolation_scheme_ptr->get_interpolated_local_rotation_vector_derivative(
        theta_s, L_i_xi, jacobi_[numgp]);


    // re-interpolation of collocation point values
    for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; ++node)
    {
      v_epsilon_bar.update(L_i(node), v_epsilon_cp[node], 1.0);

      v_thetaperp_s_bar.update(L_i_s(node), v_thetaperp_cp[node], 1.0);

      v_thetapar_s_bar.update(L_i_s(node), v_thetapar_cp[node], 1.0);

      epsilon_bar += L_i(node) * epsilon_cp[node];
    }

    v_theta_s_bar.update(1.0, v_thetaperp_s_bar, 1.0);
    v_theta_s_bar.update(1.0, v_thetapar_s_bar, 1.0);


    // "v"-matrix which is required for inertia forces is already calculated here
    // Todo find a nicer and more independent solution here
    v_theta_gp[numgp].clear();
    for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; ++node)
    {
      v_theta_gp[numgp].update(L_i(node), v_thetaperp_cp[node], 1.0);
      v_theta_gp[numgp].update(L_i(node), v_thetapar_cp[node], 1.0);
    }


    // compute material strain K
    K.clear();
    Omega.clear();

    computestrain(theta, theta_s, K);

    for (unsigned int idim = 0; idim < 3; ++idim)
    {
      Omega(idim) = K(idim) - (k0_[numgp])(idim);
    }

    // compute material stress resultants
    M.clear();
    f_par = 0.0;


    straintostress(Omega, epsilon_bar, Cn, Cm, M, f_par);


    // compute material triad at gp
    triad_mat_gp[numgp].clear();

    triad_interpolation_scheme_ptr->get_interpolated_triad(triad_mat_gp[numgp], theta);


    // pushforward of stress resultants
    m.clear();
    m.multiply(triad_mat_gp[numgp], M);


    // residual contribution from moments
    f_int_aux.clear();
    f_int_aux.multiply(v_theta_s_bar, m);
    f_int_aux.scale(wgt * jacobi_[numgp]);

    internal_force.update(1.0, f_int_aux, 1.0);

    // residual contribution from axial force
    f_int_aux.clear();
    f_int_aux.update(1.0, v_epsilon_bar);
    f_int_aux.scale(wgt * jacobi_[numgp] * f_par);

    internal_force.update(1.0, f_int_aux, 1.0);


    if (stiffmatrix != nullptr)
    {
      calculate_stiffmat_contributions_analytic_wk<nnodecl>(*stiffmatrix, disp_totlag_centerline,
          *triad_interpolation_scheme_ptr, v_theta_s_bar, lin_theta_cp, lin_theta_gp[numgp],
          lin_v_epsilon_cp, v_epsilon_bar, f_par, m, Cn(0, 0), Cm, theta, theta_s,
          triad_mat_gp[numgp], xi_gp, jacobi_[numgp], wgt);
    }

    // Calculate internal energy and store it in class variable
    eint_ += 0.5 * Core::FADUtils::cast_to_double(epsilon_bar) *
             Core::FADUtils::cast_to_double(f_par) * wgt * jacobi_[numgp];

    for (unsigned int idim = 0; idim < 3; ++idim)
    {
      eint_ += 0.5 * Core::FADUtils::cast_to_double(Omega(idim)) *
               Core::FADUtils::cast_to_double(M(idim)) * wgt * jacobi_[numgp];
    }

    // store material strain and stress resultant values in class variables
    axial_strain_gp_[numgp] = Core::FADUtils::cast_to_double(epsilon_bar);
    twist_gp_[numgp] = Core::FADUtils::cast_to_double(Omega(0));
    curvature_2_gp_[numgp] = Core::FADUtils::cast_to_double(Omega(1));
    curvature_3_gp_[numgp] = Core::FADUtils::cast_to_double(Omega(2));

    axial_force_gp_[numgp] = Core::FADUtils::cast_to_double(f_par);
    torque_gp_[numgp] = Core::FADUtils::cast_to_double(M(0));
    bending_moment_2_gp_[numgp] = Core::FADUtils::cast_to_double(M(1));
    bending_moment_3_gp_[numgp] = Core::FADUtils::cast_to_double(M(2));
  }
  //******end: gauss integration for internal force vector and stiffness matrix*********
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calculate_stiffmat_contributions_analytic_wk(
    Core::LinAlg::SerialDenseMatrix& stiffmatrix,
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>&
        disp_totlag_centerline,
    const LargeRotations::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS, double>&
        triad_intpol,
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, double>& v_theta_s_bar,
    const std::vector<Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>>&
        lin_theta_cp,
    Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta_bar,
    const std::vector<Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
        6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>>& lin_v_epsilon_cp,
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>& v_epsilon_bar,
    double axial_force_bar, const Core::LinAlg::Matrix<3, 1, double>& moment_resultant,
    double axial_rigidity,
    const Core::LinAlg::Matrix<3, 3, double>& constitutive_matrix_moment_material,
    const Core::LinAlg::Matrix<3, 1, double>& theta_gp,
    const Core::LinAlg::Matrix<3, 1, double>& theta_s_gp,
    const Core::LinAlg::Matrix<3, 3, double>& triad_mat_gp, double xi_gp, double jacobifac_gp,
    double GPwgt) const
{
  // spatial dimension
  const unsigned int ndim = 3;
  // number of values used for centerline interpolation (Hermite: value + derivative)
  const unsigned int vpernode = 2;
  // size of Dof vector of this element
  const unsigned int numdofelement = ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS;

  // create a fixed size matrix as view on the Core::LinAlg::SerialDenseMatrix to avoid copying
  Core::LinAlg::Matrix<numdofelement, numdofelement, double> stiffmatrix_fixedsize(
      stiffmatrix, true);


  // matrices storing the assembled shape functions and s-derivatives
  Core::LinAlg::Matrix<ndim, numdofelement, double> N_s, N_ss;
  Core::LinAlg::Matrix<1, numdofelement, double> L, L_s;

  // matrices storing individual shape functions, its xi-derivatives and s-derivatives
  Core::LinAlg::Matrix<1, vpernode * nnodecl, double> N_i_xi, N_i_s, N_i_xixi;
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i, L_i_xi, L_i_s;

  // r' vector and its norm
  Core::LinAlg::Matrix<3, 1, double> r_s_cp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, double> r_ss_cp(Core::LinAlg::Initialization::zero);
  double abs_r_s_cp = 0.0;

  // first base vector and s-derivative
  Core::LinAlg::Matrix<3, 1, double> g_1_cp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, double> g_1_s_cp(Core::LinAlg::Initialization::zero);


  // re-interpolated lin_theta
  Core::LinAlg::Matrix<ndim, numdofelement, double> lin_theta_s_bar(
      Core::LinAlg::Initialization::zero);

  // linearization of re-interpolated strain variations
  std::vector<Core::LinAlg::Matrix<numdofelement, numdofelement, double>> lin_v_thetaperp_moment_cp(
      BEAM3K_COLLOCATION_POINTS),
      lin_v_thetapar_moment_cp(BEAM3K_COLLOCATION_POINTS);


  Core::LinAlg::Matrix<numdofelement, numdofelement, double> lin_v_thetaperp_s_bar_moment(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdofelement, numdofelement, double> lin_v_thetapar_s_bar_moment(
      Core::LinAlg::Initialization::zero);


  // linearization of re-interpolated strain variations
  Core::LinAlg::Matrix<numdofelement, numdofelement, double> lin_v_epsilon_bar(
      Core::LinAlg::Initialization::zero);


  Core::LinAlg::Matrix<3, 3, double> spinmatrix_of_moment_resultant(
      Core::LinAlg::Initialization::zero);
  Core::LargeRotations::computespin<double>(spinmatrix_of_moment_resultant, moment_resultant);

  // push forward constitutive matrix according to Jelenic 1999, paragraph following to (2.22) on
  // page 148
  Core::LinAlg::Matrix<3, 3, double> constitutive_matrix_moment_spatial(
      Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 3, double> temp(Core::LinAlg::Initialization::zero);
  temp.multiply(triad_mat_gp, constitutive_matrix_moment_material);
  constitutive_matrix_moment_spatial.multiply_nt(temp, triad_mat_gp);


  // linearization of stress resultant (moment)
  Core::LinAlg::Matrix<3, numdofelement, double> lin_moment_resultant(
      Core::LinAlg::Initialization::zero);


  /***********************************************************************************************/
  // note: we need an additional loop over the collocation points here for all quantities that
  //       would be third order tensors if not multiplied by the associated vector (in this case
  //       moment vector); since the vector is only available within the loop over the Gauss points
  //       (i.e. at this current GP), we compute right here the lin_v_theta*_moment terms in an
  //       extra loop over the collocation points

  double xi_cp = 0.0;    // parameter coordinate
  unsigned int ind = 0;  // position index where CP quantities have to be stored

  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    // Determine storage position for this cp
    ind = Core::LargeRotations::numbering_trafo(icp + 1, BEAM3K_COLLOCATION_POINTS);

    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    xi_cp = (double)icp / (double)(BEAM3K_COLLOCATION_POINTS - 1) * 2.0 - 1.0;


    // get all required shape function values
    L_i.clear();
    Core::FE::shape_function_1d(L_i, xi_cp, shape());

    L.clear();
    assemble_shapefunctions_l(L_i, L);

    N_i_xi.clear();
    Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi_cp, length_, Core::FE::CellType::line2);

    N_s.clear();
    assemble_shapefunctions_ns(N_i_xi, jacobi_cp_[ind], N_s);


    // Calculation of r' and g_1
    r_s_cp.clear();
    r_s_cp.multiply(N_s, disp_totlag_centerline);

    abs_r_s_cp = r_s_cp.norm2();  // Todo think about computing and storing inverse value here


    g_1_cp.clear();
    g_1_cp.update(std::pow(abs_r_s_cp, -1.0), r_s_cp);


    calc_lin_v_thetaperp_moment<nnodecl>(
        lin_v_thetaperp_moment_cp[ind], N_s, g_1_cp, abs_r_s_cp, spinmatrix_of_moment_resultant);

    calc_lin_v_thetapar_moment<nnodecl>(
        lin_v_thetapar_moment_cp[ind], L, N_s, g_1_cp, abs_r_s_cp, moment_resultant);
  }


  /***********************************************************************************************/
  // re-interpolation of quantities at xi based on CP values

  L_i.clear();
  Core::FE::shape_function_1d(L_i, xi_gp, shape());

  L_i_xi.clear();
  Core::FE::shape_function_1d_deriv1(L_i_xi, xi_gp, shape());

  L_i_s.clear();
  L_i_s.update(std::pow(jacobifac_gp, -1.0), L_i_xi, 0.0);

  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    lin_v_epsilon_bar.update(L_i(icp), lin_v_epsilon_cp[icp], 1.0);

    lin_v_thetaperp_s_bar_moment.update(L_i_s(icp), lin_v_thetaperp_moment_cp[icp], 1.0);
    lin_v_thetapar_s_bar_moment.update(L_i_s(icp), lin_v_thetapar_moment_cp[icp], 1.0);
  }

  // compute Itilde(_s) matrices required for re-interpolation of CP values of lin_theta
  std::vector<Core::LinAlg::Matrix<3, 3, double>> Itilde(BEAM3K_COLLOCATION_POINTS);
  std::vector<Core::LinAlg::Matrix<3, 3, double>> Itilde_s(BEAM3K_COLLOCATION_POINTS);

  triad_intpol.get_nodal_generalized_rotation_interpolation_matrices(Itilde, theta_gp, L_i);

  triad_intpol.get_nodal_generalized_rotation_interpolation_matrices_derivative(
      Itilde_s, theta_gp, theta_s_gp, L_i, L_i_s);


  Core::LinAlg::Matrix<3, numdofelement, double> auxmatrix(Core::LinAlg::Initialization::zero);

  lin_theta_bar.clear();
  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    auxmatrix.clear();

    auxmatrix.multiply(Itilde[icp], lin_theta_cp[icp]);

    lin_theta_bar.update(1.0, auxmatrix, 1.0);

    auxmatrix.clear();

    auxmatrix.multiply(Itilde_s[icp], lin_theta_cp[icp]);

    lin_theta_s_bar.update(1.0, auxmatrix, 1.0);
  }


  calc_lin_moment_resultant<nnodecl>(lin_moment_resultant, lin_theta_bar, lin_theta_s_bar,
      spinmatrix_of_moment_resultant, constitutive_matrix_moment_spatial);

  /***********************************************************************************************/
  // finally put everything together

  // constant pre-factor
  const double jacobifac_GPwgt = jacobifac_gp * GPwgt;

  Core::LinAlg::Matrix<numdofelement, numdofelement, double> auxmatrix2(
      Core::LinAlg::Initialization::zero);


  // linearization of the residual contributions from moments
  stiffmatrix_fixedsize.update(jacobifac_GPwgt, lin_v_thetaperp_s_bar_moment, 1.0);

  stiffmatrix_fixedsize.update(jacobifac_GPwgt, lin_v_thetapar_s_bar_moment, 1.0);

  auxmatrix2.clear();
  auxmatrix2.multiply(v_theta_s_bar, lin_moment_resultant);
  stiffmatrix_fixedsize.update(jacobifac_GPwgt, auxmatrix2, 1.0);


  // linearization of the residual contributions from axial force
  auxmatrix2.clear();
  for (unsigned int idof = 0; idof < numdofelement; ++idof)
    for (unsigned int jdof = 0; jdof < numdofelement; ++jdof)
      auxmatrix2(idof, jdof) = v_epsilon_bar(idof) * v_epsilon_bar(jdof);

  stiffmatrix_fixedsize.update(axial_rigidity * jacobifac_GPwgt, auxmatrix2, 1.0);

  stiffmatrix_fixedsize.update(axial_force_bar * jacobifac_GPwgt, lin_v_epsilon_bar, 1.0);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::pre_compute_terms_at_cp_for_stiffmat_contributions_analytic_wk(
    Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta,
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
        6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_v_epsilon,
    const Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& L,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const Core::LinAlg::Matrix<3, 1, double>& r_s, double abs_r_s,
    const Core::LinAlg::Matrix<4, 1, double>& Qref_conv) const
{
  // spatial dimension
  const unsigned int ndim = 3;
  // number of values used for centerline interpolation (Hermite: value + derivative)
  const unsigned int vpernode = 2;
  // size of Dof vector of this element
  const unsigned int numdofelement = ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS;


  Core::LinAlg::Matrix<ndim, 1, double> g_1(Core::LinAlg::Initialization::zero);
  g_1.update(std::pow(abs_r_s, -1.0), r_s);

  Core::LinAlg::Matrix<ndim, 1, double> g_1_bar(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 3, double> triad_ref_conv_cp(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::quaterniontotriad(Qref_conv, triad_ref_conv_cp);

  g_1_bar.clear();
  for (unsigned int idim = 0; idim < ndim; ++idim) g_1_bar(idim) = triad_ref_conv_cp(idim, 0);


  // CP values of strain increments
  Core::LinAlg::Matrix<ndim, numdofelement, double> lin_theta_perp(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<ndim, numdofelement, double> lin_theta_par(
      Core::LinAlg::Initialization::zero);

  calc_lin_thetapar<nnodecl>(lin_theta_par, L, N_s, g_1, g_1_bar, abs_r_s);

  calc_lin_thetaperp<nnodecl>(lin_theta_perp, N_s, r_s, abs_r_s);

  // lin_theta
  lin_theta.clear();
  lin_theta.update(1.0, lin_theta_par, 1.0, lin_theta_perp);

  calc_lin_v_epsilon<nnodecl>(lin_v_epsilon, N_s, g_1, abs_r_s);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calculate_internal_forces_and_stiff_sk(
    Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>&
        disp_totlag_centerline,
    const std::vector<Core::LinAlg::Matrix<3, 3, FAD>>& triad_mat_cp,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix,
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>& internal_force,
    std::vector<Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD>>& v_theta_gp,
    std::vector<Core::LinAlg::Matrix<3, 3, FAD>>& triad_mat_gp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Beam3k::calculate_internal_forces_and_stiff_sk");

  if (BEAM3K_COLLOCATION_POINTS != 2 and BEAM3K_COLLOCATION_POINTS != 3 and
      BEAM3K_COLLOCATION_POINTS != 4)
  {
    FOUR_C_THROW("Only the values 2,3 and 4 are valid for BEAM3K_COLLOCATION_POINTS!!!");
  }

  // internal force vector
  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> f_int_aux(
      Core::LinAlg::Initialization::zero);

  // CP values of strains and their variations needed for interpolation
  std::vector<FAD> epsilon_cp(
      BEAM3K_COLLOCATION_POINTS);  // axial tension epsilon=|r_s|-1 at collocation points
  std::vector<Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>> v_epsilon_cp(
      BEAM3K_COLLOCATION_POINTS);

  // Get integration points for exact integration
  Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);

  // interpolated values of strains and their variations evaluated at Gauss points
  FAD epsilon;
  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> v_epsilon(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD> v_thetaperp_s(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD> v_thetapartheta_s(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD> v_theta_s(
      Core::LinAlg::Initialization::zero);  //=v_thetaperp_s+v_thetapartheta_s(+v_thetapard_s)

  // interpolated spin vector variation required for inertia forces:
  // v_theta=v_thetaperp+v_thetapartheta(+v_thetapard)
  v_theta_gp.resize(gausspoints.nquad);

  // Further material and spatial strains and forces to be evaluated at the Gauss points
  Core::LinAlg::Matrix<3, 1, FAD> K(Core::LinAlg::Initialization::zero);  // material curvature
  Core::LinAlg::Matrix<3, 1, FAD> Omega(
      Core::LinAlg::Initialization::zero);  // material deformation measure Omega:=K-K0
  Core::LinAlg::Matrix<3, 1, FAD> m(
      Core::LinAlg::Initialization::zero);  // spatial moment stress resultant
  Core::LinAlg::Matrix<3, 1, FAD> M(
      Core::LinAlg::Initialization::zero);  // material moment stress resultant
  FAD f_par = 0.0;                          // material=spatial axial force component

  // Triads at collocation points
  std::vector<FAD> phi_cp(BEAM3K_COLLOCATION_POINTS, 0.0);  // relative angle at collocation points

  // Interpolated material triad and angles evaluated at Gauss point
  triad_mat_gp.resize(gausspoints.nquad);
  FAD phi = 0.0;    // interpolated relative angle phi
  FAD phi_s = 0.0;  // derivative of interpolated relative angle phi with respect to arc-length s

  // matrices holding the assembled shape functions and s-derivatives
  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD> N_s(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD> N_ss(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD> L(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD> L_s(
      Core::LinAlg::Initialization::zero);

  // Matrices for individual shape functions and xi-derivatives
  Core::LinAlg::Matrix<1, 2 * nnodecl, FAD> N_i_xi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 2 * nnodecl, FAD> N_i_xixi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, FAD> L_i(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, FAD> L_i_xi(
      Core::LinAlg::Initialization::zero);

  // Matrices for individual s-derivatives
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, FAD> L_i_s(Core::LinAlg::Initialization::zero);

  // Additional kinematic quantities
  Core::LinAlg::Matrix<3, 1, FAD> r_s(Core::LinAlg::Initialization::zero);   // Matrix to store r'
  Core::LinAlg::Matrix<3, 1, FAD> r_ss(Core::LinAlg::Initialization::zero);  // Matrix to store r''
  Core::LinAlg::Matrix<3, 1, FAD> g1(Core::LinAlg::Initialization::zero);    // g1:=r'/||r'||
  Core::LinAlg::Matrix<3, 1, FAD> g1_s(Core::LinAlg::Initialization::zero);  // g1'
  Core::LinAlg::Matrix<3, 1, FAD> ttilde(
      Core::LinAlg::Initialization::zero);  //\tilde{t}:=g1/||r'||=r'/||r'||^2
  Core::LinAlg::Matrix<3, 1, FAD> ttilde_s(Core::LinAlg::Initialization::zero);  //\tilde{t}'
  Core::LinAlg::Matrix<3, 1, FAD> kappacl(
      Core::LinAlg::Initialization::zero);  // centerline (cl) curvature vector
  FAD abs_r_s = 0.0;                        // ||r'||
  FAD rsTrss = 0.0;                         // r'^Tr''
  Core::LinAlg::Matrix<3, 3, FAD> auxmatrix1(
      Core::LinAlg::Initialization::zero);  // auxiliary matrix
  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD> auxmatrix2(
      Core::LinAlg::Initialization::zero);  // auxiliary matrix

#ifdef CONSISTENTSPINSK
  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD> v_thetapard_s(
      Core::LinAlg::Initialization::zero);
  std::vector<Core::LinAlg::Matrix<3, 1, FAD>> g1_cp(BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<3, 1, FAD>(Core::LinAlg::Initialization::zero));
  std::vector<Core::LinAlg::Matrix<3, 1, FAD>> ttilde_cp(BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<3, 1, FAD>(Core::LinAlg::Initialization::zero));
  std::vector<Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD>> N_s_cp(
      BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD>(
          Core::LinAlg::Initialization::zero));
  std::vector<Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>> v1_cp(
      BEAM3K_COLLOCATION_POINTS,
      Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>(
          Core::LinAlg::Initialization::zero));
  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> v1(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> v1_s(
      Core::LinAlg::Initialization::zero);
#endif

  // MISC
  double xi = 0.0;       // parameter coordinated
  unsigned int ind = 0;  // position index where CP quantities have to be stored

  // material constitutive matrices
  Core::LinAlg::Matrix<3, 3, FAD> Cn, Cm;
  get_constitutive_matrices(Cn, Cm);


  //********begin: evaluate quantities at collocation points********************************
  for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; node++)
  {
    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    xi = (double)node / (BEAM3K_COLLOCATION_POINTS - 1) * 2 - 1.0;

    // get value of interpolating function of theta (lagrange polynomials) at xi
    N_i_xi.clear();
    Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi, length_, Core::FE::CellType::line2);

    // Determine storage position for the node node
    ind = Core::LargeRotations::numbering_trafo(node + 1, BEAM3K_COLLOCATION_POINTS);

    N_s.clear();
    assemble_shapefunctions_ns(N_i_xi, jacobi_cp_[ind], N_s);
    r_s.clear();
    // Calculation of r' at xi
    r_s.multiply(N_s, disp_totlag_centerline);

    // calculate epsilon at collocation point
    abs_r_s = Core::FADUtils::norm<FAD>(r_s);
    epsilon_cp[ind] = abs_r_s - 1.0;

    v_epsilon_cp[ind].clear();
    v_epsilon_cp[ind].multiply_tn(N_s, r_s);
    v_epsilon_cp[ind].scale(1.0 / abs_r_s);

#ifdef CONSISTENTSPINSK
    N_s_cp[ind].update(1.0, N_s, 0.0);
    g1_cp[ind].update(1.0 / abs_r_s, r_s, 0.0);
    ttilde_cp[ind].update(1.0 / (abs_r_s * abs_r_s), r_s, 0.0);
#endif

  }  // for (int node=0; node<BEAM3K_COLLOCATION_POINTS; node++)

  // calculate angle at cp (this has to be done in a SEPARATE loop as follows)
  for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; node++)
  {
    Core::LinAlg::Matrix<3, 3, FAD> Lambdabarref(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, FAD> tangentref(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, FAD> phivec(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; i++)
    {
      tangentref(i) = triad_mat_cp[node](i, 0);
    }
    Core::LargeRotations::calculate_sr_triads<FAD>(
        tangentref, triad_mat_cp[REFERENCE_NODE], Lambdabarref);
    Core::LargeRotations::triadtoangleleft(phivec, Lambdabarref, triad_mat_cp[node]);
    phi_cp[node] = 0.0;
    for (unsigned int i = 0; i < 3; i++)
    {
      phi_cp[node] += tangentref(i) * phivec(i);
    }

#ifdef CONSISTENTSPINSK
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> auxmatrix3(
        Core::LinAlg::Initialization::zero);
    compute_triple_product<6 * nnodecl + BEAM3K_COLLOCATION_POINTS>(
        N_s_cp[node], g1_cp[REFERENCE_NODE], ttilde_cp[node], auxmatrix3);
    v1_cp[node].update(1.0, auxmatrix3, 0.0);
    auxmatrix3.clear();
    compute_triple_product<6 * nnodecl + BEAM3K_COLLOCATION_POINTS>(
        N_s_cp[REFERENCE_NODE], g1_cp[node], ttilde_cp[REFERENCE_NODE], auxmatrix3);
    v1_cp[node].update(-1.0, auxmatrix3, 1.0);
    Core::LinAlg::Matrix<1, 1, FAD> auxscalar(Core::LinAlg::Initialization::zero);
    auxscalar.multiply_tn(g1_cp[node], g1_cp[REFERENCE_NODE]);
    v1_cp[node].scale(1.0 / (1.0 + auxscalar(0, 0)));
#endif
  }
  //********end: evaluate quantities at collocation points********************************

  // Clear energy in the beginning
  eint_ = 0.0;

  // re-assure correct size of strain and stress resultant class variables
  axial_strain_gp_.resize(gausspoints.nquad);
  twist_gp_.resize(gausspoints.nquad);
  curvature_2_gp_.resize(gausspoints.nquad);
  curvature_3_gp_.resize(gausspoints.nquad);

  axial_force_gp_.resize(gausspoints.nquad);
  torque_gp_.resize(gausspoints.nquad);
  bending_moment_2_gp_.resize(gausspoints.nquad);
  bending_moment_3_gp_.resize(gausspoints.nquad);


  //******begin: gauss integration for internal force vector and stiffness matrix*********
  for (int numgp = 0; numgp < gausspoints.nquad; numgp++)
  {
    // Get location and weight of GP in parameter space
    const double xi = gausspoints.qxg[numgp][0];
    const double wgt = gausspoints.qwgt[numgp];

    // Evaluate and assemble shape functions
    L_i.clear();
    L_i_xi.clear();
    L_i_s.clear();
    L.clear();
    L_s.clear();
    N_i_xi.clear();
    N_i_xixi.clear();
    N_s.clear();
    N_ss.clear();
    Core::FE::shape_function_1d(L_i, xi, shape());
    Core::FE::shape_function_1d_deriv1(L_i_xi, xi, shape());
    L_i_s.update(1.0 / jacobi_[numgp], L_i_xi, 0.0);
    assemble_shapefunctions_l(L_i, L);
    // The assemble routine is identical for L and L_s
    assemble_shapefunctions_l(L_i_s, L_s);
    Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi, length_, Core::FE::CellType::line2);
    assemble_shapefunctions_ns(N_i_xi, jacobi_[numgp], N_s);
    Core::FE::shape_function_hermite_1d_deriv2(N_i_xixi, xi, length_, Core::FE::CellType::line2);
    assemble_shapefunctions_nss(N_i_xi, N_i_xixi, jacobi_[numgp], jacobi2_[numgp], N_ss);

    // Calculate collocation point interpolations
    v_epsilon.clear();
    epsilon = 0.0;
    phi = 0.0;
    phi_s = 0.0;
    for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; node++)
    {
      // calculate interpolated axial tension and variation
      v_epsilon.update(L_i(node), v_epsilon_cp[node], 1.0);
      epsilon += L_i(node) * epsilon_cp[node];
      // calculate interpolated relative angle
      phi += L_i(node) * phi_cp[node];
      phi_s += L_i_s(node) * phi_cp[node];
    }

    // Calculation of r' and r'' at xi
    r_s.clear();
    r_s.multiply(N_s, disp_totlag_centerline);
    r_ss.clear();
    r_ss.multiply(N_ss, disp_totlag_centerline);

    //*****************************************************************************************************************************
    //************************Begin: Determine "v"-vectors representing the discrete strain
    // variations*****************************
    //*****************************************************************************************************************************
    // Auxiliary quantities
    abs_r_s = 0.0;
    rsTrss = 0.0;
    abs_r_s = Core::FADUtils::norm<FAD>(r_s);
    for (unsigned int i = 0; i < 3; i++)
    {
      rsTrss += r_s(i) * r_ss(i);
    }
    g1.clear();
    g1_s.clear();
    g1.update(1.0 / abs_r_s, r_s, 0.0);
    g1_s.update(1.0 / abs_r_s, r_ss, 0.0);
    g1_s.update(-rsTrss / (abs_r_s * abs_r_s * abs_r_s), r_s, 1.0);
    ttilde.clear();
    ttilde_s.clear();
    ttilde.update(1.0 / (abs_r_s * abs_r_s), r_s, 0.0);
    ttilde_s.update(1.0 / (abs_r_s * abs_r_s), r_ss, 0.0);
    ttilde_s.update(-2 * rsTrss / (abs_r_s * abs_r_s * abs_r_s * abs_r_s), r_s, 1.0);

    //************** I) Compute v_theta_s=v_thetaperp_s+v_thetapartheta_s(+v_thetapard_s)
    //*****************************************
    // I a) Compute v_thetapartheta_s
    v_thetapartheta_s.clear();
    for (unsigned int row = 0; row < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; row++)
    {
      for (unsigned int column = 0; column < 3; column++)
      {
        v_thetapartheta_s(row, column) += L_s(0, row) * g1(column, 0) + L(0, row) * g1_s(column, 0);
      }
    }

    // I b) Compute v_thetaperp_s
    v_thetaperp_s.clear();
    auxmatrix1.clear();
    auxmatrix2.clear();
    Core::LargeRotations::computespin(auxmatrix1, ttilde);
    auxmatrix2.multiply_tn(N_ss, auxmatrix1);
    v_thetaperp_s.update(-1.0, auxmatrix2, 0.0);
    auxmatrix1.clear();
    auxmatrix2.clear();
    Core::LargeRotations::computespin(auxmatrix1, ttilde_s);
    auxmatrix2.multiply_tn(N_s, auxmatrix1);
    v_thetaperp_s.update(-1.0, auxmatrix2, 1.0);

    // I c) Calculate sum v_theta_s=v_thetaperp_s+v_thetapartheta_s
    v_theta_s.clear();
    v_theta_s.update(1.0, v_thetaperp_s, 1.0);
    v_theta_s.update(1.0, v_thetapartheta_s, 1.0);

    //************** II) Compute v_theta=v_thetaperp_+v_thetapartheta_(+v_thetapard_)  which is
    // required for inertia forces ********
    // II a) v_thetapartheta contribution
    v_theta_gp[numgp].clear();
    for (unsigned int row = 0; row < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; row++)
    {
      for (unsigned int column = 0; column < 3; column++)
      {
        v_theta_gp[numgp](row, column) += L(0, row) * g1(column, 0);
      }
    }

    // II b) Compute v_thetaperp contribution
    auxmatrix1.clear();
    auxmatrix2.clear();
    Core::LargeRotations::computespin(auxmatrix1, ttilde);
    auxmatrix2.multiply_tn(N_s, auxmatrix1);
    v_theta_gp[numgp].update(-1.0, auxmatrix2, 1.0);


// Compute contributions stemming from CONSISTENTSPINSK
#ifdef CONSISTENTSPINSK
    //************** to I) Compute v_thetapard_s of
    // v_theta_s=v_thetaperp_s+v_thetapartheta_s(+v_thetapard_s) *******************
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> auxmatrix3(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<1, 1, FAD> auxscalar1(Core::LinAlg::Initialization::zero);

    // Calculate v1:
    v1.clear();
    auxmatrix3.clear();
    compute_triple_product<6 * nnodecl + BEAM3K_COLLOCATION_POINTS>(
        N_s, g1_cp[REFERENCE_NODE], ttilde, auxmatrix3);
    v1.update(1.0, auxmatrix3, 0.0);
    auxmatrix3.clear();
    compute_triple_product<6 * nnodecl + BEAM3K_COLLOCATION_POINTS>(
        N_s_cp[REFERENCE_NODE], g1, ttilde_cp[REFERENCE_NODE], auxmatrix3);
    v1.update(-1.0, auxmatrix3, 1.0);
    auxscalar1.clear();
    auxscalar1.multiply_tn(g1, g1_cp[REFERENCE_NODE]);
    v1.scale(1.0 / (1.0 + auxscalar1(0, 0)));

    // Calculate v1_s:
    v1_s.clear();
    auxmatrix3.clear();
    compute_triple_product<6 * nnodecl + BEAM3K_COLLOCATION_POINTS>(
        N_s, g1_cp[REFERENCE_NODE], ttilde_s, auxmatrix3);
    v1_s.update(1.0, auxmatrix3, 0.0);
    auxmatrix3.clear();
    compute_triple_product<6 * nnodecl + BEAM3K_COLLOCATION_POINTS>(
        N_ss, g1_cp[REFERENCE_NODE], ttilde, auxmatrix3);
    v1_s.update(1.0, auxmatrix3, 1.0);
    auxmatrix3.clear();
    compute_triple_product<6 * nnodecl + BEAM3K_COLLOCATION_POINTS>(
        N_s_cp[REFERENCE_NODE], g1_s, ttilde_cp[REFERENCE_NODE], auxmatrix3);
    v1_s.update(-1.0, auxmatrix3, 1.0);
    auxscalar1.clear();
    auxscalar1.multiply_tn(g1_s, g1_cp[REFERENCE_NODE]);
    v1_s.update(-auxscalar1(0, 0), v1, 1.0);
    auxscalar1.clear();
    auxscalar1.multiply_tn(g1, g1_cp[REFERENCE_NODE]);
    v1_s.scale(1.0 / (1.0 + auxscalar1(0, 0)));

    // Calculate vec1 and vec2
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> vec1(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> vec2(
        Core::LinAlg::Initialization::zero);
    for (int node = 0; node < BEAM3K_COLLOCATION_POINTS; node++)
    {
      vec1.update(L_i_s(node), v1_cp[node], 1.0);
      vec2.update(L_i(node), v1_cp[node], 1.0);
    }
    vec1.update(-1.0, v1_s, 1.0);
    vec2.update(-1.0, v1, 1.0);

    // Compute v_thetapard_s
    v_thetapard_s.clear();
    for (int i = 0; i < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        v_thetapard_s(i, j) += vec1(i) * g1(j) + vec2(i) * g1_s(j);
      }
    }
    // I d) Add v_thetapard_s contribution according to v_theta_s+=v_thetapard_s
    v_theta_s.update(1.0, v_thetapard_s, 1.0);

    // to II)  Compute v_thetapard_ of v_theta=v_thetaperp_+v_thetapartheta_(+v_thetapard_)  which
    // is required for inertia forces******* II c) v_thetapard_ contribution

    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> vec3(
        Core::LinAlg::Initialization::zero);
    for (int node = 0; node < BEAM3K_COLLOCATION_POINTS; node++)
    {
      vec3.update(L_i(node), v1_cp[node], 1.0);
    }
    vec3.update(-1.0, v1, 1.0);
    for (int i = 0; i < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        v_theta_gp[numgp](i, j) += vec3(i) * g1(j);
      }
    }
#endif
    //***************************************************************************************************************************
    //************************End: Determine "v"-vectors representing the discrete strain
    // variations*****************************
    //***************************************************************************************************************************

    // Compute material triad and centerline curvature at Gauss point
    triad_mat_gp[numgp].clear();
    compute_triad_sk(phi, r_s, triad_mat_cp[REFERENCE_NODE], triad_mat_gp[numgp]);
    kappacl.clear();
    calculate_clcurvature(r_s, r_ss, kappacl);

    // compute material strain K at Gauss point
    K.clear();
    Omega.clear();
    computestrain_sk(phi_s, kappacl, triad_mat_cp[REFERENCE_NODE], triad_mat_gp[numgp], K);
    for (unsigned int i = 0; i < 3; i++)
    {
      Omega(i) = K(i) - k0_[numgp](i);
    }

    // compute material stress resultants at Gauss point
    M.clear();
    f_par = 0.0;
    straintostress(Omega, epsilon, Cn, Cm, M, f_par);

    // Calculate internal energy and store it in class variable
    eint_ += 0.5 * epsilon.val() * f_par.val() * wgt * jacobi_[numgp];
    for (unsigned int i = 0; i < 3; i++)
    {
      eint_ += 0.5 * Omega(i).val() * M(i).val() * wgt * jacobi_[numgp];
    }

    // pushforward of stress resultants
    m.clear();
    m.multiply(triad_mat_gp[numgp], M);

    // residual contribution from moments
    f_int_aux.clear();
    f_int_aux.multiply(v_theta_s, m);
    f_int_aux.scale(wgt * jacobi_[numgp]);
    internal_force.update(1.0, f_int_aux, 1.0);

    // residual contribution from axial forces
    f_int_aux.clear();
    f_int_aux.update(1.0, v_epsilon, 0.0);
    f_int_aux.scale(wgt * jacobi_[numgp] * f_par);
    internal_force.update(1.0, f_int_aux, 1.0);


    // store material strain and stress resultant values in class variables
    axial_strain_gp_[numgp] = Core::FADUtils::cast_to_double(epsilon);
    twist_gp_[numgp] = Core::FADUtils::cast_to_double(Omega(0));
    curvature_2_gp_[numgp] = Core::FADUtils::cast_to_double(Omega(1));
    curvature_3_gp_[numgp] = Core::FADUtils::cast_to_double(Omega(2));

    axial_force_gp_[numgp] = Core::FADUtils::cast_to_double(f_par);
    torque_gp_[numgp] = Core::FADUtils::cast_to_double(M(0));
    bending_moment_2_gp_[numgp] = Core::FADUtils::cast_to_double(M(1));
    bending_moment_3_gp_[numgp] = Core::FADUtils::cast_to_double(M(2));
  }
  //******end: gauss integration for internal force vector and stiffness matrix*********
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void Discret::Elements::Beam3k::calculate_inertia_forces_and_mass_matrix(
    Teuchos::ParameterList& params, const std::vector<Core::LinAlg::Matrix<3, 3, T>>& triad_mat_gp,
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
        disp_totlag_centerline,
    const std::vector<Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, T>>&
        v_theta_gp,
    const std::vector<Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, T>>&
        lin_theta_gp,
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& f_inert,
    Core::LinAlg::SerialDenseMatrix* massmatrix)
{
  // number of values used for centerline interpolation (Hermite: value + derivative)
  const unsigned int vpernode = 2;
  // size of Dof vector of this element
  const unsigned int numdofelement = 3 * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS;

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

  double dt = 1000.0;
  double beta = -1.0;
  double gamma = -1.0;
  double alpha_f = -1.0;
  double alpha_m = -1.0;

  if (this->is_params_interface())
  {
    dt = params_interface().get_delta_time();
    beta = params_interface().get_beam_params_interface_ptr()->get_beta();
    gamma = params_interface().get_beam_params_interface_ptr()->get_gamma();
    alpha_f = params_interface().get_beam_params_interface_ptr()->get_alphaf();
    alpha_m = params_interface().get_beam_params_interface_ptr()->get_alpham();
  }
  else
  {
    beta = params.get<double>("rot_beta", 1000);
    gamma = params.get<double>("rot_gamma", 1000);
    alpha_f = params.get<double>("rot_alphaf", 1000);
    alpha_m = params.get<double>("rot_alpham", 1000);
    dt = params.get<double>("delta time", 1000);
  }


  // tensor of mass moments of inertia for translational and rotational motion
  double mass_inertia_translational = 0.0;
  Core::LinAlg::Matrix<3, 3, T> Jp(Core::LinAlg::Initialization::zero);

  get_translational_and_rotational_mass_inertia_tensor(mass_inertia_translational, Jp);


  Core::LinAlg::Matrix<3, numdofelement, T> N(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 2 * nnodecl, double> N_i(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> rnewmass(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, T> triad_mat_old(Core::LinAlg::Initialization::zero);

  // auxiliary internal force vector
  Core::LinAlg::Matrix<numdofelement, 1, T> f_inert_aux(Core::LinAlg::Initialization::zero);
  //  Core::LinAlg::Matrix<numdofelement,3,T> v_thetaperp(true);  // Todo unused?
  //  Core::LinAlg::Matrix<numdofelement,3,T> v_thetapar(true);

  // Get integration points for exact integration
  Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);

  ekin_ = 0.0;

  // loop through Gauss points
  for (int numgp = 0; numgp < gausspoints.nquad; ++numgp)
  {
    // get location and weight of GP in parameter space
    const double xi = gausspoints.qxg[numgp][0];
    const double wgt = gausspoints.qwgt[numgp];

    // get shape function values and assemble them
    N_i.clear();
    Core::FE::shape_function_hermite_1d(N_i, xi, length_, Core::FE::CellType::line2);

    N.clear();
    assemble_shapefunctions_n(N_i, N);


    // calculation of centroid position at this gp in current state
    rnewmass.clear();
    rnewmass.multiply(N, disp_totlag_centerline);


    // get quaternion in converged state at gp and compute corresponding triad
    triad_mat_old.clear();
    Core::LinAlg::Matrix<4, 1, T> Qconv(Core::LinAlg::Initialization::zero);
    for (unsigned int i = 0; i < 4; ++i) Qconv(i) = (qconvmass_[numgp])(i);

    Core::LargeRotations::quaterniontotriad(Qconv, triad_mat_old);

    // compute quaternion of relative rotation from converged to current state
    Core::LinAlg::Matrix<3, 3, T> deltatriad(Core::LinAlg::Initialization::zero);
    deltatriad.multiply_nt(triad_mat_gp[numgp], triad_mat_old);
    Core::LinAlg::Matrix<4, 1, T> deltaQ(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::triadtoquaternion(deltatriad, deltaQ);
    Core::LinAlg::Matrix<3, 1, T> deltatheta(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::quaterniontoangle(deltaQ, deltatheta);

    // compute material counterparts of spatial vectors
    Core::LinAlg::Matrix<3, 1, T> deltaTHETA(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, T> Wconvmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, T> Aconvmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, T> Amodconvmass(Core::LinAlg::Initialization::zero);

    deltaTHETA.multiply_tn(triad_mat_gp[numgp], deltatheta);

    Core::LinAlg::Matrix<3, 1, T> auxvector(Core::LinAlg::Initialization::zero);
    for (unsigned int i = 0; i < 3; ++i) auxvector(i) = (wconvmass_[numgp])(i);
    Wconvmass.multiply_tn(triad_mat_old, auxvector);

    for (unsigned int i = 0; i < 3; ++i) auxvector(i) = (aconvmass_[numgp])(i);
    Aconvmass.multiply_tn(triad_mat_old, auxvector);

    for (unsigned int i = 0; i < 3; ++i) auxvector(i) = (amodconvmass_[numgp])(i);
    Amodconvmass.multiply_tn(triad_mat_old, auxvector);

    Core::LinAlg::Matrix<3, 1, T> deltar(Core::LinAlg::Initialization::zero);
    for (unsigned int i = 0; i < 3; ++i) deltar(i) = rnewmass(i) - rconvmass_[numgp](i);

    Core::LinAlg::Matrix<3, 1, T> Anewmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, T> Wnewmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, T> Amodnewmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, T> rttnewmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, T> rtnewmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, T> rttmodnewmass(Core::LinAlg::Initialization::zero);

    /* update angular velocities and accelerations according to Newmark time integration scheme in
     * material description (see Jelenic, 1999, p. 146, equations (2.8) and (2.9)).
     * The corresponding equations are adapted according to the gen-alpha Lie group time
     * integration scheme proposed in [Arnold, Bruels (2007)], [Bruels, Cardona, 2010] and
     * [Bruels, Cardona, Arnold (2012)].
     * In the predictor step of the time integration the following formulas automatically
     * deliver a constant displacement (deltatheta=0), consistent velocity and consistent
     * acceleration predictor. This fact has to be reflected in a consistent manner by
     * the choice of the predictor in the input file: */
    const double lin_prefactor_acc = (1.0 - alpha_m) / (beta * dt * dt * (1.0 - alpha_f));
    const double lin_prefactor_vel = gamma / (beta * dt);

    for (unsigned int i = 0; i < 3; ++i)
    {
      Anewmass(i) =
          lin_prefactor_acc * deltaTHETA(i) -
          (1.0 - alpha_m) / (beta * dt * (1.0 - alpha_f)) * Wconvmass(i) -
          alpha_f / (1.0 - alpha_f) * Aconvmass(i) +
          (alpha_m / (1.0 - alpha_f) - (0.5 - beta) * (1.0 - alpha_m) / (beta * (1.0 - alpha_f))) *
              Amodconvmass(i);

      Wnewmass(i) = lin_prefactor_vel * deltaTHETA(i) + (1 - gamma / beta) * Wconvmass(i) +
                    dt * (1 - gamma / (2 * beta)) * Amodconvmass(i);

      Amodnewmass(i) =
          1.0 / (1.0 - alpha_m) *
          ((1.0 - alpha_f) * Anewmass(i) + alpha_f * Aconvmass(i) - alpha_m * Amodconvmass(i));
    }

    for (unsigned int i = 0; i < 3; ++i)
    {
      rttnewmass(i) =
          lin_prefactor_acc * deltar(i) -
          (1.0 - alpha_m) / (beta * dt * (1.0 - alpha_f)) * rtconvmass_[numgp](i) -
          alpha_f / (1.0 - alpha_f) * rttconvmass_[numgp](i) +
          (alpha_m / (1.0 - alpha_f) - (0.5 - beta) * (1.0 - alpha_m) / (beta * (1.0 - alpha_f))) *
              rttmodconvmass_[numgp](i);

      rtnewmass(i) = lin_prefactor_vel * deltar(i) + (1 - gamma / beta) * rtconvmass_[numgp](i) +
                     dt * (1 - gamma / (2 * beta)) * rttmodconvmass_[numgp](i);

      rttmodnewmass(i) = 1.0 / (1.0 - alpha_m) *
                         ((1.0 - alpha_f) * rttnewmass(i) + alpha_f * rttconvmass_[numgp](i) -
                             alpha_m * rttmodconvmass_[numgp](i));
    }

    // spin matrix of the material angular velocity, i.e. S(W)
    Core::LinAlg::Matrix<3, 3, T> SWnewmass(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::computespin(SWnewmass, Wnewmass);

    Core::LinAlg::Matrix<3, 1, T> Jp_Wnewmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, T> auxvector1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, T> Pi_t(Core::LinAlg::Initialization::zero);
    Jp_Wnewmass.multiply(Jp, Wnewmass);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        auxvector1(i) += SWnewmass(i, j) * Jp_Wnewmass(j) + Jp(i, j) * Anewmass(j);

    Pi_t.multiply(triad_mat_gp[numgp], auxvector1);

    Core::LinAlg::Matrix<3, 1, T> L_t(Core::LinAlg::Initialization::zero);
    L_t.update(mass_inertia_translational, rttnewmass, 1.0);

    // residual contribution from inertia moment
    f_inert_aux.clear();
    f_inert_aux.multiply(v_theta_gp[numgp], Pi_t);
    f_inert_aux.scale(wgt * jacobi_[numgp]);
    f_inert.update(1.0, f_inert_aux, 1.0);

    // residual contribution from inertia force
    f_inert_aux.clear();
    f_inert_aux.multiply_tn(N, L_t);
    f_inert_aux.scale(wgt * jacobi_[numgp]);
    f_inert.update(1.0, f_inert_aux, 1.0);


    // compute analytic mass matrix if required
    if (massmatrix != nullptr)
    {
      // temporal derivative of angular momentum equals negative inertia moment
      Core::LinAlg::Matrix<3, 1, T> moment_rho(Pi_t);
      moment_rho.scale(-1.0);

      if (weakkirchhoff_)
      {
        calculate_mass_matrix_contributions_analytic_wk<nnodecl>(*massmatrix,
            disp_totlag_centerline, v_theta_gp[numgp], lin_theta_gp[numgp], moment_rho, deltatheta,
            Wnewmass, triad_mat_gp[numgp], triad_mat_old, N, mass_inertia_translational, Jp,
            lin_prefactor_acc, lin_prefactor_vel, xi, jacobi_[numgp], wgt);
      }
      else
      {
        FOUR_C_THROW(
            "you tried to calculate the analytic contributions to mass matrix which is not "
            "implemented yet in case of SK!");
      }
    }


    // Calculation of kinetic energy
    Core::LinAlg::Matrix<1, 1, T> ekinrot(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<1, 1, T> ekintrans(Core::LinAlg::Initialization::zero);
    ekinrot.multiply_tn(Wnewmass, Jp_Wnewmass);
    ekintrans.multiply_tn(rtnewmass, rtnewmass);
    ekin_ += 0.5 *
             (Core::FADUtils::cast_to_double(ekinrot(0, 0)) +
                 mass_inertia_translational * Core::FADUtils::cast_to_double(ekintrans(0, 0))) *
             wgt * jacobi_[numgp];

    //**********begin: update class variables needed for storage**************
    Core::LinAlg::Matrix<3, 1, T> wnewmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, T> anewmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, T> amodnewmass(Core::LinAlg::Initialization::zero);

    wnewmass.multiply(triad_mat_gp[numgp], Wnewmass);
    anewmass.multiply(triad_mat_gp[numgp], Anewmass);
    amodnewmass.multiply(triad_mat_gp[numgp], Amodnewmass);


    // compute quaternion of current material triad at gp
    Core::LinAlg::Matrix<4, 1, T> Qnewmass(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::triadtoquaternion(triad_mat_gp[numgp], Qnewmass);

    for (unsigned int i = 0; i < 4; ++i)
    {
      (qnewmass_[numgp])(i) = Core::FADUtils::cast_to_double(Qnewmass(i));
    }

    for (unsigned int i = 0; i < 3; ++i)
    {
      (wnewmass_[numgp])(i) = Core::FADUtils::cast_to_double(wnewmass(i));
      (anewmass_[numgp])(i) = Core::FADUtils::cast_to_double(anewmass(i));
      (amodnewmass_[numgp])(i) = Core::FADUtils::cast_to_double(amodnewmass(i));

      (rnewmass_[numgp])(i) = Core::FADUtils::cast_to_double(rnewmass(i));
      (rtnewmass_[numgp])(i) = Core::FADUtils::cast_to_double(rtnewmass(i));
      (rttnewmass_[numgp])(i) = Core::FADUtils::cast_to_double(rttnewmass(i));
      (rttmodnewmass_[numgp])(i) = Core::FADUtils::cast_to_double(rttmodnewmass(i));
    }
    //**********end: update class variables needed for storage**************
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::calculate_mass_matrix_contributions_analytic_wk(
    Core::LinAlg::SerialDenseMatrix& massmatrix,
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>&
        disp_totlag_centerline,
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, double>& v_theta_bar,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta_bar,
    const Core::LinAlg::Matrix<3, 1, double>& moment_rho,
    const Core::LinAlg::Matrix<3, 1, double>& deltatheta,
    const Core::LinAlg::Matrix<3, 1, double>& angular_velocity_material,
    const Core::LinAlg::Matrix<3, 3, double>& triad_mat,
    const Core::LinAlg::Matrix<3, 3, double>& triad_mat_conv,
    const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N,
    double mass_inertia_translational,
    const Core::LinAlg::Matrix<3, 3, double>& tensor_mass_moment_of_inertia,
    double lin_prefactor_acc, double lin_prefactor_vel, double xi_gp, double jacobifac_gp,
    double GPwgt) const
{
  // spatial dimension
  const unsigned int ndim = 3;
  // number of values used for centerline interpolation (Hermite: value + derivative)
  const unsigned int vpernode = 2;
  // size of Dof vector of this element
  const unsigned int numdofelement = ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS;

  // create a fixed size matrix as view on the Core::LinAlg::SerialDenseMatrix to avoid copying
  Core::LinAlg::Matrix<numdofelement, numdofelement, double> massmatrix_fixedsize(massmatrix, true);


  // matrices storing the assembled shape functions or s-derivative
  Core::LinAlg::Matrix<ndim, numdofelement, double> N_s;
  Core::LinAlg::Matrix<1, numdofelement, double> L;

  // matrices storing individual shape functions, its xi-derivatives and s-derivatives
  Core::LinAlg::Matrix<1, vpernode * nnodecl, double> N_i_xi;
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i;

  // r' vector and its norm
  Core::LinAlg::Matrix<3, 1, double> r_s_cp(Core::LinAlg::Initialization::zero);
  double abs_r_s_cp = 0.0;

  // first base vector
  Core::LinAlg::Matrix<3, 1, double> g_1_cp(Core::LinAlg::Initialization::zero);


  // linearization of re-interpolated strain variations
  std::vector<Core::LinAlg::Matrix<numdofelement, numdofelement, double>> lin_v_thetaperp_moment_cp(
      BEAM3K_COLLOCATION_POINTS),
      lin_v_thetapar_moment_cp(BEAM3K_COLLOCATION_POINTS);

  Core::LinAlg::Matrix<numdofelement, numdofelement, double> lin_v_thetaperp_bar_moment(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdofelement, numdofelement, double> lin_v_thetapar_bar_moment(
      Core::LinAlg::Initialization::zero);


  Core::LinAlg::Matrix<3, 3, double> spinmatrix_of_moment_rho(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::computespin<double>(spinmatrix_of_moment_rho, moment_rho);


  // linearization of stress resultant (moment)
  Core::LinAlg::Matrix<3, numdofelement, double> lin_moment_rho(Core::LinAlg::Initialization::zero);


  /***********************************************************************************************/
  // note: we need an additional loop over the collocation points here for all quantities that
  //       would be third order tensors if not multiplied by the associated vector (in this case
  //       moment vector); since the vector is only available within the loop over the Gauss points
  //       (i.e. at this current GP), we compute right here the lin_v_theta*_moment terms in an
  //       extra loop over the collocation points

  double xi_cp = 0.0;    // parameter coordinate
  unsigned int ind = 0;  // position index where CP quantities have to be stored

  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    // Determine storage position for this cp
    ind = Core::LargeRotations::numbering_trafo(icp + 1, BEAM3K_COLLOCATION_POINTS);

    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    xi_cp = (double)icp / (double)(BEAM3K_COLLOCATION_POINTS - 1) * 2.0 - 1.0;


    // get all required shape function values
    L_i.clear();
    Core::FE::shape_function_1d(L_i, xi_cp, shape());

    L.clear();
    assemble_shapefunctions_l(L_i, L);

    N_i_xi.clear();
    Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi_cp, length_, Core::FE::CellType::line2);

    N_s.clear();
    assemble_shapefunctions_ns(N_i_xi, jacobi_cp_[ind], N_s);


    // Calculation of r' and g_1
    r_s_cp.clear();
    r_s_cp.multiply(N_s, disp_totlag_centerline);

    abs_r_s_cp = r_s_cp.norm2();  // Todo think about computing and storing inverse value here

    g_1_cp.clear();
    g_1_cp.update(std::pow(abs_r_s_cp, -1.0), r_s_cp);


    calc_lin_v_thetaperp_moment<nnodecl>(
        lin_v_thetaperp_moment_cp[ind], N_s, g_1_cp, abs_r_s_cp, spinmatrix_of_moment_rho);

    calc_lin_v_thetapar_moment<nnodecl>(
        lin_v_thetapar_moment_cp[ind], L, N_s, g_1_cp, abs_r_s_cp, moment_rho);
  }


  /***********************************************************************************************/
  // re-interpolation of quantities at xi based on CP values

  L_i.clear();
  Core::FE::shape_function_1d(L_i, xi_gp, shape());

  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    lin_v_thetaperp_bar_moment.update(L_i(icp), lin_v_thetaperp_moment_cp[icp], 1.0);
    lin_v_thetapar_bar_moment.update(L_i(icp), lin_v_thetapar_moment_cp[icp], 1.0);
  }
  /***********************************************************************************************/

  calc_lin_moment_inertia<nnodecl>(lin_moment_rho, triad_mat, triad_mat_conv, deltatheta,
      angular_velocity_material, lin_theta_bar, spinmatrix_of_moment_rho,
      tensor_mass_moment_of_inertia, lin_prefactor_acc, lin_prefactor_vel);

  /***********************************************************************************************/
  // finally put everything together

  // constant pre-factor
  const double jacobifac_GPwgt = jacobifac_gp * GPwgt;

  Core::LinAlg::Matrix<numdofelement, numdofelement, double> auxmatrix(
      Core::LinAlg::Initialization::zero);

  // linearization of residual from inertia force
  auxmatrix.multiply_tn(N, N);
  auxmatrix.scale(mass_inertia_translational * lin_prefactor_acc);

  massmatrix_fixedsize.update(jacobifac_GPwgt, auxmatrix, 1.0);


  // linearization of residual from inertia moment
  massmatrix_fixedsize.update(-1.0 * jacobifac_GPwgt, lin_v_thetaperp_bar_moment, 1.0);

  massmatrix_fixedsize.update(-1.0 * jacobifac_GPwgt, lin_v_thetapar_bar_moment, 1.0);

  auxmatrix.clear();
  auxmatrix.multiply(v_theta_bar, lin_moment_rho);

  massmatrix_fixedsize.update(-1.0 * jacobifac_GPwgt, auxmatrix, 1.0);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
int Discret::Elements::Beam3k::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  set_params_interface_ptr(params);

  /* As long as only endpoint forces and moments as well as distributed forces
   * (i.e. no distributed moments) are considered, the method evaluate_neumann is identical
   * for the WK and the SK case. */

  if (BEAM3K_COLLOCATION_POINTS != 2 and BEAM3K_COLLOCATION_POINTS != 3 and
      BEAM3K_COLLOCATION_POINTS != 4)
  {
    FOUR_C_THROW("Only the values 2,3 and 4 are valid for BEAM3K_COLLOCATION_POINTS!!!");
  }

  // number of nodes used for centerline interpolation
  const unsigned int nnodecl = 2;

  // get element displacements
  std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
      discretization.get_state("displacement new");
  if (disp == nullptr)
    FOUR_C_THROW(
        "Cannot get state vector 'displacement new'. "
        "No linearization provided for load (add 'LOADLIN: true' to input file)");
  std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);

  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double> disp_totlag(
      Core::LinAlg::Initialization::zero);

  // Set current positions and orientations at all nodes:
  update_disp_totlag<nnodecl, double>(mydisp, disp_totlag);


  // find out whether we will use a time curve
  double time = -1.0;
  if (this->is_params_interface())
    time = this->params_interface_ptr()->get_total_time();
  else
    time = params.get("total time", -1.0);

  // get values and switches from the condition:
  // onoff is related to the first 6 flags of a line Neumann condition in the input file;
  // value 1 for flag i says that condition is active for i-th degree of freedom
  const auto onoff = condition.parameters().get<std::vector<int>>("ONOFF");

  // val is related to the 6 "VAL" fields after the onoff flags of the Neumann condition
  // in the input file; val gives the values of the force as a multiple of the prescribed load curve
  const auto val = condition.parameters().get<std::vector<double>>("VAL");

  // compute the load vector based on value, scaling factor and whether condition is active
  Core::LinAlg::Matrix<6, 1, double> load_vector_neumann(Core::LinAlg::Initialization::zero);
  for (unsigned int i = 0; i < 6; ++i) load_vector_neumann(i) = onoff[i] * val[i];

  /***********************************************************************************************/

  // if a point neumann condition needs to be linearized
  if (condition.type() == Core::Conditions::PointNeumannEB)
  {
    // find out whether we will use a time curve and get the factor
    const auto funct = condition.parameters().get<std::vector<std::optional<int>>>("FUNCT");
    // amplitude of load curve at current time called
    std::vector<double> functtimefac(6, 1.0);

    for (unsigned int i = 0; i < 6; ++i)
    {
      // number of the load curve related with a specific line Neumann condition called
      if (funct[i].has_value() && funct[i].value() > 0)
        functtimefac[i] = Global::Problem::instance()
                              ->function_by_id<Core::Utils::FunctionOfTime>(funct[i].value())
                              .evaluate(time);

      load_vector_neumann(i) *= functtimefac[i];
    }

    // find out at which node the condition is applied
    const std::vector<int>* nodeids = condition.get_nodes();
    if (nodeids == nullptr) FOUR_C_THROW("failed to retrieve node IDs from condition!");

    /* find out local node number --> this is done since the first element of a Neumann point
     * condition is used for this function in this case we do not know whether it is the left
     * or the right node. In addition to that, xi is assigned in order to determine the index
     * of the base vectors for the smallest rotation system */
    int node = -1;

    if ((*nodeids)[0] == nodes()[0]->id())
    {
      node = 0;
    }
    else if ((*nodeids)[0] == nodes()[1]->id())
    {
      node = 1;
    }

    if (node == -1) FOUR_C_THROW("Node could not be found on nodemap!");


    evaluate_point_neumann_eb<nnodecl>(elevec1, elemat1, disp_totlag, load_vector_neumann, node);
  }
  // if a line neumann condition needs to be linearized
  else if (condition.type() == Core::Conditions::LineNeumann)
  {
    // funct is related to the 6 "FUNCT" fields after the val field of the Neumann condition
    // in the input file; funct gives the number of the function defined in the section FUNCT
    const auto function_numbers =
        condition.parameters().get<std::vector<std::optional<int>>>("FUNCT");

    // Check if distributed moment load is applied and throw error
    for (unsigned int idof = 3; idof < 6; ++idof)
    {
      if (function_numbers[idof].has_value() && function_numbers[idof].value() > 0)
        FOUR_C_THROW(
            "Line Neumann conditions for distributed moments are not implemented for beam3k"
            " so far! Only the function flag 1, 2 and 3 can be set!");
    }

    evaluate_line_neumann<nnodecl>(
        elevec1, elemat1, disp_totlag, load_vector_neumann, function_numbers, time);
  }

  return 0;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::evaluate_point_neumann_eb(Core::LinAlg::SerialDenseVector& forcevec,
    Core::LinAlg::SerialDenseMatrix* stiffmat,
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>& disp_totlag,
    const Core::LinAlg::Matrix<6, 1, double>& load_vector_neumann, int node) const
{
  /***********************************************************************************************/
  // external point force
  /***********************************************************************************************/
  /* handling of external point force is the same for rotvec_=true/false and does not need
   * linearization. So we are done here */
  for (unsigned int idim = 0; idim < 3; ++idim)
  {
    forcevec(node * 7 + idim) += load_vector_neumann(idim);
  }

  /***********************************************************************************************/
  // external point moment
  /***********************************************************************************************/

  // here, rotation vector-based formulation is easy because we can directly write specified
  // force and moment from Neumann point condition to corresponding element force vector entries
  // moreover, there is no contribution to the element stiffness matrix
  if (rotvec_ == true)
  {
    for (unsigned int idim = 0; idim < 3; ++idim)
    {
      // external moment
      forcevec(node * 7 + 3 + idim) += load_vector_neumann(idim + 3);
    }
  }
  else
  {
    /* in tangent based formulation, we need to linearize the residual contributions from
     * external moments */
    // analytic linearization
    if (not use_fad_)
    {
      Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>
          disp_totlag_centerline(Core::LinAlg::Initialization::zero);
      std::vector<Core::LinAlg::Matrix<3, 3, double>> triad_mat_cp(BEAM3K_COLLOCATION_POINTS);
      std::vector<Core::LinAlg::Matrix<4, 1>> Qref(BEAM3K_COLLOCATION_POINTS);

      update_nodal_variables<nnodecl, double>(
          disp_totlag, disp_totlag_centerline, triad_mat_cp, Qref);

      // create view on external force vector to avoid copying
      // IMPORTANT: fext is multiplied by (-1) in 4C, consequently we need no minus sign here
      Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double> f_ext_fixedsize(
          forcevec, true);

      // r' at node
      Core::LinAlg::Matrix<3, 1, double> r_s(Core::LinAlg::Initialization::zero);
      // |r'| at node
      double abs_r_s = 0.0;

      for (unsigned int i = 0; i < 3; ++i) r_s(i) = disp_totlag_centerline(node * 7 + 3 + i);

      abs_r_s = Core::FADUtils::norm(r_s);

      // matrix for moment at node
      Core::LinAlg::Matrix<3, 1, double> moment(Core::LinAlg::Initialization::zero);

      for (unsigned int i = 0; i < 3; ++i)
      {
        moment(i) = load_vector_neumann(i + 3);
      }

      evaluate_residual_from_point_neumann_moment<nnodecl, double>(
          f_ext_fixedsize, moment, r_s, abs_r_s, node);


      if (stiffmat != nullptr)
      {
        evaluate_stiff_matrix_analytic_from_point_neumann_moment<nnodecl>(
            *stiffmat, moment, r_s, abs_r_s, node);
      }
    }
    // automatic linearization
    else
    {
      Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> disp_totlag_FAD;

      for (unsigned int idof = 0; idof < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; ++idof)
        disp_totlag_FAD(idof) = disp_totlag(idof);

      // Next, we have to set variables for FAD
      set_automatic_differentiation_variables<nnodecl>(disp_totlag_FAD);

      Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>
          disp_totlag_centerline_FAD(Core::LinAlg::Initialization::zero);
      std::vector<Core::LinAlg::Matrix<3, 3, FAD>> triad_mat_cp_FAD(BEAM3K_COLLOCATION_POINTS);
      std::vector<Core::LinAlg::Matrix<4, 1>> Qref(BEAM3K_COLLOCATION_POINTS);

      update_nodal_variables<nnodecl, FAD>(
          disp_totlag_FAD, disp_totlag_centerline_FAD, triad_mat_cp_FAD, Qref);

      // external force vector
      Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> f_ext_FAD(
          Core::LinAlg::Initialization::zero);


      // r' at node
      Core::LinAlg::Matrix<3, 1, FAD> r_s_FAD(Core::LinAlg::Initialization::zero);
      // |r'| at node
      FAD abs_r_s_FAD = 0.0;

      for (unsigned int i = 0; i < 3; ++i)
        r_s_FAD(i) = disp_totlag_centerline_FAD(node * 7 + 3 + i);

      abs_r_s_FAD = Core::FADUtils::norm(r_s_FAD);


      // matrix for moment at node
      Core::LinAlg::Matrix<3, 1, FAD> moment(Core::LinAlg::Initialization::zero);

      for (unsigned int i = 0; i < 3; ++i)
      {
        moment(i) = load_vector_neumann(i + 3);
      }


      evaluate_residual_from_point_neumann_moment<nnodecl, FAD>(
          f_ext_FAD, moment, r_s_FAD, abs_r_s_FAD, node);

      // IMPORTANT: fext is multiplied by (-1) in 4C, consequently we need no minus sign here
      for (unsigned int i = 0; i < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; ++i)
      {
        forcevec(i) += f_ext_FAD(i).val();
      }

      // IMPORTANT: in contrast to f_ext, elemat1 it is directly added to the stiffness matrix,
      // therefore there has to be a sign change!
      if (stiffmat != nullptr)
      {
        // Calculating stiffness matrix with FAD
        for (unsigned int i = 0; i < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; ++i)
        {
          for (unsigned int j = 0; j < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; ++j)
          {
            (*stiffmat)(i, j) -= f_ext_FAD(i).dx(j);
          }
        }
      }
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void Discret::Elements::Beam3k::evaluate_residual_from_point_neumann_moment(
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& force_ext,
    const Core::LinAlg::Matrix<3, 1, T>& moment_ext, const Core::LinAlg::Matrix<3, 1, T>& r_s,
    T abs_r_s, int node) const
{
  // S(r') at node
  Core::LinAlg::Matrix<3, 3, T> Srs(Core::LinAlg::Initialization::zero);

  // auxiliary quantities
  Core::LinAlg::Matrix<3, 1, T> auxvector(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 1, T> auxscalar(Core::LinAlg::Initialization::zero);

  Core::LargeRotations::computespin(Srs, r_s);
  auxvector.multiply(Srs, moment_ext);
  auxvector.scale(-1.0 * std::pow(abs_r_s, -2.0));

  auxscalar.multiply_tn(r_s, moment_ext);
  auxscalar.scale(1.0 / abs_r_s);

  for (unsigned int j = 0; j < 3; ++j)
  {
    force_ext(node * 7 + 3 + j) += auxvector(j);
  }

  force_ext(node * 7 + 6) += auxscalar(0, 0);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::evaluate_stiff_matrix_analytic_from_point_neumann_moment(
    Core::LinAlg::SerialDenseMatrix& stiffmat, const Core::LinAlg::Matrix<3, 1, double>& moment_ext,
    const Core::LinAlg::Matrix<3, 1, double>& r_s, double abs_r_s, int node) const
{
  double xi_node = 0.0;
  double jacobi_node = 0.0;

  if (node == 0)
  {
    xi_node = -1.0;
    jacobi_node = jacobi_cp_[0];
  }
  else if (node == 1)
  {
    xi_node = 1.0;
    jacobi_node = jacobi_cp_[1];
  }
  else
  {
    FOUR_C_THROW("{} is an invalid value for element local node ID! Expected 0 or 1", node);
  }

  // matrices storing the assembled shape functions or s-derivative
  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> N_s;
  Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> L;

  // matrices storing individual shape functions, its xi-derivatives and s-derivatives
  Core::LinAlg::Matrix<1, 2 * nnodecl, double> N_i_xi;
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i;

  // get all required shape function values
  L_i.clear();
  Core::FE::shape_function_1d(L_i, xi_node, shape());

  L.clear();
  assemble_shapefunctions_l(L_i, L);

  N_i_xi.clear();
  Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi_node, length_, Core::FE::CellType::line2);

  N_s.clear();
  assemble_shapefunctions_ns(N_i_xi, jacobi_node, N_s);


  // Calculation of first base vector
  Core::LinAlg::Matrix<3, 1, double> g_1(Core::LinAlg::Initialization::zero);
  g_1.update(std::pow(abs_r_s, -1.0), r_s);

  Core::LinAlg::Matrix<3, 3, double> spinmatrix_of_moment_ext(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::computespin(spinmatrix_of_moment_ext, moment_ext);

  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
      6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>
      lin_v_thetaperp_moment(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
      6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>
      lin_v_thetapar_moment(Core::LinAlg::Initialization::zero);

  calc_lin_v_thetaperp_moment<nnodecl>(
      lin_v_thetaperp_moment, N_s, g_1, abs_r_s, spinmatrix_of_moment_ext);

  calc_lin_v_thetapar_moment<nnodecl>(lin_v_thetapar_moment, L, N_s, g_1, abs_r_s, moment_ext);


  // IMPORTANT: in contrast to f_ext, elemat1 it is directly added to the stiffness matrix,
  // therefore there has to be a sign change!
  for (unsigned int irow = 0; irow < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; ++irow)
    for (unsigned int icol = 0; icol < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; ++icol)
    {
      stiffmat(irow, icol) -= lin_v_thetaperp_moment(irow, icol);
      stiffmat(irow, icol) -= lin_v_thetapar_moment(irow, icol);
    }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void Discret::Elements::Beam3k::evaluate_line_neumann(Core::LinAlg::SerialDenseVector& forcevec,
    Core::LinAlg::SerialDenseMatrix* stiffmat,
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>& disp_totlag,
    const Core::LinAlg::Matrix<6, 1, double>& load_vector_neumann,
    const std::vector<std::optional<int>>& function_numbers, double time) const
{
  if (not use_fad_)
  {
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double> disp_totlag_centerline(
        Core::LinAlg::Initialization::zero);
    std::vector<Core::LinAlg::Matrix<3, 3, double>> triad_mat_cp(BEAM3K_COLLOCATION_POINTS);
    std::vector<Core::LinAlg::Matrix<4, 1>> Qref(BEAM3K_COLLOCATION_POINTS);

    update_nodal_variables<nnodecl, double>(
        disp_totlag, disp_totlag_centerline, triad_mat_cp, Qref);

    // create view on external force vector to avoid copying
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double> f_ext(forcevec, true);

    evaluate_line_neumann_forces<nnodecl, double>(
        f_ext, load_vector_neumann, function_numbers, time);

    // in tangent-based formulation (rotvec_=false), there is no contribution to stiffmat,
    // so we are done after calculation of forces

    // safety check for rotation-vector based formulation
    if (rotvec_ == true)
      FOUR_C_THROW(
          "Beam3k: analytic linearization of LineNeumann condition (distributed forces) "
          "not implemented yet for ROTVEC variant! Activate automatic linearization via FAD");
  }
  else
  {
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> disp_totlag_FAD;

    for (unsigned int idof = 0; idof < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; ++idof)
      disp_totlag_FAD(idof) = disp_totlag(idof);

    // Next, we have to set variables for FAD
    set_automatic_differentiation_variables<nnodecl>(disp_totlag_FAD);

    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>
        disp_totlag_centerline_FAD(Core::LinAlg::Initialization::zero);
    std::vector<Core::LinAlg::Matrix<3, 3, FAD>> triad_mat_cp_FAD(BEAM3K_COLLOCATION_POINTS);
    std::vector<Core::LinAlg::Matrix<4, 1>> Qref(BEAM3K_COLLOCATION_POINTS);

    update_nodal_variables<nnodecl, FAD>(
        disp_totlag_FAD, disp_totlag_centerline_FAD, triad_mat_cp_FAD, Qref);


    // external force vector
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> f_ext(
        Core::LinAlg::Initialization::zero);

    evaluate_line_neumann_forces<nnodecl, FAD>(f_ext, load_vector_neumann, function_numbers, time);


    if (rotvec_ == true)
    {
      apply_rot_vec_trafo<nnodecl, FAD>(disp_totlag_centerline_FAD, f_ext);
    }


    // IMPORTANT: fext is multiplied by (-1) in 4C, consequently we need no minus sign here
    for (unsigned int i = 0; i < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; i++)
    {
      forcevec(i) = f_ext(i).val();
    }


    if (rotvec_ == true and stiffmat != nullptr)
    {
      // IMPORTANT: in contrast to f_ext, elemat1 it is directly added to the stiffness matrix,
      // therefore there has to be a sign change!

      // Calculating stiffness matrix with FAD
      for (unsigned int i = 0; i < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; i++)
      {
        for (unsigned int j = 0; j < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; j++)
        {
          (*stiffmat)(i, j) = -f_ext(i).dx(j);
        }
      }

      transform_stiff_matrix_multiplicative<nnodecl, FAD>(stiffmat, disp_totlag_FAD);
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void Discret::Elements::Beam3k::evaluate_line_neumann_forces(
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& force_ext,
    const Core::LinAlg::Matrix<6, 1, double>& load_vector_neumann,
    const std::vector<std::optional<int>>& function_numbers, double time) const
{
  std::vector<Core::LinAlg::Matrix<3, 3>> Gref(2);

  for (unsigned int node = 0; node < 2; ++node)
  {
    Gref[node].clear();
    Core::LargeRotations::angletotriad(theta0_[node], Gref[node]);
  }

  // gaussian points
  Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);

  Core::LinAlg::Matrix<1, 4, double> N_i;
  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, T> N;

  // auxiliary external force vector
  Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T> f_ext_aux(
      Core::LinAlg::Initialization::zero);

  // integration loops
  for (int numgp = 0; numgp < gausspoints.nquad; ++numgp)
  {
    // integration points in parameter space and weights
    const double xi = gausspoints.qxg[numgp][0];
    const double wgt = gausspoints.qwgt[numgp];

    // Clear matrix for shape functions
    N_i.clear();
    N.clear();
    Core::FE::shape_function_hermite_1d(N_i, xi, length_, Core::FE::CellType::line2);
    assemble_shapefunctions_n(N_i, N);

    // position vector at the gauss point at reference configuration needed for function evaluation
    std::vector<double> X_ref(3, 0.0);
    // calculate coordinates of corresponding Gauss point in reference configuration
    for (unsigned int node = 0; node < 2; ++node)
    {
      for (unsigned int dof = 0; dof < 3; ++dof)
      {
        X_ref[dof] +=
            nodes()[node]->x()[dof] * N_i(2 * node) + (Gref[node])(dof, 0) * N_i(2 * node + 1);
      }
    }


    double functionfac = 1.0;
    Core::LinAlg::Matrix<3, 1, T> force_gp(Core::LinAlg::Initialization::zero);

    // sum up load components
    for (unsigned int idof = 0; idof < 3; ++idof)
    {
      if (function_numbers[idof].has_value() && function_numbers[idof].value() > 0)
      {
        functionfac =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(function_numbers[idof].value())
                .evaluate(X_ref, time, idof);
      }
      else
        functionfac = 1.0;

      force_gp(idof) = load_vector_neumann(idof) * functionfac;
    }

    f_ext_aux.clear();
    f_ext_aux.multiply_tn(N, force_gp);

    force_ext.update(wgt * jacobi_[numgp], f_ext_aux, 1.0);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
inline void Discret::Elements::Beam3k::calc_brownian_forces_and_stiff(
    Teuchos::ParameterList& params, std::vector<double>& vel, std::vector<double>& disp,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseVector* force)
{
  if (weakkirchhoff_ == false)
    FOUR_C_THROW(
        "calculation of viscous damping moments not implemented for"
        "WK=0 ('strong' Kirchhoff) yet. Use BEAM3WK elements (set WK=1)!");

  if (rotvec_ == true)
    FOUR_C_THROW(
        "Beam3k: Calculation of Brownian forces not tested yet for ROTVEC, "
        "i.e. nodal rotation vectors as primary variables. Use ROTVEC=0 (tangent based "
        "formulation)");

  // unshift node positions, i.e. manipulate element displacement vector
  // as if there where no periodic boundary conditions
  if (brownian_dyn_params_interface_ptr() != nullptr)
    un_shift_node_position(disp, *brownian_dyn_params_interface().get_periodic_bounding_box());


  // total position state of element

  /* vector for current nodal DoFs in total Lagrangian style, i.e. displacement + initial values:
   * rotvec_==true:  disp_totlag=[\v{d}_1, \v{theta}_1, t_1, \v{d}_2, \v{theta}_2, t_2, \alpha_3]
   * rotvec_==false: disp_totlag=[\v{d}_1, \v{t}_1, \alpha_1, \v{d}_2, \v{t}_2, \alpha_2, \alpha_3]
   */
  Core::LinAlg::Matrix<nnode * vpernode * ndim + BEAM3K_COLLOCATION_POINTS, 1, double> disp_totlag(
      Core::LinAlg::Initialization::zero);

  // Set current positions and tangents and triads at all nodes
  update_disp_totlag<nnode, double>(disp, disp_totlag);


  // velocity state of element

  // export current velocity state of element to fixed size matrix
  Core::LinAlg::Matrix<nnode * vpernode * ndim + BEAM3K_COLLOCATION_POINTS, 1> vel_fixedsize(
      vel.data(), true);
  Core::LinAlg::Matrix<nnode * vpernode * ndim, 1> vel_centerline(
      Core::LinAlg::Initialization::zero);

  // update current values of centerline (i.e. translational) velocity
  extract_centerline_dof_values_from_element_state_vector<nnode, vpernode, double>(
      vel_fixedsize, vel_centerline);


  // analytic linearization of residual contributions
  if (not use_fad_)
  {
    // force vector resulting from Brownian dynamics
    Core::LinAlg::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, double>
        force_brownian(Core::LinAlg::Initialization::zero);

    if (force != nullptr)
    {
      // set view on Core::LinAlg::SerialDenseVector to avoid copying of data
      force_brownian.set_view(&((*force)(0)));
    }

    // vector containing locally assembled nodal positions and tangents required for centerline:
    // r(s)=N(s)*disp_totlag_centerline, with disp_totlag_centerline=[\v{d}_1, \v{t}_1, 0, \v{d}_2,
    // \v{t}_2, 0, 0]
    Core::LinAlg::Matrix<nnode * vpernode * ndim + BEAM3K_COLLOCATION_POINTS, 1, double>
        disp_totlag_centerline(Core::LinAlg::Initialization::zero);

    // material triads at collocation points
    std::vector<Core::LinAlg::Matrix<3, 3, double>> triad_mat_cp(BEAM3K_COLLOCATION_POINTS);


    update_nodal_variables<nnode, double>(disp_totlag, disp_totlag_centerline, triad_mat_cp,
        qrefnew_);  // Todo @grill: do we need to update Qrefnew_ here? doesn't seem to be a problem
                    // but anyway ...

    Core::LinAlg::Matrix<nnode * vpernode * ndim, 1, double> disp_totlag_centerlineDOFs_only(
        Core::LinAlg::Initialization::zero);
    extract_centerline_dof_values_from_element_state_vector<nnode, vpernode, double>(
        disp_totlag_centerline, disp_totlag_centerlineDOFs_only);


    // Evaluation of force vectors and stiffness matrices

    // add stiffness and forces (i.e. moments) due to rotational damping effects
    evaluate_rotational_damping<double, nnode, vpernode, ndim>(
        disp_totlag_centerline, triad_mat_cp, stiffmatrix, force_brownian);

    if (stiffmatrix != nullptr) stiff_ptc_ = *stiffmatrix;

    // add stiffness and forces due to translational damping effects
    evaluate_translational_damping<double, nnode, vpernode, ndim>(
        params, vel_centerline, disp_totlag_centerlineDOFs_only, stiffmatrix, force_brownian);

    // add stochastic forces and (if required) resulting stiffness
    evaluate_stochastic_forces<double, nnode, vpernode, ndim, 3>(
        disp_totlag_centerlineDOFs_only, stiffmatrix, force_brownian);
  }
  // automatic linearization of residual contributions via FAD
  else
  {
    // force vector resulting from Brownian dynamics
    Core::LinAlg::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, FAD>
        force_brownian_FAD(Core::LinAlg::Initialization::zero);

    // copy pre-computed disp_totlag to a FAD matrix
    Core::LinAlg::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, FAD>
        disp_totlag_FAD(Core::LinAlg::Initialization::zero);

    for (unsigned int idof = 0; idof < ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS; ++idof)
      disp_totlag_FAD(idof) = disp_totlag(idof);


    // vector containing locally assembled nodal positions and tangents required for centerline:
    // r(s)=N(s)*disp_totlag_centerline, with disp_totlag_centerline=[\v{d}_1, \v{t}_1, 0, \v{d}_2,
    // \v{t}_2, 0, 0]
    Core::LinAlg::Matrix<nnode * vpernode * ndim + BEAM3K_COLLOCATION_POINTS, 1, FAD>
        disp_totlag_centerline_FAD(Core::LinAlg::Initialization::zero);

    // material triads at collocation points
    std::vector<Core::LinAlg::Matrix<3, 3, FAD>> triad_mat_cp_FAD(BEAM3K_COLLOCATION_POINTS);


    // Next, we have to set variables for FAD
    set_automatic_differentiation_variables<nnode>(disp_totlag_FAD);

    update_nodal_variables<nnode, FAD>(disp_totlag_FAD, disp_totlag_centerline_FAD,
        triad_mat_cp_FAD,
        qrefnew_);  // Todo do we need to update Qrefnew_ here? doesn't seem to be a problem but
                    // anyway ...

    Core::LinAlg::Matrix<nnode * vpernode * ndim, 1, FAD> disp_totlag_centerlineDOFs_only(
        Core::LinAlg::Initialization::zero);
    extract_centerline_dof_values_from_element_state_vector<nnode, vpernode, FAD>(
        disp_totlag_centerline_FAD, disp_totlag_centerlineDOFs_only);


    // Evaluation of force vectors and stiffness matrices

    // add stiffness and forces due to translational damping effects
    evaluate_translational_damping<FAD, nnode, vpernode, ndim>(
        params, vel_centerline, disp_totlag_centerlineDOFs_only, stiffmatrix, force_brownian_FAD);

    // add stochastic forces and (if required) resulting stiffness
    evaluate_stochastic_forces<FAD, nnode, vpernode, ndim, 3>(
        disp_totlag_centerlineDOFs_only, stiffmatrix, force_brownian_FAD);

    // add stiffness and forces (i.e. moments) due to rotational damping effects
    evaluate_rotational_damping<FAD, nnode, vpernode, ndim>(
        disp_totlag_centerline_FAD, triad_mat_cp_FAD, stiffmatrix, force_brownian_FAD);


    // Update stiffness matrix and force vector
    if (stiffmatrix != nullptr)
    {
      // Calculating stiffness matrix with FAD
      for (unsigned int i = 0; i < ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS; i++)
        for (unsigned int j = 0; j < ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS; j++)
          (*stiffmatrix)(i, j) = force_brownian_FAD(i).dx(j);
    }

    if (force != nullptr)
    {
      for (unsigned int i = 0; i < ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS; i++)
        (*force)(i) = force_brownian_FAD(i).val();
    }
  }
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <typename T, unsigned int nnode, unsigned int vpernode, unsigned int ndim>
void Discret::Elements::Beam3k::evaluate_translational_damping(Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<ndim * vpernode * nnode, 1, double>& vel,
    const Core::LinAlg::Matrix<ndim * vpernode * nnode, 1, T>& disp_totlag,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix,
    Core::LinAlg::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, T>& f_int)
{
  /* only nodes for centerline interpolation are considered here (first two nodes of this element)
     each of these nodes holds 3*vpernode translational DoFs AND 1 rotational DoFs */
  const unsigned int dofpernode = ndim * vpernode + 1;

  // get damping coefficients for translational and rotational degrees of freedom (the latter is
  // unused in this element)
  Core::LinAlg::Matrix<ndim, 1> gamma(Core::LinAlg::Initialization::zero);
  get_damping_coefficients(gamma);

  // velocity and gradient of background velocity field
  Core::LinAlg::Matrix<ndim, 1, T> velbackground(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<ndim, ndim, T> velbackgroundgrad(Core::LinAlg::Initialization::zero);

  // evaluation point in physical space corresponding to a certain Gauss point in parameter space
  Core::LinAlg::Matrix<ndim, 1, T> evaluationpoint(Core::LinAlg::Initialization::zero);
  // tangent vector (derivative of beam centerline curve r with respect to arc-length parameter s)
  Core::LinAlg::Matrix<ndim, 1, T> r_s(Core::LinAlg::Initialization::zero);
  // velocity of beam centerline point relative to background fluid velocity
  Core::LinAlg::Matrix<ndim, 1, T> vel_rel(Core::LinAlg::Initialization::zero);

  // viscous force vector per unit length at current GP
  Core::LinAlg::Matrix<ndim, 1, T> f_visc(Core::LinAlg::Initialization::zero);
  // damping matrix
  Core::LinAlg::Matrix<ndim, ndim, T> damp_mat(Core::LinAlg::Initialization::zero);


  // get Gauss points and weights
  Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);

  // matrix to store individual Hermite shape functions and their derivatives evaluated at a certain
  // Gauss point
  Core::LinAlg::Matrix<1, nnode * vpernode, double> N_i(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, nnode * vpernode, double> N_i_xi(Core::LinAlg::Initialization::zero);


  for (int gp = 0; gp < gausspoints.nquad; gp++)
  {
    Discret::Utils::Beam::evaluate_shape_functions_and_derivs_at_xi<nnode, vpernode>(
        gausspoints.qxg[gp][0], N_i, N_i_xi, this->shape(), this->ref_length());

    // compute position vector r of point in physical space corresponding to Gauss point
    calc_r<nnode, vpernode, T>(disp_totlag, N_i, evaluationpoint);

    // compute tangent vector t_{\par}=r' at current Gauss point
    calc_r_s<nnode, vpernode, T>(disp_totlag, N_i_xi, jacobi_[gp], r_s);

    // compute velocity and gradient of background flow field at point r
    get_background_velocity<ndim, T>(params, evaluationpoint, velbackground, velbackgroundgrad);

    /* compute velocity vector at this Gauss point via same interpolation as for centerline
     * position vector
     *
     * Be careful here:
     * special treatment is required if Fad is used. see method with Fad parameters for details */
    calc_velocity<nnode, vpernode, ndim>(vel, N_i, vel_rel, evaluationpoint, gp);

    vel_rel -= velbackground;

    // loop over lines and columns of damping matrix
    for (unsigned int idim = 0; idim < ndim; idim++)
      for (unsigned int jdim = 0; jdim < ndim; jdim++)
        damp_mat(idim, jdim) =
            (idim == jdim) * gamma(1) + (gamma(0) - gamma(1)) * r_s(idim) * r_s(jdim);

    // compute viscous force vector per unit length at current GP
    f_visc.multiply(damp_mat, vel_rel);

    const double jacobifac_gp_weight = jacobi_[gp] * gausspoints.qwgt[gp];


    // loop over all nodes used for centerline interpolation
    for (unsigned int inode = 0; inode < nnode; inode++)
      // loop over dimensions
      for (unsigned int idim = 0; idim < ndim; idim++)
      {
        f_int(inode * dofpernode + idim) +=
            N_i(vpernode * inode) * f_visc(idim) * jacobifac_gp_weight;
        f_int(inode * dofpernode + 3 + idim) +=
            N_i(vpernode * inode + 1) * f_visc(idim) * jacobifac_gp_weight;
      }

    if (stiffmatrix != nullptr)
    {
      evaluate_analytic_stiffmat_contributions_from_translational_damping<nnode, vpernode, ndim>(
          *stiffmatrix, damp_mat, r_s, vel_rel, gamma, velbackgroundgrad, N_i, N_i_xi, jacobi_[gp],
          gausspoints.qwgt[gp]);
    }
  }
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
void Discret::Elements::Beam3k::evaluate_analytic_stiffmat_contributions_from_translational_damping(
    Core::LinAlg::SerialDenseMatrix& stiffmatrix,
    const Core::LinAlg::Matrix<ndim, ndim, double>& damping_matrix,
    const Core::LinAlg::Matrix<ndim, 1, double>& r_s,
    const Core::LinAlg::Matrix<ndim, 1, double>& vel_rel,
    const Core::LinAlg::Matrix<ndim, 1, double>& gamma,
    const Core::LinAlg::Matrix<ndim, ndim, double>& velbackgroundgrad,
    const Core::LinAlg::Matrix<1, nnode * vpernode, double>& N_i,
    const Core::LinAlg::Matrix<1, nnode * vpernode, double>& N_i_xi, double jacobifactor,
    double gp_weight) const
{
  /* only nodes for centerline interpolation are considered here (first two nodes of this element)
     each of these nodes holds 3*vpernode translational DoFs AND 1 rotational DoFs */
  const unsigned int dofpernode = ndim * vpernode + 1;

  // get time step size
  const double dt = params_interface().get_delta_time();

  // compute matrix product of damping matrix and gradient of background velocity
  Core::LinAlg::Matrix<ndim, ndim> dampmatvelbackgroundgrad(Core::LinAlg::Initialization::zero);
  dampmatvelbackgroundgrad.multiply(damping_matrix, velbackgroundgrad);

  const double jacobifac_gp_wgt = jacobifactor * gp_weight;

  // loop over all shape functions in row dimension
  for (unsigned int inode = 0; inode < nnode; ++inode)
    // loop over all shape functions in column dimension
    for (unsigned int jnode = 0; jnode < nnode; ++jnode)
    {
      for (unsigned int idim = 0; idim < ndim; ++idim)
        for (unsigned int jdim = 0; jdim < ndim; ++jdim)
        {
          stiffmatrix(inode * dofpernode + idim, jnode * dofpernode + jdim) +=
              jacobifac_gp_wgt * N_i(vpernode * inode) * N_i(vpernode * jnode) *
              damping_matrix(idim, jdim) / dt;
          stiffmatrix(inode * dofpernode + idim, jnode * dofpernode + jdim) -=
              jacobifac_gp_wgt * N_i(vpernode * inode) * N_i(vpernode * jnode) *
              dampmatvelbackgroundgrad(idim, jdim);
          stiffmatrix(inode * dofpernode + idim, jnode * dofpernode + idim) +=
              gp_weight * N_i(vpernode * inode) * N_i_xi(vpernode * jnode) * (gamma(0) - gamma(1)) *
              r_s(jdim) * vel_rel(jdim);
          stiffmatrix(inode * dofpernode + idim, jnode * dofpernode + jdim) +=
              gp_weight * N_i(vpernode * inode) * N_i_xi(vpernode * jnode) * (gamma(0) - gamma(1)) *
              r_s(idim) * vel_rel(jdim);

          stiffmatrix(inode * dofpernode + 3 + idim, jnode * dofpernode + jdim) +=
              jacobifac_gp_wgt * N_i(vpernode * inode + 1) * N_i(vpernode * jnode) *
              damping_matrix(idim, jdim) / dt;
          stiffmatrix(inode * dofpernode + 3 + idim, jnode * dofpernode + jdim) -=
              jacobifac_gp_wgt * N_i(vpernode * inode + 1) * N_i(vpernode * jnode) *
              dampmatvelbackgroundgrad(idim, jdim);
          stiffmatrix(inode * dofpernode + 3 + idim, jnode * dofpernode + idim) +=
              gp_weight * N_i(vpernode * inode + 1) * N_i_xi(vpernode * jnode) *
              (gamma(0) - gamma(1)) * r_s(jdim) * vel_rel(jdim);
          stiffmatrix(inode * dofpernode + 3 + idim, jnode * dofpernode + jdim) +=
              gp_weight * N_i(vpernode * inode + 1) * N_i_xi(vpernode * jnode) *
              (gamma(0) - gamma(1)) * r_s(idim) * vel_rel(jdim);

          stiffmatrix(inode * dofpernode + idim, jnode * dofpernode + 3 + jdim) +=
              jacobifac_gp_wgt * N_i(vpernode * inode) * N_i(vpernode * jnode + 1) *
              damping_matrix(idim, jdim) / dt;
          stiffmatrix(inode * dofpernode + idim, jnode * dofpernode + 3 + jdim) -=
              jacobifac_gp_wgt * N_i(vpernode * inode) * N_i(vpernode * jnode + 1) *
              dampmatvelbackgroundgrad(idim, jdim);
          stiffmatrix(inode * dofpernode + idim, jnode * dofpernode + 3 + idim) +=
              gp_weight * N_i(vpernode * inode) * N_i_xi(vpernode * jnode + 1) *
              (gamma(0) - gamma(1)) * r_s(jdim) * vel_rel(jdim);
          stiffmatrix(inode * dofpernode + idim, jnode * dofpernode + 3 + jdim) +=
              gp_weight * N_i(vpernode * inode) * N_i_xi(vpernode * jnode + 1) *
              (gamma(0) - gamma(1)) * r_s(idim) * vel_rel(jdim);

          stiffmatrix(inode * dofpernode + 3 + idim, jnode * dofpernode + 3 + jdim) +=
              jacobifac_gp_wgt * N_i(vpernode * inode + 1) * N_i(vpernode * jnode + 1) *
              damping_matrix(idim, jdim) / dt;
          stiffmatrix(inode * dofpernode + 3 + idim, jnode * dofpernode + 3 + jdim) -=
              jacobifac_gp_wgt * N_i(vpernode * inode + 1) * N_i(vpernode * jnode + 1) *
              dampmatvelbackgroundgrad(idim, jdim);
          stiffmatrix(inode * dofpernode + 3 + idim, jnode * dofpernode + 3 + idim) +=
              gp_weight * N_i(vpernode * inode + 1) * N_i_xi(vpernode * jnode + 1) *
              (gamma(0) - gamma(1)) * r_s(jdim) * vel_rel(jdim);
          stiffmatrix(inode * dofpernode + 3 + idim, jnode * dofpernode + 3 + jdim) +=
              gp_weight * N_i(vpernode * inode + 1) * N_i_xi(vpernode * jnode + 1) *
              (gamma(0) - gamma(1)) * r_s(idim) * vel_rel(jdim);
        }
    }
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <typename T, unsigned int nnode, unsigned int vpernode, unsigned int ndim,
    unsigned int randompergauss>
void Discret::Elements::Beam3k::evaluate_stochastic_forces(
    const Core::LinAlg::Matrix<ndim * vpernode * nnode, 1, T>& disp_totlag,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix,
    Core::LinAlg::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, T>& f_int)
{
  /* only nodes for centerline interpolation are considered here (first two nodes of this element)
     each of these nodes holds 3*vpernode translational DoFs AND 1 rotational DoFs */
  const unsigned int dofpernode = ndim * vpernode + 1;

  // damping coefficients for three translational and one rotational degree of freedom
  Core::LinAlg::Matrix<3, 1> gamma(Core::LinAlg::Initialization::zero);
  get_damping_coefficients(gamma);

  /* get pointer at multivector in parameter list linking to random numbers for stochastic
   * forces with zero mean and standard deviation (2*kT / dt)^0.5 */
  std::shared_ptr<Core::LinAlg::MultiVector<double>> randomforces =
      brownian_dyn_params_interface().get_random_forces();

  // tangent vector (derivative of beam centerline curve r with respect to arc-length parameter s)
  Core::LinAlg::Matrix<ndim, 1, T> r_s(Core::LinAlg::Initialization::zero);

  // my random number vector at current GP
  Core::LinAlg::Matrix<ndim, 1, double> randnumvec(Core::LinAlg::Initialization::zero);

  // stochastic force vector per unit length at current GP
  Core::LinAlg::Matrix<ndim, 1, T> f_stoch(Core::LinAlg::Initialization::zero);


  // get Gauss points and weights for evaluation of damping matrix
  Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);

  // matrix to store Hermite shape functions and their derivatives evaluated at a certain Gauss
  // point
  Core::LinAlg::Matrix<1, nnode * vpernode, double> N_i;
  Core::LinAlg::Matrix<1, nnode * vpernode, double> N_i_xi;

  for (int gp = 0; gp < gausspoints.nquad; gp++)
  {
    Discret::Utils::Beam::evaluate_shape_functions_and_derivs_at_xi<nnode, vpernode>(
        gausspoints.qxg[gp][0], N_i, N_i_xi, this->shape(), this->ref_length());

    // compute tangent vector t_{\par}=r' at current Gauss point
    calc_r_s<nnode, vpernode, T>(disp_totlag, N_i_xi, jacobi_[gp], r_s);

    // extract random numbers from global vector
    for (unsigned int idim = 0; idim < ndim; idim++)
      randnumvec(idim) = (*randomforces)(gp * randompergauss + idim).local_values_as_span()[lid()];

    // compute stochastic force vector per unit length at current GP
    f_stoch.clear();
    for (unsigned int idim = 0; idim < ndim; idim++)
      for (unsigned int jdim = 0; jdim < ndim; jdim++)
        f_stoch(idim) += (std::sqrt(gamma(1)) * (idim == jdim) +
                             (std::sqrt(gamma(0)) - std::sqrt(gamma(1))) * r_s(idim) * r_s(jdim)) *
                         randnumvec(jdim);

    const double sqrt_jacobifac_gp_weight = std::sqrt(jacobi_[gp] * gausspoints.qwgt[gp]);


    // loop over all nodes used for centerline interpolation
    for (unsigned int inode = 0; inode < nnode; inode++)
      // loop over dimensions
      for (unsigned int idim = 0; idim < ndim; idim++)
      {
        f_int(inode * dofpernode + idim) -=
            N_i(vpernode * inode) * f_stoch(idim) * sqrt_jacobifac_gp_weight;
        f_int(inode * dofpernode + 3 + idim) -=
            N_i(vpernode * inode + 1) * f_stoch(idim) * sqrt_jacobifac_gp_weight;
      }


    if (stiffmatrix != nullptr)
    {
      evaluate_analytic_stiffmat_contributions_from_stochastic_forces<nnode, vpernode, ndim>(
          *stiffmatrix, r_s, randnumvec, gamma, N_i, N_i_xi, jacobi_[gp], gausspoints.qwgt[gp]);
    }
  }
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
void Discret::Elements::Beam3k::evaluate_analytic_stiffmat_contributions_from_stochastic_forces(
    Core::LinAlg::SerialDenseMatrix& stiffmatrix, const Core::LinAlg::Matrix<ndim, 1, double>& r_s,
    const Core::LinAlg::Matrix<ndim, 1, double>& randnumvec,
    const Core::LinAlg::Matrix<ndim, 1, double>& gamma,
    const Core::LinAlg::Matrix<1, nnode * vpernode, double>& N_i,
    const Core::LinAlg::Matrix<1, nnode * vpernode, double>& N_i_xi, double jacobifactor,
    double gp_weight) const
{
  /* only nodes for centerline interpolation are considered here (first two nodes of this element)
     each of these nodes holds 3*vpernode translational DoFs AND 1 rotational DoFs */
  const unsigned int dofpernode = ndim * vpernode + 1;

  // note: division by sqrt of jacobi factor, because H_i_s = H_i_xi / jacobifactor
  const double sqrt_gp_weight_jacobifac_inv = std::sqrt(gp_weight / jacobifactor);

  const double prefactor =
      sqrt_gp_weight_jacobifac_inv * (std::sqrt(gamma(0)) - std::sqrt(gamma(1)));


  // loop over all nodes used for centerline interpolation
  for (unsigned int inode = 0; inode < nnode; inode++)
    // loop over all column nodes used for centerline interpolation
    for (unsigned int jnode = 0; jnode < nnode; jnode++)
    {
      for (unsigned int idim = 0; idim < ndim; idim++)
        for (unsigned int jdim = 0; jdim < ndim; jdim++)
        {
          stiffmatrix(inode * dofpernode + idim, jnode * dofpernode + idim) -=
              N_i(vpernode * inode) * N_i_xi(vpernode * jnode) * r_s(jdim) * randnumvec(jdim) *
              prefactor;
          stiffmatrix(inode * dofpernode + idim, jnode * dofpernode + jdim) -=
              N_i(vpernode * inode) * N_i_xi(vpernode * jnode) * r_s(idim) * randnumvec(jdim) *
              prefactor;

          stiffmatrix(inode * dofpernode + 3 + idim, jnode * dofpernode + idim) -=
              N_i(vpernode * inode + 1) * N_i_xi(vpernode * jnode) * r_s(jdim) * randnumvec(jdim) *
              prefactor;
          stiffmatrix(inode * dofpernode + 3 + idim, jnode * dofpernode + jdim) -=
              N_i(vpernode * inode + 1) * N_i_xi(vpernode * jnode) * r_s(idim) * randnumvec(jdim) *
              prefactor;

          stiffmatrix(inode * dofpernode + idim, jnode * dofpernode + 3 + idim) -=
              N_i(vpernode * inode) * N_i_xi(vpernode * jnode + 1) * r_s(jdim) * randnumvec(jdim) *
              prefactor;
          stiffmatrix(inode * dofpernode + idim, jnode * dofpernode + 3 + jdim) -=
              N_i(vpernode * inode) * N_i_xi(vpernode * jnode + 1) * r_s(idim) * randnumvec(jdim) *
              prefactor;

          stiffmatrix(inode * dofpernode + 3 + idim, jnode * dofpernode + 3 + idim) -=
              N_i(vpernode * inode + 1) * N_i_xi(vpernode * jnode + 1) * r_s(jdim) *
              randnumvec(jdim) * prefactor;
          stiffmatrix(inode * dofpernode + 3 + idim, jnode * dofpernode + 3 + jdim) -=
              N_i(vpernode * inode + 1) * N_i_xi(vpernode * jnode + 1) * r_s(idim) *
              randnumvec(jdim) * prefactor;
        }
    }
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <typename T, unsigned int nnode, unsigned int vpernode, unsigned int ndim>
void Discret::Elements::Beam3k::evaluate_rotational_damping(
    const Core::LinAlg::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, T>&
        disp_totlag_centerline,
    const std::vector<Core::LinAlg::Matrix<ndim, ndim, T>>& triad_mat_cp,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix,
    Core::LinAlg::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, T>& f_int)
{
  // get time step size
  const double dt = params_interface().get_delta_time();

  // get damping coefficients for translational and rotational degrees of freedom
  Core::LinAlg::Matrix<3, 1> gamma(Core::LinAlg::Initialization::zero);
  get_damping_coefficients(gamma);

  // get Gauss points and weights for evaluation of viscous damping contributions
  Core::FE::IntegrationPoints1D gausspoints = Core::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);

  Core::LinAlg::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, T> f_int_aux(
      Core::LinAlg::Initialization::zero);

  // CP values of strains and their variations needed for interpolation
  std::vector<Core::LinAlg::Matrix<6 * nnode + BEAM3K_COLLOCATION_POINTS, 3, T>> v_thetapar_cp(
      BEAM3K_COLLOCATION_POINTS);

  // re-interpolated values of strains and their variations evaluated at Gauss points
  Core::LinAlg::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 3, T> v_thetapar_bar(
      Core::LinAlg::Initialization::zero);

  std::vector<Core::LinAlg::Matrix<ndim, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, T>>
      lin_theta_cp(BEAM3K_COLLOCATION_POINTS);

  // Interpolated material triad and local rotation vector evaluated at Gauss point
  Core::LinAlg::Matrix<3, 3, T> triad_mat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> theta(Core::LinAlg::Initialization::zero);


  // matrices holding the assembled shape functions and s-derivatives
  Core::LinAlg::Matrix<3, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, T> N_s;
  Core::LinAlg::Matrix<1, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, T> L;

  // Matrices for individual shape functions and xi-derivatives
  Core::LinAlg::Matrix<1, vpernode * nnode, double> N_i_xi;
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i;

  // Additional kinematic quantities
  Core::LinAlg::Matrix<3, 1, T> r_s;  // Matrix to store r'
  T abs_r_s;                          // ||r'||

  // create object of triad interpolation scheme
  std::shared_ptr<
      LargeRotations::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS, T>>
      triad_interpolation_scheme_ptr(
          new LargeRotations::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS,
              T>());

  // reset scheme with nodal triads
  triad_interpolation_scheme_ptr->reset(triad_mat_cp);


  //********begin: evaluate quantities at collocation points********************************
  for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; ++node)
  {
    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    const double xi_cp = (double)node / (double)(BEAM3K_COLLOCATION_POINTS - 1) * 2.0 - 1.0;

    // Determine storage position for the node node
    const unsigned int ind =
        Core::LargeRotations::numbering_trafo(node + 1, BEAM3K_COLLOCATION_POINTS);

    // get value of interpolating function of theta (Lagrange polynomials) at xi
    L_i.clear();
    Core::FE::shape_function_1d(L_i, xi_cp, shape());

    L.clear();
    assemble_shapefunctions_l(L_i, L);

    N_i_xi.clear();
    Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi_cp, length_, Core::FE::CellType::line2);

    N_s.clear();
    assemble_shapefunctions_ns(N_i_xi, jacobi_cp_[ind], N_s);

    // Calculation of r' at xi
    r_s.clear();
    r_s.multiply(N_s, disp_totlag_centerline);

    abs_r_s = Core::FADUtils::norm<T>(r_s);

    calc_v_thetapartheta<2, T>(v_thetapar_cp[ind], L, r_s, abs_r_s);

    pre_compute_terms_at_cp_for_analytic_stiffmat_contributions_from_rotational_damping<2, 2, 3>(
        lin_theta_cp[ind], L, N_s, r_s, abs_r_s, qrefconv_[ind]);
  }


  // loop through Gauss points
  for (int gp = 0; gp < gausspoints.nquad; ++gp)
  {
    // get location and weight of GP in parameter space
    const double xi_gp = gausspoints.qxg[gp][0];
    const double wgt = gausspoints.qwgt[gp];

    // evaluate shape functions
    L_i.clear();
    Core::FE::shape_function_1d(L_i, xi_gp, this->shape());

    v_thetapar_bar.clear();
    for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; ++node)
    {
      v_thetapar_bar.update(L_i(node), v_thetapar_cp[node], 1.0);
    }

    // compute material triad at gp
    triad_mat.clear();

    // compute quaternion of material triad at gp
    Core::LinAlg::Matrix<4, 1, T> Qnewmass(Core::LinAlg::Initialization::zero);

    triad_interpolation_scheme_ptr->get_interpolated_local_rotation_vector(theta, L_i);

    triad_interpolation_scheme_ptr->get_interpolated_quaternion(Qnewmass, theta);

    Core::LargeRotations::quaterniontotriad(Qnewmass, triad_mat);

    // store in class variable in order to get QconvGPmass_ in subsequent time step
    for (unsigned int i = 0; i < 4; ++i)
      (qnewmass_[gp])(i) = Core::FADUtils::cast_to_double(Qnewmass(i));

    Core::LinAlg::Matrix<4, 1, T> Qconv(Core::LinAlg::Initialization::zero);
    for (unsigned int i = 0; i < 4; ++i) Qconv(i) = (qconvmass_[gp])(i);

    Core::LinAlg::Matrix<3, 3, T> triad_mat_conv(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::quaterniontotriad(Qconv, triad_mat_conv);


    // compute quaternion of relative rotation from converged to current state
    Core::LinAlg::Matrix<3, 3, T> deltatriad(Core::LinAlg::Initialization::zero);
    deltatriad.multiply_nt(triad_mat, triad_mat_conv);

    Core::LinAlg::Matrix<4, 1, T> deltaQ(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::triadtoquaternion(deltatriad, deltaQ);

    Core::LinAlg::Matrix<3, 1, T> deltatheta(Core::LinAlg::Initialization::zero);
    Core::LargeRotations::quaterniontoangle(deltaQ, deltatheta);


    // angular velocity at this Gauss point according to backward Euler scheme
    Core::LinAlg::Matrix<3, 1, T> omega(Core::LinAlg::Initialization::zero);
    omega.update(1.0 / dt, deltatheta);

    // compute matrix Lambda*[gamma(2) 0 0 \\ 0 0 0 \\ 0 0 0]*Lambda^t = gamma(2) * g_1 \otimes g_1
    // where g_1 is first base vector, i.e. first column of Lambda
    Core::LinAlg::Matrix<3, 3, T> g1g1gamma(Core::LinAlg::Initialization::zero);
    for (unsigned int k = 0; k < 3; ++k)
      for (unsigned int j = 0; j < 3; ++j)
        g1g1gamma(k, j) = triad_mat(k, 0) * triad_mat(j, 0) * gamma(2);

    // compute vector gamma(2) * g_1 \otimes g_1 * \omega (viscous moment per unit length)
    Core::LinAlg::Matrix<3, 1, T> m_visc(Core::LinAlg::Initialization::zero);
    m_visc.multiply(g1g1gamma, omega);


    // residual contribution from viscous damping moment
    f_int_aux.clear();
    f_int_aux.multiply(v_thetapar_bar, m_visc);

    f_int.update(wgt * jacobi_[gp], f_int_aux, 1.0);


    if (stiffmatrix != nullptr)
    {
      evaluate_analytic_stiffmat_contributions_from_rotational_damping<2, 2, 3>(*stiffmatrix,
          disp_totlag_centerline, *triad_interpolation_scheme_ptr, theta, deltatheta, triad_mat,
          triad_mat_conv, v_thetapar_bar, lin_theta_cp, m_visc, gamma(2), dt, xi_gp,
          wgt * jacobi_[gp]);
    }
  }
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, unsigned int vpernode, unsigned int ndim>
void Discret::Elements::Beam3k::evaluate_analytic_stiffmat_contributions_from_rotational_damping(
    Core::LinAlg::SerialDenseMatrix& stiffmatrix,
    const Core::LinAlg::Matrix<ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>&
        disp_totlag_centerline,
    const LargeRotations::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS, double>&
        triad_intpol,
    const Core::LinAlg::Matrix<3, 1, double> theta_gp,
    const Core::LinAlg::Matrix<3, 1, double>& deltatheta_gp,
    const Core::LinAlg::Matrix<3, 3, double>& triad_mat_gp,
    const Core::LinAlg::Matrix<3, 3, double>& triad_mat_conv_gp,
    const Core::LinAlg::Matrix<ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS, ndim, double>&
        v_theta_par_bar,
    const std::vector<Core::LinAlg::Matrix<ndim,
        ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS, double>>& lin_theta_cp,
    const Core::LinAlg::Matrix<3, 1, double> moment_viscous, double gamma_polar, double dt,
    double xi_gp, double jacobifac_GPwgt) const
{
  // size of Dof vector of this element
  const unsigned int numdofelement = ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS;

  // create a fixed size matrix as view on the Core::LinAlg::SerialDenseMatrix to avoid copying
  Core::LinAlg::Matrix<numdofelement, numdofelement, double> stiffmatrix_fixedsize(
      stiffmatrix, true);


  // matrices storing the assembled shape functions or s-derivative
  Core::LinAlg::Matrix<ndim, numdofelement, double> N_s;
  Core::LinAlg::Matrix<1, numdofelement, double> L;

  // matrices storing individual shape functions, its xi-derivatives and s-derivatives
  Core::LinAlg::Matrix<1, vpernode * nnodecl, double> N_i_xi;
  Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i;

  // r' vector and its norm
  Core::LinAlg::Matrix<3, 1, double> r_s_cp(Core::LinAlg::Initialization::zero);
  double abs_r_s_cp = 0.0;

  // first base vector
  Core::LinAlg::Matrix<3, 1, double> g_1_cp(Core::LinAlg::Initialization::zero);


  Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> lin_theta_bar(
      Core::LinAlg::Initialization::zero);


  // linearization of re-interpolated strain variations
  std::vector<Core::LinAlg::Matrix<numdofelement, numdofelement, double>> lin_v_thetapar_moment_cp(
      BEAM3K_COLLOCATION_POINTS);

  Core::LinAlg::Matrix<numdofelement, numdofelement, double> lin_v_thetapar_bar_moment(
      Core::LinAlg::Initialization::zero);


  Core::LinAlg::Matrix<3, 3, double> spinmatrix_of_moment_visc(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::computespin<double>(spinmatrix_of_moment_visc, moment_viscous);


  // linearization of moment_visc
  Core::LinAlg::Matrix<3, numdofelement, double> lin_moment_viscous(
      Core::LinAlg::Initialization::zero);


  /***********************************************************************************************/
  // note: we need an additional loop over the collocation points here for all quantities that
  //       would be third order tensors if not multiplied by the associated vector (in this case
  //       moment vector); since the vector is only available within the loop over the Gauss points
  //       (i.e. at this current GP), we compute right here the lin_v_theta*_moment terms in an
  //       extra loop over the collocation points

  double xi_cp = 0.0;    // parameter coordinate
  unsigned int ind = 0;  // position index where CP quantities have to be stored

  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    // Determine storage position for this cp
    ind = Core::LargeRotations::numbering_trafo(icp + 1, BEAM3K_COLLOCATION_POINTS);

    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    xi_cp = (double)icp / (double)(BEAM3K_COLLOCATION_POINTS - 1) * 2.0 - 1.0;


    // get all required shape function values
    L_i.clear();
    Core::FE::shape_function_1d(L_i, xi_cp, shape());

    L.clear();
    assemble_shapefunctions_l(L_i, L);

    N_i_xi.clear();
    Core::FE::shape_function_hermite_1d_deriv1(N_i_xi, xi_cp, length_, Core::FE::CellType::line2);

    N_s.clear();
    assemble_shapefunctions_ns(N_i_xi, jacobi_cp_[ind], N_s);


    // Calculation of r' and g_1
    r_s_cp.clear();
    r_s_cp.multiply(N_s, disp_totlag_centerline);

    abs_r_s_cp = r_s_cp.norm2();  // Todo think about computing and storing inverse value here

    g_1_cp.clear();
    g_1_cp.update(std::pow(abs_r_s_cp, -1.0), r_s_cp);


    calc_lin_v_thetapar_moment<nnodecl>(
        lin_v_thetapar_moment_cp[ind], L, N_s, g_1_cp, abs_r_s_cp, moment_viscous);
  }

  /***********************************************************************************************/
  // re-interpolation of quantities at xi based on CP values

  L_i.clear();
  Core::FE::shape_function_1d(L_i, xi_gp, shape());

  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    lin_v_thetapar_bar_moment.update(L_i(icp), lin_v_thetapar_moment_cp[icp], 1.0);
  }
  /***********************************************************************************************/

  std::vector<Core::LinAlg::Matrix<3, 3, double>> Itilde(BEAM3K_COLLOCATION_POINTS);

  // compute Itilde matrices required for re-interpolation of CP values of lin_theta
  triad_intpol.get_nodal_generalized_rotation_interpolation_matrices(Itilde, theta_gp, L_i);

  Core::LinAlg::Matrix<3, numdofelement, double> auxmatrix(Core::LinAlg::Initialization::zero);

  lin_theta_bar.clear();
  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    auxmatrix.clear();

    auxmatrix.multiply(Itilde[icp], lin_theta_cp[icp]);

    lin_theta_bar.update(1.0, auxmatrix, 1.0);
  }

  calc_lin_moment_viscous<nnodecl>(lin_moment_viscous, triad_mat_gp, triad_mat_conv_gp,
      deltatheta_gp, lin_theta_bar, spinmatrix_of_moment_visc, gamma_polar, dt);

  /***********************************************************************************************/
  // finally put everything together
  Core::LinAlg::Matrix<numdofelement, numdofelement, double> auxmatrix2(
      Core::LinAlg::Initialization::zero);

  // linearization of residual from rotational damping moment
  stiffmatrix_fixedsize.update(jacobifac_GPwgt, lin_v_thetapar_bar_moment, 1.0);

  auxmatrix2.clear();
  auxmatrix2.multiply(v_theta_par_bar, lin_moment_viscous);

  stiffmatrix_fixedsize.update(jacobifac_GPwgt, auxmatrix2, 1.0);
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
void Discret::Elements::Beam3k::
    pre_compute_terms_at_cp_for_analytic_stiffmat_contributions_from_rotational_damping(
        Core::LinAlg::Matrix<ndim, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, double>&
            lin_theta,
        const Core::LinAlg::Matrix<1, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, double>&
            L,
        const Core::LinAlg::Matrix<ndim, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS,
            double>& N_s,
        const Core::LinAlg::Matrix<ndim, 1, double>& r_s, double abs_r_s,
        const Core::LinAlg::Matrix<4, 1, double>& Qref_conv) const
{
  // size of Dof vector of this element
  const unsigned int numdofelement = ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS;

  Core::LinAlg::Matrix<ndim, 1, double> g_1(Core::LinAlg::Initialization::zero);
  g_1.update(std::pow(abs_r_s, -1.0), r_s);

  Core::LinAlg::Matrix<ndim, 1, double> g_1_bar(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 3, double> triad_ref_conv_cp(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::quaterniontotriad(Qref_conv, triad_ref_conv_cp);

  g_1_bar.clear();
  for (unsigned int idim = 0; idim < ndim; ++idim) g_1_bar(idim) = triad_ref_conv_cp(idim, 0);

  // CP values of strain increments
  Core::LinAlg::Matrix<ndim, numdofelement, double> lin_theta_perp(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<ndim, numdofelement, double> lin_theta_par(
      Core::LinAlg::Initialization::zero);

  calc_lin_thetapar<nnode>(lin_theta_par, L, N_s, g_1, g_1_bar, abs_r_s);

  calc_lin_thetaperp<nnode>(lin_theta_perp, N_s, r_s, abs_r_s);

  // lin_theta
  lin_theta.clear();
  lin_theta.update(1.0, lin_theta_par, 1.0, lin_theta_perp);
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
int Discret::Elements::Beam3k::how_many_random_numbers_i_need() const
{
  // get Gauss rule for evaluation of stochastic force contributions
  Core::FE::GaussRule1D gaussrule = MYGAUSSRULEBEAM3K;
  Core::FE::IntegrationPoints1D gausspoints(gaussrule);

  /* at each Gauss point one needs as many random numbers as randomly excited degrees of freedom,
   * i.e. three random numbers for the translational degrees of freedom */
  return (3 * gausspoints.nquad);
}

/*-----------------------------------------------------------------------------------------------------------*
 |  Assemble C shape function meier 01/16|
 *-----------------------------------------------------------------------------------------------------------*/
template <typename T1, typename T2>
void Discret::Elements::Beam3k::assemble_shapefunctions_l(
    Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, T1>& L_i,
    Core::LinAlg::Matrix<1, 2 * 6 + BEAM3K_COLLOCATION_POINTS, T2>& L) const
{
#if defined(BEAM3K_COLLOCATION_POINTS) && (BEAM3K_COLLOCATION_POINTS == 2)

  unsigned int assembly_L[2 * 6 + BEAM3K_COLLOCATION_POINTS] = {
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2};

#elif defined(BEAM3K_COLLOCATION_POINTS) && (BEAM3K_COLLOCATION_POINTS == 3)

  unsigned int assembly_L[2 * 6 + BEAM3K_COLLOCATION_POINTS] = {
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3};

#elif defined(BEAM3K_COLLOCATION_POINTS) && (BEAM3K_COLLOCATION_POINTS == 4)

  unsigned int assembly_L[2 * 6 + BEAM3K_COLLOCATION_POINTS] = {
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 3, 4};

#else
  FOUR_C_THROW(
      "BEAM3K_COLLOCATION_POINTS has to be defined. Only the values 2,3 and 4 are valid for "
      "BEAM3K_COLLOCATION_POINTS!!!");
#endif

  // Assemble the matrices of the shape functions
  for (unsigned int i = 0; i < 2 * 6 + BEAM3K_COLLOCATION_POINTS; ++i)
  {
    if (assembly_L[i] == 0)
    {
      L(i) = 0.0;
    }
    else
    {
      L(i) = L_i(assembly_L[i] - 1);
    }
  }
}

/*-----------------------------------------------------------------------------------------------------------*
 |  Assemble the N_s shape functions meier 01/16|
 *-----------------------------------------------------------------------------------------------------------*/
template <typename T1, typename T2>
void Discret::Elements::Beam3k::assemble_shapefunctions_nss(Core::LinAlg::Matrix<1, 4, T1>& N_i_xi,
    Core::LinAlg::Matrix<1, 4, T1>& N_i_xixi, double jacobi, double jacobi2,
    Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, T2>& N_ss) const
{
  Core::LinAlg::Matrix<1, 4, T1> N_i_ss(Core::LinAlg::Initialization::zero);
  N_i_ss.update(std::pow(jacobi, -2.0), N_i_xixi, 1.0);
  N_i_ss.update(-1.0 * jacobi2 * std::pow(jacobi, -4.0), N_i_xi, 1.0);

  assemble_shapefunctions_n(N_i_ss, N_ss);
}

/*-----------------------------------------------------------------------------------------------------------*
 |  Assemble the N_s shape functions meier 01/16|
 *-----------------------------------------------------------------------------------------------------------*/
template <typename T1, typename T2>
void Discret::Elements::Beam3k::assemble_shapefunctions_ns(Core::LinAlg::Matrix<1, 4, T1>& N_i_xi,
    double jacobi, Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, T2>& N_s) const
{
  Core::LinAlg::Matrix<1, 4, T1> N_i_s(Core::LinAlg::Initialization::zero);

  // Calculate the derivatives in s
  N_i_s = N_i_xi;
  N_i_s.scale(std::pow(jacobi, -1.0));

  assemble_shapefunctions_n(N_i_s, N_s);
}

/*-----------------------------------------------------------------------------------------------------------*
 |  Assemble the N shape functions meier 01/16|
 *-----------------------------------------------------------------------------------------------------------*/
template <typename T1, typename T2>
void Discret::Elements::Beam3k::assemble_shapefunctions_n(Core::LinAlg::Matrix<1, 4, T1>& N_i,
    Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, T2>& N) const
{
#if defined(BEAM3K_COLLOCATION_POINTS) && (BEAM3K_COLLOCATION_POINTS == 2)

  unsigned int assembly_N[3][2 * 6 + BEAM3K_COLLOCATION_POINTS] = {
      {1, 0, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0}, {0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0, 0},
      {0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0}};

#elif defined(BEAM3K_COLLOCATION_POINTS) && (BEAM3K_COLLOCATION_POINTS == 3)

  unsigned int assembly_N[3][2 * 6 + BEAM3K_COLLOCATION_POINTS] = {
      {1, 0, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0, 0}, {0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0},
      {0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0, 0}};

#elif defined(BEAM3K_COLLOCATION_POINTS) && (BEAM3K_COLLOCATION_POINTS == 4)

  unsigned int assembly_N[3][2 * 6 + BEAM3K_COLLOCATION_POINTS] = {
      {1, 0, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0, 0, 0},
      {0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0, 0},
      {0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0}};

#else
  FOUR_C_THROW(
      "BEAM3K_COLLOCATION_POINTS has to be defined. Only the values 2,3 and 4 are valid for "
      "BEAM3K_COLLOCATION_POINTS!!!");
#endif

  // Assemble the matrices of the shape functions
  for (unsigned int i = 0; i < 2 * 6 + BEAM3K_COLLOCATION_POINTS; ++i)
  {
    for (unsigned int j = 0; j < 3; ++j)
    {
      if (assembly_N[j][i] == 0)
      {
        N(j, i) = 0.0;
      }
      else
      {
        N(j, i) = N_i(assembly_N[j][i] - 1);
      }
    }
  }
}

/*-----------------------------------------------------------------------------------------------------------*
 |  Pre-multiply trafo matrix if rotvec_==true: \tilde{\vec{f}_int}=\mat{T}^T*\vec{f}_int meier
 01/16|
 *-----------------------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void Discret::Elements::Beam3k::apply_rot_vec_trafo(
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
        disp_totlag_centerline,
    Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& f_int) const
{
  // Trafo matrices:
  Core::LinAlg::Matrix<4, 4, T> trafomat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> g_1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, T> auxmatrix(Core::LinAlg::Initialization::zero);
  T t = 0.0;
  Core::LinAlg::Matrix<4, 1, T> f_aux1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1, T> f_aux2(Core::LinAlg::Initialization::zero);

  for (unsigned int node = 0; node < 2; ++node)
  {
    g_1.clear();
    t = 0.0;
    auxmatrix.clear();

    for (unsigned int i = 0; i < 3; ++i)
    {
      g_1(i) = disp_totlag_centerline(7 * node + 3 + i);
    }

    t = Core::FADUtils::norm<T>(g_1);
    g_1.scale(1.0 / t);
    Core::LargeRotations::computespin(auxmatrix, g_1);
    auxmatrix.scale(-1.0 * t);
    trafomat.clear();

    for (unsigned int i = 0; i < 3; ++i)
    {
      for (unsigned int j = 0; j < 3; ++j)
      {
        trafomat(i, j) = auxmatrix(i, j);
      }
      trafomat(i, 3) = g_1(i);
      trafomat(3, i) = g_1(i);
    }

    f_aux1.clear();
    f_aux2.clear();
    for (unsigned int i = 0; i < 4; ++i)
    {
      f_aux1(i) = f_int(7 * node + 3 + i);
    }

    f_aux2.multiply_tn(trafomat, f_aux1);

    for (unsigned int i = 0; i < 4; ++i)
    {
      f_int(7 * node + 3 + i) = f_aux2(i);
    }
  }
}

/*-----------------------------------------------------------------------------------------------------------*
 |  Transform stiffness matrix in order to solve for multiplicative rotation vector increments meier
 01/16|
 *-----------------------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void Discret::Elements::Beam3k::transform_stiff_matrix_multiplicative(
    Core::LinAlg::SerialDenseMatrix* stiffmatrix,
    const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag) const
{
  // we need to transform the stiffmatrix because its entries are derivatives with respect to
  // additive rotational increments we want a stiffmatrix containing derivatives with respect to
  // multiplicative rotational increments therefore apply a trafo matrix to all those 3x3 blocks in
  // stiffmatrix which correspond to derivation with respect to rotational DOFs the trafo matrix is
  // simply the T-Matrix (see Jelenic1999, (2.4)): \Delta_{mult} \vec \theta_{inode} = \mat T(\vec
  // \theta_{inode} * \Delta_{addit} \vec \theta_{inode}
  Core::LinAlg::Matrix<2 * 6 + BEAM3K_COLLOCATION_POINTS, 3> tempmat(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<2 * 6 + BEAM3K_COLLOCATION_POINTS, 3> newstiffmat(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> Tmat(Core::LinAlg::Initialization::zero);
  std::vector<Core::LinAlg::Matrix<3, 1>> theta(
      2, Core::LinAlg::Matrix<3, 1>(Core::LinAlg::Initialization::zero));

  // Loop over the two boundary nodes
  for (unsigned int node = 0; node < 2; node++)
  {
    for (unsigned int i = 0; i < 3; ++i)
      theta[node](i) = Core::FADUtils::cast_to_double(disp_totlag(7 * node + 3 + i));
  }

  // Loop over the two boundary nodes
  for (unsigned int node = 0; node < 2; ++node)
  {
    Tmat.clear();
    Tmat = Core::LargeRotations::tmatrix(theta[node]);

    tempmat.clear();
    for (unsigned int i = 0; i < 2 * 6 + BEAM3K_COLLOCATION_POINTS; ++i)
      for (unsigned int j = 0; j < 3; ++j) tempmat(i, j) = (*stiffmatrix)(i, 7 * node + 3 + j);

    newstiffmat.clear();
    newstiffmat.multiply_nn(tempmat, Tmat);

    for (unsigned int i = 0; i < 2 * 6 + BEAM3K_COLLOCATION_POINTS; ++i)
      for (unsigned int j = 0; j < 3; ++j) (*stiffmatrix)(i, 7 * node + 3 + j) = newstiffmat(i, j);
  }
}

template <typename T>
void Discret::Elements::Beam3k::straintostress(const Core::LinAlg::Matrix<3, 1, T>& Omega,
    const T& epsilon, const Core::LinAlg::Matrix<3, 3, T>& Cn,
    const Core::LinAlg::Matrix<3, 3, T>& Cm, Core::LinAlg::Matrix<3, 1, T>& M, T& f_par) const
{
  f_par = 0.0;
  f_par = Cn(0, 0) * epsilon;

  M.clear();
  M(0) = Cm(0, 0) * Omega(0);
  M(1) = Cm(1, 1) * Omega(1);
  M(2) = Cm(2, 2) * Omega(2);
}


void Discret::Elements::Beam3k::calc_stiff_contributions_ptc(
    Core::LinAlg::SerialDenseMatrix& elemat1)
{
  elemat1 = stiff_ptc_;
}

//      //*******************************Begin:
//      FD-CHECK************************************************************
//      //the following code block can be used to check quickly whether the nonlinear stiffness
//      matrix is calculated
//      //correctly or not by means of a numerically approximated stiffness matrix. Uncomment this
//      code block and copy
//      //it to the marked place in the method Discret::Elements::Beam3k::evaluate() on the top of
//      this file! if(Id() == 0) //limiting the following tests to certain element numbers
//      {
//
//        //variable to store numerically approximated stiffness matrix
//        Core::LinAlg::SerialDenseMatrix stiff_approx;
//        stiff_approx.Shape(6*2+BEAM3K_COLLOCATION_POINTS,6*2+BEAM3K_COLLOCATION_POINTS);
//
//
//        //relative error of numerically approximated stiffness matrix
//        Core::LinAlg::SerialDenseMatrix stiff_relerr;
//        stiff_relerr.Shape(6*2+BEAM3K_COLLOCATION_POINTS,6*2+BEAM3K_COLLOCATION_POINTS);
//
//        //characteristic length for numerical approximation of stiffness
//        double h_rel = 1e-7;
//
//        //flag indicating whether approximation leads to significant relative error
//        int outputflag = 0;
//
//        //calculating strains in new configuration
//        for(int i=0; i<6*2+BEAM3K_COLLOCATION_POINTS; i++) //for all dof
//        {
//          Core::LinAlg::SerialDenseVector force_aux;
//          force_aux.Size(6*2+BEAM3K_COLLOCATION_POINTS);
//
//          //create new displacement and velocity vectors in order to store artificially modified
//          displacements std::vector<double> vel_aux(myvel); std::vector<double> disp_aux(mydisp);
//
//          //modifying displacement artificially (for numerical derivative of internal forces):
//          disp_aux[i] += h_rel;
//          vel_aux[i] += h_rel / params.get<double>("delta time",0.01);
//
//          if(weakkirchhoff_)
//            CalculateInternalForcesWK(params,disp_aux,nullptr,nullptr,&force_aux,nullptr,false);
//          else
//            CalculateInternalForcesSK(params,disp_aux,nullptr,nullptr,&force_aux,nullptr,false);
//
//          //computing derivative d(fint)/du numerically by finite difference
//          for(int u = 0 ; u < 6*2+BEAM3K_COLLOCATION_POINTS ; u++ )
//          {
//            stiff_approx(u,i)= ( force_aux[u] - elevec1(u) )/ h_rel ;
//          }
//        } //for(int i=0; i<3; i++) //for all dof
//
//
//        for(int line=0; line<6*2+BEAM3K_COLLOCATION_POINTS; line++)
//        {
//          for(int col=0; col<6*2+BEAM3K_COLLOCATION_POINTS; col++)
//          {
//            if (fabs(elemat1(line,col)) > 1.0e-10)
//              stiff_relerr(line,col)= fabs( ( elemat1(line,col) - stiff_approx(line,col) )/
//              elemat1(line,col) );
//            else if (fabs(stiff_approx(line,col)) < 1.0e-5)
//              stiff_relerr(line,col)=0.0;
//            else
//              stiff_relerr(line,col)=1000.0;
//
//            //suppressing small entries whose effect is only confusing and NaN entries (which
//            arise due to zero entries) if ( fabs( stiff_relerr(line,col) ) < h_rel*500 || isnan(
//            stiff_relerr(line,col))) //isnan = is not a number
//              stiff_relerr(line,col) = 0;
//
//            //if ( stiff_relerr(line,col) > 0)
//              outputflag = 1;
//          } //for(int col=0; col<3*nnode; col++)
//
//        } //for(int line=0; line<3*nnode; line++)
//
//        if(outputflag ==1)
//        {
//
//          std::cout<<"\n\n actually calculated stiffness matrix\n";
//          for(int line=0; line<6*2+BEAM3K_COLLOCATION_POINTS; line++)
//          {
//            for(int col=0; col<6*2+BEAM3K_COLLOCATION_POINTS; col++)
//            {
//              if(isnan(elemat1(line,col)))
//                std::cout<<"     nan   ";
//              else if(elemat1(line,col) == 0)
//                std::cout<<"     0     ";
//              else if(elemat1(line,col) >= 0)
//                std::cout<<"  "<< std::scientific << std::setprecision(3)<<elemat1(line,col);
//              else
//                std::cout<<" "<< std::scientific << std::setprecision(3)<<elemat1(line,col);
//            }
//            std::cout<<"\n";
//          }
//
//          std::cout<<"\n\n approximated stiffness matrix\n";
//          for(int line=0; line<6*2+BEAM3K_COLLOCATION_POINTS; line++)
//          {
//            for(int col=0; col<6*2+BEAM3K_COLLOCATION_POINTS; col++)
//            {
//              if(isnan(stiff_approx(line,col)))
//                std::cout<<"     nan   ";
//              else if(stiff_approx(line,col) == 0)
//                std::cout<<"     0     ";
//              else if(stiff_approx(line,col) >= 0)
//                std::cout<<"  "<< std::scientific << std::setprecision(3)<<stiff_approx(line,col);
//              else
//                std::cout<<" "<< std::scientific << std::setprecision(3)<<stiff_approx(line,col);
//            }
//            std::cout<<"\n";
//          }
//
//          std::cout<<"\n\n rel error stiffness matrix\n";
//          for(int line=0; line<6*2+BEAM3K_COLLOCATION_POINTS; line++)
//          {
//            for(int col=0; col<6*2+BEAM3K_COLLOCATION_POINTS; col++)
//            {
//              if(isnan(stiff_relerr(line,col)))
//                std::cout<<"     nan   ";
//              else if(stiff_relerr(line,col) == 0)
//                std::cout<<"     0     ";
//              else if(stiff_relerr(line,col) >= 0)
//                std::cout<<"  "<< std::scientific << std::setprecision(3)<<stiff_relerr(line,col);
//              else
//                std::cout<<" "<< std::scientific << std::setprecision(3)<<stiff_relerr(line,col);
//            }
//            std::cout<<"\n";
//          }
//        }
//
//      } //end of section in which numerical approximation for stiffness matrix is computed
//      //*******************************End:
//      FD-CHECK************************************************************


// explicit template instantiations
template void Discret::Elements::Beam3k::calculate_internal_forces_and_stiff_wk<2, double>(
    Teuchos::ParameterList&,
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<3, 3, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    std::vector<Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 3, double>>&,
    std::vector<Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>>&,
    std::vector<Core::LinAlg::Matrix<3, 3, double>>&);
template void Discret::Elements::Beam3k::calculate_internal_forces_and_stiff_wk<2, FAD>(
    Teuchos::ParameterList&, const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    const std::vector<Core::LinAlg::Matrix<3, 3, FAD>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    std::vector<Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 3, FAD>>&,
    std::vector<Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, FAD>>&,
    std::vector<Core::LinAlg::Matrix<3, 3, FAD>>&);

template void Discret::Elements::Beam3k::calculate_internal_forces_and_stiff_sk<2>(
    Teuchos::ParameterList&, const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    const std::vector<Core::LinAlg::Matrix<3, 3, FAD>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    std::vector<Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 3, FAD>>&,
    std::vector<Core::LinAlg::Matrix<3, 3, FAD>>&);

template void Discret::Elements::Beam3k::calculate_inertia_forces_and_mass_matrix<2, double>(
    Teuchos::ParameterList&, const std::vector<Core::LinAlg::Matrix<3, 3, double>>&,
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 3, double>>&,
    const std::vector<Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>>&,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    Core::LinAlg::SerialDenseMatrix*);
template void Discret::Elements::Beam3k::calculate_inertia_forces_and_mass_matrix<2, FAD>(
    Teuchos::ParameterList&, const std::vector<Core::LinAlg::Matrix<3, 3, FAD>>&,
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    const std::vector<Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 3, FAD>>&,
    const std::vector<Core::LinAlg::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, FAD>>&,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    Core::LinAlg::SerialDenseMatrix*);

template void Discret::Elements::Beam3k::evaluate_point_neumann_eb<2>(
    Core::LinAlg::SerialDenseVector&, Core::LinAlg::SerialDenseMatrix*,
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const Core::LinAlg::Matrix<6, 1, double>&, int) const;

template void Discret::Elements::Beam3k::evaluate_residual_from_point_neumann_moment<2, double>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&, double,
    int) const;
template void Discret::Elements::Beam3k::evaluate_residual_from_point_neumann_moment<2, FAD>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    const Core::LinAlg::Matrix<3, 1, FAD>&, const Core::LinAlg::Matrix<3, 1, FAD>&, FAD, int) const;

template void
Discret::Elements::Beam3k::evaluate_stiff_matrix_analytic_from_point_neumann_moment<2>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, double, int) const;

template void Discret::Elements::Beam3k::evaluate_line_neumann<2>(Core::LinAlg::SerialDenseVector&,
    Core::LinAlg::SerialDenseMatrix*,
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const Core::LinAlg::Matrix<6, 1, double>&, const std::vector<std::optional<int>>&,
    double) const;

template void Discret::Elements::Beam3k::evaluate_line_neumann_forces<2, double>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const Core::LinAlg::Matrix<6, 1, double>&, const std::vector<std::optional<int>>&,
    double) const;
template void Discret::Elements::Beam3k::evaluate_line_neumann_forces<2, FAD>(
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    const Core::LinAlg::Matrix<6, 1, double>&, const std::vector<std::optional<int>>&,
    double) const;

template void Discret::Elements::Beam3k::evaluate_translational_damping<double, 2, 2, 3>(
    Teuchos::ParameterList&, const Core::LinAlg::Matrix<3 * 2 * 2, 1, double>&,
    const Core::LinAlg::Matrix<3 * 2 * 2, 1, double>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&);
template void Discret::Elements::Beam3k::evaluate_translational_damping<FAD, 2, 2, 3>(
    Teuchos::ParameterList&, const Core::LinAlg::Matrix<3 * 2 * 2, 1, double>&,
    const Core::LinAlg::Matrix<3 * 2 * 2, 1, FAD>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&);

template void
Discret::Elements::Beam3k::evaluate_analytic_stiffmat_contributions_from_translational_damping<2, 2,
    3>(Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 3, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 3, double>&,
    const Core::LinAlg::Matrix<1, 2 * 2, double>&, const Core::LinAlg::Matrix<1, 2 * 2, double>&,
    double, double) const;
template void
Discret::Elements::Beam3k::evaluate_analytic_stiffmat_contributions_from_translational_damping<2, 2,
    3>(Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 3, FAD>&,
    const Core::LinAlg::Matrix<3, 1, FAD>&, const Core::LinAlg::Matrix<3, 1, FAD>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 3, FAD>&,
    const Core::LinAlg::Matrix<1, 2 * 2, double>&, const Core::LinAlg::Matrix<1, 2 * 2, double>&,
    double, double) const;

template void Discret::Elements::Beam3k::evaluate_stochastic_forces<double, 2, 2, 3, 3>(
    const Core::LinAlg::Matrix<3 * 2 * 2, 1, double>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&);
template void Discret::Elements::Beam3k::evaluate_stochastic_forces<FAD, 2, 2, 3, 3>(
    const Core::LinAlg::Matrix<3 * 2 * 2, 1, FAD>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&);

template void
Discret::Elements::Beam3k::evaluate_analytic_stiffmat_contributions_from_stochastic_forces<2, 2, 3>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<1, 2 * 2, double>&, const Core::LinAlg::Matrix<1, 2 * 2, double>&,
    double, double) const;
template void
Discret::Elements::Beam3k::evaluate_analytic_stiffmat_contributions_from_stochastic_forces<2, 2, 3>(
    Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::Matrix<3, 1, FAD>&,
    const Core::LinAlg::Matrix<3, 1, double>&, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<1, 2 * 2, double>&, const Core::LinAlg::Matrix<1, 2 * 2, double>&,
    double, double) const;

template void Discret::Elements::Beam3k::evaluate_rotational_damping<double, 2, 2, 3>(
    const Core::LinAlg::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const std::vector<Core::LinAlg::Matrix<3, 3, double>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&);
template void Discret::Elements::Beam3k::evaluate_rotational_damping<FAD, 2, 2, 3>(
    const Core::LinAlg::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    const std::vector<Core::LinAlg::Matrix<3, 3, FAD>>&, Core::LinAlg::SerialDenseMatrix*,
    Core::LinAlg::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&);

template void
Discret::Elements::Beam3k::evaluate_analytic_stiffmat_contributions_from_rotational_damping<2, 2,
    3>(Core::LinAlg::SerialDenseMatrix&,
    const Core::LinAlg::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const LargeRotations::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS,
        double>&,
    const Core::LinAlg::Matrix<3, 1, double>, const Core::LinAlg::Matrix<3, 1, double>&,
    const Core::LinAlg::Matrix<3, 3, double>&, const Core::LinAlg::Matrix<3, 3, double>&,
    const Core::LinAlg::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 3, double>&,
    const std::vector<Core::LinAlg::Matrix<3, 3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, double>>&,
    const Core::LinAlg::Matrix<3, 1, double>, double, double, double, double) const;

template void Discret::Elements::Beam3k::
    pre_compute_terms_at_cp_for_analytic_stiffmat_contributions_from_rotational_damping<2, 2, 3>(
        Core::LinAlg::Matrix<3, 3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
        const Core::LinAlg::Matrix<1, 3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
        const Core::LinAlg::Matrix<3, 3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
        const Core::LinAlg::Matrix<3, 1, double>&, double,
        const Core::LinAlg::Matrix<4, 1, double>&) const;

template void Discret::Elements::Beam3k::assemble_shapefunctions_l<double, double>(
    Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, double>&,
    Core::LinAlg::Matrix<1, 2 * 6 + BEAM3K_COLLOCATION_POINTS, double>&) const;
template void Discret::Elements::Beam3k::assemble_shapefunctions_l<Sacado::Fad::DFad<double>,
    Sacado::Fad::DFad<double>>(
    Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&,
    Core::LinAlg::Matrix<1, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;
template void Discret::Elements::Beam3k::assemble_shapefunctions_l<double,
    Sacado::Fad::DFad<double>>(Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, double>&,
    Core::LinAlg::Matrix<1, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;

template void Discret::Elements::Beam3k::assemble_shapefunctions_nss<double, double>(
    Core::LinAlg::Matrix<1, 4, double>&, Core::LinAlg::Matrix<1, 4, double>&, double, double,
    Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, double>&) const;
template void Discret::Elements::Beam3k::assemble_shapefunctions_nss<Sacado::Fad::DFad<double>,
    Sacado::Fad::DFad<double>>(Core::LinAlg::Matrix<1, 4, Sacado::Fad::DFad<double>>&,
    Core::LinAlg::Matrix<1, 4, Sacado::Fad::DFad<double>>&, double, double,
    Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;
template void
Discret::Elements::Beam3k::assemble_shapefunctions_nss<double, Sacado::Fad::DFad<double>>(
    Core::LinAlg::Matrix<1, 4, double>&, Core::LinAlg::Matrix<1, 4, double>&, double, double,
    Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;

template void Discret::Elements::Beam3k::assemble_shapefunctions_ns<double, double>(
    Core::LinAlg::Matrix<1, 4, double>&, double,
    Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, double>&) const;
template void Discret::Elements::Beam3k::assemble_shapefunctions_ns<Sacado::Fad::DFad<double>,
    Sacado::Fad::DFad<double>>(Core::LinAlg::Matrix<1, 4, Sacado::Fad::DFad<double>>&, double,
    Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;
template void Discret::Elements::Beam3k::assemble_shapefunctions_ns<double,
    Sacado::Fad::DFad<double>>(Core::LinAlg::Matrix<1, 4, double>&, double,
    Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;

template void Discret::Elements::Beam3k::assemble_shapefunctions_n<double, double>(
    Core::LinAlg::Matrix<1, 4, double>&,
    Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, double>&) const;
template void Discret::Elements::Beam3k::assemble_shapefunctions_n<Sacado::Fad::DFad<double>,
    Sacado::Fad::DFad<double>>(Core::LinAlg::Matrix<1, 4, Sacado::Fad::DFad<double>>&,
    Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;
template void Discret::Elements::Beam3k::assemble_shapefunctions_n<double,
    Sacado::Fad::DFad<double>>(Core::LinAlg::Matrix<1, 4, double>&,
    Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;

template void Discret::Elements::Beam3k::apply_rot_vec_trafo<2, double>(
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&) const;
template void Discret::Elements::Beam3k::apply_rot_vec_trafo<2, Sacado::Fad::DFad<double>>(
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&,
    Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&) const;

template void Discret::Elements::Beam3k::transform_stiff_matrix_multiplicative<2, double>(
    Core::LinAlg::SerialDenseMatrix*,
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&) const;
template void
Discret::Elements::Beam3k::transform_stiff_matrix_multiplicative<2, Sacado::Fad::DFad<double>>(
    Core::LinAlg::SerialDenseMatrix*,
    const Core::LinAlg::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&)
    const;

template void Discret::Elements::Beam3k::straintostress<double>(
    const Core::LinAlg::Matrix<3, 1, double>&, const double&,
    const Core::LinAlg::Matrix<3, 3, double>&, const Core::LinAlg::Matrix<3, 3, double>&,
    Core::LinAlg::Matrix<3, 1, double>&, double&) const;
template void Discret::Elements::Beam3k::straintostress<Sacado::Fad::DFad<double>>(
    const Core::LinAlg::Matrix<3, 1, Sacado::Fad::DFad<double>>&, const Sacado::Fad::DFad<double>&,
    const Core::LinAlg::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    const Core::LinAlg::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    Core::LinAlg::Matrix<3, 1, Sacado::Fad::DFad<double>>&, Sacado::Fad::DFad<double>&) const;

FOUR_C_NAMESPACE_CLOSE

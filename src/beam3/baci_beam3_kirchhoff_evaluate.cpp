/*-----------------------------------------------------------------------------------------------*/
/*! \file

\brief three dimensional nonlinear Kirchhoff beam element based on a C1 curve

\level 2

*/
/*-----------------------------------------------------------------------------------------------*/

// Todo check for obsolete header inclusions
#include "baci_beam3_kirchhoff.hpp"
#include "baci_beam3_spatial_discretization_utils.hpp"
#include "baci_beam3_triad_interpolation_local_rotation_vectors.hpp"
#include "baci_beaminteraction_periodic_boundingbox.hpp"
#include "baci_discretization_fem_general_extract_values.hpp"
#include "baci_discretization_fem_general_largerotations.hpp"
#include "baci_discretization_fem_general_utils_fem_shapefunctions.hpp"
#include "baci_discretization_fem_general_utils_integration.hpp"
#include "baci_global_data.hpp"
#include "baci_inpar_structure.hpp"
#include "baci_lib_discret.hpp"
#include "baci_linalg_fixedsizematrix.hpp"
#include "baci_linalg_utils_sparse_algebra_math.hpp"
#include "baci_structure_new_elements_paramsinterface.hpp"
#include "baci_structure_new_model_evaluator_data.hpp"
#include "baci_utils_exceptions.hpp"
#include "baci_utils_function.hpp"
#include "baci_utils_function_of_time.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------------------*
 |  evaluate the element (public) meier 01/16|
 *----------------------------------------------------------------------------------------------------------*/
int DRT::ELEMENTS::Beam3k::Evaluate(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, std::vector<int>& lm,
    CORE::LINALG::SerialDenseMatrix& elemat1,  // stiffness matrix
    CORE::LINALG::SerialDenseMatrix& elemat2,  // mass matrix
    CORE::LINALG::SerialDenseVector& elevec1,  // internal forces
    CORE::LINALG::SerialDenseVector& elevec2,  // inertia forces
    CORE::LINALG::SerialDenseVector& elevec3)
{
  SetParamsInterfacePtr(params);

  // Set brownian params interface pointer
  if (IsParamsInterface()) SetBrownianDynParamsInterfacePtr();

  // start with "none"
  ELEMENTS::ActionType act = ELEMENTS::none;

  if (IsParamsInterface())
  {
    act = ParamsInterface().GetActionType();
  }
  else
  {
    // get the action required
    std::string action = params.get<std::string>("action", "calc_none");
    if (action == "calc_none")
      dserror("No action supplied");
    else if (action == "calc_struct_linstiff")
      act = ELEMENTS::struct_calc_linstiff;
    else if (action == "calc_struct_nlnstiff")
      act = ELEMENTS::struct_calc_nlnstiff;
    else if (action == "calc_struct_internalforce")
      act = ELEMENTS::struct_calc_internalforce;
    else if (action == "calc_struct_linstiffmass")
      act = ELEMENTS::struct_calc_linstiffmass;
    else if (action == "calc_struct_nlnstiffmass")
      act = ELEMENTS::struct_calc_nlnstiffmass;
    else if (action == "calc_struct_nlnstifflmass")
      act = ELEMENTS::struct_calc_nlnstifflmass;  // with lumped mass matrix
    else if (action == "calc_struct_stress")
      act = ELEMENTS::struct_calc_stress;
    else if (action == "calc_struct_eleload")
      act = ELEMENTS::struct_calc_eleload;
    else if (action == "calc_struct_fsiload")
      act = ELEMENTS::struct_calc_fsiload;
    else if (action == "calc_struct_update_istep")
      act = ELEMENTS::struct_calc_update_istep;
    else if (action == "calc_struct_reset_istep")
      act = ELEMENTS::struct_calc_reset_istep;
    else if (action == "calc_struct_ptcstiff")
      act = ELEMENTS::struct_calc_ptcstiff;
    else if (action == "calc_struct_energy")
      act = ELEMENTS::struct_calc_energy;
    else
      dserror("Unknown type of action for Beam3k");
  }

  switch (act)
  {
    case ELEMENTS::struct_calc_ptcstiff:
    {
      dserror("no ptc implemented for Beam3k element");
      break;
    }

    case ELEMENTS::struct_calc_linstiff:
    {
      // only nonlinear case implemented!
      dserror("linear stiffness matrix called, but not implemented");
      break;
    }

    // nonlinear stiffness and mass matrix are calculated even if only nonlinear stiffness matrix is
    // required
    case ELEMENTS::struct_calc_nlnstiffmass:
    case ELEMENTS::struct_calc_nlnstifflmass:
    case ELEMENTS::struct_calc_nlnstiff:
    case ELEMENTS::struct_calc_internalforce:
    case ELEMENTS::struct_calc_internalinertiaforce:
    {
      // need current global displacement and residual forces and get them from discretization
      // making use of the local-to-global map lm one can extract current displacement and residual
      // values for each degree of freedom get element displacements
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");

      if (disp == Teuchos::null) dserror("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp(lm.size());
      CORE::FE::ExtractMyValues(*disp, mydisp, lm);


      if (act == ELEMENTS::struct_calc_nlnstiffmass)
      {
        CalcInternalAndInertiaForcesAndStiff(
            params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
      }
      else if (act == ELEMENTS::struct_calc_nlnstifflmass)
      {
        dserror("The action calc_struct_nlnstifflmass is not implemented yet!");
      }
      else if (act == ELEMENTS::struct_calc_nlnstiff)
      {
        CalcInternalAndInertiaForcesAndStiff(params, mydisp, &elemat1, nullptr, &elevec1, nullptr);
      }
      else if (act == ELEMENTS::struct_calc_internalforce)
      {
        CalcInternalAndInertiaForcesAndStiff(params, mydisp, nullptr, nullptr, &elevec1, nullptr);
      }
      else if (act == ELEMENTS::struct_calc_internalinertiaforce)
      {
        CalcInternalAndInertiaForcesAndStiff(params, mydisp, nullptr, nullptr, &elevec1, &elevec2);
      }

      // ATTENTION: In order to perform a brief finite difference check of the nonlinear stiffness
      // matrix the code block "FD-CHECK" from the end of this file has to be copied to this place:
      //***************************************************************************************************************
      // Insert code block here!
      //***************************************************************************************************************

      break;
    }

    case ELEMENTS::struct_calc_brownianforce:
    case ELEMENTS::struct_calc_brownianstiff:
    {
      // get element displacements
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
      if (disp == Teuchos::null) dserror("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp(lm.size());
      CORE::FE::ExtractMyValues(*disp, mydisp, lm);

      // get element velocity
      Teuchos::RCP<const Epetra_Vector> vel = discretization.GetState("velocity");
      if (vel == Teuchos::null) dserror("Cannot get state vectors 'velocity'");
      std::vector<double> myvel(lm.size());
      CORE::FE::ExtractMyValues(*vel, myvel, lm);

      if (act == ELEMENTS::struct_calc_brownianforce)
        CalcBrownianForcesAndStiff<2, 2, 3>(params, myvel, mydisp, nullptr, &elevec1);
      else if (act == ELEMENTS::struct_calc_brownianstiff)
        CalcBrownianForcesAndStiff<2, 2, 3>(params, myvel, mydisp, &elemat1, &elevec1);
      else
        dserror("You shouldn't be here.");

      break;
    }

    case ELEMENTS::struct_calc_energy:
    {
      if (elevec1 != Teuchos::null)  // old structural time integration
      {
        if (elevec1.numRows() != 1)
          dserror(
              "energy vector of invalid size %i, expected row dimension 1 (total elastic energy of "
              "element)!",
              elevec1.numRows());
        elevec1(0) = Eint_;
      }
      else if (IsParamsInterface())  // new structural time integration
      {
        ParamsInterface().AddContributionToEnergyType(Eint_, STR::internal_energy);
        ParamsInterface().AddContributionToEnergyType(Ekin_, STR::kinetic_energy);
      }
      break;
    }

    case ELEMENTS::struct_calc_stress:
    {
      break;
    }

    case ELEMENTS::struct_calc_update_istep:
    {
      /* the action calc_struct_update_istep is called in the very end of a time step when the new
       * dynamic equilibrium has finally been found; this is the point where the variable
       * representing the geometric
       * status of the beam at the end of the time step has to be stored */
      Qconvmass_ = Qnewmass_;
      wconvmass_ = wnewmass_;
      aconvmass_ = anewmass_;
      amodconvmass_ = amodnewmass_;
      rttconvmass_ = rttnewmass_;
      rttmodconvmass_ = rttmodnewmass_;
      rtconvmass_ = rtnewmass_;
      rconvmass_ = rnewmass_;

      Qrefconv_ = Qrefnew_;

      break;
    }

    case ELEMENTS::struct_calc_reset_istep:
    {
      /* the action calc_struct_reset_istep is called by the adaptive time step controller; carries
       * out one test step whose purpose is only figuring out a suitable timestep; thus this step
       * may be a very bad one in order to iterated towards the new dynamic equilibrium and the
       * thereby gained new geometric configuration should not be applied as starting point for any
       * further iteration step; as a consequence the thereby generated change of the geometric
       * configuration should be canceled and the configuration should be reset to the value at the
       * beginning of the time step */

      Qnewmass_ = Qconvmass_;
      wnewmass_ = wconvmass_;
      anewmass_ = aconvmass_;
      amodnewmass_ = amodconvmass_;
      rttnewmass_ = rttconvmass_;
      rttmodnewmass_ = rttmodconvmass_;
      rtnewmass_ = rtconvmass_;
      rnewmass_ = rconvmass_;

      Qrefnew_ = Qrefconv_;

      break;
    }

    case ELEMENTS::struct_calc_recover:
    {
      // do nothing here
      break;
    }

    case ELEMENTS::struct_calc_predict:
    {
      // do nothing here
      break;
    }

    // element based PTC scaling
    case ELEMENTS::struct_calc_addjacPTC:
    {
      CalcStiffContributionsPTC(elemat1);
      break;
    }

    default:
    {
      std::cout << "\ncalled element with action type " << ActionType2String(act);
      dserror("This action type is not implemented for Beam3k");
      break;
    }
  }

  return 0;
}

/*------------------------------------------------------------------------------------------------------------*
 | lump mass matrix             (private) meier 01/16|
 *------------------------------------------------------------------------------------------------------------*/
void DRT::ELEMENTS::Beam3k::Lumpmass(CORE::LINALG::SerialDenseMatrix* emass)
{
  dserror("Lumped mass matrix not implemented yet!!!");
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void DRT::ELEMENTS::Beam3k::CalcInternalAndInertiaForcesAndStiff(Teuchos::ParameterList& params,
    std::vector<double>& disp, CORE::LINALG::SerialDenseMatrix* stiffmatrix,
    CORE::LINALG::SerialDenseMatrix* massmatrix, CORE::LINALG::SerialDenseVector* force,
    CORE::LINALG::SerialDenseVector* force_inert)
{
  // number of nodes used for centerline discretization fixed for this element
  const unsigned int nnodecl = 2;
  const unsigned int numdofelement = 2 * 3 * nnodecl + BEAM3K_COLLOCATION_POINTS;


  if (disp.size() != numdofelement)
    dserror(
        "size mismatch: Number of BEAM3K_COLLOCATION_POINTS does not match number of nodes "
        "defined in the input file!");

  // unshift node positions, i.e. manipulate element displacement vector
  // as if there where no periodic boundary conditions
  if (BrownianDynParamsInterfacePtr() != Teuchos::null)
    UnShiftNodePosition(disp, *BrownianDynParamsInterface().GetPeriodicBoundingBox());


  // vector for current nodal DoFs in total Lagrangian style, i.e. displacement + initial values:
  // rotvec_==true:  disp_totlag=[\v{d}_1, \v{theta}_1, t_1, \v{d}_2, \v{theta}_2, t_2, \alpha_3]
  // rotvec_==false: disp_totlag=[\v{d}_1, \v{t}_1, \alpha_1, \v{d}_2, \v{t}_2, \alpha_2, \alpha_3]
  // The Number of collocation points can take on the values 2, 3 and 4. 3 and 4 are interior nodes.
  // This leads e.g. in the case rotvec_==true to the following ordering:
  // if BEAM3K_COLLOCATION_POINTS = 2: [ \v{d}_1 \v{theta}_1 t_1 \v{d}_2 \v{theta}_1 t_2]
  // if BEAM3K_COLLOCATION_POINTS = 3: [ \v{d}_1 \v{theta}_1 t_1 \v{d}_2 \v{theta}_1 t_2 \alpha_3]
  // if BEAM3K_COLLOCATION_POINTS = 4: [ \v{d}_1 \v{theta}_1 t_1 \v{d}_2 \v{theta}_1 t_2 \alpha_3
  // \alpha_4]
  CORE::LINALG::Matrix<numdofelement, 1, double> disp_totlag(true);

  // Set current positions and orientations at all nodes:
  UpdateDispTotlag<nnodecl, double>(disp, disp_totlag);


  // analytic linearization of internal forces
  if (not useFAD_)
  {
    if (rotvec_ == true)
    {
      dserror(
          "Beam3k: analytic linearization of internal forces only implemented for "
          "tangent-based variant so far! activate FAD!");
    }

    // internal force vector
    CORE::LINALG::Matrix<numdofelement, 1, double> internal_force(true);

    if (force != nullptr)
    {
      // set view on CORE::LINALG::SerialDenseVector to avoid copying of data
      internal_force.SetView(&((*force)(0)));
    }

    // vector containing locally assembled nodal positions and tangents required for centerline:
    // r(s)=N(s)*disp_totlag_centerline, with disp_totlag_centerline=[\v{d}_1, \v{t}_1, 0, \v{d}_2,
    // \v{t}_2, 0, 0]
    CORE::LINALG::Matrix<numdofelement, 1, double> disp_totlag_centerline(true);

    // material triads at collocation points
    std::vector<CORE::LINALG::Matrix<3, 3, double>> triad_mat_cp(
        BEAM3K_COLLOCATION_POINTS, CORE::LINALG::Matrix<3, 3, double>(true));

    UpdateNodalVariables<nnodecl, double>(
        disp_totlag, disp_totlag_centerline, triad_mat_cp, Qrefnew_);

    // Store nodal tangents in class variable
    for (unsigned int i = 0; i < 3; ++i)
    {
      T_[0](i) = disp_totlag_centerline(3 + i);
      T_[1](i) = disp_totlag_centerline(10 + i);
    }


    // pre-computed variables that are later re-used in computation of inertia terms

    // interpolated spin vector variation at Gauss points required for inertia forces:
    // v_theta = v_thetaperp + v_thetapar
    std::vector<CORE::LINALG::Matrix<numdofelement, 3, double>> v_theta_gp;

    // multiplicative rotation increment at GP
    // required for analytic linearization and re-used for analytic mass matrix
    std::vector<CORE::LINALG::Matrix<3, numdofelement, double>> lin_theta_gp;

    // Interpolated material triad at Gauss points
    std::vector<CORE::LINALG::Matrix<3, 3, double>> triad_mat_gp;


    if (weakkirchhoff_)
    {
      CalculateInternalForcesAndStiffWK<nnodecl, double>(params, disp_totlag_centerline,
          triad_mat_cp, stiffmatrix, internal_force, v_theta_gp, lin_theta_gp, triad_mat_gp);
    }
    else
    {
      dserror(
          "Beam3k: analytic linearization only implemented for variant WK so far! "
          "activate FAD!");
    }

    // *************** INERTIA *******************************************
    if (massmatrix != nullptr or force_inert != nullptr)
    {
      // construct as a view on CORE::LINALG::SerialDenseVector to avoid copying of data
      CORE::LINALG::Matrix<numdofelement, 1, double> inertia_force(*force_inert, true);

      CalculateInertiaForcesAndMassMatrix<nnodecl, double>(params, triad_mat_gp,
          disp_totlag_centerline, v_theta_gp, lin_theta_gp, inertia_force, massmatrix);
    }
  }
  // automatic linearization of internal forces via FAD
  else
  {
    // internal force vector
    CORE::LINALG::Matrix<numdofelement, 1, FAD> internal_force_FAD(true);

    // copy pre-computed disp_totlag to a FAD matrix
    CORE::LINALG::Matrix<numdofelement, 1, FAD> disp_totlag_FAD(true);

    for (unsigned int idof = 0; idof < numdofelement; ++idof)
      disp_totlag_FAD(idof) = disp_totlag(idof);

    // vector containing locally assembled nodal positions and tangents required for centerline:
    // r(s)=N(s)*disp_totlag_centerline, with disp_totlag_centerline=[\v{d}_1, \v{t}_1, 0, \v{d}_2,
    // \v{t}_2, 0, 0]
    CORE::LINALG::Matrix<numdofelement, 1, FAD> disp_totlag_centerline_FAD(true);

    // material triads at collocation points
    std::vector<CORE::LINALG::Matrix<3, 3, FAD>> triad_mat_cp_FAD(
        BEAM3K_COLLOCATION_POINTS, CORE::LINALG::Matrix<3, 3, FAD>(true));

    // Next, we have to set variables for FAD
    SetAutomaticDifferentiationVariables<nnodecl>(disp_totlag_FAD);

    UpdateNodalVariables<nnodecl, FAD>(
        disp_totlag_FAD, disp_totlag_centerline_FAD, triad_mat_cp_FAD, Qrefnew_);

    // Store nodal tangents in class variable
    for (unsigned int i = 0; i < 3; ++i)
    {
      T_[0](i) = disp_totlag_centerline_FAD(3 + i).val();
      T_[1](i) = disp_totlag_centerline_FAD(10 + i).val();
    }


    // pre-computed variables that are later re-used in computation of inertia terms

    // interpolated spin vector variation at Gauss points required for inertia forces:
    // v_theta = v_thetaperp + v_thetapar
    std::vector<CORE::LINALG::Matrix<numdofelement, 3, FAD>> v_theta_gp_FAD;

    // multiplicative rotation increment at GP
    // required for analytic linearization and re-used for analytic mass matrix
    std::vector<CORE::LINALG::Matrix<3, numdofelement, FAD>> lin_theta_gp_FAD;

    // Interpolated material triad at Gauss points
    std::vector<CORE::LINALG::Matrix<3, 3, FAD>> triad_mat_gp_FAD;


    if (weakkirchhoff_)
    {
      CalculateInternalForcesAndStiffWK<nnodecl, FAD>(params, disp_totlag_centerline_FAD,
          triad_mat_cp_FAD, nullptr, internal_force_FAD, v_theta_gp_FAD, lin_theta_gp_FAD,
          triad_mat_gp_FAD);
    }
    else
    {
      CalculateInternalForcesAndStiffSK<nnodecl>(params, disp_totlag_centerline_FAD,
          triad_mat_cp_FAD, nullptr, internal_force_FAD, v_theta_gp_FAD, triad_mat_gp_FAD);
    }

    if (rotvec_ == true)
    {
      ApplyRotVecTrafo<nnodecl, FAD>(disp_totlag_centerline_FAD, internal_force_FAD);
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

      if (rotvec_ == true) TransformStiffMatrixMultipl<nnodecl, double>(stiffmatrix, disp_totlag);
    }


    // *************** INERTIA *******************************************
    if (massmatrix != nullptr or force_inert != nullptr)
    {
      CORE::LINALG::Matrix<numdofelement, 1, FAD> inertia_force_FAD(true);

      CalculateInertiaForcesAndMassMatrix<nnodecl, FAD>(params, triad_mat_gp_FAD,
          disp_totlag_centerline_FAD, v_theta_gp_FAD, lin_theta_gp_FAD, inertia_force_FAD, nullptr);

      if (rotvec_ == true)
      {
        ApplyRotVecTrafo<nnodecl, FAD>(disp_totlag_centerline_FAD, inertia_force_FAD);
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

        if (rotvec_ == true) TransformStiffMatrixMultipl<nnodecl, double>(massmatrix, disp_totlag);
      }
    }
  }



  if (massmatrix != nullptr)
  {
    double dt = 1000.0;
    double beta = -1.0;
    double alpha_f = -1.0;
    double alpha_m = -1.0;

    if (this->IsParamsInterface())
    {
      dt = ParamsInterface().GetDeltaTime();
      beta = ParamsInterface().GetBeamParamsInterfacePtr()->GetBeta();
      alpha_f = ParamsInterface().GetBeamParamsInterfacePtr()->GetAlphaf();
      alpha_m = ParamsInterface().GetBeamParamsInterfacePtr()->GetAlpham();
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
void DRT::ELEMENTS::Beam3k::CalculateInternalForcesAndStiffWK(Teuchos::ParameterList& params,
    const CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
        disp_totlag_centerline,
    const std::vector<CORE::LINALG::Matrix<3, 3, T>>& triad_mat_cp,
    CORE::LINALG::SerialDenseMatrix* stiffmatrix,
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& internal_force,
    std::vector<CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, T>>& v_theta_gp,
    std::vector<CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, T>>& lin_theta_gp,
    std::vector<CORE::LINALG::Matrix<3, 3, T>>& triad_mat_gp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Beam3k::CalculateInternalForcesAndStiffWK");

  if (BEAM3K_COLLOCATION_POINTS != 2 and BEAM3K_COLLOCATION_POINTS != 3 and
      BEAM3K_COLLOCATION_POINTS != 4)
    dserror("Only the values 2,3 and 4 are valid for BEAM3K_COLLOCATION_POINTS!!!");

  // number of nodes fixed for this element
  const unsigned int nnode = 2;

  const unsigned int numdofelement = 2 * 3 * nnodecl + BEAM3K_COLLOCATION_POINTS;

  // internal force vector
  CORE::LINALG::Matrix<numdofelement, 1, T> f_int_aux(true);


  // CP values of strains and their variations needed for interpolation
  std::vector<T> epsilon_cp(BEAM3K_COLLOCATION_POINTS);  // axial tension
  std::vector<CORE::LINALG::Matrix<numdofelement, 1, T>> v_epsilon_cp(BEAM3K_COLLOCATION_POINTS);
  std::vector<CORE::LINALG::Matrix<numdofelement, 3, T>> v_thetaperp_cp(BEAM3K_COLLOCATION_POINTS);
  std::vector<CORE::LINALG::Matrix<numdofelement, 3, T>> v_thetapar_cp(BEAM3K_COLLOCATION_POINTS);

  // linearization of strain variations at CPs
  std::vector<CORE::LINALG::Matrix<numdofelement, numdofelement, T>> lin_v_epsilon_cp(
      BEAM3K_COLLOCATION_POINTS);

  // Get integration points for exact integration
  CORE::FE::IntegrationPoints1D gausspoints = CORE::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);



  // re-interpolated values of strains and their variations evaluated at a specific Gauss point
  T epsilon_bar;
  CORE::LINALG::Matrix<numdofelement, 1, T> v_epsilon_bar(true);
  CORE::LINALG::Matrix<numdofelement, 3, T> v_thetaperp_s_bar(true);
  CORE::LINALG::Matrix<numdofelement, 3, T> v_thetapar_s_bar(true);
  CORE::LINALG::Matrix<numdofelement, 3, T> v_theta_s_bar(
      true);  //=v_thetaperp_s_bar+v_thetapar_s_bar


  // interpolated spin vector variation required for inertia forces: v_theta=v_thetaperp+v_thetapar
  v_theta_gp.resize(gausspoints.nquad);

  // CP values of increments lin_theta_cp = lin_theta_perp_cp + lin_theta_par_cp
  // (required for analytic linearization)
  std::vector<CORE::LINALG::Matrix<3, numdofelement, T>> lin_theta_cp(
      BEAM3K_COLLOCATION_POINTS, CORE::LINALG::Matrix<3, numdofelement, T>(true));

  lin_theta_gp.resize(gausspoints.nquad);

  // Further material and spatial strains and forces to be evaluated at a specific Gauss point
  CORE::LINALG::Matrix<3, 1, T> K;      // material curvature
  CORE::LINALG::Matrix<3, 1, T> Omega;  // material deformation measure Omega:=K-K0
  CORE::LINALG::Matrix<3, 1, T> m;      // spatial moment stress resultant
  CORE::LINALG::Matrix<3, 1, T> M;      // material moment stress resultant
  T f_par;                              // material=spatial axial force component


  // Additional kinematic quantities at a specific Gauss point
  CORE::LINALG::Matrix<3, 1, T> r_s;  // vector to store r'
  T abs_r_s;                          // ||r'||


  // Interpolated material triad and angles evaluated at Gauss point
  triad_mat_gp.resize(gausspoints.nquad);
  CORE::LINALG::Matrix<3, 1, T> theta;    // interpolated angle theta
  CORE::LINALG::Matrix<3, 1, T> theta_s;  // derivative of theta with respect to arc-length s


  // matrices storing the assembled shape functions and s-derivatives
  CORE::LINALG::Matrix<3, numdofelement, T> N_s;
  CORE::LINALG::Matrix<1, numdofelement, T> L;

  // matrices storing individual shape functions, its xi-derivatives and s-derivatives
  CORE::LINALG::Matrix<1, 2 * nnode, double> N_i_xi;
  //  CORE::LINALG::Matrix<1,2*nnode,double> N_i_s;
  CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i;
  CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i_xi;
  CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i_s;

  // material constitutive matrices
  CORE::LINALG::Matrix<3, 3, T> Cn, Cm;
  GetConstitutiveMatrices(Cn, Cm);


  // parameter coordinate
  double xi_cp = 0.0;
  // position index where CP quantities have to be stored
  unsigned int ind = 0;


  // create object of triad interpolation scheme
  Teuchos::RCP<LARGEROTATIONS::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS, T>>
      triad_interpolation_scheme_ptr = Teuchos::rcp(
          new LARGEROTATIONS::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS,
              T>());

  // reset scheme with nodal triads
  triad_interpolation_scheme_ptr->Reset(triad_mat_cp);


  //********begin: evaluate quantities at collocation points********************************
  for (unsigned int inode = 0; inode < BEAM3K_COLLOCATION_POINTS; inode++)
  {
    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    xi_cp = (double)inode / (double)(BEAM3K_COLLOCATION_POINTS - 1) * 2.0 - 1.0;

    // get value of interpolating function of theta (lagrange polynomials) at xi
    L_i.Clear();
    CORE::FE::shape_function_1D(L_i, xi_cp, Shape());

    N_i_xi.Clear();
    CORE::FE::shape_function_hermite_1D_deriv1(N_i_xi, xi_cp, length_, CORE::FE::CellType::line2);

    // Determine storage position for the node node
    ind = CORE::LARGEROTATIONS::NumberingTrafo(inode + 1, BEAM3K_COLLOCATION_POINTS);

    N_s.Clear();
    AssembleShapefunctionsNs(N_i_xi, jacobi_cp_[ind], N_s);

    r_s.Clear();
    // Calculation of r' at xi
    r_s.Multiply(N_s, disp_totlag_centerline);

    // calculate epsilon at collocation point
    abs_r_s = CORE::FADUTILS::Norm<T>(r_s);
    epsilon_cp[ind] = abs_r_s - 1.0;

    AssembleShapefunctionsL(L_i, L);


    v_epsilon_cp[ind].Clear();
    v_epsilon_cp[ind].MultiplyTN(N_s, r_s);
    v_epsilon_cp[ind].Scale(1.0 / abs_r_s);

    Calc_v_thetaperp<nnodecl>(v_thetaperp_cp[ind], N_s, r_s, abs_r_s);

    Calc_v_thetapartheta<nnodecl>(v_thetapar_cp[ind], L, r_s, abs_r_s);


    if (stiffmatrix != nullptr)
    {
      PreComputeTermsAtCPForStiffmatContributionsAnalyticWK<nnodecl>(
          lin_theta_cp[ind], lin_v_epsilon_cp[ind], L, N_s, r_s, abs_r_s, Qrefconv_[ind]);
    }
  }
  //********end: evaluate quantities at collocation points********************************


  // Clear energy in the beginning
  Eint_ = 0.0;

  // re-assure correct size of strain and stress resultant class variables
  axial_strain_GP_.resize(gausspoints.nquad);
  twist_GP_.resize(gausspoints.nquad);
  curvature_2_GP_.resize(gausspoints.nquad);
  curvature_3_GP_.resize(gausspoints.nquad);

  axial_force_GP_.resize(gausspoints.nquad);
  torque_GP_.resize(gausspoints.nquad);
  bending_moment_2_GP_.resize(gausspoints.nquad);
  bending_moment_3_GP_.resize(gausspoints.nquad);


  //******begin: gauss integration for internal force vector and stiffness matrix*********
  for (int numgp = 0; numgp < gausspoints.nquad; numgp++)
  {
    // Get location and weight of GP in parameter space
    const double xi_gp = gausspoints.qxg[numgp][0];
    const double wgt = gausspoints.qwgt[numgp];

    // Evaluate shape functions
    L_i.Clear();
    L_i_xi.Clear();
    L_i_s.Clear();
    CORE::FE::shape_function_1D(L_i, xi_gp, Shape());
    CORE::FE::shape_function_1D_deriv1(L_i_xi, xi_gp, Shape());
    L_i_s.Update(1.0 / jacobi_[numgp], L_i_xi);


    // Calculate collocation point interpolations ("v"-vectors and epsilon)
    v_epsilon_bar.Clear();
    v_thetaperp_s_bar.Clear();
    v_thetapar_s_bar.Clear();
    v_theta_s_bar.Clear();
    epsilon_bar = 0.0;
    theta.Clear();
    theta_s.Clear();


    triad_interpolation_scheme_ptr->GetInterpolatedLocalRotationVector(theta, L_i);

    triad_interpolation_scheme_ptr->GetInterpolatedLocalRotationVectorDerivative(
        theta_s, L_i_xi, jacobi_[numgp]);


    // re-interpolation of collocation point values
    for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; ++node)
    {
      v_epsilon_bar.Update(L_i(node), v_epsilon_cp[node], 1.0);

      v_thetaperp_s_bar.Update(L_i_s(node), v_thetaperp_cp[node], 1.0);

      v_thetapar_s_bar.Update(L_i_s(node), v_thetapar_cp[node], 1.0);

      epsilon_bar += L_i(node) * epsilon_cp[node];
    }

    v_theta_s_bar.Update(1.0, v_thetaperp_s_bar, 1.0);
    v_theta_s_bar.Update(1.0, v_thetapar_s_bar, 1.0);


    // "v"-matrix which is required for inertia forces is already calculated here
    // Todo find a nicer and more independent solution here
    v_theta_gp[numgp].Clear();
    for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; ++node)
    {
      v_theta_gp[numgp].Update(L_i(node), v_thetaperp_cp[node], 1.0);
      v_theta_gp[numgp].Update(L_i(node), v_thetapar_cp[node], 1.0);
    }


    // compute material strain K
    K.Clear();
    Omega.Clear();

    computestrain(theta, theta_s, K);

    for (unsigned int idim = 0; idim < 3; ++idim)
    {
      Omega(idim) = K(idim) - (K0_[numgp])(idim);
    }

    // compute material stress resultants
    M.Clear();
    f_par = 0.0;


    straintostress(Omega, epsilon_bar, Cn, Cm, M, f_par);


    // compute material triad at gp
    triad_mat_gp[numgp].Clear();

    triad_interpolation_scheme_ptr->GetInterpolatedTriad(triad_mat_gp[numgp], theta);


    // pushforward of stress resultants
    m.Clear();
    m.Multiply(triad_mat_gp[numgp], M);


    // residual contribution from moments
    f_int_aux.Clear();
    f_int_aux.Multiply(v_theta_s_bar, m);
    f_int_aux.Scale(wgt * jacobi_[numgp]);

    internal_force.Update(1.0, f_int_aux, 1.0);

    // residual contribution from axial force
    f_int_aux.Clear();
    f_int_aux.Update(1.0, v_epsilon_bar);
    f_int_aux.Scale(wgt * jacobi_[numgp] * f_par);

    internal_force.Update(1.0, f_int_aux, 1.0);


    if (stiffmatrix != nullptr)
    {
      CalculateStiffmatContributionsAnalyticWK<nnodecl>(*stiffmatrix, disp_totlag_centerline,
          *triad_interpolation_scheme_ptr, v_theta_s_bar, lin_theta_cp, lin_theta_gp[numgp],
          lin_v_epsilon_cp, v_epsilon_bar, f_par, m, Cn(0, 0), Cm, theta, theta_s,
          triad_mat_gp[numgp], xi_gp, jacobi_[numgp], wgt);
    }

    // Calculate internal energy and store it in class variable
    Eint_ += 0.5 * CORE::FADUTILS::CastToDouble(epsilon_bar) * CORE::FADUTILS::CastToDouble(f_par) *
             wgt * jacobi_[numgp];

    for (unsigned int idim = 0; idim < 3; ++idim)
    {
      Eint_ += 0.5 * CORE::FADUTILS::CastToDouble(Omega(idim)) *
               CORE::FADUTILS::CastToDouble(M(idim)) * wgt * jacobi_[numgp];
    }

    // store material strain and stress resultant values in class variables
    axial_strain_GP_[numgp] = CORE::FADUTILS::CastToDouble(epsilon_bar);
    twist_GP_[numgp] = CORE::FADUTILS::CastToDouble(Omega(0));
    curvature_2_GP_[numgp] = CORE::FADUTILS::CastToDouble(Omega(1));
    curvature_3_GP_[numgp] = CORE::FADUTILS::CastToDouble(Omega(2));

    axial_force_GP_[numgp] = CORE::FADUTILS::CastToDouble(f_par);
    torque_GP_[numgp] = CORE::FADUTILS::CastToDouble(M(0));
    bending_moment_2_GP_[numgp] = CORE::FADUTILS::CastToDouble(M(1));
    bending_moment_3_GP_[numgp] = CORE::FADUTILS::CastToDouble(M(2));
  }
  //******end: gauss integration for internal force vector and stiffness matrix*********
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void DRT::ELEMENTS::Beam3k::CalculateStiffmatContributionsAnalyticWK(
    CORE::LINALG::SerialDenseMatrix& stiffmatrix,
    const CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>&
        disp_totlag_centerline,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS, double>&
        triad_intpol,
    const CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, double>& v_theta_s_bar,
    const std::vector<CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>>&
        lin_theta_cp,
    CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta_bar,
    const std::vector<CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
        6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>>& lin_v_epsilon_cp,
    const CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>& v_epsilon_bar,
    double axial_force_bar, const CORE::LINALG::Matrix<3, 1, double>& moment_resultant,
    double axial_rigidity,
    const CORE::LINALG::Matrix<3, 3, double>& constitutive_matrix_moment_material,
    const CORE::LINALG::Matrix<3, 1, double>& theta_gp,
    const CORE::LINALG::Matrix<3, 1, double>& theta_s_gp,
    const CORE::LINALG::Matrix<3, 3, double>& triad_mat_gp, double xi_gp, double jacobifac_gp,
    double GPwgt) const
{
  // spatial dimension
  const unsigned int ndim = 3;
  // number of values used for centerline interpolation (Hermite: value + derivative)
  const unsigned int vpernode = 2;
  // size of Dof vector of this element
  const unsigned int numdofelement = ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS;

  // create a fixed size matrix as view on the CORE::LINALG::SerialDenseMatrix to avoid copying
  CORE::LINALG::Matrix<numdofelement, numdofelement, double> stiffmatrix_fixedsize(
      stiffmatrix, true);


  // matrices storing the assembled shape functions and s-derivatives
  CORE::LINALG::Matrix<ndim, numdofelement, double> N_s, N_ss;
  CORE::LINALG::Matrix<1, numdofelement, double> L, L_s;

  // matrices storing individual shape functions, its xi-derivatives and s-derivatives
  CORE::LINALG::Matrix<1, vpernode * nnodecl, double> N_i_xi, N_i_s, N_i_xixi;
  CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i, L_i_xi, L_i_s;

  // r' vector and its norm
  CORE::LINALG::Matrix<3, 1, double> r_s_cp(true), r_ss_cp(true);
  double abs_r_s_cp = 0.0;

  // first base vector and s-derivative
  CORE::LINALG::Matrix<3, 1, double> g_1_cp(true), g_1_s_cp(true);


  // re-interpolated lin_theta
  CORE::LINALG::Matrix<ndim, numdofelement, double> lin_theta_s_bar(true);

  // linearization of re-interpolated strain variations
  std::vector<CORE::LINALG::Matrix<numdofelement, numdofelement, double>> lin_v_thetaperp_moment_cp(
      BEAM3K_COLLOCATION_POINTS),
      lin_v_thetapar_moment_cp(BEAM3K_COLLOCATION_POINTS);


  CORE::LINALG::Matrix<numdofelement, numdofelement, double> lin_v_thetaperp_s_bar_moment(true),
      lin_v_thetapar_s_bar_moment(true);


  // linearization of re-interpolated strain variations
  CORE::LINALG::Matrix<numdofelement, numdofelement, double> lin_v_epsilon_bar(true);


  CORE::LINALG::Matrix<3, 3, double> spinmatrix_of_moment_resultant(true);
  CORE::LARGEROTATIONS::computespin<double>(spinmatrix_of_moment_resultant, moment_resultant);

  // push forward constitutive matrix according to Jelenic 1999, paragraph following to (2.22) on
  // page 148
  CORE::LINALG::Matrix<3, 3, double> constitutive_matrix_moment_spatial(true);

  CORE::LINALG::Matrix<3, 3, double> temp(true);
  temp.Multiply(triad_mat_gp, constitutive_matrix_moment_material);
  constitutive_matrix_moment_spatial.MultiplyNT(temp, triad_mat_gp);


  // linearization of stress resultant (moment)
  CORE::LINALG::Matrix<3, numdofelement, double> lin_moment_resultant(true);


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
    ind = CORE::LARGEROTATIONS::NumberingTrafo(icp + 1, BEAM3K_COLLOCATION_POINTS);

    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    xi_cp = (double)icp / (double)(BEAM3K_COLLOCATION_POINTS - 1) * 2.0 - 1.0;


    // get all required shape function values
    L_i.Clear();
    CORE::FE::shape_function_1D(L_i, xi_cp, Shape());

    L.Clear();
    AssembleShapefunctionsL(L_i, L);

    N_i_xi.Clear();
    CORE::FE::shape_function_hermite_1D_deriv1(N_i_xi, xi_cp, length_, CORE::FE::CellType::line2);

    N_s.Clear();
    AssembleShapefunctionsNs(N_i_xi, jacobi_cp_[ind], N_s);


    // Calculation of r' and g_1
    r_s_cp.Clear();
    r_s_cp.Multiply(N_s, disp_totlag_centerline);

    abs_r_s_cp = r_s_cp.Norm2();  // Todo think about computing and storing inverse value here


    g_1_cp.Clear();
    g_1_cp.Update(std::pow(abs_r_s_cp, -1.0), r_s_cp);


    Calc_lin_v_thetaperp_moment<nnodecl>(
        lin_v_thetaperp_moment_cp[ind], N_s, g_1_cp, abs_r_s_cp, spinmatrix_of_moment_resultant);

    Calc_lin_v_thetapar_moment<nnodecl>(
        lin_v_thetapar_moment_cp[ind], L, N_s, g_1_cp, abs_r_s_cp, moment_resultant);
  }


  /***********************************************************************************************/
  // re-interpolation of quantities at xi based on CP values

  L_i.Clear();
  CORE::FE::shape_function_1D(L_i, xi_gp, Shape());

  L_i_xi.Clear();
  CORE::FE::shape_function_1D_deriv1(L_i_xi, xi_gp, Shape());

  L_i_s.Clear();
  L_i_s.Update(std::pow(jacobifac_gp, -1.0), L_i_xi, 0.0);

  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    lin_v_epsilon_bar.Update(L_i(icp), lin_v_epsilon_cp[icp], 1.0);

    lin_v_thetaperp_s_bar_moment.Update(L_i_s(icp), lin_v_thetaperp_moment_cp[icp], 1.0);
    lin_v_thetapar_s_bar_moment.Update(L_i_s(icp), lin_v_thetapar_moment_cp[icp], 1.0);
  }

  // compute Itilde(_s) matrices required for re-interpolation of CP values of lin_theta
  std::vector<CORE::LINALG::Matrix<3, 3, double>> Itilde(BEAM3K_COLLOCATION_POINTS);
  std::vector<CORE::LINALG::Matrix<3, 3, double>> Itilde_s(BEAM3K_COLLOCATION_POINTS);

  triad_intpol.GetNodalGeneralizedRotationInterpolationMatrices(Itilde, theta_gp, L_i);

  triad_intpol.GetNodalGeneralizedRotationInterpolationMatricesDerivative(
      Itilde_s, theta_gp, theta_s_gp, L_i, L_i_s);


  CORE::LINALG::Matrix<3, numdofelement, double> auxmatrix(true);

  lin_theta_bar.Clear();
  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    auxmatrix.Clear();

    auxmatrix.Multiply(Itilde[icp], lin_theta_cp[icp]);

    lin_theta_bar.Update(1.0, auxmatrix, 1.0);

    auxmatrix.Clear();

    auxmatrix.Multiply(Itilde_s[icp], lin_theta_cp[icp]);

    lin_theta_s_bar.Update(1.0, auxmatrix, 1.0);
  }


  Calc_lin_moment_resultant<nnodecl>(lin_moment_resultant, lin_theta_bar, lin_theta_s_bar,
      spinmatrix_of_moment_resultant, constitutive_matrix_moment_spatial);

  /***********************************************************************************************/
  // finally put everything together

  // constant pre-factor
  const double jacobifac_GPwgt = jacobifac_gp * GPwgt;

  CORE::LINALG::Matrix<numdofelement, numdofelement, double> auxmatrix2(true);


  // linearization of the residual contributions from moments
  stiffmatrix_fixedsize.Update(jacobifac_GPwgt, lin_v_thetaperp_s_bar_moment, 1.0);

  stiffmatrix_fixedsize.Update(jacobifac_GPwgt, lin_v_thetapar_s_bar_moment, 1.0);

  auxmatrix2.Clear();
  auxmatrix2.Multiply(v_theta_s_bar, lin_moment_resultant);
  stiffmatrix_fixedsize.Update(jacobifac_GPwgt, auxmatrix2, 1.0);


  // linearization of the residual contributions from axial force
  auxmatrix2.Clear();
  for (unsigned int idof = 0; idof < numdofelement; ++idof)
    for (unsigned int jdof = 0; jdof < numdofelement; ++jdof)
      auxmatrix2(idof, jdof) = v_epsilon_bar(idof) * v_epsilon_bar(jdof);

  stiffmatrix_fixedsize.Update(axial_rigidity * jacobifac_GPwgt, auxmatrix2, 1.0);

  stiffmatrix_fixedsize.Update(axial_force_bar * jacobifac_GPwgt, lin_v_epsilon_bar, 1.0);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void DRT::ELEMENTS::Beam3k::PreComputeTermsAtCPForStiffmatContributionsAnalyticWK(
    CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta,
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
        6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_v_epsilon,
    const CORE::LINALG::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& L,
    const CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
    const CORE::LINALG::Matrix<3, 1, double>& r_s, double abs_r_s,
    const CORE::LINALG::Matrix<4, 1, double>& Qref_conv) const
{
  // spatial dimension
  const unsigned int ndim = 3;
  // number of values used for centerline interpolation (Hermite: value + derivative)
  const unsigned int vpernode = 2;
  // size of Dof vector of this element
  const unsigned int numdofelement = ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS;


  CORE::LINALG::Matrix<ndim, 1, double> g_1(true);
  g_1.Update(std::pow(abs_r_s, -1.0), r_s);

  CORE::LINALG::Matrix<ndim, 1, double> g_1_bar(true);

  CORE::LINALG::Matrix<3, 3, double> triad_ref_conv_cp(true);
  CORE::LARGEROTATIONS::quaterniontotriad(Qref_conv, triad_ref_conv_cp);

  g_1_bar.Clear();
  for (unsigned int idim = 0; idim < ndim; ++idim) g_1_bar(idim) = triad_ref_conv_cp(idim, 0);


  // CP values of strain increments
  CORE::LINALG::Matrix<ndim, numdofelement, double> lin_theta_perp(true), lin_theta_par(true);

  Calc_lin_thetapar<nnodecl>(lin_theta_par, L, N_s, g_1, g_1_bar, abs_r_s);

  Calc_lin_thetaperp<nnodecl>(lin_theta_perp, N_s, r_s, abs_r_s);

  // lin_theta
  lin_theta.Clear();
  lin_theta.Update(1.0, lin_theta_par, 1.0, lin_theta_perp);

  Calc_lin_v_epsilon<nnodecl>(lin_v_epsilon, N_s, g_1, abs_r_s);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void DRT::ELEMENTS::Beam3k::CalculateInternalForcesAndStiffSK(Teuchos::ParameterList& params,
    const CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>&
        disp_totlag_centerline,
    const std::vector<CORE::LINALG::Matrix<3, 3, FAD>>& triad_mat_cp,
    CORE::LINALG::SerialDenseMatrix* stiffmatrix,
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>& internal_force,
    std::vector<CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD>>& v_theta_gp,
    std::vector<CORE::LINALG::Matrix<3, 3, FAD>>& triad_mat_gp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Beam3k::CalculateInternalForcesAndStiffSK");

  if (BEAM3K_COLLOCATION_POINTS != 2 and BEAM3K_COLLOCATION_POINTS != 3 and
      BEAM3K_COLLOCATION_POINTS != 4)
  {
    dserror("Only the values 2,3 and 4 are valid for BEAM3K_COLLOCATION_POINTS!!!");
  }

  // internal force vector
  CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> f_int_aux(true);

  // CP values of strains and their variations needed for interpolation
  std::vector<FAD> epsilon_cp(
      BEAM3K_COLLOCATION_POINTS);  // axial tension epsilon=|r_s|-1 at collocation points
  std::vector<CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>> v_epsilon_cp(
      BEAM3K_COLLOCATION_POINTS);

  // Get integration points for exact integration
  CORE::FE::IntegrationPoints1D gausspoints = CORE::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);

  // interpolated values of strains and their variations evaluated at Gauss points
  FAD epsilon;
  CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> v_epsilon(true);
  CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD> v_thetaperp_s(true);
  CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD> v_thetapartheta_s(true);
  CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD> v_theta_s(
      true);  //=v_thetaperp_s+v_thetapartheta_s(+v_thetapard_s)

  // interpolated spin vector variation required for inertia forces:
  // v_theta=v_thetaperp+v_thetapartheta(+v_thetapard)
  v_theta_gp.resize(gausspoints.nquad);

  // Further material and spatial strains and forces to be evaluated at the Gauss points
  CORE::LINALG::Matrix<3, 1, FAD> K(true);      // material curvature
  CORE::LINALG::Matrix<3, 1, FAD> Omega(true);  // material deformation measure Omega:=K-K0
  CORE::LINALG::Matrix<3, 1, FAD> m(true);      // spatial moment stress resultant
  CORE::LINALG::Matrix<3, 1, FAD> M(true);      // material moment stress resultant
  FAD f_par = 0.0;                              // material=spatial axial force component

  // Triads at collocation points
  std::vector<FAD> phi_cp(BEAM3K_COLLOCATION_POINTS, 0.0);  // relative angle at collocation points

  // Interpolated material triad and angles evaluated at Gauss point
  triad_mat_gp.resize(gausspoints.nquad);
  FAD phi = 0.0;    // interpolated relative angle phi
  FAD phi_s = 0.0;  // derivative of interpolated relative angle phi with respect to arc-length s

  // matrices holding the assembled shape functions and s-derivatives
  CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD> N_s(true);
  CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD> N_ss(true);
  CORE::LINALG::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD> L(true);
  CORE::LINALG::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD> L_s(true);

  // Matrices for individual shape functions and xi-derivatives
  CORE::LINALG::Matrix<1, 2 * nnodecl, FAD> N_i_xi(true);
  CORE::LINALG::Matrix<1, 2 * nnodecl, FAD> N_i_xixi(true);
  CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, FAD> L_i(true);
  CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, FAD> L_i_xi(true);

  // Matrices for individual s-derivatives
  CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, FAD> L_i_s(true);

  // Additional kinematic quantities
  CORE::LINALG::Matrix<3, 1, FAD> r_s(true);         // Matrix to store r'
  CORE::LINALG::Matrix<3, 1, FAD> r_ss(true);        // Matrix to store r''
  CORE::LINALG::Matrix<3, 1, FAD> g1(true);          // g1:=r'/||r'||
  CORE::LINALG::Matrix<3, 1, FAD> g1_s(true);        // g1'
  CORE::LINALG::Matrix<3, 1, FAD> ttilde(true);      //\tilde{t}:=g1/||r'||=r'/||r'||^2
  CORE::LINALG::Matrix<3, 1, FAD> ttilde_s(true);    //\tilde{t}'
  CORE::LINALG::Matrix<3, 1, FAD> kappacl(true);     // centerline (cl) curvature vector
  FAD abs_r_s = 0.0;                                 // ||r'||
  FAD rsTrss = 0.0;                                  // r'^Tr''
  CORE::LINALG::Matrix<3, 3, FAD> auxmatrix1(true);  // auxilliary matrix
  CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD> auxmatrix2(
      true);  // auxilliary matrix

#ifdef CONSISTENTSPINSK
  CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD> v_thetapard_s(true);
  std::vector<CORE::LINALG::Matrix<3, 1, FAD>> g1_cp(
      BEAM3K_COLLOCATION_POINTS, CORE::LINALG::Matrix<3, 1, FAD>(true));
  std::vector<CORE::LINALG::Matrix<3, 1, FAD>> ttilde_cp(
      BEAM3K_COLLOCATION_POINTS, CORE::LINALG::Matrix<3, 1, FAD>(true));
  std::vector<CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD>> N_s_cp(
      BEAM3K_COLLOCATION_POINTS,
      CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD>(true));
  std::vector<CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>> v1_cp(
      BEAM3K_COLLOCATION_POINTS,
      CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>(true));
  CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> v1(true);
  CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> v1_s(true);
#endif

  // MISC
  double xi = 0.0;       // parameter coordinated
  unsigned int ind = 0;  // position index where CP quantities have to be stored

  // material constitutive matrices
  CORE::LINALG::Matrix<3, 3, FAD> Cn, Cm;
  GetConstitutiveMatrices(Cn, Cm);


  //********begin: evaluate quantities at collocation points********************************
  for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; node++)
  {
    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    xi = (double)node / (BEAM3K_COLLOCATION_POINTS - 1) * 2 - 1.0;

    // get value of interpolating function of theta (lagrange polynomials) at xi
    N_i_xi.Clear();
    CORE::FE::shape_function_hermite_1D_deriv1(N_i_xi, xi, length_, CORE::FE::CellType::line2);

    // Determine storage position for the node node
    ind = CORE::LARGEROTATIONS::NumberingTrafo(node + 1, BEAM3K_COLLOCATION_POINTS);

    N_s.Clear();
    AssembleShapefunctionsNs(N_i_xi, jacobi_cp_[ind], N_s);
    r_s.Clear();
    // Calculation of r' at xi
    r_s.Multiply(N_s, disp_totlag_centerline);

    // calculate epsilon at collocation point
    abs_r_s = CORE::FADUTILS::Norm<FAD>(r_s);
    epsilon_cp[ind] = abs_r_s - 1.0;

    v_epsilon_cp[ind].Clear();
    v_epsilon_cp[ind].MultiplyTN(N_s, r_s);
    v_epsilon_cp[ind].Scale(1.0 / abs_r_s);

#ifdef CONSISTENTSPINSK
    N_s_cp[ind].Update(1.0, N_s, 0.0);
    g1_cp[ind].Update(1.0 / abs_r_s, r_s, 0.0);
    ttilde_cp[ind].Update(1.0 / (abs_r_s * abs_r_s), r_s, 0.0);
#endif

  }  // for (int node=0; node<BEAM3K_COLLOCATION_POINTS; node++)

  // calculate angle at cp (this has to be done in a SEPARATE loop as follows)
  for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; node++)
  {
    CORE::LINALG::Matrix<3, 3, FAD> Lambdabarref(true);
    CORE::LINALG::Matrix<3, 1, FAD> tangentref(true);
    CORE::LINALG::Matrix<3, 1, FAD> phivec(true);
    for (int i = 0; i < 3; i++)
    {
      tangentref(i) = triad_mat_cp[node](i, 0);
    }
    CORE::LARGEROTATIONS::CalculateSRTriads<FAD>(
        tangentref, triad_mat_cp[REFERENCE_NODE], Lambdabarref);
    CORE::LARGEROTATIONS::triadtoangleleft(phivec, Lambdabarref, triad_mat_cp[node]);
    phi_cp[node] = 0.0;
    for (unsigned int i = 0; i < 3; i++)
    {
      phi_cp[node] += tangentref(i) * phivec(i);
    }

#ifdef CONSISTENTSPINSK
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> auxmatrix3(true);
    ComputeTripleProduct<6 * nnodecl + BEAM3K_COLLOCATION_POINTS>(
        N_s_cp[node], g1_cp[REFERENCE_NODE], ttilde_cp[node], auxmatrix3);
    v1_cp[node].Update(1.0, auxmatrix3, 0.0);
    auxmatrix3.Clear();
    ComputeTripleProduct<6 * nnodecl + BEAM3K_COLLOCATION_POINTS>(
        N_s_cp[REFERENCE_NODE], g1_cp[node], ttilde_cp[REFERENCE_NODE], auxmatrix3);
    v1_cp[node].Update(-1.0, auxmatrix3, 1.0);
    CORE::LINALG::Matrix<1, 1, FAD> auxscalar(true);
    auxscalar.MultiplyTN(g1_cp[node], g1_cp[REFERENCE_NODE]);
    v1_cp[node].Scale(1.0 / (1.0 + auxscalar(0, 0)));
#endif
  }
  //********end: evaluate quantities at collocation points********************************

  // Clear energy in the beginning
  Eint_ = 0.0;

  // re-assure correct size of strain and stress resultant class variables
  axial_strain_GP_.resize(gausspoints.nquad);
  twist_GP_.resize(gausspoints.nquad);
  curvature_2_GP_.resize(gausspoints.nquad);
  curvature_3_GP_.resize(gausspoints.nquad);

  axial_force_GP_.resize(gausspoints.nquad);
  torque_GP_.resize(gausspoints.nquad);
  bending_moment_2_GP_.resize(gausspoints.nquad);
  bending_moment_3_GP_.resize(gausspoints.nquad);


  //******begin: gauss integration for internal force vector and stiffness matrix*********
  for (int numgp = 0; numgp < gausspoints.nquad; numgp++)
  {
    // Get location and weight of GP in parameter space
    const double xi = gausspoints.qxg[numgp][0];
    const double wgt = gausspoints.qwgt[numgp];

    // Evaluate and assemble shape functions
    L_i.Clear();
    L_i_xi.Clear();
    L_i_s.Clear();
    L.Clear();
    L_s.Clear();
    N_i_xi.Clear();
    N_i_xixi.Clear();
    N_s.Clear();
    N_ss.Clear();
    CORE::FE::shape_function_1D(L_i, xi, Shape());
    CORE::FE::shape_function_1D_deriv1(L_i_xi, xi, Shape());
    L_i_s.Update(1.0 / jacobi_[numgp], L_i_xi, 0.0);
    AssembleShapefunctionsL(L_i, L);
    // The assemble routine is identical for L and L_s
    AssembleShapefunctionsL(L_i_s, L_s);
    CORE::FE::shape_function_hermite_1D_deriv1(N_i_xi, xi, length_, CORE::FE::CellType::line2);
    AssembleShapefunctionsNs(N_i_xi, jacobi_[numgp], N_s);
    CORE::FE::shape_function_hermite_1D_deriv2(N_i_xixi, xi, length_, CORE::FE::CellType::line2);
    AssembleShapefunctionsNss(N_i_xi, N_i_xixi, jacobi_[numgp], jacobi2_[numgp], N_ss);

    // Calculate collocation piont interpolations
    v_epsilon.Clear();
    epsilon = 0.0;
    phi = 0.0;
    phi_s = 0.0;
    for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; node++)
    {
      // calculate interpolated axial tension and variation
      v_epsilon.Update(L_i(node), v_epsilon_cp[node], 1.0);
      epsilon += L_i(node) * epsilon_cp[node];
      // calculate interpolated relative angle
      phi += L_i(node) * phi_cp[node];
      phi_s += L_i_s(node) * phi_cp[node];
    }

    // Calculation of r' and r'' at xi
    r_s.Clear();
    r_s.Multiply(N_s, disp_totlag_centerline);
    r_ss.Clear();
    r_ss.Multiply(N_ss, disp_totlag_centerline);

    //*****************************************************************************************************************************
    //************************Begin: Determine "v"-vectors representing the discrete strain
    // variations*****************************
    //*****************************************************************************************************************************
    // Auxilliary quantities
    abs_r_s = 0.0;
    rsTrss = 0.0;
    abs_r_s = CORE::FADUTILS::Norm<FAD>(r_s);
    for (unsigned int i = 0; i < 3; i++)
    {
      rsTrss += r_s(i) * r_ss(i);
    }
    g1.Clear();
    g1_s.Clear();
    g1.Update(1.0 / abs_r_s, r_s, 0.0);
    g1_s.Update(1.0 / abs_r_s, r_ss, 0.0);
    g1_s.Update(-rsTrss / (abs_r_s * abs_r_s * abs_r_s), r_s, 1.0);
    ttilde.Clear();
    ttilde_s.Clear();
    ttilde.Update(1.0 / (abs_r_s * abs_r_s), r_s, 0.0);
    ttilde_s.Update(1.0 / (abs_r_s * abs_r_s), r_ss, 0.0);
    ttilde_s.Update(-2 * rsTrss / (abs_r_s * abs_r_s * abs_r_s * abs_r_s), r_s, 1.0);

    //************** I) Compute v_theta_s=v_thetaperp_s+v_thetapartheta_s(+v_thetapard_s)
    //*****************************************
    // I a) Compute v_thetapartheta_s
    v_thetapartheta_s.Clear();
    for (unsigned int row = 0; row < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; row++)
    {
      for (unsigned int column = 0; column < 3; column++)
      {
        v_thetapartheta_s(row, column) += L_s(0, row) * g1(column, 0) + L(0, row) * g1_s(column, 0);
      }
    }

    // I b) Compute v_thetaperp_s
    v_thetaperp_s.Clear();
    auxmatrix1.Clear();
    auxmatrix2.Clear();
    CORE::LARGEROTATIONS::computespin(auxmatrix1, ttilde);
    auxmatrix2.MultiplyTN(N_ss, auxmatrix1);
    v_thetaperp_s.Update(-1.0, auxmatrix2, 0.0);
    auxmatrix1.Clear();
    auxmatrix2.Clear();
    CORE::LARGEROTATIONS::computespin(auxmatrix1, ttilde_s);
    auxmatrix2.MultiplyTN(N_s, auxmatrix1);
    v_thetaperp_s.Update(-1.0, auxmatrix2, 1.0);

    // I c) Calculate sum v_theta_s=v_thetaperp_s+v_thetapartheta_s
    v_theta_s.Clear();
    v_theta_s.Update(1.0, v_thetaperp_s, 1.0);
    v_theta_s.Update(1.0, v_thetapartheta_s, 1.0);

    //************** II) Compute v_theta=v_thetaperp_+v_thetapartheta_(+v_thetapard_)  which is
    // required for inertia forces ********
    // II a) v_thetapartheta contribution
    v_theta_gp[numgp].Clear();
    for (unsigned int row = 0; row < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; row++)
    {
      for (unsigned int column = 0; column < 3; column++)
      {
        v_theta_gp[numgp](row, column) += L(0, row) * g1(column, 0);
      }
    }

    // II b) Compute v_thetaperp contribution
    auxmatrix1.Clear();
    auxmatrix2.Clear();
    CORE::LARGEROTATIONS::computespin(auxmatrix1, ttilde);
    auxmatrix2.MultiplyTN(N_s, auxmatrix1);
    v_theta_gp[numgp].Update(-1.0, auxmatrix2, 1.0);


// Compute contributions stemming from CONSISTENTSPINSK
#ifdef CONSISTENTSPINSK
    //************** to I) Compute v_thetapard_s of
    // v_theta_s=v_thetaperp_s+v_thetapartheta_s(+v_thetapard_s) *******************
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> auxmatrix3(true);
    CORE::LINALG::Matrix<1, 1, FAD> auxscalar1(true);

    // Calculate v1:
    v1.Clear();
    auxmatrix3.Clear();
    ComputeTripleProduct<6 * nnodecl + BEAM3K_COLLOCATION_POINTS>(
        N_s, g1_cp[REFERENCE_NODE], ttilde, auxmatrix3);
    v1.Update(1.0, auxmatrix3, 0.0);
    auxmatrix3.Clear();
    ComputeTripleProduct<6 * nnodecl + BEAM3K_COLLOCATION_POINTS>(
        N_s_cp[REFERENCE_NODE], g1, ttilde_cp[REFERENCE_NODE], auxmatrix3);
    v1.Update(-1.0, auxmatrix3, 1.0);
    auxscalar1.Clear();
    auxscalar1.MultiplyTN(g1, g1_cp[REFERENCE_NODE]);
    v1.Scale(1.0 / (1.0 + auxscalar1(0, 0)));

    // Calculate v1_s:
    v1_s.Clear();
    auxmatrix3.Clear();
    ComputeTripleProduct<6 * nnodecl + BEAM3K_COLLOCATION_POINTS>(
        N_s, g1_cp[REFERENCE_NODE], ttilde_s, auxmatrix3);
    v1_s.Update(1.0, auxmatrix3, 0.0);
    auxmatrix3.Clear();
    ComputeTripleProduct<6 * nnodecl + BEAM3K_COLLOCATION_POINTS>(
        N_ss, g1_cp[REFERENCE_NODE], ttilde, auxmatrix3);
    v1_s.Update(1.0, auxmatrix3, 1.0);
    auxmatrix3.Clear();
    ComputeTripleProduct<6 * nnodecl + BEAM3K_COLLOCATION_POINTS>(
        N_s_cp[REFERENCE_NODE], g1_s, ttilde_cp[REFERENCE_NODE], auxmatrix3);
    v1_s.Update(-1.0, auxmatrix3, 1.0);
    auxscalar1.Clear();
    auxscalar1.MultiplyTN(g1_s, g1_cp[REFERENCE_NODE]);
    v1_s.Update(-auxscalar1(0, 0), v1, 1.0);
    auxscalar1.Clear();
    auxscalar1.MultiplyTN(g1, g1_cp[REFERENCE_NODE]);
    v1_s.Scale(1.0 / (1.0 + auxscalar1(0, 0)));

    // Calculate vec1 and vec2
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> vec1(true);
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> vec2(true);
    for (int node = 0; node < BEAM3K_COLLOCATION_POINTS; node++)
    {
      vec1.Update(L_i_s(node), v1_cp[node], 1.0);
      vec2.Update(L_i(node), v1_cp[node], 1.0);
    }
    vec1.Update(-1.0, v1_s, 1.0);
    vec2.Update(-1.0, v1, 1.0);

    // Compute v_thetapard_s
    v_thetapard_s.Clear();
    for (int i = 0; i < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        v_thetapard_s(i, j) += vec1(i) * g1(j) + vec2(i) * g1_s(j);
      }
    }
    // I d) Add v_thetapard_s contribution according to v_theta_s+=v_thetapard_s
    v_theta_s.Update(1.0, v_thetapard_s, 1.0);

    // to II)  Compute v_thetapard_ of v_theta=v_thetaperp_+v_thetapartheta_(+v_thetapard_)  which
    // is required for inertia forces******* II c) v_thetapard_ contribution

    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> vec3(true);
    for (int node = 0; node < BEAM3K_COLLOCATION_POINTS; node++)
    {
      vec3.Update(L_i(node), v1_cp[node], 1.0);
    }
    vec3.Update(-1.0, v1, 1.0);
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
    triad_mat_gp[numgp].Clear();
    ComputeTriadSK(phi, r_s, triad_mat_cp[REFERENCE_NODE], triad_mat_gp[numgp]);
    kappacl.Clear();
    Calculate_clcurvature(r_s, r_ss, kappacl);

    // compute material strain K at Gauss point
    K.Clear();
    Omega.Clear();
    computestrainSK(phi_s, kappacl, triad_mat_cp[REFERENCE_NODE], triad_mat_gp[numgp], K);
    for (unsigned int i = 0; i < 3; i++)
    {
      Omega(i) = K(i) - K0_[numgp](i);
    }

    // compute material stress resultants at Gauss point
    M.Clear();
    f_par = 0.0;
    straintostress(Omega, epsilon, Cn, Cm, M, f_par);

    // Calculate internal energy and store it in class variable
    Eint_ += 0.5 * epsilon.val() * f_par.val() * wgt * jacobi_[numgp];
    for (unsigned int i = 0; i < 3; i++)
    {
      Eint_ += 0.5 * Omega(i).val() * M(i).val() * wgt * jacobi_[numgp];
    }

    // pushforward of stress resultants
    m.Clear();
    m.Multiply(triad_mat_gp[numgp], M);

    // residual contribution from moments
    f_int_aux.Clear();
    f_int_aux.Multiply(v_theta_s, m);
    f_int_aux.Scale(wgt * jacobi_[numgp]);
    internal_force.Update(1.0, f_int_aux, 1.0);

    // residual contribution from axial forces
    f_int_aux.Clear();
    f_int_aux.Update(1.0, v_epsilon, 0.0);
    f_int_aux.Scale(wgt * jacobi_[numgp] * f_par);
    internal_force.Update(1.0, f_int_aux, 1.0);


    // store material strain and stress resultant values in class variables
    axial_strain_GP_[numgp] = CORE::FADUTILS::CastToDouble(epsilon);
    twist_GP_[numgp] = CORE::FADUTILS::CastToDouble(Omega(0));
    curvature_2_GP_[numgp] = CORE::FADUTILS::CastToDouble(Omega(1));
    curvature_3_GP_[numgp] = CORE::FADUTILS::CastToDouble(Omega(2));

    axial_force_GP_[numgp] = CORE::FADUTILS::CastToDouble(f_par);
    torque_GP_[numgp] = CORE::FADUTILS::CastToDouble(M(0));
    bending_moment_2_GP_[numgp] = CORE::FADUTILS::CastToDouble(M(1));
    bending_moment_3_GP_[numgp] = CORE::FADUTILS::CastToDouble(M(2));
  }
  //******end: gauss integration for internal force vector and stiffness matrix*********
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void DRT::ELEMENTS::Beam3k::CalculateInertiaForcesAndMassMatrix(Teuchos::ParameterList& params,
    const std::vector<CORE::LINALG::Matrix<3, 3, T>>& triad_mat_gp,
    const CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
        disp_totlag_centerline,
    const std::vector<CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, T>>&
        v_theta_gp,
    const std::vector<CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, T>>&
        lin_theta_gp,
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& f_inert,
    CORE::LINALG::SerialDenseMatrix* massmatrix)
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
   * Since BACI does all displacement updates in an additive manner, the global vector of
   * rotational displacements has no physical meaning and, consequently the global velocity
   * and acceleration vectors resulting from the BACI time integration schemes have no
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

  if (this->IsParamsInterface())
  {
    dt = ParamsInterface().GetDeltaTime();
    beta = ParamsInterface().GetBeamParamsInterfacePtr()->GetBeta();
    gamma = ParamsInterface().GetBeamParamsInterfacePtr()->GetGamma();
    alpha_f = ParamsInterface().GetBeamParamsInterfacePtr()->GetAlphaf();
    alpha_m = ParamsInterface().GetBeamParamsInterfacePtr()->GetAlpham();
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
  CORE::LINALG::Matrix<3, 3, T> Jp(true);

  GetTranslationalAndRotationalMassInertiaTensor(mass_inertia_translational, Jp);


  CORE::LINALG::Matrix<3, numdofelement, T> N(true);
  CORE::LINALG::Matrix<1, 2 * nnodecl, double> N_i(true);
  CORE::LINALG::Matrix<3, 1, T> rnewmass(true);
  CORE::LINALG::Matrix<3, 3, T> triad_mat_old(true);

  // auxiliary internal force vector
  CORE::LINALG::Matrix<numdofelement, 1, T> f_inert_aux(true);
  //  CORE::LINALG::Matrix<numdofelement,3,T> v_thetaperp(true);  // Todo unused?
  //  CORE::LINALG::Matrix<numdofelement,3,T> v_thetapar(true);

  // Get integration points for exact integration
  CORE::FE::IntegrationPoints1D gausspoints = CORE::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);

  Ekin_ = 0.0;

  // loop through Gauss points
  for (int numgp = 0; numgp < gausspoints.nquad; ++numgp)
  {
    // get location and weight of GP in parameter space
    const double xi = gausspoints.qxg[numgp][0];
    const double wgt = gausspoints.qwgt[numgp];

    // get shape function values and assemble them
    N_i.Clear();
    CORE::FE::shape_function_hermite_1D(N_i, xi, length_, CORE::FE::CellType::line2);

    N.Clear();
    AssembleShapefunctionsN(N_i, N);


    // calculation of centroid position at this gp in current state
    rnewmass.Clear();
    rnewmass.Multiply(N, disp_totlag_centerline);


    // get quaternion in converged state at gp and compute corresponding triad
    triad_mat_old.Clear();
    CORE::LINALG::Matrix<4, 1, T> Qconv(true);
    for (unsigned int i = 0; i < 4; ++i) Qconv(i) = (Qconvmass_[numgp])(i);

    CORE::LARGEROTATIONS::quaterniontotriad(Qconv, triad_mat_old);

    // compute quaternion of relative rotation from converged to current state
    CORE::LINALG::Matrix<3, 3, T> deltatriad(true);
    deltatriad.MultiplyNT(triad_mat_gp[numgp], triad_mat_old);
    CORE::LINALG::Matrix<4, 1, T> deltaQ(true);
    CORE::LARGEROTATIONS::triadtoquaternion(deltatriad, deltaQ);
    CORE::LINALG::Matrix<3, 1, T> deltatheta(true);
    CORE::LARGEROTATIONS::quaterniontoangle(deltaQ, deltatheta);

    // compute material counterparts of spatial vectors
    CORE::LINALG::Matrix<3, 1, T> deltaTHETA(true);
    CORE::LINALG::Matrix<3, 1, T> Wconvmass(true);
    CORE::LINALG::Matrix<3, 1, T> Aconvmass(true);
    CORE::LINALG::Matrix<3, 1, T> Amodconvmass(true);

    deltaTHETA.MultiplyTN(triad_mat_gp[numgp], deltatheta);

    CORE::LINALG::Matrix<3, 1, T> auxvector(true);
    for (unsigned int i = 0; i < 3; ++i) auxvector(i) = (wconvmass_[numgp])(i);
    Wconvmass.MultiplyTN(triad_mat_old, auxvector);

    for (unsigned int i = 0; i < 3; ++i) auxvector(i) = (aconvmass_[numgp])(i);
    Aconvmass.MultiplyTN(triad_mat_old, auxvector);

    for (unsigned int i = 0; i < 3; ++i) auxvector(i) = (amodconvmass_[numgp])(i);
    Amodconvmass.MultiplyTN(triad_mat_old, auxvector);

    CORE::LINALG::Matrix<3, 1, T> deltar(true);
    for (unsigned int i = 0; i < 3; ++i) deltar(i) = rnewmass(i) - rconvmass_[numgp](i);

    CORE::LINALG::Matrix<3, 1, T> Anewmass(true);
    CORE::LINALG::Matrix<3, 1, T> Wnewmass(true);
    CORE::LINALG::Matrix<3, 1, T> Amodnewmass(true);
    CORE::LINALG::Matrix<3, 1, T> rttnewmass(true);
    CORE::LINALG::Matrix<3, 1, T> rtnewmass(true);
    CORE::LINALG::Matrix<3, 1, T> rttmodnewmass(true);

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
    CORE::LINALG::Matrix<3, 3, T> SWnewmass(true);
    CORE::LARGEROTATIONS::computespin(SWnewmass, Wnewmass);

    CORE::LINALG::Matrix<3, 1, T> Jp_Wnewmass(true);
    CORE::LINALG::Matrix<3, 1, T> auxvector1(true);
    CORE::LINALG::Matrix<3, 1, T> Pi_t(true);
    Jp_Wnewmass.Multiply(Jp, Wnewmass);
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        auxvector1(i) += SWnewmass(i, j) * Jp_Wnewmass(j) + Jp(i, j) * Anewmass(j);

    Pi_t.Multiply(triad_mat_gp[numgp], auxvector1);

    CORE::LINALG::Matrix<3, 1, T> L_t(true);
    L_t.Update(mass_inertia_translational, rttnewmass, 1.0);

    // residual contribution from inertia moment
    f_inert_aux.Clear();
    f_inert_aux.Multiply(v_theta_gp[numgp], Pi_t);
    f_inert_aux.Scale(wgt * jacobi_[numgp]);
    f_inert.Update(1.0, f_inert_aux, 1.0);

    // residual contribution from inertia force
    f_inert_aux.Clear();
    f_inert_aux.MultiplyTN(N, L_t);
    f_inert_aux.Scale(wgt * jacobi_[numgp]);
    f_inert.Update(1.0, f_inert_aux, 1.0);


    // compute analytic mass matrix if required
    if (massmatrix != nullptr)
    {
      // temporal derivative of angular momentum equals negative inertia moment
      CORE::LINALG::Matrix<3, 1, T> moment_rho(Pi_t);
      moment_rho.Scale(-1.0);

      if (weakkirchhoff_)
      {
        CalculateMassMatrixContributionsAnalyticWK<nnodecl>(*massmatrix, disp_totlag_centerline,
            v_theta_gp[numgp], lin_theta_gp[numgp], moment_rho, deltatheta, Wnewmass,
            triad_mat_gp[numgp], triad_mat_old, N, mass_inertia_translational, Jp,
            lin_prefactor_acc, lin_prefactor_vel, xi, jacobi_[numgp], wgt);
      }
      else
      {
        dserror(
            "you tried to calculate the analytic contributions to mass matrix which is not "
            "implemented yet in case of SK!");
      }
    }


    // Calculation of kinetic energy
    CORE::LINALG::Matrix<1, 1, T> ekinrot(true);
    CORE::LINALG::Matrix<1, 1, T> ekintrans(true);
    ekinrot.MultiplyTN(Wnewmass, Jp_Wnewmass);
    ekintrans.MultiplyTN(rtnewmass, rtnewmass);
    Ekin_ += 0.5 *
             (CORE::FADUTILS::CastToDouble(ekinrot(0, 0)) +
                 mass_inertia_translational * CORE::FADUTILS::CastToDouble(ekintrans(0, 0))) *
             wgt * jacobi_[numgp];

    //**********begin: update class variables needed for storage**************
    CORE::LINALG::Matrix<3, 1, T> wnewmass(true);
    CORE::LINALG::Matrix<3, 1, T> anewmass(true);
    CORE::LINALG::Matrix<3, 1, T> amodnewmass(true);

    wnewmass.Multiply(triad_mat_gp[numgp], Wnewmass);
    anewmass.Multiply(triad_mat_gp[numgp], Anewmass);
    amodnewmass.Multiply(triad_mat_gp[numgp], Amodnewmass);


    // compute quaterion of current material triad at gp
    CORE::LINALG::Matrix<4, 1, T> Qnewmass(true);
    CORE::LARGEROTATIONS::triadtoquaternion(triad_mat_gp[numgp], Qnewmass);

    for (unsigned int i = 0; i < 4; ++i)
    {
      (Qnewmass_[numgp])(i) = CORE::FADUTILS::CastToDouble(Qnewmass(i));
    }

    for (unsigned int i = 0; i < 3; ++i)
    {
      (wnewmass_[numgp])(i) = CORE::FADUTILS::CastToDouble(wnewmass(i));
      (anewmass_[numgp])(i) = CORE::FADUTILS::CastToDouble(anewmass(i));
      (amodnewmass_[numgp])(i) = CORE::FADUTILS::CastToDouble(amodnewmass(i));

      (rnewmass_[numgp])(i) = CORE::FADUTILS::CastToDouble(rnewmass(i));
      (rtnewmass_[numgp])(i) = CORE::FADUTILS::CastToDouble(rtnewmass(i));
      (rttnewmass_[numgp])(i) = CORE::FADUTILS::CastToDouble(rttnewmass(i));
      (rttmodnewmass_[numgp])(i) = CORE::FADUTILS::CastToDouble(rttmodnewmass(i));
    }
    //**********end: update class variables needed for storage**************
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void DRT::ELEMENTS::Beam3k::CalculateMassMatrixContributionsAnalyticWK(
    CORE::LINALG::SerialDenseMatrix& massmatrix,
    const CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>&
        disp_totlag_centerline,
    const CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, double>& v_theta_bar,
    const CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta_bar,
    const CORE::LINALG::Matrix<3, 1, double>& moment_rho,
    const CORE::LINALG::Matrix<3, 1, double>& deltatheta,
    const CORE::LINALG::Matrix<3, 1, double>& angular_velocity_material,
    const CORE::LINALG::Matrix<3, 3, double>& triad_mat,
    const CORE::LINALG::Matrix<3, 3, double>& triad_mat_conv,
    const CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N,
    double mass_inertia_translational,
    const CORE::LINALG::Matrix<3, 3, double>& tensor_mass_moment_of_inertia,
    double lin_prefactor_acc, double lin_prefactor_vel, double xi_gp, double jacobifac_gp,
    double GPwgt) const
{
  // spatial dimension
  const unsigned int ndim = 3;
  // number of values used for centerline interpolation (Hermite: value + derivative)
  const unsigned int vpernode = 2;
  // size of Dof vector of this element
  const unsigned int numdofelement = ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS;

  // create a fixed size matrix as view on the CORE::LINALG::SerialDenseMatrix to avoid copying
  CORE::LINALG::Matrix<numdofelement, numdofelement, double> massmatrix_fixedsize(massmatrix, true);


  // matrices storing the assembled shape functions or s-derivative
  CORE::LINALG::Matrix<ndim, numdofelement, double> N_s;
  CORE::LINALG::Matrix<1, numdofelement, double> L;

  // matrices storing individual shape functions, its xi-derivatives and s-derivatives
  CORE::LINALG::Matrix<1, vpernode * nnodecl, double> N_i_xi;
  CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i;

  // r' vector and its norm
  CORE::LINALG::Matrix<3, 1, double> r_s_cp(true);
  double abs_r_s_cp = 0.0;

  // first base vector
  CORE::LINALG::Matrix<3, 1, double> g_1_cp(true);


  // linearization of re-interpolated strain variations
  std::vector<CORE::LINALG::Matrix<numdofelement, numdofelement, double>> lin_v_thetaperp_moment_cp(
      BEAM3K_COLLOCATION_POINTS),
      lin_v_thetapar_moment_cp(BEAM3K_COLLOCATION_POINTS);

  CORE::LINALG::Matrix<numdofelement, numdofelement, double> lin_v_thetaperp_bar_moment(true),
      lin_v_thetapar_bar_moment(true);


  CORE::LINALG::Matrix<3, 3, double> spinmatrix_of_moment_rho(true);
  CORE::LARGEROTATIONS::computespin<double>(spinmatrix_of_moment_rho, moment_rho);


  // linearization of stress resultant (moment)
  CORE::LINALG::Matrix<3, numdofelement, double> lin_moment_rho(true);


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
    ind = CORE::LARGEROTATIONS::NumberingTrafo(icp + 1, BEAM3K_COLLOCATION_POINTS);

    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    xi_cp = (double)icp / (double)(BEAM3K_COLLOCATION_POINTS - 1) * 2.0 - 1.0;


    // get all required shape function values
    L_i.Clear();
    CORE::FE::shape_function_1D(L_i, xi_cp, Shape());

    L.Clear();
    AssembleShapefunctionsL(L_i, L);

    N_i_xi.Clear();
    CORE::FE::shape_function_hermite_1D_deriv1(N_i_xi, xi_cp, length_, CORE::FE::CellType::line2);

    N_s.Clear();
    AssembleShapefunctionsNs(N_i_xi, jacobi_cp_[ind], N_s);


    // Calculation of r' and g_1
    r_s_cp.Clear();
    r_s_cp.Multiply(N_s, disp_totlag_centerline);

    abs_r_s_cp = r_s_cp.Norm2();  // Todo think about computing and storing inverse value here

    g_1_cp.Clear();
    g_1_cp.Update(std::pow(abs_r_s_cp, -1.0), r_s_cp);


    Calc_lin_v_thetaperp_moment<nnodecl>(
        lin_v_thetaperp_moment_cp[ind], N_s, g_1_cp, abs_r_s_cp, spinmatrix_of_moment_rho);

    Calc_lin_v_thetapar_moment<nnodecl>(
        lin_v_thetapar_moment_cp[ind], L, N_s, g_1_cp, abs_r_s_cp, moment_rho);
  }


  /***********************************************************************************************/
  // re-interpolation of quantities at xi based on CP values

  L_i.Clear();
  CORE::FE::shape_function_1D(L_i, xi_gp, Shape());

  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    lin_v_thetaperp_bar_moment.Update(L_i(icp), lin_v_thetaperp_moment_cp[icp], 1.0);
    lin_v_thetapar_bar_moment.Update(L_i(icp), lin_v_thetapar_moment_cp[icp], 1.0);
  }
  /***********************************************************************************************/

  Calc_lin_moment_inertia<nnodecl>(lin_moment_rho, triad_mat, triad_mat_conv, deltatheta,
      angular_velocity_material, lin_theta_bar, spinmatrix_of_moment_rho,
      tensor_mass_moment_of_inertia, lin_prefactor_acc, lin_prefactor_vel);

  /***********************************************************************************************/
  // finally put everything together

  // constant pre-factor
  const double jacobifac_GPwgt = jacobifac_gp * GPwgt;

  CORE::LINALG::Matrix<numdofelement, numdofelement, double> auxmatrix(true);

  // linearization of residual from inertia force
  auxmatrix.MultiplyTN(N, N);
  auxmatrix.Scale(mass_inertia_translational * lin_prefactor_acc);

  massmatrix_fixedsize.Update(jacobifac_GPwgt, auxmatrix, 1.0);


  // linearization of residual from inertia moment
  massmatrix_fixedsize.Update(-1.0 * jacobifac_GPwgt, lin_v_thetaperp_bar_moment, 1.0);

  massmatrix_fixedsize.Update(-1.0 * jacobifac_GPwgt, lin_v_thetapar_bar_moment, 1.0);

  auxmatrix.Clear();
  auxmatrix.Multiply(v_theta_bar, lin_moment_rho);

  massmatrix_fixedsize.Update(-1.0 * jacobifac_GPwgt, auxmatrix, 1.0);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
int DRT::ELEMENTS::Beam3k::EvaluateNeumann(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Condition& condition, std::vector<int>& lm,
    CORE::LINALG::SerialDenseVector& elevec1, CORE::LINALG::SerialDenseMatrix* elemat1)
{
  SetParamsInterfacePtr(params);

  /* As long as only endpoint forces and moments as well as distributed forces
   * (i.e. no distributed moments) are considered, the method EvaluateNeumann is identical
   * for the WK and the SK case. */

  if (BEAM3K_COLLOCATION_POINTS != 2 and BEAM3K_COLLOCATION_POINTS != 3 and
      BEAM3K_COLLOCATION_POINTS != 4)
  {
    dserror("Only the values 2,3 and 4 are valid for BEAM3K_COLLOCATION_POINTS!!!");
  }

  // number of nodes used for centerline interpolation
  const unsigned int nnodecl = 2;

  // get element displacements
  Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement new");
  if (disp == Teuchos::null) dserror("Cannot get state vector 'displacement new'");
  std::vector<double> mydisp(lm.size());
  CORE::FE::ExtractMyValues(*disp, mydisp, lm);

  CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double> disp_totlag(true);

  // Set current positions and orientations at all nodes:
  UpdateDispTotlag<nnodecl, double>(mydisp, disp_totlag);


  // find out whether we will use a time curve
  double time = -1.0;
  if (this->IsParamsInterface())
    time = this->ParamsInterfacePtr()->GetTotalTime();
  else
    time = params.get("total time", -1.0);

  // get values and switches from the condition:
  // onoff is related to the first 6 flags of a line Neumann condition in the input file;
  // value 1 for flag i says that condition is active for i-th degree of freedom
  const auto* onoff = condition.Get<std::vector<int>>("onoff");

  // val is related to the 6 "val" fields after the onoff flags of the Neumann condition
  // in the input file; val gives the values of the force as a multiple of the prescribed load curve
  const auto* val = condition.Get<std::vector<double>>("val");

  // compute the load vector based on value, scaling factor and whether condition is active
  CORE::LINALG::Matrix<6, 1, double> load_vector_neumann(true);
  for (unsigned int i = 0; i < 6; ++i) load_vector_neumann(i) = (*onoff)[i] * (*val)[i];

  /***********************************************************************************************/

  // if a point neumann condition needs to be linearized
  if (condition.Type() == DRT::Condition::PointNeumannEB)
  {
    // find out whether we will use a time curve and get the factor
    const auto* funct = condition.Get<std::vector<int>>("funct");
    // amplitude of load curve at current time called
    std::vector<double> functtimefac(6, 1.0);

    for (unsigned int i = 0; i < 6; ++i)
    {
      int functnum = -1;
      // number of the load curve related with a specific line Neumann condition called
      if (funct) functnum = (*funct)[i];

      if (functnum > 0)
        functtimefac[i] = GLOBAL::Problem::Instance()
                              ->FunctionById<CORE::UTILS::FunctionOfTime>(functnum - 1)
                              .Evaluate(time);

      load_vector_neumann(i) *= functtimefac[i];
    }

    // find out at which node the condition is applied
    const std::vector<int>* nodeids = condition.GetNodes();
    if (nodeids == nullptr) dserror("failed to retrieve node IDs from condition!");

    /* find out local node number --> this is done since the first element of a Neumann point
     * condition is used for this function in this case we do not know whether it is the left
     * or the right node. In addition to that, xi is assigned in order to determine the index
     * of the base vectors for the smallest rotation system */
    int node = -1;

    if ((*nodeids)[0] == Nodes()[0]->Id())
    {
      node = 0;
    }
    else if ((*nodeids)[0] == Nodes()[1]->Id())
    {
      node = 1;
    }

    if (node == -1) dserror("Node could not be found on nodemap!");


    EvaluatePointNeumannEB<nnodecl>(elevec1, elemat1, disp_totlag, load_vector_neumann, node);
  }
  // if a line neumann condition needs to be linearized
  else if (condition.Type() == DRT::Condition::LineNeumann)
  {
    // funct is related to the 6 "funct" fields after the val field of the Neumann condition
    // in the input file; funct gives the number of the function defined in the section FUNCT
    const auto* function_numbers = condition.Get<std::vector<int>>("funct");

    // Check if distributed moment load is applied and throw error
    if (function_numbers != nullptr)
    {
      for (unsigned int idof = 3; idof < 6; ++idof)
      {
        if ((*function_numbers)[idof] > 0)
          dserror(
              "Line Neumann conditions for distributed moments are not implemented for beam3k"
              " so far! Only the function flag 1, 2 and 3 can be set!");
      }
    }


    EvaluateLineNeumann<nnodecl>(
        elevec1, elemat1, disp_totlag, load_vector_neumann, function_numbers, time);
  }

  return 0;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void DRT::ELEMENTS::Beam3k::EvaluatePointNeumannEB(CORE::LINALG::SerialDenseVector& forcevec,
    CORE::LINALG::SerialDenseMatrix* stiffmat,
    const CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>& disp_totlag,
    const CORE::LINALG::Matrix<6, 1, double>& load_vector_neumann, int node) const
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
    if (not useFAD_)
    {
      CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>
          disp_totlag_centerline(true);
      std::vector<CORE::LINALG::Matrix<3, 3, double>> triad_mat_cp(BEAM3K_COLLOCATION_POINTS);
      std::vector<CORE::LINALG::Matrix<4, 1>> Qref(BEAM3K_COLLOCATION_POINTS);

      UpdateNodalVariables<nnodecl, double>(
          disp_totlag, disp_totlag_centerline, triad_mat_cp, Qref);

      // create view on external force vector to avoid copying
      // IMPORTANT: fext is multiplied by (-1) in BACI, consequently we need no minus sign here
      CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double> f_ext_fixedsize(
          forcevec, true);

      // r' at node
      CORE::LINALG::Matrix<3, 1, double> r_s(true);
      // |r'| at node
      double abs_r_s = 0.0;

      for (unsigned int i = 0; i < 3; ++i) r_s(i) = disp_totlag_centerline(node * 7 + 3 + i);

      abs_r_s = CORE::FADUTILS::Norm(r_s);

      // matrix for moment at node
      CORE::LINALG::Matrix<3, 1, double> moment(true);

      for (unsigned int i = 0; i < 3; ++i)
      {
        moment(i) = load_vector_neumann(i + 3);
      }

      EvaluateResidualFromPointNeumannMoment<nnodecl, double>(
          f_ext_fixedsize, moment, r_s, abs_r_s, node);


      if (stiffmat != nullptr)
      {
        EvaluateStiffMatrixAnalyticFromPointNeumannMoment<nnodecl>(
            *stiffmat, moment, r_s, abs_r_s, node);
      }
    }
    // automatic linearization
    else
    {
      CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> disp_totlag_FAD;

      for (unsigned int idof = 0; idof < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; ++idof)
        disp_totlag_FAD(idof) = disp_totlag(idof);

      // Next, we have to set variables for FAD
      SetAutomaticDifferentiationVariables<nnodecl>(disp_totlag_FAD);

      CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>
          disp_totlag_centerline_FAD(true);
      std::vector<CORE::LINALG::Matrix<3, 3, FAD>> triad_mat_cp_FAD(BEAM3K_COLLOCATION_POINTS);
      std::vector<CORE::LINALG::Matrix<4, 1>> Qref(BEAM3K_COLLOCATION_POINTS);

      UpdateNodalVariables<nnodecl, FAD>(
          disp_totlag_FAD, disp_totlag_centerline_FAD, triad_mat_cp_FAD, Qref);

      // external force vector
      CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> f_ext_FAD(true);


      // r' at node
      CORE::LINALG::Matrix<3, 1, FAD> r_s_FAD(true);
      // |r'| at node
      FAD abs_r_s_FAD = 0.0;

      for (unsigned int i = 0; i < 3; ++i)
        r_s_FAD(i) = disp_totlag_centerline_FAD(node * 7 + 3 + i);

      abs_r_s_FAD = CORE::FADUTILS::Norm(r_s_FAD);


      // matrix for moment at node
      CORE::LINALG::Matrix<3, 1, FAD> moment(true);

      for (unsigned int i = 0; i < 3; ++i)
      {
        moment(i) = load_vector_neumann(i + 3);
      }


      EvaluateResidualFromPointNeumannMoment<nnodecl, FAD>(
          f_ext_FAD, moment, r_s_FAD, abs_r_s_FAD, node);

      // IMPORTANT: fext is multiplied by (-1) in BACI, consequently we need no minus sign here
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
void DRT::ELEMENTS::Beam3k::EvaluateResidualFromPointNeumannMoment(
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& force_ext,
    const CORE::LINALG::Matrix<3, 1, T>& moment_ext, const CORE::LINALG::Matrix<3, 1, T>& r_s,
    T abs_r_s, int node) const
{
  // S(r') at node
  CORE::LINALG::Matrix<3, 3, T> Srs(true);

  // auxiliary quantities
  CORE::LINALG::Matrix<3, 1, T> auxvector(true);
  CORE::LINALG::Matrix<1, 1, T> auxscalar(true);

  CORE::LARGEROTATIONS::computespin(Srs, r_s);
  auxvector.Multiply(Srs, moment_ext);
  auxvector.Scale(-1.0 * std::pow(abs_r_s, -2.0));

  auxscalar.MultiplyTN(r_s, moment_ext);
  auxscalar.Scale(1.0 / abs_r_s);

  for (unsigned int j = 0; j < 3; ++j)
  {
    force_ext(node * 7 + 3 + j) += auxvector(j);
  }

  force_ext(node * 7 + 6) += auxscalar(0, 0);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl>
void DRT::ELEMENTS::Beam3k::EvaluateStiffMatrixAnalyticFromPointNeumannMoment(
    CORE::LINALG::SerialDenseMatrix& stiffmat, const CORE::LINALG::Matrix<3, 1, double>& moment_ext,
    const CORE::LINALG::Matrix<3, 1, double>& r_s, double abs_r_s, int node) const
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
    dserror("%d is an invalid value for element local node ID! Expected 0 or 1", node);
  }

  // matrices storing the assembled shape functions or s-derivative
  CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> N_s;
  CORE::LINALG::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> L;

  // matrices storing individual shape functions, its xi-derivatives and s-derivatives
  CORE::LINALG::Matrix<1, 2 * nnodecl, double> N_i_xi;
  CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i;

  // get all required shape function values
  L_i.Clear();
  CORE::FE::shape_function_1D(L_i, xi_node, Shape());

  L.Clear();
  AssembleShapefunctionsL(L_i, L);

  N_i_xi.Clear();
  CORE::FE::shape_function_hermite_1D_deriv1(N_i_xi, xi_node, length_, CORE::FE::CellType::line2);

  N_s.Clear();
  AssembleShapefunctionsNs(N_i_xi, jacobi_node, N_s);


  // Calculation of first base vector
  CORE::LINALG::Matrix<3, 1, double> g_1(true);
  g_1.Update(std::pow(abs_r_s, -1.0), r_s);

  CORE::LINALG::Matrix<3, 3, double> spinmatrix_of_moment_ext(true);
  CORE::LARGEROTATIONS::computespin(spinmatrix_of_moment_ext, moment_ext);

  CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
      6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>
      lin_v_thetaperp_moment(true), lin_v_thetapar_moment(true);

  Calc_lin_v_thetaperp_moment<nnodecl>(
      lin_v_thetaperp_moment, N_s, g_1, abs_r_s, spinmatrix_of_moment_ext);

  Calc_lin_v_thetapar_moment<nnodecl>(lin_v_thetapar_moment, L, N_s, g_1, abs_r_s, moment_ext);


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
void DRT::ELEMENTS::Beam3k::EvaluateLineNeumann(CORE::LINALG::SerialDenseVector& forcevec,
    CORE::LINALG::SerialDenseMatrix* stiffmat,
    const CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>& disp_totlag,
    const CORE::LINALG::Matrix<6, 1, double>& load_vector_neumann,
    const std::vector<int>* function_numbers, double time) const
{
  if (not useFAD_)
  {
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double> disp_totlag_centerline(
        true);
    std::vector<CORE::LINALG::Matrix<3, 3, double>> triad_mat_cp(BEAM3K_COLLOCATION_POINTS);
    std::vector<CORE::LINALG::Matrix<4, 1>> Qref(BEAM3K_COLLOCATION_POINTS);

    UpdateNodalVariables<nnodecl, double>(disp_totlag, disp_totlag_centerline, triad_mat_cp, Qref);

    // create view on external force vector to avoid copying
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double> f_ext(forcevec, true);

    EvaluateLineNeumannForces<nnodecl, double>(f_ext, load_vector_neumann, function_numbers, time);

    // in tangent-based formulation (rotvec_=false), there is no contribution to stiffmat,
    // so we are done after calculation of forces

    // safety check for rotation-vector based formulation
    if (rotvec_ == true)
      dserror(
          "Beam3k: analytic linearization of LineNeumann condition (distributed forces) "
          "not implemented yet for ROTVEC variant! Activate automatic linearization via FAD");
  }
  else
  {
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> disp_totlag_FAD;

    for (unsigned int idof = 0; idof < 6 * nnodecl + BEAM3K_COLLOCATION_POINTS; ++idof)
      disp_totlag_FAD(idof) = disp_totlag(idof);

    // Next, we have to set variables for FAD
    SetAutomaticDifferentiationVariables<nnodecl>(disp_totlag_FAD);

    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>
        disp_totlag_centerline_FAD(true);
    std::vector<CORE::LINALG::Matrix<3, 3, FAD>> triad_mat_cp_FAD(BEAM3K_COLLOCATION_POINTS);
    std::vector<CORE::LINALG::Matrix<4, 1>> Qref(BEAM3K_COLLOCATION_POINTS);

    UpdateNodalVariables<nnodecl, FAD>(
        disp_totlag_FAD, disp_totlag_centerline_FAD, triad_mat_cp_FAD, Qref);


    // external force vector
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD> f_ext(true);

    EvaluateLineNeumannForces<nnodecl, FAD>(f_ext, load_vector_neumann, function_numbers, time);


    if (rotvec_ == true)
    {
      ApplyRotVecTrafo<nnodecl, FAD>(disp_totlag_centerline_FAD, f_ext);
    }


    // IMPORTANT: fext is multiplied by (-1) in BACI, consequently we need no minus sign here
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

      TransformStiffMatrixMultipl<nnodecl, FAD>(stiffmat, disp_totlag_FAD);
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, typename T>
void DRT::ELEMENTS::Beam3k::EvaluateLineNeumannForces(
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& force_ext,
    const CORE::LINALG::Matrix<6, 1, double>& load_vector_neumann,
    const std::vector<int>* function_numbers, double time) const
{
  std::vector<CORE::LINALG::Matrix<3, 3>> Gref(2);

  for (unsigned int node = 0; node < 2; ++node)
  {
    Gref[node].Clear();
    CORE::LARGEROTATIONS::angletotriad(theta0_[node], Gref[node]);
  }

  // gaussian points
  CORE::FE::IntegrationPoints1D gausspoints = CORE::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);

  CORE::LINALG::Matrix<1, 4, double> N_i;
  CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, T> N;

  // auxiliary external force vector
  CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T> f_ext_aux(true);

  // integration loops
  for (int numgp = 0; numgp < gausspoints.nquad; ++numgp)
  {
    // integration points in parameter space and weights
    const double xi = gausspoints.qxg[numgp][0];
    const double wgt = gausspoints.qwgt[numgp];

    // Clear matrix for shape functions
    N_i.Clear();
    N.Clear();
    CORE::FE::shape_function_hermite_1D(N_i, xi, length_, CORE::FE::CellType::line2);
    AssembleShapefunctionsN(N_i, N);

    // position vector at the gauss point at reference configuration needed for function evaluation
    std::vector<double> X_ref(3, 0.0);
    // calculate coordinates of corresponding Guass point in reference configuration
    for (unsigned int node = 0; node < 2; ++node)
    {
      for (unsigned int dof = 0; dof < 3; ++dof)
      {
        X_ref[dof] +=
            Nodes()[node]->X()[dof] * N_i(2 * node) + (Gref[node])(dof, 0) * N_i(2 * node + 1);
      }
    }


    double functionfac = 1.0;
    CORE::LINALG::Matrix<3, 1, T> force_gp(true);

    // sum up load components
    for (unsigned int idof = 0; idof < 3; ++idof)
    {
      if (function_numbers != nullptr and (*function_numbers)[idof] > 0)
      {
        functionfac =
            GLOBAL::Problem::Instance()
                ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>((*function_numbers)[idof] - 1)
                .Evaluate(X_ref.data(), time, idof);
      }
      else
        functionfac = 1.0;

      force_gp(idof) = load_vector_neumann(idof) * functionfac;
    }

    f_ext_aux.Clear();
    f_ext_aux.MultiplyTN(N, force_gp);

    force_ext.Update(wgt * jacobi_[numgp], f_ext_aux, 1.0);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
inline void DRT::ELEMENTS::Beam3k::CalcBrownianForcesAndStiff(Teuchos::ParameterList& params,
    std::vector<double>& vel, std::vector<double>& disp,
    CORE::LINALG::SerialDenseMatrix* stiffmatrix, CORE::LINALG::SerialDenseVector* force)
{
  if (weakkirchhoff_ == false)
    dserror(
        "calculation of viscous damping moments not implemented for"
        "WK=0 ('strong' Kirchhoff) yet. Use BEAM3WK elements (set WK=1)!");

  if (rotvec_ == true)
    dserror(
        "Beam3k: Calculation of Brownian forces not tested yet for ROTVEC, "
        "i.e. nodal rotation vectors as primary variables. Use ROTVEC=0 (tangent based "
        "formulation)");

  // unshift node positions, i.e. manipulate element displacement vector
  // as if there where no periodic boundary conditions
  if (BrownianDynParamsInterfacePtr() != Teuchos::null)
    UnShiftNodePosition(disp, *BrownianDynParamsInterface().GetPeriodicBoundingBox());


  // total position state of element

  /* vector for current nodal DoFs in total Lagrangian style, i.e. displacement + initial values:
   * rotvec_==true:  disp_totlag=[\v{d}_1, \v{theta}_1, t_1, \v{d}_2, \v{theta}_2, t_2, \alpha_3]
   * rotvec_==false: disp_totlag=[\v{d}_1, \v{t}_1, \alpha_1, \v{d}_2, \v{t}_2, \alpha_2, \alpha_3]
   */
  CORE::LINALG::Matrix<nnode * vpernode * ndim + BEAM3K_COLLOCATION_POINTS, 1, double> disp_totlag(
      true);

  // Set current positions and tangents and triads at all nodes
  UpdateDispTotlag<nnode, double>(disp, disp_totlag);


  // velocity state of element

  // export current velocity state of element to fixed size matrix
  CORE::LINALG::Matrix<nnode * vpernode * ndim + BEAM3K_COLLOCATION_POINTS, 1> vel_fixedsize(
      vel.data(), true);
  CORE::LINALG::Matrix<nnode * vpernode * ndim, 1> vel_centerline(true);

  // update current values of centerline (i.e. translational) velocity
  ExtractCenterlineDofValuesFromElementStateVector<nnode, vpernode, double>(
      vel_fixedsize, vel_centerline);


  // analytic linearization of residual contributions
  if (not useFAD_)
  {
    // force vector resulting from Brownian dynamics
    CORE::LINALG::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, double>
        force_brownian(true);

    if (force != nullptr)
    {
      // set view on CORE::LINALG::SerialDenseVector to avoid copying of data
      force_brownian.SetView(&((*force)(0)));
    }

    // vector containing locally assembled nodal positions and tangents required for centerline:
    // r(s)=N(s)*disp_totlag_centerline, with disp_totlag_centerline=[\v{d}_1, \v{t}_1, 0, \v{d}_2,
    // \v{t}_2, 0, 0]
    CORE::LINALG::Matrix<nnode * vpernode * ndim + BEAM3K_COLLOCATION_POINTS, 1, double>
        disp_totlag_centerline(true);

    // material triads at collocation points
    std::vector<CORE::LINALG::Matrix<3, 3, double>> triad_mat_cp(BEAM3K_COLLOCATION_POINTS);


    UpdateNodalVariables<nnode, double>(disp_totlag, disp_totlag_centerline, triad_mat_cp,
        Qrefnew_);  // Todo @grill: do we need to update Qrefnew_ here? doesn't seem to be a problem
                    // but anyway ...

    CORE::LINALG::Matrix<nnode * vpernode * ndim, 1, double> disp_totlag_centerlineDOFs_only(true);
    ExtractCenterlineDofValuesFromElementStateVector<nnode, vpernode, double>(
        disp_totlag_centerline, disp_totlag_centerlineDOFs_only);


    // Evaluation of force vectors and stiffness matrices

    // add stiffness and forces (i.e. moments) due to rotational damping effects
    EvaluateRotationalDamping<double, nnode, vpernode, ndim>(
        disp_totlag_centerline, triad_mat_cp, stiffmatrix, force_brownian);

    if (stiffmatrix != nullptr) stiff_ptc_ = *stiffmatrix;

    // add stiffness and forces due to translational damping effects
    EvaluateTranslationalDamping<double, nnode, vpernode, ndim>(
        params, vel_centerline, disp_totlag_centerlineDOFs_only, stiffmatrix, force_brownian);

    // add stochastic forces and (if required) resulting stiffness
    EvaluateStochasticForces<double, nnode, vpernode, ndim, 3>(
        disp_totlag_centerlineDOFs_only, stiffmatrix, force_brownian);
  }
  // automatic linearization of residual contributions via FAD
  else
  {
    // force vector resulting from Brownian dynamics
    CORE::LINALG::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, FAD>
        force_brownian_FAD(true);

    // copy pre-computed disp_totlag to a FAD matrix
    CORE::LINALG::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, FAD>
        disp_totlag_FAD(true);

    for (unsigned int idof = 0; idof < ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS; ++idof)
      disp_totlag_FAD(idof) = disp_totlag(idof);


    // vector containing locally assembled nodal positions and tangents required for centerline:
    // r(s)=N(s)*disp_totlag_centerline, with disp_totlag_centerline=[\v{d}_1, \v{t}_1, 0, \v{d}_2,
    // \v{t}_2, 0, 0]
    CORE::LINALG::Matrix<nnode * vpernode * ndim + BEAM3K_COLLOCATION_POINTS, 1, FAD>
        disp_totlag_centerline_FAD(true);

    // material triads at collocation points
    std::vector<CORE::LINALG::Matrix<3, 3, FAD>> triad_mat_cp_FAD(BEAM3K_COLLOCATION_POINTS);


    // Next, we have to set variables for FAD
    SetAutomaticDifferentiationVariables<nnode>(disp_totlag_FAD);

    UpdateNodalVariables<nnode, FAD>(disp_totlag_FAD, disp_totlag_centerline_FAD, triad_mat_cp_FAD,
        Qrefnew_);  // Todo do we need to update Qrefnew_ here? doesn't seem to be a problem but
                    // anyway ...

    CORE::LINALG::Matrix<nnode * vpernode * ndim, 1, FAD> disp_totlag_centerlineDOFs_only(true);
    ExtractCenterlineDofValuesFromElementStateVector<nnode, vpernode, FAD>(
        disp_totlag_centerline_FAD, disp_totlag_centerlineDOFs_only);


    // Evaluation of force vectors and stiffness matrices

    // add stiffness and forces due to translational damping effects
    EvaluateTranslationalDamping<FAD, nnode, vpernode, ndim>(
        params, vel_centerline, disp_totlag_centerlineDOFs_only, stiffmatrix, force_brownian_FAD);

    // add stochastic forces and (if required) resulting stiffness
    EvaluateStochasticForces<FAD, nnode, vpernode, ndim, 3>(
        disp_totlag_centerlineDOFs_only, stiffmatrix, force_brownian_FAD);

    // add stiffness and forces (i.e. moments) due to rotational damping effects
    EvaluateRotationalDamping<FAD, nnode, vpernode, ndim>(
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
void DRT::ELEMENTS::Beam3k::EvaluateTranslationalDamping(Teuchos::ParameterList& params,
    const CORE::LINALG::Matrix<ndim * vpernode * nnode, 1, double>& vel,
    const CORE::LINALG::Matrix<ndim * vpernode * nnode, 1, T>& disp_totlag,
    CORE::LINALG::SerialDenseMatrix* stiffmatrix,
    CORE::LINALG::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, T>& f_int)
{
  /* only nodes for centerline interpolation are considered here (first two nodes of this element)
     each of these nodes holds 3*vpernode translational DoFs AND 1 rotational DoFs */
  const unsigned int dofpernode = ndim * vpernode + 1;

  // get damping coefficients for translational and rotational degrees of freedom (the latter is
  // unused in this element)
  CORE::LINALG::Matrix<ndim, 1> gamma(true);
  GetDampingCoefficients(gamma);

  // velocity and gradient of background velocity field
  CORE::LINALG::Matrix<ndim, 1, T> velbackground(true);
  CORE::LINALG::Matrix<ndim, ndim, T> velbackgroundgrad(true);

  // evaluation point in physical space corresponding to a certain Gauss point in parameter space
  CORE::LINALG::Matrix<ndim, 1, T> evaluationpoint(true);
  // tangent vector (derivative of beam centerline curve r with respect to arc-length parameter s)
  CORE::LINALG::Matrix<ndim, 1, T> r_s(true);
  // velocity of beam centerline point relative to background fluid velocity
  CORE::LINALG::Matrix<ndim, 1, T> vel_rel(true);

  // viscous force vector per unit length at current GP
  CORE::LINALG::Matrix<ndim, 1, T> f_visc(true);
  // damping matrix
  CORE::LINALG::Matrix<ndim, ndim, T> damp_mat(true);


  // get Gauss points and weights
  CORE::FE::IntegrationPoints1D gausspoints = CORE::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);

  // matrix to store individual Hermite shape functions and their derivatives evaluated at a certain
  // Gauss point
  CORE::LINALG::Matrix<1, nnode * vpernode, double> N_i(true);
  CORE::LINALG::Matrix<1, nnode * vpernode, double> N_i_xi(true);


  for (int gp = 0; gp < gausspoints.nquad; gp++)
  {
    DRT::UTILS::BEAM::EvaluateShapeFunctionsAndDerivsAtXi<nnode, vpernode>(
        gausspoints.qxg[gp][0], N_i, N_i_xi, this->Shape(), this->RefLength());

    // compute position vector r of point in physical space corresponding to Gauss point
    Calc_r<nnode, vpernode, T>(disp_totlag, N_i, evaluationpoint);

    // compute tangent vector t_{\par}=r' at current Gauss point
    Calc_r_s<nnode, vpernode, T>(disp_totlag, N_i_xi, jacobi_[gp], r_s);

    // compute velocity and gradient of background flow field at point r
    GetBackgroundVelocity<ndim, T>(params, evaluationpoint, velbackground, velbackgroundgrad);

    /* compute velocity vector at this Gauss point via same interpolation as for centerline
     * position vector
     *
     * Be careful here:
     * special treatment is required if Fad is used. see method with Fad parameters for details */
    Calc_velocity<nnode, vpernode, ndim>(vel, N_i, vel_rel, evaluationpoint, gp);

    vel_rel -= velbackground;

    // loop over lines and columns of damping matrix
    for (unsigned int idim = 0; idim < ndim; idim++)
      for (unsigned int jdim = 0; jdim < ndim; jdim++)
        damp_mat(idim, jdim) =
            (idim == jdim) * gamma(1) + (gamma(0) - gamma(1)) * r_s(idim) * r_s(jdim);

    // compute viscous force vector per unit length at current GP
    f_visc.Multiply(damp_mat, vel_rel);

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
      EvaluateAnalyticStiffmatContributionsFromTranslationalDamping<nnode, vpernode, ndim>(
          *stiffmatrix, damp_mat, r_s, vel_rel, gamma, velbackgroundgrad, N_i, N_i_xi, jacobi_[gp],
          gausspoints.qwgt[gp]);
    }
  }
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
void DRT::ELEMENTS::Beam3k::EvaluateAnalyticStiffmatContributionsFromTranslationalDamping(
    CORE::LINALG::SerialDenseMatrix& stiffmatrix,
    const CORE::LINALG::Matrix<ndim, ndim, double>& damping_matrix,
    const CORE::LINALG::Matrix<ndim, 1, double>& r_s,
    const CORE::LINALG::Matrix<ndim, 1, double>& vel_rel,
    const CORE::LINALG::Matrix<ndim, 1, double>& gamma,
    const CORE::LINALG::Matrix<ndim, ndim, double>& velbackgroundgrad,
    const CORE::LINALG::Matrix<1, nnode * vpernode, double>& N_i,
    const CORE::LINALG::Matrix<1, nnode * vpernode, double>& N_i_xi, double jacobifactor,
    double gp_weight) const
{
  /* only nodes for centerline interpolation are considered here (first two nodes of this element)
     each of these nodes holds 3*vpernode translational DoFs AND 1 rotational DoFs */
  const unsigned int dofpernode = ndim * vpernode + 1;

  // get time step size
  const double dt = ParamsInterface().GetDeltaTime();

  // compute matrix product of damping matrix and gradient of background velocity
  CORE::LINALG::Matrix<ndim, ndim> dampmatvelbackgroundgrad(true);
  dampmatvelbackgroundgrad.Multiply(damping_matrix, velbackgroundgrad);

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
void DRT::ELEMENTS::Beam3k::EvaluateStochasticForces(
    const CORE::LINALG::Matrix<ndim * vpernode * nnode, 1, T>& disp_totlag,
    CORE::LINALG::SerialDenseMatrix* stiffmatrix,
    CORE::LINALG::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, T>& f_int)
{
  /* only nodes for centerline interpolation are considered here (first two nodes of this element)
     each of these nodes holds 3*vpernode translational DoFs AND 1 rotational DoFs */
  const unsigned int dofpernode = ndim * vpernode + 1;

  // damping coefficients for three translational and one rotational degree of freedom
  CORE::LINALG::Matrix<3, 1> gamma(true);
  GetDampingCoefficients(gamma);

  /* get pointer at Epetra multivector in parameter list linking to random numbers for stochastic
   * forces with zero mean and standard deviation (2*kT / dt)^0.5 */
  Teuchos::RCP<Epetra_MultiVector> randomforces = BrownianDynParamsInterface().GetRandomForces();

  // tangent vector (derivative of beam centerline curve r with respect to arc-length parameter s)
  CORE::LINALG::Matrix<ndim, 1, T> r_s(true);

  // my random number vector at current GP
  CORE::LINALG::Matrix<ndim, 1, double> randnumvec(true);

  // stochastic force vector per unit length at current GP
  CORE::LINALG::Matrix<ndim, 1, T> f_stoch(true);


  // get Gauss points and weights for evaluation of damping matrix
  CORE::FE::IntegrationPoints1D gausspoints = CORE::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);

  // matrix to store Hermite shape functions and their derivatives evaluated at a certain Gauss
  // point
  CORE::LINALG::Matrix<1, nnode * vpernode, double> N_i;
  CORE::LINALG::Matrix<1, nnode * vpernode, double> N_i_xi;

  for (int gp = 0; gp < gausspoints.nquad; gp++)
  {
    DRT::UTILS::BEAM::EvaluateShapeFunctionsAndDerivsAtXi<nnode, vpernode>(
        gausspoints.qxg[gp][0], N_i, N_i_xi, this->Shape(), this->RefLength());

    // compute tangent vector t_{\par}=r' at current Gauss point
    Calc_r_s<nnode, vpernode, T>(disp_totlag, N_i_xi, jacobi_[gp], r_s);

    // extract random numbers from global vector
    for (unsigned int idim = 0; idim < ndim; idim++)
      randnumvec(idim) = (*randomforces)[gp * randompergauss + idim][LID()];

    // compute stochastic force vector per unit length at current GP
    f_stoch.Clear();
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
      EvaluateAnalyticStiffmatContributionsFromStochasticForces<nnode, vpernode, ndim>(
          *stiffmatrix, r_s, randnumvec, gamma, N_i, N_i_xi, jacobi_[gp], gausspoints.qwgt[gp]);
    }
  }
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
void DRT::ELEMENTS::Beam3k::EvaluateAnalyticStiffmatContributionsFromStochasticForces(
    CORE::LINALG::SerialDenseMatrix& stiffmatrix, const CORE::LINALG::Matrix<ndim, 1, double>& r_s,
    const CORE::LINALG::Matrix<ndim, 1, double>& randnumvec,
    const CORE::LINALG::Matrix<ndim, 1, double>& gamma,
    const CORE::LINALG::Matrix<1, nnode * vpernode, double>& N_i,
    const CORE::LINALG::Matrix<1, nnode * vpernode, double>& N_i_xi, double jacobifactor,
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
void DRT::ELEMENTS::Beam3k::EvaluateRotationalDamping(
    const CORE::LINALG::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, T>&
        disp_totlag_centerline,
    const std::vector<CORE::LINALG::Matrix<ndim, ndim, T>>& triad_mat_cp,
    CORE::LINALG::SerialDenseMatrix* stiffmatrix,
    CORE::LINALG::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, T>& f_int)
{
  // get time step size
  const double dt = ParamsInterface().GetDeltaTime();

  // get damping coefficients for translational and rotational degrees of freedom
  CORE::LINALG::Matrix<3, 1> gamma(true);
  GetDampingCoefficients(gamma);

  // get Gauss points and weights for evaluation of viscous damping contributions
  CORE::FE::IntegrationPoints1D gausspoints = CORE::FE::IntegrationPoints1D(MYGAUSSRULEBEAM3K);

  CORE::LINALG::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, T> f_int_aux(true);

  // CP values of strains and their variations needed for interpolation
  std::vector<CORE::LINALG::Matrix<6 * nnode + BEAM3K_COLLOCATION_POINTS, 3, T>> v_thetapar_cp(
      BEAM3K_COLLOCATION_POINTS);

  // re-interpolated values of strains and their variations evaluated at Gauss points
  CORE::LINALG::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 3, T> v_thetapar_bar(
      true);

  std::vector<CORE::LINALG::Matrix<ndim, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, T>>
      lin_theta_cp(BEAM3K_COLLOCATION_POINTS);

  // Interpolated material triad and local rotation vector evaluated at Gauss point
  CORE::LINALG::Matrix<3, 3, T> triad_mat(true);
  CORE::LINALG::Matrix<3, 1, T> theta(true);


  // matrices holding the assembled shape functions and s-derivatives
  CORE::LINALG::Matrix<3, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, T> N_s;
  CORE::LINALG::Matrix<1, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, T> L;

  // Matrices for individual shape functions and xi-derivatives
  CORE::LINALG::Matrix<1, vpernode * nnode, double> N_i_xi;
  CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i;

  // Additional kinematic quantities
  CORE::LINALG::Matrix<3, 1, T> r_s;  // Matrix to store r'
  T abs_r_s;                          // ||r'||

  // create object of triad interpolation scheme
  Teuchos::RCP<LARGEROTATIONS::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS, T>>
      triad_interpolation_scheme_ptr = Teuchos::rcp(
          new LARGEROTATIONS::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS,
              T>());

  // reset scheme with nodal triads
  triad_interpolation_scheme_ptr->Reset(triad_mat_cp);


  //********begin: evaluate quantities at collocation points********************************
  for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; ++node)
  {
    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    const double xi_cp = (double)node / (double)(BEAM3K_COLLOCATION_POINTS - 1) * 2.0 - 1.0;

    // Determine storage position for the node node
    const unsigned int ind =
        CORE::LARGEROTATIONS::NumberingTrafo(node + 1, BEAM3K_COLLOCATION_POINTS);

    // get value of interpolating function of theta (Lagrange polynomials) at xi
    L_i.Clear();
    CORE::FE::shape_function_1D(L_i, xi_cp, Shape());

    L.Clear();
    AssembleShapefunctionsL(L_i, L);

    N_i_xi.Clear();
    CORE::FE::shape_function_hermite_1D_deriv1(N_i_xi, xi_cp, length_, CORE::FE::CellType::line2);

    N_s.Clear();
    AssembleShapefunctionsNs(N_i_xi, jacobi_cp_[ind], N_s);

    // Calculation of r' at xi
    r_s.Clear();
    r_s.Multiply(N_s, disp_totlag_centerline);

    abs_r_s = CORE::FADUTILS::Norm<T>(r_s);

    Calc_v_thetapartheta<2, T>(v_thetapar_cp[ind], L, r_s, abs_r_s);

    PreComputeTermsAtCPForAnalyticStiffmatContributionsFromRotationalDamping<2, 2, 3>(
        lin_theta_cp[ind], L, N_s, r_s, abs_r_s, Qrefconv_[ind]);
  }


  // loop through Gauss points
  for (int gp = 0; gp < gausspoints.nquad; ++gp)
  {
    // get location and weight of GP in parameter space
    const double xi_gp = gausspoints.qxg[gp][0];
    const double wgt = gausspoints.qwgt[gp];

    // evaluate shape functions
    L_i.Clear();
    CORE::FE::shape_function_1D(L_i, xi_gp, this->Shape());

    v_thetapar_bar.Clear();
    for (unsigned int node = 0; node < BEAM3K_COLLOCATION_POINTS; ++node)
    {
      v_thetapar_bar.Update(L_i(node), v_thetapar_cp[node], 1.0);
    }

    // compute material triad at gp
    triad_mat.Clear();

    // compute quaterion of material triad at gp
    CORE::LINALG::Matrix<4, 1, T> Qnewmass(true);

    triad_interpolation_scheme_ptr->GetInterpolatedLocalRotationVector(theta, L_i);

    triad_interpolation_scheme_ptr->GetInterpolatedQuaternion(Qnewmass, theta);

    CORE::LARGEROTATIONS::quaterniontotriad(Qnewmass, triad_mat);

    // store in class variable in order to get QconvGPmass_ in subsequent time step
    for (unsigned int i = 0; i < 4; ++i)
      (Qnewmass_[gp])(i) = CORE::FADUTILS::CastToDouble(Qnewmass(i));

    CORE::LINALG::Matrix<4, 1, T> Qconv(true);
    for (unsigned int i = 0; i < 4; ++i) Qconv(i) = (Qconvmass_[gp])(i);

    CORE::LINALG::Matrix<3, 3, T> triad_mat_conv(true);
    CORE::LARGEROTATIONS::quaterniontotriad(Qconv, triad_mat_conv);


    // compute quaternion of relative rotation from converged to current state
    CORE::LINALG::Matrix<3, 3, T> deltatriad(true);
    deltatriad.MultiplyNT(triad_mat, triad_mat_conv);

    CORE::LINALG::Matrix<4, 1, T> deltaQ(true);
    CORE::LARGEROTATIONS::triadtoquaternion(deltatriad, deltaQ);

    CORE::LINALG::Matrix<3, 1, T> deltatheta(true);
    CORE::LARGEROTATIONS::quaterniontoangle(deltaQ, deltatheta);


    // angular velocity at this Gauss point according to backward Euler scheme
    CORE::LINALG::Matrix<3, 1, T> omega(true);
    omega.Update(1.0 / dt, deltatheta);

    // compute matrix Lambda*[gamma(2) 0 0 \\ 0 0 0 \\ 0 0 0]*Lambda^t = gamma(2) * g_1 \otimes g_1
    // where g_1 is first base vector, i.e. first column of Lambda
    CORE::LINALG::Matrix<3, 3, T> g1g1gamma(true);
    for (unsigned int k = 0; k < 3; ++k)
      for (unsigned int j = 0; j < 3; ++j)
        g1g1gamma(k, j) = triad_mat(k, 0) * triad_mat(j, 0) * gamma(2);

    // compute vector gamma(2) * g_1 \otimes g_1 * \omega (viscous moment per unit length)
    CORE::LINALG::Matrix<3, 1, T> m_visc(true);
    m_visc.Multiply(g1g1gamma, omega);


    // residual contribution from viscous damping moment
    f_int_aux.Clear();
    f_int_aux.Multiply(v_thetapar_bar, m_visc);

    f_int.Update(wgt * jacobi_[gp], f_int_aux, 1.0);


    if (stiffmatrix != nullptr)
    {
      EvaluateAnalyticStiffmatContributionsFromRotationalDamping<2, 2, 3>(*stiffmatrix,
          disp_totlag_centerline, *triad_interpolation_scheme_ptr, theta, deltatheta, triad_mat,
          triad_mat_conv, v_thetapar_bar, lin_theta_cp, m_visc, gamma(2), dt, xi_gp,
          wgt * jacobi_[gp]);
    }
  }
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <unsigned int nnodecl, unsigned int vpernode, unsigned int ndim>
void DRT::ELEMENTS::Beam3k::EvaluateAnalyticStiffmatContributionsFromRotationalDamping(
    CORE::LINALG::SerialDenseMatrix& stiffmatrix,
    const CORE::LINALG::Matrix<ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>&
        disp_totlag_centerline,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS, double>&
        triad_intpol,
    const CORE::LINALG::Matrix<3, 1, double> theta_gp,
    const CORE::LINALG::Matrix<3, 1, double>& deltatheta_gp,
    const CORE::LINALG::Matrix<3, 3, double>& triad_mat_gp,
    const CORE::LINALG::Matrix<3, 3, double>& triad_mat_conv_gp,
    const CORE::LINALG::Matrix<ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS, ndim, double>&
        v_theta_par_bar,
    const std::vector<CORE::LINALG::Matrix<ndim,
        ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS, double>>& lin_theta_cp,
    const CORE::LINALG::Matrix<3, 1, double> moment_viscous, double gamma_polar, double dt,
    double xi_gp, double jacobifac_GPwgt) const
{
  // size of Dof vector of this element
  const unsigned int numdofelement = ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS;

  // create a fixed size matrix as view on the CORE::LINALG::SerialDenseMatrix to avoid copying
  CORE::LINALG::Matrix<numdofelement, numdofelement, double> stiffmatrix_fixedsize(
      stiffmatrix, true);


  // matrices storing the assembled shape functions or s-derivative
  CORE::LINALG::Matrix<ndim, numdofelement, double> N_s;
  CORE::LINALG::Matrix<1, numdofelement, double> L;

  // matrices storing individual shape functions, its xi-derivatives and s-derivatives
  CORE::LINALG::Matrix<1, vpernode * nnodecl, double> N_i_xi;
  CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, double> L_i;

  // r' vector and its norm
  CORE::LINALG::Matrix<3, 1, double> r_s_cp(true);
  double abs_r_s_cp = 0.0;

  // first base vector
  CORE::LINALG::Matrix<3, 1, double> g_1_cp(true);


  CORE::LINALG::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double> lin_theta_bar(true);


  // linearization of re-interpolated strain variations
  std::vector<CORE::LINALG::Matrix<numdofelement, numdofelement, double>> lin_v_thetapar_moment_cp(
      BEAM3K_COLLOCATION_POINTS);

  CORE::LINALG::Matrix<numdofelement, numdofelement, double> lin_v_thetapar_bar_moment(true);


  CORE::LINALG::Matrix<3, 3, double> spinmatrix_of_moment_visc(true);
  CORE::LARGEROTATIONS::computespin<double>(spinmatrix_of_moment_visc, moment_viscous);


  // linearization of moment_visc
  CORE::LINALG::Matrix<3, numdofelement, double> lin_moment_viscous(true);


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
    ind = CORE::LARGEROTATIONS::NumberingTrafo(icp + 1, BEAM3K_COLLOCATION_POINTS);

    // calculate xi of cp
    // node=0->xi=-1  node=1->xi=0  node=2->xi=1
    xi_cp = (double)icp / (double)(BEAM3K_COLLOCATION_POINTS - 1) * 2.0 - 1.0;


    // get all required shape function values
    L_i.Clear();
    CORE::FE::shape_function_1D(L_i, xi_cp, Shape());

    L.Clear();
    AssembleShapefunctionsL(L_i, L);

    N_i_xi.Clear();
    CORE::FE::shape_function_hermite_1D_deriv1(N_i_xi, xi_cp, length_, CORE::FE::CellType::line2);

    N_s.Clear();
    AssembleShapefunctionsNs(N_i_xi, jacobi_cp_[ind], N_s);


    // Calculation of r' and g_1
    r_s_cp.Clear();
    r_s_cp.Multiply(N_s, disp_totlag_centerline);

    abs_r_s_cp = r_s_cp.Norm2();  // Todo think about computing and storing inverse value here

    g_1_cp.Clear();
    g_1_cp.Update(std::pow(abs_r_s_cp, -1.0), r_s_cp);


    Calc_lin_v_thetapar_moment<nnodecl>(
        lin_v_thetapar_moment_cp[ind], L, N_s, g_1_cp, abs_r_s_cp, moment_viscous);
  }

  /***********************************************************************************************/
  // re-interpolation of quantities at xi based on CP values

  L_i.Clear();
  CORE::FE::shape_function_1D(L_i, xi_gp, Shape());

  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    lin_v_thetapar_bar_moment.Update(L_i(icp), lin_v_thetapar_moment_cp[icp], 1.0);
  }
  /***********************************************************************************************/

  std::vector<CORE::LINALG::Matrix<3, 3, double>> Itilde(BEAM3K_COLLOCATION_POINTS);

  // compute Itilde matrices required for re-interpolation of CP values of lin_theta
  triad_intpol.GetNodalGeneralizedRotationInterpolationMatrices(Itilde, theta_gp, L_i);

  CORE::LINALG::Matrix<3, numdofelement, double> auxmatrix(true);

  lin_theta_bar.Clear();
  for (unsigned int icp = 0; icp < BEAM3K_COLLOCATION_POINTS; ++icp)
  {
    auxmatrix.Clear();

    auxmatrix.Multiply(Itilde[icp], lin_theta_cp[icp]);

    lin_theta_bar.Update(1.0, auxmatrix, 1.0);
  }

  Calc_lin_moment_viscous<nnodecl>(lin_moment_viscous, triad_mat_gp, triad_mat_conv_gp,
      deltatheta_gp, lin_theta_bar, spinmatrix_of_moment_visc, gamma_polar, dt);

  /***********************************************************************************************/
  // finally put everything together
  CORE::LINALG::Matrix<numdofelement, numdofelement, double> auxmatrix2(true);

  // linearization of residual from rotational damping moment
  stiffmatrix_fixedsize.Update(jacobifac_GPwgt, lin_v_thetapar_bar_moment, 1.0);

  auxmatrix2.Clear();
  auxmatrix2.Multiply(v_theta_par_bar, lin_moment_viscous);

  stiffmatrix_fixedsize.Update(jacobifac_GPwgt, auxmatrix2, 1.0);
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
void DRT::ELEMENTS::Beam3k::
    PreComputeTermsAtCPForAnalyticStiffmatContributionsFromRotationalDamping(
        CORE::LINALG::Matrix<ndim, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, double>&
            lin_theta,
        const CORE::LINALG::Matrix<1, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, double>&
            L,
        const CORE::LINALG::Matrix<ndim, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS,
            double>& N_s,
        const CORE::LINALG::Matrix<ndim, 1, double>& r_s, double abs_r_s,
        const CORE::LINALG::Matrix<4, 1, double>& Qref_conv) const
{
  // size of Dof vector of this element
  const unsigned int numdofelement = ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS;

  CORE::LINALG::Matrix<ndim, 1, double> g_1(true);
  g_1.Update(std::pow(abs_r_s, -1.0), r_s);

  CORE::LINALG::Matrix<ndim, 1, double> g_1_bar(true);

  CORE::LINALG::Matrix<3, 3, double> triad_ref_conv_cp(true);
  CORE::LARGEROTATIONS::quaterniontotriad(Qref_conv, triad_ref_conv_cp);

  g_1_bar.Clear();
  for (unsigned int idim = 0; idim < ndim; ++idim) g_1_bar(idim) = triad_ref_conv_cp(idim, 0);

  // CP values of strain increments
  CORE::LINALG::Matrix<ndim, numdofelement, double> lin_theta_perp(true), lin_theta_par(true);

  Calc_lin_thetapar<nnode>(lin_theta_par, L, N_s, g_1, g_1_bar, abs_r_s);

  Calc_lin_thetaperp<nnode>(lin_theta_perp, N_s, r_s, abs_r_s);

  // lin_theta
  lin_theta.Clear();
  lin_theta.Update(1.0, lin_theta_par, 1.0, lin_theta_perp);
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
int DRT::ELEMENTS::Beam3k::HowManyRandomNumbersINeed() const
{
  // get Gauss rule for evaluation of stochastic force contributions
  CORE::FE::GaussRule1D gaussrule = MYGAUSSRULEBEAM3K;
  CORE::FE::IntegrationPoints1D gausspoints(gaussrule);

  /* at each Gauss point one needs as many random numbers as randomly excited degrees of freedom,
   * i.e. three random numbers for the translational degrees of freedom */
  return (3 * gausspoints.nquad);
}

/*-----------------------------------------------------------------------------------------------------------*
 |  Assemble C shape function meier 01/16|
 *-----------------------------------------------------------------------------------------------------------*/
template <typename T1, typename T2>
void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsL(
    CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, T1>& L_i,
    CORE::LINALG::Matrix<1, 2 * 6 + BEAM3K_COLLOCATION_POINTS, T2>& L) const
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
  dserror(
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
void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsNss(CORE::LINALG::Matrix<1, 4, T1>& N_i_xi,
    CORE::LINALG::Matrix<1, 4, T1>& N_i_xixi, double jacobi, double jacobi2,
    CORE::LINALG::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, T2>& N_ss) const
{
  CORE::LINALG::Matrix<1, 4, T1> N_i_ss(true);
  N_i_ss.Update(std::pow(jacobi, -2.0), N_i_xixi, 1.0);
  N_i_ss.Update(-1.0 * jacobi2 * std::pow(jacobi, -4.0), N_i_xi, 1.0);

  AssembleShapefunctionsN(N_i_ss, N_ss);
}

/*-----------------------------------------------------------------------------------------------------------*
 |  Assemble the N_s shape functions meier 01/16|
 *-----------------------------------------------------------------------------------------------------------*/
template <typename T1, typename T2>
void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsNs(CORE::LINALG::Matrix<1, 4, T1>& N_i_xi,
    double jacobi, CORE::LINALG::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, T2>& N_s) const
{
  CORE::LINALG::Matrix<1, 4, T1> N_i_s(true);

  // Calculate the derivatives in s
  N_i_s = N_i_xi;
  N_i_s.Scale(std::pow(jacobi, -1.0));

  AssembleShapefunctionsN(N_i_s, N_s);
}

/*-----------------------------------------------------------------------------------------------------------*
 |  Assemble the N shape functions meier 01/16|
 *-----------------------------------------------------------------------------------------------------------*/
template <typename T1, typename T2>
void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsN(CORE::LINALG::Matrix<1, 4, T1>& N_i,
    CORE::LINALG::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, T2>& N) const
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
  dserror(
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
void DRT::ELEMENTS::Beam3k::ApplyRotVecTrafo(
    const CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
        disp_totlag_centerline,
    CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& f_int) const
{
  // Trafo matrices:
  CORE::LINALG::Matrix<4, 4, T> trafomat(true);
  CORE::LINALG::Matrix<3, 1, T> g_1(true);
  CORE::LINALG::Matrix<3, 3, T> auxmatrix(true);
  T t = 0.0;
  CORE::LINALG::Matrix<4, 1, T> f_aux1(true);
  CORE::LINALG::Matrix<4, 1, T> f_aux2(true);

  for (unsigned int node = 0; node < 2; ++node)
  {
    g_1.Clear();
    t = 0.0;
    auxmatrix.Clear();

    for (unsigned int i = 0; i < 3; ++i)
    {
      g_1(i) = disp_totlag_centerline(7 * node + 3 + i);
    }

    t = CORE::FADUTILS::Norm<T>(g_1);
    g_1.Scale(1.0 / t);
    CORE::LARGEROTATIONS::computespin(auxmatrix, g_1);
    auxmatrix.Scale(-1.0 * t);
    trafomat.Clear();

    for (unsigned int i = 0; i < 3; ++i)
    {
      for (unsigned int j = 0; j < 3; ++j)
      {
        trafomat(i, j) = auxmatrix(i, j);
      }
      trafomat(i, 3) = g_1(i);
      trafomat(3, i) = g_1(i);
    }

    f_aux1.Clear();
    f_aux2.Clear();
    for (unsigned int i = 0; i < 4; ++i)
    {
      f_aux1(i) = f_int(7 * node + 3 + i);
    }

    f_aux2.MultiplyTN(trafomat, f_aux1);

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
void DRT::ELEMENTS::Beam3k::TransformStiffMatrixMultipl(
    CORE::LINALG::SerialDenseMatrix* stiffmatrix,
    const CORE::LINALG::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag) const
{
  // we need to transform the stiffmatrix because its entries are derivatives with respect to
  // additive rotational increments we want a stiffmatrix containing derivatives with respect to
  // multiplicative rotational increments therefore apply a trafo matrix to all those 3x3 blocks in
  // stiffmatrix which correspond to derivation with respect to rotational DOFs the trafo matrix is
  // simply the T-Matrix (see Jelenic1999, (2.4)): \Delta_{mult} \vec \theta_{inode} = \mat T(\vec
  // \theta_{inode} * \Delta_{addit} \vec \theta_{inode}
  CORE::LINALG::Matrix<2 * 6 + BEAM3K_COLLOCATION_POINTS, 3> tempmat(true);
  CORE::LINALG::Matrix<2 * 6 + BEAM3K_COLLOCATION_POINTS, 3> newstiffmat(true);
  CORE::LINALG::Matrix<3, 3> Tmat(true);
  std::vector<CORE::LINALG::Matrix<3, 1>> theta(2, CORE::LINALG::Matrix<3, 1>(true));

  // Loop over the two boundary nodes
  for (unsigned int node = 0; node < 2; node++)
  {
    for (unsigned int i = 0; i < 3; ++i)
      theta[node](i) = CORE::FADUTILS::CastToDouble(disp_totlag(7 * node + 3 + i));
  }

  // Loop over the two boundary nodes
  for (unsigned int node = 0; node < 2; ++node)
  {
    Tmat.Clear();
    Tmat = CORE::LARGEROTATIONS::Tmatrix(theta[node]);

    tempmat.Clear();
    for (unsigned int i = 0; i < 2 * 6 + BEAM3K_COLLOCATION_POINTS; ++i)
      for (unsigned int j = 0; j < 3; ++j) tempmat(i, j) = (*stiffmatrix)(i, 7 * node + 3 + j);

    newstiffmat.Clear();
    newstiffmat.MultiplyNN(tempmat, Tmat);

    for (unsigned int i = 0; i < 2 * 6 + BEAM3K_COLLOCATION_POINTS; ++i)
      for (unsigned int j = 0; j < 3; ++j) (*stiffmatrix)(i, 7 * node + 3 + j) = newstiffmat(i, j);
  }
}

template <typename T>
void DRT::ELEMENTS::Beam3k::straintostress(const CORE::LINALG::Matrix<3, 1, T>& Omega,
    const T& epsilon, const CORE::LINALG::Matrix<3, 3, T>& Cn,
    const CORE::LINALG::Matrix<3, 3, T>& Cm, CORE::LINALG::Matrix<3, 1, T>& M, T& f_par) const
{
  f_par = 0.0;
  f_par = Cn(0, 0) * epsilon;

  M.Clear();
  M(0) = Cm(0, 0) * Omega(0);
  M(1) = Cm(1, 1) * Omega(1);
  M(2) = Cm(2, 2) * Omega(2);
}


void DRT::ELEMENTS::Beam3k::CalcStiffContributionsPTC(CORE::LINALG::SerialDenseMatrix& elemat1)
{
  elemat1 = stiff_ptc_;
}

//      //*******************************Begin:
//      FD-CHECK************************************************************
//      //the following code block can be used to check quickly whether the nonlinear stiffness
//      matrix is calculated
//      //correctly or not by means of a numerically approximated stiffness matrix. Uncomment this
//      code block and copy
//      //it to the marked place in the method DRT::ELEMENTS::Beam3k::Evaluate() on the top of this
//      file! if(Id() == 0) //limiting the following tests to certain element numbers
//      {
//
//        //variable to store numerically approximated stiffness matrix
//        CORE::LINALG::SerialDenseMatrix stiff_approx;
//        stiff_approx.Shape(6*2+BEAM3K_COLLOCATION_POINTS,6*2+BEAM3K_COLLOCATION_POINTS);
//
//
//        //relative error of numerically approximated stiffness matrix
//        CORE::LINALG::SerialDenseMatrix stiff_relerr;
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
//          CORE::LINALG::SerialDenseVector force_aux;
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
//            //suppressing small entries whose effect is only confusing and NaN entires (which
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
//          std::cout<<"\n\n acutally calculated stiffness matrix\n";
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
template void DRT::ELEMENTS::Beam3k::CalculateInternalForcesAndStiffWK<2, double>(
    Teuchos::ParameterList&,
    const CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<3, 3, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    std::vector<CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 3, double>>&,
    std::vector<CORE::LINALG::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>>&,
    std::vector<CORE::LINALG::Matrix<3, 3, double>>&);
template void DRT::ELEMENTS::Beam3k::CalculateInternalForcesAndStiffWK<2, FAD>(
    Teuchos::ParameterList&, const CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    const std::vector<CORE::LINALG::Matrix<3, 3, FAD>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    std::vector<CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 3, FAD>>&,
    std::vector<CORE::LINALG::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, FAD>>&,
    std::vector<CORE::LINALG::Matrix<3, 3, FAD>>&);

template void DRT::ELEMENTS::Beam3k::CalculateInternalForcesAndStiffSK<2>(Teuchos::ParameterList&,
    const CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    const std::vector<CORE::LINALG::Matrix<3, 3, FAD>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    std::vector<CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 3, FAD>>&,
    std::vector<CORE::LINALG::Matrix<3, 3, FAD>>&);

template void DRT::ELEMENTS::Beam3k::CalculateInertiaForcesAndMassMatrix<2, double>(
    Teuchos::ParameterList&, const std::vector<CORE::LINALG::Matrix<3, 3, double>>&,
    const CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 3, double>>&,
    const std::vector<CORE::LINALG::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, double>>&,
    CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    CORE::LINALG::SerialDenseMatrix*);
template void DRT::ELEMENTS::Beam3k::CalculateInertiaForcesAndMassMatrix<2, FAD>(
    Teuchos::ParameterList&, const std::vector<CORE::LINALG::Matrix<3, 3, FAD>>&,
    const CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    const std::vector<CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 3, FAD>>&,
    const std::vector<CORE::LINALG::Matrix<3, 6 * 2 + BEAM3K_COLLOCATION_POINTS, FAD>>&,
    CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    CORE::LINALG::SerialDenseMatrix*);

template void DRT::ELEMENTS::Beam3k::EvaluatePointNeumannEB<2>(CORE::LINALG::SerialDenseVector&,
    CORE::LINALG::SerialDenseMatrix*,
    const CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const CORE::LINALG::Matrix<6, 1, double>&, int) const;

template void DRT::ELEMENTS::Beam3k::EvaluateResidualFromPointNeumannMoment<2, double>(
    CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&, double,
    int) const;
template void DRT::ELEMENTS::Beam3k::EvaluateResidualFromPointNeumannMoment<2, FAD>(
    CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    const CORE::LINALG::Matrix<3, 1, FAD>&, const CORE::LINALG::Matrix<3, 1, FAD>&, FAD, int) const;

template void DRT::ELEMENTS::Beam3k::EvaluateStiffMatrixAnalyticFromPointNeumannMoment<2>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, double, int) const;

template void DRT::ELEMENTS::Beam3k::EvaluateLineNeumann<2>(CORE::LINALG::SerialDenseVector&,
    CORE::LINALG::SerialDenseMatrix*,
    const CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const CORE::LINALG::Matrix<6, 1, double>&, const std::vector<int>*, double) const;

template void DRT::ELEMENTS::Beam3k::EvaluateLineNeumannForces<2, double>(
    CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const CORE::LINALG::Matrix<6, 1, double>&, const std::vector<int>*, double) const;
template void DRT::ELEMENTS::Beam3k::EvaluateLineNeumannForces<2, FAD>(
    CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    const CORE::LINALG::Matrix<6, 1, double>&, const std::vector<int>*, double) const;

template void DRT::ELEMENTS::Beam3k::EvaluateTranslationalDamping<double, 2, 2, 3>(
    Teuchos::ParameterList&, const CORE::LINALG::Matrix<3 * 2 * 2, 1, double>&,
    const CORE::LINALG::Matrix<3 * 2 * 2, 1, double>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&);
template void DRT::ELEMENTS::Beam3k::EvaluateTranslationalDamping<FAD, 2, 2, 3>(
    Teuchos::ParameterList&, const CORE::LINALG::Matrix<3 * 2 * 2, 1, double>&,
    const CORE::LINALG::Matrix<3 * 2 * 2, 1, FAD>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&);

template void
DRT::ELEMENTS::Beam3k::EvaluateAnalyticStiffmatContributionsFromTranslationalDamping<2, 2, 3>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 3, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 3, double>&,
    const CORE::LINALG::Matrix<1, 2 * 2, double>&, const CORE::LINALG::Matrix<1, 2 * 2, double>&,
    double, double) const;
template void
DRT::ELEMENTS::Beam3k::EvaluateAnalyticStiffmatContributionsFromTranslationalDamping<2, 2, 3>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 3, FAD>&,
    const CORE::LINALG::Matrix<3, 1, FAD>&, const CORE::LINALG::Matrix<3, 1, FAD>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 3, FAD>&,
    const CORE::LINALG::Matrix<1, 2 * 2, double>&, const CORE::LINALG::Matrix<1, 2 * 2, double>&,
    double, double) const;

template void DRT::ELEMENTS::Beam3k::EvaluateStochasticForces<double, 2, 2, 3, 3>(
    const CORE::LINALG::Matrix<3 * 2 * 2, 1, double>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&);
template void DRT::ELEMENTS::Beam3k::EvaluateStochasticForces<FAD, 2, 2, 3, 3>(
    const CORE::LINALG::Matrix<3 * 2 * 2, 1, FAD>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&);

template void DRT::ELEMENTS::Beam3k::EvaluateAnalyticStiffmatContributionsFromStochasticForces<2, 2,
    3>(CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<1, 2 * 2, double>&, const CORE::LINALG::Matrix<1, 2 * 2, double>&,
    double, double) const;
template void DRT::ELEMENTS::Beam3k::EvaluateAnalyticStiffmatContributionsFromStochasticForces<2, 2,
    3>(CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, FAD>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<1, 2 * 2, double>&, const CORE::LINALG::Matrix<1, 2 * 2, double>&,
    double, double) const;

template void DRT::ELEMENTS::Beam3k::EvaluateRotationalDamping<double, 2, 2, 3>(
    const CORE::LINALG::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<3, 3, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&);
template void DRT::ELEMENTS::Beam3k::EvaluateRotationalDamping<FAD, 2, 2, 3>(
    const CORE::LINALG::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&,
    const std::vector<CORE::LINALG::Matrix<3, 3, FAD>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, FAD>&);

template void DRT::ELEMENTS::Beam3k::EvaluateAnalyticStiffmatContributionsFromRotationalDamping<2,
    2, 3>(CORE::LINALG::SerialDenseMatrix&,
    const CORE::LINALG::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS,
        double>&,
    const CORE::LINALG::Matrix<3, 1, double>, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&, const CORE::LINALG::Matrix<3, 3, double>&,
    const CORE::LINALG::Matrix<3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, 3, double>&,
    const std::vector<CORE::LINALG::Matrix<3, 3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, double>>&,
    const CORE::LINALG::Matrix<3, 1, double>, double, double, double, double) const;

template void
DRT::ELEMENTS::Beam3k::PreComputeTermsAtCPForAnalyticStiffmatContributionsFromRotationalDamping<2,
    2, 3>(CORE::LINALG::Matrix<3, 3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const CORE::LINALG::Matrix<1, 3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const CORE::LINALG::Matrix<3, 3 * 2 * 2 + BEAM3K_COLLOCATION_POINTS, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, double,
    const CORE::LINALG::Matrix<4, 1, double>&) const;

template void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsL<double, double>(
    CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, double>&,
    CORE::LINALG::Matrix<1, 2 * 6 + BEAM3K_COLLOCATION_POINTS, double>&) const;
template void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsL<Sacado::Fad::DFad<double>,
    Sacado::Fad::DFad<double>>(
    CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<1, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;
template void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsL<double, Sacado::Fad::DFad<double>>(
    CORE::LINALG::Matrix<1, BEAM3K_COLLOCATION_POINTS, double>&,
    CORE::LINALG::Matrix<1, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;

template void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsNss<double, double>(
    CORE::LINALG::Matrix<1, 4, double>&, CORE::LINALG::Matrix<1, 4, double>&, double, double,
    CORE::LINALG::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, double>&) const;
template void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsNss<Sacado::Fad::DFad<double>,
    Sacado::Fad::DFad<double>>(CORE::LINALG::Matrix<1, 4, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<1, 4, Sacado::Fad::DFad<double>>&, double, double,
    CORE::LINALG::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;
template void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsNss<double, Sacado::Fad::DFad<double>>(
    CORE::LINALG::Matrix<1, 4, double>&, CORE::LINALG::Matrix<1, 4, double>&, double, double,
    CORE::LINALG::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;

template void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsNs<double, double>(
    CORE::LINALG::Matrix<1, 4, double>&, double,
    CORE::LINALG::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, double>&) const;
template void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsNs<Sacado::Fad::DFad<double>,
    Sacado::Fad::DFad<double>>(CORE::LINALG::Matrix<1, 4, Sacado::Fad::DFad<double>>&, double,
    CORE::LINALG::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;
template void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsNs<double, Sacado::Fad::DFad<double>>(
    CORE::LINALG::Matrix<1, 4, double>&, double,
    CORE::LINALG::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;

template void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsN<double, double>(
    CORE::LINALG::Matrix<1, 4, double>&,
    CORE::LINALG::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, double>&) const;
template void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsN<Sacado::Fad::DFad<double>,
    Sacado::Fad::DFad<double>>(CORE::LINALG::Matrix<1, 4, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;
template void DRT::ELEMENTS::Beam3k::AssembleShapefunctionsN<double, Sacado::Fad::DFad<double>>(
    CORE::LINALG::Matrix<1, 4, double>&,
    CORE::LINALG::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, Sacado::Fad::DFad<double>>&) const;

template void DRT::ELEMENTS::Beam3k::ApplyRotVecTrafo<2, double>(
    const CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&,
    CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&) const;
template void DRT::ELEMENTS::Beam3k::ApplyRotVecTrafo<2, Sacado::Fad::DFad<double>>(
    const CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&) const;

template void DRT::ELEMENTS::Beam3k::TransformStiffMatrixMultipl<2, double>(
    CORE::LINALG::SerialDenseMatrix*,
    const CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, double>&) const;
template void DRT::ELEMENTS::Beam3k::TransformStiffMatrixMultipl<2, Sacado::Fad::DFad<double>>(
    CORE::LINALG::SerialDenseMatrix*,
    const CORE::LINALG::Matrix<6 * 2 + BEAM3K_COLLOCATION_POINTS, 1, Sacado::Fad::DFad<double>>&)
    const;

template void DRT::ELEMENTS::Beam3k::straintostress<double>(
    const CORE::LINALG::Matrix<3, 1, double>&, const double&,
    const CORE::LINALG::Matrix<3, 3, double>&, const CORE::LINALG::Matrix<3, 3, double>&,
    CORE::LINALG::Matrix<3, 1, double>&, double&) const;
template void DRT::ELEMENTS::Beam3k::straintostress<Sacado::Fad::DFad<double>>(
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&, const Sacado::Fad::DFad<double>&,
    const CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&, Sacado::Fad::DFad<double>&) const;

FOUR_C_NAMESPACE_CLOSE

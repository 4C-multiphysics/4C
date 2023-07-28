/*-----------------------------------------------------------------------------------------------*/
/*! \file

\brief evaluation methods for 3D nonlinear Reissner beam element

\level 2

*/
/*-----------------------------------------------------------------------------------------------*/

#include "baci_beam3_reissner.H"

#include "baci_beam3_triad_interpolation_local_rotation_vectors.H"
#include "baci_beam3_spatial_discretization_utils.H"

#include "baci_lib_discret.H"
#include "baci_lib_globalproblem.H"
#include "baci_lib_exporter.H"
#include "baci_utils_exceptions.H"
#include "baci_lib_utils.H"
#include "baci_mat_beam_elasthyper.H"
#include "baci_linalg_utils_sparse_algebra_math.H"
#include "baci_discretization_fem_general_utils_fem_shapefunctions.H"
#include "baci_linalg_fixedsizematrix.H"
#include "baci_discretization_fem_general_largerotations.H"
#include "baci_utils_fad.H"
#include "baci_structure_new_elements_paramsinterface.H"
#include "baci_structure_new_model_evaluator_data.H"
#include "baci_structure_new_enum_lists.H"

#include <iostream>
#include <iomanip>

#include <Teuchos_RCP.hpp>

/*-----------------------------------------------------------------------------------------------------------*
 |  evaluate the element (public) cyron 01/08|
 *----------------------------------------------------------------------------------------------------------*/
int DRT::ELEMENTS::Beam3r::Evaluate(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, std::vector<int>& lm,
    CORE::LINALG::SerialDenseMatrix& elemat1,  // nonlinear stiffness matrix
    CORE::LINALG::SerialDenseMatrix& elemat2,  // nonlinear mass matrix
    CORE::LINALG::SerialDenseVector& elevec1,  // nonlinear internal (elastic) forces
    CORE::LINALG::SerialDenseVector& elevec2,  // nonlinear inertia forces
    CORE::LINALG::SerialDenseVector& elevec3)
{
  // Set structure params interface pointer
  SetParamsInterfacePtr(params);
  // Set brwonian params interface pointer
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
      dserror("Unknown type of action for Beam3r");
  }

  // nnodetriad: number of nodes used for interpolation of triad field
  const int nnodetriad = NumNode();

  switch (act)
  {
    case ELEMENTS::struct_calc_ptcstiff:
    {
      switch (nnodetriad)
      {
        case 2:
          EvaluatePTC<2>(params, elemat1);
          break;
        case 3:
          EvaluatePTC<3>(params, elemat1);
          break;
        case 4:
          EvaluatePTC<4>(params, elemat1);
          break;
        case 5:
          EvaluatePTC<5>(params, elemat1);
          break;
        default:
          dserror("Only Line2, Line3, Line4 and Line5 Elements implemented.");
      }
      break;
    }

    case ELEMENTS::struct_calc_linstiff:
    {
      // only nonlinear case implemented!
      dserror("linear stiffness matrix called, but not implemented");
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

    case ELEMENTS::struct_calc_nlnstiffmass:
    case ELEMENTS::struct_calc_nlnstifflmass:
    case ELEMENTS::struct_calc_nlnstiff:
    case ELEMENTS::struct_calc_internalforce:
    case ELEMENTS::struct_calc_internalinertiaforce:
    {
      // need current global displacement and residual forces and get them from discretization
      // making use of the local-to-global map lm one can extract current displacement and residual
      // values for each degree of freedom

      // get element displacements
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
      if (disp == Teuchos::null) dserror("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp, mydisp, lm);

      if (act == ELEMENTS::struct_calc_nlnstiffmass)
      {
        switch (nnodetriad)
        {
          case 2:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<2, 2, 1>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            else
              CalcInternalAndInertiaForcesAndStiff<2, 2, 2>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            break;
          }
          case 3:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<3, 3, 1>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            else
              CalcInternalAndInertiaForcesAndStiff<3, 2, 2>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            break;
          }
          case 4:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<4, 4, 1>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            else
              CalcInternalAndInertiaForcesAndStiff<4, 2, 2>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            break;
          }
          case 5:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<5, 5, 1>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            else
              CalcInternalAndInertiaForcesAndStiff<5, 2, 2>(
                  params, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
            break;
          }
        }
      }

      else if (act == ELEMENTS::struct_calc_nlnstifflmass)
      {
        // TODO there is a method 'Beam3r::lumpmass'; check generality and functionality and enable
        // action here
        dserror("Lumped mass matrix not implemented for beam3r elements so far!");
      }

      else if (act == ELEMENTS::struct_calc_nlnstiff)
      {
        switch (nnodetriad)
        {
          case 2:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<2, 2, 1>(
                  params, mydisp, &elemat1, NULL, &elevec1, NULL);
            else
              CalcInternalAndInertiaForcesAndStiff<2, 2, 2>(
                  params, mydisp, &elemat1, NULL, &elevec1, NULL);
            break;
          }
          case 3:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<3, 3, 1>(
                  params, mydisp, &elemat1, NULL, &elevec1, NULL);
            else
              CalcInternalAndInertiaForcesAndStiff<3, 2, 2>(
                  params, mydisp, &elemat1, NULL, &elevec1, NULL);
            break;
          }
          case 4:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<4, 4, 1>(
                  params, mydisp, &elemat1, NULL, &elevec1, NULL);
            else
              CalcInternalAndInertiaForcesAndStiff<4, 2, 2>(
                  params, mydisp, &elemat1, NULL, &elevec1, NULL);
            break;
          }
          case 5:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<5, 5, 1>(
                  params, mydisp, &elemat1, NULL, &elevec1, NULL);
            else
              CalcInternalAndInertiaForcesAndStiff<5, 2, 2>(
                  params, mydisp, &elemat1, NULL, &elevec1, NULL);
            break;
          }
          default:
            dserror("Only Line2, Line3, Line4, and Line5 Elements implemented.");
        }
      }
      else if (act == ELEMENTS::struct_calc_internalforce)
      {
        switch (nnodetriad)
        {
          case 2:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<2, 2, 1>(
                  params, mydisp, NULL, NULL, &elevec1, NULL);
            else
              CalcInternalAndInertiaForcesAndStiff<2, 2, 2>(
                  params, mydisp, NULL, NULL, &elevec1, NULL);
            break;
          }
          case 3:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<3, 3, 1>(
                  params, mydisp, NULL, NULL, &elevec1, NULL);
            else
              CalcInternalAndInertiaForcesAndStiff<3, 2, 2>(
                  params, mydisp, NULL, NULL, &elevec1, NULL);
            break;
          }
          case 4:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<4, 4, 1>(
                  params, mydisp, NULL, NULL, &elevec1, NULL);
            else
              CalcInternalAndInertiaForcesAndStiff<4, 2, 2>(
                  params, mydisp, NULL, NULL, &elevec1, NULL);
            break;
          }
          case 5:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<5, 5, 1>(
                  params, mydisp, NULL, NULL, &elevec1, NULL);
            else
              CalcInternalAndInertiaForcesAndStiff<5, 2, 2>(
                  params, mydisp, NULL, NULL, &elevec1, NULL);
            break;
          }
          default:
            dserror("Only Line2, Line3, Line4, and Line5 Elements implemented.");
        }
      }

      else if (act == ELEMENTS::struct_calc_internalinertiaforce)
      {
        switch (nnodetriad)
        {
          case 2:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<2, 2, 1>(
                  params, mydisp, NULL, NULL, &elevec1, &elevec2);
            else
              CalcInternalAndInertiaForcesAndStiff<2, 2, 2>(
                  params, mydisp, NULL, NULL, &elevec1, &elevec2);
            break;
          }
          case 3:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<3, 3, 1>(
                  params, mydisp, NULL, NULL, &elevec1, &elevec2);
            else
              CalcInternalAndInertiaForcesAndStiff<3, 2, 2>(
                  params, mydisp, NULL, NULL, &elevec1, &elevec2);
            break;
          }
          case 4:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<4, 4, 1>(
                  params, mydisp, NULL, NULL, &elevec1, &elevec2);
            else
              CalcInternalAndInertiaForcesAndStiff<4, 2, 2>(
                  params, mydisp, NULL, NULL, &elevec1, &elevec2);
            break;
          }
          case 5:
          {
            if (!centerline_hermite_)
              CalcInternalAndInertiaForcesAndStiff<5, 5, 1>(
                  params, mydisp, NULL, NULL, &elevec1, &elevec2);
            else
              CalcInternalAndInertiaForcesAndStiff<5, 2, 2>(
                  params, mydisp, NULL, NULL, &elevec1, &elevec2);
            break;
          }
          default:
            dserror("Only Line2, Line3, Line4, and Line5 Elements implemented.");
        }
      }

      break;
    }

    case ELEMENTS::struct_calc_update_istep:
    {
      /* the action calc_struct_update_istep is called in the very end of a time step when the new
       * dynamic equilibrium has finally been found; this is the point where the variable
       * representing the geometric status of the beam at the end of the time step has to be
       * stored*/
      Qconvnode_ = Qnewnode_;
      QconvGPmass_ = QnewGPmass_;
      wconvGPmass_ = wnewGPmass_;
      aconvGPmass_ = anewGPmass_;
      amodconvGPmass_ = amodnewGPmass_;
      rttconvGPmass_ = rttnewGPmass_;
      rttmodconvGPmass_ = rttmodnewGPmass_;
      rtconvGPmass_ = rtnewGPmass_;
      rconvGPmass_ = rnewGPmass_;
      QconvGPdampstoch_ = QnewGPdampstoch_;
      GetBeamMaterial().Update();
      break;
    }

    case ELEMENTS::struct_calc_reset_istep:
    {
      /* the action calc_struct_reset_istep is called by the adaptive time step controller; carries
       * out one test step whose purpose is only figuring out a suitable time step; thus this step
       * may be a very bad one in order to iterated towards the new dynamic equilibrium and the
       * thereby gained new geometric configuration should not be applied as starting point for any
       * further iteration step; as a consequence the thereby generated change of the geometric
       * configuration should be canceled and the configuration should be reset to the value at the
       * beginning of the time step*/
      Qnewnode_ = Qconvnode_;
      QnewGPmass_ = QconvGPmass_;
      wnewGPmass_ = wconvGPmass_;
      anewGPmass_ = aconvGPmass_;
      amodnewGPmass_ = amodconvGPmass_;
      rttnewGPmass_ = rttconvGPmass_;
      rttmodnewGPmass_ = rttmodconvGPmass_;
      rtnewGPmass_ = rtconvGPmass_;
      rnewGPmass_ = rconvGPmass_;
      QnewGPdampstoch_ = QconvGPdampstoch_;
      GetBeamMaterial().Reset();
      break;
    }

    case ELEMENTS::struct_calc_brownianforce:
    case ELEMENTS::struct_calc_brownianstiff:
    {
      // get element displacements
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
      if (disp == Teuchos::null) dserror("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp, mydisp, lm);

      // get element velocity
      Teuchos::RCP<const Epetra_Vector> vel = discretization.GetState("velocity");
      if (vel == Teuchos::null) dserror("Cannot get state vectors 'velocity'");
      std::vector<double> myvel(lm.size());
      DRT::UTILS::ExtractMyValues(*vel, myvel, lm);

      if (act == ELEMENTS::struct_calc_brownianforce)
      {
        switch (nnodetriad)
        {
          case 2:
          {
            if (!centerline_hermite_)
              CalcBrownianForcesAndStiff<2, 2, 1>(params, myvel, mydisp, NULL, &elevec1);
            else
              CalcBrownianForcesAndStiff<2, 2, 2>(params, myvel, mydisp, NULL, &elevec1);
            break;
          }
          case 3:
          {
            if (!centerline_hermite_)
              CalcBrownianForcesAndStiff<3, 3, 1>(params, myvel, mydisp, NULL, &elevec1);
            else
              CalcBrownianForcesAndStiff<3, 2, 2>(params, myvel, mydisp, NULL, &elevec1);
            break;
          }
          case 4:
          {
            if (!centerline_hermite_)
              CalcBrownianForcesAndStiff<4, 4, 1>(params, myvel, mydisp, NULL, &elevec1);
            else
              CalcBrownianForcesAndStiff<4, 2, 2>(params, myvel, mydisp, NULL, &elevec1);
            break;
          }
          case 5:
          {
            if (!centerline_hermite_)
              CalcBrownianForcesAndStiff<5, 5, 1>(params, myvel, mydisp, NULL, &elevec1);
            else
              CalcBrownianForcesAndStiff<5, 2, 2>(params, myvel, mydisp, NULL, &elevec1);
            break;
          }
          default:
            dserror("Only Line2, Line3, Line4, and Line5 Elements implemented.");
        }
      }
      else if (act == ELEMENTS::struct_calc_brownianstiff)
      {
        switch (nnodetriad)
        {
          case 2:
          {
            if (!centerline_hermite_)
              CalcBrownianForcesAndStiff<2, 2, 1>(params, myvel, mydisp, &elemat1, &elevec1);
            else
              CalcBrownianForcesAndStiff<2, 2, 2>(params, myvel, mydisp, &elemat1, &elevec1);
            break;
          }
          case 3:
          {
            if (!centerline_hermite_)
              CalcBrownianForcesAndStiff<3, 3, 1>(params, myvel, mydisp, &elemat1, &elevec1);
            else
              CalcBrownianForcesAndStiff<3, 2, 2>(params, myvel, mydisp, &elemat1, &elevec1);
            break;
          }
          case 4:
          {
            if (!centerline_hermite_)
              CalcBrownianForcesAndStiff<4, 4, 1>(params, myvel, mydisp, &elemat1, &elevec1);
            else
              CalcBrownianForcesAndStiff<4, 2, 2>(params, myvel, mydisp, &elemat1, &elevec1);
            break;
          }
          case 5:
          {
            if (!centerline_hermite_)
              CalcBrownianForcesAndStiff<5, 5, 1>(params, myvel, mydisp, &elemat1, &elevec1);
            else
              CalcBrownianForcesAndStiff<5, 2, 2>(params, myvel, mydisp, &elemat1, &elevec1);
            break;
          }
          default:
            dserror("Only Line2, Line3, Line4, and Line5 Elements implemented.");
        }
      }
      else
        dserror("You shouldn't be here.");

      break;
    }

    // write stress and strain output
    case ELEMENTS::struct_calc_stress:
    {
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
      std::cout << "\ncalled element with action type " << ActionType2String(act);
      dserror("This action type is not implemented for Beam3r");
      break;
  }
  return 0;
}

/*-----------------------------------------------------------------------------------------------------------*
 |  Integrate a Surface Neumann boundary condition (public) cyron 03/08|
 *----------------------------------------------------------------------------------------------------------*/
int DRT::ELEMENTS::Beam3r::EvaluateNeumann(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Condition& condition, std::vector<int>& lm,
    CORE::LINALG::SerialDenseVector& elevec1, CORE::LINALG::SerialDenseMatrix* elemat1)
{
  SetParamsInterfacePtr(params);

  // find out whether we will use a time curve
  double time = -1.0;

  if (IsParamsInterface())
    time = ParamsInterface().GetTotalTime();
  else
    time = params.get<double>("total time", -1.0);

  // nnodetriad: number of nodes used for interpolation of triad field
  const unsigned int nnodetriad = NumNode();
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

  const DiscretizationType distype = this->Shape();

  // gaussian points
  const CORE::DRT::UTILS::IntegrationPoints1D intpoints(MyGaussRule(neumann_lineload));

  // declaration of variables in order to store shape functions
  // used for interpolation of triad field
  CORE::LINALG::SerialDenseVector I_i(nnodetriad);
  // used for interpolation of centerline
  CORE::LINALG::SerialDenseVector H_i(vpernode * nnodecl);

  // get values and switches from the condition

  // onoff is related to the first numdf flags of a line Neumann condition in the input file;
  // value 1 for flag i says that condition is active for i-th degree of freedom
  const std::vector<int>* onoff = condition.Get<std::vector<int>>("onoff");
  // val is related to the numdf "val" fields after the onoff flags of the Neumann condition
  // in the input file; val gives the values of the force as a multiple of the prescribed load curve
  const std::vector<double>* val = condition.Get<std::vector<double>>("val");
  // funct is related to the numdf "funct" fields after the val field of the Neumann condition
  // in the input file; funct gives the number of the function defined in the section FUNCT
  const std::vector<int>* functions = condition.Get<std::vector<int>>("funct");

  // integration points in parameter space and weights
  double xi = 0.0;
  double wgt = 0.0;

  // integration loops
  for (int numgp = 0; numgp < intpoints.nquad; ++numgp)
  {
    xi = intpoints.qxg[numgp][0];
    wgt = intpoints.qwgt[numgp];

    // evaluation of shape functions at Gauss points
    CORE::DRT::UTILS::shape_function_1D(I_i, xi, distype);
    if (centerline_hermite_)
      CORE::DRT::UTILS::shape_function_hermite_1D(H_i, xi, reflength_, line2);
    else
      CORE::DRT::UTILS::shape_function_1D(H_i, xi, distype);

    // position vector at the gauss point at reference configuration needed for function evaluation
    std::vector<double> X_ref(3, 0.0);

    // calculate coordinates of corresponding Gauss point in reference configuration
    for (unsigned int node = 0; node < nnodecl; node++)
    {
      for (unsigned int dim = 0; dim < 3; dim++)
      {
        X_ref[dim] += H_i[vpernode * node] * Nodes()[node]->X()[dim];

        if (centerline_hermite_) X_ref[dim] += H_i[vpernode * node + 1] * (Tref_[node])(dim);
      }
    }

    double fac = 0;
    fac = wgt * jacobiGPneumannline_[numgp];

    // load vector ar
    double ar[6];

    // loop the relevant dofs of a node
    for (int dof = 0; dof < 6; ++dof) ar[dof] = fac * (*onoff)[dof] * (*val)[dof];
    double functionfac = 1.0;
    int functnum = -1;

    // sum up load components
    for (unsigned int dof = 0; dof < 6; ++dof)
    {
      if (functions)
        functnum = (*functions)[dof];
      else
        functnum = -1;

      // evaluate function at the position of the current GP
      if (functnum > 0)
        functionfac = DRT::Problem::Instance()
                          ->FunctionById<DRT::UTILS::FunctionOfSpaceTime>(functnum - 1)
                          .Evaluate(X_ref.data(), time, dof);
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
inline void DRT::ELEMENTS::Beam3r::pushforward(const CORE::LINALG::Matrix<3, 3, T>& Lambda,
    const CORE::LINALG::Matrix<3, 1, T>& stress_mat, const CORE::LINALG::Matrix<3, 3, T>& C_mat,
    CORE::LINALG::Matrix<3, 1, T>& stress_spatial, CORE::LINALG::Matrix<3, 3, T>& c_spatial) const
{
  // introduce auxiliary variable for pushforward of rotational matrices
  CORE::LINALG::Matrix<3, 3, T> temp;

  // push forward stress vector
  stress_spatial.Multiply(Lambda, stress_mat);

  // push forward constitutive matrix according to Jelenic 1999, paragraph following to (2.22) on
  // page 148
  temp.Multiply(Lambda, C_mat);
  c_spatial.MultiplyNT(temp, Lambda);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff(Teuchos::ParameterList& params,
    std::vector<double>& disp, CORE::LINALG::SerialDenseMatrix* stiffmatrix,
    CORE::LINALG::SerialDenseMatrix* massmatrix, CORE::LINALG::SerialDenseVector* force,
    CORE::LINALG::SerialDenseVector* inertia_force)
{
  //************ periodic boundary conditions **********************
  /* unshift node positions, i.e. manipulate element displacement vector
   * as if there where no periodic boundary conditions */
  if (BrownianDynParamsInterfacePtr() != Teuchos::null)
    UnShiftNodePosition(disp, *BrownianDynParamsInterface().GetPeriodicBoundingBox());

  /* current nodal DOFs relevant for centerline interpolation in total Lagrangian
   * style, i.e. initial values + displacements */
  CORE::LINALG::Matrix<3 * vpernode * nnodecl, 1, double> disp_totlag_centerline(true);

  // quaternions of all nodal triads
  std::vector<CORE::LINALG::Matrix<4, 1, double>> Qnode(nnodetriad);

  UpdateDispTotLagAndNodalTriads<nnodetriad, nnodecl, vpernode, double>(
      disp, disp_totlag_centerline, Qnode);

  CalcInternalAndInertiaForcesAndStiff<nnodetriad, nnodecl, vpernode>(
      disp_totlag_centerline, Qnode, stiffmatrix, massmatrix, force, inertia_force);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff(
    CORE::LINALG::Matrix<3 * vpernode * nnodecl, 1, double>& disp_totlag_centerline,
    std::vector<CORE::LINALG::Matrix<4, 1, double>>& Qnode,
    CORE::LINALG::SerialDenseMatrix* stiffmatrix, CORE::LINALG::SerialDenseMatrix* massmatrix,
    CORE::LINALG::SerialDenseVector* force, CORE::LINALG::SerialDenseVector* inertia_force)
{
  const unsigned int numdofelement = 3 * vpernode * nnodecl + 3 * nnodetriad;

  if (not useFAD_)
  {
    // internal force vector
    CORE::LINALG::Matrix<numdofelement, 1, double> internal_force(true);

    if (force != NULL)
    {
      internal_force.SetView(&((*force)(0)));
    }

    CalcInternalForceAndStiff<nnodetriad, nnodecl, vpernode, double>(
        disp_totlag_centerline, Qnode, stiffmatrix, internal_force);

    // calculation of inertia forces/moments and mass matrix
    if (massmatrix != NULL or inertia_force != NULL)
    {
      CalcInertiaForceAndMassMatrix<nnodetriad, nnodecl, vpernode>(
          disp_totlag_centerline, Qnode, massmatrix, inertia_force);
    }
  }
  else
  {
    // internal force vector
    CORE::LINALG::Matrix<numdofelement, 1, Sacado::Fad::DFad<double>> internal_force(true);

    /* current nodal DOFs relevant for centerline interpolation in total Lagrangian
     * style, i.e. initial values + displacements */
    CORE::LINALG::Matrix<3 * vpernode * nnodecl, 1, Sacado::Fad::DFad<double>>
        disp_totlag_centerline_FAD;

    for (unsigned int i = 0; i < 3 * vpernode * nnodecl; ++i)
      disp_totlag_centerline_FAD(i) = disp_totlag_centerline(i);

    // quaternions of all nodal triads
    std::vector<CORE::LINALG::Matrix<4, 1, Sacado::Fad::DFad<double>>> Qnode_FAD(nnodetriad);

    for (unsigned int inode = 0; inode < nnodetriad; ++inode)
      for (unsigned int j = 0; j < 4; ++j) Qnode_FAD[inode](j) = Qnode[inode](j);

    SetAutomaticDifferentiationVariables<nnodetriad, nnodecl, vpernode>(
        disp_totlag_centerline_FAD, Qnode_FAD);

    CalcInternalForceAndStiff<nnodetriad, nnodecl, vpernode, Sacado::Fad::DFad<double>>(
        disp_totlag_centerline_FAD, Qnode_FAD, NULL, internal_force);

    if (force != NULL)
    {
      for (unsigned int idof = 0; idof < numdofelement; ++idof)
        (*force)(idof) = CORE::FADUTILS::CastToDouble(internal_force(idof));
    }

    if (stiffmatrix != NULL)
    {
      CalcStiffmatAutomaticDifferentiation<nnodetriad, nnodecl, vpernode>(
          *stiffmatrix, Qnode, internal_force);
    }

    // calculation of inertia forces/moments and mass matrix
    if (massmatrix != NULL or inertia_force != NULL)
    {
      CalcInertiaForceAndMassMatrix<nnodetriad, nnodecl, vpernode>(
          disp_totlag_centerline, Qnode, massmatrix, inertia_force);
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode, typename T>
void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff(
    const CORE::LINALG::Matrix<3 * vpernode * nnodecl, 1, T>& disp_totlag_centerline,
    const std::vector<CORE::LINALG::Matrix<4, 1, T>>& Qnode,
    CORE::LINALG::SerialDenseMatrix* stiffmatrix,
    CORE::LINALG::Matrix<3 * vpernode * nnodecl + 3 * nnodetriad, 1, T>& internal_force)
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
  Eint_ = 0.0;

  //*************************** physical quantities evaluated at a certain GP
  //***************************

  // derivation of beam centerline with respect to arc-length parameter: r'(x) from (2.12), Jelenic
  // 1999
  CORE::LINALG::Matrix<3, 1, T> r_s;
  // spin matrix related to vector r_s
  CORE::LINALG::Matrix<3, 3, T> r_s_hat;
  // interpolated local relative rotation \Psi^l at a certain Gauss point according to (3.11),
  // Jelenic 1999
  CORE::LINALG::Matrix<3, 1, T> Psi_l;
  /* derivative of interpolated local relative rotation \Psi^l with respect to arc-length parameter
   * at a certain Gauss point according to (3.11), Jelenic 1999*/
  CORE::LINALG::Matrix<3, 1, T> Psi_l_s;
  // triad at GP
  CORE::LINALG::Matrix<3, 3, T> Lambda;

  // 3D vector related to spin matrix \hat{\kappa} from (2.1), Jelenic 1999
  CORE::LINALG::Matrix<3, 1, T> Cur;
  // 3D vector of material axial and shear strains from (2.1), Jelenic 1999
  CORE::LINALG::Matrix<3, 1, T> Gamma;

  // convected stresses N and M and constitutive matrices C_N and C_M according to section 2.4,
  // Jelenic 1999
  CORE::LINALG::Matrix<3, 1, T> stressN;
  CORE::LINALG::Matrix<3, 1, T> stressM;
  CORE::LINALG::Matrix<3, 3, T> CN;
  CORE::LINALG::Matrix<3, 3, T> CM;

  // spatial stresses n and m according to (3.10), Romero 2004 and spatial constitutive matrices c_n
  // and c_m according to page 148, Jelenic 1999
  CORE::LINALG::Matrix<3, 1, T> stressn;
  CORE::LINALG::Matrix<3, 1, T> stressm;
  CORE::LINALG::Matrix<3, 3, T> cn;
  CORE::LINALG::Matrix<3, 3, T> cm;

  //********************************** (generalized) shape functions
  //************************************
  /* Note: index i refers to the i-th shape function (i = 0 ... nnode*vpernode-1)
   * the vectors store individual shape functions, NOT an assembled matrix of shape functions)*/

  /* vector whose numgp-th element is a 1xnnode-matrix with all Lagrange polynomial shape functions
   * evaluated at the numgp-th Gauss point these shape functions are used for the interpolation of
   * the triad field*/
  std::vector<CORE::LINALG::Matrix<1, nnodetriad, double>> I_i;
  // same for the derivatives
  std::vector<CORE::LINALG::Matrix<1, nnodetriad, double>> I_i_xi;

  /* vector whose numgp-th element is a 1x(vpernode*nnode)-matrix with all (Lagrange/Hermite) shape
   * functions evaluated at the numgp-th GP
   * these shape functions are used for the interpolation of the beam centerline*/
  std::vector<CORE::LINALG::Matrix<1, vpernode * nnodecl, double>> H_i;
  // same for the derivatives
  std::vector<CORE::LINALG::Matrix<1, vpernode * nnodecl, double>> H_i_xi;


  /*************************** update/compute quantities valid for entire element
   ***********************
   *****************************************************************************************************/
  // setup constitutive matrices
  GetTemplatedBeamMaterial<T>().ComputeConstitutiveParameter(CN, CM);

  // create object of triad interpolation scheme
  Teuchos::RCP<LARGEROTATIONS::TriadInterpolationLocalRotationVectors<nnodetriad, T>>
      triad_interpolation_scheme_ptr =
          Teuchos::rcp(new LARGEROTATIONS::TriadInterpolationLocalRotationVectors<nnodetriad, T>());

  // reset triad interpolation scheme based on nodal quaternions
  triad_interpolation_scheme_ptr->Reset(Qnode);

  // matrix containing contributions to the jacobian depending on the material model
  CORE::LINALG::Matrix<3, 3, T> stiffness_contribution(true);

  /******************************* elasticity: compute fint and stiffmatrix
   *****************************
   *****************************************************************************************************/

  //************************* residual and stiffmatrix contributions from forces
  //***********************
  // for these contributions, reduced integration is applied to avoid locking

  // get integration points for elasticity
  CORE::DRT::UTILS::IntegrationPoints1D gausspoints_elast_force(MyGaussRule(res_elastic_force));

  // reuse variables for individual shape functions and resize to new numgp
  I_i.resize(gausspoints_elast_force.nquad);
  H_i_xi.resize(gausspoints_elast_force.nquad);

  // evaluate all shape functions and derivatives with respect to element parameter xi at all
  // specified Gauss points
  DRT::UTILS::BEAM::EvaluateShapeFunctionsAllGPs<nnodetriad, 1>(
      gausspoints_elast_force, I_i, this->Shape());

  DRT::UTILS::BEAM::EvaluateShapeFunctionDerivsAllGPs<nnodecl, vpernode>(
      gausspoints_elast_force, H_i_xi, this->Shape(), this->RefLength());

  // re-assure correct size of strain and stress resultant class variables
  axial_strain_GP_elastf_.resize(gausspoints_elast_force.nquad);
  shear_strain_2_GP_elastf_.resize(gausspoints_elast_force.nquad);
  shear_strain_3_GP_elastf_.resize(gausspoints_elast_force.nquad);

  material_axial_force_GP_elastf_.resize(gausspoints_elast_force.nquad);
  material_shear_force_2_GP_elastf_.resize(gausspoints_elast_force.nquad);
  material_shear_force_3_GP_elastf_.resize(gausspoints_elast_force.nquad);

  spatial_x_force_GP_elastf_.resize(gausspoints_elast_force.nquad);
  spatial_y_force_2_GP_elastf_.resize(gausspoints_elast_force.nquad);
  spatial_z_force_3_GP_elastf_.resize(gausspoints_elast_force.nquad);

  // Loop through all GP and calculate their contribution to the forcevector and stiffnessmatrix
  for (int numgp = 0; numgp < gausspoints_elast_force.nquad; ++numgp)
  {
    // weight of GP in parameter space
    const double wgt = gausspoints_elast_force.qwgt[numgp];

    Calc_r_s<nnodecl, vpernode, T>(
        disp_totlag_centerline, H_i_xi[numgp], jacobiGPelastf_[numgp], r_s);

    triad_interpolation_scheme_ptr->GetInterpolatedTriadAtXi(
        Lambda, gausspoints_elast_force.qxg[numgp][0]);

    // compute spin matrix related to vector rprime for later use
    CORE::LARGEROTATIONS::computespin<T>(r_s_hat, r_s);

    // compute material strains Gamma and Cur
    computeGamma<T>(r_s, Lambda, GammarefGP_[numgp], Gamma);

    GetTemplatedBeamMaterial<T>().EvaluateForceContributionsToStress(stressN, CN, Gamma, numgp);
    GetTemplatedBeamMaterial<T>().GetStiffnessMatrixOfForces(stiffness_contribution, CN, numgp);

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
              r_s_hat(i, j) * stressn(j) * I_i[numgp](node) * wgt * jacobiGPelastf_[numgp];
    }
    for (unsigned int node = nnodecl; node < nnodetriad;
         ++node)  // this loop is only entered in case of nnodetriad>nnodecl
    {
      // lower left block
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          internal_force(dofperclnode * nnodecl + dofpertriadnode * node + i) -=
              r_s_hat(i, j) * stressn(j) * I_i[numgp](node) * wgt * jacobiGPelastf_[numgp];
    }

    if (stiffmatrix != NULL)
    {
      CalcStiffmatAnalyticForceContributions<nnodetriad, nnodecl, vpernode>(*stiffmatrix, stressn,
          cn, r_s_hat, *triad_interpolation_scheme_ptr, I_i[numgp], H_i_xi[numgp], wgt,
          jacobiGPelastf_[numgp]);
    }

    // add elastic energy from forces at this GP
    for (unsigned int dim = 0; dim < 3; ++dim)
    {
      Eint_ += 0.5 * CORE::FADUTILS::CastToDouble(Gamma(dim)) *
               CORE::FADUTILS::CastToDouble(stressN(dim)) * jacobiGPelastf_[numgp] * wgt;
    }

    // store material strain and stress values in class variables
    axial_strain_GP_elastf_[numgp] = CORE::FADUTILS::CastToDouble(Gamma(0));
    shear_strain_2_GP_elastf_[numgp] = CORE::FADUTILS::CastToDouble(Gamma(1));
    shear_strain_3_GP_elastf_[numgp] = CORE::FADUTILS::CastToDouble(Gamma(2));

    material_axial_force_GP_elastf_[numgp] = CORE::FADUTILS::CastToDouble(stressN(0));
    material_shear_force_2_GP_elastf_[numgp] = CORE::FADUTILS::CastToDouble(stressN(1));
    material_shear_force_3_GP_elastf_[numgp] = CORE::FADUTILS::CastToDouble(stressN(2));

    spatial_x_force_GP_elastf_[numgp] = CORE::FADUTILS::CastToDouble(stressn(0));
    spatial_y_force_2_GP_elastf_[numgp] = CORE::FADUTILS::CastToDouble(stressn(1));
    spatial_z_force_3_GP_elastf_[numgp] = CORE::FADUTILS::CastToDouble(stressn(2));
  }


  //************************* residual and stiffmatrix contributions from moments
  //***********************

  // get integration points for elasticity
  CORE::DRT::UTILS::IntegrationPoints1D gausspoints_elast_moment(MyGaussRule(res_elastic_moment));

  // reuse variables for individual shape functions and resize to new numgp
  I_i.resize(gausspoints_elast_moment.nquad);
  I_i_xi.resize(gausspoints_elast_moment.nquad);

  // evaluate all shape functions and derivatives with respect to element parameter xi at all
  // specified Gauss points
  DRT::UTILS::BEAM::EvaluateShapeFunctionsAndDerivsAllGPs<nnodetriad, 1>(
      gausspoints_elast_moment, I_i, I_i_xi, this->Shape());

  // reset norm of maximal bending curvature
  Kmax_ = 0.0;

  // assure correct size of strain and stress resultant class variables
  twist_GP_elastm_.resize(gausspoints_elast_moment.nquad);
  curvature_2_GP_elastm_.resize(gausspoints_elast_moment.nquad);
  curvature_3_GP_elastm_.resize(gausspoints_elast_moment.nquad);

  material_torque_GP_elastm_.resize(gausspoints_elast_moment.nquad);
  material_bending_moment_2_GP_elastm_.resize(gausspoints_elast_moment.nquad);
  material_bending_moment_3_GP_elastm_.resize(gausspoints_elast_moment.nquad);

  spatial_x_moment_GP_elastm_.resize(gausspoints_elast_moment.nquad);
  spatial_y_moment_2_GP_elastm_.resize(gausspoints_elast_moment.nquad);
  spatial_z_moment_3_GP_elastm_.resize(gausspoints_elast_moment.nquad);


  // Loop through all GP and calculate their contribution to the forcevector and stiffnessmatrix
  for (int numgp = 0; numgp < gausspoints_elast_moment.nquad; numgp++)
  {
    // weight of GP in parameter space
    const double wgt = gausspoints_elast_moment.qwgt[numgp];

    triad_interpolation_scheme_ptr->GetInterpolatedLocalRotationVector(Psi_l, I_i[numgp]);

    triad_interpolation_scheme_ptr->GetInterpolatedLocalRotationVectorDerivative(
        Psi_l_s, I_i_xi[numgp], jacobiGPelastm_[numgp]);

    triad_interpolation_scheme_ptr->GetInterpolatedTriad(Lambda, Psi_l);

    // compute material curvature Cur
    computeK<T>(Psi_l, Psi_l_s, KrefGP_[numgp], Cur);

    // determine norm of maximal bending curvature at this GP and store in class variable if needed
    double Kmax =
        std::sqrt(CORE::FADUTILS::CastToDouble(Cur(1)) * CORE::FADUTILS::CastToDouble(Cur(1)) +
                  CORE::FADUTILS::CastToDouble(Cur(2)) * CORE::FADUTILS::CastToDouble(Cur(2)));
    if (Kmax > Kmax_) Kmax_ = Kmax;

    GetTemplatedBeamMaterial<T>().EvaluateMomentContributionsToStress(stressM, CM, Cur, numgp);
    GetTemplatedBeamMaterial<T>().GetStiffnessMatrixOfMoments(stiffness_contribution, CM, numgp);

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


    if (stiffmatrix != NULL)
    {
      CalcStiffmatAnalyticMomentContributions<nnodetriad, nnodecl, vpernode>(*stiffmatrix, stressm,
          cm, *triad_interpolation_scheme_ptr, Psi_l, Psi_l_s, I_i[numgp], I_i_xi[numgp], wgt,
          jacobiGPelastm_[numgp]);
    }

    // add elastic energy from moments at this GP
    for (unsigned int dim = 0; dim < 3; dim++)
    {
      Eint_ += 0.5 * CORE::FADUTILS::CastToDouble(Cur(dim)) *
               CORE::FADUTILS::CastToDouble(stressM(dim)) * jacobiGPelastm_[numgp] * wgt;
    }

    // store material strain and stress values in class variables
    twist_GP_elastm_[numgp] = CORE::FADUTILS::CastToDouble(Cur(0));
    curvature_2_GP_elastm_[numgp] = CORE::FADUTILS::CastToDouble(Cur(1));
    curvature_3_GP_elastm_[numgp] = CORE::FADUTILS::CastToDouble(Cur(2));

    material_torque_GP_elastm_[numgp] = CORE::FADUTILS::CastToDouble(stressM(0));
    material_bending_moment_2_GP_elastm_[numgp] = CORE::FADUTILS::CastToDouble(stressM(1));
    material_bending_moment_3_GP_elastm_[numgp] = CORE::FADUTILS::CastToDouble(stressM(2));

    spatial_x_moment_GP_elastm_[numgp] = CORE::FADUTILS::CastToDouble(stressm(0));
    spatial_y_moment_2_GP_elastm_[numgp] = CORE::FADUTILS::CastToDouble(stressm(1));
    spatial_z_moment_3_GP_elastm_[numgp] = CORE::FADUTILS::CastToDouble(stressm(2));
  }
}

template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
void DRT::ELEMENTS::Beam3r::CalcInertiaForceAndMassMatrix(
    const CORE::LINALG::Matrix<3 * vpernode * nnodecl, 1, double>& disp_totlag_centerline,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>& Qnode,
    CORE::LINALG::SerialDenseMatrix* massmatrix, CORE::LINALG::SerialDenseVector* inertia_force)
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

  const double dt = ParamsInterface().GetDeltaTime();
  const double beta = ParamsInterface().GetBeamParamsInterfacePtr()->GetBeta();
  const double gamma = ParamsInterface().GetBeamParamsInterfacePtr()->GetGamma();
  const double alpha_f = ParamsInterface().GetBeamParamsInterfacePtr()->GetAlphaf();
  const double alpha_m = ParamsInterface().GetBeamParamsInterfacePtr()->GetAlpham();

  const bool materialintegration = true;  // TODO unused? remove or realize coverage by test case
  const double diff_factor_vel = gamma / (beta * dt);
  const double diff_factor_acc = (1.0 - alpha_m) / (beta * dt * dt * (1.0 - alpha_f));

  CORE::LINALG::Matrix<3, 3> Lambdanewmass(true);
  CORE::LINALG::Matrix<3, 3> Lambdaconvmass(true);

  // tensor of mass moments of inertia for translational and rotational motion
  double mass_inertia_translational = 0.0;
  CORE::LINALG::Matrix<3, 3> Jp(true);

  GetTranslationalAndRotationalMassInertiaTensor(mass_inertia_translational, Jp);

  //********************************** shape functions ************************************
  /* Note: index i refers to the i-th shape function (i = 0 ... nnode*vpernode-1)
   * the vectors store individual shape functions, NOT an assembled matrix of shape functions)*/

  /* vector whose numgp-th element is a 1xnnode-matrix with all Lagrange polynomial shape functions
   * evaluated at the numgp-th Gauss point these shape functions are used for the interpolation of
   * the triad field*/
  std::vector<CORE::LINALG::Matrix<1, nnodetriad, double>> I_i;

  /* vector whose numgp-th element is a 1x(vpernode*nnode)-matrix with all (Lagrange/Hermite) shape
   * functions evaluated at the numgp-th GP
   * these shape functions are used for the interpolation of the beam centerline*/
  std::vector<CORE::LINALG::Matrix<1, vpernode * nnodecl, double>> H_i;

  // get integration scheme for inertia forces and mass matrix
  CORE::DRT::UTILS::IntegrationPoints1D gausspoints_mass(MyGaussRule(res_inertia));
  // reuse variables for individual shape functions and resize to new numgp
  I_i.resize(gausspoints_mass.nquad);
  H_i.resize(gausspoints_mass.nquad);

  // evaluate all shape functions at all specified Gauss points
  DRT::UTILS::BEAM::EvaluateShapeFunctionsAllGPs<nnodetriad, 1>(
      gausspoints_mass, I_i, this->Shape());
  DRT::UTILS::BEAM::EvaluateShapeFunctionsAllGPs<nnodecl, vpernode>(
      gausspoints_mass, H_i, this->Shape(), this->RefLength());

  // Calculate current centerline position at gauss points (needed for element intern time
  // integration)
  for (int gp = 0; gp < gausspoints_mass.nquad; gp++)  // loop through Gauss points
    Calc_r<nnodecl, vpernode, double>(disp_totlag_centerline, H_i[gp], rnewGPmass_[gp]);

  // create object of triad interpolation scheme
  Teuchos::RCP<LARGEROTATIONS::TriadInterpolationLocalRotationVectors<nnodetriad, double>>
      triad_interpolation_scheme_ptr = Teuchos::rcp(
          new LARGEROTATIONS::TriadInterpolationLocalRotationVectors<nnodetriad, double>());

  // reset triad interpolation scheme with nodal quaternions
  triad_interpolation_scheme_ptr->Reset(Qnode);

  Ekin_ = 0.0;
  L_ = 0.0;
  P_ = 0.0;

  // interpolated local relative rotation \Psi^l at a certain Gauss point according to (3.11),
  // Jelenic 1999
  CORE::LINALG::Matrix<3, 1, double> Psi_l(true);

  // vector with nnode elements, who represent the 3x3-matrix-shaped interpolation function
  // \tilde{I}^nnode at a certain Gauss point according to (3.18), Jelenic 1999
  std::vector<CORE::LINALG::Matrix<3, 3, double>> Itilde(nnodetriad);

  for (int gp = 0; gp < gausspoints_mass.nquad; gp++)  // loop through Gauss points
  {
    // weight of GP in parameter space
    const double wgtmass = gausspoints_mass.qwgt[gp];

    CORE::LINALG::Matrix<3, 3> Jp_bar(Jp);
    Jp_bar.Scale(diff_factor_acc);

    CORE::LINALG::Matrix<3, 1> dL(true);

    triad_interpolation_scheme_ptr->GetInterpolatedLocalRotationVector(Psi_l, I_i[gp]);

    triad_interpolation_scheme_ptr->GetInterpolatedQuaternion(QnewGPmass_[gp], Psi_l);

    triad_interpolation_scheme_ptr->GetNodalGeneralizedRotationInterpolationMatrices(
        Itilde, Psi_l, I_i[gp]);

    Lambdanewmass.Clear();
    Lambdaconvmass.Clear();
    // compute current and old triad at Gauss point
    CORE::LARGEROTATIONS::quaterniontotriad<double>(QnewGPmass_[gp], Lambdanewmass);
    CORE::LARGEROTATIONS::quaterniontotriad<double>(QconvGPmass_[gp], Lambdaconvmass);

    // rotation between last converged position and current position expressed as a quaternion
    CORE::LINALG::Matrix<4, 1> deltaQ(true);
    CORE::LARGEROTATIONS::quaternionproduct(
        CORE::LARGEROTATIONS::inversequaternion<double>(QconvGPmass_[gp]), QnewGPmass_[gp], deltaQ);

    // spatial rotation between last converged position and current position expressed as a three
    // element rotation vector
    CORE::LINALG::Matrix<3, 1> deltatheta(true);
    CORE::LARGEROTATIONS::quaterniontoangle<double>(deltaQ, deltatheta);

    // compute material counterparts of spatial vectors
    CORE::LINALG::Matrix<3, 1> deltaTHETA(true);
    CORE::LINALG::Matrix<3, 1> Wconvmass(true);
    CORE::LINALG::Matrix<3, 1> Wnewmass(true);
    CORE::LINALG::Matrix<3, 1> Aconvmass(true);
    CORE::LINALG::Matrix<3, 1> Anewmass(true);
    CORE::LINALG::Matrix<3, 1> Amodconvmass(true);
    CORE::LINALG::Matrix<3, 1> Amodnewmass(true);
    deltaTHETA.MultiplyTN(Lambdanewmass, deltatheta);
    Wconvmass.MultiplyTN(Lambdaconvmass, wconvGPmass_[gp]);
    Aconvmass.MultiplyTN(Lambdaconvmass, aconvGPmass_[gp]);
    Amodconvmass.MultiplyTN(Lambdaconvmass, amodconvGPmass_[gp]);

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
      wnewGPmass_[gp].Multiply(Lambdanewmass, Wnewmass);
      anewGPmass_[gp].Multiply(Lambdanewmass, Anewmass);
      amodnewGPmass_[gp].Multiply(Lambdanewmass, Amodnewmass);
    }
    else
    {
      for (unsigned int i = 0; i < 3; i++)
      {
        wnewGPmass_[gp](i) = gamma / (beta * dt) * deltatheta(i) +
                             (1 - gamma / beta) * wconvGPmass_[gp](i) +
                             dt * (1 - gamma / (2 * beta)) * amodconvGPmass_[gp](i);

        anewGPmass_[gp](i) = (1.0 - alpha_m) / (beta * dt * dt * (1.0 - alpha_f)) * deltatheta(i) -
                             (1.0 - alpha_m) / (beta * dt * (1.0 - alpha_f)) * wconvGPmass_[gp](i) -
                             alpha_f / (1.0 - alpha_f) * aconvGPmass_[gp](i) +
                             (alpha_m / (1.0 - alpha_f) -
                                 (0.5 - beta) * (1.0 - alpha_m) / (beta * (1.0 - alpha_f))) *
                                 amodconvGPmass_[gp](i);

        amodnewGPmass_[gp](i) =
            1.0 / (1.0 - alpha_m) *
            ((1.0 - alpha_f) * anewGPmass_[gp](i) + alpha_f * aconvGPmass_[gp](i) -
                alpha_m * amodconvGPmass_[gp](i));
      }
      Wnewmass.MultiplyTN(Lambdanewmass, wnewGPmass_[gp]);
      Anewmass.MultiplyTN(Lambdanewmass, anewGPmass_[gp]);
      Amodnewmass.MultiplyTN(Lambdanewmass, amodnewGPmass_[gp]);
    }

    CORE::LINALG::Matrix<3, 1> deltar(true);
    for (unsigned int i = 0; i < 3; i++)
    {
      deltar(i) = rnewGPmass_[gp](i) - rconvGPmass_[gp](i);
    }
    for (unsigned int i = 0; i < 3; i++)
    {
      rttnewGPmass_[gp](i) =
          (1.0 - alpha_m) / (beta * dt * dt * (1.0 - alpha_f)) * deltar(i) -
          (1.0 - alpha_m) / (beta * dt * (1.0 - alpha_f)) * rtconvGPmass_[gp](i) -
          alpha_f / (1.0 - alpha_f) * rttconvGPmass_[gp](i) +
          (alpha_m / (1.0 - alpha_f) - (0.5 - beta) * (1.0 - alpha_m) / (beta * (1.0 - alpha_f))) *
              rttmodconvGPmass_[gp](i);

      rtnewGPmass_[gp](i) = gamma / (beta * dt) * deltar(i) +
                            (1 - gamma / beta) * rtconvGPmass_[gp](i) +
                            dt * (1 - gamma / (2 * beta)) * rttmodconvGPmass_[gp](i);

      rttmodnewGPmass_[gp](i) =
          1.0 / (1.0 - alpha_m) *
          ((1.0 - alpha_f) * rttnewGPmass_[gp](i) + alpha_f * rttconvGPmass_[gp](i) -
              alpha_m * rttmodconvGPmass_[gp](i));
    }

    // spin matrix of the material angular velocity, i.e. S(W)
    CORE::LINALG::Matrix<3, 3> SWnewmass(true);
    CORE::LARGEROTATIONS::computespin<double>(SWnewmass, Wnewmass);
    CORE::LINALG::Matrix<3, 1> Jp_Wnewmass(true);
    CORE::LINALG::Matrix<3, 1> auxvector1(true);
    CORE::LINALG::Matrix<3, 1> Pi_t(true);
    Jp_Wnewmass.Multiply(Jp, Wnewmass);
    for (unsigned int i = 0; i < 3; i++)
      for (unsigned int j = 0; j < 3; j++)
        auxvector1(i) += SWnewmass(i, j) * Jp_Wnewmass(j) + Jp(i, j) * Anewmass(j);

    Pi_t.Multiply(Lambdanewmass, auxvector1);
    CORE::LINALG::Matrix<3, 1> r_tt(true);
    CORE::LINALG::Matrix<3, 1> r_t(true);
    CORE::LINALG::Matrix<3, 1> r(true);

    r_tt = rttnewGPmass_[gp];
    r_t = rtnewGPmass_[gp];
    r = rnewGPmass_[gp];

    CORE::LINALG::Matrix<3, 3> S_r(true);
    CORE::LARGEROTATIONS::computespin<double>(S_r, r);
    dL.Multiply(S_r, r_t);
    dL.Scale(mass_inertia_translational);
    CORE::LINALG::Matrix<3, 1> Lambdanewmass_Jp_Wnewmass(true);
    Lambdanewmass_Jp_Wnewmass.Multiply(Lambdanewmass, Jp_Wnewmass);
    dL.Update(1.0, Lambdanewmass_Jp_Wnewmass, 1.0);
    for (unsigned int i = 0; i < 3; i++)
    {
      L_(i) += wgtmass * jacobiGPmass_[gp] * dL(i);
      P_(i) += wgtmass * jacobiGPmass_[gp] * mass_inertia_translational * r_t(i);
    }

    CORE::LINALG::Matrix<3, 3> S_Pit(true);
    CORE::LARGEROTATIONS::computespin<double>(S_Pit, Pi_t);
    CORE::LINALG::Matrix<3, 3> SJpWnewmass(true);
    CORE::LARGEROTATIONS::computespin<double>(SJpWnewmass, Jp_Wnewmass);
    CORE::LINALG::Matrix<3, 3> SWnewmass_Jp(true);
    SWnewmass_Jp.Multiply(SWnewmass, Jp);
    Jp_bar.Update(diff_factor_vel, SWnewmass_Jp, 1.0);
    Jp_bar.Update(-diff_factor_vel, SJpWnewmass, 1.0);

    CORE::LINALG::Matrix<3, 3> Tmatrix(true);
    Tmatrix = CORE::LARGEROTATIONS::Tmatrix(deltatheta);

    CORE::LINALG::Matrix<3, 3> Lambdanewmass_Jpbar(true);
    Lambdanewmass_Jpbar.Multiply(Lambdanewmass, Jp_bar);
    CORE::LINALG::Matrix<3, 3> LambdaconvmassT_Tmatrix(true);
    LambdaconvmassT_Tmatrix.MultiplyTN(Lambdaconvmass, Tmatrix);
    CORE::LINALG::Matrix<3, 3> Lambdanewmass_Jpbar_LambdaconvmassT_Tmatrix(true);
    Lambdanewmass_Jpbar_LambdaconvmassT_Tmatrix.Multiply(
        Lambdanewmass_Jpbar, LambdaconvmassT_Tmatrix);
    CORE::LINALG::Matrix<3, 3> auxmatrix1(true);
    auxmatrix1.Update(-1.0, S_Pit, 1.0);
    auxmatrix1.Update(1.0, Lambdanewmass_Jpbar_LambdaconvmassT_Tmatrix, 1.0);

    if (inertia_force != NULL)
    {
      // inertia forces
      for (unsigned int i = 0; i < 3; i++)
      {
        for (unsigned int node = 0; node < nnodecl; node++)
        {
          // translational contribution
          (*inertia_force)(dofpercombinode * node + i) += jacobiGPmass_[gp] * wgtmass *
                                                          mass_inertia_translational *
                                                          H_i[gp](vpernode * node) * r_tt(i);
          if (centerline_hermite_)
            (*inertia_force)(dofpercombinode * node + 6 + i) +=
                jacobiGPmass_[gp] * wgtmass * mass_inertia_translational *
                H_i[gp](vpernode * node + 1) * r_tt(i);
          // rotational contribution
          (*inertia_force)(dofpercombinode * node + 3 + i) +=
              jacobiGPmass_[gp] * wgtmass * I_i[gp](node) * Pi_t(i);
        }
        for (unsigned int node = nnodecl; node < nnodetriad;
             node++)  // this loop is only entered in case of nnodetriad>nnodecl
        {
          // rotational contribution
          (*inertia_force)(dofperclnode * nnodecl + dofpertriadnode * node + i) +=
              jacobiGPmass_[gp] * wgtmass * I_i[gp](node) * Pi_t(i);
        }
      }
    }

    if (massmatrix != NULL)
    {
      // linearization of inertia forces: massmatrix
      for (unsigned int jnode = 0; jnode < nnodecl; jnode++)
      {
        // translational contribution
        for (unsigned int inode = 0; inode < nnodecl; inode++)
          for (unsigned int k = 0; k < 3; k++)
          {
            (*massmatrix)(dofpercombinode * inode + k, dofpercombinode * jnode + k) +=
                diff_factor_acc * jacobiGPmass_[gp] * wgtmass * mass_inertia_translational *
                H_i[gp](vpernode * inode) * H_i[gp](vpernode * jnode);
            if (centerline_hermite_)
            {
              (*massmatrix)(dofpercombinode * inode + 6 + k, dofpercombinode * jnode + 6 + k) +=
                  diff_factor_acc * jacobiGPmass_[gp] * wgtmass * mass_inertia_translational *
                  H_i[gp](vpernode * inode + 1) * H_i[gp](vpernode * jnode + 1);
              (*massmatrix)(dofpercombinode * inode + k, dofpercombinode * jnode + 6 + k) +=
                  diff_factor_acc * jacobiGPmass_[gp] * wgtmass * mass_inertia_translational *
                  H_i[gp](vpernode * inode) * H_i[gp](vpernode * jnode + 1);
              (*massmatrix)(dofpercombinode * inode + 6 + k, dofpercombinode * jnode + k) +=
                  diff_factor_acc * jacobiGPmass_[gp] * wgtmass * mass_inertia_translational *
                  H_i[gp](vpernode * inode + 1) * H_i[gp](vpernode * jnode);
            }
          }

        // rotational contribution
        CORE::LINALG::Matrix<3, 3> auxmatrix2(true);
        auxmatrix2.Multiply(auxmatrix1, Itilde[jnode]);
        for (unsigned int inode = 0; inode < nnodecl; inode++)
        {
          for (unsigned int i = 0; i < 3; i++)
            for (unsigned int j = 0; j < 3; j++)
              (*massmatrix)(dofpercombinode * inode + 3 + i, dofpercombinode * jnode + 3 + j) +=
                  jacobiGPmass_[gp] * wgtmass * I_i[gp](inode) * auxmatrix2(i, j);
        }
        for (unsigned int inode = nnodecl; inode < nnodetriad;
             inode++)  // this loop is only entered in case of nnodetriad>nnodecl
        {
          for (unsigned int i = 0; i < 3; i++)
            for (unsigned int j = 0; j < 3; j++)
              (*massmatrix)(dofperclnode * nnodecl + dofpertriadnode * inode + i,
                  dofpercombinode * jnode + 3 + j) +=
                  jacobiGPmass_[gp] * wgtmass * I_i[gp](inode) * auxmatrix2(i, j);
        }
      }
      for (unsigned int jnode = nnodecl; jnode < nnodetriad;
           ++jnode)  // this loop is only entered in case of nnodetriad>nnodecl
      {
        // rotational contribution
        CORE::LINALG::Matrix<3, 3> auxmatrix2(true);
        auxmatrix2.Multiply(auxmatrix1, Itilde[jnode]);
        for (unsigned int inode = 0; inode < nnodecl; inode++)
        {
          for (unsigned int i = 0; i < 3; i++)
            for (unsigned int j = 0; j < 3; j++)
              (*massmatrix)(dofpercombinode * inode + 3 + i,
                  dofperclnode * nnodecl + dofpertriadnode * jnode + j) +=
                  jacobiGPmass_[gp] * wgtmass * I_i[gp](inode) * auxmatrix2(i, j);
        }
        for (unsigned int inode = nnodecl; inode < nnodetriad;
             inode++)  // this loop is only entered in case of nnodetriad>nnodecl
        {
          for (unsigned int i = 0; i < 3; i++)
            for (unsigned int j = 0; j < 3; j++)
              (*massmatrix)(dofperclnode * nnodecl + dofpertriadnode * inode + i,
                  dofperclnode * nnodecl + dofpertriadnode * jnode + j) +=
                  jacobiGPmass_[gp] * wgtmass * I_i[gp](inode) * auxmatrix2(i, j);
        }
      }
    }

    // Calculation of kinetic energy
    CORE::LINALG::Matrix<1, 1> ekinrot(true);
    CORE::LINALG::Matrix<1, 1> ekintrans(true);
    ekinrot.MultiplyTN(Wnewmass, Jp_Wnewmass);
    ekintrans.MultiplyTN(r_t, r_t);
    Ekin_ += 0.5 * (ekinrot.Norm2() + mass_inertia_translational * ekintrans.Norm2()) *
             jacobiGPmass_[gp] * wgtmass;
    Ekintorsion_ += 0.5 * Wnewmass(0) * Jp_Wnewmass(0) * jacobiGPmass_[gp] * wgtmass;
    Ekinbending_ += 0.5 * Wnewmass(1) * Jp_Wnewmass(1) * jacobiGPmass_[gp] * wgtmass;
    Ekinbending_ += 0.5 * Wnewmass(2) * Jp_Wnewmass(2) * jacobiGPmass_[gp] * wgtmass;
    Ekintrans_ +=
        0.5 * mass_inertia_translational * ekintrans.Norm2() * jacobiGPmass_[gp] * wgtmass;

    Jp_Wnewmass.Multiply(Jp, Wnewmass);
  }

  // In Lie group GenAlpha algorithm, the mass matrix is multiplied with factor
  // (1.0-alpham_)/(beta_*dt*dt*(1.0-alphaf_)) later. so we apply inverse factor here because the
  // correct prefactors for displacement/velocity/acceleration dependent terms have been applied
  // individually above
  if (massmatrix != NULL) massmatrix->scale(beta * dt * dt * (1.0 - alpha_f) / (1.0 - alpha_m));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticForceContributions(
    CORE::LINALG::SerialDenseMatrix& stiffmatrix, const CORE::LINALG::Matrix<3, 1, double>& stressn,
    const CORE::LINALG::Matrix<3, 3, double>& cn, const CORE::LINALG::Matrix<3, 3, double>& r_s_hat,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<nnodetriad, double>& triad_intpol,
    const CORE::LINALG::Matrix<1, nnodetriad, double>& I_i,
    const CORE::LINALG::Matrix<1, vpernode * nnodecl, double>& H_i_xi, const double wgt,
    const double jacobifactor) const
{
  const unsigned int dofperclnode = 3 * vpernode;
  const unsigned int dofpertriadnode = 3;
  const unsigned int dofpercombinode = dofperclnode + dofpertriadnode;

  /* computation of stiffness matrix according to Jelenic 1999, eq. (4.7); computation split up with
   * respect to single blocks of matrix in eq. (4.7). note: again, jacobi factor cancels out in
   * terms whith I^{i'}=I_i_s=I_i_xi*(dxi/ds) (see comment above) but be careful: Itildeprime and
   * rprime are indeed derivatives with respect to arc-length parameter in reference configuration s
   */

  // vector with nnode elements, who represent the 3x3-matrix-shaped interpolation function
  // \tilde{I}^nnode at a certain Gauss point according to (3.18), Jelenic 1999
  std::vector<CORE::LINALG::Matrix<3, 3, double>> Itilde(nnodetriad);

  CORE::LINALG::Matrix<3, 1, double> Psi_l(true);
  triad_intpol.GetInterpolatedLocalRotationVector(Psi_l, I_i);
  triad_intpol.GetNodalGeneralizedRotationInterpolationMatrices(Itilde, Psi_l, I_i);


  // auxiliary variables for storing intermediate matrices in computation of entries of stiffness
  // matrix
  CORE::LINALG::Matrix<3, 3, double> auxmatrix1;
  CORE::LINALG::Matrix<3, 3, double> auxmatrix2;
  CORE::LINALG::Matrix<3, 3, double> auxmatrix3;

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
      auxmatrix2.Multiply(r_s_hat, cn);
      CORE::LARGEROTATIONS::computespin(auxmatrix1, stressn);
      auxmatrix1 -= auxmatrix2;
      auxmatrix1.Scale(I_i(nodei));
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
      auxmatrix2.Multiply(cn, r_s_hat);
      CORE::LARGEROTATIONS::computespin(auxmatrix1, stressn);
      auxmatrix2 -= auxmatrix1;  // auxmatrix2: term in parantheses

      auxmatrix3.Multiply(auxmatrix2, Itilde[nodej]);
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
      // third summand; note: error in eq. (4.7), Jelenic 1999: the first summand in the parantheses
      // should be \hat{\Lambda N} instead of \Lambda N
      auxmatrix1.Multiply(
          auxmatrix2, Itilde[nodej]);  // term in parantheses is the same as in upper right block
                                       // but with opposite sign (note '-=' below)

      auxmatrix3.Multiply(r_s_hat, auxmatrix1);
      auxmatrix3.Scale(I_i(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * nodei + 3 + i, dofpercombinode * nodej + 3 + j) -=
              auxmatrix3(i, j) * jacobifactor * wgt;
    }
    for (unsigned int nodej = nnodecl; nodej < nnodetriad;
         nodej++)  // this loop is only entered in case of nnodetriad>nnodecl
    {
      // upper right block
      auxmatrix2.Multiply(cn, r_s_hat);
      CORE::LARGEROTATIONS::computespin(auxmatrix1, stressn);
      auxmatrix2 -= auxmatrix1;  // auxmatrix2: term in parantheses

      auxmatrix3.Multiply(auxmatrix2, Itilde[nodej]);
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
      // third summand; note: error in eq. (4.7), Jelenic 1999: the first summand in the parantheses
      // should be \hat{\Lambda N} instead of \Lambda N
      auxmatrix1.Multiply(
          auxmatrix2, Itilde[nodej]);  // term in parantheses is the same as in upper right block
                                       // but with opposite sign (note '-=' below)

      auxmatrix3.Multiply(r_s_hat, auxmatrix1);
      auxmatrix3.Scale(I_i(nodei));
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
      auxmatrix2.Multiply(r_s_hat, cn);
      CORE::LARGEROTATIONS::computespin(auxmatrix1, stressn);
      auxmatrix1 -= auxmatrix2;
      auxmatrix1.Scale(I_i(nodei));
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
      // third summand; note: error in eq. (4.7), Jelenic 1999: the first summand in the parantheses
      // should be \hat{\Lambda N} instead of \Lambda N
      auxmatrix2.Multiply(cn, r_s_hat);
      CORE::LARGEROTATIONS::computespin(auxmatrix1, stressn);
      auxmatrix2 -= auxmatrix1;  // auxmatrix2: term in parantheses

      auxmatrix1.Multiply(
          auxmatrix2, Itilde[nodej]);  // term in parantheses is the same as in upper right block
                                       // but with opposite sign (note '-=' below)

      auxmatrix3.Multiply(r_s_hat, auxmatrix1);
      auxmatrix3.Scale(I_i(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * nodei + i,
              dofpercombinode * nodej + 3 + j) -= auxmatrix3(i, j) * jacobifactor * wgt;
    }
    for (unsigned int nodej = nnodecl; nodej < nnodetriad; nodej++)
    {
      // lower right block
      // third summand; note: error in eq. (4.7), Jelenic 1999: the first summand in the parantheses
      // should be \hat{\Lambda N} instead of \Lambda N
      auxmatrix2.Multiply(cn, r_s_hat);
      CORE::LARGEROTATIONS::computespin(auxmatrix1, stressn);
      auxmatrix2 -= auxmatrix1;  // auxmatrix2: term in parantheses

      auxmatrix1.Multiply(
          auxmatrix2, Itilde[nodej]);  // term in parantheses is the same as in upper right block
                                       // but with opposite sign (note '-=' below)

      auxmatrix3.Multiply(r_s_hat, auxmatrix1);
      auxmatrix3.Scale(I_i(nodei));
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
void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticMomentContributions(
    CORE::LINALG::SerialDenseMatrix& stiffmatrix, const CORE::LINALG::Matrix<3, 1, double>& stressm,
    const CORE::LINALG::Matrix<3, 3, double>& cm,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<nnodetriad, double>& triad_intpol,
    const CORE::LINALG::Matrix<3, 1, double>& Psi_l,
    const CORE::LINALG::Matrix<3, 1, double>& Psi_l_s,
    const CORE::LINALG::Matrix<1, nnodetriad, double>& I_i,
    const CORE::LINALG::Matrix<1, nnodetriad, double>& I_i_xi, const double wgt,
    const double jacobifactor) const
{
  const unsigned int dofperclnode = 3 * vpernode;
  const unsigned int dofpertriadnode = 3;
  const unsigned int dofpercombinode = dofperclnode + dofpertriadnode;

  /* computation of stiffness matrix according to Jelenic 1999, eq. (4.7)*/

  // vector with nnode elements, who represent the 3x3-matrix-shaped interpolation function
  // \tilde{I}^nnode at a certain Gauss point according to (3.18), Jelenic 1999
  std::vector<CORE::LINALG::Matrix<3, 3, double>> Itilde(nnodetriad);

  // vector with nnode elements, who represent the 3x3-matrix-shaped interpolation function
  // \tilde{I'}^nnode at a certain Gauss point according to (3.19), Jelenic 1999
  std::vector<CORE::LINALG::Matrix<3, 3, double>> Itildeprime(nnodetriad);

  triad_intpol.GetNodalGeneralizedRotationInterpolationMatrices(Itilde, Psi_l, I_i);

  triad_intpol.GetNodalGeneralizedRotationInterpolationMatricesDerivative(
      Itildeprime, Psi_l, Psi_l_s, I_i, I_i_xi, jacobifactor);


  // auxiliary variables for storing intermediate matrices in computation of entries of stiffness
  // matrix
  CORE::LINALG::Matrix<3, 3, double> auxmatrix1;
  CORE::LINALG::Matrix<3, 3, double> auxmatrix2;

  for (unsigned int nodei = 0; nodei < nnodecl; nodei++)
  {
    for (unsigned int nodej = 0; nodej < nnodecl; nodej++)
    {
      // lower right block
      // first summand
      auxmatrix1.Multiply(cm, Itildeprime[nodej]);
      auxmatrix1.Scale(I_i_xi(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * nodei + 3 + i, dofpercombinode * nodej + 3 + j) +=
              auxmatrix1(i, j) * wgt;

      // second summand
      CORE::LARGEROTATIONS::computespin(auxmatrix2, stressm);
      auxmatrix1.Multiply(auxmatrix2, Itilde[nodej]);
      auxmatrix1.Scale(I_i_xi(nodei));
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
      auxmatrix1.Multiply(cm, Itildeprime[nodej]);
      auxmatrix1.Scale(I_i_xi(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * nodei + 3 + i,
              dofperclnode * nnodecl + dofpertriadnode * nodej + j) += auxmatrix1(i, j) * wgt;

      // second summand
      CORE::LARGEROTATIONS::computespin(auxmatrix2, stressm);
      auxmatrix1.Multiply(auxmatrix2, Itilde[nodej]);
      auxmatrix1.Scale(I_i_xi(nodei));
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
      auxmatrix1.Multiply(cm, Itildeprime[nodej]);
      auxmatrix1.Scale(I_i_xi(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * nodei + i,
              dofpercombinode * nodej + 3 + j) += auxmatrix1(i, j) * wgt;

      // second summand
      CORE::LARGEROTATIONS::computespin(auxmatrix2, stressm);
      auxmatrix1.Multiply(auxmatrix2, Itilde[nodej]);
      auxmatrix1.Scale(I_i_xi(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * nodei + i,
              dofpercombinode * nodej + 3 + j) -= auxmatrix1(i, j) * wgt;
    }
    for (unsigned int nodej = nnodecl; nodej < nnodetriad; nodej++)
    {
      // lower right block
      // first summand
      auxmatrix1.Multiply(cm, Itildeprime[nodej]);
      auxmatrix1.Scale(I_i_xi(nodei));
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofperclnode * nnodecl + dofpertriadnode * nodei + i,
              dofperclnode * nnodecl + dofpertriadnode * nodej + j) += auxmatrix1(i, j) * wgt;

      // second summand
      CORE::LARGEROTATIONS::computespin(auxmatrix2, stressm);
      auxmatrix1.Multiply(auxmatrix2, Itilde[nodej]);
      auxmatrix1.Scale(I_i_xi(nodei));
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
void DRT::ELEMENTS::Beam3r::CalcStiffmatAutomaticDifferentiation(
    CORE::LINALG::SerialDenseMatrix& stiffmatrix,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>& Qnode,
    CORE::LINALG::Matrix<3 * vpernode * nnodecl + 3 * nnodetriad, 1, Sacado::Fad::DFad<double>>
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

  CORE::LINALG::Matrix<3, 3, double> tempmat(true);
  CORE::LINALG::Matrix<3, 3, double> newstiffmat(true);
  CORE::LINALG::Matrix<3, 3, double> Tmat(true);
  CORE::LINALG::Matrix<3, 1, double> theta_totlag_j(true);

  for (unsigned int jnode = 0; jnode < nnodecl; jnode++)
  {
    // compute physical total angle theta_totlag
    CORE::LARGEROTATIONS::quaterniontoangle(Qnode[jnode], theta_totlag_j);

    // compute Tmatrix of theta_totlag_i
    Tmat = CORE::LARGEROTATIONS::Tmatrix(theta_totlag_j);

    for (unsigned int inode = 0; inode < nnodecl; inode++)
    {
      // block1: derivative of nodal positions with respect to theta (rotational DOFs)
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          tempmat(i, j) = stiffmatrix(dofpercombinode * inode + i, dofpercombinode * jnode + 3 + j);

      newstiffmat.Clear();
      newstiffmat.MultiplyNN(tempmat, Tmat);

      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * inode + i, dofpercombinode * jnode + 3 + j) =
              newstiffmat(i, j);

      // block2: derivative of nodal theta with respect to theta (rotational DOFs)
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          tempmat(i, j) =
              stiffmatrix(dofpercombinode * inode + 3 + i, dofpercombinode * jnode + 3 + j);

      newstiffmat.Clear();
      newstiffmat.MultiplyNN(tempmat, Tmat);

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

        newstiffmat.Clear();
        newstiffmat.MultiplyNN(tempmat, Tmat);

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

      newstiffmat.Clear();
      newstiffmat.MultiplyNN(tempmat, Tmat);

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
    CORE::LARGEROTATIONS::quaterniontoangle(Qnode[jnode], theta_totlag_j);

    // compute Tmatrix of theta_totlag_i
    Tmat = CORE::LARGEROTATIONS::Tmatrix(theta_totlag_j);

    for (unsigned int inode = 0; inode < nnodecl; inode++)
    {
      // block1: derivative of nodal positions with respect to theta (rotational DOFs)
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          tempmat(i, j) = stiffmatrix(
              dofpercombinode * inode + i, dofperclnode * nnodecl + dofpertriadnode * jnode + j);

      newstiffmat.Clear();
      newstiffmat.MultiplyNN(tempmat, Tmat);

      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          stiffmatrix(dofpercombinode * inode + i,
              dofperclnode * nnodecl + dofpertriadnode * jnode + j) = newstiffmat(i, j);

      // block2: derivative of nodal theta with respect to theta (rotational DOFs)
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          tempmat(i, j) = stiffmatrix(dofpercombinode * inode + 3 + i,
              dofperclnode * nnodecl + dofpertriadnode * jnode + j);

      newstiffmat.Clear();
      newstiffmat.MultiplyNN(tempmat, Tmat);

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

        newstiffmat.Clear();
        newstiffmat.MultiplyNN(tempmat, Tmat);

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

      newstiffmat.Clear();
      newstiffmat.MultiplyNN(tempmat, Tmat);

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
void DRT::ELEMENTS::Beam3r::CalcBrownianForcesAndStiff(Teuchos::ParameterList& params,
    std::vector<double>& vel, std::vector<double>& disp,
    CORE::LINALG::SerialDenseMatrix* stiffmatrix, CORE::LINALG::SerialDenseVector* force)
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
  if (BrownianDynParamsInterfacePtr() != Teuchos::null)
    UnShiftNodePosition(disp, *BrownianDynParamsInterface().GetPeriodicBoundingBox());

  /****** update/compute key variables describing displacement and velocity state of this element
   * *****/

  // current nodal DOFs relevant for centerline interpolation in total Lagrangian style, i.e.
  // initial values + displacements
  CORE::LINALG::Matrix<3 * vpernode * nnodecl, 1, double> disp_totlag_centerline(true);

  // discrete centerline (i.e. translational) velocity vector
  CORE::LINALG::Matrix<3 * vpernode * nnodecl, 1, double> vel_centerline(true);

  // quaternions of all nodal triads
  std::vector<CORE::LINALG::Matrix<4, 1, double>> Q_i(nnodetriad);

  // update disp_totlag_centerline and nodal triads
  UpdateDispTotLagAndNodalTriads<nnodetriad, nnodecl, vpernode, double>(
      disp, disp_totlag_centerline, Q_i);

  // update current values of centerline (i.e. translational) velocity
  ExtractCenterlineDofValuesFromElementStateVector<nnodecl, vpernode, double>(vel, vel_centerline);

  /****** compute and assemble force and stiffness contributions from viscous damping and stochastic
   * forces *****/

  // add stiffness and forces (i.e. moments) due to rotational damping effects
  EvaluateRotationalDamping<nnodetriad, nnodecl, vpernode, 3>(params, Q_i, stiffmatrix, force);

  if (stiffmatrix != NULL) stiff_ptc_ = *stiffmatrix;

  // add stiffness and forces due to translational damping effects
  EvaluateTranslationalDamping<nnodecl, vpernode, 3>(
      params, vel_centerline, disp_totlag_centerline, stiffmatrix, force);


  // add stochastic forces and (if required) resulting stiffness
  EvaluateStochasticForces<nnodecl, vpernode, 3, 3>(
      params, disp_totlag_centerline, stiffmatrix, force);
}

void DRT::ELEMENTS::Beam3r::CalcStiffContributionsPTC(CORE::LINALG::SerialDenseMatrix& elemat1)
{
  elemat1 = stiff_ptc_;
}

/*------------------------------------------------------------------------------------------------------------*
 | lump mass matrix             (private)                                                   cyron
 01/08|
 *------------------------------------------------------------------------------------------------------------*/
template <unsigned int nnode>
void DRT::ELEMENTS::Beam3r::lumpmass(CORE::LINALG::SerialDenseMatrix* massmatrix)
{
  // lump mass matrix
  if (massmatrix != NULL)
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
void DRT::ELEMENTS::Beam3r::EvaluatePTC(
    Teuchos::ParameterList& params, CORE::LINALG::SerialDenseMatrix& elemat1)
{
  // apply PTC rotation damping term using a Lobatto integration rule; implemented for 2 nodes only
  if (nnode > 2 or centerline_hermite_)
    dserror(
        "PTC was originally implemented for 2-noded Reissner beam element only. Check "
        "functionality for "
        "numnodes>2 and/or Hermite interpolation and extend if needed!");

  for (unsigned int node = 0; node < nnode; node++)
  {
    // computing angle increment from current position in comparison with last converged position
    // for damping
    CORE::LINALG::Matrix<4, 1> deltaQ;
    CORE::LARGEROTATIONS::quaternionproduct(
        CORE::LARGEROTATIONS::inversequaternion(Qconvnode_[node]), Qnewnode_[node], deltaQ);
    CORE::LINALG::Matrix<3, 1> deltatheta;
    CORE::LARGEROTATIONS::quaterniontoangle(deltaQ, deltatheta);

    // isotropic artificial stiffness
    CORE::LINALG::Matrix<3, 3> artstiff;
    artstiff = CORE::LARGEROTATIONS::Tmatrix(deltatheta);

    // scale artificial damping with crotptc parameter for PTC method
    artstiff.Scale(params.get<double>("crotptc", 0.0));

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
int DRT::ELEMENTS::Beam3r::HowManyRandomNumbersINeed() const
{
  // get Gauss rule for evaluation of stochastic force contributions
  CORE::DRT::UTILS::GaussRule1D gaussrule = MyGaussRule(res_damp_stoch);
  CORE::DRT::UTILS::IntegrationPoints1D gausspoints(gaussrule);

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
void DRT::ELEMENTS::Beam3r::EvaluateRotationalDamping(
    Teuchos::ParameterList& params,  //!< parameter list
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>& Qnode,
    CORE::LINALG::SerialDenseMatrix* stiffmatrix,  //!< element stiffness matrix
    CORE::LINALG::SerialDenseVector* force)        //!< element internal force vector
{
  const unsigned int dofperclnode = 3 * vpernode;
  const unsigned int dofpertriadnode = 3;
  const unsigned int dofpercombinode = dofperclnode + dofpertriadnode;

  // get time step size
  double dt_inv = 0.0001;
  if (IsParamsInterface())
    dt_inv = 1.0 / ParamsInterface().GetDeltaTime();
  else
    dt_inv = 1.0 / params.get<double>("delta time", 1000);

  // get damping coefficients for translational and rotational degrees of freedom
  CORE::LINALG::Matrix<3, 1> gamma(true);
  GetDampingCoefficients(gamma);

  // get Gauss points and weights for evaluation of viscous damping contributions
  CORE::DRT::UTILS::GaussRule1D gaussrule = MyGaussRule(res_damp_stoch);
  CORE::DRT::UTILS::IntegrationPoints1D gausspoints(gaussrule);

  //*************************** physical quantities evaluated at a certain GP
  //***************************

  // interpolated local relative rotation \Psi^l at a certain Gauss point according to (3.11),
  // Jelenic 1999
  CORE::LINALG::Matrix<3, 1> Psi_l;

  // material triad and corresponding quaternion at a certain Gauss point
  CORE::LINALG::Matrix<3, 3> LambdaGP;
  CORE::LINALG::Matrix<4, 1> QnewGP;

  //********************************** (generalized) shape functions
  //************************************
  /* Note: index i refers to the i-th shape function (i = 0 ... nnodetriad-1)
   * the vectors store individual shape functions, NOT an assembled matrix of shape functions)*/

  /* vector whose numgp-th element is a 1xnnodetriad-matrix with all Lagrange polynomial shape
   * functions evaluated at the numgp-th Gauss point these shape functions are used for the
   * interpolation of the triad field */
  std::vector<CORE::LINALG::Matrix<1, nnodetriad, double>> I_i(gausspoints.nquad);

  // evaluate all shape functions at all specified Gauss points
  DRT::UTILS::BEAM::EvaluateShapeFunctionsAllGPs<nnodetriad, 1>(gausspoints, I_i, this->Shape());

  /* vector with nnodetriad elements, who represent the 3x3-matrix-shaped interpolation function
   * \tilde{I}^nnode according to (3.19), Jelenic 1999*/
  std::vector<CORE::LINALG::Matrix<3, 3, double>> Itilde(nnodetriad);


  // create an object of the triad interpolation scheme
  Teuchos::RCP<LARGEROTATIONS::TriadInterpolationLocalRotationVectors<nnodetriad, double>>
      triad_interpolation_scheme_ptr = Teuchos::rcp(
          new LARGEROTATIONS::TriadInterpolationLocalRotationVectors<nnodetriad, double>());

  // reset the scheme with nodal quaternions
  triad_interpolation_scheme_ptr->Reset(Qnode);


  for (int gp = 0; gp < gausspoints.nquad; gp++)
  {
    triad_interpolation_scheme_ptr->GetInterpolatedLocalRotationVector(Psi_l, I_i[gp]);

    triad_interpolation_scheme_ptr->GetInterpolatedQuaternion(QnewGP, Psi_l);

    // store in class variable in order to get QconvGPmass_ in subsequent time step
    QnewGPdampstoch_[gp] = QnewGP;

    // compute triad at Gauss point
    CORE::LARGEROTATIONS::quaterniontotriad(QnewGP, LambdaGP);


    // rotation between last converged state and current state expressed as a quaternion

    // ******** alternative 1 *************** Todo @grill
    //    CORE::LINALG::Matrix<4,1> deltaQ;
    //    CORE::LARGEROTATIONS::quaternionproduct(
    //        CORE::LARGEROTATIONS::inversequaternion(QconvGPdampstoch_[gp]), QnewGP, deltaQ);

    // ******** alternative 2 ***************

    // get quaternion in converged state at gp and compute corresponding triad
    CORE::LINALG::Matrix<3, 3, double> triad_mat_conv(true);
    CORE::LINALG::Matrix<4, 1, double> Qconv(true);
    for (unsigned int i = 0; i < 4; ++i) Qconv(i) = (QconvGPdampstoch_[gp])(i);

    CORE::LARGEROTATIONS::quaterniontotriad(Qconv, triad_mat_conv);

    // compute quaternion of relative rotation from converged to current state
    CORE::LINALG::Matrix<3, 3, double> deltatriad(true);
    deltatriad.MultiplyNT(LambdaGP, triad_mat_conv);

    CORE::LINALG::Matrix<4, 1, double> deltaQ(true);
    CORE::LARGEROTATIONS::triadtoquaternion(deltatriad, deltaQ);

    // **************************************

    // extract rotation vector from quaternion
    CORE::LINALG::Matrix<3, 1> deltatheta;
    CORE::LARGEROTATIONS::quaterniontoangle(deltaQ, deltatheta);

    // angular velocity at this Gauss point according to backward Euler scheme
    CORE::LINALG::Matrix<3, 1> omega(true);
    omega.Update(dt_inv, deltatheta);

    // compute matrix Lambda*[gamma(2) 0 0 \\ 0 0 0 \\ 0 0 0]*Lambda^t = gamma(2) * g_1 \otimes g_1
    // where g_1 is first base vector, i.e. first column of Lambda
    CORE::LINALG::Matrix<3, 3> g1g1gamma;
    for (int k = 0; k < 3; k++)
      for (int j = 0; j < 3; j++) g1g1gamma(k, j) = LambdaGP(k, 0) * LambdaGP(j, 0) * gamma(2);

    // compute vector gamma(2) * g_1 \otimes g_1 * \omega
    CORE::LINALG::Matrix<3, 1> g1g1gammaomega;
    g1g1gammaomega.Multiply(g1g1gamma, omega);

    const double jacobifac_gp_weight = jacobiGPdampstoch_[gp] * gausspoints.qwgt[gp];

    if (force != NULL)
    {
      // loop over all nodes
      for (unsigned int inode = 0; inode < nnodecl; inode++)
      {
        // loop over spatial dimensions
        for (unsigned int idim = 0; idim < ndim; idim++)
          (*force)(dofpercombinode * inode + 3 + idim) +=
              g1g1gammaomega(idim) * (I_i[gp])(inode)*jacobifac_gp_weight;
      }
      for (unsigned int inode = nnodecl; inode < nnodetriad; inode++)
      {
        // loop over spatial dimensions
        for (unsigned int idim = 0; idim < ndim; idim++)
          (*force)(dofperclnode * nnodecl + dofpertriadnode * inode + idim) +=
              g1g1gammaomega(idim) * (I_i[gp])(inode)*jacobifac_gp_weight;
      }
    }

    if (stiffmatrix != NULL)
    {
      triad_interpolation_scheme_ptr->GetNodalGeneralizedRotationInterpolationMatrices(
          Itilde, Psi_l, I_i[gp]);

      CORE::LINALG::Matrix<3, 3> g1g1oldgamma;
      for (int k = 0; k < 3; k++)
        for (int j = 0; j < 3; j++)
          g1g1oldgamma(k, j) = LambdaGP(k, 0) * triad_mat_conv(j, 0) * gamma(2);


      // compute matrix gamma(2) * g_1 \otimes g_1 * \omega * Tmat
      CORE::LINALG::Matrix<3, 3> g1g1oldgammaTmat;
      g1g1oldgammaTmat.Multiply(g1g1oldgamma, CORE::LARGEROTATIONS::Tmatrix(deltatheta));

      // compute spin matrix S(\omega)
      CORE::LINALG::Matrix<3, 3> Sofomega;
      CORE::LARGEROTATIONS::computespin(Sofomega, omega);

      // compute matrix gamma(2) * g_1 \otimes g_1 *S(\omega)
      CORE::LINALG::Matrix<3, 3> g1g1gammaSofomega;
      g1g1gammaSofomega.Multiply(g1g1gamma, Sofomega);

      // compute spin matrix S(gamma(2) * g_1 \otimes g_1 *\omega)
      CORE::LINALG::Matrix<3, 3> Sofg1g1gammaomega;
      CORE::LARGEROTATIONS::computespin(Sofg1g1gammaomega, g1g1gammaomega);

      // auxiliary matrices
      CORE::LINALG::Matrix<3, 3> sum(true);
      CORE::LINALG::Matrix<3, 3> auxmatrix(true);

      sum += g1g1oldgammaTmat;
      sum.Scale(dt_inv);
      sum += g1g1gammaSofomega;
      sum -= Sofg1g1gammaomega;

      // loop over first nnodecl row nodes
      for (unsigned int inode = 0; inode < nnodecl; inode++)
      {
        // loop over first nnodecl column nodes
        for (unsigned int jnode = 0; jnode < nnodecl; jnode++)
        {
          auxmatrix.Multiply(sum, Itilde[jnode]);

          // loop over three dimensions in row and column direction
          for (unsigned int idim = 0; idim < ndim; idim++)
            for (unsigned int jdim = 0; jdim < 3; jdim++)
            {
              (*stiffmatrix)(
                  dofpercombinode * inode + 3 + idim, dofpercombinode * jnode + 3 + jdim) +=
                  auxmatrix(idim, jdim) * (I_i[gp])(inode)*jacobifac_gp_weight;
            }
        }
        for (unsigned int jnode = nnodecl; jnode < nnodetriad;
             jnode++)  // this loop is only entered in case of nnodetriad>nnodecl
        {
          auxmatrix.Multiply(sum, Itilde[jnode]);

          // loop over three dimensions in row and column direction
          for (unsigned int idim = 0; idim < ndim; idim++)
            for (unsigned int jdim = 0; jdim < 3; jdim++)
            {
              (*stiffmatrix)(dofpercombinode * inode + 3 + idim,
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
          auxmatrix.Multiply(sum, Itilde[jnode]);

          // loop over three dimensions in row and column direction
          for (unsigned int idim = 0; idim < ndim; idim++)
            for (unsigned int jdim = 0; jdim < 3; jdim++)
            {
              (*stiffmatrix)(dofperclnode * nnodecl + dofpertriadnode * inode + idim,
                  dofpercombinode * jnode + 3 + jdim) +=
                  auxmatrix(idim, jdim) * (I_i[gp])(inode)*jacobifac_gp_weight;
            }
        }
        for (unsigned int jnode = nnodecl; jnode < nnodetriad;
             jnode++)  // this loop is only entered in case of nnodetriad>nnodecl
        {
          auxmatrix.Multiply(sum, Itilde[jnode]);

          // loop over three dimensions in row and column direction
          for (unsigned int idim = 0; idim < ndim; idim++)
            for (unsigned int jdim = 0; jdim < 3; jdim++)
            {
              (*stiffmatrix)(dofperclnode * nnodecl + dofpertriadnode * inode + idim,
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
void DRT::ELEMENTS::Beam3r::EvaluateTranslationalDamping(Teuchos::ParameterList& params,
    const CORE::LINALG::Matrix<ndim * vpernode * nnodecl, 1, double>& vel_centerline,
    const CORE::LINALG::Matrix<ndim * vpernode * nnodecl, 1, double>& disp_totlag_centerline,
    CORE::LINALG::SerialDenseMatrix* stiffmatrix, CORE::LINALG::SerialDenseVector* force) const
{
  /* only nodes for centerline interpolation are considered here (= first nnodecl nodes of this
     element); each of these nodes holds 3*vpernode translational DoFs AND 3 rotational DoFs */
  const unsigned int dofpernode = 3 * vpernode + 3;

  // get time step size
  double dt_inv = 0.0001;
  if (IsParamsInterface())
    dt_inv = 1.0 / ParamsInterface().GetDeltaTime();
  else
    dt_inv = 1.0 / params.get<double>("delta time", 1000);

  // velocity and gradient of background velocity field
  CORE::LINALG::Matrix<ndim, 1> velbackground;
  CORE::LINALG::Matrix<ndim, ndim> velbackgroundgrad;

  // position of beam centerline point corresponding to a certain Gauss point
  CORE::LINALG::Matrix<ndim, 1> r(true);
  // tangent vector (derivative of beam centerline curve r with respect to arc-length parameter s)
  CORE::LINALG::Matrix<ndim, 1> r_s(true);
  // velocity of beam centerline point relative to background fluid velocity
  CORE::LINALG::Matrix<ndim, 1> vel_rel(true);

  // damping coefficients for translational and rotational degrees of freedom
  CORE::LINALG::Matrix<3, 1> gamma(true);
  GetDampingCoefficients(gamma);

  // viscous force vector per unit length at current GP
  CORE::LINALG::Matrix<ndim, 1> f_visc(true);
  // damping matrix
  CORE::LINALG::Matrix<ndim, ndim> damp_mat(true);

  // get Gauss points and weights for evaluation of damping matrix
  CORE::DRT::UTILS::GaussRule1D gaussrule = MyGaussRule(res_damp_stoch);
  CORE::DRT::UTILS::IntegrationPoints1D gausspoints(gaussrule);

  /* vector whose numgp-th element is a 1x(vpernode*nnode)-matrix with all (Lagrange/Hermite) shape
   * functions evaluated at the numgp-th GP
   * these shape functions are used for the interpolation of the beam centerline*/
  std::vector<CORE::LINALG::Matrix<1, vpernode * nnodecl, double>> H_i(gausspoints.nquad);
  // same for the derivatives
  std::vector<CORE::LINALG::Matrix<1, vpernode * nnodecl, double>> H_i_xi(gausspoints.nquad);

  // evaluate all shape functions and derivatives with respect to element parameter xi at all
  // specified Gauss points
  DRT::UTILS::BEAM::EvaluateShapeFunctionsAndDerivsAllGPs<nnodecl, vpernode>(
      gausspoints, H_i, H_i_xi, this->Shape(), this->RefLength());

  for (int gp = 0; gp < gausspoints.nquad; gp++)
  {
    // compute position vector r of point in physical space corresponding to Gauss point
    Calc_r<nnodecl, vpernode, double>(disp_totlag_centerline, H_i[gp], r);

    // compute tangent vector t_{\par}=r' at current Gauss point
    Calc_r_s<nnodecl, vpernode, double>(
        disp_totlag_centerline, H_i_xi[gp], jacobiGPdampstoch_[gp], r_s);

    // compute velocity and gradient of background flow field at point r
    GetBackgroundVelocity<ndim, double>(params, r, velbackground, velbackgroundgrad);

    // compute velocity vector at this Gauss point via same interpolation as for centerline position
    // vector
    DRT::UTILS::BEAM::CalcInterpolation<nnodecl, vpernode, 3, double>(
        vel_centerline, H_i[gp], vel_rel);
    vel_rel -= velbackground;

    // loop over lines and columns of damping matrix
    for (unsigned int idim = 0; idim < ndim; idim++)
      for (unsigned int jdim = 0; jdim < ndim; jdim++)
        damp_mat(idim, jdim) =
            (idim == jdim) * gamma(1) + (gamma(0) - gamma(1)) * r_s(idim) * r_s(jdim);

    // compute viscous force vector per unit length at current GP
    f_visc.Multiply(damp_mat, vel_rel);

    const double jacobifac_gp_weight = jacobiGPdampstoch_[gp] * gausspoints.qwgt[gp];

    if (force != NULL)
    {
      // loop over all nodes used for centerline interpolation
      for (unsigned int inode = 0; inode < nnodecl; inode++)
        // loop over dimensions
        for (unsigned int idim = 0; idim < ndim; idim++)
        {
          (*force)(inode * dofpernode + idim) +=
              H_i[gp](vpernode * inode) * f_visc(idim) * jacobifac_gp_weight;
          if (centerline_hermite_)
            (*force)(inode * dofpernode + 6 + idim) +=
                H_i[gp](vpernode * inode + 1) * f_visc(idim) * jacobifac_gp_weight;
        }
    }

    if (stiffmatrix != NULL)
    {
      // compute matrix product of damping matrix and gradient of background velocity
      CORE::LINALG::Matrix<ndim, ndim> dampmatvelbackgroundgrad(true);
      dampmatvelbackgroundgrad.Multiply(damp_mat, velbackgroundgrad);


      // loop over all nodes used for centerline interpolation
      for (unsigned int inode = 0; inode < nnodecl; inode++)
        // loop over all column nodes used for centerline interpolation
        for (unsigned int jnode = 0; jnode < nnodecl; jnode++)
        {
          for (unsigned int idim = 0; idim < ndim; idim++)
            for (unsigned int jdim = 0; jdim < ndim; jdim++)
            {
              (*stiffmatrix)(inode * dofpernode + idim, jnode * dofpernode + jdim) +=
                  gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) * H_i[gp](vpernode * jnode) *
                  jacobiGPdampstoch_[gp] * damp_mat(idim, jdim) * dt_inv;
              (*stiffmatrix)(inode * dofpernode + idim, jnode * dofpernode + jdim) -=
                  gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) * H_i[gp](vpernode * jnode) *
                  jacobiGPdampstoch_[gp] * dampmatvelbackgroundgrad(idim, jdim);
              (*stiffmatrix)(inode * dofpernode + idim, jnode * dofpernode + idim) +=
                  gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) * H_i_xi[gp](vpernode * jnode) *
                  (gamma(0) - gamma(1)) * r_s(jdim) * vel_rel(jdim);
              (*stiffmatrix)(inode * dofpernode + idim, jnode * dofpernode + jdim) +=
                  gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) * H_i_xi[gp](vpernode * jnode) *
                  (gamma(0) - gamma(1)) * r_s(idim) * vel_rel(jdim);

              if (centerline_hermite_)
              {
                (*stiffmatrix)(inode * dofpernode + 6 + idim, jnode * dofpernode + jdim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i[gp](vpernode * jnode) * jacobiGPdampstoch_[gp] * damp_mat(idim, jdim) *
                    dt_inv;
                (*stiffmatrix)(inode * dofpernode + 6 + idim, jnode * dofpernode + jdim) -=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i[gp](vpernode * jnode) * jacobiGPdampstoch_[gp] *
                    dampmatvelbackgroundgrad(idim, jdim);
                (*stiffmatrix)(inode * dofpernode + 6 + idim, jnode * dofpernode + idim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i_xi[gp](vpernode * jnode) * (gamma(0) - gamma(1)) * r_s(jdim) *
                    vel_rel(jdim);
                (*stiffmatrix)(inode * dofpernode + 6 + idim, jnode * dofpernode + jdim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i_xi[gp](vpernode * jnode) * (gamma(0) - gamma(1)) * r_s(idim) *
                    vel_rel(jdim);

                (*stiffmatrix)(inode * dofpernode + idim, jnode * dofpernode + 6 + jdim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) *
                    H_i[gp](vpernode * jnode + 1) * jacobiGPdampstoch_[gp] * damp_mat(idim, jdim) *
                    dt_inv;
                (*stiffmatrix)(inode * dofpernode + idim, jnode * dofpernode + 6 + jdim) -=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) *
                    H_i[gp](vpernode * jnode + 1) * jacobiGPdampstoch_[gp] *
                    dampmatvelbackgroundgrad(idim, jdim);
                (*stiffmatrix)(inode * dofpernode + idim, jnode * dofpernode + 6 + idim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) *
                    H_i_xi[gp](vpernode * jnode + 1) * (gamma(0) - gamma(1)) * r_s(jdim) *
                    vel_rel(jdim);
                (*stiffmatrix)(inode * dofpernode + idim, jnode * dofpernode + 6 + jdim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode) *
                    H_i_xi[gp](vpernode * jnode + 1) * (gamma(0) - gamma(1)) * r_s(idim) *
                    vel_rel(jdim);

                (*stiffmatrix)(inode * dofpernode + 6 + idim, jnode * dofpernode + 6 + jdim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i[gp](vpernode * jnode + 1) * jacobiGPdampstoch_[gp] * damp_mat(idim, jdim) *
                    dt_inv;
                (*stiffmatrix)(inode * dofpernode + 6 + idim, jnode * dofpernode + 6 + jdim) -=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i[gp](vpernode * jnode + 1) * jacobiGPdampstoch_[gp] *
                    dampmatvelbackgroundgrad(idim, jdim);
                (*stiffmatrix)(inode * dofpernode + 6 + idim, jnode * dofpernode + 6 + idim) +=
                    gausspoints.qwgt[gp] * H_i[gp](vpernode * inode + 1) *
                    H_i_xi[gp](vpernode * jnode + 1) * (gamma(0) - gamma(1)) * r_s(jdim) *
                    vel_rel(jdim);
                (*stiffmatrix)(inode * dofpernode + 6 + idim, jnode * dofpernode + 6 + jdim) +=
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
void DRT::ELEMENTS::Beam3r::EvaluateStochasticForces(Teuchos::ParameterList& params,
    const CORE::LINALG::Matrix<ndim * vpernode * nnodecl, 1, double>& disp_totlag_centerline,
    CORE::LINALG::SerialDenseMatrix* stiffmatrix, CORE::LINALG::SerialDenseVector* force) const
{
  /* only nodes for centerline interpolation are considered here (= first nnodecl nodes of this
     element); each of these nodes holds 3*vpernode translational DoFs AND 3 rotational DoFs */
  const unsigned int dofpernode = 3 * vpernode + 3;

  // damping coefficients for three translational and one rotational degree of freedom
  CORE::LINALG::Matrix<3, 1> sqrt_gamma(true);
  GetDampingCoefficients(sqrt_gamma);
  for (unsigned int i = 0; i < 2; ++i) sqrt_gamma(i) = std::sqrt(sqrt_gamma(i));

  /* get pointer at Epetra multivector in parameter list linking to random numbers for stochastic
   * forces with zero mean and standard deviation (2*kT / dt)^0.5 */
  Teuchos::RCP<Epetra_MultiVector> randomforces = BrownianDynParamsInterface().GetRandomForces();

  // my random number vector at current GP
  CORE::LINALG::Matrix<ndim, 1> randnumvec(true);

  // tangent vector (derivative of beam centerline curve r with respect to arc-length parameter s)
  CORE::LINALG::Matrix<ndim, 1> r_s(true);

  // stochastic force vector per unit length at current GP
  CORE::LINALG::Matrix<ndim, 1> f_stoch(true);

  // get Gauss points and weights for evaluation of damping matrix
  CORE::DRT::UTILS::GaussRule1D gaussrule = MyGaussRule(res_damp_stoch);
  CORE::DRT::UTILS::IntegrationPoints1D gausspoints(gaussrule);

  /* vector whose numgp-th element is a 1x(vpernode*nnode)-matrix with all (Lagrange/Hermite) shape
   * functions evaluated at the numgp-th GP
   * these shape functions are used for the interpolation of the beam centerline*/
  std::vector<CORE::LINALG::Matrix<1, vpernode * nnodecl, double>> H_i(gausspoints.nquad);
  // same for the derivatives
  std::vector<CORE::LINALG::Matrix<1, vpernode * nnodecl, double>> H_i_xi(gausspoints.nquad);

  // evaluate all shape function derivatives with respect to element parameter xi at all specified
  // Gauss points
  DRT::UTILS::BEAM::EvaluateShapeFunctionsAndDerivsAllGPs<nnodecl, vpernode>(
      gausspoints, H_i, H_i_xi, this->Shape(), this->RefLength());


  for (int gp = 0; gp < gausspoints.nquad; gp++)
  {
    // compute tangent vector t_{\par}=r' at current Gauss point
    Calc_r_s<nnodecl, vpernode, double>(
        disp_totlag_centerline, H_i_xi[gp], jacobiGPdampstoch_[gp], r_s);

    // extract random numbers from global vector
    for (unsigned int idim = 0; idim < ndim; idim++)
    {
#ifndef BEAM3RCONSTSTOCHFORCE
      randnumvec(idim) = (*randomforces)[gp * randompergauss + idim][LID()];
#else
      randnumvec(idim) = (*randomforces)[idim][LID()];
#endif
    }

    // compute stochastic force vector per unit length at current GP
    f_stoch.Clear();
    for (unsigned int idim = 0; idim < ndim; idim++)
      for (unsigned int jdim = 0; jdim < ndim; jdim++)
        f_stoch(idim) += (sqrt_gamma(1) * (idim == jdim) +
                             (sqrt_gamma(0) - sqrt_gamma(1)) * r_s(idim) * r_s(jdim)) *
                         randnumvec(jdim);

    const double sqrt_jacobifac_gp_weight =
        std::sqrt(jacobiGPdampstoch_[gp] * gausspoints.qwgt[gp]);

    if (force != NULL)
    {
      // loop over all nodes
      for (unsigned int inode = 0; inode < nnodecl; inode++)
        // loop dimensions with respect to lines
        for (unsigned int idim = 0; idim < ndim; idim++)
        {
          (*force)(inode * dofpernode + idim) -=
              H_i[gp](vpernode * inode) * f_stoch(idim) * sqrt_jacobifac_gp_weight;
          if (centerline_hermite_)
          {
            (*force)(inode * dofpernode + 6 + idim) -=
                H_i[gp](vpernode * inode + 1) * f_stoch(idim) * sqrt_jacobifac_gp_weight;
          }
        }
    }

    if (stiffmatrix != NULL)
    {
      // note: division by sqrt of jacobi factor, because H_i_s = H_i_xi / jacobifactor
      const double sqrt_gp_weight_jacobifac_inv =
          std::sqrt(gausspoints.qwgt[gp] / jacobiGPdampstoch_[gp]);

      const double prefactor = sqrt_gp_weight_jacobifac_inv * (sqrt_gamma(0) - sqrt_gamma(1));

      // loop over all nodes used for centerline interpolation
      for (unsigned int inode = 0; inode < nnodecl; inode++)
        // loop over all column nodes used for centerline interpolation
        for (unsigned int jnode = 0; jnode < nnodecl; jnode++)
        {
          for (unsigned int idim = 0; idim < ndim; idim++)
            for (unsigned int jdim = 0; jdim < ndim; jdim++)
            {
              (*stiffmatrix)(inode * dofpernode + idim, jnode * dofpernode + idim) -=
                  H_i[gp](vpernode * inode) * H_i_xi[gp](vpernode * jnode) * r_s(jdim) *
                  randnumvec(jdim) * prefactor;
              (*stiffmatrix)(inode * dofpernode + idim, jnode * dofpernode + jdim) -=
                  H_i[gp](vpernode * inode) * H_i_xi[gp](vpernode * jnode) * r_s(idim) *
                  randnumvec(jdim) * prefactor;

              if (centerline_hermite_)
              {
                (*stiffmatrix)(inode * dofpernode + 6 + idim, jnode * dofpernode + idim) -=
                    H_i[gp](vpernode * inode + 1) * H_i_xi[gp](vpernode * jnode) * r_s(jdim) *
                    randnumvec(jdim) * prefactor;
                (*stiffmatrix)(inode * dofpernode + 6 + idim, jnode * dofpernode + jdim) -=
                    H_i[gp](vpernode * inode + 1) * H_i_xi[gp](vpernode * jnode) * r_s(idim) *
                    randnumvec(jdim) * prefactor;

                (*stiffmatrix)(inode * dofpernode + idim, jnode * dofpernode + 6 + idim) -=
                    H_i[gp](vpernode * inode) * H_i_xi[gp](vpernode * jnode + 1) * r_s(jdim) *
                    randnumvec(jdim) * prefactor;
                (*stiffmatrix)(inode * dofpernode + idim, jnode * dofpernode + 6 + jdim) -=
                    H_i[gp](vpernode * inode) * H_i_xi[gp](vpernode * jnode + 1) * r_s(idim) *
                    randnumvec(jdim) * prefactor;

                (*stiffmatrix)(inode * dofpernode + 6 + idim, jnode * dofpernode + 6 + idim) -=
                    H_i[gp](vpernode * inode + 1) * H_i_xi[gp](vpernode * jnode + 1) * r_s(jdim) *
                    randnumvec(jdim) * prefactor;
                (*stiffmatrix)(inode * dofpernode + 6 + idim, jnode * dofpernode + 6 + jdim) -=
                    H_i[gp](vpernode * inode + 1) * H_i_xi[gp](vpernode * jnode + 1) * r_s(idim) *
                    randnumvec(jdim) * prefactor;
              }
            }
        }
    }
  }
}


// explicit template instantiations
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<2, 2, 1>(
    Teuchos::ParameterList&, std::vector<double>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseVector*,
    CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<3, 3, 1>(
    Teuchos::ParameterList&, std::vector<double>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseVector*,
    CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<4, 4, 1>(
    Teuchos::ParameterList&, std::vector<double>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseVector*,
    CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<5, 5, 1>(
    Teuchos::ParameterList&, std::vector<double>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseVector*,
    CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<2, 2, 2>(
    Teuchos::ParameterList&, std::vector<double>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseVector*,
    CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<3, 2, 2>(
    Teuchos::ParameterList&, std::vector<double>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseVector*,
    CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<4, 2, 2>(
    Teuchos::ParameterList&, std::vector<double>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseVector*,
    CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<5, 2, 2>(
    Teuchos::ParameterList&, std::vector<double>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseVector*,
    CORE::LINALG::SerialDenseVector*);

template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<2, 2, 1>(
    CORE::LINALG::Matrix<6, 1, double>&, std::vector<CORE::LINALG::Matrix<4, 1, double>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*, CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<3, 3, 1>(
    CORE::LINALG::Matrix<9, 1, double>&, std::vector<CORE::LINALG::Matrix<4, 1, double>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*, CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<4, 4, 1>(
    CORE::LINALG::Matrix<12, 1, double>&, std::vector<CORE::LINALG::Matrix<4, 1, double>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*, CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<5, 5, 1>(
    CORE::LINALG::Matrix<15, 1, double>&, std::vector<CORE::LINALG::Matrix<4, 1, double>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*, CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<2, 2, 2>(
    CORE::LINALG::Matrix<12, 1, double>&, std::vector<CORE::LINALG::Matrix<4, 1, double>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*, CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<3, 2, 2>(
    CORE::LINALG::Matrix<12, 1, double>&, std::vector<CORE::LINALG::Matrix<4, 1, double>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*, CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<4, 2, 2>(
    CORE::LINALG::Matrix<12, 1, double>&, std::vector<CORE::LINALG::Matrix<4, 1, double>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*, CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInternalAndInertiaForcesAndStiff<5, 2, 2>(
    CORE::LINALG::Matrix<12, 1, double>&, std::vector<CORE::LINALG::Matrix<4, 1, double>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*, CORE::LINALG::SerialDenseVector*);

template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<2, 2, 1, double>(
    const CORE::LINALG::Matrix<6, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<12, 1, double>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<3, 3, 1, double>(
    const CORE::LINALG::Matrix<9, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<18, 1, double>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<4, 4, 1, double>(
    const CORE::LINALG::Matrix<12, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<24, 1, double>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<5, 5, 1, double>(
    const CORE::LINALG::Matrix<15, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<30, 1, double>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<2, 2, 2, double>(
    const CORE::LINALG::Matrix<12, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<18, 1, double>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<3, 2, 2, double>(
    const CORE::LINALG::Matrix<12, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<21, 1, double>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<4, 2, 2, double>(
    const CORE::LINALG::Matrix<12, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<24, 1, double>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<5, 2, 2, double>(
    const CORE::LINALG::Matrix<12, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::Matrix<27, 1, double>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<2, 2, 1, Sacado::Fad::DFad<double>>(
    const CORE::LINALG::Matrix<6, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::Matrix<12, 1, Sacado::Fad::DFad<double>>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<3, 3, 1, Sacado::Fad::DFad<double>>(
    const CORE::LINALG::Matrix<9, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::Matrix<18, 1, Sacado::Fad::DFad<double>>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<4, 4, 1, Sacado::Fad::DFad<double>>(
    const CORE::LINALG::Matrix<12, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::Matrix<24, 1, Sacado::Fad::DFad<double>>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<5, 5, 1, Sacado::Fad::DFad<double>>(
    const CORE::LINALG::Matrix<15, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::Matrix<30, 1, Sacado::Fad::DFad<double>>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<2, 2, 2, Sacado::Fad::DFad<double>>(
    const CORE::LINALG::Matrix<12, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::Matrix<18, 1, Sacado::Fad::DFad<double>>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<3, 2, 2, Sacado::Fad::DFad<double>>(
    const CORE::LINALG::Matrix<12, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::Matrix<21, 1, Sacado::Fad::DFad<double>>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<4, 2, 2, Sacado::Fad::DFad<double>>(
    const CORE::LINALG::Matrix<12, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::Matrix<24, 1, Sacado::Fad::DFad<double>>&);
template void DRT::ELEMENTS::Beam3r::CalcInternalForceAndStiff<5, 2, 2, Sacado::Fad::DFad<double>>(
    const CORE::LINALG::Matrix<12, 1, Sacado::Fad::DFad<double>>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, Sacado::Fad::DFad<double>>>&,
    CORE::LINALG::SerialDenseMatrix*, CORE::LINALG::Matrix<27, 1, Sacado::Fad::DFad<double>>&);

template void DRT::ELEMENTS::Beam3r::CalcInertiaForceAndMassMatrix<2, 2, 1>(
    const CORE::LINALG::Matrix<6, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInertiaForceAndMassMatrix<3, 3, 1>(
    const CORE::LINALG::Matrix<9, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInertiaForceAndMassMatrix<4, 4, 1>(
    const CORE::LINALG::Matrix<12, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInertiaForceAndMassMatrix<5, 5, 1>(
    const CORE::LINALG::Matrix<15, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInertiaForceAndMassMatrix<2, 2, 2>(
    const CORE::LINALG::Matrix<12, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInertiaForceAndMassMatrix<3, 2, 2>(
    const CORE::LINALG::Matrix<12, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInertiaForceAndMassMatrix<4, 2, 2>(
    const CORE::LINALG::Matrix<12, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*);
template void DRT::ELEMENTS::Beam3r::CalcInertiaForceAndMassMatrix<5, 2, 2>(
    const CORE::LINALG::Matrix<12, 1, double>&,
    const std::vector<CORE::LINALG::Matrix<4, 1, double>>&, CORE::LINALG::SerialDenseMatrix*,
    CORE::LINALG::SerialDenseVector*);

template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticForceContributions<2, 2, 1>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&, const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<2, double>&,
    const CORE::LINALG::Matrix<1, 2, double>&, const CORE::LINALG::Matrix<1, 2, double>&,
    const double, const double) const;
template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticForceContributions<3, 3, 1>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&, const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<3, double>&,
    const CORE::LINALG::Matrix<1, 3, double>&, const CORE::LINALG::Matrix<1, 3, double>&,
    const double, const double) const;
template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticForceContributions<4, 4, 1>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&, const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<4, double>&,
    const CORE::LINALG::Matrix<1, 4, double>&, const CORE::LINALG::Matrix<1, 4, double>&,
    const double, const double) const;
template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticForceContributions<5, 5, 1>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&, const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<5, double>&,
    const CORE::LINALG::Matrix<1, 5, double>&, const CORE::LINALG::Matrix<1, 5, double>&,
    const double, const double) const;
template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticForceContributions<2, 2, 2>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&, const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<2, double>&,
    const CORE::LINALG::Matrix<1, 2, double>&, const CORE::LINALG::Matrix<1, 4, double>&,
    const double, const double) const;
template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticForceContributions<3, 2, 2>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&, const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<3, double>&,
    const CORE::LINALG::Matrix<1, 3, double>&, const CORE::LINALG::Matrix<1, 4, double>&,
    const double, const double) const;
template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticForceContributions<4, 2, 2>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&, const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<4, double>&,
    const CORE::LINALG::Matrix<1, 4, double>&, const CORE::LINALG::Matrix<1, 4, double>&,
    const double, const double) const;
template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticForceContributions<5, 2, 2>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&, const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<5, double>&,
    const CORE::LINALG::Matrix<1, 5, double>&, const CORE::LINALG::Matrix<1, 4, double>&,
    const double, const double) const;

template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticMomentContributions<2, 2, 1>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<2, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<1, 2, double>&, const CORE::LINALG::Matrix<1, 2, double>&,
    const double, const double) const;
template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticMomentContributions<3, 3, 1>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<3, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<1, 3, double>&, const CORE::LINALG::Matrix<1, 3, double>&,
    const double, const double) const;
template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticMomentContributions<4, 4, 1>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<4, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<1, 4, double>&, const CORE::LINALG::Matrix<1, 4, double>&,
    const double, const double) const;
template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticMomentContributions<5, 5, 1>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<5, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<1, 5, double>&, const CORE::LINALG::Matrix<1, 5, double>&,
    const double, const double) const;
template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticMomentContributions<2, 2, 2>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<2, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<1, 2, double>&, const CORE::LINALG::Matrix<1, 2, double>&,
    const double, const double) const;
template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticMomentContributions<3, 2, 2>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<3, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<1, 3, double>&, const CORE::LINALG::Matrix<1, 3, double>&,
    const double, const double) const;
template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticMomentContributions<4, 2, 2>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<4, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<1, 4, double>&, const CORE::LINALG::Matrix<1, 4, double>&,
    const double, const double) const;
template void DRT::ELEMENTS::Beam3r::CalcStiffmatAnalyticMomentContributions<5, 2, 2>(
    CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&,
    const LARGEROTATIONS::TriadInterpolationLocalRotationVectors<5, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<1, 5, double>&, const CORE::LINALG::Matrix<1, 5, double>&,
    const double, const double) const;

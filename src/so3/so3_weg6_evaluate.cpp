/*----------------------------------------------------------------------*/
/*! \file
\brief Solid Wedge6 Element
\level 1


*----------------------------------------------------------------------*/
#include "so3_weg6.H"
#include "so3_prestress.H"
#include "lib_discret.H"
#include "lib_utils.H"
#include "lib_dserror.H"
#include "lib_prestress_service.H"
#include "linalg_utils_densematrix_inverse.H"
#include "linalg_utils_densematrix_eigen.H"
#include "linalg_serialdensematrix.H"
#include "linalg_serialdensevector.H"
#include "discretization_fem_general_utils_integration.H"
#include "discretization_fem_general_utils_fem_shapefunctions.H"
#include "mat_elasthyper.H"
#include "mat_constraintmixture.H"
#include "mat_so3_material.H"
#include "lib_globalproblem.H"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include "structure_new_elements_paramsinterface.H"

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                              maf 04/07|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::So_weg6::Evaluate(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, std::vector<int>& lm,
    Epetra_SerialDenseMatrix& elemat1_epetra, Epetra_SerialDenseMatrix& elemat2_epetra,
    Epetra_SerialDenseVector& elevec1_epetra, Epetra_SerialDenseVector& elevec2_epetra,
    Epetra_SerialDenseVector& elevec3_epetra)
{
  // Check whether the solid material PostSetup() routine has already been called and call it if not
  EnsureMaterialPostSetup(params);

  LINALG::Matrix<NUMDOF_WEG6, NUMDOF_WEG6> elemat1(elemat1_epetra.A(), true);
  LINALG::Matrix<NUMDOF_WEG6, NUMDOF_WEG6> elemat2(elemat2_epetra.A(), true);
  LINALG::Matrix<NUMDOF_WEG6, 1> elevec1(elevec1_epetra.A(), true);
  LINALG::Matrix<NUMDOF_WEG6, 1> elevec2(elevec2_epetra.A(), true);
  LINALG::Matrix<NUMDOF_WEG6, 1> elevec3(elevec3_epetra.A(), true);

  // start with "none"
  DRT::ELEMENTS::So_weg6::ActionType act = So_weg6::none;

  // get the required action
  std::string action = params.get<std::string>("action", "none");
  if (action == "none")
    dserror("No action supplied");
  else if (action == "calc_struct_linstiff")
    act = So_weg6::calc_struct_linstiff;
  else if (action == "calc_struct_nlnstiff")
    act = So_weg6::calc_struct_nlnstiff;
  else if (action == "calc_struct_internalforce")
    act = So_weg6::calc_struct_internalforce;
  else if (action == "calc_struct_linstiffmass")
    act = So_weg6::calc_struct_linstiffmass;
  else if (action == "calc_struct_nlnstiffmass")
    act = So_weg6::calc_struct_nlnstiffmass;
  else if (action == "calc_struct_stress")
    act = So_weg6::calc_struct_stress;
  else if (action == "calc_struct_eleload")
    act = So_weg6::calc_struct_eleload;
  else if (action == "calc_struct_fsiload")
    act = So_weg6::calc_struct_fsiload;
  else if (action == "calc_struct_update_istep")
    act = So_weg6::calc_struct_update_istep;
  else if (action == "calc_struct_reset_istep")
    act = So_weg6::calc_struct_reset_istep;
  else if (action == "calc_struct_reset_all")
    act = So_weg6::calc_struct_reset_all;
  else if (action == "calc_struct_energy")
    act = So_weg6::calc_struct_energy;
  else if (action == "calc_struct_prestress_update")
    act = So_weg6::prestress_update;
  else if (action == "calc_global_gpstresses_map")
    act = So_weg6::calc_global_gpstresses_map;
  else if (action == "calc_struct_recover")
    return 0;
  else if (action == "calc_struct_predict")
    return 0;
  else
    dserror("Unknown type of action for So_weg6");

  // what should the element do
  switch (act)
  {
    //==================================================================================
    // linear stiffness
    case calc_struct_linstiff:
    {
      // need current displacement and residual forces
      std::vector<double> mydisp(lm.size());
      for (double& i : mydisp) i = 0.0;
      std::vector<double> myres(lm.size());
      for (double& myre : myres) myre = 0.0;
      std::vector<double> mydispmat(lm.size(), 0.0);

      sow6_nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, mydispmat, &elemat1, nullptr, &elevec1,
          nullptr, nullptr, nullptr, nullptr, params, INPAR::STR::stress_none,
          INPAR::STR::strain_none);
    }
    break;

    //==================================================================================
    // nonlinear stiffness and internal force vector
    case calc_struct_nlnstiff:
    {
      // need current displacement and residual forces
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
      Teuchos::RCP<const Epetra_Vector> res = discretization.GetState("residual displacement");
      if (disp == Teuchos::null || res == Teuchos::null)
        dserror("Cannot get state vectors 'displacement' and/or residual");
      std::vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res, myres, lm);
      std::vector<double> mydispmat(lm.size(), 0.0);

      sow6_nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, mydispmat, &elemat1, nullptr, &elevec1,
          nullptr, nullptr, nullptr, nullptr, params, INPAR::STR::stress_none,
          INPAR::STR::strain_none);
    }
    break;

    //==================================================================================
    // internal force vector only
    case calc_struct_internalforce:
    {
      // need current displacement and residual forces
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
      Teuchos::RCP<const Epetra_Vector> res = discretization.GetState("residual displacement");
      if (disp == Teuchos::null || res == Teuchos::null)
        dserror("Cannot get state vectors 'displacement' and/or residual");
      std::vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res, myres, lm);
      // create a dummy element matrix to apply linearised EAS-stuff onto
      LINALG::Matrix<NUMDOF_WEG6, NUMDOF_WEG6> myemat(true);  // set to zero
      std::vector<double> mydispmat(lm.size(), 0.0);

      sow6_nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, mydispmat, &myemat, nullptr, &elevec1,
          nullptr, nullptr, nullptr, nullptr, params, INPAR::STR::stress_none,
          INPAR::STR::strain_none);
    }
    break;

    //==================================================================================
    // linear stiffness and consistent mass matrix
    case calc_struct_linstiffmass:
      dserror("Case 'calc_struct_linstiffmass' not yet implemented");
      break;

    //==================================================================================
    // nonlinear stiffness, internal force vector, and consistent mass matrix
    case calc_struct_nlnstiffmass:
    {
      // need current displacement and residual forces
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
      Teuchos::RCP<const Epetra_Vector> res = discretization.GetState("residual displacement");
      // need current velocities and accelerations (for non constant mass matrix)
      Teuchos::RCP<const Epetra_Vector> vel = discretization.GetState("velocity");
      Teuchos::RCP<const Epetra_Vector> acc = discretization.GetState("acceleration");
      if (disp == Teuchos::null || res == Teuchos::null)
        dserror("Cannot get state vectors 'displacement' and/or residual");
      if (vel == Teuchos::null) dserror("Cannot get state vectors 'velocity'");
      if (acc == Teuchos::null) dserror("Cannot get state vectors 'acceleration'");

      std::vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp, mydisp, lm);
      std::vector<double> myvel(lm.size());
      DRT::UTILS::ExtractMyValues(*vel, myvel, lm);
      std::vector<double> myacc(lm.size());
      DRT::UTILS::ExtractMyValues(*acc, myacc, lm);
      std::vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res, myres, lm);
      std::vector<double> mydispmat(lm.size(), 0.0);

      sow6_nlnstiffmass(lm, mydisp, &myvel, &myacc, myres, mydispmat, &elemat1, &elemat2, &elevec1,
          &elevec2, &elevec3, nullptr, nullptr, params, INPAR::STR::stress_none,
          INPAR::STR::strain_none);
    }
    break;

    //==================================================================================
    // evaluate stresses and strains at gauss points
    case calc_struct_stress:
    {
      // nothing to do for ghost elements
      if (discretization.Comm().MyPID() == Owner())
      {
        Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
        Teuchos::RCP<const Epetra_Vector> res = discretization.GetState("residual displacement");
        Teuchos::RCP<std::vector<char>> stressdata =
            params.get<Teuchos::RCP<std::vector<char>>>("stress", Teuchos::null);
        Teuchos::RCP<std::vector<char>> straindata =
            params.get<Teuchos::RCP<std::vector<char>>>("strain", Teuchos::null);
        if (disp == Teuchos::null) dserror("Cannot get state vectors 'displacement'");
        if (stressdata == Teuchos::null) dserror("Cannot get stress 'data'");
        if (straindata == Teuchos::null) dserror("Cannot get strain 'data'");
        std::vector<double> mydisp(lm.size());
        DRT::UTILS::ExtractMyValues(*disp, mydisp, lm);
        std::vector<double> myres(lm.size());
        DRT::UTILS::ExtractMyValues(*res, myres, lm);
        LINALG::Matrix<NUMGPT_WEG6, MAT::NUM_STRESS_3D> stress;
        LINALG::Matrix<NUMGPT_WEG6, MAT::NUM_STRESS_3D> strain;
        auto iostress =
            DRT::INPUT::get<INPAR::STR::StressType>(params, "iostress", INPAR::STR::stress_none);
        auto iostrain =
            DRT::INPUT::get<INPAR::STR::StrainType>(params, "iostrain", INPAR::STR::strain_none);

        std::vector<double> mydispmat(lm.size(), 0.0);

        sow6_nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, mydispmat, nullptr, nullptr, nullptr,
            nullptr, nullptr, &stress, &strain, params, iostress, iostrain);

        {
          DRT::PackBuffer data;
          AddtoPack(data, stress);
          data.StartPacking();
          AddtoPack(data, stress);
          std::copy(data().begin(), data().end(), std::back_inserter(*stressdata));
        }
        {
          DRT::PackBuffer data;
          AddtoPack(data, strain);
          data.StartPacking();
          AddtoPack(data, strain);
          std::copy(data().begin(), data().end(), std::back_inserter(*straindata));
        }
      }
    }
    break;
    case calc_struct_energy:
    {
      // check length of elevec1
      if (elevec1_epetra.Length() < 1) dserror("The given result vector is too short.");

      // initialization of internal energy
      double intenergy = 0.0;
      /* ============================================================================*
      ** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for Wedge_6 with 6 GAUSS POINTS*
      ** ============================================================================*/
      const static std::vector<LINALG::Matrix<NUMNOD_WEG6, 1>> shapefcts = sow6_shapefcts();
      const static std::vector<LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6>> derivs = sow6_derivs();
      const static std::vector<double> gpweights = sow6_weights();
      /* ============================================================================*/


      // get displacements of this processor
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
      if (disp == Teuchos::null) dserror("Cannot get state displacement vector");

      // get displacements of this element
      std::vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp, mydisp, lm);

      // update element geometry
      LINALG::Matrix<NUMNOD_WEG6, NUMDIM_WEG6> xrefe;  // material coord. of element
      LINALG::Matrix<NUMNOD_WEG6, NUMDIM_WEG6> xcurr;  // current  coord. of element
      LINALG::Matrix<NUMNOD_WEG6, NUMDIM_WEG6> xdisp;

      DRT::Node** nodes = Nodes();
      for (int i = 0; i < NUMNOD_WEG6; ++i)
      {
        const double* x = nodes[i]->X();
        xrefe(i, 0) = x[0];
        xrefe(i, 1) = x[1];
        xrefe(i, 2) = x[2];

        xcurr(i, 0) = xrefe(i, 0) + mydisp[i * NODDOF_WEG6 + 0];
        xcurr(i, 1) = xrefe(i, 1) + mydisp[i * NODDOF_WEG6 + 1];
        xcurr(i, 2) = xrefe(i, 2) + mydisp[i * NODDOF_WEG6 + 2];

        if (::UTILS::PRESTRESS::IsMulf(pstype_))
        {
          xdisp(i, 0) = mydisp[i * NODDOF_WEG6 + 0];
          xdisp(i, 1) = mydisp[i * NODDOF_WEG6 + 1];
          xdisp(i, 2) = mydisp[i * NODDOF_WEG6 + 2];
        }
      }

      /* =========================================================================*/
      /* ================================================= Loop over Gauss Points */
      /* =========================================================================*/
      for (int gp = 0; gp < NUMGPT_WEG6; ++gp)
      {
        /* get the inverse of the Jacobian matrix which looks like:
        **            [ x_,r  y_,r  z_,r ]^-1
        **     J^-1 = [ x_,s  y_,s  z_,s ]
        **            [ x_,t  y_,t  z_,t ]
        */
        LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6> N_XYZ;
        // compute derivatives N_XYZ at gp w.r.t. material coordinates
        // by N_XYZ = J^-1 * N_rst
        N_XYZ.Multiply(invJ_[gp], derivs[gp]);

        // Gauss weights and Jacobian determinant
        double fac = detJ_[gp] * gpweights[gp];

        LINALG::Matrix<NUMDIM_WEG6, NUMDIM_WEG6> defgrd(false);

        if (::UTILS::PRESTRESS::IsMulf(pstype_))
        {
          // get Jacobian mapping wrt to the stored configuration
          LINALG::Matrix<3, 3> invJdef;
          prestress_->StoragetoMatrix(gp, invJdef, prestress_->JHistory());
          // get derivatives wrt to last spatial configuration
          LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6> N_xyz;
          N_xyz.Multiply(invJdef, derivs[gp]);

          // build multiplicative incremental defgrd
          defgrd.MultiplyTT(xdisp, N_xyz);
          defgrd(0, 0) += 1.0;
          defgrd(1, 1) += 1.0;
          defgrd(2, 2) += 1.0;

          // get stored old incremental F
          LINALG::Matrix<3, 3> Fhist;
          prestress_->StoragetoMatrix(gp, Fhist, prestress_->FHistory());

          // build total defgrd = delta F * F_old
          LINALG::Matrix<3, 3> Fnew;
          Fnew.Multiply(defgrd, Fhist);
          defgrd = Fnew;
        }
        else
          // (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
          defgrd.MultiplyTT(xcurr, N_XYZ);

        // Right Cauchy-Green tensor = F^T * F
        LINALG::Matrix<NUMDIM_WEG6, NUMDIM_WEG6> cauchygreen;
        cauchygreen.MultiplyTN(defgrd, defgrd);

        // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
        // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
        LINALG::Matrix<6, 1> glstrain(false);
        glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
        glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
        glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
        glstrain(3) = cauchygreen(0, 1);
        glstrain(4) = cauchygreen(1, 2);
        glstrain(5) = cauchygreen(2, 0);



        // call material for evaluation of strain energy function
        double psi = 0.0;
        SolidMaterial()->StrainEnergy(glstrain, psi, gp, Id());

        // sum up GP contribution to internal energy
        intenergy += fac * psi;
      }

      // return result
      elevec1_epetra(0) = intenergy;
    }
    break;

    //==================================================================================
    case calc_struct_eleload:
      dserror("this method is not supposed to evaluate a load, use EvaluateNeumann(...)");
      break;

    //==================================================================================
    case calc_struct_fsiload:
      dserror("Case not yet implemented");
      break;

    //==================================================================================
    case calc_struct_update_istep:
    {
      SolidMaterial()->Update();
    }
    break;

    //==================================================================================
    case calc_struct_reset_istep:
    {
      // Reset of history (if needed)
      SolidMaterial()->ResetStep();
    }
    break;

    //==================================================================================
    case calc_struct_reset_all:
    {
      // Reset of history for materials
      SolidMaterial()->ResetAll(NUMGPT_WEG6);

      // Reset prestress
      if (::UTILS::PRESTRESS::IsMulf(pstype_))
      {
        time_ = 0.0;
        LINALG::Matrix<3, 3> Id(true);
        Id(0, 0) = Id(1, 1) = Id(2, 2) = 1.0;
        for (int gp = 0; gp < NUMGPT_WEG6; ++gp)
        {
          prestress_->MatrixtoStorage(gp, Id, prestress_->FHistory());
          prestress_->MatrixtoStorage(gp, invJ_[gp], prestress_->JHistory());
        }
      }
    }
    break;

    //==================================================================================
    // in case of prestressing, make a snapshot of the current green-Lagrange strains and add them
    // to the previously stored GL strains in an incremental manner
    case prestress_update:
    {
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
      if (disp == Teuchos::null) dserror("Cannot get displacement state");
      std::vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp, mydisp, lm);

      // build def gradient for every gauss point
      LINALG::SerialDenseMatrix gpdefgrd(NUMGPT_WEG6, 9);
      DefGradient(mydisp, gpdefgrd, *prestress_);

      // update deformation gradient and put back to storage
      LINALG::Matrix<3, 3> deltaF;
      LINALG::Matrix<3, 3> Fhist;
      LINALG::Matrix<3, 3> Fnew;
      for (int gp = 0; gp < NUMGPT_WEG6; ++gp)
      {
        prestress_->StoragetoMatrix(gp, deltaF, gpdefgrd);
        prestress_->StoragetoMatrix(gp, Fhist, prestress_->FHistory());
        Fnew.Multiply(deltaF, Fhist);
        prestress_->MatrixtoStorage(gp, Fnew, prestress_->FHistory());
      }

      // push-forward invJ for every gaussian point
      UpdateJacobianMapping(mydisp, *prestress_);

      // Update constraintmixture material
      if (Material()->MaterialType() == INPAR::MAT::m_constraintmixture)
      {
        SolidMaterial()->Update();
      }
    }
    break;

    //==================================================================================
    // evaluate stresses and strains at gauss points and store gpstresses in map <EleId, gpstresses
    // >
    case calc_global_gpstresses_map:
    {
      // nothing to do for ghost elements
      if (discretization.Comm().MyPID() == Owner())
      {
        Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
        Teuchos::RCP<const Epetra_Vector> res = discretization.GetState("residual displacement");
        Teuchos::RCP<std::vector<char>> stressdata =
            params.get<Teuchos::RCP<std::vector<char>>>("stress", Teuchos::null);
        Teuchos::RCP<std::vector<char>> straindata =
            params.get<Teuchos::RCP<std::vector<char>>>("strain", Teuchos::null);
        if (disp == Teuchos::null) dserror("Cannot get state vectors 'displacement'");
        if (stressdata == Teuchos::null) dserror("Cannot get 'stress' data");
        if (straindata == Teuchos::null) dserror("Cannot get 'strain' data");
        const Teuchos::RCP<std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix>>> gpstressmap =
            params.get<Teuchos::RCP<std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix>>>>(
                "gpstressmap", Teuchos::null);
        if (gpstressmap == Teuchos::null)
          dserror("no gp stress map available for writing gpstresses");
        const Teuchos::RCP<std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix>>> gpstrainmap =
            params.get<Teuchos::RCP<std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix>>>>(
                "gpstrainmap", Teuchos::null);
        if (gpstrainmap == Teuchos::null)
          dserror("no gp strain map available for writing gpstrains");
        std::vector<double> mydisp(lm.size());
        DRT::UTILS::ExtractMyValues(*disp, mydisp, lm);
        std::vector<double> myres(lm.size());
        DRT::UTILS::ExtractMyValues(*res, myres, lm);
        LINALG::Matrix<NUMGPT_WEG6, MAT::NUM_STRESS_3D> stress;
        LINALG::Matrix<NUMGPT_WEG6, MAT::NUM_STRESS_3D> strain;
        auto iostress =
            DRT::INPUT::get<INPAR::STR::StressType>(params, "iostress", INPAR::STR::stress_none);
        auto iostrain =
            DRT::INPUT::get<INPAR::STR::StrainType>(params, "iostrain", INPAR::STR::strain_none);

        std::vector<double> mydispmat(lm.size(), 0.0);

        // if a linear analysis is desired
        if (kintype_ == INPAR::STR::kinem_linear)
        {
          dserror("Linear case not implemented");
        }
        else
        {
          sow6_nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, mydispmat, nullptr, nullptr,
              nullptr, nullptr, nullptr, &stress, &strain, params, iostress, iostrain);
        }
        // add stresses to global map
        // get EleID Id()
        int gid = Id();
        Teuchos::RCP<Epetra_SerialDenseMatrix> gpstress =
            Teuchos::rcp(new Epetra_SerialDenseMatrix);
        gpstress->Shape(NUMGPT_WEG6, MAT::NUM_STRESS_3D);

        // move stresses to serial dense matrix
        for (int i = 0; i < NUMGPT_WEG6; i++)
        {
          for (int j = 0; j < MAT::NUM_STRESS_3D; j++)
          {
            (*gpstress)(i, j) = stress(i, j);
          }
        }

        // strains
        Teuchos::RCP<Epetra_SerialDenseMatrix> gpstrain =
            Teuchos::rcp(new Epetra_SerialDenseMatrix);
        gpstrain->Shape(NUMGPT_WEG6, MAT::NUM_STRESS_3D);

        // move stresses to serial dense matrix
        for (int i = 0; i < NUMGPT_WEG6; i++)
        {
          for (int j = 0; j < MAT::NUM_STRESS_3D; j++)
          {
            (*gpstrain)(i, j) = strain(i, j);
          }
        }

        // add to map
        (*gpstressmap)[gid] = gpstress;
        (*gpstrainmap)[gid] = gpstrain;

        {
          DRT::PackBuffer data;
          AddtoPack(data, stress);
          data.StartPacking();
          AddtoPack(data, stress);
          std::copy(data().begin(), data().end(), std::back_inserter(*stressdata));
        }

        {
          DRT::PackBuffer data;
          AddtoPack(data, strain);
          data.StartPacking();
          AddtoPack(data, strain);
          std::copy(data().begin(), data().end(), std::back_inserter(*straindata));
        }
      }
    }
    break;


    default:
      dserror("Unknown type of action for Solid3");
      break;
  }
  return 0;
}



/*----------------------------------------------------------------------*
 |  Integrate a Volume Neumann boundary condition (public)     maf 04/07|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::So_weg6::EvaluateNeumann(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Condition& condition, std::vector<int>& lm,
    Epetra_SerialDenseVector& elevec1, Epetra_SerialDenseMatrix* elemat1)
{
  dserror("Body force of wedge6 not implemented");
  return 0;
}

/*----------------------------------------------------------------------*
 |  init the element jacobian mapping (protected)              gee 04/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_weg6::InitJacobianMapping()
{
  /* pointer to (static) shape function array
   * for each node, evaluated at each gp*/
  LINALG::Matrix<NUMNOD_WEG6, NUMGPT_WEG6>* shapefct;
  /* pointer to (static) shape function derivatives array
   * for each node wrt to each direction, evaluated at each gp*/
  LINALG::Matrix<NUMGPT_WEG6 * NUMDIM_WEG6, NUMNOD_WEG6>* deriv;
  /* pointer to (static) weight factors at each gp */
  LINALG::Matrix<NUMGPT_WEG6, 1>* weights;
  sow6_shapederiv(&shapefct, &deriv, &weights);  // call to evaluate

  LINALG::Matrix<NUMNOD_WEG6, NUMDIM_WEG6> xrefe;
  for (int i = 0; i < NUMNOD_WEG6; ++i)
  {
    xrefe(i, 0) = Nodes()[i]->X()[0];
    xrefe(i, 1) = Nodes()[i]->X()[1];
    xrefe(i, 2) = Nodes()[i]->X()[2];
  }
  invJ_.resize(NUMGPT_WEG6);
  detJ_.resize(NUMGPT_WEG6);
  LINALG::Matrix<NUMDIM_WEG6, NUMGPT_WEG6> deriv_gp;
  for (int gp = 0; gp < NUMGPT_WEG6; ++gp)
  {
    // get submatrix of deriv at actual gp
    for (int m = 0; m < NUMDIM_WEG6; ++m)
      for (int n = 0; n < NUMGPT_WEG6; ++n) deriv_gp(m, n) = (*deriv)(NUMDIM_WEG6 * gp + m, n);

    invJ_[gp].Multiply(deriv_gp, xrefe);
    detJ_[gp] = invJ_[gp].Invert();
    if (detJ_[gp] <= 0.0) dserror("Element Jacobian mapping %10.5e <= 0.0", detJ_[gp]);

    if (::UTILS::PRESTRESS::IsMulfActive(time_, pstype_, pstime_))
      if (!(prestress_->IsInit()))
        prestress_->MatrixtoStorage(gp, invJ_[gp], prestress_->JHistory());

  }  // for (int gp=0; gp<NUMGPT_WEG6; ++gp)

  if (::UTILS::PRESTRESS::IsMulfActive(time_, pstype_, pstime_)) prestress_->IsInit() = true;
}

/*----------------------------------------------------------------------*
 |  evaluate the element (private)                             maf 04/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_weg6::sow6_nlnstiffmass(std::vector<int>& lm,  // location matrix
    std::vector<double>& disp,                                        // current displacements
    std::vector<double>* vel,                                         // current velocities
    std::vector<double>* acc,                                         // current accelerations
    std::vector<double>& residual,                                    // current residual displ
    std::vector<double>& dispmat,                                // current material displacements
    LINALG::Matrix<NUMDOF_WEG6, NUMDOF_WEG6>* stiffmatrix,       // element stiffness matrix
    LINALG::Matrix<NUMDOF_WEG6, NUMDOF_WEG6>* massmatrix,        // element mass matrix
    LINALG::Matrix<NUMDOF_WEG6, 1>* force,                       // element internal force vector
    LINALG::Matrix<NUMDOF_WEG6, 1>* forceinert,                  // element inertial force vector
    LINALG::Matrix<NUMDOF_WEG6, 1>* force_str,                   // element structural force vector
    LINALG::Matrix<NUMGPT_WEG6, MAT::NUM_STRESS_3D>* elestress,  // stresses at GP
    LINALG::Matrix<NUMGPT_WEG6, MAT::NUM_STRESS_3D>* elestrain,  // strains at GP
    Teuchos::ParameterList& params,                              // algorithmic parameters e.g. time
    const INPAR::STR::StressType iostress,                       // stress output option
    const INPAR::STR::StrainType iostrain)                       // strain output option
{
  /* ============================================================================*
  ** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for Wedge_6 with 6 GAUSS POINTS*
  ** ============================================================================*/
  const static std::vector<LINALG::Matrix<NUMNOD_WEG6, 1>> shapefcts = sow6_shapefcts();
  const static std::vector<LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6>> derivs = sow6_derivs();
  const static std::vector<double> gpweights = sow6_weights();
  /* ============================================================================*/

  // update element geometry
  LINALG::Matrix<NUMNOD_WEG6, NUMDIM_WEG6> xrefe;  // material coord. of element
  LINALG::Matrix<NUMNOD_WEG6, NUMDIM_WEG6> xcurr;  // current  coord. of element
  LINALG::Matrix<NUMNOD_WEG6, NUMDIM_WEG6> xdisp;

  DRT::Node** nodes = Nodes();
  for (int i = 0; i < NUMNOD_WEG6; ++i)
  {
    const double* x = nodes[i]->X();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];

    xcurr(i, 0) = xrefe(i, 0) + disp[i * NODDOF_WEG6 + 0];
    xcurr(i, 1) = xrefe(i, 1) + disp[i * NODDOF_WEG6 + 1];
    xcurr(i, 2) = xrefe(i, 2) + disp[i * NODDOF_WEG6 + 2];

    if (::UTILS::PRESTRESS::IsMulf(pstype_))
    {
      xdisp(i, 0) = disp[i * NODDOF_WEG6 + 0];
      xdisp(i, 1) = disp[i * NODDOF_WEG6 + 1];
      xdisp(i, 2) = disp[i * NODDOF_WEG6 + 2];
    }
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  for (int gp = 0; gp < NUMGPT_WEG6; ++gp)
  {
    /* get the inverse of the Jacobian matrix which looks like:
    **            [ x_,r  y_,r  z_,r ]^-1
    **     J^-1 = [ x_,s  y_,s  z_,s ]
    **            [ x_,t  y_,t  z_,t ]
    */
    LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6> N_XYZ;
    // compute derivatives N_XYZ at gp w.r.t. material coordinates
    // by N_XYZ = J^-1 * N_rst
    N_XYZ.Multiply(invJ_[gp], derivs[gp]);
    double detJ = detJ_[gp];

    LINALG::Matrix<NUMDIM_WEG6, NUMDIM_WEG6> defgrd(false);

    if (::UTILS::PRESTRESS::IsMulf(pstype_))
    {
      // get Jacobian mapping wrt to the stored configuration
      LINALG::Matrix<3, 3> invJdef;
      prestress_->StoragetoMatrix(gp, invJdef, prestress_->JHistory());
      // get derivatives wrt to last spatial configuration
      LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6> N_xyz;
      N_xyz.Multiply(invJdef, derivs[gp]);

      // build multiplicative incremental defgrd
      defgrd.MultiplyTT(xdisp, N_xyz);
      defgrd(0, 0) += 1.0;
      defgrd(1, 1) += 1.0;
      defgrd(2, 2) += 1.0;

      // get stored old incremental F
      LINALG::Matrix<3, 3> Fhist;
      prestress_->StoragetoMatrix(gp, Fhist, prestress_->FHistory());

      // build total defgrd = delta F * F_old
      LINALG::Matrix<3, 3> Fnew;
      Fnew.Multiply(defgrd, Fhist);
      defgrd = Fnew;
    }
    else
      // (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
      defgrd.MultiplyTT(xcurr, N_XYZ);

    // Right Cauchy-Green tensor = F^T * F
    LINALG::Matrix<NUMDIM_WEG6, NUMDIM_WEG6> cauchygreen;
    cauchygreen.MultiplyTN(defgrd, defgrd);

    // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
    // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
    LINALG::Matrix<6, 1> glstrain(false);
    glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
    glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
    glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
    glstrain(3) = cauchygreen(0, 1);
    glstrain(4) = cauchygreen(1, 2);
    glstrain(5) = cauchygreen(2, 0);

    // return gp strains (only in case of stress/strain output)
    switch (iostrain)
    {
      case INPAR::STR::strain_gl:
      {
        if (elestrain == nullptr) dserror("no strain data available");
        for (int i = 0; i < 3; ++i) (*elestrain)(gp, i) = glstrain(i);
        for (int i = 3; i < 6; ++i) (*elestrain)(gp, i) = 0.5 * glstrain(i);
      }
      break;
      case INPAR::STR::strain_ea:
      {
        if (elestrain == nullptr) dserror("no strain data available");

        // rewriting Green-Lagrange strains in matrix format
        LINALG::Matrix<NUMDIM_WEG6, NUMDIM_WEG6> gl;
        gl(0, 0) = glstrain(0);
        gl(0, 1) = 0.5 * glstrain(3);
        gl(0, 2) = 0.5 * glstrain(5);
        gl(1, 0) = gl(0, 1);
        gl(1, 1) = glstrain(1);
        gl(1, 2) = 0.5 * glstrain(4);
        gl(2, 0) = gl(0, 2);
        gl(2, 1) = gl(1, 2);
        gl(2, 2) = glstrain(2);

        // inverse of deformation gradient
        LINALG::Matrix<NUMDIM_WEG6, NUMDIM_WEG6>
            invdefgrd;  // make a copy here otherwise defgrd is destroyed!
        invdefgrd.Invert(defgrd);

        LINALG::Matrix<NUMDIM_WEG6, NUMDIM_WEG6> temp;
        LINALG::Matrix<NUMDIM_WEG6, NUMDIM_WEG6> euler_almansi;
        temp.Multiply(gl, invdefgrd);
        euler_almansi.MultiplyTN(invdefgrd, temp);

        (*elestrain)(gp, 0) = euler_almansi(0, 0);
        (*elestrain)(gp, 1) = euler_almansi(1, 1);
        (*elestrain)(gp, 2) = euler_almansi(2, 2);
        (*elestrain)(gp, 3) = euler_almansi(0, 1);
        (*elestrain)(gp, 4) = euler_almansi(1, 2);
        (*elestrain)(gp, 5) = euler_almansi(0, 2);
      }
      break;
      case INPAR::STR::strain_none:
        break;
      default:
        dserror("requested strain type not available");
        break;
    }

    /* non-linear B-operator (may so be called, meaning
    ** of B-operator is not so sharp in the non-linear realm) *
    ** B = F . Bl *
    **
    **      [ ... | F_11*N_{,1}^k  F_21*N_{,1}^k  F_31*N_{,1}^k | ... ]
    **      [ ... | F_12*N_{,2}^k  F_22*N_{,2}^k  F_32*N_{,2}^k | ... ]
    **      [ ... | F_13*N_{,3}^k  F_23*N_{,3}^k  F_33*N_{,3}^k | ... ]
    ** B =  [ ~~~   ~~~~~~~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~   ~~~ ]
    **      [       F_11*N_{,2}^k+F_12*N_{,1}^k                       ]
    **      [ ... |          F_21*N_{,2}^k+F_22*N_{,1}^k        | ... ]
    **      [                       F_31*N_{,2}^k+F_32*N_{,1}^k       ]
    **      [                                                         ]
    **      [       F_12*N_{,3}^k+F_13*N_{,2}^k                       ]
    **      [ ... |          F_22*N_{,3}^k+F_23*N_{,2}^k        | ... ]
    **      [                       F_32*N_{,3}^k+F_33*N_{,2}^k       ]
    **      [                                                         ]
    **      [       F_13*N_{,1}^k+F_11*N_{,3}^k                       ]
    **      [ ... |          F_23*N_{,1}^k+F_21*N_{,3}^k        | ... ]
    **      [                       F_33*N_{,1}^k+F_31*N_{,3}^k       ]
    */
    LINALG::Matrix<MAT::NUM_STRESS_3D, NUMDOF_WEG6> bop;
    for (int i = 0; i < NUMNOD_WEG6; ++i)
    {
      bop(0, NODDOF_WEG6 * i + 0) = defgrd(0, 0) * N_XYZ(0, i);
      bop(0, NODDOF_WEG6 * i + 1) = defgrd(1, 0) * N_XYZ(0, i);
      bop(0, NODDOF_WEG6 * i + 2) = defgrd(2, 0) * N_XYZ(0, i);
      bop(1, NODDOF_WEG6 * i + 0) = defgrd(0, 1) * N_XYZ(1, i);
      bop(1, NODDOF_WEG6 * i + 1) = defgrd(1, 1) * N_XYZ(1, i);
      bop(1, NODDOF_WEG6 * i + 2) = defgrd(2, 1) * N_XYZ(1, i);
      bop(2, NODDOF_WEG6 * i + 0) = defgrd(0, 2) * N_XYZ(2, i);
      bop(2, NODDOF_WEG6 * i + 1) = defgrd(1, 2) * N_XYZ(2, i);
      bop(2, NODDOF_WEG6 * i + 2) = defgrd(2, 2) * N_XYZ(2, i);
      /* ~~~ */
      bop(3, NODDOF_WEG6 * i + 0) = defgrd(0, 0) * N_XYZ(1, i) + defgrd(0, 1) * N_XYZ(0, i);
      bop(3, NODDOF_WEG6 * i + 1) = defgrd(1, 0) * N_XYZ(1, i) + defgrd(1, 1) * N_XYZ(0, i);
      bop(3, NODDOF_WEG6 * i + 2) = defgrd(2, 0) * N_XYZ(1, i) + defgrd(2, 1) * N_XYZ(0, i);
      bop(4, NODDOF_WEG6 * i + 0) = defgrd(0, 1) * N_XYZ(2, i) + defgrd(0, 2) * N_XYZ(1, i);
      bop(4, NODDOF_WEG6 * i + 1) = defgrd(1, 1) * N_XYZ(2, i) + defgrd(1, 2) * N_XYZ(1, i);
      bop(4, NODDOF_WEG6 * i + 2) = defgrd(2, 1) * N_XYZ(2, i) + defgrd(2, 2) * N_XYZ(1, i);
      bop(5, NODDOF_WEG6 * i + 0) = defgrd(0, 2) * N_XYZ(0, i) + defgrd(0, 0) * N_XYZ(2, i);
      bop(5, NODDOF_WEG6 * i + 1) = defgrd(1, 2) * N_XYZ(0, i) + defgrd(1, 0) * N_XYZ(2, i);
      bop(5, NODDOF_WEG6 * i + 2) = defgrd(2, 2) * N_XYZ(0, i) + defgrd(2, 0) * N_XYZ(2, i);
    }

    // call material law cccccccccccccccccccccccccccccccccccccccccccccccccccccc
    LINALG::Matrix<MAT::NUM_STRESS_3D, MAT::NUM_STRESS_3D> cmat(true);
    LINALG::Matrix<MAT::NUM_STRESS_3D, 1> stress(true);

    if (Material()->MaterialType() == INPAR::MAT::m_constraintmixture ||
        Material()->MaterialType() == INPAR::MAT::m_mixture)
    {
      // gp reference coordinates
      LINALG::Matrix<NUMNOD_WEG6, 1> funct(true);
      funct = shapefcts[gp];
      LINALG::Matrix<1, NUMDIM_WEG6> point(true);
      point.MultiplyTN(funct, xrefe);
      params.set("gprefecoord", point);
    }

    SolidMaterial()->Evaluate(&defgrd, &glstrain, params, &stress, &cmat, gp, Id());
    // end of call material law ccccccccccccccccccccccccccccccccccccccccccccccc

    // return gp stresses
    switch (iostress)
    {
      case INPAR::STR::stress_2pk:
      {
        if (elestress == nullptr) dserror("no stress data available");
        for (int i = 0; i < MAT::NUM_STRESS_3D; ++i) (*elestress)(gp, i) = stress(i);
      }
      break;
      case INPAR::STR::stress_cauchy:
      {
        if (elestress == nullptr) dserror("no stress data available");
        const double detF = defgrd.Determinant();

        LINALG::Matrix<NUMDIM_WEG6, NUMDIM_WEG6> pkstress;
        pkstress(0, 0) = stress(0);
        pkstress(0, 1) = stress(3);
        pkstress(0, 2) = stress(5);
        pkstress(1, 0) = pkstress(0, 1);
        pkstress(1, 1) = stress(1);
        pkstress(1, 2) = stress(4);
        pkstress(2, 0) = pkstress(0, 2);
        pkstress(2, 1) = pkstress(1, 2);
        pkstress(2, 2) = stress(2);

        LINALG::Matrix<NUMDIM_WEG6, NUMDIM_WEG6> temp;
        LINALG::Matrix<NUMDIM_WEG6, NUMDIM_WEG6> cauchystress;
        temp.Multiply(1.0 / detF, defgrd, pkstress);
        cauchystress.MultiplyNT(temp, defgrd);

        (*elestress)(gp, 0) = cauchystress(0, 0);
        (*elestress)(gp, 1) = cauchystress(1, 1);
        (*elestress)(gp, 2) = cauchystress(2, 2);
        (*elestress)(gp, 3) = cauchystress(0, 1);
        (*elestress)(gp, 4) = cauchystress(1, 2);
        (*elestress)(gp, 5) = cauchystress(0, 2);
      }
      break;
      case INPAR::STR::stress_none:
        break;
      default:
        dserror("requested stress type not available");
        break;
    }

    const double detJ_w = detJ * gpweights[gp];
    // update internal force vector
    if (force != nullptr)
    {
      // integrate internal force vector f = f + (B^T . sigma) * detJ * w(gp)
      force->MultiplyTN(detJ_w, bop, stress, 1.0);
    }

    // update stiffness matrix
    if (stiffmatrix != nullptr)
    {
      // integrate `elastic' and `initial-displacement' stiffness matrix
      // keu = keu + (B^T . C . B) * detJ * w(gp)
      LINALG::Matrix<MAT::NUM_STRESS_3D, NUMDOF_WEG6> cb;
      cb.Multiply(cmat, bop);  // temporary C . B
      stiffmatrix->MultiplyTN(detJ_w, bop, cb, 1.0);

      // integrate `geometric' stiffness matrix and add to keu *****************
      LINALG::Matrix<MAT::NUM_STRESS_3D, 1> sfac(stress);  // auxiliary integrated stress
      sfac.Scale(detJ_w);                      // detJ*w(gp)*[S11,S22,S33,S12=S21,S23=S32,S13=S31]
      std::vector<double> SmB_L(NUMDIM_WEG6);  // intermediate Sm.B_L
      // kgeo += (B_L^T . sigma . B_L) * detJ * w(gp)  with B_L = Ni,Xj see NiliFEM-Skript
      for (int inod = 0; inod < NUMNOD_WEG6; ++inod)
      {
        SmB_L[0] = sfac(0) * N_XYZ(0, inod) + sfac(3) * N_XYZ(1, inod) + sfac(5) * N_XYZ(2, inod);
        SmB_L[1] = sfac(3) * N_XYZ(0, inod) + sfac(1) * N_XYZ(1, inod) + sfac(4) * N_XYZ(2, inod);
        SmB_L[2] = sfac(5) * N_XYZ(0, inod) + sfac(4) * N_XYZ(1, inod) + sfac(2) * N_XYZ(2, inod);
        for (int jnod = 0; jnod < NUMNOD_WEG6; ++jnod)
        {
          double bopstrbop = 0.0;  // intermediate value
          for (int idim = 0; idim < NUMDIM_WEG6; ++idim)
            bopstrbop += N_XYZ(idim, jnod) * SmB_L[idim];
          (*stiffmatrix)(NUMDIM_WEG6 * inod + 0, NUMDIM_WEG6 * jnod + 0) += bopstrbop;
          (*stiffmatrix)(NUMDIM_WEG6 * inod + 1, NUMDIM_WEG6 * jnod + 1) += bopstrbop;
          (*stiffmatrix)(NUMDIM_WEG6 * inod + 2, NUMDIM_WEG6 * jnod + 2) += bopstrbop;
        }
      }  // end of integrate `geometric' stiffness ******************************
    }    // if (stiffmatrix != nullptr)

    if (massmatrix != nullptr)
    {  // evaluate mass matrix +++++++++++++++++++++++++
      // integrate consistent mass matrix
      double density = Material()->Density(gp);
      const double factor = detJ_w * density;
      double ifactor, massfactor;
      for (int inod = 0; inod < NUMNOD_WEG6; ++inod)
      {
        ifactor = shapefcts[gp](inod) * factor;
        for (int jnod = 0; jnod < NUMNOD_WEG6; ++jnod)
        {
          massfactor = ifactor * shapefcts[gp](jnod);  // intermediate factor
          (*massmatrix)(NUMDIM_WEG6 * inod + 0, NUMDIM_WEG6 * jnod + 0) += massfactor;
          (*massmatrix)(NUMDIM_WEG6 * inod + 1, NUMDIM_WEG6 * jnod + 1) += massfactor;
          (*massmatrix)(NUMDIM_WEG6 * inod + 2, NUMDIM_WEG6 * jnod + 2) += massfactor;
        }

        // check for non constant mass matrix
        if (SolidMaterial()->VaryingDensity())
        {
          /*
           If the density, i.e. the mass matrix, is not constant, a linearization is neccessary.
           In general, the mass matrix can be dependent on the displacements, the velocities and the
           accelerations. We write all the additional terms into the mass matrix, hence, conversion
           from accelerations to velocities and displacements are needed. As those conversions
           depend on the time integration scheme, the factors are set within the respective time
           integrators and read from the parameter list inside the element (this is a little
           ugly...). */
          double timintfac_dis = 0.0;
          double timintfac_vel = 0.0;
          if (IsParamsInterface())
          {
            timintfac_dis = StrParamsInterface().GetTimIntFactorDisp();
            timintfac_vel = StrParamsInterface().GetTimIntFactorVel();
          }
          else
          {
            timintfac_dis = params.get<double>("timintfac_dis");
            timintfac_vel = params.get<double>("timintfac_vel");
          }
          LINALG::Matrix<MAT::NUM_STRESS_3D, 1> linmass_disp(true);
          LINALG::Matrix<MAT::NUM_STRESS_3D, 1> linmass_vel(true);
          LINALG::Matrix<MAT::NUM_STRESS_3D, 1> linmass(true);

          // evaluate derivative of mass w.r.t. to right cauchy green tensor
          SolidMaterial()->EvaluateNonLinMass(
              &defgrd, &glstrain, params, &linmass_disp, &linmass_vel, gp, Id());

          // multiply by 2.0 to get derivative w.r.t green lagrange strains and multiply by time
          // integration factor
          linmass_disp.Scale(2.0 * timintfac_dis);
          linmass_vel.Scale(2.0 * timintfac_vel);
          linmass.Update(1.0, linmass_disp, 1.0, linmass_vel, 0.0);

          // evaluate accelerations at time n+1 at gauss point
          LINALG::Matrix<NUMDIM_WEG6, 1> myacc(true);
          for (int idim = 0; idim < NUMDIM_WEG6; ++idim)
            for (int inod = 0; inod < NUMNOD_WEG6; ++inod)
              myacc(idim) += shapefcts[gp](inod) * (*acc)[idim + (inod * NUMDIM_WEG6)];

          if (stiffmatrix != nullptr)
          {
            // integrate linearisation of mass matrix
            //(B^T . d\rho/d disp . a) * detJ * w(gp)
            LINALG::Matrix<1, NUMDOF_WEG6> cb;
            cb.MultiplyTN(linmass_disp, bop);
            for (int inod = 0; inod < NUMNOD_WEG6; ++inod)
            {
              double factor = detJ_w * shapefcts[gp](inod);
              for (int idim = 0; idim < NUMDIM_WEG6; ++idim)
              {
                double massfactor = factor * myacc(idim);
                for (int jnod = 0; jnod < NUMNOD_WEG6; ++jnod)
                  for (int jdim = 0; jdim < NUMDIM_WEG6; ++jdim)
                    (*massmatrix)(inod * NUMDIM_WEG6 + idim, jnod * NUMDIM_WEG6 + jdim) +=
                        massfactor * cb(jnod * NUMDIM_WEG6 + jdim);
              }
            }
          }

          // internal force vector without EAS terms
          if (forceinert != nullptr)
          {
            // integrate nonlinear inertia force term
            for (int inod = 0; inod < NUMNOD_WEG6; ++inod)
            {
              double forcefactor = shapefcts[gp](inod) * detJ_w;
              for (int idim = 0; idim < NUMDIM_WEG6; ++idim)
                (*forceinert)(inod * NUMDIM_WEG6 + idim) += forcefactor * density * myacc(idim);
            }
          }
        }
      }
    }  // end of mass matrix +++++++++++++++++++++++++++++++++++++++++++++++++++
       /* =========================================================================*/
  }    /* ==================================================== end of Loop over GP */
       /* =========================================================================*/

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate Wedge6 Shape fcts at all 6 Gauss Points           maf 09/08|
 *----------------------------------------------------------------------*/
const std::vector<LINALG::Matrix<NUMNOD_WEG6, 1>> DRT::ELEMENTS::So_weg6::sow6_shapefcts()
{
  std::vector<LINALG::Matrix<NUMNOD_WEG6, 1>> shapefcts(NUMGPT_WEG6);
  // (r,s,t) gp-locations of fully integrated linear 6-node Wedge
  // fill up nodal f at each gp
  const DRT::UTILS::GaussRule3D gaussrule = DRT::UTILS::GaussRule3D::wedge_6point;
  const DRT::UTILS::IntegrationPoints3D intpoints(gaussrule);
  for (int igp = 0; igp < intpoints.nquad; ++igp)
  {
    const double r = intpoints.qxg[igp][0];
    const double s = intpoints.qxg[igp][1];
    const double t = intpoints.qxg[igp][2];

    DRT::UTILS::shape_function_3D(shapefcts[igp], r, s, t, wedge6);
  }
  return shapefcts;
}

/*----------------------------------------------------------------------*
 |  Evaluate Wedge6 Shape fct-derivs at all 6 Gauss Points     maf 09/08|
 *----------------------------------------------------------------------*/
const std::vector<LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6>> DRT::ELEMENTS::So_weg6::sow6_derivs()
{
  std::vector<LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6>> derivs(NUMGPT_WEG6);
  // (r,s,t) gp-locations of fully integrated linear 6-node Wedge
  // fill up df w.r.t. rst directions (NUMDIM) at each gp
  const DRT::UTILS::GaussRule3D gaussrule = DRT::UTILS::GaussRule3D::wedge_6point;
  const DRT::UTILS::IntegrationPoints3D intpoints(gaussrule);
  for (int igp = 0; igp < intpoints.nquad; ++igp)
  {
    const double r = intpoints.qxg[igp][0];
    const double s = intpoints.qxg[igp][1];
    const double t = intpoints.qxg[igp][2];

    DRT::UTILS::shape_function_3D_deriv1(derivs[igp], r, s, t, wedge6);
  }
  return derivs;
}

/*----------------------------------------------------------------------*
 |  Evaluate Wedge6 Weights at all 6 Gauss Points              maf 09/08|
 *----------------------------------------------------------------------*/
const std::vector<double> DRT::ELEMENTS::So_weg6::sow6_weights()
{
  std::vector<double> weights(NUMGPT_WEG6);
  const DRT::UTILS::GaussRule3D gaussrule = DRT::UTILS::GaussRule3D::wedge_6point;
  const DRT::UTILS::IntegrationPoints3D intpoints(gaussrule);
  for (int i = 0; i < NUMGPT_WEG6; ++i)
  {
    weights[i] = intpoints.qwgt[i];
  }
  return weights;
}


/*----------------------------------------------------------------------*
 |  shape functions and derivatives for So_hex8                maf 04/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_weg6::sow6_shapederiv(
    LINALG::Matrix<NUMNOD_WEG6, NUMGPT_WEG6>** shapefct,  // pointer to pointer of shapefct
    LINALG::Matrix<NUMDOF_WEG6, NUMNOD_WEG6>** deriv,     // pointer to pointer of derivs
    LINALG::Matrix<NUMGPT_WEG6, 1>** weights)             // pointer to pointer of weights
{
  // static matrix objects, kept in memory
  static LINALG::Matrix<NUMNOD_WEG6, NUMGPT_WEG6> f;    // shape functions
  static LINALG::Matrix<NUMDOF_WEG6, NUMNOD_WEG6> df;   // derivatives
  static LINALG::Matrix<NUMGPT_WEG6, 1> weightfactors;  // weights for each gp
  static bool fdf_eval;                                 // flag for re-evaluate everything


  if (fdf_eval == true)
  {                             // if true f,df already evaluated
    *shapefct = &f;             // return adress of static object to target of pointer
    *deriv = &df;               // return adress of static object to target of pointer
    *weights = &weightfactors;  // return adress of static object to target of pointer
    return;
  }
  else
  {
    // (r,s,t) gp-locations of fully integrated linear 6-node Wedge
    // fill up nodal f at each gp
    // fill up df w.r.t. rst directions (NUMDIM) at each gp
    const DRT::UTILS::GaussRule3D gaussrule_ = DRT::UTILS::GaussRule3D::wedge_6point;
    const DRT::UTILS::IntegrationPoints3D intpoints(gaussrule_);
    for (int igp = 0; igp < intpoints.nquad; ++igp)
    {
      const double r = intpoints.qxg[igp][0];
      const double s = intpoints.qxg[igp][1];
      const double t = intpoints.qxg[igp][2];

      LINALG::Matrix<NUMNOD_WEG6, 1> funct;
      LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6> deriv;
      DRT::UTILS::shape_function_3D(funct, r, s, t, wedge6);
      DRT::UTILS::shape_function_3D_deriv1(deriv, r, s, t, wedge6);
      for (int inode = 0; inode < NUMNOD_WEG6; ++inode)
      {
        f(inode, igp) = funct(inode);
        df(igp * NUMDIM_WEG6 + 0, inode) = deriv(0, inode);
        df(igp * NUMDIM_WEG6 + 1, inode) = deriv(1, inode);
        df(igp * NUMDIM_WEG6 + 2, inode) = deriv(2, inode);
        weightfactors(igp) = intpoints.qwgt[igp];
      }
    }
    // return adresses of just evaluated matrices
    *shapefct = &f;             // return adress of static object to target of pointer
    *deriv = &df;               // return adress of static object to target of pointer
    *weights = &weightfactors;  // return adress of static object to target of pointer
    fdf_eval = true;            // now all arrays are filled statically
  }
  return;
}  // of sow6_shapederiv

/*----------------------------------------------------------------------*
 |  lump mass matrix                                         bborn 07/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_weg6::sow6_lumpmass(LINALG::Matrix<NUMDOF_WEG6, NUMDOF_WEG6>* emass)
{
  // lump mass matrix
  if (emass != nullptr)
  {
    // we assume #elemat2 is a square matrix
    for (unsigned c = 0; c < (*emass).N(); ++c)  // parse columns
    {
      double d = 0.0;
      for (unsigned r = 0; r < (*emass).M(); ++r)  // parse rows
      {
        d += (*emass)(r, c);  // accumulate row entries
        (*emass)(r, c) = 0.0;
      }
      (*emass)(c, c) = d;  // apply sum of row entries on diagonal
    }
  }
}

/*----------------------------------------------------------------------*
 |  init the element (public)                                  gee 04/08|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::So_weg6Type::Initialize(DRT::Discretization& dis)
{
  for (int i = 0; i < dis.NumMyColElements(); ++i)
  {
    if (dis.lColElement(i)->ElementType() != *this) continue;
    auto* actele = dynamic_cast<DRT::ELEMENTS::So_weg6*>(dis.lColElement(i));
    if (!actele) dserror("cast to So_weg6* failed");
    actele->InitJacobianMapping();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  compute def gradient at every gaussian point (protected)   gee 07/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_weg6::DefGradient(const std::vector<double>& disp,
    Epetra_SerialDenseMatrix& gpdefgrd, DRT::ELEMENTS::PreStress& prestress)
{
  const static std::vector<LINALG::Matrix<NUMNOD_WEG6, 1>> shapefcts = sow6_shapefcts();
  const static std::vector<LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6>> derivs = sow6_derivs();

  // update element geometry
  LINALG::Matrix<NUMNOD_WEG6, NUMDIM_WEG6> xdisp;  // current  coord. of element
  for (int i = 0; i < NUMNOD_WEG6; ++i)
  {
    xdisp(i, 0) = disp[i * NODDOF_WEG6 + 0];
    xdisp(i, 1) = disp[i * NODDOF_WEG6 + 1];
    xdisp(i, 2) = disp[i * NODDOF_WEG6 + 2];
  }

  for (int gp = 0; gp < NUMGPT_WEG6; ++gp)
  {
    // get Jacobian mapping wrt to the stored deformed configuration
    LINALG::Matrix<3, 3> invJdef;
    prestress.StoragetoMatrix(gp, invJdef, prestress.JHistory());

    // by N_XYZ = J^-1 * N_rst
    LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6> N_xyz;
    N_xyz.Multiply(invJdef, derivs[gp]);

    // build defgrd (independent of xrefe!)
    LINALG::Matrix<3, 3> defgrd;
    defgrd.MultiplyTT(xdisp, N_xyz);
    defgrd(0, 0) += 1.0;
    defgrd(1, 1) += 1.0;
    defgrd(2, 2) += 1.0;

    prestress.MatrixtoStorage(gp, defgrd, gpdefgrd);
  }
  return;
}

/*----------------------------------------------------------------------*
 |  compute Jac.mapping wrt deformed configuration (protected) gee 07/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_weg6::UpdateJacobianMapping(
    const std::vector<double>& disp, DRT::ELEMENTS::PreStress& prestress)
{
  const static std::vector<LINALG::Matrix<NUMNOD_WEG6, 1>> shapefcts = sow6_shapefcts();
  const static std::vector<LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6>> derivs = sow6_derivs();

  // get incremental disp
  LINALG::Matrix<NUMNOD_WEG6, NUMDIM_WEG6> xdisp;
  for (int i = 0; i < NUMNOD_WEG6; ++i)
  {
    xdisp(i, 0) = disp[i * NODDOF_WEG6 + 0];
    xdisp(i, 1) = disp[i * NODDOF_WEG6 + 1];
    xdisp(i, 2) = disp[i * NODDOF_WEG6 + 2];
  }

  LINALG::Matrix<3, 3> invJhist;
  LINALG::Matrix<3, 3> invJ;
  LINALG::Matrix<3, 3> defgrd;
  LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6> N_xyz;
  LINALG::Matrix<3, 3> invJnew;
  for (int gp = 0; gp < NUMGPT_WEG6; ++gp)
  {
    // get the invJ old state
    prestress.StoragetoMatrix(gp, invJhist, prestress.JHistory());
    // get derivatives wrt to invJhist
    N_xyz.Multiply(invJhist, derivs[gp]);
    // build defgrd \partial x_new / \parial x_old , where x_old != X
    defgrd.MultiplyTT(xdisp, N_xyz);
    defgrd(0, 0) += 1.0;
    defgrd(1, 1) += 1.0;
    defgrd(2, 2) += 1.0;
    // make inverse of this defgrd
    defgrd.Invert();
    // push-forward of Jinv
    invJnew.MultiplyTN(defgrd, invJhist);
    // store new reference configuration
    prestress.MatrixtoStorage(gp, invJnew, prestress.JHistory());
  }  // for (int gp=0; gp<NUMGPT_WEG6; ++gp)

  return;
}

/*----------------------------------------------------------------------*
 |  remodeling of fiber directions (protected)               tinkl 01/10|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_weg6::sow6_remodel(std::vector<int>& lm,  // location matrix
    std::vector<double>& disp,                                   // current displacements
    Teuchos::ParameterList& params,                              // algorithmic parameters e.g. time
    const Teuchos::RCP<MAT::Material>& mat)                      // material
{
  if ((Material()->MaterialType() == INPAR::MAT::m_constraintmixture) ||
      (Material()->MaterialType() == INPAR::MAT::m_elasthyper))
  {
    // in a first step ommit everything with prestress and EAS!!
    const static std::vector<LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6>> derivs = sow6_derivs();

    // update element geometry
    LINALG::Matrix<NUMNOD_WEG6, NUMDIM_WEG6> xcurr;  // current  coord. of element
    LINALG::Matrix<NUMNOD_WEG6, NUMDIM_WEG6> xdisp;
    DRT::Node** nodes = Nodes();
    for (int i = 0; i < NUMNOD_WEG6; ++i)
    {
      const double* x = nodes[i]->X();
      xcurr(i, 0) = x[0] + disp[i * NODDOF_WEG6 + 0];
      xcurr(i, 1) = x[1] + disp[i * NODDOF_WEG6 + 1];
      xcurr(i, 2) = x[2] + disp[i * NODDOF_WEG6 + 2];

      if (::UTILS::PRESTRESS::IsMulf(pstype_))
      {
        xdisp(i, 0) = disp[i * NODDOF_WEG6 + 0];
        xdisp(i, 1) = disp[i * NODDOF_WEG6 + 1];
        xdisp(i, 2) = disp[i * NODDOF_WEG6 + 2];
      }
    }
    /* =========================================================================*/
    /* ================================================= Loop over Gauss Points */
    /* =========================================================================*/
    LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6> N_XYZ;
    // interpolated values of stress and defgrd for remodeling
    LINALG::Matrix<3, 3> avg_stress(true);
    LINALG::Matrix<3, 3> avg_defgrd(true);

    // build deformation gradient wrt to material configuration
    LINALG::Matrix<NUMDIM_WEG6, NUMDIM_WEG6> defgrd(false);
    for (int gp = 0; gp < NUMGPT_WEG6; ++gp)
    {
      /* get the inverse of the Jacobian matrix which looks like:
      **            [ x_,r  y_,r  z_,r ]^-1
      **     J^-1 = [ x_,s  y_,s  z_,s ]
      **            [ x_,t  y_,t  z_,t ]
      */
      // compute derivatives N_XYZ at gp w.r.t. material coordinates
      // by N_XYZ = J^-1 * N_rst
      N_XYZ.Multiply(invJ_[gp], derivs[gp]);

      if (::UTILS::PRESTRESS::IsMulf(pstype_))
      {
        // get Jacobian mapping wrt to the stored configuration
        LINALG::Matrix<3, 3> invJdef;
        prestress_->StoragetoMatrix(gp, invJdef, prestress_->JHistory());
        // get derivatives wrt to last spatial configuration
        LINALG::Matrix<NUMDIM_WEG6, NUMNOD_WEG6> N_xyz;
        N_xyz.Multiply(invJdef, derivs[gp]);

        // build multiplicative incremental defgrd
        defgrd.MultiplyTT(xdisp, N_xyz);
        defgrd(0, 0) += 1.0;
        defgrd(1, 1) += 1.0;
        defgrd(2, 2) += 1.0;

        // get stored old incremental F
        LINALG::Matrix<3, 3> Fhist;
        prestress_->StoragetoMatrix(gp, Fhist, prestress_->FHistory());

        // build total defgrd = delta F * F_old
        LINALG::Matrix<3, 3> Fnew;
        Fnew.Multiply(defgrd, Fhist);
        defgrd = Fnew;
      }
      else
        // (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
        defgrd.MultiplyTT(xcurr, N_XYZ);

      // Right Cauchy-Green tensor = F^T * F
      LINALG::Matrix<NUMDIM_WEG6, NUMDIM_WEG6> cauchygreen;
      cauchygreen.MultiplyTN(defgrd, defgrd);

      // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
      // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
      //    Epetra_SerialDenseVector glstrain_epetra(MAT::NUM_STRESS_3D);
      //    LINALG::Matrix<MAT::NUM_STRESS_3D,1> glstrain(glstrain_epetra.A(),true);
      LINALG::Matrix<6, 1> glstrain(false);
      glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
      glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
      glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
      glstrain(3) = cauchygreen(0, 1);
      glstrain(4) = cauchygreen(1, 2);
      glstrain(5) = cauchygreen(2, 0);

      // call material law cccccccccccccccccccccccccccccccccccccccccccccccccccccc
      LINALG::Matrix<MAT::NUM_STRESS_3D, MAT::NUM_STRESS_3D> cmat(true);
      LINALG::Matrix<MAT::NUM_STRESS_3D, 1> stress(true);

      SolidMaterial()->Evaluate(&defgrd, &glstrain, params, &stress, &cmat, gp, Id());
      // end of call material law ccccccccccccccccccccccccccccccccccccccccccccccc

      // Cauchy stress
      const double detF = defgrd.Determinant();

      LINALG::Matrix<3, 3> pkstress;
      pkstress(0, 0) = stress(0);
      pkstress(0, 1) = stress(3);
      pkstress(0, 2) = stress(5);
      pkstress(1, 0) = pkstress(0, 1);
      pkstress(1, 1) = stress(1);
      pkstress(1, 2) = stress(4);
      pkstress(2, 0) = pkstress(0, 2);
      pkstress(2, 1) = pkstress(1, 2);
      pkstress(2, 2) = stress(2);

      LINALG::Matrix<3, 3> temp(true);
      LINALG::Matrix<3, 3> cauchystress(true);
      temp.Multiply(1.0 / detF, defgrd, pkstress);
      cauchystress.MultiplyNT(temp, defgrd);

      // evaluate eigenproblem based on stress of previous step
      LINALG::Matrix<3, 3> lambda(true);
      LINALG::Matrix<3, 3> locsys(true);
      LINALG::SYEV(cauchystress, lambda, locsys);

      if (mat->MaterialType() == INPAR::MAT::m_constraintmixture)
      {
        auto* comi = dynamic_cast<MAT::ConstraintMixture*>(mat.get());
        comi->EvaluateFiberVecs(gp, locsys, defgrd);
      }
      else if (mat->MaterialType() == INPAR::MAT::m_elasthyper)
      {
        // we only have fibers at element center, thus we interpolate stress and defgrd
        avg_stress.Update(1.0 / NUMGPT_WEG6, cauchystress, 1.0);
        avg_defgrd.Update(1.0 / NUMGPT_WEG6, defgrd, 1.0);
      }
      else
        dserror("material not implemented for remodeling");

    }  // end loop over gauss points

    if (mat->MaterialType() == INPAR::MAT::m_elasthyper)
    {
      // evaluate eigenproblem based on stress of previous step
      LINALG::Matrix<3, 3> lambda(true);
      LINALG::Matrix<3, 3> locsys(true);
      LINALG::SYEV(avg_stress, lambda, locsys);

      // modulation function acc. Hariton: tan g = 2nd max lambda / max lambda
      double newgamma = atan2(lambda(1, 1), lambda(2, 2));
      // compression in 2nd max direction, thus fibers are alligned to max principal direction
      if (lambda(1, 1) < 0) newgamma = 0.0;

      // new fiber vectors
      auto* elast = dynamic_cast<MAT::ElastHyper*>(mat.get());
      elast->EvaluateFiberVecs(newgamma, locsys, avg_defgrd);
    }
  }
}

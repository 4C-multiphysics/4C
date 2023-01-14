/*----------------------------------------------------------------------*/
/*! \file
\brief pyramid shaped solid element
\level 2

*/
/*----------------------------------------------------------------------*/

#include "Epetra_SerialDenseSolver.h"
#include "so3_pyramid5fbar.H"
#include "linalg_serialdensematrix.H"
#include "lib_discret.H"
#include "lib_utils.H"
#include "lib_dserror.H"
#include "lib_prestress_service.H"
#include "mat_so3_material.H"
#include "mat_thermoplastichyperelast.H"
#include "linalg_utils_densematrix_eigen.H"
#include "linalg_serialdensevector.H"
#include "fem_general_utils_integration.H"
#include "fem_general_utils_fem_shapefunctions.H"
#include "lib_globalproblem.H"
#include "so3_prestress.H"
#include "patspec.H"

#include "structure_new_elements_paramsinterface.H"

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                                       |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::So_pyramid5fbar::Evaluate(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, std::vector<int>& lm,
    Epetra_SerialDenseMatrix& elemat1_epetra, Epetra_SerialDenseMatrix& elemat2_epetra,
    Epetra_SerialDenseVector& elevec1_epetra, Epetra_SerialDenseVector& elevec2_epetra,
    Epetra_SerialDenseVector& elevec3_epetra)
{
  // Check whether the solid material PostSetup() routine has already been called and call it if not
  EnsureMaterialPostSetup(params);

  LINALG::Matrix<NUMDOF_SOP5, NUMDOF_SOP5> elemat1(elemat1_epetra.A(), true);
  LINALG::Matrix<NUMDOF_SOP5, NUMDOF_SOP5> elemat2(elemat2_epetra.A(), true);
  LINALG::Matrix<NUMDOF_SOP5, 1> elevec1(elevec1_epetra.A(), true);
  LINALG::Matrix<NUMDOF_SOP5, 1> elevec2(elevec2_epetra.A(), true);
  LINALG::Matrix<NUMDOF_SOP5, 1> elevec3(elevec3_epetra.A(), true);

  // start with "none"
  DRT::ELEMENTS::So_pyramid5fbar::ActionType act = So_pyramid5fbar::none;

  // get the required action
  std::string action = params.get<std::string>("action", "none");
  if (action == "none")
    dserror("No action supplied");
  else if (action == "calc_struct_linstiff")
    act = So_pyramid5fbar::calc_struct_linstiff;
  else if (action == "calc_struct_nlnstiff")
    act = So_pyramid5fbar::calc_struct_nlnstiff;
  else if (action == "calc_struct_internalforce")
    act = So_pyramid5fbar::calc_struct_internalforce;
  else if (action == "calc_struct_linstiffmass")
    act = So_pyramid5fbar::calc_struct_linstiffmass;
  else if (action == "calc_struct_nlnstiffmass")
    act = So_pyramid5fbar::calc_struct_nlnstiffmass;
  else if (action == "calc_struct_nlnstifflmass")
    act = So_pyramid5fbar::calc_struct_nlnstifflmass;
  else if (action == "calc_struct_stress")
    act = So_pyramid5fbar::calc_struct_stress;
  else if (action == "calc_struct_eleload")
    act = So_pyramid5fbar::calc_struct_eleload;
  else if (action == "calc_struct_fsiload")
    act = So_pyramid5fbar::calc_struct_fsiload;
  else if (action == "calc_struct_update_istep")
    act = So_pyramid5fbar::calc_struct_update_istep;
  else if (action == "calc_struct_reset_istep")
    act = So_pyramid5fbar::calc_struct_reset_istep;
  else if (action == "calc_struct_reset_all")
    act = So_pyramid5fbar::calc_struct_reset_all;
  else if (action == "postprocess_stress")
    act = So_pyramid5fbar::postprocess_stress;
  else if (action == "multi_readrestart")
    act = So_pyramid5fbar::multi_readrestart;
  else if (action == "multi_calc_dens")
    act = So_pyramid5fbar::multi_calc_dens;
  else if (action == "calc_struct_prestress_update")
    act = So_pyramid5fbar::prestress_update;
  else if (action == "calc_struct_energy")
    act = So_pyramid5::calc_struct_energy;
  else if (action == "calc_struct_recover")
    return 0;
  else if (action == "calc_struct_predict")
    return 0;
  else
    dserror("Unknown type of action for So_pyramid5fbar");

  // check for patient specific data
  PATSPEC::GetILTDistance(Id(), params, discretization);
  PATSPEC::GetLocalRadius(Id(), params, discretization);
  PATSPEC::GetInnerRadius(Id(), params, discretization);

  // what should the element do
  switch (act)
  {
    // linear stiffness
    case calc_struct_linstiff:
    {
      // need current displacement and residual forces
      std::vector<double> mydisp(lm.size());
      for (double& i : mydisp) i = 0.0;
      std::vector<double> myres(lm.size());
      for (double& myre : myres) myre = 0.0;
      std::vector<double> mydispmat(lm.size(), 0.0);

      nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, mydispmat, &elemat1, nullptr, &elevec1,
          nullptr, nullptr, nullptr, nullptr, nullptr, params, INPAR::STR::stress_none,
          INPAR::STR::strain_none, INPAR::STR::strain_none);
    }
    break;

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
      LINALG::Matrix<NUMDOF_SOP5, NUMDOF_SOP5>* matptr = nullptr;
      if (elemat1.IsInitialized()) matptr = &elemat1;
      std::vector<double> mydispmat(lm.size(), 0.0);

      nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, mydispmat, matptr, nullptr, &elevec1,
          nullptr, nullptr, nullptr, nullptr, nullptr, params, INPAR::STR::stress_none,
          INPAR::STR::strain_none, INPAR::STR::strain_none);
    }
    break;

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
      LINALG::Matrix<NUMDOF_SOP5, NUMDOF_SOP5> myemat(true);
      std::vector<double> mydispmat(lm.size(), 0.0);

      nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, mydispmat, &myemat, nullptr, &elevec1,
          nullptr, nullptr, nullptr, nullptr, nullptr, params, INPAR::STR::stress_none,
          INPAR::STR::strain_none, INPAR::STR::strain_none);
    }
    break;

    // linear stiffness and consistent mass matrix
    case calc_struct_linstiffmass:
      dserror("Case 'calc_struct_linstiffmass' not yet implemented");
      break;

    // nonlinear stiffness, internal force vector, and consistent mass matrix
    case calc_struct_nlnstiffmass:
    case calc_struct_nlnstifflmass:
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

      nlnstiffmass(lm, mydisp, &myvel, &myacc, myres, mydispmat, &elemat1, &elemat2, &elevec1,
          &elevec2, &elevec3, nullptr, nullptr, nullptr, params, INPAR::STR::stress_none,
          INPAR::STR::strain_none, INPAR::STR::strain_none);

      if (act == calc_struct_nlnstifflmass) sop5_lumpmass(&elemat2);
    }
    break;

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
        if (stressdata == Teuchos::null) dserror("Cannot get 'stress' data");
        if (straindata == Teuchos::null) dserror("Cannot get 'strain' data");
        std::vector<double> mydisp(lm.size());
        DRT::UTILS::ExtractMyValues(*disp, mydisp, lm);
        std::vector<double> myres(lm.size());
        DRT::UTILS::ExtractMyValues(*res, myres, lm);
        LINALG::Matrix<NUMGPT_SOP5, MAT::NUM_STRESS_3D> stress;
        LINALG::Matrix<NUMGPT_SOP5, MAT::NUM_STRESS_3D> strain;
        auto iostress =
            DRT::INPUT::get<INPAR::STR::StressType>(params, "iostress", INPAR::STR::stress_none);
        auto iostrain =
            DRT::INPUT::get<INPAR::STR::StrainType>(params, "iostrain", INPAR::STR::strain_none);

        std::vector<double> mydispmat(lm.size(), 0.0);

        // in case of small strain materials calculate plastic strains for post processing
        Teuchos::RCP<std::vector<char>> plstraindata =
            params.get<Teuchos::RCP<std::vector<char>>>("plstrain", Teuchos::null);
        if (plstraindata == Teuchos::null) dserror("Cannot get 'plastic strain' data");
        LINALG::Matrix<NUMGPT_SOP5, MAT::NUM_STRESS_3D> plstrain;
        auto ioplstrain =
            DRT::INPUT::get<INPAR::STR::StrainType>(params, "ioplstrain", INPAR::STR::strain_none);

        nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, mydispmat, nullptr, nullptr, nullptr,
            nullptr, nullptr, &stress, &strain, &plstrain, params, iostress, iostrain, ioplstrain);
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
        {
          DRT::PackBuffer data;
          AddtoPack(data, plstrain);
          data.StartPacking();
          AddtoPack(data, plstrain);
          std::copy(data().begin(), data().end(), std::back_inserter(*plstraindata));
        }
      }
    }
    break;

    // postprocess stresses/strains at gauss points

    // note that in the following, quantities are always referred to as
    // "stresses" etc. although they might also apply to strains
    // (depending on what this routine is called for from the post filter)
    case postprocess_stress:
    {
      const Teuchos::RCP<std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix>>> gpstressmap =
          params.get<Teuchos::RCP<std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix>>>>(
              "gpstressmap", Teuchos::null);
      if (gpstressmap == Teuchos::null)
        dserror("no gp stress/strain map available for postprocessing");
      std::string stresstype = params.get<std::string>("stresstype", "ndxyz");
      int gid = Id();
      LINALG::Matrix<NUMGPT_SOP5, MAT::NUM_STRESS_3D> gpstress(((*gpstressmap)[gid])->A(), true);

      Teuchos::RCP<Epetra_MultiVector> poststress =
          params.get<Teuchos::RCP<Epetra_MultiVector>>("poststress", Teuchos::null);
      if (poststress == Teuchos::null) dserror("No element stress/strain vector available");

      if (stresstype == "ndxyz")
      {
        // extrapolate stresses/strains at Gauss points to nodes
        sop5_expol(gpstress, *poststress);
      }
      else if (stresstype == "cxyz")
      {
        const Epetra_BlockMap& elemap = poststress->Map();
        int lid = elemap.LID(Id());
        if (lid != -1)
        {
          for (int i = 0; i < MAT::NUM_STRESS_3D; ++i)
          {
            double& s = (*((*poststress)(i)))[lid];  // resolve pointer for faster access
            s = 0.;
            for (int j = 0; j < NUMGPT_SOP5; ++j)
            {
              s += gpstress(j, i);
            }
            s *= 1.0 / NUMGPT_SOP5;
          }
        }
      }
      else
      {
        dserror("unknown type of stress/strain output on element level");
      }
    }
    break;

    case calc_struct_eleload:
      dserror("this method is not supposed to evaluate a load, use EvaluateNeumann(...)");
      break;

    case calc_struct_fsiload:
      dserror("Case not yet implemented");
      break;

    case calc_struct_update_istep:
    {
      // Update of history for materials
      SolidMaterial()->Update();
    }
    break;

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
      SolidMaterial()->ResetAll(NUMGPT_SOP5);

      // Reset prestress
      if (::UTILS::PRESTRESS::IsMulf(pstype_))
      {
        time_ = 0.0;
        LINALG::Matrix<3, 3> Id(true);
        Id(0, 0) = Id(1, 1) = Id(2, 2) = 1.0;
        for (int gp = 0; gp < NUMGPT_SOP5; ++gp)
        {
          prestress_->MatrixtoStorage(gp, Id, prestress_->FHistory());
          prestress_->MatrixtoStorage(gp, invJ_[gp], prestress_->JHistory());
        }
        prestress_->MatrixtoStorage(NUMGPT_SOP5, Id, prestress_->FHistory());
        LINALG::Matrix<NUMNOD_SOP5, NUMDIM_SOP5> xrefe;
        for (int i = 0; i < NUMNOD_SOP5; ++i)
        {
          xrefe(i, 0) = Nodes()[i]->X()[0];
          xrefe(i, 1) = Nodes()[i]->X()[1];
          xrefe(i, 2) = Nodes()[i]->X()[2];
        }
        LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5> N_rst_0;
        DRT::UTILS::shape_function_3D_deriv1(N_rst_0, 0.0, 0.0, 0.25, pyramid5);
        LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> invJ_0;
        invJ_0.Multiply(N_rst_0, xrefe);
        invJ_0.Invert();
        prestress_->MatrixtoStorage(NUMGPT_SOP5, invJ_0, prestress_->JHistory());
      }
      if (::UTILS::PRESTRESS::IsInverseDesign(pstype_))
        dserror("Reset of Inverse Design not yet implemented");
    }
    break;

    case multi_calc_dens:
    {
      sop5_homog(params);
    }
    break;

    //==================================================================================
    case prestress_update:
    {
      time_ = params.get<double>("total time");
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
      if (disp == Teuchos::null) dserror("Cannot get displacement state");
      std::vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp, mydisp, lm);

      // build def gradient for every gauss point
      LINALG::SerialDenseMatrix gpdefgrd(NUMGPT_SOP5 + 1, 9);
      DefGradient(mydisp, gpdefgrd, *prestress_);

      // update deformation gradient and put back to storage
      LINALG::Matrix<3, 3> deltaF;
      LINALG::Matrix<3, 3> Fhist;
      LINALG::Matrix<3, 3> Fnew;
      for (int gp = 0; gp < NUMGPT_SOP5 + 1; ++gp)
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
    case inversedesign_update:
      dserror("The sopyramid5fbar element does not support inverse design analysis");
      break;


    // read restart of microscale
    case multi_readrestart:
    {
      Teuchos::RCP<MAT::Material> mat = Material();

      if (mat->MaterialType() == INPAR::MAT::m_struct_multiscale) sop5_read_restart_multi();
    }
    break;

    //==================================================================================
    case calc_struct_energy:
    {
      // check length of elevec1
      if (elevec1_epetra.Length() < 1) dserror("The given result vector is too short.");

      // initialization of internal energy
      double intenergy = 0.0;

      // shape functions and Gauss weights
      const static std::vector<LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5>> derivs = sop5_derivs();
      const static std::vector<double> weights = sop5_weights();

      // get displacements of this processor
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
      if (disp == Teuchos::null) dserror("Cannot get state displacement vector");

      // get displacements of this element
      std::vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp, mydisp, lm);

      // update element geometry
      LINALG::Matrix<NUMNOD_SOP5, NUMDIM_SOP5> xrefe;  // material coord. of element
      LINALG::Matrix<NUMNOD_SOP5, NUMDIM_SOP5> xcurr;  // current  coord. of element
      LINALG::Matrix<NUMNOD_SOP5, NUMDIM_SOP5> xdisp;
      DRT::Node** nodes = Nodes();
      for (int i = 0; i < NUMNOD_SOP5; ++i)
      {
        const double* x = nodes[i]->X();
        xrefe(i, 0) = x[0];
        xrefe(i, 1) = x[1];
        xrefe(i, 2) = x[2];

        xcurr(i, 0) = xrefe(i, 0) + mydisp[i * NODDOF_SOP5 + 0];
        xcurr(i, 1) = xrefe(i, 1) + mydisp[i * NODDOF_SOP5 + 1];
        xcurr(i, 2) = xrefe(i, 2) + mydisp[i * NODDOF_SOP5 + 2];

        if (::UTILS::PRESTRESS::IsMulf(pstype_))
        {
          xdisp(i, 0) = mydisp[i * NODDOF_SOP5 + 0];
          xdisp(i, 1) = mydisp[i * NODDOF_SOP5 + 1];
          xdisp(i, 2) = mydisp[i * NODDOF_SOP5 + 2];
        }
      }


      //****************************************************************************
      // deformation gradient at centroid of element
      //****************************************************************************
      double detF_0 = -1.0;
      LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> invdefgrd_0;
      LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5> N_XYZ_0;
      // element coordinate derivatives at centroid
      LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5> N_rst_0;
      DRT::UTILS::shape_function_3D_deriv1(N_rst_0, 0.0, 0.0, 0.25, pyramid5);
      {
        // inverse jacobian matrix at centroid
        LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> invJ_0;
        invJ_0.Multiply(N_rst_0, xrefe);
        invJ_0.Invert();
        // material derivatives at centroid
        N_XYZ_0.Multiply(invJ_0, N_rst_0);
      }

      if (::UTILS::PRESTRESS::IsMulf(pstype_))
      {
        // get Jacobian mapping wrt to the stored configuration
        // centroid is 9th Gaussian point in storage
        LINALG::Matrix<3, 3> invJdef_0;
        prestress_->StoragetoMatrix(NUMGPT_SOP5, invJdef_0, prestress_->JHistory());
        // get derivatives wrt to last spatial configuration
        LINALG::Matrix<3, 5> N_xyz_0;
        N_xyz_0.Multiply(invJdef_0, N_rst_0);  // if (!Id()) std::cout << invJdef_0;

        // build multiplicative incremental defgrd
        LINALG::Matrix<3, 3> defgrd_0(false);
        defgrd_0.MultiplyTT(xdisp, N_xyz_0);
        defgrd_0(0, 0) += 1.0;
        defgrd_0(1, 1) += 1.0;
        defgrd_0(2, 2) += 1.0;

        // get stored old incremental F
        LINALG::Matrix<3, 3> Fhist;
        prestress_->StoragetoMatrix(NUMGPT_SOP5, Fhist, prestress_->FHistory());

        // build total defgrd = delta F * F_old
        LINALG::Matrix<3, 3> tmp;
        tmp.Multiply(defgrd_0, Fhist);
        defgrd_0 = tmp;

        // build inverse and detF
        invdefgrd_0.Invert(defgrd_0);
        detF_0 = defgrd_0.Determinant();
      }
      else  // no prestressing
      {
        // deformation gradient and its determinant at centroid
        LINALG::Matrix<3, 3> defgrd_0(false);
        defgrd_0.MultiplyTT(xcurr, N_XYZ_0);
        invdefgrd_0.Invert(defgrd_0);
        detF_0 = defgrd_0.Determinant();
      }



      // loop over all Gauss points
      for (int gp = 0; gp < NUMGPT_SOP5; gp++)
      {
        // Gauss weights and Jacobian determinant
        double fac = detJ_[gp] * weights[gp];

        /* get the inverse of the Jacobian matrix which looks like:
        **            [ x_,r  y_,r  z_,r ]^-1
        **     J^-1 = [ x_,s  y_,s  z_,s ]
        **            [ x_,t  y_,t  z_,t ]
        */
        // compute derivatives N_XYZ at gp w.r.t. material coordinates
        // by N_XYZ = J^-1 * N_rst
        LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5> N_XYZ(true);
        N_XYZ.Multiply(invJ_[gp], derivs[gp]);

        // (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
        LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> defgrd(true);
        defgrd.MultiplyTT(xcurr, N_XYZ);

        // F_bar deformation gradient =(detF_0/detF)^1/3*F
        double detF = defgrd.Determinant();
        LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> defgrd_bar(defgrd);
        double f_bar_factor = pow(detF_0 / detF, 1.0 / 3.0);
        defgrd_bar.Scale(f_bar_factor);

        // right Cauchy-Green tensor = F^T * F
        LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> cauchygreen;
        cauchygreen.MultiplyTN(defgrd_bar, defgrd_bar);

        // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
        // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
        LINALG::Matrix<MAT::NUM_STRESS_3D, 1> glstrain;
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


    default:
      dserror("Unknown type of action for So_pyramid5fbar");
      break;
  }
  return 0;
}


/*----------------------------------------------------------------------*
 |  init the element jacobian mapping (protected)           seitz 03/15 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_pyramid5fbar::InitJacobianMapping()
{
  const static std::vector<LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5>> derivs = sop5_derivs();
  LINALG::Matrix<NUMNOD_SOP5, NUMDIM_SOP5> xrefe;
  for (int i = 0; i < NUMNOD_SOP5; ++i)
  {
    xrefe(i, 0) = Nodes()[i]->X()[0];
    xrefe(i, 1) = Nodes()[i]->X()[1];
    xrefe(i, 2) = Nodes()[i]->X()[2];
  }
  invJ_.resize(NUMGPT_SOP5);
  detJ_.resize(NUMGPT_SOP5);
  for (int gp = 0; gp < NUMGPT_SOP5; ++gp)
  {
    // invJ_[gp].Shape(NUMDIM_SOP5,NUMDIM_SOP5);
    invJ_[gp].Multiply(derivs[gp], xrefe);
    detJ_[gp] = invJ_[gp].Invert();
    if (detJ_[gp] <= 0.0) dserror("Element Jacobian mapping %10.5e <= 0.0", detJ_[gp]);

    if (::UTILS::PRESTRESS::IsMulfActive(time_, pstype_, pstime_))
      if (!(prestress_->IsInit()))
        prestress_->MatrixtoStorage(gp, invJ_[gp], prestress_->JHistory());
  }

  // init the centroid invJ
  if (::UTILS::PRESTRESS::IsMulfActive(time_, pstype_, pstime_))
    if (!(prestress_->IsInit()))
    {
      LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5> N_rst_0;
      DRT::UTILS::shape_function_3D_deriv1(N_rst_0, 0.0, 0.0, 0.25, pyramid5);
      LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> invJ_0;
      invJ_0.Multiply(N_rst_0, xrefe);
      invJ_0.Invert();
      prestress_->MatrixtoStorage(NUMGPT_SOP5, invJ_0, prestress_->JHistory());
    }


  if (::UTILS::PRESTRESS::IsMulfActive(time_)) prestress_->IsInit() = true;

  return;
}


/*----------------------------------------------------------------------*
 |  Integrate a Volume Neumann boundary condition (public)               |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::So_pyramid5fbar::EvaluateNeumann(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Condition& condition, std::vector<int>& lm,
    Epetra_SerialDenseVector& elevec1, Epetra_SerialDenseMatrix* elemat1)
{
  // get values and switches from the condition
  const auto* onoff = condition.Get<std::vector<int>>("onoff");
  const auto* val = condition.Get<std::vector<double>>("val");

  /*
  **    TIME CURVE BUSINESS
  */
  // find out whether we will use a time curve
  const double time = params.get("total time", -1.0);

  // (SPATIAL) FUNCTION BUSINESS
  const auto* funct = condition.Get<std::vector<int>>("funct");
  LINALG::Matrix<NUMDIM_SOP5, 1> xrefegp(false);
  bool havefunct = false;
  if (funct)
    for (int dim = 0; dim < NUMDIM_SOP5; dim++)
      if ((*funct)[dim] > 0) havefunct = havefunct or true;

  /* ================================================================================*
  ** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for PYRAMID_% with ? GAUSS POINTS*
  ** ================================================================================*/
  const static std::vector<LINALG::Matrix<NUMNOD_SOP5, 1>> shapefcts = sop5_shapefcts();
  const static std::vector<LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5>> derivs = sop5_derivs();
  const static std::vector<double> gpweights = sop5_weights();
  /* ============================================================================*/

  // update element geometry
  LINALG::Matrix<NUMNOD_SOP5, NUMDIM_SOP5> xrefe;  // material coord. of element
  DRT::Node** nodes = Nodes();
  for (int i = 0; i < NUMNOD_SOP5; ++i)
  {
    const double* x = nodes[i]->X();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];
  }
  /* ================================================= Loop over Gauss Points */
  for (int gp = 0; gp < NUMGPT_SOP5; ++gp)
  {
    // compute the Jacobian matrix
    LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> jac;
    jac.Multiply(derivs[gp], xrefe);

    // compute determinant of Jacobian
    const double detJ = jac.Determinant();
    if (detJ == 0.0)
      dserror("ZERO JACOBIAN DETERMINANT");
    else if (detJ < 0.0)
      dserror("NEGATIVE JACOBIAN DETERMINANT");

    // material/reference co-ordinates of Gauss point
    if (havefunct)
    {
      for (int dim = 0; dim < NUMDIM_SOP5; dim++)
      {
        xrefegp(dim) = 0.0;
        for (int nodid = 0; nodid < NUMNOD_SOP5; ++nodid)
          xrefegp(dim) += shapefcts[gp](nodid) * xrefe(nodid, dim);
      }
    }

    // integration factor
    const double fac = gpweights[gp] * detJ;
    // distribute/add over element load vector
    for (int dim = 0; dim < NUMDIM_SOP5; dim++)
    {
      // function evaluation
      const int functnum = (funct) ? (*funct)[dim] : -1;
      const double functfac =
          (functnum > 0) ? DRT::Problem::Instance()
                               ->FunctionById<DRT::UTILS::FunctionOfSpaceTime>(functnum - 1)
                               .Evaluate(xrefegp.A(), time, dim)
                         : 1.0;
      const double dim_fac = (*onoff)[dim] * (*val)[dim] * fac * functfac;
      for (int nodid = 0; nodid < NUMNOD_SOP5; ++nodid)
      {
        elevec1[nodid * NUMDIM_SOP5 + dim] += shapefcts[gp](nodid) * dim_fac;
      }
    }

  } /* ==================================================== end of Loop over GP */

  return 0;
}  // DRT::ELEMENTS::So_pyramid5fbar::EvaluateNeumann


/*----------------------------------------------------------------------*
 |  evaluate the element (private)                                      |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_pyramid5fbar::nlnstiffmass(std::vector<int>& lm,  // location matrix
    std::vector<double>& disp,                                           // current displacements
    std::vector<double>* vel,                                            // current velocities
    std::vector<double>* acc,                                            // current accelerations
    std::vector<double>& residual,                                       // current residual displ
    std::vector<double>& dispmat,                                // current material displacements
    LINALG::Matrix<NUMDOF_SOP5, NUMDOF_SOP5>* stiffmatrix,       // element stiffness matrix
    LINALG::Matrix<NUMDOF_SOP5, NUMDOF_SOP5>* massmatrix,        // element mass matrix
    LINALG::Matrix<NUMDOF_SOP5, 1>* force,                       // element internal force vector
    LINALG::Matrix<NUMDOF_SOP5, 1>* forceinert,                  // element inertial force vector
    LINALG::Matrix<NUMDOF_SOP5, 1>* force_str,                   // element structural force vector
    LINALG::Matrix<NUMGPT_SOP5, MAT::NUM_STRESS_3D>* elestress,  // stresses at GP
    LINALG::Matrix<NUMGPT_SOP5, MAT::NUM_STRESS_3D>* elestrain,  // strains at GP
    LINALG::Matrix<NUMGPT_SOP5, MAT::NUM_STRESS_3D>* eleplstrain,  // plastic strains at GP
    Teuchos::ParameterList& params,           // algorithmic parameters e.g. time
    const INPAR::STR::StressType iostress,    // stress output option
    const INPAR::STR::StrainType iostrain,    // strain output option
    const INPAR::STR::StrainType ioplstrain)  // plastic strain output option
{
  /* ================================================================================*
  ** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for PYRAMID_5 with ? GAUSS POINTS*
  ** ================================================================================*/
  const static std::vector<LINALG::Matrix<NUMNOD_SOP5, 1>> shapefcts = sop5_shapefcts();
  const static std::vector<LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5>> derivs = sop5_derivs();
  const static std::vector<double> gpweights = sop5_weights();
  /* ============================================================================*/

  // update element geometry
  LINALG::Matrix<NUMNOD_SOP5, NUMDIM_SOP5> xrefe;  // material coord. of element
  LINALG::Matrix<NUMNOD_SOP5, NUMDIM_SOP5> xcurr;  // current  coord. of element
  LINALG::Matrix<NUMNOD_SOP5, NUMDIM_SOP5> xdisp;
  DRT::Node** nodes = Nodes();
  for (int i = 0; i < NUMNOD_SOP5; ++i)
  {
    const double* x = nodes[i]->X();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];

    xcurr(i, 0) = xrefe(i, 0) + disp[i * NODDOF_SOP5 + 0];
    xcurr(i, 1) = xrefe(i, 1) + disp[i * NODDOF_SOP5 + 1];
    xcurr(i, 2) = xrefe(i, 2) + disp[i * NODDOF_SOP5 + 2];

    if (::UTILS::PRESTRESS::IsMulf(pstype_))
    {
      xdisp(i, 0) = disp[i * NODDOF_SOP5 + 0];
      xdisp(i, 1) = disp[i * NODDOF_SOP5 + 1];
      xdisp(i, 2) = disp[i * NODDOF_SOP5 + 2];
    }
  }

  //****************************************************************************
  // deformation gradient at centroid of element
  //****************************************************************************
  double detF_0 = -1.0;
  LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> invdefgrd_0;
  LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5> N_XYZ_0;
  // element coordinate derivatives at centroid
  LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5> N_rst_0;
  DRT::UTILS::shape_function_3D_deriv1(N_rst_0, 0.0, 0.0, 0.25, pyramid5);
  {
    // inverse jacobian matrix at centroid
    LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> invJ_0;
    invJ_0.Multiply(N_rst_0, xrefe);
    invJ_0.Invert();
    // material derivatives at centroid
    N_XYZ_0.Multiply(invJ_0, N_rst_0);
  }

  if (::UTILS::PRESTRESS::IsMulf(pstype_))
  {
    // get Jacobian mapping wrt to the stored configuration
    // centroid is 9th Gaussian point in storage
    LINALG::Matrix<3, 3> invJdef_0;
    prestress_->StoragetoMatrix(NUMGPT_SOP5, invJdef_0, prestress_->JHistory());
    // get derivatives wrt to last spatial configuration
    LINALG::Matrix<3, 5> N_xyz_0;
    N_xyz_0.Multiply(invJdef_0, N_rst_0);  // if (!Id()) std::cout << invJdef_0;

    // build multiplicative incremental defgrd
    LINALG::Matrix<3, 3> defgrd_0(false);
    defgrd_0.MultiplyTT(xdisp, N_xyz_0);
    defgrd_0(0, 0) += 1.0;
    defgrd_0(1, 1) += 1.0;
    defgrd_0(2, 2) += 1.0;

    // get stored old incremental F
    LINALG::Matrix<3, 3> Fhist;
    prestress_->StoragetoMatrix(NUMGPT_SOP5, Fhist, prestress_->FHistory());

    // build total defgrd = delta F * F_old
    LINALG::Matrix<3, 3> tmp;
    tmp.Multiply(defgrd_0, Fhist);
    defgrd_0 = tmp;

    // build inverse and detF
    invdefgrd_0.Invert(defgrd_0);
    detF_0 = defgrd_0.Determinant();
  }
  else  // no prestressing
  {
    // deformation gradient and its determinant at centroid
    LINALG::Matrix<3, 3> defgrd_0(false);
    defgrd_0.MultiplyTT(xcurr, N_XYZ_0);
    invdefgrd_0.Invert(defgrd_0);
    detF_0 = defgrd_0.Determinant();
  }
  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5> N_XYZ;
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> defgrd(false);
  for (int gp = 0; gp < NUMGPT_SOP5; ++gp)
  {
    /* get the inverse of the Jacobian matrix which looks like:
    **            [ x_,r  y_,r  z_,r ]^-1
    **     J^-1 = [ x_,s  y_,s  z_,s ]
    **            [ x_,t  y_,t  z_,t ]
    */
    // compute derivatives N_XYZ at gp w.r.t. material coordinates
    // by N_XYZ = J^-1 * N_rst
    N_XYZ.Multiply(invJ_[gp], derivs[gp]);
    double detJ = detJ_[gp];

    if (::UTILS::PRESTRESS::IsMulf(pstype_))
    {
      // get Jacobian mapping wrt to the stored configuration
      LINALG::Matrix<3, 3> invJdef;
      prestress_->StoragetoMatrix(gp, invJdef, prestress_->JHistory());
      // get derivatives wrt to last spatial configuration
      LINALG::Matrix<3, 5> N_xyz;
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
    else  // no prestressing
    {
      // (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
      defgrd.MultiplyTT(xcurr, N_XYZ);
    }
    LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> invdefgrd;
    invdefgrd.Invert(defgrd);
    double detF = defgrd.Determinant();

    // Right Cauchy-Green tensor = F^T * F
    LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> cauchygreen;
    cauchygreen.MultiplyTN(defgrd, defgrd);

    // check for negative jacobian
    if (detF_0 < 0. || detF < 0.)
    {
      // check, if errors are tolerated or should throw a dserror
      bool error_tol = false;
      if (params.isParameter("tolerate_errors")) error_tol = params.get<bool>("tolerate_errors");
      if (error_tol)
      {
        params.set<bool>("eval_error", true);
        stiffmatrix->Clear();
        force->Clear();
        return;
      }
      else
        dserror("negative jacobian determinant");
    }
    // F_bar deformation gradient =(detF_0/detF)^1/3*F
    LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> defgrd_bar(defgrd);
    double f_bar_factor = pow(detF_0 / detF, 1.0 / 3.0);
    defgrd_bar.Scale(f_bar_factor);

    // Right Cauchy-Green tensor(Fbar) = F_bar^T * F_bar
    LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> cauchygreen_bar;
    cauchygreen_bar.MultiplyTN(defgrd_bar, defgrd_bar);

    // Green-Lagrange strains(F_bar) matrix E = 0.5 * (Cauchygreen(F_bar) - Identity)
    // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
    Epetra_SerialDenseVector glstrain_bar_epetra(MAT::NUM_STRESS_3D);
    LINALG::Matrix<MAT::NUM_STRESS_3D, 1> glstrain_bar(glstrain_bar_epetra.A(), true);
    glstrain_bar(0) = 0.5 * (cauchygreen_bar(0, 0) - 1.0);
    glstrain_bar(1) = 0.5 * (cauchygreen_bar(1, 1) - 1.0);
    glstrain_bar(2) = 0.5 * (cauchygreen_bar(2, 2) - 1.0);
    glstrain_bar(3) = cauchygreen_bar(0, 1);
    glstrain_bar(4) = cauchygreen_bar(1, 2);
    glstrain_bar(5) = cauchygreen_bar(2, 0);

    // return gp strains (only in case of stress/strain output)
    switch (iostrain)
    {
      case INPAR::STR::strain_gl:
      {
        if (elestrain == nullptr) dserror("strain data not available");
        for (int i = 0; i < 3; ++i) (*elestrain)(gp, i) = glstrain_bar(i);
        for (int i = 3; i < 6; ++i) (*elestrain)(gp, i) = 0.5 * glstrain_bar(i);
      }
      break;
      case INPAR::STR::strain_ea:
      {
        if (elestrain == nullptr) dserror("strain data not available");
        // rewriting Green-Lagrange strains in matrix format
        LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> gl_bar;
        gl_bar(0, 0) = glstrain_bar(0);
        gl_bar(0, 1) = 0.5 * glstrain_bar(3);
        gl_bar(0, 2) = 0.5 * glstrain_bar(5);
        gl_bar(1, 0) = gl_bar(0, 1);
        gl_bar(1, 1) = glstrain_bar(1);
        gl_bar(1, 2) = 0.5 * glstrain_bar(4);
        gl_bar(2, 0) = gl_bar(0, 2);
        gl_bar(2, 1) = gl_bar(1, 2);
        gl_bar(2, 2) = glstrain_bar(2);

        // inverse of fbar deformation gradient
        LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> invdefgrd_bar;
        invdefgrd_bar.Invert(defgrd_bar);

        LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> temp;
        LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> euler_almansi_bar;
        temp.Multiply(gl_bar, invdefgrd_bar);
        euler_almansi_bar.MultiplyTN(invdefgrd_bar, temp);

        (*elestrain)(gp, 0) = euler_almansi_bar(0, 0);
        (*elestrain)(gp, 1) = euler_almansi_bar(1, 1);
        (*elestrain)(gp, 2) = euler_almansi_bar(2, 2);
        (*elestrain)(gp, 3) = euler_almansi_bar(0, 1);
        (*elestrain)(gp, 4) = euler_almansi_bar(1, 2);
        (*elestrain)(gp, 5) = euler_almansi_bar(0, 2);
      }
      break;
      case INPAR::STR::strain_log:
      {
        if (elestrain == nullptr) dserror("strain data not available");

        /// the Eularian logarithmic strain is defined as the natural logarithm of the left stretch
        /// tensor [1,2]: \f[
        ///    e_{log} = e_{hencky} = ln (\mathbf{V}) = \sum_{i=1}^3 (ln \lambda_i) \mathbf{n}_i
        ///    \otimes \mathbf{n}_i
        /// \f]
        ///< h3>References</h3>
        /// <ul>
        /// <li> [1] H. Xiao, Beijing, China, O. T. Bruhns and A. Meyers (1997) Logarithmic strain,
        /// logarithmic spin and logarithmic rate, Eq. 5 <li> [2] Caminero et al. (2011) Modeling
        /// large strain anisotropic elasto-plasticity with logarithmic strain and stress measures,
        /// Eq. 70
        /// </ul>
        ///
        /// \author HdV
        /// \date 08/13

        // eigenvalue decomposition (from elasthyper.cpp)
        LINALG::Matrix<3, 3> prstr2(true);  // squared principal stretches
        LINALG::Matrix<3, 1> prstr(true);   // principal stretch
        LINALG::Matrix<3, 3> prdir(true);   // principal directions
        LINALG::SYEV(cauchygreen, prstr2, prdir);

        // THE principal stretches
        for (int al = 0; al < 3; ++al) prstr(al) = std::sqrt(prstr2(al, al));

        // populating the logarithmic strain matrix
        LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> lnv(true);

        // checking if cauchy green is correctly determined to ensure eigen vectors in correct
        // direction i.e. a flipped eigenvector is also a valid solution C = \sum_{i=1}^3
        // (\lambda_i^2) \mathbf{n}_i \otimes \mathbf{n}_i
        LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> tempCG(true);

        for (int k = 0; k < 3; ++k)
        {
          double n_00, n_01, n_02, n_11, n_12, n_22 = 0.0;

          n_00 = prdir(0, k) * prdir(0, k);
          n_01 = prdir(0, k) * prdir(1, k);
          n_02 = prdir(0, k) * prdir(2, k);
          n_11 = prdir(1, k) * prdir(1, k);
          n_12 = prdir(1, k) * prdir(2, k);
          n_22 = prdir(2, k) * prdir(2, k);

          // only compute the symmetric components from a single eigenvector,
          // because eigenvalue directions are not consistent (it can be flipped)
          tempCG(0, 0) += (prstr(k)) * (prstr(k)) * n_00;
          tempCG(0, 1) += (prstr(k)) * (prstr(k)) * n_01;
          tempCG(0, 2) += (prstr(k)) * (prstr(k)) * n_02;
          tempCG(1, 0) += (prstr(k)) * (prstr(k)) * n_01;  // symmetry
          tempCG(1, 1) += (prstr(k)) * (prstr(k)) * n_11;
          tempCG(1, 2) += (prstr(k)) * (prstr(k)) * n_12;
          tempCG(2, 0) += (prstr(k)) * (prstr(k)) * n_02;  // symmetry
          tempCG(2, 1) += (prstr(k)) * (prstr(k)) * n_12;  // symmetry
          tempCG(2, 2) += (prstr(k)) * (prstr(k)) * n_22;

          // Computation of the Logarithmic strain tensor

          lnv(0, 0) += (std::log(prstr(k))) * n_00;
          lnv(0, 1) += (std::log(prstr(k))) * n_01;
          lnv(0, 2) += (std::log(prstr(k))) * n_02;
          lnv(1, 0) += (std::log(prstr(k))) * n_01;  // symmetry
          lnv(1, 1) += (std::log(prstr(k))) * n_11;
          lnv(1, 2) += (std::log(prstr(k))) * n_12;
          lnv(2, 0) += (std::log(prstr(k))) * n_02;  // symmetry
          lnv(2, 1) += (std::log(prstr(k))) * n_12;  // symmetry
          lnv(2, 2) += (std::log(prstr(k))) * n_22;
        }

        // compare CG computed with deformation gradient with CG computed
        // with eigenvalues and -vectors to determine/ensure the correct
        // orientation of the eigen vectors
        LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> diffCG(true);

        for (int i = 0; i < 3; ++i)
        {
          for (int j = 0; j < 3; ++j)
          {
            diffCG(i, j) = cauchygreen(i, j) - tempCG(i, j);
            // the solution to this problem is to evaluate the cauchygreen tensor with tempCG
            // computed with every combination of eigenvector orientations -- up to nine comparisons
            if (diffCG(i, j) > 1e-10)
              dserror(
                  "eigenvector orientation error with the diffCG giving problems: %10.5e \n BUILD "
                  "SOLUTION TO FIX IT",
                  diffCG(i, j));
          }
        }

        (*elestrain)(gp, 0) = lnv(0, 0);
        (*elestrain)(gp, 1) = lnv(1, 1);
        (*elestrain)(gp, 2) = lnv(2, 2);
        (*elestrain)(gp, 3) = lnv(0, 1);
        (*elestrain)(gp, 4) = lnv(1, 2);
        (*elestrain)(gp, 5) = lnv(0, 2);
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
    LINALG::Matrix<MAT::NUM_STRESS_3D, NUMDOF_SOP5> bop;
    for (int i = 0; i < NUMNOD_SOP5; ++i)
    {
      bop(0, NODDOF_SOP5 * i + 0) = defgrd(0, 0) * N_XYZ(0, i);
      bop(0, NODDOF_SOP5 * i + 1) = defgrd(1, 0) * N_XYZ(0, i);
      bop(0, NODDOF_SOP5 * i + 2) = defgrd(2, 0) * N_XYZ(0, i);
      bop(1, NODDOF_SOP5 * i + 0) = defgrd(0, 1) * N_XYZ(1, i);
      bop(1, NODDOF_SOP5 * i + 1) = defgrd(1, 1) * N_XYZ(1, i);
      bop(1, NODDOF_SOP5 * i + 2) = defgrd(2, 1) * N_XYZ(1, i);
      bop(2, NODDOF_SOP5 * i + 0) = defgrd(0, 2) * N_XYZ(2, i);
      bop(2, NODDOF_SOP5 * i + 1) = defgrd(1, 2) * N_XYZ(2, i);
      bop(2, NODDOF_SOP5 * i + 2) = defgrd(2, 2) * N_XYZ(2, i);
      /* ~~~ */
      bop(3, NODDOF_SOP5 * i + 0) = defgrd(0, 0) * N_XYZ(1, i) + defgrd(0, 1) * N_XYZ(0, i);
      bop(3, NODDOF_SOP5 * i + 1) = defgrd(1, 0) * N_XYZ(1, i) + defgrd(1, 1) * N_XYZ(0, i);
      bop(3, NODDOF_SOP5 * i + 2) = defgrd(2, 0) * N_XYZ(1, i) + defgrd(2, 1) * N_XYZ(0, i);
      bop(4, NODDOF_SOP5 * i + 0) = defgrd(0, 1) * N_XYZ(2, i) + defgrd(0, 2) * N_XYZ(1, i);
      bop(4, NODDOF_SOP5 * i + 1) = defgrd(1, 1) * N_XYZ(2, i) + defgrd(1, 2) * N_XYZ(1, i);
      bop(4, NODDOF_SOP5 * i + 2) = defgrd(2, 1) * N_XYZ(2, i) + defgrd(2, 2) * N_XYZ(1, i);
      bop(5, NODDOF_SOP5 * i + 0) = defgrd(0, 2) * N_XYZ(0, i) + defgrd(0, 0) * N_XYZ(2, i);
      bop(5, NODDOF_SOP5 * i + 1) = defgrd(1, 2) * N_XYZ(0, i) + defgrd(1, 0) * N_XYZ(2, i);
      bop(5, NODDOF_SOP5 * i + 2) = defgrd(2, 2) * N_XYZ(0, i) + defgrd(2, 0) * N_XYZ(2, i);
    }

    // call material law
    LINALG::Matrix<MAT::NUM_STRESS_3D, MAT::NUM_STRESS_3D> cmat(true);
    LINALG::Matrix<MAT::NUM_STRESS_3D, 1> stress_bar(true);

    // in case of temperature-dependent material parameters, e.g. Young's modulus,
    // i.e. E(T), current element temperature T_{n+1} required for stress and cmat
    if (Material()->MaterialType() == INPAR::MAT::m_thermoplhyperelast)
    {
      GetTemperatureForStructuralMaterial(shapefcts[gp], params);
    }

    if (Material()->MaterialType() == INPAR::MAT::m_constraintmixture ||
        Material()->MaterialType() == INPAR::MAT::m_mixture)
    {
      // gp reference coordinates
      LINALG::Matrix<NUMNOD_SOP5, 1> funct(true);
      funct = shapefcts[gp];
      LINALG::Matrix<1, NUMDIM_SOP5> point(true);
      point.MultiplyTN(funct, xrefe);
      params.set("gprefecoord", point);
    }

    SolidMaterial()->Evaluate(&defgrd_bar, &glstrain_bar, params, &stress_bar, &cmat, gp, Id());
    // end of call material law

    // print plastic strains
    // CAUTION: print plastic strains ONLY in case of small strain regime!
    switch (ioplstrain)
    {
      case INPAR::STR::strain_gl:
      {
        if (eleplstrain == nullptr) dserror("plastic strain data not available");
        LINALG::Matrix<MAT::NUM_STRESS_3D, 1> plglstrain_bar =
            params.get<LINALG::Matrix<MAT::NUM_STRESS_3D, 1>>("plglstrain");
        for (int i = 0; i < 3; ++i) (*eleplstrain)(gp, i) = plglstrain_bar(i);
        for (int i = 3; i < 6; ++i) (*eleplstrain)(gp, i) = 0.5 * plglstrain_bar(i);
      }
      break;
      case INPAR::STR::strain_ea:
      {
        if (eleplstrain == nullptr) dserror("plastic strain data not available");
        LINALG::Matrix<MAT::NUM_STRESS_3D, 1> plglstrain_bar =
            params.get<LINALG::Matrix<MAT::NUM_STRESS_3D, 1>>("plglstrain");

        // e = F^{T-1} . E . F^{-1}
        LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> euler_almansi_bar;
        GLtoEA(&plglstrain_bar, &defgrd_bar, &euler_almansi_bar);

        (*eleplstrain)(gp, 0) = euler_almansi_bar(0, 0);
        (*eleplstrain)(gp, 1) = euler_almansi_bar(1, 1);
        (*eleplstrain)(gp, 2) = euler_almansi_bar(2, 2);
        (*eleplstrain)(gp, 3) = euler_almansi_bar(0, 1);
        (*eleplstrain)(gp, 4) = euler_almansi_bar(1, 2);
        (*eleplstrain)(gp, 5) = euler_almansi_bar(0, 2);
      }
      break;
      case INPAR::STR::strain_none:
        break;

      default:
        dserror("requested plastic strain type not available");
        break;
    }  // switch (ioplstrain)

    // return gp stresses
    switch (iostress)
    {
      case INPAR::STR::stress_2pk:
      {
        if (elestress == nullptr) dserror("stress data not available");
        for (int i = 0; i < MAT::NUM_STRESS_3D; ++i) (*elestress)(gp, i) = stress_bar(i);
      }
      break;
      case INPAR::STR::stress_cauchy:
      {
        if (elestress == nullptr) dserror("stress data not available");
        const double detF_bar = defgrd_bar.Determinant();

        LINALG::Matrix<3, 3> pkstress_bar;
        pkstress_bar(0, 0) = stress_bar(0);
        pkstress_bar(0, 1) = stress_bar(3);
        pkstress_bar(0, 2) = stress_bar(5);
        pkstress_bar(1, 0) = pkstress_bar(0, 1);
        pkstress_bar(1, 1) = stress_bar(1);
        pkstress_bar(1, 2) = stress_bar(4);
        pkstress_bar(2, 0) = pkstress_bar(0, 2);
        pkstress_bar(2, 1) = pkstress_bar(1, 2);
        pkstress_bar(2, 2) = stress_bar(2);

        LINALG::Matrix<3, 3> temp;
        LINALG::Matrix<3, 3> cauchystress_bar;
        temp.Multiply(1.0 / detF_bar, defgrd_bar, pkstress_bar);
        cauchystress_bar.MultiplyNT(temp, defgrd_bar);

        (*elestress)(gp, 0) = cauchystress_bar(0, 0);
        (*elestress)(gp, 1) = cauchystress_bar(1, 1);
        (*elestress)(gp, 2) = cauchystress_bar(2, 2);
        (*elestress)(gp, 3) = cauchystress_bar(0, 1);
        (*elestress)(gp, 4) = cauchystress_bar(1, 2);
        (*elestress)(gp, 5) = cauchystress_bar(0, 2);
      }
      break;
      case INPAR::STR::stress_none:
        break;
      default:
        dserror("requested stress type not available");
        break;
    }

    double detJ_w = detJ * gpweights[gp];

    // update internal force vector
    if (force != nullptr)
    {
      // integrate internal force vector f = f + (B^T . sigma) * detJ * w(gp)
      force->MultiplyTN(detJ_w / f_bar_factor, bop, stress_bar, 1.0);
    }

    // update stiffness matrix
    if (stiffmatrix != nullptr)
    {
      // integrate `elastic' and `initial-displacement' stiffness matrix
      // keu = keu + (B^T . C . B) * detJ * w(gp)
      LINALG::Matrix<6, NUMDOF_SOP5> cb;
      cb.Multiply(cmat, bop);
      stiffmatrix->MultiplyTN(detJ_w * f_bar_factor, bop, cb, 1.0);

      // integrate `geometric' stiffness matrix and add to keu *****************
      LINALG::Matrix<6, 1> sfac(stress_bar);  // auxiliary integrated stress
      sfac.Scale(detJ_w / f_bar_factor);      // detJ*w(gp)*[S11,S22,S33,S12=S21,S23=S32,S13=S31]
      std::vector<double> SmB_L(3);           // intermediate Sm.B_L
      // kgeo += (B_L^T . sigma . B_L) * detJ * w(gp)  with B_L = Ni,Xj see NiliFEM-Skript
      for (int inod = 0; inod < NUMNOD_SOP5; ++inod)
      {
        SmB_L[0] = sfac(0) * N_XYZ(0, inod) + sfac(3) * N_XYZ(1, inod) + sfac(5) * N_XYZ(2, inod);
        SmB_L[1] = sfac(3) * N_XYZ(0, inod) + sfac(1) * N_XYZ(1, inod) + sfac(4) * N_XYZ(2, inod);
        SmB_L[2] = sfac(5) * N_XYZ(0, inod) + sfac(4) * N_XYZ(1, inod) + sfac(2) * N_XYZ(2, inod);
        for (int jnod = 0; jnod < NUMNOD_SOP5; ++jnod)
        {
          double bopstrbop = 0.0;  // intermediate value
          for (int idim = 0; idim < NUMDIM_SOP5; ++idim)
            bopstrbop += N_XYZ(idim, jnod) * SmB_L[idim];
          (*stiffmatrix)(3 * inod + 0, 3 * jnod + 0) += bopstrbop;
          (*stiffmatrix)(3 * inod + 1, 3 * jnod + 1) += bopstrbop;
          (*stiffmatrix)(3 * inod + 2, 3 * jnod + 2) += bopstrbop;
        }
      }  // end of integrate `geometric' stiffness******************************

      // integrate additional fbar matrix
      LINALG::Matrix<MAT::NUM_STRESS_3D, 1> cauchygreenvector;
      cauchygreenvector(0) = cauchygreen(0, 0);
      cauchygreenvector(1) = cauchygreen(1, 1);
      cauchygreenvector(2) = cauchygreen(2, 2);
      cauchygreenvector(3) = 2 * cauchygreen(0, 1);
      cauchygreenvector(4) = 2 * cauchygreen(1, 2);
      cauchygreenvector(5) = 2 * cauchygreen(2, 0);

      LINALG::Matrix<MAT::NUM_STRESS_3D, 1> ccg;
      ccg.Multiply(cmat, cauchygreenvector);

      LINALG::Matrix<NUMDOF_SOP5, 1> bopccg(false);  // auxiliary integrated stress
      bopccg.MultiplyTN(detJ_w * f_bar_factor / 3.0, bop, ccg);

      double htensor[NUMDOF_SOP5];
      for (int n = 0; n < NUMDOF_SOP5; n++)
      {
        htensor[n] = 0;
        for (int i = 0; i < NUMDIM_SOP5; i++)
        {
          htensor[n] +=
              invdefgrd_0(i, n % 3) * N_XYZ_0(i, n / 3) - invdefgrd(i, n % 3) * N_XYZ(i, n / 3);
        }
      }

      LINALG::Matrix<NUMDOF_SOP5, 1> bops(false);  // auxiliary integrated stress
      bops.MultiplyTN(-detJ_w / f_bar_factor / 3.0, bop, stress_bar);
      for (int i = 0; i < NUMDOF_SOP5; i++)
      {
        for (int j = 0; j < NUMDOF_SOP5; j++)
        {
          (*stiffmatrix)(i, j) += htensor[j] * (bops(i, 0) + bopccg(i, 0));
        }
      }  // end of integrate additional `fbar' stiffness**********************
    }    // if (stiffmatrix != nullptr)

    if (massmatrix != nullptr)  // evaluate mass matrix +++++++++++++++++++++++++
    {
      double density = Material()->Density(gp);
      // integrate consistent mass matrix
      const double factor = detJ_w * density;
      double ifactor, massfactor;
      for (int inod = 0; inod < NUMNOD_SOP5; ++inod)
      {
        ifactor = shapefcts[gp](inod) * factor;
        for (int jnod = 0; jnod < NUMNOD_SOP5; ++jnod)
        {
          massfactor = shapefcts[gp](jnod) * ifactor;  // intermediate factor
          (*massmatrix)(NUMDIM_SOP5 * inod + 0, NUMDIM_SOP5 * jnod + 0) += massfactor;
          (*massmatrix)(NUMDIM_SOP5 * inod + 1, NUMDIM_SOP5 * jnod + 1) += massfactor;
          (*massmatrix)(NUMDIM_SOP5 * inod + 2, NUMDIM_SOP5 * jnod + 2) += massfactor;
        }
      }

      // check for non constant mass matrix
      if (SolidMaterial()->VaryingDensity())
      {
        /*
         If the density, i.e. the mass matrix, is not constant, a linearization is neccessary.
         In general, the mass matrix can be dependent on the displacements, the velocities and the
         accelerations. We write all the additional terms into the mass matrix, hence, conversion
         from accelerations to velocities and displacements are needed. As those conversions depend
         on the time integration scheme, the factors are set within the respective time integrators
         and read from the parameter list inside the element (this is a little ugly...).
         */
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
        // Right Cauchy-Green tensor = F^T * F
        LINALG::Matrix<NUMDIM_SOP5, NUMDIM_SOP5> cauchygreen;
        cauchygreen.MultiplyTN(defgrd, defgrd);

        // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
        Epetra_SerialDenseVector glstrain_epetra(MAT::NUM_STRESS_3D);
        LINALG::Matrix<MAT::NUM_STRESS_3D, 1> glstrain(glstrain_epetra.A(), true);
        // if (kintype_ == INPAR::STR::kinem_nonlinearTotLag)
        //{
        // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
        glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
        glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
        glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
        glstrain(3) = cauchygreen(0, 1);
        glstrain(4) = cauchygreen(1, 2);
        glstrain(5) = cauchygreen(2, 0);
        //}

        SolidMaterial()->EvaluateNonLinMass(
            &defgrd, &glstrain, params, &linmass_disp, &linmass_vel, gp, Id());


        // multiply by 2.0 to get derivative w.r.t green lagrange strains and multiply by time
        // integration factor
        linmass_disp.Scale(2.0 * timintfac_dis);
        linmass_vel.Scale(2.0 * timintfac_vel);
        linmass.Update(1.0, linmass_disp, 1.0, linmass_vel, 0.0);

        // evaluate accelerations at time n+1 at gauss point
        LINALG::Matrix<NUMDIM_SOP5, 1> myacc(true);
        for (int idim = 0; idim < NUMDIM_SOP5; ++idim)
          for (int inod = 0; inod < NUMNOD_SOP5; ++inod)
            myacc(idim) += shapefcts[gp](inod) * (*acc)[idim + (inod * NUMDIM_SOP5)];

        if (stiffmatrix != nullptr)
        {
          // integrate linearisation of mass matrix
          //(B^T . d\rho/d disp . a) * detJ * w(gp)
          LINALG::Matrix<1, NUMDOF_SOP5> cb;
          cb.MultiplyTN(linmass_disp, bop);
          for (int inod = 0; inod < NUMNOD_SOP5; ++inod)
          {
            double factor = detJ_w * shapefcts[gp](inod);
            for (int idim = 0; idim < NUMDIM_SOP5; ++idim)
            {
              double massfactor = factor * myacc(idim);
              for (int jnod = 0; jnod < NUMNOD_SOP5; ++jnod)
                for (int jdim = 0; jdim < NUMDIM_SOP5; ++jdim)
                  (*massmatrix)(inod * NUMDIM_SOP5 + idim, jnod * NUMDIM_SOP5 + jdim) +=
                      massfactor * cb(jnod * NUMDIM_SOP5 + jdim);
            }
          }
        }

        // internal force vector without EAS terms
        if (forceinert != nullptr)
        {
          // integrate nonlinear inertia force term
          for (int inod = 0; inod < NUMNOD_SOP5; ++inod)
          {
            double forcefactor = shapefcts[gp](inod) * detJ_w;
            for (int idim = 0; idim < NUMDIM_SOP5; ++idim)
              (*forceinert)(inod * NUMDIM_SOP5 + idim) += forcefactor * density * myacc(idim);
          }
        }
      }
    }  // end of mass matrix +++++++++++++++++++++++++++++++++++++++++++++++++++

  } /* ==================================================== end of Loop over GP */

  return;
}  // DRT::ELEMENTS::So_pyramid5fbar::nlnstiffmass


/*----------------------------------------------------------------------*
 |  init the element (public)                                           |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::So_pyramid5fbarType::Initialize(DRT::Discretization& dis)
{
  for (int i = 0; i < dis.NumMyColElements(); ++i)
  {
    if (dis.lColElement(i)->ElementType() != *this) continue;
    auto* actele = dynamic_cast<DRT::ELEMENTS::So_pyramid5fbar*>(dis.lColElement(i));
    if (!actele) dserror("cast to So_pyramid5fbar* failed");
    actele->InitJacobianMapping();
  }
  return 0;
}


/*----------------------------------------------------------------------*
 | compute def gradient at every gaussian point (protected) seitz 03/15 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_pyramid5fbar::DefGradient(const std::vector<double>& disp,
    Epetra_SerialDenseMatrix& gpdefgrd, DRT::ELEMENTS::PreStress& prestress)
{
  const static std::vector<LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5>> derivs = sop5_derivs();
  // derivatives at centroid point
  LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5> N_rst_0;
  DRT::UTILS::shape_function_3D_deriv1(N_rst_0, 0.0, 0.0, 0.25, pyramid5);

  // update element geometry
  LINALG::Matrix<NUMNOD_SOP5, NUMDIM_SOP5> xdisp;  // current  coord. of element
  for (int i = 0; i < NUMNOD_SOP5; ++i)
  {
    xdisp(i, 0) = disp[i * NODDOF_SOP5 + 0];
    xdisp(i, 1) = disp[i * NODDOF_SOP5 + 1];
    xdisp(i, 2) = disp[i * NODDOF_SOP5 + 2];
  }

  for (int gp = 0; gp < NUMGPT_SOP5; ++gp)
  {
    // get Jacobian mapping wrt to the stored deformed configuration
    LINALG::Matrix<3, 3> invJdef;
    prestress.StoragetoMatrix(gp, invJdef, prestress.JHistory());

    // by N_XYZ = J^-1 * N_rst
    LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5> N_xyz;
    N_xyz.Multiply(invJdef, derivs[gp]);

    // build defgrd (independent of xrefe!)
    LINALG::Matrix<3, 3> defgrd;
    defgrd.MultiplyTT(xdisp, N_xyz);
    defgrd(0, 0) += 1.0;
    defgrd(1, 1) += 1.0;
    defgrd(2, 2) += 1.0;

    prestress.MatrixtoStorage(gp, defgrd, gpdefgrd);
  }

  {
    // get Jacobian mapping wrt to the stored deformed configuration
    LINALG::Matrix<3, 3> invJdef;
    prestress.StoragetoMatrix(NUMGPT_SOP5, invJdef, prestress.JHistory());

    // by N_XYZ = J^-1 * N_rst
    LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5> N_xyz;
    N_xyz.Multiply(invJdef, N_rst_0);

    // build defgrd (independent of xrefe!)
    LINALG::Matrix<3, 3> defgrd;
    defgrd.MultiplyTT(xdisp, N_xyz);
    defgrd(0, 0) += 1.0;
    defgrd(1, 1) += 1.0;
    defgrd(2, 2) += 1.0;

    prestress.MatrixtoStorage(NUMGPT_SOP5, defgrd, gpdefgrd);
  }

  return;
}


/*------------------------------------------------------------------------*
 | compute Jac.mapping wrt deformed configuration (protected) seitz 03/15 |
 *------------------------------------------------------------------------*/
void DRT::ELEMENTS::So_pyramid5fbar::UpdateJacobianMapping(
    const std::vector<double>& disp, DRT::ELEMENTS::PreStress& prestress)
{
  const static std::vector<LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5>> derivs = sop5_derivs();
  // derivatives at centroid
  LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5> N_rst_0;
  DRT::UTILS::shape_function_3D_deriv1(N_rst_0, 0.0, 0.0, 0.25, pyramid5);

  // get incremental disp
  LINALG::Matrix<NUMNOD_SOP5, NUMDIM_SOP5> xdisp;
  for (int i = 0; i < NUMNOD_SOP5; ++i)
  {
    xdisp(i, 0) = disp[i * NODDOF_SOP5 + 0];
    xdisp(i, 1) = disp[i * NODDOF_SOP5 + 1];
    xdisp(i, 2) = disp[i * NODDOF_SOP5 + 2];
  }

  LINALG::Matrix<3, 3> invJhist;
  LINALG::Matrix<3, 3> invJ;
  LINALG::Matrix<3, 3> defgrd;
  LINALG::Matrix<NUMDIM_SOP5, NUMNOD_SOP5> N_xyz;
  LINALG::Matrix<3, 3> invJnew;
  for (int gp = 0; gp < NUMGPT_SOP5; ++gp)
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
  }  // for (int gp=0; gp<NUMGPT_SOP5; ++gp)

  {
    // get the invJ old state
    prestress.StoragetoMatrix(NUMGPT_SOP5, invJhist, prestress.JHistory());
    // get derivatives wrt to invJhist
    N_xyz.Multiply(invJhist, N_rst_0);
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
    prestress.MatrixtoStorage(NUMGPT_SOP5, invJnew, prestress.JHistory());
  }

  return;
}

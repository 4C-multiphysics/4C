/*----------------------------------------------------------------------------*/
/*! \file
\brief Evaluate methods for 2D wall element for structure part of porous medium.

\level 2


*/
/*---------------------------------------------------------------------------*/


#include "baci_lib_discret.H"
#include "baci_linalg_utils_densematrix_multiply.H"
#include "baci_linalg_utils_sparse_algebra_math.H"
#include "baci_mat_fluidporo.H"
#include "baci_mat_fluidporo_multiphase.H"
#include "baci_mat_list.H"
#include "baci_mat_structporo.H"
#include "baci_nurbs_discret_nurbs_utils.H"
#include "baci_structure_new_elements_paramsinterface.H"
#include "baci_w1_poro.H"

#include <Teuchos_SerialDenseSolver.hpp>

#include <iterator>

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::PreEvaluate(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Element::LocationArray& la)
{
  if (scatra_coupling_)
  {
    if (la.Size() > 2)
    {
      if (discretization.HasState(2, "scalar"))
      {
        // check if you can get the scalar state
        Teuchos::RCP<const Epetra_Vector> scalarnp = discretization.GetState(2, "scalar");

        // extract local values of the global vectors
        std::vector<double> myscalar(la[2].lm_.size());
        DRT::UTILS::ExtractMyValues(*scalarnp, myscalar, la[2].lm_);

        if (NumMaterial() < 3) dserror("no third material defined for Wall poro element!");
        Teuchos::RCP<MAT::Material> scatramat = Material(2);

        int numscal = 1;
        if (scatramat->MaterialType() == INPAR::MAT::m_matlist or
            scatramat->MaterialType() == INPAR::MAT::m_matlist_reactions)
        {
          Teuchos::RCP<MAT::MatList> matlist = Teuchos::rcp_dynamic_cast<MAT::MatList>(scatramat);
          numscal = matlist->NumMat();
        }

        Teuchos::RCP<std::vector<double>> scalar =
            Teuchos::rcp(new std::vector<double>(numscal, 0.0));
        if ((int)myscalar.size() != numscal * numnod_) dserror("sizes do not match!");

        for (int i = 0; i < numnod_; i++)
          for (int j = 0; j < numscal; j++) scalar->at(j) += myscalar[numscal * i + j] / numnod_;

        params.set("scalar", scalar);
      }
    }
  }
}

template <CORE::FE::CellType distype>
int DRT::ELEMENTS::Wall1_Poro<distype>::Evaluate(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Element::LocationArray& la,
    CORE::LINALG::SerialDenseMatrix& elemat1_epetra,
    CORE::LINALG::SerialDenseMatrix& elemat2_epetra,
    CORE::LINALG::SerialDenseVector& elevec1_epetra,
    CORE::LINALG::SerialDenseVector& elevec2_epetra,
    CORE::LINALG::SerialDenseVector& elevec3_epetra)
{
  if (not init_) dserror("internal element data not initialized!");

  this->SetParamsInterfacePtr(params);
  ELEMENTS::ActionType act = ELEMENTS::none;

  if (this->IsParamsInterface())
  {
    act = this->ParamsInterface().GetActionType();
  }
  else
  {
    // get the required action
    std::string action = params.get<std::string>("action", "none");
    if (action == "none")
      dserror("No action supplied");
    else if (action == "struct_poro_calc_fluidcoupling")
      act = ELEMENTS::struct_poro_calc_fluidcoupling;
    else if (action == "struct_poro_calc_scatracoupling")
      act = ELEMENTS::struct_poro_calc_scatracoupling;
    else if (action == "struct_poro_calc_prescoupling")
      act = ELEMENTS::struct_poro_calc_prescoupling;
  }

  // what should the element do
  switch (act)
  {
    //==================================================================================
    // off diagonal terms in stiffness matrix for monolithic coupling
    case ELEMENTS::struct_poro_calc_fluidcoupling:
    case ELEMENTS::struct_poro_calc_scatracoupling:
    case ELEMENTS::struct_poro_calc_prescoupling:
    {
      // in some cases we need to write/change some data before evaluating
      PreEvaluate(params, discretization, la);

      MyEvaluate(params, discretization, la, elemat1_epetra, elemat2_epetra, elevec1_epetra,
          elevec2_epetra, elevec3_epetra);
    }
    break;
    //==================================================================================
    default:
    {
      // in some cases we need to write/change some data before evaluating
      PreEvaluate(params, discretization, la);

      // evaluate parent solid element
      DRT::ELEMENTS::Wall1::Evaluate(params, discretization, la[0].lm_, elemat1_epetra,
          elemat2_epetra, elevec1_epetra, elevec2_epetra, elevec3_epetra);

      // add volume coupling specific terms
      MyEvaluate(params, discretization, la, elemat1_epetra, elemat2_epetra, elevec1_epetra,
          elevec2_epetra, elevec3_epetra);
    }
    break;
  }

  return 0;
}

template <CORE::FE::CellType distype>
int DRT::ELEMENTS::Wall1_Poro<distype>::MyEvaluate(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Element::LocationArray& la,
    CORE::LINALG::SerialDenseMatrix& elemat1_epetra,
    CORE::LINALG::SerialDenseMatrix& elemat2_epetra,
    CORE::LINALG::SerialDenseVector& elevec1_epetra,
    CORE::LINALG::SerialDenseVector& elevec2_epetra,
    CORE::LINALG::SerialDenseVector& elevec3_epetra)
{
  this->SetParamsInterfacePtr(params);
  ELEMENTS::ActionType act = ELEMENTS::none;

  if (this->IsParamsInterface())
  {
    act = this->ParamsInterface().GetActionType();
  }
  else
  {
    // get the required action
    std::string action = params.get<std::string>("action", "none");
    if (action == "none")
      dserror("No action supplied");
    else if (action == "calc_struct_internalforce")
      act = ELEMENTS::struct_calc_internalforce;
    else if (action == "calc_struct_nlnstiff")
      act = ELEMENTS::struct_calc_nlnstiff;
    else if (action == "calc_struct_nlnstiffmass")
      act = ELEMENTS::struct_calc_nlnstiffmass;
    else if (action == "struct_poro_calc_fluidcoupling")
      act = ELEMENTS::struct_poro_calc_fluidcoupling;
    else if (action == "calc_struct_stress")
      act = ELEMENTS::struct_calc_stress;
    else if (action == "struct_poro_calc_prescoupling")
      act = ELEMENTS::struct_poro_calc_prescoupling;
  }

  // --------------------------------------------------
  // Now do the nurbs specific stuff
  if (Shape() == CORE::FE::CellType::nurbs4 || Shape() == CORE::FE::CellType::nurbs9)
  {
    myknots_.resize(2);

    switch (act)
    {
      case ELEMENTS::struct_calc_nlnstiffmass:
      case ELEMENTS::struct_calc_nlnstiff:
      case ELEMENTS::struct_calc_internalforce:
      case ELEMENTS::struct_poro_calc_fluidcoupling:
      case ELEMENTS::struct_poro_calc_prescoupling:
      case ELEMENTS::struct_calc_stress:
      {
        auto* nurbsdis = dynamic_cast<DRT::NURBS::NurbsDiscretization*>(&(discretization));

        bool zero_sized = (*((*nurbsdis).GetKnotVector())).GetEleKnots(myknots_, Id());

        // skip zero sized elements in knot span --- they correspond to interpolated nodes
        if (zero_sized)
        {
          return (0);
        }

        break;
      }
      default:
        myknots_.clear();
        break;
    }
  }

  // what should the element do
  switch (act)
  {
    //==================================================================================
    // nonlinear stiffness, damping and internal force vector for poroelasticity
    case ELEMENTS::struct_calc_nlnstiff:
    {
      // stiffness
      CORE::LINALG::Matrix<numdof_, numdof_> elemat1(elemat1_epetra.values(), true);
      // damping
      CORE::LINALG::Matrix<numdof_, numdof_> elemat2(elemat2_epetra.values(), true);
      // internal force vector
      CORE::LINALG::Matrix<numdof_, 1> elevec1(elevec1_epetra.values(), true);

      // elevec2+3 are not used anyway

      std::vector<int> lm = la[0].lm_;

      CORE::LINALG::Matrix<numdim_, numnod_> mydisp(true);
      ExtractValuesFromGlobalVector(discretization, 0, lm, &mydisp, nullptr, "displacement");

      CORE::LINALG::Matrix<numdof_, numdof_>* matptr = nullptr;
      if (elemat1.IsInitialized()) matptr = &elemat1;

      enum INPAR::STR::DampKind damping =
          params.get<enum INPAR::STR::DampKind>("damping", INPAR::STR::damp_none);
      CORE::LINALG::Matrix<numdof_, numdof_>* matptr2 = nullptr;
      if (elemat2.IsInitialized() and (damping == INPAR::STR::damp_material)) matptr2 = &elemat2;

      if (la.Size() > 1)
      {
        if (discretization.HasState(1, "fluidvel"))
        {
          // need current fluid state,
          // call the fluid discretization: fluid equates 2nd dofset
          // disassemble velocities and pressures
          CORE::LINALG::Matrix<numdim_, numnod_> myvel(true);
          CORE::LINALG::Matrix<numdim_, numnod_> myfluidvel(true);
          CORE::LINALG::Matrix<numnod_, 1> myepreaf(true);

          if (discretization.HasState(0, "velocity"))
            ExtractValuesFromGlobalVector(
                discretization, 0, la[0].lm_, &myvel, nullptr, "velocity");

          // extract local values of the global vectors
          ExtractValuesFromGlobalVector(
              discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");

          // calculate tangent stiffness matrix
          NonlinearStiffnessPoroelast(
              lm, mydisp, myvel, myfluidvel, myepreaf, matptr, matptr2, &elevec1, params);
        }
        else if (la.Size() > 2)
        {
          if (discretization.HasState(1, "porofluid"))
          {
            // get primary variables of multiphase porous medium flow
            std::vector<double> myephi(la[1].Size());
            Teuchos::RCP<const Epetra_Vector> matrix_state =
                discretization.GetState(1, "porofluid");
            DRT::UTILS::ExtractMyValues(*matrix_state, myephi, la[1].lm_);

            // calculate tangent stiffness matrix
            NonlinearStiffnessPoroelastPressureBased(lm, mydisp, myephi, matptr, &elevec1, params);
          }
        }
      }
    }
    break;

    //==================================================================================
    // nonlinear stiffness, mass matrix and internal force vector for poroelasticity
    case ELEMENTS::struct_calc_nlnstiffmass:
    {
      // stiffness
      CORE::LINALG::Matrix<numdof_, numdof_> elemat1(elemat1_epetra.values(), true);
      // internal force vector
      CORE::LINALG::Matrix<numdof_, 1> elevec1(elevec1_epetra.values(), true);

      // elemat2,elevec2+3 are not used anyway

      // build the location vector only for the structure field
      std::vector<int> lm = la[0].lm_;

      CORE::LINALG::Matrix<numdim_, numnod_> mydisp(true);
      ExtractValuesFromGlobalVector(discretization, 0, la[0].lm_, &mydisp, nullptr, "displacement");

      CORE::LINALG::Matrix<numdof_, numdof_>* matptr = nullptr;
      if (elemat1.IsInitialized()) matptr = &elemat1;

      // we skip this evaluation if the coupling is not setup yet, i.e.
      // if the secondary dofset or the secondary material was not set
      // this can happen during setup of the time integrator or restart
      // there might be a better way. For instance do not evaluate
      // before the setup of the multiphysics problem is completed.
      if (la.Size() > 1 and NumMaterial() > 1)
      {
        // need current fluid state,
        // call the fluid discretization: fluid equates 2nd dofset
        // disassemble velocities and pressures

        CORE::LINALG::Matrix<numdim_, numnod_> myvel(true);
        CORE::LINALG::Matrix<numdim_, numnod_> myfluidvel(true);
        CORE::LINALG::Matrix<numnod_, 1> myepreaf(true);

        if (discretization.HasState(0, "velocity"))
          ExtractValuesFromGlobalVector(discretization, 0, la[0].lm_, &myvel, nullptr, "velocity");

        // this is kind of a hack. Find a better way! (e.g. move the pressure based variant
        // into own element)
        if (discretization.HasState(1, "fluidvel"))
        {
          // extract local values of the global vectors
          ExtractValuesFromGlobalVector(
              discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");

          NonlinearStiffnessPoroelast(
              lm, mydisp, myvel, myfluidvel, myepreaf, matptr, nullptr, &elevec1, params);
        }
        else if (la.Size() > 2)
        {
          if (discretization.HasState(1, "porofluid"))
          {
            // get primary variables of multiphase porous medium flow
            std::vector<double> myephi(la[1].Size());
            Teuchos::RCP<const Epetra_Vector> matrix_state =
                discretization.GetState(1, "porofluid");
            DRT::UTILS::ExtractMyValues(*matrix_state, myephi, la[1].lm_);

            // calculate tangent stiffness matrix
            NonlinearStiffnessPoroelastPressureBased(lm, mydisp, myephi, matptr, &elevec1, params);
          }
        }
      }
    }
    break;

    //==================================================================================
    // coupling terms in force-vector and stiffness matrix for poroelasticity
    case ELEMENTS::struct_poro_calc_fluidcoupling:
    {
      // stiffness
      CORE::LINALG::Matrix<numdof_, (numdim_ + 1) * numnod_> elemat1(elemat1_epetra.values(), true);

      // elemat2,elevec1-3 are not used anyway

      // build the location vector only for the structure field
      std::vector<int> lm = la[0].lm_;

      CORE::LINALG::Matrix<numdof_, (numdim_ + 1)* numnod_>* matptr = nullptr;
      if (elemat1.IsInitialized()) matptr = &elemat1;

      // need current fluid state,
      // call the fluid discretization: fluid equates 2nd dofset
      // disassemble velocities and pressures
      if (discretization.HasState(1, "fluidvel"))
      {
        CORE::LINALG::Matrix<numdim_, numnod_> myvel(true);
        CORE::LINALG::Matrix<numdim_, numnod_> myfluidvel(true);
        CORE::LINALG::Matrix<numnod_, 1> myepreaf(true);

        CORE::LINALG::Matrix<numdim_, numnod_> mydisp(true);
        ExtractValuesFromGlobalVector(
            discretization, 0, la[0].lm_, &mydisp, nullptr, "displacement");

        if (discretization.HasState(0, "velocity"))
          ExtractValuesFromGlobalVector(discretization, 0, la[0].lm_, &myvel, nullptr, "velocity");

        // extract local values of the global vectors
        ExtractValuesFromGlobalVector(
            discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");

        CouplingPoroelast(
            lm, mydisp, myvel, myfluidvel, myepreaf, matptr, nullptr, nullptr, params);
      }
      else if (la.Size() > 2)
      {
        if (discretization.HasState(1, "porofluid"))
        {
          // get primary variables of multiphase porous medium flow
          std::vector<double> myephi(la[1].Size());
          Teuchos::RCP<const Epetra_Vector> matrix_state = discretization.GetState(1, "porofluid");
          DRT::UTILS::ExtractMyValues(*matrix_state, myephi, la[1].lm_);

          CORE::LINALG::Matrix<numdim_, numnod_> mydisp(true);
          ExtractValuesFromGlobalVector(
              discretization, 0, la[0].lm_, &mydisp, nullptr, "displacement");

          // calculate OD-Matrix
          CouplingPoroelastPressureBased(lm, mydisp, myephi, elemat1_epetra, params);
        }
        else
          dserror("cannot find global states displacement or solidpressure");
      }
    }
    break;

    //==================================================================================
    // nonlinear stiffness and internal force vector for poroelasticity
    case ELEMENTS::struct_calc_internalforce:
    {
      // internal force vector
      CORE::LINALG::Matrix<numdof_, 1> elevec1(elevec1_epetra.values(), true);

      // elemat1+2,elevec2+3 are not used anyway

      // build the location vector only for the structure field
      std::vector<int> lm = la[0].lm_;

      CORE::LINALG::Matrix<numdim_, numnod_> mydisp(true);
      ExtractValuesFromGlobalVector(discretization, 0, lm, &mydisp, nullptr, "displacement");

      // need current fluid state,
      // call the fluid discretization: fluid equates 2nd dofset
      // disassemble velocities and pressures
      if (discretization.HasState(1, "fluidvel"))
      {
        // extract local values of the global vectors
        CORE::LINALG::Matrix<numdim_, numnod_> myfluidvel(true);
        CORE::LINALG::Matrix<numnod_, 1> myepreaf(true);
        ExtractValuesFromGlobalVector(
            discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");

        CORE::LINALG::Matrix<numdim_, numnod_> myvel(true);
        ExtractValuesFromGlobalVector(discretization, 0, la[0].lm_, &myvel, nullptr, "velocity");

        // calculate tangent stiffness matrix
        NonlinearStiffnessPoroelast(
            lm, mydisp, myvel, myfluidvel, myepreaf, nullptr, nullptr, &elevec1, params);
      }
      else if (la.Size() > 2)
      {
        if (discretization.HasState(1, "porofluid"))
        {
          // get primary variables of multiphase porous medium flow
          std::vector<double> myephi(la[1].Size());
          Teuchos::RCP<const Epetra_Vector> matrix_state = discretization.GetState(1, "porofluid");
          DRT::UTILS::ExtractMyValues(*matrix_state, myephi, la[1].lm_);

          // calculate tangent stiffness matrix
          NonlinearStiffnessPoroelastPressureBased(lm, mydisp, myephi, nullptr, &elevec1, params);
        }
      }
    }
    break;
    //==================================================================================
    // evaluate stresses and strains at gauss points
    case ELEMENTS::struct_calc_stress:
    {
      // elemat1+2,elevec1-3 are not used anyway

      // nothing to do for ghost elements
      if (discretization.Comm().MyPID() == Owner())
      {
        // get the location vector only for the structure field
        std::vector<int> lm = la[0].lm_;

        CORE::LINALG::Matrix<numdim_, numnod_> mydisp(true);
        ExtractValuesFromGlobalVector(discretization, 0, lm, &mydisp, nullptr, "displacement");

        Teuchos::RCP<std::vector<char>> couplstressdata =
            params.get<Teuchos::RCP<std::vector<char>>>("couplstress", Teuchos::null);

        if (couplstressdata == Teuchos::null) dserror("Cannot get 'couplstress' data");

        CORE::LINALG::SerialDenseMatrix couplstress(numgpt_, Wall1::numstr_);

        auto iocouplstress = DRT::INPUT::get<INPAR::STR::StressType>(
            params, "iocouplstress", INPAR::STR::stress_none);

        // need current fluid state,
        // call the fluid discretization: fluid equates 2nd dofset
        // disassemble velocities and pressures
        if (discretization.HasState(1, "fluidvel"))
        {
          // extract local values of the global vectors
          CORE::LINALG::Matrix<numdim_, numnod_> myfluidvel(true);
          CORE::LINALG::Matrix<numnod_, 1> myepreaf(true);
          ExtractValuesFromGlobalVector(
              discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");

          CouplingStressPoroelast(
              mydisp, myfluidvel, myepreaf, &couplstress, nullptr, params, iocouplstress);
        }
        else if (la.Size() > 2)
        {
          if (discretization.HasState(1, "porofluid"))
            dserror("coupl stress poroelast not yet implemented for pressure-based variant");
        }

        // pack the data for postprocessing
        {
          CORE::COMM::PackBuffer data;
          // get the size of stress
          Wall1::AddtoPack(data, couplstress);
          data.StartPacking();
          // pack the stresses
          Wall1::AddtoPack(data, couplstress);
          std::copy(data().begin(), data().end(), std::back_inserter(*couplstressdata));
        }
      }
    }
    break;

    //==================================================================================
    default:
      // do nothing
      break;
  }
  return 0;
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::NonlinearStiffnessPoroelast(
    std::vector<int>& lm,                                 // location matrix
    CORE::LINALG::Matrix<numdim_, numnod_>& disp,         // current displacements
    CORE::LINALG::Matrix<numdim_, numnod_>& vel,          // current velocities
    CORE::LINALG::Matrix<numdim_, numnod_>& evelnp,       // current fluid velocities
    CORE::LINALG::Matrix<numnod_, 1>& epreaf,             // current fluid pressure
    CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix,  // element stiffness matrix
    CORE::LINALG::Matrix<numdof_, numdof_>* reamatrix,    // element reactive matrix
    CORE::LINALG::Matrix<numdof_, 1>* force,              // element internal force vector
    Teuchos::ParameterList& params                        // algorithmic parameters e.g. time
)
{
  GetMaterials();

  // update element geometry
  CORE::LINALG::Matrix<numdim_, numnod_> xrefe;  // material coord. of element
  CORE::LINALG::Matrix<numdim_, numnod_> xcurr;  // current  coord. of element

  DRT::Node** nodes = Nodes();
  for (int i = 0; i < numnod_; ++i)
  {
    const auto& x = nodes[i]->X();
    for (int j = 0; j < numdim_; j++)
    {
      xrefe(j, i) = x[j];
      xcurr(j, i) = xrefe(j, i) + disp(j, i);
    }
  }

  CORE::LINALG::Matrix<numdof_, numdof_> erea_v(true);

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  GaussPointLoop(params, xrefe, xcurr, disp, vel, evelnp, epreaf, nullptr, erea_v, stiffmatrix,
      reamatrix, force);

  if (reamatrix != nullptr)
  {
    /* additional "reactive darcy-term"
     detJ * w(gp) * ( J * reacoeff * phi^2  ) * D(v_s)
     */
    reamatrix->Update(1.0, erea_v, 1.0);
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::NonlinearStiffnessPoroelastPressureBased(
    std::vector<int>& lm,                          // location matrix
    CORE::LINALG::Matrix<numdim_, numnod_>& disp,  // current displacements
    const std::vector<double>& ephi,               // primary variable for poro-multiphase flow
    CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix,  // element stiffness matrix
    CORE::LINALG::Matrix<numdof_, 1>* force,              // element internal force vector
    Teuchos::ParameterList& params                        // algorithmic parameters e.g. time
)
{
  GetMaterialsPressureBased();

  // update element geometry
  CORE::LINALG::Matrix<numdim_, numnod_> xrefe;  // material coord. of element
  CORE::LINALG::Matrix<numdim_, numnod_> xcurr;  // current  coord. of element

  DRT::Node** nodes = Nodes();
  for (int i = 0; i < numnod_; ++i)
  {
    const auto& x = nodes[i]->X();
    for (int j = 0; j < numdim_; j++)
    {
      xrefe(j, i) = x[j];
      xcurr(j, i) = xrefe(j, i) + disp(j, i);
    }
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  GaussPointLoopPressureBased(params, xrefe, xcurr, disp, ephi, stiffmatrix, force);
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::GaussPointLoop(Teuchos::ParameterList& params,
    const CORE::LINALG::Matrix<numdim_, numnod_>& xrefe,
    const CORE::LINALG::Matrix<numdim_, numnod_>& xcurr,
    const CORE::LINALG::Matrix<numdim_, numnod_>& nodaldisp,
    const CORE::LINALG::Matrix<numdim_, numnod_>& nodalvel,
    const CORE::LINALG::Matrix<numdim_, numnod_>& evelnp,
    const CORE::LINALG::Matrix<numnod_, 1>& epreaf,
    const CORE::LINALG::Matrix<numnod_, 1>* porosity_dof,
    CORE::LINALG::Matrix<numdof_, numdof_>& erea_v,
    CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix,
    CORE::LINALG::Matrix<numdof_, numdof_>* reamatrix, CORE::LINALG::Matrix<numdof_, 1>* force)
{
  /*--------------------------------- get node weights for nurbs elements */
  if (distype == CORE::FE::CellType::nurbs4 || distype == CORE::FE::CellType::nurbs9)
  {
    for (int inode = 0; inode < numnod_; ++inode)
    {
      auto* cp = dynamic_cast<DRT::NURBS::ControlPoint*>(Nodes()[inode]);

      weights_(inode) = cp->W();
    }
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  // first derivatives N_XYZ at gp w.r.t. material coordinates
  CORE::LINALG::Matrix<numdim_, numnod_> N_XYZ;
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  // CAUTION: defgrd(true): filled with zeros!
  CORE::LINALG::Matrix<numdim_, numdim_> defgrd(true);
  // shape function at gp w.r.t. reference coordinates
  CORE::LINALG::Matrix<numnod_, 1> shapefct;
  // first derivatives at gp w.r.t. reference coordinates
  CORE::LINALG::Matrix<numdim_, numnod_> deriv;

  CORE::LINALG::Matrix<numstr_, 1> fstress(true);

  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // evaluate shape functions and derivatives at integration point
    ComputeShapeFunctionsAndDerivatives(gp, shapefct, deriv, N_XYZ);

    // compute deformation gradient
    ComputeDefGradient(defgrd, N_XYZ, xcurr);

    // inverse deformation gradient F^-1
    CORE::LINALG::Matrix<numdim_, numdim_> defgrd_inv(false);
    defgrd_inv.Invert(defgrd);

    // jacobian determinant of transformation between spatial and material space "|dx/dX|"
    double J = 0.0;
    //------linearization of jacobi determinant detF=J w.r.t. structure displacement   dJ/d(us) =
    // dJ/dF : dF/dus = J * F^-T * N,X
    static CORE::LINALG::Matrix<1, numdof_> dJ_dus;
    // volume change (used for porosity law). Same as J in nonlinear theory.
    double volchange = 0.0;
    //------linearization of volume change w.r.t. structure displacement
    static CORE::LINALG::Matrix<1, numdof_> dvolchange_dus;

    // compute J, the volume change and the respctive linearizations w.r.t. structure displacement
    ComputeJacobianDeterminantVolumeChangeAndLinearizations(
        J, volchange, dJ_dus, dvolchange_dus, defgrd, defgrd_inv, N_XYZ, nodaldisp);

    // non-linear B-operator
    CORE::LINALG::Matrix<numstr_, numdof_> bop;
    ComputeBOperator(bop, defgrd, N_XYZ);

    //----------------------------------------------------
    // pressure at integration point
    double press = shapefct.Dot(epreaf);

    // pressure gradient at integration point
    CORE::LINALG::Matrix<numdim_, 1> Gradp;
    Gradp.Multiply(N_XYZ, epreaf);

    // fluid velocity at integration point
    CORE::LINALG::Matrix<numdim_, 1> fvelint;
    fvelint.Multiply(evelnp, shapefct);

    // material fluid velocity gradient at integration point
    CORE::LINALG::Matrix<numdim_, numdim_> fvelder;
    fvelder.MultiplyNT(evelnp, N_XYZ);

    // structure displacement and velocity at integration point
    CORE::LINALG::Matrix<numdim_, 1> velint(true);

    for (int i = 0; i < numnod_; i++)
      for (int j = 0; j < numdim_; j++) velint(j) += nodalvel(j, i) * shapefct(i);

    // Right Cauchy-Green tensor = F^T * F
    CORE::LINALG::Matrix<numdim_, numdim_> cauchygreen;
    cauchygreen.MultiplyTN(defgrd, defgrd);

    // inverse Right Cauchy-Green tensor
    CORE::LINALG::Matrix<numdim_, numdim_> C_inv(false);
    C_inv.Invert(cauchygreen);

    // compute some auxiliary matrixes for computation of linearization
    // dF^-T/dus
    CORE::LINALG::Matrix<numdim_ * numdim_, numdof_> dFinvTdus(true);
    // F^-T * Grad p
    CORE::LINALG::Matrix<numdim_, 1> Finvgradp;
    // dF^-T/dus * Grad p
    CORE::LINALG::Matrix<numdim_, numdof_> dFinvdus_gradp(true);
    // dC^-1/dus * Grad p
    CORE::LINALG::Matrix<numstr_, numdof_> dCinv_dus(true);

    ComputeAuxiliaryValues(
        N_XYZ, defgrd_inv, C_inv, Gradp, dFinvTdus, Finvgradp, dFinvdus_gradp, dCinv_dus);

    //--------------------------------------------------------------------

    // linearization of porosity w.r.t structure displacement d\phi/d(us) = d\phi/dJ*dJ/d(us)
    CORE::LINALG::Matrix<1, numdof_> dphi_dus;
    double porosity = 0.0;

    ComputePorosityAndLinearization(
        params, press, volchange, gp, shapefct, porosity_dof, dvolchange_dus, porosity, dphi_dus);

    // **********************evaluate stiffness matrix and force vector**********************
    if (fluid_mat_->Type() == MAT::PAR::darcy_brinkman)
    {
      FillMatrixAndVectorsBrinkman(gp, J, porosity, fvelder, defgrd_inv, bop, C_inv, dphi_dus,
          dJ_dus, dCinv_dus, dFinvTdus, stiffmatrix, force, fstress);
    }

    FillMatrixAndVectors(gp, shapefct, N_XYZ, J, press, porosity, velint, fvelint, fvelder,
        defgrd_inv, bop, C_inv, Finvgradp, dphi_dus, dJ_dus, dCinv_dus, dFinvdus_gradp, dFinvTdus,
        erea_v, stiffmatrix, force, fstress);
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::GaussPointLoopPressureBased(Teuchos::ParameterList& params,
    const CORE::LINALG::Matrix<numdim_, numnod_>& xrefe,
    const CORE::LINALG::Matrix<numdim_, numnod_>& xcurr,
    const CORE::LINALG::Matrix<numdim_, numnod_>& nodaldisp, const std::vector<double>& ephi,
    CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix, CORE::LINALG::Matrix<numdof_, 1>* force)
{
  /*--------------------------------- get node weights for nurbs elements */
  if (distype == CORE::FE::CellType::nurbs4 || distype == CORE::FE::CellType::nurbs9)
  {
    for (int inode = 0; inode < numnod_; ++inode)
    {
      auto* cp = dynamic_cast<DRT::NURBS::ControlPoint*>(Nodes()[inode]);

      weights_(inode) = cp->W();
    }
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  // first derivatives N_XYZ at gp w.r.t. material coordinates
  CORE::LINALG::Matrix<numdim_, numnod_> N_XYZ;
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  // CAUTION: defgrd(true): filled with zeros!
  CORE::LINALG::Matrix<numdim_, numdim_> defgrd(true);
  // shape function at gp w.r.t. reference coordinates
  CORE::LINALG::Matrix<numnod_, 1> shapefct;
  // first derivatives at gp w.r.t. reference coordinates
  CORE::LINALG::Matrix<numdim_, numnod_> deriv;

  CORE::LINALG::Matrix<numstr_, 1> fstress(true);

  // Initialize
  const int totalnumdofpernode = fluidmulti_mat_->NumMat();
  const int numfluidphases = fluidmulti_mat_->NumFluidPhases();
  const int numvolfrac = fluidmulti_mat_->NumVolFrac();
  const bool hasvolfracs = (totalnumdofpernode > numfluidphases);
  std::vector<double> phiAtGP(totalnumdofpernode);

  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // evaluate shape functions and derivatives at integration point
    ComputeShapeFunctionsAndDerivatives(gp, shapefct, deriv, N_XYZ);

    // compute deformation gradient
    ComputeDefGradient(defgrd, N_XYZ, xcurr);

    // inverse deformation gradient F^-1
    CORE::LINALG::Matrix<numdim_, numdim_> defgrd_inv(false);
    defgrd_inv.Invert(defgrd);

    // jacobian determinant of transformation between spatial and material space "|dx/dX|"
    double J = 0.0;
    //------linearization of jacobi determinant detF=J w.r.t. structure displacement   dJ/d(us) =
    // dJ/dF : dF/dus = J * F^-T * N,X
    static CORE::LINALG::Matrix<1, numdof_> dJ_dus;
    // volume change (used for porosity law). Same as J in nonlinear theory.
    double volchange = 0.0;
    //------linearization of volume change w.r.t. structure displacement
    static CORE::LINALG::Matrix<1, numdof_> dvolchange_dus;

    // compute J, the volume change and the respctive linearizations w.r.t. structure displacement
    ComputeJacobianDeterminantVolumeChangeAndLinearizations(
        J, volchange, dJ_dus, dvolchange_dus, defgrd, defgrd_inv, N_XYZ, nodaldisp);

    // non-linear B-operator
    CORE::LINALG::Matrix<numstr_, numdof_> bop;
    ComputeBOperator(bop, defgrd, N_XYZ);

    // derivative of press w.r.t. displacements (only in case of vol fracs)
    CORE::LINALG::Matrix<1, numdof_> dps_dus(true);

    //----------------------------------------------------
    // pressure at integration point
    ComputePrimaryVariableAtGP(ephi, totalnumdofpernode, shapefct, phiAtGP);
    double press = ComputeSolPressureAtGP(totalnumdofpernode, numfluidphases, phiAtGP);
    // recalculate for the case of volume fractions
    if (hasvolfracs)
    {
      CORE::LINALG::Matrix<1, numdof_> dphi_dus;
      double porosity = 0.0;

      ComputePorosityAndLinearization(
          params, press, volchange, gp, shapefct, nullptr, dvolchange_dus, porosity, dphi_dus);
      // save the pressure coming from the fluid S_i*p_i
      const double fluidpress = press;
      press = RecalculateSolPressureAtGP(
          fluidpress, porosity, totalnumdofpernode, numfluidphases, numvolfrac, phiAtGP);
      ComputeLinearizationOfSolPressWrtDisp(fluidpress, porosity, totalnumdofpernode,
          numfluidphases, numvolfrac, phiAtGP, dphi_dus, dps_dus);
    }

    // Right Cauchy-Green tensor = F^T * F
    CORE::LINALG::Matrix<numdim_, numdim_> cauchygreen;
    cauchygreen.MultiplyTN(defgrd, defgrd);

    // inverse Right Cauchy-Green tensor
    CORE::LINALG::Matrix<numdim_, numdim_> C_inv(false);
    C_inv.Invert(cauchygreen);

    // compute some auxiliary matrixes for computation of linearization
    // dC^-1/dus
    CORE::LINALG::Matrix<numstr_, numdof_> dCinv_dus(true);
    for (int n = 0; n < numnod_; ++n)
    {
      for (int k = 0; k < numdim_; ++k)
      {
        const int gid = n * numdim_ + k;
        for (int i = 0; i < numdim_; ++i)
        {
          dCinv_dus(0, gid) += -2 * C_inv(0, i) * N_XYZ(i, n) * defgrd_inv(0, k);
          dCinv_dus(1, gid) += -2 * C_inv(1, i) * N_XYZ(i, n) * defgrd_inv(1, k);
          /* ~~~ */
          dCinv_dus(2, gid) += -C_inv(0, i) * N_XYZ(i, n) * defgrd_inv(1, k) -
                               defgrd_inv(0, k) * N_XYZ(i, n) * C_inv(1, i);
        }
      }
    }

    // **********************evaluate stiffness matrix and force vector**********************
    FillMatrixAndVectorsPressureBased(
        gp, shapefct, N_XYZ, J, press, bop, C_inv, dJ_dus, dCinv_dus, dps_dus, stiffmatrix, force);
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::FillMatrixAndVectors(const int& gp,
    const CORE::LINALG::Matrix<numnod_, 1>& shapefct,
    const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& press,
    const double& porosity, const CORE::LINALG::Matrix<numdim_, 1>& velint,
    const CORE::LINALG::Matrix<numdim_, 1>& fvelint,
    const CORE::LINALG::Matrix<numdim_, numdim_>& fvelder,
    const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd_inv,
    const CORE::LINALG::Matrix<numstr_, numdof_>& bop,
    const CORE::LINALG::Matrix<numdim_, numdim_>& C_inv,
    const CORE::LINALG::Matrix<numdim_, 1>& Finvgradp,
    const CORE::LINALG::Matrix<1, numdof_>& dphi_dus,
    const CORE::LINALG::Matrix<1, numdof_>& dJ_dus,
    const CORE::LINALG::Matrix<numstr_, numdof_>& dCinv_dus,
    const CORE::LINALG::Matrix<numdim_, numdof_>& dFinvdus_gradp,
    const CORE::LINALG::Matrix<numdim_ * numdim_, numdof_>& dFinvTdus,
    CORE::LINALG::Matrix<numdof_, numdof_>& erea_v,
    CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix, CORE::LINALG::Matrix<numdof_, 1>* force,
    CORE::LINALG::Matrix<numstr_, 1>& fstress)
{
  // const double reacoeff = fluid_mat_->ComputeReactionCoeff();

  static CORE::LINALG::Matrix<numdim_, numdim_> matreatensor(true);
  static CORE::LINALG::Matrix<numdim_, numdim_> reatensor(true);
  static CORE::LINALG::Matrix<numdim_, numdim_> linreac_dphi(true);
  static CORE::LINALG::Matrix<numdim_, numdim_> linreac_dJ(true);
  static CORE::LINALG::Matrix<numdim_, 1> reafvel(true);
  static CORE::LINALG::Matrix<numdim_, 1> reavel(true);
  {
    static CORE::LINALG::Matrix<numdim_, numdim_> temp(false);
    std::vector<double> anisotropic_permeability_coeffs =
        ComputeAnisotropicPermeabilityCoeffsAtGP(shapefct);
    fluid_mat_->ComputeReactionTensor(matreatensor, J, porosity,
        anisotropic_permeability_directions_, anisotropic_permeability_coeffs);
    fluid_mat_->ComputeLinMatReactionTensor(linreac_dphi, linreac_dJ, J, porosity);
    temp.Multiply(1.0, matreatensor, defgrd_inv);
    reatensor.MultiplyTN(defgrd_inv, temp);
    reavel.Multiply(reatensor, velint);
    reafvel.Multiply(reatensor, fvelint);
  }

  const double detJ_w = detJ_[gp] * intpoints_.Weight(gp) * thickness_;

  {
    for (int k = 0; k < numnod_; k++)
    {
      const int fk = numdim_ * k;
      const double fac = detJ_w * shapefct(k);
      const double v = fac * porosity * porosity * J * J;

      for (int j = 0; j < numdim_; j++)
      {
        /*-------structure- velocity coupling:  RHS
         "darcy-terms"
         - reacoeff * J^2 *  phi^2 *  v^f
         */
        (*force)(fk + j) += -v * reafvel(j);

        /* "reactive darcy-terms"
         reacoeff * J^2 *  phi^2 *  v^s
         */
        (*force)(fk + j) += v * reavel(j);

        /*-------structure- fluid pressure coupling: RHS
         *                        "pressure gradient terms"
         - J *  F^-T * Grad(p) * phi
         */
        (*force)(fk + j) += fac * J * Finvgradp(j) * (-porosity);

        for (int i = 0; i < numnod_; i++)
        {
          const int fi = numdim_ * i;

          for (int l = 0; l < numdim_; l++)
          {
            /* additional "reactive darcy-term"
             detJ * w(gp) * ( J^2 * reacoeff * phi^2  ) * D(v_s)
             */
            erea_v(fk + j, fi + l) += v * reatensor(j, l) * shapefct(i);

            /* additional "pressure gradient term"
             -  detJ * w(gp) * phi *  ( dJ/d(us) * F^-T * Grad(p) - J * d(F^-T)/d(us) *Grad(p) ) *
             D(us)
             - detJ * w(gp) * d(phi)/d(us) * J * F^-T * Grad(p) * D(us)
             */
            (*stiffmatrix)(fk + j, fi + l) += fac * (-porosity * dJ_dus(fi + l) * Finvgradp(j) -
                                                        porosity * J * dFinvdus_gradp(j, fi + l) -
                                                        dphi_dus(fi + l) * J * Finvgradp(j));

            /* additional "reactive darcy-term"
               detJ * w(gp) * 2 * ( dJ/d(us) * vs * reacoeff * phi^2 + J * reacoeff * phi *
             d(phi)/d(us) * vs ) * D(us)
             - detJ * w(gp) *  2 * ( J * dJ/d(us) * v^f * reacoeff * phi^2 + J * reacoeff * phi *
             d(phi)/d(us) * v^f ) * D(us)
             */
            (*stiffmatrix)(fk + j, fi + l) += fac * J * porosity * 2.0 * (reavel(j) - reafvel(j)) *
                                              (porosity * dJ_dus(fi + l) + J * dphi_dus(fi + l));

            for (int m = 0; m < numdim_; ++m)
            {
              for (int n = 0; n < numdim_; ++n)
              {
                for (int p = 0; p < numdim_; ++p)
                {
                  (*stiffmatrix)(fk + j, fi + l) +=
                      v * (velint(p) - fvelint(p)) *
                      (dFinvTdus(j * numdim_ + m, fi + l) * matreatensor(m, n) * defgrd_inv(n, p) +
                          defgrd_inv(m, j) * matreatensor(m, n) *
                              dFinvTdus(p * numdim_ + n, fi + l));
                }
              }
            }
            // check if derivatives of reaction tensor are zero --> significant speed up
            if (fluid_mat_->PermeabilityFunction() != MAT::PAR::constant)
            {
              for (int m = 0; m < numdim_; ++m)
              {
                for (int n = 0; n < numdim_; ++n)
                {
                  for (int p = 0; p < numdim_; ++p)
                  {
                    (*stiffmatrix)(fk + j, fi + l) += v * (velint(p) - fvelint(p)) *
                                                      (+defgrd_inv(m, j) *
                                                          (linreac_dphi(m, n) * dphi_dus(fi + l) +
                                                              linreac_dJ(m, n) * dJ_dus(fi + l)) *
                                                          defgrd_inv(n, p));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // inverse Right Cauchy-Green tensor as vector
  static CORE::LINALG::Matrix<numstr_, 1> C_inv_vec;
  C_inv_vec(0) = C_inv(0, 0);
  C_inv_vec(1) = C_inv(1, 1);
  C_inv_vec(2) = C_inv(0, 1);

  // B^T . C^-1
  static CORE::LINALG::Matrix<numdof_, 1> cinvb(true);
  cinvb.MultiplyTN(bop, C_inv_vec);

  const double fac1 = -detJ_w * press;
  const double fac2 = fac1 * J;

  // update internal force vector
  if (force != nullptr)
  {
    // additional fluid stress- stiffness term RHS -(B^T .  C^-1  * J * p^f * detJ * w(gp))
    force->Update(fac2, cinvb, 1.0);
  }

  // update stiffness matrix
  if (stiffmatrix != nullptr)
  {
    static CORE::LINALG::Matrix<numdof_, numdof_> tmp;

    // additional fluid stress- stiffness term -(B^T . C^-1 . dJ/d(us) * p^f * detJ * w(gp))
    tmp.Multiply(fac1, cinvb, dJ_dus);
    stiffmatrix->Update(1.0, tmp, 1.0);

    // additional fluid stress- stiffness term -(B^T .  dC^-1/d(us) * J * p^f * detJ * w(gp))
    tmp.MultiplyTN(fac2, bop, dCinv_dus);
    stiffmatrix->Update(1.0, tmp, 1.0);

    // integrate `geometric' stiffness matrix and add to keu *****************
    CORE::LINALG::Matrix<numstr_, 1> sfac(C_inv_vec);  // auxiliary integrated stress

    // scale and add viscous stress
    sfac.Update(detJ_w, fstress, fac2);  // detJ*w(gp)*[S11,S22,S33,S12=S21,S23=S32,S13=S31]

    std::vector<double> SmB_L(2);  // intermediate Sm.B_L
    // kgeo += (B_L^T . sigma . B_L) * detJ * w(gp)  with B_L = Ni,Xj see NiliFEM-Skript
    for (int inod = 0; inod < numnod_; ++inod)
    {
      SmB_L[0] = sfac(0) * N_XYZ(0, inod) + sfac(2) * N_XYZ(1, inod);
      SmB_L[1] = sfac(2) * N_XYZ(0, inod) + sfac(1) * N_XYZ(1, inod);
      for (int jnod = 0; jnod < numnod_; ++jnod)
      {
        double bopstrbop = 0.0;  // intermediate value
        for (int idim = 0; idim < numdim_; ++idim) bopstrbop += N_XYZ(idim, jnod) * SmB_L[idim];
        (*stiffmatrix)(numdim_ * inod + 0, numdim_ * jnod + 0) += bopstrbop;
        (*stiffmatrix)(numdim_ * inod + 1, numdim_ * jnod + 1) += bopstrbop;
      }
    }
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::FillMatrixAndVectorsPressureBased(const int& gp,
    const CORE::LINALG::Matrix<numnod_, 1>& shapefct,
    const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& press,
    const CORE::LINALG::Matrix<numstr_, numdof_>& bop,
    const CORE::LINALG::Matrix<numdim_, numdim_>& C_inv,
    const CORE::LINALG::Matrix<1, numdof_>& dJ_dus,
    const CORE::LINALG::Matrix<numstr_, numdof_>& dCinv_dus,
    const CORE::LINALG::Matrix<1, numdof_>& dps_dus,
    CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix, CORE::LINALG::Matrix<numdof_, 1>* force)
{
  const double detJ_w = detJ_[gp] * intpoints_.Weight(gp) * thickness_;

  // inverse Right Cauchy-Green tensor as vector
  static CORE::LINALG::Matrix<numstr_, 1> C_inv_vec;
  C_inv_vec(0) = C_inv(0, 0);
  C_inv_vec(1) = C_inv(1, 1);
  C_inv_vec(2) = C_inv(0, 1);

  // B^T . C^-1
  static CORE::LINALG::Matrix<numdof_, 1> cinvb(true);
  cinvb.MultiplyTN(bop, C_inv_vec);

  const double fac1 = -detJ_w * press;
  const double fac2 = fac1 * J;

  // update internal force vector
  if (force != nullptr)
  {
    // additional fluid stress- stiffness term RHS -(B^T .  C^-1  * J * p^f * detJ * w(gp))
    force->Update(fac2, cinvb, 1.0);
  }

  // update stiffness matrix
  if (stiffmatrix != nullptr)
  {
    static CORE::LINALG::Matrix<numdof_, numdof_> tmp;

    // additional fluid stress- stiffness term -(B^T . C^-1 . dJ/d(us) * p^f * detJ * w(gp))
    tmp.Multiply(fac1, cinvb, dJ_dus);
    stiffmatrix->Update(1.0, tmp, 1.0);

    // additional fluid stress- stiffness term -(B^T .  dC^-1/d(us) * J * p^f * detJ * w(gp))
    tmp.MultiplyTN(fac2, bop, dCinv_dus);
    stiffmatrix->Update(1.0, tmp, 1.0);

    // additional fluid stress- stiffness term -(B^T .  dC^-1 * J * dp^s/d(us) * detJ * w(gp))
    tmp.Multiply(-detJ_w * J, cinvb, dps_dus);
    stiffmatrix->Update(1.0, tmp, 1.0);

    // integrate `geometric' stiffness matrix and add to keu *****************
    CORE::LINALG::Matrix<numstr_, 1> sfac(C_inv_vec);  // auxiliary integrated stress
    sfac.Scale(fac2);

    std::vector<double> SmB_L(2);  // intermediate Sm.B_L
    // kgeo += (B_L^T . sigma . B_L) * detJ * w(gp)  with B_L = Ni,Xj see NiliFEM-Skript
    for (int inod = 0; inod < numnod_; ++inod)
    {
      SmB_L[0] = sfac(0) * N_XYZ(0, inod) + sfac(2) * N_XYZ(1, inod);
      SmB_L[1] = sfac(2) * N_XYZ(0, inod) + sfac(1) * N_XYZ(1, inod);
      for (int jnod = 0; jnod < numnod_; ++jnod)
      {
        double bopstrbop = 0.0;  // intermediate value
        for (int idim = 0; idim < numdim_; ++idim) bopstrbop += N_XYZ(idim, jnod) * SmB_L[idim];
        (*stiffmatrix)(numdim_ * inod + 0, numdim_ * jnod + 0) += bopstrbop;
        (*stiffmatrix)(numdim_ * inod + 1, numdim_ * jnod + 1) += bopstrbop;
      }
    }
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::FillMatrixAndVectorsBrinkman(const int& gp,
    const double& J, const double& porosity, const CORE::LINALG::Matrix<numdim_, numdim_>& fvelder,
    const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd_inv,
    const CORE::LINALG::Matrix<numstr_, numdof_>& bop,
    const CORE::LINALG::Matrix<numdim_, numdim_>& C_inv,
    const CORE::LINALG::Matrix<1, numdof_>& dphi_dus,
    const CORE::LINALG::Matrix<1, numdof_>& dJ_dus,
    const CORE::LINALG::Matrix<numstr_, numdof_>& dCinv_dus,
    const CORE::LINALG::Matrix<numdim_ * numdim_, numdof_>& dFinvTdus,
    CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix, CORE::LINALG::Matrix<numdof_, 1>* force,
    CORE::LINALG::Matrix<numstr_, 1>& fstress)
{
  const double detJ_w = detJ_[gp] * intpoints_.Weight(gp) * thickness_;

  const double visc = fluid_mat_->Viscosity();
  CORE::LINALG::Matrix<numdim_, numdim_> CinvFvel;
  CORE::LINALG::Matrix<numdim_, numdim_> tmp;
  CinvFvel.Multiply(C_inv, fvelder);
  tmp.MultiplyNT(CinvFvel, defgrd_inv);
  CORE::LINALG::Matrix<numdim_, numdim_> tmp2(tmp);
  tmp.UpdateT(1.0, tmp2, 1.0);

  fstress(0) = tmp(0, 0);
  fstress(1) = tmp(1, 1);
  fstress(2) = tmp(0, 1);

  fstress.Scale(detJ_w * visc * J * porosity);

  // B^T . C^-1
  CORE::LINALG::Matrix<numdof_, 1> fstressb(true);
  fstressb.MultiplyTN(bop, fstress);

  if (force != nullptr) force->Update(1.0, fstressb, 1.0);

  // evaluate viscous terms (for darcy-brinkman flow only)
  if (stiffmatrix != nullptr)
  {
    CORE::LINALG::Matrix<numdim_, numdim_> tmp4;
    tmp4.MultiplyNT(fvelder, defgrd_inv);

    double fac = detJ_w * visc;

    CORE::LINALG::Matrix<numstr_, numdof_> fstress_dus(true);
    for (int n = 0; n < numnod_; ++n)
    {
      for (int k = 0; k < numdim_; ++k)
      {
        const int gid = n * numdim_ + k;

        fstress_dus(0, gid) +=
            2 * (dCinv_dus(0, gid) * tmp4(0, 0) + dCinv_dus(2, gid) * tmp4(1, 0));
        fstress_dus(1, gid) +=
            2 * (dCinv_dus(2, gid) * tmp4(0, 1) + dCinv_dus(1, gid) * tmp4(1, 1));
        /* ~~~ */
        fstress_dus(2, gid) += +dCinv_dus(0, gid) * tmp4(0, 1) + dCinv_dus(2, gid) * tmp4(1, 1) +
                               dCinv_dus(2, gid) * tmp4(0, 0) + dCinv_dus(1, gid) * tmp4(1, 0);

        for (int j = 0; j < numdim_; j++)
        {
          fstress_dus(0, gid) += 2 * CinvFvel(0, j) * dFinvTdus(j * numdim_, gid);
          fstress_dus(1, gid) += 2 * CinvFvel(1, j) * dFinvTdus(j * numdim_ + 1, gid);
          /* ~~~ */
          fstress_dus(2, gid) += +CinvFvel(0, j) * dFinvTdus(j * numdim_ + 1, gid) +
                                 CinvFvel(1, j) * dFinvTdus(j * numdim_, gid);
        }
      }
    }

    CORE::LINALG::Matrix<numdof_, numdof_> tmp;

    // additional viscous fluid stress- stiffness term (B^T . fstress . dJ/d(us) * porosity * detJ *
    // w(gp))
    tmp.Multiply(fac * porosity, fstressb, dJ_dus);
    stiffmatrix->Update(1.0, tmp, 1.0);

    // additional fluid stress- stiffness term (B^T .  d\phi/d(us) . fstress  * J * w(gp))
    tmp.Multiply(fac * J, fstressb, dphi_dus);
    stiffmatrix->Update(1.0, tmp, 1.0);

    // additional fluid stress- stiffness term (B^T .  phi . dfstress/d(us)  * J * w(gp))
    tmp.MultiplyTN(detJ_w * visc * J * porosity, bop, fstress_dus);
    stiffmatrix->Update(1.0, tmp, 1.0);
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::CouplingPoroelast(std::vector<int>& lm,  // location matrix
    CORE::LINALG::Matrix<numdim_, numnod_>& disp,    // current displacements
    CORE::LINALG::Matrix<numdim_, numnod_>& vel,     // current velocities
    CORE::LINALG::Matrix<numdim_, numnod_>& evelnp,  // current fluid velocity
    CORE::LINALG::Matrix<numnod_, 1>& epreaf,        // current fluid pressure
    CORE::LINALG::Matrix<numdof_, (numdim_ + 1) * numnod_>*
        stiffmatrix,                                                    // element stiffness matrix
    CORE::LINALG::Matrix<numdof_, (numdim_ + 1) * numnod_>* reamatrix,  // element reactive matrix
    CORE::LINALG::Matrix<numdof_, 1>* force,  // element internal force vector
    Teuchos::ParameterList& params)           // algorithmic parameters e.g. time
{
  GetMaterials();

  //=======================================================================

  // update element geometry
  CORE::LINALG::Matrix<numdim_, numnod_> xrefe;  // material coord. of element
  CORE::LINALG::Matrix<numdim_, numnod_> xcurr;  // current  coord. of element

  DRT::Node** nodes = Nodes();
  for (int i = 0; i < numnod_; ++i)
  {
    const auto& x = nodes[i]->X();
    for (int j = 0; j < numdim_; j++)
    {
      xrefe(j, i) = x[j];
      xcurr(j, i) = xrefe(j, i) + disp(j, i);
    }
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  if (stiffmatrix != nullptr)
    GaussPointLoopOD(params, xrefe, xcurr, disp, vel, evelnp, epreaf, nullptr, *stiffmatrix);
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::CouplingPoroelastPressureBased(
    std::vector<int>& lm,                          // location matrix
    CORE::LINALG::Matrix<numdim_, numnod_>& disp,  // current displacements
    const std::vector<double>& ephi,            // current primary variable for poro-multiphase flow
    CORE::LINALG::SerialDenseMatrix& couplmat,  // element stiffness matrix
    Teuchos::ParameterList& params)             // algorithmic parameters e.g. time
{
  GetMaterialsPressureBased();

  //=======================================================================

  // update element geometry
  CORE::LINALG::Matrix<numdim_, numnod_> xrefe;  // material coord. of element
  CORE::LINALG::Matrix<numdim_, numnod_> xcurr;  // current  coord. of element

  DRT::Node** nodes = Nodes();
  for (int i = 0; i < numnod_; ++i)
  {
    const auto& x = nodes[i]->X();
    for (int j = 0; j < numdim_; j++)
    {
      xrefe(j, i) = x[j];
      xcurr(j, i) = xrefe(j, i) + disp(j, i);
    }
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/

  GaussPointLoopODPressureBased(params, xrefe, xcurr, disp, ephi, couplmat);
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::GaussPointLoopOD(Teuchos::ParameterList& params,
    const CORE::LINALG::Matrix<numdim_, numnod_>& xrefe,
    const CORE::LINALG::Matrix<numdim_, numnod_>& xcurr,
    const CORE::LINALG::Matrix<numdim_, numnod_>& nodaldisp,
    const CORE::LINALG::Matrix<numdim_, numnod_>& nodalvel,
    const CORE::LINALG::Matrix<numdim_, numnod_>& evelnp,
    const CORE::LINALG::Matrix<numnod_, 1>& epreaf,
    const CORE::LINALG::Matrix<numnod_, 1>* porosity_dof,
    CORE::LINALG::Matrix<numdof_, (numdim_ + 1) * numnod_>& ecoupl)
{
  /*--------------------------------- get node weights for nurbs elements */
  if (distype == CORE::FE::CellType::nurbs4 || distype == CORE::FE::CellType::nurbs9)
  {
    for (int inode = 0; inode < numnod_; ++inode)
    {
      auto* cp = dynamic_cast<DRT::NURBS::ControlPoint*>(Nodes()[inode]);

      weights_(inode) = cp->W();
    }
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  CORE::LINALG::Matrix<numdim_, numnod_> N_XYZ;  //  first derivatives at gausspoint w.r.t. X, Y,Z
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  // CAUTION: defgrd(true): filled with zeros!
  CORE::LINALG::Matrix<numdim_, numdim_> defgrd(
      true);                                  //  deformation gradiant evaluated at gauss point
  CORE::LINALG::Matrix<numnod_, 1> shapefct;  //  shape functions evalulated at gauss point
  CORE::LINALG::Matrix<numdim_, numnod_> deriv(
      true);  //  first derivatives at gausspoint w.r.t. r,s,t
  // CORE::LINALG::Matrix<numdim_,1> xsi;
  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // evaluate shape functions and derivatives at integration point
    ComputeShapeFunctionsAndDerivatives(gp, shapefct, deriv, N_XYZ);

    // (material) deformation gradient F = d xcurr / d xrefe = xcurr * N_XYZ^T
    ComputeDefGradient(defgrd, N_XYZ, xcurr);

    // jacobian determinant of transformation between spatial and material space "|dx/dX|"
    double J = 0.0;
    // volume change (used for porosity law). Same as J in nonlinear theory.
    double volchange = 0.0;

    // compute J, the volume change and the respctive linearizations w.r.t. structure displacement
    ComputeJacobianDeterminantVolumeChange(J, volchange, defgrd, N_XYZ, nodaldisp);

    // non-linear B-operator (may so be called, meaning
    CORE::LINALG::Matrix<numstr_, numdof_> bop;
    ComputeBOperator(bop, defgrd, N_XYZ);

    // Right Cauchy-Green tensor = F^T * F
    CORE::LINALG::Matrix<numdim_, numdim_> cauchygreen;
    cauchygreen.MultiplyTN(defgrd, defgrd);

    // inverse Right Cauchy-Green tensor
    CORE::LINALG::Matrix<numdim_, numdim_> C_inv(false);
    C_inv.Invert(cauchygreen);

    // inverse deformation gradient F^-1
    CORE::LINALG::Matrix<numdim_, numdim_> defgrd_inv(false);
    defgrd_inv.Invert(defgrd);

    //---------------- get pressure at integration point
    double press = shapefct.Dot(epreaf);

    //------------------ get material pressure gradient at integration point
    CORE::LINALG::Matrix<numdim_, 1> Gradp;
    Gradp.Multiply(N_XYZ, epreaf);

    //--------------------- get fluid velocity at integration point
    CORE::LINALG::Matrix<numdim_, 1> fvelint;
    fvelint.Multiply(evelnp, shapefct);

    // material fluid velocity gradient at integration point
    CORE::LINALG::Matrix<numdim_, numdim_> fvelder;
    fvelder.MultiplyNT(evelnp, N_XYZ);

    //----------------structure displacement and velocity at integration point
    CORE::LINALG::Matrix<numdim_, 1> velint(true);
    for (int i = 0; i < numnod_; i++)
      for (int j = 0; j < numdim_; j++) velint(j) += nodalvel(j, i) * shapefct(i);

    // auxilary variables for computing the porosity and linearization
    double dphi_dp = 0.0;
    double porosity = 0.0;

    ComputePorosityAndLinearizationOD(
        params, press, volchange, gp, shapefct, porosity_dof, porosity, dphi_dp);

    // **********************evaluate stiffness matrix and force vector**********************
    FillMatrixAndVectorsOD(gp, shapefct, N_XYZ, J, porosity, dphi_dp, velint, fvelint, defgrd_inv,
        Gradp, bop, C_inv, ecoupl);

    if (fluid_mat_->Type() == MAT::PAR::darcy_brinkman)
    {
      FillMatrixAndVectorsBrinkmanOD(
          gp, shapefct, N_XYZ, J, porosity, dphi_dp, fvelder, defgrd_inv, bop, C_inv, ecoupl);
    }
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::GaussPointLoopODPressureBased(
    Teuchos::ParameterList& params, const CORE::LINALG::Matrix<numdim_, numnod_>& xrefe,
    const CORE::LINALG::Matrix<numdim_, numnod_>& xcurr,
    const CORE::LINALG::Matrix<numdim_, numnod_>& nodaldisp, const std::vector<double>& ephi,
    CORE::LINALG::SerialDenseMatrix& couplmat)
{
  /*--------------------------------- get node weights for nurbs elements */
  if (distype == CORE::FE::CellType::nurbs4 || distype == CORE::FE::CellType::nurbs9)
  {
    for (int inode = 0; inode < numnod_; ++inode)
    {
      auto* cp = dynamic_cast<DRT::NURBS::ControlPoint*>(Nodes()[inode]);

      weights_(inode) = cp->W();
    }
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  CORE::LINALG::Matrix<numdim_, numnod_> N_XYZ;  //  first derivatives at gausspoint w.r.t. X, Y,Z
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  // CAUTION: defgrd(true): filled with zeros!
  CORE::LINALG::Matrix<numdim_, numdim_> defgrd(
      true);                                  //  deformation gradiant evaluated at gauss point
  CORE::LINALG::Matrix<numnod_, 1> shapefct;  //  shape functions evalulated at gauss point
  CORE::LINALG::Matrix<numdim_, numnod_> deriv(
      true);  //  first derivatives at gausspoint w.r.t. r,s,t

  // Initialize
  const int numfluidphases = fluidmulti_mat_->NumFluidPhases();
  const int totalnumdofpernode = fluidmulti_mat_->NumMat();
  const int numvolfrac = fluidmulti_mat_->NumVolFrac();
  const bool hasvolfracs = (totalnumdofpernode - numfluidphases);
  std::vector<double> phiAtGP(totalnumdofpernode);
  std::vector<double> solpressderiv(totalnumdofpernode);

  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // evaluate shape functions and derivatives at integration point
    ComputeShapeFunctionsAndDerivatives(gp, shapefct, deriv, N_XYZ);

    // (material) deformation gradient F = d xcurr / d xrefe = xcurr * N_XYZ^T
    ComputeDefGradient(defgrd, N_XYZ, xcurr);

    // jacobian determinant of transformation between spatial and material space "|dx/dX|"
    double J = 0.0;
    // volume change (used for porosity law). Same as J in nonlinear theory.
    double volchange = 0.0;

    // compute J, the volume change and the respctive linearizations w.r.t. structure displacement
    ComputeJacobianDeterminantVolumeChange(J, volchange, defgrd, N_XYZ, nodaldisp);

    // non-linear B-operator (may so be called, meaning
    CORE::LINALG::Matrix<numstr_, numdof_> bop;
    ComputeBOperator(bop, defgrd, N_XYZ);

    // Right Cauchy-Green tensor = F^T * F
    CORE::LINALG::Matrix<numdim_, numdim_> cauchygreen;
    cauchygreen.MultiplyTN(defgrd, defgrd);

    // inverse Right Cauchy-Green tensor
    CORE::LINALG::Matrix<numdim_, numdim_> C_inv(false);
    C_inv.Invert(cauchygreen);

    // compute derivative of solid pressure w.r.t primary variable phi at node
    ComputePrimaryVariableAtGP(ephi, totalnumdofpernode, shapefct, phiAtGP);
    ComputeSolPressureDeriv(phiAtGP, numfluidphases, solpressderiv);
    // in case of volume fractions --> recalculate
    if (hasvolfracs)
    {
      double dphi_dp = 0.0;
      double porosity = 0.0;

      double press = ComputeSolPressureAtGP(totalnumdofpernode, numfluidphases, phiAtGP);

      ComputePorosityAndLinearizationOD(
          params, press, volchange, gp, shapefct, nullptr, porosity, dphi_dp);

      RecalculateSolPressureDeriv(
          phiAtGP, totalnumdofpernode, numfluidphases, numvolfrac, press, porosity, solpressderiv);
    }

    // **********************evaluate stiffness matrix and force vector**********************
    FillMatrixAndVectorsODPressureBased(
        gp, shapefct, N_XYZ, J, bop, C_inv, solpressderiv, couplmat);
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::FillMatrixAndVectorsOD(const int& gp,
    const CORE::LINALG::Matrix<numnod_, 1>& shapefct,
    const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& porosity,
    const double& dphi_dp, const CORE::LINALG::Matrix<numdim_, 1>& velint,
    const CORE::LINALG::Matrix<numdim_, 1>& fvelint,
    const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd_inv,
    const CORE::LINALG::Matrix<numdim_, 1>& Gradp,
    const CORE::LINALG::Matrix<numstr_, numdof_>& bop,
    const CORE::LINALG::Matrix<numdim_, numdim_>& C_inv,
    CORE::LINALG::Matrix<numdof_, (numdim_ + 1) * numnod_>& ecoupl)
{
  CORE::LINALG::Matrix<numdim_, numdim_> matreatensor(true);
  CORE::LINALG::Matrix<numdim_, numdim_> reatensor(true);
  CORE::LINALG::Matrix<numdim_, numdim_> linreac_dphi(true);
  CORE::LINALG::Matrix<numdim_, numdim_> linreac_dJ(true);
  CORE::LINALG::Matrix<numdim_, 1> reafvel(true);
  CORE::LINALG::Matrix<numdim_, 1> reavel(true);
  {
    CORE::LINALG::Matrix<numdim_, numdim_> temp(true);
    std::vector<double> anisotropic_permeability_coeffs =
        ComputeAnisotropicPermeabilityCoeffsAtGP(shapefct);
    fluid_mat_->ComputeReactionTensor(matreatensor, J, porosity,
        anisotropic_permeability_directions_, anisotropic_permeability_coeffs);
    fluid_mat_->ComputeLinMatReactionTensor(linreac_dphi, linreac_dJ, J, porosity);
    temp.Multiply(1.0, matreatensor, defgrd_inv);
    reatensor.MultiplyTN(defgrd_inv, temp);
    reavel.Multiply(reatensor, velint);
    reafvel.Multiply(reatensor, fvelint);
  }

  const double detJ_w = detJ_[gp] * intpoints_.Weight(gp) * thickness_;

  // inverse Right Cauchy-Green tensor as vector
  CORE::LINALG::Matrix<numstr_, 1> C_inv_vec;
  C_inv_vec(0) = C_inv(0, 0);
  C_inv_vec(1) = C_inv(1, 1);
  C_inv_vec(2) = C_inv(0, 1);

  // B^T . C^-1
  CORE::LINALG::Matrix<numdof_, 1> cinvb(true);
  cinvb.MultiplyTN(bop, C_inv_vec);

  // F^-T * Grad p
  CORE::LINALG::Matrix<numdim_, 1> Finvgradp;
  Finvgradp.MultiplyTN(defgrd_inv, Gradp);

  // F^-T * N_XYZ
  CORE::LINALG::Matrix<numdim_, numnod_> FinvNXYZ;
  FinvNXYZ.MultiplyTN(defgrd_inv, N_XYZ);

  {
    for (int i = 0; i < numnod_; i++)
    {
      const int fi = numdim_ * i;
      const double fac = detJ_w * shapefct(i);

      for (int j = 0; j < numdim_; j++)
      {
        for (int k = 0; k < numnod_; k++)
        {
          const int fk = (numdim_ + 1) * k;
          const int fk_press = fk + numdim_;

          /*-------structure- fluid pressure coupling: "stress terms" + "pressure gradient terms"
           -B^T . ( -1*J*C^-1 ) * Dp
           - J * F^-T * Grad(p) * dphi/dp * Dp - J * F^-T * d(Grad((p))/(dp) * phi * Dp
           */
          ecoupl(fi + j, fk_press) +=
              detJ_w * cinvb(fi + j) * (-1.0) * J * shapefct(k) -
              fac * J * (dphi_dp * Finvgradp(j) * shapefct(k) + porosity * FinvNXYZ(j, k));

          /*-------structure- fluid pressure coupling:  "darcy-terms" + "reactive darcy-terms"
           - 2 * reacoeff * J * v^f * phi * d(phi)/dp  Dp
           + 2 * reacoeff * J * v^s * phi * d(phi)/dp  Dp
           + J * J * phi * phi * defgrd_^-T * d(mat_reacoeff)/d(phi) * defgrd_^-1 * (v^s-v^f) *
           d(phi)/dp Dp
           */
          const double tmp = fac * J * J * 2 * porosity * dphi_dp * shapefct(k);
          ecoupl(fi + j, fk_press) += -tmp * reafvel(j);

          ecoupl(fi + j, fk_press) += tmp * reavel(j);

          // check if derivatives of reaction tensor are zero --> significant speed up
          if (fluid_mat_->PermeabilityFunction() != MAT::PAR::constant)
          {
            const double tmp2 = 0.5 * tmp * porosity;
            for (int m = 0; m < numdim_; ++m)
            {
              for (int n = 0; n < numdim_; ++n)
              {
                for (int p = 0; p < numdim_; ++p)
                {
                  ecoupl(fi + j, fk_press) += tmp2 * defgrd_inv(m, j) * linreac_dphi(m, n) *
                                              defgrd_inv(n, p) * (velint(p) - fvelint(p));
                }
              }
            }
          }

          /*-------structure- fluid velocity coupling:  "darcy-terms"
           -reacoeff * J * J *  phi^2 *  Dv^f
           */
          const double v = fac * J * J * porosity * porosity;
          for (int l = 0; l < numdim_; l++)
            ecoupl(fi + j, fk + l) += -v * reatensor(j, l) * shapefct(k);
        }
      }
    }
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::FillMatrixAndVectorsODPressureBased(const int& gp,
    const CORE::LINALG::Matrix<numnod_, 1>& shapefct,
    const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ, const double& J,
    const CORE::LINALG::Matrix<numstr_, numdof_>& bop,
    const CORE::LINALG::Matrix<numdim_, numdim_>& C_inv, const std::vector<double>& solpressderiv,
    CORE::LINALG::SerialDenseMatrix& couplmat)
{
  const double detJ_w = detJ_[gp] * intpoints_.Weight(gp) * thickness_;

  // inverse Right Cauchy-Green tensor as vector
  CORE::LINALG::Matrix<numstr_, 1> C_inv_vec;
  C_inv_vec(0) = C_inv(0, 0);
  C_inv_vec(1) = C_inv(1, 1);
  C_inv_vec(2) = C_inv(0, 1);

  // B^T . C^-1
  CORE::LINALG::Matrix<numdof_, 1> cinvb(true);
  cinvb.MultiplyTN(bop, C_inv_vec);

  const int totalnumdofpernode = fluidmulti_mat_->NumMat();

  {
    for (int i = 0; i < numnod_; i++)
    {
      const int fi = numdim_ * i;

      for (int j = 0; j < numdim_; j++)
      {
        for (int k = 0; k < numnod_; k++)
        {
          for (int iphase = 0; iphase < totalnumdofpernode; iphase++)
          {
            int fk_press = k * totalnumdofpernode + iphase;

            /*-------structure- fluid pressure coupling: "stress term"
             -B^T . ( -1*J*C^-1 ) * Dp
             */
            couplmat(fi + j, fk_press) +=
                detJ_w * cinvb(fi + j) * (-1.0) * J * shapefct(k) * solpressderiv[iphase];
          }
        }
      }
    }
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::FillMatrixAndVectorsBrinkmanOD(const int& gp,
    const CORE::LINALG::Matrix<numnod_, 1>& shapefct,
    const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& porosity,
    const double& dphi_dp, const CORE::LINALG::Matrix<numdim_, numdim_>& fvelder,
    const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd_inv,
    const CORE::LINALG::Matrix<numstr_, numdof_>& bop,
    const CORE::LINALG::Matrix<numdim_, numdim_>& C_inv,
    CORE::LINALG::Matrix<numdof_, (numdim_ + 1) * numnod_>& ecoupl)
{
  const double detJ_w = detJ_[gp] * intpoints_.Weight(gp) * thickness_;
  const double visc = fluid_mat_->Viscosity();

  CORE::LINALG::Matrix<numstr_, 1> fstress;

  CORE::LINALG::Matrix<numdim_, numdim_> CinvFvel;
  CORE::LINALG::Matrix<numdim_, numdim_> tmp;
  CinvFvel.Multiply(C_inv, fvelder);
  tmp.MultiplyNT(CinvFvel, defgrd_inv);
  CORE::LINALG::Matrix<numdim_, numdim_> tmp2(tmp);
  tmp.UpdateT(1.0, tmp2, 1.0);

  fstress(0) = tmp(0, 0);
  fstress(1) = tmp(1, 1);
  fstress(2) = tmp(0, 1);

  // B^T . \sigma
  CORE::LINALG::Matrix<numdof_, 1> fstressb;
  fstressb.MultiplyTN(bop, fstress);
  CORE::LINALG::Matrix<numdim_, numnod_> N_XYZ_Finv;
  N_XYZ_Finv.Multiply(defgrd_inv, N_XYZ);

  // dfstress/dv^f
  CORE::LINALG::Matrix<numstr_, numdof_> dfstressb_dv;
  for (int i = 0; i < numnod_; i++)
  {
    const int fi = numdim_ * i;
    for (int j = 0; j < numdim_; j++)
    {
      int k = fi + j;
      dfstressb_dv(0, k) = 2 * N_XYZ_Finv(0, i) * C_inv(0, j);
      dfstressb_dv(1, k) = 2 * N_XYZ_Finv(1, i) * C_inv(1, j);

      dfstressb_dv(2, k) = N_XYZ_Finv(0, i) * C_inv(1, j) + N_XYZ_Finv(1, i) * C_inv(0, j);
    }
  }

  // B^T . dfstress/dv^f
  CORE::LINALG::Matrix<numdof_, numdof_> dfstressb_dv_bop(true);
  dfstressb_dv_bop.MultiplyTN(bop, dfstressb_dv);

  for (int i = 0; i < numnod_; i++)
  {
    const int fi = numdim_ * i;

    for (int j = 0; j < numdim_; j++)
    {
      for (int k = 0; k < numnod_; k++)
      {
        const int fk_sub = numdim_ * k;
        const int fk = (numdim_ + 1) * k;
        const int fk_press = fk + numdim_;

        /*-------structure- fluid pressure coupling: "darcy-brinkman stress terms"
         B^T . ( \mu*J - d(phi)/(dp) * fstress ) * Dp
         */
        ecoupl(fi + j, fk_press) += detJ_w * fstressb(fi + j) * dphi_dp * visc * J * shapefct(k);
        for (int l = 0; l < numdim_; l++)
        {
          /*-------structure- fluid velocity coupling: "darcy-brinkman stress terms"
           B^T . ( \mu*J - phi * dfstress/dv^f ) * Dp
           */
          ecoupl(fi + j, fk + l) +=
              detJ_w * visc * J * porosity * dfstressb_dv_bop(fi + j, fk_sub + l);
        }
      }
    }
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::CouplingStressPoroelast(
    CORE::LINALG::Matrix<numdim_, numnod_>& disp,    // current displacements
    CORE::LINALG::Matrix<numdim_, numnod_>& evelnp,  // current fluid velocities
    CORE::LINALG::Matrix<numnod_, 1>& epreaf,        // current fluid pressure
    CORE::LINALG::SerialDenseMatrix* elestress,      // stresses at GP
    CORE::LINALG::SerialDenseMatrix* elestrain,      // strains at GP
    Teuchos::ParameterList& params,                  // algorithmic parameters e.g. time
    const INPAR::STR::StressType iostress            // stress output option
)
{
  // update element geometry
  CORE::LINALG::Matrix<numdim_, numnod_> xrefe;  // material coord. of element
  CORE::LINALG::Matrix<numdim_, numnod_> xcurr;  // current  coord. of element

  DRT::Node** nodes = Nodes();
  for (int i = 0; i < numnod_; ++i)
  {
    const auto& x = nodes[i]->X();
    for (int j = 0; j < numdim_; j++)
    {
      xrefe(j, i) = x[j];
      xcurr(j, i) = xrefe(j, i) + disp(j, i);
    }
  }
  CORE::LINALG::Matrix<numnod_, 1> shapefct;
  CORE::LINALG::Matrix<numdim_, numdim_> defgrd(true);
  CORE::LINALG::Matrix<numdim_, numnod_> N_XYZ;
  CORE::LINALG::Matrix<numdim_, numnod_> deriv;

  // get structure material
  Teuchos::RCP<MAT::StructPoro> structmat = Teuchos::rcp_dynamic_cast<MAT::StructPoro>(Material());
  if (structmat->MaterialType() != INPAR::MAT::m_structporo)
    dserror("invalid structure material for poroelasticity");

  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // evaluate shape functions and derivatives at integration point
    ComputeShapeFunctionsAndDerivatives(gp, shapefct, deriv, N_XYZ);

    // (material) deformation gradient F = d xcurr / d xrefe = xcurr * N_XYZ^T
    ComputeDefGradient(defgrd, N_XYZ, xcurr);

    //----------------------------------------------------
    // pressure at integration point
    double press = shapefct.Dot(epreaf);

    CORE::LINALG::Matrix<Wall1::numstr_, 1> couplstress(true);

    structmat->CouplStress(defgrd, press, couplstress);

    // return gp stresses
    switch (iostress)
    {
      case INPAR::STR::stress_2pk:
      {
        if (elestress == nullptr) dserror("stress data not available");
        for (int i = 0; i < numstr_; ++i) (*elestress)(gp, i) = couplstress(i);
      }
      break;
      case INPAR::STR::stress_cauchy:
      {
        if (elestress == nullptr) dserror("stress data not available");

        // push forward of material stress to the spatial configuration
        CORE::LINALG::Matrix<numdim_, numdim_> cauchycouplstress;
        PK2toCauchy(couplstress, defgrd, cauchycouplstress);

        (*elestress)(gp, 0) = cauchycouplstress(0, 0);
        (*elestress)(gp, 1) = cauchycouplstress(1, 1);
        (*elestress)(gp, 2) = 0.0;
        (*elestress)(gp, 3) = cauchycouplstress(0, 1);
      }
      break;
      case INPAR::STR::stress_none:
        break;

      default:
        dserror("requested stress type not available");
        break;
    }
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::InitElement()
{
  CORE::LINALG::Matrix<numdim_, numnod_> deriv;
  CORE::LINALG::Matrix<numnod_, numdim_> xrefe;
  for (int i = 0; i < numnod_; ++i)
  {
    Node** nodes = Nodes();
    if (!nodes) dserror("Nodes() returned null pointer");
    for (int j = 0; j < numdim_; ++j) xrefe(i, j) = Nodes()[i]->X()[j];
  }
  invJ_.resize(numgpt_);
  detJ_.resize(numgpt_);
  xsi_.resize(numgpt_);

  for (int gp = 0; gp < numgpt_; ++gp)
  {
    const double* gpcoord = intpoints_.Point(gp);
    for (int idim = 0; idim < numdim_; idim++)
    {
      xsi_[gp](idim) = gpcoord[idim];
    }
  }

  if (distype != CORE::FE::CellType::nurbs4 and distype != CORE::FE::CellType::nurbs9)
  {
    for (int gp = 0; gp < numgpt_; ++gp)
    {
      CORE::DRT::UTILS::shape_function_deriv1<distype>(xsi_[gp], deriv);

      invJ_[gp].Multiply(deriv, xrefe);
      detJ_[gp] = invJ_[gp].Invert();
      if (detJ_[gp] <= 0.0) dserror("Element Jacobian mapping %10.5e <= 0.0", detJ_[gp]);
    }
  }

  scatra_coupling_ = false;

  ProblemType probtype = DRT::Problem::Instance()->GetProblemType();
  if (probtype == ProblemType::poroscatra) scatra_coupling_ = true;

  init_ = true;
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::ComputeJacobianDeterminantVolumeChangeAndLinearizations(
    double& J, double& volchange, CORE::LINALG::Matrix<1, numdof_>& dJ_dus,
    CORE::LINALG::Matrix<1, numdof_>& dvolchange_dus,
    const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd,
    const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd_inv,
    const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ,
    const CORE::LINALG::Matrix<numdim_, numnod_>& nodaldisp)
{
  // compute J
  J = defgrd.Determinant();
  // compute linearization of J
  ComputeLinearizationOfJacobian(dJ_dus, J, N_XYZ, defgrd_inv);

  if (kintype_ == INPAR::STR::kinem_nonlinearTotLag)  // total lagrange (nonlinear)
  {
    // for nonlinear kinematics the Jacobian of the deformation gradient is the volume change
    volchange = J;
    dvolchange_dus = dJ_dus;
  }
  else if (kintype_ == INPAR::STR::kinem_linear)  // linear kinematics
  {
    // for linear kinematics the volume change is the trace of the linearized strains

    // gradient of displacements
    static CORE::LINALG::Matrix<numdim_, numdim_> dispgrad;
    dispgrad.Clear();
    // gradient of displacements
    dispgrad.MultiplyNT(nodaldisp, N_XYZ);

    volchange = 1.0;
    // volchange = 1 + trace of the linearized strains (= trace of displacement gradient)
    for (int i = 0; i < numdim_; ++i) volchange += dispgrad(i, i);

    for (int i = 0; i < numdim_; ++i)
      for (int j = 0; j < numnod_; ++j) dvolchange_dus(numdim_ * j + i) = N_XYZ(i, j);
  }
  else
    dserror("invalid kinematic type!");
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::ComputeJacobianDeterminantVolumeChange(double& J,
    double& volchange, const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd,
    const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ,
    const CORE::LINALG::Matrix<numdim_, numnod_>& nodaldisp)
{
  // compute J
  J = defgrd.Determinant();

  if (kintype_ == INPAR::STR::kinem_nonlinearTotLag)  // total lagrange (nonlinear)
  {
    // for nonlinear kinematics the Jacobian of the deformation gradient is the volume change
    volchange = J;
  }
  else if (kintype_ == INPAR::STR::kinem_linear)  // linear kinematics
  {
    // for linear kinematics the volume change is the trace of the linearized strains

    // gradient of displacements
    static CORE::LINALG::Matrix<numdim_, numdim_> dispgrad;
    dispgrad.Clear();
    // gradient of displacements
    dispgrad.MultiplyNT(nodaldisp, N_XYZ);

    volchange = 1.0;
    // volchange = 1 + trace of the linearized strains (= trace of displacement gradient)
    for (int i = 0; i < numdim_; ++i) volchange += dispgrad(i, i);
  }
  else
    dserror("invalid kinematic type!");
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::PK2toCauchy(
    CORE::LINALG::Matrix<Wall1::numstr_, 1>& stress, CORE::LINALG::Matrix<numdim_, numdim_>& defgrd,
    CORE::LINALG::Matrix<numdim_, numdim_>& cauchystress)
{
  // calculate the Jacobi-deterinant
  const double detF = (defgrd).Determinant();

  // sigma = 1/J . F . S . F^T
  CORE::LINALG::Matrix<numdim_, numdim_> pkstress;
  pkstress(0, 0) = (stress)(0);
  pkstress(0, 1) = (stress)(2);
  pkstress(1, 0) = pkstress(0, 1);
  pkstress(1, 1) = (stress)(1);

  CORE::LINALG::Matrix<numdim_, numdim_> temp;
  temp.Multiply((1.0 / detF), (defgrd), pkstress);
  (cauchystress).MultiplyNT(temp, (defgrd));
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::ComputeDefGradient(
    CORE::LINALG::Matrix<numdim_, numdim_>& defgrd,
    const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ,
    const CORE::LINALG::Matrix<numdim_, numnod_>& xcurr)
{
  if (kintype_ == INPAR::STR::kinem_nonlinearTotLag)
  {
    // (material) deformation gradient F = d xcurr / d xrefe = xcurr * N_XYZ^T
    defgrd.MultiplyNT(xcurr, N_XYZ);  //  (6.17)
  }
  else if (kintype_ == INPAR::STR::kinem_linear)
  {
    defgrd.Clear();
    for (int i = 0; i < numdim_; i++) defgrd(i, i) = 1.0;
  }
  else
    dserror("invalid kinematic type!");
}

template <CORE::FE::CellType distype>
inline void DRT::ELEMENTS::Wall1_Poro<distype>::ComputeBOperator(
    CORE::LINALG::Matrix<numstr_, numdof_>& bop,
    const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd,
    const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ)
{
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
  for (int i = 0; i < numnod_; ++i)
  {
    bop(0, noddof_ * i + 0) = defgrd(0, 0) * N_XYZ(0, i);
    bop(0, noddof_ * i + 1) = defgrd(1, 0) * N_XYZ(0, i);
    bop(1, noddof_ * i + 0) = defgrd(0, 1) * N_XYZ(1, i);
    bop(1, noddof_ * i + 1) = defgrd(1, 1) * N_XYZ(1, i);
    /* ~~~ */
    bop(2, noddof_ * i + 0) = defgrd(0, 0) * N_XYZ(1, i) + defgrd(0, 1) * N_XYZ(0, i);
    bop(2, noddof_ * i + 1) = defgrd(1, 0) * N_XYZ(1, i) + defgrd(1, 1) * N_XYZ(0, i);
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::ComputeShapeFunctionsAndDerivatives(const int& gp,
    CORE::LINALG::Matrix<numnod_, 1>& shapefct, CORE::LINALG::Matrix<numdim_, numnod_>& deriv,
    CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ)
{
  // get values of shape functions and derivatives in the gausspoint
  if (distype != CORE::FE::CellType::nurbs4 and distype != CORE::FE::CellType::nurbs9)
  {
    // shape functions and their derivatives for polynomials
    CORE::DRT::UTILS::shape_function<distype>(xsi_[gp], shapefct);
    CORE::DRT::UTILS::shape_function_deriv1<distype>(xsi_[gp], deriv);
  }
  else
  {
    // nurbs version
    CORE::DRT::NURBS::UTILS::nurbs_get_funct_deriv(
        shapefct, deriv, xsi_[gp], myknots_, weights_, distype);

    CORE::LINALG::Matrix<numnod_, numdim_> xrefe;
    for (int i = 0; i < numnod_; ++i)
    {
      Node** nodes = Nodes();
      if (!nodes) dserror("Nodes() returned null pointer");
      xrefe(i, 0) = Nodes()[i]->X()[0];
      xrefe(i, 1) = Nodes()[i]->X()[1];
    }
    invJ_[gp].Multiply(deriv, xrefe);
    detJ_[gp] = invJ_[gp].Invert();
    if (detJ_[gp] <= 0.0) dserror("Element Jacobian mapping %10.5e <= 0.0", detJ_[gp]);
  }

  /* get the inverse of the Jacobian matrix which looks like:
   **            [ X_,r  Y_,r  Z_,r ]^-1
   **     J^-1 = [ X_,s  Y_,s  Z_,s ]
   **            [ X_,t  Y_,t  Z_,t ]
   */

  // compute derivatives N_XYZ at gp w.r.t. material coordinates
  // by N_XYZ = J^-1 * N_rst
  N_XYZ.Multiply(invJ_[gp], deriv);  // (6.21)
}

template <CORE::FE::CellType distype>
double DRT::ELEMENTS::Wall1_Poro<distype>::ComputeJacobianDeterminant(const int& gp,
    const CORE::LINALG::Matrix<numdim_, numnod_>& xcurr,
    const CORE::LINALG::Matrix<numdim_, numnod_>& deriv)
{
  // get Jacobian matrix and determinant w.r.t. spatial configuration
  // transposed jacobian "dx/ds"
  CORE::LINALG::Matrix<numdim_, numdim_> xjm;
  // inverse of transposed jacobian "ds/dx"
  CORE::LINALG::Matrix<numdim_, numdim_> xji;
  xjm.MultiplyNT(deriv, xcurr);
  const double det = xji.Invert(xjm);

  // determinant of deformationgradient: det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds) )^-1
  const double J = det / detJ_[gp];

  return J;
}

template <CORE::FE::CellType distype>
inline void DRT::ELEMENTS::Wall1_Poro<distype>::ComputeLinearizationOfJacobian(
    CORE::LINALG::Matrix<1, numdof_>& dJ_dus, const double& J,
    const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ,
    const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd_inv)
{
  //--------------------------- build N_X operator (wrt material config)
  CORE::LINALG::Matrix<numdim_ * numdim_, numdof_> N_X(true);  // set to zero
  for (int i = 0; i < numnod_; ++i)
  {
    N_X(0, numdim_ * i + 0) = N_XYZ(0, i);
    N_X(1, numdim_ * i + 1) = N_XYZ(0, i);

    N_X(2, numdim_ * i + 0) = N_XYZ(1, i);
    N_X(3, numdim_ * i + 1) = N_XYZ(1, i);
  }

  //------------------------------------ build F^-1 as vector 4x1
  CORE::LINALG::Matrix<numdim_ * numdim_, 1> defgrd_inv_vec;
  defgrd_inv_vec(0) = defgrd_inv(0, 0);
  defgrd_inv_vec(1) = defgrd_inv(0, 1);
  defgrd_inv_vec(2) = defgrd_inv(1, 0);
  defgrd_inv_vec(3) = defgrd_inv(1, 1);

  //------linearization of jacobi determinant detF=J w.r.t. strucuture displacement   dJ/d(us) =
  // dJ/dF : dF/dus = J * F^-T * N,X
  dJ_dus.MultiplyTN(J, defgrd_inv_vec, N_X);
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::ComputeAuxiliaryValues(
    const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ,
    const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd_inv,
    const CORE::LINALG::Matrix<numdim_, numdim_>& C_inv,
    const CORE::LINALG::Matrix<numdim_, 1>& Gradp,
    CORE::LINALG::Matrix<numdim_ * numdim_, numdof_>& dFinvTdus,
    CORE::LINALG::Matrix<numdim_, 1>& Finvgradp,
    CORE::LINALG::Matrix<numdim_, numdof_>& dFinvdus_gradp,
    CORE::LINALG::Matrix<numstr_, numdof_>& dCinv_dus)
{
  // F^-T * Grad p
  Finvgradp.MultiplyTN(defgrd_inv, Gradp);

  if (kintype_ != INPAR::STR::kinem_linear)
  {
    // dF^-T/dus
    for (int i = 0; i < numdim_; i++)
    {
      for (int n = 0; n < numnod_; n++)
      {
        for (int j = 0; j < numdim_; j++)
        {
          const int gid = numdim_ * n + j;
          for (int k = 0; k < numdim_; k++)
            for (int l = 0; l < numdim_; l++)
              dFinvTdus(i * numdim_ + l, gid) += -defgrd_inv(l, j) * N_XYZ(k, n) * defgrd_inv(k, i);
        }
      }
    }

    // dF^-T/dus * Grad p
    for (int i = 0; i < numdim_; i++)
    {
      for (int n = 0; n < numnod_; n++)
      {
        for (int j = 0; j < numdim_; j++)
        {
          const int gid = numdim_ * n + j;
          for (int l = 0; l < numdim_; l++)
            dFinvdus_gradp(i, gid) += dFinvTdus(i * numdim_ + l, gid) * Gradp(l);
        }
      }
    }
  }

  for (int n = 0; n < numnod_; ++n)
  {
    for (int k = 0; k < numdim_; ++k)
    {
      const int gid = n * numdim_ + k;
      for (int i = 0; i < numdim_; ++i)
      {
        dCinv_dus(0, gid) += -2 * C_inv(0, i) * N_XYZ(i, n) * defgrd_inv(0, k);
        dCinv_dus(1, gid) += -2 * C_inv(1, i) * N_XYZ(i, n) * defgrd_inv(1, k);
        /* ~~~ */
        dCinv_dus(2, gid) += -C_inv(0, i) * N_XYZ(i, n) * defgrd_inv(1, k) -
                             defgrd_inv(0, k) * N_XYZ(i, n) * C_inv(1, i);
      }
    }
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::ComputePorosityAndLinearization(
    Teuchos::ParameterList& params, const double& press, const double& J, const int& gp,
    const CORE::LINALG::Matrix<numnod_, 1>& shapfct,
    const CORE::LINALG::Matrix<numnod_, 1>* myporosity,
    const CORE::LINALG::Matrix<1, numdof_>& dJ_dus, double& porosity,
    CORE::LINALG::Matrix<1, numdof_>& dphi_dus)
{
  double dphi_dJ = 0.0;

  struct_mat_->ComputePorosity(params, press, J, gp, porosity, nullptr, &dphi_dJ, nullptr, nullptr,
      nullptr  // dphi_dpp not needed
  );

  // linearization of porosity w.r.t structure displacement d\phi/d(us) = d\phi/dJ*dJ/d(us)
  dphi_dus.Update(dphi_dJ, dJ_dus);
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::ComputePorosityAndLinearizationOD(
    Teuchos::ParameterList& params, const double& press, const double& J, const int& gp,
    const CORE::LINALG::Matrix<numnod_, 1>& shapfct,
    const CORE::LINALG::Matrix<numnod_, 1>* myporosity, double& porosity, double& dphi_dp)
{
  struct_mat_->ComputePorosity(params, press, J, gp, porosity, &dphi_dp,
      nullptr,  // dphi_dJ not needed
      nullptr,  // dphi_dJdp not needed
      nullptr,  // dphi_dJJ not needed
      nullptr   // dphi_dpp not needed
  );
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::ComputeSolPressureDeriv(const std::vector<double>& phiAtGP,
    const int numfluidphases, std::vector<double>& solidpressderiv)
{
  // zero out everything
  std::fill(solidpressderiv.begin(), solidpressderiv.end(), 0.0);

  // initialize auxiliary variables
  std::vector<double> genpress(numfluidphases);
  std::vector<double> press(numfluidphases);
  std::vector<double> sat(numfluidphases);
  CORE::LINALG::SerialDenseMatrix helpderiv(numfluidphases, numfluidphases, true);
  CORE::LINALG::SerialDenseMatrix satderiv(numfluidphases, numfluidphases, true);
  CORE::LINALG::SerialDenseMatrix pressderiv(numfluidphases, numfluidphases, true);
  std::vector<double> fluidphi(phiAtGP.data(), phiAtGP.data() + numfluidphases);

  // evaluate the pressures
  fluidmulti_mat_->EvaluateGenPressure(genpress, fluidphi);

  // transform generalized pressures to true pressure values
  fluidmulti_mat_->TransformGenPresToTruePres(genpress, press);

  // explicit evaluation of saturation
  fluidmulti_mat_->EvaluateSaturation(sat, fluidphi, press);

  // calculate the derivative of the pressure (actually first its inverse)
  fluidmulti_mat_->EvaluateDerivOfDofWrtPressure(pressderiv, fluidphi);

  // now invert the derivatives of the dofs w.r.t. pressure to get the derivatives
  // of the pressure w.r.t. the dofs
  {
    using ordinalType = CORE::LINALG::SerialDenseMatrix::ordinalType;
    using scalarType = CORE::LINALG::SerialDenseMatrix::scalarType;
    Teuchos::SerialDenseSolver<ordinalType, scalarType> inverse;
    inverse.setMatrix(Teuchos::rcpFromRef(pressderiv));
    int err = inverse.invert();
    if (err != 0)
      dserror("Inversion of matrix for pressure derivative failed with error code %d.", err);
  }

  // calculate derivatives of saturation w.r.t. pressure
  fluidmulti_mat_->EvaluateDerivOfSaturationWrtPressure(helpderiv, press);

  // chain rule: the derivative of saturation w.r.t. dof =
  // (derivative of saturation w.r.t. pressure) * (derivative of pressure w.r.t. dof)
  CORE::LINALG::multiply(satderiv, helpderiv, pressderiv);

  // compute derivative of solid pressure w.r.t. dofs with product rule
  for (int iphase = 0; iphase < numfluidphases; iphase++)
  {
    for (int jphase = 0; jphase < numfluidphases; jphase++)
      solidpressderiv[iphase] +=
          pressderiv(jphase, iphase) * sat[jphase] + satderiv(jphase, iphase) * press[jphase];
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::ComputeLinearizationOfSolPressWrtDisp(
    const double fluidpress, const double porosity, const int totalnumdofpernode,
    const int numfluidphases, const int numvolfrac, const std::vector<double>& phiAtGP,
    const CORE::LINALG::Matrix<1, numdof_>& dphi_dus, CORE::LINALG::Matrix<1, numdof_>& dps_dus)
{
  // get volume fraction primary variables
  std::vector<double> volfracphi(
      phiAtGP.data() + numfluidphases, phiAtGP.data() + numfluidphases + numvolfrac);
  double sumaddvolfrac = 0.0;
  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++) sumaddvolfrac += volfracphi[ivolfrac];

  // get volume fraction pressure at [numfluidphases+numvolfrac...totalnumdofpernode-1]
  std::vector<double> volfracpressure(
      phiAtGP.data() + numfluidphases + numvolfrac, phiAtGP.data() + totalnumdofpernode);

  // p_s = (porosity - sumaddvolfrac)/porosity * fluidpress
  //       + 1.0 / porosity sum_i=1^numvolfrac (volfrac_i*pressure_i)
  // d (p_s) / d porosity = + sumaddvolfrac/porosity/porosity * fluidpress
  double dps_dphi = sumaddvolfrac / (porosity * porosity) * fluidpress;

  // ... + 1.0 / porosity / porosity sum_i=1^numvolfrac (volfrac_i*pressure_i)
  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++)
    dps_dphi -= volfracphi[ivolfrac] * volfracpressure[ivolfrac] / (porosity * porosity);

  // d (p_s) / d u_s = d (p_s) / d porosity * d porosity / d u_s
  dps_dus.Update(dps_dphi, dphi_dus);
}

/*----------------------------------------------------------------------*
 * derivative of sol. pres. at GP for multiphase flow   kremheller 10/17|
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::RecalculateSolPressureDeriv(
    const std::vector<double>& phiAtGP, const int totalnumdofpernode, const int numfluidphases,
    const int numvolfrac, const double press, const double porosity,
    std::vector<double>& solidpressderiv)
{
  // get volume fraction primary variables
  std::vector<double> volfracphi(
      phiAtGP.data() + numfluidphases, phiAtGP.data() + numfluidphases + numvolfrac);
  double sumaddvolfrac = 0.0;
  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++) sumaddvolfrac += volfracphi[ivolfrac];

  // p_s = (porosity - sumaddvolfrac)/porosity * fluidpress
  //      + 1.0 / porosity sum_i=1^numvolfrac (volfrac_i*pressure_i)
  const double scale = (porosity - sumaddvolfrac) / porosity;

  // scale original fluid press deriv with (porosity - sumaddvolfrac)/porosity
  for (int iphase = 0; iphase < numfluidphases; iphase++) solidpressderiv[iphase] *= scale;

  // get volfrac pressures at [numfluidphases+numvolfrac...totalnumdofpernode-1]
  std::vector<double> volfracpressure(
      phiAtGP.data() + numfluidphases + numvolfrac, phiAtGP.data() + totalnumdofpernode);


  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++)
  {
    // d p_s / d volfrac = - fluidpress/porosity + volfracpressure/porosity
    solidpressderiv[ivolfrac + numfluidphases] =
        -1.0 / porosity * press + 1.0 / porosity * volfracpressure[ivolfrac];
    // d p_s / d volfracpress = + volfracphi/porosity
    solidpressderiv[ivolfrac + numfluidphases + numvolfrac] = volfracphi[ivolfrac] / porosity;
  }
}

template <CORE::FE::CellType distype>
double DRT::ELEMENTS::Wall1_Poro<distype>::ComputeSolPressureAtGP(
    const int totalnumdofpernode, const int numfluidphases, const std::vector<double>& phiAtGP)
{
  // initialize auxiliary variables
  std::vector<double> genpress(numfluidphases, 0.0);
  std::vector<double> sat(numfluidphases, 0.0);
  std::vector<double> press(numfluidphases, 0.0);
  std::vector<double> fluidphi(phiAtGP.data(), phiAtGP.data() + numfluidphases);

  // evaluate the pressures
  fluidmulti_mat_->EvaluateGenPressure(genpress, fluidphi);

  // transform generalized pressures to true pressure values
  fluidmulti_mat_->TransformGenPresToTruePres(genpress, press);

  // explicit evaluation of saturation
  fluidmulti_mat_->EvaluateSaturation(sat, fluidphi, press);

  // solid pressure = sum (S_i*p_i)
  const double solidpressure = std::inner_product(sat.begin(), sat.end(), press.begin(), 0.0);

  return solidpressure;
}

template <CORE::FE::CellType distype>
double DRT::ELEMENTS::Wall1_Poro<distype>::RecalculateSolPressureAtGP(double press,
    const double porosity, const int totalnumdofpernode, const int numfluidphases,
    const int numvolfrac, const std::vector<double>& phiAtGP)
{
  // get volume fraction primary variables at [numfluidphases-1...numfluidphase-1+numvolfrac]
  std::vector<double> volfracphi(
      phiAtGP.data() + numfluidphases, phiAtGP.data() + numfluidphases + numvolfrac);
  double sumaddvolfrac = 0.0;
  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++) sumaddvolfrac += volfracphi[ivolfrac];

  // p_s = (porosity - sumaddvolfrac)/porosity * fluidpress
  //      + 1.0 / porosity sum_i=1^numvolfrac (volfrac_i*pressure_i)
  // first part
  press *= (porosity - sumaddvolfrac) / porosity;

  // get volfrac pressures at [numfluidphases+numvolfrac...totalnumdofpernode-1]
  std::vector<double> volfracpressure(
      phiAtGP.data() + numfluidphases + numvolfrac, phiAtGP.data() + totalnumdofpernode);

  // second part
  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++)
    press += volfracphi[ivolfrac] / porosity * volfracpressure[ivolfrac];

  // note: in RecalculateSolidPressure in porofluid_phasemanager calculation is performed a bit
  //       differently since we already pass porosity = porosity - sumaddvolfrac, but result is
  //       equivalent

  return press;
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::ComputePrimaryVariableAtGP(const std::vector<double>& ephi,
    const int totalnumdofpernode, const CORE::LINALG::Matrix<numnod_, 1>& shapefct,
    std::vector<double>& phiAtGP)
{
  // zero out everything
  std::fill(phiAtGP.begin(), phiAtGP.end(), 0.0);
  // compute phi at GP = phi * shapefunction
  for (int i = 0; i < numnod_; i++)
  {
    for (int j = 0; j < totalnumdofpernode; j++)
    {
      phiAtGP[j] += shapefct(i) * ephi[i * totalnumdofpernode + j];
    }
  }
}

template <CORE::FE::CellType distype>
void DRT::ELEMENTS::Wall1_Poro<distype>::ExtractValuesFromGlobalVector(
    const DRT::Discretization& discretization, const int& dofset, const std::vector<int>& lm,
    CORE::LINALG::Matrix<numdim_, numnod_>* matrixtofill,
    CORE::LINALG::Matrix<numnod_, 1>* vectortofill, const std::string state)
{
  // put on higher level
  // get state of the global vector
  Teuchos::RCP<const Epetra_Vector> matrix_state = discretization.GetState(dofset, state);
  if (matrix_state == Teuchos::null) dserror("Cannot get state vector %s", state.c_str());

  const int numdofpernode = discretization.NumDof(dofset, Nodes()[0]);

  // extract local values of the global vectors
  std::vector<double> mymatrix(lm.size());
  DRT::UTILS::ExtractMyValues(*matrix_state, mymatrix, lm);

  if (numdofpernode == numdim_ + 1)
  {
    for (int inode = 0; inode < numnod_; ++inode)  // number of nodes
    {
      // fill a vector field via a pointer
      if (matrixtofill != nullptr)
      {
        for (int idim = 0; idim < numdim_; ++idim)  // number of dimensions
        {
          (*matrixtofill)(idim, inode) = mymatrix[idim + (inode * numdofpernode)];
        }
      }
      // fill a scalar field via a pointer
      if (vectortofill != nullptr)
        (*vectortofill)(inode, 0) = mymatrix[numdim_ + (inode * numdofpernode)];
    }
  }
  else if (numdofpernode == numdim_)
  {
    for (int inode = 0; inode < numnod_; ++inode)  // number of nodes
    {
      // fill a vector field via a pointer
      if (matrixtofill != nullptr)
      {
        for (int idim = 0; idim < numdim_; ++idim)  // number of dimensions
        {
          (*matrixtofill)(idim, inode) = mymatrix[idim + (inode * numdofpernode)];
        }
      }
    }
  }
  else if (numdofpernode == 1)
  {
    for (int inode = 0; inode < numnod_; ++inode)  // number of nodes
    {
      if (vectortofill != nullptr) (*vectortofill)(inode, 0) = mymatrix[inode * numdofpernode];
    }
  }
  else
  {
    for (int inode = 0; inode < numnod_; ++inode)  // number of nodes
    {
      if (vectortofill != nullptr) (*vectortofill)(inode, 0) = mymatrix[inode * numdofpernode];
    }
  }
}

template <CORE::FE::CellType distype>
std::vector<double> DRT::ELEMENTS::Wall1_Poro<distype>::ComputeAnisotropicPermeabilityCoeffsAtGP(
    const CORE::LINALG::Matrix<numnod_, 1>& shapefct) const
{
  std::vector<double> anisotropic_permeability_coeffs(numdim_, 0.0);

  for (int node = 0; node < numnod_; ++node)
  {
    const double shape_val = shapefct(node);
    for (int dim = 0; dim < numdim_; ++dim)
    {
      anisotropic_permeability_coeffs[dim] +=
          shape_val * anisotropic_permeability_nodal_coeffs_[dim][node];
    }
  }

  return anisotropic_permeability_coeffs;
}

template class DRT::ELEMENTS::Wall1_Poro<CORE::FE::CellType::tri3>;
template class DRT::ELEMENTS::Wall1_Poro<CORE::FE::CellType::quad4>;
template class DRT::ELEMENTS::Wall1_Poro<CORE::FE::CellType::quad9>;
template class DRT::ELEMENTS::Wall1_Poro<CORE::FE::CellType::nurbs4>;
template class DRT::ELEMENTS::Wall1_Poro<CORE::FE::CellType::nurbs9>;

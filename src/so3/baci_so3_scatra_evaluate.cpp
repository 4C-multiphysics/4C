/*----------------------------------------------------------------------*/
/*! \file

\brief Solid-scatra elements evaluate

\level 2


*----------------------------------------------------------------------*/

#include "baci_lib_utils.H"
#include "baci_mat_so3_material.H"
#include "baci_so3_element_service.H"
#include "baci_so3_scatra.H"
#include "baci_structure_new_enum_lists.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <class so3_ele, DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::So3_Scatra<so3_ele, distype>::PreEvaluate(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Element::LocationArray& la)
{
  if (la.Size() > 1)
  {
    // ask for the number of dofs of second dofset (scatra)
    const int numscal = discretization.NumDof(1, Nodes()[0]);

    if (la[1].Size() != numnod_ * numscal)
      dserror("So3_Scatra: PreEvaluate: Location vector length for concentrations does not match!");

    if (discretization.HasState(1, "scalarfield"))  // if concentrations were set
    {
      if (not(distype == DRT::Element::DiscretizationType::hex8 or
              distype == DRT::Element::DiscretizationType::hex27 or
              distype == DRT::Element::DiscretizationType::tet4 or
              distype == DRT::Element::DiscretizationType::tet10))
      {
        dserror(
            "The Solidscatra elements are only tested for the Hex8, Hex27, Tet4, and Tet10 case. "
            "The following should work, but keep your eyes open (especially with the order of the "
            "Gauss points)");
      }

      /* =========================================================================*/
      // start concentration business
      /* =========================================================================*/
      auto gpconc = Teuchos::rcp(
          new std::vector<std::vector<double>>(numgpt_, std::vector<double>(numscal, 0.0)));

      // check if you can get the scalar state
      Teuchos::RCP<const Epetra_Vector> concnp = discretization.GetState(1, "scalarfield");

      if (concnp == Teuchos::null)
        dserror("calc_struct_nlnstiff: Cannot get state vector 'scalarfield' ");

      // extract local values of the global vectors
      auto myconc = std::vector<double>(la[1].lm_.size(), 0.0);

      DRT::UTILS::ExtractMyValues(*concnp, myconc, la[1].lm_);

      // element vector for k-th scalar
      std::vector<CORE::LINALG::Matrix<numnod_, 1>> econc(numscal);
      for (int k = 0; k < numscal; ++k)
        for (int i = 0; i < numnod_; ++i) (econc.at(k))(i, 0) = myconc.at(numscal * i + k);

      /* =========================================================================*/
      /* ================================================= Loop over Gauss Points */
      /* =========================================================================*/
      // volume of current element in reference configuration
      double volume_ref = 0.0;
      // mass in current element in reference configuration
      std::vector<double> mass_ref(numscal, 0.0);

      for (int igp = 0; igp < numgpt_; ++igp)
      {
        // detJrefpar_wgp = det(dX/dr) * w_gp to calculate volume in reference configuration
        const double detJrefpar_wgp = detJ_[igp] * intpoints_.qwgt[igp];

        volume_ref += detJrefpar_wgp;

        // concentrations at current gauss point
        std::vector<double> conc_gp_k(numscal, 0.0);

        // shape functions evaluated at current gauss point
        CORE::LINALG::Matrix<numnod_, 1> shapefunct_gp(true);
        CORE::DRT::UTILS::shape_function<distype>(xsi_[igp], shapefunct_gp);

        for (int k = 0; k < numscal; ++k)
        {
          // identical shapefunctions for displacements and temperatures
          conc_gp_k.at(k) = shapefunct_gp.Dot(econc.at(k));

          mass_ref.at(k) += conc_gp_k.at(k) * detJrefpar_wgp;
        }

        gpconc->at(igp) = conc_gp_k;
      }

      params.set<Teuchos::RCP<std::vector<std::vector<double>>>>("gp_conc", gpconc);

      // compute average concentrations. Now mass_ref is the element averaged concentration
      for (int k = 0; k < numscal; ++k) mass_ref.at(k) /= volume_ref;

      auto avgconc = Teuchos::rcp(new std::vector<std::vector<double>>(numgpt_, mass_ref));

      params.set<Teuchos::RCP<std::vector<std::vector<double>>>>("avg_conc", avgconc);

    }  // if (discretization.HasState(1,"scalarfield"))

    // if temperatures were set
    if (discretization.NumDofSets() == 3)
    {
      if (discretization.HasState(2, "tempfield"))
      {
        if (not(distype == DRT::Element::DiscretizationType::hex8 or
                distype == DRT::Element::DiscretizationType::hex27 or
                distype == DRT::Element::DiscretizationType::tet4 or
                distype == DRT::Element::DiscretizationType::tet10))
        {
          dserror(
              "The Solidscatra elements are only tested for the Hex8, Hex27, Tet4, and Tet10 case. "
              "The following should work, but keep your eyes open (especially with the order of "
              "the Gauss points");
        }

        /* =========================================================================*/
        // start temperature business
        /* =========================================================================*/
        auto gptemp = Teuchos::rcp(new std::vector<double>(std::vector<double>(numgpt_, 0.0)));

        Teuchos::RCP<const Epetra_Vector> tempnp = discretization.GetState(2, "tempfield");

        if (tempnp == Teuchos::null)
          dserror("calc_struct_nlnstiff: Cannot get state vector 'tempfield' ");

        // extract local values of the global vectors
        auto mytemp = std::vector<double>(la[2].lm_.size(), 0.0);

        DRT::UTILS::ExtractMyValues(*tempnp, mytemp, la[2].lm_);

        // element vector for k-th scalar
        CORE::LINALG::Matrix<numnod_, 1> etemp;

        for (int i = 0; i < numnod_; ++i) etemp(i, 0) = mytemp.at(i);

        /* =========================================================================*/
        /* ================================================= Loop over Gauss Points */
        /* =========================================================================*/

        for (int igp = 0; igp < numgpt_; ++igp)
        {
          // shape functions evaluated at current gauss point
          CORE::LINALG::Matrix<numnod_, 1> shapefunct_gp(true);
          CORE::DRT::UTILS::shape_function<distype>(xsi_[igp], shapefunct_gp);

          // temperature at Gauss point withidentical shapefunctions for displacements and
          // temperatures
          gptemp->at(igp) = shapefunct_gp.Dot(etemp);
        }

        params.set<Teuchos::RCP<std::vector<double>>>("gp_temp", gptemp);
      }
    }

    // If you need a pointer to the scatra material, use these lines:
    // we assume that the second material of the structure is the scatra element material
    // Teuchos::RCP<MAT::Material> scatramat = so3_ele::Material(1);
    // params.set< Teuchos::RCP<MAT::Material> >("scatramat",scatramat);
  }

  // TODO: (thon) actually we do not want this here, since it has nothing to do with scatra specific
  // stuff. But for now we let it be...
  std::vector<double> center = DRT::UTILS::ElementCenterRefeCoords(this);
  auto xrefe = Teuchos::rcp(new std::vector<double>(center));
  params.set<Teuchos::RCP<std::vector<double>>>("position", xrefe);
}

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                                       |
 *----------------------------------------------------------------------*/
template <class so3_ele, DRT::Element::DiscretizationType distype>
int DRT::ELEMENTS::So3_Scatra<so3_ele, distype>::Evaluate(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Element::LocationArray& la,
    CORE::LINALG::SerialDenseMatrix& elemat1_epetra,
    CORE::LINALG::SerialDenseMatrix& elemat2_epetra,
    CORE::LINALG::SerialDenseVector& elevec1_epetra,
    CORE::LINALG::SerialDenseVector& elevec2_epetra,
    CORE::LINALG::SerialDenseVector& elevec3_epetra)
{
  // start with ActionType "none"
  typename So3_Scatra::ActionType act = So3_Scatra::none;

  // get the required action
  std::string action = params.get<std::string>("action", "none");

  // get the required action and safety check
  if (action == "none")
    dserror("No action supplied");
  else if (action == "calc_struct_stiffscalar")
    act = So3_Scatra::calc_struct_stiffscalar;

  // at the moment all cases need the PreEvaluate routine, since we always need the concentration
  // value at the gp
  PreEvaluate(params, discretization, la);

  // what action shall be performed
  switch (act)
  {
    // coupling terms K_dS of stiffness matrix K^{SSI} for monolithic SSI
    case So3_Scatra::calc_struct_stiffscalar:
    {
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState(0, "displacement");
      if (disp == Teuchos::null) dserror("Cannot get state vectors 'displacement'");

      // get my displacement vector
      std::vector<double> mydisp((la[0].lm_).size());
      DRT::UTILS::ExtractMyValues(*disp, mydisp, la[0].lm_);

      // calculate the stiffness matrix
      nln_kdS_ssi(la, mydisp, elemat1_epetra, params);

      break;
    }

    default:
    {
      // call the base class routine
      so3_ele::Evaluate(params, discretization, la[0].lm_, elemat1_epetra, elemat2_epetra,
          elevec1_epetra, elevec2_epetra, elevec3_epetra);
      break;
    }  // default
  }    // switch(act)

  return 0;
}  // Evaluate


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <class so3_ele, DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::So3_Scatra<so3_ele, distype>::GetCauchyNDirAndDerivativesAtXi(
    const CORE::LINALG::Matrix<3, 1>& xi, const std::vector<double>& disp_nodal_values,
    const std::vector<double>& scalar_nodal_values, const CORE::LINALG::Matrix<3, 1>& n,
    const CORE::LINALG::Matrix<3, 1>& dir, double& cauchy_n_dir,
    CORE::LINALG::SerialDenseMatrix* d_cauchyndir_dd,
    CORE::LINALG::SerialDenseMatrix* d_cauchyndir_ds, CORE::LINALG::Matrix<3, 1>* d_cauchyndir_dn,
    CORE::LINALG::Matrix<3, 1>* d_cauchyndir_ddir, CORE::LINALG::Matrix<3, 1>* d_cauchyndir_dxi)
{
  auto scalar_values_at_xi =
      DRT::ELEMENTS::ProjectNodalQuantityToXi<distype>(xi, scalar_nodal_values);
  double d_cauchyndir_ds_gp(0.0);
  // call base class
  so3_ele::GetCauchyNDirAndDerivativesAtXi(xi, disp_nodal_values, n, dir, cauchy_n_dir,
      d_cauchyndir_dd, nullptr, nullptr, nullptr, nullptr, d_cauchyndir_dn, d_cauchyndir_ddir,
      d_cauchyndir_dxi, nullptr, nullptr, nullptr, scalar_values_at_xi.data(), &d_cauchyndir_ds_gp);

  if (d_cauchyndir_ds != nullptr)
  {
    d_cauchyndir_ds->shape(numnod_, 1);
    // get the shape functions
    CORE::LINALG::Matrix<numnod_, 1> shapefunct(true);
    CORE::DRT::UTILS::shape_function<distype>(xi, shapefunct);
    // calculate DsntDs
    CORE::LINALG::Matrix<numnod_, 1>(d_cauchyndir_ds->values(), true)
        .Update(d_cauchyndir_ds_gp, shapefunct, 1.0);
  }
}

/*----------------------------------------------------------------------*
 | evaluate only the mechanical-scatra stiffness term     schmidt 10/17 |
 | for monolithic SSI, contribution to k_dS (private)                   |
 *----------------------------------------------------------------------*/
template <class so3_ele, DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::So3_Scatra<so3_ele, distype>::nln_kdS_ssi(DRT::Element::LocationArray& la,
    std::vector<double>& disp,                         // current displacement
    CORE::LINALG::SerialDenseMatrix& stiffmatrix_kdS,  // (numdim_*numnod_ ; numnod_)
    Teuchos::ParameterList& params)
{
  // calculate current and material coordinates of element
  CORE::LINALG::Matrix<numnod_, numdim_> xrefe(true);  // X, material coord. of element
  CORE::LINALG::Matrix<numnod_, numdim_> xcurr(true);  // x, current  coord. of element
  DRT::Node** nodes = Nodes();
  for (int i = 0; i < numnod_; ++i)
  {
    const double* x = nodes[i]->X();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];

    xcurr(i, 0) = xrefe(i, 0) + disp[i * numdofpernode_ + 0];
    xcurr(i, 1) = xrefe(i, 1) + disp[i * numdofpernode_ + 1];
    xcurr(i, 2) = xrefe(i, 2) + disp[i * numdofpernode_ + 2];
  }

  // shape functions and their first derivatives
  CORE::LINALG::Matrix<numnod_, 1> shapefunct(true);
  CORE::LINALG::Matrix<numdim_, numnod_> deriv(true);
  // compute derivatives N_XYZ at gp w.r.t. material coordinates
  CORE::LINALG::Matrix<numdim_, numnod_> N_XYZ(true);
  // compute deformation gradient w.r.t. to material configuration
  CORE::LINALG::Matrix<numdim_, numdim_> defgrad(true);

  // evaluation of linearization w.r.t. certain primary variable
  const int differentiationtype =
      params.get<int>("differentiationtype", static_cast<int>(STR::DifferentiationType::none));
  if (differentiationtype == static_cast<int>(STR::DifferentiationType::none))
    dserror("Cannot get differentation type");

  // get numscatradofspernode from parameter list in case of elch linearizations
  int numscatradofspernode(-1);
  if (differentiationtype == static_cast<int>(STR::DifferentiationType::elch))
  {
    numscatradofspernode = params.get<int>("numscatradofspernode", -1);
    if (numscatradofspernode == -1)
      dserror("Could not read 'numscatradofspernode' from parameter list!");
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // get shape functions and their derivatives
    CORE::DRT::UTILS::shape_function<distype>(xsi_[gp], shapefunct);
    CORE::DRT::UTILS::shape_function_deriv1<distype>(xsi_[gp], deriv);

    // compute derivatives N_XYZ at gp w.r.t. material coordinates
    // by N_XYZ = J^-1 . N_rst
    N_XYZ.Multiply(invJ_[gp], deriv);

    // (material) deformation gradient
    // F = d xcurr / d xrefe = xcurr^T . N_XYZ^T
    defgrad.MultiplyTT(xcurr, N_XYZ);

    // right Cauchy-Green tensor = F^T . F
    CORE::LINALG::Matrix<3, 3> cauchygreen;
    cauchygreen.MultiplyTN(defgrad, defgrad);

    // calculate vector of right Cauchy-Green tensor
    CORE::LINALG::Matrix<numstr_, 1> cauchygreenvec;
    cauchygreenvec(0) = cauchygreen(0, 0);
    cauchygreenvec(1) = cauchygreen(1, 1);
    cauchygreenvec(2) = cauchygreen(2, 2);
    cauchygreenvec(3) = 2 * cauchygreen(0, 1);
    cauchygreenvec(4) = 2 * cauchygreen(1, 2);
    cauchygreenvec(5) = 2 * cauchygreen(2, 0);

    // Green Lagrange strain
    CORE::LINALG::Matrix<numstr_, 1> glstrain;
    // Green-Lagrange strain matrix E = 0.5 * (Cauchygreen - Identity)
    glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
    glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
    glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
    glstrain(3) = cauchygreen(0, 1);
    glstrain(4) = cauchygreen(1, 2);
    glstrain(5) = cauchygreen(2, 0);

    // calculate nonlinear B-operator
    CORE::LINALG::Matrix<numstr_, numdofperelement_> bop(true);
    CalculateBop(&bop, &defgrad, &N_XYZ);

    /*==== call material law ======================================================*/
    // init derivative of second Piola-Kirchhoff stresses w.r.t. concentrations dSdc
    CORE::LINALG::Matrix<numstr_, 1> dSdc(true);

    // get dSdc, hand in nullptr as 'cmat' to evaluate the off-diagonal block
    Teuchos::RCP<MAT::So3Material> so3mat = Teuchos::rcp_static_cast<MAT::So3Material>(Material());
    so3mat->Evaluate(&defgrad, &glstrain, params, &dSdc, nullptr, gp, Id());

    /*==== end of call material law ===============================================*/

    // k_dS = B^T . dS/dc * detJ * N * w(gp)
    const double detJ_w = detJ_[gp] * intpoints_.qwgt[gp];
    CORE::LINALG::Matrix<numdofperelement_, 1> BdSdc(true);
    BdSdc.MultiplyTN(detJ_w, bop, dSdc);

    // loop over rows
    for (int rowi = 0; rowi < numdofperelement_; ++rowi)
    {
      const double BdSdc_rowi = BdSdc(rowi, 0);
      // loop over columns
      for (int coli = 0; coli < numnod_; ++coli)
      {
        // stiffness matrix w.r.t. elch dofs
        if (differentiationtype == static_cast<int>(STR::DifferentiationType::elch))
          stiffmatrix_kdS(rowi, coli * numscatradofspernode) += BdSdc_rowi * shapefunct(coli, 0);
        else if (differentiationtype == static_cast<int>(STR::DifferentiationType::temp))
          stiffmatrix_kdS(rowi, coli) += BdSdc_rowi * shapefunct(coli, 0);
        else
          dserror("Unknown differentation type");
      }
    }
  }  // gauss point loop
}  // nln_kdS_ssi


/*----------------------------------------------------------------------*
 | calculate the nonlinear B-operator (private)           schmidt 10/17 |
 *----------------------------------------------------------------------*/
template <class so3_ele, DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::So3_Scatra<so3_ele, distype>::CalculateBop(
    CORE::LINALG::Matrix<numstr_, numdofperelement_>* bop,  //!< (o): nonlinear B-operator
    const CORE::LINALG::Matrix<numdim_, numdim_>* defgrad,  //!< (i): deformation gradient
    const CORE::LINALG::Matrix<numdim_, numnod_>* N_XYZ)
    const  //!< (i): (material) derivative of shape functions
{
  // calc bop matrix if provided
  if (bop != nullptr)
  {
    /* non-linear B-operator (may so be called, meaning of B-operator is not so
    **  sharp in the non-linear realm) *
    **   B = F^{i,T} . B_L *
    ** with linear B-operator B_L =  N_XYZ (6x24) = (3x8)
    **
    **   B    =   F^T  . N_XYZ
    ** (6x24)    (3x3)   (3x8)
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
      (*bop)(0, numdofpernode_ * i + 0) = (*defgrad)(0, 0) * (*N_XYZ)(0, i);
      (*bop)(0, numdofpernode_ * i + 1) = (*defgrad)(1, 0) * (*N_XYZ)(0, i);
      (*bop)(0, numdofpernode_ * i + 2) = (*defgrad)(2, 0) * (*N_XYZ)(0, i);
      (*bop)(1, numdofpernode_ * i + 0) = (*defgrad)(0, 1) * (*N_XYZ)(1, i);
      (*bop)(1, numdofpernode_ * i + 1) = (*defgrad)(1, 1) * (*N_XYZ)(1, i);
      (*bop)(1, numdofpernode_ * i + 2) = (*defgrad)(2, 1) * (*N_XYZ)(1, i);
      (*bop)(2, numdofpernode_ * i + 0) = (*defgrad)(0, 2) * (*N_XYZ)(2, i);
      (*bop)(2, numdofpernode_ * i + 1) = (*defgrad)(1, 2) * (*N_XYZ)(2, i);
      (*bop)(2, numdofpernode_ * i + 2) = (*defgrad)(2, 2) * (*N_XYZ)(2, i);
      /* ~~~ */
      (*bop)(3, numdofpernode_ * i + 0) =
          (*defgrad)(0, 0) * (*N_XYZ)(1, i) + (*defgrad)(0, 1) * (*N_XYZ)(0, i);
      (*bop)(3, numdofpernode_ * i + 1) =
          (*defgrad)(1, 0) * (*N_XYZ)(1, i) + (*defgrad)(1, 1) * (*N_XYZ)(0, i);
      (*bop)(3, numdofpernode_ * i + 2) =
          (*defgrad)(2, 0) * (*N_XYZ)(1, i) + (*defgrad)(2, 1) * (*N_XYZ)(0, i);
      (*bop)(4, numdofpernode_ * i + 0) =
          (*defgrad)(0, 1) * (*N_XYZ)(2, i) + (*defgrad)(0, 2) * (*N_XYZ)(1, i);
      (*bop)(4, numdofpernode_ * i + 1) =
          (*defgrad)(1, 1) * (*N_XYZ)(2, i) + (*defgrad)(1, 2) * (*N_XYZ)(1, i);
      (*bop)(4, numdofpernode_ * i + 2) =
          (*defgrad)(2, 1) * (*N_XYZ)(2, i) + (*defgrad)(2, 2) * (*N_XYZ)(1, i);
      (*bop)(5, numdofpernode_ * i + 0) =
          (*defgrad)(0, 2) * (*N_XYZ)(0, i) + (*defgrad)(0, 0) * (*N_XYZ)(2, i);
      (*bop)(5, numdofpernode_ * i + 1) =
          (*defgrad)(1, 2) * (*N_XYZ)(0, i) + (*defgrad)(1, 0) * (*N_XYZ)(2, i);
      (*bop)(5, numdofpernode_ * i + 2) =
          (*defgrad)(2, 2) * (*N_XYZ)(0, i) + (*defgrad)(2, 0) * (*N_XYZ)(2, i);
    }
  }
}  // CalculateBop


/*----------------------------------------------------------------------*
 | initialize element (private)                            schmidt 10/17|
 *----------------------------------------------------------------------*/
template <class so3_ele, DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::So3_Scatra<so3_ele, distype>::InitElement()
{
  // resize gauss point coordinates, inverse of the jacobian and determinant of the jacobian
  xsi_.resize(numgpt_);
  invJ_.resize(numgpt_);
  detJ_.resize(numgpt_);

  // calculate coordinates in reference (material) configuration
  CORE::LINALG::Matrix<numnod_, numdim_> xrefe;
  for (int i = 0; i < numnod_; ++i)
  {
    Node** nodes = Nodes();
    if (!nodes) dserror("Nodes() returned null pointer");
    xrefe(i, 0) = Nodes()[i]->X()[0];
    xrefe(i, 1) = Nodes()[i]->X()[1];
    xrefe(i, 2) = Nodes()[i]->X()[2];
  }

  // calculate gauss point coordinates, the inverse jacobian and the determinant of the jacobian
  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // gauss point coordinates
    const double* gpcoord = intpoints_.Point(gp);
    for (int idim = 0; idim < numdim_; idim++) xsi_[gp](idim) = gpcoord[idim];

    // get derivative of shape functions w.r.t. parameter coordinates, needed for calculation of the
    // inverse of the jacobian
    CORE::LINALG::Matrix<numdim_, numnod_> deriv;
    CORE::DRT::UTILS::shape_function_deriv1<distype>(xsi_[gp], deriv);

    // get the inverse of the Jacobian matrix which looks like:
    /*
                 [ X_,r  Y_,r  Z_,r ]^-1
          J^-1 = [ X_,s  Y_,s  Z_,s ]
                 [ X_,t  Y_,t  Z_,t ]
     */

    invJ_[gp].Multiply(deriv, xrefe);
    // here Jacobian is inverted and det(J) is calculated
    detJ_[gp] = invJ_[gp].Invert();

    // make sure determinant of jacobian is positive
    if (detJ_[gp] <= 0.0) dserror("Element Jacobian mapping %10.5e <= 0.0", detJ_[gp]);
  }
}


template class DRT::ELEMENTS::So3_Scatra<DRT::ELEMENTS::So_hex8,
    DRT::Element::DiscretizationType::hex8>;
template class DRT::ELEMENTS::So3_Scatra<DRT::ELEMENTS::So_hex27,
    DRT::Element::DiscretizationType::hex27>;
template class DRT::ELEMENTS::So3_Scatra<DRT::ELEMENTS::So_hex8fbar,
    DRT::Element::DiscretizationType::hex8>;
template class DRT::ELEMENTS::So3_Scatra<DRT::ELEMENTS::So_tet4,
    DRT::Element::DiscretizationType::tet4>;
template class DRT::ELEMENTS::So3_Scatra<DRT::ELEMENTS::So_tet10,
    DRT::Element::DiscretizationType::tet10>;
template class DRT::ELEMENTS::So3_Scatra<DRT::ELEMENTS::So_weg6,
    DRT::Element::DiscretizationType::wedge6>;

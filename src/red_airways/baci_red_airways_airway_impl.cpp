/*---------------------------------------------------------------------*/
/*! \file

\brief Internal implementation of RedAirway element. Methods implemented here
       are called by airway_evaluate.cpp by DRT::ELEMENTS::RedAirway::Evaluate()
       with the corresponding action.


\level 3

*/
/*---------------------------------------------------------------------*/



#include "baci_red_airways_airway_impl.H"

#include "baci_discretization_fem_general_utils_fem_shapefunctions.H"
#include "baci_lib_discret.H"
#include "baci_lib_globalproblem.H"
#include "baci_lib_utils.H"
#include "baci_mat_air_0d_O2_saturation.H"
#include "baci_mat_hemoglobin_0d_O2_saturation.H"
#include "baci_mat_list.H"
#include "baci_mat_newtonianfluid.H"
#include "baci_mat_par_bundle.H"
#include "baci_red_airways_elem_params.h"
#include "baci_red_airways_evaluation_data.h"
#include "baci_utils_function.H"
#include "baci_utils_function_of_time.H"

#include <fstream>
#include <iomanip>


namespace
{
  /*----------------------------------------------------------------------*
  |  get element length                                      ismail 08/13|
  |                                                                      |
  *----------------------------------------------------------------------*/
  template <CORE::FE::CellType distype>
  double GetElementLength(DRT::ELEMENTS::RedAirway* ele)
  {
    double length = 0.0;
    // get node coordinates and number of elements per node
    static const int numnode = CORE::DRT::UTILS::DisTypeToNumNodePerEle<distype>::numNodePerElement;
    DRT::Node** nodes = ele->Nodes();
    // get airway length
    CORE::LINALG::Matrix<3, numnode> xyze;
    for (int inode = 0; inode < numnode; inode++)
    {
      const double* x = nodes[inode]->X();
      xyze(0, inode) = x[0];
      xyze(1, inode) = x[1];
      xyze(2, inode) = x[2];
    }
    // Calculate the length of airway element
    length = sqrt(pow(xyze(0, 0) - xyze(0, 1), 2) + pow(xyze(1, 0) - xyze(1, 1), 2) +
                  pow(xyze(2, 0) - xyze(2, 1), 2));
    // get airway area

    return length;
  }

  /*----------------------------------------------------------------------*
  |  calculate curve value a node with a certain BC          ismail 06/13|
  |                                                                      |
  *----------------------------------------------------------------------*/
  template <CORE::FE::CellType distype>
  bool GetCurveValAtCond(double& bcVal, DRT::Node* node, std::string condName,
      std::string optionName, std::string condType, double time)
  {
    // initialize bc value
    bcVal = 0.0;

    // check if node exists
    if (!node)
    {
      // return BC doesn't exist
      return false;
    }

    // check if condition exists
    if (node->GetCondition(condName))
    {
      DRT::Condition* condition = node->GetCondition(condName);
      // Get the type of prescribed bc
      std::string Bc = *(condition->Get<std::string>(optionName));
      if (Bc == condType)
      {
        const std::vector<int>* curve = condition->Get<std::vector<int>>("curve");
        double curvefac = 1.0;
        const std::vector<double>* vals = condition->Get<std::vector<double>>("val");

        // -----------------------------------------------------------------
        // Read in the value of the applied BC
        //  Val = curve1*val1 + curve2*func
        // -----------------------------------------------------------------
        // get curve1 and val1
        int curvenum = -1;
        if (curve) curvenum = (*curve)[0];
        if (curvenum >= 0)
          curvefac = DRT::Problem::Instance()
                         ->FunctionById<CORE::UTILS::FunctionOfTime>(curvenum)
                         .Evaluate(time);

        bcVal = (*vals)[0] * curvefac;

        // get funct 1
        const std::vector<int>* functions = condition->Get<std::vector<int>>("funct");
        int functnum = -1;
        if (functions)
          functnum = (*functions)[0];
        else
          functnum = -1;

        double functionfac = 0.0;
        if (functnum > 0)
        {
          functionfac = DRT::Problem::Instance()
                            ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>(functnum - 1)
                            .Evaluate(node->X(), time, 0);
        }
        // get curve2
        int curve2num = -1;
        double curve2fac = 1.0;
        if (curve) curve2num = (*curve)[1];
        if (curve2num >= 0)
          curve2fac = DRT::Problem::Instance()
                          ->FunctionById<CORE::UTILS::FunctionOfTime>(curve2num)
                          .Evaluate(time);

        bcVal += functionfac * curve2fac;

        // return BC exists
        return true;
      }
    }
    // return BC doesn't exist
    return false;
  }

  /*!
  \brief calculate element matrix and rhs

  \param ele              (i) the element those matrix is calculated
  \param eqnp             (i) nodal volumetric flow rate at n+1
  \param evelnp           (i) nodal velocity at n+1
  \param eareanp          (i) nodal cross-sectional area at n+1
  \param eprenp           (i) nodal pressure at n+1
  \param estif            (o) element matrix to calculate
  \param eforce           (o) element rhs to calculate
  \param material         (i) airway material/dimesion
  \param time             (i) current simulation time
  \param dt               (i) timestep
  \param compute_awacinter(i) computing airway-acinus interdependency
  */
  template <CORE::FE::CellType distype>
  void Sysmat(DRT::ELEMENTS::RedAirway* ele, CORE::LINALG::SerialDenseVector& epnp,
      CORE::LINALG::SerialDenseVector& epn, CORE::LINALG::SerialDenseVector& epnm,
      CORE::LINALG::SerialDenseMatrix& sysmat, CORE::LINALG::SerialDenseVector& rhs,
      Teuchos::RCP<const MAT::Material> material, DRT::REDAIRWAYS::ElemParams& params, double time,
      double dt, bool compute_awacinter)
  {
    double dens = 0.0;
    double visc = 0.0;

    if (material->MaterialType() == INPAR::MAT::m_fluid)
    {
      // get actual material
      const MAT::NewtonianFluid* actmat = static_cast<const MAT::NewtonianFluid*>(material.get());

      // get density
      dens = actmat->Density();

      // get dynamic viscosity
      visc = actmat->Viscosity();
    }
    else
    {
      dserror("Material law is not a Newtonian fluid");
      exit(1);
    }

    rhs.putScalar(0.0);
    sysmat.putScalar(0.0);

    // Calculate the length of airway element
    const double L = GetElementLength<distype>(ele);

    double qout_n = params.qout_n;
    double qout_np = params.qout_np;
    double qin_n = params.qin_n;
    double qin_np = params.qin_np;

    // get the generation number
    int generation = 0;
    ele->getParams("Generation", generation);

    double R = -1.0;

    // get element information
    double Ao = 0.0;
    double A = 0.0;
    double velPow = 0.0;

    ele->getParams("Area", Ao);
    A = Ao;
    ele->getParams("PowerOfVelocityProfile", velPow);

    if (ele->ElemSolvingType() == "Linear")
    {
      A = Ao;
    }
    else if (ele->ElemSolvingType() == "NonLinear")
    {
      A = params.volnp / L;
    }
    else
    {
      dserror("[%s] is not a defined ElemSolvingType of a RED_AIRWAY element",
          ele->ElemSolvingType().c_str());
    }

    // Get airway branch length
    double l_branch = 0.0;
    ele->getParams("BranchLength", l_branch);

    if (l_branch < 0.0) l_branch = L;

    // evaluate Poiseuille resistance
    double Rp = 2.0 * (2.0 + velPow) * M_PI * visc * L / (pow(A, 2));

    // evaluate the Reynolds number
    const double Re = 2.0 * fabs(qout_np) / (visc / dens * sqrt(A * M_PI));

    if (ele->Resistance() == "Poiseuille")
    {
      R = Rp;
    }
    else if (ele->Resistance() == "Pedley")
    {
      //-----------------------------------------------------------------
      // resistance evaluated using Pedley's model from :
      // Pedley et al (1970)
      //-----------------------------------------------------------------
      double gamma = 0.327;
      R = gamma * (sqrt(Re * 2.0 * sqrt(A / M_PI) / l_branch)) * Rp;

      //-----------------------------------------------------------------
      // Correct any resistance smaller than Poiseuille's one
      //-----------------------------------------------------------------
      //    if (R < Rp)
      //    {
      //      R = Rp;
      //    }
      double alfa = sqrt(2.0 * sqrt(A / M_PI) / l_branch);

      double Rep = 1.0 / ((gamma * alfa) * (gamma * alfa));
      double k = 0.50;
      double st = 1.0 / (1.0 + exp(-2 * k * (Re - Rep)));

      R = R * st + Rp * (1.0 - st);
    }
    else if (ele->Resistance() == "Generation_Dependent_Pedley")
    {
      //-----------------------------------------------------------------
      // Gamma is taken from Ertbruggen et al
      //-----------------------------------------------------------------
      double gamma = 0.327;
      switch (generation)
      {
        case 0:
          gamma = 0.162;
          break;
        case 1:
          gamma = 0.239;
          break;
        case 2:
          gamma = 0.244;
          break;
        case 3:
          gamma = 0.295;
          break;
        case 4:
          gamma = 0.175;
          break;
        case 5:
          gamma = 0.303;
          break;
        case 6:
          gamma = 0.356;
          break;
        case 7:
          gamma = 0.566;
          break;
        default:
          gamma = 0.327;
          break;
      }
      //-----------------------------------------------------------------
      // resistance evaluated using Pedley's model from :
      // Pedley et al (1970)
      //-----------------------------------------------------------------
      R = gamma * (sqrt(Re * 2.0 * sqrt(A / M_PI) / l_branch)) * Rp;

      //-----------------------------------------------------------------
      // Correct any resistance smaller than Poiseuille's one
      //-----------------------------------------------------------------
      if (R < Rp)
      {
        R = Rp;
      }
    }
    else if (ele->Resistance() == "Cont_Pedley")
    {
      //-----------------------------------------------------------------
      // resistance evaluated using Pedley's model from :
      // Pedley et al (1970)
      //-----------------------------------------------------------------
      double gamma = 0.327;
      double D = sqrt(A / M_PI) * 2.0;
      double Rel = (l_branch / D) * (1.0 / (gamma * gamma));
      double lambda = 1.2;
      double Ret = lambda * Rel;

      //-----------------------------------------------------------------
      // Correct any resistance smaller than Poiseuille's one
      //-----------------------------------------------------------------
      if (Re >= Ret)
        R = gamma * (sqrt(Re * 2.0 * sqrt(A / M_PI) / l_branch)) * Rp;
      else
      {
        double St = gamma * sqrt((D / l_branch) * Ret);
        double bRe = 2.0 * St / (St - 1.0);
        double aRe = (St - 1.0) / pow(Ret, bRe);
        R = (aRe * pow(Re, bRe) + 1.0) * Rp;
      }
    }
    else if (ele->Resistance() == "Generation_Dependent_Cont_Pedley")
    {
      //-----------------------------------------------------------------
      // Gamma is taken from Ertbruggen et al
      //-----------------------------------------------------------------
      double gamma = 0.327;
      switch (generation)
      {
        case 0:
          gamma = 0.162;
          break;
        case 1:
          gamma = 0.239;
          break;
        case 2:
          gamma = 0.244;
          break;
        case 3:
          gamma = 0.295;
          break;
        case 4:
          gamma = 0.175;
          break;
        case 5:
          gamma = 0.303;
          break;
        case 6:
          gamma = 0.356;
          break;
        case 7:
          gamma = 0.566;
          break;
        default:
          gamma = 0.327;
          break;
      }
      //-----------------------------------------------------------------
      // resistance evaluated using Pedley's model from :
      // Pedley et al (1970)
      //-----------------------------------------------------------------
      double D = sqrt(A / M_PI) * 2.0;
      double Rel = (l_branch / D) * (1.0 / (gamma * gamma));
      double lambda = 1.2;
      double Ret = lambda * Rel;

      //-----------------------------------------------------------------
      // Correct any resistance smaller than Poiseuille's one
      //-----------------------------------------------------------------
      if (Re >= Ret)
        R = gamma * (sqrt(Re * 2.0 * sqrt(A / M_PI) / l_branch)) * Rp;
      else
      {
        double St = gamma * sqrt((D / l_branch) * Ret);
        double bRe = 2.0 * St / (St - 1.0);
        double aRe = (St - 1.0) / pow(Ret, bRe);
        R = (aRe * pow(Re, bRe) + 1.0) * Rp;
      }
    }
    else if (ele->Resistance() == "Reynolds")
    {
      R = Rp * (3.4 + 2.1e-3 * Re);
    }
    else
    {
      dserror("[%s] is not a defined resistance model", ele->Resistance().c_str());
    }

    //------------------------------------------------------------
    // Set high resistance for collapsed airway
    //------------------------------------------------------------
    double airwayColl = 0;
    ele->getParams("AirwayColl", airwayColl);

    if (airwayColl == 1)
    {
      double opennp = params.open;
      if (opennp == 0)
      {
        // R = 10000000000;
        R = 10000000;  // 000 before: 10^10, Bates: 10^8
      }
    }

    //------------------------------------------------------------
    // get airway compliance
    //------------------------------------------------------------
    double Ew, tw, nu;
    // Get element compliance
    ele->getParams("WallElasticity", Ew);
    ele->getParams("WallThickness", tw);
    ele->getParams("PoissonsRatio", nu);
    double C = 0.0;
    double Ec = 0.0;
    Ec = (Ew * tw * sqrt(M_PI)) / ((1.0 - nu * nu) * 2.0 * sqrt(A) * Ao * L);
    if (Ec != 0.0)
    {
      C = 1.0 / Ec;
    }

    //------------------------------------------------------------
    // get airway viscous resistance
    //------------------------------------------------------------
    double Ts, phis;
    // define 0D airway components
    ele->getParams("ViscousPhaseShift", phis);
    ele->getParams("ViscousTs", Ts);
    double gammas = Ts * tan(phis) * (Ew * tw * sqrt(M_PI) / (1.0 - nu * nu)) / (4.0 * M_PI);
    double Rvis = gammas / (Ao * sqrt(Ao) * L);

    //------------------------------------------------------------
    // get airway inductance
    //------------------------------------------------------------
    double I = dens * L / Ao;

    //------------------------------------------------------------
    // get airway convective resistance
    //------------------------------------------------------------
    // get Poiseuille resistance with parabolic profile
    double Rp2nd = 2.0 * (2.0 + 2.0) * M_PI * visc * L / (pow(A, 2));
    // get the power of velocity profile for the currently used resistance
    double gamma = 4.0 / (Rp2nd / R) - 2.0;
    // get the Coriolis coefficient
    double alpha = (2.0 + gamma) / (1.0 + gamma);
    double Rconv = 2.0 * alpha * dens * (qout_np - qin_np) / (A * A);

    //------------------------------------------------------------
    // get airway external pressure
    //------------------------------------------------------------
    double pextn = 0.0;
    double pextnp = 0.0;

    // loop over all nodes
    // pext is the average pressure over the nodes
    for (int i = 0; i < ele->NumNode(); i++)
    {
      double pextVal = 0.0;
      // get Pext at time step n
      GetCurveValAtCond<distype>(pextVal, ele->Nodes()[i], "RedAirwayPrescribedExternalPressure",
          "boundarycond", "ExternalPressure", time - dt);
      pextn += pextVal / double(ele->NumNode());

      // get Pext at time step n+1e
      GetCurveValAtCond<distype>(pextVal, ele->Nodes()[i], "RedAirwayPrescribedExternalPressure",
          "boundarycond", "ExternalPressure", time);
      pextnp += pextVal / double(ele->NumNode());
    }

    // Routine to compute pextnp andd pextn from neighbourung acinus pressure
    // ComputePext() analog zu EvaluateCollapse()
    // bool compute_awacinter = params.get<bool>("compute_awacinter");
    if (compute_awacinter)
    {
      pextn = params.p_extn;
      pextnp = params.p_extnp;
    }

    if (ele->Type() == "Resistive")
    {
      C = 0.0;
      I = 0.0;
      Rconv = 0.0;
      Rvis = 0.0;
    }
    else if (ele->Type() == "InductoResistive")
    {
      C = 0.0;
      Rconv = 0.0;
      Rvis = 0.0;
    }
    else if (ele->Type() == "ComplientResistive")
    {
      I = 0.0;
      Rconv = 0.0;
      Rvis = 0.0;
    }
    else if (ele->Type() == "RLC")
    {
      Rconv = 0.0;
      Rvis = 0.0;
    }
    else if (ele->Type() == "ViscoElasticRLC")
    {
      Rconv = 0.0;
    }
    else if (ele->Type() == "ConvectiveViscoElasticRLC")
    {
    }
    else
    {
      dserror("[%s] is not an implemented element yet", (ele->Type()).c_str());
      exit(1);
    }

    double Ainv = -0.5 * C / (dt + Rvis * C);
    double B = 0.5 * I / dt + 0.5 * (Rconv + R);
    double P1 = epn(0) + epn(1) - 2.0 * Rvis * (qin_n - qout_n) + 2.0 * (pextnp - pextn);
    double P2 = -I * (qin_n + qout_n) / (2.0 * dt);

    sysmat(0, 0) = 0.5 * Ainv - 0.5 / B;
    sysmat(0, 1) = 0.5 * Ainv + 0.5 / B;
    sysmat(1, 0) = 0.5 * Ainv + 0.5 / B;
    sysmat(1, 1) = 0.5 * Ainv - 0.5 / B;

    rhs(0) = 0.5 * (P1 * Ainv - P2 / B);
    rhs(1) = 0.5 * (P1 * Ainv + P2 / B);

    // If airway is collapsed, set pressure equal in the downstream airway to
    // force zero flow downstream of the collapse
    if (airwayColl == 1)
    {
      double opennp = params.open;

      if (opennp == 0)
      {
        sysmat(1, 0) = 0;
        sysmat(1, 1) = 0;
        // rhs(0) = 0;
        rhs(1) = 0;
      }
    }
  }
}  // namespace

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::RedAirwayImplInterface* DRT::ELEMENTS::RedAirwayImplInterface::Impl(
    DRT::ELEMENTS::RedAirway* red_airway)
{
  switch (red_airway->Shape())
  {
    case CORE::FE::CellType::line2:
    {
      static AirwayImpl<CORE::FE::CellType::line2>* airway;
      if (airway == nullptr)
      {
        airway = new AirwayImpl<CORE::FE::CellType::line2>;
      }
      return airway;
    }
    default:
      dserror("shape %d (%d nodes) not supported", red_airway->Shape(), red_airway->NumNode());
      break;
  }
  return nullptr;
}


/*----------------------------------------------------------------------*
 | evaluate (public)                                       ismail 01/10 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
int DRT::ELEMENTS::AirwayImpl<distype>::Evaluate(RedAirway* ele, Teuchos::ParameterList& params,
    DRT::Discretization& discretization, std::vector<int>& lm,
    CORE::LINALG::SerialDenseMatrix& elemat1_epetra,
    CORE::LINALG::SerialDenseMatrix& elemat2_epetra,
    CORE::LINALG::SerialDenseVector& elevec1_epetra,
    CORE::LINALG::SerialDenseVector& elevec2_epetra,
    CORE::LINALG::SerialDenseVector& elevec3_epetra, Teuchos::RCP<MAT::Material> mat)
{
  const int elemVecdim = elevec1_epetra.length();

  std::vector<int>::iterator it_vcr;

  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  //----------------------------------------------------------------------
  // get control parameters for time integration
  //----------------------------------------------------------------------
  // get time-step size
  const double dt = evaluation_data.dt;
  // get time
  const double time = evaluation_data.time;

  // ---------------------------------------------------------------------
  // get control parameters for stabilization and higher-order elements
  //----------------------------------------------------------------------
  // flag for higher order elements
  // bool higher_order_ele = ele->isHigherOrderElement(distype);

  // ---------------------------------------------------------------------
  // get all general state vectors: flow, pressure,
  // ---------------------------------------------------------------------

  Teuchos::RCP<const Epetra_Vector> pnp = discretization.GetState("pnp");
  Teuchos::RCP<const Epetra_Vector> pn = discretization.GetState("pn");
  Teuchos::RCP<const Epetra_Vector> pnm = discretization.GetState("pnm");

  if (pnp == Teuchos::null || pn == Teuchos::null || pnm == Teuchos::null)
    dserror("Cannot get state vectors 'pnp', 'pn', and/or 'pnm''");

  // extract local values from the global vectors
  std::vector<double> mypnp(lm.size());
  DRT::UTILS::ExtractMyValues(*pnp, mypnp, lm);

  // extract local values from the global vectors
  std::vector<double> mypn(lm.size());
  DRT::UTILS::ExtractMyValues(*pn, mypn, lm);

  // extract local values from the global vectors
  std::vector<double> mypnm(lm.size());
  DRT::UTILS::ExtractMyValues(*pnm, mypnm, lm);

  // create objects for element arrays
  CORE::LINALG::SerialDenseVector epnp(elemVecdim);
  CORE::LINALG::SerialDenseVector epn(elemVecdim);
  CORE::LINALG::SerialDenseVector epnm(elemVecdim);
  for (int i = 0; i < elemVecdim; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    epnp(i) = mypnp[i];
    epn(i) = mypn[i];
    epnm(i) = mypnm[i];
  }

  double e_acin_e_vnp;
  double e_acin_e_vn;

  for (int i = 0; i < elemVecdim; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    e_acin_e_vnp = (*evaluation_data.acinar_vnp)[ele->LID()];
    e_acin_e_vn = (*evaluation_data.acinar_vn)[ele->LID()];
  }

  // get the volumetric flow rate from the previous time step
  DRT::REDAIRWAYS::ElemParams elem_params;
  elem_params.qout_np = (*evaluation_data.qout_np)[ele->LID()];
  elem_params.qout_n = (*evaluation_data.qout_n)[ele->LID()];
  elem_params.qout_nm = (*evaluation_data.qout_nm)[ele->LID()];
  elem_params.qin_np = (*evaluation_data.qin_np)[ele->LID()];
  elem_params.qin_n = (*evaluation_data.qin_n)[ele->LID()];
  elem_params.qin_nm = (*evaluation_data.qin_nm)[ele->LID()];
  elem_params.volnp = (*evaluation_data.elemVolumenp)[ele->LID()];
  elem_params.voln = (*evaluation_data.elemVolumen)[ele->LID()];

  elem_params.acin_vnp = e_acin_e_vnp;
  elem_params.acin_vn = e_acin_e_vn;

  elem_params.lungVolume_np = evaluation_data.lungVolume_np;
  elem_params.lungVolume_n = evaluation_data.lungVolume_n;
  elem_params.lungVolume_nm = evaluation_data.lungVolume_nm;

  // Routine for computing pextn and pextnp
  if (evaluation_data.compute_awacinter)
  {
    ComputePext(ele, pn, pnp, params);
    elem_params.p_extn = (*evaluation_data.p_extn)[ele->LID()];
    elem_params.p_extnp = (*evaluation_data.p_extnp)[ele->LID()];
  }


  // Routine for open/collapsed decision
  double airwayColl = 0.0;
  ele->getParams("AirwayColl", airwayColl);
  if (airwayColl == 1)
  {
    EvaluateCollapse(ele, epnp, params, dt);
    elem_params.open = (*evaluation_data.open)[ele->LID()];
  }

  // ---------------------------------------------------------------------
  // call routine for calculating element matrix and right hand side
  // ---------------------------------------------------------------------
  Sysmat<distype>(ele, epnp, epn, epnm, elemat1_epetra, elevec1_epetra, mat, elem_params, time, dt,
      evaluation_data.compute_awacinter);

  return 0;
}


/*----------------------------------------------------------------------*
 |  calculate element matrix and right hand side (private)  ismail 01/10|
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::Initial(RedAirway* ele, Teuchos::ParameterList& params,
    DRT::Discretization& discretization, std::vector<int>& lm,
    CORE::LINALG::SerialDenseVector& radii_in, CORE::LINALG::SerialDenseVector& radii_out,
    Teuchos::RCP<const MAT::Material> material)
{
  const int myrank = discretization.Comm().MyPID();

  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  std::vector<int> lmstride;
  Teuchos::RCP<std::vector<int>> lmowner = Teuchos::rcp(new std::vector<int>);
  ele->LocationVector(discretization, lm, *lmowner, lmstride);

  // Calculate the length of airway element
  const double L = GetElementLength<distype>(ele);

  //--------------------------------------------------------------------
  // Initialize the pressure vectors
  //--------------------------------------------------------------------
  if (myrank == (*lmowner)[0])
  {
    int gid = lm[0];
    double val = 0.0;
    evaluation_data.p0np->ReplaceGlobalValues(1, &val, &gid);
    evaluation_data.p0n->ReplaceGlobalValues(1, &val, &gid);
    evaluation_data.p0nm->ReplaceGlobalValues(1, &val, &gid);
  }
  {
    int gid = lm[1];
    double val = 0.0;
    if (myrank == ele->Nodes()[1]->Owner())
    {
      evaluation_data.p0np->ReplaceGlobalValues(1, &val, &gid);
      evaluation_data.p0n->ReplaceGlobalValues(1, &val, &gid);
      evaluation_data.p0nm->ReplaceGlobalValues(1, &val, &gid);
    }

    double A;
    ele->getParams("Area", A);

    if (evaluation_data.solveScatra)
    {
      evaluation_data.junVolMix_Corrector->ReplaceGlobalValues(1, &A, &gid);
      // Initialize scatra
      for (unsigned int sci = 0; sci < lm.size(); sci++)
      {
        int sgid = lm[sci];
        int esgid = ele->Id();
        // -----------------------------------------------------
        // Fluid volume
        // -----------------------------------------------------
        // get length
        const double L = GetElementLength<distype>(ele);

        // get Area
        double area;
        ele->getParams("Area", area);

        // find volume
        double vFluid = area * L;

        // -----------------------------------------------------
        // Initialize concentration of the fluid
        // -----------------------------------------------------
        if (ele->Nodes()[sci]->GetCondition("RedAirwayScatraHemoglobinCond"))
        {
          double intSat = ele->Nodes()[sci]
                              ->GetCondition("RedAirwayScatraHemoglobinCond")
                              ->GetDouble("INITIAL_CONCENTRATION");

          int id = DRT::Problem::Instance()->Materials()->FirstIdByType(
              INPAR::MAT::m_0d_o2_hemoglobin_saturation);
          // check if O2 properties material exists
          if (id == -1)
          {
            dserror("A material defining O2 properties in blood could not be found");
            exit(1);
          }
          const MAT::PAR::Parameter* smat =
              DRT::Problem::Instance()->Materials()->ParameterById(id);
          const MAT::PAR::Hemoglobin_0d_O2_saturation* actmat =
              static_cast<const MAT::PAR::Hemoglobin_0d_O2_saturation*>(smat);

          // how much of blood satisfies this rule
          double per_volume_blood = actmat->per_volume_blood_;
          double o2_sat_per_vol_blood = actmat->o2_sat_per_vol_blood_;
          double nO2perVO2 = actmat->nO2_per_VO2_;

          // get the ratio of blood volume to the reference saturation volume
          double alpha = vFluid / per_volume_blood;

          // get VO2
          double vO2 = alpha * intSat * o2_sat_per_vol_blood;

          // get initial concentration
          double intConc = nO2perVO2 * vO2 / vFluid;

          evaluation_data.scatranp->ReplaceGlobalValues(1, &intConc, &sgid);
          evaluation_data.e1scatranp->ReplaceGlobalValues(1, &intConc, &esgid);
          evaluation_data.e2scatranp->ReplaceGlobalValues(1, &intConc, &esgid);
        }
        else if (ele->Nodes()[sci]->GetCondition("RedAirwayScatraAirCond"))
        {
          double intSat = ele->Nodes()[sci]
                              ->GetCondition("RedAirwayScatraAirCond")
                              ->GetDouble("INITIAL_CONCENTRATION");
          int id = DRT::Problem::Instance()->Materials()->FirstIdByType(
              INPAR::MAT::m_0d_o2_air_saturation);
          // check if O2 properties material exists
          if (id == -1)
          {
            dserror("A material defining O2 properties in air could not be found");
            exit(1);
          }
          const MAT::PAR::Parameter* smat =
              DRT::Problem::Instance()->Materials()->ParameterById(id);
          const MAT::PAR::Air_0d_O2_saturation* actmat =
              static_cast<const MAT::PAR::Air_0d_O2_saturation*>(smat);

          // get atmospheric pressure
          double patm = actmat->atmospheric_p_;
          // get number of O2 moles per unit volume of O2
          double nO2perVO2 = actmat->nO2_per_VO2_;

          // calculate the PO2 at nodes
          double pO2 = intSat * patm;

          // calculate VO2
          double vO2 = vFluid * (pO2 / patm);

          // evaluate initial concentration
          double intConc = nO2perVO2 * vO2 / vFluid;

          evaluation_data.scatranp->ReplaceGlobalValues(1, &intConc, &sgid);
          evaluation_data.e1scatranp->ReplaceGlobalValues(1, &intConc, &esgid);
          evaluation_data.e2scatranp->ReplaceGlobalValues(1, &intConc, &esgid);
        }
        else
        {
          dserror("0D scatra must be predefined as either \"air\" or \"blood\"");
          exit(1);
        }
      }
    }

    val = sqrt(A / M_PI);
    radii_in(0) = val;
    radii_in(1) = 0.0;
    radii_out(0) = 0.0;
    radii_out(1) = val;
  }

  //--------------------------------------------------------------------
  // get the generation numbers
  //--------------------------------------------------------------------
  //  if(myrank == ele->Owner())
  {
    int gid = ele->Id();
    int generation = 0;
    ele->getParams("Generation", generation);

    double val = double(generation);
    evaluation_data.generations->ReplaceGlobalValues(1, &val, &gid);


    double A;
    ele->getParams("Area", A);
    double V = A * L;
    evaluation_data.elemVolume->ReplaceGlobalValues(1, &V, &gid);
    evaluation_data.elemArea0->ReplaceGlobalValues(1, &A, &gid);
  }

}  // AirwayImpl::Initial


/*----------------------------------------------------------------------*
 |  Evaluate open/collapsed state of an airway element following        |
 |  Bates and Irvin (2002), J. Appl. Physiol., 93:705-713.              |
 |                                                         roth 12/2015 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::EvaluateCollapse(
    RedAirway* ele, CORE::LINALG::SerialDenseVector& epn, Teuchos::ParameterList& params, double dt)
{
  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  double s_c, s_o, Pcrit_o, Pcrit_c;
  ele->getParams("S_Close", s_c);
  ele->getParams("S_Open", s_o);
  ele->getParams("Pcrit_Close", Pcrit_c);
  ele->getParams("Pcrit_Open", Pcrit_o);

  double xnp = (*evaluation_data.x_np)[ele->LID()];
  double xn = (*evaluation_data.x_n)[ele->LID()];
  double opennp = (*evaluation_data.open)[ele->LID()];

  // as decisive quantity the pressure value at the first node of the airway element is chosen;
  // using the mean pressure of the airway element caused convergence problems
  double tmp = epn(0);

  /*if (epn(0)-Pcrit_o > 0)
  {
    xnp=xn + s_o*dt*(epn(0)-Pcrit_o);
  }
  else if (epn(0)-Pcrit_c < 0)
  {
    xnp=xn + s_c*dt*(epn(0)-Pcrit_c);
  }*/

  if (tmp > Pcrit_o)
  {
    xnp = xn + s_o * dt * (tmp - Pcrit_o);
  }
  else if (tmp < Pcrit_c)
  {
    xnp = xn + s_c * dt * (tmp - Pcrit_c);
  }

  if (xnp > 1.0)
  {
    xnp = 1.0;
    opennp = 1;
  }
  else if (xnp < 0.0)
  {
    xnp = 0.0;
    opennp = 0;
  }

  int gid = ele->Id();
  evaluation_data.x_np->ReplaceGlobalValues(1, &xnp, &gid);
  evaluation_data.open->ReplaceGlobalValues(1, &opennp, &gid);
}

/*----------------------------------------------------------------------*
 |  Neighbour search for computing pressure prevailing on the outside   |
 |  of an airway.                                                       |
 |                                                         roth 02/2016 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::ComputePext(RedAirway* ele,
    Teuchos::RCP<const Epetra_Vector> pn, Teuchos::RCP<const Epetra_Vector> pnp,
    Teuchos::ParameterList& params)
{
  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  // get node-Id of nearest acinus
  int node_id = (*evaluation_data.airway_acinus_dep)[ele->LID()];

  // Set pextn and pextnp
  double pextnp = (*pnp)[node_id];
  double pextn = (*pn)[node_id];


  int gid = ele->Id();
  evaluation_data.p_extnp->ReplaceGlobalValues(1, &pextnp, &gid);
  evaluation_data.p_extn->ReplaceGlobalValues(1, &pextn, &gid);
}

/*----------------------------------------------------------------------*
 |  Evaluate the values of the degrees of freedom           ismail 01/10|
 |  at terminal nodes.                                                  |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::EvaluateTerminalBC(RedAirway* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization, std::vector<int>& lm,
    CORE::LINALG::SerialDenseVector& rhs, Teuchos::RCP<MAT::Material> material)
{
  const int myrank = discretization.Comm().MyPID();

  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  // get total time
  const double time = evaluation_data.time;

  // get time-step size
  const double dt = evaluation_data.dt;

  // the number of nodes
  const int numnode = lm.size();
  std::vector<int>::iterator it_vcr;

  Teuchos::RCP<const Epetra_Vector> pn = discretization.GetState("pn");

  if (pn == Teuchos::null) dserror("Cannot get state vectors 'pn'");

  // extract local values from the global vectors
  std::vector<double> mypn(lm.size());
  DRT::UTILS::ExtractMyValues(*pn, mypn, lm);

  // create objects for element arrays
  CORE::LINALG::SerialDenseVector epn(numnode);

  // get all values at the last computed time step
  for (int i = 0; i < numnode; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    epn(i) = mypn[i];
  }

  CORE::LINALG::SerialDenseVector eqn(2);
  eqn(0) = (*evaluation_data.qin_n)[ele->LID()];
  eqn(1) = (*evaluation_data.qout_n)[ele->LID()];
  // ---------------------------------------------------------------------------------
  // Resolve the BCs
  // ---------------------------------------------------------------------------------
  for (int i = 0; i < ele->NumNode(); i++)
  {
    if (ele->Nodes()[i]->Owner() == myrank)
    {
      if (ele->Nodes()[i]->GetCondition("RedAirwayPrescribedCond") ||
          ele->Nodes()[i]->GetCondition("Art_redD_3D_CouplingCond") ||
          ele->Nodes()[i]->GetCondition("RedAirwayVentilatorCond"))
      {
        std::string Bc;
        double BCin = 0.0;
        if (ele->Nodes()[i]->GetCondition("RedAirwayPrescribedCond"))
        {
          DRT::Condition* condition = ele->Nodes()[i]->GetCondition("RedAirwayPrescribedCond");
          // Get the type of prescribed bc
          Bc = *(condition->Get<std::string>("boundarycond"));

          if (Bc == "switchFlowPressure")
          {
            // get switch condition variables
            DRT::Condition* switchCondition =
                ele->Nodes()[i]->GetCondition("RedAirwaySwitchFlowPressureCond");

            const int funct_id_flow = switchCondition->GetInt("FUNCT_ID_FLOW");
            const int funct_id_pressure = switchCondition->GetInt("FUNCT_ID_PRESSURE");
            const int funct_id_switch = switchCondition->GetInt("FUNCT_ID_PRESSURE_ACTIVE");

            const double pressure_active =
                DRT::Problem::Instance()
                    ->FunctionById<CORE::UTILS::FunctionOfTime>(funct_id_switch - 1)
                    .Evaluate(time);

            int funct_id_current = 0;
            if (std::abs(pressure_active - 1.0) < 10e-8)
            {
              // phase with pressure bc
              Bc = "pressure";
              funct_id_current = funct_id_pressure;
            }
            else if (std::abs(pressure_active) < 10e-8)
            {
              // phase with flow bc
              Bc = "flow";
              funct_id_current = funct_id_flow;
            }
            else
            {
              dserror(
                  "FUNCTION %i has to take either value 0.0 or 1.0. Not clear if flow or pressure "
                  "boundary condition should be active.",
                  (funct_id_switch - 1));
              exit(1);
            }

            BCin = DRT::Problem::Instance()
                       ->FunctionById<CORE::UTILS::FunctionOfTime>(funct_id_current - 1)
                       .Evaluate(time);
          }
          else
          {
            // -----------------------------------------------------------------
            // Read in the value of the applied BC
            //  Val = curve1*val1 + curve2*func
            // -----------------------------------------------------------------
            const std::vector<int>* curve = condition->Get<std::vector<int>>("curve");
            const std::vector<double>* vals = condition->Get<std::vector<double>>("val");

            // get factor of curve1 or curve2
            const auto curvefac = [&](unsigned id)
            {
              int curvenum = -1;
              if (curve)
                if ((curvenum = (*curve)[id]) >= 0)
                  return DRT::Problem::Instance()
                      ->FunctionById<CORE::UTILS::FunctionOfTime>(curvenum)
                      .Evaluate(time);
                else
                  return 1.0;
              else
                return 1.0;
            };

            // get factor of func
            const double functfac = std::invoke(
                [&]()
                {
                  int functnum = -1;
                  const std::vector<int>* functions = condition->Get<std::vector<int>>("funct");
                  if (functions)
                    if ((functnum = (*functions)[0]) > 0)
                      return DRT::Problem::Instance()
                          ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>(functnum - 1)
                          .Evaluate((ele->Nodes()[i])->X(), time, 0);
                    else
                      return 0.0;
                  else
                    return 0.0;
                });

            BCin = (*vals)[0] * curvefac(0) + functfac * curvefac(1);
          }
          // -----------------------------------------------------------------------------
          // get the local id of the node to whome the bc is prescribed
          // -----------------------------------------------------------------------------
          int local_id = discretization.NodeRowMap()->LID(ele->Nodes()[i]->Id());
          if (local_id < 0)
          {
            dserror("node (%d) doesn't exist on proc(%d)", ele->Nodes()[i]->Id(),
                discretization.Comm().MyPID());
            exit(1);
          }
        }
        else if (ele->Nodes()[i]->GetCondition("Art_redD_3D_CouplingCond"))
        {
          const DRT::Condition* condition =
              ele->Nodes()[i]->GetCondition("Art_redD_3D_CouplingCond");

          Teuchos::RCP<Teuchos::ParameterList> CoupledTo3DParams =
              params.get<Teuchos::RCP<Teuchos::ParameterList>>("coupling with 3D fluid params");
          // -----------------------------------------------------------------
          // If the parameter list is empty, then something is wrong!
          // -----------------------------------------------------------------
          if (CoupledTo3DParams.get() == nullptr)
          {
            dserror(
                "Cannot prescribe a boundary condition from 3D to reduced D, if the parameters "
                "passed don't exist");
            exit(1);
          }

          // -----------------------------------------------------------------
          // Read in Condition type
          // -----------------------------------------------------------------
          //        Type = *(condition->Get<std::string>("CouplingType"));
          // -----------------------------------------------------------------
          // Read in coupling variable rescribed by the 3D simulation
          //
          //     In this case a map called map3D has the following form:
          //     +-----------------------------------------------------------+
          //     |           std::map< std::string               ,  double        >    |
          //     |     +------------------------------------------------+    |
          //     |     |  ID  | coupling variable name | variable value |    |
          //     |     +------------------------------------------------+    |
          //     |     |  1   |   flow1                |     0.12116    |    |
          //     |     +------+------------------------+----------------+    |
          //     |     |  2   |   pressure2            |    10.23400    |    |
          //     |     +------+------------------------+----------------+    |
          //     |     .  .   .   ....                 .     .......    .    |
          //     |     +------+------------------------+----------------+    |
          //     |     |  N   |   variableN            |    value(N)    |    |
          //     |     +------+------------------------+----------------+    |
          //     +-----------------------------------------------------------+
          // -----------------------------------------------------------------

          int ID = condition->GetInt("ConditionID");
          Teuchos::RCP<std::map<std::string, double>> map3D;
          map3D = CoupledTo3DParams->get<Teuchos::RCP<std::map<std::string, double>>>(
              "3D map of values");

          // find the applied boundary variable
          std::stringstream stringID;
          stringID << "_" << ID;
          for (std::map<std::string, double>::iterator itr = map3D->begin(); itr != map3D->end();
               itr++)
          {
            std::string VariableWithId = itr->first;
            size_t found;
            found = VariableWithId.rfind(stringID.str());
            if (found != std::string::npos)
            {
              Bc = std::string(VariableWithId, 0, found);
              BCin = itr->second;
              break;
            }
          }
        }
        else if (ele->Nodes()[i]->GetCondition("RedAirwayVentilatorCond"))
        {
          DRT::Condition* condition = ele->Nodes()[i]->GetCondition("RedAirwayVentilatorCond");
          // Get the type of prescribed bc
          Bc = *(condition->Get<std::string>("phase1"));

          // get the smoothness flag of the two different phases
          std::string phase1Smooth = *(condition->Get<std::string>("Phase1Smoothness"));
          std::string phase2Smooth = *(condition->Get<std::string>("Phase2Smoothness"));

          double period = condition->GetDouble("period");
          double period1 = condition->GetDouble("phase1_period");

          double smoothnessT1 = condition->GetDouble("smoothness_period1");
          double smoothnessT2 = condition->GetDouble("smoothness_period2");

          unsigned int phase_number = 0;

          if (fmod(time, period) >= period1)
          {
            phase_number = 1;
            Bc = *(condition->Get<std::string>("phase2"));
          }

          const std::vector<int>* curve = condition->Get<std::vector<int>>("curve");
          double curvefac = 1.0;
          const std::vector<double>* vals = condition->Get<std::vector<double>>("val");

          // -----------------------------------------------------------------
          // Read in the value of the applied BC
          // -----------------------------------------------------------------
          int curvenum = -1;
          if (curve) curvenum = (*curve)[phase_number];
          if (curvenum >= 0)
            curvefac = DRT::Problem::Instance()
                           ->FunctionById<CORE::UTILS::FunctionOfTime>(curvenum)
                           .Evaluate(time);

          BCin = (*vals)[phase_number] * curvefac;

          // -----------------------------------------------------------------
          // Compute flow value in case a volume is prescribed in the RedAirwayVentilatorCond
          // -----------------------------------------------------------------
          if (Bc == "volume")
          {
            if (fmod(time, period) < period1)
            {
              double Vnp = BCin;
              double Vn =
                  (*vals)[phase_number] * DRT::Problem::Instance()
                                              ->FunctionById<CORE::UTILS::FunctionOfTime>(curvenum)
                                              .Evaluate(time - dt);
              BCin = (Vnp - Vn) / dt;
              Bc = "flow";
            }
          }

          // -----------------------------------------------------------------
          // treat smoothness of the solution
          // -----------------------------------------------------------------
          // if phase 1
          if ((fmod(time, period) < smoothnessT1 && phase_number == 0) ||
              (fmod(time, period) < period1 + smoothnessT2 && phase_number == 1))
          {
            double tsmooth = period;
            if (phase_number == 0 && phase1Smooth == "smooth")
            {
              tsmooth = fmod(time, period);
              double tau = smoothnessT2 / 6.0;
              double Xo = 0.0;
              double Xinf = BCin;
              double Xn = 0.0;
              if (Bc == "pressure")
              {
                Xn = epn(i);
              }
              if (Bc == "flow")
              {
                Xn = eqn(i);
              }
              Xo = (Xn - Xinf) / (exp(-(tsmooth - dt) / tau));
              BCin = Xo * exp(-tsmooth / tau) + Xinf;
            }
            if (phase_number == 1 && phase2Smooth == "smooth")
            {
              tsmooth = fmod(time, period) - period1;
              double tau = smoothnessT2 / 6.0;
              double Xo = 0.0;
              double Xinf = BCin;
              double Xn = 0.0;
              if (Bc == "pressure")
              {
                Xn = epn(i);
              }
              if (Bc == "flow")
              {
                Xn = eqn(i);
              }
              Xo = (Xn - Xinf) / (exp(-(tsmooth - dt) / tau));
              BCin = Xo * exp(-tsmooth / tau) + Xinf;
            }
          }

          // -----------------------------------------------------------------------------
          // get the local id of the node to whome the bc is prescribed
          // -----------------------------------------------------------------------------
          int local_id = discretization.NodeRowMap()->LID(ele->Nodes()[i]->Id());
          if (local_id < 0)
          {
            dserror("node (%d) doesn't exist on proc(%d)", ele->Nodes()[i]->Id(),
                discretization.Comm().MyPID());
            exit(1);
          }
        }
        else
        {
        }

        if (Bc == "pressure")
        {
          // set pressure at node i
          int gid;
          double val;

          gid = lm[i];
          val = BCin;
          evaluation_data.bcval->ReplaceGlobalValues(1, &val, &gid);

          gid = lm[i];
          val = 1;
          evaluation_data.dbctog->ReplaceGlobalValues(1, &val, &gid);
        }
        else if (Bc == "flow")
        {
          // ----------------------------------------------------------
          // Since a node might belong to multiple elements then the
          // flow might be added to the rhs multiple time.
          // To fix this the flow is devided by the number of elements
          // (which is the number of branches). Thus the sum of the
          // final added values is the actual prescribed flow.
          // ----------------------------------------------------------
          int numOfElems = (ele->Nodes()[i])->NumElement();
          BCin /= double(numOfElems);

          rhs(i) += -BCin + rhs(i);
        }
        else
        {
          dserror("precribed [%s] is not defined for reduced airways", Bc.c_str());
          exit(1);
        }
      }
      else
      {
        // ---------------------------------------------------------------
        // If the node is a terminal node, but no b.c is prescribed to it
        // then a zero output pressure is assumed
        // ---------------------------------------------------------------
        if (ele->Nodes()[i]->NumElement() == 1)
        {
          // -------------------------------------------------------------
          // get the local id of the node to whome the bc is prescribed
          // -------------------------------------------------------------

          int local_id = discretization.NodeRowMap()->LID(ele->Nodes()[i]->Id());
          if (local_id < 0)
          {
            dserror("node (%d) doesn't exist on proc(%d)", ele->Nodes()[i],
                discretization.Comm().MyPID());
            exit(1);
          }

          // set pressure at node i
          int gid;
          double val;

          gid = lm[i];
          val = 0.0;
          evaluation_data.bcval->ReplaceGlobalValues(1, &val, &gid);

          gid = lm[i];
          val = 1;
          evaluation_data.dbctog->ReplaceGlobalValues(1, &val, &gid);
        }
      }  // END of if there is no BC but the node still is at the terminal

    }  // END of if node is available on this processor
  }    // End of node i has a condition
}


/*----------------------------------------------------------------------*
 |  Evaluate the values of the degrees of freedom           ismail 01/10|
 |  at terminal nodes.                                                  |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::CalcFlowRates(RedAirway* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization, std::vector<int>& lm,
    Teuchos::RCP<MAT::Material> material)
{
  const int elemVecdim = lm.size();

  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  //----------------------------------------------------------------------
  // get control parameters for time integration
  //----------------------------------------------------------------------
  // get time-step size
  const double dt = evaluation_data.dt;

  // get time
  const double time = evaluation_data.time;

  // ---------------------------------------------------------------------
  // get all general state vectors: flow, pressure,
  // ---------------------------------------------------------------------

  Teuchos::RCP<const Epetra_Vector> pnp = discretization.GetState("pnp");
  Teuchos::RCP<const Epetra_Vector> pn = discretization.GetState("pn");
  Teuchos::RCP<const Epetra_Vector> pnm = discretization.GetState("pnm");

  if (pnp == Teuchos::null || pn == Teuchos::null || pnm == Teuchos::null)
    dserror("Cannot get state vectors 'pnp', 'pn', and/or 'pnm''");

  // extract local values from the global vectors
  std::vector<double> mypnp(lm.size());
  DRT::UTILS::ExtractMyValues(*pnp, mypnp, lm);

  // extract local values from the global vectors
  std::vector<double> mypn(lm.size());
  DRT::UTILS::ExtractMyValues(*pn, mypn, lm);

  // extract local values from the global vectors
  std::vector<double> mypnm(lm.size());
  DRT::UTILS::ExtractMyValues(*pnm, mypnm, lm);

  // create objects for element arrays
  CORE::LINALG::SerialDenseVector epnp(elemVecdim);
  CORE::LINALG::SerialDenseVector epn(elemVecdim);
  CORE::LINALG::SerialDenseVector epnm(elemVecdim);
  for (int i = 0; i < elemVecdim; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    epnp(i) = mypnp[i];
    epn(i) = mypn[i];
    epnm(i) = mypnm[i];
  }

  double e_acin_vnp = 0.0;
  double e_acin_vn = 0.0;

  for (int i = 0; i < elemVecdim; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    e_acin_vnp = (*evaluation_data.acinar_vnp)[ele->LID()];
    e_acin_vn = (*evaluation_data.acinar_vn)[ele->LID()];
  }


  // get the volumetric flow rate from the previous time step
  DRT::REDAIRWAYS::ElemParams elem_params;
  elem_params.qout_np = (*evaluation_data.qout_np)[ele->LID()];
  elem_params.qout_n = (*evaluation_data.qout_n)[ele->LID()];
  elem_params.qout_nm = (*evaluation_data.qout_nm)[ele->LID()];
  elem_params.qin_np = (*evaluation_data.qin_np)[ele->LID()];
  elem_params.qin_n = (*evaluation_data.qin_n)[ele->LID()];
  elem_params.qin_nm = (*evaluation_data.qin_nm)[ele->LID()];

  // TODO same volume is used is this correct?
  elem_params.volnp = (*evaluation_data.elemVolumenp)[ele->LID()];
  elem_params.voln = (*evaluation_data.elemVolumenp)[ele->LID()];

  elem_params.acin_vnp = e_acin_vnp;
  elem_params.acin_vn = e_acin_vn;

  elem_params.lungVolume_np = 0.0;
  elem_params.lungVolume_n = 0.0;
  elem_params.lungVolume_nm = 0.0;

  elem_params.x_np = (*evaluation_data.x_np)[ele->LID()];
  elem_params.x_n = (*evaluation_data.x_n)[ele->LID()];
  elem_params.open = (*evaluation_data.open)[ele->LID()];

  elem_params.p_extn = (*evaluation_data.p_extn)[ele->LID()];
  elem_params.p_extnp = (*evaluation_data.p_extnp)[ele->LID()];

  CORE::LINALG::SerialDenseMatrix sysmat(elemVecdim, elemVecdim, true);
  CORE::LINALG::SerialDenseVector rhs(elemVecdim);


  // ---------------------------------------------------------------------
  // call routine for calculating element matrix and right hand side
  // ---------------------------------------------------------------------
  Sysmat<distype>(ele, epnp, epn, epnm, sysmat, rhs, material, elem_params, time, dt,
      evaluation_data.compute_awacinter);

  double qinnp = -1.0 * (sysmat(0, 0) * epnp(0) + sysmat(0, 1) * epnp(1) - rhs(0));
  double qoutnp = 1.0 * (sysmat(1, 0) * epnp(0) + sysmat(1, 1) * epnp(1) - rhs(1));

  int gid = ele->Id();

  evaluation_data.qin_np->ReplaceGlobalValues(1, &qinnp, &gid);
  evaluation_data.qout_np->ReplaceGlobalValues(1, &qoutnp, &gid);
}  // CalcFlowRates


/*----------------------------------------------------------------------*
 |  Evaluate the elements volume from the change in flow    ismail 07/13|
 |  rates.                                                              |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::CalcElemVolume(RedAirway* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization, std::vector<int>& lm,
    Teuchos::RCP<MAT::Material> material)
{
  // get all essential vector variables

  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  // extract all essential element variables from their corresponding variables
  double qinnp = (*evaluation_data.qin_np)[ele->LID()];
  double qoutnp = (*evaluation_data.qout_np)[ele->LID()];
  double eVolumen = (*evaluation_data.elemVolumen)[ele->LID()];
  double eVolumenp = (*evaluation_data.elemVolumenp)[ele->LID()];

  // get time-step size
  const double dt = evaluation_data.dt;

  // get element global ID
  int gid = ele->Id();

  // -------------------------------------------------------------------
  // find the change of volume from the conservation equation
  // par(V)/par(t) = Qin - Qout
  // numerically
  // (v^n+1 - v^n)/dt = (Qin^n+1 - Qout^n+1)
  // -------------------------------------------------------------------
  double dVol = dt * (qinnp - qoutnp);
  // new volume
  eVolumenp = eVolumen + dVol;

  // -------------------------------------------------------------------
  // Treat possible collapses
  // -------------------------------------------------------------------
  // Calculate the length of airway element
  const double L = GetElementLength<distype>(ele);

  // get area0
  double area0 = 0.0;
  ele->getParams("Area", area0);

  // calculate the current area
  double area = eVolumenp / L;

  // if the airway is near collapsing then fix area to 0.01*area0
  if (area / area0 < 0.01)
  {
    eVolumenp = L * area0 * 0.01;
  }
  // update elem
  evaluation_data.elemVolumenp->ReplaceGlobalValues(1, &eVolumenp, &gid);

  // calculate and update element radius
  double eRadiusnp = std::sqrt(eVolumenp / L * M_1_PI);
  evaluation_data.elemRadiusnp->ReplaceGlobalValues(1, &eRadiusnp, &gid);
}  // CalcElemVolume


/*----------------------------------------------------------------------*
 |  Get the coupled the values on the coupling interface    ismail 07/10|
 |  of the 3D/reduced-D problem                                         |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::GetCoupledValues(RedAirway* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization, std::vector<int>& lm,
    Teuchos::RCP<MAT::Material> material)
{
  const int myrank = discretization.Comm().MyPID();

  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  // get total time
  const double time = evaluation_data.time;

  // the number of nodes
  const int numnode = lm.size();
  std::vector<int>::iterator it_vcr;

  Teuchos::RCP<const Epetra_Vector> pnp = discretization.GetState("pnp");

  if (pnp == Teuchos::null) dserror("Cannot get state vectors 'pnp'");

  // extract local values from the global vectors
  std::vector<double> mypnp(lm.size());
  DRT::UTILS::ExtractMyValues(*pnp, mypnp, lm);

  // create objects for element arrays
  CORE::LINALG::SerialDenseVector epnp(numnode);

  // get all values at the last computed time step
  for (int i = 0; i < numnode; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    epnp(i) = mypnp[i];
  }

  // ---------------------------------------------------------------------------------
  // Resolve the BCs
  // ---------------------------------------------------------------------------------
  for (int i = 0; i < ele->NumNode(); i++)
  {
    if (ele->Nodes()[i]->Owner() == myrank)
    {
      if (ele->Nodes()[i]->GetCondition("Art_redD_3D_CouplingCond"))
      {
        const DRT::Condition* condition = ele->Nodes()[i]->GetCondition("Art_redD_3D_CouplingCond");
        Teuchos::RCP<Teuchos::ParameterList> CoupledTo3DParams =
            params.get<Teuchos::RCP<Teuchos::ParameterList>>("coupling with 3D fluid params");
        // -----------------------------------------------------------------
        // If the parameter list is empty, then something is wrong!
        // -----------------------------------------------------------------
        if (CoupledTo3DParams.get() == nullptr)
        {
          dserror(
              "Cannot prescribe a boundary condition from 3D to reduced D, if the parameters "
              "passed don't exist");
          exit(1);
        }


        // -----------------------------------------------------------------
        // Compute the variable solved by the reduced D simulation to be
        // passed to the 3D simulation
        //
        //     In this case a map called map1D has the following form:
        //     +-----------------------------------------------------------+
        //     |              std::map< std::string            ,  double        > >  |
        //     |     +------------------------------------------------+    |
        //     |     |  ID  | coupling variable name | variable value |    |
        //     |     +------------------------------------------------+    |
        //     |     |  1   |   flow1                |     xxxxxxx    |    |
        //     |     +------+------------------------+----------------+    |
        //     |     |  2   |   pressure2            |     xxxxxxx    |    |
        //     |     +------+------------------------+----------------+    |
        //     |     .  .   .   ....                 .     .......    .    |
        //     |     +------+------------------------+----------------+    |
        //     |     |  N   |   variable(N)          | trash value(N) |    |
        //     |     +------+------------------------+----------------+    |
        //     +-----------------------------------------------------------+
        // -----------------------------------------------------------------

        int ID = condition->GetInt("ConditionID");
        Teuchos::RCP<std::map<std::string, double>> map1D;
        map1D = CoupledTo3DParams->get<Teuchos::RCP<std::map<std::string, double>>>(
            "reducedD map of values");

        std::string returnedBC = *(condition->Get<std::string>("ReturnedVariable"));

        double BC3d = 0.0;
        if (returnedBC == "flow")
        {
          // MUST BE DONE
        }
        else if (returnedBC == "pressure")
        {
          BC3d = epnp(i);
        }
        else
        {
          std::string str = (*condition->Get<std::string>("ReturnedVariable"));
          dserror("%s, is an unimplimented type of coupling", str.c_str());
          exit(1);
        }
        std::stringstream returnedBCwithId;
        returnedBCwithId << returnedBC << "_" << ID;
        std::cout << "COND [" << ID << "] Returning at time " << time << " " << returnedBC << "= "
                  << BC3d << std::endl;
        // -----------------------------------------------------------------
        // Check whether the coupling wrapper has already initialized this
        // map else wise we will have problems with parallelization, that's
        // because of the preassumption that the map is filled and sorted
        // Thus we can use parallel addition
        // -----------------------------------------------------------------

        std::map<std::string, double>::iterator itrMap1D;
        itrMap1D = map1D->find(returnedBCwithId.str());
        if (itrMap1D == map1D->end())
        {
          dserror("The 3D map for (1D - 3D coupling) has no variable (%s) for ID [%d]",
              returnedBC.c_str(), ID);
          exit(1);
        }

        // update the 1D map
        (*map1D)[returnedBCwithId.str()] = BC3d;
      }
    }  // END of if node is available on this processor
  }    // End of node i has a condition
}


/*----------------------------------------------------------------------*
 |  calculate the ammount of fluid mixing inside a          ismail 02/13|
 |  junction                                                            |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::GetJunctionVolumeMix(RedAirway* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization,
    CORE::LINALG::SerialDenseVector& volumeMix_np, std::vector<int>& lm,
    Teuchos::RCP<MAT::Material> material)
{
  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  // get the elements Qin and Qout
  double qoutnp = (*evaluation_data.qout_np)[ele->LID()];
  double qinnp = (*evaluation_data.qin_np)[ele->LID()];
  double evolnp = (*evaluation_data.elemVolumenp)[ele->LID()];

  //--------------------------------------------------------------------
  // get element length
  //--------------------------------------------------------------------
  const double L = GetElementLength<distype>(ele);

  // Check if the node is attached to any other elements
  if (qoutnp >= 0.0)
  {
    volumeMix_np(1) = evolnp / L;
    //  ele->getParams("Area",volumeMix_np(1));
  }
  if (qinnp < 0.0)
  {
    volumeMix_np(0) = evolnp / L;
    //  ele->getParams("Area",volumeMix_np(0));
  }

  for (int i = 0; i < iel; i++)
  {
    {
      if (ele->Nodes()[i]->NumElement() == 1) volumeMix_np(i) = evolnp / L;
      // ele->getParams("Area",volumeMix_np(i));
    }
  }

  if (ele->Nodes()[0]->GetCondition("RedAirwayPrescribedScatraCond"))
  {
    if (qinnp >= 0) volumeMix_np(0) = evolnp / L;
  }
  if (ele->Nodes()[1]->GetCondition("RedAirwayPrescribedScatraCond"))
  {
    if (qoutnp < 0) volumeMix_np(1) = evolnp / L;
  }
}


/*----------------------------------------------------------------------*
 |  calculate the scalar transport                          ismail 02/13|
 |                                                                      |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::SolveScatra(RedAirway* ele, Teuchos::ParameterList& params,
    DRT::Discretization& discretization, CORE::LINALG::SerialDenseVector& scatranp,
    CORE::LINALG::SerialDenseVector& volumeMix_np, std::vector<int>& lm,
    Teuchos::RCP<MAT::Material> material)
{
  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  // get the elements Qin and Qout
  double q_out = (*evaluation_data.qout_np)[ele->LID()];
  double q_in = (*evaluation_data.qin_np)[ele->LID()];
  double eVoln = (*evaluation_data.elemVolumen)[ele->LID()];
  double eVolnp = (*evaluation_data.elemVolumenp)[ele->LID()];
  double e1s = (*evaluation_data.e1scatran)[ele->LID()];
  double e2s = (*evaluation_data.e2scatran)[ele->LID()];

  // get time step size
  const double dt = evaluation_data.dt;

  // get time
  const double time = evaluation_data.time;

  //--------------------------------------------------------------------
  // get element length
  //--------------------------------------------------------------------
  const double L = GetElementLength<distype>(ele);


  // get area
  double areanp = eVolnp / L;
  // ele->getParams("Area",area);

  // evaluate velocity at nodes (1) and (2)
  double vel1 = q_in / areanp;
  double vel2 = q_out / areanp;

  CORE::LINALG::Matrix<2, 1> velv;
  velv(0, 0) = vel1;
  velv(1, 0) = vel2;

  // Get CFL number
  double cfl1 = fabs(vel1) * dt / L;
  double cfl2 = fabs(vel2) * dt / L;

  if (cfl1 >= 1.0 || cfl2 >= 1.0)
  {
    dserror("Error 0D scatra solver detected a CFL numbers > 1.0: CFL(%f,%f)\n", cfl1, cfl2);
    exit(0);
  }
  //--------------------------------------------------------------------
  // if vel>=0 then node(2) is analytically evaluated;
  //  ---> node(1) is either prescribed or comes from the junction
  // if vel< 0 then node(1) is analytically evaluated;
  //  ---> node(2) is either prescribed or comes from the junction
  //--------------------------------------------------------------------
  // solve transport for upstream when velocity > 0
  if (vel2 >= 0.0)
  {
    double scnp = 0.0;
    scnp = e2s - dt * (q_out * e2s - q_out * e1s) / eVoln;
    //    scnp = e2s - dt*(q_out*e2s-q_in*e1s)/eVoln;
    int gid = ele->Id();
    // Update the upstream transport
    evaluation_data.e2scatranp->ReplaceGlobalValues(1, &scnp, &gid);
  }
  // solve transport for upstream when velocity < 0
  if (vel1 < 0.0)
  {
    double scnp = 0.0;
    scnp = e1s + dt * (q_in * e1s - q_in * e2s) / eVoln;
    //    scnp = e1s + dt*(q_in*e1s-q_out*e2s)/eVoln;
    int gid = ele->Id();
    // Update the upstream transport
    evaluation_data.e1scatranp->ReplaceGlobalValues(1, &scnp, &gid);
  }

  //--------------------------------------------------------------------
  // Prescribing boundary condition
  //--------------------------------------------------------------------
  for (int i = 0; i < 2; i++)
  {
    if (ele->Nodes()[i]->GetCondition("RedAirwayPrescribedScatraCond"))
    {
      double scnp = 0.0;
      DRT::Condition* condition = ele->Nodes()[i]->GetCondition("RedAirwayPrescribedScatraCond");
      // Get the type of prescribed bc

      const std::vector<int>* curve = condition->Get<std::vector<int>>("curve");
      double curvefac = 1.0;
      const std::vector<double>* vals = condition->Get<std::vector<double>>("val");

      // -----------------------------------------------------------------
      // Read in the value of the applied BC
      // -----------------------------------------------------------------
      int curvenum = -1;
      if (curve) curvenum = (*curve)[0];
      if (curvenum >= 0)
        curvefac =
            DRT::Problem::Instance()->FunctionById<CORE::UTILS::FunctionOfTime>(curvenum).Evaluate(
                time);

      scnp = (*vals)[0] * curvefac;

      const std::vector<int>* functions = condition->Get<std::vector<int>>("funct");
      int functnum = -1;
      if (functions)
        functnum = (*functions)[0];
      else
        functnum = -1;

      double functionfac = 0.0;
      if (functnum > 0)
      {
        functionfac = DRT::Problem::Instance()
                          ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>(functnum - 1)
                          .Evaluate((ele->Nodes()[i])->X(), time, 0);
      }
      scnp += functionfac;

      // ----------------------------------------------------
      // convert O2 saturation to O2 concentration
      // ---------------------------------------------------

      // -------------------------------------------------------------------
      // find out if the material type is Air or Blood
      // -------------------------------------------------------------------
      std::string fluidType = "none";
      // if RedAirwayScatraAirCond then material type is air
      if (ele->Nodes()[0]->GetCondition("RedAirwayScatraAirCond") != nullptr &&
          ele->Nodes()[1]->GetCondition("RedAirwayScatraAirCond") != nullptr)
      {
        fluidType = "air";
      }
      // if RedAirwayScatraHemoglobinCond then material type is blood
      else if (ele->Nodes()[0]->GetCondition("RedAirwayScatraHemoglobinCond") != nullptr &&
               ele->Nodes()[1]->GetCondition("RedAirwayScatraHemoglobinCond") != nullptr)
      {
        fluidType = "blood";
      }
      else
      {
        dserror("A scalar transport element must be defined either as \"air\" or \"blood\"");
        exit(1);
      }

      // -------------------------------------------------------------------
      // Convert O2 concentration to PO2
      // -------------------------------------------------------------------
      // Calculate the length of airway element
      const double length = GetElementLength<distype>(ele);

      double vFluid = length * areanp;
      if (fluidType == "air")
      {
        // -----------------------------------------------------------------
        // Get O2 properties in air
        // -----------------------------------------------------------------

        int id = DRT::Problem::Instance()->Materials()->FirstIdByType(
            INPAR::MAT::m_0d_o2_air_saturation);
        // check if O2 properties material exists
        if (id == -1)
        {
          dserror("A material defining O2 properties in air could not be found");
          exit(1);
        }
        const MAT::PAR::Parameter* smat = DRT::Problem::Instance()->Materials()->ParameterById(id);
        const MAT::PAR::Air_0d_O2_saturation* actmat =
            static_cast<const MAT::PAR::Air_0d_O2_saturation*>(smat);

        // get atmospheric pressure
        double patm = actmat->atmospheric_p_;
        // get number of O2 moles per unit volume of O2
        double nO2perVO2 = actmat->nO2_per_VO2_;

        // -----------------------------------------------------------------
        // Calculate Vo2 in air
        // -----------------------------------------------------------------
        // calculate the PO2 at nodes
        double pO2 = scnp * patm;

        // calculate the VO2 at nodes
        double vO2 = vFluid * (pO2 / patm);

        // calculate O2 concentration
        scnp = nO2perVO2 * vO2 / vFluid;
      }
      else if (fluidType == "blood")
      {
        // -----------------------------------------------------------------
        // Get O2 properties in blood
        // -----------------------------------------------------------------
        int id = DRT::Problem::Instance()->Materials()->FirstIdByType(
            INPAR::MAT::m_0d_o2_hemoglobin_saturation);
        // check if O2 properties material exists
        if (id == -1)
        {
          dserror("A material defining O2 properties in blood could not be found");
          exit(1);
        }
        const MAT::PAR::Parameter* smat = DRT::Problem::Instance()->Materials()->ParameterById(id);
        const MAT::PAR::Hemoglobin_0d_O2_saturation* actmat =
            static_cast<const MAT::PAR::Hemoglobin_0d_O2_saturation*>(smat);

        // how much of blood satisfies this rule
        double per_volume_blood = actmat->per_volume_blood_;
        double o2_sat_per_vol_blood = actmat->o2_sat_per_vol_blood_;
        double nO2perVO2 = actmat->nO2_per_VO2_;

        // -----------------------------------------------------------------
        // Calculate Vo2 in blood
        // -----------------------------------------------------------------
        // get the ratio of blood volume to the reference saturation volume
        double alpha = vFluid / per_volume_blood;

        // get VO2
        double vO2 = alpha * scnp * o2_sat_per_vol_blood;

        // get concentration
        scnp = nO2perVO2 * vO2 / vFluid;
      }

      //
      //
      //
      // ------------------
      if (i == 0)
      {
        int gid = ele->Id();
        double val = scnp;
        if (vel1 < 0.0) val = (*evaluation_data.e1scatranp)[ele->LID()];
        {
          evaluation_data.e1scatranp->ReplaceGlobalValues(1, &val, &gid);
        }
        scatranp(0) = val * areanp;
      }
      else
      {
        int gid = ele->Id();
        double val = scnp;
        if (vel2 >= 0.0) val = (*evaluation_data.e2scatranp)[ele->LID()];
        {
          evaluation_data.e2scatranp->ReplaceGlobalValues(1, &val, &gid);
        }
        scatranp(1) = val * areanp;
      }
    }
  }

  if (vel2 >= 0.0)
  {
    scatranp(1) = (*evaluation_data.e2scatranp)[ele->LID()] * areanp;
  }
  if (vel1 < 0.0)
  {
    scatranp(0) = (*evaluation_data.e1scatranp)[ele->LID()] * areanp;
  }
}


/*----------------------------------------------------------------------*
 |  calculate the scalar transport                          ismail 02/13|
 |                                                                      |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::SolveScatraBifurcations(RedAirway* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization,
    CORE::LINALG::SerialDenseVector& scatra_np, CORE::LINALG::SerialDenseVector& volumeMix_np,
    std::vector<int>& lm, Teuchos::RCP<MAT::Material> material)
{
  const int myrank = discretization.Comm().MyPID();

  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  // get the elements Qin and Qout
  double q_out = (*evaluation_data.qout_n)[ele->LID()];
  double q_in = (*evaluation_data.qin_n)[ele->LID()];
  double eVolnp = (*evaluation_data.elemVolumenp)[ele->LID()];

  // extract local values from the global vectors
  std::vector<double> myscatran(lm.size());
  DRT::UTILS::ExtractMyValues(*evaluation_data.scatran, myscatran, lm);

  //--------------------------------------------------------------------
  // get element length
  //--------------------------------------------------------------------
  // Calculate the length of airway element
  const double L = GetElementLength<distype>(ele);

  // get area
  double areanp = eVolnp / L;
  //  ele->getParams("Area",area);

  // evaluate velocity at nodes (1) and (2)
  double vel1 = q_in / areanp;
  double vel2 = q_out / areanp;

  CORE::LINALG::Matrix<2, 1> velv;
  velv(0, 0) = vel1;
  velv(1, 0) = vel2;

  //--------------------------------------------------------------------
  // if vel>=0 then node(2) is analytically evaluated;
  //  ---> node(1) is either prescribed or comes from the junction
  // if vel< 0 then node(1) is analytically evaluated;
  //  ---> node(2) is either prescribed or comes from the junction
  //--------------------------------------------------------------------
  //  double dx = vel*dt;
  if (vel1 >= 0.0)
  {
    // extrapolate the analytical solution
    double scnp = myscatran[0];
    int gid = ele->Id();

    //    if(myrank == ele->Owner())
    {
      evaluation_data.e1scatranp->ReplaceGlobalValues(1, &scnp, &gid);
    }
  }
  if (vel2 < 0.0)
  {
    // extrapolate the analytical solution
    double scnp = myscatran[1];
    int gid = ele->Id();
    if (myrank == ele->Nodes()[1]->Owner())
    {
      evaluation_data.e2scatranp->ReplaceGlobalValues(1, &scnp, &gid);
    }
    // get the juction solution
  }
}  // SolveScatraBifurcations

/*----------------------------------------------------------------------*
 |  calculate element CFL                                   ismail 02/13|
 |                                                                      |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::CalcCFL(RedAirway* ele, Teuchos::ParameterList& params,
    DRT::Discretization& discretization, std::vector<int>& lm, Teuchos::RCP<MAT::Material> material)
{
  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  // get the elements Qin and Qout
  double q_outnp = (*evaluation_data.qout_np)[ele->LID()];
  double q_innp = (*evaluation_data.qin_np)[ele->LID()];
  double eVolnp = (*evaluation_data.elemVolumenp)[ele->LID()];


  // get time step size
  const double dt = evaluation_data.dt;

  // get time
  //  const double time = evaluation_data.time;

  //--------------------------------------------------------------------
  // get element length
  //--------------------------------------------------------------------
  // Calculate the length of airway element
  const double L = GetElementLength<distype>(ele);

  // get area
  double area = eVolnp / L;
  //  ele->getParams("Area",area);

  // evaluate velocity at nodes (1) and (2)
  double vel1np = q_innp / area;
  double vel2np = q_outnp / area;

  double cfl1np = fabs(vel1np) * dt / L;
  double cfl2np = fabs(vel2np) * dt / L;

  double cflmax = 0.0;
  cflmax = (cfl1np > cflmax) ? cfl1np : cflmax;
  cflmax = (cfl2np > cflmax) ? cfl2np : cflmax;

  int gid = ele->Id();
  if (ele->Nodes()[1]->Owner()) evaluation_data.cfl->ReplaceGlobalValues(1, &cflmax, &gid);
}


/*----------------------------------------------------------------------*
 |  calculate the scalar transport                          ismail 02/13|
 |                                                                      |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::UpdateScatra(RedAirway* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization, std::vector<int>& lm,
    Teuchos::RCP<MAT::Material> material)
{
  const int myrank = discretization.Comm().MyPID();

  // ---------------------------------------------------------------------
  // perform this step only for capillaries
  // ---------------------------------------------------------------------
  if (ele->Nodes()[0]->GetCondition("RedAirwayScatraCapillaryCond") == nullptr ||
      ele->Nodes()[1]->GetCondition("RedAirwayScatraCapillaryCond") == nullptr)
  {
    return;
  }
  else
  {
    Teuchos::RCP<const Epetra_Vector> scatranp = discretization.GetState("scatranp");
    Teuchos::RCP<const Epetra_Vector> avgscatranp = discretization.GetState("avg_scatranp");
    Teuchos::RCP<const Epetra_Vector> dscatranp = discretization.GetState("dscatranp");

    DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

    // extract local values from the global vectors
    std::vector<double> mydscatranp(lm.size());
    DRT::UTILS::ExtractMyValues(*dscatranp, mydscatranp, lm);

    // extract local values from the global vectors
    std::vector<double> myscatranp(lm.size());
    DRT::UTILS::ExtractMyValues(*scatranp, myscatranp, lm);

    // extract local values from the global vectors
    std::vector<double> myavgscatranp(lm.size());
    DRT::UTILS::ExtractMyValues(*avgscatranp, myavgscatranp, lm);

    // get flowrate

    // Get the average concentration

    double scatra_avg = 0.0;
    for (unsigned int i = 0; i < lm.size(); i++)
    {
      scatra_avg += myavgscatranp[i];
    }
    scatra_avg /= double(lm.size());
    for (unsigned int i = 0; i < lm.size(); i++)
    {
      scatra_avg += mydscatranp[i];
    }

    // modify dscatranp to have the new average scatranp
    for (unsigned int i = 0; i < lm.size(); i++)
    {
      int gid = lm[i];
      mydscatranp[i] = scatra_avg - myscatranp[i];
      double val = mydscatranp[i];
      if (myrank == ele->Nodes()[i]->Owner())
      {
        evaluation_data.dscatranp->ReplaceGlobalValues(1, &val, &gid);
      }
    }
  }
}  // UpdateScatra



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::UpdateElem12Scatra(RedAirway* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization, std::vector<int>& lm,
    Teuchos::RCP<MAT::Material> material)
{
  // ---------------------------------------------------------------------
  // perform this step only for capillaries
  // ---------------------------------------------------------------------
  if (ele->Nodes()[0]->GetCondition("RedAirwayScatraCapillaryCond") == nullptr ||
      ele->Nodes()[1]->GetCondition("RedAirwayScatraCapillaryCond") == nullptr)
  {
    return;
  }


  Teuchos::RCP<const Epetra_Vector> dscatranp = discretization.GetState("dscatranp");
  Teuchos::RCP<const Epetra_Vector> scatranp = discretization.GetState("scatranp");
  Teuchos::RCP<const Epetra_Vector> volumeMix = discretization.GetState("junctionVolumeInMix");

  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  // extract local values from the global vectors
  std::vector<double> mydscatranp(lm.size());
  DRT::UTILS::ExtractMyValues(*dscatranp, mydscatranp, lm);

  // extract local values from the global vectors
  std::vector<double> myscatranp(lm.size());
  DRT::UTILS::ExtractMyValues(*scatranp, myscatranp, lm);

  // extract local values from the global vectors
  std::vector<double> myvolmix(lm.size());
  DRT::UTILS::ExtractMyValues(*volumeMix, myvolmix, lm);

  // ---------------------------------------------------------------------
  // element scatra must be updated only at the capillary nodes.
  // ---------------------------------------------------------------------
  double e1s = myscatranp[0];
  double e2s = myscatranp[1];

  int gid = ele->Id();
  evaluation_data.e1scatranp->ReplaceGlobalValues(1, &e1s, &gid);
  evaluation_data.e2scatranp->ReplaceGlobalValues(1, &e2s, &gid);
}


/*----------------------------------------------------------------------*
 |  calculate the scalar transport                          ismail 06/13|
 |                                                                      |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::EvalPO2FromScatra(RedAirway* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization, std::vector<int>& lm,
    Teuchos::RCP<MAT::Material> material)
{
  const int myrank = discretization.Comm().MyPID();

  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  // -------------------------------------------------------------------
  // extract scatra values
  // -------------------------------------------------------------------
  // extract local values from the global vectors
  std::vector<double> myscatranp(lm.size());
  DRT::UTILS::ExtractMyValues(*evaluation_data.scatranp, myscatranp, lm);

  // -------------------------------------------------------------------
  // find out if the material type is Air or Blood
  // -------------------------------------------------------------------
  std::string fluidType = "none";
  // if RedAirwayScatraAirCond then material type is air
  if (ele->Nodes()[0]->GetCondition("RedAirwayScatraAirCond") != nullptr &&
      ele->Nodes()[1]->GetCondition("RedAirwayScatraAirCond") != nullptr)
  {
    fluidType = "air";
  }
  // if RedAirwayScatraHemoglobinCond then material type is blood
  else if (ele->Nodes()[0]->GetCondition("RedAirwayScatraHemoglobinCond") != nullptr &&
           ele->Nodes()[1]->GetCondition("RedAirwayScatraHemoglobinCond") != nullptr)
  {
    fluidType = "blood";
  }
  else
  {
    dserror("A scalar transport element must be defined either as \"air\" or \"blood\"");
    exit(1);
  }

  double eVolnp = (*evaluation_data.elemVolumenp)[ele->LID()];
  // define a empty pO2 vector
  std::vector<double> pO2(lm.size());

  // -------------------------------------------------------------------
  // Convert O2 concentration to PO2
  // -------------------------------------------------------------------
  // Calculate the length of airway element
  const double length = GetElementLength<distype>(ele);

  // get airway area
  double area = eVolnp / length;
  // ele->getParams("Area",area);

  // -------------------------------------------------------------------
  // Get O2 properties in air
  // -------------------------------------------------------------------
  if (fluidType == "air")
  {
    // -----------------------------------------------------------------
    // Get O2 properties in air
    // -----------------------------------------------------------------

    int id =
        DRT::Problem::Instance()->Materials()->FirstIdByType(INPAR::MAT::m_0d_o2_air_saturation);
    // check if O2 properties material exists
    if (id == -1)
    {
      dserror("A material defining O2 properties in air could not be found");
      exit(1);
    }
    const MAT::PAR::Parameter* smat = DRT::Problem::Instance()->Materials()->ParameterById(id);
    const MAT::PAR::Air_0d_O2_saturation* actmat =
        static_cast<const MAT::PAR::Air_0d_O2_saturation*>(smat);

    // get atmospheric pressure
    double patm = actmat->atmospheric_p_;
    // get number of O2 moles per unit volume of O2
    double nO2perVO2 = actmat->nO2_per_VO2_;

    // -----------------------------------------------------------------
    // Calculate Vo2 in air
    // -----------------------------------------------------------------
    // get airway volume
    double vAir = area * length;
    // calculate the VO2 at nodes
    std::vector<double> vO2(lm.size());
    for (unsigned int i = 0; i < vO2.size(); i++)
    {
      vO2[i] = (myscatranp[i] * vAir) / nO2perVO2;
    }
    // calculate PO2 at nodes
    for (unsigned int i = 0; i < pO2.size(); i++)
    {
      pO2[i] = patm * vO2[i] / vAir;
    }
  }
  // -------------------------------------------------------------------
  // Get O2 properties in blood
  // -------------------------------------------------------------------
  else if (fluidType == "blood")
  {
    int id = DRT::Problem::Instance()->Materials()->FirstIdByType(
        INPAR::MAT::m_0d_o2_hemoglobin_saturation);
    // check if O2 properties material exists
    if (id == -1)
    {
      dserror("A material defining O2 properties in blood could not be found");
      exit(1);
    }
    const MAT::PAR::Parameter* smat = DRT::Problem::Instance()->Materials()->ParameterById(id);
    const MAT::PAR::Hemoglobin_0d_O2_saturation* actmat =
        static_cast<const MAT::PAR::Hemoglobin_0d_O2_saturation*>(smat);

    // how much of blood satisfies this rule
    double per_volume_blood = actmat->per_volume_blood_;
    double o2_sat_per_vol_blood = actmat->o2_sat_per_vol_blood_;
    double ph = actmat->p_half_;
    double power = actmat->power_;
    double nO2perVO2 = actmat->nO2_per_VO2_;

    // -----------------------------------------------------------------
    // Calculate Vo2 in blood
    // -----------------------------------------------------------------
    // get airway volume
    double vBlood = area * length;
    // get the ratio of blood volume to the reference saturation volume
    double alpha = vBlood / per_volume_blood;
    double kv = o2_sat_per_vol_blood * alpha;
    // calculate the VO2 at nodes
    std::vector<double> vO2(lm.size());
    for (unsigned int i = 0; i < vO2.size(); i++)
    {
      vO2[i] = (myscatranp[i] * vBlood) / nO2perVO2;
    }

    // calculate PO2 at nodes
    for (unsigned int i = 0; i < pO2.size(); i++)
    {
      pO2[i] = pow(vO2[i] / kv, 1.0 / power) * pow(1.0 - vO2[i] / kv, -1.0 / power) * ph;
    }
  }
  else
  {
    dserror("A scalar transport element must be defined either as \"air\" or \"blood\"");
    exit(1);
  }

  // -------------------------------------------------------------------
  // Set element pO2 to PO2 vector
  // -------------------------------------------------------------------
  for (unsigned int i = 0; i < lm.size(); i++)
  {
    int gid = lm[i];
    double val = pO2[i];
    if (myrank == ele->Nodes()[i]->Owner())
    {
      evaluation_data.po2->ReplaceGlobalValues(1, &val, &gid);
    }
  }
}  // EvalPO2FromScatra



/*----------------------------------------------------------------------*
 |  calculate the scalar transport                          ismail 06/13|
 |                                                                      |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::AirwayImpl<distype>::EvalNodalEssentialValues(RedAirway* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization,
    CORE::LINALG::SerialDenseVector& nodal_surface, CORE::LINALG::SerialDenseVector& nodal_volume,
    CORE::LINALG::SerialDenseVector& nodal_avg_scatra, std::vector<int>& lm,
    Teuchos::RCP<MAT::Material> material)
{
  // ---------------------------------------------------------------------
  // perform this step only for capillaries
  // ---------------------------------------------------------------------
  if (ele->Nodes()[0]->GetCondition("RedAirwayScatraCapillaryCond") == nullptr ||
      ele->Nodes()[1]->GetCondition("RedAirwayScatraCapillaryCond") == nullptr)
  {
    return;
  }

  DRT::REDAIRWAYS::EvaluationData& evaluation_data = DRT::REDAIRWAYS::EvaluationData::get();

  // get time-step size
  const double dt = evaluation_data.dt;

  // ---------------------------------------------------------------------
  // get all general state vectors: flow, pressure,
  // ---------------------------------------------------------------------
  Teuchos::RCP<const Epetra_Vector> scatranp = discretization.GetState("scatranp");

  // ---------------------------------------------------------------------
  // extract scatra values
  // ---------------------------------------------------------------------
  // extract local values from the global vectors
  std::vector<double> myscatranp(lm.size());
  DRT::UTILS::ExtractMyValues(*scatranp, myscatranp, lm);

  double qin = (*evaluation_data.qin_np)[ele->LID()];
  double eVolnp = (*evaluation_data.elemVolumenp)[ele->LID()];

  // ---------------------------------------------------------------------
  // get volume of capillaries
  // ---------------------------------------------------------------------
  // Calculate the length of airway element
  const double length = GetElementLength<distype>(ele);
  // get airway area
  double area = eVolnp / length;

  // get node coordinates and number of elements per node
  {
    nodal_volume[0] = length * area;
    nodal_volume[1] = length * area;

    double avg_scatra = 0.0;
    double vel = fabs(qin / area);
    double dx = dt * vel;
    if (qin >= 0.0)
    {
      //      avg_scatra = myscatranp[1]- 0.5*(myscatranp[1]-myscatranp[0])*dx/length;
      avg_scatra = myscatranp[1] - (myscatranp[1] - myscatranp[0]) * dx / length;
    }
    else
    {
      //      avg_scatra = myscatranp[0]- 0.5*(myscatranp[0]-myscatranp[1])*dx/length;
      avg_scatra = myscatranp[0] - (myscatranp[0] - myscatranp[1]) * dx / length;
    }

    nodal_avg_scatra[0] = avg_scatra;
    nodal_avg_scatra[1] = avg_scatra;
  }
}

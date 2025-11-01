// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_red_airways_airway_impl.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_red_airways_elem_params.hpp"
#include "4C_red_airways_evaluation_data.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_of_time.hpp"

#include <fstream>
#include <iomanip>

FOUR_C_NAMESPACE_OPEN

namespace
{
  /*----------------------------------------------------------------------*
  |  get element length                                      ismail 08/13|
  |                                                                      |
  *----------------------------------------------------------------------*/
  template <Core::FE::CellType distype>
  double get_element_length(Discret::Elements::RedAirway* ele)
  {
    double length = 0.0;
    // get node coordinates and number of elements per node
    static const int numnode = Core::FE::num_nodes(distype);
    Core::Nodes::Node** nodes = ele->nodes();
    // get airway length
    Core::LinAlg::Matrix<3, numnode> xyze;
    for (int inode = 0; inode < numnode; inode++)
    {
      const auto& x = nodes[inode]->x();
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
  template <Core::FE::CellType distype>
  bool get_curve_val_at_cond(const Core::FE::Discretization& discretization, double& bcVal,
      Core::Nodes::Node* node, std::string condName, std::string optionName, std::string condType,
      double time)
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
    if (Discret::Elements::AirwayImpl<distype>::get_first_condition(discretization, node, condName))
    {
      const Core::Conditions::Condition* condition =
          Discret::Elements::AirwayImpl<distype>::get_first_condition(
              discretization, node, condName);
      // Get the type of prescribed bc
      std::string Bc = (condition->parameters().get<std::string>(optionName));
      if (Bc == condType)
      {
        const auto curve = condition->parameters().get<std::vector<std::optional<int>>>("curve");
        double curvefac = 1.0;
        const auto vals = condition->parameters().get<std::vector<double>>("VAL");

        // -----------------------------------------------------------------
        // Read in the value of the applied BC
        //  Val = curve1*val1 + curve2*func
        // -----------------------------------------------------------------
        // get curve1 and val1
        if (curve[0].has_value() && curve[0].value() > 0)
          curvefac = Global::Problem::instance()
                         ->function_by_id<Core::Utils::FunctionOfTime>(curve[0].value())
                         .evaluate(time);

        bcVal = vals[0] * curvefac;

        // get funct 1
        const int functnum = condition->parameters().get_or<int>("VAL", -1);
        double functionfac = 0.0;
        if (functnum > 0)
        {
          functionfac = Global::Problem::instance()
                            ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functnum)
                            .evaluate(node->x(), time, 0);
        }
        // get curve2
        double curve2fac = 1.0;
        if (curve[1].has_value() && curve[1].value() > 0)
          curve2fac = Global::Problem::instance()
                          ->function_by_id<Core::Utils::FunctionOfTime>(curve[1].value())
                          .evaluate(time);

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
  template <Core::FE::CellType distype>
  void sysmat(const Core::FE::Discretization& discretization, Discret::Elements::RedAirway* ele,
      Core::LinAlg::SerialDenseVector& epnp, Core::LinAlg::SerialDenseVector& epn,
      Core::LinAlg::SerialDenseVector& epnm, Core::LinAlg::SerialDenseMatrix& sysmat,
      Core::LinAlg::SerialDenseVector& rhs, std::shared_ptr<const Core::Mat::Material> material,
      Discret::ReducedLung::ElemParams& params, double time, double dt, bool compute_awacinter)
  {
    const auto airway_params = ele->get_airway_params();

    double dens = 0.0;
    double visc = 0.0;

    if (material->material_type() == Core::Materials::m_fluid)
    {
      // get actual material
      const Mat::NewtonianFluid* actmat = static_cast<const Mat::NewtonianFluid*>(material.get());

      // get density
      dens = actmat->density();

      // get dynamic viscosity
      visc = actmat->viscosity();
    }
    else
    {
      FOUR_C_THROW("Material law is not a Newtonian fluid");
    }

    rhs.putScalar(0.0);
    sysmat.putScalar(0.0);

    // Calculate the length of airway element
    const double L = get_element_length<distype>(ele);

    double qout_n = params.qout_n;
    double qout_np = params.qout_np;
    double qin_n = params.qin_n;
    double qin_np = params.qin_np;

    // get the generation number
    const int generation = airway_params.generation;

    double R = -1.0;

    // get element information
    const double Ao = airway_params.area;
    double A = Ao;
    const double velPow = airway_params.power_velocity_profile;

    if (ele->elem_solving_type() == "Linear")
    {
    }
    else if (ele->elem_solving_type() == "NonLinear")
    {
      A = params.volnp / L;
    }
    else
    {
      FOUR_C_THROW("[{}] is not a defined ElemSolvingType of a RED_AIRWAY element",
          ele->elem_solving_type());
    }

    // Get airway branch length
    double l_branch = airway_params.branch_length;
    if (l_branch < 0.0) l_branch = L;

    // evaluate Poiseuille resistance
    double Rp = 2.0 * (2.0 + velPow) * std::numbers::pi * visc * L / (pow(A, 2));

    // evaluate the Reynolds number
    const double Re = 2.0 * fabs(qout_np) / (visc / dens * sqrt(A * std::numbers::pi));

    if (ele->resistance() == "Poiseuille")
    {
      R = Rp;
    }
    else if (ele->resistance() == "Pedley")
    {
      //-----------------------------------------------------------------
      // resistance evaluated using Pedley's model from :
      // Pedley et al (1970)
      //-----------------------------------------------------------------
      double gamma = 0.327;
      R = gamma * (sqrt(Re * 2.0 * sqrt(A / std::numbers::pi) / l_branch)) * Rp;

      //-----------------------------------------------------------------
      // Correct any resistance smaller than Poiseuille's one
      //-----------------------------------------------------------------
      //    if (R < Rp)
      //    {
      //      R = Rp;
      //    }
      double alfa = sqrt(2.0 * sqrt(A / std::numbers::pi) / l_branch);

      double Rep = 1.0 / ((gamma * alfa) * (gamma * alfa));
      double k = 0.50;
      double st = 1.0 / (1.0 + exp(-2 * k * (Re - Rep)));

      R = R * st + Rp * (1.0 - st);
    }
    else if (ele->resistance() == "Generation_Dependent_Pedley")
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
      R = gamma * (sqrt(Re * 2.0 * sqrt(A / std::numbers::pi) / l_branch)) * Rp;

      //-----------------------------------------------------------------
      // Correct any resistance smaller than Poiseuille's one
      //-----------------------------------------------------------------
      if (R < Rp)
      {
        R = Rp;
      }
    }
    else if (ele->resistance() == "Cont_Pedley")
    {
      //-----------------------------------------------------------------
      // resistance evaluated using Pedley's model from :
      // Pedley et al (1970)
      //-----------------------------------------------------------------
      double gamma = 0.327;
      double D = sqrt(A / std::numbers::pi) * 2.0;
      double Rel = (l_branch / D) * (1.0 / (gamma * gamma));
      double lambda = 1.2;
      double Ret = lambda * Rel;

      //-----------------------------------------------------------------
      // Correct any resistance smaller than Poiseuille's one
      //-----------------------------------------------------------------
      if (Re >= Ret)
        R = gamma * (sqrt(Re * 2.0 * sqrt(A / std::numbers::pi) / l_branch)) * Rp;
      else
      {
        double St = gamma * sqrt((D / l_branch) * Ret);
        double bRe = 2.0 * St / (St - 1.0);
        double aRe = (St - 1.0) / pow(Ret, bRe);
        R = (aRe * pow(Re, bRe) + 1.0) * Rp;
      }
    }
    else if (ele->resistance() == "Generation_Dependent_Cont_Pedley")
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
      double D = sqrt(A / std::numbers::pi) * 2.0;
      double Rel = (l_branch / D) * (1.0 / (gamma * gamma));
      double lambda = 1.2;
      double Ret = lambda * Rel;

      //-----------------------------------------------------------------
      // Correct any resistance smaller than Poiseuille's one
      //-----------------------------------------------------------------
      if (Re >= Ret)
        R = gamma * (sqrt(Re * 2.0 * sqrt(A / std::numbers::pi) / l_branch)) * Rp;
      else
      {
        double St = gamma * sqrt((D / l_branch) * Ret);
        double bRe = 2.0 * St / (St - 1.0);
        double aRe = (St - 1.0) / pow(Ret, bRe);
        R = (aRe * pow(Re, bRe) + 1.0) * Rp;
      }
    }
    else if (ele->resistance() == "Reynolds")
    {
      R = Rp * (3.4 + 2.1e-3 * Re);
    }
    else
    {
      FOUR_C_THROW("[{}] is not a defined resistance model", ele->resistance());
    }

    //------------------------------------------------------------
    // Set high resistance for collapsed airway
    //------------------------------------------------------------
    const double airwayColl = airway_params.airway_coll;

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
    const double Ew = airway_params.wall_elasticity;
    const double tw = airway_params.wall_thickness;
    const double nu = airway_params.poisson_ratio;

    // Get element compliance
    double C = 0.0;
    double Ec = 0.0;
    Ec = (Ew * tw * sqrt(std::numbers::pi)) / ((1.0 - nu * nu) * 2.0 * sqrt(A) * Ao * L);
    if (Ec != 0.0)
    {
      C = 1.0 / Ec;
    }

    //------------------------------------------------------------
    // get airway viscous resistance
    //------------------------------------------------------------
    const double Ts = airway_params.viscous_Ts;
    const double phis = airway_params.viscous_phase_shift;
    // define 0D airway components
    double gammas = Ts * tan(phis) * (Ew * tw * sqrt(std::numbers::pi) / (1.0 - nu * nu)) /
                    (4.0 * std::numbers::pi);
    double Rvis = gammas / (Ao * sqrt(Ao) * L);

    //------------------------------------------------------------
    // get airway inductance
    //------------------------------------------------------------
    double I = dens * L / Ao;

    //------------------------------------------------------------
    // get airway convective resistance
    //------------------------------------------------------------
    // get Poiseuille resistance with parabolic profile
    double Rp2nd = 2.0 * (2.0 + 2.0) * std::numbers::pi * visc * L / (pow(A, 2));
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
    for (int i = 0; i < ele->num_node(); i++)
    {
      double pextVal = 0.0;
      // get Pext at time step n
      get_curve_val_at_cond<distype>(discretization, pextVal, ele->nodes()[i],
          "RedAirwayPrescribedExternalPressure", "boundarycond", "ExternalPressure", time - dt);
      pextn += pextVal / double(ele->num_node());

      // get Pext at time step n+1e
      get_curve_val_at_cond<distype>(discretization, pextVal, ele->nodes()[i],
          "RedAirwayPrescribedExternalPressure", "boundarycond", "ExternalPressure", time);
      pextnp += pextVal / double(ele->num_node());
    }

    // Routine to compute pextnp and pextn from neighbourung acinus pressure
    // ComputePext() analog zu EvaluateCollapse()
    // bool compute_awacinter = params.get<bool>("compute_awacinter");
    if (compute_awacinter)
    {
      pextn = params.p_extn;
      pextnp = params.p_extnp;
    }

    if (ele->type() == "Resistive")
    {
      C = 0.0;
      I = 0.0;
      Rconv = 0.0;
      Rvis = 0.0;
    }
    else if (ele->type() == "InductoResistive")
    {
      C = 0.0;
      Rconv = 0.0;
      Rvis = 0.0;
    }
    else if (ele->type() == "CompliantResistive")
    {
      I = 0.0;
      Rconv = 0.0;
      Rvis = 0.0;
    }
    else if (ele->type() == "RLC")
    {
      Rconv = 0.0;
      Rvis = 0.0;
    }
    else if (ele->type() == "ViscoElasticRLC")
    {
      Rconv = 0.0;
    }
    else if (ele->type() == "ConvectiveViscoElasticRLC")
    {
    }
    else
    {
      FOUR_C_THROW("[{}] is not an implemented element yet", (ele->type()));
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
Discret::Elements::RedAirwayImplInterface* Discret::Elements::RedAirwayImplInterface::impl(
    Discret::Elements::RedAirway* red_airway)
{
  switch (red_airway->shape())
  {
    case Core::FE::CellType::line2:
    {
      static AirwayImpl<Core::FE::CellType::line2>* airway;
      if (airway == nullptr)
      {
        airway = new AirwayImpl<Core::FE::CellType::line2>;
      }
      return airway;
    }
    default:
      FOUR_C_THROW(
          "shape {} ({} nodes) not supported", red_airway->shape(), red_airway->num_node());
      break;
  }
  return nullptr;
}


/*----------------------------------------------------------------------*
 | evaluate (public)                                       ismail 01/10 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::AirwayImpl<distype>::evaluate(RedAirway* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3, std::shared_ptr<Core::Mat::Material> mat)
{
  const int elemVecdim = elevec1.length();

  Discret::ReducedLung::EvaluationData& evaluation_data =
      Discret::ReducedLung::EvaluationData::get();
  const auto airway_params = ele->get_airway_params();

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
  // bool higher_order_ele = ele->is_higher_order_element(distype);

  // ---------------------------------------------------------------------
  // get all general state vectors: flow, pressure,
  // ---------------------------------------------------------------------

  std::shared_ptr<const Core::LinAlg::Vector<double>> pnp = discretization.get_state("pnp");
  std::shared_ptr<const Core::LinAlg::Vector<double>> on = discretization.get_state("on");
  std::shared_ptr<const Core::LinAlg::Vector<double>> pnm = discretization.get_state("pnm");

  if (pnp == nullptr || on == nullptr || pnm == nullptr)
    FOUR_C_THROW("Cannot get state vectors 'pnp', 'on', and/or 'pnm''");

  // extract local values from the global vectors
  std::vector<double> mypnp = Core::FE::extract_values(*pnp, lm);

  // extract local values from the global vectors
  std::vector<double> mypn = Core::FE::extract_values(*on, lm);

  // extract local values from the global vectors
  std::vector<double> mypnm = Core::FE::extract_values(*pnm, lm);

  // create objects for element arrays
  Core::LinAlg::SerialDenseVector epnp(elemVecdim);
  Core::LinAlg::SerialDenseVector epn(elemVecdim);
  Core::LinAlg::SerialDenseVector epnm(elemVecdim);
  for (int i = 0; i < elemVecdim; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    epnp(i) = mypnp[i];
    epn(i) = mypn[i];
    epnm(i) = mypnm[i];
  }

  double e_acin_e_vnp;
  double e_acin_e_vn;

  // split area and volumetric flow rate, insert into element arrays
  e_acin_e_vnp = evaluation_data.acinar_vnp->local_values_as_span()[ele->lid()];
  e_acin_e_vn = evaluation_data.acinar_vn->local_values_as_span()[ele->lid()];

  // get the volumetric flow rate from the previous time step
  Discret::ReducedLung::ElemParams elem_params;
  elem_params.qout_np = evaluation_data.qout_np->local_values_as_span()[ele->lid()];
  elem_params.qout_n = evaluation_data.qout_n->local_values_as_span()[ele->lid()];
  elem_params.qout_nm = evaluation_data.qout_nm->local_values_as_span()[ele->lid()];
  elem_params.qin_np = evaluation_data.qin_np->local_values_as_span()[ele->lid()];
  elem_params.qin_n = evaluation_data.qin_n->local_values_as_span()[ele->lid()];
  elem_params.qin_nm = evaluation_data.qin_nm->local_values_as_span()[ele->lid()];
  elem_params.volnp = evaluation_data.elemVolumenp->local_values_as_span()[ele->lid()];
  elem_params.voln = evaluation_data.elemVolumen->local_values_as_span()[ele->lid()];

  elem_params.acin_vnp = e_acin_e_vnp;
  elem_params.acin_vn = e_acin_e_vn;

  elem_params.lungVolume_np = evaluation_data.lungVolume_np;
  elem_params.lungVolume_n = evaluation_data.lungVolume_n;
  elem_params.lungVolume_nm = evaluation_data.lungVolume_nm;

  // Routine for computing pextn and pextnp
  if (evaluation_data.compute_awacinter)
  {
    compute_pext(ele, *on, *pnp, params);
    elem_params.p_extn = evaluation_data.p_extn->local_values_as_span()[ele->lid()];
    elem_params.p_extnp = evaluation_data.p_extnp->local_values_as_span()[ele->lid()];
  }

  // Routine for open/collapsed decision
  const double airwayColl = airway_params.airway_coll;
  if (airwayColl == 1)
  {
    evaluate_collapse(ele, epnp, params, dt);
    elem_params.open = evaluation_data.open->local_values_as_span()[ele->lid()];
  }

  // ---------------------------------------------------------------------
  // call routine for calculating element matrix and right hand side
  // ---------------------------------------------------------------------
  sysmat<distype>(discretization, ele, epnp, epn, epnm, elemat1, elevec1, mat, elem_params, time,
      dt, evaluation_data.compute_awacinter);

  return 0;
}


/*----------------------------------------------------------------------*
 |  calculate element matrix and right hand side (private)  ismail 01/10|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::AirwayImpl<distype>::initial(RedAirway* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& radii_in, Core::LinAlg::SerialDenseVector& radii_out,
    std::shared_ptr<const Core::Mat::Material> material)
{
  const int myrank = Core::Communication::my_mpi_rank(discretization.get_comm());

  Discret::ReducedLung::EvaluationData& evaluation_data =
      Discret::ReducedLung::EvaluationData::get();
  const auto airway_params = ele->get_airway_params();

  std::vector<int> lmstride;
  std::vector<int> lmowner;
  ele->location_vector(discretization, lm, lmowner, lmstride);

  // Calculate the length of airway element
  const double L = get_element_length<distype>(ele);

  //--------------------------------------------------------------------
  // Initialize the pressure vectors
  //--------------------------------------------------------------------
  if (myrank == (lmowner)[0])
  {
    int gid = lm[0];
    double val = 0.0;
    evaluation_data.p0np->replace_global_values(1, &val, &gid);
    evaluation_data.p0n->replace_global_values(1, &val, &gid);
    evaluation_data.p0nm->replace_global_values(1, &val, &gid);
  }
  {
    int gid = lm[1];
    double val = 0.0;
    if (myrank == ele->nodes()[1]->owner())
    {
      evaluation_data.p0np->replace_global_values(1, &val, &gid);
      evaluation_data.p0n->replace_global_values(1, &val, &gid);
      evaluation_data.p0nm->replace_global_values(1, &val, &gid);
    }

    const double A = airway_params.area;

    val = sqrt(A / std::numbers::pi);
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
    int gid = ele->id();
    const int generation = airway_params.generation;

    double val = double(generation);
    evaluation_data.generations->replace_global_values(1, &val, &gid);


    const double A = airway_params.area;
    double V = A * L;
    evaluation_data.elemVolume->replace_global_values(1, &V, &gid);
    evaluation_data.elemArea0->replace_global_values(1, &A, &gid);
  }

}  // AirwayImpl::Initial


/*----------------------------------------------------------------------*
 |  Evaluate open/collapsed state of an airway element following        |
 |  Bates and Irvin (2002), J. Appl. Physiol., 93:705-713.              |
 |                                                         roth 12/2015 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::AirwayImpl<distype>::evaluate_collapse(
    RedAirway* ele, Core::LinAlg::SerialDenseVector& epn, Teuchos::ParameterList& params, double dt)
{
  Discret::ReducedLung::EvaluationData& evaluation_data =
      Discret::ReducedLung::EvaluationData::get();
  const auto airway_params = ele->get_airway_params();

  const double s_c = airway_params.s_close;
  const double s_o = airway_params.s_open;
  const double Pcrit_o = airway_params.p_crit_open;
  const double Pcrit_c = airway_params.p_crit_close;

  double xnp = evaluation_data.x_np->local_values_as_span()[ele->lid()];
  double xn = evaluation_data.x_n->local_values_as_span()[ele->lid()];
  double opennp = evaluation_data.open->local_values_as_span()[ele->lid()];

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

  int gid = ele->id();
  evaluation_data.x_np->replace_global_values(1, &xnp, &gid);
  evaluation_data.open->replace_global_values(1, &opennp, &gid);
}

/*----------------------------------------------------------------------*
 |  Neighbour search for computing pressure prevailing on the outside   |
 |  of an airway.                                                       |
 |                                                         roth 02/2016 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::AirwayImpl<distype>::compute_pext(RedAirway* ele,
    const Core::LinAlg::Vector<double>& on, const Core::LinAlg::Vector<double>& pnp,
    Teuchos::ParameterList& params)
{
  Discret::ReducedLung::EvaluationData& evaluation_data =
      Discret::ReducedLung::EvaluationData::get();

  // get node-Id of nearest acinus
  int node_id = evaluation_data.airway_acinus_dep->local_values_as_span()[ele->lid()];

  // Set pextn and pextnp
  double pextnp = pnp.local_values_as_span()[node_id];
  double pextn = on.local_values_as_span()[node_id];


  int gid = ele->id();
  evaluation_data.p_extnp->replace_global_values(1, &pextnp, &gid);
  evaluation_data.p_extn->replace_global_values(1, &pextn, &gid);
}

/*----------------------------------------------------------------------*
 |  Evaluate the values of the degrees of freedom           ismail 01/10|
 |  at terminal nodes.                                                  |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::AirwayImpl<distype>::evaluate_terminal_bc(RedAirway* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& rhs, std::shared_ptr<Core::Mat::Material> material)
{
  const int myrank = Core::Communication::my_mpi_rank(discretization.get_comm());

  Discret::ReducedLung::EvaluationData& evaluation_data =
      Discret::ReducedLung::EvaluationData::get();

  // get total time
  const double time = evaluation_data.time;

  // the number of nodes
  const int numnode = lm.size();

  std::shared_ptr<const Core::LinAlg::Vector<double>> on = discretization.get_state("on");

  if (on == nullptr) FOUR_C_THROW("Cannot get state vectors 'on'");

  // extract local values from the global vectors
  std::vector<double> mypn = Core::FE::extract_values(*on, lm);

  // create objects for element arrays
  Core::LinAlg::SerialDenseVector epn(numnode);

  // get all values at the last computed time step
  for (int i = 0; i < numnode; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    epn(i) = mypn[i];
  }

  Core::LinAlg::SerialDenseVector eqn(2);
  eqn(0) = evaluation_data.qin_n->local_values_as_span()[ele->lid()];
  eqn(1) = evaluation_data.qout_n->local_values_as_span()[ele->lid()];
  // ---------------------------------------------------------------------------------
  // Resolve the BCs
  // ---------------------------------------------------------------------------------
  for (int i = 0; i < ele->num_node(); i++)
  {
    if (ele->nodes()[i]->owner() == myrank)
    {
      if (get_first_condition(discretization, ele->nodes()[i], "RedAirwayPrescribedCond") ||
          get_first_condition(discretization, ele->nodes()[i], "Art_redD_3D_CouplingCond"))
      {
        std::string Bc;
        double BCin = 0.0;
        if (get_first_condition(discretization, ele->nodes()[i], "RedAirwayPrescribedCond"))
        {
          const Core::Conditions::Condition* condition =
              get_first_condition(discretization, ele->nodes()[i], "RedAirwayPrescribedCond");
          // Get the type of prescribed bc
          Bc = (condition->parameters().get<std::string>("boundarycond"));

          if (Bc == "switchFlowPressure")
          {
            // get switch condition variables
            const Core::Conditions::Condition* switchCondition = get_first_condition(
                discretization, ele->nodes()[i], "RedAirwaySwitchFlowPressureCond");

            const int funct_id_flow = switchCondition->parameters().get<int>("FUNCT_ID_FLOW");
            const int funct_id_pressure =
                switchCondition->parameters().get<int>("FUNCT_ID_PRESSURE");
            const int funct_id_switch =
                switchCondition->parameters().get<int>("FUNCT_ID_PRESSURE_ACTIVE");

            const double pressure_active =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfTime>(funct_id_switch)
                    .evaluate(time);

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
              FOUR_C_THROW(
                  "FUNCTION {} has to take either value 0.0 or 1.0. Not clear if flow or pressure "
                  "boundary condition should be active.",
                  (funct_id_switch - 1));
            }

            BCin = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(funct_id_current)
                       .evaluate(time);
          }
          else
          {
            // -----------------------------------------------------------------
            // Read in the value of the applied BC
            //  Val = curve1*val1 + curve2*func
            // -----------------------------------------------------------------
            const auto* curve =
                condition->parameters().get_if<std::vector<std::optional<int>>>("curve");
            const auto vals = condition->parameters().get<std::vector<double>>("VAL");

            // get factor of curve1 or curve2
            const auto curvefac = [&](unsigned id)
            {
              if (curve)
              {
                if ((*curve)[id].has_value() && (*curve)[id].value() > 0)
                  return Global::Problem::instance()
                      ->function_by_id<Core::Utils::FunctionOfTime>((*curve)[id].value())
                      .evaluate(time);
                else
                  return 1.0;
              }
              else
                return 1.0;
            };

            // get factor of func
            const double functfac = std::invoke(
                [&]()
                {
                  const auto* functions =
                      condition->parameters().get_if<std::vector<std::optional<int>>>("funct");
                  if (functions)
                  {
                    if ((*functions)[0].has_value() && (*functions)[0].value() > 0)
                      return Global::Problem::instance()
                          ->function_by_id<Core::Utils::FunctionOfSpaceTime>(
                              (*functions)[0].value())
                          .evaluate((ele->nodes()[i])->x(), time, 0);
                    else
                      return 0.0;
                  }
                  else
                    return 0.0;
                });

            BCin = vals[0] * curvefac(0) + functfac * curvefac(1);
          }
          // -----------------------------------------------------------------------------
          // get the local id of the node to whom the bc is prescribed
          // -----------------------------------------------------------------------------
          int local_id = discretization.node_row_map()->lid(ele->nodes()[i]->id());
          if (local_id < 0)
          {
            FOUR_C_THROW("node ({}) doesn't exist on proc({})", ele->nodes()[i]->id(),
                Core::Communication::my_mpi_rank(discretization.get_comm()));
          }
        }
        else if (get_first_condition(discretization, ele->nodes()[i], "Art_redD_3D_CouplingCond"))
        {
          const Core::Conditions::Condition* condition =
              get_first_condition(discretization, ele->nodes()[i], "Art_redD_3D_CouplingCond");

          std::shared_ptr<Teuchos::ParameterList> CoupledTo3DParams =
              params.get<std::shared_ptr<Teuchos::ParameterList>>("coupling with 3D fluid params");
          // -----------------------------------------------------------------
          // If the parameter list is empty, then something is wrong!
          // -----------------------------------------------------------------
          if (CoupledTo3DParams.get() == nullptr)
          {
            FOUR_C_THROW(
                "Cannot prescribe a boundary condition from 3D to reduced D, if the parameters "
                "passed don't exist");
          }

          // -----------------------------------------------------------------
          // Read in Condition type
          // -----------------------------------------------------------------
          //        Type = (condition->parameters().get<std::string>("CouplingType"));
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

          int ID = condition->parameters().get<int>("ConditionID");
          std::shared_ptr<std::map<std::string, double>> map3D;
          map3D = CoupledTo3DParams->get<std::shared_ptr<std::map<std::string, double>>>(
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
          evaluation_data.bcval->replace_global_values(1, &val, &gid);

          gid = lm[i];
          val = 1;
          evaluation_data.dbctog->replace_global_values(1, &val, &gid);
        }
        else if (Bc == "flow")
        {
          // ----------------------------------------------------------
          // Since a node might belong to multiple elements then the
          // flow might be added to the rhs multiple time.
          // To fix this the flow is divided by the number of elements
          // (which is the number of branches). Thus the sum of the
          // final added values is the actual prescribed flow.
          // ----------------------------------------------------------
          int numOfElems = (ele->nodes()[i])->num_element();
          BCin /= double(numOfElems);

          rhs(i) += -BCin + rhs(i);
        }
        else
        {
          FOUR_C_THROW("precribed [{}] is not defined for reduced airways", Bc);
        }
      }
      else
      {
        // ---------------------------------------------------------------
        // If the node is a terminal node, but no b.c is prescribed to it
        // then a zero output pressure is assumed
        // ---------------------------------------------------------------
        if (ele->nodes()[i]->num_element() == 1)
        {
          // -------------------------------------------------------------
          // get the local id of the node to whom the bc is prescribed
          // -------------------------------------------------------------

          int local_id = discretization.node_row_map()->lid(ele->nodes()[i]->id());
          if (local_id < 0)
          {
            FOUR_C_THROW("node ({}) doesn't exist on proc({})", ele->nodes()[i]->id(),
                Core::Communication::my_mpi_rank(discretization.get_comm()));
          }

          // set pressure at node i
          int gid;
          double val;

          gid = lm[i];
          val = 0.0;
          evaluation_data.bcval->replace_global_values(1, &val, &gid);

          gid = lm[i];
          val = 1;
          evaluation_data.dbctog->replace_global_values(1, &val, &gid);
        }
      }  // END of if there is no BC but the node still is at the terminal

    }  // END of if node is available on this processor
  }  // End of node i has a condition
}


/*----------------------------------------------------------------------*
 |  Evaluate the values of the degrees of freedom           ismail 01/10|
 |  at terminal nodes.                                                  |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::AirwayImpl<distype>::calc_flow_rates(RedAirway* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material> material)
{
  const int elemVecdim = lm.size();

  Discret::ReducedLung::EvaluationData& evaluation_data =
      Discret::ReducedLung::EvaluationData::get();

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

  std::shared_ptr<const Core::LinAlg::Vector<double>> pnp = discretization.get_state("pnp");
  std::shared_ptr<const Core::LinAlg::Vector<double>> on = discretization.get_state("on");
  std::shared_ptr<const Core::LinAlg::Vector<double>> pnm = discretization.get_state("pnm");

  if (pnp == nullptr || on == nullptr || pnm == nullptr)
    FOUR_C_THROW("Cannot get state vectors 'pnp', 'on', and/or 'pnm''");

  // extract local values from the global vectors
  std::vector<double> mypnp = Core::FE::extract_values(*pnp, lm);

  // extract local values from the global vectors
  std::vector<double> mypn = Core::FE::extract_values(*on, lm);

  // extract local values from the global vectors
  std::vector<double> mypnm = Core::FE::extract_values(*pnm, lm);

  // create objects for element arrays
  Core::LinAlg::SerialDenseVector epnp(elemVecdim);
  Core::LinAlg::SerialDenseVector epn(elemVecdim);
  Core::LinAlg::SerialDenseVector epnm(elemVecdim);
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
    e_acin_vnp = evaluation_data.acinar_vnp->local_values_as_span()[ele->lid()];
    e_acin_vn = evaluation_data.acinar_vn->local_values_as_span()[ele->lid()];
  }


  // get the volumetric flow rate from the previous time step
  Discret::ReducedLung::ElemParams elem_params;
  elem_params.qout_np = evaluation_data.qout_np->local_values_as_span()[ele->lid()];
  elem_params.qout_n = evaluation_data.qout_n->local_values_as_span()[ele->lid()];
  elem_params.qout_nm = evaluation_data.qout_nm->local_values_as_span()[ele->lid()];
  elem_params.qin_np = evaluation_data.qin_np->local_values_as_span()[ele->lid()];
  elem_params.qin_n = evaluation_data.qin_n->local_values_as_span()[ele->lid()];
  elem_params.qin_nm = evaluation_data.qin_nm->local_values_as_span()[ele->lid()];

  // TODO same volume is used is this correct?
  elem_params.volnp = evaluation_data.elemVolumenp->local_values_as_span()[ele->lid()];
  elem_params.voln = evaluation_data.elemVolumenp->local_values_as_span()[ele->lid()];

  elem_params.acin_vnp = e_acin_vnp;
  elem_params.acin_vn = e_acin_vn;

  elem_params.lungVolume_np = 0.0;
  elem_params.lungVolume_n = 0.0;
  elem_params.lungVolume_nm = 0.0;

  elem_params.x_np = evaluation_data.x_np->local_values_as_span()[ele->lid()];
  elem_params.x_n = evaluation_data.x_n->local_values_as_span()[ele->lid()];
  elem_params.open = evaluation_data.open->local_values_as_span()[ele->lid()];

  elem_params.p_extn = evaluation_data.p_extn->local_values_as_span()[ele->lid()];
  elem_params.p_extnp = evaluation_data.p_extnp->local_values_as_span()[ele->lid()];

  Core::LinAlg::SerialDenseMatrix system_matrix(elemVecdim, elemVecdim, true);
  Core::LinAlg::SerialDenseVector rhs(elemVecdim);


  // ---------------------------------------------------------------------
  // call routine for calculating element matrix and right hand side
  // ---------------------------------------------------------------------
  sysmat<distype>(discretization, ele, epnp, epn, epnm, system_matrix, rhs, material, elem_params,
      time, dt, evaluation_data.compute_awacinter);

  double qinnp = -1.0 * (system_matrix(0, 0) * epnp(0) + system_matrix(0, 1) * epnp(1) - rhs(0));
  double qoutnp = 1.0 * (system_matrix(1, 0) * epnp(0) + system_matrix(1, 1) * epnp(1) - rhs(1));

  int gid = ele->id();

  evaluation_data.qin_np->replace_global_values(1, &qinnp, &gid);
  evaluation_data.qout_np->replace_global_values(1, &qoutnp, &gid);
}  // CalcFlowRates


/*----------------------------------------------------------------------*
 |  Evaluate the elements volume from the change in flow    ismail 07/13|
 |  rates.                                                              |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::AirwayImpl<distype>::calc_elem_volume(RedAirway* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material> material)
{
  // get all essential vector variables

  Discret::ReducedLung::EvaluationData& evaluation_data =
      Discret::ReducedLung::EvaluationData::get();
  const auto airway_params = ele->get_airway_params();

  // extract all essential element variables from their corresponding variables
  double qinnp = evaluation_data.qin_np->local_values_as_span()[ele->lid()];
  double qoutnp = evaluation_data.qout_np->local_values_as_span()[ele->lid()];
  double eVolumen = evaluation_data.elemVolumen->local_values_as_span()[ele->lid()];
  double eVolumenp = evaluation_data.elemVolumenp->local_values_as_span()[ele->lid()];

  // get time-step size
  const double dt = evaluation_data.dt;

  // get element global ID
  int gid = ele->id();

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
  const double L = get_element_length<distype>(ele);

  // get area0
  const double area0 = airway_params.area;

  // calculate the current area
  double area = eVolumenp / L;

  // if the airway is near collapsing then fix area to 0.01*area0
  if (area / area0 < 0.01)
  {
    eVolumenp = L * area0 * 0.01;
  }
  // update elem
  evaluation_data.elemVolumenp->replace_global_values(1, &eVolumenp, &gid);

  // calculate and update element radius
  double eRadiusnp = std::sqrt(eVolumenp / L * std::numbers::inv_pi);
  evaluation_data.elemRadiusnp->replace_global_values(1, &eRadiusnp, &gid);
}  // CalcElemVolume


/*----------------------------------------------------------------------*
 |  Get the coupled the values on the coupling interface    ismail 07/10|
 |  of the 3D/reduced-D problem                                         |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::AirwayImpl<distype>::get_coupled_values(RedAirway* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material> material)
{
  const int myrank = Core::Communication::my_mpi_rank(discretization.get_comm());

  Discret::ReducedLung::EvaluationData& evaluation_data =
      Discret::ReducedLung::EvaluationData::get();

  // get total time
  const double time = evaluation_data.time;

  // the number of nodes
  const int numnode = lm.size();

  std::shared_ptr<const Core::LinAlg::Vector<double>> pnp = discretization.get_state("pnp");

  if (pnp == nullptr) FOUR_C_THROW("Cannot get state vectors 'pnp'");

  // extract local values from the global vectors
  std::vector<double> mypnp = Core::FE::extract_values(*pnp, lm);

  // create objects for element arrays
  Core::LinAlg::SerialDenseVector epnp(numnode);

  // get all values at the last computed time step
  for (int i = 0; i < numnode; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    epnp(i) = mypnp[i];
  }

  // ---------------------------------------------------------------------------------
  // Resolve the BCs
  // ---------------------------------------------------------------------------------
  for (int i = 0; i < ele->num_node(); i++)
  {
    if (ele->nodes()[i]->owner() == myrank)
    {
      if (get_first_condition(discretization, ele->nodes()[i], "Art_redD_3D_CouplingCond"))
      {
        const Core::Conditions::Condition* condition =
            get_first_condition(discretization, ele->nodes()[i], "Art_redD_3D_CouplingCond");
        std::shared_ptr<Teuchos::ParameterList> CoupledTo3DParams =
            params.get<std::shared_ptr<Teuchos::ParameterList>>("coupling with 3D fluid params");
        // -----------------------------------------------------------------
        // If the parameter list is empty, then something is wrong!
        // -----------------------------------------------------------------
        if (CoupledTo3DParams.get() == nullptr)
        {
          FOUR_C_THROW(
              "Cannot prescribe a boundary condition from 3D to reduced D, if the parameters "
              "passed don't exist");
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

        int ID = condition->parameters().get<int>("ConditionID");
        std::shared_ptr<std::map<std::string, double>> map1D;
        map1D = CoupledTo3DParams->get<std::shared_ptr<std::map<std::string, double>>>(
            "reducedD map of values");

        std::string returnedBC = (condition->parameters().get<std::string>("ReturnedVariable"));

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
          std::string str = (condition->parameters().get<std::string>("ReturnedVariable"));
          FOUR_C_THROW("{}, is an unimplemented type of coupling", str);
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
          FOUR_C_THROW(
              "The 3D map for (1D - 3D coupling) has no variable ({}) for ID [{}]", returnedBC, ID);
        }

        // update the 1D map
        (*map1D)[returnedBCwithId.str()] = BC3d;
      }
    }  // END of if node is available on this processor
  }  // End of node i has a condition
}


template <Core::FE::CellType distype>
const Core::Conditions::Condition* Discret::Elements::AirwayImpl<distype>::get_first_condition(
    const Core::FE::Discretization& discretization, const Core::Nodes::Node* node,
    const std::string& name)
{
  auto conds = discretization.get_conditions_on_node(name, node);
  return conds.empty() ? nullptr : conds.front();
}

FOUR_C_NAMESPACE_CLOSE

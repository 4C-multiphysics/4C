// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_red_airways_acinus_impl.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_gder2.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_mat_maxwell_0d_acinus.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_red_airways_elem_params.hpp"
#include "4C_red_airways_evaluation_data.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_of_time.hpp"

#include <fstream>
#include <iomanip>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::RedAcinusImplInterface* Discret::Elements::RedAcinusImplInterface::impl(
    Discret::Elements::RedAcinus* red_acinus)
{
  switch (red_acinus->shape())
  {
    case Core::FE::CellType::line2:
    {
      static AcinusImpl<Core::FE::CellType::line2>* acinus;
      if (acinus == nullptr)
      {
        acinus = new AcinusImpl<Core::FE::CellType::line2>;
      }
      return acinus;
    }
    default:
      FOUR_C_THROW(
          "shape {} ({} nodes) not supported", red_acinus->shape(), red_acinus->num_node());
      break;
  }
  return nullptr;
}


/*----------------------------------------------------------------------*
  | constructor (public)                                    ismail 01/10 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::AcinusImpl<distype>::AcinusImpl()
{
}


/*----------------------------------------------------------------------*
 |  calculate element matrix and right hand side (private)  ismail 01/10|
 |                                                                      |
 *----------------------------------------------------------------------*/
/*!
        \brief calculate element matrix and rhs

\param ele              (i) the element those matrix is calculated
        \param eqnp             (i) nodal volumetric flow rate at n+1
        \param evelnp           (i) nodal velocity at n+1
        \param eareanp          (i) nodal cross-sectional area at n+1
        \param eprenp           (i) nodal pressure at n+1
        \param estif            (o) element matrix to calculate
        \param eforce           (o) element rhs to calculate
        \param material         (i) acinus material/dimesion
        \param time             (i) current simulation time
        \param dt               (i) timestep
        */
template <Core::FE::CellType distype>
void sysmat(Discret::Elements::RedAcinus* ele, Core::LinAlg::SerialDenseVector& epnp,
    Core::LinAlg::SerialDenseVector& epn, Core::LinAlg::SerialDenseVector& epnm,
    Core::LinAlg::SerialDenseMatrix& sysmat, Core::LinAlg::SerialDenseVector& rhs,
    const Core::Mat::Material& material, Discret::ReducedLung::ElemParams& params, double time,
    double dt)
{
  const auto acinus_params = ele->get_acinus_params();

  // Decide which acinus material should be used
  if ((material.material_type() == Core::Materials::m_0d_maxwell_acinus_neohookean) ||
      (material.material_type() == Core::Materials::m_0d_maxwell_acinus_exponential) ||
      (material.material_type() == Core::Materials::m_0d_maxwell_acinus_doubleexponential) ||
      (material.material_type() == Core::Materials::m_0d_maxwell_acinus_ogden))
  {
    const double VolAcinus = acinus_params.volume_relaxed;
    const double volAlvDuct = acinus_params.alveolar_duct_volume;
    const auto NumOfAcini = double(floor(VolAcinus / volAlvDuct));

    const std::shared_ptr<Mat::Maxwell0dAcinus> acinus_mat =
        std::dynamic_pointer_cast<Mat::Maxwell0dAcinus>(ele->material());

    // Evaluate material law for acinus
    acinus_mat->evaluate(epnp, epn, epnm, sysmat, rhs, params, NumOfAcini, volAlvDuct, time, dt);
  }
  else
  {
    FOUR_C_THROW("Material law is not a valid reduced dimensional lung acinus material.");
  }
}


/*----------------------------------------------------------------------*
 | evaluate (public)                                       ismail 01/10 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::AcinusImpl<distype>::evaluate(RedAcinus* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3, std::shared_ptr<Core::Mat::Material> mat)
{
  const int elemVecdim = elevec1.length();

  Discret::ReducedLung::EvaluationData& evaluation_data =
      Discret::ReducedLung::EvaluationData::get();

  // Get control parameters for time integration
  // get time-step size
  const double dt = evaluation_data.dt;
  // get time
  const double time = evaluation_data.time;

  // Get all general state vectors: flow, pressure,
  std::shared_ptr<const Core::LinAlg::Vector<double>> pnp = discretization.get_state("pnp");
  std::shared_ptr<const Core::LinAlg::Vector<double>> on = discretization.get_state("on");
  std::shared_ptr<const Core::LinAlg::Vector<double>> pnm = discretization.get_state("pnm");

  std::shared_ptr<const Core::LinAlg::Vector<double>> ial =
      discretization.get_state("intr_ac_link");

  if (pnp == nullptr || on == nullptr || pnm == nullptr)
    FOUR_C_THROW("Cannot get state vectors 'pnp', 'on', and/or 'pnm''");

  // Extract local values from the global vectors
  std::vector<double> mypnp = Core::FE::extract_values(*pnp, lm);

  // Extract local values from the global vectors
  std::vector<double> mypn = Core::FE::extract_values(*on, lm);

  // Extract local values from the global vectors
  std::vector<double> mypnm = Core::FE::extract_values(*pnm, lm);

  // Extract local values from the global vectors
  std::vector<double> myial = Core::FE::extract_values(*ial, lm);

  // Create objects for element arrays
  Core::LinAlg::SerialDenseVector epnp(elemVecdim);
  Core::LinAlg::SerialDenseVector epn(elemVecdim);
  Core::LinAlg::SerialDenseVector epnm(elemVecdim);
  for (int i = 0; i < elemVecdim; ++i)
  {
    // Split area and volumetric flow rate, insert into element arrays
    epnp(i) = mypnp[i];
    epn(i) = mypn[i];
    epnm(i) = mypnm[i];
  }

  double e_acin_e_vnp;
  double e_acin_e_vn;

  // Split area and volumetric flow rate, insert into element arrays
  e_acin_e_vnp = evaluation_data.acinar_vnp->local_values_as_span()[ele->lid()];
  e_acin_e_vn = evaluation_data.acinar_vn->local_values_as_span()[ele->lid()];

  // Get the volumetric flow rate from the previous time step
  Discret::ReducedLung::ElemParams elem_params;
  elem_params.qout_np = evaluation_data.qout_np->local_values_as_span()[ele->lid()];
  elem_params.qout_n = evaluation_data.qout_n->local_values_as_span()[ele->lid()];
  elem_params.qout_nm = evaluation_data.qout_nm->local_values_as_span()[ele->lid()];
  elem_params.qin_np = evaluation_data.qin_np->local_values_as_span()[ele->lid()];
  elem_params.qin_n = evaluation_data.qin_n->local_values_as_span()[ele->lid()];
  elem_params.qin_nm = evaluation_data.qin_nm->local_values_as_span()[ele->lid()];

  elem_params.acin_vnp = e_acin_e_vnp;
  elem_params.acin_vn = e_acin_e_vn;

  elem_params.lungVolume_np = evaluation_data.lungVolume_np;
  elem_params.lungVolume_n = evaluation_data.lungVolume_n;
  elem_params.lungVolume_nm = evaluation_data.lungVolume_nm;

  // Call routine for calculating element matrix and right hand side
  sysmat<distype>(ele, epnp, epn, epnm, elemat1, elevec1, *mat, elem_params, time, dt);

  // Put zeros on second line of matrix and rhs in case of interacinar linker
  if (myial[1] > 0.0)
  {
    elemat1(1, 0) = 0.0;
    elemat1(1, 1) = 0.0;
    elevec1(1) = 0.0;
  }

  return 0;
}

static const Core::Conditions::Condition* get_first_condition(
    const Core::FE::Discretization& discretization, const Core::Nodes::Node* node,
    const std::string& name)
{
  auto conds = discretization.get_conditions_on_node(name, node);
  return conds.empty() ? nullptr : conds.front();
}


/*----------------------------------------------------------------------*
 |  calculate element matrix and right hand side (private)  ismail 01/10|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::AcinusImpl<distype>::initial(RedAcinus* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<const Core::Mat::Material> material)
{
  const int myrank = Core::Communication::my_mpi_rank(discretization.get_comm());

  Discret::ReducedLung::EvaluationData& evaluation_data =
      Discret::ReducedLung::EvaluationData::get();
  const auto acinus_params = ele->get_acinus_params();

  std::vector<int> lmstride;
  std::vector<int> lmowner;
  ele->location_vector(discretization, lm, lmowner, lmstride);

  // Initialize the pressure vectors
  if (myrank == (lmowner)[0])
  {
    int gid = lm[0];
    double val = 0.0;
    evaluation_data.p0np->replace_global_values(1, &val, &gid);
    evaluation_data.p0n->replace_global_values(1, &val, &gid);
    evaluation_data.p0nm->replace_global_values(1, &val, &gid);
  }

  // Find the volume of an acinus element
  {
    int gid2 = ele->id();
    double acin_vol = acinus_params.volume_relaxed;
    evaluation_data.acini_e_volume->replace_global_values(1, &acin_vol, &gid2);
  }

  // Get the generation numbers
  for (int i = 0; i < 2; i++)
  {
    if (get_first_condition(discretization, ele->nodes()[i], "RedAirwayEvalLungVolCond"))
    {
      // find the acinus condition
      int gid = ele->id();
      double val = 1.0;
      evaluation_data.acini_bc->replace_global_values(1, &val, &gid);
    }
  }
  {
    int gid = ele->id();
    int generation = -1;
    double val = double(generation);
    evaluation_data.generations->replace_global_values(1, &val, &gid);
  }

}  // AcinusImpl::Initial


/*----------------------------------------------------------------------*
 |  Evaluate the values of the degrees of freedom           ismail 01/10|
 |  at terminal nodes.                                                  |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::AcinusImpl<distype>::evaluate_terminal_bc(RedAcinus* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& rhs, std::shared_ptr<Core::Mat::Material> material)
{
  const int myrank = Core::Communication::my_mpi_rank(discretization.get_comm());

  Discret::ReducedLung::EvaluationData& evaluation_data =
      Discret::ReducedLung::EvaluationData::get();

  // Get total time
  const double time = evaluation_data.time;

  // The number of nodes
  const int numnode = lm.size();

  std::shared_ptr<const Core::LinAlg::Vector<double>> pnp = discretization.get_state("pnp");

  if (pnp == nullptr) FOUR_C_THROW("Cannot get state vectors 'pnp'");

  // Extract local values from the global vectors
  std::vector<double> mypnp = Core::FE::extract_values(*pnp, lm);

  // Create objects for element arrays
  Core::LinAlg::SerialDenseVector epnp(numnode);

  // Get all values at the last computed time step
  for (int i = 0; i < numnode; ++i)
  {
    // Split area and volumetric flow rate, insert into element arrays
    epnp(i) = mypnp[i];
  }

  /**
   * Resolve the BCs
   **/
  for (int i = 0; i < ele->num_node(); i++)
  {
    if (ele->nodes()[i]->owner() == myrank)
    {
      if (get_first_condition(discretization, ele->nodes()[i], "RedAirwayPrescribedCond") ||
          get_first_condition(discretization, ele->nodes()[i], "Art_redD_3D_CouplingCond") ||
          get_first_condition(discretization, ele->nodes()[i], "RedAcinusVentilatorCond"))
      {
        std::string Bc;
        double BCin = 0.0;
        if (get_first_condition(discretization, ele->nodes()[i], "RedAirwayPrescribedCond"))
        {
          const Core::Conditions::Condition* condition =
              get_first_condition(discretization, ele->nodes()[i], "RedAirwayPrescribedCond");
          // Get the type of prescribed bc
          Bc = (condition->parameters().get<std::string>("boundarycond"));

          const auto vals = condition->parameters().get<std::vector<double>>("VAL");
          const auto curve = condition->parameters().get<std::vector<std::optional<int>>>("curve");
          const auto functions =
              condition->parameters().get<std::vector<std::optional<int>>>("funct");

          // Read in the value of the applied BC
          // Get factor of first CURVE
          double curvefac = 1.0;
          if (curve[0].has_value() && curve[0].value() > 0)
          {
            curvefac = Global::Problem::instance()
                           ->function_by_id<Core::Utils::FunctionOfTime>(curve[0].value())
                           .evaluate(time);
            BCin = vals[0] * curvefac;
          }
          else
          {
            FOUR_C_THROW("no boundary condition defined!");
          }

          // Get factor of FUNCT
          double functionfac = 0.0;
          if (functions[0].has_value() && functions[0].value() > 0)
          {
            functionfac =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functions[0].value())
                    .evaluate((ele->nodes()[i])->x(), time, 0);
          }

          // Get factor of second CURVE
          double curve2fac = 1.0;
          if (curve[1].has_value() && curve[1].value() > 0)
            curve2fac = Global::Problem::instance()
                            ->function_by_id<Core::Utils::FunctionOfTime>(curve[1].value())
                            .evaluate(time);

          // Add first_CURVE + FUNCTION * second_CURVE
          BCin += functionfac * curve2fac;

          // Get the local id of the node to whom the bc is prescribed
          int local_id = discretization.node_row_map()->lid(ele->nodes()[i]->id());
          if (local_id < 0)
          {
            FOUR_C_THROW("node ({}) doesn't exist on proc({})", ele->nodes()[i]->id(),
                Core::Communication::my_mpi_rank(discretization.get_comm()));
          }
        }
        /**
         * For Art_redD_3D_CouplingCond bc
         **/
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
          //     |           std::map< string               ,  double        >    |
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
        /**
         * For RedAcinusVentilatorCond bc
         **/
        else if (get_first_condition(discretization, ele->nodes()[i], "RedAcinusVentilatorCond"))
        {
          const Core::Conditions::Condition* condition =
              get_first_condition(discretization, ele->nodes()[i], "RedAcinusVentilatorCond");
          // Get the type of prescribed bc
          Bc = (condition->parameters().get<std::string>("phase1"));

          double period = condition->parameters().get<double>("period");
          double period1 = condition->parameters().get<double>("phase1_period");

          unsigned int phase_number = 0;

          if (fmod(time, period) > period1)
          {
            phase_number = 1;
            Bc = (condition->parameters().get<std::string>("phase2"));
          }

          const auto curve = condition->parameters().get<std::vector<std::optional<int>>>("curve");
          double curvefac = 1.0;
          const auto vals = condition->parameters().get<std::vector<double>>("VAL");

          // Read in the value of the applied BC
          if (curve[phase_number].has_value() && curve[phase_number].value() > 0)
          {
            curvefac =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfTime>(curve[phase_number].value())
                    .evaluate(time);
            BCin = vals[phase_number] * curvefac;
          }
          else
          {
            FOUR_C_THROW("no boundary condition defined!");
          }

          // Get the local id of the node to whom the bc is prescribed
          int local_id = discretization.node_row_map()->lid(ele->nodes()[i]->id());
          if (local_id < 0)
          {
            FOUR_C_THROW("node ({}) doesn't exist on proc({})", ele->nodes()[i]->id(),
                Core::Communication::my_mpi_rank(discretization.get_comm()));
          }
        }
        else
        {
        }

        /**
         * For pressure or VolumeDependentPleuralPressure bc
         **/
        if (Bc == "pressure" || Bc == "VolumeDependentPleuralPressure")
        {
          if (Bc == "VolumeDependentPleuralPressure")
          {
            const Core::Conditions::Condition* pplCond = get_first_condition(
                discretization, ele->nodes()[i], "RedAirwayVolDependentPleuralPressureCond");
            double Pp_np = 0.0;
            if (pplCond)
            {
              const auto curve =
                  pplCond->parameters().get<std::vector<std::optional<int>>>("curve");
              double curvefac = 1.0;
              const auto vals = pplCond->parameters().get<std::vector<double>>("VAL");

              // Read in the value of the applied BC
              if (curve[0].has_value() && curve[0].value() > 0)
              {
                curvefac = Global::Problem::instance()
                               ->function_by_id<Core::Utils::FunctionOfTime>(curve[0].value())
                               .evaluate(time);
              }

              // Get parameters for VolumeDependentPleuralPressure condition
              std::string ppl_Type = (pplCond->parameters().get<std::string>("TYPE"));
              double ap = pplCond->parameters().get<double>("P_PLEURAL_0");
              double bp = pplCond->parameters().get<double>("P_PLEURAL_LIN");
              double cp = pplCond->parameters().get<double>("P_PLEURAL_NONLIN");
              double dp = pplCond->parameters().get<double>("TAU");
              double RV = pplCond->parameters().get<double>("RV");
              double TLC = pplCond->parameters().get<double>("TLC");

              Discret::ReducedLung::EvaluationData& evaluation_data =
                  Discret::ReducedLung::EvaluationData::get();

              // Safety check: in case of polynomial TLC is not used
              if (((ppl_Type == "Linear_Polynomial") or (ppl_Type == "Nonlinear_Polynomial")) and
                  (TLC != 0.0))
              {
                FOUR_C_THROW(
                    "TLC is not used for the following type of VolumeDependentPleuralPressure BC: "
                    "{}.\n Set TLC = 0.0",
                    ppl_Type.c_str());
              }

              // Safety check: in case of Ogden TLC, P_PLEURAL_0, and P_PLEURAL_LIN
              if ((ppl_Type == "Nonlinear_Ogden") and
                  ((TLC != 0.0) or (ap != 0.0) or (bp != 0.0) or (dp == 0.0)))
              {
                FOUR_C_THROW(
                    "Parameters are not set correctly for Nonlinear_Ogden. Only P_PLEURAL_NONLIN, "
                    "TAU and RV are used. Set all others to zero. TAU is not allowed to be zero.");
              }

              if (ppl_Type == "Linear_Polynomial")
              {
                const double lungVolumenp = evaluation_data.lungVolume_n;
                Pp_np = ap + bp * (lungVolumenp - RV) + cp * pow((lungVolumenp - RV), dp);
              }
              else if (ppl_Type == "Linear_Exponential")
              {
                const double lungVolumenp = evaluation_data.lungVolume_n;
                const double TLCnp = (lungVolumenp - RV) / (TLC - RV);
                Pp_np = ap + bp * TLCnp + cp * exp(dp * TLCnp);
              }
              else if (ppl_Type == "Linear_Ogden")
              {
                const double lungVolumenp = evaluation_data.lungVolume_n;
                Pp_np = RV / lungVolumenp * cp / dp * (1 - pow(RV / lungVolumenp, dp));
              }
              else if (ppl_Type == "Nonlinear_Polynomial")
              {
                const double lungVolumenp = evaluation_data.lungVolume_np;
                Pp_np = ap + bp * (lungVolumenp - RV) + cp * pow((lungVolumenp - RV), dp);
              }
              else if (ppl_Type == "Nonlinear_Exponential")
              {
                const double lungVolumenp = evaluation_data.lungVolume_np;
                const double TLCnp = (lungVolumenp - RV) / (TLC - RV);
                Pp_np = ap + bp * TLCnp + cp * exp(dp * TLCnp);
              }
              else if (ppl_Type == "Nonlinear_Ogden")
              {
                const double lungVolumenp = evaluation_data.lungVolume_np;
                Pp_np = RV / lungVolumenp * cp / dp * (1 - pow(RV / lungVolumenp, dp));
              }
              else
              {
                FOUR_C_THROW("Unknown volume pleural pressure type: {}", ppl_Type);
              }
              Pp_np *= curvefac * (vals[0]);
            }
            else
            {
              FOUR_C_THROW("No volume dependent pleural pressure condition was defined");
            }
            BCin += Pp_np;
          }

          Discret::ReducedLung::EvaluationData& evaluation_data =
              Discret::ReducedLung::EvaluationData::get();

          // Set pressure at node i
          int gid;
          double val;

          gid = lm[i];
          val = BCin;
          evaluation_data.bcval->replace_global_values(1, &val, &gid);

          gid = lm[i];
          val = 1;
          evaluation_data.dbctog->replace_global_values(1, &val, &gid);
        }
        /**
         * For flow bc
         **/
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
          FOUR_C_THROW("prescribed [{}] is not defined for reduced acinuss", Bc);
        }
      }
      /**
       * If the node is a terminal node, but no b.c is prescribed to it
       * then a zero output pressure is assumed
       **/
      else
      {
        if (ele->nodes()[i]->num_element() == 1)
        {
          // Get the local id of the node to whom the bc is prescribed
          int local_id = discretization.node_row_map()->lid(ele->nodes()[i]->id());
          if (local_id < 0)
          {
            FOUR_C_THROW("node ({}) doesn't exist on proc({})", ele->nodes()[i]->id(),
                Core::Communication::my_mpi_rank(discretization.get_comm()));
          }

          Discret::ReducedLung::EvaluationData& evaluation_data =
              Discret::ReducedLung::EvaluationData::get();

          // Set pressure=0.0 at node i
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
 |  Calculate flowrate at current iteration step and the    ismail 01/10|
 |  corresponding acinus volume via dV = 0.5*(qnp+qn)*dt                 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::AcinusImpl<distype>::calc_flow_rates(RedAcinus* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material> material)

{
  const int elemVecdim = lm.size();

  Discret::ReducedLung::EvaluationData& evaluation_data =
      Discret::ReducedLung::EvaluationData::get();
  const auto acinus_params = ele->get_acinus_params();

  // Get control parameters for time integration
  // Get time-step size
  const double dt = evaluation_data.dt;
  // Get time
  const double time = evaluation_data.time;

  // Get all general state vectors: flow, pressure,
  std::shared_ptr<const Core::LinAlg::Vector<double>> pnp = discretization.get_state("pnp");
  std::shared_ptr<const Core::LinAlg::Vector<double>> on = discretization.get_state("on");
  std::shared_ptr<const Core::LinAlg::Vector<double>> pnm = discretization.get_state("pnm");

  if (pnp == nullptr || on == nullptr || pnm == nullptr)
    FOUR_C_THROW("Cannot get state vectors 'pnp', 'on', and/or 'pnm''");

  // Extract local values from the global vectors
  std::vector<double> mypnp = Core::FE::extract_values(*pnp, lm);

  // Extract local values from the global vectors
  std::vector<double> mypn = Core::FE::extract_values(*on, lm);

  // Extract local values from the global vectors
  std::vector<double> mypnm = Core::FE::extract_values(*pnm, lm);

  // Create objects for element arrays
  Core::LinAlg::SerialDenseVector epnp(elemVecdim);
  Core::LinAlg::SerialDenseVector epn(elemVecdim);
  Core::LinAlg::SerialDenseVector epnm(elemVecdim);
  for (int i = 0; i < elemVecdim; ++i)
  {
    // Split area and volumetric flow rate, insert into element arrays
    epnp(i) = mypnp[i];
    epn(i) = mypn[i];
    epnm(i) = mypnm[i];
  }

  double e_acin_vnp = 0.0;
  double e_acin_vn = 0.0;

  for (int i = 0; i < elemVecdim; ++i)
  {
    // Split area and volumetric flow rate, insert into element arrays
    e_acin_vnp = evaluation_data.acinar_vnp->local_values_as_span()[ele->lid()];
    e_acin_vn = evaluation_data.acinar_vn->local_values_as_span()[ele->lid()];
  }

  // Get the volumetric flow rate from the previous time step
  Discret::ReducedLung::ElemParams elem_params;
  elem_params.qout_np = evaluation_data.qout_np->local_values_as_span()[ele->lid()];
  elem_params.qout_n = evaluation_data.qout_n->local_values_as_span()[ele->lid()];
  elem_params.qout_nm = evaluation_data.qout_nm->local_values_as_span()[ele->lid()];
  elem_params.qin_np = evaluation_data.qin_np->local_values_as_span()[ele->lid()];
  elem_params.qin_n = evaluation_data.qin_n->local_values_as_span()[ele->lid()];
  elem_params.qin_nm = evaluation_data.qin_nm->local_values_as_span()[ele->lid()];

  elem_params.acin_vnp = e_acin_vnp;
  elem_params.acin_vn = e_acin_vn;

  Core::LinAlg::SerialDenseMatrix system_matrix(elemVecdim, elemVecdim, true);
  Core::LinAlg::SerialDenseVector rhs(elemVecdim);

  // Call routine for calculating element matrix and right hand side
  sysmat<distype>(ele, epnp, epn, epnm, system_matrix, rhs, *material, elem_params, time, dt);

  double qn = evaluation_data.qin_n->local_values_as_span()[ele->lid()];
  double qnp = -1.0 * (system_matrix(0, 0) * epnp(0) + system_matrix(0, 1) * epnp(1) - rhs(0));

  int gid = ele->id();

  evaluation_data.qin_np->replace_global_values(1, &qnp, &gid);
  evaluation_data.qout_np->replace_global_values(1, &qnp, &gid);

  // Calculate the new volume of the acinus due to the incoming flow; 0.5*(qnp+qn)*dt
  {
    double acinus_volume = e_acin_vn;
    acinus_volume += 0.5 * (qnp + qn) * dt;
    evaluation_data.acinar_vnp->replace_global_values(1, &acinus_volume, &gid);

    // Calculate corresponding acinar strain
    const double vo = acinus_params.volume_relaxed;
    double avs_np = (acinus_volume - vo) / vo;
    evaluation_data.acinar_vnp_strain->replace_global_values(1, &avs_np, &gid);
  }
}


/*----------------------------------------------------------------------*
 |  Calculate element volume                                ismail 07/13|
 |                                                                      |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::AcinusImpl<distype>::calc_elem_volume(RedAcinus* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material> material)

{
  Discret::ReducedLung::EvaluationData& evaluation_data =
      Discret::ReducedLung::EvaluationData::get();

  // Get acinus size
  double evolnp = evaluation_data.elemVolumenp->local_values_as_span()[ele->lid()];

  // Get element global ID
  int gid = ele->id();

  // Update elem
  evaluation_data.elemVolumenp->replace_global_values(1, &evolnp, &gid);

  // calculate and update element radius
  double eRadiusnp = std::pow(evolnp * 0.75 * std::numbers::inv_pi, 1.0 / 3.0);
  evaluation_data.elemRadiusnp->replace_global_values(1, &eRadiusnp, &gid);
}

/*----------------------------------------------------------------------*
 |  Get the coupled the values on the coupling interface    ismail 07/10|
 |  of the 3D/reduced-D problem                                         |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::AcinusImpl<distype>::get_coupled_values(RedAcinus* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material> material)
{
  const int myrank = Core::Communication::my_mpi_rank(discretization.get_comm());

  // The number of nodes
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
        //     |              std::map< string            ,  double        > >  |
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

        //        std::cout<<"Return ["<<returnedBC<<"] form 1D problem to 3D SURFACE of
        //        ID["<<ID<<"]: "<<BC3d<<std::endl;

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

FOUR_C_NAMESPACE_CLOSE

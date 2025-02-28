// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_art_net_art_terminal_bc.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_of_time.hpp"

#include <stdio.h>

FOUR_C_NAMESPACE_OPEN


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Solve (public)                                          ismail 08/09|
 |                                                                      |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::Utils::solve_prescribed_terminal_bc(Core::FE::Discretization& actdis,
    const Core::Conditions::Condition* condition, Teuchos::ParameterList& params)
{
  // define BC name std::string (e.g: BC   = "flow")
  std::string BC;
  // define BC type std::string (e.g: Type = "forced")
  std::string Type;
  // Define the reflection cooficient
  double Rf = 0.0;
  // Define bc variable
  double BCin = 0.0;

  // -------------------------------------------------------------------
  // Read in the 3D parameters exported to the reduced D problem
  // -------------------------------------------------------------------
  std::shared_ptr<Teuchos::ParameterList> CoupledTo3DParams;

  // -------------------------------------------------------------------
  // Read in global time
  // -------------------------------------------------------------------
  double time = params.get<double>("total time");

  // -------------------------------------------------------------------
  // Check whether the condition is prescribed from the input file
  // or from a 3D fluid simulation
  // -------------------------------------------------------------------
  if (params.get<std::string>("Condition Name") == "ArtPrescribedCond")
  {
    // -----------------------------------------------------------------
    // Read in Condition type and name
    // -----------------------------------------------------------------
    Type = (condition->parameters().get<std::string>("type"));
    BC = (condition->parameters().get<std::string>("boundarycond"));

    // -----------------------------------------------------------------
    // Read in the bc curve information
    // -----------------------------------------------------------------
    const auto& curve = condition->parameters().get<std::vector<std::optional<int>>>("curve");
    double curvefac = 1.0;
    const auto& vals = condition->parameters().get<std::vector<double>>("VAL");

    // -----------------------------------------------------------------
    // Check whether the BC is absorbing or forced
    // -----------------------------------------------------------------
    if (Type == "absorbing")  // => without Reflection
    {
      Rf = 0.0;
    }
    else if (Type == "forced")  // => with Reflection
    {
      // If forced curve exists => Rf = curve
      if (curve[1].has_value())
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[1].value())
                       .evaluate(time);
        Rf = vals[1] * curvefac;
      }
      // else the BC is totally forced
      else
      {
        Rf = 1.0;
      }
      if (Rf < 0.0 || Rf > 1.0)
      {
        FOUR_C_THROW("forced reflection (Rf = %f) should always belong to the range :[0  1.0]", Rf);
        exit(1);
      }
    }
    else
    {
      FOUR_C_THROW("%s is not defined as a 1D artery's inlet BC type", Type.c_str());
      exit(1);
    }

    // -----------------------------------------------------------------
    // Read in the value of the applied BC
    // -----------------------------------------------------------------
    if (curve[0].has_value())
    {
      curvefac = Global::Problem::instance()
                     ->function_by_id<Core::Utils::FunctionOfTime>(curve[0].value())
                     .evaluate(time);
      BCin = vals[0] * curvefac;
    }
    else
    {
      FOUR_C_THROW("no inlet boundary condition defined!");
      exit(1);
    }
  }
  else if (params.get<std::string>("Condition Name") == "Art_redD_3D_CouplingCond")
  {
    // -------------------------------------------------------------------
    // Read in the 3D parameters exported to the reduced D problem
    // -------------------------------------------------------------------
    CoupledTo3DParams =
        params.get<std::shared_ptr<Teuchos::ParameterList>>("coupling with 3D fluid params");

    // -----------------------------------------------------------------
    // If the parameter list is empty, then something is wrong!
    // -----------------------------------------------------------------
    if (CoupledTo3DParams.get() == nullptr)
    {
      FOUR_C_THROW(
          "Cannot prescribe a boundary condition from 3D to reduced D, if the parameters passed "
          "don't exist");
      exit(1);
    }

    // -----------------------------------------------------------------
    // Read in Condition type
    // -----------------------------------------------------------------
    Type = (condition->parameters().get<std::string>("CouplingType"));

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
    map3D =
        CoupledTo3DParams->get<std::shared_ptr<std::map<std::string, double>>>("3D map of values");

    // find the applied boundary variable
    std::stringstream stringID;
    stringID << "_" << ID;
    for (std::map<std::string, double>::iterator itr = map3D->begin(); itr != map3D->end(); itr++)
    {
      std::string VariableWithId = itr->first;
      size_t found;
      found = VariableWithId.rfind(stringID.str());
      if (found != std::string::npos)
      {
        BC = std::string(VariableWithId, 0, found);
        BCin = itr->second;
        break;
      }
    }

    std::cout << "Return [" << BC << "] form 3D problem to 1D POINT of ID[" << ID << "]: " << BCin
              << std::endl;
    if (Type == "forced")
    {
      Rf = 1.0;
    }
    else if (Type == "absorbing")
    {
      Rf = 0.0;
    }
    else
    {
      FOUR_C_THROW("%s, is an unimplemented type of coupling", Type.c_str());
      exit(1);
    }
  }
  else
  {
    FOUR_C_THROW("No such condition Name");
    exit(1);
  }

  // -------------------------------------------------------------------
  // Read in the parameters asosciated with the artery terminal to whom
  // the BC will be applied
  // -------------------------------------------------------------------

  double Wfnp = 0.0;
  double Wbnp = 0.0;
  // IO = -1 if terminal is an inlet
  // IO =  1 if terminal is an outlet
  const int IO = params.get<int>("in out flag");
  const double beta = params.get<double>("artery beta");
  const double Ao = params.get<double>("artery area");
  const double dens = params.get<double>("blood density");
  const double pext = params.get<double>("external pressure");

  if (IO == -1)  // If BC is prescribed at the inlet
  {
    // read in backward characteristic wave speed
    Wbnp = params.get<double>("backward characteristic wave speed");

    // Initial backward characteristic speed at terminal 1
    const double Wbo = -4.0 * sqrt(beta / (2.0 * dens * sqrt(Ao)));
    // backward characteristic wave,
    const double Wb = (Rf * Wbnp + (1.0 - Rf) * Wbo);

    if (BC == "flow")
    {
      /*
       Prescribed Volumetric flow rate:

              /2.rho.Ao\2  /Wf - Wb\4  /Wf - Wb\
       Q    = |--------| . |-------| . |-------|
              \ beta   /   \   8   /   \   2   /

              /2.rho.Ao\2  /Wf - Wb\4  /Wf - Wb\
       f    = |--------| . |-------| . |-------|  - Q = 0
              \ beta   /   \   8   /   \   2   /

        df    /2.rho.Ao\2  /Wf - Wb\3  /5*Wf - 3*Wb\
       ---- = |--------| . |-------| . |-----------|
       dWf    \ beta   /   \   8   /   \     16    /

       The nonlinear equation: f could be solve using Newton-Raphson
       method as following:

         1- U(first guess) = Q*(Ao) => W1(first guess) = 2Q/Ao - W2
         2- Calculate df/dWf
         3- Find Wf,i+1 = Wf,i - f,i/(df/dWf),i
         4- Calculate the Error f
         5- if Tolarance is Lager than Tolarance go to step (2)
      */
      double f;
      double dfdw;
      int itrs;
      itrs = 0;
      // step 1
      Wfnp = 2.0 * BCin / Ao - Wb;
      f = std::pow(2.0 * dens * Ao / beta, 2) * pow((Wfnp - Wb) / 8.0, 4) * (Wfnp + Wb) / 2.0;
      while (fabs(f) > 0.00000001)
      {
        // step 2
        dfdw = pow(2.0 * dens * Ao / beta, 2) * pow((Wfnp - Wb) / 8.0, 3) *
               (5.0 * Wfnp + 3.0 * Wb) / 16.0;
        // step 3
        Wfnp = Wfnp - f / dfdw;
        // step 4
        f = pow(2.0 * dens * Ao / beta, 2) * pow((Wfnp - Wb) / 8.0, 4) * (Wfnp + Wb) / 2.0 - BCin;
        // an escape routine to prevent infinite loop
        itrs++;
        if (itrs >= 100)
        {
          FOUR_C_THROW(
              "Inflow boundary condition for Newton-Raphson exceeded the maximum allowed "
              "iterations");
          exit(1);
        }
      }
    }
    else if (BC == "velocity")
    {
      /*
       Prescribed Inlet Velocity
       Wf1 = 2*U - Wb1
      */
      Wfnp = 2.0 * BCin - Wb;
    }
    else if (BC == "pressure")
    {
      /*
       Prescribed Inlet Pressure
       Wf1 = Wb1 + 8*sqrt((p-pext) + beta/sqrt(Ao))/(2.rho))
      */
      Wfnp = Wb + 8.0 * sqrt((BCin - pext + beta / sqrt(Ao)) / (2.0 * dens));
    }
    else if (BC == "area")
    {
      /*
       Prescribed Inlet Area
       Wf1 = Wb1 + 8*sqrt(beta.A/(2.rho.Ao))
      */
      Wfnp = Wb + 8.0 * sqrt(beta * sqrt(BCin) / (2.0 * dens * Ao));
    }
    else if (BC == "characteristicWave")
    {
      /*
       Charachteristic wave
      */
      Wfnp = BCin;
    }
    else
    {
      FOUR_C_THROW("%s is not defined!", BC.c_str());
      exit(1);
    }
  }  // If BC is prescribed at the inlet
  else if (IO == 1)  // If BC is prescribed at the outlet
  {
    Wfnp = params.get<double>("forward characteristic wave speed");
    // Initial forward characteristic speed at terminal 2
    const double Wfo = 4.0 * sqrt(beta / (2.0 * dens * sqrt(Ao)));
    // forward characteristic wave,
    const double Wf = (Rf * Wfnp + (1.0 - Rf) * Wfo);

    if (BC == "flow")
    {
      BCin *= -1.0;
      /*
       Prescribed Volumetric flow rate:

                             2           4
                   /2.rho.Ao\   /Wf - Wb\   /Wf + Wb\
      #    Q   =   |--------| . |-------| . |-------|
                   \ beta   /   \   8   /   \   2   /

                             2           4
                   /2.rho.Ao\   /Wf - Wb\   /Wf + Wb\
       #   f   =   |--------| . |-------| . |-------|  - Q = 0
                   \ beta   /   \   8   /   \   2   /

                             2           3
           df      /2.rho.Ao\   /Wf - Wb\   /3*Wf + 5*Wb\
       #  ---- = - |--------| . |-------| . |-----------|
          dWb      \ beta   /   \   8   /   \     16    /

       The nonlinear equation: f could be solve using Newton-Raphson
       method as following:

         1- U(first guess) = Q*(Ao) => W2(first guess) = 2Q/Ao - W1
         2- Calculate df/dWb
         3- Find Wb,i+1 = Wb,i - f,i/(df/dWb),i
         4- Calculate the Error f
         5- if Tolarance is Lager than Tolarance go to step (2)

       */
      double f;
      double dfdw;
      int itrs;
      itrs = 0;
      // step 1
      Wbnp = 2.0 * BCin / Ao - Wf;
      f = std::pow(2.0 * dens * Ao / beta, 2) * pow((Wf - Wbnp) / 8.0, 4) * (Wf + Wbnp) / 2.0;
      while (fabs(f) > 0.000001)
      {
        // step 2
        dfdw = -pow(2.0 * dens * Ao / beta, 2) * pow((Wf - Wbnp) / 8.0, 3) *
               (3.0 * Wf + 5.0 * Wbnp) / 16.0;
        // step 3
        Wbnp = Wbnp - f / dfdw;
        // step 4
        f = pow(2.0 * dens * Ao / beta, 2) * pow((Wf - Wbnp) / 8.0, 4) * (Wf + Wbnp) / 2.0 - BCin;

        // a small routine to prevent infinite loop
        itrs++;
        if (itrs >= 30)
        {
          FOUR_C_THROW(
              "Inflow boundary condition for Newton-Raphson exceeded the maximum allowed "
              "iterations");
          exit(1);
        }
      }
    }
    else if (BC == "velocity")
    {
      BCin *= -1.0;
      /*
       Prescribed Inlet Velocity
       Wb2 = 2*U - Wf2
      */
      Wbnp = 2.0 * BCin - Wf;
    }
    else if (BC == "pressure")
    {
      /*
       Prescribed Inlet Pressure
       Wb2 = Wf2 - 8*sqrt((p-pext) + beta/sqrt(Ao))/(2.rho))
      */
      Wbnp = Wf - 8.0 * sqrt((BCin - pext + beta / sqrt(Ao)) / (2.0 * dens));
    }
    else if (BC == "area")
    {
      /*
       Prescribed Inlet Area
       Wb2 = Wf2 - 8*sqrt(beta.A/(2.rho.Ao))
      */
      Wbnp = Wf - 8.0 * sqrt(beta * sqrt(BCin) / (2.0 * dens * Ao));
    }
    else if (BC == "characteristicWave")
    {
      /*
       Charachteristic wave
      */
      Wbnp = BCin;
    }
  }  // If BC is prescribed at the outlet
  else
  {
    FOUR_C_THROW("IO flag must be either 1 (for outlets) or 0 (for inlets)\n");
    exit(1);
  }

  // -------------------------------------------------------------------
  // return the computed 3D values
  // -------------------------------------------------------------------
  if (params.get<std::string>("Condition Name") == "Art_redD_3D_CouplingCond")
  {
    // -----------------------------------------------------------------
    // If the parameter list is empty, then something is wrong!
    // -----------------------------------------------------------------
    if (CoupledTo3DParams.get() == nullptr)
    {
      FOUR_C_THROW(
          "Cannot prescribe a boundary condition from 3D to reduced D, if the parameters passed "
          "don't exist");
      exit(1);
    }

    // -----------------------------------------------------------------
    // Compute the variable solved by the 1D simulation to be passed to
    // the 3D simulation
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
      double c = (Wfnp - Wbnp) / 8.0;
      double A = std::pow(c, 4) * 4.0 * pow(dens * Ao / beta, 2);
      BC3d = (Wfnp + Wbnp) / 2.0 * A;
      std::cout << "1D is returning flowrate = " << BC3d << std::endl;
    }
    else if (returnedBC == "pressure")
    {
      double c = (Wfnp - Wbnp) / 8.0;
      double A = std::pow(c, 4) * 4.0 * pow(dens * Ao / beta, 2);
      BC3d = beta * (sqrt(A) - sqrt(Ao)) / Ao + pext;
    }
    else
    {
      std::string str = (condition->parameters().get<std::string>("ReturnedVariable"));
      FOUR_C_THROW("%s, is an unimplemented type of coupling", str.c_str());
      exit(1);
    }
    std::stringstream returnedBCwithId;
    returnedBCwithId << returnedBC << "_" << ID;

    std::cout << "Return [" << returnedBC << "] form 1D problem to 3D SURFACE of ID[" << ID
              << "]: " << BC3d << std::endl;
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
      FOUR_C_THROW("The 3D map for (1D - 3D coupling) has no variable (%s) for ID [%d]",
          returnedBC.c_str(), ID);
      exit(1);
    }

    // update the 1D map
    (*map1D)[returnedBCwithId.str()] = BC3d;
  }

  // -------------------------------------------------------------------
  // finally return the updated value if the characteristic speeds
  // -------------------------------------------------------------------
  params.set<double>("forward characteristic wave speed", Wfnp);
  params.set<double>("backward characteristic wave speed", Wbnp);

}  // void Arteries::Utils::SolvePrescribedTerminalBC


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  SolveReflectiveTerminal (public)                        ismail 08/09|
 |  Solves the BC value for a certain Reflective value                  |
 |                                                                      |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::Utils::solve_reflective_terminal(Core::FE::Discretization& actdis,
    const Core::Conditions::Condition* condition, Teuchos::ParameterList& params)
{
  // Define the reflection cooficient
  double Rf;

  // -------------------------------------------------------------------
  // Read in the bc curve information
  // -------------------------------------------------------------------
  const auto& curve = condition->parameters().get<std::vector<std::optional<int>>>("curve");
  double curvefac = 1.0;
  const auto& vals = condition->parameters().get<std::vector<double>>("VAL");

  // if the curve exist => Rf = val*curve(time)
  if (curve[0].has_value())
  {
    double time = params.get<double>("total time");
    curvefac = Global::Problem::instance()
                   ->function_by_id<Core::Utils::FunctionOfTime>(curve[0].value())
                   .evaluate(time);
    Rf = vals[0] * curvefac;
  }
  // else Rf = val
  else
  {
    Rf = vals[0];
  }

  // -------------------------------------------------------------------
  // Read in the parameters asosciated with the artery terminal to whom
  // the BC will be applied
  // -------------------------------------------------------------------
  const double beta = params.get<double>("artery beta");
  const double Ao = params.get<double>("artery area");
  const double dens = params.get<double>("blood density");
  const double Wfnp = params.get<double>("forward characteristic wave speed");


  // -------------------------------------------------------------------
  // Calculate the BC results
  // -------------------------------------------------------------------

  // Initial backward characteristic speed at terminal 1
  const double Wbo = -4.0 * sqrt(beta / (2.0 * dens * sqrt(Ao)));
  const double Wfo = 4.0 * sqrt(beta / (2.0 * dens * sqrt(Ao)));
  // backward characteristic wave,
  const double Wbnp = -Rf * (Wfnp - Wfo) + Wbo;

  // -------------------------------------------------------------------
  // finally return the updated value if the characteristic speeds
  // -------------------------------------------------------------------
  params.set<double>("forward characteristic wave speed", Wfnp);
  params.set<double>("backward characteristic wave speed", Wbnp);

}  // void Arteries::Utils::SolveReflectiveTerminal



//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  SolveExplWindkesselBC (public)                          ismail 08/09|
 |  Solves the windkessel boundary condition                            |
 |                                                                      |
 |  This code                                                           |
 |                                                                      |
 |                                                                      |
 |                                                                      |
 |                                                                      |
 |                                                                      |
 |                                                                      |
 |                                                                      |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::Utils::solve_expl_windkessel_bc(Core::FE::Discretization& actdis,
    const Core::Conditions::Condition* condition, Teuchos::ParameterList& params)
{
  // define BC windkessel inigration type std::string (e.g: BC   = "flow")
  std::string int_type = (condition->parameters().get<std::string>("intigrationType"));
  // define windkessel BC type std::string (e.g: Type = "forced")
  std::string wk_type = (condition->parameters().get<std::string>("windkesselType"));

  // -------------------------------------------------------------------
  // Read in the bc curve information
  // -------------------------------------------------------------------
  const auto& curve = condition->parameters().get<std::vector<std::optional<int>>>("curve");
  double curvefac = 1.0;
  const auto& vals = condition->parameters().get<std::vector<double>>("VAL");

  double Wb;
  if (int_type == "ExplicitWindkessel")
  {
    const double time = params.get<double>("total time");
    const double Wf = params.get<double>("forward characteristic wave speed");
    const double Ao = params.get<double>("artery area");
    const double beta = params.get<double>("artery beta");
    const double dens = params.get<double>("blood density");
    const double dt = params.get<double>("time step size");
    const double Pext = params.get<double>("external pressure");
    double Q = params.get<double>("terminal volumetric flow rate");
    double A = params.get<double>("terminal cross-sectional area");
    double P = beta / Ao * (sqrt(A) - sqrt(Ao)) + Pext;


    // calculate the initial wave speed
    //    const double co = sqrt(beta/(2.0*dens*Ao));

    // -----------------------------------------------------------------
    // Find the type of windkessel model
    // -----------------------------------------------------------------
    if (wk_type == "R")  // a resister with a peripheral Pressure (Pout)
    {
      FOUR_C_THROW("So far, only the 3 element windkessel model is implemented\n");
      exit(1);
      // ---------------------------------------------------------------
      // Read in the wind kessel parameters
      // ---------------------------------------------------------------
      // define a terminal resistance
      double R = 0.0;
      double Pout = 0.0;
      double dFdA = 0.0;
      // read in the reflection value
      if (curve[1].has_value())
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[1].value())
                       .evaluate(time);
        R = vals[1] * curvefac;
      }
      else
      {
        R = vals[1];
      }

      if (R < 0.0)
      {
        FOUR_C_THROW("terminal resistance must be greater or equal to zero\n");
        exit(1);
      }

      if (curve[0].has_value())
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[0].value())
                       .evaluate(time);
        Pout = vals[0] * curvefac;
      }
      else
      {
        Pout = vals[0];
      }

      // ---------------------------------------------------------------
      // Solve the nonlinear problem using Newton-Raphson scheme
      // ---------------------------------------------------------------

      const double Wo = 4.0 * sqrt(beta / (2.0 * dens * sqrt(Ao)));
      //  Calculate the residual
      double F = R * Wf * A - 4.0 * R * sqrt(beta / (2.0 * dens * Ao)) * pow(A, 5.0 / 4.0) +
                 beta / Ao * (sqrt(Ao) - sqrt(A)) + Pout - Pext;

      // loop until convergence
      int count = 0;
      while (fabs(F / (R * Ao * Wo)) > 0.0000000001)
      {
        dFdA = R * Wf - 5.0 * R * sqrt(beta / (2.0 * dens * Ao) * sqrt(A)) -
               0.5 * beta / (Ao * sqrt(A));
        A -= F / dFdA;
        F = R * Wf * A - 4.0 * R * sqrt(beta / (2.0 * dens * Ao)) * pow(A, 5.0 / 4.0) +
            beta / Ao * (sqrt(Ao) - sqrt(A)) + Pout - Pext;

        count++;
        if (count > 40)
        {
          FOUR_C_THROW("1 windkessel element (resistive) boundary condition didn't converge!");
          exit(1);
        }
      }

      // finally find evaluate Wb
      Wb = Wf - 8.0 * sqrt(beta / (2.0 * dens * Ao) * sqrt(A));

    }  // if(wk_type == "R")
    else if (wk_type == "RC")  // an RC circuit with a peripheral Pressure (Pout)
    {
      FOUR_C_THROW("So far, only the 3 element windkessel model is implemented\n");
      exit(1);

      // ---------------------------------------------------------------
      // Read in the wind kessel parameters
      // ---------------------------------------------------------------
      // define the 2 element windkessel parameters
      //      double Pout = 0.0;
      double R = 0.0;
      double C = 0.0;

      // Read in the periferal pressure of the wind kessel model
      if (curve[0].has_value())
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[0].value())
                       .evaluate(time);
        //        Pout = vals[0]*curvefac;
      }
      else
      {
        //        Pout = vals[0];
      }
      // read in the resistance value
      if (curve[1].has_value())
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[1].value())
                       .evaluate(time);
        R = vals[1] * curvefac;
      }
      else
      {
        R = vals[1];
      }
      // Read in the capacitance value
      if (curve[2].has_value())
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[2].value())
                       .evaluate(time);
        C = vals[2] * curvefac;
      }
      else
      {
        C = vals[2];
      }

      if (R <= 0.0 || C <= 0.0)
      {
        FOUR_C_THROW("terminal resistance and capacitance must be always greater than zero\n");
        exit(1);
      }

      // Calculate W2


    }  // if (wk_type == "RC")
    else if (wk_type == "RCR")  // The famous 3 element wind kessel model
    {
      // ---------------------------------------------------------------
      // Read in the wind kessel parameters
      // ---------------------------------------------------------------

      // define the 3 element windkessel parameters
      //      double Pout   = 0.0;
      double R1 = 0.0;
      double C = 0.0;
      double R2 = 0.0;
      double Poutnm = 0.0;

      // Read in the periferal pressure of the windkessel model
      // Read in Pout at time step n-1
      if (curve[0].has_value())
      {
        double t;
        if (time <= dt)
          t = time;
        else
          t = time - dt;
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[0].value())
                       .evaluate(t);
        Poutnm = vals[0] * curvefac;
      }
      else
      {
        Poutnm = vals[2];
      }
      // read in the source resistance value
      if (curve[1].has_value())
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[1].value())
                       .evaluate(time);
        R1 = vals[1] * curvefac;
      }
      else
      {
        R1 = vals[1];
      }
      // Read in the capacitance value
      if (curve[2].has_value())
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[2].value())
                       .evaluate(time);
        C = vals[2] * curvefac;
      }
      else
      {
        C = vals[2];
      }
      // read in the periferal resistance value
      if (curve[3].has_value())
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[3].value())
                       .evaluate(time);
        R2 = vals[3] * curvefac;
      }
      else
      {
        R2 = vals[3];
      }

      if (R1 < 0.0 || C <= 0.0 || R2 <= 0.0)
      {
        FOUR_C_THROW("terminal resistances and capacitance must always be greater than zero\n");
        exit(1);
      }

      // ---------------------------------------------------------------
      // Calculate Wb (backward characteristic wave speed)
      // ---------------------------------------------------------------

      // define important valriable
      double Pc = 0.0;
      double Qout = 0.0;
      double F = 0.0;
      double F_A = 0.0;
      double dFdA = 0.0;

      // find Pc at time step n
      Pc = P - Q * R1;

      // find Qout at time step n
      Qout = (Pc - Poutnm) / R2;

      // find Pc at n+1
      Pc = Pc + dt / C * (Q - Qout);

      // solving the nonlinear equation F(A) = 0 using Newton-Raphson scheme

      F = R1 * Wf * A - 4.0 * R1 * sqrt(beta / (2.0 * dens * Ao) * sqrt(pow(A, 5))) - Pext -
          beta / Ao * (sqrt(A) - sqrt(Ao)) + Pc;
      int i = 0;
      F_A = std::pow(F * sqrt(Ao) / beta + 1.0, 2);
      while (fabs(F_A) > 0.0000001)
      {
        dFdA = R1 * Wf - 5.0 * R1 * sqrt(beta / (2.0 * dens * Ao) * sqrt(A)) -
               0.5 * beta / (Ao * sqrt(A));
        A = A - F / dFdA;
        F = R1 * Wf * A - 4.0 * R1 * sqrt(beta / (2.0 * dens * Ao) * sqrt(pow(A, 5))) - Pext -
            beta / Ao * (sqrt(A) - sqrt(Ao)) + Pc;
        i++;
        F_A = std::pow(F * sqrt(Ao) / beta + 1.0, 2) - 1.0;
        if (i > 40)
        {
          FOUR_C_THROW("3 element windkessel Newton Raphson is not converging\n");
          exit(1);
        }
      }

      // finally find evaluate Wb
      Wb = Wf - 8.0 * sqrt(beta / (2.0 * dens * Ao) * sqrt(A));

    }  // if (wk_type == "RCR")
    else if (wk_type == "RCRL")  // four element windkessel model
    {
      FOUR_C_THROW("So far, only the 3 element windkessel model is implemented\n");
      exit(1);
      // ---------------------------------------------------------------
      // Read in the wind kessel parameters
      // ---------------------------------------------------------------

      // define the 4 element windkessel parameters
      //      double Pout = 0.0;
      double R1 = 0.0;
      double C = 0.0;
      double R2 = 0.0;
      double L = 0.0;

      // Read in the periferal pressure of the wind kessel model
      if (curve[0].has_value())
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[0].value())
                       .evaluate(time);
        //        Pout = vals[0]*curvefac;
      }
      else
      {
        //        Pout = vals[2];
      }
      // read in the source resistance value
      if (curve[1].has_value())
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[1].value())
                       .evaluate(time);
        R1 = vals[1] * curvefac;
      }
      else
      {
        R1 = vals[1];
      }
      // Read in the capacitance value
      if (curve[2].has_value())
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[2].value())
                       .evaluate(time);
        C = vals[2] * curvefac;
      }
      else
      {
        C = vals[2];
      }
      // read in the periferal resistance value
      if (curve[3].has_value())
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[3].value())
                       .evaluate(time);
        R2 = vals[3] * curvefac;
      }
      else
      {
        R2 = vals[3];
      }
      // read in the inductance value
      if (curve[4].has_value())
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curve[4].value())
                       .evaluate(time);
        L = vals[4] * curvefac;
      }
      else
      {
        L = vals[4];
      }

      if (R1 <= 0.0 || C <= 0.0 || R2 <= 0.0 || L <= 0.0)
      {
        FOUR_C_THROW("terminal resistance and capacitance must be always greater than zero\n");
        exit(1);
      }
      // Calculate W2

    }  // if (wk_type == "RCRL")
    else if (wk_type == "none")
    {
      FOUR_C_THROW("So far, only the 3 element windkessel model is implemented\n");
      exit(1);
    }
    else
    {
      FOUR_C_THROW("\"%s\" is not supported type of windkessel model\n", wk_type.c_str());
      exit(1);
    }

    // -----------------------------------------------------------------
    // return the calculated backward characteristic wave speed
    // -----------------------------------------------------------------
    params.set<double>("backward characteristic wave speed", Wb);
  }
  else
  {
    FOUR_C_THROW("so far windkessel BC supports only ExplicitWindkessel");
    exit(1);
  }

}  // void Arteries::Utils::SolveExplWindkesselBC

FOUR_C_NAMESPACE_CLOSE

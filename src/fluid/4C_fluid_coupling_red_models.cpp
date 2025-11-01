// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_coupling_red_models.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fluid_ele_action.hpp"

#include <math.h>
#include <stdio.h>

FOUR_C_NAMESPACE_OPEN
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Constructor (public)                                    ismail 11/09|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
FLD::Utils::FluidCouplingWrapperBase::FluidCouplingWrapperBase(
    std::shared_ptr<Core::FE::Discretization> dis_3D,
    std::shared_ptr<Core::FE::Discretization> dis_redD,
    //                                                         std::shared_ptr<red_D_time_int>
    //                                                         RedD_Time_integ,
    Core::IO::DiscretizationWriter& output, double dt_3D, double dt_redD)
    :  // call constructor for "nontrivial" objects
      discret_3d_(dis_3D),
      discret_red_d_(dis_redD),
      //  reduced_D_time_integ_(RedD_Time_integ),
      output_(output)
{
  // ---------------------------------------------------------------------
  // Read in the time step
  // ---------------------------------------------------------------------
  dt_f3_ = dt_3D;
  dt_rm_ = dt_redD;

  // check whether the "reduceD time step" is a deviser  of "3D time step"
  //    This is essential since it is not advisable to change the time
  //    step of the reduced-D problem during the simulation
  int quotient = int(floor(dt_f3_ / dt_rm_));
  double remainder = dt_f3_ - double(quotient) * dt_rm_;
  if (remainder != 0.0)
  {
    FOUR_C_THROW("\"Fluid 3D\" must have a time step multiple of that of \"reduced-D\" problem");
  }

  // ---------------------------------------------------------------------
  // Read in all conditions
  // ---------------------------------------------------------------------
  std::vector<const Core::Conditions::Condition*> couplingcond;
  discret_3d_->get_condition("Art_3D_redD_CouplingCond", couplingcond);

  std::vector<const Core::Conditions::Condition*> couplingcond2;
  discret_red_d_->get_condition("Art_redD_3D_CouplingCond", couplingcond2);

  // the number of lines of coupling boundary conditions found in the input
  // note that some of these lines could belong to the same physical condition
  // which is then marked by the same 'ConditionID'
  // The corresponding resorting of result values has to be done later
  unsigned int numcondlines = couplingcond.size();

  // ---------------------------------------------------------------------
  // Check whether the 2nd conditions are similar in number as the first
  // ---------------------------------------------------------------------
  if (numcondlines != couplingcond2.size())
  {
    FOUR_C_THROW(
        "coupled problem between reduced-D and 3D must have equal number of condition on both "
        "sides "
        "of the discretization boundaries");
  }

  if (numcondlines > 0)  // if there is at least one coupling bc
  {
    map3_dnp_ = std::make_shared<std::map<std::string, double>>();
    map3_dn_ = std::make_shared<std::map<std::string, double>>();
    map_red_dnp_ = std::make_shared<std::map<std::string, double>>();
    map_red_dn_ = std::make_shared<std::map<std::string, double>>();
    // -------------------------------------------------------------------
    // get the maximum allowable number of iterations at the boundary
    // which should be the same!
    // -------------------------------------------------------------------

    int N_iter = couplingcond[0]->parameters().get<int>("MaximumIterations");

    // -------------------------------------------------------------------
    // make sure that each coupling has two conditions of same ID
    // -------------------------------------------------------------------
    for (unsigned int i = 0; i < numcondlines; i++)
    {
      bool CondIsFine = false;
      int condid = couplingcond[i]->parameters().get<int>("ConditionID");
      for (unsigned int j = 0; j < numcondlines; j++)
      {
        int condid2 = couplingcond2[j]->parameters().get<int>("ConditionID");
        if (condid2 == condid)
        {
          CondIsFine = true;
          break;
        }
      }
      if (!CondIsFine)
      {
        FOUR_C_THROW(
            "[3D/Reduced-D COUPLING] condition [{}] is defined only on the 3D side", condid);
      }
    }

    // -------------------------------------------------------------------
    // now care for the fact that there could be more than one input line
    // belonging to the same coupling boundary condition
    // -------------------------------------------------------------------
    for (unsigned int i = 0; i < numcondlines; i++)
    {
      int condid = couplingcond[i]->parameters().get<int>("ConditionID");

      // -----------------------------------------------------------------
      // Find the second condition number which has the same id as the
      // first
      // -----------------------------------------------------------------
      unsigned int j = 0;
      for (j = 0; j < numcondlines; j++)
      {
        if (condid == couplingcond2[j]->parameters().get<int>("ConditionID"))
        {
          break;
        }
      }
      int thisN_iter = couplingcond[i]->parameters().get<int>("MaximumIterations");
      if (thisN_iter != N_iter)
        FOUR_C_THROW(
            "all maximum number of iterations on the coupling boundary between 3-D and reduced-D "
            "boundary should be the same!!!");


      // ------------------------------------------------------------------
      // allocate the coupling bc class members for every case
      // ------------------------------------------------------------------
      std::shared_ptr<FluidCouplingBc> couplingbc = std::make_shared<FluidCouplingBc>(
          discret_3d_, discret_red_d_, output_, dt_f3_, dt_rm_, condid, i, j);

      // -----------------------------------------------------------------
      // sort coupling bc's in map and test, if one condition ID appears
      // more than once. Currently this case is forbidden.
      // -----------------------------------------------------------------
      bool inserted = coup_map_3d_.insert(std::make_pair(condid, couplingbc)).second;
      if (!inserted)
        FOUR_C_THROW(
            "There are more than one 3D-to-OneD coupling condition lines with the same ID. This "
            "can not yet be handled.");
    }  // end loop over condition lines from input

    // -------------------------------------------------------------------
    // Fill the coupled variables boundary condition
    //    +-----------------------------+--------------------+
    //    |  BoundaryVariable           |  BoundaryValue     |
    //    +-----------------------------+--------------------+
    //    |         pressure1           |       300.25       |
    //    |         .......             .                    .
    //    |         flow10              .        20.43       |
    //    +-----------------------------+--------------------+
    // -------------------------------------------------------------------

    // -------------------------------------------------------------------
    // Fill the 3D coupling variable
    // -------------------------------------------------------------------
    for (unsigned int i = 0; i < numcondlines; i++)
    {
      // Get condition ID
      int id = couplingcond[i]->parameters().get<int>("ConditionID");

      // Get returned coupling variable
      std::string variable = ((couplingcond[i])->parameters().get<std::string>("ReturnedVariable"));

      // Build a new std::string from [coupling Variable name][Condition Id]
      std::stringstream VariableWithId;
      VariableWithId << variable << "_" << id;
      double value = 0.0;

      // Build the map

      map3_dnp_->insert(std::make_pair(VariableWithId.str(), value));
      map3_dn_->insert(std::make_pair(VariableWithId.str(), value));
    }

    // ------------------------------------------------------------------
    // Fill the reduced-D coupling variable
    // ------------------------------------------------------------------
    for (unsigned int i = 0; i < numcondlines; i++)
    {
      // Get condition ID
      int id = couplingcond2[i]->parameters().get<int>("ConditionID");

      // Get returned coupling variable
      std::string variable =
          ((couplingcond2[i])->parameters().get<std::string>("ReturnedVariable"));

      // Build a new std::string from [coupling Variable name][Condition Id]
      std::stringstream VariableWithId;
      VariableWithId << variable << "_" << id;
      double value = 0.0;

      // Build the map
      map_red_dnp_->insert(std::make_pair(VariableWithId.str(), value));
      map_red_dn_->insert(std::make_pair(VariableWithId.str(), value));
    }

  }  // end if there were conditions

  return;
}  // end FluidCouplingWrapperBase


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Wrap flow rate calculation                             ismail 12/09 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::Utils::FluidCouplingWrapperBase ::flow_rate_calculation(double time, double dta)
{
  // get an iterator to my map
  std::map<const int, std::shared_ptr<class FluidCouplingBc>>::iterator mapiter;

  for (mapiter = coup_map_3d_.begin(); mapiter != coup_map_3d_.end(); mapiter++)
  {
    mapiter->second->FluidCouplingBc ::flow_rate_calculation(time, dta, mapiter->first);
  }

  return;
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Wrap flow rate calculation                             ismail 12/09 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::Utils::FluidCouplingWrapperBase::pressure_calculation(double time, double dta)
{
  // get an iterator to my map
  std::map<const int, std::shared_ptr<class FluidCouplingBc>>::iterator mapiter;

  for (mapiter = coup_map_3d_.begin(); mapiter != coup_map_3d_.end(); mapiter++)
  {
    mapiter->second->FluidCouplingBc::pressure_calculation(time, dta, mapiter->first);
  }

  return;
}


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Wrap outflow boundary pressure application             ismail 12/09 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::Utils::FluidCouplingWrapperBase::apply_boundary_conditions(
    double time, double dta, double theta)
{
  // get an iterator to my map
  std::map<const int, std::shared_ptr<class FluidCouplingBc>>::iterator mapiter;

  // ---------------------------------------------------------------------
  // Read in all conditions
  // ---------------------------------------------------------------------

  // Read in the 3D coupling conditions
  std::vector<const Core::Conditions::Condition*> conds3D;
  discret_3d_->get_condition("Art_3D_redD_CouplingCond", conds3D);

  // Read in the reduced-D coupling conditions
  std::vector<const Core::Conditions::Condition*> conds_redD;
  discret_red_d_->get_condition("Art_redD_3D_CouplingCond", conds_redD);

  int condID;
  for (mapiter = coup_map_3d_.begin(); mapiter != coup_map_3d_.end(); mapiter++)
  {
    // Get condition ID
    condID = mapiter->first;

    //----------------------------------------------------------------
    // find the parameters that must be calculated and returned by
    // the 3D problem
    //----------------------------------------------------------------
    for (auto& i : conds3D)
    {
      if (i->parameters().get<int>("ConditionID") == condID)
      {
        // get returned value name from 3D boundary
        std::string variable_str = (i->parameters().get<std::string>("ReturnedVariable"));

        // concatenate the variable name with the variable id
        std::stringstream CouplingVariable;
        CouplingVariable << variable_str << "_" << condID;

        if (variable_str == "flow")
        {
          auto itr = map3_dnp_->find(CouplingVariable.str());
          if (itr == map3_dnp_->end())
          {
            FOUR_C_THROW("[3D/Reduced-D COUPLING] 3D map has no variable {} for condition [{}]",
                variable_str, condID);
          }
          (*map3_dnp_)[CouplingVariable.str()] =
              mapiter->second->FluidCouplingBc::flow_rate_calculation(time, dta, condID);
        }
        else if (variable_str == "pressure")
        {
          std::map<std::string, double>::iterator itr = map3_dnp_->find(CouplingVariable.str());
          if (itr == map3_dnp_->end())
          {
            FOUR_C_THROW("[3D/Reduced-D COUPLING] 3D map has no variable {} for condition [{}]",
                variable_str, condID);
          }
          double density = 0.0;
          double viscosity = 0.0;
          double area = 0.0;
          area = mapiter->second->FluidCouplingBc::area(density, viscosity, condID);
          (*map3_dnp_)[CouplingVariable.str()] =
              mapiter->second->FluidCouplingBc::pressure_calculation(time, dta, condID);
          (*map3_dnp_)[CouplingVariable.str()] /= area;
        }
        else
        {
          FOUR_C_THROW(
              "({}): No such coupling variable on the 3D side is defined yet", variable_str);
        }
        if (Core::Communication::my_mpi_rank(discret_3d_->get_comm()) == 0)
        {
          std::cout << "3D condition "
                    << " [" << condID << "] returns " << variable_str << " "
                    << (*map3_dnp_)[CouplingVariable.str()] << " at time " << time << std::endl;
        }
        break;
      }
    }
  }

  // -------------------------------------------------------------------
  // Solve the reduced-D problem
  // -------------------------------------------------------------------

  int NumOfSteps = int(dt_f3_ / dt_rm_);

  for (int N = 0; N < NumOfSteps; N++)
  {
    for (mapiter = coup_map_3d_.begin(); mapiter != coup_map_3d_.end(); mapiter++)
    {
      // Get condition ID
      condID = mapiter->first;
      //----------------------------------------------------------------
      // find the parameters that must be calculated and returned by
      // the reducedD problem
      //----------------------------------------------------------------
      for (unsigned int i = 0; i < conds3D.size(); i++)
      {
        if (conds_redD[i]->parameters().get<int>("ConditionID") == condID)
        {
          // get returned value name from 3D boundary
          std::string variable_str =
              (conds_redD[i]->parameters().get<std::string>("ReturnedVariable"));

          // concatenate the variable name with the variable id
          std::stringstream CouplingVariable;
          CouplingVariable << variable_str << "_" << condID;

          if (variable_str == "pressure")
          {
            auto itr = map_red_dnp_->find(CouplingVariable.str());
            if (itr == map_red_dnp_->end())
            {
              FOUR_C_THROW(
                  "[3D/Reduced-D COUPLING] reduced-D map has no variable {} for condition [{}]",
                  variable_str.c_str(), condID);
            }
            (*map_red_dnp_)[CouplingVariable.str()] = 0.0;
          }
          else if (variable_str == "flow")
          {
            std::map<std::string, double>::iterator itr =
                map_red_dnp_->find(CouplingVariable.str());
            if (itr == map_red_dnp_->end())
            {
              FOUR_C_THROW(
                  "[3D/Reduced-D COUPLING] reduced-D map has no variable {} for condition [{}]",
                  variable_str.c_str(), condID);
            }
            (*map_red_dnp_)[CouplingVariable.str()] = 0.0;
          }

          else
          {
            FOUR_C_THROW(
                "({}): No such coupling variable on the 3D side is defined yet", variable_str);
          }
          break;
        }
      }
    }
    // -----------------------------------------------------------------
    // Define a map that will have the interpolated values at the
    // reduced-D time subscale
    // -----------------------------------------------------------------
    std::shared_ptr<std::map<std::string, double>> map3D_inter_to_Red =
        std::make_shared<std::map<std::string, double>>();
    double dstep = 1.0 / double(NumOfSteps);

    // -----------------------------------------------------------------
    // Calculate the variables with in the reduced-D time subscale
    //
    //
    //    ^
    // V  |
    //    |                                      +
    //    |                                   .  .
    //    |                               o
    //    |                           .   .      .
    //    |                       //
    //    |                   .   .       .      .
    //    |               o
    //    |           .   .       .       .      .
    //    |       +
    //    |       .       .       .       .      .
    //    |
    //    |       .       .       .       .      .
    //    +-------+-------+-------//------+------+--------->
    //    |       0       1       i      N-2     N-1     Step
    //
    //
    //  ds = 1/N
    //                /  i  \                                           .
    //  V|   =  V|  + |-----|*
    //    i       1   \ N-1 /
    //
    //
    //
    // -----------------------------------------------------------------

    std::map<std::string, double>::iterator itr_sub;
    for (itr_sub = map3_dnp_->begin(); itr_sub != map3_dnp_->end(); itr_sub++)
    {
      std::string var_str = itr_sub->first;
      double var = (*map3_dn_)[var_str];
      double dvar = itr_sub->second - (*map3_dn_)[var_str];
      var = var + double(N + 1) * dstep * (dvar);

      (*map3D_inter_to_Red)[var_str] = var;
    }

    std::shared_ptr<Teuchos::ParameterList> params = std::make_shared<Teuchos::ParameterList>();
    //    params->set("3D map of values", map3_Dnp_);
    params->set("3D map of values", map3D_inter_to_Red);
    params->set("reducedD map of values", map_red_dnp_);
    double subscale_time = time - (dt_rm_ * double(NumOfSteps - N - 1));
    params->set("time", subscale_time);
    // #endif

    //    std::shared_ptr<Teuchos::ParameterList> params = Teuchos::rcp( new
    //    Teuchos::ParameterList); params->set("3D map of values", map3_Dnp_); params->set("reducedD
    //    map of values", mapRed_Dnp_);


    this->integrate(true, params);
  }

  // -------------------------------------------------------------------
  // Get the reduced-D results from all of the processors
  // if a processor doesn't have any reduced-D results then it must
  // return values equivalent to zero
  // -------------------------------------------------------------------

  for (unsigned int i = 0; i < conds3D.size(); i++)
  {
    // Get condition ID
    int ID = conds_redD[i]->parameters().get<int>("ConditionID");

    // Concatenate the returned value with the condition ID
    std::string ReturnedVariable =
        (conds_redD[i]->parameters().get<std::string>("ReturnedVariable"));

    std::stringstream VariableWithId;
    VariableWithId << ReturnedVariable << "_" << ID;

    // Get the variable with is returned
    double var = (*map_red_dnp_)[VariableWithId.str()];

    // update the variable on all of the processors
    double par_var = 0.0;
    par_var = Core::Communication::sum_all(var, discret_3d_->get_comm());
    (*map_red_dnp_)[VariableWithId.str()] = par_var;

    // Apply the boundary condition to the outlets
    if (ReturnedVariable == "pressure")
    {
      coup_map_3d_[ID]->outflow_boundary(par_var, time, dta, theta, ID);
    }
    else if (ReturnedVariable == "flow")
    {
      coup_map_3d_[ID]->inflow_boundary(par_var, time, dta, theta, ID);
    }
    else
    {
      FOUR_C_THROW(
          "Reduced-dimensional problem, returned a value of type [{}] at the condition ({})",
          ReturnedVariable.c_str(), ID);
    }
  }
  return;
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Wrap update of residual                                ismail 12/09 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::Utils::FluidCouplingWrapperBase::update_residual(Core::LinAlg::Vector<double>& residual)
{
  std::map<const int, std::shared_ptr<class FluidCouplingBc>>::iterator mapiter;

  (*map_red_dn_) = (*map_red_dnp_);
  (*map3_dn_) = (*map3_dnp_);

  for (mapiter = coup_map_3d_.begin(); mapiter != coup_map_3d_.end(); mapiter++)
  {
    mapiter->second->FluidCouplingBc::update_residual(residual);
  }

  return;
}


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Wrap update of residual                                ismail 05/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::Utils::FluidCouplingWrapperBase::evaluate_dirichlet(
    Core::LinAlg::Vector<double>& velnp, const Core::LinAlg::Map& condmap, double time)
{
  std::map<const int, std::shared_ptr<class FluidCouplingBc>>::iterator mapiter;

  (*map_red_dn_) = (*map_red_dnp_);
  (*map3_dn_) = (*map3_dnp_);

  for (mapiter = coup_map_3d_.begin(); mapiter != coup_map_3d_.end(); mapiter++)
  {
    mapiter->second->FluidCouplingBc::evaluate_dirichlet(velnp, condmap, time);
  }

  return;
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Wrap restart writing                                   ismail 12/09 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::Utils::FluidCouplingWrapperBase::write_restart(Core::IO::DiscretizationWriter& output)
{
  std::map<std::string, double>::iterator it;
  //! map of coupling variables returned by the 3-D model at time step n+1
  for (it = map3_dnp_->begin(); it != map3_dnp_->end(); it++)
  {
    std::stringstream stream;
    stream << it->first << "_3D_np";
    output.write_double(stream.str(), it->second);
  }
  //! map of coupling variables returned by the 3-D model at time step n
  for (it = map3_dn_->begin(); it != map3_dn_->end(); it++)
  {
    std::stringstream stream;
    stream << it->first << "_3D_n";
    output.write_double(stream.str(), it->second);
  }
  //! map of coupling variables returned by the reduced-D model at time step n+1
  for (it = map3_dnp_->begin(); it != map3_dnp_->end(); it++)
  {
    std::stringstream stream;
    stream << it->first << "_Red_np";
    output.write_double(stream.str(), it->second);
  }
  //! map of coupling variables returned by the reduced-D model at time step n
  for (it = map_red_dn_->begin(); it != map_red_dn_->end(); it++)
  {
    std::stringstream stream;
    stream << it->first << "_Red_n";
    output.write_double(stream.str(), it->second);
  }

  std::map<const int, std::shared_ptr<class FluidCouplingBc>>::iterator mapiter;

  for (mapiter = coup_map_3d_.begin(); mapiter != coup_map_3d_.end(); mapiter++)
  {
    mapiter->second->FluidCouplingBc::write_restart(output, mapiter->first);
  }

  return;
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Wrap restart reading                                   ismail 12/09 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::Utils::FluidCouplingWrapperBase::read_restart(Core::IO::DiscretizationReader& reader)
{
  std::map<std::string, double>::iterator it;
  //! map of coupling variables returned by the 3-D model at time step n+1
  for (it = map3_dnp_->begin(); it != map3_dnp_->end(); it++)
  {
    std::stringstream stream;
    double val = 0.0;
    stream << it->first << "_3D_np";
    val = reader.read_double(stream.str());
    it->second = val;
  }
  //! map of coupling variables returned by the 3-D model at time step n
  for (it = map3_dn_->begin(); it != map3_dn_->end(); it++)
  {
    std::stringstream stream;
    double val = 0.0;
    stream << it->first << "_3D_n";
    val = reader.read_double(stream.str());
    it->second = val;
  }
  //! map of coupling variables returned by the reduced-D model at time step n+1
  for (it = map3_dnp_->begin(); it != map3_dnp_->end(); it++)
  {
    std::stringstream stream;
    double val = 0.0;
    stream << it->first << "_Red_np";
    val = reader.read_double(stream.str());
    it->second = val;
  }
  //! map of coupling variables returned by the reduced-D model at time step n
  for (it = map_red_dn_->begin(); it != map_red_dn_->end(); it++)
  {
    std::stringstream stream;
    double val = 0.0;
    stream << it->first << "_Red_n";
    val = reader.read_double(stream.str());
    it->second = val;
  }

  std::map<const int, std::shared_ptr<class FluidCouplingBc>>::iterator mapiter;

  for (mapiter = coup_map_3d_.begin(); mapiter != coup_map_3d_.end(); mapiter++)
    mapiter->second->FluidCouplingBc::read_restart(reader, mapiter->first);

  return;
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Constructor (public)                                    ismail 12/09|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
FLD::Utils::FluidCouplingBc::FluidCouplingBc(std::shared_ptr<Core::FE::Discretization> dis_3D,
    std::shared_ptr<Core::FE::Discretization> dis_redD, Core::IO::DiscretizationWriter& output,
    double dt_3d, double dt_rm, int condid, int numcond,
    int numcond2)
    :  // call constructor for "nontrivial" objects
      condid_(condid),
      discret_3d_(dis_3D),
      discret_red_d_(dis_redD),
      output_(output)
{
  // ---------------------------------------------------------------------
  // read in all 3D to reducedD boundary conditions
  // ---------------------------------------------------------------------
  std::vector<const Core::Conditions::Condition*> couplingcond;
  dis_redD->get_condition("Art_redD_3D_CouplingCond", couplingcond);


  // ---------------------------------------------------------------------
  // get time steps size
  // ---------------------------------------------------------------------
  dt_f3_ = dt_3d;
  dt_rm_ = dt_rm;

  // ---------------------------------------------------------------------
  // get the processor ID from the communicator
  // ---------------------------------------------------------------------
  myrank_ = Core::Communication::my_mpi_rank(discret_3d_->get_comm());

  // ---------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  // ---------------------------------------------------------------------
  const Core::LinAlg::Map* dofrowmap = discret_3d_->dof_row_map();
  couplingbc_ = Core::LinAlg::create_vector(*dofrowmap, true);

  flowrate_ = 0.0;

  // ---------------------------------------------------------------------
  // Calculate alfa, the velocity correction factor:
  //
  //
  //  -------------+                     -------------+<------
  //               +<------                           +<------
  //       3D      +<------                   3D      +<------
  //    GEOMETRY   +<------                GEOMETRY   +<------
  //               +<------                           +<------
  //               +<------                           +<------
  //  -------------+                     -------------+<------
  //                   ^                                  ^
  //                   |                                  |
  //           Actual applied                      Actual calculated
  //              velocity                             velocity
  //
  //              Q|
  //               |calculated
  //      alfa = --------
  //              Q|
  //               |applied
  // ---------------------------------------------------------------------

  if ((couplingcond[numcond2]->parameters().get<std::string>("ReturnedVariable")) == "flow")
  {
    double density = 0.0;
    double viscosity = 0.0;
    double time = 0.0;
    double flowrate = this->flow_rate_calculation(time, dt_3d, condid);
    double area = this->area(density, viscosity, condid);
    if (flowrate == 0.0)
    {
      FOUR_C_THROW(
          "3D SURF condition ({}) expects a flowrate from 1D problem,\nthus it must have a "
          "Dirichlet BC of 1 in the direction of flow",
          condid);
    }
    else
    {
      alfa_ = fabs(area / flowrate);
    }
    if (myrank_ == 0)
    {
      std::cout << "Velocity correction factor cond(" << condid << ") is: " << alfa_ << std::endl;
    }
  }
  else
  {
    alfa_ = 1.0;
  }

  velocity_ = 0.0;
  return;
}


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Restart writing                                        ismail 12/09 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::Utils::FluidCouplingBc::write_restart(Core::IO::DiscretizationWriter& output, int condnum)
{
  // condnum contains the number of the present condition
  // condition Id numbers must not change at restart!!!!
  std::stringstream stream1, stream2, stream3;

  // also write vector couplingbc_
  stream3 << "couplingbc" << condnum;
  output.write_vector(stream3.str(), couplingbc_);


  // write time steps size
  output.write_double("dta_3D", dt_f3_);
  output.write_double("reduced_D_dta", dt_rm_);

  return;
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Restart reading                                        ismail 12/09 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::Utils::FluidCouplingBc::read_restart(Core::IO::DiscretizationReader& reader, int condnum)
{
  std::stringstream stream1, stream2, stream3;

  // also read vector couplingbc_
  stream3 << "couplingbc" << condnum;
  reader.read_vector(couplingbc_, stream3.str());

  // read time steps size
  dt_f3_ = reader.read_double("dta_3D");
  dt_rm_ = reader.read_double("reduced_D_dta");


  return;
}


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | Area calculation                                        ismail 12/09 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*!

*/
double FLD::Utils::FluidCouplingBc::area(double& density, double& viscosity, int condid)
{
  // fill in parameter list for subsequent element evaluation
  // there's no assembly required here
  Teuchos::ParameterList eleparams;
  eleparams.set<FLD::BoundaryAction>("action", FLD::calc_area);
  eleparams.set<double>("area", 0.0);
  eleparams.set<double>("viscosity", 0.0);
  eleparams.set<double>("density", 0.0);

  const std::string condstring("Art_3D_redD_CouplingCond");

  discret_3d_->evaluate_condition(eleparams, condstring, condid);

  double actarea = eleparams.get<double>("area");
  density = eleparams.get<double>("density");
  viscosity = eleparams.get<double>("viscosity");

  // find the lowest proc number that knows the material data
  int numproc = Core::Communication::num_mpi_ranks(discret_3d_->get_comm());
  int theproc = -1;  // the lowest proc that has the desired information
  std::vector<double> alldens(numproc);

  Core::Communication::gather_all(&density, alldens.data(), 1, discret_3d_->get_comm());
  for (int i = 0; i < numproc; i++)
    if (alldens[i] > 0.0)
    {
      theproc = i;
      break;
    }
  if (theproc < 0) FOUR_C_THROW("Something parallel went terribly wrong!");

  // do the actual communication of density ...
  Core::Communication::broadcast(&density, 1, theproc, discret_3d_->get_comm());
  // ... and viscosity
  Core::Communication::broadcast(&viscosity, 1, theproc, discret_3d_->get_comm());

  // get total area in parallel case
  double pararea = 0.0;
  pararea = Core::Communication::sum_all(actarea, discret_3d_->get_comm());

  if (myrank_ == 0)
  {
    std::cout << "3D/Reduced-D coupling condition Id: " << condid << " area = " << pararea
              << std::endl;
  }
  return pararea;
}  // FluidImplicitTimeInt::Area



//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Flow rate calculation                                  ismail 12/09 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*!
  modified by chfoe 04/08

  Calculate the flow rate across an impedance boundary surface

  Flow rates are
  (1) calculated for single element surfaces
  (2) added up over the elements of the single procs
  (3) communicated and added over the procs
  (4) and finally stored within the vector 'flowrates_'

  The vector of the flowrates holds the flow rate history of the
  very last cycle!

*/
double FLD::Utils::FluidCouplingBc::flow_rate_calculation(double time, double dta, int condid)
{
  // fill in parameter list for subsequent element evaluation
  // there's no assembly required here
  Teuchos::ParameterList eleparams;
  eleparams.set<FLD::BoundaryAction>("action", FLD::calc_flowrate);
  eleparams.set("total time", time);

  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  const Core::LinAlg::Map* dofrowmap = discret_3d_->dof_row_map();

  // create vector (+ initialization with zeros)
  std::shared_ptr<Core::LinAlg::Vector<double>> flowrates =
      Core::LinAlg::create_vector(*dofrowmap, true);
  const std::string condstring("Art_3D_redD_CouplingCond");
  discret_3d_->evaluate_condition(eleparams, flowrates, condstring, condid);

  double local_flowrate = 0.0;
  for (int i = 0; i < dofrowmap->num_my_elements(); i++)
  {
    local_flowrate += flowrates->local_values_as_span()[i];
  }

  double flowrate = 0.0;
  flowrate = Core::Communication::sum_all(local_flowrate, dofrowmap->get_comm());

  return flowrate;
}  // FluidImplicitTimeInt::flow_rate_calculation


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Pressure calculation                                   ismail 04/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*!
  Calculate the pressure across a coupling boundary surface

  Flow rates are
  (1) calculated for single element surfaces
  (2) added up over the elements of the single procs
  (3) communicated and added over the procs
  (4) divide the integrated pressure over the cross-sectional area
  (4) and finally stored within the vector 'pressures_'

  The vector of the flowrates holds the flow rate history of the
  very last cycle!

*/
double FLD::Utils::FluidCouplingBc::pressure_calculation(double time, double dta, int condid)
{
  // fill in parameter list for subsequent element evaluation
  // there's no assembly required here
  Teuchos::ParameterList eleparams;
  eleparams.set<FLD::BoundaryAction>("action", FLD::calc_pressure_bou_int);
  eleparams.set<double>("pressure boundary integral", 0.0);
  eleparams.set("total time", time);

  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  const Core::LinAlg::Map* dofrowmap = discret_3d_->dof_row_map();

  // get elemental flowrates ...
  std::shared_ptr<Core::LinAlg::Vector<double>> myStoredPressures =
      std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, 100);
  const std::string condstring("Art_3D_redD_CouplingCond");
  discret_3d_->evaluate_condition(eleparams, myStoredPressures, condstring, condid);

  // ... as well as actual total flowrate on this proc
  double actpressure = eleparams.get<double>("pressure boundary integral");

  // get total flowrate in parallel case
  double parpressure = 0.0;
  parpressure = Core::Communication::sum_all(actpressure, discret_3d_->get_comm());

  return parpressure;
}  // FluidImplicitTimeInt::pressure_calculation


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Apply outflow boundary to the coupled surface          ismail 12/09 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*

  This routine contains major parts of the following paper:

  Olufsen et al.: "Numerical Simulation and Experimental Validation of
  Blood Flow in Arteries with Structured-Tree Outflow Conditions",
  Annals of Biomedical Eingineering, Vol. 28, pp. 1281--1299, 2000.

  Basic Idea:
  (1) Evaluate convolution integral over one cycle to obtain outflow
      pressure
  (2) Apply this pressure as a Neumann-load type at the outflow boundary

*/
void FLD::Utils::FluidCouplingBc::outflow_boundary(
    double pressure, double time, double dta, double theta, int condid)
{
  // call the element to apply the pressure
  Teuchos::ParameterList eleparams;
  // action for elements
  // the reason we have Outlet impedance as action is because we don't
  // want to rewrite the implemented code
  eleparams.set<FLD::BoundaryAction>("action", FLD::Outletimpedance);

  eleparams.set("total time", time);
  eleparams.set("delta time", dta);
  eleparams.set("thsl", theta * dta);
  eleparams.set("WindkesselPressure", pressure);

  if (myrank_ == 0)
    printf(
        "3D/reduced-D coupling condition Id: %d Pressure %f at time %f\n", condid, pressure, time);


  couplingbc_->put_scalar(0.0);
  const std::string condstring("Art_3D_redD_CouplingCond");
  discret_3d_->evaluate_condition(eleparams, couplingbc_, condstring, condid);

  //  discret_3D_->ClearState();

  return;
}  // FluidImplicitTimeInt::outflow_boundary

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Apply inflow boundary to the coupled surface           ismail 05/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*
  (2) Apply this flowrate as a Dirichlet boundary at the inflow boundary

*/
void FLD::Utils::FluidCouplingBc::inflow_boundary(
    double flowrate, double time, double dta, double theta, int condid)
{
  // call the element to apply the pressure
  Teuchos::ParameterList eleparams;
  // action for elements
  // the reason we have Outlet impedance as action is because we don't
  // want to rewrite the implemented code


  if (myrank_ == 0)
    printf("3D/reduced-D coupling condition Id: %d flowrate = %f\n", condid, flowrate);


  std::vector<const Core::Conditions::Condition*> cond3D;
  discret_3d_->get_condition("Art_3D_redD_CouplingCond", cond3D);

  double density = 0.0;
  double viscosity = 0.0;

  velocity_ = flowrate / area(density, viscosity, condid_);
}



//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Update of residual vector                              ismail 12/09 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*!
 */
void FLD::Utils::FluidCouplingBc::update_residual(Core::LinAlg::Vector<double>& residual)
{
  residual.update(1.0, *couplingbc_, 1.0);
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Update dirichlet values                                ismail 05/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*!
 */
void FLD::Utils::FluidCouplingBc::evaluate_dirichlet(
    Core::LinAlg::Vector<double>& velnp, const Core::LinAlg::Map& condmap, double time)
{
  return;
  std::cout << "Evaluating Dirich!" << std::endl;
  // FOUR_C_THROW("Dirichlet coupling is not fixed yet, if you see the message then something is
  // wrong!");
  //  std::cout<<"3D discretization:"<<std::endl<<*discret_3D_<<std::endl;
  std::vector<const Core::Conditions::Condition*> conds_red;
  discret_red_d_->get_condition("Art_redD_3D_CouplingCond", conds_red);

  const Core::Conditions::Condition* cond_red = nullptr;
  for (unsigned int i = 0; i != conds_red.size(); i++)
  {
    if (conds_red[i]->parameters().get<int>("ConditionID") == condid_)
    {
      cond_red = conds_red[i];
      break;
    }
  }

  if ((cond_red->parameters().get<std::string>("ReturnedVariable")) != "flow")
  {
    return;
  }

  //  residual->Update(1.0,*couplingbc_,1.0);
  std::vector<const Core::Conditions::Condition*> conds;
  discret_3d_->get_condition("Art_3D_redD_CouplingCond", conds);

  const Core::Conditions::Condition* cond;
  for (unsigned int i = 0; i != conds.size(); i++)
  {
    if (conds[i]->parameters().get<int>("ConditionID") == condid_)
    {
      cond = conds[i];
      break;
    }
  }

  double area = 0.0;
  double density = 0.0;
  double viscosity = 0.0;

  area = this->area(density, viscosity, condid_);

  double Dflowrate = this->flow_rate_calculation(time, dt_f3_, condid_);
  alfa_ = fabs(area / Dflowrate);

  alfa_ = 1.0;
  velocity_ *= alfa_;

  std::cout << "velocity: " << std::endl;
  std::cout << "Dflowrate: " << Dflowrate << std::endl;
  std::cout << "area: " << area << std::endl;
  const std::vector<int>* nodes = cond->get_nodes();

  for (unsigned int i = 0; i < nodes->size(); i++)
  {
    int gid = (*nodes)[i];
    std::cout << "Node(" << gid << "): ";

    if (discret_3d_->have_global_node(gid))
    {
      Core::Nodes::Node* node = discret_3d_->g_node(gid);
      unsigned int numDof = discret_3d_->num_dof(node);
      std::cout << "(" << numDof << ") dof --> ";
      for (unsigned int dof = 0; dof < numDof - 1; dof++)
      {
        int dof_gid = discret_3d_->dof(node, dof);
        //        std::cout<<"("<<dof<<")+>["<<dof_gid<<"]\t";
        if (condmap.my_gid(dof_gid))
        {
          int lid = discret_3d_->dof_row_map()->lid(dof_gid);

          double val = (velnp).local_values_as_span()[lid] * velocity_;
          //        std::cout<<"Vel["<<gid<<"]: "<<(*velnp) [lid]<<std::endl;
          if ((velnp).local_values_as_span()[lid] > 1.0)
          {
            FOUR_C_THROW("coupled 3D/Reduced-D must have Dirichlet BC = 1");
          }
          std::cout << "[" << dof_gid << "]\t|" << val << "\t<-<"
                    << (velnp).local_values_as_span()[lid] << "|\t";
          velnp.replace_global_values(1, &val, &dof_gid);
        }
      }
    }
    std::cout << std::endl;
  }
}

FOUR_C_NAMESPACE_CLOSE

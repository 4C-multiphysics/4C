// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_xfluid_timeInt_std_SemiLagrange.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_cut_cutwizard.hpp"
#include "4C_cut_elementhandle.hpp"
#include "4C_cut_integrationcell.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_inpar_xfem.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_xfem_dofset.hpp"

FOUR_C_NAMESPACE_OPEN


// #define DEBUG_SEMILAGRANGE

/*------------------------------------------------------------------------------------------------*
 * Semi-Lagrange Back-Tracking algorithm constructor                                 schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
XFEM::XfluidSemiLagrange::XfluidSemiLagrange(
    XFEM::XfluidTimeintBase& timeInt,  /// time integration base class object
    const std::map<int, std::vector<Inpar::XFEM::XFluidTimeInt>>&
        reconstr_method,                      /// reconstruction map for nodes and its dofsets
    Inpar::XFEM::XFluidTimeInt& timeIntType,  /// type of time integration
    const std::shared_ptr<Core::LinAlg::Vector<double>> veln,  /// velocity at time t^n in col map
    const double& dt,                                          /// time step size
    const double& theta,                                       /// OST theta
    bool initialize                                            /// is initialization?
    )
    : XfluidStd(timeInt, reconstr_method, timeIntType, veln, dt, initialize),
      theta_default_(theta),
      rel_tol_incr_(1.0e-10),
      rel_tol_res_(1.0e-10)
{
  return;
}  // end constructor



/*------------------------------------------------------------------------------------------------*
 * Semi-Lagrangean Back-Tracking main algorithm                                      schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidSemiLagrange::compute(
    std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>& newRowVectorsn  // row
)
{
  const int nsd = 3;  // 3 dimensions for a 3d fluid element
  handle_vectors(newRowVectorsn);

  // REMARK: in case of a new FGI iteration we have values at new position
  // they are used to compute nodal gradients used for non-predictor case
  // at the end of new_iteration_prepare newVectors will be cleared to fill it with new information
  // from SL-algo
  new_iteration_prepare(newVectors_);

  switch (FGIType_)
  {
    case FRS1FGI1_:
    {
      if (myrank_ == 0)
        Core::IO::cout << "\nXFLUID_SemiLagrange::compute: case FRS1FGI1_ ..." << Core::IO::flush;
      reset_state(TimeIntData::basicStd_, TimeIntData::currSL_);
      break;
    }
    case FRSNot1_:
    {
      if (myrank_ == 0)
        Core::IO::cout << "\nXFLUID_SemiLagrange::compute: case FRSNot1_ ..." << Core::IO::flush;
      reset_state(TimeIntData::doneStd_, TimeIntData::currSL_);
      break;
    }
    case FRS1FGINot1_:
    {
      if (myrank_ == 0)
        Core::IO::cout << "\nXFLUID_SemiLagrange::compute: case FRS1FGINot1_ ..."
                       << Core::IO::flush;
      reinitialize_data();
      reset_state(TimeIntData::basicStd_, TimeIntData::currSL_);
      reset_state(TimeIntData::doneStd_, TimeIntData::currSL_);
      break;
    }
    default:
      FOUR_C_THROW("not implemented");
      break;
  }  // end switch


#ifdef DEBUG_SEMILAGRANGE
  Core::IO::cout
      << "\n----------------------------------------------------------------------------------"
         "------- ";
  Core::IO::cout << "\nReconstruct data with SEMILAGRANGEAN algorithm for " << timeIntData_->size()
                 << " dofsets ";
  Core::IO::cout
      << "\n----------------------------------------------------------------------------------"
         "------- "
      << Core::IO::endl;
#endif

  /*----------------------------------------------------*
   * first part: get the correct origin for the node    *
   * in a lagrangian point of view using a newton loop  *
   *----------------------------------------------------*/
  int counter = 0;  // loop counter to avoid infinite loops

  // loop over nodes which still don't have and may get a good startvalue
  while (true)
  {
    counter += 1;

    // counter limit because maximal max_iter Newton iterations with maximal
    // numproc processor changes per iteration (avoids infinite loop)
    if (!global_newton_finished(counter))
    {
#ifdef DEBUG_SEMILAGRANGE
      Core::IO::cout << "\n==============================================";
      Core::IO::cout << "\n CONTINUE GLOBAL NEWTON (" << counter << ") on proc " << myrank_;
      Core::IO::cout << "\n==============================================" << Core::IO::endl;
#endif

      // loop over all nodes (their std-dofsets) that have been chosen for SEMI-Lagrangean
      // reconstruction
      for (std::vector<TimeIntData>::iterator data = timeIntData_->begin();
          data != timeIntData_->end(); data++)
      {
#ifdef DEBUG_SEMILAGRANGE
        Core::IO::cout << "\n\t * STD-SL algorithm for node " << data->node_.Id();
#endif

        //------------------------------------
        // find element the initial startpoint lies in, if not done yet
        //------------------------------------
        if (data->initial_eid_ == -1)  // initial point not found yet
        {
          bool initial_elefound =
              false;  // true if the element for a point was found on the processor
          Core::Elements::Element* initial_ele =
              nullptr;  // pointer to the element where start point lies in
          Core::LinAlg::Matrix<nsd, 1> initial_xi(
              Core::LinAlg::Initialization::zero);  // local transformed coordinates of x w.r.t
                                                    // found ele

          // set the element pointer where the initial point lies in!
          element_search(initial_ele, data->initialpoint_, initial_xi, initial_elefound);

          if (!initial_elefound)
          {
            if (data->searchedProcs_ < numproc_)
            {
              // set state to nextSL to proceed with these data on the next proc
              data->state_ = TimeIntData::nextSL_;
              data->searchedProcs_ += 1;  // increment counter that the element the point lies in
                                          // has not been found on this processor
              data->initial_eid_ = -1;
            }
            else  // all procs searched -> initial point not in domain
            {
              data->state_ = TimeIntData::failedSL_;
              FOUR_C_THROW(
                  "<<< WARNING! Initial point for node {} for finding the Lagrangean origin not in "
                  "domain! >>>",
                  data->node_.id());
            }
          }
          else
          {
            // check if the initial point lies in the fluid domain
            //----------------------------------------------
            // get dofset w.r.t to old interface position
            bool step_np = false;
            std::vector<int> nds_curr;
            get_nodal_dof_set(
                initial_ele, data->initialpoint_, nds_curr, data->last_valid_vc_, step_np);

            if (nds_curr.size() == 0) FOUR_C_THROW("no valid nds-vector for initial point found");
            if (data->last_valid_vc_ != nullptr and  // not an uncut element
                data->last_valid_vc_->position() != Cut::Point::outside)
            {
              FOUR_C_THROW("initial point does not lie in the fluid");
            }

            data->initial_eid_ = initial_ele->id();
            data->startpoint_ =
                data->initialpoint_;  // start with the initial point as startpoint approximation
            data->initial_ele_owner_ = myrank_;

#ifdef DEBUG_SEMILAGRANGE
            Core::IO::cout << "\n\t\t -> Initial point found in element " << data->initial_eid_;
#endif
          }
        }

#ifdef DEBUG_SEMILAGRANGE
        Core::IO::cout << "\n\t\t -> start with start point approximation: " << data->startpoint_;
#endif

        if (data->state_ ==
            TimeIntData::currSL_)  // do not proceed when nextSL is set for current data
        {
          //------------------------------------
          // find element the initial startpoint lies in
          // if found, then get the element information
          //------------------------------------

          // Initialization
          bool elefound = false;  // true if the element for a point was found on the processor
          Core::Elements::Element* ele =
              nullptr;  // pointer to the element where start point lies in
          Core::LinAlg::Matrix<nsd, 1> xi(
              Core::LinAlg::Initialization::zero);  // local transformed coordinates of x w.r.t
                                                    // found ele
          Core::LinAlg::Matrix<nsd, 1> vel(
              Core::LinAlg::Initialization::zero);  // velocity of the start point approximation

          // search for an element where the current startpoint lies in
          element_search(ele, data->startpoint_, xi, elefound);

          //------------------------------------
          // if element is found on this proc, the newton iteration to find a better startpoint can
          // start
          //------------------------------------
          if (elefound)
          {
#ifdef DEBUG_SEMILAGRANGE
            Core::IO::cout << "\n\t\t\t ... start point approximation found in element: "
                           << ele->Id();
#endif
            //----------------------------------------------
            Core::Elements::Element* initial_ele = nullptr;

            if (!discret_->have_global_element(data->initial_eid_))
            {
              // ChangedSide check for intersections with all sides in the boundary-dis
              // this is not so efficient but should not be called so often
              // the check itself does not need information about the background elements
            }
            else
            {
              initial_ele = discret_->g_element(data->initial_eid_);
            }

            data->changedside_ = changed_side(
                ele, data->startpoint_, false, initial_ele, data->initialpoint_, false);

            //----------------------------------------------
            // get dofset w.r.t to old interface position
            bool step_np = false;
            std::vector<int> nds_curr;
            get_nodal_dof_set(ele, data->startpoint_, nds_curr, data->last_valid_vc_, step_np);


            //=========================================================
            // how to continue if no side changing comparison possible...
            //=========================================================
            if (data->changedside_)  // how to continue in newton loop, or stop the newton loop
            {
              if (!continue_for_changing_side(&*data, ele, nds_curr))
                continue;  // continue with next TimintData
            }
            else  // point did not change the side
            {
              // this set is a valid fluid set on the right side of the interface
              data->last_valid_nds_ = nds_curr;
              data->last_valid_ele_ = ele->id();
              data->nds_ = nds_curr;
            }

            if (data->changedside_ == false and nds_curr.size() == 0)
            {
              // std::cout << "node" << data->node_.Id() << std::endl;
              // std::cout << "initial point " << data->initialpoint_ << std::endl;
              // std::cout << "current startpoint " << data->startpoint_ << std::endl;
              // std::cout << "initial element Id " << data->initial_eid_ << std::endl;
              // std::cout << "initial element owner " << data->initial_ele_owner_ << std::endl;

              FOUR_C_THROW("point did not change the side, but nds = empty?!.");
            }

            //-------------------------------------------------
            // Newton loop just for sensible points
            //-------------------------------------------------

            //----------------------------------------------
            // compute the velocity at startpoint
            Core::LinAlg::Matrix<nsd, nsd> vel_deriv(
                Core::LinAlg::Initialization::zero);  // dummy matrix for velocity derivatives
            double pres = 0.0;                        // dummy variable for pressure
            Core::LinAlg::Matrix<1, nsd> pres_deriv(
                Core::LinAlg::Initialization::zero);  // dummy matrix for the pressure
                                                      // derivatives

            get_gp_values(
                ele, xi, nds_curr, *dofset_old_, vel, vel_deriv, pres, pres_deriv, veln_, false);

#ifdef DEBUG_SEMILAGRANGE
            Core::IO::cout << "\n\t\t\t ... computed velocity at start point approximation: "
                           << vel;
#endif

            //----------------------------------------------------------------------------------------
            // call the Newton loop to get the right Lagrangean origin
            // REMARK: if newton loop is converged then return the element,
            //         local coordinates and velocity at lagrangean origin
            newton_loop(ele, &*data, xi, vel, elefound);
            //----------------------------------------------------------------------------------------


            // if iteration can go on (that is when startpoint is on
            // correct interface side and iter < max_iter)
            if ((data->counter_ < newton_max_iter_) and (data->state_ == TimeIntData::currSL_))
            {
              // if element is not found in a newton step, look at another processor and so add
              // all according data to the vectors which will be sent to the next processor
              if (!elefound)
              {
                // here, a point has not been found on this proc for a second time
                data->searchedProcs_ = 2;
                data->state_ = TimeIntData::nextSL_;
              }
              else
              {
                //----------------------------------------------------------------------------------------
                // newton iteration converged to a good startpoint and so the data can be used to go
                // on
                if (data->accepted_)
                {
                  call_back_tracking(ele, &*data, xi, "standard");
                }
                else  // converged Lagrangean origin, however not accepted
                {
                  //----------------------------------------------------------------------
                  // a Lagrangean origin has been found but it does not lie in the fluid
                  //----------------------------------------------------------------------

                  Core::LinAlg::Matrix<nsd, 1> proj_x(Core::LinAlg::Initialization::zero);

                  find_nearest_surf_point(
                      data->startpoint_, proj_x, data->last_valid_vc_, "idispn");

                  proj_x = data->initialpoint_;

                  element_search(ele, proj_x, xi, elefound);

                  if (elefound)
                  {
                    // check if the projected point still remains in the same element
                    if (ele->id() == data->last_valid_ele_)
                    {
                      // TODO: check if the point lies on Boundary of the last valid vc

                      // set the new startpoint
                      data->startpoint_ = proj_x;

                      call_back_tracking(ele, &*data, xi, "standard");
                    }
                    else
                    {
                      FOUR_C_THROW(
                          "projection of startpoint lies in another element compared to the point "
                          "to be projected");
                      data->state_ = TimeIntData::failedSL_;
                    }
                  }
                  else
                    FOUR_C_THROW(
                        "element where the projection point lies in not available on this proc");
                }
                //----------------------------------------------------------------------------------------
              }
            }  // end if
            // maximum number of iterations reached or converged origin on wrong interface side
            else if (data->counter_ == newton_max_iter_ or (!data->accepted_))
            {
              // do not use the lagrangian origin since this case is strange and potential dangerous
              data->state_ = TimeIntData::failedSL_;

#ifdef DEBUG_SEMILAGRANGE
              Core::IO::cout
                  << " <<< WARNING: newton iteration to find start value did not converge! >>>"
                  << Core::IO::endl;
#endif
            }  // not converged in max_iter
          }  // if(elefound)
          // if element is not found, look at another processor and so add all
          // according data to the vectors which will be sent to the next processor
          else  // (!elefound)
          {
            if (data->searchedProcs_ < numproc_)
            {
              data->state_ = TimeIntData::nextSL_;
              data->searchedProcs_ += 1;
            }
            else  // all procs searched -> point not in domain
            {
              data->state_ = TimeIntData::failedSL_;
              Core::IO::cout << " <<< WARNING! Lagrangian start point not in domain! >>>"
                             << Core::IO::endl;
            }
          }  // end if elefound
        }
      }  // end loop over all nodes stored in TiminitData marked for Semilagrangean algorithm
    }  // !global_newton_finished(counter)
    else
    {
      // reset the state to failed
      reset_state(TimeIntData::currSL_, TimeIntData::failedSL_);
    }

    //===================================================================
    //                     PARALLEL COMMUNICATION

    // export nodes and according data for which the startpoint isn't still found (next_ vector) to
    // next proc
    bool procDone = global_newton_finished();

#ifdef DEBUG_SEMILAGRANGE
    if (procDone)
    {
      Core::IO::cout << "\n==============================================";
      Core::IO::cout << "\n FINISHED GLOBAL NEWTON on proc " << myrank_;
      Core::IO::cout << "\n==============================================" << Core::IO::endl;
    }
#endif

    export_iter_data(procDone);

    // convergencecheck: procfinished == 1 just if all procs have finished
    if (procDone)
    {
#ifdef DEBUG_SEMILAGRANGE
      Core::IO::cout << "\n-------------------------------------------------";
      Core::IO::cout << "\n\t\t\t !!!!!!!!!! procDone!!!!!!!!";
      Core::IO::cout << "\n-------------------------------------------------" << Core::IO::endl;
#endif
      break;
    }

    //===================================================================

  }  // end while loop over searched nodes


  /*-----------------------------------------------------------------------------*
   * second part: get sensible startvalues for nodes where the algorithm failed, *
   * using another algorithm, and combine the "Done" and the "Failed" - vectors  *
   *-----------------------------------------------------------------------------*/
  if (FGIType_ == FRSNot1_)  // failed nodes stay equal after the first computation
    clear_state(TimeIntData::failedSL_);
  else
  {
    //===================================================================
    //                     PARALLEL COMMUNICATION
    export_alternativ_algo_data();  // export data of failed nodes
    //===================================================================

    get_data_for_not_converged_nodes();  // compute final data for failed nodes
  }



  /*-----------------------------------------------------------*
   * third part: set the computed values into the state vector *
   *-----------------------------------------------------------*/
  // send the computed startvalues for every node which needs
  // new start data to the processor where the node is
  export_final_data();

  // now every proc has the whole data for the nodes and so the data can be set to the right place
  // now
  set_final_data();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (counter > 8 * numproc_)  // too much loops shouldnt be if all this works
    std::cout << "WARNING: semiLagrangeExtrapolation seems to run an infinite loop!" << std::endl;
#endif
}  // end semiLagrangeExtrapolation



/*------------------------------------------------------------------------------------------------*
 * Main Newton loop of the Semi-Lagrangian Back-Tracking algorithm                   schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidSemiLagrange::newton_loop(Core::Elements::Element*& ele,  /// pointer to element
    TimeIntData* data,                                                     /// current data
    Core::LinAlg::Matrix<3, 1>& xi,   /// local coordinates of point
    Core::LinAlg::Matrix<3, 1>& vel,  /// velocity at current point
    bool& elefound                    /// is element found ?
)
{
#ifdef DEBUG_SEMILAGRANGE
  Core::IO::cout << "\n\t\t -> XFLUID_SemiLagrange::newton_loop" << Core::IO::endl;
#endif

  const int nsd = 3;  // 3 dimensions for a 3d fluid element

  // Initialization
  Core::LinAlg::Matrix<nsd, 1> residuum(
      Core::LinAlg::Initialization::zero);  // residuum of the newton iteration
  Core::LinAlg::Matrix<nsd, 1> incr(
      Core::LinAlg::Initialization::zero);  // increment of the newton system

  // coordinates of endpoint of Lagrangian characteristics
  Core::LinAlg::Matrix<nsd, 1> origNodeCoords(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < nsd; i++) origNodeCoords(i) = data->node_.x()[i] + data->dispnp_(i);

  //-------------------------------------------------------
  // initialize residual (Theta = 0 at predictor step)
  residuum.clear();

  // data->vel_ = vel^(n+1) for FGI>1, vel = vel^n
  residuum.update((1.0 - theta(data)), vel, theta(data), data->vel_);  // dt*v(data->startpoint_)
  residuum.update(1.0, data->startpoint_, -1.0, origNodeCoords,
      dt_);  // R = data->startpoint_ - data->node_ + dt*v(data->startpoint_)

  //==========================================================
  // (re-)start the Newton-loop on this processor
  //==========================================================
  while (data->counter_ < newton_max_iter_)  // newton loop
  {
#ifdef DEBUG_SEMILAGRANGE
    Core::IO::cout << "\n\t\t\t newton_loop(" << data->counter_ << "): residuum " << residuum
                   << Core::IO::endl;
#endif

    data->counter_ += 1;

    //-------------------------------------
    // compute a new Newton iteration
    //-------------------------------------
    newton_iter(ele, data, xi, residuum, incr, elefound);


    //=========================================================
    // continue on this proc if the new startpoint approximation is also on this processor
    //=========================================================
    if (elefound)  // element of data->startpoint_ at this processor
    {
#ifdef DEBUG_SEMILAGRANGE
      Core::IO::cout << "\n\t\t\t\t ... elefound " << ele->Id();
#endif

      Core::Elements::Element* initial_ele = nullptr;

      if (!discret_->have_global_element(data->initial_eid_))
      {
        // TODO: modify the ChangedSide check for intersections with all sides in the boundary-dis
        // this is not so efficient but should not be called not so often
        // the check itself does not need information about the background elements
        //               FOUR_C_THROW("element where initial point lies in not available on proc {},
        //              no ChangedSide comparison possible", myrank_);
      }
      else
      {
        initial_ele = discret_->g_element(data->initial_eid_);
      }

      data->changedside_ =
          changed_side(ele, data->startpoint_, false, initial_ele, data->initialpoint_, false);

      bool step_np = false;  // new timestep or old timestep
      std::vector<int> nds_curr;
      get_nodal_dof_set(ele, data->startpoint_, nds_curr, data->last_valid_vc_, step_np);

      //=========================================================
      // how to continue if no side changing comparison possible...
      //=========================================================
      if (data->changedside_)  // how to continue in newton loop, or stop the newton loop
      {
        if (!continue_for_changing_side(data, ele, nds_curr)) break;
      }
      else  // point did not change the side
      {
        // this set is a valid fluid set on the right side of the interface
        data->last_valid_nds_ = nds_curr;
        data->last_valid_ele_ = ele->id();
        data->nds_ = nds_curr;
      }

      //=========================================================
      // continue if the semi-lagrangean origin approximation did not change the side w.r.t the
      // initial start point
      //=========================================================

      //-------------------------------------------------------
      // compute the velocity at startpoint
      Core::LinAlg::Matrix<nsd, nsd> vel_deriv(Core::LinAlg::Initialization::zero);  // dummy matrix
      double pres = 0.0;  // dummy variable for pressure
      Core::LinAlg::Matrix<1, nsd> pres_deriv(
          Core::LinAlg::Initialization::zero);  // dummy matrix for the pressure derivatives

      get_gp_values(
          ele, xi, nds_curr, *dofset_old_, vel, vel_deriv, pres, pres_deriv, veln_, false);


#ifdef DEBUG_SEMILAGRANGE
      Core::IO::cout << "\n\t\t\t ... computed velocity at start point approximation: " << vel;
#endif

      //-------------------------------------------------------
      // reset residual
      residuum.clear();
      residuum.update(
          (1.0 - theta(data)), vel, theta(data), data->vel_);  // dt*v(data->startpoint_)
      residuum.update(1.0, data->startpoint_, -1.0, origNodeCoords,
          dt_);  // R = data->startpoint_ - data->movNode_ + dt*v(data->startpoint_)

      //-------------------------------------------------------
      // convergence criterion
      if (data->startpoint_.norm2() > 1e-3)
      {
        if (incr.norm2() / data->startpoint_.norm2() < rel_tol_incr_ &&
            residuum.norm2() / data->startpoint_.norm2() < rel_tol_res_)
        {
          if (data->changedside_ == false)
          {
            data->accepted_ = true;

#ifdef DEBUG_SEMILAGRANGE
            Core::IO::cout << "\n\t*******************************";
            Core::IO::cout << "\n\t    newton_loop: converged!";
            Core::IO::cout << "\n\t  LAGRANGEAN ORIGIN ACCEPTED";
            Core::IO::cout << "\n\t*******************************" << Core::IO::endl;
#endif
          }
          else
          {
            data->accepted_ = false;

#ifdef DEBUG_SEMILAGRANGE
            Core::IO::cout << "\n\t*******************************";
            Core::IO::cout << "\n\t    newton_loop: converged!";
            Core::IO::cout << "\n\t  LAGRANGEAN ORIGIN NOT (!!!) ACCEPTED";
            Core::IO::cout << "\n\t*******************************" << Core::IO::endl;
#endif
          }

          break;
        }
      }
      else
      {
        if (incr.norm2() < rel_tol_incr_ && residuum.norm2() < rel_tol_res_)
        {
          if (data->changedside_ == false)
          {
            data->accepted_ = true;

#ifdef DEBUG_SEMILAGRANGE
            Core::IO::cout << "\n\t*******************************";
            Core::IO::cout << "\n\t    newton_loop: converged!";
            Core::IO::cout << "\n\t  LAGRANGEAN ORIGIN ACCEPTED";
            Core::IO::cout << "\n\t*******************************" << Core::IO::endl;
#endif
          }
          else
          {
            data->accepted_ = false;

#ifdef DEBUG_SEMILAGRANGE
            Core::IO::cout << "\n\t*******************************";
            Core::IO::cout << "\n\t    newton_loop: converged!";
            Core::IO::cout << "\n\t  LAGRANGEAN ORIGIN NOT (!!!) ACCEPTED";
            Core::IO::cout << "\n\t*******************************" << Core::IO::endl;
#endif
          }

          break;
        }
      }

    }  // end if elefound is true
    //=========================================================
    // stop Newton loop on this proc since the new startpoint approximation is not on this processor
    //=========================================================
    else  // element of data->startpoint_ not at this processor
    {
#ifdef DEBUG_SEMILAGRANGE
      Core::IO::cout
          << "\t <<< !!! element not found on this proc -> stop Newton loop on this proc !!! >>>"
          << Core::IO::endl;
#endif
      break;  // stop newton loop on this proc
    }
  }  // end while Newton loop



#ifdef DEBUG_SEMILAGRANGE
  // did newton iteration converge?
  if (data->counter_ == newton_max_iter_)
  {
    Core::IO::cout
        << "\t <<< WARNING: newton iteration for finding start value not converged for point "
           "!!! >>>"
        << Core::IO::endl;
  }
#endif


}  // end function newton_loop



/*------------------------------------------------------------------------------------------------*
 * One Newton iteration of the Semi-Lagrangian Back-Tracking algorithm               schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidSemiLagrange::newton_iter(
    Core::Elements::Element*& ele,         /// pointer to element to be updated
    TimeIntData* data,                     /// current data to be updated
    Core::LinAlg::Matrix<3, 1>& xi,        /// local coordinates w.r.t ele to be updated
    Core::LinAlg::Matrix<3, 1>& residuum,  /// residual for semilagrangean backtracking
    Core::LinAlg::Matrix<3, 1>& incr,  /// computed increment for lagrangean origin to be updated
    bool& elefound                     /// element found ?
)
{
#ifdef DEBUG_SEMILAGRANGE
  Core::IO::cout << "\n\t\t\t\t ... new iteration";
#endif

  const int nsd = 3;  // 3 dimensions for a 3d fluid element

  // Initialization
  Core::LinAlg::Matrix<nsd, 1> vel_dummy(
      Core::LinAlg::Initialization::zero);  // dummy matrix for the velocity
  Core::LinAlg::Matrix<nsd, nsd> vel_deriv(
      Core::LinAlg::Initialization::zero);  // matrix for the velocity derivatives
  double pres_dummy = 0.0;                  // dummy variable for pressure
  Core::LinAlg::Matrix<1, nsd> pres_deriv(
      Core::LinAlg::Initialization::zero);  // dummy matrix for the pressure derivatives

  Core::LinAlg::Matrix<nsd, nsd> sysmat(
      Core::LinAlg::Initialization::zero);  // matrix for the newton system

  // compute the velocity derivatives at startpoint
  get_gp_values(
      ele, xi, data->nds_, *dofset_old_, vel_dummy, vel_deriv, pres_dummy, pres_deriv, veln_, true);

  // build sysmat
  // JAC = I + dt(1-theta)*velDerivXY
  sysmat.update((1.0 - theta(data)) * dt_, vel_deriv);  // dt*(1-theta)dN/dx

  for (int i = 0; i < nsd; i++) sysmat(i, i) += 1.0;  // I + dt*velDerivXY

  // inverse system Matrix built
  sysmat.invert();


  // solve Newton iteration
  incr.clear();
  incr.multiply(-1.0, sysmat, residuum);  // incr = -Systemmatrix^-1 * residuum

  // update iteration
  for (int i = 0; i < nsd; i++) data->startpoint_(i) += incr(i);

#ifdef DEBUG_SEMILAGRANGE
  Core::IO::cout << "\n\t\t\t\t ... new approximate startvalue is " << data->startpoint_(0) << " "
                 << data->startpoint_(1) << " " << data->startpoint_(2) << Core::IO::endl;
#endif

  // find the element the new approximation lies in
  element_search(ele, data->startpoint_, xi, elefound);


  return;
}  // end function newton_iter


/*------------------------------------------------------------------------------------------------*
 * check if newton iteration searching for the Lagrangian origin has finished        schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
bool XFEM::XfluidSemiLagrange::global_newton_finished(int counter) const
{
  if (counter == newton_max_iter_ * numproc_) return true;  // maximal number of iterations reached
  for (std::vector<TimeIntData>::iterator data = timeIntData_->begin(); data != timeIntData_->end();
      data++)
  {
    if ((data->state_ == TimeIntData::currSL_) or (data->state_ == TimeIntData::nextSL_))
    {
      return false;  // one node requires more data
    }
  }
  return true;  // if no more node requires data, we are done
}


/*------------------------------------------------------------------------------------------------*
 * Decide how or if to continue when the startpoint approximation changed the side  schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
bool XFEM::XfluidSemiLagrange::continue_for_changing_side(
    TimeIntData* data,             ///< current data to be updated
    Core::Elements::Element* ele,  ///< pointer to element the current point lies in
    std::vector<int>&
        nds_curr  ///< nds-vector of current volumecell the current startpoint approximation lies in
)
{
  //--------------------------------------------------------------------------------------
  // ALTERNATIVE: CONTINUE NEWTON-ALGO when startvalue changed side during newton
  // maybe the newton turns back to the right interface side


  if (nds_curr == data->last_valid_nds_ and ele->id() == data->last_valid_ele_)
  {
    // the new newton step is within the same element and has the same nds-vector(same cell set)
    // but changed the side, then we are at the tip of a thin structure -> failed
#ifdef DEBUG_SEMILAGRANGE
    Core::IO::cout
        << "\n "
           "----------------------------------------------------------------------------------"
           "-------------";
    Core::IO::cout
        << "\n <<< Startpoint approximation moved within one fld-vc, but the trace intersects "
           "the side >>>";
    Core::IO::cout << "\n                          CHANGED SIDE ";
    Core::IO::cout
        << "\n Newton stopped! We are at the tip of a thin structure! -> leave newton loop >>>";
    Core::IO::cout
        << "\n "
           "----------------------------------------------------------------------------------"
           "-------------"
        << Core::IO::endl;
#endif
    data->state_ = TimeIntData::failedSL_;

    return false;
  }
  else if (nds_curr != data->last_valid_nds_ and ele->id() == data->last_valid_ele_)
  {
    // the new newton step is within the same element but has a different nds-vector
    // we are within the structure or changed the side completely
    // it can be that the newton iterations goes back on the right side
    // -> continue within this element using the last valid nds-vector

    nds_curr = data->last_valid_nds_;

    return true;
  }
  else if (ele->id() != data->last_valid_ele_)
  {
    // within the newton the element and the side have changed
#ifdef DEBUG_SEMILAGRANGE
    Core::IO::cout
        << " <<< Newton for lagrangian origin can not be continued, iteration changed the "
           "side and the element! -> leave newton loop >>>"
        << Core::IO::endl;
#endif
    data->state_ = TimeIntData::failedSL_;

    return false;
  }
  else
    FOUR_C_THROW("case not possible");

  return false;
}


/*------------------------------------------------------------------------------------------------*
 * Computing final data where Semi-Lagrangian approach failed                        schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidSemiLagrange::get_data_for_not_converged_nodes()
{
  const int nsd = 3;  // 3 dimensions for a 3d fluid element

  // remark: all data have to be sent to the processor where
  //         the startpoint lies before calling this function
  for (std::vector<TimeIntData>::iterator data = timeIntData_->begin(); data != timeIntData_->end();
      data++)
  {
    if (data->state_ == TimeIntData::failedSL_)
    {
#ifdef DEBUG_SEMILAGRANGE
      Core::IO::cout << "P " << myrank_ << " WARNING: failedSL -> alternative algo!"
                     << Core::IO::endl;
      Core::IO::cout << "P " << myrank_ << " node " << data->node_.Id() << Core::IO::endl;
      Core::IO::cout << "P " << myrank_ << " use initial point: " << data->initialpoint_
                     << Core::IO::endl;
#endif

      // Initialization
      Core::Elements::Element* ele =
          nullptr;  // pointer to the element where pseudo-Lagrangian origin lies in
      Core::LinAlg::Matrix<nsd, 1> xi(
          Core::LinAlg::Initialization::zero);  // local coordinates of pseudo-Lagrangian origin
      Core::LinAlg::Matrix<nsd, 1> vel(
          Core::LinAlg::Initialization::zero);  // velocity at pseudo-Lagrangian origin
      bool elefound = false;  // true if an element for a point was found on the processor

      // search for an element where the current startpoint lies in
      element_search(ele, data->initialpoint_, xi, elefound);

      // if found, give out all data at the startpoint
      if (elefound)
      {
        bool step_np = false;  // data w.r.t old interface position
        get_nodal_dof_set(ele, data->initialpoint_, data->nds_, data->last_valid_vc_, step_np);
      }
      else  // possibly slave node looked for element of master node or vice versa
      {
        FOUR_C_THROW("element not found");
      }

      //-------------------------------------------------------------------
      // call the back Tracking computation based on the initial point
      // which is a rough approximation of the lagrangian origin
      call_back_tracking(ele, &*data, xi, static_cast<const char*>("failing"));
      //-------------------------------------------------------------------

    }  // if(failedSL_)
  }  // end loop over nodes

  return;
}  // end get_data_for_not_converged_nodes


/*------------------------------------------------------------------------------------------------*
 * rewrite data for new computation                                                  schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidSemiLagrange::new_iteration_prepare(
    std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> newRowVectors)
{
  for (std::vector<TimeIntData>::iterator data = timeIntData_->begin(); data != timeIntData_->end();
      data++)
  {
    data->searchedProcs_ = 1;
    data->counter_ = 0;
    data->velValues_.clear();
    data->presValues_.clear();
  }

  new_iteration_nodal_data(newRowVectors);  // data at t^n+1 not used in predictor
  newRowVectors.clear();                    // no more needed
}



/*------------------------------------------------------------------------------------------------*
 * compute Gradients at side-changing nodes                                          schott 04/13 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidSemiLagrange::new_iteration_nodal_data(
    std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> newRowVectors)
{
  const int nsd = 3;

  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> newColVectors;

  for (size_t index = 0; index < newRowVectors.size(); index++)
  {
    const Core::LinAlg::Map* newdofcolmap = discret_->dof_col_map();

    std::shared_ptr<Core::LinAlg::Vector<double>> tmpColVector =
        std::make_shared<Core::LinAlg::Vector<double>>(*newdofcolmap, true);
    newColVectors.push_back(tmpColVector);
    Core::LinAlg::export_to(*newRowVectors[index], *newColVectors[index]);
  }

  for (std::vector<TimeIntData>::iterator data = timeIntData_->begin(); data != timeIntData_->end();
      data++)
  {
    if (data->type_ == TimeIntData::predictor_)
      continue;  // no info at new interface position required

    if (data->type_ != TimeIntData::predictor_)
      FOUR_C_THROW(
          "this function is used for the first time here, check the compute gradient "
          "functionality, check also the parallel export!");

    Core::Nodes::Node* node = &data->node_;

    //----------------------------------------------------------
    // Reconstruct nodal gradients for all vectors
    //----------------------------------------------------------

    // node velocities of the element nodes for the fields that should be changed
    std::vector<Core::LinAlg::Matrix<nsd, nsd>> avg_nodevelgraddata(
        newColVectors.size(), Core::LinAlg::Matrix<nsd, nsd>(Core::LinAlg::Initialization::zero));

    // node pressures of the element nodes for the data that should be changed
    std::vector<Core::LinAlg::Matrix<1, nsd>> avg_nodepresgraddata(
        newColVectors.size(), Core::LinAlg::Matrix<1, nsd>(Core::LinAlg::Initialization::zero));


    // determine the elements used for the nodal gradient computation
    // determine the corresponding nodal dofset vectors used for averaging the nodal gradients
    std::vector<Core::Elements::Element*> eles_avg;
    std::vector<std::vector<int>> eles_avg_nds;


    Cut::Node* n = wizard_new_->get_node(node->id());

    if (n != nullptr)
    {
      // get the nodal dofset w.r.t the Lagrangean origin
      const std::set<Cut::plain_volumecell_set, Cut::Cmp>& cellset =
          n->get_nodal_dof_set(0)->volume_cell_composite();  // always the standard dofset

      // get for each adjacent element contained in the nodal dofset the first vc, its parent
      // element is used for the reconstruction REMARK: adjacent elements for that no elementhandle
      // exists in the cut won't be found here
      for (std::set<Cut::plain_volumecell_set, Cut::Cmp>::const_iterator cellset_it =
               cellset.begin();
          cellset_it != cellset.end(); cellset_it++)
      {
        // the first vc representing the set
        Cut::VolumeCell* vc = *((*cellset_it).begin());
        int peid = vc->parent_element()->get_parent_id();

        if (!discret_->have_global_element(peid))
          FOUR_C_THROW("element {} for averaging not on proc {}", peid, myrank_);

        // get the element
        Core::Elements::Element* e = discret_->g_element(peid);
        // get the vc's nds vector
        const std::vector<int> e_nds = vc->nodal_dof_set();

        // add the element and the nds vector
        eles_avg.push_back(e);
        eles_avg_nds.push_back(e_nds);
      }
    }

    // check all surrounding elements
    int numele = node->num_element();
    Core::Elements::Element** eles = node->elements();

    // add surrounding std uncut elements for that no elementhandle is available
    for (int i = 0; i < numele; i++)
    {
      Cut::ElementHandle* eh = wizard_new_->get_element(eles[i]);

      if (eh != nullptr)
        continue;  // element and the right nds-vec should have been found using the for-loop before

      // if we are here, then the element is a standard uncut element
      // and it is ensured that it has not been added to eles_avg yet
      std::vector<int> std_nds(eles[i]->num_node(), 0);

      eles_avg.push_back(eles[i]);
      eles_avg_nds.push_back(std_nds);
    }

    if (eles_avg.size() == 0) FOUR_C_THROW("there is no element for averaging");

    // compute the nodal gradients for velocity/acc and pressure component
    compute_nodal_gradient(newColVectors, node, eles_avg, eles_avg_nds, *dofset_new_,
        avg_nodevelgraddata, avg_nodepresgraddata);


    data->velDeriv_ = avg_nodevelgraddata;
    data->presDeriv_ = avg_nodepresgraddata;

    //----------------------------------------------------------
    // get the transport velocity at t^(n+1) at the node
    //----------------------------------------------------------

    //-------------------------------------------------------
    // get node location vector, dirichlet flags and ownerships (discret, nds, la, doDirichlet)
    std::vector<int> lm;
    std::vector<int> dofs;

    dofset_new_->dof(dofs, node, 0);  // dofs for standard dofset

    for (int j = 0; j < 4; ++j)
    {
      lm.push_back(dofs[j]);
    }

    //-------------------------------------------------------
    // the first vector contains the velocity information
    Core::LinAlg::Matrix<3, 1> nodevel(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<1, 1> nodepre(Core::LinAlg::Initialization::zero);
    extract_nodal_values_from_vector<1>(nodevel, nodepre, newColVectors[0], lm);

    data->vel_ = nodevel;

    Core::LinAlg::Matrix<3, 1> nodedispnp(Core::LinAlg::Initialization::zero);
    if (dispnp_ != nullptr)  // is alefluid
    {
      //------------------------------------------------------- add ale disp
      // get node location vector, dirichlet flags and ownerships (discret, nds, la, doDirichlet)

      Core::LinAlg::Matrix<1, 1> nodepredummy(Core::LinAlg::Initialization::zero);
      extract_nodal_values_from_vector<1>(nodedispnp, nodepredummy, dispnp_, lm);
    }

    data->dispnp_ = nodedispnp;
  }
  //----------------------------------------------------------------------------------
}



/*------------------------------------------------------------------------------------------------*
 * reinitialize data for new computation                                         winklmaier 11/11 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidSemiLagrange::reinitialize_data()
{
  FOUR_C_THROW("adapt implementation of this function");
  FOUR_C_THROW("adapt, how to get nds_np?");

  //  int nds_np = -1;
  //
  //  std::cout << "in SemiLagrange::reinitialize_data" << std::endl;
  //  const int nsd = 3; // dimension
  //  Core::LinAlg::Matrix<nsd,1> dummyStartpoint; // dummy startpoint for comparison
  //  for (int i=0;i<nsd;i++) dummyStartpoint(i) = 777.777;
  //
  //  // fill curr_ structure with the data for the nodes which changed interface side
  //  for (int lnodeid=0; lnodeid<discret_->NumMyColNodes(); lnodeid++)  // loop over processor
  //  nodes
  //  {
  //    Core::Nodes::Node* currnode = discret_->lColNode(lnodeid);
  //    // node on current processor which changed interface side
  //    if ((currnode->Owner() == myrank_) &&
  //        (interfaceSideCompare((*phinp_)[lnodeid],(*phinpi_)[lnodeid])==false))
  //    {
  //      if (interfaceSideCompare((*phinp_)[lnodeid],(*phin_)[lnodeid]) == false) // real new side
  //        timeIntData_->push_back(TimeIntData(
  //            *currnode,
  //            nds_np,
  //            Core::LinAlg::Matrix<nsd,1>(Core::LinAlg::Initialization::zero),
  //            std::vector<Core::LinAlg::Matrix<nsd,nsd>
  //            >(oldVectors_.size(),Core::LinAlg::Matrix<nsd,nsd>(Core::LinAlg::Initialization::zero)),
  //            std::vector<Core::LinAlg::Matrix<1,nsd>
  //            >(oldVectors_.size(),Core::LinAlg::Matrix<1,nsd>(Core::LinAlg::Initialization::zero)),
  //            dummyStartpoint,
  ////            (*phinp_)[lnodeid],
  //            1,
  //            0,
  //            std::vector<int>(1,-1),
  //            std::vector<int>(1,-1),
  //            INFINITY,
  //            TimeIntData::predictor_));
  //      else // other side than last FSI, but same side as old solution at last time step
  //      {
  //        for (std::vector<TimeIntData>::iterator data=timeIntData_->begin();
  //            data!=timeIntData_->end(); data++)
  //        {
  //          const int nodeid = currnode->Id();
  //
  //          // 1) delete data
  //          if (data->node_.Id()==nodeid)
  //            timeIntData_->erase(data);
  //
  //          // 2) reset value of old solution
  //          // get nodal velocities and pressures with help of the field set of node
  //          const std::set<XFEM::FieldEnr>& fieldEnrSet(newdofman_->getNodeDofSet(nodeid));
  //          for (std::set<XFEM::FieldEnr>::const_iterator fieldenr = fieldEnrSet.begin();
  //              fieldenr != fieldEnrSet.end();++fieldenr)
  //          {
  //            const DofKey dofkey(nodeid, *fieldenr);
  //            const int newdofpos = newNodalDofRowDistrib_.find(dofkey)->second;
  //            const int olddofpos = oldNodalDofColDistrib_.find(dofkey)->second;
  //            switch (fieldenr->getEnrichment().Type())
  //            {
  //            case XFEM::Enrichment::typeJump :
  //            case XFEM::Enrichment::typeKink : break; // just standard dofs
  //            case XFEM::Enrichment::typeStandard :
  //            case XFEM::Enrichment::typeVoid :
  //            {
  //              for (size_t index=0;index<newVectors_.size();index++) // reset standard dofs due
  //              to old solution
  //                (*newVectors_[index])[newdofrowmap_.LID(newdofpos)] =
  //                    (*oldVectors_[index])[olddofcolmap_.LID(olddofpos)];
  //              break;
  //            }
  //            case XFEM::Enrichment::typeUndefined : break;
  //            default :
  //            {
  //              std::cout <<
  //              fieldenr->getEnrichment().enrTypeToString(fieldenr->getEnrichment().Type()) <<
  //              std::endl; FOUR_C_THROW("unknown enrichment type"); break;
  //            }
  //            } // end switch enrichment
  //          } // end loop over fieldenr
  //        } // end loop over nodes
  //      }
  //    }
  //  } // end loop over processor nodes
  //
  //  startpoints();
  //
  //  // test loop if all initial startpoints have been computed
  //  for (std::vector<TimeIntData>::iterator data=timeIntData_->begin();
  //      data!=timeIntData_->end(); data++)
  //  {
  //    if (data->startpoint_==dummyStartpoint)
  //      FOUR_C_THROW("WARNING! No enriched node on one interface side found!\nThis "
  //          "indicates that the whole area is at one side of the interface!");
  //  } // end loop over nodes
}  // end function reinitialize_data


/*------------------------------------------------------------------------------------------------*
 * call back-tracking of data at final Lagrangian origin of a point                  schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidSemiLagrange::call_back_tracking(
    Core::Elements::Element*& ele,   /// pointer to element
    TimeIntData* data,               /// data
    Core::LinAlg::Matrix<3, 1>& xi,  /// local coordinates
    const char* backTrackingType     /// type of back_tracking
)
{
  switch (ele->shape())
  {
    case Core::FE::CellType::hex8:
    {
      const int numnode = Core::FE::num_nodes<Core::FE::CellType::hex8>;
      back_tracking<numnode, Core::FE::CellType::hex8>(ele, data, xi, backTrackingType);
    }
    break;
    case Core::FE::CellType::hex20:
    {
      const int numnode = Core::FE::num_nodes<Core::FE::CellType::hex20>;
      back_tracking<numnode, Core::FE::CellType::hex20>(ele, data, xi, backTrackingType);
    }
    break;
    default:
      FOUR_C_THROW("xfem assembly type not yet implemented in time integration");
      break;
  };
}  // end back_tracking


/*------------------------------------------------------------------------------------------------*
 * back-tracking of data at final Lagrangian origin of a point                       schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
template <const int numnode, Core::FE::CellType distype>
void XFEM::XfluidSemiLagrange::back_tracking(
    Core::Elements::Element*& fittingele,  /// pointer to element
    TimeIntData* data,                     /// data
    Core::LinAlg::Matrix<3, 1>& xi,        /// local coordinates
    const char* backTrackingType           /// type of back_tracking
)
{
#ifdef DEBUG_SEMILAGRANGE
  Core::IO::cout << "\n==============================================";
  Core::IO::cout << "\n BACK-TRACKING on proc " << myrank_;
  Core::IO::cout << "\n==============================================" << Core::IO::endl;
#endif


  const int nsd = 3;  // dimension

  if ((strcmp(backTrackingType, static_cast<const char*>("standard")) != 0) and
      (strcmp(backTrackingType, static_cast<const char*>("failing")) != 0))
    FOUR_C_THROW("backTrackingType not implemented");

#ifdef DEBUG_SEMILAGRANGE
  if (strcmp(backTrackingType, static_cast<const char*>("standard")) == 0)
  {
    std::cout << "\n--------------------------------------------------\n"
              << "\nnode: " << data->node_ << "\ncomputed LAGRANGEAN ORIGIN  (startpoint) "
              << data->startpoint_ << "with xi-coord. " << xi << "in element " << *fittingele
              << "\n--------------------------------------------------" << std::endl;
  }
  if (strcmp(backTrackingType, static_cast<const char*>("failing")) == 0)
  {
    std::cout << "\n--------------------------------------------------\n"
              << "\nnode: " << data->node_
              << "\nused <<<PSEUDO>>> LAGRANGEAN ORIGIN (initialpoint) " << data->initialpoint_
              << "with xi-coord. " << xi << "in element " << *fittingele
              << "\n--------------------------------------------------" << std::endl;
  }
#endif

  //---------------------------------------------------------------------------------
  // Initialization

  Core::LinAlg::Matrix<3, 1> lagrangeanOrigin(
      Core::LinAlg::Initialization::zero);  // the applied Lagrangean origin (the real computed
                                            // or an approximated)

  if (strcmp(backTrackingType, static_cast<const char*>("standard")) == 0)
  {
    lagrangeanOrigin = data->startpoint_;  // use the computed start point approximation or the
                                           // projected start point
  }
  else if (strcmp(backTrackingType, static_cast<const char*>("failing")) == 0)
  {
    lagrangeanOrigin = data->initialpoint_;  // use the initial guess for the Lagrangean origin

    Core::IO::cout << "\n\tWARNING: Proc " << myrank_
                   << ": SEMI-LAGRANGEAN algorithm: used the initial-guess instead of the real "
                      "Lagrangean origin for node "
                   << data->node_.id() << Core::IO::endl;
  }
  else
    FOUR_C_THROW("backTrackingType not implemented");


  Core::LinAlg::Matrix<numnode, 1> shapeFcn(Core::LinAlg::Initialization::zero);  // shape function
  Core::LinAlg::Matrix<3, numnode> shapeFcnDeriv(
      Core::LinAlg::Initialization::zero);  // shape function derivatives w.r.t xyz
  Core::LinAlg::Matrix<nsd, nsd> xji(Core::LinAlg::Initialization::zero);  // inverse of jacobian

  double deltaT = 0;  // pseudo time-step size, used when the initial point is used instead of the
                      // computed lagrangean startpoint

  // data for the final back-tracking
  Core::LinAlg::Matrix<nsd, 1> vel(Core::LinAlg::Initialization::zero);  // velocity data
  std::vector<Core::LinAlg::Matrix<nsd, nsd>> velnDeriv1(oldVectors_.size(),
      Core::LinAlg::Matrix<nsd, nsd>(
          Core::LinAlg::Initialization::zero));  // first derivation of velocity data

  Core::LinAlg::Matrix<1, 1> pres(Core::LinAlg::Initialization::zero);  // pressure data
  std::vector<Core::LinAlg::Matrix<1, nsd>> presnDeriv1(oldVectors_.size(),
      Core::LinAlg::Matrix<1, nsd>(
          Core::LinAlg::Initialization::zero));  // first derivation of pressure data

  std::vector<Core::LinAlg::Matrix<nsd, 1>> veln(oldVectors_.size(),
      Core::LinAlg::Matrix<nsd, 1>(Core::LinAlg::Initialization::zero));  // velocity at t^n
  Core::LinAlg::Matrix<nsd, 1> transportVeln(
      Core::LinAlg::Initialization::zero);  // transport velocity at Lagrangian origin
                                            // (x_Lagr(t^n))


  //---------------------------------------------------------------------------------
  // fill velocity and pressure data at nodes of element ...

  // node velocities of the element nodes for transport velocity
  Core::LinAlg::Matrix<nsd, numnode> nodevel(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numnode, 1> nodepre(Core::LinAlg::Initialization::zero);

  // node velocities of the element nodes for the data that should be changed
  std::vector<Core::LinAlg::Matrix<nsd, numnode>> nodeveldata(
      oldVectors_.size(), Core::LinAlg::Matrix<nsd, numnode>(Core::LinAlg::Initialization::zero));
  // node pressures of the element nodes for the data that should be changed
  std::vector<Core::LinAlg::Matrix<numnode, 1>> nodepresdata(
      oldVectors_.size(), Core::LinAlg::Matrix<numnode, 1>(Core::LinAlg::Initialization::zero));

  // velocity of the data that shall be changed
  std::vector<Core::LinAlg::Matrix<nsd, 1>> velValues(
      oldVectors_.size(), Core::LinAlg::Matrix<nsd, 1>(Core::LinAlg::Initialization::zero));
  // pressures of the data that shall be changed
  std::vector<double> presValues(oldVectors_.size(), 0);


  for (size_t index = 0; index < oldVectors_.size(); index++)
  {
    nodeveldata[index].clear();
    nodepresdata[index].clear();
  }

  Core::Elements::Element* ele = fittingele;  // current element

  //---------------------------------------------------------------------------------
  // get shape functions and derivatives at local coordinates

  bool compute_deriv = true;

  eval_shape_and_deriv<numnode, distype>(ele, xi, xji, shapeFcn, shapeFcnDeriv, compute_deriv);

  //-------------------------------------------------------
  // get element location vector, dirichlet flags and ownerships (discret, nds, la, doDirichlet)
  std::vector<int> lm;

  for (int inode = 0; inode < numnode; inode++)
  {
    Core::Nodes::Node* node = ele->nodes()[inode];
    std::vector<int> dofs;
    dofset_old_->dof(dofs, node, data->nds_[inode]);


    for (int j = 0; j < 4; ++j)
    {
      lm.push_back(dofs[j]);
    }
  }

  //-------------------------------------------------------
  // all vectors are based on the same map

  extract_nodal_values_from_vector<numnode>(nodevel, nodepre, veln_, lm);

  for (size_t index = 0; index < oldVectors_.size(); index++)
    extract_nodal_values_from_vector<numnode>(
        nodeveldata[index], nodepresdata[index], oldVectors_[index], lm);



  //----------------------------------------------------------------------------------

  // node velocities of the element nodes for the fields that should be changed
  std::vector<std::vector<Core::LinAlg::Matrix<nsd, nsd>>> avg_nodevelgraddata;
  avg_nodevelgraddata.reserve(numnode);

  // node pressures of the element nodes for the data that should be changed
  std::vector<std::vector<Core::LinAlg::Matrix<1, nsd>>> avg_nodepresgraddata;
  avg_nodepresgraddata.reserve(numnode);

  for (size_t i = 0; i < numnode; i++)
  {
    std::vector<Core::LinAlg::Matrix<nsd, nsd>> tmp_vec(
        oldVectors_.size(), Core::LinAlg::Matrix<nsd, nsd>(Core::LinAlg::Initialization::zero));
    avg_nodevelgraddata.push_back(tmp_vec);

    std::vector<Core::LinAlg::Matrix<1, nsd>> tmp_vec2(
        oldVectors_.size(), Core::LinAlg::Matrix<1, nsd>(Core::LinAlg::Initialization::zero));
    avg_nodepresgraddata.push_back(tmp_vec2);
  }

  // Reconstruct nodal gradients
  for (int inode = 0; inode < numnode; inode++)
  {
    Core::Nodes::Node* node = (ele->nodes())[inode];

    // determine the elements used for the nodal gradient computation
    // determine the corresponding nodal dofset vectors used for averaging the nodal gradients
    std::vector<Core::Elements::Element*> eles_avg;
    std::vector<std::vector<int>> eles_avg_nds;


    Cut::Node* n = wizard_old_->get_node(node->id());

    if (n != nullptr)
    {
      // get the nodal dofset w.r.t the Lagrangean origin
      const std::set<Cut::plain_volumecell_set, Cut::Cmp>& cellset =
          n->get_nodal_dof_set(data->nds_[inode])->volume_cell_composite();


      // get for each adjacent element contained in the nodal dofset the first vc, its parent
      // element is used for the reconstruction REMARK: adjacent elements for that no elementhandle
      // exists in the cut won't be found here
      for (std::set<Cut::plain_volumecell_set, Cut::Cmp>::const_iterator cellset_it =
               cellset.begin();
          cellset_it != cellset.end(); cellset_it++)
      {
        // the first vc representing the set
        Cut::VolumeCell* vc = *((*cellset_it).begin());
        int peid = vc->parent_element()->get_parent_id();

        if (!discret_->have_global_element(peid))
          FOUR_C_THROW("element {} for averaging not on proc {}", peid, myrank_);

        // get the element
        Core::Elements::Element* e = discret_->g_element(peid);
        // get the vc's nds vector
        const std::vector<int> e_nds = vc->nodal_dof_set();

        // add the element and the nds vector
        eles_avg.push_back(e);
        eles_avg_nds.push_back(e_nds);
      }
    }

    // check all surrounding elements
    int numele = node->num_element();
    Core::Elements::Element** eles = node->elements();

    // add surrounding std uncut elements for that no elementhandle is available
    for (int i = 0; i < numele; i++)
    {
      Cut::ElementHandle* eh = wizard_old_->get_element(eles[i]);

      if (eh != nullptr)
        continue;  // element and the right nds-vec should have been found using the for-loop before

      // if we are here, then the element is a standard uncut element
      // and it is ensured that it has not been added to eles_avg yet
      std::vector<int> std_nds(eles[i]->num_node(), 0);

      eles_avg.push_back(eles[i]);
      eles_avg_nds.push_back(std_nds);
    }

    if (eles_avg.size() == 0) FOUR_C_THROW("there is no element for averaging");

    // compute the nodal gradients for velocity/acc and pressure component
    compute_nodal_gradient(oldVectors_, node, eles_avg, eles_avg_nds, *dofset_old_,
        avg_nodevelgraddata[inode], avg_nodepresgraddata[inode]);
  }
  //----------------------------------------------------------------------------------


  //---------------------------------------------------------------------------------
  // interpolate velocity and pressure values at starting point
  transportVeln.multiply(nodevel, shapeFcn);

#ifdef DEBUG_SEMILAGRANGE
  Core::IO::cout << "\t transportVeln\t" << transportVeln << Core::IO::endl;
#endif

  //---------------------------------------------------------------------------------
  // computing pseudo time-step deltaT
  // remark: if x is the Lagrange-origin of node, deltaT = dt with respect to small errors.
  // if it is not, deltaT estimates the time x needs to move to node)
  if (data->type_ == TimeIntData::predictor_)
  {
    Core::LinAlg::Matrix<nsd, 1> diff(data->node_.x().data());
    for (int i = 0; i < nsd; ++i) diff(i) += data->dispnp_(i);
    diff -= lagrangeanOrigin;  // diff = x_Node - x_Appr

    double numerator = transportVeln.dot(diff);             // numerator = v^T*(x_Node-x_Appr)
    double denominator = transportVeln.dot(transportVeln);  // denominator = v^T*v

    if (denominator > 1e-15) deltaT = numerator / denominator;  // else deltaT = 0 as initialized

#ifdef DEBUG_SEMILAGRANGE
    Core::IO::cout << " \t recomputed modified pseudo time-step size: " << deltaT << Core::IO::endl;
#endif
  }
  else
    deltaT = dt_;


  // interpolate velocity and pressure gradients for all fields at starting point and get final
  // values
  for (size_t index = 0; index < oldVectors_.size(); index++)
  {
    veln[index].multiply(nodeveldata[index], shapeFcn);
    // use the averaged nodal gradients
    velnDeriv1[index].clear();

    for (int i = 0; i < numnode; i++)
    {
      velnDeriv1[index].update(shapeFcn(i), avg_nodevelgraddata[i][index], 1.0);
      presnDeriv1[index].update(shapeFcn(i), avg_nodepresgraddata[i][index], 1.0);
    }
  }  // end loop over vectors to be read from


  for (size_t index = 0; index < oldVectors_.size(); index++)
  {
    vel.multiply(1.0 - theta(data), velnDeriv1[index], transportVeln);  // v = (1-theta)*Dv^n/Dx*v^n
    vel.multiply(theta(data), data->velDeriv_[index], data->vel_,
        1.0);  // v = theta*Dv^n+1/Dx*v^n+1+(1-theta)*Dv^n/Dx*v^n
    vel.update(
        1.0, veln[index], deltaT);  // v = v_n + dt*(theta*Dv^n+1/Dx*v^n+1+(1-theta)*Dv^n/Dx*v^n)
    velValues[index] = vel;

    pres.multiply(
        1.0 - theta(data), presnDeriv1[index], transportVeln);  // p = (1-theta)*Dp^n/Dx*v^n
    pres.multiply(theta(data), data->presDeriv_[index], data->vel_,
        1.0);  // p = theta*Dp^n+1/Dx*v^n+1+(1-theta)*Dp^n/Dx*v^n
    pres.multiply_tn(1.0, nodepresdata[index], shapeFcn,
        deltaT);  // p = p_n + dt*(theta*Dp^n+1/Dx*v^n+1+(1-theta)*Dp^n/Dx*v^n)
    presValues[index] = pres(0);

#ifdef DEBUG_SEMILAGRANGE
    Core::IO::cout << "\n***********************************************";
    Core::IO::cout << "\n           RECONSTRUCTED VALUES for node " << (data->node_).Id();
    Core::IO::cout << "\nvelocity entry in vector \t" << index << "\t " << vel;
    Core::IO::cout << "pressure entry in vector \t" << index << "\t " << pres(0);
    Core::IO::cout << "\n***********************************************" << Core::IO::endl;
#endif
  }  // loop over vectors to be set

  data->velValues_ = velValues;
  data->presValues_ = presValues;
  data->state_ = TimeIntData::doneStd_;


  return;
}  // end back_tracking


/*------------------------------------------------------------------------------------------------*
 * determine point's dofset in element ele w.r.t old or new interface position       schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidSemiLagrange::get_nodal_dof_set(
    Core::Elements::Element* ele,   /// pointer to element
    Core::LinAlg::Matrix<3, 1>& x,  /// global coordinates of point
    std::vector<int>& nds,          /// determine the points dofset w.r.t old/new interface position
    Cut::VolumeCell*& vc,           /// valid fluid volumecell the point x lies in
    bool step_np                    /// computation w.r.t old or new interface position?
)
{
  nds.clear();


#ifdef DEBUG_SEMILAGRANGE
  Core::IO::cout << "\n\t\t\t ... get_nodal_dof_set";
#endif


  std::shared_ptr<Cut::CutWizard> wizard = step_np ? wizard_new_ : wizard_old_;

  Cut::ElementHandle* e = wizard->get_element(ele);

  bool inside_structure = false;

  if (e != nullptr)  // element in cut involved
  {
    Cut::plain_volumecell_set cells;
    e->get_volume_cells(cells);

    if (cells.size() == 0)
      FOUR_C_THROW("Cut::Element {} does not contain any volume cell", ele->id());

    for (Cut::plain_volumecell_set::iterator cell_it = cells.begin(); cell_it != cells.end();
        cell_it++)
    {
      Cut::VolumeCell* cell = *cell_it;
      //      if(cell->Contains(x))
      // cell contains the point inside or on one of its boundaries and the cell is an outside
      // (fluid) cell
      if (((cell->is_this_point_inside(x) == "inside") or
              (cell->is_this_point_inside(x) == "onBoundary")) and
          cell->position() == Cut::Point::outside)
      {
#ifdef DEBUG_SEMILAGRANGE
        Core::IO::cout << "\n\t\t\t -> Position of point w.r.t volumecell is "
                       << cell->IsThisPointInside(x) << " \t cell pos = " << cell->Position()
                       << Core::IO::endl;
#endif
        nds = cell->nodal_dof_set();

        if ((int)nds.size() != ele->num_node())
          FOUR_C_THROW(
              " size of nds-vector != number of element nodes, why does the vc does not have nodal "
              "dofsets?!");

        vc = cell;

#ifdef DEBUG_SEMILAGRANGE
        Core::IO::cout << "nds-vector ";
        for (int i = 0; i < (int)nds.size(); i++)
        {
          Core::IO::cout << " " << nds[i];
        }
        Core::IO::cout << "\n";
#endif

        return;
      }
      // point lies within the structure or on Boundary
      else if ((cell->is_this_point_inside(x) == "inside" or
                   cell->is_this_point_inside(x) == "onBoundary") and
               (cell->position() == Cut::Point::inside))
      {
#ifdef DEBUG_SEMILAGRANGE
        Core::IO::cout << "\n\t\t\t -> Position of point w.r.t volumecell is "
                       << cell->IsThisPointInside(x) << " \t cell pos = " << cell->Position()
                       << Core::IO::endl;
#endif
        // do not return before all the other vcs have been tested, maybe a fluid-cell with
        // onBoundary can be found
        inside_structure = true;
      }
      else
      {
        // the point does not lie inside this vc !
        // #ifdef DEBUG_SEMILAGRANGE
        //        Core::IO::cout << "\n\t\t\t -> Position of cell " << cell->Position() << " and
        //        IsThisPointInside "<< cell->IsThisPointInside(x) << Core::IO::endl;
        // #endif
      }
    }

    if (!inside_structure and (int)(nds.size()) != ele->num_node())
    {
      Cut::plain_volumecell_set cells;
      e->get_volume_cells(cells);

      Core::IO::cout << "point: " << x << Core::IO::endl;
      for (Cut::plain_volumecell_set::iterator cell_it = cells.begin(); cell_it != cells.end();
          cell_it++)
      {
        Cut::VolumeCell* cell = *cell_it;
        Core::IO::cout << "vc-pos: " << cell->position() << Core::IO::endl;
      }

      FOUR_C_THROW(
          "no valid nds-vector could be determined and point does not lie within the structure!");
    }

    // return if the structural volume cell is the only one which was found
    if (inside_structure)
    {
      nds.clear();
#ifdef DEBUG_SEMILAGRANGE
      Core::IO::cout
          << "\n\t\t\t -> Position of point inside structure and not onBoundary of other "
             "fluid-vcs -> reset nds to empty vector"
          << Core::IO::endl;
#endif

      return;
    }

    Core::IO::cout << "error: coordinates of point x " << x
                   << " number of volumecells: " << cells.size() << Core::IO::endl;
    FOUR_C_THROW(
        "there is no volume cell in element {} which contains point with coordinates ({},{},{}) -> "
        "void element???",
        ele->id(), x(0), x(1), x(2));
  }
  else  // standard element, all its nodes have dofset 0
  {
    int numnode = ele->num_node();

    for (int inode = 0; inode < numnode; inode++)
    {
      nds.push_back(0);
    }
  }

  return;
}


/*------------------------------------------------------------------------------------------------*
 * compute gradients at nodes for that SL-reconstruction is called                   schott 04/13 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidSemiLagrange::compute_nodal_gradient(
    const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
        colVectors,           ///< all vectors for that we reconstruct the their gradients
    Core::Nodes::Node* node,  ///< node at which we reconstruct the gradients
    std::vector<Core::Elements::Element*>&
        eles,                                ///< elements around node used for the reconstruction
    std::vector<std::vector<int>>& ele_nds,  ///< corresponding elements nodal dofset information
    XFEM::XFEMDofSet& dofset,                ///< XFEM dofset
    std::vector<Core::LinAlg::Matrix<3, 3>>&
        velDeriv_avg,  ///< velocity/acc component derivatives for several vectors
    std::vector<Core::LinAlg::Matrix<1, 3>>&
        preDeriv_avg  ///< pressure-component derivatives for several vectors
) const
{
  const int nsd = 3;  // dimension

  for (size_t vec_index = 0; vec_index < colVectors.size(); vec_index++)
  {
    velDeriv_avg[vec_index].clear();
    preDeriv_avg[vec_index].clear();
  }

  int numele = (int)eles.size();

  if (numele != (int)ele_nds.size()) FOUR_C_THROW("number of nds-vector != number of elements!");


  for (int iele = 0; iele < numele; iele++)
  {
    Core::Elements::Element* e = eles[iele];

    // xi coordinates of node w.r.t this element
    Core::LinAlg::Matrix<nsd, 1> tmp_xi(Core::LinAlg::Initialization::zero);
    bool indomain = false;  // dummy variable

    // xyz coordinates of the node
    Core::LinAlg::Matrix<nsd, 1> x_node(node->x().data());

    // get the local coordinates of the node w.r.t current element
    // Comment to the configuration:
    // Here we use the fact that x_node is a node position in reference configuration,
    // which means we get the same tmp_xi if we use the element in reference config, as
    // if callXToXi would be done in (n+1) config and x_node^(n+1)
    call_x_to_xi_coords(e, x_node, tmp_xi, "reference", indomain);

    for (size_t tmp_index = 0; tmp_index < colVectors.size(); tmp_index++)
    {
      Core::LinAlg::Matrix<nsd, 1> vel(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd, nsd> vel_deriv(Core::LinAlg::Initialization::zero);
      double pres = 0.0;
      Core::LinAlg::Matrix<1, nsd> pres_deriv(Core::LinAlg::Initialization::zero);

      get_gp_values(e, tmp_xi, ele_nds[iele], dofset, vel, vel_deriv, pres, pres_deriv,
          colVectors[tmp_index], true);

      // add the current gradients
      velDeriv_avg[tmp_index].update(1.0, vel_deriv, 1.0);
      preDeriv_avg[tmp_index].update(1.0, pres_deriv, 1.0);
    }
  }  // end loop ele

  for (size_t vec_index = 0; vec_index < colVectors.size(); vec_index++)
  {
    velDeriv_avg[vec_index].scale(1. / numele);
    preDeriv_avg[vec_index].scale(1. / numele);
  }

}  // end function compute nodal gradient



/*------------------------------------------------------------------------------------------------*
 * get the time integration factor theta fitting to the computation type             schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
double XFEM::XfluidSemiLagrange::theta(TimeIntData* data) const
{
  double theta = -1.0;

  switch (data->type_)
  {
    case TimeIntData::predictor_:
      theta = 0.0;
      break;
    case TimeIntData::standard_:
      theta = theta_default_;
      break;
    default:
      FOUR_C_THROW("type not implemented");
      break;
  }

  if (theta < 0.0) FOUR_C_THROW("something wrong");

  return theta;
}  // end function theta



/*------------------------------------------------------------------------------------------------*
 * export alternative algo data to neighbour proc                                    schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidSemiLagrange::export_alternativ_algo_data()
{
  const int nsd = 3;  // 3 dimensions for a 3d fluid element

  // array of vectors which stores data for
  // every processor in one vector
  std::vector<std::vector<TimeIntData>> dataVec(numproc_);

  // fill vectors with the data
  for (std::vector<TimeIntData>::iterator data = timeIntData_->begin(); data != timeIntData_->end();
      data++)
  {
    if (data->state_ == TimeIntData::failedSL_)
    {
      dataVec[data->initial_ele_owner_].push_back(*data);
    }
  }

  clear_state(TimeIntData::failedSL_);
  timeIntData_->insert(timeIntData_->end(), dataVec[myrank_].begin(), dataVec[myrank_].end());

  dataVec[myrank_].clear();  // clear the set data from the vector

  // send data to the processor where the point lies (1. nearest higher neighbour 2. 2nd nearest
  // higher neighbour...)
  for (int dest = (myrank_ + 1) % numproc_; dest != myrank_;
      dest = (dest + 1) % numproc_)  // dest is the target processor
  {
    // Initialization of sending
    Core::Communication::PackBuffer
        dataSend;  // vector including all data that has to be send to dest proc

    // Initialization
    int source = myrank_ - (dest - myrank_);  // source proc (sends (dest-myrank_) far and gets from
                                              // (dest-myrank_) earlier)
    if (source < 0)
      source += numproc_;
    else if (source >= numproc_)
      source -= numproc_;

    for (std::vector<TimeIntData>::iterator data = dataVec[dest].begin();
        data != dataVec[dest].end(); data++)
    {
      if (data->state_ == TimeIntData::failedSL_)
      {
        pack_node(dataSend, data->node_);
        add_to_pack(dataSend, data->nds_np_);
        add_to_pack(dataSend, data->vel_);
        add_to_pack(dataSend, data->velDeriv_);
        add_to_pack(dataSend, data->presDeriv_);
        add_to_pack(dataSend, data->dispnp_);
        add_to_pack(dataSend, data->initialpoint_);
        add_to_pack(dataSend, data->initial_eid_);
        add_to_pack(dataSend, data->initial_ele_owner_);
        add_to_pack(dataSend, data->type_);
      }
    }

    // clear the no more needed data
    dataVec[dest].clear();

    std::vector<char> dataRecv;
    send_data(dataSend, dest, source, dataRecv);

    Core::Communication::UnpackBuffer buffer(dataRecv);
    // unpack received data
    while (!buffer.at_end())
    {
      std::vector<double> coords(nsd, 0.0);
      Core::Nodes::Node node(0, coords, 0);
      int nds_np;
      Core::LinAlg::Matrix<nsd, 1> vel;
      std::vector<Core::LinAlg::Matrix<nsd, nsd>> velDeriv;
      std::vector<Core::LinAlg::Matrix<1, nsd>> presDeriv;
      Core::LinAlg::Matrix<nsd, 1> dispnp;
      Core::LinAlg::Matrix<nsd, 1> initialpoint;
      int initial_eid;
      int initial_ele_owner;
      TimeIntData::Type newtype;

      unpack_node(buffer, node);
      extract_from_pack(buffer, nds_np);
      extract_from_pack(buffer, vel);
      extract_from_pack(buffer, velDeriv);
      extract_from_pack(buffer, presDeriv);
      extract_from_pack(buffer, dispnp);
      extract_from_pack(buffer, initialpoint);
      extract_from_pack(buffer, initial_eid);
      extract_from_pack(buffer, initial_ele_owner);
      extract_from_pack(buffer, newtype);

      timeIntData_->push_back(TimeIntData(node, nds_np, vel, velDeriv, presDeriv, dispnp,
          initialpoint, initial_eid, initial_ele_owner,
          newtype));  // startOwner is current proc
    }  // end loop over number of nodes to get

    // processors wait for each other
    Core::Communication::barrier(discret_->get_comm());
  }  // end loop over processors
}  // end export_alternativ_algo_data



/*------------------------------------------------------------------------------------------------*
 * export data while Newton loop to neighbour proc                                   schott 07/12 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::XfluidSemiLagrange::export_iter_data(bool& procDone)
{
#ifdef DEBUG_SEMILAGRANGE
  Core::IO::cout << "\n\t=============================";
  Core::IO::cout << "\n\t  export Iteration Data  ";
  Core::IO::cout << "\n\t=============================" << Core::IO::endl;
#endif

  const int nsd = 3;  // 3 dimensions for a 3d fluid element

  // Initialization
  int dest = myrank_ + 1;  // destination proc (the "next" one)
  if (myrank_ == (numproc_ - 1)) dest = 0;

  int source = myrank_ - 1;  // source proc (the "last" one)
  if (myrank_ == 0) source = numproc_ - 1;


  /*-------------------------------------------*
   * first part: send procfinished in order to *
   * check whether all procs have finished     *
   *-------------------------------------------*/
  for (int iproc = 0; iproc < numproc_ - 1; iproc++)
  {
    Core::Communication::PackBuffer dataSend;

    add_to_pack(dataSend, procDone);

    std::vector<char> dataRecv;
    send_data(dataSend, dest, source, dataRecv);

    // pointer to current position of group of cells in global std::string (counts bytes)
    bool allProcsDone;

    // unpack received data
    Core::Communication::UnpackBuffer buffer(dataRecv);
    extract_from_pack(buffer, allProcsDone);

    if (allProcsDone == false) procDone = false;

    // processors wait for each other
    Core::Communication::barrier(discret_->get_comm());
  }

  /*--------------------------------------*
   * second part: if not all procs have   *
   * finished send data to neighbour proc *
   *--------------------------------------*/
  if (!procDone)
  {
    Core::Communication::PackBuffer dataSend;

    for (std::vector<TimeIntData>::iterator data = timeIntData_->begin();
        data != timeIntData_->end(); data++)
    {
      if (data->state_ == TimeIntData::nextSL_)
      {
        pack_node(dataSend, data->node_);
        add_to_pack(dataSend, data->nds_np_);
        add_to_pack(dataSend, data->vel_);
        add_to_pack(dataSend, data->velDeriv_);
        add_to_pack(dataSend, data->presDeriv_);
        add_to_pack(dataSend, data->dispnp_);
        add_to_pack(dataSend, data->initialpoint_);
        add_to_pack(dataSend, data->initial_eid_);
        add_to_pack(dataSend, data->initial_ele_owner_);
        add_to_pack(dataSend, data->startpoint_);
        add_to_pack(dataSend, data->searchedProcs_);
        add_to_pack(dataSend, data->counter_);
        add_to_pack(dataSend, data->type_);
      }
    }

    clear_state(TimeIntData::nextSL_);

    std::vector<char> dataRecv;
    send_data(dataSend, dest, source, dataRecv);


    // unpack received data
    Core::Communication::UnpackBuffer buffer(dataRecv);
    while (!buffer.at_end())
    {
      std::vector<double> coords(nsd, 0.0);
      Core::Nodes::Node node(0, coords, 0);
      int nds_np;
      Core::LinAlg::Matrix<nsd, 1> vel;
      std::vector<Core::LinAlg::Matrix<nsd, nsd>> velDeriv;
      std::vector<Core::LinAlg::Matrix<1, nsd>> presDeriv;
      Core::LinAlg::Matrix<nsd, 1> dispnp;
      Core::LinAlg::Matrix<nsd, 1> initialpoint;
      int initial_eid;
      int initial_ele_owner;
      Core::LinAlg::Matrix<nsd, 1> startpoint;
      int searchedProcs;
      int iter;
      TimeIntData::Type newtype;

      unpack_node(buffer, node);
      extract_from_pack(buffer, nds_np);
      extract_from_pack(buffer, vel);
      extract_from_pack(buffer, velDeriv);
      extract_from_pack(buffer, presDeriv);
      extract_from_pack(buffer, dispnp);
      extract_from_pack(buffer, initialpoint);
      extract_from_pack(buffer, initial_eid);
      extract_from_pack(buffer, initial_ele_owner);
      extract_from_pack(buffer, startpoint);
      extract_from_pack(buffer, searchedProcs);
      extract_from_pack(buffer, iter);
      extract_from_pack(buffer, newtype);

      timeIntData_->push_back(TimeIntData(node, nds_np, vel, velDeriv, presDeriv, dispnp,
          initialpoint, initial_eid, initial_ele_owner, startpoint, searchedProcs, iter, newtype));
    }  // end loop over number of points to get

    // processors wait for each other
    Core::Communication::barrier(discret_->get_comm());
  }  // end if procfinished == false
}  // end export_iter_data

FOUR_C_NAMESPACE_CLOSE

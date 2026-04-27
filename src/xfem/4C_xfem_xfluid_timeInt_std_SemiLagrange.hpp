// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_XFEM_XFLUID_TIMEINT_STD_SEMILAGRANGE_HPP
#define FOUR_C_XFEM_XFLUID_TIMEINT_STD_SEMILAGRANGE_HPP

#include "4C_config.hpp"

#include "4C_xfem_xfluid_timeInt_base.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Core::Elements
{
  class Element;
}


namespace XFEM
{
  /*========================================================================*/
  // forward declarations
  /*========================================================================*/

  class XfluidTimeintBase;
  class TimeIntData;


  /*!
    \brief this class is used in XFEM to compute new values for standard degrees of freedom
    for nodes which change their interface side between two consecutive time steps.
    It bases on the Semi-Lagrangian approach described in
    "/intern/arbeiten/diplomarbeiten/WinklmaierMartin2010.pdf"
   */
  class XfluidSemiLagrange : public XfluidStd
  {
   public:
    /*========================================================================*/
    //! constructor/destructor
    /*========================================================================*/

    //! constructor
    explicit XfluidSemiLagrange(
        XFEM::XfluidTimeintBase& timeInt,  ///< time integration base class object
        const std::map<int, std::vector<XFEM::XFluidTimeIntMethod>>&
            reconstr_method,                     ///< reconstruction map for nodes and its dofsets
        XFEM::XFluidTimeIntMethod& timeIntType,  ///< type of time integration
        const std::shared_ptr<Core::LinAlg::Vector<double>> veln,  ///< velocity at time t^n
        const double& dt,                                          ///< time step size
        const double& theta,                                       ///< OST theta
        bool initialize                                            ///< is initialization?

    );

    /*========================================================================*/
    //! compute routines
    /*========================================================================*/

    //! get startvalues in time step for nodes which changed interface-side
    void compute(
        std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>& newRowVectorsn) override;

   private:
    /*========================================================================*/
    //! control routines for Semi-Lagrangean Newton-loop
    /*========================================================================*/

    //! run a Newton loop in order to compute the exact Lagrangian origin for a node which changed
    //! interface side
    void newton_loop(Core::Elements::Element*& ele,  ///< pointer to element
        TimeIntData* data,                           ///< current data
        Core::LinAlg::Matrix<3, 1>& xi,              ///< local coordinates of point
        Core::LinAlg::Matrix<3, 1>& vel,             ///< velocity at current point
        bool& elefound                               ///< is element found ?
    );

    //! perform one Newton iteration in order to compute the exact Lagrangian origin for a node
    //! which changed its interface side
    void newton_iter(Core::Elements::Element*& ele,  ///< pointer to element to be updated
        TimeIntData* data,                           ///< current data to be updated
        Core::LinAlg::Matrix<3, 1>& xi,              ///< local coordinates w.r.t ele to be updated
        Core::LinAlg::Matrix<3, 1>&
            residuum,  ///< residual for semilagrangean backtracking to be updated
        Core::LinAlg::Matrix<3, 1>&
            incr,       ///< computed increment for lagrangean origin to be updated
        bool& elefound  ///< element found ?
    );

    //! check if newton iteration has finished
    bool global_newton_finished(int counter = 0) const;

    //! Decide how or if to continue when the startpoint approximation changed the side
    bool continue_for_changing_side(TimeIntData* data,  ///< current data to be updated
        Core::Elements::Element* ele,  ///< pointer to element the current point lies in
        std::vector<int>& nds_curr     ///< nds-vector of current volumecell the current startpoint
                                       ///< approximation lies in
    );

    //! determine velocity and pressure for nodes where the "normal" semi-lagrange startfinder
    //! failed
    void get_data_for_not_converged_nodes();

    //! prepare new iteration
    void new_iteration_prepare(
        std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> newRowVectors);

    //! gradients at a node
    void new_iteration_nodal_data(
        std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> newRowVectors);

    //! reinitialize some data for new computations, e.g. at a new FGI
    void reinitialize_data();


    /*========================================================================*/
    //! Semi-Lagrangean backtracking routines
    /*========================================================================*/

    //! call the back tracking which computes the final values
    void call_back_tracking(Core::Elements::Element*& ele,  ///< pointer to element
        TimeIntData* data,                                  ///< data
        Core::LinAlg::Matrix<3, 1>& xi,                     ///< local coordinates
        const char* backTrackingType                        ///< type of back_tracking
    );

    //! track back the Lagrangian origin to get final values
    template <const int numnode, Core::FE::CellType distype>
    void back_tracking(Core::Elements::Element*& fittingele,  ///< pointer to element
        TimeIntData* data,                                    ///< data
        Core::LinAlg::Matrix<3, 1>& xi,                       ///< local coordinates
        const char* backTrackingType                          ///< type of backTrackingwVectors
    );

    /*========================================================================*/
    //! element/dofset based routines
    /*========================================================================*/

    //! determine point's dofset in element ele w.r.t old or new interface position
    void get_nodal_dof_set(Core::Elements::Element* ele,  ///< pointer to element
        Core::LinAlg::Matrix<3, 1>& x,                    ///< global coordinates of point
        std::vector<int>& nds,  ///< determine the points dofset w.r.t old/new interface position
        Cut::VolumeCell*& vc,   ///< valid fluid volumecell the point x lies in
        bool step_np            ///< computation w.r.t old or new interface position?
    );

    //! compute the nodal gradient
    void compute_nodal_gradient(
        const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
            colVectors,           ///< all vectors for that we reconstruct the their gradients
        Core::Nodes::Node* node,  ///< node at which we reconstruct the gradients
        std::vector<Core::Elements::Element*>&
            eles,  ///< elements around node used for the reconstruction
        std::vector<std::vector<int>>&
            ele_nds,               ///< corresponding elements nodal dofset information
        XFEM::XFEMDofSet& dofset,  ///< XFEM dofset
        std::vector<Core::LinAlg::Matrix<3, 3>>&
            velDeriv_avg,  ///< velocity/acc component derivatives for several vectors
        std::vector<Core::LinAlg::Matrix<1, 3>>&
            preDeriv_avg  ///< pressure-component derivatives for several vectors
    ) const;


    /*========================================================================*/
    //! others
    /*========================================================================*/

    //! compute the theta which has to be used for computation
    double theta(TimeIntData* data) const;



    /*========================================================================*/
    //! parallel routines
    /*========================================================================*/

    //! export data to startpoint processor when Semi-Lagrange algorithm failed
    void export_alternativ_algo_data();

    //! export data to neighbour proc in Newton loop
    void export_iter_data(bool& procDone);


    /*========================================================================*/
    //! constants
    /*========================================================================*/

    const double theta_default_;  //! factor of one-step theta scheme

    const double rel_tol_incr_;  //! tolerance for the increment
    const double rel_tol_res_;   //! tolerance for the residual

  };  // class XFLUID_SemiLagrange
}  // namespace XFEM


FOUR_C_NAMESPACE_CLOSE

#endif

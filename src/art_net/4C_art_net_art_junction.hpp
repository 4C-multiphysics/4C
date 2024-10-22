// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ART_NET_ART_JUNCTION_HPP
#define FOUR_C_ART_NET_ART_JUNCTION_HPP


#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include <Epetra_MpiComm.h>
#include <Teuchos_RCP.hpp>
#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN



namespace Arteries
{
  namespace Utils
  {
    //--------------------------------------------------------------------
    // nodal information at a junction
    //--------------------------------------------------------------------

    /*!
    \brief structure of required nodal information at a junction
    this structure is meant to do some organisation stuff
    */

    struct JunctionNodeParams
    {
      double Ao_;    // Initial cross-sectional area
      double beta_;  // Material constant
      double Pext_;  // external pressure at that node
      double Q_;     // Volumetric flow rate
      double A_;     // Cross-sectional area
      double W_;     // Charechteristic velocity
      double rho_;   // density of blood
    };

    //--------------------------------------------------------------------
    // Wrapper class (to be called from outside) for junction bc
    //--------------------------------------------------------------------

    /*!
    \brief 1d-artery junction boundary condition wrapper
    this class is meant to do some organisation stuff
    */
    class ArtJunctionWrapper
    {
      friend class ArtNetExplicitTimeInt;


     public:
      /*!
      \brief Standard Constructor
      */
      ArtJunctionWrapper(Teuchos::RCP<Core::FE::Discretization> actdis,
          Core::IO::DiscretizationWriter &output, Teuchos::ParameterList &params, double dta);

      /*!
      \brief Destructor
      */
      virtual ~ArtJunctionWrapper() = default;


      /*!
      \brief Wrapper for ArtJunctionBc::update_residual
     */
      void update_residual(Teuchos::RCP<Core::LinAlg::Vector<double>> residual);

      /*!
      \brief Standard solver
      */
      int solve(Teuchos::ParameterList &params);

      /*!
      \brief Standard solver
      */
      void apply_bc(Teuchos::ParameterList &params);



     private:
      /*!
      \brief all single junction conditions
     */
      std::map<const int, Teuchos::RCP<class ArtJunctionBc>> ajunmap_;

      /*!
      \brief all parameters connected to junctions
      */
      //      Teuchos::RCP<map<const int, Teuchos::RCP<JunctionNodeParams> > >  nodalParams_;
      // map<const int, Teuchos::RCP<Teuchos::ParameterList> >  nodalParams_;

      //! 1d artery discretization
      Teuchos::RCP<Core::FE::Discretization> discret_;

      //! the output writer
      Core::IO::DiscretizationWriter &output_;

    };  // class ArtJunctionWrapper



    //--------------------------------------------------------------------
    // Actual junction bc calculation
    //--------------------------------------------------------------------
    /*!
    \brief 1d-artery junction boundary condition

    */

    class ArtJunctionBc
    {
      friend class ArtJunctionWrapper;

     public:
      using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
      using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;

      /*!
      \brief Standard Constructor
     */
      ArtJunctionBc(Teuchos::RCP<Core::FE::Discretization> actdis,
          Core::IO::DiscretizationWriter &output, std::vector<Core::Conditions::Condition *> conds,
          std::vector<int> IOart_flag, double dta, int condid, int numcond);

      /*!
      \brief Empty Constructor
      */
      ArtJunctionBc();

      /*!
      \brief Destructor
      */
      virtual ~ArtJunctionBc() = default;

     protected:
      /*!
      \Apply the boundary condition to the elements
      */
      void apply_bc(double time, double dta, int condid);

      /*!
      \Solve the boundary condition to the elements
      */
      int solve(Teuchos::ParameterList &params);


      /*!
      \Evaluate the Jacobian matrix to solve the nonlinear problem
      */
      void jacobian_eval(Core::LinAlg::SerialDenseMatrix &Jacobian, std::vector<double> &A,
          std::vector<double> &Q, std::vector<double> &W, std::vector<double> &Ao,
          std::vector<double> &rho, std::vector<double> &beta, std::vector<double> &Pext);

      /*!
      \Evaluate the residual vector needed to solve the nonlinear problem
      */
      void residual_eval(Core::LinAlg::SerialDenseVector &f, std::vector<double> &A,
          std::vector<double> &Q, std::vector<double> &W, std::vector<double> &Ao,
          std::vector<double> &rho, std::vector<double> &beta, std::vector<double> &Pext);

      void residual_eval(Core::LinAlg::SerialDenseMatrix &f, std::vector<double> &A,
          std::vector<double> &Q, std::vector<double> &W, std::vector<double> &Ao,
          std::vector<double> &rho, std::vector<double> &beta, std::vector<double> &Pext);

      /*!
      \Evaluate the residual vector needed to solve the nonlinear problem
      */
      void update_result(Core::LinAlg::SerialDenseVector &xn, Core::LinAlg::SerialDenseVector &dx);

      /*!
      \Evaluate the residual vector needed to solve the nonlinear problem
      */
      double two_norm(Core::LinAlg::SerialDenseVector &x);



      /*!
      \Update the Residual
      */
      void update_residual(Teuchos::RCP<Core::LinAlg::Vector<double>> residual);


     protected:
      Teuchos::RCP<Core::LinAlg::Vector<double>> junctionbc_;

     private:
      //! ID of present condition
      int condid_;

      //! vector associated to the pressure head loss constant
      std::vector<double> kr_;

      //! time step size
      double dta_;

      //! the processor ID from the communicator
      int myrank_;

      //! fluid discretization
      Teuchos::RCP<Core::FE::Discretization> discret_;

      //! the output writer
      Core::IO::DiscretizationWriter &output_;

      //! the vector defining whethe an element is inlet or outlet
      std::vector<int> io_art_flag_;

      //! Size of the nonlinear problem matrix
      int prob_size_;

      //! vector of nodes connected to the junction
      std::vector<int> nodes_;

      //! A Teuchos wrapper for a dense matrix solver
      Teuchos::SerialDenseSolver<ordinalType, scalarType> solver_;


    };  // class ArtJunctionBc

  }  // namespace Utils
}  // namespace Arteries

FOUR_C_NAMESPACE_CLOSE

#endif

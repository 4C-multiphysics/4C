/*----------------------------------------------------------------------*/
/*! \file
\brief Method to deal with one dimensional artery juction bcs


\level 3

*----------------------------------------------------------------------*/

#ifndef BACI_ART_NET_ART_JUNCTION_HPP
#define BACI_ART_NET_ART_JUNCTION_HPP


#include "baci_config.hpp"

#include "baci_io.hpp"
#include "baci_lib_discret.hpp"
#include "baci_linalg_utils_sparse_algebra_math.hpp"

#include <Epetra_MpiComm.h>
#include <Teuchos_RCP.hpp>
#include <Teuchos_SerialDenseSolver.hpp>

BACI_NAMESPACE_OPEN



namespace ART
{
  namespace UTILS
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
      ArtJunctionWrapper(Teuchos::RCP<DRT::Discretization> actdis, IO::DiscretizationWriter &output,
          Teuchos::ParameterList &params, double dta);

      /*!
      \brief Destructor
      */
      virtual ~ArtJunctionWrapper() = default;


      /*!
      \brief Wrapper for ArtJunctionBc::UpdateResidual
     */
      void UpdateResidual(Teuchos::RCP<Epetra_Vector> residual);

      /*!
      \brief Standard solver
      */
      int Solve(Teuchos::ParameterList &params);

      /*!
      \brief Standard solver
      */
      void ApplyBC(Teuchos::ParameterList &params);



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
      Teuchos::RCP<DRT::Discretization> discret_;

      //! the output writer
      IO::DiscretizationWriter &output_;

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
      using ordinalType = CORE::LINALG::SerialDenseMatrix::ordinalType;
      using scalarType = CORE::LINALG::SerialDenseMatrix::scalarType;

      /*!
      \brief Standard Constructor
     */
      ArtJunctionBc(Teuchos::RCP<DRT::Discretization> actdis, IO::DiscretizationWriter &output,
          std::vector<DRT::Condition *> conds, std::vector<int> IOart_flag, double dta, int condid,
          int numcond);

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
      void ApplyBc(double time, double dta, int condid);

      /*!
      \Solve the boundary condition to the elements
      */
      int Solve(Teuchos::ParameterList &params);


      /*!
      \Evaluate the Jacobian matrix to solve the nonlinear problem
      */
      void Jacobian_Eval(CORE::LINALG::SerialDenseMatrix &Jacobian, std::vector<double> &A,
          std::vector<double> &Q, std::vector<double> &W, std::vector<double> &Ao,
          std::vector<double> &rho, std::vector<double> &beta, std::vector<double> &Pext);

      /*!
      \Evaluate the residual vector needed to solve the nonlinear problem
      */
      void Residual_Eval(CORE::LINALG::SerialDenseVector &f, std::vector<double> &A,
          std::vector<double> &Q, std::vector<double> &W, std::vector<double> &Ao,
          std::vector<double> &rho, std::vector<double> &beta, std::vector<double> &Pext);

      void Residual_Eval(CORE::LINALG::SerialDenseMatrix &f, std::vector<double> &A,
          std::vector<double> &Q, std::vector<double> &W, std::vector<double> &Ao,
          std::vector<double> &rho, std::vector<double> &beta, std::vector<double> &Pext);

      /*!
      \Evaluate the residual vector needed to solve the nonlinear problem
      */
      void Update_Result(CORE::LINALG::SerialDenseVector &xn, CORE::LINALG::SerialDenseVector &dx);

      /*!
      \Evaluate the residual vector needed to solve the nonlinear problem
      */
      double twoNorm(CORE::LINALG::SerialDenseVector &x);



      /*!
      \Update the Residual
      */
      void UpdateResidual(Teuchos::RCP<Epetra_Vector> residual);


     protected:
      Teuchos::RCP<Epetra_Vector> junctionbc_;

     private:
      //! ID of present condition
      int condid_;

      //! vector associated to the pressure head loss constant
      std::vector<double> Kr;

      //! time step size
      double dta_;

      //! the processor ID from the communicator
      int myrank_;

      //! fluid discretization
      Teuchos::RCP<DRT::Discretization> discret_;

      //! the output writer
      IO::DiscretizationWriter &output_;

      //! the vector defining whethe an element is inlet or outlet
      std::vector<int> IOart_flag_;

      //! Size of the nonlinear problem matrix
      int ProbSize_;

      //! vector of nodes connected to the junction
      std::vector<int> nodes_;

      //! A Teuchos wrapper for a dense matrix solver
      Teuchos::SerialDenseSolver<ordinalType, scalarType> solver_;


    };  // class ArtJunctionBc

  }  // namespace UTILS
}  // namespace ART

BACI_NAMESPACE_CLOSE

#endif  // ART_NET_ART_JUNCTION_H

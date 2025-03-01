// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_STATUSTEST_HPP
#define FOUR_C_FSI_STATUSTEST_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_linalg_mapextractor.hpp"

#include <NOX_Abstract_Group.H>
#include <NOX_StatusTest_Generic.H>
#include <NOX_Utils.H>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Coupling::Adapter
{
  class CouplingConverter;
}

namespace NOX
{
  namespace FSI
  {
    class Newton;
  }
}  // namespace NOX

namespace NOX
{
  namespace FSI
  {
    /// a copy of ::NOX::StatusTest::NormF that works on a vector part only
    /*!

      This is a copy of ::NOX::StatusTest::NormF that allows subclasses to
      define what vector to test. This way we can specify tests on
      vectors that cover only parts of our field.

     */
    class GenericNormF : public ::NOX::StatusTest::Generic
    {
     public:
      //! Type that determines whether to scale the norm by the problem size.
      enum ScaleType
      {
        //! No norm scaling
        Unscaled,
        //! Scale the norm by the length of the vector
        Scaled
      };

      //! Type that determines whether the norm is absolute or relative to the initial guess
      enum ToleranceType
      {
        //! Relative to starting guess
        Relative,
        //! Absolute
        Absolute
      };

      GenericNormF(std::string name, double tolerance,
          ::NOX::Abstract::Vector::NormType normType = ::NOX::Abstract::Vector::TwoNorm,
          ScaleType stype = Scaled);

      /// do the check
      ::NOX::StatusTest::StatusType checkStatus(
          const ::NOX::Solver::Generic& problem, ::NOX::StatusTest::CheckType checkType) override;

      /// get the result
      ::NOX::StatusTest::StatusType getStatus() const override;

      /// output
      std::ostream& print(std::ostream& stream, int indent = 0) const override;

      /* @name Accessor Functions
         Used to query current values of variables in the status test.
      */
      //@{

      //! Returns the value of the F-norm computed in the last call to checkStatus.
      virtual double get_norm_f() const;

      //! Returns the true tolerance.
      virtual double get_true_tolerance() const;

      //! Returns the specified tolerance set in the constructor.
      virtual double get_specified_tolerance() const;

      //! Returns the initial tolerance.
      virtual double get_initial_tolerance() const;

      //@}

     protected:
      /*! \brief Calculate the norm of F for the given group according to
        the scaling type, norm type, and tolerance type.

        \note Returns -1.0 if F(x) has not been calculated for the given
        grp (i.e., grp.isF() is false).
      */
      virtual double compute_norm(const ::NOX::Abstract::Group& grp) = 0;

      /// compute the norm of a given vector
      double compute_norm(const Epetra_Vector& v);

      /*! In the case of a relative norm calculation, initializes
        \c trueTolerance based on the F-value at the initial guess.*/
      // void relative_setup(::NOX::Abstract::Group& initialGuess);

      double tolerance() const { return true_tolerance_; }

     private:
      //! %Status
      ::NOX::StatusTest::StatusType status_;

      //! Type of norm to use
      ::NOX::Abstract::Vector::NormType norm_type_;

      //! Scaling to use
      ScaleType scale_type_;

      //! Tolerance required for convergence.
      double specified_tolerance_;

      //! Initial tolerance
      double initial_tolerance_;

      //! True tolerance value, i.e., specifiedTolerance / initialTolerance
      double true_tolerance_;

      //! Norm of F to be compared to trueTolerance
      double norm_f_;

      //! name of this test
      std::string name_;
    };


    /// generic norm F enhanced for adaptive Newton
    class AdaptiveNewtonNormF : public GenericNormF
    {
     public:
      AdaptiveNewtonNormF(std::string name, double tolerance,
          ::NOX::Abstract::Vector::NormType normType = ::NOX::Abstract::Vector::TwoNorm,
          ScaleType stype = Scaled)
          : GenericNormF(name, tolerance, normType, stype)
      {
      }

      void set_newton(Teuchos::RCP<NOX::FSI::Newton> newton) { newton_ = newton; }

     protected:
      Teuchos::RCP<NOX::FSI::Newton> newton() { return newton_; }

     private:
      Teuchos::RCP<NOX::FSI::Newton> newton_;
    };


    /// Norm of inner dofs of one of our fields
    /*!

      For FSI status tests independent norms of the inner dofs of
      structural and fluid field are required.

     */
    class PartialNormF : public AdaptiveNewtonNormF
    {
     public:
      PartialNormF(std::string name, const Core::LinAlg::MultiMapExtractor& extractor, int blocknum,
          double tolerance,
          ::NOX::Abstract::Vector::NormType normType = ::NOX::Abstract::Vector::TwoNorm,
          ScaleType stype = Scaled);

     protected:
      double compute_norm(const ::NOX::Abstract::Group& grp) override;

     private:
      const Core::LinAlg::MultiMapExtractor extractor_;
      int blocknum_;
    };


    /// norm of sum of two (matching) parts of a large vector
    class PartialSumNormF : public AdaptiveNewtonNormF
    {
     public:
      PartialSumNormF(std::string name, const Core::LinAlg::MapExtractor& extractor1, double scale1,
          const Core::LinAlg::MapExtractor& extractor2, double scale2,
          std::shared_ptr<Coupling::Adapter::CouplingConverter> converter, double tolerance,
          ScaleType stype = Scaled);

     protected:
      double compute_norm(const ::NOX::Abstract::Group& grp) override;

     private:
      const Core::LinAlg::MapExtractor extractor1_;
      const Core::LinAlg::MapExtractor extractor2_;
      double scale1_;
      double scale2_;
      std::shared_ptr<Coupling::Adapter::CouplingConverter> converter_;
    };


    /// Copy of ::NOX::StatusTest::NormUpdate
    /*!
     *  Compute norm of the iterative solution increment.
     *
     *  The iterative solution increment is computed as difference of solutions
     *  of two subsequent nonlinear iterations.
     */
    class GenericNormUpdate : public ::NOX::StatusTest::Generic
    {
     public:
      //! Type that determines whether to scale the norm by the problem size.
      enum ScaleType
      {
        //! No norm scaling
        Unscaled,
        //! Scale the norm by the length of the vector
        Scaled
      };

      //! Constructor for absolute norm.
      /*! This constructor defaults to the \c Absolute tolerance type. */
      GenericNormUpdate(std::string name, double tolerance, ::NOX::Abstract::Vector::NormType ntype,
          ScaleType stype = Scaled);

      //! Constructor for absolute norm
      /*! This constructor defaults to the \c Absolute ToleranceType and \c TWO NormType. */
      GenericNormUpdate(std::string name, double tol, ScaleType stype = Scaled);


      ::NOX::StatusTest::StatusType checkStatus(
          const ::NOX::Solver::Generic& problem, ::NOX::StatusTest::CheckType checkType) override;

      ::NOX::StatusTest::StatusType getStatus() const override;

      std::ostream& print(std::ostream& stream, int indent = 0) const override;

      /* @name Access Functions
         Used to query current values of variables in the status test.
      */
      //@{

      //! Returns the value of the Update-norm computed in the last call to checkStatus.
      virtual double get_norm_update() const;

      //! Returns the true tolerance.
      virtual double get_tolerance() const;

      //@}

     protected:
      /// compute the norm of a given vector
      virtual double compute_norm(const Epetra_Vector& v);

     private:
      //! %Status
      ::NOX::StatusTest::StatusType status_;

      //! Vector containing the update for the current outer iteration
      std::shared_ptr<::NOX::Abstract::Vector> update_vector_ptr_;

      //! Type of norm to use
      ::NOX::Abstract::Vector::NormType norm_type_;

      //! Scaling to use
      ScaleType scale_type_;

      //! Tolerance value specified by user in constructor
      double tolerance_;

      //! Norm of the update to be compared to trueTolerance
      double norm_update_;

      //! name of this test
      std::string name_;
    };


    /// Norm of inner dofs of one of our fields
    class PartialNormUpdate : public GenericNormUpdate
    {
     public:
      //! Constructor for absolute norm
      /*! This constructor defaults to the \c Absolute tolerance type. */
      PartialNormUpdate(std::string name, const Core::LinAlg::MultiMapExtractor& extractor,
          int blocknum, double tolerance,
          ::NOX::Abstract::Vector::NormType ntype = ::NOX::Abstract::Vector::TwoNorm,
          ScaleType stype = Scaled);

      //! Constructor for absolute norm
      /*! This constructor defaults to the \c Absolute ToleranceType and \c TWO NormType. */
      PartialNormUpdate(std::string name, const Core::LinAlg::MultiMapExtractor& extractor,
          int blocknum, double tolerance, ScaleType stype = Scaled);

     protected:
      /// compute the norm of a given vector
      double compute_norm(const Epetra_Vector& v) override;

     private:
      const Core::LinAlg::MultiMapExtractor extractor_;
      int blocknum_;
    };


    /// require some iterations at least
    class MinIters : public ::NOX::StatusTest::Generic
    {
     public:
      /*! \brief Constructor
       *
       *  Specify the minimum number of nonlinear solver iterations, \f$k_{\min}\f$
       *  and optionally an error stream for printing errors.
       *
       */
      MinIters(int minIterations, const ::NOX::Utils* u = nullptr);


      ::NOX::StatusTest::StatusType checkStatus(
          const ::NOX::Solver::Generic& problem, ::NOX::StatusTest::CheckType checkType) override;

      ::NOX::StatusTest::StatusType getStatus() const override;

      std::ostream& print(std::ostream& stream, int indent = 0) const override;

      //! Returns the Minimum number of iterations set in the constructor.
      virtual int get_min_iters() const;

      /*!
      \brief Returns the current number of iterations taken by the solver.

      Returns -1 if the status of this test is ::NOX::StatusTest::Unevaluated.
      */
      virtual int get_num_iters() const;

     private:
      //! Minimum number of iterations
      int miniters_;

      //! Current number of iterations (if known)
      int niters_;

      //! Status
      ::NOX::StatusTest::StatusType status_;

      //! Ostream used to print errors
      ::NOX::Utils utils_;
    };

  }  // namespace FSI
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif

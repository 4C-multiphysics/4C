/*-----------------------------------------------------------*/
/*! \file

\brief Wrapper for all Dirichlet boundary condition tasks.


\level 3

*/
/*-----------------------------------------------------------*/

#ifndef FOUR_C_STRUCTURE_NEW_DBC_HPP
#define FOUR_C_STRUCTURE_NEW_DBC_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_abstract_prepostoperator.hpp"

#include <Teuchos_RCP.hpp>

// forward declarations
class Epetra_Map;
class Epetra_Vector;
namespace Teuchos
{
  class ParameterList;
}  // namespace Teuchos

FOUR_C_NAMESPACE_OPEN

namespace DRT
{
  class Discretization;

  namespace UTILS
  {
    class LocsysManager;
  }  // namespace UTILS
}  // namespace DRT

namespace CORE::LINALG
{
  class SparseOperator;
  class SparseMatrix;
  class MapExtractor;
}  // namespace CORE::LINALG

namespace STR
{
  namespace TIMINT
  {
    class Base;
    class BaseDataGlobalState;
  }  // namespace TIMINT

  /*! \brief Object to handle Dirichlet boundary conditions for solid dynamics
   *
   *  This class provides capabilities to
   *  - handle Dirichlet boundary conditions
   *  - rotate the coordinate system (referred to as 'locSys')
   */
  class Dbc
  {
   public:
    //! Constructor
    Dbc();

    //! Destructor
    virtual ~Dbc() = default;

    //! Initialize class variables
    virtual void Init(const Teuchos::RCP<DRT::Discretization>& discret,
        const Teuchos::RCP<Epetra_Vector>& freact,
        const Teuchos::RCP<const STR::TIMINT::Base>& timint_ptr);

    //! Setup class variables
    virtual void Setup();

    /*! \brief Apply the DBC to system of equations
     *
     *  \note Stay in the local coordinate system and do not rotate back (if locSys is defined).*/
    void ApplyDirichletToLocalSystem(
        Teuchos::RCP<CORE::LINALG::SparseOperator> A, Teuchos::RCP<Epetra_Vector>& b) const;

    /*! \brief Apply the DBC to a vector
     *
     *  \note Stay in the global coordinate system (Rotation: global-->local-->global).*/
    void ApplyDirichletToVector(Teuchos::RCP<Epetra_Vector>& vec) const;

    /*! \brief Apply the DBC to the rhs vector and calculate and save the reaction forces
     *
     *  \note Stay in the global coordinate system (Rotation: global-->local-->global).*/
    void ApplyDirichletToRhs(Teuchos::RCP<Epetra_Vector>& b) const;

    //! Update the locsys manager
    void UpdateLocSysManager();

    //! Calculate the dirichlet increment of the current (time) step
    Teuchos::RCP<Epetra_Vector> GetDirichletIncrement();

    /*! \brief Evaluate and apply the DBC
     *
     * \note Stay in the global coordinate system (Rotation: global-->local-->global).*/
    virtual void ApplyDirichletBC(const double& time, Teuchos::RCP<Epetra_Vector> dis,
        Teuchos::RCP<Epetra_Vector> vel, Teuchos::RCP<Epetra_Vector> acc, bool recreatemap) const;

    /*! \brief Insert non-dbc dof values of source vector into the non-dbc dofs of target vector
     *
     *  \param[in] source_ptr Source vector with values to be inserted
     *  \param[in/out] target_ptr Target vector where values shall be inserted into
     */
    void InsertVectorInNonDbcDofs(
        Teuchos::RCP<const Epetra_Vector> source_ptr, Teuchos::RCP<Epetra_Vector> target_ptr) const;

    //! @name Access functions
    //!@{

    //! Get the Dirichlet Boundary Condition map extractor
    Teuchos::RCP<const CORE::LINALG::MapExtractor> GetDBCMapExtractor() const;

    //! Get a pointer to the local system manager
    Teuchos::RCP<DRT::UTILS::LocsysManager> LocSysManagerPtr();

    //! Get the zeros vector
    const Epetra_Vector& GetZeros() const;
    Teuchos::RCP<const Epetra_Vector> GetZerosPtr() const;

    //!@}

    //! Allows to expand dbc map with provided maptoadd
    void AddDirichDofs(const Teuchos::RCP<const Epetra_Map> maptoadd);

    //! Allows to contract dbc map with provided maptoremove
    void RemoveDirichDofs(const Teuchos::RCP<const Epetra_Map> maptoremove);

    /*! \brief Rotate the system matrix from a global to a local coordinate system
     *
     *  \pre #locsysman_ has to be defined.
     *
     *  \note Works only for CORE::LINALG::SparseMatrices.
     **/
    bool RotateGlobalToLocal(const Teuchos::RCP<CORE::LINALG::SparseOperator>& A) const;

    /*! \brief Rotate the rhs vector from the global to the local coordinate system
     *
     *  \pre #locsysman_ has to be defined.
     *
     *  \param[in] v Vector to be rotated
     */
    bool RotateGlobalToLocal(const Teuchos::RCP<Epetra_Vector>& v) const;

    /*! \brief Rotate the rhs vector from the global to the local coordinate system
     *
     *  \pre #locsysman_ has to be defined.
     *
     *  \param[in] v Vector to be rotated
     *  \param[in] offset ??
     */
    bool RotateGlobalToLocal(const Teuchos::RCP<Epetra_Vector>& v, bool offset) const;

    /*! \brief Rotate a vector from the local to the global coordinate system
     *
     *  \pre #locsysman_ has to be defined.
     *
     *  \param[in] v Vector to be rotated
     */
    bool RotateLocalToGlobal(const Teuchos::RCP<Epetra_Vector>& v) const;

    /*! \brief Rotate a vector from the local to the global coordinate system
     *
     *  \pre #locsysman_ has to be defined.
     *
     *  \param[in] v Vector to be rotated
     *  \param[in] offset ??
     */
    bool RotateLocalToGlobal(const Teuchos::RCP<Epetra_Vector>& v, bool offset) const;

   protected:
    //! Returns the initialization status
    const bool& IsInit() const { return isinit_; };

    //! Returns the setup status
    const bool& IsSetup() const { return issetup_; };

    //! Checks the initialization status
    void CheckInit() const;

    //! Checks the initialization and setup status
    void CheckInitSetup() const;

    //! Get discretization pointer
    Teuchos::RCP<DRT::Discretization> DiscretPtr();
    Teuchos::RCP<const DRT::Discretization> DiscretPtr() const;

    //! Access the reaction force
    Epetra_Vector& Freact() const;

    //! Get the locsys transformation matrix
    Teuchos::RCP<const CORE::LINALG::SparseMatrix> GetLocSysTrafo() const;

    //! Get the global state
    const STR::TIMINT::BaseDataGlobalState& GState() const;

    //! Has #locsysman_ be defined?
    bool IsLocSys() const { return islocsys_; };

    /*! \brief Extract the reaction forces
     *
     *  \param b ??
     */
    void ExtractFreact(Teuchos::RCP<Epetra_Vector>& b) const;

    /*! Apply the DBC to the right hand side in the local coordinate system and
     *  do not rotate it back to the global coordinate system. */
    void ApplyDirichletToLocalRhs(Teuchos::RCP<Epetra_Vector>& b) const;

    /*! \brief Apply the DBC to the Jacobian in the local coordinate system
     *
     *  \note This does not rotate the resutl back to the global coordinate system.
     *
     *  \param[in/out] A Jacobian matrix
     */
    void ApplyDirichletToLocalJacobian(Teuchos::RCP<CORE::LINALG::SparseOperator> A) const;

   protected:
    //! Flag indicating the initialization status.
    bool isinit_;

    //! Flag indicating the setup status.
    bool issetup_;

    //! Flag indicating if a #locsysman_ was defined.
    bool islocsys_;

    //! Discretization pointer
    Teuchos::RCP<DRT::Discretization> discret_ptr_;

    //! pointer to the overlying time integrator (read-only)
    Teuchos::RCP<const STR::TIMINT::Base> timint_ptr_;

    //! Pointer to the local coordinate system manager
    Teuchos::RCP<DRT::UTILS::LocsysManager> locsysman_ptr_;

    //! Some vector with system size and filled with zeros.
    Teuchos::RCP<Epetra_Vector> zeros_ptr_;

    //! Dirichlet boundary condition map extractor.
    Teuchos::RCP<CORE::LINALG::MapExtractor> dbcmap_ptr_;

   private:
    //! Reaction force
    Epetra_Vector* freact_ptr_;

  };  // namespace STR
}  // namespace STR

namespace NOX
{
  namespace NLN
  {
    namespace LinSystem
    {
      namespace PrePostOp
      {
        /*! \brief PrePostOperator class to modify the linear system before the linear system is
         * going to be solved.
         *
         * We use this pre/post operator to apply the DBC on the linear system of equations before
         * the linear system is going to be solved. This gives us the opportunity to rotate the
         * matrix only once, if locSys is defined and to apply possible modifications to the linear
         * system at different places without the need to re-apply the DBC (see PTC for an example).
         *
         * \author Hiermeier */
        class Dbc : public NOX::NLN::Abstract::PrePostOperator
        {
         public:
          //! constructor
          Dbc(const Teuchos::RCP<const STR::Dbc>& dbc_ptr);

          //! \brief Apply the DBC and rotate the system of equations if necessary (derived)
          void runPreApplyJacobianInverse(::NOX::Abstract::Vector& rhs,
              CORE::LINALG::SparseOperator& jac, const NOX::NLN::LinearSystem& linsys) override;

         private:
          //! pointer to the underlying class, which provides the whole functionality
          Teuchos::RCP<const STR::Dbc> dbc_ptr_;

        };  // class Dbc (pre/post operator)
      }     // namespace PrePostOp
    }       // namespace LinSystem
  }         // namespace NLN
}  // namespace NOX


FOUR_C_NAMESPACE_CLOSE

#endif

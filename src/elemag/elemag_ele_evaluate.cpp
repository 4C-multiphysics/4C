/*--------------------------------------------------------------------------*/
/*! \file
\brief electromagnetic element evaluation base routines

\level 2

*/
/*--------------------------------------------------------------------------*/

#include "elemag_ele_factory.H"
#include "inpar_elemag.H"
#include "elemag_ele_interface.H"
#include "elemag_ele.H"
#include "elemag_diff_ele.H"

/*---------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::ElemagType::PreEvaluate(DRT::Discretization& dis, Teuchos::ParameterList& p,
    Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix1,
    Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix2,
    Teuchos::RCP<Epetra_Vector> systemvector1, Teuchos::RCP<Epetra_Vector> systemvector2,
    Teuchos::RCP<Epetra_Vector> systemvector3)
{
  return;
}

/*---------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::Elemag::Evaluate(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, std::vector<int>& lm, Epetra_SerialDenseMatrix& elemat1,
    Epetra_SerialDenseMatrix& elemat2, Epetra_SerialDenseVector& elevec1,
    Epetra_SerialDenseVector& elevec2, Epetra_SerialDenseVector& elevec3)
{
  Teuchos::RCP<MAT::Material> mat = Material();

  if (dynamic_cast<const DRT::ELEMENTS::ElemagDiff*>(this))
    return DRT::ELEMENTS::ElemagFactory::ProvideImpl(Shape(), "diff")
        ->Evaluate(
            this, discretization, lm, params, mat, elemat1, elemat2, elevec1, elevec2, elevec3);
  else
    return DRT::ELEMENTS::ElemagFactory::ProvideImpl(Shape(), "std")
        ->Evaluate(
            this, discretization, lm, params, mat, elemat1, elemat2, elevec1, elevec2, elevec3);

  return -1;
}

/*---------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::Elemag::EvaluateNeumann(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Condition& condition, std::vector<int>& lm,
    Epetra_SerialDenseVector& elevec1, Epetra_SerialDenseMatrix* elemat1)
{
  return 0;
}

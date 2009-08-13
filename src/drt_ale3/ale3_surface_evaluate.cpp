//-----------------------------------------------------------------------
/*!
\file ale3_surface_evaluate.cpp

<pre>

</pre>
*/
//-----------------------------------------------------------------------
#ifdef D_ALE
#ifdef CCADISCRET

#include "ale3.H"
#include "../drt_lib/drt_discret.H"


int DRT::ELEMENTS::Ale3Surface::EvaluateNeumann(
  ParameterList& params,
  DRT::Discretization&      discretization,
  DRT::Condition&           condition,
  vector<int>&              lm,
  Epetra_SerialDenseVector& elevec1,
  Epetra_SerialDenseMatrix* elemat1)
{
  return 0;
}

#endif
#endif

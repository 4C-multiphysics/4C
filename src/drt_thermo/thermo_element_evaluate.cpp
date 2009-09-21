/*!
\file thermo_element_evaluate.cpp
\brief

<pre>
Maintainer: Caroline Danowski
            danowski@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15253
</pre>

*/
/*----------------------------------------------------------------------*
 |  definitions                                                gjb 01/08|
 *----------------------------------------------------------------------*/
#ifdef CCADISCRET
#ifdef D_THERMO

// This is just here to get the c++ mpi header, otherwise it would
// use the c version included inside standardtypes.h
#ifdef PARALLEL
#include "mpi.h"
#endif

/*----------------------------------------------------------------------*
 |  headers                                                    gjb 01/08|
 *----------------------------------------------------------------------*/
#include "thermo_element.H"
#include "../drt_thermo/thermo_ele_impl.H"

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                            dano 09/09 |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::Thermo::Evaluate(
    Teuchos::ParameterList&   params,
    DRT::Discretization&      discretization,
    std::vector<int>&         lm,
    Epetra_SerialDenseMatrix& elemat1,
    Epetra_SerialDenseMatrix& elemat2,
    Epetra_SerialDenseVector& elevec1,
    Epetra_SerialDenseVector& elevec2,
    Epetra_SerialDenseVector& elevec3)
{

  // all physics-related stuff is included in the implementation class that can
  // be used in principle inside any element (at the moment: only Thermo element)
  // If this element has special features/ methods that do not fit in the
  // generalized implementation class, you have to do a switch here in order to
  // call element-specific routines

  return DRT::ELEMENTS::TemperImplInterface::Impl(this)->Evaluate(
      this,
      params,
      discretization,
      lm,
      elemat1,
      elemat2,
      elevec1,
      elevec2,
      elevec3
      );

} //DRT::ELEMENTS::Thermo::Evaluate

/*----------------------------------------------------------------------*
 |  do nothing (public)                                      dano 09/09 |
 |                                                                      |
 |  The function is just a dummy. For the thermo elements, the          |
 |  integration of the volume neumann (body forces) loads takes place   |
 |  in the element. We need it there for the stabilisation terms!       |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::Thermo::EvaluateNeumann(Teuchos::ParameterList& params,
    DRT::Discretization&      discretization,
    DRT::Condition&           condition,
    std::vector<int>&         lm,
    Epetra_SerialDenseVector& elevec1,
    Epetra_SerialDenseMatrix* elemat1)
{
  return DRT::ELEMENTS::TemperImplInterface::Impl(this)->EvaluateNeumann(
      this,
      params,
      discretization,
      lm,
      elevec1,
      elemat1
      );
}


#endif  // #ifdef D_THERMO
#endif  // #ifdef CCADISCRET

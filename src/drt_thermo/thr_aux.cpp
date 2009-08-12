/*----------------------------------------------------------------------*/
/*!
\file thr_utils.cpp
\brief various auxiliar methods needed in thermal analysis

<pre>
Maintainer: Burkhard Bornemann
            bornemann@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15237

            05.08.09 changed by bb, cd
</pre>
*/

/*----------------------------------------------------------------------*/
/* definitions */
#ifdef CCADISCRET

/*----------------------------------------------------------------------*/
/* headers */
#include "thr_aux.H"

/*----------------------------------------------------------------------*/
/* Calculate vector norm */
double THR::AUX::CalculateVectorNorm
(
  const enum INPAR::THR::VectorNorm norm,
  const Teuchos::RCP<Epetra_Vector> vect
)
{
  // L1 norm
  if (norm == INPAR::THR::norm_l1)
  {
    double vectnorm;
    vect->Norm1(&vectnorm);
    return vectnorm;
  }
  // L2/Euclidian norm
  else if (norm == INPAR::THR::norm_l2)
  {
    double vectnorm;
    vect->Norm2(&vectnorm);
    return vectnorm;
  }
  // RMS norm
  else if (norm == INPAR::THR::norm_rms)
  {
    double vectnorm;
    vect->Norm2(&vectnorm);
    return vectnorm/std::sqrt((double) vect->GlobalLength());
  }
  // infinity/maximum norm
  else if (norm == INPAR::THR::norm_inf)
  {
    double vectnorm;
    vect->NormInf(&vectnorm);
    return vectnorm;
  }
  else
  {
    dserror("Cannot handle vector norm");
    return 0;
  }
}

/*----------------------------------------------------------------------*/
#endif  // #ifdef CCADISCRET

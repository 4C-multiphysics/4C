/*
 * solver_mlpreconditioner.cpp
 *
 *  Created on: Jul 4, 2011
 *      Author: wiesner
 */

#include "../drt_lib/drt_dserror.H"

#include "ml_common.h"
#include "ml_include.h"
#include "ml_epetra_utils.h"
#include "ml_epetra.h"
#include "ml_epetra_operator.h"
#include "ml_MultiLevelPreconditioner.h"

#include "linalg_mlapi_operator.H"  // Michael's MLAPI based ML preconditioner
#include "amgpreconditioner.H"      // Tobias' smoothed aggregation AMG implementation in BACI (only for fluids)

#include "solver_mlpreconditioner.H"

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
LINALG::SOLVER::MLPreconditioner::MLPreconditioner( FILE * outfile, Teuchos::ParameterList & mllist )
  : PreconditionerType( outfile ),
    mllist_( mllist )
{
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void LINALG::SOLVER::MLPreconditioner::Setup( bool create,
                                              Epetra_Operator * matrix,
                                              Epetra_MultiVector * x,
                                              Epetra_MultiVector * b )
{
  SetupLinearProblem( matrix, x, b );

  if ( create )
  {
    Epetra_CrsMatrix* A = dynamic_cast<Epetra_CrsMatrix*>( matrix );
    if ( A==NULL )
      dserror( "CrsMatrix expected" );

    // free old matrix first
    P_       = Teuchos::null;
    Pmatrix_ = Teuchos::null;

    // create a copy of the scaled matrix
    // so we can reuse the preconditioner
    Pmatrix_ = Teuchos::rcp(new Epetra_CrsMatrix(*A));

    // see whether we use standard ml or our own mlapi operator
    const bool domlapioperator = mllist_.get<bool>("LINALG::AMG_Operator",false);
    const bool doamgpreconditioner = mllist_.get<bool>("LINALG::AMGPreconditioner",false);
    if (domlapioperator)
    {
      P_ = Teuchos::rcp(new LINALG::AMG_Operator(Pmatrix_,mllist_,true));
    }
    else if (doamgpreconditioner)
    {
      P_ = Teuchos::rcp(new LINALG::AMGPreconditioner(Pmatrix_,mllist_,outfile_));
    }
    else
    {
      P_ = Teuchos::rcp(new ML_Epetra::MultiLevelPreconditioner(*Pmatrix_,mllist_,true));

      // for debugging ML
      //dynamic_cast<ML_Epetra::MultiLevelPreconditioner&>(*P_).PrintUnused(0);
    }
  }
}

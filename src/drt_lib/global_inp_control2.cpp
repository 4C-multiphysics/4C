/*!----------------------------------------------------------------------
\file
\brief

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>

#ifdef PARALLEL
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "global_inp_control2.H"

#include "drt_inputreader.H"


/*----------------------------------------------------------------------*
  |                                                       m.gee 06/01    |
  | general problem data                                                 |
  | global variable GENPROB genprob is defined in global_control.c       |
 *----------------------------------------------------------------------*/
extern struct _GENPROB     genprob;


/*!----------------------------------------------------------------------
  \brief file pointers

  <pre>                                                         m.gee 8/00
  This structure struct _FILES allfiles is defined in input_control_global.c
  and the type is in standardtypes.h
  It holds all file pointers and some variables needed for the FRSYSTEM
  </pre>
 *----------------------------------------------------------------------*/
extern struct _FILES  allfiles;


/*----------------------------------------------------------------------*
  | input of control, element and load information         m.gee 10/06  |
  | This version of the routine uses the new discretization subsystem   |
  | ccadiscret                                                          |
 *----------------------------------------------------------------------*/
void ntainp_ccadiscret()
{
#ifdef PARALLEL
  Epetra_MpiComm* com = new Epetra_MpiComm(MPI_COMM_WORLD);
  Teuchos::RCP<Epetra_Comm> comm = rcp(com);
#else
  Epetra_SerialComm* com = new Epetra_SerialComm();
  Teuchos::RCP<Epetra_Comm> comm = rcp(com);
#endif

  Teuchos::RCP<DRT::Problem> problem = DRT::Problem::Instance();

  // and now the actual reading
  DRT::INPUT::DatFileReader reader(allfiles.inputfile_name, comm);
  reader.Activate();

  problem->ReadParameter(reader);

  /* input of not mesh or time based problem data  */
  problem->InputControl();

  /* input of materials */
  problem->ReadMaterial();

  /* input of fields */
  problem->ReadFields(reader);

  // read dynamic control data
  if (genprob.timetyp==time_dynamic)
  {
    // nothing to do! We do not use alldyn anymore!
  }
  // read static control data
  else inpctrstat();

  // read all types of geometry related conditions (e.g. boundary conditions)
  // Also read time and space functions and local coord systems
  problem->ReadConditions(reader);

  // read all knot information for isogeometric analysis
  // and add it to the (derived) nurbs discretization
  problem->ReadKnots(reader);

  /*---------------------------------------- input of result descriptions */
  inp_resultdescr();

  // all reading is done at this point!

  // create control file for output and read restart data if required
  problem->OpenControlFile(*comm,
                           allfiles.inputfile_name,
                           allfiles.outputfile_kenner);

  return;
} // end of ntainp_ccadiscret()


#endif  // #ifdef CCADISCRET

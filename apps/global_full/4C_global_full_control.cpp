/*----------------------------------------------------------------------*/
/*! \file

\brief the basic 4C routine

\level 0


*/
/*----------------------------------------------------------------------*/

#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_global_full_init_control.hpp"
#include "4C_global_full_inp_control.hpp"
#include "4C_io_pstream.hpp"


namespace
{
  double walltime_in_seconds()
  {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
               .count() *
           1.0e-3;
  }
}  // namespace

void ntacal();

/*----------------------------------------------------------------------*
 | main routine                                           m.gee 8/00    |
 *----------------------------------------------------------------------*/
void ntam(int argc, char *argv[])
{
  using namespace FourC;

  Teuchos::RCP<Epetra_Comm> gcomm = Global::Problem::Instance()->GetCommunicators()->GlobalComm();

  double t0, ti, tc;

  /// IO file names and kenners
  std::string inputfile_name;
  std::string outputfile_kenner;
  std::string restartfile_kenner;

  ntaini_ccadiscret(argc, argv, inputfile_name, outputfile_kenner, restartfile_kenner);

  /* input phase, input of all information */

  t0 = walltime_in_seconds();

  ntainp_ccadiscret(inputfile_name, outputfile_kenner, restartfile_kenner);

  ti = walltime_in_seconds() - t0;
  if (gcomm->MyPID() == 0)
  {
    Core::IO::cout << "\nTotal CPU Time for INPUT:       " << std::setw(10) << std::setprecision(3)
                   << std::scientific << ti << " sec \n\n";
  }

  /*--------------------------------------------------calculation phase */
  t0 = walltime_in_seconds();

  ntacal();

  tc = walltime_in_seconds() - t0;
  if (gcomm->MyPID() == 0)
  {
    Core::IO::cout << "\nTotal CPU Time for CALCULATION: " << std::setw(10) << std::setprecision(3)
                   << std::scientific << tc << " sec \n\n";
  }
}

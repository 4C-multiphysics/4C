/*----------------------------------------------------------------------*/
/*! \file

\brief A collection of dense matrix printing methods for namespace CORE::LINALG

\level 0
*/
/*----------------------------------------------------------------------*/

#include "baci_linalg_utils_densematrix_print.H"

#include <fstream>

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void CORE::LINALG::PrintSerialDenseMatrixInMatlabFormat(
    std::string filename, const CORE::LINALG::SerialDenseMatrix& A, const bool newfile)
{
  std::ofstream os;

  // open file for writing
  if (newfile)
    os.open(filename.c_str(), std::fstream::trunc);
  else
    os.open(filename.c_str(), std::fstream::ate | std::fstream::app);

  const int NumMyRows = A.numRows();
  const int NumMyColumns = A.numCols();

  for (int i = 0; i < NumMyRows; i++)
  {
    for (int j = 0; j < NumMyColumns; j++)
    {
      os << std::setw(10) << i + 1;  // increase index by one for matlab
      os << std::setw(10) << j + 1;  // increase index by one for matlab
      os << std::setw(30) << std::setprecision(16) << std::scientific << A(i, j);
      os << std::endl;
    }
  }

  os << std::flush;

  // close file
  os.close();

  // just to be sure
  if (os.is_open()) os.close();
}

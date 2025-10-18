// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cardiovascular0d_mor_pod.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_utils_string.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Cardiovascular0D::ProperOrthogonalDecomposition::ProperOrthogonalDecomposition(
    std::shared_ptr<const Core::LinAlg::Map> full_model_dof_row_map,
    const std::string& pod_matrix_file_name, const std::string& absolute_path_to_input_file)
    : full_model_dof_row_map_(full_model_dof_row_map)
{
  // check if model order reduction is given in input file
  {
    std::vector<std::string> components_of_absolute_path =
        Core::Utils::split_string_list(pod_matrix_file_name, "/");
    if (components_of_absolute_path.back() != std::string("none")) havemor_ = true;
  }

  // no mor? -> exit
  if (not havemor_) return;

  // A multi-vector to store basis vectors to be read from file
  std::shared_ptr<Core::LinAlg::MultiVector<double>> reduced_basis = nullptr;

  // read projection matrix from binary file
  {
    std::string absolute_path_to_pod_file = pod_matrix_file_name;

    // Make sure that we have the absolute path to the file
    if (pod_matrix_file_name[0] != '/')
    {
      std::string::size_type pos = absolute_path_to_input_file.rfind('/');
      if (pos != std::string::npos)
      {
        std::string path = absolute_path_to_input_file.substr(0, pos + 1);
        absolute_path_to_pod_file.insert(
            absolute_path_to_pod_file.begin(), path.begin(), path.end());
      }
    }

    read_pod_basis_vectors_from_file(absolute_path_to_pod_file, reduced_basis);
  }

  // build an importer
  Core::LinAlg::Import dofrowimporter(*full_model_dof_row_map_, reduced_basis->get_map());
  projmatrix_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *full_model_dof_row_map_, reduced_basis->num_vectors(), true);
  projmatrix_->import(*reduced_basis, dofrowimporter, Insert, nullptr);

  // check row dimension
  if (projmatrix_->global_length() != full_model_dof_row_map_->num_global_elements())
    FOUR_C_THROW("Projection matrix does not match discretization.");

  // check orthogonality
  if (not is_pod_basis_orthogonal(*projmatrix_))
    FOUR_C_THROW("Projection matrix is not orthogonal.");

  // maps for reduced system
  structmapr_ = std::make_shared<Core::LinAlg::Map>(
      projmatrix_->num_vectors(), 0, full_model_dof_row_map_->get_comm());
  redstructmapr_ = std::make_shared<Core::LinAlg::Map>(projmatrix_->num_vectors(),
      projmatrix_->num_vectors(), 0, full_model_dof_row_map_->get_comm());
  // Core::LinAlg::allreduce_e_map cant't be used here, because NumGlobalElements will be chosen
  // wrong

  // importers for reduced system
  structrimpo_ = std::make_shared<Core::LinAlg::Import>(*structmapr_, *redstructmapr_);
  structrinvimpo_ = std::make_shared<Core::LinAlg::Import>(*redstructmapr_, *structmapr_);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix>
Cardiovascular0D::ProperOrthogonalDecomposition::reduce_diagonal(Core::LinAlg::SparseMatrix& M)
{
  // right multiply M * V
  Core::LinAlg::MultiVector<double> M_tmp(M.row_map(), projmatrix_->num_vectors(), true);
  int err = M.multiply(false, *projmatrix_, M_tmp);
  if (err) FOUR_C_THROW("Multiplication M * V failed.");

  // left multiply V^T * (M * V)
  std::shared_ptr<Core::LinAlg::MultiVector<double>> M_red_mvec =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*structmapr_, M_tmp.num_vectors(), true);
  multiply_multi_vectors(
      *projmatrix_, 'T', M_tmp, 'N', *redstructmapr_, *structrimpo_, *M_red_mvec);

  // convert Core::LinAlg::MultiVector<double> to Core::LinAlg::SparseMatrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> M_red =
      std::make_shared<Core::LinAlg::SparseMatrix>(*structmapr_, 0, false, true);
  Core::LinAlg::multi_vector_to_linalg_sparse_matrix(
      *M_red_mvec, *structmapr_, *structmapr_, *M_red);

  return M_red;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix>
Cardiovascular0D::ProperOrthogonalDecomposition::reduce_off_diagonal(Core::LinAlg::SparseMatrix& M)
{
  // right multiply M * V
  std::shared_ptr<Core::LinAlg::MultiVector<double>> M_tmp =
      std::make_shared<Core::LinAlg::MultiVector<double>>(
          M.domain_map(), projmatrix_->num_vectors(), true);
  int err = M.multiply(true, *projmatrix_, *M_tmp);
  if (err) FOUR_C_THROW("Multiplication V^T * M failed.");

  // convert Core::LinAlg::MultiVector<double> to Core::LinAlg::SparseMatrix
  std::shared_ptr<Core::LinAlg::Map> rangemap = std::make_shared<Core::LinAlg::Map>(M.domain_map());
  std::shared_ptr<Core::LinAlg::SparseMatrix> M_red =
      std::make_shared<Core::LinAlg::SparseMatrix>(*rangemap, 0, false, true);
  Core::LinAlg::multi_vector_to_linalg_sparse_matrix(*M_tmp, *rangemap, *structmapr_, *M_red);

  return M_red;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>>
Cardiovascular0D::ProperOrthogonalDecomposition::reduce_rhs(Core::LinAlg::MultiVector<double>& v)
{
  std::shared_ptr<Core::LinAlg::MultiVector<double>> v_red =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*structmapr_, 1, true);
  multiply_multi_vectors(*projmatrix_, 'T', v, 'N', *redstructmapr_, *structrimpo_, *v_red);

  return v_red;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Cardiovascular0D::ProperOrthogonalDecomposition::reduce_residual(Core::LinAlg::Vector<double>& v)
{
  Core::LinAlg::Vector<double> v_tmp(*redstructmapr_);
  int err = v_tmp.multiply('T', 'N', 1.0, *projmatrix_, v, 0.0);
  if (err) FOUR_C_THROW("Multiplication V^T * v failed.");

  std::shared_ptr<Core::LinAlg::Vector<double>> v_red =
      std::make_shared<Core::LinAlg::Vector<double>>(*structmapr_);
  v_red->import(v_tmp, *structrimpo_, Insert, nullptr);

  return v_red;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Cardiovascular0D::ProperOrthogonalDecomposition::extend_solution(
    Core::LinAlg::Vector<double>& v_red)
{
  Core::LinAlg::Vector<double> v_tmp(*redstructmapr_, true);
  v_tmp.import(v_red, *structrinvimpo_, Insert, nullptr);
  std::shared_ptr<Core::LinAlg::Vector<double>> v =
      std::make_shared<Core::LinAlg::Vector<double>>(*full_model_dof_row_map_);
  int err = v->multiply('N', 'N', 1.0, *projmatrix_, v_tmp, 0.0);
  if (err) FOUR_C_THROW("Multiplication V * v_red failed.");

  return v;
}


/*----------------------------------------------------------------------*
 | read binary projection matrix from file                pfaller Oct17 |
 |                                                                      |
 | MATLAB code to write projmatrix(DIM x dim):                          |
 |                                                                      |
 |   fid=fopen(filename, 'w');                                          |
 |   fwrite(fid, size(projmatrix, 1), 'int');                           |
 |   fwrite(fid, size(projmatrix, 2), 'int');                           |
 |   fwrite(fid, projmatrix', 'single');                                |
 |   fclose(fid);                                                       |
 |                                                                      |
 *----------------------------------------------------------------------*/
void Cardiovascular0D::ProperOrthogonalDecomposition::read_pod_basis_vectors_from_file(
    const std::string& absolute_path_to_pod_file,
    std::shared_ptr<Core::LinAlg::MultiVector<double>>& projmatrix)
{
  // ***************************
  // PART1: Read in Matrix Sizes
  // ***************************

  // open binary file
  std::ifstream file1(absolute_path_to_pod_file.c_str(),
      std::ifstream::in | std::ifstream::binary | std::ifstream::ate);

  if (!file1.good())
    FOUR_C_THROW("File containing the matrix could not be opened. Check Input-File.");

  // allocation of a memory-block to read the data
  char* sizeblock = new char[8];

  // jump to beginning of file
  file1.seekg(0, std::ifstream::beg);

  // read the first 8 byte into the memory-block
  file1.read(sizeblock, std::streamoff(8));

  // close the file
  file1.close();

  // union for conversion of 4 bytes to int or float
  union CharIntFloat
  {
    char ValueAsChars[4];
    int ValueAsInt;
    float ValueAsFloat;
  } NumRows, NumCols;

  // number of rows
  for (int k = 0; k < 4; k++) NumRows.ValueAsChars[k] = sizeblock[k];

  // number of columns
  for (int k = 0; k < 4; k++) NumCols.ValueAsChars[k] = sizeblock[k + 4];

  delete[] sizeblock;

  // allocate multivector according to matrix size:
  std::shared_ptr<Core::LinAlg::Map> mymap = std::make_shared<Core::LinAlg::Map>(
      NumRows.ValueAsInt, 0, full_model_dof_row_map_->get_comm());
  projmatrix = std::make_shared<Core::LinAlg::MultiVector<double>>(*mymap, NumCols.ValueAsInt);


  // ***************************
  // PART2: Read In Matrix
  // ***************************

  // select the format we want to import matrices in
  const int formatfactor = 4;  // 8 is double 4 is single

  // union for conversion of some bytes to int or float
  union NewCharIntFloat
  {
    char VAsChar[formatfactor];
    double VAsDbl;
    float VAsFlt;
  } Val;

  // calculate a number of bytes that are needed to be reserved in order to fill the multivector
  int mysize = projmatrix->num_vectors() * projmatrix->local_length() * formatfactor;

  // open binary file (again)
  std::ifstream file2(absolute_path_to_pod_file.c_str(),
      std::ifstream::in | std::ifstream::binary | std::ifstream::ate);

  // allocation of a memory-block to read the data
  char* memblock = new char[mysize];

  // calculation of starting points in matrix for each processor
  MPI_Comm comm(full_model_dof_row_map_->get_comm());
  const int numproc(Core::Communication::num_mpi_ranks(comm));
  const int mypid(Core::Communication::my_mpi_rank(comm));
  std::vector<int> localnumbers(numproc, 0);
  std::vector<int> globalnumbers(numproc, 0);
  localnumbers[mypid] = mymap->num_my_elements();
  globalnumbers = Core::Communication::sum_all(localnumbers, comm);

  int factor(0);
  for (int i = 0; i < mypid; i++) factor += globalnumbers[i];

  Core::Communication::barrier(comm);

  // 64 bit number necessary, as integer can overflow for large matrices
  long long start =
      (long long)factor * (long long)projmatrix->num_vectors() * (long long)formatfactor;

  // leads to a starting point:
  file2.seekg(8 + start, std::ifstream::beg);

  // read into memory
  file2.read(memblock, std::streamoff(mysize));

  // close the file
  file2.close();

  // loop over columns and fill double array
  for (int i = 0; i < projmatrix->num_vectors(); i++)
  {
    // loop over all rows owned by the calling processor
    for (int j = 0; j < projmatrix->local_length(); j++)
    {
      // current value
      for (int k = 0; k < formatfactor; k++)
        Val.VAsChar[k] = memblock[j * formatfactor * NumCols.ValueAsInt + i * formatfactor + k];

      // write current value to Multivector
      projmatrix->replace_local_value(j, i, double(Val.VAsFlt));
    }
  }

  // delete memory-block
  delete[] memblock;

  // all procs wait until proc number 0 did finish the stuff before
  Core::Communication::barrier(comm);

  // Inform user
  if (Core::Communication::my_mpi_rank(comm) == 0) std::cout << " --> Successful\n" << std::endl;

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Cardiovascular0D::ProperOrthogonalDecomposition::is_pod_basis_orthogonal(
    const Core::LinAlg::MultiVector<double>& M)
{
  const int n = M.num_vectors();

  // calculate V^T * V (should be an nxn identity matrix)
  Core::LinAlg::Map map = Core::LinAlg::Map(n, n, 0, full_model_dof_row_map_->get_comm());
  Core::LinAlg::MultiVector<double> identity = Core::LinAlg::MultiVector<double>(map, n, true);
  identity.multiply('T', 'N', 1.0, M, M, 0.0);

  // subtract one from diagonal
  for (int i = 0; i < n; ++i) identity.sum_into_global_value(i, i, -1.0);

  // inf norm of columns
  double* norms = new double[n];
  identity.norm_inf(norms);

  for (int i = 0; i < n; ++i)
    if (norms[i] > 1.0e-7) return false;

  return true;
}

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_COMM_UTILS_HPP
#define FOUR_C_COMM_UTILS_HPP


#include "4C_config.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_sparsematrix.hpp"

#include <Teuchos_DefaultMpiComm.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::Communication
{
  // forward declaration
  class Communicators;

  /**
   * The known types for nested parallelism.
   */
  enum class NestedParallelismType
  {
    every_group_read_input_file,
    separate_input_files,
    nested_multiscale,
    no_nested_parallelism
  };

  /**
   * Configuration for communicators.
   */
  struct CommConfig
  {
    int ngroup = 1;
    std::vector<int> group_layout;
    NestedParallelismType np_type;
    int diffgroup = -1;
  };

  //! create a local and a global communicator for the problem
  Communicators create_comm(CommConfig& config);

  /*! \brief debug routine to compare vectors from different parallel 4C runs
   *
   * You can add Core::Communication::are_distributed_vectors_identical in your code which will lead
   * to a comparison of the given vector for different executables and/or configurations. Command
   * for using this feature:
   * @code
   * mpirun -np 1 ./4C --nptype=diffgroup0 <input_1> xxx_set \
   * : -np 3 ./other-4C --nptype=diffgroup1 <input_2> xxx_par
   * @endcode
   *
   * A further nice option is to compare results from different executables used for
   * running the same simulation.
   *
   * \note You need to add the are_distributed_vectors_identical method in both executables at the
   * same position in the code
   *
   * \param communicators (in): communicators containing local and global comm
   * \param vec           (in): vector to compare
   * \param name          (in): user given name for the vector (needs to match within gcomm)
   * \param tol           (in): comparison tolerance for infinity norm
   * \return boolean to indicate if compared vectors are identical
   */
  bool are_distributed_vectors_identical(const Communicators& communicators,
      const Core::LinAlg::MultiVector<double>& vec, const char* name, double tol = 1.0e-14);

  /*! \brief debug routine to compare sparse matrices from different parallel 4C runs
   *
   * You can add Core::Communication::are_distributed_sparse_matrices_identical in your code which
   * will lead to a comparison of the given sparse matrices for different executables and/or
   * configurations. Command for using this feature:
   * @code
   * mpirun -np 1 ./4C --nptype=diffgroup0 <input_1> xxx_set \
   * : -np 3 ./other-4C --nptype=diffgroup1 <input_2> xxx_par
   * @endcode
   *
   * A further nice option is to compare results from different executables used for
   * running the same simulation.
   *
   * \note You need to add the are_distributed_sparse_matrices_identical method in both executables
   * at the same position in the code.
   *
   * \param communicators (in): communicators containing local and global comm
   * \param matrix        (in): matrix to compare
   * \param name          (in): user given name for the matrix (needs to match within gcomm)
   * \param tol           (in): comparison tolerance for infinity norm
   * \return boolean to indicate if compared vectors are identical
   */
  bool are_distributed_sparse_matrices_identical(const Communicators& communicators,
      const Core::LinAlg::SparseMatrix& matrix, const char* name, double tol = 1.0e-14);

  //! transform MPI_Comm to Teuchos::Comm, std::shared_ptr version
  template <class Datatype>
  std::shared_ptr<const Teuchos::Comm<Datatype>> to_teuchos_comm(MPI_Comm comm)
  {
    return std::make_shared<Teuchos::MpiComm<Datatype>>(comm);
  }


  /**
   * A class to gather various MPI_Comm objects.
   */
  class Communicators
  {
   public:
    Communicators(int groupId, int ngroup, std::map<int, int> lpidgpid, MPI_Comm lcomm,
        MPI_Comm gcomm, NestedParallelismType npType);

    /// return group id
    int group_id() const { return group_id_; }

    /// return number of groups
    int num_groups() const { return ngroup_; }

    /// return local communicator
    MPI_Comm local_comm() const { return lcomm_; }

    /// return local communicator
    MPI_Comm global_comm() const { return gcomm_; }

    /// set a sub group communicator
    void set_sub_comm(MPI_Comm subcomm);

    /// return sub group communicator
    MPI_Comm sub_comm() const { return subcomm_; }

    /// return nested parallelism type
    NestedParallelismType np_type() const { return np_type_; }

    //! Cleanup MPI communicators
    void finalize()
    {
      if (lcomm_ != MPI_COMM_NULL)
      {
        MPI_Comm_free(&lcomm_);
        lcomm_ = MPI_COMM_NULL;
      }

      if (subcomm_ != MPI_COMM_NULL)
      {
        MPI_Comm_free(&subcomm_);
        subcomm_ = MPI_COMM_NULL;
      }
    }

   private:
    /// group id
    int group_id_;

    /// number of groups
    int ngroup_;

    /// map from local processor ids to global processor ids
    std::map<int, int> lpidgpid_;

    /// local communicator
    MPI_Comm lcomm_;

    /// global communicator
    MPI_Comm gcomm_;

    /// sub communicator
    MPI_Comm subcomm_;

    /// nested parallelism type
    NestedParallelismType np_type_;
  };


}  // namespace Core::Communication

FOUR_C_NAMESPACE_CLOSE

#endif

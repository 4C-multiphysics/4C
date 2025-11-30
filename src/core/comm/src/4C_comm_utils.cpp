// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_utils.hpp"

#include "4C_io_pstream.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_transfer.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Communication
{
  /*----------------------------------------------------------------------*
   | create communicator                                      ghamm 02/12 |
   *----------------------------------------------------------------------*/
  Communicators create_comm(std::vector<std::string> argv)
  {
    // for coupled simulations: color = 1 for 4C and color = 0 for other programs
    // so far: either nested parallelism within 4C or coupling with further
    // executables is possible
    // default values without nested parallelism
    int myrank = -1;
    int size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int color = 0;
    int ngroup = 1;
    NestedParallelismType npType = NestedParallelismType::no_nested_parallelism;

    // parse command line and separate configuration arguments
    std::vector<std::string> conf;
    for (std::size_t i = 1; i < argv.size(); i++)
    {
      std::string temp = argv[i];
      if (temp.substr(0, 1) == "-")
      {
        conf.push_back(argv[i]);
      }
    }

    // grouplayout will be filled accordingly to the user given input
    std::vector<int> grouplayout;
    bool ngroupisset = false;
    bool nptypeisset = false;
    for (std::size_t i = 0; i < conf.size(); i++)
    {
      // fill std::string with current argument
      std::string argument(conf[i]);
      //----------------------------------------------------------------
      // determine number of groups and the proc distribution
      //----------------------------------------------------------------
      if (argument.substr(0, 8) == "-ngroup=")
      {
        ngroupisset = true;
        ngroup = atoi(argument.substr(8, std::string::npos).c_str());

        // read out argument after ngroup=
        std::string glayout;
        if (i + 1 < conf.size())
          glayout = conf[i + 1];
        else
          glayout = "dummy";

        // case with given group layout
        if (glayout.substr(0, 9) == "-glayout=")
        {
          glayout = glayout.substr(9, std::string::npos).c_str();

          std::istringstream layout(glayout);
          int sumprocs = 0;

          while (layout)
          {
            std::string s;
            if (!getline(layout, s, ',')) break;
            grouplayout.push_back(atoi(s.c_str()));
            sumprocs += atoi(s.c_str());
          }

          // final check whether a correct group layout is specified
          if (ngroup != int(grouplayout.size()) or size != sumprocs or ngroup < 1)
          {
            if (myrank == (size - 1))  // myrank == 0 is eventually not within 4C (i.e. coupling
                                       // to external codes)
            {
              printf(
                  "\n\nNumber of procs (%d) and number of groups (%d) does not match given group "
                  "layout! \n",
                  size, ngroup);
              printf("Example mpirun -np 4 ./4C -ngroup=2 -glayout=1,3 \n");
              printf("Try again!\n");
            }
            MPI_Finalize();
            exit(EXIT_FAILURE);
          }
        }
        // case without given group layout
        else
        {
          if (myrank == (size - 1))  // myrank == 0 is eventually not within 4C (i.e. coupling to
                                     // external codes)
          {
            printf(
                "\n\n\nINFO: Group layout is not specified. Default is equal size of the "
                "groups.\n");
          }
          if ((size % ngroup) != 0)
          {
            if (myrank == (size - 1))
            {
              printf(
                  "\n\nNumber of processors (%d) cannot be divided by the number of groups (%d)!\n",
                  size, ngroup);
              printf("Try again!\n");
            }
            MPI_Finalize();
            exit(EXIT_FAILURE);
          }

          // equal size of the groups
          for (int k = 0; k < ngroup; k++)
          {
            grouplayout.push_back(size / ngroup);
          }
        }

        // the color is specified: procs are distributed to the groups with increasing global rank
        color = -1;
        int gsum = 0;
        do
        {
          color++;
          gsum += grouplayout[color];
        } while (gsum <= myrank);

#ifdef FOUR_C_ENABLE_ASSERTIONS
        std::cout << "Nested parallelism layout: Global rank: " << myrank
                  << " is in group: " << color << std::endl;
#endif

      }  // end if (argument.substr( 0, 8 ) == "-ngroup=")

      //----------------------------------------------------------------
      // nested parallelism type
      //----------------------------------------------------------------
      else if (argument.substr(0, 8) == "-nptype=")
      {
        nptypeisset = true;
        argument = argument.substr(8, std::string::npos).c_str();
        if (argument == "everyGroupReadInputFile")
          npType = NestedParallelismType::every_group_read_input_file;
        else if (argument == "separateInputFiles")
          npType = NestedParallelismType::separate_input_files;
        else if (argument == "nestedMultiscale")
        {
          npType = NestedParallelismType::separate_input_files;
          // the color is specified: only two groups and group one (macro problem) is distributed
          // over all processors
          color = -1;
          if (myrank % (int)(size / grouplayout[0]) == 0 and
              myrank < (grouplayout[0] * (int)(size / grouplayout[0])))
            color = 0;
          else
            color = 1;
        }
        else if (argument.substr(0, 9) == "diffgroup")
        {
          npType = NestedParallelismType::no_nested_parallelism;
          ngroup = 2;
          color = atoi(argument.substr(9, std::string::npos).c_str());
        }
        else
        {
          if (myrank == (size - 1))  // myrank == 0 is eventually not within 4C (i.e. coupling to
                                     // external codes)
          {
            printf(
                "\n\nOnly everyGroupReadInputFile and separateInputFiles is available for "
                "nptype=  \n\n");
            printf("Try again!\n");
          }
          MPI_Finalize();
          exit(EXIT_FAILURE);
        }
      }

      //----------------------------------------------------------------
      // check for valid arguments that can be used in 4C.cpp
      //----------------------------------------------------------------
      else if ((argument.substr(0, 9) != "-glayout=") and (argument.substr(0, 2) != "-v") and
               (argument.substr(0, 2) != "-h") and (argument.substr(0, 6) != "--help") and
               (argument.substr(0, 2) != "-p") and (argument.substr(0, 12) != "--parameters") and
               (argument.substr(0, 2) != "-d") and (argument.substr(0, 9) != "--datfile") and
               (argument.substr(0, 13) != "--interactive"))
      {
        printf(
            "\n\n You have specified an argument ( %s ) for 4C starting with a \"-\" that is not "
            "valid!\n",
            argument.c_str());
        printf("Please refer to ./4C --help and try again!\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
      }

    }  // end for(int i=0; i<int(conf.size()); i++)


    if ((int(conf.size()) > 1) and (ngroupisset == false or nptypeisset == false))
    {
      if (myrank ==
          (size - 1))  // myrank == 0 is eventually not within 4C (i.e. coupling to external codes)
      {
        printf(
            "\n\nAt least -nptype= and -ngroup= must be specified for nested parallelism. -glayout "
            "is optional (behind -ngroup).  \n\n");
        printf("Try again!\n");
      }
      MPI_Finalize();
      exit(EXIT_FAILURE);
    }

    // do the splitting of the communicator
    MPI_Comm lcomm;
    MPI_Comm_split(MPI_COMM_WORLD, color, myrank, &lcomm);

    // the global communicator is created
    MPI_Comm gcomm;

    if (ngroup == 1)
    {
      gcomm = lcomm;
    }
    else
    {
      // TODO: consider a second executable that is coupled to 4C in case of nested parallelism
      // TODO: the procs owned by another executable have to be removed from world_group, e.g.
      // MPI_Group_excl
      MPI_Comm mpi_global_comm;
      MPI_Group world_group;
      MPI_Comm_group(MPI_COMM_WORLD, &world_group);
      MPI_Comm_create(MPI_COMM_WORLD, world_group, &mpi_global_comm);
      MPI_Group_free(&world_group);

      gcomm = mpi_global_comm;
    }

    // mapping of local proc ids to global proc ids
    std::map<int, int> lpidgpid;
    int localsize = Core::Communication::num_mpi_ranks(lcomm);
    for (int lpid = 0; lpid < localsize; lpid++)
    {
      lpidgpid[lpid] =
          Core::Communication::my_mpi_rank(gcomm) - Core::Communication::my_mpi_rank(lcomm) + lpid;
    }

    // nested parallelism group is created
    Communicators communicators(color, ngroup, lpidgpid, lcomm, gcomm, npType);

    // info for the nested parallelism user
    if (Core::Communication::my_mpi_rank(lcomm) == 0 && ngroup > 1)
      printf("Nested parallelism layout: Group %d has %d processors.\n ", color,
          Core::Communication::num_mpi_ranks(lcomm));
    fflush(stdout);

    // for sync of output
    Core::Communication::barrier(gcomm);

    return communicators;
  }

  /*----------------------------------------------------------------------*
   | constructor communicators                                ghamm 03/12 |
   *----------------------------------------------------------------------*/
  Communicators::Communicators(int groupId, int ngroup, std::map<int, int> lpidgpid, MPI_Comm lcomm,
      MPI_Comm gcomm, NestedParallelismType npType)
      : group_id_(groupId),
        ngroup_(ngroup),
        lpidgpid_(lpidgpid),
        lcomm_(lcomm),
        gcomm_(gcomm),
        subcomm_(MPI_COMM_NULL),
        np_type_(npType)
  {
    return;
  }


  /*----------------------------------------------------------------------*
   | set sub communicator                                     ghamm 04/12 |
   *----------------------------------------------------------------------*/
  void Communicators::set_sub_comm(MPI_Comm subcomm)
  {
    subcomm_ = subcomm;
    return;
  }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  bool are_distributed_vectors_identical(const Communicators& communicators,
      const Core::LinAlg::MultiVector<double>& vec, const char* name, double tol /*= 1.0e-14*/
  )
  {
    MPI_Comm lcomm = communicators.local_comm();
    MPI_Comm gcomm = communicators.global_comm();

    int result = -1;
    MPI_Comm_compare(gcomm, lcomm, &result);
    if (result == 0)
    {
      Core::IO::cout << "WARNING:: Vectors " << name
                     << " cannot be compared because second 4C run is missing" << Core::IO::endl;
      return false;
    }

    // gather data of vector to compare on gcomm proc 0 and last gcomm proc
    std::shared_ptr<Core::LinAlg::Map> proc0map;
    if (Core::Communication::my_mpi_rank(lcomm) == Core::Communication::my_mpi_rank(gcomm))
      proc0map = Core::LinAlg::allreduce_overlapping_e_map(vec.get_map(), 0);
    else
      proc0map = Core::LinAlg::allreduce_overlapping_e_map(
          vec.get_map(), Core::Communication::num_mpi_ranks(lcomm) - 1);

    // export full vectors to the two desired processors
    Core::LinAlg::MultiVector<double> fullvec(*proc0map, vec.num_vectors(), true);
    Core::LinAlg::export_to(vec, fullvec);

    const int myglobalrank = Core::Communication::my_mpi_rank(gcomm);
    double maxdiff = 0.0;
    // last proc in gcomm sends its data to proc 0 which does the comparison
    if (myglobalrank == 0)
    {
      // compare names
      int lengthRecv = 0;
      std::vector<char> receivename;
      MPI_Status status;
      // first: receive length of name
      int tag = 1336;
      MPI_Recv(&lengthRecv, 1, MPI_INT, Core::Communication::num_mpi_ranks(gcomm) - 1, tag, gcomm,
          &status);
      if (lengthRecv == 0) FOUR_C_THROW("Length of name received from second run is zero.");

      // second: receive name
      tag = 2672;
      receivename.resize(lengthRecv);
      MPI_Recv(receivename.data(), lengthRecv, MPI_CHAR,
          Core::Communication::num_mpi_ranks(gcomm) - 1, tag, gcomm, &status);

      // do comparison of names
      if (std::strcmp(name, receivename.data()))
        FOUR_C_THROW(
            "comparison of different vectors: communicators 0 ({}) and communicators 1 ({})", name,
            receivename.data());

      // compare data
      lengthRecv = 0;
      std::vector<double> receivebuf;
      // first: receive length of data
      tag = 1337;
      MPI_Recv(&lengthRecv, 1, MPI_INT, Core::Communication::num_mpi_ranks(gcomm) - 1, tag, gcomm,
          &status);
      // also enable comparison of empty vectors
      if (lengthRecv == 0 && fullvec.local_length() != lengthRecv)
        FOUR_C_THROW("Length of data received from second run is incorrect.");

      // second: receive data
      tag = 2674;
      receivebuf.resize(lengthRecv);
      MPI_Recv(receivebuf.data(), lengthRecv, MPI_DOUBLE,
          Core::Communication::num_mpi_ranks(gcomm) - 1, tag, gcomm, &status);

      // start comparison
      int mylength = fullvec.local_length() * vec.num_vectors();
      if (mylength != lengthRecv)
        FOUR_C_THROW(
            "length of received data ({}) does not match own data ({})", lengthRecv, mylength);

      for (int i = 0; i < mylength; ++i)
      {
        double difference = std::abs(fullvec.get_values()[i] - receivebuf[i]);
        if (difference > tol)
        {
          std::stringstream diff;
          diff << std::scientific << std::setprecision(16) << maxdiff;
          std::cout << "vectors " << name << " do not match, difference in row "
                    << fullvec.get_map().gid(i) << " between entries is: " << diff.str().c_str()
                    << std::endl;
        }
        maxdiff = std::max(maxdiff, difference);
      }
      if (maxdiff <= tol)
      {
        Core::IO::cout << "compared vectors " << name << " of length: " << mylength
                       << " which are identical." << Core::IO::endl;
        result = 1;
      }
    }
    else if (myglobalrank == Core::Communication::num_mpi_ranks(gcomm) - 1)
    {
      // compare names
      // include terminating \0 of char array
      int lengthSend = std::strlen(name) + 1;
      // first: send length of name
      int tag = 1336;
      MPI_Send(&lengthSend, 1, MPI_INT, 0, tag, gcomm);

      // second: send name
      tag = 2672;
      MPI_Send(const_cast<char*>(name), lengthSend, MPI_CHAR, 0, tag, gcomm);

      // compare data
      lengthSend = fullvec.local_length() * vec.num_vectors();
      // first: send length of data
      tag = 1337;
      MPI_Send(&lengthSend, 1, MPI_INT, 0, tag, gcomm);

      // second: send data
      tag = 2674;
      MPI_Send(fullvec.get_values(), lengthSend, MPI_DOUBLE, 0, tag, gcomm);
    }

    // force all procs to stay here until proc 0 has checked the vectors
    Core::Communication::broadcast(&maxdiff, 1, 0, gcomm);
    if (maxdiff > tol)
    {
      std::stringstream diff;
      diff << std::scientific << std::setprecision(16) << maxdiff;
      FOUR_C_THROW(
          "vectors {} do not match, maximum difference between entries is: {}", name, diff.str());
    }

    return true;
  }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  bool are_distributed_sparse_matrices_identical(const Communicators& communicators,
      const Core::LinAlg::SparseMatrix& matrix, const char* name, double tol /*= 1.0e-14*/
  )
  {
    MPI_Comm lcomm = communicators.local_comm();
    MPI_Comm gcomm = communicators.global_comm();
    const int myglobalrank = Core::Communication::my_mpi_rank(gcomm);

    int result = -1;
    MPI_Comm_compare(gcomm, lcomm, &result);
    if (result == 0)
    {
      Core::IO::cout << "WARNING:: Matrices " << name
                     << " cannot be compared because second 4C run is missing" << Core::IO::endl;
      return false;
    }

    const Core::LinAlg::Map& rowmap = Core::LinAlg::Map(matrix.row_map());
    const Core::LinAlg::Map& domainmap = Core::LinAlg::Map(matrix.domain_map());

    // gather data of vector to compare on gcomm proc 0 and last gcomm proc
    std::shared_ptr<Core::LinAlg::Map> serialrowmap;
    if (Core::Communication::my_mpi_rank(lcomm) == Core::Communication::my_mpi_rank(gcomm))
      serialrowmap = Core::LinAlg::allreduce_overlapping_e_map(rowmap, 0);
    else
      serialrowmap = Core::LinAlg::allreduce_overlapping_e_map(
          rowmap, Core::Communication::num_mpi_ranks(lcomm) - 1);

    std::shared_ptr<Core::LinAlg::Map> serialdomainmap;
    if (Core::Communication::my_mpi_rank(lcomm) == Core::Communication::my_mpi_rank(gcomm))
      serialdomainmap = Core::LinAlg::allreduce_overlapping_e_map(domainmap, 0);
    else
      serialdomainmap = Core::LinAlg::allreduce_overlapping_e_map(
          domainmap, Core::Communication::num_mpi_ranks(lcomm) - 1);

    // export full matrices to the two desired processors
    Core::LinAlg::Import serialimporter(*serialrowmap, rowmap);
    Core::LinAlg::SparseMatrix serialCrsMatrix(*serialrowmap, 0);
    serialCrsMatrix.import(matrix, serialimporter, Core::LinAlg::CombineMode::insert);
    serialCrsMatrix.complete(*serialdomainmap, *serialrowmap);

    // fill data of matrices to container which can be easily communicated via MPI
    std::vector<int> data_indices;
    data_indices.reserve(serialCrsMatrix.num_my_nonzeros() * 2);
    std::vector<double> data_values;
    data_values.reserve(serialCrsMatrix.num_my_nonzeros());
    if (myglobalrank == 0 || myglobalrank == Core::Communication::num_mpi_ranks(gcomm) - 1)
    {
      for (int i = 0; i < serialrowmap->num_my_elements(); ++i)
      {
        int rowgid = serialrowmap->gid(i);
        int NumEntries;
        double* Values;
        int* Indices;
        serialCrsMatrix.extract_my_row_view(i, NumEntries, Values, Indices);

        for (int j = 0; j < NumEntries; ++j)
        {
          // store row and col gid in order to compare them on proc 0 and for detailed error output
          // information
          data_indices.push_back(rowgid);
          data_indices.push_back(Indices[j]);
          data_values.push_back(Values[j]);
        }
      }
    }

    // last proc in gcomm sends its data to proc 0 which does the comparison
    double maxdiff = 0.0;
    if (myglobalrank == 0)
    {
      // compare names
      int lengthRecv = 0;
      std::vector<char> receivename;
      MPI_Status status;
      // first: receive length of name
      int tag = 1336;
      MPI_Recv(&lengthRecv, 1, MPI_INT, Core::Communication::num_mpi_ranks(gcomm) - 1, tag, gcomm,
          &status);
      if (lengthRecv == 0) FOUR_C_THROW("Length of name received from second run is zero.");

      // second: receive name
      tag = 2672;
      receivename.resize(lengthRecv);
      MPI_Recv(receivename.data(), lengthRecv, MPI_CHAR,
          Core::Communication::num_mpi_ranks(gcomm) - 1, tag, gcomm, &status);

      // do comparison of names
      if (std::strcmp(name, receivename.data()))
        FOUR_C_THROW(
            "comparison of different vectors: communicators 0 ({}) and communicators 1 ({})", name,
            receivename.data());

      // compare data: indices
      lengthRecv = 0;
      std::vector<int> receivebuf_indices;
      // first: receive length of data
      tag = 1337;
      MPI_Recv(&lengthRecv, 1, MPI_INT, Core::Communication::num_mpi_ranks(gcomm) - 1, tag, gcomm,
          &status);
      // also enable comparison of empty matrices
      if (lengthRecv == 0 && (int)data_indices.size() != lengthRecv)
        FOUR_C_THROW("Length of data received from second run is incorrect.");

      // second: receive data
      tag = 2674;
      receivebuf_indices.resize(lengthRecv);
      MPI_Recv(receivebuf_indices.data(), lengthRecv, MPI_INT,
          Core::Communication::num_mpi_ranks(gcomm) - 1, tag, gcomm, &status);

      // start comparison
      int mylength = data_indices.size();
      if (mylength != lengthRecv)
        FOUR_C_THROW(
            "length of received data ({}) does not match own data ({})", lengthRecv, mylength);

      for (int i = 0; i < mylength; ++i)
      {
        if (data_indices[i] != receivebuf_indices[i])
        {
          bool iscolindex = data_indices[i] % 2;
          FOUR_C_THROW(
              "{} index of matrix {} does not match: communicators 0 ({}) and communicators 1 ({})",
              iscolindex == 0 ? "row" : "col", name, data_indices[i], receivebuf_indices[i]);
        }
      }
      Core::IO::cout << "indices of compared matrices " << name << " of length: " << mylength
                     << " are identical." << Core::IO::endl;

      // compare data: values
      lengthRecv = 0;
      std::vector<double> receivebuf_values;
      // first: receive length of data
      tag = 1338;
      MPI_Recv(&lengthRecv, 1, MPI_INT, Core::Communication::num_mpi_ranks(gcomm) - 1, tag, gcomm,
          &status);
      // also enable comparison of empty matrices
      if (lengthRecv == 0 && (int)data_values.size() != lengthRecv)
        FOUR_C_THROW("Length of data received from second run is incorrect.");

      // second: receive data
      tag = 2676;
      receivebuf_values.resize(lengthRecv);
      MPI_Recv(receivebuf_values.data(), lengthRecv, MPI_DOUBLE,
          Core::Communication::num_mpi_ranks(gcomm) - 1, tag, gcomm, &status);

      // start comparison
      mylength = data_values.size();
      if (mylength != lengthRecv)
        FOUR_C_THROW(
            "length of received data ({}) does not match own data ({})", lengthRecv, mylength);

      for (int i = 0; i < mylength; ++i)
      {
        double difference = std::abs(data_values[i] - receivebuf_values[i]);
        if (difference > tol)
        {
          std::stringstream diff;
          diff << std::scientific << std::setprecision(16) << maxdiff;
          std::cout << "matrices " << name << " do not match, difference in row "
                    << data_indices[2 * i] << " , col: " << data_indices[2 * i + 1]
                    << " between entries is: " << diff.str().c_str() << std::endl;
        }
        maxdiff = std::max(maxdiff, difference);
      }
      if (maxdiff <= tol)
      {
        Core::IO::cout << "values of compared matrices " << name << " of length: " << mylength
                       << " are identical." << Core::IO::endl;
      }
    }
    else if (myglobalrank == Core::Communication::num_mpi_ranks(gcomm) - 1)
    {
      // compare names
      // include terminating \0 of char array
      int lengthSend = std::strlen(name) + 1;
      // first: send length of name
      int tag = 1336;
      MPI_Send(&lengthSend, 1, MPI_INT, 0, tag, gcomm);

      // second: send name
      tag = 2672;
      MPI_Send(const_cast<char*>(name), lengthSend, MPI_CHAR, 0, tag, gcomm);

      // compare data: indices
      lengthSend = data_indices.size();
      // first: send length of data
      tag = 1337;
      MPI_Send(&lengthSend, 1, MPI_INT, 0, tag, gcomm);

      // second: send data
      tag = 2674;
      MPI_Send(data_indices.data(), lengthSend, MPI_INT, 0, tag, gcomm);

      // compare data: values
      lengthSend = data_values.size();
      // first: send length of data
      tag = 1338;
      MPI_Send(&lengthSend, 1, MPI_INT, 0, tag, gcomm);

      // second: send data
      tag = 2676;
      MPI_Send(data_values.data(), lengthSend, MPI_DOUBLE, 0, tag, gcomm);
    }

    // force all procs to stay here until proc 0 has checked the matrices
    Core::Communication::broadcast(&maxdiff, 1, 0, gcomm);
    if (maxdiff > tol)
    {
      std::stringstream diff;
      diff << std::scientific << std::setprecision(16) << maxdiff;
      FOUR_C_THROW("matrices {} do not match, maximum difference between entries is: {} in row",
          name, diff.str());
    }

    return true;
  }
}  // namespace Core::Communication

FOUR_C_NAMESPACE_CLOSE

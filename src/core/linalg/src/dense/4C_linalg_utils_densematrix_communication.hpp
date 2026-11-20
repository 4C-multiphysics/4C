// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_UTILS_DENSEMATRIX_COMMUNICATION_HPP
#define FOUR_C_LINALG_UTILS_DENSEMATRIX_COMMUNICATION_HPP

#include "4C_config.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_map.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /*!
   \brief Gather information of type vector<T> on a subset of processors

   This template gathers information provided in sdata on a subset of
   processors tprocs, where the length of the array tprocs is ntargetprocs.
   The redistributed data is returned in rdata which has appropriate size
   on output (size of rdata on input is arbitrary). ntargetprocs can be
   one to reduce data to one proc, it also can be equal to the total number
   of processors to make sdata redundant on all procs.

   \note Functionality of this method is equal to that of Epetra_Comm::GatherAll
   except for that the Epetra version demands the data to be of constant
   size over all procs which this method does not require!

   \param sdata (in) : Information to be gathered on tprocs.
   Length of sdata can be different on every proc.
   \param rdata (out): Information from sdata gathered on a subset of procs.
   size of rdata on input is arbitrary, it is exact on output.
   \param ntargetprocs (in): length of tprocs
   \param tprocs (in): vector of procs ids the information in sdata shall be
   gathered on.
   \param comm (in):   communicator to be used.


   */
  template <typename T>
  void gather(std::vector<T>& sdata, std::vector<T>& rdata, const int ntargetprocs,
      const int* tprocs, MPI_Comm comm)
  {
    const int myrank = Core::Communication::my_mpi_rank(comm);
    const int numproc = Core::Communication::num_mpi_ranks(comm);
    if (numproc == 1)
    {
      rdata = sdata;
      return;  // nothing to do in serial
    }
    // build a map of data
    std::map<int, std::vector<T>> datamap;
    datamap[myrank] = sdata;
    // build a source map
    Core::LinAlg::Map source(numproc, std::span<const int>(&myrank, 1), 0, comm);
    // build a target map which is redundant on all target procs and zero everywhere else
    bool iamtarget = false;
    for (int i = 0; i < ntargetprocs; ++i)
      if (tprocs[i] == myrank)
      {
        iamtarget = true;
        break;
      }
    std::vector<int> targetvec;
    if (iamtarget)
    {
      targetvec.resize(numproc);
      for (int i = 0; i < numproc; ++i) targetvec[i] = i;
    }
    const int tnummyelements = (int)targetvec.size();
    Core::LinAlg::Map target(-1, std::span<const int>(targetvec.data(), tnummyelements), 0, comm);
    // build an exporter and export data
    Core::Communication::Exporter exporter(source, target, comm);
    exporter.do_export(datamap);
    // put data from map in rdata
    rdata.clear();
    int count = 0;
    typename std::map<int, std::vector<T>>::const_iterator curr;
    for (curr = datamap.begin(); curr != datamap.end(); ++curr)
    {
      const std::vector<T>& current = curr->second;
      const int size = (int)current.size();
      rdata.resize((int)rdata.size() + size);
      for (int i = 0; i < size; ++i) rdata[count + i] = current[i];
      count += size;
    }
    return;
  }

  /*!
   \brief Gather information of type set<T> on a subset of processors

   This template gathers information provided in sdata on a subset of
   processors tprocs, where the length of the array tprocs is ntargetprocs.
   The redistributed data is returned in rdata which has appropriate size
   on output (size of rdata on input is arbitrary). ntargetprocs can be
   one to reduce data to one proc, it also can be equal to the total number
   of processors to make sdata redundant on all procs.

   \note Functionality of this method is equal to that of Epetra_Comm::GatherAll
   except for that the Epetra version demands the data to be of constant
   size over all procs which this method does not require!

   \param sdata (in) : Information to be gathered on tprocs.
   Length of sdata can be different on every proc.
   \param rdata (out): Information from sdata gathered on a subset of procs.
   size of rdata on input is arbitrary, it is exact on output.
   \param ntargetprocs (in): length of tprocs
   \param tprocs (in): vector of procs ids the information in sdata shall be
   gathered on.
   \param comm (in):   communicator to be used.


   */
  template <typename T>
  void gather(std::set<T>& sdata, std::set<T>& rdata, const int ntargetprocs, const int* tprocs,
      MPI_Comm comm)
  {
    const int myrank = Core::Communication::my_mpi_rank(comm);
    const int numproc = Core::Communication::num_mpi_ranks(comm);
    if (numproc == 1)
    {
      rdata = sdata;
      return;  // nothing to do in serial
    }
    // build a map of data
    std::map<int, std::set<T>> datamap;
    datamap[myrank] = sdata;
    // build a source map
    Core::LinAlg::Map source(numproc, std::span<const int>(&myrank, 1), 0, comm);
    // build a target map which is redundant on all target procs and zero everywhere else
    bool iamtarget = false;
    for (int i = 0; i < ntargetprocs; ++i)
      if (tprocs[i] == myrank)
      {
        iamtarget = true;
        break;
      }
    std::vector<int> targetvec;
    if (iamtarget)
    {
      targetvec.resize(numproc);
      for (int i = 0; i < numproc; ++i) targetvec[i] = i;
    }
    const int tnummyelements = (int)targetvec.size();
    Core::LinAlg::Map target(-1, std::span<const int>(targetvec.data(), tnummyelements), 0, comm);
    // build an exporter and export data
    Core::Communication::Exporter exporter(source, target, comm);
    exporter.do_export(datamap);
    // put data from map in rdata
    rdata.clear();
    typename std::map<int, std::set<T>>::const_iterator curr;
    for (curr = datamap.begin(); curr != datamap.end(); ++curr)
    {
      const std::set<T>& current = curr->second;
      typename std::set<T>::const_iterator setiter;
      for (setiter = current.begin(); setiter != current.end(); ++setiter)
      {
        rdata.insert(*setiter);
      }
    }
    return;
  }

  /*!
   \brief Gather information of type std::map<int, std::set<T> > on a subset of processors

   This template gathers information provided in sdata on a subset of
   processors tprocs, where the length of the array tprocs is ntargetprocs.
   The redistributed data is returned in rdata which has appropriate size
   on output (size of rdata on input is arbitrary). ntargetprocs can be
   one to reduce data to one proc, it also can be equal to the total number
   of processors to make sdata redundant on all procs.

   \note Functionality of this method is equal to that of Epetra_Comm::GatherAll
   except for that the Epetra version demands the data to be of constant
   size over all procs which this method does not require!

   \param sdata (in) : Information to be gathered on tprocs.
   Length of sdata can be different on every proc.
   \param rdata (out): Information from sdata gathered on a subset of procs.
   size of rdata on input is arbitrary, it is exact on output.
   \param ntargetprocs (in): length of tprocs
   \param tprocs (in): vector of procs ids the information in sdata shall be
   gathered on.
   \param comm (in):   communicator to be used.


   */
  template <typename T>
  void gather(std::map<int, std::set<T>>& sdata, std::map<int, std::set<T>>& rdata,
      const int ntargetprocs, const int* tprocs, MPI_Comm comm)
  {
    const int myrank = Core::Communication::my_mpi_rank(comm);
    const int numproc = Core::Communication::num_mpi_ranks(comm);
    if (numproc == 1)
    {
      rdata = sdata;
      return;  // nothing to do in serial
    }
    // build a map of data
    std::map<int, std::map<int, std::set<T>>> datamap;
    datamap[myrank] = sdata;
    // build a source map
    Core::LinAlg::Map source(numproc, std::span<const int>(&myrank, 1), 0, comm);
    // build a target map which is redundant on all target procs and zero everywhere else
    bool iamtarget = false;
    for (int i = 0; i < ntargetprocs; ++i)
      if (tprocs[i] == myrank)
      {
        iamtarget = true;
        break;
      }
    std::vector<int> targetvec;
    if (iamtarget)
    {
      targetvec.resize(numproc);
      for (int i = 0; i < numproc; ++i) targetvec[i] = i;
    }
    const int tnummyelements = (int)targetvec.size();
    Core::LinAlg::Map target(-1, std::span<const int>(targetvec.data(), tnummyelements), 0, comm);
    // build an exporter and export data
    Core::Communication::Exporter exporter(source, target, comm);
    exporter.do_export(datamap);
    // put data from map in rdata
    rdata.clear();
    typename std::map<int, std::map<int, std::set<T>>>::const_iterator curr;
    for (curr = datamap.begin(); curr != datamap.end(); ++curr)
    {
      const std::map<int, std::set<T>>& innercurr = curr->second;
      typename std::map<int, std::set<T>>::const_iterator inneriter;
      for (inneriter = innercurr.begin(); inneriter != innercurr.end(); ++inneriter)
      {
        rdata[inneriter->first].insert(inneriter->second.begin(), inneriter->second.end());
      }
    }
    return;
  }

  /*!
   \brief Gather information of type std::map<int, std::vector<T> > on a subset of processors

   This template gathers information provided in sdata on a subset of
   processors tprocs, where the length of the array tprocs is ntargetprocs.
   The redistributed data is returned in rdata which has appropriate size
   on output (size of rdata on input is arbitrary). ntargetprocs can be
   one to reduce data to one proc, it also can be equal to the total number
   of processors to make sdata redundant on all procs.

   \note Functionality of this method is equal to that of Epetra_Comm::GatherAll
   except for that the Epetra version demands the data to be of constant
   size over all procs which this method does not require!

   \param sdata (in) : Information to be gathered on tprocs.
   Length of sdata can be different on every proc.
   \param rdata (out): Information from sdata gathered on a subset of procs.
   size of rdata on input is arbitrary, it is exact on output.
   \param ntargetprocs (in): length of tprocs
   \param tprocs (in): vector of procs ids the information in sdata shall be
   gathered on.
   \param comm (in):   communicator to be used.


   */
  template <typename T>
  void gather(std::map<int, std::vector<T>>& sdata, std::map<int, std::vector<T>>& rdata,
      const int ntargetprocs, const int* tprocs, MPI_Comm comm)
  {
    const int myrank = Core::Communication::my_mpi_rank(comm);
    const int numproc = Core::Communication::num_mpi_ranks(comm);
    if (numproc == 1)
    {
      rdata = sdata;
      return;  // nothing to do in serial
    }
    // build a map of data
    std::map<int, std::map<int, std::vector<T>>> datamap;
    datamap[myrank] = sdata;
    // build a source map
    Core::LinAlg::Map source(numproc, std::span<const int>(&myrank, 1), 0, comm);
    // build a target map which is redundant on all target procs and zero everywhere else
    bool iamtarget = false;
    for (int i = 0; i < ntargetprocs; ++i)
      if (tprocs[i] == myrank)
      {
        iamtarget = true;
        break;
      }
    std::vector<int> targetvec;
    if (iamtarget)
    {
      targetvec.resize(numproc);
      for (int i = 0; i < numproc; ++i) targetvec[i] = i;
    }
    const int tnummyelements = (int)targetvec.size();
    Core::LinAlg::Map target(-1, std::span<const int>(targetvec.data(), tnummyelements), 0, comm);
    // build an exporter and export data
    Core::Communication::Exporter exporter(source, target, comm);
    exporter.do_export(datamap);
    // put data from map in rdata
    rdata.clear();
    typename std::map<int, std::map<int, std::vector<T>>>::const_iterator curr1;
    typename std::map<int, std::vector<T>>::const_iterator curr2;
    typename std::vector<T>::const_iterator curr3;
    for (curr1 = datamap.begin(); curr1 != datamap.end(); ++curr1)
    {
      const std::map<int, std::vector<T>>& data = curr1->second;
      for (curr2 = data.begin(); curr2 != data.end(); ++curr2)
      {
        const std::vector<T>& vectordata = curr2->second;
        for (curr3 = vectordata.begin(); curr3 != vectordata.end(); ++curr3)
        {
          rdata[curr2->first].push_back(*curr3);
        }
      }
    }
    return;
  }

  /*!
   \brief Gather information of type map<T,U> on a subset of processors

   This template gathers information provided in sdata on a subset of
   processors tprocs, where the length of the array tprocs is ntargetprocs.
   The redistributed data is returned in rdata which has appropriate size
   on output (size of rdata on input is arbitrary). ntargetprocs can be
   one to reduce data to one proc, it also can be equal to the total number
   of processors to make sdata redundant on all procs.

   \param sdata (in) : Information to be gathered on tprocs.
   Length of sdata can be different on every proc.
   \param rdata (out): Information from sdata gathered on a subset of procs.
   size of rdata on input is arbitrary, it is exact on output.
   \param ntargetprocs (in): length of tprocs
   \param tprocs (in): vector of procs ids the information in sdata shall be
   gathered on.
   \param comm (in):   communicator to be used.

   */

  template <typename T, typename U>
  void gather(std::map<T, U>& sdata, std::map<T, U>& rdata, const int ntargetprocs,
      const int* tprocs, MPI_Comm comm)
  {
    const int myrank = Core::Communication::my_mpi_rank(comm);
    const int numproc = Core::Communication::num_mpi_ranks(comm);
    if (numproc == 1)
    {
      rdata = sdata;
      return;  // nothing to do in serial
    }

    // build a map of data
    std::map<int, std::map<T, U>> datamap;
    datamap[myrank] = sdata;

    // build a source map
    Core::LinAlg::Map source(numproc, std::span<const int>(&myrank, 1), 0, comm);
    // build a target map which is redundant on all target procs and zero everywhere else
    bool iamtarget = false;
    for (int i = 0; i < ntargetprocs; ++i)
      if (tprocs[i] == myrank)
      {
        iamtarget = true;
        break;
      }
    std::vector<int> targetvec;
    if (iamtarget)
    {
      targetvec.resize(numproc);
      for (int i = 0; i < numproc; ++i) targetvec[i] = i;
    }
    const int tnummyelements = (int)targetvec.size();
    Core::LinAlg::Map target(-1, std::span<const int>(targetvec.data(), tnummyelements), 0, comm);
    // build an exporter and export data
    Core::Communication::Exporter exporter(source, target, comm);
    exporter.do_export(datamap);
    // put data from map in rdata
    rdata.clear();
    typename std::map<int, std::map<T, U>>::const_iterator curr;
    for (curr = datamap.begin(); curr != datamap.end(); ++curr)
    {
      const std::map<T, U>& current = curr->second;
      typename std::map<T, U>::const_iterator mapiter;
      for (mapiter = current.begin(); mapiter != current.end(); ++mapiter)
      {
        rdata.insert(std::make_pair(mapiter->first, mapiter->second));
      }
    }
    return;
  }

  /*!
   \brief Gather information of type set<T> from all processors

   This template gathers information provided in data on all processors.
   The redistributed data is returned in data which has appropriate size
   on output.

   \note Functionality of this method is equal to that of Epetra_Comm::GatherAll
   except for that the Epetra version demands the data to be of constant
   size over all procs which this method does not require!

   \param data (in/out) : Information to be gathered.
   Length of data can be different on every proc.
   \param comm (in):   communicator to be used.

   */
  template <typename T>
  void gather_all(std::set<T>& data, MPI_Comm comm)
  {
    // ntargetprocs is equal to the total number of processors to make data redundant on all procs
    const int numprocs = Core::Communication::num_mpi_ranks(comm);
    std::vector<int> allproc(numprocs);
    for (int i = 0; i < numprocs; ++i) allproc[i] = i;

    gather<T>(data, data, numprocs, allproc.data(), comm);
    return;
  }

  // nagler 07/2012
  /*!
   \brief Gather information of type map<T,U> from all processors

   This template gathers information provided in data on all processors.
   The redistributed data is returned in data which has appropriate size
   on output.

   \param data (in/out) : Information to be gathered.
   Length of data can be different on every proc.
   \param comm (in):   communicator to be used.

   */

  template <typename T, typename U>
  void gather_all(std::map<T, U>& data, MPI_Comm comm)
  {
    const int numprocs = Core::Communication::num_mpi_ranks(comm);
    std::vector<int> allproc(numprocs);
    for (int i = 0; i < numprocs; ++i) allproc[i] = i;

    gather<T, U>(data, data, numprocs, allproc.data(), comm);

    return;
  }

  //                                                              sudhakar 02/2014
  /*!
   \brief Gather information of type map<int,vector<T> > from all processors

   This template gathers information provided in data on all processors.
   The redistributed data is returned in data which has appropriate size
   on output.

   \param data (in/out) : Information to be gathered.
   Length of data can be different on every proc.
   \param comm (in):   communicator to be used.

   */

  template <typename T>
  void gather_all(std::map<int, std::vector<T>>& data, MPI_Comm comm)
  {
    const int numprocs = Core::Communication::num_mpi_ranks(comm);
    std::vector<int> allproc(numprocs);
    for (int i = 0; i < numprocs; ++i) allproc[i] = i;

    gather<T>(data, data, numprocs, allproc.data(), comm);

    return;
  }

  /*!
   \brief Gather information of type vector<T> from all processors

   This template gathers information provided in data on all processors.
   The redistributed data is returned in data which has appropriate size
   on output.

   \note Functionality of this method is equal to that of Epetra_Comm::GatherAll
   except for that the Epetra version demands the data to be of constant
   size over all procs which this method does not require!

   \param data (in/out) : Information to be gathered.
   Length of data can be different on every proc.
   \param comm (in):   communicator to be used.

   */
  template <typename T>
  void gather_all(std::vector<T>& data, MPI_Comm comm)
  {
    // ntargetprocs is equal to the total number of processors to make data redundant on all procs
    const int numprocs = Core::Communication::num_mpi_ranks(comm);
    std::vector<int> allproc(numprocs);
    for (int i = 0; i < numprocs; ++i) allproc[i] = i;

    gather<T>(data, data, numprocs, allproc.data(), comm);
    return;
  }

  /*!
   \brief Create an allreduced vector of gids from the given Core::LinAlg::Map

   We have nodes and elements arbitrary global ids. On rare occasions, however,
   we need to allreduce a particular map to one or more processors.
   This is a building block for such occasions. We allreduce the gids
   of the given Core::LinAlg::Map into a vector ordered by processor number.

   \note You are not supposed to use redundant vectors in normal
   situations. If you happen to need this method you are probably
   about to do something illegal.

   \param rredundant (o) redundant vector of global ids
   \param emap (i) unique distributed Core::LinAlg::Map

   */
  void allreduce_e_map(std::vector<int>& rredundant, const Core::LinAlg::Map& emap);

  /// Create an allreduced gid to index map from the given Core::LinAlg::Map
  /*!
   We have nodes and elements with unique but otherwise arbitrary
   global ids. But unfortunately we need an allreduced vector of dof
   numbers during the dof assignment phase. In order to use such a
   vector we need to map from global ids to vector indexes. Here we
   provide that map.

   \note You are not supposed to use redundant vectors in normal
   situations. If you happen to need this method you are probably
   about to do something illegal.

   \param idxmap (o) map from global ids to (redundant) vector indexes
   \param emap (i) unique distributed Core::LinAlg::Map

   */
  void allreduce_e_map(std::map<int, int>& idxmap, const Core::LinAlg::Map& emap);

  /*!
   \brief Create an allreduced gid to index map from the given Core::LinAlg::Map
          on a distinct processor, all other procs create empty maps instead.

   This method is currently used within the parallel post_ensight
   filter in order to import all values stored in a distributed Core::LinAlg::Vector<double>
   to processor 0 for writing them into file.

   \note see also documentation for the usual allreduce_e_map methods

   \param emap (i) any distributed Core::LinAlg::Map
   \param pid (i)  processor id where you want to have the allreduced map
   exclusively

   */
  std::shared_ptr<Core::LinAlg::Map> allreduce_e_map(const Core::LinAlg::Map& emap, const int pid);

  /*!
   \brief Create an allreduced Core::LinAlg::Map from the given Core::LinAlg::Map
          and give it to all processors.

   This method is currently used within the constraint management, since
   current values of constraint values and Lagrange multipliers are distributed
   uniquely for computation. At some places we need the full information of these
   values on every processor, so this method has to be used.

   \note You are not supposed to use redundant vectors in normal
   situations. If you happen to need this method you are probably
   about to do something illegal.

   \param emap (i) any distributed Core::LinAlg::Map

   */
  std::shared_ptr<Core::LinAlg::Map> allreduce_e_map(const Core::LinAlg::Map& emap);

  /*!
   \brief Create an allreduced Core::LinAlg::Map from the given Core::LinAlg::Map
          and give it to all processors.

   Here, we have a overlapping source map and still want to have a fully
   redundant map on all processors without duplicated entries.

   */
  std::shared_ptr<Core::LinAlg::Map> allreduce_overlapping_e_map(const Core::LinAlg::Map& emap);

  /*!
   \brief Create an allreduced Core::LinAlg::Map from the given Core::LinAlg::Map
          on a distinct processor, all other procs create empty maps instead.

   \param emap (i) any distributed overlapping Core::LinAlg::Map
   \param pid (i)  processor id where you want to have the allreduced and sorted map exclusively

   */
  std::shared_ptr<Core::LinAlg::Map> allreduce_overlapping_e_map(
      const Core::LinAlg::Map& emap, const int pid);

  /*!
   \brief Find position of my map elements in a consecutive vector

   The idea is to put the entries of a given map into a redundant
   vector, ordered by processor number. The map is assumed to be
   nonoverlapping. Here we figure out the index of our first entry in
   that vector.

   \note You are not supposed to use redundant vectors in normal
   situations. If you happen to need this method you are probably
   about to do something illegal.

   \param nummyelements (i) number of elements on this proc
   \param comm (i) communicator

   \return vector position of first entry on each processor

   */
  int find_my_pos(int nummyelements, MPI_Comm comm);

  /// create an allreduced sorted copy of the source vectors
  void allreduce_vector(const std::vector<int>& src, std::vector<int>& dest, MPI_Comm comm);

  /*!
   \brief Communication between all pairs of processes, with distinct data for each.

   Sends a different vector<int> to each processes. The size of each vector may
   be different, zero-length vectors are allowed.
   Communication is implemented with the MPI function MPI_Alltoallv.

   \param comm (i) communicator
   \param send (i) vector of length Core::Communication::num_mpi_ranks(comm), j-th element to be
   send to j-th processor. \param recv (o) vector of length
   Core::Communication::num_mpi_ranks(comm), j-th element received from j-th processor.

   */
  void all_to_all_communication(MPI_Comm comm, const std::vector<std::vector<int>>& send,
      std::vector<std::vector<int>>& recv);

  /*!
   \brief Communication between all pairs of processes, with distinct data for each.

   Sends a different vector<int> to each processes. The size of each vector may
   be different, zero-length vectors are allowed.
   Communication is implemented with the MPI function MPI_Alltoallv.

   \param[in] comm communicator
   \param[in] send vector of length Core::Communication::num_mpi_ranks(comm), j-th element to be
   send to j-th processor. \param[out] recv vector of received elements without knowledge of the
   sending processor
   */
  void all_to_all_communication(
      MPI_Comm comm, const std::vector<std::vector<int>>& send, std::vector<int>& recv);


}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif

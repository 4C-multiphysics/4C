// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_hdf.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 * The Constructor of the HDFReader (num_proc defaults to 1)
 *----------------------------------------------------------------------*/
Core::IO::HDFReader::HDFReader(std::string dir)
    : filenames_(0), files_(0), input_dir_(dir), num_output_proc_(0)
{
  // inhibit delayed closure, throws error if file contents still in use
  h5_plist_ = H5Pcreate(H5P_FILE_ACCESS);
  herr_t status = H5Pset_fclose_degree(h5_plist_, H5F_CLOSE_WEAK);
  if (status < 0) FOUR_C_THROW("Failed to set file access list");
}

/*----------------------------------------------------------------------*
 * The Destructor
 *----------------------------------------------------------------------*/
Core::IO::HDFReader::~HDFReader()
{
  close();
  herr_t status = H5Pclose(h5_plist_);
  if (status < 0) FOUR_C_THROW("Failed to close file access list");
}

/*----------------------------------------------------------------------*
 * With num_output_proc_ == 1 this function opens the result data file
 * with name basename. When num_output_proc_ > 1 it opens the result
 * files of all processors, by appending .p<proc_num> to the basename.
 *----------------------------------------------------------------------*/
void Core::IO::HDFReader::open(
    std::string basename, int num_output_procs, int new_proc_num, int my_id)
{
  int start;
  int end;
  num_output_proc_ = num_output_procs;
  calculate_range(new_proc_num, my_id, start, end);
  close();
  for (int i = 0; i < num_output_proc_; ++i)
  {
    std::ostringstream buf;
    buf << input_dir_ << basename;
    if (num_output_proc_ > 1)
    {
      buf << ".p" << i;
    }
    if (i >= start and i < end)
    {
      filenames_.push_back(buf.str());
      files_.push_back(H5Fopen(buf.str().c_str(), H5F_ACC_RDONLY, h5_plist_));
      if (files_[i] < 0) FOUR_C_THROW("Failed to open HDF-file {}", filenames_[i]);
    }
    else
    {
      filenames_.push_back("");
      files_.push_back(-1);
    }
  }
}
/*----------------------------------------------------------------------*
 * reads the packed element data from the mesh files
 * Note: this function should only be called when the HDFReader opened
 * the mesh files
 *----------------------------------------------------------------------*/
std::shared_ptr<std::vector<char>> Core::IO::HDFReader::read_element_data(
    int step, int new_proc_num, int my_id) const
{
  if (files_.size() == 0) FOUR_C_THROW("Tried to read data without opening any file");
  std::ostringstream path;
  path << "/step" << step << "/elements";
  int start, end;
  if (new_proc_num == 0 && my_id == 0)
  {
    start = 0;
    end = num_output_proc_;
  }
  else
  {
    calculate_range(new_proc_num, my_id, start, end);
  }
  return read_char_data(path.str(), start, end);
}


/*----------------------------------------------------------------------*
 * reads the packed bc data from the mesh files
 * Note: this function should only be called when the HDFReader opened
 * the mesh files                                          gammi 05/07
 *----------------------------------------------------------------------*/
std::shared_ptr<std::vector<char>> Core::IO::HDFReader::read_condition(
    const int step, const int new_proc_num, const int my_id, const std::string condname) const
{
  if (files_.size() == 0) FOUR_C_THROW("Tried to read data without opening any file");

  std::ostringstream path;
  path << "/step" << step << "/" << condname;

  // only one proc (PROC 0) wrote this conditions to the mesh file
  const int start = 0;
  const int end = 1;

  std::shared_ptr<std::vector<char>> block;
  block = read_char_data(path.str(), start, end);

  return block;
}


/*----------------------------------------------------------------------*
 * reads the packed knotvector data from the mesh files
 * Note: this function should only be called when the HDFReader opened
 * the mesh files                                          gammi 05/08
 *----------------------------------------------------------------------*/
std::shared_ptr<std::vector<char>> Core::IO::HDFReader::read_knotvector(const int step) const
{
  if (files_.size() == 0) FOUR_C_THROW("Tried to read data without opening any file");

  std::ostringstream path;
  path << "/step" << step << "/"
       << "knotvector";

  // only one proc (PROC 0) wrote the knotvector to the mesh file
  const int start = 0;
  const int end = 1;

  std::shared_ptr<std::vector<char>> block;
  block = read_char_data(path.str(), start, end);

  return block;
}


/*----------------------------------------------------------------------*
 * reads the packed node data from the mesh files
 * Note: this function should only be called when the HDFReader opened
 * the mesh files
 *----------------------------------------------------------------------*/
std::shared_ptr<std::vector<char>> Core::IO::HDFReader::read_node_data(
    int step, int new_proc_num, int my_id) const
{
  if (files_.size() == 0) FOUR_C_THROW("Tried to read data without opening any file");
  std::ostringstream path;
  path << "/step" << step << "/nodes";
  int start, end;
  if (new_proc_num == 0 && my_id == 0)
  {
    start = 0;
    end = num_output_proc_;
  }
  else
  {
    calculate_range(new_proc_num, my_id, start, end);
  }
  std::shared_ptr<std::vector<char>> d = read_char_data(path.str(), start, end);
  return d;
}

/*----------------------------------------------------------------------*
 * reads the dataset 'path' in all the files in the range [start,end)
 * and returns all the data in one vector. The data is assumed to by
 * of type char (private)
 *----------------------------------------------------------------------*/
std::shared_ptr<std::vector<char>> Core::IO::HDFReader::read_char_data(
    std::string path, int start, int end) const
{
  if (end == -1) end = num_output_proc_;
  hsize_t offset = 0;
  std::shared_ptr<std::vector<char>> data = std::make_shared<std::vector<char>>();
  for (int i = start; i < end; ++i)
  {
    const char* cpath = path.c_str();
    hid_t dataset = H5Dopen(files_[i], cpath, H5P_DEFAULT);
    if (dataset < 0)
    {
      FOUR_C_THROW("Failed to open dataset {} in HDF-file {}", cpath, filenames_[i]);
    }
    hid_t dataspace = H5Dget_space(dataset);
    if (dataspace < 0)
      FOUR_C_THROW("Failed to get dataspace from dataset {} in HDF-file {}", path, filenames_[i]);
    int rank = H5Sget_simple_extent_ndims(dataspace);
    switch (rank)
    {
      case 0:
        break;
      case 1:
      {
        hsize_t dim, maxdim;
        int res = H5Sget_simple_extent_dims(dataspace, &dim, &maxdim);
        if (res < 0)
          FOUR_C_THROW("Failed to get size from dataspace in HDF-file {}", filenames_[i]);
        data->resize(offset + dim);
        herr_t status =
            H5Dread(dataset, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, &((*data)[offset]));
        offset += dim;
        if (status < 0)
          printf(
              "Failed to read data from dataset %s in HDF-file %s. "
              "This can be tolerated in case you have procs without row elements!",
              path.c_str(), filenames_[i].c_str());
        break;
      }
      default:
        FOUR_C_THROW("HDF5 rank={} unsupported", rank);
        break;
    }

    herr_t status = H5Sclose(dataspace);
    if (status < 0) FOUR_C_THROW("Failed to close node dataspace", path, filenames_[i]);
    status = H5Dclose(dataset);
    if (status < 0) FOUR_C_THROW("Failed to close node dataset", path, filenames_[i]);
  }
  return data;
}

/*----------------------------------------------------------------------*
 * reads the dataset 'path' in all the files in the range [start,end)
 * and returns all the data in one std::vector<int> (private)
 *----------------------------------------------------------------------*/
std::shared_ptr<std::vector<int>> Core::IO::HDFReader::read_int_data(
    std::string path, int start, int end) const
{
  if (end == -1) end = num_output_proc_;
  int offset = 0;
  std::shared_ptr<std::vector<int>> data = std::make_shared<std::vector<int>>();
  for (int i = start; i < end; ++i)
  {
    hid_t dataset = H5Dopen(files_[i], path.c_str(), H5P_DEFAULT);
    if (dataset < 0)
    {
      FOUR_C_THROW("Failed to open dataset {} in HDF-file {}", path, filenames_[i]);
    }
    hid_t dataspace = H5Dget_space(dataset);
    if (dataspace < 0)
      FOUR_C_THROW("Failed to get dataspace from dataset {} in HDF-file {}", path, filenames_[i]);
    int rank = H5Sget_simple_extent_ndims(dataspace);
    switch (rank)
    {
      case 0:
        break;
      case 1:
      {
        hsize_t dim, maxdim;
        int res = H5Sget_simple_extent_dims(dataspace, &dim, &maxdim);
        if (res < 0)
          FOUR_C_THROW("Failed to get size from dataspace in HDF-file {}", filenames_[i]);
        data->resize(offset + dim);
        herr_t status =
            H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &((*data)[offset]));
        offset += dim;
        if (status < 0)
          FOUR_C_THROW("Failed to read data from dataset {} in HDF-file {}", path, filenames_[i]);
        break;
      }
      default:
        FOUR_C_THROW("HDF5 rank={} unsupported", rank);
        break;
    }

    herr_t status = H5Sclose(dataspace);
    if (status < 0)
      FOUR_C_THROW("Failed to close node dataspace {} in HDF-file {}", path, filenames_[i]);
    status = H5Dclose(dataset);
    if (status < 0)
      FOUR_C_THROW("Failed to close node dataset {} in HDF-file {}", path, filenames_[i]);
  }
  return data;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<std::vector<double>> Core::IO::HDFReader::read_double_data(
    std::string path, int start, int end, std::vector<int>& lengths) const
{
  if (end == -1) end = num_output_proc_;
  int offset = 0;
  std::shared_ptr<std::vector<double>> data = std::make_shared<std::vector<double>>();
  for (int i = start; i < end; ++i)
  {
    hid_t dataset = H5Dopen(files_[i], path.c_str(), H5P_DEFAULT);
    if (dataset < 0) FOUR_C_THROW("Failed to open dataset {} in HDF-file {}", path, filenames_[i]);
    hid_t dataspace = H5Dget_space(dataset);
    if (dataspace < 0)
      FOUR_C_THROW("Failed to get dataspace from dataset {} in HDF-file {}", path, filenames_[i]);
    int rank = H5Sget_simple_extent_ndims(dataspace);
    switch (rank)
    {
      case 0:
        lengths.push_back(0);
        break;
      case 1:
      {
        hsize_t dim, maxdim;
        int res = H5Sget_simple_extent_dims(dataspace, &dim, &maxdim);
        if (res < 0)
          FOUR_C_THROW("Failed to get size from dataspace in HDF-file {}", filenames_[i]);
        data->resize(offset + dim);
        herr_t status =
            H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &((*data)[offset]));
        lengths.push_back(dim);
        offset += dim;
        if (status < 0)
          FOUR_C_THROW("Failed to read data from dataset {} in HDF-file {}", path, filenames_[i]);
        break;
      }
      default:
        FOUR_C_THROW("HDF5 rank={} unsupported", rank);
        break;
    }

    herr_t status = H5Sclose(dataspace);
    if (status < 0)
      FOUR_C_THROW("Failed to close node dataspace {} in HDF-file {}", path, filenames_[i]);
    status = H5Dclose(dataset);
    if (status < 0)
      FOUR_C_THROW("Failed to close node dataset {} in HDF-file {}", path, filenames_[i]);
  }
  return data;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> Core::IO::HDFReader::read_result_data(
    std::string id_path, std::string value_path, int columns, MPI_Comm Comm) const
{
  int new_proc_num = Core::Communication::num_mpi_ranks(Comm);
  int my_id = Core::Communication::my_mpi_rank(Comm);

  if (files_.size() == 0) FOUR_C_THROW("Tried to read data without opening any file");
  int start, end;
  calculate_range(new_proc_num, my_id, start, end);

  std::shared_ptr<std::vector<int>> ids = read_int_data(id_path, start, end);
  Core::LinAlg::Map map(-1, std::span<const int>(ids->data(), ids->size()), 0, Comm);

  std::shared_ptr<Core::LinAlg::MultiVector<double>> res =
      std::make_shared<Core::LinAlg::MultiVector<double>>(map, columns, false);

  std::vector<int> lengths;
  std::shared_ptr<std::vector<double>> values = read_double_data(value_path, start, end, lengths);

  if (static_cast<int>(values->size()) != res->local_length() * res->num_vectors())
    FOUR_C_THROW("vector value size mismatch: {} != {}", values->size(),
        res->local_length() * res->num_vectors());

  // Rearrange multi vectors that are read with fewer processors than written.
  int offset = 0;
  for (int i = start; i < end; ++i)
  {
    int l = lengths[i - start];
    for (int c = 0; c < columns; ++c)
    {
      std::copy(values->data() + offset + c * l / columns,
          values->data() + offset + (c + 1) * l / columns,
          res->get_values() + c * res->local_length() + offset / columns);
    }
    offset += l;
  }

  return res;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<std::vector<char>> Core::IO::HDFReader::read_result_data_vec_char(
    std::string id_path, std::string value_path, int columns, MPI_Comm Comm,
    std::shared_ptr<Core::LinAlg::Map>& elemap) const
{
  if (columns != 1) FOUR_C_THROW("got multivector, std::vector<char> expected");

  int new_proc_num = Core::Communication::num_mpi_ranks(Comm);
  int my_id = Core::Communication::my_mpi_rank(Comm);

  if (files_.size() == 0) FOUR_C_THROW("Tried to read data without opening any file");
  int start, end;
  calculate_range(new_proc_num, my_id, start, end);

  std::shared_ptr<std::vector<int>> ids = read_int_data(id_path, start, end);
  // cout << "size of ids:" << (*ids).size() << endl;
  Core::LinAlg::Map map(-1, std::span<const int>(ids->data(), ids->size()), 0, Comm);
  elemap = std::make_shared<Core::LinAlg::Map>(map);

  std::shared_ptr<std::vector<char>> res = read_char_data(value_path, start, end);
  return res;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<std::vector<char>> Core::IO::HDFReader::read_char_vector(
    std::string value_path, MPI_Comm Comm) const
{
  int new_proc_num = Core::Communication::num_mpi_ranks(Comm);
  int my_id = Core::Communication::my_mpi_rank(Comm);

  if (files_.size() == 0) FOUR_C_THROW("Tried to read data without opening any file");
  int start, end;
  calculate_range(new_proc_num, my_id, start, end);

  std::shared_ptr<std::vector<char>> res = read_char_data(value_path, start, end);
  return res;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::HDFReader::close()
{
  for (int i = 0; i < num_output_proc_ and i < static_cast<int>(files_.size()); ++i)
  {
    if (files_[i] != -1)
    {
      herr_t status = H5Fclose(files_[i]);
      if (status < 0) FOUR_C_THROW("Failed to close HDF-file {}", filenames_[i]);
      files_[i] = -1;
    }
  }
  filenames_.resize(0);
  files_.resize(0);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::HDFReader::calculate_range(int new_proc_num, int my_id, int& start, int& end) const
{
  const int mod = num_output_proc_ % new_proc_num;
  if (my_id < mod)
  {
    start = (num_output_proc_ / new_proc_num + 1) * my_id;
    end = (num_output_proc_ / new_proc_num + 1) * (my_id + 1);
  }
  else
  {
    start = (num_output_proc_ / new_proc_num + 1) * mod +
            (num_output_proc_ / new_proc_num) * (my_id - mod);
    end = (num_output_proc_ / new_proc_num + 1) * mod +
          (num_output_proc_ / new_proc_num) * (my_id - mod + 1);
  }
}

FOUR_C_NAMESPACE_CLOSE

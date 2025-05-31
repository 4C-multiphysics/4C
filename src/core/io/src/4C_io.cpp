// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_io_control.hpp"
#include "4C_io_legacy_table.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::DiscretizationReader::DiscretizationReader() /* [PROTECTED] */
    : dis_(nullptr), input_(nullptr), restart_step_(nullptr), reader_(nullptr), meshreader_(nullptr)
{
  // intentionally left blank
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::DiscretizationReader::DiscretizationReader(std::shared_ptr<Core::FE::Discretization> dis,
    std::shared_ptr<Core::IO::InputControl> input, int step)
    : dis_(dis), input_(input)
{
  find_result_group(step, input_->control_file());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Core::IO::DiscretizationReader::has_int(std::string name)
{
  int integer;
  return map_find_int(restart_step_, name.c_str(), &integer);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> Core::IO::DiscretizationReader::read_vector(
    std::string name)
{
  MAP* result = map_read_map(restart_step_, name.c_str());
  int columns;
  if (map_find_int(result, "columns", &columns))
  {
    if (columns != 1) FOUR_C_THROW("got multivector with name '{}', vector expected", name);
  }
  return read_multi_vector(name);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationReader::read_vector(
    std::shared_ptr<Core::LinAlg::Vector<double>> vec, std::string name)
{
  read_vector(Utils::shared_ptr_from_ref(vec->as_multi_vector()), name);
}


void Core::IO::DiscretizationReader::read_vector(
    std::shared_ptr<Core::LinAlg::MultiVector<double>> vec, std::string name)
{
  MAP* result = map_read_map(restart_step_, name.c_str());
  int columns;
  if (map_find_int(result, "columns", &columns))
  {
    if (columns != 1) FOUR_C_THROW("got multivector with name '{}', vector expected", name);
  }
  read_multi_vector(vec, name);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>>
Core::IO::DiscretizationReader::read_multi_vector(const std::string name)
{
  MAP* result = map_read_map(restart_step_, name.c_str());
  const std::string id_path = map_read_string(result, "ids");
  const std::string value_path = map_read_string(result, "values");
  int columns;
  if (not map_find_int(result, "columns", &columns))
  {
    columns = 1;
  }
  return reader_->read_result_data(id_path, value_path, columns, get_comm());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationReader::read_multi_vector(
    std::shared_ptr<Core::LinAlg::MultiVector<double>> vec, std::string name)
{
  // check if vec is a null pointer
  if (vec == nullptr)
  {
    FOUR_C_THROW("vec is a null pointer. You need to allocate memory before calling this function");
  }

  std::shared_ptr<Core::LinAlg::MultiVector<double>> nv = read_multi_vector(name);

  if (nv->GlobalLength() > vec->GlobalLength())
    FOUR_C_THROW(
        "Reading vector \"{}\": Global length of source exceeds target "
        "(Multi-) Vector length! This case is not supported ! "
        "Source size: {} Target size: {}",
        name.c_str(), nv->GlobalLength(), vec->GlobalLength());

  Core::LinAlg::export_to(*nv, *vec);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationReader::read_map_data_of_char_vector(
    std::map<int, std::vector<char>>& mapdata, std::string name) const
{
  MAP* result = map_read_map(restart_step_, name.c_str());
  std::string id_path = map_read_string(result, "ids");
  std::string value_path = map_read_string(result, "values");
  int columns = map_find_int(result, "columns", &columns);
  if (not map_find_int(result, "columns", &columns))
  {
    columns = 1;
  }
  if (columns != 1)
    FOUR_C_THROW("got multivector with name '{}', std::vector<char> expected", name);

  std::shared_ptr<Core::LinAlg::Map> elemap;
  std::shared_ptr<std::vector<char>> data =
      reader_->read_result_data_vec_char(id_path, value_path, columns, get_comm(), elemap);

  Communication::UnpackBuffer buffer(*data);
  for (int i = 0; i < elemap->num_my_elements(); ++i)
  {
    std::vector<char> single_entry_data;
    extract_from_pack(buffer, single_entry_data);
    (mapdata)[elemap->gid(i)] = single_entry_data;
  }
}


/*----------------------------------------------------------------------*
 * Read the mesh from restart files                                     *
 *----------------------------------------------------------------------*/
void Core::IO::DiscretizationReader::read_mesh(int step)
{
  dis_->delete_nodes();
  dis_->delete_elements();

  find_mesh_group(step, input_->control_file());

  std::shared_ptr<std::vector<char>> nodedata = meshreader_->read_node_data(step,
      Core::Communication::num_mpi_ranks(get_comm()), Core::Communication::my_mpi_rank(get_comm()));

  std::shared_ptr<std::vector<char>> elementdata = meshreader_->read_element_data(step,
      Core::Communication::num_mpi_ranks(get_comm()), Core::Communication::my_mpi_rank(get_comm()));

  // unpack nodes and elements and redistributed to current layout
  // take care --- we are just adding elements to the discretisation
  // that means depending on the current distribution and the
  // distribution of the data read we might increase the
  // number of elements in dis_
  // the call to redistribute deletes the unnecessary elements,
  // so everything should be OK
  dis_->unpack_my_nodes(*nodedata);
  dis_->unpack_my_elements(*elementdata);

  dis_->setup_ghosting(true, false, false);

  dis_->fill_complete();

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationReader::read_nodes_only(int step)
{
  find_mesh_group(step, input_->control_file());

  std::shared_ptr<std::vector<char>> nodedata = meshreader_->read_node_data(step,
      Core::Communication::num_mpi_ranks(get_comm()), Core::Communication::my_mpi_rank(get_comm()));

  // unpack nodes; fill_complete() has to be called manually
  dis_->unpack_my_nodes(*nodedata);
  return;
}


/*----------------------------------------------------------------------*
 * Read history data from restart files                                 *
 *----------------------------------------------------------------------*/
void Core::IO::DiscretizationReader::read_history_data(int step)
{
  find_mesh_group(step, input_->control_file());

  std::shared_ptr<std::vector<char>> nodedata = meshreader_->read_node_data(step,
      Core::Communication::num_mpi_ranks(get_comm()), Core::Communication::my_mpi_rank(get_comm()));

  std::shared_ptr<std::vector<char>> elementdata = meshreader_->read_element_data(step,
      Core::Communication::num_mpi_ranks(get_comm()), Core::Communication::my_mpi_rank(get_comm()));

  // before we unpack nodes/elements we store a copy of the nodal row/col map
  Core::LinAlg::Map noderowmap(*dis_->node_row_map());
  Core::LinAlg::Map nodecolmap(*dis_->node_col_map());

  // before we unpack nodes/elements we store a copy of the nodal row/col map
  Core::LinAlg::Map elerowmap(*dis_->element_row_map());
  Core::LinAlg::Map elecolmap(*dis_->element_col_map());

  // unpack nodes and elements and redistributed to current layout

  // take care --- we are just adding elements to the discretisation
  // that means depending on the current distribution and the
  // distribution of the data read we might increase the
  // number of elements in dis_
  // the call to redistribute deletes the unnecessary elements,
  // so everything should be OK
  dis_->unpack_my_nodes(*nodedata);
  dis_->unpack_my_elements(*elementdata);
  dis_->redistribute(noderowmap, nodecolmap, elerowmap, elecolmap);
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationReader::read_char_vector(
    std::shared_ptr<std::vector<char>>& charvec, const std::string name)
{
  // read vector properties
  MAP* result = map_read_map(restart_step_, name.c_str());
  std::string value_path = map_read_string(result, "values");

  charvec = reader_->read_char_vector(value_path, dis_->get_comm());

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationReader::read_redundant_double_vector(
    std::shared_ptr<std::vector<double>>& doublevec, const std::string name)
{
  int length;

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    // only proc0 reads the vector entities
    MAP* result = map_read_map(restart_step_, name.c_str());
    std::string value_path = map_read_string(result, "values");

    doublevec = reader_->read_double_vector(value_path);

    length = doublevec->size();
  }

  // communicate the length of the vector to come
  Core::Communication::broadcast(&length, 1, 0, get_comm());

  // make vector having the correct length on all procs
  doublevec->resize(length);

  // now distribute information to all procs
  Core::Communication::broadcast(doublevec->data(), length, 0, get_comm());
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationReader::read_redundant_int_vector(
    std::shared_ptr<std::vector<int>>& intvec, const std::string name)
{
  int length;

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    // only proc0 reads the vector entities
    MAP* result = map_read_map(restart_step_, name.c_str());
    std::string value_path = map_read_string(result, "values");

    intvec = reader_->read_int_vector(value_path);

    length = intvec->size();
  }

  // communicate the length of the vector to come
  Core::Communication::broadcast(&length, 1, 0, get_comm());

  // make vector having the correct length on all procs
  intvec->resize(length);

  // now distribute information to all procs
  Core::Communication::broadcast(intvec->data(), length, 0, get_comm());
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Core::IO::DiscretizationReader::read_int(std::string name)
{
  return map_read_int(restart_step_, name.c_str());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Core::IO::DiscretizationReader::read_double(std::string name)
{
  return map_read_real(restart_step_, name.c_str());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationReader::find_group(int step, MAP* file, const char* caption,
    const char* filestring, MAP*& result_info, MAP*& file_info)
{
  SYMBOL* symbol;

  /* Iterate all symbols under the name "result" and get the one that
   * matches the given step. Note that this iteration starts from the
   * last result group and goes backward. */

  std::string name = dis_->name();

  symbol = map_find_symbol(file, caption);
  while (symbol != nullptr)
  {
    if (symbol_is_map(symbol))
    {
      MAP* map;
      symbol_get_map(symbol, &map);
      if (map_has_string(map, "field", name.c_str()) and map_has_int(map, "step", step))
      {
        result_info = map;
        break;
      }
    }
    symbol = symbol->next;
  }
  if (symbol == nullptr)
  {
    FOUR_C_THROW(
        "No restart entry for discretization '{}' step {} in symbol table. "
        "Control file corrupt?\n\nLooking for control file at: {}",
        name.c_str(), step, input_->file_name().c_str());
  }

  /*--------------------------------------------------------------------*/
  /* open file to read */

  /* We have a symbol and its map that corresponds to the step we are
   * interested in. Now we need to continue our search to find the
   * step that defines the output file used for our step. */

  while (symbol != nullptr)
  {
    if (symbol_is_map(symbol))
    {
      MAP* map;
      symbol_get_map(symbol, &map);
      if (map_has_string(map, "field", name.c_str()))
      {
        /*
         * If one of these files is here the other one has to be
         * here, too. If it's not, it's a bug in the input. */
        if (map_symbol_count(map, filestring) > 0)
        {
          file_info = map;
          break;
        }
      }
    }
    symbol = symbol->next;
  }

  /* No restart files defined? */
  if (symbol == nullptr)
  {
    FOUR_C_THROW("no restart file definitions found in control file");
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationReader::find_result_group(int step, MAP* file)
{
  MAP* result_info = nullptr;
  MAP* file_info = nullptr;

  find_group(step, file, "result", "result_file", result_info, file_info);
  reader_ = open_files("result_file", file_info);

  restart_step_ = result_info;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MPI_Comm Core::IO::DiscretizationReader::get_comm() const { return dis_->get_comm(); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationReader::find_mesh_group(int step, MAP* file)
{
  MAP* result_info = nullptr;
  MAP* file_info = nullptr;

  find_group(step, file, "field", "mesh_file", result_info, file_info);
  meshreader_ = open_files("mesh_file", file_info);

  // We do not need result_info as we are interested in the mesh files only.
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::IO::HDFReader> Core::IO::DiscretizationReader::open_files(
    const char* filestring, MAP* result_step)
{
  int numoutputproc;
  if (!map_find_int(result_step, "num_output_proc", &numoutputproc))
  {
    numoutputproc = 1;
  }

  const std::string name = input_->file_name();

  std::string dirname;
  const std::string::size_type pos = name.find_last_of('/');
  if (pos == std::string::npos)
  {
    dirname = "";
  }
  else
  {
    dirname = name.substr(0, pos + 1);
  }

  const std::string filename = map_read_string(result_step, filestring);

  std::shared_ptr<HDFReader> reader = std::make_shared<HDFReader>(dirname);
  reader->open(filename, numoutputproc, Core::Communication::num_mpi_ranks(get_comm()),
      Core::Communication::my_mpi_rank(get_comm()));
  return reader;
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::DiscretizationWriter::DiscretizationWriter() /* PROTECTED */
    : dis_(nullptr),
      step_(-1),
      time_(-1.0),
      meshfile_(-1),
      resultfile_(-1),
      meshfilename_(),
      resultfilename_(),
      meshgroup_(-1),
      resultgroup_(-1),
      resultfile_changed_(-1),
      meshfile_changed_(-1),
      output_(nullptr),
      binio_(false),
      spatial_approx_(Core::FE::ShapeFunctionType::undefined)
{
  // intentionally left blank
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::DiscretizationWriter::DiscretizationWriter(std::shared_ptr<Core::FE::Discretization> dis,
    std::shared_ptr<OutputControl> output_control,
    const Core::FE::ShapeFunctionType shape_function_type)
    : dis_(Core::Utils::shared_ptr_from_ref(
          *dis.get())),  // no ownership to break circle discretization<>writer
      step_(-1),
      time_(-1.0),
      meshfile_(-1),
      resultfile_(-1),
      meshfilename_(),
      resultfilename_(),
      meshgroup_(-1),
      resultgroup_(-1),
      resultfile_changed_(-1),
      meshfile_changed_(-1),
      output_(output_control),
      spatial_approx_(shape_function_type)
{
  if (output_ != nullptr)
    binio_ = output_->write_binary_output();
  else
    binio_ = false;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::DiscretizationWriter::DiscretizationWriter(const Core::IO::DiscretizationWriter& writer,
    const std::shared_ptr<OutputControl>& control, enum CopyType type)
    : dis_(Core::Utils::shared_ptr_from_ref(*writer.dis_.get())),
      step_(-1),
      time_(-1.0),
      meshfile_(-1),
      resultfile_(-1),
      meshfilename_(),
      resultfilename_(),
      meshgroup_(-1),
      resultgroup_(-1),
      resultfile_changed_(-1),
      meshfile_changed_(-1),
      output_(nullptr),
      binio_(false),
      spatial_approx_(writer.spatial_approx_)
{
  output_ = (!control ? writer.output_ : control);
  if (output_) binio_ = output_->write_binary_output();

  if (type == CopyType::deep)
  {
    step_ = writer.step_;
    time_ = writer.time_;
    meshfile_ = writer.meshfile_;
    resultfile_ = writer.resultfile_;
    meshfilename_ = writer.resultfile_;
    resultfilename_ = writer.resultfile_;
    meshgroup_ = writer.resultfile_;
    resultgroup_ = writer.resultfile_;
    resultfile_changed_ = writer.resultfile_;
    meshfile_changed_ = writer.resultfile_;
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::DiscretizationWriter::~DiscretizationWriter()
{
  if (meshfile_ != -1)
  {
    const herr_t status = H5Fclose(meshfile_);
    if (status < 0)
    {
      FOUR_C_THROW("Failed to close HDF file {}", meshfilename_);
    }
  }
  if (resultfile_ != -1)
  {
    // apparently H5Fclose(resultfile_); does not close the file, if there are still
    // some open groups or datasets, so close them first

    // Get the number of open groups
    int num_og = H5Fget_obj_count(resultfile_, H5F_OBJ_GROUP);

    // get vector to store ids of open groups
    std::vector<hid_t> oid_list(num_og, -1);

    herr_t status = H5Fget_obj_ids(resultfile_, H5F_OBJ_GROUP, num_og, oid_list.data());
    if (status < 0) FOUR_C_THROW("Failed to get id's of open groups in resultfile");

    // loop over open groups
    for (int i = 0; i < num_og; i++)
    {
      const herr_t status_g = H5Gclose(oid_list[i]);
      if (status_g < 0) FOUR_C_THROW("Failed to close HDF-group in resultfile");
    }
    // now close the result file
    const herr_t status_c = H5Fclose(resultfile_);
    if (status_c < 0)
    {
      FOUR_C_THROW("Failed to close HDF file {}", resultfilename_);
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MPI_Comm Core::IO::DiscretizationWriter::get_comm() const { return dis_->get_comm(); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::create_mesh_file(const int step)
{
  if (binio_)
  {
    std::ostringstream meshname;

    meshname << output_->file_name() << ".mesh." << dis_->name() << ".s" << step;
    meshfilename_ = meshname.str();
    if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    {
      meshname << ".p" << Core::Communication::my_mpi_rank(get_comm());
    }

    if (meshfile_ != -1)
    {
      const herr_t status = H5Fclose(meshfile_);
      if (status < 0)
      {
        FOUR_C_THROW("Failed to close HDF file {}", meshfilename_);
      }
    }

    meshfile_ = H5Fcreate(meshname.str().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (meshfile_ < 0) FOUR_C_THROW("Failed to open file {}", meshname.str());
    meshfile_changed_ = step;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::create_result_file(const int step)
{
  if (binio_)
  {
    std::ostringstream resultname;
    resultname << output_->file_name() << ".result." << dis_->name() << ".s" << step;

    resultfilename_ = resultname.str();
    if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    {
      resultname << ".p" << Core::Communication::my_mpi_rank(get_comm());
    }
    if (resultfile_ != -1)
    {
      herr_t status = H5Fclose(resultfile_);
      if (status < 0)
      {
        FOUR_C_THROW("Failed to close HDF file {}", resultfilename_);
      }
    }

    // we will never refer to maps stored in other files
    mapcache_.clear();
    mapstack_.clear();

    resultfile_ = H5Fcreate(resultname.str().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (resultfile_ < 0) FOUR_C_THROW("Failed to open file {}", resultname.str());
    resultfile_changed_ = step;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::new_result_file(int numb_run)
{
  if (binio_)
  {
    create_new_result_and_mesh_file();
    output_->new_result_file(numb_run, spatial_approx_);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::new_result_file(std::string name_appendix, int numb_run)
{
  if (binio_)
  {
    create_new_result_and_mesh_file();
    output_->new_result_file(name_appendix, numb_run, spatial_approx_);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::new_result_file(std::string name)
{
  if (binio_)
  {
    create_new_result_and_mesh_file();
    output_->new_result_file(name, spatial_approx_);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::overwrite_result_file()
{
  if (binio_)
  {
    create_new_result_and_mesh_file();
    output_->overwrite_result_file(spatial_approx_);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::new_step(const int step, const double time)
{
  if (binio_)
  {
    bool write_file = false;

    if (not have_result_or_mesh_file_changed())
    {
      // do not perform the step if already called
      if (step_ == step and fabs(time_ - time) < 1e-14)
      {
        return;
      }
    }

    step_ = step;
    time_ = time;
    std::ostringstream groupname;
    groupname << "step" << step_;

    if (resultgroup_ != -1)
    {
      const herr_t status = H5Gclose(resultgroup_);
      if (status < 0)
      {
        FOUR_C_THROW("Failed to close HDF group in file {}", resultfilename_);
      }
    }

    if (step_ - resultfile_changed_ >= output_->file_steps() or resultfile_changed_ == -1)
    {
      create_result_file(step_);
      write_file = true;
    }

    resultgroup_ =
        H5Gcreate(resultfile_, groupname.str().c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (resultgroup_ < 0) FOUR_C_THROW("Failed to write HDF-group in resultfile");

    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      output_->control_file() << "result:\n"
                              << "    field = \"" << dis_->name() << "\"\n"
                              << std::setprecision(16) << "    time = " << time << "\n"
                              << "    step = " << step << "\n\n";

      if (write_file)
      {
        if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
        {
          output_->control_file() << "    num_output_proc = "
                                  << Core::Communication::num_mpi_ranks(get_comm()) << "\n";
        }
        std::string filename;
        const std::string::size_type pos = resultfilename_.find_last_of('/');
        if (pos == std::string::npos)
          filename = resultfilename_;
        else
          filename = resultfilename_.substr(pos + 1);
        output_->control_file() << "    result_file = \"" << filename << "\"\n\n";
      }
      output_->control_file() << std::flush;
    }
    const herr_t status = H5Fflush(resultgroup_, H5F_SCOPE_LOCAL);
    if (status < 0)
    {
      FOUR_C_THROW("Failed to flush HDF file {}", resultfilename_);
    }
  }
}

/*----------------------------------------------------------------------*/
/*write double to control file                                  tk 04/08*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::write_double(const std::string name, const double value)
{
  if (binio_)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      // using a local stringstream we make sure that we do not change
      // the output formatting of control file permanently
      std::stringstream s;
      s << "    " << name << " = " << std::scientific << std::setprecision(16) << value << "\n\n"
        << std::flush;
      output_->control_file() << s.str() << std::flush;
    }
  }
}

/*----------------------------------------------------------------------*/
/*write int to control file                                     tk 04/08*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::write_int(const std::string name, const int value)
{
  if (binio_)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      output_->control_file() << "    " << name << " = " << value << "\n\n" << std::flush;
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::write_vector(const std::string name,
    std::shared_ptr<const Core::LinAlg::Vector<double>> vec, IO::VectorType vt)
{
  write_multi_vector(name, *vec, vt);
}

void Core::IO::DiscretizationWriter::write_multi_vector(
    const std::string name, const Core::LinAlg::MultiVector<double>& vec, IO::VectorType vt)
{
  if (binio_)
  {
    std::string valuename = name + ".values";
    double* data = vec.Values();
    const hsize_t size = vec.MyLength() * vec.NumVectors();
    if (size != 0)
    {
      const herr_t make_status =
          H5LTmake_dataset_double(resultgroup_, valuename.c_str(), 1, &size, data);
      if (make_status < 0)
        FOUR_C_THROW("Failed to create dataset in HDF-resultfile. status={}", make_status);
    }
    else
    {
      const herr_t make_status =
          H5LTmake_dataset_double(resultgroup_, valuename.c_str(), 0, &size, data);
      if (make_status < 0)
        FOUR_C_THROW("Failed to create dataset in HDF-resultfile. status={}", make_status);
    }

    std::string idname;

    // We maintain a map cache to avoid rewriting the same map all the
    // time. The idea is that a map is never modified once it is
    // constructed. Thus the internal data class can be used to find
    // identical maps easily. This will not find all identical maps, but
    // all maps with the same data pointer are guaranteed to be
    // identical.

    std::ostringstream groupname;
    groupname << "/step" << step_ << "/";

    valuename = groupname.str() + valuename;

    const Epetra_BlockMapData* mapdata = vec.get_map().get_data_ptr();
    std::map<const Epetra_BlockMapData*, std::string>::const_iterator m = mapcache_.find(mapdata);
    if (m != mapcache_.end())
    {
      // the map has been written already, just link to it again
      idname = m->second;
    }
    else
    {
      const hsize_t mapsize = vec.MyLength();
      idname = name + ".ids";
      int* ids = vec.get_map().my_global_elements();
      if (size != 0)
      {
        const herr_t make_status =
            H5LTmake_dataset_int(resultgroup_, idname.c_str(), 1, &mapsize, ids);
        if (make_status < 0) FOUR_C_THROW("Failed to create dataset in HDF-resultfile");
      }
      else
      {
        const herr_t make_status =
            H5LTmake_dataset_int(resultgroup_, idname.c_str(), 0, &mapsize, ids);
        if (make_status < 0) FOUR_C_THROW("Failed to create dataset in HDF-resultfile");
      }

      idname = groupname.str() + idname;

      // remember where we put the map
      mapcache_[mapdata] = idname;

      /* Make a copy of the map. This is a std::shared_ptr copy internally. We
       * just make sure here the map stays alive as long as we keep our cache.
       * Otherwise subtle errors could occur. */
      mapstack_.push_back(vec.get_map());
      /* BUT: If a problem relies on fill_complete()-calls in every time step,
       * new maps are created in every time step. Storing all old maps in
       * mapstack_ leads to an unbounded increase in memory consumption which
       * has to be strictly avoided.
       * Remedy: clear_map_cache() can be called to get rid of old maps when too
       * many are stored. The following limit of 20 is somehow arbitrary and
       * could be increased. The basic idea is: 3 maps (dofrow, noderow and
       * elerow) per field involved in the problem plus some more in case mapstack_
       * is not cleared after each time step */
      if (mapstack_.size() > 20)
        FOUR_C_THROW(
            "Careful! Due to repeated fill_complete()-calls many maps are stored in the output "
            "process.");
    }

    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      std::string vectortype;
      switch (vt)
      {
        case dofvector:
          vectortype = "dof";
          break;
        case nodevector:
          vectortype = "node";
          break;
        case elementvector:
          vectortype = "element";
          break;
        default:
          FOUR_C_THROW("unknown vector type {}", vt);
          break;
      }
      output_->control_file() << "    " << name << ":\n"
                              << "        type = \"" << vectortype << "\"\n"
                              << "        columns = " << vec.NumVectors() << "\n"
                              << "        values = \"" << valuename.c_str() << "\"\n"
                              << "        ids = \"" << idname.c_str()
                              << "\"\n\n"  // different names + other information?
                              << std::flush;
    }
    const herr_t flush_status = H5Fflush(resultgroup_, H5F_SCOPE_LOCAL);
    if (flush_status < 0)
    {
      FOUR_C_THROW("Failed to flush HDF file {}", resultfilename_);
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::write_vector(const std::string name,
    const std::vector<char>& vec, const Core::LinAlg::Map& elemap, IO::VectorType vt)
{
  if (binio_)
  {
    std::string valuename = name + ".values";
    const hsize_t size = vec.size();
    const char* data = vec.data();
    if (size != 0)
    {
      const herr_t make_status =
          H5LTmake_dataset_char(resultgroup_, valuename.c_str(), 1, &size, data);
      if (make_status < 0)
        FOUR_C_THROW("Failed to create dataset in HDF-resultfile. status={}", make_status);
    }
    else
    {
      const herr_t make_status =
          H5LTmake_dataset_char(resultgroup_, valuename.c_str(), 0, &size, data);
      if (make_status < 0)
        FOUR_C_THROW("Failed to create dataset in HDF-resultfile. status={}", make_status);
    }

    std::string idname;

    // We maintain a map cache to avoid rewriting the same map all the
    // time. The idea is that a map is never modified once it is
    // constructed. Thus the internal data class can be used to find
    // identical maps easily. This will not find all identical maps, but
    // all maps with the same data pointer are guaranteed to be
    // identical.

    std::ostringstream groupname;
    groupname << "/step" << step_ << "/";

    valuename = groupname.str() + valuename;

    const Epetra_BlockMapData* mapdata = elemap.get_data_ptr();
    std::map<const Epetra_BlockMapData*, std::string>::const_iterator m = mapcache_.find(mapdata);
    if (m != mapcache_.end())
    {
      // the map has been written already, just link to it again
      idname = m->second;
    }
    else
    {
      const hsize_t mapsize = elemap.num_my_elements();
      idname = name + ".ids";
      int* ids = elemap.my_global_elements();
      const herr_t make_status =
          H5LTmake_dataset_int(resultgroup_, idname.c_str(), 1, &mapsize, ids);
      if (make_status < 0) FOUR_C_THROW("Failed to create dataset in HDF-resultfile");

      idname = groupname.str() + idname;

      // remember where we put the map
      mapcache_[mapdata] = idname;

      /* Make a copy of the map. This is a std::shared_ptr copy internally. We
       * just make sure here the map stays alive as long as we keep our cache.
       * Otherwise subtle errors could occur. */
      mapstack_.push_back(elemap);
      /* BUT: If a problem relies on fill_complete()-calls in every time step,
       * new maps are created in every time step. Storing all old maps in
       * mapstack_ leads to an unbounded increase in memory consumption which
       * has to be strictly avoided.
       * Remedy: clear_map_cache() can be called to get rid of old maps when too
       * many are stored. The following limit of 20 is somehow arbitrary and
       * could be increased. The basic idea is: 3 maps (dofrow, noderow and
       * elerow) per field involved in the problem plus some more in case mapstack_
       * is not cleared after each time step */
      if (mapstack_.size() > 20)
        FOUR_C_THROW(
            "Careful! Due to repeated fill_complete()-calls many maps are "
            "stored in the output process.");
    }

    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      std::string vectortype;
      switch (vt)
      {
        case dofvector:
          vectortype = "dof";
          break;
        case nodevector:
          vectortype = "node";
          break;
        case elementvector:
          vectortype = "element";
          break;
        default:
          FOUR_C_THROW("unknown vector type {}", vt);
          break;
      }
      output_->control_file() << "    " << name << ":\n"
                              << "        type = \"" << vectortype << "\"\n"
                              << "        columns = 1\n"
                              << "        values = \"" << valuename << "\"\n"
                              << "        ids = \"" << idname
                              << "\"\n\n"  // different names + other information?
                              << std::flush;
    }
    const herr_t flush_status = H5Fflush(resultgroup_, H5F_SCOPE_LOCAL);
    if (flush_status < 0)
    {
      FOUR_C_THROW("Failed to flush HDF file {}", resultfilename_);
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::write_mesh(const int step, const double time)
{
  if (binio_)
  {
    if (step - meshfile_changed_ >= output_->file_steps() or meshfile_changed_ == -1)
    {
      create_mesh_file(step);
    }
    std::ostringstream name;
    name << "step" << step;
    meshgroup_ = H5Gcreate(meshfile_, name.str().c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (meshgroup_ < 0) FOUR_C_THROW("Failed to write group in HDF-meshfile");

    // only procs with row elements need to write data
    std::shared_ptr<std::vector<char>> elementdata = dis_->pack_my_elements();
    hsize_t dim = static_cast<hsize_t>(elementdata->size());
    if (dim != 0)
    {
      const herr_t element_status =
          H5LTmake_dataset_char(meshgroup_, "elements", 1, &dim, elementdata->data());
      if (element_status < 0) FOUR_C_THROW("Failed to create dataset in HDF-meshfile");
    }
    else
    {
      const herr_t element_status =
          H5LTmake_dataset_char(meshgroup_, "elements", 0, &dim, elementdata->data());
      if (element_status < 0)
        FOUR_C_THROW(
            "Failed to create dataset in HDF-meshfile on proc {} which does"
            " not have row elements",
            Core::Communication::my_mpi_rank(get_comm()));
    }

    // only procs with row nodes need to write data
    std::shared_ptr<std::vector<char>> nodedata = dis_->pack_my_nodes();
    dim = static_cast<hsize_t>(nodedata->size());
    if (dim != 0)
    {
      const herr_t node_status =
          H5LTmake_dataset_char(meshgroup_, "nodes", 1, &dim, nodedata->data());
      if (node_status < 0) FOUR_C_THROW("Failed to create dataset in HDF-meshfile");
    }
    else
    {
      const herr_t node_status =
          H5LTmake_dataset_char(meshgroup_, "nodes", 0, &dim, nodedata->data());
      if (node_status < 0)
        FOUR_C_THROW(
            "Failed to create dataset in HDF-meshfile on proc {} which"
            " does not have row nodes",
            Core::Communication::my_mpi_rank(get_comm()));
    }

    int max_nodeid = dis_->node_row_map()->max_all_gid();

    // ... write other mesh information
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      output_->control_file() << "field:\n"
                              << "    field = \"" << dis_->name() << "\"\n"
                              << std::setprecision(16) << "    time = " << time << "\n"
                              << "    step = " << step << "\n\n"
                              << "    num_nd = " << dis_->num_global_nodes() << "\n"
                              << "    max_nodeid = " << max_nodeid << "\n"
                              << "    num_ele = " << dis_->num_global_elements() << "\n"
                              << "    num_dof = " << dis_->dof_row_map(0)->num_global_elements()
                              << "\n"
                              << "    num_dim = " << dis_->n_dim() << "\n\n";

      // knotvectors for nurbs-discretisation
      write_knotvector();

      if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
      {
        output_->control_file() << "    num_output_proc = "
                                << Core::Communication::num_mpi_ranks(get_comm()) << "\n";
      }
      std::string filename;
      std::string::size_type pos = meshfilename_.find_last_of('/');
      if (pos == std::string::npos)
        filename = meshfilename_;
      else
        filename = meshfilename_.substr(pos + 1);
      output_->control_file() << "    mesh_file = \"" << filename << "\"\n\n";
      output_->control_file() << std::flush;
    }
    const herr_t flush_status = H5Fflush(meshgroup_, H5F_SCOPE_LOCAL);
    if (flush_status < 0)
    {
      FOUR_C_THROW("Failed to flush HDF file {}", meshfilename_);
    }
    const herr_t close_status = H5Gclose(meshgroup_);
    if (close_status < 0)
    {
      FOUR_C_THROW("Failed to close HDF group in file {}", meshfilename_);
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::write_mesh(
    const int step, const double time, std::string name_base_file)
{
  if (binio_)
  {
    // ... write other mesh information
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      output_->control_file() << "field:\n"
                              << "    field = \"" << dis_->name() << "\"\n"
                              << std::setprecision(16) << "    time = " << time << "\n"
                              << "    step = " << step << "\n\n"
                              << "    num_nd = " << dis_->num_global_nodes() << "\n"
                              << "    num_ele = " << dis_->num_global_elements() << "\n"
                              << "    num_dof = " << dis_->dof_row_map(0)->num_global_elements()
                              << "\n"
                              << "    num_dim = " << dis_->n_dim() << "\n\n";

      // knotvectors for nurbs-discretisation
      // write_knotvector();
      // create name for meshfile as in createmeshfile which is not called here
      std::ostringstream meshname;

      meshname << name_base_file << ".mesh." << dis_->name() << ".s" << step;
      meshfilename_ = meshname.str();

      if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
      {
        output_->control_file() << "    num_output_proc = "
                                << Core::Communication::num_mpi_ranks(get_comm()) << "\n";
      }
      std::string filename;
      std::string::size_type pos = meshfilename_.find_last_of('/');
      if (pos == std::string::npos)
        filename = meshfilename_;
      else
        filename = meshfilename_.substr(pos + 1);
      output_->control_file() << "    mesh_file = \"" << filename << "\"\n\n";
      // << "    mesh_file = \"" << name_base_file << "\"\n\n";

      output_->control_file() << std::flush;
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::write_only_nodes_in_new_field_group_to_control_file(
    const int step, const double time, const bool writerestart)
{
  if (binio_)
  {
    if (step - meshfile_changed_ >= output_->file_steps() or meshfile_changed_ == -1)
    {
      create_mesh_file(step);
    }
    std::ostringstream name;
    name << "step" << step;
    meshgroup_ = H5Gcreate(meshfile_, name.str().c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (meshgroup_ < 0) FOUR_C_THROW("Failed to write group in HDF-meshfile");

    if (writerestart)
    {
      // only for restart: procs with row nodes need to write data
      std::shared_ptr<std::vector<char>> nodedata = dis_->pack_my_nodes();
      hsize_t dim = static_cast<hsize_t>(nodedata->size());
      if (dim != 0)
      {
        const herr_t node_status =
            H5LTmake_dataset_char(meshgroup_, "nodes", 1, &dim, nodedata->data());
        if (node_status < 0) FOUR_C_THROW("Failed to create dataset in HDF-meshfile");
      }
      else
      {
        const herr_t node_status =
            H5LTmake_dataset_char(meshgroup_, "nodes", 0, &dim, nodedata->data());
        if (node_status < 0)
          FOUR_C_THROW(
              "Failed to create dataset in HDF-meshfile on proc {} which "
              "does not have row nodes",
              Core::Communication::my_mpi_rank(get_comm()));
      }
    }

    /* nodes do not have to be written for standard output; only number of
     * nodes is important more exactly: the maximum nodal id is used to
     * determine number of particles during output unused particles are
     * located at the origin and are waiting for activation */
    int max_nodeid = dis_->node_row_map()->max_all_gid();

    // ... write other mesh information
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      /* number of nodes and elements is set to zero to suppress reading of
       * nodes during post-processing only maxnodeid is important */
      output_->control_file() << "field:\n"
                              << "    field = \"" << dis_->name() << "\"\n"
                              << std::setprecision(16) << "    time = " << time << "\n"
                              << "    step = " << step << "\n\n"
                              << "    num_nd = " << 0 << "\n"
                              << "    max_nodeid = " << max_nodeid << "\n"
                              << "    num_ele = " << 0 << "\n"
                              << "    num_dof = " << dis_->dof_row_map(0)->num_global_elements()
                              << "\n"
                              << "    num_dim = " << dis_->n_dim() << "\n\n";

      /* name of the output file must be specified for changing geometries in
       * each time step */
      if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
      {
        output_->control_file() << "    num_output_proc = "
                                << Core::Communication::num_mpi_ranks(get_comm()) << "\n";
      }
      std::string filename;
      std::string::size_type pos = meshfilename_.find_last_of('/');
      if (pos == std::string::npos)
        filename = meshfilename_;
      else
        filename = meshfilename_.substr(pos + 1);
      output_->control_file() << "    mesh_file = \"" << filename << "\"\n\n";

      output_->control_file() << std::flush;
    }
    const herr_t flush_status = H5Fflush(meshgroup_, H5F_SCOPE_LOCAL);
    if (flush_status < 0)
    {
      FOUR_C_THROW("Failed to flush HDF file {}", meshfilename_);
    }
    const herr_t close_status = H5Gclose(meshgroup_);
    if (close_status < 0)
    {
      FOUR_C_THROW("Failed to close HDF group in file {}", meshfilename_);
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::write_element_data(bool writeowner)
{
  if (binio_)
  {
    std::map<std::string, int> names;  // contains name and dimension of data

    // loop all elements and build map of data names and dimensions
    if (writeowner == true)
    {
      for (auto* ele : dis_->my_row_element_range())
      {
        // write owner of every element
        ele->vis_owner(names);
      }
    }

    for (auto* ele : dis_->my_row_element_range())
    {
      // get names and dimensions from every element
      ele->vis_names(names);
    }

    // By applying gather_all we get the combined map including all elemental values
    // which where found by vis_names
    Core::LinAlg::gather_all(names, get_comm());

    FOUR_C_ASSERT_ALWAYS(
        std::all_of(names.begin(), names.end(), [](const auto& pair) { return pair.second >= 1; }),
        "Dimension of all data must be at least 1");

    // loop all names acquired form the elements and fill data vectors
    for (const auto& [name, dimension] : names)
    {
      std::vector<double> eledata(dimension);

      // MultiVector stuff from the elements is put in
      Core::LinAlg::MultiVector<double> sysdata(*dis_->element_row_map(), dimension, true);

      int ele_counter = 0;
      for (auto* ele : dis_->my_row_element_range())
      {
        std::fill(eledata.begin(), eledata.end(), 0.0);

        // get data for a given name from element & put in sysdata
        ele->vis_data(name, eledata);

        FOUR_C_ASSERT_ALWAYS(
            (int)eledata.size() == dimension, "element manipulated size of visualization data");
        for (int j = 0; j < dimension; ++j) sysdata(j)[ele_counter] = eledata[j];

        ele_counter++;
      }

      write_multi_vector(name, sysdata, elementvector);
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::write_node_data(bool writeowner)
{
  if (binio_)
  {
    std::map<std::string, int>::const_iterator fool;
    std::map<std::string, int> names;  // contains name and dimension of data

    // loop over all nodes and build map of data names and dimensions
    const Core::LinAlg::Map* noderowmap = dis_->node_row_map();
    if (writeowner == true)
    {
      for (int i = 0; i < noderowmap->num_my_elements(); ++i)
      {
        // write owner of every node
        dis_->l_row_node(i)->vis_owner(names);
      }
    }

    for (int i = 0; i < noderowmap->num_my_elements(); ++i)
    {
      // get names and dimensions from every node
      dis_->l_row_node(i)->vis_names(names);
    }

    /* By applying gather_all we get the combined map including all nodal values
     * which where found by vis_names
     */
    Core::LinAlg::gather_all(names, get_comm());

    // make sure there's no name with a dimension of less than 1
    for (fool = names.begin(); fool != names.end(); ++fool)
      if (fool->second < 1) FOUR_C_THROW("Dimension of data must be at least 1");

    // loop all names acquired form the nodes and fill data vectors
    for (fool = names.begin(); fool != names.end(); ++fool)
    {
      const int dimension = fool->second;
      std::vector<double> nodedata(dimension);

      // MultiVector stuff from the nodes is put in
      Core::LinAlg::MultiVector<double> sysdata(*noderowmap, dimension, true);

      for (int i = 0; i < noderowmap->num_my_elements(); ++i)
      {
        // zero is the default value if not all nodes write the same node data
        for (int idim = 0; idim < dimension; ++idim) nodedata[idim] = 0.0;

        // get data for a given name from node and put in sysdata
        dis_->l_row_node(i)->vis_data(fool->first, nodedata);
        if ((int)nodedata.size() != dimension)
          FOUR_C_THROW("element manipulated size of visualization data");

        for (int j = 0; j < dimension; ++j) sysdata(j)[i] = nodedata[j];
      }

      write_multi_vector(fool->first, sysdata, Core::IO::nodevector);

    }  // for (fool = names.begin(); fool!= names.end(); ++fool)
  }
}

/*----------------------------------------------------------------------*
 *                                                          gammi 05/08 *
 *----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::write_knotvector() const
{
  if (binio_)
  {
    // try a dynamic cast of the discretisation to a nurbs discretisation
    Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
        dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(dis_.get());

    if (nurbsdis != nullptr)
    {
      // get knotvector from nurbsdis
      std::shared_ptr<Core::FE::Nurbs::Knotvector> knots = nurbsdis->get_knot_vector();

      // put knotvector into block
      Core::Communication::PackBuffer block;
      knots->pack(block);

      // write block to file
      if (!block().empty())
      {
        hsize_t dim = static_cast<hsize_t>(block().size());
        const herr_t status =
            H5LTmake_dataset_char(meshgroup_, "knotvector", 1, &dim, block().data());
        if (status < 0) FOUR_C_THROW("Failed to create dataset in HDF-meshfile");
      }
      else
      {
        FOUR_C_THROW("block empty --- couldn't write knots\n");
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*/
/* write a stl vector of chars                                          */
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::write_char_data(
    const std::string name, std::vector<char>& charvec)
{
  if (binio_)
  {
    // only proc0 writes the vector entities to the binary data
    // an appropriate name has to be provided
    std::string valuename = name + ".values";
    const hsize_t size = charvec.size();
    if (size != 0)
    {
      const herr_t make_status =
          H5LTmake_dataset_char(resultgroup_, valuename.c_str(), 1, &size, charvec.data());
      if (make_status < 0)
        FOUR_C_THROW("Failed to create dataset in HDF-resultfile. status={}", make_status);
    }
    else
    {
      const herr_t make_status =
          H5LTmake_dataset_char(resultgroup_, valuename.c_str(), 0, &size, charvec.data());
      if (make_status < 0)
        FOUR_C_THROW("Failed to create dataset in HDF-resultfile. status={}", make_status);
    }

    // ... write other mesh information
    if (Core::Communication::my_mpi_rank(dis_->get_comm()) == 0)
    {
      // do I need the following naming stuff?
      std::ostringstream groupname;

      groupname << "/step" << step_ << "/";

      valuename = groupname.str() + valuename;

      // a comment is also added to the control file
      output_->control_file() << "    " << name << ":\n"
                              << "        values = \"" << valuename.c_str() << "\"\n\n"
                              << std::flush;
    }

    const herr_t flush_status = H5Fflush(resultgroup_, H5F_SCOPE_LOCAL);
    if (flush_status < 0) FOUR_C_THROW("Failed to flush HDF file {}", resultfilename_);
  }
}

/*----------------------------------------------------------------------*/
/* write a stl vector of doubles from proc0                             */
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::write_redundant_double_vector(
    const std::string name, std::vector<double>& doublevec)
{
  if (binio_)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      // only proc0 writes the vector entities to the binary data
      // an appropriate name has to be provided
      std::string valuename = name + ".values";
      const hsize_t size = doublevec.size();
      if (size != 0)
      {
        const herr_t make_status =
            H5LTmake_dataset_double(resultgroup_, valuename.c_str(), 1, &size, doublevec.data());
        if (make_status < 0)
          FOUR_C_THROW("Failed to create dataset in HDF-resultfile. status={}", make_status);
      }
      else
      {
        const herr_t make_status =
            H5LTmake_dataset_double(resultgroup_, valuename.c_str(), 0, &size, doublevec.data());
        if (make_status < 0)
          FOUR_C_THROW("Failed to create dataset in HDF-resultfile. status={}", make_status);
      }

      // do I need the following naming stuff?
      std::ostringstream groupname;

      groupname << "/step" << step_ << "/";

      valuename = groupname.str() + valuename;

      // a comment is also added to the control file
      output_->control_file() << "    " << name << ":\n"
                              << "        values = \"" << valuename.c_str() << "\"\n\n"
                              << std::flush;

      const herr_t flush_status = H5Fflush(resultgroup_, H5F_SCOPE_LOCAL);
      if (flush_status < 0) FOUR_C_THROW("Failed to flush HDF file {}", resultfilename_);
    }  // endif proc0
  }
}

/*----------------------------------------------------------------------*/
/* write a stl set of integers from proc0                             */
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::write_redundant_int_vector(
    const std::string name, std::vector<int>& vectorint)
{
  if (binio_)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      // only proc0 writes the entities to the binary data
      // an appropriate name has to be provided
      std::string valuename = name + ".values";
      const hsize_t size = vectorint.size();
      if (size != 0)
      {
        const herr_t make_status =
            H5LTmake_dataset_int(resultgroup_, valuename.c_str(), 1, &size, vectorint.data());
        if (make_status < 0)
          FOUR_C_THROW("Failed to create dataset in HDF-resultfile. status={}", make_status);
      }
      else
      {
        const herr_t make_status =
            H5LTmake_dataset_int(resultgroup_, valuename.c_str(), 0, &size, vectorint.data());
        if (make_status < 0)
          FOUR_C_THROW("Failed to create dataset in HDF-resultfile. status={}", make_status);
      }

      // do I need the following naming stuff?
      std::ostringstream groupname;

      groupname << "/step" << step_ << "/";

      valuename = groupname.str() + valuename;

      // a comment is also added to the control file
      output_->control_file() << "    " << name << ":\n"
                              << "        values = \"" << valuename.c_str() << "\"\n\n"
                              << std::flush;

      const herr_t flush_status = H5Fflush(resultgroup_, H5F_SCOPE_LOCAL);
      if (flush_status < 0) FOUR_C_THROW("Failed to flush HDF file {}", resultfilename_);
    }  // endif proc0
  }
}


/*----------------------------------------------------------------------*
 |  set output control                               (public) nis Jan14 |
 *----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::set_output(std::shared_ptr<OutputControl> output)
{
  output_ = output;
  binio_ = output_->write_binary_output();
}

/*----------------------------------------------------------------------*/
/* clear all stored map data                                            */
/*----------------------------------------------------------------------*/
void Core::IO::DiscretizationWriter::clear_map_cache()
{
  mapcache_.clear();
  mapstack_.clear();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::FE::Discretization& Core::IO::DiscretizationWriter::get_discretization() const
{
  if (!dis_) FOUR_C_THROW("The discretization pointer has not been initialized!");
  return *dis_;
}

FOUR_C_NAMESPACE_CLOSE

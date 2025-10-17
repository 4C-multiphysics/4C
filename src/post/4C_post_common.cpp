// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_post_common.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_fem_condition_periodic.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_global_legacy_module.hpp"
#include "4C_inpar_problemtype.hpp"
#include "4C_io_legacy_table.hpp"
#include "4C_rigidsphere.hpp"

#include <stack>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 * The main part of this file. All the functions of the three classes
 * PostProblem, PostField and PostResult are defined here.
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 * the Constructor of PostProblem
 *----------------------------------------------------------------------*/
PostProblem::PostProblem(Teuchos::CommandLineProcessor& CLP, int argc, char** argv)
    : start_(0), end_(-1), step_(1), mortar_(false)
{
  using namespace FourC;

  MPI_Init(&argc, &argv);

  global_legacy_module_callbacks().RegisterParObjectTypes();

  std::string file = "xxx";
  std::string output;
  filter_ = "ensight";
  struct_mat_disp_ = "no";
  struct_rot_ = "no";
  std::string mortar_string = "no";

  CLP.throwExceptions(false);
  CLP.setOption("filter", &filter_, "filter to run [ensight, gid, vtu, vtu_node_based, vti]");
  CLP.setOption("start", &start_, "first time step to read");
  CLP.setOption("end", &end_, "last time step to read");
  CLP.setOption("step", &step_, "number of time steps to jump");
  CLP.setOption("file", &file, "control file to open");
  CLP.setOption("output", &output, "output file name [defaults to control file name]");
  CLP.setOption("stresstype", &stresstype_,
      "stress output type [cxyz, ndxyz, cxyz_ndxyz, c123, nd123, c123_nd123]");
  CLP.setOption("stress", &stresstype_,
      "stress output type [cxyz, ndxyz, cxyz_ndxyz, c123, nd123, c123_nd123]");
  CLP.setOption("straintype", &straintype_,
      "strain output type [cxyz, ndxyz, cxyz_ndxyz, c123, nd123, c123_nd123]");
  CLP.setOption("strain", &straintype_,
      "strain output type [cxyz, ndxyz, cxyz_ndxyz, c123, nd123, c123_nd123]");
  CLP.setOption("mortar", &mortar_string, "Do post-processing of mortar interfaces [yes]");
  CLP.setOption("heatfluxtype", &heatfluxtype_,
      "heatflux output type [cxyz, ndxyz, cxyz_ndxyz, c123, nd123, c123_nd123]");
  CLP.setOption("heatflux", &heatfluxtype_,
      "heatflux output type [cxyz, ndxyz, cxyz_ndxyz, c123, nd123, c123_nd123]");
  CLP.setOption("tempgradtype", &tempgradtype_,
      "tempgrad output type [cxyz, ndxyz, cxyz_ndxyz, c123, nd123, c123_nd123]");
  CLP.setOption("tempgrad", &tempgradtype_,
      "tempgrad output type [cxyz, ndxyz, cxyz_ndxyz, c123, nd123, c123_nd123]");
  CLP.setOption("rotation", &struct_rot_, "structural rotation matrix R [yes]");
  CLP.setOption("structmatdisp", &struct_mat_disp_, "material displacement output output [yes]");
  CLP.setOption("outputtype", &outputtype_,
      "binary (bin) or ascii (ascii) output, option works for vtu filter only");
  Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn = CLP.parse(argc, argv);

  if (parseReturn != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
    exit(EXIT_FAILURE);
  }

  if (file == "")
  {
    CLP.printHelpMessage(argv[0], std::cout);
    exit(EXIT_FAILURE);
  }

  if (file.length() <= 8 or file.substr(file.length() - 8, 8) != ".control")
  {
    file += ".control";
  }

  if (output == "")
  {
    output = file.substr(0, file.length() - 8);
  }

  if (stresstype_ == "")
  {
    stresstype_ = "none";
  }

  if (straintype_ == "")
  {
    straintype_ = "none";
  }

  if (heatfluxtype_ == "")
  {
    heatfluxtype_ = "none";
  }

  if (tempgradtype_ == "")
  {
    tempgradtype_ = "none";
  }

  if (outputtype_ == "")
  {
    outputtype_ = "bin";
  }

  if (mortar_string == "yes")
  {
    mortar_ = true;
  }

  result_group_ = std::vector<MAP*>();
  setup_filter(file, output);

  MAP* general_data = map_read_map(&control_table_, "general");
  ndim_ = map_read_int(general_data, "ndim");
  FOUR_C_ASSERT((ndim_ == 1) || (ndim_ == 2) || (ndim_ == 3), "illegal dimension");

  const char* type = map_read_string(general_data, "problem_type");
  const std::string probtype(type);
  problemtype_ = Inpar::PROBLEMTYPE::string_to_problem_type(probtype);

  spatial_approx_ = Core::FE::string_to_shape_function_type(
      map_read_string(general_data, "spatial_approximation"));

  /*--------------------------------------------------------------------*/
  /* collect all result groups */
  SYMBOL* symbol = map_find_symbol(&control_table_, "result");
  while (symbol != nullptr)
  {
    if (!symbol_is_map(symbol))
    {
      FOUR_C_THROW("failed to get result group");
    }
    result_group_.push_back(symbol_map(symbol));
    symbol = symbol->next;
  }

  read_meshes();
}

/*----------------------------------------------------------------------*
 * the Destructor
 *----------------------------------------------------------------------*/
PostProblem::~PostProblem()
{
  destroy_map(&control_table_);
  MPI_Finalize();
}


/*----------------------------------------------------------------------*
 * returns a pointer to the num-th discretization
 *----------------------------------------------------------------------*/
PostField* PostProblem::get_discretization(const int num)
{
  if (num >= static_cast<int>(fields_.size()))
  {
    std::cout << "You asked for discretization " << num
              << " (counting from zero), but there are only " << fields_.size()
              << " discretization(s)!";
    FOUR_C_THROW("This is a bug!");
  }
  return fields_.data() + num;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int PostProblem::field_pos(const PostField* field) const
{
  for (std::vector<PostField>::const_iterator i = fields_.begin(); i != fields_.end(); ++i)
  {
    if (&*i == field)
    {
      return field - fields_.data();
    }
  }
  FOUR_C_THROW("field not in list");
  return -1;
}


/*----------------------------------------------------------------------*
 * returns the MPI communicator object
 *----------------------------------------------------------------------*/
MPI_Comm PostProblem::get_comm() { return comm_; }


/*----------------------------------------------------------------------*
 * initializes all the data a filter needs. This function is called by
 * the Constructor.  (private)
 *----------------------------------------------------------------------*/
void PostProblem::setup_filter(std::string control_file_name, std::string output_name)
{
  MAP temp_table;

  comm_ = MPI_COMM_WORLD;

  /* The warning system is not set up.*/

  basename_ = control_file_name.substr(0, control_file_name.length() - 8);
  outname_ = output_name;

  parse_control_file(&control_table_, control_file_name.c_str(), MPI_COMM_WORLD);

  /*
   * Now that we've read the control file given by the user we have to
   * take care of any previous (restarted) control files. These files
   * build a chain. So as long as a previous file exists we have to
   * open it and read any groups of results with smaller step numbers
   * than the results we've already read. */

  /* The general idea is to merge the different control files in
   * memory. If one step is written several times the last version is
   * used. */

  MAP* table = &control_table_;

  /* copy directory information */
  std::string::size_type separator = basename_.rfind('/', std::string::npos);
  if (separator != std::string::npos)
  {
    input_dir_ = basename_.substr(0, separator + 1);
  }
  else
  {
    input_dir_ = "";
  }

  while (map_symbol_count(table, "restarted_run") > 0)
  {
    /* copy directory information */
    control_file_name = input_dir_;

    /* copy file name */
    control_file_name += map_read_string(table, "restarted_run");
    control_file_name += ".control";

    /* test open to see if it exists */
    FILE* f = fopen(control_file_name.c_str(), "rb");
    if (f == nullptr)
    {
      printf("Restarted control file '%s' does not exist. Skip previous results.\n",
          control_file_name.c_str());
      break;
    }
    fclose(f);

    /* copy all the result steps that are previous to this file */
    /* We assume that the results are ordered! */

    /*------------------------------------------------------------------*/
    /* find the first result in the current table */
    SYMBOL* first_result = map_find_symbol(&control_table_, "result");
    if (first_result == nullptr)
    {
      FOUR_C_THROW("no result sections in control file '{}'\n", control_file_name);
    }
    while (first_result->next != nullptr)
    {
      first_result = first_result->next;
    }
    const int first_step = map_read_int(symbol_map(first_result), "step");


    /*------------------------------------------------------------------*/
    /* done with this control file */
    if (table != &control_table_)
    {
      destroy_map(table);
    }

    /* The first time we reach this place we had just used the main
     * control table. But from now on we are interested in the
     * previous control files we read. */
    table = &temp_table;

    /* read the previous control file */
    parse_control_file(table, control_file_name.c_str(), MPI_COMM_WORLD);
    if (Core::Communication::my_mpi_rank(comm_) == 0)
      printf("read restarted control file: %s\n", control_file_name.c_str());

    /* find the previous results */
    {
      int counter = 0;

      /*
       * the dummy_symbol is a hack that allows us to treat all results
       * in the list the same way (use the same code). Without it we'd
       * need special conditions for the first entry. */
      SYMBOL dummy_symbol;
      SYMBOL* previous_results = &dummy_symbol;
      previous_results->next = map_find_symbol(table, "result");
      while (previous_results->next != nullptr)
      {
        SYMBOL* result;
        int step;
        result = previous_results->next;
        step = map_read_int(symbol_map(result), "step");

        if (step < first_step)
        {
          /* found it */
          /* Now we simply switch all previous results to our main
           * map. The assumption is a perfect ordering */
          map_prepend_symbols(
              &control_table_, "result", result, map_symbol_count(table, "result") - counter);
          previous_results->next = nullptr;

          /*
           * In case all results go to the main map we have to disconnect
           * them explicitly. */
          if (previous_results == &dummy_symbol)
          {
            map_disconnect_symbols(table, "result");
          }
          break;
        }

        /* Not found yet. Go up one result. */
        previous_results = previous_results->next;
        counter += 1;
      }
    }

    /* find the previous mesh files */
    {
      int counter = 0;

      /*
       * the dummy_symbol is a hack that allows us to treat all results
       * in the list the same way (use the same code). Without it we'd
       * need special conditions for the first entry. */
      SYMBOL dummy_symbol;
      SYMBOL* previous_fields = &dummy_symbol;
      previous_fields->next = map_find_symbol(table, "field");
      while (previous_fields->next != nullptr)
      {
        SYMBOL* field;
        int step;
        field = previous_fields->next;
        step = map_read_int(symbol_map(field), "step");

        if (step < first_step)
        {
          /* found it */
          /* Now we simply switch all previous fields to our main
           * map. The assumption is a perfect ordering */
          map_prepend_symbols(
              &control_table_, "field", field, map_symbol_count(table, "field") - counter);
          previous_fields->next = nullptr;

          /*
           * In case all fields go to the main map we have to disconnect
           * them explicitly. */
          if (previous_fields == &dummy_symbol)
          {
            map_disconnect_symbols(table, "field");
          }
          break;
        }

        /* Not found yet. Go up one field. */
        previous_fields = previous_fields->next;
        counter += 1;
      }
    }
  }
}

/*----------------------------------------------------------------------*
 * reads the mesh files and calls 'getfield()' for each 'field'-entry
 * in the mesh file (currently it reads only the fields with step ==
 * 0). This function is called by the Constructor. (private)
 *----------------------------------------------------------------------*/
void PostProblem::read_meshes()
{
  using namespace FourC;

  SYMBOL* mesh = map_find_symbol(&control_table_, "field");
  if (mesh == nullptr) FOUR_C_THROW("No field found.");

  // We have to reverse the traversal of meshes we get from the control file
  // in order to get the same dof numbers in all discretizations as we had
  // during our calculation.
  // The order inside the control file is important!
  // Discretizations have to be fill_complete()ed in the same order as during
  // the calculation!
  std::stack<MAP*> meshstack;

  while (mesh != nullptr)
  {
    // only those fields with a mesh file entry are readable here
    // (each control file is bound to include at least one of those)
    if (map_find_symbol(symbol_map(mesh), "mesh_file") != nullptr)
    {
      meshstack.push(symbol_map(mesh));
    }
    mesh = mesh->next;
  }
  mesh = nullptr;

  while (not meshstack.empty())
  {
    MAP* meshmap = meshstack.top();
    meshstack.pop();

    std::string name = map_read_string(meshmap, "field");

    bool havefield = false;
    for (unsigned i = 0; i < fields_.size(); ++i)
    {
      if (fields_[i].name() == name)
      {
        havefield = true;
        break;
      }
    }

    // only read a field that has not yet been read
    // for now we do not care at which step this field was defined
    // if we want to support changing meshes one day, we'll need to
    // change this code...
    if (not havefield)
    {
      int step;
      if (!map_find_int(meshmap, "step", &step)) FOUR_C_THROW("No step information in field.");

      PostField currfield = getfield(meshmap);

      int num_output_procs;
      if (!map_find_int(meshmap, "num_output_proc", &num_output_procs))
      {
        num_output_procs = 1;
      }
      currfield.set_num_output_procs(num_output_procs);
      const char* fn;
      if (!map_find_string(meshmap, "mesh_file", &fn))
        FOUR_C_THROW("No meshfile name for discretization {}.", currfield.discretization()->name());
      std::string filename = fn;
      Core::IO::HDFReader reader = Core::IO::HDFReader(input_dir_);
      reader.open(filename, num_output_procs, Core::Communication::num_mpi_ranks(comm_),
          Core::Communication::my_mpi_rank(comm_));

      if (currfield.num_nodes() != 0)
      {
        std::shared_ptr<std::vector<char>> node_data = reader.read_node_data(step,
            Core::Communication::num_mpi_ranks(comm_), Core::Communication::my_mpi_rank(comm_));
        currfield.discretization()->unpack_my_nodes(*node_data);
      }

      if (currfield.num_elements() != 0)
      {
        std::shared_ptr<std::vector<char>> element_data = reader.read_element_data(step,
            Core::Communication::num_mpi_ranks(comm_), Core::Communication::my_mpi_rank(comm_));
        currfield.discretization()->unpack_my_elements(*element_data);
      }

      std::shared_ptr<std::vector<char>> cond_pbcsline;
      std::shared_ptr<std::vector<char>> cond_pbcssurf;

      // read knot vectors for nurbs discretisations
      switch (spatial_approx_)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          // try a dynamic cast of the discretisation to a nurbs discretisation
          Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
              dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(*currfield.discretization()));

          if (nurbsdis == nullptr)
            FOUR_C_THROW("discretization {} is not a NurbsDiscretization",
                currfield.discretization()->name());

          std::shared_ptr<std::vector<char>> packed_knots;
          if (Core::Communication::my_mpi_rank(comm_) == 0)
            packed_knots = reader.read_knotvector(step);
          else
            packed_knots = std::make_shared<std::vector<char>>();

          // distribute knots to all procs
          if (Core::Communication::num_mpi_ranks(comm_) > 1)
          {
            Core::Communication::Exporter exporter(nurbsdis->get_comm());

            if (Core::Communication::my_mpi_rank(comm_) == 0)
            {
              MPI_Request request;
              int tag = -1;
              int frompid = 0;
              int topid = -1;

              for (int np = 1; np < Core::Communication::num_mpi_ranks(comm_); ++np)
              {
                tag = np;
                topid = np;

                exporter.i_send(
                    frompid, topid, packed_knots->data(), packed_knots->size(), tag, request);
              }
            }
            else
            {
              int length = -1;
              int frompid = 0;
              int mypid = Core::Communication::my_mpi_rank(comm_);

              std::vector<char> rblock;

              exporter.receive_any(frompid, mypid, rblock, length);

              *packed_knots = rblock;
            }
          }

          std::shared_ptr<Core::FE::Nurbs::Knotvector> knots =
              std::make_shared<Core::FE::Nurbs::Knotvector>();

          Core::Communication::UnpackBuffer knot_buffer(*packed_knots);
          knots->unpack(knot_buffer);

          if (nurbsdis == nullptr)
          {
            FOUR_C_THROW("expected a nurbs discretisation for spatial approx. Nurbs\n");
          }

          if (Core::Communication::num_mpi_ranks(nurbsdis->get_comm()) != 1)
            nurbsdis->setup_ghosting(Core::FE::OptionsFillComplete::none());
          else
            nurbsdis->fill_complete(Core::FE::OptionsFillComplete::none());


          if (!(nurbsdis->filled()))
          {
            FOUR_C_THROW("nurbsdis was not fc\n");
          }

          int smallest_gid_in_dis = nurbsdis->element_row_map()->min_all_gid();

          knots->finish_knots(smallest_gid_in_dis);

          nurbsdis->set_knot_vector(knots);

          // do initialisation
          currfield.discretization()->fill_complete();

          break;
        }
        default:
        {
          // setup of parallel layout: create ghosting of already distributed nodes+elems
          if (Core::Communication::num_mpi_ranks(currfield.discretization()->get_comm()) != 1)
            currfield.discretization()->setup_ghosting();
          else
            currfield.discretization()->fill_complete();

          break;
        }
      }

      // -------------------------------------------------------------------
      // connect degrees of freedom for periodic boundary conditions
      // -------------------------------------------------------------------
      // parallel execution?!
      if ((cond_pbcssurf != nullptr and not cond_pbcssurf->empty()) or
          (cond_pbcsline != nullptr and not cond_pbcsline->empty()))
      {
        Core::Conditions::PeriodicBoundaryConditions pbc(currfield.discretization());
        pbc.update_dofs_for_periodic_boundary_conditions();
      }

      fields_.push_back(currfield);
    }
  }
}

/*----------------------------------------------------------------------*
 * creates and returns a PostField instance from a field MAP. (private)
 *----------------------------------------------------------------------*/
PostField PostProblem::getfield(MAP* field_info)
{
  using namespace FourC;

  const char* field_name = map_read_string(field_info, "field");
  const int numnd = map_read_int(field_info, "num_nd");
  const int numele = map_read_int(field_info, "num_ele");
  const int ndim = map_read_int(field_info, "num_dim");

  std::shared_ptr<Core::FE::Discretization> dis;

  switch (spatial_approx_)
  {
    case Core::FE::ShapeFunctionType::polynomial:
    case Core::FE::ShapeFunctionType::hdg:
    {
      dis = std::make_shared<Core::FE::Discretization>(field_name, comm_, ndim);
      break;
    }
    case Core::FE::ShapeFunctionType::nurbs:
    {
      dis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(field_name, comm_, ndim);
      break;
    }
    default:
    {
      FOUR_C_THROW("Undefined spatial approximation type.\n");
      break;
    }
  }

  return PostField(dis, this, field_name, numnd, numele);
}

/*--------------------------------------------------------------------------*
 * loop all fields and get maximum node id for given field name ghamm 03/13
 *--------------------------------------------------------------------------*/
int PostProblem::get_max_nodeid(const std::string& fieldname)
{
  SYMBOL* mesh = map_find_symbol(&control_table_, "field");

  if (mesh == nullptr) FOUR_C_THROW("No field found.");

  int maxnodeid = -1;
  while (mesh != nullptr)
  {
    MAP* meshmap = symbol_map(mesh);
    mesh = mesh->next;

    std::string name = map_read_string(meshmap, "field");

    if (name == fieldname)
    {
      const int stepmaxnodeid = map_read_int(meshmap, "max_nodeid");
      if (stepmaxnodeid > maxnodeid) maxnodeid = stepmaxnodeid;
    }
  }

  return maxnodeid;
}

int PostProblem::num_discr() { return static_cast<int>(fields_.size()); }


/*----------------------------------------------------------------------*
 * Constructor of PostField.
 *----------------------------------------------------------------------*/
PostField::PostField(std::shared_ptr<Core::FE::Discretization> dis, PostProblem* problem,
    std::string field_name, const int numnd, const int numele)
    : dis_(dis), problem_(problem), field_name_(field_name), numnd_(numnd), numele_(numele)
{
}



/*----------------------------------------------------------------------*
 * The Constructor of PostResult
 *----------------------------------------------------------------------*/
PostResult::PostResult(PostField* field)
    : field_(field), pos_(-1), group_(nullptr), file_((field->problem()->input_dir()))
{
}


/*----------------------------------------------------------------------*
 * The Destructor of PostResult
 *----------------------------------------------------------------------*/
PostResult::~PostResult() { close_result_files(); }

/*----------------------------------------------------------------------*
 * get timesteps when the solution is written
 *----------------------------------------------------------------------*/
std::vector<double> PostResult::get_result_times(const std::string& fieldname)
{
  std::vector<double> times;  // timesteps when the solution is written

  while (this->next_result()) times.push_back(this->time());

  if (times.size() == 0)
  {
    FOUR_C_THROW(
        "PostResult::get_result_times(fieldname='{}'):\n  no solution steps found in specified "
        "timestep range! Check --start, --end, --step parameters.",
        fieldname.c_str());
  }

  return times;
}

/*----------------------------------------------------------------------*
 * get timesteps when the specific solution vector >name< is written
 *                                                               gjb02/08
 *----------------------------------------------------------------------*/
std::vector<double> PostResult::get_result_times(
    const std::string& fieldname, const std::string& groupname)
{
  std::vector<double> times;  // timesteps when the solution is written

  if (this->next_result(groupname))
    times.push_back(this->time());
  else
    FOUR_C_THROW("no solution found in field '{}'", fieldname);

  while (this->next_result(groupname)) times.push_back(this->time());

  if (times.size() == 0)
  {
    FOUR_C_THROW(
        "PostResult::get_result_times(fieldname='{}', groupname='{}'):\n  no solution steps found "
        "in specified timestep range! Check --start, --end, --step parameters.",
        fieldname.c_str(), groupname.c_str());
  }

  return times;
}

/*----------------------------------------------------------------------*
 * get times and steps when the solution is written
 *----------------------------------------------------------------------*/
void PostResult::get_result_timesandsteps(
    const std::string& fieldname, std::vector<double>& times, std::vector<int>& steps)
{
  while (this->next_result())
  {
    times.push_back(this->time());
    steps.push_back(this->step());
  }

  if (times.size() == 0 or steps.size() == 0)
  {
    FOUR_C_THROW(
        "PostResult::get_result_timesandsteps(fieldname='{}'):\n  no solution steps found in "
        "specified range! Check --start, --end, --step parameters.",
        fieldname.c_str());
  }

  return;
}

/*----------------------------------------------------------------------*
 * loads the next result block and opens new result files if there are
 * any. Returns 1 when a new result block has been found, otherwise
 * returns 0
 *----------------------------------------------------------------------*/
int PostResult::next_result()
{
  PostProblem* problem = field_->problem();
  int ret = 0;

  for (int i = pos_ + 1; i < problem->num_results(); ++i)
  {
    MAP* map = (*problem->result_groups())[problem->num_results() - 1 - i];

    if (match_field_result(map))
    {
      /*
       * Open the new files if there are any.
       *
       * If one of these files is here the other one has to be
       * here, too. If it's not, it's a bug in the input. */
      if ((map_symbol_count(map, "result_file") > 0))
      {
        close_result_files();
        open_result_files(map);
      }

      /*
       * We use the real step numbers here. That is a user has to give
       * the real numbers, too. Maybe that's the best way to handle
       * it. */
      /* In case of FSI everything else hurts even more. */
      const int step = map_read_int(map, "step");

      /* we are only interested if the result matches the slice */
      if ((step >= problem->start()) && ((step <= problem->end()) || (problem->end() == -1)) &&
          ((step - problem->start()) % problem->step() == 0))
      {
        pos_ = i;
        group_ = map;
        ret = 1;
        break;
      }
    }
  }
  return ret;
}


/*----------------------------------------------------------------------*
 * loads the next result block that contains written result values
 * specified by a given groupname. Returns 1 when a new result block has
 * been found, otherwise returns 0                              gjb 02/08
 *----------------------------------------------------------------------*/
int PostResult::next_result(const std::string& groupname)
{
  int ret = next_result();
  // go on, until the specified result is contained or end of time slice reached
  while ((!map_has_map(group_, groupname.c_str())) && (ret == 1))
  {
    ret = next_result();
  }
  return ret;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Tell whether a given result group belongs to this result.

*/
/*----------------------------------------------------------------------*/
int PostResult::match_field_result(MAP* result_group) const
{
  return field_->name() == map_read_string(result_group, "field");
}

/*----------------------------------------------------------------------*
 * closes all the currently open result files
 *----------------------------------------------------------------------*/
void PostResult::close_result_files() { file_.close(); }

/*----------------------------------------------------------------------*
 * opens result files. The name is taken from the "result_file" entry
 * in the block 'field_info'
 *----------------------------------------------------------------------*/
void PostResult::open_result_files(MAP* field_info)
{
  int num_output_procs;
  if (!map_find_int(field_info, "num_output_proc", &num_output_procs))
  {
    num_output_procs = 1;
  }
  const std::string basename = map_read_string(field_info, "result_file");
  // field_->problem()->set_basename(basename);
  auto comm = field_->problem()->get_comm();
  file_.open(basename, num_output_procs, Core::Communication::num_mpi_ranks(comm),
      Core::Communication::my_mpi_rank(comm));
}

/*----------------------------------------------------------------------*
 * reads the data of the result vector 'name' from the current result
 * block and returns it as an vector.
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> PostResult::read_result(const std::string name)
{
  MAP* result = map_read_map(group_, name.c_str());
  int columns;
  if (map_find_int(result, "columns", &columns))
  {
    if (columns != 1) FOUR_C_THROW("got multivector with name '{}', vector expected", name);
  }
  auto test = read_multi_result(name);
  return std::make_shared<Core::LinAlg::Vector<double>>((*test)(0));
}

/*----------------------------------------------------------------------*
 * reads the data of the result vector 'name' from the current result
 * block and returns it as an std::vector<char>. the corresponding
 * elemap is returned, too.
 *----------------------------------------------------------------------*/
std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>
PostResult::read_result_serialdensematrix(const std::string name)
{
  using namespace FourC;

  MPI_Comm comm = field_->problem()->get_comm();
  MAP* result = map_read_map(group_, name.c_str());
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
      file_.read_result_data_vec_char(id_path, value_path, columns, comm, elemap);

  std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>> mapdata =
      std::make_shared<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>();

  Core::Communication::UnpackBuffer data_buffer(*data);
  for (int i = 0; i < elemap->num_my_elements(); ++i)
  {
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> gpstress =
        std::make_shared<Core::LinAlg::SerialDenseMatrix>();
    extract_from_pack(data_buffer, *gpstress);
    (*mapdata)[elemap->gid(i)] = gpstress;
  }

  const Core::LinAlg::Map& elecolmap = *field_->discretization()->element_col_map();
  Core::Communication::Exporter ex(*elemap, elecolmap, comm);
  ex.do_export(*mapdata);

  return mapdata;
}

/*----------------------------------------------------------------------*
 * reads the data of the result vector 'name' from the current result
 * block and returns it as an multi vector.
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> PostResult::read_multi_result(
    const std::string name)
{
  const MPI_Comm comm = field_->problem()->get_comm();
  MAP* result = map_read_map(group_, name.c_str());
  const std::string id_path = map_read_string(result, "ids");
  const std::string value_path = map_read_string(result, "values");
  int columns;
  if (not map_find_int(result, "columns", &columns))
  {
    columns = 1;
  }
  return file_.read_result_data(id_path, value_path, columns, comm);
}

//! returns time of this result
double PostResult::time() const { return map_read_real(group_, "time"); }

//! returns step number of this result
int PostResult::step() const { return map_read_int(group_, "step"); }



//! returns the number of global Dof-Ids
int PostField::global_id_num() const { return dis_->dof_row_map()->num_global_elements(); }

FOUR_C_NAMESPACE_CLOSE

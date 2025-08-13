// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_meshreader.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element_definition.hpp"
#include "4C_fem_general_fiber_node.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_nurbs_discretization_control_point.hpp"
#include "4C_io_exodus.hpp"
#include "4C_io_gridgenerator.hpp"
#include "4C_io_input_file.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_rebalance.hpp"
#include "4C_rebalance_graph_based.hpp"
#include "4C_rebalance_print.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <string>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO::Internal
{
  /**
   * Internal support class to read a mesh from an exodus file.
   */
  struct ExodusReader
  {
    /**
     *The discretization that should be filled with the information from the exodus file.
     */
    Core::FE::Discretization& target_discretization;

    /**
     * The section in the input file that has the necessary data for the reader (e.g. the file
     * name).
     */
    std::string section_name;

    /**
     * The actual exodus mesh object. This is only created on rank 0.
     */
    std::shared_ptr<Exodus::Mesh> mesh_on_rank_zero{};
  };
}  // namespace Core::IO::Internal

namespace
{

  struct CellInfo
  {
    Core::FE::CellType cell_type;
    std::string cell_type_str;
    std::vector<int> nodal_ids;
  };

  //! Get a primitive cell type with number of nodes + the nodal ids from the parser.
  //! e.g. reads a line like `HEX8 1 2 3 4 5 6 7 8`
  void read_cell_info(Core::IO::ValueParser& parser, CellInfo& cell_info)
  {
    cell_info.cell_type_str = parser.read<std::string>();
    cell_info.cell_type = Core::FE::string_to_cell_type(cell_info.cell_type_str);

    const auto num_nodes = Core::FE::num_nodes(cell_info.cell_type);

    // Read the nodal ids
    cell_info.nodal_ids = parser.read<std::vector<int>>(num_nodes);
  }

  class ElementReader
  {
   public:
    /*!
    \brief Construct element reader for a given field that reads a given section

    Create empty discretization and append it to given field.

    \param dis (i) the new discretization
    \param comm (i) our communicator
    \param sectionname (i) the section that contains the element lines
    */
    ElementReader(std::shared_ptr<Core::FE::Discretization> dis, const Core::IO::InputFile& input,
        std::string sectionname);

    /// give the discretization this reader fills
    std::shared_ptr<Core::FE::Discretization> get_dis() const { return dis_; }

    /// Return the list of row elements
    std::shared_ptr<Core::LinAlg::Map> get_row_elements() const { return roweles_; }

    /*! Read elements and partition the node graph

    - read global ids of elements of this discretization
      (this is one fully redundant vector for elements)
    - determine a preliminary element distribution. The fully redundant
      vector is trashed after the construction.
    - define blocksizes for blocks of elements we read (not necessarily
      the same as it was used to construct the map --- we may have a
      smaller blocksize here).
    - read elements of this discretization and distribute according
      to a linear map. While reading, remember node gids and assemble
      them into a second fully redundant vector (mapping node id->gid).
      In addition, we keep them in a fully redundant set (required by
      node reader). Construct reverse lookup from gids to node ids.
      Again, this is a global, fully redundant map!
    - define preliminary linear distributed nodal row map
    - determine adjacency array (i.e. the infos for the node graph)
      using the nodal row distribution and a round robin communication
      of element connectivity information.
      Use adjacency array to build an initial Crsgraph on the linear map.
    - do partitioning using parmetis
      Results are distributed to other procs using two global vectors!
    - build final nodal row map, export graph to the new map
    */
    void read_and_distribute();

    /*!
    \brief Tell whether the given node belongs to us

    \note This is based on the redundant nodes_ set and only available on processor 0.
    */
    bool has_node(const int nodeid) const { return nodes_.find(nodeid) != nodes_.end(); }

   private:
    /// Get the overall number of elements and their corresponding global IDs
    std::vector<int> get_element_size_and_ids() const;

    /// Read the file and get element information, distribute them to each processor
    void get_and_distribute_elements(const int nblock, const int bsize);

    /// discretization name
    std::string name_;

    /// the main input file reader
    const Core::IO::InputFile& input_;

    /// my comm
    MPI_Comm comm_;

    /// my section to read
    std::string sectionname_;

    /*!
    \brief All global node ids of a discretization on processor 0

    This is a redundant set of all node numbers. But it is only valid
    on processor 0. We need it to easily figure out to which
    discretization a node belongs.
    */
    std::set<int> nodes_;

    /// my discretization
    std::shared_ptr<Core::FE::Discretization> dis_;

    /// element row map
    std::shared_ptr<Core::LinAlg::Map> roweles_;
  };


  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  ElementReader::ElementReader(std::shared_ptr<Core::FE::Discretization> dis,
      const Core::IO::InputFile& input, std::string sectionname)
      : name_(dis->name()),
        input_(input),
        comm_(dis->get_comm()),
        sectionname_(sectionname),
        dis_(dis)
  {
  }


  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void ElementReader::read_and_distribute()
  {
    const int myrank = Core::Communication::my_mpi_rank(comm_);
    const int numproc = Core::Communication::num_mpi_ranks(comm_);

    const auto& eids = get_element_size_and_ids();

    if (eids.empty())
    {
      // If the element section is empty, we create an empty input and return
      roweles_ = std::make_shared<Core::LinAlg::Map>(-1, 0, nullptr, 0, comm_);

      return;
    }

    // determine a preliminary element distribution
    int nblock, mysize, bsize;
    const int numele = static_cast<int>(eids.size());
    {
      // number of element chunks to split the reading process in
      // approximate block size (just a guess!)
      nblock = numproc;
      bsize = numele / nblock;

      // create a simple (pseudo linear) map
      mysize = bsize;
      if (myrank == numproc - 1) mysize = numele - (numproc - 1) * bsize;

      // construct the map
      roweles_ = std::make_shared<Core::LinAlg::Map>(-1, mysize, &eids[myrank * bsize], 0, comm_);
    }

    // define blocksizes for blocks of elements we read
    {
      // for block sizes larger than about 250000 elements (empirical value !) the code sometimes
      // hangs during ExportRowElements call for the second block (block 1). Therefore an upper
      // limit of 100000 for bsize is ensured below.
      const int maxblocksize = 100000;

      if (bsize > maxblocksize)
      {
        // without an additional increase of nblock by 1 the last block size
        // could reach a maximum value of (2*maxblocksize)-1, potentially
        // violating the intended upper limit!
        nblock = 1 + numele / maxblocksize;
        bsize = maxblocksize;
      }
    }

    get_and_distribute_elements(nblock, bsize);
  }


  std::vector<int> ElementReader::get_element_size_and_ids() const
  {
    // vector of all global element ids
    std::vector<int> eids;

    // all reading is done on proc 0
    if (Core::Communication::my_mpi_rank(comm_) == 0)
    {
      for (const auto& element_line : input_.in_section_rank_0_only(sectionname_))
      {
        std::istringstream t{std::string{element_line.get_as_dat_style_string()}};
        int elenumber;
        std::string eletype;
        t >> elenumber >> eletype;
        elenumber -= 1;

        // only read registered element types or all elements if nothing is registered
        eids.push_back(elenumber);
      }
    }

    Core::Communication::broadcast(eids, 0, comm_);

    return eids;
  }


  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void ElementReader::get_and_distribute_elements(const int nblock, const int bsize)
  {
    Core::Elements::ElementDefinition ed;

    // All ranks > 0 will receive the node ids of the elements from rank 0.
    // We know that we will read nblock blocks of elements, so call the
    // collective function an appropriate number of times.
    if (Core::Communication::my_mpi_rank(comm_) > 0)
    {
      for (int i = 0; i < nblock; ++i)
      {
        std::vector<int> gidlist;
        dis_->proc_zero_distribute_elements_to_all(*roweles_, gidlist);
      }
    }
    // Rank 0 does the actual work
    else
    {
      std::vector<int> gidlist;
      gidlist.reserve(bsize);
      int bcount = 0;
      int block = 0;

      CellInfo cell_info;
      for (const auto& element_line : input_.in_section_rank_0_only(sectionname_))
      {
        Core::IO::ValueParser parser{element_line.get_as_dat_style_string(),
            {.user_scope_message = "While reading element line: "}};
        const int elenumber = parser.read<int>() - 1;
        gidlist.push_back(elenumber);

        const auto eletype = parser.read<std::string>();
        read_cell_info(parser, cell_info);

        // let the factory create a matching empty element
        std::shared_ptr<Core::Elements::Element> ele =
            Core::Communication::factory(eletype, cell_info.cell_type_str, elenumber, 0);
        if (!ele) FOUR_C_THROW("element creation failed");

        const auto& linedef = ed.get(eletype, cell_info.cell_type);

        Core::IO::InputParameterContainer data;
        linedef.fully_parse(parser, data);

        ele->set_node_ids_one_based_index(cell_info.nodal_ids);
        ele->read_element(eletype, cell_info.cell_type_str, data);

        // add element to discretization
        dis_->add_element(ele);

        // get the node ids of this element
        const int numnode = ele->num_node();
        const int* nodeids = ele->node_ids();

        // all node gids of this element are inserted into a set of
        // node ids --- it will be used later during reading of nodes
        // to add the node to one or more discretisations
        std::copy(nodeids, nodeids + numnode, std::inserter(nodes_, nodes_.begin()));

        ++bcount;

        // Distribute the block if it is full. Never distribute the last block here because it
        // could be longer than expected and is therefore always distributed at the end.
        if (block != nblock - 1 && bcount == bsize)
        {
          dis_->proc_zero_distribute_elements_to_all(*roweles_, gidlist);
          gidlist.clear();
          bcount = 0;
          ++block;
        }
      }

      // Ensure that the last block is distributed. Since the loop might abort a lot earlier
      // than expected by the number of blocks, make sure to call the collective function
      // the appropriate number of times to match the action of the other ranks.
      for (; block < nblock; ++block)
      {
        dis_->proc_zero_distribute_elements_to_all(*roweles_, gidlist);
        gidlist.clear();
      }
    }
  }

  struct DomainReader
  {
    Core::FE::Discretization& target_discretization;

    std::string section_name;
  };


  std::pair<std::shared_ptr<Core::LinAlg::Map>, std::shared_ptr<Core::LinAlg::Map>>
  do_rebalance_discretization(const std::shared_ptr<const Core::LinAlg::Graph>& graph,
      Core::FE::Discretization& discret, Core::Rebalance::RebalanceType rebalanceMethod,
      Teuchos::ParameterList& rebalanceParams,
      const Core::IO::MeshReader::MeshReaderParameters& parameters, MPI_Comm comm)
  {
    std::shared_ptr<Core::LinAlg::Map> rowmap, colmap;

    switch (rebalanceMethod)
    {
      case Core::Rebalance::RebalanceType::hypergraph:
      {
        if (!Core::Communication::my_mpi_rank(comm))
          std::cout << "Redistributing using hypergraph .........\n";

        rebalanceParams.set("partitioning method", "HYPERGRAPH");
        std::tie(rowmap, colmap) = Core::Rebalance::rebalance_node_maps(*graph, rebalanceParams);
        break;
      }
      case Core::Rebalance::RebalanceType::recursive_coordinate_bisection:
      {
        if (!Core::Communication::my_mpi_rank(comm))
          std::cout << "Redistributing using recursive coordinate bisection .........\n";

        rebalanceParams.set("partitioning method", "RCB");

        rowmap = std::make_shared<Core::LinAlg::Map>(
            -1, graph->row_map().num_my_elements(), graph->row_map().my_global_elements(), 0, comm);
        colmap = std::make_shared<Core::LinAlg::Map>(
            -1, graph->col_map().num_my_elements(), graph->col_map().my_global_elements(), 0, comm);

        discret.redistribute(*rowmap, *colmap,
            {.assign_degrees_of_freedom = false,
                .init_elements = false,
                .do_boundary_conditions = false});

        std::shared_ptr<Core::LinAlg::MultiVector<double>> coordinates =
            discret.build_node_coordinates();

        std::tie(rowmap, colmap) = Core::Rebalance::rebalance_node_maps(
            *graph, rebalanceParams, nullptr, nullptr, coordinates);
        break;
      }
      case Core::Rebalance::RebalanceType::monolithic:
      {
        if (!Core::Communication::my_mpi_rank(comm))
          std::cout << "Redistributing using monolithic hypergraph .........\n";

        rebalanceParams.set("partitioning method", "HYPERGRAPH");

        rowmap = std::make_shared<Core::LinAlg::Map>(
            -1, graph->row_map().num_my_elements(), graph->row_map().my_global_elements(), 0, comm);
        colmap = std::make_shared<Core::LinAlg::Map>(
            -1, graph->col_map().num_my_elements(), graph->col_map().my_global_elements(), 0, comm);

        discret.redistribute(*rowmap, *colmap, {.do_boundary_conditions = false});

        std::shared_ptr<const Core::LinAlg::Graph> enriched_graph =
            Core::Rebalance::build_monolithic_node_graph(
                discret, Core::GeometricSearch::GeometricSearchParams(
                             parameters.geometric_search_parameters, parameters.io_parameters));

        std::tie(rowmap, colmap) =
            Core::Rebalance::rebalance_node_maps(*enriched_graph, rebalanceParams);
        break;
      }
      default:
        FOUR_C_THROW("Appropriate partitioning has to be set!");
    }

    return {rowmap, colmap};
  }

  void rebalance_discretization(Core::FE::Discretization& discret,
      const Core::LinAlg::Map& row_elements,
      const Core::IO::MeshReader::MeshReaderParameters& parameters, MPI_Comm comm)
  {
    std::shared_ptr<const Core::LinAlg::Graph> graph = nullptr;

    // Skip building the node graph if there are no elements
    if (row_elements.num_global_elements() > 0)
      graph = Core::Rebalance::build_graph(discret, row_elements);

    // Create partitioning parameters
    const double imbalance_tol =
        parameters.mesh_partitioning_parameters.get<double>("IMBALANCE_TOL");

    Teuchos::ParameterList rebalanceParams;
    rebalanceParams.set<std::string>("imbalance tol", std::to_string(imbalance_tol));

    const int minele_per_proc =
        parameters.mesh_partitioning_parameters.get<int>("MIN_ELE_PER_PROC");
    const int max_global_procs = Core::Communication::num_mpi_ranks(comm);
    int min_global_procs = max_global_procs;

    if (minele_per_proc > 0)
      min_global_procs = row_elements.num_global_elements() / minele_per_proc;
    const int num_procs = std::min(max_global_procs, min_global_procs);
    rebalanceParams.set<std::string>("num parts", std::to_string(num_procs));

    const auto rebalanceMethod = Teuchos::getIntegralValue<Core::Rebalance::RebalanceType>(
        parameters.mesh_partitioning_parameters, "METHOD");

    if (!Core::Communication::my_mpi_rank(comm))
      std::cout << "\nNumber of procs used for redistribution: " << num_procs << "\n";

    std::shared_ptr<Core::LinAlg::Map> rowmap, colmap;

    if (graph)
    {
      std::tie(rowmap, colmap) = do_rebalance_discretization(
          graph, discret, rebalanceMethod, rebalanceParams, parameters, comm);
    }
    else
    {
      rowmap = colmap = std::make_shared<Core::LinAlg::Map>(-1, 0, nullptr, 0, comm);
    }

    auto options_redistribution = Core::FE::OptionsRedistribution();
    if (rebalanceMethod == Core::Rebalance::RebalanceType::monolithic)
      options_redistribution.do_extended_ghosting = true;

    options_redistribution.assign_degrees_of_freedom = false;
    options_redistribution.init_elements = false;
    options_redistribution.do_boundary_conditions = false;

    discret.redistribute(*rowmap, *colmap, options_redistribution);

    Core::Rebalance::Utils::print_parallel_distribution(discret);
  }

  std::vector<std::shared_ptr<Core::FE::Discretization>> find_dis_node(
      const std::vector<ElementReader>& element_readers, int global_node_id)
  {
    std::vector<std::shared_ptr<Core::FE::Discretization>> list_of_discretizations;
    for (const auto& element_reader : element_readers)
      if (element_reader.has_node(global_node_id))
        list_of_discretizations.emplace_back(element_reader.get_dis());

    return list_of_discretizations;
  }

  void read_nodes(const Core::IO::InputFile& input, const std::string& node_section_name,
      std::vector<ElementReader>& element_readers, int& max_node_id)
  {
    const int myrank = Core::Communication::my_mpi_rank(input.get_comm());
    if (myrank > 0) return;

    int line_count = 0;
    for (const auto& node_line : input.in_section_rank_0_only(node_section_name))
    {
      Core::IO::ValueParser parser{
          node_line.get_as_dat_style_string(), {.user_scope_message = "While reading node data: "}};
      auto type = parser.read<std::string>();

      if (type == "NODE")
      {
        int nodeid = parser.read<int>() - 1;
        parser.consume("COORD");
        auto coords = parser.read<std::vector<double>>(3);

        max_node_id = std::max(max_node_id, nodeid) + 1;
        std::vector<std::shared_ptr<Core::FE::Discretization>> dis =
            find_dis_node(element_readers, nodeid);

        for (const auto& di : dis)
        {
          // create node and add to discretization
          std::shared_ptr<Core::Nodes::Node> node =
              std::make_shared<Core::Nodes::Node>(nodeid, coords, myrank);
          di->add_node(node);
        }
      }
      // this node is a Nurbs control point
      else if (type == "CP")
      {
        int cpid = parser.read<int>() - 1;
        parser.consume("COORD");
        auto coords = parser.read<std::vector<double>>(3);
        double weight = parser.read<double>();

        max_node_id = std::max(max_node_id, cpid) + 1;
        if (cpid != line_count)
          FOUR_C_THROW(
              "Reading of control points {} failed: They must be numbered consecutive!!", cpid);
        std::vector<std::shared_ptr<Core::FE::Discretization>> diss =
            find_dis_node(element_readers, cpid);

        for (auto& dis : diss)
        {
          // create node/control point and add to discretization
          std::shared_ptr<Core::FE::Nurbs::ControlPoint> node =
              std::make_shared<Core::FE::Nurbs::ControlPoint>(cpid, coords, weight, myrank);
          dis->add_node(node);
        }
      }
      // this is a special node with additional fiber information
      else if (type == "FNODE")
      {
        enum class FiberType
        {
          Unknown,
          Angle,
          Fiber,
          CosyDirection
        };

        // read fiber node
        std::map<Core::Nodes::CoordinateSystemDirection, std::array<double, 3>> cosyDirections;
        std::vector<std::array<double, 3>> fibers;
        std::map<Core::Nodes::AngleType, double> angles;

        int nodeid = parser.read<int>() - 1;
        parser.consume("COORD");
        auto coords = parser.read<std::vector<double>>(3);
        max_node_id = std::max(max_node_id, nodeid) + 1;

        while (!parser.at_end())
        {
          auto next = parser.read<std::string>();

          if (next == "FIBER" + std::to_string(1 + fibers.size()))
          {
            fibers.emplace_back(parser.read<std::array<double, 3>>());
          }
          else if (next == "CIR")
          {
            cosyDirections[Core::Nodes::CoordinateSystemDirection::Circular] =
                parser.read<std::array<double, 3>>();
          }
          else if (next == "TAN")
          {
            cosyDirections[Core::Nodes::CoordinateSystemDirection::Tangential] =
                parser.read<std::array<double, 3>>();
          }
          else if (next == "RAD")
          {
            cosyDirections[Core::Nodes::CoordinateSystemDirection::Radial] =
                parser.read<std::array<double, 3>>();
          }
          else if (next == "HELIX")
          {
            angles[Core::Nodes::AngleType::Helix] = parser.read<double>();
          }
          else if (next == "TRANS")
          {
            angles[Core::Nodes::AngleType::Transverse] = parser.read<double>();
          }
        }

        // add fiber information to node
        std::vector<std::shared_ptr<Core::FE::Discretization>> discretizations =
            find_dis_node(element_readers, nodeid);
        for (auto& dis : discretizations)
        {
          auto node = std::make_shared<Core::Nodes::FiberNode>(
              nodeid, coords, cosyDirections, fibers, angles, myrank);
          dis->add_node(node);
        }
      }
      else
        FOUR_C_THROW("Unknown node type '{}'", type);

      ++line_count;
    }
  }

  void generate_mesh(
      const Core::IO::InputFile& input, const DomainReader& domain_reader, int& node_count)
  {
    const int myrank = Core::Communication::my_mpi_rank(input.get_comm());
    auto& target_discretization = domain_reader.target_discretization;

    {
      Teuchos::Time time("", true);

      if (myrank == 0)
        Core::IO::cout << "Entering domain generation mode for " << target_discretization.name()
                       << " discretization ...\nCreate and partition elements      in...."
                       << Core::IO::endl;

      Core::IO::InputParameterContainer container;
      input.match_section(domain_reader.section_name, container);
      auto inputData = Core::IO::GridGenerator::RectangularCuboidInputs::from_input(
          container.group(domain_reader.section_name));
      inputData.node_gid_of_first_new_node_ = node_count;

      Core::IO::GridGenerator::create_rectangular_cuboid_discretization(
          target_discretization, inputData, false);

      if (!myrank)
        Core::IO::cout << "............................................... " << std::setw(10)
                       << std::setprecision(5) << std::scientific << time.totalElapsedTime(true)
                       << " secs" << Core::IO::endl;
    }

    {
      Teuchos::Time time("", true);

      if (!myrank)
        Core::IO::cout << "Complete discretization " << std::left << std::setw(16)
                       << target_discretization.name() << " in...." << Core::IO::flush;

      int err = target_discretization.fill_complete(false, false, false);
      if (err) FOUR_C_THROW("dis_->fill_complete() returned {}", err);

      if (!myrank) Core::IO::cout << time.totalElapsedTime(true) << " secs" << Core::IO::endl;

      Core::Rebalance::Utils::print_parallel_distribution(target_discretization);
    }
  }

  void read_mesh_from_exodus(const Core::IO::InputFile& input,
      Core::IO::Internal::ExodusReader& exodus_reader,
      const Core::IO::MeshReader::MeshReaderParameters& parameters, int& ele_count, MPI_Comm comm)
  {
    TEUCHOS_FUNC_TIME_MONITOR("Core::IO::MeshReader::read_mesh_from_exodus");
    auto my_rank = Core::Communication::my_mpi_rank(comm);

    // We cannot create the map right away. First, we need to figure out how many elements there
    // are. Since the code is rather different on rank 0 and other ranks, we will set this pointer
    // to nullptr and create it later.
    std::unique_ptr<Core::LinAlg::Map> linear_element_map;

    // All the work is done on rank 0. The other ranks will receive the data.
    if (my_rank == 0)
    {
      // Initial implementation:
      // - read all information on rank 0, construct discretization, rebalance afterwards

      FOUR_C_ASSERT(exodus_reader.mesh_on_rank_zero != nullptr, "Internal error.");
      const auto& mesh = *exodus_reader.mesh_on_rank_zero;

      Core::IO::InputParameterContainer data;
      input.match_section(exodus_reader.section_name, data);

      const auto& geometry_data = data.group(exodus_reader.section_name);
      const auto& element_block_data = geometry_data.get_list("ELEMENT_BLOCKS");

      Core::Elements::ElementDefinition element_definition;

      std::vector<int> skipped_blocks;
      int ele_count_before = ele_count;
      for (const auto& [eb_id, eb] : mesh.get_element_blocks())
      {
        // Look into the input file to find out which elements we need to assign to this block.
        const int eb_id_copy = eb_id;  // work around compiler warning in clang18
        auto current_block_data = std::ranges::find_if(element_block_data,
            [eb_id_copy](const auto& e) { return e.template get<int>("ID") == eb_id_copy; });
        if (current_block_data == element_block_data.end())
        {
          skipped_blocks.emplace_back(eb_id);
          continue;
        }

        const auto [element_name, cell_type, specific_data] =
            element_definition.unpack_element_data(*current_block_data);
        const std::string cell_type_string = Core::FE::cell_type_to_string(eb.cell_type);

        FOUR_C_ASSERT_ALWAYS(cell_type == eb.cell_type,
            "Element block '{}' has cell type '{}' but your given element definition for '{}' has "
            "cell type '{}'.",
            eb_id, eb.cell_type, element_name, cell_type);

        for (const auto& ele_nodes : eb.elements | std::views::values)
        {
          auto ele = Core::Communication::factory(element_name, cell_type_string, ele_count, 0);
          if (!ele) FOUR_C_THROW("element creation failed");
          ele->set_node_ids(ele_nodes.size(), ele_nodes.data());
          ele->read_element(element_name, cell_type_string, specific_data);
          exodus_reader.target_discretization.add_element(ele);

          ele_count++;
        }
      }

      int num_read_ele = ele_count - ele_count_before;
      FOUR_C_ASSERT_ALWAYS(num_read_ele > 0,
          "No element block of the mesh was used. This does not make any sense. "
          "If you supply an Exodus mesh file, you need to use at least one of its blocks.");

      int first_ele_id = ele_count_before;
      Core::Communication::broadcast(num_read_ele, 0, comm);
      Core::Communication::broadcast(first_ele_id, 0, comm);
      linear_element_map =
          std::make_unique<Core::LinAlg::Map>(num_read_ele, ele_count_before, comm);

      std::vector<int> gid_list(num_read_ele);
      std::iota(gid_list.begin(), gid_list.end(), ele_count_before);
      exodus_reader.target_discretization.proc_zero_distribute_elements_to_all(
          *linear_element_map, gid_list);

      // Now add all the nodes to the discretization on rank 0. They are distributed later during
      // the rebalancing process.
      for (const auto& [id, coords] : mesh.get_nodes())
      {
        auto node = std::make_shared<Core::Nodes::Node>(id, coords, 0);
        exodus_reader.target_discretization.add_node(node);
      }
    }
    // Other ranks
    else
    {
      int num_read_ele;
      int first_ele_id;
      Core::Communication::broadcast(num_read_ele, 0, comm);
      Core::Communication::broadcast(first_ele_id, 0, comm);
      linear_element_map = std::make_unique<Core::LinAlg::Map>(num_read_ele, first_ele_id, comm);

      std::vector<int> gid_list;
      exodus_reader.target_discretization.proc_zero_distribute_elements_to_all(
          *linear_element_map, gid_list);
    }

    FOUR_C_ASSERT(linear_element_map, "Internal error: nullptr.");
    rebalance_discretization(
        exodus_reader.target_discretization, *linear_element_map, parameters, comm);
  }
}  // namespace

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::MeshReader::MeshReader(const Core::IO::InputFile& input, MeshReaderParameters parameters)
    : comm_(input.get_comm()), input_(input), parameters_(std::move(parameters))
{
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::MeshReader::attach_discretization(
    std::shared_ptr<Core::FE::Discretization> dis, const std::string& section_prefix)
{
  target_discretizations_.emplace_back(section_prefix, dis);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::MeshReader::read_and_partition()
{
  // We need to track the max global node ID to offset node numbering and for sanity checks
  int max_node_id = 0;

  std::vector<ElementReader> element_readers;
  std::vector<DomainReader> domain_readers;

  for (const auto& [section_name, dis] : target_discretizations_)
  {
    // Find out which section we have available for input. We can only do this on rank zero due
    // to large legacy sections that are not available everywhere. Communicate the result to all
    // ranks.
    std::map<std::string, bool> available_section;
    const int my_rank = Core::Communication::my_mpi_rank(comm_);
    if (my_rank == 0)
    {
      available_section[section_name + " ELEMENTS"] =
          input_.has_section(section_name + " ELEMENTS");
      available_section[section_name + " DOMAIN"] = input_.has_section(section_name + " DOMAIN");
      available_section[section_name + " GEOMETRY"] =
          input_.has_section(section_name + " GEOMETRY");
      Core::Communication::broadcast(available_section, 0, comm_);
    }
    else
    {
      Core::Communication::broadcast(available_section, 0, comm_);
    }

    const int num_sections_in_file =
        std::ranges::count_if(available_section, [](const auto& pair) { return pair.second; });
    if (num_sections_in_file > 1)
    {
      std::string found_sections;
      for (const auto& [section, exists] : available_section)
      {
        if (exists) found_sections += "'" + section + "' ";
      }
      FOUR_C_THROW(
          "Multiple options to read mesh for discretization '{}'. Only one is allowed.\n Found "
          "sections: {}",
          dis->name(), found_sections);
    }

    if (num_sections_in_file == 0 || available_section[section_name + " ELEMENTS"])
    {
      // This used to be the default, so we use it for backwards compatibility.
      element_readers.emplace_back(ElementReader(dis, input_, section_name + " ELEMENTS"));
    }
    else if (available_section[section_name + " DOMAIN"])
    {
      domain_readers.emplace_back(DomainReader{
          .target_discretization = *dis,
          .section_name = section_name + " DOMAIN",
      });
    }
    else if (available_section[section_name + " GEOMETRY"])
    {
      exodus_readers_.emplace_back(
          std::make_unique<Internal::ExodusReader>(*dis, section_name + " GEOMETRY"));
    }
  }

  // Read all the elements first
  for (auto& element_reader : element_readers)
  {
    element_reader.read_and_distribute();
  }

  // Only now read the nodes since they must belong to one of the read elements.
  read_nodes(input_, "NODE COORDS", element_readers, max_node_id);

  for (auto& element_reader : element_readers)
  {
    rebalance_discretization(
        *element_reader.get_dis(), *element_reader.get_row_elements(), parameters_, comm_);
  }

  Core::Communication::broadcast(max_node_id, 0, comm_);
  for (auto& domain_reader : domain_readers)
  {
    generate_mesh(input_, domain_reader, max_node_id);
    max_node_id = domain_reader.target_discretization.node_row_map()->max_all_gid() + 1;
  }

  // First, we look at all the mesh files we are going to read and determine if they are
  // duplicated. For now, we only support the case where all files are the same.
  if (Core::Communication::my_mpi_rank(comm_) == 0)
  {
    // We only support one mesh file at the moment. We check if all the files are the same.
    std::shared_ptr<Exodus::Mesh> mesh;
    std::filesystem::path mesh_file;
    for (auto& exodus_reader : exodus_readers_)
    {
      FOUR_C_ASSERT(input_.has_section(exodus_reader->section_name), "Internal error.");

      Core::IO::InputParameterContainer data;
      input_.match_section(exodus_reader->section_name, data);

      const auto& geometry_data = data.group(exodus_reader->section_name);
      const auto& exodus_file = geometry_data.get<std::filesystem::path>("FILE");
      const auto& exodus_verbosity =
          geometry_data.get<Core::IO::Exodus::VerbosityLevel>("SHOW_INFO");
      if (mesh)
      {
        FOUR_C_ASSERT_ALWAYS(mesh_file == exodus_file,
            "All Exodus mesh input must come from the same file. Found different files '{}' and "
            "'{}'.",
            exodus_file.string(), mesh_file.string());
      }
      else
      {
        mesh_file = exodus_file;
        mesh = std::make_unique<Core::IO::Exodus::Mesh>(
            exodus_file.string(), Core::IO::Exodus::MeshParameters{
                                      // We internally depend on node numbers starting at 0.
                                      .node_start_id = 0,
                                  });
        mesh->print(std::cout, exodus_verbosity);
      }
      exodus_reader->mesh_on_rank_zero = mesh;
    }
  }

  int ele_count = 0;
  for (auto& exodus_reader : exodus_readers_)
  {
    read_mesh_from_exodus(input_, *exodus_reader, parameters_, ele_count, comm_);
  }
}

// Default destructor in implementation to enable unique_ptr in header.
Core::IO::MeshReader::~MeshReader() = default;

MPI_Comm Core::IO::MeshReader::get_comm() const { return comm_; }


const Core::IO::Exodus::Mesh* Core::IO::MeshReader::get_exodus_mesh_on_rank_zero() const
{
  if (exodus_readers_.empty()) return nullptr;

  FOUR_C_ASSERT(std::ranges::all_of(exodus_readers_,
                    [&](const auto& exodus_reader)
                    {
                      return exodus_reader->mesh_on_rank_zero ==
                             exodus_readers_.front()->mesh_on_rank_zero;
                    }),
      "Internal error: all meshes are supposed to be the same.");

  return exodus_readers_.front()->mesh_on_rank_zero.get();
}

FOUR_C_NAMESPACE_CLOSE

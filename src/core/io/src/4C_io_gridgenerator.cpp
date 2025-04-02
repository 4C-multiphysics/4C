// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_gridgenerator.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_element_definition.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_input_spec_validators.hpp"
#include "4C_io_pstream.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_rebalance_graph_based.hpp"

#include <Teuchos_ParameterList.hpp>

#include <algorithm>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO::GridGenerator
{
  // forward declarations
  std::shared_ptr<Core::Elements::Element> create_hex_element(int eleid, int nodeoffset, int myrank,
      const Core::IO::InputParameterContainer& ele_data, std::array<int, 3> interval,
      std::string elementtype, Core::FE::CellType cell_type);

  std::shared_ptr<Core::Elements::Element> create_wedge_element(int eleid, int nodeoffset,
      int myrank, const Core::IO::InputParameterContainer& ele_data, std::array<int, 3> interval,
      std::string elementtype, Core::FE::CellType cell_type);

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void create_rectangular_cuboid_discretization(Core::FE::Discretization& dis,
      const Core::IO::GridGenerator::RectangularCuboidInputs& inputData, bool outputFlag)
  {
    MPI_Comm comm = dis.get_comm();
    const int myrank = Core::Communication::my_mpi_rank(comm);
    const int numproc = Core::Communication::num_mpi_ranks(comm);

    const Core::Elements::ElementDefinition ed;
    // safety checks
    for (int i = 0; i < 3; ++i)
    {
      if (inputData.bottom_corner_point_[i] >= inputData.top_corner_point_[i])
        FOUR_C_THROW("lower bound in domain reader must be smaller than upper bound");

      if (inputData.interval_[i] <= 0)
        FOUR_C_THROW("intervals in domain reader must be greater than zero");
    }

    std::shared_ptr<Core::LinAlg::Map> nodeRowMap;
    std::shared_ptr<Core::LinAlg::Map> nodeColMap;
    std::shared_ptr<Core::LinAlg::Map> elementRowMap;
    std::shared_ptr<Core::LinAlg::Map> elementColMap;

    // Create initial (or final) map of row elements
    int numnewele = inputData.interval_[0] * inputData.interval_[1] * inputData.interval_[2];
    if (inputData.autopartition_)  // linear map
    {
      int scale = 1;
      if (inputData.cell_type == Core::FE::CellType::wedge6 or
          inputData.cell_type == Core::FE::CellType::wedge15)
      {
        scale = 2;
      }
      elementRowMap = std::make_shared<Core::LinAlg::Map>(scale * numnewele, 0, comm);
    }
    else  // fancy final box map
    {
      // Error for invalid element types!!!
      if (inputData.cell_type != Core::FE::CellType::hex8 and
          inputData.cell_type != Core::FE::CellType::hex20 and
          inputData.cell_type != Core::FE::CellType::hex27)
      {
        FOUR_C_THROW("This map-partition is only available for HEX-elements!");
      }

      std::vector<int> factors;
      int nproc = numproc;
      for (int fac = 2; fac < nproc + 1;)
      {
        if (nproc % fac == 0)
        {
          factors.push_back(fac);
          nproc /= fac;
        }
        else
        {
          fac++;
        }
      }
      if (nproc != 1) FOUR_C_THROW("Could not split numproc.");

      unsigned int subdivisions[] = {1, 1, 1};
      const double dinterval[] = {static_cast<double>(inputData.interval_[0]),
          static_cast<double>(inputData.interval_[1]), static_cast<double>(inputData.interval_[2])};
      for (std::vector<int>::const_reverse_iterator fac = factors.rbegin(); fac != factors.rend();
          ++fac)
      {
        const double ratios[] = {dinterval[0] / subdivisions[0], dinterval[1] / subdivisions[1],
            dinterval[2] / subdivisions[2]};
        if (ratios[0] >= ratios[1] && ratios[0] >= ratios[2])
          subdivisions[0] *= *fac;
        else if (ratios[1] >= ratios[0] && ratios[1] >= ratios[2])
          subdivisions[1] *= *fac;
        else if (ratios[2] >= ratios[0] && ratios[2] >= ratios[1])
          subdivisions[2] *= *fac;
      }

      if (myrank == 0 && outputFlag)
        Core::IO::cout << "Determined domain subdivision to: " << subdivisions[0] << "x"
                       << subdivisions[1] << "x" << subdivisions[2] << "\n";

      std::vector<unsigned int> xranges(subdivisions[0] + 1ul);
      for (size_t i = 0; i < subdivisions[0] + 1ul; ++i)
        xranges[i] = std::max(0, std::min(inputData.interval_[0],
                                     static_cast<int>(round(i * dinterval[0] / subdivisions[0]))));

      std::vector<unsigned int> yranges(subdivisions[1] + 1ul);
      for (size_t i = 0; i < subdivisions[1] + 1ul; ++i)
        yranges[i] = std::max(0, std::min(inputData.interval_[1],
                                     static_cast<int>(round(i * dinterval[1] / subdivisions[1]))));

      std::vector<unsigned int> zranges(subdivisions[2] + 1ul);
      for (size_t i = 0; i < subdivisions[2] + 1ul; ++i)
        zranges[i] = std::max(0, std::min(inputData.interval_[2],
                                     static_cast<int>(round(i * dinterval[2] / subdivisions[2]))));

      const unsigned int mysection[] = {myrank % subdivisions[0],
          (myrank / subdivisions[0]) % subdivisions[1],
          myrank / (subdivisions[0] * subdivisions[1])};
      const int nummynewele = (xranges[mysection[0] + 1] - xranges[mysection[0]]) *
                              (yranges[mysection[1] + 1] - yranges[mysection[1]]) *
                              (zranges[mysection[2] + 1] - zranges[mysection[2]]);
      std::vector<int> mynewele(nummynewele);

      size_t idx = 0;
      for (size_t iz = zranges[mysection[2]]; iz < zranges[mysection[2] + 1]; ++iz)
        for (size_t it = yranges[mysection[1]]; it < yranges[mysection[1] + 1]; ++it)
          for (size_t ix = xranges[mysection[0]]; ix < xranges[mysection[0] + 1]; ++ix)
            mynewele[idx++] = (iz * inputData.interval_[1] + it) * inputData.interval_[0] + ix;

      elementRowMap =
          std::make_shared<Core::LinAlg::Map>(-1, nummynewele, mynewele.data(), 0, comm);
    }

    // Create the actual elements according to the row map
    for (int lid = 0; lid < elementRowMap->num_my_elements(); ++lid)
    {
      int eleid = elementRowMap->gid(lid);
      FOUR_C_ASSERT(eleid >= 0, "Missing gid");

      const Core::IO::InputParameterContainer& ele_data = inputData.element_arguments;

      // Create specified elements
      switch (inputData.cell_type)
      {
        case Core::FE::CellType::hex8:
        case Core::FE::CellType::hex20:
        case Core::FE::CellType::hex27:
        {
          std::shared_ptr<Core::Elements::Element> ele =
              create_hex_element(eleid, inputData.node_gid_of_first_new_node_, myrank, ele_data,
                  inputData.interval_, inputData.elementtype_, inputData.cell_type);
          // add element to discretization
          dis.add_element(ele);
          break;
        }
        case Core::FE::CellType::wedge6:
        case Core::FE::CellType::wedge15:
        {
          std::shared_ptr<Core::Elements::Element> ele = IO::GridGenerator::create_wedge_element(
              eleid, inputData.node_gid_of_first_new_node_, myrank, ele_data, inputData.interval_,
              inputData.elementtype_, inputData.cell_type);
          dis.add_element(ele);
          break;
        }
        default:
          FOUR_C_THROW(
              "The discretization type {}, is not implemented. Currently only HEX(8,20,27) and "
              "WEDGE(6,15) are implemented for the box geometry generation.",
              inputData.cell_type);
      }
    }

    // redistribute the elements
    if (inputData.autopartition_)
    {
      std::shared_ptr<const Core::LinAlg::Graph> nodeGraph =
          Core::Rebalance::build_graph(dis, *elementRowMap);

      Teuchos::ParameterList rebalanceParams;
      rebalanceParams.set<std::string>(
          "num_global_parts", std::to_string(Core::Communication::num_mpi_ranks(comm)));

      std::tie(nodeRowMap, nodeColMap) =
          Core::Rebalance::rebalance_node_maps(*nodeGraph, rebalanceParams);
    }
    else  // do not destroy our manual partitioning
    {
      std::shared_ptr<const Core::LinAlg::Graph> graph =
          Core::Rebalance::build_graph(dis, *elementRowMap);
      nodeRowMap = std::make_shared<Core::LinAlg::Map>(
          -1, graph->row_map().num_my_elements(), graph->row_map().my_global_elements(), 0, comm);
      nodeColMap = std::make_shared<Core::LinAlg::Map>(
          -1, graph->col_map().num_my_elements(), graph->col_map().my_global_elements(), 0, comm);
    }


    // now we have all elements in a linear map roweles
    // build reasonable maps for elements from the
    // already valid and final node maps
    // note that nothing is actually redistributed in here
    std::tie(elementRowMap, elementColMap) = dis.build_element_row_column(*nodeRowMap, *nodeColMap);

    // we can now export elements to reasonable row element distribution
    dis.export_row_elements(*elementRowMap);

    // export to the column map / create ghosting of elements
    dis.export_column_elements(*elementColMap);

    // Create the nodes according to their elements
    // number of nodes per direction
    const size_t nx = 2 * inputData.interval_[0] + 1;
    const size_t ny = 2 * inputData.interval_[1] + 1;
    int maxgid = -1;

    // as we are using the redistributed row node map, the nodes are directly created on the
    // correct processors

    // Compute midpoint for rotations of the box geometry
    std::vector<double> coordm(3, 0.0);
    if (inputData.rotation_angle_[0] != 0.0 || inputData.rotation_angle_[1] != 0.0 ||
        inputData.rotation_angle_[2] != 0.0)
    {
      coordm[0] = (inputData.top_corner_point_[0] + inputData.bottom_corner_point_[0]) / 2.;
      coordm[1] = (inputData.top_corner_point_[1] + inputData.bottom_corner_point_[1]) / 2.;
      coordm[2] = (inputData.top_corner_point_[2] + inputData.bottom_corner_point_[2]) / 2.;
    }

    for (int lid = 0; lid < nodeRowMap->num_my_elements(); ++lid)
    {
      const int gid = nodeRowMap->gid(lid);
      maxgid = std::max(gid, maxgid);

      const int posid = gid - inputData.node_gid_of_first_new_node_;
      FOUR_C_ASSERT(posid >= 0, "Tried to access a node gid that was not on this proc");
      size_t i = posid % nx;
      size_t j = (posid / nx) % ny;
      size_t k = posid / (nx * ny);

      std::array<double, 3> coords;
      coords[0] = static_cast<double>(i) / (2 * inputData.interval_[0]) *
                      (inputData.top_corner_point_[0] - inputData.bottom_corner_point_[0]) +
                  inputData.bottom_corner_point_[0];
      coords[1] = static_cast<double>(j) / (2 * inputData.interval_[1]) *
                      (inputData.top_corner_point_[1] - inputData.bottom_corner_point_[1]) +
                  inputData.bottom_corner_point_[1];
      coords[2] = static_cast<double>(k) / (2 * inputData.interval_[2]) *
                      (inputData.top_corner_point_[2] - inputData.bottom_corner_point_[2]) +
                  inputData.bottom_corner_point_[2];

      // If set perform rotations, applied in the order, x,y,z-axis
      for (int rotaxis = 0; rotaxis < 3; ++rotaxis)
      {
        if (inputData.rotation_angle_[rotaxis] != 0.0)
        {
          // add rotation around mitpoint here.
          double dx[3];
          dx[0] = coords[0] - coordm[0];
          dx[1] = coords[1] - coordm[1];
          dx[2] = coords[2] - coordm[2];

          double calpha = cos(inputData.rotation_angle_[rotaxis] * std::numbers::pi / 180);
          double salpha = sin(inputData.rotation_angle_[rotaxis] * std::numbers::pi / 180);

          coords[0] = coordm[0];  //+ calpha*dx[0] + salpha*dx[1];
          coords[1] = coordm[1];  //+ -salpha*dx[0] + calpha*dx[1];
          coords[2] = coordm[2];

          coords[(rotaxis + 1) % 3] +=
              calpha * dx[(rotaxis + 1) % 3] + salpha * dx[(rotaxis + 2) % 3];
          coords[(rotaxis + 2) % 3] +=
              calpha * dx[(rotaxis + 2) % 3] - salpha * dx[(rotaxis + 1) % 3];
          coords[rotaxis] += dx[rotaxis];
        }
      }

      dis.add_node(coords, gid, nullptr);
    }
    dis.export_column_nodes(*nodeColMap);
  }


  InputSpec RectangularCuboidInputs::spec()
  {
    using namespace Core::IO::InputSpecBuilders;
    using namespace Core::IO::InputSpecBuilders::Validators;

    Elements::ElementDefinition element_definition;

    return all_of({
        parameter<std::array<double, 3>>("bottom_corner_point",
            {.description = "Coordinates of the first point specifying the cuboid."}),
        parameter<std::array<double, 3>>("top_corner_point",
            {.description = "Coordinates of the second point specifying the cuboid. Every "
                            "coordinate should be strictly greater than the corresponding "
                            "one in bottom_corner_point."}),
        parameter<std::array<int, 3>>("subdivisions",
            {
                .description = "The number of elements to generate per dimension.",
                .validator = all_elements(positive<int>()),
            }),
        parameter<std::array<double, 3>>("rotation_angle",
            {
                .description = "Optional rotation in Euler angles, i.e., rotation "
                               "about the x-, y-, and z-axis (in that order).",
                .default_value = std::array{0.0, 0.0, 0.0},
                .validator = all_elements(in_range(0.0, excl(360.0))),
            }),
        parameter<bool>("auto_partition",
            {.description = "Partition the geometry with 4C's standard rebalancing "
                            "techniques (when true) or manually partition based on the knowledge "
                            "of the domain intervals (when false).",
                .default_value = false}),
        group("elements", {element_definition.element_data_spec()},
            {.description = "Specify which elements should be generated."}),
    });
  }


  RectangularCuboidInputs RectangularCuboidInputs::from_input(
      const Core::IO::InputParameterContainer& input)
  {
    RectangularCuboidInputs result;

    result.bottom_corner_point_ = input.get<std::array<double, 3>>("bottom_corner_point");
    result.top_corner_point_ = input.get<std::array<double, 3>>("top_corner_point");
    result.interval_ = input.get<std::array<int, 3>>("subdivisions");
    result.rotation_angle_ = input.get<std::array<double, 3>>("rotation_angle");
    result.autopartition_ = input.get<bool>("auto_partition");

    Elements::ElementDefinition element_definition;
    std::tie(result.elementtype_, result.cell_type, result.element_arguments) =
        element_definition.unpack_element_data(input.group("elements"));

    return result;
  }


  /*----------------------------------------------------------------------*
   | create HEX type elements for the partition                           |
   *----------------------------------------------------------------------*/
  std::shared_ptr<Core::Elements::Element> create_hex_element(int eleid, int nodeOffset, int myrank,
      const Core::IO::InputParameterContainer& ele_data, std::array<int, 3> interval,
      std::string elementtype, Core::FE::CellType cell_type)
  {
    // Reserve nodeids for this element type
    std::vector<int> nodeids(Core::FE::get_number_of_element_nodes(cell_type));

    // current element position
    const size_t ex = 2 * (eleid % interval[0]);
    const size_t ey = 2 * ((eleid / interval[0]) % interval[1]);
    const size_t ez = 2 * (eleid / (interval[0] * interval[1]));

    // number of nodes per direction
    const size_t nx = 2 * interval[0] + 1;
    const size_t ny = 2 * interval[1] + 1;

    switch (nodeids.size())
    {
      case 27:
        nodeids[20] = nodeOffset + (ez * ny + ey + 1) * nx + ex + 1;
        nodeids[21] = nodeOffset + ((ez + 1) * ny + ey) * nx + ex + 1;
        nodeids[22] = nodeOffset + ((ez + 1) * ny + ey + 1) * nx + ex + 2;
        nodeids[23] = nodeOffset + ((ez + 1) * ny + ey + 2) * nx + ex + 1;
        nodeids[24] = nodeOffset + ((ez + 1) * ny + ey + 1) * nx + ex;
        nodeids[25] = nodeOffset + ((ez + 2) * ny + ey + 1) * nx + ex + 1;
        nodeids[26] = nodeOffset + ((ez + 1) * ny + ey + 1) * nx + ex + 1;
        [[fallthrough]];
      case 20:
        nodeids[8] = nodeOffset + (ez * ny + ey) * nx + ex + 1;
        nodeids[9] = nodeOffset + (ez * ny + ey + 1) * nx + ex + 2;
        nodeids[10] = nodeOffset + (ez * ny + ey + 2) * nx + ex + 1;
        nodeids[11] = nodeOffset + (ez * ny + ey + 1) * nx + ex;
        nodeids[12] = nodeOffset + ((ez + 1) * ny + ey) * nx + ex;
        nodeids[13] = nodeOffset + ((ez + 1) * ny + ey) * nx + ex + 2;
        nodeids[14] = nodeOffset + ((ez + 1) * ny + ey + 2) * nx + ex + 2;
        nodeids[15] = nodeOffset + ((ez + 1) * ny + ey + 2) * nx + ex;
        nodeids[16] = nodeOffset + ((ez + 2) * ny + ey) * nx + ex + 1;
        nodeids[17] = nodeOffset + ((ez + 2) * ny + ey + 1) * nx + ex + 2;
        nodeids[18] = nodeOffset + ((ez + 2) * ny + ey + 2) * nx + ex + 1;
        nodeids[19] = nodeOffset + ((ez + 2) * ny + ey + 1) * nx + ex;
        [[fallthrough]];
      case 8:
        nodeids[0] = nodeOffset + (ez * ny + ey) * nx + ex;
        nodeids[1] = nodeOffset + (ez * ny + ey) * nx + ex + 2;
        nodeids[2] = nodeOffset + (ez * ny + ey + 2) * nx + ex + 2;
        nodeids[3] = nodeOffset + (ez * ny + ey + 2) * nx + ex;
        nodeids[4] = nodeOffset + ((ez + 2) * ny + ey) * nx + ex;
        nodeids[5] = nodeOffset + ((ez + 2) * ny + ey) * nx + ex + 2;
        nodeids[6] = nodeOffset + ((ez + 2) * ny + ey + 2) * nx + ex + 2;
        nodeids[7] = nodeOffset + ((ez + 2) * ny + ey + 2) * nx + ex;
        break;
      default:
        FOUR_C_THROW("The number of nodeids: {}, does not correspond to a supported HEX-element.",
            nodeids.size());
        break;
    }
    // let the factory create a matching empty element
    std::shared_ptr<Core::Elements::Element> ele = Core::Communication::factory(
        elementtype, Core::FE::cell_type_to_string(cell_type), eleid, myrank);
    ele->set_node_ids(nodeids.size(), &(nodeids[0]));
    ele->read_element(elementtype, Core::FE::cell_type_to_string(cell_type), ele_data);
    return ele;
  }

  /*----------------------------------------------------------------------*
   | Create WEDGE type elements for the partition.                        |
   | For even eleids -> create 1st part of HEX equivalent, odd -> 2nd     |
   | part of HEX equivalent.                                              |
   | Wedges aligned in z-direction.                                       |
   *----------------------------------------------------------------------*/
  std::shared_ptr<Core::Elements::Element> create_wedge_element(int eleid, int nodeoffset,
      int myrank, const Core::IO::InputParameterContainer& ele_data, std::array<int, 3> interval,
      std::string elementtype, Core::FE::CellType cell_type)
  {
    // Reserve nodeids for this element type
    std::vector<int> nodeids(Core::FE::get_number_of_element_nodes(cell_type));

    // HEX-equivalent element
    int hex_equiv_eleid = int(eleid / 2);

    // current element position
    const size_t ex = 2 * (hex_equiv_eleid % interval[0]);
    const size_t ey = 2 * ((hex_equiv_eleid / interval[0]) % interval[1]);
    const size_t ez = 2 * (hex_equiv_eleid / (interval[0] * interval[1]));

    // number of nodes per direction
    const size_t nx = 2 * interval[0] + 1;
    const size_t ny = 2 * interval[1] + 1;

    // Create 2 elements for every hex element. Even - Odd pairs.
    if (eleid % 2 == 0)  // Even - elements
    {
      switch (nodeids.size())
      {
        case 15:
          nodeids[6] = nodeoffset + (ez * ny + ey) * nx + ex + 1;             // HEX-eqvi: 8
          nodeids[8] = nodeoffset + (ez * ny + ey + 1) * nx + ex;             // HEX-eqvi: 11
          nodeids[9] = nodeoffset + ((ez + 1) * ny + ey) * nx + ex;           // HEX-eqvi: 12
          nodeids[10] = nodeoffset + ((ez + 1) * ny + ey) * nx + ex + 2;      // HEX-eqvi: 13
          nodeids[11] = nodeoffset + ((ez + 1) * ny + ey + 2) * nx + ex;      // HEX-eqvi: 15
          nodeids[12] = nodeoffset + ((ez + 2) * ny + ey) * nx + ex + 1;      // HEX-eqvi: 16
          nodeids[14] = nodeoffset + ((ez + 2) * ny + ey + 1) * nx + ex;      // HEX-eqvi: 19
          nodeids[7] = nodeoffset + (ez * ny + ey + 1) * nx + ex + 1;         // HEX-eqvi: 20
          nodeids[13] = nodeoffset + ((ez + 2) * ny + ey + 1) * nx + ex + 1;  // HEX-eqvi: 25
          [[fallthrough]];
        case 6:
          nodeids[0] = nodeoffset + (ez * ny + ey) * nx + ex;            // HEX-eqvi: 0
          nodeids[1] = nodeoffset + (ez * ny + ey) * nx + ex + 2;        // HEX-eqvi: 1
          nodeids[2] = nodeoffset + (ez * ny + ey + 2) * nx + ex;        // HEX-eqvi: 3
          nodeids[3] = nodeoffset + ((ez + 2) * ny + ey) * nx + ex;      // HEX-eqvi: 4
          nodeids[4] = nodeoffset + ((ez + 2) * ny + ey) * nx + ex + 2;  // HEX-eqvi: 5
          nodeids[5] = nodeoffset + ((ez + 2) * ny + ey + 2) * nx + ex;  // HEX-eqvi: 7
          break;
          //---------------------
        default:
          FOUR_C_THROW(
              "The number of nodeids: {}, does not correspond to a supported WEDGE-element.",
              nodeids.size());
          break;
      }
    }
    else  // Odd - elements
    {
      switch (nodeids.size())
      {
        case 15:
          nodeids[6] = nodeoffset + (ez * ny + ey + 1) * nx + ex + 2;         // HEX-eqvi: 9
          nodeids[7] = nodeoffset + (ez * ny + ey + 2) * nx + ex + 1;         // HEX-eqvi: 10
          nodeids[9] = nodeoffset + ((ez + 1) * ny + ey) * nx + ex + 2;       // HEX-eqvi: 13
          nodeids[10] = nodeoffset + ((ez + 1) * ny + ey + 2) * nx + ex + 2;  // HEX-eqvi: 14
          nodeids[11] = nodeoffset + ((ez + 1) * ny + ey + 2) * nx + ex;      // HEX-eqvi: 15
          nodeids[12] = nodeoffset + ((ez + 2) * ny + ey + 1) * nx + ex + 2;  // HEX-eqvi: 17
          nodeids[13] = nodeoffset + ((ez + 2) * ny + ey + 2) * nx + ex + 1;  // HEX-eqvi: 18
          nodeids[8] = nodeoffset + (ez * ny + ey + 1) * nx + ex + 1;         // HEX-eqvi: 20
          nodeids[14] = nodeoffset + ((ez + 2) * ny + ey + 1) * nx + ex + 1;  // HEX-eqvi: 25
          [[fallthrough]];
        case 6:
          nodeids[0] = nodeoffset + (ez * ny + ey) * nx + ex + 2;            // HEX-eqvi: 1
          nodeids[1] = nodeoffset + (ez * ny + ey + 2) * nx + ex + 2;        // HEX-eqvi: 2
          nodeids[2] = nodeoffset + (ez * ny + ey + 2) * nx + ex;            // HEX-eqvi: 3
          nodeids[3] = nodeoffset + ((ez + 2) * ny + ey) * nx + ex + 2;      // HEX-eqvi: 5
          nodeids[4] = nodeoffset + ((ez + 2) * ny + ey + 2) * nx + ex + 2;  // HEX-eqvi: 6
          nodeids[5] = nodeoffset + ((ez + 2) * ny + ey + 2) * nx + ex;      // HEX-eqvi: 7
          break;
          //---------------------
        default:
          FOUR_C_THROW(
              "The number of nodeids: {}, does not correspond to a supported WEDGE-element.",
              nodeids.size());
          break;
      }
    }

    // let the factory create a matching empty element
    std::shared_ptr<Core::Elements::Element> ele = Core::Communication::factory(
        elementtype, FE::cell_type_to_string(cell_type), eleid, myrank);
    ele->set_node_ids(nodeids.size(), &(nodeids[0]));
    ele->read_element(elementtype, FE::cell_type_to_string(cell_type), ele_data);
    return ele;
  }



}  // namespace Core::IO::GridGenerator

FOUR_C_NAMESPACE_CLOSE

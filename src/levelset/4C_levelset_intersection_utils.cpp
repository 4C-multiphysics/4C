// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_levelset_intersection_utils.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_cut_integrationcell.hpp"
#include "4C_cut_levelsetintersection.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_geometry_element_coordtrafo.hpp"
#include "4C_fem_geometry_element_volume.hpp"
#include "4C_fem_geometry_integrationcell.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
ScaTra::LevelSet::Intersection::Intersection()
    : check_lsv_(false), desired_positions_(0), volumeplus_(0.0), volumeminus_(0.0), surface_(0.0)
{
  desired_positions_.reserve(2);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::LevelSet::Intersection::reset()
{
  volumeplus_ = 0.0;
  volumeminus_ = 0.0;
  surface_ = 0.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::LevelSet::Intersection::capture_zero_level_set(const Core::LinAlg::Vector<double>& phi,
    const Core::FE::Discretization& scatradis, double& volumedomainminus, double& volumedomainplus,
    double& zerosurface, std::map<int, Core::Geo::BoundaryIntCells>& elementBoundaryIntCells)
{
  // reset, just to be sure
  reset();
  volumedomainminus = 0.0;
  volumedomainplus = 0.0;
  zerosurface = 0.0;
  elementBoundaryIntCells.clear();

  // herein the actual capturing happens
  get_zero_level_set(phi, scatradis, elementBoundaryIntCells);

  // collect contributions from all procs and store in respective variables
  volumedomainplus = Core::Communication::sum_all(volume_plus(), scatradis.get_comm());
  volumedomainminus = Core::Communication::sum_all(volume_minus(), scatradis.get_comm());
  zerosurface = Core::Communication::sum_all(surface(), scatradis.get_comm());

  // export also interface to all procs
  export_interface(elementBoundaryIntCells, scatradis.get_comm());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <typename T>
void ScaTra::LevelSet::Intersection::get_zero_level_set(const Core::LinAlg::Vector<double>& phi,
    const Core::FE::Discretization& scatradis, std::map<int, T>& elementBoundaryIntCells,
    bool cut_screenoutput)
{
  // export phi from row to column map
  Core::LinAlg::Vector<double> phicol(*scatradis.dof_col_map());
  Core::LinAlg::export_to(phi, phicol);

  // remark: loop over row elements is sufficient
  for (int iele = 0; iele < scatradis.num_my_row_elements(); ++iele)
  {
    // get element from discretization
    const Core::Elements::Element* ele = scatradis.l_row_element(iele);
    const Core::FE::CellType distype = ele->shape();

    // clear vector each loop
    boundary_int_cells_per_ele<T>().clear();

    // ------------------------------------------------------------------------
    // Prepare cut
    // ------------------------------------------------------------------------
    Cut::LevelSetIntersection levelset(Core::Communication::my_mpi_rank(scatradis.get_comm()));
    Core::LinAlg::SerialDenseMatrix xyze;
    std::vector<double> phi_nodes;
    std::vector<int> nids;
    prepare_cut(ele, scatradis, phicol, xyze, phi_nodes, nids);

    // check if this element is cut, according to its level-set values
    // -> add it to 'levelset'
    // note: cut is performed in physical space
    if (!levelset.add_element(1, nids, xyze, ele->shape(), phi_nodes.data(), false, check_lsv_))
      continue;

    // ------------------------------------------------------------------------
    // call Cut algorithm and process cut data
    // ------------------------------------------------------------------------
    Cut::ElementHandle* ehandle = cut(levelset, xyze, phi_nodes, cut_screenoutput);

    // =========================================================
    // cell is in contact with the interface (cut or touched)
    // =========================================================
    if (ehandle != nullptr)
    {
      Cut::plain_element_set cuteles;

      collect_cut_eles(*ehandle, cuteles, distype);

      // ----------------------------------------------------------------------
      // get zero level-set contour
      // ----------------------------------------------------------------------
      get_zero_level_set_contour(cuteles, xyze, distype);
    }
    // =========================================================
    // element is uncut
    // =========================================================
    else
    {
      double elevol = Core::Geo::element_volume(distype, xyze);

      // it is sufficient to check the first node, since the element entirely
      // lies within the plus or minus domain
      if (phi_nodes[0] > 0.0)
        volume_plus() += elevol;
      else
        volume_minus() += elevol;
    }

    // store interface of element
    if (boundary_int_cells_per_ele<T>().size() > 0)
      elementBoundaryIntCells[ele->id()] = boundary_int_cells_per_ele<T>();
  }

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::LevelSet::Intersection::get_zero_level_set_contour(
    const Cut::plain_element_set& cuteles, const Core::LinAlg::SerialDenseMatrix& xyze,
    Core::FE::CellType distype)
{
  for (Cut::plain_element_set::const_iterator icutele = cuteles.begin(); icutele != cuteles.end();
      ++icutele)
  {
    // get pointer to cut element
    Cut::Element* cutele = *icutele;

    Cut::plain_volumecell_set volcells;
    volcells = cutele->volume_cells();

    for (Cut::plain_volumecell_set::const_iterator ivolcell = volcells.begin();
        ivolcell != volcells.end(); ++ivolcell)
    {
      Cut::VolumeCell* volcell = *ivolcell;
      const Cut::Point::PointPosition vol_pos = volcell->position();
      if (is_point_position(vol_pos))
      {
        add_to_volume(vol_pos, volcell->volume());
        // get boundary integration cells for this volume cell
        // we consider only the cells for one position, otherwise we would have the boundary
        // cells twice
        const Cut::plain_boundarycell_set& bcells = volcell->boundary_cells();
        for (Cut::plain_boundarycell_set::const_iterator ibcell = bcells.begin();
            ibcell != bcells.end(); ++ibcell)
        {
          Cut::BoundaryCell* bcell = *ibcell;

          add_to_boundary_int_cells_per_ele(xyze, *bcell, distype);

          surface() += bcell->area();
        }
      }
      else
      {
        add_to_volume(vol_pos, volcell->volume());
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::LevelSet::Intersection::add_to_boundary_int_cells_per_ele(
    const Core::LinAlg::SerialDenseMatrix& xyze, const Cut::BoundaryCell& bcell,
    Core::FE::CellType distype_ele)
{
  Core::FE::CellType distype_bc = bcell.shape();
  check_boundary_cell_type(distype_bc);

  const int numnodebc = Core::FE::get_number_of_element_nodes(distype_bc);

  // get physical coordinates of this cell
  Core::LinAlg::SerialDenseMatrix coord = bcell.coordinates();

  // transfer to element coordinates
  Core::LinAlg::SerialDenseMatrix localcoord(3, numnodebc, true);

  for (int ivert = 0; ivert < numnodebc; ivert++)
  {
    Core::LinAlg::Matrix<3, 1> lcoord;
    Core::LinAlg::Matrix<3, 1> pcoord;
    for (int ll = 0; ll < 3; ll++) pcoord(ll, 0) = coord(ll, ivert);

    Core::Geo::current_to_volume_element_coordinates(distype_ele, xyze, pcoord, lcoord);

    // write as 'physCoord'
    for (int ll = 0; ll < 3; ll++) localcoord(ll, ivert) = lcoord(ll, 0);
  }

  // store boundary element and sum area into surface
  // be careful, we only set physical coordinates
  Core::LinAlg::SerialDenseMatrix dummyMat;
  boundary_int_cells_per_ele<Core::Geo::BoundaryIntCells>().push_back(
      Core::Geo::BoundaryIntCell(distype_bc, -1, localcoord, dummyMat, coord, true));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::LevelSet::Intersection::check_boundary_cell_type(Core::FE::CellType distype_bc) const
{
  if (distype_bc != Core::FE::CellType::tri3 and distype_bc != Core::FE::CellType::quad4)
  {
    FOUR_C_THROW("unexpected type of boundary integration cell: {}",
        Core::FE::cell_type_to_string(distype_bc));
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::LevelSet::Intersection::add_to_volume(Cut::Point::PointPosition pos, double vol)
{
  switch (pos)
  {
    case Cut::Point::outside:
      volume_plus() += vol;
      break;
    case Cut::Point::inside:
      volume_minus() += vol;
      break;
    default:
      /* do nothing for the undecided case */
      break;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::LevelSet::Intersection::collect_cut_eles(
    Cut::ElementHandle& ehandle, Cut::plain_element_set& cuteles, Core::FE::CellType distype) const
{
  ehandle.collect_elements(cuteles);

  switch (distype)
  {
    case Core::FE::CellType::line2:
    case Core::FE::CellType::hex8:
    {
      if (cuteles.size() != 1) FOUR_C_THROW("one cut element expected for linear elements");
      break;
    }
    case Core::FE::CellType::hex20:
    case Core::FE::CellType::hex27:
    {
      if (cuteles.size() != 8) FOUR_C_THROW("eight cut elements expected for quadratic elements");
      break;
    }
    default:
    {
      FOUR_C_THROW("distype unknown for level set cut algorithm");
      break;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::LevelSet::Intersection::prepare_cut(const Core::Elements::Element* ele,
    const Core::FE::Discretization& scatradis, const Core::LinAlg::Vector<double>& phicol,
    Core::LinAlg::SerialDenseMatrix& xyze, std::vector<double>& phi_nodes,
    std::vector<int>& node_ids) const
{
  const Core::FE::CellType distype = ele->shape();
  unsigned numnode = Core::FE::get_number_of_element_nodes(distype);
  const unsigned probdim = Global::Problem::instance()->n_dim();

  xyze.shape(3, numnode);
  switch (distype)
  {
    case Core::FE::CellType::hex8:
      Core::Geo::fill_initial_position_array<Core::FE::CellType::hex8, 3>(ele, xyze);
      break;
    case Core::FE::CellType::hex20:
      Core::Geo::fill_initial_position_array<Core::FE::CellType::hex20, 3>(ele, xyze);
      break;
    case Core::FE::CellType::hex27:
      Core::Geo::fill_initial_position_array<Core::FE::CellType::hex27, 3>(ele, xyze);
      break;
    case Core::FE::CellType::line2:
      switch (probdim)
      {
        case 2:
          Core::Geo::fill_initial_position_array<Core::FE::CellType::line2, 2>(ele, xyze);
          break;
        case 3:
          Core::Geo::fill_initial_position_array<Core::FE::CellType::line2, 3>(ele, xyze);
          break;
        default:
          FOUR_C_THROW("Unsupported problem dimension! (probdim = {})", probdim);
      }
      break;
    default:
      FOUR_C_THROW("Unknown element type ( type = {} )", Core::FE::cell_type_to_string(distype));
      break;
  }

  // we assume one dof per node here
  phi_nodes.resize(ele->num_node(), 0.0);
  std::vector<int> lm;
  std::vector<int> lmowner;
  std::vector<int> lmstride;

  // get element location vector
  lm.clear();
  lmowner.clear();
  lmstride.clear();
  ele->location_vector(scatradis, lm, lmowner, lmstride);
  phi_nodes = Core::FE::extract_values(phicol, lm);

  // define nodal ID's
  node_ids.resize(numnode, 0.0);
  for (std::size_t i = 0; i < numnode; ++i) node_ids[i] = i;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::ElementHandle* ScaTra::LevelSet::Intersection::cut(Cut::LevelSetIntersection& levelset,
    const Core::LinAlg::SerialDenseMatrix& xyze, const std::vector<double>& phi_nodes,
    bool cut_screenoutput) const
{
  try
  {
    levelset.cut(true, cut_screenoutput);
  }
  catch (Core::Exception& err)
  {
    std::cerr << "\n--- failed to cut element ---\n"
              << "coordinates:\n";
    std::cerr << xyze;
    std::cerr << "g-function values:\n" << std::setprecision(16);
    std::copy(phi_nodes.begin(), phi_nodes.end(), std::ostream_iterator<double>(std::cerr, ", "));
    std::cerr << "\n";
    throw;
  }

  return levelset.get_element(1);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool ScaTra::LevelSet::Intersection::is_point_position(const Cut::Point::PointPosition& curr_pos,
    const std::vector<Cut::Point::PointPosition>& desired_pos) const
{
  for (std::vector<Cut::Point::PointPosition>::const_iterator cit = desired_pos.begin();
      cit != desired_pos.end(); ++cit)
  {
    // OR - combination
    if (curr_pos == *cit) return true;
  }

  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const std::vector<Cut::Point::PointPosition>& ScaTra::LevelSet::Intersection::desired_positions()
{
  if (desired_positions_.empty()) desired_positions_.push_back(Cut::Point::outside);
  return desired_positions_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::LevelSet::Intersection::set_desired_positions(
    const std::vector<Cut::Point::PointPosition>& desired_pos)
{
  desired_positions_.resize(desired_pos.size(), Cut::Point::undecided);
  std::copy(desired_pos.begin(), desired_pos.end(), desired_positions_.begin());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::LevelSet::Intersection::export_interface(
    std::map<int, Core::Geo::BoundaryIntCells>& myinterface, MPI_Comm comm)
{
  //-------------------------------
  // prepare parallel communication
  //-------------------------------
  const int myrank = Core::Communication::my_mpi_rank(comm);
  const int numproc = Core::Communication::num_mpi_ranks(comm);

  int size_one = 1;

  Core::Communication::Exporter exporter(comm);

  // destination proc (the "next" one)
  int dest = myrank + 1;
  if (myrank == (numproc - 1)) dest = 0;

  // source proc (the "last" one)
  int source = myrank - 1;
  if (myrank == 0) source = numproc - 1;

#ifdef FOUR_C_ENABLE_ASSERTIONS
  Core::IO::cout << "proc " << myrank << " interface pieces for " << myinterface.size()
                 << " elements available before export" << Core::IO::endl;
#endif

  Core::Communication::PackBuffer data;
  pack_boundary_int_cells(myinterface, data);

  //-----------------------------------------------------------------
  // pack data (my boundary integration cell groups) for initial send
  //-----------------------------------------------------------------
  std::vector<char> dataSend;
  swap(dataSend, data());

  //-----------------------------------------------
  // send data around in a circle to all processors
  //-----------------------------------------------
  // loop over processors
  for (int num = 0; num < numproc - 1; num++)
  {
    std::vector<int> lengthSend(1, 0);
    lengthSend[0] = dataSend.size();

#ifdef FOUR_C_ENABLE_ASSERTIONS
    Core::IO::cout << "--- sending " << lengthSend[0] << " bytes: from proc " << myrank
                   << " to proc " << dest << Core::IO::endl;
#endif

    // send length of the data to be received ...
    MPI_Request req_length_data;
    int length_tag = 0;
    exporter.i_send(myrank, dest, lengthSend.data(), size_one, length_tag, req_length_data);
    // ... and receive length
    std::vector<int> lengthRecv(1, 0);
    exporter.receive(source, length_tag, lengthRecv, size_one);
    exporter.wait(req_length_data);

    // send actual data ...
    int data_tag = 4;
    MPI_Request req_data;
    exporter.i_send(myrank, dest, dataSend.data(), lengthSend[0], data_tag, req_data);
    // ... and receive data
    std::vector<char> dataRecv(lengthRecv[0]);
    exporter.receive_any(source, data_tag, dataRecv, lengthRecv[0]);
    exporter.wait(req_data);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    Core::IO::cout << "--- receiving " << lengthRecv[0] << " bytes: to proc " << myrank
                   << " from proc " << source << Core::IO::endl;
#endif

    //-----------------------------------------------
    // unpack data (boundary integration cell groups)
    //-----------------------------------------------
    std::map<int, Core::Geo::BoundaryIntCells> interface_recv;

    unpack_boundary_int_cells(dataRecv, interface_recv);

    // add group of cells to my interface map
    /* remark: all groups of boundary integration cells (interface pieces
     * within an element) are collected here */
    for (std::map<int, Core::Geo::BoundaryIntCells>::const_iterator cellgroup =
             interface_recv.begin();
        cellgroup != interface_recv.end(); ++cellgroup)
    {
      myinterface.insert(*cellgroup);
    }

    // make received data the new 'to be sent' data
    dataSend = dataRecv;

    // processors wait for each other
    Core::Communication::barrier(comm);
  }
#ifdef FOUR_C_ENABLE_ASSERTIONS
  Core::IO::cout << "proc " << myrank << " interface pieces for " << myinterface.size()
                 << " elements available after export" << Core::IO::endl;
#endif
}


/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void ScaTra::LevelSet::Intersection::pack_boundary_int_cells(
    const std::map<int, Core::Geo::BoundaryIntCells>& intcellmap,
    Core::Communication::PackBuffer& dataSend)
{
  // pack data on all processors
  // loop entries of map (groups of boundary integration cells)
  for (std::map<int, Core::Geo::BoundaryIntCells>::const_iterator cellgroup = intcellmap.begin();
      cellgroup != intcellmap.end(); ++cellgroup)
  {
    // pack data of all boundary integrations cells belonging to an element
    const int elegid = cellgroup->first;
    add_to_pack(dataSend, elegid);

    const int numcells = (cellgroup->second).size();
    add_to_pack(dataSend, numcells);

    for (int icell = 0; icell < numcells; ++icell)
    {
      Core::Geo::BoundaryIntCell cell = cellgroup->second[icell];
      // get all member variables from a single boundary integration cell
      const Core::FE::CellType distype = cell.shape();
      add_to_pack(dataSend, distype);

      // coordinates of cell vertices in (scatra) element parameter space
      const Core::LinAlg::SerialDenseMatrix vertices_xi = cell.cell_nodal_pos_xi_domain();
      add_to_pack(dataSend, vertices_xi);

      // coordinates of cell vertices in physical space
      const Core::LinAlg::SerialDenseMatrix vertices_xyz = cell.cell_nodal_pos_xyz();
      add_to_pack(dataSend, vertices_xyz);
    }
  }
}


/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void ScaTra::LevelSet::Intersection::unpack_boundary_int_cells(
    const std::vector<char>& data, std::map<int, Core::Geo::BoundaryIntCells>& intcellmap)
{
  // pointer to current position in a group of cells in local std::string (counts bytes)

  Core::Communication::UnpackBuffer buffer(data);
  while (!buffer.at_end())
  {
    // extract fluid element gid
    int elegid = -1;
    extract_from_pack(buffer, elegid);
    if (elegid < 0) FOUR_C_THROW("extraction of element gid failed");

    // extract number of boundary integration cells for this element
    int numvecs = -1;
    extract_from_pack(buffer, numvecs);

    // vector holding group of boundary integration cells belonging to this element
    Core::Geo::BoundaryIntCells intcellvector;

    for (int icell = 0; icell < numvecs; ++icell)
    {
      //--------------------------------------------------------------------
      // extract all member variables for a single boundary integration cell
      //--------------------------------------------------------------------
      // distype of cell
      Core::FE::CellType distype;
      extract_from_pack(buffer, distype);
      if (!(distype == Core::FE::CellType::tri3 || distype == Core::FE::CellType::quad4))
        FOUR_C_THROW("unexpected distype {}", distype);

      Core::LinAlg::SerialDenseMatrix vertices_xi;
      extract_from_pack(buffer, vertices_xi);

      // coordinates of cell vertices in physical space
      Core::LinAlg::SerialDenseMatrix vertices_xyz;
      extract_from_pack(buffer, vertices_xyz);

      // store boundary integration cells in boundaryintcelllist
      Core::LinAlg::SerialDenseMatrix dummyMat;
      intcellvector.push_back(
          Core::Geo::BoundaryIntCell(distype, -1, vertices_xi, dummyMat, vertices_xyz));
    }

    // add group of cells for this element to the map
    intcellmap.insert(std::make_pair(elegid, intcellvector));
  }
}


template void ScaTra::LevelSet::Intersection::get_zero_level_set<Core::Geo::BoundaryIntCells>(
    const Core::LinAlg::Vector<double>& phi, const Core::FE::Discretization& scatradis,
    std::map<int, Core::Geo::BoundaryIntCells>& elementBoundaryIntCells, bool cut_screenoutput);
template void ScaTra::LevelSet::Intersection::get_zero_level_set<Core::Geo::BoundaryIntCellPtrs>(
    const Core::LinAlg::Vector<double>& phi, const Core::FE::Discretization& scatradis,
    std::map<int, Core::Geo::BoundaryIntCellPtrs>& elementBoundaryIntCells, bool cut_screenoutput);

FOUR_C_NAMESPACE_CLOSE

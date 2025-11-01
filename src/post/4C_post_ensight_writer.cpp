// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_post_ensight_writer.hpp"

#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fluid_rotsym_periodicbc_utils.hpp"
#include "4C_inpar_problemtype.hpp"
#include "4C_io_legacy_table.hpp"
#include "4C_io_legacy_table_iter.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_post_common.hpp"
#include "4C_utils_enum.hpp"
#include "4C_xfem_discretization.hpp"

#include <numeric>
#include <string>

FOUR_C_NAMESPACE_OPEN

//! 6 Surfaces of a Hex27 element with 9 nodes per surface
const int Hex20_FourCToEnsightGold[20] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15};

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
EnsightWriter::EnsightWriter(PostField* field, const std::string& filename)
    : PostWriterBase(field, filename), nodeidgiven_(true), writecp_(false)
{
  using namespace FourC;

  // initialize proc0map_ correctly
  const std::shared_ptr<Core::FE::Discretization> dis = field_->discretization();
  const Core::LinAlg::Map* noderowmap = dis->node_row_map();
  proc0map_ = Core::LinAlg::allreduce_e_map(*noderowmap, 0);

  // sort proc0map_ so that we can loop it and get nodes in ascending order.
  std::vector<int> sortmap;
  sortmap.reserve(proc0map_->num_my_elements());
  sortmap.assign(proc0map_->my_global_elements(),
      proc0map_->my_global_elements() + proc0map_->num_my_elements());
  std::sort(sortmap.begin(), sortmap.end());
  proc0map_ = std::make_shared<Core::LinAlg::Map>(
      -1, sortmap.size(), sortmap.data(), 0, proc0map_->get_comm());

  // get the number of elements for each distype (global numbers)
  numElePerDisType_ = get_num_ele_per_dis_type(*dis);

  // get the global ids of elements for each distype (global numbers)
  eleGidPerDisType_ = get_ele_gid_per_dis_type(*dis, numElePerDisType_);

  // map between distypes in 4C and existing Ensight strings
  // it includes only strings for cell types known in ensight
  // you need to manually switch to other types distypes before querying this map
  distype2ensightstring_.clear();
  distype2ensightstring_[Core::FE::CellType::point1] = "point";
  distype2ensightstring_[Core::FE::CellType::line2] = "bar2";
  distype2ensightstring_[Core::FE::CellType::line3] = "bar2";  //"bar3";
  distype2ensightstring_[Core::FE::CellType::hex8] = "hexa8";
  distype2ensightstring_[Core::FE::CellType::hex20] = "hexa20";
  distype2ensightstring_[Core::FE::CellType::tet4] = "tetra4";
  distype2ensightstring_[Core::FE::CellType::tet10] = "tetra10";
  distype2ensightstring_[Core::FE::CellType::nurbs8] = "hexa8";
  distype2ensightstring_[Core::FE::CellType::nurbs27] = "hexa8";
  distype2ensightstring_[Core::FE::CellType::nurbs4] = "quad4";
  distype2ensightstring_[Core::FE::CellType::nurbs9] = "quad4";
  distype2ensightstring_[Core::FE::CellType::quad4] = "quad4";
  distype2ensightstring_[Core::FE::CellType::quad8] = "quad8";
  distype2ensightstring_[Core::FE::CellType::tri3] = "tria3";
  distype2ensightstring_[Core::FE::CellType::tri6] = "tria6";
  distype2ensightstring_[Core::FE::CellType::wedge6] = "penta6";
  distype2ensightstring_[Core::FE::CellType::wedge15] = "penta15";
  distype2ensightstring_[Core::FE::CellType::pyramid5] = "pyramid5";
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EnsightWriter::write_files(PostFilterBase& filter)
{
  PostResult result(field_);

  // timesteps when the solution is written
  const std::vector<double> soltime = result.get_result_times(field_->name());
  const unsigned int numsoltimes = soltime.size();

  // When spatial approximation is based on NURBS, we want to write
  // real geometry data and control point information. Therefore,
  // we perform one standard writing step and one additional step
  // for the control points. Here, a new .case file is created which
  // ends with "_cp".
  int iter = 1;
  if (field_->problem()->spatial_approximation_type() == Core::FE::ShapeFunctionType::nurbs) iter++;

  // For none-NURBS cases, this loop is just passed through once!
  for (int i = 0; i < iter; ++i)
  {
    // for control point output:
    // define aux string for control point (cp) files and set writecp_ to true
    // if necessary
    std::string aux = "";
    writecp_ = false;
    if (i == 1)
    {
      writecp_ = true;
      aux = aux + "_cp";

      // should be cleared for control point output
      filesetmap_.clear();
    }

    ///////////////////////////////////
    //  write geometry file          //
    ///////////////////////////////////
    const std::string geofilename = filename_ + "_" + field_->name() + aux + ".geo";
    const size_t found_path = geofilename.find_last_of("/\\");
    const std::string geofilename_nopath = geofilename.substr(found_path + 1);
    write_geo_file(geofilename);
    std::vector<int> filesteps;
    filesteps.push_back(1);
    filesetmap_["geo"] = filesteps;
    std::vector<double> timesteps;
    if (soltime.size() > 0)
      timesteps.push_back(soltime[0]);
    else
      timesteps.push_back(0.0);
    timesetmap_["geo"] = timesteps;
    // at the moment, we can only print out the first step -> to be changed

    ///////////////////////////////////
    //  write solution fields files  //
    ///////////////////////////////////
    filter.write_all_results(field_);

    // prepare the time sets and file sets for case file creation
    int setcounter = 0;
    int allresulttimeset = 0;
    for (std::map<std::string, std::vector<double>>::const_iterator entry = timesetmap_.begin();
        entry != timesetmap_.end(); ++entry)
    {
      std::string key = entry->first;
      if ((entry->second).size() == numsoltimes)
      {
        if (allresulttimeset == 0)
        {
          setcounter++;
          allresulttimeset = setcounter;
        }
        timesetnumbermap_[key] =
            allresulttimeset;  // reuse the default result time set, when possible
      }
      else
      {
        setcounter++;
        timesetnumbermap_[key] = setcounter;  // a new time set number is needed
      }
    }

    // Paraview wants the geo file to be fileset number one
    setcounter = 1;
    for (std::map<std::string, std::vector<int>>::const_iterator entry = filesetmap_.begin();
        entry != filesetmap_.end(); ++entry)
    {
      std::string key = entry->first;
      if (entry->first == "geo")
      {
        filesetnumbermap_[key] = 1;
      }
      else
      {
        setcounter++;
        filesetnumbermap_[key] = setcounter;
      }
    }

    ///////////////////////////////////
    //  now write the case file      //
    ///////////////////////////////////
    if (myrank_ == 0)
    {
      const std::string casefilename = filename_ + "_" + field_->name() + aux + ".case";
      std::ofstream casefile;
      casefile.open(casefilename.c_str());
      casefile << "# created using post_ensight\n"
               << "FORMAT\n\n"
               << "type:\tensight gold\n";

      casefile << "\nGEOMETRY\n\n";
      casefile << "model:\t" << timesetnumbermap_["geo"] << "\t" << filesetnumbermap_["geo"] << "\t"
               << geofilename_nopath << "\n";

      casefile << "\nVARIABLE\n\n";
      casefile << get_variable_section(filesetmap_, variablenumdfmap_, variablefilenamemap_);

      casefile << "\nTIME\n\n";
      casefile << get_time_section_string_from_timesets(timesetmap_);

      casefile << "\nFILE\n\n";
      casefile << get_file_section_string_from_filesets(filesetmap_);

      casefile.close();
    }
  }  // end of loop

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EnsightWriter::write_geo_file(const std::string& geofilename)
{
  // open file
  std::ofstream geofile;
  if (myrank_ == 0)
  {
    geofile.open(geofilename.c_str());
    if (!geofile) FOUR_C_THROW("failed to open file: {}", geofilename);
  }

  // header
  write(geofile, "C Binary");

  // print out one
  // if more are needed, this has to go into a loop
  std::map<std::string, std::vector<std::ofstream::pos_type>> resultfilepos;

  {
    write_geo_file_one_time_step(geofile, resultfilepos, "geo");
  }

  // append index table
  // TODO: ens_checker complains if this is turned!!!! but I can't see, whats wrong here a.ger 11/07
  // it is also correct to omit write_index_table, however the EnsightGold Format manual says,
  // it would improve performance to have it on...
  // Writing the index for the result fields is fine. Complains only for the geometry-file  gb 02/10
  // write_index_table(geofile, resultfilepos["geo"]);

  if (geofile.is_open()) geofile.close();

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EnsightWriter::write_geo_file_one_time_step(std::ofstream& file,
    std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
    const std::string name)
{
  using namespace FourC;

  std::vector<std::ofstream::pos_type>& filepos = resultfilepos[name];
  write(file, "BEGIN TIME STEP");
  filepos.push_back(file.tellp());

  write(file, field_->name() + " geometry");
  write(file, "Comment");

  // nodeidgiven_ is set to true inside the class constructor
  if (nodeidgiven_)
    write(file, "node id given");
  else
    write(file, "node id assign");

  write(file, "element id off");

  // part + partnumber + comment
  write(file, "part");
  write(file, field_->field_pos() + 1);
  write(file, field_->name() + " field");


  // switch between nurbs an others
  if (field_->problem()->spatial_approximation_type() == Core::FE::ShapeFunctionType::nurbs &&
      !writecp_)
  {
    // cast dis to NurbsDiscretisation
    Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
        dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(*(field_->discretization())));

    if (nurbsdis == nullptr)
    {
      FOUR_C_THROW("This probably isn't a NurbsDiscretization\n");
    }

    // get number of patches
    int npatches = (nurbsdis->get_knot_vector())->return_np();

    int totalnumvisp = 0;

    // loop all patches
    for (int np = 0; np < npatches; ++np)
    {
      // get nurbs dis' knotvector sizes
      std::vector<int> nele_x_mele_x_lele(nurbsdis->return_nele_x_mele_x_lele(np));

      int numvisp = 1;

      for (unsigned rr = 0; rr < nele_x_mele_x_lele.size(); ++rr)
      {
        numvisp *= 2 * (nele_x_mele_x_lele[rr]) + 1;
      }
      totalnumvisp += numvisp;
    }
    if (myrank_ == 0)
    {
      std::cout << "Writing coordinates for " << totalnumvisp << " visualisation points\n";
    }
    write(file, "coordinates");
    write(file, totalnumvisp);
  }
  else
  {
    write(file, "coordinates");
    write(file, field_->num_nodes());
  }

  // write the grid information
  std::shared_ptr<Core::LinAlg::Map> proc0map = write_coordinates(file, *field_->discretization());
  proc0map_ = proc0map;  // update the internal map
  write_cells(file, field_->discretization(), proc0map);

  write(file, "END TIME STEP");
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> EnsightWriter::write_coordinates(
    std::ofstream& geofile, Core::FE::Discretization& dis)
{
  using namespace FourC;

  Core::FE::ShapeFunctionType distype = field_->problem()->spatial_approximation_type();
  if (myrank_ == 0)
  {
    std::cout << "(computing) coordinates for a ";
    std::cout << Core::FE::shape_function_type_to_string(distype);
    std::cout << " approximation\n";
  }

  // map for all visualisation points after they have been
  // communicated to proc 0
  std::shared_ptr<Core::LinAlg::Map> proc0map;

  switch (distype)
  {
    case Core::FE::ShapeFunctionType::polynomial:
    case Core::FE::ShapeFunctionType::hdg:
    {
      write_coordinates_for_polynomial_shapefunctions(geofile, dis, proc0map);
      break;
    }
    case Core::FE::ShapeFunctionType::nurbs:
    {
      // write real geometry coordinates
      if (!writecp_) write_coordinates_for_nurbs_shapefunctions(geofile, dis, proc0map);
      // write control point coordinates
      else
        write_coordinates_for_polynomial_shapefunctions(geofile, dis, proc0map);
      break;
    }
    default:
    {
      FOUR_C_THROW("Undefined spatial approximation type.");
    }
  }

  return proc0map;
}


/*----------------------------------------------------------------------*
  | write node connectivity for every element                  gjb 12/07 |
  *----------------------------------------------------------------------*/
void EnsightWriter::write_cells(std::ofstream& geofile,
    const std::shared_ptr<Core::FE::Discretization> dis,
    const std::shared_ptr<Core::LinAlg::Map>& proc0map) const
{
  using namespace FourC;

  const Core::LinAlg::Map* elementmap = dis->element_row_map();

  std::vector<int> nodevector;
  if (myrank_ > 0)
  {
    // reserve sufficient memory for storing the node connectivity
    //(ghosted nodes included)
    nodevector.reserve(dis->num_my_col_nodes());
  }

  // for each found distype write block of the same typed elements
  NumElePerDisType::const_iterator iter;
  for (iter = numElePerDisType_.begin(); iter != numElePerDisType_.end(); ++iter)
  {
    const Core::FE::CellType distypeiter = iter->first;
    const int ne = get_num_ele_output(distypeiter, iter->second);
    const std::string ensightCellType = get_ensight_string(distypeiter);

    if (myrank_ == 0)
    {
      std::cout << "writing " << iter->second << " " << Core::FE::cell_type_to_string(distypeiter)
                << " element(s) as " << ne << " " << ensightCellType << " ensight cell(s)..."
                << std::endl;
      write(geofile, ensightCellType);
      write(geofile, ne);
    }

    nodevector.clear();

    // loop all available elements
    for (int iele = 0; iele < elementmap->num_my_elements(); ++iele)
    {
      Core::Elements::Element* const actele = dis->g_element(elementmap->gid(iele));
      if (actele->shape() == distypeiter)
      {
        Core::Nodes::Node** const nodes = actele->nodes();
        switch (actele->shape())
        {
          case Core::FE::CellType::point1:
          case Core::FE::CellType::line2:
            // case Core::FE::CellType::line3: // Ensight format supports line3,
            // Paraview does not.
          case Core::FE::CellType::hex8:
          case Core::FE::CellType::quad4:
          case Core::FE::CellType::quad8:
          case Core::FE::CellType::tet4:
          case Core::FE::CellType::tet10:
          case Core::FE::CellType::tri3:
          case Core::FE::CellType::tri6:
          case Core::FE::CellType::wedge6:
          case Core::FE::CellType::wedge15:
          case Core::FE::CellType::pyramid5:
          {
            // standard case with direct support
            const int numnp = actele->num_node();
            for (int inode = 0; inode < numnp; ++inode)
            {
              if (myrank_ == 0)  // proc0 can write its elements immediately
                write(geofile, proc0map->lid(nodes[inode]->id()) + 1);
              else  // elements on other procs have to store their global node ids
                nodevector.push_back(nodes[inode]->id());
            }
            break;
          }
          case Core::FE::CellType::hex20:
          {
            // standard case with direct support
            const int numnp = actele->num_node();
            for (int inode = 0; inode < numnp; ++inode)
            {
              if (myrank_ == 0)  // proc0 can write its elements immediately
                write(geofile, proc0map->lid(nodes[Hex20_FourCToEnsightGold[inode]]->id()) + 1);
              else  // elements on other procs have to store their global node ids
                nodevector.push_back(nodes[Hex20_FourCToEnsightGold[inode]]->id());
            }
            break;
          }
          case Core::FE::CellType::hex16:
          {
            // write subelements
            for (int isubele = 0; isubele < get_num_sub_ele(Core::FE::CellType::hex16); ++isubele)
              for (int isubnode = 0; isubnode < 8; ++isubnode)
                if (myrank_ == 0)  // proc0 can write its elements immediately
                  write(geofile, proc0map->lid(nodes[subhex16map[isubele][isubnode]]->id()) + 1);
                else  // elements on other procs have to store their global node ids
                  nodevector.push_back(nodes[subhex16map[isubele][isubnode]]->id());
            break;
          }
          case Core::FE::CellType::hex18:
          {
            // write subelements
            for (int isubele = 0; isubele < get_num_sub_ele(Core::FE::CellType::hex18); ++isubele)
              for (int isubnode = 0; isubnode < 8; ++isubnode)
                if (myrank_ == 0)  // proc0 can write its elements immediately
                  write(geofile, proc0map->lid(nodes[subhex18map[isubele][isubnode]]->id()) + 1);
                else  // elements on other procs have to store their global node ids
                  nodevector.push_back(nodes[subhex18map[isubele][isubnode]]->id());
            break;
          }
          case Core::FE::CellType::hex27:
          {
            // write subelements
            for (int isubele = 0; isubele < get_num_sub_ele(Core::FE::CellType::hex27); ++isubele)
              for (int isubnode = 0; isubnode < 8; ++isubnode)
                if (myrank_ == 0)  // proc0 can write its elements immediately
                  write(geofile, proc0map->lid(nodes[subhexmap[isubele][isubnode]]->id()) + 1);
                else  // elements on other procs have to store their global node ids
                  nodevector.push_back(nodes[subhexmap[isubele][isubnode]]->id());
            break;
          }
          case Core::FE::CellType::quad9:
          {
            // write subelements
            for (int isubele = 0; isubele < get_num_sub_ele(Core::FE::CellType::quad9); ++isubele)
              for (int isubnode = 0; isubnode < 4; ++isubnode)
                if (myrank_ == 0)  // proc0 can write its elements immediately
                  write(geofile, proc0map->lid(nodes[subquadmap[isubele][isubnode]]->id()) + 1);
                else  // elements on other procs have to store their global node ids
                  nodevector.push_back(nodes[subquadmap[isubele][isubnode]]->id());
            break;
          }
          case Core::FE::CellType::line3:
          {
            // write subelements
            for (int isubele = 0; isubele < get_num_sub_ele(Core::FE::CellType::line3); ++isubele)
              for (int isubnode = 0; isubnode < 2; ++isubnode)
                if (myrank_ == 0)  // proc0 can write its elements immediately
                  write(geofile, proc0map->lid(nodes[sublinemap[isubele][isubnode]]->id()) + 1);
                else  // elements on other procs have to store their global node ids
                  nodevector.push_back(nodes[sublinemap[isubele][isubnode]]->id());
            break;
          }
          case Core::FE::CellType::nurbs4:
          {
            if (!writecp_)
              write_nurbs_cell(actele->shape(), actele->id(), geofile, nodevector, *dis, *proc0map);
            else
            {
              // standard case with direct support
              const int numnp = actele->num_node();
              for (int inode = 0; inode < numnp; ++inode)
              {
                if (myrank_ == 0)  // proc0 can write its elements immediately
                  write(geofile, proc0map->lid(nodes[inode]->id()) + 1);
                else  // elements on other procs have to store their global node ids
                  nodevector.push_back(nodes[inode]->id());
              }
            }
            break;
          }
          case Core::FE::CellType::nurbs9:
          {
            if (!writecp_)
              write_nurbs_cell(actele->shape(), actele->id(), geofile, nodevector, *dis, *proc0map);
            else
            {
              // write subelements
              for (int isubele = 0; isubele < get_num_sub_ele(Core::FE::CellType::quad9); ++isubele)
                for (int isubnode = 0; isubnode < 4; ++isubnode)
                  if (myrank_ == 0)  // proc0 can write its elements immediately
                    write(geofile, proc0map->lid(nodes[subquadmap[isubele][isubnode]]->id()) + 1);
                  else  // elements on other procs have to store their global node ids
                    nodevector.push_back(nodes[subquadmap[isubele][isubnode]]->id());
            }
            break;
          }
          case Core::FE::CellType::nurbs27:
          {
            if (!writecp_)
              write_nurbs_cell(actele->shape(), actele->id(), geofile, nodevector, *dis, *proc0map);
            else
            {
              // write subelements
              for (int isubele = 0; isubele < get_num_sub_ele(Core::FE::CellType::hex27); ++isubele)
                for (int isubnode = 0; isubnode < 8; ++isubnode)
                  if (myrank_ == 0)  // proc0 can write its elements immediately
                    write(geofile, proc0map->lid(nodes[subhexmap[isubele][isubnode]]->id()) + 1);
                  else  // elements on other procs have to store their global node ids
                    nodevector.push_back(nodes[subhexmap[isubele][isubnode]]->id());
            }
            break;
          }
          break;
          default:
            FOUR_C_THROW("don't know, how to write this element type as a cell");
        }
      }
    }

    // now do some communicative work for the parallel case:
    // proc 1 to proc n have to send their stored node connectivity to proc0
    // which does the writing

    write_node_connectivity_par(geofile, *dis, nodevector, *proc0map);
  }
  return;
}


/*!
 * \brief communicate and write node connectivity in parallel case

*/
void EnsightWriter::write_node_connectivity_par(std::ofstream& geofile,
    Core::FE::Discretization& dis, const std::vector<int>& nodevector,
    Core::LinAlg::Map& proc0map) const
{
  using namespace FourC;

  // now we have communicated the connectivity infos from proc 1...proc n to proc 0

  std::vector<char> sblock;  // sending block
  std::vector<char> rblock;  // receiving block

  // create an exporter for communication
  Core::Communication::Exporter exporter(dis.get_comm());

  // pack my node ids into sendbuffer
  sblock.clear();

  Core::Communication::PackBuffer data;
  add_to_pack(data, nodevector);
  swap(sblock, data());

  // now we start the communication
  for (unsigned int pid = 0;
      pid < static_cast<unsigned int>(Core::Communication::num_mpi_ranks(dis.get_comm())); ++pid)
  {
    MPI_Request request;
    int tag = 0;
    int frompid = pid;
    int topid = 0;
    int length = sblock.size();

    //--------------------------------------------------
    // proc pid sends its values to proc 0
    if (myrank_ == pid)
    {
      exporter.i_send(frompid, topid, sblock.data(), sblock.size(), tag, request);
    }

    //--------------------------------------------------
    // proc 0 receives from proc pid
    rblock.clear();
    if (myrank_ == 0)
    {
      exporter.receive_any(frompid, tag, rblock, length);
      if (tag != 0)
      {
        FOUR_C_THROW("Proc 0 received wrong message (ReceiveAny)");
      }
      exporter.wait(request);
    }

    // for safety
    Core::Communication::barrier(exporter.get_comm());

    //--------------------------------------------------
    // Unpack received block and write the data
    if (myrank_ == 0)
    {
      std::vector<int> nodeids;
      Core::Communication::UnpackBuffer buffer(rblock);
      // extract data from received package
      while (!buffer.at_end())
      {
        extract_from_pack(buffer, nodeids);
      }
      // compute node lid based on proc0map and write it to file
      for (int i = 0; i < (int)nodeids.size(); ++i)
      {
        // using the same map as for the writing the node coordinates
        int id = (proc0map.lid(nodeids[i])) + 1;
        write(geofile, id);
      }
      nodeids.clear();
    }  // end unpack

    // for safety
    Core::Communication::barrier(exporter.get_comm());

  }  // for pid

  return;
}


/*!
 * \brief parse all elements and get the global(!) number of elements for each distype

 */
EnsightWriter::NumElePerDisType EnsightWriter::get_num_ele_per_dis_type(
    Core::FE::Discretization& dis) const
{
  using namespace FourC;

  const Core::LinAlg::Map* elementmap = dis.element_row_map();

  NumElePerDisType numElePerDisType;
  for (int iele = 0; iele < elementmap->num_my_elements(); ++iele)
  {
    Core::Elements::Element* actele = dis.g_element(elementmap->gid(iele));
    const Core::FE::CellType distype = actele->shape();
    // update counter for current distype
    numElePerDisType[distype]++;
  }

  // in parallel case we have to sum up the local element distype numbers

  // determine maximum number of possible element discretization types
  auto numeledistypes = static_cast<int>(Core::FE::CellType::max_distype);

  // write the final local numbers into a vector
  std::vector<int> myNumElePerDisType(numeledistypes);
  NumElePerDisType::const_iterator iter;
  for (iter = numElePerDisType.begin(); iter != numElePerDisType.end(); ++iter)
  {
    const Core::FE::CellType distypeiter = iter->first;
    const int ne = iter->second;
    myNumElePerDisType[static_cast<int>(distypeiter)] += ne;
  }

  // wait for all procs before communication is started
  Core::Communication::barrier((dis.get_comm()));

  // form the global sum
  std::vector<int> globalnumeleperdistype(numeledistypes);
  globalnumeleperdistype = Core::Communication::sum_all(myNumElePerDisType, (dis.get_comm()));

  // create return argument containing the global element numbers per distype
  NumElePerDisType globalNumElePerDisType;
  for (int i = 0; i < numeledistypes; ++i)
  {
    if (globalnumeleperdistype[i] > 0)  // no entry when we have no element of this type
      globalNumElePerDisType[Core::FE::CellType(i)] = globalnumeleperdistype[i];
  }

  return globalNumElePerDisType;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int EnsightWriter::get_num_ele_output(const Core::FE::CellType distype, const int numele) const
{
  return get_num_sub_ele(distype) * numele;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int EnsightWriter::get_num_sub_ele(const Core::FE::CellType distype) const
{
  using namespace FourC;

  switch (distype)
  {
    case Core::FE::CellType::hex18:
      return 4;
      break;
    case Core::FE::CellType::hex27:
      return 8;
      break;
    case Core::FE::CellType::nurbs27:
      return 8;
      break;
    case Core::FE::CellType::quad9:
      return 4;
      break;
    case Core::FE::CellType::nurbs9:
      return 4;
      break;
    case Core::FE::CellType::line3:
      return 2;
      break;
    default:
      return 1;  // no element splitting necessary
  }
}

/*!
 * \brief parse all elements and get the global ids of the elements for each distype
 */
EnsightWriter::EleGidPerDisType EnsightWriter::get_ele_gid_per_dis_type(
    Core::FE::Discretization& dis, NumElePerDisType numeleperdistype) const
{
  using namespace FourC;

  const Core::LinAlg::Map* elementmap = dis.element_row_map();

  EleGidPerDisType eleGidPerDisType;

  // allocate memory
  NumElePerDisType::const_iterator iter;
  for (iter = numElePerDisType_.begin(); iter != numElePerDisType_.end(); ++iter)
  {
    eleGidPerDisType[iter->first].reserve(iter->second);
  }

  for (int iele = 0; iele < elementmap->num_my_elements(); ++iele)
  {
    const int gid = elementmap->gid(iele);
    Core::Elements::Element* actele = dis.g_element(gid);
    const Core::FE::CellType distype = actele->shape();
    // update counter for current distype
    eleGidPerDisType[distype].push_back(gid);
  }

  // in parallel case we have to provide the ele gids located on other procs as well
  EleGidPerDisType globaleleGidPerDisType;
  NumElePerDisType::const_iterator iterator;

  for (iterator = numElePerDisType_.begin(); iterator != numElePerDisType_.end(); ++iterator)
  {
    // wait for all procs before communication is started
    Core::Communication::barrier((dis.get_comm()));

    // no we have to communicate everything from proc 1...proc n to proc 0
    std::vector<char> sblock;  // sending block
    std::vector<char> rblock;  // receiving block

    // create an exporter for communication
    Core::Communication::Exporter exporter(dis.get_comm());

    // pack my element gids of this discretization type into sendbuffer
    sblock.clear();

    Core::Communication::PackBuffer data;
    add_to_pack(data, eleGidPerDisType[iterator->first]);
    swap(sblock, data());

    // now we start the communication
    for (unsigned int pid = 0;
        pid < static_cast<unsigned int>(Core::Communication::num_mpi_ranks(dis.get_comm())); ++pid)
    {
      MPI_Request request;
      int tag = 0;
      int frompid = pid;
      int topid = 0;
      int length = sblock.size();

      //--------------------------------------------------
      // proc pid sends its values to proc 0
      if (myrank_ == pid)
      {
        exporter.i_send(frompid, topid, sblock.data(), sblock.size(), tag, request);
      }

      //--------------------------------------------------
      // proc 0 receives from proc pid
      rblock.clear();
      if (myrank_ == 0)
      {
        exporter.receive_any(frompid, tag, rblock, length);
        if (tag != 0)
        {
          FOUR_C_THROW("Proc 0 received wrong message (ReceiveAny)");
        }
        exporter.wait(request);
      }

      // for safety
      Core::Communication::barrier(exporter.get_comm());

      //--------------------------------------------------
      // Unpack received block and write the data
      if (myrank_ == 0)
      {
        std::vector<int> elegids;
        // extract data from received package
        Core::Communication::UnpackBuffer buffer(rblock);
        while (!buffer.at_end())
        {
          extract_from_pack(buffer, elegids);
        }
        for (int i = 0; i < (int)elegids.size(); ++i)
        {
          globaleleGidPerDisType[iterator->first].push_back(elegids[i]);
        }
        elegids.clear();
      }  // end unpack

    }  // for pid
  }  // for iter over type

  // note: this map is only filled on proc 0 !!!!
  return globaleleGidPerDisType;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::string EnsightWriter::get_ensight_string(const Core::FE::CellType distype) const
{
  using namespace FourC;

  std::map<Core::FE::CellType, std::string>::const_iterator entry;
  switch (distype)
  {
    case Core::FE::CellType::hex18:
    case Core::FE::CellType::hex27:
      entry = distype2ensightstring_.find(Core::FE::CellType::hex8);
      break;
    case Core::FE::CellType::quad9:
      entry = distype2ensightstring_.find(Core::FE::CellType::quad4);
      break;
    case Core::FE::CellType::tet10:
      entry = distype2ensightstring_.find(Core::FE::CellType::tet10);
      break;
    default:
      entry = distype2ensightstring_.find(distype);
      break;
  }
  if (entry == distype2ensightstring_.end())
    FOUR_C_THROW(
        "No entry in distype2ensightstring_ found for Core::FE::CellType = '{}'.", distype);
  return entry->second;
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EnsightWriter::write_result(const std::string groupname, const std::string name,
    const ResultType restype, const int numdf, const int from /*=0*/,
    const bool fillzeros /*=false*/)
{
  PostResult result(field_);
  bool foundit = false;
  while (result.next_result(groupname))
  {
    if (map_has_map(result.group(), groupname.c_str()))
    {
      foundit = true;
      break;
    }
  }
  if (!foundit) return;

  // FixMe -- hiermeier
  //  if (restype == dofbased)
  //  {
  //    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
  //    std::cout << "!!!Write Dof results is currently not working and thus skipped!!!\n";
  //    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
  //    return;
  //  }
  // new for file continuation
  bool multiple_files = false;

  // For NURBS control point output filename is extended by "_cp"
  std::string aux = "";
  if (writecp_)
  {
    aux = aux + "_cp";
  }

  // open file
  const std::string filename = filename_ + "_" + field_->name() + aux + "." + name;
  std::ofstream file;
  int startfilepos = 0;
  if (myrank_ == 0)
  {
    file.open(filename.c_str());
    startfilepos = file.tellp();  // file position should be zero, but we stay flexible
  }

  std::map<std::string, std::vector<std::ofstream::pos_type>> resultfilepos;
  int stepsize = 0;

  // distinguish between node- and element-based results
  switch (restype)
  {
    case dofbased:
    {
      if (myrank_ == 0) std::cout << "writing dof-based field " << name << std::endl;
      // store information for later case file creation
      variableresulttypemap_[name] = "node";

      // const Core::LinAlg::Map* nodemap = field_->discretization()->NodeRowMap();
      // const int numnp = nodemap->NumGlobalElements();
      // int effnumdf = numdf;
      // if (numdf==2) effnumdf=3; // in 2D we still have to write a 3D vector with zero
      // z-components!!!
      // get the number of bits to be written each time step
      // const int stepsize = 5*80+sizeof(int)+effnumdf*numnp*sizeof(float);

      write_dof_result_step(file, result, resultfilepos, groupname, name, numdf, from, fillzeros);
      // how many bits are necessary per time step (we assume a fixed size)?
      if (myrank_ == 0)
      {
        stepsize = ((int)file.tellp()) - startfilepos;
        if (stepsize <= 0) FOUR_C_THROW("found invalid step size for result file");
      }
      else
        stepsize = 1;  // use dummy value on other procs

      while (result.next_result(groupname))
      {
        if (map_has_map(result.group(), groupname.c_str()))
        {
          const int indexsize = 80 + 2 * sizeof(int) + (file.tellp() / stepsize + 2) * sizeof(long);
          if (static_cast<long unsigned int>(file.tellp()) + stepsize + indexsize >=
              FILE_SIZE_LIMIT_)
          {
            file_switcher(
                file, multiple_files, filesetmap_, resultfilepos, stepsize, name, filename);
          }
          write_dof_result_step(
              file, result, resultfilepos, groupname, name, numdf, from, fillzeros);
        }
      }
    }
    break;

    case nodebased:
    {
      if (myrank_ == 0) std::cout << "writing node-based field " << name << std::endl;
      // store information for later case file creation
      variableresulttypemap_[name] = "node";

      write_nodal_result_step(file, result, resultfilepos, groupname, name, numdf);
      // how many bits are necessary per time step (we assume a fixed size)?
      if (myrank_ == 0)
      {
        stepsize = ((int)file.tellp()) - startfilepos;
        if (stepsize <= 0) FOUR_C_THROW("found invalid step size for result file");
      }
      else
        stepsize = 1;  // use dummy value on other procs

      while (result.next_result(groupname))
      {
        const int indexsize = 80 + 2 * sizeof(int) + (file.tellp() / stepsize + 2) * sizeof(long);
        if (static_cast<long unsigned int>(file.tellp()) + stepsize + indexsize >= FILE_SIZE_LIMIT_)
        {
          file_switcher(file, multiple_files, filesetmap_, resultfilepos, stepsize, name, filename);
        }
        write_nodal_result_step(file, result, resultfilepos, groupname, name, numdf);
      }
    }
    break;

    case elementdof:
    {
      if (myrank_ == 0) std::cout << "writing element based field " << name << std::endl;
      // store information for later case file creation
      variableresulttypemap_[name] = "element";

      write_element_dof_result_step(file, result, resultfilepos, groupname, name, numdf, from);
      // how many bits are necessary per time step (we assume a fixed size)?
      if (myrank_ == 0)
      {
        stepsize = ((int)file.tellp()) - startfilepos;
        if (stepsize <= 0) FOUR_C_THROW("found invalid step size for result file");
      }
      else
        stepsize = 1;  // use dummy value on other procs

      while (result.next_result(groupname))
      {
        const int indexsize = 80 + 2 * sizeof(int) + (file.tellp() / stepsize + 2) * sizeof(long);
        if (static_cast<long unsigned int>(file.tellp()) + stepsize + indexsize >= FILE_SIZE_LIMIT_)
        {
          file_switcher(file, multiple_files, filesetmap_, resultfilepos, stepsize, name, filename);
        }
        write_element_dof_result_step(file, result, resultfilepos, groupname, name, numdf, from);
      }
    }
    break;

    case elementbased:
    {
      if (myrank_ == 0) std::cout << "writing element-based field " << name << std::endl;
      // store information for later case file creation
      variableresulttypemap_[name] = "element";

      write_element_result_step(file, result, resultfilepos, groupname, name, numdf, from);
      // how many bits are necessary per time step (we assume a fixed size)?
      if (myrank_ == 0)
      {
        stepsize = ((int)file.tellp()) - startfilepos;
        if (stepsize <= 0) FOUR_C_THROW("found invalid step size for result file");
      }
      else
        stepsize = 1;  // use dummy value on other procs

      while (result.next_result(groupname))
      {
        const int indexsize = 80 + 2 * sizeof(int) + (file.tellp() / stepsize + 2) * sizeof(long);
        if (static_cast<long unsigned int>(file.tellp()) + stepsize + indexsize >= FILE_SIZE_LIMIT_)
        {
          file_switcher(file, multiple_files, filesetmap_, resultfilepos, stepsize, name, filename);
        }
        write_element_result_step(file, result, resultfilepos, groupname, name, numdf, from);
      }
    }
    break;

    case no_restype:
    case max_restype:
      FOUR_C_THROW("found invalid result type");
      break;
    default:
      FOUR_C_THROW("Invalid output type in WriteResult");
  }  // end of switch(restype)

  // store information for later case file creation
  filesetmap_[name].push_back(
      file.tellp() / stepsize);  // has to be done BEFORE writing the index table
  variablenumdfmap_[name] = numdf;
  variablefilenamemap_[name] = filename;
  // store solution times vector for later case file creation
  {
    PostResult res = PostResult(field_);  // this is needed!
    std::vector<double> restimes = res.get_result_times(field_->name(), groupname);
    timesetmap_[name] = restimes;
  }

  // append index table
  write_index_table(file, resultfilepos[name]);
  resultfilepos[name].clear();

  // close result file
  if (file.is_open()) file.close();

  return;
}

void EnsightWriter::write_result_one_time_step(PostResult& result, const std::string groupname,
    const std::string name, const ResultType restype, const int numdf, bool firststep,
    bool laststep, const int from)
{
  if (not map_has_map(result.group(), groupname.c_str()))
    FOUR_C_THROW("expected result: {} in step {}. Probably a return is missing here. But check!",
        groupname, result.step());

  // FILE CONTINUATION NOT SUPPORTED DUE TO ITS COMPLEXITY FOR VARIABLE GEOMETRY ghamm 03.03.2014
  // guessing whether a new file is necessary should be possible with the current method because the
  // file size is no hard limit more tricky are the things that happen in file_switcher --> need
  // adaptation for variable geometry
  bool multiple_files = false;

  // For NURBS control point output filename is extended by "_cp"
  std::string aux = "";
  if (writecp_)
  {
    aux = aux + "_cp";
  }

  // open file
  const std::string filename = filename_ + "_" + field_->name() + aux + "." + name;
  std::ofstream file;
  int startfilepos = 0;
  if (myrank_ == 0)
  {
    if (firststep)
    {
      file.open(filename.c_str());
      startfilepos = file.tellp();  // file position should be zero, but we stay flexible
    }
    else
    {
      file.open(filename.c_str(), std::ofstream::app | std::ofstream::binary);
      startfilepos = file.tellp();  // file position should be zero, but we stay flexible
    }
  }

  int stepsize = 0;

  // distinguish between node- and element-based results
  switch (restype)
  {
    case dofbased:
    {
      if (myrank_ == 0) std::cout << "writing node-based field " << name << std::endl;
      // store information for later case file creation
      variableresulttypemap_[name] = "node";

      //    const Core::LinAlg::Map* nodemap = field_->discretization()->NodeRowMap();
      //    const int numnp = nodemap->NumGlobalElements();
      //    int effnumdf = numdf;
      //    if (numdf==2) effnumdf=3; // in 2D we still have to write a 3D vector with zero
      //    z-components!!!
      //    // get the number of bits to be written each time step
      //    const int stepsize = 5*80+sizeof(int)+effnumdf*numnp*sizeof(float);

      if (map_has_map(result.group(), groupname.c_str()))
      {
        write_dof_result_step(file, result, resultfilepos_, groupname, name, numdf, from, false);

        // how many bits are necessary per time step (we assume a fixed size)?
        if (myrank_ == 0)
        {
          stepsize = ((int)file.tellp()) - startfilepos;
          if (stepsize <= 0) FOUR_C_THROW("found invalid step size for result file");
        }
        else
          stepsize = 1;  // use dummy value on other procs

        const int indexsize = 80 + 2 * sizeof(int) + (file.tellp() / stepsize + 2) * sizeof(long);
        if (static_cast<long unsigned int>(file.tellp()) + stepsize + indexsize >= FILE_SIZE_LIMIT_)
        {
          FOUR_C_THROW("file continuation not supported for variable geometries");
          file_switcher(
              file, multiple_files, filesetmap_, resultfilepos_, stepsize, name, filename);
        }
      }
    }
    break;

    case nodebased:
    {
      if (myrank_ == 0) std::cout << "writing node-based field " << name << std::endl;
      // store information for later case file creation
      variableresulttypemap_[name] = "node";

      write_nodal_result_step(file, result, resultfilepos_, groupname, name, numdf);

      // how many bits are necessary per time step (we assume a fixed size)?
      if (myrank_ == 0)
      {
        stepsize = ((int)file.tellp()) - startfilepos;
        if (stepsize <= 0) FOUR_C_THROW("found invalid step size for result file");
      }
      else
        stepsize = 1;  // use dummy value on other procs

      const int indexsize = 80 + 2 * sizeof(int) + (file.tellp() / stepsize + 2) * sizeof(long);
      if (static_cast<long unsigned int>(file.tellp()) + stepsize + indexsize >= FILE_SIZE_LIMIT_)
      {
        FOUR_C_THROW("file continuation not supported for variable geometries");
        file_switcher(file, multiple_files, filesetmap_, resultfilepos_, stepsize, name, filename);
      }
    }
    break;

    case elementdof:
    {
      if (myrank_ == 0) std::cout << "writing element based field " << name << std::endl;
      // store information for later case file creation
      variableresulttypemap_[name] = "element";

      write_element_dof_result_step(file, result, resultfilepos_, groupname, name, numdf, from);

      // how many bits are necessary per time step (we assume a fixed size)?
      if (myrank_ == 0)
      {
        stepsize = ((int)file.tellp()) - startfilepos;
        if (stepsize <= 0) FOUR_C_THROW("found invalid step size for result file");
      }
      else
        stepsize = 1;  // use dummy value on other procs

      const int indexsize = 80 + 2 * sizeof(int) + (file.tellp() / stepsize + 2) * sizeof(long);
      if (static_cast<long unsigned int>(file.tellp()) + stepsize + indexsize >= FILE_SIZE_LIMIT_)
      {
        FOUR_C_THROW("file continuation not supported for variable geometries");
        file_switcher(file, multiple_files, filesetmap_, resultfilepos_, stepsize, name, filename);
      }
    }
    break;

    case elementbased:
    {
      if (myrank_ == 0) std::cout << "writing element-based field " << name << std::endl;
      // store information for later case file creation
      variableresulttypemap_[name] = "element";

      write_element_result_step(file, result, resultfilepos_, groupname, name, numdf, from);

      // how many bits are necessary per time step (we assume a fixed size)?
      if (myrank_ == 0)
      {
        stepsize = ((int)file.tellp()) - startfilepos;
        if (stepsize <= 0) FOUR_C_THROW("found invalid step size for result file");
      }
      else
        stepsize = 1;  // use dummy value on other procs

      const int indexsize = 80 + 2 * sizeof(int) + (file.tellp() / stepsize + 2) * sizeof(long);
      if (static_cast<long unsigned int>(file.tellp()) + stepsize + indexsize >= FILE_SIZE_LIMIT_)
      {
        FOUR_C_THROW("file continuation not supported for variable geometries");
        file_switcher(file, multiple_files, filesetmap_, resultfilepos_, stepsize, name, filename);
      }
    }
    break;

    case no_restype:
    case max_restype:
      FOUR_C_THROW("found invalid result type");
      break;
    default:
      FOUR_C_THROW("Invalid output type in WriteResult");
      break;
  }  // end of switch(restype)

  // store information for later case file creation
  timesetmap_[name].push_back(result.time());
  if (laststep)
  {
    filesetmap_[name].push_back((int)(timesetmap_[name].size()));
    variablenumdfmap_[name] = numdf;
    variablefilenamemap_[name] = filename;

    // append index table
    write_index_table(file, resultfilepos_[name]);
    resultfilepos_[name].clear();
  }

  // close result file
  if (file.is_open()) file.close();

  return;
}



void EnsightWriter::write_special_field(SpecialFieldInterface& special, PostResult& result,
    const ResultType restype, const std::string& groupname,
    const std::vector<std::string>& fieldnames, const std::string& outinfo)
{
  const int numfiles = fieldnames.size();

  // new for file continuation
  std::vector<bool> multiple_files(numfiles);
  for (int i = 0; i < numfiles; ++i)
  {
    multiple_files[i] = false;
  }

  // open file
  std::vector<std::string> filenames(numfiles);
  for (int i = 0; i < numfiles; ++i)
  {
    filenames[i] = filename_ + "_" + field_->name() + "." + fieldnames[i];
  }

  std::vector<std::shared_ptr<std::ofstream>> files(numfiles);
  std::vector<int> startfilepos(numfiles);
  for (int i = 0; i < numfiles; ++i) startfilepos[i] = 0;
  for (int i = 0; i < numfiles; ++i)
  {
    files[i] = std::make_shared<std::ofstream>();

    if (myrank_ == 0)
    {
      files[i]->open(filenames[i].c_str());
      startfilepos[i] = files[i]->tellp();  // file position should be zero, but we stay flexible
    }
  }

  std::map<std::string, std::vector<std::ofstream::pos_type>> resultfilepos;
  std::vector<int> stepsize(numfiles);
  for (int i = 0; i < numfiles; ++i)
  {
    stepsize[i] = 0;
  }

  if (myrank_ == 0)
  {
    if (restype == nodebased)
      std::cout << "writing node-based ";
    else if (restype == elementbased)
      std::cout << "writing element-based ";
    else
      std::cout << "writing [unknown type] ";
    std::cout << outinfo << std::endl;
  }

  // store information for later case file creation
  for (int i = 0; i < numfiles; ++i)
  {
    variableresulttypemap_[fieldnames[i]] = (restype == nodebased ? "node" : "element");
  }

  special(files, result, resultfilepos, groupname, fieldnames);

  // how many bits are necessary per time step (we assume a fixed size)?
  if (myrank_ == 0)
  {
    for (int i = 0; i < numfiles; ++i)
    {
      stepsize[i] = ((int)files[i]->tellp()) - startfilepos[i];
      if (stepsize[i] <= 0) FOUR_C_THROW("found invalid step size for result file");
    }
  }
  else
  {
    for (int i = 0; i < numfiles; ++i)
    {
      stepsize[i] = 1;  // use dummy value on other procs
    }
  }

  while (result.next_result())
  {
    for (int i = 0; i < numfiles; ++i)
    {
      const int indexsize =
          80 + 2 * sizeof(int) + (files[i]->tellp() / stepsize[i] + 2) * sizeof(long);
      if (static_cast<long unsigned int>(files[i]->tellp()) + stepsize[i] + indexsize >=
          FILE_SIZE_LIMIT_)
      {
        bool mf = multiple_files[i];
        file_switcher(
            *(files[i]), mf, filesetmap_, resultfilepos, stepsize[i], fieldnames[i], filenames[i]);
        multiple_files[i] = mf;
      }
    }

    special(files, result, resultfilepos, groupname, fieldnames);
  }
  // store information for later case file creation

  const std::vector<int> numdfmap = special.num_df_map();
  FOUR_C_ASSERT(
      static_cast<int>(numdfmap.size()) == numfiles, "Wrong number of components in NumDfMap.");
  for (int i = 0; i < numfiles; ++i)
  {
    filesetmap_[fieldnames[i]].push_back(
        files[i]->tellp() / stepsize[i]);  // has to be done BEFORE writing the index table
    variablenumdfmap_[fieldnames[i]] = numdfmap[i];
    variablefilenamemap_[fieldnames[i]] = filenames[i];
  }

  // store solution times vector for later case file creation
  for (int i = 0; i < numfiles; ++i)
  {
    PostResult res = PostResult(field_);  // this is needed!
    std::vector<double> restimes = res.get_result_times(field_->name(), groupname);
    timesetmap_[fieldnames[i]] = restimes;
  }

  // append index table
  for (int i = 0; i < numfiles; ++i)
  {
    write_index_table(*(files[i]), resultfilepos[fieldnames[i]]);
    resultfilepos[fieldnames[i]].clear();
    if (files[i]->is_open()) files[i]->close();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EnsightWriter::file_switcher(std::ofstream& file, bool& multiple_files,
    std::map<std::string, std::vector<int>>& filesetmap,
    std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos, const int stepsize,
    const std::string name, const std::string filename) const
{
  if (myrank_ == 0)
  {
    std::ostringstream newfilename;

    if (multiple_files == false)
    {
      multiple_files = true;

      std::vector<int> numsteps;
      numsteps.push_back(file.tellp() / stepsize);
      filesetmap[name] = numsteps;

      // append index table
      write_index_table(file, resultfilepos[name]);
      resultfilepos[name].clear();
      file.close();
      rename(filename.c_str(), (filename + "001").c_str());

      newfilename << filename << "002";
    }
    else
    {
      filesetmap[name].push_back(file.tellp() / stepsize);

      // append index table
      write_index_table(file, resultfilepos[name]);
      resultfilepos[name].clear();
      file.close();

      newfilename << filename;
      newfilename.width(3);
      newfilename.fill('0');
      newfilename << filesetmap[name].size() + 1;
    }
    file.open(newfilename.str().c_str());
  }  // if (myrank_==0)
  return;
}


/*!
  \brief Write nodal values for one timestep for dof-based vectors
  Each node has to have the same number of dofs.
*/
void EnsightWriter::write_dof_result_step(std::ofstream& file, PostResult& result,
    std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
    const std::string& groupname, const std::string& name, const int numdf, const int frompid,
    const bool fillzeros) const
{
  using namespace FourC;

  //-------------------------------------------
  // write some key words and read result data
  //-------------------------------------------

  std::vector<std::ofstream::pos_type>& filepos = resultfilepos[name];
  write(file, "BEGIN TIME STEP");
  filepos.push_back(file.tellp());
  write(file, "description");
  write(file, "part");
  write(file, field_->field_pos() + 1);
  write(file, "coordinates");

  const std::shared_ptr<Core::FE::Discretization> dis = field_->discretization();
  const Core::LinAlg::Map* nodemap = dis->node_row_map();  // local node row map
  const int numnp = nodemap->num_global_elements();

  const std::shared_ptr<Core::LinAlg::Vector<double>> data = result.read_result(groupname);
  const Core::LinAlg::Map& datamap = data->get_map();


  // determine offset of dofs in case of multiple discretizations in
  // separate files (e.g. multi-scale problems). during calculation,
  // dofs are numbered consecutively for all discretizations. in the
  // post-processing phase, when only one discretization is called,
  // numbering always starts with 0, so a potential offset needs to be
  // taken into account.

  // find min. GID over all procs. 'datamap->MinAllGID()' / 'dis->dof_row_map()->MinAllGID()'
  // cannot be used, as it would return 0 for procs without elements
  const int num_my_datamap = datamap.num_my_elements();
  const int num_my_dofrowmap = dis->dof_row_map()->num_my_elements();

  // get min. value on this proc or set to max. value of integers if this proc has no elements
  int min_gid_my_datamap =
      num_my_datamap > 0 ? datamap.min_my_gid() : std::numeric_limits<int>::max();
  int min_gid_my_dofrowmap =
      num_my_dofrowmap > 0 ? dis->dof_row_map()->min_my_gid() : std::numeric_limits<int>::max();

  // find min. GID over all procs
  int min_gid_glob_datamap = std::numeric_limits<int>::max();
  int min_gid_glob_dofrowmap = std::numeric_limits<int>::max();

  min_gid_glob_datamap = Core::Communication::min_all(min_gid_my_datamap, dis->get_comm());
  min_gid_glob_dofrowmap = Core::Communication::min_all(min_gid_my_dofrowmap, dis->get_comm());

  // get offset in dofs
  const int offset = min_gid_glob_datamap - min_gid_glob_dofrowmap;

  // switch between nurbs an others
  if (field_->problem()->spatial_approximation_type() == Core::FE::ShapeFunctionType::nurbs &&
      !writecp_)
  {
    write_dof_result_step_for_nurbs(file, numdf, *data, name, offset);
  }
  else if (field_->problem()->spatial_approximation_type() ==
               Core::FE::ShapeFunctionType::polynomial or
           field_->problem()->spatial_approximation_type() == Core::FE::ShapeFunctionType::hdg or
           (field_->problem()->spatial_approximation_type() == Core::FE::ShapeFunctionType::nurbs &&
               writecp_))
  {
    //------------------------------------------------------
    // each processor provides its result values for proc 0
    //------------------------------------------------------

    std::shared_ptr<Core::LinAlg::Map> proc0datamap;
    proc0datamap = Core::LinAlg::allreduce_e_map(datamap, 0);

    // contract result values on proc0 (proc0 gets everything, other procs empty)
    Core::LinAlg::Import proc0dataimporter(*proc0datamap, datamap);
    Core::LinAlg::Vector<double> proc0data(*proc0datamap);
    proc0data.import(*data, proc0dataimporter, Insert);

    const Core::LinAlg::Map& finaldatamap = proc0data.get_map();

    //------------------------------------------------------------------
    // each processor provides its dof global id information for proc 0
    //------------------------------------------------------------------

    // would be nice to have an int vector, instead of casting to doubles
    Core::LinAlg::MultiVector<double> dofgidpernodelid(*nodemap, numdf);
    dofgidpernodelid.put_scalar(-1.0);

    const int mynumnp = nodemap->num_my_elements();
    for (int idf = 0; idf < numdf; ++idf)
    {
      for (int inode = 0; inode < mynumnp; inode++)
      {
        Core::Nodes::Node* n = dis->l_row_node(inode);

        const double dofgid = (double)dis->dof(n, frompid + idf) + offset;
        if (dofgid > -1.0)
        {
          dofgidpernodelid.replace_local_value(inode, idf, dofgid);
        }
        else
        {
          FOUR_C_THROW(
              "Error while creating Core::LinAlg::MultiVector<double> dofgidperlocalnodeid");
        }
      }
    }

    // contract Core::LinAlg::MultiVector<double> on proc0 (proc0 gets everything, other procs
    // empty)
    Core::LinAlg::MultiVector<double> dofgidpernodelid_proc0(*proc0map_, numdf);
    Core::LinAlg::Import proc0dofimporter(*proc0map_, *nodemap);
    dofgidpernodelid_proc0.import(dofgidpernodelid, proc0dofimporter, Insert);


    //---------------
    // write results
    //---------------

    const int finalnumnode = proc0map_->num_global_elements();
    if (myrank_ == 0)  // ensures pointer dofgids is valid
    {
      // care for rotationally symmetric periodic boundary conditions
      // note: only vector fields with numdf > 1 require checking!
      std::map<int, double> pbcslavenodemap;
      std::map<int, double>::iterator iter;
      if (numdf > 1) FLD::get_relevant_slave_nodes_of_rot_sym_pbc(pbcslavenodemap, dis);

      double* dofgids = (dofgidpernodelid_proc0.get_values());  // columnwise data storage
      for (int idf = 0; idf < numdf; ++idf)
      {
        for (int inode = 0; inode < finalnumnode;
            inode++)  // inode == lid of node because we use proc0map_
        {
          // local storage position of desired dof gid
          const int doflid = inode + (idf * numnp);
          // get the dof global id
          const int actdofgid = (int)(dofgids[doflid]);
          FOUR_C_ASSERT(actdofgid >= 0, "error while getting dof global id");
          // get the dof local id w.r.t. the finaldatamap
          int lid = finaldatamap.lid(actdofgid);
          if (lid > -1)
          {
            // is the current node a slave of a rot. symm. periodic boundary condition?
            const int nodegid = proc0map_->gid(inode);
            iter = pbcslavenodemap.find(nodegid);
            if (iter != pbcslavenodemap.end())
            {
              // this is the desired component of the rotated vector field result
              double value =
                  FLD::get_component_of_rotated_vector_field(idf, proc0data, lid, iter->second);
              write(file, static_cast<float>(value));
            }
            else
            {
              // the standard case
              write(file, static_cast<float>(proc0data.local_values_as_span()[lid]));
            }
          }
          else
          {
            if (fillzeros)
              write<float>(file, 0.);
            else
              FOUR_C_THROW("received illegal dof local id: {} (gid={})", lid, actdofgid);
          }
        }
      }  // for idf

      // 2 component vectors in a 3d problem require a row of zeros.
      // do we really need this?
      if (numdf == 2)
      {
        for (int inode = 0; inode < numnp; inode++)
        {
          write<float>(file, 0.);
        }
      }
    }  // if (myrank_==0)
  }
  else
  {
    FOUR_C_THROW("Undefined spatial approximation type.\n");
  }

  write(file, "END TIME STEP");
  return;
}



/*!
  \brief Write nodal values for one timestep for node-based vectors
  Each node has to have the same number of dofs.
*/
void EnsightWriter::write_nodal_result_step(std::ofstream& file, PostResult& result,
    std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
    const std::string& groupname, const std::string& name, const int numdf)
{
  const std::shared_ptr<Core::LinAlg::MultiVector<double>> data =
      result.read_multi_result(groupname);
  write_nodal_result_step(file, data, resultfilepos, groupname, name, numdf);
}



/*!
  \brief Write nodal values for one timestep for node-based vectors
  Each node has to have the same number of dofs.
*/
void EnsightWriter::write_nodal_result_step(std::ofstream& file,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& data,
    std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
    const std::string& groupname, const std::string& name, const int numdf)
{
  using namespace FourC;

  //-------------------------------------------
  // write some key words and read result data
  //-------------------------------------------

  std::vector<std::ofstream::pos_type>& filepos = resultfilepos[name];
  write(file, "BEGIN TIME STEP");
  filepos.push_back(file.tellp());
  write(file, "description");
  write(file, "part");
  write(file, field_->field_pos() + 1);
  write(file, "coordinates");

  const Core::LinAlg::Map& datamap = data->get_map();

  // switch between nurbs an others
  if (field_->problem()->spatial_approximation_type() == Core::FE::ShapeFunctionType::nurbs &&
      !writecp_)
  {
    write_nodal_result_step_for_nurbs(file, numdf, *data, name, 0);
  }
  else if (field_->problem()->spatial_approximation_type() ==
               Core::FE::ShapeFunctionType::polynomial or
           field_->problem()->spatial_approximation_type() == Core::FE::ShapeFunctionType::hdg or
           (field_->problem()->spatial_approximation_type() == Core::FE::ShapeFunctionType::nurbs &&
               writecp_))
  {
    // contract Core::LinAlg::MultiVector<double> on proc0 (proc0 gets everything, other procs
    // empty)
    Core::LinAlg::MultiVector<double> data_proc0(*proc0map_, numdf);
    Core::LinAlg::Import proc0dofimporter(*proc0map_, datamap);
    data_proc0.import(*data, proc0dofimporter, Insert);

    //---------------
    // write results
    //---------------

    const int finalnumnode = proc0map_->num_global_elements();

    if (myrank_ == 0)
    {
      std::vector<int> mycols(numdf);
      std::iota(std::begin(mycols), std::end(mycols), 0);
      // swap entries 5 and 6 (inside 4C we use XY, YZ, XZ, however, ensight-format expects
      // ordering XY, XZ, YZ, for symmetric tensors)
      if (numdf == 6)
      {
        std::cout << "6x1 vector " << name
                  << " interpreted as symmetric 3x3 tensor --> entries 5 and 6 are switched"
                  << std::endl;
        std::iter_swap(mycols.begin() + 4, mycols.begin() + 5);
      }
      for (int idf = 0; idf < numdf; ++idf)
      {
        Core::LinAlg::Vector<double> column((data_proc0)(mycols[idf]));

        for (int inode = 0; inode < finalnumnode;
            inode++)  // inode == lid of node because we use proc0map_
        {
          write(file, static_cast<float>(column.local_values_as_span()[inode]));
        }
      }
    }  // if (myrank_==0)

    // 2 component vectors in a 3d problem require a row of zeros.
    if (numdf == 2)
    {
      for (int inode = 0; inode < finalnumnode; inode++)
      {
        write<float>(file, 0.);
      }
    }
  }  // polynomial || meshfree
  else
  {
    FOUR_C_THROW("Undefined spatial approximation type.\n");
  }

  write(file, "END TIME STEP");
  return;
}


/*!
  \brief Write element dof values for one timestep

  Each element has to have the same number of dofs.
*/
void EnsightWriter::write_element_dof_result_step(std::ofstream& file, PostResult& result,
    std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
    const std::string& groupname, const std::string& name, const int numdof, const int from) const
{
  using namespace FourC;

  //-------------------------------------------
  // write some key words and read result data
  //-------------------------------------------

  std::vector<std::ofstream::pos_type>& filepos = resultfilepos[name];
  write(file, "BEGIN TIME STEP");
  filepos.push_back(file.tellp());
  write(file, "description");
  write(file, "part");
  write(file, field_->field_pos() + 1);

  const std::shared_ptr<Core::FE::Discretization> dis = field_->discretization();
  const Core::LinAlg::Map* elementmap = dis->element_row_map();  // local node row map

  const std::shared_ptr<Core::LinAlg::Vector<double>> data = result.read_result(groupname);
  const Core::LinAlg::Map& datamap = data->get_map();


  //------------------------------------------------------
  // each processor provides its result values for proc 0
  //------------------------------------------------------

  std::shared_ptr<Core::LinAlg::Map> proc0datamap;
  proc0datamap = Core::LinAlg::allreduce_e_map(datamap, 0);

  // contract result values on proc0 (proc0 gets everything, other procs empty)
  Core::LinAlg::Import proc0dataimporter(*proc0datamap, datamap);
  Core::LinAlg::Vector<double> proc0data(*proc0datamap);
  proc0data.import(*data, proc0dataimporter, Insert);

  const Core::LinAlg::Map& finaldatamap = proc0data.get_map();

  //------------------------------------------------------------------
  // each processor provides its dof global id information for proc 0
  //------------------------------------------------------------------

  Core::LinAlg::MultiVector<double> dofgidperelementlid(*elementmap, numdof);
  dofgidperelementlid.put_scalar(-1.0);

  const int nummyelem = elementmap->num_my_elements();
  for (int idof = 0; idof < numdof; ++idof)
  {
    for (int ielem = 0; ielem < nummyelem; ielem++)
    {
      Core::Elements::Element* n = dis->l_row_element(ielem);
      const double dofgid = (double)dis->dof(n, from + idof);
      if (dofgid > -1.0)
      {
        dofgidperelementlid.replace_local_value(ielem, idof, dofgid);
      }
      else
      {
        FOUR_C_THROW("Error while creating Core::LinAlg::MultiVector<double> dofgidperlocalnodeid");
      }
    }
  }

  // contract Core::LinAlg::MultiVector<double> on proc0 (proc0 gets everything, other procs empty)
  Core::LinAlg::MultiVector<double> dofgidperelementlid_proc0(*proc0map_, numdof);
  Core::LinAlg::Import proc0dofimporter(*proc0map_, *elementmap);
  dofgidperelementlid_proc0.import(dofgidperelementlid, proc0dofimporter, Insert);

  const int numglobelem = elementmap->num_global_elements();

  //-------------------------
  // specify the element type
  //-------------------------
  // loop over the different element types present
  EleGidPerDisType::const_iterator iter;
  for (iter = eleGidPerDisType_.begin(); iter != eleGidPerDisType_.end(); ++iter)
  {
    const std::string ensighteleString = get_ensight_string(iter->first);
    const int numelepertype = (iter->second).size();
    std::vector<int> actelegids(numelepertype);
    actelegids = iter->second;
    // write element type
    write(file, ensighteleString);

    //---------------
    // write results
    //---------------
    if (myrank_ == 0)
    {
      if (eleGidPerDisType_.empty() == true) FOUR_C_THROW("no element types available");
    }

    if (myrank_ == 0)  // ensures pointer dofgids is valid
    {
      double* dofgids = (dofgidperelementlid_proc0.get_values());  // columnwise data storage
      for (int idof = 0; idof < numdof; ++idof)
      {
        for (int ielem = 0; ielem < numelepertype;
            ielem++)  // inode == lid of node because we use proc0map_
        {
          // local storage position of desired dof gid
          const int doflid = ielem + (idof * numglobelem);
          // get the dof global id
          const int actdofgid = (int)(dofgids[doflid]);
          FOUR_C_ASSERT(actdofgid >= 0, "error while getting dof global id");
          // get the dof local id w.r.t. the finaldatamap
          int lid = finaldatamap.lid(actdofgid);
          if (lid > -1)
          {
            write(file, static_cast<float>(proc0data.local_values_as_span()[lid]));
          }
          else
            FOUR_C_THROW("received illegal dof local id: {}", lid);
        }
      }
    }  // for idf

    // 2 component vectors in a 3d problem require a row of zeros.
    // do we really need this?
    if (numdof == 2)
    {
      for (int ielem = 0; ielem < numelepertype; ielem++)
      {
        write<float>(file, 0.);
      }
    }

  }  // eledistype


  write(file, "END TIME STEP");
  return;
}



/*----------------------------------------------------------------------*/
/*!
  \brief Write element values for one timestep

  Each element has to have the same number of dofs.

*/
/*----------------------------------------------------------------------*/
void EnsightWriter::write_element_result_step(std::ofstream& file, PostResult& result,
    std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
    const std::string& groupname, const std::string& name, const int numdf, const int from)
{
  const std::shared_ptr<Core::LinAlg::MultiVector<double>> data =
      result.read_multi_result(groupname);
  write_element_result_step(file, data, resultfilepos, groupname, name, numdf, from);
}



/*----------------------------------------------------------------------*/
/*!
  \brief Write element values for one timestep

  Each element has to have the same number of dofs.

*/
/*----------------------------------------------------------------------*/
void EnsightWriter::write_element_result_step(std::ofstream& file,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& data,
    std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
    const std::string& groupname, const std::string& name, const int numdf, const int from)
{
  using namespace FourC;

  //-------------------------------------------
  // write some key words and read result data
  //-------------------------------------------
  std::vector<std::ofstream::pos_type>& filepos = resultfilepos[name];
  write(file, "BEGIN TIME STEP");
  filepos.push_back(file.tellp());
  write(file, "description");
  write(file, "part");
  write(file, field_->field_pos() + 1);

  const Core::LinAlg::Map& datamap = data->get_map();
  const int numcol = data->num_vectors();

  //------------------------------------------------------
  // each processor provides its result values for proc 0
  //------------------------------------------------------

  std::shared_ptr<Core::LinAlg::Map> proc0datamap;
  proc0datamap = Core::LinAlg::allreduce_e_map(datamap, 0);

  // contract result values on proc0 (proc0 gets everything, other procs empty)
  Core::LinAlg::Import proc0dataimporter(*proc0datamap, datamap);
  Core::LinAlg::MultiVector<double> proc0data(*proc0datamap, numcol);
  proc0data.import(*data, proc0dataimporter, Insert);

  const Core::LinAlg::Map& finaldatamap = proc0data.get_map();

  //-------------------------
  // specify the element type
  //-------------------------
  if (myrank_ == 0)
  {
    if (eleGidPerDisType_.empty() == true) FOUR_C_THROW("no element types available");
  }
  // loop over the different element types present
  EleGidPerDisType::const_iterator iter;
  for (iter = eleGidPerDisType_.begin(); iter != eleGidPerDisType_.end(); ++iter)
  {
    const std::string ensighteleString = get_ensight_string(iter->first);

    int numsubele = get_num_sub_ele(iter->first);

    const int numelepertype = (iter->second).size();
    std::vector<int> actelegids(numelepertype);
    actelegids = iter->second;
    // write element type
    write(file, ensighteleString);

    //---------------
    // write results
    //---------------
    if (myrank_ == 0)
    {
      if (numdf + from > numcol)
        FOUR_C_THROW("violated column range of Core::LinAlg::MultiVector<double>: {}", numcol);
      std::vector<int> mycols(numdf);
      std::iota(std::begin(mycols), std::end(mycols), 0);
      // swap entries 5 and 6 (inside 4C we use XY, YZ, XZ, however, ensight-format expects
      // ordering XY, XZ, YZ, for symmetric tensors)
      if (numdf == 6)
      {
        std::cout << "6x1 vector " << name
                  << " interpreted as symmetric 3x3 tensor --> entries 5 and 6 are switched"
                  << std::endl;
        std::iter_swap(mycols.begin() + 4, mycols.begin() + 5);
      }
      for (int col = 0; col < numdf; ++col)
      {
        // extract actual column
        Core::LinAlg::Vector<double> datacolumn((proc0data)(mycols[col] + from));
        for (int iele = 0; iele < numelepertype; iele++)
        {
          // extract element global id
          const int gid = actelegids[iele];
          // get the dof local id w.r.t. the finaldatamap
          // int lid = datamap.LID(gid);
          int lid = finaldatamap.lid(gid);
          if (lid > -1)
          {
            for (int i = 0; i < numsubele; ++i)
              write(file, static_cast<float>(datacolumn.local_values_as_span()[lid]));
          }
          else
            FOUR_C_THROW("received illegal dof local id: {}", lid);
        }
      }
    }  // if (myrank_==0)

    // 2 component vectors in a 3d problem require a row of zeros.
    if (numdf == 2)
    {
      for (int iele = 0; iele < numelepertype; iele++)
      {
        write<float>(file, 0.);
      }
    }

  }  // end iteration over eleGidPerDisType_;

  // finish writing the current time step
  write(file, "END TIME STEP");
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EnsightWriter::write_index_table(
    std::ofstream& file, const std::vector<std::ofstream::pos_type>& filepos) const
{
  std::ofstream::pos_type lastpos = file.tellp();
  const unsigned steps = filepos.size();
  write(file, steps);
  for (unsigned i = 0; i < steps; ++i)
  {
    write<long>(file, filepos[i]);
  }
  write(file, 0);
  write<long>(file, lastpos);
  write(file, "FILE_INDEX");
  return;
}


/*----------------------------------------------------------------------*/
/*!
  \brief write strings of exactly 80 chars
*/
/*----------------------------------------------------------------------*/
void EnsightWriter::write_string(std::ofstream& stream, const std::string str) const
{
  // we need to write 80 bytes per string
  std::vector<char> s(str.begin(), str.end());
  while (s.size() < 80)
  {
    s.push_back('\0');
  }
  stream.write(s.data(), 80);

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::string EnsightWriter::get_variable_section(std::map<std::string, std::vector<int>> filesetmap,
    std::map<std::string, int> variablenumdfmap,
    std::map<std::string, std::string> variablefilenamemap) const
{
  std::stringstream str;

  std::map<std::string, int>::const_iterator variable;

  for (variable = variablenumdfmap.begin(); variable != variablenumdfmap.end(); ++variable)
  {
    const std::string key = variable->first;
    const int numdf = variable->second;
    const std::string filename = variablefilenamemap[key];

    // Get rid of path
    const size_t found_path = filename.find_last_of("/\\");
    const std::string filename_nopath = filename.substr(found_path + 1);

    std::map<std::string, int>::const_iterator timeentry = timesetnumbermap_.find(key);
    if (timeentry == timesetnumbermap_.end()) FOUR_C_THROW("key not found!");
    const int timesetnumber = timeentry->second;

    std::map<std::string, int>::const_iterator entry1 = filesetnumbermap_.find(key);
    if (entry1 == filesetnumbermap_.end()) FOUR_C_THROW("key not found!");
    const int setnumber = entry1->second;

    std::map<std::string, std::vector<int>>::const_iterator entry2 = filesetmap.find(key);
    if (entry2 == filesetmap.end()) FOUR_C_THROW("filesetmap not defined for '{}'", key);

    const int numsubfilesteps = entry2->second.size();
    std::string filename_for_casefile;
    if (numsubfilesteps > 1)
    {
      filename_for_casefile = filename_nopath + "***";
    }
    else
    {
      filename_for_casefile = filename_nopath;
    }
    str << get_variable_entry_for_case_file(
        numdf, setnumber, key, filename_for_casefile, timesetnumber);
  }

  return str.str();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::string EnsightWriter::get_variable_entry_for_case_file(const int numdf,
    const unsigned int fileset, const std::string name, const std::string filename,
    const int timeset) const
{
  std::stringstream str;

  // determine the type of this result variable (node-/element-based)
  std::map<std::string, std::string>::const_iterator entry = variableresulttypemap_.find(name);
  if (entry == variableresulttypemap_.end()) FOUR_C_THROW("key not found!");
  const std::string restypestring = entry->second;

  // create variable entry in the case-file
  switch (numdf)
  {
    case 9:
      str << "tensor asymm per " << restypestring << ":\t" << timeset << "\t" << fileset << "\t"
          << name << "\t" << filename << "\n";
      break;
    case 6:
      str << "tensor symm per " << restypestring << ":\t" << timeset << "\t" << fileset << "\t"
          << name << "\t" << filename << "\n";
      break;
    case 3:
    case 2:
      str << "vector per " << restypestring << ":\t" << timeset << "\t" << fileset << "\t" << name
          << "\t" << filename << "\n";
      break;
    case 1:
      str << "scalar per " << restypestring << ":\t" << timeset << "\t" << fileset << "\t" << name
          << "\t" << filename << "\n";
      break;
    default:
      FOUR_C_THROW("unknown number of dof per node");
  };
  return str.str();
}


/*----------------------------------------------------------------------*/
/*!
  \brief create string for one TIME set in the case file
*/
/*----------------------------------------------------------------------*/
std::string EnsightWriter::get_time_section_string(
    const int timeset, const std::vector<double>& times) const
{
  std::stringstream s;
  s << "time set:\t\t" << timeset << "\n"
    << "number of steps:\t" << times.size() << "\ntime values: ";
  for (unsigned i = 0; i < times.size(); ++i)
  {
    s << std::setprecision(16) << times[i] << " ";
    if (i % 8 == 0 && i != 0)
    {
      s << "\n";
    }
  }
  s << "\n";
  return s.str();
}


/*----------------------------------------------------------------------*/
/*!
  \brief create string for the TIME section in the case file
*/
/*----------------------------------------------------------------------*/
std::string EnsightWriter::get_time_section_string_from_timesets(
    const std::map<std::string, std::vector<double>>& timesetmap) const
{
  std::stringstream s;
  std::map<std::string, std::vector<double>>::const_iterator timeset;
  std::set<int> donetimesets;

  for (timeset = timesetmap.begin(); timeset != timesetmap.end(); ++timeset)
  {
    std::string key = timeset->first;
    std::map<std::string, int>::const_iterator entry = timesetnumbermap_.find(key);
    if (entry == timesetnumbermap_.end()) FOUR_C_THROW("key not found!");
    const int timesetnumber = entry->second;
    const std::vector<double> soltimes = timeset->second;
    if (donetimesets.find(timesetnumber) == donetimesets.end())  // do not write redundant time sets
    {
      donetimesets.insert(timesetnumber);
      std::string outstring = get_time_section_string(timesetnumber, soltimes);
      s << outstring << std::endl;
    }
  }
  return s.str();
}

/*----------------------------------------------------------------------*/
/*!
  \brief create string for the FILE section in the case file
*/
/*----------------------------------------------------------------------*/
std::string EnsightWriter::get_file_section_string_from_filesets(
    const std::map<std::string, std::vector<int>>& filesetmap) const
{
  std::stringstream s;
  std::map<std::string, std::vector<int>>::const_iterator fileset;

  // print filesets in increasing numbering, starting with "geo"

  std::map<std::string, int>::const_iterator entry = filesetnumbermap_.find("geo");
  if (entry == filesetnumbermap_.end()) FOUR_C_THROW("key 'geo' not found!");
  const int setnumber = entry->second;
  if (setnumber != 1) FOUR_C_THROW("geometry file must have file set number 1");
  fileset = filesetmap.find("geo");
  std::vector<int> stepsperfile = fileset->second;
  s << "file set:\t\t" << setnumber << "\n";
  if (stepsperfile.size() == 1)
  {
    s << "number of steps:\t" << stepsperfile[0] << "\n\n";
  }
  else
  {
    for (unsigned int j = 0; j < stepsperfile.size(); ++j)
    {
      s << "filename index:\t" << 1 + j << "\n";
      s << "number of steps:\t" << stepsperfile[j] << "\n";
    }
    s << "\n";
  }

  for (fileset = filesetmap.begin(); fileset != filesetmap.end(); ++fileset)
  {
    std::string key = fileset->first;
    // skip geometry file since it was already considered above!
    if (key == "geo") continue;

    std::map<std::string, int>::const_iterator entry = filesetnumbermap_.find(key);
    if (entry == filesetnumbermap_.end()) FOUR_C_THROW("key not found!");
    const int setnumber = entry->second;
    std::vector<int> stepsperfile = fileset->second;
    s << "file set:\t\t" << setnumber << "\n";
    if (stepsperfile.size() == 1)
    {
      s << "number of steps:\t" << stepsperfile[0] << "\n\n";
    }
    else
    {
      for (unsigned int j = 0; j < stepsperfile.size(); ++j)
      {
        s << "filename index:\t" << 1 + j << "\n";
        s << "number of steps:\t" << stepsperfile[j] << "\n";
      }
      s << "\n";
    }
  }
  return s.str();
}

/*----------------------------------------------------------------------*/
/*
    Write the coordinates for a Polynomial discretization
    The coordinates of the visualisation points (i.e. the corner
    nodes of elements displayed in paraview) are just the node
    coordinates of the nodes in the discretization.
*/
/*----------------------------------------------------------------------*/
void EnsightWriter::write_coordinates_for_polynomial_shapefunctions(std::ofstream& geofile,
    Core::FE::Discretization& dis, std::shared_ptr<Core::LinAlg::Map>& proc0map)
{
  using namespace FourC;

  // refcountpointer to vector of all coordinates
  // distributed among all procs
  std::shared_ptr<Core::LinAlg::MultiVector<double>> nodecoords;

  const Core::LinAlg::Map* nodemap = dis.node_row_map();
  const int numnp = nodemap->num_my_elements();
  nodecoords = std::make_shared<Core::LinAlg::MultiVector<double>>(*nodemap, 3);

  // loop over the nodes on this proc and store the coordinate information
  for (int inode = 0; inode < numnp; inode++)
  {
    int gid = nodemap->gid(inode);
    const Core::Nodes::Node* actnode = dis.g_node(gid);
    auto x = actnode->x();
    for (size_t isd = 0; isd < x.size(); ++isd)
    {
      nodecoords->replace_local_value(inode, isd, x[isd]);
    }
  }

  // put all coordinate information on proc 0
  proc0map = Core::LinAlg::allreduce_e_map(*nodemap, 0);

  // import my new values (proc0 gets everything, other procs empty)
  Core::LinAlg::Import proc0importer(*proc0map, *nodemap);
  Core::LinAlg::MultiVector<double> allnodecoords(*proc0map, 3);
  allnodecoords.import(*nodecoords, proc0importer, Insert);

  // write the node coordinates (only proc 0)
  // ensight format requires x_1 .. x_n, y_1 .. y_n, z_1 ... z_n
  // this is fulfilled automatically due to Core::LinAlg::MultiVector<double> usage (columnwise
  // writing data)
  if (myrank_ == 0)
  {
    double* coords = allnodecoords.get_values();
    int numentries = (3 * (allnodecoords.global_length()));
    FOUR_C_ASSERT(numentries == (3 * nodemap->num_global_elements()),
        "proc 0 has not all of the node coordinates");
    if (nodeidgiven_)
    {
      // first write node global ids (default)
      for (int inode = 0; inode < proc0map->num_global_elements(); ++inode)
      {
        write(geofile, proc0map->gid(inode) + 1);
        // gid+1 delivers the node numbering of the input file starting with 1
      }
    }
    // now write the coordinate information
    for (int i = 0; i < numentries; ++i)
    {
      write(geofile, static_cast<float>(coords[i]));
    }
  }

  return;
}

FOUR_C_NAMESPACE_CLOSE

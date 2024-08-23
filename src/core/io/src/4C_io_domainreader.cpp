/*----------------------------------------------------------------------*/
/*! \file

\brief Read domain sections of dat files.

\level 0


*/
/*----------------------------------------------------------------------*/

#include "4C_io_domainreader.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element_definition.hpp"
#include "4C_io_gridgenerator.hpp"
#include "4C_io_pstream.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_rebalance_print.hpp"
#include "4C_utils_string.hpp"

#include <Teuchos_Time.hpp>

#include <algorithm>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  /// forward declarations
  void broadcast_input_data_to_all_procs(
      Teuchos::RCP<Epetra_Comm> comm, Core::IO::GridGenerator::RectangularCuboidInputs& inputData);

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  DomainReader::DomainReader(Teuchos::RCP<Core::FE::Discretization> dis,
      const Core::IO::DatFileReader& reader, std::string sectionname)
      : name_(dis->name()),
        reader_(reader),
        comm_(reader.get_comm()),
        sectionname_(sectionname),
        dis_(dis)
  {
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void DomainReader::create_partitioned_mesh(int nodeGIdOfFirstNewNode) const
  {
    const int myrank = comm_->MyPID();

    Teuchos::Time time("", true);

    if (!reader_.my_output_flag() && myrank == 0)
      Core::IO::cout << "Entering domain generation mode for " << name_
                     << " discretization ...\nCreate and partition elements      in...."
                     << Core::IO::endl;

    GridGenerator::RectangularCuboidInputs inputData =
        DomainReader::read_rectangular_cuboid_input_data();
    inputData.node_gid_of_first_new_node_ = nodeGIdOfFirstNewNode;

    Core::IO::GridGenerator::create_rectangular_cuboid_discretization(
        *dis_, inputData, static_cast<bool>(reader_.my_output_flag()));

    if (!myrank && reader_.my_output_flag() == 0)
      Core::IO::cout << "............................................... " << std::setw(10)
                     << std::setprecision(5) << std::scientific << time.totalElapsedTime(true)
                     << " secs" << Core::IO::endl;

    return;
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  GridGenerator::RectangularCuboidInputs DomainReader::read_rectangular_cuboid_input_data() const
  {
    Core::IO::GridGenerator::RectangularCuboidInputs inputData;
    // all reading is done on proc 0
    if (comm_->MyPID() == 0)
    {
      // open input file at the right position
      std::string inputFileName = reader_.my_inputfile_name();
      std::ifstream file(inputFileName.c_str());
      std::ifstream::pos_type pos = reader_.excluded_section_position(sectionname_);
      if (pos != std::ifstream::pos_type(-1))
      {
        file.seekg(pos);

        // read domain info
        std::string line;
        while (getline(file, line))
        {
          // remove comments, trailing and leading whitespaces
          // compact internal whitespaces
          line = Core::UTILS::strip_comment(line);

          // line is now empty
          if (line.size() == 0) continue;

          if (line.find("--") == 0)
          {
            break;
          }
          else
          {
            std::istringstream t;
            t.str(line);
            std::string key;
            t >> key;
            if (key == "LOWER_BOUND")
              t >> inputData.bottom_corner_point_[0] >> inputData.bottom_corner_point_[1] >>
                  inputData.bottom_corner_point_[2];
            else if (key == "UPPER_BOUND")
              t >> inputData.top_corner_point_[0] >> inputData.top_corner_point_[1] >>
                  inputData.top_corner_point_[2];
            else if (key == "INTERVALS")
              t >> inputData.interval_[0] >> inputData.interval_[1] >> inputData.interval_[2];
            else if (key == "ROTATION")
              t >> inputData.rotation_angle_[0] >> inputData.rotation_angle_[1] >>
                  inputData.rotation_angle_[2];
            else if (key == "ELEMENTS")
            {
              t >> inputData.elementtype_ >> inputData.distype_;
              getline(t, inputData.elearguments_);
            }
            else if (key == "PARTITION")
            {
              std::string tmp;
              t >> tmp;
              std::transform(tmp.begin(), tmp.end(), tmp.begin(),
                  [](unsigned char c) { return std::tolower(c); });
              if (tmp == "auto")
                inputData.autopartition_ = true;
              else if (tmp == "structured")
                inputData.autopartition_ = false;
              else
                FOUR_C_THROW(
                    "Invalid argument for PARTITION in DOMAIN reader. Valid options are \"auto\" "
                    "and \"structured\".");
            }
            else
              FOUR_C_THROW("Unknown Key in DOMAIN section");
          }
        }
      }
      else
      {
        FOUR_C_THROW("No DOMAIN specified but box geometry selected!");
      }
    }

    // broadcast if necessary
    if (comm_->NumProc() > 1)
    {
      IO::broadcast_input_data_to_all_procs(comm_, inputData);
    }

    return inputData;
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void broadcast_input_data_to_all_procs(
      Teuchos::RCP<Epetra_Comm> comm, Core::IO::GridGenerator::RectangularCuboidInputs& inputData)
  {
    const int myrank = comm->MyPID();

    std::vector<char> data;
    if (myrank == 0)
    {
      Core::Communication::PackBuffer buffer;
      Core::Communication::ParObject::add_to_pack<double, 3>(
          buffer, inputData.bottom_corner_point_);
      Core::Communication::ParObject::add_to_pack<double, 3>(buffer, inputData.top_corner_point_);
      Core::Communication::ParObject::add_to_pack<int, 3>(buffer, inputData.interval_);
      Core::Communication::ParObject::add_to_pack<double, 3>(buffer, inputData.rotation_angle_);
      Core::Communication::ParObject::add_to_pack(
          buffer, static_cast<int>(inputData.autopartition_));
      Core::Communication::ParObject::add_to_pack(buffer, inputData.elementtype_);
      Core::Communication::ParObject::add_to_pack(buffer, inputData.distype_);
      Core::Communication::ParObject::add_to_pack(buffer, inputData.elearguments_);
      std::swap(data, buffer());
    }

    ssize_t data_size = data.size();
    comm->Broadcast(&data_size, 1, 0);
    if (myrank != 0) data.resize(data_size, 0);
    comm->Broadcast(data.data(), data.size(), 0);

    Communication::UnpackBuffer buffer(data);
    if (myrank != 0)
    {
      Core::Communication::ParObject::extract_from_pack<double, 3>(
          buffer, inputData.bottom_corner_point_);
      Core::Communication::ParObject::extract_from_pack<double, 3>(
          buffer, inputData.top_corner_point_);
      Core::Communication::ParObject::extract_from_pack<int, 3>(buffer, inputData.interval_);
      Core::Communication::ParObject::extract_from_pack<double, 3>(
          buffer, inputData.rotation_angle_);
      int autopartitionInteger;
      Core::Communication::ParObject::extract_from_pack(buffer, autopartitionInteger);
      inputData.autopartition_ = autopartitionInteger;
      Core::Communication::ParObject::extract_from_pack(buffer, inputData.elementtype_);
      Core::Communication::ParObject::extract_from_pack(buffer, inputData.distype_);
      Core::Communication::ParObject::extract_from_pack(buffer, inputData.elearguments_);
    }
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void DomainReader::complete() const
  {
    const int myrank = comm_->MyPID();

    Teuchos::Time time("", true);

    if (!myrank && !reader_.my_output_flag())
      Core::IO::cout << "Complete discretization " << std::left << std::setw(16) << name_
                     << " in...." << Core::IO::flush;

    int err = dis_->fill_complete(false, false, false);
    if (err) FOUR_C_THROW("dis_->fill_complete() returned %d", err);

    if (!myrank && !reader_.my_output_flag())
      Core::IO::cout << time.totalElapsedTime(true) << " secs" << Core::IO::endl;

    Core::Rebalance::UTILS::print_parallel_distribution(*dis_);
  }

}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

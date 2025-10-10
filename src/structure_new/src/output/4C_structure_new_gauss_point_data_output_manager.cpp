// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_gauss_point_data_output_manager.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"


FOUR_C_NAMESPACE_OPEN

Solid::ModelEvaluator::GaussPointDataOutputManager::GaussPointDataOutputManager(
    Inpar::Solid::GaussPointDataOutputType output_type)
    : output_type_(output_type),
      max_num_gp_(0),
      data_nodes_({}),
      data_nodes_count_({}),
      data_element_center_({}),
      data_gauss_point_({}),
      quantities_({})
{
}

void Solid::ModelEvaluator::GaussPointDataOutputManager::add_quantity_if_not_existent(
    const std::string& name, int size)
{
  const auto item = quantities_.find(name);
  if (item != quantities_.end())
  {
    if (item->second != size)
    {
      FOUR_C_THROW(
          "The quantity {} is already registered, but with a different size ({} vs. {}). This is "
          "fatal!",
          name.c_str(), size, item->second);
    }
  }
  else
  {
    if (name.find(MPI_DELIMITER) != std::string::npos)
    {
      FOUR_C_THROW(
          "The quantity name {} for Gauss Point VTK runtime output contains the delimiter {} that "
          "is used for MPI communication. This is not allowed.",
          name.c_str(), MPI_DELIMITER);
    }
    quantities_.insert({name, size});
  }
}

void Solid::ModelEvaluator::GaussPointDataOutputManager::merge_quantities(
    const std::unordered_map<std::string, int>& quantities)
{
  for (const auto& name_and_size : quantities)
  {
    const std::string& name = name_and_size.first;
    const int size = name_and_size.second;

    add_quantity_if_not_existent(name, size);
  }
}

void Solid::ModelEvaluator::GaussPointDataOutputManager::add_element_number_of_gauss_points(
    const int numgp)
{
  if (numgp > max_num_gp_)
  {
    max_num_gp_ = numgp;
  }
}

void Solid::ModelEvaluator::GaussPointDataOutputManager::prepare_data(
    const Core::LinAlg::Map& node_col_map, const Core::LinAlg::Map& element_row_map)
{
  switch (output_type_)
  {
    case Inpar::Solid::GaussPointDataOutputType::nodes:
      prepare_nodal_data_vectors(node_col_map);
      break;
    case Inpar::Solid::GaussPointDataOutputType::element_center:
      prepare_element_center_data_vectors(element_row_map);
      break;
    case Inpar::Solid::GaussPointDataOutputType::gauss_points:
      prepare_gauss_point_data_vectors(element_row_map);
      break;
    case Inpar::Solid::GaussPointDataOutputType::none:
      FOUR_C_THROW("Your Gauss point data output type is none, so you don't need to prepare data!");
    default:
      FOUR_C_THROW("Unknown Gauss point data output type");
  }
}

void Solid::ModelEvaluator::GaussPointDataOutputManager::prepare_nodal_data_vectors(
    const Core::LinAlg::Map& node_col_map)
{
  for (const auto& name_and_size : quantities_)
  {
    const std::string& name = name_and_size.first;
    const int size = name_and_size.second;

    data_nodes_[name] =
        std::make_shared<Core::LinAlg::MultiVector<double>>(node_col_map, size, true);
    data_nodes_count_[name] = std::make_shared<Core::LinAlg::Vector<int>>(node_col_map, true);
  }
}

void Solid::ModelEvaluator::GaussPointDataOutputManager::prepare_element_center_data_vectors(
    const Core::LinAlg::Map& element_col_map)
{
  for (const auto& name_and_size : quantities_)
  {
    const std::string& name = name_and_size.first;
    const int size = name_and_size.second;

    data_element_center_[name] =
        std::make_shared<Core::LinAlg::MultiVector<double>>(element_col_map, size, true);
  }
}

void Solid::ModelEvaluator::GaussPointDataOutputManager::prepare_gauss_point_data_vectors(
    const Core::LinAlg::Map& element_col_map)
{
  for (const auto& name_and_size : quantities_)
  {
    const std::string& name = name_and_size.first;
    const int size = name_and_size.second;

    data_gauss_point_.emplace(
        name, std::vector<std::shared_ptr<Core::LinAlg::MultiVector<double>>>());
    std::vector<std::shared_ptr<Core::LinAlg::MultiVector<double>>>& gp_data =
        data_gauss_point_[name];

    gp_data.resize(max_num_gp_);
    for (auto& data_i : gp_data)
    {
      data_i = std::make_shared<Core::LinAlg::MultiVector<double>>(element_col_map, size, true);
    }
  }
}

void Solid::ModelEvaluator::GaussPointDataOutputManager::post_evaluate()
{
  if (output_type_ == Inpar::Solid::GaussPointDataOutputType::nodes)
  {
    // divide nodal quantities by the nodal count
    for (const auto& name_and_size : quantities_)
    {
      const std::string& name = name_and_size.first;

      Core::LinAlg::MultiVector<double>& nodal_data = *data_nodes_[name];
      const Core::LinAlg::Vector<int>& nodal_count = *data_nodes_count_[name];

      for (int col = 0; col < nodal_data.num_vectors(); ++col)
      {
        auto& data_item = nodal_data(col);

        for (int i = 0; i < data_item.local_length(); ++i)
        {
          if (nodal_count.get_local_values()[i] != 0)
          {
            data_item.get_values()[i] /= nodal_count.get_local_values()[i];
          }
        }
      }
    }
  }
}

void Solid::ModelEvaluator::GaussPointDataOutputManager::distribute_quantities(MPI_Comm comm)
{
  const Core::Communication::Exporter exporter(comm);

  int max_quantities = quantities_.size();
  int max_quantities_local = quantities_.size();
  max_quantities = Core::Communication::max_all(max_quantities_local, comm);

  if (max_quantities == 0)
  {
    // Nothing to distribute
    return;
  }

  // Communicate max number of Gauss points
  int max_num_gp_local = max_num_gp_;
  max_num_gp_ = Core::Communication::max_all(max_num_gp_local, comm);

  // Collect all quantities on proc 0
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    // receive everything from all other procs
    for (int i = 1; i < Core::Communication::num_mpi_ranks(comm); ++i)
    {
      std::unique_ptr<std::unordered_map<std::string, int>> received_quantities =
          receive_quantities_from_proc(exporter, i);

      merge_quantities(*received_quantities);
    }
  }
  else
  {
    send_my_quantities_to_proc(exporter, 0);
  }

  // Broadcast merged quantities to every proc
  broadcast_my_quantities(exporter);
}

void Solid::ModelEvaluator::GaussPointDataOutputManager::send_my_quantities_to_proc(
    const Core::Communication::Exporter& exporter, int to_proc) const
{
  // Pack quantities
  std::vector<char> sdata;
  pack_my_quantities(sdata);

  MPI_Request request;
  exporter.i_send(Core::Communication::my_mpi_rank(exporter.get_comm()), 0, sdata.data(),
      sdata.size(), MPI_TAG, request);
  exporter.wait(request);
}

std::unique_ptr<std::unordered_map<std::string, int>>
Solid::ModelEvaluator::GaussPointDataOutputManager::receive_quantities_from_proc(
    const Core::Communication::Exporter& exporter, int from_proc) const
{
  std::vector<char> rdata;
  int size;
  exporter.receive(from_proc, MPI_TAG, rdata, size);

  auto quantities = std::unique_ptr<std::unordered_map<std::string, int>>(
      new std::unordered_map<std::string, int>());

  Core::Communication::UnpackBuffer buffer(rdata);
  unpack_quantities(buffer, *quantities);

  return quantities;
}

void Solid::ModelEvaluator::GaussPointDataOutputManager::broadcast_my_quantities(
    const Core::Communication::Exporter& exporter)
{
  std::vector<char> data;
  if (Core::Communication::my_mpi_rank(exporter.get_comm()) == 0)
  {
    pack_my_quantities(data);
  }

  exporter.broadcast(0, data, MPI_TAG);

  if (Core::Communication::my_mpi_rank(exporter.get_comm()) != 0)
  {
    std::unordered_map<std::string, int> received_quantities{};
    Core::Communication::UnpackBuffer buffer(data);
    unpack_quantities(buffer, received_quantities);

    merge_quantities(received_quantities);
  }
}

void Solid::ModelEvaluator::GaussPointDataOutputManager::pack_my_quantities(
    std::vector<char>& data) const
{
  Core::Communication::PackBuffer packBuffer;
  add_to_pack(packBuffer, quantities_);
  std::swap(data, packBuffer());
}

void Solid::ModelEvaluator::GaussPointDataOutputManager::unpack_quantities(
    Core::Communication::UnpackBuffer& buffer,
    std::unordered_map<std::string, int>& quantities) const
{
  extract_from_pack(buffer, quantities);
}

FOUR_C_NAMESPACE_CLOSE

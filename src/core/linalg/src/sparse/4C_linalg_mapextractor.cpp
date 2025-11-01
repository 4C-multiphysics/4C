// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_mapextractor.hpp"

#include "4C_linalg_utils_sparse_algebra_create.hpp"

#include <Teuchos_getConst.hpp>

#include <cmath>
#include <numeric>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::LinAlg::MultiMapExtractor::MultiMapExtractor(const Core::LinAlg::Map& fullmap,
    const std::vector<std::shared_ptr<const Core::LinAlg::Map>>& maps)
{
  setup(fullmap, maps);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::LinAlg::MultiMapExtractor::setup(const Core::LinAlg::Map& fullmap,
    const std::vector<std::shared_ptr<const Core::LinAlg::Map>>& maps)
{
  fullmap_ = std::make_shared<Core::LinAlg::Map>(fullmap);
  maps_ = maps;

  importer_.resize(maps_.size());
  for (unsigned i = 0; i < importer_.size(); ++i)
  {
    if (maps_[i] != nullptr)
    {
      importer_[i] = std::make_shared<Core::LinAlg::Import>(*maps_[i], *fullmap_);
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::LinAlg::MultiMapExtractor::check_for_valid_map_extractor() const
{
  if (maps_.size() == 0)
  {
    FOUR_C_THROW("no maps_ available");
  }

  for (unsigned i = 0; i < maps_.size(); ++i)
  {
    if (maps_[i] != nullptr)
    {
      if (maps_[i]->get_data_ptr() == nullptr)
      {
        FOUR_C_THROW("Got zero data pointer on setup of block {} of maps_\n", i);
      }
      if (not maps_[i]->unique_gids())
      {
        FOUR_C_THROW("map {} not unique", i);
      }
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Core::LinAlg::MultiMapExtractor::merge_maps(
    const std::vector<std::shared_ptr<const Core::LinAlg::Map>>& maps)
{
  if (maps.size() == 0) FOUR_C_THROW("no maps to merge");
  for (unsigned i = 0; i < maps.size(); ++i)
  {
    if (maps[i] == nullptr) FOUR_C_THROW("can not merge extractor with null maps");
    if (not maps[i]->unique_gids()) FOUR_C_THROW("map {} not unique", i);
  }
  std::set<int> mapentries;
  for (unsigned i = 0; i < maps.size(); ++i)
  {
    const Core::LinAlg::Map& map = *maps[i];
    std::copy(map.my_global_elements(), map.my_global_elements() + map.num_my_elements(),
        std::inserter(mapentries, mapentries.begin()));
  }
  return Core::LinAlg::create_map(
      std::vector<int>(mapentries.begin(), mapentries.end()), maps[0]->get_comm());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Core::LinAlg::MultiMapExtractor::merge_maps_keep_order(
    const std::vector<std::shared_ptr<const Core::LinAlg::Map>>& maps)
{
  if (maps.empty()) FOUR_C_THROW("no maps to merge");

  // sanity checks
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    if (maps[i] == nullptr) FOUR_C_THROW("can not merge extractor with null maps");
    if (not maps[i]->unique_gids()) FOUR_C_THROW("map {} not unique", i);
  }

  // collect gids
  std::vector<int> gids;
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    const Core::LinAlg::Map& map = *maps[i];
    for (int j = 0; j < map.num_my_elements(); ++j) gids.push_back(map.gid(j));
  }

  // build combined map
  std::shared_ptr<Core::LinAlg::Map> fullmap =
      std::make_shared<Core::LinAlg::Map>(-1, gids.size(), gids.data(), 0, maps[0]->get_comm());
  return fullmap;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Core::LinAlg::MultiMapExtractor::intersect_maps(
    const std::vector<std::shared_ptr<const Core::LinAlg::Map>>& maps)
{
  if (maps.size() == 0) FOUR_C_THROW("no maps to intersect");
  for (unsigned i = 0; i < maps.size(); ++i)
  {
    if (maps[i] == nullptr) FOUR_C_THROW("can not intersect extractor with null maps");
    if (not maps[i]->unique_gids()) FOUR_C_THROW("map {} not unique", i);
  }
  std::set<int> mapentries(
      maps[0]->my_global_elements(), maps[0]->my_global_elements() + maps[0]->num_my_elements());
  for (unsigned i = 1; i < maps.size(); ++i)
  {
    const Core::LinAlg::Map& map = *maps[i];
    std::set<int> newset;
    int numele = map.num_my_elements();
    int* ele = map.my_global_elements();
    for (int j = 0; j < numele; ++j)
    {
      if (mapentries.find(ele[j]) != mapentries.end())
      {
        newset.insert(ele[j]);
      }
    }
    std::swap(mapentries, newset);
  }
  return Core::LinAlg::create_map(
      std::vector<int>(mapentries.begin(), mapentries.end()), maps[0]->get_comm());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Core::LinAlg::MultiMapExtractor::extract_vector(
    const Core::LinAlg::Vector<double>& full, int block) const
{
  if (maps_[block] == nullptr) FOUR_C_THROW("null map at block {}", block);
  std::shared_ptr<Core::LinAlg::Vector<double>> vec =
      std::make_shared<Core::LinAlg::Vector<double>>(*maps_[block]);
  extract_vector(full, block, *vec);
  return vec;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> Core::LinAlg::MultiMapExtractor::extract_vector(
    const Core::LinAlg::MultiVector<double>& full, int block) const
{
  if (maps_[block] == nullptr) FOUR_C_THROW("null map at block {}", block);
  std::shared_ptr<Core::LinAlg::MultiVector<double>> vec =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*maps_[block], full.num_vectors());
  extract_vector(full, block, *vec);
  return vec;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::LinAlg::MultiMapExtractor::extract_vector(const Core::LinAlg::MultiVector<double>& full,
    int block, Core::LinAlg::MultiVector<double>& partial) const
{
  if (maps_[block] == nullptr) FOUR_C_THROW("null map at block {}", block);

  partial.import(full, *importer_[block], Insert);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Core::LinAlg::MultiMapExtractor::insert_vector(
    const Core::LinAlg::Vector<double>& partial, int block) const
{
  std::shared_ptr<Core::LinAlg::Vector<double>> full =
      std::make_shared<Core::LinAlg::Vector<double>>(*fullmap_);
  insert_vector(partial, block, *full);
  return full;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> Core::LinAlg::MultiMapExtractor::insert_vector(
    const Core::LinAlg::MultiVector<double>& partial, int block) const
{
  std::shared_ptr<Core::LinAlg::MultiVector<double>> full =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*fullmap_, partial.num_vectors());
  insert_vector(partial, block, *full);
  return full;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::LinAlg::MultiMapExtractor::insert_vector(
    const Core::LinAlg::MultiVector<double>& partial, int block,
    Core::LinAlg::MultiVector<double>& full) const
{
  if (maps_[block] == nullptr) FOUR_C_THROW("null map at block {}", block);

  full.export_to(partial, *importer_[block], Insert);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::LinAlg::MultiMapExtractor::add_vector(const Core::LinAlg::MultiVector<double>& partial,
    int block, Core::LinAlg::MultiVector<double>& full, double scale) const
{
  std::shared_ptr<Core::LinAlg::MultiVector<double>> v = extract_vector(full, block);
  if (not v->get_map().same_as(partial.get_map()))
    FOUR_C_THROW("The maps of the vectors must be the same!");
  v->update(scale, partial, 1.0);
  insert_vector(*v, block, full);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::LinAlg::MultiMapExtractor::put_scalar(
    Core::LinAlg::Vector<double>& full, int block, double scalar) const
{
  const Core::LinAlg::Map& bm = *map(block);
  const Core::LinAlg::Map& fm = *full_map();

  int numv = bm.num_my_elements();
  int* v = bm.my_global_elements();

  for (int i = 0; i < numv; ++i)
  {
    int lid = fm.lid(v[i]);
    if (lid == -1) FOUR_C_THROW("maps do not match");
    full.get_values()[lid] = scalar;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Core::LinAlg::MultiMapExtractor::norm2(
    const Core::LinAlg::Vector<double>& full, int block) const
{
  const Core::LinAlg::Map& bm = *map(block);
  const Core::LinAlg::Map& fm = *full_map();

  int numv = bm.num_my_elements();
  int* v = bm.my_global_elements();

  double local_norm = 0;

  for (int i = 0; i < numv; ++i)
  {
    int lid = fm.lid(v[i]);
    if (lid == -1) FOUR_C_THROW("maps do not match");
    double value = full.local_values_as_span()[lid];
    local_norm += value * value;
  }

  double global_norm = 0;
  global_norm = Core::Communication::sum_all(local_norm, fm.get_comm());
  return std::sqrt(global_norm);
}


/*----------------------------------------------------------------------*
 | Scale one block only                                      fang 08/16 |
 *----------------------------------------------------------------------*/
void Core::LinAlg::MultiMapExtractor::scale(
    Core::LinAlg::Vector<double>& full, int block, double scalar) const
{
  const Core::LinAlg::Map& bm = *map(block);
  const Core::LinAlg::Map& fm = *full_map();

  int numv = bm.num_my_elements();
  int* v = bm.my_global_elements();

  for (int i = 0; i < numv; ++i)
  {
    int lid = fm.lid(v[i]);
    if (lid == -1) FOUR_C_THROW("maps do not match");
    full.get_values()[lid] *= scalar;
  }
}


/*----------------------------------------------------------------------*
 | Scale one block only                                      fang 08/16 |
 *----------------------------------------------------------------------*/
void Core::LinAlg::MultiMapExtractor::scale(
    Core::LinAlg::MultiVector<double>& full, int block, double scalar) const
{
  for (int i = 0; i < full.num_vectors(); ++i) scale(full(i), block, scalar);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::LinAlg::MapExtractor::MapExtractor(const Core::LinAlg::Map& fullmap,
    std::shared_ptr<const Core::LinAlg::Map> condmap,
    std::shared_ptr<const Core::LinAlg::Map> othermap)
{
  setup(fullmap, condmap, othermap);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::LinAlg::MapExtractor::MapExtractor(const Core::LinAlg::Map& fullmap,
    std::shared_ptr<const Core::LinAlg::Map> partialmap, bool iscondmap)
{
  // initialise other DOFs by inserting all DOFs of full map
  std::set<int> othergids;
  const int* fullgids = fullmap.my_global_elements();
  copy(fullgids, fullgids + fullmap.num_my_elements(), inserter(othergids, othergids.begin()));

  // throw away all DOFs which are in condmap
  if (partialmap->num_my_elements() > 0)
  {
    const int* condgids = partialmap->my_global_elements();
    for (int lid = 0; lid < partialmap->num_my_elements(); ++lid) othergids.erase(condgids[lid]);
  }

  // create (non-overlapping) othermap for non-condmap DOFs

  std::shared_ptr<Core::LinAlg::Map> othermap = Core::LinAlg::create_map(
      std::vector<int>(othergids.begin(), othergids.end()), fullmap.get_comm());

  // create the extractor based on choice 'iscondmap'
  if (iscondmap)
    setup(fullmap, partialmap, othermap);
  else
    setup(fullmap, othermap, partialmap);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::LinAlg::MapExtractor::setup(const Core::LinAlg::Map& full_map,
    const std::shared_ptr<const Core::LinAlg::Map>& cond_map,
    const std::shared_ptr<const Core::LinAlg::Map>& other_map)
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps;
  maps.push_back(other_map);
  maps.push_back(cond_map);
  MultiMapExtractor::setup(full_map, maps);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::LinAlg::MapExtractor::setup(
    const Core::LinAlg::Map& full_map, const std::shared_ptr<const Core::LinAlg::Map>& cond_map)
{
  std::span<const int> full_gids(full_map.my_global_elements(), full_map.num_my_elements());
  std::span<const int> cond_gids(cond_map->my_global_elements(), cond_map->num_my_elements());

  // The set_difference algorithm requires the input ranges to be sorted.
  FOUR_C_ASSERT(std::ranges::is_sorted(full_gids), "Internal error: GIDs in map not sorted.");
  FOUR_C_ASSERT(std::ranges::is_sorted(cond_gids), "Internal error: GIDs in map not sorted.");

  std::vector<int> other_gids;
  std::ranges::set_difference(full_gids, cond_gids, std::back_inserter(other_gids));

  auto other_map = Core::LinAlg::create_map(other_gids, full_map.get_comm());

  setup(full_map, cond_map, other_map);
}

FOUR_C_NAMESPACE_CLOSE

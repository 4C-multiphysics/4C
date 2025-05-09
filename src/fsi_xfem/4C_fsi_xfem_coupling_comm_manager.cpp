// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_xfem_coupling_comm_manager.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_condition_selector.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

#include <iostream>
#include <string>

FOUR_C_NAMESPACE_OPEN


/*-----------------------------------------------------------------------------------------*
| constructor - simplified for 1 discretization to couple (public)             ager 03/2016|
*-----------------------------------------------------------------------------------------*/
XFEM::CouplingCommManager::CouplingCommManager(std::shared_ptr<const Core::FE::Discretization> dis0,
    std::string cond_name, int startdim, int enddim)
    : cond_name_(cond_name), startdim_(startdim), enddim_(enddim)
{
  std::map<int, std::shared_ptr<const Core::FE::Discretization>> dis_map;
  dis_map.insert(std::pair<int, std::shared_ptr<const Core::FE::Discretization>>(0, dis0));

  setup(dis_map);
}

/*-----------------------------------------------------------------------------------------*
| constructor - simplified for 2 discretizations to couple (public)            ager 03/2016|
*-----------------------------------------------------------------------------------------*/
XFEM::CouplingCommManager::CouplingCommManager(std::shared_ptr<const Core::FE::Discretization> dis0,
    std::shared_ptr<const Core::FE::Discretization> dis1, std::string cond_name, int startdim,
    int enddim)
    : cond_name_(cond_name), startdim_(startdim), enddim_(enddim)
{
  std::map<int, std::shared_ptr<const Core::FE::Discretization>> dis_map;
  dis_map.insert(std::pair<int, std::shared_ptr<const Core::FE::Discretization>>(0, dis0));
  dis_map.insert(std::pair<int, std::shared_ptr<const Core::FE::Discretization>>(1, dis1));

  setup(dis_map);
}

/*-----------------------------------------------------------------------------------------*
| constructor (public)                                                         ager 03/2016|
*-----------------------------------------------------------------------------------------*/
XFEM::CouplingCommManager::CouplingCommManager(
    std::map<int, std::shared_ptr<const Core::FE::Discretization>> dis, std::string cond_name,
    int startdim, int enddim)
    : cond_name_(cond_name), startdim_(startdim), enddim_(enddim)
{
  setup(dis);
}


/*--------------------------------------------------------------------------------------------------------------*
| Transfer conditioned part of Vector from discretization A --> B with different transfer types ager
03/2016|
*--------------------------------------------------------------------------------------------------------------*/
void XFEM::CouplingCommManager::insert_vector(const int idxA,
    std::shared_ptr<const Core::LinAlg::Vector<double>> vecA, const int idxB,
    std::shared_ptr<Core::LinAlg::Vector<double>> vecB,
    const CouplingCommManager::TransferType ttype, bool add, double scale)
{
  switch (ttype)
  {
    case CouplingCommManager::full_to_full:
    {
      std::shared_ptr<Core::LinAlg::MultiMapExtractor> mmeb = get_map_extractor(idxB);
      std::shared_ptr<Core::LinAlg::Vector<double>> tmpvec =
          std::make_shared<Core::LinAlg::Vector<double>>(*mmeb->map(1), true);
      insert_vector(idxA, vecA, idxB, tmpvec, CouplingCommManager::full_to_partial, false, scale);
      if (!add)
        mmeb->insert_vector(*tmpvec, 1, *vecB);
      else
        mmeb->add_vector(*tmpvec, 1, *vecB);
      break;
    }
    case CouplingCommManager::full_to_partial:
    {
      insert_vector(idxA, get_map_extractor(idxA)->extract_vector(*vecA, 1), idxB, vecB,
          CouplingCommManager::partial_to_partial, add, scale);
      break;
    }
    case CouplingCommManager::partial_to_full:
    {
      std::shared_ptr<Core::LinAlg::MultiMapExtractor> mmeb = get_map_extractor(idxB);
      std::shared_ptr<Core::LinAlg::Vector<double>> tmpvec =
          std::make_shared<Core::LinAlg::Vector<double>>(*mmeb->map(1), true);
      insert_vector(
          idxA, vecA, idxB, tmpvec, CouplingCommManager::partial_to_partial, false, scale);
      if (!add)
        mmeb->insert_vector(*tmpvec, 1, *vecB);
      else
        mmeb->add_vector(*tmpvec, 1, *vecB);
      break;
    }
    case CouplingCommManager::partial_to_partial:
    {
      if (vecB == nullptr) FOUR_C_THROW("Coupling_Comm_Manager::InsertVector: vecB is nullptr!");
      if (idxA < idxB)  // this Coupling Object is directly stored
      {
        if (!add)
          *vecB = *get_coupling(idxA, idxB)->master_to_slave(*vecA);
        else
          vecB->update(scale, *get_coupling(idxA, idxB)->master_to_slave(*vecA), 1.0);
      }
      else if (idxA > idxB)  // just the inverse Coupling Object is stored
      {
        if (!add)
          *vecB = *get_coupling(idxB, idxA)->slave_to_master(*vecA);
        else
          vecB->update(scale, *get_coupling(idxB, idxA)->slave_to_master(*vecA), 1.0);
      }
      else
      {
        if (!add)
        {
          // *vecB = *vecA; //we don't do any transformation!
          vecB->update(1.0, *vecA, 0.0);
        }
        else
          vecB->update(scale, *vecA, 1.0);
      }
      if (!add && scale != 1.0) vecB->scale(scale);
      break;
    }
    case CouplingCommManager::partial_to_global:
    {
      std::shared_ptr<Core::LinAlg::MultiMapExtractor> mme = get_full_map_extractor();
      std::shared_ptr<Core::LinAlg::Vector<double>> fullvec =
          std::make_shared<Core::LinAlg::Vector<double>>(*mme->map(idxB), true);
      insert_vector(idxA, vecA, idxB, fullvec, CouplingCommManager::partial_to_full, false, scale);
      if (!add)
        mme->insert_vector(*fullvec, idxB, *vecB);
      else
        mme->add_vector(*fullvec, idxB, *vecB);
      break;
    }
    case CouplingCommManager::full_to_global:
    {
      std::shared_ptr<Core::LinAlg::MultiMapExtractor> mme = get_full_map_extractor();
      std::shared_ptr<Core::LinAlg::Vector<double>> fullvec =
          std::make_shared<Core::LinAlg::Vector<double>>(*mme->map(idxB), true);
      insert_vector(idxA, vecA, idxB, fullvec, CouplingCommManager::full_to_full, false, scale);
      if (!add)
        mme->insert_vector(*fullvec, idxB, *vecB);
      else
        mme->add_vector(*fullvec, idxB, *vecB);
      break;
    }
    default:
      FOUR_C_THROW("Coupling_Comm_Manager::InsertVector: Transfer Type not implemented!");
  }
  return;
}

/*----------------------------------------------------------------------------------------------*
| //! Insert a Matrix A into Matrix B (choose type of transfer, add or scaling)     ager 03/2016|
*----------------------------------------------------------------------------------------------*/
bool XFEM::CouplingCommManager::insert_matrix(int transform_id, int idxA,
    const Core::LinAlg::SparseMatrix& matA, int idxB, Core::LinAlg::SparseMatrix& matB,
    const CouplingCommManager::MatrixTransferType mttype, double scale, bool exactmatch,
    bool addmatrix)
{
  switch (mttype)
  {
    case CouplingCommManager::col:
    {
      return get_transform(transform_id)
          ->operator()(matA, matA.range_map(), matA.domain_map(), scale, nullptr,
              get_coupling_converter(idxA, idxB).get(), matB, exactmatch, addmatrix);
      break;
    }
    case CouplingCommManager::row:
    {
      return get_transform(transform_id)
          ->operator()(matA, matA.range_map(), matA.domain_map(), scale,
              get_coupling_converter(idxA, idxB).get(), nullptr, matB, true, addmatrix);
      break;
    }
    case CouplingCommManager::row_and_col:
    {
      return get_transform(transform_id)
          ->operator()(matA, matA.range_map(), matA.domain_map(), scale,
              get_coupling_converter(idxA, idxB).get(), get_coupling_converter(idxA, idxB).get(),
              matB, exactmatch, addmatrix);
      break;
    }
    default:
      FOUR_C_THROW("Coupling_Comm_Manager::InsertMatrix: Matrix Transfer Type not implemented!");
  }
  return true;
}

/*-----------------------------------------------------------------------------------------*
| Setup Coupling_Comm_Manager                                                  ager 06/2016|
*-----------------------------------------------------------------------------------------*/
void XFEM::CouplingCommManager::setup(
    std::map<int, std::shared_ptr<const Core::FE::Discretization>> dis)
{
  if (cond_name_ != "")  // Setup for Communication on Condition
  {
    setup_multi_map_extractors(dis);
    setup_couplings(dis);
  }
  else  // Setup for Communication on full discretization
  {
    setup_full_couplings(dis);       // First Couple full discretizations (from startdim to enddim)
    setup_full_map_extractors(dis);  // Setup Extractors between Couplingpart of discretization and
                                     // full discretization (e.g. pres <==> vel&pres)
  }
  setup_full_extractor(dis);
  return;
}

/*-----------------------------------------------------------------------------------------*
| Setup MultiMapExtractors for all fields                                      ager 03/2016|
*-----------------------------------------------------------------------------------------*/
void XFEM::CouplingCommManager::setup_multi_map_extractors(
    std::map<int, std::shared_ptr<const Core::FE::Discretization>> dis)
{
  for (std::map<int, std::shared_ptr<const Core::FE::Discretization>>::iterator dit = dis.begin();
      dit != dis.end(); ++dit)
  {
    mme_[dit->first] = std::make_shared<Core::LinAlg::MultiMapExtractor>();
    Core::Conditions::setup_extractor(*dit->second, *mme_[dit->first],
        std::vector<Core::Conditions::Selector>{
            Core::Conditions::Selector(cond_name_, startdim_, enddim_)});
  }
}

/*-----------------------------------------------------------------------------------------*
| Setup MultiMapExtractors for all fields                                      ager 03/2016|
*-----------------------------------------------------------------------------------------*/
void XFEM::CouplingCommManager::setup_full_map_extractors(
    std::map<int, std::shared_ptr<const Core::FE::Discretization>> dis)
{
  if (dis.size() < 2)
    FOUR_C_THROW(
        "setup_full_map_extractors: Just extract from discretization to coupled map! (e.g. Fluid "
        "vel "
        "<==> Fluid vel&pres)");

  for (std::map<int, std::shared_ptr<const Core::FE::Discretization>>::iterator dit = dis.begin();
      dit != dis.end(); ++dit)
  {
    std::shared_ptr<Core::LinAlg::MapExtractor> me = std::make_shared<Core::LinAlg::MapExtractor>();
    if (static_cast<std::size_t>(dit->first) < dis.size() - 1)
    {
      std::shared_ptr<Coupling::Adapter::Coupling> coup = get_coupling(dit->first, dit->first + 1);
      me->setup(*dit->second->dof_row_map(), coup->master_dof_map(),
          Core::LinAlg::split_map(*dit->second->dof_row_map(), *coup->master_dof_map()));
    }
    else
    {
      std::shared_ptr<Coupling::Adapter::Coupling> coup = get_coupling(dit->first - 1, dit->first);
      me->setup(*dit->second->dof_row_map(), coup->slave_dof_map(),
          Core::LinAlg::split_map(*dit->second->dof_row_map(), *coup->slave_dof_map()));
    }
    mme_[dit->first] = me;
  }
  return;
}

/*------------------------------------------------------------------------------------------------*
| Setup Couplings between different discretizations with the same condition on in  -- ager 03/2016|
*------------------------------------------------------------------------------------------------*/
void XFEM::CouplingCommManager::setup_couplings(
    std::map<int, std::shared_ptr<const Core::FE::Discretization>> dis)
{
  if (dis.size() < 2) return;

  for (std::map<int, std::shared_ptr<Core::LinAlg::MultiMapExtractor>>::iterator mmealpha =
           mme_.begin();
      mmealpha != mme_.end(); ++mmealpha)
  {
    for (std::map<int, std::shared_ptr<Core::LinAlg::MultiMapExtractor>>::iterator mmebeta =
             mme_.begin();
        mmebeta != mme_.end(); ++mmebeta)
    {
      if ((*mmealpha).first >= (*mmebeta).first)
        continue;  // we  create couplings just for idxa < idxb

      std::pair<int, int> key = std::pair<int, int>((*mmealpha).first, (*mmebeta).first);

      coup_[key] = std::make_shared<Coupling::Adapter::Coupling>();

      std::map<int, std::shared_ptr<const Core::FE::Discretization>>::iterator alphadis =
          dis.find((*mmealpha).first);
      if (alphadis == dis.end())
        FOUR_C_THROW("Couldn't find discretization for key {}", (*mmealpha).first);
      std::map<int, std::shared_ptr<const Core::FE::Discretization>>::iterator betadis =
          dis.find((*mmebeta).first);
      if (betadis == dis.end())
        FOUR_C_THROW("Couldn't find discretization for key {}", (*mmebeta).first);

      coup_[key]->setup_condition_coupling(*(*alphadis).second, (*mmealpha).second->map(1),
          *(*betadis).second, (*mmebeta).second->map(1), cond_name_, enddim_ - startdim_, true);
    }
  }
  return;
}

/*------------------------------------------------------------------------------------------------*
| Setup Couplings between different discretizations full coupling                  -- ager 07/2016|
*------------------------------------------------------------------------------------------------*/
void XFEM::CouplingCommManager::setup_full_couplings(
    std::map<int, std::shared_ptr<const Core::FE::Discretization>> dis)
{
  if (dis.size() < 2) return;

  for (std::size_t idx_a = 0; idx_a < dis.size(); ++idx_a)
  {
    for (std::size_t idx_b = 0; idx_b < dis.size(); ++idx_b)
    {
      if (idx_a >= idx_b) continue;  // we  create couplings just for idxa < idxb

      std::pair<int, int> key = std::pair<int, int>(idx_a, idx_b);

      coup_[key] = std::make_shared<Coupling::Adapter::Coupling>();

      std::map<int, std::shared_ptr<const Core::FE::Discretization>>::iterator alphadis =
          dis.find(idx_a);
      if (alphadis == dis.end()) FOUR_C_THROW("Couldn't find discretization for key {}", idx_a);
      std::map<int, std::shared_ptr<const Core::FE::Discretization>>::iterator betadis =
          dis.find(idx_b);
      if (betadis == dis.end()) FOUR_C_THROW("Couldn't find discretization for key {}", idx_b);

      coup_[key]->setup_coupling(*(*alphadis).second, *(*betadis).second,
          *(*alphadis).second->node_row_map(), *(*betadis).second->node_row_map(),
          enddim_ - startdim_, false);
    }
  }
  return;
}

/*------------------------------------------------------------------------------------------------*
| Setup Full Extractor of all involved discretisations                                ager 03/2016|
*------------------------------------------------------------------------------------------------*/
void XFEM::CouplingCommManager::setup_full_extractor(
    std::map<int, std::shared_ptr<const Core::FE::Discretization>> dis)
{
  if (dis.size() < 2) return;

  std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps;
  for (std::map<int, std::shared_ptr<const Core::FE::Discretization>>::iterator dit = dis.begin();
      dit != dis.end(); ++dit)
  {
    maps.push_back(std::make_shared<Core::LinAlg::Map>(*(*dit).second->dof_row_map()));
  }

  std::shared_ptr<Core::LinAlg::Map> fullmap = Core::LinAlg::MultiMapExtractor::merge_maps(maps);
  fullextractor_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(*fullmap, maps);
}

/*------------------------------------------------------------------------------------------------*
| Get Coupling Converter between Discret A and B                                      ager 03/2016|
*------------------------------------------------------------------------------------------------*/
std::shared_ptr<Coupling::Adapter::CouplingConverter>
XFEM::CouplingCommManager::get_coupling_converter(int idxA, int idxB)
{
  if (idxA < idxB)
  {
    return std::make_shared<Coupling::Adapter::CouplingMasterConverter>(*get_coupling(idxA, idxB));
  }
  else if (idxA > idxB)
  {
    return std::make_shared<Coupling::Adapter::CouplingSlaveConverter>(*get_coupling(idxB, idxA));
  }
  else
  {
    FOUR_C_THROW(
        "Coupling_Comm_Manager::get_coupling_converter: Coupling Converter from {} to {} makes not "
        "really sense, does it?",
        idxA, idxB);
  }
  return nullptr;  // 2 make Compiler happy :-)
}

/*------------------------------------------------------------------------------------------------*
| Get Coupling Object between Discret A and B                                      -- ager 03/2016|
*------------------------------------------------------------------------------------------------*/
std::shared_ptr<Coupling::Adapter::Coupling> XFEM::CouplingCommManager::get_coupling(
    int idxA, int idxB)
{
  if (idxA < idxB)
  {
    std::pair<int, int> key = std::pair<int, int>(idxA, idxB);
    std::map<std::pair<int, int>, std::shared_ptr<Coupling::Adapter::Coupling>>::iterator cit =
        coup_.find(key);
    if (cit != coup_.end())
    {
      return cit->second;
    }
    else
    {
      FOUR_C_THROW("Coupling_Comm_Manager::GetCoupling: Couldn't find Coupling for key ({},{})",
          key.first, key.second);
    }
  }
  else
  {
    FOUR_C_THROW(
        "Coupling_Comm_Manager::GetCoupling: Just Coupling Objects for idxA < idxB are stored!");
  }
  return nullptr;
}

/*------------------------------------------------------------------------------------------------*
| Get Transform Object between Discret A and B                                        ager 03/2016|
*------------------------------------------------------------------------------------------------*/
std::shared_ptr<Coupling::Adapter::MatrixLogicalSplitAndTransform>
XFEM::CouplingCommManager::get_transform(int transform_id)
{
  std::map<int, std::shared_ptr<Coupling::Adapter::MatrixLogicalSplitAndTransform>>::iterator tit =
      transform_.find(transform_id);
  if (tit != transform_.end() && transform_id != -1)
  {
    return tit->second;
  }
  else
  {
    transform_[transform_id] =
        std::make_shared<Coupling::Adapter::MatrixLogicalSplitAndTransform>();
    return transform_[transform_id];
  }
  return nullptr;
}

/*------------------------------------------------------------------------------------------------*
| Get Map Extractor Object for idx                                                    ager 03/2016|
*------------------------------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiMapExtractor> XFEM::CouplingCommManager::get_map_extractor(
    int idx)
{
  std::map<int, std::shared_ptr<Core::LinAlg::MultiMapExtractor>>::iterator mit = mme_.find(idx);
  if (mit != mme_.end())
  {
    return mit->second;
  }
  else
  {
    FOUR_C_THROW(
        "Coupling_Comm_Manager::GetMapExtractor: Couldn't find Map Extractor for key ({})", idx);
  }
  return nullptr;  // to guarantee that the compiler feels comfortable in his skin...
}

/*---------------------------------------------------------------------------------------------------*
| Coupling_Comm_Manager Debug Output                                                     ager
03/2016|
*---------------------------------------------------------------------------------------------------*/
void XFEM::CouplingCommManager::debug_out(
    std::string str1, std::string str2, std::string str3, std::string str4)
{
#ifdef COUP_MANAGER_DEBUG_OUT
  std::cout << str1 << str2 << str3 << str4 << std::flush;
#endif
  return;
}

FOUR_C_NAMESPACE_CLOSE

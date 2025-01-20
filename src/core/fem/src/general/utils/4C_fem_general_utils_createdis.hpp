// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_UTILS_CREATEDIS_HPP
#define FOUR_C_FEM_GENERAL_UTILS_CREATEDIS_HPP

#include "4C_config.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_immersed_node.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_io_pstream.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  /// class providing basic functionality for cloning discretizations
  class DiscretizationCreatorBase
  {
   public:
    //! constructor
    explicit DiscretizationCreatorBase() : numeleskips_(0) {}

    //! destructor
    virtual ~DiscretizationCreatorBase() = default;
    //! copy given conditions from source to target discretization
    void copy_conditions(const Core::FE::Discretization& sourcedis,
        Core::FE::Discretization& targetdis,
        const std::map<std::string, std::string>& conditions_to_copy) const;


    /**
     * generate identical clone of sourcedis (eles, nodes and conditions, dofs are only cloned
     * with flag clonedofs = true)
     * @param sourcedis: discretization based on which clone will be generated
     * @param targetdisname: name of cloned discretization
     * @param clonedofs: should Dofs be cloned?
     * @param assigndegreesoffreedom: flag for call to fill complete on cloned dis
     * @param initelements: flag for call to fill complete on cloned dis
     * @param doboundaryconditions: flag for call to fill complete on cloned dis
     * @return the cloned discretization
     */
    std::shared_ptr<Core::FE::Discretization> create_matching_discretization(
        Core::FE::Discretization& sourcedis, const std::string& targetdisname,
        bool clonedofs = true, bool assigndegreesoffreedom = true, bool initelements = true,
        bool doboundaryconditions = true) const;

    //! Base class version for creation of matching discretization without material
    std::shared_ptr<Core::FE::Discretization> create_matching_discretization_from_condition(
        const Core::FE::Discretization& sourcedis,  ///< discretization with condition
        const Core::Conditions::Condition&
            cond,  ///< condition, from which the derived discretization is derived
        const std::string& discret_name,  ///< name of the new discretization
        const std::string& element_name,  ///< name/type of the elements to be created
        const std::vector<std::string>& conditions_to_copy  ///< list of conditions that will be
                                                            ///< copied to the new discretization
    )
    {
      if (cond.get_nodes() == nullptr or cond.get_nodes()->size() == 0)
        FOUR_C_THROW("The condition has no nodes!");

      // make sure connectivity is all set
      // we don't care, whether dofs exist or not
      if (!sourcedis.filled()) FOUR_C_THROW("sourcedis is not filled");

      // get this condition's elements
      std::map<int, std::shared_ptr<Core::Elements::Element>> sourceelements;
      const std::map<int, std::shared_ptr<Core::Elements::Element>>& geo = cond.geometry();
      sourceelements.insert(geo.begin(), geo.end());

      return create_matching_discretization_from_condition(
          sourcedis, sourceelements, discret_name, element_name, conditions_to_copy);
    };  // create_matching_discretization_from_condition

    //! Base class version for creation of matching discretization without material
    std::shared_ptr<Core::FE::Discretization> create_matching_discretization_from_condition(
        const Core::FE::Discretization& sourcedis,  ///< discretization with condition
        const std::string& condname,                ///< name of the condition, by which the derived
                                                    ///< discretization is identified
        const std::string& discret_name,            ///< name of the new discretization
        const std::string&
            element_name,  ///< name/type of the elements to be created, if set to "" the natural
                           ///< element will be created (e.g. Fluid-->FluidBoundary,...)
        const std::vector<std::string>& conditions_to_copy,  ///< list of conditions that will be
                                                             ///< copied to the new discretization
        const int label = -1  ///< consider only conditions with specified label
    )
    {
      // make sure connectivity is all set
      // we don't care, whether dofs exist or not
      if (!sourcedis.filled()) FOUR_C_THROW("sourcedis is not filled");

      // We need to test for all elements (including ghosted ones) to
      // catch all nodes
      std::map<int, std::shared_ptr<Core::Elements::Element>> sourceelements;
      Core::Conditions::find_condition_objects(sourcedis, sourceelements, condname, label);

      return create_matching_discretization_from_condition(
          sourcedis, sourceelements, discret_name, element_name, conditions_to_copy);
    };  // create_discretization_from_condition

    //! method for cloning a new discretization from an existing condition without material
    std::shared_ptr<Core::FE::Discretization> create_matching_discretization_from_condition(
        const Core::FE::Discretization& sourcedis,  ///< discretization with condition
        const std::map<int, std::shared_ptr<Core::Elements::Element>>&
            sourceelements,               ///< element map/geometry of the condition
        const std::string& discret_name,  ///< name of the new discretization
        const std::string&
            element_name,  ///< name/type of the elements to be created, if set to "" the natural
                           ///< element will be created (e.g. Fluid-->FluidBoundary,...)
        const std::vector<std::string>& conditions_to_copy  ///< list of conditions that will be
                                                            ///< copied to the new discretization
    )
    {
      MPI_Comm com = sourcedis.get_comm();
      const int myrank = Core::Communication::my_mpi_rank(com);
      const Epetra_Map* sourcenoderowmap = sourcedis.node_row_map();

      std::shared_ptr<Core::FE::Discretization> targetdis;

      // try to cast sourcedis to NurbsDiscretization
      const Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
          dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(&sourcedis);

      if (nurbsdis != nullptr)
        targetdis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
            discret_name, com, sourcedis.n_dim());
      else
        targetdis =
            std::make_shared<Core::FE::Discretization>(discret_name, com, sourcedis.n_dim());

      // construct new elements
      for (std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator sourceele_iter =
               sourceelements.begin();
          sourceele_iter != sourceelements.end(); ++sourceele_iter)
      {
        const std::shared_ptr<Core::Elements::Element> sourceele = sourceele_iter->second;

        // get global node ids
        std::vector<int> nids;
        nids.reserve(sourceele->num_node());
        transform(sourceele->nodes(), sourceele->nodes() + sourceele->num_node(),
            back_inserter(nids), std::mem_fn(&Core::Nodes::Node::id));

        // check if element has nodes which are not in col map on this proc.
        // this should not be the case since each proc should have all nodes of
        // all owned or ghosted elements in the col map.
        if (std::count_if(nids.begin(), nids.end(),
                Core::Conditions::MyGID(sourcedis.node_col_map())) != (int)(nids.size()))
        {
          FOUR_C_THROW("element %d owned by proc %d has remote non-ghost nodes", sourceele->id(),
              sourceele->owner());
        }

        copy(nids.begin(), nids.end(), inserter(colnodeset_, colnodeset_.begin()));

        // copy node ids of condition ele to rownodeset but leave those that do
        // not belong to this processor
        remove_copy_if(nids.begin(), nids.end(), inserter(rownodeset_, rownodeset_.begin()),
            std::not_fn(Core::Conditions::MyGID(sourcenoderowmap)));

        // Do not clone ghost elements here! Those will be handled by the
        // discretization itself.
        if (sourceele->owner() == myrank)
        {
          std::shared_ptr<Core::Elements::Element> condele;
          if (element_name == "")
          {
            // copy the source ele (created in fill complete of the discretization)
            condele = std::shared_ptr<Core::Elements::Element>(sourceele->clone());
          }
          else
          {
            // create an element with the same global element id
            condele =
                Core::Communication::factory(element_name, "Polynomial", sourceele->id(), myrank);
            // set the same global node ids to the new element
            condele->set_node_ids(nids.size(), nids.data());
          }
          // add element
          targetdis->add_element(condele);
          roweleset_.insert(sourceele->id());
        }
        coleleset_.insert(sourceele->id());
      }  // loop over all source elements

      // construct new nodes, which use the same global id as the source nodes
      for (int i = 0; i < sourcenoderowmap->NumMyElements(); ++i)
      {
        const int gid = sourcenoderowmap->GID(i);
        if (rownodeset_.find(gid) != rownodeset_.end())
        {
          const Core::Nodes::Node* sourcenode = sourcedis.l_row_node(i);

          // check if this node is a NURBS control point
          const Core::FE::Nurbs::ControlPoint* control_point =
              dynamic_cast<const Core::FE::Nurbs::ControlPoint*>(sourcenode);

          // if the node cannot be dynamic casted to a control point, add the point as a node
          if (!control_point)
            targetdis->add_node(std::make_shared<Core::Nodes::Node>(gid, sourcenode->x(), myrank));
          else
            targetdis->add_node(std::make_shared<Core::FE::Nurbs::ControlPoint>(
                gid, control_point->x(), control_point->w(), myrank));
        }
      }

      // we get the element maps almost for free
      targetelerowmap_ = create_map(roweleset_, *targetdis);
      targetelecolmap_ = create_map(coleleset_, *targetdis);
      // we get the node maps almost for free
      targetnoderowmap_ = create_map(rownodeset_, *targetdis);
      targetnodecolmap_ = create_map(colnodeset_, *targetdis);

      // copy selected conditions to the new discretization
      for (const auto& cond_name : conditions_to_copy)
      {
        std::vector<Core::Conditions::Condition*> conds;
        sourcedis.get_condition(cond_name, conds);
        for (const auto& cond : conds)
        {
          // We use the same nodal ids and therefore we can just copy the conditions.
          targetdis->set_condition(cond_name, cond->copy_without_geometry());
        }
      }

      // we always skip the safety checks in Finalize()
      // because we create a discretization from a
      // conditioned subset of the source discretiztation.
      numeleskips_++;

      // call redistribute, fill_complete etc.
      finalize(sourcedis, *targetdis);

      return targetdis;
    };  // create_discretization_from_condition without material

   protected:
    //! construct row nodes for cloned target discretization
    void create_nodes(const Core::FE::Discretization& sourcedis,
        Core::FE::Discretization& targetdis, const std::set<int>& rownodeset,
        const std::set<int>& colnodeset, const bool isnurbsdis, const bool buildimmersednode) const;

    //! construct and return Epetra_Map
    std::shared_ptr<Epetra_Map> create_map(
        std::set<int>& gidset, const Core::FE::Discretization& targetdis) const;

    //! do some checks
    void initial_checks(
        const Core::FE::Discretization& sourcedis, const Core::FE::Discretization& targetdis) const;

    //! export target nodes and elements and perform some checks
    void finalize(
        const Core::FE::Discretization& sourcedis, Core::FE::Discretization& targetdis) const;

   protected:
    //! set of row nodes
    std::set<int> rownodeset_;
    //! set of column nodes
    std::set<int> colnodeset_;
    //! set of row elements
    std::set<int> roweleset_;
    //! set of column elements
    std::set<int> coleleset_;
    //! vector for holding each (desired) element type std::string
    std::vector<std::string> eletype_;
    //! map containing gids of owned nodes
    std::shared_ptr<Epetra_Map> targetnoderowmap_;
    //! map containing gids of owned + ghosted nodes
    std::shared_ptr<Epetra_Map> targetnodecolmap_;
    //! map containing gids of owned elements
    std::shared_ptr<Epetra_Map> targetelerowmap_;
    //! map containing gids of owned + ghosted elements
    std::shared_ptr<Epetra_Map> targetelecolmap_;
    //! local number of skipped elements during cloning
    int numeleskips_;

  };  // class DiscretizationCreatorBase


  /// class for cloning a new discretization from an existing one
  template <class CloneStrategy>
  class DiscretizationCreator : DiscretizationCreatorBase, CloneStrategy
  {
   public:
    /// constructor
    explicit DiscretizationCreator() {};

    /// Create the clone field material map from the input file
    void create_clone_field_mat_map(std::map<int, int>& matmap,
        const Core::FE::Discretization& sourcedis, const Core::FE::Discretization& targetdis,
        const std::map<std::pair<std::string, std::string>, std::map<int, int>>& clonefieldmatmap)
        const
    {
      if (matmap.size()) FOUR_C_THROW("The input material map is supposed to be empty!");

      if (clonefieldmatmap.size() < 1)
        FOUR_C_THROW("At least one material pairing required in --CLONING MATERIAL MAP.");

      std::pair<std::string, std::string> key(sourcedis.name(), targetdis.name());
      matmap = clonefieldmatmap.at(key);
      if (matmap.size() < 1)
        FOUR_C_THROW("Key pair '%s/%s' not defined in --CLONING MATERIAL MAP.",
            sourcedis.name().c_str(), targetdis.name().c_str());

      return;
    };  // create_clone_field_mat_map

    /// method for cloning a new discretization from an existing one
    void create_matching_discretization(
        Core::FE::Discretization& sourcedis,  ///< std::shared_ptr to source discretization
        Core::FE::Discretization& targetdis,  ///< std::shared_ptr to empty target discretization
        const int matid  ///< ID of the material which generated elements will get
    )
    {
      // obsolete function call which should not be used anymore!
      // let's use the more general version using an explicit material id mapping
      // as defined in the input file section "--CLONING MATERIAL MAP"

      // check and analyze source discretization (sorcedis must be filled!)
      initial_checks(sourcedis, targetdis);

      // We have to find out all the material ids of the source discretization.
      // All cloned elements will receive the same material with the provided matid.
      std::map<int, int> matmap;
      int numelements = sourcedis.num_my_col_elements();
      if (numelements < 1) FOUR_C_THROW("At least one processor has no element");
      for (int i = 0; i < numelements; ++i)
      {
        Core::Elements::Element* sourceele = sourcedis.l_col_element(i);
        int src_matid = sourceele->material()->parameter()->id();
        // if a new material id is found -> extend the map
        std::map<int, int>::iterator mat_iter = matmap.find(src_matid);
        if (mat_iter == matmap.end())
        {
          std::pair<int, int> matmappair(src_matid, matid);
          matmap.insert(matmappair);
        }
      }

      create_matching_discretization(sourcedis, targetdis, matmap);

      return;
    };  // create_matching_discretization

    /// method for cloning a new discretization from an existing one
    void create_matching_discretization(
        Core::FE::Discretization& sourcedis,  ///< std::shared_ptr to source discretization
        Core::FE::Discretization& targetdis,  ///< std::shared_ptr to empty target discretization
        const std::map<int, int>&
            matmap  ///< map of material IDs (source element -> target element)
    )
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (!(sourcedis.have_global_node(sourcedis.node_row_map()->GID(0))))
        FOUR_C_THROW("Cloning not possible since node with GID %d is not stored on this proc!",
            sourcedis.node_row_map()->GID(0));
#endif
      // try to cast sourcedis to NurbsDiscretisation
      Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
          dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&((sourcedis)));
      bool isnurbsdis(nurbsdis != nullptr);

      // try to cast source node to immersed node
      Core::Nodes::ImmersedNode* inode = dynamic_cast<Core::Nodes::ImmersedNode*>(
          sourcedis.g_node(sourcedis.node_row_map()->GID(0)));
      bool buildimmersednode(inode != nullptr);

      // check and analyze source discretization
      initial_checks(sourcedis, targetdis);
      analyze_source_dis(sourcedis, eletype_, rownodeset_, colnodeset_, roweleset_, coleleset_);

      // do the node business
      create_nodes(sourcedis, targetdis, rownodeset_, colnodeset_, isnurbsdis, buildimmersednode);
      targetnoderowmap_ = create_map(rownodeset_, targetdis);
      targetnodecolmap_ = create_map(colnodeset_, targetdis);

      // create elements
      create_elements(sourcedis, targetdis, matmap, isnurbsdis);
      targetelerowmap_ = create_map(roweleset_, targetdis);
      targetelecolmap_ = create_map(coleleset_, targetdis);

      // copy desired conditions from source to target discretization
      const auto conditions_to_copy = CloneStrategy::conditions_to_copy();
      copy_conditions(sourcedis, targetdis, conditions_to_copy);

      // call redistribute, fill_complete etc.
      finalize(sourcedis, targetdis);
    };  // create_matching_discretization

    /// method for cloning a new discretization from an existing condition using the actual
    /// condition
    void create_matching_discretization_from_condition(
        const Core::FE::Discretization& sourcedis,  ///< ref. to source discretization
        const std::vector<Core::Conditions::Condition*>&
            conds,  ///< vector of conditions containing the elements to clone
        Core::FE::Discretization& targetdis,  ///< std::shared_ptr to empty target discretization
        const std::map<int, int>&
            matmap  ///< map of material IDs (source element -> target element)
    )
    {
      // check and analyze source and target discretization
      initial_checks(sourcedis, targetdis);

      std::vector<Core::Conditions::Condition*>::const_iterator cit;
      for (cit = conds.begin(); cit != conds.end(); ++cit)
      {
        // check the source condition
        if ((*cit)->get_nodes() == nullptr or (*cit)->get_nodes()->size() == 0)
          FOUR_C_THROW("The condition has no nodes!");
      }

      // get this condition vector's elements
      std::map<int, std::shared_ptr<Core::Elements::Element>> sourceelements;
      Core::Conditions::find_condition_objects(sourceelements, conds);

      create_matching_discretization_from_condition(sourcedis, sourceelements, targetdis, matmap);
      return;
    };  // create_matching_discretization_from_condition

    /// method for cloning a new discretization from an existing condition using the condition
    /// name
    void create_matching_discretization_from_condition(
        const Core::FE::Discretization& sourcedis,  ///< ref. to source discretization
        const std::string& condname,          ///< string to identify conditioned elements to clone
        Core::FE::Discretization& targetdis,  ///< std::shared_ptr to empty target discretization
        const std::map<int, int>&
            matmap  ///< map of material IDs (source element -> target element)
    )
    {
      // check and analyze source discretization
      initial_checks(sourcedis, targetdis);
      std::map<int, std::shared_ptr<Core::Elements::Element>> sourceelements;
      Core::Conditions::find_condition_objects(sourcedis, sourceelements, condname);

      create_matching_discretization_from_condition(sourcedis, sourceelements, targetdis, matmap);
      return;
    };  // create_matching_discretization_from_condition

   private:
    /// method for cloning a new discretization from an existing condition with material
    void create_matching_discretization_from_condition(
        const Core::FE::Discretization& sourcedis,  ///< ref. to source discretization
        const std::map<int, std::shared_ptr<Core::Elements::Element>>&
            sourceelements,                   ///< conditioned element map to clone
        Core::FE::Discretization& targetdis,  ///< std::shared_ptr to empty target discretization
        const std::map<int, int>&
            matmap  ///< map of material IDs (source element -> target element)
    )
    {
      // try to cast sourcedis to NurbsDiscretisation
      const Core::FE::Nurbs::NurbsDiscretization* nurbsdis_ptr =
          dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(&sourcedis);
      bool isnurbsdis(nurbsdis_ptr != nullptr);

      // try to cast source node to immersed node
      Core::Nodes::ImmersedNode* inode = dynamic_cast<Core::Nodes::ImmersedNode*>(
          sourcedis.g_node(sourcedis.node_row_map()->GID(0)));
      bool buildimmersednode(inode != nullptr);

      analyze_conditioned_source_dis(
          sourcedis, sourceelements, eletype_, rownodeset_, colnodeset_, roweleset_, coleleset_);

      // do the node business
      create_nodes(sourcedis, targetdis, rownodeset_, colnodeset_, isnurbsdis, buildimmersednode);
      targetnoderowmap_ = create_map(rownodeset_, targetdis);
      targetnodecolmap_ = create_map(colnodeset_, targetdis);

      // create elements
      create_elements_from_condition(sourceelements, targetdis, matmap, isnurbsdis);
      targetelerowmap_ = create_map(roweleset_, targetdis);
      targetelecolmap_ = create_map(coleleset_, targetdis);

      // copy desired conditions from source to target discretization
      const auto conditions_to_copy = CloneStrategy::conditions_to_copy();
      copy_conditions(sourcedis, targetdis, conditions_to_copy);

      // call redistribute, fill_complete etc.
      finalize(sourcedis, targetdis);
    };  // create_matching_discretization_from_condition with material

    /// get element type std::strings and global id's and nodes from source discretization
    void analyze_source_dis(Core::FE::Discretization& sourcedis, std::vector<std::string>& eletype,
        std::set<int>& rownodeset, std::set<int>& colnodeset, std::set<int>& roweleset,
        std::set<int>& coleleset)
    {
      const Epetra_Map* noderowmap = sourcedis.node_row_map();

      // We need to test for all elements (including ghosted ones) to
      // catch all nodes attached to the elements of the source discretization
      // we will clone only those (-> support for ALE sub-meshes)
      int numelements = sourcedis.num_my_col_elements();

      for (int i = 0; i < numelements; ++i)
      {
        Core::Elements::Element* actele = sourcedis.l_col_element(i);
        bool ismyele = sourcedis.element_row_map()->MyGID(actele->id());

        // we get the element type std::string and a boolean if this element
        // is considered! (see submeshes for Fluid-ALE case!)
        if (CloneStrategy::determine_ele_type(actele, ismyele, eletype))
        {
          // we make sure, that the cloned discretization
          // has the same parallel distribution as the
          // source discretization.
          if (ismyele) roweleset.insert(actele->id());

          coleleset.insert(actele->id());

          // copy node ids of actele to rownodeset but leave those that do
          // not belong to this processor
          remove_copy_if(actele->node_ids(), actele->node_ids() + actele->num_node(),
              inserter(rownodeset, rownodeset.begin()),
              std::not_fn(Core::Conditions::MyGID(noderowmap)));

          copy(actele->node_ids(), actele->node_ids() + actele->num_node(),
              inserter(colnodeset, colnodeset.begin()));
        }
        else
          numeleskips_++;
      }  // loop over my elements
      return;
    };  // analyze_source_dis

    /// get element type std::strings and global id's and nodes from conditioned source
    /// discretization
    void analyze_conditioned_source_dis(const Core::FE::Discretization& sourcedis,
        const std::map<int, std::shared_ptr<Core::Elements::Element>>& sourceelements,
        std::vector<std::string>& eletype, std::set<int>& rownodeset, std::set<int>& colnodeset,
        std::set<int>& roweleset, std::set<int>& coleleset)
    {
      const int myrank = Core::Communication::my_mpi_rank(sourcedis.get_comm());
      const Epetra_Map* sourcenoderowmap = sourcedis.node_row_map();
      const Epetra_Map* sourcenodecolmap = sourcedis.node_col_map();

      // construct new elements
      std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator sourceele_iter;
      for (sourceele_iter = sourceelements.begin(); sourceele_iter != sourceelements.end();
          ++sourceele_iter)
      {
        const std::shared_ptr<Core::Elements::Element> actele = sourceele_iter->second;
        const bool ismyele = (actele->owner() == myrank);

        // we get the element type std::string and a boolean if this element
        // is considered! (see submeshes for Fluid-ALE case!)
        if (CloneStrategy::determine_ele_type(&(*actele), ismyele, eletype))
        {
          // we make sure, that the cloned discretization
          // has the same parallel distribution as the
          // source discretization.
          // as prerequisite it is required that the parallel
          // distribution of the condition geometry matches
          // the distribution of the underlying parent elements.
          // this is made sure in drt_discret_condition.cpp
          // (rauch 10/16).
          if (ismyele) roweleset.insert(actele->id());

          coleleset.insert(actele->id());

          // get global node ids
          std::vector<int> nids;
          nids.reserve(actele->num_node());
          transform(actele->nodes(), actele->nodes() + actele->num_node(), back_inserter(nids),
              std::mem_fn(&Core::Nodes::Node::id));

          // check if element has nodes, which are not in col map on this proc.
          // this should not be, since each proc should have all nodes of all
          // owned, or ghosted elements in the col map.
          if (std::count_if(nids.begin(), nids.end(), Core::Conditions::MyGID(sourcenodecolmap)) !=
              (int)(nids.size()))
          {
            FOUR_C_THROW("element %d owned by proc %d has remote non-ghost nodes", actele->id(),
                actele->owner());
          }

          // copy node ids of condition ele to set of column nodes
          copy(nids.begin(), nids.end(), inserter(colnodeset, colnodeset.begin()));

          // copy node ids of condition ele to rownodeset except for those which do
          // not belong to this processor
          remove_copy_if(nids.begin(), nids.end(), inserter(rownodeset, rownodeset.begin()),
              std::not_fn(Core::Conditions::MyGID(sourcenoderowmap)));
        }
      }

      // we always skip the safety checks in Finalize()
      // because we create a discretization from a
      // conditioned subset of the source discretiztation.
      numeleskips_++;
      return;
    };  // analyze_conditioned_source_dis

    /// create new elements and add them to the target discretization
    void create_elements(Core::FE::Discretization& sourcedis, Core::FE::Discretization& targetdis,
        std::map<int, int> matmap, const bool isnurbsdis)
    {
      // now do the elements
      for (std::map<int, int>::iterator mapit = matmap.begin(); mapit != matmap.end(); ++mapit)
      {
        int target_id = mapit->second;
        CloneStrategy::check_material_type(target_id);
      }

      // prepare some variables we need
      int myrank = Core::Communication::my_mpi_rank(targetdis.get_comm());

      // construct new elements
      // The order of the elements might be different from that of the
      // source elements. We don't care. There are no dofs to these elements.
      std::set<int>::iterator it = roweleset_.begin();
      for (std::size_t i = 0; i < roweleset_.size(); ++i)
      {
        Core::Elements::Element* sourceele = sourcedis.g_element(*it);

        std::string approxtype = "Polynomial";
        if (isnurbsdis)
        {
          if (sourceele->num_node() == 8)
          {
            approxtype = "NURBS8";
          }
          else if (sourceele->num_node() == 9)
          {
            approxtype = "NURBS9";
          }
          else if (sourceele->num_node() == 4)
          {
            approxtype = "NURBS4";
          }
          else if (sourceele->num_node() == 27)
          {
            approxtype = "NURBS27";
          }
          else if (sourceele->num_node() == 2)
          {
            approxtype = "NURBS2";
          }
          else if (sourceele->num_node() == 3)
          {
            approxtype = "NURBS3";
          }
          else
          {
            FOUR_C_THROW("unknown type of nurbs element\n");
          }
        }

        // create a new element of desired type with the same global element id
        std::shared_ptr<Core::Elements::Element> newele =
            Core::Communication::factory(eletype_[i], approxtype, *it, myrank);

        // get global node ids of source element
        std::vector<int> nids;
        nids.reserve(sourceele->num_node());
        transform(sourceele->nodes(), sourceele->nodes() + sourceele->num_node(),
            back_inserter(nids), std::mem_fn(&Core::Nodes::Node::id));

        // set the same global node ids to the new element
        newele->set_node_ids(nids.size(), nids.data());

        // We need to set material and gauss points to complete element setup.
        // This is again really ugly as we have to extract the actual
        // element type in order to access the material property
        // note: set_material() was reimplemented by the transport element!

        int src_matid = sourceele->material()->parameter()->id();
        std::map<int, int>::iterator mat_iter = matmap.find(src_matid);
        if (mat_iter != matmap.end())
        {
          int tar_matid = mat_iter->second;
          CloneStrategy::set_element_data(newele, sourceele, tar_matid, isnurbsdis);

          // add new element to discretization
          targetdis.add_element(newele);
        }
        else
        {
          // before we stop, print the material id map
          std::cout << "Material map on PROC " << myrank << ":" << std::endl;
          for (mat_iter = matmap.begin(); mat_iter != matmap.end(); mat_iter++)
            std::cout << mat_iter->first << " -> " << mat_iter->second << std::endl;

          FOUR_C_THROW("no matching material ID (%d) in map", src_matid);
        }
        it++;
      }
      return;
    };  // create_elements

    /// create new elements from the condition and add them to the target discretization
    void create_elements_from_condition(
        const std::map<int, std::shared_ptr<Core::Elements::Element>>& sourceelements,
        Core::FE::Discretization& targetdis, const std::map<int, int>& matmap,
        const bool& isnurbsdis)
    {
      // now do the elements
      for (std::map<int, int>::const_iterator mapit = matmap.begin(); mapit != matmap.end();
          ++mapit)
      {
        int target_id = mapit->second;
        CloneStrategy::check_material_type(target_id);
      }

      // prepare some variables we need
      int myrank = Core::Communication::my_mpi_rank(targetdis.get_comm());

      // construct new elements
      // The order of the elements might be different from that of the
      // source elements. We don't care. There are no dofs to these elements.
      std::set<int>::iterator it = roweleset_.begin();
      for (std::size_t i = 0; i < roweleset_.size(); ++i)
      {
        std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator src_ele_citer =
            sourceelements.find(*it);
        if (src_ele_citer == sourceelements.end())
          FOUR_C_THROW(
              "The source element %d could not be found in the source "
              "condition element map!",
              *it);

        Core::Elements::Element* sourceele = src_ele_citer->second.get();
        if (sourceele == nullptr) FOUR_C_THROW("The sourceele pointer is nullptr!");

        std::string approxtype = "Polynomial";
        if (isnurbsdis)
        {
          if (sourceele->num_node() == 8)
          {
            approxtype = "NURBS8";
          }
          else if (sourceele->num_node() == 9)
          {
            approxtype = "NURBS9";
          }
          else if (sourceele->num_node() == 4)
          {
            approxtype = "NURBS4";
          }
          else if (sourceele->num_node() == 27)
          {
            approxtype = "NURBS27";
          }
          else if (sourceele->num_node() == 2)
          {
            approxtype = "NURBS2";
          }
          else if (sourceele->num_node() == 3)
          {
            approxtype = "NURBS3";
          }
          else
          {
            FOUR_C_THROW("unknown type of nurbs element\n");
          }
        }

        // get owner of source element
        const int sourceeleowner = sourceele->owner();
        if (myrank != sourceeleowner)
          FOUR_C_THROW("roweleset_ should only contain my element gids!");

        // create a new element of desired type with the same global element id and same owner as
        // source element
        std::shared_ptr<Core::Elements::Element> newele =
            Core::Communication::factory(eletype_[i], approxtype, *it, sourceeleowner);

        // get global node ids of fluid element
        std::vector<int> nids;
        nids.reserve(sourceele->num_node());
        transform(sourceele->nodes(), sourceele->nodes() + sourceele->num_node(),
            back_inserter(nids), std::mem_fn(&Core::Nodes::Node::id));

        // set the same global node ids to the new element
        newele->set_node_ids(nids.size(), nids.data());

        // We need to set material and gauss points to complete element setup.
        // This is again really ugly as we have to extract the actual
        // element type in order to access the material property
        // note: set_material() was reimplemented by the transport element!
        std::shared_ptr<Core::Mat::Material> mat_ptr = sourceele->material();
        /* Check if the material pointer is null. If necessary, try to cast
         * the condition element to a FaceElement and ask the parent element for
         * the material.                                                      */
        if (!mat_ptr)
        {
          Core::Elements::FaceElement* src_face_element =
              dynamic_cast<Core::Elements::FaceElement*>(sourceele);
          if (src_face_element != nullptr) mat_ptr = src_face_element->parent_element()->material();
        }
        // It is no FaceElement or the material pointer of the parent element is nullptr.
        if (!mat_ptr) FOUR_C_THROW("The condition element has no material!");

        int src_matid = mat_ptr->parameter()->id();
        std::map<int, int>::const_iterator mat_iter = matmap.find(src_matid);
        if (mat_iter != matmap.end())
        {
          int tar_matid = mat_iter->second;
          CloneStrategy::set_element_data(newele, sourceele, tar_matid, isnurbsdis);

          // add new element to discretization
          targetdis.add_element(newele);
        }
        else
        {
          // before we stop, print the material id map
          std::cout << "Material map on PROC " << myrank << ":" << std::endl;
          for (mat_iter = matmap.begin(); mat_iter != matmap.end(); mat_iter++)
            std::cout << mat_iter->first << " -> " << mat_iter->second << std::endl;

          FOUR_C_THROW("no matching material ID (%d) in map", src_matid);
        }
        it++;
      }
      return;
    }  // create_elements_from_condition

  };  // class DiscretizationCreator


  /// clone target discretization @p targetdis from a given source discretization @p sourcedis.
  /// The @p clonefieldmatmap is required from the global CloningMaterialMap.
  template <class CloneStrategy>
  void clone_discretization(Core::FE::Discretization& sourcedis,
      Core::FE::Discretization& targetdis,
      const std::map<std::pair<std::string, std::string>, std::map<int, int>>& clonefieldmatmap)
  {
    // access the communicator for time measurement
    MPI_Comm comm = sourcedis.get_comm();
    Teuchos::Time time("", true);

    // create target discretization using a given clone strategy
    {
      std::shared_ptr<Core::FE::DiscretizationCreator<CloneStrategy>> clonewizard =
          std::make_shared<Core::FE::DiscretizationCreator<CloneStrategy>>();

      std::map<int, int> matmap;
      clonewizard->create_clone_field_mat_map(matmap, sourcedis, targetdis, clonefieldmatmap);

      clonewizard->create_matching_discretization(sourcedis, targetdis, matmap);
    }
    if (Core::Communication::my_mpi_rank(comm) == 0)
    {
      Core::IO::cout << "Created discretization " << targetdis.name()
                     << " as a clone of discretization " << sourcedis.name() << " in...."
                     << time.totalElapsedTime(true) << " secs\n\n";
    }
    return;
  };  // clone_discretization

  /// clone target discretization @p targetdis from a given source discretization @p sourcedis
  /// based on conditions @p conds. The @p clonefieldmatmap is required from the global
  /// CloningMaterialMap.
  template <class CloneStrategy>
  void clone_discretization_from_condition(const Core::FE::Discretization& sourcedis,
      Core::FE::Discretization& targetdis, const std::vector<Core::Conditions::Condition*>& conds,
      const std::map<std::pair<std::string, std::string>, std::map<int, int>>& clonefieldmatmap)
  {
    const Core::FE::Discretization* sourcedis_ptr =
        dynamic_cast<const Core::FE::Discretization*>(&sourcedis);
    if (sourcedis_ptr == nullptr) FOUR_C_THROW("Cast of the source discretization failed!");
    Core::FE::Discretization* targetdis_ptr = dynamic_cast<Core::FE::Discretization*>(&targetdis);
    if (targetdis_ptr == nullptr) FOUR_C_THROW("Cast of the target discretization failed!");

    // access the communicator for time measurement
    MPI_Comm comm = sourcedis_ptr->get_comm();
    Teuchos::Time time("", true);

    // create target discretization using a given clone strategy
    {
      std::shared_ptr<Core::FE::DiscretizationCreator<CloneStrategy>> clonewizard =
          std::make_shared<Core::FE::DiscretizationCreator<CloneStrategy>>();

      std::map<int, int> matmap;
      clonewizard->create_clone_field_mat_map(
          matmap, *sourcedis_ptr, *targetdis_ptr, clonefieldmatmap);

      clonewizard->create_matching_discretization_from_condition(
          *sourcedis_ptr, conds, *targetdis_ptr, matmap);
    }
    if (Core::Communication::my_mpi_rank(comm) == 0)
    {
      Core::IO::cout << "Created discretization " << targetdis_ptr->name()
                     << " as a clone from the condition(s) with ID(s)=";
      for (unsigned int i = 0; i < conds.size(); ++i) Core::IO::cout << " " << conds[i]->id();
      Core::IO::cout << " of the discretization " << sourcedis_ptr->name() << " in...."
                     << time.totalElapsedTime(true) << " secs\n\n";
    }
    return;
  };  // clone_discretization_from_condition

  /// clone target discretization @p targetdis from a given source discretization @p sourcedis
  /// based on the name of a condition @p condname. The @p clonefieldmatmap is required from the
  /// global CloningMaterialMap.
  template <class CloneStrategy>
  void clone_discretization_from_condition(const Core::FE::Discretization& sourcedis,
      Core::FE::Discretization& targetdis, const std::string& condname,
      const std::map<std::pair<std::string, std::string>, std::map<int, int>>& clonefieldmatmap)
  {
    // access the communicator for time measurement
    MPI_Comm comm = sourcedis.get_comm();
    Teuchos::Time time("", true);

    // create target discretization using a given clone strategy
    {
      std::shared_ptr<Core::FE::DiscretizationCreator<CloneStrategy>> clonewizard =
          std::make_shared<Core::FE::DiscretizationCreator<CloneStrategy>>();

      std::map<int, int> matmap;
      clonewizard->create_clone_field_mat_map(matmap, sourcedis, targetdis, clonefieldmatmap);

      clonewizard->create_matching_discretization_from_condition(
          sourcedis, condname, targetdis, matmap);
    }
    if (Core::Communication::my_mpi_rank(comm) == 0)
    {
      Core::IO::cout << "Created discretization " << targetdis.name()
                     << " as a clone from the condition \"" << condname.c_str()
                     << "\" of the discretization " << sourcedis.name() << " in...."
                     << time.totalElapsedTime(true) << " secs\n\n";
    }
    return;
  };  // clone_discretization_from_condition

  //! Return valid cloning material map input lines.
  IO::InputSpec valid_cloning_material_map();

}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif

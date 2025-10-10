// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization_hdg.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fem_general_dg_element.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_legacy_enum_definitions_problem_type.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_utils_parameter_list.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

Core::FE::DiscretizationHDG::DiscretizationHDG(
    const std::string name, MPI_Comm comm, const unsigned int n_dim)
    : DiscretizationFaces(name, comm, n_dim)
{
  this->doboundaryfaces_ = true;
}

/*----------------------------------------------------------------------*
 |  Finalize construction (public)                     kronbichler 12/13|
 *----------------------------------------------------------------------*/
int Core::FE::DiscretizationHDG::fill_complete(OptionsFillComplete options)
{
  // call FillCompleteFaces of base class with create_faces set to true
  this->fill_complete_faces(options, true);

  // get the correct face orientation from the owner. since the elements in general do not allow
  // packing, extract the node ids, communicate them, and change the node ids in the element
  Core::Communication::Exporter nodeexporter(*facerowmap_, *facecolmap_, get_comm());
  std::map<int, std::vector<int>> nodeIds, trafoMap;
  for (std::map<int, std::shared_ptr<Core::Elements::FaceElement>>::const_iterator f =
           faces_.begin();
      f != faces_.end(); ++f)
  {
    std::vector<int> ids(f->second->num_node());
    for (int i = 0; i < f->second->num_node(); ++i) ids[i] = f->second->node_ids()[i];
    nodeIds[f->first] = ids;
    trafoMap[f->first] = f->second->get_local_trafo_map();
  }

  nodeexporter.do_export(nodeIds);
  nodeexporter.do_export(trafoMap);

  for (std::map<int, std::shared_ptr<Core::Elements::FaceElement>>::iterator f = faces_.begin();
      f != faces_.end(); ++f)
  {
    if (f->second->owner() == Core::Communication::my_mpi_rank(get_comm())) continue;
    std::vector<int>& ids = nodeIds[f->first];
    FOUR_C_ASSERT(ids.size() > 0, "Lost a face during communication");
    f->second->set_node_ids(ids.size(), ids.data());
    f->second->set_local_trafo_map(trafoMap[f->first]);

    // refresh node pointers if they have been set up
    Core::Nodes::Node** oldnodes = f->second->nodes();
    if (oldnodes != nullptr)
    {
      std::vector<Core::Nodes::Node*> nodes(ids.size(), nullptr);

      for (unsigned int i = 0; i < ids.size(); ++i)
      {
        for (unsigned int j = 0; j < ids.size(); ++j)
          if (oldnodes[j]->id() == ids[i])
          {
            nodes[i] = oldnodes[j];
          }
        FOUR_C_ASSERT(nodes[i] != nullptr, "Could not find node.");
      }
      f->second->build_nodal_pointers(nodes.data());
    }

    // check master/slave relation of current face in terms of the local trafo map
    FOUR_C_ASSERT(f->second->parent_master_element() != nullptr,
        "Unexpected topology between face and parent");
    const int* nodeIdsMaster = f->second->parent_master_element()->node_ids();
    const int* nodeIds = f->second->node_ids();

    std::vector<std::vector<int>> faceNodeOrder =
        Core::FE::get_ele_node_numbering_faces(f->second->parent_master_element()->shape());

    bool exchangeMasterAndSlave = false;
    for (int i = 0; i < f->second->num_node(); ++i)
    {
      // TODO (MK): check that this is enough also on periodic B.C. where the
      // node ids are different in any case...
      if (nodeIdsMaster[faceNodeOrder[f->second->face_master_number()][i]] != nodeIds[i])
        exchangeMasterAndSlave = true;
    }
    if (exchangeMasterAndSlave)
    {
      Core::Elements::Element* faceMaster = f->second->parent_master_element();
      const int faceMasterNo = f->second->face_master_number();
      // new master element might be nullptr on MPI computations
      f->second->set_parent_master_element(f->second->parent_slave_element(),
          f->second->parent_slave_element() != nullptr ? f->second->face_slave_number() : -1);
      f->second->set_parent_slave_element(faceMaster, faceMasterNo);
    }
  }

  // add dofsets:
  // nds = 0 used for trace values
  // nds = 1 used for interior values
  // nds = 2 used for nodal ALE values
  if (this->num_my_row_elements())
    if (this->l_row_element(0)->element_type().name() == "FluidHDGWeakCompType")
    {
      // add nds 1
      if (this->num_dof_sets() == 1)
      {
        int ndof_ele = this->num_my_row_elements() > 0
                           ? dynamic_cast<Core::Elements::DgElement*>(this->l_row_element(0))
                                 ->num_dof_per_element_auxiliary()
                           : 0;
        std::shared_ptr<Core::DOFSets::DofSetInterface> dofset_ele =
            std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(0, ndof_ele, 0, false);

        this->add_dof_set(dofset_ele);
      }

      // add nds 2
      if (this->num_dof_sets() == 2)
      {
        int ndof_node = this->num_my_row_elements() > 0
                            ? dynamic_cast<Core::Elements::DgElement*>(this->l_row_element(0))
                                  ->num_dof_per_node_auxiliary()
                            : 0;
        std::shared_ptr<Core::DOFSets::DofSetInterface> dofset_node =
            std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(ndof_node, 0, 0, false);

        this->add_dof_set(dofset_node);
      }
    }

  return 0;
}


/*----------------------------------------------------------------------*
 | assign_global_ids                                        schoeder 06/14|
 *----------------------------------------------------------------------*/
void Core::FE::DiscretizationHDG::assign_global_ids(MPI_Comm comm,
    const std::map<std::vector<int>, std::shared_ptr<Core::Elements::Element>>& elementmap,
    std::map<int, std::shared_ptr<Core::Elements::Element>>& finalelements)
{
  // The point here is to make sure the element gid are the same on any
  // parallel distribution of the elements. Thus we allreduce thing to
  // processor 0 and sort the element descriptions (vectors of nodal ids)
  // there. We also communicate the element degree! This is the difference
  // the base class function!
  //
  // This routine has not been optimized for efficiency. I don't think that is
  // needed.
  //
  // pack elements on all processors

  int size = 0;
  std::map<std::vector<int>, std::shared_ptr<Core::Elements::Element>>::const_iterator elemsiter;
  for (elemsiter = elementmap.begin(); elemsiter != elementmap.end(); ++elemsiter)
  {
    size += elemsiter->first.size() + 2;
  }

  std::vector<int> sendblock;
  sendblock.reserve(size);
  for (elemsiter = elementmap.begin(); elemsiter != elementmap.end(); ++elemsiter)
  {
    sendblock.push_back(elemsiter->first.size());
    sendblock.push_back(elemsiter->second->degree());
    std::copy(elemsiter->first.begin(), elemsiter->first.end(), std::back_inserter(sendblock));
  }

  // communicate elements to processor 0

  int mysize = sendblock.size();
  size = Core::Communication::sum_all(mysize, comm);
  int mypos = Core::LinAlg::find_my_pos(sendblock.size(), comm);

  std::vector<int> send(size);
  std::ranges::fill(send, 0);
  std::ranges::copy(sendblock, send.data() + mypos);
  sendblock.clear();
  std::vector<int> recv(size);
  recv = Core::Communication::sum_all(send, comm);
  send.clear();

  // unpack, unify and sort elements on processor 0

  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::map<std::vector<int>, int> elementsanddegree;
    int index = 0;
    while (index < static_cast<int>(recv.size()))
    {
      int esize = recv[index];
      int degree = recv[index + 1];
      index += 2;
      std::vector<int> element;
      element.reserve(esize);
      std::copy_n(recv.data() + index, esize, std::back_inserter(element));
      index += esize;

      // check if we already have this and if so, check for max degree
      std::map<std::vector<int>, int>::const_iterator iter = elementsanddegree.find(element);
      if (iter != elementsanddegree.end())
      {
        degree = iter->second > degree ? iter->second : degree;
        elementsanddegree.erase(
            element);  // is only inserted in the next line, if the entry does not exist
      }
      elementsanddegree.insert(std::pair<std::vector<int>, int>(element, degree));
    }
    recv.clear();

    // pack again to distribute pack to all processors
    std::map<std::vector<int>, int>::const_iterator iter;
    send.reserve(index);

    for (iter = elementsanddegree.begin(); iter != elementsanddegree.end(); ++iter)
    {
      send.push_back(iter->first.size());
      send.push_back(iter->second);
      std::copy(iter->first.begin(), iter->first.end(), std::back_inserter(send));
    }
    size = send.size();
  }
  else
  {
    recv.clear();
  }

  // broadcast sorted elements to all processors

  Core::Communication::broadcast(&size, 1, 0, comm);
  send.resize(size);
  Core::Communication::broadcast(send.data(), send.size(), 0, comm);

  // Unpack sorted elements. Take element position for gid.

  int index = 0;
  int gid = 0;
  while (index < static_cast<int>(send.size()))
  {
    int esize = send[index];
    index += 2;
    std::vector<int> element;
    element.reserve(esize);
    std::copy_n(send.data() + index, esize, std::back_inserter(element));
    index += esize;

    // set gid to my elements
    std::map<std::vector<int>, std::shared_ptr<Core::Elements::Element>>::const_iterator iter =
        elementmap.find(element);
    if (iter != elementmap.end())
    {
      iter->second->set_id(gid);

      finalelements[gid] = iter->second;
    }

    gid += 1;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::ostream& operator<<(std::ostream& os, const Core::FE::DiscretizationHDG& dis)
{
  // print standard discretization info
  dis.print(os);
  // print additional info about internal faces
  dis.print_faces(os);

  return os;
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::DbcHDG::read_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::Discretization& discret, const Core::Conditions::Condition& cond, double time,
    Core::FE::Dbc::DbcInfo& info, const std::shared_ptr<std::set<int>>* dbcgids,
    int hierarchical_order) const
{
  // no need to check the cast, because it has been done during
  // the build process (see build_dbc())
  const Core::FE::DiscretizationFaces& face_discret =
      static_cast<const Core::FE::DiscretizationFaces&>(discret);

  read_dirichlet_condition(params, face_discret, cond, time, info, dbcgids, hierarchical_order);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::DbcHDG::read_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::DiscretizationFaces& discret, const Core::Conditions::Condition& cond,
    double time, Core::FE::Dbc::DbcInfo& info, const std::shared_ptr<std::set<int>>* dbcgids,
    int hierarchical_order) const

{
  // call to corresponding method in base class; safety checks inside
  Core::FE::Dbc::read_dirichlet_condition(
      params, discret, cond, time, info, dbcgids, hierarchical_order);

  // say good bye if there are no face elements
  if (discret.face_row_map() == nullptr) return;

  // get onoff toggles
  const auto& onoff = cond.parameters().get<std::vector<int>>("ONOFF");

  if (discret.num_my_row_faces() > 0)
  {
    // initialize with true on each proc except proc 0
    bool pressureDone = Core::Communication::my_mpi_rank(discret.get_comm()) != 0;

    // loop over all faces
    for (int i = 0; i < discret.num_my_row_faces(); ++i)
    {
      const Core::Elements::FaceElement* faceele =
          dynamic_cast<const Core::Elements::FaceElement*>(discret.l_row_face(i));
      const unsigned int dofperface =
          faceele->parent_master_element()->num_dof_per_face(faceele->face_master_number());
      const unsigned int dofpercomponent =
          faceele->parent_master_element()->num_dof_per_component(faceele->face_master_number());
      const unsigned int component = dofperface / dofpercomponent;

      if (onoff.size() <= component || onoff[component] == 0 ||
          *params.get<const Core::ProblemType*>("problem_type") != Core::ProblemType::fluid)
        pressureDone = true;
      if (!pressureDone)
      {
        if (discret.num_my_row_elements() > 0 &&
            Core::Communication::my_mpi_rank(discret.get_comm()) == 0)
        {
          std::vector<int> predof = discret.dof(0, discret.l_row_element(0));
          const int gid = predof[0];
          const int lid = discret.dof_row_map(0)->lid(gid);

          // set toggle vector
          info.toggle.get_local_values()[lid] = 1;
          // amend vector of DOF-IDs which are Dirichlet BCs
          if (dbcgids[set_row] != nullptr) (*dbcgids[set_row]).insert(gid);
          pressureDone = true;
        }
      }

      // do only faces where all nodes are present in the node list
      bool faceRelevant = true;
      int nummynodes = discret.l_row_face(i)->num_node();
      const int* mynodes = discret.l_row_face(i)->node_ids();
      for (int j = 0; j < nummynodes; ++j)
        if (!cond.contains_node(mynodes[j]))
        {
          faceRelevant = false;
          break;
        }
      if (!faceRelevant) continue;

      // get dofs of current face element
      std::vector<int> dofs = discret.dof(0, discret.l_row_face(i));

      // loop over dofs
      for (unsigned int j = 0; j < dofperface; ++j)
      {
        // get global id
        const int gid = dofs[j];
        // get corresponding local id
        const int lid = info.toggle.get_map().lid(gid);
        if (lid < 0)
          FOUR_C_THROW("Global id {} not on this proc {} in system vector", dofs[j],
              Core::Communication::my_mpi_rank(discret.get_comm()));
        // get position of label for this dof in condition line
        int onesetj = j / dofpercomponent;

        if (onoff[onesetj] == 0)
        {
          // no DBC on this dof, set toggle zero
          info.toggle.get_local_values()[lid] = 0;
          // get rid of entry in DBC map - if it exists
          if (dbcgids[set_row] != nullptr) (*dbcgids[set_row]).erase(gid);
          continue;
        }
        else  // if (onoff[onesetj]==1)
        {
          // dof has DBC, set toggle vector one
          info.toggle.get_local_values()[lid] = 1;
          // amend vector of DOF-IDs which are dirichlet BCs
          if (dbcgids[set_row] != nullptr) (*dbcgids[set_row]).insert(gid);
        }

      }  // loop over DOFs of face
    }  // loop over all faces
  }  // if there are faces

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::DbcHDG::do_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::Discretization& discret, const Core::Conditions::Condition& cond, double time,

    const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
    const Core::LinAlg::Vector<int>& toggle, const std::shared_ptr<std::set<int>>* dbcgids) const
{
  // no need to check the cast, because it has been done during
  // the build process (see build_dbc())
  const Core::FE::DiscretizationFaces& face_discret =
      static_cast<const Core::FE::DiscretizationFaces&>(discret);

  do_dirichlet_condition(params, face_discret, cond, time, systemvectors, toggle);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::DbcHDG::do_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::DiscretizationFaces& discret, const Core::Conditions::Condition& cond,
    double time, const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
    const Core::LinAlg::Vector<int>& toggle) const
{
  // call corresponding method from base class; safety checks inside
  Core::FE::Dbc::do_dirichlet_condition(
      params, discret, cond, time, systemvectors, toggle, nullptr);

  // say good bye if there are no face elements
  if (discret.face_row_map() == nullptr) return;

  // get ids of conditioned nodes
  const std::vector<int>* nodeids = cond.get_nodes();
  if (!nodeids) FOUR_C_THROW("Dirichlet condition does not have nodal cloud");

  // get curves, functs, vals, and onoff toggles from the condition
  const auto funct = cond.parameters().get<std::vector<std::optional<int>>>("FUNCT");
  const auto val = cond.parameters().get<std::vector<double>>("VAL");
  const auto onoff = cond.parameters().get<std::vector<int>>("ONOFF");

  // determine highest degree of time derivative
  // and first existent system vector to apply DBC to
  unsigned deg = 0;  // highest degree of requested time derivative
  std::shared_ptr<Core::LinAlg::Vector<double>> systemvectoraux =
      nullptr;  // auxiliary system vector
  if (systemvectors[0] != nullptr)
  {
    deg = 0;
    systemvectoraux = systemvectors[0];
  }
  if (systemvectors[1] != nullptr)
  {
    deg = 1;
    if (systemvectoraux == nullptr) systemvectoraux = systemvectors[1];
  }
  if (systemvectors[2] != nullptr)
  {
    deg = 2;
    if (systemvectoraux == nullptr) systemvectoraux = systemvectors[2];
  }

  // do we have faces?
  if (discret.num_my_row_faces() > 0)
  {
    Core::LinAlg::SerialDenseVector elevec1, elevec2, elevec3;
    Core::LinAlg::SerialDenseMatrix elemat1, elemat2;
    Core::Elements::LocationArray dummy(1);
    Teuchos::ParameterList initParams;

    const auto problem_type = *params.get<const Core::ProblemType*>("problem_type");
    if (problem_type == Core::ProblemType::scatra)
    {
      initParams.set("hdg_action", true);
      Core::Utils::add_enum_class_to_parameter_list<Core::FE::HDGAction>(
          "action", Core::FE::HDGAction::project_dirich_field, initParams);
    }

    // TODO: Introduce a general action type that is
    // valid for all problems
    std::vector<int> funct_without_nones(funct.size());
    for (unsigned int i = 0; i < funct.size(); ++i)
    {
      if (funct[i].has_value() && funct[i].value() > 0)
        funct_without_nones[i] = funct[i].value();
      else
        funct_without_nones[i] = -1;
    }

    Teuchos::Array<int> functarray(funct_without_nones);
    initParams.set("funct", functarray);

    Teuchos::Array<int> onoffarray(onoff);
    initParams.set("onoff", onoffarray);
    initParams.set("time", time);

    // initialize with true if proc is not proc 0
    bool pressureDone = Core::Communication::my_mpi_rank(discret.get_comm()) != 0;

    // loop over all faces
    for (int i = 0; i < discret.num_my_row_faces(); ++i)
    {
      const Core::Elements::FaceElement* faceele =
          dynamic_cast<const Core::Elements::FaceElement*>(discret.l_row_face(i));
      const unsigned int dofperface =
          faceele->parent_master_element()->num_dof_per_face(faceele->face_master_number());
      const unsigned int dofpercomponent =
          faceele->parent_master_element()->num_dof_per_component(faceele->face_master_number());
      const unsigned int component = dofperface / dofpercomponent;

      if (onoff.size() <= component || onoff[component] == 0 ||
          *params.get<const Core::ProblemType*>("problem_type") != Core::ProblemType::fluid)
        pressureDone = true;
      if (!pressureDone)
      {
        if (discret.num_my_row_elements() > 0 &&
            Core::Communication::my_mpi_rank(discret.get_comm()) == 0)
        {
          std::vector<int> predof = discret.dof(0, discret.l_row_element(0));
          const int gid = predof[0];
          const int lid = discret.dof_row_map(0)->lid(gid);

          // amend vector of DOF-IDs which are Dirichlet BCs
          if (systemvectors[0] != nullptr) (*systemvectors[0]).get_values()[lid] = 0.0;
          if (systemvectors[1] != nullptr) (*systemvectors[1]).get_values()[lid] = 0.0;
          if (systemvectors[2] != nullptr) (*systemvectors[2]).get_values()[lid] = 0.0;

          // --------------------------------------------------------------------------------------
          pressureDone = true;
        }
      }
      int nummynodes = discret.l_row_face(i)->num_node();
      const int* mynodes = discret.l_row_face(i)->node_ids();

      // do only faces where all nodes are present in the node list
      bool faceRelevant = true;
      for (int j = 0; j < nummynodes; ++j)
        if (!cond.contains_node(mynodes[j]))
        {
          faceRelevant = false;
          break;
        }
      if (!faceRelevant) continue;

      initParams.set<unsigned int>(
          "faceconsider", static_cast<unsigned int>(faceele->face_master_number()));
      if (static_cast<unsigned int>(elevec1.numRows()) != dofperface) elevec1.shape(dofperface, 1);
      std::vector<int> dofs = discret.dof(0, discret.l_row_face(i));

      bool do_evaluate = false;
      for (unsigned int i = 0; i < component; ++i)
        if (funct[i] > 0) do_evaluate = true;

      if (do_evaluate)
      {
        // cast the const qualifier away, thus the Evaluate routine can be called.
        Core::FE::DiscretizationFaces& non_const_dis =
            const_cast<Core::FE::DiscretizationFaces&>(discret);
        faceele->parent_master_element()->evaluate(
            initParams, non_const_dis, dummy, elemat1, elemat2, elevec1, elevec2, elevec3);
      }
      else
        for (unsigned int i = 0; i < dofperface; ++i) elevec1(i) = 1.;

      // loop over face dofs
      for (unsigned int j = 0; j < dofperface; ++j)
      {
        // get global id
        const int gid = dofs[j];
        // get corresponding local id
        const int lid = toggle.get_map().lid(gid);
        if (lid < 0)
          FOUR_C_THROW("Global id {} not on this proc {} in system vector", dofs[j],
              Core::Communication::my_mpi_rank(discret.get_comm()));
        // get position of label for this dof in condition line
        int onesetj = j / dofpercomponent;

        // check whether dof gid is a dbc gid
        if (toggle.get_local_values()[lid] == 0) continue;

        std::vector<double> value(deg + 1, val[onesetj]);

        // assign value
        if (systemvectors[0] != nullptr)
          (*systemvectors[0]).get_values()[lid] = value[0] * elevec1(j);
        if (systemvectors[1] != nullptr)
          (*systemvectors[1]).get_values()[lid] = value[1] * elevec1(j);
        if (systemvectors[2] != nullptr)
          (*systemvectors[2]).get_values()[lid] = value[2] * elevec1(j);

      }  // loop over all DOFs
    }  // loop over all faces

  }  // if there are faces

  return;
}

FOUR_C_NAMESPACE_CLOSE

/*!
\file xfem_condition_manager.cpp

\brief manages the different types of mesh and level-set based coupling conditions and thereby builds the bridge between the
xfluid class and the cut-library

<pre>
Maintainer: Benedikt Schott
            schott@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15241
</pre>
*/



#include <Teuchos_TimeMonitor.hpp>

#include "xfem_condition_manager.H"
#include "xfem_utils.H"

#include "../drt_inpar/inpar_xfem.H"

#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_globalproblem.H"

#include "../drt_lib/drt_condition_utils.H"
#include "../drt_lib/drt_condition_selector.H"
#include "../drt_lib/drt_discret_faces.H"
#include "../drt_lib/drt_discret_xfem.H"
#include "../drt_lib/drt_dofset_transparent_independent.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_utils_parallel.H"

#include "../drt_fluid_ele/fluid_ele_action.H"
#include "../drt_fluid_ele/fluid_ele_parameter_xfem.H"

#include "../linalg/linalg_utils.H"

#include "../drt_inpar/inpar_xfem.H"
#include "../drt_inpar/inpar_fluid.H"

#include "../drt_crack/crackUtils.H"

#include "../drt_io/io.H"
#include "../drt_io/io_gmsh.H"
#include "../drt_io/io_control.H"
#include "../drt_io/io_pstream.H"

void XFEM::CouplingBase::SetElementConditions()
{
  // number of column cutter boundary elements
  int nummycolele = cutter_dis_->NumMyColElements();

  cutterele_conds_.clear();
  cutterele_conds_.reserve(nummycolele);

  // initialize the vector invalid coupling-condition type "NONE"
  EleCoupCond init_pair = EleCoupCond(INPAR::XFEM::CouplingCond_NONE,NULL);
  for(int lid=0; lid<nummycolele; lid++) cutterele_conds_.push_back(init_pair);

  //-----------------------------------------------------------------------------------
  // loop all column cutting elements on this processor
  for(int lid=0; lid<nummycolele; lid++)
  {
    DRT::Element* cutele = cutter_dis_->lColElement(lid);

    // loop all possible XFEM-coupling conditions
    for(size_t cond=0; cond < conditions_to_copy_.size(); cond++)
    {
      // get all conditions with given condition name
      std::vector<DRT::Condition*> mycond;
      DRT::UTILS::FindElementConditions(cutele, conditions_to_copy_[cond], mycond);

      // safety checks
      if (mycond.size()>1)
      {
//        dserror("%i conditions of the same name for element %i! %s coupling-condition not unique!", mycond.size(), cutele->Id(), conditions_to_copy_[cond].c_str());
      }
      else if(mycond.size() == 0)
      {
        continue; // try the next condition type
      }
      else
      {
//        std::cout << "unique condition found!" << std::endl;
      }

      INPAR::XFEM::EleCouplingCondType cond_type = CondType_stringToEnum(conditions_to_copy_[cond]);

      // non-coupling condition found (e.g. FSI coupling)
      if(cond_type == INPAR::XFEM::CouplingCond_NONE) continue;

      // non-unique conditions for one cutter element
      if( cutterele_conds_[lid].first != INPAR::XFEM::CouplingCond_NONE )
      {
        dserror("There are two different condition types for the same cutter dis element with id %i: 1st %i, 2nd %i. Make the XFEM coupling conditions unique!",
            cutele->Id(), cutterele_conds_[lid].first, cond_type);
      }

      // store the unique condition pointer to the cutting element
      cutterele_conds_[lid] = EleCoupCond(cond_type, mycond[0]);
    }
  }

  //-----------------------------------------------------------------------------------
  // check if all column cutter elements have a valid condition type
  // loop all column cutting elements on this processor
  for(int lid=0; lid<nummycolele; lid++)
  {
    if(cutterele_conds_[lid].first == INPAR::XFEM::CouplingCond_NONE)
      dserror("cutter element with local id %i has no valid coupling-condition", lid);
  }

}

void XFEM::CouplingBase::SetCouplingStrategy()
{
  const INPAR::XFEM::EleCouplingCondType cond_type = CondType_stringToEnum(cond_name_);

  switch(cond_type)
  {
  case INPAR::XFEM::CouplingCond_SURF_FSI_MONO:
  {
    coupling_strategy_ = INPAR::XFEM::Xfluid_Sided_Coupling; //TODO: rename to xfluid (only one for coupling and wdbc/Neumann)
    break;
  }
  case INPAR::XFEM::CouplingCond_SURF_FLUIDFLUID:
  {
    // ask the first cutter element
    const int lid=0;
    const int val = cutterele_conds_[lid].second->GetInt("COUPSTRATEGY");
    coupling_strategy_ = static_cast<INPAR::XFEM::CouplingStrategy>(val);
    break;
  }
  case INPAR::XFEM::CouplingCond_LEVELSET_TWOPHASE:
  case INPAR::XFEM::CouplingCond_LEVELSET_COMBUSTION:
  {
    coupling_strategy_ = INPAR::XFEM::Harmonic;
    break;
  }
  case INPAR::XFEM::CouplingCond_SURF_FSI_PART:
  case INPAR::XFEM::CouplingCond_SURF_CRACK_FSI_PART:
  case INPAR::XFEM::CouplingCond_SURF_WEAK_DIRICHLET: // set this to Teuchos::null when the values are read from the function instead of the ivelnp vector
  case INPAR::XFEM::CouplingCond_SURF_NEUMANN:
  case INPAR::XFEM::CouplingCond_LEVELSET_WEAK_DIRICHLET:
  case INPAR::XFEM::CouplingCond_LEVELSET_NEUMANN:
  {
    coupling_strategy_ = INPAR::XFEM::Xfluid_Sided_weak_DBC; //TODO: rename to xfluid
    break;
  }
  default: dserror("which is the coupling discretization for this type of coupling %i?", cond_type); break;
  }
}


void XFEM::CouplingBase::SetCouplingDiscretization()
{
  const INPAR::XFEM::EleCouplingCondType cond_type = CondType_stringToEnum(cond_name_);

  switch(cond_type)
  {
  case INPAR::XFEM::CouplingCond_SURF_FSI_MONO:
  {
    coupl_dis_ = cutter_dis_;
    break;
  }
  case INPAR::XFEM::CouplingCond_SURF_FLUIDFLUID:
  {
    // depending on the weighting strategy
    if(coupling_strategy_==INPAR::XFEM::Xfluid_Sided_Coupling)
    {
      coupl_dis_ = cutter_dis_;
    }
    else if(coupling_strategy_==INPAR::XFEM::Embedded_Sided_Coupling or
        coupling_strategy_==INPAR::XFEM::Two_Sided_Coupling )
    {
      coupl_dis_ = cond_dis_;
    }
    else dserror("invalid coupling strategy for fluid-fluid application");
    break;
  }
  case INPAR::XFEM::CouplingCond_LEVELSET_TWOPHASE:
  case INPAR::XFEM::CouplingCond_LEVELSET_COMBUSTION:
  {
    coupl_dis_ = bg_dis_;
    break;
  }
  case INPAR::XFEM::CouplingCond_SURF_FSI_PART:
  case INPAR::XFEM::CouplingCond_SURF_CRACK_FSI_PART:
  case INPAR::XFEM::CouplingCond_SURF_WEAK_DIRICHLET: // set this to Teuchos::null when the values are read from the function instead of the ivelnp vector
  case INPAR::XFEM::CouplingCond_SURF_NEUMANN:
  {
    coupl_dis_ = cutter_dis_;
    break;
  }
  case INPAR::XFEM::CouplingCond_LEVELSET_WEAK_DIRICHLET:
  case INPAR::XFEM::CouplingCond_LEVELSET_NEUMANN:
  {
    coupl_dis_ = Teuchos::null;
    break;
  }
  default: dserror("which is the coupling discretization for this type of coupling %i?", cond_type); break;
  }
}



XFEM::MeshCoupling::MeshCoupling(
    Teuchos::RCP<DRT::Discretization>&  bg_dis,   ///< background discretization
    const std::string &                 cond_name,///< name of the condition, by which the derived cutter discretization is identified
    Teuchos::RCP<DRT::Discretization>&  cond_dis,  ///< discretization from which the cutter discretization is derived
    const double                        time,      ///< time
    const int                           step       ///< time step
) : CouplingBase(bg_dis, cond_name, cond_dis, time, step)
{

  // set list of conditions that will be copied to the new cutter discretization
  SetConditionsToCopy();

  // create a cutter discretization from conditioned nodes of the given coupling discretization
  CreateCutterDisFromCondition();

  // set unique element conditions
  SetElementConditions();

  // set the coupling strategy
  SetCouplingStrategy();

  // set coupling discretization
  SetCouplingDiscretization();

  // initialize state vectors based on cutter discretization
  InitStateVectors();

}


void XFEM::MeshCoupling::SetConditionsToCopy()
{
  // fill list of conditions that will be copied to the new cutter discretization
  conditions_to_copy_.push_back(cond_name_);

  // additional conditions required for the new boundary conditions
  conditions_to_copy_.push_back("FSICoupling");  // for partitioned and monolithic XFSI

  // additional conditions required for the displacements of the cutter mesh
  conditions_to_copy_.push_back("XFEMSurfDisplacement");
}


/*--------------------------------------------------------------------------*
 | Create the cutter discretization                                         |
 *--------------------------------------------------------------------------*/
void XFEM::MeshCoupling::CreateCutterDisFromCondition()
{
  // create name string for new cutter discretization (e.g, "boundary_of_struct" or "boundary_of_fluid")
  std::string cutterdis_name ("boundary_of_");
  cutterdis_name += cond_dis_->Name();


  //--------------------------------
  // create the new cutter discretization form the conditioned coupling discretization
  cutter_dis_ = DRT::UTILS::CreateDiscretizationFromCondition(
      cond_dis_,                ///< discretization with condition
      cond_name_,               ///< name of the condition, by which the derived discretization is identified
      cutterdis_name,           ///< name of the new discretization
      GetBELEName(cond_dis_),   ///< name/type of the elements to be created
      conditions_to_copy_       ///< list of conditions that will be copied to the new discretization
  );
  //--------------------------------


  if (cutter_dis_->NumGlobalNodes() == 0)
  {
    dserror("Empty cutter discretization detected. No coupling can be performed...");
  }

  // for parallel jobs we have to call TransparentDofSet with additional flag true
  bool parallel = cond_dis_->Comm().NumProc() > 1;
  Teuchos::RCP<DRT::DofSet> newdofset = Teuchos::rcp(new
      DRT::TransparentIndependentDofSet(cond_dis_,parallel));

  cutter_dis_->ReplaceDofSet(newdofset); //do not call this with true!!
  cutter_dis_->FillComplete();

  // create node and element distribution with elements and nodes ghosted on all processors
  const Epetra_Map noderowmap = *cutter_dis_->NodeRowMap();
  const Epetra_Map elemrowmap = *cutter_dis_->ElementRowMap();

  // put all boundary nodes and elements onto all processors
  const Epetra_Map nodecolmap = *LINALG::AllreduceEMap(noderowmap);
  const Epetra_Map elemcolmap = *LINALG::AllreduceEMap(elemrowmap);

  // redistribute nodes and elements to column (ghost) map
  cutter_dis_->ExportColumnNodes(nodecolmap);
  cutter_dis_->ExportColumnElements(elemcolmap);

  cutter_dis_->FillComplete();
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCoupling::GmshOutputDiscretization(
  std::ostream& gmshfilecontent
)
{
  // compute the current boundary position
  std::map<int,LINALG::Matrix<3,1> > currinterfacepositions;

  // output of cutting discretization
  XFEM::UTILS::ExtractNodeVectors(cutter_dis_, currinterfacepositions);
  XFEM::UTILS::PrintDiscretizationToStream(cutter_dis_,
     cutter_dis_->Name(), true, true,  true, true,  false, false, gmshfilecontent, &currinterfacepositions);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
XFEM::MeshCouplingFluidFluid::MeshCouplingFluidFluid(
    Teuchos::RCP<DRT::Discretization>&  bg_dis,   ///< background discretization
    const std::string &                 cond_name,///< name of the condition, by which the derived cutter discretization is identified
    Teuchos::RCP<DRT::Discretization>&  cond_dis,  ///< discretization from which cutter discretization can be derived
    const double                        time,      ///< time
    const int                           step       ///< time step
) : MeshCoupling(bg_dis,cond_name,cond_dis,time,step)
{
  if (GetCouplingStrategy() == INPAR::XFEM::Embedded_Sided_Coupling ||
      GetCouplingStrategy() == INPAR::XFEM::Two_Sided_Coupling)
  {
    // ghost coupling elements, that contribute to the cutting discretization
    RedistributeEmbeddedDiscretization();
    // create map from side to embedded element ID
    CreateCuttingToEmbeddedElementMap();

    // Todo: create only for Nitsche+EVP & EOS on outer embedded elements
    CreateAuxiliaryDiscretization();
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFluidFluid::RedistributeForErrorCalculation()
{
  if (GetCouplingStrategy() == INPAR::XFEM::Embedded_Sided_Coupling ||
      GetCouplingStrategy() == INPAR::XFEM::Two_Sided_Coupling)
    return;
  // ghost coupling elements, that contribute to the cutting discretization
  RedistributeEmbeddedDiscretization();
  // create map from side to embedded element ID
  CreateCuttingToEmbeddedElementMap();
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFluidFluid::RedistributeEmbeddedDiscretization()
{
//#ifdef DEBUG
//  // collect conditioned nodes and compare to the overall number of nodes in
//  // the surface discretization
//  std::vector<DRT::Condition*> cnd;
//  cond_dis_->GetCondition(cond_name_,cnd);
//
//  // get the set of ids of all xfem nodes
//  std::set<int> cond_nodeset;
//  {
//    for (size_t cond = 0; cond< cnd.size(); ++ cond)
//    {
//      // conditioned node ids
//      const std::vector<int>* nodeids_cnd = cnd[cond]->Nodes();
//      for (std::vector<int>::const_iterator c = nodeids_cnd->begin();
//           c != nodeids_cnd->end(); ++c)
//        cond_nodeset.insert(*c);
//    }
//  }
//
//  if (cond_nodeset.size() != static_cast<size_t>(cutter_dis_->NumGlobalNodes()))
//    dserror("Got %d %s nodes but have % dnodes in the boundary discretization created from the condition",
//        cond_nodeset.size(), cond_name_.c_str(), cutter_dis_->NumGlobalNodes());
//#endif

  // get gids of elements (and associated notes), that contribute to the fluid-fluid interface
  std::set<int> adj_eles_row;
  std::set<int> adj_ele_nodes_row;

  const int mypid = cond_dis_->Comm().MyPID();

  // STEP 1: Query
  // loop over nodes of cutter discretization (conditioned nodes)
  for (int icondn = 0; icondn != cutter_dis_->NodeRowMap()->NumMyElements(); ++icondn)
  {
    // get node GID
    const int cond_node_gid = cutter_dis_->NodeRowMap()->GID(icondn);

    // node from coupling discretization (is on this proc, as cutter_dis nodes are
    // a subset!)
    const DRT::Node* cond_node = cond_dis_->gNode(cond_node_gid);

    // get associated elements
    const DRT::Element*const* cond_eles = cond_node->Elements();
    const int num_cond_ele = cond_node->NumElement();

    // loop over associated elements
    for (int ie = 0; ie < num_cond_ele; ++ ie)
    {
      if (cond_eles[ie]->Owner() == mypid)
        adj_eles_row.insert(cond_eles[ie]->Id());

      const int * node_ids = cond_eles[ie]->NodeIds();
      for (int in = 0; in < cond_eles[ie]->NumNode(); ++ in)
      {
        if (cond_dis_->gNode(node_ids[in])->Owner() == mypid)
          adj_ele_nodes_row.insert(node_ids[in]);
      }
    }
  }

  // STEP 2 : ghost interface-contributing elements from coupl_dis on all proc

  // collect node & element gids from the auxiliary discetization and
  // store in vector full_{nodes;eles}, which will be appended by the standard
  // column elements/nodes of the discretization we couple with

  LINALG::GatherAll(adj_ele_nodes_row,cond_dis_->Comm());
  LINALG::GatherAll(adj_eles_row,cond_dis_->Comm());

  std::set<int> full_ele_nodes_col(adj_ele_nodes_row);
  std::set<int> full_eles_col(adj_eles_row);

  for (int in = 0; in < cond_dis_->NumMyColNodes(); in++)
  {
    full_ele_nodes_col.insert(cond_dis_->lColNode(in)->Id());
  }
  for (int ie=0; ie < cond_dis_->NumMyColElements(); ie++)
  {
    full_eles_col.insert(cond_dis_->lColElement(ie)->Id());
  }

  // create the final column maps
  {
    std::vector<int> full_nodes(full_ele_nodes_col.begin(),full_ele_nodes_col.end());
    std::vector<int> full_eles(full_eles_col.begin(),full_eles_col.end());

    Teuchos::RCP<const Epetra_Map> full_nodecolmap = Teuchos::rcp(new Epetra_Map(-1, full_nodes.size(), &full_nodes[0], 0, cond_dis_->Comm()));
    Teuchos::RCP<const Epetra_Map> full_elecolmap  = Teuchos::rcp(new Epetra_Map(-1, full_eles.size(), &full_eles[0], 0, cond_dis_->Comm()));

    // redistribute nodes and elements to column (ghost) map
    cond_dis_->ExportColumnNodes(*full_nodecolmap);
    cond_dis_->ExportColumnElements(*full_elecolmap);

    cond_dis_->FillComplete(true,true, true);
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFluidFluid::CreateAuxiliaryDiscretization()
{
  std::string aux_coup_disname("auxiliary_coupling_");
  aux_coup_disname += cond_dis_->Name();
  aux_coup_dis_ = Teuchos::rcp(new DRT::Discretization(aux_coup_disname,
      Teuchos::rcp(cond_dis_->Comm().Clone())));

  // make the condition known to the auxiliary discretization
  // we use the same nodal ids and therefore we can just copy the conditions
  // get the set of ids of all xfem nodes
  std::vector<DRT::Condition*> xfemcnd;
  cond_dis_->GetCondition(cond_name_,xfemcnd);

  std::set<int> xfemnodeset;

  for (size_t cond = 0; cond < xfemcnd.size(); ++ cond)
  {
    aux_coup_dis_->SetCondition(cond_name_,Teuchos::rcp(new DRT::Condition(*xfemcnd[cond])));
    const std::vector<int>* nodeids_cnd = xfemcnd[cond]->Nodes();
    for (std::vector<int>::const_iterator c = nodeids_cnd->begin();
         c != nodeids_cnd->end(); ++c)
      xfemnodeset.insert(*c);
  }

  // determine sets of nodes next to xfem nodes
  std::set<int> adjacent_row;
  std::set<int> adjacent_col;

  // loop all column elements and label all row nodes next to a xfem node
  for (int i=0; i<cond_dis_->NumMyColElements(); ++i)
  {
    DRT::Element* actele = cond_dis_->lColElement(i);

    // get the node ids of this element
    const int  numnode = actele->NumNode();
    const int* nodeids = actele->NodeIds();

    bool found=false;

    // loop the element's nodes, check if a xfem condition is active
    for (int n=0; n<numnode; ++n)
    {
      const int node_gid(nodeids[n]);
      std::set<int>::iterator curr = xfemnodeset.find(node_gid);
      found = (curr!=xfemnodeset.end());
      if (found) break;
    }

    if (!found) continue;

    // if at least one of the element's nodes holds a xfem condition,
    // add all node gids to the adjecent node sets
    for (int n=0; n<numnode; ++n)
    {
      const int node_gid(nodeids[n]);
      // yes, we have a xfem condition:
      // node stored on this proc? add to the set of row nodes!
      if (coupl_dis_->NodeRowMap()->MyGID(node_gid))
        adjacent_row.insert(node_gid);

      // always add to set of col nodes
      adjacent_col.insert(node_gid);
    }

    // add the element to the discretization
    if (cond_dis_->ElementRowMap()->MyGID(actele->Id()))
    {
      Teuchos::RCP<DRT::Element> bndele =Teuchos::rcp(actele->Clone());
      aux_coup_dis_->AddElement(bndele);
    }
  } // end loop over column elements

  // all row nodes next to a xfem node are now added to the auxiliary discretization
  for (std::set<int>::iterator id=adjacent_row.begin();
       id!=adjacent_row.end(); ++id)
  {
    DRT::Node* actnode = cond_dis_->gNode(*id);
    Teuchos::RCP<DRT::Node> bndnode =Teuchos::rcp(actnode->Clone());
    aux_coup_dis_->AddNode(bndnode);
  }

  // build nodal row & col maps to redistribute the discretization
  Teuchos::RCP<Epetra_Map> newnoderowmap;
  Teuchos::RCP<Epetra_Map> newnodecolmap;

  {
    // copy row/col node gids to std::vector
    // (expected by Epetra_Map ctor)
    std::vector<int> rownodes(adjacent_row.begin(),adjacent_row.end());
    // build noderowmap for new distribution of nodes
    newnoderowmap = Teuchos::rcp(new Epetra_Map(-1,
                                                rownodes.size(),
                                                &rownodes[0],
                                                0,
                                                aux_coup_dis_->Comm()));

    std::vector<int> colnodes(adjacent_col.begin(),adjacent_col.end());

    // build nodecolmap for new distribution of nodes
    newnodecolmap = Teuchos::rcp(new Epetra_Map(-1,
                                                colnodes.size(),
                                                &colnodes[0],
                                                0,
                                                aux_coup_dis_->Comm()));

    aux_coup_dis_->Redistribute(*newnoderowmap,*newnodecolmap,false,false,false);

    // make auxiliary discretization have the same dofs as the coupling discretization
    Teuchos::RCP<DRT::DofSet> newdofset = Teuchos::rcp(new DRT::TransparentIndependentDofSet(cond_dis_,true));
    aux_coup_dis_->ReplaceDofSet(newdofset,false); // do not call this with true (no replacement in static dofsets intended)
    aux_coup_dis_->FillComplete(true,true,true);
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFluidFluid::CreateCuttingToEmbeddedElementMap()
{
  // fill map between boundary (cutting) element id and its corresponding embedded (coupling) element id
  for (int ibele=0; ibele< cutter_dis_->NumMyColElements(); ++ ibele)
  {
    // boundary element and its nodes
    DRT::Element* bele = cutter_dis_->lColElement(ibele);
    const int * bele_node_ids = bele->NodeIds();

    bool bele_found = false;

    // ask all conditioned embedded elements for this boundary element
    for(int iele = 0; iele< cond_dis_->NumMyColElements(); ++iele)
    {
      DRT::Element* ele = cond_dis_->lColElement(iele);
      const int * ele_node_ids = ele->NodeIds();

      // get nodes for every face of the embedded element
      std::vector<std::vector<int> > face_node_map = DRT::UTILS::getEleNodeNumberingFaces(ele->Shape());

      // loop the faces of the element and check node equality for every boundary element
      // Todo: Efficiency?
      for (int f = 0; f < ele->NumFace(); f++)
      {
        bele_found = true;

        const int face_numnode = face_node_map[f].size();

        if(bele->NumNode() != face_numnode) continue; // this face cannot be the right one

        // check all nodes of the boundary element
        for(int inode=0; inode<bele->NumNode();  ++inode)
        {
          // boundary node
          const int belenodeId = bele_node_ids[inode];

          bool node_found = false;
          for (int fnode=0; fnode<face_numnode; ++fnode)
          {
            const int facenodeId = ele_node_ids[face_node_map[f][fnode]];

            if(facenodeId == belenodeId)
            {
              // nodes are the same
              node_found = true;
              break;
            }
          } // loop nodes of element's face
          if (node_found==false) // this node is not contained in this face
          {
            bele_found = false; // element not the right one, if at least one boundary node is not found
            break; // node not found
          }
        } // loop nodes of boundary element


        if (bele_found)
        {
          cutting_emb_gid_map_.insert(std::pair<int,int>(bele->Id(),ele->Id()));
          cutting_emb_face_lid_map_.insert(std::pair<int,int>(bele->Id(),f));
          break;
        }
      } // loop element faces
      if (bele_found) break; // do not continue the search

    }

    if(bele_found == false)
      dserror("Corresponding embedded element for boundary element id %i not found on proc %i ! Please ghost corresponding embedded elements on all procs!",
        bele->Id(), cond_dis_->Comm().MyPID());
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFluidFluid::EstimateNitscheTraceMaxEigenvalue(
    const Teuchos::RCP<const Epetra_Vector> & dispnp) const
{
  Teuchos::ParameterList params;

  // set action for elements
  params.set<int>("action",FLD::estimate_Nitsche_trace_maxeigenvalue_);

  Teuchos::RCP<Epetra_Vector> aux_coup_dispnp = LINALG::CreateVector(*aux_coup_dis_->DofRowMap(),true);
  LINALG::Export(*dispnp,*aux_coup_dispnp);

  aux_coup_dis_->SetState("dispnp", aux_coup_dispnp);

  /// map of embedded element ID to the value of it's Nitsche parameter
  Teuchos::RCP<std::map<int,double> > ele_to_max_eigenvalue =  Teuchos::rcp(new std::map<int,double> ());
  params.set<Teuchos::RCP<std::map<int,double > > >("trace_estimate_max_eigenvalue_map", ele_to_max_eigenvalue);

  Teuchos::RCP<LINALG::SparseOperator> systemmatrixA;
  Teuchos::RCP<LINALG::SparseOperator> systemmatrixB;

  // Evaluate the general eigenvalue problem Ax = lambda Bx for local for the elements of aux_coup_dis_
  aux_coup_dis_->EvaluateCondition(  params,
                                     systemmatrixA,
                                     systemmatrixB,
                                     Teuchos::null,
                                     Teuchos::null,
                                     Teuchos::null,
                                     "XFEMSurfFluidFluid");

  // gather the information form all processors
  Teuchos::RCP<std::map<int,double> > tmp_map = params.get<Teuchos::RCP<std::map<int,double > > >("trace_estimate_max_eigenvalue_map");

  // information how many processors work at all
  std::vector<int> allproc(aux_coup_dis_->Comm().NumProc());

  // in case of n processors allproc becomes a vector with entries (0,1,...,n-1)
  for (int i = 0; i < aux_coup_dis_->Comm().NumProc(); ++ i) allproc[i] = i;

  // gather the information from all procs
  LINALG::Gather<int,double>(*tmp_map,*ele_to_max_eigenvalue,(int)aux_coup_dis_->Comm().NumProc(),&allproc[0], aux_coup_dis_->Comm());

  // update the estimate of the maximal eigenvalues in the parameter list to access on element level
  DRT::ELEMENTS::FluidEleParameterXFEM::Instance()->Update_TraceEstimate_MaxEigenvalue(ele_to_max_eigenvalue);
}

void XFEM::MeshCoupling::InitStateVectors()
{
  const Epetra_Map* cutterdofrowmap = cutter_dis_->DofRowMap();

  ivelnp_  = LINALG::CreateVector(*cutterdofrowmap,true);
  iveln_   = LINALG::CreateVector(*cutterdofrowmap,true);
  ivelnm_  = LINALG::CreateVector(*cutterdofrowmap,true);

  idispnp_ = LINALG::CreateVector(*cutterdofrowmap,true);
  idispn_  = LINALG::CreateVector(*cutterdofrowmap,true);
}


void XFEM::MeshCoupling::SetState()
{
  // set general vector values of cutterdis needed by background element evaluate routine
  cutter_dis_->ClearState();

  cutter_dis_->SetState("ivelnp", ivelnp_ );
  cutter_dis_->SetState("iveln",  iveln_  );
  cutter_dis_->SetState("idispnp",idispnp_);
}

void XFEM::MeshCoupling::SetStateDisplacement()
{
  // set general vector values of cutterdis needed by background element evaluate routine
  cutter_dis_->ClearState();

  cutter_dis_->SetState("idispnp",idispnp_);
  cutter_dis_->SetState("idispn",idispn_);

}


void XFEM::MeshCoupling::UpdateStateVectors()
{
  // update velocity n-1
  ivelnm_->Update(1.0,*iveln_,0.0);

  // update velocity n
  iveln_->Update(1.0,*ivelnp_,0.0);

  // update displacement n
  idispn_->Update(1.0,*idispnp_,0.0);
}



Teuchos::RCP<const Epetra_Vector> XFEM::MeshCoupling::GetCutterDispCol()
{
  // export cut-discretization mesh displacements
  Teuchos::RCP<Epetra_Vector> idispcol = LINALG::CreateVector(*cutter_dis_->DofColMap(),true);
  LINALG::Export(*idispnp_,*idispcol);

  return idispcol;
}


//! constructor
XFEM::MeshCouplingBC::MeshCouplingBC(
    Teuchos::RCP<DRT::Discretization>&  bg_dis,   ///< background discretization
    const std::string &                 cond_name,///< name of the condition, by which the derived cutter discretization is identified
    Teuchos::RCP<DRT::Discretization>&  cond_dis,  ///< discretization from which cutter discretization can be derived
    const double                        time,      ///< time
    const int                           step       ///< time step
) : MeshCoupling(bg_dis,cond_name,cond_dis, time, step)
{
  // set the initial interface displacements as they are used for initial cut position at the end of Xfluid::Init()
  SetInterfaceDisplacement();

  // set the interface displacements also to idispn
  idispn_->Update(1.0,*idispnp_,0.0);

  // set the initial interface velocity and possible initialization function
  SetInterfaceVelocity();

  // set the initial interface velocities also to iveln
  iveln_->Update(1.0,*ivelnp_,0.0);
}

void XFEM::MeshCouplingBC::PrepareSolve()
{
  // set the new interface displacements where DBCs or Neumann BCs have to be evaluted
  SetInterfaceDisplacement();

  // set or compute the current prescribed interface velocities, just for XFEM WDBC
  SetInterfaceVelocity();
}

bool XFEM::MeshCouplingBC::HasMovingInterface()
{
  // get the first local col(!) node
  if(cutter_dis_->NumMyColNodes() == 0) dserror("no col node on proc %i", myrank_);

  DRT::Node* lnode = cutter_dis_->lColNode(0);

  std::vector<DRT::Condition*> mycond;
  lnode->GetCondition("XFEMSurfDisplacement",mycond);

  DRT::Condition* cond = mycond[0];

  const std::string* evaltype = cond->Get<std::string>("evaltype");

  if(*evaltype == "zero") return false;

  return true;
}

void XFEM::MeshCouplingBC::EvaluateCondition(
    Teuchos::RCP<Epetra_Vector> ivec,
    const std::string& condname,
    const double time,
    const double dt)
{
  // loop all nodes on the processor
  for(int lnodeid=0;lnodeid<cutter_dis_->NumMyRowNodes();lnodeid++)
  {
    // get the processor local node
    DRT::Node*  lnode      = cutter_dis_->lRowNode(lnodeid);
    // the set of degrees of freedom associated with the node
    const std::vector<int> nodedofset = cutter_dis_->Dof(lnode);

    const int numdof = nodedofset.size();

    if (numdof==0) dserror("node has no dofs");
    std::vector<DRT::Condition*> mycond;
    lnode->GetCondition(condname,mycond);

    // safety check for unique condition
    // TODO: check more if there is more than one condition with the same Couplinglabel
    // check if the coupling label is equal to the coupling condition label
    //      if (mycond.size()!=1)
    //        dserror("Exact one condition is expected on node!");
    //std::cout << "number of conditions " << mycond.size() << std::endl;

    // initial value for all nodal dofs to zero
    std::vector<double> final_values(numdof, 0.0);


    DRT::Condition* cond = mycond[0];

    if(condname == "XFEMSurfDisplacement")
      EvaluateInterfaceDisplacement(final_values,lnode,cond,time);
    else if(condname == "XFEMSurfWeakDirichlet")
      EvaluateInterfaceVelocity(final_values,lnode,cond,time,dt);
    else dserror("non supported condname for evaluation %s", condname.c_str());


    // set final values to vector
    for(int dof=0;dof<numdof;++dof)
    {
      int gid = nodedofset[dof];
      ivec->ReplaceGlobalValues(1,&final_values[dof],&gid);
    }

  } // loop row nodes
}


void XFEM::MeshCouplingBC::EvaluateInterfaceVelocity(
    std::vector<double>& final_values,
    DRT::Node* node,
    DRT::Condition* cond,
    const double time,
    const double dt
)
{
  const std::string* evaltype = cond->Get<std::string>("evaltype");


  if(*evaltype == "zero")
  {
    // take initialized vector with zero values
  }
  else if(*evaltype == "funct_interpolated")
  {
    // evaluate function at node at current time
    EvaluateFunction(final_values,node->X(),cond,time);
  }
  else if(*evaltype == "funct_gausspoint")
  {
    //do nothing, the evaluate routine is called again directly from the Gaussian point
  }
  else if(*evaltype =="displacement_1storder_wo_initfunct" or
          *evaltype =="displacement_2ndorder_wo_initfunct")
  {
    if(step_ == 0) return; // do not compute velocities from displacements at the beginning and do not set

    ComputeInterfaceVelocityFromDisplacement(final_values, node, dt, evaltype);
  }
  else if(
      *evaltype =="displacement_1storder_with_initfunct" or
      *evaltype =="displacement_2ndorder_with_initfunct")
  {
    if(step_ == 0) // evaluate initialization function at node at current time
    {
      EvaluateFunction(final_values,node->X(),cond,time);
    }
    else
      ComputeInterfaceVelocityFromDisplacement(final_values, node, dt, evaltype);
  }
  else dserror("evaltype not supported %s", evaltype->c_str());
}


void XFEM::MeshCouplingBC::EvaluateInterfaceDisplacement(
    std::vector<double>& final_values,
    DRT::Node* node,
    DRT::Condition* cond,
    const double time
)
{
  const std::string* evaltype = cond->Get<std::string>("evaltype");

  if(*evaltype == "zero")
  {
    // take initialized vector with zero values
  }
  else if(*evaltype == "funct")
  {
    // evaluate function at node at current time
    EvaluateFunction(final_values,node->X(),cond,time);
  }
  else if(*evaltype == "implementation")
  {
    // evaluate implementation
    //TODO: get the function name from the condition!!!
    std::string function_name = "ROTATING_BEAM";
    EvaluateImplementation(final_values,node->X(),cond,time,function_name);
  }
  else dserror("evaltype not supported %s", evaltype->c_str());
}


void XFEM::MeshCouplingBC::EvaluateFunction(
    std::vector<double>& final_values,
    const double* x,
    DRT::Condition* cond,
    const double time
)
{
  const int numdof = cond->GetInt("numdof");

  if(numdof != (int)final_values.size())
    dserror("you specified NUMDOF %i in the input file, however, only %i dofs allowed!", numdof, (int)final_values.size());

  // check for potential time curve
  const std::vector<int>* curve  = cond->Get<std::vector<int> >("curve");
  int curvenum = -1;
  if (curve) curvenum = (*curve)[0];

  // initialization of time-curve factor
  double curvefac = 0.0;

  // compute potential time curve or set time-curve factor to one
  if (curvenum >= 0)
  {
    // time factor (negative time indicating error)
    if (time >= 0.0)
      curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);
    else dserror("Negative time in function evaluation: time = %f", time);
  }
  else curvefac = 1.0;

  // get values and switches from the condition
  const std::vector<int>*    onoff     = cond->Get<std::vector<int>    >("onoff");
  const std::vector<double>* val       = cond->Get<std::vector<double> >("val"  );
  const std::vector<int>*    functions = cond->Get<std::vector<int>    >("funct");

  // factor given by spatial function
  double functionfac = 0.0;
  int functnum = -1;

  // set this condition
  for(int dof=0;dof<numdof;++dof)
  {
    // get factor given by spatial function
    if (functions) functnum = (*functions)[dof];
    else functnum = -1;

    double num = (*onoff)[dof]*(*val)[dof]*curvefac;

    if (functnum>0)
      functionfac = DRT::Problem::Instance()->Funct(functnum-1).Evaluate(dof%numdof,x,time,NULL);
    else
      functionfac = 1.0;

    final_values[dof] = num*functionfac;
  } // loop dofs
}


void XFEM::MeshCouplingBC::ComputeInterfaceVelocityFromDisplacement(
    std::vector<double>& final_values,
    DRT::Node* node,
    const double dt,
    const std::string* evaltype
)
{
  if(dt < 1e-14) dserror("zero or negative time step size not allowed!!!");


  double thetaiface = 0.0;

  if( *evaltype == "displacement_1storder_wo_initfunct" or
      *evaltype == "displacement_1storder_with_initfunct")
    thetaiface = 1.0; // for backward Euler, OST(1.0)
  else if( *evaltype == "displacement_2ndorder_wo_initfunct" or
           *evaltype == "displacement_2ndorder_with_initfunct")
    thetaiface = 0.5; // for Crank Nicholson, OST(0.5)
  else dserror("not supported");


  const std::vector<int> nodedofset = cutter_dis_->Dof(node);
  const int numdof = nodedofset.size();

  // loop dofs of node
  for(int dof=0;dof<numdof;++dof)
  {

    int gid = nodedofset[dof];
    int lid = idispnp_->Map().LID(gid);

    const double dispnp = (*idispnp_)[lid];
    const double dispn  = (*idispn_ )[lid];
    const double veln   = (*iveln_  )[lid];

    final_values[dof] = 1.0/(thetaiface*dt) * (dispnp-dispn) - (1.0-thetaiface)/thetaiface * veln;
  } // loop dofs
}

void XFEM::MeshCouplingBC::EvaluateImplementation(
    std::vector<double>& final_values,
    const double* x,
    DRT::Condition* cond,
    const double time,
    const std::string& function_name
)
{
  const int numdof = final_values.size();

  if(function_name != "ROTATING_BEAM") dserror("currently only the rotating beam function is available!");

  double t_1 = 1.0;         // ramp the rotation
  double t_2 = t_1+1.0;     // reached the constant angle velocity
  double t_3 = t_2+12.0;    // decrease the velocity and turn around
  double t_4 = t_3+2.0;     // constant negative angle velocity

  double arg= 0.0; // prescribe a time-dependent angle

  double T = 16.0;  // time period for 2*Pi
  double angle_vel = 2.*M_PI/T;

  if(time <= t_1)
  {
    arg = 0.0;
  }
  else if(time> t_1 and time<= t_2)
  {
    arg = angle_vel / 2.0 * (time-t_1) - angle_vel*(t_2-t_1)/(2.0*M_PI)*sin(M_PI * (time-t_1)/(t_2-t_1));
  }
  else if(time>t_2 and time<= t_3)
  {
    arg = angle_vel * (time-t_2) + M_PI/T*(t_2-t_1);
  }
  else if(time>t_3 and time<=t_4)
  {
    arg = angle_vel*(t_4-t_3)/(M_PI)*sin(M_PI*(time-t_3)/(t_4-t_3)) + 2.0*M_PI/T*(t_3-t_2)+ M_PI/T*(t_2-t_1);
  }
  else if(time>t_4)
  {
    arg = -angle_vel * (time-t_4) + M_PI/T*(t_2-t_1)+ 2.0*M_PI/T*(t_3-t_2);
  }
  else dserror("for that time we did not define an implemented rotation %d", time);


  // rotation with constant angle velocity around point
  LINALG::Matrix<3,1> center(true);

  center(0) = 0.0;
  center(1) = 0.0;
  center(2) = 0.0;

  LINALG::Matrix<3,1> diff(true);
  diff(0) = x[0]-center(0);
  diff(1) = x[1]-center(1);
  diff(2) = x[2]-center(2);

  // rotation matrix
  LINALG::Matrix<3,3> rot(true);

  rot(0,0) = cos(arg);            rot(0,1) = -sin(arg);          rot(0,2) = 0.0;
  rot(1,0) = sin(arg);            rot(1,1) = cos(arg);           rot(1,2) = 0.0;
  rot(2,0) = 0.0;                 rot(2,1) = 0.0;                rot(2,2) = 1.0;

  //          double r= diff.Norm2();
  //
  //          rot.Scale(r);

  LINALG::Matrix<3,1> x_new(true);
  LINALG::Matrix<3,1> rotated(true);

  rotated.Multiply(rot,diff);

  x_new.Update(1.0,rotated,-1.0, diff);


  for(int dof=0;dof<numdof;++dof)
  {
    final_values[dof] = x_new(dof);
  }
}

/*----------------------------------------------------------------------*
 |  set interface displacement at current time             schott 03/12 |
 *----------------------------------------------------------------------*/
void XFEM::MeshCouplingBC::SetInterfaceDisplacement()
{
  if(myrank_ == 0) IO::cout << "\t set interface displacement, time " << time_ << IO::endl;

  std::string condname = "XFEMSurfDisplacement";

  EvaluateCondition( idispnp_, condname, time_);
}

/*----------------------------------------------------------------------*
 |  set interface velocity at current time                 schott 03/12 |
 *----------------------------------------------------------------------*/
void XFEM::MeshCouplingBC::SetInterfaceVelocity()
{
  if(myrank_ == 0) IO::cout << "\t set interface velocity, time " << time_ << IO::endl;

  EvaluateCondition( ivelnp_, cond_name_, time_, dt_);
}


//! constructor
XFEM::MeshCouplingFSI::MeshCouplingFSI(
    Teuchos::RCP<DRT::Discretization>&  bg_dis,   ///< background discretization
    const std::string &                 cond_name,///< name of the condition, by which the derived cutter discretization is identified
    Teuchos::RCP<DRT::Discretization>&  cond_dis,  ///< discretization from which cutter discretization can be derived
    const double                        time,      ///< time
    const int                           step       ///< time step
) : MeshCoupling(bg_dis,cond_name,cond_dis, time, step), firstoutputofrun_(true)
{
  InitStateVectors_FSI();
  PrepareCutterOutput();
}



void XFEM::MeshCouplingFSI::InitStateVectors_FSI()
{
  const Epetra_Map* cutterdofrowmap = cutter_dis_->DofRowMap();
  const Epetra_Map* cutterdofcolmap = cutter_dis_->DofColMap();

  itrueresidual_ = LINALG::CreateVector(*cutterdofrowmap,true);
  iforcecol_     = LINALG::CreateVector(*cutterdofcolmap,true);
}


void XFEM::MeshCouplingFSI::CompleteStateVectors()
{
  //-------------------------------------------------------------------------------
  // finalize itrueresidual vector

  // need to export the interface forces
  Epetra_Vector iforce_tmp(itrueresidual_->Map(),true);
  Epetra_Export exporter_iforce(iforcecol_->Map(),iforce_tmp.Map());
  int err1 = iforce_tmp.Export(*iforcecol_,exporter_iforce,Add);
  if (err1) dserror("Export using exporter returned err=%d",err1);

  // scale the interface trueresidual with -1.0 to get the forces acting on structural side (no residual-scaling!)
  itrueresidual_->Update(-1.0,iforce_tmp,0.0);
}


void XFEM::MeshCouplingFSI::ZeroStateVectors_FSI()
{
  itrueresidual_->PutScalar(0.0);
  iforcecol_->PutScalar(0.0);
}

// -------------------------------------------------------------------
// Read Restart data for cutter discretization
// -------------------------------------------------------------------
void XFEM::MeshCouplingFSI::ReadRestart(
    const int step
)
{
  const int myrank = cutter_dis_->Comm().MyPID();

  if(myrank) IO::cout << "ReadRestart for boundary discretization " << IO::endl;

  //-------- boundary discretization
  IO::DiscretizationReader boundaryreader(cutter_dis_, step);

  const double time = boundaryreader.ReadDouble("time");
//  const int    step = boundaryreader.ReadInt("step");

  if(myrank == 0)
  {
    IO::cout << "time: " << time << IO::endl;
    IO::cout << "step: " << step << IO::endl;
  }

  boundaryreader.ReadVector(iveln_,   "iveln_res");
  boundaryreader.ReadVector(idispn_,  "idispn_res");

  // REMARK: ivelnp_ and idispnp_ are set again for the new time step in PrepareSolve()
  boundaryreader.ReadVector(ivelnp_,  "ivelnp_res");
  boundaryreader.ReadVector(idispnp_, "idispnp_res");

  if (not (cutter_dis_->DofRowMap())->SameAs(ivelnp_->Map()))
    dserror("Global dof numbering in maps does not match");
  if (not (cutter_dis_->DofRowMap())->SameAs(iveln_->Map()))
    dserror("Global dof numbering in maps does not match");
  if (not (cutter_dis_->DofRowMap())->SameAs(idispnp_->Map()))
    dserror("Global dof numbering in maps does not match");
  if (not (cutter_dis_->DofRowMap())->SameAs(idispn_->Map()))
    dserror("Global dof numbering in maps does not match");


}


void XFEM::MeshCouplingFSI::GmshOutput(
    const std::string & filename_base,
    const int step,
    const int gmsh_step_diff,
    const bool gmsh_debug_out_screen
)
{
  std::ostringstream filename_base_fsi;
  filename_base_fsi << filename_base << "_force";

  // compute the current boundary position
  std::map<int,LINALG::Matrix<3,1> > currinterfacepositions;
  XFEM::UTILS::ExtractNodeVectors(cutter_dis_, currinterfacepositions);


  const std::string filename =
      IO::GMSH::GetNewFileNameAndDeleteOldFiles(
          filename_base_fsi.str(),
          step,
          gmsh_step_diff,
          gmsh_debug_out_screen,
          cutter_dis_->Comm().MyPID()
      );

  std::ofstream gmshfilecontent(filename.c_str());

  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" " << "iforce \" {" << std::endl;
    // draw vector field 'force' for every node
    IO::GMSH::SurfaceVectorFieldDofBasedToGmsh(cutter_dis_,itrueresidual_,currinterfacepositions,gmshfilecontent,3,3);
    gmshfilecontent << "};" << std::endl;
  }

  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" " << "idispnp \" {" << std::endl;
    // draw vector field 'idispnp' for every node
    IO::GMSH::SurfaceVectorFieldDofBasedToGmsh(cutter_dis_,idispnp_,currinterfacepositions,gmshfilecontent,3,3);
    gmshfilecontent << "};" << std::endl;
  }

  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" " << "ivelnp \" {" << std::endl;
    // draw vector field 'ivelnp' for every node
    IO::GMSH::SurfaceVectorFieldDofBasedToGmsh(cutter_dis_,ivelnp_,currinterfacepositions,gmshfilecontent,3,3);
    gmshfilecontent << "};" << std::endl;
  }

  gmshfilecontent.close();
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void XFEM::MeshCouplingFSI::GmshOutputDiscretization(
  std::ostream& gmshfilecontent
)
{
  // print surface discretization
  XFEM::MeshCoupling::GmshOutputDiscretization(gmshfilecontent);

  // compute the current solid and boundary position
  std::map<int,LINALG::Matrix<3,1> > currsolidpositions;

  XFEM::UTILS::ExtractNodeVectors(cond_dis_, currsolidpositions);

  XFEM::UTILS::PrintDiscretizationToStream(cond_dis_,
      cond_dis_->Name(), true, false, true, false, false, false, gmshfilecontent, &currsolidpositions);
}

void XFEM::MeshCouplingFSI::PrepareCutterOutput()
{
  // -------------------------------------------------------------------
  // prepare output
  // -------------------------------------------------------------------

  cutter_dis_->SetWriter(Teuchos::rcp(new IO::DiscretizationWriter(cutter_dis_)));
  cutter_output_ = cutter_dis_->Writer();
  cutter_output_->WriteMesh(0,0.0);
}

void XFEM::MeshCouplingFSI::Output(
    const int step,
    const double time,
    const bool write_restart_data
)
{
  // output for interface
  cutter_output_->NewStep(step,time);

  cutter_output_->WriteVector("ivelnp", ivelnp_);
  cutter_output_->WriteVector("idispnp", idispnp_);
  cutter_output_->WriteVector("itrueresnp", itrueresidual_);

  cutter_output_->WriteElementData(firstoutputofrun_);
  firstoutputofrun_ = false;

  // write restart
  if (write_restart_data)
  {
    cutter_output_->WriteVector("iveln_res",   iveln_);
    cutter_output_->WriteVector("idispn_res",  idispn_);
    cutter_output_->WriteVector("ivelnp_res",  ivelnp_);
    cutter_output_->WriteVector("idispnp_res", idispnp_);
  }
}

//----------------------------------------------------------------------
// LiftDrag                                                  chfoe 11/07
//----------------------------------------------------------------------
//calculate lift&drag forces
//
//Lift and drag forces are based upon the right hand side true-residual entities
//of the corresponding nodes. The contribution of the end node of a line is entirely
//added to a present L&D force.
/*----------------------------------------------------------------------*/
void XFEM::MeshCouplingFSI::LiftDrag(
    const int step,
    const double time
) const
{
  // get forces on all procs
  // create interface DOF vectors using the fluid parallel distribution
  Teuchos::RCP<const Epetra_Vector> iforcecol = DRT::UTILS::GetColVersionOfRowVector(cutter_dis_, itrueresidual_);

  if (cutter_dis_->Comm().MyPID() == 0)
  {
    // compute force components
    const int nsd = 3;
    const Epetra_Map* dofcolmap = cutter_dis_->DofColMap();
    LINALG::Matrix<3,1> c(true);
    for (int inode = 0; inode < cutter_dis_->NumMyColNodes(); ++inode)
    {
      const DRT::Node* node = cutter_dis_->lColNode(inode);
      const std::vector<int> dof = cutter_dis_->Dof(node);
      for (int isd = 0; isd < nsd; ++isd)
      {
        // [// minus to get correct sign of lift and drag (force acting on the body) ]
        c(isd) += (*iforcecol)[dofcolmap->LID(dof[isd])];
      }
    }

    // print to file
    std::ostringstream s;
    std::ostringstream header;

    header << std::left  << std::setw(10) << "Time"
        << std::right << std::setw(16) << "F_x"
        << std::right << std::setw(16) << "F_y"
        << std::right << std::setw(16) << "F_z";
    s << std::left  << std::setw(10) << std::scientific << time
        << std::right << std::setw(16) << std::scientific << c(0)
        << std::right << std::setw(16) << std::scientific << c(1)
        << std::right << std::setw(16) << std::scientific << c(2);

    std::ofstream f;
    const std::string fname = DRT::Problem::Instance()->OutputControlFile()->FileName()
                                        + ".liftdrag."
                                        + cond_name_
                                        + ".txt";
    if (step <= 1)
    {
      f.open(fname.c_str(),std::fstream::trunc);
      f << header.str() << std::endl;
    }
    else
    {
      f.open(fname.c_str(),std::fstream::ate | std::fstream::app);
    }
    f << s.str() << "\n";
    f.close();

    std::cout << header.str() << std::endl << s.str() << std::endl;
  }
}



//! constructor
XFEM::MeshCouplingFSICrack::MeshCouplingFSICrack(
    Teuchos::RCP<DRT::Discretization>&  bg_dis,   ///< background discretization
    const std::string &                 cond_name,///< name of the condition, by which the derived cutter discretization is identified
    Teuchos::RCP<DRT::Discretization>&  cond_dis,  ///< discretization from which cutter discretization can be derived
    const double                        time,      ///< time
    const int                           step       ///< time step
) : MeshCouplingFSI(bg_dis,cond_name,cond_dis,time,step)
{
  InitCrackInitiationsPoints();

  {
    // @ Sudhakar
    // keep a pointer to the original boundary discretization
    // note: for crack problems, the discretization is replaced by new ones during the simulation.
    // Paraview output based on changing discretizations is not possible so far.
    // To enable at least restarts, the IO::DiscretizationWriter(boundarydis_) has to be kept alive,
    // however, in case that the initial boundary dis, used for creating the Writer, is replaced, it will be deleted,
    // as now other RCP points to it anymore. Then the functionality of the Writer breaks down. Therefore, we artificially
    // hold second pointer to the original boundary dis for Crack-problems.
    cutterdis_init_output_ = cutter_dis_;
  }
}


void XFEM::MeshCouplingFSICrack::SetCutterDis(Teuchos::RCP<DRT::Discretization> cutter_dis_new)
{
  cutter_dis_ = cutter_dis_new;

  // update the Coupling object

  // set unique element conditions
  SetElementConditions();

  // set the coupling strategy
  SetCouplingStrategy();

  // set coupling discretization
  SetCouplingDiscretization();

  // NOTE: do not create new state vectors, this is done in UpdateBoundaryValuesAfterCrack
  // NOTE: do not create new specific state vectors, this is done in UpdateBoundaryValuesAfterCrack

  // create new iforcecol vector as it is not updated in UpdateBoundaryValuesAfterCrack
  iforcecol_ = LINALG::CreateVector(*cutter_dis_->DofColMap(),true);
}


void XFEM::MeshCouplingFSICrack::InitCrackInitiationsPoints()
{
  tip_nodes_.clear();

  DRT::Condition* crackpts = cond_dis_->GetCondition( "CrackInitiationPoints" );

  const std::vector<int>* crackpt_nodes = const_cast<std::vector<int>* >(crackpts->Nodes());


  for(std::vector<int>::const_iterator inod=crackpt_nodes->begin(); inod!=crackpt_nodes->end();inod++ )
  {
    const int nodid = *inod;
    LINALG::Matrix<3, 1> xnod( true );

    tip_nodes_[nodid] = xnod;
  }

  if( tip_nodes_.size() == 0 )
    dserror("crack initiation points unspecified\n");

/*---------------------- POSSIBILITY 2 --- adding crack tip elements ----------------------------*/
/*
{
DRT::Condition* crackpts = soliddis_->GetCondition( "CrackInitiationPoints" );

const std::vector<int>* tipnodes = const_cast<std::vector<int>* >(crackpts->Nodes());

if( tipnodes->size() == 0 )
  dserror("crack initiation points unspecified\n");

addCrackTipElements( tipnodes );
}*/

/*  Teuchos::RCP<DRT::DofSet> newdofset1 = Teuchos::rcp(new DRT::TransparentIndependentDofSet(soliddis_,true));

boundarydis_->ReplaceDofSet(newdofset1);//do not call this with true!!
boundarydis_->FillComplete();*/
}


void XFEM::MeshCouplingFSICrack::UpdateBoundaryValuesAfterCrack(
    const std::map<int,int>& oldnewIds
)
{
  // NOTE: these routines create new vectors, transfer data from the original to the new one and set the pointers to
  // the newly created vectors

  DRT::CRACK::UTILS::UpdateThisEpetraVectorCrack( cutter_dis_, ivelnp_, oldnewIds );
  DRT::CRACK::UTILS::UpdateThisEpetraVectorCrack( cutter_dis_, iveln_,  oldnewIds );
  DRT::CRACK::UTILS::UpdateThisEpetraVectorCrack( cutter_dis_, ivelnm_, oldnewIds );

  DRT::CRACK::UTILS::UpdateThisEpetraVectorCrack( cutter_dis_, idispnp_, oldnewIds );
  DRT::CRACK::UTILS::UpdateThisEpetraVectorCrack( cutter_dis_, idispn_,  oldnewIds );

  // update necessary for partitioned FSI, where structure is solved first and
  // crack values have been updated at the end of the last time step,
  // Then interface forces have to be transfered to the new vector based on the new boundary discretization
  DRT::CRACK::UTILS::UpdateThisEpetraVectorCrack( cutter_dis_, itrueresidual_, oldnewIds );


  //TODO: I guess the following lines are unnecessary (Sudhakar)
  {
    //iforcenp_ = LINALG::CreateVector(*boundarydis_->DofRowMap(),true);
    //LINALG::Export( *itrueresidual_, *iforcenp_ );

  }

  //TODO: Check whether the output in case of crack-FSI work properly (Sudhakar)
  //boundarydis_->SetWriter(Teuchos::rcp(new IO::DiscretizationWriter(boundarydis_)));
  //boundary_output_ = boundarydis_->Writer();
}



XFEM::LevelSetCoupling::LevelSetCoupling(
    Teuchos::RCP<DRT::Discretization>&  bg_dis,   ///< background discretization
    const std::string &                 cond_name, ///< name of the condition, by which the derived cutter discretization is identified
    const double                        time,      ///< time
    const int                           step       ///< time step
) : CouplingBase(bg_dis, cond_name, bg_dis,time,step)
{
  /// level-set field is given w.r.t background mesh
  /// NOTE: more generally it would be possible cutterdis != bg_dis for the single LevelSetCoupling,
  /// however, the unique bg_phinp vector stored in the ConditionManager has to be given w.r.t bgdis
  cutter_dis_ = bg_dis;

  SetConditionsToCopy();

  SetElementConditions();

  // set the coupling strategy
  SetCouplingStrategy();

  // set coupling discretization
  SetCouplingDiscretization();

  // create node-based vector with level-set values
  phinp_ = Teuchos::rcp(new Epetra_Vector(*cutter_dis_->NodeRowMap()));

  // read initial level-set field
  SetLevelSetField(time_);
}


void XFEM::LevelSetCoupling::SetConditionsToCopy()
{
  // set only the unique given condition name
  conditions_to_copy_.push_back(cond_name_);

  // additional conditions required for the levelset field based on the cutter (background) mesh
  conditions_to_copy_.push_back("XFEMSurfDisplacement");
}


/*----------------------------------------------------------------------*
 | ... |
 *----------------------------------------------------------------------*/
void XFEM::LevelSetCoupling::SetLevelSetField(
   Teuchos::RCP<const Epetra_Vector> scalaraf,
   Teuchos::RCP<DRT::Discretization> scatradis
   )
{
  // initializations
  int err(0);
  double value(0.0);
  std::vector<int> nodedofs;


  // loop all nodes on the processor
  for(int lnodeid=0;lnodeid<cutter_dis_->NumMyRowNodes();lnodeid++)
  {
    // get the processor's local scatra node
    DRT::Node* lscatranode = scatradis->lRowNode(lnodeid);

    // find out the global dof id of the last(!) dof at the scatra node
    const int numscatradof = scatradis->NumDof(0,lscatranode);
    const int globalscatradofid = scatradis->Dof(0,lscatranode,numscatradof-1);
    const int localscatradofid = scalaraf->Map().LID(globalscatradofid);
    if (localscatradofid < 0)
      dserror("localdofid not found in map for given globaldofid");

    // now copy the values
    value = (*scalaraf)[localscatradofid];
    err = phinp_->ReplaceMyValue(lnodeid,0,value);
    if (err != 0) dserror("error while inserting value into phinp_");

  }

  return;
}


/*----------------------------------------------------------------------*
 | ... |
 *----------------------------------------------------------------------*/
void XFEM::LevelSetCoupling::SetLevelSetField(const double time)
{
  // initializations
  int err(0);
  double value(0.0);
  std::vector<int> nodedofs;

  // get the function from the first element
  const int lid=0;
  DRT::Condition* cond = cutterele_conds_[lid].second;
  const int func_no = cond->GetInt("levelsetfieldno");

  // check for potential time curve
  const int curvenum  = cond->GetInt("levelsetcurve");

  // initialization of time-curve factor
  double curvefac = 0.0;

  // compute potential time curve or set time-curve factor to one
  if (curvenum >= 0)
  {
    // time factor (negative time indicating error)
    if (time >= 0.0)
      curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);
    else dserror("Negative time in function evaluation: time = %f", time);
  }
  else curvefac = 1.0;

  // loop all nodes on the processor
  for(int lnodeid=0;lnodeid<cutter_dis_->NumMyRowNodes();lnodeid++)
  {
    // get the processor's local scatra node
    DRT::Node* lnode = cutter_dis_->lRowNode(lnodeid);

    // get value
    value=DRT::Problem::Instance()->Funct(func_no-1).Evaluate(0,lnode->X(),time,NULL);

    double final_val = curvefac*value;

    // now copy the values
    err = phinp_->ReplaceMyValue(lnodeid,0,final_val);
    if (err != 0) dserror("error while inserting value into phinp_");
  }

  return;
}


// -------------------------------------------------------------------
// Read Restart data for cutter discretization
// -------------------------------------------------------------------
void XFEM::LevelSetCoupling::ReadRestart(
    const int step
)
{
  dserror("read restart for Level-set coupling objects not implemented yet");
}


//constructor
XFEM::ConditionManager::ConditionManager(
      Teuchos::RCP<DRT::Discretization> &               bg_dis,          ///< background discretization
      std::vector<Teuchos::RCP<DRT::Discretization> > & meshcoupl_dis,   ///< mesh coupling discretizations
      const double                                      time,            ///< time
      const int                                         step             ///< time step
  ) :
  bg_dis_(bg_dis),
  levelset_gid_(-1),
  time_(time),
  step_(step),
  is_levelset_uptodate_(false),
  bg_phinp_(Teuchos::null)
{

  // create Levelset Coupling objects
  {
    std::vector<std::string> conditions_to_check;

    conditions_to_check.push_back("XFEMLevelsetWeakDirichlet");
    conditions_to_check.push_back("XFEMLevelsetNeumann");
    conditions_to_check.push_back("XFEMLevelsetTwophase");
    conditions_to_check.push_back("XFEMLevelsetCombustion");

    std::vector<std::string> names;
    bg_dis_->GetConditionNames( names );

    // check if background discretization has relevant conditioned nodes
    // create new coupling object for each type of condition
    for(size_t c=0; c<conditions_to_check.size(); c++)
    {
      if(std::find(names.begin(), names.end(), conditions_to_check[c]) != names.end())
        CreateNewLevelSetCoupling(conditions_to_check[c]);

    }
  }

  // create Mesh Coupling objects
  {
    std::vector<std::string> conditions_to_check;
    conditions_to_check.push_back("XFEMSurfFSIPart");
    conditions_to_check.push_back("XFEMSurfFSIMono");
    conditions_to_check.push_back("XFEMSurfCrackFSIPart");
    conditions_to_check.push_back("XFEMSurfFluidFluid");
    conditions_to_check.push_back("XFEMSurfWeakDirichlet");
    conditions_to_check.push_back("XFEMSurfNeumann");


    // check if a coupling discretization has relevant conditioned nodes
    // create new coupling object for each type of condition and each coupling discretization
    for(size_t mc_idx=0; mc_idx<meshcoupl_dis.size(); mc_idx++) // loop all specified mesh coupling discretizations
    {
      if(meshcoupl_dis[mc_idx] == Teuchos::null) continue;

      std::vector<std::string> names;
      meshcoupl_dis[mc_idx]->GetConditionNames( names );

      for(size_t c=0; c<conditions_to_check.size(); c++)
      {
        if(std::find(names.begin(), names.end(), conditions_to_check[c]) != names.end())
          CreateNewMeshCoupling(conditions_to_check[c], meshcoupl_dis[mc_idx]);
      }
    }
  }

}


void XFEM::ConditionManager::IncrementTimeAndStep(
    const double dt)
{
  step_ += 1;
  time_ += dt;

  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->IncrementTimeAndStep(dt);
  }

  // loop all levelset coupling objects
  for(int lsc=0; lsc<(int)levelset_coupl_.size(); lsc++)
  {
    levelset_coupl_[lsc]->IncrementTimeAndStep(dt);
  }

}


void XFEM::ConditionManager::SetTimeAndStep(
    const double time,
    const int step)
{
  time_ = time;
  step_ = step;

  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->SetTimeAndStep(time, step);
  }

  // loop all levelset coupling objects
  for(int lsc=0; lsc<(int)levelset_coupl_.size(); lsc++)
  {
    levelset_coupl_[lsc]->SetTimeAndStep(time, step);
  }
}



void XFEM::ConditionManager::CreateNewLevelSetCoupling(const std::string& cond_name)
{
  AddLevelSetCoupling( cond_name );
}


void XFEM::ConditionManager::CreateNewMeshCoupling(
    const std::string& cond_name,
    Teuchos::RCP<DRT::Discretization> cond_dis         ///< discretization from which the cutter discretization can be derived
)
{
  AddMeshCoupling( cond_name, cond_dis );
}


void XFEM::ConditionManager::Create()
{
  numglobal_coupling_sides = 0;
  mesh_coupl_start_gid_.reserve(mesh_coupl_.size());
  levelset_gid_ = -1;

  // set global side Ids for all Mesh coupling discretizations and level-set sides

  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    Teuchos::RCP<DRT::Discretization> mc_cutdis = mesh_coupl_[mc]->GetCutterDis();
    if(mc_cutdis == Teuchos::null) dserror("cutter dis is Teuchos::null");

    // set current number of global coupling sides as start index for global id this coupling object
    mesh_coupl_start_gid_[mc] = numglobal_coupling_sides;

    if (mc_cutdis->Comm().MyPID() == 0)
      std::cout << "mesh coupling object " << mc << " starts with global side index " << mesh_coupl_start_gid_[mc] << std::endl;

    // increase total number of sides with number of global side elements of this mesh coupling object
    numglobal_coupling_sides += mc_cutdis->NumGlobalElements();
  }

  // TODO: unify the level-set coupling objects to one unique level-set field
  // combine the level-set values

  if(levelset_coupl_.size() > 0)
  {

    // add one global levelset side used in the cut library
    levelset_gid_ = numglobal_coupling_sides;
    numglobal_coupling_sides+=1;

    std::cout << "levelset coupling object " << " has global side id index " << levelset_gid_ << std::endl;

    bg_phinp_= LINALG::CreateVector(*bg_dis_->NodeRowMap(), true);

    //TODO: note: information about the coupling condition for level-sets is obtained via the background element
    // for which we store the index of the level-set coupling object
    // we allow for multiple level-set coupling objects however only for one level-set side
  }



  //TODO: Status-routine to print to screen

}


void XFEM::ConditionManager::SetLevelSetField( const double time )
{
  if(levelset_coupl_.size() != 1)
    dserror("level-set field is not unique, which level-set field to be set?");

  is_levelset_uptodate_ = false;

  // update the unique level-set field
  levelset_coupl_[0]->SetLevelSetField(time);
}


void XFEM::ConditionManager::SetLevelSetField(
   Teuchos::RCP<const Epetra_Vector> scalaraf,
   Teuchos::RCP<DRT::Discretization> scatradis
   )
{
  if(levelset_coupl_.size() != 1)
    dserror("level-set field is not unique, which level-set field to be set by given vector?");

  is_levelset_uptodate_ = false;

  // update the unique level-set field
  levelset_coupl_[0]->SetLevelSetField(scalaraf, scatradis);
}


Teuchos::RCP<const Epetra_Vector> XFEM::ConditionManager::GetLevelSetFieldCol()
{
  if(levelset_coupl_.size()==0) return Teuchos::null;

  // export nodal level-set values to node column map
  Teuchos::RCP<Epetra_Vector> bg_phinp_col = Teuchos::rcp(new Epetra_Vector(*bg_dis_->NodeColMap()));
  LINALG::Export(*GetLevelSetField(),*bg_phinp_col);

  return bg_phinp_col;
}

void XFEM::ConditionManager::UpdateLevelSetField()
{

  if(levelset_coupl_.size() != 1)
      dserror("level-set field is not unique, update of the global bg_phinp by more than one phinp not implemented yet");

  // assume same maps between background fluid dis and the cutterdis (scatra dis)
  bg_phinp_->Update(1.0, *(levelset_coupl_[0]->GetLevelSetField()), 0.0);

  is_levelset_uptodate_ = true;
}

void XFEM::ConditionManager::SetState()
{
  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->SetState();
  }
}

void XFEM::ConditionManager::SetStateDisplacement()
{
  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->SetStateDisplacement();
  }
}

void XFEM::ConditionManager::UpdateStateVectors()
{
  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->UpdateStateVectors();
  }
}

void XFEM::ConditionManager::CompleteStateVectors()
{
  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->CompleteStateVectors();
  }
}

void XFEM::ConditionManager::ZeroStateVectors_FSI()
{
  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->ZeroStateVectors_FSI();
  }
}

void XFEM::ConditionManager::GmshOutput(
    const std::string & filename_base,
    const int step,
    const int gmsh_step_diff,
    const bool gmsh_debug_out_screen
)
{
  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->GmshOutput(filename_base, step, gmsh_step_diff, gmsh_debug_out_screen);
  }
}

void XFEM::ConditionManager::GmshOutputDiscretization(
  std::ostream& gmshfilecontent
)
{
  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->GmshOutputDiscretization(gmshfilecontent);
  }
}

void XFEM::ConditionManager::Output(
    const int step,
    const double time,
    const bool write_restart_data
)
{
  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->Output(step, time, write_restart_data);
  }
}

void XFEM::ConditionManager::LiftDrag(
    const int step,
    const double time
)
{
  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->LiftDrag(step, time);
  }
}

void XFEM::ConditionManager::ReadRestart(
    const int step
)
{
  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->ReadRestart(step);
  }

  // loop all levelset coupling objects
  for(int lsc=0; lsc<(int)levelset_coupl_.size(); lsc++)
  {
    levelset_coupl_[lsc]->ReadRestart(step);
  }
}


void XFEM::ConditionManager::PrepareSolve()
{
  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    mesh_coupl_[mc]->PrepareSolve();
  }

  // loop all levelset coupling objects
  for(int lsc=0; lsc<(int)levelset_coupl_.size(); lsc++)
  {
    levelset_coupl_[lsc]->PrepareSolve();
  }
}

bool XFEM::ConditionManager::HasMovingInterface()
{
  // loop all mesh coupling objects
  for(int mc=0; mc<(int)mesh_coupl_.size(); mc++)
  {
    if(mesh_coupl_[mc]->HasMovingInterface()) return true;
  }

  // loop all levelset coupling objects
  for(int lsc=0; lsc<(int)levelset_coupl_.size(); lsc++)
  {
    if(levelset_coupl_[lsc]->HasMovingInterface()) return true;
  }

  return false;
}


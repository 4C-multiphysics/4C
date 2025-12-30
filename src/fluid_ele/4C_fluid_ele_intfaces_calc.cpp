// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_intfaces_calc.hpp"

#include "4C_fem_discretization_faces.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_calc_intfaces_stab.hpp"
#include "4C_fluid_ele_parameter_intface.hpp"
#include "4C_fluid_ele_parameter_std.hpp"
#include "4C_fluid_ele_parameter_timint.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::FluidIntFaceImplInterface* Discret::Elements::FluidIntFaceImplInterface::impl(
    const Core::Elements::Element* ele)
{
  switch (ele->shape())
  {
    case Core::FE::CellType::quad4:
    {
      return FluidIntFaceImpl<Core::FE::CellType::quad4>::instance();
    }
    case Core::FE::CellType::quad8:
    {
      return FluidIntFaceImpl<Core::FE::CellType::quad8>::instance();
    }
    case Core::FE::CellType::quad9:
    {
      return FluidIntFaceImpl<Core::FE::CellType::quad9>::instance();
    }
    case Core::FE::CellType::tri3:
    {
      return FluidIntFaceImpl<Core::FE::CellType::tri3>::instance();
    }
    case Core::FE::CellType::tri6:
    {
      return FluidIntFaceImpl<Core::FE::CellType::tri6>::instance();
    }
    case Core::FE::CellType::line2:
    {
      return FluidIntFaceImpl<Core::FE::CellType::line2>::instance();
    }
    case Core::FE::CellType::line3:
    {
      return FluidIntFaceImpl<Core::FE::CellType::line3>::instance();
    }
    default:
      FOUR_C_THROW(
          "Element shape {} ({} nodes) not activated. Just do it.", ele->shape(), ele->num_node());
      break;
  }
  return nullptr;
}

template <Core::FE::CellType distype>
Discret::Elements::FluidIntFaceImpl<distype>*
Discret::Elements::FluidIntFaceImpl<distype>::instance(Core::Utils::SingletonAction action)
{
  static auto singleton_owner = Core::Utils::make_singleton_owner(
      []()
      {
        return std::unique_ptr<Discret::Elements::FluidIntFaceImpl<distype>>(
            new Discret::Elements::FluidIntFaceImpl<distype>());
      });

  return singleton_owner.instance(action);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::FluidIntFaceImpl<distype>::FluidIntFaceImpl()
{
  // pointer to class FluidImplParameterTimInt (access to the time-integration parameter)
  fldparatimint_ = Discret::Elements::FluidEleParameterTimInt::instance();
  // pointer to class FluidEleParameterIntFace (access to the faces specific parameter)
  fldpara_intface_ = Discret::Elements::FluidEleParameterIntFace::instance();

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidIntFaceImpl<distype>::assemble_internal_faces_using_neighbor_data(
    Discret::Elements::FluidIntFace* intface,        ///< internal face element
    std::shared_ptr<Core::Mat::Material>& material,  ///< material for face stabilization
    std::vector<int>& nds_master,                    ///< nodal dofset w.r.t. master element
    std::vector<int>& nds_slave,                     ///< nodal dofset w.r.t. slave element
    const Inpar::XFEM::FaceType& face_type,  ///< which type of face std, ghost, ghost-penalty
    Teuchos::ParameterList& params,          ///< parameter list
    Core::FE::DiscretizationFaces& discretization,              ///< faces discretization
    std::shared_ptr<Core::LinAlg::SparseMatrix> systemmatrix,   ///< systemmatrix
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector  ///< systemvector
)
{
  TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: assemble_internal_faces_using_neighbor_data");

  // decide which terms have to be assembled for the current face and decide the assembly pattern,
  // return if no assembly required
  bool stab_required = fldpara_intface_->set_face_specific_fluid_xfem_parameter(face_type, params);

  // do not assemble if no stabilization terms activated for this face
  if (!stab_required) return;

  if (!discretization.filled()) FOUR_C_THROW("fill_complete() was not called");
  if (!discretization.have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  const bool assemblemat = systemmatrix != nullptr;
  const bool assemblevec = systemvector != nullptr;

  //--------------------------------------------------------
  /// number of space dimensions of the FluidIntFace element
  static const int facensd = Core::FE::dim<distype>;

  /// number of space dimensions of the parent element
  static const int nsd = facensd + 1;

  /// number of dof's per node
  static const int numdofpernode = nsd + 1;


  //---------------------- check for PBCS ------------------
  std::shared_ptr<const std::map<int, int>> pbcconnectivity =
      discretization.get_pbc_slave_to_master_node_connectivity();

  //----------------------- create patchlm -----------------

  const int numnode_master = intface->parent_master_element()->num_node();
  const int numnode_slave = intface->parent_slave_element()->num_node();
  const int numnode_face = intface->num_node();

  const int numnodeinpatch = numnode_master + numnode_slave - numnode_face;

  // local maps for patch dofs
  std::vector<int> lm_patch;
  lm_patch.reserve(numnodeinpatch * numdofpernode);

  // local maps between master/slave dofs and position in patch dofs (lm_patch)
  std::vector<int> lm_masterToPatch;
  lm_masterToPatch.reserve(numnode_master * numdofpernode);
  std::vector<int> lm_slaveToPatch;
  lm_slaveToPatch.reserve(numnode_slave * numdofpernode);
  std::vector<int> lm_faceToPatch;
  lm_faceToPatch.reserve(numnode_face * numdofpernode);

  // local maps between master/slave nodes and position in patch nodes
  std::vector<int> lm_masterNodeToPatch;
  lm_masterNodeToPatch.reserve(numnode_master);
  std::vector<int> lm_slaveNodeToPatch;
  lm_slaveNodeToPatch.reserve(numnode_slave);


  // create patch location vector combining master element, slave element and face element
  intface->patch_location_vector(discretization, nds_master, nds_slave, lm_patch,
      // lm_master, lm_slave, lm_face,
      lm_masterToPatch, lm_slaveToPatch, lm_faceToPatch, lm_masterNodeToPatch, lm_slaveNodeToPatch,
      pbcconnectivity);


  // patch_lm for Velx,Vely,Velz, Pres field
  std::vector<std::vector<int>> patch_components_lm(numdofpernode);
  std::vector<std::vector<int>> patch_components_lmowner(numdofpernode);


  for (int i = 0; i < numdofpernode; i++)
  {
    patch_components_lm[i].reserve(numnodeinpatch);
    patch_components_lmowner[i].reserve(numnodeinpatch);
  }

  // modify the patch owner to the owner of the internal face element
  const int owner = intface->owner();
  std::vector<int> patchlm_owner(lm_patch.size(), owner);

  for (unsigned i = 0; i < lm_patch.size(); i++)
  {
    const int field = i % numdofpernode;

    // i%4 yields the Velx,Vely,Velz,Pres field
    patch_components_lm[field].push_back(lm_patch[i]);
    patch_components_lmowner[field].push_back(owner);
  }

#ifdef FOUR_C_ENABLE_ASSERTIONS
  for (int isd = 0; isd < numdofpernode; isd++)
    if ((int)(patch_components_lm[isd].size()) != numnodeinpatch)
      FOUR_C_THROW("patch_components_lm[{}] has wrong size: size is {} but expected {}", isd,
          (int)(patch_components_lm[isd].size()), numnodeinpatch);
#endif

  //------------- create and evaluate block element matrices -----------------

  //------------------------------------------------------------------------------------
  // decide which pattern
  // pattern = "u-v-w-p-diagonal-block matrix pattern";    // assembles each component block
  // separated pattern = "u-p-block matrix pattern";                 // assembles u-block and
  // p-block separated pattern = "full matrix pattern";                      // assembles the whole
  // u-p matrix

  Inpar::FLUID::EosGpPattern eos_gp_pattern = fldpara_intface_->face_eos_gp_pattern();


  int numblocks = 0;

  if (eos_gp_pattern == Inpar::FLUID::EOS_GP_Pattern_uvwp)
  {
    // 3D: 4 blocks =  u-u block, v-v block, w-w block and p-p block
    // 2D: 3 blocks =  u-u block, v-v block and p-p block
    numblocks = numdofpernode;
  }
  else if (eos_gp_pattern == Inpar::FLUID::EOS_GP_Pattern_up)
  {
    // 3D: 10 blocks = 3x3 u-u blocks + 1x1 p-p block
    // 3D: 5 blocks  = 2x2 u-u blocks + 1x1 p-p block
    numblocks = nsd * nsd + 1;  // 10 blocks = 3x3 u-u blocks + 1x1 p-p block
  }
  else if (eos_gp_pattern == Inpar::FLUID::EOS_GP_Pattern_full)
  {
    // 3D: 16 blocks = 4x4 uvwp blocks
    // 2D: 9  blocks = 3x3 uvp blocks
    numblocks = numdofpernode * numdofpernode;
  }

  else
    FOUR_C_THROW("unknown matrix pattern");


  // define element matrices and vectors
  std::vector<Core::LinAlg::SerialDenseMatrix> elemat_blocks(numblocks);
  std::vector<Core::LinAlg::SerialDenseVector> elevec_blocks(
      numdofpernode);  // 3D: 4 vectors for u,v,w,p components, 2D: 3 vectors for u,v,p


  for (int b = 0; b < numblocks; b++)
  {
    elemat_blocks[b].shape(numnodeinpatch, numnodeinpatch);  // new shape and init values to zero
  }

  for (int b = 0; b < numdofpernode; b++)
  {
    int err = elevec_blocks[b].size(numnodeinpatch);  // new size and init values to zero

    if (err != 0) FOUR_C_THROW("element matrix Shape not successful");
  }


  //---------------------------------------------------------------------
  // call the element specific evaluate method

  int err = evaluate_internal_faces(intface, material, params, discretization, lm_patch,
      lm_masterToPatch, lm_slaveToPatch, lm_faceToPatch, lm_masterNodeToPatch, lm_slaveNodeToPatch,
      elemat_blocks, elevec_blocks);
  if (err) FOUR_C_THROW("error while evaluating elements");

  //---------------------------------------------------------------------------------------
  // assemble systemmatrix
  {
    TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: assemble FE matrix");

    // calls the Assemble function for matrices including communication of non-row entries
    if (assemblemat)
    {
      if (eos_gp_pattern == Inpar::FLUID::EOS_GP_Pattern_uvwp)
      {
        for (int ij = 0; ij < numdofpernode; ij++)
        {
          systemmatrix->fe_assemble(elemat_blocks[ij], patch_components_lm[ij],
              patch_components_lmowner[ij], patch_components_lm[ij]);
        }
      }
      else if (eos_gp_pattern == Inpar::FLUID::EOS_GP_Pattern_up)
      {
        for (int i = 0; i < nsd; i++)
        {
          for (int j = 0; j < nsd; j++)
          {
            systemmatrix->fe_assemble(elemat_blocks[i * nsd + j], patch_components_lm[i],
                patch_components_lmowner[i], patch_components_lm[j]);
          }
        }
        systemmatrix->fe_assemble(elemat_blocks[nsd * nsd], patch_components_lm[nsd],
            patch_components_lmowner[nsd], patch_components_lm[nsd]);
      }
      else if (eos_gp_pattern == Inpar::FLUID::EOS_GP_Pattern_full)
      {
        for (int i = 0; i < numdofpernode; i++)
        {
          for (int j = 0; j < numdofpernode; j++)
          {
            systemmatrix->fe_assemble(elemat_blocks[i * numdofpernode + j], patch_components_lm[i],
                patch_components_lmowner[i], patch_components_lm[j]);
          }
        }
      }
      else
        FOUR_C_THROW("unknown matrix pattern");
    }
  }
  //---------------------------------------------------------------------------------------
  // assemble systemvector
  if (assemblevec)
  {
    TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: assemble vector");

    // REMARK:: call Assemble without lmowner
    // to assemble the residual_col vector on only row elements also column nodes have to be
    // assembled do not exclude non-row nodes (modify the real owner to myowner) after assembly the
    // col vector it has to be exported to the row residual_ vector using the 'Add' flag to get the
    // right value for shared nodes
    for (int i = 0; i < numdofpernode; i++)
    {
      Core::LinAlg::assemble(
          *systemvector, elevec_blocks[i], patch_components_lm[i], patch_components_lmowner[i]);
    }
  }

  return;
}



/*----------------------------------------------------------------------*
 |  Evaluate internal faces (public)                        schott 01/12|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::FluidIntFaceImpl<distype>::evaluate_internal_faces(
    Discret::Elements::FluidIntFace* intface,        ///< internal face element
    std::shared_ptr<Core::Mat::Material>& material,  ///< material associated with the faces
    Teuchos::ParameterList& params,                  ///< parameter list
    Core::FE::Discretization& discretization,        ///< discretization
    std::vector<int>& patchlm,                       ///< patch local map
    std::vector<int>& lm_masterToPatch,              ///< local map between master dofs and patchlm
    std::vector<int>& lm_slaveToPatch,               ///< local map between slave dofs and patchlm
    std::vector<int>& lm_faceToPatch,                ///< local map between face dofs and patchlm
    std::vector<int>& lm_masterNodeToPatch,  ///< local map between master nodes and nodes in patch
    std::vector<int>& lm_slaveNodeToPatch,   ///< local map between slave nodes and nodes in patch
    std::vector<Core::LinAlg::SerialDenseMatrix>& elemat_blocks,  ///< element matrix blocks
    std::vector<Core::LinAlg::SerialDenseVector>& elevec_blocks   ///< element vector blocks
)
{
  FLD::IntFaceAction act = FLD::ifa_none;
  act = Teuchos::getIntegralValue<FLD::IntFaceAction>(params, "action");

  switch (act)
  {
    case FLD::EOS_and_GhostPenalty_stabilization:
    {
      return Discret::Elements::FluidIntFaceStab::impl(intface)->evaluate_edge_based_stabilization(
          intface, material, *fldparatimint_, *fldpara_intface_, params, discretization, patchlm,
          lm_masterToPatch, lm_slaveToPatch, lm_faceToPatch, lm_masterNodeToPatch,
          lm_slaveNodeToPatch, elemat_blocks, elevec_blocks);
      break;
    }
    default:
      FOUR_C_THROW("Unknown type of action for FluidIntFace");
      break;
  }  // end of switch(act)

  return 0;
}

FOUR_C_NAMESPACE_CLOSE

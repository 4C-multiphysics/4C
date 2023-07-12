/*----------------------------------------------------------------------*/
/*! \file

\brief Internal implementation of ScaTraHDG internal faces elements

Integrate internal face terms on an internal faces element

\level 3


*----------------------------------------------------------------------*/
#include <Teuchos_TimeMonitor.hpp>

#include "scatra_ele_action.H"
#include "scatra_ele_hdg_intfaces_calc.H"

#include "linalg_utils_sparse_algebra_math.H"

#include "lib_discret_faces.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::ScaTraHDGIntFaceImplInterface* DRT::ELEMENTS::ScaTraHDGIntFaceImplInterface::Impl(
    const DRT::Element* ele)
{
  switch (ele->Shape())
  {
    case DRT::Element::quad4:
    {
      return ScaTraHDGIntFaceImpl<DRT::Element::quad4>::Instance();
    }
    case DRT::Element::quad8:
    {
      return ScaTraHDGIntFaceImpl<DRT::Element::quad8>::Instance();
    }
    case DRT::Element::quad9:
    {
      return ScaTraHDGIntFaceImpl<DRT::Element::quad9>::Instance();
    }
    case DRT::Element::tri3:
    {
      return ScaTraHDGIntFaceImpl<DRT::Element::tri3>::Instance();
    }
    case DRT::Element::tri6:
    {
      return ScaTraHDGIntFaceImpl<DRT::Element::tri6>::Instance();
    }
    case DRT::Element::line2:
    {
      return ScaTraHDGIntFaceImpl<DRT::Element::line2>::Instance();
    }
    case DRT::Element::line3:
    {
      return ScaTraHDGIntFaceImpl<DRT::Element::line3>::Instance();
    }
    default:
      dserror(
          "Element shape %d (%d nodes) not activated. Just do it.", ele->Shape(), ele->NumNode());
      break;
  }
  return NULL;
}

template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::ScaTraHDGIntFaceImpl<distype>*
DRT::ELEMENTS::ScaTraHDGIntFaceImpl<distype>::Instance(CORE::UTILS::SingletonAction action)
{
  static auto singleton_owner = CORE::UTILS::MakeSingletonOwner(
      []()
      {
        return std::unique_ptr<DRT::ELEMENTS::ScaTraHDGIntFaceImpl<distype>>(
            new DRT::ELEMENTS::ScaTraHDGIntFaceImpl<distype>());
      });

  return singleton_owner.Instance(action);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::ScaTraHDGIntFaceImpl<distype>::ScaTraHDGIntFaceImpl()
{
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraHDGIntFaceImpl<distype>::AssembleInternalFacesUsingNeighborData(
    DRT::ELEMENTS::ScaTraHDGIntFace* intface,               ///< internal face element
    std::vector<int>& nds_master,                           ///< nodal dofset w.r.t. master element
    std::vector<int>& nds_slave,                            ///< nodal dofset w.r.t. slave element
    Teuchos::ParameterList& params,                         ///< parameter list
    DRT::DiscretizationFaces& discretization,               ///< faces discretization
    Teuchos::RCP<CORE::LINALG::SparseMatrix> systemmatrix,  ///< systemmatrix
    Teuchos::RCP<Epetra_Vector> systemvector                ///< systemvector
)
{
  return;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
int DRT::ELEMENTS::ScaTraHDGIntFaceImpl<distype>::EvaluateInternalFaces(
    DRT::ELEMENTS::ScaTraHDGIntFace* intface,  ///< internal face element
    Teuchos::ParameterList& params,            ///< parameter list
    DRT::Discretization& discretization,       ///< discretization
    std::vector<int>& patchlm,                 ///< patch local map
    std::vector<int>& lm_masterToPatch,        ///< local map between master dofs and patchlm
    std::vector<int>& lm_slaveToPatch,         ///< local map between slave dofs and patchlm
    std::vector<int>& lm_faceToPatch,          ///< local map between face dofs and patchlm
    std::vector<int>& lm_masterNodeToPatch,  ///< local map between master nodes and nodes in patch
    std::vector<int>& lm_slaveNodeToPatch,   ///< local map between slave nodes and nodes in patch
    std::vector<Epetra_SerialDenseMatrix>& elemat_blocks,  ///< element matrix blocks
    std::vector<Epetra_SerialDenseVector>& elevec_blocks   ///< element vector blocks
)
{
  return 0;
}

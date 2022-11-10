/*----------------------------------------------------------------------*/
/*! \file

\brief transparent independent dofset

\level 1

*/
/*----------------------------------------------------------------------*/

#include "xfem_dofset_transparent_independent.H"

#include "cut_node.H"
#include "cut_cutwizard.H"


XFEM::XFEMTransparentIndependentDofSet::XFEMTransparentIndependentDofSet(
    Teuchos::RCP<DRT::Discretization> sourcedis, bool parallel, Teuchos::RCP<GEO::CutWizard> wizard)
    : DRT::TransparentIndependentDofSet(sourcedis, parallel), wizard_(wizard)
{
  return;
}

int XFEM::XFEMTransparentIndependentDofSet::NumDofPerNode(const DRT::Node &node) const
{
  if (wizard_ != Teuchos::null)
  {
    GEO::CUT::Node *n = wizard_->GetNode(node.Id());
    if (n != NULL)
    {
      int numdofpernode = DRT::DofSet::NumDofPerNode(node);
      return numdofpernode * n->NumDofSets();
    }
  }
  return DRT::DofSet::NumDofPerNode(node);
}

/*!----------------------------------------------------------------------
\file manager_nurbs.cpp

<pre>
Maintainer: Philipp Farah
            farah@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15257
</pre>

*----------------------------------------------------------------------*/

#include "../drt_mortar/mortar_manager_base.H"
#include "../drt_mortar/mortar_node.H"
#include "../drt_mortar/mortar_element.H"

#include "../drt_nurbs_discret/drt_control_point.H"
#include "../drt_nurbs_discret/drt_nurbs_discret.H"
#include "../drt_nurbs_discret/drt_knotvector.H"

/*----------------------------------------------------------------------*
 |  Prepare mortar element for nurbs-case                    farah 11/14|
 *----------------------------------------------------------------------*/
void MORTAR::ManagerBase::PrepareNURBSElement(
    DRT::Discretization& discret,
    Teuchos::RCP<DRT::Element> ele,
    Teuchos::RCP<MORTAR::MortarElement> cele,
    int dim)
{
  DRT::NURBS::NurbsDiscretization* nurbsdis =
      dynamic_cast<DRT::NURBS::NurbsDiscretization*>(&(discret));

  Teuchos::RCP<DRT::NURBS::Knotvector> knots =
      (*nurbsdis).GetKnotVector();
  std::vector<Epetra_SerialDenseVector> parentknots(dim);
  std::vector<Epetra_SerialDenseVector> mortarknots(dim - 1);

  double normalfac = 0.0;
  bool zero_size = knots->GetBoundaryEleAndParentKnots(
      parentknots,
      mortarknots,
      normalfac,
      ele->ParentMasterElement()->Id(),
      ele->FaceMasterNumber());

  // store nurbs specific data to node
  cele->ZeroSized() = zero_size;
  cele->Knots()     = mortarknots;
  cele->NormalFac() = normalfac;

 return;
}


/*----------------------------------------------------------------------*
 |  Prepare mortar node for nurbs-case                       farah 11/14|
 *----------------------------------------------------------------------*/
void MORTAR::ManagerBase::PrepareNURBSNode(
    DRT::Node* node,
    Teuchos::RCP<MORTAR::MortarNode> mnode)
{
  DRT::NURBS::ControlPoint* cp =
      dynamic_cast<DRT::NURBS::ControlPoint*>(node);

  mnode->NurbsW() = cp->W();

  return;
}

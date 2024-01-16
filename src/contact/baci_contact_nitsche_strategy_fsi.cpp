/*---------------------------------------------------------------------*/
/*! \file

\brief Nitsche contact solving strategy for problems with FSI

\level 3


*/
/*---------------------------------------------------------------------*/

#include "baci_contact_nitsche_strategy_fsi.H"

#include "baci_contact_element.H"
#include "baci_contact_interface.H"
#include "baci_contact_nitsche_integrator_fsi.H"
#include "baci_lib_discret.H"
#include "baci_mortar_projector.H"

BACI_NAMESPACE_OPEN

void CONTACT::NitscheStrategyFsi::ApplyForceStiffCmt(Teuchos::RCP<Epetra_Vector> dis,
    Teuchos::RCP<CORE::LINALG::SparseOperator>& kt, Teuchos::RCP<Epetra_Vector>& f, const int step,
    const int iter, bool predictor)
{
  if (predictor) return;
  CONTACT::NitscheStrategy::ApplyForceStiffCmt(dis, kt, f, step, iter, predictor);
}

void CONTACT::NitscheStrategyFsi::SetState(
    const enum MORTAR::StateType& statename, const Epetra_Vector& vec)
{
  CONTACT::NitscheStrategy::SetState(statename, vec);
  if (statename == MORTAR::state_new_displacement)
  {
    DoContactSearch();
  }
}

void CONTACT::NitscheStrategyFsi::DoContactSearch()
{
  for (auto& interface : interface_)
  {
    interface->Initialize();
    interface->EvaluateSearchBinarytree();
    interface->EvaluateNodalNormals();
    interface->ExportNodalNormals();
  }
}

bool CONTACT::NitscheStrategyFsi::CheckNitscheContactState(CONTACT::Element* cele,
    const CORE::LINALG::Matrix<2, 1>& xsi, const double& full_fsi_traction, double& gap)
{
  return CONTACT::UTILS::CheckNitscheContactState(
      *ContactInterfaces()[0], pen_n_, weighting_, cele, xsi, full_fsi_traction, gap);
}

bool CONTACT::UTILS::CheckNitscheContactState(CONTACT::Interface& contactinterface,
    const double& pen_n, INPAR::CONTACT::NitscheWeighting weighting, CONTACT::Element* cele,
    const CORE::LINALG::Matrix<2, 1>& xsi, const double& full_fsi_traction, double& gap)
{
  // No master elements found
  if (!cele->MoData().NumSearchElements())
  {
    gap = 1.e12;
    return true;
  }
  if (!(cele->Shape() == CORE::FE::CellType::quad4 || cele->Shape() == CORE::FE::CellType::quad8 ||
          cele->Shape() == CORE::FE::CellType::quad9))
    dserror("This element shape is not yet implemented!");

  // find the corresponding master element
  CONTACT::Element* other_cele = nullptr;
  double mxi[2] = {0.0, 0.0};
  double projalpha = 0.0;
  static const double tol = 1e-4;
  double near = 0.;
  double max_relevant_gap = gap * 2.;  // safety factor 2
  for (int m = 0; m < cele->MoData().NumSearchElements(); ++m)
  {
    if (other_cele) break;
    auto* test_ele = dynamic_cast<CONTACT::Element*>(
        contactinterface.Discret().gElement(cele->MoData().SearchElements()[m]));
    if (!test_ele) dserror("Cannot find element with gid %d", cele->MoData().SearchElements()[m]);

    MORTAR::Projector::Impl(*cele, *test_ele)
        ->ProjectGaussPoint3D(*cele, xsi.A(), *test_ele, mxi, projalpha);
    bool is_inside = false;
    switch (test_ele->Shape())
    {
      case CORE::FE::CellType::quad4:
      case CORE::FE::CellType::quad8:
      case CORE::FE::CellType::quad9:
        if (abs(mxi[0]) < 1. + tol && abs(mxi[1]) < 1. + tol) is_inside = true;
        break;
      default:
        dserror("This element shape is not yet implemented (%d)!", test_ele->Shape());
    }
    if (is_inside) other_cele = test_ele;
    // distance check
    if (other_cele)
    {
      double center[2] = {0., 0.};
      CORE::LINALG::Matrix<3, 1> sc, mc;
      cele->LocalToGlobal(center, sc.A(), 0);
      other_cele->LocalToGlobal(center, mc.A(), 0);
      near = 2. * std::max(cele->MaxEdgeSize(), other_cele->MaxEdgeSize());
      sc.Update(-1., mc, 1.);
      if (sc.Norm2() > std::max(near, max_relevant_gap)) other_cele = nullptr;
    }
  }
  // orientation check
  if (other_cele)
  {
    double center[2] = {0., 0.};
    CORE::LINALG::Matrix<3, 1> sn, mn;
    cele->ComputeUnitNormalAtXi(center, sn.A());
    other_cele->ComputeUnitNormalAtXi(center, mn.A());
    if (sn.Dot(mn) > 0.) other_cele = nullptr;
  }
  // no master element hit
  if (other_cele == nullptr)
  {
    gap = 1e12;
    return true;
  }

  CORE::LINALG::Matrix<2, 1> mxi_m(mxi, true);
  double mx_glob[3];
  double sx_glob[3];
  cele->LocalToGlobal(xsi.A(), sx_glob, 0);
  other_cele->LocalToGlobal(mxi, mx_glob, 0);
  CORE::LINALG::Matrix<3, 1> mx(mx_glob, true);
  CORE::LINALG::Matrix<3, 1> sx(sx_glob, true);

  CORE::LINALG::Matrix<3, 1> n(mx);
  n.Update(-1., sx, 1.);
  CORE::LINALG::Matrix<3, 1> diff(n);
  n.Scale(1. / n.Norm2());
  gap = diff.Dot(n);
  CORE::LINALG::Matrix<3, 1> myN;
  cele->ComputeUnitNormalAtXi(xsi.A(), myN.A());
  double dir = n.Dot(myN);
  if (dir > 0)
    gap *= 1.;
  else
    gap *= -1.;

  // master element on the other side
  if (gap < -near)
  {
    gap = 1e12;
    return true;
  }


  double ws = 0.;
  double wm = 0.;
  double my_pen = pen_n;
  double my_pen_t = 0.0;
  CONTACT::UTILS::NitscheWeightsAndScaling(
      *cele, *other_cele, weighting, 1., ws, wm, my_pen, my_pen_t);

  CORE::LINALG::Matrix<3, 1> ele_n;
  cele->ComputeUnitNormalAtXi(xsi.A(), ele_n.A());

  double stress_plus_penalty =
      ws * CONTACT::UTILS::SolidCauchyAtXi(cele, xsi, ele_n, ele_n) +
      wm * CONTACT::UTILS::SolidCauchyAtXi(other_cele, mxi_m, ele_n, ele_n) + my_pen * gap;

  if (stress_plus_penalty >= full_fsi_traction)
    return true;  // aka evaluate FSI
  else
    return false;  // aka evaluate contact
}

BACI_NAMESPACE_CLOSE

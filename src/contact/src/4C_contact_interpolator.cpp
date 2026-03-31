// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_interpolator.hpp"

#include "4C_contact_defines.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_integrator.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_projector.hpp"
#include "4C_mortar_shape_utils.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                            farah 09/14|
 *----------------------------------------------------------------------*/
NTS::Interpolator::Interpolator(Teuchos::ParameterList& params, const int& dim)
    : iparams_(params),
      dim_(dim),
      pwslip_(iparams_.get<bool>("GP_SLIP_INCR")),
      wearlaw_(Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(iparams_, "WEARLAW")),
      wearimpl_(false),
      wearside_(Inpar::Wear::wear_slave),
      weartype_(Inpar::Wear::wear_intstate),
      wearshapefcn_(Inpar::Wear::wear_shape_standard),
      wearcoeff_(-1.0),
      wearcoeffm_(-1.0),
      sswear_(iparams_.get<bool>("SSWEAR")),
      ssslip_(iparams_.get<double>("SSSLIP"))
{
  // wear specific
  if (wearlaw_ != Inpar::Wear::wear_none)
  {
    // wear time integration
    auto wtimint = Teuchos::getIntegralValue<Inpar::Wear::WearTimInt>(params, "WEARTIMINT");
    if (wtimint == Inpar::Wear::wear_impl) wearimpl_ = true;

    // wear surface
    wearside_ = Teuchos::getIntegralValue<Inpar::Wear::WearSide>(iparams_, "BOTH_SIDED_WEAR");

    // wear algorithm
    weartype_ = Teuchos::getIntegralValue<Inpar::Wear::WearType>(iparams_, "WEARTYPE");

    // wear shape function
    wearshapefcn_ = Teuchos::getIntegralValue<Inpar::Wear::WearShape>(iparams_, "WEAR_SHAPEFCN");

    // wear coefficient
    wearcoeff_ = iparams_.get<double>("WEARCOEFF");

    // wear coefficient
    wearcoeffm_ = iparams_.get<double>("WEARCOEFF_MASTER");
  }

  return;
}


/*----------------------------------------------------------------------*
 |  interpolate (public)                                     farah 02/16|
 *----------------------------------------------------------------------*/
bool NTS::Interpolator::interpolate(Mortar::Node& snode, std::vector<Mortar::Element*> meles)
{
  // call sub functions for 2 and 3 dimensions
  if (dim_ == 2)
    interpolate_2d(snode, meles);
  else if (dim_ == 3)
    return interpolate_3d(snode, meles);
  else
    FOUR_C_THROW("wrong dimension");

  return true;
}


/*----------------------------------------------------------------------*
 |  interpolate (public)                                     farah 09/14|
 *----------------------------------------------------------------------*/
void NTS::Interpolator::interpolate_2d(Mortar::Node& snode, std::vector<Mortar::Element*> meles)
{
  // ********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************
  // check input data
  for (int i = 0; i < (int)meles.size(); ++i)
  {
    if ((!snode.is_slave()) || (meles[i]->is_slave()))
      FOUR_C_THROW("IntegrateAndDerivSegment called on a wrong type of Mortar::Element pair!");
  }

  // contact with wear
  bool wear = false;
  if (iparams_.get<double>("WEARCOEFF") > 1e-12) wear = true;

  // bool for node to node projection
  bool kink_projection = false;

  // calculate area -- simplified version
  double area = 0.0;

  // get first element (this is a dummy to use established algorithms)
  Mortar::Element* sele =
      dynamic_cast<Mortar::Element*>(snode.adjacent_elements()[0].user_element());

  CONTACT::Node& mynode = dynamic_cast<CONTACT::Node&>(snode);

  int lid = -1;
  for (int i = 0; i < sele->num_node(); ++i)
  {
    if ((sele->nodes()[i])->id() == snode.id())
    {
      lid = i;
      break;
    }
  }

  //**************************************************************
  //                loop over all Master Elements
  //**************************************************************
  for (int nummaster = 0; nummaster < (int)meles.size(); ++nummaster)
  {
    // project Gauss point onto master element
    double mxi[2] = {0.0, 0.0};
    Mortar::Projector::impl(*meles[nummaster])->project_nodal_normal(snode, *meles[nummaster], mxi);

    // node on mele?
    if ((mxi[0] >= -1.0) && (mxi[0] <= 1.0) && (kink_projection == false))
    {
      kink_projection = true;
      snode.has_proj() = true;

      int ndof = 2;
      int ncol = meles[nummaster]->num_node();
      Core::LinAlg::SerialDenseVector mval(ncol);
      Core::LinAlg::SerialDenseMatrix mderiv(ncol, 1);
      meles[nummaster]->evaluate_shape(mxi, mval, mderiv, ncol, false);

      // get slave and master nodal coords for Jacobian / GP evaluation
      Core::LinAlg::SerialDenseMatrix scoord(3, sele->num_node());
      Core::LinAlg::SerialDenseMatrix mcoord(3, ncol);
      sele->get_nodal_coords(scoord);
      meles[nummaster]->get_nodal_coords(mcoord);

      // nodal coords from previous time step and lagrange multiplier
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> scoordold;
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> mcoordold;
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> lagmult;

      scoordold = std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, sele->num_node());
      mcoordold = std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, ncol);
      lagmult = std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, sele->num_node());
      sele->get_nodal_coords_old(*scoordold);
      meles[nummaster]->get_nodal_coords_old(*mcoordold);
      sele->get_nodal_lag_mult(*lagmult);

      // TODO: calculate reasonable linsize
      int linsize = 100;
      double gpn[3] = {0.0, 0.0, 0.0};
      double jumpval = 0.0;
      Core::Gen::Pairedvector<int, double> dgap(linsize + ndof * ncol);
      Core::Gen::Pairedvector<int, double> dslipmatrix(linsize + ndof * ncol);
      Core::Gen::Pairedvector<int, double> dwear(linsize + ndof * ncol);
      //**************************************************************
      std::array<double, 2> sxi = {0.0, 0.0};

      if (sele->shape() == Core::FE::CellType::line2)
      {
        if (lid == 0) sxi[0] = -1;
        if (lid == 1) sxi[0] = 1;
      }
      else if (sele->shape() == Core::FE::CellType::line3)
      {
        if (lid == 0) sxi[0] = -1;
        if (lid == 1) sxi[0] = 1;
        if (lid == 2) sxi[0] = 0;
      }
      else
      {
        FOUR_C_THROW("Chosen element type not supported for NTS!");
      }
      //**************************************************************

      // evaluate the GP slave coordinate derivatives --> no entries
      Core::Gen::Pairedvector<int, double> dsxi(linsize + ndof * ncol);

      // evaluate the GP master coordinate derivatives
      Core::Gen::Pairedvector<int, double> dmxi(linsize + ndof * ncol);
      deriv_xi_gp_2d(*sele, *meles[nummaster], sxi[0], mxi[0], dsxi, dmxi, linsize);

      // calculate node-wise DM
      nw_d_m_2d(mynode, *sele, *meles[nummaster], mval, mderiv, dmxi);

      // calculate node-wise un-weighted gap
      nw_gap_2d(mynode, *sele, *meles[nummaster], mval, mderiv, dmxi, gpn);

      // calculate node-wise wear
      if (wear)
      {
        FOUR_C_THROW("stop");
        nw_wear_2d(mynode, *meles[nummaster], mval, mderiv, scoord, mcoord, *scoordold, *mcoordold,
            *lagmult, lid, linsize, jumpval, area, gpn, dmxi, dslipmatrix, dwear);
      }

      // calculate node-wise slip
      if (pwslip_)
      {
        nw_slip_2d(mynode, *meles[nummaster], mval, mderiv, scoord, mcoord, *scoordold, *mcoordold,
            lid, linsize, dmxi);
      }

      // calculate node-wise wear (prim. var.)
      if (weartype_ == Inpar::Wear::wear_primvar)
      {
        FOUR_C_THROW("stop");
        nw_t_e_2d(mynode, area, jumpval, dslipmatrix);
      }
    }  // End hit ele
  }  // End Loop over all Master Elements

  //**************************************************************

  return;
}


/*----------------------------------------------------------------------*
 |  interpolate (public)                                     farah 09/14|
 *----------------------------------------------------------------------*/
bool NTS::Interpolator::interpolate_3d(Mortar::Node& snode, std::vector<Mortar::Element*> meles)
{
  bool success = false;

  // ********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************

  bool kink_projection = false;

  // get first element (this is a dummy to use established algorithms)
  Mortar::Element* sele =
      dynamic_cast<Mortar::Element*>(snode.adjacent_elements()[0].user_element());

  CONTACT::Node& mynode = dynamic_cast<CONTACT::Node&>(snode);

  int lid = -1;
  for (int i = 0; i < sele->num_node(); ++i)
  {
    if ((sele->nodes()[i])->id() == snode.id())
    {
      lid = i;
      break;
    }
  }

  double sxi[2] = {0.0, 0.0};

  if (sele->shape() == Core::FE::CellType::quad4 or sele->shape() == Core::FE::CellType::quad8 or
      sele->shape() == Core::FE::CellType::quad9)
  {
    if (lid == 0)
    {
      sxi[0] = -1;
      sxi[1] = -1;
    }
    else if (lid == 1)
    {
      sxi[0] = 1;
      sxi[1] = -1;
    }
    else if (lid == 2)
    {
      sxi[0] = 1;
      sxi[1] = 1;
    }
    else if (lid == 3)
    {
      sxi[0] = -1;
      sxi[1] = 1;
    }
    else if (lid == 4)
    {
      sxi[0] = 0;
      sxi[1] = -1;
    }
    else if (lid == 5)
    {
      sxi[0] = 1;
      sxi[1] = 0;
    }
    else if (lid == 6)
    {
      sxi[0] = 0;
      sxi[1] = 1;
    }
    else if (lid == 7)
    {
      sxi[0] = -1;
      sxi[1] = 0;
    }
    else if (lid == 8)
    {
      sxi[0] = 0;
      sxi[1] = 0;
    }
    else
      FOUR_C_THROW("ERROR: wrong node LID");
  }
  else if (sele->shape() == Core::FE::CellType::tri3 or sele->shape() == Core::FE::CellType::tri6)
  {
    if (lid == 0)
    {
      sxi[0] = 0;
      sxi[1] = 0;
    }
    else if (lid == 1)
    {
      sxi[0] = 1;
      sxi[1] = 0;
    }
    else if (lid == 2)
    {
      sxi[0] = 0;
      sxi[1] = 1;
    }
    else if (lid == 3)
    {
      sxi[0] = 0.5;
      sxi[1] = 0;
    }
    else if (lid == 4)
    {
      sxi[0] = 0.5;
      sxi[1] = 0.5;
    }
    else if (lid == 5)
    {
      sxi[0] = 0;
      sxi[1] = 0.5;
    }
    else
      FOUR_C_THROW("ERROR: wrong node LID");
  }
  else
  {
    FOUR_C_THROW("Chosen element type not supported for NTS!");
  }

  //**************************************************************
  //                loop over all Master Elements
  //**************************************************************
  for (int nummaster = 0; nummaster < (int)meles.size(); ++nummaster)
  {
    // project Gauss point onto master element
    double mxi[2] = {0.0, 0.0};
    double projalpha = 0.0;
    Mortar::Projector::impl(*sele, *meles[nummaster])
        ->project_gauss_point_3d(*sele, sxi, *meles[nummaster], mxi, projalpha);

    bool is_on_mele = true;

    // check GP projection
    Core::FE::CellType dt = meles[nummaster]->shape();
    const double tol = 1e-8;
    if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
        dt == Core::FE::CellType::quad9)
    {
      if (mxi[0] < -1.0 - tol || mxi[1] < -1.0 - tol || mxi[0] > 1.0 + tol || mxi[1] > 1.0 + tol)
      {
        is_on_mele = false;
      }
    }
    else
    {
      if (mxi[0] < -tol || mxi[1] < -tol || mxi[0] > 1.0 + tol || mxi[1] > 1.0 + tol ||
          mxi[0] + mxi[1] > 1.0 + 2 * tol)
      {
        is_on_mele = false;
      }
    }

    // node on mele?
    if ((kink_projection == false) && (is_on_mele))
    {
      kink_projection = true;
      mynode.has_proj() = true;
      success = true;

      int ndof = 3;
      int ncol = meles[nummaster]->num_node();
      Core::LinAlg::SerialDenseVector mval(ncol);
      Core::LinAlg::SerialDenseMatrix mderiv(ncol, 2);
      meles[nummaster]->evaluate_shape(mxi, mval, mderiv, ncol, false);

      // get slave and master nodal coords for Jacobian / GP evaluation
      Core::LinAlg::SerialDenseMatrix scoord(3, sele->num_node());
      Core::LinAlg::SerialDenseMatrix mcoord(3, ncol);
      sele->get_nodal_coords(scoord);
      meles[nummaster]->get_nodal_coords(mcoord);

      // nodal coords from previous time step and lagrange multiplier
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> scoordold;
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> mcoordold;
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> lagmult;

      scoordold = std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, sele->num_node());
      mcoordold = std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, ncol);
      lagmult = std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, sele->num_node());
      sele->get_nodal_coords_old(*scoordold);
      meles[nummaster]->get_nodal_coords_old(*mcoordold);
      sele->get_nodal_lag_mult(*lagmult);

      int linsize = mynode.get_linsize();
      double gpn[3] = {0.0, 0.0, 0.0};
      //**************************************************************

      linsize *= 100;
      // evaluate the GP slave coordinate derivatives --> no entries
      std::vector<Core::Gen::Pairedvector<int, double>> dsxi(2, 0);
      std::vector<Core::Gen::Pairedvector<int, double>> dmxi(2, 4 * linsize + ncol * ndof);
      deriv_xi_gp_3d(*sele, *meles[nummaster], sxi, mxi, dsxi, dmxi, projalpha);

      // calculate node-wise DM
      nw_d_m_3d(mynode, *meles[nummaster], mval, mderiv, dmxi);

      // calculate node-wise un-weighted gap
      nw_gap_3d(mynode, *meles[nummaster], mval, mderiv, dmxi, gpn);

    }  // End hit ele
  }  // End Loop over all Master Elements

  //**************************************************************

  return success;
}


/*----------------------------------------------------------------------*
 |  interpolate (public)                                     seitz 08/15|
 *----------------------------------------------------------------------*/
void NTS::Interpolator::interpolate_master_temp_3d(
    Mortar::Element& sele, std::vector<Mortar::Element*> meles)
{
  // if it's not a TSI problem, there's nothing to do here
  if (dynamic_cast<CONTACT::Node*>(sele.nodes()[0])->has_tsi_data() == false) return;

  // ********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************
  // check input data
  for (int i = 0; i < (int)meles.size(); ++i)
  {
    if ((!sele.is_slave()) || (meles[i]->is_slave()))
      FOUR_C_THROW("interpolate_master_temp_3d called on a wrong type of Mortar::Element pair!");
  }

  //**************************************************************
  //                loop over all Slave nodes
  //**************************************************************
  for (int snodes = 0; snodes < sele.num_node(); ++snodes)
  {
    CONTACT::Node* mynode = dynamic_cast<CONTACT::Node*>(sele.nodes()[snodes]);

    double sxi[2] = {0.0, 0.0};

    if (sele.shape() == Core::FE::CellType::quad4 or sele.shape() == Core::FE::CellType::quad8 or
        sele.shape() == Core::FE::CellType::quad9)
    {
      if (snodes == 0)
      {
        sxi[0] = -1;
        sxi[1] = -1;
      }
      else if (snodes == 1)
      {
        sxi[0] = 1;
        sxi[1] = -1;
      }
      else if (snodes == 2)
      {
        sxi[0] = 1;
        sxi[1] = 1;
      }
      else if (snodes == 3)
      {
        sxi[0] = -1;
        sxi[1] = 1;
      }
      else if (snodes == 4)
      {
        sxi[0] = 0;
        sxi[1] = -1;
      }
      else if (snodes == 5)
      {
        sxi[0] = 1;
        sxi[1] = 0;
      }
      else if (snodes == 6)
      {
        sxi[0] = 0;
        sxi[1] = 1;
      }
      else if (snodes == 7)
      {
        sxi[0] = -1;
        sxi[1] = 0;
      }
      else if (snodes == 8)
      {
        sxi[0] = 0;
        sxi[1] = 0;
      }
      else
        FOUR_C_THROW("ERROR: wrong node LID");
    }
    else if (sele.shape() == Core::FE::CellType::tri3 or sele.shape() == Core::FE::CellType::tri6)
    {
      if (snodes == 0)
      {
        sxi[0] = 0;
        sxi[1] = 0;
      }
      else if (snodes == 1)
      {
        sxi[0] = 1;
        sxi[1] = 0;
      }
      else if (snodes == 2)
      {
        sxi[0] = 0;
        sxi[1] = 1;
      }
      else if (snodes == 3)
      {
        sxi[0] = 0.5;
        sxi[1] = 0;
      }
      else if (snodes == 4)
      {
        sxi[0] = 0.5;
        sxi[1] = 0.5;
      }
      else if (snodes == 5)
      {
        sxi[0] = 0;
        sxi[1] = 0.5;
      }
      else
        FOUR_C_THROW("ERROR: wrong node LID");
    }
    else
    {
      FOUR_C_THROW("Chosen element type not supported for NTS!");
    }

    //**************************************************************
    //                loop over all Master Elements
    //**************************************************************
    for (int nummaster = 0; nummaster < (int)meles.size(); ++nummaster)
    {
      // project Gauss point onto master element
      double mxi[2] = {0.0, 0.0};
      double projalpha = 0.0;
      Mortar::Projector::impl(sele, *meles[nummaster])
          ->project_gauss_point_3d(sele, sxi, *meles[nummaster], mxi, projalpha);

      bool is_on_mele = true;

      // check GP projection
      Core::FE::CellType dt = meles[nummaster]->shape();
      const double tol = 0.00;
      if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
          dt == Core::FE::CellType::quad9)
      {
        if (mxi[0] < -1.0 - tol || mxi[1] < -1.0 - tol || mxi[0] > 1.0 + tol || mxi[1] > 1.0 + tol)
        {
          is_on_mele = false;
        }
      }
      else
      {
        if (mxi[0] < -tol || mxi[1] < -tol || mxi[0] > 1.0 + tol || mxi[1] > 1.0 + tol ||
            mxi[0] + mxi[1] > 1.0 + 2 * tol)
        {
          is_on_mele = false;
        }
      }

      // node on mele?
      if (is_on_mele)
      {
        mynode->has_proj() = true;

        int ndof = 3;
        int ncol = meles[nummaster]->num_node();
        Core::LinAlg::SerialDenseVector mval(ncol);
        Core::LinAlg::SerialDenseMatrix mderiv(ncol, 2);
        meles[nummaster]->evaluate_shape(mxi, mval, mderiv, ncol, false);

        // get slave and master nodal coords for Jacobian / GP evaluation
        Core::LinAlg::SerialDenseMatrix scoord(3, sele.num_node());
        Core::LinAlg::SerialDenseMatrix mcoord(3, ncol);
        sele.get_nodal_coords(scoord);
        meles[nummaster]->get_nodal_coords(mcoord);

        int linsize = mynode->get_linsize();
        //**************************************************************

        // evaluate the GP slave coordinate derivatives --> no entries
        std::vector<Core::Gen::Pairedvector<int, double>> dsxi(2, 0);
        std::vector<Core::Gen::Pairedvector<int, double>> dmxi(2, 4 * linsize + ncol * ndof);
        deriv_xi_gp_3d(sele, *meles[nummaster], sxi, mxi, dsxi, dmxi, projalpha);

        // interpolate master side temperatures
        nw_master_temp(*mynode, *meles[nummaster], mval, mderiv, dmxi);
      }  // End hit ele
    }  // End Loop over all Master Elements
  }
  //**************************************************************

  return;
}


/*----------------------------------------------------------------------*
 |  node-wise TE for primary variable wear                  farah 09/14 |
 *----------------------------------------------------------------------*/
void NTS::Interpolator::nw_t_e_2d(CONTACT::Node& mynode, double& area, double& jumpval,
    Core::Gen::Pairedvector<int, double>& dslipmatrix)
{
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // multiply the two shape functions
  double prod1 = abs(jumpval);
  double prod2 = 1.0 * area;

  int col = mynode.dofs()[0];
  int row = 0;

  if (abs(prod1) > MORTARINTTOL)
    dynamic_cast<CONTACT::FriNode&>(mynode).add_t_value(row, col, prod1);
  if (abs(prod2) > MORTARINTTOL)
    dynamic_cast<CONTACT::FriNode&>(mynode).add_e_value(row, col, prod2);

  std::map<int, double>& tmmap_jk =
      dynamic_cast<CONTACT::FriNode&>(mynode).wear_data().get_deriv_tw()[mynode.id()];

  if (!sswear_)
  {
    double fac = 1.0;
    for (CI p = dslipmatrix.begin(); p != dslipmatrix.end(); ++p)
      tmmap_jk[p->first] += fac * (p->second);
  }
  return;
}


/*----------------------------------------------------------------------*
 |  node-wise slip                                          farah 09/14 |
 *----------------------------------------------------------------------*/
void NTS::Interpolator::nw_slip_2d(CONTACT::Node& mynode, Mortar::Element& mele,
    Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& mderiv,
    Core::LinAlg::SerialDenseMatrix& scoord, Core::LinAlg::SerialDenseMatrix& mcoord,
    Core::LinAlg::SerialDenseMatrix& scoordold, Core::LinAlg::SerialDenseMatrix& mcoordold,
    int& snodes, int& linsize, Core::Gen::Pairedvector<int, double>& dmxi)
{
  const int ncol = mele.num_node();
  const int ndof = mynode.num_dof();

  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  Core::Gen::Pairedvector<int, double> dslipgp(linsize + ndof * ncol);

  // LIN OF TANGENT
  Core::Gen::Pairedvector<int, double> dmap_txsl_gp(ncol * ndof + linsize);
  Core::Gen::Pairedvector<int, double> dmap_tysl_gp(ncol * ndof + linsize);

  // build interpolation of slave GP normal and coordinates
  std::array<double, 3> sjumpv = {0.0, 0.0, 0.0};
  std::array<double, 3> mjumpv = {0.0, 0.0, 0.0};
  std::array<double, 3> jumpv = {0.0, 0.0, 0.0};
  std::array<double, 3> tanv = {0.0, 0.0, 0.0};

  double tanlength = 0.0;
  double pwjump = 0.0;

  // nodal tangent interpolation
  tanv[0] += mynode.data().txi()[0];
  tanv[1] += mynode.data().txi()[1];
  tanv[2] += mynode.data().txi()[2];

  // delta D
  sjumpv[0] += (scoord(0, snodes) - (scoordold)(0, snodes));
  sjumpv[1] += (scoord(1, snodes) - (scoordold)(1, snodes));
  sjumpv[2] += (scoord(2, snodes) - (scoordold)(2, snodes));

  for (int i = 0; i < ncol; ++i)
  {
    mjumpv[0] += mval[i] * (mcoord(0, i) - (mcoordold)(0, i));
    mjumpv[1] += mval[i] * (mcoord(1, i) - (mcoordold)(1, i));
    mjumpv[2] += mval[i] * (mcoord(2, i) - (mcoordold)(2, i));
  }

  // normalize interpolated GP tangent back to length 1.0 !!!
  tanlength = sqrt(tanv[0] * tanv[0] + tanv[1] * tanv[1] + tanv[2] * tanv[2]);
  if (tanlength < 1.0e-12) FOUR_C_THROW("nw_slip_2d: Divide by zero!");

  for (int i = 0; i < 3; i++) tanv[i] /= tanlength;

  // jump
  jumpv[0] = sjumpv[0] - mjumpv[0];
  jumpv[1] = sjumpv[1] - mjumpv[1];
  jumpv[2] = sjumpv[2] - mjumpv[2];

  // multiply with tangent
  // value of relative tangential jump
  for (int i = 0; i < 3; ++i) pwjump += tanv[i] * jumpv[i];

  // *****************************************************************************
  // add everything to dslipgp                                                   *
  // *****************************************************************************
  Core::Gen::Pairedvector<int, double>& dmap_txsl_i = mynode.data().get_deriv_txi()[0];
  Core::Gen::Pairedvector<int, double>& dmap_tysl_i = mynode.data().get_deriv_txi()[1];

  for (CI p = dmap_txsl_i.begin(); p != dmap_txsl_i.end(); ++p)
    dmap_txsl_gp[p->first] += 1.0 * (p->second);
  for (CI p = dmap_tysl_i.begin(); p != dmap_tysl_i.end(); ++p)
    dmap_tysl_gp[p->first] += 1.0 * (p->second);

  // build directional derivative of slave GP tagent (unit)
  Core::Gen::Pairedvector<int, double> dmap_txsl_gp_unit(ncol * ndof + linsize);
  Core::Gen::Pairedvector<int, double> dmap_tysl_gp_unit(ncol * ndof + linsize);

  const double llv = tanlength * tanlength;
  const double linv = 1.0 / tanlength;
  const double lllinv = 1.0 / (tanlength * tanlength * tanlength);
  const double sxsxv = tanv[0] * tanv[0] * llv;
  const double sxsyv = tanv[0] * tanv[1] * llv;
  const double sysyv = tanv[1] * tanv[1] * llv;

  for (CI p = dmap_txsl_gp.begin(); p != dmap_txsl_gp.end(); ++p)
  {
    dmap_txsl_gp_unit[p->first] += linv * (p->second);
    dmap_txsl_gp_unit[p->first] -= lllinv * sxsxv * (p->second);
    dmap_tysl_gp_unit[p->first] -= lllinv * sxsyv * (p->second);
  }

  for (CI p = dmap_tysl_gp.begin(); p != dmap_tysl_gp.end(); ++p)
  {
    dmap_tysl_gp_unit[p->first] += linv * (p->second);
    dmap_tysl_gp_unit[p->first] -= lllinv * sysyv * (p->second);
    dmap_txsl_gp_unit[p->first] -= lllinv * sxsyv * (p->second);
  }

  for (CI p = dmap_txsl_gp_unit.begin(); p != dmap_txsl_gp_unit.end(); ++p)
    dslipgp[p->first] += jumpv[0] * (p->second);

  for (CI p = dmap_tysl_gp_unit.begin(); p != dmap_tysl_gp_unit.end(); ++p)
    dslipgp[p->first] += jumpv[1] * (p->second);

  // coord lin
  for (int k = 0; k < 2; ++k)
  {
    dslipgp[mynode.dofs()[k]] += tanv[k];
  }

  for (int z = 0; z < ncol; ++z)
  {
    CONTACT::Node* mnode = dynamic_cast<CONTACT::Node*>(mele.nodes()[z]);
    for (int k = 0; k < 2; ++k)
    {
      dslipgp[mnode->dofs()[k]] -= mval[z] * tanv[k];

      for (CI p = dmxi.begin(); p != dmxi.end(); ++p)
        dslipgp[p->first] -=
            tanv[k] * mderiv(z, 0) * (mcoord(k, z) - (mcoordold)(k, z)) * (p->second);
    }
  }

  // ***************************
  // Add to node!
  double prod = pwjump;

  // add current Gauss point's contribution to jump
  dynamic_cast<CONTACT::FriNode&>(mynode).add_jump_value(prod, 0);

  // get the corresponding map as a reference
  std::map<int, double>& djumpmap =
      dynamic_cast<CONTACT::FriNode&>(mynode).fri_data().get_deriv_var_jump()[0];

  double fac = 1.0;
  for (CI p = dslipgp.begin(); p != dslipgp.end(); ++p) djumpmap[p->first] += fac * (p->second);

  return;
}


/*----------------------------------------------------------------------*
 |  node-wise un-weighted gap                               farah 09/14 |
 *----------------------------------------------------------------------*/
void NTS::Interpolator::nw_wear_2d(CONTACT::Node& mynode, Mortar::Element& mele,
    Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& mderiv,
    Core::LinAlg::SerialDenseMatrix& scoord, Core::LinAlg::SerialDenseMatrix& mcoord,
    Core::LinAlg::SerialDenseMatrix& scoordold, Core::LinAlg::SerialDenseMatrix& mcoordold,
    Core::LinAlg::SerialDenseMatrix& lagmult, int& snodes, int& linsize, double& jumpval,
    double& area, double* gpn, Core::Gen::Pairedvector<int, double>& dmxi,
    Core::Gen::Pairedvector<int, double>& dslipmatrix, Core::Gen::Pairedvector<int, double>& dwear)
{
  const int ncol = mele.num_node();
  const int ndof = mynode.num_dof();

  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  std::array<double, 3> gpt = {0.0, 0.0, 0.0};
  std::array<double, 3> gplm = {0.0, 0.0, 0.0};
  std::array<double, 3> sgpjump = {0.0, 0.0, 0.0};
  std::array<double, 3> mgpjump = {0.0, 0.0, 0.0};
  std::array<double, 3> jump = {0.0, 0.0, 0.0};

  // for linearization
  double lm_lin = 0.0;
  double lengtht = 0.0;
  double wearval = 0.0;

  // nodal tangent interpolation
  gpt[0] += mynode.data().txi()[0];
  gpt[1] += mynode.data().txi()[1];
  gpt[2] += mynode.data().txi()[2];

  // delta D
  sgpjump[0] += (scoord(0, snodes) - ((scoordold)(0, snodes)));
  sgpjump[1] += (scoord(1, snodes) - ((scoordold)(1, snodes)));
  sgpjump[2] += (scoord(2, snodes) - ((scoordold)(2, snodes)));

  // LM interpolation
  gplm[0] += ((lagmult)(0, snodes));
  gplm[1] += ((lagmult)(1, snodes));
  gplm[2] += ((lagmult)(2, snodes));

  // normalize interpolated GP tangent back to length 1.0 !!!
  lengtht = sqrt(gpt[0] * gpt[0] + gpt[1] * gpt[1] + gpt[2] * gpt[2]);
  if (abs(lengtht) < 1.0e-12) FOUR_C_THROW("IntegrateAndDerivSegment: Divide by zero!");

  for (int i = 0; i < 3; i++) gpt[i] /= lengtht;

  // interpolation of master GP jumps (relative displacement increment)
  for (int i = 0; i < ncol; ++i)
  {
    mgpjump[0] += mval[i] * (mcoord(0, i) - (mcoordold)(0, i));
    mgpjump[1] += mval[i] * (mcoord(1, i) - (mcoordold)(1, i));
    mgpjump[2] += mval[i] * (mcoord(2, i) - (mcoordold)(2, i));
  }

  // jump
  jump[0] = sgpjump[0] - mgpjump[0];
  jump[1] = sgpjump[1] - mgpjump[1];
  jump[2] = sgpjump[2] - mgpjump[2];

  // evaluate wear
  // normal contact stress -- normal LM value
  for (int i = 0; i < 2; ++i)
  {
    wearval += gpn[i] * gplm[i];
    lm_lin += gpn[i] * gplm[i];  // required for linearization
  }

  // value of relative tangential jump
  for (int i = 0; i < 3; ++i) jumpval += gpt[i] * jump[i];

  if (sswear_) jumpval = ssslip_;

  // no jump --> no wear
  if (abs(jumpval) < 1e-12) return;

  // product
  // use non-abs value for implicit-wear algorithm
  // just for simple linear. maybe we change this in future
  wearval = abs(wearval) * abs(jumpval);

  double prod = wearval / area;

  // add current node wear to w
  dynamic_cast<CONTACT::FriNode&>(mynode).add_delta_weighted_wear_value(prod);

  //****************************************************************
  //   linearization for implicit algorithms
  //****************************************************************
  if ((wearimpl_ || weartype_ == Inpar::Wear::wear_primvar) and abs(jumpval) > 1e-12)
  {
    // lin. abs(x) = x/abs(x) * lin x.
    double xabsx = (jumpval / abs(jumpval)) * lm_lin;
    double xabsxT = (jumpval / abs(jumpval));

    // **********************************************************************
    // (1) Lin of normal for LM -- deriv normal maps from weighted gap lin.
    for (CI p = mynode.data().get_deriv_n()[0].begin(); p != mynode.data().get_deriv_n()[0].end();
        ++p)
      dwear[p->first] += abs(jumpval) * gplm[0] * (p->second);

    for (CI p = mynode.data().get_deriv_n()[1].begin(); p != mynode.data().get_deriv_n()[1].end();
        ++p)
      dwear[p->first] += abs(jumpval) * gplm[1] * (p->second);

    // **********************************************************************
    // (3) absolute incremental slip linearization:
    // (a) build directional derivative of slave GP tagent (non-unit)
    Core::Gen::Pairedvector<int, double> dmap_txsl_gp(ndof * ncol + linsize);
    Core::Gen::Pairedvector<int, double> dmap_tysl_gp(ndof * ncol + linsize);

    Core::Gen::Pairedvector<int, double>& dmap_txsl_i = mynode.data().get_deriv_txi()[0];
    Core::Gen::Pairedvector<int, double>& dmap_tysl_i = mynode.data().get_deriv_txi()[1];

    for (CI p = dmap_txsl_i.begin(); p != dmap_txsl_i.end(); ++p)
      dmap_txsl_gp[p->first] += (p->second);
    for (CI p = dmap_tysl_i.begin(); p != dmap_tysl_i.end(); ++p)
      dmap_tysl_gp[p->first] += (p->second);

    // (b) build directional derivative of slave GP tagent (unit)
    Core::Gen::Pairedvector<int, double> dmap_txsl_gp_unit(ndof * ncol + linsize);
    Core::Gen::Pairedvector<int, double> dmap_tysl_gp_unit(ndof * ncol + linsize);

    const double ll = lengtht * lengtht;
    const double linv = 1.0 / lengtht;
    const double lllinv = 1.0 / (lengtht * lengtht * lengtht);
    const double sxsx = gpt[0] * gpt[0] * ll;
    const double sxsy = gpt[0] * gpt[1] * ll;
    const double sysy = gpt[1] * gpt[1] * ll;

    for (CI p = dmap_txsl_gp.begin(); p != dmap_txsl_gp.end(); ++p)
    {
      dmap_txsl_gp_unit[p->first] += linv * (p->second);
      dmap_txsl_gp_unit[p->first] -= lllinv * sxsx * (p->second);
      dmap_tysl_gp_unit[p->first] -= lllinv * sxsy * (p->second);
    }

    for (CI p = dmap_tysl_gp.begin(); p != dmap_tysl_gp.end(); ++p)
    {
      dmap_tysl_gp_unit[p->first] += linv * (p->second);
      dmap_tysl_gp_unit[p->first] -= lllinv * sysy * (p->second);
      dmap_txsl_gp_unit[p->first] -= lllinv * sxsy * (p->second);
    }

    // add tangent lin. to dweargp
    for (CI p = dmap_txsl_gp_unit.begin(); p != dmap_txsl_gp_unit.end(); ++p)
      dwear[p->first] += xabsx * jump[0] * (p->second);

    for (CI p = dmap_tysl_gp_unit.begin(); p != dmap_tysl_gp_unit.end(); ++p)
      dwear[p->first] += xabsx * jump[1] * (p->second);

    // add tangent lin. to slip linearization for wear Tmatrix
    for (CI p = dmap_txsl_gp_unit.begin(); p != dmap_txsl_gp_unit.end(); ++p)
      dslipmatrix[p->first] += xabsxT * jump[0] * (p->second);

    for (CI p = dmap_tysl_gp_unit.begin(); p != dmap_tysl_gp_unit.end(); ++p)
      dslipmatrix[p->first] += xabsxT * jump[1] * (p->second);

    // **********************************************************************
    // (c) build directional derivative of jump
    Core::Gen::Pairedvector<int, double> dmap_slcoord_gp_x(ndof * ncol + linsize);
    Core::Gen::Pairedvector<int, double> dmap_slcoord_gp_y(ndof * ncol + linsize);

    Core::Gen::Pairedvector<int, double> dmap_mcoord_gp_x(ndof * ncol + linsize);
    Core::Gen::Pairedvector<int, double> dmap_mcoord_gp_y(ndof * ncol + linsize);

    Core::Gen::Pairedvector<int, double> dmap_coord_x(ndof * ncol + linsize);
    Core::Gen::Pairedvector<int, double> dmap_coord_y(ndof * ncol + linsize);

    // lin master part -- mxi
    for (int i = 0; i < ncol; ++i)
    {
      for (CI p = dmxi.begin(); p != dmxi.end(); ++p)
      {
        double valx = mderiv(i, 0) * (mcoord(0, i) - ((mcoordold)(0, i)));
        dmap_mcoord_gp_x[p->first] += valx * (p->second);
        double valy = mderiv(i, 0) * (mcoord(1, i) - ((mcoordold)(1, i)));
        dmap_mcoord_gp_y[p->first] += valy * (p->second);
      }
    }

    // deriv slave x-coords
    dmap_slcoord_gp_x[mynode.dofs()[0]] += 1.0;
    dmap_slcoord_gp_y[mynode.dofs()[1]] += 1.0;

    // deriv master x-coords
    for (int i = 0; i < ncol; ++i)
    {
      Mortar::Node* mnode = dynamic_cast<Mortar::Node*>(mele.nodes()[i]);

      dmap_mcoord_gp_x[mnode->dofs()[0]] += mval[i];
      dmap_mcoord_gp_y[mnode->dofs()[1]] += mval[i];
    }

    // slave: add to jumplin
    for (CI p = dmap_slcoord_gp_x.begin(); p != dmap_slcoord_gp_x.end(); ++p)
      dmap_coord_x[p->first] += (p->second);
    for (CI p = dmap_slcoord_gp_y.begin(); p != dmap_slcoord_gp_y.end(); ++p)
      dmap_coord_y[p->first] += (p->second);

    // master: add to jumplin
    for (CI p = dmap_mcoord_gp_x.begin(); p != dmap_mcoord_gp_x.end(); ++p)
      dmap_coord_x[p->first] -= (p->second);
    for (CI p = dmap_mcoord_gp_y.begin(); p != dmap_mcoord_gp_y.end(); ++p)
      dmap_coord_y[p->first] -= (p->second);

    // add to dweargp
    for (CI p = dmap_coord_x.begin(); p != dmap_coord_x.end(); ++p)
      dwear[p->first] += xabsx * gpt[0] * (p->second);

    for (CI p = dmap_coord_y.begin(); p != dmap_coord_y.end(); ++p)
      dwear[p->first] += xabsx * gpt[1] * (p->second);

    // add tangent lin. to slip linearization for wear Tmatrix
    for (CI p = dmap_coord_x.begin(); p != dmap_coord_x.end(); ++p)
      dslipmatrix[p->first] += xabsxT * gpt[0] * (p->second);

    for (CI p = dmap_coord_y.begin(); p != dmap_coord_y.end(); ++p)
      dslipmatrix[p->first] += xabsxT * gpt[1] * (p->second);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  node-wise un-weighted gap                               farah 09/14 |
 *----------------------------------------------------------------------*/
void NTS::Interpolator::nw_gap_2d(CONTACT::Node& mynode, Mortar::Element& sele,
    Mortar::Element& mele, Core::LinAlg::SerialDenseVector& mval,
    Core::LinAlg::SerialDenseMatrix& mderiv, Core::Gen::Pairedvector<int, double>& dmxi,
    double* gpn)
{
  const int ncol = mele.num_node();
  std::array<double, 3> sgpx = {0.0, 0.0, 0.0};
  std::array<double, 3> mgpx = {0.0, 0.0, 0.0};

  gpn[0] += mynode.mo_data().n()[0];
  gpn[1] += mynode.mo_data().n()[1];
  gpn[2] += mynode.mo_data().n()[2];

  sgpx[0] += mynode.xspatial()[0];
  sgpx[1] += mynode.xspatial()[1];
  sgpx[2] += mynode.xspatial()[2];

  // build interpolation of master GP coordinates
  for (int i = 0; i < ncol; ++i)
  {
    CONTACT::Node* mnode = dynamic_cast<CONTACT::Node*>(mele.nodes()[i]);

    mgpx[0] += mval[i] * mnode->xspatial()[0];
    mgpx[1] += mval[i] * mnode->xspatial()[1];
    mgpx[2] += mval[i] * mnode->xspatial()[2];
  }

  // normalize interpolated GP normal back to length 1.0 !!!
  double lengthn = sqrt(gpn[0] * gpn[0] + gpn[1] * gpn[1] + gpn[2] * gpn[2]);
  if (lengthn < 1.0e-12) FOUR_C_THROW("Divide by zero!");

  for (int i = 0; i < 3; ++i) gpn[i] /= lengthn;

  // build gap function at current GP
  double gap = 0.0;
  for (int i = 0; i < 2; ++i) gap += (mgpx[i] - sgpx[i]) * gpn[i];

  // **************************
  // add to node
  // **************************
  mynode.addnts_gap_value(gap);

  // **************************
  // linearization
  // **************************
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;
  Core::Gen::Pairedvector<int, double> dgapgp(10 * ncol);

  //*************************************************************
  for (CI p = mynode.data().get_deriv_n()[0].begin(); p != mynode.data().get_deriv_n()[0].end();
      ++p)
    dgapgp[p->first] += (mgpx[0] - sgpx[0]) * (p->second);

  for (CI p = mynode.data().get_deriv_n()[1].begin(); p != mynode.data().get_deriv_n()[1].end();
      ++p)
    dgapgp[p->first] += (mgpx[1] - sgpx[1]) * (p->second);


  for (int k = 0; k < 2; ++k)
  {
    dgapgp[mynode.dofs()[k]] -= (gpn[k]);
  }

  for (int z = 0; z < ncol; ++z)
  {
    Mortar::Node* mnode = dynamic_cast<Mortar::Node*>(mele.nodes()[z]);

    for (int k = 0; k < 2; ++k)
    {
      dgapgp[mnode->dofs()[k]] += mval[z] * gpn[k];

      for (CI p = dmxi.begin(); p != dmxi.end(); ++p)
        dgapgp[p->first] += gpn[k] * mderiv(z, 0) * mnode->xspatial()[k] * (p->second);
    }
  }

  std::map<int, double>& dgmap = mynode.data().get_deriv_gnts();

  // (1) Lin(g) - gap function
  for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += (p->second);

  return;
}


/*----------------------------------------------------------------------*
 |  node-wise un-weighted gap                               farah 09/14 |
 *----------------------------------------------------------------------*/
void NTS::Interpolator::nw_gap_3d(CONTACT::Node& mynode, Mortar::Element& mele,
    Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& mderiv,
    std::vector<Core::Gen::Pairedvector<int, double>>& dmxi, double* gpn)
{
  const int ncol = mele.num_node();

  std::array<double, 3> sgpx = {0.0, 0.0, 0.0};
  std::array<double, 3> mgpx = {0.0, 0.0, 0.0};

  gpn[0] += mynode.mo_data().n()[0];
  gpn[1] += mynode.mo_data().n()[1];
  gpn[2] += mynode.mo_data().n()[2];

  sgpx[0] += mynode.xspatial()[0];
  sgpx[1] += mynode.xspatial()[1];
  sgpx[2] += mynode.xspatial()[2];

  // build interpolation of master GP coordinates
  for (int i = 0; i < ncol; ++i)
  {
    CONTACT::Node* mnode = dynamic_cast<CONTACT::Node*>(mele.nodes()[i]);

    mgpx[0] += mval[i] * mnode->xspatial()[0];
    mgpx[1] += mval[i] * mnode->xspatial()[1];
    mgpx[2] += mval[i] * mnode->xspatial()[2];
  }

  // normalize interpolated GP normal back to length 1.0 !!!
  double lengthn = sqrt(gpn[0] * gpn[0] + gpn[1] * gpn[1] + gpn[2] * gpn[2]);
  if (lengthn < 1.0e-12) FOUR_C_THROW("Divide by zero!");

  for (int i = 0; i < 3; ++i) gpn[i] /= lengthn;

  // build gap function at current GP
  double gap = 0.0;
  for (int i = 0; i < 3; ++i) gap += (mgpx[i] - sgpx[i]) * gpn[i];

  // **************************
  // add to node
  // **************************
  mynode.addnts_gap_value(gap);

  // **************************
  // linearization
  // **************************
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // TODO: linsize for parallel simulations buggy. 100 for safety
  Core::Gen::Pairedvector<int, double> dgapgp(3 * ncol + 3 * mynode.get_linsize() + 100);

  //*************************************************************
  for (CI p = mynode.data().get_deriv_n()[0].begin(); p != mynode.data().get_deriv_n()[0].end();
      ++p)
    dgapgp[p->first] += (mgpx[0] - sgpx[0]) * (p->second);

  for (CI p = mynode.data().get_deriv_n()[1].begin(); p != mynode.data().get_deriv_n()[1].end();
      ++p)
    dgapgp[p->first] += (mgpx[1] - sgpx[1]) * (p->second);

  for (CI p = mynode.data().get_deriv_n()[2].begin(); p != mynode.data().get_deriv_n()[2].end();
      ++p)
    dgapgp[p->first] += (mgpx[2] - sgpx[2]) * (p->second);


  for (int k = 0; k < 3; ++k)
  {
    dgapgp[mynode.dofs()[k]] -= (gpn[k]);
  }

  for (int z = 0; z < ncol; ++z)
  {
    Mortar::Node* mnode = dynamic_cast<Mortar::Node*>(mele.nodes()[z]);

    for (int k = 0; k < 3; ++k)
    {
      dgapgp[mnode->dofs()[k]] += mval[z] * gpn[k];

      for (CI p = dmxi[0].begin(); p != dmxi[0].end(); ++p)
        dgapgp[p->first] += gpn[k] * mderiv(z, 0) * mnode->xspatial()[k] * (p->second);

      for (CI p = dmxi[1].begin(); p != dmxi[1].end(); ++p)
        dgapgp[p->first] += gpn[k] * mderiv(z, 1) * mnode->xspatial()[k] * (p->second);
    }
  }

  std::map<int, double>& dgmap = mynode.data().get_deriv_gnts();

  // (1) Lin(g) - gap function
  double fac = 1.0;
  for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += fac * (p->second);

  return;
}


/*----------------------------------------------------------------------*
 |  projected master temperature at the slave node          seitz 08/15 |
 *----------------------------------------------------------------------*/
void NTS::Interpolator::nw_master_temp(CONTACT::Node& mynode, Mortar::Element& mele,
    const Core::LinAlg::SerialDenseVector& mval, const Core::LinAlg::SerialDenseMatrix& mderiv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dmxi)
{
  const int ncol = mele.num_node();

  // build interpolation of master GP coordinates
  double mtemp = 0.;
  for (int i = 0; i < ncol; ++i)
  {
    CONTACT::Node* mnode = dynamic_cast<CONTACT::Node*>(mele.nodes()[i]);
    mtemp += mval[i] * mnode->tsi_data().temp();
  }
  mynode.tsi_data().temp_master() = mtemp;

  // **************************
  // linearization
  // **************************
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  std::map<int, double>& dTpdT = mynode.tsi_data().deriv_temp_master_temp();
  dTpdT.clear();
  for (int i = 0; i < mele.num_node(); ++i)
    dTpdT[dynamic_cast<Mortar::Node*>(mele.nodes()[i])->dofs()[0]] = mval[i];

  std::map<int, double>& dTpdd = mynode.tsi_data().deriv_temp_master_disp();
  dTpdd.clear();
  for (int d = 0; d < 2; ++d)
    for (CI p = dmxi[d].begin(); p != dmxi[d].end(); ++p)
    {
      double& dest = dTpdd[p->first];
      for (int mn = 0; mn < mele.num_node(); ++mn)
        dest += mderiv(mn, d) *
                (dynamic_cast<CONTACT::Node*>(mele.nodes()[mn])->tsi_data().temp()) * p->second;
    }

  return;
}


/*----------------------------------------------------------------------*
 |  node-wise D/M calculation                               farah 09/14 |
 *----------------------------------------------------------------------*/
void NTS::Interpolator::nw_d_m_2d(CONTACT::Node& mynode, Mortar::Element& sele,
    Mortar::Element& mele, Core::LinAlg::SerialDenseVector& mval,
    Core::LinAlg::SerialDenseMatrix& mderiv, Core::Gen::Pairedvector<int, double>& dmxi)
{
  const int ncol = mele.num_node();
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // node-wise M value
  for (int k = 0; k < ncol; ++k)
  {
    CONTACT::Node* mnode = dynamic_cast<CONTACT::Node*>(mele.nodes()[k]);

    // multiply the two shape functions
    double prod = mval[k];

    if (abs(prod) > MORTARINTTOL) mynode.add_mnts_value(mnode->id(), prod);
    if (abs(prod) > MORTARINTTOL) mynode.add_m_node(mnode->id());  // only for friction!
  }

  // integrate dseg
  // multiply the two shape functions
  double prod = 1.0;
  if (abs(prod) > MORTARINTTOL) mynode.add_dnts_value(mynode.id(), prod);
  if (abs(prod) > MORTARINTTOL) mynode.add_s_node(mynode.id());  // only for friction!

  // integrate LinM
  for (int k = 0; k < ncol; ++k)
  {
    // global master node ID
    int mgid = mele.nodes()[k]->id();
    double fac = 0.0;

    // get the correct map as a reference
    std::map<int, double>& dmmap_jk = mynode.data().get_deriv_mnts()[mgid];

    // (3) Lin(NMaster) - master GP coordinates
    fac = mderiv(k, 0);
    for (CI p = dmxi.begin(); p != dmxi.end(); ++p) dmmap_jk[p->first] += fac * (p->second);
  }  // loop over master nodes

  return;
}


/*----------------------------------------------------------------------*
 |  node-wise D/M calculation                               farah 09/14 |
 *----------------------------------------------------------------------*/
void NTS::Interpolator::nw_d_m_3d(CONTACT::Node& mynode, Mortar::Element& mele,
    Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& mderiv,
    std::vector<Core::Gen::Pairedvector<int, double>>& dmxi)
{
  const int ncol = mele.num_node();

  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // node-wise M value
  for (int k = 0; k < ncol; ++k)
  {
    CONTACT::Node* mnode = dynamic_cast<CONTACT::Node*>(mele.nodes()[k]);

    // multiply the two shape functions
    double prod = mval[k];

    if (abs(prod) > MORTARINTTOL) mynode.add_mnts_value(mnode->id(), prod);
    if (abs(prod) > MORTARINTTOL) mynode.add_m_node(mnode->id());  // only for friction!
  }

  // integrate dseg
  // multiply the two shape functions
  double prod = 1.0;
  if (abs(prod) > MORTARINTTOL) mynode.add_dnts_value(mynode.id(), prod);
  if (abs(prod) > MORTARINTTOL) mynode.add_s_node(mynode.id());  // only for friction!

  // integrate LinM
  for (int k = 0; k < ncol; ++k)
  {
    // global master node ID
    int mgid = mele.nodes()[k]->id();
    double fac = 0.0;

    // get the correct map as a reference
    std::map<int, double>& dmmap_jk = mynode.data().get_deriv_mnts()[mgid];

    fac = mderiv(k, 0);
    for (CI p = dmxi[0].begin(); p != dmxi[0].end(); ++p) dmmap_jk[p->first] += fac * (p->second);

    fac = mderiv(k, 1);
    for (CI p = dmxi[1].begin(); p != dmxi[1].end(); ++p) dmmap_jk[p->first] += fac * (p->second);
  }  // loop over master nodes

  return;
}


/*----------------------------------------------------------------------*
 |  Compute directional derivative of XiGP master (2D)       popp 05/08 |
 *----------------------------------------------------------------------*/
void NTS::Interpolator::deriv_xi_gp_2d(Mortar::Element& sele, Mortar::Element& mele, double& sxigp,
    double& mxigp, const Core::Gen::Pairedvector<int, double>& derivsxi,
    Core::Gen::Pairedvector<int, double>& derivmxi, int& linsize)
{
  // check for problem dimension

  // we need the participating slave and master nodes
  Core::Nodes::Node** snodes = nullptr;
  Core::Nodes::Node** mnodes = nullptr;
  int numsnode = sele.num_node();
  int nummnode = mele.num_node();

  int ndof = 2;
  snodes = sele.nodes();
  mnodes = mele.nodes();

  std::vector<Mortar::Node*> smrtrnodes(numsnode);
  std::vector<Mortar::Node*> mmrtrnodes(nummnode);

  for (int i = 0; i < numsnode; ++i)
  {
    smrtrnodes[i] = dynamic_cast<Mortar::Node*>(snodes[i]);
    if (!smrtrnodes[i]) FOUR_C_THROW("DerivXiAB2D: Null pointer!");
  }

  for (int i = 0; i < nummnode; ++i)
  {
    mmrtrnodes[i] = dynamic_cast<Mortar::Node*>(mnodes[i]);
    if (!mmrtrnodes[i]) FOUR_C_THROW("DerivXiAB2D: Null pointer!");
  }

  // we also need shape function derivs in A and B
  double psxigp[2] = {sxigp, 0.0};
  double pmxigp[2] = {mxigp, 0.0};
  Core::LinAlg::SerialDenseVector valsxigp(numsnode);
  Core::LinAlg::SerialDenseVector valmxigp(nummnode);
  Core::LinAlg::SerialDenseMatrix derivsxigp(numsnode, 1);
  Core::LinAlg::SerialDenseMatrix derivmxigp(nummnode, 1);

  sele.evaluate_shape(psxigp, valsxigp, derivsxigp, numsnode, false);
  mele.evaluate_shape(pmxigp, valmxigp, derivmxigp, nummnode, false);

  // we also need the GP slave coordinates + normal
  std::array<double, 3> sgpn = {0.0, 0.0, 0.0};
  std::array<double, 3> sgpx = {0.0, 0.0, 0.0};
  for (int i = 0; i < numsnode; ++i)
  {
    sgpn[0] += valsxigp[i] * smrtrnodes[i]->mo_data().n()[0];
    sgpn[1] += valsxigp[i] * smrtrnodes[i]->mo_data().n()[1];
    sgpn[2] += valsxigp[i] * smrtrnodes[i]->mo_data().n()[2];

    sgpx[0] += valsxigp[i] * smrtrnodes[i]->xspatial()[0];
    sgpx[1] += valsxigp[i] * smrtrnodes[i]->xspatial()[1];
    sgpx[2] += valsxigp[i] * smrtrnodes[i]->xspatial()[2];
  }

  // normalize interpolated GP normal back to length 1.0 !!!
  const double length = sqrt(sgpn[0] * sgpn[0] + sgpn[1] * sgpn[1] + sgpn[2] * sgpn[2]);
  if (length < 1.0e-12) FOUR_C_THROW("deriv_xi_gp_2d: Divide by zero!");
  for (int i = 0; i < 3; ++i) sgpn[i] /= length;

  // compute factors and leading constants for master
  double cmxigp = 0.0;
  double fac_dxm_gp = 0.0;
  double fac_dym_gp = 0.0;
  double fac_xmsl_gp = 0.0;
  double fac_ymsl_gp = 0.0;

  for (int i = 0; i < nummnode; ++i)
  {
    fac_dxm_gp += derivmxigp(i, 0) * (mmrtrnodes[i]->xspatial()[0]);
    fac_dym_gp += derivmxigp(i, 0) * (mmrtrnodes[i]->xspatial()[1]);

    fac_xmsl_gp += valmxigp[i] * (mmrtrnodes[i]->xspatial()[0]);
    fac_ymsl_gp += valmxigp[i] * (mmrtrnodes[i]->xspatial()[1]);
  }

  cmxigp = -1 / (fac_dxm_gp * sgpn[1] - fac_dym_gp * sgpn[0]);
  // std::cout << "cmxigp: " << cmxigp << std::endl;

  fac_xmsl_gp -= sgpx[0];
  fac_ymsl_gp -= sgpx[1];

  // prepare linearization
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // build directional derivative of slave GP coordinates
  Core::Gen::Pairedvector<int, double> dmap_xsl_gp(linsize + nummnode * ndof);
  Core::Gen::Pairedvector<int, double> dmap_ysl_gp(linsize + nummnode * ndof);

  for (int i = 0; i < numsnode; ++i)
  {
    dmap_xsl_gp[smrtrnodes[i]->dofs()[0]] += valsxigp[i];
    dmap_ysl_gp[smrtrnodes[i]->dofs()[1]] += valsxigp[i];

    for (CI p = derivsxi.begin(); p != derivsxi.end(); ++p)
    {
      double facx = derivsxigp(i, 0) * (smrtrnodes[i]->xspatial()[0]);
      double facy = derivsxigp(i, 0) * (smrtrnodes[i]->xspatial()[1]);
      dmap_xsl_gp[p->first] += facx * (p->second);
      dmap_ysl_gp[p->first] += facy * (p->second);
    }
  }

  // build directional derivative of slave GP normal
  Core::Gen::Pairedvector<int, double> dmap_nxsl_gp(linsize + nummnode * ndof);
  Core::Gen::Pairedvector<int, double> dmap_nysl_gp(linsize + nummnode * ndof);

  std::array<double, 3> sgpnmod = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i) sgpnmod[i] = sgpn[i] * length;

  Core::Gen::Pairedvector<int, double> dmap_nxsl_gp_mod(linsize + nummnode * ndof);
  Core::Gen::Pairedvector<int, double> dmap_nysl_gp_mod(linsize + nummnode * ndof);

  for (int i = 0; i < numsnode; ++i)
  {
    Core::Gen::Pairedvector<int, double>& dmap_nxsl_i =
        dynamic_cast<CONTACT::Node*>(smrtrnodes[i])->data().get_deriv_n()[0];
    Core::Gen::Pairedvector<int, double>& dmap_nysl_i =
        dynamic_cast<CONTACT::Node*>(smrtrnodes[i])->data().get_deriv_n()[1];

    for (CI p = dmap_nxsl_i.begin(); p != dmap_nxsl_i.end(); ++p)
      dmap_nxsl_gp_mod[p->first] += valsxigp[i] * (p->second);
    for (CI p = dmap_nysl_i.begin(); p != dmap_nysl_i.end(); ++p)
      dmap_nysl_gp_mod[p->first] += valsxigp[i] * (p->second);

    for (CI p = derivsxi.begin(); p != derivsxi.end(); ++p)
    {
      double valx = derivsxigp(i, 0) * smrtrnodes[i]->mo_data().n()[0];
      dmap_nxsl_gp_mod[p->first] += valx * (p->second);
      double valy = derivsxigp(i, 0) * smrtrnodes[i]->mo_data().n()[1];
      dmap_nysl_gp_mod[p->first] += valy * (p->second);
    }
  }

  const double sxsx = sgpnmod[0] * sgpnmod[0];
  const double sxsy = sgpnmod[0] * sgpnmod[1];
  const double sysy = sgpnmod[1] * sgpnmod[1];
  const double linv = 1.0 / length;
  const double lllinv = 1.0 / (length * length * length);

  for (CI p = dmap_nxsl_gp_mod.begin(); p != dmap_nxsl_gp_mod.end(); ++p)
  {
    dmap_nxsl_gp[p->first] += linv * (p->second);
    dmap_nxsl_gp[p->first] -= lllinv * sxsx * (p->second);
    dmap_nysl_gp[p->first] -= lllinv * sxsy * (p->second);
  }

  for (CI p = dmap_nysl_gp_mod.begin(); p != dmap_nysl_gp_mod.end(); ++p)
  {
    dmap_nysl_gp[p->first] += linv * (p->second);
    dmap_nysl_gp[p->first] -= lllinv * sysy * (p->second);
    dmap_nxsl_gp[p->first] -= lllinv * sxsy * (p->second);
  }

  // *********************************************************************
  // finally compute Lin(XiGP_master)
  // *********************************************************************

  // add derivative of slave GP coordinates
  for (CI p = dmap_xsl_gp.begin(); p != dmap_xsl_gp.end(); ++p)
    derivmxi[p->first] -= sgpn[1] * (p->second);
  for (CI p = dmap_ysl_gp.begin(); p != dmap_ysl_gp.end(); ++p)
    derivmxi[p->first] += sgpn[0] * (p->second);

  // add derivatives of master node coordinates
  for (int i = 0; i < nummnode; ++i)
  {
    derivmxi[mmrtrnodes[i]->dofs()[0]] += valmxigp[i] * sgpn[1];
    derivmxi[mmrtrnodes[i]->dofs()[1]] -= valmxigp[i] * sgpn[0];
  }

  // add derivative of slave GP normal
  for (CI p = dmap_nxsl_gp.begin(); p != dmap_nxsl_gp.end(); ++p)
    derivmxi[p->first] -= fac_ymsl_gp * (p->second);
  for (CI p = dmap_nysl_gp.begin(); p != dmap_nysl_gp.end(); ++p)
    derivmxi[p->first] += fac_xmsl_gp * (p->second);

  // multiply all entries with cmxigp
  for (CI p = derivmxi.begin(); p != derivmxi.end(); ++p) derivmxi[p->first] = cmxigp * (p->second);

  return;
}


/*----------------------------------------------------------------------*
 |  Compute directional derivative of XiGP master (3D)        popp 02/09|
 *----------------------------------------------------------------------*/
void NTS::Interpolator::deriv_xi_gp_3d(Mortar::Element& sele, Mortar::Element& mele, double* sxigp,
    double* mxigp, const std::vector<Core::Gen::Pairedvector<int, double>>& derivsxi,
    std::vector<Core::Gen::Pairedvector<int, double>>& derivmxi, double& alpha)
{
  // we need the participating slave and master nodes
  Core::Nodes::Node** snodes = sele.nodes();
  Core::Nodes::Node** mnodes = mele.nodes();
  std::vector<Mortar::Node*> smrtrnodes(sele.num_node());
  std::vector<Mortar::Node*> mmrtrnodes(mele.num_node());
  const int numsnode = sele.num_node();
  const int nummnode = mele.num_node();

  for (int i = 0; i < numsnode; ++i)
  {
    smrtrnodes[i] = dynamic_cast<Mortar::Node*>(snodes[i]);
    if (!smrtrnodes[i]) FOUR_C_THROW("DerivXiGP3D: Null pointer!");
  }

  for (int i = 0; i < nummnode; ++i)
  {
    mmrtrnodes[i] = dynamic_cast<Mortar::Node*>(mnodes[i]);
    if (!mmrtrnodes[i]) FOUR_C_THROW("DerivXiGP3D: Null pointer!");
  }

  // we also need shape function derivs at the GP
  Core::LinAlg::SerialDenseVector valsxigp(numsnode);
  Core::LinAlg::SerialDenseVector valmxigp(nummnode);
  Core::LinAlg::SerialDenseMatrix derivsxigp(numsnode, 2, true);
  Core::LinAlg::SerialDenseMatrix derivmxigp(nummnode, 2, true);

  sele.evaluate_shape(sxigp, valsxigp, derivsxigp, numsnode);
  mele.evaluate_shape(mxigp, valmxigp, derivmxigp, nummnode);

  // we also need the GP slave coordinates + normal
  std::array<double, 3> sgpn = {0.0, 0.0, 0.0};
  std::array<double, 3> sgpx = {0.0, 0.0, 0.0};
  for (int i = 0; i < numsnode; ++i)
    for (int k = 0; k < 3; ++k)
    {
      sgpn[k] += valsxigp[i] * smrtrnodes[i]->mo_data().n()[k];
      sgpx[k] += valsxigp[i] * smrtrnodes[i]->xspatial()[k];
    }

  // build 3x3 factor matrix L
  Core::LinAlg::Matrix<3, 3> lmatrix(Core::LinAlg::Initialization::zero);
  for (int k = 0; k < 3; ++k) lmatrix(k, 2) = -sgpn[k];
  for (int z = 0; z < nummnode; ++z)
    for (int k = 0; k < 3; ++k)
    {
      lmatrix(k, 0) += derivmxigp(z, 0) * mmrtrnodes[z]->xspatial()[k];
      lmatrix(k, 1) += derivmxigp(z, 1) * mmrtrnodes[z]->xspatial()[k];
    }

  // get inverse of the 3x3 matrix L (in place)
  if (abs(lmatrix.determinant()) < 1e-12) FOUR_C_THROW("Singular lmatrix for derivgp3d");

  lmatrix.invert();

  // build directional derivative of slave GP normal
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  int linsize = 0;
  for (int i = 0; i < numsnode; ++i)
  {
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(snodes[i]);
    linsize += cnode->get_linsize();
  }

  // TODO: this is for safety. Change to reasonable value!
  linsize *= 100;

  Core::Gen::Pairedvector<int, double> dmap_nxsl_gp(linsize);
  Core::Gen::Pairedvector<int, double> dmap_nysl_gp(linsize);
  Core::Gen::Pairedvector<int, double> dmap_nzsl_gp(linsize);

  for (int i = 0; i < numsnode; ++i)
  {
    Core::Gen::Pairedvector<int, double>& dmap_nxsl_i =
        dynamic_cast<CONTACT::Node*>(smrtrnodes[i])->data().get_deriv_n()[0];
    Core::Gen::Pairedvector<int, double>& dmap_nysl_i =
        dynamic_cast<CONTACT::Node*>(smrtrnodes[i])->data().get_deriv_n()[1];
    Core::Gen::Pairedvector<int, double>& dmap_nzsl_i =
        dynamic_cast<CONTACT::Node*>(smrtrnodes[i])->data().get_deriv_n()[2];

    for (CI p = dmap_nxsl_i.begin(); p != dmap_nxsl_i.end(); ++p)
      dmap_nxsl_gp[p->first] += valsxigp[i] * (p->second);
    for (CI p = dmap_nysl_i.begin(); p != dmap_nysl_i.end(); ++p)
      dmap_nysl_gp[p->first] += valsxigp[i] * (p->second);
    for (CI p = dmap_nzsl_i.begin(); p != dmap_nzsl_i.end(); ++p)
      dmap_nzsl_gp[p->first] += valsxigp[i] * (p->second);

    for (CI p = derivsxi[0].begin(); p != derivsxi[0].end(); ++p)
    {
      double valx = derivsxigp(i, 0) * smrtrnodes[i]->mo_data().n()[0];
      dmap_nxsl_gp[p->first] += valx * (p->second);
      double valy = derivsxigp(i, 0) * smrtrnodes[i]->mo_data().n()[1];
      dmap_nysl_gp[p->first] += valy * (p->second);
      double valz = derivsxigp(i, 0) * smrtrnodes[i]->mo_data().n()[2];
      dmap_nzsl_gp[p->first] += valz * (p->second);
    }

    for (CI p = derivsxi[1].begin(); p != derivsxi[1].end(); ++p)
    {
      double valx = derivsxigp(i, 1) * smrtrnodes[i]->mo_data().n()[0];
      dmap_nxsl_gp[p->first] += valx * (p->second);
      double valy = derivsxigp(i, 1) * smrtrnodes[i]->mo_data().n()[1];
      dmap_nysl_gp[p->first] += valy * (p->second);
      double valz = derivsxigp(i, 1) * smrtrnodes[i]->mo_data().n()[2];
      dmap_nzsl_gp[p->first] += valz * (p->second);
    }
  }

  // start to fill linearization maps for master GP
  // (1) all master nodes coordinates part
  for (int z = 0; z < nummnode; ++z)
  {
    for (int k = 0; k < 3; ++k)
    {
      derivmxi[0][mmrtrnodes[z]->dofs()[k]] -= valmxigp[z] * lmatrix(0, k);
      derivmxi[1][mmrtrnodes[z]->dofs()[k]] -= valmxigp[z] * lmatrix(1, k);
    }
  }

  // (2) slave Gauss point coordinates part
  for (int z = 0; z < numsnode; ++z)
  {
    for (int k = 0; k < 3; ++k)
    {
      derivmxi[0][smrtrnodes[z]->dofs()[k]] += valsxigp[z] * lmatrix(0, k);
      derivmxi[1][smrtrnodes[z]->dofs()[k]] += valsxigp[z] * lmatrix(1, k);

      for (CI p = derivsxi[0].begin(); p != derivsxi[0].end(); ++p)
      {
        derivmxi[0][p->first] +=
            derivsxigp(z, 0) * smrtrnodes[z]->xspatial()[k] * lmatrix(0, k) * (p->second);
        derivmxi[1][p->first] +=
            derivsxigp(z, 0) * smrtrnodes[z]->xspatial()[k] * lmatrix(1, k) * (p->second);
      }

      for (CI p = derivsxi[1].begin(); p != derivsxi[1].end(); ++p)
      {
        derivmxi[0][p->first] +=
            derivsxigp(z, 1) * smrtrnodes[z]->xspatial()[k] * lmatrix(0, k) * (p->second);
        derivmxi[1][p->first] +=
            derivsxigp(z, 1) * smrtrnodes[z]->xspatial()[k] * lmatrix(1, k) * (p->second);
      }
    }
  }

  // (3) slave Gauss point normal part
  for (CI p = dmap_nxsl_gp.begin(); p != dmap_nxsl_gp.end(); ++p)
  {
    derivmxi[0][p->first] += alpha * lmatrix(0, 0) * (p->second);
    derivmxi[1][p->first] += alpha * lmatrix(1, 0) * (p->second);
  }
  for (CI p = dmap_nysl_gp.begin(); p != dmap_nysl_gp.end(); ++p)
  {
    derivmxi[0][p->first] += alpha * lmatrix(0, 1) * (p->second);
    derivmxi[1][p->first] += alpha * lmatrix(1, 1) * (p->second);
  }
  for (CI p = dmap_nzsl_gp.begin(); p != dmap_nzsl_gp.end(); ++p)
  {
    derivmxi[0][p->first] += alpha * lmatrix(0, 2) * (p->second);
    derivmxi[1][p->first] += alpha * lmatrix(1, 2) * (p->second);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Implementation for meshtying interpolator                farah 10/14|
 *----------------------------------------------------------------------*/
NTS::MTInterpolator* NTS::MTInterpolator::impl(std::vector<Mortar::Element*> meles)
{
  // TODO: maybe this object should be crearted for one mele
  // and note for a vector of meles...
  switch (meles[0]->shape())
  {
    // 2D surface elements
    case Core::FE::CellType::quad4:
    {
      return MTInterpolatorCalc<Core::FE::CellType::quad4>::instance(
          Core::Utils::SingletonAction::create);
    }
    case Core::FE::CellType::quad8:
    {
      return MTInterpolatorCalc<Core::FE::CellType::quad8>::instance(
          Core::Utils::SingletonAction::create);
    }
    case Core::FE::CellType::quad9:
    {
      return MTInterpolatorCalc<Core::FE::CellType::quad9>::instance(
          Core::Utils::SingletonAction::create);
    }
    case Core::FE::CellType::tri3:
    {
      return MTInterpolatorCalc<Core::FE::CellType::tri3>::instance(
          Core::Utils::SingletonAction::create);
    }
    case Core::FE::CellType::tri6:
    {
      return MTInterpolatorCalc<Core::FE::CellType::tri6>::instance(
          Core::Utils::SingletonAction::create);
    }
      // 1D surface elements
    case Core::FE::CellType::line2:
    {
      return MTInterpolatorCalc<Core::FE::CellType::line2>::instance(
          Core::Utils::SingletonAction::create);
    }
    case Core::FE::CellType::line3:
    {
      return MTInterpolatorCalc<Core::FE::CellType::line3>::instance(
          Core::Utils::SingletonAction::create);
    }
    default:
      FOUR_C_THROW("Chosen element type not supported!");
      break;
  }
  return nullptr;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            farah 10/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_m>
NTS::MTInterpolatorCalc<distype_m>::MTInterpolatorCalc()
{
  //...
}

template <Core::FE::CellType distype_m>
NTS::MTInterpolatorCalc<distype_m>* NTS::MTInterpolatorCalc<distype_m>::instance(
    Core::Utils::SingletonAction action)
{
  static auto singleton_owner = Core::Utils::make_singleton_owner(
      []()
      {
        return std::unique_ptr<NTS::MTInterpolatorCalc<distype_m>>(
            new NTS::MTInterpolatorCalc<distype_m>());
      });

  return singleton_owner.instance(action);
}


/*----------------------------------------------------------------------*
 |  interpolate (public)                                     farah 10/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_m>
void NTS::MTInterpolatorCalc<distype_m>::interpolate(
    Mortar::Node& snode, std::vector<Mortar::Element*> meles)
{
  if (ndim_ == 2)
    interpolate_2d(snode, meles);
  else if (ndim_ == 3)
    interpolate_3d(snode, meles);
  else
    FOUR_C_THROW("wrong dimension!");

  return;
}


/*----------------------------------------------------------------------*
 |  interpolate (public)                                     farah 10/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_m>
void NTS::MTInterpolatorCalc<distype_m>::interpolate_2d(
    Mortar::Node& snode, std::vector<Mortar::Element*> meles)
{
  // ********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************
  // check input data
  for (int i = 0; i < (int)meles.size(); ++i)
  {
    if ((!snode.is_slave()) || (meles[i]->is_slave()))
      FOUR_C_THROW("IntegrateAndDerivSegment called on a wrong type of Mortar::Element pair!");
  }

  // bool for projection onto a master node
  bool kink_projection = false;

  //**************************************************************
  //                loop over all Master Elements
  //**************************************************************
  for (int nummaster = 0; nummaster < (int)meles.size(); ++nummaster)
  {
    // project Gauss point onto master element
    double mxi[2] = {0.0, 0.0};
    Mortar::Projector::impl(*meles[nummaster])->project_nodal_normal(snode, *meles[nummaster], mxi);

    // node on mele?
    if ((mxi[0] >= -1.0) && (mxi[0] <= 1.0) && (kink_projection == false))
    {
      kink_projection = true;
      snode.has_proj() = true;

      static Core::LinAlg::Matrix<nm_, 1> mval;
      Mortar::Utils::evaluate_shape_displ(mxi, mval, *meles[nummaster], false);

      // node-wise M value
      for (int k = 0; k < nm_; ++k)
      {
        Mortar::Node* mnode = dynamic_cast<Mortar::Node*>(meles[nummaster]->nodes()[k]);

        // multiply the two shape functions
        double prod = mval(k);

        if (abs(prod) > MORTARINTTOL) snode.add_m_value(mnode->id(), prod);
      }

      // dseg reduces to 1.0 for nts
      double prod = 1.0;

      if (abs(prod) > MORTARINTTOL) snode.add_d_value(snode.id(), prod);
    }  // End hit ele
  }  // End Loop over all Master Elements

  //**************************************************************

  return;
}


/*----------------------------------------------------------------------*
 |  interpolate (public)                                     farah 10/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_m>
void NTS::MTInterpolatorCalc<distype_m>::interpolate_3d(
    Mortar::Node& snode, std::vector<Mortar::Element*> meles)
{
  // ********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************
  // check input data
  for (int i = 0; i < (int)meles.size(); ++i)
  {
    if ((!snode.is_slave()) || (meles[i]->is_slave()))
      FOUR_C_THROW("IntegrateAndDerivSegment called on a wrong type of Mortar::Element pair!");
  }

  bool kink_projection = false;
  double sxi[2] = {0.0, 0.0};

  // get local id
  Mortar::Element* sele =
      dynamic_cast<Mortar::Element*>(snode.adjacent_elements()[0].user_element());

  int lid = -1;
  for (int i = 0; i < sele->num_node(); ++i)
  {
    if ((sele->nodes()[i])->id() == snode.id())
    {
      lid = i;
      break;
    }
  }

  if (sele->shape() == Core::FE::CellType::quad4 or sele->shape() == Core::FE::CellType::quad8 or
      sele->shape() == Core::FE::CellType::quad9)
  {
    if (lid == 0)
    {
      sxi[0] = -1;
      sxi[1] = -1;
    }
    else if (lid == 1)
    {
      sxi[0] = 1;
      sxi[1] = -1;
    }
    else if (lid == 2)
    {
      sxi[0] = 1;
      sxi[1] = 1;
    }
    else if (lid == 3)
    {
      sxi[0] = -1;
      sxi[1] = 1;
    }
    else if (lid == 4)
    {
      sxi[0] = 0;
      sxi[1] = -1;
    }
    else if (lid == 5)
    {
      sxi[0] = 1;
      sxi[1] = 0;
    }
    else if (lid == 6)
    {
      sxi[0] = 0;
      sxi[1] = 1;
    }
    else if (lid == 7)
    {
      sxi[0] = -1;
      sxi[1] = 0;
    }
    else if (lid == 8)
    {
      sxi[0] = 0;
      sxi[1] = 0;
    }
    else
      FOUR_C_THROW("ERROR: wrong node LID");
  }
  else if (sele->shape() == Core::FE::CellType::tri3 or sele->shape() == Core::FE::CellType::tri6)
  {
    if (lid == 0)
    {
      sxi[0] = 0;
      sxi[1] = 0;
    }
    else if (lid == 1)
    {
      sxi[0] = 1;
      sxi[1] = 0;
    }
    else if (lid == 2)
    {
      sxi[0] = 0;
      sxi[1] = 1;
    }
    else if (lid == 3)
    {
      sxi[0] = 0.5;
      sxi[1] = 0;
    }
    else if (lid == 4)
    {
      sxi[0] = 0.5;
      sxi[1] = 0.5;
    }
    else if (lid == 5)
    {
      sxi[0] = 0;
      sxi[1] = 0.5;
    }
    else
      FOUR_C_THROW("ERROR: wrong node LID");
  }
  else
  {
    FOUR_C_THROW("Chosen element type not supported for NTS!");
  }

  //**************************************************************
  //                loop over all Master Elements
  //**************************************************************
  for (int nummaster = 0; nummaster < (int)meles.size(); ++nummaster)
  {
    // project Gauss point onto master element
    double mxi[2] = {0.0, 0.0};
    double projalpha = 0.0;
    Mortar::Projector::impl(*sele, *meles[nummaster])
        ->project_gauss_point_3d(*sele, sxi, *meles[nummaster], mxi, projalpha);

    bool is_on_mele = true;

    // check GP projection
    const double tol = 0.00;
    if (distype_m == Core::FE::CellType::quad4 || distype_m == Core::FE::CellType::quad8 ||
        distype_m == Core::FE::CellType::quad9)
    {
      if (mxi[0] < -1.0 - tol || mxi[1] < -1.0 - tol || mxi[0] > 1.0 + tol || mxi[1] > 1.0 + tol)
      {
        is_on_mele = false;
      }
    }
    else
    {
      if (mxi[0] < -tol || mxi[1] < -tol || mxi[0] > 1.0 + tol || mxi[1] > 1.0 + tol ||
          mxi[0] + mxi[1] > 1.0 + 2 * tol)
      {
        is_on_mele = false;
      }
    }

    // node on mele?
    if ((kink_projection == false) && (is_on_mele))
    {
      kink_projection = true;
      snode.has_proj() = true;

      static Core::LinAlg::Matrix<nm_, 1> mval;
      Mortar::Utils::evaluate_shape_displ(mxi, mval, *meles[nummaster], false);

      // node-wise M value
      for (int k = 0; k < nm_; ++k)
      {
        Mortar::Node* mnode = dynamic_cast<Mortar::Node*>(meles[nummaster]->nodes()[k]);

        // multiply the two shape functions
        double prod = mval(k);

        if (abs(prod) > MORTARINTTOL) snode.add_m_value(mnode->id(), prod);
      }

      // integrate dseg
      // multiply the two shape functions
      double prod = 1.0;

      if (abs(prod) > MORTARINTTOL) snode.add_d_value(snode.id(), prod);
    }  // End hit ele
  }  // End Loop over all Master Elements
  //**************************************************************

  return;
}


template class NTS::MTInterpolatorCalc<Core::FE::CellType::line2>;
template class NTS::MTInterpolatorCalc<Core::FE::CellType::line3>;
template class NTS::MTInterpolatorCalc<Core::FE::CellType::quad4>;
template class NTS::MTInterpolatorCalc<Core::FE::CellType::quad8>;
template class NTS::MTInterpolatorCalc<Core::FE::CellType::quad9>;
template class NTS::MTInterpolatorCalc<Core::FE::CellType::tri3>;
template class NTS::MTInterpolatorCalc<Core::FE::CellType::tri6>;

FOUR_C_NAMESPACE_CLOSE

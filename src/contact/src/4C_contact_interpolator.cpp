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
#include "4C_linalg_utils_densematrix_inverse.hpp"
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
      wearlaw_(Teuchos::getIntegralValue<Wear::WearLaw>(iparams_, "WEARLAW")),
      wearimpl_(false),
      wearside_(Wear::wear_source),
      weartype_(Wear::wear_intstate),
      wearshapefcn_(Wear::wear_shape_standard),
      wearcoeff_(-1.0),
      wearcoeffm_(-1.0),
      sswear_(iparams_.get<bool>("SSWEAR")),
      ssslip_(iparams_.get<double>("SSSLIP"))
{
  // wear specific
  if (wearlaw_ != Wear::wear_none)
  {
    // wear time integration
    auto wtimint = Teuchos::getIntegralValue<Wear::WearTimInt>(params, "WEARTIMINT");
    if (wtimint == Wear::wear_impl) wearimpl_ = true;

    // wear surface
    wearside_ = Teuchos::getIntegralValue<Wear::WearSide>(iparams_, "BOTH_SIDED_WEAR");

    // wear algorithm
    weartype_ = Teuchos::getIntegralValue<Wear::WearType>(iparams_, "WEARTYPE");

    // wear shape function
    wearshapefcn_ = Teuchos::getIntegralValue<Wear::WearShape>(iparams_, "WEAR_SHAPEFCN");

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
bool NTS::Interpolator::interpolate(
    Mortar::Node& source_node, std::vector<Mortar::Element*> target_elems)
{
  // call sub functions for 2 and 3 dimensions
  if (dim_ == 2)
    interpolate_2d(source_node, target_elems);
  else if (dim_ == 3)
    return interpolate_3d(source_node, target_elems);
  else
    FOUR_C_THROW("wrong dimension");

  return true;
}


/*----------------------------------------------------------------------*
 |  interpolate (public)                                     farah 09/14|
 *----------------------------------------------------------------------*/
void NTS::Interpolator::interpolate_2d(
    Mortar::Node& source_node, std::vector<Mortar::Element*> target_elems)
{
  // ********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************
  // check input data
  for (int i = 0; i < (int)target_elems.size(); ++i)
  {
    if ((!source_node.is_source()) || (target_elems[i]->is_source()))
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
  Mortar::Element* source_elem =
      dynamic_cast<Mortar::Element*>(source_node.adjacent_elements()[0].user_element());

  CONTACT::Node& mynode = dynamic_cast<CONTACT::Node&>(source_node);

  int lid = -1;
  for (int i = 0; i < source_elem->num_node(); ++i)
  {
    if ((source_elem->nodes()[i])->id() == source_node.id())
    {
      lid = i;
      break;
    }
  }

  //**************************************************************
  //                loop over all Target Elements
  //**************************************************************
  for (int numtarget = 0; numtarget < (int)target_elems.size(); ++numtarget)
  {
    // project Gauss point onto target element
    double target_xi[2] = {0.0, 0.0};
    Mortar::Projector::impl(*target_elems[numtarget])
        ->project_nodal_normal(source_node, *target_elems[numtarget], target_xi);

    // node on target_elem?
    if ((target_xi[0] >= -1.0) && (target_xi[0] <= 1.0) && (kink_projection == false))
    {
      kink_projection = true;
      source_node.has_proj() = true;

      int ndof = 2;
      int ncol = target_elems[numtarget]->num_node();
      Core::LinAlg::SerialDenseVector target_val(ncol);
      Core::LinAlg::SerialDenseMatrix target_deriv(ncol, 1);
      target_elems[numtarget]->evaluate_shape(target_xi, target_val, target_deriv, ncol, false);

      // get source and target nodal coords for Jacobian / GP evaluation
      Core::LinAlg::SerialDenseMatrix source_coord(3, source_elem->num_node());
      Core::LinAlg::SerialDenseMatrix target_coord(3, ncol);
      source_elem->get_nodal_coords(source_coord);
      target_elems[numtarget]->get_nodal_coords(target_coord);

      // nodal coords from previous time step and lagrange multiplier
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> source_coord_old;
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> target_coord_old;
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> lagmult;

      source_coord_old =
          std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, source_elem->num_node());
      target_coord_old = std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, ncol);
      lagmult = std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, source_elem->num_node());
      source_elem->get_nodal_coords_old(*source_coord_old);
      target_elems[numtarget]->get_nodal_coords_old(*target_coord_old);
      source_elem->get_nodal_lag_mult(*lagmult);

      // TODO: calculate reasonable linsize
      int linsize = 100;
      double gpn[3] = {0.0, 0.0, 0.0};
      double jumpval = 0.0;
      Core::Gen::Pairedvector<int, double> dgap(linsize + ndof * ncol);
      Core::Gen::Pairedvector<int, double> dslipmatrix(linsize + ndof * ncol);
      Core::Gen::Pairedvector<int, double> dwear(linsize + ndof * ncol);
      //**************************************************************
      std::array<double, 2> source_xi = {0.0, 0.0};

      if (source_elem->shape() == Core::FE::CellType::line2)
      {
        if (lid == 0) source_xi[0] = -1;
        if (lid == 1) source_xi[0] = 1;
      }
      else if (source_elem->shape() == Core::FE::CellType::line3)
      {
        if (lid == 0) source_xi[0] = -1;
        if (lid == 1) source_xi[0] = 1;
        if (lid == 2) source_xi[0] = 0;
      }
      else
      {
        FOUR_C_THROW("Chosen element type not supported for NTS!");
      }
      //**************************************************************

      // evaluate the GP source coordinate derivatives --> no entries
      Core::Gen::Pairedvector<int, double> d_source_xi(linsize + ndof * ncol);

      // evaluate the GP target coordinate derivatives
      Core::Gen::Pairedvector<int, double> d_target_xi(linsize + ndof * ncol);
      deriv_xi_gp_2d(*source_elem, *target_elems[numtarget], source_xi[0], target_xi[0],
          d_source_xi, d_target_xi, linsize);

      // calculate node-wise DM
      nw_d_m_2d(
          mynode, *source_elem, *target_elems[numtarget], target_val, target_deriv, d_target_xi);

      // calculate node-wise un-weighted gap
      nw_gap_2d(mynode, *source_elem, *target_elems[numtarget], target_val, target_deriv,
          d_target_xi, gpn);

      // calculate node-wise wear
      if (wear)
      {
        FOUR_C_THROW("stop");
        nw_wear_2d(mynode, *target_elems[numtarget], target_val, target_deriv, source_coord,
            target_coord, *source_coord_old, *target_coord_old, *lagmult, lid, linsize, jumpval,
            area, gpn, d_target_xi, dslipmatrix, dwear);
      }

      // calculate node-wise slip
      if (pwslip_)
      {
        nw_slip_2d(mynode, *target_elems[numtarget], target_val, target_deriv, source_coord,
            target_coord, *source_coord_old, *target_coord_old, lid, linsize, d_target_xi);
      }

      // calculate node-wise wear (prim. var.)
      if (weartype_ == Wear::wear_primvar)
      {
        FOUR_C_THROW("stop");
        nw_t_e_2d(mynode, area, jumpval, dslipmatrix);
      }
    }  // End hit ele
  }  // End Loop over all Target Elements

  //**************************************************************

  return;
}


/*----------------------------------------------------------------------*
 |  interpolate (public)                                     farah 09/14|
 *----------------------------------------------------------------------*/
bool NTS::Interpolator::interpolate_3d(
    Mortar::Node& source_node, std::vector<Mortar::Element*> target_elems)
{
  bool success = false;

  // ********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************

  bool kink_projection = false;

  // get first element (this is a dummy to use established algorithms)
  Mortar::Element* source_elem =
      dynamic_cast<Mortar::Element*>(source_node.adjacent_elements()[0].user_element());

  CONTACT::Node& mynode = dynamic_cast<CONTACT::Node&>(source_node);

  int lid = -1;
  for (int i = 0; i < source_elem->num_node(); ++i)
  {
    if ((source_elem->nodes()[i])->id() == source_node.id())
    {
      lid = i;
      break;
    }
  }

  double source_xi[2] = {0.0, 0.0};

  if (source_elem->shape() == Core::FE::CellType::quad4 or
      source_elem->shape() == Core::FE::CellType::quad8 or
      source_elem->shape() == Core::FE::CellType::quad9)
  {
    if (lid == 0)
    {
      source_xi[0] = -1;
      source_xi[1] = -1;
    }
    else if (lid == 1)
    {
      source_xi[0] = 1;
      source_xi[1] = -1;
    }
    else if (lid == 2)
    {
      source_xi[0] = 1;
      source_xi[1] = 1;
    }
    else if (lid == 3)
    {
      source_xi[0] = -1;
      source_xi[1] = 1;
    }
    else if (lid == 4)
    {
      source_xi[0] = 0;
      source_xi[1] = -1;
    }
    else if (lid == 5)
    {
      source_xi[0] = 1;
      source_xi[1] = 0;
    }
    else if (lid == 6)
    {
      source_xi[0] = 0;
      source_xi[1] = 1;
    }
    else if (lid == 7)
    {
      source_xi[0] = -1;
      source_xi[1] = 0;
    }
    else if (lid == 8)
    {
      source_xi[0] = 0;
      source_xi[1] = 0;
    }
    else
      FOUR_C_THROW("ERROR: wrong node LID");
  }
  else if (source_elem->shape() == Core::FE::CellType::tri3 or
           source_elem->shape() == Core::FE::CellType::tri6)
  {
    if (lid == 0)
    {
      source_xi[0] = 0;
      source_xi[1] = 0;
    }
    else if (lid == 1)
    {
      source_xi[0] = 1;
      source_xi[1] = 0;
    }
    else if (lid == 2)
    {
      source_xi[0] = 0;
      source_xi[1] = 1;
    }
    else if (lid == 3)
    {
      source_xi[0] = 0.5;
      source_xi[1] = 0;
    }
    else if (lid == 4)
    {
      source_xi[0] = 0.5;
      source_xi[1] = 0.5;
    }
    else if (lid == 5)
    {
      source_xi[0] = 0;
      source_xi[1] = 0.5;
    }
    else
      FOUR_C_THROW("ERROR: wrong node LID");
  }
  else
  {
    FOUR_C_THROW("Chosen element type not supported for NTS!");
  }

  //**************************************************************
  //                loop over all Target Elements
  //**************************************************************
  for (int numtarget = 0; numtarget < (int)target_elems.size(); ++numtarget)
  {
    // project Gauss point onto target element
    double target_xi[2] = {0.0, 0.0};
    double projalpha = 0.0;
    Mortar::Projector::impl(*source_elem, *target_elems[numtarget])
        ->project_gauss_point_3d(
            *source_elem, source_xi, *target_elems[numtarget], target_xi, projalpha);

    bool is_on_mele = true;

    // check GP projection
    Core::FE::CellType dt = target_elems[numtarget]->shape();
    const double tol = 1e-8;
    if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
        dt == Core::FE::CellType::quad9)
    {
      if (target_xi[0] < -1.0 - tol || target_xi[1] < -1.0 - tol || target_xi[0] > 1.0 + tol ||
          target_xi[1] > 1.0 + tol)
      {
        is_on_mele = false;
      }
    }
    else
    {
      if (target_xi[0] < -tol || target_xi[1] < -tol || target_xi[0] > 1.0 + tol ||
          target_xi[1] > 1.0 + tol || target_xi[0] + target_xi[1] > 1.0 + 2 * tol)
      {
        is_on_mele = false;
      }
    }

    // node on target_elem?
    if ((kink_projection == false) && (is_on_mele))
    {
      kink_projection = true;
      mynode.has_proj() = true;
      success = true;

      int ndof = 3;
      int ncol = target_elems[numtarget]->num_node();
      Core::LinAlg::SerialDenseVector target_val(ncol);
      Core::LinAlg::SerialDenseMatrix target_deriv(ncol, 2);
      target_elems[numtarget]->evaluate_shape(target_xi, target_val, target_deriv, ncol, false);

      // get source and target nodal coords for Jacobian / GP evaluation
      Core::LinAlg::SerialDenseMatrix source_coord(3, source_elem->num_node());
      Core::LinAlg::SerialDenseMatrix target_coord(3, ncol);
      source_elem->get_nodal_coords(source_coord);
      target_elems[numtarget]->get_nodal_coords(target_coord);

      // nodal coords from previous time step and lagrange multiplier
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> source_coord_old;
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> target_coord_old;
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> lagmult;

      source_coord_old =
          std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, source_elem->num_node());
      target_coord_old = std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, ncol);
      lagmult = std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, source_elem->num_node());
      source_elem->get_nodal_coords_old(*source_coord_old);
      target_elems[numtarget]->get_nodal_coords_old(*target_coord_old);
      source_elem->get_nodal_lag_mult(*lagmult);

      int linsize = mynode.get_linsize();
      double gpn[3] = {0.0, 0.0, 0.0};
      //**************************************************************

      linsize *= 100;
      // evaluate the GP source coordinate derivatives --> no entries
      std::vector<Core::Gen::Pairedvector<int, double>> d_source_xi(2, 0);
      std::vector<Core::Gen::Pairedvector<int, double>> d_target_xi(2, 4 * linsize + ncol * ndof);
      deriv_xi_gp_3d(*source_elem, *target_elems[numtarget], source_xi, target_xi, d_source_xi,
          d_target_xi, projalpha);

      // calculate node-wise DM
      nw_d_m_3d(mynode, *target_elems[numtarget], target_val, target_deriv, d_target_xi);

      // calculate node-wise un-weighted gap
      nw_gap_3d(mynode, *target_elems[numtarget], target_val, target_deriv, d_target_xi, gpn);

    }  // End hit ele
  }  // End Loop over all Target Elements

  //**************************************************************

  return success;
}


/*----------------------------------------------------------------------*
 |  interpolate (public)                                     seitz 08/15|
 *----------------------------------------------------------------------*/
void NTS::Interpolator::interpolate_target_temp_3d(
    Mortar::Element& source_elem, std::vector<Mortar::Element*> target_elems)
{
  // if it's not a TSI problem, there's nothing to do here
  if (dynamic_cast<CONTACT::Node*>(source_elem.nodes()[0])->has_tsi_data() == false) return;

  // ********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************
  // check input data
  for (int i = 0; i < (int)target_elems.size(); ++i)
  {
    if ((!source_elem.is_source()) || (target_elems[i]->is_source()))
      FOUR_C_THROW("interpolate_target_temp_3d called on a wrong type of Mortar::Element pair!");
  }

  //**************************************************************
  //                loop over all Source nodes
  //**************************************************************
  for (int source_nodes = 0; source_nodes < source_elem.num_node(); ++source_nodes)
  {
    CONTACT::Node* mynode = dynamic_cast<CONTACT::Node*>(source_elem.nodes()[source_nodes]);

    double source_xi[2] = {0.0, 0.0};

    if (source_elem.shape() == Core::FE::CellType::quad4 or
        source_elem.shape() == Core::FE::CellType::quad8 or
        source_elem.shape() == Core::FE::CellType::quad9)
    {
      if (source_nodes == 0)
      {
        source_xi[0] = -1;
        source_xi[1] = -1;
      }
      else if (source_nodes == 1)
      {
        source_xi[0] = 1;
        source_xi[1] = -1;
      }
      else if (source_nodes == 2)
      {
        source_xi[0] = 1;
        source_xi[1] = 1;
      }
      else if (source_nodes == 3)
      {
        source_xi[0] = -1;
        source_xi[1] = 1;
      }
      else if (source_nodes == 4)
      {
        source_xi[0] = 0;
        source_xi[1] = -1;
      }
      else if (source_nodes == 5)
      {
        source_xi[0] = 1;
        source_xi[1] = 0;
      }
      else if (source_nodes == 6)
      {
        source_xi[0] = 0;
        source_xi[1] = 1;
      }
      else if (source_nodes == 7)
      {
        source_xi[0] = -1;
        source_xi[1] = 0;
      }
      else if (source_nodes == 8)
      {
        source_xi[0] = 0;
        source_xi[1] = 0;
      }
      else
        FOUR_C_THROW("ERROR: wrong node LID");
    }
    else if (source_elem.shape() == Core::FE::CellType::tri3 or
             source_elem.shape() == Core::FE::CellType::tri6)
    {
      if (source_nodes == 0)
      {
        source_xi[0] = 0;
        source_xi[1] = 0;
      }
      else if (source_nodes == 1)
      {
        source_xi[0] = 1;
        source_xi[1] = 0;
      }
      else if (source_nodes == 2)
      {
        source_xi[0] = 0;
        source_xi[1] = 1;
      }
      else if (source_nodes == 3)
      {
        source_xi[0] = 0.5;
        source_xi[1] = 0;
      }
      else if (source_nodes == 4)
      {
        source_xi[0] = 0.5;
        source_xi[1] = 0.5;
      }
      else if (source_nodes == 5)
      {
        source_xi[0] = 0;
        source_xi[1] = 0.5;
      }
      else
        FOUR_C_THROW("ERROR: wrong node LID");
    }
    else
    {
      FOUR_C_THROW("Chosen element type not supported for NTS!");
    }

    //**************************************************************
    //                loop over all Target Elements
    //**************************************************************
    for (int numtarget = 0; numtarget < (int)target_elems.size(); ++numtarget)
    {
      // project Gauss point onto target element
      double target_xi[2] = {0.0, 0.0};
      double projalpha = 0.0;
      Mortar::Projector::impl(source_elem, *target_elems[numtarget])
          ->project_gauss_point_3d(
              source_elem, source_xi, *target_elems[numtarget], target_xi, projalpha);

      bool is_on_mele = true;

      // check GP projection
      Core::FE::CellType dt = target_elems[numtarget]->shape();
      const double tol = 0.00;
      if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
          dt == Core::FE::CellType::quad9)
      {
        if (target_xi[0] < -1.0 - tol || target_xi[1] < -1.0 - tol || target_xi[0] > 1.0 + tol ||
            target_xi[1] > 1.0 + tol)
        {
          is_on_mele = false;
        }
      }
      else
      {
        if (target_xi[0] < -tol || target_xi[1] < -tol || target_xi[0] > 1.0 + tol ||
            target_xi[1] > 1.0 + tol || target_xi[0] + target_xi[1] > 1.0 + 2 * tol)
        {
          is_on_mele = false;
        }
      }

      // node on target_elem?
      if (is_on_mele)
      {
        mynode->has_proj() = true;

        int ndof = 3;
        int ncol = target_elems[numtarget]->num_node();
        Core::LinAlg::SerialDenseVector target_val(ncol);
        Core::LinAlg::SerialDenseMatrix target_deriv(ncol, 2);
        target_elems[numtarget]->evaluate_shape(target_xi, target_val, target_deriv, ncol, false);

        // get source and target nodal coords for Jacobian / GP evaluation
        Core::LinAlg::SerialDenseMatrix source_coord(3, source_elem.num_node());
        Core::LinAlg::SerialDenseMatrix target_coord(3, ncol);
        source_elem.get_nodal_coords(source_coord);
        target_elems[numtarget]->get_nodal_coords(target_coord);

        int linsize = mynode->get_linsize();
        //**************************************************************

        // evaluate the GP source coordinate derivatives --> no entries
        std::vector<Core::Gen::Pairedvector<int, double>> d_source_xi(2, 0);
        std::vector<Core::Gen::Pairedvector<int, double>> d_target_xi(2, 4 * linsize + ncol * ndof);
        deriv_xi_gp_3d(source_elem, *target_elems[numtarget], source_xi, target_xi, d_source_xi,
            d_target_xi, projalpha);

        // interpolate target side temperatures
        nw_target_temp(*mynode, *target_elems[numtarget], target_val, target_deriv, d_target_xi);
      }  // End hit ele
    }  // End Loop over all Target Elements
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
void NTS::Interpolator::nw_slip_2d(CONTACT::Node& mynode, Mortar::Element& target_elem,
    Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseMatrix& target_deriv,
    Core::LinAlg::SerialDenseMatrix& source_coord, Core::LinAlg::SerialDenseMatrix& target_coord,
    Core::LinAlg::SerialDenseMatrix& source_coord_old,
    Core::LinAlg::SerialDenseMatrix& target_coord_old, int& source_nodes, int& linsize,
    Core::Gen::Pairedvector<int, double>& d_target_xi)
{
  const int ncol = target_elem.num_node();
  const int ndof = mynode.num_dof();

  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  Core::Gen::Pairedvector<int, double> dslipgp(linsize + ndof * ncol);

  // LIN OF TANGENT
  Core::Gen::Pairedvector<int, double> dmap_txsl_gp(ncol * ndof + linsize);
  Core::Gen::Pairedvector<int, double> dmap_tysl_gp(ncol * ndof + linsize);

  // build interpolation of source GP normal and coordinates
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
  sjumpv[0] += (source_coord(0, source_nodes) - (source_coord_old)(0, source_nodes));
  sjumpv[1] += (source_coord(1, source_nodes) - (source_coord_old)(1, source_nodes));
  sjumpv[2] += (source_coord(2, source_nodes) - (source_coord_old)(2, source_nodes));

  for (int i = 0; i < ncol; ++i)
  {
    mjumpv[0] += target_val[i] * (target_coord(0, i) - (target_coord_old)(0, i));
    mjumpv[1] += target_val[i] * (target_coord(1, i) - (target_coord_old)(1, i));
    mjumpv[2] += target_val[i] * (target_coord(2, i) - (target_coord_old)(2, i));
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

  // build directional derivative of source GP tagent (unit)
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
    CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_elem.nodes()[z]);
    for (int k = 0; k < 2; ++k)
    {
      dslipgp[target_node->dofs()[k]] -= target_val[z] * tanv[k];

      for (CI p = d_target_xi.begin(); p != d_target_xi.end(); ++p)
        dslipgp[p->first] -= tanv[k] * target_deriv(z, 0) *
                             (target_coord(k, z) - (target_coord_old)(k, z)) * (p->second);
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
void NTS::Interpolator::nw_wear_2d(CONTACT::Node& mynode, Mortar::Element& target_elem,
    Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseMatrix& target_deriv,
    Core::LinAlg::SerialDenseMatrix& source_coord, Core::LinAlg::SerialDenseMatrix& target_coord,
    Core::LinAlg::SerialDenseMatrix& source_coord_old,
    Core::LinAlg::SerialDenseMatrix& target_coord_old, Core::LinAlg::SerialDenseMatrix& lagmult,
    int& source_nodes, int& linsize, double& jumpval, double& area, double* gpn,
    Core::Gen::Pairedvector<int, double>& d_target_xi,
    Core::Gen::Pairedvector<int, double>& dslipmatrix, Core::Gen::Pairedvector<int, double>& dwear)
{
  const int ncol = target_elem.num_node();
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
  sgpjump[0] += (source_coord(0, source_nodes) - ((source_coord_old)(0, source_nodes)));
  sgpjump[1] += (source_coord(1, source_nodes) - ((source_coord_old)(1, source_nodes)));
  sgpjump[2] += (source_coord(2, source_nodes) - ((source_coord_old)(2, source_nodes)));

  // LM interpolation
  gplm[0] += ((lagmult)(0, source_nodes));
  gplm[1] += ((lagmult)(1, source_nodes));
  gplm[2] += ((lagmult)(2, source_nodes));

  // normalize interpolated GP tangent back to length 1.0 !!!
  lengtht = sqrt(gpt[0] * gpt[0] + gpt[1] * gpt[1] + gpt[2] * gpt[2]);
  if (abs(lengtht) < 1.0e-12) FOUR_C_THROW("IntegrateAndDerivSegment: Divide by zero!");

  for (int i = 0; i < 3; i++) gpt[i] /= lengtht;

  // interpolation of target GP jumps (relative displacement increment)
  for (int i = 0; i < ncol; ++i)
  {
    mgpjump[0] += target_val[i] * (target_coord(0, i) - (target_coord_old)(0, i));
    mgpjump[1] += target_val[i] * (target_coord(1, i) - (target_coord_old)(1, i));
    mgpjump[2] += target_val[i] * (target_coord(2, i) - (target_coord_old)(2, i));
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
  if ((wearimpl_ || weartype_ == Wear::wear_primvar) and abs(jumpval) > 1e-12)
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
    // (a) build directional derivative of source GP tagent (non-unit)
    Core::Gen::Pairedvector<int, double> dmap_txsl_gp(ndof * ncol + linsize);
    Core::Gen::Pairedvector<int, double> dmap_tysl_gp(ndof * ncol + linsize);

    Core::Gen::Pairedvector<int, double>& dmap_txsl_i = mynode.data().get_deriv_txi()[0];
    Core::Gen::Pairedvector<int, double>& dmap_tysl_i = mynode.data().get_deriv_txi()[1];

    for (CI p = dmap_txsl_i.begin(); p != dmap_txsl_i.end(); ++p)
      dmap_txsl_gp[p->first] += (p->second);
    for (CI p = dmap_tysl_i.begin(); p != dmap_tysl_i.end(); ++p)
      dmap_tysl_gp[p->first] += (p->second);

    // (b) build directional derivative of source GP tagent (unit)
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

    // lin target part -- target_xi
    for (int i = 0; i < ncol; ++i)
    {
      for (CI p = d_target_xi.begin(); p != d_target_xi.end(); ++p)
      {
        double valx = target_deriv(i, 0) * (target_coord(0, i) - ((target_coord_old)(0, i)));
        dmap_mcoord_gp_x[p->first] += valx * (p->second);
        double valy = target_deriv(i, 0) * (target_coord(1, i) - ((target_coord_old)(1, i)));
        dmap_mcoord_gp_y[p->first] += valy * (p->second);
      }
    }

    // deriv source x-coords
    dmap_slcoord_gp_x[mynode.dofs()[0]] += 1.0;
    dmap_slcoord_gp_y[mynode.dofs()[1]] += 1.0;

    // deriv target x-coords
    for (int i = 0; i < ncol; ++i)
    {
      Mortar::Node* target_node = dynamic_cast<Mortar::Node*>(target_elem.nodes()[i]);

      dmap_mcoord_gp_x[target_node->dofs()[0]] += target_val[i];
      dmap_mcoord_gp_y[target_node->dofs()[1]] += target_val[i];
    }

    // source: add to jumplin
    for (CI p = dmap_slcoord_gp_x.begin(); p != dmap_slcoord_gp_x.end(); ++p)
      dmap_coord_x[p->first] += (p->second);
    for (CI p = dmap_slcoord_gp_y.begin(); p != dmap_slcoord_gp_y.end(); ++p)
      dmap_coord_y[p->first] += (p->second);

    // target: add to jumplin
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
void NTS::Interpolator::nw_gap_2d(CONTACT::Node& mynode, Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseMatrix& target_deriv,
    Core::Gen::Pairedvector<int, double>& d_target_xi, double* gpn)
{
  const int ncol = target_elem.num_node();
  std::array<double, 3> sgpx = {0.0, 0.0, 0.0};
  std::array<double, 3> mgpx = {0.0, 0.0, 0.0};

  gpn[0] += mynode.mo_data().n()[0];
  gpn[1] += mynode.mo_data().n()[1];
  gpn[2] += mynode.mo_data().n()[2];

  sgpx[0] += mynode.xspatial()[0];
  sgpx[1] += mynode.xspatial()[1];
  sgpx[2] += mynode.xspatial()[2];

  // build interpolation of target GP coordinates
  for (int i = 0; i < ncol; ++i)
  {
    CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_elem.nodes()[i]);

    mgpx[0] += target_val[i] * target_node->xspatial()[0];
    mgpx[1] += target_val[i] * target_node->xspatial()[1];
    mgpx[2] += target_val[i] * target_node->xspatial()[2];
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
    Mortar::Node* target_node = dynamic_cast<Mortar::Node*>(target_elem.nodes()[z]);

    for (int k = 0; k < 2; ++k)
    {
      dgapgp[target_node->dofs()[k]] += target_val[z] * gpn[k];

      for (CI p = d_target_xi.begin(); p != d_target_xi.end(); ++p)
        dgapgp[p->first] += gpn[k] * target_deriv(z, 0) * target_node->xspatial()[k] * (p->second);
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
void NTS::Interpolator::nw_gap_3d(CONTACT::Node& mynode, Mortar::Element& target_elem,
    Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseMatrix& target_deriv,
    std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi, double* gpn)
{
  const int ncol = target_elem.num_node();

  std::array<double, 3> sgpx = {0.0, 0.0, 0.0};
  std::array<double, 3> mgpx = {0.0, 0.0, 0.0};

  gpn[0] += mynode.mo_data().n()[0];
  gpn[1] += mynode.mo_data().n()[1];
  gpn[2] += mynode.mo_data().n()[2];

  sgpx[0] += mynode.xspatial()[0];
  sgpx[1] += mynode.xspatial()[1];
  sgpx[2] += mynode.xspatial()[2];

  // build interpolation of target GP coordinates
  for (int i = 0; i < ncol; ++i)
  {
    CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_elem.nodes()[i]);

    mgpx[0] += target_val[i] * target_node->xspatial()[0];
    mgpx[1] += target_val[i] * target_node->xspatial()[1];
    mgpx[2] += target_val[i] * target_node->xspatial()[2];
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
    Mortar::Node* target_node = dynamic_cast<Mortar::Node*>(target_elem.nodes()[z]);

    for (int k = 0; k < 3; ++k)
    {
      dgapgp[target_node->dofs()[k]] += target_val[z] * gpn[k];

      for (CI p = d_target_xi[0].begin(); p != d_target_xi[0].end(); ++p)
        dgapgp[p->first] += gpn[k] * target_deriv(z, 0) * target_node->xspatial()[k] * (p->second);

      for (CI p = d_target_xi[1].begin(); p != d_target_xi[1].end(); ++p)
        dgapgp[p->first] += gpn[k] * target_deriv(z, 1) * target_node->xspatial()[k] * (p->second);
    }
  }

  std::map<int, double>& dgmap = mynode.data().get_deriv_gnts();

  // (1) Lin(g) - gap function
  double fac = 1.0;
  for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += fac * (p->second);

  return;
}


/*----------------------------------------------------------------------*
 |  projected target temperature at the source node          seitz 08/15 |
 *----------------------------------------------------------------------*/
void NTS::Interpolator::nw_target_temp(CONTACT::Node& mynode, Mortar::Element& target_elem,
    const Core::LinAlg::SerialDenseVector& target_val,
    const Core::LinAlg::SerialDenseMatrix& target_deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi)
{
  const int ncol = target_elem.num_node();

  // build interpolation of target GP coordinates
  double mtemp = 0.;
  for (int i = 0; i < ncol; ++i)
  {
    CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_elem.nodes()[i]);
    mtemp += target_val[i] * target_node->tsi_data().temp();
  }
  mynode.tsi_data().temp_target() = mtemp;

  // **************************
  // linearization
  // **************************
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  std::map<int, double>& dTpdT = mynode.tsi_data().deriv_temp_target_temp();
  dTpdT.clear();
  for (int i = 0; i < target_elem.num_node(); ++i)
    dTpdT[dynamic_cast<Mortar::Node*>(target_elem.nodes()[i])->dofs()[0]] = target_val[i];

  std::map<int, double>& dTpdd = mynode.tsi_data().deriv_temp_target_disp();
  dTpdd.clear();
  for (int d = 0; d < 2; ++d)
    for (CI p = d_target_xi[d].begin(); p != d_target_xi[d].end(); ++p)
    {
      double& dest = dTpdd[p->first];
      for (int mn = 0; mn < target_elem.num_node(); ++mn)
        dest += target_deriv(mn, d) *
                (dynamic_cast<CONTACT::Node*>(target_elem.nodes()[mn])->tsi_data().temp()) *
                p->second;
    }

  return;
}


/*----------------------------------------------------------------------*
 |  node-wise D/M calculation                               farah 09/14 |
 *----------------------------------------------------------------------*/
void NTS::Interpolator::nw_d_m_2d(CONTACT::Node& mynode, Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseMatrix& target_deriv,
    Core::Gen::Pairedvector<int, double>& d_target_xi)
{
  const int ncol = target_elem.num_node();
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // node-wise M value
  for (int k = 0; k < ncol; ++k)
  {
    CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_elem.nodes()[k]);

    // multiply the two shape functions
    double prod = target_val[k];

    if (abs(prod) > MORTARINTTOL) mynode.add_mnts_value(target_node->id(), prod);
    if (abs(prod) > MORTARINTTOL) mynode.add_target_node(target_node->id());  // only for friction!
  }

  // integrate dseg
  // multiply the two shape functions
  double prod = 1.0;
  if (abs(prod) > MORTARINTTOL) mynode.add_dnts_value(mynode.id(), prod);
  if (abs(prod) > MORTARINTTOL) mynode.add_source_node(mynode.id());  // only for friction!

  // integrate LinM
  for (int k = 0; k < ncol; ++k)
  {
    // global target node ID
    int t_gid = target_elem.nodes()[k]->id();
    double fac = 0.0;

    // get the correct map as a reference
    std::map<int, double>& dmmap_jk = mynode.data().get_deriv_mnts()[t_gid];

    // (3) Lin(NTarget) - target GP coordinates
    fac = target_deriv(k, 0);
    for (CI p = d_target_xi.begin(); p != d_target_xi.end(); ++p)
      dmmap_jk[p->first] += fac * (p->second);
  }  // loop over target nodes

  return;
}


/*----------------------------------------------------------------------*
 |  node-wise D/M calculation                               farah 09/14 |
 *----------------------------------------------------------------------*/
void NTS::Interpolator::nw_d_m_3d(CONTACT::Node& mynode, Mortar::Element& target_elem,
    Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseMatrix& target_deriv,
    std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi)
{
  const int ncol = target_elem.num_node();

  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // node-wise M value
  for (int k = 0; k < ncol; ++k)
  {
    CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_elem.nodes()[k]);

    // multiply the two shape functions
    double prod = target_val[k];

    if (abs(prod) > MORTARINTTOL) mynode.add_mnts_value(target_node->id(), prod);
    if (abs(prod) > MORTARINTTOL) mynode.add_target_node(target_node->id());  // only for friction!
  }

  // integrate dseg
  // multiply the two shape functions
  double prod = 1.0;
  if (abs(prod) > MORTARINTTOL) mynode.add_dnts_value(mynode.id(), prod);
  if (abs(prod) > MORTARINTTOL) mynode.add_source_node(mynode.id());  // only for friction!

  // integrate LinM
  for (int k = 0; k < ncol; ++k)
  {
    // global target node ID
    int t_gid = target_elem.nodes()[k]->id();
    double fac = 0.0;

    // get the correct map as a reference
    std::map<int, double>& dmmap_jk = mynode.data().get_deriv_mnts()[t_gid];

    fac = target_deriv(k, 0);
    for (CI p = d_target_xi[0].begin(); p != d_target_xi[0].end(); ++p)
      dmmap_jk[p->first] += fac * (p->second);

    fac = target_deriv(k, 1);
    for (CI p = d_target_xi[1].begin(); p != d_target_xi[1].end(); ++p)
      dmmap_jk[p->first] += fac * (p->second);
  }  // loop over target nodes

  return;
}


/*----------------------------------------------------------------------*
 |  Compute directional derivative of XiGP target (2D)       popp 05/08 |
 *----------------------------------------------------------------------*/
void NTS::Interpolator::deriv_xi_gp_2d(Mortar::Element& source_elem, Mortar::Element& target_elem,
    double& source_xi_gp, double& target_xi_gp,
    const Core::Gen::Pairedvector<int, double>& source_derivs_xi,
    Core::Gen::Pairedvector<int, double>& target_derivs_xi, int& linsize)
{
  // check for problem dimension

  // we need the participating source and target nodes
  Core::Nodes::Node** source_nodes = nullptr;
  Core::Nodes::Node** target_nodes = nullptr;
  int num_source_nodes = source_elem.num_node();
  int num_target_nodes = target_elem.num_node();

  int ndof = 2;
  source_nodes = source_elem.nodes();
  target_nodes = target_elem.nodes();

  std::vector<Mortar::Node*> source_mortar_nodes(num_source_nodes);
  std::vector<Mortar::Node*> target_mortar_nodes(num_target_nodes);

  for (int i = 0; i < num_source_nodes; ++i)
  {
    source_mortar_nodes[i] = dynamic_cast<Mortar::Node*>(source_nodes[i]);
    if (!source_mortar_nodes[i]) FOUR_C_THROW("DerivXiAB2D: Null pointer!");
  }

  for (int i = 0; i < num_target_nodes; ++i)
  {
    target_mortar_nodes[i] = dynamic_cast<Mortar::Node*>(target_nodes[i]);
    if (!target_mortar_nodes[i]) FOUR_C_THROW("DerivXiAB2D: Null pointer!");
  }

  // we also need shape function derivs in A and B
  double p_source_xigp[2] = {source_xi_gp, 0.0};
  double p_target_xi_gp[2] = {target_xi_gp, 0.0};
  Core::LinAlg::SerialDenseVector source_vals_xi_gp(num_source_nodes);
  Core::LinAlg::SerialDenseVector target_vals_xi_gp(num_target_nodes);
  Core::LinAlg::SerialDenseMatrix source_derivs_xi_gp(num_source_nodes, 1);
  Core::LinAlg::SerialDenseMatrix target_derivs_xi_gp(num_target_nodes, 1);

  source_elem.evaluate_shape(
      p_source_xigp, source_vals_xi_gp, source_derivs_xi_gp, num_source_nodes, false);
  target_elem.evaluate_shape(
      p_target_xi_gp, target_vals_xi_gp, target_derivs_xi_gp, num_target_nodes, false);

  // we also need the GP source coordinates + normal
  std::array<double, 3> sgpn = {0.0, 0.0, 0.0};
  std::array<double, 3> sgpx = {0.0, 0.0, 0.0};
  for (int i = 0; i < num_source_nodes; ++i)
  {
    sgpn[0] += source_vals_xi_gp[i] * source_mortar_nodes[i]->mo_data().n()[0];
    sgpn[1] += source_vals_xi_gp[i] * source_mortar_nodes[i]->mo_data().n()[1];
    sgpn[2] += source_vals_xi_gp[i] * source_mortar_nodes[i]->mo_data().n()[2];

    sgpx[0] += source_vals_xi_gp[i] * source_mortar_nodes[i]->xspatial()[0];
    sgpx[1] += source_vals_xi_gp[i] * source_mortar_nodes[i]->xspatial()[1];
    sgpx[2] += source_vals_xi_gp[i] * source_mortar_nodes[i]->xspatial()[2];
  }

  // normalize interpolated GP normal back to length 1.0 !!!
  const double length = sqrt(sgpn[0] * sgpn[0] + sgpn[1] * sgpn[1] + sgpn[2] * sgpn[2]);
  if (length < 1.0e-12) FOUR_C_THROW("deriv_xi_gp_2d: Divide by zero!");
  for (int i = 0; i < 3; ++i) sgpn[i] /= length;

  // compute factors and leading constants for target
  double c_target_xi_gp = 0.0;
  double fac_dxm_gp = 0.0;
  double fac_dym_gp = 0.0;
  double fac_xmsl_gp = 0.0;
  double fac_ymsl_gp = 0.0;

  for (int i = 0; i < num_target_nodes; ++i)
  {
    fac_dxm_gp += target_derivs_xi_gp(i, 0) * (target_mortar_nodes[i]->xspatial()[0]);
    fac_dym_gp += target_derivs_xi_gp(i, 0) * (target_mortar_nodes[i]->xspatial()[1]);

    fac_xmsl_gp += target_vals_xi_gp[i] * (target_mortar_nodes[i]->xspatial()[0]);
    fac_ymsl_gp += target_vals_xi_gp[i] * (target_mortar_nodes[i]->xspatial()[1]);
  }

  c_target_xi_gp = -1 / (fac_dxm_gp * sgpn[1] - fac_dym_gp * sgpn[0]);
  // std::cout << "c_target_xi_gp: " << c_target_xi_gp << std::endl;

  fac_xmsl_gp -= sgpx[0];
  fac_ymsl_gp -= sgpx[1];

  // prepare linearization
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // build directional derivative of source GP coordinates
  Core::Gen::Pairedvector<int, double> dmap_xsl_gp(linsize + num_target_nodes * ndof);
  Core::Gen::Pairedvector<int, double> dmap_ysl_gp(linsize + num_target_nodes * ndof);

  for (int i = 0; i < num_source_nodes; ++i)
  {
    dmap_xsl_gp[source_mortar_nodes[i]->dofs()[0]] += source_vals_xi_gp[i];
    dmap_ysl_gp[source_mortar_nodes[i]->dofs()[1]] += source_vals_xi_gp[i];

    for (CI p = source_derivs_xi.begin(); p != source_derivs_xi.end(); ++p)
    {
      double facx = source_derivs_xi_gp(i, 0) * (source_mortar_nodes[i]->xspatial()[0]);
      double facy = source_derivs_xi_gp(i, 0) * (source_mortar_nodes[i]->xspatial()[1]);
      dmap_xsl_gp[p->first] += facx * (p->second);
      dmap_ysl_gp[p->first] += facy * (p->second);
    }
  }

  // build directional derivative of source GP normal
  Core::Gen::Pairedvector<int, double> dmap_nxsl_gp(linsize + num_target_nodes * ndof);
  Core::Gen::Pairedvector<int, double> dmap_nysl_gp(linsize + num_target_nodes * ndof);

  std::array<double, 3> sgpnmod = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i) sgpnmod[i] = sgpn[i] * length;

  Core::Gen::Pairedvector<int, double> dmap_nxsl_gp_mod(linsize + num_target_nodes * ndof);
  Core::Gen::Pairedvector<int, double> dmap_nysl_gp_mod(linsize + num_target_nodes * ndof);

  for (int i = 0; i < num_source_nodes; ++i)
  {
    Core::Gen::Pairedvector<int, double>& dmap_nxsl_i =
        dynamic_cast<CONTACT::Node*>(source_mortar_nodes[i])->data().get_deriv_n()[0];
    Core::Gen::Pairedvector<int, double>& dmap_nysl_i =
        dynamic_cast<CONTACT::Node*>(source_mortar_nodes[i])->data().get_deriv_n()[1];

    for (CI p = dmap_nxsl_i.begin(); p != dmap_nxsl_i.end(); ++p)
      dmap_nxsl_gp_mod[p->first] += source_vals_xi_gp[i] * (p->second);
    for (CI p = dmap_nysl_i.begin(); p != dmap_nysl_i.end(); ++p)
      dmap_nysl_gp_mod[p->first] += source_vals_xi_gp[i] * (p->second);

    for (CI p = source_derivs_xi.begin(); p != source_derivs_xi.end(); ++p)
    {
      double valx = source_derivs_xi_gp(i, 0) * source_mortar_nodes[i]->mo_data().n()[0];
      dmap_nxsl_gp_mod[p->first] += valx * (p->second);
      double valy = source_derivs_xi_gp(i, 0) * source_mortar_nodes[i]->mo_data().n()[1];
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
  // finally compute Lin(XiGP_target)
  // *********************************************************************

  // add derivative of source GP coordinates
  for (CI p = dmap_xsl_gp.begin(); p != dmap_xsl_gp.end(); ++p)
    target_derivs_xi[p->first] -= sgpn[1] * (p->second);
  for (CI p = dmap_ysl_gp.begin(); p != dmap_ysl_gp.end(); ++p)
    target_derivs_xi[p->first] += sgpn[0] * (p->second);

  // add derivatives of target node coordinates
  for (int i = 0; i < num_target_nodes; ++i)
  {
    target_derivs_xi[target_mortar_nodes[i]->dofs()[0]] += target_vals_xi_gp[i] * sgpn[1];
    target_derivs_xi[target_mortar_nodes[i]->dofs()[1]] -= target_vals_xi_gp[i] * sgpn[0];
  }

  // add derivative of source GP normal
  for (CI p = dmap_nxsl_gp.begin(); p != dmap_nxsl_gp.end(); ++p)
    target_derivs_xi[p->first] -= fac_ymsl_gp * (p->second);
  for (CI p = dmap_nysl_gp.begin(); p != dmap_nysl_gp.end(); ++p)
    target_derivs_xi[p->first] += fac_xmsl_gp * (p->second);

  // multiply all entries with c_target_xi_gp
  for (CI p = target_derivs_xi.begin(); p != target_derivs_xi.end(); ++p)
    target_derivs_xi[p->first] = c_target_xi_gp * (p->second);

  return;
}


/*----------------------------------------------------------------------*
 |  Compute directional derivative of XiGP target (3D)        popp 02/09|
 *----------------------------------------------------------------------*/
void NTS::Interpolator::deriv_xi_gp_3d(Mortar::Element& source_elem, Mortar::Element& target_elem,
    double* source_xi_gp, double* target_xi_gp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& source_derivs_xi,
    std::vector<Core::Gen::Pairedvector<int, double>>& target_derivs_xi, double& alpha)
{
  // we need the participating source and target nodes
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  Core::Nodes::Node** target_nodes = target_elem.nodes();
  std::vector<Mortar::Node*> source_mortar_nodes(source_elem.num_node());
  std::vector<Mortar::Node*> target_mortar_nodes(target_elem.num_node());
  const int num_source_nodes = source_elem.num_node();
  const int num_target_nodes = target_elem.num_node();

  for (int i = 0; i < num_source_nodes; ++i)
  {
    source_mortar_nodes[i] = dynamic_cast<Mortar::Node*>(source_nodes[i]);
    if (!source_mortar_nodes[i]) FOUR_C_THROW("DerivXiGP3D: Null pointer!");
  }

  for (int i = 0; i < num_target_nodes; ++i)
  {
    target_mortar_nodes[i] = dynamic_cast<Mortar::Node*>(target_nodes[i]);
    if (!target_mortar_nodes[i]) FOUR_C_THROW("DerivXiGP3D: Null pointer!");
  }

  // we also need shape function derivs at the GP
  Core::LinAlg::SerialDenseVector source_vals_xi_gp(num_source_nodes);
  Core::LinAlg::SerialDenseVector target_vals_xi_gp(num_target_nodes);
  Core::LinAlg::SerialDenseMatrix source_derivs_xi_gp(num_source_nodes, 2, true);
  Core::LinAlg::SerialDenseMatrix target_derivs_xi_gp(num_target_nodes, 2, true);

  source_elem.evaluate_shape(
      source_xi_gp, source_vals_xi_gp, source_derivs_xi_gp, num_source_nodes);
  target_elem.evaluate_shape(
      target_xi_gp, target_vals_xi_gp, target_derivs_xi_gp, num_target_nodes);

  // we also need the GP source coordinates + normal
  std::array<double, 3> sgpn = {0.0, 0.0, 0.0};
  std::array<double, 3> sgpx = {0.0, 0.0, 0.0};
  for (int i = 0; i < num_source_nodes; ++i)
    for (int k = 0; k < 3; ++k)
    {
      sgpn[k] += source_vals_xi_gp[i] * source_mortar_nodes[i]->mo_data().n()[k];
      sgpx[k] += source_vals_xi_gp[i] * source_mortar_nodes[i]->xspatial()[k];
    }

  // build 3x3 factor matrix L
  Core::LinAlg::Matrix<3, 3> lmatrix(Core::LinAlg::Initialization::zero);
  for (int k = 0; k < 3; ++k) lmatrix(k, 2) = -sgpn[k];
  for (int z = 0; z < num_target_nodes; ++z)
    for (int k = 0; k < 3; ++k)
    {
      lmatrix(k, 0) += target_derivs_xi_gp(z, 0) * target_mortar_nodes[z]->xspatial()[k];
      lmatrix(k, 1) += target_derivs_xi_gp(z, 1) * target_mortar_nodes[z]->xspatial()[k];
    }

  // get inverse of the 3x3 matrix L (in place)
  if (abs(lmatrix.determinant()) < 1e-12) FOUR_C_THROW("Singular lmatrix for derivgp3d");

  Core::LinAlg::invert(lmatrix);

  // build directional derivative of source GP normal
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  int linsize = 0;
  for (int i = 0; i < num_source_nodes; ++i)
  {
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(source_nodes[i]);
    linsize += cnode->get_linsize();
  }

  // TODO: this is for safety. Change to reasonable value!
  linsize *= 100;

  Core::Gen::Pairedvector<int, double> dmap_nxsl_gp(linsize);
  Core::Gen::Pairedvector<int, double> dmap_nysl_gp(linsize);
  Core::Gen::Pairedvector<int, double> dmap_nzsl_gp(linsize);

  for (int i = 0; i < num_source_nodes; ++i)
  {
    Core::Gen::Pairedvector<int, double>& dmap_nxsl_i =
        dynamic_cast<CONTACT::Node*>(source_mortar_nodes[i])->data().get_deriv_n()[0];
    Core::Gen::Pairedvector<int, double>& dmap_nysl_i =
        dynamic_cast<CONTACT::Node*>(source_mortar_nodes[i])->data().get_deriv_n()[1];
    Core::Gen::Pairedvector<int, double>& dmap_nzsl_i =
        dynamic_cast<CONTACT::Node*>(source_mortar_nodes[i])->data().get_deriv_n()[2];

    for (CI p = dmap_nxsl_i.begin(); p != dmap_nxsl_i.end(); ++p)
      dmap_nxsl_gp[p->first] += source_vals_xi_gp[i] * (p->second);
    for (CI p = dmap_nysl_i.begin(); p != dmap_nysl_i.end(); ++p)
      dmap_nysl_gp[p->first] += source_vals_xi_gp[i] * (p->second);
    for (CI p = dmap_nzsl_i.begin(); p != dmap_nzsl_i.end(); ++p)
      dmap_nzsl_gp[p->first] += source_vals_xi_gp[i] * (p->second);

    for (CI p = source_derivs_xi[0].begin(); p != source_derivs_xi[0].end(); ++p)
    {
      double valx = source_derivs_xi_gp(i, 0) * source_mortar_nodes[i]->mo_data().n()[0];
      dmap_nxsl_gp[p->first] += valx * (p->second);
      double valy = source_derivs_xi_gp(i, 0) * source_mortar_nodes[i]->mo_data().n()[1];
      dmap_nysl_gp[p->first] += valy * (p->second);
      double valz = source_derivs_xi_gp(i, 0) * source_mortar_nodes[i]->mo_data().n()[2];
      dmap_nzsl_gp[p->first] += valz * (p->second);
    }

    for (CI p = source_derivs_xi[1].begin(); p != source_derivs_xi[1].end(); ++p)
    {
      double valx = source_derivs_xi_gp(i, 1) * source_mortar_nodes[i]->mo_data().n()[0];
      dmap_nxsl_gp[p->first] += valx * (p->second);
      double valy = source_derivs_xi_gp(i, 1) * source_mortar_nodes[i]->mo_data().n()[1];
      dmap_nysl_gp[p->first] += valy * (p->second);
      double valz = source_derivs_xi_gp(i, 1) * source_mortar_nodes[i]->mo_data().n()[2];
      dmap_nzsl_gp[p->first] += valz * (p->second);
    }
  }

  // start to fill linearization maps for target GP
  // (1) all target nodes coordinates part
  for (int z = 0; z < num_target_nodes; ++z)
  {
    for (int k = 0; k < 3; ++k)
    {
      target_derivs_xi[0][target_mortar_nodes[z]->dofs()[k]] -=
          target_vals_xi_gp[z] * lmatrix(0, k);
      target_derivs_xi[1][target_mortar_nodes[z]->dofs()[k]] -=
          target_vals_xi_gp[z] * lmatrix(1, k);
    }
  }

  // (2) source Gauss point coordinates part
  for (int z = 0; z < num_source_nodes; ++z)
  {
    for (int k = 0; k < 3; ++k)
    {
      target_derivs_xi[0][source_mortar_nodes[z]->dofs()[k]] +=
          source_vals_xi_gp[z] * lmatrix(0, k);
      target_derivs_xi[1][source_mortar_nodes[z]->dofs()[k]] +=
          source_vals_xi_gp[z] * lmatrix(1, k);

      for (CI p = source_derivs_xi[0].begin(); p != source_derivs_xi[0].end(); ++p)
      {
        target_derivs_xi[0][p->first] += source_derivs_xi_gp(z, 0) *
                                         source_mortar_nodes[z]->xspatial()[k] * lmatrix(0, k) *
                                         (p->second);
        target_derivs_xi[1][p->first] += source_derivs_xi_gp(z, 0) *
                                         source_mortar_nodes[z]->xspatial()[k] * lmatrix(1, k) *
                                         (p->second);
      }

      for (CI p = source_derivs_xi[1].begin(); p != source_derivs_xi[1].end(); ++p)
      {
        target_derivs_xi[0][p->first] += source_derivs_xi_gp(z, 1) *
                                         source_mortar_nodes[z]->xspatial()[k] * lmatrix(0, k) *
                                         (p->second);
        target_derivs_xi[1][p->first] += source_derivs_xi_gp(z, 1) *
                                         source_mortar_nodes[z]->xspatial()[k] * lmatrix(1, k) *
                                         (p->second);
      }
    }
  }

  // (3) source Gauss point normal part
  for (CI p = dmap_nxsl_gp.begin(); p != dmap_nxsl_gp.end(); ++p)
  {
    target_derivs_xi[0][p->first] += alpha * lmatrix(0, 0) * (p->second);
    target_derivs_xi[1][p->first] += alpha * lmatrix(1, 0) * (p->second);
  }
  for (CI p = dmap_nysl_gp.begin(); p != dmap_nysl_gp.end(); ++p)
  {
    target_derivs_xi[0][p->first] += alpha * lmatrix(0, 1) * (p->second);
    target_derivs_xi[1][p->first] += alpha * lmatrix(1, 1) * (p->second);
  }
  for (CI p = dmap_nzsl_gp.begin(); p != dmap_nzsl_gp.end(); ++p)
  {
    target_derivs_xi[0][p->first] += alpha * lmatrix(0, 2) * (p->second);
    target_derivs_xi[1][p->first] += alpha * lmatrix(1, 2) * (p->second);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Implementation for meshtying interpolator                farah 10/14|
 *----------------------------------------------------------------------*/
NTS::MTInterpolator* NTS::MTInterpolator::impl(std::vector<Mortar::Element*> target_elems)
{
  // TODO: maybe this object should be crearted for one target_elem
  // and note for a vector of target_elems...
  switch (target_elems[0]->shape())
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
    Mortar::Node& source_node, std::vector<Mortar::Element*> target_elems)
{
  if (ndim_ == 2)
    interpolate_2d(source_node, target_elems);
  else if (ndim_ == 3)
    interpolate_3d(source_node, target_elems);
  else
    FOUR_C_THROW("wrong dimension!");

  return;
}


/*----------------------------------------------------------------------*
 |  interpolate (public)                                     farah 10/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_m>
void NTS::MTInterpolatorCalc<distype_m>::interpolate_2d(
    Mortar::Node& source_node, std::vector<Mortar::Element*> target_elems)
{
  // ********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************
  // check input data
  for (int i = 0; i < (int)target_elems.size(); ++i)
  {
    if ((!source_node.is_source()) || (target_elems[i]->is_source()))
      FOUR_C_THROW("IntegrateAndDerivSegment called on a wrong type of Mortar::Element pair!");
  }

  // bool for projection onto a target node
  bool kink_projection = false;

  //**************************************************************
  //                loop over all Target Elements
  //**************************************************************
  for (int numtarget = 0; numtarget < (int)target_elems.size(); ++numtarget)
  {
    // project Gauss point onto target element
    double target_xi[2] = {0.0, 0.0};
    Mortar::Projector::impl(*target_elems[numtarget])
        ->project_nodal_normal(source_node, *target_elems[numtarget], target_xi);

    // node on target_elem?
    if ((target_xi[0] >= -1.0) && (target_xi[0] <= 1.0) && (kink_projection == false))
    {
      kink_projection = true;
      source_node.has_proj() = true;

      static Core::LinAlg::Matrix<nm_, 1> target_val;
      Mortar::Utils::evaluate_shape_displ(target_xi, target_val, *target_elems[numtarget], false);

      // node-wise M value
      for (int k = 0; k < nm_; ++k)
      {
        Mortar::Node* target_node =
            dynamic_cast<Mortar::Node*>(target_elems[numtarget]->nodes()[k]);

        // multiply the two shape functions
        double prod = target_val(k);

        if (abs(prod) > MORTARINTTOL) source_node.add_m_value(target_node->id(), prod);
      }

      // dseg reduces to 1.0 for nts
      double prod = 1.0;

      if (abs(prod) > MORTARINTTOL) source_node.add_d_value(source_node.id(), prod);
    }  // End hit ele
  }  // End Loop over all Target Elements

  //**************************************************************

  return;
}


/*----------------------------------------------------------------------*
 |  interpolate (public)                                     farah 10/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_m>
void NTS::MTInterpolatorCalc<distype_m>::interpolate_3d(
    Mortar::Node& source_node, std::vector<Mortar::Element*> target_elems)
{
  // ********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************
  // check input data
  for (int i = 0; i < (int)target_elems.size(); ++i)
  {
    if ((!source_node.is_source()) || (target_elems[i]->is_source()))
      FOUR_C_THROW("IntegrateAndDerivSegment called on a wrong type of Mortar::Element pair!");
  }

  bool kink_projection = false;
  double source_xi[2] = {0.0, 0.0};

  // get local id
  Mortar::Element* source_elem =
      dynamic_cast<Mortar::Element*>(source_node.adjacent_elements()[0].user_element());

  int lid = -1;
  for (int i = 0; i < source_elem->num_node(); ++i)
  {
    if ((source_elem->nodes()[i])->id() == source_node.id())
    {
      lid = i;
      break;
    }
  }

  if (source_elem->shape() == Core::FE::CellType::quad4 or
      source_elem->shape() == Core::FE::CellType::quad8 or
      source_elem->shape() == Core::FE::CellType::quad9)
  {
    if (lid == 0)
    {
      source_xi[0] = -1;
      source_xi[1] = -1;
    }
    else if (lid == 1)
    {
      source_xi[0] = 1;
      source_xi[1] = -1;
    }
    else if (lid == 2)
    {
      source_xi[0] = 1;
      source_xi[1] = 1;
    }
    else if (lid == 3)
    {
      source_xi[0] = -1;
      source_xi[1] = 1;
    }
    else if (lid == 4)
    {
      source_xi[0] = 0;
      source_xi[1] = -1;
    }
    else if (lid == 5)
    {
      source_xi[0] = 1;
      source_xi[1] = 0;
    }
    else if (lid == 6)
    {
      source_xi[0] = 0;
      source_xi[1] = 1;
    }
    else if (lid == 7)
    {
      source_xi[0] = -1;
      source_xi[1] = 0;
    }
    else if (lid == 8)
    {
      source_xi[0] = 0;
      source_xi[1] = 0;
    }
    else
      FOUR_C_THROW("ERROR: wrong node LID");
  }
  else if (source_elem->shape() == Core::FE::CellType::tri3 or
           source_elem->shape() == Core::FE::CellType::tri6)
  {
    if (lid == 0)
    {
      source_xi[0] = 0;
      source_xi[1] = 0;
    }
    else if (lid == 1)
    {
      source_xi[0] = 1;
      source_xi[1] = 0;
    }
    else if (lid == 2)
    {
      source_xi[0] = 0;
      source_xi[1] = 1;
    }
    else if (lid == 3)
    {
      source_xi[0] = 0.5;
      source_xi[1] = 0;
    }
    else if (lid == 4)
    {
      source_xi[0] = 0.5;
      source_xi[1] = 0.5;
    }
    else if (lid == 5)
    {
      source_xi[0] = 0;
      source_xi[1] = 0.5;
    }
    else
      FOUR_C_THROW("ERROR: wrong node LID");
  }
  else
  {
    FOUR_C_THROW("Chosen element type not supported for NTS!");
  }

  //**************************************************************
  //                loop over all Target Elements
  //**************************************************************
  for (int numtarget = 0; numtarget < (int)target_elems.size(); ++numtarget)
  {
    // project Gauss point onto target element
    double target_xi[2] = {0.0, 0.0};
    double projalpha = 0.0;
    Mortar::Projector::impl(*source_elem, *target_elems[numtarget])
        ->project_gauss_point_3d(
            *source_elem, source_xi, *target_elems[numtarget], target_xi, projalpha);

    bool is_on_mele = true;

    // check GP projection
    const double tol = 0.00;
    if (distype_m == Core::FE::CellType::quad4 || distype_m == Core::FE::CellType::quad8 ||
        distype_m == Core::FE::CellType::quad9)
    {
      if (target_xi[0] < -1.0 - tol || target_xi[1] < -1.0 - tol || target_xi[0] > 1.0 + tol ||
          target_xi[1] > 1.0 + tol)
      {
        is_on_mele = false;
      }
    }
    else
    {
      if (target_xi[0] < -tol || target_xi[1] < -tol || target_xi[0] > 1.0 + tol ||
          target_xi[1] > 1.0 + tol || target_xi[0] + target_xi[1] > 1.0 + 2 * tol)
      {
        is_on_mele = false;
      }
    }

    // node on target_elem?
    if ((kink_projection == false) && (is_on_mele))
    {
      kink_projection = true;
      source_node.has_proj() = true;

      static Core::LinAlg::Matrix<nm_, 1> target_val;
      Mortar::Utils::evaluate_shape_displ(target_xi, target_val, *target_elems[numtarget], false);

      // node-wise M value
      for (int k = 0; k < nm_; ++k)
      {
        Mortar::Node* target_node =
            dynamic_cast<Mortar::Node*>(target_elems[numtarget]->nodes()[k]);

        // multiply the two shape functions
        double prod = target_val(k);

        if (abs(prod) > MORTARINTTOL) source_node.add_m_value(target_node->id(), prod);
      }

      // integrate dseg
      // multiply the two shape functions
      double prod = 1.0;

      if (abs(prod) > MORTARINTTOL) source_node.add_d_value(source_node.id(), prod);
    }  // End hit ele
  }  // End Loop over all Target Elements
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

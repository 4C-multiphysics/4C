// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_calc_xwall.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_gder2.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_parameter_std.hpp"
#include "4C_fluid_ele_parameter_timint.hpp"
#include "4C_fluid_rotsym_periodicbc.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_mat_newtonianfluid.hpp"

FOUR_C_NAMESPACE_OPEN


/*-----------------------------------------------------------------------------*
 | Constructor                                                      bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
Discret::Elements::FluidEleCalcXWall<distype, enrtype>::FluidEleCalcXWall()
    : Discret::Elements::FluidEleCalc<distype, enrtype>::FluidEleCalc(),
      ewdist_(Core::LinAlg::Initialization::zero),
      etauw_(Core::LinAlg::Initialization::zero),
      einctauw_(Core::LinAlg::Initialization::zero),
      eramp_(Core::LinAlg::Initialization::zero),
      epsi_(Core::LinAlg::Initialization::zero),
      epsinew_(Core::LinAlg::Initialization::zero),
      epsiold_(Core::LinAlg::Initialization::zero),
      eincwdist_(Core::LinAlg::Initialization::zero),
      visc_(0.0),
      viscinv_(0.0),
      xyze_(Core::LinAlg::Initialization::zero),
      functenr_(Core::LinAlg::Initialization::zero),
      funct_(Core::LinAlg::Initialization::zero),
      derxyenr_(Core::LinAlg::Initialization::zero),
      derxy_(Core::LinAlg::Initialization::zero),
      derxyenr2_(Core::LinAlg::Initialization::zero),
      derxy2_(Core::LinAlg::Initialization::zero),
      deriv_(Core::LinAlg::Initialization::zero),
      deriv2_(Core::LinAlg::Initialization::zero),
      k_(0.41),
      b_(5.17),
      expmkmb_(exp(-k_ * b_)),
      mk_(-1.0)
{
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
Discret::Elements::FluidEleCalcXWall<distype, enrtype>*
Discret::Elements::FluidEleCalcXWall<distype, enrtype>::instance(
    Core::Utils::SingletonAction action)
{
  static Core::Utils::SingletonOwner<Discret::Elements::FluidEleCalcXWall<distype, enrtype>>
      singleton_owner(
          []()
          {
            return std::unique_ptr<Discret::Elements::FluidEleCalcXWall<distype, enrtype>>(
                new Discret::Elements::FluidEleCalcXWall<distype, enrtype>());
          });

  return singleton_owner.instance(action);
}

/*-----------------------------------------------------------------------------*
 | Entry supporting methods of the element                          bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalcXWall<distype, enrtype>::evaluate_service(
    Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
    std::shared_ptr<Core::Mat::Material>& mat, Core::FE::Discretization& discretization,
    std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
    Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3)
{
  calcoldandnewpsi_ = false;
  const auto act = Teuchos::getIntegralValue<FLD::Action>(params, "action");
  if (act == FLD::xwall_l2_projection) calcoldandnewpsi_ = true;
  get_ele_properties(ele, discretization, lm, params, mat);

  // non-enriched case, solve problem as usual
  if (enrtype ==
      Discret::Elements::Fluid::xwall)  // this element has the same no of dofs on each node
  {
    std::vector<int> assembletoggle;
    int nodecount = 0;
    for (std::vector<int>::const_iterator i = lm.begin(); i != lm.end(); ++i)
    {
      ++nodecount;
      if (nodecount == 8)  // change here if I use a different number of dofs
      {
        assembletoggle.push_back(0);
        nodecount = 0;
      }
      else
      {
        assembletoggle.push_back(1);
      }
    }

    // the last dof must have been an unused pressure dof
    if (nodecount != 0)
      FOUR_C_THROW("something is wrong in this element with the number of virtual nodes vs dofs");

    int err = evaluate_service_x_wall(
        ele, params, mat, discretization, lm, elemat1, elemat2, elevec1, elevec2, elevec3);

    // for some evaluate_service actions, elevec1 is not necessary
    if (elevec1.length() != 0 && act != FLD::tauw_via_gradient && act != FLD::calc_div_u &&
        act != FLD::calc_dt_via_cfl && act != FLD::xwall_calc_mk &&
        act != FLD::calc_mass_flow_periodic_hill)
    {
      int row1 = 0;
      // assembly back into the old vector
      for (std::vector<int>::const_iterator i = assembletoggle.begin(); i != assembletoggle.end();
          ++i)
      {
        if (*i == 0) elevec1[row1] = 0.0;
        ++row1;
      }
    }

    return err;
  }
  else
    FOUR_C_THROW("not xwall element");


  return 1;
}


/*----------------------------------------------------------------------*
 * Evaluate supporting methods of the element
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalcXWall<distype, enrtype>::evaluate_service_x_wall(
    Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
    std::shared_ptr<Core::Mat::Material>& mat, Core::FE::Discretization& discretization,
    std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
    Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3)
{
  // get the action required
  const auto act = Teuchos::getIntegralValue<FLD::Action>(params, "action");

  switch (act)
  {
    case FLD::xwall_l2_projection:
    {
      // this action only considers enriched elements
      // I only need funct_ and maybe the first derivative
      my::is_higher_order_ele_ = false;
      return x_wall_projection(ele, params, discretization, lm, mat, elemat1,
          elemat2);  // here we can go further in the element and implement the matrix and rhs terms
    }
    break;
    case FLD::tauw_via_gradient:
    {
      // this action only considers enriched elements
      // I only need funct_ and maybe the first derivative
      my::is_higher_order_ele_ = false;
      return tau_w_via_gradient(ele, params, discretization, lm, mat, elevec1,
          elevec2);  // here we can go further in the element and implement the matrix and rhs terms
    }
    break;
    case FLD::xwall_calc_mk:
    {
      // this action only considers enriched elements
      // It is essential that the second derivatives exist!
      my::is_higher_order_ele_ = true;
      return calc_mk(ele, params, discretization, lm, mat, elevec1,
          elevec2);  // here we can go further in the element and implement the matrix and rhs terms
    }
    break;
    default:
      return my::evaluate_service(
          ele, params, mat, discretization, lm, elemat1, elemat2, elevec1, elevec2, elevec3);
      break;
  }  // end of switch(act)

  return 0;
}

/*-----------------------------------------------------------------------------*
 | Action type: Evaluate                                            bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalcXWall<distype, enrtype>::evaluate(Discret::Elements::Fluid* ele,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3, bool offdiag)
{
  calcoldandnewpsi_ = false;

  get_ele_properties(ele, discretization, lm, params, mat);

  if (enrtype ==
      Discret::Elements::Fluid::xwall)  // this element has the same no of dofs on each node
  {
    std::vector<int> assembletoggle;
    int nodecount = 0;
    for (std::vector<int>::const_iterator i = lm.begin(); i != lm.end(); ++i)
    {
      ++nodecount;
      if (nodecount == 8)
      {
        nodecount = 0;
        assembletoggle.push_back(0);
      }
      else
      {
        assembletoggle.push_back(1);
      }
    }

    // the last dof must have been an unused pressure dof
    if (nodecount != 0)
      FOUR_C_THROW("something is wrong in this element with the number of virtual nodes vs dofs");

    int err = my::evaluate(ele, discretization, lm, params, mat, elemat1, elemat2, elevec1, elevec2,
        elevec3, my::intpoints_);

    int row1 = 0;
    int col1 = 0;
    // assembly back into the old matrix
    for (std::vector<int>::const_iterator i = assembletoggle.begin(); i != assembletoggle.end();
        ++i)
    {
      row1 = 0;
      for (std::vector<int>::const_iterator j = assembletoggle.begin(); j != assembletoggle.end();
          ++j)
      {
        if (*i == 0 && *j == 0 && row1 == col1)
          elemat1(row1, col1) = 1.0;
        else if (*i == 0 or *j == 0)
          elemat1(row1, col1) = 0.0;
        ++row1;
      }
      ++col1;
    }

    row1 = 0;
    // assembly back into the old matrix
    for (std::vector<int>::const_iterator i = assembletoggle.begin(); i != assembletoggle.end();
        ++i)
    {
      if (*i == 0) elevec1[row1] = 0.0;
      ++row1;
    }

    return err;
  }
  else
    FOUR_C_THROW(
        "this should not have happened: some nodes have too many dofs in the LM vector, because "
        "they are dof-blending nodes and the wrong LocationVector() function is called");


  return 1;
}

/*-----------------------------------------------------------------------------*
 | Get properties for this element                                  bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalcXWall<distype, enrtype>::get_ele_properties(
    Discret::Elements::Fluid* ele, Core::FE::Discretization& discretization,
    const std::vector<int>& lm, Teuchos::ParameterList& params,
    std::shared_ptr<Core::Mat::Material>& mat)
{
  is_blending_ele_ = false;
  visc_ = 0.0;

  if (!(enrtype == Discret::Elements::Fluid::xwall))
    FOUR_C_THROW("This class is exclusively for the xwall enrichment type up to now");

  // rotate the vector field in the case of rotationally symmetric boundary conditions
  if (my::rotsymmpbc_->has_rot_symm_pbc()) FOUR_C_THROW("rotsymm pbc don't work with xwall");

  // get xwall toggle
  {
    const std::shared_ptr<Core::LinAlg::Vector<double>> xwalltoggle =
        params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("xwalltoggle");

    std::vector<double> mylocal(ele->num_node());
    Core::FE::extract_my_node_based_values(ele, mylocal, *xwalltoggle);

    for (unsigned inode = 0; inode < (unsigned)enren_; ++inode)  // number of nodes
    {
      etoggle_(inode) = mylocal.at(inode);
    }
  }

  int xwallnodes = 0;
  // init nodal ramp values
  for (int pq = 0; pq < enren_; ++pq)
  {
    if (etoggle_(pq) > 0.5)
    {
      ++xwallnodes;
      eramp_(pq) = 1.0;
    }
    else
      eramp_(pq) = 0.0;
  }

  if (xwallnodes > 0 && xwallnodes < enren_) is_blending_ele_ = true;

  // get wall distance
  {
    const std::shared_ptr<Core::LinAlg::Vector<double>> walldist =
        params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("walldist");
    //      std::cout << *walldist << std::endl;
    std::vector<double> mylocal(ele->num_node());
    Core::FE::extract_my_node_based_values(ele, mylocal, *walldist);

    for (unsigned inode = 0; inode < (unsigned)enren_; ++inode)  // number of nodes
    {
      ewdist_(inode) = mylocal.at(inode);
    }
  }

  // get tauw
  {
    const std::shared_ptr<Core::LinAlg::Vector<double>> tauw =
        params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("tauw");

    std::vector<double> mylocal(ele->num_node());
    Core::FE::extract_my_node_based_values(ele, mylocal, *tauw);

    for (unsigned inode = 0; inode < (unsigned)enren_; ++inode)  // number of nodes
    {
      etauw_(inode) = mylocal.at(inode);
    }
  }

  // get increment of tauw
  {
    const std::shared_ptr<Core::LinAlg::Vector<double>> inctauw =
        params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("inctauw");

    std::vector<double> mylocal(ele->num_node());
    Core::FE::extract_my_node_based_values(ele, mylocal, *inctauw);

    for (unsigned inode = 0; inode < (unsigned)enren_; ++inode)  // number of nodes
    {
      einctauw_(inode) = mylocal.at(inode);
    }
  }

  // get viscosity and density
  {
    const Mat::NewtonianFluid* actmat = static_cast<const Mat::NewtonianFluid*>(mat.get());
    if (!actmat) FOUR_C_THROW("not a newtonian fluid");
    // get constant dynamic viscosity
    dens_ = actmat->density();
    densinv_ = 1.0 / dens_;
    visc_ = actmat->viscosity() * densinv_;
    viscinv_ = 1.0 / visc_;
  }

  // calculate nodal shape values of psi
  for (int inode = 0; inode < enren_; inode++)
  {
    double utaunode = sqrt(etauw_(inode) * densinv_);
    double psinode = spaldings_law(ewdist_(inode), utaunode);

    epsi_(inode) = psinode;
  }

  if (calcoldandnewpsi_ == true)
  {
    // get old wall distance in case of ale
    if (ele->is_ale())
    {
      const std::shared_ptr<Core::LinAlg::Vector<double>> incwdist =
          params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("incwalldist");

      std::vector<double> mylocal(ele->num_node());
      Core::FE::extract_my_node_based_values(ele, mylocal, *incwdist);

      for (unsigned inode = 0; inode < (unsigned)enren_; ++inode)  // number of nodes
      {
        eincwdist_(inode) = mylocal.at(inode);
      }
    }

    for (int inode = 0; inode < enren_; inode++)
    {
      epsinew_(inode) = epsi_(inode);
      double utaunode = sqrt((etauw_(inode) - einctauw_(inode)) * densinv_);
      double psinode = spaldings_law(ewdist_(inode) - eincwdist_(inode), utaunode);

      epsiold_(inode) = psinode;
    }
  }


  // get element mk for stabilization
  const std::shared_ptr<Core::LinAlg::Vector<double>> mkvec =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("mk");
  mk_ = mkvec->local_values_as_span()[mkvec->get_map().lid(ele->id())];

  numgpnorm_ = params.get<int>("gpnorm");
  numgpnormow_ = params.get<int>("gpnormow");
  numgpplane_ = params.get<int>("gppar");
  // get node coordinates and number of elements per node
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, my::xyze_);
  Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
  if (ele->is_ale()) get_grid_disp_ale(discretization, lm, edispnp);
  prepare_gauss_rule();

  return;
}

/*-----------------------------------------------------------------------------*
 | Go wall shear stress increment backwards                         bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalcXWall<distype, enrtype>::x_wall_tau_w_inc_back()
{
  for (unsigned inode = 0; inode < (unsigned)enren_; ++inode)  // number of nodes
  {
    etauw_(inode) -= einctauw_(inode);
    epsi_(inode) = epsiold_(inode);
    ewdist_(inode) -= eincwdist_(inode);
  }

  return;
}

/*-----------------------------------------------------------------------------*
 | Go wall shear stress increment forward                           bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalcXWall<distype, enrtype>::x_wall_tau_w_inc_forward()
{
  for (unsigned inode = 0; inode < (unsigned)enren_; ++inode)  // number of nodes
  {
    etauw_(inode) += einctauw_(inode);
    epsi_(inode) = epsinew_(inode);
    ewdist_(inode) += eincwdist_(inode);
  }

  return;
}

/*-----------------------------------------------------------------------------*
 | Calculate shape functions at integration point                   bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalcXWall<distype, enrtype>::
    eval_shape_func_and_derivs_at_int_point(
        const double* gpcoord,  // actual integration point (coords)
        double gpweight         // actual integration point (weight)
    )
{
  eval_std_shape_func_and_derivs_at_int_point(gpcoord, gpweight);

  eval_enrichment();
  return;
}

/*-----------------------------------------------------------------------------*
 | Calculate shape functions at integration point                   bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalcXWall<distype, enrtype>::
    eval_std_shape_func_and_derivs_at_int_point(
        const double* gpcoord,  // actual integration point (coords)
        double gpweight         // actual integration point (weight)
    )
{
  funct_.clear();
  derxy_.clear();
  derxy2_.clear();

  // put the geometry in the local copy
  for (int inode = 0; inode < enren_; ++inode)
  {
    for (int sdm = 0; sdm < nsd_; ++sdm) xyze_(sdm, inode) = my::xyze_(sdm, inode);
  }

  // coordinates of the current integration point
  for (int idim = 0; idim < nsd_; idim++)
  {
    my::xsi_(idim) = gpcoord[idim];
  }

  // shape functions and their first derivatives
  Core::FE::shape_function<distype>(my::xsi_, funct_);
  Core::FE::shape_function_deriv1<distype>(my::xsi_, deriv_);
  if (my::is_higher_order_ele_ && distype == Core::FE::CellType::hex8)
  {
    // get the second derivatives of standard element at current GP
    Core::FE::shape_function_deriv2<distype>(my::xsi_, deriv2_);
  }
  else
    deriv2_.clear();

  // get Jacobian matrix and determinant
  // actually compute its transpose....
  /*
    +-            -+ T      +-            -+
    | dx   dx   dx |        | dx   dy   dz |
    | --   --   -- |        | --   --   -- |
    | dr   ds   dt |        | dr   dr   dr |
    |              |        |              |
    | dy   dy   dy |        | dx   dy   dz |
    | --   --   -- |   =    | --   --   -- |
    | dr   ds   dt |        | ds   ds   ds |
    |              |        |              |
    | dz   dz   dz |        | dx   dy   dz |
    | --   --   -- |        | --   --   -- |
    | dr   ds   dt |        | dt   dt   dt |
    +-            -+        +-            -+
  */
  my::xjm_.multiply_nt(deriv_, xyze_);

  my::det_ = my::xji_.invert(my::xjm_);

  if (my::det_ < 1E-16)
    FOUR_C_THROW(
        "GLOBAL ELEMENT NO.{}\nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", my::eid_, my::det_);

  // compute integration factor
  my::fac_ = gpweight * my::det_;

  // compute global first derivatives
  derxy_.multiply(my::xji_, deriv_);

  //--------------------------------------------------------------
  //             compute global second derivatives
  //--------------------------------------------------------------
  if (my::is_higher_order_ele_ && distype == Core::FE::CellType::hex8)
  {
    Core::FE::gder2<distype, enren_>(my::xjm_, derxy_, deriv2_, xyze_, derxy2_);
  }
  else
    derxy2_.clear();

  return;
}

/*-----------------------------------------------------------------------------*
 | Calculate enrichment shape functions                             bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalcXWall<distype, enrtype>::eval_enrichment()
{
  // first clear everything
  functenr_.clear();
  derxyenr_.clear();
  derxyenr2_.clear();

  Core::LinAlg::Matrix<nsd_, 1> derpsigp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numderiv2_, 1> der2psigp(Core::LinAlg::Initialization::zero);

  double psigp = enrichment_shape_der(derpsigp, der2psigp);

  // shape function
  for (int inode = 0; inode < enren_; ++inode)
    functenr_(inode) = funct_(inode) * (psigp - epsi_(inode));

  // first derivative
  for (int inode = 0; inode < enren_; ++inode)
  {
    for (int sdm = 0; sdm < nsd_; ++sdm)
    {
      derxyenr_(sdm, inode) =
          derxy_(sdm, inode) * (psigp - epsi_(inode)) + funct_(inode) * derpsigp(sdm);
    }
  }

  if (my::is_higher_order_ele_)
  {
    for (int inode = 0; inode < enren_; ++inode)
    {
      for (int sdm = 0; sdm < numderiv2_; ++sdm)
      {
        const std::array<int, 6> i = {0, 1, 2, 0, 0, 1};
        const std::array<int, 6> j = {0, 1, 2, 1, 2, 2};
        derxyenr2_(sdm, inode) += derxy2_(sdm, inode) * (psigp - epsi_(inode)) +
                                  derxy_(i[sdm], inode) * derpsigp(j[sdm]) +
                                  derxy_(j[sdm], inode) * derpsigp(i[sdm]) +
                                  funct_(inode) * der2psigp(sdm);
      }
    }
  }

  // treat blending elements with ramp functions
  if (is_blending_ele_)
  {
    Core::LinAlg::Matrix<nsd_, 1> derramp(Core::LinAlg::Initialization::zero);
    derramp.multiply(derxy_, eramp_);
    Core::LinAlg::Matrix<numderiv2_, 1> der2ramp(Core::LinAlg::Initialization::zero);
    der2ramp.multiply(derxy2_, eramp_);
    double ramp = eramp_.dot(funct_);

    for (int inode = 0; inode < enren_; ++inode)
    {
      if (my::is_higher_order_ele_)
      {
        for (int sdm = 0; sdm < numderiv2_; ++sdm)
        {
          const std::array<int, 6> i = {0, 1, 2, 0, 0, 1};
          const std::array<int, 6> j = {0, 1, 2, 1, 2, 2};

          derxyenr2_(sdm, inode) *= ramp;
          derxyenr2_(sdm, inode) += derxyenr_(i[sdm], inode) * derramp(j[sdm]) +
                                    derxyenr_(j[sdm], inode) * derramp(i[sdm]);
          derxyenr2_(sdm, inode) += functenr_(inode) * der2ramp(sdm);
        }
      }
      for (int sdm = 0; sdm < nsd_; ++sdm)
      {
        derxyenr_(sdm, inode) = derxyenr_(sdm, inode) * ramp + functenr_(inode) * derramp(sdm);
      }
      // treat derivative first, because we use functenr_ without ramp_
      functenr_(inode) *= ramp;
    }
  }

  // put everything in standard shape function
  for (int inode = 0; inode < enren_; ++inode)
  {
    const int inodetwo = inode * 2;
    my::funct_(inodetwo) = funct_(inode);
    my::funct_(inodetwo + 1) = functenr_(inode);
    for (int sdm = 0; sdm < nsd_; ++sdm)
    {
      my::derxy_(sdm, inodetwo) = derxy_(sdm, inode);
      my::derxy_(sdm, inodetwo + 1) = derxyenr_(sdm, inode);
    }

    if (my::is_higher_order_ele_)
    {
      for (int sdm = 0; sdm < numderiv2_; ++sdm)
      {
        my::derxy2_(sdm, inodetwo) = derxy2_(sdm, inode);
        my::derxy2_(sdm, inodetwo + 1) = derxyenr2_(sdm, inode);
      }
    }
  }

  return;
}

/*-----------------------------------------------------------------------------*
 | Calculate enrichment shape functions and derivatives                        |
 | including transformation from y+ to (x,y,z)                      bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
double Discret::Elements::FluidEleCalcXWall<distype, enrtype>::enrichment_shape_der(
    Core::LinAlg::Matrix<nsd_, 1>& derpsigp, Core::LinAlg::Matrix<numderiv2_, 1>& der2psigp)
{
  // calculate transformation ---------------------------------------
  double wdist = ewdist_.dot(funct_);
  double tauw = etauw_.dot(funct_);
  Core::LinAlg::Matrix<nsd_, 1> derwdist(Core::LinAlg::Initialization::zero);
  derwdist.multiply(derxy_, ewdist_);
  Core::LinAlg::Matrix<nsd_, 1> dertauw(Core::LinAlg::Initialization::zero);
  dertauw.multiply(derxy_, etauw_);
  Core::LinAlg::Matrix<numderiv2_, 1> der2wdist(Core::LinAlg::Initialization::zero);
  if (my::is_higher_order_ele_) der2wdist.multiply(derxy2_, ewdist_);
  Core::LinAlg::Matrix<numderiv2_, 1> der2tauw(Core::LinAlg::Initialization::zero);
  if (my::is_higher_order_ele_) der2tauw.multiply(derxy2_, etauw_);
  Core::LinAlg::Matrix<nsd_, 1> dertrans(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numderiv2_, 1> der2trans_1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numderiv2_, 1> der2trans_2(Core::LinAlg::Initialization::zero);

  if (tauw < 1.0e-10) FOUR_C_THROW("tauw is almost zero");
  if (dens_ < 1.0e-10) FOUR_C_THROW("density is almost zero");

  const double utau = sqrt(tauw * densinv_);
  const double fac = 1.0 / (2.0 * sqrt(dens_ * tauw));
  const double wdistfac = wdist * fac;

  for (int sdm = 0; sdm < nsd_; ++sdm)
    dertrans(sdm) = (utau * derwdist(sdm) + wdistfac * dertauw(sdm)) * viscinv_;


  // second derivative, first part: to be multiplied with der2psigpsc
  // second derivative, second part: to be multiplied with derpsigpsc
  if (my::is_higher_order_ele_)
  {
    const double wdistfactauwtwoinv = wdistfac / (tauw * 2.0);

    for (int sdm = 0; sdm < numderiv2_; ++sdm)
    {
      const std::array<int, 6> i = {0, 1, 2, 0, 0, 1};
      const std::array<int, 6> j = {0, 1, 2, 1, 2, 2};

      der2trans_1(sdm) = dertrans(i[sdm]) * dertrans(j[sdm]);

      der2trans_2(sdm) = (derwdist(j[sdm]) * fac * dertauw(i[sdm]) + wdistfac * der2tauw(sdm) -
                             wdistfactauwtwoinv * dertauw(i[sdm]) * dertauw(j[sdm]) +
                             dertauw(j[sdm]) * fac * derwdist(i[sdm]) + utau * der2wdist(sdm)) *
                         viscinv_;
    }
  }
  // calculate transformation done ----------------------------------

  // get enrichment function and scalar derivatives
  const double psigp = spaldings_law(wdist, utau);
  const double derpsigpsc = der_spaldings_law(wdist, utau, psigp);
  const double der2psigpsc = der2_spaldings_law(wdist, utau, psigp, derpsigpsc);

  // calculate final derivatives
  for (int sdm = 0; sdm < nsd_; ++sdm)
  {
    derpsigp(sdm) = derpsigpsc * dertrans(sdm);
  }
  if (my::is_higher_order_ele_)
    for (int sdm = 0; sdm < numderiv2_; ++sdm)
    {
      der2psigp(sdm) = der2psigpsc * der2trans_1(sdm);
      der2psigp(sdm) += derpsigpsc * der2trans_2(sdm);
    }

  return psigp;
}

/*-----------------------------------------------------------------------------*
 | Enrichment function (modification of Spalding's law)             bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
double Discret::Elements::FluidEleCalcXWall<distype, enrtype>::spaldings_law(
    double dist, double utau)
{
  // watch out, this is not exactly Spalding's law but psi=u_+*k, which saves quite some
  // multiplications

  double yplus = dist * utau * viscinv_;
  double psi = 0.0;

  const double km1 = 1.0 / k_;

  if (yplus >
      11.0)  // this is approximately where the intersection of log law and linear region lies
    psi = log(yplus) + b_ * k_;
  else
    psi = yplus * k_;

  double inc = 10.0;
  double fn = 10.0;
  int count = 0;
  while (abs(inc) > 1.0E-14 && abs(fn) > 1.0E-14 && 1000 > count++)
  {
    double psiquad = psi * psi;
    double exppsi = exp(psi);
    fn = -yplus + psi * km1 +
         expmkmb_ *
             (exppsi - 1.0 - psi - psiquad * 0.5 - psiquad * psi / 6.0 - psiquad * psiquad / 24.0);
    double dfn = km1 + expmkmb_ * (exppsi - 1.0 - psi - psiquad * 0.5 - psiquad * psi / 6.0);

    inc = fn / dfn;

    psi -= inc;
  }

  return psi;

  // Reichardt's law 1951
  // return (1.0/k_*log(1.0+0.4*yplus)+7.8*(1.0-exp(-yplus/11.0)-(yplus/11.0)*exp(-yplus/3.0)))*k_;
}

/*-----------------------------------------------------------------------------*
 | Derivative of enrichment function w.r.t. y+                         bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
double Discret::Elements::FluidEleCalcXWall<distype, enrtype>::der_spaldings_law(
    double dist, double utau, double psi)
{
  // derivative with respect to y+!
  // spaldings law according to paper (derivative)
  return 1.0 /
         (1.0 / k_ + expmkmb_ * (exp(psi) - 1.0 - psi - psi * psi * 0.5 - psi * psi * psi / 6.0));

  // Reichardt's law
  //  double yplus=dist*utau*viscinv_;
  //  return
  //  (0.4/(k_*(1.0+0.4*yplus))+7.8*(1.0/11.0*exp(-yplus/11.0)-1.0/11.0*exp(-yplus/3.0)+yplus/33.0*exp(-yplus/3.0)))*k_;
}

/*-----------------------------------------------------------------------------*
 | Second derivative of enrichment function w.r.t. y+               bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
double Discret::Elements::FluidEleCalcXWall<distype, enrtype>::der2_spaldings_law(
    double dist, double utau, double psi, double derpsi)
{
  // derivative with respect to y+!
  // spaldings law according to paper (2nd derivative)
  return -expmkmb_ * (exp(psi) - 1 - psi - psi * psi * 0.5) * derpsi * derpsi * derpsi;

  // Reichardt's law
  //  double yplus=dist*utau*viscinv_;
  //  return
  //  (-0.4*0.4/(k_*(1.0+0.4*yplus)*(1.0+0.4*yplus))+7.8*(-1.0/121.0*exp(-yplus/11.0)+(2.0/33.0-yplus/99.0)*exp(-yplus/3.0)))*k_;
}

/*-----------------------------------------------------------------------------*
 | Calculate matrix for l2 projection                               bk 07/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalcXWall<distype, enrtype>::tau_w_via_gradient(
    Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material>& mat, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2)
{
  //----------------------------------------------------------------------------
  //   Extract velocity/pressure from global vectors
  //----------------------------------------------------------------------------

  Core::LinAlg::Matrix<nsd_, nen_> evel(Core::LinAlg::Initialization::zero);
  my::extract_values_from_global_vector(
      discretization, lm, *my::rotsymmpbc_, &evel, nullptr, "vel");

  //----------------------------------------------------------------------------
  //                         ELEMENT GEOMETRY
  //----------------------------------------------------------------------------

  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, my::xyze_);

  if (ele->is_ale())
  {
    Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nsd_, nen_> egridv(Core::LinAlg::Initialization::zero);
    my::get_grid_disp_vel_ale(discretization, lm, edispnp, egridv);
    evel -= egridv;
  }


  //------------------------------------------------------------------
  //                       INTEGRATION LOOP
  //------------------------------------------------------------------


  // number of nodes, the derivative should be calculated at the nodes
  for (int inode = 0; inode < enren_; ++inode)
  {
    // calculate only for the wall nodes
    if (ewdist_(inode) < 1e-4)
    {
      Core::LinAlg::Matrix<3, 1> test = Core::FE::get_node_coordinates(inode, distype);
      const std::array<double, 3> gp = {test(0, 0), test(1, 0), test(2, 0)};
      const double* gpc = gp.data();
      // evaluate shape functions and derivatives at integration point
      eval_shape_func_and_derivs_at_int_point(gpc, 1.0);

      // calculate wall-normal vector
      Core::LinAlg::Matrix<nsd_, 1> normwall(Core::LinAlg::Initialization::zero);
      normwall.multiply(derxy_, ewdist_);

      // at certain corner elements, it can happen, that the normal vector calculated at the
      // boundary is zero. so we move a bit inside the element and calculate the properties there
      // instead
      if (normwall.norm2() < 1.0e-10)
      {
        test.scale(0.95);
        const std::array<double, 3> gp = {test(0, 0), test(1, 0), test(2, 0)};
        const double* gpc = gp.data();
        // evaluate shape functions and derivatives at integration point
        eval_shape_func_and_derivs_at_int_point(gpc, 1.0);

        normwall.multiply(derxy_, ewdist_);
        if (normwall.norm2() < 1.0e-10)
          FOUR_C_THROW("normal vector has length zero, even in the second try");
      }

      // unit vector
      normwall.scale(1.0 / normwall.norm2());

      Core::LinAlg::Matrix<nsd_, nsd_> velderxy(Core::LinAlg::Initialization::zero);
      velderxy.multiply_nt(evel, my::derxy_);

      // remove normal part

      //      normwall.scale(1.0/normwall.norm2());
      Core::LinAlg::Matrix<nsd_, nsd_> velderxywoun(Core::LinAlg::Initialization::zero);
      for (int idim = 0; idim < nsd_; idim++)
        for (int jdim = 0; jdim < nsd_; jdim++)
          velderxywoun(idim, jdim) = velderxy(idim, jdim) * (1.0 - abs(normwall(idim)));

      // now transform to derivative w.r.t. n
      Core::LinAlg::Matrix<nsd_, 1> veldern(Core::LinAlg::Initialization::zero);
      for (int idim = 0; idim < nsd_; idim++)
        for (int jdim = 0; jdim < nsd_; jdim++)
          veldern(idim) += velderxywoun(idim, jdim) * normwall(jdim);

      // calculate wall shear stress with dynamic! viscosity
      elevec1(inode) = visc_ * dens_ * veldern.norm2();
      // the following vector counts, how often this node is assembled
      //(e.g. from neighboring elements)
      elevec2(inode) = 1.0;
    }
  }  // end of integration loop

  return 0;
}

/*-----------------------------------------------------------------------------*
 | Calculate stabilization parameter mk                             bk 07/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
double Discret::Elements::FluidEleCalcXWall<distype, enrtype>::get_mk()
{
  if (mk_ < 0.0)
    return calc_mk();
  else
    return mk_;
  FOUR_C_THROW("mk could not be determined for xwall");
  return 0.0;
}


/*-----------------------------------------------------------------------------*
 | Calculate stabilization parameter mk                             bk 07/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
double Discret::Elements::FluidEleCalcXWall<distype, enrtype>::calc_mk()
{
  if (my::is_higher_order_ele_ == false)
    FOUR_C_THROW("It is essential that the second derivatives exist!");

  Core::FE::GaussIntegration intpoints(Core::FE::CellType::hex8, 1);
  if (distype == Core::FE::CellType::hex8)
  {
    Core::FE::GaussIntegration intpointstmp(cgp_);
    intpoints = intpointstmp;
  }
  else if (distype == Core::FE::CellType::tet4)
  {
    Core::FE::GaussIntegration intpointsplane(Core::FE::CellType::tet4, 2 * numgpnorm_ - 1);
    intpoints = intpointsplane;
  }

  Core::LinAlg::SerialDenseMatrix elemat1;
  Core::LinAlg::SerialDenseMatrix elemat2;
  elemat1.shape(nen_, nen_);
  elemat2.shape(nen_, nen_);
  Core::LinAlg::Matrix<nen_, nen_> Amat(elemat1.values(), true);
  Core::LinAlg::Matrix<nen_, nen_> Bmat(elemat2.values(), true);

  double vol = 0.0;
  //------------------------------------------------------------------
  //                       INTEGRATION LOOP
  //------------------------------------------------------------------
  for (Core::FE::GaussIntegration::iterator iquad = intpoints.begin(); iquad != intpoints.end();
      ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    const unsigned Velx = 0;
    const unsigned Vely = 1;
    const unsigned Velz = 2;

    for (int vi = 0; vi < nen_; ++vi)
    {
      for (int ui = 0; ui < nen_; ++ui)
      {
        //(x,x)
        // mixed derivatives are neglected such that the standard values are recovered for ideal
        // higher order cube elements, i.e. C_l={12,60} for polynomial orders p={2,3}. See Harari &
        // Hughes 1991 mixed derivatives practically don't have any influence
        Amat(vi, ui) += my::fac_ * (my::derxy2_(Velx, vi) * my::derxy2_(Velx, ui)) +
                        my::fac_ * (my::derxy2_(Vely, vi) * my::derxy2_(Vely, ui)) +
                        my::fac_ * (my::derxy2_(Velz, vi) * my::derxy2_(Velz, ui));
      }
    }

    /*
  //    /                \
  //   |                  |
  //   |Nabla(v),Nabla(u) |
  //   |                  |
  //    \                / volume
     */

    for (int vi = 0; vi < nen_; ++vi)
    {
      for (int ui = 0; ui < nen_; ++ui)
      {
        //(x,x)
        Bmat(vi, ui) += my::fac_ * (my::derxy_(Velx, vi) * my::derxy_(Velx, ui)) +
                        my::fac_ * (my::derxy_(Vely, vi) * my::derxy_(Vely, ui)) +
                        my::fac_ * (my::derxy_(Velz, vi) * my::derxy_(Velz, ui));
      }
    }

    vol += my::fac_;
  }  // gauss loop

  const double maxeigenvalue =
      Core::LinAlg::generalized_eigen_max_real_eigenvalue(elemat1, elemat2);

  double h_u = 0.0;
  if (my::fldpara_->which_tau() == Inpar::FLUID::tau_franca_barrenechea_valentin_frey_wall ||
      my::fldpara_->which_tau() == Inpar::FLUID::tau_codina ||
      my::fldpara_->which_tau() == Inpar::FLUID::tau_codina_convscaled)
  {
    if (!(my::fldpara_->char_ele_length_u() == Inpar::FLUID::volume_equivalent_diameter_u))
      FOUR_C_THROW("only volume equivalent diameter defined up to now");

    // volume equivalent diameter
    h_u = std::pow((6. * vol / std::numbers::pi), (1.0 / 3.0)) / sqrt(3.0);
  }
  else
    FOUR_C_THROW(
        "Element length not defined for dynamic determination of mk for your stabilization "
        "parameter");

  if (abs(maxeigenvalue) < 1.e-9)
  {
    std::cout << "Warning: maxeigenvalue zero:  " << maxeigenvalue << std::endl;
    return 0.33333333333;
  }
  else if (1.0 / (maxeigenvalue * h_u * h_u) > 0.33)
  {
    std::cout << "Warning: mk larger than 0.33:  " << maxeigenvalue * h_u * h_u << std::endl;

    return 0.33333333333;
  }

  // safety factor
  const double sfac = 1.0;
  //  std::cout << sfac/(maxeigenvalue*h_u*h_u) << std::endl;
  return sfac / (maxeigenvalue * h_u * h_u);
}

/*-----------------------------------------------------------------------------*
 | Calculate stabilization parameter mk                             bk 07/2014 |
 | (call for action type)                                                      |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalcXWall<distype, enrtype>::calc_mk(Discret::Elements::Fluid* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    const std::vector<int>& lm, std::shared_ptr<Core::Mat::Material>& mat,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2)
{
  //----------------------------------------------------------------------------
  //                         ELEMENT GEOMETRY
  //----------------------------------------------------------------------------

  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, my::xyze_);

  Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
  if (ele->is_ale()) get_grid_disp_ale(discretization, lm, edispnp);

  elevec1[0] = calc_mk();
  return 0;
}

/*-----------------------------------------------------------------------------*
 | Calculate matrix for l2 projection                               bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
int Discret::Elements::FluidEleCalcXWall<distype, enrtype>::x_wall_projection(
    Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material>& mat, Core::LinAlg::SerialDenseMatrix& elemat1,
    Core::LinAlg::SerialDenseMatrix& elemat2)
{
  const int numdof = 3;

  //----------------------------------------------------------------------------
  //   Extract velocity/pressure from global vectors
  //----------------------------------------------------------------------------

  Core::LinAlg::Matrix<nsd_, nen_> eveln(Core::LinAlg::Initialization::zero);
  my::extract_values_from_global_vector(
      discretization, lm, *my::rotsymmpbc_, &eveln, nullptr, "veln");

  Core::LinAlg::Matrix<nsd_, nen_> eaccn(Core::LinAlg::Initialization::zero);
  bool switchonaccn = discretization.has_state("accn");
  if (switchonaccn)
    my::extract_values_from_global_vector(
        discretization, lm, *my::rotsymmpbc_, &eaccn, nullptr, "accn");

  Core::LinAlg::Matrix<nsd_, nen_> evelnp(Core::LinAlg::Initialization::zero);
  bool switchonvelnp = discretization.has_state("velnp");
  if (switchonvelnp)
    my::extract_values_from_global_vector(
        discretization, lm, *my::rotsymmpbc_, &evelnp, nullptr, "velnp");

  //----------------------------------------------------------------------------
  //                         ELEMENT GEOMETRY
  //----------------------------------------------------------------------------

  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, my::xyze_);

  Core::LinAlg::Matrix<nsd_, nen_> edispnp(Core::LinAlg::Initialization::zero);
  if (ele->is_ale()) get_grid_disp_ale(discretization, lm, edispnp);

  //  Core::FE::GaussIntegration intpoints(Core::FE::CellType::line6);

  //------------------------------------------------------------------
  //                       INTEGRATION LOOP
  //------------------------------------------------------------------
  for (Core::FE::GaussIntegration::iterator iquad = my::intpoints_.begin();
      iquad != my::intpoints_.end(); ++iquad)
  {
    // evaluate shape functions and derivatives at integration point
    eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    Core::LinAlg::Matrix<nen_, 1> newfunct(my::funct_);

    x_wall_tau_w_inc_back();

    // evaluate shape functions and derivatives at integration point
    eval_shape_func_and_derivs_at_int_point(iquad.point(), iquad.weight());

    Core::LinAlg::Matrix<nen_, 1> oldfunct(my::funct_);

    x_wall_tau_w_inc_forward();

    //----------------------------------------------------------------------------
    //                         MASS MATRIX
    //----------------------------------------------------------------------------
    std::array<int, nsd_> idim_nsd_p_idim;
    Core::LinAlg::Matrix<nsd_ * nsd_, enren_> lin_resM_Du(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<enren_ * nsd_, enren_ * nsd_> estif_u(Core::LinAlg::Initialization::zero);

    for (int idim = 0; idim < nsd_; ++idim)
    {
      idim_nsd_p_idim[idim] = idim * nsd_ + idim;
    }

    for (int ui = 1; ui < nen_; ui += 2)
    {
      const double v = my::fac_ * newfunct(ui);
      int nodeui = 0;
      if (ui % 2 == 1)
        nodeui = (ui - 1) / 2;
      else
        FOUR_C_THROW("something wrong with indices. also correct in all following terms");

      for (int idim = 0; idim < nsd_; ++idim)
      {
        lin_resM_Du(idim_nsd_p_idim[idim], nodeui) += v;
      }
    }

    for (int ui = 1; ui < nen_; ui += 2)
    {
      const int nodeui = (ui - 1) / 2;
      const int nsd_nodeui = nsd_ * nodeui;
      for (int vi = 1; vi < nen_; vi += 2)
      {
        const int nsd_nodevi = nsd_ * (vi - 1) / 2;

        for (int idim = 0; idim < nsd_; ++idim)
        {
          estif_u(nsd_nodevi + idim, nsd_nodeui + idim) +=
              newfunct(vi) * lin_resM_Du(idim_nsd_p_idim[idim], nodeui);
        }  // end for (idim)
      }  // vi
    }  // ui

    // add velocity-velocity part to matrix
    for (int ui = 0; ui < enren_; ++ui)
    {
      const int numdof_ui = numdof * ui;
      const int nsd_ui = nsd_ * ui;

      for (int jdim = 0; jdim < nsd_; ++jdim)
      {
        const int numdof_ui_jdim = numdof_ui + jdim;
        const int nsd_ui_jdim = nsd_ui + jdim;

        for (int vi = 0; vi < enren_; ++vi)
        {
          const int numdof_vi = numdof * vi;
          const int nsd_vi = nsd_ * vi;

          for (int idim = 0; idim < nsd_; ++idim)
          {
            elemat1(numdof_vi + idim, numdof_ui_jdim) += estif_u(nsd_vi + idim, nsd_ui_jdim);
          }
        }
      }
    }

    //----------------------------------------------------------------------------
    //                         RHS
    //----------------------------------------------------------------------------
    estif_u.clear();
    lin_resM_Du.clear();

    for (int ui = 1; ui < nen_; ui += 2)
    {
      const double v = my::fac_ * (oldfunct(ui) - newfunct(ui));
      const int nodeui = (ui - 1) / 2;

      for (int idim = 0; idim < nsd_; ++idim)
      {
        lin_resM_Du(idim_nsd_p_idim[idim], nodeui) += v;
      }
    }

    for (int ui = 1; ui < nen_; ui += 2)
    {
      const int nodeui = (ui - 1) / 2;
      const int nsd_nodeui = nsd_ * nodeui;
      for (int vi = 1; vi < nen_; vi += 2)
      {
        const int nsd_nodevi = nsd_ * (vi - 1) / 2;
        for (int idim = 0; idim < nsd_; ++idim)
        {
          estif_u(nsd_nodevi + idim, nsd_nodeui + idim) +=
              newfunct(vi) * lin_resM_Du(idim_nsd_p_idim[idim], nodeui);
        }  // end for (idim)
      }  // vi
    }  // ui

    // veln
    // add velocity-velocity part to rhs
    for (int ui = 0; ui < enren_; ++ui)
    {
      const int nsd_ui = nsd_ * ui;
      const int uix = 2 * ui + 1;

      for (int jdim = 0; jdim < nsd_; ++jdim)
      {
        const int nsd_ui_jdim = nsd_ui + jdim;

        for (int vi = 0; vi < enren_; ++vi)
        {
          const int numdof_vi = numdof * vi;
          const int nsd_vi = nsd_ * vi;

          for (int idim = 0; idim < nsd_; ++idim)
          {
            elemat2(numdof_vi + idim, 0) += estif_u(nsd_vi + idim, nsd_ui_jdim) * eveln(jdim, uix);
          }
        }
      }
    }

    // accn
    if (switchonaccn)
    {
      // add velocity-velocity part to rhs
      for (int ui = 0; ui < enren_; ++ui)
      {
        const int nsd_ui = nsd_ * ui;
        const int uix = 2 * ui + 1;

        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          const int nsd_ui_jdim = nsd_ui + jdim;

          for (int vi = 0; vi < enren_; ++vi)
          {
            const int numdof_vi = numdof * vi;
            const int nsd_vi = nsd_ * vi;

            for (int idim = 0; idim < nsd_; ++idim)
            {
              elemat2(numdof_vi + idim, 1) +=
                  estif_u(nsd_vi + idim, nsd_ui_jdim) * eaccn(jdim, uix);
            }
          }
        }
      }
    }

    if (switchonvelnp)
    {
      // velnp
      // add velocity-velocity part to rhs
      for (int ui = 0; ui < enren_; ++ui)
      {
        const int nsd_ui = nsd_ * ui;
        const int uix = 2 * ui + 1;

        for (int jdim = 0; jdim < nsd_; ++jdim)
        {
          const int nsd_ui_jdim = nsd_ui + jdim;

          for (int vi = 0; vi < enren_; ++vi)
          {
            const int numdof_vi = numdof * vi;
            const int nsd_vi = nsd_ * vi;

            for (int idim = 0; idim < nsd_; ++idim)
            {
              elemat2(numdof_vi + idim, 2) +=
                  estif_u(nsd_vi + idim, nsd_ui_jdim) * evelnp(jdim, uix);
            }
          }
        }
      }
    }
  }


  return 0;
}

/*---------------------------------------------------------------------------*
 | get ALE grid displacements only for element                      bk 02/15 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalcXWall<distype, enrtype>::get_grid_disp_ale(
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Core::LinAlg::Matrix<nsd_, nen_>& edispnp)
{
  my::extract_values_from_global_vector(
      discretization, lm, *my::rotsymmpbc_, &edispnp, nullptr, "dispnp");

  // add displacement when fluid nodes move in the ALE case
  // xyze_ does only know 8 nodes
  // edispnp also knows the virtual ones but doesn't do anything with them
  for (unsigned inode = 0; inode < (unsigned)enren_; ++inode)  // number of nodes
    for (int sdm = 0; sdm < nsd_; ++sdm) my::xyze_(sdm, inode) += edispnp(sdm, inode * 2);
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalcXWall<distype, enrtype>::lin_mesh_motion_3d(
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& emesh,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const double& press, const double& timefac,
    const double& timefacfac)
{
  // xGderiv_ = sum(gridx(k,i) * deriv_(j,k), k);
  // xGderiv_ == xjm_
  FOUR_C_THROW("wrong");

  return;
}


/*-----------------------------------------------------------------------------*
 | Prepare custom (direction-dependent) Gauss rule                  bk 06/2014 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalcXWall<distype, enrtype>::prepare_gauss_rule()
{
  // which is the wall-normal element direction?
  // calculate jacobian at element center
  my::is_higher_order_ele_ = false;
  my::eval_shape_func_and_derivs_at_ele_center();
  my::is_higher_order_ele_ = true;

  // the derivative of the wall distance with respect to the local coordinates
  // shows how the local axes are oriented with respect to the wall-normal vector
  Core::LinAlg::Matrix<nsd_, 1> normwallrst(Core::LinAlg::Initialization::zero);
  normwallrst.multiply(deriv_, ewdist_);
  double normwallrstnorm2 = normwallrst.norm2();
  const double dot1 = abs(normwallrst(0) / normwallrstnorm2);
  const double dot2 = abs(normwallrst(1) / normwallrstnorm2);
  const double dot3 = abs(normwallrst(2) / normwallrstnorm2);

  double minyp = 1e9;
  for (int inode = 0; inode < enren_; inode++)
  {
    double utaunode = sqrt(etauw_(inode) * densinv_);
    double yp = ewdist_(inode) * viscinv_ * utaunode;
    if (yp < minyp) minyp = yp;
  }

  if (minyp > 15.0) numgpnorm_ = numgpnormow_;

  if (distype == Core::FE::CellType::tet4)
  {
    Core::FE::GaussIntegration intpointsplane(Core::FE::CellType::tet4, 2 * numgpnorm_ - 1);
    my::intpoints_ = intpointsplane;
  }
  else  // hex8
  {
    cgp_ = std::make_shared<Core::FE::CollectedGaussPoints>(numgpnorm_ * numgpplane_ * numgpplane_);
    // get the quad9 gaussrule for the in plane integration
    Core::FE::GaussIntegration intpointsplane(Core::FE::CellType::quad8, 2 * numgpplane_ - 1);
    // get the quad9 gaussrule for the in normal integration
    Core::FE::GaussIntegration intpointsnormal(Core::FE::CellType::line3, 2 * numgpnorm_ - 1);

    // 0.9 corresponds to an angle of 25.8 deg
    if (dot1 < 0.90 && dot2 < 0.90 && dot3 < 0.90)
    {  // element, where the wall normal direction does not point in one specific element direction,
       // e.g. in corners
      cgp_->increase_reserved(
          (numgpnorm_ * numgpnorm_ * numgpnorm_) - (numgpnorm_ * numgpplane_ * numgpplane_));
      Core::FE::GaussIntegration intpointsplane(Core::FE::CellType::quad8, 2 * numgpnorm_ - 1);
      // start loop over integration points in layer
      for (Core::FE::GaussIntegration::iterator iquadplane = intpointsplane.begin();
          iquadplane != intpointsplane.end(); ++iquadplane)
      {
        // start loop over integration points in layer
        for (Core::FE::GaussIntegration::iterator iquadnorm = intpointsnormal.begin();
            iquadnorm != intpointsnormal.end(); ++iquadnorm)
        {
          cgp_->append(iquadnorm.point()[0], iquadplane.point()[0], iquadplane.point()[1],
              iquadplane.weight() * iquadnorm.weight());
        }
      }
    }
    else if (dot1 > dot2 && dot1 > dot3)
    {
      // start loop over integration points in layer
      for (Core::FE::GaussIntegration::iterator iquadplane = intpointsplane.begin();
          iquadplane != intpointsplane.end(); ++iquadplane)
      {
        // start loop over integration points in layer
        for (Core::FE::GaussIntegration::iterator iquadnorm = intpointsnormal.begin();
            iquadnorm != intpointsnormal.end(); ++iquadnorm)
        {
          cgp_->append(iquadnorm.point()[0], iquadplane.point()[0], iquadplane.point()[1],
              iquadplane.weight() * iquadnorm.weight());
        }
      }
    }
    else if (dot2 > dot3)
    {
      // start loop over integration points in layer
      for (Core::FE::GaussIntegration::iterator iquadplane = intpointsplane.begin();
          iquadplane != intpointsplane.end(); ++iquadplane)
      {
        // start loop over integration points in layer
        for (Core::FE::GaussIntegration::iterator iquadnorm = intpointsnormal.begin();
            iquadnorm != intpointsnormal.end(); ++iquadnorm)
        {
          cgp_->append(iquadplane.point()[0], iquadnorm.point()[0], iquadplane.point()[1],
              iquadplane.weight() * iquadnorm.weight());
        }
      }
    }
    else
    {
      // start loop over integration points in layer
      for (Core::FE::GaussIntegration::iterator iquadplane = intpointsplane.begin();
          iquadplane != intpointsplane.end(); ++iquadplane)
      {
        // start loop over integration points in layer
        for (Core::FE::GaussIntegration::iterator iquadnorm = intpointsnormal.begin();
            iquadnorm != intpointsnormal.end(); ++iquadnorm)
        {
          cgp_->append(iquadplane.point()[0], iquadplane.point()[1], iquadnorm.point()[0],
              iquadplane.weight() * iquadnorm.weight());
        }
      }
    }
    Core::FE::GaussIntegration grule(cgp_);
    my::intpoints_ = grule;
  }

  return;
}

template <Core::FE::CellType distype, Discret::Elements::Fluid::EnrichmentType enrtype>
void Discret::Elements::FluidEleCalcXWall<distype, enrtype>::sysmat(
    const Core::LinAlg::Matrix<nsd_, nen_>& ebofoaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& eprescpgaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& ebofon,
    const Core::LinAlg::Matrix<nsd_, nen_>& eprescpgn,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelaf, const Core::LinAlg::Matrix<nsd_, nen_>& evelam,
    const Core::LinAlg::Matrix<nsd_, nen_>& eveln, const Core::LinAlg::Matrix<nsd_, nen_>& evelnp,
    const Core::LinAlg::Matrix<nsd_, nen_>& fsevelaf, const Core::LinAlg::Matrix<nen_, 1>& fsescaaf,
    const Core::LinAlg::Matrix<nsd_, nen_>& evel_hat,
    const Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& ereynoldsstress_hat,
    const Core::LinAlg::Matrix<nen_, 1>& epreaf, const Core::LinAlg::Matrix<nen_, 1>& epream,
    const Core::LinAlg::Matrix<nen_, 1>& epren, const Core::LinAlg::Matrix<nen_, 1>& eprenp,
    const Core::LinAlg::Matrix<nsd_, nen_>& eaccam, const Core::LinAlg::Matrix<nen_, 1>& escaaf,
    const Core::LinAlg::Matrix<nen_, 1>& escaam, const Core::LinAlg::Matrix<nen_, 1>& escadtam,
    const Core::LinAlg::Matrix<nsd_, nen_>& eveldtam, const Core::LinAlg::Matrix<nen_, 1>& epredtam,
    const Core::LinAlg::Matrix<nen_, 1>& escabofoaf, const Core::LinAlg::Matrix<nen_, 1>& escabofon,
    const Core::LinAlg::Matrix<nsd_, nen_>& emhist, const Core::LinAlg::Matrix<nsd_, nen_>& edispnp,
    const Core::LinAlg::Matrix<nsd_, nen_>& egridv,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& estif,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, (nsd_ + 1) * nen_>& emesh,
    Core::LinAlg::Matrix<(nsd_ + 1) * nen_, 1>& eforce, const Core::LinAlg::Matrix<nen_, 1>& eporo,
    const Core::LinAlg::Matrix<nsd_, 2 * nen_>& egradphi,
    const Core::LinAlg::Matrix<nen_, 2 * 1>& ecurvature, const double thermpressaf,
    const double thermpressam, const double thermpressdtaf, const double thermpressdtam,
    std::shared_ptr<const Core::Mat::Material> material, double& Cs_delta_sq, double& Ci_delta_sq,
    double& Cv, bool isale, double* saccn, double* sveln, double* svelnp,
    const Core::FE::GaussIntegration& intpoints)
{
  my::sysmat(ebofoaf, eprescpgaf, ebofon, eprescpgn, evelaf, evelam, eveln, evelnp, fsevelaf,
      fsescaaf, evel_hat, ereynoldsstress_hat, epreaf, epream, epren, eprenp, eaccam, escaaf,
      escaam, escadtam, eveldtam, epredtam, escabofoaf, escabofon, emhist, edispnp, egridv, estif,
      emesh,  // -> emesh
      eforce, eporo, egradphi, ecurvature, thermpressaf, thermpressam, thermpressdtaf,
      thermpressdtam, material, Cs_delta_sq, Ci_delta_sq, Cv, isale, saccn, sveln, svelnp,
      intpoints);
  return;
}

template class Discret::Elements::FluidEleCalcXWall<Core::FE::CellType::tet4,
    Discret::Elements::Fluid::xwall>;
template class Discret::Elements::FluidEleCalcXWall<Core::FE::CellType::hex8,
    Discret::Elements::Fluid::xwall>;

FOUR_C_NAMESPACE_CLOSE

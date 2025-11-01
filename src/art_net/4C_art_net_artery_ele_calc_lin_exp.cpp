// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_art_net_artery_ele_calc_lin_exp.hpp"

#include "4C_art_net_art_junction.hpp"
#include "4C_art_net_art_terminal_bc.hpp"
#include "4C_art_net_artery_ele_calc.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_cnst_1d_art.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_of_time.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <fstream>
#include <iomanip>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ArteryEleCalcLinExp<distype>::ArteryEleCalcLinExp(
    const int numdofpernode, const std::string& disname)
    : Discret::Elements::ArteryEleCalc<distype>(numdofpernode, disname),
      qn_(),
      an_(),
      area0_(),
      th_(),
      young_(),
      pext_()
{
}

/*----------------------------------------------------------------------*
 | singleton access method                                   vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ArteryEleCalcLinExp<distype>*
Discret::Elements::ArteryEleCalcLinExp<distype>::instance(
    const int numdofpernode, const std::string& disname)
{
  using Key = std::pair<std::string, int>;
  static auto singleton_map = Core::Utils::make_singleton_map<Key>(
      [](const int numdofpernode, const std::string& disname)
      {
        return std::unique_ptr<ArteryEleCalcLinExp<distype>>(
            new ArteryEleCalcLinExp<distype>(numdofpernode, disname));
      });

  std::pair<std::string, int> key(disname, numdofpernode);

  return singleton_map[key].instance(Core::Utils::SingletonAction::create, numdofpernode, disname);
}


template <Core::FE::CellType distype>
int Discret::Elements::ArteryEleCalcLinExp<distype>::evaluate(Artery* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
    Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3,
    std::shared_ptr<Core::Mat::Material> mat)
{
  // the number of nodes
  const int numnode = my::iel_;

  // construct views
  Core::LinAlg::Matrix<2 * my::iel_, 2 * my::iel_> elemat_1(elemat1.values(), true);
  Core::LinAlg::Matrix<2 * my::iel_, 1> elevec_1(elevec1.values(), true);
  // elemat2, elevec2, and elevec3 are never used anyway

  //----------------------------------------------------------------------
  // get control parameters for time integration
  //----------------------------------------------------------------------

  // get time-step size
  const double dt = params.get<double>("time step size");

  // ---------------------------------------------------------------------
  // get control parameters for stabilization and higher-order elements
  //----------------------------------------------------------------------


  // flag for higher order elements
  //  bool higher_order_ele = ele->is_higher_order_element(distype);

  // ---------------------------------------------------------------------
  // get all general state vectors: flow./area.,
  // ---------------------------------------------------------------------

  std::shared_ptr<const Core::LinAlg::Vector<double>> qanp = discretization.get_state("qanp");
  //  std::shared_ptr<Core::LinAlg::Vector<double>> Wfnp        = discretization.GetState("Wfnp");

  if (qanp == nullptr) FOUR_C_THROW("Cannot get state vectors 'qanp'");

  // extract local values from the global vectors
  std::vector<double> myqanp = Core::FE::extract_values(*qanp, la[0].lm_);

  // create objects for element arrays
  Core::LinAlg::Matrix<numnode, 1> eareanp;
  Core::LinAlg::Matrix<numnode, 1> eqnp;
  for (int i = 0; i < numnode; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    eqnp(i) = myqanp[1 + (i * 2)];
    qn_(i) = myqanp[1 + (i * 2)];
    eareanp(i) = myqanp[0 + (i * 2)];
    an_(i) = myqanp[0 + (i * 2)];
  }
  // ---------------------------------------------------------------------
  // call routine for calculating element matrix and right hand side
  // ---------------------------------------------------------------------
  sysmat(ele, eqnp, eareanp, elemat_1, elevec_1, mat, dt);


  return 0;
}


template <Core::FE::CellType distype>
int Discret::Elements::ArteryEleCalcLinExp<distype>::evaluate_service(Artery* ele,
    const Arteries::Action action, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3, std::shared_ptr<Core::Mat::Material> mat)
{
  switch (action)
  {
    case Arteries::get_initial_artery_state:
    {
      initial(ele, params, discretization, la[0].lm_, mat);
    }
    break;
    case Arteries::set_term_bc:
    {
      evaluate_terminal_bc(ele, params, discretization, la[0].lm_, mat);
    }
    break;
    case Arteries::set_scatra_term_bc:
    {
      evaluate_scatra_bc(ele, params, discretization, la[0].lm_, mat);
    }
    break;
    case Arteries::solve_riemann_problem:
    {
      solve_riemann(ele, params, discretization, la[0].lm_, mat);
    }
    break;
    case Arteries::calc_postpro_vals:
    {
      calc_postprocessing_values(ele, params, discretization, la[0].lm_, mat);
    }
    break;
    case Arteries::calc_scatra_from_scatra_fb:
    {
      calc_scatra_from_scatra_fw(ele, params, discretization, la[0].lm_, mat);
    }
    break;
    case Arteries::evaluate_wf_wb:
    {
      evaluate_wf_and_wb(ele, params, discretization, la[0].lm_, mat);
    }
    break;
    case Arteries::evaluate_scatra_analytically:
    {
      solve_scatra_analytically(ele, params, discretization, la[0].lm_, mat);
    }
    break;
    default:
      FOUR_C_THROW("Unknown type of action {} for Artery (LinExp formulation)", action);
  }  // end of switch(act)

  return 0;
}

template <Core::FE::CellType distype>
int Discret::Elements::ArteryEleCalcLinExp<distype>::scatra_evaluate(Artery* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3, std::shared_ptr<Core::Mat::Material> mat)
{
  // the number of nodes
  const int numnode = my::iel_;

  // construct views
  Core::LinAlg::Matrix<2 * my::iel_, 2 * my::iel_> elemat_1(elemat1.values(), true);
  Core::LinAlg::Matrix<2 * my::iel_, 1> elevec_1(elevec1.values(), true);
  // elemat2, elevec2, and elevec3 are never used anyway

  //----------------------------------------------------------------------
  // get control parameters for time integration
  //----------------------------------------------------------------------

  // get time-step size
  const double dt = params.get<double>("time step size");

  // ---------------------------------------------------------------------
  // get control parameters for stabilization and higher-order elements
  //----------------------------------------------------------------------


  // flag for higher order elements
  //  bool higher_order_ele = ele->is_higher_order_element(distype);

  // ---------------------------------------------------------------------
  // get all general state vectors: flow./area.,
  // ---------------------------------------------------------------------

  std::shared_ptr<Core::LinAlg::Vector<double>> qanp =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("qanp");
  std::shared_ptr<Core::LinAlg::Vector<double>> qan =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("qan");
  std::shared_ptr<Core::LinAlg::Vector<double>> wfnp =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wfnp");
  std::shared_ptr<Core::LinAlg::Vector<double>> wbnp =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wbnp");
  std::shared_ptr<Core::LinAlg::Vector<double>> wfo =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wfo");
  std::shared_ptr<Core::LinAlg::Vector<double>> wbo =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wbo");

  std::shared_ptr<Core::LinAlg::Vector<double>> scatran =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("scatran");

  if (qanp == nullptr) FOUR_C_THROW("Cannot get state vectors 'qanp'");

  // extract local values from the global vectors
  std::vector<double> myqanp(lm.size());
  std::vector<double> myqan(lm.size());
  std::vector<double> myescatran = Core::FE::extract_values(*scatran, lm);
  //  Core::FE::extract_my_values(*qan ,myqan ,lm);

  // create objects for element arrays
  Core::LinAlg::Matrix<numnode, 1> eareanp;
  Core::LinAlg::Matrix<numnode, 1> earean;
  Core::LinAlg::Matrix<numnode, 1> eqnp;
  Core::LinAlg::Matrix<numnode, 1> eqn;
  Core::LinAlg::Matrix<2 * numnode, 1> escatran;
  Core::LinAlg::Matrix<numnode, 1> ewfnp;
  Core::LinAlg::Matrix<numnode, 1> ewbnp;
  for (int i = 0; i < numnode; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    //    eqnp(i)    = myqanp[1+(i*2)];
    //    eqn(i)     = myqan[1+(i*2)];
    //    eareanp(i) = myqanp[0+(i*2)];
    //    earean(i)     = myqan[0+(i*2)];
    escatran(2 * i) = myescatran[2 * i];
    escatran(2 * i + 1) = myescatran[2 * i + 1];
    // get element scalar transport
    int local_id = discretization.node_row_map()->lid(ele->nodes()[i]->id());
    //    escatran(i)     = (*scatran)[local_id];
    ewfnp(i) = wfnp->local_values_as_span()[local_id] - wfo->local_values_as_span()[local_id];
    ewbnp(i) = wbnp->local_values_as_span()[local_id] - wbo->local_values_as_span()[local_id];
    std::cout << "ewfnp(" << i << ") : " << ewfnp(i) << std::endl;
    std::cout << "ewbnp(" << i << ") : " << ewbnp(i) << std::endl;
  }

  // call routine for calculating element matrix and right hand side
  scatra_sysmat(ele, escatran, ewfnp, ewbnp, eareanp, earean, elemat_1, elevec_1, *mat, dt);
  return 0;
}



/*----------------------------------------------------------------------*
 |  calculate element matrix and right hand side (private)  ismail 07/09|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ArteryEleCalcLinExp<distype>::initial(Artery* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<const Core::Mat::Material> material)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> qa0 =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("qa0");
  std::shared_ptr<Core::LinAlg::Vector<double>> wfo =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("wfo");
  std::shared_ptr<Core::LinAlg::Vector<double>> wbo =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("wbo");

  Core::Nodes::Node** nodes = ele->nodes();

  int myrank = Core::Communication::my_mpi_rank(discretization.get_comm());
  if (material->material_type() == Core::Materials::m_cnst_art)
  {
    const Mat::Cnst1dArt* actmat = static_cast<const Mat::Cnst1dArt*>(material.get());
    //    std::vector<int>::iterator it = lm.begin();

    if (myrank == nodes[0]->owner())
    {
      int gid = lm[0];
      double val = std::numbers::pi * pow(actmat->diam() / 2, 2);
      qa0->replace_global_values(1, &val, &gid);
    }
    if (myrank == nodes[0]->owner())
    {
      int gid = lm[1];
      double val = 0.0;
      qa0->replace_global_values(1, &val, &gid);
    }
    if (myrank == nodes[1]->owner())
    {
      int gid = lm[2];
      double val = std::numbers::pi * pow(actmat->diam() / 2, 2);
      qa0->replace_global_values(1, &val, &gid);
    }
    if (myrank == nodes[1]->owner())
    {
      int gid = lm[3];
      double val = 0.0;
      qa0->replace_global_values(1, &val, &gid);
    }
    // Calculate Wfo and Wbo
    // Read in initial cross-sectional area at node 1
    const double Ao1 = std::numbers::pi * pow(actmat->diam() / 2, 2);
    // Read in initial cross-sectional area at node 2
    const double Ao2 = Ao1;
    // Read in blood density
    const double dens = actmat->density();
    // Read in blood viscosity
    // const double visc  = actmat->Viscosity();
    // Read in artery's thickness at node 1
    const double t1 = actmat->th();
    // Read in artery's thickness at node 2
    const double t2 = t1;
    // Read in artery's Youngs modulus of elasticity thickness at node 1
    const double E1 = actmat->young();
    // Read in artery's Youngs modulus of elasticity thickness at node 2
    const double E2 = E1;
    // Read in artery's Poisson's ratio
    const double nue = actmat->nue();
    const double co1 = sqrt(
        sqrt(std::numbers::pi) * E1 * t1 / (1.0 - pow(nue, 2)) * sqrt(Ao1) / (2.0 * Ao1 * dens));
    const double co2 = sqrt(
        sqrt(std::numbers::pi) * E2 * t2 / (1.0 - pow(nue, 2)) * sqrt(Ao2) / (2.0 * Ao2 * dens));

    int gid = ele->nodes()[0]->id();
    double val = 4.0 * co1;
    wfo->replace_global_values(1, &val, &gid);

    gid = ele->nodes()[0]->id();
    val = -4.0 * co2;
    wbo->replace_global_values(1, &val, &gid);

    gid = ele->nodes()[1]->id();
    val = 4.0 * co2;
    wfo->replace_global_values(1, &val, &gid);

    gid = ele->nodes()[1]->id();
    val = -4.0 * co2;
    wbo->replace_global_values(1, &val, &gid);
  }
  else
    FOUR_C_THROW("Material law is not an artery");
}

// ArteryEleCalcLinExp::Initial

/*----------------------------------------------------------------------*
 |  calculate element matrix and right hand side (private)  ismail 07/09|
 |                                                                      |
 |                                                                      |
 |                                                                      |
 |                               ______                                 |
 |                     _____-----      -----_____                       |
 |           _______---                          ---______              |
 | ->       / \                                         / \   ->        |
 | -->     |   |                                       |   |  -->       |
 | ---->  |     |                                     |     | ---->     |
 | ---->  |     |                                     |     | ---->     |
 | ---->  |     |                                     |     | ---->     |
 | -->     |   |                                       |   |  -->       |
 | ->       \_/_____                                ____\_/   ->        |
 |                  ---_____                _____---                    |
 |                          -----______-----                            |
 |                                                                      |
 |                                                                      |
 |                                                                      |
 |                                                                      |
 |                                                                      |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ArteryEleCalcLinExp<distype>::sysmat(Artery* ele,
    const Core::LinAlg::Matrix<my::iel_, 1>& eqnp, const Core::LinAlg::Matrix<my::iel_, 1>& eareanp,
    Core::LinAlg::Matrix<2 * my::iel_, 2 * my::iel_>& sysmat,
    Core::LinAlg::Matrix<2 * my::iel_, 1>& rhs, std::shared_ptr<const Core::Mat::Material> material,
    double dt)
{
  Core::LinAlg::Matrix<2 * my::iel_, 1> qan;
  for (int i = 0; i < my::iel_; i++)
  {
    qan(2 * i, 0) = eareanp(i);
    qan(2 * i + 1, 0) = eqnp(i);
  }
  // set element data
  const int numnode = my::iel_;
  // get node coordinates and number of elements per node
  Core::Nodes::Node** nodes = ele->nodes();
  Core::LinAlg::Matrix<3, my::iel_> xyze;
  for (int inode = 0; inode < numnode; inode++)
  {
    const auto& x = nodes[inode]->x();
    xyze(0, inode) = x[0];
    xyze(1, inode) = x[1];
    xyze(2, inode) = x[2];
  }
  rhs.clear();
  sysmat.clear();

  // Define Geometric variables
  double Ao1 = 0.0;
  double Ao2 = 0.0;
  // Define blood material variables
  double visc = 0.0;
  double dens = 0.0;
  double Kr = 0.0;
  // Define artery's material variables
  double t1 = 0.0;
  double t2 = 0.0;
  double E1 = 0.0;
  double E2 = 0.0;
  double nue = 0.0;
  // Define artery's external forces
  double pext1 = 0.0;
  double pext2 = 0.0;
  // check here, if we really have an artery !!
  if (material->material_type() == Core::Materials::m_cnst_art)
  {
    const Mat::Cnst1dArt* actmat = static_cast<const Mat::Cnst1dArt*>(material.get());
    // Read in initial cross-sectional area at node 1
    Ao1 = std::numbers::pi * pow(actmat->diam() / 2, 2);
    // Read in initial cross-sectional area at node 2
    Ao2 = Ao1;
    // Read in blood density
    dens = actmat->density();
    // Read in blood viscosity
    visc = actmat->viscosity();
    // Read in artery's thickness at node 1
    t1 = actmat->th();
    // Read in artery's thickness at node 2
    t2 = t1;
    // Read in artery's Youngs modulus of elasticity thickness at node 1
    E1 = actmat->young();
    // Read in artery's Youngs modulus of elasticity thickness at node 2
    E2 = E1;
    // Read in artery's Poisson's ratio
    nue = actmat->nue();
    // Read in artery's external forces at node 1
    pext1 = actmat->pext(0);
    // Read in artery's external forces at node 1
    pext2 = actmat->pext(1);

    // Set up all the needed vectors for further calculations
    area0_(0, 0) = Ao1;
    area0_(1, 0) = Ao2;
    th_(0, 0) = t1;
    th_(1, 0) = t2;
    young_(0, 0) = E1;
    young_(1, 0) = E2;
    pext_(0, 0) = pext1;
    pext_(1, 0) = pext2;
  }
  else
    FOUR_C_THROW("Material law is not an artery");

  // Calculate the length of artery element
  const double L = sqrt(pow(xyze(0, 0) - xyze(0, 1), 2) + pow(xyze(1, 0) - xyze(1, 1), 2) +
                        pow(xyze(2, 0) - xyze(2, 1), 2));

  // defining some redundantly used matrices

  // Defining the shape functions
  Core::LinAlg::Matrix<2 * my::iel_, 2> Nxi;
  Nxi.clear();
  // Defining the derivative of shape functions
  Core::LinAlg::Matrix<2 * my::iel_, 2> dNdxi;
  dNdxi.clear();

  Core::LinAlg::Matrix<2 * my::iel_, 1> temp1;
  Core::LinAlg::Matrix<2, 1> temp2;
  Core::LinAlg::Matrix<2 * my::iel_, 1> rhs_temp;
  rhs_temp.clear();

  Core::LinAlg::Matrix<2, 1> BLW;
  Core::LinAlg::Matrix<2, 1> FLW;
  Core::LinAlg::Matrix<2, 1> dFdxi;
  Core::LinAlg::Matrix<2, 2> H;
  Core::LinAlg::Matrix<2, 2> Bu;
  Core::LinAlg::Matrix<2, 1> B;
  Core::LinAlg::Matrix<2, 1> F;

  // Defining essential variables at the Gauss points
  double th;
  double Young;
  double beta;
  double Q;
  double A;
  double Ao;

  // Defining essential derivatives at the Gauss points
  double dbeta_dxi;
  double dAodxi;
  double dQdxi;
  double dAdxi;
  double dpext_dxi;

  //--------------------------------------------------------------
  //               Calculate the System Matrix
  //--------------------------------------------------------------
  //
  /*
    In the case of the linear elastic material behavior of arteries,
    the system matrix is the same as the mass matrix, and thus it
    could be derived analytically.

                    +-    .      .      .     -+
                    |   2 .      .      .      |
                    |  N  .   0  . N  N .  0   |
                 _1 |   1 .      .  1  2.      |
                /   |..........................|
               |    |     .   2  .      .      |
               |    |  0  .  N   .   0  .N  N  |
               |    |     .   1  .      . 1  2 |  par s
    MassMat =  |    |..........................|  ------ dxi
               |    |     .      .   2  .      |  par xi
               |    |N  N .   0  .  N   .  0   |
               |    | 2  1.      .   2  .      |
             _/     |..........................|
           -1       |     .      .      .   2  |
                    |  0  . N  N .   0  .  N   |
                    |     .  2  1.      .   2  |
                    +-                        -+


                    +-    .      .      .     -+
                    |  2  .      .   1  .      |
                    | --- .   0  .  --- .  0   |
                    |  3  .      .   3  .      |
                    |..........................|
                    |     .  2   .      .  1   |
                    |  0  . ---  .   0  . ---  |
                    |     .  3   .      .  3   |    L
    MassMat =       |..........................| * ---
                    |  1  .      .   2  .      |    2
                    | --- .   0  .  --- .  0   |
                    |  3  .      .   3  .      |
                    |..........................|
                    |     .  1   .      .  2   |
                    |  0  . ---  .   0  . ---  |
                    |     .  3   .      .  3   |
                    +-                        -+

   */
  sysmat(0, 0) = L / 3.0;
  sysmat(0, 1) = 0.0;
  sysmat(0, 2) = L / 6.0;
  sysmat(0, 3) = 0.0;
  sysmat(1, 0) = 0.0;
  sysmat(1, 1) = L / 3.0;
  sysmat(1, 2) = 0.0;
  sysmat(1, 3) = L / 6.0;
  sysmat(2, 0) = L / 6.0;
  sysmat(2, 1) = 0.0;
  sysmat(2, 2) = L / 3.0;
  sysmat(2, 3) = 0.0;
  sysmat(3, 0) = 0.0;
  sysmat(3, 1) = L / 6.0;
  sysmat(3, 2) = 0.0;
  sysmat(3, 3) = L / 3.0;


  // gaussian points
  const Core::FE::IntegrationPoints1D intpoints(ele->gauss_rule());

  // integration loop

  for (int iquad = 0; iquad < intpoints.nquad; ++iquad)
  {
    // coordinates of the current integration point
    const double xi = intpoints.qxg[iquad][0];
    const double wgt = intpoints.qwgt[iquad];

    // shape functions and their derivatives
    Core::FE::shape_function_1d(my::funct_, xi, distype);
    Core::FE::shape_function_1d_deriv1(my::deriv_, xi, distype);

    // get Jacobian matrix and determinant
    // actually compute its transpose....
    /*
                             _____________________________________
        ds     L            /         2            2            2
        --- = ---   ;  L = / ( x - x )  + ( y - y )  + ( z - z )
        dxi    2          v     1   2        2    2       1   2

    */

    my::xjm_ = L / 2.0;
    //
    for (int r = 0; r < my::iel_; r++)
    {
      my::tderiv_(r) = my::deriv_(r);
      dNdxi(2 * r, 0) = my::deriv_(r);
      dNdxi(2 * r + 1, 1) = my::deriv_(r);
      Nxi(2 * r, 0) = my::funct_(r);
      Nxi(2 * r + 1, 1) = my::funct_(r);
    }
    // Calculating essential variables at the Gauss points
    th = my::funct_.dot(th_);
    Young = my::funct_.dot(young_);
    beta = sqrt(std::numbers::pi) * Young * th / (1.0 - pow(nue, 2));
    Q = my::funct_.dot(qn_);
    A = my::funct_.dot(an_);
    Ao = my::funct_.dot(area0_);
    // Calculating essential derivatives at the Gauss points
    dbeta_dxi = sqrt(std::numbers::pi) / (1.0 - pow(nue, 2)) *
                (th * my::tderiv_.dot(young_) + my::tderiv_.dot(th_) * Young);
    dAodxi = my::tderiv_.dot(area0_);
    dQdxi = my::tderiv_.dot(qn_);
    dAdxi = my::tderiv_.dot(an_);
    dpext_dxi = my::tderiv_.dot(pext_);

    //--------------------------------------------------------------
    //                   compute the rhs vector
    //--------------------------------------------------------------
    /*
       Compute the rhs of the linear elastic artery with a
       Taylor-Galerkin skeme for the nonlinear diff. equation

         1- Since this element is explicitly solved in time domain
            the results from time step "n" will only be used to compute
            the results from time step "n+1"

            /   \
         2- |.,.| is the Lebesgue inner product
            \   /

         3- Psi is the weight function for the Galerkin approximation




                        n                     n                         n
            /          \        /            \      2 /                \
        n   |          |        |       dPsi |    Dt  |    par F       |
  o  rhs =  | U  , Psi |   + Dt | F   , ---- |  - --- | B  ----- , Psi |
            |          |        |  LW    ds  |     2  |  U par s       |
            \          /        \            /        \                /
           +------------+    +----------------+ +-----------------------+
              (Term 1)            (Term 2)             (Term 3)

                                   n                    n
               2 /                \        /           \
             Dt  |   par F   dPsi |        |           |
           - --- | H ----- , ---- |   + Dt | B   , Psi |
              2  |   par s    ds  |        |  LW       |
                 \                /        \           /
          +-------------------------+   +----------------+
                  (Term 4)                  (Term 5)



                                +-     -+
          +- -+                 | psi   |
          | A |                 |    a  |
  o U   = |   |     ;     Psi = |       |
          | Q |                 | psi   |
          +- -+                 |    q  |
                                +-     -+


               Dt
  o F   = F + --- H B
     LW        2

               Dt
  o B   = B + --- B  B
     LW        2   U


        +-                  -+
        |                    |
        |         Q          |
        |                    |
  o F = |....................|
        |  2                 |
        | Q      beta    3/2 |
        |--- + -------- A    |
        | A    3 rho Ao      |
        +-                  -+


        +-                                                -+
        |                                                  |
        |                         0                        |
    dF  |                                                  |
  o -- =|..................................................|
    ds  |             /      2                  \          |
        |   Q   dQ    |  / Q \      beta    1/2 | dA       |
        | 2---.---  + | -|---|  + -------- A    | --- +    |
        |   A   ds    \  \ A /    2.rho.Ao      / ds       |
        |                                                  |
        |                  3/2  /                     \    |
        |                 A     | d beta    beta  dAo |    |
        |           +  -------- | ------ - -----  --- |    |
        |              3.rho.Ao \ d x        Ao   ds  /    |
        +-                                                -+


        +-                                                          -+
        |                                                            |
        |                            0                               |
        |                                                            |
  o B = |............................................................|
        |                                                            |
        |     Q        A    / 2   1/2    1/2 \ par beta              |
        |- Kr---  -  ------ |--- A    - Ao   | --------              |
        |     A      Ao rho \ 3              /  par s                |
        |                                                            |
        |                                                            |
        |             beta   A  / 2  1/2     1  1/2 \ par Ao         |
        |         +  ------ --- |---A     - ---Ao   | ------         |
        |            Ao rho  Ao \ 3          2      /  par s         |
        |                                                            |
        |                                                            |
        |              A    par Pext                                 |
        |         -  -----.---------                                 |
        |             rho    par s                                   |
        |                                                            |
        +-                                                          -+


        +-                                                   .      -+
        |                                                    .       |
        |                            0                       .   0   |
    dB  |                                                    .       |
  o -- =|............................................................|
    dU  |                                                    .       |
        |     Q        1    /  1/2    1/2 \ par beta         .   Kr  |
        |- Kr---  -  ------ | A    - Ao   | --------         . - --- |
        |     A^2    Ao rho \             /  par s           .    A  |
        |                                                    .       |
        |                                                    .       |
        |             beta   1  /  1/2    1  1/2 \ par Ao    .       |
        |         +  ------ --- | A    - -- Ao   | ------    .       |
        |            Ao rho  Ao \         2      /  par s    .       |
        |                                                    .       |
        |                                                    .       |
        |              1     par Pext                        .       |
        |         -  -----. ---------                        .       |
        |             rho     par s                          .       |
        |                                                    .       |
        +-                                                          -+


        +-                           .        -+
        |                            .         |
        |             0              .   1     |
  o H = |......................................|
        |                            .         |
        |        2                   .         |
        |   / Q \       beta    1/2  .     Q   |
        | - |---|  +  -------- A     .  2 ---  |
        |   \ A /     2 rho Ao       .     A   |
        |                            .         |
        +-                           .        -+


    */
    // Calculate Kr
    Kr = 8.0 * std::numbers::pi * visc / dens;

    // Calculate H
    H(0, 0) = 0.0;
    H(0, 1) = 1.0;
    H(1, 0) = -pow(Q / A, 2) + beta / (2.0 * dens * Ao) * sqrt(A);
    H(1, 1) = 2.0 * Q / A;

    // Calculating F
    F(0, 0) = Q;
    F(1, 0) = pow(Q, 2) / A + beta / (3.0 * dens * Ao) * pow(A, 1.5);

    // Calculating B
    B(0, 0) = 0.0;
    B(1, 0) =
        -Kr * Q / A - A / (Ao * dens) * (2.0 / 3.0 * sqrt(A) - sqrt(Ao)) * dbeta_dxi * 2.0 / L +
        beta * A / (dens * pow(Ao, 2)) * (2.0 / 3.0 * sqrt(A) - 0.5 * sqrt(Ao)) * dAodxi * 2.0 / L -
        A * dpext_dxi / dens * 2.0 / L;

    // Calculating Bu
    Bu(0, 0) = 0.0;
    Bu(0, 1) = 0.0;
    Bu(1, 0) = Kr * Q / pow(A, 2) - 1.0 / (Ao * dens) * (sqrt(A) - sqrt(Ao)) * dbeta_dxi * 2.0 / L +
               beta / (pow(Ao, 2) * dens) * (sqrt(A) - 0.5 * sqrt(Ao)) * dAodxi * 2.0 / L -
               dpext_dxi / dens * 2.0 / L;
    Bu(1, 1) = -Kr / A;

    // Calculating dF/dxi
    dFdxi(0, 0) = dQdxi;
    dFdxi(1, 0) = 2.0 * Q / A * dQdxi +
                  (-pow(Q / A, 2) + beta / (2.0 * dens * Ao) * sqrt(A)) * dAdxi +
                  pow(A, 1.5) / (3.0 * dens * Ao) * (dbeta_dxi - beta / Ao * dAodxi);

    // Calculating FLW
    FLW.multiply(dt / 2.0, H, B);
    FLW += F;

    // Calculating BLW
    BLW.multiply(dt / 2.0, Bu, B);
    BLW += B;

    // Term 1 is constant and can be evaluated analytically, therefore it will
    // be added in the end

    // Adding the contribution of Term 2
    rhs_temp.multiply(dNdxi, FLW);
    rhs_temp.scale(dt);


    // Adding the contribution of Term 3
    temp2.multiply(Bu, dFdxi);
    temp2.scale(-0.5 * pow(dt, 2));
    temp1.multiply(Nxi, temp2);
    rhs_temp += temp1;

    // Adding the contribution of Term 4
    temp2.multiply(H, dFdxi);
    temp2.scale(-pow(dt, 2) / L);
    temp1.multiply(dNdxi, temp2);
    rhs_temp += temp1;

    // Adding the contribution of Term 5
    temp1.multiply(Nxi, BLW);
    temp1.scale(0.5 * dt * L);
    rhs_temp += temp1;

    // Final Addition
    rhs_temp.scale(wgt);
    rhs += rhs_temp;
  }  // loop gausspoints

  temp1.multiply(sysmat, qan);
  rhs += temp1;
}


template <Core::FE::CellType distype>
void Discret::Elements::ArteryEleCalcLinExp<distype>::scatra_sysmat(Artery* ele,
    const Core::LinAlg::Matrix<2 * my::iel_, 1>& escatran,
    const Core::LinAlg::Matrix<my::iel_, 1>& ewfnp, const Core::LinAlg::Matrix<my::iel_, 1>& ewbnp,
    const Core::LinAlg::Matrix<my::iel_, 1>& eareanp,
    const Core::LinAlg::Matrix<my::iel_, 1>& earean,
    Core::LinAlg::Matrix<2 * my::iel_, 2 * my::iel_>& sysmat,
    Core::LinAlg::Matrix<2 * my::iel_, 1>& rhs, const Core::Mat::Material& material, double dt)
{
  // get the nodal coordinates of the element
  Core::Nodes::Node** nodes = ele->nodes();
  Core::LinAlg::Matrix<3, my::iel_> xyze;
  for (int inode = 0; inode < my::iel_; inode++)
  {
    const auto& x = nodes[inode]->x();
    xyze(0, inode) = x[0];
    xyze(1, inode) = x[1];
    xyze(2, inode) = x[2];
  }
  // Evaluate element length
  const double L = sqrt(pow(xyze(0, 0) - xyze(0, 1), 2) + pow(xyze(1, 0) - xyze(1, 1), 2) +
                        pow(xyze(2, 0) - xyze(2, 1), 2));


  // evaluate CFL number
  double f_cfl_np = 0.5 * (0.5 * ewfnp(1) + 0.5 * ewfnp(0)) * dt / L;
  double b_cfl_np = 0.5 * (0.5 * ewbnp(1) + 0.5 * ewbnp(0)) * dt / L;

  // Evaluate the system mass matrix
  sysmat(0, 0) = 0.5 + f_cfl_np;
  sysmat(0, 2) = 0.5 - f_cfl_np;
  sysmat(1, 1) = 0.5 + b_cfl_np;
  sysmat(1, 3) = 0.5 - b_cfl_np;
  sysmat(2, 0) = -0.5 - f_cfl_np;
  sysmat(2, 2) = -0.5 + f_cfl_np;
  sysmat(3, 1) = -0.5 - b_cfl_np;
  sysmat(3, 3) = -0.5 + b_cfl_np;

  // Evaluate rhs vector
  rhs(0) = 0.5 * (escatran(2) + escatran(0));
  rhs(1) = 0.5 * (escatran(3) + escatran(1));
  rhs(2) = -0.5 * (escatran(2) + escatran(0));
  rhs(3) = -0.5 * (escatran(3) + escatran(1));
}


static const Core::Conditions::Condition* get_first_condition(
    const Core::FE::Discretization& discretization, const Core::Nodes::Node* node,
    const std::string& name)
{
  auto conds = discretization.get_conditions_on_node(name, node);
  return conds.empty() ? nullptr : conds.front();
}


/*----------------------------------------------------------------------*
 |  Solve the Riemann problem at the terminal elements      ismail 07/09|
 |                                                                      |
 |    1- Check whether any of the element nodes has a condition         |
 |    2- If a condition exists on the first node (i.e it is an inlet):  |
 |           Evaluate the backward characteristics wave speed           |
 |    3- If a condition exists on the second node (i.e it is an outlet):|
 |           Evaluate the forward characteristics wave speed            |
 |    4- If no conditions exist the nodes are not boundary nodes and    |
 |       no Riemann solution is required                                |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Discret::Elements::ArteryEleCalcLinExp<distype>::solve_riemann(Artery* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<const Core::Mat::Material> material)
{
  // Define Geometric variables
  double Ao1 = 0.0;
  double Ao2 = 0.0;
  // Define blood material variables
  //  double visc = 0.0;
  double dens = 0.0;
  // Define artery's material variables
  double t1 = 0.0;
  double t2 = 0.0;
  double E1 = 0.0;
  double E2 = 0.0;
  double nue = 0.0;
  // Define artery's external forces
  double pext1 = 0.0;
  double pext2 = 0.0;
  // check here, if we really have an artery !!
  if (material->material_type() == Core::Materials::m_cnst_art)
  {
    const Mat::Cnst1dArt* actmat = static_cast<const Mat::Cnst1dArt*>(material.get());
    // Read in initial cross-sectional area at node 1
    Ao1 = std::numbers::pi * pow(actmat->diam() / 2, 2);
    // Read in initial cross-sectional area at node 2
    Ao2 = Ao1;
    // Read in blood density
    dens = actmat->density();
    // Read in blood viscosity
    //    visc   = actmat->Viscosity();
    // Read in artery's thickness at node 1
    t1 = actmat->th();
    // Read in artery's thickness at node 2
    t2 = t1;
    // Read in artery's Youngs modulus of elasticity thickness at node 1
    E1 = actmat->young();
    // Read in artery's Youngs modulus of elasticity thickness at node 2
    E2 = E1;
    // Read in artery's Poisson's ratio
    nue = actmat->nue();
    // Read in artery's external forces at node 1
    pext1 = actmat->pext(0);
    // Read in artery's external forces at node 1
    pext2 = actmat->pext(1);

    // Set up all the needed vectors for further calculations
    area0_(0, 0) = Ao1;
    area0_(1, 0) = Ao2;
    th_(0, 0) = t1;
    th_(1, 0) = t2;
    young_(0, 0) = E1;
    young_(1, 0) = E2;
    pext_(0, 0) = pext1;
    pext_(1, 0) = pext2;
  }
  else
    FOUR_C_THROW("Material law is not an artery");



  // the number of nodes
  const int numnode = my::iel_;

  std::shared_ptr<const Core::LinAlg::Vector<double>> qanp = discretization.get_state("qanp");
  std::shared_ptr<Core::LinAlg::Vector<double>> Wfnp =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wfnp");
  std::shared_ptr<Core::LinAlg::Vector<double>> Wbnp =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wbnp");

  if (qanp == nullptr) FOUR_C_THROW("Cannot get state vectors 'qanp'");

  // extract local values from the global vectors
  std::vector<double> myqanp = Core::FE::extract_values(*qanp, lm);

  // create objects for element arrays
  Core::LinAlg::Matrix<numnode, 1> earean;
  Core::LinAlg::Matrix<numnode, 1> eqn;

  // get time step size
  const double dt = params.get<double>("time step size");

  // get all values at the last computed time step
  for (int i = 0; i < numnode; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    eqn(i) = myqanp[1 + (i * 2)];
    qn_(i) = myqanp[1 + (i * 2)];
    earean(i) = myqanp[0 + (i * 2)];
    an_(i) = myqanp[0 + (i * 2)];
  }

  // get the nodal coordinates of the element
  Core::Nodes::Node** nodes = ele->nodes();
  Core::LinAlg::Matrix<3, my::iel_> xyze;
  for (int inode = 0; inode < my::iel_; inode++)
  {
    const auto& x = nodes[inode]->x();
    xyze(0, inode) = x[0];
    xyze(1, inode) = x[1];
    xyze(2, inode) = x[2];
  }
  const double L = sqrt(pow(xyze(0, 0) - xyze(0, 1), 2) + pow(xyze(1, 0) - xyze(1, 1), 2) +
                        pow(xyze(2, 0) - xyze(2, 1), 2));
  bool BCnodes = false;

  // get the number of nodes per element
  const int numnds = ele->num_node();

  if (numnds != 2) FOUR_C_THROW("An element with {} nodes is not supported", numnds);

  // check for the CFL number CFL = Max(abs(3/sqrt(3) * lambda2_i * dt/dx), abs(3/sqrt(3) *
  // lambda1_i * dt/dx))
  double c_0 = sqrt(sqrt(std::numbers::pi) * young_(0) * th_(0) / (1.0 - pow(nue, 2)) *
                    sqrt(earean(0)) / (2.0 * area0_(0) * dens));
  double c_1 = sqrt(sqrt(std::numbers::pi) * young_(1) * th_(1) / (1.0 - pow(nue, 2)) *
                    sqrt(earean(1)) / (2.0 * area0_(1) * dens));
  double lambda2_0 = eqn(0) / earean(0) - c_0;
  double lambda2_1 = eqn(1) / earean(1) - c_1;
  double lambda1_0 = eqn(0) / earean(0) + c_0;
  double lambda1_1 = eqn(1) / earean(1) + c_1;
  double lambda_max = fabs(lambda2_0);
  if (lambda_max < fabs(lambda2_1)) lambda_max = fabs(lambda2_1);
  if (lambda_max < fabs(lambda1_0)) lambda_max = fabs(lambda1_0);
  if (lambda_max < fabs(lambda1_1)) lambda_max = fabs(lambda1_1);

  if (3.0 / sqrt(3.0) * lambda_max * dt / L >= 1.0)
  {
    FOUR_C_THROW(
        "CFL number at element {} is {}", ele->id(), 3.0 / sqrt(3.0) * lambda_max * dt / L);
  }

  // Solve Riemann problem at the terminals
  // loop over the terminal nodes
  for (int i = 0; i < 2; i++)
  {
    if (get_first_condition(discretization, ele->nodes()[i], "ArtInOutCond"))
    {
      double TermIO = 0.0;
      // Get the in/out terminal condition
      std::string TerminalType =
          get_first_condition(discretization, ele->nodes()[i], "ArtInOutCond")
              ->parameters()
              .get<std::string>("terminaltype");
      if (TerminalType == "inlet")
        TermIO = -1.0;
      else if (TerminalType == "outlet")
        TermIO = 1.0;
      else
        FOUR_C_THROW(
            "Something is severely wrong! In/Out terminal condition should be either \"outlet\" or "
            "\"inlet\"");

      // sound speed at node 1 = sqrt(beta/(2*Ao*rho)) and Lambda2 = Q/A - c
      const double c = sqrt(sqrt(std::numbers::pi) * young_(i) * th_(i) / (1.0 - pow(nue, 2)) *
                            sqrt(earean(i)) / (2.0 * area0_(i) * dens));
      const double lambda = eqn(i) / earean(i) + TermIO * c;
      //      const double N1     = (0.5*(-TermIO + 1.0)*L + dt*lambda)/L;
      const double N1 = (-TermIO * dt * lambda) / L;
      //      const double N2     = (0.5*( TermIO + 1.0)*L - dt*lambda)/L;
      const double N2 = (L + TermIO * dt * lambda) / L;
      //      const double A_l    = N1*earean(0) + N2*earean(1);
      const double A_l = N1 * earean(i) + N2 * earean((i + 1) % 2);
      //      const double beta_l = sqrt(PI)*(young_(0)*N1 + young_(1)*N2)*(th_(0)*N1 +
      //      th_(1)*N2)/(1.0-pow(nue,2));
      const double beta_l = sqrt(std::numbers::pi) * (young_(i) * N1 + young_((i + 1) % 2) * N2) *
                            (th_(i) * N1 + th_((i + 1) % 2) * N2) / (1.0 - pow(nue, 2));
      //      const double Q_l    = N1*eqn(0)    + N2*eqn(1);
      const double Q_l = N1 * eqn(i) + N2 * eqn((i + 1) % 2);
      //      const double Ao_l   = N1*area0_(0) + N2*area0_(1);
      const double Ao_l = N1 * area0_(i) + N2 * area0_((i + 1) % 2);
      const double c_l = sqrt(beta_l * sqrt(A_l) / (2.0 * Ao_l * dens));

      // defining W2n at dt*lambda2
      const double Wn_l = Q_l / A_l + TermIO * 4.0 * c_l;
      const double Won_l = TermIO * 4.0 * sqrt(beta_l * sqrt(Ao_l) / (2.0 * Ao_l * dens));
      const double co = sqrt(sqrt(std::numbers::pi) * young_(i) * th_(i) / (1.0 - pow(nue, 2)) *
                             sqrt(area0_(i)) / (2.0 * area0_(i) * dens));

      double Wnp = Wn_l - Won_l + TermIO * 4.0 * co;

      // ---------------------------------------------------------------------------
      // Modify the global backkward characteristics speeds vector
      // ---------------------------------------------------------------------------
      int myrank = Core::Communication::my_mpi_rank(discretization.get_comm());
      if (myrank == ele->nodes()[i]->owner())
      {
        int gid = ele->nodes()[i]->id();
        double val = Wnp;
        if (TermIO == -1.0)
          Wbnp->replace_global_values(1, &val, &gid);
        else if (TermIO == 1.0)
          Wfnp->replace_global_values(1, &val, &gid);
      }

      // -----------------------------------------------------------------------------
      // Update the information needed for solving the junctions
      // -----------------------------------------------------------------------------
      if (get_first_condition(discretization, ele->nodes()[i], "ArtJunctionCond"))
      {
        // Update the characteristic wave speed
        std::shared_ptr<std::map<const int, std::shared_ptr<Arteries::Utils::JunctionNodeParams>>>
            junc_nodal_vals = params.get<std::shared_ptr<
                std::map<const int, std::shared_ptr<Arteries::Utils::JunctionNodeParams>>>>(
                "Junctions Parameters");

        int local_id = discretization.node_row_map()->lid(ele->nodes()[i]->id());
        if (local_id < 0)
        {
          FOUR_C_THROW("node ({}) doesn't exist on proc({})", ele->nodes()[i]->id(),
              Core::Communication::my_mpi_rank(discretization.get_comm()));
        }
        if (TermIO == -1.0)
          (*junc_nodal_vals)[local_id]->W_ = Wbnp->local_values_as_span()[local_id];
        else if (TermIO == 1.0)
          (*junc_nodal_vals)[local_id]->W_ = Wfnp->local_values_as_span()[local_id];
        (*junc_nodal_vals)[local_id]->A_ = earean(i);
        (*junc_nodal_vals)[local_id]->Q_ = eqn(i);
        (*junc_nodal_vals)[local_id]->Ao_ = area0_(i);
        (*junc_nodal_vals)[local_id]->rho_ = dens;
        (*junc_nodal_vals)[local_id]->Pext_ = pext_(i);
        (*junc_nodal_vals)[local_id]->beta_ =
            sqrt(std::numbers::pi) * young_(i) * th_(i) / (1.0 - pow(nue, 2));
      }

      BCnodes = true;
    }
  }

  return BCnodes;
}

/*----------------------------------------------------------------------*
 |  Evaluate the values of the degrees of freedom           ismail 07/09|
 |  at terminal nodes.                                                  |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ArteryEleCalcLinExp<distype>::evaluate_terminal_bc(Artery* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material> material)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> Wfnp =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wfnp");
  std::shared_ptr<Core::LinAlg::Vector<double>> Wbnp =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wbnp");

  // get time-step size
  const double dt = params.get<double>("time step size");

  // check here, if we really have an artery !!
  // Define Geometric variables
  double Ao1 = 0.0;
  double Ao2 = 0.0;
  // Define blood material variables
  //  double visc=0.0;
  double dens = 0.0;
  // Define artery's material variables
  double t1 = 0.0;
  double t2 = 0.0;
  double E1 = 0.0;
  double E2 = 0.0;
  double nue = 0.0;
  // Define artery's external forces
  double pext1 = 0.0;
  double pext2 = 0.0;
  // check here, if we really have an artery !!
  if (material->material_type() == Core::Materials::m_cnst_art)
  {
    const Mat::Cnst1dArt* actmat = static_cast<const Mat::Cnst1dArt*>(material.get());
    // Read in initial cross-sectional area at node 1
    Ao1 = std::numbers::pi * pow(actmat->diam() / 2, 2);
    // Read in initial cross-sectional area at node 2
    Ao2 = Ao1;
    // Read in blood density
    dens = actmat->density();
    // Read in blood viscosity
    //    visc   = actmat->Viscosity();
    // Read in artery's thickness at node 1
    t1 = actmat->th();
    // Read in artery's thickness at node 2
    t2 = t1;
    // Read in artery's Youngs modulus of elasticity thickness at node 1
    E1 = actmat->young();
    // Read in artery's Youngs modulus of elasticity thickness at node 2
    E2 = E1;
    // Read in artery's Poisson's ratio
    nue = actmat->nue();
    // Read in artery's external forces at node 1
    pext1 = actmat->pext(0);
    // Read in artery's external forces at node 1
    pext2 = actmat->pext(1);

    // Set up all the needed vectors for further calculations
    area0_(0, 0) = Ao1;
    area0_(1, 0) = Ao2;
    th_(0, 0) = t1;
    th_(1, 0) = t2;
    young_(0, 0) = E1;
    young_(1, 0) = E2;
    pext_(0, 0) = pext1;
    pext_(1, 0) = pext2;
  }
  else
    FOUR_C_THROW("Material law is not an artery");


  // the number of nodes
  const int numnode = my::iel_;

  std::shared_ptr<const Core::LinAlg::Vector<double>> qanp = discretization.get_state("qanp");

  if (qanp == nullptr) FOUR_C_THROW("Cannot get state vectors 'qanp'");

  // extract local values from the global vectors
  std::vector<double> myqanp = Core::FE::extract_values(*qanp, lm);

  // create objects for element arrays
  Core::LinAlg::Matrix<numnode, 1> eareanp;
  Core::LinAlg::Matrix<numnode, 1> eqnp;

  // get time step size
  //  const double dt = params.get<double>("time step size");

  // get all values at the last computed time step
  for (int i = 0; i < numnode; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    eqnp(i) = myqanp[1 + (i * 2)];
    qn_(i) = myqanp[1 + (i * 2)];
    eareanp(i) = myqanp[0 + (i * 2)];
    an_(i) = myqanp[0 + (i * 2)];
  }

  // ---------------------------------------------------------------------------------
  // Resolve the BC at the inlet and outlet
  // ---------------------------------------------------------------------------------
  // IO_BC_HERE
  for (int i = 0; i < 2; i++)
  {
    if (get_first_condition(discretization, ele->nodes()[i], "ArtInOutCond"))
    {
      double TermIO = 0.0;
      // Get the in/out terminal condition
      std::string TerminalType =
          (get_first_condition(discretization, ele->nodes()[i], "ArtInOutCond")
                  ->parameters()
                  .get<std::string>("terminaltype"));
      if (TerminalType == "inlet")
        TermIO = -1.0;
      else if (TerminalType == "outlet")
        TermIO = 1.0;
      else
        FOUR_C_THROW(
            "Something is severely wrong! In/Out terminal condition should be either \"outlet\" or "
            "\"inlet\"");

      std::shared_ptr<Core::LinAlg::Vector<double>> bcval =
          params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("bcval");
      std::shared_ptr<Core::LinAlg::Vector<double>> dbctog =
          params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("dbctog");

      if (bcval == nullptr || dbctog == nullptr)
        FOUR_C_THROW("Cannot get state vectors 'bcval' and 'dbctog'");


      th_(0, 0) = t1;
      th_(1, 0) = t2;
      young_(0, 0) = E1;
      young_(1, 0) = E2;
      const double beta = sqrt(std::numbers::pi) * young_(i) * th_(i) / (1.0 - nue * nue);
      double Wf, Wb;

      // -----------------------------------------------------------------------------
      // fill the required parameters to solve the inlet BC
      // -----------------------------------------------------------------------------
      Teuchos::ParameterList Cparams;
      Cparams.set<int>("in out flag", int(TermIO));
      Cparams.set<double>("total time", params.get<double>("total time"));
      Cparams.set<double>("artery beta", beta);
      Cparams.set<double>("artery area", area0_(i));
      Cparams.set<double>("blood density", dens);
      int local_id = discretization.node_row_map()->lid(ele->nodes()[i]->id());
      if (local_id < 0)
      {
        FOUR_C_THROW("node ({}) doesn't exist on proc({})", ele->nodes()[i]->id(),
            Core::Communication::my_mpi_rank(discretization.get_comm()));
      }

      if (TermIO == -1)
      {
        Cparams.set<double>(
            "backward characteristic wave speed", Wbnp->local_values_as_span()[local_id]);
      }
      else
      {
        Cparams.set<double>(
            "forward characteristic wave speed", Wfnp->local_values_as_span()[local_id]);
      }
      Cparams.set<double>("external pressure", pext_(i));

      // -----------------------------------------------------------------------------
      // Solve any possible prescribed boundary condition
      // -----------------------------------------------------------------------------
      if (get_first_condition(discretization, ele->nodes()[i], "ArtPrescribedCond"))
      {
        const Core::Conditions::Condition* condition =
            get_first_condition(discretization, ele->nodes()[i], "ArtPrescribedCond");
        Cparams.set<std::string>("Condition Name", "ArtPrescribedCond");
        Arteries::Utils::solve_prescribed_terminal_bc(discretization, condition, Cparams);
      }

      // -----------------------------------------------------------------------------
      // Solve any possible 3-D/reduced-D coupled boundary condition
      // -----------------------------------------------------------------------------
      if (get_first_condition(discretization, ele->nodes()[i], "Art_redD_3D_CouplingCond"))
      {
        std::shared_ptr<Teuchos::ParameterList> CoupledTo3DParams =
            params.get<std::shared_ptr<Teuchos::ParameterList>>("coupling with 3D fluid params");
        const Core::Conditions::Condition* condition =
            get_first_condition(discretization, ele->nodes()[i], "Art_redD_3D_CouplingCond");
        Cparams.set<std::shared_ptr<Teuchos::ParameterList>>(
            "coupling with 3D fluid params", CoupledTo3DParams);
        Cparams.set<std::string>("Condition Name", "Art_redD_3D_CouplingCond");

        Arteries::Utils::solve_prescribed_terminal_bc(discretization, condition, Cparams);
      }

      // -----------------------------------------------------------------------------
      // Solve any possible reflection boundary condition
      // -----------------------------------------------------------------------------
      if (get_first_condition(discretization, ele->nodes()[i], "ArtRfCond"))
      {
        const Core::Conditions::Condition* condition =
            get_first_condition(discretization, ele->nodes()[i], "ArtRfCond");
        Arteries::Utils::solve_reflective_terminal(discretization, condition, Cparams);
      }

      // -----------------------------------------------------------------------------
      // Solve any possible windkessel boundary condition
      // -----------------------------------------------------------------------------
      if (get_first_condition(discretization, ele->nodes()[i], "ArtWkCond"))
      {
        const Core::Conditions::Condition* condition =
            get_first_condition(discretization, ele->nodes()[i], "ArtWkCond");
        Cparams.set<double>("time step size", dt);
        Cparams.set<double>("external pressure", pext_(i));
        Cparams.set<double>("terminal volumetric flow rate", qn_(i));
        Cparams.set<double>("terminal cross-sectional area", an_(i));
        Arteries::Utils::solve_expl_windkessel_bc(discretization, condition, Cparams);
      }

      // -----------------------------------------------------------------------------
      // break the for loop if the boundary condition is a junction,
      // since it will be solved later
      // -----------------------------------------------------------------------------
      if (get_first_condition(discretization, ele->nodes()[i], "ArtJunctionCond") == nullptr)
      {
        Wf = Cparams.get<double>("forward characteristic wave speed");
        Wb = Cparams.get<double>("backward characteristic wave speed");

        // -----------------------------------------------------------------------------
        // Modify the global forward and backward characteristics speeds vector
        // -----------------------------------------------------------------------------
        int myrank = Core::Communication::my_mpi_rank(discretization.get_comm());
        if (myrank == ele->nodes()[i]->owner())
        {
          int gid = ele->nodes()[i]->id();
          if (TermIO == -1.0)
          {
            double val1 = Wf;
            Wfnp->replace_global_values(1, &val1, &gid);
          }
          else
          {
            double val2 = Wb;
            Wbnp->replace_global_values(1, &val2, &gid);
            int local_id = discretization.node_row_map()->lid(ele->nodes()[i]->id());
            Wf = Wfnp->local_values_as_span()[local_id];
          }
        }

        // calculating A at node i
        int gid;
        double val;
        double cross_area;

        cross_area = std::pow(2.0 * dens * area0_(i) / beta, 2.0) * pow((Wf - Wb) / 8.0, 4.0);

        gid = lm[2 * i];
        val = cross_area;
        bcval->replace_global_values(1, &val, &gid);

        gid = lm[2 * i];
        val = 1;
        dbctog->replace_global_values(1, &val, &gid);

        // calculating Q at node i
        gid = lm[2 * i + 1];
        val = (cross_area) * (Wf + Wb) / 2.0;
        bcval->replace_global_values(1, &val, &gid);

        gid = lm[2 * i + 1];
        val = 1;
        dbctog->replace_global_values(1, &val, &gid);
      }
    }  // End of node i has a condition
  }  // End of for loop

  // ---------------------------------------------------------------------------------
  // Solve the any available junction boundary conditions
  // ---------------------------------------------------------------------------------
  for (int i = 0; i < 2; i++)
  {
    if (get_first_condition(discretization, ele->nodes()[i], "ArtJunctionCond"))
    {
      std::shared_ptr<std::map<const int, std::shared_ptr<Arteries::Utils::JunctionNodeParams>>>
          junc_nodal_vals;
      junc_nodal_vals = params.get<std::shared_ptr<
          std::map<const int, std::shared_ptr<Arteries::Utils::JunctionNodeParams>>>>(
          "Junctions Parameters");

      std::shared_ptr<Core::LinAlg::Vector<double>> bcval =
          params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("bcval");
      std::shared_ptr<Core::LinAlg::Vector<double>> dbctog =
          params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("dbctog");

      // -------------------------------------------------------------------------------
      // Update the Dirichlet BC vector
      // -------------------------------------------------------------------------------
      int local_id = discretization.node_row_map()->lid(ele->nodes()[i]->id());
      int gid;
      double val;
      // set A at node i
      gid = lm[2 * i];
      val = (*junc_nodal_vals)[local_id]->A_;
      bcval->replace_global_values(1, &val, &gid);

      val = 1;
      dbctog->replace_global_values(1, &val, &gid);

      // set Q at node i
      gid = lm[2 * i + 1];
      val = (*junc_nodal_vals)[local_id]->Q_;
      bcval->replace_global_values(1, &val, &gid);

      val = 1;
      dbctog->replace_global_values(1, &val, &gid);
    }
  }
}

/*----------------------------------------------------------------------*
 |  Evaluate the values of the degrees of freedom           ismail 07/09|
 |  at terminal nodes.                                                  |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ArteryEleCalcLinExp<distype>::evaluate_scatra_bc(Artery* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material> material)
{
  //  const int numnode = my::iel_;

  // loop over the nodes
  for (int i = 0; i < 2; i++)
  {
    if (get_first_condition(discretization, ele->nodes()[i], "ArtInOutCond"))
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> bcval =
          params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("bcval");
      std::shared_ptr<Core::LinAlg::Vector<double>> dbctog =
          params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("dbctog");
      double time = params.get<double>("time");

      //
      // calculating Q at node i
      const Core::Conditions::Condition* condition =
          get_first_condition(discretization, ele->nodes()[i], "ArtPrescribedScatraCond");

      const auto* curve = condition->parameters().get_if<int>("curve");

      double curvefac = condition->parameters().get<double>("VAL");
      int curvenum = -1;
      if (curve) curvenum = *curve;
      if (curvenum > 0)
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curvenum)
                       .evaluate(time);
      }


      std::string TerminalType =
          (get_first_condition(discretization, ele->nodes()[i], "ArtInOutCond")
                  ->parameters()
                  .get<std::string>("terminaltype"));
      int dof = 0;
      if (TerminalType == "inlet")
      {
        dof = 0;
      }
      else if (TerminalType == "outlet")
      {
        dof = 1;
      }
      else
      {
        FOUR_C_THROW(
            "double check input file, ArtInOutCond must be either \"inlet\" or \"outlet\"");
      }

      int gid = lm[2 * i + dof];
      double val = 1;
      dbctog->replace_global_values(1, &val, &gid);

      gid = lm[2 * i + dof];
      val = curvefac;
      bcval->replace_global_values(1, &val, &gid);
    }
  }
}

/*----------------------------------------------------------------------*
 |  Evaluate the values of the degrees of freedom           ismail 07/09|
 |  at terminal nodes.                                                  |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ArteryEleCalcLinExp<distype>::calc_postprocessing_values(Artery* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material> material)
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> qanp = discretization.get_state("qanp");
  //  std::shared_ptr<const Core::LinAlg::Vector<double>> Wfnp  = discretization.GetState("Wfnp");
  //  std::shared_ptr<const Core::LinAlg::Vector<double>> Wbnp  = discretization.GetState("Wbnp");

  std::shared_ptr<Core::LinAlg::Vector<double>> on =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("pressure");
  std::shared_ptr<Core::LinAlg::Vector<double>> qn =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("flow");
  std::shared_ptr<Core::LinAlg::Vector<double>> an =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("art_area");

  // get time-step size
  //  const double dt = params.get<double>("time step size");

  // check here, if we really have an artery !!
  // Define Geometric variables
  double Ao1 = 0.0;
  double Ao2 = 0.0;
  // Define blood material variables
  //  double visc=0.0;
  //  double dens=0.0;
  // Define artery's material variables
  double t1 = 0.0;
  double t2 = 0.0;
  double E1 = 0.0;
  double E2 = 0.0;
  double nue = 0.0;
  // Define artery's external forces
  double pext1 = 0.0;
  double pext2 = 0.0;
  // check here, if we really have an artery !!
  if (material->material_type() == Core::Materials::m_cnst_art)
  {
    const Mat::Cnst1dArt* actmat = static_cast<const Mat::Cnst1dArt*>(material.get());
    // Read in initial cross-sectional area at node 1
    Ao1 = std::numbers::pi * pow(actmat->diam() / 2, 2);
    // Read in initial cross-sectional area at node 2
    Ao2 = Ao1;
    // Read in blood density
    //    dens   = actmat->Density();
    // Read in blood viscosity
    //    visc   = actmat->Viscosity();
    // Read in artery's thickness at node 1
    t1 = actmat->th();
    // Read in artery's thickness at node 2
    t2 = t1;
    // Read in artery's Youngs modulus of elasticity thickness at node 1
    E1 = actmat->young();
    // Read in artery's Youngs modulus of elasticity thickness at node 2
    E2 = E1;
    // Read in artery's Poisson's ratio
    nue = actmat->nue();
    // Read in artery's external forces at node 1
    pext1 = actmat->pext(0);
    // Read in artery's external forces at node 1
    pext2 = actmat->pext(1);

    // Set up all the needed vectors for further calculations
    area0_(0, 0) = Ao1;
    area0_(1, 0) = Ao2;
    th_(0, 0) = t1;
    th_(1, 0) = t2;
    young_(0, 0) = E1;
    young_(1, 0) = E2;
    pext_(0, 0) = pext1;
    pext_(1, 0) = pext2;
  }
  else
    FOUR_C_THROW("Material law is not an artery");


  // the number of nodes
  const int numnode = my::iel_;

  if (qanp == nullptr) FOUR_C_THROW("Cannot get state vectors 'qanp'");

  // extract local values from the global vectors
  std::vector<double> myqanp = Core::FE::extract_values(*qanp, lm);

  // create objects for element arrays
  Core::LinAlg::Matrix<numnode, 1> eareanp;
  Core::LinAlg::Matrix<numnode, 1> eqnp;

  // get time step size
  //  const double dt = params.get<double>("time step size");

  // get all values at the last computed time step
  for (int i = 0; i < numnode; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    eqnp(i) = myqanp[1 + (i * 2)];
    qn_(i) = myqanp[1 + (i * 2)];
    eareanp(i) = myqanp[0 + (i * 2)];
    an_(i) = myqanp[0 + (i * 2)];
  }


  // ---------------------------------------------------------------------------------
  // Evaluate flowrate and pressure values at both sides of the artery
  // ---------------------------------------------------------------------------------
  area0_(0, 0) = Ao1;
  area0_(1, 0) = Ao2;
  th_(0, 0) = t1;
  th_(1, 0) = t2;
  young_(0, 0) = E1;
  young_(1, 0) = E2;
  pext_(0, 0) = pext1;
  pext_(1, 0) = pext2;

  for (int i = 0; i < 2; i++)
  {
    const double beta = sqrt(std::numbers::pi) * young_(i) * th_(i) / (1.0 - nue * nue);

    int myrank = Core::Communication::my_mpi_rank(discretization.get_comm());
    if (myrank == ele->nodes()[i]->owner())
    {
      int gid = ele->nodes()[i]->id();
      double val;

      // calculating P at node i
      double pressure = 0.0;
      pressure = beta * (sqrt(an_(i)) - sqrt(area0_(i))) / area0_(i) + pext_(i);

      gid = ele->nodes()[i]->id();
      val = pressure;
      on->replace_global_values(1, &val, &gid);

      // calculating Q at node i
      val = qn_(i);
      qn->replace_global_values(1, &val, &gid);

      // evaluate area
      val = an_(i);
      an->replace_global_values(1, &val, &gid);
    }
  }  // End of node i has a condition
}

/*----------------------------------------------------------------------*
 |  Evaluate the values of the scalar transport             ismail 12/12|
 |  from the forward and backward transport                             |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ArteryEleCalcLinExp<distype>::calc_scatra_from_scatra_fw(Artery* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material> material)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> scatra_fb =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("scatra_fb");
  std::shared_ptr<Core::LinAlg::Vector<double>> scatra =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("scatra");

  // the number of nodes
  const int numnode = my::iel_;

  // extract local values from the global vectors
  std::vector<double> myscatra_fb = Core::FE::extract_values(*scatra_fb, lm);

  // get all values at the last computed time step
  double val = 0.0;
  int gid = 0;
  for (int i = 0; i < numnode; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    val = myscatra_fb[i * 2] + myscatra_fb[i * 2 + 1];
    gid = ele->nodes()[i]->id();
    scatra->replace_global_values(1, &val, &gid);
  }
}


/*----------------------------------------------------------------------*
 |  Evaluate Wf and Wb                                      ismail 12/12|
 |                                                                      |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ArteryEleCalcLinExp<distype>::evaluate_wf_and_wb(Artery* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material> material)
{
  // Define Geometric variables
  double Ao1 = 0.0;
  double Ao2 = 0.0;
  // Define blood material variables
  //  double visc = 0.0;
  double dens = 0.0;
  // Define artery's material variables
  double t1 = 0.0;
  double t2 = 0.0;
  double E1 = 0.0;
  double E2 = 0.0;
  double nue = 0.0;
  // Define artery's external forces
  double pext1 = 0.0;
  double pext2 = 0.0;
  // check here, if we really have an artery !!
  if (material->material_type() == Core::Materials::m_cnst_art)
  {
    const Mat::Cnst1dArt* actmat = static_cast<const Mat::Cnst1dArt*>(material.get());
    // Read in initial cross-sectional area at node 1
    Ao1 = std::numbers::pi * pow(actmat->diam() / 2, 2);
    // Read in initial cross-sectional area at node 2
    Ao2 = Ao1;
    // Read in blood density
    dens = actmat->density();
    // Read in blood viscosity
    //    visc   = actmat->Viscosity();
    // Read in artery's thickness at node 1
    t1 = actmat->th();
    // Read in artery's thickness at node 2
    t2 = t1;
    // Read in artery's Youngs modulus of elasticity thickness at node 1
    E1 = actmat->young();
    // Read in artery's Youngs modulus of elasticity thickness at node 2
    E2 = E1;
    // Read in artery's Poisson's ratio
    nue = actmat->nue();
    // Read in artery's external forces at node 1
    pext1 = actmat->pext(0);
    // Read in artery's external forces at node 1
    pext2 = actmat->pext(1);

    // Set up all the needed vectors for further calculations
    area0_(0, 0) = Ao1;
    area0_(1, 0) = Ao2;
    th_(0, 0) = t1;
    th_(1, 0) = t2;
    young_(0, 0) = E1;
    young_(1, 0) = E2;
    pext_(0, 0) = pext1;
    pext_(1, 0) = pext2;
  }
  else
    FOUR_C_THROW("Material law is not an artery");

  // the number of nodes
  const int numnode = my::iel_;

  std::shared_ptr<const Core::LinAlg::Vector<double>> qanp = discretization.get_state("qanp");
  std::shared_ptr<Core::LinAlg::Vector<double>> Wfnp =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wfnp");
  std::shared_ptr<Core::LinAlg::Vector<double>> Wbnp =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wbnp");

  if (qanp == nullptr) FOUR_C_THROW("Cannot get state vectors 'qanp'");

  // extract local values from the global vectors
  std::vector<double> myqanp = Core::FE::extract_values(*qanp, lm);

  // create objects for element arrays
  Core::LinAlg::Matrix<numnode, 1> earean;
  Core::LinAlg::Matrix<numnode, 1> eqn;

  // get time step size
  //  const double dt = params.get<double>("time step size");

  // get all values at the last computed time step
  for (int i = 0; i < numnode; ++i)
  {
    // split area and volumetric flow rate, insert into element arrays
    eqn(i) = myqanp[1 + (i * 2)];
    qn_(i) = myqanp[1 + (i * 2)];
    earean(i) = myqanp[0 + (i * 2)];
    an_(i) = myqanp[0 + (i * 2)];
  }

  // get the nodal coordinates of the element
  Core::Nodes::Node** nodes = ele->nodes();
  Core::LinAlg::Matrix<3, my::iel_> xyze;
  for (int inode = 0; inode < my::iel_; inode++)
  {
    const auto& x = nodes[inode]->x();
    xyze(0, inode) = x[0];
    xyze(1, inode) = x[1];
    xyze(2, inode) = x[2];
  }
  //  const double L=sqrt(
  //            pow(xyze(0,0) - xyze(0,1),2)
  //          + pow(xyze(1,0) - xyze(1,1),2)
  //          + pow(xyze(2,0) - xyze(2,1),2));
  //  bool BCnodes= false;

  // get the number of nodes per element
  const int numnds = ele->num_node();

  if (numnds != 2) FOUR_C_THROW("An element with {} nodes is not supported", numnds);

  // check for the CFL number CFL = Max(abs(3/sqrt(3) * lambda2_i * dt/dx), abs(3/sqrt(3) *
  // lambda1_i * dt/dx))
  for (int i = 0; i < numnode; ++i)
  {
    const double c = sqrt(sqrt(std::numbers::pi) * young_(i) * th_(i) / (1.0 - pow(nue, 2)) *
                          sqrt(earean(i)) / (2.0 * area0_(i) * dens));
    double Wf = eqn(i) / earean(i) + 4.0 * c;
    double Wb = eqn(i) / earean(i) - 4.0 * c;

    //    std::cout<<"Wb:  "<<Wb<<std::endl;
    int gid = ele->nodes()[i]->id();
    Wbnp->replace_global_values(1, &Wb, &gid);
    Wfnp->replace_global_values(1, &Wf, &gid);
  }
}


/*----------------------------------------------------------------------*
 |  Solve scatra analytically                               ismail 12/12|
 |                                                                      |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ArteryEleCalcLinExp<distype>::solve_scatra_analytically(Artery* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    std::shared_ptr<Core::Mat::Material> material)
{
  // the number of nodes
  const int numnode = my::iel_;

  //----------------------------------------------------------------------
  // get control parameters for time integration
  //----------------------------------------------------------------------

  // get time-step size
  const double dt = params.get<double>("time step size");

  // ---------------------------------------------------------------------
  // get control parameters for stabilization and higher-order elements
  //----------------------------------------------------------------------


  // flag for higher order elements
  //  bool higher_order_ele = ele->is_higher_order_element(distype);

  // ---------------------------------------------------------------------
  // get all general state vectors: flow./area.,
  // ---------------------------------------------------------------------
  std::shared_ptr<Core::LinAlg::Vector<double>> wfn =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wfn");
  std::shared_ptr<Core::LinAlg::Vector<double>> wbn =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wbn");
  std::shared_ptr<Core::LinAlg::Vector<double>> wfo =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wfo");
  std::shared_ptr<Core::LinAlg::Vector<double>> wbo =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wbo");

  std::shared_ptr<Core::LinAlg::Vector<double>> scatran =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("scatran");
  std::shared_ptr<Core::LinAlg::Vector<double>> scatranp =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("scatranp");

  // extract local values from the global vectors
  std::vector<double> myescatran = Core::FE::extract_values(*scatran, lm);
  //  Core::FE::extract_my_values(*qan ,myqan ,lm);

  // create objects for element arrays
  Core::LinAlg::Matrix<2 * numnode, 1> escatran;
  Core::LinAlg::Matrix<numnode, 1> ewfn;
  Core::LinAlg::Matrix<numnode, 1> ewbn;
  for (int i = 0; i < numnode; ++i)
  {
    escatran(2 * i) = myescatran[2 * i];
    escatran(2 * i + 1) = myescatran[2 * i + 1];
    // get element scalar transport
    int local_id = discretization.node_row_map()->lid(ele->nodes()[i]->id());
    //    escatran(i)     = (*scatran)[local_id];
    ewfn(i) = wfn->local_values_as_span()[local_id] - wfo->local_values_as_span()[local_id];
    ewbn(i) = wbn->local_values_as_span()[local_id] - wbo->local_values_as_span()[local_id];
  }

  // Get length of the element
  // get node coordinates and number of elements per node
  Core::Nodes::Node** nodes = ele->nodes();
  Core::LinAlg::Matrix<3, my::iel_> xyze;
  for (int inode = 0; inode < numnode; inode++)
  {
    const auto& x = nodes[inode]->x();
    xyze(0, inode) = x[0];
    xyze(1, inode) = x[1];
    xyze(2, inode) = x[2];
  }

  const double L = sqrt(pow(xyze(0, 0) - xyze(0, 1), 2) + pow(xyze(1, 0) - xyze(1, 1), 2) +
                        pow(xyze(2, 0) - xyze(2, 1), 2));

  // Evaluate forward Scalar transport at n+1
  //  if(! ele->Nodes()[0]->GetCondition("ArtInOutCond"))
  {
    int nodenum = 1;
    // evaluta forward scatra
    int gid = lm[2 * nodenum];

    // get forward speed
    const double cf = ewfn(nodenum);
    double x = cf * dt;

    // get shape functions of forward speed
    const double N1 = (L - x) / L;
    const double N2 = (x) / L;

    double cn1 = escatran(0);
    double cn2 = escatran(2);

    double val = cn1 * N1 + cn2 * N2;
    scatranp->replace_global_values(1, &val, &gid);
  }

  // Evaluate backward Scalar transport at n+1
  //  if(! ele->Nodes()[1]->GetCondition("ArtInOutCond"))
  {
    int nodenum = 0;
    // evaluta forward scatra
    int gid = lm[2 * nodenum + 1];

    // get forward speed
    const double cf = ewbn(nodenum);
    double x = cf * dt;

    // get shape functions of forward speed
    const double N1 = (L - x) / L;
    const double N2 = (x) / L;

    double cn1 = escatran(1);
    double cn2 = escatran(3);

    double val = cn1 * N1 + cn2 * N2;
    scatranp->replace_global_values(1, &val, &gid);
  }
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// template classes

// 1D elements
template class Discret::Elements::ArteryEleCalcLinExp<Core::FE::CellType::line2>;

FOUR_C_NAMESPACE_CLOSE

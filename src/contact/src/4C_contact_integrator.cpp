// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_integrator.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_contact_defines.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_node.hpp"
#include "4C_contact_paramsinterface.hpp"
#include "4C_contact_wear_input.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_mortar_calc_utils.hpp"
#include "4C_mortar_coupling3d_classes.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_projector.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                            farah 10/13|
 *----------------------------------------------------------------------*/
CONTACT::Integrator::Integrator(
    Teuchos::ParameterList& params, Core::FE::CellType eletype, MPI_Comm comm)
    : imortar_(params),
      Comm_(comm),
      dim_(imortar_.get<int>("DIMENSION")),
      shapefcn_(Teuchos::getIntegralValue<Mortar::ShapeFcn>(imortar_, "LM_SHAPEFCN")),
      lagmultquad_(Teuchos::getIntegralValue<Mortar::LagMultQuad>(imortar_, "LM_QUAD")),
      gpslip_(imortar_.get<bool>("GP_SLIP_INCR")),
      algo_(Teuchos::getIntegralValue<Mortar::AlgorithmType>(imortar_, "ALGORITHM")),
      stype_(Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(imortar_, "STRATEGY")),
      cppnormal_(imortar_.get<bool>("CPP_NORMALS")),
      wearlaw_(Teuchos::getIntegralValue<Wear::WearLaw>(imortar_, "WEARLAW")),
      wearimpl_(false),
      wearside_(Wear::wear_source),
      weartype_(Wear::wear_intstate),
      wearshapefcn_(Wear::wear_shape_standard),
      sswear_(imortar_.get<bool>("SSWEAR")),
      wearcoeff_(-1.0),
      wearcoeffm_(-1.0),
      ssslip_(imortar_.get<double>("SSSLIP")),
      nonsmooth_(false),
      nonsmoothselfcontactsurface_(imortar_.get<bool>("NONSMOOTH_CONTACT_SURFACE")),
      integrationtype_(Teuchos::getIntegralValue<Mortar::IntType>(imortar_, "INTTYPE"))
{
  // init gp
  initialize_gp(eletype);

  // wear specific
  if (wearlaw_ != Wear::wear_none)
  {
    // set wear contact status
    auto wtimint = Teuchos::getIntegralValue<Wear::WearTimInt>(params, "WEARTIMINT");
    if (wtimint == Wear::wear_impl) wearimpl_ = true;

    // wear surface
    wearside_ = Teuchos::getIntegralValue<Wear::WearSide>(imortar_, "WEAR_SIDE");

    // wear algorithm
    weartype_ = Teuchos::getIntegralValue<Wear::WearType>(imortar_, "WEARTYPE");

    // wear shape function
    wearshapefcn_ = Teuchos::getIntegralValue<Wear::WearShape>(imortar_, "WEAR_SHAPEFCN");

    // wear coefficient
    wearcoeff_ = imortar_.get<double>("WEARCOEFF");

    // wear coefficient
    wearcoeffm_ = imortar_.get<double>("WEARCOEFF_MASTER");
  }

  nonsmooth_ = imortar_.get<bool>("NONSMOOTH_GEOMETRIES");
}

/*----------------------------------------------------------------------*
 |  check for boundary elements                              farah 02/14|
 *----------------------------------------------------------------------*/
bool CONTACT::Integrator::boundary_segm_check_2d(
    Mortar::Element& source_elem, std::vector<Mortar::Element*> target_elems)
{
  double source_xi_test[2] = {0.0, 0.0};
  bool proj_test = false;
  bool boundary_ele = false;

  double glob_test[3] = {0.0, 0.0, 0.0};

  Core::Nodes::Node** mynodes_test = source_elem.nodes();
  if (!mynodes_test) FOUR_C_THROW("has_proj_status: Null pointer!");

  if (source_elem.shape() == Core::FE::CellType::line2 ||
      source_elem.shape() == Core::FE::CellType::nurbs2)
  {
    for (int s_test = 0; s_test < 2; ++s_test)
    {
      if (s_test == 0)
        source_xi_test[0] = -1.0;
      else
        source_xi_test[0] = 1.0;

      proj_test = false;
      for (int bs_test = 0; bs_test < (int)target_elems.size(); ++bs_test)
      {
        double target_xi_test[2] = {0.0, 0.0};
        Mortar::Projector::impl(source_elem, *target_elems[bs_test])
            ->project_gauss_point_2d(
                source_elem, source_xi_test, *target_elems[bs_test], target_xi_test);

        if ((target_xi_test[0] >= -1.0) && (target_xi_test[0] <= 1.0))
        {
          // get hasproj
          source_elem.local_to_global(source_xi_test, glob_test, 0);
          for (int ii = 0; ii < source_elem.num_node(); ++ii)
          {
            Mortar::Node* mycnode_test = dynamic_cast<Mortar::Node*>(mynodes_test[ii]);
            if (!mycnode_test) FOUR_C_THROW("has_proj_status: Null pointer!");

            if (glob_test[0] == mycnode_test->xspatial()[0] &&
                glob_test[1] == mycnode_test->xspatial()[1] &&
                glob_test[2] == mycnode_test->xspatial()[2])
              mycnode_test->has_proj() = true;
          }

          glob_test[0] = 0.0;
          glob_test[1] = 0.0;
          glob_test[2] = 0.0;

          proj_test = true;
        }
      }
      if (proj_test == false) boundary_ele = true;
    }
  }
  else if (source_elem.shape() == Core::FE::CellType::line3 ||
           source_elem.shape() == Core::FE::CellType::nurbs3)
  {
    for (int s_test = 0; s_test < 3; ++s_test)
    {
      if (s_test == 0)
        source_xi_test[0] = -1.0;
      else if (s_test == 1)
        source_xi_test[0] = 0.0;
      else if (s_test == 2)
        source_xi_test[0] = 1.0;

      proj_test = false;
      for (int bs_test = 0; bs_test < (int)target_elems.size(); ++bs_test)
      {
        double target_xi_test[2] = {0.0, 0.0};
        Mortar::Projector::impl(source_elem, *target_elems[bs_test])
            ->project_gauss_point_2d(
                source_elem, source_xi_test, *target_elems[bs_test], target_xi_test);

        if ((target_xi_test[0] >= -1.0) && (target_xi_test[0] <= 1.0))
        {
          // get hasproj
          source_elem.local_to_global(source_xi_test, glob_test, 0);
          for (int ii = 0; ii < source_elem.num_node(); ++ii)
          {
            Mortar::Node* mycnode_test = dynamic_cast<Mortar::Node*>(mynodes_test[ii]);
            if (!mycnode_test) FOUR_C_THROW("has_proj_status: Null pointer!");

            if (glob_test[0] == mycnode_test->xspatial()[0] &&
                glob_test[1] == mycnode_test->xspatial()[1] &&
                glob_test[2] == mycnode_test->xspatial()[2])
              mycnode_test->has_proj() = true;
          }

          glob_test[0] = 0.0;
          glob_test[1] = 0.0;
          glob_test[2] = 0.0;

          proj_test = true;
        }
      }
      if (proj_test == false) boundary_ele = true;
    }
  }
  else
  {
    FOUR_C_THROW("No valid element type for source discretization!");
  }

  return boundary_ele;
}


/*----------------------------------------------------------------------*
 |  Initialize gauss points                                   popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::initialize_gp(Core::FE::CellType eletype)
{
  //**********************************************************************
  // Create integration points according to eletype!
  //
  // For segment-based integration, we have pre-defined default
  // values for the Gauss rules according to the segment type.
  //
  // default for integrals on 1D lines:
  // --> 5 GP (degree of precision: 9)
  //
  // default for integrals on 2D triangles:
  // --> 7 GP (degree of precision: 5)
  //
  // default for integrals on 2D quadrilaterals:
  // --> 9 GP (degree of precision: 5)
  //
  // For element-based integration, we choose the Gauss rules according
  // to the user's wish (i.e. according to the parameter NUMGP_PER_DIM).
  //
  // possibilities for integrals on 1D lines:
  // --> 1,2,3,4,5,6,7,8,9,10,16,20,32 GPs
  //
  // possibilities for integrals on 2D triangles:
  // --> 1,3,6,7,12,37,64 GPs
  //
  // possibilities for integrals on 2D quadrilaterals
  // --> 1,4,9,16,25,36,49,64,81,100,256,400,1024 GPs
  //**********************************************************************

  // get numgp (for element-based integration)
  int numgp = imortar_.get<int>("NUMGP_PER_DIM");

  // get integration type
  auto integrationtype = Teuchos::getIntegralValue<Mortar::IntType>(imortar_, "INTTYPE");

  //**********************************************************************
  // choose Gauss rule according to (a) element type (b) input parameter
  //**********************************************************************
  switch (eletype)
  {
    case Core::FE::CellType::line2:
    case Core::FE::CellType::line3:
    case Core::FE::CellType::nurbs2:
    case Core::FE::CellType::nurbs3:
    {
      // set default value for segment-based version first
      Core::FE::GaussRule1D mygaussrule = Core::FE::GaussRule1D::line_5point;

      // GP switch if element-based version and non-zero value provided by user
      if (integrationtype == Mortar::inttype_elements ||
          integrationtype == Mortar::inttype_elements_BS)
      {
        if (numgp > 0)
        {
          switch (numgp)
          {
            case 1:
            {
              FOUR_C_THROW("Our experience says that 1 GP per source element is not enough.");
              break;
            }
            case 2:
            {
              mygaussrule = Core::FE::GaussRule1D::line_2point;
              break;
            }
            case 3:
            {
              mygaussrule = Core::FE::GaussRule1D::line_3point;
              break;
            }
            case 4:
            {
              mygaussrule = Core::FE::GaussRule1D::line_4point;
              break;
            }
            case 5:
            {
              mygaussrule = Core::FE::GaussRule1D::line_5point;
              break;
            }
            case 6:
            {
              mygaussrule = Core::FE::GaussRule1D::line_6point;
              break;
            }
            case 7:
            {
              mygaussrule = Core::FE::GaussRule1D::line_7point;
              break;
            }
            case 8:
            {
              mygaussrule = Core::FE::GaussRule1D::line_8point;
              break;
            }
            case 9:
            {
              mygaussrule = Core::FE::GaussRule1D::line_9point;
              break;
            }
            case 10:
            {
              mygaussrule = Core::FE::GaussRule1D::line_10point;
              break;
            }
            case 16:
            {
              mygaussrule = Core::FE::GaussRule1D::line_16point;
              break;
            }
            case 20:
            {
              mygaussrule = Core::FE::GaussRule1D::line_20point;
              break;
            }
            case 32:
            {
              mygaussrule = Core::FE::GaussRule1D::line_32point;
              break;
            }
            case 50:
            {
              mygaussrule = Core::FE::GaussRule1D::line_50point;
              break;
            }
            default:
            {
              FOUR_C_THROW("Requested GP-Number is not implemented!");
              break;
            }
          }
        }
      }

      const Core::FE::IntegrationPoints1D intpoints(mygaussrule);
      ngp_ = intpoints.nquad;
      coords_.reshape(n_gp(), 2);
      weights_.resize(n_gp());
      for (int i = 0; i < n_gp(); ++i)
      {
        coords_(i, 0) = intpoints.qxg[i][0];
        coords_(i, 1) = 0.0;
        weights_[i] = intpoints.qwgt[i];
      }
      break;
    }
    case Core::FE::CellType::tri3:
    case Core::FE::CellType::tri6:
    {
      // set default value for segment-based version first
      Core::FE::GaussRule2D mygaussrule = Core::FE::GaussRule2D::tri_7point;
      if (integrationtype == Mortar::inttype_segments)
      {
        if (numgp > 0) switch (numgp)
          {
            case 1:
              mygaussrule = Core::FE::GaussRule2D::tri_1point;
              break;
            case 3:
              mygaussrule = Core::FE::GaussRule2D::tri_3point;
              break;
            case 7:
              mygaussrule = Core::FE::GaussRule2D::tri_7point;
              break;
            case 16:
              mygaussrule = Core::FE::GaussRule2D::tri_16point;
              break;
            case 37:
              mygaussrule = Core::FE::GaussRule2D::tri_37point;
              break;
            default:
              FOUR_C_THROW("unknown tri gauss rule");
              break;
          }
      }

      // GP switch if element-based version and non-zero value provided by user
      if (integrationtype == Mortar::inttype_elements ||
          integrationtype == Mortar::inttype_elements_BS)
      {
        if (numgp > 0)
        {
          switch (numgp)
          {
            case 1:
            {
              mygaussrule = Core::FE::GaussRule2D::tri_3point;
              break;
            }
            case 2:
            {
              mygaussrule = Core::FE::GaussRule2D::tri_6point;
              break;
            }
            case 3:
            {
              mygaussrule = Core::FE::GaussRule2D::tri_7point;
              break;
            }
            case 4:
            {
              mygaussrule = Core::FE::GaussRule2D::tri_12point;
              break;
            }
            case 5:
            {
              mygaussrule = Core::FE::GaussRule2D::tri_12point;
              break;
            }
            case 6:
            {
              mygaussrule = Core::FE::GaussRule2D::tri_37point;
              break;
            }
            case 7:
            {
              mygaussrule = Core::FE::GaussRule2D::tri_37point;
              break;
            }
            case 8:
            {
              mygaussrule = Core::FE::GaussRule2D::tri_64point;
              break;
            }
            case 9:
            {
              mygaussrule = Core::FE::GaussRule2D::tri_64point;
              break;
            }
            case 10:
            {
              mygaussrule = Core::FE::GaussRule2D::tri_64point;
              break;
            }
            case 20:
            {
              mygaussrule = Core::FE::GaussRule2D::tri_64point;
              break;
            }
            default:
            {
              FOUR_C_THROW("Requested GP-Number is not implemented!");
              break;
            }
          }
        }
      }

      const Core::FE::IntegrationPoints2D intpoints(mygaussrule);
      ngp_ = intpoints.nquad;
      coords_.reshape(n_gp(), 2);
      weights_.resize(n_gp());
      for (int i = 0; i < n_gp(); ++i)
      {
        coords_(i, 0) = intpoints.qxg[i][0];
        coords_(i, 1) = intpoints.qxg[i][1];
        weights_[i] = intpoints.qwgt[i];
      }
      break;
    }
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
    case Core::FE::CellType::nurbs4:
    case Core::FE::CellType::nurbs8:
    case Core::FE::CellType::nurbs9:
    {
      // set default value for segment-based version first
      Core::FE::GaussRule2D mygaussrule = Core::FE::GaussRule2D::quad_9point;

      // GP switch if element-based version and non-zero value provided by user
      if (integrationtype == Mortar::inttype_elements ||
          integrationtype == Mortar::inttype_elements_BS)
      {
        if (numgp > 0)
        {
          switch (numgp)
          {
            case 1:
            {
              mygaussrule = Core::FE::GaussRule2D::quad_1point;
              break;
            }
            case 2:
            {
              mygaussrule = Core::FE::GaussRule2D::quad_4point;
              break;
            }
            case 3:
            {
              mygaussrule = Core::FE::GaussRule2D::quad_9point;
              break;
            }
            case 4:
            {
              mygaussrule = Core::FE::GaussRule2D::quad_16point;
              break;
            }
            case 5:
            {
              mygaussrule = Core::FE::GaussRule2D::quad_25point;
              break;
            }
            case 6:
            {
              mygaussrule = Core::FE::GaussRule2D::quad_36point;
              break;
            }
            case 7:
            {
              mygaussrule = Core::FE::GaussRule2D::quad_49point;
              break;
            }
            case 8:
            {
              mygaussrule = Core::FE::GaussRule2D::quad_64point;
              break;
            }
            case 9:
            {
              mygaussrule = Core::FE::GaussRule2D::quad_81point;
              break;
            }
            case 10:
            {
              mygaussrule = Core::FE::GaussRule2D::quad_100point;
              break;
            }
            case 16:
            {
              mygaussrule = Core::FE::GaussRule2D::quad_256point;
              break;
            }
            case 20:
            {
              mygaussrule = Core::FE::GaussRule2D::quad_400point;
              break;
            }
            case 32:
            {
              mygaussrule = Core::FE::GaussRule2D::quad_1024point;
              break;
            }
            default:
            {
              FOUR_C_THROW("Requested GP-Number is not implemented!");
              break;
            }
          }
        }
      }

      const Core::FE::IntegrationPoints2D intpoints(mygaussrule);
      ngp_ = intpoints.nquad;
      coords_.reshape(n_gp(), 2);
      weights_.resize(n_gp());
      for (int i = 0; i < n_gp(); ++i)
      {
        coords_(i, 0) = intpoints.qxg[i][0];
        coords_(i, 1) = intpoints.qxg[i][1];
        weights_[i] = intpoints.qwgt[i];
      }
      break;
    }
    default:
    {
      FOUR_C_THROW("This contact element type is not implemented!");
      break;
    }
  }  // switch(eletype)
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_deriv_segment_2d(Mortar::Element& source_elem,
    double& source_xi_a, double& source_xi_b, Mortar::Element& target_elem, double& target_xi_a,
    double& target_xi_b, MPI_Comm comm, const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  std::shared_ptr<CONTACT::ParamsInterface> cparams_ptr = nullptr;
  if (mparams_ptr)
  {
    cparams_ptr = std::dynamic_pointer_cast<CONTACT::ParamsInterface>(mparams_ptr);
  }
  integrate_deriv_segment_2d(source_elem, source_xi_a, source_xi_b, target_elem, target_xi_a,
      target_xi_b, comm, cparams_ptr);
}

/*----------------------------------------------------------------------*
 |  Integrate and linearize a 1D source / target overlap (2D)  popp 02/09|
 |  This method integrates the overlap M matrix and weighted gap g~     |
 |  and stores it in mseg and gseg respectively. Moreover, derivatives  |
 |  LinM and Ling are built and stored directly into the adjacent nodes.|
 |  (Thus this method combines EVERYTHING before done separately in     |
 |  IntegrateM, IntegrateG, DerivM and DerivG!)                         |
 |  Also wear is integrated.                                            |
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_deriv_segment_2d(Mortar::Element& source_elem,
    double& source_xi_a, double& source_xi_b, Mortar::Element& target_elem, double& target_xi_a,
    double& target_xi_b, MPI_Comm comm,
    const std::shared_ptr<CONTACT::ParamsInterface>& cparams_ptr)
{
  // skip this segment, if too small
  if (source_xi_b - source_xi_a < 4. * MORTARINTLIM) return;

  // *********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************

  // explicitly defined shape function type needed
  if (shape_fcn() == Mortar::shape_undefined)
    FOUR_C_THROW("integrate_deriv_segment_2d called without specific shape function defined!");

  // Petrov-Galerkin approach for LM not yet implemented for quadratic FE
  if (source_elem.shape() == Core::FE::CellType::line3 &&
      shape_fcn() == Mortar::shape_petrovgalerkin)
    FOUR_C_THROW("Petrov-Galerkin approach not yet implemented for 2-D quadratic FE interpolation");

  // check for problem dimension
  if (n_dim() != 2) FOUR_C_THROW("2D integration method called for non-2D problem");

  // check input data
  if ((!source_elem.is_source()) || (target_elem.is_source()))
    FOUR_C_THROW("integrate_deriv_segment_2d called on a wrong type of Mortar::Element pair!");
  if ((source_xi_a < -1.0) || (source_xi_b > 1.0))
    FOUR_C_THROW("integrate_deriv_segment_2d called with infeasible source limits!");
  if ((target_xi_a < -1.0) || (target_xi_b > 1.0))
    FOUR_C_THROW("integrate_deriv_segment_2d called with infeasible target limits!");

  // *********************************************************************
  // Prepare integration
  // *********************************************************************
  // number of nodes (source, target)
  int nrow = source_elem.num_node();
  int ncol = target_elem.num_node();
  int ndof = n_dim();

  // get source element nodes themselves
  Core::Nodes::Node** mynodes = source_elem.nodes();
  if (!mynodes) FOUR_C_THROW("Null pointer!");

  // decide whether boundary modification has to be considered or not
  // this is element-specific (is there a boundary node in this element?)
  for (int k = 0; k < nrow; ++k)
  {
    Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[k]);
    if (!mymrtrnode) FOUR_C_THROW("Null pointer!");
  }

  // decide whether linear LM are used for quadratic FE here
  // and whether displacement shape fct. modification has to be considered or not
  // this is the case for dual linear Lagrange multipliers on line3 elements
  bool linlm = false;
  bool dualquad = false;
  if (lag_mult_quad() == Mortar::lagmult_lin && source_elem.shape() == Core::FE::CellType::line3)
  {
    linlm = true;
    if (shapefcn_ == Mortar::shape_dual) dualquad = true;
  }

  // prepare directional derivative of dual shape functions
  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> dualmap(
      2 * nrow, 0, Core::LinAlg::SerialDenseMatrix(nrow, nrow));
  if ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
      (source_elem.shape() == Core::FE::CellType::line3 ||
          source_elem.shape() == Core::FE::CellType::nurbs3 ||
          source_elem.mo_data().deriv_dual_shape() != nullptr))
    source_elem.deriv_shape_dual(dualmap);

  // create empty vectors for shape fct. evaluation
  Core::LinAlg::SerialDenseVector source_val(nrow);
  Core::LinAlg::SerialDenseMatrix source_deriv(nrow, 1);
  Core::LinAlg::SerialDenseVector target_val(ncol);
  Core::LinAlg::SerialDenseMatrix target_deriv(ncol, 1);
  Core::LinAlg::SerialDenseVector lm_val(nrow);
  Core::LinAlg::SerialDenseMatrix lm_deriv(nrow, 1);

  Core::LinAlg::SerialDenseVector m2val(ncol);
  Core::LinAlg::SerialDenseMatrix m2deriv(ncol, 1);
  Core::LinAlg::SerialDenseVector lm2val(ncol);
  Core::LinAlg::SerialDenseMatrix lm2deriv(ncol, 1);

  // create empty vectors for shape fct. evaluation
  Core::LinAlg::SerialDenseMatrix ssecderiv(nrow, 1);

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  int linsize = 0;
  for (int i = 0; i < nrow; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(mynodes[i]);
    linsize += cnode->get_linsize();
  }

  // safety
  linsize = linsize * 2;

  // *********************************************************************
  // Find out about whether start / end of overlap are source or target!
  // CAUTION: be careful with positive rotation direction ("Umlaufsinn")
  // source_xi_a -> belongs to source_elem.Nodes()[0]
  // source_xi_b -> belongs to source_elem.Nodes()[1]
  // target_xi_a -> belongs to target_elem.Nodes()[0]
  // target_xi_b -> belongs to target_elem.Nodes()[1]
  // but source and target have different positive rotation directions,
  // counter-clockwise for source side, clockwise for target side!
  // this means that target_xi_a belongs to source_xi_b and vice versa!
  // *********************************************************************

  bool startsource = false;
  bool endsource = false;

  if (source_elem.normal_fac() * target_elem.normal_fac() > 0.)
  {
    if (source_xi_a != -1.0 && target_xi_b != 1.0)
      FOUR_C_THROW("First outer node is neither source nor target node");
    if (source_xi_b != 1.0 && target_xi_a != -1.0)
      FOUR_C_THROW("Second outer node is neither source nor target node");
  }
  else
  {
    if (source_xi_a != -1. && target_xi_a != -1.)
      FOUR_C_THROW("First outer node is neither source nor target node");
    if (source_xi_b != 1. && target_xi_b != 1.)
      FOUR_C_THROW("Second outer node is neither source nor target node");
  }
  if (source_xi_a == -1.0)
    startsource = true;
  else
    startsource = false;
  if (source_xi_b == 1.0)
    endsource = true;
  else
    endsource = false;

  // get directional derivatives of source_xi_a, source_xi_b, target_xi_a, target_xi_b
  std::vector<Core::Gen::Pairedvector<int, double>> ximaps(4, linsize + ndof * ncol);
  deriv_xi_a_b_2d(source_elem, source_xi_a, source_xi_b, target_elem, target_xi_a, target_xi_b,
      ximaps, startsource, endsource, linsize);

  //**********************************************************************
  // loop over all Gauss points for integration
  //**********************************************************************
  for (int gp = 0; gp < n_gp(); ++gp)
  {
    // coordinates and weight
    std::array<double, 2> eta = {coordinate(gp, 0), 0.0};
    double wgt = weight(gp);

    // coordinate transformation source_xi->eta (source Mortar::Element->Overlap)
    double source_xi[2] = {0.0, 0.0};
    source_xi[0] = 0.5 * (1.0 - eta[0]) * source_xi_a + 0.5 * (1.0 + eta[0]) * source_xi_b;

    // project Gauss point onto target element
    double target_xi[2] = {0.0, 0.0};
    Mortar::Projector::impl(source_elem, target_elem)
        ->project_gauss_point_2d(source_elem, source_xi, target_elem, target_xi);

    // check GP projection
    if ((target_xi[0] < target_xi_a - 1e-4) || (target_xi[0] > target_xi_b + 1e-4))
    {
      std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                << std::endl;
      std::cout << "Gauss point: " << source_xi[0] << " " << source_xi[1] << std::endl;
      std::cout << "Projection: " << target_xi[0] << " " << target_xi[1] << std::endl;
      std::cout << "Projection bounds: " << target_xi_a << " " << target_xi_b << std::endl;
      //  FOUR_C_THROW("IntegrateAndDerivSegment: Gauss point projection failed!
      //  target_xi={}",target_xi[0]);
    }

    // evaluate Lagrange multiplier shape functions (on source element)
    if (lag_mult_quad() == Mortar::lagmult_const)
      source_elem.evaluate_shape_lag_mult_const(shape_fcn(), source_xi, lm_val, lm_deriv, nrow);
    else if (linlm)
      source_elem.evaluate_shape_lag_mult_lin(shape_fcn(), source_xi, lm_val, lm_deriv, nrow);
    else
      source_elem.evaluate_shape_lag_mult(shape_fcn(), source_xi, lm_val, lm_deriv, nrow);

    // evaluate trace space shape functions (on both elements)
    source_elem.evaluate_shape(source_xi, source_val, source_deriv, nrow, dualquad);
    target_elem.evaluate_shape(target_xi, target_val, target_deriv, ncol, false);

    // evaluate the two source side Jacobians
    double dxdsxi = source_elem.jacobian(source_xi);
    double dsxideta = -0.5 * source_xi_a + 0.5 * source_xi_b;

    // evaluate linearizations *******************************************
    // evaluate 2nd deriv of trace space shape functions (on source element)
    source_elem.evaluate2nd_deriv_shape(source_xi, ssecderiv, nrow);

    // evaluate the derivative dxdsxidsxi = Jac,xi
    double djacdxi[2] = {0.0, 0.0};
    dynamic_cast<CONTACT::Element&>(source_elem).d_jac_d_xi(djacdxi, source_xi, ssecderiv);
    double dxdsxidsxi = djacdxi[0];  // only 2D here

    // evaluate the GP source coordinate derivatives
    std::vector<Core::Gen::Pairedvector<int, double>> d_source_xi_gp(
        1, Core::Gen::Pairedvector<int, double>(linsize + ndof * ncol));
    for (CI p = ximaps[0].begin(); p != ximaps[0].end(); ++p)
      d_source_xi_gp[0][p->first] += 0.5 * (1.0 - eta[0]) * (p->second);
    for (CI p = ximaps[1].begin(); p != ximaps[1].end(); ++p)
      d_source_xi_gp[0][p->first] += 0.5 * (1.0 + eta[0]) * (p->second);

    // evaluate the GP target coordinate derivatives
    std::vector<Core::Gen::Pairedvector<int, double>> d_target_xi_gp(
        1, Core::Gen::Pairedvector<int, double>(linsize + ndof * ncol));
    deriv_xi_gp_2d(source_elem, target_elem, source_xi[0], target_xi[0], d_source_xi_gp[0],
        d_target_xi_gp[0], linsize);

    // evaluate the Jacobian derivative
    Core::Gen::Pairedvector<int, double> derivjacSele(nrow * ndof);
    source_elem.deriv_jacobian(source_xi, derivjacSele);

    double jac = dsxideta * dxdsxi;
    Core::Gen::Pairedvector<int, double> derivjac(nrow * ndof + linsize);
    for (Core::Gen::Pairedvector<int, double>::const_iterator p = derivjacSele.begin();
        p != derivjacSele.end(); ++p)
      derivjac[p->first] += dsxideta * p->second;
    for (CI p = ximaps[0].begin(); p != ximaps[0].end(); ++p)
      derivjac[p->first] -= .5 * dxdsxi * p->second;
    for (CI p = ximaps[1].begin(); p != ximaps[1].end(); ++p)
      derivjac[p->first] += .5 * dxdsxi * p->second;
    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      derivjac[p->first] += dsxideta * dxdsxidsxi * p->second;

    //**********************************************************************
    // frequently reused quantities
    //**********************************************************************
    double gpn[2] = {0.0, 0.0};  // normalized normal at gp
    double gap = 0.0;            // gap
    Core::Gen::Pairedvector<int, double> dgapgp(
        linsize + ndof * ncol);  // gap lin without weighting and jac
    std::vector<Core::Gen::Pairedvector<int, double>> dnmap_unit(
        2, (linsize + ndof * ncol));  // deriv of x and y comp. of gpn (unit)

    //**********************************************************************
    // evaluate at GP and lin char. quantities
    //**********************************************************************
    if (dualquad && linlm)
    {
      // declare and compute shape functions as well as derivatives for computation of gap
      // -> standard quadratic functions instead of only linear functions
      Core::LinAlg::SerialDenseVector svalgap(nrow);
      Core::LinAlg::SerialDenseMatrix sderivgap(nrow, 1);
      source_elem.evaluate_shape(source_xi, svalgap, sderivgap, nrow);

      gap_2d(source_elem, target_elem, svalgap, target_val, sderivgap, target_deriv, &gap, gpn,
          d_source_xi_gp, d_target_xi_gp, dgapgp, dnmap_unit);
    }
    else
      gap_2d(source_elem, target_elem, source_val, target_val, source_deriv, target_deriv, &gap,
          gpn, d_source_xi_gp, d_target_xi_gp, dgapgp, dnmap_unit);

    integrate_gp_2d(source_elem, target_elem, source_val, lm_val, target_val, source_deriv,
        target_deriv, lm_deriv, dualmap, wgt, jac, derivjac, gpn, dnmap_unit, gap, dgapgp,
        source_xi, target_xi, d_source_xi_gp, d_target_xi_gp);

  }  // gp-loop
}

/*----------------------------------------------------------------------*
 |  check for boundary elements                              farah 07/14|
 *----------------------------------------------------------------------*/
bool CONTACT::Integrator::boundary_segm_check_3d(
    Mortar::Element& source_elem, std::vector<Mortar::Element*> target_elems)
{
  double source_xi_test[2] = {0.0, 0.0};
  bool proj_test = false;
  bool boundary_ele = false;
  double alpha_test = 0.0;
  double glob_test[3] = {0.0, 0.0, 0.0};
  const double tol = 1e-8;
  Core::Nodes::Node** mynodes_test = source_elem.nodes();
  if (!mynodes_test) FOUR_C_THROW("has_proj_status: Null pointer!");

  Core::FE::CellType dt_s = source_elem.shape();

  if (dt_s == Core::FE::CellType::quad4)  //|| dt_s==Core::FE::CellType::quad8
                                          //|| dt_s==Core::FE::CellType::quad9)
  {
    for (int s_test = 0; s_test < 4; ++s_test)
    {
      if (s_test == 0)
      {
        source_xi_test[0] = -1.0;
        source_xi_test[1] = -1.0;
      }
      else if (s_test == 1)
      {
        source_xi_test[0] = -1.0;
        source_xi_test[1] = 1.0;
      }
      else if (s_test == 2)
      {
        source_xi_test[0] = 1.0;
        source_xi_test[1] = -1.0;
      }
      else if (s_test == 3)
      {
        source_xi_test[0] = 1.0;
        source_xi_test[1] = 1.0;
      }

      proj_test = false;
      for (int bs_test = 0; bs_test < (int)target_elems.size(); ++bs_test)
      {
        double target_xi_test[2] = {0.0, 0.0};
        Mortar::Projector::impl(source_elem, *target_elems[bs_test])
            ->project_gauss_point_3d(
                source_elem, source_xi_test, *target_elems[bs_test], target_xi_test, alpha_test);
        Core::FE::CellType dt = target_elems[bs_test]->shape();

        if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
            dt == Core::FE::CellType::quad9)
        {
          if (target_xi_test[0] >= -1.0 && target_xi_test[1] >= -1.0 && target_xi_test[0] <= 1.0 &&
              target_xi_test[1] <= 1.0)
          {
            // get hasproj
            source_elem.local_to_global(source_xi_test, glob_test, 0);
            for (int ii = 0; ii < source_elem.num_node(); ++ii)
            {
              Mortar::Node* mycnode_test = dynamic_cast<Mortar::Node*>(mynodes_test[ii]);
              if (!mycnode_test) FOUR_C_THROW("has_proj_status: Null pointer!");

              if (glob_test[0] > mycnode_test->xspatial()[0] - tol &&
                  glob_test[0] < mycnode_test->xspatial()[0] + tol &&
                  glob_test[1] > mycnode_test->xspatial()[1] - tol &&
                  glob_test[1] < mycnode_test->xspatial()[1] + tol &&
                  glob_test[2] > mycnode_test->xspatial()[2] - tol &&
                  glob_test[2] < mycnode_test->xspatial()[2] + tol)
                mycnode_test->has_proj() = true;
            }

            glob_test[0] = 0.0;
            glob_test[1] = 0.0;
            glob_test[2] = 0.0;

            proj_test = true;
          }
        }
        else if (dt == Core::FE::CellType::tri3 || dt == Core::FE::CellType::tri6)
        {
          if (target_xi_test[0] >= 0.0 && target_xi_test[1] >= 0.0 && target_xi_test[0] <= 1.0 &&
              target_xi_test[1] <= 1.0 && target_xi_test[0] + target_xi_test[1] <= 1.0)
          {
            // get hasproj
            source_elem.local_to_global(source_xi_test, glob_test, 0);
            for (int ii = 0; ii < source_elem.num_node(); ++ii)
            {
              Mortar::Node* mycnode_test = dynamic_cast<Mortar::Node*>(mynodes_test[ii]);
              if (!mycnode_test) FOUR_C_THROW("has_proj_status: Null pointer!");

              if (glob_test[0] > mycnode_test->xspatial()[0] - tol &&
                  glob_test[0] < mycnode_test->xspatial()[0] + tol &&
                  glob_test[1] > mycnode_test->xspatial()[1] - tol &&
                  glob_test[1] < mycnode_test->xspatial()[1] + tol &&
                  glob_test[2] > mycnode_test->xspatial()[2] - tol &&
                  glob_test[2] < mycnode_test->xspatial()[2] + tol)
                mycnode_test->has_proj() = true;
            }
            glob_test[0] = 0.0;
            glob_test[1] = 0.0;
            glob_test[2] = 0.0;

            proj_test = true;
          }
        }
        else
        {
          FOUR_C_THROW("Non valid element type for target discretization!");
        }
      }
      if (proj_test == false) boundary_ele = true;
    }
  }
  else if (dt_s == Core::FE::CellType::quad9)  //|| dt_s==Core::FE::CellType::quad8 ||
                                               // dt_s==Core::FE::CellType::quad9)
  {
    for (int s_test = 0; s_test < 9; ++s_test)
    {
      if (s_test == 0)
      {
        source_xi_test[0] = -1.0;
        source_xi_test[1] = -1.0;
      }
      else if (s_test == 1)
      {
        source_xi_test[0] = 0.0;
        source_xi_test[1] = -1.0;
      }
      else if (s_test == 2)
      {
        source_xi_test[0] = 1.0;
        source_xi_test[1] = -1.0;
      }
      else if (s_test == 3)
      {
        source_xi_test[0] = -1.0;
        source_xi_test[1] = 0.0;
      }
      else if (s_test == 4)
      {
        source_xi_test[0] = 0.0;
        source_xi_test[1] = 0.0;
      }
      else if (s_test == 5)
      {
        source_xi_test[0] = 1.0;
        source_xi_test[1] = 0.0;
      }
      else if (s_test == 6)
      {
        source_xi_test[0] = -1.0;
        source_xi_test[1] = 1.0;
      }
      else if (s_test == 7)
      {
        source_xi_test[0] = 0.0;
        source_xi_test[1] = 1.0;
      }
      else if (s_test == 8)
      {
        source_xi_test[0] = 1.0;
        source_xi_test[1] = 1.0;
      }

      proj_test = false;
      for (int bs_test = 0; bs_test < (int)target_elems.size(); ++bs_test)
      {
        double target_xi_test[2] = {0.0, 0.0};
        Mortar::Projector::impl(source_elem, *target_elems[bs_test])
            ->project_gauss_point_3d(
                source_elem, source_xi_test, *target_elems[bs_test], target_xi_test, alpha_test);
        Core::FE::CellType dt = target_elems[bs_test]->shape();

        if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
            dt == Core::FE::CellType::quad9)
        {
          if (target_xi_test[0] >= -1.0 && target_xi_test[1] >= -1.0 && target_xi_test[0] <= 1.0 &&
              target_xi_test[1] <= 1.0)
          {
            // get hasproj
            source_elem.local_to_global(source_xi_test, glob_test, 0);
            for (int ii = 0; ii < source_elem.num_node(); ++ii)
            {
              Mortar::Node* mycnode_test = dynamic_cast<Mortar::Node*>(mynodes_test[ii]);
              if (!mycnode_test) FOUR_C_THROW("has_proj_status: Null pointer!");

              if (glob_test[0] > mycnode_test->xspatial()[0] - tol &&
                  glob_test[0] < mycnode_test->xspatial()[0] + tol &&
                  glob_test[1] > mycnode_test->xspatial()[1] - tol &&
                  glob_test[1] < mycnode_test->xspatial()[1] + tol &&
                  glob_test[2] > mycnode_test->xspatial()[2] - tol &&
                  glob_test[2] < mycnode_test->xspatial()[2] + tol)
                mycnode_test->has_proj() = true;
            }

            glob_test[0] = 0.0;
            glob_test[1] = 0.0;
            glob_test[2] = 0.0;

            proj_test = true;
          }
        }
        else if (dt == Core::FE::CellType::tri3 || dt == Core::FE::CellType::tri6)
        {
          if (target_xi_test[0] >= 0.0 && target_xi_test[1] >= 0.0 && target_xi_test[0] <= 1.0 &&
              target_xi_test[1] <= 1.0 && target_xi_test[0] + target_xi_test[1] <= 1.0)
          {
            // get hasproj
            source_elem.local_to_global(source_xi_test, glob_test, 0);
            for (int ii = 0; ii < source_elem.num_node(); ++ii)
            {
              Mortar::Node* mycnode_test = dynamic_cast<Mortar::Node*>(mynodes_test[ii]);
              if (!mycnode_test) FOUR_C_THROW("has_proj_status: Null pointer!");

              if (glob_test[0] > mycnode_test->xspatial()[0] - tol &&
                  glob_test[0] < mycnode_test->xspatial()[0] + tol &&
                  glob_test[1] > mycnode_test->xspatial()[1] - tol &&
                  glob_test[1] < mycnode_test->xspatial()[1] + tol &&
                  glob_test[2] > mycnode_test->xspatial()[2] - tol &&
                  glob_test[2] < mycnode_test->xspatial()[2] + tol)
                mycnode_test->has_proj() = true;
            }
            glob_test[0] = 0.0;
            glob_test[1] = 0.0;
            glob_test[2] = 0.0;

            proj_test = true;
          }
        }
        else
        {
          FOUR_C_THROW("Non valid element type for target discretization!");
        }
      }
      if (proj_test == false) boundary_ele = true;
    }
  }
  else if (dt_s == Core::FE::CellType::quad8)  //||
                                               // dt_s==Core::FE::CellType::quad8
                                               //|| dt_s==Core::FE::CellType::quad9)
  {
    for (int s_test = 0; s_test < 8; ++s_test)
    {
      if (s_test == 0)
      {
        source_xi_test[0] = -1.0;
        source_xi_test[1] = -1.0;
      }
      else if (s_test == 1)
      {
        source_xi_test[0] = 0.0;
        source_xi_test[1] = -1.0;
      }
      else if (s_test == 2)
      {
        source_xi_test[0] = 1.0;
        source_xi_test[1] = -1.0;
      }
      else if (s_test == 3)
      {
        source_xi_test[0] = -1.0;
        source_xi_test[1] = 0.0;
      }
      else if (s_test == 4)
      {
        source_xi_test[0] = 1.0;
        source_xi_test[1] = 0.0;
      }
      else if (s_test == 5)
      {
        source_xi_test[0] = -1.0;
        source_xi_test[1] = 1.0;
      }
      else if (s_test == 6)
      {
        source_xi_test[0] = 0.0;
        source_xi_test[1] = 1.0;
      }
      else if (s_test == 7)
      {
        source_xi_test[0] = 1.0;
        source_xi_test[1] = 1.0;
      }

      proj_test = false;
      for (int bs_test = 0; bs_test < (int)target_elems.size(); ++bs_test)
      {
        double target_xi_test[2] = {0.0, 0.0};
        Mortar::Projector::impl(source_elem, *target_elems[bs_test])
            ->project_gauss_point_3d(
                source_elem, source_xi_test, *target_elems[bs_test], target_xi_test, alpha_test);
        Core::FE::CellType dt = target_elems[bs_test]->shape();

        if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
            dt == Core::FE::CellType::quad9)
        {
          if (target_xi_test[0] >= -1.0 && target_xi_test[1] >= -1.0 && target_xi_test[0] <= 1.0 &&
              target_xi_test[1] <= 1.0)
          {
            // get hasproj
            source_elem.local_to_global(source_xi_test, glob_test, 0);
            for (int ii = 0; ii < source_elem.num_node(); ++ii)
            {
              Mortar::Node* mycnode_test = dynamic_cast<Mortar::Node*>(mynodes_test[ii]);
              if (!mycnode_test) FOUR_C_THROW("has_proj_status: Null pointer!");

              if (glob_test[0] > mycnode_test->xspatial()[0] - tol &&
                  glob_test[0] < mycnode_test->xspatial()[0] + tol &&
                  glob_test[1] > mycnode_test->xspatial()[1] - tol &&
                  glob_test[1] < mycnode_test->xspatial()[1] + tol &&
                  glob_test[2] > mycnode_test->xspatial()[2] - tol &&
                  glob_test[2] < mycnode_test->xspatial()[2] + tol)
                mycnode_test->has_proj() = true;
            }

            glob_test[0] = 0.0;
            glob_test[1] = 0.0;
            glob_test[2] = 0.0;

            proj_test = true;
          }
        }
        else if (dt == Core::FE::CellType::tri3 || dt == Core::FE::CellType::tri6)
        {
          if (target_xi_test[0] >= 0.0 && target_xi_test[1] >= 0.0 && target_xi_test[0] <= 1.0 &&
              target_xi_test[1] <= 1.0 && target_xi_test[0] + target_xi_test[1] <= 1.0)
          {
            // get hasproj
            source_elem.local_to_global(source_xi_test, glob_test, 0);
            for (int ii = 0; ii < source_elem.num_node(); ++ii)
            {
              Mortar::Node* mycnode_test = dynamic_cast<Mortar::Node*>(mynodes_test[ii]);
              if (!mycnode_test) FOUR_C_THROW("has_proj_status: Null pointer!");

              if (glob_test[0] > mycnode_test->xspatial()[0] - tol &&
                  glob_test[0] < mycnode_test->xspatial()[0] + tol &&
                  glob_test[1] > mycnode_test->xspatial()[1] - tol &&
                  glob_test[1] < mycnode_test->xspatial()[1] + tol &&
                  glob_test[2] > mycnode_test->xspatial()[2] - tol &&
                  glob_test[2] < mycnode_test->xspatial()[2] + tol)
                mycnode_test->has_proj() = true;
            }
            glob_test[0] = 0.0;
            glob_test[1] = 0.0;
            glob_test[2] = 0.0;

            proj_test = true;
          }
        }
        else
        {
          FOUR_C_THROW("Non valid element type for target discretization!");
        }
      }
      if (proj_test == false) boundary_ele = true;
    }
  }
  // TRI-ELE
  else if (dt_s == Core::FE::CellType::tri3)
  {
    for (int s_test = 0; s_test < 3; ++s_test)
    {
      if (s_test == 0)
      {
        source_xi_test[0] = 0.0;
        source_xi_test[1] = 0.0;
      }
      else if (s_test == 1)
      {
        source_xi_test[0] = 1.0;
        source_xi_test[1] = 0.0;
      }
      else if (s_test == 2)
      {
        source_xi_test[0] = 0.0;
        source_xi_test[1] = 1.0;
      }

      proj_test = false;
      for (int bs_test = 0; bs_test < (int)target_elems.size(); ++bs_test)
      {
        double target_xi_test[2] = {0.0, 0.0};
        Mortar::Projector::impl(source_elem, *target_elems[bs_test])
            ->project_gauss_point_3d(
                source_elem, source_xi_test, *target_elems[bs_test], target_xi_test, alpha_test);
        Core::FE::CellType dt = target_elems[bs_test]->shape();

        if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
            dt == Core::FE::CellType::quad9)
        {
          if (target_xi_test[0] >= -1.0 && target_xi_test[1] >= -1.0 && target_xi_test[0] <= 1.0 &&
              target_xi_test[1] <= 1.0)  // Falls Position auf Element
          {
            // get hasproj
            source_elem.local_to_global(source_xi_test, glob_test, 0);
            for (int ii = 0; ii < source_elem.num_node(); ++ii)
            {
              Mortar::Node* mycnode_test = dynamic_cast<Mortar::Node*>(mynodes_test[ii]);
              if (!mycnode_test) FOUR_C_THROW("has_proj_status: Null pointer!");

              if (glob_test[0] > mycnode_test->xspatial()[0] - tol &&
                  glob_test[0] < mycnode_test->xspatial()[0] + tol &&
                  glob_test[1] > mycnode_test->xspatial()[1] - tol &&
                  glob_test[1] < mycnode_test->xspatial()[1] + tol &&
                  glob_test[2] > mycnode_test->xspatial()[2] - tol &&
                  glob_test[2] < mycnode_test->xspatial()[2] + tol)
                mycnode_test->has_proj() = true;
            }

            glob_test[0] = 0.0;
            glob_test[1] = 0.0;
            glob_test[2] = 0.0;

            proj_test = true;
          }
        }
        else if (dt == Core::FE::CellType::tri3 || dt == Core::FE::CellType::tri6)
        {
          if (target_xi_test[0] >= 0.0 && target_xi_test[1] >= 0.0 && target_xi_test[0] <= 1.0 &&
              target_xi_test[1] <= 1.0 &&
              target_xi_test[0] + target_xi_test[1] <= 1.0)  // Falls Position auf Element
          {
            // get hasproj
            source_elem.local_to_global(source_xi_test, glob_test, 0);
            for (int ii = 0; ii < source_elem.num_node(); ++ii)
            {
              Mortar::Node* mycnode_test = dynamic_cast<Mortar::Node*>(mynodes_test[ii]);
              if (!mycnode_test) FOUR_C_THROW("has_proj_status: Null pointer!");

              if (glob_test[0] > mycnode_test->xspatial()[0] - tol &&
                  glob_test[0] < mycnode_test->xspatial()[0] + tol &&
                  glob_test[1] > mycnode_test->xspatial()[1] - tol &&
                  glob_test[1] < mycnode_test->xspatial()[1] + tol &&
                  glob_test[2] > mycnode_test->xspatial()[2] - tol &&
                  glob_test[2] < mycnode_test->xspatial()[2] + tol)
                mycnode_test->has_proj() = true;
            }

            glob_test[0] = 0.0;
            glob_test[1] = 0.0;
            glob_test[2] = 0.0;

            proj_test = true;
          }
        }
        else
        {
          FOUR_C_THROW("Non valid element type for target discretization!");
        }
      }
      if (proj_test == false) boundary_ele = true;
    }
  }
  else if (dt_s == Core::FE::CellType::tri6)
  {
    for (int s_test = 0; s_test < 6; ++s_test)
    {
      if (s_test == 0)
      {
        source_xi_test[0] = 0.0;
        source_xi_test[1] = 0.0;
      }
      else if (s_test == 1)
      {
        source_xi_test[0] = 0.5;
        source_xi_test[1] = 0.0;
      }
      else if (s_test == 2)
      {
        source_xi_test[0] = 1.0;
        source_xi_test[1] = 0.0;
      }
      else if (s_test == 3)
      {
        source_xi_test[0] = 0.0;
        source_xi_test[1] = 0.5;
      }
      else if (s_test == 4)
      {
        source_xi_test[0] = 0.5;
        source_xi_test[1] = 0.5;
      }
      else if (s_test == 5)
      {
        source_xi_test[0] = 0.0;
        source_xi_test[1] = 1.0;
      }

      proj_test = false;
      for (int bs_test = 0; bs_test < (int)target_elems.size(); ++bs_test)
      {
        double target_xi_test[2] = {0.0, 0.0};
        Mortar::Projector::impl(source_elem, *target_elems[bs_test])
            ->project_gauss_point_3d(
                source_elem, source_xi_test, *target_elems[bs_test], target_xi_test, alpha_test);
        Core::FE::CellType dt = target_elems[bs_test]->shape();

        if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
            dt == Core::FE::CellType::quad9)
        {
          if (target_xi_test[0] >= -1.0 && target_xi_test[1] >= -1.0 && target_xi_test[0] <= 1.0 &&
              target_xi_test[1] <= 1.0)  // Falls Position auf Element
          {
            // get hasproj
            source_elem.local_to_global(source_xi_test, glob_test, 0);
            for (int ii = 0; ii < source_elem.num_node(); ++ii)
            {
              Mortar::Node* mycnode_test = dynamic_cast<Mortar::Node*>(mynodes_test[ii]);
              if (!mycnode_test) FOUR_C_THROW("has_proj_status: Null pointer!");

              if (glob_test[0] > mycnode_test->xspatial()[0] - tol &&
                  glob_test[0] < mycnode_test->xspatial()[0] + tol &&
                  glob_test[1] > mycnode_test->xspatial()[1] - tol &&
                  glob_test[1] < mycnode_test->xspatial()[1] + tol &&
                  glob_test[2] > mycnode_test->xspatial()[2] - tol &&
                  glob_test[2] < mycnode_test->xspatial()[2] + tol)
                mycnode_test->has_proj() = true;
            }

            glob_test[0] = 0.0;
            glob_test[1] = 0.0;
            glob_test[2] = 0.0;

            proj_test = true;
          }
        }
        else if (dt == Core::FE::CellType::tri3 || dt == Core::FE::CellType::tri6)
        {
          if (target_xi_test[0] >= 0.0 && target_xi_test[1] >= 0.0 && target_xi_test[0] <= 1.0 &&
              target_xi_test[1] <= 1.0 &&
              target_xi_test[0] + target_xi_test[1] <= 1.0)  // Falls Position auf Element
          {
            // get hasproj
            source_elem.local_to_global(source_xi_test, glob_test, 0);
            for (int ii = 0; ii < source_elem.num_node(); ++ii)
            {
              Mortar::Node* mycnode_test = dynamic_cast<Mortar::Node*>(mynodes_test[ii]);
              if (!mycnode_test) FOUR_C_THROW("has_proj_status: Null pointer!");

              if (glob_test[0] > mycnode_test->xspatial()[0] - tol &&
                  glob_test[0] < mycnode_test->xspatial()[0] + tol &&
                  glob_test[1] > mycnode_test->xspatial()[1] - tol &&
                  glob_test[1] < mycnode_test->xspatial()[1] + tol &&
                  glob_test[2] > mycnode_test->xspatial()[2] - tol &&
                  glob_test[2] < mycnode_test->xspatial()[2] + tol)
                mycnode_test->has_proj() = true;
            }

            glob_test[0] = 0.0;
            glob_test[1] = 0.0;
            glob_test[2] = 0.0;

            proj_test = true;
          }
        }
        else
        {
          FOUR_C_THROW("Non valid element type for target discretization!");
        }
      }
      if (proj_test == false) boundary_ele = true;
    }
  }
  else
  {
    FOUR_C_THROW("Non valid element type for source discretization!");
  }

  return boundary_ele;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_deriv_ele_3d(Mortar::Element& source_elem,
    std::vector<Mortar::Element*> target_elems, bool* boundary_ele, bool* proj_, MPI_Comm comm,
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  std::shared_ptr<CONTACT::ParamsInterface> cparams_ptr = nullptr;
  if (mparams_ptr)
  {
    cparams_ptr = std::dynamic_pointer_cast<CONTACT::ParamsInterface>(mparams_ptr);
  }
  integrate_deriv_ele_3d(source_elem, target_elems, boundary_ele, proj_, comm, cparams_ptr);
}

/*----------------------------------------------------------------------*
 |  Integrate and linearize for lin and quad elements        farah 01/13|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_deriv_ele_3d(Mortar::Element& source_elem,
    std::vector<Mortar::Element*> target_elems, bool* boundary_ele, bool* proj_, MPI_Comm comm,
    const std::shared_ptr<CONTACT::ParamsInterface>& cparams_ptr)
{
  // explicitly defined shape function type needed
  if (shape_fcn() == Mortar::shape_undefined)
    FOUR_C_THROW("Function is called without specific shape function defined!");

  // Petrov-Galerkin approach for LM not yet implemented
  if (shape_fcn() == Mortar::shape_petrovgalerkin &&
      source_elem.shape() != Core::FE::CellType::nurbs9)
    FOUR_C_THROW("Petrov-Galerkin approach not yet implemented for 3-D quadratic FE interpolation");

  // check for problem dimension
  if (n_dim() != 3) FOUR_C_THROW("3D integration method called for non-3D problem");

  // get source element nodes themselves for normal evaluation
  Core::Nodes::Node** mynodes = source_elem.nodes();
  if (!mynodes) FOUR_C_THROW("Null pointer!");

  // check input data
  for (int test = 0; test < (int)target_elems.size(); ++test)
  {
    if (((!source_elem.is_source()) || (target_elems[test]->is_source())) and
        (!imortar_.get<bool>("Two_half_pass")))
      FOUR_C_THROW("Function is called on a wrong type of Mortar::Element pair!");
  }

  int msize = target_elems.size();
  int nrow = source_elem.num_node();
  int ndof = dynamic_cast<Mortar::Node*>(source_elem.nodes()[0])->num_dof();

  // create empty vectors for shape fct. evaluation
  Core::LinAlg::SerialDenseVector source_val(nrow);
  Core::LinAlg::SerialDenseMatrix source_deriv(nrow, 2, true);
  Core::LinAlg::SerialDenseVector lm_val(nrow);
  Core::LinAlg::SerialDenseMatrix lm_deriv(nrow, 2, true);

  // prepare directional derivative of dual shape functions
  // this is necessary for all source element types except tri3
  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> dualmap(
      nrow * ndof, 0, Core::LinAlg::SerialDenseMatrix(nrow, nrow));
  if ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
      (source_elem.shape() != Core::FE::CellType::tri3 ||
          source_elem.mo_data().deriv_dual_shape() != nullptr) &&
      lag_mult_quad() != Mortar::lagmult_const)
    source_elem.deriv_shape_dual(dualmap);

  // check whether quadratic elements with linear interpolation are present
  bool linlm = false;
  if ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
      (lag_mult_quad() == Mortar::lagmult_lin) &&
      (source_elem.shape() == Core::FE::CellType::quad8 ||
          source_elem.shape() == Core::FE::CellType::tri6))
    linlm = true;

  //********************************************************************
  //  Boundary_segmentation test -- HasProj() check
  //  if a source-node has no projection onto each target element
  //  --> Boundary_ele==true
  //********************************************************************
  auto integrationtype = Teuchos::getIntegralValue<Mortar::IntType>(imortar_, "INTTYPE");

  //************************************************************************
  // Boundary Segmentation check -- HasProj()-check
  //************************************************************************
  if (source_elem.shape() != Core::FE::CellType::nurbs4 and
      source_elem.shape() != Core::FE::CellType::nurbs9)
    *boundary_ele = boundary_segm_check_3d(source_elem, target_elems);

  int linsize = 0;
  for (int i = 0; i < nrow; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(mynodes[i]);
    linsize += cnode->get_linsize();
  }

  // Start integration if fast integration should be used or if there is no boundary element
  // for the fast_BS integration
  if (*boundary_ele == false || integrationtype == Mortar::inttype_elements)
  {
    //**********************************************************************
    // loop over all Gauss points for integration
    //**********************************************************************
    for (int gp = 0; gp < n_gp(); ++gp)
    {
      int iter_proj = 0;
      // coordinates and weight
      std::array<double, 2> eta = {coordinate(gp, 0), coordinate(gp, 1)};
      double wgt = weight(gp);

      // note that the third component of source_xi is necessary!
      // (although it will always be 0.0 of course)
      double source_xi[2] = {0.0, 0.0};
      double target_xi[2] = {0.0, 0.0};
      double projalpha = 0.0;

      // get Gauss point in source element coordinates
      source_xi[0] = eta[0];
      source_xi[1] = eta[1];

      bool is_on_mele = true;

      // evaluate Lagrange multiplier shape functions (on source element)
      if (lag_mult_quad() == Mortar::lagmult_const)
        source_elem.evaluate_shape_lag_mult_const(shape_fcn(), source_xi, lm_val, lm_deriv, nrow);
      else if (lag_mult_quad() == Mortar::lagmult_lin)
        source_elem.evaluate_shape_lag_mult_lin(shape_fcn(), source_xi, lm_val, lm_deriv, nrow);
      else
        source_elem.evaluate_shape_lag_mult(shape_fcn(), source_xi, lm_val, lm_deriv, nrow);

      // evaluate trace space shape functions (on both elements)
      source_elem.evaluate_shape(source_xi, source_val, source_deriv, nrow, linlm);

      // evaluate the two Jacobians (int. cell and source element)
      double jacsource = source_elem.jacobian(source_xi);

      // evaluate linearizations *******************************************
      // evaluate the source Jacobian derivative
      Core::Gen::Pairedvector<int, double> jacsourcemap(nrow * ndof + linsize);
      source_elem.deriv_jacobian(source_xi, jacsourcemap);

      //**********************************************************************
      // loop over all target_elem
      //**********************************************************************
      for (int numtarget = 0; numtarget < msize; ++numtarget)
      {
        Core::FE::CellType dt = target_elems[numtarget]->shape();

        int nmnode = target_elems[numtarget]->num_node();
        Core::LinAlg::SerialDenseVector target_val(nmnode);
        Core::LinAlg::SerialDenseMatrix target_deriv(nmnode, 2, true);

        // project Gauss point onto target element
        Mortar::Projector::impl(source_elem, *target_elems[numtarget])
            ->project_gauss_point_3d(
                source_elem, source_xi, *target_elems[numtarget], target_xi, projalpha);

        // check GP projection
        const double tol = 0.00;
        is_on_mele = true;
        if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
            dt == Core::FE::CellType::quad9 || dt == Core::FE::CellType::nurbs9)
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

        // gp is valid
        if (is_on_mele == true)
        {
          *proj_ = true;
          iter_proj += 1;

          // get target_val
          target_elems[numtarget]->evaluate_shape(target_xi, target_val, target_deriv, nmnode);

          // evaluate the GP source coordinate derivatives
          std::vector<Core::Gen::Pairedvector<int, double>> d_source_xi_gp(2, 0);
          std::vector<Core::Gen::Pairedvector<int, double>> d_target_xi_gp(
              2, 4 * linsize + nmnode * ndof);
          deriv_xi_gp_3d(source_elem, *target_elems[numtarget], source_xi, target_xi,
              d_source_xi_gp, d_target_xi_gp, projalpha);

          //**********************************************************************
          // geometric quantities
          //**********************************************************************
          double gpn[3] = {0.0, 0.0, 0.0};
          Core::Gen::Pairedvector<int, double> dgapgp(
              (nmnode * ndof) + linsize);  // gap lin. without lm and jac.
          double gap = 0.0;
          std::vector<Core::Gen::Pairedvector<int, double>> dnmap_unit(
              3, ((nmnode * ndof) + linsize));  // deriv of x,y and z comp. of gpn (unit)

          //**********************************************************************
          // evaluate at GP and lin char. quantities
          //**********************************************************************
          if (linlm)
          {
            // declare and compute shape functions as well as derivatives for computation of gap
            // -> standard quadratic functions instead of only linear functions
            Core::LinAlg::SerialDenseVector svalgap(nrow);
            Core::LinAlg::SerialDenseMatrix sderivgap(nrow, 2, true);
            source_elem.evaluate_shape(source_xi, svalgap, sderivgap, nrow);

            gap_3d(source_elem, *target_elems[numtarget], svalgap, target_val, sderivgap,
                target_deriv, &gap, gpn, d_source_xi_gp, d_target_xi_gp, dgapgp, dnmap_unit);
          }
          else
            gap_3d(source_elem, *target_elems[numtarget], source_val, target_val, source_deriv,
                target_deriv, &gap, gpn, d_source_xi_gp, d_target_xi_gp, dgapgp, dnmap_unit);

          if (algo_ == Mortar::algorithm_mortar)
            dynamic_cast<CONTACT::Element&>(source_elem).prepare_mderiv(target_elems, numtarget);

          integrate_gp_3d(source_elem, *target_elems[numtarget], source_val, lm_val, target_val,
              source_deriv, target_deriv, lm_deriv, dualmap, wgt, jacsource, jacsourcemap, gpn,
              dnmap_unit, gap, dgapgp, source_xi, target_xi, d_source_xi_gp, d_target_xi_gp);

          // assemble m-matrix for this source/target pair
          if (algo_ == Mortar::algorithm_mortar)
            dynamic_cast<CONTACT::Element&>(source_elem)
                .assemble_mderiv_to_nodes(*target_elems[numtarget]);

        }  // is_on_mele
        if (is_on_mele == true) break;
      }  // target_elem loop

      // warning, if an element which is declared not to be on the boundary by the above test
      // has non-projectable Gauss points
      if (is_on_mele == false && *boundary_ele == false)
      {
        std::cout << "*** warning *** Non-boundary element has non-projectable Gauss point \n";
        Core::Nodes::Node** source_nodes = source_elem.nodes();
        for (int e = 0; e < source_elem.num_node(); ++e)
        {
          Node* cnode = dynamic_cast<Node*>(source_nodes[e]);
          std::cout << "#eID " << source_elem.id() << " | #nID " << cnode->id() << ": "
                    << cnode->x()[0] << ", " << cnode->x()[1] << ", " << cnode->x()[2] << std::endl;
        }
      }

      // if one gp has counterparts on 2 elements --> non-uniqueness
      if (iter_proj > 1) FOUR_C_THROW("Multiple feasible projections of one integration point!");
    }  // GP-loop
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_deriv_cell_3d_aux_plane(Mortar::Element& source_elem,
    Mortar::Element& target_elem, std::shared_ptr<Mortar::IntCell> cell, double* auxn,
    MPI_Comm comm, const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  std::shared_ptr<CONTACT::ParamsInterface> cparams_ptr = nullptr;
  if (mparams_ptr)
  {
    cparams_ptr = std::dynamic_pointer_cast<CONTACT::ParamsInterface>(mparams_ptr);
  }
  integrate_deriv_cell_3d_aux_plane(source_elem, target_elem, cell, auxn, comm, cparams_ptr);
}

/*----------------------------------------------------------------------*
 |  Integrate and linearize a 2D source / target cell (3D)     popp 03/09|
 |  This method integrates the cell M matrix and weighted gap g~        |
 |  and stores it in mseg and gseg respectively. Moreover, derivatives  |
 |  LinM and Ling are built and stored directly into the adjacent nodes.|
 |  (Thus this method combines EVERYTHING before done separately in     |
 |  IntegrateM3D, IntegrateG3D, DerivM3D and DerivG3D!)                 |
 |  This is the auxiliary plane coupling version!!!                     |
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_deriv_cell_3d_aux_plane(Mortar::Element& source_elem,
    Mortar::Element& target_elem, std::shared_ptr<Mortar::IntCell> cell, double* auxn,
    MPI_Comm comm, const std::shared_ptr<CONTACT::ParamsInterface>& cparams_ptr)
{
  // explicitly defined shape function type needed
  if (shape_fcn() == Mortar::shape_undefined)
    FOUR_C_THROW(
        "integrate_deriv_cell_3d_aux_plane called without specific shape function defined!");

  // check for problem dimension
  if (n_dim() != 3) FOUR_C_THROW("3D integration method called for non-3D problem");

  // discretization type of target element
  Core::FE::CellType sdt = source_elem.shape();
  Core::FE::CellType mdt = target_elem.shape();

  // check input data
  if (((!source_elem.is_source()) || (target_elem.is_source())) and
      (!imortar_.get<bool>("Two_half_pass")))
    FOUR_C_THROW(
        "integrate_deriv_cell_3d_aux_plane called on a wrong type of Mortar::Element pair!");
  if (cell == nullptr)
    FOUR_C_THROW("integrate_deriv_cell_3d_aux_plane called without integration cell");

  // number of nodes (source, target)
  int nrow = source_elem.num_node();
  int ncol = target_elem.num_node();
  int ndof = n_dim();

  // get source element nodes themselves for normal evaluation
  Core::Nodes::Node** mynodes = source_elem.nodes();
  if (!mynodes) FOUR_C_THROW("Null pointer!");

  // create empty vectors for shape fct. evaluation
  Core::LinAlg::SerialDenseVector source_val(nrow);
  Core::LinAlg::SerialDenseMatrix source_deriv(nrow, 2, true);
  Core::LinAlg::SerialDenseVector target_val(ncol);
  Core::LinAlg::SerialDenseMatrix target_deriv(ncol, 2, true);
  Core::LinAlg::SerialDenseVector lm_val(nrow);
  Core::LinAlg::SerialDenseMatrix lm_deriv(nrow, 2, true);
  Core::LinAlg::SerialDenseVector lmvalScale(nrow);
  Core::LinAlg::SerialDenseMatrix lmderivScale(nrow, 2, true);



  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  int linsize = 0;
  if (cppnormal_)
  {
    for (int i = 0; i < ncol; ++i)
    {
      Node* cnode = dynamic_cast<Node*>(target_elem.nodes()[i]);
      linsize += cnode->get_linsize();
    }
    linsize += 1 + target_elem.num_node();
  }
  else
  {
    for (int i = 0; i < nrow; ++i)
    {
      Node* cnode = dynamic_cast<Node*>(mynodes[i]);
      linsize += cnode->get_linsize();
    }
  }

  // for cpp normal we have more lin entries
  linsize *= 10;

  // prepare directional derivative of dual shape functions
  // this is necessary for all source element types except tri3
  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> dualmap(
      (nrow + ncol) * ndof, 0, Core::LinAlg::SerialDenseMatrix(nrow, nrow));
  if ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
      (source_elem.shape() != Core::FE::CellType::tri3 ||
          source_elem.mo_data().deriv_dual_shape() != nullptr))
    source_elem.deriv_shape_dual(dualmap);

  // check if the cells are tri3
  // there's nothing wrong about other shapes, but as long as they are all
  // tri3 we can perform the jacobian calculation ( and its deriv) outside
  // the Gauss point loop
  if (cell->shape() != Core::FE::CellType::tri3)
    FOUR_C_THROW("only tri3 integration cells at the moment. See comment in the code");

  double jac = cell->jacobian();
  // directional derivative of cell Jacobian
  Core::Gen::Pairedvector<int, double> jacintcellmap((nrow + ncol) * ndof);
  cell->deriv_jacobian(jacintcellmap);

  //**********************************************************************
  // loop over all Gauss points for integration
  //**********************************************************************
  for (int gp = 0; gp < n_gp(); ++gp)
  {
    // coordinates and weight
    double eta[2] = {coordinate(gp, 0), coordinate(gp, 1)};
    double wgt = weight(gp);

    // get global Gauss point coordinates
    double globgp[3] = {0.0, 0.0, 0.0};
    cell->local_to_global(eta, globgp, 0);

    double source_xi[2] = {0.0, 0.0};
    double target_xi[2] = {0.0, 0.0};

    // project Gauss point onto source element
    // project Gauss point onto target element
    double sprojalpha = 0.0;
    double mprojalpha = 0.0;
    Mortar::Projector::impl(source_elem)
        ->project_gauss_point_auxn_3d(globgp, auxn, source_elem, source_xi, sprojalpha);
    Mortar::Projector::impl(target_elem)
        ->project_gauss_point_auxn_3d(globgp, auxn, target_elem, target_xi, mprojalpha);

    // check GP projection (SOURCE)
    double tol = 0.01;
    if (sdt == Core::FE::CellType::quad4 || sdt == Core::FE::CellType::quad8 ||
        sdt == Core::FE::CellType::quad9)
    {
      if (source_xi[0] < -1.0 - tol || source_xi[1] < -1.0 - tol || source_xi[0] > 1.0 + tol ||
          source_xi[1] > 1.0 + tol)
      {
        std::cout
            << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Gauss point projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Source GP projection: " << source_xi[0] << " " << source_xi[1] << std::endl;
      }
    }
    else
    {
      if (source_xi[0] < -tol || source_xi[1] < -tol || source_xi[0] > 1.0 + tol ||
          source_xi[1] > 1.0 + tol || source_xi[0] + source_xi[1] > 1.0 + 2 * tol)
      {
        std::cout
            << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Gauss point projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Source GP projection: " << source_xi[0] << " " << source_xi[1] << std::endl;
      }
    }

    // check GP projection (TARGET)
    if (mdt == Core::FE::CellType::quad4 || mdt == Core::FE::CellType::quad8 ||
        mdt == Core::FE::CellType::quad9)
    {
      if (target_xi[0] < -1.0 - tol || target_xi[1] < -1.0 - tol || target_xi[0] > 1.0 + tol ||
          target_xi[1] > 1.0 + tol)
      {
        std::cout
            << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Gauss point projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Target GP projection: " << target_xi[0] << " " << target_xi[1] << std::endl;
      }
    }
    else
    {
      if (target_xi[0] < -tol || target_xi[1] < -tol || target_xi[0] > 1.0 + tol ||
          target_xi[1] > 1.0 + tol || target_xi[0] + target_xi[1] > 1.0 + 2 * tol)
      {
        std::cout
            << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Gauss point projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Target GP projection: " << target_xi[0] << " " << target_xi[1] << std::endl;
      }
    }

    // evaluate Lagrange multiplier shape functions (on source element)
    source_elem.evaluate_shape_lag_mult(shape_fcn(), source_xi, lm_val, lm_deriv, nrow);
    source_elem.evaluate_shape_lag_mult(
        shape_fcn(), source_xi, lmvalScale, lmderivScale, nrow, false);

    // evaluate trace space shape functions (on both elements)
    source_elem.evaluate_shape(source_xi, source_val, source_deriv, nrow);
    target_elem.evaluate_shape(target_xi, target_val, target_deriv, ncol);

    // evaluate linearizations *******************************************

    // evaluate global GP coordinate derivative
    static Core::LinAlg::Matrix<3, 1> svalcell;
    static Core::LinAlg::Matrix<3, 2> sderivcell;
    cell->evaluate_shape(eta, svalcell, sderivcell);

    Core::Gen::Pairedvector<int, Core::LinAlg::Matrix<3, 1>> lingp((nrow + ncol) * ndof);

    for (int v = 0; v < 3; ++v)
      for (int d = 0; d < 3; ++d)
        for (CI p = (cell->get_deriv_vertex(v))[d].begin();
            p != (cell->get_deriv_vertex(v))[d].end(); ++p)
          lingp[p->first](d) += svalcell(v) * (p->second);

    // evaluate the GP source coordinate derivatives
    std::vector<Core::Gen::Pairedvector<int, double>> d_source_xi_gp(2, (nrow + ncol) * ndof);
    deriv_xi_gp_3d_aux_plane(source_elem, source_xi, cell->auxn(), d_source_xi_gp, sprojalpha,
        cell->get_deriv_auxn(), lingp);

    // evaluate the GP target coordinate derivatives
    std::vector<Core::Gen::Pairedvector<int, double>> d_target_xi_gp(2, (nrow + ncol) * ndof);
    deriv_xi_gp_3d_aux_plane(target_elem, target_xi, cell->auxn(), d_target_xi_gp, mprojalpha,
        cell->get_deriv_auxn(), lingp);

    //**********************************************************************
    // geometric quantities
    //**********************************************************************
    double gpn[3] = {0.0, 0.0, 0.0};
    Core::Gen::Pairedvector<int, double> dgapgp(
        (ncol * ndof) + linsize);  // gap lin. without lm and jac.
    double gap = 0.0;
    std::vector<Core::Gen::Pairedvector<int, double>> dnmap_unit(
        3, linsize);  // deriv of x,y and z comp. of gpn (unit)
    gap_3d(source_elem, target_elem, source_val, target_val, source_deriv, target_deriv, &gap, gpn,
        d_source_xi_gp, d_target_xi_gp, dgapgp, dnmap_unit);

    // integrate for area scaling:
    for (int i = 0; i < source_elem.num_node(); ++i)
    {
      dynamic_cast<CONTACT::Node*>(source_elem.nodes()[i])->mo_data().get_dscale() +=
          wgt * jac * source_val[i] * lmvalScale[i];
    }

    //**********************************************************************
    // evaluate at GP and lin char. quantities
    //**********************************************************************
    if (!nonsmoothselfcontactsurface_)
      integrate_gp_3d(source_elem, target_elem, source_val, lm_val, target_val, source_deriv,
          target_deriv, lm_deriv, dualmap, wgt, jac, jacintcellmap, gpn, dnmap_unit, gap, dgapgp,
          source_xi, target_xi, d_source_xi_gp, d_target_xi_gp);
    // for non-smooth (self) contact surfaces we do not use the smoothed normal, but instead we use
    // the normal that is already used for the projection
    else
      integrate_gp_3d(source_elem, target_elem, source_val, lm_val, target_val, source_deriv,
          target_deriv, lm_deriv, dualmap, wgt, jac, jacintcellmap, cell->auxn(),
          cell->get_deriv_auxn(), gap, dgapgp, source_xi, target_xi, d_source_xi_gp,
          d_target_xi_gp);
  }  // end gp loop
}



/*----------------------------------------------------------------------*
 |  Integrate and linearize a 1D source / target cell (2D)    farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_deriv_cell_3d_aux_plane_stl(Mortar::Element& target_elem,
    Mortar::Element& lele, Mortar::Element& source_elem, std::shared_ptr<Mortar::IntCell> cell,
    double* auxn, MPI_Comm comm)
{
  // explicitly defined shape function type needed
  if (shape_fcn() == Mortar::shape_undefined)
    FOUR_C_THROW(
        "integrate_deriv_cell_3d_aux_plane called without specific shape function defined!");

  // check for problem dimension
  if (n_dim() != 3) FOUR_C_THROW("3D integration method called for non-3D problem");

  // discretization type of target element
  Core::FE::CellType sdt = source_elem.shape();
  Core::FE::CellType mdt = target_elem.shape();

  // check input data
  if ((!source_elem.is_source()) || (target_elem.is_source()))
    FOUR_C_THROW(
        "integrate_deriv_cell_3d_aux_plane called on a wrong type of Mortar::Element pair!");
  if (cell == nullptr)
    FOUR_C_THROW("integrate_deriv_cell_3d_aux_plane called without integration cell");

  // number of nodes (source, target)
  //  int nrowL = lele.num_node();
  int nrow = source_elem.num_node();
  int ncolP = target_elem.num_node();
  int ncolL = lele.num_node();
  int ndof = n_dim();

  // get source element nodes themselves for normal evaluation
  Core::Nodes::Node** mynodes = source_elem.nodes();
  if (!mynodes) FOUR_C_THROW("Null pointer!");
  Core::Nodes::Node** target_nodes = lele.nodes();
  if (!target_nodes) FOUR_C_THROW("Null pointer!");

  // create empty vectors for shape fct. evaluation
  Core::LinAlg::SerialDenseVector source_val(nrow);
  Core::LinAlg::SerialDenseMatrix source_deriv(nrow, 2, true);
  Core::LinAlg::SerialDenseVector target_val(ncolL);
  Core::LinAlg::SerialDenseMatrix target_deriv(ncolL, 1, true);
  Core::LinAlg::SerialDenseVector lm_val(nrow);
  Core::LinAlg::SerialDenseMatrix lm_deriv(nrow, 2, true);

  // prepare directional derivative of dual shape functions
  // this is necessary for all source element types except tri3
  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> dualmap(
      (nrow + ncolL) * ndof, 0, Core::LinAlg::SerialDenseMatrix(nrow, nrow));
  if ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
      (lele.shape() != Core::FE::CellType::line2 ||
          source_elem.mo_data().deriv_dual_shape() != nullptr))
    source_elem.deriv_shape_dual(dualmap);

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  int linsize = 0;
  for (int i = 0; i < nrow; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(mynodes[i]);
    linsize += cnode->get_linsize();
  }

  // check if the cells are tri3
  // there's nothing wrong about other shapes, but as long as they are all
  // tri3 we can perform the jacobian calculation ( and its deriv) outside
  // the Gauss point loop
  if (cell->shape() != Core::FE::CellType::line2)
    FOUR_C_THROW("only line2 integration cells for LTS");

  double jac = cell->jacobian();

  // directional derivative of cell Jacobian
  Core::Gen::Pairedvector<int, double> jacintcellmap((nrow + ncolL) * ndof);
  cell->deriv_jacobian(jacintcellmap);

  //**********************************************************************
  // loop over all Gauss points for integration
  //**********************************************************************
  for (int gp = 0; gp < n_gp(); ++gp)
  {
    // coordinates and weight
    double eta[2] = {coordinate(gp, 0), coordinate(gp, 1)};
    double wgt = weight(gp);

    // get global Gauss point coordinates
    double globgp[3] = {0.0, 0.0, 0.0};
    cell->local_to_global(eta, globgp, 0);

    // para coordinate on surface source ele
    double source_xi[2] = {0.0, 0.0};

    // para coordinate on surface target ele
    double target_xi[2] = {0.0, 0.0};

    // para coordinate on line (edge) source ele
    double lxi[2] = {0.0, 0.0};

    // project Gauss point onto source/target surface element
    double sprojalpha = 0.0;
    double mprojalpha = 0.0;
    Mortar::Projector::impl(source_elem)
        ->project_gauss_point_auxn_3d(globgp, auxn, source_elem, source_xi, sprojalpha);
    Mortar::Projector::impl(target_elem)
        ->project_gauss_point_auxn_3d(globgp, auxn, target_elem, target_xi, mprojalpha);

    // check GP projection (SOURCE)
    double tol = 0.01;
    if (sdt == Core::FE::CellType::quad4 || sdt == Core::FE::CellType::quad8 ||
        sdt == Core::FE::CellType::quad9)
    {
      if (source_xi[0] < -1.0 - tol || source_xi[1] < -1.0 - tol || source_xi[0] > 1.0 + tol ||
          source_xi[1] > 1.0 + tol)
      {
        std::cout
            << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Gauss point projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Source GP projection: " << source_xi[0] << " " << source_xi[1] << std::endl;
      }
    }
    else
    {
      if (source_xi[0] < -tol || source_xi[1] < -tol || source_xi[0] > 1.0 + tol ||
          source_xi[1] > 1.0 + tol || source_xi[0] + source_xi[1] > 1.0 + 2 * tol)
      {
        std::cout
            << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Gauss point projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Source GP projection: " << source_xi[0] << " " << source_xi[1] << std::endl;
      }
    }

    // check GP projection (TARGET)
    if (mdt == Core::FE::CellType::quad4 || mdt == Core::FE::CellType::quad8 ||
        mdt == Core::FE::CellType::quad9)
    {
      if (target_xi[0] < -1.0 - tol || target_xi[1] < -1.0 - tol || target_xi[0] > 1.0 + tol ||
          target_xi[1] > 1.0 + tol)
      {
        std::cout
            << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Gauss point projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Target GP projection: " << target_xi[0] << " " << target_xi[1] << std::endl;
      }
    }
    else
    {
      if (target_xi[0] < -tol || target_xi[1] < -tol || target_xi[0] > 1.0 + tol ||
          target_xi[1] > 1.0 + tol || target_xi[0] + target_xi[1] > 1.0 + 2 * tol)
      {
        std::cout
            << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Gauss point projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Target GP projection: " << target_xi[0] << " " << target_xi[1] << std::endl;
      }
    }

    // convert source_xi to lsxi!!!
    if (lele.id() == 0)
    {
      lxi[0] = target_xi[0];
    }
    else if (lele.id() == 1)
    {
      lxi[0] = target_xi[1];
    }
    else if (lele.id() == 2)
    {
      lxi[0] = -target_xi[0];
    }
    else if (lele.id() == 3)
    {
      lxi[0] = -target_xi[1];
    }
    else
      FOUR_C_THROW("invalid local id of line ele");

    // evaluate Lagrange multiplier shape functions (on source element)
    source_elem.evaluate_shape_lag_mult(shape_fcn(), source_xi, lm_val, lm_deriv, nrow);

    // evaluate trace space shape functions (on both elements)
    source_elem.evaluate_shape(source_xi, source_val, source_deriv, nrow);
    lele.evaluate_shape(lxi, target_val, target_deriv, ncolL);

    // evaluate linearizations *******************************************

    // evaluate global GP coordinate derivative
    static Core::LinAlg::Matrix<3, 1> svalcell;
    static Core::LinAlg::Matrix<3, 2> sderivcell;
    cell->evaluate_shape(eta, svalcell, sderivcell);

    Core::Gen::Pairedvector<int, Core::LinAlg::Matrix<3, 1>> lingp(100 * (nrow + ncolP) * ndof);

    for (int v = 0; v < 2; ++v)
      for (int d = 0; d < 3; ++d)
        for (CI p = (cell->get_deriv_vertex(v))[d].begin();
            p != (cell->get_deriv_vertex(v))[d].end(); ++p)
          lingp[p->first](d) += svalcell(v) * (p->second);

    // evaluate the GP source coordinate derivatives
    std::vector<Core::Gen::Pairedvector<int, double>> d_source_xi_gp(2, (nrow + ncolP) * ndof);
    deriv_xi_gp_3d_aux_plane(source_elem, source_xi, cell->auxn(), d_source_xi_gp, sprojalpha,
        cell->get_deriv_auxn(), lingp);

    // evaluate the GP target coordinate derivatives
    std::vector<Core::Gen::Pairedvector<int, double>> d_target_xi_gp(
        2, (nrow + ncolP) * 100 * ndof);
    deriv_xi_gp_3d_aux_plane(target_elem, target_xi, cell->auxn(), d_target_xi_gp, mprojalpha,
        cell->get_deriv_auxn(), lingp);

    // convert deriv source_xi
    std::vector<Core::Gen::Pairedvector<int, double>> dlxigp(1, 100 * (nrow + ncolL) * ndof);

    if (lele.id() == 0)
    {
      for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
        dlxigp[0][p->first] += (p->second);
    }
    else if (lele.id() == 1)
    {
      for (CI p = d_target_xi_gp[1].begin(); p != d_target_xi_gp[1].end(); ++p)
        dlxigp[0][p->first] += (p->second);
    }
    else if (lele.id() == 2)
    {
      for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
        dlxigp[0][p->first] -= (p->second);
    }
    else if (lele.id() == 3)
    {
      for (CI p = d_target_xi_gp[1].begin(); p != d_target_xi_gp[1].end(); ++p)
        dlxigp[0][p->first] -= (p->second);
    }
    else
      FOUR_C_THROW("invalid local id of line ele");

    //**********************************************************************
    // geometric quantities
    //**********************************************************************
    std::array<double, 3> gpn = {0.0, 0.0, 0.0};
    Core::Gen::Pairedvector<int, double> dgapgp(
        (ncolL * ndof) + 10 * linsize);  // gap lin. without lm and jac.
    double gap = 0.0;
    std::vector<Core::Gen::Pairedvector<int, double>> dnmap_unit(
        3, 10 * linsize);  // deriv of x,y and z comp. of gpn (unit)

    //**********************************************************************
    // evaluate at GP and lin char. quantities
    //**********************************************************************
    std::array<double, 3> sgpx = {0.0, 0.0, 0.0};
    std::array<double, 3> mgpx = {0.0, 0.0, 0.0};

    for (int i = 0; i < nrow; ++i)
    {
      Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[i]);
      gpn[0] += source_val[i] * mymrtrnode->mo_data().n()[0];
      gpn[1] += source_val[i] * mymrtrnode->mo_data().n()[1];
      gpn[2] += source_val[i] * mymrtrnode->mo_data().n()[2];

      sgpx[0] += source_val[i] * source_elem.get_nodal_coords(0, i);
      sgpx[1] += source_val[i] * source_elem.get_nodal_coords(1, i);
      sgpx[2] += source_val[i] * source_elem.get_nodal_coords(2, i);
    }

    // build interpolation of target GP coordinates
    for (int i = 0; i < ncolL; ++i)
    {
      mgpx[0] += target_val[i] * lele.get_nodal_coords(0, i);
      mgpx[1] += target_val[i] * lele.get_nodal_coords(1, i);
      mgpx[2] += target_val[i] * lele.get_nodal_coords(2, i);
    }

    // normalize interpolated GP normal back to length 1.0 !!!
    double lengthn = sqrt(gpn[0] * gpn[0] + gpn[1] * gpn[1] + gpn[2] * gpn[2]);
    if (lengthn < 1.0e-12) FOUR_C_THROW("Divide by zero!");

    for (int i = 0; i < 3; ++i) gpn[i] /= lengthn;

    // build gap function at current GP
    for (int i = 0; i < n_dim(); ++i) gap += (mgpx[i] - sgpx[i]) * gpn[i];

    // **************************
    // Linearization
    // **************************
    int linsize = 0;
    for (int i = 0; i < nrow; ++i)
    {
      Node* cnode = dynamic_cast<Node*>(mynodes[i]);
      linsize += 10 * cnode->get_linsize();
    }

    // build directional derivative of source GP normal (non-unit)
    Core::Gen::Pairedvector<int, double> dmap_nxsl_gp(linsize);
    Core::Gen::Pairedvector<int, double> dmap_nysl_gp(linsize);
    Core::Gen::Pairedvector<int, double> dmap_nzsl_gp(linsize);

    for (int i = 0; i < nrow; ++i)
    {
      Node* cnode = dynamic_cast<Node*>(mynodes[i]);

      Core::Gen::Pairedvector<int, double>& dmap_nxsl_i = cnode->data().get_deriv_n()[0];
      Core::Gen::Pairedvector<int, double>& dmap_nysl_i = cnode->data().get_deriv_n()[1];
      Core::Gen::Pairedvector<int, double>& dmap_nzsl_i = cnode->data().get_deriv_n()[2];

      for (CI p = dmap_nxsl_i.begin(); p != dmap_nxsl_i.end(); ++p)
        dmap_nxsl_gp[p->first] += source_val[i] * (p->second);
      for (CI p = dmap_nysl_i.begin(); p != dmap_nysl_i.end(); ++p)
        dmap_nysl_gp[p->first] += source_val[i] * (p->second);
      for (CI p = dmap_nzsl_i.begin(); p != dmap_nzsl_i.end(); ++p)
        dmap_nzsl_gp[p->first] += source_val[i] * (p->second);

      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      {
        double valx = source_deriv(i, 0) * cnode->mo_data().n()[0];
        dmap_nxsl_gp[p->first] += valx * (p->second);
        double valy = source_deriv(i, 0) * cnode->mo_data().n()[1];
        dmap_nysl_gp[p->first] += valy * (p->second);
        double valz = source_deriv(i, 0) * cnode->mo_data().n()[2];
        dmap_nzsl_gp[p->first] += valz * (p->second);
      }

      for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
      {
        double valx = source_deriv(i, 1) * cnode->mo_data().n()[0];
        dmap_nxsl_gp[p->first] += valx * (p->second);
        double valy = source_deriv(i, 1) * cnode->mo_data().n()[1];
        dmap_nysl_gp[p->first] += valy * (p->second);
        double valz = source_deriv(i, 1) * cnode->mo_data().n()[2];
        dmap_nzsl_gp[p->first] += valz * (p->second);
      }
    }

    const double ll = lengthn * lengthn;
    const double linv = 1.0 / (lengthn);
    const double lllinv = 1.0 / (lengthn * lengthn * lengthn);
    const double sxsx = gpn[0] * gpn[0] * ll;
    const double sxsy = gpn[0] * gpn[1] * ll;
    const double sxsz = gpn[0] * gpn[2] * ll;
    const double sysy = gpn[1] * gpn[1] * ll;
    const double sysz = gpn[1] * gpn[2] * ll;
    const double szsz = gpn[2] * gpn[2] * ll;

    for (CI p = dmap_nxsl_gp.begin(); p != dmap_nxsl_gp.end(); ++p)
    {
      dnmap_unit[0][p->first] += linv * (p->second);
      dnmap_unit[0][p->first] -= lllinv * sxsx * (p->second);
      dnmap_unit[1][p->first] -= lllinv * sxsy * (p->second);
      dnmap_unit[2][p->first] -= lllinv * sxsz * (p->second);
    }

    for (CI p = dmap_nysl_gp.begin(); p != dmap_nysl_gp.end(); ++p)
    {
      dnmap_unit[1][p->first] += linv * (p->second);
      dnmap_unit[1][p->first] -= lllinv * sysy * (p->second);
      dnmap_unit[0][p->first] -= lllinv * sxsy * (p->second);
      dnmap_unit[2][p->first] -= lllinv * sysz * (p->second);
    }

    for (CI p = dmap_nzsl_gp.begin(); p != dmap_nzsl_gp.end(); ++p)
    {
      dnmap_unit[2][p->first] += linv * (p->second);
      dnmap_unit[2][p->first] -= lllinv * szsz * (p->second);
      dnmap_unit[0][p->first] -= lllinv * sxsz * (p->second);
      dnmap_unit[1][p->first] -= lllinv * sysz * (p->second);
    }

    // add everything to dgapgp
    for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      dgapgp[p->first] += (mgpx[0] - sgpx[0]) * (p->second);

    for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      dgapgp[p->first] += (mgpx[1] - sgpx[1]) * (p->second);

    for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      dgapgp[p->first] += (mgpx[2] - sgpx[2]) * (p->second);

    // lin source nodes
    for (int z = 0; z < nrow; ++z)
    {
      Node* cnode = dynamic_cast<Node*>(mynodes[z]);
      for (int k = 0; k < 3; ++k) dgapgp[cnode->dofs()[k]] -= source_val[z] * gpn[k];
    }

    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
    {
      double& dg = dgapgp[p->first];
      const double& ps = p->second;
      for (int z = 0; z < nrow; ++z)
      {
        Node* cnode = dynamic_cast<Node*>(mynodes[z]);
        for (int k = 0; k < 3; ++k) dg -= gpn[k] * source_deriv(z, 0) * cnode->xspatial()[k] * ps;
      }
    }
    for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
    {
      double& dg = dgapgp[p->first];
      const double& ps = p->second;
      for (int z = 0; z < nrow; ++z)
      {
        Node* cnode = dynamic_cast<Node*>(mynodes[z]);
        for (int k = 0; k < 3; ++k) dg -= gpn[k] * source_deriv(z, 1) * cnode->xspatial()[k] * ps;
      }
    }

    //        TARGET
    // lin target nodes
    for (int z = 0; z < ncolL; ++z)
    {
      Node* cnode = dynamic_cast<Node*>(target_nodes[z]);
      for (int k = 0; k < 3; ++k) dgapgp[cnode->dofs()[k]] += target_val[z] * gpn[k];
    }

    for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
    {
      double& dg = dgapgp[p->first];
      const double& ps = p->second;
      for (int z = 0; z < ncolL; ++z)
      {
        Node* cnode = dynamic_cast<Node*>(target_nodes[z]);
        for (int k = 0; k < 3; ++k) dg += gpn[k] * target_deriv(z, 0) * cnode->xspatial()[k] * ps;
      }
    }



    // weighted gap
    for (int j = 0; j < nrow; ++j)
    {
      CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(mynodes[j]);

      double prod = 0.0;
      // Petrov-Galerkin approach (dual LM for D/M but standard LM for gap)
      if (shape_fcn() == Mortar::shape_petrovgalerkin) prod = source_val[j] * gap * jac * wgt;
      // usual standard or dual LM approach
      else
        prod = lm_val[j] * gap * jac * wgt;

      // do not process source side boundary nodes
      // (their row entries would be zero anyway!)
      if (cnode->is_on_bound()) continue;

      // add current Gauss point's contribution to gseg
      cnode->addlts_gap_value(prod);
    }

    for (int iter = 0; iter < nrow; ++iter)
    {
      // map iterator
      using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

      Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[iter]);
      if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

      static double fac = 0.0;

      // get the corresponding map as a reference
      std::map<int, double>& dgmap =
          dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_glts();

      // switch if Petrov-Galerkin approach for LM is applied
      if (shape_fcn() == Mortar::shape_petrovgalerkin)
      {
        // (1) Lin(Phi) - does not exist here for Petrov-Galerkin approach

        // (2) Lin(N) - source GP coordinates
        for (int d = 0; d < n_dim() - 1; ++d)
        {
          fac = wgt * source_deriv(iter, d) * gap * jac;
          for (CI p = d_source_xi_gp[d].begin(); p != d_source_xi_gp[d].end(); ++p)
            dgmap[p->first] += fac * (p->second);
        }

        // (3) Lin(g) - gap function
        fac = wgt * source_val[iter] * jac;
        for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += fac * (p->second);

        // (4) Lin(dsxideta) - intcell GP Jacobian
        fac = wgt * source_val[iter] * gap;
        for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
          dgmap[p->first] += fac * (p->second);
      }

      // the usual standard or dual LM approach
      else
      {
        // (1) Lin(Phi) - dual shape functions
        // only line2 --> not required!

        // (2) Lin(Phi) - source GP coordinates
        for (int d = 0; d < n_dim() - 1; ++d)
        {
          fac = wgt * lm_deriv(iter, d) * gap * jac;
          for (CI p = d_source_xi_gp[d].begin(); p != d_source_xi_gp[d].end(); ++p)
            dgmap[p->first] += fac * (p->second);
        }

        // (3) Lin(g) - gap function
        fac = wgt * lm_val[iter] * jac;
        for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += fac * (p->second);

        // (4) Lin(dsxideta) - intcell GP Jacobian
        fac = wgt * lm_val[iter] * gap;
        for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
          dgmap[p->first] += fac * (p->second);
      }
    }

    // integrate D and M matrix
    // dual shape functions
    if (shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin)
    {
      for (int j = 0; j < nrow; ++j)
      {
        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(mynodes[j]);

        // integrate mseg
        for (int k = 0; k < ncolL; ++k)
        {
          CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_nodes[k]);

          // multiply the two shape functions
          double prod = lm_val[j] * target_val[k] * jac * wgt;

          if (abs(prod) > MORTARINTTOL) cnode->add_dlts_value(cnode->id(), prod);
          if (abs(prod) > MORTARINTTOL) cnode->add_source_node(cnode->id());  // only for friction!

          if (abs(prod) > MORTARINTTOL) cnode->add_mlts_value(target_node->id(), prod);
          if (abs(prod) > MORTARINTTOL)
            cnode->add_target_node(target_node->id());  // only for friction!
        }
      }
    }
    else
    {
      for (int j = 0; j < nrow; ++j)
      {
        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(mynodes[j]);

        // integrate dseg
        for (int k = 0; k < nrow; ++k)
        {
          CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(mynodes[k]);

          // multiply the two shape functions
          double prod = lm_val[j] * source_val[k] * jac * wgt;

          if (abs(prod) > MORTARINTTOL) cnode->add_dlts_value(target_node->id(), prod);
          if (abs(prod) > MORTARINTTOL)
            cnode->add_source_node(target_node->id());  // only for friction!
        }

        // integrate mseg
        for (int k = 0; k < ncolL; ++k)
        {
          CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_nodes[k]);

          // multiply the two shape functions
          double prod = lm_val[j] * target_val[k] * jac * wgt;

          if (abs(prod) > MORTARINTTOL) cnode->add_mlts_value(target_node->id(), prod);
          if (abs(prod) > MORTARINTTOL)
            cnode->add_target_node(target_node->id());  // only for friction!
        }
      }
    }

    if (shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin)
    {
      // integrate LinD
      for (int j = 0; j < nrow; ++j)
      {
        Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[j]);
        if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

        int s_gid = mymrtrnode->id();
        std::map<int, double>& ddmap_jk =
            dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_dlts()[s_gid];


        // integrate LinM
        for (int k = 0; k < ncolL; ++k)
        {
          // global target node ID
          int t_gid = target_elem.nodes()[k]->id();
          static double fac = 0.0;

          // get the correct map as a reference
          std::map<int, double>& dmmap_jk =
              dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_mlts()[t_gid];

          // (1) Lin(Phi) - dual shape functions
          // this vanishes here since there are no deformation-dependent dual functions

          // (2) Lin(NSource) - source GP coordinates
          fac = wgt * lm_deriv(j, 0) * target_val[k] * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          {
            dmmap_jk[p->first] += fac * (p->second);
            ddmap_jk[p->first] += fac * (p->second);
          }

          fac = wgt * lm_deriv(j, 1) * target_val[k] * jac;
          for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
          {
            dmmap_jk[p->first] += fac * (p->second);
            ddmap_jk[p->first] += fac * (p->second);
          }

          // (3) Lin(NTarget) - target GP coordinates
          fac = wgt * lm_val[j] * target_deriv(k, 0) * jac;
          for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
          {
            dmmap_jk[p->first] += fac * (p->second);
            ddmap_jk[p->first] += fac * (p->second);
          }

          //        fac = wgt*lm_val[j]*target_deriv(k, 1)*jac;
          //        for (CI p=d_target_xi_gp[1].begin(); p!=d_target_xi_gp[1].end(); ++p)
          //        {
          //          dmmap_jk[p->first] += fac*(p->second);
          //          ddmap_jk[p->first] += fac*(p->second);
          //        }

          // (4) Lin(dsxideta) - intcell GP Jacobian
          fac = wgt * lm_val[j] * target_val[k];
          for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
          {
            dmmap_jk[p->first] += fac * (p->second);
            ddmap_jk[p->first] += fac * (p->second);
          }
        }  // loop over target nodes
      }
    }
    else  // standard case
    {
      // integrate LinD
      for (int j = 0; j < nrow; ++j)
      {
        Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[j]);
        if (!mymrtrnode) FOUR_C_THROW("Null pointer!");


        // integrate LinD
        for (int k = 0; k < nrow; ++k)
        {
          // global target node ID
          int t_gid = source_elem.nodes()[k]->id();
          static double fac = 0.0;

          // get the correct map as a reference
          std::map<int, double>& ddmap_jk =
              dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_dlts()[t_gid];

          // (1) Lin(Phi) - dual shape functions
          // this vanishes here since there are no deformation-dependent dual functions

          // (2) Lin(NSource) - source GP coordinates
          fac = wgt * lm_deriv(j, 0) * source_val[k] * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          {
            ddmap_jk[p->first] += fac * (p->second);
          }

          fac = wgt * lm_deriv(j, 1) * source_val[k] * jac;
          for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
          {
            ddmap_jk[p->first] += fac * (p->second);
          }

          // (3) Lin(NTarget) - target GP coordinates
          fac = wgt * lm_val[j] * source_deriv(k, 0) * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          {
            ddmap_jk[p->first] += fac * (p->second);
          }

          fac = wgt * lm_val[j] * source_deriv(k, 1) * jac;
          for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
          {
            ddmap_jk[p->first] += fac * (p->second);
          }

          //        fac = wgt*lm_val[j]*target_deriv(k, 1)*jac;
          //        for (CI p=d_target_xi_gp[1].begin(); p!=d_target_xi_gp[1].end(); ++p)
          //        {
          //          dmmap_jk[p->first] += fac*(p->second);
          //          ddmap_jk[p->first] += fac*(p->second);
          //        }

          // (4) Lin(dsxideta) - intcell GP Jacobian
          fac = wgt * lm_val[j] * source_val[k];
          for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
          {
            ddmap_jk[p->first] += fac * (p->second);
          }
        }  // loop over source nodes

        // integrate LinM
        for (int k = 0; k < ncolL; ++k)
        {
          // global target node ID
          int t_gid = lele.nodes()[k]->id();
          static double fac = 0.0;

          // get the correct map as a reference
          std::map<int, double>& dmmap_jk =
              dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_mlts()[t_gid];

          // (1) Lin(Phi) - dual shape functions
          // this vanishes here since there are no deformation-dependent dual functions

          // (2) Lin(NSource) - source GP coordinates
          fac = wgt * lm_deriv(j, 0) * target_val[k] * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          {
            dmmap_jk[p->first] += fac * (p->second);
          }

          fac = wgt * lm_deriv(j, 1) * target_val[k] * jac;
          for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
          {
            dmmap_jk[p->first] += fac * (p->second);
          }

          // (3) Lin(NTarget) - target GP coordinates
          fac = wgt * lm_val[j] * target_deriv(k, 0) * jac;
          for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
          {
            dmmap_jk[p->first] += fac * (p->second);
          }

          //        fac = wgt*lm_val[j]*target_deriv(k, 1)*jac;
          //        for (CI p=d_target_xi_gp[1].begin(); p!=d_target_xi_gp[1].end(); ++p)
          //        {
          //          dmmap_jk[p->first] += fac*(p->second);
          //          ddmap_jk[p->first] += fac*(p->second);
          //        }

          // (4) Lin(dsxideta) - intcell GP Jacobian
          fac = wgt * lm_val[j] * target_val[k];
          for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
          {
            dmmap_jk[p->first] += fac * (p->second);
          }
        }  // loop over target nodes
      }
    }
  }  // end gp loop
}


/*----------------------------------------------------------------------*
 |  Integrate and linearize a 1D source / target cell (2D)    farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_deriv_cell_3d_aux_plane_lts(Mortar::Element& source_elem,
    Mortar::Element& lsele, Mortar::Element& target_elem, std::shared_ptr<Mortar::IntCell> cell,
    double* auxn, MPI_Comm comm)
{
  // explicitly defined shape function type needed
  if (shape_fcn() == Mortar::shape_undefined)
    FOUR_C_THROW(
        "integrate_deriv_cell_3d_aux_plane called without specific shape function defined!");

  // check for problem dimension
  if (n_dim() != 3) FOUR_C_THROW("3D integration method called for non-3D problem");

  // discretization type of target element
  Core::FE::CellType sdt = source_elem.shape();
  Core::FE::CellType mdt = target_elem.shape();

  // check input data
  //  if ((!source_elem.IsSource()) || (target_elem.IsSource()))
  //    FOUR_C_THROW("integrate_deriv_cell_3d_aux_plane called on a wrong type of Mortar::Element
  //    pair!");
  if (cell == nullptr)
    FOUR_C_THROW("integrate_deriv_cell_3d_aux_plane called without integration cell");

  // number of nodes (source, target)
  int nrowL = lsele.num_node();
  int nrowS = source_elem.num_node();
  int ncol = target_elem.num_node();
  int ndof = n_dim();

  // get source element nodes themselves for normal evaluation
  Core::Nodes::Node** mynodes = lsele.nodes();
  if (!mynodes) FOUR_C_THROW("Cannot access source nodes. Null pointer!");
  Core::Nodes::Node** target_nodes = target_elem.nodes();
  if (!target_nodes) FOUR_C_THROW("Cannot access target nodes. Null pointer!");

  // create empty vectors for shape fct. evaluation
  Core::LinAlg::SerialDenseVector source_val(nrowL);
  Core::LinAlg::SerialDenseMatrix source_deriv(nrowL, 1, true);
  Core::LinAlg::SerialDenseVector target_val(ncol);
  Core::LinAlg::SerialDenseMatrix target_deriv(ncol, 2, true);
  Core::LinAlg::SerialDenseVector lm_val(nrowL);
  Core::LinAlg::SerialDenseMatrix lm_deriv(nrowL, 1, true);

  // prepare directional derivative of dual shape functions
  // this is necessary for all source element types except tri3
  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> dualmap(
      (nrowL + ncol) * ndof, 0, Core::LinAlg::SerialDenseMatrix(nrowL, nrowL));
  if ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
      (lsele.shape() != Core::FE::CellType::line2 ||
          source_elem.mo_data().deriv_dual_shape() != nullptr))
    source_elem.deriv_shape_dual(dualmap);

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  int linsize = 0;
  for (int i = 0; i < nrowL; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(mynodes[i]);
    linsize += cnode->get_linsize();
  }
  linsize *= 100;  // TODO: remove this safety scaling

  // check if the cells are tri3
  // there's nothing wrong about other shapes, but as long as they are all
  // tri3 we can perform the jacobian calculation ( and its deriv) outside
  // the Gauss point loop
  if (cell->shape() != Core::FE::CellType::line2)
    FOUR_C_THROW("only line2 integration cells for LTS");

  const double jac = cell->jacobian();

  // directional derivative of cell Jacobian
  Core::Gen::Pairedvector<int, double> jacintcellmap((nrowL + ncol) * ndof + linsize);
  cell->deriv_jacobian(jacintcellmap);

  //**********************************************************************
  // loop over all Gauss points for integration
  //**********************************************************************
  for (int gp = 0; gp < n_gp(); ++gp)
  {
    // coordinates and weight
    double eta[2] = {coordinate(gp, 0), coordinate(gp, 1)};
    double wgt = weight(gp);

    // get global Gauss point coordinates
    double globgp[3] = {0.0, 0.0, 0.0};
    cell->local_to_global(eta, globgp, 0);

    // coordinate in parameter space on surface source ele
    double source_xi[2] = {0.0, 0.0};

    // coordinate in parameter space on surface target ele
    double target_xi[2] = {0.0, 0.0};

    // coordinate in parameter space on line (edge) source ele
    double lxi[2] = {0.0, 0.0};

    // project Gauss point onto source/target surface element
    double sprojalpha = 0.0;
    double mprojalpha = 0.0;
    bool success =
        Mortar::Projector::impl(source_elem)
            ->project_gauss_point_auxn_3d(globgp, auxn, source_elem, source_xi, sprojalpha);
    if (!success)
    {
      //      FOUR_C_THROW("projection not possible!");
      return;
    }

    success = Mortar::Projector::impl(target_elem)
                  ->project_gauss_point_auxn_3d(globgp, auxn, target_elem, target_xi, mprojalpha);
    if (!success)
    {
      //      FOUR_C_THROW("projection not possible!");
      return;
    }

    // check
    double checkpos[3] = {0.0, 0.0, 0.0};
    target_elem.local_to_global(target_xi, checkpos, 0);

    // check GP projection (SOURCE)
    const double sminedge = source_elem.min_edge_size();
    const double mminedge = target_elem.min_edge_size();
    const double tol = MORTARCLIPTOL * std::min(sminedge, mminedge);
    if (sdt == Core::FE::CellType::quad4 || sdt == Core::FE::CellType::quad8 ||
        sdt == Core::FE::CellType::quad9)
    {
      if (source_xi[0] < -1.0 - tol || source_xi[1] < -1.0 - tol || source_xi[0] > 1.0 + tol ||
          source_xi[1] > 1.0 + tol)
      {
        std::cout << "\n***Warning: LTS: Gauss point projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Source GP projection: " << source_xi[0] << " " << source_xi[1] << std::endl;
      }
    }
    else
    {
      if (source_xi[0] < -tol || source_xi[1] < -tol || source_xi[0] > 1.0 + tol ||
          source_xi[1] > 1.0 + tol || source_xi[0] + source_xi[1] > 1.0 + 2 * tol)
      {
        std::cout << "\n***Warning: LTS: Gauss point projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Source GP projection: " << source_xi[0] << " " << source_xi[1] << std::endl;
      }
    }

    // check GP projection (TARGET)
    if (mdt == Core::FE::CellType::quad4 || mdt == Core::FE::CellType::quad8 ||
        mdt == Core::FE::CellType::quad9)
    {
      if (target_xi[0] < -1.0 - tol || target_xi[1] < -1.0 - tol || target_xi[0] > 1.0 + tol ||
          target_xi[1] > 1.0 + tol)
      {
        std::cout << "\n***Warning: LTS: Gauss point projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Target GP projection: " << target_xi[0] << " " << target_xi[1] << std::endl;
        std::cout << "globgp=   " << globgp[0] << "  " << globgp[1] << "  " << globgp[2]
                  << std::endl;
        std::cout << "checkpos= " << checkpos[0] << "  " << checkpos[1] << "  " << checkpos[2]
                  << std::endl;
      }
    }
    else
    {
      if (target_xi[0] < -tol || target_xi[1] < -tol || target_xi[0] > 1.0 + tol ||
          target_xi[1] > 1.0 + tol || target_xi[0] + target_xi[1] > 1.0 + 2 * tol)
      {
        std::cout << "\n***Warning: LTS: Gauss point projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Target GP projection: " << target_xi[0] << " " << target_xi[1] << std::endl;
        std::cout << "globgp=   " << globgp[0] << "  " << globgp[1] << "  " << globgp[2]
                  << std::endl;
        std::cout << "checkpos= " << checkpos[0] << "  " << checkpos[1] << "  " << checkpos[2]
                  << std::endl;
      }
    }

    // convert source_xi to lsxi!!!
    if (lsele.id() == 0)
    {
      lxi[0] = source_xi[0];
    }
    else if (lsele.id() == 1)
    {
      lxi[0] = source_xi[1];
    }
    else if (lsele.id() == 2)
    {
      lxi[0] = -source_xi[0];
    }
    else if (lsele.id() == 3)
    {
      lxi[0] = -source_xi[1];
    }
    else
      FOUR_C_THROW("invalid local id of line ele");

    // evaluate Lagrange multiplier shape functions (on source element)
    lsele.evaluate_shape_lag_mult(shape_fcn(), lxi, lm_val, lm_deriv, nrowL);

    // evaluate trace space shape functions (on both elements)
    lsele.evaluate_shape(lxi, source_val, source_deriv, nrowL);
    target_elem.evaluate_shape(target_xi, target_val, target_deriv, ncol);

    // evaluate linearizations *******************************************

    // evaluate global GP coordinate derivative
    static Core::LinAlg::Matrix<3, 1> svalcell;
    static Core::LinAlg::Matrix<3, 2> sderivcell;
    cell->evaluate_shape(eta, svalcell, sderivcell);

    Core::Gen::Pairedvector<int, Core::LinAlg::Matrix<3, 1>> lingp((nrowS + ncol) * ndof + linsize);

    for (int v = 0; v < 2; ++v)
      for (int d = 0; d < 3; ++d)
        for (CI p = (cell->get_deriv_vertex(v))[d].begin();
            p != (cell->get_deriv_vertex(v))[d].end(); ++p)
          lingp[p->first](d) += svalcell(v) * (p->second);

    // evaluate the GP source coordinate derivatives
    std::vector<Core::Gen::Pairedvector<int, double>> d_source_xi_gp(
        2, (nrowS + ncol) * ndof + linsize);
    deriv_xi_gp_3d_aux_plane(source_elem, source_xi, cell->auxn(), d_source_xi_gp, sprojalpha,
        cell->get_deriv_auxn(), lingp);

    // evaluate the GP target coordinate derivatives
    std::vector<Core::Gen::Pairedvector<int, double>> d_target_xi_gp(
        2, (nrowS + ncol) * ndof + linsize);
    deriv_xi_gp_3d_aux_plane(target_elem, target_xi, cell->auxn(), d_target_xi_gp, mprojalpha,
        cell->get_deriv_auxn(), lingp);

    // convert deriv source_xi
    std::vector<Core::Gen::Pairedvector<int, double>> dlxigp(1, (nrowS + ncol) * ndof + linsize);

    if (lsele.id() == 0)
    {
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        dlxigp[0][p->first] += (p->second);
    }
    else if (lsele.id() == 1)
    {
      for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
        dlxigp[0][p->first] += (p->second);
    }
    else if (lsele.id() == 2)
    {
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        dlxigp[0][p->first] -= (p->second);
    }
    else if (lsele.id() == 3)
    {
      for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
        dlxigp[0][p->first] -= (p->second);
    }
    else
      FOUR_C_THROW("invalid local id of line ele");

    //**********************************************************************
    // geometric quantities
    //**********************************************************************
    std::array<double, 3> gpn = {0.0, 0.0, 0.0};
    Core::Gen::Pairedvector<int, double> dgapgp(
        (ncol * ndof) + 10 * linsize);  // gap lin. without lm and jac.
    double gap = 0.0;
    std::vector<Core::Gen::Pairedvector<int, double>> dnmap_unit(
        3, 10 * linsize);  // deriv of x,y and z comp. of gpn (unit)

    //**********************************************************************
    // evaluate at GP and lin char. quantities
    //**********************************************************************
    std::array<double, 3> sgpx = {0.0, 0.0, 0.0};
    std::array<double, 3> mgpx = {0.0, 0.0, 0.0};

    for (int i = 0; i < nrowL; ++i)
    {
      Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[i]);
      gpn[0] += source_val[i] * mymrtrnode->mo_data().n()[0];
      gpn[1] += source_val[i] * mymrtrnode->mo_data().n()[1];
      gpn[2] += source_val[i] * mymrtrnode->mo_data().n()[2];

      sgpx[0] += source_val[i] * lsele.get_nodal_coords(0, i);
      sgpx[1] += source_val[i] * lsele.get_nodal_coords(1, i);
      sgpx[2] += source_val[i] * lsele.get_nodal_coords(2, i);
    }

    // build interpolation of target GP coordinates
    for (int i = 0; i < ncol; ++i)
    {
      mgpx[0] += target_val[i] * target_elem.get_nodal_coords(0, i);
      mgpx[1] += target_val[i] * target_elem.get_nodal_coords(1, i);
      mgpx[2] += target_val[i] * target_elem.get_nodal_coords(2, i);
    }

    // normalize interpolated GP normal back to length 1.0 !!!
    double lengthn = sqrt(gpn[0] * gpn[0] + gpn[1] * gpn[1] + gpn[2] * gpn[2]);
    if (lengthn < 1.0e-12) FOUR_C_THROW("Divide by zero!");

    for (int i = 0; i < 3; ++i) gpn[i] /= lengthn;

    // build gap function at current GP
    for (int i = 0; i < n_dim(); ++i) gap += (mgpx[i] - sgpx[i]) * gpn[i];

    // **************************
    // Linearization
    // **************************
    int linsize = 0;
    for (int i = 0; i < nrowL; ++i)
    {
      Node* cnode = dynamic_cast<Node*>(mynodes[i]);
      linsize += 10 * cnode->get_linsize();
    }

    // build directional derivative of source GP normal (non-unit)
    Core::Gen::Pairedvector<int, double> dmap_nxsl_gp(linsize);
    Core::Gen::Pairedvector<int, double> dmap_nysl_gp(linsize);
    Core::Gen::Pairedvector<int, double> dmap_nzsl_gp(linsize);

    for (int i = 0; i < nrowL; ++i)
    {
      Node* cnode = dynamic_cast<Node*>(mynodes[i]);

      Core::Gen::Pairedvector<int, double>& dmap_nxsl_i = cnode->data().get_deriv_n()[0];
      Core::Gen::Pairedvector<int, double>& dmap_nysl_i = cnode->data().get_deriv_n()[1];
      Core::Gen::Pairedvector<int, double>& dmap_nzsl_i = cnode->data().get_deriv_n()[2];

      for (CI p = dmap_nxsl_i.begin(); p != dmap_nxsl_i.end(); ++p)
        dmap_nxsl_gp[p->first] += source_val[i] * (p->second);
      for (CI p = dmap_nysl_i.begin(); p != dmap_nysl_i.end(); ++p)
        dmap_nysl_gp[p->first] += source_val[i] * (p->second);
      for (CI p = dmap_nzsl_i.begin(); p != dmap_nzsl_i.end(); ++p)
        dmap_nzsl_gp[p->first] += source_val[i] * (p->second);

      for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
      {
        double valx = source_deriv(i, 0) * cnode->mo_data().n()[0];
        dmap_nxsl_gp[p->first] += valx * (p->second);
        double valy = source_deriv(i, 0) * cnode->mo_data().n()[1];
        dmap_nysl_gp[p->first] += valy * (p->second);
        double valz = source_deriv(i, 0) * cnode->mo_data().n()[2];
        dmap_nzsl_gp[p->first] += valz * (p->second);
      }
    }

    const double ll = lengthn * lengthn;
    const double linv = 1.0 / (lengthn);
    const double lllinv = 1.0 / (lengthn * lengthn * lengthn);
    const double sxsx = gpn[0] * gpn[0] * ll;
    const double sxsy = gpn[0] * gpn[1] * ll;
    const double sxsz = gpn[0] * gpn[2] * ll;
    const double sysy = gpn[1] * gpn[1] * ll;
    const double sysz = gpn[1] * gpn[2] * ll;
    const double szsz = gpn[2] * gpn[2] * ll;

    for (CI p = dmap_nxsl_gp.begin(); p != dmap_nxsl_gp.end(); ++p)
    {
      dnmap_unit[0][p->first] += linv * (p->second);
      dnmap_unit[0][p->first] -= lllinv * sxsx * (p->second);
      dnmap_unit[1][p->first] -= lllinv * sxsy * (p->second);
      dnmap_unit[2][p->first] -= lllinv * sxsz * (p->second);
    }

    for (CI p = dmap_nysl_gp.begin(); p != dmap_nysl_gp.end(); ++p)
    {
      dnmap_unit[1][p->first] += linv * (p->second);
      dnmap_unit[1][p->first] -= lllinv * sysy * (p->second);
      dnmap_unit[0][p->first] -= lllinv * sxsy * (p->second);
      dnmap_unit[2][p->first] -= lllinv * sysz * (p->second);
    }

    for (CI p = dmap_nzsl_gp.begin(); p != dmap_nzsl_gp.end(); ++p)
    {
      dnmap_unit[2][p->first] += linv * (p->second);
      dnmap_unit[2][p->first] -= lllinv * szsz * (p->second);
      dnmap_unit[0][p->first] -= lllinv * sxsz * (p->second);
      dnmap_unit[1][p->first] -= lllinv * sysz * (p->second);
    }

    // add everything to dgapgp
    for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      dgapgp[p->first] += (mgpx[0] - sgpx[0]) * (p->second);

    for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      dgapgp[p->first] += (mgpx[1] - sgpx[1]) * (p->second);

    for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      dgapgp[p->first] += (mgpx[2] - sgpx[2]) * (p->second);

    // lin source nodes
    for (int z = 0; z < nrowL; ++z)
    {
      Node* cnode = dynamic_cast<Node*>(mynodes[z]);
      for (int k = 0; k < 3; ++k) dgapgp[cnode->dofs()[k]] -= source_val[z] * gpn[k];
    }

    for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
    {
      double& dg = dgapgp[p->first];
      const double& ps = p->second;
      for (int z = 0; z < nrowL; ++z)
      {
        Node* cnode = dynamic_cast<Node*>(mynodes[z]);
        for (int k = 0; k < 3; ++k) dg -= gpn[k] * source_deriv(z, 0) * cnode->xspatial()[k] * ps;
      }
    }
    //        TARGET
    // lin target nodes
    for (int z = 0; z < ncol; ++z)
    {
      Node* cnode = dynamic_cast<Node*>(target_nodes[z]);
      for (int k = 0; k < 3; ++k) dgapgp[cnode->dofs()[k]] += target_val[z] * gpn[k];
    }

    for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
    {
      double& dg = dgapgp[p->first];
      const double& ps = p->second;
      for (int z = 0; z < ncol; ++z)
      {
        Node* cnode = dynamic_cast<Node*>(target_nodes[z]);
        for (int k = 0; k < 3; ++k) dg += gpn[k] * target_deriv(z, 0) * cnode->xspatial()[k] * ps;
      }
    }

    for (CI p = d_target_xi_gp[1].begin(); p != d_target_xi_gp[1].end(); ++p)
    {
      double& dg = dgapgp[p->first];
      const double& ps = p->second;
      for (int z = 0; z < ncol; ++z)
      {
        Node* cnode = dynamic_cast<Node*>(target_nodes[z]);
        for (int k = 0; k < 3; ++k) dg += gpn[k] * target_deriv(z, 1) * cnode->xspatial()[k] * ps;
      }
    }

    // weighted gap
    for (int j = 0; j < nrowL; ++j)
    {
      CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(mynodes[j]);

      double prod = 0.0;
      // Petrov-Galerkin approach (dual LM for D/M but standard LM for gap)
      if (shape_fcn() == Mortar::shape_petrovgalerkin) prod = source_val[j] * gap * jac * wgt;
      // usual standard or dual LM approach
      else
        prod = lm_val[j] * gap * jac * wgt;

      // do not process source side boundary nodes
      // (their row entries would be zero anyway!)
      if (cnode->is_on_corner()) continue;

      // add current Gauss point's contribution to gseg
      cnode->addlts_gap_value(prod);
    }

    for (int iter = 0; iter < nrowL; ++iter)
    {
      // map iterator
      using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

      Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[iter]);
      if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

      if (mymrtrnode->is_on_corner()) continue;

      static double fac = 0.0;

      // get the corresponding map as a reference
      std::map<int, double>& dgmap =
          dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_glts();

      // switch if Petrov-Galerkin approach for LM is applied
      if (shape_fcn() == Mortar::shape_petrovgalerkin)
      {
        // (1) Lin(Phi) - does not exist here for Petrov-Galerkin approach

        // (2) Lin(N) - source GP coordinates
        for (int d = 0; d < n_dim() - 2; ++d)
        {
          fac = wgt * source_deriv(iter, d) * gap * jac;
          for (CI p = dlxigp[d].begin(); p != dlxigp[d].end(); ++p)
            dgmap[p->first] += fac * (p->second);
        }

        // (3) Lin(g) - gap function
        fac = wgt * source_val[iter] * jac;
        for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += fac * (p->second);

        // (4) Lin(dsxideta) - intcell GP Jacobian
        fac = wgt * source_val[iter] * gap;
        for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
          dgmap[p->first] += fac * (p->second);
      }

      // the usual standard or dual LM approach
      else
      {
        // (1) Lin(Phi) - dual shape functions
        // only line2 --> not required!

        // (2) Lin(Phi) - source GP coordinates
        for (int d = 0; d < n_dim() - 2; ++d)
        {
          fac = wgt * lm_deriv(iter, d) * gap * jac;
          for (CI p = dlxigp[d].begin(); p != dlxigp[d].end(); ++p)
            dgmap[p->first] += fac * (p->second);
        }

        // (3) Lin(g) - gap function
        fac = wgt * lm_val[iter] * jac;
        for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += fac * (p->second);

        // (4) Lin(dsxideta) - intcell GP Jacobian
        fac = wgt * lm_val[iter] * gap;
        for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
          dgmap[p->first] += fac * (p->second);
      }
    }
    //--------------------------------------------------------
    // integrate D and M matrix
    // dual shape functions
    if (shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin)
    {
      bool bound = false;
      for (int j = 0; j < nrowL; ++j)
      {
        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(mynodes[j]);
        if (cnode->is_on_corner()) bound = true;
      }

      for (int j = 0; j < nrowL; ++j)
      {
        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(mynodes[j]);
        if (cnode->is_on_corner()) continue;

        // integrate mseg
        for (int k = 0; k < ncol; ++k)
        {
          CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_nodes[k]);

          // multiply the two shape functions
          double prod = lm_val[j] * target_val[k] * jac * wgt;

          if (!bound and abs(prod) > MORTARINTTOL)
          {
            if (abs(prod) > MORTARINTTOL) cnode->add_dlts_value(cnode->id(), prod);
            if (abs(prod) > MORTARINTTOL)
              cnode->add_source_node(cnode->id());  // only for friction!
          }

          if (abs(prod) > MORTARINTTOL) cnode->add_mlts_value(target_node->id(), prod);
          if (abs(prod) > MORTARINTTOL)
            cnode->add_target_node(target_node->id());  // only for friction!
        }

        // integrate dseg (boundary modification)
        if (bound)
        {
          //        bool j_boundnode = cnode->IsOnBoundorCE();

          for (int k = 0; k < nrowL; ++k)
          {
            CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(mynodes[k]);
            //          bool k_boundnode = target_node->IsOnBoundorCE();

            // do not assemble off-diagonal terms if j,k are both non-boundary nodes
            //          if (!j_boundnode && !k_boundnode && (j!=k))
            //            continue;

            // multiply the two shape functions
            double prod = lm_val[j] * source_val[k] * jac * wgt;

            // isolate the dseg entries to be filled
            // (both the main diagonal and every other secondary diagonal)
            // and add current Gauss point's contribution to dseg
            if (target_node->is_on_corner())
            {
              if (abs(prod) > MORTARINTTOL) cnode->add_dlts_value(target_node->id(), prod);
              if (abs(prod) > MORTARINTTOL)
                cnode->add_source_node(target_node->id());  // only for friction!
            }
            else
            {
              if (abs(prod) > MORTARINTTOL) cnode->add_dlts_value(cnode->id(), prod);
              if (abs(prod) > MORTARINTTOL)
                cnode->add_source_node(cnode->id());  // only for friction!
            }
          }
        }
      }

      // OLD DM DUAL IMPLEMENTATION
      //      for (int j=0;j<nrowL;++j)
      //      {
      //        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(mynodes[j]);
      //        if (cnode->IsOnCorner())
      //          continue;
      //
      //        // integrate mseg
      //        for (int k=0; k<ncol; ++k)
      //        {
      //          CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_nodes[k]);
      //
      //          // multiply the two shape functions
      //          double prod = lm_val[j]*target_val[k]*jac*wgt;
      //
      //          if(abs(prod)>MORTARINTTOL) cnode->AddDltsValue(cnode->Id(),prod);
      //          if(abs(prod)>MORTARINTTOL) cnode->AddSNode(cnode->Id()); // only for friction!
      //
      //          if(abs(prod)>MORTARINTTOL) cnode->AddMltsValue(target_node->Id(),prod);
      //          if(abs(prod)>MORTARINTTOL) cnode->AddMNode(target_node->Id());  // only for
      //          friction!
      //        }
      //      }
    }
    // STANDARD SHAPE FUNCTIONS
    else  // std
    {
      for (int j = 0; j < nrowL; ++j)
      {
        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(mynodes[j]);

        if (cnode->is_on_corner()) continue;

        // integrate mseg
        for (int k = 0; k < ncol; ++k)
        {
          CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_nodes[k]);

          // multiply the two shape functions
          double prod = lm_val[j] * target_val[k] * jac * wgt;

          if (abs(prod) > MORTARINTTOL) cnode->add_mlts_value(target_node->id(), prod);
          if (abs(prod) > MORTARINTTOL)
            cnode->add_target_node(target_node->id());  // only for friction!
        }

        // integrate dseg
        for (int k = 0; k < nrowL; ++k)
        {
          CONTACT::Node* source_node = dynamic_cast<CONTACT::Node*>(mynodes[k]);

          if (source_node->is_on_corner())
          {
            // multiply the two shape functions
            double prod = lm_val[j] * source_val[k] * jac * wgt;

            if (abs(prod) > MORTARINTTOL) cnode->add_dlts_value(source_node->id(), prod);
            if (abs(prod) > MORTARINTTOL)
              cnode->add_source_node(source_node->id());  // only for friction!
          }
          else
          {
            // multiply the two shape functions
            double prod = lm_val[j] * source_val[k] * jac * wgt;

            if (abs(prod) > MORTARINTTOL) cnode->add_dlts_value(source_node->id(), prod);
            if (abs(prod) > MORTARINTTOL)
              cnode->add_source_node(source_node->id());  // only for friction!
          }
        }
      }
    }

    // dual shape functions
    if (shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin)
    {
      bool bound = false;
      for (int j = 0; j < nrowL; ++j)
      {
        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(mynodes[j]);
        if (cnode->is_on_corner()) bound = true;
      }

      if (bound)
      {
        // integrate LinD
        for (int j = 0; j < nrowL; ++j)
        {
          Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[j]);
          if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

          // node j is boundary node
          if (mymrtrnode->is_on_corner()) continue;

          // integrate LinM
          for (int k = 0; k < ncol; ++k)
          {
            // global target node ID
            int t_gid = target_elem.nodes()[k]->id();
            static double fac = 0.0;

            // get the correct map as a reference
            std::map<int, double>& dmmap_jk =
                dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_mlts()[t_gid];

            // (1) Lin(Phi) - dual shape functions
            //          if (dualmap.size()>0)
            //          {
            //            for (int m=0;m<nrowL;++m)
            //            {
            //              fac = wgt*source_val[m]*target_val[k]*jac;
            //              for
            //              (Core::Gen::Pairedvector<int,Core::LinAlg::SerialDenseMatrix>::const_iterator
            //              p=dualmap.begin();
            //                  p!=dualmap.end();++p)
            //                dmmap_jk[p->first] += fac*(p->second)(j,m);
            //            }
            //          }

            // (2) Lin(NSource) - source GP coordinates
            fac = wgt * lm_deriv(j, 0) * target_val[k] * jac;
            for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
              dmmap_jk[p->first] += fac * (p->second);

            //            fac = wgt*lm_deriv(j, 1)*target_val[k]*jac;
            //            for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
            //              dmmap_jk[p->first] += fac*(p->second);

            // (3) Lin(NTarget) - target GP coordinates
            fac = wgt * lm_val[j] * target_deriv(k, 0) * jac;
            for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
              dmmap_jk[p->first] += fac * (p->second);

            fac = wgt * lm_val[j] * target_deriv(k, 1) * jac;
            for (CI p = d_target_xi_gp[1].begin(); p != d_target_xi_gp[1].end(); ++p)
              dmmap_jk[p->first] += fac * (p->second);

            // (4) Lin(dsxideta) - intcell GP Jacobian
            fac = wgt * lm_val[j] * target_val[k];
            for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
              dmmap_jk[p->first] += fac * (p->second);
          }  // loop over target nodes

          // loop over source nodes
          for (int k = 0; k < nrowL; ++k)
          {
            Mortar::Node* mymrtrnode2 = dynamic_cast<Mortar::Node*>(mynodes[k]);
            if (!mymrtrnode2) FOUR_C_THROW("Null pointer!");

            // global target node ID
            int s_gid = mymrtrnode2->id();
            static double fac = 0.0;

            // node k is boundary node
            if (mymrtrnode2->is_on_corner())
            {
              // move entry to derivM (with minus sign)
              // get the correct map as a reference
              std::map<int, double>& dmmap_jk =
                  dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_dlts()[s_gid];

              // (1) Lin(Phi) - dual shape functions
              //            if (dualmap.size()>0)
              //            {
              //              for (int m=0;m<nrow;++m)
              //              {
              //                fac = wgt*source_val[m]*source_val[k]*jac;
              //                for
              //                (Core::Gen::Pairedvector<int,Core::LinAlg::SerialDenseMatrix>::const_iterator
              //                p=dualmap.begin();
              //                    p!=dualmap.end();++p)
              //                  dmmap_jk[p->first] -= fac*(p->second)(j,m);
              //              }
              //            }

              // (2) Lin(NSource) - source GP coordinates
              fac = wgt * lm_deriv(j, 0) * source_val[k] * jac;
              for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
                dmmap_jk[p->first] += fac * (p->second);

              //              fac = wgt*lm_deriv(j, 1)*source_val[k]*jac;
              //              for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
              //                dmmap_jk[p->first] -= fac*(p->second);

              // (3) Lin(NSource) - source GP coordinates
              fac = wgt * lm_val[j] * source_deriv(k, 0) * jac;
              for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
                dmmap_jk[p->first] += fac * (p->second);

              //              fac = wgt*lm_val[j]*source_deriv(k, 1)*jac;
              //              for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
              //                dmmap_jk[p->first] -= fac*(p->second);

              // (4) Lin(dsxideta) - intcell GP Jacobian
              fac = wgt * lm_val[j] * source_val[k];
              for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
                dmmap_jk[p->first] += fac * (p->second);
            }

            // node k is NO boundary node
            else
            {
              // get the correct map as a reference
              std::map<int, double>& ddmap_jk = dynamic_cast<CONTACT::Node*>(mymrtrnode)
                                                    ->data()
                                                    .get_deriv_dlts()[mymrtrnode->id()];

              // (1) Lin(Phi) - dual shape functions
              //            if (dualmap.size()>0)
              //            {
              //              for (int m=0;m<nrow;++m)
              //              {
              //                fac = wgt*source_val[m]*source_val[k]*jac;
              //                for
              //                (Core::Gen::Pairedvector<int,Core::LinAlg::SerialDenseMatrix>::const_iterator
              //                p=dualmap.begin();
              //                    p!=dualmap.end();++p)
              //                  ddmap_jk[p->first] += fac*(p->second)(j,m);
              //              }
              //            }

              // (2) Lin(NSource) - source GP coordinates
              fac = wgt * lm_deriv(j, 0) * source_val[k] * jac;
              for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
                ddmap_jk[p->first] += fac * (p->second);

              //              fac = wgt*lm_deriv(j, 1)*source_val[k]*jac;
              //              for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
              //                ddmap_jk[p->first] += fac*(p->second);

              // (3) Lin(NSource) - source GP coordinates
              fac = wgt * lm_val[j] * source_deriv(k, 0) * jac;
              for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
                ddmap_jk[p->first] += fac * (p->second);

              //              fac = wgt*lm_val[j]*source_deriv(k, 1)*jac;
              //              for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
              //                ddmap_jk[p->first] += fac*(p->second);

              // (4) Lin(dsxideta) - intcell GP Jacobian
              fac = wgt * lm_val[j] * source_val[k];
              for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
                ddmap_jk[p->first] += fac * (p->second);
            }
          }  // loop over source nodes
        }
      }
      else
      {
        // OLD DM DUAL IMPLEMENTATION
        // integrate LinD
        for (int j = 0; j < nrowL; ++j)
        {
          Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[j]);
          if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

          int s_gid = mymrtrnode->id();
          std::map<int, double>& ddmap_jk =
              dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_dlts()[s_gid];

          // integrate LinM
          for (int k = 0; k < ncol; ++k)
          {
            // global target node ID
            int t_gid = target_elem.nodes()[k]->id();
            static double fac = 0.0;

            // get the correct map as a reference
            std::map<int, double>& dmmap_jk =
                dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_mlts()[t_gid];

            // (1) Lin(Phi) - dual shape functions
            // this vanishes here since there are no deformation-dependent dual functions

            // (2) Lin(NSource) - source GP coordinates
            fac = wgt * lm_deriv(j, 0) * target_val[k] * jac;
            for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
            {
              dmmap_jk[p->first] += fac * (p->second);
              ddmap_jk[p->first] += fac * (p->second);
            }

            // (3) Lin(NTarget) - target GP coordinates
            fac = wgt * lm_val[j] * target_deriv(k, 0) * jac;
            for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
            {
              dmmap_jk[p->first] += fac * (p->second);
              ddmap_jk[p->first] += fac * (p->second);
            }

            fac = wgt * lm_val[j] * target_deriv(k, 1) * jac;
            for (CI p = d_target_xi_gp[1].begin(); p != d_target_xi_gp[1].end(); ++p)
            {
              dmmap_jk[p->first] += fac * (p->second);
              ddmap_jk[p->first] += fac * (p->second);
            }

            // (4) Lin(dsxideta) - intcell GP Jacobian
            fac = wgt * lm_val[j] * target_val[k];
            for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
            {
              dmmap_jk[p->first] += fac * (p->second);
              ddmap_jk[p->first] += fac * (p->second);
            }
          }  // loop over target nodes
        }
      }
    }
    // STANDARD SHAPE FUNCTIONS
    else  // std
    {
      // integrate LinD
      for (int j = 0; j < nrowL; ++j)
      {
        Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[j]);
        if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

        if (mymrtrnode->is_on_corner()) continue;

        // integrate LinM
        for (int k = 0; k < ncol; ++k)
        {
          // global target node ID
          int t_gid = target_elem.nodes()[k]->id();
          static double fac = 0.0;

          // get the correct map as a reference
          std::map<int, double>& dmmap_jk =
              dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_mlts()[t_gid];

          // (1) Lin(Phi) - dual shape functions
          // this vanishes here since there are no deformation-dependent dual functions

          // (2) Lin(NSource) - source GP coordinates
          fac = wgt * lm_deriv(j, 0) * target_val[k] * jac;
          for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
          {
            dmmap_jk[p->first] += fac * (p->second);
          }

          // (3) Lin(NTarget) - target GP coordinates
          fac = wgt * lm_val[j] * target_deriv(k, 0) * jac;
          for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
          {
            dmmap_jk[p->first] += fac * (p->second);
          }

          fac = wgt * lm_val[j] * target_deriv(k, 1) * jac;
          for (CI p = d_target_xi_gp[1].begin(); p != d_target_xi_gp[1].end(); ++p)
          {
            dmmap_jk[p->first] += fac * (p->second);
          }

          // (4) Lin(dsxideta) - intcell GP Jacobian
          fac = wgt * lm_val[j] * target_val[k];
          for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
          {
            dmmap_jk[p->first] += fac * (p->second);
          }
        }  // loop over target nodes

        // integrate LinD
        for (int k = 0; k < nrowL; ++k)
        {
          // global target node ID
          int t_gid = mynodes[k]->id();
          static double fac = 0.0;

          if (dynamic_cast<Node*>(mynodes[k])->is_on_corner())
          {
            // get the correct map as a reference
            std::map<int, double>& dmmap_jk =
                dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_dlts()[t_gid];

            // (1) Lin(Phi) - dual shape functions
            // this vanishes here since there are no deformation-dependent dual functions

            // (2) Lin(NSource) - source GP coordinates
            fac = wgt * lm_deriv(j, 0) * source_val[k] * jac;
            for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
            {
              dmmap_jk[p->first] += fac * (p->second);
            }

            // (3) Lin(NTarget) - target GP coordinates
            fac = wgt * lm_val[j] * source_deriv(k, 0) * jac;
            for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
            {
              dmmap_jk[p->first] += fac * (p->second);
            }

            // (4) Lin(dsxideta) - intcell GP Jacobian
            fac = wgt * lm_val[j] * source_val[k];
            for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
            {
              dmmap_jk[p->first] += fac * (p->second);
            }
          }
          else
          {
            // get the correct map as a reference
            std::map<int, double>& dmmap_jk =
                dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_dlts()[t_gid];

            // (1) Lin(Phi) - dual shape functions
            // this vanishes here since there are no deformation-dependent dual functions

            // (2) Lin(NSource) - source GP coordinates
            fac = wgt * lm_deriv(j, 0) * source_val[k] * jac;
            for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
            {
              dmmap_jk[p->first] += fac * (p->second);
            }

            // (3) Lin(NTarget) - target GP coordinates
            fac = wgt * lm_val[j] * source_deriv(k, 0) * jac;
            for (CI p = dlxigp[0].begin(); p != dlxigp[0].end(); ++p)
            {
              dmmap_jk[p->first] += fac * (p->second);
            }

            // (4) Lin(dsxideta) - intcell GP Jacobian
            fac = wgt * lm_val[j] * source_val[k];
            for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
            {
              dmmap_jk[p->first] += fac * (p->second);
            }
          }
        }  // loop over target nodes
      }
    }
  }  // end gp loop
}


/*----------------------------------------------------------------------*
 |  Integrate and linearize a 2D source / target cell (3D)     popp 03/09|
 |  This method integrates the cell M matrix and weighted gap g~        |
 |  and stores it in mseg and gseg respectively. Moreover, derivatives  |
 |  LinM and Ling are built and stored directly into the adjacent nodes.|
 |  (Thus this method combines EVERYTHING before done separately in     |
 |  IntegrateM3D, IntegrateG3D, DerivM3D and DerivG3D!)                 |
 |  This is the QUADRATIC auxiliary plane coupling version!!!           |
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_deriv_cell_3d_aux_plane_quad(Mortar::Element& source_elem,
    Mortar::Element& target_elem, Mortar::IntElement& sintele, Mortar::IntElement& mintele,
    std::shared_ptr<Mortar::IntCell> cell, double* auxn)
{
  // get LMtype
  Mortar::LagMultQuad lmtype = lag_mult_quad();

  // explicitly defined shape function type needed
  if (shape_fcn() == Mortar::shape_undefined)
    FOUR_C_THROW("Function is called without specific shape function defined!");

  // Petrov-Galerkin approach for LM not yet implemented
  if (shape_fcn() == Mortar::shape_petrovgalerkin &&
      source_elem.shape() != Core::FE::CellType::nurbs9)
    FOUR_C_THROW("Petrov-Galerkin approach not yet implemented for 3-D quadratic FE interpolation");

  // check for problem dimension
  if (n_dim() != 3) FOUR_C_THROW("3D integration method called for non-3D problem");

  // discretization type of source and target IntElement
  Core::FE::CellType sdt = sintele.shape();
  Core::FE::CellType mdt = mintele.shape();

  // discretization type of source and target Element
  Core::FE::CellType psdt = source_elem.shape();
  Core::FE::CellType pmdt = target_elem.shape();

  // check input data
  if (((!source_elem.is_source()) || (target_elem.is_source())) and
      (!imortar_.get<bool>("Two_half_pass")))
    FOUR_C_THROW("Function is called on a wrong type of Mortar::Element pair!");
  if (cell == nullptr) FOUR_C_THROW("Function is called without integration cell");

  // contact with wear
  bool wear = false;
  if (wearlaw_ != Wear::wear_none) wear = true;

  // number of nodes (source, target)
  int nrow = source_elem.num_node();
  int ncol = target_elem.num_node();
  int ndof = n_dim();
  int nintrow = sintele.num_node();

  // create empty vectors for shape fct. evaluation
  Core::LinAlg::SerialDenseVector source_val(nrow);
  Core::LinAlg::SerialDenseMatrix source_deriv(nrow, 2, true);
  Core::LinAlg::SerialDenseVector svalmod(nrow);
  Core::LinAlg::SerialDenseMatrix sderivmod(nrow, 2, true);
  Core::LinAlg::SerialDenseVector target_val(ncol);
  Core::LinAlg::SerialDenseMatrix target_deriv(ncol, 2, true);
  Core::LinAlg::SerialDenseVector lm_val(nrow);
  Core::LinAlg::SerialDenseMatrix lm_deriv(nrow, 2, true);
  Core::LinAlg::SerialDenseVector lmintval(nintrow);
  Core::LinAlg::SerialDenseMatrix lmintderiv(nintrow, 2, true);

  // create empty vectors for shape fct. evaluation
  Core::LinAlg::SerialDenseMatrix ssecderiv(nrow, 3);

  // get source and target nodal coords for Jacobian / GP evaluation
  Core::LinAlg::SerialDenseMatrix source_coord(3, source_elem.num_node());
  source_elem.get_nodal_coords(source_coord);
  Core::LinAlg::SerialDenseMatrix target_coord(3, target_elem.num_node());
  target_elem.get_nodal_coords(target_coord);

  // nodal coords from previous time step and lagrange multiplier
  std::shared_ptr<Core::LinAlg::SerialDenseMatrix> source_coord_old;
  std::shared_ptr<Core::LinAlg::SerialDenseMatrix> target_coord_old;
  std::shared_ptr<Core::LinAlg::SerialDenseMatrix> lagmult;

  // get them in the case of wear
  if (wear or imortar_.get<bool>("GP_SLIP_INCR"))
  {
    source_coord_old = std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, source_elem.num_node());
    target_coord_old = std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, target_elem.num_node());
    lagmult = std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, source_elem.num_node());
    source_elem.get_nodal_coords_old(*source_coord_old);
    target_elem.get_nodal_coords_old(*target_coord_old);
    source_elem.get_nodal_lag_mult(*lagmult);
  }

  // get source element nodes themselves for normal evaluation
  Core::Nodes::Node** mynodes = source_elem.nodes();
  if (!mynodes) FOUR_C_THROW("Null pointer!");
  Core::Nodes::Node** myintnodes = sintele.nodes();
  if (!myintnodes) FOUR_C_THROW("Null pointer!");

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // prepare directional derivative of dual shape functions
  // this is necessary for all source element types except tri3
  bool duallin = false;
  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> dualmap(
      (nrow + ncol) * ndof, 0, Core::LinAlg::SerialDenseMatrix(nrow, nrow));
  if ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
      (source_elem.shape() != Core::FE::CellType::tri3 ||
          source_elem.mo_data().deriv_dual_shape() != nullptr) &&
      lag_mult_quad() != Mortar::lagmult_const)
  {
    duallin = true;
    source_elem.deriv_shape_dual(dualmap);
  }

  // decide whether boundary modification has to be considered or not
  // this is element-specific (is there a boundary node in this element?)
  bool bound = false;
  for (int k = 0; k < nrow; ++k)
  {
    Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[k]);
    if (!mymrtrnode) FOUR_C_THROW("Null pointer!");
    bound += mymrtrnode->is_on_bound();
  }

  // decide whether displacement shape fct. modification has to be considered or not
  // this is the case for dual quadratic Lagrange multipliers on quad8 and tri6 elements
  // check whether quadratic element with linear interpolation is present
  bool dualquad3d = false;
  bool linlm = false;
  if ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
      (lmtype == Mortar::lagmult_quad || lmtype == Mortar::lagmult_lin) &&
      (source_elem.shape() == Core::FE::CellType::quad8 ||
          source_elem.shape() == Core::FE::CellType::tri6))
  {
    if (lmtype == Mortar::lagmult_quad)
      dualquad3d = true;
    else
      linlm = true;
  }

  // get linearization size
  int linsize = 0;
  for (int i = 0; i < nrow; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(mynodes[i]);
    linsize += cnode->get_linsize();
  }

  // check if the cells are tri3
  // there's nothing wrong about other shapes, but as long as they are all
  // tri3 we can perform the jacobian calculation ( and its deriv) outside
  // the Gauss point loop
  if (cell->shape() != Core::FE::CellType::tri3)
    FOUR_C_THROW("only tri3 integration cells at the moment. See comment in the code");

  double jac = cell->jacobian();
  // directional derivative of cell Jacobian
  Core::Gen::Pairedvector<int, double> jacintcellmap((nrow + ncol) * ndof);
  cell->deriv_jacobian(jacintcellmap);

  //**********************************************************************
  // loop over all Gauss points for integration
  //**********************************************************************
  for (int gp = 0; gp < n_gp(); ++gp)
  {
    // coordinates and weight
    double eta[2] = {coordinate(gp, 0), coordinate(gp, 1)};
    double wgt = weight(gp);

    // get global Gauss point coordinates
    double globgp[3] = {0.0, 0.0, 0.0};
    cell->local_to_global(eta, globgp, 0);

    double source_xi[2] = {0.0, 0.0};
    double target_xi[2] = {0.0, 0.0};

    // project Gauss point onto source integration element
    // project Gauss point onto target integration element
    double sprojalpha = 0.0;
    double mprojalpha = 0.0;
    Mortar::Projector::impl(sintele)->project_gauss_point_auxn_3d(
        globgp, auxn, sintele, source_xi, sprojalpha);
    Mortar::Projector::impl(mintele)->project_gauss_point_auxn_3d(
        globgp, auxn, mintele, target_xi, mprojalpha);

    // check GP projection (SOURCE)
    const double tol = 0.01;
    if (sdt == Core::FE::CellType::quad4 || sdt == Core::FE::CellType::quad8 ||
        sdt == Core::FE::CellType::quad9)
    {
      if (source_xi[0] < -1.0 - tol || source_xi[1] < -1.0 - tol || source_xi[0] > 1.0 + tol ||
          source_xi[1] > 1.0 + tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane_quad: Source Gauss point "
                     "projection "
                     "outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Source GP (IntElement) projection: " << source_xi[0] << " " << source_xi[1]
                  << std::endl;
      }
    }
    else
    {
      if (source_xi[0] < -tol || source_xi[1] < -tol || source_xi[0] > 1.0 + tol ||
          source_xi[1] > 1.0 + tol || source_xi[0] + source_xi[1] > 1.0 + 2 * tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane_quad: Source Gauss point "
                     "projection "
                     "outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Source GP (IntElement) projection: " << source_xi[0] << " " << source_xi[1]
                  << std::endl;
      }
    }

    // check GP projection (TARGET)
    if (mdt == Core::FE::CellType::quad4 || mdt == Core::FE::CellType::quad8 ||
        mdt == Core::FE::CellType::quad9)
    {
      if (target_xi[0] < -1.0 - tol || target_xi[1] < -1.0 - tol || target_xi[0] > 1.0 + tol ||
          target_xi[1] > 1.0 + tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane_quad: Target Gauss point "
                     "projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Target GP (IntElement) projection: " << target_xi[0] << " " << target_xi[1]
                  << std::endl;
      }
    }
    else
    {
      if (target_xi[0] < -tol || target_xi[1] < -tol || target_xi[0] > 1.0 + tol ||
          target_xi[1] > 1.0 + tol || target_xi[0] + target_xi[1] > 1.0 + 2 * tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane_quad: Target Gauss point "
                     "projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Target GP (IntElement) projection: " << target_xi[0] << " " << target_xi[1]
                  << std::endl;
      }
    }

    // project Gauss point back to source (parent) element
    // project Gauss point back to target (parent) element
    double p_source_xi[2] = {0.0, 0.0};
    double p_target_xi[2] = {0.0, 0.0};
    double psprojalpha = 0.0;
    double pmprojalpha = 0.0;
    Mortar::Projector::impl(source_elem)
        ->project_gauss_point_auxn_3d(globgp, auxn, source_elem, p_source_xi, psprojalpha);
    Mortar::Projector::impl(target_elem)
        ->project_gauss_point_auxn_3d(globgp, auxn, target_elem, p_target_xi, pmprojalpha);
    // sintele.MapToParent(source_xi, p_source_xi); // old way of doing it via affine map... wrong
    // (popp 05/2016) mintele.MapToParent(target_xi, p_target_xi); // old way of doing it via affine
    // map... wrong (popp 05/2016)

    // check GP projection (SOURCE)
    if (psdt == Core::FE::CellType::quad4 || psdt == Core::FE::CellType::quad8 ||
        psdt == Core::FE::CellType::quad9 || psdt == Core::FE::CellType::nurbs9)
    {
      if (p_source_xi[0] < -1.0 - tol || p_source_xi[1] < -1.0 - tol ||
          p_source_xi[0] > 1.0 + tol || p_source_xi[1] > 1.0 + tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane_quad: Source Gauss point "
                     "projection "
                     "outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Source GP projection: " << p_source_xi[0] << " " << p_source_xi[1]
                  << std::endl;
      }
    }
    else
    {
      if (p_source_xi[0] < -tol || p_source_xi[1] < -tol || p_source_xi[0] > 1.0 + tol ||
          p_source_xi[1] > 1.0 + tol || p_source_xi[0] + p_source_xi[1] > 1.0 + 2 * tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane_quad: Source Gauss point "
                     "projection "
                     "outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Source GP projection: " << p_source_xi[0] << " " << p_source_xi[1]
                  << std::endl;
      }
    }

    // check GP projection (TARGET)
    if (pmdt == Core::FE::CellType::quad4 || pmdt == Core::FE::CellType::quad8 ||
        pmdt == Core::FE::CellType::quad9 || pmdt == Core::FE::CellType::nurbs9)
    {
      if (p_target_xi[0] < -1.0 - tol || p_target_xi[1] < -1.0 - tol ||
          p_target_xi[0] > 1.0 + tol || p_target_xi[1] > 1.0 + tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane_quad: Target Gauss point "
                     "projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Target GP projection: " << p_target_xi[0] << " " << p_target_xi[1]
                  << std::endl;
      }
    }
    else
    {
      if (p_target_xi[0] < -tol || p_target_xi[1] < -tol || p_target_xi[0] > 1.0 + tol ||
          p_target_xi[1] > 1.0 + tol || p_target_xi[0] + p_target_xi[1] > 1.0 + 2 * tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane_quad: Target Gauss point "
                     "projection outside!";
        std::cout << "Source ID: " << source_elem.id() << " Target ID: " << target_elem.id()
                  << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Target GP projection: " << p_target_xi[0] << " " << p_target_xi[1]
                  << std::endl;
      }
    }

    // evaluate Lagrange multiplier shape functions (on source element)
    if (lag_mult_quad() == Mortar::lagmult_const)
      source_elem.evaluate_shape_lag_mult_const(shape_fcn(), source_xi, lm_val, lm_deriv, nrow);
    else if (bound)
      source_elem.evaluate_shape_lag_mult(shape_fcn(), p_source_xi, lm_val, lm_deriv, nrow);
    else
    {
      source_elem.evaluate_shape_lag_mult(shape_fcn(), p_source_xi, lm_val, lm_deriv, nrow);
      sintele.evaluate_shape_lag_mult(shape_fcn(), source_xi, lmintval, lmintderiv, nintrow, false);
    }

    // evaluate trace space shape functions (on both elements)
    source_elem.evaluate_shape(p_source_xi, source_val, source_deriv, nrow, linlm);
    if (dualquad3d) source_elem.evaluate_shape(p_source_xi, svalmod, sderivmod, nrow, true);
    target_elem.evaluate_shape(p_target_xi, target_val, target_deriv, ncol);

    // evaluate 2nd deriv of trace space shape functions (on source element)
    source_elem.evaluate2nd_deriv_shape(p_source_xi, ssecderiv, nrow);

    // evaluate global GP coordinate derivative
    Core::LinAlg::Matrix<3, 1> svalcell;
    Core::LinAlg::Matrix<3, 2> sderivcell;
    cell->evaluate_shape(eta, svalcell, sderivcell);

    Core::Gen::Pairedvector<int, Core::LinAlg::Matrix<3, 1>> lingp((nrow + ncol) * ndof);

    for (int v = 0; v < 3; ++v)
      for (int d = 0; d < 3; ++d)
        for (CI p = (cell->get_deriv_vertex(v))[d].begin();
            p != (cell->get_deriv_vertex(v))[d].end(); ++p)
          lingp[p->first](d) += svalcell(v) * (p->second);

    // evaluate the GP source coordinate derivatives
    std::vector<Core::Gen::Pairedvector<int, double>> d_source_xi_gp(2, (nrow + ncol) * ndof);
    deriv_xi_gp_3d_aux_plane(sintele, source_xi, cell->auxn(), d_source_xi_gp, sprojalpha,
        cell->get_deriv_auxn(), lingp);

    // evaluate the GP target coordinate derivatives
    std::vector<Core::Gen::Pairedvector<int, double>> d_target_xi_gp(2, (nrow + ncol) * ndof);
    deriv_xi_gp_3d_aux_plane(mintele, target_xi, cell->auxn(), d_target_xi_gp, mprojalpha,
        cell->get_deriv_auxn(), lingp);

    // evaluate the GP source coordinate derivatives (parent element)
    // evaluate the GP target coordinate derivatives (parent element)
    std::vector<Core::Gen::Pairedvector<int, double>> dp_source_xigp(2, (nrow + ncol) * ndof);
    std::vector<Core::Gen::Pairedvector<int, double>> dp_target_xi_gp(2, (nrow + ncol) * ndof);
    deriv_xi_gp_3d_aux_plane(source_elem, p_source_xi, cell->auxn(), dp_source_xigp, psprojalpha,
        cell->get_deriv_auxn(), lingp);
    deriv_xi_gp_3d_aux_plane(target_elem, p_target_xi, cell->auxn(), dp_target_xi_gp, pmprojalpha,
        cell->get_deriv_auxn(), lingp);
    // sintele.MapToParent(d_source_xi_gp,dp_source_xigp); // old way of doing it via affine map...
    // wrong (popp 05/2016) mintele.MapToParent(d_target_xi_gp,dp_target_xi_gp); // old way of doing
    // it via affine map... wrong (popp 05/2016)

    //**********************************************************************
    // frequently reused quantities
    //**********************************************************************
    double gpn[3] = {0.0, 0.0, 0.0};
    double gap = 0.0;
    double jumpval[2] = {
        0.0, 0.0};  // jump for wear
                    // double jumpvalv[2] = {0.0,0.0};  // jump for slipincr --> equal to jumpval
    double wearval = 0.0;  // wear value
    double lengthn = 0.0;  // length of gp normal gpn
    Core::Gen::Pairedvector<int, double> dsliptmatrixgp(
        (ncol * ndof) + linsize);  // deriv. of slip for wear
    Core::Gen::Pairedvector<int, double> dgapgp(
        (ncol * ndof) + linsize);  // gap lin. without lm and jac.
    Core::Gen::Pairedvector<int, double> dweargp(
        (ncol * ndof) + linsize);  // wear lin. without lm and jac.
    std::vector<Core::Gen::Pairedvector<int, double>> dslipgp(
        2, ((ncol * ndof) + linsize));  // deriv. of slip for slipincr (xi, eta)
    std::vector<Core::Gen::Pairedvector<int, double>> dnmap_unit(
        3, linsize);  // deriv of x,y and z comp. of gpn (unit)

    //**********************************************************************
    // evaluate at GP and lin char. quantities
    //**********************************************************************
    // these cases don't need any special treatment for quadratic elements
    if (algo_ == Mortar::algorithm_gpts ||
        (lmtype == Mortar::lagmult_const && algo_ == Mortar::algorithm_mortar))
    {
      gap_3d(source_elem, target_elem, source_val, target_val, source_deriv, target_deriv, &gap,
          gpn, dp_source_xigp, dp_target_xi_gp, dgapgp, dnmap_unit);
      integrate_gp_3d(source_elem, target_elem, source_val, lm_val, target_val, source_deriv,
          target_deriv, lm_deriv, dualmap, wgt, jac, jacintcellmap, gpn, dnmap_unit, gap, dgapgp,
          p_source_xi, p_target_xi, dp_source_xigp, dp_target_xi_gp);
    }
    // special treatment for quadratic elements
    else
    {
      // integrate and lin gp gap
      if (shape_fcn() == Mortar::shape_standard && lmtype == Mortar::lagmult_pwlin)
      {
        gp_3d_g_quad_pwlin(source_elem, sintele, target_elem, source_val, target_val, lmintval,
            source_coord, target_coord, source_deriv, target_deriv, &gap, gpn, &lengthn, jac, wgt,
            d_source_xi_gp, d_target_xi_gp, dgapgp, dnmap_unit);
      }
      else
      {
        if (linlm)
        {
          // declare and compute shape functions as well as derivatives for computation of gap
          // -> standard quadratic functions instead of only linear functions
          Core::LinAlg::SerialDenseVector svalgap(nrow);
          Core::LinAlg::SerialDenseMatrix sderivgap(nrow, 2, true);
          source_elem.evaluate_shape(p_source_xi, svalgap, sderivgap, nrow);

          gap_3d(source_elem, target_elem, svalgap, target_val, sderivgap, target_deriv, &gap, gpn,
              dp_source_xigp, dp_target_xi_gp, dgapgp, dnmap_unit);
        }
        else
          gap_3d(source_elem, target_elem, source_val, target_val, source_deriv, target_deriv, &gap,
              gpn, dp_source_xigp, dp_target_xi_gp, dgapgp, dnmap_unit);

        gp_3d_w_gap(source_elem, source_val, lm_val, &gap, jac, wgt, true);
      }

      // compute cell D/M matrix *******************************************
      gp_3d_dm_quad(source_elem, target_elem, sintele, lm_val, lmintval, source_val, target_val,
          jac, wgt, nrow, nintrow, ncol, ndof, bound);

      //*******************************
      // WEAR stuff
      //*******************************
      // std. wear for all wear-algorithm types
      if (wear)
        gp_3d_wear(source_elem, target_elem, source_val, source_deriv, target_val, target_deriv,
            lm_val, lm_deriv, *lagmult, gpn, jac, wgt, jumpval, &wearval, dsliptmatrixgp, dweargp,
            d_source_xi_gp, d_target_xi_gp, dnmap_unit, dualmap);

      // integrate T and E matrix for discr. wear
      if (wear_type() == Wear::wear_primvar)
        gp_te(source_elem, lm_val, source_val, jac, wgt, jumpval);

      //********************************************************************
      // compute cell linearization
      //********************************************************************
      if (shape_fcn() == Mortar::shape_standard && lmtype == Mortar::lagmult_pwlin)
      {
        for (int iter = 0; iter < nintrow; ++iter)
        {
          // Lin DM
          gp_3d_dm_quad_pwlin_lin(iter, source_elem, sintele, target_elem, source_val, target_val,
              lmintval, source_deriv, target_deriv, lmintderiv, wgt, jac, d_source_xi_gp,
              dp_source_xigp, dp_target_xi_gp, jacintcellmap);

          // Lin gap
          gp_3d_g_quad_pwlin_lin(iter, sintele, source_val, lmintval, source_deriv, lmintderiv, gap,
              gpn, jac, wgt, dgapgp, jacintcellmap, d_source_xi_gp);
        }
      }
      else
      {
        // Lin DM
        gp_3d_dm_quad_lin(duallin, source_elem, target_elem, source_val, svalmod, target_val,
            lm_val, source_deriv, target_deriv, lm_deriv, wgt, jac, dp_source_xigp, dp_target_xi_gp,
            jacintcellmap, dualmap, dualquad3d);

        for (int iter = 0; iter < nrow; ++iter)
        {
          // Lin gap
          gp_3d_g_quad_lin(iter, source_elem, target_elem, source_val, svalmod, lm_val,
              source_deriv, lm_deriv, gap, gpn, jac, wgt, duallin, dgapgp, jacintcellmap,
              dp_source_xigp, dualmap, dualquad3d);

          // Lin wear matrices T and E for discr. wear
          if (wear_type() == Wear::wear_primvar)
            gp_3d_te_lin(iter, source_elem, source_val, lm_val, source_deriv, lm_deriv, jac, wgt,
                jumpval, d_source_xi_gp, jacintcellmap, dsliptmatrixgp, dualmap);
        }
      }
    }
  }  // gp loop
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_deriv_ele_2d(Mortar::Element& source_elem,
    std::vector<Mortar::Element*> target_elems, bool* boundary_ele,
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  std::shared_ptr<CONTACT::ParamsInterface> cparams_ptr = nullptr;
  if (mparams_ptr)
  {
    cparams_ptr = std::dynamic_pointer_cast<CONTACT::ParamsInterface>(mparams_ptr);
  }
  integrate_deriv_ele_2d(source_elem, target_elems, boundary_ele, cparams_ptr);
}

/*----------------------------------------------------------------------*
 |  Integrate and linearize mortar terms                     farah 01/13|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_deriv_ele_2d(Mortar::Element& source_elem,
    std::vector<Mortar::Element*> target_elems, bool* boundary_ele,
    const std::shared_ptr<CONTACT::ParamsInterface>& cparams_ptr)
{
  // ********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************

  // explicitly defined shape function type needed
  if (shape_fcn() == Mortar::shape_undefined)
    FOUR_C_THROW("IntegrateDerivEle2D called without specific shape function defined!");

  // Petrov-Galerkin approach for LM not yet implemented for quadratic FE
  if (source_elem.shape() == Core::FE::CellType::line3 &&
      shape_fcn() == Mortar::shape_petrovgalerkin)
    FOUR_C_THROW("Petrov-Galerkin approach not yet implemented for 2-D quadratic FE interpolation");

  // check for problem dimension
  if (n_dim() != 2) FOUR_C_THROW("2D integration method called for non-2D problem");

  // check input data
  for (int i = 0; i < (int)target_elems.size(); ++i)
  {
    if ((!source_elem.is_source()) || (target_elems[i]->is_source()))
      FOUR_C_THROW("IntegrateDerivEle2D called on a wrong type of Mortar::Element pair!");
  }

  // *********************************************************************
  // Define source quantities
  // *********************************************************************

  // consider entire source element --> parameter space [-1,1]
  double source_xi_a = -1.0;
  double source_xi_b = 1.0;

  // number of nodes (source, target)
  int ndof = n_dim();
  int nrow = 0;

  Core::Nodes::Node** mynodes = nullptr;
  nrow = source_elem.num_node();
  mynodes = source_elem.nodes();

  // get source element nodes themselves
  if (!mynodes) FOUR_C_THROW("Null pointer!");

  // create empty vectors for shape fct. evaluation
  Core::LinAlg::SerialDenseVector source_val(nrow);
  Core::LinAlg::SerialDenseMatrix source_deriv(nrow, 1);
  Core::LinAlg::SerialDenseVector lm_val(nrow);
  Core::LinAlg::SerialDenseMatrix lm_deriv(nrow, 1);

  // get source nodal coords for Jacobian / GP evaluation
  Core::LinAlg::SerialDenseMatrix source_coord(3, nrow);
  source_elem.get_nodal_coords(source_coord);

  // nodal coords from previous time step and lagrange multiplier
  std::shared_ptr<Core::LinAlg::SerialDenseMatrix> source_coord_old;
  std::shared_ptr<Core::LinAlg::SerialDenseMatrix> target_coord_old;
  std::shared_ptr<Core::LinAlg::SerialDenseMatrix> lagmult;

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // prepare directional derivative of dual shape functions
  // this is only necessary for quadratic dual shape functions in 2D
  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> dualmap(
      nrow * ndof, 0, Core::LinAlg::SerialDenseMatrix(nrow, nrow));
  if ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
      (source_elem.shape() == Core::FE::CellType::line3 ||
          source_elem.mo_data().deriv_dual_shape() != nullptr ||
          source_elem.shape() == Core::FE::CellType::nurbs3))
    source_elem.deriv_shape_dual(dualmap);

  // decide whether boundary modification has to be considered or not
  // this is element-specific (is there a boundary node in this element?)
  for (int k = 0; k < nrow; ++k)
  {
    Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[k]);
    if (!mymrtrnode) FOUR_C_THROW("Null pointer!");
  }

  // decide whether linear LM are used for quadratic FE here
  // and whether displacement shape fct. modification has to be considered or not
  // this is the case for dual linear Lagrange multipliers on line3 elements
  bool linlm = false;
  bool dualquad = false;
  if (lag_mult_quad() == Mortar::lagmult_lin && source_elem.shape() == Core::FE::CellType::line3)
  {
    linlm = true;
    if (shape_fcn() == Mortar::shape_dual) dualquad = true;
  }

  // get numerical integration type
  auto inttype = Teuchos::getIntegralValue<Mortar::IntType>(imortar_, "INTTYPE");

  //************************************************************************
  // Boundary Segmentation check -- HasProj()-check
  //************************************************************************
  if (inttype == Mortar::inttype_elements_BS)
    *boundary_ele = boundary_segm_check_2d(source_elem, target_elems);

  int linsize = 0;
  for (int i = 0; i < nrow; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(mynodes[i]);
    linsize += cnode->get_linsize();
  }

  if (*boundary_ele == false || inttype == Mortar::inttype_elements)
  {
    //*************************************************************************
    //                loop over all Gauss points for integration
    //*************************************************************************
    for (int gp = 0; gp < n_gp(); ++gp)
    {
      bool kink_projection = false;

      // coordinates and weight
      std::array<double, 2> eta = {coordinate(gp, 0), 0.0};
      double wgt = weight(gp);

      // coordinate transformation source_xi->eta (source Mortar::Element->Overlap)
      double source_xi[2] = {0.0, 0.0};
      source_xi[0] = eta[0];

      // evaluate the two source side Jacobians
      double dxdsxi = source_elem.jacobian(source_xi);

      // evaluate Lagrange multiplier shape functions (on source element)
      if (lag_mult_quad() == Mortar::lagmult_const)
        source_elem.evaluate_shape_lag_mult_const(shape_fcn(), source_xi, lm_val, lm_deriv, nrow);
      else if (linlm)
        source_elem.evaluate_shape_lag_mult_lin(shape_fcn(), source_xi, lm_val, lm_deriv, nrow);
      else
        source_elem.evaluate_shape_lag_mult(shape_fcn(), source_xi, lm_val, lm_deriv, nrow);

      // evaluate trace space shape functions
      source_elem.evaluate_shape(source_xi, source_val, source_deriv, nrow, dualquad);

      //****************************************************************************************************************
      //                loop over all Target Elements
      //****************************************************************************************************************
      for (int numtarget = 0; numtarget < (int)target_elems.size(); ++numtarget)
      {
        // project Gauss point onto target element
        double target_xi[2] = {0.0, 0.0};
        Mortar::Projector::impl(source_elem, *target_elems[numtarget])
            ->project_gauss_point_2d(source_elem, source_xi, *target_elems[numtarget], target_xi);

        // gp on target_elem?
        if ((target_xi[0] >= -1.0) && (target_xi[0] <= 1.0) && (kink_projection == false))
        {
          kink_projection = true;

          int ncol = 0;

          ncol = target_elems[numtarget]->num_node();

          Core::LinAlg::SerialDenseVector target_val(ncol);
          Core::LinAlg::SerialDenseMatrix target_deriv(ncol, 1);

          // get target nodal coords for Jacobian / GP evaluation
          Core::LinAlg::SerialDenseMatrix target_coord(3, ncol);
          target_elems[numtarget]->get_nodal_coords(target_coord);

          // evaluate trace space shape functions
          target_elems[numtarget]->evaluate_shape(target_xi, target_val, target_deriv, ncol, false);

          // get directional derivatives of source_xi_a, source_xi_b, target_xi_a, target_xi_b -->
          // derivatives of target_xi_a/target_xi_b not required
          std::vector<Core::Gen::Pairedvector<int, double>> ximaps(4, linsize + ndof * ncol);
          bool startsource = true;
          bool endsource = true;
          double target_xi_a = -0.1;  //--> arbitrary value
          double target_xi_b = 0.1;   //--> arbitrary value
          deriv_xi_a_b_2d(source_elem, source_xi_a, source_xi_b, *target_elems[numtarget],
              target_xi_a, target_xi_b, ximaps, startsource, endsource, linsize);

          // evaluate the GP source coordinate derivatives --> no entries
          std::vector<Core::Gen::Pairedvector<int, double>> d_source_xi_gp(
              1, Core::Gen::Pairedvector<int, double>(linsize + ndof * ncol));
          for (CI p = ximaps[0].begin(); p != ximaps[0].end(); ++p)
            d_source_xi_gp[0][p->first] = 0.0;

          // evaluate the GP target coordinate derivatives
          std::vector<Core::Gen::Pairedvector<int, double>> d_target_xi_gp(
              1, Core::Gen::Pairedvector<int, double>(linsize + ndof * ncol));
          deriv_xi_gp_2d(source_elem, *target_elems[numtarget], source_xi[0], target_xi[0],
              d_source_xi_gp[0], d_target_xi_gp[0], linsize);

          // evaluate the Jacobian derivative
          Core::Gen::Pairedvector<int, double> derivjac(nrow * ndof);
          source_elem.deriv_jacobian(
              source_xi, derivjac);  // direct derivative if xi^1_g does not change

          //**********************************************************************
          // frequently reused quantities
          //**********************************************************************
          double gpn[2] = {0.0, 0.0};  // normalized normal at gp
          std::vector<Core::Gen::Pairedvector<int, double>> dnmap_unit(
              2, (linsize + ndof * ncol));  // deriv of x and y comp. of gpn (unit)
          double gap = 0.0;                 // gap
          Core::Gen::Pairedvector<int, double> dgapgp(
              linsize + ndof * ncol);  // gap lin without weighting and jac

          //**********************************************************************
          // evaluate at GP and lin char. quantities
          //**********************************************************************
          if (dualquad && linlm)
          {
            // declare and compute shape functions as well as derivatives for computation of gap
            // -> standard quadratic functions instead of only linear functions
            Core::LinAlg::SerialDenseVector svalgap(nrow);
            Core::LinAlg::SerialDenseMatrix sderivgap(nrow, 1);
            source_elem.evaluate_shape(source_xi, svalgap, sderivgap, nrow);

            gap_2d(source_elem, *target_elems[numtarget], svalgap, target_val, sderivgap,
                target_deriv, &gap, gpn, d_source_xi_gp, d_target_xi_gp, dgapgp, dnmap_unit);
          }
          else
            gap_2d(source_elem, *target_elems[numtarget], source_val, target_val, source_deriv,
                target_deriv, &gap, gpn, d_source_xi_gp, d_target_xi_gp, dgapgp, dnmap_unit);

          // integrate and lin gp gap
          integrate_gp_2d(source_elem, *target_elems[numtarget], source_val, lm_val, target_val,
              source_deriv, target_deriv, lm_deriv, dualmap, wgt, dxdsxi, derivjac, gpn, dnmap_unit,
              gap, dgapgp, source_xi, target_xi, d_source_xi_gp, d_target_xi_gp);
        }
      }  // End Loop over all Target Elements
    }  // End Loop over all GP
  }  // boundary_ele check
}

/*----------------------------------------------------------------------*
 |  Integrate D                                              farah 09/14|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_d(Mortar::Element& source_elem, MPI_Comm comm, bool lin)
{
  // ********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************

  // explicitly defined shape function type needed
  if (shape_fcn() == Mortar::shape_undefined)
    FOUR_C_THROW("IntegrateD called without specific shape function defined!");

  // *********************************************************************
  // Define source quantities
  // *********************************************************************
  // number of nodes (source, target)
  int ndof = n_dim();
  int nrow = 0;

  Core::Nodes::Node** mynodes = nullptr;
  nrow = source_elem.num_node();
  mynodes = source_elem.nodes();

  // get source element nodes themselves
  if (!mynodes) FOUR_C_THROW("Null pointer!");

  // create empty vectors for shape fct. evaluation
  Core::LinAlg::SerialDenseVector source_val(nrow);
  Core::LinAlg::SerialDenseMatrix source_deriv(nrow, ndof - 1);
  Core::LinAlg::SerialDenseVector lm_val(nrow);
  Core::LinAlg::SerialDenseMatrix lm_deriv(nrow, ndof - 1);

  int linsize = 0;
  for (int i = 0; i < nrow; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(mynodes[i]);
    linsize += cnode->get_linsize();
  }

  // decide whether boundary modification has to be considered or not
  // this is element-specific (is there a boundary node in this element?)
  bool bound = false;
  for (int k = 0; k < nrow; ++k)
  {
    Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[k]);
    if (!mymrtrnode) FOUR_C_THROW("Null pointer!");
    bound += mymrtrnode->is_on_bound();
  }

  // prepare directional derivative of dual shape functions
  // this is necessary for all source element types except tri3
  bool duallin = false;
  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> dualmap(
      nrow * ndof, 0, Core::LinAlg::SerialDenseMatrix(nrow, nrow));
  if ((source_elem.shape() != Core::FE::CellType::tri3 and
          source_elem.shape() != Core::FE::CellType::line2) ||
      source_elem.mo_data().deriv_dual_shape() != nullptr)
  {
    duallin = true;
    source_elem.deriv_shape_dual(dualmap);
  }

  // d-matrix derivative
  // local for this element to avoid direct assembly into the nodes for performance reasons
  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> dMatrixDeriv(
      source_elem.num_node() * n_dim(), 0,
      Core::LinAlg::SerialDenseMatrix(source_elem.num_node(), source_elem.num_node()));

  //*************************************************************************
  //                loop over all Gauss points for integration
  //*************************************************************************
  for (int gp = 0; gp < n_gp(); ++gp)
  {
    // coordinates and weight
    std::array<double, 2> eta = {coordinate(gp, 0), 0.0};
    double wgt = weight(gp);
    if (ndof == 3) eta[1] = coordinate(gp, 1);

    // coordinate transformation source_xi->eta (source Mortar::Element->Overlap)
    double source_xi[2] = {0.0, 0.0};
    source_xi[0] = eta[0];
    source_xi[1] = eta[1];

    // evaluate the two source side Jacobians
    double dxdsxi = source_elem.jacobian(source_xi);

    // evaluate linearizations *******************************************
    // evaluate the source Jacobian derivative
    Core::Gen::Pairedvector<int, double> jacsourcemap(nrow * ndof + linsize);
    source_elem.deriv_jacobian(source_xi, jacsourcemap);

    // evaluate Lagrange multiplier shape functions (on source element)
    source_elem.evaluate_shape_lag_mult(Mortar::shape_dual, source_xi, lm_val, lm_deriv, nrow);

    // evaluate trace space shape functions
    source_elem.evaluate_shape(source_xi, source_val, source_deriv, nrow, false);

    //**********************************************************************
    // evaluate at GP and lin char. quantities
    //**********************************************************************
    // integrate D and M
    for (int j = 0; j < nrow; ++j)
    {
      CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(mynodes[j]);

      // integrate dseg (boundary modification)
      if (bound)
      {
        bool j_boundnode = cnode->is_on_bound();

        for (int k = 0; k < nrow; ++k)
        {
          CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(mynodes[k]);
          bool k_boundnode = target_node->is_on_bound();

          // do not assemble off-diagonal terms if j,k are both non-boundary nodes
          if (!j_boundnode && !k_boundnode && (j != k)) continue;

          // multiply the two shape functions
          double prod = lm_val[j] * source_val[k] * dxdsxi * wgt;

          // isolate the dseg entries to be filled
          // (both the main diagonal and every other secondary diagonal)
          // and add current Gauss point's contribution to dseg
          if (target_node->is_on_bound())
          {
            if (abs(prod) > MORTARINTTOL) cnode->add_m_value(target_node->id(), -prod);
            if (abs(prod) > MORTARINTTOL)
              cnode->add_target_node(target_node->id());  // only for friction!
          }
          else
          {
            if (abs(prod) > MORTARINTTOL) cnode->add_d_value(target_node->id(), prod);
            if (abs(prod) > MORTARINTTOL)
              cnode->add_source_node(target_node->id());  // only for friction!
          }
        }
      }
      else
      {
        // integrate dseg
        for (int k = 0; k < nrow; ++k)
        {
          CONTACT::Node* source_node = dynamic_cast<CONTACT::Node*>(mynodes[k]);

          // multiply the two shape functions
          double prod = lm_val[j] * source_val[k] * dxdsxi * wgt;


          if (source_elem.is_source())
          {
            if (abs(prod) > MORTARINTTOL) cnode->add_d_value(source_node->id(), prod);
          }

          // loop over source dofs
          for (int jdof = 0; jdof < ndof; ++jdof)
          {
            int col = source_node->dofs()[jdof];

            if (source_elem.is_source())
            {
            }
            else
            {
              if (source_elem.owner() == Core::Communication::my_mpi_rank(comm))
              {
                if (abs(prod) > MORTARINTTOL)
                  dynamic_cast<CONTACT::FriNode*>(cnode)->add_d2_value(jdof, col, prod);
              }
            }
          }
        }
      }
    }

    if (lin)
    {
      using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

      // (1) Lin(Phi) - dual shape functions
      if (duallin)
      {
        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 dualmap.begin();
            p != dualmap.end(); ++p)
        {
          Core::LinAlg::SerialDenseMatrix& dderivtmp = dMatrixDeriv[p->first];
          for (int j = 0; j < nrow; ++j)
            for (int k = 0; k < nrow; ++k)
              for (int m = 0; m < nrow; ++m)
                dderivtmp(j, j) += wgt * source_val[m] * source_val[k] * dxdsxi * (p->second)(j, m);
        }
      }

      // (4) Lin(dsxideta) - intcell GP Jacobian
      for (CI p = jacsourcemap.begin(); p != jacsourcemap.end(); ++p)
      {
        Core::LinAlg::SerialDenseMatrix& dderivtmp = dMatrixDeriv[p->first];
        for (int j = 0; j < nrow; ++j)
          for (int k = 0; k < nrow; ++k)
            dderivtmp(j, j) += wgt * lm_val[j] * source_val[k] * (p->second);
      }
    }
  }  // End Loop over all GP

  source_elem.mo_data().reset_dual_shape();
  source_elem.mo_data().reset_deriv_dual_shape();
}


/*----------------------------------------------------------------------*
 |  Compute penalty scaling factor kappa                     farah 11/16|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_kappa_penalty_lts(Mortar::Element& source_elem)
{
  // number of nodes (source)
  int nrow = source_elem.num_node();

  // create empty objects for shape fct. evaluation
  Core::LinAlg::SerialDenseVector val(nrow);
  Core::LinAlg::SerialDenseMatrix deriv(nrow, 2, true);

  // get source element nodes themselves
  Core::Nodes::Node** mynodes = source_elem.nodes();

  //**********************************************************************
  // loop over all Gauss points for integration
  //**********************************************************************
  for (int gp = 0; gp < n_gp(); ++gp)
  {
    // coordinates and weight
    double eta[2] = {coordinate(gp, 0), 0.0};
    if (n_dim() == 3) eta[1] = coordinate(gp, 1);
    double wgt = weight(gp);

    // evaluate shape functions
    source_elem.evaluate_shape_lag_mult(shape_fcn(), eta, val, deriv, nrow);

    // evaluate the Jacobian det
    const double jac = source_elem.jacobian(eta);

    // compute cell gap vector *******************************************
    // loop over all gseg vector entries
    // nrow represents the source side dofs !!!  */
    for (int j = 0; j < nrow; ++j)
    {
      Node* mymrtrnode = dynamic_cast<Node*>(mynodes[j]);

      if (mymrtrnode->is_on_corner()) continue;

      // add current Gauss point's contribution to kappa
      mymrtrnode->data().kappa() += val[j] * jac * wgt;
    }
  }
}


/*----------------------------------------------------------------------*
 |  Compute penalty scaling factor kappa                      popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_kappa_penalty(Mortar::Element& source_elem, double* source_xi_a,
    double* source_xi_b, Core::LinAlg::SerialDenseVector& gseg)
{
  // explicitly defined shape function type needed
  if (shape_fcn() == Mortar::shape_undefined)
    FOUR_C_THROW("Function is called without specific shape function defined!");

  // check input data
  if (!source_elem.is_source()) FOUR_C_THROW("Function is called on a non-source Mortar::Element!");
  if ((source_xi_a[0] < -1.0) || (source_xi_a[1] < -1.0) || (source_xi_b[0] > 1.0) ||
      (source_xi_b[1] > 1.0))
    FOUR_C_THROW("Function is called with infeasible source limits!");

  // number of nodes (source)
  int nrow = source_elem.num_node();

  // create empty objects for shape fct. evaluation
  Core::LinAlg::SerialDenseVector val(nrow);
  Core::LinAlg::SerialDenseMatrix deriv(nrow, 2, true);


  // get source element nodes themselves
  Core::Nodes::Node** mynodes = source_elem.nodes();
  if (!mynodes) FOUR_C_THROW("Null pointer!");

  // decide whether boundary modification has to be considered or not
  // this is element-specific (is there a boundary node in this element?)
  bool bound = false;
  for (int k = 0; k < nrow; ++k)
  {
    Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[k]);
    if (!mymrtrnode) FOUR_C_THROW("Null pointer!");
    bound += mymrtrnode->is_on_bound();
  }

  //**********************************************************************
  // loop over all Gauss points for integration
  //**********************************************************************
  for (int gp = 0; gp < n_gp(); ++gp)
  {
    // coordinates and weight
    double eta[2] = {coordinate(gp, 0), 0.0};
    if (n_dim() == 3) eta[1] = coordinate(gp, 1);
    double wgt = weight(gp);

    // evaluate shape functions
    if (bound)
      source_elem.evaluate_shape_lag_mult_lin(shape_fcn(), eta, val, deriv, nrow);
    else
      source_elem.evaluate_shape_lag_mult(shape_fcn(), eta, val, deriv, nrow);

    // evaluate the Jacobian det
    const double jac = source_elem.jacobian(eta);

    // compute cell gap vector *******************************************
    // loop over all gseg vector entries
    // nrow represents the source side dofs !!!  */
    for (int j = 0; j < nrow; ++j)
    {
      // add current Gauss point's contribution to gseg
      (gseg)(j) += val[j] * jac * wgt;
    }
    // compute cell gap vector *******************************************
  }
}

/*----------------------------------------------------------------------*
 |  Compute penalty scaling factor kappa (3D piecewise lin)   popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_kappa_penalty(Mortar::Element& source_elem,
    Mortar::IntElement& sintele, double* source_xi_a, double* source_xi_b,
    Core::LinAlg::SerialDenseVector& gseg)
{
  // get LMtype
  Mortar::LagMultQuad lmtype = lag_mult_quad();

  // explicitly defined shape function type needed
  if (shape_fcn() != Mortar::shape_standard)
    FOUR_C_THROW("integrate_kappa_penalty -> you should not be here!");

  // check input data
  if (!source_elem.is_source())
    FOUR_C_THROW("integrate_kappa_penalty called on a non-source Mortar::Element!");
  if ((source_xi_a[0] < -1.0) || (source_xi_a[1] < -1.0) || (source_xi_b[0] > 1.0) ||
      (source_xi_b[1] > 1.0))
    FOUR_C_THROW("integrate_kappa_penalty called with infeasible source limits!");

  // number of nodes (source)
  int nrow = source_elem.num_node();
  int nintrow = sintele.num_node();

  // create empty objects for shape fct. evaluation
  Core::LinAlg::SerialDenseVector val(nrow);
  Core::LinAlg::SerialDenseMatrix deriv(nrow, 2, true);
  Core::LinAlg::SerialDenseVector intval(nintrow);
  Core::LinAlg::SerialDenseMatrix intderiv(nintrow, 2, true);

  //**********************************************************************
  // loop over all Gauss points for integration
  //**********************************************************************
  for (int gp = 0; gp < n_gp(); ++gp)
  {
    // coordinates and weight
    double eta[2] = {coordinate(gp, 0), 0.0};
    if (n_dim() == 3) eta[1] = coordinate(gp, 1);
    double wgt = weight(gp);

    // get global Gauss point coordinates
    double globgp[3] = {0.0, 0.0, 0.0};
    sintele.local_to_global(eta, globgp, 0);

    // get normal vector
    double auxn[3] = {0.0, 0.0, 0.0};
    sintele.compute_unit_normal_at_xi(eta, auxn);

    // project Gauss point back to source (parent) element
    double p_source_xi[2] = {0.0, 0.0};
    double psprojalpha = 0.0;
    Mortar::Projector::impl(source_elem)
        ->project_gauss_point_auxn_3d(globgp, auxn, source_elem, p_source_xi, psprojalpha);
    // sintele.MapToParent(eta,p_source_xi); //old way of doing it via affine map... wrong (popp
    // 05/2016)

    // evaluate shape functions
    source_elem.evaluate_shape(p_source_xi, val, deriv, nrow);
    sintele.evaluate_shape(eta, intval, intderiv, nintrow);

    // evaluate the Jacobian det
    const double jac = sintele.jacobian(eta);

    // compute cell gap vector *******************************************
    if (lmtype == Mortar::lagmult_pwlin)
    {
      for (int j = 0; j < nintrow; ++j)
      {
        // add current Gauss point's contribution to gseg
        (gseg)(j) += intval[j] * jac * wgt;
      }
    }
    else
    {
      FOUR_C_THROW("Invalid LM interpolation case!");
    }
    // compute cell gap vector *******************************************
  }
}

/*----------------------------------------------------------------------*
 |  Compute directional derivative of XiAB (2D)               popp 05/08|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::deriv_xi_a_b_2d(const Mortar::Element& source_elem, double source_xi_a,
    double source_xi_b, const Mortar::Element& target_elem, double target_xi_a, double target_xi_b,
    std::vector<Core::Gen::Pairedvector<int, double>>& derivxi, bool startsource, bool endsource,
    int linsize) const
{
  // check for problem dimension
  if (n_dim() != 2) FOUR_C_THROW("2D integration method called for non-2D problem");

  // we need the participating source and target nodes
  const Core::Nodes::Node* const* source_nodes = nullptr;
  const Core::Nodes::Node* const* target_nodes = nullptr;
  int num_source_nodes = source_elem.num_node();
  int num_target_nodes = target_elem.num_node();

  int ndof = n_dim();

  source_nodes = source_elem.nodes();
  target_nodes = target_elem.nodes();

  std::vector<const Mortar::Node*> source_mortar_nodes(num_source_nodes);
  std::vector<const Mortar::Node*> target_mortar_nodes(num_target_nodes);

  for (int i = 0; i < num_source_nodes; ++i)
  {
    source_mortar_nodes[i] = dynamic_cast<const Mortar::Node*>(source_nodes[i]);
    if (!source_mortar_nodes[i]) FOUR_C_THROW("Null pointer!");
  }

  for (int i = 0; i < num_target_nodes; ++i)
  {
    target_mortar_nodes[i] = dynamic_cast<const Mortar::Node*>(target_nodes[i]);
    if (!target_mortar_nodes[i]) FOUR_C_THROW("Null pointer!");
  }

  // we also need shape function derivs in A and B
  double p_source_xi_a[2] = {source_xi_a, 0.0};
  double p_source_xi_b[2] = {source_xi_b, 0.0};
  double p_target_xi_a[2] = {target_xi_a, 0.0};
  double p_target_xi_b[2] = {target_xi_b, 0.0};
  Core::LinAlg::SerialDenseVector source_vals_xi_a(num_source_nodes);
  Core::LinAlg::SerialDenseVector source_vals_xi_b(num_source_nodes);
  Core::LinAlg::SerialDenseVector target_vals_xi_a(num_target_nodes);
  Core::LinAlg::SerialDenseVector target_vals_xi_b(num_target_nodes);
  Core::LinAlg::SerialDenseMatrix source_derivs_xi_a(num_source_nodes, 1);
  Core::LinAlg::SerialDenseMatrix source_derivs_xi_b(num_source_nodes, 1);
  Core::LinAlg::SerialDenseMatrix target_derivs_xi_a(num_target_nodes, 1);
  Core::LinAlg::SerialDenseMatrix target_derivs_xi_b(num_target_nodes, 1);

  source_elem.evaluate_shape(
      p_source_xi_a, source_vals_xi_a, source_derivs_xi_a, num_source_nodes, false);
  source_elem.evaluate_shape(
      p_source_xi_b, source_vals_xi_b, source_derivs_xi_b, num_source_nodes, false);
  target_elem.evaluate_shape(
      p_target_xi_a, target_vals_xi_a, target_derivs_xi_a, num_target_nodes, false);
  target_elem.evaluate_shape(
      p_target_xi_b, target_vals_xi_b, target_derivs_xi_b, num_target_nodes, false);

  // prepare linearizations
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // compute leading constant for DerivXiBTarget if start node = source node
  if (startsource == true)
  {
    // compute factors and leading constants for target
    double c_target_xi_b = 0.0;
    double fac_dxm_b = 0.0;
    double fac_dym_b = 0.0;
    double fac_xmsl_b = 0.0;
    double fac_ymsl_b = 0.0;

    // there are only actual 2 components for the normal in 2D but compute_unit_normal_at_xi expects
    // 3 components (otherwise invalid memory access)
    double normal[3] = {0., 0., 0.};
    source_elem.compute_unit_normal_at_xi(p_source_xi_a, normal);
    std::vector<Core::Gen::Pairedvector<int, double>> derivN;
    dynamic_cast<const CONTACT::Element*>(&source_elem)
        ->deriv_unit_normal_at_xi(p_source_xi_a, derivN);

    Core::LinAlg::SerialDenseVector* target_val = nullptr;
    Core::LinAlg::SerialDenseMatrix* target_deriv = nullptr;
    if (source_elem.normal_fac() * target_elem.normal_fac() > 0.)
    {
      target_val = &target_vals_xi_b;
      target_deriv = &target_derivs_xi_b;
    }
    else
    {
      target_val = &target_vals_xi_a;
      target_deriv = &target_derivs_xi_a;
    }

    for (int i = 0; i < num_target_nodes; ++i)
    {
      fac_dxm_b += (*target_deriv)(i, 0) * (target_mortar_nodes[i]->xspatial()[0]);
      fac_dym_b += (*target_deriv)(i, 0) * (target_mortar_nodes[i]->xspatial()[1]);
      fac_xmsl_b += (*target_val)[i] * (target_mortar_nodes[i]->xspatial()[0]);
      fac_ymsl_b += (*target_val)[i] * (target_mortar_nodes[i]->xspatial()[1]);
    }

    c_target_xi_b = -1 / (fac_dxm_b * (normal[1]) - fac_dym_b * (normal[0]));
    // std::cout << "c_target_xi_b: " << c_target_xi_b << std::endl;

    for (int i = 0; i < num_source_nodes; ++i)
    {
      fac_xmsl_b -= source_vals_xi_a[i] * (source_mortar_nodes[i]->xspatial()[0]);
      fac_ymsl_b -= source_vals_xi_a[i] * (source_mortar_nodes[i]->xspatial()[1]);
    }

    Core::Gen::Pairedvector<int, double> dmap_target_xi_b(num_target_nodes * ndof + linsize);

    // add derivative of source node coordinates
    for (int i = 0; i < num_source_nodes; ++i)
    {
      dmap_target_xi_b[source_mortar_nodes[i]->dofs()[0]] -= source_vals_xi_a[i] * normal[1];
      dmap_target_xi_b[source_mortar_nodes[i]->dofs()[1]] += source_vals_xi_a[i] * normal[0];
    }
    // add derivatives of target node coordinates
    for (int i = 0; i < num_target_nodes; ++i)
    {
      dmap_target_xi_b[target_mortar_nodes[i]->dofs()[0]] += (*target_val)[i] * (normal[1]);
      dmap_target_xi_b[target_mortar_nodes[i]->dofs()[1]] -= (*target_val)[i] * (normal[0]);
    }

    // add derivative of source node normal
    for (CI p = derivN[0].begin(); p != derivN[0].end(); ++p)
      dmap_target_xi_b[p->first] -= fac_ymsl_b * (p->second);
    for (CI p = derivN[1].begin(); p != derivN[1].end(); ++p)
      dmap_target_xi_b[p->first] += fac_xmsl_b * (p->second);

    // multiply all entries with c_target_xi_b
    for (CI p = dmap_target_xi_b.begin(); p != dmap_target_xi_b.end(); ++p)
      dmap_target_xi_b[p->first] = c_target_xi_b * (p->second);

    // return map to DerivM() method
    if (source_elem.normal_fac() * target_elem.normal_fac() > 0.)
      derivxi[3] = dmap_target_xi_b;
    else
      derivxi[2] = dmap_target_xi_b;
  }

  // compute leading constant for DerivXiATarget if end node = source node
  if (endsource == true)
  {
    // compute factors and leading constants for target
    double c_target_xi_a = 0.0;
    double fac_dxm_a = 0.0;
    double fac_dym_a = 0.0;
    double fac_xmsl_a = 0.0;
    double fac_ymsl_a = 0.0;

    // there are only 2 actual components for the normal in 2D but compute_unit_normal_at_xi expects
    // 3 components (otherwise invalid memory access)
    double normal[3] = {0., 0., 0.};
    source_elem.compute_unit_normal_at_xi(p_source_xi_b, normal);
    std::vector<Core::Gen::Pairedvector<int, double>> derivN;
    dynamic_cast<const CONTACT::Element*>(&source_elem)
        ->deriv_unit_normal_at_xi(p_source_xi_b, derivN);

    Core::LinAlg::SerialDenseVector* target_val = nullptr;
    Core::LinAlg::SerialDenseMatrix* target_deriv = nullptr;
    if (source_elem.normal_fac() * target_elem.normal_fac() > 0.)
    {
      target_val = &target_vals_xi_a;
      target_deriv = &target_derivs_xi_a;
    }
    else
    {
      target_val = &target_vals_xi_b;
      target_deriv = &target_derivs_xi_b;
    }

    for (int i = 0; i < num_target_nodes; ++i)
    {
      fac_dxm_a += (*target_deriv)(i, 0) * (target_mortar_nodes[i]->xspatial()[0]);
      fac_dym_a += (*target_deriv)(i, 0) * (target_mortar_nodes[i]->xspatial()[1]);
      fac_xmsl_a += (*target_val)[i] * (target_mortar_nodes[i]->xspatial()[0]);
      fac_ymsl_a += (*target_val)[i] * (target_mortar_nodes[i]->xspatial()[1]);
    }

    c_target_xi_a = -1 / (fac_dxm_a * (source_mortar_nodes[1]->mo_data().n()[1]) -
                             fac_dym_a * (source_mortar_nodes[1]->mo_data().n()[0]));
    // std::cout << "c_target_xi_a: " << c_target_xi_a << std::endl;

    for (int i = 0; i < num_source_nodes; ++i)
    {
      fac_xmsl_a -= source_vals_xi_b[i] * (source_mortar_nodes[i]->xspatial()[0]);
      fac_ymsl_a -= source_vals_xi_b[i] * (source_mortar_nodes[i]->xspatial()[1]);
    }

    Core::Gen::Pairedvector<int, double> dmap_target_xi_a(num_target_nodes * ndof + linsize);

    // add derivative of source node coordinates
    for (int i = 0; i < num_source_nodes; ++i)
    {
      dmap_target_xi_a[source_mortar_nodes[i]->dofs()[0]] -= source_vals_xi_b[i] * normal[1];
      dmap_target_xi_a[source_mortar_nodes[i]->dofs()[1]] += source_vals_xi_b[i] * normal[0];
    }

    // add derivatives of target node coordinates
    for (int i = 0; i < num_target_nodes; ++i)
    {
      dmap_target_xi_a[target_mortar_nodes[i]->dofs()[0]] += (*target_val)[i] * (normal[1]);
      dmap_target_xi_a[target_mortar_nodes[i]->dofs()[1]] -= (*target_val)[i] * (normal[0]);
    }

    // add derivative of source node normal
    for (CI p = derivN[0].begin(); p != derivN[0].end(); ++p)
      dmap_target_xi_a[p->first] -= fac_ymsl_a * (p->second);
    for (CI p = derivN[1].begin(); p != derivN[1].end(); ++p)
      dmap_target_xi_a[p->first] += fac_xmsl_a * (p->second);

    // multiply all entries with c_target_xi_a
    for (CI p = dmap_target_xi_a.begin(); p != dmap_target_xi_a.end(); ++p)
      dmap_target_xi_a[p->first] = c_target_xi_a * (p->second);

    // return map to DerivM() method
    if (source_elem.normal_fac() * target_elem.normal_fac() > 0.)
      derivxi[2] = dmap_target_xi_a;
    else
      derivxi[3] = dmap_target_xi_a;
  }

  // compute leading constant for DerivXiASource if start node = target node
  if (startsource == false)
  {
    // compute factors and leading constants for source
    double csxia = 0.0;
    double fac_dxsl_a = 0.0;
    double fac_dysl_a = 0.0;
    double fac_xslm_a = 0.0;
    double fac_yslm_a = 0.0;
    double fac_dnx_a = 0.0;
    double fac_dny_a = 0.0;
    double fac_nx_a = 0.0;
    double fac_ny_a = 0.0;

    Core::LinAlg::SerialDenseVector* target_val = nullptr;
    if (source_elem.normal_fac() * target_elem.normal_fac() > 0.)
      target_val = &target_vals_xi_b;
    else
      target_val = &target_vals_xi_a;

    for (int i = 0; i < num_source_nodes; ++i)
    {
      fac_dxsl_a += source_derivs_xi_a(i, 0) * (source_mortar_nodes[i]->xspatial()[0]);
      fac_dysl_a += source_derivs_xi_a(i, 0) * (source_mortar_nodes[i]->xspatial()[1]);
      fac_xslm_a += source_vals_xi_a[i] * (source_mortar_nodes[i]->xspatial()[0]);
      fac_yslm_a += source_vals_xi_a[i] * (source_mortar_nodes[i]->xspatial()[1]);
      fac_dnx_a += source_derivs_xi_a(i, 0) * (source_mortar_nodes[i]->mo_data().n()[0]);
      fac_dny_a += source_derivs_xi_a(i, 0) * (source_mortar_nodes[i]->mo_data().n()[1]);
      fac_nx_a += source_vals_xi_a[i] * (source_mortar_nodes[i]->mo_data().n()[0]);
      fac_ny_a += source_vals_xi_a[i] * (source_mortar_nodes[i]->mo_data().n()[1]);
    }

    for (int i = 0; i < num_target_nodes; ++i)
    {
      fac_xslm_a -= (*target_val)[i] * (target_mortar_nodes[i]->xspatial()[0]);
      fac_yslm_a -= (*target_val)[i] * (target_mortar_nodes[i]->xspatial()[1]);
    }

    csxia = -1. / (fac_dxsl_a * fac_ny_a - fac_dysl_a * fac_nx_a + fac_xslm_a * fac_dny_a -
                      fac_yslm_a * fac_dnx_a);
    // std::cout << "csxia: " << csxia << std::endl;


    Core::Gen::Pairedvector<int, double> dmap_sxia(num_target_nodes * ndof + linsize);

    // add derivative of target node coordinates
    for (int i = 0; i < num_target_nodes; ++i)
    {
      dmap_sxia[target_mortar_nodes[i]->dofs()[0]] -= (*target_val)[i] * fac_ny_a;
      dmap_sxia[target_mortar_nodes[i]->dofs()[1]] += (*target_val)[i] * fac_nx_a;
    }

    // add derivatives of source node coordinates
    for (int i = 0; i < num_source_nodes; ++i)
    {
      dmap_sxia[source_mortar_nodes[i]->dofs()[0]] += source_vals_xi_a[i] * fac_ny_a;
      dmap_sxia[source_mortar_nodes[i]->dofs()[1]] -= source_vals_xi_a[i] * fac_nx_a;
    }

    // add derivatives of source node normals
    for (int i = 0; i < num_source_nodes; ++i)
    {
      const Core::Gen::Pairedvector<int, double>& nxmap_curr =
          static_cast<const CONTACT::Node*>(source_mortar_nodes[i])->data().get_deriv_n()[0];
      const Core::Gen::Pairedvector<int, double>& nymap_curr =
          static_cast<const CONTACT::Node*>(source_mortar_nodes[i])->data().get_deriv_n()[1];

      for (CI p = nxmap_curr.begin(); p != nxmap_curr.end(); ++p)
        dmap_sxia[p->first] -= source_vals_xi_a[i] * fac_yslm_a * (p->second);
      for (CI p = nymap_curr.begin(); p != nymap_curr.end(); ++p)
        dmap_sxia[p->first] += source_vals_xi_a[i] * fac_xslm_a * (p->second);
    }

    // multiply all entries with csxia
    for (CI p = dmap_sxia.begin(); p != dmap_sxia.end(); ++p)
      dmap_sxia[p->first] = csxia * (p->second);

    // return map to DerivM() method
    derivxi[0] = dmap_sxia;
  }

  // compute leading constant for DerivXiBSource if end node = target node
  if (endsource == false)
  {
    // compute factors and leading constants for source
    double csxib = 0.0;
    double fac_dxsl_b = 0.0;
    double fac_dysl_b = 0.0;
    double fac_xslm_b = 0.0;
    double fac_yslm_b = 0.0;
    double fac_dnx_b = 0.0;
    double fac_dny_b = 0.0;
    double fac_nx_b = 0.0;
    double fac_ny_b = 0.0;

    Core::LinAlg::SerialDenseVector* target_val = nullptr;
    if (source_elem.normal_fac() * target_elem.normal_fac() > 0.)
      target_val = &target_vals_xi_a;
    else
      target_val = &target_vals_xi_b;

    for (int i = 0; i < num_source_nodes; ++i)
    {
      fac_dxsl_b += source_derivs_xi_b(i, 0) * (source_mortar_nodes[i]->xspatial()[0]);
      fac_dysl_b += source_derivs_xi_b(i, 0) * (source_mortar_nodes[i]->xspatial()[1]);
      fac_xslm_b += source_vals_xi_b[i] * (source_mortar_nodes[i]->xspatial()[0]);
      fac_yslm_b += source_vals_xi_b[i] * (source_mortar_nodes[i]->xspatial()[1]);
      fac_dnx_b += source_derivs_xi_b(i, 0) * (source_mortar_nodes[i]->mo_data().n()[0]);
      fac_dny_b += source_derivs_xi_b(i, 0) * (source_mortar_nodes[i]->mo_data().n()[1]);
      fac_nx_b += source_vals_xi_b[i] * (source_mortar_nodes[i]->mo_data().n()[0]);
      fac_ny_b += source_vals_xi_b[i] * (source_mortar_nodes[i]->mo_data().n()[1]);
    }

    for (int i = 0; i < num_target_nodes; ++i)
    {
      fac_xslm_b -= (*target_val)[i] * (target_mortar_nodes[i]->xspatial()[0]);
      fac_yslm_b -= (*target_val)[i] * (target_mortar_nodes[i]->xspatial()[1]);
    }

    csxib = -1 / (fac_dxsl_b * fac_ny_b - fac_dysl_b * fac_nx_b + fac_xslm_b * fac_dny_b -
                     fac_yslm_b * fac_dnx_b);
    // std::cout << "csxib: " << csxib << std::endl;

    Core::Gen::Pairedvector<int, double> dmap_sxib(num_target_nodes * ndof + linsize);

    // add derivative of target node coordinates
    for (int i = 0; i < num_target_nodes; ++i)
    {
      dmap_sxib[target_mortar_nodes[i]->dofs()[0]] -= (*target_val)[i] * fac_ny_b;
      dmap_sxib[target_mortar_nodes[i]->dofs()[1]] += (*target_val)[i] * fac_nx_b;
    }

    // add derivatives of source node coordinates
    for (int i = 0; i < num_source_nodes; ++i)
    {
      dmap_sxib[source_mortar_nodes[i]->dofs()[0]] += source_vals_xi_b[i] * fac_ny_b;
      dmap_sxib[source_mortar_nodes[i]->dofs()[1]] -= source_vals_xi_b[i] * fac_nx_b;
    }

    // add derivatives of source node normals
    for (int i = 0; i < num_source_nodes; ++i)
    {
      const Core::Gen::Pairedvector<int, double>& nxmap_curr =
          static_cast<const CONTACT::Node*>(source_mortar_nodes[i])->data().get_deriv_n()[0];
      const Core::Gen::Pairedvector<int, double>& nymap_curr =
          static_cast<const CONTACT::Node*>(source_mortar_nodes[i])->data().get_deriv_n()[1];

      for (CI p = nxmap_curr.begin(); p != nxmap_curr.end(); ++p)
        dmap_sxib[p->first] -= source_vals_xi_b[i] * fac_yslm_b * (p->second);
      for (CI p = nymap_curr.begin(); p != nymap_curr.end(); ++p)
        dmap_sxib[p->first] += source_vals_xi_b[i] * fac_xslm_b * (p->second);
    }

    // multiply all entries with csxib
    for (CI p = dmap_sxib.begin(); p != dmap_sxib.end(); ++p)
      dmap_sxib[p->first] = csxib * (p->second);

    // return map to DerivM() method
    derivxi[1] = dmap_sxib;
  }
}

/*----------------------------------------------------------------------*
 |  Compute directional derivative of XiGP target (2D)        popp 05/08|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::deriv_xi_gp_2d(const Mortar::Element& source_elem,
    const Mortar::Element& target_elem, double source_xi_gp, double target_xi_gp,
    const Core::Gen::Pairedvector<int, double>& source_derivs_xi,
    Core::Gen::Pairedvector<int, double>& target_derivs_xi, int linsize) const
{
  // check for problem dimension
  if (n_dim() != 2) FOUR_C_THROW("2D integration method called for non-2D problem");

  // we need the participating source and target nodes
  const Core::Nodes::Node* const* source_nodes = nullptr;
  const Core::Nodes::Node* const* target_nodes = nullptr;
  int num_source_nodes = source_elem.num_node();
  int num_target_nodes = target_elem.num_node();

  int ndof = n_dim();

  source_nodes = source_elem.nodes();
  target_nodes = target_elem.nodes();

  std::vector<const Mortar::Node*> source_mortar_nodes(num_source_nodes);
  std::vector<const Mortar::Node*> target_mortar_nodes(num_target_nodes);

  for (int i = 0; i < num_source_nodes; ++i)
  {
    source_mortar_nodes[i] = dynamic_cast<const Mortar::Node*>(source_nodes[i]);
    if (!source_mortar_nodes[i]) FOUR_C_THROW("Null pointer!");
  }

  for (int i = 0; i < num_target_nodes; ++i)
  {
    target_mortar_nodes[i] = dynamic_cast<const Mortar::Node*>(target_nodes[i]);
    if (!target_mortar_nodes[i]) FOUR_C_THROW("Null pointer!");
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

  // FIXME: This does not have to be the UNIT normal (see 3D)!
  // The reason for this is that we linearize the Gauss point
  // projection from source to target side here and this condition
  // only includes the Gauss point normal in a cross product.
  // When looking at Mortar::Projector::ProjectGaussPoint, one can see
  // that we do NOT use a unit normal there, either. Thus, why here?
  // First results suggest that it really makes no difference!

  // normalize interpolated GP normal back to length 1.0 !!!
  const double length = sqrt(sgpn[0] * sgpn[0] + sgpn[1] * sgpn[1] + sgpn[2] * sgpn[2]);
  if (length < 1.0e-12) FOUR_C_THROW("Divide by zero!");
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
    const Core::Gen::Pairedvector<int, double>& dmap_nxsl_i =
        static_cast<const CONTACT::Node*>(source_mortar_nodes[i])->data().get_deriv_n()[0];
    const Core::Gen::Pairedvector<int, double>& dmap_nysl_i =
        static_cast<const CONTACT::Node*>(source_mortar_nodes[i])->data().get_deriv_n()[1];

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
}

/*----------------------------------------------------------------------*
 |  Compute directional derivative of XiGP target (3D)        popp 02/09|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::deriv_xi_gp_3d(const Mortar::Element& source_elem,
    const Mortar::Element& target_elem, const double* source_xi_gp, const double* target_xi_gp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& source_derivs_xi,
    std::vector<Core::Gen::Pairedvector<int, double>>& target_derivs_xi, const double alpha) const
{
  // check for problem dimension
  if (n_dim() != 3) FOUR_C_THROW("3D integration method called for non-3D problem");

  // we need the participating source and target nodes
  const Core::Nodes::Node* const* source_nodes = source_elem.nodes();
  const Core::Nodes::Node* const* target_nodes = target_elem.nodes();
  std::vector<const Mortar::Node*> source_mortar_nodes(source_elem.num_node());
  std::vector<const Mortar::Node*> target_mortar_nodes(target_elem.num_node());
  const int num_source_nodes = source_elem.num_node();
  const int num_target_nodes = target_elem.num_node();

  for (int i = 0; i < num_source_nodes; ++i)
  {
    source_mortar_nodes[i] = dynamic_cast<const Mortar::Node*>(source_nodes[i]);
    if (!source_mortar_nodes[i]) FOUR_C_THROW("Null pointer!");
  }

  for (int i = 0; i < num_target_nodes; ++i)
  {
    target_mortar_nodes[i] = dynamic_cast<const Mortar::Node*>(target_nodes[i]);
    if (!target_mortar_nodes[i]) FOUR_C_THROW("Null pointer!");
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

  lmatrix.invert();

  // build directional derivative of source GP normal
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  int linsize = 0;
  for (int i = 0; i < num_source_nodes; ++i)
  {
    const Node* cnode = dynamic_cast<const Node*>(source_nodes[i]);
    linsize += cnode->get_linsize();
  }

  Core::Gen::Pairedvector<int, double> dmap_nxsl_gp(linsize);
  Core::Gen::Pairedvector<int, double> dmap_nysl_gp(linsize);
  Core::Gen::Pairedvector<int, double> dmap_nzsl_gp(linsize);

  for (int i = 0; i < num_source_nodes; ++i)
  {
    const Core::Gen::Pairedvector<int, double>& dmap_nxsl_i =
        static_cast<const CONTACT::Node*>(source_mortar_nodes[i])->data().get_deriv_n()[0];
    const Core::Gen::Pairedvector<int, double>& dmap_nysl_i =
        static_cast<const CONTACT::Node*>(source_mortar_nodes[i])->data().get_deriv_n()[1];
    const Core::Gen::Pairedvector<int, double>& dmap_nzsl_i =
        static_cast<const CONTACT::Node*>(source_mortar_nodes[i])->data().get_deriv_n()[2];

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
}

/*----------------------------------------------------------------------*
 |  Compute deriv. of XiGP source / target AuxPlane (3D)       popp 03/09|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::deriv_xi_gp_3d_aux_plane(const Mortar::Element& ele, const double* xigp,
    const double* auxn, std::vector<Core::Gen::Pairedvector<int, double>>& derivxi, double alpha,
    std::vector<Core::Gen::Pairedvector<int, double>>& derivauxn,
    Core::Gen::Pairedvector<int, Core::LinAlg::Matrix<3, 1>>& derivgp) const
{
  // check for problem dimension
  if (n_dim() != 3) FOUR_C_THROW("3D integration method called for non-3D problem");

  // we need the participating element nodes
  const Core::Nodes::Node* const* nodes = ele.nodes();
  std::vector<const Mortar::Node*> mrtrnodes(ele.num_node());
  const int numnode = ele.num_node();

  for (int i = 0; i < numnode; ++i)
  {
    mrtrnodes[i] = dynamic_cast<const Mortar::Node*>(nodes[i]);
    if (!mrtrnodes[i]) FOUR_C_THROW("Null pointer!");
  }

  // we also need shape function derivs at the GP
  Core::LinAlg::SerialDenseVector valxigp(numnode);
  Core::LinAlg::SerialDenseMatrix derivxigp(numnode, 2, true);
  ele.evaluate_shape(xigp, valxigp, derivxigp, numnode);

  // build 3x3 factor matrix L
  Core::LinAlg::Matrix<3, 3> lmatrix(Core::LinAlg::Initialization::zero);
  for (int k = 0; k < 3; ++k) lmatrix(k, 2) = -auxn[k];
  for (int z = 0; z < numnode; ++z)
    for (int k = 0; k < 3; ++k)
    {
      lmatrix(k, 0) += derivxigp(z, 0) * mrtrnodes[z]->xspatial()[k];
      lmatrix(k, 1) += derivxigp(z, 1) * mrtrnodes[z]->xspatial()[k];
    }

  // get inverse of the 3x3 matrix L (in place)
  lmatrix.invert();

  // start to fill linearization maps for element GP
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // see if this is an IntEle
  const Mortar::IntElement* ie = dynamic_cast<const Mortar::IntElement*>(&ele);

  // (1) all nodes coordinates part
  if (!ie)
    for (int z = 0; z < numnode; ++z)
      for (int k = 0; k < 3; ++k)
      {
        derivxi[0][mrtrnodes[z]->dofs()[k]] -= valxigp[z] * lmatrix(0, k);
        derivxi[1][mrtrnodes[z]->dofs()[k]] -= valxigp[z] * lmatrix(1, k);
      }
  else
  {
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> nodelin;
    ie->node_linearization(nodelin);
    for (int z = 0; z < numnode; ++z)
      for (int k = 0; k < 3; ++k)
        for (CI p = nodelin[z][k].begin(); p != nodelin[z][k].end(); ++p)
        {
          derivxi[0][p->first] -= valxigp[z] * lmatrix(0, k) * p->second;
          derivxi[1][p->first] -= valxigp[z] * lmatrix(1, k) * p->second;
        }
  }

  // (2) Gauss point coordinates part
  for (Core::Gen::Pairedvector<int, Core::LinAlg::Matrix<3, 1>>::const_iterator p = derivgp.begin();
      p != derivgp.end(); ++p)
  {
    const int pf = p->first;
    double& derivxi0 = derivxi[0][pf];
    double& derivxi1 = derivxi[1][pf];
    const Core::LinAlg::Matrix<3, 1>& tmp = p->second;
    for (int d = 0; d < 3; ++d)
    {
      derivxi0 += lmatrix(0, d) * tmp(d);
      derivxi1 += lmatrix(1, d) * tmp(d);
    }
  }

  // (3) AuxPlane normal part
  for (CI p = derivauxn[0].begin(); p != derivauxn[0].end(); ++p)
  {
    derivxi[0][p->first] += alpha * lmatrix(0, 0) * (p->second);
    derivxi[1][p->first] += alpha * lmatrix(1, 0) * (p->second);
  }
  for (CI p = derivauxn[1].begin(); p != derivauxn[1].end(); ++p)
  {
    derivxi[0][p->first] += alpha * lmatrix(0, 1) * (p->second);
    derivxi[1][p->first] += alpha * lmatrix(1, 1) * (p->second);
  }
  for (CI p = derivauxn[2].begin(); p != derivauxn[2].end(); ++p)
  {
    derivxi[0][p->first] += alpha * lmatrix(0, 2) * (p->second);
    derivxi[1][p->first] += alpha * lmatrix(1, 2) * (p->second);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_gp_3d(Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& lm_val, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& target_deriv,
    Core::LinAlg::SerialDenseMatrix& lm_deriv,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, double& wgt,
    double& jac, Core::Gen::Pairedvector<int, double>& derivjac, double* normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double& gap,
    Core::Gen::Pairedvector<int, double>& deriv_gap, double* source_xi, double* target_xi,
    std::vector<Core::Gen::Pairedvector<int, double>>& source_derivs_xi,
    std::vector<Core::Gen::Pairedvector<int, double>>& target_derivs_xi)
{
  // different algorithms integrate different sh*t
  switch (algo_)
  {
    case Mortar::algorithm_mortar:
    {
      // is quadratic case?
      bool quad = source_elem.is_quad();

      // weighted gap
      gp_3d_w_gap(source_elem, source_val, lm_val, &gap, jac, wgt, quad);
      for (int j = 0; j < source_elem.num_node(); ++j)
        gp_g_lin(j, source_elem, target_elem, source_val, target_val, lm_val, source_deriv,
            lm_deriv, gap, normal, jac, wgt, deriv_gap, derivjac, source_derivs_xi, dualmap);

      // check bound
      bool bound = false;
      for (int i = 0; i < source_elem.num_node(); ++i)
        if (dynamic_cast<CONTACT::Node*>(source_elem.nodes()[i])->is_on_boundor_ce()) bound = true;

      // integrate D and M matrix
      gp_dm(source_elem, target_elem, lm_val, source_val, target_val, jac, wgt, bound);

      // compute segment D/M linearization
      if (bound)
      {
        gp_3d_dm_lin_bound(source_elem, target_elem, source_val, target_val, lm_val, source_deriv,
            lm_deriv, target_deriv, jac, wgt, derivjac, source_derivs_xi, target_derivs_xi,
            dualmap);
      }
      else
      {
        gp_3d_dm_lin(source_elem, target_elem, source_val, target_val, lm_val, source_deriv,
            target_deriv, lm_deriv, wgt, jac, source_derivs_xi, target_derivs_xi, derivjac,
            dualmap);
      }

      // Creating the WEIGHTED tangential relative slip increment (non-objective)
      if (gpslip_)
      {
        int linsize = 0;
        for (int i = 0; i < source_elem.num_node(); ++i)
          linsize += dynamic_cast<Node*>(source_elem.nodes()[i])->get_linsize();

        double jumpvalv[2] = {0.0, 0.0};  // jump for slipincr --> equal to jumpval
        std::vector<Core::Gen::Pairedvector<int, double>> dslipgp(
            2, ((target_elem.num_node() * n_dim()) +
                   linsize));  // deriv. of slip for slipincr (xi, eta)

        // Creating the WEIGHTED tangential relative slip increment (non-objective)
        gp_3d_slip_incr(source_elem, target_elem, source_val, target_val, lm_val, source_deriv,
            target_deriv, jac, wgt, jumpvalv, source_derivs_xi, target_derivs_xi, dslipgp);
        // Lin weighted slip
        for (int iter = 0; iter < source_elem.num_node(); ++iter)
          gp_3d_slip_incr_lin(iter, source_elem, source_val, lm_val, source_deriv, lm_deriv, jac,
              wgt, jumpvalv, derivjac, dslipgp, source_derivs_xi, dualmap);
      }

      //*******************************
      // WEAR stuff
      //*******************************
      // std. wear for all wear-algorithm types
      if (wearlaw_ != Wear::wear_none)
      {
        int linsize = 0;
        for (int i = 0; i < source_elem.num_node(); ++i)
          linsize += dynamic_cast<Node*>(source_elem.nodes()[i])->get_linsize();

        double jumpval[2] = {0.0, 0.0};  // jump for wear
        double wearval = 0.0;            // wear value
        Core::Gen::Pairedvector<int, double> dsliptmatrixgp(
            (target_elem.num_node() * n_dim()) + linsize);  // deriv. of slip for wear
        Core::Gen::Pairedvector<int, double> dweargp(
            (target_elem.num_node() * n_dim()) + linsize);  // wear lin. without lm and jac.
        Core::LinAlg::SerialDenseMatrix lagmult(3, source_elem.num_node());
        source_elem.get_nodal_lag_mult(lagmult);

        gp_3d_wear(source_elem, target_elem, source_val, source_deriv, target_val, target_deriv,
            lm_val, lm_deriv, lagmult, normal, jac, wgt, jumpval, &wearval, dsliptmatrixgp, dweargp,
            source_derivs_xi, target_derivs_xi, dnmap_unit, dualmap);

        if (wear_type() == Wear::wear_intstate and wearimpl_ == true)
          for (int j = 0; j < source_elem.num_node(); ++j)
            gp_3d_wear_lin(j, source_elem, source_val, lm_val, source_deriv, lm_deriv, jac, normal,
                wgt, wearval, jumpval, dweargp, derivjac, source_derivs_xi, dualmap);

        // integrate T and E matrix for discr. wear
        if (wear_type() == Wear::wear_primvar)
          gp_te(source_elem, lm_val, source_val, jac, wgt, jumpval);

        // both-sided discr wear specific stuff
        if (wear_side() == Wear::wear_both and wear_type() == Wear::wear_primvar)
        {
          Core::LinAlg::SerialDenseVector lm2val(target_elem.num_node());
          Core::LinAlg::SerialDenseMatrix lm2deriv(target_elem.num_node(), n_dim() - 1, true);
          target_elem.evaluate_shape_lag_mult(
              shape_fcn(), target_xi, lm2val, lm2deriv, target_elem.num_node());

          gp_te_target(
              source_elem, target_elem, lm_val, lm2val, target_val, jac, wgt, jumpval, Comm_);
        }

        // both-sided wear specific stuff
        if (wear_side() == Wear::wear_both and wear_type() == Wear::wear_intstate)
        {
          Core::LinAlg::SerialDenseVector lm2val(target_elem.num_node());
          Core::LinAlg::SerialDenseMatrix lm2deriv(target_elem.num_node(), 2, true);
          target_elem.evaluate_shape_lag_mult(
              shape_fcn(), target_xi, lm2val, lm2deriv, target_elem.num_node());

          gp_d2(source_elem, target_elem, lm2val, target_val, jac, wgt, Comm_);
        }

        // Lin wear matrices T and E for discr. wear
        if (wearimpl_ == true and wear_type() == Wear::wear_primvar)
          for (int j = 0; j < source_elem.num_node(); ++j)
            gp_3d_te_lin(j, source_elem, source_val, lm_val, source_deriv, lm_deriv, jac, wgt,
                jumpval, source_derivs_xi, derivjac, dsliptmatrixgp, dualmap);



        if (wear_side() == Wear::wear_both and wear_type() == Wear::wear_primvar and
            wearimpl_ == true)
        {
          Core::LinAlg::SerialDenseVector lm2val(target_elem.num_node());
          Core::LinAlg::SerialDenseMatrix lm2deriv(target_elem.num_node(), n_dim() - 1, true);
          Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> dual2map(
              (source_elem.num_node() + target_elem.num_node()) * n_dim(), 0,
              Core::LinAlg::SerialDenseMatrix(target_elem.num_node(), target_elem.num_node()));
          target_elem.evaluate_shape_lag_mult(
              shape_fcn(), target_xi, lm2val, lm2deriv, target_elem.num_node());
          target_elem.deriv_shape_dual(dual2map);

          for (int iter = 0; iter < target_elem.num_node(); ++iter)
            gp_3d_te_target_lin(iter, source_elem, target_elem, source_val, target_val, lm_val,
                lm2val, source_deriv, target_deriv, lm_deriv, lm2deriv, jac, wgt, jumpval,
                source_derivs_xi, target_derivs_xi, derivjac, dsliptmatrixgp, dualmap, dual2map,
                Comm_);
        }
      }

      //*******************************
      // PORO stuff
      //*******************************
      bool poroprob = false;
      if (imortar_.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroelast ||
          imortar_.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroscatra)
        if (imortar_.get<bool>("CONTACT_NO_PENETRATION"))  // evaluate additional terms just in case
                                                           // of no penectration condition
          poroprob = true;
      if (poroprob)
      {
        double ncoup = 0.0;
        std::map<int, double> dncoupgp;      // ncoup lin. without lm and jac.
        std::map<int, double> dvelncoupgp;   // velocity ncoup lin. without lm and jac.
        std::map<int, double> dpresncoupgp;  // pressure ncoup lin. without lm and jac.

        gp_ncoup_deriv(source_elem, target_elem, source_val, target_val, lm_val, source_deriv,
            target_deriv, &ncoup, normal, jac, wgt, source_xi, source_derivs_xi, target_derivs_xi,
            dncoupgp, dvelncoupgp, dpresncoupgp, dnmap_unit, false);
        // Lin ncoup condition
        for (int j = 0; j < source_elem.num_node(); ++j)
          gp_ncoup_lin(j, source_elem, target_elem, source_val, target_val, lm_val, source_deriv,
              lm_deriv, ncoup, normal, jac, wgt, dncoupgp, dvelncoupgp, dpresncoupgp, derivjac,
              source_derivs_xi, target_derivs_xi, dualmap);
      }
      break;
    }

    default:
      FOUR_C_THROW("contact integrator doesn't know what to do with this algorithm");
  };
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::integrate_gp_2d(Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& lm_val, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& target_deriv,
    Core::LinAlg::SerialDenseMatrix& lm_deriv,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, double& wgt,
    double& jac, Core::Gen::Pairedvector<int, double>& derivjac, double* normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double& gap,
    Core::Gen::Pairedvector<int, double>& deriv_gap, double* source_xi, double* target_xi,
    std::vector<Core::Gen::Pairedvector<int, double>>& source_derivs_xi,
    std::vector<Core::Gen::Pairedvector<int, double>>& target_derivs_xi)
{
  // different algorithms integrate different sh*t
  switch (algo_)
  {
    case Mortar::algorithm_mortar:
    {
      // decide whether boundary modification has to be considered or not
      // this is element-specific (is there a boundary node in this element?)
      bool bound = false;
      for (int k = 0; k < source_elem.num_node(); ++k)
      {
        Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_elem.nodes()[k]);
        if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

        if (mymrtrnode->is_on_boundor_ce())
        {
          bound = true;
          break;
        }
      }

      // decide whether linear LM are used for quadratic FE here
      bool linlm = false;
      if (lag_mult_quad() == Mortar::lagmult_lin &&
          source_elem.shape() == Core::FE::CellType::line3)
      {
        bound = false;  // crosspoints and linear LM NOT at the same time!!!!
        linlm = true;
      }

      // weighted gap
      gp_2d_w_gap(source_elem, source_val, lm_val, &gap, jac, wgt);
      for (int j = 0; j < source_val.numRows(); ++j)
        gp_2d_g_lin(j, source_elem, target_elem, source_val, target_val, lm_val, source_deriv,
            lm_deriv, gap, normal, jac, wgt, deriv_gap, derivjac, source_derivs_xi, dualmap);

      // integrate D and M matrix
      gp_dm(source_elem, target_elem, lm_val, source_val, target_val, jac, wgt, bound);

      // compute segment D/M linearization  -- bound
      if (bound)
      {
        gp_2d_dm_lin_bound(source_elem, target_elem, source_val, target_val, lm_val, source_deriv,
            target_deriv, lm_deriv, jac, wgt, derivjac, source_derivs_xi, target_derivs_xi,
            dualmap);
      }
      else
      {
        // D/M linearization
        for (int j = 0; j < source_val.numRows(); ++j)
        {
          gp_2d_dm_lin(j, bound, linlm, source_elem, target_elem, source_val, target_val, lm_val,
              source_deriv, target_deriv, lm_deriv, jac, wgt, source_derivs_xi, target_derivs_xi,
              derivjac, dualmap);
        }
      }

      // Creating the tangential relative slip increment (non-objective)
      if (imortar_.get<bool>("GP_SLIP_INCR"))
      {
        int linsize = 0;
        for (int i = 0; i < source_elem.num_node(); ++i)
          linsize += dynamic_cast<Node*>(source_elem.nodes()[i])->get_linsize();
        linsize = linsize * 2;

        double jumpvalv = 0.0;  // jump for slipincr --> equal to jumpval
        Core::Gen::Pairedvector<int, double> dslipgp(
            linsize + n_dim() * target_elem.num_node());  // deriv. of slip for slipincr

        gp_2d_slip_incr(source_elem, target_elem, source_val, target_val, lm_val, source_deriv,
            target_deriv, jac, wgt, &jumpvalv, source_derivs_xi, target_derivs_xi, dslipgp,
            linsize);

        for (int iter = 0; iter < source_elem.num_node(); ++iter)
          gp_2d_slip_incr_lin(iter, source_elem, source_val, lm_val, source_deriv, lm_deriv, jac,
              wgt, &jumpvalv, source_derivs_xi, dslipgp, derivjac, dualmap);
      }

      // wear specific stuff
      if (wearlaw_ != Wear::wear_none)
      {
        // nodal Lagrange multiplier
        Core::LinAlg::SerialDenseMatrix lagmult(3, source_elem.num_node());
        source_elem.get_nodal_lag_mult(lagmult);

        int linsize = 0;
        for (int i = 0; i < source_elem.num_node(); ++i)
          linsize += dynamic_cast<Node*>(source_elem.nodes()[i])->get_linsize();
        linsize = linsize * 2;

        double jumpval = 0.0;  // jump for wear
        double wearval = 0.0;  // wear value
        Core::Gen::Pairedvector<int, double> dsliptmatrixgp(
            linsize + n_dim() * target_elem.num_node());  // deriv. of slip for wear
        Core::Gen::Pairedvector<int, double> dweargp(
            linsize + n_dim() * target_elem.num_node());  // wear lin without weighting and jac

        // std. wear for all wear-algorithm types
        gp_2d_wear(source_elem, target_elem, source_val, source_deriv, target_val, target_deriv,
            lm_val, lm_deriv, lagmult, normal, jac, wgt, &jumpval, &wearval, dsliptmatrixgp,
            dweargp, source_derivs_xi, target_derivs_xi, dnmap_unit, dualmap);

        // integrate T and E matrix for discr. wear
        if (wear_type() == Wear::wear_primvar)
          gp_te(source_elem, lm_val, source_val, jac, wgt, &jumpval);

        // both-sided discr wear specific stuff
        if (wear_side() == Wear::wear_both and wear_type() == Wear::wear_primvar)
        {
          Core::LinAlg::SerialDenseVector lm2val(target_elem.num_node());
          Core::LinAlg::SerialDenseMatrix lm2deriv(target_elem.num_node(), n_dim() - 1, true);
          target_elem.evaluate_shape_lag_mult(
              shape_fcn(), target_xi, lm2val, lm2deriv, target_elem.num_node());

          gp_te_target(
              source_elem, target_elem, lm_val, lm2val, target_val, jac, wgt, &jumpval, Comm_);
        }

        // both-sided map wear specific stuff
        if (wear_side() == Wear::wear_both and wear_type() == Wear::wear_intstate)
        {
          Core::LinAlg::SerialDenseVector lm2val(target_elem.num_node());
          Core::LinAlg::SerialDenseMatrix lm2deriv(target_elem.num_node(), 2, true);
          target_elem.evaluate_shape_lag_mult(
              shape_fcn(), target_xi, lm2val, lm2deriv, target_elem.num_node());

          gp_d2(source_elem, target_elem, lm2val, target_val, jac, wgt, Comm_);
        }

        // Lin wear for impl. alg.
        if (wearimpl_ == true and wear_type() == Wear::wear_intstate)
          for (int j = 0; j < source_elem.num_node(); ++j)
            gp_2d_wear_lin(j, source_elem, source_val, lm_val, source_deriv, lm_deriv, jac, normal,
                wgt, wearval, &jumpval, dweargp, derivjac, source_derivs_xi, dualmap);

        // Lin wear T and E matrix
        if (wearimpl_ == true and wear_type() == Wear::wear_primvar)
          for (int j = 0; j < source_elem.num_node(); ++j)
            gp_2d_te_lin(j, source_elem, source_val, lm_val, source_deriv, lm_deriv, jac, wgt,
                &jumpval, source_derivs_xi, derivjac, dsliptmatrixgp, dualmap);

        if (wear_side() == Wear::wear_both and wear_type() == Wear::wear_primvar and
            wearimpl_ == true)
        {
          Core::LinAlg::SerialDenseVector lm2val(target_elem.num_node());
          Core::LinAlg::SerialDenseMatrix lm2deriv(target_elem.num_node(), n_dim() - 1, true);
          Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> dual2map(
              (source_elem.num_node() + target_elem.num_node()) * n_dim(), 0,
              Core::LinAlg::SerialDenseMatrix(target_elem.num_node(), target_elem.num_node()));
          target_elem.evaluate_shape_lag_mult(
              shape_fcn(), target_xi, lm2val, lm2deriv, target_elem.num_node());
          target_elem.deriv_shape_dual(dual2map);

          for (int iter = 0; iter < target_elem.num_node(); ++iter)
            gp_3d_te_target_lin(iter, source_elem, target_elem, source_val, target_val, lm_val,
                lm2val, source_deriv, target_deriv, lm_deriv, lm2deriv, jac, wgt, &jumpval,
                source_derivs_xi, target_derivs_xi, derivjac, dsliptmatrixgp, dualmap, dual2map,
                Comm_);
        }
      }  // if wear

      //*******************************
      // PORO stuff
      //*******************************
      bool poroprob = false;
      if (imortar_.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroelast ||
          imortar_.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroscatra)
        if (imortar_.get<bool>("CONTACT_NO_PENETRATION"))  // evaluate additional terms just in case
                                                           // of no penectration condition
          poroprob = true;
      if (poroprob)
      {
        double ncoup = 0.0;
        std::map<int, double> dncoupgp;      // ncoup lin. without lm and jac.
        std::map<int, double> dvelncoupgp;   // velocity ncoup lin. without lm and jac.
        std::map<int, double> dpresncoupgp;  // pressure ncoup lin. without lm and jac.

        gp_ncoup_deriv(source_elem, target_elem, source_val, target_val, lm_val, source_deriv,
            target_deriv, &ncoup, normal, jac, wgt, source_xi, source_derivs_xi, target_derivs_xi,
            dncoupgp, dvelncoupgp, dpresncoupgp, dnmap_unit, false);
        // Lin ncoup condition
        for (int j = 0; j < source_elem.num_node(); ++j)
          gp_ncoup_lin(j, source_elem, target_elem, source_val, target_val, lm_val, source_deriv,
              lm_deriv, ncoup, normal, jac, wgt, dncoupgp, dvelncoupgp, dpresncoupgp, derivjac,
              source_derivs_xi, target_derivs_xi, dualmap);
      }

      break;
    }
    default:
      FOUR_C_THROW("contact integrator doesn't know what to do with this algorithm");
  };
}


/*----------------------------------------------------------------------*
 |  Compute entries for D and M matrix at GP                 farah 09/13|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::gp_dm(Mortar::Element& source_elem, Mortar::Element& target_elem,
    Core::LinAlg::SerialDenseVector& lm_val, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& target_val, double& jac, double& wgt, bool& bound)
{
  int nrow;
  int ncol;
  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = nullptr;
  Core::Nodes::Node** target_nodes = nullptr;
  nrow = source_elem.num_node();
  source_nodes = source_elem.nodes();
  ncol = target_elem.num_node();
  target_nodes = target_elem.nodes();

  // BOUNDARY NODE MODIFICATION **********************************
  // We have modified their neighbors' dual shape functions, so we
  // now have a problem with off-diagonal entries occurring in D.
  // Of course we want to keep the diagonality property of the D
  // matrix, but still we may not modify the whole Mortar coupling
  // setting! We achieve both by applying a quite simple but very
  // effective trick: The boundary nodes have already been defined
  // as being target nodes, so all we have to do here, is to shift
  // the off-diagonal terms from D to the respective place in M,
  // which is not diagonal anyway! (Mind the MINUS sign!!!)
  // *************************************************************

  // compute segment D/M matrix ****************************************
  // dual shape functions without locally linear Lagrange multipliers
  if ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
      lag_mult_quad() != Mortar::lagmult_lin)
  {
    for (int j = 0; j < nrow; ++j)
    {
      CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(source_nodes[j]);

      if (cnode->is_on_boundor_ce()) continue;

      // integrate mseg
      for (int k = 0; k < ncol; ++k)
      {
        const CONTACT::Node* target_node = dynamic_cast<const CONTACT::Node*>(target_nodes[k]);

        // multiply the two shape functions
        double prod = lm_val[j] * target_val[k] * jac * wgt;
        if (abs(prod) > MORTARINTTOL)
        {
          cnode->add_m_value(target_node->id(), prod);
          cnode->add_target_node(target_node->id());  // only for friction!
          if (!bound)
          {
            cnode->add_d_value(cnode->id(), prod);
            cnode->add_source_node(cnode->id());  // only for friction!
          }
        }
      }

      // integrate dseg (boundary modification)
      if (bound)
      {
        //        bool j_boundnode = cnode->IsOnBoundorCE();

        for (int k = 0; k < nrow; ++k)
        {
          const CONTACT::Node* target_node = dynamic_cast<const CONTACT::Node*>(source_nodes[k]);

          //          bool k_boundnode = target_node->IsOnBoundorCE();

          // do not assemble off-diagonal terms if j,k are both non-boundary nodes
          //          if (!j_boundnode && !k_boundnode && (j!=k))
          //            continue;

          // multiply the two shape functions
          double prod = lm_val[j] * source_val[k] * jac * wgt;
          if (abs(prod) > MORTARINTTOL)
          {
            // for the "bound" flag, the node on the source side does not carry a LM value
            // therefore, we assemble the negative value to the M-matrix (which contains
            // all nodes, that do not have a LM).
            if (target_node->is_on_bound())
            {
              cnode->add_m_value(target_node->id(), -prod);
              cnode->add_target_node(target_node->id());  // only for friction!
            }
            // in the non-smooth case, the nodes at corners/edges will have a LM value,
            // however the corresponding integration is not performed here (here we only
            // do the surface integration). Since the nodes will have a LM value, the corresponding
            // integrals should be assembled in the D-matrix.
            else if (target_node->is_on_boundor_ce())
            {
              // isolate the dseg entries to be filled
              // (both the main diagonal and every other secondary diagonal)
              // and add current Gauss point's contribution to dseg
              // NONSMOOTH MODIFICATION
              cnode->add_d_value(target_node->id(), prod);
              cnode->add_source_node(target_node->id());  // only for friction!
            }
            else
            {
              cnode->add_d_value(cnode->id(), prod);
              cnode->add_source_node(cnode->id());  // only for friction!
            }
          }
        }
      }
    }
  }
  // standard shape functions or dual shape functions with locally linear Lagrange multipliers
  else
  {
    for (int j = 0; j < nrow; ++j)
    {
      CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(source_nodes[j]);

      if ((shapefcn_ == Mortar::shape_standard && cnode->is_on_boundor_ce()) ||
          ((shapefcn_ == Mortar::shape_dual || shapefcn_ == Mortar::shape_petrovgalerkin) &&
              cnode->is_on_bound()))
        continue;

      // integrate mseg
      for (int k = 0; k < ncol; ++k)
      {
        CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_nodes[k]);

        // multiply the two shape functions
        double prod = lm_val[j] * target_val[k] * jac * wgt;
        if (abs(prod) > MORTARINTTOL)
        {
          cnode->add_m_value(target_node->id(), prod);
          cnode->add_target_node(target_node->id());  // only for friction!
        }
      }

      // integrate dseg
      for (int k = 0; k < nrow; ++k)
      {
        CONTACT::Node* source_node = dynamic_cast<CONTACT::Node*>(source_nodes[k]);

        // multiply the two shape functions
        double prod = lm_val[j] * source_val[k] * jac * wgt;
        if (abs(prod) > MORTARINTTOL)
        {
          if ((shape_fcn() == Mortar::shape_standard && source_node->is_on_boundor_ce()) ||
              ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
                  source_node->is_on_bound()))
          {
            if (shape_fcn() == Mortar::shape_standard && nonsmooth_)
            {
              if (source_node->is_on_bound() and !source_node->is_on_corner_edge())
              {
                cnode->add_m_value(source_node->id(), -prod);
                cnode->add_target_node(source_node->id());  // only for friction!
              }
              else
              {
                cnode->add_d_value(source_node->id(), prod);
                cnode->add_source_node(source_node->id());  // only for friction!
              }
            }
            // this still needs to be checked for dual shape functions with locally linear Lagrange
            // multipliers
            else if ((shape_fcn() == Mortar::shape_dual ||
                         shape_fcn() == Mortar::shape_petrovgalerkin) &&
                     lag_mult_quad() == Mortar::lagmult_lin && nonsmooth_)
              FOUR_C_THROW(
                  "dual shape functions with locally linear Lagrange multipliers in "
                  "combination non-smooth approach still needs to be checked!");
            else
            {
              cnode->add_m_value(source_node->id(), -prod);
              cnode->add_target_node(source_node->id());  // only for friction!
            }
          }
          else
          {
            cnode->add_d_value(source_node->id(), prod);
            cnode->add_source_node(source_node->id());  // only for friction!
          }
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Compute entries for D and M matrix at GP (3D Quad)       farah 12/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_3d_dm_quad(Mortar::Element& source_elem,
    Mortar::Element& target_elem, Mortar::IntElement& sintele,
    Core::LinAlg::SerialDenseVector& lm_val, Core::LinAlg::SerialDenseVector& lmintval,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& target_val,
    const double& jac, double& wgt, const int& nrow, const int& nintrow, const int& ncol,
    const int& ndof, bool& bound)
{
  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  if (!source_nodes) FOUR_C_THROW("Null pointer!");
  Core::Nodes::Node** target_nodes = target_elem.nodes();
  if (!target_nodes) FOUR_C_THROW("Null pointer!");
  Core::Nodes::Node** sintnodes = sintele.nodes();
  if (!sintnodes) FOUR_C_THROW("Null pointer for sintnodes!");

  // CASE 1/2: Standard LM shape functions and quadratic, linear or constant interpolation
  // CASE 5: dual LM shape functions and linear interpolation
  if ((shape_fcn() == Mortar::shape_standard &&
          (lag_mult_quad() == Mortar::lagmult_quad || lag_mult_quad() == Mortar::lagmult_lin ||
              lag_mult_quad() == Mortar::lagmult_const)) ||
      ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
          lag_mult_quad() == Mortar::lagmult_lin))
  {
    // compute all mseg and dseg matrix entries
    // loop over Lagrange multiplier dofs j
    for (int j = 0; j < nrow; ++j)
    {
      Node* cnode = dynamic_cast<Node*>(source_nodes[j]);

      // integrate mseg
      for (int k = 0; k < ncol; ++k)
      {
        Node* target_node = dynamic_cast<Node*>(target_nodes[k]);

        // multiply the two shape functions
        double prod = lm_val[j] * target_val[k] * jac * wgt;
        if (abs(prod) > MORTARINTTOL)
        {
          cnode->add_m_value(target_node->id(), prod);
          cnode->add_target_node(target_node->id());
        }
      }

      // integrate dseg
      for (int k = 0; k < nrow; ++k)
      {
        Node* source_node = dynamic_cast<Node*>(source_nodes[k]);

        // multiply the two shape functions
        double prod = lm_val[j] * source_val[k] * jac * wgt;
        if (abs(prod) > MORTARINTTOL)
        {
          if (source_node->is_on_bound())
          {
            cnode->add_m_value(source_node->id(), -prod);
            cnode->add_target_node(source_node->id());
          }
          else
          {
            cnode->add_d_value(source_node->id(), prod);
            cnode->add_source_node(source_node->id());
          }
        }
      }
    }
  }
  // CASE 3: Standard LM shape functions and piecewise linear interpolation
  else if (shape_fcn() == Mortar::shape_standard && lag_mult_quad() == Mortar::lagmult_pwlin)
  {
    // compute all mseg and dseg matrix entries
    // loop over Lagrange multiplier dofs j
    for (int j = 0; j < nintrow; ++j)
    {
      Node* cnode = dynamic_cast<Node*>(sintnodes[j]);

      // integrate mseg
      for (int k = 0; k < ncol; ++k)
      {
        Node* target_node = dynamic_cast<Node*>(target_nodes[k]);

        // multiply the two shape functions
        double prod = lmintval[j] * target_val[k] * jac * wgt;
        if (abs(prod) > MORTARINTTOL)
        {
          cnode->add_m_value(target_node->id(), prod);
          cnode->add_target_node(target_node->id());
        }
      }

      // integrate dseg
      for (int k = 0; k < nrow; ++k)
      {
        Node* source_node = dynamic_cast<Node*>(source_nodes[k]);

        // multiply the two shape functions
        double prod = lmintval[j] * source_val[k] * jac * wgt;
        if (abs(prod) > MORTARINTTOL)
        {
          if (source_node->is_on_bound())
          {
            cnode->add_m_value(source_node->id(), -prod);
            cnode->add_target_node(source_node->id());
          }
          else
          {
            cnode->add_d_value(source_node->id(), prod);
            cnode->add_source_node(source_node->id());
          }
        }
      }
    }
  }

  // CASE 4: Dual LM shape functions and quadratic or constant interpolation
  else if ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
           (lag_mult_quad() == Mortar::lagmult_quad || lag_mult_quad() == Mortar::lagmult_const))
  {
    // compute all mseg and dseg matrix entries
    // loop over Lagrange multiplier dofs j
    for (int j = 0; j < nrow; ++j)
    {
      Node* cnode = dynamic_cast<Node*>(source_nodes[j]);

      // integrate mseg
      for (int k = 0; k < ncol; ++k)
      {
        Node* target_node = dynamic_cast<Node*>(target_nodes[k]);

        // multiply the two shape functions
        double prod = lm_val[j] * target_val[k] * jac * wgt;
        if (abs(prod) > MORTARINTTOL)
        {
          cnode->add_d_value(cnode->id(), prod);
          cnode->add_source_node(cnode->id());
          cnode->add_m_value(target_node->id(), prod);
          cnode->add_target_node(target_node->id());
        }
      }
    }
  }
  // INVALID CASES
  else
  {
    FOUR_C_THROW("Invalid integration case for 3D quadratic mortar!");
  }
}


/*----------------------------------------------------------------------*
 |  Compute entries for weighted Gap at GP                   farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_2d_w_gap(Mortar::Element& source_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& lm_val,
    double* gap, double& jac, double& wgt)
{
  Core::Nodes::Node** mynodes = nullptr;
  int nrow = 0;
  nrow = source_elem.num_node();
  mynodes = source_elem.nodes();

  // **************************
  // add to node
  // **************************
  for (int j = 0; j < nrow; ++j)
  {
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(mynodes[j]);

    double prod = 0.0;
    // Petrov-Galerkin approach (dual LM for D/M but standard LM for gap)
    if (shape_fcn() == Mortar::shape_petrovgalerkin) prod = source_val[j] * gap[0] * jac * wgt;
    // usual standard or dual LM approach
    else
      prod = lm_val[j] * gap[0] * jac * wgt;


    // do not process source side boundary nodes
    // (their row entries would be zero anyway!)
    if (cnode->is_on_boundor_ce()) continue;

    // add current Gauss point's contribution to gseg
    cnode->addg_value(prod);
  }
}

/*----------------------------------------------------------------------*
 |  Compute entries for weighted Gap at GP                   farah 09/13|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::gp_3d_w_gap(Mortar::Element& source_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& lm_val,
    double* gap, double& jac, double& wgt, bool quadratic, int nintrow)
{
  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  if (!source_nodes) FOUR_C_THROW("Null pointer!");

  // number of nodes (source, target)
  const int nrow = source_elem.num_node();

  // **************************
  // add to node
  // **************************
  if (!quadratic)
  {
    for (int j = 0; j < nrow; ++j)
    {
      CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(source_nodes[j]);

      double prod = 0.0;
      // Petrov-Galerkin approach (dual LM for D/M but standard LM for gap)
      if (shape_fcn() == Mortar::shape_petrovgalerkin) prod = source_val[j] * gap[0] * jac * wgt;
      // usual standard or dual LM approach
      else
        prod = lm_val[j] * gap[0] * jac * wgt;

      // do not process source side boundary nodes
      // (their row entries would be zero anyway!)
      if (cnode->is_on_boundor_ce()) continue;

      // add current Gauss point's contribution to gseg
      cnode->addg_value(prod);
    }
  }
  else
  {
    // compute cell gap vector *******************************************
    // CASE 1/2: Standard LM shape functions and quadratic or linear interpolation
    if (shape_fcn() == Mortar::shape_standard &&
        (lag_mult_quad() == Mortar::lagmult_quad || lag_mult_quad() == Mortar::lagmult_lin ||
            lag_mult_quad() == Mortar::lagmult_const))
    {
      for (int j = 0; j < nrow; ++j)
      {
        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(source_nodes[j]);

        double prod = 0.0;
        prod = lm_val[j] * gap[0] * jac * wgt;

        if (cnode->is_on_boundor_ce()) continue;

        // add current Gauss point's contribution to gseg
        cnode->addg_value(prod);
      }
    }

    // CASE 3: Standard LM shape functions and piecewise linear interpolation
    // Attention:  for this case, lm_val represents lmintval !!!
    else if (shape_fcn() == Mortar::shape_standard && lag_mult_quad() == Mortar::lagmult_pwlin)
    {
      if (nintrow == 0) FOUR_C_THROW("ERROR!");

      for (int j = 0; j < nintrow; ++j)
      {
        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(source_nodes[j]);

        double prod = 0.0;
        prod = lm_val[j] * gap[0] * jac * wgt;

        if (cnode->is_on_bound()) continue;

        // add current Gauss point's contribution to gseg
        cnode->addg_value(prod);
      }
    }

    // CASE 4: Dual LM shape functions and quadratic, linear or constant interpolation
    else if ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
             (lag_mult_quad() == Mortar::lagmult_quad || lag_mult_quad() == Mortar::lagmult_lin ||
                 lag_mult_quad() == Mortar::lagmult_const))
    {
      for (int j = 0; j < nrow; ++j)
      {
        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(source_nodes[j]);

        if (cnode->is_on_bound()) continue;

        double prod = 0.0;
        // Petrov-Galerkin approach (dual LM for D/M but standard LM for gap)
        if (shape_fcn() == Mortar::shape_petrovgalerkin) prod = source_val[j] * gap[0] * jac * wgt;
        // usual standard or dual LM approach
        else
          prod = lm_val[j] * gap[0] * jac * wgt;

        // add current Gauss point's contribution to gseg
        cnode->addg_value(prod);
      }
    }

    // INVALID CASES
    else
    {
      FOUR_C_THROW("Invalid integration case for 3D quadratic contact!");
    }
  }
}

void CONTACT::Integrator::gap_3d(Mortar::Element& source_elem, Mortar::Element& target_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& target_deriv,
    double* gap, double* gpn, std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_gp,
    Core::Gen::Pairedvector<int, double>& dgapgp,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit)
{
  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  Core::Nodes::Node** target_nodes = target_elem.nodes();
  if (!source_nodes) FOUR_C_THROW("Null pointer!");
  if (!target_nodes) FOUR_C_THROW("Null pointer!");

  // number of nodes (source, target)
  const int nrow = source_elem.num_node();
  const int ncol = target_elem.num_node();

  std::array<double, 3> sgpx = {0.0, 0.0, 0.0};
  std::array<double, 3> mgpx = {0.0, 0.0, 0.0};

  for (int i = 0; i < nrow; ++i)
  {
    Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[i]);
    gpn[0] += source_val[i] * mymrtrnode->mo_data().n()[0];
    gpn[1] += source_val[i] * mymrtrnode->mo_data().n()[1];
    gpn[2] += source_val[i] * mymrtrnode->mo_data().n()[2];

    if (wear_type() == Wear::wear_primvar)
    {
      FriNode* myfricnode = dynamic_cast<FriNode*>(mymrtrnode);
      sgpx[0] +=
          source_val[i] * (source_elem.get_nodal_coords(0, i) -
                              (myfricnode->mo_data().n()[0]) * myfricnode->wear_data().wcurr()[0]);
      sgpx[1] +=
          source_val[i] * (source_elem.get_nodal_coords(1, i) -
                              (myfricnode->mo_data().n()[1]) * myfricnode->wear_data().wcurr()[0]);
      sgpx[2] +=
          source_val[i] * (source_elem.get_nodal_coords(2, i) -
                              (myfricnode->mo_data().n()[2]) * myfricnode->wear_data().wcurr()[0]);
    }
    else
    {
      sgpx[0] += source_val[i] * source_elem.get_nodal_coords(0, i);
      sgpx[1] += source_val[i] * source_elem.get_nodal_coords(1, i);
      sgpx[2] += source_val[i] * source_elem.get_nodal_coords(2, i);
    }
  }

  // build interpolation of target GP coordinates
  for (int i = 0; i < ncol; ++i)
  {
    if (wear_side() == Wear::wear_both and wear_type() == Wear::wear_primvar)
    {
      FriNode* targetnode = dynamic_cast<FriNode*>(target_nodes[i]);

      mgpx[0] +=
          target_val[i] * (target_elem.get_nodal_coords(0, i) -
                              (targetnode->mo_data().n()[0] * targetnode->wear_data().wcurr()[0]));
      mgpx[1] +=
          target_val[i] * (target_elem.get_nodal_coords(1, i) -
                              (targetnode->mo_data().n()[1] * targetnode->wear_data().wcurr()[0]));
      mgpx[2] +=
          target_val[i] * (target_elem.get_nodal_coords(2, i) -
                              (targetnode->mo_data().n()[2] * targetnode->wear_data().wcurr()[0]));
    }
    else
    {
      mgpx[0] += target_val[i] * target_elem.get_nodal_coords(0, i);
      mgpx[1] += target_val[i] * target_elem.get_nodal_coords(1, i);
      mgpx[2] += target_val[i] * target_elem.get_nodal_coords(2, i);
    }
  }

  // normalize interpolated GP normal back to length 1.0 !!!
  double lengthn = sqrt(gpn[0] * gpn[0] + gpn[1] * gpn[1] + gpn[2] * gpn[2]);
  if (lengthn < 1.0e-12) FOUR_C_THROW("Divide by zero!");

  for (int i = 0; i < 3; ++i) gpn[i] /= lengthn;

  // build gap function at current GP
  for (int i = 0; i < n_dim(); ++i) gap[0] += (mgpx[i] - sgpx[i]) * gpn[i];

  // **************************
  // Linearization
  // **************************
  int linsize = 0;
  if (cppnormal_)
  {
    for (int i = 0; i < ncol; ++i)
    {
      Node* cnode = dynamic_cast<Node*>(target_nodes[i]);
      linsize += cnode->get_linsize();
    }
    linsize += 1 + target_elem.num_node();
  }
  else
  {
    for (int i = 0; i < nrow; ++i)
    {
      Node* cnode = dynamic_cast<Node*>(source_nodes[i]);
      linsize += cnode->get_linsize();
    }
  }
  linsize *= 10;

  // build directional derivative of source GP normal (non-unit)
  Core::Gen::Pairedvector<int, double> dmap_nxsl_gp(linsize);
  Core::Gen::Pairedvector<int, double> dmap_nysl_gp(linsize);
  Core::Gen::Pairedvector<int, double> dmap_nzsl_gp(linsize);

  for (int i = 0; i < nrow; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(source_nodes[i]);

    Core::Gen::Pairedvector<int, double>& dmap_nxsl_i = cnode->data().get_deriv_n()[0];
    Core::Gen::Pairedvector<int, double>& dmap_nysl_i = cnode->data().get_deriv_n()[1];
    Core::Gen::Pairedvector<int, double>& dmap_nzsl_i = cnode->data().get_deriv_n()[2];

    for (CI p = dmap_nxsl_i.begin(); p != dmap_nxsl_i.end(); ++p)
      dmap_nxsl_gp[p->first] += source_val[i] * (p->second);
    for (CI p = dmap_nysl_i.begin(); p != dmap_nysl_i.end(); ++p)
      dmap_nysl_gp[p->first] += source_val[i] * (p->second);
    for (CI p = dmap_nzsl_i.begin(); p != dmap_nzsl_i.end(); ++p)
      dmap_nzsl_gp[p->first] += source_val[i] * (p->second);

    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
    {
      double valx = source_deriv(i, 0) * cnode->mo_data().n()[0];
      dmap_nxsl_gp[p->first] += valx * (p->second);
      double valy = source_deriv(i, 0) * cnode->mo_data().n()[1];
      dmap_nysl_gp[p->first] += valy * (p->second);
      double valz = source_deriv(i, 0) * cnode->mo_data().n()[2];
      dmap_nzsl_gp[p->first] += valz * (p->second);
    }

    for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
    {
      double valx = source_deriv(i, 1) * cnode->mo_data().n()[0];
      dmap_nxsl_gp[p->first] += valx * (p->second);
      double valy = source_deriv(i, 1) * cnode->mo_data().n()[1];
      dmap_nysl_gp[p->first] += valy * (p->second);
      double valz = source_deriv(i, 1) * cnode->mo_data().n()[2];
      dmap_nzsl_gp[p->first] += valz * (p->second);
    }
  }

  const double ll = lengthn * lengthn;
  const double linv = 1.0 / (lengthn);
  const double lllinv = 1.0 / (lengthn * lengthn * lengthn);
  const double sxsx = gpn[0] * gpn[0] * ll;
  const double sxsy = gpn[0] * gpn[1] * ll;
  const double sxsz = gpn[0] * gpn[2] * ll;
  const double sysy = gpn[1] * gpn[1] * ll;
  const double sysz = gpn[1] * gpn[2] * ll;
  const double szsz = gpn[2] * gpn[2] * ll;

  for (CI p = dmap_nxsl_gp.begin(); p != dmap_nxsl_gp.end(); ++p)
  {
    dnmap_unit[0][p->first] += linv * (p->second);
    dnmap_unit[0][p->first] -= lllinv * sxsx * (p->second);
    dnmap_unit[1][p->first] -= lllinv * sxsy * (p->second);
    dnmap_unit[2][p->first] -= lllinv * sxsz * (p->second);
  }

  for (CI p = dmap_nysl_gp.begin(); p != dmap_nysl_gp.end(); ++p)
  {
    dnmap_unit[1][p->first] += linv * (p->second);
    dnmap_unit[1][p->first] -= lllinv * sysy * (p->second);
    dnmap_unit[0][p->first] -= lllinv * sxsy * (p->second);
    dnmap_unit[2][p->first] -= lllinv * sysz * (p->second);
  }

  for (CI p = dmap_nzsl_gp.begin(); p != dmap_nzsl_gp.end(); ++p)
  {
    dnmap_unit[2][p->first] += linv * (p->second);
    dnmap_unit[2][p->first] -= lllinv * szsz * (p->second);
    dnmap_unit[0][p->first] -= lllinv * sxsz * (p->second);
    dnmap_unit[1][p->first] -= lllinv * sysz * (p->second);
  }

  // add everything to dgapgp
  for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    dgapgp[p->first] += (mgpx[0] - sgpx[0]) * (p->second);

  for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    dgapgp[p->first] += (mgpx[1] - sgpx[1]) * (p->second);

  for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    dgapgp[p->first] += (mgpx[2] - sgpx[2]) * (p->second);

  // for wear as own discretization
  // lin slave nodes
  if (wear_type() == Wear::wear_primvar)
  {
    for (int z = 0; z < nrow; ++z)
    {
      for (int k = 0; k < 3; ++k)
      {
        FriNode* frinode = dynamic_cast<FriNode*>(source_nodes[z]);

        dgapgp[frinode->dofs()[k]] -= source_val[z] * gpn[k];

        for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          dgapgp[p->first] -= gpn[k] * source_deriv(z, 0) *
                              (frinode->xspatial()[k] -
                                  frinode->mo_data().n()[k] * frinode->wear_data().wcurr()[0]) *
                              (p->second);

        for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
          dgapgp[p->first] -= gpn[k] * source_deriv(z, 1) *
                              (frinode->xspatial()[k] -
                                  frinode->mo_data().n()[k] * frinode->wear_data().wcurr()[0]) *
                              (p->second);

        for (CI p = frinode->data().get_deriv_n()[k].begin();
            p != frinode->data().get_deriv_n()[k].end(); ++p)
          dgapgp[p->first] +=
              gpn[k] * source_val[z] * frinode->wear_data().wcurr()[0] * (p->second);
      }
    }
  }
  else
  {
    for (int z = 0; z < nrow; ++z)
    {
      Node* cnode = dynamic_cast<Node*>(source_nodes[z]);
      for (int k = 0; k < 3; ++k) dgapgp[cnode->dofs()[k]] -= source_val[z] * gpn[k];
    }

    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
    {
      double& dg = dgapgp[p->first];
      const double& ps = p->second;
      for (int z = 0; z < nrow; ++z)
      {
        Node* cnode = dynamic_cast<Node*>(source_nodes[z]);
        for (int k = 0; k < 3; ++k) dg -= gpn[k] * source_deriv(z, 0) * cnode->xspatial()[k] * ps;
      }
    }

    for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
    {
      double& dg = dgapgp[p->first];
      const double& ps = p->second;
      for (int z = 0; z < nrow; ++z)
      {
        Node* cnode = dynamic_cast<Node*>(source_nodes[z]);
        for (int k = 0; k < 3; ++k) dg -= gpn[k] * source_deriv(z, 1) * cnode->xspatial()[k] * ps;
      }
    }
  }

  //        MASTER
  if (wear_side() == Wear::wear_both and wear_type() == Wear::wear_primvar)
  {
    for (int z = 0; z < ncol; ++z)
    {
      FriNode* frinode = dynamic_cast<FriNode*>(target_nodes[z]);

      for (int k = 0; k < 3; ++k)
      {
        dgapgp[frinode->dofs()[k]] += target_val[z] * gpn[k];

        for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
          dgapgp[p->first] += gpn[k] * target_deriv(z, 0) *
                              (frinode->xspatial()[k] -
                                  frinode->mo_data().n()[k] * frinode->wear_data().wcurr()[0]) *
                              (p->second);

        for (CI p = d_target_xi_gp[1].begin(); p != d_target_xi_gp[1].end(); ++p)
          dgapgp[p->first] += gpn[k] * target_deriv(z, 1) *
                              (frinode->xspatial()[k] -
                                  frinode->mo_data().n()[k] * frinode->wear_data().wcurr()[0]) *
                              (p->second);

        for (CI p = frinode->data().get_deriv_n()[k].begin();
            p != frinode->data().get_deriv_n()[k].end(); ++p)
          dgapgp[p->first] -=
              gpn[k] * target_val[z] * frinode->wear_data().wcurr()[0] * (p->second);
      }
    }
  }
  else
  {
    // lin target nodes
    for (int z = 0; z < ncol; ++z)
    {
      Node* cnode = dynamic_cast<Node*>(target_nodes[z]);
      for (int k = 0; k < 3; ++k) dgapgp[cnode->dofs()[k]] += target_val[z] * gpn[k];
    }

    for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
    {
      double& dg = dgapgp[p->first];
      const double& ps = p->second;
      for (int z = 0; z < ncol; ++z)
      {
        Node* cnode = dynamic_cast<Node*>(target_nodes[z]);
        for (int k = 0; k < 3; ++k) dg += gpn[k] * target_deriv(z, 0) * cnode->xspatial()[k] * ps;
      }
    }

    for (CI p = d_target_xi_gp[1].begin(); p != d_target_xi_gp[1].end(); ++p)
    {
      double& dg = dgapgp[p->first];
      const double& ps = p->second;
      for (int z = 0; z < ncol; ++z)
      {
        Node* cnode = dynamic_cast<Node*>(target_nodes[z]);
        for (int k = 0; k < 3; ++k) dg += gpn[k] * target_deriv(z, 1) * cnode->xspatial()[k] * ps;
      }
    }
  }
}

void CONTACT::Integrator::gap_2d(Mortar::Element& source_elem, Mortar::Element& target_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& target_deriv,
    double* gap, double* gpn, std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_gp,
    Core::Gen::Pairedvector<int, double>& dgapgp,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit)
{
  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = nullptr;
  Core::Nodes::Node** target_nodes = nullptr;
  int nrow = source_elem.num_node();
  int ncol = target_elem.num_node();

  int linsize = 0;
  for (int i = 0; i < nrow; ++i)
    linsize += dynamic_cast<Node*>(source_elem.nodes()[i])->get_linsize();
  // safety
  linsize = linsize * 2;

  int ndof = n_dim();

  source_nodes = source_elem.nodes();
  target_nodes = target_elem.nodes();

  // get source nodal coords for GP evaluation
  Core::LinAlg::SerialDenseMatrix source_coord(3, nrow);
  source_elem.get_nodal_coords(source_coord);

  // get target nodal coords for GP evaluation
  Core::LinAlg::SerialDenseMatrix target_coord(3, ncol);
  target_elem.get_nodal_coords(target_coord);

  std::array<double, 2> sgpx = {0.0, 0.0};
  std::array<double, 2> mgpx = {0.0, 0.0};

  for (int i = 0; i < nrow; ++i)
  {
    Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[i]);
    gpn[0] += source_val[i] * mymrtrnode->mo_data().n()[0];
    gpn[1] += source_val[i] * mymrtrnode->mo_data().n()[1];

    if (wear_type() == Wear::wear_primvar)
    {
      FriNode* myfricnode = dynamic_cast<FriNode*>(mymrtrnode);
      double w = myfricnode->wear_data().wcurr()[0] + myfricnode->wear_data().waccu()[0];
      sgpx[0] += source_val[i] * (source_coord(0, i) - (myfricnode->mo_data().n()[0]) * w);
      sgpx[1] += source_val[i] * (source_coord(1, i) - (myfricnode->mo_data().n()[1]) * w);
    }
    else
    {
      sgpx[0] += source_val[i] * source_coord(0, i);
      sgpx[1] += source_val[i] * source_coord(1, i);
    }
  }

  // build interpolation of target GP coordinates
  for (int i = 0; i < ncol; ++i)
  {
    if (wear_side() == Wear::wear_both and wear_type() == Wear::wear_primvar)
    {
      FriNode* mymrtrnodeM = dynamic_cast<FriNode*>(target_nodes[i]);
      double w = mymrtrnodeM->wear_data().wcurr()[0] + mymrtrnodeM->wear_data().waccu()[0];
      mgpx[0] += target_val[i] * (target_coord(0, i) - (mymrtrnodeM->mo_data().n()[0]) * w);
      mgpx[1] += target_val[i] * (target_coord(1, i) - (mymrtrnodeM->mo_data().n()[1]) * w);
    }
    else
    {
      mgpx[0] += target_val[i] * target_coord(0, i);
      mgpx[1] += target_val[i] * target_coord(1, i);
    }
  }

  // normalize interpolated GP normal back to length 1.0 !!!
  double lengthn = sqrt(gpn[0] * gpn[0] + gpn[1] * gpn[1]);
  if (lengthn < 1.0e-12) FOUR_C_THROW("Divide by zero!");

  for (int i = 0; i < n_dim(); ++i) gpn[i] /= lengthn;

  // build gap function at current GP
  for (int i = 0; i < n_dim(); ++i) gap[0] += (mgpx[i] - sgpx[i]) * gpn[i];

  // **************************
  // Linearization
  // **************************

  // build directional derivative of source GP normal (non-unit)
  Core::Gen::Pairedvector<int, double> dmap_nxsl_gp(ncol * ndof + linsize);
  Core::Gen::Pairedvector<int, double> dmap_nysl_gp(ncol * ndof + linsize);

  for (int i = 0; i < nrow; ++i)
  {
    Mortar::Node* source_node = dynamic_cast<Mortar::Node*>(source_nodes[i]);

    Core::Gen::Pairedvector<int, double>& dmap_nxsl_i =
        dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().get_deriv_n()[0];
    Core::Gen::Pairedvector<int, double>& dmap_nysl_i =
        dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().get_deriv_n()[1];

    for (CI p = dmap_nxsl_i.begin(); p != dmap_nxsl_i.end(); ++p)
      dmap_nxsl_gp[p->first] += source_val[i] * (p->second);
    for (CI p = dmap_nysl_i.begin(); p != dmap_nysl_i.end(); ++p)
      dmap_nysl_gp[p->first] += source_val[i] * (p->second);

    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
    {
      double valx = source_deriv(i, 0) * source_node->mo_data().n()[0];
      dmap_nxsl_gp[p->first] += valx * (p->second);
      double valy = source_deriv(i, 0) * source_node->mo_data().n()[1];
      dmap_nysl_gp[p->first] += valy * (p->second);
    }
  }

  // build directional derivative of source GP normal (unit)
  const double ll = lengthn * lengthn;
  const double linv = 1.0 / lengthn;
  const double lllinv = 1.0 / (lengthn * lengthn * lengthn);
  const double sxsx = gpn[0] * gpn[0] * ll;  // gpn is the unit normal --> multiplication with ll
  const double sxsy = gpn[0] * gpn[1] * ll;  // to get the non-unit normal
  const double sysy = gpn[1] * gpn[1] * ll;

  for (CI p = dmap_nxsl_gp.begin(); p != dmap_nxsl_gp.end(); ++p)
  {
    dnmap_unit[0][p->first] += linv * (p->second);
    dnmap_unit[0][p->first] -= lllinv * sxsx * (p->second);
    dnmap_unit[1][p->first] -= lllinv * sxsy * (p->second);
  }

  for (CI p = dmap_nysl_gp.begin(); p != dmap_nysl_gp.end(); ++p)
  {
    dnmap_unit[1][p->first] += linv * (p->second);
    dnmap_unit[1][p->first] -= lllinv * sysy * (p->second);
    dnmap_unit[0][p->first] -= lllinv * sxsy * (p->second);
  }

  // *****************************************************************************
  // add everything to dgapgp                                                    *
  // *****************************************************************************
  for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    dgapgp[p->first] += (mgpx[0] - sgpx[0]) * (p->second);

  for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    dgapgp[p->first] += (mgpx[1] - sgpx[1]) * (p->second);

  // for wear as own discretization
  // slave nodes
  if (wear_type() == Wear::wear_primvar)
  {
    for (int z = 0; z < nrow; ++z)
    {
      for (int k = 0; k < 2; ++k)
      {
        FriNode* frinode = dynamic_cast<FriNode*>(source_nodes[z]);
        double w = frinode->wear_data().wcurr()[0] + frinode->wear_data().waccu()[0];

        dgapgp[frinode->dofs()[k]] -= source_val[z] * gpn[k];

        for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          dgapgp[p->first] -= gpn[k] * source_deriv(z, 0) *
                              (frinode->xspatial()[k] - frinode->mo_data().n()[k] * w) *
                              (p->second);

        for (CI p = frinode->data().get_deriv_n()[k].begin();
            p != frinode->data().get_deriv_n()[k].end(); ++p)
          dgapgp[p->first] += gpn[k] * source_val[z] * w * (p->second);
      }
    }
  }
  else
  {
    for (int z = 0; z < nrow; ++z)
    {
      Mortar::Node* source_node = dynamic_cast<Mortar::Node*>(source_nodes[z]);

      for (int k = 0; k < n_dim(); ++k)
      {
        dgapgp[source_node->dofs()[k]] -= source_val[z] * (gpn[k]);

        for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          dgapgp[p->first] -=
              gpn[k] * source_deriv(z, 0) * source_node->xspatial()[k] * (p->second);
      }
    }
  }

  // **************************************************
  // master nodes
  if (wear_side() == Wear::wear_both and wear_type() == Wear::wear_primvar)
  {
    for (int z = 0; z < ncol; ++z)
    {
      for (int k = 0; k < 2; ++k)
      {
        FriNode* frinode = dynamic_cast<FriNode*>(target_nodes[z]);
        const double w = frinode->wear_data().wcurr()[0] + frinode->wear_data().waccu()[0];

        dgapgp[frinode->dofs()[k]] += target_val[z] * gpn[k];

        for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
          dgapgp[p->first] += gpn[k] * target_deriv(z, 0) *
                              (frinode->xspatial()[k] - frinode->mo_data().n()[k] * w) *
                              (p->second);

        for (CI p = frinode->data().get_deriv_n()[k].begin();
            p != frinode->data().get_deriv_n()[k].end(); ++p)
          dgapgp[p->first] -= gpn[k] * target_val[z] * w * (p->second);
      }
    }
  }
  else
  {
    for (int z = 0; z < ncol; ++z)
    {
      Mortar::Node* target_node = dynamic_cast<Mortar::Node*>(target_nodes[z]);

      for (int k = 0; k < n_dim(); ++k)
      {
        dgapgp[target_node->dofs()[k]] += target_val[z] * gpn[k];

        for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
          dgapgp[p->first] +=
              gpn[k] * target_deriv(z, 0) * target_node->xspatial()[k] * (p->second);
      }
    }
  }
}


/*----------------------------------------------------------------------*
 |  Compute entries for weighted Gap at GP                   farah 12/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_3d_g_quad_pwlin(Mortar::Element& source_elem,
    Mortar::IntElement& sintele, Mortar::Element& target_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseVector& lmintval, Core::LinAlg::SerialDenseMatrix& source_coord,
    Core::LinAlg::SerialDenseMatrix& target_coord, Core::LinAlg::SerialDenseMatrix& source_deriv,
    Core::LinAlg::SerialDenseMatrix& target_deriv, double* gap, double* gpn, double* lengthn,
    double& jac, double& wgt,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_gp,
    Core::Gen::Pairedvector<int, double>& dgapgp,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit)
{
  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  Core::Nodes::Node** sintnodes = sintele.nodes();
  Core::Nodes::Node** target_nodes = target_elem.nodes();
  if (!source_nodes) FOUR_C_THROW("Null pointer!");
  if (!sintnodes) FOUR_C_THROW("Null pointer!");
  if (!target_nodes) FOUR_C_THROW("Null pointer!");

  // number of nodes (source, target)
  const int nrow = source_elem.num_node();
  const int nintrow = sintele.num_node();
  const int ncol = target_elem.num_node();

  std::array<double, 3> sgpx = {0.0, 0.0, 0.0};
  std::array<double, 3> mgpx = {0.0, 0.0, 0.0};

  for (int i = 0; i < nrow; ++i)
  {
    Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[i]);
    gpn[0] += source_val[i] * mymrtrnode->mo_data().n()[0];
    gpn[1] += source_val[i] * mymrtrnode->mo_data().n()[1];
    gpn[2] += source_val[i] * mymrtrnode->mo_data().n()[2];

    if (wear_type() == Wear::wear_primvar)
    {
      FriNode* myfrinode = dynamic_cast<FriNode*>(source_nodes[i]);
      sgpx[0] += source_val[i] * (source_coord(0, i) - (myfrinode->mo_data().n()[0]) *
                                                           myfrinode->wear_data().wcurr()[0]);
      sgpx[1] += source_val[i] * (source_coord(1, i) - (myfrinode->mo_data().n()[1]) *
                                                           myfrinode->wear_data().wcurr()[0]);
      sgpx[2] += source_val[i] * (source_coord(2, i) - (myfrinode->mo_data().n()[2]) *
                                                           myfrinode->wear_data().wcurr()[0]);
    }
    else
    {
      sgpx[0] += source_val[i] * source_coord(0, i);
      sgpx[1] += source_val[i] * source_coord(1, i);
      sgpx[2] += source_val[i] * source_coord(2, i);
    }
  }

  // build interpolation of target GP coordinates
  for (int i = 0; i < ncol; ++i)
  {
    if (wear_side() == Wear::wear_both and wear_type() == Wear::wear_primvar)
    {
      FriNode* targetnode = dynamic_cast<FriNode*>(target_nodes[i]);

      mgpx[0] += target_val[i] * (target_coord(0, i) - (targetnode->mo_data().n()[0] *
                                                           targetnode->wear_data().wcurr()[0]));
      mgpx[1] += target_val[i] * (target_coord(1, i) - (targetnode->mo_data().n()[1] *
                                                           targetnode->wear_data().wcurr()[0]));
      mgpx[2] += target_val[i] * (target_coord(2, i) - (targetnode->mo_data().n()[2] *
                                                           targetnode->wear_data().wcurr()[0]));
    }
    else
    {
      mgpx[0] += target_val[i] * target_coord(0, i);
      mgpx[1] += target_val[i] * target_coord(1, i);
      mgpx[2] += target_val[i] * target_coord(2, i);
    }
  }

  // normalize interpolated GP normal back to length 1.0 !!!
  lengthn[0] = sqrt(gpn[0] * gpn[0] + gpn[1] * gpn[1] + gpn[2] * gpn[2]);
  if (lengthn[0] < 1.0e-12) FOUR_C_THROW("Divide by zero!");

  for (int i = 0; i < 3; ++i) gpn[i] /= lengthn[0];

  // build gap function at current GP
  for (int i = 0; i < n_dim(); ++i) gap[0] += (mgpx[i] - sgpx[i]) * gpn[i];

  // **************************
  // add to node
  // **************************
  // CASE 3: Standard LM shape functions and piecewise linear interpolation
  // Attention:  for this case, lm_val represents lmintval !!!
  if (shape_fcn() == Mortar::shape_standard && lag_mult_quad() == Mortar::lagmult_pwlin)
  {
    for (int j = 0; j < nintrow; ++j)
    {
      CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(sintnodes[j]);

      double prod = 0.0;
      prod = lmintval[j] * gap[0] * jac * wgt;

      if (cnode->is_on_bound()) continue;

      // add current Gauss point's contribution to gseg
      cnode->addg_value(prod);
    }
  }
  // INVALID CASES
  else
  {
    FOUR_C_THROW("Invalid integration case for 3D quadratic contact!");
  }

  // **************************
  // Linearization
  // **************************
  int linsize = 0;
  for (int i = 0; i < nrow; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(source_nodes[i]);
    linsize += cnode->get_linsize();
  }

  // build directional derivative of source GP normal (non-unit)
  Core::Gen::Pairedvector<int, double> dmap_nxsl_gp(linsize);
  Core::Gen::Pairedvector<int, double> dmap_nysl_gp(linsize);
  Core::Gen::Pairedvector<int, double> dmap_nzsl_gp(linsize);

  for (int i = 0; i < nrow; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(source_nodes[i]);

    Core::Gen::Pairedvector<int, double>& dmap_nxsl_i = cnode->data().get_deriv_n()[0];
    Core::Gen::Pairedvector<int, double>& dmap_nysl_i = cnode->data().get_deriv_n()[1];
    Core::Gen::Pairedvector<int, double>& dmap_nzsl_i = cnode->data().get_deriv_n()[2];

    for (CI p = dmap_nxsl_i.begin(); p != dmap_nxsl_i.end(); ++p)
      dmap_nxsl_gp[p->first] += source_val[i] * (p->second);
    for (CI p = dmap_nysl_i.begin(); p != dmap_nysl_i.end(); ++p)
      dmap_nysl_gp[p->first] += source_val[i] * (p->second);
    for (CI p = dmap_nzsl_i.begin(); p != dmap_nzsl_i.end(); ++p)
      dmap_nzsl_gp[p->first] += source_val[i] * (p->second);

    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
    {
      double valx = source_deriv(i, 0) * cnode->mo_data().n()[0];
      dmap_nxsl_gp[p->first] += valx * (p->second);
      double valy = source_deriv(i, 0) * cnode->mo_data().n()[1];
      dmap_nysl_gp[p->first] += valy * (p->second);
      double valz = source_deriv(i, 0) * cnode->mo_data().n()[2];
      dmap_nzsl_gp[p->first] += valz * (p->second);
    }

    for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
    {
      double valx = source_deriv(i, 1) * cnode->mo_data().n()[0];
      dmap_nxsl_gp[p->first] += valx * (p->second);
      double valy = source_deriv(i, 1) * cnode->mo_data().n()[1];
      dmap_nysl_gp[p->first] += valy * (p->second);
      double valz = source_deriv(i, 1) * cnode->mo_data().n()[2];
      dmap_nzsl_gp[p->first] += valz * (p->second);
    }
  }

  const double ll = lengthn[0] * lengthn[0];
  const double linv = 1.0 / (lengthn[0]);
  const double lllinv = 1.0 / (lengthn[0] * lengthn[0] * lengthn[0]);
  const double sxsx = gpn[0] * gpn[0] * ll;
  const double sxsy = gpn[0] * gpn[1] * ll;
  const double sxsz = gpn[0] * gpn[2] * ll;
  const double sysy = gpn[1] * gpn[1] * ll;
  const double sysz = gpn[1] * gpn[2] * ll;
  const double szsz = gpn[2] * gpn[2] * ll;

  for (CI p = dmap_nxsl_gp.begin(); p != dmap_nxsl_gp.end(); ++p)
  {
    dnmap_unit[0][p->first] += linv * (p->second);
    dnmap_unit[0][p->first] -= lllinv * sxsx * (p->second);
    dnmap_unit[1][p->first] -= lllinv * sxsy * (p->second);
    dnmap_unit[2][p->first] -= lllinv * sxsz * (p->second);
  }

  for (CI p = dmap_nysl_gp.begin(); p != dmap_nysl_gp.end(); ++p)
  {
    dnmap_unit[1][p->first] += linv * (p->second);
    dnmap_unit[1][p->first] -= lllinv * sysy * (p->second);
    dnmap_unit[0][p->first] -= lllinv * sxsy * (p->second);
    dnmap_unit[2][p->first] -= lllinv * sysz * (p->second);
  }

  for (CI p = dmap_nzsl_gp.begin(); p != dmap_nzsl_gp.end(); ++p)
  {
    dnmap_unit[2][p->first] += linv * (p->second);
    dnmap_unit[2][p->first] -= lllinv * szsz * (p->second);
    dnmap_unit[0][p->first] -= lllinv * sxsz * (p->second);
    dnmap_unit[1][p->first] -= lllinv * sysz * (p->second);
  }

  // add everything to dgapgp
  for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    dgapgp[p->first] += (mgpx[0] - sgpx[0]) * (p->second);

  for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    dgapgp[p->first] += (mgpx[1] - sgpx[1]) * (p->second);

  for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    dgapgp[p->first] += (mgpx[2] - sgpx[2]) * (p->second);

  // for wear as own discretization
  // lin slave nodes
  if (wear_type() == Wear::wear_primvar)
  {
    for (int z = 0; z < nrow; ++z)
    {
      for (int k = 0; k < 3; ++k)
      {
        FriNode* frinode = dynamic_cast<FriNode*>(source_nodes[z]);

        dgapgp[frinode->dofs()[k]] -= source_val[z] * gpn[k];

        for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          dgapgp[p->first] -= gpn[k] * source_deriv(z, 0) *
                              (frinode->xspatial()[k] -
                                  frinode->mo_data().n()[k] * frinode->wear_data().wcurr()[0]) *
                              (p->second);

        for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
          dgapgp[p->first] -= gpn[k] * source_deriv(z, 1) *
                              (frinode->xspatial()[k] -
                                  frinode->mo_data().n()[k] * frinode->wear_data().wcurr()[0]) *
                              (p->second);

        for (CI p = frinode->data().get_deriv_n()[k].begin();
            p != frinode->data().get_deriv_n()[k].end(); ++p)
          dgapgp[p->first] +=
              gpn[k] * source_val[z] * frinode->wear_data().wcurr()[0] * (p->second);
      }
    }
  }
  else
  {
    for (int z = 0; z < nrow; ++z)
    {
      Node* cnode = dynamic_cast<Node*>(source_nodes[z]);

      for (int k = 0; k < 3; ++k)
      {
        dgapgp[cnode->dofs()[k]] -= source_val[z] * gpn[k];

        for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          dgapgp[p->first] -= gpn[k] * source_deriv(z, 0) * cnode->xspatial()[k] * (p->second);

        for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
          dgapgp[p->first] -= gpn[k] * source_deriv(z, 1) * cnode->xspatial()[k] * (p->second);
      }
    }
  }

  //        MASTER
  if (wear_side() == Wear::wear_both and wear_type() == Wear::wear_primvar)
  {
    for (int z = 0; z < ncol; ++z)
    {
      FriNode* frinode = dynamic_cast<FriNode*>(target_nodes[z]);

      for (int k = 0; k < 3; ++k)
      {
        dgapgp[frinode->dofs()[k]] += target_val[z] * gpn[k];

        for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
          dgapgp[p->first] += gpn[k] * target_deriv(z, 0) *
                              (frinode->xspatial()[k] -
                                  frinode->mo_data().n()[k] * frinode->wear_data().wcurr()[0]) *
                              (p->second);

        for (CI p = d_target_xi_gp[1].begin(); p != d_target_xi_gp[1].end(); ++p)
          dgapgp[p->first] += gpn[k] * target_deriv(z, 1) *
                              (frinode->xspatial()[k] -
                                  frinode->mo_data().n()[k] * frinode->wear_data().wcurr()[0]) *
                              (p->second);

        for (CI p = frinode->data().get_deriv_n()[k].begin();
            p != frinode->data().get_deriv_n()[k].end(); ++p)
          dgapgp[p->first] -=
              gpn[k] * target_val[z] * frinode->wear_data().wcurr()[0] * (p->second);
      }
    }
  }
  else
  {
    // lin target nodes
    for (int z = 0; z < ncol; ++z)
    {
      Node* cnode = dynamic_cast<Node*>(target_nodes[z]);

      for (int k = 0; k < 3; ++k)
      {
        dgapgp[cnode->dofs()[k]] += target_val[z] * gpn[k];

        for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
          dgapgp[p->first] += gpn[k] * target_deriv(z, 0) * cnode->xspatial()[k] * (p->second);

        for (CI p = d_target_xi_gp[1].begin(); p != d_target_xi_gp[1].end(); ++p)
          dgapgp[p->first] += gpn[k] * target_deriv(z, 1) * cnode->xspatial()[k] * (p->second);
      }
    }
  }
}


/*----------------------------------------------------------------------*
 |  Do lin. entries for weighted Gap at GP                   farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_2d_g_lin(int& iter, Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    double& gap, double* gpn, double& jac, double& wgt,
    Core::Gen::Pairedvector<int, double>& dgapgp,
    Core::Gen::Pairedvector<int, double>& jacintcellmap,
    std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = nullptr;
  int nrow = source_elem.num_node();
  source_nodes = source_elem.nodes();

  Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[iter]);
  if (!mymrtrnode) FOUR_C_THROW("Null pointer!");
  if (mymrtrnode->is_on_boundor_ce()) return;

  double fac = 0.0;

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // get the corresponding map as a reference
  std::map<int, double>& dgmap = dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_g();

  // switch if Petrov-Galerkin approach for LM is applied
  if (shape_fcn() == Mortar::shape_petrovgalerkin)
  {
    // (1) Lin(Phi) - does not exist in gap for Petrov-Galerkin interpolation
    // as std shape functions are used here

    // (2) Lin(Phi) - source GP coordinates
    fac = wgt * source_deriv(iter, 0) * gap * jac;
    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      dgmap[p->first] += fac * (p->second);

    // (3) Lin(g) - gap function
    fac = wgt * source_val[iter] * jac;
    for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += fac * (p->second);

    // (4) Lin(dxdsxi) - source GP Jacobian
    fac = wgt * source_val[iter] * gap;
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
      dgmap[p->first] += fac * (p->second);
  }

  // the usual standard or dual LM interpolation
  else
  {
    // (1) Lin(Phi) - dual shape functions
    if (shape_fcn() == Mortar::shape_dual)
    {
      for (int m = 0; m < nrow; ++m)
      {
        fac = wgt * source_val[m] * gap * jac;
        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 dualmap.begin();
            p != dualmap.end(); ++p)
          dgmap[p->first] += fac * (p->second)(iter, m);
      }
    }

    // (2) Lin(Phi) - source GP coordinates
    fac = wgt * lm_deriv(iter, 0) * gap * jac;
    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      dgmap[p->first] += fac * (p->second);

    // (3) Lin(g) - gap function
    fac = wgt * lm_val[iter] * jac;
    for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += fac * (p->second);

    // (4) Lin(dxdsxi) - source GP Jacobian
    fac = wgt * lm_val[iter] * gap;
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
      dgmap[p->first] += fac * (p->second);
  }

  //****************************************************************
  // LIN WEAR W.R.T. LM
  //****************************************************************
  if (wear_type() == Wear::wear_primvar)
  {
    // get target element nodes themselves
    const int ncol = target_elem.num_node();
    Core::Nodes::Node** target_nodes = target_elem.nodes();

    std::map<int, double>& dgwmmap =
        dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_gw();

    for (int bl = 0; bl < nrow; ++bl)
    {
      Mortar::Node* wearnode = dynamic_cast<Mortar::Node*>(source_nodes[bl]);
      for (int z = 0; z < n_dim(); ++z)
        dgwmmap[wearnode->dofs()[0]] +=
            jac * wgt * lm_val[iter] * (gpn[z] * source_val[bl] * wearnode->mo_data().n()[z]);
    }

    if (wear_side() == Wear::wear_both)
    {
      for (int bl = 0; bl < ncol; ++bl)
      {
        Mortar::Node* wearnodeM = dynamic_cast<Mortar::Node*>(target_nodes[bl]);
        for (int z = 0; z < n_dim(); ++z)
          dgwmmap[wearnodeM->dofs()[0]] -=
              jac * wgt * lm_val[iter] * (gpn[z] * target_val[bl] * wearnodeM->mo_data().n()[z]);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Do lin. entries for weighted Gap at GP                   farah 12/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_3d_g_quad_pwlin_lin(int& iter, Mortar::IntElement& sintele,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& lmintval,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& lmintderiv,
    double& gap, double* gpn, double& jac, double& wgt,
    const Core::Gen::Pairedvector<int, double>& dgapgp,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp)
{
  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // get source element nodes themselves
  Core::Nodes::Node** sintnodes = sintele.nodes();

  Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(sintnodes[iter]);
  if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

  double fac = 0.0;

  // CASE 3: Standard LM shape functions and piecewise linear interpolation
  if (shape_fcn() == Mortar::shape_standard && lag_mult_quad() == Mortar::lagmult_pwlin)
  {
    // get the corresponding map as a reference
    std::map<int, double>& dgmap = dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_g();

    // (1) Lin(Phi) - dual shape functions
    // this vanishes here since there are no deformation-dependent dual functions

    // (2) Lin(Phi) - source GP coordinates
    fac = wgt * lmintderiv(iter, 0) * gap * jac;
    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      dgmap[p->first] += fac * (p->second);

    fac = wgt * lmintderiv(iter, 1) * gap * jac;
    for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
      dgmap[p->first] += fac * (p->second);

    // (3) Lin(g) - gap function
    fac = wgt * lmintval[iter] * jac;
    for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += fac * (p->second);

    // (4) Lin(dsxideta) - intcell GP Jacobian
    fac = wgt * lmintval[iter] * gap;
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
      dgmap[p->first] += fac * (p->second);
  }
  else
    FOUR_C_THROW("shapefcn-lagmult combination not supported!");
}

/*----------------------------------------------------------------------*
 |  Do lin. entries for weighted Gap at GP                   farah 12/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_3d_g_quad_lin(int& iter, Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& svalmod, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    double& gap, double* gpn, double& jac, double& wgt, bool& duallin,
    const Core::Gen::Pairedvector<int, double>& dgapgp,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dp_source_xigp,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, bool dualquad3d)
{
  const int nrow = source_elem.num_node();

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();

  Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[iter]);
  if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

  double fac = 0.0;

  // compute cell gap linearization ************************************
  // CASE 1/2: Standard LM shape functions and quadratic or linear interpolation
  if (shape_fcn() == Mortar::shape_standard &&
      (lag_mult_quad() == Mortar::lagmult_quad || lag_mult_quad() == Mortar::lagmult_lin))
  {
    // get the corresponding map as a reference
    std::map<int, double>& dgmap = dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_g();

    // (1) Lin(Phi) - dual shape functions
    // this vanishes here since there are no deformation-dependent dual functions

    // (2) Lin(Phi) - source GP coordinates
    fac = wgt * lm_deriv(iter, 0) * gap * jac;
    for (CI p = dp_source_xigp[0].begin(); p != dp_source_xigp[0].end(); ++p)
      dgmap[p->first] += fac * (p->second);

    fac = wgt * lm_deriv(iter, 1) * gap * jac;
    for (CI p = dp_source_xigp[1].begin(); p != dp_source_xigp[1].end(); ++p)
      dgmap[p->first] += fac * (p->second);

    // (3) Lin(g) - gap function
    fac = wgt * lm_val[iter] * jac;
    for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += fac * (p->second);

    // (4) Lin(dsxideta) - intcell GP Jacobian
    fac = wgt * lm_val[iter] * gap;
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
      dgmap[p->first] += fac * (p->second);
  }

  // CASE 4: Dual LM shape functions and quadratic or linear interpolation
  else if ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
           (lag_mult_quad() == Mortar::lagmult_quad || lag_mult_quad() == Mortar::lagmult_lin))
  {
    // get the corresponding map as a reference
    std::map<int, double>& dgmap = dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_g();

    if (shape_fcn() == Mortar::shape_dual)
    {
      // (1) Lin(Phi) - dual shape functions
      if (duallin)
      {
        for (int m = 0; m < nrow; ++m)
        {
          if (dualquad3d)
            fac = wgt * svalmod[m] * gap * jac;
          else
            fac = wgt * source_val[m] * gap * jac;

          for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                   dualmap.begin();
              p != dualmap.end(); ++p)
            dgmap[p->first] += fac * (p->second)(iter, m);
        }
      }

      // (2) Lin(Phi) - source GP coordinates
      fac = wgt * lm_deriv(iter, 0) * gap * jac;
      for (CI p = dp_source_xigp[0].begin(); p != dp_source_xigp[0].end(); ++p)
        dgmap[p->first] += fac * (p->second);

      fac = wgt * lm_deriv(iter, 1) * gap * jac;
      for (CI p = dp_source_xigp[1].begin(); p != dp_source_xigp[1].end(); ++p)
        dgmap[p->first] += fac * (p->second);

      // (3) Lin(g) - gap function
      fac = wgt * lm_val[iter] * jac;
      for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += fac * (p->second);

      // (4) Lin(dsxideta) - intcell GP Jacobian
      fac = wgt * lm_val[iter] * gap;
      for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
        dgmap[p->first] += fac * (p->second);
    }
    else if (shape_fcn() == Mortar::shape_petrovgalerkin)
    {
      // (1) Lin(Phi) - dual shape functions

      // (2) Lin(Phi) - source GP coordinates
      fac = wgt * source_deriv(iter, 0) * gap * jac;
      for (CI p = dp_source_xigp[0].begin(); p != dp_source_xigp[0].end(); ++p)
        dgmap[p->first] += fac * (p->second);

      fac = wgt * source_deriv(iter, 1) * gap * jac;
      for (CI p = dp_source_xigp[1].begin(); p != dp_source_xigp[1].end(); ++p)
        dgmap[p->first] += fac * (p->second);

      // (3) Lin(g) - gap function
      fac = wgt * source_val[iter] * jac;
      for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += fac * (p->second);

      // (4) Lin(dsxideta) - intcell GP Jacobian
      fac = wgt * source_val[iter] * gap;
      for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
        dgmap[p->first] += fac * (p->second);
    }
  }
}

/*----------------------------------------------------------------------*
 |  Do lin. entries for weighted Gap at GP                   farah 09/13|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::gp_g_lin(int& iter, Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    double& gap, double* gpn, double& jac, double& wgt,
    Core::Gen::Pairedvector<int, double>& dgapgp,
    Core::Gen::Pairedvector<int, double>& jacintcellmap,
    std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  const int nrow = source_elem.num_node();
  const int ncol = target_elem.num_node();

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  Core::Nodes::Node** target_nodes = target_elem.nodes();

  Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[iter]);
  if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

  if (mymrtrnode->is_on_boundor_ce()) return;

  static double fac = 0.0;

  // get the corresponding map as a reference
  std::map<int, double>& dgmap = dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_g();

  // switch if Petrov-Galerkin approach for LM is applied
  if (shape_fcn() == Mortar::shape_petrovgalerkin)
  {
    // (1) Lin(Phi) - does not exist here for Petrov-Galerkin approach

    // (2) Lin(N) - source GP coordinates
    for (int d = 0; d < n_dim() - 1; ++d)
    {
      fac = wgt * source_deriv(iter, d) * gap * jac;
      for (CI p = d_source_xi_gp[d].begin(); p != d_source_xi_gp[d].end(); ++p)
        dgmap[p->first] += fac * (p->second);
    }

    // (3) Lin(g) - gap function
    fac = wgt * source_val[iter] * jac;
    for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += fac * (p->second);

    // (4) Lin(dsxideta) - intcell GP Jacobian
    fac = wgt * source_val[iter] * gap;
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
      dgmap[p->first] += fac * (p->second);
  }

  // the usual standard or dual LM approach
  else
  {
    // (1) Lin(Phi) - dual shape functions
    if (dualmap.size() > 0)
      for (int m = 0; m < nrow; ++m)
      {
        fac = wgt * source_val[m] * gap * jac;
        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 dualmap.begin();
            p != dualmap.end(); ++p)
          dgmap[p->first] += fac * (p->second)(iter, m);
      }

    // (2) Lin(Phi) - source GP coordinates
    for (int d = 0; d < n_dim() - 1; ++d)
    {
      fac = wgt * lm_deriv(iter, d) * gap * jac;
      for (CI p = d_source_xi_gp[d].begin(); p != d_source_xi_gp[d].end(); ++p)
        dgmap[p->first] += fac * (p->second);
    }

    // (3) Lin(g) - gap function
    fac = wgt * lm_val[iter] * jac;
    for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p) dgmap[p->first] += fac * (p->second);

    // (4) Lin(dsxideta) - intcell GP Jacobian
    fac = wgt * lm_val[iter] * gap;
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
      dgmap[p->first] += fac * (p->second);
  }

  //****************************************************************
  // LIN WEAR W.R.T. W
  //****************************************************************
  if (wear_type() == Wear::wear_primvar)
  {
    std::map<int, double>& dgwmmap =
        dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_gw();

    for (int bl = 0; bl < nrow; ++bl)
    {
      Mortar::Node* wearnode = dynamic_cast<Mortar::Node*>(source_nodes[bl]);
      for (int z = 0; z < n_dim(); ++z)
        dgwmmap[wearnode->dofs()[0]] +=
            jac * wgt * lm_val[iter] * (gpn[z] * source_val[bl] * wearnode->mo_data().n()[z]);
    }

    if (wear_side() == Wear::wear_both)
    {
      for (int bl = 0; bl < ncol; ++bl)
      {
        Mortar::Node* wearnodeM = dynamic_cast<Mortar::Node*>(target_nodes[bl]);
        for (int z = 0; z < n_dim(); ++z)
          dgwmmap[wearnodeM->dofs()[0]] -=
              jac * wgt * lm_val[iter] * (gpn[z] * target_val[bl] * wearnodeM->mo_data().n()[z]);
      }
    }
  }
}


/*----------------------------------------------------------------------*
 |  Compute entries for bound DM                             farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::gp_3d_dm_lin_bound(Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    Core::LinAlg::SerialDenseMatrix& target_deriv, double& jac, double& wgt,
    const Core::Gen::Pairedvector<int, double>& derivjac,
    std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_gp,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  const int nrow = source_elem.num_node();
  const int ncol = target_elem.num_node();

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  if (shape_fcn() == Mortar::shape_standard)
  {
    // integrate LinD
    for (int j = 0; j < nrow; ++j)
    {
      Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[j]);
      if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

      // node j is boundary node
      if (mymrtrnode->is_on_boundor_ce()) continue;

      // integrate LinM
      for (int k = 0; k < ncol; ++k)
      {
        // global target node ID
        int t_gid = target_elem.nodes()[k]->id();
        static double fac = 0.0;

        // get the correct map as a reference
        std::map<int, double>& dmmap_jk =
            dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[t_gid];

        // (1) Lin(Phi) - dual shape functions
        // this vanishes here since there are no deformation-dependent dual functions

        // (2) Lin(NSource) - source GP coordinates
        fac = wgt * lm_deriv(j, 0) * target_val[k] * jac;
        for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        fac = wgt * lm_deriv(j, 1) * target_val[k] * jac;
        for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        // (3) Lin(NTarget) - target GP coordinates
        fac = wgt * lm_val[j] * target_deriv(k, 0) * jac;
        for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        fac = wgt * lm_val[j] * target_deriv(k, 1) * jac;
        for (CI p = d_target_xi_gp[1].begin(); p != d_target_xi_gp[1].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        // (4) Lin(dsxideta) - intcell GP Jacobian
        fac = wgt * lm_val[j] * target_val[k];
        for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);
      }  // loop over target nodes

      for (int k = 0; k < nrow; ++k)
      {
        Mortar::Node* mymrtrnode2 = dynamic_cast<Mortar::Node*>(source_nodes[k]);
        if (!mymrtrnode2) FOUR_C_THROW("Null pointer!");

        // global target node ID
        int s_gid = mymrtrnode2->id();
        static double fac = 0.0;

        // node k is boundary node
        if (mymrtrnode2->is_on_boundor_ce())
        {
          // get the correct map pointer
          std::map<int, double>* dmmap_jk = nullptr;
          double sign = 0.;

          // for boundary nodes, we assemble to M-matrix, since the corresponding nodes do not
          // carry a LM-dof
          if (mymrtrnode2->is_on_bound())
          {
            dmmap_jk = &(dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[s_gid]);
            sign = -1.;
          }
          // for corners/edges, we assemble to D-matrix, since the corresponding nodes
          // carry a LM-dof, but are integrated some place else. Here we only do surface
          // integration.
          else if (mymrtrnode2->is_on_corner_edge())
          {
            dmmap_jk = &(dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_d()[s_gid]);
            sign = +1.;
          }
          else
            FOUR_C_THROW("you should never be here");

          // (1) Lin(Phi) - dual shape functions
          // this vanishes here since there are no deformation-dependent dual functions

          // (2) Lin(NSource) - source GP coordinates
          fac = sign * wgt * lm_deriv(j, 0) * source_val[k] * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
            (*dmmap_jk)[p->first] += fac * (p->second);

          fac = sign * wgt * lm_deriv(j, 1) * source_val[k] * jac;
          for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
            (*dmmap_jk)[p->first] += fac * (p->second);

          // (3) Lin(NSource) - source GP coordinates
          fac = sign * wgt * lm_val[j] * source_deriv(k, 0) * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
            (*dmmap_jk)[p->first] += fac * (p->second);

          fac = sign * wgt * lm_val[j] * source_deriv(k, 1) * jac;
          for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
            (*dmmap_jk)[p->first] += fac * (p->second);

          // (4) Lin(dsxideta) - intcell GP Jacobian
          fac = sign * wgt * lm_val[j] * source_val[k];
          for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
            (*dmmap_jk)[p->first] += fac * (p->second);
        }

        // node k is NO boundary node
        else
        {
          // get the correct map as a reference
          std::map<int, double>& ddmap_jk =
              dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_d()[s_gid];

          // (1) Lin(Phi) - dual shape functions
          // this vanishes here since there are no deformation-dependent dual functions

          // (2) Lin(NSource) - source GP coordinates
          fac = wgt * lm_deriv(j, 0) * source_val[k] * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);

          fac = wgt * lm_deriv(j, 1) * source_val[k] * jac;
          for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);

          // (3) Lin(NSource) - source GP coordinates
          fac = wgt * lm_val[j] * source_deriv(k, 0) * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);

          fac = wgt * lm_val[j] * source_deriv(k, 1) * jac;
          for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);

          // (4) Lin(dsxideta) - intcell GP Jacobian
          fac = wgt * lm_val[j] * source_val[k];
          for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);
        }
      }  // loop over source nodes
    }
  }
  //--------------------------------------------------------
  else if (shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin)
  {
    // integrate LinD
    for (int j = 0; j < nrow; ++j)
    {
      Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[j]);
      if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

      // node j is boundary node
      if (mymrtrnode->is_on_boundor_ce()) continue;

      // integrate LinM
      for (int k = 0; k < ncol; ++k)
      {
        // global target node ID
        int t_gid = target_elem.nodes()[k]->id();
        static double fac = 0.0;

        // get the correct map as a reference
        std::map<int, double>& dmmap_jk =
            dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[t_gid];

        // (1) Lin(Phi) - dual shape functions
        if (dualmap.size() > 0)
        {
          for (int m = 0; m < nrow; ++m)
          {
            fac = wgt * source_val[m] * target_val[k] * jac;
            for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                     dualmap.begin();
                p != dualmap.end(); ++p)
              dmmap_jk[p->first] += fac * (p->second)(j, m);
          }
        }

        // (2) Lin(NSource) - source GP coordinates
        fac = wgt * lm_deriv(j, 0) * target_val[k] * jac;
        for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        fac = wgt * lm_deriv(j, 1) * target_val[k] * jac;
        for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        // (3) Lin(NTarget) - target GP coordinates
        fac = wgt * lm_val[j] * target_deriv(k, 0) * jac;
        for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        fac = wgt * lm_val[j] * target_deriv(k, 1) * jac;
        for (CI p = d_target_xi_gp[1].begin(); p != d_target_xi_gp[1].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        // (4) Lin(dsxideta) - intcell GP Jacobian
        fac = wgt * lm_val[j] * target_val[k];
        for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);
      }  // loop over target nodes

      // loop over source nodes
      for (int k = 0; k < nrow; ++k)
      {
        Mortar::Node* mymrtrnode2 = dynamic_cast<Mortar::Node*>(source_nodes[k]);
        if (!mymrtrnode2) FOUR_C_THROW("Null pointer!");

        // global target node ID
        int s_gid = mymrtrnode2->id();
        static double fac = 0.0;

        // node k is boundary node
        if (mymrtrnode2->is_on_boundor_ce())
        {
          // get the correct map pointer
          std::map<int, double>* dmmap_jk = nullptr;
          double sign = 0.;

          // for boundary nodes, we assemble to M-matrix, since the corresponding nodes do not
          // carry a LM-dof
          if (mymrtrnode2->is_on_bound())
          {
            dmmap_jk = &(dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[s_gid]);
            sign = -1.;
          }
          // for corners/edges, we assemble to D-matrix, since the corresponding nodes
          // carry a LM-dof, but are integrated some place else. Here we only do surface
          // integration.
          else if (mymrtrnode2->is_on_corner_edge())
          {
            dmmap_jk = &(dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_d()[s_gid]);
            sign = +1.;
          }
          else
            FOUR_C_THROW("you should never be here");

          // (1) Lin(Phi) - dual shape functions
          if (dualmap.size() > 0)
          {
            for (int m = 0; m < nrow; ++m)
            {
              fac = sign * wgt * source_val[m] * source_val[k] * jac;
              for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                       dualmap.begin();
                  p != dualmap.end(); ++p)
                (*dmmap_jk)[p->first] += fac * (p->second)(j, m);
            }
          }

          // (2) Lin(NSource) - source GP coordinates
          fac = sign * wgt * lm_deriv(j, 0) * source_val[k] * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
            (*dmmap_jk)[p->first] += fac * (p->second);

          fac = sign * wgt * lm_deriv(j, 1) * source_val[k] * jac;
          for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
            (*dmmap_jk)[p->first] += fac * (p->second);

          // (3) Lin(NSource) - source GP coordinates
          fac = sign * wgt * lm_val[j] * source_deriv(k, 0) * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
            (*dmmap_jk)[p->first] += fac * (p->second);

          fac = sign * wgt * lm_val[j] * source_deriv(k, 1) * jac;
          for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
            (*dmmap_jk)[p->first] += fac * (p->second);

          // (4) Lin(dsxideta) - intcell GP Jacobian
          fac = sign * wgt * lm_val[j] * source_val[k];
          for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
            (*dmmap_jk)[p->first] += fac * (p->second);
        }

        // node k is NO boundary node
        else
        {
          // get the correct map as a reference
          std::map<int, double>& ddmap_jk =
              dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_d()[mymrtrnode->id()];

          // (1) Lin(Phi) - dual shape functions
          if (dualmap.size() > 0)
          {
            for (int m = 0; m < nrow; ++m)
            {
              fac = wgt * source_val[m] * source_val[k] * jac;
              for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                       dualmap.begin();
                  p != dualmap.end(); ++p)
                ddmap_jk[p->first] += fac * (p->second)(j, m);
            }
          }

          // (2) Lin(NSource) - source GP coordinates
          fac = wgt * lm_deriv(j, 0) * source_val[k] * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);

          fac = wgt * lm_deriv(j, 1) * source_val[k] * jac;
          for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);

          // (3) Lin(NSource) - source GP coordinates
          fac = wgt * lm_val[j] * source_deriv(k, 0) * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);

          fac = wgt * lm_val[j] * source_deriv(k, 1) * jac;
          for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);

          // (4) Lin(dsxideta) - intcell GP Jacobian
          fac = wgt * lm_val[j] * source_val[k];
          for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);
        }
      }  // loop over source nodes
    }
  }
  else
    FOUR_C_THROW("unknown shape function type!");
}

/*----------------------------------------------------------------------*
 |  Compute entries for weighted Gap at GP                   farah 07/16|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_2d_dm_lin_bound(Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& target_deriv,
    Core::LinAlg::SerialDenseMatrix& lm_deriv, double& jac, double& wgt,
    const Core::Gen::Pairedvector<int, double>& derivjac,
    std::vector<Core::Gen::Pairedvector<int, double>>& source_derivs_xi,
    std::vector<Core::Gen::Pairedvector<int, double>>& target_derivs_xi,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  const int nrow = source_elem.num_node();
  const int ncol = target_elem.num_node();

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  if (shape_fcn() == Mortar::shape_standard)
  {
    // integrate LinD
    for (int j = 0; j < nrow; ++j)
    {
      Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[j]);
      if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

      // node j is boundary node
      if (mymrtrnode->is_on_boundor_ce()) continue;

      // integrate LinM
      for (int k = 0; k < ncol; ++k)
      {
        // global target node ID
        int t_gid = target_elem.nodes()[k]->id();
        static double fac = 0.0;

        // get the correct map as a reference
        std::map<int, double>& dmmap_jk =
            dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[t_gid];

        // (1) Lin(Phi) - dual shape functions
        // this vanishes here since there are no deformation-dependent dual functions

        // (2) Lin(NSource) - source GP coordinates
        fac = wgt * lm_deriv(j, 0) * target_val[k] * jac;
        for (CI p = source_derivs_xi[0].begin(); p != source_derivs_xi[0].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        //        fac = wgt*lm_deriv(j, 1)*target_val[k]*jac;
        //        for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
        //          dmmap_jk[p->first] += fac*(p->second);

        // (3) Lin(NTarget) - target GP coordinates
        fac = wgt * lm_val[j] * target_deriv(k, 0) * jac;
        for (CI p = target_derivs_xi[0].begin(); p != target_derivs_xi[0].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        //        fac = wgt*lm_val[j]*target_deriv(k, 1)*jac;
        //        for (CI p=d_target_xi_gp[1].begin(); p!=d_target_xi_gp[1].end(); ++p)
        //          dmmap_jk[p->first] += fac*(p->second);

        // (4) Lin(dsxideta) - intcell GP Jacobian
        fac = wgt * lm_val[j] * target_val[k];
        for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);
      }  // loop over target nodes

      for (int k = 0; k < nrow; ++k)
      {
        Mortar::Node* mymrtrnode2 = dynamic_cast<Mortar::Node*>(source_nodes[k]);
        if (!mymrtrnode2) FOUR_C_THROW("Null pointer!");

        // global target node ID
        int s_gid = mymrtrnode2->id();
        static double fac = 0.0;

        // node k is boundary node
        if (mymrtrnode2->is_on_boundor_ce())
        {
          // move entry to derivM (with minus sign)
          // get the correct map as a reference
          std::map<int, double>& dmmap_jk =
              dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[s_gid];

          // (1) Lin(Phi) - dual shape functions
          // this vanishes here since there are no deformation-dependent dual functions

          // (2) Lin(NSource) - source GP coordinates
          fac = wgt * lm_deriv(j, 0) * source_val[k] * jac;
          for (CI p = source_derivs_xi[0].begin(); p != source_derivs_xi[0].end(); ++p)
            dmmap_jk[p->first] -= fac * (p->second);

          //          fac = wgt*lm_deriv(j, 1)*source_val[k]*jac;
          //          for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
          //            dmmap_jk[p->first] -= fac*(p->second);

          // (3) Lin(NSource) - source GP coordinates
          fac = wgt * lm_val[j] * source_deriv(k, 0) * jac;
          for (CI p = source_derivs_xi[0].begin(); p != source_derivs_xi[0].end(); ++p)
            dmmap_jk[p->first] -= fac * (p->second);

          //          fac = wgt*lm_val[j]*source_deriv(k, 1)*jac;
          //          for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
          //            dmmap_jk[p->first] -= fac*(p->second);

          // (4) Lin(dsxideta) - intcell GP Jacobian
          fac = wgt * lm_val[j] * source_val[k];
          for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
            dmmap_jk[p->first] -= fac * (p->second);
        }

        // node k is NO boundary node
        else
        {
          // get the correct map as a reference
          std::map<int, double>& ddmap_jk =
              dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_d()[s_gid];

          // (1) Lin(Phi) - dual shape functions
          // this vanishes here since there are no deformation-dependent dual functions

          // (2) Lin(NSource) - source GP coordinates
          fac = wgt * lm_deriv(j, 0) * source_val[k] * jac;
          for (CI p = source_derivs_xi[0].begin(); p != source_derivs_xi[0].end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);

          //          fac = wgt*lm_deriv(j, 1)*source_val[k]*jac;
          //          for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
          //            ddmap_jk[p->first] += fac*(p->second);

          // (3) Lin(NSource) - source GP coordinates
          fac = wgt * lm_val[j] * source_deriv(k, 0) * jac;
          for (CI p = source_derivs_xi[0].begin(); p != source_derivs_xi[0].end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);

          //          fac = wgt*lm_val[j]*source_deriv(k, 1)*jac;
          //          for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
          //            ddmap_jk[p->first] += fac*(p->second);

          // (4) Lin(dsxideta) - intcell GP Jacobian
          fac = wgt * lm_val[j] * source_val[k];
          for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);
        }
      }  // loop over source nodes
    }
  }
  //--------------------------------------------------------
  else if (shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin)
  {
    // integrate LinD
    for (int j = 0; j < nrow; ++j)
    {
      Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[j]);
      if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

      // node j is boundary node
      if (mymrtrnode->is_on_boundor_ce()) continue;

      // integrate LinM
      for (int k = 0; k < ncol; ++k)
      {
        // global target node ID
        int t_gid = target_elem.nodes()[k]->id();
        static double fac = 0.0;

        // get the correct map as a reference
        std::map<int, double>& dmmap_jk =
            dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[t_gid];

        // (1) Lin(Phi) - dual shape functions
        if (dualmap.size() > 0)
        {
          for (int m = 0; m < nrow; ++m)
          {
            fac = wgt * source_val[m] * target_val[k] * jac;
            for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                     dualmap.begin();
                p != dualmap.end(); ++p)
              dmmap_jk[p->first] += fac * (p->second)(j, m);
          }
        }

        // (2) Lin(NSource) - source GP coordinates
        fac = wgt * lm_deriv(j, 0) * target_val[k] * jac;
        for (CI p = source_derivs_xi[0].begin(); p != source_derivs_xi[0].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        //        fac = wgt*lm_deriv(j, 1)*target_val[k]*jac;
        //        for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
        //          dmmap_jk[p->first] += fac*(p->second);

        // (3) Lin(NTarget) - target GP coordinates
        fac = wgt * lm_val[j] * target_deriv(k, 0) * jac;
        for (CI p = target_derivs_xi[0].begin(); p != target_derivs_xi[0].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        //        fac = wgt*lm_val[j]*target_deriv(k, 1)*jac;
        //        for (CI p=d_target_xi_gp[1].begin(); p!=d_target_xi_gp[1].end(); ++p)
        //          dmmap_jk[p->first] += fac*(p->second);

        // (4) Lin(dsxideta) - intcell GP Jacobian
        fac = wgt * lm_val[j] * target_val[k];
        for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);
      }  // loop over target nodes

      // loop over source nodes
      for (int k = 0; k < nrow; ++k)
      {
        Mortar::Node* mymrtrnode2 = dynamic_cast<Mortar::Node*>(source_nodes[k]);
        if (!mymrtrnode2) FOUR_C_THROW("Null pointer!");

        // global target node ID
        int s_gid = mymrtrnode2->id();
        static double fac = 0.0;

        // node k is boundary node
        if (mymrtrnode2->is_on_boundor_ce())
        {
          // move entry to derivM (with minus sign)
          // get the correct map as a reference
          std::map<int, double>& dmmap_jk =
              dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[s_gid];

          // (1) Lin(Phi) - dual shape functions
          if (dualmap.size() > 0)
          {
            for (int m = 0; m < nrow; ++m)
            {
              fac = wgt * source_val[m] * source_val[k] * jac;
              for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                       dualmap.begin();
                  p != dualmap.end(); ++p)
                dmmap_jk[p->first] -= fac * (p->second)(j, m);
            }
          }

          // (2) Lin(NSource) - source GP coordinates
          fac = wgt * lm_deriv(j, 0) * source_val[k] * jac;
          for (CI p = source_derivs_xi[0].begin(); p != source_derivs_xi[0].end(); ++p)
            dmmap_jk[p->first] -= fac * (p->second);

          //          fac = wgt*lm_deriv(j, 1)*source_val[k]*jac;
          //          for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
          //            dmmap_jk[p->first] -= fac*(p->second);

          // (3) Lin(NSource) - source GP coordinates
          fac = wgt * lm_val[j] * source_deriv(k, 0) * jac;
          for (CI p = source_derivs_xi[0].begin(); p != source_derivs_xi[0].end(); ++p)
            dmmap_jk[p->first] -= fac * (p->second);

          //          fac = wgt*lm_val[j]*source_deriv(k, 1)*jac;
          //          for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
          //            dmmap_jk[p->first] -= fac*(p->second);

          // (4) Lin(dsxideta) - intcell GP Jacobian
          fac = wgt * lm_val[j] * source_val[k];
          for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
            dmmap_jk[p->first] -= fac * (p->second);
        }

        // node k is NO boundary node
        else
        {
          // get the correct map as a reference
          std::map<int, double>& ddmap_jk =
              dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_d()[mymrtrnode->id()];

          // (1) Lin(Phi) - dual shape functions
          if (dualmap.size() > 0)
          {
            for (int m = 0; m < nrow; ++m)
            {
              fac = wgt * source_val[m] * source_val[k] * jac;
              for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                       dualmap.begin();
                  p != dualmap.end(); ++p)
                ddmap_jk[p->first] += fac * (p->second)(j, m);
            }
          }

          // (2) Lin(NSource) - source GP coordinates
          fac = wgt * lm_deriv(j, 0) * source_val[k] * jac;
          for (CI p = source_derivs_xi[0].begin(); p != source_derivs_xi[0].end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);

          //          fac = wgt*lm_deriv(j, 1)*source_val[k]*jac;
          //          for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
          //            ddmap_jk[p->first] += fac*(p->second);

          // (3) Lin(NSource) - source GP coordinates
          fac = wgt * lm_val[j] * source_deriv(k, 0) * jac;
          for (CI p = source_derivs_xi[0].begin(); p != source_derivs_xi[0].end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);

          //          fac = wgt*lm_val[j]*source_deriv(k, 1)*jac;
          //          for (CI p=d_source_xi_gp[1].begin(); p!=d_source_xi_gp[1].end(); ++p)
          //            ddmap_jk[p->first] += fac*(p->second);

          // (4) Lin(dsxideta) - intcell GP Jacobian
          fac = wgt * lm_val[j] * source_val[k];
          for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);
        }
      }  // loop over source nodes
    }
  }
  else
    FOUR_C_THROW("unknown shape function type!");
}


/*----------------------------------------------------------------------*
 |  Lin D and M matrix entries at GP                         farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_2d_dm_ele_lin(int& iter, bool& bound,
    Mortar::Element& source_elem, Mortar::Element& target_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseVector& lm_val, Core::LinAlg::SerialDenseMatrix& target_deriv,
    double& dxdsxi, double& wgt, const Core::Gen::Pairedvector<int, double>& d_target_xi_gp,
    const Core::Gen::Pairedvector<int, double>& derivjac,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  Core::Nodes::Node** source_nodes = nullptr;
  Core::Nodes::Node** target_nodes = nullptr;

  int nrow = source_elem.num_node();
  int ncol = target_elem.num_node();

  source_nodes = source_elem.nodes();
  target_nodes = target_elem.nodes();

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[iter]);
  if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

  const int s_gid = mymrtrnode->id();

  // standard shape functions
  if (shape_fcn() == Mortar::shape_standard)
  {
    // integrate LinM
    for (int k = 0; k < ncol; ++k)
    {
      // global target node ID
      int t_gid = target_nodes[k]->id();
      double fac = 0.0;


      // get the correct map as a reference
      std::map<int, double>& dmmap_jk =
          dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[t_gid];

      // (1) Lin(Phi) - dual shape functions    --> 0

      // (2) Lin(NSource) - source GP coordinates --> 0

      // (3) Lin(NTarget) - target GP coordinates
      fac = wgt * lm_val[iter] * target_deriv(k, 0) * dxdsxi;
      for (CI p = d_target_xi_gp.begin(); p != d_target_xi_gp.end(); ++p)
        dmmap_jk[p->first] += fac * (p->second);

      // (4) Lin(dsxideta) - segment end coordinates --> 0

      // (5) Lin(dxdsxi) - source GP Jacobian
      fac = wgt * lm_val[iter] * target_val[k];
      for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
        dmmap_jk[p->first] += fac * (p->second);

      // (6) Lin(dxdsxi) - source GP coordinates --> 0
    }  // loop over target nodes

    // integrate LinD
    for (int k = 0; k < nrow; ++k)
    {
      // global source node ID
      int s_gid = source_nodes[k]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& ddmap_jk =
          dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_d()[s_gid];

      // (1) Lin(Phi) - dual shape functions --> 0

      // (2) Lin(NSource) - source GP coordinates --> 0

      // (3) Lin(NSource) - source GP coordinates --> 0

      // (4) Lin(dsxideta) - segment end coordinates --> 0

      // (5) Lin(dxdsxi) - source GP Jacobian
      fac = wgt * lm_val[iter] * source_val[k];
      for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
        ddmap_jk[p->first] += fac * (p->second);

      // (6) Lin(dxdsxi) - source GP coordinates --> 0
    }  // loop over source nodes
  }

  // dual shape functions
  else if (shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin)
  {
    // get the D-map as a reference
    std::map<int, double>& ddmap_jk =
        dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_d()[s_gid];

    // integrate LinM and LinD (NO boundary modification)
    for (int k = 0; k < ncol; ++k)
    {
      // global target node ID
      int t_gid = target_nodes[k]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& dmmap_jk =
          dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[t_gid];

      // (1) Lin(Phi) - dual shape functions
      for (int m = 0; m < nrow; ++m)
      {
        fac = wgt * source_val[m] * target_val[k] * dxdsxi;
        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 dualmap.begin();
            p != dualmap.end(); ++p)
        {
          dmmap_jk[p->first] += fac * (p->second)(iter, m);
          if (!bound) ddmap_jk[p->first] += fac * (p->second)(iter, m);
        }
      }

      // (2) Lin(Phi) - source GP coordinates --> 0

      // (3) Lin(NTarget) - target GP coordinates
      fac = wgt * lm_val[iter] * target_deriv(k, 0) * dxdsxi;
      for (CI p = d_target_xi_gp.begin(); p != d_target_xi_gp.end(); ++p)
      {
        dmmap_jk[p->first] += fac * (p->second);
        if (!bound) ddmap_jk[p->first] += fac * (p->second);
      }

      // (4) Lin(dsxideta) - segment end coordinates --> 0

      // (5) Lin(dxdsxi) - source GP Jacobian
      fac = wgt * lm_val[iter] * target_val[k];
      for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
      {
        dmmap_jk[p->first] += fac * (p->second);
        if (!bound) ddmap_jk[p->first] += fac * (p->second);
      }

      // (6) Lin(dxdsxi) - source GP coordinates --> 0
    }  // loop over target nodes
  }  // shape_fcn() switch
}

/*----------------------------------------------------------------------*
 |  Lin D and M matrix entries at GP                         farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_2d_dm_lin(int& iter, bool& bound, bool& linlm,
    Mortar::Element& source_elem, Mortar::Element& target_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseVector& lm_val, Core::LinAlg::SerialDenseMatrix& source_deriv,
    Core::LinAlg::SerialDenseMatrix& target_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    double& jac, double& wgt,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_gp,
    const Core::Gen::Pairedvector<int, double>& derivjac,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  Core::Nodes::Node** source_nodes = nullptr;
  Core::Nodes::Node** target_nodes = nullptr;
  int nrow = source_elem.num_node();
  int ncol = target_elem.num_node();
  source_nodes = source_elem.nodes();
  target_nodes = target_elem.nodes();

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // **************** no edge modification *****************************
  // (and LinM also for edge node modification case)

  // check for linear LM interpolation in quadratic FE
  if (linlm)
  {
    Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[iter]);
    if (!mymrtrnode) FOUR_C_THROW("Null pointer!");
    bool jbound = mymrtrnode->is_on_bound();

    // node j is boundary node
    if (jbound)
    {
      // do nothing as respective D and M entries are zero anyway
    }
    // node j is NO boundary node
    else
    {
      // integrate LinM
      for (int k = 0; k < ncol; ++k)
      {
        // global target node ID
        int t_gid = target_nodes[k]->id();
        double fac = 0.0;

        // get the correct map as a reference
        std::map<int, double>& dmmap_jk =
            dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[t_gid];

        // (1) Lin(Phi) - dual shape functions
        // this vanishes here since there are no deformation-dependent dual functions

        // (2) Lin(NSource) - source GP coordinates
        fac = wgt * lm_deriv(iter, 0) * target_val[k] * jac;
        for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        // (3) Lin(NTarget) - target GP coordinates
        fac = wgt * lm_val[iter] * target_deriv(k, 0) * jac;
        for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        // (4) Lin(dxdsxi) - source GP Jacobian
        fac = wgt * lm_val[iter] * target_val[k];
        for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);
      }  // loop over target nodes

      // integrate LinD
      for (int k = 0; k < nrow; ++k)
      {
        Mortar::Node* mymrtrnode2 = dynamic_cast<Mortar::Node*>(source_nodes[k]);
        if (!mymrtrnode2) FOUR_C_THROW("Null pointer!");
        bool kbound = mymrtrnode2->is_on_bound();

        // global target node ID
        int s_gid = mymrtrnode2->id();
        double fac = 0.0;

        // node k is boundary node
        if (kbound)
        {
          // move entry to derivM (with minus sign)
          // get the correct map as a reference
          std::map<int, double>& dmmap_jk =
              dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[s_gid];

          // (1) Lin(Phi) - dual shape functions
          // this vanishes here since there are no deformation-dependent dual functions

          // (2) Lin(NSource) - source GP coordinates
          fac = wgt * lm_deriv(iter, 0) * source_val[k] * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
            dmmap_jk[p->first] -= fac * (p->second);

          // (3) Lin(NSource) - source GP coordinates
          fac = wgt * lm_val[iter] * source_deriv(k, 0) * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
            dmmap_jk[p->first] -= fac * (p->second);

          // (4) Lin(dxdsxi) - source GP Jacobian
          fac = wgt * lm_val[iter] * source_val[k];
          for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
            dmmap_jk[p->first] -= fac * (p->second);
        }

        // node k is NO boundary node
        else
        {
          // get the correct map as a reference
          std::map<int, double>& ddmap_jk =
              dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_d()[s_gid];

          // (1) Lin(Phi) - dual shape functions
          // this vanishes here since there are no deformation-dependent dual functions

          // (2) Lin(NSource) - source GP coordinates
          fac = wgt * lm_deriv(iter, 0) * source_val[k] * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);

          // (3) Lin(NSource) - source GP coordinates
          fac = wgt * lm_val[iter] * source_deriv(k, 0) * jac;
          for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);

          // (4) Lin(dxdsxi) - source GP Jacobian
          fac = wgt * lm_val[iter] * source_val[k];
          for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
            ddmap_jk[p->first] += fac * (p->second);
        }
      }  // loop over source nodes
    }
  }
  // no linear LM interpolation for quadratic FE
  else
  {
    Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[iter]);
    if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

    int s_gid = mymrtrnode->id();

    // standard shape functions
    if (shape_fcn() == Mortar::shape_standard)
    {
      // integrate LinM
      for (int k = 0; k < ncol; ++k)
      {
        // global target node ID
        int t_gid = target_nodes[k]->id();
        double fac = 0.0;

        // get the correct map as a reference
        std::map<int, double>& dmmap_jk =
            dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[t_gid];

        // (1) Lin(Phi) - dual shape functions
        // this vanishes here since there are no deformation-dependent dual functions

        // (2) Lin(NSource) - source GP coordinates
        fac = wgt * lm_deriv(iter, 0) * target_val[k] * jac;
        for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        // (3) Lin(NTarget) - target GP coordinates
        fac = wgt * lm_val[iter] * target_deriv(k, 0) * jac;
        for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);

        // (4) Lin(dxdsxi) - source GP Jacobian
        fac = wgt * lm_val[iter] * target_val[k];
        for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
          dmmap_jk[p->first] += fac * (p->second);
      }  // loop over target nodes

      // integrate LinD
      for (int k = 0; k < nrow; ++k)
      {
        // global source node ID
        int s_gid = source_nodes[k]->id();
        double fac = 0.0;

        // get the correct map as a reference
        std::map<int, double>& ddmap_jk =
            dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_d()[s_gid];

        // (1) Lin(Phi) - dual shape functions
        // this vanishes here since there are no deformation-dependent dual functions

        // (2) Lin(NSource) - source GP coordinates
        fac = wgt * lm_deriv(iter, 0) * source_val[k] * jac;
        for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          ddmap_jk[p->first] += fac * (p->second);

        // (3) Lin(NSource) - source GP coordinates
        fac = wgt * lm_val[iter] * source_deriv(k, 0) * jac;
        for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
          ddmap_jk[p->first] += fac * (p->second);

        // (4) Lin(dxdsxi) - source GP Jacobian
        fac = wgt * lm_val[iter] * source_val[k];
        for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
          ddmap_jk[p->first] += fac * (p->second);
      }  // loop over source nodes
    }

    // dual shape functions
    else if (shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin)
    {
      // get the D-map as a reference
      std::map<int, double>& ddmap_jk =
          dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_d()[s_gid];

      // integrate LinM and LinD (NO boundary modification)
      for (int k = 0; k < ncol; ++k)
      {
        // global target node ID
        int t_gid = target_nodes[k]->id();
        double fac = 0.0;

        // get the correct map as a reference
        std::map<int, double>& dmmap_jk =
            dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[t_gid];

        // (1) Lin(Phi) - dual shape functions
        for (int m = 0; m < nrow; ++m)
        {
          fac = wgt * source_val[m] * target_val[k] * jac;
          for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                   dualmap.begin();
              p != dualmap.end(); ++p)
          {
            dmmap_jk[p->first] += fac * (p->second)(iter, m);
            if (!bound) ddmap_jk[p->first] += fac * (p->second)(iter, m);
          }
        }

        // (2) Lin(Phi) - source GP coordinates
        fac = wgt * lm_deriv(iter, 0) * target_val[k] * jac;

        for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        {
          dmmap_jk[p->first] += fac * (p->second);
          if (!bound) ddmap_jk[p->first] += fac * (p->second);
        }

        // (3) Lin(NTarget) - target GP coordinates
        fac = wgt * lm_val[iter] * target_deriv(k, 0) * jac;

        for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
        {
          dmmap_jk[p->first] += fac * (p->second);
          if (!bound) ddmap_jk[p->first] += fac * (p->second);
        }

        // (4) Lin(dxdsxi) - source GP Jacobian
        fac = wgt * lm_val[iter] * target_val[k];

        for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
        {
          dmmap_jk[p->first] += fac * (p->second);
          if (!bound) ddmap_jk[p->first] += fac * (p->second);
        }
      }  // loop over target nodes
    }  // shape_fcn() switch
  }
  // compute segment D/M linearization *********************************
}

/*----------------------------------------------------------------------*
 |  Lin D and M matrix entries at GP - pwlin                 farah 12/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_3d_dm_quad_pwlin_lin(int& iter, Mortar::Element& source_elem,
    Mortar::Element& sintele, Mortar::Element& target_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseVector& lmintval, Core::LinAlg::SerialDenseMatrix& source_deriv,
    Core::LinAlg::SerialDenseMatrix& target_deriv, Core::LinAlg::SerialDenseMatrix& lmintderiv,
    double& wgt, double& jac,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dp_source_xigp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dp_target_xi_gp,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap)
{
  const int nrow = source_elem.num_node();
  const int ncol = target_elem.num_node();

  // get source element nodes themselves
  Core::Nodes::Node** sintnodes = sintele.nodes();

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // **************** no edge modification *****************************
  // (and LinM also for edge node modification case)

  Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(sintnodes[iter]);
  if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

  // integrate LinM
  for (int k = 0; k < ncol; ++k)
  {
    // global target node ID
    int t_gid = target_elem.nodes()[k]->id();
    double fac = 0.0;

    // get the correct map as a reference
    std::map<int, double>& dmmap_jk =
        dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[t_gid];

    // (1) Lin(Phi) - dual shape functions
    // this vanishes here since there are no deformation-dependent dual functions

    // (2) Lin(NSource) - source GP coordinates
    fac = wgt * lmintderiv(iter, 0) * target_val[k] * jac;
    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      dmmap_jk[p->first] += fac * (p->second);

    fac = wgt * lmintderiv(iter, 1) * target_val[k] * jac;
    for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
      dmmap_jk[p->first] += fac * (p->second);

    // (3) Lin(NTarget) - target GP coordinates
    fac = wgt * lmintval[iter] * target_deriv(k, 0) * jac;
    for (CI p = dp_target_xi_gp[0].begin(); p != dp_target_xi_gp[0].end(); ++p)
      dmmap_jk[p->first] += fac * (p->second);

    fac = wgt * lmintval[iter] * target_deriv(k, 1) * jac;
    for (CI p = dp_target_xi_gp[1].begin(); p != dp_target_xi_gp[1].end(); ++p)
      dmmap_jk[p->first] += fac * (p->second);

    // (4) Lin(dsxideta) - intcell GP Jacobian
    fac = wgt * lmintval[iter] * target_val[k];
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
      dmmap_jk[p->first] += fac * (p->second);
  }  // loop over target nodes

  // integrate LinD
  for (int k = 0; k < nrow; ++k)
  {
    // global target node ID
    int s_gid = source_elem.nodes()[k]->id();
    double fac = 0.0;

    // get the correct map as a reference
    std::map<int, double>& ddmap_jk =
        dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_d()[s_gid];

    // (1) Lin(Phi) - dual shape functions
    // this vanishes here since there are no deformation-dependent dual functions

    // (2) Lin(NSource) - source GP coordinates
    fac = wgt * lmintderiv(iter, 0) * source_val[k] * jac;
    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      ddmap_jk[p->first] += fac * (p->second);

    fac = wgt * lmintderiv(iter, 1) * source_val[k] * jac;
    for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
      ddmap_jk[p->first] += fac * (p->second);

    // (3) Lin(NSource) - source GP coordinates
    fac = wgt * lmintval[iter] * source_deriv(k, 0) * jac;
    for (CI p = dp_source_xigp[0].begin(); p != dp_source_xigp[0].end(); ++p)
      ddmap_jk[p->first] += fac * (p->second);

    fac = wgt * lmintval[iter] * source_deriv(k, 1) * jac;
    for (CI p = dp_source_xigp[1].begin(); p != dp_source_xigp[1].end(); ++p)
      ddmap_jk[p->first] += fac * (p->second);

    // (4) Lin(dsxideta) - intcell GP Jacobian
    fac = wgt * lmintval[iter] * source_val[k];
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
      ddmap_jk[p->first] += fac * (p->second);
  }  // loop over source nodes
}

/*----------------------------------------------------------------------*
 |  Lin D and M matrix entries at GP                         farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_3d_dm_quad_lin(bool& duallin, Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& svalmod, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseVector& lm_val, Core::LinAlg::SerialDenseMatrix& source_deriv,
    Core::LinAlg::SerialDenseMatrix& target_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    double& wgt, double& jac,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dp_source_xigp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dp_target_xi_gp,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, bool dualquad3d)
{
  const int nrow = source_elem.num_node();
  const int ncol = target_elem.num_node();

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();

  std::vector<Mortar::Node*> source_mortar_nodes(nrow);
  for (int i = 0; i < nrow; ++i)
    source_mortar_nodes[i] = dynamic_cast<Mortar::Node*>(source_nodes[i]);

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // **************** no edge modification *****************************
  // (and LinM also for edge node modification case)

  // compute cell D/M linearization ************************************
  // CASE 1: Standard LM shape functions and quadratic interpolation
  if (shape_fcn() == Mortar::shape_standard && lag_mult_quad() == Mortar::lagmult_quad)
  {
    static double fac1 = 0.;
    static double fac2 = 0.;
    // (1) Lin(Phi) - dual shape functions
    // this vanishes here since there are no deformation-dependent dual functions

    // (2) Lin(NSource) - source GP coordinates
    // (3) Lin(NSource) - source GP coordinates
    for (int d = 0; d < 2; ++d)
      for (CI p = dp_source_xigp[d].begin(); p != dp_source_xigp[d].end(); ++p)
      {
        Core::LinAlg::SerialDenseMatrix& dMderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_dderiv()[p->first];
        for (int j = 0; j < nrow; ++j)
          for (int k = 0; k < nrow; ++k)
          {
            fac1 = wgt * lm_deriv(j, d) * source_val[k] * jac;
            fac2 = wgt * lm_val[j] * source_deriv(k, d) * jac;
            dMderiv(j, k) += (fac1 + fac2) * (p->second);
          }
      }
    // (4) Lin(dsxideta) - intcell GP Jacobian
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
    {
      Core::LinAlg::SerialDenseMatrix& dMderiv =
          dynamic_cast<CONTACT::Element&>(source_elem).get_dderiv()[p->first];
      for (int j = 0; j < nrow; ++j)
        for (int k = 0; k < nrow; ++k)
          dMderiv(j, k) += wgt * lm_val[j] * source_val[k] * (p->second);
    }

    // (1) Lin(Phi) - dual shape functions
    // this vanishes here since there are no deformation-dependent dual functions

    // (2) Lin(NSource) - source GP coordinates
    for (int d = 0; d < 2; ++d)
      for (CI p = dp_source_xigp[d].begin(); p != dp_source_xigp[d].end(); ++p)
      {
        Core::LinAlg::SerialDenseMatrix& mMderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_mderiv()[p->first];
        for (int j = 0; j < nrow; ++j)
          for (int k = 0; k < ncol; ++k)
            mMderiv(j, k) += wgt * lm_deriv(j, d) * target_val[k] * jac * (p->second);
      }

    // (3) Lin(NTarget) - target GP coordinates
    for (int d = 0; d < 2; ++d)
      for (CI p = dp_target_xi_gp[d].begin(); p != dp_target_xi_gp[d].end(); ++p)
      {
        Core::LinAlg::SerialDenseMatrix& mMderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_mderiv()[p->first];
        for (int j = 0; j < nrow; ++j)
          for (int k = 0; k < ncol; ++k)
            mMderiv(j, k) += wgt * lm_val[j] * target_deriv(k, d) * jac * (p->second);
      }

    // (4) Lin(dsxideta) - intcell GP Jacobian
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
    {
      Core::LinAlg::SerialDenseMatrix& mMderiv =
          dynamic_cast<CONTACT::Element&>(source_elem).get_mderiv()[p->first];
      for (int j = 0; j < nrow; ++j)
        for (int k = 0; k < ncol; ++k)
          mMderiv(j, k) += wgt * lm_val[j] * target_val[k] * (p->second);
    }
  }

  // CASE 2: Standard LM shape functions and linear interpolation
  // CASE 5: dual LM shape functions and linear interpolation
  // (this has to be treated separately here for LinDM because of bound)
  else if ((shape_fcn() == Mortar::shape_standard || shape_fcn() == Mortar::shape_dual ||
               shape_fcn() == Mortar::shape_petrovgalerkin) &&
           lag_mult_quad() == Mortar::lagmult_lin)
  {
    // integrate LinD
    for (int j = 0; j < nrow; ++j)
    {
      Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[j]);
      if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

      // node j is boundary node
      if (mymrtrnode->set_bound())
      {
        // do nothing as respective D and M entries are zero anyway
      }

      // node j is NO boundary node
      else
      {
        // integrate LinM
        for (int k = 0; k < ncol; ++k)
        {
          // global target node ID
          int t_gid = target_elem.nodes()[k]->id();
          static double fac = 0.0;

          // get the correct map as a reference
          std::map<int, double>& dmmap_jk =
              dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[t_gid];

          // (1) Lin(Phi) - dual shape functions
          if (duallin)
          {
            using CIM =
                Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator;
            for (CIM p = dualmap.begin(); p != dualmap.end(); ++p)
            {
              double lmderiv_d = 0.0;
              for (int m = 0; m < nrow; ++m) lmderiv_d += (p->second)(j, m) * source_val[m];
              fac = wgt * lmderiv_d * target_val[k] * jac;
              dmmap_jk[p->first] += fac;
            }
          }

          // (2) Lin(NSource) - source GP coordinates
          fac = wgt * lm_deriv(j, 0) * target_val[k] * jac;
          for (CI p = dp_source_xigp[0].begin(); p != dp_source_xigp[0].end(); ++p)
            dmmap_jk[p->first] += fac * (p->second);

          fac = wgt * lm_deriv(j, 1) * target_val[k] * jac;
          for (CI p = dp_source_xigp[1].begin(); p != dp_source_xigp[1].end(); ++p)
            dmmap_jk[p->first] += fac * (p->second);

          // (3) Lin(NTarget) - target GP coordinates
          fac = wgt * lm_val[j] * target_deriv(k, 0) * jac;
          for (CI p = dp_target_xi_gp[0].begin(); p != dp_target_xi_gp[0].end(); ++p)
            dmmap_jk[p->first] += fac * (p->second);

          fac = wgt * lm_val[j] * target_deriv(k, 1) * jac;
          for (CI p = dp_target_xi_gp[1].begin(); p != dp_target_xi_gp[1].end(); ++p)
            dmmap_jk[p->first] += fac * (p->second);

          // (4) Lin(dsxideta) - intcell GP Jacobian
          fac = wgt * lm_val[j] * target_val[k];
          for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
            dmmap_jk[p->first] += fac * (p->second);
        }  // loop over target nodes

        for (int k = 0; k < nrow; ++k)
        {
          Mortar::Node* mymrtrnode2 = dynamic_cast<Mortar::Node*>(source_nodes[k]);
          if (!mymrtrnode2) FOUR_C_THROW("Null pointer!");

          // global target node ID
          int s_gid = mymrtrnode2->id();
          static double fac = 0.0;

          // node k is boundary node
          if (mymrtrnode2->is_on_bound())
          {
            // move entry to derivM (with minus sign)
            // get the correct map as a reference
            std::map<int, double>& dmmap_jk =
                dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_m()[s_gid];

            // (1) Lin(Phi) - dual shape functions
            if (duallin)
            {
              using CIM =
                  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator;
              fac = wgt * source_val[k] * jac;
              for (CIM p = dualmap.begin(); p != dualmap.end(); ++p)
                for (int m = 0; m < nrow; ++m)
                  dmmap_jk[p->first] -= fac * (p->second)(j, m) * source_val[m];

              for (CIM p = dualmap.begin(); p != dualmap.end(); ++p)
              {
                double lmderiv_d = 0.0;
                for (int m = 0; m < nrow; ++m) lmderiv_d += (p->second)(j, m) * source_val[m];
                fac = wgt * lmderiv_d * target_val[k] * jac;
                dmmap_jk[p->first] -= fac;
              }
            }

            // (2) Lin(NSource) - source GP coordinates
            fac = wgt * lm_deriv(j, 0) * source_val[k] * jac;
            for (CI p = dp_source_xigp[0].begin(); p != dp_source_xigp[0].end(); ++p)
              dmmap_jk[p->first] -= fac * (p->second);

            fac = wgt * lm_deriv(j, 1) * source_val[k] * jac;
            for (CI p = dp_source_xigp[1].begin(); p != dp_source_xigp[1].end(); ++p)
              dmmap_jk[p->first] -= fac * (p->second);

            // (3) Lin(NSource) - source GP coordinates
            fac = wgt * lm_val[j] * source_deriv(k, 0) * jac;
            for (CI p = dp_source_xigp[0].begin(); p != dp_source_xigp[0].end(); ++p)
              dmmap_jk[p->first] -= fac * (p->second);

            fac = wgt * lm_val[j] * source_deriv(k, 1) * jac;
            for (CI p = dp_source_xigp[1].begin(); p != dp_source_xigp[1].end(); ++p)
              dmmap_jk[p->first] -= fac * (p->second);

            // (4) Lin(dsxideta) - intcell GP Jacobian
            fac = wgt * lm_val[j] * source_val[k];
            for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
              dmmap_jk[p->first] -= fac * (p->second);
          }

          // node k is NO boundary node
          else
          {
            // get the correct map as a reference
            std::map<int, double>& ddmap_jk =
                dynamic_cast<CONTACT::Node*>(mymrtrnode)->data().get_deriv_d()[s_gid];

            // (1) Lin(Phi) - dual shape functions
            if (duallin)
            {
              using CIM =
                  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator;
              fac = wgt * source_val[k] * jac;
              for (CIM p = dualmap.begin(); p != dualmap.end(); ++p)
                for (int m = 0; m < nrow; ++m)
                  ddmap_jk[p->first] += fac * (p->second)(j, m) * source_val[m];

              for (CIM p = dualmap.begin(); p != dualmap.end(); ++p)
              {
                double lmderiv_d = 0.0;
                for (int m = 0; m < nrow; ++m) lmderiv_d += (p->second)(j, m) * source_val[m];
                fac = wgt * lmderiv_d * target_val[k] * jac;
                ddmap_jk[p->first] += fac;
              }
            }

            // (2) Lin(NSource) - source GP coordinates
            fac = wgt * lm_deriv(j, 0) * source_val[k] * jac;
            for (CI p = dp_source_xigp[0].begin(); p != dp_source_xigp[0].end(); ++p)
              ddmap_jk[p->first] += fac * (p->second);

            fac = wgt * lm_deriv(j, 1) * source_val[k] * jac;
            for (CI p = dp_source_xigp[1].begin(); p != dp_source_xigp[1].end(); ++p)
              ddmap_jk[p->first] += fac * (p->second);

            // (3) Lin(NSource) - source GP coordinates
            fac = wgt * lm_val[j] * source_deriv(k, 0) * jac;
            for (CI p = dp_source_xigp[0].begin(); p != dp_source_xigp[0].end(); ++p)
              ddmap_jk[p->first] += fac * (p->second);

            fac = wgt * lm_val[j] * source_deriv(k, 1) * jac;
            for (CI p = dp_source_xigp[1].begin(); p != dp_source_xigp[1].end(); ++p)
              ddmap_jk[p->first] += fac * (p->second);

            // (4) Lin(dsxideta) - intcell GP Jacobian
            fac = wgt * lm_val[j] * source_val[k];
            for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
              ddmap_jk[p->first] += fac * (p->second);
          }
        }  // loop over source nodes
      }
    }
  }
  // CASE 4: Dual LM shape functions and quadratic interpolation
  else if ((shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin) &&
           lag_mult_quad() == Mortar::lagmult_quad)
  {
    double fac = 0.;
    // (1) Lin(Phi) - dual shape functions
    if (duallin)
    {
      for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
               dualmap.begin();
          p != dualmap.end(); ++p)
      {
        Core::LinAlg::SerialDenseMatrix& dDderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_dderiv()[p->first];
        Core::LinAlg::SerialDenseMatrix& dMderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_mderiv()[p->first];
        for (int m = 0; m < nrow; ++m)
          for (int k = 0; k < ncol; ++k)
          {
            if (dualquad3d)
              fac = wgt * svalmod[m] * target_val[k] * jac;
            else
              fac = wgt * source_val[m] * target_val[k] * jac;
            for (int j = 0; j < nrow; ++j)
            {
              dMderiv(j, k) += fac * (p->second)(j, m);
              dDderiv(j, j) += fac * (p->second)(j, m);
            }
          }
      }
    }

    // (2) Lin(Phi) - source GP coordinates
    for (int d = 0; d < 2; ++d)
      for (CI p = dp_source_xigp[d].begin(); p != dp_source_xigp[d].end(); ++p)
      {
        Core::LinAlg::SerialDenseMatrix& dDderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_dderiv()[p->first];
        Core::LinAlg::SerialDenseMatrix& dMderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_mderiv()[p->first];
        for (int j = 0; j < nrow; ++j)
          for (int k = 0; k < ncol; ++k)
          {
            fac = wgt * lm_deriv(j, d) * target_val[k] * jac;
            dMderiv(j, k) += fac * (p->second);
            dDderiv(j, j) += fac * (p->second);
          }
      }

    // (3) Lin(NTarget) - target GP coordinates
    for (int d = 0; d < 2; ++d)
      for (CI p = dp_target_xi_gp[d].begin(); p != dp_target_xi_gp[d].end(); ++p)
      {
        Core::LinAlg::SerialDenseMatrix& dDderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_dderiv()[p->first];
        Core::LinAlg::SerialDenseMatrix& dMderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_mderiv()[p->first];
        for (int j = 0; j < nrow; ++j)
          for (int k = 0; k < ncol; ++k)
          {
            fac = wgt * lm_val[j] * target_deriv(k, d) * jac;
            dMderiv(j, k) += fac * (p->second);
            dDderiv(j, j) += fac * (p->second);
          }
      }

    // (4) Lin(dsxideta) - intcell GP Jacobian
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
    {
      Core::LinAlg::SerialDenseMatrix& dDderiv =
          dynamic_cast<CONTACT::Element&>(source_elem).get_dderiv()[p->first];
      Core::LinAlg::SerialDenseMatrix& dMderiv =
          dynamic_cast<CONTACT::Element&>(source_elem).get_mderiv()[p->first];
      for (int j = 0; j < nrow; ++j)
        for (int k = 0; k < ncol; ++k)
        {
          fac = wgt * lm_val[j] * target_val[k];
          dMderiv(j, k) += fac * (p->second);
          dDderiv(j, j) += fac * (p->second);
        }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Lin D and M matrix entries at GP                         farah 09/13|
 *----------------------------------------------------------------------*/
void CONTACT::Integrator::gp_3d_dm_lin(Mortar::Element& source_elem, Mortar::Element& target_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseVector& lm_val, Core::LinAlg::SerialDenseMatrix& source_deriv,
    Core::LinAlg::SerialDenseMatrix& target_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    double& wgt, double& jac, std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_gp,
    Core::Gen::Pairedvector<int, double>& jacintcellmap,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  const int nrow = source_elem.num_node();
  const int ncol = target_elem.num_node();

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // standard shape functions
  if (shape_fcn() == Mortar::shape_standard)
  {
    // integrate LinM
    // (1) Lin(Phi) - dual shape functions
    // this vanishes here since there are no deformation-dependent dual functions

    // (2) Lin(NSource) - source GP coordinates
    for (int d = 0; d < 2; ++d)
      for (CI p = d_source_xi_gp[d].begin(); p != d_source_xi_gp[d].end(); ++p)
      {
        Core::LinAlg::SerialDenseMatrix& dMderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_mderiv()[p->first];
        for (int j = 0; j < nrow; ++j)
          for (int k = 0; k < ncol; ++k)
            dMderiv(j, k) += wgt * lm_deriv(j, d) * target_val[k] * jac * (p->second);
      }

    // (3) Lin(NTarget) - target GP coordinates
    for (int d = 0; d < 2; ++d)
      for (CI p = d_target_xi_gp[d].begin(); p != d_target_xi_gp[d].end(); ++p)
      {
        Core::LinAlg::SerialDenseMatrix& dMderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_mderiv()[p->first];
        for (int j = 0; j < nrow; ++j)
          for (int k = 0; k < ncol; ++k)
            dMderiv(j, k) += wgt * lm_val[j] * target_deriv(k, d) * jac * (p->second);
      }

    // (4) Lin(dsxideta) - intcell GP Jacobian
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
    {
      Core::LinAlg::SerialDenseMatrix& dMderiv =
          dynamic_cast<CONTACT::Element&>(source_elem).get_mderiv()[p->first];
      for (int j = 0; j < nrow; ++j)
        for (int k = 0; k < ncol; ++k)
          dMderiv(j, k) += wgt * lm_val[j] * target_val[k] * (p->second);
    }

    // integrate LinD
    // (1) Lin(Phi) - dual shape functions
    // this vanishes here since there are no deformation-dependent dual functions

    // (2) Lin(NSource) - source GP coordinates
    // (3) Lin(NSource) - source GP coordinates
    for (int d = 0; d < 2; ++d)
      for (CI p = d_source_xi_gp[d].begin(); p != d_source_xi_gp[d].end(); ++p)
      {
        Core::LinAlg::SerialDenseMatrix& dDderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_dderiv()[p->first];
        for (int j = 0; j < nrow; ++j)
          for (int k = 0; k < nrow; ++k)
            dDderiv(j, k) += (wgt * lm_deriv(j, d) * source_val[k] * jac +
                                 wgt * lm_val[j] * source_deriv(k, d) * jac) *
                             (p->second);
      }

    // (4) Lin(dsxideta) - intcell GP Jacobian
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
    {
      Core::LinAlg::SerialDenseMatrix& dDderiv =
          dynamic_cast<CONTACT::Element&>(source_elem).get_dderiv()[p->first];
      for (int j = 0; j < nrow; ++j)
        for (int k = 0; k < nrow; ++k)
          dDderiv(j, k) += wgt * lm_val[j] * source_val[k] * (p->second);
    }
  }

  // dual shape functions
  else if (shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin)
  {
    double fac = 0.;

    // (1) Lin(Phi) - dual shape functions
    if (dualmap.size() > 0)
      for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
               dualmap.begin();
          p != dualmap.end(); ++p)
      {
        Core::LinAlg::SerialDenseMatrix& dDderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_dderiv()[p->first];
        Core::LinAlg::SerialDenseMatrix& dMderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_mderiv()[p->first];
        for (int j = 0; j < nrow; ++j)
          for (int k = 0; k < ncol; ++k)
            for (int m = 0; m < nrow; ++m)
            {
              fac = wgt * source_val[m] * target_val[k] * jac * (p->second)(j, m);
              dDderiv(j, j) += fac;
              dMderiv(j, k) += fac;
            }
      }

    // (2) Lin(Phi) - source GP coordinates
    for (int d = 0; d < 2; ++d)
      for (CI p = d_source_xi_gp[d].begin(); p != d_source_xi_gp[d].end(); ++p)
      {
        Core::LinAlg::SerialDenseMatrix& dDderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_dderiv()[p->first];
        Core::LinAlg::SerialDenseMatrix& dMderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_mderiv()[p->first];
        for (int j = 0; j < nrow; ++j)
          for (int k = 0; k < ncol; ++k)
          {
            fac = wgt * lm_deriv(j, d) * target_val[k] * jac * (p->second);
            dDderiv(j, j) += fac;
            dMderiv(j, k) += fac;
          }
      }

    // (3) Lin(NTarget) - target GP coordinates
    for (int d = 0; d < 2; ++d)
      for (CI p = d_target_xi_gp[d].begin(); p != d_target_xi_gp[d].end(); ++p)
      {
        Core::LinAlg::SerialDenseMatrix& dDderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_dderiv()[p->first];
        Core::LinAlg::SerialDenseMatrix& dMderiv =
            dynamic_cast<CONTACT::Element&>(source_elem).get_mderiv()[p->first];
        for (int j = 0; j < nrow; ++j)
          for (int k = 0; k < ncol; ++k)
          {
            fac = wgt * lm_val[j] * target_deriv(k, d) * jac * (p->second);
            dDderiv(j, j) += fac;
            dMderiv(j, k) += fac;
          }
      }

    // (4) Lin(dsxideta) - intcell GP Jacobian
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
    {
      Core::LinAlg::SerialDenseMatrix& dDderiv =
          dynamic_cast<CONTACT::Element&>(source_elem).get_dderiv()[p->first];
      Core::LinAlg::SerialDenseMatrix& dMderiv =
          dynamic_cast<CONTACT::Element&>(source_elem).get_mderiv()[p->first];
      for (int j = 0; j < nrow; ++j)
        for (int k = 0; k < ncol; ++k)
        {
          fac = wgt * lm_val[j] * target_val[k] * (p->second);
          dDderiv(j, j) += fac;
          dMderiv(j, k) += fac;
        }
    }
  }  // shape_fcn() switch
  // compute segment D/M linearization *********************************
}

/*----------------------------------------------------------------------*
 |  Compute entries for D2 matrix at GP                      farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_d2(Mortar::Element& source_elem, Mortar::Element& target_elem,
    Core::LinAlg::SerialDenseVector& lm2val, Core::LinAlg::SerialDenseVector& m2val, double& jac,
    double& wgt, MPI_Comm comm)
{
  int ncol = target_elem.num_node();
  int ndof = n_dim();

  // get source element nodes themselves
  Core::Nodes::Node** target_nodes = target_elem.nodes();
  if (!target_nodes) FOUR_C_THROW("Null pointer!");

  if (shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin)
  {
    for (int j = 0; j < ncol; ++j)
    {
      CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(target_nodes[j]);

      // IMPORTANT: assembling to node is only allowed for target nodes
      //            associated to owned source elements. This results
      //            to an unique entry distribution!
      if (source_elem.owner() == Core::Communication::my_mpi_rank(comm))
      {
        for (int jdof = 0; jdof < ndof; ++jdof)
        {
          for (int k = 0; k < ncol; ++k)
          {
            CONTACT::FriNode* target_node = dynamic_cast<CONTACT::FriNode*>(target_nodes[k]);

            for (int kdof = 0; kdof < ndof; ++kdof)
            {
              int col = target_node->dofs()[kdof];

              // multiply the two shape functions
              double prod = lm2val[j] * m2val[k] * jac * wgt;

              if ((jdof == kdof) and (j == k))
              {
                if (abs(prod) > MORTARINTTOL) cnode->involved_m() = true;
                if (abs(prod) > MORTARINTTOL) cnode->add_d2_value(jdof, col, prod);
              }
            }
          }
        }
      }
    }
  }
  else
    FOUR_C_THROW("Both-sided wear just for dual shape functions!");
}



/*----------------------------------------------------------------------*
 |  Compute wear at GP (for expl/impl algor.)                farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_2d_wear(Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseMatrix& target_deriv, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& lm_deriv, Core::LinAlg::SerialDenseMatrix& lagmult,
    double* gpn, double& jac, double& wgt, double* jumpval, double* wearval,
    Core::Gen::Pairedvector<int, double>& dsliptmatrixgp,
    Core::Gen::Pairedvector<int, double>& dweargp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_gp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  const int nrow = source_elem.num_node();
  const int ncol = target_elem.num_node();
  const int ndof = n_dim();

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  Core::Nodes::Node** target_nodes = target_elem.nodes();

  int linsize = 0;
  for (int i = 0; i < source_elem.num_node(); ++i)
    linsize += dynamic_cast<Node*>(source_elem.nodes()[i])->get_linsize();
  linsize = linsize * 2;

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  //***********************************************************************
  // Here, the tangential relative slip increment is used and NOT the
  // nodal weighted tangential relative slip increment !!!
  // The reason for that is the slip for wear calculation is written
  // within the integral --> no double weighting allowed !
  // The wearcoefficient is not included in this calculation
  //***********************************************************************
  // for wearval
  std::array<double, 2> gpt = {0.0, 0.0};
  std::array<double, 2> gplm = {0.0, 0.0};
  std::array<double, 2> sgpjump = {0.0, 0.0};
  std::array<double, 2> mgpjump = {0.0, 0.0};
  std::array<double, 2> jump = {0.0, 0.0};

  // for linearization
  double lm_lin = 0.0;
  double lengtht = 0.0;

  for (int i = 0; i < nrow; ++i)
  {
    CONTACT::Node* myconode = dynamic_cast<CONTACT::Node*>(source_nodes[i]);

    // nodal tangent interpolation
    gpt[0] += source_val[i] * myconode->data().txi()[0];
    gpt[1] += source_val[i] * myconode->data().txi()[1];

    // delta D
    sgpjump[0] += source_val[i] *
                  (source_elem.get_nodal_coords(0, i) - (source_elem.get_nodal_coords_old(0, i)));
    sgpjump[1] += source_val[i] *
                  (source_elem.get_nodal_coords(1, i) - (source_elem.get_nodal_coords_old(1, i)));

    // LM interpolation
    gplm[0] += lm_val[i] * ((lagmult)(0, i));
    gplm[1] += lm_val[i] * ((lagmult)(1, i));
  }

  // normalize interpolated GP tangent back to length 1.0 !!!
  lengtht = sqrt(gpt[0] * gpt[0] + gpt[1] * gpt[1]);
  if (abs(lengtht) < 1.0e-12) FOUR_C_THROW("Divide by zero!");

  for (int i = 0; i < n_dim(); i++) gpt[i] /= lengtht;

  // interpolation of target GP jumps (relative displacement increment)
  for (int i = 0; i < ncol; ++i)
  {
    mgpjump[0] += target_val[i] *
                  (target_elem.get_nodal_coords(0, i) - target_elem.get_nodal_coords_old(0, i));
    mgpjump[1] += target_val[i] *
                  (target_elem.get_nodal_coords(1, i) - target_elem.get_nodal_coords_old(1, i));
  }

  // jump
  jump[0] = sgpjump[0] - mgpjump[0];
  jump[1] = sgpjump[1] - mgpjump[1];

  // evaluate wear
  // normal contact stress -- normal LM value
  for (int i = 0; i < n_dim(); ++i)
  {
    wearval[0] += gpn[i] * gplm[i];
    lm_lin += gpn[i] * gplm[i];  // required for linearization
  }

  // value of relative tangential jump
  for (int i = 0; i < n_dim(); ++i) jumpval[0] += gpt[i] * jump[i];

  // steady state slip
  if (sswear_) jumpval[0] = ssslip_;

  // no jump --> no wear
  if (abs(jumpval[0]) < 1e-12) return;

  // product
  // use non-abs value for implicit-wear algorithm
  // just for simple linear. maybe we change this in future
  if (wearimpl_ and wear_type() != Wear::wear_primvar)
    wearval[0] = (wearval[0]) * abs(jumpval[0]);
  else
    wearval[0] = abs(wearval[0]) * abs(jumpval[0]);

  // compute segment wear vector ***************************************
  // nrow represents the source side dofs !!!
  for (int j = 0; j < nrow; ++j)
  {
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(source_nodes[j]);

    double prod = 0.0;
    if (shape_fcn() == Mortar::shape_petrovgalerkin)
      prod = source_val[j] * wearval[0] * jac * wgt;
    else
      prod = lm_val[j] * wearval[0] * jac * wgt;

    // add current Gauss point's contribution to wseg
    cnode->add_delta_weighted_wear_value(prod);
  }

  //****************************************************************
  //   linearization for implicit algorithms
  //****************************************************************
  if (wearimpl_ || wear_type() == Wear::wear_primvar)
  {
    // evaluate the GP wear function derivatives
    Core::Gen::Pairedvector<int, double> ddualgp_x(ndof * ncol + linsize);
    Core::Gen::Pairedvector<int, double> ddualgp_y(ndof * ncol + linsize);

    Core::Gen::Pairedvector<int, double> ddualgp_x_sxi(ndof * ncol + linsize);
    Core::Gen::Pairedvector<int, double> ddualgp_y_sxi(ndof * ncol + linsize);

    // lin. abs(x) = x/abs(x) * lin x.
    double xabsx = (jumpval[0] / abs(jumpval[0])) * lm_lin;
    double xabsxT = (jumpval[0] / abs(jumpval[0]));

    // **********************************************************************
    // (1) Lin of normal for LM -- deriv normal maps from weighted gap lin.
    for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gplm[0] * (p->second);

    for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gplm[1] * (p->second);

    // **********************************************************************
    // (2) Lin. of dual shape function of LM mult.
    for (int i = 0; i < nrow; ++i)
    {
      for (int m = 0; m < nrow; ++m)
      {
        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 dualmap.begin();
            p != dualmap.end(); ++p)
        {
          ddualgp_x[p->first] += ((lagmult)(0, m)) * source_val[m] * (p->second)(i, m);
          ddualgp_y[p->first] += ((lagmult)(1, m)) * source_val[m] * (p->second)(i, m);
        }
      }
    }
    for (CI p = ddualgp_x.begin(); p != ddualgp_x.end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gpn[0] * (p->second);
    for (CI p = ddualgp_y.begin(); p != ddualgp_y.end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gpn[1] * (p->second);

    // LM deriv
    for (int i = 0; i < nrow; ++i)
    {
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      {
        ddualgp_x_sxi[p->first] += ((lagmult)(0, i)) * lm_deriv(i, 0) * (p->second);
        ddualgp_y_sxi[p->first] += ((lagmult)(1, i)) * lm_deriv(i, 0) * (p->second);
      }
    }
    for (CI p = ddualgp_x_sxi.begin(); p != ddualgp_x_sxi.end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gpn[0] * (p->second);
    for (CI p = ddualgp_y_sxi.begin(); p != ddualgp_y_sxi.end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gpn[1] * (p->second);


    // **********************************************************************
    // (3) absolute incremental slip linearization:
    // (a) build directional derivative of source GP tagent (non-unit)
    Core::Gen::Pairedvector<int, double> dmap_txsl_gp(ndof * ncol + linsize);
    Core::Gen::Pairedvector<int, double> dmap_tysl_gp(ndof * ncol + linsize);

    for (int i = 0; i < nrow; ++i)
    {
      CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(source_nodes[i]);

      Core::Gen::Pairedvector<int, double>& dmap_txsl_i =
          dynamic_cast<CONTACT::Node*>(cnode)->data().get_deriv_txi()[0];
      Core::Gen::Pairedvector<int, double>& dmap_tysl_i =
          dynamic_cast<CONTACT::Node*>(cnode)->data().get_deriv_txi()[1];

      for (CI p = dmap_txsl_i.begin(); p != dmap_txsl_i.end(); ++p)
        dmap_txsl_gp[p->first] += source_val[i] * (p->second);
      for (CI p = dmap_tysl_i.begin(); p != dmap_tysl_i.end(); ++p)
        dmap_tysl_gp[p->first] += source_val[i] * (p->second);

      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      {
        double valx = source_deriv(i, 0) * dynamic_cast<CONTACT::Node*>(cnode)->data().txi()[0];
        dmap_txsl_gp[p->first] += valx * (p->second);
        double valy = source_deriv(i, 0) * dynamic_cast<CONTACT::Node*>(cnode)->data().txi()[1];
        dmap_tysl_gp[p->first] += valy * (p->second);
      }
    }

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
      dweargp[p->first] += xabsx * jump[0] * (p->second);

    for (CI p = dmap_tysl_gp_unit.begin(); p != dmap_tysl_gp_unit.end(); ++p)
      dweargp[p->first] += xabsx * jump[1] * (p->second);

    // add tangent lin. to slip linearization for wear Tmatrix
    for (CI p = dmap_txsl_gp_unit.begin(); p != dmap_txsl_gp_unit.end(); ++p)
      dsliptmatrixgp[p->first] += xabsxT * jump[0] * (p->second);

    for (CI p = dmap_tysl_gp_unit.begin(); p != dmap_tysl_gp_unit.end(); ++p)
      dsliptmatrixgp[p->first] += xabsxT * jump[1] * (p->second);

    // **********************************************************************
    // (c) build directional derivative of jump
    Core::Gen::Pairedvector<int, double> dmap_slcoord_gp_x(ndof * ncol + linsize);
    Core::Gen::Pairedvector<int, double> dmap_slcoord_gp_y(ndof * ncol + linsize);

    Core::Gen::Pairedvector<int, double> dmap_mcoord_gp_x(ndof * ncol + linsize);
    Core::Gen::Pairedvector<int, double> dmap_mcoord_gp_y(ndof * ncol + linsize);

    Core::Gen::Pairedvector<int, double> dmap_coord_x(ndof * ncol + linsize);
    Core::Gen::Pairedvector<int, double> dmap_coord_y(ndof * ncol + linsize);

    // lin source part -- source_xi
    for (int i = 0; i < nrow; ++i)
    {
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      {
        double valx = source_deriv(i, 0) * (source_elem.get_nodal_coords(0, i) -
                                               (source_elem.get_nodal_coords_old(0, i)));
        dmap_slcoord_gp_x[p->first] += valx * (p->second);
        double valy = source_deriv(i, 0) * (source_elem.get_nodal_coords(1, i) -
                                               (source_elem.get_nodal_coords_old(1, i)));
        dmap_slcoord_gp_y[p->first] += valy * (p->second);
      }
    }

    // lin target part -- target_xi
    for (int i = 0; i < ncol; ++i)
    {
      for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
      {
        double valx = target_deriv(i, 0) * (target_elem.get_nodal_coords(0, i) -
                                               (target_elem.get_nodal_coords_old(0, i)));
        dmap_mcoord_gp_x[p->first] += valx * (p->second);
        double valy = target_deriv(i, 0) * (target_elem.get_nodal_coords(1, i) -
                                               (target_elem.get_nodal_coords_old(1, i)));
        dmap_mcoord_gp_y[p->first] += valy * (p->second);
      }
    }

    // deriv source x-coords
    for (int i = 0; i < nrow; ++i)
    {
      CONTACT::Node* source_node = dynamic_cast<CONTACT::Node*>(source_nodes[i]);

      dmap_slcoord_gp_x[source_node->dofs()[0]] += source_val[i];
      dmap_slcoord_gp_y[source_node->dofs()[1]] += source_val[i];
    }
    // deriv target x-coords
    for (int i = 0; i < ncol; ++i)
    {
      CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_nodes[i]);

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
      dweargp[p->first] += xabsx * gpt[0] * (p->second);

    for (CI p = dmap_coord_y.begin(); p != dmap_coord_y.end(); ++p)
      dweargp[p->first] += xabsx * gpt[1] * (p->second);

    // add tangent lin. to slip linearization for wear Tmatrix
    for (CI p = dmap_coord_x.begin(); p != dmap_coord_x.end(); ++p)
      dsliptmatrixgp[p->first] += xabsxT * gpt[0] * (p->second);

    for (CI p = dmap_coord_y.begin(); p != dmap_coord_y.end(); ++p)
      dsliptmatrixgp[p->first] += xabsxT * gpt[1] * (p->second);
  }
}

/*----------------------------------------------------------------------*
 |  Compute wear at GP (for expl/impl algor.)                farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_3d_wear(Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseMatrix& target_deriv, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& lm_deriv, Core::LinAlg::SerialDenseMatrix& lagmult,
    double* gpn, double& jac, double& wgt, double* jumpval, double* wearval,
    Core::Gen::Pairedvector<int, double>& dsliptmatrixgp,
    Core::Gen::Pairedvector<int, double>& dweargp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_gp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  const int nrow = source_elem.num_node();
  const int ncol = target_elem.num_node();
  const int ndof = n_dim();

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  Core::Nodes::Node** target_nodes = target_elem.nodes();

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  //***********************************************************************
  // Here, the tangential relative slip increment is used and NOT the
  // nodal weighted tangential relative slip increment !!!
  // The reason for that is the slip for wear calculation is written
  // within the integral --> no double weighting allowed !
  // The wearcoefficient is not included in this calculation
  //***********************************************************************

  // wear-lin
  Core::LinAlg::Matrix<3, 1> jump;
  Core::LinAlg::Matrix<3, 1> jumptan;
  Core::LinAlg::Matrix<3, 1> lm;
  Core::LinAlg::Matrix<3, 1> lmtan;
  Core::LinAlg::Matrix<3, 3> tanplane;

  std::array<double, 3> gplm = {0.0, 0.0, 0.0};
  double lm_lin = 0.0;

  // tangent plane
  tanplane(0, 0) = 1.0 - (gpn[0] * gpn[0]);
  tanplane(0, 1) = -(gpn[0] * gpn[1]);
  tanplane(0, 2) = -(gpn[0] * gpn[2]);
  tanplane(1, 0) = -(gpn[1] * gpn[0]);
  tanplane(1, 1) = 1.0 - (gpn[1] * gpn[1]);
  tanplane(1, 2) = -(gpn[1] * gpn[2]);
  tanplane(2, 0) = -(gpn[2] * gpn[0]);
  tanplane(2, 1) = -(gpn[2] * gpn[1]);
  tanplane(2, 2) = 1.0 - (gpn[2] * gpn[2]);

  // interpolation of source GP jumps (relative displacement increment)
  std::array<double, 3> sgpjump = {0.0, 0.0, 0.0};
  for (int i = 0; i < nrow; ++i)
  {
    sgpjump[0] += source_val[i] *
                  (source_elem.get_nodal_coords(0, i) - source_elem.get_nodal_coords_old(0, i));
    sgpjump[1] += source_val[i] *
                  (source_elem.get_nodal_coords(1, i) - source_elem.get_nodal_coords_old(1, i));
    sgpjump[2] += source_val[i] *
                  (source_elem.get_nodal_coords(2, i) - source_elem.get_nodal_coords_old(2, i));
  }

  // interpolation of target GP jumps (relative displacement increment)
  std::array<double, 3> mgpjump = {0.0, 0.0, 0.0};
  for (int i = 0; i < ncol; ++i)
  {
    mgpjump[0] += target_val[i] *
                  (target_elem.get_nodal_coords(0, i) - target_elem.get_nodal_coords_old(0, i));
    mgpjump[1] += target_val[i] *
                  (target_elem.get_nodal_coords(1, i) - target_elem.get_nodal_coords_old(1, i));
    mgpjump[2] += target_val[i] *
                  (target_elem.get_nodal_coords(2, i) - target_elem.get_nodal_coords_old(2, i));
  }

  // build jump (relative displacement increment) at current GP
  jump(0, 0) = (sgpjump[0] - mgpjump[0]);
  jump(1, 0) = (sgpjump[1] - mgpjump[1]);
  jump(2, 0) = (sgpjump[2] - mgpjump[2]);

  // build lagrange multiplier at current GP
  for (int i = 0; i < nrow; ++i)
  {
    lm(0, 0) += lm_val[i] * (lagmult)(0, i);
    lm(1, 0) += lm_val[i] * (lagmult)(1, i);
    lm(2, 0) += lm_val[i] * (lagmult)(2, i);
  }

  // build tangential jump
  jumptan.multiply(tanplane, jump);

  // build tangential lm
  lmtan.multiply(tanplane, lm);

  // evaluate wear
  // not including wearcoefficient
  // normal contact stress
  for (int i = 0; i < 3; ++i) wearval[0] += lm(i, 0) * gpn[i];

  // absolute value of relative tangential jump
  jumpval[0] = sqrt(jumptan(0, 0) * jumptan(0, 0) + jumptan(1, 0) * jumptan(1, 0) +
                    jumptan(2, 0) * jumptan(2, 0));

  // steady state wear
  if (sswear_) jumpval[0] = ssslip_;

  // no jump --> no wear
  if (abs(jumpval[0]) < 1e-12) return;

  // product
  if (wearimpl_)
    wearval[0] = (wearval[0]) * jumpval[0];
  else
    wearval[0] = abs(wearval[0]) * jumpval[0];

  // normal contact stress
  for (int i = 0; i < 3; ++i) gplm[i] = lm(i, 0);

  for (int i = 0; i < 3; ++i) lm_lin += gpn[i] * gplm[i];

  // add to node
  for (int j = 0; j < nrow; ++j)
  {
    CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(source_nodes[j]);

    double prod = 0.0;
    if (shape_fcn() == Mortar::shape_petrovgalerkin)
      prod = source_val[j] * wearval[0] * jac * wgt;
    else
      prod = lm_val[j] * wearval[0] * jac * wgt;

    // add current Gauss point's contribution to wseg
    cnode->add_delta_weighted_wear_value(prod);
  }

  // linearization without lm weighting and jac.
  if (wearimpl_ or wear_type() == Wear::wear_primvar)
  {
    int linsize = 0;
    for (int i = 0; i < nrow; ++i)
    {
      Node* cnode = dynamic_cast<Node*>(source_nodes[i]);
      linsize += cnode->get_linsize();
    }

    // evaluate the GP wear function derivatives
    Core::Gen::Pairedvector<int, double> ddualgp_x((nrow + ncol) * ndof + linsize);
    Core::Gen::Pairedvector<int, double> ddualgp_y((nrow + ncol) * ndof + linsize);
    Core::Gen::Pairedvector<int, double> ddualgp_z((nrow + ncol) * ndof + linsize);

    Core::Gen::Pairedvector<int, double> ddualgp_x_sxi((nrow + ncol) * ndof + linsize);
    Core::Gen::Pairedvector<int, double> ddualgp_y_sxi((nrow + ncol) * ndof + linsize);
    Core::Gen::Pairedvector<int, double> ddualgp_z_sxi((nrow + ncol) * ndof + linsize);

    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> tanggp(
        3, std::vector<Core::Gen::Pairedvector<int, double>>(3, (ncol * ndof) + linsize));

    // lin. abs(x) = x/abs(x) * lin x.
    double xabsx = (1.0 / abs(jumpval[0])) * lm_lin;
    double absx = (1.0 / abs(jumpval[0]));


    // **********************************************************************
    // (1) Lin of normal for LM -- deriv normal maps from weighted gap lin.
    for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gplm[0] * (p->second);

    for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gplm[1] * (p->second);

    for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gplm[2] * (p->second);

    // **********************************************************************
    // (2) Lin. of dual shape function of LM mult.
    for (int i = 0; i < nrow; ++i)
    {
      for (int m = 0; m < nrow; ++m)
      {
        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 dualmap.begin();
            p != dualmap.end(); ++p)
        {
          ddualgp_x[p->first] += (lagmult)(0, m) * source_val[m] * (p->second)(i, m);
          ddualgp_y[p->first] += (lagmult)(1, m) * source_val[m] * (p->second)(i, m);
          ddualgp_z[p->first] += (lagmult)(2, m) * source_val[m] * (p->second)(i, m);
        }
      }
    }
    for (CI p = ddualgp_x.begin(); p != ddualgp_x.end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gpn[0] * (p->second);
    for (CI p = ddualgp_y.begin(); p != ddualgp_y.end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gpn[1] * (p->second);
    for (CI p = ddualgp_z.begin(); p != ddualgp_z.end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gpn[2] * (p->second);

    // LM deriv
    for (int i = 0; i < nrow; ++i)
    {
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      {
        ddualgp_x_sxi[p->first] += (lagmult)(0, i) * lm_deriv(i, 0) * (p->second);
        ddualgp_y_sxi[p->first] += (lagmult)(1, i) * lm_deriv(i, 0) * (p->second);
        ddualgp_z_sxi[p->first] += (lagmult)(2, i) * lm_deriv(i, 0) * (p->second);
      }
      for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
      {
        ddualgp_x_sxi[p->first] += (lagmult)(0, i) * lm_deriv(i, 1) * (p->second);
        ddualgp_y_sxi[p->first] += (lagmult)(1, i) * lm_deriv(i, 1) * (p->second);
        ddualgp_z_sxi[p->first] += (lagmult)(2, i) * lm_deriv(i, 1) * (p->second);
      }
    }
    for (CI p = ddualgp_x_sxi.begin(); p != ddualgp_x_sxi.end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gpn[0] * (p->second);
    for (CI p = ddualgp_y_sxi.begin(); p != ddualgp_y_sxi.end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gpn[1] * (p->second);
    for (CI p = ddualgp_z_sxi.begin(); p != ddualgp_z_sxi.end(); ++p)
      dweargp[p->first] += abs(jumpval[0]) * gpn[2] * (p->second);

    // **********************************************************************
    // (3) absolute incremental slip linearization:
    // (a) build directional derivative of source GP tagent

    // lin tangplane: 1-n x n --> - ( dn x n + n x dn )
    for (int i = 0; i < 3; ++i)
    {
      for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
        tanggp[0][i][p->first] -= gpn[i] * (p->second);

      for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
        tanggp[1][i][p->first] -= gpn[i] * (p->second);

      for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
        tanggp[2][i][p->first] -= gpn[i] * (p->second);
    }
    for (int i = 0; i < 3; ++i)
    {
      for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
        tanggp[i][0][p->first] -= gpn[i] * (p->second);

      for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
        tanggp[i][1][p->first] -= gpn[i] * (p->second);

      for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
        tanggp[i][2][p->first] -= gpn[i] * (p->second);
    }

    Core::Gen::Pairedvector<int, double> dt0((ncol * ndof) + linsize);
    Core::Gen::Pairedvector<int, double> dt1((ncol * ndof) + linsize);
    Core::Gen::Pairedvector<int, double> dt2((ncol * ndof) + linsize);

    // xccord from tang jump lin --> lin tangplane * jump
    for (CI p = tanggp[0][0].begin(); p != tanggp[0][0].end(); ++p)
      dt0[p->first] += (p->second) * jump(0, 0);

    for (CI p = tanggp[0][1].begin(); p != tanggp[0][1].end(); ++p)
      dt0[p->first] += (p->second) * jump(1, 0);

    for (CI p = tanggp[0][2].begin(); p != tanggp[0][2].end(); ++p)
      dt0[p->first] += (p->second) * jump(2, 0);

    // yccord from tang jump lin
    for (CI p = tanggp[1][0].begin(); p != tanggp[1][0].end(); ++p)
      dt1[p->first] += (p->second) * jump(0, 0);

    for (CI p = tanggp[1][1].begin(); p != tanggp[1][1].end(); ++p)
      dt1[p->first] += (p->second) * jump(1, 0);

    for (CI p = tanggp[1][2].begin(); p != tanggp[1][2].end(); ++p)
      dt1[p->first] += (p->second) * jump(2, 0);

    // zccord from tang jump lin
    for (CI p = tanggp[2][0].begin(); p != tanggp[2][0].end(); ++p)
      dt2[p->first] += (p->second) * jump(0, 0);

    for (CI p = tanggp[2][1].begin(); p != tanggp[2][1].end(); ++p)
      dt2[p->first] += (p->second) * jump(1, 0);

    for (CI p = tanggp[2][2].begin(); p != tanggp[2][2].end(); ++p)
      dt2[p->first] += (p->second) * jump(2, 0);


    // add to weargp :  1/abs(tangplane*jump) * LM * tanjump^T * (lin. tangplane * jump)
    for (CI p = dt0.begin(); p != dt0.end(); ++p)
      dweargp[p->first] += xabsx * (p->second) * jumptan(0, 0);
    for (CI p = dt1.begin(); p != dt1.end(); ++p)
      dweargp[p->first] += xabsx * (p->second) * jumptan(1, 0);
    for (CI p = dt2.begin(); p != dt2.end(); ++p)
      dweargp[p->first] += xabsx * (p->second) * jumptan(2, 0);

    // slip lin. for discrete wear
    // u/abs(u) * lin tang * jump
    if (wear_type() == Wear::wear_primvar)
    {
      for (CI p = dt0.begin(); p != dt0.end(); ++p)
        dsliptmatrixgp[p->first] += absx * (p->second) * jumptan(0, 0);

      for (CI p = dt1.begin(); p != dt1.end(); ++p)
        dsliptmatrixgp[p->first] += absx * (p->second) * jumptan(1, 0);

      for (CI p = dt2.begin(); p != dt2.end(); ++p)
        dsliptmatrixgp[p->first] += absx * (p->second) * jumptan(2, 0);
    }

    // **********************************************************************
    // (c) build directional derivative of jump
    Core::Gen::Pairedvector<int, double> dmap_slcoord_gp_x((nrow + ncol) * ndof + linsize);
    Core::Gen::Pairedvector<int, double> dmap_slcoord_gp_y((nrow + ncol) * ndof + linsize);
    Core::Gen::Pairedvector<int, double> dmap_slcoord_gp_z((nrow + ncol) * ndof + linsize);

    Core::Gen::Pairedvector<int, double> dmap_mcoord_gp_x((nrow + ncol) * ndof + linsize);
    Core::Gen::Pairedvector<int, double> dmap_mcoord_gp_y((nrow + ncol) * ndof + linsize);
    Core::Gen::Pairedvector<int, double> dmap_mcoord_gp_z((nrow + ncol) * ndof + linsize);

    Core::Gen::Pairedvector<int, double> dmap_coord_x((nrow + ncol) * ndof + linsize);
    Core::Gen::Pairedvector<int, double> dmap_coord_y((nrow + ncol) * ndof + linsize);
    Core::Gen::Pairedvector<int, double> dmap_coord_z((nrow + ncol) * ndof + linsize);

    // lin source part -- source_xi
    for (int i = 0; i < nrow; ++i)
    {
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      {
        double valx = source_deriv(i, 0) *
                      (source_elem.get_nodal_coords(0, i) - source_elem.get_nodal_coords_old(0, i));
        dmap_slcoord_gp_x[p->first] += valx * (p->second);
        double valy = source_deriv(i, 0) *
                      (source_elem.get_nodal_coords(1, i) - source_elem.get_nodal_coords_old(1, i));
        dmap_slcoord_gp_y[p->first] += valy * (p->second);
        double valz = source_deriv(i, 0) *
                      (source_elem.get_nodal_coords(2, i) - source_elem.get_nodal_coords_old(2, i));
        dmap_slcoord_gp_z[p->first] += valz * (p->second);
      }
      for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
      {
        double valx = source_deriv(i, 1) *
                      (source_elem.get_nodal_coords(0, i) - source_elem.get_nodal_coords_old(0, i));
        dmap_slcoord_gp_x[p->first] += valx * (p->second);
        double valy = source_deriv(i, 1) *
                      (source_elem.get_nodal_coords(1, i) - source_elem.get_nodal_coords_old(1, i));
        dmap_slcoord_gp_y[p->first] += valy * (p->second);
        double valz = source_deriv(i, 1) *
                      (source_elem.get_nodal_coords(2, i) - source_elem.get_nodal_coords_old(2, i));
        dmap_slcoord_gp_z[p->first] += valz * (p->second);
      }
    }

    // lin target part -- target_xi
    for (int i = 0; i < ncol; ++i)
    {
      for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
      {
        double valx = target_deriv(i, 0) *
                      (target_elem.get_nodal_coords(0, i) - target_elem.get_nodal_coords_old(0, i));
        dmap_mcoord_gp_x[p->first] += valx * (p->second);
        double valy = target_deriv(i, 0) *
                      (target_elem.get_nodal_coords(1, i) - target_elem.get_nodal_coords_old(1, i));
        dmap_mcoord_gp_y[p->first] += valy * (p->second);
        double valz = target_deriv(i, 0) *
                      (target_elem.get_nodal_coords(2, i) - target_elem.get_nodal_coords_old(2, i));
        dmap_mcoord_gp_z[p->first] += valz * (p->second);
      }
      for (CI p = d_target_xi_gp[1].begin(); p != d_target_xi_gp[1].end(); ++p)
      {
        double valx = target_deriv(i, 1) *
                      (target_elem.get_nodal_coords(0, i) - target_elem.get_nodal_coords_old(0, i));
        dmap_mcoord_gp_x[p->first] += valx * (p->second);
        double valy = target_deriv(i, 1) *
                      (target_elem.get_nodal_coords(1, i) - target_elem.get_nodal_coords_old(1, i));
        dmap_mcoord_gp_y[p->first] += valy * (p->second);
        double valz = target_deriv(i, 1) *
                      (target_elem.get_nodal_coords(2, i) - target_elem.get_nodal_coords_old(2, i));
        dmap_mcoord_gp_z[p->first] += valz * (p->second);
      }
    }

    // deriv source x-coords
    for (int i = 0; i < nrow; ++i)
    {
      CONTACT::Node* source_node = dynamic_cast<CONTACT::Node*>(source_nodes[i]);

      dmap_slcoord_gp_x[source_node->dofs()[0]] += source_val[i];
      dmap_slcoord_gp_y[source_node->dofs()[1]] += source_val[i];
      dmap_slcoord_gp_z[source_node->dofs()[2]] += source_val[i];
    }
    // deriv target x-coords
    for (int i = 0; i < ncol; ++i)
    {
      CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_nodes[i]);

      dmap_mcoord_gp_x[target_node->dofs()[0]] += target_val[i];
      dmap_mcoord_gp_y[target_node->dofs()[1]] += target_val[i];
      dmap_mcoord_gp_z[target_node->dofs()[2]] += target_val[i];
    }

    // source: add to jumplin
    for (CI p = dmap_slcoord_gp_x.begin(); p != dmap_slcoord_gp_x.end(); ++p)
      dmap_coord_x[p->first] += (p->second);
    for (CI p = dmap_slcoord_gp_y.begin(); p != dmap_slcoord_gp_y.end(); ++p)
      dmap_coord_y[p->first] += (p->second);
    for (CI p = dmap_slcoord_gp_z.begin(); p != dmap_slcoord_gp_z.end(); ++p)
      dmap_coord_z[p->first] += (p->second);

    // target: add to jumplin
    for (CI p = dmap_mcoord_gp_x.begin(); p != dmap_mcoord_gp_x.end(); ++p)
      dmap_coord_x[p->first] -= (p->second);
    for (CI p = dmap_mcoord_gp_y.begin(); p != dmap_mcoord_gp_y.end(); ++p)
      dmap_coord_y[p->first] -= (p->second);
    for (CI p = dmap_mcoord_gp_z.begin(); p != dmap_mcoord_gp_z.end(); ++p)
      dmap_coord_z[p->first] -= (p->second);

    // matrix vector prod -- tan
    Core::Gen::Pairedvector<int, double> lintan0((nrow + ncol) * ndof + linsize);
    Core::Gen::Pairedvector<int, double> lintan1((nrow + ncol) * ndof + linsize);
    Core::Gen::Pairedvector<int, double> lintan2((nrow + ncol) * ndof + linsize);

    for (CI p = dmap_coord_x.begin(); p != dmap_coord_x.end(); ++p)
      lintan0[p->first] += tanplane(0, 0) * (p->second);
    for (CI p = dmap_coord_y.begin(); p != dmap_coord_y.end(); ++p)
      lintan0[p->first] += tanplane(0, 1) * (p->second);
    for (CI p = dmap_coord_z.begin(); p != dmap_coord_z.end(); ++p)
      lintan0[p->first] += tanplane(0, 2) * (p->second);

    for (CI p = dmap_coord_x.begin(); p != dmap_coord_x.end(); ++p)
      lintan1[p->first] += tanplane(1, 0) * (p->second);
    for (CI p = dmap_coord_y.begin(); p != dmap_coord_y.end(); ++p)
      lintan1[p->first] += tanplane(1, 1) * (p->second);
    for (CI p = dmap_coord_z.begin(); p != dmap_coord_z.end(); ++p)
      lintan1[p->first] += tanplane(1, 2) * (p->second);

    for (CI p = dmap_coord_x.begin(); p != dmap_coord_x.end(); ++p)
      lintan2[p->first] += tanplane(2, 0) * (p->second);
    for (CI p = dmap_coord_y.begin(); p != dmap_coord_y.end(); ++p)
      lintan2[p->first] += tanplane(2, 1) * (p->second);
    for (CI p = dmap_coord_z.begin(); p != dmap_coord_z.end(); ++p)
      lintan2[p->first] += tanplane(2, 2) * (p->second);

    // add to dweargp
    for (CI p = lintan0.begin(); p != lintan0.end(); ++p)
      dweargp[p->first] += xabsx * jumptan(0, 0) * (p->second);
    for (CI p = lintan1.begin(); p != lintan1.end(); ++p)
      dweargp[p->first] += xabsx * jumptan(1, 0) * (p->second);
    for (CI p = lintan2.begin(); p != lintan2.end(); ++p)
      dweargp[p->first] += xabsx * jumptan(2, 0) * (p->second);

    // slip lin. for discrete wear
    // u/abs(u) * tang * lin jump
    if (wear_type() == Wear::wear_primvar)
    {
      for (CI p = lintan0.begin(); p != lintan0.end(); ++p)
        dsliptmatrixgp[p->first] += absx * (p->second) * jumptan(0, 0);

      for (CI p = lintan1.begin(); p != lintan1.end(); ++p)
        dsliptmatrixgp[p->first] += absx * (p->second) * jumptan(1, 0);

      for (CI p = lintan2.begin(); p != lintan2.end(); ++p)
        dsliptmatrixgp[p->first] += absx * (p->second) * jumptan(2, 0);
    }
  }
}


/*----------------------------------------------------------------------*
 |  Compute entries for E and T matrix at GP (Source)         farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_te(Mortar::Element& source_elem,
    Core::LinAlg::SerialDenseVector& lm_val, Core::LinAlg::SerialDenseVector& source_val,
    double& jac, double& wgt, double* jumpval)
{
  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  if (!source_nodes) FOUR_C_THROW("Null pointer!");

  int nrow = source_elem.num_node();

  if (wear_shape_fcn() == Wear::wear_shape_standard)
  {
    for (int k = 0; k < nrow; ++k)
    {
      CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(source_nodes[k]);

      for (int j = 0; j < nrow; ++j)
      {
        CONTACT::FriNode* source_node = dynamic_cast<CONTACT::FriNode*>(source_nodes[j]);

        // multiply the two shape functions
        double prod1 = source_val[k] * lm_val[j] * abs(*jumpval) * jac * wgt;
        double prod2 = source_val[k] * source_val[j] * jac * wgt;

        int col = source_node->dofs()[0];
        int row = 0;

        if (abs(prod1) > MORTARINTTOL) cnode->add_t_value(row, col, prod1);
        if (abs(prod2) > MORTARINTTOL) cnode->add_e_value(row, col, prod2);
      }
    }
  }
  else if (wear_shape_fcn() == Wear::wear_shape_dual)
  {
    for (int k = 0; k < nrow; ++k)
    {
      CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(source_nodes[k]);

      for (int j = 0; j < nrow; ++j)
      {
        CONTACT::FriNode* source_node = dynamic_cast<CONTACT::FriNode*>(source_nodes[j]);

        // multiply the two shape functions
        double prod1 = lm_val[k] * lm_val[j] * abs(*jumpval) * jac * wgt;
        double prod2 = lm_val[k] * source_val[j] * jac * wgt;

        int col = source_node->dofs()[0];
        int row = 0;

        if (abs(prod1) > MORTARINTTOL) cnode->add_t_value(row, col, prod1);

        // diagonal E matrix
        if (j == k)
          if (abs(prod2) > MORTARINTTOL) cnode->add_e_value(row, col, prod2);
      }
    }
  }
  else
    FOUR_C_THROW("Chosen wear shape function not supported!");
}

/*----------------------------------------------------------------------*
 |  Compute entries for E and T matrix at GP (Target)        farah 11/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_te_target(Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseVector& lm2val, Core::LinAlg::SerialDenseVector& target_val,
    double& jac, double& wgt, double* jumpval, MPI_Comm comm)
{
  if (source_elem.owner() != Core::Communication::my_mpi_rank(comm)) return;

  // target_elem is involved for both-sided wear
  target_elem.set_attached() = true;

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  if (!source_nodes) FOUR_C_THROW("Null pointer!");

  // get target element nodes themselves
  Core::Nodes::Node** target_nodes = target_elem.nodes();
  if (!target_nodes) FOUR_C_THROW("Null pointer!");

  int nrow = target_elem.num_node();
  int ncol = source_elem.num_node();

  if (wear_shape_fcn() == Wear::wear_shape_standard)
  {
    for (int k = 0; k < nrow; ++k)
    {
      CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(target_nodes[k]);
      int row = 0;

      for (int j = 0; j < nrow; ++j)
      {
        CONTACT::FriNode* source_node = dynamic_cast<CONTACT::FriNode*>(target_nodes[j]);

        // multiply the two shape functions
        double prod2 = target_val[k] * target_val[j] * jac * wgt;

        int col = source_node->dofs()[0];
        if (abs(prod2) > MORTARINTTOL) cnode->add_e_value(row, col, prod2);
      }
      for (int j = 0; j < ncol; ++j)
      {
        CONTACT::FriNode* source_node = dynamic_cast<CONTACT::FriNode*>(source_nodes[j]);

        // multiply the two shape functions
        double prod1 = target_val[k] * lm_val[j] * abs(*jumpval) * jac * wgt;

        int col = source_node->dofs()[0];
        if (abs(prod1) > MORTARINTTOL) cnode->add_t_value(row, col, prod1);
      }
    }
  }
  else if (wear_shape_fcn() == Wear::wear_shape_dual)
  {
    for (int k = 0; k < nrow; ++k)
    {
      CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(target_nodes[k]);
      int row = 0;

      for (int j = 0; j < nrow; ++j)
      {
        CONTACT::FriNode* source_node = dynamic_cast<CONTACT::FriNode*>(target_nodes[j]);

        // multiply the two shape functions
        double prod2 = lm2val[k] * target_val[j] * jac * wgt;

        int col = source_node->dofs()[0];
        if (abs(prod2) > MORTARINTTOL and j == k) cnode->add_e_value(row, col, prod2);
      }
      for (int j = 0; j < ncol; ++j)
      {
        CONTACT::FriNode* source_node = dynamic_cast<CONTACT::FriNode*>(source_nodes[j]);

        // multiply the two shape functions
        double prod1 = lm2val[k] * lm_val[j] * abs(*jumpval) * jac * wgt;

        int col = source_node->dofs()[0];
        if (abs(prod1) > MORTARINTTOL) cnode->add_t_value(row, col, prod1);
      }
    }
  }
  else
    FOUR_C_THROW("Chosen wear shape function not supported!");
}

/*----------------------------------------------------------------------*
 |  Compute lin for T and E matrix -- discr. wear            farah 11/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_2d_te_target_lin(int& iter,  // like k
    Mortar::Element& source_elem, Mortar::Element& target_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseVector& lm_val, Core::LinAlg::SerialDenseMatrix& target_deriv,
    Core::LinAlg::SerialDenseMatrix& lm_deriv, double& dsxideta, double& dxdsxi, double& dxdsxidsxi,
    double& wgt, double* jumpval, const Core::Gen::Pairedvector<int, double>& d_source_xi_gp,
    const Core::Gen::Pairedvector<int, double>& d_target_xi_gp,
    const Core::Gen::Pairedvector<int, double>& derivjac,
    const Core::Gen::Pairedvector<int, double>& dsliptmatrixgp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& ximaps,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, MPI_Comm comm)
{
  if (source_elem.owner() != Core::Communication::my_mpi_rank(comm)) return;

  const int nrow = source_elem.num_node();
  const int ncol = target_elem.num_node();

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  if (!source_nodes) FOUR_C_THROW("Null pointer!");

  // get target element nodes themselves
  Core::Nodes::Node** target_nodes = target_elem.nodes();
  if (!target_nodes) FOUR_C_THROW("Null pointer!");

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(target_nodes[iter]);
  if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

  if (wear_shape_fcn() == Wear::wear_shape_standard)
  {
    // integrate LinT
    for (int j = 0; j < nrow; ++j)
    {
      // global target node ID
      int t_gid = source_elem.nodes()[j]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& tmmap_jk =
          dynamic_cast<CONTACT::FriNode*>(mymrtrnode)->wear_data().get_deriv_tw()[t_gid];

      // (1) Lin(Phi) - dual shape functions
      for (int m = 0; m < nrow; ++m)
      {
        fac = wgt * target_val[m] * source_val[j] * dsxideta * dxdsxi * abs(jumpval[0]);
        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 dualmap.begin();
            p != dualmap.end(); ++p)
          tmmap_jk[p->first] += fac * (p->second)(iter, m);
      }

      // (2) Lin(Phi) - source GP coordinates
      fac = wgt * lm_deriv(j, 0) * target_val[iter] * dsxideta * dxdsxi * abs(jumpval[0]);
      for (CI p = d_source_xi_gp.begin(); p != d_source_xi_gp.end(); ++p)
        tmmap_jk[p->first] += fac * (p->second);

      // (3) Lin(Phi) - source GP coordinates
      fac = wgt * lm_val[j] * target_deriv(iter, 0) * dsxideta * dxdsxi * abs(jumpval[0]);
      for (CI p = d_target_xi_gp.begin(); p != d_target_xi_gp.end(); ++p)
        tmmap_jk[p->first] += fac * (p->second);

      // (4) Lin(dsxideta) - segment end coordinates
      fac = wgt * lm_val[j] * target_val[iter] * dxdsxi * abs(jumpval[0]);
      for (CI p = ximaps[0].begin(); p != ximaps[0].end(); ++p)
        tmmap_jk[p->first] -= 0.5 * fac * (p->second);
      for (CI p = ximaps[1].begin(); p != ximaps[1].end(); ++p)
        tmmap_jk[p->first] += 0.5 * fac * (p->second);

      // (5) Lin(dxdsxi) - source GP Jacobian
      fac = wgt * lm_val[j] * target_val[iter] * dsxideta * abs(jumpval[0]);
      for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
        tmmap_jk[p->first] += fac * (p->second);

      // (6) Lin(dxdsxi) - source GP coordinates
      fac = wgt * lm_val[j] * target_val[iter] * dsxideta * dxdsxidsxi * abs(jumpval[0]);
      for (CI p = d_source_xi_gp.begin(); p != d_source_xi_gp.end(); ++p)
        tmmap_jk[p->first] += fac * (p->second);

      // (7) Lin(wear)
      fac = wgt * lm_val[j] * target_val[iter] * dsxideta * dxdsxi;
      for (CI p = dsliptmatrixgp.begin(); p != dsliptmatrixgp.end(); ++p)  // dsliptmatrixgp
      {
        tmmap_jk[p->first] += fac * (p->second);
      }
    }  // end integrate linT

    //********************************************************************************
    // integrate LinE
    for (int j = 0; j < ncol; ++j)
    {
      // global target node ID
      int t_gid = target_elem.nodes()[j]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& emmap_jk =
          dynamic_cast<CONTACT::FriNode*>(mymrtrnode)->wear_data().get_deriv_e()[t_gid];

      // (1) Lin(Phi) - source GP coordinates
      fac = wgt * target_deriv(iter, 0) * target_val[j] * dsxideta * dxdsxi;
      for (CI p = d_target_xi_gp.begin(); p != d_target_xi_gp.end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      // (2) Lin(Phi) - source GP coordinates
      fac = wgt * target_val[iter] * target_deriv(j, 0) * dsxideta * dxdsxi;
      for (CI p = d_target_xi_gp.begin(); p != d_target_xi_gp.end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      // (3) Lin(dsxideta) - segment end coordinates
      fac = wgt * target_val[iter] * target_val[j] * dxdsxi;
      for (CI p = ximaps[0].begin(); p != ximaps[0].end(); ++p)
        emmap_jk[p->first] -= 0.5 * fac * (p->second);
      for (CI p = ximaps[1].begin(); p != ximaps[1].end(); ++p)
        emmap_jk[p->first] += 0.5 * fac * (p->second);

      // (4) Lin(dxdsxi) - source GP Jacobian
      fac = wgt * target_val[iter] * target_val[j] * dsxideta;
      for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      // (5) Lin(dxdsxi) - source GP coordinates
      fac = wgt * target_val[iter] * target_val[j] * dsxideta * dxdsxidsxi;
      for (CI p = d_source_xi_gp.begin(); p != d_source_xi_gp.end(); ++p)
        emmap_jk[p->first] += fac * (p->second);
    }  // end integrate linE
  }
  else if (wear_shape_fcn() == Wear::wear_shape_dual)  //******************************************
  {
    FOUR_C_THROW("Chosen shapefunctions 'wear_shape_dual' not supported!");
  }
  else
    FOUR_C_THROW("Chosen shapefunctions not supported!");
}

/*----------------------------------------------------------------------*
 |  Compute lin for T and E matrix -- discr. wear            farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_2d_te_lin(int& iter, Mortar::Element& source_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    double& jac, double& wgt, double* jumpval,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const Core::Gen::Pairedvector<int, double>& derivjac,
    const Core::Gen::Pairedvector<int, double>& dsliptmatrixgp,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  const int nrow = source_elem.num_node();

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  if (!source_nodes) FOUR_C_THROW("Null pointer!");

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[iter]);
  if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

  if (wear_shape_fcn() == Wear::wear_shape_standard)
  {
    // integrate LinT
    for (int j = 0; j < nrow; ++j)
    {
      // global target node ID
      int t_gid = source_elem.nodes()[j]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& tmmap_jk =
          dynamic_cast<CONTACT::FriNode*>(mymrtrnode)->wear_data().get_deriv_tw()[t_gid];

      // (1) Lin(Phi) - dual shape functions
      for (int m = 0; m < nrow; ++m)
      {
        fac = wgt * source_val[m] * source_val[j] * jac * abs(jumpval[0]);
        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 dualmap.begin();
            p != dualmap.end(); ++p)
          tmmap_jk[p->first] += fac * (p->second)(iter, m);
      }

      // (2) Lin(Phi) - source GP coordinates
      fac = wgt * lm_deriv(j, 0) * source_val[iter] * jac * abs(jumpval[0]);
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        tmmap_jk[p->first] += fac * (p->second);

      // (3) Lin(Phi) - source GP coordinates
      fac = wgt * lm_val[j] * source_deriv(iter, 0) * jac * abs(jumpval[0]);
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        tmmap_jk[p->first] += fac * (p->second);

      // (4) Lin(dxdsxi) - source GP Jacobian
      fac = wgt * lm_val[j] * source_val[iter] * abs(jumpval[0]);
      for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
        tmmap_jk[p->first] += fac * (p->second);

      // (7) Lin(wear)
      if (!sswear_)
      {
        fac = wgt * lm_val[j] * source_val[iter] * jac;
        for (CI p = dsliptmatrixgp.begin(); p != dsliptmatrixgp.end(); ++p)  // dsliptmatrixgp
        {
          tmmap_jk[p->first] += fac * (p->second);
        }
      }

    }  // end integrate linT

    //********************************************************************************
    // integrate LinE
    for (int j = 0; j < nrow; ++j)
    {
      // global target node ID
      int t_gid = source_elem.nodes()[j]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& emmap_jk =
          dynamic_cast<CONTACT::FriNode*>(mymrtrnode)->wear_data().get_deriv_e()[t_gid];

      // (1) Lin(Phi) - source GP coordinates
      fac = wgt * source_deriv(iter, 0) * source_val[j] * jac;
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      // (2) Lin(Phi) - source GP coordinates
      fac = wgt * source_val[iter] * source_deriv(j, 0) * jac;
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      // (4) Lin(dxdsxi) - source GP Jacobian
      fac = wgt * source_val[iter] * source_val[j];
      for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
        emmap_jk[p->first] += fac * (p->second);
    }  // end integrate linE
  }
  else if (wear_shape_fcn() == Wear::wear_shape_dual)  //******************************************
  {
    // integrate LinT
    for (int j = 0; j < nrow; ++j)
    {
      // global target node ID
      int t_gid = source_elem.nodes()[j]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& tmmap_jk =
          dynamic_cast<CONTACT::FriNode*>(mymrtrnode)->wear_data().get_deriv_tw()[t_gid];

      // (1) Lin(Phi) - dual shape functions
      for (int m = 0; m < nrow; ++m)
      {
        fac = wgt * source_val[m] * lm_val[j] * jac * abs(jumpval[0]);
        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 dualmap.begin();
            p != dualmap.end(); ++p)
          tmmap_jk[p->first] += fac * (p->second)(iter, m);
      }

      // (2) Lin(Phi) - source GP coordinates
      fac = wgt * lm_deriv(j, 0) * lm_val[iter] * jac * abs(jumpval[0]);
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        tmmap_jk[p->first] += fac * (p->second);


      // (1) Lin(Phi) - dual shape functions
      for (int m = 0; m < nrow; ++m)
      {
        fac = wgt * source_val[m] * source_val[j] * jac * abs(jumpval[0]);
        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 dualmap.begin();
            p != dualmap.end(); ++p)
          tmmap_jk[p->first] += fac * (p->second)(iter, m);
      }

      // (2) Lin(Phi) - source GP coordinates
      fac = wgt * lm_deriv(iter, 0) * lm_val[j] * jac * abs(jumpval[0]);
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        tmmap_jk[p->first] += fac * (p->second);

      // (3) Lin(dxdsxi) - source GP Jacobian
      fac = wgt * lm_val[j] * lm_val[iter] * abs(jumpval[0]);
      for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
        tmmap_jk[p->first] += fac * (p->second);

      // (4) Lin(wear)
      fac = wgt * lm_val[j] * lm_val[iter] * jac;
      for (CI p = dsliptmatrixgp.begin(); p != dsliptmatrixgp.end(); ++p)  // dsliptmatrixgp
        tmmap_jk[p->first] += fac * (p->second);
    }  // end integrate linT

    //********************************************************************************
    // integrate LinE
    for (int j = 0; j < nrow; ++j)
    {
      // global target node ID
      int t_gid = source_elem.nodes()[j]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& emmap_jk =
          dynamic_cast<CONTACT::FriNode*>(mymrtrnode)->wear_data().get_deriv_e()[t_gid];

      // (1) Lin(Phi) - dual shape functions
      for (int m = 0; m < nrow; ++m)
      {
        fac = wgt * source_val[m] * source_val[j] * jac;
        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 dualmap.begin();
            p != dualmap.end(); ++p)
          emmap_jk[p->first] += fac * (p->second)(iter, m);
      }

      // (1) Lin(Phi) - source GP coordinates
      fac = wgt * lm_deriv(iter, 0) * source_val[j] * jac;
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      // (2) Lin(Phi) - source GP coordinates
      fac = wgt * lm_val[iter] * source_deriv(j, 0) * jac;
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      // (3) Lin(dxdsxi) - source GP Jacobian
      fac = wgt * lm_val[iter] * source_val[j];
      for (CI p = derivjac.begin(); p != derivjac.end(); ++p)
        emmap_jk[p->first] += fac * (p->second);
    }  // end integrate linE
  }
  else
    FOUR_C_THROW("Chosen shape functions not supported!");
}

/*----------------------------------------------------------------------*
 |  Compute lin for T and E matrix -- discr. wear            farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_3d_te_lin(int& iter, Mortar::Element& source_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    double& jac, double& wgt, double* jumpval,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap,
    const Core::Gen::Pairedvector<int, double>& dsliptmatrixgp,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  const int nrow = source_elem.num_node();

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  if (!source_nodes) FOUR_C_THROW("Null pointer!");

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(source_nodes[iter]);
  if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

  if (wear_shape_fcn() == Wear::wear_shape_standard)
  {
    // integrate LinT
    for (int j = 0; j < nrow; ++j)
    {
      // global target node ID
      int t_gid = source_elem.nodes()[j]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& dtmap_jk =
          dynamic_cast<CONTACT::FriNode*>(mymrtrnode)->wear_data().get_deriv_tw()[t_gid];

      // (1) Lin(Phi) - dual shape functions
      if (dualmap.size() > 0)
        for (int m = 0; m < nrow; ++m)
        {
          fac = wgt * source_val[m] * source_val[iter] * jac * abs(jumpval[0]);
          for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                   dualmap.begin();
              p != dualmap.end(); ++p)
            dtmap_jk[p->first] += fac * (p->second)(j, m);
        }

      // (2) Lin(Phi) - source GP coordinates
      fac = wgt * lm_deriv(j, 0) * source_val[iter] * jac * abs(jumpval[0]);
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);

      fac = wgt * lm_deriv(j, 1) * source_val[iter] * jac * abs(jumpval[0]);
      for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);


      // (3) Lin(NTarget) - target GP coordinates
      fac = wgt * lm_val[j] * source_deriv(iter, 0) * jac * abs(jumpval[0]);
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);

      fac = wgt * lm_val[j] * source_deriv(iter, 1) * jac * abs(jumpval[0]);
      for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);


      // (4) Lin(dsxideta) - intcell GP Jacobian
      fac = wgt * lm_val[j] * source_val[iter] * abs(jumpval[0]);
      for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);

      if (!sswear_)
      {
        fac = wgt * lm_val[j] * source_val[iter] * jac;
        for (CI p = dsliptmatrixgp.begin(); p != dsliptmatrixgp.end(); ++p)
          dtmap_jk[p->first] += fac * (p->second);
      }
    }

    //********************************************************************************
    // integrate LinE
    for (int j = 0; j < nrow; ++j)
    {
      // global target node ID
      int t_gid = source_elem.nodes()[j]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& emmap_jk =
          dynamic_cast<CONTACT::FriNode*>(mymrtrnode)->wear_data().get_deriv_e()[t_gid];

      // (2) Lin(Phi) - source GP coordinates
      fac = wgt * source_deriv(j, 0) * source_val[iter] * jac;
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      fac = wgt * source_deriv(j, 1) * source_val[iter] * jac;
      for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      // (3) Lin(NTarget) - target GP coordinates
      fac = wgt * source_val[j] * source_deriv(iter, 0) * jac;
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      fac = wgt * source_val[j] * source_deriv(iter, 1) * jac;
      for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      // (4) Lin(dsxideta) - intcell GP Jacobian
      fac = wgt * source_val[j] * source_val[iter];
      for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

    }  // end integrate linE
  }
  else if (wear_shape_fcn() == Wear::wear_shape_dual)
  {
    // integrate LinT
    for (int j = 0; j < nrow; ++j)
    {
      // global target node ID
      int t_gid = source_elem.nodes()[j]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& dtmap_jk =
          dynamic_cast<CONTACT::FriNode*>(mymrtrnode)->wear_data().get_deriv_tw()[t_gid];

      //**********************************************
      // LM-shape function lin...
      //**********************************************
      // (1) Lin(Phi) - dual shape functions
      if (dualmap.size() > 0)
        for (int m = 0; m < nrow; ++m)
        {
          fac = wgt * source_val[m] * lm_val[j] * jac * abs(jumpval[0]);
          for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                   dualmap.begin();
              p != dualmap.end(); ++p)
            dtmap_jk[p->first] += fac * (p->second)(iter, m);
        }

      // (2) Lin(Phi) - source GP coordinates
      fac = wgt * lm_deriv(j, 0) * lm_val[iter] * jac * abs(jumpval[0]);
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);

      fac = wgt * lm_deriv(j, 1) * lm_val[iter] * jac * abs(jumpval[0]);
      for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);

      //**********************************************
      // wear weighting lin...
      //**********************************************
      // (1) Lin(Phi) - dual shape functions
      if (dualmap.size() > 0)
        for (int m = 0; m < nrow; ++m)
        {
          fac = wgt * lm_val[iter] * source_val[m] * jac * abs(jumpval[0]);
          for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                   dualmap.begin();
              p != dualmap.end(); ++p)
            dtmap_jk[p->first] += fac * (p->second)(j, m);
        }

      // (2) Lin(Phi) - source GP coordinates
      fac = wgt * lm_val[j] * lm_deriv(iter, 0) * jac * abs(jumpval[0]);
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);

      fac = wgt * lm_val[j] * lm_deriv(iter, 1) * jac * abs(jumpval[0]);
      for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);

      //**********************************************
      // rest
      //**********************************************
      // (4) Lin(dsxideta) - intcell GP Jacobian
      fac = wgt * lm_val[j] * lm_val[iter] * abs(jumpval[0]);
      for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);

      fac = wgt * lm_val[j] * lm_val[iter] * jac;
      for (CI p = dsliptmatrixgp.begin(); p != dsliptmatrixgp.end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);
    }

    //********************************************************************************
    // integrate LinE
    for (int j = 0; j < nrow; ++j)
    {
      // global target node ID
      int t_gid = source_elem.nodes()[j]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& emmap_jk =
          dynamic_cast<CONTACT::FriNode*>(mymrtrnode)->wear_data().get_deriv_e()[t_gid];

      //**********************************************
      // wear weighting lin...
      //**********************************************
      // (1) Lin(Phi) - dual shape functions
      if (dualmap.size() > 0)
        for (int m = 0; m < nrow; ++m)
        {
          fac = wgt * source_val[m] * source_val[iter] * jac;
          for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                   dualmap.begin();
              p != dualmap.end(); ++p)
            emmap_jk[p->first] += fac * (p->second)(j, m);
        }

      // (2) Lin(Phi) - source GP coordinates
      fac = wgt * source_deriv(j, 0) * lm_val[iter] * jac;
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      fac = wgt * source_deriv(j, 1) * lm_val[iter] * jac;
      for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      // (3) Lin(NTarget) - target GP coordinates
      fac = wgt * source_val[j] * lm_deriv(iter, 0) * jac;
      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      fac = wgt * source_val[j] * lm_deriv(iter, 1) * jac;
      for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

      // (4) Lin(dsxideta) - intcell GP Jacobian
      fac = wgt * source_val[j] * lm_val[iter];
      for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

    }  // end integrate linE
  }
  else
    FOUR_C_THROW("Chosen shapefunctions not supported!");
}

/*----------------------------------------------------------------------*
 |  Compute lin for T and E matrix -- discr. wear (target)   farah 11/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_3d_te_target_lin(int& iter, Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseVector& lm2val, Core::LinAlg::SerialDenseMatrix& source_deriv,
    Core::LinAlg::SerialDenseMatrix& target_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    Core::LinAlg::SerialDenseMatrix& lm2deriv, double& jac, double& wgt, double* jumpval,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_gp,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap,
    const Core::Gen::Pairedvector<int, double>& dsliptmatrixgp,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dual2map, MPI_Comm comm)
{
  if (source_elem.owner() != Core::Communication::my_mpi_rank(comm)) return;

  const int ncol = target_elem.num_node();
  const int nrow = source_elem.num_node();

  // get source element nodes themselves
  Core::Nodes::Node** target_nodes = target_elem.nodes();
  if (!target_nodes) FOUR_C_THROW("Null pointer!");

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(target_nodes[iter]);
  if (!mymrtrnode) FOUR_C_THROW("Null pointer!");

  if (wear_shape_fcn() == Wear::wear_shape_standard)
  {
    // integrate LinT
    for (int j = 0; j < nrow; ++j)
    {
      // global target node ID
      int t_gid = source_elem.nodes()[j]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& dtmap_jk =
          dynamic_cast<CONTACT::FriNode*>(mymrtrnode)->wear_data().get_deriv_tw()[t_gid];

      // (1) Lin(Phi) - dual shape functions
      if (dualmap.size() > 0)
        for (int m = 0; m < nrow; ++m)
        {
          fac = wgt * source_val[m] * target_val[iter] * jac * abs(jumpval[0]);
          for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                   dualmap.begin();
              p != dualmap.end(); ++p)
            dtmap_jk[p->first] += fac * (p->second)(j, m);
        }

      // (2) Lin(Phi) - source GP coordinates
      for (int d = 0; d < n_dim() - 1; ++d)
      {
        fac = wgt * lm_deriv(j, d) * target_val[iter] * jac * abs(jumpval[0]);
        for (CI p = d_source_xi_gp[d].begin(); p != d_source_xi_gp[d].end(); ++p)
          dtmap_jk[p->first] += fac * (p->second);
      }

      // (3) Lin(NTarget) - target GP coordinates
      for (int d = 0; d < n_dim() - 1; ++d)
      {
        fac = wgt * lm_val[j] * target_deriv(iter, d) * jac * abs(jumpval[0]);
        for (CI p = d_target_xi_gp[d].begin(); p != d_target_xi_gp[d].end(); ++p)
          dtmap_jk[p->first] += fac * (p->second);
      }

      // (4) Lin(dsxideta) - intcell GP Jacobian
      fac = wgt * lm_val[j] * target_val[iter] * abs(jumpval[0]);
      for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);

      fac = wgt * lm_val[j] * target_val[iter] * jac;
      for (CI p = dsliptmatrixgp.begin(); p != dsliptmatrixgp.end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);
    }

    //********************************************************************************
    // integrate LinE
    for (int j = 0; j < ncol; ++j)
    {
      // global target node ID
      int t_gid = target_elem.nodes()[j]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& emmap_jk =
          dynamic_cast<CONTACT::FriNode*>(mymrtrnode)->wear_data().get_deriv_e()[t_gid];

      // (2) Lin(Phi) - source GP coordinates
      for (int d = 0; d < n_dim() - 1; ++d)
      {
        fac = wgt * target_deriv(j, d) * target_val[iter] * jac;
        for (CI p = d_target_xi_gp[d].begin(); p != d_target_xi_gp[d].end(); ++p)
          emmap_jk[p->first] += fac * (p->second);
      }

      // (3) Lin(NTarget) - target GP coordinates
      for (int d = 0; d < n_dim() - 1; ++d)
      {
        fac = wgt * target_val[j] * target_deriv(iter, d) * jac;
        for (CI p = d_target_xi_gp[d].begin(); p != d_target_xi_gp[d].end(); ++p)
          emmap_jk[p->first] += fac * (p->second);
      }

      // (4) Lin(dsxideta) - intcell GP Jacobian
      fac = wgt * target_val[j] * target_val[iter];
      for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

    }  // end integrate linE
  }
  //------------------------------------------------------------
  //------------------------------------------------------------
  //------------------------------------------------------------
  else if (wear_shape_fcn() == Wear::wear_shape_dual)
  {
    // integrate LinT
    for (int j = 0; j < nrow; ++j)
    {
      // global target node ID
      int t_gid = source_elem.nodes()[j]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& dtmap_jk =
          dynamic_cast<CONTACT::FriNode*>(mymrtrnode)->wear_data().get_deriv_tw()[t_gid];

      // (1) Lin(Phi) - dual shape functions
      if (dualmap.size() > 0)
        for (int m = 0; m < nrow; ++m)
        {
          fac = wgt * source_val[m] * lm2val[iter] * jac * abs(jumpval[0]);
          for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                   dualmap.begin();
              p != dualmap.end(); ++p)
            dtmap_jk[p->first] += fac * (p->second)(j, m);
        }

      // (2) Lin(Phi) - source GP coordinates
      for (int d = 0; d < n_dim() - 1; ++d)
      {
        fac = wgt * lm_deriv(j, d) * lm2val[iter] * jac * abs(jumpval[0]);
        for (CI p = d_source_xi_gp[d].begin(); p != d_source_xi_gp[d].end(); ++p)
          dtmap_jk[p->first] += fac * (p->second);
      }

      //----------------------

      // (1) Lin(Phi) - dual shape functions
      if (dual2map.size() > 0)
        for (int m = 0; m < ncol; ++m)
        {
          fac = wgt * lm_val[m] * target_val[iter] * jac * abs(jumpval[0]);
          for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                   dual2map.begin();
              p != dual2map.end(); ++p)
            dtmap_jk[p->first] += fac * (p->second)(j, m);
        }

      // (3) Lin(NTarget) - target GP coordinates
      for (int d = 0; d < n_dim() - 1; ++d)
      {
        fac = wgt * lm_val[j] * lm2deriv(iter, d) * jac * abs(jumpval[0]);
        for (CI p = d_target_xi_gp[d].begin(); p != d_target_xi_gp[d].end(); ++p)
          dtmap_jk[p->first] += fac * (p->second);
      }

      //----------------------

      // (4) Lin(dsxideta) - intcell GP Jacobian
      fac = wgt * lm_val[j] * lm2val[iter] * abs(jumpval[0]);
      for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);

      fac = wgt * lm_val[j] * lm2val[iter] * jac;
      for (CI p = dsliptmatrixgp.begin(); p != dsliptmatrixgp.end(); ++p)
        dtmap_jk[p->first] += fac * (p->second);
    }

    //********************************************************************************
    // integrate LinE
    for (int j = 0; j < ncol; ++j)
    {
      // global target node ID
      int t_gid = target_elem.nodes()[j]->id();
      double fac = 0.0;

      // get the correct map as a reference
      std::map<int, double>& emmap_jk =
          dynamic_cast<CONTACT::FriNode*>(mymrtrnode)->wear_data().get_deriv_e()[t_gid];

      // (2) Lin(Phi) - source GP coordinates
      for (int d = 0; d < n_dim() - 1; ++d)
      {
        fac = wgt * target_deriv(j, d) * lm2val[iter] * jac;
        for (CI p = d_target_xi_gp[d].begin(); p != d_target_xi_gp[d].end(); ++p)
          emmap_jk[p->first] += fac * (p->second);
      }

      //----------------------

      // (1) Lin(Phi) - dual shape functions
      if (dual2map.size() > 0)
        for (int m = 0; m < ncol; ++m)
        {
          fac = wgt * target_val[m] * target_val[iter] * jac;
          for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                   dual2map.begin();
              p != dual2map.end(); ++p)
            emmap_jk[p->first] += fac * (p->second)(j, m);
        }

      // (3) Lin(NTarget) - target GP coordinates
      for (int d = 0; d < n_dim() - 1; ++d)
      {
        fac = wgt * target_val[j] * lm2deriv(iter, d) * jac;
        for (CI p = d_target_xi_gp[d].begin(); p != d_target_xi_gp[d].end(); ++p)
          emmap_jk[p->first] += fac * (p->second);
      }

      //----------------------

      // (4) Lin(dsxideta) - intcell GP Jacobian
      fac = wgt * target_val[j] * target_val[iter];
      for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
        emmap_jk[p->first] += fac * (p->second);

    }  // end integrate linE
  }
  else
    FOUR_C_THROW("Chosen shape functions not supported!");
}

/*----------------------------------------------------------------------*
 |  Compute entries for weighted slip increment at GP        farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_2d_slip_incr(Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& target_deriv,
    double& jac, double& wgt, double* jumpvalv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_gp,
    Core::Gen::Pairedvector<int, double>& dslipgp, int& linsize)
{
  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  Core::Nodes::Node** target_nodes = target_elem.nodes();
  if (!source_nodes) FOUR_C_THROW("Null pointer!");
  if (!target_nodes) FOUR_C_THROW("Null pointer!");

  // number of nodes (source, target)
  const int nrow = source_elem.num_node();
  const int ncol = target_elem.num_node();
  const int ndof = n_dim();

  // LIN OF TANGENT
  Core::Gen::Pairedvector<int, double> dmap_txsl_gp(ncol * ndof + linsize);
  Core::Gen::Pairedvector<int, double> dmap_tysl_gp(ncol * ndof + linsize);

  // build interpolation of source GP normal and coordinates
  std::array<double, 2> source_jump_v = {0.0, 0.0};
  std::array<double, 2> target_jump_v = {0.0, 0.0};
  std::array<double, 2> jump_v = {0.0, 0.0};
  std::array<double, 2> tanv = {0.0, 0.0};

  double tanlength = 0.0;
  for (int i = 0; i < nrow; ++i)
  {
    CONTACT::Node* myconode = dynamic_cast<CONTACT::Node*>(source_nodes[i]);

    // nodal tangent interpolation
    tanv[0] += source_val[i] * myconode->data().txi()[0];
    tanv[1] += source_val[i] * myconode->data().txi()[1];

    // delta D
    source_jump_v[0] += source_val[i] * (source_elem.get_nodal_coords(0, i) -
                                            source_elem.get_nodal_coords_old(0, i));
    source_jump_v[1] += source_val[i] * (source_elem.get_nodal_coords(1, i) -
                                            source_elem.get_nodal_coords_old(1, i));
  }

  for (int i = 0; i < ncol; ++i)
  {
    target_jump_v[0] += target_val[i] * (target_elem.get_nodal_coords(0, i) -
                                            target_elem.get_nodal_coords_old(0, i));
    target_jump_v[1] += target_val[i] * (target_elem.get_nodal_coords(1, i) -
                                            target_elem.get_nodal_coords_old(1, i));
  }

  // normalize interpolated GP tangent back to length 1.0 !!!
  tanlength = sqrt(tanv[0] * tanv[0] + tanv[1] * tanv[1]);
  if (tanlength < 1.0e-12) FOUR_C_THROW("Divide by zero!");

  for (int i = 0; i < 2; i++) tanv[i] /= tanlength;

  // jump
  jump_v[0] = source_jump_v[0] - target_jump_v[0];
  jump_v[1] = source_jump_v[1] - target_jump_v[1];

  // multiply with tangent
  // value of relative tangential jump
  for (int i = 0; i < 2; ++i) jumpvalv[0] += tanv[i] * jump_v[i];


  // *****************************************************************************
  // add everything to dslipgp                                                   *
  // *****************************************************************************
  for (int i = 0; i < nrow; ++i)
  {
    Core::Gen::Pairedvector<int, double>& dmap_txsl_i =
        dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().get_deriv_txi()[0];
    Core::Gen::Pairedvector<int, double>& dmap_tysl_i =
        dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().get_deriv_txi()[1];

    for (CI p = dmap_txsl_i.begin(); p != dmap_txsl_i.end(); ++p)
      dmap_txsl_gp[p->first] += source_val[i] * (p->second);
    for (CI p = dmap_tysl_i.begin(); p != dmap_tysl_i.end(); ++p)
      dmap_tysl_gp[p->first] += source_val[i] * (p->second);

    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
    {
      double valx =
          source_deriv(i, 0) * dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().txi()[0];
      dmap_txsl_gp[p->first] += valx * (p->second);
      double valy =
          source_deriv(i, 0) * dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().txi()[1];
      dmap_tysl_gp[p->first] += valy * (p->second);
    }
  }

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
    dslipgp[p->first] += jump_v[0] * (p->second);

  for (CI p = dmap_tysl_gp_unit.begin(); p != dmap_tysl_gp_unit.end(); ++p)
    dslipgp[p->first] += jump_v[1] * (p->second);

  // coord lin
  for (int z = 0; z < nrow; ++z)
  {
    FriNode* source_node = dynamic_cast<FriNode*>(source_nodes[z]);
    for (int k = 0; k < 2; ++k)
    {
      dslipgp[source_node->dofs()[k]] += source_val[z] * tanv[k];

      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
        dslipgp[p->first] +=
            tanv[k] * source_deriv(z, 0) *
            (source_elem.get_nodal_coords(k, z) - source_elem.get_nodal_coords_old(k, z)) *
            (p->second);
    }
  }

  for (int z = 0; z < ncol; ++z)
  {
    FriNode* target_node = dynamic_cast<FriNode*>(target_nodes[z]);
    for (int k = 0; k < 2; ++k)
    {
      dslipgp[target_node->dofs()[k]] -= target_val[z] * tanv[k];

      for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
        dslipgp[p->first] -=
            tanv[k] * target_deriv(z, 0) *
            (target_elem.get_nodal_coords(k, z) - target_elem.get_nodal_coords_old(k, z)) *
            (p->second);
    }
  }

  // ***************************
  // Add to node!
  for (int j = 0; j < nrow; ++j)
  {
    FriNode* source_node = dynamic_cast<FriNode*>(source_nodes[j]);
    if (source_node->is_on_boundor_ce()) continue;

    double prod = lm_val[j] * jumpvalv[0] * jac * wgt;

    // add current Gauss point's contribution to jump
    source_node->add_jump_value(prod, 0);
  }
}


/*----------------------------------------------------------------------*
 |  Compute entries for slip increment at GP                 farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_3d_slip_incr(Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& target_deriv,
    double& jac, double& wgt, double* jumpvalv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_gp,
    std::vector<Core::Gen::Pairedvector<int, double>>& dslipgp)
{
  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  Core::Nodes::Node** target_nodes = target_elem.nodes();
  if (!source_nodes) FOUR_C_THROW("Null pointer!");
  if (!target_nodes) FOUR_C_THROW("Null pointer!");

  // number of nodes (source, target)
  const int nrow = source_elem.num_node();
  const int ncol = target_elem.num_node();
  const int ndof = n_dim();

  int linsize = 0;
  for (int i = 0; i < nrow; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(source_nodes[i]);
    linsize += cnode->get_linsize();
  }

  // build interpolation of source GP normal and coordinates
  std::array<double, 3> source_jump_v = {0.0, 0.0, 0.0};
  std::array<double, 3> target_jump_v = {0.0, 0.0, 0.0};
  std::array<double, 3> jump_v = {0.0, 0.0, 0.0};
  std::array<double, 3> tanv1 = {0.0, 0.0, 0.0};
  std::array<double, 3> tanv2 = {0.0, 0.0, 0.0};

  double jumpvalv1 = 0.0;
  double jumpvalv2 = 0.0;
  double tanlength1 = 0.0;
  double tanlength2 = 0.0;

  // LIN OF TANGENT
  std::map<int, double> dmap_txsl_gp;
  std::map<int, double> dmap_tysl_gp;

  for (int i = 0; i < nrow; ++i)
  {
    CONTACT::Node* myconode = dynamic_cast<CONTACT::Node*>(source_nodes[i]);

    // nodal tangent interpolation
    tanv1[0] += source_val[i] * myconode->data().txi()[0];
    tanv1[1] += source_val[i] * myconode->data().txi()[1];
    tanv1[2] += source_val[i] * myconode->data().txi()[2];

    tanv2[0] += source_val[i] * myconode->data().teta()[0];
    tanv2[1] += source_val[i] * myconode->data().teta()[1];
    tanv2[2] += source_val[i] * myconode->data().teta()[2];
    // delta D
    source_jump_v[0] += source_val[i] * (source_elem.get_nodal_coords(0, i) -
                                            source_elem.get_nodal_coords_old(0, i));
    source_jump_v[1] += source_val[i] * (source_elem.get_nodal_coords(1, i) -
                                            source_elem.get_nodal_coords_old(1, i));
    source_jump_v[2] += source_val[i] * (source_elem.get_nodal_coords(2, i) -
                                            source_elem.get_nodal_coords_old(2, i));
  }

  for (int i = 0; i < ncol; ++i)
  {
    target_jump_v[0] += target_val[i] * (target_elem.get_nodal_coords(0, i) -
                                            target_elem.get_nodal_coords_old(0, i));
    target_jump_v[1] += target_val[i] * (target_elem.get_nodal_coords(1, i) -
                                            target_elem.get_nodal_coords_old(1, i));
    target_jump_v[2] += target_val[i] * (target_elem.get_nodal_coords(2, i) -
                                            target_elem.get_nodal_coords_old(2, i));
  }

  // normalize interpolated GP tangent back to length 1.0 !!!
  tanlength1 = sqrt(tanv1[0] * tanv1[0] + tanv1[1] * tanv1[1] + tanv1[2] * tanv1[2]);
  if (tanlength1 < 1.0e-12) FOUR_C_THROW("Divide by zero!");

  tanlength2 = sqrt(tanv2[0] * tanv2[0] + tanv2[1] * tanv2[1] + tanv2[2] * tanv2[2]);
  if (tanlength2 < 1.0e-12) FOUR_C_THROW("Divide by zero!");

  for (int i = 0; i < 3; i++)
  {
    tanv1[i] /= tanlength1;
    tanv2[i] /= tanlength2;
  }

  // jump
  jump_v[0] = source_jump_v[0] - target_jump_v[0];
  jump_v[1] = source_jump_v[1] - target_jump_v[1];
  jump_v[2] = source_jump_v[2] - target_jump_v[2];

  // multiply with tangent
  // value of relative tangential jump
  for (int i = 0; i < 3; ++i)
  {
    jumpvalv1 += tanv1[i] * jump_v[i];
    jumpvalv2 += tanv2[i] * jump_v[i];
  }

  jumpvalv[0] = jumpvalv1;
  jumpvalv[1] = jumpvalv2;

  // ***************************
  // Add to node!
  for (int j = 0; j < nrow; ++j)
  {
    FriNode* source_node = dynamic_cast<FriNode*>(source_nodes[j]);

    if (source_node->is_on_boundor_ce()) continue;

    double prod1 = lm_val[j] * jumpvalv1 * jac * wgt;
    double prod2 = lm_val[j] * jumpvalv2 * jac * wgt;

    // add current Gauss point's contribution to gseg
    source_node->add_jump_value(prod1, 0);
    source_node->add_jump_value(prod2, 1);
  }

  //************* LIN TANGENT TXI *********************
  // build directional derivative of source GP txi (non-unit)
  Core::Gen::Pairedvector<int, double> dmap_txix_gp(linsize + ncol * ndof);
  Core::Gen::Pairedvector<int, double> dmap_txiy_gp(linsize + ncol * ndof);
  Core::Gen::Pairedvector<int, double> dmap_txiz_gp(linsize + ncol * ndof);

  // source GP txi (non-unit)
  Core::Gen::Pairedvector<int, double> dmap_txix_gp_unit(linsize + ncol * ndof);
  Core::Gen::Pairedvector<int, double> dmap_txiy_gp_unit(linsize + ncol * ndof);
  Core::Gen::Pairedvector<int, double> dmap_txiz_gp_unit(linsize + ncol * ndof);

  for (int i = 0; i < nrow; ++i)
  {
    Core::Gen::Pairedvector<int, double>& dmap_txsl_i =
        dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().get_deriv_txi()[0];
    Core::Gen::Pairedvector<int, double>& dmap_tysl_i =
        dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().get_deriv_txi()[1];
    Core::Gen::Pairedvector<int, double>& dmap_tzsl_i =
        dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().get_deriv_txi()[2];

    for (CI p = dmap_txsl_i.begin(); p != dmap_txsl_i.end(); ++p)
      dmap_txix_gp[p->first] += source_val[i] * (p->second);
    for (CI p = dmap_tysl_i.begin(); p != dmap_tysl_i.end(); ++p)
      dmap_txiy_gp[p->first] += source_val[i] * (p->second);
    for (CI p = dmap_tzsl_i.begin(); p != dmap_tzsl_i.end(); ++p)
      dmap_txiz_gp[p->first] += source_val[i] * (p->second);

    const double txi_x = dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().txi()[0];
    const double txi_y = dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().txi()[1];
    const double txi_z = dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().txi()[2];

    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
    {
      double valx = source_deriv(i, 0) * txi_x;
      dmap_txix_gp[p->first] += valx * (p->second);
      double valy = source_deriv(i, 0) * txi_y;
      dmap_txiy_gp[p->first] += valy * (p->second);
      double valz = source_deriv(i, 0) * txi_z;
      dmap_txiz_gp[p->first] += valz * (p->second);
    }

    for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
    {
      double valx = source_deriv(i, 1) * txi_x;
      dmap_txix_gp[p->first] += valx * (p->second);
      double valy = source_deriv(i, 1) * txi_y;
      dmap_txiy_gp[p->first] += valy * (p->second);
      double valz = source_deriv(i, 1) * txi_z;
      dmap_txiz_gp[p->first] += valz * (p->second);
    }
  }

  // build directional derivative of source GP txi (unit)
  const double ll1 = tanlength1 * tanlength1;
  const double linv1 = 1.0 / tanlength1;
  const double lllinv1 = 1.0 / (tanlength1 * tanlength1 * tanlength1);
  const double sxsx1 = tanv1[0] * tanv1[0] * ll1;
  const double sxsy1 = tanv1[0] * tanv1[1] * ll1;
  const double sxsz1 = tanv1[0] * tanv1[2] * ll1;
  const double sysy1 = tanv1[1] * tanv1[1] * ll1;
  const double sysz1 = tanv1[1] * tanv1[2] * ll1;
  const double szsz1 = tanv1[2] * tanv1[2] * ll1;

  for (CI p = dmap_txix_gp.begin(); p != dmap_txix_gp.end(); ++p)
  {
    dmap_txix_gp_unit[p->first] += linv1 * (p->second);
    dmap_txix_gp_unit[p->first] -= lllinv1 * sxsx1 * (p->second);
    dmap_txiy_gp_unit[p->first] -= lllinv1 * sxsy1 * (p->second);
    dmap_txiz_gp_unit[p->first] -= lllinv1 * sxsz1 * (p->second);
  }

  for (CI p = dmap_txiy_gp.begin(); p != dmap_txiy_gp.end(); ++p)
  {
    dmap_txiy_gp_unit[p->first] += linv1 * (p->second);
    dmap_txiy_gp_unit[p->first] -= lllinv1 * sysy1 * (p->second);
    dmap_txix_gp_unit[p->first] -= lllinv1 * sxsy1 * (p->second);
    dmap_txiz_gp_unit[p->first] -= lllinv1 * sysz1 * (p->second);
  }

  for (CI p = dmap_txiz_gp.begin(); p != dmap_txiz_gp.end(); ++p)
  {
    dmap_txiz_gp_unit[p->first] += linv1 * (p->second);
    dmap_txiz_gp_unit[p->first] -= lllinv1 * szsz1 * (p->second);
    dmap_txix_gp_unit[p->first] -= lllinv1 * sxsz1 * (p->second);
    dmap_txiy_gp_unit[p->first] -= lllinv1 * sysz1 * (p->second);
  }


  //************* LIN TANGENT TETA *********************
  // build directional derivative of source GP teta (non-unit)
  Core::Gen::Pairedvector<int, double> dmap_tetax_gp(linsize + ncol * ndof);
  Core::Gen::Pairedvector<int, double> dmap_tetay_gp(linsize + ncol * ndof);
  Core::Gen::Pairedvector<int, double> dmap_tetaz_gp(linsize + ncol * ndof);

  // source GP teta (unit)
  Core::Gen::Pairedvector<int, double> dmap_tetax_gp_unit(linsize + ncol * ndof);
  Core::Gen::Pairedvector<int, double> dmap_tetay_gp_unit(linsize + ncol * ndof);
  Core::Gen::Pairedvector<int, double> dmap_tetaz_gp_unit(linsize + ncol * ndof);

  for (int i = 0; i < nrow; ++i)
  {
    Core::Gen::Pairedvector<int, double>& dmap_txsl_i =
        dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().get_deriv_teta()[0];
    Core::Gen::Pairedvector<int, double>& dmap_tysl_i =
        dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().get_deriv_teta()[1];
    Core::Gen::Pairedvector<int, double>& dmap_tzsl_i =
        dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().get_deriv_teta()[2];

    for (CI p = dmap_txsl_i.begin(); p != dmap_txsl_i.end(); ++p)
      dmap_tetax_gp[p->first] += source_val[i] * (p->second);
    for (CI p = dmap_tysl_i.begin(); p != dmap_tysl_i.end(); ++p)
      dmap_tetay_gp[p->first] += source_val[i] * (p->second);
    for (CI p = dmap_tzsl_i.begin(); p != dmap_tzsl_i.end(); ++p)
      dmap_tetaz_gp[p->first] += source_val[i] * (p->second);

    const double teta_x = dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().teta()[0];
    const double teta_y = dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().teta()[1];
    const double teta_z = dynamic_cast<CONTACT::Node*>(source_nodes[i])->data().teta()[2];

    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
    {
      double valx = source_deriv(i, 0) * teta_x;
      dmap_tetax_gp[p->first] += valx * (p->second);
      double valy = source_deriv(i, 0) * teta_y;
      dmap_tetay_gp[p->first] += valy * (p->second);
      double valz = source_deriv(i, 0) * teta_z;
      dmap_tetaz_gp[p->first] += valz * (p->second);
    }

    for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
    {
      double valx = source_deriv(i, 1) * teta_x;
      dmap_tetax_gp[p->first] += valx * (p->second);
      double valy = source_deriv(i, 1) * teta_y;
      dmap_tetay_gp[p->first] += valy * (p->second);
      double valz = source_deriv(i, 1) * teta_z;
      dmap_tetaz_gp[p->first] += valz * (p->second);
    }
  }

  // build directional derivative of source GP teta (unit)
  const double ll2 = tanlength2 * tanlength2;
  const double linv2 = 1.0 / tanlength2;
  const double lllinv2 = 1.0 / (tanlength2 * tanlength2 * tanlength2);
  const double sxsx2 = tanv2[0] * tanv2[0] * ll2;
  const double sxsy2 = tanv2[0] * tanv2[1] * ll2;
  const double sxsz2 = tanv2[0] * tanv2[2] * ll2;
  const double sysy2 = tanv2[1] * tanv2[1] * ll2;
  const double sysz2 = tanv2[1] * tanv2[2] * ll2;
  const double szsz2 = tanv2[2] * tanv2[2] * ll2;

  for (CI p = dmap_tetax_gp.begin(); p != dmap_tetax_gp.end(); ++p)
  {
    dmap_tetax_gp_unit[p->first] += linv2 * (p->second);
    dmap_tetax_gp_unit[p->first] -= lllinv2 * sxsx2 * (p->second);
    dmap_tetay_gp_unit[p->first] -= lllinv2 * sxsy2 * (p->second);
    dmap_tetaz_gp_unit[p->first] -= lllinv2 * sxsz2 * (p->second);
  }

  for (CI p = dmap_tetay_gp.begin(); p != dmap_tetay_gp.end(); ++p)
  {
    dmap_tetay_gp_unit[p->first] += linv2 * (p->second);
    dmap_tetay_gp_unit[p->first] -= lllinv2 * sysy2 * (p->second);
    dmap_tetax_gp_unit[p->first] -= lllinv2 * sxsy2 * (p->second);
    dmap_tetaz_gp_unit[p->first] -= lllinv2 * sysz2 * (p->second);
  }

  for (CI p = dmap_tetaz_gp.begin(); p != dmap_tetaz_gp.end(); ++p)
  {
    dmap_tetaz_gp_unit[p->first] += linv2 * (p->second);
    dmap_tetaz_gp_unit[p->first] -= lllinv2 * szsz2 * (p->second);
    dmap_tetax_gp_unit[p->first] -= lllinv2 * sxsz2 * (p->second);
    dmap_tetay_gp_unit[p->first] -= lllinv2 * sysz2 * (p->second);
  }

  // TXI
  for (CI p = dmap_txix_gp_unit.begin(); p != dmap_txix_gp_unit.end(); ++p)
    dslipgp[0][p->first] += jump_v[0] * (p->second);

  for (CI p = dmap_txiy_gp_unit.begin(); p != dmap_txiy_gp_unit.end(); ++p)
    dslipgp[0][p->first] += jump_v[1] * (p->second);

  for (CI p = dmap_txiz_gp_unit.begin(); p != dmap_txiz_gp_unit.end(); ++p)
    dslipgp[0][p->first] += jump_v[2] * (p->second);

  // TETA
  for (CI p = dmap_tetax_gp_unit.begin(); p != dmap_tetax_gp_unit.end(); ++p)
    dslipgp[1][p->first] += jump_v[0] * (p->second);

  for (CI p = dmap_tetay_gp_unit.begin(); p != dmap_tetay_gp_unit.end(); ++p)
    dslipgp[1][p->first] += jump_v[1] * (p->second);

  for (CI p = dmap_tetaz_gp_unit.begin(); p != dmap_tetaz_gp_unit.end(); ++p)
    dslipgp[1][p->first] += jump_v[2] * (p->second);

  // coord lin
  for (int z = 0; z < nrow; ++z)
  {
    FriNode* source_node = dynamic_cast<FriNode*>(source_nodes[z]);

    for (int k = 0; k < 3; ++k)
    {
      dslipgp[0][source_node->dofs()[k]] += source_val[z] * tanv1[k];
      dslipgp[1][source_node->dofs()[k]] += source_val[z] * tanv2[k];

      for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      {
        dslipgp[0][p->first] +=
            tanv1[k] * source_deriv(z, 0) *
            (source_elem.get_nodal_coords(k, z) - source_elem.get_nodal_coords_old(k, z)) *
            (p->second);
        dslipgp[1][p->first] +=
            tanv2[k] * source_deriv(z, 0) *
            (source_elem.get_nodal_coords(k, z) - source_elem.get_nodal_coords_old(k, z)) *
            (p->second);
      }

      for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
      {
        dslipgp[0][p->first] +=
            tanv1[k] * source_deriv(z, 1) *
            (source_elem.get_nodal_coords(k, z) - source_elem.get_nodal_coords_old(k, z)) *
            (p->second);
        dslipgp[1][p->first] +=
            tanv2[k] * source_deriv(z, 1) *
            (source_elem.get_nodal_coords(k, z) - source_elem.get_nodal_coords_old(k, z)) *
            (p->second);
      }
    }
  }

  for (int z = 0; z < ncol; ++z)
  {
    FriNode* target_node = dynamic_cast<FriNode*>(target_nodes[z]);

    for (int k = 0; k < 3; ++k)
    {
      dslipgp[0][target_node->dofs()[k]] -= target_val[z] * tanv1[k];
      dslipgp[1][target_node->dofs()[k]] -= target_val[z] * tanv2[k];

      for (CI p = d_target_xi_gp[0].begin(); p != d_target_xi_gp[0].end(); ++p)
      {
        dslipgp[0][p->first] -=
            tanv1[k] * target_deriv(z, 0) *
            (target_elem.get_nodal_coords(k, z) - target_elem.get_nodal_coords_old(k, z)) *
            (p->second);
        dslipgp[1][p->first] -=
            tanv2[k] * target_deriv(z, 0) *
            (target_elem.get_nodal_coords(k, z) - target_elem.get_nodal_coords_old(k, z)) *
            (p->second);
      }

      for (CI p = d_target_xi_gp[1].begin(); p != d_target_xi_gp[1].end(); ++p)
      {
        dslipgp[0][p->first] -=
            tanv1[k] * target_deriv(z, 1) *
            (target_elem.get_nodal_coords(k, z) - target_elem.get_nodal_coords_old(k, z)) *
            (p->second);
        dslipgp[1][p->first] -=
            tanv2[k] * target_deriv(z, 1) *
            (target_elem.get_nodal_coords(k, z) - target_elem.get_nodal_coords_old(k, z)) *
            (p->second);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Compute slipincr lin at GP                               farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_2d_slip_incr_lin(int& iter, Mortar::Element& source_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    double& jac, double& wgt, double* jumpvalv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const Core::Gen::Pairedvector<int, double>& dslipgp,
    const Core::Gen::Pairedvector<int, double>& derivjac,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  Core::Nodes::Node** source_nodes = source_elem.nodes();

  const int nrow = source_elem.num_node();
  double fac = 0.0;

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  FriNode* source_node = dynamic_cast<FriNode*>(source_nodes[iter]);
  if (source_node->is_on_boundor_ce()) return;

  // get the corresponding map as a reference
  std::map<int, double>& djumpmap = source_node->fri_data().get_deriv_var_jump()[0];

  // (1) Lin(Phi) - dual shape functions
  if (shape_fcn() == Mortar::shape_dual)
  {
    for (int m = 0; m < nrow; ++m)
    {
      fac = wgt * source_val[m] * jumpvalv[0] * jac;
      for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
               dualmap.begin();
          p != dualmap.end(); ++p)
        djumpmap[p->first] += fac * (p->second)(iter, m);
    }
  }

  // (2) Lin(Phi) - source GP coordinates
  fac = wgt * lm_deriv(iter, 0) * jumpvalv[0] * jac;
  for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
    djumpmap[p->first] += fac * (p->second);

  // (3) Lin(g) - gap function
  fac = wgt * lm_val[iter] * jac;
  for (CI p = dslipgp.begin(); p != dslipgp.end(); ++p) djumpmap[p->first] += fac * (p->second);

  // (4) Lin(dxdsxi) - source GP Jacobian
  fac = wgt * lm_val[iter] * jumpvalv[0];
  for (CI p = derivjac.begin(); p != derivjac.end(); ++p) djumpmap[p->first] += fac * (p->second);
}

/*----------------------------------------------------------------------*
 |  Compute slipincr lin at   GP                             farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_3d_slip_incr_lin(int& iter, Mortar::Element& source_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    double& jac, double& wgt, double* jumpvalv,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dslipgp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  Core::Nodes::Node** source_nodes = source_elem.nodes();

  double nrow = source_elem.num_node();

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  FriNode* source_node = dynamic_cast<FriNode*>(source_nodes[iter]);

  if (source_node->is_on_boundor_ce()) return;

  // get the corresponding map as a reference
  std::map<int, double>& djumpmap1 = source_node->fri_data().get_deriv_var_jump()[0];
  std::map<int, double>& djumpmap2 = source_node->fri_data().get_deriv_var_jump()[1];

  double fac1 = 0.0;
  double fac2 = 0.0;

  // (1) Lin(Phi) - dual shape functions
  if (shape_fcn() == Mortar::shape_dual)
  {
    for (int m = 0; m < nrow; ++m)
    {
      fac1 = wgt * source_val[m] * jumpvalv[0] * jac;
      fac2 = wgt * source_val[m] * jumpvalv[1] * jac;

      for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
               dualmap.begin();
          p != dualmap.end(); ++p)
      {
        djumpmap1[p->first] += fac1 * (p->second)(iter, m);
        djumpmap2[p->first] += fac2 * (p->second)(iter, m);
      }
    }
  }

  // (2) Lin(Phi) - source GP coordinates --> because of duality
  fac1 = wgt * lm_deriv(iter, 0) * jumpvalv[0] * jac;
  fac2 = wgt * lm_deriv(iter, 0) * jumpvalv[1] * jac;
  for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
  {
    djumpmap1[p->first] += fac1 * (p->second);
    djumpmap2[p->first] += fac2 * (p->second);
  }

  fac1 = wgt * lm_deriv(iter, 1) * jumpvalv[0] * jac;
  fac2 = wgt * lm_deriv(iter, 1) * jumpvalv[1] * jac;
  for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
  {
    djumpmap1[p->first] += fac1 * (p->second);
    djumpmap2[p->first] += fac2 * (p->second);
  }

  // (3) Lin(w) - wear function
  fac1 = wgt * lm_val[iter] * jac;
  for (CI p = dslipgp[0].begin(); p != dslipgp[0].end(); ++p)
    djumpmap1[p->first] += fac1 * (p->second);
  for (CI p = dslipgp[1].begin(); p != dslipgp[1].end(); ++p)
    djumpmap2[p->first] += fac1 * (p->second);

  // (5) Lin(dxdsxi) - source GP Jacobian
  fac1 = wgt * lm_val[iter] * jumpvalv[0];
  fac2 = wgt * lm_val[iter] * jumpvalv[1];
  for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
  {
    djumpmap1[p->first] += fac1 * (p->second);
    djumpmap2[p->first] += fac2 * (p->second);
  }
}

/*----------------------------------------------------------------------*
 |  Lin wear for impl. algor.                                farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_2d_wear_lin(int& iter, Mortar::Element& source_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    double& jac, double* gpn, double& wgt, double& wearval, double* jumpval,
    const Core::Gen::Pairedvector<int, double>& dweargp,
    const Core::Gen::Pairedvector<int, double>& derivjac,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  const double wcoeff = wearcoeff_ + wearcoeffm_;

  double facw = 0.0;
  const int nrow = source_elem.num_node();

  Core::Nodes::Node** source_nodes = source_elem.nodes();

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // get the corresponding map as a reference
  CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(source_nodes[iter]);

  std::map<int, double>& dwmap = cnode->data().get_deriv_w();

  if (shape_fcn() == Mortar::shape_petrovgalerkin)
  {
    // (1) Lin(Phi) - dual shape functions resulting from weighting the wear
    // we use std. shape functions for shape_petrovgalerkin

    // (2) Lin(Phi) - source GP coordinates --> because of duality
    facw = wcoeff * wgt * source_deriv(iter, 0) * wearval * jac;
    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      dwmap[p->first] += facw * (p->second);

    // (3) Lin(w) - wear function
    facw = wcoeff * wgt * source_val[iter] * jac;
    for (CI p = dweargp.begin(); p != dweargp.end(); ++p) dwmap[p->first] += facw * (p->second);

    // (4) Lin(dxdsxi) - source GP Jacobian
    facw = wcoeff * wgt * source_val[iter] * wearval;
    for (CI p = derivjac.begin(); p != derivjac.end(); ++p) dwmap[p->first] += facw * (p->second);
  }
  else  // no petrov_galerkin
  {
    // (1) Lin(Phi) - dual shape functions resulting from weighting the wear
    if (shape_fcn() == Mortar::shape_dual)
    {
      for (int m = 0; m < nrow; ++m)
      {
        facw = wcoeff * wgt * source_val[m] * wearval * jac;
        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 dualmap.begin();
            p != dualmap.end(); ++p)
          dwmap[p->first] += facw * (p->second)(iter, m);
      }
    }

    // (2) Lin(Phi) - source GP coordinates --> because of duality
    facw = wcoeff * wgt * lm_deriv(iter, 0) * wearval * jac;
    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      dwmap[p->first] += facw * (p->second);

    // (3) Lin(w) - wear function
    facw = wcoeff * wgt * lm_val[iter] * jac;
    for (CI p = dweargp.begin(); p != dweargp.end(); ++p) dwmap[p->first] += facw * (p->second);

    // (4) Lin(dxdsxi) - source GP Jacobian
    facw = wcoeff * wgt * lm_val[iter] * wearval;
    for (CI p = derivjac.begin(); p != derivjac.end(); ++p) dwmap[p->first] += facw * (p->second);
  }

  //****************************************************************
  // LIN WEAR W.R.T. LM
  //****************************************************************
  std::map<int, double>& dwlmmap = cnode->data().get_deriv_wlm();

  for (int bl = 0; bl < nrow; ++bl)
  {
    Mortar::Node* wearnode = dynamic_cast<Mortar::Node*>(source_nodes[bl]);

    if (shape_fcn() == Mortar::shape_petrovgalerkin)
    {
      dwlmmap[wearnode->dofs()[0]] +=
          wcoeff * source_val[iter] * jac * wgt * abs(jumpval[0]) * gpn[0] * lm_val[bl];
      dwlmmap[wearnode->dofs()[1]] +=
          wcoeff * source_val[iter] * jac * wgt * abs(jumpval[0]) * gpn[1] * lm_val[bl];
    }
    else
    {
      dwlmmap[wearnode->dofs()[0]] +=
          wcoeff * lm_val[iter] * jac * wgt * abs(jumpval[0]) * gpn[0] * lm_val[bl];
      dwlmmap[wearnode->dofs()[1]] +=
          wcoeff * lm_val[iter] * jac * wgt * abs(jumpval[0]) * gpn[1] * lm_val[bl];
    }
  }
}

/*----------------------------------------------------------------------*
 |  Lin wear for impl. algor.                                farah 09/13|
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_3d_wear_lin(int& iter, Mortar::Element& source_elem,
    Core::LinAlg::SerialDenseVector& source_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    double& jac, double* gpn, double& wgt, double& wearval, double* jumpval,
    const Core::Gen::Pairedvector<int, double>& dweargp,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  double facw = 0.0;
  const int nrow = source_elem.num_node();

  Core::Nodes::Node** source_nodes = source_elem.nodes();

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // get the corresponding map as a reference
  CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(source_nodes[iter]);

  // get the corresponding map as a reference
  std::map<int, double>& dwmap = cnode->data().get_deriv_w();

  if (shape_fcn() == Mortar::shape_petrovgalerkin)
  {
    // (1) Lin(Phi) - dual shape functions resulting from weighting the wear
    // --

    // (2) Lin(Phi) - source GP coordinates --> because of duality
    facw = wearcoeff_ * wgt * source_deriv(iter, 0) * wearval * jac;
    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      dwmap[p->first] += facw * (p->second);

    facw = wearcoeff_ * wgt * source_deriv(iter, 1) * wearval * jac;
    for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
      dwmap[p->first] += facw * (p->second);

    // (3) Lin(w) - wear function
    facw = wearcoeff_ * wgt * source_val[iter] * jac;
    for (CI p = dweargp.begin(); p != dweargp.end(); ++p) dwmap[p->first] += facw * (p->second);

    // (5) Lin(dxdsxi) - source GP Jacobian
    facw = wearcoeff_ * wgt * source_val[iter] * wearval;
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
      dwmap[p->first] += facw * (p->second);
  }
  else  // no pg
  {
    // (1) Lin(Phi) - dual shape functions resulting from weighting the wear
    if (shape_fcn() == Mortar::shape_dual)
    {
      for (int m = 0; m < nrow; ++m)
      {
        facw = wearcoeff_ * wgt * source_val[m] * wearval * jac;
        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 dualmap.begin();
            p != dualmap.end(); ++p)
          dwmap[p->first] += facw * (p->second)(iter, m);
      }
    }

    // (2) Lin(Phi) - source GP coordinates --> because of duality
    facw = wearcoeff_ * wgt * lm_deriv(iter, 0) * wearval * jac;
    for (CI p = d_source_xi_gp[0].begin(); p != d_source_xi_gp[0].end(); ++p)
      dwmap[p->first] += facw * (p->second);

    facw = wearcoeff_ * wgt * lm_deriv(iter, 1) * wearval * jac;
    for (CI p = d_source_xi_gp[1].begin(); p != d_source_xi_gp[1].end(); ++p)
      dwmap[p->first] += facw * (p->second);

    // (3) Lin(w) - wear function
    facw = wearcoeff_ * wgt * lm_val[iter] * jac;
    for (CI p = dweargp.begin(); p != dweargp.end(); ++p) dwmap[p->first] += facw * (p->second);

    // (5) Lin(dxdsxi) - source GP Jacobian
    facw = wearcoeff_ * wgt * lm_val[iter] * wearval;
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
      dwmap[p->first] += facw * (p->second);
  }


  //****************************************************************
  // LIN WEAR W.R.T. LM
  //****************************************************************
  std::map<int, double>& dwlmmap = cnode->data().get_deriv_wlm();

  for (int bl = 0; bl < nrow; ++bl)
  {
    Mortar::Node* wearnode = dynamic_cast<Mortar::Node*>(source_nodes[bl]);

    if (shape_fcn() == Mortar::shape_petrovgalerkin)
    {
      dwlmmap[wearnode->dofs()[0]] +=
          wearcoeff_ * source_val[iter] * jac * wgt * abs(jumpval[0]) * gpn[0] * lm_val[bl];
      dwlmmap[wearnode->dofs()[1]] +=
          wearcoeff_ * source_val[iter] * jac * wgt * abs(jumpval[0]) * gpn[1] * lm_val[bl];
      dwlmmap[wearnode->dofs()[2]] +=
          wearcoeff_ * source_val[iter] * jac * wgt * abs(jumpval[0]) * gpn[2] * lm_val[bl];
    }
    else
    {
      dwlmmap[wearnode->dofs()[0]] +=
          wearcoeff_ * lm_val[iter] * jac * wgt * abs(jumpval[0]) * gpn[0] * lm_val[bl];
      dwlmmap[wearnode->dofs()[1]] +=
          wearcoeff_ * lm_val[iter] * jac * wgt * abs(jumpval[0]) * gpn[1] * lm_val[bl];
      dwlmmap[wearnode->dofs()[2]] +=
          wearcoeff_ * lm_val[iter] * jac * wgt * abs(jumpval[0]) * gpn[2] * lm_val[bl];
    }
  }
}


/*----------------------------------------------------------------------*
 |  Compute entries for poro normal coupling condition      07/14 ager  |
 *----------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_ncoup_deriv(Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& target_deriv,
    double* ncoup, double* gpn, double& jac, double& wgt, double* gpcoord,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_gp,
    std::map<int, double>& dncoupgp, std::map<int, double>& dvelncoupgp,
    std::map<int, double>& dpresncoupgp,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, bool quadratic, int nintrow)
{
  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  Core::Nodes::Node** target_nodes = target_elem.nodes();
  if (!source_nodes) FOUR_C_THROW("Null pointer!");
  if (!target_nodes) FOUR_C_THROW("Null pointer!");

  // bool to decide if fluid quantities and structural velocities
  // exist for source side on Node level and contribute to porofluid meshtying
  bool sourceporo = (source_elem.phys_type() == Mortar::Element::poro);
  // bool to decide if fluid quantities and structural velocities
  // exist for target side on Node level and contribute to porofluid meshtying
  bool targetporo = (target_elem.phys_type() == Mortar::Element::poro);

  if (!sourceporo and targetporo)
    FOUR_C_THROW(
        "poroelastic mesh tying method needs the source side to be poroelastic (invert "
        "target/source "
        "definition)");
  // see CONTACT::LagrangeStrategyPoro::LagrangeStrategyPoro for comment

  // number of nodes (source, target)
  int nrow = source_elem.num_node();
  // if(twosided) //needed only for target side
  int ncol = target_elem.num_node();  // not used for onesided porocontact!!!

  // get fluid velocities in GP
  std::array<double, 3> sgpfvel = {0.0, 0.0, 0.0};
  std::array<double, 3> sgpsvel = {0.0, 0.0, 0.0};

  std::array<double, 3> mgpfvel = {0.0, 0.0, 0.0};
  std::array<double, 3> mgpsvel = {0.0, 0.0, 0.0};

  // fill local vectors for corresponding problem dimension
  for (int k = 0; k < n_dim(); ++k)
  {
    if (sourceporo)
      for (int i = 0; i < nrow; ++i)
      {
        Node* mymrtrnode = dynamic_cast<Node*>(source_nodes[i]);
        sgpfvel[k] += source_val[i] * mymrtrnode->poro_data().fvel()[k];
        sgpsvel[k] += source_val[i] * mymrtrnode->poro_data().svel()[k];
      }

    if (targetporo)
    {
      for (int i = 0; i < ncol; ++i)
      {
        Node* mymrtrnode = dynamic_cast<Node*>(target_nodes[i]);
        mgpfvel[k] += target_val[i] * mymrtrnode->poro_data().fvel()[k];
        mgpsvel[k] += target_val[i] * mymrtrnode->poro_data().svel()[k];
      }
    }
  }

  ////////////////////////////////!!!Calculate source side Porosity!!!////////////////////////////
  // get J
  std::map<int, double>
      sJLin;  // source map for linearizations of the Deformation Gradient Determinant
  // get fluid pressure in GP
  double sgpfpres = 0.0;
  // porosity to be written into
  double source_porosity;
  double sdphi_dp;  // linearization terms to be written into
  double sdphi_dJ;  // linearization terms to be written into

  if (sourceporo)
  {
    const double sJ = det_deformation_gradient(source_elem, wgt, gpcoord, sJLin);

    for (int i = 0; i < nrow; ++i)
    {
      Node* mymrtrnode = dynamic_cast<Node*>(source_nodes[i]);
      sgpfpres += source_val[i] * mymrtrnode->poro_data().fpres()[0];
    }
    Teuchos::ParameterList sparams;  // empty parameter list;

    std::shared_ptr<Mat::StructPoro> sstructmat =
        std::dynamic_pointer_cast<Mat::StructPoro>(source_elem.parent_element()->material(0));
    if (sstructmat == nullptr)
      sstructmat =
          std::dynamic_pointer_cast<Mat::StructPoro>(source_elem.parent_element()->material(1));
    if (sstructmat == nullptr) FOUR_C_THROW("Cast to StructPoro failed!");
    sstructmat->compute_surf_porosity(sparams, sgpfpres, sJ, source_elem.face_parent_number(),
        1,  // finally check what to do here Todo:
        source_porosity,
        &sdphi_dp,  // linearization of phi w.r.t. pressure
        &sdphi_dJ,  // linearization of phi w.r.t. defo-grad determinant
        nullptr,    // dphi_dJdp not needed
        nullptr,    // dphi_dJJ not needed
        nullptr,    // dphi_dpp not needed
        false,
        source_elem.id());
  }

  ////////////////////////////////!!!Calculate target side Porosity!!!////////////////////////////
  // get J

  std::map<int, double>
      mJLin;  // target map for linearizations of the Deformation Gradient Determinant
  // get fluid pressure in GP
  double mgpfpres = 0.0;
  // porosity to be written into
  double target_porosity;
  double mdphi_dp;  // linearization terms to be written into
  double mdphi_dJ;  // linearization terms to be written into

  if (targetporo)
  {
    const double mJ = det_deformation_gradient(target_elem, wgt, gpcoord, mJLin);

    for (int i = 0; i < ncol; ++i)
    {
      Node* mymrtrnode = dynamic_cast<Node*>(target_nodes[i]);
      mgpfpres += target_val[i] * mymrtrnode->poro_data().fpres()[0];
    }
    Teuchos::ParameterList mparams;  // empty parameter list;

    std::shared_ptr<Mat::StructPoro> mstructmat =
        std::dynamic_pointer_cast<Mat::StructPoro>(target_elem.parent_element()->material(0));
    if (mstructmat == nullptr)
      mstructmat =
          std::dynamic_pointer_cast<Mat::StructPoro>(target_elem.parent_element()->material(1));
    if (mstructmat == nullptr) FOUR_C_THROW("Cast to StructPoro failed!");

    mstructmat->compute_surf_porosity(mparams, mgpfpres, mJ,
        target_elem.face_parent_number(),  // may not work
        1,                                 // finally check what to do here Todo:
        target_porosity,
        &mdphi_dp,  // linearization of phi w.r.t. pressure
        &mdphi_dJ,  // linearization of phi w.r.t. defo-grad determinant
        nullptr,    // dphi_dJdp not needed
        nullptr,    // dphi_dJJ not needed
        nullptr,    // dphi_dpp not needed
        false,
        source_elem.id()
      );
  }
  ////////////////////////////////!!!Calculate Porosity done!!!////////////////////////////
  // build normal coupling term at current GP
  for (int i = 0; i < n_dim(); ++i)
  {
    if (sourceporo) ncoup[0] += (sgpsvel[i] - sgpfvel[i]) * gpn[i] * source_porosity;
    if (targetporo) ncoup[0] -= (mgpsvel[i] - mgpfvel[i]) * gpn[i] * target_porosity;
  }
  // **************************
  // add to node
  // **************************
  if (!quadratic)
  {
    for (int j = 0; j < nrow; ++j)
    {
      CONTACT::Node* mrtrnode = dynamic_cast<CONTACT::Node*>(source_nodes[j]);

      double prod = 0.0;
      // Petrov-Galerkin approach (dual LM for D/M but standard LM for normal coupling)
      if (shape_fcn() == Mortar::shape_petrovgalerkin) prod = source_val[j] * ncoup[0] * jac * wgt;
      // usual standard or dual LM approach
      else
        prod = lm_val[j] * ncoup[0] * jac * wgt;

      // do not process source side boundary nodes
      // (their row entries would be zero anyway!)
      if (mrtrnode->is_on_bound())
      {
        FOUR_C_THROW("CHECKME: isonbound_ active; should not happen here");
        continue;
      }

      // add current Gauss point's contribution to gseg
      mrtrnode->add_ncoup_value(prod);
    }
  }

  //    // CASE 4: Dual LM shape functions and quadratic interpolation
  //    else if ((ShapeFcn() == Mortar::shape_dual || shape_fcn() ==
  //    Mortar::shape_petrovgalerkin) &&
  //        LagMultQuad() == Mortar::lagmult_quad)
  //    {
  //      for (int j=0;j<nrow;++j)
  //      {
  //        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(source_nodes[j]);
  //
  //        double prod = 0.0;
  //        prod = lm_val[j]*gap[0]*jac*wgt;
  //
  //        // add current Gauss point's contribution to gseg
  //        cnode->AddgValue(prod);
  //      }
  //    }

  // INVALID CASES
  else
  {
    FOUR_C_THROW("Invalid integration case for 3D quadratic normal coupling mortar!");
  }
  //}
  // **************************
  // Linearization w.r.t. displacements
  // **************************

  // add everything to dncoupgp
  // dncoupgp ... (v(struct)-v(fluid)) * delta n
  for (int k = 0; k < n_dim(); ++k)
  {
    for (CI p = dnmap_unit[k].begin(); p != dnmap_unit[k].end(); ++p)
    {
      double sourceside = 0.0;
      if (sourceporo)
        sourceside =
            source_porosity * (sgpsvel[k] - sgpfvel[k]);  // h.Willmann source terms preparation

      double targetside = 0.0;
      if (targetporo)
        targetside =
            target_porosity * (mgpsvel[k] - mgpfvel[k]);  // h.Willmann target terms preparation

      dncoupgp[p->first] += (sourceside - targetside) * (p->second);  // linearization of n,
    }
  }

  double timefac =
      imortar_.get<double>("porotimefac");  // TODO: move in final version to other place ChrAg

  if (sourceporo)
  {
    for (int i = 0; i < nrow; ++i)
    {
      Node* mrtrnode = dynamic_cast<Node*>(source_nodes[i]);

      for (int k = 0; k < n_dim(); ++k)
      {
        dncoupgp[mrtrnode->dofs()[k]] += source_porosity * source_val[i] * gpn[k] *
                                         timefac;  // linearisation of vs add for target side
        for (unsigned int d = 0; d < d_source_xi_gp.size(); ++d)
        {
          for (CI p = d_source_xi_gp[d].begin(); p != d_source_xi_gp[d].end();
              ++p)  // linearization of shape function N
          {
            dncoupgp[p->first] -= source_porosity * gpn[k] * source_deriv(i, d) *
                                  mrtrnode->poro_data().fvel()[k] * (p->second);
            dncoupgp[p->first] += source_porosity * gpn[k] * source_deriv(i, d) *
                                  mrtrnode->poro_data().svel()[k] * (p->second);
          }
        }
      }
    }
  }  // if(sourceporo)

  // h.Willmann target side is considered now
  // TARGET
  if (targetporo)
  {
    // lin target nodes
    for (int i = 0; i < ncol; ++i)
    {
      Node* mrtrnode = dynamic_cast<Node*>(target_nodes[i]);

      for (int k = 0; k < n_dim(); ++k)
      {
        dncoupgp[mrtrnode->dofs()[k]] -= target_porosity * target_val[i] * gpn[k] * timefac;
        for (unsigned int d = 0; d < d_target_xi_gp.size(); ++d)
        {
          for (CI p = d_target_xi_gp[d].begin(); p != d_target_xi_gp[d].end(); ++p)
          {
            dncoupgp[p->first] += target_porosity * gpn[k] * target_deriv(i, d) *
                                  mrtrnode->poro_data().fvel()[k] * (p->second);
            dncoupgp[p->first] -= target_porosity * gpn[k] * target_deriv(i, d) *
                                  mrtrnode->poro_data().svel()[k] * (p->second);
          }
        }
      }
    }
  }

  // h.Willmann Write Deformation Gradient Determinant Linearization to map
  using I = std::map<int, double>::iterator;

  if (sourceporo)
  {
    for (I p = sJLin.begin(); p != sJLin.end(); ++p)
    {
      for (int i = 0; i < n_dim(); ++i)
      {
        dncoupgp[p->first] += (sgpsvel[i] - sgpfvel[i]) * sdphi_dJ * gpn[i] * p->second;
      }
    }
  }
  if (targetporo)
  {
    for (I p = mJLin.begin(); p != mJLin.end(); ++p)
    {
      for (int i = 0; i < n_dim(); ++i)
      {
        dncoupgp[p->first] -= (mgpsvel[i] - mgpfvel[i]) * mdphi_dJ * gpn[i] * p->second;
      }
    }
  }

  // **************************
  // Linearization w.r.t. fluid velocities
  // **************************

  if (sourceporo)
  {
    for (int z = 0; z < nrow; ++z)
    {
      Node* mrtrnode = dynamic_cast<Node*>(source_nodes[z]);

      for (int k = 0; k < n_dim(); ++k)
      {
        dvelncoupgp[mrtrnode->dofs()[k]] -= source_porosity * source_val[z] * gpn[k];
        // Because Source Discretisation should be the poro dis!!!  --- add target for two sided
        // poro contact!!!

        // h.Willmann linearization w.r.t. fluid pressure:
        dpresncoupgp[mrtrnode->dofs()[0]] +=
            sdphi_dp * source_val[z] * (sgpsvel[k] - sgpfvel[k]) * gpn[k];
        //      std::cout<<mrtrnode->Dofs()[0]<<"mrtrnode->Dofs()[0]"<<std::endl;
      }
    }
  }
  // h.Willmann terms added for target side
  if (targetporo)
  {
    for (int z = 0; z < ncol; ++z)
    {
      Node* mrtrnode = dynamic_cast<Node*>(target_nodes[z]);

      for (int k = 0; k < n_dim(); ++k)
      {
        dvelncoupgp[mrtrnode->dofs()[k]] += target_porosity * target_val[z] * gpn[k];

        // h.Willmann linearization w.r.t. fluid pressure:
        dpresncoupgp[mrtrnode->dofs()[0]] -=
            mdphi_dp * target_val[z] * (mgpsvel[k] - mgpfvel[k]) * gpn[k];
        //       std::cout<<mrtrnode->Dofs()[0]<<"mrtrnode->Dofs()[0]"<<std::endl;
      }
    }
  }
}

/*-----------------------------------------------------------------------------*
 |  Do lin. entries for weighted normal coupling condition at GP     ager 06/14|
 |  modified by h.Willmann 2015                                                |
 *----------------------------------------------------------------------------*/
void inline CONTACT::Integrator::gp_ncoup_lin(int& iter, Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& target_val, Core::LinAlg::SerialDenseVector& lm_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& lm_deriv,
    double& ncoup, double* gpn, double& jac, double& wgt, const std::map<int, double>& dncoupgp,
    const std::map<int, double>& dvelncoupgp, const std::map<int, double>& dpresncoupgp,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi_gp,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap)
{
  int nrow = source_elem.num_node();
  // only source_elem.num_node() is needed because the integration of the interface constraint is
  // evaluated on the source side interface surface

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // get source element nodes themselves
  Core::Nodes::Node** source_nodes = source_elem.nodes();
  // Core::Nodes::Node** target_nodes = target_elem.Nodes();

  CONTACT::Node* mymrtrnode = dynamic_cast<CONTACT::Node*>(source_nodes[iter]);
  if (!mymrtrnode) FOUR_C_THROW("CONTACT::Integrator::gp_ncoup_lin: mymrtnode: Null pointer!");

  double fac = 0.0;

  // get the corresponding map as a reference
  std::map<int, double>& dgmap = mymrtrnode->poro_data().get_derivn_coup();
  // switch if Petrov-Galerkin approach for LM is applied
  // for meshtying shape_dual is the only allowed case as long as porofluid and skeleton
  // meshtying conditions share the same input lines
  //(for contact they still share the input lines but shape_petrovgalerkin is allowed for
  // skeleton/structural contact) shape_petrovgalerkin
  if (shape_fcn() == Mortar::shape_petrovgalerkin)
  {
    // (1) Lin(Phi) - does not exist here for Petrov-Galerkin approach

    // (2) Lin(N) - source GP coordinates
    for (unsigned int k = 0; k < d_source_xi_gp.size(); ++k)
    {
      fac = wgt * source_deriv(iter, k) * ncoup * jac;
      for (CI p = d_source_xi_gp[k].begin(); p != d_source_xi_gp[k].end(); ++p)
        dgmap[p->first] += fac * (p->second);
    }
    // (3) Lin(g) - normal coupling condition ncoup function
    fac = wgt * source_val[iter] * jac;
    for (auto p = dncoupgp.begin(); p != dncoupgp.end(); ++p) dgmap[p->first] += fac * (p->second);

    // (4) Lin(dsxideta) - intcell GP Jacobian
    fac = wgt * source_val[iter] * ncoup;
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
      dgmap[p->first] += fac * (p->second);
  }

  // the usual standard or dual LM approach
  else
  {
    // (1) Lin(Phi) - dual shape functions
    if (dualmap.size() > 0)
      for (int m = 0; m < nrow; ++m)
      {
        fac = wgt * source_val[m] * ncoup * jac;
        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 dualmap.begin();
            p != dualmap.end(); ++p)
        {
          dgmap[p->first] += fac * (p->second)(iter, m);
        }
      }

    // (2) Lin(Phi) - source GP coordinates
    for (unsigned int k = 0; k < d_source_xi_gp.size(); ++k)
    {
      fac = wgt * lm_deriv(iter, k) * ncoup * jac;
      for (CI p = d_source_xi_gp[k].begin(); p != d_source_xi_gp[k].end(); ++p)
      {
        dgmap[p->first] += fac * (p->second);
      }
    }

    // (3) Lin(g) - normal coupling condition ncoup function
    fac = wgt * lm_val[iter] * jac;
    for (auto p = dncoupgp.begin(); p != dncoupgp.end(); ++p)
    {
      dgmap[p->first] += fac * (p->second);
    }

    // (4) Lin(dsxideta) - intcell GP Jacobian
    fac = wgt * lm_val[iter] * ncoup;
    for (CI p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
    {
      dgmap[p->first] += fac * (p->second);
    }
  }

  // velocity linearisation of the ncoupling condition!
  // get the corresponding map as a reference
  std::map<int, double>& dvelncoupmap = mymrtrnode->poro_data().get_vel_derivn_coup();
  // switch if Petrov-Galerkin approach for LM is applied
  if (shape_fcn() == Mortar::shape_petrovgalerkin)
  {
    // (3) Lin(g) - normal coupling condition function
    fac = wgt * source_val[iter] * jac;
    for (auto p = dvelncoupgp.begin(); p != dvelncoupgp.end(); ++p)
      dvelncoupmap[p->first] += fac * (p->second);
  }
  // the usual standard or dual LM approach
  else
  {
    // (3) Lin(g) - normal coupling condition function
    fac = wgt * lm_val[iter] * jac;
    for (auto p = dvelncoupgp.begin(); p != dvelncoupgp.end(); ++p)
    {
      dvelncoupmap[p->first] += fac * (p->second);
    }
  }

  // h.Willmann
  // pressure linearisation of the ncoupling condition!
  // get the corresponding map as a reference
  std::map<int, double>& dpresncoupmap = mymrtrnode->poro_data().get_pres_derivn_coup();
  // switch if Petrov-Galerkin approach for LM is applied
  if (shape_fcn() == Mortar::shape_petrovgalerkin)
  {
    // (3) Lin(g) - normal coupling condition function
    fac = wgt * source_val[iter] * jac;
    for (auto p = dpresncoupgp.begin(); p != dpresncoupgp.end(); ++p)
      dpresncoupmap[p->first] += fac * (p->second);
  }
  // the usual standard or dual LM approach
  else
  {
    // (3) Lin(g) - normal coupling condition function
    fac = wgt * lm_val[iter] * jac;
    for (auto p = dpresncoupgp.begin(); p != dpresncoupgp.end(); ++p)
    {
      dpresncoupmap[p->first] += fac * (p->second);
    }
  }
}

/*-----------------------------------------------------------------------------*
 |  Calculate Determinate of the Deformation Gradient at GP          ager 10/14|
 *----------------------------------------------------------------------------*/
double CONTACT::Integrator::det_deformation_gradient(
    Mortar::Element& source_elem, double& wgt, double* gpcoord, std::map<int, double>& JLin)
{
  double J;
  Core::FE::CellType distype = source_elem.parent_element()->shape();
  switch (distype)
  {
    case Core::FE::CellType::hex8:
      J = t_det_deformation_gradient<Core::FE::CellType::hex8, 3>(source_elem, wgt, gpcoord, JLin);
      break;
    case Core::FE::CellType::quad4:
      J = t_det_deformation_gradient<Core::FE::CellType::quad4, 2>(source_elem, wgt, gpcoord, JLin);
      break;
    default:
      FOUR_C_THROW(
          "DetDeformationGradient3D: Parent Element Type not templated yet, just add it here!");
      J = 0.0;  // To avoid warning!
      break;
  }

  return J;
}
/*-------------------------------------------------------------------------------*
 |  Templated Calculate Determinant of the Deformation Gradient at GP  ager 10/14|
 |  modified by h.Willmann 2015                                                  |
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType parentdistype, int dim>
double CONTACT::Integrator::t_det_deformation_gradient(
    Mortar::Element& source_elem, double& wgt, double* gpcoord, std::map<int, double>& JLin)
{
  //! nen_: number of element nodes (T. Hughes: The Finite Element Method)
  static const int numnodes = Core::FE::num_nodes(parentdistype);

  Core::FE::CollectedGaussPoints intpoints =
      Core::FE::CollectedGaussPoints(1);  // reserve just for 1 entry ...
  intpoints.append(gpcoord[0], gpcoord[1], 0.0, wgt);

  // get coordinates of gauss point w.r.t. local parent coordinate system
  Core::LinAlg::SerialDenseMatrix pqxg(1, dim);
  Core::LinAlg::Matrix<dim, dim> derivtrafo(Core::LinAlg::Initialization::zero);

  Core::FE::boundary_gp_to_parent_gp<dim>(pqxg, derivtrafo, intpoints,
      source_elem.parent_element()->shape(), source_elem.shape(), source_elem.face_parent_number());

  Core::LinAlg::Matrix<dim, 1> pxsi(Core::LinAlg::Initialization::zero);

  // coordinates of the current integration point in parent coordinate system
  for (int idim = 0; idim < dim; idim++)
  {
    pxsi(idim) = pqxg(0, idim);
  }

  Core::LinAlg::Matrix<dim, numnodes> pderiv_loc(
      Core::LinAlg::Initialization::zero);  // derivatives of parent element shape functions in
                                            // parent element coordinate system

  // evaluate derivatives of parent element shape functions at current integration point in parent
  // coordinate system
  Core::FE::shape_function_deriv1<parentdistype>(pxsi, pderiv_loc);
  //
  // get Jacobian matrix and determinant w.r.t. spatial configuration
  //
  // |J| = det(xjm) * det(Jmat^-1) = det(xjm) * 1/det(Jmat)
  //
  //    _                     _
  //   |  x_1,1  x_2,1  x_3,1  |           d x_i
  //   |  x_1,2  x_2,2  x_3,2  | = xjm  = --------
  //   |_ x_1,3  x_2,3  x_3,3 _|           d s_j
  //    _                     _
  //   |  X_1,1  X_2,1  X_3,1  |           d X_i
  //   |  X_1,2  X_2,2  X_3,2  | = Jmat = --------
  //   |_ X_1,3  X_2,3  X_3,3 _|           d s_j
  //
  Core::LinAlg::Matrix<dim, dim> xjm;
  Core::LinAlg::Matrix<dim, dim> Jmat;

  Core::LinAlg::Matrix<dim, numnodes> xrefe(
      Core::LinAlg::Initialization::zero);  // material coord. of parent element
  Core::LinAlg::Matrix<dim, numnodes> xcurr(
      Core::LinAlg::Initialization::zero);  // current  coord. of parent element

  // update element geometry of parent element
  {
    Core::Nodes::Node** nodes = source_elem.parent_element()->nodes();
    for (int inode = 0; inode < numnodes; ++inode)
    {
      for (unsigned int idof = 0; idof < dim; ++idof)
      {
        const auto& x = nodes[inode]->x();
        xrefe(idof, inode) = x[idof];
        xcurr(idof, inode) =
            xrefe(idof, inode) + source_elem.mo_data().parent_disp()[inode * dim + idof];
        //          std::cout<< source_elem.MoData().ParentDisp()[inode*dim+idof] <<",
        //          source_elem.MoData().ParentDisp()[inode*dim+idof]"<<std::endl;
      }
    }
  }
  // std::cout<<xcurr<<"xcurr, "<<xrefe<<"xrefe"<<std::endl;
  xjm.multiply_nt(pderiv_loc, xcurr);
  Jmat.multiply_nt(pderiv_loc, xrefe);
  double det = xjm.determinant();
  double detJ = Jmat.determinant();
  const double J = det / detJ;

  // Linearisation of Jacobian (atm missing!)
  // D J[d] = J div(d) = J div(N_i d_i) = J dN_i/dx_j d_ij = J dN_i/dzeta_k dzeta_k/dx_j d_ij

  //  Core::LinAlg::Matrix<dim,numnodes>  auxJLin (true);   // matrix to be filled with
  //  linearization scalars
  xjm.invert();
  {
    //    Core::Nodes::Node** nodes = source_elem.parent_element()->Nodes();
    for (int inode = 0; inode < numnodes; ++inode)
    {
      for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = 0; j < dim; ++j)
        {
          JLin[source_elem.mo_data().parent_dof().at(inode * dim + i)] +=
              J * pderiv_loc(j, inode) * xjm(i, j);
          //          auxJLin(idof,inode)+=J*pderiv_loc(j,inode)*xjm(i,j); //matrix that could be
          //          filled with linearizations
        }
      }
    }
  }
  return J;
}

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mortar_integrator.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_mortar_calc_utils.hpp"
#include "4C_mortar_coupling3d_classes.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_node.hpp"
#include "4C_mortar_projector.hpp"
#include "4C_mortar_shape_utils.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  impl...                                                  farah 01/14|
 *----------------------------------------------------------------------*/
Mortar::Integrator* Mortar::Integrator::impl(
    Mortar::Element& sele, Mortar::Element& mele, Teuchos::ParameterList& params)
{
  switch (sele.shape())
  {
    // 2D surface elements
    case Core::FE::CellType::quad4:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::quad4:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad4,
              Core::FE::CellType::quad4>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::quad8:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad4,
              Core::FE::CellType::quad8>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::quad9:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad4,
              Core::FE::CellType::quad9>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::tri3:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad4,
              Core::FE::CellType::tri3>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::tri6:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad4,
              Core::FE::CellType::tri6>::instance(Core::Utils::SingletonAction::create, params);
        }
        default:
          FOUR_C_THROW("Element combination not allowed!");
      }
      break;
    }
    case Core::FE::CellType::quad8:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::quad4:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad8,
              Core::FE::CellType::quad4>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::quad8:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad8,
              Core::FE::CellType::quad8>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::quad9:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad8,
              Core::FE::CellType::quad9>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::tri3:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad8,
              Core::FE::CellType::tri3>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::tri6:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad8,
              Core::FE::CellType::tri6>::instance(Core::Utils::SingletonAction::create, params);
        }
        default:
          FOUR_C_THROW("Element combination not allowed!");
      }
      break;
    }
    case Core::FE::CellType::quad9:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::quad4:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad9,
              Core::FE::CellType::quad4>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::quad8:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad9,
              Core::FE::CellType::quad8>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::quad9:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad9,
              Core::FE::CellType::quad9>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::tri3:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad9,
              Core::FE::CellType::tri3>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::tri6:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::quad9,
              Core::FE::CellType::tri6>::instance(Core::Utils::SingletonAction::create, params);
        }
        default:
          FOUR_C_THROW("Element combination not allowed!");
      }
      break;
    }
    case Core::FE::CellType::tri3:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::quad4:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::tri3,
              Core::FE::CellType::quad4>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::quad8:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::tri3,
              Core::FE::CellType::quad8>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::quad9:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::tri3,
              Core::FE::CellType::quad9>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::tri3:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::tri3,
              Core::FE::CellType::tri3>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::tri6:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::tri3,
              Core::FE::CellType::tri6>::instance(Core::Utils::SingletonAction::create, params);
        }
        default:
          FOUR_C_THROW("Element combination not allowed!");
      }
      break;
    }
    case Core::FE::CellType::tri6:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::quad4:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::tri6,
              Core::FE::CellType::quad4>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::quad8:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::tri6,
              Core::FE::CellType::quad8>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::quad9:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::tri6,
              Core::FE::CellType::quad9>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::tri3:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::tri6,
              Core::FE::CellType::tri3>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::tri6:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::tri6,
              Core::FE::CellType::tri6>::instance(Core::Utils::SingletonAction::create, params);
        }
        default:
          FOUR_C_THROW("Element combination not allowed!");
      }
      break;
    }
      // 1D surface elements
    case Core::FE::CellType::line2:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::line2:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::line2,
              Core::FE::CellType::line2>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::line3:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::line2,
              Core::FE::CellType::line3>::instance(Core::Utils::SingletonAction::create, params);
        }
        default:
          FOUR_C_THROW("Element combination not allowed!");
      }
      break;
    }
    case Core::FE::CellType::line3:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::line2:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::line3,
              Core::FE::CellType::line2>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::line3:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::line3,
              Core::FE::CellType::line3>::instance(Core::Utils::SingletonAction::create, params);
        }
        default:
          FOUR_C_THROW("Element combination not allowed!");
      }
      break;
    }

      //==================================================
      //                     NURBS
      //==================================================
      // 1D surface elements
    case Core::FE::CellType::nurbs2:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::nurbs2:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::nurbs2,
              Core::FE::CellType::nurbs2>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::nurbs3:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::nurbs2,
              Core::FE::CellType::nurbs3>::instance(Core::Utils::SingletonAction::create, params);
        }
        default:
          FOUR_C_THROW("Element combination not allowed!");
      }
      break;
    }
    case Core::FE::CellType::nurbs3:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::nurbs2:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::nurbs3,
              Core::FE::CellType::nurbs2>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::nurbs3:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::nurbs3,
              Core::FE::CellType::nurbs3>::instance(Core::Utils::SingletonAction::create, params);
        }
        default:
          FOUR_C_THROW("Element combination not allowed!");
      }
      break;
    }
    case Core::FE::CellType::nurbs9:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::nurbs9:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::nurbs9,
              Core::FE::CellType::nurbs9>::instance(Core::Utils::SingletonAction::create, params);
        }
        case Core::FE::CellType::nurbs4:
        {
          return Mortar::IntegratorCalc<Core::FE::CellType::nurbs9,
              Core::FE::CellType::nurbs4>::instance(Core::Utils::SingletonAction::create, params);
        }
        default:
          FOUR_C_THROW("Element combination not allowed!");
      }
      break;
    }
    default:
      FOUR_C_THROW("Error...");
      break;
  }
  return nullptr;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            farah 01/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
Mortar::IntegratorCalc<distype_s, distype_m>::IntegratorCalc(const Teuchos::ParameterList& params)
    : imortar_(params),
      shapefcn_(Teuchos::getIntegralValue<Mortar::ShapeFcn>(params, "LM_SHAPEFCN")),
      lmquadtype_(Teuchos::getIntegralValue<Mortar::LagMultQuad>(params, "LM_QUAD"))
{
  initialize_gp();
}

template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
Mortar::IntegratorCalc<distype_s, distype_m>*
Mortar::IntegratorCalc<distype_s, distype_m>::instance(
    Core::Utils::SingletonAction action, const Teuchos::ParameterList& params)
{
  static auto singleton_owner = Core::Utils::make_singleton_owner(
      [](const Teuchos::ParameterList& p)
      {
        return std::unique_ptr<Mortar::IntegratorCalc<distype_s, distype_m>>(
            new Mortar::IntegratorCalc<distype_s, distype_m>(p));
      });

  return singleton_owner.instance(action, params);
}


/*----------------------------------------------------------------------*
 |  Initialize gauss points                                   popp 06/09|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void Mortar::IntegratorCalc<distype_s, distype_m>::initialize_gp()
{
  // get numgp (for element-based integration)
  int numgp = imortar_.get<int>("NUMGP_PER_DIM");

  // get integration type
  auto integrationtype = Teuchos::getIntegralValue<Mortar::IntType>(imortar_, "INTTYPE");

  // if we use segment-based integration, the shape of the cells has to be considered!
  Core::FE::CellType intshape;
  if (integrationtype == Mortar::inttype_segments)
  {
    if (ndim_ == 2)
      intshape = Core::FE::CellType::line2;
    else if (ndim_ == 3)
      intshape = Core::FE::CellType::tri3;
    else
      FOUR_C_THROW("wrong dimension!");
  }
  else
    intshape = distype_s;

  //**********************************************************************
  // choose Gauss rule according to (a) element type (b) input parameter
  //**********************************************************************
  switch (intshape)
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
              FOUR_C_THROW("Our experience says that 1 GP per slave element is not enough.");
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
            case 7:
              mygaussrule = Core::FE::GaussRule2D::tri_7point;
              break;
            case 12:
              mygaussrule = Core::FE::GaussRule2D::tri_12point;
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
      else if (integrationtype == Mortar::inttype_elements ||
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
      else
        FOUR_C_THROW("unknown integration type!");

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
      Core::FE::GaussRule2D mygaussrule = Core::FE::GaussRule2D::quad_25point;

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
              FOUR_C_THROW("Our experience says that 1 GP per slave element is not enough.");
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
      FOUR_C_THROW("Mortar::Integrator: This contact element type is not implemented!");
      break;
    }
  }  // switch(eletype)

  return;
}


/*--------------------------------------------------------------------------------------*
 | Integrate without segmentation --> more GP required                       farah 01/13|
 | Integration over the entire Slave-Element: no mapping sxi->eta                       |
 | required                                                                             |
 *--------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void Mortar::IntegratorCalc<distype_s, distype_m>::integrate_ele_based_2d(
    Mortar::Element& sele, std::vector<Mortar::Element*> meles, bool* boundary_ele, MPI_Comm comm)
{
  // check for problem dimension
  if (ndim_ != 2) FOUR_C_THROW("2D integration method called for non-2D problem");

  // number of nodes (slave, master)
  int nrow = sele.num_node();
  int ndof = dynamic_cast<Mortar::Node*>(sele.nodes()[0])->num_dof();
  int nodemaster = meles[0]->num_node();

  // create empty vectors for shape fct. evaluation
  static Core::LinAlg::Matrix<ns_, 1> sval;
  static Core::LinAlg::Matrix<nm_, 1> mval;
  static Core::LinAlg::Matrix<ns_, 1> lmval;

  // get slave element nodes themselves
  Core::Nodes::Node** mynodes = sele.nodes();
  if (!mynodes) FOUR_C_THROW("IntegrateAndDerivSegment: Null pointer!");

  // decide whether boundary modification has to be considered or not
  // this is element-specific (is there a boundary node in this element?)
  bool bound = false;
  for (int k = 0; k < nrow; ++k)
  {
    Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[k]);
    if (!mymrtrnode) FOUR_C_THROW("integrate_deriv_segment_2d: Null pointer!");
    bound += mymrtrnode->is_on_bound();
  }

  // decide whether linear LM are used for quadratic FE here
  bool linlm = false;
  bool dualquad = false;
  if (lmquadtype_ == Mortar::lagmult_lin && sele.shape() == Core::FE::CellType::line3)
  {
    bound = false;  // crosspoints and linear LM NOT at the same time!!!!
    linlm = true;
    if (shapefcn_ == Mortar::shape_dual) dualquad = true;
  }

  double sxia = -1;
  double sxib = 1;
  double sxi[2] = {0.0, 0.0};

  //**********************************************************************
  // loop over all Gauss points for integration
  //**********************************************************************
  for (int gp = 0; gp < n_gp(); ++gp)  // loop to the end //Number of the GP
  {
    bool is_on_mele = false;
    bool kink_projection = false;
    // coordinates and weight of the GP
    double eta[2] = {coordinate(gp, 0), 0.0};
    double wgt = weight(gp);
    sxi[0] = 0.5 * (1 - eta[0]) * sxia + 0.5 * (1 + eta[0]) * sxib;

    // evaluate the two slave side Jacobians
    double dxdsxi = sele.jacobian(sxi);
    double dsxideta = -0.5 * sxia + 0.5 * sxib;

    // evaluate Lagrange multiplier shape functions (on slave element)
    if (lmquadtype_ == Mortar::lagmult_const)
      Utils::evaluate_shape_lm_const(shapefcn_, sxi, lmval, sele, nrow);
    else if (linlm)
      Utils::evaluate_shape_lm_lin(shapefcn_, sxi, lmval, sele, nrow);
    else
      Utils::evaluate_shape_lm(shapefcn_, sxi, lmval, sele, nrow);

    // evaluate trace space shape functions (on both elements)
    Utils::evaluate_shape_displ(sxi, sval, sele, dualquad);

    //********************************************************************
    //  loop over all involved masterelements
    //********************************************************************
    for (int nummaster = 0; nummaster < (int)meles.size(); ++nummaster)
    {
      // get Master element nodes themselves
      Core::Nodes::Node** mnodes = meles[nummaster]->nodes();
      if (!mnodes) FOUR_C_THROW("EleBased_Integration: Null pointer!");

      // project Gauss point onto master element
      double mxi[2] = {0.0, 0.0};
      sxi[0] = eta[0];

      Mortar::Projector::impl(sele, *meles[nummaster])
          ->project_gauss_point_2d(sele, eta, *meles[nummaster], mxi);

      // evaluate trace space shape functions (on both elements)
      Utils::evaluate_shape_displ(mxi, mval, *meles[nummaster], false);

      // check GP projection
      if ((mxi[0] >= -1) && (mxi[0] <= 1) && (kink_projection == false))
      {
        kink_projection = true;
        is_on_mele = true;

        // compute segment D/M matrix ****************************************
        double jac = dsxideta * dxdsxi;
        gp_dm(sele, *meles[nummaster], lmval, sval, mval, jac, wgt, nrow, nodemaster, ndof, bound,
            comm);

      }  // end - if Pojection on MasterElement
    }  // loop-end over all involved master elements

    // strong discontinuity --> Boundary Segmentation
    if (is_on_mele == false)
    {
      *boundary_ele = true;
      std::cout << "WARNING: strong disc. occurs !!!" << std::endl;
    }

  }  // loop-end over all gp
}


/*----------------------------------------------------------------------*
 |  Integrate and linearize a 1D slave / master overlap (2D)  popp 02/09|
 |  This method integrates the overlap D/M matrix and weighted gap g~   |
 |  and stores it in mseg and gseg respectively. Moreover, derivatives  |
 |  LinD/M and Ling are built and stored directly into adjacent nodes.  |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void Mortar::IntegratorCalc<distype_s, distype_m>::integrate_segment_2d(Mortar::Element& sele,
    double& sxia, double& sxib, Mortar::Element& mele, double& mxia, double& mxib, MPI_Comm comm)
{
  // get LMtype
  Mortar::LagMultQuad lmtype = lmquadtype_;

  // explicitly defined shape function type needed
  if (shapefcn_ == Mortar::shape_undefined)
    FOUR_C_THROW("integrate_deriv_segment_2d called without specific shape function defined!");

  // check for problem dimension
  if (ndim_ != 2) FOUR_C_THROW("2D integration method called for non-2D problem");

  // check input data
  if ((!sele.is_source()) || (mele.is_source()))
    FOUR_C_THROW("IntegrateAndDerivSegment called on a wrong type of Mortar::Element pair!");
  if ((sxia < -1.0) || (sxib > 1.0))
    FOUR_C_THROW("IntegrateAndDerivSegment called with infeasible slave limits!");
  if ((mxia < -1.0) || (mxib > 1.0))
    FOUR_C_THROW("IntegrateAndDerivSegment called with infeasible master limits!");

  // number of nodes (slave, master)
  int nrow = sele.num_node();
  int ncol = mele.num_node();
  int ndof = dynamic_cast<Mortar::Node*>(sele.nodes()[0])->num_dof();

  // create empty vectors for shape fct. evaluation
  static Core::LinAlg::Matrix<ns_, 1> sval;
  static Core::LinAlg::Matrix<nm_, 1> mval;
  static Core::LinAlg::Matrix<ns_, 1> lmval;

  // get slave element nodes themselves
  Core::Nodes::Node** mynodes = sele.nodes();
  if (!mynodes) FOUR_C_THROW("IntegrateAndDerivSegment: Null pointer!");

  // decide whether boundary modification has to be considered or not
  // this is element-specific (is there a boundary node in this element?)
  //---------------------------------
  // do trafo for bound elements
  Core::LinAlg::SerialDenseMatrix trafo(nrow, nrow, true);
  bool bound = false;

  // get number of bound nodes
  std::vector<int> ids;
  for (int i = 0; i < nrow; ++i)
  {
    Node* mymrtrnode = dynamic_cast<Node*>(sele.nodes()[i]);
    if (mymrtrnode->is_on_boundor_ce())
    {
      // get local bound id
      ids.push_back(i);
      bound = true;
    }
  }

  int numbound = (int)ids.size();

  // if all bound: error
  if ((nrow - numbound) < 1e-12) return;

  const double factor = 1.0 / (nrow - numbound);
  // row loop
  for (int i = 0; i < nrow; ++i)
  {
    Node* mymrtrnode = dynamic_cast<Node*>(sele.nodes()[i]);
    if (!mymrtrnode->is_on_boundor_ce())
    {
      trafo(i, i) = 1.0;
      for (int j = 0; j < (int)ids.size(); ++j) trafo(i, ids[j]) = factor;
    }
  }

  // decide whether linear LM are used for quadratic FE here
  bool linlm = false;
  bool dualquad = false;
  if (lmtype == Mortar::lagmult_lin && sele.shape() == Core::FE::CellType::line3)
  {
    bound = false;  // crosspoints and linear LM NOT at the same time!!!!
    linlm = true;
    if (shapefcn_ == Mortar::shape_dual) dualquad = true;
  }

  //**********************************************************************
  // loop over all Gauss points for integration
  //**********************************************************************
  for (int gp = 0; gp < n_gp(); ++gp)
  {
    // coordinates and weight
    double eta[2] = {coordinate(gp, 0), 0.0};
    double wgt = weight(gp);

    // coordinate transformation sxi->eta (slave Mortar::Element->Overlap)
    double sxi[2] = {0.0, 0.0};
    sxi[0] = 0.5 * (1 - eta[0]) * sxia + 0.5 * (1 + eta[0]) * sxib;

    // project Gauss point onto master element
    double mxi[2] = {0.0, 0.0};
    Mortar::Projector::impl(sele, mele)->project_gauss_point_2d(sele, sxi, mele, mxi);

    // check GP projection
    if ((mxi[0] < mxia) || (mxi[0] > mxib))
    {
      std::cout << "Slave ID: " << sele.id() << " Master ID: " << mele.id() << std::endl;
      std::cout << "Gauss point: " << sxi[0] << " " << sxi[1] << std::endl;
      std::cout << "Projection: " << mxi[0] << " " << mxi[1] << std::endl;
      FOUR_C_THROW("IntegrateAndDerivSegment: Gauss point projection failed! mxi={}", mxi[0]);
    }

    // evaluate Lagrange multiplier shape functions (on slave element)
    if (lmtype == Mortar::lagmult_const)
      Utils::evaluate_shape_lm_const(shapefcn_, sxi, lmval, sele, nrow);
    else if (linlm)
      Utils::evaluate_shape_lm_lin(shapefcn_, sxi, lmval, sele, nrow);
    else
      Utils::evaluate_shape_lm(shapefcn_, sxi, lmval, sele, nrow);

    // transform shape functions for bound case
    if (bound)
    {
      Core::LinAlg::SerialDenseVector tempval(nrow, true);
      for (int i = 0; i < nrow; ++i)
        for (int j = 0; j < nrow; ++j) tempval(i) += trafo(i, j) * lmval(j);

      for (int i = 0; i < nrow; ++i) lmval(i) = tempval(i);
    }

    // evaluate trace space shape functions (on both elements)
    Utils::evaluate_shape_displ(sxi, sval, sele, dualquad);
    Utils::evaluate_shape_displ(mxi, mval, mele, false);

    // evaluate the two slave side Jacobians
    double dxdsxi = sele.jacobian(sxi);
    double dsxideta = -0.5 * sxia + 0.5 * sxib;

    // compute segment D/M matrix ****************************************
    double jac = dsxideta * dxdsxi;
    gp_dm(sele, mele, lmval, sval, mval, jac, wgt, nrow, ncol, ndof, bound, comm);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Compute entries for D and M matrix at GP                 farah 12/13|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void inline Mortar::IntegratorCalc<distype_s, distype_m>::gp_dm(Mortar::Element& sele,
    Mortar::Element& mele, Core::LinAlg::Matrix<ns_, 1>& lmval, Core::LinAlg::Matrix<ns_, 1>& sval,
    Core::LinAlg::Matrix<nm_, 1>& mval, double& jac, double& wgt, int& nrow, int& ncol, int& ndof,
    bool& bound, MPI_Comm comm)
{
  // get slave element nodes themselves
  Core::Nodes::Node** snodes = sele.nodes();
  if (!snodes) FOUR_C_THROW("IntegrateAndDerivSegment: Null pointer!");
  Core::Nodes::Node** mnodes = mele.nodes();
  if (!mnodes) FOUR_C_THROW("IntegrateAndDerivSegment: Null pointer!");

  // BOUNDARY NODE MODIFICATION **********************************
  // We have modified their neighbors' dual shape functions, so we
  // now have a problem with off-diagonal entries occurring in D.
  // Of course we want to keep the diagonality property of the D
  // matrix, but still we may not modify the whole Mortar coupling
  // setting! We achieve both by applying a quite simple but very
  // effective trick: The boundary nodes have already been defined
  // as being master nodes, so all we have to do here, is to shift
  // the off-diagonal terms from D to the respective place in M,
  // which is not diagonal anyway! (Mind the MINUS sign!!!)
  // *************************************************************

  // compute segment D/M matrix ****************************************
  // dual shape functions without locally linear Lagrange multipliers
  if ((shapefcn_ == Mortar::shape_dual || shapefcn_ == Mortar::shape_petrovgalerkin) &&
      lmquadtype_ != Mortar::lagmult_lin)
  {
    for (int j = 0; j < nrow; ++j)
    {
      Mortar::Node* cnode = dynamic_cast<Mortar::Node*>(snodes[j]);

      if (cnode->owner() != Core::Communication::my_mpi_rank(comm)) continue;
      if (cnode->is_on_boundor_ce()) continue;

      cnode->mo_data().interface_weight() += lmval(j) * jac * wgt;

      // integrate mseg
      for (int k = 0; k < ncol; ++k)
      {
        Mortar::Node* mnode = dynamic_cast<Mortar::Node*>(mnodes[k]);

        // multiply the two shape functions
        double prod = lmval(j) * mval(k) * jac * wgt;
        if (abs(prod) > MORTARINTTOL)
        {
          cnode->add_m_value(mnode->id(), prod);
          if (!bound) cnode->add_d_value(cnode->id(), prod);
        }
      }

      // integrate dseg (boundary modification)
      if (bound)
      {
        bool j_boundnode = cnode->is_on_boundor_ce();

        for (int k = 0; k < nrow; ++k)
        {
          Mortar::Node* mnode = dynamic_cast<Mortar::Node*>(snodes[k]);

          bool k_boundnode = mnode->is_on_boundor_ce();

          // do not assemble off-diagonal terms if j,k are both non-boundary nodes
          if (!j_boundnode && !k_boundnode && (j != k)) continue;

          // multiply the two shape functions
          double prod = lmval(j) * sval(k) * jac * wgt;
          if (abs(prod) > MORTARINTTOL)
          {
            // isolate the dseg entries to be filled
            // (both the main diagonal and every other secondary diagonal)
            // and add current Gauss point's contribution to dseg
            if (mnode->is_on_boundor_ce())
              cnode->add_m_value(mnode->id(), -prod);
            else
              cnode->add_d_value(mnode->id(), prod);
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
      Mortar::Node* cnode = dynamic_cast<Mortar::Node*>(snodes[j]);

      if (cnode->owner() != Core::Communication::my_mpi_rank(comm)) continue;
      if ((shapefcn_ == Mortar::shape_standard && cnode->is_on_boundor_ce()) ||
          ((shapefcn_ == Mortar::shape_dual || shapefcn_ == Mortar::shape_petrovgalerkin) &&
              cnode->is_on_bound()))
        continue;

      cnode->mo_data().interface_weight() += lmval(j) * jac * wgt;

      // integrate mseg
      for (int k = 0; k < ncol; ++k)
      {
        Mortar::Node* mnode = dynamic_cast<Mortar::Node*>(mnodes[k]);

        // multiply the two shape functions
        double prod = lmval(j) * mval(k) * jac * wgt;
        if (abs(prod) > MORTARINTTOL) cnode->add_m_value(mnode->id(), prod);
      }

      // integrate dseg
      for (int k = 0; k < nrow; ++k)
      {
        Mortar::Node* snode = dynamic_cast<Mortar::Node*>(snodes[k]);

        // multiply the two shape functions
        double prod = lmval(j) * sval(k) * jac * wgt;
        if (abs(prod) > MORTARINTTOL)
        {
          if ((shapefcn_ == Mortar::shape_standard && snode->is_on_boundor_ce()) ||
              ((shapefcn_ == Mortar::shape_dual || shapefcn_ == Mortar::shape_petrovgalerkin) &&
                  snode->is_on_bound()))
            cnode->add_m_value(snode->id(), -prod);
          else
            cnode->add_d_value(snode->id(), prod);
        }
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Compute entries for D and M matrix at GP (3D Quad)       farah 12/13|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void inline Mortar::IntegratorCalc<distype_s, distype_m>::gp_3d_dm_quad(Mortar::Element& sele,
    Mortar::Element& mele, Mortar::IntElement& sintele, Core::LinAlg::SerialDenseVector& lmval,
    Core::LinAlg::SerialDenseVector& lmintval, Core::LinAlg::Matrix<ns_, 1>& sval,
    Core::LinAlg::Matrix<nm_, 1>& mval, double& jac, double& wgt, int& nrow, int& nintrow,
    int& ncol, int& ndof, bool& bound)
{
  // get slave element nodes themselves
  Core::Nodes::Node** snodes = sele.nodes();
  if (!snodes) FOUR_C_THROW("Null pointer!");
  Core::Nodes::Node** mnodes = mele.nodes();
  if (!mnodes) FOUR_C_THROW("Null pointer!");
  Core::Nodes::Node** sintnodes = sintele.nodes();
  if (!sintnodes) FOUR_C_THROW("Null pointer for sintnodes!");

  // CASES 1/2: standard LM shape functions and quadratic or linear interpolation
  // CASE 5: dual LM shape functions and linear interpolation
  // (here, we must NOT ignore the small off-diagonal terms for accurate convergence)
  if ((shapefcn_ == Mortar::shape_standard &&
          (lmquadtype_ == Mortar::lagmult_quad || lmquadtype_ == Mortar::lagmult_lin)) ||
      ((shapefcn_ == Mortar::shape_dual || shapefcn_ == Mortar::shape_petrovgalerkin) &&
          (lmquadtype_ == Mortar::lagmult_lin || lmquadtype_ == Mortar::lagmult_const)))
  {
    // compute all mseg and dseg matrix entries
    // loop over Lagrange multiplier dofs j
    for (int j = 0; j < nrow; ++j)
    {
      Mortar::Node* cnode = dynamic_cast<Mortar::Node*>(snodes[j]);

      cnode->mo_data().interface_weight() += lmval[j] * jac * wgt;

      // integrate mseg
      for (int k = 0; k < ncol; ++k)
      {
        Mortar::Node* mnode = dynamic_cast<Mortar::Node*>(mnodes[k]);

        // multiply the two shape functions
        double prod = lmval[j] * mval(k) * jac * wgt;
        if (abs(prod) > MORTARINTTOL) cnode->add_m_value(mnode->id(), prod);
      }

      // integrate dseg
      for (int k = 0; k < nrow; ++k)
      {
        Mortar::Node* snode = dynamic_cast<Mortar::Node*>(snodes[k]);

        // multiply the two shape functions
        double prod = lmval[j] * sval(k) * jac * wgt;
        if (abs(prod) > MORTARINTTOL)
        {
          if ((shapefcn_ == Mortar::shape_standard && snode->is_on_boundor_ce()) ||
              ((shapefcn_ == Mortar::shape_dual || shapefcn_ == Mortar::shape_petrovgalerkin) &&
                  snode->is_on_bound()))
            cnode->add_m_value(snode->id(), -prod);
          else
            cnode->add_d_value(snode->id(), prod);
        }
      }
    }
  }
  // CASE 3: standard LM shape functions and piecewise linear interpolation
  else if (shapefcn_ == Mortar::shape_standard && lmquadtype_ == Mortar::lagmult_pwlin)
  {
    // compute all mseg and dseg matrix entries
    // loop over Lagrange multiplier dofs j
    for (int j = 0; j < nintrow; ++j)
    {
      Mortar::Node* cnode = dynamic_cast<Mortar::Node*>(sintnodes[j]);

      cnode->mo_data().interface_weight() += lmintval[j] * jac * wgt;

      // integrate mseg
      for (int k = 0; k < ncol; ++k)
      {
        Mortar::Node* mnode = dynamic_cast<Mortar::Node*>(mnodes[k]);

        // multiply the two shape functions
        double prod = lmintval[j] * mval(k) * jac * wgt;
        if (abs(prod) > MORTARINTTOL) cnode->add_m_value(mnode->id(), prod);
      }

      // integrate dseg
      for (int k = 0; k < nrow; ++k)
      {
        Mortar::Node* snode = dynamic_cast<Mortar::Node*>(snodes[k]);

        // multiply the two shape functions
        double prod = lmintval[j] * sval(k) * jac * wgt;
        if (abs(prod) > MORTARINTTOL)
        {
          if (snode->is_on_bound())
            cnode->add_m_value(snode->id(), -prod);
          else
            cnode->add_d_value(snode->id(), prod);
        }
      }
    }
  }
  // CASE 4: dual LM shape functions and quadratic interpolation
  else if ((shapefcn_ == Mortar::shape_dual || shapefcn_ == Mortar::shape_petrovgalerkin) &&
           lmquadtype_ == Mortar::lagmult_quad)
  {
    // compute all mseg and dseg matrix entries
    // loop over Lagrange multiplier dofs j
    for (int j = 0; j < nrow; ++j)
    {
      Mortar::Node* cnode = dynamic_cast<Mortar::Node*>(snodes[j]);

      // integrate mseg
      for (int k = 0; k < ncol; ++k)
      {
        Mortar::Node* mnode = dynamic_cast<Mortar::Node*>(mnodes[k]);

        // multiply the two shape functions
        double prod = lmval[j] * mval(k) * jac * wgt;
        if (abs(prod) > MORTARINTTOL)
        {
          cnode->add_d_value(cnode->id(), prod);
          cnode->add_m_value(mnode->id(), prod);
        }
      }
    }
  }
  // INVALID CASES
  else
    FOUR_C_THROW("Invalid integration case for 3D quadratic mortar!");

  return;
}


/*----------------------------------------------------------------------*
 |  Integrate Mmod on slave / master overlap (2D)             popp 01/08|
 |  This method integrates the modification to the Mortar matrix M      |
 |  for curved interface (Paper by Puso/Wohlmuth) from given local      |
 |  coordinates sxia to sxib. The corresponding master side local       |
 |  element coordinates given by mxia and mxib                          |
 |  Output is an LinAlg::SerialDenseMatrix holding the int. values       |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
std::shared_ptr<Core::LinAlg::SerialDenseMatrix>
Mortar::IntegratorCalc<distype_s, distype_m>::integrate_mmod_2d(Mortar::Element& sele, double& sxia,
    double& sxib, Mortar::Element& mele, double& mxia, double& mxib)
{
  //**********************************************************************
  FOUR_C_THROW("IntegrateMmod2D method is outdated!");
  //**********************************************************************

  // check for problem dimension
  if (ndim_ != 2) FOUR_C_THROW("2D integration method called for non-2D problem");

  // check input data
  if ((!sele.is_source()) || (mele.is_source()))
    FOUR_C_THROW("IntegrateMmod2D called on a wrong type of Mortar::Element pair!");
  if ((sxia < -1.0) || (sxib > 1.0))
    FOUR_C_THROW("IntegrateMmod2D called with infeasible slave limits!");
  if ((mxia < -1.0) || (mxib > 1.0))
    FOUR_C_THROW("IntegrateMmod2D called with infeasible master limits!");

  // create empty mmodseg object and wrap it with std::shared_ptr
  int nrow = sele.num_node();
  int nrowdof = ndim_;
  int ncol = mele.num_node();
  int ncoldof = ndim_;
  std::shared_ptr<Core::LinAlg::SerialDenseMatrix> mmodseg =
      std::make_shared<Core::LinAlg::SerialDenseMatrix>(nrow * nrowdof, ncol * ncoldof);

  // create empty vectors for shape fct. evaluation
  Core::LinAlg::SerialDenseVector sval(nrow);
  Core::LinAlg::SerialDenseMatrix sderiv(nrow, 1);
  Core::LinAlg::SerialDenseVector mval(ncol);
  Core::LinAlg::SerialDenseMatrix mderiv(ncol, 1);
  Core::LinAlg::SerialDenseVector lmval(nrow);
  Core::LinAlg::SerialDenseMatrix lmderiv(nrow, 1);

  // loop over all Gauss points for integration
  for (int gp = 0; gp < n_gp(); ++gp)
  {
    double eta[2] = {coordinate(gp, 0), 0.0};
    double wgt = weight(gp);

    double sxi[2] = {0.0, 0.0};
    double mxi[2] = {0.0, 0.0};

    // coordinate transformation sxi->eta (slave Mortar::Element->Overlap)
    sxi[0] = 0.5 * (1 - eta[0]) * sxia + 0.5 * (1 + eta[0]) * sxib;

    // project Gauss point onto master element
    Mortar::Projector::impl(sele, mele)->project_gauss_point_2d(sele, sxi, mele, mxi);

    // check GP projection
    if ((mxi[0] < mxia) || (mxi[0] > mxib))
    {
      std::cout << "Slave ID: " << sele.id() << " Master ID: " << mele.id() << std::endl;
      std::cout << "Gauss point: " << sxi[0] << " " << sxi[1] << std::endl;
      std::cout << "Projection: " << mxi[0] << " " << mxi[1] << std::endl;
      FOUR_C_THROW("IntegrateMmod2D: Gauss point projection failed!");
    }

    // evaluate trace space shape functions (on both elements)
    sele.evaluate_shape(sxi, sval, sderiv, nrow);
    mele.evaluate_shape(mxi, mval, mderiv, ncol);

    // build the delta function of slave side shape functions
    double deltasval = sval[0] - sval[1];

    // evaluate the two slave side Jacobians
    double dxdsxi = sele.jacobian(sxi);
    double dsxideta = -0.5 * sxia + 0.5 * sxib;

    /* loop over all mmodseg matrix entries
     nrow represents the slave Lagrange multipliers !!!
     ncol represents the master dofs !!!
     (this DOES matter here for mmodseg, as it might
     sometimes be rectangular, not quadratic!)              */
    for (int j = 0; j < nrow * nrowdof; ++j)
    {
      for (int k = 0; k < ncol * ncoldof; ++k)
      {
        // multiply the two shape functions
        int mindex = (int)(k / ncoldof);
        double prod = 0.5 * deltasval * mval[mindex];
        // add current Gauss point's contribution to mmodseg
        (*mmodseg)(j, k) += prod * dxdsxi * dsxideta * wgt;
      }
    }
  }  // for (int gp=0;gp<nGP();++gp)

  // prepare computation of purely geometric part of Mmod entries
  Node* snode0 = dynamic_cast<Node*>(sele.nodes()[0]);
  Node* snode1 = dynamic_cast<Node*>(sele.nodes()[1]);

  // normals
  double n[2][2];
  n[0][0] = snode0->mo_data().n()[0];
  n[0][1] = snode0->mo_data().n()[1];
  n[1][0] = snode1->mo_data().n()[0];
  n[1][1] = snode1->mo_data().n()[1];

  // scalar product n1 * n2
  double n1n2 = 0.0;
  for (int i = 0; i < 2; ++i) n1n2 += n[0][i] * n[1][i];

  // vector product n1 x n2
  double n1xn2 = n[0][0] * n[1][1] - n[0][1] * n[1][0];

  // // multiply geometric part onto Mmod
  for (int i = 0; i < ncol; ++i)
  {
    (*mmodseg)(0, 0 + i * ncoldof) *= (1.0 - n1n2);
    (*mmodseg)(1, 0 + i * ncoldof) *= n1xn2;
    (*mmodseg)(0, 1 + i * ncoldof) *= -n1xn2;
    (*mmodseg)(1, 1 + i * ncoldof) *= (1.0 - n1n2);

    (*mmodseg)(2, 0 + i * ncoldof) *= (n1n2 - 1.0);
    (*mmodseg)(3, 0 + i * ncoldof) *= n1xn2;
    (*mmodseg)(2, 1 + i * ncoldof) *= -n1xn2;
    (*mmodseg)(3, 1 + i * ncoldof) *= (n1n2 - 1.0);
  }

  return mmodseg;
}


/*----------------------------------------------------------------------*
 |  Integrate and linearize without segmentation             farah 01/13|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void Mortar::IntegratorCalc<distype_s, distype_m>::integrate_ele_based_3d(
    Mortar::Element& sele, std::vector<Mortar::Element*> meles, bool* boundary_ele, MPI_Comm comm)
{
  // explicitly defined shape function type needed
  if (shapefcn_ == Mortar::shape_undefined)
    FOUR_C_THROW(
        "integrate_deriv_cell_3d_aux_plane called without specific shape function defined!");

  // check for problem dimension
  if (ndim_ != 3) FOUR_C_THROW("3D integration method called for non-3D problem");

  // discretization type of master element
  Core::FE::CellType dt = meles[0]->shape();

  // check input data
  for (int test = 0; test < (int)meles.size(); ++test)
  {
    if ((!sele.is_source()) || (meles[test]->is_source()))
      FOUR_C_THROW("IntegrateDerivCell3D called on a wrong type of Mortar::Element pair!");
  }

  int msize = meles.size();
  int nrow = sele.num_node();
  int nmnode = meles[0]->num_node();
  int ndof = dynamic_cast<Mortar::Node*>(sele.nodes()[0])->num_dof();

  // create empty vectors for shape fct. evaluation
  static Core::LinAlg::Matrix<ns_, 1> sval;
  static Core::LinAlg::Matrix<nm_, 1> mval;
  static Core::LinAlg::Matrix<ns_, 1> lmval;

  //**********************************************************************
  // loop over all Gauss points for integration
  //**********************************************************************
  for (int gp = 0; gp < n_gp(); ++gp)
  {
    // coordinates and weight
    double eta[2] = {coordinate(gp, 0), coordinate(gp, 1)};
    double wgt = weight(gp);

    // note that the third component of sxi is necessary!
    // (although it will always be 0.0 of course)
    // double tempsxi[3] = {0.0, 0.0, 0.0};
    double sxi[2] = {0.0, 0.0};
    double mxi[2] = {0.0, 0.0};
    double projalpha = 0.0;

    sxi[0] = eta[0];
    sxi[1] = eta[1];

    // evaluate the two Jacobians (int. cell and slave element)
    // double jaccell = cell->Jacobian(eta);
    double jacslave = sele.jacobian(sxi);

    // evaluate Lagrange multiplier shape functions (on slave element)
    if (lmquadtype_ == Mortar::lagmult_const)
      Utils::evaluate_shape_lm_const(shapefcn_, sxi, lmval, sele, nrow);
    else
      Utils::evaluate_shape_lm(shapefcn_, sxi, lmval, sele, nrow);

    // evaluate trace space shape functions (on both elements)
    Utils::evaluate_shape_displ(sxi, sval, sele, false);

    // check for Boundary Segmentation
    bool projactable_gp = false;

    //*******************************************************************
    // loop over meles
    //*******************************************************************
    for (int nummaster = 0; nummaster < msize; ++nummaster)
    {
      // project Gauss point onto master element
      bool is_on_mele = Mortar::Projector::impl(sele, *meles[nummaster])
                            ->project_gauss_point_3d(sele, sxi, *meles[nummaster], mxi, projalpha);
      if (not is_on_mele) continue;

      // check GP projection
      double tol = 0.00;
      if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
          dt == Core::FE::CellType::quad9 || dt == Core::FE::CellType::nurbs8 ||
          dt == Core::FE::CellType::nurbs9)
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

      if (is_on_mele == true)
      {
        projactable_gp = true;

        // evaluate trace space shape functions (on both elements)
        Utils::evaluate_shape_displ(mxi, mval, *meles[nummaster], false);

        // compute cell D/M matrix *******************************************
        bool bound = false;
        gp_dm(sele, *meles[nummaster], lmval, sval, mval, jacslave, wgt, nrow, nmnode, ndof, bound,
            comm);
      }  // is_on_mele==true
    }  // loop over meles

    // strong discontinuity --> Boundary Segmentation
    if (projactable_gp == false) *boundary_ele = true;
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Integrate and linearize a 2D slave / master cell (3D)     popp 03/09|
 |  This method integrates the cell M matrix and weighted gap g~        |
 |  and stores it in mseg and gseg respectively. Moreover, derivatives  |
 |  LinM and Ling are built and stored directly into the adjacent nodes.|
 |  (Thus this method combines EVERYTHING before done separately in     |
 |  IntegrateM3D, IntegrateG3D, DerivM3D and DerivG3D!)                 |
 |  This is the auxiliary plane coupling version!!!                     |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void Mortar::IntegratorCalc<distype_s, distype_m>::integrate_cell_3d_aux_plane(
    Mortar::Element& sele, Mortar::Element& mele, std::shared_ptr<Mortar::IntCell> cell,
    double* auxn, MPI_Comm comm)
{
  // explicitly defined shape function type needed
  if (shapefcn_ == Mortar::shape_undefined)
    FOUR_C_THROW(
        "integrate_deriv_cell_3d_aux_plane called without specific shape function defined!");

  // check for problem dimension
  if (ndim_ != 3) FOUR_C_THROW("3D integration method called for non-3D problem");

  // discretization type of master element
  Core::FE::CellType sdt = sele.shape();
  Core::FE::CellType mdt = mele.shape();

  // check input data
  if ((!sele.is_source()) || (mele.is_source()))
    FOUR_C_THROW(
        "integrate_deriv_cell_3d_aux_plane called on a wrong type of Mortar::Element pair!");
  if (cell == nullptr)
    FOUR_C_THROW("integrate_deriv_cell_3d_aux_plane called without integration cell");

  // number of nodes (slave, master)
  int nrow = sele.num_node();
  int ncol = mele.num_node();
  int ndof = dynamic_cast<Mortar::Node*>(sele.nodes()[0])->num_dof();

  // create empty vectors for shape fct. evaluation
  static Core::LinAlg::Matrix<ns_, 1> sval;
  static Core::LinAlg::Matrix<nm_, 1> mval;
  static Core::LinAlg::Matrix<ns_, 1> lmval;

  //---------------------------------
  // do trafo for bound elements
  Core::LinAlg::SerialDenseMatrix trafo(nrow, nrow, true);
  bool bound = false;

  // get number of bound nodes
  std::vector<int> ids;
  for (int i = 0; i < nrow; ++i)
  {
    Node* mymrtrnode = dynamic_cast<Node*>(sele.nodes()[i]);
    if (mymrtrnode->is_on_boundor_ce())
    {
      // get local bound id
      ids.push_back(i);
      bound = true;
    }
  }

  int numbound = (int)ids.size();

  // if all bound: error
  if ((nrow - numbound) < 1e-12) return;

  const double factor = 1.0 / (nrow - numbound);
  // row loop
  for (int i = 0; i < nrow; ++i)
  {
    Node* mymrtrnode = dynamic_cast<Node*>(sele.nodes()[i]);
    if (!mymrtrnode->is_on_boundor_ce())
    {
      trafo(i, i) = 1.0;
      for (int j = 0; j < (int)ids.size(); ++j) trafo(i, ids[j]) = factor;
    }
  }

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

    double sxi[2] = {0.0, 0.0};
    double mxi[2] = {0.0, 0.0};

    // project Gauss point onto slave element
    // project Gauss point onto master element
    double sprojalpha = 0.0;
    double mprojalpha = 0.0;
    Mortar::Projector::impl(sele)->project_gauss_point_auxn_3d(globgp, auxn, sele, sxi, sprojalpha);
    Mortar::Projector::impl(mele)->project_gauss_point_auxn_3d(globgp, auxn, mele, mxi, mprojalpha);

    // check GP projection (SLAVE)
    double tol = 0.01;
    if (sdt == Core::FE::CellType::quad4 || sdt == Core::FE::CellType::quad8 ||
        sdt == Core::FE::CellType::quad9)
    {
      if (sxi[0] < -1.0 - tol || sxi[1] < -1.0 - tol || sxi[0] > 1.0 + tol || sxi[1] > 1.0 + tol)
      {
        std::cout
            << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Gauss point projection outside!";
        std::cout << "Slave ID: " << sele.id() << " Master ID: " << mele.id() << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Slave GP projection: " << sxi[0] << " " << sxi[1] << std::endl;
      }
    }
    else
    {
      if (sxi[0] < -tol || sxi[1] < -tol || sxi[0] > 1.0 + tol || sxi[1] > 1.0 + tol ||
          sxi[0] + sxi[1] > 1.0 + 2 * tol)
      {
        std::cout
            << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Gauss point projection outside!";
        std::cout << "Slave ID: " << sele.id() << " Master ID: " << mele.id() << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Slave GP projection: " << sxi[0] << " " << sxi[1] << std::endl;
      }
    }

    // check GP projection (MASTER)
    if (mdt == Core::FE::CellType::quad4 || mdt == Core::FE::CellType::quad8 ||
        mdt == Core::FE::CellType::quad9)
    {
      if (mxi[0] < -1.0 - tol || mxi[1] < -1.0 - tol || mxi[0] > 1.0 + tol || mxi[1] > 1.0 + tol)
      {
        std::cout
            << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Gauss point projection outside!";
        std::cout << "Slave ID: " << sele.id() << " Master ID: " << mele.id() << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Master GP projection: " << mxi[0] << " " << mxi[1] << std::endl;
      }
    }
    else
    {
      if (mxi[0] < -tol || mxi[1] < -tol || mxi[0] > 1.0 + tol || mxi[1] > 1.0 + tol ||
          mxi[0] + mxi[1] > 1.0 + 2 * tol)
      {
        std::cout
            << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Gauss point projection outside!";
        std::cout << "Slave ID: " << sele.id() << " Master ID: " << mele.id() << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Master GP projection: " << mxi[0] << " " << mxi[1] << std::endl;
      }
    }

    // evaluate Lagrange multiplier shape functions (on slave element)
    Utils::evaluate_shape_lm(shapefcn_, sxi, lmval, sele, nrow);

    // transform shape functions for bound case
    if (bound)
    {
      Core::LinAlg::SerialDenseVector tempval(nrow, true);
      for (int i = 0; i < nrow; ++i)
        for (int j = 0; j < nrow; ++j) tempval(i) += trafo(i, j) * lmval(j);

      for (int i = 0; i < nrow; ++i) lmval(i) = tempval(i);
    }


    // evaluate trace space shape functions (on both elements)
    Utils::evaluate_shape_displ(sxi, sval, sele, false);
    Utils::evaluate_shape_displ(mxi, mval, mele, false);

    // evaluate the integration cell Jacobian
    double jac = cell->jacobian();

    // compute cell D/M matrix *******************************************
    gp_dm(sele, mele, lmval, sval, mval, jac, wgt, nrow, ncol, ndof, bound, comm);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Integrate and linearize a 2D slave / master cell (3D)     popp 03/09|
 |  This method integrates the cell M matrix (and possibly D matrix)    |
 |  and stores it in mseg and dseg respectively.                        |
 |  This is the QUADRATIC auxiliary plane coupling version!!!           |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void Mortar::IntegratorCalc<distype_s, distype_m>::integrate_cell_3d_aux_plane_quad(
    Mortar::Element& sele, Mortar::Element& mele, Mortar::IntElement& sintele,
    Mortar::IntElement& mintele, std::shared_ptr<Mortar::IntCell> cell, double* auxn)
{
  // get LMtype
  Mortar::LagMultQuad lmtype = lmquadtype_;

  // explicitly defined shape function type needed
  if (shapefcn_ == Mortar::shape_undefined)
    FOUR_C_THROW(
        "ERROR: integrate_deriv_cell_3d_aux_plane_quad called without specific shape function "
        "defined!");

  // check for problem dimension
  if (ndim_ != 3) FOUR_C_THROW("3D integration method called for non-3D problem");

  if (cell->shape() != Core::FE::CellType::tri3) FOUR_C_THROW("wrong cell shape!");

  // discretization type of slave and master IntElement
  Core::FE::CellType sdt = sintele.shape();
  Core::FE::CellType mdt = mintele.shape();

  // discretization type of slave and master Element
  Core::FE::CellType psdt = sele.shape();
  Core::FE::CellType pmdt = mele.shape();

  // check input data
  if ((!sele.is_source()) || (mele.is_source()))
    FOUR_C_THROW(
        "ERROR: integrate_deriv_cell_3d_aux_plane_quad called on a wrong type of Mortar::Element "
        "pair!");
  if (cell == nullptr)
    FOUR_C_THROW("integrate_deriv_cell_3d_aux_plane_quad called without integration cell");

  // number of nodes (slave, master)
  int nrow = sele.num_node();
  int ncol = mele.num_node();
  int nintrow = sintele.num_node();
  int ndof = dynamic_cast<Mortar::Node*>(sele.nodes()[0])->num_dof();

  // create empty vectors for shape fct. evaluation
  Core::LinAlg::Matrix<ns_, 1> sval;
  Core::LinAlg::Matrix<nm_, 1> mval;
  Core::LinAlg::SerialDenseVector lmval(nrow);
  Core::LinAlg::SerialDenseMatrix lmderiv(nrow, 2, true);
  Core::LinAlg::SerialDenseVector lmintval(nintrow);
  Core::LinAlg::SerialDenseMatrix lmintderiv(nintrow, 2, true);

  // get slave element nodes themselves
  Core::Nodes::Node** mynodes = sele.nodes();
  if (!mynodes) FOUR_C_THROW("integrate_deriv_cell_3d_aux_plane_quad: Null pointer!");

  // decide whether boundary modification has to be considered or not
  // this is element-specific (is there a boundary node in this element?)
  bool bound = false;
  for (int k = 0; k < nrow; ++k)
  {
    Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[k]);
    if (!mymrtrnode) FOUR_C_THROW("integrate_deriv_segment_2d: Null pointer!");
    bound += mymrtrnode->is_on_boundor_ce();
  }

  // decide whether displacement shape fct. modification has to be considered or not
  // this is the case for dual quadratic and linear Lagrange multipliers on quad9/quad8/tri6
  // elements
  bool dualquad3d = false;
  if ((shapefcn_ == Mortar::shape_dual) &&
      (lmtype == Mortar::lagmult_quad || lmtype == Mortar::lagmult_lin) &&
      (sele.shape() == Core::FE::CellType::quad9 || sele.shape() == Core::FE::CellType::quad8 ||
          sele.shape() == Core::FE::CellType::tri6))
  {
    dualquad3d = true;
  }

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

    double sxi[2] = {0.0, 0.0};
    double mxi[2] = {0.0, 0.0};

    // project Gauss point onto slave integration element
    // project Gauss point onto master integration element
    double sprojalpha = 0.0;
    double mprojalpha = 0.0;
    Mortar::Projector::impl(sintele)->project_gauss_point_auxn_3d(
        globgp, auxn, sintele, sxi, sprojalpha);
    Mortar::Projector::impl(mintele)->project_gauss_point_auxn_3d(
        globgp, auxn, mintele, mxi, mprojalpha);

    // check GP projection (SLAVE)
    double tol = 0.01;
    if (sdt == Core::FE::CellType::quad4 || sdt == Core::FE::CellType::quad8 ||
        sdt == Core::FE::CellType::quad9)
    {
      if (sxi[0] < -1.0 - tol || sxi[1] < -1.0 - tol || sxi[0] > 1.0 + tol || sxi[1] > 1.0 + tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Slave Gauss point "
                     "projection outside!";
        std::cout << "Slave ID: " << sele.id() << " Master ID: " << mele.id() << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Slave GP (IntElement) projection: " << sxi[0] << " " << sxi[1] << std::endl;
      }
    }
    else
    {
      if (sxi[0] < -tol || sxi[1] < -tol || sxi[0] > 1.0 + tol || sxi[1] > 1.0 + tol ||
          sxi[0] + sxi[1] > 1.0 + 2 * tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Slave Gauss point "
                     "projection outside!";
        std::cout << "Slave ID: " << sele.id() << " Master ID: " << mele.id() << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Slave GP (IntElement) projection: " << sxi[0] << " " << sxi[1] << std::endl;
      }
    }

    // check GP projection (MASTER)
    if (mdt == Core::FE::CellType::quad4 || mdt == Core::FE::CellType::quad8 ||
        mdt == Core::FE::CellType::quad9)
    {
      if (mxi[0] < -1.0 - tol || mxi[1] < -1.0 - tol || mxi[0] > 1.0 + tol || mxi[1] > 1.0 + tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Master Gauss point "
                     "projection outside!";
        std::cout << "Slave ID: " << sele.id() << " Master ID: " << mele.id() << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Master GP (IntElement) projection: " << mxi[0] << " " << mxi[1] << std::endl;
      }
    }
    else
    {
      if (mxi[0] < -tol || mxi[1] < -tol || mxi[0] > 1.0 + tol || mxi[1] > 1.0 + tol ||
          mxi[0] + mxi[1] > 1.0 + 2 * tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Master Gauss point "
                     "projection outside!";
        std::cout << "Slave ID: " << sele.id() << " Master ID: " << mele.id() << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Master GP (IntElement) projection: " << mxi[0] << " " << mxi[1] << std::endl;
      }
    }

    // project Gauss point back to slave (parent) element
    // project Gauss point back to master (parent) element
    double psxi[2] = {0.0, 0.0};
    double pmxi[2] = {0.0, 0.0};
    double psprojalpha = 0.0;
    double pmprojalpha = 0.0;
    Mortar::Projector::impl(sele)->project_gauss_point_auxn_3d(
        globgp, auxn, sele, psxi, psprojalpha);
    Mortar::Projector::impl(mele)->project_gauss_point_auxn_3d(
        globgp, auxn, mele, pmxi, pmprojalpha);
    // sintele.MapToParent(sxi, psxi); // old way of doing it via affine map... wrong (popp 05/2016)
    // mintele.MapToParent(mxi, pmxi); // old way of doing it via affine map... wrong (popp 05/2016)

    // check GP projection (SLAVE)
    if (psdt == Core::FE::CellType::quad4 || psdt == Core::FE::CellType::quad8 ||
        psdt == Core::FE::CellType::quad9 || psdt == Core::FE::CellType::nurbs9)
    {
      if (psxi[0] < -1.0 - tol || psxi[1] < -1.0 - tol || psxi[0] > 1.0 + tol ||
          psxi[1] > 1.0 + tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Slave Gauss point "
                     "projection outside!";
        std::cout << "Slave ID: " << sele.id() << " Master ID: " << mele.id() << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Slave GP projection: " << psxi[0] << " " << psxi[1] << std::endl;
      }
    }
    else
    {
      if (psxi[0] < -tol || psxi[1] < -tol || psxi[0] > 1.0 + tol || psxi[1] > 1.0 + tol ||
          psxi[0] + psxi[1] > 1.0 + 2 * tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Slave Gauss point "
                     "projection outside!";
        std::cout << "Slave ID: " << sele.id() << " Master ID: " << mele.id() << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Slave GP projection: " << psxi[0] << " " << psxi[1] << std::endl;
      }
    }

    // check GP projection (MASTER)
    if (pmdt == Core::FE::CellType::quad4 || pmdt == Core::FE::CellType::quad8 ||
        pmdt == Core::FE::CellType::quad9 || pmdt == Core::FE::CellType::nurbs9)
    {
      if (pmxi[0] < -1.0 - tol || pmxi[1] < -1.0 - tol || pmxi[0] > 1.0 + tol ||
          pmxi[1] > 1.0 + tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Master Gauss point "
                     "projection outside!";
        std::cout << "Slave ID: " << sele.id() << " Master ID: " << mele.id() << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Master GP projection: " << pmxi[0] << " " << pmxi[1] << std::endl;
      }
    }
    else
    {
      if (pmxi[0] < -tol || pmxi[1] < -tol || pmxi[0] > 1.0 + tol || pmxi[1] > 1.0 + tol ||
          pmxi[0] + pmxi[1] > 1.0 + 2 * tol)
      {
        std::cout << "\n***Warning: integrate_deriv_cell_3d_aux_plane: Master Gauss point "
                     "projection outside!";
        std::cout << "Slave ID: " << sele.id() << " Master ID: " << mele.id() << std::endl;
        std::cout << "GP local: " << eta[0] << " " << eta[1] << std::endl;
        std::cout << "Master GP projection: " << pmxi[0] << " " << pmxi[1] << std::endl;
      }
    }

    // evaluate Lagrange multiplier shape functions (on slave element)
    //    if (bound)
    //    {
    //      sele.evaluate_shape_lag_mult_lin(shapefcn_, psxi, lmval, lmderiv, nrow);
    //    }
    //    else
    //    {
    //      sele.evaluate_shape_lag_mult(shapefcn_, psxi, lmval, lmderiv, nrow);
    //      sintele.evaluate_shape_lag_mult(shapefcn_, sxi, lmintval, lmintderiv, nintrow);
    //    }

    if (lmtype == Mortar::lagmult_const)
      Utils::evaluate_shape_lm_const(shapefcn_, psxi, lmval, sele, nrow);
    else if (bound)
    {
      sele.evaluate_shape_lag_mult(shapefcn_, psxi, lmval, lmderiv, nrow);
    }
    else
    {
      sele.evaluate_shape_lag_mult(shapefcn_, psxi, lmval, lmderiv, nrow);
      sintele.evaluate_shape_lag_mult(shapefcn_, sxi, lmintval, lmintderiv, nintrow, false);
    }

    // evaluate trace space shape functions (on both elements)
    Utils::evaluate_shape_displ(psxi, sval, sele, dualquad3d);
    Utils::evaluate_shape_displ(pmxi, mval, mele, false);

    // evaluate the integration cell Jacobian
    double jac = cell->jacobian();

    // compute cell D/M matrix *******************************************
    gp_3d_dm_quad(sele, mele, sintele, lmval, lmintval, sval, mval, jac, wgt, nrow, nintrow, ncol,
        ndof, bound);
  }
  //**********************************************************************

  return;
}


// line2 slave
template class Mortar::IntegratorCalc<Core::FE::CellType::line2, Core::FE::CellType::line2>;
template class Mortar::IntegratorCalc<Core::FE::CellType::line2, Core::FE::CellType::line3>;

// line3 slave
template class Mortar::IntegratorCalc<Core::FE::CellType::line3, Core::FE::CellType::line2>;
template class Mortar::IntegratorCalc<Core::FE::CellType::line3, Core::FE::CellType::line3>;

// quad4 slave
template class Mortar::IntegratorCalc<Core::FE::CellType::quad4, Core::FE::CellType::quad4>;
template class Mortar::IntegratorCalc<Core::FE::CellType::quad4, Core::FE::CellType::quad8>;
template class Mortar::IntegratorCalc<Core::FE::CellType::quad4, Core::FE::CellType::quad9>;
template class Mortar::IntegratorCalc<Core::FE::CellType::quad4, Core::FE::CellType::tri3>;
template class Mortar::IntegratorCalc<Core::FE::CellType::quad4, Core::FE::CellType::tri6>;

// quad8 slave
template class Mortar::IntegratorCalc<Core::FE::CellType::quad8, Core::FE::CellType::quad4>;
template class Mortar::IntegratorCalc<Core::FE::CellType::quad8, Core::FE::CellType::quad8>;
template class Mortar::IntegratorCalc<Core::FE::CellType::quad8, Core::FE::CellType::quad9>;
template class Mortar::IntegratorCalc<Core::FE::CellType::quad8, Core::FE::CellType::tri3>;
template class Mortar::IntegratorCalc<Core::FE::CellType::quad8, Core::FE::CellType::tri6>;

// quad9 slave
template class Mortar::IntegratorCalc<Core::FE::CellType::quad9, Core::FE::CellType::quad4>;
template class Mortar::IntegratorCalc<Core::FE::CellType::quad9, Core::FE::CellType::quad8>;
template class Mortar::IntegratorCalc<Core::FE::CellType::quad9, Core::FE::CellType::quad9>;
template class Mortar::IntegratorCalc<Core::FE::CellType::quad9, Core::FE::CellType::tri3>;
template class Mortar::IntegratorCalc<Core::FE::CellType::quad9, Core::FE::CellType::tri6>;

// tri3 slave
template class Mortar::IntegratorCalc<Core::FE::CellType::tri3, Core::FE::CellType::quad4>;
template class Mortar::IntegratorCalc<Core::FE::CellType::tri3, Core::FE::CellType::quad8>;
template class Mortar::IntegratorCalc<Core::FE::CellType::tri3, Core::FE::CellType::quad9>;
template class Mortar::IntegratorCalc<Core::FE::CellType::tri3, Core::FE::CellType::tri3>;
template class Mortar::IntegratorCalc<Core::FE::CellType::tri3, Core::FE::CellType::tri6>;

// tri6 slave
template class Mortar::IntegratorCalc<Core::FE::CellType::tri6, Core::FE::CellType::quad4>;
template class Mortar::IntegratorCalc<Core::FE::CellType::tri6, Core::FE::CellType::quad8>;
template class Mortar::IntegratorCalc<Core::FE::CellType::tri6, Core::FE::CellType::quad9>;
template class Mortar::IntegratorCalc<Core::FE::CellType::tri6, Core::FE::CellType::tri3>;
template class Mortar::IntegratorCalc<Core::FE::CellType::tri6, Core::FE::CellType::tri6>;

//==================================================
//                     NURBS
//==================================================
// nurbs2 slave
template class Mortar::IntegratorCalc<Core::FE::CellType::nurbs2, Core::FE::CellType::nurbs2>;
template class Mortar::IntegratorCalc<Core::FE::CellType::nurbs2, Core::FE::CellType::nurbs3>;

// nurbs3 slave
template class Mortar::IntegratorCalc<Core::FE::CellType::nurbs3, Core::FE::CellType::nurbs2>;
template class Mortar::IntegratorCalc<Core::FE::CellType::nurbs3, Core::FE::CellType::nurbs3>;

FOUR_C_NAMESPACE_CLOSE

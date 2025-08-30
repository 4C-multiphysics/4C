// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cardiovascular0d.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 |  ctor (public)                                              mhv 10/13|
 *----------------------------------------------------------------------*/
Utils::Cardiovascular0D::Cardiovascular0D(std::shared_ptr<Core::FE::Discretization> discr,
    const std::string& conditionname, std::vector<int>& curID)
    : actdisc_(discr),
      cardiovascular0dcond_(0),
      cardiovascular0dstructcoupcond_(0),
      cardiovascular0dtype_(none),
      atrium_model_(Teuchos::getIntegralValue<Inpar::Cardiovascular0D::Cardvasc0DAtriumModel>(
          Global::Problem::instance()->cardiovascular0_d_structural_params().sublist(
              "SYS-PUL CIRCULATION PARAMETERS"),
          "ATRIUM_MODEL")),
      ventricle_model_(Teuchos::getIntegralValue<Inpar::Cardiovascular0D::Cardvasc0DVentricleModel>(
          Global::Problem::instance()->cardiovascular0_d_structural_params().sublist(
              "SYS-PUL CIRCULATION PARAMETERS"),
          "VENTRICLE_MODEL")),
      respiratory_model_(
          Teuchos::getIntegralValue<Inpar::Cardiovascular0D::Cardvasc0DRespiratoryModel>(
              Global::Problem::instance()->cardiovascular0_d_structural_params().sublist(
                  "RESPIRATORY PARAMETERS"),
              "RESPIRATORY_MODEL")),
      gaussrule_(Core::FE::GaussRule2D::undefined)
{
  actdisc_->get_condition(conditionname, cardiovascular0dcond_);
  if (cardiovascular0dcond_.size())
  {
    cardiovascular0dtype_ = get_cardiovascular0_d_type(conditionname);
    std::vector<int> curcoupID;
    for (auto& i : cardiovascular0dcond_)
    {
      curID.push_back(i->parameters().get<int>("id"));
    }

    std::shared_ptr<Core::FE::Discretization> structdis =
        Global::Problem::instance()->get_dis("structure");
    if (structdis == nullptr) FOUR_C_THROW("no structure discretization available");

    // first get all Neumann conditions on structure
    structdis->get_condition("SurfaceNeumannCardiovascular0D", cardiovascular0dstructcoupcond_);

    unsigned int numcoupcond = cardiovascular0dstructcoupcond_.size();
    if (numcoupcond == 0) FOUR_C_THROW("no coupling conditions found");

    std::vector<int> wkID(cardiovascular0dcond_.size());
    for (unsigned int i = 0; i < cardiovascular0dcond_.size(); i++)
    {
      wkID[i] = cardiovascular0dcond_[i]->parameters().get<int>("id");
    }

    // safety checks for closed-loop vascular model
    if (cardiovascular0dtype_ == cardvasc0d_syspulcirculation or
        cardiovascular0dtype_ == cardvascrespir0d_syspulperiphcirculation)
    {
      std::vector<const std::string*> condtype(cardiovascular0dcond_.size());
      for (unsigned int i = 0; i < cardiovascular0dcond_.size(); i++)
      {
        condtype[i] = &cardiovascular0dcond_[i]->parameters().get<std::string>("TYPE");

        if (atrium_model_ == Inpar::Cardiovascular0D::atr_elastance_0d or
            atrium_model_ == Inpar::Cardiovascular0D::atr_prescribed)
        {
          if (*condtype[i] == "atrium_left" or *condtype[i] == "atrium_right")
            FOUR_C_THROW(
                "Set ATRIUM_MODEL to '3D' if you want to couple the 0D vascular system to a 3D "
                "atrial structure!");
        }
      }

      switch (atrium_model_)
      {
        case Inpar::Cardiovascular0D::atr_elastance_0d:
        case Inpar::Cardiovascular0D::atr_prescribed:
        {
          if (ventricle_model_ == Inpar::Cardiovascular0D::ventr_structure_3d)
          {
            if (cardiovascular0dcond_.size() == 2)
            {
              if (*condtype[0] != "ventricle_left" and *condtype[1] != "ventricle_left")
                FOUR_C_THROW("No left/right ventricle type of condition specified!");
              if (*condtype[0] != "ventricle_right" and *condtype[1] != "ventricle_right")
                FOUR_C_THROW("No left/right ventricle type of condition specified!");
            }
            else
              FOUR_C_THROW("You need 2 conditions (left + right ventricle)!");
          }
          if (ventricle_model_ == Inpar::Cardiovascular0D::ventr_elastance_0d or
              ventricle_model_ == Inpar::Cardiovascular0D::ventr_prescribed)
          {
            if (cardiovascular0dcond_.size() == 1)
            {
              if (*condtype[0] != "dummy") FOUR_C_THROW("Only specify 1 dummy condition!");
            }
            else
              FOUR_C_THROW("You're only allowed to specify 1 (dummy) condition!");
          }
        }
        break;
        case Inpar::Cardiovascular0D::atr_structure_3d:
        {
          if (ventricle_model_ == Inpar::Cardiovascular0D::ventr_elastance_0d)
            FOUR_C_THROW("You cannot use 3D atria with 0D ventricles!");

          if (cardiovascular0dcond_.size() == 4)
          {
            if (*condtype[0] != "atrium_left" and *condtype[1] != "atrium_left" and
                *condtype[2] != "atrium_left" and *condtype[3] != "atrium_left")
              FOUR_C_THROW(
                  "ATRIUM_MODEL is set to '3D' but you don't have a left/right atrium type of "
                  "condition specified!");
            if (*condtype[0] != "atrium_right" and *condtype[1] != "atrium_right" and
                *condtype[2] != "atrium_right" and *condtype[3] != "atrium_right")
              FOUR_C_THROW(
                  "ATRIUM_MODEL is set to '3D' but you don't have a left/right atrium type of "
                  "condition specified!");

            if (*condtype[0] != "ventricle_left" and *condtype[1] != "ventricle_left" and
                *condtype[2] != "ventricle_left" and *condtype[3] != "ventricle_left")
              FOUR_C_THROW("No left/right ventricle type of condition specified!");
            if (*condtype[0] != "ventricle_right" and *condtype[1] != "ventricle_right" and
                *condtype[2] != "ventricle_right" and *condtype[3] != "ventricle_right")
              FOUR_C_THROW("No left/right ventricle type of condition specified!");
          }
          else
            FOUR_C_THROW(
                "You need 4 conditions (left + right ventricle, and left + right atrium)!");
        }
        break;
        default:
          FOUR_C_THROW("Unknown ATRIUM_MODEL!");
          break;
      }  // end of switch
    }  // end if (cardiovascular0dtype_ == cardvasc0d_syspulcirculation)

    std::vector<int> coupcondID(cardiovascular0dstructcoupcond_.size());
    // set Neumann line to condition
    for (unsigned int i = 0; i < cardiovascular0dstructcoupcond_.size(); i++)
    {
      coupcondID[i] = cardiovascular0dstructcoupcond_[i]->parameters().get<int>("coupling_id");

      // TODO: This hacks the condition parameters.
      auto& parameters = const_cast<Core::IO::InputParameterContainer&>(
          cardiovascular0dstructcoupcond_[i]->parameters());
      std::string type = "orthopressure";
      parameters.add("TYPE", type);
      std::vector<int> onoff(6, 0);
      onoff[0] = 1;
      parameters.add("ONOFF", onoff);
      std::vector<double> val(6, 0.0);
      parameters.add("VAL", val);
    }

    if (cardiovascular0dcond_.size() != cardiovascular0dstructcoupcond_.size())
      FOUR_C_THROW(
          "Number of cardiovascular 0D conditions has to be equal to number of cardiovascular 0d "
          "structure coupling conditions!");

    if (*std::min_element(wkID.begin(), wkID.end()) != 0)
      FOUR_C_THROW("Start your ID numbering from 0 on!");
    if (*std::min_element(coupcondID.begin(), coupcondID.end()) != 0)
      FOUR_C_THROW("Start your coupling_id numbering from 0 on!");

    if (*std::min_element(wkID.begin(), wkID.end()) !=
        *std::min_element(coupcondID.begin(), coupcondID.end()))
      FOUR_C_THROW(
          "Min cardiovascular0d id not equal to min cardiovascular0d structure coupling id!");
    if (*std::max_element(wkID.begin(), wkID.end()) !=
        *std::max_element(coupcondID.begin(), coupcondID.end()))
      FOUR_C_THROW(
          "Max cardiovascular0d id not equal to max cardiovascular0d structure coupling id!");

    if (*std::max_element(wkID.begin(), wkID.end()) !=
        static_cast<int>(cardiovascular0dcond_.size()) - 1)
      FOUR_C_THROW("Max ID should be the number of conditions minus 1!");
    if (*std::max_element(coupcondID.begin(), coupcondID.end()) !=
        static_cast<int>(cardiovascular0dstructcoupcond_.size()) - 1)
      FOUR_C_THROW("Max coupling_id should be the number of conditions minus 1!");
  }
  else
  {
    cardiovascular0dtype_ = none;
  }
}

/*-----------------------------------------------------------------------*
|(private)                                                      mhv 10/13|
 *-----------------------------------------------------------------------*/
Utils::Cardiovascular0D::Cardiovascular0DType Utils::Cardiovascular0D::get_cardiovascular0_d_type(
    const std::string& name)
{
  if (name == "Cardiovascular0D4ElementWindkesselStructureCond")
    return cardvasc0d_4elementwindkessel;
  else if (name == "Cardiovascular0DArterialProxDistStructureCond")
    return cardvasc0d_arterialproxdist;
  else if (name == "Cardiovascular0DSysPulCirculationStructureCond")
    return cardvasc0d_syspulcirculation;
  else if (name == "CardiovascularRespiratory0DSysPulPeriphCirculationStructureCond")
    return cardvascrespir0d_syspulperiphcirculation;
  return none;
}

/*------------------------------------------------------------------------*
|(public)                                                      mhv 10/13  |
|Initialization routine computes ref base values and activates conditions |
 *------------------------------------------------------------------------*/
void Utils::Cardiovascular0D::initialize(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec1,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec2)
{
  FOUR_C_THROW("Overridden by derived class!");
  return;
}


/*-----------------------------------------------------------------------*
|(public)                                                       mhv 10/13|
|Evaluate Cardiovascular0D functions, choose the right action based on type    |
 *-----------------------------------------------------------------------*/
void Utils::Cardiovascular0D::evaluate(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat1,
    std::shared_ptr<Core::LinAlg::SparseOperator> sysmat2,
    std::shared_ptr<Core::LinAlg::SparseOperator> sysmat3,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec1,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec2,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec3,
    const std::shared_ptr<Core::LinAlg::Vector<double>> sysvec4,
    std::shared_ptr<Core::LinAlg::Vector<double>> sysvec5)
{
  FOUR_C_THROW("Overridden by derived class!");
  return;
}


void Utils::Cardiovascular0D::evaluate_d_struct_dp(
    Teuchos::ParameterList& params, Core::LinAlg::SparseOperator& sysmat)
{
  // get structural time-integrator dependent values
  double sc_strtimint = params.get("scale_timint", 1.0);

  int numdof_per_cond = 1;

  // choose action
  switch (cardiovascular0dtype_)
  {
    case cardvasc0d_4elementwindkessel:
      numdof_per_cond = 3;
      break;
    case cardvasc0d_arterialproxdist:
      numdof_per_cond = 4;
      break;
    case cardvasc0d_syspulcirculation:
      break;
    case cardvascrespir0d_syspulperiphcirculation:
      break;
    case none:
      return;
    default:
      FOUR_C_THROW("Unknown Cardiovascular0D type to be evaluated in Cardiovascular0D class!");
      break;
  }

  const int offsetID = params.get<int>("OffsetID");

  std::vector<int> gindex_syspulcirculation(16);
  gindex_syspulcirculation[0] = offsetID;
  for (int j = 1; j < 16; j++) gindex_syspulcirculation[j] = gindex_syspulcirculation[0] + j;

  std::vector<int> gindex_syspulperiphcirculation(34);
  gindex_syspulperiphcirculation[0] = offsetID;
  for (int j = 1; j < 34; j++)
    gindex_syspulperiphcirculation[j] = gindex_syspulperiphcirculation[0] + j;

  // loop over cardiovascular0d structure coupling conditions
  /* here we do the loop to assemble the offdiagonal stiffness block dfext/dcvdof (0,1 block)
  this is the derivative of the orthopressure Neumann load (external load vector fext) w.r.t. the
  pressure*/
  for (auto* coupcond : cardiovascular0dstructcoupcond_)
  {
    int coupcondID = coupcond->parameters().get<int>("coupling_id");
    params.set("coupling_id", coupcondID);

    std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
        params.get<std::shared_ptr<const Core::LinAlg::Vector<double>>>("new disp");
    actdisc_->set_state("displacement", *disp);

    // global and local ID of this bc in the redundant vectors
    std::vector<int> gindex(numdof_per_cond);
    gindex[0] = numdof_per_cond * coupcondID + offsetID;
    for (int j = 1; j < numdof_per_cond; j++) gindex[j] = gindex[0] + j;

    const std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = coupcond->geometry();
    // if (geom.empty()) FOUR_C_THROW("evaluation of condition with empty geometry");
    // no check for empty geometry here since in parallel computations
    // can exist processors which do not own a portion of the elements belonging
    // to the condition geometry
    for (const auto& [id, ele] : geom)
    {
      // get element location vector and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      ele->location_vector(*actdisc_, lm, lmowner, lmstride);

      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      const int eledim = (int)lm.size();

      Core::LinAlg::SerialDenseVector elevector;
      elevector.size(eledim);

      Core::Elements::Element* element = ele.get();
      int numnode = element->num_node();

      // allocate vector for shape functions and matrix for derivatives
      Core::LinAlg::SerialDenseVector funct(numnode);
      Core::LinAlg::SerialDenseMatrix deriv(2, numnode);
      Core::LinAlg::SerialDenseMatrix xc;

      xc.shape(numnode, 3);

      if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement new'");
      std::shared_ptr<const Core::LinAlg::Vector<double>> curdispl =
          actdisc_->get_state("displacement");
      std::vector<double> mydisp = Core::FE::extract_values(*curdispl, lm);

      for (int j = 0; j < numnode; ++j)
      {
        xc(j, 0) = element->nodes()[j]->x()[0] + mydisp[j * 3 + 0];
        xc(j, 1) = element->nodes()[j]->x()[1] + mydisp[j * 3 + 1];
        xc(j, 2) = element->nodes()[j]->x()[2] + mydisp[j * 3 + 2];
      }

      /*----------------------------------------------------------------------*
      |               start loop over integration points                     |
      *----------------------------------------------------------------------*/
      Core::FE::CellType shape = element->shape();
      // type of gaussian integration
      switch (shape)
      {
        case Core::FE::CellType::tri3:
          gaussrule_ = Core::FE::GaussRule2D::tri_3point;
          break;
        case Core::FE::CellType::tri6:
          gaussrule_ = Core::FE::GaussRule2D::tri_6point;
          break;
        case Core::FE::CellType::quad4:
          gaussrule_ = Core::FE::GaussRule2D::quad_4point;
          break;
        case Core::FE::CellType::quad8:
          gaussrule_ = Core::FE::GaussRule2D::quad_9point;
          break;
        case Core::FE::CellType::quad9:
          gaussrule_ = Core::FE::GaussRule2D::quad_9point;
          break;
        case Core::FE::CellType::nurbs9:
          gaussrule_ = Core::FE::GaussRule2D::quad_9point;
          break;
        default:
          FOUR_C_THROW("shape type unknown!\n");
          break;
      }

      const Core::FE::IntegrationPoints2D intpoints(gaussrule_);
      for (int gp = 0; gp < intpoints.nquad; gp++)
      {
        // set gausspoints from integration rule
        Core::LinAlg::SerialDenseVector e(2);
        e(0) = intpoints.qxg[gp][0];
        e(1) = intpoints.qxg[gp][1];

        // get shape functions and derivatives in the plane of the element

        Core::FE::shape_function_2d(funct, e(0), e(1), shape);
        Core::FE::shape_function_2d_deriv1(deriv, e(0), e(1), shape);

        // stuff to get spatial Neumann
        const int numdim = 3;
        Core::LinAlg::SerialDenseMatrix gp_coord(1, numdim);

        std::vector<double> normal(3);

        // note that the length of this normal is the area dA
        // compute dXYZ / drs
        Core::LinAlg::SerialDenseMatrix dxyzdrs(2, 3);
        Core::LinAlg::multiply(dxyzdrs, deriv, xc);

        normal[0] = dxyzdrs(0, 1) * dxyzdrs(1, 2) - dxyzdrs(0, 2) * dxyzdrs(1, 1);
        normal[1] = dxyzdrs(0, 2) * dxyzdrs(1, 0) - dxyzdrs(0, 0) * dxyzdrs(1, 2);
        normal[2] = dxyzdrs(0, 0) * dxyzdrs(1, 1) - dxyzdrs(0, 1) * dxyzdrs(1, 0);

        const double fac = intpoints.qwgt[gp];
        for (int node = 0; node < numnode; ++node)
          for (int dim = 0; dim < 3; dim++)
            elevector[node * 3 + dim] += funct[node] * normal[dim] * fac;
      }

      int eid = ele->id();

      // assemble the offdiagonal stiffness block (0,1 block) arising from dR_struct/dcvdof
      // assemble to rectangular matrix. The col corresponds to the Cardiovascular0D ID.
      std::vector<int> colvec(1);

      // choose action
      switch (cardiovascular0dtype_)
      {
        case cardvasc0d_4elementwindkessel:
          colvec[0] = gindex[0];
          break;
        case cardvasc0d_arterialproxdist:
          colvec[0] = gindex[0];
          break;
        case cardvasc0d_syspulcirculation:
        {
          for (unsigned int j = 0; j < cardiovascular0dcond_.size(); ++j)
          {
            const Core::Conditions::Condition& cond = *(cardiovascular0dcond_[j]);
            int id_cardvasc0d = cond.parameters().get<int>("id");
            if (coupcondID == id_cardvasc0d)
            {
              // get the type of the corresponding cardiovascular0D condition
              const std::string* conditiontype =
                  &cardiovascular0dcond_[j]->parameters().get<std::string>("TYPE");
              if (*conditiontype == "ventricle_left") colvec[0] = gindex_syspulcirculation[3];
              if (*conditiontype == "ventricle_right") colvec[0] = gindex_syspulcirculation[11];
              if (*conditiontype == "atrium_left") colvec[0] = gindex_syspulcirculation[0];
              if (*conditiontype == "atrium_right") colvec[0] = gindex_syspulcirculation[8];
              if (*conditiontype == "dummy") colvec[0] = gindex_syspulcirculation[0];
            }
          }
        }
        break;
        case cardvascrespir0d_syspulperiphcirculation:
        {
          for (unsigned int j = 0; j < cardiovascular0dcond_.size(); ++j)
          {
            const Core::Conditions::Condition& cond = *(cardiovascular0dcond_[j]);
            int id_cardvasc0d = cond.parameters().get<int>("id");
            if (coupcondID == id_cardvasc0d)
            {
              // get the type of the corresponding cardiovascular0D condition
              const std::string* conditiontype =
                  &cardiovascular0dcond_[j]->parameters().get<std::string>("TYPE");
              if (*conditiontype == "ventricle_left") colvec[0] = gindex_syspulperiphcirculation[3];
              if (*conditiontype == "ventricle_right")
                colvec[0] = gindex_syspulperiphcirculation[27];
              if (*conditiontype == "atrium_left") colvec[0] = gindex_syspulperiphcirculation[0];
              if (*conditiontype == "atrium_right") colvec[0] = gindex_syspulperiphcirculation[24];
              if (*conditiontype == "dummy") colvec[0] = gindex_syspulperiphcirculation[0];
            }
          }
        }
        break;
        case none:
          return;
        default:
          FOUR_C_THROW("Unknown Cardiovascular0D type to be evaluated in Cardiovascular0D class!");
          break;
      }

      elevector.scale(sc_strtimint);
      sysmat.assemble(eid, lmstride, elevector, lm, lmowner, colvec);
    }
  }

  return;
}


/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void Utils::Cardiovascular0D::set_state(const std::string& state, Core::LinAlg::Vector<double>& V)
{
  actdisc_->set_state(state, V);
}

FOUR_C_NAMESPACE_CLOSE

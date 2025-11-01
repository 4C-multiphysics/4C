// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_volumetric_surfaceFlow_condition.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_of_time.hpp"

#include <stdio.h>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                    ismail 09/10|
 *----------------------------------------------------------------------*/
FLD::Utils::FluidVolumetricSurfaceFlowWrapper::FluidVolumetricSurfaceFlowWrapper(
    std::shared_ptr<Core::FE::Discretization> actdis, double dta)
    :  // call constructor for "nontrivial" objects
      discret_(actdis)
{
  //--------------------------------------------------------------------
  // extract the womersley boundary dof
  //--------------------------------------------------------------------

  // Get the surfaces to whom the Womersley flow profile must be applied
  std::vector<const Core::Conditions::Condition*> womersleycond;
  discret_->get_condition("VolumetricSurfaceFlowCond", womersleycond);
  int num_of_wom_conds = womersleycond.size();

  // Get the lines which define the surrounding nodes of the womersley surface
  std::vector<const Core::Conditions::Condition*> womersley_border_nodes_cond;
  discret_->get_condition("VolumetricFlowBorderNodesCond", womersley_border_nodes_cond);
  int num_of_borders = womersley_border_nodes_cond.size();

  //--------------------------------------------------------------------
  // Make sure that both each surface has one and only one border
  //--------------------------------------------------------------------
  if (num_of_wom_conds != num_of_borders)
  {
    FOUR_C_THROW("Each Womersley surface condition must have one and only one border condition");
  }
  // Check if each surface has it's corresponding border
  for (unsigned int i = 0; i < womersleycond.size(); i++)
  {
    bool ConditionIsWrong = true;
    // get the Womersley surface ID
    int surfID = womersleycond[i]->parameters().get<int>("ConditionID");

    // loop over all of the border conditions
    for (unsigned int j = 0; j < womersley_border_nodes_cond.size(); j++)
    {
      // get the border ID
      int lineID = womersley_border_nodes_cond[j]->parameters().get<int>("ConditionID");
      if (lineID == surfID)
      {
        // Since the condition is ok then create the corresponding the condition
        std::shared_ptr<FluidVolumetricSurfaceFlowBc> fvsf_bc =
            std::make_shared<FluidVolumetricSurfaceFlowBc>(discret_, dta,
                "VolumetricSurfaceFlowCond", "VolumetricFlowBorderNodesCond", surfID, i, j);
        bool inserted = fvsf_map_.insert(std::make_pair(surfID, fvsf_bc)).second;
        if (!inserted)
        {
          FOUR_C_THROW(
              "There are more than one Womersley condition lines with the same ID. This can not "
              "yet be handled.");
        }
        ConditionIsWrong = false;
        break;
      }
    }

    // if a surface womersley doesn't have a correspondiong border defined!
    if (ConditionIsWrong)
    {
      FOUR_C_THROW("Each Womersley surface condition must have one and only one border condition");
    }
  }

  return;
}  // end FluidVolumetricSurfaceFlowWrapper


/*----------------------------------------------------------------------*
 |  Output (public)                                         ismail 10/10|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowWrapper::output(Core::IO::DiscretizationWriter& output)
{
  std::map<const int, std::shared_ptr<class FluidVolumetricSurfaceFlowBc>>::iterator mapiter;

  for (mapiter = fvsf_map_.begin(); mapiter != fvsf_map_.end(); mapiter++)
  {
    mapiter->second->FluidVolumetricSurfaceFlowBc::output(
        output, "VolumetricSurfaceFlowCond", mapiter->first);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  read_restart (public)                                    ismail 10/10|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowWrapper::read_restart(
    Core::IO::DiscretizationReader& reader)
{
  std::map<const int, std::shared_ptr<class FluidVolumetricSurfaceFlowBc>>::iterator mapiter;

  for (mapiter = fvsf_map_.begin(); mapiter != fvsf_map_.end(); mapiter++)
  {
    mapiter->second->FluidVolumetricSurfaceFlowBc::read_restart(
        reader, "VolumetricSurfaceFlowCond", mapiter->first);
  }

  return;
}  // FluidVolumetricSurfaceFlowWrapper::read_restart


/*----------------------------------------------------------------------*
 | Evaluate the velocities of the dof and the map          ismail 09/10 |
 | extractor of boundary condition                                      |
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowWrapper::evaluate_velocities(
    Core::LinAlg::Vector<double>& velocities, const double time)
{
  std::map<const int, std::shared_ptr<class FluidVolumetricSurfaceFlowBc>>::iterator mapiter;

  for (mapiter = fvsf_map_.begin(); mapiter != fvsf_map_.end(); mapiter++)
  {
    {
      double flowrate = mapiter->second->FluidVolumetricSurfaceFlowBc::evaluate_flowrate(
          "VolumetricSurfaceFlowCond", time);
      mapiter->second->FluidVolumetricSurfaceFlowBc::evaluate_velocities(
          flowrate, "VolumetricSurfaceFlowCond", time);
    }

    {
      Teuchos::ParameterList eleparams;
      mapiter->second->FluidVolumetricSurfaceFlowBc::correct_flow_rate(
          eleparams, "VolumetricSurfaceFlowCond", FLD::calc_flowrate, time, false);

      mapiter->second->FluidVolumetricSurfaceFlowBc::set_velocities(velocities);
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Constructor (public)                                    ismail 09/10|
 *----------------------------------------------------------------------*/
FLD::Utils::FluidVolumetricSurfaceFlowBc::FluidVolumetricSurfaceFlowBc(
    std::shared_ptr<Core::FE::Discretization> actdis, double dta, std::string ds_condname,
    std::string dl_condname, int condid, int surf_numcond, int line_numcond)
    :  // call constructor for "nontrivial" objects
      discret_(actdis)
{
  // -------------------------------------------------------------------
  // get the processor ID from the communicator
  // -------------------------------------------------------------------
  myrank_ = Core::Communication::my_mpi_rank(discret_->get_comm());

  // -------------------------------------------------------------------
  // get dof row map
  // -------------------------------------------------------------------
  const Core::LinAlg::Map* dofrowmap = actdis->dof_row_map();

  // -------------------------------------------------------------------
  // get condition
  // -------------------------------------------------------------------
  std::vector<const Core::Conditions::Condition*> conditions;
  discret_->get_condition(ds_condname, conditions);
  // -------------------------------------------------------------------
  // some standard initialized variables
  // -------------------------------------------------------------------
  condid_ = condid;
  condnum_s_ = surf_numcond;
  condnum_l_ = line_numcond;
  dta_ = dta;

  // get the cycle period size
  period_ = conditions[surf_numcond]->parameters().get<double>("Period");

  // get the polynomial order of the profile
  order_ = conditions[surf_numcond]->parameters().get<int>("Order");

  // get the number of harmonics
  n_harmonics_ = conditions[surf_numcond]->parameters().get<int>("Harmonics");

  // get the profile type
  flowprofile_type_ = ((conditions[surf_numcond])->parameters().get<std::string>("ConditionType"));

  // get the prebiasing flag
  prebiasing_flag_ = ((conditions[surf_numcond])->parameters().get<std::string>("prebiased"));

  // -------------------------------------------------------------------
  // calculate the center of mass and varage normal of the surface
  // condition
  // -------------------------------------------------------------------
  std::shared_ptr<std::vector<double>> cmass = std::make_shared<std::vector<double>>();
  std::shared_ptr<std::vector<double>> normal = std::make_shared<std::vector<double>>();
  this->center_of_mass_calculation(cmass, normal, ds_condname);

  // get the normal
  normal_ = std::make_shared<std::vector<double>>(*normal);
  std::string normal_info = (conditions[surf_numcond])->parameters().get<std::string>("NORMAL");
  if (normal_info == "SelfEvaluateNormal")
  {
    if (!myrank_)
    {
      std::cout << "Normal is automatically evaluated" << std::endl;
    }
    vnormal_ = std::make_shared<std::vector<double>>(*normal);
  }
  else if (normal_info == "UsePrescribedNormal")
  {
    if (!myrank_)
    {
      std::cout << "Normal is manually setup" << std::endl;
    }
    vnormal_ = std::make_shared<std::vector<double>>();
    (*vnormal_)[0] = conditions[surf_numcond]->parameters().get<double>("n1");
    (*vnormal_)[1] = conditions[surf_numcond]->parameters().get<double>("n2");
    (*vnormal_)[2] = conditions[surf_numcond]->parameters().get<double>("n3");
  }
  else
  {
    FOUR_C_THROW("[{}]: is not a defined normal evaluation type", normal_info);
  }

  // get the center of mass
  std::string c_mass_info =
      (conditions[surf_numcond])->parameters().get<std::string>("CenterOfMass");
  if (c_mass_info == "SelfEvaluateCenterOfMass")
  {
    if (!myrank_)
    {
      std::cout << "Center of mass is automatically evaluated" << std::endl;
    }
    cmass_ = std::make_shared<std::vector<double>>(*cmass);
  }
  else if (c_mass_info == "UsePrescribedCenterOfMass")
  {
    if (!myrank_)
    {
      std::cout << "Center of mass is manually setup" << std::endl;
    }
    normal_ = std::make_shared<std::vector<double>>();
    (*cmass_)[0] = conditions[surf_numcond]->parameters().get<double>("c1");
    (*cmass_)[1] = conditions[surf_numcond]->parameters().get<double>("c2");
    (*cmass_)[2] = conditions[surf_numcond]->parameters().get<double>("c3");
  }
  else
  {
    FOUR_C_THROW("[{}]: is not a defined center-of-mass evaluation type", normal_info);
  }


  // check if the condition surface is a inlet or outlet
  std::string flow_dir = (conditions[surf_numcond])->parameters().get<std::string>("FlowType");
  if (flow_dir == "InFlow")
  {
    flow_dir_ = -1.0;
  }
  else if (flow_dir == "OutFlow")
  {
    flow_dir_ = 1.0;
  }
  else
  {
    FOUR_C_THROW("[{}]: is not a defined flow-direction-type", normal_info);
  }

  // check if the flow is with correction
  std::string corr_flag =
      (conditions[surf_numcond])->parameters().get<std::string>("CorrectionFlag");
  correct_flow_ = (corr_flag == "WithCorrection");

  // -------------------------------------------------------------------
  // create the flow rates vector
  // -------------------------------------------------------------------
  int num_steps = int(period_ / dta) + 1;

  flowrates_ = std::make_shared<std::vector<double>>(num_steps, 0.0);

  if (prebiasing_flag_ == "PREBIASED" || prebiasing_flag_ == "FORCED")
  {
    for (unsigned int i = 0; i < flowrates_->size(); i++)
    {
      (*flowrates_)[i] = this->evaluate_flowrate(ds_condname, dta * double(i));
    }
  }

  // -------------------------------------------------------------------
  // get the node row maps of the condition node
  // -------------------------------------------------------------------
  // evaluate the surface node row map
  this->build_condition_node_row_map(
      discret_, ds_condname, condid_, condnum_s_, cond_surfnoderowmap_);

  // evaluate the line node row map
  const std::string border_cond_name = dl_condname;
  this->build_condition_node_row_map(
      discret_, ds_condname, condid_, condnum_l_, cond_linenoderowmap_);

  // -------------------------------------------------------------------
  // get the dof row maps of the condition node
  // -------------------------------------------------------------------
  // evaluate the surface dof row map
  this->build_condition_dof_row_map(discret_, ds_condname, condid_, condnum_s_, cond_dofrowmap_);
  const Core::LinAlg::Map* drt_dofrowMap = discret_->dof_row_map();

  // -------------------------------------------------------------------
  // calculate the normalized  of mass of the surface condition
  // -------------------------------------------------------------------
  this->eval_local_normalized_radii(ds_condname, dl_condname);

  // -------------------------------------------------------------------
  // create cond_velocities and condition traction velocity terms
  // -------------------------------------------------------------------
  cond_velocities_ = Core::LinAlg::create_vector(*cond_dofrowmap_, true);
  cond_traction_vel_ = Core::LinAlg::create_vector(*dofrowmap, true);
  drt_velocities_ = Core::LinAlg::create_vector(*drt_dofrowMap, true);

  // -------------------------------------------------------------------
  // Evaluate the area of the design surface.
  // This will also return the viscosity and density of the fluid
  // -------------------------------------------------------------------
  area_ = this->area(density_, viscosity_, ds_condname, condid_);
}

/*----------------------------------------------------------------------*
 |  Calculate center of mass (public)                       ismail 10/10|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::center_of_mass_calculation(
    std::shared_ptr<std::vector<double>> coords, std::shared_ptr<std::vector<double>> normal,
    std::string ds_condname)
{
  // Evaluate center of mass
  *coords = std::vector<double>(3, 0.0);
  *normal = std::vector<double>(3, 0.0);

  // fill the list od element evaluation
  Teuchos::ParameterList eleparams;
  eleparams.set<FLD::BoundaryAction>("action", FLD::center_of_mass_calc);
  eleparams.set<double>("total area", 0.0);
  eleparams.set<std::shared_ptr<std::vector<double>>>("center of mass", coords);
  eleparams.set<std::shared_ptr<std::vector<double>>>("normal", normal);

  const std::string condstring(ds_condname);

  discret_->evaluate_condition(eleparams, nullptr, condstring, condid_);

  // get center of mass in parallel case
  std::vector<double> par_coord(3, 0.0);
  for (unsigned int i = 0; i < par_coord.size(); i++)
  {
    // get the actual area on the local processor
    double act_area = eleparams.get<double>("total area");

    // get the actual coordinate values on the local processor
    double act_val = (*coords)[i] * act_area;
    double act_n_val = (*normal)[i] * act_area;

    // define the parallel values that will be summed over all of the processors
    double par_area = 0.0;
    double par_val = 0.0;
    double par_n_val = 0.0;

    // Summ all of the local processor values in one values
    par_val = Core::Communication::sum_all(act_val, discret_->get_comm());
    par_n_val = Core::Communication::sum_all(act_n_val, discret_->get_comm());
    par_area = Core::Communication::sum_all(act_area, discret_->get_comm());

    // finally evaluate the actual center of mass and average normal
    (*coords)[i] = par_val / par_area;
    (*normal)[i] = par_n_val / par_area;
  }


  // Print out the results
  if (myrank_ == 0)
  {
    // print the center of mass
    printf("center of mass of cond(%d) is evaluated:\n", condid_);
    for (unsigned int i = 0; i < coords->size(); i++)
    {
      printf("| %f |", (*coords)[i]);
    }
    printf("\n");

    // print the average normal
    printf("average normal of cond(%d) surface is:\n", condid_);
    for (unsigned int i = 0; i < coords->size(); i++)
    {
      printf("| %f |", (*normal)[i]);
    }
    printf("\n");
  }
}

/*----------------------------------------------------------------------*
 |  Calculate local normalized radii (public)               ismail 10/10|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::eval_local_normalized_radii(
    std::string ds_condname, std::string dl_condname)
// std::shared_ptr<std::vector<double> > center_of_mass,
// std::shared_ptr<std::vector<double> > avg_normal,
// int condid_)
{
  //--------------------------------------------------------------------
  // get the processor rank
  //--------------------------------------------------------------------
  int myrank = Core::Communication::my_mpi_rank(discret_->get_comm());

  local_radii_ = Core::LinAlg::create_vector(*cond_surfnoderowmap_, true);

  border_radii_ = Core::LinAlg::create_vector(*cond_surfnoderowmap_, true);
  //--------------------------------------------------------------------
  // get all of the border nodes
  //--------------------------------------------------------------------
  std::vector<const Core::Conditions::Condition*> conditions;
  discret_->get_condition(dl_condname, conditions);

  const Core::Conditions::Condition* cond = conditions[condnum_l_];

  //--------------------------------------------------------------------
  // get the nodes of the condition
  //--------------------------------------------------------------------
  const std::vector<int>* border_nodes = cond->get_nodes();

  //--------------------------------------------------------------------
  // Create a map of the border nodes
  //--------------------------------------------------------------------
  std::map<int, std::vector<double>> border_nodes_coords;
  for (unsigned int i = 0; i < border_nodes->size(); i++)
  {
    std::vector<double> coords(3, 0.0);
    int node_num = (*border_nodes)[i];

    bool inserted = border_nodes_coords.insert(std::make_pair(node_num, coords)).second;
    if (!inserted)
    {
      FOUR_C_THROW("There are more than one node of the same number. something went wrong");
    }
  }

  //--------------------------------------------------------------------
  // find the coordinates of each border node and get it parallel
  //--------------------------------------------------------------------
  for (std::map<int, std::vector<double>>::iterator it = border_nodes_coords.begin();
      it != border_nodes_coords.end(); it++)
  {
    // define the coordinate of a border node
    std::vector<double> xyze(3, 0.0);

    // if a node doesn't exist on the proc, then use the (0,0,0) origin
    for (unsigned int i = 0; i < xyze.size(); i++)
    {
      xyze[i] = 0.0;
    }

    // get the coordinate of a node, if it exists on the current proc
    if (discret_->have_global_node(it->first))
    {
      // check if the node is not a gohst node
      if (discret_->g_node(it->first)->owner() == myrank)
      {
        // get the coordinate of a node
        const auto& x = discret_->g_node(it->first)->x();
        for (unsigned int i = 0; i < xyze.size(); i++)
        {
          xyze[i] = x[i];
        }
      }
    }

    //------------------------------------------------------------------
    // distribute the coordinates to all of the processors
    //------------------------------------------------------------------
    for (unsigned int i = 0; i < xyze.size(); i++)
    {
      // define the actual value
      double act_val = xyze[i];

      // define the parallel value
      double par_val = 0.0;

      // Summ all of the local processor values in one values
      par_val = Core::Communication::sum_all(act_val, discret_->get_comm());

      // set the coordinate of the nodes on the local processor
      (it->second)[i] = par_val;
    }
  }

  // -------------------------------------------------------------------
  // loop over each node and compare its distance to the
  // [CenterOfMass BorderNodes)
  // -------------------------------------------------------------------

  // get the dimension of the node
  const int dim = 3;

  // define the vector between center-of-mass and current-node
  Core::LinAlg::Matrix<(dim), 1> c_cnd(Core::LinAlg::Initialization::zero);

  // define the vector between center-of-mass and border-node
  Core::LinAlg::Matrix<(dim), 1> c_bnd(Core::LinAlg::Initialization::zero);

  // define the vector that is the nearest from right
  Core::LinAlg::Matrix<(dim), 1> v_right(Core::LinAlg::Initialization::zero);

  // define the vector that is the nearest from left
  Core::LinAlg::Matrix<(dim), 1> v_left(Core::LinAlg::Initialization::zero);


  // define a direction vector perpendicular to the vector
  // of center-of-mass and current-node and to the surface normal.
  // This vector is also used to define whether a certain vector
  // is in the [0,pi] or [pi,2pi] awy from the
  // "center-of-mass and current-node" vector.
  Core::LinAlg::Matrix<(dim), 1> dir_vec(Core::LinAlg::Initialization::zero);

  for (int lid = 0; lid < cond_surfnoderowmap_->num_my_elements(); lid++)
  {
    // get the global id of the current node
    int gid = cond_surfnoderowmap_->gid(lid);

    double R_r = 0.0;
    double R_l = 0.0;
    double R = 0.0;
    // check if the node exists on the current processor
    if (discret_->have_global_node(gid))
    {
      // check if the node is not a gohst node
      if (discret_->g_node(gid)->owner() == myrank)
      {
        double border_radius = 0.0;
        const auto& curr_xyze = discret_->g_node(gid)->x();

        //----------------------------------------------------------------
        // loop over all of the border nodes
        //----------------------------------------------------------------
        double diff_error_l = 2.0;
        double diff_error_r = 2.0;

        bool isBorderNode = false;
        for (std::map<int, std::vector<double>>::iterator it = border_nodes_coords.begin();
            it != border_nodes_coords.end(); it++)
        {
          isBorderNode = false;

          const std::vector<double> bord_xyze = it->second;

          //--------------------------------------------------------------
          // build the cener-of-mass to current-node vector
          // build the cener-of-mass to border-node vector
          //--------------------------------------------------------------
          for (int index = 0; index < dim; index++)
          {
            c_cnd(index) = curr_xyze[index] - (*cmass_)[index];
            c_bnd(index) = bord_xyze[index] - (*cmass_)[index];
          }

          // calculate the radius of the border node
          R = c_cnd.norm2();
          border_radius = c_bnd.norm2();

          if (it->first == gid)
          {
            isBorderNode = true;
            break;
          }

          // normalize the two vectors
          c_cnd.scale(1.0 / c_cnd.norm2());
          c_bnd.scale(1.0 / c_bnd.norm2());

          //--------------------------------------------------------------
          // find the closest two vectors by calculating the norm of the
          // difference between the two vectors
          //--------------------------------------------------------------
          Core::LinAlg::Matrix<(dim), 1> diff = c_cnd;
          diff -= c_bnd;

          //--------------------------------------------------------------
          // evaluate the direction vector = normal_vec X ref_vec
          //--------------------------------------------------------------
          dir_vec(0) = c_cnd(1) * (*normal_)[2] - c_cnd(2) * (*normal_)[1];
          dir_vec(1) = c_cnd(2) * (*normal_)[0] - c_cnd(0) * (*normal_)[2];
          dir_vec(2) = c_cnd(0) * (*normal_)[1] - c_cnd(1) * (*normal_)[0];

          // if the boundary is from the left
          if (dir_vec.dot(c_bnd) > 0)
          {
            if (diff.norm2() <= diff_error_l)
            {
              diff_error_l = diff.norm2();
              R_l = border_radius;
              v_left = c_bnd;
            }
          }
          else
          {
            if (diff.norm2() <= diff_error_r)
            {
              diff_error_r = diff.norm2();
              R_r = border_radius;
              v_right = c_bnd;
            }
          }
        }
        // ---------------------------------------------------------------
        // calculate the local radius of the node
        //  1- Calculate the angle between R_right and [center  curr-node]
        //  2- Calculate the angle between R_left and R_right
        //  3- Calculate the local Radius by interpolation
        //  P.S: intersection method might be more accurate. But since
        //       some nodes might be slightly out of the plane, such a
        //       method might fail.
        // ---------------------------------------------------------------

        if (!isBorderNode)
        {
          double cos_rl = v_right.dot(v_left);
          double cos_r = v_right.dot(c_cnd);

          double angle_rl = 0.0;
          double angle_r = 0.0;
          double border_radius = 0.0;

          if (cos_r >= 1.0 || cos_r <= -1.0)
          {
            border_radius = R_r;
          }
          else if (cos_rl >= 1.0 || cos_rl <= -1.0)
          {
            border_radius = R_l;
          }
          else
          {
            angle_rl = acos(v_right.dot(v_left));
            angle_r = acos(v_right.dot(c_cnd));
            border_radius = R_r + (R_l - R_r) * angle_r / angle_rl;
          }

          // update local radius
          R /= border_radius;
        }
        else
        {
          R = 1.0;
        }
        // update local radius
        local_radii_->replace_global_values(1, &R, &gid);

        // update border radius
        border_radii_->replace_global_values(1, &border_radius, &gid);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Calculate local normalized radii (public)               ismail 10/10|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::build_condition_node_row_map(
    std::shared_ptr<Core::FE::Discretization> dis, const std::string condname, int condid,
    int condnum, std::shared_ptr<Core::LinAlg::Map>& cond_noderowmap)
{
  //--------------------------------------------------------------------
  // get the processor rank
  //--------------------------------------------------------------------
  int myrank = Core::Communication::my_mpi_rank(dis->get_comm());

  //--------------------------------------------------------------------
  // read in the condition
  //--------------------------------------------------------------------
  std::vector<const Core::Conditions::Condition*> conditions;
  discret_->get_condition(condname, conditions);

  const Core::Conditions::Condition* cond = conditions[condnum];

  //--------------------------------------------------------------------
  // get the nodes of the condition
  //--------------------------------------------------------------------
  const std::vector<int>* nodes = cond->get_nodes();
  std::vector<int> nodeids;

  //--------------------------------------------------------------------
  // check which nodes belong to the current proc
  //--------------------------------------------------------------------
  for (unsigned int i = 0; i < nodes->size(); i++)
  {
    int Id = (*nodes)[i];
    if (dis->have_global_node(Id))
    {
      // check if the node is not a gohst node
      if (dis->g_node(Id)->owner() == myrank)
      {
        nodeids.push_back(Id);
      }
    }
  }

  //--------------------------------------------------------------------
  // create the node row map of the nodes on the current proc
  //--------------------------------------------------------------------
  cond_noderowmap =
      std::make_shared<Core::LinAlg::Map>(-1, nodeids.size(), nodeids.data(), 0, dis->get_comm());

}  // build_condition_node_row_map

/*----------------------------------------------------------------------*
 |  Build condition dof_row_map (public)                      ismail 10/10|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::build_condition_dof_row_map(
    std::shared_ptr<Core::FE::Discretization> dis, const std::string condname, int condid,
    int condnum, std::shared_ptr<Core::LinAlg::Map>& cond_dofrowmap)
{
  //--------------------------------------------------------------------
  // get the processor rank
  //--------------------------------------------------------------------
  int myrank = Core::Communication::my_mpi_rank(dis->get_comm());

  //--------------------------------------------------------------------
  // read in the condition
  //--------------------------------------------------------------------
  std::vector<const Core::Conditions::Condition*> conditions;
  discret_->get_condition(condname, conditions);

  const Core::Conditions::Condition* cond = conditions[condnum];

  //--------------------------------------------------------------------
  // get the nodes of the condition
  //--------------------------------------------------------------------
  const std::vector<int>* nodes = cond->get_nodes();
  std::vector<int> dofids;

  //--------------------------------------------------------------------
  // check which nodes belong to the current proc
  //--------------------------------------------------------------------
  for (unsigned int i = 0; i < nodes->size(); i++)
  {
    int Id = (*nodes)[i];
    if (dis->have_global_node(Id))
    {
      // check if the node is not a gohst node
      Core::Nodes::Node* node = dis->g_node(Id);
      if (node->owner() == myrank)
      {
        for (int ldof = 0; ldof < 3; ldof++)
        {
          dofids.push_back(dis->dof(node, ldof));
        }
      }
    }
  }

  //--------------------------------------------------------------------
  // create the node row map of the nodes on the current proc
  //--------------------------------------------------------------------
  cond_dofrowmap =
      std::make_shared<Core::LinAlg::Map>(-1, dofids.size(), dofids.data(), 0, dis->get_comm());

}  // FluidVolumetricSurfaceFlowBc::build_condition_dof_row_map


/*----------------------------------------------------------------------*
 |  Output (public)                                         ismail 10/10|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::output(
    Core::IO::DiscretizationWriter& output, std::string ds_condname, int condnum)
{
  // condnum contains the number of the present condition
  // condition Id numbers must not change at restart!!!!

  std::stringstream stream1, stream2, stream3, stream4;

  // write the flowrates of the previous period
  stream1 << ds_condname << "_flowrates" << condnum;
  output.write_redundant_double_vector(stream1.str(), *flowrates_);

  // write the time step
  stream2 << ds_condname << "_dt" << condnum;
  output.write_double(stream2.str(), dta_);

  // write the condition velocities
  stream3 << ds_condname << "_velocities" << condnum;
  output.write_vector(stream3.str(), cond_velocities_);

  stream4 << ds_condname << "_traction_vel_component" << condnum;
  output.write_vector(stream4.str(), cond_traction_vel_);
  return;
}

/*----------------------------------------------------------------------*
 |  read_restart (public)                                    ismail 11/10|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::read_restart(
    Core::IO::DiscretizationReader& reader, std::string ds_condname, int condnum)
{
  // condnum contains the number of the present condition
  // condition Id numbers must not change at restart!!!!
  std::stringstream stream1, stream2, stream3, stream4;

  // old time step size
  stream2 << ds_condname << "_dt" << condnum;
  double odta = reader.read_double(stream2.str());

  // write the condition velocities
  stream3 << ds_condname << "_velocities" << condnum;
  reader.read_vector(cond_velocities_, stream3.str());

  stream4 << ds_condname << "_traction_vel_component" << condnum;
  reader.read_vector(cond_traction_vel_, stream4.str());

  // get time step of the current problems
  double ndta = dta_;

  // -------------------------------------------------------------------
  // Read in the flowrates values and the flowrates position
  // -------------------------------------------------------------------
  stream1 << ds_condname << "_flowrates" << condnum;

  // read in flowrates
  reader.read_redundant_double_vector(flowrates_, stream1.str());

  // read in the flowrates' position
  flowratespos_ = 0;

  // evaluate the new flowrate vector and position if time stepping
  // is changed
  if (odta != ndta)
  {
    // Get old flowrates Vector size
    int oQSize = (int)flowrates_->size();

    // Calculate new flowrates Vector size
    int nQSize = (int)(double(oQSize) * odta / ndta);

    // evaluate the new flowrates vector
    int nq_pos = 0;
    std::shared_ptr<std::vector<double>> nq = std::make_shared<std::vector<double>>(nQSize, 0.0);
    this->interpolate(*flowrates_, *nq, flowratespos_, nq_pos, period_);

    // store new values in class
    flowratespos_ = nq_pos;
    flowrates_ = nq;
  }
}

/*----------------------------------------------------------------------*
 |  Evaluates the Velocities (public)                       ismail 10/10|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::evaluate_velocities(
    const double flowrate, const std::string ds_condname, const double time)
{
  const double time_in_a_period = fmod(time, period_);  // time - period_*floor(time/period_);
  // get the flowrate position
  const int position = int(time_in_a_period / dta_ + 0.5);

  // insert flowrate into the flowrates vector
  (*flowrates_)[position] = flowrate;

  if (time_in_a_period < dta_)
  {
    (*flowrates_)[0] = flowrate;
    (*flowrates_)[flowrates_->size() - 1] = flowrate;
  }

  std::shared_ptr<Teuchos::ParameterList> params = std::make_shared<Teuchos::ParameterList>();

  params->set<int>("Number of Harmonics", n_harmonics_);
  // condition id
  params->set<int>("Condition ID", condid_);
  // history of flowrates at the outlet
  params->set<std::shared_ptr<std::vector<double>>>("Flowrates", flowrates_);
  // the velocity position
  params->set<int>("Velocity Position", position);
  // the flow type
  params->set<std::string>("flowrate type", flowprofile_type_);
  // time
  params->set<double>("time", time_in_a_period);
  // period of a cycle
  params->set<double>("period", period_);
  // polynomial order
  params->set<int>("polynomial order", order_);
  // surface area
  params->set<double>("area", area_);


  this->velocities(*discret_, *cond_velocities_, *cond_surfnoderowmap_, *local_radii_,
      *border_radii_, *vnormal_, *params);

}  // FLD::Utils::FluidWomersleyBc::EvaluateVelocities

/*----------------------------------------------------------------------*
 |  Evaluates the Velocity components of the traction        ismail 05/11|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::reset_traction_velocity_comp()
{
  cond_traction_vel_->put_scalar(0.0);
}


/*----------------------------------------------------------------------*
 |  Evaluates the Velocity components of the traction        ismail 05/11|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::evaluate_traction_velocity_comp(
    Teuchos::ParameterList eleparams, std::string condname, double flowrate, int condid_,
    double time, double theta, double dta)
{
  //  double norm2= 0.0;

  eleparams.set("thsl", theta * dta);
  eleparams.set("condition velocities", cond_velocities_);
  eleparams.set("condition dofrowmap", cond_dofrowmap_);


  // Export and set state
  export_and_set_boundary_values(*cond_velocities_, *drt_velocities_, "velaf");

  eleparams.set("flowrate", flowrate);
  eleparams.set("area", area_);

  eleparams.set<FLD::BoundaryAction>("action", FLD::traction_velocity_component);
  eleparams.set("velocities", cond_velocities_);
  discret_->evaluate_condition(eleparams, cond_traction_vel_, condname, condid_);
}


/*----------------------------------------------------------------------*
 |  Evaluates the Flowrate  (public)                        ismail 04/11|
 *----------------------------------------------------------------------*/
double FLD::Utils::FluidVolumetricSurfaceFlowBc::evaluate_flowrate(
    const std::string ds_condname, const double time)
{
  // -------------------------------------------------------------------
  // get curve information
  // -------------------------------------------------------------------

  // get condition
  std::vector<const Core::Conditions::Condition*> conditions;
  discret_->get_condition(ds_condname, conditions);
  const Core::Conditions::Condition* condition = conditions[condnum_s_];

  // get curve and curve_factor
  const int functnum = condition->parameters().get<int>("Funct");
  const double val = condition->parameters().get<double>("Val");

  //  if ( val < 1e-14 )
  //    FOUR_C_THROW("Val must be positive!");

  // evaluate the current flowrate value
  double functfac = 0.0;
  double flowrate = 0.0;

  if (functnum > 0)
  {
    functfac =
        Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfTime>(functnum).evaluate(
            time);
    flowrate = val * functfac;
  }

  return flowrate;
}

/*----------------------------------------------------------------------*
 |  Evaluates the Velocities (public)                       ismail 10/10|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::velocities(Core::FE::Discretization& disc,
    Core::LinAlg::Vector<double>& bcdof, Core::LinAlg::Map& cond_noderowmap,
    Core::LinAlg::Vector<double>& local_radii, Core::LinAlg::Vector<double>& border_radii,
    std::vector<double>& normal, Teuchos::ParameterList& params)


{
  //--------------------------------------------------------------------
  // get the processor rank
  //--------------------------------------------------------------------
  int myrank = Core::Communication::my_mpi_rank(disc.get_comm());


  // -------------------------------------------------------------------
  // Read in all of the parameters
  // -------------------------------------------------------------------
  // number of harmonics
  int n_harmonics = params.get<int>("Number of Harmonics");

  // condition id
  int condid = params.get<int>("Condition ID");
  // history of average velocities at the outlet
  std::shared_ptr<std::vector<double>> flowrates =
      params.get<std::shared_ptr<std::vector<double>>>("Flowrates");
  std::shared_ptr<std::vector<double>> velocities =
      std::make_shared<std::vector<double>>(*flowrates);

  // the velocity position
  int velocityposition = params.get<int>("Velocity Position");

  // the flow type
  std::string flowType = params.get<std::string>("flowrate type");
  // time
  double time = params.get<double>("time");
  // period of a cycle
  double period = params.get<double>("period");
  // polynomial order
  int order = params.get<int>("polynomial order");

  // surface area
  double area = params.get<double>("area");

  // -------------------------------------------------------------------
  // Get the volumetric flowrates Fourier coefficients
  // -------------------------------------------------------------------
  std::shared_ptr<std::vector<std::complex<double>>> Vn;
  std::vector<double> Bn;

  //
  int cycle_num = 0;
  double in_cycle_time = 0.0;
  double time2 = time - dta_;
  cycle_num = int(floor(time2 / period));
  in_cycle_time = time2 - period * double(cycle_num);
  flowratespos_ = int((in_cycle_time / dta_));

  for (unsigned int i = 0; i < velocities->size(); i++)
  {
    (*velocities)[i] /= area * flow_dir_;
  }

  if (flowType == "WOMERSLEY")
  {
    if (n_harmonics < 1)
    {
      FOUR_C_THROW("The number of Womersley harmonics is {} (less than 1)", n_harmonics);
    }

    this->dft(velocities, Vn, flowratespos_);

    Bn = std::vector<double>(n_harmonics, 0.0);

    double rl = real((*Vn)[0]);
    double im = imag((*Vn)[0]);

    Bn[0] = 0.5 * rl;

    for (unsigned int k = 1; k < Bn.size(); k++)
    {
      rl = real((*Vn)[k]);
      im = imag((*Vn)[k]);
      const double Mk = sqrt(rl * rl + im * im);
      const double Phik = atan2(-imag((*Vn)[k]), real((*Vn)[k]));

      Bn[k] = Mk * cos(2.0 * std::numbers::pi * double(k) - Phik);
    }
  }

  // -------------------------------------------------------------------
  // evaluate the average velocity and apply it to the design surface
  // -------------------------------------------------------------------
  // loop over all of the nodes
  for (int lid = 0; lid < cond_noderowmap.num_my_elements(); lid++)
  {
    int gid = cond_noderowmap.gid(lid);

    // check if the node exists on the current processor
    if (disc.have_global_node(gid))
    {
      const Core::Nodes::Node* node = disc.g_node(gid);

      // check if the node is not a gohst node
      if (node->owner() == myrank)
      {
        // loop over the dof of a map
        // eval the velocity of a dof
        double velocity = 0.0;
        double r = local_radii.local_values_as_span()[cond_noderowmap.lid(gid)];

        //------------------------------------------------------------
        // Check for the velocity profile type
        //------------------------------------------------------------

        // check for the polynomial type
        if (flowType == "POLYNOMIAL")
        {
          if (order != 0)
          {
            velocity = (1.0 + 2.0 / double(order)) * (*velocities)[velocityposition];
            velocity *= this->polynomail_velocity(r, order);
          }
          else
          {
            velocity = (*velocities)[velocityposition];
          }
        }
        // else check for Womersley type
        else if (flowType == "WOMERSLEY")
        {
          double R = border_radii.local_values_as_span()[cond_noderowmap.lid(gid)];

          // first calculate the parabolic profile of the 0th
          // harmonics
          if (order != 0)
          {
            velocity = (1.0 + 2.0 / double(order)) * Bn[0] * this->polynomail_velocity(r, order);
          }
          else
          {
            velocity = Bn[0];
          }
          // velocity = 0.0;
          double velocity_wom = 0.0;
          double rl = real((*Vn)[0]);
          double im = imag((*Vn)[0]);
          for (unsigned int k = 1; k < Bn.size(); k++)
          {
            //    double Mk   = sqrt(norm((*Qn)[k]));
            rl = real((*Vn)[k]);
            im = imag((*Vn)[k]);
            const double Mk = sqrt(rl * rl + im * im);
            const double Phik = atan2(-imag((*Vn)[k]), real((*Vn)[k]));
            Bn[k] = Mk * cos(2.0 * std::numbers::pi * double(k) - Phik);

            velocity_wom += this->womersley_velocity(r, R, Mk, Phik, k, period);
          }
          velocity += velocity_wom;
        }
        else
        {
          FOUR_C_THROW(
              "[{}] in cond ({}): No such profile is defined. Please correct the input file ",
              flowType.c_str(), condid);
        }
        velocity *= flow_dir_;
        for (unsigned int ldof = 0; ldof < normal.size(); ldof++)
        {
          // get the global dof from using the local one
          int gdof = disc.dof(node, ldof);

          //------------------------------------------------------------
          // Apply the velocity in the normal direction
          //------------------------------------------------------------
          double Vdof = velocity * (normal)[ldof];

          bcdof.replace_global_values(1, &Vdof, &gdof);
        }
      }
    }
  }

}  // FLD::Utils::FluidVolumetricSurfaceFlowBc::Velocities


/*----------------------------------------------------------------------*
 |  Corrects the Flow Rate   (public)                       ismail 10/10|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::correct_flow_rate(
    const Teuchos::ParameterList eleparams, const std::string ds_condname,
    const FLD::BoundaryAction action, const double time, const bool force_correction)
{
  if (!force_correction)
  {
    if (!correct_flow_)
    {
      return;
    }
  }

  // -------------------------------------------------------------------
  // get the processor ID from the communicator
  // -------------------------------------------------------------------
  int myrank = Core::Communication::my_mpi_rank(discret_->get_comm());

  // -------------------------------------------------------------------
  // calculate flow rate
  // -------------------------------------------------------------------
  // Export and set state
  export_and_set_boundary_values(*cond_velocities_, *drt_velocities_, "velaf");

  double actflowrate = this->flow_rate_calculation(eleparams, time, ds_condname, action, condid_);

  std::shared_ptr<Teuchos::ParameterList> params = std::make_shared<Teuchos::ParameterList>();

  // -------------------------------------------------------------------
  // evaluate the wanted flow rate
  // -------------------------------------------------------------------

  // get curve information
  double time_in_a_period = fmod(time, period_);  // time - period_*floor(time/period_);

  // get the flowrate position
  int position = int(time_in_a_period / dta_ + 0.5);

  // insert flowrate into the flowrates vector
  double flowrate = (*flowrates_)[position];
  //  double flowrate = this->flow_rate_calculation(time,ds_condname,action,condid_);

  if (myrank == 0)
  {
    double flow_error = 0.0;
    flow_error = actflowrate - flowrate;
    printf("DIR(%f): Flow_estimated = %f : Flow_wanted = %f : Flow_correction = %f\n", flow_dir_,
        actflowrate, flowrate, flow_error);
  }

  // loop over all of the nodes
  std::shared_ptr<Core::LinAlg::Vector<double>> correction_velnp =
      Core::LinAlg::create_vector(*cond_dofrowmap_, true);

  params->set<int>("Number of Harmonics", 0);
  // condition id
  params->set<int>("Condition ID", condid_);
  // history of average velocities at the outlet
  std::shared_ptr<std::vector<double>> flowrates = std::make_shared<std::vector<double>>();
  flowrates->push_back(1.0 * area_);
  params->set<std::shared_ptr<std::vector<double>>>("Flowrates", flowrates);
  // the velocity position
  params->set<int>("Velocity Position", 0);
  // the flow type
  params->set<std::string>("flowrate type", "POLYNOMIAL");
  // time
  params->set<double>("time", time_in_a_period);
  // period of a cycle
  params->set<double>("period", period_);
  // polynomial order
  params->set<int>("polynomial order", order_);
  // surface area
  params->set<double>("area", area_);

  this->velocities(*discret_, *correction_velnp, *cond_surfnoderowmap_, *local_radii_,
      *border_radii_, *vnormal_, *params);

  // Export and set state
  export_and_set_boundary_values(*correction_velnp, *drt_velocities_, "velaf");

  double corrective_flowrate =
      this->flow_rate_calculation(eleparams, time, ds_condname, action, condid_);

  if (myrank_ == 0)
  {
    std::cout << "+------- corrective_flowrate -------+" << std::endl;
    std::cout << "|      Q_corr: " << corrective_flowrate << std::endl;
    std::cout << "+-----------------------------------+" << std::endl;
  }

  if (action == FLD::calc_flowrate)
  {
    double correction_factor = (flowrate - actflowrate) / (corrective_flowrate);

    double correction = 0.0;
    for (int lid = 0; lid < correction_velnp->local_length(); lid++)
    {
      int gid = correction_velnp->get_map().gid(lid);
      correction = correction_factor * correction_velnp->local_values_as_span()[lid];

      int bc_lid = cond_velocities_->get_map().lid(gid);
      (*cond_velocities_).get_values()[bc_lid] += correction;
    }
  }
  else
  {
    double correction_factor = flowrate / (corrective_flowrate);
    correction_factor = sqrt(correction_factor);
    double correction = 0.0;
    for (int lid = 0; lid < correction_velnp->local_length(); lid++)
    {
      int gid = correction_velnp->get_map().gid(lid);
      correction = correction_factor * correction_velnp->local_values_as_span()[lid];

      int bc_lid = cond_velocities_->get_map().lid(gid);
      (*cond_velocities_).get_values()[bc_lid] = correction;
    }
  }
}


/*----------------------------------------------------------------------*
 |  Apply velocities         (public)                       ismail 10/10|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::set_velocities(
    Core::LinAlg::Vector<double>& velocities)
{
  for (int lid = 0; lid < cond_velocities_->local_length(); lid++)
  {
    int gid = cond_velocities_->get_map().gid(lid);
    double val = cond_velocities_->local_values_as_span()[lid];

    velocities.replace_global_values(1, &val, &gid);
  }
}

/*----------------------------------------------------------------------*
 |  Flow rate calculation                                      ac 03/08 |
 *----------------------------------------------------------------------*/
/*!
  modified by chfoe 04/08

  Calculate the flow rate across an impedance boundary surface

  Flow rates are
  (1) calculated for single element surfaces
  (2) added up over the elements of the single procs
  (3) communicated and added over the procs
  (4) and finally stored within the vector 'flowrates_'

  The vector of the flowrates holds the flow rate history of the
  very last cycle!

*/
double FLD::Utils::FluidVolumetricSurfaceFlowBc::flow_rate_calculation(
    Teuchos::ParameterList eleparams, double time, std::string ds_condname,
    FLD::BoundaryAction action, int condid)
{
  // fill in parameter list for subsequent element evaluation
  // there's no assembly required here
  eleparams.set<FLD::BoundaryAction>("action", action);
  eleparams.set("total time", time);

  // get a vector layout from the discretization to construct matching
  // vectors and matrices local <-> global dof numbering
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // create vector (+ initialization with zeros)
  std::shared_ptr<Core::LinAlg::Vector<double>> flowrates =
      Core::LinAlg::create_vector(*dofrowmap, true);

  const std::string condstring(ds_condname);

  discret_->evaluate_condition(eleparams, flowrates, condstring, condid);

  double local_flowrate = 0.0;
  for (int i = 0; i < dofrowmap->num_my_elements(); i++)
  {
    local_flowrate += (flowrates->local_values_as_span()[i]);
  }

  double flowrate = 0.0;
  flowrate = Core::Communication::sum_all(local_flowrate, dofrowmap->get_comm());

  return flowrate;

}  // FluidImplicitTimeInt::flow_rate_calculation


double FLD::Utils::FluidVolumetricSurfaceFlowBc::pressure_calculation(
    double time, std::string ds_condname, std::string action, int condid)
{
  // fill in parameter list for subsequent element evaluation
  // there's no assembly required here
  Teuchos::ParameterList eleparams;
  double pressure = 0.0;
  eleparams.set("action", action);
  eleparams.set("Inlet integrated pressure", pressure);
  eleparams.set("total time", time);

  // get a vector layout from the discretization to construct matching
  // vectors and matrices local <-> global dof numbering
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // create vector (+ initialization with zeros)
  std::shared_ptr<Core::LinAlg::Vector<double>> flowrates =
      Core::LinAlg::create_vector(*dofrowmap, true);

  const std::string condstring(ds_condname);
  discret_->evaluate_condition(eleparams, flowrates, condstring, condid);

  pressure = eleparams.get<double>("Inlet integrated pressure");
  std::cout << "avg pressure: " << pressure / area_ << std::endl;

  return pressure / area_;

}  // FluidImplicitTimeInt::pressure_calculation

/*----------------------------------------------------------------------*
 |  Parabolic velocity at certain radius and time         mueller 04/10 |
 *----------------------------------------------------------------------*/
double FLD::Utils::FluidVolumetricSurfaceFlowBc::polynomail_velocity(double r, int order)
{
  return (1.0 - pow(r, double(order)));
}  // FLD::Utils::FluidVolumetricSurfaceFlowBc::PolynomailVelocity


/*----------------------------------------------------------------------*
 |  Womersley velocity at certain radius and time         mueller 04/10 |
 *----------------------------------------------------------------------*/
double FLD::Utils::FluidVolumetricSurfaceFlowBc::womersley_velocity(double r, double R, double Bn,
    double Phi,
    // complex<double> Bn,
    int n, double time)
{
  //--------------------------------------------------------------------
  // define some variables
  //--------------------------------------------------------------------

  // velocity
  std::complex<double> velocity;

  // imaginary unit
  std::complex<double> i = std::complex<double>(0.0, 1.0);

  // exp^( i*w*t)
  double constexp = 2.0 * std::numbers::pi * double(n) * time / period_ - Phi;
  double realpart = cos(constexp);
  double imagpart = sin(constexp);
  std::complex<double> eiwt_phi(realpart, imagpart);

  // Jo_z
  std::complex<double> Jo_z;

  // J1_z
  std::complex<double> J1_z;

  // Jo_rz
  std::complex<double> Jo_rz;

  //--------------------------------------------------------------------
  // evaluate the nth harmonic velocity
  //--------------------------------------------------------------------

  // Womersley number
  //  double          alpha = R*sqrt(2.0*M_PI*double(n)/period_/viscosity_);
  double alpha = R * sqrt(2.0 * std::numbers::pi * double(n) / period_ / (viscosity_ / density_));

  // Bessel variable
  std::complex<double> z = alpha * pow(i, 1.5);

  // bessel functions
  Jo_z = this->bessel_j01(z, false);
  J1_z = this->bessel_j01(z, true);
  Jo_rz = this->bessel_j01(z * r, false);

  // velocity
  velocity = (Bn) * (z * (Jo_z - Jo_rz) / (z * Jo_z - 2.0 * J1_z)) * eiwt_phi;

  // return the real part of the Womersley velocity
  return real(velocity);

}  // FLD::Utils::FluidVolumetricSurfaceFlowBc::PolynomailVelocity


/*----------------------------------------------------------------------*
 |  Womersley: Bessel functions of order 0 and 1          mueller 04/10 |
 *----------------------------------------------------------------------*/
std::complex<double> FLD::Utils::FluidVolumetricSurfaceFlowBc::bessel_j01(
    std::complex<double> z, bool order)
{
  // DESCRIPTION:
  // Bessel functions of order 0 (order==false) or 1 (order==true) are calculated for
  // a given argument z
  std::complex<double> J(0.0, 0.0);
  std::complex<double> Jm(0.0, 0.0);

  // Convergence tolerance of the Bessel function
  double tol = 1e-10;
  double error = 10.0 * tol;
  const int maxItr = 200;
  // Bessel function of the first kind and order 1
  if (order == false)
  {
    // J0[N] = sum_{m=0}^{N} {Sn}
    // Sn = ((-(0.5*z)^2))^(m))/(m!*m!)
    // Error = | J0[N]-J0[N-1] |
    int m = 0;
    J = std::complex<double>(0.0, 0.0);
    if (z == std::complex<double>(0.0, 0.0))
      J = std::complex<double>(1.0, 0.0);
    else
    {
      do
      {
        std::complex<double> Sn = std::complex<double>(1.0, 0.0);
        // -------------------------------------------------------------
        // calculate Sn
        // Warning: do not use pow function. The total denominator in
        // the Sn function can become larger than 1e300 (i.e NaN)
        // -------------------------------------------------------------
        for (int i = 1; i <= m; i++)
        {
          Sn *= -(z * z / 4.0) / (double(m + 1 - i) * double(m + 1 - i));
        }
        J += Sn;
        // -------------------------------------------------------------
        // Evaluate the convergence error
        // -------------------------------------------------------------
        if (m < 1)
        {
          error = 10.0 * tol;
        }
        else
        {
          error = abs(J - Jm);
        }
        Jm = J;
        m += 1;
      } while (error > tol and m < maxItr);
    }
  }
  // Bessel function of the first kind and order 1
  else
  {
    // -----------------------------------------------------------------
    // J1[N] = sum_{m=0}^{N} {Sn}
    // Sn = ((-1.0)^m)*(0.5*z)^(2*m+1))/(m!*(m+1)!)
    // Error = | J1[N]-J1[N-1] |
    // -----------------------------------------------------------------
    int m = 0;
    J = std::complex<double>(0.0, 0.0);
    if (z == std::complex<double>(0.0, 0.0))
      J = std::complex<double>(0.0, 0.0);
    else
    {
      do
      {
        std::complex<double> Sn = std::complex<double>(1.0, 0.0);
        // -------------------------------------------------------------
        // calculate Sn
        // Warning: do not use pow function. The total denominator in
        // the Sn function can become larger than 1e300 (i.e NaN)
        // -------------------------------------------------------------
        Sn = (z / 2.0) / (double(m + 1));
        for (int i = 1; i <= m; i++)
        {
          Sn *= -(z * z / 4.0) / (double(m + 1 - i) * double(m + 1 - i));
        }
        J += Sn;
        // -------------------------------------------------------------
        // Evaluate the convergence error
        // -------------------------------------------------------------
        if (m < 1)
        {
          error = 10.0 * tol;
        }
        else
        {
          error = abs(J - Jm);
        }
        Jm = J;
        m += 1;
      } while (error > tol and m < maxItr);
    }
  }
  return J;
}

/*----------------------------------------------------------------------*
 | Area calculation                                         chfoe 05/08 |
 *----------------------------------------------------------------------*/
/*!

*/
double FLD::Utils::FluidVolumetricSurfaceFlowBc::area(
    double& density, double& viscosity, std::string ds_condname, int condid)
{
  // fill in parameter list for subsequent element evaluation
  // there's no assembly required here
  Teuchos::ParameterList eleparams;
  eleparams.set<FLD::BoundaryAction>("action", FLD::calc_area);
  eleparams.set<double>("area", 0.0);
  eleparams.set<double>("viscosity", 0.0);
  eleparams.set<double>("density", 0.0);

  const std::string condstring(ds_condname);

  discret_->evaluate_condition(eleparams, condstring, condid);

  double actarea = eleparams.get<double>("area");
  density = eleparams.get<double>("density");
  viscosity = eleparams.get<double>("viscosity");

  // find the lowest proc number that knows the material data
  int numproc = Core::Communication::num_mpi_ranks(discret_->get_comm());
  int theproc = -1;  // the lowest proc that has the desired information
  std::vector<double> alldens(numproc);

  Core::Communication::gather_all(&density, alldens.data(), 1, discret_->get_comm());
  for (int i = 0; i < numproc; i++)
    if (alldens[i] > 0.0)
    {
      theproc = i;
      break;
    }
  if (theproc < 0) FOUR_C_THROW("Something parallel went terribly wrong!");

  // do the actual communication of density ...
  Core::Communication::broadcast(&density, 1, theproc, discret_->get_comm());
  // ... and viscosity
  Core::Communication::broadcast(&viscosity, 1, theproc, discret_->get_comm());

  // get total area in parallel case
  double pararea = 0.0;
  pararea = Core::Communication::sum_all(actarea, discret_->get_comm());

  if (myrank_ == 0)
  {
    std::cout << "Volumetric surface flow rate condition Id: " << condid << " area = " << pararea
              << std::endl;
  }
  return pararea;
}  // FLD::Utils::FluidVolumetricSurfaceFlowBc::Area

/*----------------------------------------------------------------------*
 |  Womersley: Discrete Fourier Transformation              ismail 10/10 |
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::dft(std::shared_ptr<std::vector<double>> f,
    std::shared_ptr<std::vector<std::complex<double>>>& F, int starting_pos)
{
  //--------------------------------------------------------------------
  // Initialise the Fourier values
  //--------------------------------------------------------------------
  F = std::make_shared<std::vector<std::complex<double>>>(f->size(), 0.0);

  const double N = double(f->size());
  const int fsize = f->size();

  //--------------------------------------------------------------------
  // Compute the Fourier values
  //--------------------------------------------------------------------
  for (int k = 0; k < fsize; k++)
  {
    (*F)[k] = std::complex<double>(0.0, 0.0);
    for (int n = 0; n < fsize; n++)
    {
      int pos = 0;
      if (starting_pos - n >= 0)
      {
        pos = (starting_pos - n);
      }
      else
      {
        pos = fsize - (n - starting_pos);
      }

      double rl = (*f)[pos] * 2.0 / N *
                  (cos(2.0 * std::numbers::pi * double(k) * double(fsize - 1 - n) / N));
      double im = (*f)[pos] * 2.0 / N *
                  (-sin(2.0 * std::numbers::pi * double(k) * double(fsize - 1 - n) / N));

      (*F)[k] += std::complex<double>(rl, im);
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Interpolate values of Vector1 to fit in Vector2         ismail 04/10|
 |                                                                      |
 | V ^                                                                  |
 |   |                              +                                   |
 |   |                            , .                                   |
 |   |                          ,   .                                   |
 |   |                        ,     .                                   |
 |   |                      ,+      .                                   |
 |   |                    ,  .      .                                   |
 |   |                  //   .      .                                   |
 |   |                ,      .      .                                   |
 |   |              ,+       .      .                                   |
 |   |            ,  .       .      .                                   |
 |   |          ,    .       .      .                                   |
 |   |        +      .       .      .                                   |
 |   |        .      .       .      .                                   |
 |   |        .      .       .      .                                   |
 |   |        .      .       .      .                                   |
 |   |        .      .       .      .                                   |
 |   |        .      .       .      .                                   |
 |   |        .      .       .      .                                   |
 |   |        .      .       .      .                                   |
 |   +--------+------o---//--o------+----------->                       |
 |            T(i)   .              T(i+1)      t                       |
 |            .      .              .                                   |
 |            t(m)   t(m+j)         t(m+k)                              |
 |                                                                      |
 |                                                                      |
 |  (T) is the time step of the original vector                         |
 |  (t) is the time step of the new vector                              |
 |  1 - Loop over all intervals (T(i) and T(i+1))                       |
 |  2 - Check if V2 has any time steps between T(i) and T(i+1)          |
 |  3 - Using linear interpolation check get the value of V2 at t(m+j)  |
 |                                                                      |
 | *The advantage of this method is that is works for finer and coarser |
 |  interpolations.                                                     |
 |                                                                      |
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::interpolate(
    std::vector<double>& V1, std::vector<double>& V2, int index1, int& index2, double period)
{
  // Get size of V1 and V2
  int n1 = V1.size();
  int n2 = V2.size();

  double TotalTime = period;

  // Get time step size of V1 and V2
  double dt1 = (TotalTime) / double(n1 - 1);
  double dt2 = (TotalTime) / double(n2 - 1);

  // defining some necessary variables
  double t1_1, t1_2;
  double v1_1, v1_2;

  // define t (time step of V2) and k (index of V2)
  double t = 0.0;
  int k = 0;
  for (int i = 0; i < n1 - 1; i++)
  {
    // -----------------------------------------------------------------
    // Get V1 values at T(i) and T(i+1)
    // -----------------------------------------------------------------
    v1_1 = (V1)[i];
    v1_2 = (V1)[i + 1];

    // -----------------------------------------------------------------
    // Calculate T(i) and T(i+1)
    // -----------------------------------------------------------------
    t1_1 = double(i) * dt1;
    t1_2 = double(i + 1) * dt1;

    // -----------------------------------------------------------------
    // Evaluate V2 values between T(i) and  T(i+1)
    // -----------------------------------------------------------------
    while (t < t1_2)
    {
      // Evaluate value of V2 using Interpolation
      (V2)[k] = (t1_2 - t) / dt1 * v1_1 + (t - t1_1) / dt1 * v1_2;
      // Increment k
      k++;
      // Increment t
      t += dt2;
    }
  }

  // -------------------------------------------------------------------
  // Finally resolve the last step where V2(n2) = V1(n1)
  // -------------------------------------------------------------------
  (V2)[V2.size() - 1] = (V1)[V1.size() - 1];


  // -------------------------------------------------------------------
  // Get the index of V2
  // where t = t     => dt1*index1 = dt2*index2
  //                              dt1
  //                 => index2 = ----- index1
  //                              dt2
  // -------------------------------------------------------------------
  index2 = int(double(index1) * (dt1 / dt2));

}  // FLD::Utils::FluidVolumetricSurfaceFlowBc::interpolate

void FLD::Utils::FluidVolumetricSurfaceFlowBc::update_residual(
    Core::LinAlg::Vector<double>& residual)
{
  residual.update(1.0, *cond_traction_vel_, 1.0);
}

/*----------------------------------------------------------------------*
 |  Constructor (public)                                    ismail 04/11|
 *----------------------------------------------------------------------*/
FLD::Utils::TotalTractionCorrector::TotalTractionCorrector(
    std::shared_ptr<Core::FE::Discretization> actdis, double dta)
    :  // call constructor for "nontrivial" objects
      discret_(actdis)
{
  //--------------------------------------------------------------------
  // extract the womersley boundary dof
  //--------------------------------------------------------------------

  // Get the surfaces to whom the traction flow profile must be applied
  std::vector<const Core::Conditions::Condition*> tractioncond;
  discret_->get_condition("TotalTractionCorrectionCond", tractioncond);
  int num_of_tr_conds = tractioncond.size();

  // Get the lines which define the surrounding nodes of the traction surface
  std::vector<const Core::Conditions::Condition*> traction_border_nodes_cond;
  discret_->get_condition("TotalTractionCorrectionBorderNodesCond", traction_border_nodes_cond);
  int num_of_borders = traction_border_nodes_cond.size();

  //--------------------------------------------------------------------
  // Make sure that both each surface has one and only one border
  //--------------------------------------------------------------------
  if (num_of_tr_conds != num_of_borders)
  {
    FOUR_C_THROW("Each Womersley surface condition must have one and only one border condition");
  }
  // Check if each surface has it's corresponding border
  for (unsigned int i = 0; i < tractioncond.size(); i++)
  {
    bool ConditionIsWrong = true;
    // get the traction surface ID
    int surfID = tractioncond[i]->parameters().get<int>("ConditionID");

    // loop over all of the border conditions
    for (unsigned int j = 0; j < traction_border_nodes_cond.size(); j++)
    {
      // get the border ID
      int lineID = traction_border_nodes_cond[j]->parameters().get<int>("ConditionID");
      if (lineID == surfID)
      {
        // Since the condition is ok then create the corresponding the condition
        std::shared_ptr<FluidVolumetricSurfaceFlowBc> fvsf_bc =
            std::make_shared<FluidVolumetricSurfaceFlowBc>(discret_, dta,
                "TotalTractionCorrectionCond", "TotalTractionCorrectionBorderNodesCond", surfID, i,
                j);
        bool inserted = fvsf_map_.insert(std::make_pair(surfID, fvsf_bc)).second;
        if (!inserted)
        {
          FOUR_C_THROW(
              "There are more than one impedance condition lines with the same ID. This can not "
              "yet be handled.");
        }

        ConditionIsWrong = false;
        break;
      }
    }

    // if a surface traction doesn't have a correspondiong border defined!
    if (ConditionIsWrong)
    {
      FOUR_C_THROW(
          "Each Total traction correction surface condition must have one and only one border "
          "condition");
    }
  }

  return;
}  // end TotalTractionCorrector


/*----------------------------------------------------------------------*
 | Evaluate the velocities of the dof and the map          ismail 04/11 |
 | extractor of boundary condition                                      |
 *----------------------------------------------------------------------*/
void FLD::Utils::TotalTractionCorrector::evaluate_velocities(
    Core::LinAlg::Vector<double>& velocities, double time, double theta, double dta)
{
  std::map<const int, std::shared_ptr<class FluidVolumetricSurfaceFlowBc>>::iterator mapiter;

  for (mapiter = fvsf_map_.begin(); mapiter != fvsf_map_.end(); mapiter++)
  {
    double flowrate = 0.0;

    if (mapiter->second->prebiasing_flag() == "FORCED")
    {
      flowrate = mapiter->second->evaluate_flowrate("TotalTractionCorrectionCond", time);
    }
    else
    {
      Teuchos::ParameterList eleparams;

      discret_->set_state("velaf", velocities);

      flowrate = mapiter->second->FluidVolumetricSurfaceFlowBc::flow_rate_calculation(
          eleparams, time, "TotalTractionCorrectionCond", FLD::calc_flowrate, mapiter->first);
      std::cout << "Traction Corrector_1: Q=" << flowrate << std::endl;
    }

    mapiter->second->FluidVolumetricSurfaceFlowBc::evaluate_velocities(
        flowrate, "TotalTractionCorrectionCond", time);

    Teuchos::ParameterList eleparams;
    mapiter->second->FluidVolumetricSurfaceFlowBc::correct_flow_rate(
        eleparams, "TotalTractionCorrectionCond", FLD::calc_flowrate, time, true);
    mapiter->second->FluidVolumetricSurfaceFlowBc::reset_traction_velocity_comp();
    mapiter->second->FluidVolumetricSurfaceFlowBc::evaluate_traction_velocity_comp(
        eleparams, "TotalTractionCorrectionCond", flowrate, mapiter->first, time, theta, dta);
  }

  return;
}


/*----------------------------------------------------------------------*
 | Update residual                                         ismail 04/11 |
 *----------------------------------------------------------------------*/
void FLD::Utils::TotalTractionCorrector::update_residual(Core::LinAlg::Vector<double>& residual)
{
  std::map<const int, std::shared_ptr<class FluidVolumetricSurfaceFlowBc>>::iterator mapiter;

  for (mapiter = fvsf_map_.begin(); mapiter != fvsf_map_.end(); mapiter++)
  {
    mapiter->second->FluidVolumetricSurfaceFlowBc::update_residual(residual);
  }
}


/*----------------------------------------------------------------------*
 |  Output (public)                                         ismail 04/11|
 *----------------------------------------------------------------------*/
void FLD::Utils::TotalTractionCorrector::output(Core::IO::DiscretizationWriter& output)
{
  std::map<const int, std::shared_ptr<class FluidVolumetricSurfaceFlowBc>>::iterator mapiter;

  for (mapiter = fvsf_map_.begin(); mapiter != fvsf_map_.end(); mapiter++)
  {
    mapiter->second->FluidVolumetricSurfaceFlowBc::output(
        output, "TotalTractionCorrectionCond", mapiter->first);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  read_restart (public)                                    ismail 04/11|
 *----------------------------------------------------------------------*/
void FLD::Utils::TotalTractionCorrector::read_restart(Core::IO::DiscretizationReader& reader)
{
  std::map<const int, std::shared_ptr<class FluidVolumetricSurfaceFlowBc>>::iterator mapiter;

  for (mapiter = fvsf_map_.begin(); mapiter != fvsf_map_.end(); mapiter++)
  {
    mapiter->second->FluidVolumetricSurfaceFlowBc::read_restart(
        reader, "TotalTractionCorrectionCond", mapiter->first);
  }

  return;
}  // FluidVolumetricSurfaceFlowWrapper::read_restart

/*----------------------------------------------------------------------*
 |  Export boundary values and setstate                     ismail 07/14|
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidVolumetricSurfaceFlowBc::export_and_set_boundary_values(
    Core::LinAlg::Vector<double>& source, Core::LinAlg::Vector<double>& target, std::string name)
{
  // define the exporter
  Core::LinAlg::Export exporter(source.get_map(), target.get_map());
  // Export source vector to target vector
  target.export_to(source, exporter, Zero);
  // Set state
  discret_->set_state(name, target);
}

/*----------------------------------------------------------------------*
 |  Export boundary values and setstate                     ismail 07/14|
 *----------------------------------------------------------------------*/
void FLD::Utils::TotalTractionCorrector::export_and_set_boundary_values(
    Core::LinAlg::Vector<double>& source, Core::LinAlg::Vector<double>& target, std::string name)
{
  // define the exporter
  Core::LinAlg::Export exporter(source.get_map(), target.get_map());
  // Export source vector to target vector
  target.export_to(source, exporter, Zero);
  // Set state
  discret_->set_state(name, target);
}

FOUR_C_NAMESPACE_CLOSE

/*-----------------------------------------------------------*/
/*! \file

\brief Methods needed to apply rotationally symmetric periodic boundary
       conditions for fluid problems


\level 1

*/
/*-----------------------------------------------------------*/

#include "4C_fluid_rotsym_periodicbc_utils.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double FLD::get_component_of_rotated_vector_field(const int idf,
    const Teuchos::RCP<const Epetra_Vector> proc0data, const int lid, const double rotangle)
{
  switch (idf)
  {
    case 0:
    {
      // we assume that local dof id of y-component is lid+1
      double xvalue = (*proc0data)[lid];
      double yvalue = (*proc0data)[lid + 1];
      return (xvalue * cos(rotangle) - yvalue * sin(rotangle));
      break;
    }
    case 1:
    {
      // we assume that local dof id of x-component is lid-1
      double xvalue = (*proc0data)[lid - 1];
      double yvalue = (*proc0data)[lid];
      return (xvalue * sin(rotangle) + yvalue * (cos(rotangle)));
      break;
    }
    default:  // we only allow rotation around z-axis!
      break;
  }

  return (*proc0data)[lid];  // case > 1: return unchanged value
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FLD::is_slave_node_of_rot_sym_pbc(const Core::Nodes::Node* node, double& rotangle)
{
  // get periodic surface/line boundary conditions
  std::vector<Core::Conditions::Condition*> pbc;
  node->get_condition("SurfacePeriodic", pbc);
  if (pbc.empty()) node->get_condition("LinePeriodic", pbc);

  bool isrotsymslave(false);
  for (unsigned int j = 0; j < pbc.size(); ++j)
  {
    const std::string& isslave =
        pbc[j]->parameters().get<std::string>("Is slave periodic boundary condition");
    if (isslave == "Slave")
    {
      rotangle = get_rot_angle_from_condition(pbc[j]);
      if (abs(rotangle) > 1e-13)  // angle is not zero
      {
        if (isrotsymslave) FOUR_C_THROW("Node is slave of more than one rot.sym. periodic bc");
        isrotsymslave = true;
      }
    }
  }

  return isrotsymslave;  // yes, it is a slave node with non-zero angle
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double FLD::get_rot_angle_from_condition(const Core::Conditions::Condition* cond)
{
  const double rotangle_deg = cond->parameters().get<double>("ANGLE");

  return rotangle_deg * M_PI / 180.0;  // angle of rotation (RAD);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FLD::get_relevant_slave_nodes_of_rot_sym_pbc(
    std::map<int, double>& pbcslavenodemap, Teuchos::RCP<Core::FE::Discretization> dis)
{
  // get all periodic boundary conditions
  std::vector<Core::Conditions::Condition*> mypbccond;
  dis->get_condition("SurfacePeriodic", mypbccond);
  if (mypbccond.empty())
  {
    dis->get_condition("LinePeriodic", mypbccond);
  }

  // loop the periodic boundary conditions
  for (unsigned numcond = 0; numcond < mypbccond.size(); ++numcond)
  {
    const std::string& mymasterslavetoggle =
        mypbccond[numcond]->parameters().get<std::string>("Is slave periodic boundary condition");
    const double rotangle = FLD::get_rot_angle_from_condition(mypbccond[numcond]);

    // only slave nodes with non-zero angle of rotation require rotation
    // of vector results!
    if ((mymasterslavetoggle == "Slave") && (abs(rotangle) > 1e-13))
    {
      const std::vector<int>* nodes = mypbccond[numcond]->get_nodes();
      for (unsigned int inode = 0; inode < nodes->size(); inode++)
      {
        const int nodegid = nodes->at(inode);
        pbcslavenodemap[nodegid] = rotangle;
      }
    }  // end is slave condition?
  }    // end loop periodic boundary conditions
}

FOUR_C_NAMESPACE_CLOSE

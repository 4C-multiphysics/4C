// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition_locsys.hpp"

#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN



/*-------------------------------------------------------------------*
 |  ctor (public)                                         meier 06/13|
 *-------------------------------------------------------------------*/
Core::Conditions::LocsysManager::LocsysManager(Core::FE::Discretization& discret, const int dim)
    : discret_(discret), dim_(dim), numlocsys_(-1), locsysfunct_(false)
{
  if (dim != 2 && dim != 3) FOUR_C_THROW("Locsys problem must be 2D or 3D");

  // get node row layout of discretization
  const Core::LinAlg::Map* noderowmap = discret_.node_row_map();

  // create locsys vector and initialize to -1
  locsystoggle_ = Core::LinAlg::create_vector(*noderowmap, false);
  locsystoggle_->put_scalar(-1.0);

  // check for locsys boundary conditions
  LocsysManager::discret().get_condition("Locsys", locsysconds_);
  numlocsys_ = (int)locsysconds_.size();
  id_.resize(numlocsys_);
  typelocsys_.resize(numlocsys_);

  for (int i = 0; i < num_locsys(); ++i)
  {
    // Assign internal locsys id (is this still being used?)
    id_[i] = locsysconds_[i]->id();

    // TODO: this hacks the condition parameters!
    auto& parameters = const_cast<IO::InputParameterContainer&>(locsysconds_[i]->parameters());
    // ConditionID not supplied via the input line, thus adding it here
    parameters.add("ConditionID", i);
  }

  // Set boolean that indicates, if a locsys warning has already been thrown, to false
  warning_thrown_ = false;
}

/*-------------------------------------------------------------------*
 *-------------------------------------------------------------------*/
void Core::Conditions::LocsysManager::update(const double time,
    std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> nodenormals,
    const Core::Utils::FunctionManager& function_manager)
{
  nodenormals_ = std::move(nodenormals);
  // IMPORTANT NOTE:
  // The definition of local coordinate systems only makes sense in
  // combination with Dirichlet boundary conditions. This means that
  // in order to define a boundary condition, both locsys
  // AND Dirichlet condition have to formulated for the same entity
  // (i.e. point, line, surface, volume).

  // LIMITATIONS:
  // - So far locsys only works for 2D and 3D solids
  // - Due to this limitation it's necessary to distinguish between this different element types
  //   by means of there nodal DoFs. If further element types are integrated into locsys
  //   more elaborate criteria might be useful.

  // If we have no functions in the locsys conditions the whole Setup method is only
  // conducted once in the constructor (where time is set to -1.0).
  if (time >= 0.0 and !locsysfunct_) return;

  // get dof row map of discretization
  const Core::LinAlg::Map* dofrowmap = discret_.dof_row_map();

  // get node row layout of discretization
  const Core::LinAlg::Map* noderowmap = discret_.node_row_map();

  // Since also time dependent conditions are possible we clear all local systems in the beginning
  nodalrotvectors_.clear();

  // As for Dirichlet conditions, we keep to a very strict hierarchy
  // for evaluation of the Locsys conditions: Volume locsys conditions
  // are evaluated first, followed by Surface and Line locsys conditions
  // and finally Point locsys conditions. This means that nodes carrying
  // different types of locsys conditions are dominated by the rule "Point
  // above Line above Surface above Volume". When two locsys conditions of
  // the same type are defined for one node, ordering in the input file matters!
  std::vector<Core::Conditions::ConditionType> geo_hierarchy;
  geo_hierarchy.emplace_back(Core::Conditions::VolumeLocsys);
  geo_hierarchy.emplace_back(Core::Conditions::SurfaceLocsys);
  geo_hierarchy.emplace_back(Core::Conditions::LineLocsys);
  geo_hierarchy.emplace_back(Core::Conditions::PointLocsys);

  // Ensure locsystoggle_ is on the node row map
  const Core::LinAlg::Map& node_row_map = *discret_.node_row_map();

  // Check if maps match, if not recreate the correct map
  if (locsystoggle_ == nullptr || !locsystoggle_->get_map().same_as(node_row_map))
  {
    locsystoggle_ = Core::LinAlg::create_vector(node_row_map, true);
  }

  //**********************************************************************
  // read locsys conditions in given hierarchical order
  //**************************+*******************************************
  for (auto geo_level : geo_hierarchy)
  {
    for (int i = 0; i < num_locsys(); ++i)
    {
      const Core::Conditions::Condition* currlocsys = locsysconds_[i];

      // safety check
      if (currlocsys->type() != Core::Conditions::VolumeLocsys and
          currlocsys->type() != Core::Conditions::SurfaceLocsys and
          currlocsys->type() != Core::Conditions::LineLocsys and
          currlocsys->type() != Core::Conditions::PointLocsys)
        FOUR_C_THROW("Unknown type of locsys condition!");

      if (currlocsys->type() == geo_level)
      {
        typelocsys_[i] = currlocsys->type();

        const auto rotangle = currlocsys->parameters().get<std::vector<double>>("ROTANGLE");
        const auto funct = currlocsys->parameters().get<std::vector<std::optional<int>>>("FUNCT");
        const auto useUpdatedNodePos = currlocsys->parameters().get<int>("USEUPDATEDNODEPOS");
        const std::vector<int>* nodes = currlocsys->get_nodes();
        const auto useConsistentNodeNormal =
            currlocsys->parameters().get_or<int>("USECONSISTENTNODENORMAL", -1);

        if (currlocsys->type() == Core::Conditions::SurfaceLocsys or
            currlocsys->type() == Core::Conditions::LineLocsys)
        {
          // Check, if we have time dependent locsys conditions (through functions)
          if ((((funct)[0].has_value() && (funct)[0].value() > 0) or
                  ((funct)[1].has_value() && (funct)[1].value() > 0) or
                  ((funct)[2].has_value() && (funct)[2].value() > 0)) or
              ((useConsistentNodeNormal == 1) and (useUpdatedNodePos == 1)))
            locsysfunct_ = true;
        }
        else if (currlocsys->type() == Core::Conditions::VolumeLocsys or
                 currlocsys->type() == Core::Conditions::PointLocsys)
        {
          // Check, if we have time dependent locsys conditions (through functions)
          if (((funct)[0].has_value() && (funct)[0].value() > 0) or
              ((funct)[1].has_value() && (funct)[1].value() > 0) or
              ((funct)[2].has_value() && (funct)[2].value() > 0))
            locsysfunct_ = true;
        }

        // Here we have the convention that 2D problems "live" in the global xy-plane.
        if (n_dim() == 2 and ((rotangle)[0] != 0 or (rotangle)[1] != 0))
        {
          FOUR_C_THROW(
              "For 2D problems (xy-plane) the vector ROTANGLE has to be parallel to the global "
              "z-axis!");
        }

        if ((currlocsys->type() == Core::Conditions::SurfaceLocsys or
                currlocsys->type() == Core::Conditions::LineLocsys) and
            (useConsistentNodeNormal == 1))
          calc_rotation_vector_for_normal_system(i, time);
        else
        {
          // Check, if the updated node positions shall be used for evaluation of the functions
          // 'funct'
          std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
          if (((useUpdatedNodePos) == 1) && (time >= 0.0))
          {
            dispnp = discret().get_state("dispnp");
            if (dispnp == nullptr)
            {
              FOUR_C_THROW(
                  "Locsys: Cannot find state 'dispnp'! You need to set the state 'dispnp' before "
                  "calling the locsys setup.");
            }
          }

          // Each component j of the pseudo rotation vector that rotates the global xyz system onto
          // the local system assigned to each node consists of a constant, a time dependent and
          // spatially variable part: currotangle_j(x,t) = rotangle_j * funct_j(t,x)
          Core::LinAlg::Matrix<3, 1> currotangle(Core::LinAlg::Initialization::zero);

          for (int nodeGID : *nodes)
          {
            bool havenode = discret().have_global_node(nodeGID);
            if (!havenode) continue;

            // Weights of rotations vector due to temporal and spatial function
            for (int j = 0; j < 3; j++)
            {
              // factor given by spatial function
              double functfac = 1.0;
              if ((funct)[j].has_value() && (funct)[j].value() > 0)
              {
                Core::Nodes::Node* node = discret().g_node(nodeGID);

                // Determine node position, which shall be used for evaluating the function, and
                // evaluate it
                if (((useUpdatedNodePos) == 1) && (time >= 0.0))
                {
                  // Obtain current displacement for node
                  std::vector<int> lm;
                  discret().dof(node, lm);

                  std::vector<double> currDisp = Core::FE::extract_values(*dispnp, lm);

                  // Calculate current position for node
                  std::vector<double> currPos(n_dim());
                  const auto& xp = node->x();

                  for (int dim = 0; dim < LocsysManager::n_dim(); ++dim)
                    currPos[dim] = xp[dim] + currDisp[dim];

                  // Evaluate function with current node position
                  functfac = (function_manager.function_by_id<Core::Utils::FunctionOfSpaceTime>(
                                  (funct)[j].value()))
                                 .evaluate(currPos, time, j);
                }
                else
                {
                  // Evaluate function with reference node position
                  functfac = (function_manager.function_by_id<Core::Utils::FunctionOfSpaceTime>(
                                  (funct)[j].value()))
                                 .evaluate(node->x(), time, j);
                }
              }
              currotangle(j) = (rotangle)[j] * functfac;
            }

            nodalrotvectors_[nodeGID] = currotangle;

            int indices = nodeGID;
            double value = static_cast<double>(i);

            const auto& locsystoggle_map = locsystoggle_->get_map();
            const int lid = locsystoggle_map.lid(indices);
            if (lid >= 0)
            {
              // Global setter with global ids
              locsystoggle_->replace_global_values(1, &value, &indices);
            }
          }
        }
      }
    }
  }

  if (time < 0.0) print();

  // When building the transformation matrix we apply a node-by-node
  // strategy. The global matrix trafo_ will consist of nodal blocks
  // of dimension (numdof)x(numdof). The following code block is designed
  // for 2D and 3D solid elements. If special fields are constructed with
  // more than dim geometric dofs, i.e. that have to be transformed, then
  // the following code might have to be modified!

  //**********************************************************************
  // Build transformation matrix trafo_
  //**********************************************************************

  // we need to make sure that two nodes sharing the same dofs are not
  // transformed twice. This is a NURBS/periodic boundary feature.
  std::shared_ptr<Core::LinAlg::Vector<double>> already_processed =
      Core::LinAlg::create_vector(*dofrowmap, true);
  already_processed->put_scalar(0.0);

  // Perform a check for zero diagonal elements. They will crash the SGS-like preconditioners
  bool sanity_check = false;

  // GIDs of all DoFs subjected to local co-ordinate systems
  std::set<int> locsysdofset;

  trafo_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap, 3);

  for (int i = 0; i < noderowmap->num_my_elements(); ++i)
  {
    int nodeGID = noderowmap->gid(i);
    Core::Nodes::Node* node = discret().g_node(nodeGID);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", nodeGID);
    std::vector<int> dofs = discret().dof(0, node);
    int numdof = (int)dofs.size();
    int locsysindex = (int)locsystoggle_->local_values_as_span()[i];

    // skip nodes whose dofs have already been processed
    for (int rr = 0; rr < numdof; ++rr)
      if (already_processed->local_values_as_span()[dofrowmap->lid(dofs[rr])] > 1e-9) continue;

    // unity matrix for non-locsys node
    if (locsysindex < 0)
    {
      for (int r = 0; r < numdof; ++r) trafo_->set_value(1.0, dofs[r], dofs[r]);
    }
    // trafo matrix for locsys node
    else
    {
      Core::LinAlg::SerialDenseMatrix nodetrafo(numdof, numdof);
      for (int k = 0; k < numdof; ++k) nodetrafo(k, k) = 1.0;

      Core::LinAlg::Matrix<3, 1> currrotvector = nodalrotvectors_[nodeGID];
      Core::LinAlg::Matrix<3, 3> currrotationmatrix;

      // Compute rotation matrix out of rotation angle
      Core::LargeRotations::angletotriad(currrotvector, currrotationmatrix);

      // base vectors of local system
      Core::LinAlg::Matrix<3, 1> vec1;
      Core::LinAlg::Matrix<3, 1> vec2;
      Core::LinAlg::Matrix<3, 1> vec3;

      // The columns of the rotation matrix are the base vectors
      for (int j = 0; j < 3; j++)
      {
        vec1(j) = currrotationmatrix(j, 0);
        vec2(j) = currrotationmatrix(j, 1);
        vec3(j) = currrotationmatrix(j, 2);
      }

      // Check for zero-diagonal elements
      if (fabs(vec1(0)) < 1e-9 || fabs(vec2(1)) < 1e-9 || fabs(vec3(2)) < 1e-9) sanity_check = true;

      // trafo for 2D case
      if (n_dim() == 2)
      {
        for (int dim = 0; dim < 2; dim++)
        {
          nodetrafo(0, dim) = vec1(dim);
          nodetrafo(1, dim) = vec2(dim);
        }
      }
      // trafo for 3D case
      if (n_dim() == 3)
      {
        for (int dim = 0; dim < 3; dim++)
        {
          nodetrafo(0, dim) = vec1(dim);
          nodetrafo(1, dim) = vec2(dim);
          nodetrafo(2, dim) = vec3(dim);
        }
      }

      // Assemble the rotation of this dofs ('nodetrafo') into the global matrix
      for (int r = 0; r < n_dim(); ++r)
      {
        for (int c = 0; c < n_dim(); ++c)
        {
          trafo_->set_value(nodetrafo(r, c), dofs[r], dofs[c]);
        }
      }
      for (int r = n_dim(); r < numdof; ++r)
      {
        trafo_->set_value(1.0, dofs[r], dofs[r]);
      }

      // store the DOF with locsys
      for (int r = 0; r < numdof; ++r)
      {
        locsysdofset.insert(dofs[r]);
      }

      // node dofs are marked now as already processed
      for (int rr = 0; rr < numdof; ++rr)
        (*already_processed).get_values()[dofrowmap->lid(dofs[rr])] = 1.0;
    }
  }

  // complete transformation matrix
  trafo_->complete();

  // Throw warning if transformation matrix has zero diagonal elements since
  // they end up on the diagonal of the system matrix. Show this warning only
  // once.
  if ((not warning_thrown_) && sanity_check)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      warning_thrown_ = true;

      printf("Locsys warning:\n");
      printf("A zero diagonal element on the transformation matrix occurred.\n");
      printf("This will probably cause a crash in the preconditioner.\n");
      printf("Try not to rotate your local coordinate system by 90 degrees \n");
      printf("or more or use the slow version.\n");
      printf("This warning won't be repeated anymore.\n");
    }
  }

  //**********************************************************************
  // Build map holding DOFs linked to nodes with local co-ordinate system
  //**********************************************************************

  // create unique/row map of DOFs subjected to local co-ordinate change
  // transformation matrix for relevant DOFs with local system
  int nummyentries = 0;
  int* myglobalentries = nullptr;
  std::vector<int> locsysdofs;
  if (locsysdofset.size() > 0)
  {
    locsysdofs.reserve(locsysdofset.size());
    locsysdofs.assign(locsysdofset.begin(), locsysdofset.end());
    nummyentries = static_cast<int>(locsysdofs.size());
    myglobalentries = locsysdofs.data();
  }
  locsysdofmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, nummyentries, myglobalentries, discret_.dof_row_map()->index_base(), discret_.get_comm());
  if (locsysdofmap_ == nullptr) FOUR_C_THROW("Creation failed.");

  // The matrix subtrafo_ is used in order to apply the Dirichlet Conditions in a more efficient
  // manner
  subtrafo_ = trafo_->extract_dirichlet_rows(*locsysdofmap_);

  /*
  REMARK:
   The most general approach to apply Dirichlet conditions in a rotated, local system would be:
   1) Transform the system into local coordinates by means of
      K \cdot D = F --> \tilde{K} \cdot \tilde{D} = \tilde{F}
      with \tilde{K} = trafo_ \cdot K \cdot trafo_^T, \tilde{F} = trafo_ \cdot F, \tilde{D} = trafo_
  \cdot D 2) Apply Dirichlet conditions in the rotated system 3) Transform the system back into
  global coordinates, i.e. \tilde{K} \cdot \tilde{D} = \tilde{F} --> K \cdot D = F with K = trafo_^T
  \cdot \tilde{K} \cdot trafo_, F = trafo_^T \cdot \tilde{F}, D = trafo_^T \cdot \tilde{D}

   Nevertheless, we apply a more efficient algorithm which can be shown,to deliver an equivalent
  system of equations: 1) Therefore we only apply one left transformation to our system of equations
  according K \cdot D = F --> trafo_ \cdot K \cdot D = trafo_ \cdot F 2) Afterwards we apply the
  rotated Dirichlet conditions in an appropriate manner, i.e. we zero the corresponding Dirichlet
  line and than insert the corresponding local base vector vec_i of the assigned local system into
  the corresponding 3*3-block, e.g. if the DoFs of the locsys node are represented by fourth, fifth
  and sixth column:
      (*,*,*,*,*,*,*,*,*,*,*,*) --> (0,0,0, vec_i^T, 0,0,0,0,0,0)
      We don't invert the left transformation of our system afterwards. This means, that we don't
  solve the original but a algebraic manipulated system of equations, nevertheless we still solve
  for the original, non-rotated DoFs D. However, this is actually no drawback since e.g.
  zero-diagonal elements resulting from rotated Dirichlet conditions would still exist even if we
  applied the back transformation afterwards.
  */
}


/*----------------------------------------------------------------------*
 |  print manager (public)                                   meier 06/13|
 *----------------------------------------------------------------------*/
void Core::Conditions::LocsysManager::print() const
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    Core::IO::cout << "\n-------------------------------------Core::Conditions::LocsysManager"
                   << Core::IO::endl;
    for (int i = 0; i < num_locsys(); ++i)
    {
      Core::IO::cout << "*  *  *  *  *  *  *  *  *  *  *  *  *Locsys entity ID: "
                     << locsysconds_[i]->id();
      if (type_locsys(i) == Core::Conditions::PointLocsys)
        Core::IO::cout << " Point   " << Core::IO::endl;
      else if (type_locsys(i) == Core::Conditions::LineLocsys)
        Core::IO::cout << " Line    " << Core::IO::endl;
      else if (type_locsys(i) == Core::Conditions::SurfaceLocsys)
        Core::IO::cout << " Surface " << Core::IO::endl;
      else if (type_locsys(i) == Core::Conditions::VolumeLocsys)
        Core::IO::cout << " Volume  " << Core::IO::endl;
      else
        FOUR_C_THROW("Unknown type of locsys condition!");
    }
    Core::IO::cout << "-------------------------------------------------------------\n\n";
  }
}


/*----------------------------------------------------------------------*
 |  Get the communicator                                                |
 *----------------------------------------------------------------------*/
inline MPI_Comm Core::Conditions::LocsysManager::get_comm() const { return discret_.get_comm(); }


/*----------------------------------------------------------------------*
 |  Transform system global -> local (public)                 popp 09/08|
 *----------------------------------------------------------------------*/
void Core::Conditions::LocsysManager::rotate_global_to_local(
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat, Core::LinAlg::Vector<double>& rhs) const
{
  // transform rhs vector
  rotate_global_to_local(rhs);

  // selective multiplication from left
  std::shared_ptr<Core::LinAlg::SparseMatrix> temp =
      Core::LinAlg::matrix_multiply(*subtrafo_, false, *sysmat, false, true);
  // put transformed rows back into global matrix
  Core::LinAlg::matrix_put(*temp, 1.0, locsysdofmap_, *sysmat);
}


/*----------------------------------------------------------------------*
 |  Transform system matrix global -> local (public)       mueller 05/10|
 *----------------------------------------------------------------------*/
void Core::Conditions::LocsysManager::rotate_global_to_local(
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat) const
{
  // selective multiplication from left
  std::shared_ptr<Core::LinAlg::SparseMatrix> temp = Core::LinAlg::matrix_multiply(
      *subtrafo_, false, *sysmat, false, sysmat->explicit_dirichlet(), sysmat->save_graph(), true);

  // put transformed rows back into global matrix
  Core::LinAlg::matrix_put(*temp, 1.0, locsysdofmap_, *sysmat);
}

/*----------------------------------------------------------------------*
 |  Transform vector global -> local (public)                 popp 09/08|
 *----------------------------------------------------------------------*/
void Core::Conditions::LocsysManager::rotate_global_to_local(
    Core::LinAlg::Vector<double>& vec, bool offset) const
{
  // y = trafo_ . x  with x = vec
  Core::LinAlg::Vector<double> tmp(vec);
  trafo_->multiply(false, tmp, vec);
}

/*----------------------------------------------------------------------*
 |  Transform result + system local -> global (public)        popp 09/08|
 *----------------------------------------------------------------------*/
void Core::Conditions::LocsysManager::rotate_local_to_global(Core::LinAlg::Vector<double>& result,
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat, Core::LinAlg::Vector<double>& rhs) const
{
  // transform result
  rotate_local_to_global(result);

  // transform rhs vector
  rotate_local_to_global(rhs);

  // transform system matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> temp;
  std::shared_ptr<Core::LinAlg::SparseMatrix> temp2;

  // We want to keep the SaveGraph() value of sysmat also after transformation.
  // It is not possible to keep ExplicitDirichlet()==true after transformation,
  // so we explicitly set this to false.
  temp = Core::LinAlg::matrix_multiply(
      *sysmat, false, *trafo_, false, false, sysmat->save_graph(), true);
  temp2 =
      Core::LinAlg::matrix_multiply(*trafo_, true, *temp, false, false, sysmat->save_graph(), true);

  // this is a deep copy (expensive!)
  *sysmat = *temp2;
}

/*----------------------------------------------------------------------*
 |  Transform vector local -> global (public)                 popp 09/08|
 *----------------------------------------------------------------------*/
void Core::Conditions::LocsysManager::rotate_local_to_global(
    Core::LinAlg::Vector<double>& vec, bool offset) const
{
  Core::LinAlg::Vector<double> tmp(vec);
  trafo_->multiply(true, tmp, vec);
}
/*----------------------------------------------------------------------*
 |  Transform matrix local -> global (public)              mueller 05/10|
 *----------------------------------------------------------------------*/
void Core::Conditions::LocsysManager::rotate_local_to_global(
    Core::LinAlg::SparseMatrix& sysmat) const
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> temp2 =
      Core::LinAlg::matrix_multiply(*trafo_, true, sysmat, false, false, sysmat.save_graph(), true);
  sysmat = *temp2;
}


/*----------------------------------------------------------------------*
 |  Calculate rotation vector for (mass-consistent) normal              |
 |  system for a given locsys condition                       hahn 07/14|
 *----------------------------------------------------------------------*/
void Core::Conditions::LocsysManager::calc_rotation_vector_for_normal_system(
    int numLocsysCond, double time)
{
  // Take care for "negative times", where no information about dispnp_ is available
  if (time < 0.0)
  {
    // TODO is the memory management still working?
    LinAlg::Vector<double> zeroVector(*discret().dof_row_map(), true);
    discret_.set_state("dispnp", zeroVector);
  }

  // Calculate (non-normalized) mass-consistent node normals for a given locsys condition
  // *****+++++++++++++++++**************************************************************

  // check if the normals were set
  if (nodenormals_.empty())
  {
    FOUR_C_THROW(
        "The option massConsistentNodeNormals of the Local SYS BC needs the current normals of the "
        "problem. Before calling Setup, please provide the Node Normals.");
  }

  // Declare node normal variable
  std::shared_ptr<Core::LinAlg::Vector<double>> massConsistentNodeNormals =
      nodenormals_[numLocsysCond];

  // Loop through all nodes in the condition
  // *******************************************************************
  // Obtain desired locsys condition and its corresponding nodes
  const Core::Conditions::Condition* currLocsysCond = locsysconds_[numLocsysCond];
  const std::vector<int>* nodes = currLocsysCond->get_nodes();

  // Obtain rank of calling processor
  int myrank = Core::Communication::my_mpi_rank(discret_.get_comm());

  // Loop through all nodes in the condition
  for (int nodeGID : *nodes)
  {
    // Don't care about nodes that don't exist
    bool haveNode = discret_.have_global_node(nodeGID);
    if (!haveNode) continue;

    Core::Nodes::Node* node = discret_.g_node(nodeGID);

    // Don't care about nodes that the processor doesn't own
    bool isOwner = (node->owner() == myrank);
    if (!isOwner) continue;

    // Obtain all gids of all dofs of the node
    std::vector<int> nodeGIDs;
    discret_.dof(node, nodeGIDs);

    // Obtain node normal for current node and calculate its vector length
    // *******************************************************************
    Core::LinAlg::Matrix<3, 1> nodeNormal;  // massConsistentNodeNormals contains (dim_+1) dofs
                                            // in the fluid case and (dim_) dofs in the ale case,
                                            // but only the first (dim_) are used.
    nodeNormal.clear();                     // if dim_==2, then the third component is just not used
    double length = 0.0;
    for (int jdim = 0; jdim < dim_; jdim++)
    {
      const int localId = massConsistentNodeNormals->get_map().lid(nodeGIDs[jdim]);
      nodeNormal(jdim, 0) = massConsistentNodeNormals->local_values_as_span()[localId];
      length += nodeNormal(jdim, 0) * nodeNormal(jdim, 0);
    }
    length = sqrt(length);
    if (length < 1e-12)
    {
      FOUR_C_THROW(
          "Locsys: calc_rotation_vector_for_normal_system: Node normal length is zero, what "
          "shouldn't "
          "happen! Check, if your BC nodeset really contains surface elements!");
    }

    // Normalize the node normal vector
    for (int jdim = 0; jdim < dim_; jdim++)
    {  // massConsistentNodeNormals contains (dim_+1) dofs
       // in the fluid case and (dim_) dofs in the ale case,
       // but only the first (dim_) are used.
      nodeNormal(jdim, 0) = nodeNormal(jdim, 0) / length;
    }

    // Finally, calculate the rotation vector
    // *******************************************************************
    Core::LinAlg::Matrix<3, 1> currNodalRotVector;

    // Determine rotation angle
    const double rotAngle = acos(nodeNormal(0, 0));

    // Calculate the L2-norm of the rotation vector (which is given by (0, -nodeNormal(2,0),
    // nodeNormal(1,0)))
    const double rotVecNorm =
        sqrt(nodeNormal(1, 0) * nodeNormal(1, 0) + nodeNormal(2, 0) * nodeNormal(2, 0));

    // Calculate the requested rotation vector
    if (rotVecNorm > 1e-12)
    {  // normal vector is not (+-1,0,0), thus rotate as planned
      currNodalRotVector(0, 0) = 0.0;
      currNodalRotVector(1, 0) = rotAngle * (-1) * nodeNormal(2, 0) / rotVecNorm;
      currNodalRotVector(2, 0) = rotAngle * nodeNormal(1, 0) / rotVecNorm;
    }
    else if (nodeNormal(0, 0) < 0)
    {  // normal vector is (-1,0,0), thus rotate 180 deg about z-axis, i.e. (0,0,pi)
      currNodalRotVector(0, 0) = 0.0;
      currNodalRotVector(1, 0) = 0.0;
      currNodalRotVector(2, 0) = std::numbers::pi;
    }
    else
    {  // normal vector is (+1,0,0), thus no rotation at all
      currNodalRotVector(0, 0) = 0.0;
      currNodalRotVector(1, 0) = 0.0;
      currNodalRotVector(2, 0) = 0.0;
    }

    // Write out nodal rotation vector to global list
    nodalrotvectors_[nodeGID] = currNodalRotVector;

    // Do some locsys voodoo
    int indices = nodeGID;
    double values = numLocsysCond;
    locsystoggle_->replace_global_values(1, &values, &indices);
  }
}

FOUR_C_NAMESPACE_CLOSE

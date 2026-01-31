// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_SUBMODELEVALUATOR_MPC_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_SUBMODELEVALUATOR_MPC_HPP

#include "4C_config.hpp"

#include "4C_constraint_framework_input.hpp"
#include "4C_constraint_framework_submodelevaluator_base.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_pstream.hpp"
#include "4C_structure_new_model_evaluator_generic.hpp"

#include <Teuchos_ParameterList.hpp>

#include <map>

FOUR_C_NAMESPACE_OPEN


namespace Constraints::SubmodelEvaluator
{
  class ConstraintEquationBase;


  class RveMultiPointConstraintManager : public ConstraintBase
  {
   public:
    /*!
    \brief Standard Constructor
    */
    RveMultiPointConstraintManager(std::shared_ptr<const Core::FE::Discretization> disc_ptr,
        Core::LinAlg::SparseMatrix* st_ptr);

    //! @name Public evaluation methods

    /*!
      \brief Perform basic checks of the input conditions and parameters
    */
    void check_input();

    /*!
     * \brief Reset the constraint stiffness matrix and delete node pairs
     */
    void reset() override;

    std::map<Solid::EnergyType, double> get_energy() const override;

    //@}

   private:
    //! @name member variables

    //! Map of the Corner Node IDs Ni
    std::map<std::string, Core::Nodes::Node*> rve_ref_node_map_;

    //! RVE reference length vectors
    std::array<double, 2> r_xmxp_, r_ymyp_;

    //! Vector with all Conditions
    std::vector<const Core::Conditions::Condition*> point_linear_coupled_equation_conditions_,
        point_periodic_rve_ref_conditions_, line_periodic_rve_conditions_,
        surface_periodic_rve_conditions_;

    //! Tolerance for the opposing edge node search
    double node_search_toler_;

    //! Dimension of the rve boundary
    Constraints::MultiPoint::RveDimension rve_dim_;

    //! Type of reference vector definition
    Constraints::MultiPoint::RveReferenceDeformationDefinition rve_ref_type_;

    //@}

   private:
    //! @name Private evaluation methods

    //! find the opposite edge node pairs of the periodic rve
    void build_periodic_mp_cs(std::map<std::string, const std::vector<int>*>& rveBoundaryNodeIdMap_,
        std::map<std::string, int>& rveCornerNodeIdMap_);

    //! add linear mpcs to the mpcList
    int build_linear_mp_cs();

    //! find a node that is member of edge1 and edge2
    int find_periodic_rve_corner_nodes(
        const std::vector<int>* edge1, const std::vector<int>* edge2);

    //! find a node that is member of surf1 thru surf3
    int find_periodic_rve_corner_nodes(const std::vector<int>* surf1, const std::vector<int>* surf2,
        const std::vector<int>* surf3);

    //! find the nodes contained i a mpc for the pbcs
    int find_opposite_edge_node(const int nodeID, Constraints::MultiPoint::RveEdgeIdentifiers edge,
        std::map<std::string, const std::vector<int>*>& rveBoundaryNodeIdMap_);

    //! find the corner nodes of the periodic rve
    void build_periodic_rve_corner_node_map(
        std::map<std::string, const std::vector<int>*>& rveBoundaryNodeIdMap,
        std::map<std::string, int>& rveCornerNodeIdMap_);

    //! retrieve design line periodic rve boundary condition
    void build_periodic_rve_boundary_node_map(
        std::map<std::string, const std::vector<int>*>& rveBoundaryNodeIdMap_);
    //@}
  };
}  // namespace Constraints::SubmodelEvaluator
FOUR_C_NAMESPACE_CLOSE
#endif

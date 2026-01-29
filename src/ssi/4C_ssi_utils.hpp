// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSI_UTILS_HPP
#define FOUR_C_SSI_UTILS_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_fem_condition.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Coupling::Adapter
{
  class Coupling;
  class CouplingSlaveConverter;
}  // namespace Coupling::Adapter

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class BlockSparseMatrixBase;
  class MapExtractor;
  class MultiMapExtractor;
  class SparseMatrix;
  class SparseOperator;
  enum class MatrixType;
}  // namespace Core::LinAlg

namespace SSI
{
  class SSIBase;
  class SsiMono;
  enum class Subproblem;

  namespace Utils
  {
    // forward declaration
    class SSIMaps;
    class SSISlaveSideConverter;

    //! Modification of time parameter list for problem with different time step size
    void change_time_parameter(MPI_Comm comm, Teuchos::ParameterList& ssiparams,
        Teuchos::ParameterList& scatradyn, Teuchos::ParameterList& sdyn);

    //! check for a consistent input file definition of the SSIInterfaceContact condition
    void check_consistency_of_ssi_interface_contact_condition(
        const std::vector<const Core::Conditions::Condition*>& conditionsToBeTested,
        const Core::FE::Discretization& structdis);

    /// Function for checking that the different time steps are a
    /// multiplicative of each other
    int check_time_stepping(double dt1, double dt2);

    //! clone scatra specific parameters for solver of manifold. Add manifold specific parameters
    Teuchos::ParameterList clone_scatra_manifold_params(const Teuchos::ParameterList& scatraparams,
        const Teuchos::ParameterList& sublist_manifold_params);

    //! modify scatra parameters for ssi specific values
    Teuchos::ParameterList modify_scatra_params(const Teuchos::ParameterList& scatraparams);


    /*---------------------------------------------------------------------------------*
     *---------------------------------------------------------------------------------*/
    //! sets up and holds all sub blocks of system matrices and system matrix for SSI simulations
    class SSIMatrices
    {
     public:
      /*!
       * @brief constructor
       *
       * @param[in] ssi_maps            ssi maps object containing all relevant maps
       * @param[in] ssi_matrixtype      the ssi matrix type
       * @param[in] scatra_matrixtype   the scalar transport matrix type
       * @param[in] is_scatra_manifold  flag indicating if a scatra manifold is used
       */
      SSIMatrices(const SSIMaps& ssi_maps, Core::LinAlg::MatrixType ssi_matrixtype,
          Core::LinAlg::MatrixType scatra_matrixtype, bool is_scatra_manifold);

      void complete_scatra_manifold_scatra_matrix() const;

      //! call complete on the scalar transport manifold - structure off-diagonal matrix
      void complete_scatra_manifold_structure_matrix() const;

      void complete_scatra_scatra_manifold_matrix() const;

      //! call complete on the scalar transport - structure off-diagonal matrix
      void complete_scatra_structure_matrix() const;

      //! call complete on the structure - scalar transport off-diagonal matrix
      void complete_structure_scatra_matrix() const;

      //! method that clears all ssi matrices
      void clear_matrices() const;

      //! return the system matrix
      [[nodiscard]] std::shared_ptr<Core::LinAlg::SparseOperator> system_matrix() const
      {
        return system_matrix_;
      }

      //! return sub blocks of system matrix
      //@{
      [[nodiscard]] std::shared_ptr<Core::LinAlg::SparseOperator> scatra_matrix() const
      {
        return scatra_matrix_;
      }
      [[nodiscard]] std::shared_ptr<Core::LinAlg::SparseOperator> scatra_manifold_structure_matrix()
          const
      {
        return scatramanifold_structure_matrix_;
      }
      [[nodiscard]] std::shared_ptr<Core::LinAlg::SparseOperator> scatra_structure_matrix() const
      {
        return scatra_structure_matrix_;
      }
      [[nodiscard]] std::shared_ptr<Core::LinAlg::SparseOperator> structure_scatra_matrix() const
      {
        return structure_scatra_matrix_;
      }
      [[nodiscard]] std::shared_ptr<Core::LinAlg::SparseMatrix> structure_matrix() const
      {
        return structure_matrix_;
      }
      [[nodiscard]] std::shared_ptr<Core::LinAlg::SparseOperator> manifold_matrix() const
      {
        return manifold_matrix_;
      }
      [[nodiscard]] std::shared_ptr<Core::LinAlg::SparseOperator> scatra_scatra_manifold_matrix()
          const
      {
        return scatra_scatramanifold_matrix_;
      }
      [[nodiscard]] std::shared_ptr<Core::LinAlg::SparseOperator> scatra_manifold_scatra_matrix()
          const
      {
        return scatramanifold_scatra_matrix_;
      }
      //@}

      /*!
       * @brief set up a pointer to a block matrix
       *
       * @param[in] row_map  row map the block matrix is based on
       * @param[in] col_map  column map the block matrix is based on
       * @return pointer to block matrix
       */
      static std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> setup_block_matrix(
          const Core::LinAlg::MultiMapExtractor& row_map,
          const Core::LinAlg::MultiMapExtractor& col_map);

      /*!
       * @brief set up a pointer to a sparse matrix
       *
       * @param[in] row_map  row map the sparse matrix is based on
       * @return pointer to sparse matrix
       */
      static std::shared_ptr<Core::LinAlg::SparseMatrix> setup_sparse_matrix(
          const Core::LinAlg::Map& row_map);

     private:
      /*!
       * @brief initialize the scatra-structure interaction main-diagonal matrices
       *
       * @param[in] ssi_maps            pointer to the ssi maps object containing all relevant maps
       */
      void initialize_main_diag_matrices(const SSIMaps& ssi_maps);

      /*!
       * @brief initialize the scatra-structure interaction off-diagonal matrices
       *
       * @param[in] ssi_maps            pointer to the ssi maps object containing all relevant maps
       */
      void initialize_off_diag_matrices(const SSIMaps& ssi_maps);

      /*!
       * @brief initialize the system matrix
       *
       * @param[in] ssi_maps         pointer to the ssi maps object containing all relevant maps
       * @param[in] ssi_matrixtype   the ssi matrix type
       */
      void initialize_system_matrix(
          const SSIMaps& ssi_maps, Core::LinAlg::MatrixType ssi_matrixtype);

      //! flag indicating if we have a scatra manifold
      const bool is_scatra_manifold_;

      //! matrix type of scatra matrix
      const Core::LinAlg::MatrixType scatra_matrixtype_;

      //! the scalar transport dof row map
      std::shared_ptr<const Core::LinAlg::Map> scatra_dofrowmap_;

      //! the scalar transport manifold dof row map
      std::shared_ptr<const Core::LinAlg::Map> scatramanifold_dofrowmap_;

      //! the structure dof row map
      std::shared_ptr<const Core::LinAlg::Map> structure_dofrowmap_;

      //! system matrix
      std::shared_ptr<Core::LinAlg::SparseOperator> system_matrix_;
      //! sub blocks of system matrix
      //@{
      std::shared_ptr<Core::LinAlg::SparseOperator> scatra_matrix_;
      std::shared_ptr<Core::LinAlg::SparseOperator> scatramanifold_structure_matrix_;
      std::shared_ptr<Core::LinAlg::SparseOperator> scatra_structure_matrix_;
      std::shared_ptr<Core::LinAlg::SparseOperator> structure_scatra_matrix_;
      std::shared_ptr<Core::LinAlg::SparseMatrix> structure_matrix_;
      std::shared_ptr<Core::LinAlg::SparseOperator> manifold_matrix_;
      std::shared_ptr<Core::LinAlg::SparseOperator> scatra_scatramanifold_matrix_;
      std::shared_ptr<Core::LinAlg::SparseOperator> scatramanifold_scatra_matrix_;
      //@}
    };

    /*---------------------------------------------------------------------------------*
     *---------------------------------------------------------------------------------*/
    //! sets up and holds the system residuals and increment for SSI simulations
    class SSIVectors
    {
     public:
      /*!
       * @brief constructor
       *
       * @param[in] ssi_maps            ssi maps object containing all relevant maps
       * @param[in] is_scatra_manifold  flag indicating if a scatra manifold is used
       */
      explicit SSIVectors(const SSIMaps& ssi_maps, bool is_scatra_manifold);

      //! clear the increment vector
      void clear_increment() const;

      //! clear all residual vectors
      void clear_residuals() const;

      //! global increment vector for Newton-Raphson iteration
      [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> increment() const
      {
        return increment_;
      }

      //! residual vector on right-hand side of global system of equations
      [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> residual() const
      {
        return residual_;
      }

      //! residual vector on right-hand side of scalar transport system
      [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> scatra_residual() const
      {
        return scatra_residual_;
      }

      //! residual vector on right-hand side of structure system
      [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> structure_residual() const
      {
        return structure_residual_;
      }

      [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> manifold_residual() const
      {
        return manifold_residual_;
      }

     private:
      //! global increment vector for Newton-Raphson iteration
      std::shared_ptr<Core::LinAlg::Vector<double>> increment_;

      //! flag indicating if we have a scatra manifold
      const bool is_scatra_manifold_;

      //! residual vector on right-hand side of manifold scalar transport system
      std::shared_ptr<Core::LinAlg::Vector<double>> manifold_residual_;

      //! residual vector on right-hand side of global system of equations
      std::shared_ptr<Core::LinAlg::Vector<double>> residual_;

      //! residual vector on right-hand side of scalar transport system
      std::shared_ptr<Core::LinAlg::Vector<double>> scatra_residual_;

      //! residual vector on right-hand side of structure system
      std::shared_ptr<Core::LinAlg::Vector<double>> structure_residual_;
    };

    /*---------------------------------------------------------------------------------*
     *---------------------------------------------------------------------------------*/
    class SSIMaps
    {
     public:
      //! constructor
      explicit SSIMaps(const SsiMono& ssi_mono_algorithm);

      //! get vector containing positions within system matrix for specific subproblem
      [[nodiscard]] std::vector<int> get_block_positions(Subproblem subproblem) const;

      //! get position within global dof map for specific sub problem
      static int get_problem_position(Subproblem subproblem);

      //! the multi map extractor of the scalar transport field
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_scatra() const;

      //! the multi map extractor of the scalar transport on manifold field
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::MultiMapExtractor>
      block_map_scatra_manifold() const;

      //! the multi map extractor of the structure field
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_structure()
          const;

      //! map extractor associated with blocks of global system matrix
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_system_matrix()
          const
      {
        return block_map_system_matrix_;
      }

      //! all dofs of the SSI algorithm
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> map_system_matrix() const
      {
        return map_system_matrix_;
      }

      /*!
       * @brief global map extractor
       * @note only access with GetProblemPosition method
       */
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::MultiMapExtractor> maps_sub_problems() const
      {
        return maps_sub_problems_;
      }

      //! the scalar transport dof row map
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> scatra_dof_row_map() const;

      //! the scalar transport on manifolds dof row map
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> scatra_manifold_dof_row_map() const;

      //! the structure dof row map
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> structure_dof_row_map() const;

     private:
      //! create and check the block maps of all sub problems
      void create_and_check_block_maps_sub_problems(const SsiMono& ssi_mono_algorithm);

      //! block maps of all sub problems organized in std map
      std::map<Subproblem, std::shared_ptr<const Core::LinAlg::MultiMapExtractor>>
          block_maps_sub_problems_;

      //! map extractor associated with blocks of global system matrix
      std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_system_matrix_;

      //! all dofs of the SSI algorithm
      std::shared_ptr<const Core::LinAlg::Map> map_system_matrix_;

      /*!
       * @brief global map extractor
       * @note only access with GetProblemPosition method
       */
      std::shared_ptr<const Core::LinAlg::MultiMapExtractor> maps_sub_problems_;

      //! matrix type of scatra matrix
      const Core::LinAlg::MatrixType scatra_matrixtype_;

      //! matrix type of scatra manifold matrix
      const Core::LinAlg::MatrixType scatra_manifold_matrixtype_;

      //! matrix type of ssi matrix
      const Core::LinAlg::MatrixType ssi_matrixtype_;
    };

    class SSIMeshTyingHandler
    {
     public:
      explicit SSIMeshTyingHandler(
          std::shared_ptr<Coupling::Adapter::Coupling> slave_master_coupling,
          std::shared_ptr<Core::LinAlg::MultiMapExtractor> slave_master_extractor,
          std::shared_ptr<Coupling::Adapter::Coupling> slave_slave_transformation);

      //! coupling adapter between master and slave coupling
      [[nodiscard]] std::shared_ptr<Coupling::Adapter::Coupling> slave_master_coupling() const
      {
        return slave_master_coupling_;
      }

      //! map extractor for coupling adapter: 0: interior, 1: slave, 2: master
      [[nodiscard]] std::shared_ptr<Core::LinAlg::MultiMapExtractor> slave_master_extractor() const
      {
        return slave_master_extractor_;
      }

      //! converter to convert slave dofs to master side
      [[nodiscard]] std::shared_ptr<Coupling::Adapter::CouplingSlaveConverter>
      slave_side_converter() const
      {
        return slave_side_converter_;
      }

      //! coupling adapter between new slave nodes and slave nodes from input file
      [[nodiscard]] std::shared_ptr<Coupling::Adapter::Coupling> slave_slave_transformation() const
      {
        return slave_slave_transformation_;
      }

     private:
      //! coupling adapter between master and slave coupling
      std::shared_ptr<Coupling::Adapter::Coupling> slave_master_coupling_;

      //! map extractor for coupling adapter: 0: interior, 1: slave, 2: master
      std::shared_ptr<Core::LinAlg::MultiMapExtractor> slave_master_extractor_;

      //! converter to convert slave dofs to master side
      std::shared_ptr<Coupling::Adapter::CouplingSlaveConverter> slave_side_converter_;

      //! coupling adapter between new slave nodes and slave nodes from input file
      std::shared_ptr<Coupling::Adapter::Coupling> slave_slave_transformation_;
    };

    class SSIMeshTying
    {
     public:
      explicit SSIMeshTying(const std::string& conditionname_coupling,
          const Core::FE::Discretization& dis, bool build_slave_slave_transformation,
          bool check_over_constrained);

      //! check if one dof has slave side conditions and Dirichlet conditions
      void check_slave_side_has_dirichlet_conditions(
          std::shared_ptr<const Core::LinAlg::Map> struct_dbc_map) const;

      //! all master side dofs
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> full_master_side_map() const
      {
        return full_master_side_map_;
      }

      //! all slave side dofs
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> full_slave_side_map() const
      {
        return full_slave_side_map_;
      }

      //! all interior dofs
      [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> interior_map() const
      {
        return interior_map_;
      }

      //! vector of all mesh tying handlers
      [[nodiscard]] std::vector<std::shared_ptr<SSIMeshTyingHandler>> mesh_tying_handlers() const
      {
        return meshtying_handlers_;
      }

     private:
      //! Define master nodes and subsequently master slave pairs
      //! \param dis               [in] discretization
      //! \param grouped_matching_nodes   [in] vector of vector of nodes at same position
      //! \param master_gids              [out] vector of all defined master nodes
      //! \param slave_master_pair        [out] map between slave nodes (key) and master nodes
      //!                                 (value)
      //! \param check_over_constrained   [in] check if two DBCs are set on two dofs at the same
      //! position
      void define_master_slave_pairing(const Core::FE::Discretization& dis,
          const std::vector<std::vector<int>>& grouped_matching_nodes,
          std::vector<int>& master_gids, std::map<int, int>& slave_master_pair,
          bool check_over_constrained) const;

      //! Construct global pairs between matching nodes
      //! \param dis                 [in] discretization
      //! \param name_meshtying_condition   [in] name of meshtying condition
      //! \param coupling_pairs         [out] vector of pairs of matching nodes
      void find_matching_node_pairs(const Core::FE::Discretization& dis,
          const std::string& name_meshtying_condition,
          std::vector<std::pair<int, int>>& coupling_pairs) const;

      //! Find match between new slave nodes and slave nodes from input file
      //! \param dis                        [in] discretization
      //! \param name_meshtying_condition          [in] name of meshtying condition
      //! \param inodegidvec_slave                 [in] new slave nodes on this proc
      //! \param all_coupled_original_slave_gids   [out] old slave nodes that match new slave nodes
      void find_slave_slave_transformation_nodes(const Core::FE::Discretization& dis,
          const std::string& name_meshtying_condition, const std::vector<int>& inodegidvec_slave,
          std::vector<int>& all_coupled_original_slave_gids) const;

      //! Get number of slave nodes that are assigned to a master node
      //! \param slave_master_pair                   [in] map between slave nodes (key) and master
      //!                                             nodes
      //!                                            (value)
      //! \param num_assigned_slave_to_master_nodes  [out] map between master nodes (master) and
      //!                                            number of assigned slave nodes
      //! \param max_assigned_slave_nodes            [out] max. number of slave nodes assigned to a
      //!                                            master node
      void get_num_assigned_slave_to_master_nodes(const std::map<int, int>& slave_master_pair,
          std::map<int, int>& num_assigned_slave_to_master_nodes,
          int& max_assigned_slave_nodes) const;

      //! Group nodes that are at the geometrically same position
      //! \param coupling_pairs           [in] vector of pairs of matching nodes
      //! \param grouped_matching_nodes   [out] vector of vector of nodes at same position
      void group_matching_nodes(const std::vector<std::pair<int, int>>& coupling_pairs,
          std::vector<std::vector<int>>& grouped_matching_nodes) const;

      //! Check if matching_nodes has this gid.
      //! \param gid             [in] gid to be checked
      //! \param matching_nodes  [in] grouped node gids
      //! \return                index in matching_nodes (outer vector) where gid is, otherwise -1
      [[nodiscard]] int has_gid(int gid, const std::vector<std::vector<int>>& matching_nodes) const;

      //! Check if subset of matching_nodes has this gid.
      //! \param gid             [in] gid to be checked
      //! \param start           [in] first index in matching_nodes
      //! \param end             [in] last index in matching_nodes
      //! \param matching_nodes  [in] grouped node gids
      //! \return                index in matching_nodes (outer vector) between start and end where
      //!                        gid is, otherwise -1
      [[nodiscard]] int has_gid_partial(
          int gid, int start, int end, const std::vector<std::vector<int>>& matching_nodes) const;

      //! Construct mesh tying handlers based on conditions with name conditionname_coupling
      //! \param dis                  [in] discretization
      //! \param name_meshtying_condition    [in] name of meshtying condition
      //! \param build_slave_slave_transformation [in] is a map required that defines the
      //! transformation from slave nodes at the input and matched slave nodes
      void setup_mesh_tying_handlers(const Core::FE::Discretization& dis,
          const std::string& name_meshtying_condition, bool build_slave_slave_transformation,
          bool check_over_constrained);

      //! communicator
      MPI_Comm comm_;

      //! should this proc write screen output
      const bool do_print_;

      //! all master side dofs
      std::shared_ptr<const Core::LinAlg::Map> full_master_side_map_;

      //! all slave side dofs
      std::shared_ptr<const Core::LinAlg::Map> full_slave_side_map_;

      //! all interior dofs
      std::shared_ptr<const Core::LinAlg::Map> interior_map_;

      //! all mesh tying handlers
      std::vector<std::shared_ptr<SSIMeshTyingHandler>> meshtying_handlers_;

      //! number of proc ID
      const int my_rank_;

      //! number of procs
      const int num_proc_;
    };

  }  // namespace Utils
}  // namespace SSI

FOUR_C_NAMESPACE_CLOSE

#endif

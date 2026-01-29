// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSI_MANIFOLD_UTILS_HPP
#define FOUR_C_SSI_MANIFOLD_UTILS_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_ssi_utils.hpp"

#include <memory>
#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace Coupling::Adapter
{
  class Coupling;
  class CouplingSlaveConverter;
  class CouplingMasterConverter;
}  // namespace Coupling::Adapter

namespace Adapter
{
  class ScaTraBaseAlgorithm;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class MapExtractor;
  class MultiMapExtractor;
  class SparseMatrix;
  class SparseOperator;
  enum class MatrixType;
}  // namespace Core::LinAlg

namespace SSI
{

  class SsiMono;

  //! type of coupling/system matrix block
  enum class BlockMatrixType : std::uint8_t
  {
    ManifoldScaTra,     //! derivative of flux into/out of manifold w.r.t. scatra
    ManifoldStructure,  //! derivative of flux into/out of manifold w.r.t. structure
    ScaTraManifold,     //! derivative of flux into/out of scatra w.r.t. manifold
    ScaTraStructure,    //! derivative of flux into/out of scatra w.r.t. structure
    SysMatManifold,     //! derivative of flux into/out of manifold w.r.t. manifold
    SysMatScaTra        //! derivative of flux into/out of scatra w.r.t. scatra
  };

  //! holds everything to evaluate coupling between scatra and scatra manifold
  class ManifoldScaTraCoupling
  {
   public:
    ManifoldScaTraCoupling(const Core::FE::Discretization& manifold_discretization,
        const Core::FE::Discretization& scatra_discretization,
        const Core::Conditions::Condition* condition_manifold,
        const Core::Conditions::Condition* condition_kinetics, int ndof_per_node);

    //! Check if graph of matrix has changed compared to last evaluation. Afterwards, store new size
    //! of graph
    //! \param block  graph of which matrix should be checked
    //! \param size   new size of graph
    //! \return       does old size of graph match new size?
    bool check_and_set_size_of_matrix_graph(BlockMatrixType block, int size);

    //! Kinetics condition on scatra dis
    const Core::Conditions::Condition* condition_kinetics() const { return condition_kinetics_; }

    //! coupling adapter between manifold (slave) and scatra (master)
    [[nodiscard]] const Coupling::Adapter::Coupling& coupling_adapter() const
    {
      return coupling_adapter_;
    }

    //! inverse of thickness of manifold
    [[nodiscard]] double inv_thickness() const { return inv_thickness_; }

    //! condition ID of manifold condition
    int manifold_condition_id() const { return manifold_condition_id_; }

    //! from master to slave side
    std::shared_ptr<Coupling::Adapter::CouplingMasterConverter> master_converter() const
    {
      return master_converter_;
    }

    //! condition ID of kinetics condition
    int kinetics_condition_id() const { return kinetics_condition_id_; }

    //! Map extractor for dofs in this manifold condition
    std::shared_ptr<Core::LinAlg::MapExtractor> manifold_map_extractor() const
    {
      return manifold_map_extractor_;
    }

    //! Map extractor for dofs in this kinetics condition
    std::shared_ptr<Core::LinAlg::MapExtractor> scatra_map_extractor() const
    {
      return scatra_map_extractor_;
    }

   private:
    //! Kinetics condition on scatra dis
    const Core::Conditions::Condition* condition_kinetics_;

    //! coupling adapter between manifold (slave) and scatra (master)
    Coupling::Adapter::Coupling coupling_adapter_;

    //! inverse of thickness of manifold
    const double inv_thickness_;

    //! condition ID of manifold condition
    const int manifold_condition_id_;

    //! condition ID of kinetics condition
    const int kinetics_condition_id_;

    //! Map extractor for dofs in this manifold condition
    std::shared_ptr<Core::LinAlg::MapExtractor> manifold_map_extractor_;

    //! Master converter for scatra - manifold coupling
    std::shared_ptr<Coupling::Adapter::CouplingMasterConverter> master_converter_;

    //! Map extractor for dofs in this kinetics condition
    std::shared_ptr<Core::LinAlg::MapExtractor> scatra_map_extractor_;

    //! map with the size of the graph of each matrix
    std::map<BlockMatrixType, int> size_matrix_graph_;
  };

  /*---------------------------------------------------------------------------------*
   *---------------------------------------------------------------------------------*/
  //!
  class ScaTraManifoldScaTraFluxEvaluator
  {
   public:
    explicit ScaTraManifoldScaTraFluxEvaluator(const SSI::SsiMono& ssi_mono);

    //! call complete on system matrices and coupling matrices
    //@{
    void complete_matrix_manifold_scatra() const;
    void complete_matrix_manifold_structure() const;
    void complete_matrix_scatra_manifold() const;
    void complete_matrix_scatra_structure() const;
    void complete_system_matrix_manifold() const;
    void complete_system_matrix_scatra() const;
    //@}

    //! write inflow fluxes to csv file
    bool do_output() const { return do_output_; }

    //! Evaluate everything including coupling
    void evaluate();

    //! Evaluate inflow into manifold field from coupling with scatra field
    void evaluate_scatra_manifold_inflow();

    //! get all RHS
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_manifold() { return rhs_manifold_; }
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_scatra() { return rhs_scatra_; }
    //@}

    //! get all matrices
    //@{
    std::shared_ptr<Core::LinAlg::SparseOperator> system_matrix_manifold()
    {
      return systemmatrix_manifold_;
    }
    std::shared_ptr<Core::LinAlg::SparseOperator> system_matrix_scatra()
    {
      return systemmatrix_scatra_;
    }

    std::shared_ptr<Core::LinAlg::SparseOperator> matrix_manifold_scatra()
    {
      return matrix_manifold_scatra_;
    }
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix_scatra_manifold()
    {
      return matrix_scatra_manifold_;
    }
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix_manifold_structure()
    {
      return matrix_manifold_structure_;
    }
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix_scatra_structure()
    {
      return matrix_scatra_structure_;
    }
    //@}

    //! write coupling fluxes (inflow into manifold) to csv file
    void output() const;

    //! return all scatra-scatra manifold coupling operators
    std::vector<std::shared_ptr<SSI::ManifoldScaTraCoupling>>& scatra_manifold_couplings()
    {
      return scatra_manifold_couplings_;
    }

   private:
    //! Add to global matrices and rhs
    void add_condition_contribution() const;

    //! Copy and scale (-1.0) to manifold side
    void copy_scatra_scatra_manifold_side(
        const ManifoldScaTraCoupling& scatra_manifold_coupling) const;

    //! Evaluate flux and linearizations on bulk side
    void evaluate_bulk_side(ManifoldScaTraCoupling& scatra_manifold_coupling);

    //! Evaluate integral on scatra manifold over 1.0
    void evaluate_scatra_manifold_domain_integral(
        const ManifoldScaTraCoupling& scatra_manifold_coupling);

    //! Evaluate integral on scatra manifold over positive fluxes
    void evaluate_scatra_manifold_inflow_integral(
        const ManifoldScaTraCoupling& scatra_manifold_coupling);

    //! prepare evaluation of coupling condition: set elemental data
    void pre_evaluate(const ManifoldScaTraCoupling& scatra_manifold_coupling);

    //! uncomplete all global matrices if any matrices holding the condition contributions have
    //! updated graphs (i.e. zeros become non-zeros or vice versa). In this case the graph of the
    //! global matrices needs to be updated as well to be able to add the local matrices to the
    //! global matrices
    void un_complete_matrices_if_necessary(ManifoldScaTraCoupling& scatra_manifold_coupling) const;

    //! map extractor associated with all degrees of freedom inside scatra field
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_scatra_;

    //! map extractor associated with all degrees of freedom inside scatra manifold field
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_scatra_manifold_;

    //! map extractor associated with all degrees of freedom inside structure field
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_structure_;

    //! write inflow fluxes to csv file
    const bool do_output_;

    //! integral of manifold domain
    std::map<int, double> domainintegral_;

    //! map of all scatra manifold dofs
    std::shared_ptr<const Core::LinAlg::Map> full_map_manifold_;

    //! map extractor associated with all degrees of freedom inside scatra field
    std::shared_ptr<const Core::LinAlg::Map> full_map_scatra_;

    //! map extractor associated with all degrees of freedom inside structural field
    std::shared_ptr<const Core::LinAlg::Map> full_map_structure_;

    //! integrated flux for each scalar into manifold
    std::map<int, std::vector<double>> inflow_;

    //! coupling matrices
    //@{
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix_manifold_scatra_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> matrix_manifold_scatra_cond_;
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix_manifold_structure_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> matrix_manifold_structure_cond_;
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix_scatra_manifold_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> matrix_scatra_manifold_cond_;
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix_scatra_structure_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> matrix_scatra_structure_cond_;
    //@}

    //! rhs for manifold and scatra
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_manifold_;
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_manifold_cond_;
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_scatra_;
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_scatra_cond_;
    //@}

    // writes evaluated data to output
    std::optional<Core::IO::RuntimeCsvWriter> runtime_csvwriter_;

    //! scatra problem
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra_;

    //! scatra manifold problem
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra_manifold_;

    //! all scatra-scatra manifold coupling operators
    std::vector<std::shared_ptr<SSI::ManifoldScaTraCoupling>> scatra_manifold_couplings_;

    //! SSI structure mesh tying object containing coupling adapters, converters and maps
    std::shared_ptr<SSI::Utils::SSIMeshTying> ssi_structure_meshtying_;

    //! system matrices of scatra and manifold
    //@{
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix_manifold_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> systemmatrix_manifold_cond_;
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix_scatra_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> systemmatrix_scatra_cond_;
    //@}
  };

  //! data types for mesh tying handlers for sparse case: coupling adapter and MultiMapExtractor
  //! splitting dofs into interior, master, and slave
  using meshtying_handler_type = std::pair<std::shared_ptr<Coupling::Adapter::Coupling>,
      std::shared_ptr<Core::LinAlg::MultiMapExtractor>>;
  //! data types for mesh tying handlers for block case: standard mesh tying handler and
  //! MultiMapExtractor splitting slave dofs into blocks
  using meshtying_block_handler_type =
      std::pair<std::vector<std::shared_ptr<Coupling::Adapter::Coupling>>,
          std::vector<std::shared_ptr<const Core::LinAlg::Map>>>;

  //! Base class to handle mesh tying between manifold fields
  class ManifoldMeshTyingStrategyBase
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~ManifoldMeshTyingStrategyBase() = default;

    explicit ManifoldMeshTyingStrategyBase(const Core::FE::Discretization& scatra_manifold_dis,
        std::shared_ptr<SSI::Utils::SSIMaps> ssi_maps, bool is_manifold_meshtying);

    //! apply mesh tying to right hand side
    void apply_mesh_tying_to_manifold_rhs(Core::LinAlg::Vector<double>& rhs_manifold) const;

    //! apply mesh tying to manifold system matrix
    virtual void apply_meshtying_to_manifold_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_matrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_matrix) = 0;

    //! apply mesh tying to scatra manifold - scatra matrix
    virtual void apply_meshtying_to_manifold_scatra_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_scatra_matrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_scatra_matrix) = 0;

    //! apply mesh tying to scatra manifold - structure matrix
    virtual void apply_meshtying_to_manifold_structure_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_structure_matrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_structure_matrix,
        bool do_uncomplete) = 0;

    //! apply mesh tying to scatra manifold - scatra matrix
    virtual void apply_meshtying_to_scatra_manifold_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> ssi_scatra_manifold_matrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatra_manifold_matrix,
        bool do_uncomplete) = 0;

    //! coupling adapters, maps, and extractors for mesh tying
    std::shared_ptr<const SSI::Utils::SSIMeshTying> ssi_mesh_tying() const
    {
      return ssi_meshtying_;
    }

   protected:
    //! should meshtying between manifold fields be applied?
    const bool is_manifold_meshtying_;

    //! all interior and master dofs
    std::shared_ptr<const Core::LinAlg::Map> condensed_dof_map_;

    //! this object holds all maps relevant to monolithic scalar transport - structure interaction
    std::shared_ptr<SSI::Utils::SSIMaps> ssi_maps_;

   private:
    //! coupling adapters, maps, and extractors for mesh tying
    std::shared_ptr<const SSI::Utils::SSIMeshTying> ssi_meshtying_;
  };

  class ManifoldMeshTyingStrategySparse : public ManifoldMeshTyingStrategyBase
  {
   public:
    explicit ManifoldMeshTyingStrategySparse(const Core::FE::Discretization& scatra_manifold_dis,
        std::shared_ptr<Utils::SSIMaps> ssi_maps, bool is_manifold_meshtying);

    void apply_meshtying_to_manifold_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_matrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_matrix) override;

    void apply_meshtying_to_manifold_scatra_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_scatra_matrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_scatra_matrix) override;

    void apply_meshtying_to_manifold_structure_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_structure_matrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_structure_matrix,
        bool do_uncomplete) override;

    void apply_meshtying_to_scatra_manifold_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> ssi_scatra_manifold_matrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatra_manifold_matrix,
        bool do_uncomplete) override;
  };

  class ManifoldMeshTyingStrategyBlock : public ManifoldMeshTyingStrategyBase
  {
   public:
    explicit ManifoldMeshTyingStrategyBlock(const Core::FE::Discretization& scatra_manifold_dis,
        std::shared_ptr<SSI::Utils::SSIMaps> ssi_maps, bool is_manifold_meshtying);

    void apply_meshtying_to_manifold_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_matrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_matrix) override;

    void apply_meshtying_to_manifold_scatra_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_scatra_matrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_scatra_matrix) override;

    void apply_meshtying_to_manifold_structure_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_structure_matrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_structure_matrix,
        bool do_uncomplete) override;

    void apply_meshtying_to_scatra_manifold_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> ssi_scatra_manifold_matrix,
        std::shared_ptr<const Core::LinAlg::SparseOperator> scatra_manifold_matrix,
        bool do_uncomplete) override;

    //! handles all coupling adapters and multimap extractors with additional MultiMapExtractor
    //! splitting the slave dof map into blocks
    std::vector<meshtying_block_handler_type> mesh_tying_block_handler() const
    {
      return meshtying_block_handler_;
    }

   private:
    //! intersects @p block_map with @p intersecting_map. Extracts also dofs from @p perm_map if the
    //! corresponding dof in @p intersecting map intersects with @p block_map. Returns both in a
    //! tuple.
    std::tuple<std::shared_ptr<const Core::LinAlg::Map>, std::shared_ptr<const Core::LinAlg::Map>>
    intersect_coupling_maps_block_map(const Core::LinAlg::Map& block_map,
        const Core::LinAlg::Map& intersecting_map, const Core::LinAlg::Map& permuted_map,
        MPI_Comm comm);

    //! all interior and master dofs split into blocks
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> condensed_block_dof_map_;

    //! handles all coupling adapters and multimap extractors with additional multimap extractors
    //! that split the slave dof map into blocks
    std::vector<meshtying_block_handler_type> meshtying_block_handler_;
  };

  //! build specific mesh tying strategy
  std::shared_ptr<SSI::ManifoldMeshTyingStrategyBase> build_manifold_mesh_tying_strategy(
      const Core::FE::Discretization& scatra_manifold_dis, std::shared_ptr<Utils::SSIMaps> ssi_maps,
      bool is_manifold_meshtying, Core::LinAlg::MatrixType matrixtype_manifold);

}  // namespace SSI

FOUR_C_NAMESPACE_CLOSE

#endif

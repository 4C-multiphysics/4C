/*----------------------------------------------------------------------*/
/*! \file

\brief Fluid field adapter for fsi airway simulations with attached
parenchyma balloon

\level 2


*/
/*----------------------------------------------------------------------*/


#ifndef FOUR_C_ADAPTER_FLD_LUNG_HPP
#define FOUR_C_ADAPTER_FLD_LUNG_HPP

#include "4C_config.hpp"

#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_fem_condition.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_RCP.hpp>

#include <set>

FOUR_C_NAMESPACE_OPEN


// forward declarations

namespace Core::LinAlg
{
  class MapExtractor;
}


namespace Adapter
{
  class FluidLung : public FluidFSI
  {
   public:
    /// Constructor
    FluidLung(Teuchos::RCP<Fluid> fluid, Teuchos::RCP<Core::FE::Discretization> dis,
        Teuchos::RCP<Core::LinAlg::Solver> solver, Teuchos::RCP<Teuchos::ParameterList> params,
        Teuchos::RCP<Core::IO::DiscretizationWriter> output, bool isale, bool dirichletcond);

    /// initialize algorithm
    void init() override;

    /// List of fluid-structure volume constraints
    void list_lung_vol_cons(std::set<int>& LungVolConIDs, int& MinLungVolConID);

    /// Initialize fluid part of lung volume constraint
    void initialize_vol_con(Teuchos::RCP<Epetra_Vector> initflowrate, const int offsetID);

    /// Evaluate fluid/ale part of lung volume constraint
    void evaluate_vol_con(Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> FluidShapeDerivMatrix,
        Teuchos::RCP<Core::LinAlg::SparseMatrix> FluidConstrMatrix,
        Teuchos::RCP<Core::LinAlg::SparseMatrix> ConstrFluidMatrix,
        Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> AleConstrMatrix,
        Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> ConstrAleMatrix,
        Teuchos::RCP<Epetra_Vector> FluidRHS, Teuchos::RCP<Epetra_Vector> CurrFlowRates,
        Teuchos::RCP<Epetra_Vector> lagrMultVecRed, const int offsetID, const double dttheta);

    /// Write additional forces due to volume constraint
    void output_forces(Teuchos::RCP<Epetra_Vector> Forces);

    /// Get map extractor for fsi <-> full map
    Teuchos::RCP<Core::LinAlg::MapExtractor> fsi_interface() { return fsiinterface_; }

    /// Get map extractor for asi, other <-> full inner map
    Teuchos::RCP<Core::LinAlg::MapExtractor> inner_split() { return innersplit_; }

   private:
    /// conditions, that define the lung volume constraints
    std::vector<Core::Conditions::Condition*> constrcond_;

    /// map extractor for fsi <-> full map
    /// this is needed since otherwise "other_map" contains only dofs
    /// which are not part of a condition. however, asi dofs are of
    /// course also "inner" dofs for the fluid field.
    Teuchos::RCP<Core::LinAlg::MapExtractor> fsiinterface_;

    /// map extractor for asi, other <-> full inner map
    Teuchos::RCP<Core::LinAlg::MapExtractor> innersplit_;

    /// map extractor for outflow fsi <-> full map
    Teuchos::RCP<Core::LinAlg::MapExtractor> outflowfsiinterface_;
  };
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif

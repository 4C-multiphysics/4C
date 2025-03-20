// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSTI_ALGORITHM_HPP
#define FOUR_C_SSTI_ALGORITHM_HPP

#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"
#include "4C_linalg_vector.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Adapter
{
  class ScaTraBaseAlgorithm;
  class SSIStructureWrapper;
  class Structure;
  class StructureBaseAlgorithmNew;
}  // namespace Adapter

namespace Inpar
{
  namespace SSTI
  {
    enum class SolutionScheme;
  }
}  // namespace Inpar

namespace Core::LinAlg
{
  class MultiMapExtractor;
}

namespace ScaTra
{
  class MeshtyingStrategyS2I;
  class ScaTraTimIntImpl;
}  // namespace ScaTra

namespace SSI
{
  namespace Utils
  {
    class SSIMeshTying;
  }
}  // namespace SSI

namespace SSTI
{
  //! Base class of all solid-scatra algorithms
  class SSTIAlgorithm : public Adapter::AlgorithmBase
  {
   public:
    explicit SSTIAlgorithm(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);

    //! Setup of algorithm
    //! Clone Discretizations, init and setup subproblems, setup coupling adapters at interfaces,
    //! setup submatrices for coupling between fields
    //@{
    virtual void init(MPI_Comm comm, const Teuchos::ParameterList& sstitimeparams,
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& thermoparams,
        const Teuchos::ParameterList& structparams) = 0;
    virtual void setup();
    virtual void setup_system() = 0;
    //@}

    /*! @brief Perform all necessary tasks after setting up the SSTI
     * algorithm. Currently, this only calls the post_setup routine of the
     * structural field.
     *
     */
    void post_setup();

    //! increment the counter for Newton-Raphson iterations (monolithic algorithm)
    void increment_iter() { ++iter_; }

    //! return the counter for Newton-Raphson iterations (monolithic algorithm)
    unsigned int iter() const { return iter_; }

    //! reset the counter for Newton-Raphson iterations (monolithic algorithm)
    void reset_iter() { iter_ = 0; }

    //! return coupling
    //@{
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_scatra() const
    {
      return meshtying_strategy_scatra_;
    }
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_thermo() const
    {
      return meshtying_strategy_thermo_;
    }
    std::shared_ptr<const SSI::Utils::SSIMeshTying> ssti_structure_mesh_tying() const
    {
      return ssti_structure_meshtying_;
    }
    //@}

    //! return subproblems
    //@{
    std::shared_ptr<Adapter::SSIStructureWrapper> structure_field() const { return structure_; };
    std::shared_ptr<ScaTra::ScaTraTimIntImpl> scatra_field() const;
    std::shared_ptr<ScaTra::ScaTraTimIntImpl> thermo_field() const;
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra_field_base() { return scatra_; };
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> thermo_field_base() { return thermo_; };
    //@}

    //! get bool indicating if we have at least one ssi interface meshtying condition
    bool interface_meshtying() const { return interfacemeshtying_; };

    //! read restart
    void read_restart(int restart) override;

    //! timeloop of coupled problem
    virtual void timeloop() = 0;

    //! test results (if necessary)
    virtual void test_results(MPI_Comm comm) const;

   protected:
    //! clone scatra from structure and then thermo from scatra
    virtual void clone_discretizations(MPI_Comm comm);

    //! copies modified time step from scatra to structure and to this SSI algorithm
    void distribute_dt_from_scatra();

    //! distribute states between subproblems
    //@{
    void distribute_solution_all_fields();
    void distribute_scatra_solution();
    void distribute_structure_solution();
    void distribute_thermo_solution();
    //@}

   private:
    //! counter for Newton-Raphson iterations (monolithic algorithm)
    unsigned int iter_;

    //! exchange materials between discretizations
    void assign_material_pointers();

    void check_is_init();

    //! clone thermo parameters from scatra parameters and adjust where needed
    Teuchos::ParameterList clone_thermo_params(
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& thermoparams);

    //! Pointers to subproblems
    //@{
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra_;
    std::shared_ptr<Adapter::SSIStructureWrapper> structure_;
    std::shared_ptr<Adapter::StructureBaseAlgorithmNew> struct_adapterbase_ptr_;
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> thermo_;
    //@}

    //! Pointers to coupling strategies
    //@{
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_scatra_;
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_thermo_;
    std::shared_ptr<SSI::Utils::SSIMeshTying> ssti_structure_meshtying_;
    //@}

    //! bool indicating if we have at least one ssi interface meshtying condition
    const bool interfacemeshtying_;

    //! flag indicating if class is initialized
    bool isinit_;

    //! flag indicating if class is setup
    bool issetup_;
  };  // SSTI_Algorithm

  //! Construct specific SSTI algorithm
  std::shared_ptr<SSTI::SSTIAlgorithm> build_ssti(Inpar::SSTI::SolutionScheme coupling,
      MPI_Comm comm, const Teuchos::ParameterList& sstiparams);
}  // namespace SSTI
FOUR_C_NAMESPACE_CLOSE

#endif

/*----------------------------------------------------------------------*/
/*! \file

\brief general coupling algorithm for scatra-thermo interaction

\level 2

*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_STI_ALGORITHM_HPP
#define FOUR_C_STI_ALGORITHM_HPP

#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"
#include "4C_adapter_scatra_base_algorithm.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class Vector;
}

namespace Adapter
{
  class ScaTraBaseAlgorithm;
}  // namespace Adapter

namespace ScaTra
{
  class MeshtyingStrategyS2I;
  class ScaTraTimIntImpl;
}  // namespace ScaTra

namespace STI
{
  //! monolithic algorithm for scatra-thermo interaction
  class Algorithm : public Adapter::AlgorithmBase
  {
   public:
    //! return counter for Newton-Raphson iterations (monolithic algorithm) or outer coupling
    //! iterations (partitioned algorithm)
    const unsigned& iter() const { return iter_; };

    //! read restart data
    void read_restart(int step  //! time step for restart
        ) override;

    //! access scatra time integrator
    Teuchos::RCP<ScaTra::ScaTraTimIntImpl> scatra_field() const { return scatra_->scatra_field(); };

    //! access thermo time integrator
    Teuchos::RCP<ScaTra::ScaTraTimIntImpl> thermo_field() const { return thermo_->scatra_field(); };

    //! time loop
    void time_loop();

   protected:
    //! constructor
    explicit Algorithm(const Epetra_Comm& comm,  //! communicator
        const Teuchos::ParameterList& stidyn,    //! parameter list for scatra-thermo interaction
        const Teuchos::ParameterList&
            scatradyn,  //! scalar transport parameter list for scatra and thermo fields
        const Teuchos::ParameterList&
            solverparams_scatra,  //! solver parameter list for scatra field
        const Teuchos::ParameterList&
            solverparams_thermo  //! solver parameter list for thermo field
    );

    //! prepare time step
    void prepare_time_step() override;

    //! pass scatra degrees of freedom to thermo discretization
    void transfer_scatra_to_thermo(
        const Teuchos::RCP<const Core::LinAlg::Vector> scatra  //!< scatra state vector
    ) const;

    //! pass thermo degrees of freedom to scatra discretization
    void transfer_thermo_to_scatra(
        const Teuchos::RCP<const Core::LinAlg::Vector> thermo  //!< thermo state vector
    ) const;

    //! scatra time integrator
    Teuchos::RCP<Adapter::ScaTraBaseAlgorithm> scatra_;

    //! thermo time integrator
    Teuchos::RCP<Adapter::ScaTraBaseAlgorithm> thermo_;

    //! meshtying strategy for scatra-scatra interface coupling on scatra discretization
    Teuchos::RCP<ScaTra::MeshtyingStrategyS2I> strategyscatra_;

    //! meshtying strategy for scatra-scatra interface coupling on thermo discretization
    Teuchos::RCP<ScaTra::MeshtyingStrategyS2I> strategythermo_;

    //! input parameters for scatra and thermo fields
    Teuchos::RCP<Teuchos::ParameterList> fieldparameters_;

    //! counter for Newton-Raphson iterations (monolithic algorithm) or outer coupling iterations
    //! (partitioned algorithm)
    unsigned int iter_;

    //! maximum number of Newton-Raphson iterations (monolithic algorithm) or outer coupling
    //! iterations (partitioned algorithm)
    unsigned int itermax_;

    //! tolerance for Newton-Raphson iteration (monolithic algorithm) or outer coupling iteration
    //! (partitioned algorithm)
    double itertol_;

    //! input parameters for scatra-thermo interaction
    Teuchos::RCP<Teuchos::ParameterList> stiparameters_;

    //! timer for Newton-Raphson iteration (monolithic algorithm) or outer coupling iteration
    //! (partitioned algorithm)
    Teuchos::RCP<Teuchos::Time> timer_;

   private:
    //! modify field parameters for thermo field
    void modify_field_parameters_for_thermo_field();

    //! output solution to screen and files
    void output() override;

    //! evaluate time step using Newton-Raphson iteration (monolithic algorithm) or outer coupling
    //! iteration (partitioned algorithm)
    virtual void solve() = 0;

    //! update scatra and thermo fields after time step evaluation
    void update() override;
  };  // class Algorithm : public Adapter::AlgorithmBase
}  // namespace STI
FOUR_C_NAMESPACE_CLOSE

#endif

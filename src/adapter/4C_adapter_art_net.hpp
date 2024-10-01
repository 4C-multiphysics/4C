/*----------------------------------------------------------------------*/
/*! \file
 \brief adapter for artery network

   \level 3

 *----------------------------------------------------------------------*/

#ifndef FOUR_C_ADAPTER_ART_NET_HPP
#define FOUR_C_ADAPTER_ART_NET_HPP

#include "4C_config.hpp"

#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"
#include "4C_utils_result_test.hpp"

#include <Epetra_Map.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Adapter
{
  /// basic artery network adapter
  class ArtNet
  {
   public:
    /// constructor
    ArtNet(){};

    /// virtual destructor to support polymorph destruction
    virtual ~ArtNet() = default;

    /// initialization
    virtual void init(const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& arteryparams, const std::string& scatra_disname) = 0;

    /// initialization
    virtual void init_save_state() = 0;

    // restart
    virtual void read_restart(int step, bool CoupledTo3D = false) = 0;

    // time integration
    virtual void integrate(
        bool CoupledTo3D, Teuchos::RCP<Teuchos::ParameterList> CouplingParams) = 0;

    // test results
    virtual void test_results() = 0;

    // create field test
    virtual Teuchos::RCP<Core::UTILS::ResultTest> create_field_test() = 0;

    //! get discretization
    virtual Teuchos::RCP<Core::FE::Discretization> discretization() = 0;

    // get time step size
    virtual double dt() const = 0;

    // output
    virtual void output(bool CoupledTo3D, Teuchos::RCP<Teuchos::ParameterList> CouplingParams) = 0;

    // update of variables from n --> n+1
    virtual void time_update() = 0;

    // save state
    virtual void save_state() = 0;

    // load state
    virtual void load_state() = 0;

    // prepare the loop
    virtual void prepare_time_loop() = 0;

    // prepare step
    virtual void prepare_time_step() = 0;

    // solve
    virtual void solve(Teuchos::RCP<Teuchos::ParameterList> CouplingTo3DParams) = 0;

    virtual void assemble_mat_and_rhs() = 0;

    virtual void prepare_linear_solve() = 0;

    /// direct access to system matrix
    virtual Teuchos::RCP<Core::LinAlg::SparseMatrix> system_matrix() = 0;

    //! right-hand side alias the dynamic force residual
    virtual Teuchos::RCP<const Core::LinAlg::Vector<double>> rhs() const = 0;

    //! return pressure field at time n+1
    virtual Teuchos::RCP<const Core::LinAlg::Vector<double>> pressurenp() const = 0;

    //! iterative update of primary variable
    virtual void update_iter(const Teuchos::RCP<const Core::LinAlg::Vector<double>> inc) = 0;

    // solve scalar transport in arteries
    virtual void solve_scatra() = 0;

    // set solve scalar transport-flag
    virtual void set_solve_scatra(const bool solvescatra) = 0;

    //! Return MapExtractor for Dirichlet boundary conditions
    virtual Teuchos::RCP<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() const = 0;

  };  // class ArtNet

}  // namespace Adapter


FOUR_C_NAMESPACE_CLOSE

#endif

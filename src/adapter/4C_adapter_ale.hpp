// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_ALE_HPP
#define FOUR_C_ADAPTER_ALE_HPP

#include "4C_config.hpp"

#include "4C_ale_input.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"
#include "4C_utils_result_test.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::IO
{
  class DiscretizationWriter;
}

namespace Core::LinAlg
{
  class Solver;
  class SparseMatrix;
  class BlockSparseMatrixBase;
  class MapExtractor;
  class MultiMapExtractor;
}  // namespace Core::LinAlg

namespace Core::Conditions
{
  class LocsysManager;
}

namespace TimeInt
{
  template <typename>
  class TimIntMStep;
}

namespace Global
{
  class Problem;
}  // namespace Global


namespace Adapter
{
  /*! \brief General ALE field interface
   *
   *  Base class for ALE field implementations. A pure ALE problem just needs the
   *  simple ALE time integrator ALE::Ale whereas coupled problems need wrap
   *  the ALE field in an ALE adapter that provides problem specific ALE
   *  functionalities.
   *
   *  \sa ALE::Ale
   *  \sa Adapter::Structure, Adapter::Fluid
   *
   */
  class Ale
  {
   public:
    //! virtual to get polymorph destruction
    virtual ~Ale() = default;

    //! @name Vector access

    //! initial guess of Newton's method
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() const = 0;

    //! rhs of Newton's method
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() const = 0;

    //! unknown displacements at \f$t_{n+1}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp() const = 0;

    //! known displacements at \f$t_{n}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> dispn() const = 0;

    //@}

    //! @name Misc

    //! dof map of vector of unknowns
    virtual std::shared_ptr<const Core::LinAlg::Map> dof_row_map() const = 0;

    //! direct access to system matrix
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() = 0;

    //! direct access to system matrix
    virtual std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() = 0;

    // access to locsys manager
    virtual std::shared_ptr<Core::Conditions::LocsysManager> locsys_manager() = 0;

    //! direct access to discretization
    virtual std::shared_ptr<const Core::FE::Discretization> discretization() const = 0;

    /// writing access to discretization
    virtual std::shared_ptr<Core::FE::Discretization> write_access_discretization() = 0;

    //! Return MapExtractor for Dirichlet boundary conditions
    virtual std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor(
        ALE::Utils::MapExtractor::AleDBCSetType dbc_type  ///< type of dbc set
        ) = 0;

    //@}

    /// setup Dirichlet boundary condition map extractor
    virtual void setup_dbc_map_ex(ALE::Utils::MapExtractor::AleDBCSetType
                                      dbc_type,  //!< application-specific type of Dirichlet set
        std::shared_ptr<const ALE::Utils::MapExtractor>
            interface,  //!< interface for creation of additional, application-specific Dirichlet
                        //!< map extractors
        std::shared_ptr<const ALE::Utils::XFluidFluidMapExtractor>
            xff_interface  //!< interface for creation of a Dirichlet map extractor, tailored to
                           //!< XFFSI
        ) = 0;

    //! @name Time step helpers
    //@{

    virtual void reset_time(const double dtold) = 0;

    //! Return target time \f$t_{n+1}\f$
    virtual double time() const = 0;

    //! Return target step counter \f$step_{n+1}\f$
    virtual double step() const = 0;

    //! Evaluate time step
    virtual void time_step(ALE::Utils::MapExtractor::AleDBCSetType dbc_type =
                               ALE::Utils::MapExtractor::dbc_set_std) = 0;

    //! Get time step size \f$\Delta t_n\f$
    virtual double dt() const = 0;

    //! Take the time and integrate (time loop)
    virtual int integrate() = 0;

    //! start new time step
    virtual void prepare_time_step() = 0;

    //! set time step size
    virtual void set_dt(const double dtnew) = 0;

    //! Set time and step
    virtual void set_time_step(const double time, const int step) = 0;

    /*! \brief update displacement and evaluate elements
     *
     *  We use a step increment such that the update reads
     *  \f$x^n+1_i+1 = x^n + disstepinc\f$
     *
     *  with \f$n\f$ and \f$i\f$ being time and Newton iteration step
     *
     *  Note: The ALE expects an iteration increment.
     *  In case the StructureNOXCorrectionWrapper is applied, the step increment
     *  is expected which is then transformed into an iteration increment
     */
    virtual void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>>
                              disiterinc,  ///< step increment such that \f$ x_{n+1}^{k+1} =
                                           ///< x_{n}^{converged}+ stepinc \f$
        ALE::Utils::MapExtractor::AleDBCSetType
            dbc_type  ///< application-specific type of Dirichlet set
        ) = 0;

    //! iterative update of solution after solving the linear system
    virtual void update_iter() = 0;

    //! update at time step end
    virtual void update() = 0;

    //! output results
    virtual void output() = 0;

    //! read restart information for given time step
    virtual void read_restart(const int step) = 0;

    /*! \brief Reset time step
     *
     *  In case of time step size adaptivity, time steps might have to be
     *  repeated. Therefore, we need to reset the solution back to the initial
     *  solution of the time step.
     *
     */
    virtual void reset_step() = 0;

    //@}

    //! @name Solver calls

    /*!
     \brief nonlinear solve

     Do the nonlinear solve, i.e. (multiple) corrector,
     for the time step. All boundary conditions have
     been set.
     */
    virtual int solve() = 0;

    //! Access to linear solver
    virtual std::shared_ptr<Core::LinAlg::Solver> linear_solver() = 0;

    //@}

    //! @name Write access to field solution variables at \f$t^{n+1}\f$
    //@{

    //! write access to extract displacements at \f$t^{n+1}\f$
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> write_access_dispnp() const = 0;

    //@}

    //! create result test for encapsulated structure algorithm
    virtual std::shared_ptr<Core::Utils::ResultTest> create_field_test() = 0;

    //! reset state vectors to zero
    virtual void reset() = 0;

    /*! \brief Create System matrix
     *
     * We allocate the Core::LINALG object just once, the result is an empty Core::LINALG
     * object. Evaluate has to be called separately.
     */
    virtual void create_system_matrix(
        std::shared_ptr<const ALE::Utils::MapExtractor> interface = nullptr) = 0;

    //! update source dofs for multifield simulations with ale mesh tying
    virtual void update_source_dof(std::shared_ptr<Core::LinAlg::Vector<double>>& a) = 0;

  };  // class Ale

  //! Base class of algorithms that use an ale field
  class AleBaseAlgorithm
  {
   public:
    //! constructor
    explicit AleBaseAlgorithm(Global::Problem& problem,
        const Teuchos::ParameterList& prbdyn,             ///< the problem's parameter list
        std::shared_ptr<Core::FE::Discretization> actdis  ///< pointer to discretization
    );

    //! virtual destructor to support polymorph destruction
    virtual ~AleBaseAlgorithm() = default;

    //! std::shared_ptr version of ale field solver
    std::shared_ptr<Ale> ale_field() { return ale_; }

   private:
    /*! \brief Setup ALE algorithm
     *
     *  Setup the ALE algorithm. We allow for overriding some parameters with
     *  values specified in given problem-dependent ParameterList.
     */
    void setup_ale(const Teuchos::ParameterList& prbdyn,  ///< the problem's parameter list
        std::shared_ptr<Core::FE::Discretization> actdis  ///< pointer to discretization
    );

    //! explicit global problem context
    Global::Problem& problem_;

    //! ALE field solver
    std::shared_ptr<Ale> ale_;

  };  // class AleBaseAlgorithm

}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif

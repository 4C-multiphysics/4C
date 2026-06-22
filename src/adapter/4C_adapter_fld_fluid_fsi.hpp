// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_FLD_FLUID_FSI_HPP
#define FOUR_C_ADAPTER_FLD_FLUID_FSI_HPP

#include "4C_config.hpp"

#include "4C_adapter_fld_wrapper.hpp"
#include "4C_fsi_input.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class Solver;
  class MapExtractor;
}  // namespace Core::LinAlg

namespace Core::IO
{
  class DiscretizationWriter;
}

namespace FLD
{
  class FluidImplicitTimeInt;
  namespace Utils
  {
    class MapExtractor;
  }
}  // namespace FLD

namespace Global
{
  class Problem;
}  // namespace Global

namespace Adapter
{
  /*! \brief Fluid field adapter for fsi
   *
   *
   *  Can only be used in conjunction with #FLD::FluidImplicitTimeInt
   */
  class FluidFSI : public FluidWrapper
  {
   public:
    /// Constructor with explicit global problem context
    FluidFSI(Global::Problem& problem, std::shared_ptr<Fluid> fluid,
        std::shared_ptr<Core::FE::Discretization> dis, std::shared_ptr<Core::LinAlg::Solver> solver,
        std::shared_ptr<Teuchos::ParameterList> params,
        std::shared_ptr<Core::IO::DiscretizationWriter> output, bool isale, bool dirichletcond);

    /// initialize algorithm
    void init() override;

    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() override;

    std::shared_ptr<const Core::LinAlg::Map> dof_row_map(unsigned nds) override;

    /// Velocity-displacement conversion at the fsi interface
    double time_scaling() const override;

    /// take current results for converged and save for next time step
    void update() override;

    /// get the linear solver object used for this field
    std::shared_ptr<Core::LinAlg::Solver> linear_solver() override;

    std::shared_ptr<Core::LinAlg::Vector<double>> relaxation_solve(
        std::shared_ptr<Core::LinAlg::Vector<double>> ivel) override;

    /// communication object at the interface
    std::shared_ptr<FLD::Utils::MapExtractor> const& interface() const override
    {
      return interface_;
    }

    /// update source dofs for multifield simulations with fluid mesh tying
    virtual void update_source_dof(std::shared_ptr<Core::LinAlg::Vector<double>>& f);

    std::shared_ptr<const Core::LinAlg::Map> inner_velocity_row_map() override;

    std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_forces() override;

    /// Return interface velocity at new time level n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_velnp() override;

    /// Return interface velocity at old time level n
    std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_veln() override;

    void apply_interface_velocities(std::shared_ptr<Core::LinAlg::Vector<double>> ivel) override;

    /// Apply initial mesh displacement
    void apply_initial_mesh_displacement(
        std::shared_ptr<const Core::LinAlg::Vector<double>> initfluiddisp) override;

    void apply_mesh_displacement(
        std::shared_ptr<const Core::LinAlg::Vector<double>> fluiddisp) override;

    /// Update fluid griv velocity via FD approximation
    void update_gridv();

    void apply_mesh_displacement_increment(
        std::shared_ptr<const Core::LinAlg::Vector<double>> dispstepinc) override
    {
      FOUR_C_THROW("not implemented!");
    };

    void apply_mesh_velocity(std::shared_ptr<const Core::LinAlg::Vector<double>> gridvel) override;

    void set_mesh_map(
        std::shared_ptr<const Core::LinAlg::Map> mm, const int target_dofset_number = 0) override;

    //! @name Conversion between displacement and velocity at interface

    //! Conversion of displacement to velocity at the interface without predictors or inhomogeneous
    //! DBCs
    //!
    //! All input vectors have to live on the fluid field map.
    void displacement_to_velocity(std::shared_ptr<Core::LinAlg::Vector<double>>
            fcx  ///< interface displacement step increment
        ) override;

    //! Conversion of velocity to displacement at the interface without predictors or inhomogeneous
    //! DBCs
    //!
    //! All input vectors have to live on the fluid field map.
    void velocity_to_displacement(std::shared_ptr<Core::LinAlg::Vector<double>>
            fcx  ///< interface velocity step increment at interface
        ) override;

    //@}

    std::shared_ptr<Core::LinAlg::Vector<double>> integrate_interface_shape() override;

    void use_block_matrix(bool splitmatrix) override;

    /*! \brief Project the velocity field into a divergence free subspace
     *
     *  Project the velocity field into a divergence free subspace
     *  while interface and Dirichlet DOFS are not affected.
     *  The projection is done by the following operation:
     *
     *  \$f v_{divfree} = (I - B(B^TB)^{-1}B^T)) v + B(B^TB)^{-1} R\$f
     *
     *  The vector \$f R \$f ensures that interface and Dirichlet DOFs are not modified.
     *
     */
    void proj_vel_to_div_zero();

    /// reset state vectors
    void reset(int numsteps = 1, int iter = -1) override;

    /// calculate error in comparison to analytical solution
    void calculate_error() override;

    //! @name Time step size adaptivity in monolithic FSI
    //@{

    /*! \brief Do one step with auxiliary time integration scheme
     *
     *  Do a single time step with the user given auxiliary time integration
     *  scheme. Result is stored in #locerrvelnp_ and is used later to estimate
     *  the local discretization error of the marching time integration scheme.
     *
     */
    void time_step_auxiliary() override;

    /*! \brief Indicate norms of temporal discretization error
     *
     */
    void indicate_error_norms(
        double& err,       ///< L2-norm of temporal discretization error based on all DOFs
        double& errcond,   ///< L2-norm of temporal discretization error based on interface DOFs
        double& errother,  ///< L2-norm of temporal discretization error based on interior DOFs
        double& errinf,    ///< L-inf-norm of temporal discretization error based on all DOFs
        double&
            errinfcond,  ///< L-inf-norm of temporal discretization error based on interface DOFs
        double& errinfother  ///< L-inf-norm of temporal discretization error based on interior DOFs
        ) override;

    /*! \brief Error order for adaptive fluid time integration
     *
     */
    double get_tim_ada_err_order() const;

    /*! \brief Name of auxiliary time integrator
     *
     */
    std::string get_tim_ada_method_name() const;

    //! Type of adaptivity algorithm
    enum ETimAdaAux
    {
      ada_none,      ///< no time step size adaptivity
      ada_upward,    ///< of upward type, i.e. auxiliary scheme has \b higher order of accuracy than
                     ///< marching scheme
      ada_downward,  ///< of downward type, i.e. auxiliary scheme has \b lower order of accuracy
                     ///< than marching scheme
      ada_orderequal  ///< of equal order type, i.e. auxiliary scheme has the \b same order of
                      ///< accuracy like the marching method
    };

    //@}

    /// Calculate WSS vector
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> calculate_wall_shear_stresses();

   protected:
    /// create conditioned dof-map extractor for the fluid
    virtual void setup_interface(const int target_dofset_number = 0);

    /*! \brief Build inner velocity map
     *
     *  Only DOFs in the interior of the fluid domain are considered. DOFs at
     *  the interface are excluded.
     *
     *  We use only velocity DOFs and only those without Dirichlet constraint.
     */
    void build_inner_vel_map();

    /// A casted pointer to the fluid itself
    std::shared_ptr<FLD::FluidImplicitTimeInt> fluidimpl_;

    //! @name local copies of input parameters
    Global::Problem& problem_;
    std::shared_ptr<Core::FE::Discretization> dis_;
    std::shared_ptr<Teuchos::ParameterList> params_;
    std::shared_ptr<Core::IO::DiscretizationWriter> output_;
    bool dirichletcond_;
    //@}

    //! \brief interface map setup for fsi interface, interior translation
    //!
    //! Note: full map contains velocity AND pressure DOFs
    std::shared_ptr<FLD::Utils::MapExtractor> interface_;

    /// interface force at old time level t_n
    std::shared_ptr<Core::LinAlg::Vector<double>> interfaceforcen_;

    /// ALE dof map
    std::shared_ptr<Core::LinAlg::MapExtractor> meshmap_;

    /// all velocity dofs not at the interface
    std::shared_ptr<Core::LinAlg::Map> innervelmap_;

   private:
    //! Time step size adaptivity in monolithic FSI
    //@{

    /*! \brief Do a single explicit Euler step as auxiliary time integrator
     *
     *  Based on state vector \f$x_n\f$ and its time derivative \f$\dot{x}_{n}\f$
     *  at time \f$t_{n}\f$, we calculate \f$x_{n+1}\f$ using an explicit Euler
     *  step:
     *
     *  \f[
     *    x_{n+1} = x_{n} + \Delta t_{n} \dot{x}_{n}
     *  \f]
     *
     */
    void explicit_euler(const Core::LinAlg::Vector<double>& veln,  ///< velocity at \f$t_n\f$
        const Core::LinAlg::Vector<double>& accn,                  ///< acceleration at \f$t_n\f$
        Core::LinAlg::Vector<double>& velnp                        ///< velocity at \f$t_{n+1}\f$
    ) const;

    /*! \brief Do a single Adams-Bashforth 2 step as auxiliary time integrator
     *
     *  Based on state vector \f$x_n\f$ and its time derivatives \f$\dot{x}_{n}\f$
     *  and \f$\dot{x}_{n-1}\f$ at time steps \f$t_{n}\f$ and \f$t_{n-1}\f$,
     *  respectively, we calculate \f$x_{n+1}\f$ using an Adams-Bashforth 2 step
     *  with varying time step sizes:
     *
     *  \f[
     *    x_{n+1} = x_{n}
     *            + \frac{2\Delta t_{n} \Delta t_{n-1} + \Delta t_{n}^2}
     *                   {2\Delta t_{n-1}} \dot{x}_{n}
     *            - \frac{\Delta t_{n}^2}{2\Delta t_{n-1}} \dot{x}_{n-1}
     *  \f]
     *
     *  <h3> References: </h3>
     *  <ul>
     *  <li> Wall WA: Fluid-Struktur-Interaktion mit stabilisierten Finiten
     *       Elementen, PhD Thesis, Universitaet Stuttgart, 1999 </li>
     *  <li> Bornemann B: Time Integration Algorithms for the Steady States of
     *       Dissipative Non-Linear Dynamic Systems, PhD Thesis, Imperial
     *       College London, 2003 </li>
     *  <li> Gresho PM, Griffiths DF, Silvester DJ: Adaptive Time-Stepping for
     *       Incompressible Flow Part I: Scalar Advection-Diffusion,
     *       SIAM J. Sci. Comput. (30), pp. 2018-2054, 2008 </li>
     *  <li> Kay DA, Gresho PM, Griffiths DF, Silvester DJ: Adaptive Time
     *       Stepping for Incompressible Flow Part II: Navier-Stokes Equations,
     *       SIAM J. Sci. Comput. (32), pp. 111-128, 2010 </li>
     *
     */
    void adams_bashforth2(const Core::LinAlg::Vector<double>& veln,  ///< velocity at \f$t_n\f$
        const Core::LinAlg::Vector<double>& accn,                    ///< acceleration at \f$t_n\f$
        const Core::LinAlg::Vector<double>& accnm,  ///< acceleration at \f$t_{n-1}\f$
        Core::LinAlg::Vector<double>& velnp         ///< velocity at \f$t_{n+1}\f$
    ) const;

    //! Compute length-scaled L2-norm of a vector
    virtual double calculate_error_norm(
        const Core::LinAlg::Vector<double>& vec,  ///< vector to compute norm of
        const int numneglect = 0  ///< number of DOFs that have to be neglected for length scaling
    ) const;

    //! return order of accuracy of auxiliary integrator
    int aux_method_order_of_accuracy() const;

    //! return leading error coefficient of velocity of auxiliary integrator
    double aux_method_lin_err_coeff_vel() const;

    std::shared_ptr<Core::LinAlg::Vector<double>>
        locerrvelnp_;  ///< vector of temporal local discretization error

    FSI::FluidMethod auxintegrator_;  ///< auxiliary time integrator in fluid field

    int numfsidbcdofs_;  ///< number of interface DOFs with Dirichlet boundary condition

    ETimAdaAux methodadapt_;  ///< type of adaptive method

    //@}
  };
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif

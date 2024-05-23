/*----------------------------------------------------------------------*/
/*! \file

\brief Fluid field adapter for fsi. Can only be used in conjunction with FLD::FluidImplicitTimeInt

\level 2

*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_ADAPTER_FLD_FLUID_FSI_HPP
#define FOUR_C_ADAPTER_FLD_FLUID_FSI_HPP

#include "4C_config.hpp"

#include "4C_adapter_fld_wrapper.hpp"
#include "4C_inpar_fsi.hpp"

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace CORE::LINALG
{
  class Solver;
  class MapExtractor;
}  // namespace CORE::LINALG

namespace IO
{
  class DiscretizationWriter;
}

namespace FLD
{
  class FluidImplicitTimeInt;
  namespace UTILS
  {
    class MapExtractor;
  }
}  // namespace FLD

namespace ADAPTER
{
  /*! \brief Fluid field adapter for fsi
   *
   *
   *  Can only be used in conjunction with #FLD::FluidImplicitTimeInt
   */
  class FluidFSI : public FluidWrapper
  {
   public:
    /// Constructor
    FluidFSI(Teuchos::RCP<Fluid> fluid, Teuchos::RCP<DRT::Discretization> dis,
        Teuchos::RCP<CORE::LINALG::Solver> solver, Teuchos::RCP<Teuchos::ParameterList> params,
        Teuchos::RCP<IO::DiscretizationWriter> output, bool isale, bool dirichletcond);

    /// initialize algorithm
    void Init() override;

    Teuchos::RCP<const Epetra_Map> DofRowMap() override;

    Teuchos::RCP<const Epetra_Map> DofRowMap(unsigned nds) override;

    /// Velocity-displacement conversion at the fsi interface
    double TimeScaling() const override;

    /// take current results for converged and save for next time step
    void Update() override;

    /// get the linear solver object used for this field
    Teuchos::RCP<CORE::LINALG::Solver> LinearSolver() override;

    Teuchos::RCP<Epetra_Vector> RelaxationSolve(Teuchos::RCP<Epetra_Vector> ivel) override;

    /// communication object at the interface
    Teuchos::RCP<FLD::UTILS::MapExtractor> const& Interface() const override { return interface_; }

    /// update slave dofs for multifield simulations with fluid mesh tying
    virtual void UpdateSlaveDOF(Teuchos::RCP<Epetra_Vector>& f);

    Teuchos::RCP<const Epetra_Map> InnerVelocityRowMap() override;

    Teuchos::RCP<Epetra_Vector> extract_interface_forces() override;

    /// Return interface velocity at new time level n+1
    Teuchos::RCP<Epetra_Vector> extract_interface_velnp() override;

    /// Return interface velocity at old time level n
    Teuchos::RCP<Epetra_Vector> extract_interface_veln() override;

    Teuchos::RCP<Epetra_Vector> extract_free_surface_veln() override;

    void apply_interface_velocities(Teuchos::RCP<Epetra_Vector> ivel) override;

    /// Apply initial mesh displacement
    void apply_initial_mesh_displacement(Teuchos::RCP<const Epetra_Vector> initfluiddisp) override;

    void apply_mesh_displacement(Teuchos::RCP<const Epetra_Vector> fluiddisp) override;

    /// Update fluid griv velocity via FD approximation
    void UpdateGridv();

    void apply_mesh_displacement_increment(Teuchos::RCP<const Epetra_Vector> dispstepinc) override
    {
      FOUR_C_THROW("not implemented!");
    };

    void ApplyMeshVelocity(Teuchos::RCP<const Epetra_Vector> gridvel) override;

    void SetMeshMap(Teuchos::RCP<const Epetra_Map> mm, const int nds_master = 0) override;

    //! @name Conversion between displacement and velocity at interface

    //! Conversion of displacement to velocity at the interface without predictors or inhomogeneous
    //! DBCs
    //!
    //! All input vectors have to live on the fluid field map.
    void displacement_to_velocity(
        Teuchos::RCP<Epetra_Vector> fcx  ///< interface displacement step increment
        ) override;

    //! Conversion of velocity to displacement at the interface without predictors or inhomogeneous
    //! DBCs
    //!
    //! All input vectors have to live on the fluid field map.
    void velocity_to_displacement(
        Teuchos::RCP<Epetra_Vector> fcx  ///< interface velocity step increment at interface
        ) override;

    void free_surf_displacement_to_velocity(Teuchos::RCP<Epetra_Vector> fcx) override;

    void free_surf_velocity_to_displacement(Teuchos::RCP<Epetra_Vector> fcx) override;

    //@}

    Teuchos::RCP<Epetra_Vector> integrate_interface_shape() override;

    void UseBlockMatrix(bool splitmatrix) override;

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
     *  \author mayr.mt \date  06/2012
     */
    void ProjVelToDivZero();

    /// reset state vectors
    void Reset(bool completeReset = false, int numsteps = 1, int iter = -1) override;

    /// calculate error in comparison to analytical solution
    void CalculateError() override;

    //! @name Time step size adaptivity in monolithic FSI
    //@{

    /*! \brief Do one step with auxiliary time integration scheme
     *
     *  Do a single time step with the user given auxiliary time integration
     *  scheme. Result is stored in #locerrvelnp_ and is used later to estimate
     *  the local discretization error of the marching time integration scheme.
     *
     *  \author mayr.mt \date 12/2013
     */
    void TimeStepAuxiliar() override;

    /*! \brief Indicate norms of temporal discretization error
     *
     *  \author mayr.mt \date 12/2013
     */
    void IndicateErrorNorms(
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
     *  \author mayr.mt \date 04/2015
     */
    double GetTimAdaErrOrder() const;

    /*! \brief Name of auxiliary time integrator
     *
     *  \author mayr.mt \date 04/2015
     */
    std::string GetTimAdaMethodName() const;

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
    virtual Teuchos::RCP<Epetra_Vector> calculate_wall_shear_stresses();

   protected:
    /// create conditioned dof-map extractor for the fluid
    virtual void SetupInterface(const int nds_master = 0);

    /*! \brief Build inner velocity map
     *
     *  Only DOFs in the interior of the fluid domain are considered. DOFs at
     *  the interface are excluded.
     *
     *  We use only velocity DOFs and only those without Dirichlet constraint.
     */
    void BuildInnerVelMap();

    /// A casted pointer to the fluid itself
    Teuchos::RCP<FLD::FluidImplicitTimeInt> fluidimpl_;

    //! @name local copies of input parameters
    Teuchos::RCP<DRT::Discretization> dis_;
    Teuchos::RCP<Teuchos::ParameterList> params_;
    Teuchos::RCP<IO::DiscretizationWriter> output_;
    bool dirichletcond_;
    //@}

    //! \brief interface map setup for fsi interface, free surface, interior translation
    //!
    //! Note: full map contains velocity AND pressure DOFs
    Teuchos::RCP<FLD::UTILS::MapExtractor> interface_;

    /// interface force at old time level t_n
    Teuchos::RCP<Epetra_Vector> interfaceforcen_;

    /// ALE dof map
    Teuchos::RCP<CORE::LINALG::MapExtractor> meshmap_;

    /// all velocity dofs not at the interface
    Teuchos::RCP<Epetra_Map> innervelmap_;

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
     *  \author mayr.mt \date 10/2013
     */
    void ExplicitEuler(const Epetra_Vector& veln,  ///< velocity at \f$t_n\f$
        const Epetra_Vector& accn,                 ///< acceleration at \f$t_n\f$
        Epetra_Vector& velnp                       ///< velocity at \f$t_{n+1}\f$
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
     *  \author mayr.mt \date 11/2013
     */
    void AdamsBashforth2(const Epetra_Vector& veln,  ///< velocity at \f$t_n\f$
        const Epetra_Vector& accn,                   ///< acceleration at \f$t_n\f$
        const Epetra_Vector& accnm,                  ///< acceleration at \f$t_{n-1}\f$
        Epetra_Vector& velnp                         ///< velocity at \f$t_{n+1}\f$
    ) const;

    //! Compute length-scaled L2-norm of a vector
    virtual double CalculateErrorNorm(const Epetra_Vector& vec,  ///< vector to compute norm of
        const int numneglect = 0  ///< number of DOFs that have to be neglected for length scaling
    ) const;

    //! return order of accuracy of auxiliary integrator
    int aux_method_order_of_accuracy() const;

    //! return leading error coefficient of velocity of auxiliary integrator
    double aux_method_lin_err_coeff_vel() const;

    Teuchos::RCP<Epetra_Vector> locerrvelnp_;  ///< vector of temporal local discretization error

    INPAR::FSI::FluidMethod auxintegrator_;  ///< auxiliary time integrator in fluid field

    int numfsidbcdofs_;  ///< number of interface DOFs with Dirichlet boundary condition

    ETimAdaAux methodadapt_;  ///< type of adaptive method

    //@}
  };
}  // namespace ADAPTER

FOUR_C_NAMESPACE_CLOSE

#endif

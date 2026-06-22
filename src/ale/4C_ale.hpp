// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ALE_HPP
#define FOUR_C_ALE_HPP

#include "4C_config.hpp"

#include "4C_adapter_ale.hpp"
#include "4C_ale_input.hpp"
#include "4C_ale_meshtying.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class Solver;
  class SparseOperator;
  class SparseMatrix;
  class BlockSparseMatrixBase;
  class MapExtractor;
}  // namespace Core::LinAlg

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Conditions
{
  class LocsysManager;
}


namespace Core::IO
{
  class DiscretizationWriter;
}  // namespace Core::IO

namespace ALE
{
  namespace Utils
  {
    class MapExtractor;
  }  // namespace Utils
}  // namespace ALE


namespace ALE
{
  /*! \class Ale
   *  \brief ALE time integration
   *
   *  Pure ALE field for nonlinear mesh motion algorithms. To include this into a
   *  coupled problem, use a problem specific adapter that derives from
   *  Adapter::AleWrapper.
   *
   *  We provide the following ALE formulations:
   *  <ul>
   *  <li> solid: assume the ALE mesh to be a elastic, quasi-static solid body. We
   *              allow the ALE field to have any material from the
   *              Mat::ElastHyper tool box. </li>
   *  <li> springs: spring analogy where the nodes are connected by lineal springs
   *                and additional torsional springs in the element corners. </li>
   *  <li> laplace: mesh motion as a Laplacian smoother where the diffusuvity is
   *                computed based on the Jacobian determinant. </li>
   *  </ul>
   *
   *  Since all ALE formulations just differ in the element evaluation routines,
   *  there is no difference between them on the time integration level. We just
   *  use the ALE_TYPE from the input file to pass an type-specific element action
   *  kenner to the element evaluation and distinguish between the different
   *  formulation only on the element level.
   *
   *  <h3>References:</h3>
   *  For springs:
   *  <ul>
   *  <li> Batina, J. T.: Unsteady Euler algorithm with unstructured dynamic mesh
   *       for complex-aircraft aerodynamic analysis, AIAA Journal (29), No. 3,
   *       pp. 327-333, 1991 </li>
   *  <li> Farhat, C., Degand, C, Koobus, B and Lesoinne, M.: Torsional springs
   *       for two-dimensional dynamic unstructured fluid meshes, CMAME (163),
   *       No. 1-4, pp. 231-245, 1998 </li>
   *  <li> Zeng, D. and Ross Ethier, C.: A semi-torsional spring analogy model for
   *       updating unstructured meshes in 3D moving domains, Finite Elements in
   *       Analysis and Design (41), No. 11-12, pp. 1118-1139, 2005 </li>
   *  <li> Degand, C. and Farhat, C.: A three-dimensional torional spring analogy
   *       method for unstructured dynamic meshes, Computers & Structures (80),
   *       No. 3-4, pp. 305-316, 2002
   *  </ul>
   *
   *  \sa Solid::TimInt, FLD::TimInt, ALE::AleLinear
   *
   */
  class Ale : public Adapter::Ale
  {
    // friend class AleResultTest;

   public:
    Ale(std::shared_ptr<Core::FE::Discretization> actdis,       ///< pointer to discretization
        std::shared_ptr<Core::LinAlg::Solver> solver,           ///< linear solver
        std::shared_ptr<Teuchos::ParameterList> params,         ///< parameter list
        std::shared_ptr<Core::IO::DiscretizationWriter> output  ///< output writing
    );

    /*!
     *  \brief Set initial displacement field
     *
     *  Use this for pure ALE problems as well as for coupled problems.
     *
     *  \param[in]     init Initial displacement field
     *
     *  \param[in]     startfuncno Function to evaluate initial displacement
     *
     */
    virtual void set_initial_displacement(const ALE::InitialDisp init, const int startfuncno);

    /*! \brief Create Systemmatrix
     *
     * We allocate the Core::LINALG object just once, the result is an empty
     * Core::LINALG object. Evaluate has to be called separately.
     *
     */
    void create_system_matrix(
        std::shared_ptr<const ALE::Utils::MapExtractor> interface = nullptr  //!< interface
        ) override;

    /*! \brief evaluate and assemble residual #residual_ and jacobian matrix #sysmat_
     *
     *  use this as evaluate routine for pure ALE problems as well as for coupled problems.
     *  Update in case of monolithic coupling is done by passing stepinc, nullptr is assumed
     * for non monolithic case.
     */
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> stepinc =
                      nullptr,  ///< step increment such that \f$ x_{n+1}^{k+1} =
                                ///< x_{n}^{converged}+ stepinc \f$
        ALE::Utils::MapExtractor::AleDBCSetType dbc_type =
            ALE::Utils::MapExtractor::dbc_set_std  ///< application-specific type of Dirichlet set
        ) override;

    /// linear solve
    int solve() override;

    /// get the linear solver object used for this field
    std::shared_ptr<Core::LinAlg::Solver> linear_solver() override { return solver_; }

    //! update displacement with iterative increment
    void update_iter() override;

    /// take the current solution to be the final one for this time step
    void update() override;

    /// convergence test for newton
    virtual bool converged(const int iter);

    /// Evaluate all elements
    virtual void evaluate_elements();

    /// Convert element action enum to std::string
    virtual std::string element_action_string(const ALE::AleDynamic name  ///< enum to convert
    );

    //! @name Time step helpers

    /// a very simple time loop to be used for standalone ALE problems
    int integrate() override;

    /// start a new time step
    void prepare_time_step() override;

    /*! \brief Do a single time step
     *
     *  Perform Newton iteration to solve the nonlinear problem.
     */
    void time_step(ALE::Utils::MapExtractor::AleDBCSetType dbc_type =
                       ALE::Utils::MapExtractor::dbc_set_std) override;

    /// write output
    void output() override;

    /*! \brief Reset time step
     *
     *  In case of time step size adaptivity, time steps might have to be repeated.
     *  Therefore, we need to reset the solution back to the initial solution of
     *  the time step.
     *
     */
    void reset_step() override;

    /*! \brief Reset time and step in case that a time step has to be repeated
     *
     *  ALE field increments time and step at the beginning of a time step. If a
     *  time step has to be repeated, we need to take this into account and
     *  decrease time and step beforehand. They will be incremented right at the
     *  beginning of the repetition and, thus, everything will be fine. Currently,
     *  this is needed for time step size adaptivity in FSI.
     *
     */
    void reset_time(const double dtold) override;

    /// Get current simulation time
    double time() const override { return time_; }

    /// Get current step counter
    double step() const override { return step_; }

    /// Get the time step size
    double dt() const override { return dt_; }

    /// set time step step size
    void set_dt(const double dtnew) override;

    /// read restart for given step
    void read_restart(const int step) override;

    //@}

    //! @name Reading access to displacement
    //@{

    /// get the whole displacement field at time step \f$t^{n+1}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp() const override { return dispnp_; }

    /// get the whole displacement field at time step \f$t^{n}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispn() const override { return dispn_; }

    //@}

    //! @name Writing access to displacement

    /// write access to whole displacement field at time step \f$t^{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> write_access_dispnp() const override
    {
      return dispnp_;
    }

    //@}

    //! @name Vector access

    /// initial guess of Newton's method
    std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() const override
    {
      return zeros_;
    }

    /// rhs of Newton's method
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() const override { return rhs_; }

    //@}

    //! @name Misc

    /// dof map of vector of unknowns
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() const override;

    /// direct access to system matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() override;

    /// direct access to system matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() override;

    /// direct access to discretization
    std::shared_ptr<const Core::FE::Discretization> discretization() const override
    {
      return discret_;
    }

    /// writing access to discretization
    std::shared_ptr<Core::FE::Discretization> write_access_discretization() override
    {
      return discret_;
    }

    /*! \brief setup Dirichlet boundary condition map extractor.
     *
     *  Generally, all application-specific information belongs to the subsequent
     *  adapter class - this routine is an exception.
     *  This method creates application-specific Dirichlet maps and stores them in
     *  a map, together with an application key; by passing this key to routines
     *  like evaluate, an adapter classes can assure, that its very own Dirichlet
     *  map extractor is used.
     */
    void setup_dbc_map_ex(
        ALE::Utils::MapExtractor::AleDBCSetType dbc_type =
            ALE::Utils::MapExtractor::dbc_set_std,  //!< application-specific type of Dirichlet set
        std::shared_ptr<const ALE::Utils::MapExtractor> interface =
            nullptr,  //!< interface for creation of additional, application-specific
                      //!< Dirichlet map extractors
        std::shared_ptr<const ALE::Utils::XFluidFluidMapExtractor> xff_interface =
            nullptr  //!< interface for creation of a Dirichlet map extractor, tailored to
                     //!< XFFSI
        ) override;

    /// create result test for encapsulated algorithm
    std::shared_ptr<Core::Utils::ResultTest> create_field_test() override;

    std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor(
        ALE::Utils::MapExtractor::AleDBCSetType dbc_type =
            ALE::Utils::MapExtractor::dbc_set_std  //!< application-specific type of Dirichlet set
        ) override
    {
      return dbcmaps_[dbc_type];
    }

    //! Return (rotatory) transformation matrix of local co-ordinate systems
    std::shared_ptr<const Core::LinAlg::SparseMatrix> get_loc_sys_trafo() const;

    //! Update slave dofs for multifield simulations with ale
    void update_source_dof(std::shared_ptr<Core::LinAlg::Vector<double>>& a) override;

    //! Return locsys manager
    std::shared_ptr<Core::Conditions::LocsysManager> locsys_manager() override
    {
      return locsysman_;
    }

    //! Apply Dirichlet boundary conditions on provided state vectors
    void apply_dirichlet_bc(Teuchos::ParameterList& params,
        std::shared_ptr<Core::LinAlg::Vector<double>> systemvector,    //!< (may be nullptr)
        std::shared_ptr<Core::LinAlg::Vector<double>> systemvectord,   //!< (may be nullptr)
        std::shared_ptr<Core::LinAlg::Vector<double>> systemvectordd,  //!< (may be nullptr)
        bool recreatemap  //!< recreate mapextractor/toggle-vector
                          //!< which stores the DOF IDs subjected
                          //!< to Dirichlet BCs
                          //!< This needs to be true if the bounded DOFs
                          //!< have been changed.
    );

    /// Reset state vectors to zero
    void reset() override;

    //! Set time and step
    void set_time_step(const double time, const int step) override
    {
      time_ = time;
      step_ = step;
    }

    //@}

   protected:
    //! Read parameter list
    const Teuchos::ParameterList& params() const { return *params_; }

    //! write access to residual
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> write_access_residual() const
    {
      return residual_;
    }

   private:
    virtual bool update_sys_mat_every_step() const { return true; }

    //! @name Misc

    //! ALE discretization
    std::shared_ptr<Core::FE::Discretization> discret_;

    //! linear solver
    std::shared_ptr<Core::LinAlg::Solver> solver_;

    //! parameter list
    std::shared_ptr<Teuchos::ParameterList> params_;

    //! output writing
    std::shared_ptr<Core::IO::DiscretizationWriter> output_;

    //! Dirichlet BCs with local co-ordinate system
    std::shared_ptr<Core::Conditions::LocsysManager> locsysman_;

    //@}

    //! @name Algorithm core variables
    int step_;               ///< step counter
    int numstep_;            ///< max number of steps
    double time_;            ///< simulation time
    double maxtime_;         ///< max simulation time
    double dt_;              ///< time step size
    int writerestartevery_;  ///< write restart every n steps
    int writeresultsevery_;  ///< write results every n steps
    //@}

    //! @name matrices, vectors
    //@{

    std::shared_ptr<Core::LinAlg::SparseOperator> sysmat_;  ///< stiffness matrix

    /*! \brief residual vector
     *
     *  This is the "mechanical" residual \f$res = - f_{int}\f$ as it comes
     *  from the discret_->evaluate() call.
     *
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> residual_;

    /*! \brief right hand side of Newton-type algorithm
     *
     *  Use this as the right hand side for a Newton algorithm. It should equal
     *  the negative residual: #rhs_ = - #residual_
     *
     *  We update this variable only right after the discret_->evaluate() call.
     *
     *  \warning DO NOT TOUCH THIS VARIABLE AT OTHER PLACES!!!
     *
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_;

    std::shared_ptr<Core::LinAlg::Vector<double>> dispnp_;  ///< unknown solution at \f$t_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> dispn_;   ///< known solution at \f$t_{n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> disi_;    ///< iterative displacement increment
    double normdisi_;  ///< norm of iterative displacement increment
    std::shared_ptr<const Core::LinAlg::Vector<double>> zeros_;  ///< zero vector for dbc handling

    //@}

    //! @name map extractors
    //@{

    /*! \brief map with application-specific Dirichlet map extractors
     *
     *  Each adapter class can extract its map extractor via
     *  an application-specific key
     */
    std::map<int, std::shared_ptr<Core::LinAlg::MapExtractor>> dbcmaps_;

    //@}

    //! @name Assess mesh regularity and element quality
    //@!{

    //! Loop all elements to compute quality measure according to [Oddy et al. 1988a]
    virtual bool evaluate_element_quality();

    //! det of element jacobian
    std::shared_ptr<Core::LinAlg::Vector<double>> eledetjac_;

    /*! \brief Element quality measure according to [Oddy et al. 1988a]
     *
     *  Distortion metric for quadrilaterals and hexahedrals. Value is zero for
     *  squares/cubes and increases to large values for distorted elements.
     *
     *  Reference: Oddy A, Goldak J, McDill M, Bibby M (1988): A distortion metric
     *  for isoparametric finite elements, Trans. Can. Soc. Mech. Engrg.,
     *  Vol. 12 (4), pp. 213-217
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> elequality_;

    //! Flag to activate (true) and deactivate (false) assessment of mesh quality
    const bool elequalityyesno_;

    //@}


    /// print info about current time step to screen
    virtual void print_time_step_header() const;

    /// write restart data
    virtual void output_restart(bool& datawritten);

    /// write output data
    virtual void output_state(bool& datawritten);

    /// ale formulation read from inputfile
    const ALE::AleDynamic aletype_;

    //! @name solver parameters
    //@{
    //! maximum number of newton iterations
    const int maxiter_;

    //! tolerance of length scaled L2 residual norm
    const double tolres_;

    //! tolerance of length scaled L2 increment norm
    const double toldisp_;

    //! error handling in case of unconverged nonlinear solver
    const ALE::DivContAct divercont_;

    //! flag for mesh-tying
    const ALE::MeshTying msht_;

    //! flag for initial displacement
    const ALE::InitialDisp initialdisp_;

    //! start function number
    const int startfuncno_;

    //! coupling of ALE-ALE at an internal interface
    std::shared_ptr<ALE::Meshtying> meshtying_;

    //@}

  };  // class Ale

  /*! \class AleLinear
   *  \brief Ale time integrator for linear mesh motion algorithms
   *
   *  Simplification of nonlinear ALE::Ale class in case of a linear mesh motion
   *  algorithm. Only functions related to nonlinear solution techniques must be
   *  overloaded, i.e. evaluate_elements or the nonlinear solve.
   *
   *  Linear mesh motion should be sufficient in case of small or uniform/
   *  volumetric mesh deformation, but evaluates the system matrix just once and
   *  is much cheaper than the nonlinear version.
   *
   *  We allow for two options:
   *  <ul>
   *  <li> Fully linear: The system matrix is evaluated only once at the beginning
   *       of the simulation. The residual is computed as \f$r = K*d\f$. </li>
   *  <li> Pseudo-linear: The system matrix is evaluated at the beginning of each
   *       time step, while dependencies on the displacement field are considered.
   *       The residual is computed as \f$r = K*d\f$. </li>
   *  </ul>
   *
   */
  class AleLinear : public Ale
  {
   public:
    //! @name Construction / Destruction
    //@{

    //! Constructor
    AleLinear(std::shared_ptr<Core::FE::Discretization> actdis,  ///< pointer to discretization
        std::shared_ptr<Core::LinAlg::Solver> solver,            ///< linear solver
        std::shared_ptr<Teuchos::ParameterList> params_in,       ///< parameter list
        std::shared_ptr<Core::IO::DiscretizationWriter> output   ///< output writing
    );

    //@}

    //! Computation
    //@{

    /*! \brief Start a new time step
     *
     *  Prepare time step as in nonlinear case. Reset #validsysmat_ in case of an
     *  updated strategy, i.e. if #sysmat_ needs to be recomputed at the beginning
     *  of each time step.
     *
     */
    void prepare_time_step() override;

    /*! \brief Do a single time step
     *
     *  Just call the linear solver once.
     *
     */
    void time_step(ALE::Utils::MapExtractor::AleDBCSetType dbc_type =
                       ALE::Utils::MapExtractor::dbc_set_std) override;

    /*! \brief Evaluate all elements
     *
     *  In the linear case, the system matrix \f$K\f$ is kept constant throughout
     *  the entire computation. Thus, we call ALE::Ale::evaluate_elements() once in
     *  the beginning to compute the stiffness matrix. Afterwards, we only need to
     *  compute the current residual \f$f_{res}\f$ as
     *  \f[
     *    f_{res} = Kd
     *  \f]
     *  based on the current displacements \f$d\f$.
     *
     *  In order to initially provide a matrix, we call Ale::evaluate_elements() in
     *  the very first call. This is kept track of by #validsysmat_.
     *
     */
    void evaluate_elements() override;

    //@}

   protected:
   private:
    bool update_sys_mat_every_step() const override { return updateeverystep_; }

    //! Is the #sysmat_ valid (true) or does it need to be re-evaluated (false)
    bool validsysmat_;

    //! \brief Update stiffness matrix oncer per time step ?
    bool updateeverystep_;

  };  // class AleLinear

}  // namespace ALE

FOUR_C_NAMESPACE_CLOSE

#endif

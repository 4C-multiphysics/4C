/*----------------------------------------------------------------------------*/
/*! \file
\brief Basic HDG weakly compressible time-integration scheme

\level 2

*/
/*----------------------------------------------------------------------------*/


#ifndef FOUR_C_FLUID_TIMINT_HDG_WEAK_COMP_HPP
#define FOUR_C_FLUID_TIMINT_HDG_WEAK_COMP_HPP


#include "baci_config.hpp"

#include "baci_fluid_timint_genalpha.hpp"

FOUR_C_NAMESPACE_OPEN


namespace FLD
{
  /*!
  \brief time integration for HDG fluid (only gen-alpha implemented)

  */
  class TimIntHDGWeakComp : public TimIntGenAlpha
  {
   public:
    /// Standard Constructor
    TimIntHDGWeakComp(const Teuchos::RCP<DRT::Discretization>& actdis,
        const Teuchos::RCP<CORE::LINALG::Solver>& solver,
        const Teuchos::RCP<Teuchos::ParameterList>& params,
        const Teuchos::RCP<IO::DiscretizationWriter>& output, bool alefluid);

    /*!
    \brief initialization

    */
    void Init() override;

    /*!
    \brief Set theta_ to its value, dependent on integration method for GenAlpha and BDF2

    */
    void SetTheta() override;

    /*!
    \brief do explicit predictor step to start nonlinear iteration from
           a better initial value
    */
    void ExplicitPredictor() override;

    /*!
    \brief Set custom parameters in the respective time integration class (Loma, RedModels...)

    */
    void SetCustomEleParamsAssembleMatAndRHS(Teuchos::ParameterList& eleparams) override;

    /*!
    \brief Set states in the time integration schemes: additional vectors for HDG

    */
    void SetStateTimInt() override;

    /*!
    \brief Call discret_->ClearState() after assembly (HDG needs to read from state vectors...)

    */
    void ClearStateAssembleMatAndRHS() override;

    /*!
    \brief Set the part of the right hand side belonging to the last
           time step for incompressible or low-Mach-number flow

       for low-Mach-number flow: distinguish momentum and continuity part
       (continuity part only meaningful for low-Mach-number flow)

       Stationary/af-generalized-alpha:

                     mom: hist_ = 0.0
                    (con: hist_ = 0.0)

       One-step-Theta:

                     mom: hist_ = veln_  + dt*(1-Theta)*accn_
                    (con: hist_ = densn_ + dt*(1-Theta)*densdtn_)

       BDF2: for constant time step:

                     mom: hist_ = 4/3 veln_  - 1/3 velnm_
                    (con: hist_ = 4/3 densn_ - 1/3 densnm_)


    */
    void SetOldPartOfRighthandside() override;

    /*!
    \brief update within iteration

    */
    void IterUpdate(const Teuchos::RCP<const Epetra_Vector> increment) override;

    /*!
    \brief Update the solution after convergence of the nonlinear
           iteration. Current solution becomes old solution of next
           time step.
    */
    void TimeUpdate() override;

    /*!
    \brief Update the grid velocity
    */
    void UpdateGridv() override;

    /*!
    \brief set initial flow field for analytical test problems

    */
    void SetInitialFlowField(
        const INPAR::FLUID::InitialField initfield, const int startfuncno) override;

    /*!
    \brief calculate error between a analytical solution and the
           numerical solution of a test problems

    */
    Teuchos::RCP<std::vector<double>> EvaluateErrorComparedToAnalyticalSol() override;

    /*!
    \brief Reset state vectors
     */
    void Reset(bool completeReset = false, int numsteps = 1, int iter = -1) override;

    /*!
    \brief update configuration and output to file/screen

    */
    void Output() override;

    /*!
    \brief accessor to interior velocity

    */
    virtual Teuchos::RCP<Epetra_Vector> ReturnIntVelnp() { return intvelnp_; }
    virtual Teuchos::RCP<Epetra_Vector> ReturnIntVeln() { return intveln_; }
    virtual Teuchos::RCP<Epetra_Vector> ReturnIntVelnm() { return intvelnm_; }


   protected:
    /// copy constructor
    TimIntHDGWeakComp(const TimIntHDGWeakComp& old);

    /*!
    \brief update acceleration for generalized-alpha time integration

    */
    void GenAlphaUpdateAcceleration() override;

    /*!
    \brief compute values at intermediate time steps for gen.-alpha

    */
    void GenAlphaIntermediateValues() override;

    //! @name mixed variable, density and momentum at time n+1, n, n-1
    //!  and n+alpha_F for element interior in HDG
    Teuchos::RCP<Epetra_Vector> intvelnp_;
    Teuchos::RCP<Epetra_Vector> intveln_;
    Teuchos::RCP<Epetra_Vector> intvelnm_;
    Teuchos::RCP<Epetra_Vector> intvelaf_;
    //@}

    //! @name time derivatives at time n+1, n and n+alpha_M/(n+alpha_M/n)
    //!  and n-1 for element interior in HDG
    //@{
    Teuchos::RCP<Epetra_Vector> intaccnp_;  ///< acceleration at time \f$t^{n+1}\f$
    Teuchos::RCP<Epetra_Vector> intaccn_;   ///< acceleration at time \f$t^{n}\f$
    Teuchos::RCP<Epetra_Vector> intaccnm_;  ///< acceleration at time \f$t^{n-1}\f$
    Teuchos::RCP<Epetra_Vector> intaccam_;  ///< acceleration at time \f$t^{n+\alpha_M}\f$
    //@}

    //! @name other HDG-specific auxiliary vectors for output
    Teuchos::RCP<Epetra_MultiVector> interpolatedMixedVar_;
    Teuchos::RCP<Epetra_Vector> interpolatedDensity_;
    //@}


   private:
    ///< Print stabilization details to screen. Do nothing here because we do not use stabilization
    void PrintStabilizationDetails() const override {}

    ///< time algorithm flag actually set (we internally reset it)
    INPAR::FLUID::TimeIntegrationScheme timealgoset_;

    ///< Keep track of whether we do the first assembly of a time step because we reconstruct the
    ///< local HDG solution as part of assembly
    bool first_assembly_;

  };  // class TimIntHDGWeakComp

}  // namespace FLD


FOUR_C_NAMESPACE_CLOSE

#endif

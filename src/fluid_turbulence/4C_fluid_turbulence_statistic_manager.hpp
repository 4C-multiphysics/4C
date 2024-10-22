// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_TURBULENCE_STATISTIC_MANAGER_HPP
#define FOUR_C_FLUID_TURBULENCE_STATISTIC_MANAGER_HPP

#include "4C_config.hpp"

#include "4C_inpar_fluid.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_linalg_vector.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE
namespace Core::DOFSets
{
  class DofSet;
}

namespace Core::IO
{
  class DiscretizationReader;
  class DiscretizationWriter;
}  // namespace Core::IO

namespace ScaTra
{
  class ScaTraTimIntImpl;
}

namespace FLD
{
  // forward declarations
  class FluidImplicitTimeInt;
  class XWall;
  class TurbulenceStatisticsCha;
  class TurbulenceStatisticsCcy;
  class TurbulenceStatisticsLdc;
  class TurbulenceStatisticsBfs;
  class TurbulenceStatisticsSqc;
  class TurbulenceStatisticsHit;
  class TurbulenceStatisticsTgv;
  class TurbulenceStatisticsGeneralMean;
  class TurbulenceStatisticsPh;
  class TurbulenceStatisticsBfda;
  namespace Utils
  {
    class StressManager;
  }


  class TurbulenceStatisticManager
  {
   public:
    /*!
    \brief One-Step-Theta Constructor (public)

    */
    TurbulenceStatisticManager(FluidImplicitTimeInt& fluid);


    /*!
    \brief Destructor

    */
    virtual ~TurbulenceStatisticManager() = default;


    //! @name managing methods

    /*!
    \brief Store values computed during the element call

    (including Cs, visceff etc for a dynamic
     Smagorinsky model)

    */
    void store_element_values(int step);

    /*!
    \brief Get current velnp for meshtying, since it may have changed

    */
    void get_current_velnp(Teuchos::RCP<Core::LinAlg::Vector<double>>);

    /*!
    \brief Include current quantities in the time
    averaging procedure

    Include current quantities in the time
    averaging procedure. The velnp parameters with
    the default of Teuchos::null can be used in
    case:
    1. discretmatchingvelnp: the discretization has a varying
                             number of dofs.
    2. velnp:                instead of updating the original
                             velnp to which myvelnp_ points a
                             new velnp is created.

    \param step (in)       : current time step
    \param time (in)       : current time
    \param eosfac (in)     : equation of state factor
    \param velnp (in)      : velnp built with dofrowmap from standarddofset
    \param force (in)      : trueresidual vector built with standarddofset
    \param discretmatchingvelnp (in) : velnp built with dofrowmap from discretization

    */
    void do_time_sample(int step, const double eosfac, const double thermpressaf = 0.0,
        const double thermpressam = 0.0, const double thermpressdtaf = 0.0,
        const double thermpressdtam = 0.0);

    /*!
    \brief ???

    */
    void do_time_sample(int step, Core::LinAlg::Vector<double>& velnp,
        Core::LinAlg::Vector<double>& force, Core::LinAlg::Vector<double>& phi,
        const Core::DOFSets::DofSet& stddofset);

    /*!
    \brief Write (dump) the statistics to a file

    */
    void do_output(Core::IO::DiscretizationWriter& output, int step, const bool inflow = false);

    /*!
    \brief Restart collection of statistics

    */
    void read_restart(Core::IO::DiscretizationReader& reader, int step);

    /*!
    \brief Restart scatra-specific collection of statistics

    */
    void read_restart_scatra(Core::IO::DiscretizationReader& scatrareader, int step);

    /*!
    \brief Provide access to scalar transport field

    */
    void add_scatra_field(Teuchos::RCP<ScaTra::ScaTraTimIntImpl> scatra_timeint);

    /*!
    \brief   Write (dump) the scatra-specific mean fields to the result file

    */
    void do_output_for_scatra(Core::IO::DiscretizationWriter& output, int step);

    //@}

    /*!
    \brief remote access method to general mean statistics manager

    */
    Teuchos::RCP<TurbulenceStatisticsGeneralMean> get_turbulence_statistics_general_mean() const
    {
      return statistics_general_mean_;
    }

    bool with_scatra() { return withscatra_; }

   private:
    /*!
    \brief Time integration independent setup called by Constructor

    */
    void setup();

    //! time step size
    double dt_;

    //! parameters for sampling/dumping period
    //! start of sampling at step samstart
    int samstart_;
    //! stop sampling at step samstop
    int samstop_;
    //! incremental dump every dumperiod steps or standalone
    //! records (0)
    int dumperiod_;

    //! the fluid discretization
    Teuchos::RCP<Core::FE::Discretization> discret_;
    //! the scatra discretization
    Teuchos::RCP<Core::FE::Discretization> scatradis_;

    //! parameterlist of the discretization including time params,
    //! stabilization params and turbulence sublist
    Teuchos::RCP<Teuchos::ParameterList> params_;
    //! parameterlist of the scatra discretization including time params,
    //! stabilization params and turbulence sublist
    Teuchos::RCP<Teuchos::ParameterList> scatraparams_;
    //! since model parameters (Prt, Csgs, etc.) are recomputed during
    //! simulation and change their values, we need a pointer to this
    //! list and cannot use a copy stored in scatraparams_
    Teuchos::RCP<Teuchos::ParameterList> scatraextraparams_;
    Teuchos::RCP<Teuchos::ParameterList> scatratimeparams_;

    //! parameterlist specially designed for the evaluation of
    //! gausspoint statistics
    Teuchos::ParameterList eleparams_;

    //! decides whether we use an Eulerian or an ALE formulation
    bool alefluid_;

    //! all my solution vectors needed for element evaluation
    Teuchos::RCP<Core::LinAlg::Vector<double>> myaccnp_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> myaccn_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> myaccam_;

    Teuchos::RCP<Core::LinAlg::Vector<double>> myvelnp_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> myveln_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> myvelaf_;

    Teuchos::RCP<Core::LinAlg::Vector<double>> myhist_;

    Teuchos::RCP<Core::LinAlg::Vector<double>> myscaaf_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> myscaam_;

    Teuchos::RCP<Core::LinAlg::Vector<double>> mydispnp_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> mydispn_;

    Teuchos::RCP<Core::LinAlg::Vector<double>> mygridvelaf_;

    Teuchos::RCP<Core::LinAlg::Vector<double>> myforce_;

    Teuchos::RCP<Core::LinAlg::Vector<double>> myfsvelaf_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> myfsscaaf_;
    // xwall object is required for evaluating inner element planes of channel
    Teuchos::RCP<FLD::XWall> myxwall_;

    Teuchos::RCP<FLD::Utils::StressManager> mystressmanager_;

    //! scatra result vector (defined on the scatra dofrowmap!)
    Teuchos::RCP<Core::LinAlg::Vector<double>> myphinp_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> myphiaf_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> myphiam_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> myfsphi_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> myscatrahist_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> myphidtam_;

    //! specifies the special flow
    enum SpecialFlow
    {
      no_special_flow,
      channel_flow_of_height_2,
      loma_channel_flow_of_height_2,
      scatra_channel_flow_of_height_2,
      lid_driven_cavity,
      loma_lid_driven_cavity,
      backward_facing_step,
      loma_backward_facing_step,
      backward_facing_step2,
      square_cylinder,
      square_cylinder_nurbs,
      bubbly_channel_flow,
      rotating_circular_cylinder_nurbs,
      rotating_circular_cylinder_nurbs_scatra,
      decaying_homogeneous_isotropic_turbulence,
      forced_homogeneous_isotropic_turbulence,
      scatra_forced_homogeneous_isotropic_turbulence,
      taylor_green_vortex,
      periodic_hill,
      blood_fda_flow,
      time_averaging
    } flow_;

    //! flag to include scatra field
    bool withscatra_;

    //! toggle additional evaluations for turbulence models
    enum Inpar::FLUID::TurbModelAction turbmodel_;

    //! toggle evaluation of subgrid quantities, dissipation rates etc
    //! this is only possible for the genalpha implementation since
    //! we need a corresponding element implementation
    bool subgrid_dissipation_;

    //! toggle statistics output of additional inflow
    bool inflow_;

    //! mean values of velocity and pressure, independent of special flow
    //! averaging takes place in time, if hom. directions have been specified
    //! additionally along these lines
    bool out_mean_;

    //! name of statistics output file, despite the ending
    const std::string statistics_outfilename_;

    Teuchos::RCP<TurbulenceStatisticsGeneralMean> statistics_general_mean_;

    //! turbulence statistics for turbulent channel flow
    Teuchos::RCP<TurbulenceStatisticsCha> statistics_channel_;

    //! turbulence statistics for a rotating circular cylinder
    Teuchos::RCP<TurbulenceStatisticsCcy> statistics_ccy_;

    //! turbulence statistics for lid-driven cavity
    Teuchos::RCP<TurbulenceStatisticsLdc> statistics_ldc_;

    //! turbulence statistics for backward-facing step
    Teuchos::RCP<TurbulenceStatisticsBfs> statistics_bfs_;

    //! turbulence statistics for periodic hill
    Teuchos::RCP<TurbulenceStatisticsPh> statistics_ph_;

    //! turbulence statistics for blood fda flow
    Teuchos::RCP<TurbulenceStatisticsBfda> statistics_bfda_;

    //! turbulence statistics for square cylinder
    Teuchos::RCP<TurbulenceStatisticsSqc> statistics_sqc_;

    //! turbulence statistics for homogeneous isotropic turbulence
    Teuchos::RCP<TurbulenceStatisticsHit> statistics_hit_;

    //! turbulence statistics for Taylor-Green Vortex
    Teuchos::RCP<TurbulenceStatisticsTgv> statistics_tgv_;

  };  // end class turbulence_statistic_manager

}  // end namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif

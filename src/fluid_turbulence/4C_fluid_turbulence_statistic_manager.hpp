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

#include <memory>

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
  class TurbulenceStatisticsHit;
  class TurbulenceStatisticsPh;
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
    void get_current_velnp(std::shared_ptr<Core::LinAlg::Vector<double>>);

    /*!
    \brief Include current quantities in the time
    averaging procedure

    Include current quantities in the time
    averaging procedure. The velnp parameters with
    the default of nullptr can be used in
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
    void add_scatra_field(std::shared_ptr<ScaTra::ScaTraTimIntImpl> scatra_timeint);

    /*!
    \brief   Write (dump) the scatra-specific mean fields to the result file

    */
    void do_output_for_scatra(Core::IO::DiscretizationWriter& output, int step);

    //@}

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
    std::shared_ptr<Core::FE::Discretization> discret_;
    //! the scatra discretization
    std::shared_ptr<Core::FE::Discretization> scatradis_;

    //! parameterlist of the discretization including time params,
    //! stabilization params and turbulence sublist
    std::shared_ptr<Teuchos::ParameterList> params_;
    //! parameterlist of the scatra discretization including time params,
    //! stabilization params and turbulence sublist
    std::shared_ptr<Teuchos::ParameterList> scatraparams_;
    //! since model parameters (Prt, Csgs, etc.) are recomputed during
    //! simulation and change their values, we need a pointer to this
    //! list and cannot use a copy stored in scatraparams_
    std::shared_ptr<Teuchos::ParameterList> scatraextraparams_;
    std::shared_ptr<Teuchos::ParameterList> scatratimeparams_;

    //! parameterlist specially designed for the evaluation of
    //! gausspoint statistics
    Teuchos::ParameterList eleparams_;

    //! decides whether we use an Eulerian or an ALE formulation
    bool alefluid_;

    //! all my solution vectors needed for element evaluation
    std::shared_ptr<Core::LinAlg::Vector<double>> myaccnp_;
    std::shared_ptr<Core::LinAlg::Vector<double>> myaccn_;
    std::shared_ptr<Core::LinAlg::Vector<double>> myaccam_;

    std::shared_ptr<Core::LinAlg::Vector<double>> myvelnp_;
    std::shared_ptr<Core::LinAlg::Vector<double>> myveln_;
    std::shared_ptr<Core::LinAlg::Vector<double>> myvelaf_;

    std::shared_ptr<Core::LinAlg::Vector<double>> myhist_;

    std::shared_ptr<Core::LinAlg::Vector<double>> myscaaf_;
    std::shared_ptr<Core::LinAlg::Vector<double>> myscaam_;

    std::shared_ptr<Core::LinAlg::Vector<double>> mydispnp_;
    std::shared_ptr<Core::LinAlg::Vector<double>> mydispn_;

    std::shared_ptr<Core::LinAlg::Vector<double>> mygridvelaf_;

    std::shared_ptr<Core::LinAlg::Vector<double>> myforce_;

    std::shared_ptr<Core::LinAlg::Vector<double>> myfsvelaf_;
    std::shared_ptr<Core::LinAlg::Vector<double>> myfsscaaf_;
    // xwall object is required for evaluating inner element planes of channel
    std::shared_ptr<FLD::XWall> myxwall_;

    std::shared_ptr<FLD::Utils::StressManager> mystressmanager_;

    //! scatra result vector (defined on the scatra dofrowmap!)
    std::shared_ptr<Core::LinAlg::Vector<double>> myphinp_;
    std::shared_ptr<Core::LinAlg::Vector<double>> myphiaf_;
    std::shared_ptr<Core::LinAlg::Vector<double>> myphiam_;
    std::shared_ptr<Core::LinAlg::Vector<double>> myfsphi_;
    std::shared_ptr<Core::LinAlg::Vector<double>> myscatrahist_;
    std::shared_ptr<Core::LinAlg::Vector<double>> myphidtam_;

    //! specifies the special flow
    enum SpecialFlow
    {
      no_special_flow,
      channel_flow_of_height_2,
      loma_channel_flow_of_height_2,
      scatra_channel_flow_of_height_2,
      square_cylinder_nurbs,
      bubbly_channel_flow,
      rotating_circular_cylinder_nurbs,
      rotating_circular_cylinder_nurbs_scatra,
      decaying_homogeneous_isotropic_turbulence,
      forced_homogeneous_isotropic_turbulence,
      scatra_forced_homogeneous_isotropic_turbulence,
      periodic_hill,
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
    //! averaging takes place in time, if home. directions have been specified
    //! additionally along these lines
    bool out_mean_;

    //! name of statistics output file, despite the ending
    const std::string statistics_outfilename_;

    //! turbulence statistics for turbulent channel flow
    std::shared_ptr<TurbulenceStatisticsCha> statistics_channel_;

    //! turbulence statistics for a rotating circular cylinder
    std::shared_ptr<TurbulenceStatisticsCcy> statistics_ccy_;

    //! turbulence statistics for periodic hill
    std::shared_ptr<TurbulenceStatisticsPh> statistics_ph_;

    //! turbulence statistics for homogeneous isotropic turbulence
    std::shared_ptr<TurbulenceStatisticsHit> statistics_hit_;

  };  // end class turbulence_statistic_manager

}  // end namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_TURBULENCE_STATISTICS_CHA_HPP
#define FOUR_C_FLUID_TURBULENCE_STATISTICS_CHA_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fem_nurbs_discretization_control_point.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"

#include <Epetra_MpiComm.h>
#include <Teuchos_ParameterList.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN


namespace FLD
{
  class XWall;

  class TurbulenceStatisticsCha
  {
   public:
    /*!
    \brief Standard Constructor (public)

        o Create vector of homogeneous plane coordinates

    o Allocate 4 distributed toggle vectors and one distributed vector
      for squares, both for integral and pointwise means

    o allocate all sum_something vectors

    o initialise the output (open/clear files, print header)


    */
    TurbulenceStatisticsCha(std::shared_ptr<Core::FE::Discretization> actdis, bool alefluid,
        std::shared_ptr<Core::LinAlg::Vector<double>> dispnp, Teuchos::ParameterList& params,
        const std::string& statistics_outfilename, bool subgrid_dissipation,
        std::shared_ptr<FLD::XWall> xwallobj);

    /*!
    \brief Destructor

    */
    virtual ~TurbulenceStatisticsCha() = default;

    //! @name functions for (spatial) averaging


    /*!
    \brief Store scatra discretization for passive scalar transport
    to calculate mean values of multifractal subgrid-scales modeling
    parameters
    */
    void store_scatra_discret_and_params(std::shared_ptr<Core::FE::Discretization> scatradis,
        std::shared_ptr<Teuchos::ParameterList> scatraparams,
        std::shared_ptr<Teuchos::ParameterList> scatraextraparams,
        std::shared_ptr<Teuchos::ParameterList> scatratimeparams);


    /*!
    \brief Compute the in-plane mean values of first and second order
    moments for velocities, pressure and Cs are added to global
    'sum' vectors.
    */
    void do_time_sample(
        const Core::LinAlg::Vector<double>& velnp, const Core::LinAlg::Vector<double>& force);


    /*!
    \brief The values of velocity, pressure, temperature and its squared
    values are added to global vectors. This method allows to do the time
    average of the nodal values after a certain amount of timesteps.
    */
    void do_loma_time_sample(const Core::LinAlg::Vector<double>& velnp,
        const Core::LinAlg::Vector<double>& scanp, const Core::LinAlg::Vector<double>& force,
        const double eosfac);


    /*!
    \brief The values of velocity, pressure, scalar and its squared
    values are added to global vectors. This method allows to do the time
    average of the nodal values after a certain amount of timesteps.
    */
    void do_scatra_time_sample(const Core::LinAlg::Vector<double>& velnp,
        const Core::LinAlg::Vector<double>& scanp, const Core::LinAlg::Vector<double>& force);


    /*!
    \brief Compute in plane means of u,u^2 etc. (integral version)

    The averages here are calculated by integration.

    The calculated value is added to the sum**,sumsq** variables in the
    component corresponding to the plane.

    Further documentation is provided in the element subroutines
    */
    void evaluate_integral_mean_values_in_planes();

    /*!
    \brief Compute in-plane means (integral version) for low-Mach-number flow
    */
    void evaluate_loma_integral_mean_values_in_planes(const double eosfac);

    /*!
    \brief Compute in-plane means (integral version) for turbulent passive scalar transport
    */
    void evaluate_scatra_integral_mean_values_in_planes();

    /*!
    \brief Compute in plane means of u,u^2 etc. (nodal quantities)

    The averages here are calculated as the arithmetic mean of
    point values:

    - generate 4 toggle vectors (u,v,w,p), for example

                              /  1  u dof in homogeneous plane
                   toggleu_  |
                              \  0  elsewhere

    - 2 * 4 scalarproducts for in plane mean values

    - apply toggle vectors to pointwise multiplied velnp.*velnp
      for second order moments


    The calculated values are added to the pointsum**,pointsumsq** variables
    in the component corresponding to the plane.

    velnp is the solution vector provided by the time integration
    algorithm
    */
    void evaluate_pointwise_mean_values_in_planes();

    /*!
      \brief Add computed dynamic Smagorinsky quantities (Smagorinsky
             constant, effective viscosity and (Cs_delta)^2 used
             during the computation)

      The increment is computed during the computation of Cs
      (in the filtering part of time integration, i.e. during
      an element call in the nonlinear iteration)

      We just store it here and add it to the sum as soon as we do
      the time sample.
    */

    void add_dynamic_smagorinsky_quantities();

    /*!
      \brief Add parameters of multifractal
             subgrid-scales model
    */

    void add_model_params_multifractal(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> velnp,
        const std::shared_ptr<const Core::LinAlg::Vector<double>> fsvelnp, const bool withscatra);

    /*!
      \brief do averaging of residuals, dissipation rates etc
             (all gausspoint-quantities)

    */

    void evaluate_residuals(
        std::map<std::string, std::shared_ptr<Core::LinAlg::Vector<double>>> statevecs,
        std::map<std::string, std::shared_ptr<Core::LinAlg::MultiVector<double>>> statetenss,
        const double thermpressaf, const double thermpressam, const double thermpressdtaf,
        const double thermpressdtam,
        std::map<std::string, std::shared_ptr<Core::LinAlg::Vector<double>>> scatrastatevecs,
        std::map<std::string, std::shared_ptr<Core::LinAlg::MultiVector<double>>> scatrafieldvecs);

    //@}

    //! @name Miscellaneous

    /*!
    \brief Compute a time average of the mean values over all steps
    since the last output. Dump the result to file.

    step on input is used to print the timesteps which belong to the
    statistic to the file

    */

    void time_average_means_and_output_of_statistics(const int step);

    /*!
    \brief Compute a time average of the mean values over all steps
    of the sampling period so far. Dump the result to file.

    */

    void dump_statistics(const int step);

    /*!
    \brief Compute a time average of the mean values for low-Mach-number
    flow over all steps of the sampling period so far. Dump the result to
    file.

    */

    void dump_loma_statistics(const int step);

    /*!
    \brief Compute a time average of the mean values for turbulent
    passive scalar transport over all steps of the sampling period so far.
    Dump the result to file.

    */

    void dump_scatra_statistics(const int step);

    /*!
    \brief Reset sums and number of samples to 0

    */

    void clear_statistics();

    /*!
    \brief Provide the coordinates of the homogeneous planes for a
    turbulent channel flow

    */
    std::vector<double> return_node_plane_coords() { return (*nodeplanes_); };

    //@}


   protected:
    /*!
    \brief sort criterium for double values up to a tolerance of 10-9

    This is used to create sets of doubles (e.g. coordinates)

    */
    class PlaneSortCriterion
    {
     public:
      bool operator()(const double& p1, const double& p2) const { return (p1 < p2 - 1E-9); }

     protected:
     private:
    };

   private:
    //! direction normal to homogeneous plane
    int dim_;

    //! number of elements in sample plane
    int numele_;

    //! number of samples taken
    int numsamp_;

    //! number of records written
    int countrecord_;

    //! flag for physical type of fluid flow (standard: incompressible)
    Inpar::FLUID::PhysicalType physicaltype_;

    //! The discretization (required for nodes, dofs etc;)
    std::shared_ptr<Core::FE::Discretization> discret_;

    //! scatra discretization (required for additional multifractal subgrid-scale output in case on
    //! passive scalar transport)
    std::shared_ptr<Core::FE::Discretization> scatradiscret_;

    //! flag for ale discretization
    bool alefluid_;

    //! node displacements due to mesh motion
    std::shared_ptr<Core::LinAlg::Vector<double>> dispnp_;

    //! contains plane normal direction etc --- this is the original
    //! fluid dynamic parameterlist
    Teuchos::ParameterList& params_;
    //! name of statistics output file, despite the ending
    const std::string statistics_outfilename_;
    //! contains plane normal direction etc --- this is the original
    //! scatra dynamic parameterlist
    std::shared_ptr<Teuchos::ParameterList> scatraparams_;
    //! additional parameter list in scatra that contains
    //! required parameters form other sublists such as turbulence models, etc.
    std::shared_ptr<Teuchos::ParameterList> scatraextraparams_;
    //! additional parameterlist in scatra containing the parameters of the
    //! resepctive time-integration scheme
    std::shared_ptr<Teuchos::ParameterList> scatratimeparams_;

    //! toggle evaluation of dynamic Smagorinsky/Smagorinsky with
    //! wall damping quantities
    bool smagorinsky_;

    //! toggle evaluation of scale multifractal quantities
    bool multifractal_;

    //! toggle whether to evaluate residuals, taus, dissipation rates etc
    bool subgrid_dissipation_;

    //! boolean indicating turbulent inflow channel discretization
    const bool inflowchannel_;
    //! x-coordinate of outflow of inflow channel
    const double inflowmax_;

    //! parameterlist for the element call when averages of residuals
    //! are calculated --- used for communication between element
    //! and averaging methods --- for fluid field
    Teuchos::ParameterList eleparams_;

    //! parameterlist for the element call when averages of residuals
    //! are calculated --- used for communication between element
    //! and averaging methods --- for scalar field
    Teuchos::ParameterList scatraeleparams_;

    //! pointer to mean vel/pres and scalar field
    std::shared_ptr<Core::LinAlg::Vector<double>> meanvelnp_;
    std::shared_ptr<Core::LinAlg::Vector<double>> meanscanp_;

    //! pointer to vel/pres^2 field (space allocated in constructor)
    std::shared_ptr<Core::LinAlg::Vector<double>> squaredvelnp_;

    //! toggle vectors --- sums are computed by scalarproducts
    //  with these toggle vectors
    std::shared_ptr<Core::LinAlg::Vector<double>> toggleu_;
    std::shared_ptr<Core::LinAlg::Vector<double>> togglev_;
    std::shared_ptr<Core::LinAlg::Vector<double>> togglew_;
    std::shared_ptr<Core::LinAlg::Vector<double>> togglep_;

    //! the dim_-coordinates of the homogeneous planes containing nodes
    std::shared_ptr<std::vector<double>> nodeplanes_;

    //! the dim_-coordinates of the homogeneous planes --- including
    // additional sampling planes
    std::shared_ptr<std::vector<double>> planecoordinates_;

    //! a bounding box for the channel
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> boundingbox_;

    //! viscosity to calculate l_tau, y+ etc.
    double dens_;
    double visc_;
    //! specific heat capacity to calculate Temp_tau (loma)
    double shc_;
    //! Schmidt number (passive scalar)
    double scnum_;

    std::shared_ptr<FLD::XWall> myxwall_;

    //!--------------------------------------------------
    //!         integration based averaging
    //!--------------------------------------------------
    //
    //! sum over u (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumu_;
    //! sum over v (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumv_;
    //! sum over w (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumw_;
    //! sum over p (over one plane in each component)
    std::shared_ptr<std::vector<double>> sump_;
    //! sum over density (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumrho_;
    //! sum over T (over one plane in each component)
    std::shared_ptr<std::vector<double>> sum_t_;
    //! sum over density*u (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumrhou_;
    //! sum over density*u*T (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumrhou_t_;

    //! sum over u^2 (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumsqu_;
    //! sum over v^2 (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumsqv_;
    //! sum over w^2 (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumsqw_;
    //! sum over p^2 (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumsqp_;
    //! sum over density^2 (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumsqrho_;
    //! sum over T^2 (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumsq_t_;

    //! sum over uv (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumuv_;
    //! sum over uw (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumuw_;
    //! sum over vw (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumvw_;
    //! sum over uv (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumu_t_;
    //! sum over uw (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumv_t_;
    //! sum over vw (over one plane in each component)
    std::shared_ptr<std::vector<double>> sumw_t_;

    //!--------------------------------------------------
    //!       the pointwise averaged stuff
    //!--------------------------------------------------
    //
    //! vector of squared velocities and pressures
    std::shared_ptr<Core::LinAlg::Vector<double>> pointsquaredvelnp_;

    //! sum over u (over one plane in each component)
    std::shared_ptr<std::vector<double>> pointsumu_;
    //! sum over v (over one plane in each component)
    std::shared_ptr<std::vector<double>> pointsumv_;
    //! sum over w (over one plane in each component)
    std::shared_ptr<std::vector<double>> pointsumw_;
    //! sum over p (over one plane in each component)
    std::shared_ptr<std::vector<double>> pointsump_;
    //! sum over T (over one plane in each component)
    std::shared_ptr<std::vector<double>> pointsum_t_;

    //! sum over u^2 (over one plane in each component)
    std::shared_ptr<std::vector<double>> pointsumsqu_;
    //! sum over v^2 (over one plane in each component)
    std::shared_ptr<std::vector<double>> pointsumsqv_;
    //! sum over w^2 (over one plane in each component)
    std::shared_ptr<std::vector<double>> pointsumsqw_;
    //! sum over p^2 (over one plane in each component)
    std::shared_ptr<std::vector<double>> pointsumsqp_;
    //! sum over T^2 (over one plane in each component)
    std::shared_ptr<std::vector<double>> pointsumsq_t_;

    //!--------------------------------------------------
    //!   averaged forces (mean, bottom and top)
    //!--------------------------------------------------

    //! sum over nodal forces on boundary in u direction
    double sumforceu_;
    //! sum over nodal forces on boundary in v direction
    double sumforcev_;
    //! sum over nodal forces on boundary in w direction
    double sumforcew_;

    //! sum over nodal forces on boundary in u direction
    double sumforcebu_;
    //! sum over nodal forces on boundary in v direction
    double sumforcebv_;
    //! sum over nodal forces on boundary in w direction
    double sumforcebw_;

    //! sum over nodal forces on boundary in u direction
    double sumforcetu_;
    //! sum over nodal forces on boundary in v direction
    double sumforcetv_;
    //! sum over nodal forces on boundary in w direction
    double sumforcetw_;

    //! heat flux on bottom and top boundary
    double sumqwb_;
    double sumqwt_;

    //!--------------------------------------------------
    //!  averaged quantities from dynamic Smagorinsky
    //!--------------------------------------------------

    //! sum over Cs --- used only for dynamic Smagorinsky model
    std::shared_ptr<std::vector<double>> sum_cs_;
    //! sum over (Cs*delta)^2 --- used only for dynamic Smagorinsky model
    std::shared_ptr<std::vector<double>> sum_cs_delta_sq_;
    //! sum over effective viscosity --- used only for dynamic Smagorinsky model
    std::shared_ptr<std::vector<double>> sumvisceff_;
    //! increment of sumCs over in one timestep
    std::shared_ptr<std::vector<double>> incrsum_cs_;
    //! increment of sumCs_delta_sq over in one timestep
    std::shared_ptr<std::vector<double>> incrsum_cs_delta_sq_;
    //! increment of sumvisceff over in one timestep
    std::shared_ptr<std::vector<double>> incrsumvisceff_;
    //! sum over Prt --- used only for dynamic Smagorinsky model
    std::shared_ptr<std::vector<double>> sum_prt_;
    //! sum over (Cs*delta)^2/Prt --- used only for dynamic Smagorinsky model
    std::shared_ptr<std::vector<double>> sum_cs_delta_sq_prt_;
    //! sum over effective diffusivity --- used only for dynamic Smagorinsky model
    std::shared_ptr<std::vector<double>> sumdiffeff_;
    //! increment of sumCs_delta_sq_Prt over in one timestep
    std::shared_ptr<std::vector<double>> incrsum_cs_delta_sq_prt_;
    //! increment of sumPrt over in one timestep
    std::shared_ptr<std::vector<double>> incrsum_prt_;
    //! increment of sumdiffeff over in one timestep
    std::shared_ptr<std::vector<double>> incrsumdiffeff_;
    //! sum over Ci --- used only for dynamic Smagorinsky model for loma
    std::shared_ptr<std::vector<double>> sum_ci_;
    //! sum over (Ci*delta)^2 --- used only for dynamic Smagorinsky model for loma
    std::shared_ptr<std::vector<double>> sum_ci_delta_sq_;
    //! increment of sumCi over in one timestep
    std::shared_ptr<std::vector<double>> incrsum_ci_;
    //! increment of sumCi_delta_sq over in one timestep
    std::shared_ptr<std::vector<double>> incrsum_ci_delta_sq_;

    //!--------------------------------------------------
    //!  averaged quantities from multifractal subgid-scales
    //!--------------------------------------------------

    //! sum over parameter N --- used only for multifractal subgid-scale model
    std::shared_ptr<std::vector<double>> sum_n_stream_;
    std::shared_ptr<std::vector<double>> sum_n_normal_;
    std::shared_ptr<std::vector<double>> sum_n_span_;
    //! increment of parameter N over in one time step
    std::shared_ptr<std::vector<double>> incrsum_n_stream_;
    std::shared_ptr<std::vector<double>> incrsum_n_normal_;
    std::shared_ptr<std::vector<double>> incrsum_n_span_;
    //! sum over parameter B --- used only for multifractal subgid-scale model
    std::shared_ptr<std::vector<double>> sum_b_stream_;
    std::shared_ptr<std::vector<double>> sum_b_normal_;
    std::shared_ptr<std::vector<double>> sum_b_span_;
    //! increment of parameter B over in one time step
    std::shared_ptr<std::vector<double>> incrsum_b_stream_;
    std::shared_ptr<std::vector<double>> incrsum_b_normal_;
    std::shared_ptr<std::vector<double>> incrsum_b_span_;
    //! sum over parameter Csgs --- used only for multifractal subgid-scale model
    std::shared_ptr<std::vector<double>> sum_csgs_;
    //! increment of Csgs over in one time step
    std::shared_ptr<std::vector<double>> incrsum_csgs_;
    //! sum over subgrid viscosity --- used only for multifractal subgid-scale model in combination
    //! with eddy viscosity model
    std::shared_ptr<std::vector<double>> sumsgvisc_;
    //! increment of subgrid viscosity over in one time step
    std::shared_ptr<std::vector<double>> incrsumsgvisc_;
    //! sum over parameter Nphi --- used only for multifractal subgid-scale model
    std::shared_ptr<std::vector<double>> sum_nphi_;
    //! increment of parameter Nphi over in one time step
    std::shared_ptr<std::vector<double>> incrsum_nphi_;
    //! sum over parameter Dphi --- used only for multifractal subgid-scale model
    std::shared_ptr<std::vector<double>> sum_dphi_;
    //! increment of parameter Dphi over in one time step
    std::shared_ptr<std::vector<double>> incrsum_dphi_;
    //! sum over parameter Csgs_phi --- used only for multifractal subgid-scale model
    std::shared_ptr<std::vector<double>> sum_csgs_phi_;
    //! increment of Csgs_phi over in one time step
    std::shared_ptr<std::vector<double>> incrsum_csgs_phi_;

    //!--------------------------------------------------
    //!  averaged resudiuals and subscale quantities
    //!--------------------------------------------------

    //! sum over all in plane element sizes
    std::shared_ptr<std::vector<double>> sumhk_;
    //! sum over all in plane element sizes for the Bazilevs
    //! parameter in the viscous regime
    std::shared_ptr<std::vector<double>> sumhbazilevs_;
    //! sum over all in plane stream lengths
    std::shared_ptr<std::vector<double>> sumstrle_;
    //! sum over all in plane stream lengths
    std::shared_ptr<std::vector<double>> sumgradle_;

    //! sum over all in plane residuals
    std::shared_ptr<std::vector<double>> sumtau_m_;
    //! sum over all in plane squared residuals
    std::shared_ptr<std::vector<double>> sumtau_c_;

    //! sum over all in plane mk (parameter for stabilization parameter, 1/3 for lin ele)
    std::shared_ptr<std::vector<double>> summk_;

    //! sum over all in plane residuals
    std::shared_ptr<std::vector<double>> sumres_;
    //! sum over all in plane squared residuals
    std::shared_ptr<std::vector<double>> sumres_sq_;
    //! sum over all in plane residuals norms
    std::shared_ptr<std::vector<double>> sumabsres_;
    //! sum over all in plane values of svel/tau
    std::shared_ptr<std::vector<double>> sumtauinvsvel_;
    //! sum over all in plane subscale velocities
    std::shared_ptr<std::vector<double>> sumsvelaf_;
    //! sum over all in plane squared subscale velocities
    std::shared_ptr<std::vector<double>> sumsvelaf_sq_;
    //! sum over all in plane subscale velocities norms
    std::shared_ptr<std::vector<double>> sumabssvelaf_;

    //! sum over all in plane residuals of the continuity equation
    std::shared_ptr<std::vector<double>> sumres_c_;
    //! sum over all in plane squared residuals of the continuity equation
    std::shared_ptr<std::vector<double>> sumres_c_sq_;
    //! sum over all in plane subscale pressure values at current timestep
    std::shared_ptr<std::vector<double>> sumspressnp_;
    //! sum over all in plane squared subscale pressure values at current timestep
    std::shared_ptr<std::vector<double>> sumspressnp_sq_;

    //! sum over all in plane averaged dissipation rates from pspg stabilisation
    std::shared_ptr<std::vector<double>> sum_eps_pspg_;
    //! sum over all in plane averaged dissipation rates from supg stabilisation
    std::shared_ptr<std::vector<double>> sum_eps_supg_;
    //! sum over all in plane averaged dissipation rates from cross term
    std::shared_ptr<std::vector<double>> sum_eps_cross_;
    //! sum over all in plane averaged dissipation rates from reynolds term
    std::shared_ptr<std::vector<double>> sum_eps_rey_;
    //! sum over all in plane averaged dissipation rates from least squares continuity term
    std::shared_ptr<std::vector<double>> sum_eps_graddiv_;
    //! sum over all in plane averaged dissipation rates from eddy viscosity model (Smagorinsky)
    std::shared_ptr<std::vector<double>> sum_eps_eddyvisc_;
    //! sum over all in plane averaged dissipation rates from eddy viscosity model (AVM3)
    std::shared_ptr<std::vector<double>> sum_eps_avm3_;
    //! sum over all in plane averaged dissipation rates from mfs model
    std::shared_ptr<std::vector<double>> sum_eps_mfs_;
    //! sum over all in plane averaged dissipation rates from mfs model (forwardscatter)
    std::shared_ptr<std::vector<double>> sum_eps_mfscross_;
    //! sum over all in plane averaged dissipation rates from mfs model (backscatter)
    std::shared_ptr<std::vector<double>> sum_eps_mfsrey_;
    //! sum over all in plane averaged dissipation rates from Galerkin viscous term
    std::shared_ptr<std::vector<double>> sum_eps_visc_;
    //! sum over all in plane averaged dissipation rates from Galerkin convective term
    std::shared_ptr<std::vector<double>> sum_eps_conv_;

    //! sum over all in plane averaged supg+cross stress
    std::shared_ptr<std::vector<double>> sum_crossstress_;
    //! sum over all in plane averaged reynolds stress
    std::shared_ptr<std::vector<double>> sum_reystress_;

    //!--------------------------------------------------
    //!  averaged resudiuals and subscale quantities of scalar field
    //!--------------------------------------------------

    //! sum over all in plane stabilization parameters
    std::shared_ptr<std::vector<double>> sumtau_s_;

    //! sum over all in plane residuals of the convection-diffusion equation
    std::shared_ptr<std::vector<double>> sumres_s_;
    //! sum over all in plane squared residuals of the convection-diffusion equation
    std::shared_ptr<std::vector<double>> sumres_s_sq_;

    //! sum over all in plane averaged dissipation rates from supg stabilisation
    std::shared_ptr<std::vector<double>> sum_scatra_eps_supg_;
    //! sum over all in plane averaged dissipation rates from cross term
    std::shared_ptr<std::vector<double>> sum_scatra_eps_cross_;
    //! sum over all in plane averaged dissipation rates from reynolds term
    std::shared_ptr<std::vector<double>> sum_scatra_eps_rey_;
    //! sum over all in plane averaged dissipation rates from eddy viscosity model (Smagorinsky)
    std::shared_ptr<std::vector<double>> sum_scatra_eps_eddyvisc_;
    //! sum over all in plane averaged dissipation rates from eddy viscosity model (AVM3)
    std::shared_ptr<std::vector<double>> sum_scatra_eps_avm3_;
    //! sum over all in plane averaged dissipation rates from mfs model
    std::shared_ptr<std::vector<double>> sum_scatra_eps_mfs_;
    //! sum over all in plane averaged dissipation rates from mfs model (forwardscatter)
    std::shared_ptr<std::vector<double>> sum_scatra_eps_mfscross_;
    //! sum over all in plane averaged dissipation rates from mfs model (backscatter)
    std::shared_ptr<std::vector<double>> sum_scatra_eps_mfsrey_;
    //! sum over all in plane averaged dissipation rates from Galerkin viscous term
    std::shared_ptr<std::vector<double>> sum_scatra_eps_visc_;
    //! sum over all in plane averaged dissipation rates from Galerkin convective term
    std::shared_ptr<std::vector<double>> sum_scatra_eps_conv_;

    //! number of sampling planes inside the element
    const int numsubdivisions_;
  };

}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif

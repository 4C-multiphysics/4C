#ifndef FOUR_C_FLUID_TURBULENCE_HIT_FORCING_HPP
#define FOUR_C_FLUID_TURBULENCE_HIT_FORCING_HPP

#include "4C_config.hpp"

#include "4C_inpar_fluid.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace FLD
{
  // forward declarations
  class FluidImplicitTimeInt;
  class XWall;

  class ForcingInterface
  {
   public:
    //! constructor
    ForcingInterface() {}

    virtual ~ForcingInterface() = default;
    //! initialize with initial spectrum
    virtual void set_initial_spectrum(Inpar::FLUID::InitialField init_field_type) = 0;

    //! turn on forcing
    virtual void activate_forcing(const bool activate) = 0;

    //! calculate power input
    virtual void calculate_forcing(const int step) = 0;

    //! get forcing
    virtual void update_forcing(const int step) = 0;

    //! time update of energy spectrum
    virtual void time_update_forcing() = 0;
  };

  class HomIsoTurbForcing : public ForcingInterface
  {
   public:
    //! constructor
    HomIsoTurbForcing(FluidImplicitTimeInt& timeint);

    //! initialize with initial spectrum
    void set_initial_spectrum(Inpar::FLUID::InitialField init_field_type) override;

    //! turn on forcing
    void activate_forcing(const bool activate) override;

    //! calculate power input
    void calculate_forcing(const int step) override;

    //! get forcing
    void update_forcing(const int step) override;

    //! time update of energy spectrum
    void time_update_forcing() override;

   protected:
    //! sort criterium for double values up to a tolerance of 10-9
    class LineSortCriterion
    {
     public:
      bool operator()(const double& p1, const double& p2) const { return (p1 < p2 - 1E-9); }

     protected:
     private:
    };

    //! type of forcing
    Inpar::FLUID::ForcingType forcing_type_;

    //! fluid discretization
    Teuchos::RCP<Core::FE::Discretization> discret_;

    //! state vector of volume force to be computed
    Teuchos::RCP<Core::LinAlg::Vector<double>> forcing_;

    //! state vectors used to compute forcing
    Teuchos::RCP<Core::LinAlg::Vector<double>> velnp_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> velaf_;

    //! threshold wave number for forcing
    //! i.e., forcing is applied to wave numbers <= threshold wave number
    double threshold_wavenumber_;

    //! identify gen-alpha time integration
    bool is_genalpha_;

    //! number of forcing time steps for decaying case
    int num_force_steps_;

    //! specifies the special flow
    enum FlowType
    {
      decaying_homogeneous_isotropic_turbulence,
      forced_homogeneous_isotropic_turbulence
    } flow_type_;

    //! number of resolved mode
    int nummodes_;

    //! vector of coordinates in one spatial direction (same for the other two directions)
    Teuchos::RCP<std::vector<double>> coordinates_;

    //! vector of wave numbers
    Teuchos::RCP<std::vector<double>> wavenumbers_;

    //! vector energy spectrum (sum over k=const) at time n
    Teuchos::RCP<std::vector<double>> energyspectrum_n_;

    //! vector energy spectrum  (sum over k=const) at time n+1/n+af
    Teuchos::RCP<std::vector<double>> energyspectrum_np_;

    //! time step length
    double dt_;

    //! flag to activate forcing
    bool activate_;

    //! linear compensation factor
    Teuchos::RCP<Core::LinAlg::SerialDenseVector> force_fac_;

    //! fixed power input
    double Pin_;

    //! energy_contained in lowest wave numbers
    double E_kf_;

    //! fixed power input factor
    Teuchos::RCP<Core::LinAlg::SerialDenseVector> fixed_power_fac_;

    //! interpolation function
    static double interpolate(
        const double& x, const double& x_1, const double& x_2, const double& y_1, const double& y_2)
    {
      const double value = y_1 + (y_2 - y_1) / (x_2 - x_1) * (x - x_1);
      return value;
    }
  };

  // there are quite some differences for the HDG case
  // author: bk 03/15
  class HomIsoTurbForcingHDG : public HomIsoTurbForcing
  {
   public:
    //! constructor
    HomIsoTurbForcingHDG(FluidImplicitTimeInt& timeint);

    //! initialize with initial spectrum
    void set_initial_spectrum(Inpar::FLUID::InitialField init_field_type) override;

    //! calculate power input
    void calculate_forcing(const int step) override;

    //! get forcing
    void update_forcing(const int step) override;
  };

  // this is an adaptive body force for the periodic hill benchmark such
  // that the mass flow reaches the given value.
  // author: bk 12/14
  class PeriodicHillForcing : public ForcingInterface
  {
   public:
    //! constructor
    PeriodicHillForcing(FluidImplicitTimeInt& timeint);

    //! initialize with initial spectrum
    void set_initial_spectrum(Inpar::FLUID::InitialField init_field_type) override { return; }

    //! turn on forcing
    void activate_forcing(const bool activate) override { return; }

    //! calculate power input
    void calculate_forcing(const int step) override { return; }

    //! get forcing
    void update_forcing(const int step) override;

    //! time update of energy spectrum
    void time_update_forcing() override;


   private:
    //! fluid discretization
    Teuchos::RCP<Core::FE::Discretization> discret_;

    //! state vector of volume force to be computed
    Teuchos::RCP<Core::LinAlg::Vector<double>> forcing_;

    //! state vectors used to compute forcing
    Teuchos::RCP<Core::LinAlg::Vector<double>> velnp_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> velaf_;

    //! xwall object is required for evaluating inner element planes of channel
    Teuchos::RCP<FLD::XWall> myxwall_;

    //! values of previous step
    double oldforce_;
    double oldflow_;

    //! reference value for optimal control
    double idealmassflow_;

    // length of overall flow domain
    double length_;

    //! step and statistical data
    int step_;
    int count_;
    double sum_;
  };

}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_RED_AIRWAYS_TISSUE_HPP
#define FOUR_C_RED_AIRWAYS_TISSUE_HPP

#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"
#include "4C_inpar_bio.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_red_airways_resulttest.hpp"
#include "4C_utils_result_test.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


// forward declarations

namespace Adapter
{
  class StructureRedAirway;
}

namespace Airway
{
  class RedAirwayImplicitTimeInt;

  class RedAirwayTissue : public Adapter::AlgorithmBase
  {
   public:
    /// Standard Constructor
    RedAirwayTissue(MPI_Comm comm, const Teuchos::ParameterList& timeparams);


    void read_restart(const int step) override;

    void setup_red_airways();

    /// time integration of coupled problem
    void integrate();

    void do_structure_step();

    void relax_pressure(int iter);

    void do_red_airway_step();

    /// flag whether iteration between fields should be finished
    bool not_converged(int iter);

    void output_iteration(Core::LinAlg::Vector<double>& pres_inc,
        Core::LinAlg::Vector<double>& scaled_pres_inc, Core::LinAlg::Vector<double>& flux_inc,
        Core::LinAlg::Vector<double>& scaled_flux_inc, int iter);

    void update_and_output();

    /// access to structural field
    std::shared_ptr<Adapter::StructureRedAirway>& structure_field() { return structure_; }

    /// access to airway field
    std::shared_ptr<RedAirwayImplicitTimeInt>& red_airway_field() { return redairways_; }


   private:
    /// underlying structure
    std::shared_ptr<Adapter::StructureRedAirway> structure_;

    std::shared_ptr<RedAirwayImplicitTimeInt> redairways_;

    /// redundant vector of outlet pressures (new iteration step)
    std::shared_ptr<Core::LinAlg::Vector<double>> couppres_ip_;

    /// redundant vector of outlet pressures (old iteration step)
    std::shared_ptr<Core::LinAlg::Vector<double>> couppres_im_;


    // Aitken Variables:
    // Relaxation factor
    std::shared_ptr<Core::LinAlg::Vector<double>> omega_np_;

    /// redundant vector of outlet pressures (before old iteration step), p^{i}_{n+1}
    std::shared_ptr<Core::LinAlg::Vector<double>> couppres_il_;

    /// redundant vector of outlet pressures (old iteration step guess), \tilde{p}^{i+1}_{n+1}
    std::shared_ptr<Core::LinAlg::Vector<double>> couppres_im_tilde_;

    /// redundant vector of outlet pressures (new iteration step guess), \tilde{p}^{i+2}_{n+1}
    std::shared_ptr<Core::LinAlg::Vector<double>> couppres_ip_tilde_;


    /// redundant vector of outlet fluxes (new iteration step)
    std::shared_ptr<Core::LinAlg::Vector<double>> coupflux_ip_;

    /// redundant vector of outlet fluxes (old iteration step)
    std::shared_ptr<Core::LinAlg::Vector<double>> coupflux_im_;

    /// redundant vector of 3D volumes (new iteration step)
    std::shared_ptr<Core::LinAlg::Vector<double>> coupvol_ip_;

    /// redundant vector of 3D volumes (old iteration step)
    std::shared_ptr<Core::LinAlg::Vector<double>> coupvol_im_;

    /// internal iteration step
    int itermax_;

    /// restart step
    int uprestart_;

    /// internal tolerance for pressure
    double tolp_;

    /// internal tolerance for flux
    double tolq_;

    /// defined normal direction
    double normal_;

    /// fixed relaxation parameter
    double omega_;

    /// fixed relaxation parameter
    Inpar::ArteryNetwork::Relaxtype3D0D relaxtype_;
  };
}  // namespace Airway

FOUR_C_NAMESPACE_CLOSE

#endif

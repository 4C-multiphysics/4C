// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FS3I_PARTITIONED_2WC_HPP
#define FOUR_C_FS3I_PARTITIONED_2WC_HPP


#include "4C_config.hpp"

#include "4C_fs3i_partitioned.hpp"

FOUR_C_NAMESPACE_OPEN


namespace FS3I
{
  class PartFS3I2Wc : public PartFS3I
  {
   public:
    PartFS3I2Wc(MPI_Comm comm);

    //! initialize this class
    void init() override;

    //! setup this class
    void setup() override;

    void timeloop() override;

    void initial_calculations();

    void prepare_time_step() override;

    void outer_loop();

    // void SetFSIValuesInScaTra();

    void set_scatra_values_in_fsi();

    bool convergence_check(int itnum);

    bool scatra_convergence_check(int itnum) override;

    void time_update_and_output();

   private:
    void validate_structure_scatra_clone_configuration(const Core::FE::Discretization& structdis,
        const Core::FE::Discretization& structscatradis,
        const std::map<std::pair<std::string, std::string>, std::map<int, int>>& clonefieldmatmap)
        const override;

    //! @name  (preliminary) maximum number of iterations and tolerance for outer iteration
    int itmax_;
    double ittol_;
    //@}

    /// flag for constant thermodynamic pressure
    std::string consthermpress_;

    /// fluid- and structure-based scalar transport problem
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> fluidscatra_;
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> structurescatra_;
  };
}  // namespace FS3I

FOUR_C_NAMESPACE_CLOSE

#endif

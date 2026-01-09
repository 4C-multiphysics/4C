// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_MONITOR_DBC_HPP
#define FOUR_C_STRUCTURE_NEW_MONITOR_DBC_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_io_yaml.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_map.hpp"
#include "4C_structure_new_monitor_dbc_input.hpp"

#include <mpi.h>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class RuntimeCsvWriter;
  class DiscretizationWriter;
}  // namespace Core::IO
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE
namespace Solid
{
  class Dbc;
  namespace TimeInt
  {
    class BaseDataGlobalState;
    class BaseDataIO;
  }  // namespace TimeInt

  /** \brief Monitor Dirichlet boundary conditions
   *
   *  This class can be used to monitor e.g. the reaction forces and the area
   *  change of a tagged Dirichlet condition during a simulation. To tag a Dirichlet
   *  condition just add the corresponding TAG, e.g. \"monitor_reaction\"
   *
   *  E 1 - NUMDOF 3 ONOFF 1 0 0 VAL 0.0 0.0 0.0 FUNCT 0 0 0 TAG monitor_reaction
   *
   *  If the TAG can be found for any Dirichlet condition the reaction force as
   *  well as the reference and current area will be stored in a text file
   *  located at
   *
   *  <OUTPUT_PATH>/<OUTPUT_FILE_NAME>_monitor_dbc/<ID>_monitor_dbc.data
   *
   *  If no tag is found nothing is happening.
   *
   *  */
  class MonitorDbc
  {
    const static unsigned DIM = 3;

    /// constants for the SCREEN output
    const static unsigned OS_WIDTH = 14;

   public:
    MonitorDbc() = default;

    /// initialize class members
    void init(const std::shared_ptr<Solid::TimeInt::BaseDataIO>& io_ptr,
        Core::FE::Discretization& discret, Solid::TimeInt::BaseDataGlobalState& gstate,
        Solid::Dbc& dbc);

    /// setup new class members
    void setup();

    /// monitor the tensile test results and write them to a text file
    void execute(Core::IO::DiscretizationWriter& writer);

   private:
    int get_unique_id(int tagged_id, Core::Conditions::GeometryType gtype) const;

    /**
     * @brief Read in existing yaml file for restart and remove steps beyond restart step
     *
     * The opened yaml writer is appended to the dbc_monitor_yaml_file_trees_ vector.
     *
     * @param rcond Condition for which the restart file shall be read
     */
    void read_restart_yaml_file(const Core::Conditions::Condition& rcond);

    void create_reaction_force_condition(
        const Core::Conditions::Condition& tagged_cond, Core::FE::Discretization& discret) const;

    void get_tagged_condition(std::vector<const Core::Conditions::Condition*>& tagged_conds,
        const std::string& cond_name, const std::string& tag_name,
        const Core::FE::Discretization& discret) const;

    void create_reaction_maps(const Core::FE::Discretization& discret,
        const Core::Conditions::Condition& rcond,
        std::shared_ptr<Core::LinAlg::Map>* react_maps) const;

    void get_area(double area_ref[], const Core::Conditions::Condition* rcond) const;

    double get_reaction_force(Core::LinAlg::Matrix<3, 1>& rforce_xyz,
        const std::shared_ptr<Core::LinAlg::Map>* react_maps) const;

    double get_reaction_moment(Core::LinAlg::Matrix<3, 1>& rmoment_xyz,
        const std::shared_ptr<Core::LinAlg::Map>* react_maps,
        const Core::Conditions::Condition* rcond) const;

    void write_condition_header(std::ostream& os, const int col_width,
        const Core::Conditions::Condition* cond = nullptr) const;

    void write_column_header(std::ostream& os, const int col_width) const;

    void write_results_to_screen(const Core::Conditions::Condition& rcond_ptr,
        const Core::LinAlg::Matrix<DIM, 1>& rforce, const Core::LinAlg::Matrix<DIM, 1>& rmoment,
        const double& area_ref, const double& area_curr) const;

    void write_results(std::ostream& os, const int col_width, const int precision,
        const unsigned step, const double time, const Core::LinAlg::Matrix<DIM, 1>& rforce,
        const Core::LinAlg::Matrix<DIM, 1>& rmoment, const double& area_ref,
        const double& area_cur) const;

    inline MPI_Comm get_comm() const;

    inline void throw_if_not_init() const { FOUR_C_ASSERT(isinit_, "Call init() first!"); }

    inline void throw_if_not_setup() const { FOUR_C_ASSERT(issetup_, "Call setup() first!"); }

    Core::FE::Discretization* discret_ptr_ = nullptr;
    Solid::TimeInt::BaseDataGlobalState* gstate_ptr_ = nullptr;
    Solid::Dbc* dbc_ptr_ = nullptr;

    std::vector<std::string> full_filepaths_ = std::vector<std::string>();

    /// extract the dofs of the reaction forces which shall be monitored
    std::map<int, std::vector<std::shared_ptr<Core::LinAlg::Map>>> react_maps_;
    int os_precision_ = -1;
    std::vector<std::unique_ptr<Core::IO::RuntimeCsvWriter>> dbc_monitor_csvwriter_;
    std::vector<ryml::Tree> dbc_monitor_yaml_file_trees_;
    IOMonitorStructureDBC::FileType file_type_;

    bool isempty_ = true;
    bool isinit_ = false;
    bool issetup_ = false;
  };
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif

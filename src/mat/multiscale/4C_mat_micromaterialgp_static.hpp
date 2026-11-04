// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MICROMATERIALGP_STATIC_HPP
#define FOUR_C_MAT_MICROMATERIALGP_STATIC_HPP



#include "4C_config.hpp"

#include "4C_io_control.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"

#include <map>
#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations

namespace MultiScale
{
  class MicroStatic;
}

namespace Core::IO
{
  class DiscretizationWriter;
}

namespace Mat
{
  /// one Gauss point of the micro material
  class MicroMaterialGP
  {
   public:
    /// construct an instance of MicroMaterial for a given Gauss point
    /// and microscale discretization
    MicroMaterialGP(const int gp, const int ele_ID, const bool eleowner, const int microdisnum,
        const double V0, const bool initialize_runtime_output_writer);

    /// destructor
    ~MicroMaterialGP();

    /// Read restart
    void read_restart();

    /// New result file
    void new_result_file(
        bool eleowner, std::string& newfilename, bool initialize_runtime_output_writer);

    /// create path of new result file
    std::string new_result_file_path(const std::string& newprefix) const;

    /// Post setup to set time and step properly
    void post_setup();

    /// Perform microscale simulation
    void perform_micro_simulation(const Core::LinAlg::Matrix<3, 3>* defgrd,
        Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat);

    /// Update time, step, displacement and element internal history data
    void update();

    /// Calculate stresses and strains on the micro-scale
    void runtime_pre_output_step_state();

    /// write runtime output on the micro-scale
    void runtime_output_step_state_microscale(
        const std::pair<double, int>& output_time_and_step, const std::string& section_name);

    /// write restart on the micro-scale
    void write_restart();

    /// save micro scale element history data
    void extract_and_store_history_data();

    /// set micro scale element history data
    void fill_history_data_into_elements();

    /// get ele id
    int ele_id() const { return ele_id_; }

    /// get density
    double density() const { return density_; }


   private:
    /// corresponding macroscale Gauss point
    const int gp_;

    /// corresponding macroscale element
    const int ele_id_;

    /// corresponding microstructure discretization number
    const int microdisnum_;

    /// microstructure discretization writer (for restart output)
    std::shared_ptr<Core::IO::DiscretizationWriter> micro_output_;

    /// microstructure runtime output writer
    std::shared_ptr<Core::IO::DiscretizationVisualizationWriterMesh> micro_visualization_writer_;

    std::optional<Core::IO::OutputControl> micro_output_control_;

    /// homogenized density
    double density_;

    /// my vector of old displacements
    std::shared_ptr<Core::LinAlg::Vector<double>> dis_;

    /// my vector of new displacements
    std::shared_ptr<Core::LinAlg::Vector<double>> disn_;

    /// history data for each column element on microscale
    std::unordered_map<int, std::vector<char>> history_data_;

    /// my stresses and strains
    std::shared_ptr<std::vector<char>> plstrain_;
    std::shared_ptr<Core::LinAlg::MultiVector<double>> stress_data_node_postprocessed_;
    std::shared_ptr<Core::LinAlg::MultiVector<double>> stress_data_element_postprocessed_;
    std::shared_ptr<Core::LinAlg::MultiVector<double>> strain_data_node_postprocessed_;
    std::shared_ptr<Core::LinAlg::MultiVector<double>> strain_data_element_postprocessed_;

    /// Averaged tangent stiffness tensor
    std::shared_ptr<Core::LinAlg::Matrix<6, 6>> macro_cmat_output_;

    /// old absolute time
    double time_;

    /// current absolute time
    double timen_;

    /// old step
    int step_;

    /// current step
    int stepn_;

    /// timestep size
    double dt_;

    /// restart name
    std::string restartname_;

    /// flag for modified Newton on macroscale
    bool mod_newton_;

    /// flag for build of stiffness matrix
    bool build_stiff_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

#ifndef FOUR_C_IO_VISUALIZATION_MANAGER_HPP
#define FOUR_C_IO_VISUALIZATION_MANAGER_HPP

#include "4C_config.hpp"

#include "4C_io_visualization_data.hpp"
#include "4C_io_visualization_parameters.hpp"
#include "4C_io_visualization_writer_base.hpp"

#include <Epetra_Comm.h>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  /**
   * @brief This class defines a container that eases the management of visualization data and
   * writers
   *
   * This class can be used to manage either a "single" visualization data set, e.g., a mesh plus
   * field values, or multiple data sets that belong together, e.g., contact forces on the nodes and
   * segmentation visualization. From a vtu point of view, the last example does not fit into a
   * single vtu grid, since there are different fields defined on them, but they can be grouped in
   * this object to ease management of the data.
   *
   * In the context of this class and the managed classes therein, "visualization" refers to data
   * that will be written to disk to visualize the simulation results, e.g., in ParaView. This can
   * be either direct simulation results such as nodal values of post-processed results such as
   * stresses. The functionality here is only intended for visualization and post-processing and is
   * completely independent from writing restart data, that stores the current state of the
   * simulation.
   */
  class VisualizationManager
  {
   public:
    /**
     * @brief Default constructor
     *
     * @param parameters (in) Parameter container
     * @param comm (in) MPI communicator
     * @param base_output_name (in) Base name of this output data. For example, if the
     * base_output_name is "contact" and the registered visualization data names are "", "forces"
     * and "segmentation", then this object will create the following output files:
     * - "contact"
     * - "contact_forces"
     * - "contact_segmentation"
     */
    VisualizationManager(Core::IO::VisualizationParameters parameters, const Epetra_Comm& comm,
        std::string base_output_name);

    /**
     * @brief Return a const reference to the visualization data
     *
     * @param visualization_data_name (in) Name of the visualization data, if this argument is
     * empty, the default visualization data, i.e., "", will be returned.
     */
    [[nodiscard]] const VisualizationData& get_visualization_data(
        const std::string& visualization_data_name = "") const;

    /**
     * @brief Return a mutable reference to the visualization data
     *
     * @param visualization_data_name (in) Name of the visualization data, if this argument is
     * empty, the default visualization data, i.e., "", will be returned.
     */
    [[nodiscard]] VisualizationData& get_visualization_data(
        const std::string& visualization_data_name = "");

    /**
     * @brief Register a visualization data name and return a reference to the visualization data
     *
     * This method creates the visualization data and the corresponding visualization writer. It is
     * not required to register the default visualization data, i.e., visualization_data_name == "".
     *
     * @param visualization_data_name (in) Name of the visualization data
     */
    VisualizationData& register_visualization_data(const std::string& visualization_data_name);

    /**
     * @brief Check if a given visualization data name already exists
     */
    [[nodiscard]] bool visualization_data_exists(const std::string& visualization_data_name) const;

    /**
     * @brief Clear all visualization data from this container. The registered visualization data
     * names will remain.
     */
    void clear_data();

    /**
     * @brief Write all contained visualization data containers to disk
     *
     * @param visualziation_time (in) Time value of current step, this is not necessarily the same
     * as the simulation time
     * @param visualization_step (in) Time step counter of current time step (does not have to be
     * continuous) this is not necessarily the same as the simulation time step counter
     */
    void write_to_disk(const double visualziation_time, const int visualization_step);

   private:
    /**
     * @brief Return the output data name corresponding to a visualization data name
     *
     * @param visualization_data_name (in) Name of the visualization data
     */
    [[nodiscard]] std::string get_visualization_data_name_for_output_files(
        const std::string& visualization_data_name) const;

   private:
    //! Visualization parameters
    const Core::IO::VisualizationParameters parameters_;

    //! MPI communicator
    const Epetra_Comm& comm_;

    //! The visualization data containers
    std::map<std::string, std::pair<VisualizationData, std::unique_ptr<VisualizationWriterBase>>>
        visualization_map_;

    //! Base name of this output data
    const std::string base_output_name_;
  };
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
/*----------------------------------------------------------------------*/
/*! \file

\brief Object to store the beam to solid volume meshtying output (visualization) parameters.

\level 3

*/
// End doxygen header.


#ifndef BACI_BEAMINTERACTION_BEAM_TO_SOLID_VOLUME_MESHTYING_VISUALIZATION_OUTPUT_PARAMS_HPP
#define BACI_BEAMINTERACTION_BEAM_TO_SOLID_VOLUME_MESHTYING_VISUALIZATION_OUTPUT_PARAMS_HPP


#include "baci_config.hpp"

#include "baci_utils_exceptions.hpp"

#include <memory>

BACI_NAMESPACE_OPEN


namespace BEAMINTERACTION
{
  class BeamToSolidVolumeMeshtyingVisualizationOutputParams
  {
   public:
    /**
     * \brief Constructor.
     */
    BeamToSolidVolumeMeshtyingVisualizationOutputParams();

    /**
     * \brief Destructor.
     */
    virtual ~BeamToSolidVolumeMeshtyingVisualizationOutputParams() = default;

    /**
     * \brief Initialize with the stuff coming from input file.
     */
    void Init();

    /**
     * \brief Setup member variables.
     */
    void Setup();

    /**
     * \brief Return the output every iteration flag.
     */
    bool GetOutputEveryIteration() const { return output_every_iteration_; }

    /**
     * \brief Output interval regarding steps: write output every INTERVAL_STEPS steps.
     */
    int OutputIntervalInSteps() const
    {
      CheckInitSetup();
      return output_interval_steps_;
    };

    /**
     * \brief Return the output flag.
     */
    bool GetOutputFlag() const { return output_flag_; }

    /**
     * \brief Return the nodal forces flag.
     */
    bool GetNodalForceOutputFlag() const { return nodal_forces_; }

    /**
     * \brief Return the mortar lambda discrete flag.
     */
    bool GetMortarLambdaDiscretOutputFlag() const { return mortar_lambda_discret_; }

    /**
     * \brief Return the mortar lambda continuous flag.
     */
    bool GetMortarLambdaContinuousOutputFlag() const { return mortar_lambda_continuous_; }

    /**
     * \brief Return the number of segments for continuous mortar output.
     */
    unsigned int GetMortarLambdaContinuousSegments() const
    {
      return mortar_lambda_continuous_segments_;
    }

    /**
     * \brief Return the nodal forces flag.
     */
    bool GetSegmentationOutputFlag() const { return segmentation_; }

    /**
     * \brief Return the integration output flag.
     */
    bool GetIntegrationPointsOutputFlag() const { return integration_points_; }

    /**
     * \brief Return the write unique IDs output flag.
     */
    bool GetWriteUniqueIDsFlag() const { return write_unique_ids_; }

   protected:
    /**
     * \brief Checks the init and setup status.
     */
    inline void CheckInitSetup() const
    {
      if (!isinit_ or !issetup_) dserror("Call Init() and Setup() first!");
    }

    /**
     * \brief Checks the init status.
     */
    inline void CheckInit() const
    {
      if (!isinit_) dserror("Init() has not been called, yet!");
    }

    //! Flag if object is initialized.
    bool isinit_;

    //! Flag if object is set up.
    bool issetup_;

    //! Output interval regarding steps: write output every INTERVAL_STEPS steps.
    int output_interval_steps_;

    //! Whether to write output in every iteration of the nonlinear solver.
    bool output_every_iteration_;

    //! Flag whether or not to write output.
    bool output_flag_;

    //! Flag whether or not to output resulting nodal forces.
    bool nodal_forces_;

    //! Flag whether or not to output resulting discrete Lagrange multiplier values for mortar
    //! pairs.
    bool mortar_lambda_discret_;

    //! Flag whether or not to output resulting continuous Lagrange multiplier function for mortar
    //! pairs.
    bool mortar_lambda_continuous_;

    //! Number of segments to use for the continuous mortar output.
    unsigned int mortar_lambda_continuous_segments_;

    //! Flag whether or not to write segmentation data.
    bool segmentation_;

    //! Flag whether or not to write data at integration points.
    bool integration_points_;

    //! Flag whether or not to write unique cell and point data IDs.
    bool write_unique_ids_;
  };

}  // namespace BEAMINTERACTION

BACI_NAMESPACE_CLOSE

#endif
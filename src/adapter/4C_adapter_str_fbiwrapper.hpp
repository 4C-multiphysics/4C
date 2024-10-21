#ifndef FOUR_C_ADAPTER_STR_FBIWRAPPER_HPP
#define FOUR_C_ADAPTER_STR_FBIWRAPPER_HPP

#include "4C_config.hpp"

#include "4C_adapter_str_fsiwrapper.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Solid
{
  namespace TimeInt
  {
    class ParamsRuntimeOutput;
  }
}  // namespace Solid
namespace Core::LinAlg
{
  class MultiMapExtractor;
}  // namespace Core::LinAlg

namespace BEAMINTERACTION
{
  class BeamToSolidVolumeMeshtyingVisualizationOutputParams;
}
namespace Adapter
{
  class FBIStructureWrapper : public FSIStructureWrapper
  {
   public:
    /// constructor
    explicit FBIStructureWrapper(Teuchos::RCP<Structure> structure);

    /// extracts interface velocities at \f$t_{n}\f$
    virtual Teuchos::RCP<Core::LinAlg::Vector<double>> extract_interface_veln();

    /// extracts interface velocities at \f$t_{n+1}\f$
    virtual Teuchos::RCP<Core::LinAlg::Vector<double>> extract_interface_velnp();

    /// Predictor for interface velocities
    virtual Teuchos::RCP<Core::LinAlg::Vector<double>> predict_interface_velnp();

    /** \brief linear structure solve with just a interface load
     *
     * Overloads RelaxationSolve of base class with an error message, because it is not implemented
     * for fluid-beam interaction yet
     */
    Teuchos::RCP<Core::LinAlg::Vector<double>> relaxation_solve(
        Teuchos::RCP<Core::LinAlg::Vector<double>> iforce) override;

    /// switch structure field to block matrix in fsi simulations
    void use_block_matrix() override { FOUR_C_THROW("Not yet implemented\n"); };

    /// extract interface displacements at \f$t_{n}\f$
    Teuchos::RCP<Core::LinAlg::Vector<double>> extract_interface_dispn() override;

    /// extract interface displacements at \f$t_{n+1}\f$
    Teuchos::RCP<Core::LinAlg::Vector<double>> extract_interface_dispnp() override;

    /// Predictor for interface displacements
    Teuchos::RCP<Core::LinAlg::Vector<double>> predict_interface_dispnp() override;

    /** \brief Apply interface forces to structural solver
     *
     * This prepares a new solve of the structural field within one time
     * step. This implementation overloads the base class function because we are not using a
     * condition for the interface, since we expect all beam elements to be immersed.
     */
    void apply_interface_forces(Teuchos::RCP<Core::LinAlg::Vector<double>> iforce) override;

    /// rebuild FSI interface from structure side
    void rebuild_interface() override;

    /// Setup the multi map extractor after ghosting of the structure discretization
    virtual void setup_multi_map_extractor();

    /// Get Runtime Output data
    virtual Teuchos::RCP<const Solid::TimeInt::ParamsRuntimeOutput> get_io_data();

   private:
    /// Map extractor seperating the beam elements from the structure elements
    Teuchos::RCP<Core::LinAlg::MultiMapExtractor> eletypeextractor_;

  };  // class FSIStructureWrapper
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif

/*----------------------------------------------------------------------*/
/*! \file

\brief Abstract class to be overloaded by different constraint enforcement techniques for fluid-beam
interaction.

\level 2

*----------------------------------------------------------------------*/
#ifndef FOUR_C_FBI_CONSTRAINTENFORCER_HPP
#define FOUR_C_FBI_CONSTRAINTENFORCER_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <Epetra_Vector.h>
#include <Teuchos_RCP.hpp>

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ADAPTER
{
  class FSIStructureWrapper;
  class FluidMovingBoundary;

}  // namespace ADAPTER

namespace BEAMINTERACTION
{
  class BeamToFluidMeshtyingVtkOutputWriter;
}

namespace BINSTRATEGY
{
  class BinningStrategy;
}
namespace DRT
{
  class Discretization;
  class Element;
}  // namespace DRT
namespace FBI
{
  class FBIGeometryCoupler;
}
namespace CORE::LINALG
{
  class SparseMatrix;
  class SparseOperator;
  class MapExtractor;
}  // namespace CORE::LINALG
namespace ADAPTER
{
  class ConstraintEnforcerFactory;
  class FBIConstraintBridge;

  /**
   * \brief Abstract class to be overloaded by different constraint enforcement techniques for
   * fluid-beam interaction
   *
   * Depending on the constraint enforcement technique used to couple embedded meshes, e.g. through
   * the penalty method, Lagrange multiplier method, Nitsche method, ect., very different
   * information have to be passed to the participating fields.
   * This class is designed to decouple the decision of which information to pass from the actual
   * (partitioned) algorithm.
   *
   * The interface to the outside world is just a Setup and Evaluate routine, as well as the
   * FluidToStruct() and StructToFluid() routines, which only return one vector, as customary for
   * the FSI::DirichletNeumann algorithm. Everything else, like contributions to the stiffness
   * matrix of the fluid, it provides by calling internal functions. To make this possible, the
   * constraint enforcer need information on the two field.
   *
   */
  class FBIConstraintenforcer
  {
    friend ConstraintEnforcerFactory;
    friend class BEAMINTERACTION::BeamToFluidMeshtyingVtkOutputWriter;

   public:
    /// empty destructor
    virtual ~FBIConstraintenforcer() = default;

    /**
     * \brief Sets up the constraint enforcer
     *
     *\param[in] structure wrapper for the structure solver
     *\param[in] fluid moving boundary wrapper for the fluid solver
     */
    virtual void Setup(Teuchos::RCP<ADAPTER::FSIStructureWrapper> structure,
        Teuchos::RCP<ADAPTER::FluidMovingBoundary> fluid);

    /** \brief Hand the binning strategy used for the distribution of the fluid mesh
     *  to the object responsible for the element pair search in the FBI framework
     *
     *  \param[in] binning binning strategy object
     */
    void SetBinning(Teuchos::RCP<BINSTRATEGY::BinningStrategy> binning);

    /**
     * \brief Computes the coupling matrices
     *
     * This is where the magic happens. The stiffness contributions are integrated using information
     * of the structure elements, the fluid elements and their position relative to each other
     */

    virtual void Evaluate();

    /**
     * \brief Recomputes all coupling related quantities without performing a search
     */
    virtual void RecomputeCouplingWithoutPairCreation();

    /**
     * \brief Abstractly, we do everything we have to, to introduce the coupling condition into the
     * structure field.
     *
     * Depending on the constraint enforcement strategy, either only an interface force is returned
     * (Mortar-Lagrangemultiplier partitioned, linearized penalty force partitioned), or a force
     * vector as well as a stiffness matrix with additional
     * information is returned (monolithic formulation, full penalty partitioned).
     *
     *
     * \returns structure force vector
     */

    virtual Teuchos::RCP<Epetra_Vector> FluidToStructure();

    /**
     * \brief Abstractly, we do everything we have to, to introduce the coupling condition into the
     * slave field.
     *
     * Depending on the constraint enforcement strategy, either only an interface force is returned
     * (Mortar-Lagrangemultiplier partitioned, linearized penalty force partitioned), or force
     * vector with additional contributions as well as a stiffness matrix with additional
     * information is returned (monolithic formulation, full penalty partitioned, weak Dirichlet).
     *
     *
     * \returns fluid velocity on the whole domain
     */

    virtual Teuchos::RCP<Epetra_Vector> StructureToFluid(int step);

    /// Interface to do preparations to solve the fluid
    virtual void PrepareFluidSolve() = 0;

    /// Get function for the structure field #structure_
    Teuchos::RCP<const ADAPTER::FSIStructureWrapper> GetStructure() const { return structure_; };

    /// Get function for the bridge object #bridge_
    Teuchos::RCP<const ADAPTER::FBIConstraintBridge> GetBridge() const { return bridge_; };

    /// Handle fbi specific output
    virtual void Output(double time, int step) = 0;

   protected:
    /** \brief You will have to use the ADAPTER::ConstraintEnforcerFactory
     *
     * \param[in] bridge an object managing the pair contributins
     * \param[in] geometrycoupler an object managing the search, parallel communication, ect.
     */
    FBIConstraintenforcer(Teuchos::RCP<ADAPTER::FBIConstraintBridge> bridge,
        Teuchos::RCP<FBI::FBIGeometryCoupler> geometrycoupler);

    /**
     * \brief Creates all possible interaction pairs
     *
     * \param[in] pairids a map relating all beam element ids to a set of fluid
     * elements ids which they potentially cut
     */
    void CreatePairs(Teuchos::RCP<std::map<int, std::vector<int>>> pairids);

    /**
     * \brief Resets the state, i.e. the velocity of all interaction pairs
     */
    void ResetAllPairStates();

    /**
     * \brief Extracts current element dofs that are needed for the computations on pair level
     *
     *\param[in] elements elements belonging to the pair
     *\param[out] beam_dofvec current positions and velocities of the beam element
     *\param[out] fluid_dofvec current positions and velocities of the fluid element
     */
    virtual void ExtractCurrentElementDofs(std::vector<DRT::Element const*> elements,
        std::vector<double>& beam_dofvec, std::vector<double>& fluid_dofvec) const;

    /**
     * \brief Computes the contributions to the stiffness matrix of the fluid field.
     *
     * This has to be implemented differently depending on the concrete constraint enforcement
     * strategy.
     *
     * \returns coupling contributions to the fluid system matrix
     */
    virtual Teuchos::RCP<const CORE::LINALG::SparseOperator> AssembleFluidCouplingMatrix() const
    {
      FOUR_C_THROW("Not yet implemented! This has to be overloaded by a derived class.\n");
      return Teuchos::null;
    };

    /**
     * \brief Computes the contributions to the stiffness matrix of the structure field.
     *
     * This has to be implemented differently depending on the concrete constraint enforcement
     * strategy.
     *
     * \returns coupling contributions to the structure system matrix
     */
    virtual Teuchos::RCP<const CORE::LINALG::SparseMatrix> AssembleStructureCouplingMatrix() const
    {
      FOUR_C_THROW("Not yet implemented! This has to be overloaded by a derived class.\n");
      return Teuchos::null;
    };

    /**
     * \brief Computes the contributions to the rhs of the structure field.
     *
     * This has to be implemented differently depending on the concrete constraint enforcement
     * strategy.
     *
     * \returns coupling contributions to the structure residual
     */
    virtual Teuchos::RCP<Epetra_Vector> AssembleStructureCouplingResidual() const
    {
      FOUR_C_THROW("Not yet implemented! This has to be overloaded by a derived class.\n");
      return Teuchos::null;
    };

    /**
     * \brief Computes the contributions to the residuum of the fluid field.
     *
     * This has to be implemented differently depending on the concrete constraint enforcement
     * strategy.
     *
     * \returns coupling contributions to the fluid residual
     */
    virtual Teuchos::RCP<Epetra_Vector> AssembleFluidCouplingResidual() const
    {
      FOUR_C_THROW("Not yet implemented! This has to be overloaded by a derived class.\n");
      return Teuchos::null;
    };

    /// Get function for the fluid field #fluid_
    Teuchos::RCP<ADAPTER::FluidMovingBoundary> GetFluid() const { return fluid_; };

    /// Get function for the structure and the fluid discretization in the vector #discretizations_
    std::vector<Teuchos::RCP<DRT::Discretization>> GetDiscretizations() const
    {
      return discretizations_;
    }

    /// Get function for the bridge object #bridge_
    Teuchos::RCP<ADAPTER::FBIConstraintBridge> Bridge() const { return bridge_; };

    /// Get map extractor to split fluid velocity and pressure values
    Teuchos::RCP<const CORE::LINALG::MapExtractor> GetVelocityPressureSplitter() const
    {
      return velocity_pressure_splitter_;
    }

   private:
    FBIConstraintenforcer() = delete;

    /// underlying fluid of the FSI problem
    Teuchos::RCP<ADAPTER::FluidMovingBoundary> fluid_;

    /// underlying structure of the FSI problem
    Teuchos::RCP<ADAPTER::FSIStructureWrapper> structure_;

    /// Vector containing both (fluid and structure) field discretizations
    std::vector<Teuchos::RCP<DRT::Discretization>> discretizations_;

    /**
     * \brief Object bridging the gap between the specific implementation of the constraint
     * enforcement technique and the specific implementation of the meshtying discretization
     * approach
     */
    Teuchos::RCP<ADAPTER::FBIConstraintBridge> bridge_;

    /**
     * \brief Object handling geometric operations like the search of embedded pairs as well as the
     * parallel communication
     */
    Teuchos::RCP<FBI::FBIGeometryCoupler> geometrycoupler_;

    /// Displacement of the structural column nodes on the current proc
    Teuchos::RCP<const Epetra_Vector> column_structure_displacement_;
    /// Velocity of the structural column nodes on the current proc
    Teuchos::RCP<const Epetra_Vector> column_structure_velocity_;
    /// Velocity of the fluid column nodes on the current proc
    Teuchos::RCP<const Epetra_Vector> column_fluid_velocity_;
    /**
     * \brief Extractor to split fluid values into velocities and pressure DOFs
     *
     * velocities  = OtherVector
     * pressure    = CondVector
     */
    Teuchos::RCP<CORE::LINALG::MapExtractor> velocity_pressure_splitter_;
  };
}  // namespace ADAPTER
FOUR_C_NAMESPACE_CLOSE

#endif

/*----------------------------------------------------------------------*/
/*! \file
\brief  Pure Virtual Coupling Manager, to define the basic functionality of all specified coupling
managers

\level 2


*----------------------------------------------------------------------*/

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#ifndef FOUR_C_FSI_XFEM_COUPLING_MANAGER_HPP
#define FOUR_C_FSI_XFEM_COUPLING_MANAGER_HPP

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class BlockSparseMatrixBase;
  class MultiMapExtractor;
}  // namespace Core::LinAlg

namespace Core::IO
{
  class DiscretizationWriter;
  class DiscretizationReader;
}  // namespace Core::IO

namespace XFEM
{
  class CouplingManager
  {
   public:
    //! @name Destruction
    //@{

    /// virtual to get polymorph destruction
    virtual ~CouplingManager() = default;
    //! predict states in the coupling object
    virtual void predict_coupling_states() = 0;

    //! Set required states in the coupling object
    virtual void set_coupling_states() = 0;

    //! Initializes the couplings (done at the beginning of the algorithm after fields have their
    //! state for timestep n)
    virtual void init_coupling_states() = 0;

    //! Add the coupling matrixes to the global systemmatrix
    // in ... idx[0] first discretization index , idx[1] second discretization index, ... in the
    // blockmatrix in ... scaling between xfluid evaluated coupling matrixes and coupled
    // systemmatrix
    virtual void add_coupling_matrix(
        Core::LinAlg::BlockSparseMatrixBase& systemmatrix, double scaling) = 0;

    //! Add the coupling rhs
    // in ... idx[0] first discretization index , idx[1] second discretization index, ... in the
    // blockmatrix in ... scaling between xfluid evaluated coupling matrixes and coupled
    // systemmatrix
    virtual void add_coupling_rhs(Teuchos::RCP<Core::LinAlg::Vector> rhs,
        const Core::LinAlg::MultiMapExtractor& me, double scaling) = 0;

    //! Update (Perform after Each Timestep)
    virtual void update(double scaling) = 0;

    //! Write Output (For restart or write results on the interface)
    virtual void output(Core::IO::DiscretizationWriter& writer) = 0;

    //! Read Restart (For quantities stored on the interface)
    virtual void read_restart(Core::IO::DiscretizationReader& reader) = 0;
  };
}  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif

/*----------------------------------------------------------------------*/
/*! \file
\brief General algorithmic routines for partitioned solution approaches
       to fluid-porous-structure-scalar-scalar interaction (FPS3I), that is,
       algorithmic routines not specifically related to partitioned
       solution approaches to one -or two-way-coupled problem
       configurations, respectively.

\level 3



*----------------------------------------------------------------------*/


#ifndef FOUR_C_FS3I_FPS3I_PARTITIONED_HPP
#define FOUR_C_FS3I_FPS3I_PARTITIONED_HPP


#include "4C_config.hpp"

#include "4C_fs3i.hpp"

FOUR_C_NAMESPACE_OPEN


// forward declarations
namespace FPSI
{
  class MonolithicPlain;
}

namespace Adapter
{
  class ScaTraBaseAlgorithm;
}  // namespace Adapter

namespace FSI
{
  class Monolithic;

  namespace Utils
  {
    class MatrixRowTransform;
    class MatrixColTransform;
    class MatrixRowColTransform;
  }  // namespace Utils
}  // namespace FSI

namespace Core::LinAlg
{
  class MultiMapExtractor;
  class BlockSparseMatrixBase;
  class SparseMatrix;
  class Solver;
}  // namespace Core::LinAlg


namespace FS3I
{
  class PartFPS3I : public FS3IBase
  {
   public:
    //! constructor of base class for partitioned FPS3I
    PartFPS3I(const Epetra_Comm& comm);

    //! initialize this class
    void init() override;

    //! setup this class
    void setup() override;

    //! time loop to be defined in inherited classes (structure depends on
    //! considered coupling, i.e. one-way or two-way)
    void timeloop() override = 0;

    //! flag whether time loop should be finished
    bool not_finished() { return step_ < numstep_ and time_ <= timemax_; };

    //! read and set fields needed for restart
    void read_restart() override;

    /// redistribute FPS3I interface, if running on parallel
    void redistribute_interface() override;

    //! set-up of FPSI and ScaTra systems
    void setup_system() override;

    //! test results for individual fields
    void test_results(const Epetra_Comm& comm) override;

    //! evaluate ScaTra fields
    void evaluate_scatra_fields() override;

    //! information transfer FPSI -> ScaTra
    void set_fpsi_solution();

    /// set scatra solution on structure field
    void set_struct_scatra_solution();

    //! return communicator
    const Epetra_Comm& get_comm() const { return comm_; }


    /// extract fluid convective and structure convective velocities
    void extract_vel(std::vector<Teuchos::RCP<const Core::LinAlg::Vector<double>>>& vel,
        std::vector<Teuchos::RCP<const Core::LinAlg::Vector<double>>>& convel);

    /// extract Wall Shear Stresses at the interface
    void extract_wss(std::vector<Teuchos::RCP<const Core::LinAlg::Vector<double>>>& wss);

    /// extracts pressures at the interface
    void extract_pressure(std::vector<Teuchos::RCP<const Core::LinAlg::Vector<double>>>& pressure);

    /// provide velocities from FPSI subproblem for scatra subproblem
    void set_velocity_fields();

    /// provide wall shear stresses from FPSI subproblem for scatra subproblem
    void set_wall_shear_stresses();

    /// provide pressures from FPSI subproblem for scatra subproblem
    void set_pressure_fields();

    /// provide displacements from FPSI subproblem for scatra subproblem
    void set_mesh_disp();

   protected:
    /// fpsi algorithm
    Teuchos::RCP<FPSI::MonolithicPlain> fpsi_;


   private:
    /// communication (mainly for screen output)
    const Epetra_Comm& comm_;

    /// scatra field on fluid
    Teuchos::RCP<Adapter::ScaTraBaseAlgorithm> fluidscatra_;

    /// scatra field on structure
    Teuchos::RCP<Adapter::ScaTraBaseAlgorithm> structscatra_;
  };
}  // namespace FS3I

FOUR_C_NAMESPACE_CLOSE

#endif

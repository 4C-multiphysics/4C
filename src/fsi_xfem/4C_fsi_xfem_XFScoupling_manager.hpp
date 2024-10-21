#ifndef FOUR_C_FSI_XFEM_XFSCOUPLING_MANAGER_HPP
#define FOUR_C_FSI_XFEM_XFSCOUPLING_MANAGER_HPP

#include "4C_config.hpp"

#include "4C_fsi_xfem_coupling_comm_manager.hpp"
#include "4C_fsi_xfem_coupling_manager.hpp"

FOUR_C_NAMESPACE_OPEN

namespace FLD
{
  class XFluid;
}

namespace Adapter
{
  class Structure;
}

namespace XFEM
{
  class ConditionManager;
  class MeshCouplingFSI;

  class XfsCouplingManager : public CouplingManager, public CouplingCommManager
  {
   public:
    /// constructor
    // in idx ... idx[0] structureal discretization index , idx[1] fluid discretization index in the
    // blockmatrix
    explicit XfsCouplingManager(Teuchos::RCP<ConditionManager> condmanager,
        Teuchos::RCP<Adapter::Structure> structure, Teuchos::RCP<FLD::XFluid> xfluid,
        std::vector<int> idx);

    //! @name Destruction
    //@{

    //! predict states in the coupling object
    void predict_coupling_states() override{};

    //! init...
    void init_coupling_states() override;

    //! Set required displacement & velocity states in the coupling object
    void set_coupling_states() override;

    //! Add the coupling matrixes to the global systemmatrix
    // in ... scaling between xfluid evaluated coupling matrixes and coupled systemmatrix
    void add_coupling_matrix(
        Core::LinAlg::BlockSparseMatrixBase& systemmatrix, double scaling) override;

    //! Add the coupling rhs

    // in scaling ... scaling between xfluid evaluated coupling rhs and coupled rhs
    // in me ... global map extractor of coupled problem (same index used as for idx)
    void add_coupling_rhs(Teuchos::RCP<Core::LinAlg::Vector<double>> rhs,
        const Core::LinAlg::MultiMapExtractor& me, double scaling) override;

    //! Update (Perform after Each Timestep)
    void update(double scaling) override;

    //! Write Output (For restart or write results on the interface)
    void output(Core::IO::DiscretizationWriter& writer) override;

    //! Read Restart (For lambda_)
    void read_restart(Core::IO::DiscretizationReader& reader) override;

   private:
    //! Get Timeface on the interface (for OST this is 1/(theta dt))
    double get_interface_timefac();

    //! FSI Mesh Coupling Object
    Teuchos::RCP<MeshCouplingFSI> mcfsi_;

    //! Structural Object
    Teuchos::RCP<Adapter::Structure> struct_;
    //! eXtendedFluid
    Teuchos::RCP<FLD::XFluid> xfluid_;

    //"XFEMSurfFSIMono"
    const std::string cond_name_;

    // Global Index in the blockmatrix of the coupled sytem [0] = structure-, [1] = fluid- block
    std::vector<int> idx_;

    bool interface_second_order_;

    //--------------------------------------------------------------------------//
    //! @name Store the Coupling RHS of the Old Timestep in lambda

    //! Lagrange multiplier \f$\lambda_\Gamma^n\f$ at the interface (ie forces onto the structure,
    //! Robin-type forces consisting of fluid forces and the Nitsche penalty term contribution)
    //! evaluated at old time step \f$t_n\f$ but needed for next time step \f$t_{n+1}\f$
    Teuchos::RCP<Core::LinAlg::Vector<double>> lambda_;
  };
}  // namespace XFEM
FOUR_C_NAMESPACE_CLOSE

#endif

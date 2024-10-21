#ifndef FOUR_C_FSI_XFEM_XFPCOUPLING_MANAGER_HPP
#define FOUR_C_FSI_XFEM_XFPCOUPLING_MANAGER_HPP

#include "4C_config.hpp"

#include "4C_fsi_xfem_coupling_comm_manager.hpp"
#include "4C_fsi_xfem_coupling_manager.hpp"

FOUR_C_NAMESPACE_OPEN

namespace FLD
{
  class XFluid;
}

namespace PoroElast
{
  class PoroBase;
}

namespace XFEM
{
  class ConditionManager;
  class MeshCouplingFPI;

  class XfpCouplingManager : public CouplingManager, public CouplingCommManager
  {
   public:
    /// constructor
    explicit XfpCouplingManager(Teuchos::RCP<ConditionManager> condmanager,
        Teuchos::RCP<PoroElast::PoroBase> poro, Teuchos::RCP<FLD::XFluid> xfluid,
        std::vector<int> idx);

    //! @name Destruction
    //@{

    //! predict states in the coupling object
    void predict_coupling_states() override {}

    //! Initializes the couplings (done at the beginning of the algorithm after fields have their
    //! state for timestep n)
    void init_coupling_states() override;

    void set_coupling_states() override;

    void add_coupling_matrix(
        Core::LinAlg::BlockSparseMatrixBase& systemmatrix, double scaling) override;

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



    Teuchos::RCP<MeshCouplingFPI> mcfpi_ps_ps_;
    Teuchos::RCP<MeshCouplingFPI> mcfpi_ps_pf_;
    Teuchos::RCP<MeshCouplingFPI> mcfpi_pf_ps_;
    Teuchos::RCP<MeshCouplingFPI> mcfpi_pf_pf_;

    Teuchos::RCP<PoroElast::PoroBase> poro_;
    Teuchos::RCP<FLD::XFluid> xfluid_;

    std::string cond_name_ps_ps_;
    std::string cond_name_ps_pf_;
    std::string cond_name_pf_ps_;
    std::string cond_name_pf_pf_;

    // Global Index in the blockmatrix of the coupled sytem [0] = structure-, [1] = fluid- block,
    // [2] = porofluid-block
    std::vector<int> idx_;

    bool interface_second_order_;

    //--------------------------------------------------------------------------//
    //! @name Store the Coupling RHS of the Old Timestep in lambda

    //! Lagrange multiplier \f$\lambda_\Gamma^n\f$ at the interface (ie forces onto the structure,
    //! Robin-type forces consisting of fluid forces and the Nitsche penalty term contribution)
    //! evaluated at old time step \f$t_n\f$ but needed for next time step \f$t_{n+1}\f$
    Teuchos::RCP<Core::LinAlg::Vector<double>> lambda_ps_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> lambda_pf_;
  };
}  // namespace XFEM
FOUR_C_NAMESPACE_CLOSE

#endif

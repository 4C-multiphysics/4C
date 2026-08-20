// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSI_COUPLING_HPP
#define FOUR_C_SSI_COUPLING_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter_mortar.hpp"
#include "4C_coupling_adapter_volmortar.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_ssi_base.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Adapter
{
  class ScaTraBaseAlgorithm;
  class Structure;
}  // namespace Adapter

namespace SSI
{
  //! Base class of solid-scatra coupling helper classes
  class SSICouplingBase
  {
   public:
    SSICouplingBase() = default;

    virtual ~SSICouplingBase() = default;

    //! \brief init this class
    //!
    //! \param ndim                        dimension of the problem
    //! \param structdis                   underlying structure discretization
    //! \param ssi_base                    underlying scatra-structure time integrator
    virtual void init(int ndim, std::shared_ptr<Core::FE::Discretization> structdis,
        std::shared_ptr<SSI::SSIBase> ssi_base) = 0;

    //! \brief setup this class
    virtual void setup() = 0;


    //! \brief exchange material pointers of both discratizations
    //!
    //! \param structdis   underlying structure discretization
    //! \param scatradis   underlying scatra discretization
    virtual void assign_material_pointers(std::shared_ptr<Core::FE::Discretization> structdis,
        std::shared_ptr<Core::FE::Discretization> scatradis) = 0;

    //!
    //! \param scatradis      underlying scatra discretization
    //! \param stress_state   mechanical stress state vector to set
    //! \param nds            number of dofset to write state on
    virtual void set_mechanical_stress_state(Core::FE::Discretization& scatradis,
        const Core::LinAlg::Vector<double>& stress_state, unsigned nds) = 0;

    //! \brief set structure mesh displacement on other field
    //!
    //! \param scatra    underlying scatra problem of the SSI problem
    //! \param disp      displacement field to set
    virtual void set_mesh_disp(std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
        const Core::LinAlg::Vector<double>& disp) = 0;

    //! \brief set structure velocity fields on other field
    //!
    //! \param scatra    underlying scatra problem of the SSI problem
    //! \param convvel   convective velocity field to set
    //! \param vel       velocity field to set
    virtual void set_velocity_fields(std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
        const Core::LinAlg::Vector<double>& convvel, const Core::LinAlg::Vector<double>& vel) = 0;

    //! \brief set scatra solution on other field
    //!
    //! \param dis    discretization to write scatra solution on
    //! \param phi    scalar field solution
    //! \param nds    number of dofset to write state on
    virtual void set_scalar_field(
        Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& phi, unsigned nds) = 0;

    //! \brief set micro solution of scatra field other field
    //!
    //! \param dis     discretization to write micro scatra solution on
    //! \param phi     micro scatra solution
    //! \param nds     number of dofset to write micro scatra solution on
    virtual void set_scalar_field_micro(
        Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& phi, unsigned nds) = 0;

    //! set temperature field on structure field
    virtual void set_temperature_field(Core::FE::Discretization& structdis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> temp) = 0;
  };

  //! solid-scatra coupling for matching volume meshes
  class SSICouplingMatchingVolume : public SSICouplingBase
  {
   public:
    SSICouplingMatchingVolume() : issetup_(false), isinit_(false) {}

    void init(int ndim, std::shared_ptr<Core::FE::Discretization> structdis,
        std::shared_ptr<SSI::SSIBase> ssi_base) override;

    void setup() override;

    void assign_material_pointers(std::shared_ptr<Core::FE::Discretization> structdis,
        std::shared_ptr<Core::FE::Discretization> scatradis) override;

    void set_mechanical_stress_state(Core::FE::Discretization& scatradis,
        const Core::LinAlg::Vector<double>& stress_state, unsigned nds) override;

    void set_mesh_disp(std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
        const Core::LinAlg::Vector<double>& disp) override;

    void set_velocity_fields(std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
        const Core::LinAlg::Vector<double>& convvel,
        const Core::LinAlg::Vector<double>& vel) override;

    void set_scalar_field(Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& phi,
        unsigned nds) override;

    void set_scalar_field_micro(Core::FE::Discretization& dis,
        const Core::LinAlg::Vector<double>& phi, unsigned nds) override;

    void set_temperature_field(Core::FE::Discretization& structdis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> temp) override;

   private:
    //! flag indicating if class is setup
    bool issetup_;

    //! flag indicating if class is initialized
    bool isinit_;

   protected:
    //! returns true if setup() was called and is still valid
    [[nodiscard]] bool is_setup() const { return issetup_; }

    //! returns true if init(..) was called and is still valid
    [[nodiscard]] bool is_init() const { return isinit_; }

    //! returns true if class was setup and setup is still valid
    void check_is_setup() const { FOUR_C_ASSERT_ALWAYS(is_setup(), "setup() was not called."); }

    //! returns true if class was init and init is still valid
    void check_is_init() const { FOUR_C_ASSERT_ALWAYS(is_init(), "init(...) was not called."); }

   public:
    //! set flag true after setup or false if setup became invalid
    void set_is_setup(bool trueorfalse) { issetup_ = trueorfalse; }

    //! set flag true after init or false if init became invalid
    void set_is_init(bool trueorfalse) { isinit_ = trueorfalse; }
  };

  //! solid-scatra coupling for matching boundary meshes
  class SSICouplingNonMatchingBoundary : public SSICouplingBase
  {
   public:
    SSICouplingNonMatchingBoundary() = default;

    void init(int ndim, std::shared_ptr<Core::FE::Discretization> structdis,
        std::shared_ptr<SSI::SSIBase> ssi_base) override;

    void setup() override;

    void assign_material_pointers(std::shared_ptr<Core::FE::Discretization> structdis,
        std::shared_ptr<Core::FE::Discretization> scatradis) override;

    void set_mechanical_stress_state(Core::FE::Discretization& scatradis,
        const Core::LinAlg::Vector<double>& stress_state, unsigned nds) override
    {
      FOUR_C_THROW("only implemented for 'SSICouplingMatchingVolume'");
    }

    void set_mesh_disp(std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
        const Core::LinAlg::Vector<double>& disp) override;

    void set_velocity_fields(std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
        const Core::LinAlg::Vector<double>& convvel,
        const Core::LinAlg::Vector<double>& vel) override;

    void set_scalar_field(Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& phi,
        unsigned nds) override;

    void set_scalar_field_micro(Core::FE::Discretization& dis,
        const Core::LinAlg::Vector<double>& phi, unsigned nds) override;

    void set_temperature_field(Core::FE::Discretization& structdis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> temp) override
    {
      FOUR_C_THROW("only for matching nodes");
    }

   private:
    //! adapter to mortar framework
    std::shared_ptr<Coupling::Adapter::CouplingMortar> adaptermeshtying_;

    //! extractor for coupled surface of structure discretization with surface scatra
    std::shared_ptr<Core::LinAlg::MapExtractor> extractor_;

    //! flag indicating if class is setup
    bool issetup_;

    //! flag indicating if class is initialized
    bool isinit_;

    //! spatial dimension of the global problem
    int problem_dimension_;

    //! pointer to structdis_
    std::shared_ptr<Core::FE::Discretization> structdis_;

    //! pointer to scatradis_
    std::shared_ptr<Core::FE::Discretization> scatradis_;

   protected:
    //! returns true if setup() was called and is still valid
    [[nodiscard]] bool is_setup() const { return issetup_; }

    //! returns true if init(..) was called and is still valid
    [[nodiscard]] bool is_init() const { return isinit_; }

    //! returns true if class was setup and setup is still valid
    void check_is_setup() const { FOUR_C_ASSERT_ALWAYS(is_setup(), "setup() was not called."); }

    //! returns true if class was init and init is still valid
    void check_is_init() const { FOUR_C_ASSERT_ALWAYS(is_init(), "init(...) was not called."); }

   public:
    //! set flag true after setup or false if setup became invalid
    void set_is_setup(bool trueorfalse) { issetup_ = trueorfalse; }

    //! set flag true after init or false if init became invalid
    void set_is_init(bool trueorfalse) { isinit_ = trueorfalse; }
  };

  //! solid-scatra coupling for non-matching boundary meshes
  class SSICouplingNonMatchingVolume : public SSICouplingBase
  {
   public:
    SSICouplingNonMatchingVolume()
        : volcoupl_structurescatra_(nullptr), issetup_(false), isinit_(false)
    {
    }

    void init(int ndim, std::shared_ptr<Core::FE::Discretization> structdis,
        std::shared_ptr<SSI::SSIBase> ssi_base) override;

    void setup() override;

    void assign_material_pointers(std::shared_ptr<Core::FE::Discretization> structdis,
        std::shared_ptr<Core::FE::Discretization> scatradis) override;

    void set_mechanical_stress_state(Core::FE::Discretization& scatradis,
        const Core::LinAlg::Vector<double>& stress_state, unsigned nds) override
    {
      FOUR_C_THROW("only implemented for 'SSICouplingMatchingVolume'");
    }

    void set_mesh_disp(std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
        const Core::LinAlg::Vector<double>& disp) override;

    void set_velocity_fields(std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
        const Core::LinAlg::Vector<double>& convvel,
        const Core::LinAlg::Vector<double>& vel) override;

    void set_scalar_field(Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& phi,
        unsigned nds) override;

    void set_scalar_field_micro(Core::FE::Discretization& dis,
        const Core::LinAlg::Vector<double>& phi, unsigned nds) override;

    void set_temperature_field(Core::FE::Discretization& structdis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> temp) override
    {
      FOUR_C_THROW("only for matching nodes");
    }

   private:
    //! volume coupling (using mortar) adapter
    std::shared_ptr<Coupling::Adapter::MortarVolCoupl> volcoupl_structurescatra_;

    //! flag indicating if class is setup
    bool issetup_;

    //! flag indicating if class is initialized
    bool isinit_;

   protected:
    //! returns true if setup() was called and is still valid
    [[nodiscard]] bool is_setup() const { return issetup_; }

    //! returns true if init(..) was called and is still valid
    [[nodiscard]] bool is_init() const { return isinit_; }

    //! returns true if class was setup and setup is still valid
    void check_is_setup() const { FOUR_C_ASSERT_ALWAYS(is_setup(), "setup() was not called."); }

    //! returns true if class was init and init is still valid
    void check_is_init() const { FOUR_C_ASSERT_ALWAYS(is_init(), "init(...) was not called."); }

   public:
    //! set flag true after setup or false if setup became invalid
    void set_is_setup(bool trueorfalse) { issetup_ = trueorfalse; }

    //! set flag true after init or false if init became invalid
    void set_is_init(bool trueorfalse) { isinit_ = trueorfalse; }
  };

  //! solid-scatra coupling for matching volume and boundary meshes
  class SSICouplingMatchingVolumeAndBoundary : public SSICouplingBase
  {
   public:
    SSICouplingMatchingVolumeAndBoundary() : issetup_(false), isinit_(false) {}

    void init(int ndim, std::shared_ptr<Core::FE::Discretization> structdis,
        std::shared_ptr<SSI::SSIBase> ssi_base) override;

    void setup() override;


    void assign_material_pointers(std::shared_ptr<Core::FE::Discretization> structdis,
        std::shared_ptr<Core::FE::Discretization> scatradis) override;

    void set_mechanical_stress_state(Core::FE::Discretization& scatradis,
        const Core::LinAlg::Vector<double>& stress_state, unsigned nds) override
    {
      FOUR_C_THROW("only implemented for 'SSICouplingMatchingVolume'");
    }

    void set_mesh_disp(std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
        const Core::LinAlg::Vector<double>& disp) override;

    void set_velocity_fields(std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
        const Core::LinAlg::Vector<double>& convvel,
        const Core::LinAlg::Vector<double>& vel) override;

    void set_scalar_field(Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& phi,
        unsigned nds) override;

    void set_scalar_field_micro(Core::FE::Discretization& dis,
        const Core::LinAlg::Vector<double>& phi, unsigned nds) override;

    void set_temperature_field(Core::FE::Discretization& structdis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> temp) override;

   private:
    //! flag indicating if class is setup
    bool issetup_;

    //! flag indicating if class is initialized
    bool isinit_;

   protected:
    //! returns true if setup() was called and is still valid
    [[nodiscard]] bool is_setup() const { return issetup_; }

    //! returns true if init(..) was called and is still valid
    [[nodiscard]] bool is_init() const { return isinit_; }

    //! returns true if class was setup and setup is still valid
    void check_is_setup() const { FOUR_C_ASSERT_ALWAYS(is_setup(), "setup() was not called."); }

    //! returns true if class was init and init is still valid
    void check_is_init() const { FOUR_C_ASSERT_ALWAYS(is_init(), "init(...) was not called."); }

   public:
    //! set flag true after setup or false if setup became invalid
    void set_is_setup(bool trueorfalse) { issetup_ = trueorfalse; }

    //! set flag true after init or false if init became invalid
    void set_is_init(bool trueorfalse) { isinit_ = trueorfalse; }
  };
}  // namespace SSI


FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VISCOELAST_SUMMAND_HPP
#define FOUR_C_MAT_VISCOELAST_SUMMAND_HPP

#include "4C_config.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_legacy_enum_definitions_materials.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_solid_ele_fibers.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Teuchos
{
  class ParameterList;
}

FOUR_C_NAMESPACE_OPEN

namespace Mat::ViscoElast
{
  /**
   * \brief Base class for parameter-backed visco summands used by Mat::ViscoElastHyper.
   *
   * A summand adapts a material parameter object to the hooks needed by the active contribution
   * implementations. The base class provides no-op defaults so each concrete summand implements
   * only the hooks required by its model family. Mat::ViscoElastHyper constructs summands through
   * factory() and then asks them to declare formulation flags and provide model-specific
   * coefficients or scalar parameters.
   */
  class Summand : public Core::Communication::ParObject
  {
   public:
    Summand() = default;
    ~Summand() override = default;

    /// Summands are packed by their owning material, not as standalone ParObjects.
    int unique_par_object_id() const override;
    /// Pack no standalone ParObject state.
    void pack(Core::Communication::PackBuffer& data) const override;
    /// Unpack no standalone ParObject state.
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    /// Material type represented by this summand adapter.
    virtual Core::Materials::MaterialType material_type() const = 0;

    /// Construct a concrete visco summand from the global material parameter bundle.
    static std::shared_ptr<Summand> factory(int matnum);

    /// Pack model-specific summand state if a concrete summand owns any.
    virtual void pack_summand(Core::Communication::PackBuffer& data) const;
    /// Unpack model-specific summand state if a concrete summand owns any.
    virtual void unpack_summand(Core::Communication::UnpackBuffer& buffer);

    /// Prepare optional summand-local data before contribution setup.
    virtual void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system);

    /// Update optional summand-local state after a converged time step.
    virtual void update();

    /// Report invariant formulation flags needed by the owning material.
    virtual void specify_formulation(
        bool& isoprinc, bool& isomod, bool& anisoprinc, bool& anisomod, bool& viscogeneral);

    /// Add principal iso-rate coefficients; default implementation contributes nothing.
    virtual void add_coefficients_visco_principal(const Core::LinAlg::Matrix<3, 1>& inv,
        Core::LinAlg::Matrix<8, 1>& mu, Core::LinAlg::Matrix<33, 1>& xi,
        Core::LinAlg::Matrix<7, 1>& rateinv, const Teuchos::ParameterList& params, double dt,
        int gp, int eleGID);

    /// Add modified-invariant iso-rate coefficients; default implementation contributes nothing.
    virtual void add_coefficients_visco_modified(const Core::LinAlg::Matrix<3, 1>& modinv,
        Core::LinAlg::Matrix<8, 1>& modmu, Core::LinAlg::Matrix<33, 1>& modxi,
        Core::LinAlg::Matrix<7, 1>& modrateinv, const Teuchos::ParameterList& params, double dt,
        int gp, int eleGID);

    /// Read FSLS-like scalar parameters from a concrete summand.
    virtual void read_material_parameters_visco(
        double& tau, double& beta, double& alpha, std::string& solve);

    /// Read generalized Maxwell top-level parameters from a concrete summand.
    virtual void read_material_parameters(
        int& numbranch, const std::vector<int>*& matids, std::string& solve);

    /// Read generalized Maxwell branch parameters from a concrete summand.
    virtual void read_material_parameters(double& tau, int& matid);

    /// Report which contribution family this summand activates.
    virtual void specify_visco_formulation(bool& visco_iso_rate, bool& visco_generalized_maxwell,
        bool& visco_quasi_linear_generalized_maxwell, bool& visco_fsls);
  };
}  // namespace Mat::ViscoElast

FOUR_C_NAMESPACE_CLOSE

#endif

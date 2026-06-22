// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_COUPLING_ADAPTER_VOLMORTAR_HPP
#define FOUR_C_COUPLING_ADAPTER_VOLMORTAR_HPP

/*---------------------------------------------------------------------*
 | headers                                                 farah 10/13 |
 *---------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_binstrategy_utils.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <functional>
#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------*
 | forward declarations                                    farah 10/13 |
 *---------------------------------------------------------------------*/
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Core::LinAlg
{
  class SparseMatrix;
}

namespace Coupling::VolMortar
{
  namespace Utils
  {
    class DefaultMaterialStrategy;
  }
}  // namespace Coupling::VolMortar

namespace Core::IO
{
  class OutputControl;
}

namespace Coupling::Adapter
{  /// Class for calling volmortar coupling and proper parallel redistr.
  class MortarVolCoupl : public CouplingBase
  {
   public:
    /*!
    \brief Empty constructor

    */
    MortarVolCoupl();

    /*!
    \brief Call parallel redistr. and evaluate volmortar coupl.

    */
    void init(int spatial_dimension, std::shared_ptr<Core::FE::Discretization> dis1,
        std::shared_ptr<Core::FE::Discretization> dis2, std::vector<int>* coupleddof12 = nullptr,
        std::vector<int>* coupleddof21 = nullptr, std::pair<int, int>* dofsets12 = nullptr,
        std::pair<int, int>* dofsets21 = nullptr,
        std::shared_ptr<VolMortar::Utils::DefaultMaterialStrategy> materialstrategy = nullptr,
        bool createauxdofs = true);

    /*!
    \brief Setup this class based on the @p params.

    */
    void setup(const Teuchos::ParameterList& params, const Teuchos::ParameterList& cut_params);

    /*!
    \brief Redistribute discretizations to meet needs of volmortar coupling

    \note Call this method in your global control algorithm inbetween \ref init()
          and \ref setup(), in case you need parallel redistribution




    */
    void redistribute(const Teuchos::ParameterList& binning_params,
        std::shared_ptr<Core::IO::OutputControl> output_control,
        std::function<const Core::Nodes::Node&(const Core::Nodes::Node& node)> correct_node =
            nullptr,
        std::function<std::vector<std::array<double, 3>>(const Core::FE::Discretization&,
            const Core::Elements::Element&,
            std::shared_ptr<const Core::LinAlg::Vector<double>> disnp)>
            determine_relevant_points = nullptr);


    /*!
    \brief Get coupling matrices for field 1 and 2

    */
    std::shared_ptr<const Core::LinAlg::SparseMatrix> get_p_matrix12() const { return p12_; };
    std::shared_ptr<const Core::LinAlg::SparseMatrix> get_p_matrix21() const { return p21_; };

    /*!
    \brief Mortar mapping for 1 to 2 and 2 to 1 - for vectors

    */
    std::shared_ptr<const Core::LinAlg::Vector<double>> apply_vector_mapping12(
        const Core::LinAlg::Vector<double>& vec) const;
    std::shared_ptr<const Core::LinAlg::Vector<double>> apply_vector_mapping21(
        const Core::LinAlg::Vector<double>& vec) const;

    /*!
    \brief Mortar mapping for 1 to 2 and 2 to 1 - for matrices

    */
    std::shared_ptr<Core::LinAlg::SparseMatrix> apply_matrix_mapping12(
        const Core::LinAlg::SparseMatrix& mat) const;
    std::shared_ptr<Core::LinAlg::SparseMatrix> apply_matrix_mapping21(
        const Core::LinAlg::SparseMatrix& mat) const;

    //@}

    //! assign materials
    void assign_materials(std::shared_ptr<Core::FE::Discretization> dis1,
        std::shared_ptr<Core::FE::Discretization> dis2,
        const Teuchos::ParameterList& volmortar_params, const Teuchos::ParameterList& cut_params,
        std::shared_ptr<VolMortar::Utils::DefaultMaterialStrategy> materialstrategy = nullptr);


    /** \name Conversion between target and source */
    //@{
    /// There are different versions to satisfy all needs. The basic
    /// idea is the same for all of them.


    /// transfer a dof vector from target to source
    std::shared_ptr<Core::LinAlg::Vector<double>> target_to_source(
        const Core::LinAlg::Vector<double>& mv) const override;

    /// transfer a dof vector from source to target
    std::shared_ptr<Core::LinAlg::Vector<double>> source_to_target(
        const Core::LinAlg::Vector<double>& sv) const override;

    /// transfer a dof vector from target to source
    std::shared_ptr<Core::LinAlg::MultiVector<double>> target_to_source(
        const Core::LinAlg::MultiVector<double>& mv) const override;

    /// transfer a dof vector from source to target
    std::shared_ptr<Core::LinAlg::MultiVector<double>> source_to_target(
        const Core::LinAlg::MultiVector<double>& sv) const override;

    /// transfer a dof vector from target to source
    void target_to_source(const Core::LinAlg::MultiVector<double>& mv,
        Core::LinAlg::MultiVector<double>& sv) const override;

    /// transfer a dof vector from source to target
    void source_to_target(const Core::LinAlg::MultiVector<double>& sv,
        Core::LinAlg::MultiVector<double>& mv) const override;

    //@}

    /** \name Coupled maps */
    //@{

    /// the interface dof map of the target side
    std::shared_ptr<const Core::LinAlg::Map> target_dof_map() const override;

    /// the interface dof map of the source side
    std::shared_ptr<const Core::LinAlg::Map> source_dof_map() const override;

    //@}

   private:
    /*!
    \brief Create auxiliary dofsets for multiphysics if necessary

    */
    void create_aux_dofsets(Core::FE::Discretization& dis1, Core::FE::Discretization& dis2,
        std::vector<int>* coupleddof12, std::vector<int>* coupleddof21);

    /// check setup call
    const bool& is_setup() const { return issetup_; };

    /// check setup call
    const bool& is_init() const { return isinit_; };

    /// check init and setup call
    void check_setup() const
    {
      if (not is_setup()) FOUR_C_THROW("ERROR: Call setup() first!");
    }

    /// check init and setup call
    void check_init() const
    {
      if (not is_init()) FOUR_C_THROW("ERROR: Call init() first!");
    }

   private:
    bool issetup_;  ///< check for setup
    bool isinit_;   ///< check for init

    // mortar matrices and projector
    // s1 = P12 * s2
    // s2 = P21 * s1
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        p12_;  ///< global Mortar projection matrix P Omega_2 -> Omega_1
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        p21_;  ///< global Mortar projection matrix P Omega_1 -> Omega_2

    std::shared_ptr<Core::FE::Discretization> target_dis_;
    std::shared_ptr<Core::FE::Discretization> source_dis_;

    std::vector<int>* coupleddof12_;
    std::vector<int>* coupleddof21_;
    std::pair<int, int>* dofsets12_;
    std::pair<int, int>* dofsets21_;
    std::shared_ptr<VolMortar::Utils::DefaultMaterialStrategy> materialstrategy_;

    int spatial_dimension_{};
  };
}  // namespace Coupling::Adapter

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_SCATRA_3D_ELE_HPP
#define FOUR_C_SOLID_SCATRA_3D_ELE_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_solid_3D_ele_calc_eas.hpp"
#include "4C_solid_scatra_3D_ele_calc_lib_nitsche.hpp"
#include "4C_solid_scatra_3D_ele_factory.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class So3Material;
}
namespace Discret::Elements
{
  // forward declaration
  class SolidScatraType : public Core::Elements::ElementType
  {
   public:
    void setup_element_definition(
        std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
        override;

    std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
        const std::string elecelltype, const int id, const int owner) override;

    std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

    [[nodiscard]] std::string name() const override { return "SolidScatraType"; }

    void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override;

    Core::LinAlg::SerialDenseMatrix compute_null_space(
        Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

    static SolidScatraType& instance();

   private:
    static SolidScatraType instance_;

  };  // class SolidType

  class SolidScatra : public Core::Elements::Element
  {
    friend class SolidScatraType;

   public:
    SolidScatra(int id, int owner);

    [[nodiscard]] Core::Elements::Element* clone() const override;

    [[nodiscard]] int unique_par_object_id() const override
    {
      return SolidScatraType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    [[nodiscard]] Core::Elements::ElementType& element_type() const override
    {
      return SolidScatraType::instance();
    }

    [[nodiscard]] Core::FE::CellType shape() const override { return celltype_; }

    [[nodiscard]] virtual Mat::So3Material& solid_material(int nummat = 0) const;

    [[nodiscard]] int num_line() const override;

    [[nodiscard]] int num_surface() const override;

    [[nodiscard]] int num_volume() const override;

    std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

    std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

    [[nodiscard]] int num_dof_per_node(const Core::Nodes::Node& node) const override { return 3; }

    [[nodiscard]] int num_dof_per_element() const override { return 0; }

    bool read_element(const std::string& eletype, const std::string& celltype,
        const Core::IO::InputParameterContainer& container) override;

    int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
        Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseVector& elevec2,
        Core::LinAlg::SerialDenseVector& elevec3) override;

    int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        const Core::Conditions::Condition& condition, std::vector<int>& lm,
        Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

    std::shared_ptr<Core::Elements::ParamsInterface> params_interface_ptr() override
    {
      return interface_ptr_;
    }

    [[nodiscard]] inline bool is_params_interface() const override
    {
      return (interface_ptr_ != nullptr);
    }

    [[nodiscard]] inline bool is_solid_params_interface() const
    {
      return (solid_interface_ptr_ != nullptr);
    }

    [[nodiscard]] inline Core::Elements::ParamsInterface& params_interface() const
    {
      FOUR_C_ASSERT_ALWAYS(interface_ptr_.get(), "The parameter interface pointer is not set.");
      return *interface_ptr_;
    }
    [[nodiscard]] inline FourC::Solid::Elements::ParamsInterface& get_solid_params_interface() const
    {
      FOUR_C_ASSERT_ALWAYS(solid_interface_ptr_.get(),
          "The parameter interface pointer is not set or not a solid parameter interface.");
      return *solid_interface_ptr_;
    }

    void set_params_interface_ptr(const Teuchos::ParameterList& p) override;

    void vis_names(std::map<std::string, int>& names) override;

    bool vis_data(const std::string& name, std::vector<double>& data) override;

    /// return ScaTra::ImplType
    [[nodiscard]] Inpar::ScaTra::ImplType impl_type() const { return properties_.impltype; }

    [[nodiscard]] const SolidElementProperties& get_solid_element_properties() const
    {
      return properties_.solid;
    }

    /*!
     * @brief Returns the Cauchy stress in the direction @p dir at @p xi with normal @p n
     *
     * @param disp Nodal displacements of the element
     * @param scalars Scalars at the nodes of the element
     * @param xi
     * @param n
     * @param dir
     * @param linearizations [in/out] : Struct holding the linearizations that are possible for
     * evaluation
     * @return double
     */
    double get_normal_cauchy_stress_at_xi(const std::vector<double>& disp,
        const std::vector<double>& scalars, const Core::LinAlg::Tensor<double, 3>& xi,
        const Core::LinAlg::Tensor<double, 3>& n, const Core::LinAlg::Tensor<double, 3>& dir,
        SolidScatraCauchyNDirLinearizations<3>& linearizations);

   private:
    //! cell type
    Core::FE::CellType celltype_ = Core::FE::CellType::dis_none;

    //! solid-scatra properties
    SolidScatraElementProperties properties_{};

    //! interface pointer for data exchange between the element and the time integrator.
    std::shared_ptr<Core::Elements::ParamsInterface> interface_ptr_;

    //! interface pointer for data exchange between the element and the solid time integrator.
    std::shared_ptr<FourC::Solid::Elements::ParamsInterface> solid_interface_ptr_;


    //! solid element calculation holding one of the implemented variants
    SolidScatraCalcVariant solid_scatra_calc_variant_;

    //! flag, whether the post setup of materials is already called
    bool material_post_setup_ = false;

  };  // class SolidScatra

}  // namespace Discret::Elements


FOUR_C_NAMESPACE_CLOSE

#endif

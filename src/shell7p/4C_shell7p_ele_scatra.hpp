// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SHELL7P_ELE_SCATRA_HPP
#define FOUR_C_SHELL7P_ELE_SCATRA_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_shell7p_ele_calc_interface.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements
{
  class Shell7pEleCalcInterface;
  class Shell7pLine;

  class Shell7pScatraType : public Core::Elements::ElementType
  {
   public:
    void setup_element_definition(
        std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
        override;

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

    std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
        const std::string eledistype, const int id, const int owner) override;

    std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

    [[nodiscard]] std::string name() const override { return "Shell7pScatraType"; }

    int initialize(Core::FE::Discretization& dis) override;

    void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override;

    Core::LinAlg::SerialDenseMatrix compute_null_space(
        Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

    static Shell7pScatraType& instance();

   private:
    static Shell7pScatraType instance_;
  };

  class Shell7pScatra : public Core::Elements::Element
  {
    //! @name Friends
    friend class Shell7pScatraType;
    friend class Shell7pLine;

   public:
    //! @name Constructors and destructors related methods
    //! @{
    /*!
     * \brief Standard Constructor
     * @param id (in) : A unique global id
     * @param owner (in) : elements owner
     */
    Shell7pScatra(int id, int owner) : Core::Elements::Element(id, owner) {};

    //! copy Constructor
    Shell7pScatra(const Shell7pScatra& other);


    //! copy assignment operator
    Shell7pScatra& operator=(const Shell7pScatra& other);

    //! move constructor
    Shell7pScatra(Shell7pScatra&& other) noexcept = default;

    //! move assignment operator
    Shell7pScatra& operator=(Shell7pScatra&& other) noexcept = default;
    //! @}

    [[nodiscard]] Core::Elements::Element* clone() const override;

    [[nodiscard]] int unique_par_object_id() const override
    {
      return Shell7pScatraType::instance().unique_par_object_id();
    }

    [[nodiscard]] int num_line() const override;

    [[nodiscard]] int num_surface() const override;

    std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

    std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

    [[nodiscard]] int num_dof_per_node(const Core::Nodes::Node& node) const override { return 6; }

    [[nodiscard]] int num_dof_per_element() const override { return 0; }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    void print(std::ostream& os) const override;


    [[nodiscard]] Core::Elements::ElementType& element_type() const override
    {
      return Shell7pScatraType::instance();
    }

    [[nodiscard]] Core::FE::CellType shape() const override { return distype_; };

    bool read_element(const std::string& eletype, const std::string& distype,
        const Core::IO::InputParameterContainer& container) override;

    //! @name Evaluation
    //! @{
    int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
        Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseVector& elevec2,
        Core::LinAlg::SerialDenseVector& elevec3) override;

    int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        const Core::Conditions::Condition& condition, std::vector<int>& la,
        Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseMatrix* elemat1) override;

    //@}

    //! @name Query methods
    //! @{
    [[nodiscard]] inline bool is_params_interface() const override
    {
      return (interface_ptr_ != nullptr);
    }

    [[nodiscard]] inline Solid::Elements::ParamsInterface& str_params_interface() const
    {
      if (not is_params_interface()) FOUR_C_THROW("The interface ptr is not set!");
      return *interface_ptr_;
    }

    void set_params_interface_ptr(const Teuchos::ParameterList& p) override;
    //! @}

    [[nodiscard]] const std::set<Inpar::Solid::EleTech>& get_ele_tech() const { return eletech_; }

    [[nodiscard]] std::shared_ptr<Mat::So3Material> solid_material(int nummat = 0) const;

    [[nodiscard]] const Inpar::ScaTra::ImplType& impl_type() const { return impltype_; };

    void vis_names(std::map<std::string, int>& names) override;

    bool vis_data(const std::string& name, std::vector<double>& data) override;

    [[nodiscard]] const double& get_thickness() const { return thickness_; }

    [[nodiscard]] const Core::LinAlg::SerialDenseMatrix& get_directors() const
    {
      return nodal_directors_;
    }

    inline void set_all_nodal_directors(const Core::LinAlg::SerialDenseMatrix& nodal_directors)
    {
      nodal_directors_ = nodal_directors;
    }

    inline void set_nodal_director(const int& node_id, const std::vector<double>& director)
    {
      nodal_directors_(node_id, 0) = director[0];
      nodal_directors_(node_id, 1) = director[1];
      nodal_directors_(node_id, 2) = director[2];
    }

   private:
    //! discretization type
    Core::FE::CellType distype_ = Core::FE::CellType::dis_none;

    //! interface ptr, data exchange between the element and the time integrator.
    std::shared_ptr<Solid::Elements::ParamsInterface> interface_ptr_ = nullptr;

    //! element technology
    std::set<Inpar::Solid::EleTech> eletech_ = {};

    //! shell thickness in reference frame
    double thickness_ = 0.0;

    //! nodal director
    Core::LinAlg::SerialDenseMatrix nodal_directors_ = {};

    //! flag, whether the post setup of materials is already called
    bool material_post_setup_ = false;

    //! shell calculation interface
    std::shared_ptr<Shell7pEleCalcInterface> shell_interface_ = nullptr;

    //! scalar transport implementation type (physics)
    Inpar::ScaTra::ImplType impltype_ = Inpar::ScaTra::ImplType::impltype_undefined;
  };

}  // namespace Discret::Elements


FOUR_C_NAMESPACE_CLOSE

#endif

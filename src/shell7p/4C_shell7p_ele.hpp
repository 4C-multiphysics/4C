// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SHELL7P_ELE_HPP
#define FOUR_C_SHELL7P_ELE_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_shell7p_ele_calc_interface.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Mat
{
  class So3Material;
}  // namespace Mat

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    // forward declaration
    class Shell7pEleCalcInterface;
    class Shell7pLine;


    class Shell7pType : public Core::Elements::ElementType
    {
     public:
      void setup_element_definition(
          std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
          override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      int initialize(Core::FE::Discretization& dis) override;

      [[nodiscard]] std::string name() const override { return "Shell7pType"; }

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, std::span<const double> x0, const int numdof) override;

      static Shell7pType& instance();

     private:
      static Shell7pType instance_;

    };  // class Shell7pType

    /*!
      *  \brief  The 7-parameter shell element
      *
      *  A nonlinear shell element based on the 7-parameter shell formulation with relative
      *  displacements as parameterization of rotations. The formulation is based on [1]. This
      *  formulation is adopted by a scaled-director approach, see [2], to remove the
      *  extra-ill conditioning of the resulting matrices that appears with three-dimensional shell
      *  formulations as compared to formulations that neglect the thickness change of the shell.
      *
      *  The shell element can be used with tri3, tri6, quad4, quad8 and quad9 elements. The
      *  7-parameter formulation has three translational and three relative-translational degrees
      *  of freedom with an additional linear thickness change introduced via a hybrid formulation.
      *
      *  The current formulation is able to remedy different types of locking for quadrilateral
      *  elements due to the EAS and ANS method. The EAS input parameters corresponds to membrane
      *  locking, bending locking, locking due to thickness changes, locking due to transverse
      *  shear strain (constant and linear). The individual mechanism to remedy locking can be
      *  turned off/on separately. The combined EAS/ANS element can be practically considered
      *  locking free. Currently, the number of gauss points in thickness direction are prescribed
      *  by two integration points. Otherwise, the shell element would suffer from a
      *  nonlinear poisson stiffening affect, because the implemented formulation assumes that
      *  the strains are linear in thickness direction.
      *
      *  The element supports the use of 3D Solid Materials, however only St. Venant Kirchhoff,
      *  Coupled NeoHooke, IsoNeoHooke were tested now.
      *

      *  References:
      *  [1] Theorie und Numerik einer dreidimensionalen Schalenformulierung (1999), M.Bischoff,
      *      PhD-Thesis
      *  [2] Effiziente Loesungsstrategien in der nichtlinearen Schalenmechanik (2004), M.Gee,
      *      PhD-Thesis
      *
      */
    class Shell7p : public Core::Elements::Element
    {
     public:
      //! @name Friends
      friend class Shell7pType;
      friend class Shell7pLine;

      //! @name Constructors and destructors related methods
      //! @{
      /*!
      \brief Standard Constructor

      @param id    (in): A globally unique element id
      @param owner (in): owner processor of the element
      */
      Shell7p(int id, int owner) : Core::Elements::Element(id, owner) {};


      //! copy constructor
      Shell7p(const Discret::Elements::Shell7p& other);

      //! copy assignment operator
      Shell7p& operator=(const Shell7p& other);

      //! move constructor
      Shell7p(Shell7p&& other) noexcept = default;

      //! move assignment operator
      Shell7p& operator=(Shell7p&& other) noexcept = default;
      //! @}

      [[nodiscard]] Core::Elements::Element* clone() const override;

      [[nodiscard]] int unique_par_object_id() const override
      {
        return Shell7pType::instance().unique_par_object_id();
      };

      void pack(Core::Communication::PackBuffer& data) const override;

      void unpack(Core::Communication::UnpackBuffer& buffer) override;

      [[nodiscard]] Core::Elements::ElementType& element_type() const override
      {
        return Shell7pType::instance();
      }

      [[nodiscard]] Core::FE::CellType shape() const override { return distype_; };

      [[nodiscard]] int num_line() const override;

      [[nodiscard]] int num_surface() const override;

      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

      [[nodiscard]] int num_dof_per_node(const Core::Nodes::Node& node) const override { return 6; }

      [[nodiscard]] int num_dof_per_element() const override { return 0; }

      //! @name Evaluation
      //! @{
      int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          std::vector<int>& dof_index_array, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          const Core::Conditions::Condition& condition, std::vector<int>& dof_index_array,
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

      bool read_element(const std::string& eletype, const std::string& eledistype,
          const Core::IO::InputParameterContainer& container) override;

      [[nodiscard]] const std::set<Inpar::Solid::EleTech>& get_ele_tech() const { return eletech_; }

      [[nodiscard]] std::shared_ptr<Mat::So3Material> solid_material(int nummat = 0) const;

      void print(std::ostream& os) const override;

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

      //! nodal director matrix
      Core::LinAlg::SerialDenseMatrix nodal_directors_{};

      //! flag, whether the post setup of materials is already called
      bool material_post_setup_ = false;

      //! shell calculation interface
      std::shared_ptr<Shell7pEleCalcInterface> shell_interface_ = nullptr;
    };
  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
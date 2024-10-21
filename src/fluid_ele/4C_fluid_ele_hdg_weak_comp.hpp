#ifndef FOUR_C_FLUID_ELE_HDG_WEAK_COMP_HPP
#define FOUR_C_FLUID_ELE_HDG_WEAK_COMP_HPP

#include "4C_config.hpp"

#include "4C_fem_general_dg_element.hpp"
#include "4C_fem_general_utils_polynomial.hpp"
#include "4C_fluid_ele.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Input
{
  class LineDefinition;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace ELEMENTS
  {
    class FluidHDGWeakCompType : public FluidType
    {
     public:
      std::string name() const override { return "FluidHDGWeakCompType"; }

      static FluidHDGWeakCompType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      Teuchos::RCP<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      Teuchos::RCP<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(
          Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

      virtual void compute_null_space(Core::FE::Discretization& dis, std::vector<double>& ns,
          const double* x0, int numdf, int dimns);

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
          override;

     private:
      static FluidHDGWeakCompType instance_;
    };


    /*!
    \brief HDG weakly compressible fluid element
    */
    class FluidHDGWeakComp : public Fluid, public Core::Elements::DgElement
    {
     public:
      //! @name constructors and destructors and related methods

      /*!
      \brief standard constructor
      */
      FluidHDGWeakComp(int id,  ///< A unique global id
          int owner             ///< ???
      );

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      FluidHDGWeakComp(const FluidHDGWeakComp& old);

      /*!
      \brief Deep copy this instance of fluid and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;


      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int unique_par_object_id() const override
      {
        return FluidHDGWeakCompType::instance().unique_par_object_id();
      }

      /*!
      \brief Pack this class so it can be communicated

      \ref pack and \ref unpack are used to communicate this element

      */
      void pack(Core::Communication::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref pack and \ref unpack are used to communicate this element
      */
      void unpack(Core::Communication::UnpackBuffer& buffer) override;

      /*!
      \brief Read input for this element
      */
      bool read_element(const std::string& eletype, const std::string& distype,
          const Core::IO::InputParameterContainer& container) override;

      //@}

      //! @name Access methods
      /*!
      \brief Get number of degrees of freedom per node, zero for the primary
      and the secondary dof set and equal to the given number for the tertiary dof set
      */
      int num_dof_per_node(const Core::Nodes::Node&) const override { return 0; }

      /*!
      \brief Returns the number of dofs per node for the ALE displacements
       */
      int num_dof_per_node_auxiliary() const override
      {
        return (Core::FE::get_dimension(distype_) + 0);
        // maybe + 1 to mimic the standard fluid dofset with velocity + pressure
        // this is maybe needed for the fluid-ALE coupling
      }

      /*!
      \brief Get number of degrees of freedom per face

      */
      int num_dof_per_face(const unsigned face) const override
      {
        return (1 + Core::FE::get_dimension(distype_)) * num_dof_per_component(face);
      }

      /*!
      \brief Get number of dofs per component per face
      */
      int num_dof_per_component(const unsigned face) const override
      {
        return Core::FE::get_basis_size(
            Core::FE::get_ele_face_shape_type(distype_), this->degree(), completepol_);
      }

      /*!
      \brief Get number of degrees of freedom per element, zero for the primary dof set
      and equal to the given number for the secondary dof set
      */
      int num_dof_per_element() const override { return 0; }

      /*!
      \brief Returns the number of dofs per element for the interior variables
       */
      int num_dof_per_element_auxiliary() const override
      {
        const int nsd_ = Core::FE::get_dimension(distype_);
        const int msd_ = (nsd_ * (nsd_ + 1.0)) / 2.0;
        return (msd_ + nsd_ + 1.0) * Core::FE::get_basis_size(distype_, degree_, completepol_);
      }

      /*!
       \brief Returns the degree of the element
       */
      int degree() const override { return degree_; }

      /*!
       \brief Returns the degree of the element
       */
      int uses_complete_polynomial_space() const { return completepol_; }

      //@}

      //! @name Evaluation

      /*!
      \brief Evaluate an element, that is, call the element routines to evaluate fluid
      element matrices and vectors or evaluate errors, statistics or updates etc. directly.

      \param params (in/out): ParameterList for communication between control routine
                              and elements
      \param elemat1 (out)  : matrix to be filled by element. If nullptr on input,
                              the controlling method does not expect the element to fill
                              this matrix.
      \param elemat2 (out)  : matrix to be filled by element. If nullptr on input,
                              the controlling method does not expect the element to fill
                              this matrix.
      \param elevec1 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \param elevec2 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \param elevec3 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \return 0 if successful, negative otherwise
      */
      int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

      //@}

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        return FluidHDGWeakCompType::instance();
      }

     private:
      // don't want = operator
      FluidHDGWeakComp& operator=(const FluidHDGWeakComp& old);

      // stores the degree of the element
      unsigned char degree_;

      // stores the polynomial type (tensor product or complete polynomial)
      bool completepol_;
    };  // class Fluid


  }  // namespace ELEMENTS
}  // namespace Discret



FOUR_C_NAMESPACE_CLOSE

#endif

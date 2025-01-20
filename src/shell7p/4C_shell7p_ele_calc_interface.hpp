// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SHELL7P_ELE_CALC_INTERFACE_HPP
#define FOUR_C_SHELL7P_ELE_CALC_INTERFACE_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_shell7p_utils.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Solid::ModelEvaluator
{
  class GaussPointDataOutputManager;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    // forward declaration
    class Shell7p;
    struct ShellStressIO
    {
      Inpar::Solid::StressType type;
      std::vector<char>& mutable_data;
    };

    struct ShellStrainIO
    {
      Inpar::Solid::StrainType type;
      std::vector<char>& mutable_data;
    };

    class Shell7pEleCalcInterface
    {
     public:
      //! destructor
      virtual ~Shell7pEleCalcInterface() = default;

      /*!
       * @brief Setup routine for the element
       * @param ele (in) : Reference to the element
       * @param solid_material (in) : Solid material of the element
       * @param locking_types (in) : Shell locking types
       * @param shell_data (in) : Shell data (thickness, number of ANS, SDC)
       * @param container (in) : Element input container
       */
      virtual void setup(Core::Elements::Element& ele, Mat::So3Material& solid_material,
          const Core::IO::InputParameterContainer& container,
          const Solid::Elements::ShellLockingTypes& locking_types,
          const Solid::Elements::ShellData& shell_data) = 0;

      /*!
       * @brief A setup routine for the materials after the whole input is read before evaluation.
       *
       * @param ele (in) : Reference to the element
       * @param solid_material (in) : Solid material of the element
       *
       */
      virtual void material_post_setup(
          Core::Elements::Element& ele, Mat::So3Material& solid_material) = 0;

      /*!
       * @brief Evaluate the force vector, stiffness matrix and mass matrix of the element
       *
       * @param ele (in) : Reference to the element
       * @param solid_material (in) : Solid material of the element
       * @param discretization  (in) : Reference to the discretization
       * @param nodal_directors  (in) : Nodal directors in reference frame
       * @param dof_index_array (in) : Location vector of the owned dofs
       * @param params (in) : ParameterList for communication between control routine, elements and
       * materials
       * @param force_vector (out) : Pointer to force vector or nullptr
       * @param stiffness_matrix (out) : Pointer to stiffness matrix or nullptr
       * @param mass_matrix (out) : Pointer to mass matrix or nullptr
       */
      virtual void evaluate_nonlinear_force_stiffness_mass(Core::Elements::Element& ele,
          Mat::So3Material& solid_material, const Core::FE::Discretization& discretization,
          const Core::LinAlg::SerialDenseMatrix& nodal_directors,
          const std::vector<int>& dof_index_array, Teuchos::ParameterList& params,
          Core::LinAlg::SerialDenseVector* force_vector,
          Core::LinAlg::SerialDenseMatrix* stiffness_matrix,
          Core::LinAlg::SerialDenseMatrix* mass_matrix) = 0;

      /*!
       * @brief Recover condensed EAS variables
       *
       * @param ele  (in) : Reference to the element
       * @param discretization  (in) : Reference to the discretization
       * @param dof_index_array (in) : Location vector of the owned dofs
       * @param params (in) : ParameterList for communication between control routine, elements and
       * materials
       */
      virtual void recover(Core::Elements::Element& ele,
          const Core::FE::Discretization& discretization, const std::vector<int>& dof_index_array,
          Teuchos::ParameterList& params, Solid::Elements::ParamsInterface& str_interface) = 0;

      /*!
       * @brief Evaluates the stresses and strains
       *
       * @param ele (in) : Reference to the element
       * @param solid_material (in) : Solid material of the element
       * @param strainIO  (in) : Strain input/output type
       * @param stressIO (in) : Stress input/output type
       * @param discretization  (in) : Reference to the discretization
       * @param nodal_directors  (in) : Nodal directors in reference frame
       * @param dof_index_array (in) : Location vector of the owned dofs
       * @param params (in) : ParameterList for communication between control routine, elements and
       * materials
       */
      virtual void calculate_stresses_strains(Core::Elements::Element& ele,
          Mat::So3Material& solid_material, const ShellStressIO& stressIO,
          const ShellStrainIO& strainIO, const Core::FE::Discretization& discretization,
          const Core::LinAlg::SerialDenseMatrix& nodal_directors,
          const std::vector<int>& dof_index_array, Teuchos::ParameterList& params) = 0;

      /*!
       * @brief Calculates the internal energy
       *
       * @param ele (in) : Reference to the element
       * @param solid_material (in) : Solid material of the element
       * @param discretization (in) : Reference to the discretization
       * @param nodal_directors  (in) : Nodal directors in reference frame
       * @param dof_index_array (in) : Location vector of the owned dofs
       * @param params (in) : ParameterList for communication between control routine, elements and
       * materials
       */
      virtual double calculate_internal_energy(Core::Elements::Element& ele,
          Mat::So3Material& solid_material, const Core::FE::Discretization& discretization,
          const Core::LinAlg::SerialDenseMatrix& nodal_directors,
          const std::vector<int>& dof_index_array, Teuchos::ParameterList& params) = 0;

      /*!
       * @brief The update routine of the element
       *
       * @param ele  (in) : Reference to the element
       * @param solid_material (in) : Solid material of the element
       * @param discretization (in) : Reference to the discretization
       * @param nodal_directors  (in) : Nodal directors in reference frame
       * @param dof_index_array (in) : The location array of the owned dofs
       * @param params (in/[out]) : A ParameterList to pass values from the time integrator to
       * the elements/materials
       */
      virtual void update(Core::Elements::Element& ele, Mat::So3Material& solid_material,
          const Core::FE::Discretization& discretization,
          const Core::LinAlg::SerialDenseMatrix& nodal_directors,
          const std::vector<int>& dof_index_array, Teuchos::ParameterList& params) = 0;

      /*!
       * @brief Reset time step of material (for time adaptivity)
       *
       * @param ele  (in) : Reference to the element
       * @param solid_material (in) : Solid material of the element
       */
      virtual void reset_to_last_converged(
          Core::Elements::Element& ele, Mat::So3Material& solid_material) = 0;

      /*!
       * \brief Query data to be visualized using BINIO of a given name
       *
       * @param name (in):   Name of data that is currently processed for visualization
       * @param data (out):  data to be filled by element if it recognizes the name
       */
      virtual void vis_data(const std::string& name, std::vector<double>& data) = 0;
    };  // class Shell7pEleInterface
  }  // namespace Elements

}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif

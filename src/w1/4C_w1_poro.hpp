// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_W1_PORO_HPP
#define FOUR_C_W1_PORO_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_w1.hpp"
#include "4C_w1_poro_eletypes.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class StructPoro;
  class FluidPoro;
  class FluidPoroMultiPhase;
}  // namespace Mat

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    /*!
    \brief A C++ version of a 2 dimensional solid element with modifications for porous media

    */
    template <Core::FE::CellType distype>
    class Wall1Poro : public Wall1
    {
     public:
      //!@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner : elements owner
      */
      Wall1Poro(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      Wall1Poro(const Wall1Poro& old);

      //!@}

      //! number of element nodes (
      static constexpr int numnod_ = Core::FE::num_nodes(distype);

      //! number of strains per node
      static constexpr int numstr_ = 3;

      //! number or second derivatives
      static constexpr int numderiv2_ = Core::FE::DisTypeToNumDeriv2<distype>::numderiv2;

      //! number of degrees of freedom of element
      static constexpr int numdof_ = numnod_ * noddof_;

      //! total gauss points per element
      int numgpt_;

      //! @name Access methods

      /*!
      \brief Deep copy this instance of Solid3 and return pointer to the copy

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
        switch (distype)
        {
          case Core::FE::CellType::tri3:
            return Discret::Elements::WallTri3PoroType::instance().unique_par_object_id();
          case Core::FE::CellType::quad4:
            return Discret::Elements::WallQuad4PoroType::instance().unique_par_object_id();
          case Core::FE::CellType::quad9:
            return Discret::Elements::WallQuad9PoroType::instance().unique_par_object_id();
          case Core::FE::CellType::nurbs4:
            return Discret::Elements::WallNurbs4PoroType::instance().unique_par_object_id();
          case Core::FE::CellType::nurbs9:
            return Discret::Elements::WallNurbs9PoroType::instance().unique_par_object_id();
          default:
            FOUR_C_THROW("unknown element type");
            break;
        }
        return -1;
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

      //! Get vector of std::shared_ptrs to the lines of this element
      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

      //! Get vector of std::shared_ptrs to the surfaces of this element
      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

      //! @name Access methods

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        switch (distype)
        {
          case Core::FE::CellType::tri3:
            return Discret::Elements::WallTri3PoroType::instance();
          case Core::FE::CellType::quad4:
            return Discret::Elements::WallQuad4PoroType::instance();
          case Core::FE::CellType::quad9:
            return Discret::Elements::WallQuad9PoroType::instance();
          case Core::FE::CellType::nurbs4:
            return Discret::Elements::WallNurbs4PoroType::instance();
          case Core::FE::CellType::nurbs9:
            return Discret::Elements::WallNurbs9PoroType::instance();
          default:
            FOUR_C_THROW("unknown element type");
            break;
        }
        return Discret::Elements::WallQuad4PoroType::instance();
      }

      //!@}

      //! @name Evaluation

      void pre_evaluate(
          Teuchos::ParameterList&
              params,  //!< ParameterList for communication between control routine and elements
          Core::FE::Discretization& discretization,  //!< pointer to discretization for de-assembly
          Core::Elements::LocationArray& la          //!< location array for de-assembly
      );

      /*!
      \brief Evaluate an element

      Evaluate element stiffness, mass, internal forces, etc.

      If nullptr on input, the controlling method does not expect the element
      to fill these matrices or vectors.

      \return 0 if successful, negative otherwise
      */
      int evaluate(
          Teuchos::ParameterList&
              params,  //!< ParameterList for communication between control routine and elements
          Core::FE::Discretization& discretization,  //!< pointer to discretization for de-assembly
          Core::Elements::LocationArray& la,         //!< location array for de-assembly
          Core::LinAlg::SerialDenseMatrix&
              elemat1,  //!< (stiffness-)matrix to be filled by element.
          Core::LinAlg::SerialDenseMatrix& elemat2,  //!< (mass-)matrix to be filled by element.
          Core::LinAlg::SerialDenseVector&
              elevec1,  //!< (internal force-)vector to be filled by element
          Core::LinAlg::SerialDenseVector& elevec2,  //!< vector to be filled by element
          Core::LinAlg::SerialDenseVector& elevec3   //!< vector to be filled by element
          ) override;


      //! initialize the inverse of the jacobian and its determinant in the material configuration
      virtual void init_element();

      //!@}

      //! @name Input and Creation

      /*!
      \brief Read input for this element
      */
      bool read_element(const std::string& eletype, const std::string& eledistype,
          const Core::IO::InputParameterContainer& container) override;

      //!@}

      /*!
      \brief Read input for this element
      */
      /*!
      \brief Query names of element data to be visualized using BINIO

      The element fills the provided map with key names of
      visualization data the element wants to visualize AT THE CENTER
      of the element geometry. The values is supposed to be dimension of the
      data to be visualized. It can either be 1 (scalar), 3 (vector), 6 (sym. tensor)
      or 9 (nonsym. tensor)

      Example:
      \code
        // Name of data is 'Owner', dimension is 1 (scalar value)
        names.insert(std::pair<string,int>("Owner",1));
        // Name of data is 'StressesXYZ', dimension is 6 (sym. tensor value)
        names.insert(std::pair<string,int>("StressesXYZ",6));
      \endcode

      \param names (out): On return, the derived class has filled names with
                          key names of data it wants to visualize and with int dimensions
                          of that data.
      */
      void vis_names(std::map<std::string, int>& names) override;

      /*!
      \brief Query data to be visualized using BINIO of a given name

      The method is supposed to call this base method to visualize the owner of
      the element.
      If the derived method recognizes a supported data name, it shall fill it
      with corresponding data.
      If it does NOT recognizes the name, it shall do nothing.

      \warning The method must not change size of data

      \param name (in):   Name of data that is currently processed for visualization
      \param data (out):  data to be filled by element if element recognizes the name
      */
      bool vis_data(const std::string& name, std::vector<double>& data) override;


      //! return anisotropic permeability directions (used for cloning)
      const std::vector<std::vector<double>>& get_anisotropic_permeability_directions() const
      {
        return anisotropic_permeability_directions_;
      }

      //! return scaling coefficients for anisotropic permeability (used for cloning)
      const std::vector<std::vector<double>>& get_anisotropic_permeability_nodal_coeffs() const
      {
        return anisotropic_permeability_nodal_coeffs_;
      }

      //! don't want = operator
      Wall1Poro& operator=(const Wall1Poro& old) = delete;

     protected:
      /*!
      \brief Evaluate an element

      Evaluate Wall1_poro element stiffness, mass, internal forces, etc.
      Templated evaluate routine of element matrixes

      If nullptr on input, the controlling method does not expect the element
      to fill these matrices or vectors.

      \return 0 if successful, negative otherwise
      */
      virtual int my_evaluate(
          Teuchos::ParameterList&
              params,  //!< ParameterList for communication between control routine and elements
          Core::FE::Discretization& discretization,  //!< pointer to discretization for de-assembly
          Core::Elements::LocationArray& la,         //!< location array for de-assembly
          Core::LinAlg::SerialDenseMatrix&
              elemat1,  //!< (stiffness-)matrix to be filled by element.
          Core::LinAlg::SerialDenseMatrix& elemat2,  //!< (mass-)matrix to be filled by element.
          Core::LinAlg::SerialDenseVector&
              elevec1,  //!< (internal force-)vector to be filled by element
          Core::LinAlg::SerialDenseVector& elevec2,  //!< vector to be filled by element
          Core::LinAlg::SerialDenseVector& elevec3   //!< vector to be filled by element
      );

      //! vector of inverses of the jacobian in material frame
      std::vector<Core::LinAlg::Matrix<numdim_, numdim_>> invJ_;
      //! determinant of Jacobian in material frame
      std::vector<double> detJ_;
      //! vector of coordinates of current integration point in reference coordinates
      std::vector<Core::LinAlg::Matrix<numdim_, 1>> xsi_;

      //! Calculate nonlinear stiffness and internal force for poroelasticity problems
      virtual void nonlinear_stiffness_poroelast(std::vector<int>& lm,  //!< location matrix
          Core::LinAlg::Matrix<numdim_, numnod_>& disp,                 //!< current displacements
          Core::LinAlg::Matrix<numdim_, numnod_>& vel,                  //!< current velocities
          Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,       //!< fluid velocity of element
          Core::LinAlg::Matrix<numnod_, 1>& epreaf,             //!< fluid pressure of element
          Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,  //!< element stiffness matrix
          Core::LinAlg::Matrix<numdof_, numdof_>* reamatrix,    //!< element reactive matrix
          Core::LinAlg::Matrix<numdof_, 1>* force,              //!< element internal force vector
          Teuchos::ParameterList& params  //!< algorithmic parameters e.g. time
      );

      //! Calculate nonlinear stiffness and internal force for poroelasticity problems (pressure
      //! based formulation)
      virtual void nonlinear_stiffness_poroelast_pressure_based(
          std::vector<int>& lm,                          //!< location matrix
          Core::LinAlg::Matrix<numdim_, numnod_>& disp,  //!< current displacements
          const std::vector<double>& ephi,  //!< current primary variable for poro-multiphase flow
          Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,  //!< element stiffness matrix
          Core::LinAlg::Matrix<numdof_, 1>* force,              //!< element internal force vector
          Teuchos::ParameterList& params  //!< algorithmic parameters e.g. time
      );

      //! Calculate coupling terms in nonlinear stiffness and internal force for poroelasticity
      //! problems
      virtual void coupling_poroelast(std::vector<int>& lm,  //!< location matrix
          Core::LinAlg::Matrix<numdim_, numnod_>& disp,      //!< current displacements
          Core::LinAlg::Matrix<numdim_, numnod_>& vel,       //!< current velocities
          Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,    //!< fluid velocity of element
          Core::LinAlg::Matrix<numnod_, 1>& epreaf,          //!< fluid pressure of element
          Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>*
              stiffmatrix,  //!< element stiffness matrix
          Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>*
              reamatrix,                            //!< element reactive matrix
          Core::LinAlg::Matrix<numdof_, 1>* force,  //!< element internal force vector
          Teuchos::ParameterList& params            //!< algorithmic parameters e.g. time
      );

      //! Calculate coupling terms in nonlinear stiffness and internal force for poroelasticity
      //! problems (pressure based formulation)
      virtual void coupling_poroelast_pressure_based(std::vector<int>& lm,  //!< location matrix
          Core::LinAlg::Matrix<numdim_, numnod_>& disp,  //!< current displacements
          const std::vector<double>& ephi,  //! current primary variable for poro-multiphase flow
          Core::LinAlg::SerialDenseMatrix& couplmat,  //!< element stiffness matrix
          Teuchos::ParameterList& params              //!< algorithmic parameters e.g. time
      );

      //! Gauss Point Loop evaluating stiffness and force
      void gauss_point_loop(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<numdim_, numnod_>& xrefe,
          const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr,
          const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp,
          const Core::LinAlg::Matrix<numdim_, numnod_>& nodalvel,
          const Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,
          const Core::LinAlg::Matrix<numnod_, 1>& epreaf,
          const Core::LinAlg::Matrix<numnod_, 1>* porosity_dof,
          Core::LinAlg::Matrix<numdof_, numdof_>& erea_v,
          Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,
          Core::LinAlg::Matrix<numdof_, numdof_>* reamatrix,
          Core::LinAlg::Matrix<numdof_, 1>* force);

      //! Gauss Point Loop evaluating stiffness (off diagonal)
      void gauss_point_loop_od(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<numdim_, numnod_>& xrefe,
          const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr,
          const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp,
          const Core::LinAlg::Matrix<numdim_, numnod_>& nodalvel,
          const Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,
          const Core::LinAlg::Matrix<numnod_, 1>& epreaf,
          const Core::LinAlg::Matrix<numnod_, 1>* porosity_dof,
          Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>& ecoupl);

      //! Gauss Point Loop evaluating stiffness and force
      void gauss_point_loop_pressure_based(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<numdim_, numnod_>& xrefe,
          const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr,
          const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp, const std::vector<double>& ephi,
          Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,
          Core::LinAlg::Matrix<numdof_, 1>* force);

      //! Gauss Point Loop evaluating stiffness (off diagonal)
      void gauss_point_loop_od_pressure_based(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<numdim_, numnod_>& xrefe,
          const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr,
          const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp, const std::vector<double>& ephi,
          Core::LinAlg::SerialDenseMatrix& couplmat);

      //! compute porosity at gausspoint and linearization of porosity w.r.t. structural
      //! displacements
      virtual void compute_porosity_and_linearization(Teuchos::ParameterList& params,
          const double& press, const double& J, const int& gp,
          const Core::LinAlg::Matrix<numnod_, 1>& shapfct,
          const Core::LinAlg::Matrix<numnod_, 1>* myporosity,
          const Core::LinAlg::Matrix<1, numdof_>& dJ_dus, double& porosity,
          Core::LinAlg::Matrix<1, numdof_>& dphi_dus);

      //! compute porosity at gausspoint and linearization of porosity w.r.t. fluid pressure
      virtual void compute_porosity_and_linearization_od(Teuchos::ParameterList& params,
          const double& press, const double& J, const int& gp,
          const Core::LinAlg::Matrix<numnod_, 1>& shapfct,
          const Core::LinAlg::Matrix<numnod_, 1>* myporosity, double& porosity, double& dphi_dp);

      //! Compute Jacobian determinant
      void compute_jacobian_determinant_volume_change_and_linearizations(double& J,
          double& volchange, Core::LinAlg::Matrix<1, numdof_>& dJ_dus,
          Core::LinAlg::Matrix<1, numdof_>& dvolchange_dus,
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
          const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
          const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp);

      //! Compute Jacobian determinant
      void compute_jacobian_determinant_volume_change(double& J, double& volchange,
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
          const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
          const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp);

      //! fill stiffness matrix and rhs vector for darcy flow
      void fill_matrix_and_vectors(const int& gp, const Core::LinAlg::Matrix<numnod_, 1>& shapefct,
          const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& press,
          const double& porosity, const Core::LinAlg::Matrix<numdim_, 1>& velint,
          const Core::LinAlg::Matrix<numdim_, 1>& fvelint,
          const Core::LinAlg::Matrix<numdim_, numdim_>& fvelder,
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
          const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
          const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
          const Core::LinAlg::Matrix<numdim_, 1>& Finvgradp,
          const Core::LinAlg::Matrix<1, numdof_>& dphi_dus,
          const Core::LinAlg::Matrix<1, numdof_>& dJ_dus,
          const Core::LinAlg::Matrix<numstr_, numdof_>& dCinv_dus,
          const Core::LinAlg::Matrix<numdim_, numdof_>& dFinvdus_gradp,
          const Core::LinAlg::Matrix<numdim_ * numdim_, numdof_>& dFinvTdus,
          Core::LinAlg::Matrix<numdof_, numdof_>& erea_v,
          Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,
          Core::LinAlg::Matrix<numdof_, 1>* force, Core::LinAlg::Matrix<numstr_, 1>& fstress);

      //! fill stiffness matrix and rhs vector for pressure-based formulation
      void fill_matrix_and_vectors_pressure_based(const int& gp,
          const Core::LinAlg::Matrix<numnod_, 1>& shapefct,
          const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& press,
          const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
          const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
          const Core::LinAlg::Matrix<1, numdof_>& dJ_dus,
          const Core::LinAlg::Matrix<numstr_, numdof_>& dCinv_dus,
          const Core::LinAlg::Matrix<1, numdof_>& dps_dus,
          Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,
          Core::LinAlg::Matrix<numdof_, 1>* force);

      //! fill stiffness matrix and rhs vector for brinkman flow
      void fill_matrix_and_vectors_brinkman(const int& gp, const double& J, const double& porosity,
          const Core::LinAlg::Matrix<numdim_, numdim_>& fvelder,
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
          const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
          const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
          const Core::LinAlg::Matrix<1, numdof_>& dphi_dus,
          const Core::LinAlg::Matrix<1, numdof_>& dJ_dus,
          const Core::LinAlg::Matrix<numstr_, numdof_>& dCinv_dus,
          const Core::LinAlg::Matrix<numdim_ * numdim_, numdof_>& dFinvTdus,
          Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,
          Core::LinAlg::Matrix<numdof_, 1>* force, Core::LinAlg::Matrix<numstr_, 1>& fstress);

      //! fill stiffness matrix and rhs vector for darcy flow (off diagonal terms)
      void fill_matrix_and_vectors_od(const int& gp,
          const Core::LinAlg::Matrix<numnod_, 1>& shapefct,
          const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ, const double& J,
          const double& porosity, const double& dphi_dp,
          const Core::LinAlg::Matrix<numdim_, 1>& velint,
          const Core::LinAlg::Matrix<numdim_, 1>& fvelint,
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
          const Core::LinAlg::Matrix<numdim_, 1>& Gradp,
          const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
          const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
          Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>& ecoupl);

      //! fill stiffness matrix (off diagonal terms) -- pressure-based formulation
      void fill_matrix_and_vectors_od_pressure_based(const int& gp,
          const Core::LinAlg::Matrix<numnod_, 1>& shapefct,
          const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ, const double& J,
          const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
          const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
          const std::vector<double>& solpressderiv, Core::LinAlg::SerialDenseMatrix& couplmat);

      //! fill stiffness matrix and rhs vector for darcy brinkman flow (off diagonal terms)
      void fill_matrix_and_vectors_brinkman_od(const int& gp,
          const Core::LinAlg::Matrix<numnod_, 1>& shapefct,
          const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ, const double& J,
          const double& porosity, const double& dphi_dp,
          const Core::LinAlg::Matrix<numdim_, numdim_>& fvelder,
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
          const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
          const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
          Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>& ecoupl);

      //! Compute  nonlinear b-operator
      void compute_b_operator(Core::LinAlg::Matrix<numstr_, numdof_>& bop,
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
          const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ);

      //! evaluate shape functions and their derivatives at gauss point
      void compute_shape_functions_and_derivatives(const int& gp,
          Core::LinAlg::Matrix<numnod_, 1>& shapefct, Core::LinAlg::Matrix<numdim_, numnod_>& deriv,
          Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ);

      //! Compute Jacobian determinant
      double compute_jacobian_determinant(const int& gp,
          const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr,
          const Core::LinAlg::Matrix<numdim_, numnod_>& deriv);

      //! Compute Linearization Of Jacobian
      void compute_linearization_of_jacobian(Core::LinAlg::Matrix<1, numdof_>& dJ_dus,
          const double& J, const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv);

      //! helper functions to get element vectors from global vector
      void compute_auxiliary_values(const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
          const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
          const Core::LinAlg::Matrix<numdim_, 1>& Gradp,
          Core::LinAlg::Matrix<numdim_ * numdim_, numdof_>& dFinvTdus,
          Core::LinAlg::Matrix<numdim_, 1>& Finvgradp,
          Core::LinAlg::Matrix<numdim_, numdof_>& dFinvdus_gradp,
          Core::LinAlg::Matrix<numstr_, numdof_>& dCinv_dus);

      //! push forward of material stresses to the current, spatial configuration (for output only)
      void p_k2to_cauchy(Core::LinAlg::Matrix<Wall1::numstr_, 1>& stress,
          Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
          Core::LinAlg::Matrix<numdim_, numdim_>& cauchystress);

      //! Compute deformation gradient
      void compute_def_gradient(Core::LinAlg::Matrix<numdim_, numdim_>&
                                    defgrd,  //!<<    (i) deformation gradient at gausspoint
          const Core::LinAlg::Matrix<numdim_, numnod_>&
              N_XYZ,  //!<<    (i) derivatives of shape functions w.r.t. reference coordinates
          const Core::LinAlg::Matrix<numdim_, numnod_>&
              xcurr  //!<<    (i) current position of gausspoint
      );

      //! helper functions to get element vectors from global vector
      void extract_values_from_global_vector(
          const Core::FE::Discretization& discretization,        //!< discretization
          const int& dofset,                                     //!< number of dofset
          const std::vector<int>& lm,                            //!< location vector
          Core::LinAlg::Matrix<numdim_, numnod_>* matrixtofill,  //!< vector field
          Core::LinAlg::Matrix<numnod_, 1>* vectortofill,        //!< scalar field
          const std::string state                                //!< state of the global vector
      );

      //! Compute Solid-pressure derivative w.r.t. primary variable at GP
      void compute_sol_pressure_deriv(const std::vector<double>& phiAtGP,  //!<< primary variable
          const int numfluidphases,             //!<< number of fluid phases
          std::vector<double>& solidpressderiv  //!<< solid pressure derivative at GP
      );

      //! Compute solid pressure at GP
      double compute_sol_pressure_at_gp(
          const int totalnumdofpernode,       //!<< total number of multiphase dofs
          const int numfluidphases,           //!<< number of fluid phases
          const std::vector<double>& phiAtGP  //!<< primary variable
      );

      //! recalculate solid pressure at GP in case of volfracs
      double recalculate_sol_pressure_at_gp(double press, const double porosity,
          const int totalnumdofpernode, const int numfluidphases, const int numvolfrac,
          const std::vector<double>& phiAtGP);

      //! recalculate solid pressure derivative in case of volfracs
      void recalculate_sol_pressure_deriv(const std::vector<double>& phiAtGP,
          const int totalnumdofpernode, const int numfluidphases, const int numvolfrac,
          const double press, const double porosity, std::vector<double>& solidpressderiv);

      //! Compute primary variable for multiphase flow at GP
      void compute_primary_variable_at_gp(
          const std::vector<double>& ephi,                   //!<< primary variable at node
          const int totalnumdofpernode,                      //!<< total number of multiphase dofs
          const Core::LinAlg::Matrix<numnod_, 1>& shapefct,  //!<< shapefct
          std::vector<double>& phiAtGP                       //!<< primary variable at GP
      );

      //! Compute linearizaton of solid press w.r.t. displacements
      //! only needed if additional volume fractions are present and porosity depends on
      //! Jacobian of deformation gradient
      void compute_linearization_of_sol_press_wrt_disp(const double fluidpress,
          const double porosity, const int totalnumdofpernode, const int numfluidphases,
          const int numvolfrac, const std::vector<double>& phiAtGP,
          const Core::LinAlg::Matrix<1, numdof_>& dphi_dus,
          Core::LinAlg::Matrix<1, numdof_>& dps_dus);

      //! get materials (solid and fluid)
      void get_materials();

      //! get materials (solid and fluidmulti)
      void get_materials_pressure_based();

      //! anisotropic permeability directions in the element definition
      void read_anisotropic_permeability_directions_from_element_line_definition(
          const Core::IO::InputParameterContainer& container);

      //! read nodal anisotropic permeability scaling coefficients in the element definition
      void read_anisotropic_permeability_nodal_coeffs_from_element_line_definition(
          const Core::IO::InputParameterContainer& container);

      //! interpolate the anisotropic permeability coefficients at GP from nodal values
      std::vector<double> compute_anisotropic_permeability_coeffs_at_gp(
          const Core::LinAlg::Matrix<numnod_, 1>& shapefct) const;

      //! Gauss integration rule
      Core::FE::GaussIntegration intpoints_;

      //! flag indicating initialization of element
      bool init_;

      //! flag for scatra coupling
      bool scatra_coupling_;

      //! corresponding fluid material
      std::shared_ptr<Mat::FluidPoro> fluid_mat_;

      //! corresponding multiphase fluid material
      std::shared_ptr<Mat::FluidPoroMultiPhase> fluidmulti_mat_;

      //! own poro structure material
      std::shared_ptr<Mat::StructPoro> struct_mat_;

      //! weights for nurbs elements
      Core::LinAlg::Matrix<numnod_, 1> weights_;
      //! knot vector for nurbs elements
      std::vector<Core::LinAlg::SerialDenseVector> myknots_;

      //! directions for anisotropic permeability
      std::vector<std::vector<double>> anisotropic_permeability_directions_;

      //! scaling coefficients for nodal anisotropic permeability
      std::vector<std::vector<double>> anisotropic_permeability_nodal_coeffs_;
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif

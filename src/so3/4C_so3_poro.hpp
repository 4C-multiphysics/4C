// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SO3_PORO_HPP
#define FOUR_C_SO3_PORO_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_inpar_structure.hpp"

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
    \brief A C++ version of a 3 dimensional solid element with modifications for porous media

    A structural 3 dimensional solid displacement element for large deformations
    and (near)-incompressibility.

    */
    template <class So3Ele, Core::FE::CellType distype>
    class So3Poro : public So3Ele
    {
      //! @name Friends
      friend class SoTet4PoroType;
      friend class SoTet10PoroType;
      friend class SoHex8PoroType;
      friend class SoHex27PoroType;

     public:
      //!@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner : elements owner
      */
      So3Poro(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      So3Poro(const So3Poro& old);

      //!@}

      //! number of element nodes (
      static constexpr int numnod_ = Core::FE::num_nodes<distype>;

      //! number of space dimensions
      static constexpr int numdim_ = Core::FE::dim<distype>;

      //! number of dofs per node
      static constexpr int noddof_ = numdim_;

      //! total dofs per element
      static constexpr int numdof_ = noddof_ * numnod_;

      //! number of strains per node
      static constexpr int numstr_ = (numdim_ * (numdim_ + 1)) / 2;

      //! number of components necessary to store second derivatives
      /*!
       1 component  for nsd=1:  (N,xx)

       3 components for nsd=2:  (N,xx ; N,yy ; N,xy)

       6 components for nsd=3:  (N,xx ; N,yy ; N,zz ; N,xy ; N,xz ; N,yz)
      */
      static constexpr int numderiv2_ = Core::FE::DisTypeToNumDeriv2<distype>::numderiv2;

      //! total gauss points per element
      int numgpt_;

      //! @name Acess methods

      /*!
      \brief Deep copy this instance of Solid3 and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;

      //! get global id of element
      int id() const;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int unique_par_object_id() const override;

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
      \brief Get vector of std::shared_ptrs to the lines of this element

      */
      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

      /*!
      \brief Get vector of std::shared_ptrs to the surfaces of this element

      */
      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

      //! @name Access methods

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override;

      //!@}

      /*!
      \brief Return value how expensive it is to evaluate this element

      \param double (out): cost to evaluate this element
      */
      double evaluation_cost() override { return 10000.0; }

      //! @name Evaluation

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

      virtual void pre_evaluate(
          Teuchos::ParameterList&
              params,  //!< ParameterList for communication between control routine and elements
          Core::FE::Discretization& discretization,  //!< pointer to discretization for de-assembly
          Core::Elements::LocationArray& la          //!< location array for de-assembly
      );


      //! initialize the inverse of the jacobian and its determinant in the material configuration
      virtual void init_element();

      //!@}

      //! @name Input and Creation

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
        names.insert(std::pair<std::string,int>("Owner",1));
        // Name of data is 'StressesXYZ', dimension is 6 (sym. tensor value)
        names.insert(std::pair<std::string,int>("StressesXYZ",6));
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

      /*!
      \brief Read input for this element
      */
      bool read_element(const std::string& eletype, const std::string& eledistype,
          const Core::IO::InputParameterContainer& container) override;

      //!@}

      //! compute porosity at gausspoint
      virtual void compute_porosity(Teuchos::ParameterList& params, double press, double J, int gp,
          double& porosity, double* dphi_dp, double* dphi_dJ, double* dphi_dJdp, double* dphi_dJJ,
          double* dphi_dpp, bool save);

      //! compute porosity at gauss point at one face of the element
      virtual void compute_surf_porosity(Teuchos::ParameterList& params, double press, double J,
          int surfnum, int gp, double& porosity, double* dphi_dp, double* dphi_dJ,
          double* dphi_dJdp, double* dphi_dJJ, double* dphi_dpp, bool save);

      //! return time derivative of reference porosity
      virtual double ref_porosity_time_deriv();

      //! evaluate Cauchy stress at given point in parameter space
      virtual void get_cauchy_n_dir_and_derivatives_at_xi(const Core::LinAlg::Matrix<3, 1>& xi,
          const std::vector<double>& disp, const std::vector<double>& pres,
          const Core::LinAlg::Matrix<3, 1>& n, const Core::LinAlg::Matrix<3, 1>& dir,
          double& cauchy_n_dir, Core::LinAlg::SerialDenseMatrix* d_cauchyndir_dd,
          Core::LinAlg::SerialDenseMatrix* d_cauchyndir_dp,
          Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dn,
          Core::LinAlg::Matrix<3, 1>* d_cauchyndir_ddir,
          Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dxi);

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
      So3Poro& operator=(const So3Poro& old) = delete;

     protected:
      /*!
      \brief Evaluate an element

      Evaluate So3_poro element stiffness, mass, internal forces, etc.
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

      //! action parameters recognized by so_hex8
      enum ActionType
      {
        none,
        calc_struct_nlnstiff,
        calc_struct_internalforce,
        calc_struct_nlnstiffmass,
        calc_struct_stress,
        calc_struct_multidofsetcoupling,  //!< structure-fluid coupling: internal force, stiffness
                                          //!< for poroelasticity (structural part)
        interpolate_porosity_to_given_point
      };

      //! vector of inverses of the jacobian in material frame
      std::vector<Core::LinAlg::Matrix<numdim_, numdim_>> invJ_;
      //! determinant of Jacobian in material frame
      std::vector<double> detJ_;
      //! vector of coordinates of current integration point in reference coordinates
      std::vector<Core::LinAlg::Matrix<numdim_, 1>> xsi_;

      //! Calculate nonlinear stiffness and internal force for poroelasticity problems
      void nonlinear_stiffness_poroelast(std::vector<int>& lm,  //!< location matrix
          Core::LinAlg::Matrix<numdim_, numnod_>& disp,         //! current displacements
          Core::LinAlg::Matrix<numdim_, numnod_>& vel,          //! current velocities
          Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,       //! fluid velocity of element
          Core::LinAlg::Matrix<numnod_, 1>& epreaf,             //! fluid pressure of element
          Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,  //! element stiffness matrix
          Core::LinAlg::Matrix<numdof_, numdof_>* reamatrix,    //! element reactive matrix
          Core::LinAlg::Matrix<numdof_, 1>* force,              //! element internal force vector
          Teuchos::ParameterList& params                        //! algorithmic parameters e.g. time
      );

      //! Calculate nonlinear stiffness and internal force for poroelasticity problems (pressure
      //! based formulation)
      virtual void nonlinear_stiffness_poroelast_pressure_based(
          std::vector<int>& lm,                          //!< location matrix
          Core::LinAlg::Matrix<numdim_, numnod_>& disp,  //! current displacements
          const std::vector<double>& ephi,  //! current primary variable for poro-multiphase flow
          Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,  //! element stiffness matrix
          Core::LinAlg::Matrix<numdof_, 1>* force,              //! element internal force vector
          Teuchos::ParameterList& params                        //! algorithmic parameters e.g. time
      );

      //! Calculate coupling terms in nonlinear stiffness and internal force for poroelasticity
      //! problems
      void coupling_poroelast(std::vector<int>& lm,        //!< location matrix
          Core::LinAlg::Matrix<numdim_, numnod_>& disp,    //! current displacements
          Core::LinAlg::Matrix<numdim_, numnod_>& vel,     //! current velocities
          Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,  //! fluid velocity of element
          Core::LinAlg::Matrix<numnod_, 1>& epreaf,        //! fluid pressure of element
          Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>*
              stiffmatrix,  //! element stiffness matrix
          Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>*
              reamatrix,                            //! element reactive matrix
          Core::LinAlg::Matrix<numdof_, 1>* force,  //! element internal force vector
          Teuchos::ParameterList& params            //! algorithmic parameters e.g. time
      );

      //! Calculate coupling terms in nonlinear stiffness and internal force for poroelasticity
      //! problems (pressure based formulation)
      virtual void coupling_poroelast_pressure_based(std::vector<int>& lm,  //!< location matrix
          Core::LinAlg::Matrix<numdim_, numnod_>& disp,  //! current displacements
          const std::vector<double>& ephi,  //! current primary variable for poro-multiphase flow
          Core::LinAlg::SerialDenseMatrix& couplmat,  //!< element stiffness matrix
          Teuchos::ParameterList& params              //!< algorithmic parameters e.g. time
      );

      //! Calculate coupling stress for poroelasticity problems
      virtual void coupling_stress_poroelast(
          Core::LinAlg::Matrix<numdim_, numnod_>& disp,    //! current displacements
          Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,  //! fluid velocity of element
          Core::LinAlg::Matrix<numnod_, 1>& epreaf,        //! fluid pressure of element
          Core::LinAlg::SerialDenseMatrix* elestress,      //! stresses at GP
          Core::LinAlg::SerialDenseMatrix* elestrain,      //! strains at GP
          Teuchos::ParameterList& params,                  //! algorithmic parameters e.g. time
          const Inpar::Solid::StressType iostress          //! stress output option
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
          Core::LinAlg::Matrix<numdof_, 1>* force);

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
          Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>* stiffmatrix);

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
          Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>* stiffmatrix);

      //! Gauss Point Loop evaluating stiffness (off diagonal)
      void gauss_point_loop_od(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<numdim_, numnod_>& xrefe,
          const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr,
          const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp,
          const Core::LinAlg::Matrix<numdim_, numnod_>& nodalvel,
          const Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,
          const Core::LinAlg::Matrix<numnod_, 1>& epreaf,
          Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>* stiffmatrix);

      //! helper functions to get element vectors from global vector
      void extract_values_from_global_vector(
          const Core::FE::Discretization& discretization,        //!< discretization
          const int& dofset,                                     //!< number of dofset
          const std::vector<int>& lm,                            //!< location vector
          Core::LinAlg::Matrix<numdim_, numnod_>* matrixtofill,  //!< vector field
          Core::LinAlg::Matrix<numnod_, 1>* vectortofill,        //!< scalar field
          const std::string& state                               //!< state of the global vector
      );

      //! push forward of material stresses to the current, spatial configuration
      void p_k2to_cauchy(Core::LinAlg::Matrix<numstr_, 1>& stress,
          Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
          Core::LinAlg::Matrix<numdim_, numdim_>& cauchystress);

      //! get materials (solid and fluid)
      void get_materials();

      //! get materials (solid and fluid) for pressure based formulation
      void get_materials_pressure_based();

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

      //! evaluate shape functions and their derivatives at gauss point
      void compute_shape_functions_and_derivatives(const int& gp,
          Core::LinAlg::Matrix<numnod_, 1>& shapefct, Core::LinAlg::Matrix<numdim_, numnod_>& deriv,
          Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ);

      //! Compute Jacobian determinant and the volume change
      void compute_jacobian_determinant_volume_change(double& J, double& volchange,
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
          const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
          const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp);

      //! Compute Jacobian determinant and the volume change and its linearizations
      virtual void compute_jacobian_determinant_volume_change_and_linearizations(
          double& J,          //!< (o) Jacobian determinant
          double& volchange,  //!< (o) Change of Volume
          Core::LinAlg::Matrix<1, numdof_>&
              dJ_dus,  //!< (o) Linearization of Jacobian determinant  w.r.t displacements
          Core::LinAlg::Matrix<1, numdof_>&
              dvolchange_dus,  //!<  (o) Linearization of Change of Volume  w.r.t displacements
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,  //!< (i) deformation gradient
          const Core::LinAlg::Matrix<numdim_, numdim_>&
              defgrd_inv,  //!< (i) inverse of deformation gradient
          const Core::LinAlg::Matrix<numdim_, numnod_>&
              N_XYZ,  //!< (i) spatial gradient of material shape functions
          const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp  //!<  (i) nodal displacements
      );

      //! Compute Linearization Of Jacobian
      void compute_linearization_of_jacobian(Core::LinAlg::Matrix<1, numdof_>& dJ_dus,
          const double& J, const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv);

      //! Compute Linearization some Auxiliary Values in gauss point loop
      void compute_auxiliary_values(const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
          const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
          const Core::LinAlg::Matrix<numdim_, 1>& Gradp,
          Core::LinAlg::Matrix<numdim_ * numdim_, numdof_>& dFinvTdus,
          Core::LinAlg::Matrix<numdim_, 1>& Finvgradp,
          Core::LinAlg::Matrix<numdim_, numdof_>& dFinvdus_gradp,
          Core::LinAlg::Matrix<numstr_, numdof_>& dCinv_dus);

      //! Compute  nonlinear b-operator
      void compute_b_operator(Core::LinAlg::Matrix<numstr_, numdof_>& bop,
          const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
          const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ);

      //! Compute deformation gradient
      void compute_def_gradient(Core::LinAlg::Matrix<numdim_, numdim_>&
                                    defgrd,  //!<<    (i) deformation gradient at gausspoint
          const Core::LinAlg::Matrix<numdim_, numnod_>&
              N_XYZ,  //!<<    (i) derivatives of shape functions w.r.t. reference coordinates
          const Core::LinAlg::Matrix<numdim_, numnod_>&
              xcurr  //!<<    (i) current position of gausspoint
      );

      //! read anisotropic permeability directions in the element definition
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

      //! flag indicating if element has been initialized
      bool init_;

      //! flag for nurbs
      bool isNurbs_;

      //! weights for nurbs elements
      Core::LinAlg::Matrix<numnod_, 1> weights_;

      //! knot vector for nurbs elements
      std::vector<Core::LinAlg::SerialDenseVector> myknots_;

      //! corresponding fluid material
      std::shared_ptr<Mat::FluidPoro> fluid_mat_;

      //! corresponding multiphase fluid material
      std::shared_ptr<Mat::FluidPoroMultiPhase> fluidmulti_mat_;

      //! own poro structure material
      std::shared_ptr<Mat::StructPoro> struct_mat_;

      //! directions for anisotropic permeability
      std::vector<std::vector<double>> anisotropic_permeability_directions_;

      //! scaling coefficients for nodal anisotropic permeability
      std::vector<std::vector<double>> anisotropic_permeability_nodal_coeffs_;

      //! get nodes of element
      Core::Nodes::Node** nodes() override;

      //! get material of element
      std::shared_ptr<Core::Mat::Material> material() const;

      //! get global id of element
      int id() const;
    };
  }  // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif

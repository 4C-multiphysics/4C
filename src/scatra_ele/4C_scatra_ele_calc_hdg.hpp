// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_HDG_HPP
#define FOUR_C_SCATRA_ELE_CALC_HDG_HPP


#include "4C_config.hpp"

#include "4C_fem_general_utils_shapevalues_hdg.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_scatra_ele_calc.hpp"
#include "4C_scatra_ele_hdg.hpp"
#include "4C_scatra_ele_interface.hpp"

FOUR_C_NAMESPACE_OPEN



namespace Discret
{
  namespace Elements
  {
    //! Scatra HDG element implementation
    template <Core::FE::CellType distype, int probdim = Core::FE::dim<distype>>
    class ScaTraEleCalcHDG : public ScaTraEleInterface
    {
     public:
      //! nen_: number of element nodes (T. Hughes: The Finite Element Method)
      static constexpr unsigned int nen_ = Core::FE::num_nodes<distype>;

      //! number of space dimensions
      static constexpr unsigned int nsd_ = probdim;

      //! number of faces on element
      static constexpr unsigned int nfaces_ = Core::FE::num_faces<distype>;

      /// Evaluate supporting methods of the element
      /*!
        Interface function for supporting methods of the element
       */

      //! Singleton access method
      static ScaTraEleCalcHDG<distype, probdim>* instance(const int numdofpernode,
          const int numscal, const std::string& disname, bool create = true);

      //! evaluate service routine
      int evaluate_service(Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
          Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
          Core::LinAlg::SerialDenseVector& elevec1_epetra,
          Core::LinAlg::SerialDenseVector& elevec2_epetra,
          Core::LinAlg::SerialDenseVector& elevec3_epetra) override;

      //! interpolates an HDG solution to the element nodes for output
      virtual int node_based_values(Core::Elements::Element* ele,
          Core::FE::Discretization& discretization, Core::LinAlg::SerialDenseVector& elevec1);

      //! initialize the shape functions and solver to the given element (degree is runtime
      //! parameter)
      void initialize_shapes(const Core::Elements::Element* ele, const std::string& disname);

      //! Evaluate the element (Generic virtual interface function. Called via base pointer.)
      int evaluate(Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
          Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

      //! evaluate action for off-diagonal system matrix block
      int evaluate_od(Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
          Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override
      {
        FOUR_C_THROW("Not implemented!");
        return -1;
      }

      //! Setup element evaluation
      int setup_calc(
          Core::Elements::Element* ele, Core::FE::Discretization& discretization) override
      {
        return 0;
      }

      //! projection of Dirichlet function field
      int project_dirich_field(Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Core::LinAlg::SerialDenseVector& elevec1);

      //! update interior variables
      int update_interior_variables(Discret::Elements::ScaTraHDG* ele,
          Teuchos::ParameterList& params, Core::LinAlg::SerialDenseVector& elevec);

      //! set initial field
      int set_initial_field(const Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2);

      //! project field
      int project_field(const Core::Elements::Element* ele,
          Core::FE::Discretization& discretization, Teuchos::ParameterList& params,
          Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
          Core::Elements::LocationArray& la);

      //! project material field
      virtual int project_material_field(const Core::Elements::Element* ele) { return 0; };

      //! calc p-adaptivity
      int calc_p_adaptivity(const Core::Elements::Element* ele,
          Core::FE::Discretization& discretization, Teuchos::ParameterList& params);

      //! calc error
      int calc_error(const Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::LinAlg::SerialDenseVector& elevec);

     protected:
      /// (private) protected constructor, since we are a Singleton.
      /// this constructor is called from a derived class
      /// -> therefore, it has to be protected instead of private
      ScaTraEleCalcHDG(const int numdofpernode, const int numscal, const std::string& disname);

      //! get the material parameters
      virtual void get_material_params(
          Core::Elements::Element* ele  //!< the element we are dealing with
      );

      //! get the material parameters before first timestep
      virtual void prepare_material_params(
          Core::Elements::Element* ele  //!< the element we are dealing with
      );

      //! evaluate material
      virtual void materials(const std::shared_ptr<const Core::Mat::Material>
                                 material,              //!< pointer to current material
          const int k,                                  //!< id of current scalar
          Core::LinAlg::SerialDenseMatrix& difftensor,  //!< diffusion tensor
          Core::LinAlg::SerialDenseVector& ivecn,       //!< reaction term at time n
          Core::LinAlg::SerialDenseVector& ivecnp,      //!< reaction term at time n+1
          Core::LinAlg::SerialDenseMatrix& ivecnpderiv  //!< reaction term derivative
      )
      {
        return;
      };

      //! evaluate material before first timestep
      virtual void prepare_materials(
          Core::Elements::Element* ele,  //!< the element we are dealing with
          const std::shared_ptr<const Core::Mat::Material>
              material,  //!< pointer to current material
          const int k,   //!< id of current scalar
          std::shared_ptr<std::vector<Core::LinAlg::SerialDenseMatrix>>
              difftensor  //!< diffusion tensor
      );

      //! stores the material internal state in a vector for output and restart
      virtual void get_material_internal_state(const Core::Elements::Element* ele,
          Teuchos::ParameterList& params, Core::FE::Discretization& discretization)
      {
        return;
      };

      //! stores the restart information in the material internal state
      virtual void set_material_internal_state(const Core::Elements::Element* ele,
          Teuchos::ParameterList& params, Core::FE::Discretization& discretization)
      {
        return;
      };

      //! local data object
      std::shared_ptr<Core::FE::ShapeValues<distype>> shapes_;
      std::shared_ptr<Core::FE::ShapeValuesFace<distype>> shapesface_;

      //! extracted values from concentrations and gradients
      Core::LinAlg::SerialDenseVector interiorPhin_;
      //! extracted values from concentrations
      Core::LinAlg::SerialDenseVector interiorPhinp_;

      //! get time step
      double dt() { return local_solver_->scatraparatimint_->dt(); }



      //! update time dependent material
      virtual void time_update_material(
          const Core::Elements::Element* ele  //!< the element we are dealing with
      )
      {
        return;
      };

      //! element initialization at the first time step
      void element_init(Core::Elements::Element* ele);

      /*========================================================================*/
      //! @name dofs and nodes
      /*========================================================================*/

      //! number of dof per node
      const int numdofpernode_;

      //! number of transported scalars (numscal_ <= numdofpernode_)
      const int numscal_;

      //! use complete polynomial space
      bool usescompletepoly_;

      //! pointer to general scalar transport parameter class
      Discret::Elements::ScaTraEleParameterStd* scatrapara_;

     private:
      //! local solver that inverts local problem on an element and can solve with various vectors
      struct LocalSolver
      {
        static constexpr unsigned int nsd_ = ScaTraEleCalcHDG<distype, probdim>::nsd_;
        static constexpr unsigned int nfaces_ = ScaTraEleCalcHDG<distype, probdim>::nfaces_;
        int onfdofs_;

        LocalSolver(const Core::Elements::Element* ele, Core::FE::ShapeValues<distype>& shapeValues,
            Core::FE::ShapeValuesFace<distype>& shapeValuesFace, bool completepoly,
            const std::string& disname, int numscal);

        //! compute the residual
        void compute_residual(Teuchos::ParameterList& params,
            Core::LinAlg::SerialDenseVector& elevec, Core::LinAlg::SerialDenseMatrix& elemat1,
            Core::LinAlg::SerialDenseVector& interiorPhin, Core::LinAlg::SerialDenseVector& tracen,
            Core::LinAlg::SerialDenseVector& tracenp, const Discret::Elements::ScaTraHDG* hdgele);

        //! compute Neumann boundary conditions
        void compute_neumann_bc(Core::Elements::Element* ele, Teuchos::ParameterList& params,
            int face, Core::LinAlg::SerialDenseVector& elevec, int indexstart);

        //! compute interior matrices
        void compute_interior_matrices(Discret::Elements::ScaTraHDG* hdgele);

        //! compute interior matrices for Tet elements
        void compute_interior_matrices_tet(Discret::Elements::ScaTraHDG* hdgele);

        //! compute interior matrices
        void compute_interior_matrices_all(Discret::Elements::ScaTraHDG* hdgele);

        //! calls local solver to compute matrices: internal and face
        void compute_matrices(Core::Elements::Element* ele);

        //! compute face matrices
        void compute_face_matrices(
            const int face, int indexstart, Discret::Elements::ScaTraHDG* hdgele);

        //! condense the local matrix (involving interior concentration gradients and
        //! concentrations) into the element matrix for the trace and similarly for the residuals
        void condense_local_part(Discret::Elements::ScaTraHDG* hdgele);

        //! Compute divergence of current source (ELEMAG)
        void compute_source(const Core::Elements::Element* ele,
            Core::LinAlg::SerialDenseVector& elevec1, const double time);

        //! add diffusive term to element matrix
        void add_diff_mat(
            Core::LinAlg::SerialDenseMatrix& elemat, const Discret::Elements::ScaTraHDG* hdgele);

        //! add reaction term to element matrix
        void add_reac_mat(
            Core::LinAlg::SerialDenseMatrix& elemat, const Discret::Elements::ScaTraHDG* hdgele);

        //! set material parameter
        void set_material_parameter(Discret::Elements::ScaTraHDG* hdgele,
            Core::LinAlg::SerialDenseVector& ivecn, Core::LinAlg::SerialDenseVector& ivecnp,
            Core::LinAlg::SerialDenseMatrix& ivecnpderiv);

        //! prepare material parameter in first timestep
        void prepare_material_parameter(
            Discret::Elements::ScaTraHDG* hdgele, Core::LinAlg::SerialDenseMatrix& difftensor);


        // convention: we sort the entries in the matrices the following way:
        // first come the concentration, then the concentration  gradients, and finally the trace

        //! evaluated shape values
        std::shared_ptr<Core::FE::ShapeValues<distype>> shapes_;

        //! evaluated shape values on face
        std::shared_ptr<Core::FE::ShapeValuesFace<distype>> shapesface_;  /// evaluated shape values

        // Element matrices if one wants to compute them on the fly instead of storing them on the
        // element
        //      Core::LinAlg::SerialDenseMatrix  Amat;     /// concentrations - concentrations
        //      Core::LinAlg::SerialDenseMatrix  Bmat;     /// concentrations - concentrations
        //      gradients Core::LinAlg::SerialDenseMatrix  Cmat;     /// concentration - trace
        //      Core::LinAlg::SerialDenseMatrix  Dmat;     /// concentrations gradients -
        //      concentrations gradients Core::LinAlg::SerialDenseMatrix  Emat;     /// trace -
        //      concentrations gradients Core::LinAlg::SerialDenseMatrix  Gmat;     ///
        //      concentrations gradients Core::LinAlg::SerialDenseMatrix  Hmat;     /// trace -trace
        //      Core::LinAlg::SerialDenseMatrix  Mmat;     /// mass matrix (concentrations -
        //      concentrations) Core::LinAlg::SerialDenseMatrix  EmatT;    /// trace -
        //      concentrations gradients (E^T) Core::LinAlg::SerialDenseMatrix  BmatMT;   ///
        //      concentrations gradients- concentrations (-B^T) Core::LinAlg::SerialDenseMatrix
        //      Kmat;   /// condensed matrix

        // @name variables for the reaction term
        //!@{
        //! reaction term at time n
        //      Core::LinAlg::SerialDenseVector  Ivecn_;

        //! reaction term at time n+1
        // Core::LinAlg::SerialDenseVector  Ivecnp_;

        //! derivative of reaction term at time n+1
        //      Core::LinAlg::SerialDenseMatrix  Imatnpderiv_;
        //!@}

        ////      Core::LinAlg::SerialDenseMatrix  invAmat;     /// inverse of Amat
        //      Core::LinAlg::SerialDenseMatrix  invAMmat;     /// inverse of [A + (1/(dt*theta))*M]
        //
        //      // auxiliary stuff
        //      Core::LinAlg::SerialDenseMatrix  massPart;
        //      Core::LinAlg::SerialDenseMatrix  massPartW;
        //      Core::LinAlg::SerialDenseMatrix  BTAMmat;
        //      Core::LinAlg::SerialDenseMatrix  invCondmat;
        //      Core::LinAlg::SerialDenseMatrix  Xmat;

        //! pointer to general scalar transport parameter class
        Discret::Elements::ScaTraEleParameterStd* scatrapara_;

        //      std::shared_ptr<Discret::Elements::ScaTraEleParameterBase> scatrapara_; //! pointer
        //      to parameter list
        //! pointer to time parameter list
        std::shared_ptr<Discret::Elements::ScaTraEleParameterTimInt> scatraparatimint_;

        /*========================================================================*/
        //! @name diffusions and reaction coefficient
        /*========================================================================*/

        // diffusion tensor stored on the element, if necessary this can be changed
        //      //! diffusion tensor
        //      Core::LinAlg::SerialDenseMatrix diff_;
        //      //! inverse diffusion tensor
        //      Core::LinAlg::SerialDenseMatrix invdiff_;

        //! scalar raeaction coefficient
        //      std::vector<double> reacoeff_;
      };

      //! reads from global vectors
      void read_global_vectors(Core::Elements::Element* ele,
          Core::FE::Discretization& discretization, Core::Elements::LocationArray& la);

      //! local solver object
      std::shared_ptr<LocalSolver> local_solver_;

      /*========================================================================*/
      //! @name trace and interior concentrations and gradients
      /*========================================================================*/

      //! extracted values from concentrations
      Core::LinAlg::SerialDenseVector tracen_;

      //! extracted local values (concentration gradients) at n+alpha_f
      Core::LinAlg::SerialDenseVector interior_grad_phin_;

      //! extracted values from trace solution vector at n-m
      Core::LinAlg::SerialDenseVector tracenm_;
    };
  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif

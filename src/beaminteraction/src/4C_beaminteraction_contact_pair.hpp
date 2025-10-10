// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CONTACT_PAIR_HPP
#define FOUR_C_BEAMINTERACTION_CONTACT_PAIR_HPP

#include "4C_config.hpp"

#include "4C_linalg_fevector.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SerialDenseVector;
  class SerialDenseMatrix;
  class SparseMatrix;
}  // namespace Core::LinAlg
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace GeometryPair
{
  class GeometryPairBase;
  class GeometryEvaluationDataBase;
  class FaceElement;
}  // namespace GeometryPair


namespace BeamInteraction
{
  // forward declaration ...
  class BeamContactParams;
  class BeamToSolidVisualizationOutputWriterBase;
  class BeamInteractionConditions;
  class BeamToSolidMortarManager;

  enum class ContactPairType
  {
    unspecified_base,
    beam_to_beam_contact,
    beam_to_beam_point_coupling,
    beam_to_solid_base,
    beam_to_solid_edge,
    beam_to_sphere_contact,
  };

  /*!
   \brief
   */
  class BeamContactPair
  {
   public:
    //! @name Friends
    // no friend classes defined
    //@}

    //! @name Constructors and destructors and related methods

    BeamContactPair();

    /*!
    \brief Destructor
    */
    virtual ~BeamContactPair() = default;
    //! Initialization
    virtual void init(const std::shared_ptr<BeamInteraction::BeamContactParams> params_ptr,
        std::vector<Core::Elements::Element const*> elements);

    //! Setup
    virtual void setup();

    //@}

    /*!
    \brief things that need to be done in a separate loop before the actual evaluation loop
           over all contact pairs
    */
    virtual void pre_evaluate() = 0;

    //! @name Public evaluation methods
    /*!
    \brief Evaluate this contact element pair, return value indicates whether pair is active,
           i.e. non-zero values for force and stiffmat are returned
    */
    virtual bool evaluate(Core::LinAlg::SerialDenseVector* forcevec1,
        Core::LinAlg::SerialDenseVector* forcevec2, Core::LinAlg::SerialDenseMatrix* stiffmat11,
        Core::LinAlg::SerialDenseMatrix* stiffmat12, Core::LinAlg::SerialDenseMatrix* stiffmat21,
        Core::LinAlg::SerialDenseMatrix* stiffmat22) = 0;

    //! return appropriate internal implementation class (acts as a simple factory), also works if a
    //! single pair of elements results in multiple beam contact pairs
    static std::vector<std::shared_ptr<BeamContactPair>> create(
        std::vector<Core::Elements::Element const*> const& ele_ptrs,
        BeamInteraction::BeamInteractionConditions& beam_interaction_conditions_ptr);

    /*
    \brief Update state of translational nodal DoFs (absolute positions and tangents) of both
    elements
    */
    virtual void reset_state(const std::vector<double>& centerline_dofvec_ele1,
        const std::vector<double>& centerline_dofvec_ele2) = 0;

    /**
     * \brief Update state of rotational DoFs of both elements
     */
    virtual void reset_rotation_state(const Core::FE::Discretization& discret,
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& ia_discolnp) {};

    //@}

    //! @name Access methods

    inline std::shared_ptr<BeamInteraction::BeamContactParams> params() const { return params_; }

    /*!
    \brief Get an element pointer by the elements index.
    */
    inline const Core::Elements::Element* get_element(const unsigned int index) const
    {
      if (index == 0)
        return element1_;
      else if (index == 1)
        return element2_;
      else
        FOUR_C_THROW("Index has to be 0 or 1, got {}", index);
      return nullptr;
    };

    /*!
    \brief Get first element
    */
    inline const Core::Elements::Element* element1() const { return element1_; };

    /*!
    \brief Get second element
    */
    inline const Core::Elements::Element* element2() const { return element2_; };

    /*!
    \brief Get the geometry pair object. Throw error if it does not exist.
    */
    inline std::shared_ptr<GeometryPair::GeometryPairBase> geometry_pair() const
    {
      if (geometry_pair_ == nullptr)
        FOUR_C_THROW("The geometry pair is requested, but it is a null pointer!");
      return geometry_pair_;
    }

    /*!
    \brief Get flag indicating whether contact is active (true) or inactive (false)
    */
    virtual bool get_contact_flag() const = 0;

    /*!
    \brief Get number of active contact point pairs on this element pair
    */
    virtual unsigned int get_num_all_active_contact_point_pairs() const = 0;

    /*!
    \brief Get coordinates of all active contact points on element1 and element2
    */
    virtual void get_all_active_contact_point_coords_element1(
        std::vector<Core::LinAlg::Matrix<3, 1, double>>& coords) const = 0;

    virtual void get_all_active_contact_point_coords_element2(
        std::vector<Core::LinAlg::Matrix<3, 1, double>>& coords) const = 0;

    /*!
    \brief Get all active visualization values of this contact pair
    */
    virtual void get_all_active_beam_to_beam_visualization_values(std::vector<double>& forces,
        std::vector<double>& gaps, std::vector<double>& angles, std::vector<int>& types) const = 0;


    //  virtual Core::LinAlg::Matrix< 3, 1,TYPE>* GetNormalOld()=0;
    //
    //  /*!
    //    \Check, if there is a difference between the result of the new and old gap definition,
    //    i.e. if the beams centerlines have already crossed or not.
    //  */
    //  virtual bool GetNewGapStatus()=0;

    /*!
    \brief Get energy of penalty contact.
    */
    virtual double get_energy() const = 0;

    //  /*!
    //    \Get energy of perp penalty contact without transition factor contribution.
    //  */
    //  virtual double get_unscaled_perp_energy()=0;
    //
    //  /*!
    //    \Get energy of parallel penalty contact without transition factor contribution.
    //  */
    //  virtual double get_unscaled_parallel_energy()=0;
    //
    //  virtual bool FirstTimeStep()=0;
    //
    //  /*!
    //  \brief Get flag indicating whether the nodal values of one element had been shifted due to
    //  r1=r2
    //  */
    //  virtual bool GetShiftStatus()=0;
    //  //@}

    /** \brief print this beam contact element pair to screen
     *
     */
    virtual void print(std::ostream& out) const = 0;

    /** \brief print this beam contact element pair to screen
     *
     */
    virtual void print_summary_one_line_per_active_segment_pair(std::ostream& out) const = 0;

    /**
     * \brief Per default it is assumed, that the contributions of a pair can be directly assembled
     * into the global force and stiffness matrices.
     */
    inline virtual bool is_assembly_direct() const { return true; };

    /**
     * \brief Evaluate the pair and directly assemble it into the global force vector and stiffness
     * matrix.
     *
     * This method can be used for pairs that couple more DOF than the ones from the two contact
     * elements, e.g. surface patches with averaged normals also create coupling terms in the DOF of
     * the neighbouring surface elements. Those pairs write their coupling terms into the global
     * system vector / matrix and do not require an outside routine to do so.
     *
     * In some cases this can also be used to avoid the creation of variable size element vectors
     * and matrices, by directly assembling templated vectors and matrices into the global ones.
     *
     * @param discret (in) Pointer to the discretization.
     * @param force_vector (in / out) Global force vector.
     * @param stiffness_matrix (in / out) Global stiffness matrix.
     * @param displacement_vector (in) Global displacement vector.
     */
    virtual void evaluate_and_assemble(
        const std::shared_ptr<const Core::FE::Discretization>& discret,
        const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
        const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector) {};

    /**
     * \brief Evaluate the pair and directly assemble it into the global force vector and stiffness
     * matrix.
     *
     * This method can be used for pairs that couple more DOF than the ones from the two contact
     * elements, e.g. surface patches with averaged normals also create coupling terms in the DOF of
     * the neighbouring surface elements. Those pairs write their coupling terms into the global
     * system vector / matrix and do not require an outside routine to do so.
     *
     * In some cases this can also be used to avoid the creation of variable size element vectors
     * and matrices, by directly assembling templated vectors and matrices into the global ones.
     *
     * @param discret (in) Pointer to the discretization.
     * @param mortar_manager (in) Mortar manager, used to get the Lagrange multiplier GIDs.
     * @param force_vector (in / out) Global force vector.
     * @param stiffness_matrix (in / out) Global stiffness matrix.
     * @param lambda (in) Global Lagrange multiplier vector.
     * @param displacement_vector (in) Global displacement vector.
     */
    virtual void evaluate_and_assemble(const Core::FE::Discretization& discret,
        const BeamToSolidMortarManager* mortar_manager,
        const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
        const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const Core::LinAlg::Vector<double>& global_lambda,
        const Core::LinAlg::Vector<double>& displacement_vector) {};

    /**
     * \brief Evaluate the mortar matrices $D$ and $M$ for this contact element pair.
     * @param local_D (out) Local mortar matrix $D$.
     * @param local_M (out) Local mortar matrix $M$.
     * @param local_kappa (out) Local scaling vector.
     * @param local_constraint (out) Local constraint vector.
     * @return True if pair is in contact.
     */
    virtual bool evaluate_dm(Core::LinAlg::SerialDenseMatrix& local_D,
        Core::LinAlg::SerialDenseMatrix& local_M, Core::LinAlg::SerialDenseVector& local_kappa,
        Core::LinAlg::SerialDenseVector& local_constraint)
    {
      return false;
    }

    /**
     * \brief Evaluate the global matrices and vectors resulting from mortar coupling.
     * @param discret (in) discretization, used to get the beam GIDs.
     * @param mortar_manager (in) Mortar manager, used to get the Lagrange multiplier GIDs.
     * @param global_constraint_lin_beam (in/out) Constraint equations derived w.r.t the beam DOFs.
     * @param global_constraint_lin_solid (in/out) Constraint equations derived w.r.t the solid
     * DOFs.
     * @param global_force_beam_lin_lambda (in/out) Beam force vector derived w.r.t the Lagrange
     * multipliers.
     * @param global_force_solid_lin_lambda (in/out) Solid force vector derived w.r.t the Lagrange
     * multipliers.
     * @param global_constraint (in/out) Global constraint vector.
     * @param global_kappa (in/out) Global scaling matrix.
     * @param global_kappa_lin_beam (in/out) Global scaling matrix derived w.r.t. the beam DOFs.
     * @param global_kappa_lin_solid (in/out) Global scaling matrix derived w.r.t. the solid DOFs.
     * @param global_lambda_active (in/out) Global vector with active Lagrange multipliers.
     * @param displacement_vector (in) Global displacement vector.
     */
    virtual void evaluate_and_assemble_mortar_contributions(const Core::FE::Discretization& discret,
        const BeamToSolidMortarManager* mortar_manager,
        Core::LinAlg::SparseMatrix& global_constraint_lin_beam,
        Core::LinAlg::SparseMatrix& global_constraint_lin_solid,
        Core::LinAlg::SparseMatrix& global_force_beam_lin_lambda,
        Core::LinAlg::SparseMatrix& global_force_solid_lin_lambda,
        Core::LinAlg::FEVector<double>& global_constraint,
        Core::LinAlg::FEVector<double>& global_kappa,
        Core::LinAlg::SparseMatrix& global_kappa_lin_beam,
        Core::LinAlg::SparseMatrix& global_kappa_lin_solid,
        Core::LinAlg::FEVector<double>& global_lambda_active,
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector) {};

    /**
     * \brief Add the visualization of this pair to the beam to solid visualization output writer.
     *
     * This is currently only implemented for beam to solid pairs, if in the future we also want to
     * use this function in other type of pairs, the name should be adapted accordingly.
     *
     * @param visualization_writer (out) Object that manages all visualization related data for beam
     * to solid pairs.
     * @param visualization_params (in) Parameter list with possible needed data to generate the
     * visualization in the pairs.
     */
    virtual void get_pair_visualization(
        std::shared_ptr<BeamToSolidVisualizationOutputWriterBase> visualization_writer,
        Teuchos::ParameterList& visualization_params) const
    {
    }

    /**
     * \brief Create the geometry pair for this contact pair.
     *
     * Per default no geometry pair is created. The geometry pair has to be created before init is
     * called on the contact pair.
     *
     * @param element1 Pointer to the first element
     * @param element2 Pointer to the second element
     *
     * @param geometry_evaluation_data_ptr (in) Geometry evaluation data for the geometry pair.
     */
    virtual void create_geometry_pair(const Core::Elements::Element* element1,
        const Core::Elements::Element* element2,
        const std::shared_ptr<GeometryPair::GeometryEvaluationDataBase>&
            geometry_evaluation_data_ptr)
    {
      FOUR_C_THROW("CreateGeometryPair has to be implemented in the derived class.");
    };

    /**
     * \brief Set the restart displacement in this pair.
     *
     * If coupling interactions should be evaluated w.r.t the restart state. Has to be implemented
     * in derived class.
     *
     * @param centerline_restart_vec_ (in) Vector with the centerline displacements at the restart
     * step, for all contained elements (Vector of vector).
     */
    virtual void set_restart_displacement(
        const std::vector<std::vector<double>>& centerline_restart_vec_)
    {
      check_init_setup();
    };

    /**
     * \brief Link the contact pair with the face element storing information on the averaged nodal
     * normals.
     *
     * @param face_element (in) RCP to the face element.
     */
    virtual void set_face_element(std::shared_ptr<GeometryPair::FaceElement>& face_element)
    {
      FOUR_C_THROW("This method has to be implemented in the derived class.");
    }

    /**
     * \brief Returns the type of this beam contact pair.
     */
    virtual ContactPairType get_type() const { return ContactPairType::unspecified_base; }


   protected:
    //! returns init state
    inline const bool& is_init() const { return isinit_; };

    //! returns setup state
    inline const bool& is_setup() const { return issetup_; };

    //! Check the init state
    void check_init() const;

    //! Check the init and setup state
    void check_init_setup() const;

   protected:
    //! @name member variables

    //! indicates if the init() function has been called
    bool isinit_;

    //! indicates if the setup() function has been called
    bool issetup_;

    //! pointer to the geometry pair
    std::shared_ptr<GeometryPair::GeometryPairBase> geometry_pair_;

   private:
    //! beam contact parameter data container
    std::shared_ptr<BeamInteraction::BeamContactParams> params_;

    //! first element of interacting pair
    const Core::Elements::Element* element1_;

    //! second element of interacting pair
    const Core::Elements::Element* element2_;
    //@}
  };
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif

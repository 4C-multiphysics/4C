// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRY_PAIR_ELEMENT_FACES_HPP
#define FOUR_C_GEOMETRY_PAIR_ELEMENT_FACES_HPP


#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_geometry_pair_element.hpp"

#include <unordered_map>

namespace
{
  class GeometryPairLineToSurfacePatchTest;
}

FOUR_C_NAMESPACE_OPEN

// Forward declarations.
namespace GeometryPair
{
  struct ConnectedFace;
}  // namespace GeometryPair

namespace GeometryPair
{
  enum class SurfaceNormals;
}



namespace GeometryPair
{
  /**
   * \brief This structure "converts" a face element type to the underlying volume element.
   */
  template <Core::FE::CellType discretization>
  struct FaceDiscretizationTypeToVolumeElement
  {
    using volume_type_ = void;
  };
  template <>
  struct FaceDiscretizationTypeToVolumeElement<Core::FE::CellType::quad4>
  {
    using volume_type_ = t_hex8;
  };
  template <>
  struct FaceDiscretizationTypeToVolumeElement<Core::FE::CellType::quad8>
  {
    using volume_type_ = t_hex20;
  };
  template <>
  struct FaceDiscretizationTypeToVolumeElement<Core::FE::CellType::quad9>
  {
    using volume_type_ = t_hex27;
  };

  /**
   * \brief Utility structure to represent the connection of faces to a patch.
   */
  struct ConnectedFace
  {
    //! This vector stores the LIDs of the nodes of this connected face corresponding to the nodes
    //! of the patch.
    std::vector<int> my_node_patch_lid_;

    //! This map represents the link between the nodes of a connected face and the patch face. The
    //! key is the LID of the node on the connected face and the value is the LID of the node on the
    //! patch face.
    std::map<int, int> node_lid_map_;
  };

  /**
   * \brief Base, non templated class for an object that represents a surface element.
   */
  class FaceElement
  {
    friend GeometryPairLineToSurfacePatchTest;

   public:
    /**
     * \brief Constructor.
     * @param core_element (in) Pointer to the element.
     */
    FaceElement(const Core::Elements::Element* core_element)
        : core_element_(core_element), part_of_pair_(false), patch_dof_gid_() {};

    /**
     * \brief Destructor.
     */
    virtual ~FaceElement() = default;

    /**
     * \brief Get the pointer to the element.
     */
    const Core::Elements::Element* get_element() const { return core_element_; }

    /**
     * \brief Setup the object. Has to be implemented in derived class.
     * @param discret (in) Pointer to the discretization.
     * @param face_elements (in) Vector with all face elements in the surface condition.
     */
    virtual void setup(const std::shared_ptr<const Core::FE::Discretization>& discret,
        const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>&
            face_elements) = 0;

    /**
     * \brief Set the needed displacement vectors for this face. Has to be implemented in derived
     * class.
     *
     * @param displacement (in) Current displacement vector.
     * @param face_elements (in) Map with all the faces in this condition.
     */
    virtual void set_state(const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement,
        const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>&
            face_elements) = 0;

    /**
     * \brief Calculate the averaged normals on the nodes of this face. Has to be implemented in
     * derived class.
     *
     * @param face_elements (in) Vector with all face elements in the surface condition.
     */
    virtual void calculate_averaged_reference_normals(
        const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>& face_elements) {
    };

    /**
     * \brief Return the position on this face as a double.
     *
     * @param xi (in) Parameter coordinate on the face.
     * @param r (out) Position on the face.
     * @param reference (in) If the reference position or the current position should be returned.
     */
    virtual void evaluate_face_position_double(const Core::LinAlg::Matrix<2, 1, double>& xi,
        Core::LinAlg::Matrix<3, 1, double>& r, bool reference = false) const = 0;

    /**
     * \brief Return a normal on the element.
     *
     * If an averaged normal is requested and the element does not have averaged normals (e.g.
     * nurbs) a zero vector is returned.
     *
     * @param xi (in) Parameter coordinate on the face.
     * @param n (out) Normal on the face.
     * @param reference (in) If the reference position or the current position should be returned.
     * @param averaged_normal (in) If an averaged normal or an geometrical normal should be
     * returned.
     */
    virtual void evaluate_face_normal_double(const Core::LinAlg::Matrix<2, 1, double>& xi,
        Core::LinAlg::Matrix<3, 1, double>& n, const bool reference,
        const bool averaged_normal) const = 0;

    /**
     * \brief Return if the face is part of a pair.
     * @return True if the face is part of a pair, false otherwise.
     */
    bool is_part_of_pair() const { return part_of_pair_; }

    /**
     * \brief Set the part_of_pair_ flag.
     * @param part_of_pair (in) Value to set.
     */
    void set_part_of_pair(bool part_of_pair) { part_of_pair_ = part_of_pair; }

    /**
     * \brief Get the local to global indices for this surface patch.
     * @return Local to global indices for this surface patch.
     */
    const std::vector<int>& get_patch_gid() const { return patch_dof_gid_; }

   protected:
    //! Pointer to the drt element.
    const Core::Elements::Element* core_element_;

    //! Flag if this face element is part of a contact pair, i.e. if it has evaluate it's averaged
    //! normals.
    bool part_of_pair_;

    //! Global DOF IDs of this face patch.
    std::vector<int> patch_dof_gid_;
  };


  /**
   * \brief An object that represents a surface element which only depends on a single face. For
   * example nurbs faces since they are C1 continuous.
   *
   * @tparam surface Type of surface element.
   * @tparam scalar_type Scalar type for FAD evaluations.
   */
  template <typename Surface, typename ScalarType>
  class FaceElementTemplate : public FaceElement
  {
   public:
    //! Shortcut to the type of this templated object.
    using my_type = FaceElementTemplate<Surface, ScalarType>;

   public:
    /**
     * \brief Constructor (derived).
     */
    FaceElementTemplate(const Core::Elements::Element* core_element)
        : FaceElement(core_element), n_dof_other_element_(0) {};


    /**
     * \brief Get the face GIDs and set the reference configuration.
     *
     * @param discret (in) Pointer to the discretization.
     * @param face_elements (in) Vector with all face elements in the surface condition.
     */
    void setup(const std::shared_ptr<const Core::FE::Discretization>& discret,
        const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>& face_elements)
        override;

    /**
     * \brief Set the needed displacement vectors for this face (derived).
     */
    void set_state(const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement,
        const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>& face_elements)
        override;

    /**
     * \brief Return the reference element data of this face.
     */
    const auto& get_face_reference_element_data() const { return face_reference_position_; }

    /**
     * \brief Return the current element data of this face.
     */
    const auto& get_face_element_data() const { return face_position_; }

    /**
     * \brief Return the reference normals on this face.
     */
    virtual const Core::LinAlg::Matrix<3 * Surface::n_nodes_, 1, double>* get_reference_normals()
        const
    {
      return nullptr;
    }

    /**
     * \brief Return the current normals on this face.
     */
    virtual const Core::LinAlg::Matrix<3 * Surface::n_nodes_, 1, ScalarType>* get_current_normals()
        const
    {
      return nullptr;
    }

    /**
     * \brief Return the position on this face as a double. (derived)
     */
    void evaluate_face_position_double(const Core::LinAlg::Matrix<2, 1, double>& xi,
        Core::LinAlg::Matrix<3, 1, double>& r, bool reference = false) const override;

    /**
     * \brief Return a normal on the element. (derived)
     */
    void evaluate_face_normal_double(const Core::LinAlg::Matrix<2, 1, double>& xi,
        Core::LinAlg::Matrix<3, 1, double>& n, const bool reference,
        const bool averaged_normal) const override;

    /**
     * \brief Return the number of DOFs of the interacting element.
     */
    [[nodiscard]] unsigned int get_number_of_dof_other_element() const
    {
      return n_dof_other_element_;
    }

    /**
     * \brief Return the number of DOFs of the interacting element.
     */
    void set_number_of_dof_other_element(const unsigned int n_dof_other_element)
    {
      if (n_dof_other_element_ == 0)
      {
        n_dof_other_element_ = n_dof_other_element;
      }
      else if (n_dof_other_element_ != n_dof_other_element)
      {
        FOUR_C_THROW(
            "FaceElementTemplate only allows other elements with the same number of DOFs. You "
            "already have an other element with {} DOFs and now want to set another one with {} "
            "DOFs",
            n_dof_other_element_, n_dof_other_element);
      }
    }

   protected:
    //! Reference position.
    ElementData<Surface, double> face_reference_position_;

    //! Current position.
    ElementData<Surface, ScalarType> face_position_;

    //! Number of DOFs used for the element that will be interacting with this face. This is
    //! required to correctly set the FAD types. For now, a single face element can only be used in
    //! interactions with other elements of the same type.
    unsigned int n_dof_other_element_;
  };


  /**
   * \brief An object that represents a surface element and stores averaged normal information as
   * well as position information of the face.
   *
   * @tparam surface Type of surface element.
   * @tparam scalar_type Scalar type for FAD evaluations.
   */
  template <typename Surface, typename ScalarType>
  class FaceElementPatchTemplate : public FaceElementTemplate<Surface, ScalarType>
  {
    friend GeometryPairLineToSurfacePatchTest;

   public:
    //! Shortcut to the type of this templated object.
    using my_type = FaceElementPatchTemplate<Surface, ScalarType>;

    //! Shortcut to the base class.
    using base_class = FaceElementTemplate<Surface, ScalarType>;

   public:
    /**
     * \brief Constructor (derived).
     * @param evaluate_current_normals (in) If the current normals should be evaluated.
     */
    FaceElementPatchTemplate(
        const Core::Elements::Element* core_element, const bool evaluate_current_normals)
        : base_class(core_element),
          connected_faces_(),
          evaluate_current_normals_(evaluate_current_normals)
    {
    }

    /**
     * \brief Set the patch information of the patch connected to this face element.
     *
     * Find the connected faces of this patch. Order the nodes, starting with the nodes of the main
     * face element and then add the nodes of the connected faces. Also create a vector with the
     * GIDs of this patch.
     *
     * @param discret (in) Pointer to the discretization.
     * @param face_elements (in) Vector with all face elements in the surface condition.
     */
    void setup(const std::shared_ptr<const Core::FE::Discretization>& discret,
        const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>& face_elements)
        override;

    /**
     * \brief Set the needed displacement vectors for this face (derived).
     */
    void set_state(const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement,
        const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>& face_elements)
        override;

    /**
     * \brief Calculate the averaged normals on the nodes of this face (derived).
     */
    void calculate_averaged_reference_normals(
        const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>& face_elements)
        override;

    /**
     * \brief Return a normal on the element. (derived)
     */
    void evaluate_face_normal_double(const Core::LinAlg::Matrix<2, 1, double>& xi,
        Core::LinAlg::Matrix<3, 1, double>& n, const bool reference,
        const bool averaged_normal) const override;

   private:
    /**
     * \brief Average normals at the face nodes.
     * @param normals (in) Sum of all normals at a each node.
     * @param averaged_normals (out) Averaged normals, already in the vector format needed for this
     * element.
     */
    template <typename T>
    void average_nodal_normals(
        Core::LinAlg::Matrix<Surface::n_nodes_, 1, Core::LinAlg::Matrix<3, 1, T>>& normals,
        Core::LinAlg::Matrix<3 * Surface::n_nodes_, 1, T>& averaged_normals) const;

   protected:
    //! Store the relevant information of the connected faces. The key of this map is the GID of the
    //! connected face element.
    std::map<int, ConnectedFace> connected_faces_;

    //! If the current normals should be evaluated. If they should not be evaluated, then no
    //! information about the patch is required.
    bool evaluate_current_normals_;
  };


  /**
   * \brief Class to handle extended volume coupling.
   */
  template <typename Surface, typename ScalarType, typename Volume>
  class FaceElementTemplateExtendedVolume : public FaceElementTemplate<Surface, ScalarType>
  {
   public:
    //! Shortcut to the type of this templated object.
    using my_type = FaceElementTemplateExtendedVolume<Surface, ScalarType, Volume>;

    //! Shortcut to the base class.
    using base_class = FaceElementTemplate<Surface, ScalarType>;

   public:
    /**
     * \brief Constructor (derived).
     */
    FaceElementTemplateExtendedVolume(const Core::Elements::Element* core_element)
        : base_class(core_element),
          surface_dof_lid_map_(Core::LinAlg::Initialization::zero),
          face_to_volume_coordinate_axis_map_(Core::LinAlg::Initialization::zero),
          face_to_volume_coordinate_axis_factor_(Core::LinAlg::Initialization::zero),
          third_direction_(-1),
          third_direction_factor_(0) {};

    /**
     * \brief Get the face GIDs and set the reference configuration.
     *
     * @param discret (in) Pointer to the discretization.
     * @param face_elements (in) Vector with all face elements in the surface condition.
     */
    void setup(const std::shared_ptr<const Core::FE::Discretization>& discret,
        const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>& face_elements)
        override;

    /**
     * \brief Set the needed displacement vectors for this face (derived).
     */
    void set_state(const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement,
        const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>& face_elements)
        override;

    /**
     * \brief Calculate the nodal normals for this face element.
     * @tparam scalar_type_normal Scalar type for normals
     * @param volume_position (in) Face position vector of the volume.
     * @param surface_position (in) Face position vector of the surface.
     * @param normals (out) Normals on the nodes.
     */
    template <typename ScalarTypeNormal>
    void calculate_normals(
        const GeometryPair::ElementData<Volume, ScalarTypeNormal>& volume_position,
        const GeometryPair::ElementData<Surface, ScalarTypeNormal>& surface_position,
        Core::LinAlg::Matrix<3 * Surface::n_nodes_, 1, ScalarTypeNormal>& normals) const;

    /**
     * \brief Return a normal on the element. (derived)
     */
    void evaluate_face_normal_double(const Core::LinAlg::Matrix<2, 1, double>& xi,
        Core::LinAlg::Matrix<3, 1, double>& n, const bool reference,
        const bool averaged_normal) const override;

   protected:
    /**
     * \brief Convert the face element parameter coordinates to the volume element parameter
     * coordinates.
     */
    void xi_face_to_xi_volume(const Core::LinAlg::Matrix<2, 1, double>& xi_face,
        Core::LinAlg::Matrix<3, 1, double>& xi_volume) const;

   protected:
    //! Map between surface DOFs and volume DOFs.
    Core::LinAlg::Matrix<Surface::n_dof_, 1, int> surface_dof_lid_map_;

    //! Reference position.
    GeometryPair::ElementData<Volume, double> volume_reference_position_;

    //! Current position.
    GeometryPair::ElementData<Volume, ScalarType> volume_position_;

    //! Map the face coordinate axis to the volume coordinate axis.
    Core::LinAlg::Matrix<2, 1, int> face_to_volume_coordinate_axis_map_;

    //! Factor for the transformed coordinate axis.
    Core::LinAlg::Matrix<2, 1, int> face_to_volume_coordinate_axis_factor_;

    //! Third, i.e. normal direction.
    int third_direction_;

    //! Factor for normal direction.
    int third_direction_factor_;
  };


  /**
   * \brief Create the templated version of the face element.
   * @param core_element (in) Pointer to the element.
   * @param fad_order (in) Order of the created FAD type (0 means double).
   * @param surface_normal_strategy (in) strategy to be used for surface normals.
   * @return RCP to the created GEOMETRYPAIR FaceElement.
   */
  std::shared_ptr<FaceElement> face_element_factory(const Core::Elements::Element* core_element,
      const int fad_order, const GeometryPair::SurfaceNormals surface_normal_strategy);

}  // namespace GeometryPair

FOUR_C_NAMESPACE_CLOSE

#endif

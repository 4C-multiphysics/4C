// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMCONTACT_BEAM3CONTACT_OCTTREE_HPP
#define FOUR_C_BEAMCONTACT_BEAM3CONTACT_OCTTREE_HPP

#include "4C_config.hpp"

#include "4C_beamcontact_beam3contactnew.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

using namespace CONTACT;

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class SparseMatrix;
}

/*!
 \brief Octtree for beam contact search...
 Refer also to the Semesterarbeit of Christian Roth, 2011
*/
class Beam3ContactOctTree
{
 public:
  //!\brief Constructor
  Beam3ContactOctTree(Teuchos::ParameterList& params, Core::FE::Discretization& discret,
      Core::FE::Discretization& searchdis);

  //!\brief Destructor
  virtual ~Beam3ContactOctTree() = default;

  /*!\brief call octtree search routine
   * \param currentposition (in) node positions in column map format (fully overlapping)
   * \param step            (in) time step (needed for output)*/
  std::vector<std::vector<Core::Elements::Element*>> oct_tree_search(
      std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions, int step = -1);

  //!\brief checks in which octant a given bounding box lies
  std::vector<int> in_which_octant_lies(const int& thisBBoxID);

  /*!\brief intersection test of all elements in the octant in which a given bounding box lies
   * \param nodecoords  (in) nodal coordinates
   * \param nodeLID     (in) local Ids of the nodes */
  bool intersect_b_boxes_with(
      Core::LinAlg::SerialDenseMatrix& nodecoords, Core::LinAlg::SerialDenseMatrix& nodeLID);

  /*!\brief output of octree discretization, bounding boxes and contact pairs
   * \param contactpairelements (in) vector with contact pairs
   * \param step   (in) time step */
  void octree_output(
      std::vector<std::vector<Core::Elements::Element*>> contactpairelements, int step);

 private:
  // ! \brief Bounding Box Types available for this search routine
  enum BboxType
  {
    none,
    axisaligned,
    cyloriented,
    spherical
  };

  //!\brief Initialize class vectors for new Octree search
  void initialize_octree_search();
  /*!\brief generator of extended Bounding Boxes (axis aligned as well as cylindrical oriented)
   * \param currentpositions (in) map holding node positions
   */
  void create_bounding_boxes(std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions);
  //!\brief get the dimensions of the root octant
  Core::LinAlg::Matrix<6, 1> get_root_box();
  /*!\brief create axis aligned bounding boxes
   * \param coord      (in)  coordinates of the element's nodes
   * \param elecolid   (in)  element column map Id
   * \param bboxlimits (out) limits of the bounding box*/
  void create_aabb(Core::LinAlg::SerialDenseMatrix& coord, const int& elecolid,
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> bboxlimits = nullptr);
  /*!\brief create coylindrical oriented bounding boxes
   * \param coord      (in)  coordinates of the element's nodes
   * \param elecolid   (in)  element column map Id
   * \param bboxlimits (out) limits of the bounding box*/
  void create_cobb(Core::LinAlg::SerialDenseMatrix& coord, const int& elecolid,
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> bboxlimits = nullptr);
  /*! \brief create spherical bounding boxes for crosslinker
   * \param coord      (in)  coordinates of the element's nodes
   * \param elecolid   (in)  element column map Id*/
  void create_spbb(Core::LinAlg::SerialDenseMatrix& coord, const int& elecolid,
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> bboxlimits = nullptr);


  //!\brief base call for octree build. Returns false if no bounding boxes exist
  bool locate_all();

  /*!\brief recursively locates bounding boxes and maps them to the octant(s) they lie in
   * \param allbboxesstdvec   (in)     std::vector holding all bounding box limits
   * \param lim               (in)     limits of the root octant
   * \param octreelimits      (in/out) vector holding the limits of all octants
   * \param bboxesinoctants   (in/out) vector mapping bounding boxes to octants
   * \param bbox2octant       (in/out) vector mapping bounding boxes to octants they lie in
   * \param treedepth         (in) current tree depth*/
  void locate_box(std::vector<std::vector<double>>& allbboxesstdvec,
      Core::LinAlg::Matrix<6, 1>& lim, std::vector<Core::LinAlg::Matrix<6, 1>>& octreelimits,
      std::vector<std::vector<int>>& bboxesinoctants, std::vector<std::vector<int>>& bbox2octant,
      int& treedepth);

  /*! \brief Subdivide given octant
   *  \param parentoctlimits  (in)  limits of the parent octant
   *  \param suboctedgelength (out) edge length of the sub octants
   *  \param suboctlimits     (out) limits of the 8 sub octants*/
  void create_sub_octants(Core::LinAlg::Matrix<6, 1>& parentoctlimits,
      Core::LinAlg::Matrix<3, 1>& suboctedgelength,
      std::vector<Core::LinAlg::Matrix<6, 1>>& suboctlimits);

  //! \brief Check if axis aligned bounding box is in the current octant
  bool aabb_is_in_this_octant(
      Core::LinAlg::Matrix<6, 1>& suboctlimits, std::vector<double>& bboxcoords, int& shift);
  //! \brief Check if cylindrical oriented bounding box is in the current octant
  bool cobb_is_in_this_octant(Core::LinAlg::Matrix<3, 1>& octcenter,
      Core::LinAlg::Matrix<3, 1>& newedgelength, std::vector<double>& bboxcoords,
      double& extrusionvalue, int& lid, int& shift);
  //! \brief Check if spherical bounding box is in the current octant
  bool spbb_is_in_this_octant(Core::LinAlg::Matrix<3, 1>& octcenter,
      Core::LinAlg::Matrix<3, 1>& newedgelength, std::vector<double>& bboxcoords, int& lid,
      int& shift);

  /*!\brief Manages the intersection of bounding boxes of an octant
   * \param currentpositions  (in)  node positions in column map format (fully overlapping)
   * \param contactpaits      (out) vector holding all contact pairs considered after octree
   * evaluation
   */
  void bounding_box_intersection(std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
      std::vector<std::vector<Core::Elements::Element*>>& contactpairelements);

  /*!\brief intersection method applying axis-aligned bounding boxes when both boxes belong to
   * existing elements \param bboxIDs    (in) vector with bounding box Ids (element GIDs) \param
   * bboxlimits (in) limits of the bounding box */
  bool intersection_aabb(const std::vector<int>& bboxIDs,
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> bboxlimits = nullptr);
  /*!\brief intersection method applying cylindrical oriented bounding boxes when both boxes belong
   * to existing elements \param bboxIDs    (in) vector with bounding box Ids (element GIDs)
   * \param bboxlimits (in) limits of the bounding box */
  bool intersection_cobb(const std::vector<int>& bboxIDs,
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> bboxlimits = nullptr);
  /* !\brief intersection method applying spherical bounding boxes for crosslinker when both boxes
   * belong to existing elements \param bboxIDs    (in) vector with bounding box Ids (element GIDs)
   *  \param bboxlimits (in) limits of the bounding box */
  bool intersection_spbb(const std::vector<int>& bboxIDs,
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> bboxlimits = nullptr);

  /*! \brief communicate Vector to all participating processors
   *  \param InVec    (in) Source/Input vector
   *  \param OutVec   (in) Target/Output vector
   *  \param doexport (in) export flag
   *  \param doimport (in) import flag */
  void communicate_vector(Core::LinAlg::Vector<double>& InVec, Core::LinAlg::Vector<double>& OutVec,
      bool zerofy = false, bool doexport = true, bool doimport = true);
  /*! \brief communicate MultiVector to all participating processors
   *  \param InVec    (in) Source/Input vector
   *  \param OutVec   (in) Target/Output vector
   *  \param doexport (in) export flag
   *  \param doimport (in) import flag */
  void communicate_multi_vector(Core::LinAlg::MultiVector<double>& InVec,
      Core::LinAlg::MultiVector<double>& OutVec, bool zerofy = false, bool doexport = true,
      bool doimport = true);

  /*! \brief Calculate maximal and minimal x-, y- and z-value of a solid elements nodes */
  void calc_corner_pos(Core::Elements::Element* element,
      std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
      Core::LinAlg::SerialDenseMatrix& coord);

  /*! \brief Undo the shifts due periodic BCs and make coord continuous */
  void undo_effect_of_periodic_boundary_condition(
      Core::LinAlg::SerialDenseMatrix& coord, std::vector<int>& cut, int& numshifts);

  /*! \brief Retrieve bounding box specific extrusion value*/
  double get_bounding_box_extrusion_value();

  //! \brief translate std::vec<std::vec<type> > > to Core::LinAlg::MultiVector<double>
  template <class TYPE>
  void std_vec_to_epetra_multi_vec(
      std::vector<std::vector<TYPE>>& stdvector, Core::LinAlg::MultiVector<double>& vector)
  {
    if (std::strcmp(typeid(TYPE).name(), "i") != 0 && std::strcmp(typeid(TYPE).name(), "f") != 0 &&
        std::strcmp(typeid(TYPE).name(), "d") != 0)
      FOUR_C_THROW("Template of wrong type {}! Only int, float, and double are permitted!",
          typeid(TYPE).name());
    if (vector.local_length() != (int)stdvector.size()) FOUR_C_THROW("Sizes differ!");
    for (int i = 0; i < (int)stdvector.size(); i++)
    {
      if ((int)stdvector[i].size() > vector.num_vectors())
        FOUR_C_THROW("stdvector[{}].size() = {} is larger than vector.NumVectors() = {}", i,
            (int)stdvector[i].size(), vector.num_vectors());
      for (int j = 0; j < (int)stdvector[i].size(); j++)
        vector(j).get_values()[i] = (TYPE)stdvector[i][j];
    }
  }
  //! \brief translate Core::LinAlg::MultiVector<double> to std::vec<std::vec<type> > >
  template <class TYPE>
  void epetra_multi_vec_to_std_vec(
      Core::LinAlg::MultiVector<double>& vector, std::vector<std::vector<TYPE>>& stdvector)
  {
    if (std::strcmp(typeid(TYPE).name(), "i") != 0 && std::strcmp(typeid(TYPE).name(), "f") != 0 &&
        std::strcmp(typeid(TYPE).name(), "d") != 0)
      FOUR_C_THROW("Template of wrong type {}! Only int, float, and double are permitted!",
          typeid(TYPE).name());
    if (vector.local_length() != (int)stdvector.size()) FOUR_C_THROW("Sizes differ!");
    for (int i = 0; i < vector.num_vectors(); i++)
    {
      for (int j = 0; j < vector.local_length(); j++)
      {
        if ((int)stdvector[j].size() < vector.num_vectors())
          FOUR_C_THROW("stdvector[{}].size() = {} is larger than vector.NumVectors() = {}", j,
              (int)stdvector[j].size(), vector.num_vectors());
        stdvector[j][i] = vector(i).local_values_as_span()[j];
      }
    }
  }

  //!\brief flag indicating the use of periodic boundary conditions
  bool periodic_bc_;

  //!\brief flag indicating
  bool additiveextrusion_;

  //!\brief flag indicating whether search shall include (beam,solid) contact element pairs
  bool btsol_;

  //!\brief vector holding the edge lengths of the cuboid periodic volume
  std::shared_ptr<std::vector<double>> periodlength_;

  //!\brief Matrix holding the spatial limits of the root box
  Core::LinAlg::Matrix<6, 1> rootbox_;

  //!\brief Matrix holding the spatial limits of the root box in reference configuration
  Core::LinAlg::Matrix<6, 1> initbox_;

  //!\brief problem discretization
  Core::FE::Discretization& discret_;

  //!\brief contact discretization
  Core::FE::Discretization& searchdis_;

  //!\brief number of initial nodes
  int basisnodes_;

  //!\brief maximum tree depth
  int maxtreedepth_;

  //!\brief minimum number of BBs per octant
  int minbboxesinoctant_;

  //!\brief scalar extrusion values for additive or multiplicative extrusion of the bounding box
  std::shared_ptr<std::vector<double>> extrusionvalue_;

  //!\brief diameters of all beam elements
  std::shared_ptr<Core::LinAlg::Vector<double>> diameter_;

  //!\brief stores the IDs and the coordinates of all bounding boxes
  std::shared_ptr<Core::LinAlg::MultiVector<double>> allbboxes_;

  //!\brief vector listing the bounding boxes located in the octants
  std::shared_ptr<Core::LinAlg::MultiVector<double>> bboxesinoctants_;

  //!\brief mapping bounding boxes to octants they lie in
  std::shared_ptr<Core::LinAlg::MultiVector<double>> bbox2octant_;

  //!\brief storage vector for octree octant limits
  std::vector<Core::LinAlg::Matrix<6, 1>> octreelimits_;

  //!\brief vector holding information on how many times a bounding box is shifted due to periodic
  //! boundary conditions
  std::shared_ptr<Core::LinAlg::Vector<double>> numshifts_;

  //!\brief Bounding Box type
  Beam3ContactOctTree::BboxType boundingbox_;

  int numcrit1_;
  int numcrit2_;
};

FOUR_C_NAMESPACE_CLOSE

#endif

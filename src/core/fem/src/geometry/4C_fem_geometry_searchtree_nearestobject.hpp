#ifndef FOUR_C_FEM_GEOMETRY_SEARCHTREE_NEARESTOBJECT_HPP
#define FOUR_C_FEM_GEOMETRY_SEARCHTREE_NEARESTOBJECT_HPP


#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Core::Geo
{
  //! possible positions of a point with respect to an element
  enum ObjectType
  {
    NOTYPE_OBJECT,   ///< closest object not defined
    SURFACE_OBJECT,  ///< closest object is a point
    LINE_OBJECT,     ///< closest object is a line
    NODE_OBJECT      ///< closest object is a surface
  };


  /*!
  \brief  NearestObject stores and delivers all data , which is important
          during a nearest object in tree node search
  */
  class NearestObject
  {
   public:
    /*!
    \brief constructor
    */
    NearestObject();

    /*!
    \brief copy constructor
    */
    NearestObject(const Core::Geo::NearestObject& old);

    /*!
    \brief assignment operator
    */
    Core::Geo::NearestObject& operator=(const Core::Geo::NearestObject& old);

    /*!
    \brief clear nearest object
    */
    void clear();

    /*!
    \brief Set node object type
    \param nodeId       (in)        : node gid
    \param label        (in)        : label
    \param physcoord    (in)        : physical coordinates of point on object
    */
    void set_node_object_type(
        const int nodeId, const int label, const Core::LinAlg::Matrix<3, 1>& physcoord);

    /*!
    \brief Set line object type
    \param lineId       (in)        : line gid
    \param surfId       (in)        : surf gid
    \param label        (in)        : label
    \param physcoord    (in)        : physical coordinates of point on object
    */
    void set_line_object_type(const int lineId, const int surfId, const int label,
        const Core::LinAlg::Matrix<3, 1>& physcoord);

    /*!
    \brief Set surface object type
    \param surfId       (in)        : surf gid
    \param label        (in)        : label
    \param physcoord    (in)        : physical coordinates of point on object
    */
    void set_surface_object_type(
        const int surfId, const int label, const Core::LinAlg::Matrix<3, 1>& physcoord);

    /*!
    \brief Return object type
     */
    inline ObjectType get_object_type() const { return object_type_; }

    /*!
    \brief Return label
     */
    inline int get_label() const { return label_; }

    /*!
    \brief Return vector of physical coordinates
     */
    inline Core::LinAlg::Matrix<3, 1> get_phys_coord() const
    {
      if (object_type_ == NOTYPE_OBJECT)
        FOUR_C_THROW("no object type and physical coordinates are set");
      return physcoord_;
    }


   private:
    //! ObjectType NOTYPE SURFACE LINE NODE
    ObjectType object_type_;

    //! id of node
    int node_id_;

    //! id of line
    int line_id_;

    //! id of surface
    int surf_id_;

    //! label of object
    int label_;

    //! physical coordinates of point on nearest object
    Core::LinAlg::Matrix<3, 1> physcoord_;
  };

}  // namespace Core::Geo

FOUR_C_NAMESPACE_CLOSE

#endif

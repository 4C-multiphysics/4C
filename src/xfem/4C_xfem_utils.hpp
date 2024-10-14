/*----------------------------------------------------------------------*/
/*! \file
\brief Basic tools used in XFEM routines

\level 3


\warning this file should be cleaned up
*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_XFEM_UTILS_HPP
#define FOUR_C_XFEM_UTILS_HPP

#include "4C_config.hpp"

#include "4C_cut_point.hpp"
#include "4C_fem_discretization.hpp"

FOUR_C_NAMESPACE_OPEN

namespace XFEM
{
  namespace Utils
  {
    //! extract the nodal vectors and store them in node-vector-map
    //! \author schott \date 01/13
    void extract_node_vectors(Core::FE::Discretization& dis,
        std::map<int, Core::LinAlg::Matrix<3, 1>>& nodevecmap,
        Teuchos::RCP<Core::LinAlg::Vector<double>> idispnp);

    //! @name Get material properties for the Volume Cell

    /*!

    \brief Element material for the volume cell, depending on element and position.
           If an element which is not a material list is given, the provided material is chosen.
           If however a material list is given the material chosen for the volume cell is depending
    on the point position.

     */
    void get_volume_cell_material(Core::Elements::Element* actele,  // element for volume cell INPUT
        Teuchos::RCP<Core::Mat::Material>& mat,                   // material of volume cell OUTPUT
        Cut::Point::PointPosition position = Cut::Point::outside  // position of volume cell INPUT
                                                                  // to determine position
    );


    //! @name Check whether materials are identical
    /*!

    \brief A Safety check is done for XFEM-type problems. Is utilized in the edgebased framework.

     */
    void safety_check_materials(
        Teuchos::RCP<Core::Mat::Material>& pmat, Teuchos::RCP<Core::Mat::Material>& nmat);

    //! @name Extract quantities on a element
    /*!
    \brief Needs a column-vector to extract correctly in parallel
     */
    void extract_quantity_at_element(Core::LinAlg::SerialDenseMatrix::Base& element_vector,
        const Core::Elements::Element* element, const Epetra_MultiVector& global_col_vector,
        Core::FE::Discretization& dis, const int nds_vector, const int nsd);

    //! @name Extract quantities on a node
    /*!
    \brief Needs a column-vector to extract correctly in parallel
     */
    void extract_quantity_at_node(Core::LinAlg::SerialDenseMatrix::Base& element_vector,
        const Core::Nodes::Node* node, const Epetra_MultiVector& global_col_vector,
        Core::FE::Discretization& dis, const int nds_vector, const unsigned int nsd);

  }  // namespace Utils
}  // namespace XFEM


FOUR_C_NAMESPACE_CLOSE

#endif

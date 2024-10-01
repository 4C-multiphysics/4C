/*----------------------------------------------------------------------*/
/*! \file
\brief Collection of free functions to reduce code duplication between elements

\level 3
*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_SO3_ELEMENT_SERVICE_HPP
#define FOUR_C_SO3_ELEMENT_SERVICE_HPP

#include "4C_config.hpp"

#include "4C_so3_base.hpp"

#include <Epetra_MultiVector.h>

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace UTILS
  {
    class GaussIntegration;
  }
  namespace ELEMENTS
  {
    /*!
     * \brief Assemble nodal element count
     *
     * \param global_count Add a 1 to all nodes belonging to this element
     * \param ele element
     */
    void assemble_nodal_element_count(
        Core::LinAlg::Vector<int>& global_count, const Core::Elements::Element& ele);

    /*!
     * \brief Assemble Gauss point data into an array of global cell data
     *
     * \param global_data array of global cell data (length at least number of gauss points)
     * \param gp_data (numgp x size) matrix of the Gauss point data
     * \param ele element
     */
    void assemble_gauss_point_values(std::vector<Teuchos::RCP<Epetra_MultiVector>>& global_data,
        const Core::LinAlg::SerialDenseMatrix& gp_data, const Core::Elements::Element& ele);

    /*!
     * @brief calculate and return the value of the quantity at position xi based on the
     * quantity node vector
     *
     * @tparam distype        discretization type of element
     * @param xi              position to project to in local coordinates
     * @param nodal_quantity  nodal vector of the quantity to be projected
     * @return quantities projected to position xi
     */
    template <Core::FE::CellType distype>
    std::vector<double> project_nodal_quantity_to_xi(
        const Core::LinAlg::Matrix<3, 1>& xi, const std::vector<double>& nodal_quantity);
  }  // namespace ELEMENTS
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
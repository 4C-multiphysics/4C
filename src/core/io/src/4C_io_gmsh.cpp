// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_gmsh.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"  // Core::LinAlg::Export
#include "4C_rebalance_binning_based.hpp"

FOUR_C_NAMESPACE_OPEN


/*------------------------------------------------------------------------------------------------*
 | write scalar field to Gmsh postprocessing file                                     henke 12/09 |
 *------------------------------------------------------------------------------------------------*/
void Core::IO::Gmsh::scalar_field_to_gmsh(Core::FE::Discretization& discret,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> scalarfield_row, std::ostream& s)
{
  // transform solution vector from dof_row_map to DofColMap
  const std::shared_ptr<const Core::LinAlg::Vector<double>> scalarfield =
      Core::Rebalance::get_col_version_of_row_vector(discret, scalarfield_row);

  // loop all row elements on this processor
  for (int iele = 0; iele < discret.num_my_row_elements(); ++iele)
  {
    const Core::Elements::Element* ele = discret.l_row_element(iele);
    const Core::FE::CellType distype = ele->shape();
    const int numnode = distype_to_gmsh_num_node(distype);

    Core::LinAlg::SerialDenseMatrix xyze(3, numnode);

    const Core::Nodes::Node* const* nodes = ele->nodes();
    for (int inode = 0; inode < numnode; ++inode)
    {
      xyze(0, inode) = nodes[inode]->x()[0];
      xyze(1, inode) = nodes[inode]->x()[1];
      xyze(2, inode) = nodes[inode]->x()[2];
    }

    s << "S";  // scalar field indicator
    s << distype_to_gmsh_element_header(distype);

    // write node coordinates to Gmsh stream
    coordinates_to_stream(xyze, distype, s);

    std::vector<int> lm;
    std::vector<int> lmowner;
    std::vector<int> lmstride;
    ele->location_vector(discret, lm, lmowner, lmstride);

    // extract local values from the global vector
    Core::LinAlg::SerialDenseVector myscalarfield(lm.size());
    Core::FE::extract_my_values(*scalarfield, myscalarfield, lm);

    // write scalar field to Gmsh stream
    scalar_field_to_stream(myscalarfield, distype, s);

    s << "\n";
  }
}


/*------------------------------------------------------------------------------------------------*
 | write scalar field to Gmsh postprocessing file                                     henke 12/09 |
 *------------------------------------------------------------------------------------------------*/
void Core::IO::Gmsh::scalar_field_dof_based_to_gmsh(Core::FE::Discretization& discret,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> scalarfield_row, const int nds,
    std::ostream& s)
{
  // transform solution vector from dof_row_map to DofColMap
  const std::shared_ptr<const Core::LinAlg::Vector<double>> scalarfield =
      Core::Rebalance::get_col_version_of_row_vector(discret, scalarfield_row, nds);

  // loop all row elements on this processor
  for (int iele = 0; iele < discret.num_my_row_elements(); ++iele)
  {
    const Core::Elements::Element* ele = discret.l_row_element(iele);
    const Core::FE::CellType distype = ele->shape();
    const int numnode = distype_to_gmsh_num_node(distype);

    Core::LinAlg::SerialDenseMatrix xyze(3, numnode);

    const Core::Nodes::Node* const* nodes = ele->nodes();
    for (int inode = 0; inode < numnode; ++inode)
    {
      xyze(0, inode) = nodes[inode]->x()[0];
      xyze(1, inode) = nodes[inode]->x()[1];
      xyze(2, inode) = nodes[inode]->x()[2];
    }

    s << "S";  // scalar field indicator
    s << distype_to_gmsh_element_header(distype);

    // write node coordinates to Gmsh stream
    coordinates_to_stream(xyze, distype, s);

    Core::Elements::LocationArray la(discret.num_dof_sets());
    ele->location_vector(discret, la);

    // extract local values from the global vector
    Core::LinAlg::SerialDenseVector myscalarfield(la[nds].lm_.size());
    Core::FE::extract_my_values(*scalarfield, myscalarfield, la[nds].lm_);

    //    // Extract velocity from local velnp_
    //    Core::LinAlg::SerialDenseMatrix myvectorfield(nsd,numnode);
    //    for (int inode = 0; inode < numnode; ++inode)
    //    {
    //      for (int idim = 0; idim < nsd; ++idim)
    //      {
    //        double val = extractmyvectorfield(idim + inode*nsd);
    //        myvectorfield(idim,inode)= val;
    //        if( displacenodes )
    //          xyze(idim,inode) += val;
    //      }
    //    }
    //
    //    std::vector<int> lm;
    //    std::vector<int> lmowner;
    //    std::vector<int> lmstride;
    //    ele->LocationVector(*discret, lm, lmowner, lmstride);
    //
    //    // extract local values from the global vector
    //    Core::LinAlg::SerialDenseVector myscalarfield(lm.size());
    //    Core::FE::extract_my_values(*scalarfield, myscalarfield, lm);

    // write scalar field to Gmsh stream
    scalar_field_to_stream(myscalarfield, distype, s);

    s << "\n";
  }
}

/*------------------------------------------------------------------------------------------------*
 | write scalar element-based field to Gmsh postprocessing file                  winklmaier 12/09 |
 *------------------------------------------------------------------------------------------------*/
void Core::IO::Gmsh::scalar_element_field_to_gmsh(const Core::FE::Discretization& discret,
    const Core::LinAlg::Vector<double>& scalarfield_ele_row, std::ostream& s)
{
  if (scalarfield_ele_row.get_map().same_as(*discret.element_row_map()) == false)
  {
    std::cout << scalarfield_ele_row.get_map() << std::endl
              << *discret.element_row_map() << std::endl;
    FOUR_C_THROW("The written field should be based on the element row map");
  }

  // currently the element value is used for every node of the element so that
  // we result in the correct solution - gmsh writes nodal values for every
  // element, so that the same node can get different values for different
  // elements (similar to discontinuous methods)
  for (int iele = 0; iele < discret.num_my_row_elements(); ++iele)
  {
    const Core::Elements::Element* ele = discret.l_row_element(iele);
    const Core::FE::CellType distype = ele->shape();
    const int numnode = distype_to_gmsh_num_node(distype);

    Core::LinAlg::SerialDenseMatrix xyze(3, numnode);

    const Core::Nodes::Node* const* nodes = ele->nodes();
    for (int inode = 0; inode < numnode; ++inode)
    {
      xyze(0, inode) = nodes[inode]->x()[0];
      xyze(1, inode) = nodes[inode]->x()[1];
      xyze(2, inode) = nodes[inode]->x()[2];
    }

    s << "S";  // scalar field indicator
    s << distype_to_gmsh_element_header(distype);

    // write node coordinates to Gmsh stream
    coordinates_to_stream(xyze, distype, s);

    double eleval = (scalarfield_ele_row).local_values_as_span()[iele];

    // constant value for all nodes
    Core::LinAlg::SerialDenseVector myscalarfield(ele->num_node());
    for (int inode = 0; inode < numnode; ++inode) myscalarfield(inode) = eleval;

    // write scalar field to Gmsh stream
    scalar_field_to_stream(myscalarfield, distype, s);

    s << "\n";
  }
}

/*------------------------------------------------------------------------------------------------*
 | write dof-based vector field to Gmsh postprocessing file                           henke 12/09 |
 *------------------------------------------------------------------------------------------------*/
void Core::IO::Gmsh::vector_field_dof_based_to_gmsh(Core::FE::Discretization& discret,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> vectorfield_row, std::ostream& s,
    const int nds, bool displacenodes)
{
  // transform solution vector from dof_row_map to DofColMap
  const std::shared_ptr<const Core::LinAlg::Vector<double>> vectorfield =
      Core::Rebalance::get_col_version_of_row_vector(discret, vectorfield_row, nds);

  // loop all row elements on this processor
  for (int iele = 0; iele < discret.num_my_row_elements(); ++iele)
  {
    const Core::Elements::Element* ele = discret.l_row_element(iele);
    const Core::FE::CellType distype = ele->shape();
    const int numnode = distype_to_gmsh_num_node(distype);
    const int nsd = Core::FE::get_dimension(distype);

    Core::LinAlg::SerialDenseMatrix xyze(nsd, numnode);

    const Core::Nodes::Node* const* nodes = ele->nodes();
    for (int inode = 0; inode < numnode; ++inode)
    {
      for (int idim = 0; idim < nsd; ++idim) xyze(idim, inode) = nodes[inode]->x()[idim];
    }

    Core::Elements::LocationArray la(discret.num_dof_sets());
    ele->location_vector(discret, la);

    // extract local values from the global vector
    Core::LinAlg::SerialDenseVector extractmyvectorfield(la[nds].lm_.size());
    Core::FE::extract_my_values(*vectorfield, extractmyvectorfield, la[nds].lm_);

    // Extract velocity from local velnp_
    Core::LinAlg::SerialDenseMatrix myvectorfield(nsd, numnode);
    for (int inode = 0; inode < numnode; ++inode)
    {
      for (int idim = 0; idim < nsd; ++idim)
      {
        double val = extractmyvectorfield(idim + inode * nsd);
        myvectorfield(idim, inode) = val;
        if (displacenodes) xyze(idim, inode) += val;
      }
    }

    s << "V";  // vector field indicator
    s << distype_to_gmsh_element_header(distype);

    // write node coordinates to Gmsh stream
    coordinates_to_stream(xyze, distype, s);

    // write vector field to Gmsh stream
    // remark: only the first 3 components are written to the Gmsh postprocessing file
    //         -> e.g. pressure in velnp_ fluid state vector is not written ('myvectorfield':
    //         velx,vely,velz,pressure)
    vector_field_to_stream(myvectorfield, distype, s);

    s << "\n";
  }
}

/*------------------------------------------------------------------------------------------------*
 | write multivector field to Gmsh postprocessing file                               winter 04/17 |
 *------------------------------------------------------------------------------------------------*/
void Core::IO::Gmsh::vector_field_multi_vector_dof_based_to_gmsh(
    const Core::FE::Discretization& discret,
    const Core::LinAlg::MultiVector<double>& vectorfield_row, std::ostream& s, const int nds)
{
  // TODO: Remove dependence on size of Epetra_Multivector!!!

  // transform solution vector from dof_row_map to DofColMap
  Core::LinAlg::MultiVector<double> vectorfield(*discret.dof_col_map(nds), 3, true);
  Core::LinAlg::export_to(vectorfield_row, vectorfield);

  // loop all row elements on this processor
  for (int iele = 0; iele < discret.num_my_row_elements(); ++iele)
  {
    const Core::Elements::Element* ele = discret.l_row_element(iele);
    const Core::FE::CellType distype = ele->shape();
    const int numnode = distype_to_gmsh_num_node(distype);
    const int nsd = Core::FE::get_dimension(distype);

    Core::LinAlg::SerialDenseMatrix xyze(nsd, numnode);

    const Core::Nodes::Node* const* nodes = ele->nodes();
    for (int inode = 0; inode < numnode; ++inode)
    {
      for (int idim = 0; idim < nsd; ++idim)
      {
        xyze(idim, inode) = nodes[inode]->x()[idim];
      }
    }

    s << "V";  // vector field indicator
    s << distype_to_gmsh_element_header(distype);

    // write node coordinates to Gmsh stream
    coordinates_to_stream(xyze, distype, s);

    Core::Elements::LocationArray la(discret.num_dof_sets());
    ele->location_vector(discret, la);

    // extract local values from the global vector
    // Core::LinAlg::SerialDenseVector extractmyvectorfield(la[nds].lm_.size());
    Core::LinAlg::SerialDenseMatrix myvectorfield(nsd, numnode);
    std::vector<double> local_vector = Core::FE::extract_values(vectorfield, la[nds].lm_);

    for (int inode = 0; inode < numnode; ++inode)
    {
      for (int idim = 0; idim < nsd; ++idim)
      {
        myvectorfield(idim, inode) = local_vector[idim + nsd * inode];
      }
    }

    // write vector field to Gmsh stream
    vector_field_to_stream(myvectorfield, distype, s);

    s << "\n";
  }
}

/*------------------------------------------------------------------------------------------------*
 | write dof-based vector field to Gmsh postprocessing file at current position      schott 12/09 |
 *------------------------------------------------------------------------------------------------*/
void Core::IO::Gmsh::surface_vector_field_dof_based_to_gmsh(Core::FE::Discretization& discret,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> vectorfield_row,
    std::map<int, Core::LinAlg::Matrix<3, 1>>& currpos, std::ostream& s, const int nsd,
    const int numdofpernode)
{
  // transform solution vector from dof_row_map to DofColMap
  const std::shared_ptr<const Core::LinAlg::Vector<double>> vectorfield =
      Core::Rebalance::get_col_version_of_row_vector(discret, vectorfield_row);

  // loop all row elements on this processor
  for (int iele = 0; iele < discret.num_my_row_elements(); ++iele)
  {
    const Core::Elements::Element* ele = discret.l_row_element(iele);
    const Core::FE::CellType distype = ele->shape();
    const int numnode = distype_to_gmsh_num_node(distype);

    Core::LinAlg::SerialDenseMatrix xyze(nsd, numnode);

    const Core::Nodes::Node* const* nodes = ele->nodes();
    for (int inode = 0; inode < numnode; ++inode)
    {
      int nid = nodes[inode]->id();
      for (int idim = 0; idim < nsd; ++idim)
        xyze(idim, inode) = ((currpos.find(nid))->second)(idim, 0);
    }

    s << "V";  // vector field indicator
    s << distype_to_gmsh_element_header(distype);

    // write node coordinates to Gmsh stream
    coordinates_to_stream(xyze, distype, s);

    std::vector<int> lm;
    std::vector<int> lmowner;
    std::vector<int> lmstride;
    ele->location_vector(discret, lm, lmowner, lmstride);

    // extract local values from the global vector
    Core::LinAlg::SerialDenseVector extractmyvectorfield(lm.size());
    Core::FE::extract_my_values(*vectorfield, extractmyvectorfield, lm);

    // Extract velocity from local velnp_
    Core::LinAlg::SerialDenseMatrix myvectorfield(nsd, numnode);
    for (int inode = 0; inode < numnode; ++inode)
      for (int idim = 0; idim < nsd; ++idim)
        myvectorfield(idim, inode) = extractmyvectorfield(idim + inode * numdofpernode);

    // write vector field to Gmsh stream
    // remark: only the first 3 components are written to the Gmsh postprocessing file
    //         -> e.g. pressure in velnp_ fluid state vector is not written ('myvectorfield':
    //         velx,vely,velz,pressure)
    vector_field_to_stream(myvectorfield, distype, s);

    s << "\n";
  }
}

/*------------------------------------------------------------------------------------------------*
 | write scalar / vector from a dof-based vector field (e.g. velocity)                            |
 | to Gmsh postprocessing file                                                         ehrl 05/11 |
 *------------------------------------------------------------------------------------------------*/
void Core::IO::Gmsh::velocity_pressure_field_dof_based_to_gmsh(Core::FE::Discretization& discret,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> vectorfield_row,
    const std::string field, std::ostream& s, const int nds)
{
  // transform solution vector from dof_row_map to DofColMap
  const std::shared_ptr<const Core::LinAlg::Vector<double>> vectorfield =
      Core::Rebalance::get_col_version_of_row_vector(discret, vectorfield_row, nds);

  // loop all row elements on this processor
  for (int iele = 0; iele < discret.num_my_row_elements(); ++iele)
  {
    const Core::Elements::Element* ele = discret.l_row_element(iele);
    const Core::FE::CellType distype = ele->shape();
    const int numnode = distype_to_gmsh_num_node(distype);
    const int nsd = Core::FE::get_dimension(distype);

    Core::LinAlg::SerialDenseMatrix xyze(nsd, numnode);

    const Core::Nodes::Node* const* nodes = ele->nodes();
    for (int inode = 0; inode < numnode; ++inode)
    {
      for (int idim = 0; idim < nsd; ++idim) xyze(idim, inode) = nodes[inode]->x()[idim];
    }

    Core::Elements::LocationArray la(discret.num_dof_sets());
    ele->location_vector(discret, la);

    // extract local values from the global vector
    Core::LinAlg::SerialDenseVector extractmyvectorfield(la[nds].lm_.size());
    Core::FE::extract_my_values(*vectorfield, extractmyvectorfield, la[nds].lm_);

    if (field == "pressure")
    {
      // Extract scalar from local velnp_
      Core::LinAlg::SerialDenseVector myscalarfield(numnode);
      for (int inode = 0; inode < numnode; ++inode)
        myscalarfield(inode) = extractmyvectorfield(nsd + inode * (nsd + 1));

      // replace function: cellWithScalarFieldToStream(distype, myscalarfield, xyze, s);
      {
        s << "S";  // scalar field indicator
        s << distype_to_gmsh_element_header(distype);
        if (nsd == 3)
          coordinates_to_stream(xyze, distype, s);
        else if (nsd == 2)
          coordinates_to_stream2_d(xyze, distype, s);
        else
          FOUR_C_THROW("only two- and three-dimensional domains are supported");

        scalar_field_to_stream(myscalarfield, distype, s);
        s << "\n";
      }
    }
    else if (field == "velocity")
    {
      // Extract velocity from local velnp_
      Core::LinAlg::SerialDenseMatrix myvectorfield(nsd, numnode);
      for (int inode = 0; inode < numnode; ++inode)
        for (int idim = 0; idim < nsd; ++idim)
          myvectorfield(idim, inode) = extractmyvectorfield(idim + inode * (nsd + 1));

      // replace function : cellWithVectorFieldToStream(distype, myvectorfield, xyze, s);
      {
        s << "V";  // scalar field indicator
        s << distype_to_gmsh_element_header(distype);
        if (nsd == 3)
        {
          coordinates_to_stream(xyze, distype, s);
          vector_field_to_stream(myvectorfield, distype, s);
        }
        else if (nsd == 2)
        {
          coordinates_to_stream2_d(xyze, distype, s);
          vector_field_to_stream2_d(myvectorfield, distype, s);
        }
        else
          FOUR_C_THROW("");
        s << "\n";
      }
    }
    else
      FOUR_C_THROW("The chosen field does not exist (wrong writing, ...)");

    s << "\n";
  }
}

/*------------------------------------------------------------------------------------------------*
 | write node-based vector field to Gmsh postprocessing file                          henke 12/09 |
 *------------------------------------------------------------------------------------------------*/
void Core::IO::Gmsh::vector_field_node_based_to_gmsh(const Core::FE::Discretization& discret,
    const Core::LinAlg::MultiVector<double>& vectorfield_row, std::ostream& s)
{
  // transform solution vector from NodeRowMap to NodeColMap
  // remark: Core::Rebalance::get_col_version_of_row_vector() does only work for
  // Core::LinAlg::Vectors on dof_row_map
  Core::LinAlg::MultiVector<double> vectorfield(*discret.node_col_map(), 3, true);
  Core::LinAlg::export_to(vectorfield_row, vectorfield);

  // loop all row elements on this processor
  for (int iele = 0; iele < discret.num_my_row_elements(); ++iele)
  {
    const Core::Elements::Element* ele = discret.l_row_element(iele);
    const Core::FE::CellType distype = ele->shape();
    const int numnode = ele->num_node();
    const int nsd = Core::FE::get_dimension(distype);

    Core::LinAlg::SerialDenseMatrix xyze(nsd, numnode);

    const Core::Nodes::Node* const* nodes = ele->nodes();
    for (int inode = 0; inode < numnode; ++inode)
    {
      for (int idim = 0; idim < nsd; ++idim) xyze(idim, inode) = nodes[inode]->x()[idim];
    }

    s << "V";  // vector field indicator
    s << distype_to_gmsh_element_header(distype);

    // write node coordinates to Gmsh stream
    coordinates_to_stream(xyze, distype, s);

    // extract local values from the global vector
    Core::LinAlg::SerialDenseMatrix myvectorfield(nsd, numnode);
    Core::FE::extract_my_node_based_values(ele, myvectorfield, vectorfield, nsd);

    // write vector field to Gmsh stream
    vector_field_to_stream(myvectorfield, distype, s);

    s << "\n";
  }
}


/*------------------------------------------------------------------------------------------------*
 | write node-based scalar field to Gmsh postprocessing file                      rasthofer 09/13 |
 *------------------------------------------------------------------------------------------------*/
void Core::IO::Gmsh::scalar_field_node_based_to_gmsh(const Core::FE::Discretization& discret,
    const Core::LinAlg::Vector<double>& scalarfield_row, std::ostream& s)
{
  // transform solution vector from NodeRowMap to NodeColMap
  // remark: Core::Rebalance::get_col_version_of_row_vector() does only work for
  // Core::LinAlg::Vectors on dof_row_map
  //         something similar is done in COMBUST::FlameFront::ProcessFlameFront, although not for
  //         Epetra_MultiVectors
  Core::LinAlg::Vector<double> scalarfield(*discret.node_col_map(), true);
  Core::LinAlg::export_to(scalarfield_row, scalarfield);

  // loop all row elements on this processor
  for (int iele = 0; iele < discret.num_my_row_elements(); ++iele)
  {
    const Core::Elements::Element* ele = discret.l_row_element(iele);
    const Core::FE::CellType distype = ele->shape();
    const int numnode = ele->num_node();

    Core::LinAlg::SerialDenseMatrix xyze(3, numnode);

    const Core::Nodes::Node* const* nodes = ele->nodes();
    for (int inode = 0; inode < numnode; ++inode)
    {
      xyze(0, inode) = nodes[inode]->x()[0];
      xyze(1, inode) = nodes[inode]->x()[1];
      xyze(2, inode) = nodes[inode]->x()[2];
    }

    s << "S";  // vector field indicator
    s << distype_to_gmsh_element_header(distype);

    // write node coordinates to Gmsh stream
    coordinates_to_stream(xyze, distype, s);

    // extract local values from the global vector
    Core::LinAlg::SerialDenseVector myscalarfield(numnode);
    Core::FE::extract_my_node_based_values(ele, myscalarfield, scalarfield, 1);

    // write vector field to Gmsh stream
    scalar_field_to_stream(myscalarfield, distype, s);

    s << "\n";
  }
}

void Core::IO::Gmsh::scalar_to_stream(
    const Core::LinAlg::Matrix<3, 1>& pointXYZ,  ///< coordinates of point
    const double scalarvalue,                    ///< scalar value at this point
    std::ostream& s                              ///< stream
)
{
  s.setf(std::ios::scientific, std::ios::floatfield);
  s.precision(12);

  s << "SP(";  // scalar field indicator
  s << pointXYZ(0) << ",";
  s << pointXYZ(1) << ",";
  s << pointXYZ(2) << ")";

  s << "{" << scalarvalue << "};";
  s << "\n";
}

void Core::IO::Gmsh::vector_to_stream(
    const Core::LinAlg::Matrix<3, 1>& pointXYZ,     ///< coordinates of point
    const Core::LinAlg::Matrix<3, 1>& vectorvalue,  ///< vector at this point
    std::ostream& s                                 ///< stream
)
{
  s.setf(std::ios::scientific, std::ios::floatfield);
  s.precision(12);

  s << "VP(";  // vector field indicator
  s << pointXYZ(0) << ",";
  s << pointXYZ(1) << ",";
  s << pointXYZ(2) << ")";

  s << "{";
  s << vectorvalue(0);
  s << "," << vectorvalue(1);
  s << "," << vectorvalue(2);
  s << "};";  //<< endl;
  s << "\n";
}

void Core::IO::Gmsh::element_at_initial_position_to_stream(
    const double scalar, const Core::Elements::Element* ele, std::ostream& s)
{
  const Core::Nodes::Node* const* nodes = ele->nodes();

  const Core::FE::CellType distype = ele->shape();
  const int numnode = distype_to_gmsh_num_node(distype);

  s.setf(std::ios::scientific, std::ios::floatfield);
  s.precision(12);

  s << "S" << distype_to_gmsh_element_header(distype) << "(";
  for (int i = 0; i < numnode; ++i)
  {
    const Core::Nodes::Node* node = nodes[i];
    const auto& x = node->x();
    s << x[0] << ",";
    s << x[1] << ",";
    s << x[2];
    if (i < numnode - 1)
    {
      s << ",";
    }
  }
  s << ")";
  // values
  scalar_to_stream(scalar, distype, s);
  s << "\n";
}


std::string Core::IO::Gmsh::element_at_initial_position_to_string(
    const double scalar, const Core::Elements::Element* ele)
{
  std::ostringstream s;
  element_at_initial_position_to_stream(scalar, ele, s);
  return s.str();
}


void Core::IO::Gmsh::element_at_current_position_to_stream(const double scalar,
    const Core::Elements::Element* ele,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentelepositions, std::ostream& s)
{
  Core::IO::Gmsh::cell_with_scalar_to_stream(
      ele->shape(), scalar, Core::Geo::get_current_nodal_positions(ele, currentelepositions), s);
}


std::string Core::IO::Gmsh::element_at_current_position_to_string(const double scalar,
    const Core::Elements::Element* ele,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentelepositions)
{
  std::ostringstream s;
  Core::IO::Gmsh::element_at_current_position_to_stream(scalar, ele, currentelepositions, s);
  return s.str();
}


std::string Core::IO::Gmsh::text3d_to_string(
    const Core::LinAlg::Matrix<3, 1>& xyz,  ///< 3d Position of text
    const std::string& text,                ///< text to be printed
    const int fontsize                      ///< font size
)
{
  std::ostringstream s;

  s << "T3";
  // coordinates
  s << "(";
  s << std::scientific << xyz(0) << ",";
  s << std::scientific << xyz(1) << ",";
  s << std::scientific << xyz(2) << ",";
  s << fontsize << ")";
  s << "{\"" << text << "\"};";
  s << "\n";
  return s.str();
}

void Core::IO::Gmsh::dis_to_stream(const std::string& text, const double scalar,
    const Core::FE::Discretization& dis, std::ostream& s)
{
  s << "View \" " << text << " Elements \" {\n";
  for (int i = 0; i < dis.num_my_row_elements(); ++i)
  {
    const Core::Elements::Element* actele = dis.l_row_element(i);
    Core::IO::Gmsh::element_at_initial_position_to_stream(scalar, actele, s);
  }
  s << "};\n";
}

std::string Core::IO::Gmsh::dis_to_string(
    const std::string& text, const double scalar, Core::FE::Discretization& dis)
{
  std::ostringstream s;
  dis_to_stream(text, scalar, dis, s);
  return s.str();
}

void Core::IO::Gmsh::dis_to_stream(const std::string& text, const double scalar,
    const Core::FE::Discretization& dis,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions, std::ostream& s)
{
  s << "View \" " << text << " Elements \" {\n";

  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    const Core::Elements::Element* actele = dis.l_col_element(i);
    Core::IO::Gmsh::cell_with_scalar_to_stream(actele->shape(), scalar,
        Core::Geo::get_current_nodal_positions(actele, currentpositions), s);
  }
  s << "};\n";
}

std::string Core::IO::Gmsh::dis_to_string(const std::string& text, const double scalar,
    Core::FE::Discretization& dis,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions)
{
  std::ostringstream s;
  dis_to_stream(text, scalar, dis, currentpositions, s);
  return s.str();
}

std::string Core::IO::Gmsh::get_new_file_name_and_delete_old_files(const std::string& filename_base,
    const std::string& file_name_prefix, const int& actstep, const int& step_diff,
    const bool screen_out, const int pid)
{
  std::ostringstream filename;
  std::ostringstream filenamedel;

  std::ostringstream pid_stream;
  pid_stream << ".p" << std::setw(2) << std::setfill('0') << pid;

  filename << file_name_prefix << "." << filename_base << "_" << std::setw(5) << std::setfill('0')
           << actstep << pid_stream.str() << ".pos";
  filenamedel << file_name_prefix << "." << filename_base << "_" << std::setw(5)
              << std::setfill('0') << actstep - step_diff << pid_stream.str() << ".pos";
  std::remove(filenamedel.str().c_str());
  if (screen_out)
    std::cout << "writing " << std::left << std::setw(60) << filename.str() << "..." << '\n';
  return filename.str();
}

std::string Core::IO::Gmsh::get_file_name(const std::string& filename_base,
    const std::string& file_name_prefix, const int& actstep, const bool screen_out, const int pid)
{
  std::ostringstream filename;
  std::ostringstream pid_stream;
  pid_stream << ".p" << std::setw(2) << std::setfill('0') << pid;

  filename << file_name_prefix << "." << filename_base << "_" << std::setw(5) << std::setfill('0')
           << actstep << pid_stream.str() << ".pos";

  if (screen_out)
    std::cout << "writing " << std::left << std::setw(60) << filename.str() << "..." << '\n';

  return filename.str();
}

FOUR_C_NAMESPACE_CLOSE

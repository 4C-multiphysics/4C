// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_GMSH_HPP
#define FOUR_C_IO_GMSH_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  namespace Gmsh
  {
    //! for each Core::FE::CellType find the appropriate Gmsh file type
    inline std::string distype_to_gmsh_element_header(
        const Core::FE::CellType distype  ///< element shape
    )
    {
      switch (distype)
      {
        case Core::FE::CellType::hex8:
          return "H";
          break;
        case Core::FE::CellType::hex20:
          return "H";
          break;
        case Core::FE::CellType::hex27:
          return "H";
          break;
        case Core::FE::CellType::tet4:
          return "S";
          break;
        case Core::FE::CellType::tet10:
          return "S";
          break;
        case Core::FE::CellType::point1:
          return "P";
          break;
        case Core::FE::CellType::quad4:
          return "Q";
          break;
        case Core::FE::CellType::quad8:
          return "Q";
          break;
        case Core::FE::CellType::quad9:
          return "Q";
          break;
        case Core::FE::CellType::tri3:
          return "T";
          break;
        case Core::FE::CellType::tri6:
          return "T";
          break;
        case Core::FE::CellType::line2:
          return "L";
          break;
        case Core::FE::CellType::line3:
          return "L2";
          break;
        case Core::FE::CellType::wedge6:
          return "I";
          break;
        case Core::FE::CellType::wedge15:
          return "I";
          break;
        default:
          FOUR_C_THROW("distypeToGmshElementHeader: distype not supported for printout!");
      }
      return "xxx";
    }

    //! for each Core::FE::CellType find the appropriate number of element nodes for
    //! Gmsh output
    inline int distype_to_gmsh_num_node(const Core::FE::CellType distype  ///< element shape
    )
    {
      switch (distype)
      {
        case Core::FE::CellType::hex8:
          return 8;
          break;
        case Core::FE::CellType::hex20:
          return 8;
          break;
        case Core::FE::CellType::hex27:
          return 8;
          break;
        case Core::FE::CellType::tet4:
          return 4;
          break;
        case Core::FE::CellType::tet10:
          return 4;
          break;
        case Core::FE::CellType::point1:
          return 1;
          break;
        case Core::FE::CellType::quad4:
          return 4;
          break;
        case Core::FE::CellType::quad8:
          return 4;
          break;
        case Core::FE::CellType::quad9:
          return 4;
          break;
        case Core::FE::CellType::tri3:
          return 3;
          break;
        case Core::FE::CellType::tri6:
          return 3;
          break;
        case Core::FE::CellType::line2:
          return 2;
          break;
        case Core::FE::CellType::line3:
          return 3;
          break;
        case Core::FE::CellType::wedge6:
          return 6;
          break;
        case Core::FE::CellType::wedge15:
          return 6;
          break;
        default:
          FOUR_C_THROW("distypeToGmshNumNode: distype not supported for printout!");
      }
      return -1;
    }

    //! write scalar field to Gmsh postprocessing file
    void scalar_field_to_gmsh(Core::FE::Discretization& discret,  ///< discretization
        const std::shared_ptr<const Core::LinAlg::Vector<double>>
            scalarfield,  ///< scalar field to output
        std::ostream& s   ///< output stream
    );

    //! write scalar field to Gmsh postprocessing file
    //  Can we replace the scalar_field_to_gmsh function with this (nds=0)?
    void scalar_field_dof_based_to_gmsh(Core::FE::Discretization& discret,  ///< discretization
        const std::shared_ptr<const Core::LinAlg::Vector<double>>
            scalarfield,  ///< scalar field to output
        const int nds,    ///< dofset
        std::ostream& s   ///< output stream
    );

    //! write scalar field to Gmsh postprocessing file
    void scalar_element_field_to_gmsh(const Core::FE::Discretization& discret,  ///< discretization
        const Core::LinAlg::Vector<double>& scalarfield,  ///< scalar field to output
        std::ostream& s                                   ///< output stream
    );

    //! write dof-based vector field to Gmsh postprocessing file
    //! when writing the given vectorfield is displacement, displacenodes can be set true
    //! then it writes the nodal coordinates at the present time step.
    void vector_field_dof_based_to_gmsh(Core::FE::Discretization& discret,  ///< discretization
        const std::shared_ptr<const Core::LinAlg::Vector<double>>
            vectorfield,    ///< vector field to output
        std::ostream& s,    ///< output stream
        const int nds = 0,  ///< number of dofset associated with vector field
        bool displacenodes = false);

    //! write dof-based vector field to Gmsh postprocessing file
    void vector_field_multi_vector_dof_based_to_gmsh(
        const Core::FE::Discretization& discret,               ///< discretization
        const Core::LinAlg::MultiVector<double>& vectorfield,  ///< vector field to output
        std::ostream& s,                                       ///< output stream
        const int nds = 0                                      //< which dof-set to use from vector
    );

    //! write node-based vector field to Gmsh postprocessing file
    void vector_field_node_based_to_gmsh(
        const Core::FE::Discretization& discret,               ///< discretization
        const Core::LinAlg::MultiVector<double>& vectorfield,  ///< vector field to output
        std::ostream& s                                        ///< output stream
    );

    //! write dof-based vector field to Gmsh postprocessing file
    void surface_vector_field_dof_based_to_gmsh(
        Core::FE::Discretization& discret,  ///< discretization
        const std::shared_ptr<const Core::LinAlg::Vector<double>>
            vectorfield,  ///< vector field to output
        std::map<int, Core::LinAlg::Matrix<3, 1>>& currpos,
        std::ostream& s,  ///< output stream
        const int nsd, const int numdofpernode);

    //! write dof-based velocity / pressure field to Gmsh postprocessing file
    //!
    void velocity_pressure_field_dof_based_to_gmsh(
        Core::FE::Discretization& discret,  ///< discretization
        const std::shared_ptr<const Core::LinAlg::Vector<double>>
            vectorfield,          ///< vector field to output
        const std::string field,  ///< "velocity" or "pressure"
        std::ostream& s,          ///< output stream
        const int nds = 0         ///< which dof-set to use from vector
    );

    //! write node-based scalar field to Gmsh postprocessing file
    void scalar_field_node_based_to_gmsh(
        const Core::FE::Discretization& discret,          ///< discretization
        const Core::LinAlg::Vector<double>& scalarfield,  ///< scalar field to output
        std::ostream& s                                   ///< output stream
    );

    //! take an array (3,numnode) and translate it to the coordinate section of a Gmsh element entry
    template <class M>
    inline void coordinates_to_stream(const M& coord,  ///< position array (3, numnode)
        const Core::FE::CellType distype,              ///< element shape
        std::ostream& s                                ///< output stream
    )
    {
      s.setf(std::ios::scientific, std::ios::floatfield);
      s.precision(12);

      const int numnode = distype_to_gmsh_num_node(distype);

      // coordinates
      s << "(";
      for (int inen = 0; inen < numnode; ++inen)
      {
        // print position
        s << coord(0, inen) << ",";
        s << coord(1, inen) << ",";
        s << coord(2, inen);
        if (inen < numnode - 1)
        {
          s << ",";
        }
      };
      s << ")";
      return;
    }

    //! take an array (3,numnode) and translate it to the coordinate section of a Gmsh element entry
    template <class M>
    inline void coordinates_to_stream2_d(const M& coord,  ///< position array (3, numnode)
        const Core::FE::CellType distype,                 ///< element shape
        std::ostream& s)
    {
      s.setf(std::ios::scientific, std::ios::floatfield);
      s.precision(12);

      const int numnode = distype_to_gmsh_num_node(distype);

      // coordinates
      s << "(";
      for (int inen = 0; inen < numnode; ++inen)
      {
        // print position 2D
        s << coord(0, inen) << ",";
        s << coord(1, inen) << ",";
        s << "0.000000000000e+00";
        if (inen < numnode - 1)
        {
          s << ",";
        }
      };
      s << ")";
      return;
    }

    //! take a double value and translate it to the value section of an Gmsh element entry
    //! -> gives a constant value over the element; used to plot the mesh without any physical field
    inline void scalar_to_stream(
        const double scalar,               ///< constant (arbitrary) value assigned to elements
        const Core::FE::CellType distype,  ///< element shape
        std::ostream& s)
    {
      s.setf(std::ios::scientific, std::ios::floatfield);
      s.precision(12);

      const int numnode = distype_to_gmsh_num_node(distype);

      // values
      s << "{";
      for (int i = 0; i < numnode; ++i)
      {
        s << scalar;
        if (i < numnode - 1)
        {
          s << ",";
        }
      };
      s << "};";
    }

    //! take a scalar value at a point and translate it into Gmsh postprocessing format
    void scalar_to_stream(const Core::LinAlg::Matrix<3, 1>& pointXYZ,  ///< coordinates of point
        const double scalarvalue,  ///< scalar value at this point
        std::ostream& s            ///< stream
    );

    //! take a scalar value at a point and translate it into Gmsh postprocessing format
    void vector_to_stream(const Core::LinAlg::Matrix<3, 1>& pointXYZ,  ///< coordinates of point
        const Core::LinAlg::Matrix<3, 1>& vectorvalue,                 ///< vector at this point
        std::ostream& s                                                ///< stream
    );

    //! take an array (numnode) and translate it to the scalar value section of an Gmsh element
    //! entry
    template <class V>
    inline void scalar_field_to_stream(const V& scalarfield,
        const Core::FE::CellType distype,  ///< element shape
        std::ostream& s)
    {
      s.setf(std::ios::scientific, std::ios::floatfield);
      s.precision(12);

      const int numnode = distype_to_gmsh_num_node(distype);

      // values
      s << "{";
      for (int inode = 0; inode < numnode; ++inode)
      {
        s << scalarfield(inode);
        if (inode < numnode - 1)
        {
          s << ",";
        }
      };
      s << "};";
    }

    //! take an array (3,numnode) and translate it to the vector value section of an Gmsh element
    //! entry
    template <class M>
    inline void vector_field_to_stream(const M& vectorfield,  ///< vector value array (3, numnode)
        const Core::FE::CellType distype,                     ///< element shape
        std::ostream& s)
    {
      s.setf(std::ios::scientific, std::ios::floatfield);
      s.precision(12);

      const int numnode = distype_to_gmsh_num_node(distype);
      const int nsd = 3;

      // values
      s << "{";
      for (int inode = 0; inode < numnode; ++inode)
      {
        for (int isd = 0; isd < nsd; ++isd)
        {
          s << std::scientific << vectorfield(isd, inode);
          if (isd < nsd - 1) s << ",";
        }
        if (inode < numnode - 1)
        {
          s << ",";
        }
      };
      s << "};";
    }


    //! take an array (3,numnode) and translate it to the vector value section of an Gmsh element
    //! entry
    template <class M>
    inline void vector_field_to_stream2_d(
        const M& vectorfield,              ///< vector value array (3, numnode)
        const Core::FE::CellType distype,  ///< element shape
        std::ostream& s)
    {
      s.setf(std::ios::scientific, std::ios::floatfield);
      s.precision(12);

      const int numnode = distype_to_gmsh_num_node(distype);

      // values
      s << "{";
      for (int inode = 0; inode < numnode; ++inode)
      {
        // 2D
        s << std::scientific << vectorfield(0, inode) << ",";
        s << std::scientific << vectorfield(1, inode) << ",";
        s << "0.000000000000e+00";
        if (inode < numnode - 1)
        {
          s << ",";
        }
      };
      s << "};";
    }

    //! take an array (9,numnode) and translate it to the tensor value section of an Gmsh element
    //! entry
    template <class M>
    inline void tensor_field_to_stream(const M& tensorfield,
        const Core::FE::CellType distype,  ///< element shape
        std::ostream& s)
    {
      s.setf(std::ios::scientific, std::ios::floatfield);
      s.precision(12);

      const int numnode = distype_to_gmsh_num_node(distype);
      const int numTensorEntries = 9;

      // values
      s << "{";
      for (int inode = 0; inode < numnode; ++inode)
      {
        for (int ientry = 0; ientry < numTensorEntries; ++ientry)
        {
          s << std::scientific << tensorfield(ientry, inode);
          if (ientry < numTensorEntries - 1) s << ",";
        }
        if (inode < numnode - 1)
        {
          s << ",";
        }
      };
      s << "};";
    }

    //! take an entire Core::Elements::Element and print it with constant scalar value at its
    //! initial position
    void element_at_initial_position_to_stream(
        const double scalar, const Core::Elements::Element* ele, std::ostream& s);

    //! take an entire Core::Elements::Element and print it with constant scalar value at its
    //! initial position
    std::string element_at_initial_position_to_string(
        const double scalar, const Core::Elements::Element* ele);

    //! take an entire Core::Elements::Element and print it with constant scalar value at the given
    //! position
    void element_at_current_position_to_stream(
        const double scalar,                 ///< scalar value for the entire element
        const Core::Elements::Element* ele,  ///< element to print
        const std::map<int, Core::LinAlg::Matrix<3, 1>>&
            currentelepositions,  ///< nodal position array
        std::ostream& s);

    //! take an entire Core::Elements::Element and print it with constant scalar value at the given
    //! position
    std::string element_at_current_position_to_string(
        const double scalar,                 ///< scalar value for the entire element
        const Core::Elements::Element* ele,  ///< element to print
        const std::map<int, Core::LinAlg::Matrix<3, 1>>&
            currentelepositions  ///< nodal position array
    );

    //! take an array (numnode) and translate it to the scalar value section of an Gmsh element
    //! entry
    template <class M>
    void cell_with_scalar_to_stream(const Core::FE::CellType distype,  ///< element shape
        const double scalar,  ///< scalar value for the entire element
        const M& xyze,        ///< position array (3, numnode)
        std::ostream& s)
    {
      s << "S";  // scalar field indicator
      s << distype_to_gmsh_element_header(distype);
      coordinates_to_stream(xyze, distype, s);
      scalar_to_stream(scalar, distype, s);
      s << "\n";
    }

    //! take an array (numnode) and translate it to the scalar value section of an Gmsh element
    //! entry
    template <class M>
    std::string cell_with_scalar_to_string(const Core::FE::CellType distype,  ///< element shape
        const double scalar,  ///< scalar value for the entire element
        const M& xyze         ///< position array (3, numnode)
    )
    {
      std::ostringstream s;
      cellWithScalarToStream(distype, scalar, xyze, s);
      return s.str();
    }

    //! take an value array (numnode) and a position array (3,numnode) and translate it to an
    //! element section of a Gmsh postprocessing file
    template <class V, class M>
    inline void cell_with_scalar_field_to_stream(
        const Core::FE::CellType distype,  ///< element shape
        const V& scalarfield,              ///< scalar field in the element
        const M& xyze,                     ///< position array (3, numnode)
        std::ostream& s)
    {
      s << "S";  // scalar field indicator
      s << distype_to_gmsh_element_header(distype);
      CoordinatesToStream(xyze, distype, s);
      ScalarFieldToStream(scalarfield, distype, s);
      s << "\n";
    }

    //! take an value array (numnode) and a position array (3,numnode) and translate it to an
    //! element section of a Gmsh postprocessing file
    template <class V, class M>
    inline std::string cell_with_scalar_field_to_string(
        const Core::FE::CellType distype,  ///< element shape
        const V& scalarfield,              ///< scalar field in the element
        const M& xyze                      ///< position array (3, numnode)
    )
    {
      std::ostringstream s;
      cellWithScalarFieldToStream(distype, scalarfield, xyze, s);
      return s.str();
    }

    //! take an value array (3,numnode) and a position array (3,numnode) and translate it to an
    //! element section of a Gmsh postprocessing file
    template <class M1, class M2>
    inline void cell_with_vector_field_to_stream(
        const Core::FE::CellType distype,  ///< element shape
        const M1& vectorfield,             ///< vector field in the element (3, numnode)
        const M2& xyze,                    ///< position array (3, numnode)
        std::ostream& s)
    {
      s << "V";  // vector field indicator
      s << distype_to_gmsh_element_header(distype);
      CoordinatesToStream(xyze, distype, s);
      VectorFieldToStream(vectorfield, distype, s);
      s << "\n";
    }

    //! take an value array (3,numnode) and a position array (3,numnode) and translate it to an
    //! element section of a Gmsh postprocessing file
    template <class M1, class M2>
    inline std::string cell_with_vector_field_to_string(
        const Core::FE::CellType distype,  ///< element shape
        const M1& vectorfield,             ///< vector field in the element (3, numnode)
        const M2& xyze                     ///< position array (3, numnode)
    )
    {
      std::ostringstream s;
      cellWithVectorFieldToStream(distype, vectorfield, xyze, s);
      return s.str();
    }

    //! take an value array (9,numnode) and a position array (3,numnode) and translate it to an
    //! element section of a Gmsh postprocessing file
    template <class M1, class M2>
    inline void cell_with_tensor_field_to_stream(
        const Core::FE::CellType distype,  ///< element shape
        const M1& tensorfield,             ///< tensor field in the element (9,numnode)
        const M2& xyze,                    ///< position array (3, numnode)
        std::ostream& s)
    {
      s << "T";  // tensor field indicator
      s << distype_to_gmsh_element_header(distype);
      CoordinatesToStream(xyze, distype, s);
      TensorFieldToStream(tensorfield, distype, s);
      s << "\n";
    }

    //! take an value array (9,numnode) and a position array (3,numnode) and translate it to an
    //! element section of a Gmsh postprocessing file
    template <class M1, class M2>
    inline std::string cell_with_tensor_field_to_string(
        const Core::FE::CellType distype,  ///< element shape
        const M1& tensorfield,             ///< tensor field in the element (9,numnode)
        const M2& xyze                     ///< position array (3, numnode)
    )
    {
      std::ostringstream s;
      cellWithTensorFieldToStream(distype, tensorfield, xyze, s);
      return s.str();
    }

    /// print a piece of text at a given position
    std::string text3d_to_string(const Core::LinAlg::Matrix<3, 1>& xyz,  ///< 3d Position of text
        const std::string& text,                                         ///< text to be printed
        const int fontsize                                               ///< font size
    );

    //! print discretization in initial configuration (t = 0)
    void dis_to_stream(const std::string& text, const double scalar,
        const Core::FE::Discretization& dis, std::ostream& s);

    //! print discretization in initial configuration (t = 0)
    std::string dis_to_string(
        const std::string& text, const double scalar, Core::FE::Discretization& dis);

    //! print discretization in current configuration (t > 0)
    void dis_to_stream(const std::string& text, const double scalar,
        const Core::FE::Discretization& dis,
        const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions, std::ostream& s);

    //! print discretization in current configuration (t > 0)
    std::string dis_to_string(const std::string& text, const double scalar,
        Core::FE::Discretization& dis,
        const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions);

    std::string get_new_file_name_and_delete_old_files(
        const std::string& filename_base, const std::string& file_name_prefix,
        const int& actstep,    ///< generate filename for this step
        const int& step_diff,  ///< how many steps are kept
        const bool screen_out,
        const int pid = 0  ///< my processor id
    );

    //! open Gmsh output file to add data
    std::string get_file_name(const std::string& filename_base, const std::string& file_name_prefix,
        const int& actstep,  ///< generate filename for this step
        const bool screen_out,
        const int pid = 0  ///< my processor id
    );
  }  // namespace Gmsh
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif

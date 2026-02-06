// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_HPP
#define FOUR_C_IO_HPP


#include "4C_config.hpp"

#include "4C_fem_general_shape_function_type.hpp"
#include "4C_io_control.hpp"
#include "4C_io_hdf.hpp"
#include "4C_io_legacy_types.hpp"
#include "4C_linalg_serialdensematrix.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace LinAlg
{
  class SerialDenseMatrix;
  class Map;
}  // namespace LinAlg
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

enum class ShapeFunctionType;

/// IO: input/output facility
namespace Core::IO
{
  class ControlFileEntry;
  class InputControl;
  class OutputControl;
  class HDFReader;

  // supported vector maps for the input/output routines
  enum VectorType
  {
    dofvector,
    nodevector,
    elementvector
  };

  /// copy type
  enum class CopyType : char
  {
    deep,  ///< copy everything.
    shape  ///< copy only the shape and create everything else new.
  };

  /*!
    \brief base class of 4C restart
   */
  class DiscretizationReader
  {
   public:
    /// construct reader for a given discretization to read a particular time step
    DiscretizationReader(
        Core::FE::Discretization& dis, std::shared_ptr<Core::IO::InputControl> input, int step);

    /// destructor
    ~DiscretizationReader() = default;

    /**
     * \brief read in and return vector
     *
     * This method is based on the method read_multi_vector(const std::string name). Also refer to
     * the documentation therein.
     *
     * \param[in] name  name of vector to read in
     * \return          source vector as read in
     */
    std::shared_ptr<Core::LinAlg::MultiVector<double>> read_vector(std::string name);

    /**
     * \brief read into given vector
     *
     * This method is based on the method
     * read_multi_vector(std::shared_ptr<Core::LinAlg::MultiVector<double>> vec, std::string name).
     * Also refer to the documentation therein.
     *
     * \param[in,out] vec   target vector to be filled
     * \param[in]     name  name of vector to read in
     */
    void read_vector(std::shared_ptr<Core::LinAlg::MultiVector<double>> vec, std::string name);
    void read_vector(std::shared_ptr<Core::LinAlg::Vector<double>> vec, std::string name);
    /**
     * \brief read in and return multi-vector
     *
     * Read in and return the vector without modifying the underlying map.
     *
     * \note This is a special method that has to be used with care! It may be that the underlying
     * map of the vector as read in does not match the current distribution of the underlying
     * discretization.
     *
     * \param[in] name  name of vector to read in
     * \return          source vector as read in
     */
    std::shared_ptr<Core::LinAlg::MultiVector<double>> read_multi_vector(const std::string name);

    /**
     * \brief read into given multi-vector
     *
     * In case the target vector to be filled is based on a different map than the source vector as
     * read in, the source vector is exported to the target vector.
     *
     * \note This is the default method to read into a given vector.
     *
     * \param[in,out] vec   target vector to be filled
     * \param[in]     name  name of vector to read in
     */
    void read_multi_vector(
        std::shared_ptr<Core::LinAlg::MultiVector<double>> vec, std::string name);

    /// read into given data object
    void read_map_data_of_char_vector(
        std::map<int, std::vector<char>>& mapdata, std::string name) const;

    /// check if an integer value exists in the control file
    int has_int(std::string name);

    /// read an integer value from the control file
    int read_int(std::string name);

    /// read a double value from the control file
    double read_double(std::string name);

    /// read into the discretization given in the constructor
    void read_mesh(int step);

    /// read nodes into the discretization given in the constructor
    void read_nodes_only(int step);

    /// Read the history data of elements and nodes from restart files
    void read_history_data(int step);

    /// read a non discretisation based vector of chars
    void read_char_vector(std::shared_ptr<std::vector<char>>& charvec, const std::string name);

    //! read a non discretisation based vector of doubles
    /*!
      This vector should have been written only by proc0.
      It is assumed that this is a 'small' vector which has to be present on all procs.
      It is read from proc0 again and then communicated to all present procs.
     */
    void read_redundant_double_vector(
        std::shared_ptr<std::vector<double>>& doublevec, const std::string name);

    //! read a non discretisation based vector of integers
    /*!
      This vector should have been written only by proc0.
      It is assumed that this is a 'small' vector which has to be present on all procs.
      It is read from proc0 again and then communicated to all present procs.
     */
    void read_redundant_int_vector(
        std::shared_ptr<std::vector<int>>& intvec, const std::string name);

   private:
    /// find control file entry to given time step
    void find_result_group(int step);

    /// access the MPI_Comm object
    [[nodiscard]] MPI_Comm get_comm() const;

    /// find control file entry to given time step
    void find_mesh_group(int step);

    /// open data files.
    std::shared_ptr<HDFReader> open_files(
        const char* filestring, const Core::IO::ControlFileEntry& result_step);

    //! my discretization
    Core::FE::Discretization& dis_;

    /// my input control file
    std::shared_ptr<Core::IO::InputControl> input_;

    /// control file entry of this step
    Core::IO::ControlFileEntry restart_step_;

    std::shared_ptr<HDFReader> reader_;
    std::shared_ptr<HDFReader> meshreader_;
  };


  /*!
    \brief The output context of a discretization

    Create an object of this class for every discretization those mesh
    and results you want to write. Data are written in parallel to
    processor local files. The first process additionally maintains the
    (plain text) control file that glues all result files together.
  */
  class DiscretizationWriter
  {
   public:
    /*!
     * @brief construct a discretization writer object to output the mesh and results
     *
     * @param[in] dis                   discretization
     * @param[in] output_control        output control file
     * @param[in] shape_function_type   shape function type of the underlying fe discretization
     */
    DiscretizationWriter(Core::FE::Discretization& dis, OutputControl& output_control,
        const Core::FE::ShapeFunctionType shape_function_type);

    DiscretizationWriter(const DiscretizationWriter& other) = delete;
    DiscretizationWriter operator=(const DiscretizationWriter& other) = delete;
    DiscretizationWriter(DiscretizationWriter&& other) = delete;
    DiscretizationWriter operator=(DiscretizationWriter&& other) = delete;


    /// cleanup, close hdf5 files
    ~DiscretizationWriter();

    //!@name Output methods
    //@{

    //! write result header to control file
    /*!
      You will want to call this once each time step _before_ the
      result data is written.
      For full restart capabilities, call write_mesh() before this method.
      \param step : current time step
      \param time : current absolute time
    */
    void new_step(const int step, const double time);

    //! write a result double to control file
    /*!
      There will be an entry in the current result step in the control
      file that points to this vector

      \param name : control file entry name
      \param value  : the result data value
    */
    void write_double(const std::string name, const double value);

    //! write a result integer to control file
    /*!
      There will be an entry in the current result step in the control
      file that points to this vector

      \param name : control file entry name
      \param value  : the result data value
    */
    void write_int(const std::string name, const int value);

    //! write a result vector
    /*!
      There will be an entry in the current result step in the control
      file that points to this vector

      \param name : control file entry name
      \param vec  : the result data vector
      \param vt   : vector type
    */
    void write_vector(const std::string name,
        std::shared_ptr<const Core::LinAlg::Vector<double>> vec, VectorType vt = dofvector);

    void write_multi_vector(const std::string name, const Core::LinAlg::MultiVector<double>& vec,
        VectorType vt = dofvector);



    //! write a result vector
    /*!
      There will be an entry in the current result step in the control
      file that points to this vector

      \param name : control file entry name
      \param vec  : the result data vector
      \param elemap: element map of discretization
      \param vt   : vector type
    */
    void write_vector(const std::string name, const std::vector<char>& vec,
        const Core::LinAlg::Map& elemap, VectorType vt = dofvector);

    //! write new mesh and result file next time it is possible
    void create_new_result_and_mesh_file()
    {
      resultfile_changed_ = -1;
      meshfile_changed_ = -1;
    };

    bool have_result_or_mesh_file_changed()
    {
      return resultfile_changed_ == -1 or meshfile_changed_ == -1;
    }

    //! write new "field" group to control file including node and element chunks
    void write_mesh(const int step, const double time);

    // for particle simulations: write only nodes in new "field" group to control file
    void write_only_nodes_in_new_field_group_to_control_file(
        const int step, const double time, const bool writerestart);

    //! write element data to file
    void write_element_data(bool writeowner);

    //! write node data to file
    void write_node_data(bool writeowner);

    //! write a non discretisation based vector of chars
    void write_char_data(const std::string name, std::vector<char>& charvec);

    //! write a non discretisation based vector of doubles
    /*!
      Write this vector only from proc0. It is assumed that this is a 'small' vector
      which is present on all procs. It shall be read from proc0 again and then
      communicated to all present procs.
     */
    void write_redundant_double_vector(const std::string name, std::vector<double>& doublevec);

    //! write a non discretisation based vector of integers
    /*!
      Write this vector only from proc0. It is assumed that this is a 'small' vector
      which is present on all procs. It shall be read from proc0 again and then
      communicated to all present procs.
     */
    void write_redundant_int_vector(const std::string name, std::vector<int>& vectorint);

    //@}

    //!@name Data management
    //@{

    /// clear all stored map data
    void clear_map_cache();

    //@}

    /// get output control
    [[nodiscard]] OutputControl& output() const { return output_; }

    /// access discretization
    [[nodiscard]] const Core::FE::Discretization& get_discretization() const;

   protected:
    /// access the MPI_Comm object
    [[nodiscard]] MPI_Comm get_comm() const;

    /*!
      \brief write a knotvector for a nurbs discretisation
    */
    void write_knotvector() const;

    //! open new mesh file
    void create_mesh_file(const int step);

    //! open new result file
    void create_result_file(const int step);

    //! my discretization
    Core::FE::Discretization& dis_;

    int step_;
    double time_;

    hid_t meshfile_;
    hid_t resultfile_;
    std::string meshfilename_;
    std::string resultfilename_;
    hid_t meshgroup_;
    hid_t resultgroup_;

    /// cache to remember maps we have already written
    std::map<const Epetra_BlockMapData*, std::string> mapcache_;

    /// dummy stack to really save the maps we cache
    std::vector<Core::LinAlg::Map> mapstack_;

    int resultfile_changed_;
    int meshfile_changed_;

    //! Control file object
    OutputControl& output_;

    //! do we want binary output
    bool binio_;

    Core::FE::ShapeFunctionType spatial_approx_;
  };

}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif

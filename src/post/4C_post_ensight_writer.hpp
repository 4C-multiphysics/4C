/*----------------------------------------------------------------------*/
/*! \file

 \brief contains basis class for the Ensight filter


 \level 1
 */


#ifndef FOUR_C_POST_ENSIGHT_WRITER_HPP
#define FOUR_C_POST_ENSIGHT_WRITER_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"  // due to DiscretizationType
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_post_writer_base.hpp"  // base class PostWriterBase

#include <Epetra_Map.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

class PostField;
class PostResult;


// forward declaration

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Nurbs
  {
    class NurbsDiscretization;
  }
}  // namespace Discret

//! defines how 2 line2 elements are constructed from a line3
const int sublinemap[2][2] = {{0, 2}, {2, 1}};


//! defines how 4 quad4 elements are constructed from a quad9
const int subquadmap[4][4] = {{0, 4, 8, 7}, {4, 1, 5, 8}, {8, 5, 2, 6}, {7, 8, 6, 3}};

//! defines how 8 hex8 elements are constructed from a hex27
//  ;-) its symetric for some reason
const int subhexmap[8][8] = {{0, 8, 20, 11, 12, 21, 26, 24}, {8, 1, 9, 20, 21, 13, 22, 26},
    {20, 9, 2, 10, 26, 22, 14, 23}, {11, 20, 10, 3, 24, 26, 23, 15},
    {12, 21, 26, 24, 4, 16, 25, 19}, {21, 13, 22, 26, 16, 5, 17, 25},
    {26, 22, 14, 23, 25, 17, 6, 18}, {24, 26, 23, 15, 19, 25, 18, 7}};

//! defines how 4 hex8 elements are constructed from a hex16
const int subhex16map[4][8] = {
    {0, 4, 7, 8, 12, 15}, {4, 1, 5, 12, 9, 13}, {5, 2, 6, 13, 10, 14}, {7, 6, 3, 14, 14, 12}};

//! defines how 4 hex8 elements are constructed from a hex18
const int subhex18map[4][8] = {{0, 4, 8, 7, 9, 13, 17, 16}, {4, 1, 5, 8, 13, 10, 14, 17},
    {8, 5, 2, 6, 17, 14, 11, 15}, {7, 8, 6, 3, 16, 17, 15, 12}};


//! basis class for the Ensight filter
class EnsightWriter : public PostWriterBase
{
 public:
  typedef std::map<Core::FE::CellType, int> NumElePerDisType;

  typedef std::map<Core::FE::CellType, std::vector<int>> EleGidPerDisType;

  //! constructor, does nothing (SetField must be called before usage)
  EnsightWriter(PostField* field, const std::string& name);

  //! write the whole thing
  void write_files(PostFilterBase& filter) override;

 protected:
  /*!
   \brief write all time steps of a result

   Write nodal results. The results are taken from a reconstructed
   Core::LinAlg::Vector<double>. In many cases this vector will contain just one
   variable (displacements) and thus is easy to write as a whole. At
   other times, however, there is more than one result (velocity,
   pressure) and we want to write just one part of it. So we have to
   specify which part.

   \author u.kue
   \date 03/07
   */
  void write_result(const std::string groupname,  ///< name of the result group in the control file
      const std::string name,                     ///< name of the result to be written
      const ResultType restype,     ///< type of the result to be written (nodal-/element-based)
      const int numdf,              ///< number of dofs per node to this result
      const int from = 0,           ///< start position of values in nodes
      const bool fillzeros = false  ///< zeros are filled to ensight file when no data is available
      ) override;


  /*!
   \brief write all time steps of a result in one time step

   Write nodal results. The results are taken from a reconstructed
   Core::LinAlg::Vector<double>. In many cases this vector will contain just one
   variable (displacements) and thus is easy to write as a whole. At
   other times, however, there is more than one result (velocity,
   pressure) and we want to write just one part of it. So we have to
   specify which part. Currently file continuation is not supported
   because Paraview is not able to load it due to some weird wild card
   issue.

    originally
   \author u.kue
   \date 03/07
   adapted
   \author ghamm
   \date 03/13
   */
  void write_result_one_time_step(PostResult& result,  ///< result group in the control file
      const std::string groupname,  ///< name of the result group in the control file
      const std::string name,       ///< name of the result to be written
      const ResultType restype,     ///< type of the result to be written (nodal-/element-based)
      const int numdf,              ///< number of dofs per node to this result
      bool firststep,               ///< bool whether this is the first time step
      bool laststep,                ///< bool whether this is the last time step
      const int from = 0            ///< start position of values in nodes
      ) override;

  /*!
   \brief write a particular variable to file

   Write results. Some variables need interaction with the post filter,
   e.g. structural stresses that do some element computations before output.
   To allow for a generic interface, the calling site needs to supply a
   class derived from SpecialFieldInterface that knows which function to call.

   \author kronbichler
   \date 04/14
   */
  void write_special_field(SpecialFieldInterface& special,
      PostResult& result,  ///< result group in the control file
      const ResultType restype, const std::string& groupname,
      const std::vector<std::string>& fieldnames, const std::string& outinfo) override;

  template <class T>
  void write(std::ofstream& os, T i) const
  {
    // only processor 0 does the writing !!
    if (myrank_ == 0) os.write(reinterpret_cast<const char*>(&i), sizeof(T));
  }

  void write(std::ofstream& os, const std::string s) const
  {
    // only processor 0 does the writing !!
    if (myrank_ == 0) write_string(os, s);
  }

  void write(std::ofstream& os, const char* s) const
  {
    // only processor 0 does the writing !!
    if (myrank_ == 0) write_string(os, s);
  }

  void write_string(std::ofstream& stream,  ///< filestream we are writing to
      const std::string str                 ///< string to be written to file
  ) const;
  void write_geo_file(const std::string& geofilename);
  void write_geo_file_one_time_step(std::ofstream& file,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string name);

  Teuchos::RCP<Epetra_Map> write_coordinates(
      std::ofstream& geofile,        ///< filestream for the geometry
      Core::FE::Discretization& dis  ///< discretization where the nodal positions are take from
  );

  /*! \brief Write the coordinates for a Polynomial discretization
      The coordinates of the vizualisation points (i.e. the corner
      nodes of elements displayed in paraview) are just the node
      coordinates of the nodes in the discretization.
    */
  void write_coordinates_for_polynomial_shapefunctions(
      std::ofstream& geofile, Core::FE::Discretization& dis, Teuchos::RCP<Epetra_Map>& proc0map);

  /*! \brief Write the coordinates for a Nurbs discretization
    The coordinates of the vizualisation points (i.e. the corner
    nodes of elements displayed in paraview) are not the control point
    coordinates of the nodes in the discretization but the points the
    knot values are mapped to.
  */
  void write_coordinates_for_nurbs_shapefunctions(
      std::ofstream& geofile, Core::FE::Discretization& dis, Teuchos::RCP<Epetra_Map>& proc0map);

  virtual void write_cells(std::ofstream& geofile,  ///< filestream for the geometry
      const Teuchos::RCP<Core::FE::Discretization>
          dis,  ///< discretization where the nodal positions are take from
      const Teuchos::RCP<Epetra_Map>&
          proc0map  ///< current proc0 node map, created by WriteCoordinatesPar
  ) const;

  /*! \brief Write the cells for a Nurbs discretization
    quadratic nurbs split one element in knot space into
    four(2d)/eight(3d) cells. The global numbering of the
    vizualisation points (i.e. the corner points of the
    cells) is computed from the local patch numbering and
    the patch offset.                              (gammi)

    \param Core::FE::CellType (i)          the nurbs discretisation type
    \param int                              (i)          global element id
    \param std::ofstream                    (used for o) direct print to file
    \param std::vector<int>                 (o)          remember node values for parallel IO
    \param Teuchos::RCP<Core::FE::Discretization> (i)          the discretisation holding
    knots etc \param Teuchos::RCP<Epetra_Map>          (i)          an allreduced nodemap

  */
  void write_nurbs_cell(const Core::FE::CellType distype, const int gid, std::ofstream& geofile,
      std::vector<int>& nodevector, Core::FE::Discretization& dis, Epetra_Map& proc0map) const;

  /*! \brief Quadratic nurbs split one nurbs27 element
    in knot space into eight(3d) cells. The global
    numbering of the vizualisation points (i.e. the corner
    points of the cells) are computed from the local patch
    numbering and the patch offset. This method appends
    the node connectivity for one hex8 cell to the vector
    of cell nodes                                 (gammi)

    \param Core::FE::CellType (i/o)        the vector to which the node
                                                         connectivity is appended to
    \param int                              (i)          0: left      1: right (which hex to
    generate) \param int                              (i)          0: front     1: rear  (which hex
    to generate) \param int                              (i)          0: bottom    1: top   (which
    hex to generate) \param std::vector<int>                 (i)          cartesian element ids in
    patch \param int                              (i)          number of visualisation points in u
    direction \param int                              (i)          number of visualisation points in
    v direction \param int                              (i)          number of patch

    \return void

  */
  void append_nurbs_sub_hex(std::vector<int>& cellnodes, const int& l, const int& m, const int& n,
      const std::vector<int>& ele_cart_id, const int& nvpu, const int& nvpv,
      const int& npatch) const
  {
    int twoid[3];
    twoid[0] = 2 * ele_cart_id[0];
    twoid[1] = 2 * ele_cart_id[1];
    twoid[2] = 2 * ele_cart_id[2];

    cellnodes.push_back((twoid[0] + l) + ((twoid[1] + m) + (twoid[2] + n) * nvpv) * nvpu);
    cellnodes.push_back((twoid[0] + 1 + l) + ((twoid[1] + m) + (twoid[2] + n) * nvpv) * nvpu);
    cellnodes.push_back((twoid[0] + 1 + l) + ((twoid[1] + 1 + m) + (twoid[2] + n) * nvpv) * nvpu);
    cellnodes.push_back((twoid[0] + l) + ((twoid[1] + 1 + m) + (twoid[2] + n) * nvpv) * nvpu);
    cellnodes.push_back((twoid[0] + l) + ((twoid[1] + m) + (twoid[2] + 1 + n) * nvpv) * nvpu);
    cellnodes.push_back((twoid[0] + 1 + l) + ((twoid[1] + m) + (twoid[2] + 1 + n) * nvpv) * nvpu);
    cellnodes.push_back(
        (twoid[0] + 1 + l) + ((twoid[1] + 1 + m) + (twoid[2] + 1 + n) * nvpv) * nvpu);
    cellnodes.push_back((twoid[0] + l) + ((twoid[1] + 1 + m) + (twoid[2] + 1 + n) * nvpv) * nvpu);

    return;
  };


  void write_node_connectivity_par(std::ofstream& geofile, Core::FE::Discretization& dis,
      const std::vector<int>& nodevector, Epetra_Map& proc0map) const;
  void write_dof_result_step(std::ofstream& file, PostResult& result,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string& groupname, const std::string& name, const int numdf, const int from,
      const bool fillzeros) const;


  /*! \brief Write the results for a NURBS discretisation
    (dof based).

    On input, result data for an n-dimensional computation
    is provided (from the result file)

    This element data is communicated in such a way that
    all elements have access to their (dof-accessible) data.
    Here we separate velocity/displacement and pressure
    output, since for velocity/displacement and pressure
    different dofs are required.

    Then, all elements are looped and function values are
    evaluated at visualisation points. This is the place
    where we need the dof data (again, different data for
    velocity/displacement and pressure output)

    The resulting vector is allreduced on proc0 and written.

                         .                              (gammi)

    \param std::ofstream                    (used for o) direct print to file
    \param int                              (i)          number of degrees of freedom
    \param Teuchos::RCP<Core::LinAlg::Vector<double>>      (i)          the result data read from
    the 4C output \param string                           (i)          name of the thing we are
    writing (velocity, pressure etc.) \param int                              (i)          potential
    offset in dof numbering

  */
  void write_dof_result_step_for_nurbs(std::ofstream& file, const int numdf,
      Core::LinAlg::Vector<double>& data, const std::string name, const int offset) const;

  //! perform interpolation of result data to visualization points.
  void interpolate_nurbs_result_to_viz_points(Epetra_MultiVector& idata, const int dim,
      const int npatch, const std::vector<int>& vpoff, const std::vector<int>& ele_cart_id,
      const Core::Elements::Element* actele, Core::FE::Nurbs::NurbsDiscretization* nurbsdis,
      const std::vector<Core::LinAlg::SerialDenseVector>& eleknots,
      const Core::LinAlg::SerialDenseVector& weights, const int numdf,
      const std::vector<double>& my_data) const;

  void write_nodal_result_step_for_nurbs(std::ofstream& file, const int numdf,
      Epetra_MultiVector& data, const std::string name, const int offset) const;

  void write_nodal_result_step(std::ofstream& file, PostResult& result,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string& groupname, const std::string& name, const int numdf);
  void write_nodal_result_step(std::ofstream& file, const Teuchos::RCP<Epetra_MultiVector>& data,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string& groupname, const std::string& name, const int numdf) override;
  void write_element_dof_result_step(std::ofstream& file, PostResult& result,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string& groupname, const std::string& name, const int numdof,
      const int from) const;
  void write_element_result_step(std::ofstream& file, PostResult& result,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string& groupname, const std::string& name, const int numdf, const int from);
  void write_element_result_step(std::ofstream& file, const Teuchos::RCP<Epetra_MultiVector>& data,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string& groupname, const std::string& name, const int numdf,
      const int from) override;
  void write_index_table(
      std::ofstream& file, const std::vector<std::ofstream::pos_type>& filepos) const;

  /*!
   * \brief create string for the VARIABLE section
   *        that corresponds to the current field
   */
  std::string get_variable_entry_for_case_file(
      int numdf,  ///< degrees of freedom per node for this field
      unsigned int fileset, const std::string name, const std::string filename,
      const int timeset) const;

  /*!
   * \brief create string for the VARIABLE section
   *        for all fields in the variablemap
   */
  std::string get_variable_section(std::map<std::string, std::vector<int>> filesetmap,
      std::map<std::string, int> variablenumdfmap,
      std::map<std::string, std::string> variablefilenamemap) const;

  /*!
   * \brief estimate, how many elements of each distype will be written
   * \return map between distype and number of elements to be written
   */
  NumElePerDisType get_num_ele_per_dis_type(Core::FE::Discretization& dis) const;

  std::string get_ensight_string(const Core::FE::CellType distype) const;

  /*!
   * \brief if files become to big, this routine switches to a new file
   */
  void file_switcher(std::ofstream& file,                   ///< filestream that is switched
      bool& multiple_files,                                 ///< ???
      std::map<std::string, std::vector<int>>& filesetmap,  ///< ???
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,  ///< ???
      const int stepsize,                                                          ///< ???
      const std::string name,                                                      ///< ???
      const std::string filename  ///< constant part of the filename
  ) const;

  int get_num_ele_output(const Core::FE::CellType distype, const int numele) const;

  //! number of subelements (equals 1 if no element splitting has to be done)
  int get_num_sub_ele(const Core::FE::CellType distype) const;

  /*!
   * \brief store, which elements are belonging to each present distype
   * \return map between distype and vector containing the global ids of th corresponding elements
   */
  EleGidPerDisType get_ele_gid_per_dis_type(
      Core::FE::Discretization& dis, NumElePerDisType numeleperdistype) const;

  //! create string for one TIME section in the case file
  std::string get_time_section_string(const int timeset,  ///< number of timeset to be written
      const std::vector<double>& times  ///< vector with time value for each time step
  ) const;

  //! create string for TIME section in the case file
  std::string get_time_section_string_from_timesets(
      const std::map<std::string, std::vector<double>>& timesetmap) const;

  //! create string for FILE section in the case file
  std::string get_file_section_string_from_filesets(const std::map<std::string, std::vector<int>>&
          filesetmap  ///< filesets when using multiple huge binary files
  ) const;

  bool nodeidgiven_;  ///< indicates whether 4C global node ids are written to geometry file.
                      ///< default value: true
  bool
      writecp_;  ///< NURBS-specific: defines if control point information should be written to file

  NumElePerDisType numElePerDisType_;  ///< number of elements per element discretization type
  EleGidPerDisType
      eleGidPerDisType_;  ///< global ids of corresponding elements per element discretization type

  Teuchos::RCP<Epetra_Map> proc0map_;  ///< allreduced node row map for proc 0, empty on other procs

  Teuchos::RCP<Epetra_Map> vispointmap_;  ///< map for all visualisation points

  std::map<std::string, std::vector<int>> filesetmap_;
  std::map<std::string, std::vector<double>> timesetmap_;

  std::map<std::string, int> variablenumdfmap_;
  std::map<std::string, std::string> variablefilenamemap_;
  std::map<std::string, std::string>
      variableresulttypemap_;  ///< STL map for storing the result-type per variable name

  std::map<std::string, int> timesetnumbermap_;
  std::map<std::string, int> filesetnumbermap_;

  std::map<std::string, std::vector<std::ofstream::pos_type>> resultfilepos_;

  //! maps a distype to the corresponding Ensight cell type
  std::map<Core::FE::CellType, std::string> distype2ensightstring_;

  //! maximum file size
  static constexpr unsigned FILE_SIZE_LIMIT_ = 0x7fffffff;  // 2GB
  // static constexpr unsigned FILE_SIZE_LIMIT_ = 1024*10; // 10kB ... useful for debugging ;-)
};

FOUR_C_NAMESPACE_CLOSE

#endif

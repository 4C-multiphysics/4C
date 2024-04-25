/*----------------------------------------------------------------------*/
/*! \file

\brief drt binary filter library


\level 2


The drt binary filter library contains classes to open and read binary
files. Based on these different filter applications can be written.

Note: The new filter design uses the discretization, node and element
classes. So postprocessing calculations are easily possible. Element
specific output operations can be done by the elements themselves.

*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_POST_COMMON_HPP
#define FOUR_C_POST_COMMON_HPP



#include "4C_config.hpp"

#include "4C_discretization_fem_general_shape_function_type.hpp"
#include "4C_global_data_enums.hpp"
#include "4C_io_hdf.hpp"
#include "4C_io_legacy_types.hpp"
#include "4C_linalg_serialdensematrix.hpp"

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_RCP.hpp>

#include <string>
#include <vector>

class Epetra_Comm;

FOUR_C_NAMESPACE_OPEN

class PostField;


namespace DRT
{
  class Discretization;
}

/*!
  \brief Foundation class of the filter

  Each filter needs one instance of PostProblem. This sets up the
  filter environment including MPI in a parallel setting. Some global variables are handled here as
  well.
*/
class PostProblem
{
 public:
  //@{ \name Constructors and destructors

  /*! \brief Setup filter and read command line
   *
   *  \param CLP (in)  : preconfigured command line processor
   *  \param argc (in) : number of command line arguments
   *  \param argv (in) : command line arguments
   */
  PostProblem(Teuchos::CommandLineProcessor& CLP, int argc, char** argv);

  //! destructor
  ~PostProblem();

  //@}

  //@{ \name Access methods

  //! returns the filter to be used for post-processing
  std::string filter() const { return filter_; }

  //! returns the basename of the control file
  std::string basename() const { return basename_; }

  /*! \brief Sets the basename of the control file to name
   *
   *  \param name [in] Name to be used for the control file
   */
  void set_basename(std::string name) { basename_ = name; }

  //! returns the basename of the output files
  std::string outname() const { return outname_; }

  //! returns the number of result groups that are stored in the control file
  int num_results() { return static_cast<int>(result_group_.size()); }

  //! returns a pointer to a vector of all result groups
  std::vector<MAP*>* result_groups() { return &result_group_; }

  //! returns the number of discretizations
  int num_discr() { return static_cast<int>(fields_.size()); }

  //! returns the number of dimensions the problem has
  int num_dim() const { return ndim_; }

  /*! \brief Returns a pointer to the num-th discretization
   *
   *  \param num [in] Index of discretization to be returned
   */
  PostField* get_discretization(const int num);

  int field_pos(const PostField* field) const;

  int num_output_procs() const;

  /*! \brief Get the maximum node id over the total simulation for a given fieldname
   *
   *  \param fieldname [in] Name of field that is queried
   */
  int get_max_nodeid(const std::string& fieldname);

  //! returns the first step in the range, that the user specified
  int start() const { return start_; }

  //! returns the last step in the range, that the user specified
  int end() const { return end_; }

  //! returns the step parameter the user specified
  int step() const { return step_; }

  Teuchos::RCP<Epetra_Comm> comm();

  //! returns the directory name in which the control files and data files are
  std::string input_dir() { return input_dir_; }

  //! returns type of the simulation
  GLOBAL::ProblemType Problemtype() { return problemtype_; }

  //! returns type of the spatial approximation (nurbs/polynomial)
  CORE::FE::ShapeFunctionType SpatialApproximationType() { return spatial_approx_; }

  //! returns stress type
  std::string stresstype() { return stresstype_; }

  //! returns strain type
  std::string straintype() { return straintype_; }

  //! returns optional quantity type
  std::string optquantitytype() { return optquantitytype_; }

  //! returns heatflux type
  std::string heatfluxtype() { return heatfluxtype_; }

  //! returns tempgrad type
  std::string tempgradtype() { return tempgradtype_; }

  //! returns struct_vel_acc_
  std::string struct_vel_acc() { return struct_vel_acc_; }

  //! returns struct_mat_disp_
  std::string struct_mat_disp() { return struct_mat_disp_; }

  //! returns struct_rot_
  std::string struct_rot() { return struct_rot_; }

  //! returns outputtype_
  std::string outputtype() { return outputtype_; }

  //! Is post-processing of mortar interfaces activated?
  bool DoMortarInterfaces() { return mortar_; }

  //@}

 private:
  /*! brief Initializes all the data a filter needs.
   *
   *  This function is called by the Constructor.
   */
  void setup_filter(std::string control_file_name,  ///< [in] Name of control file
      std::string output_name                       ///< [in] Name of output file
  );

  /*! \brief Reads the mesh files
   *
   *  Calls 'getfield()' for each 'field'-entry in the mesh file
   *  (currently it reads only the fields with step ==  0).
   *  This function is called by the Constructor.
   */
  void read_meshes();

  /*! \brief Creates and returns a PostField instance from a field MAP.
   *
   *  Keeps track of global field variable
   */
  PostField getfield(MAP* field_info);

  //! type of filter we want to apply
  std::string filter_;

  //! type of the problem we post process
  GLOBAL::ProblemType problemtype_;

  //! type of the spatial approximation of theproblem we post process
  CORE::FE::ShapeFunctionType spatial_approx_;

  //! number of dimensions
  int ndim_;

  //! control file base name (possibly including directory)
  std::string basename_;

  //! output file base name (possibly including directory)
  std::string outname_;

  /*! \brief Memory version of control file
   *
   *  If a restarted run is post processed this is a merged control
   *  file constructed from all control files available.
   */
  MAP control_table_;

  //! the meshes of our problem (fields are discretizations here!)
  std::vector<PostField> fields_;

  /*! \brief Start, stop and step numbers
   *
   *  We don't have to read each result. This is set by command line
   *  arguments.
   */
  int start_;
  int end_;
  int step_;

  //! Activate post-processing of mortar interfaces
  bool mortar_;

  //! stress output type optionally set by command line argument
  std::string stresstype_;

  //! strain output type optionally set by command line argument
  std::string straintype_;

  //! optional quantity output type optionally set by command line argument
  std::string optquantitytype_;

  //! heatflux output type optionally set by command line argument
  std::string heatfluxtype_;

  //! spatial temperature gradient output type optionally set by command line argument
  std::string tempgradtype_;

  //! structural velocity and acceleration output optionally set by command line argument
  std::string struct_vel_acc_;

  //! material displacement output optionally set by command line argument
  std::string struct_mat_disp_;

  //! structural rotation output optionally set by command line argument
  std::string struct_rot_;

  //! output to be written in binary of ascii format optionally set by commend line argument
  std::string outputtype_;

  /*! \brief MAP pointers into the control file map.
   *
   *  We don't own these pointers, so no reference counting.
   */
  std::vector<MAP*> result_group_;

  std::string input_dir_;
  Teuchos::RCP<Epetra_Comm> comm_;
};


/*!
  \brief The representation of one mesh.

  Problems with fixed mesh topologies will want to create one instance
  of this class for each mesh of the problem.

  Problems with changing meshes can be done, but will need some more
  work.

  \note The filter design uses the discretization, node and element
  classes. So postprocessing calculations are easily possible. Element
  specific output operations can be done by the elements themselves.

  \author m.kue
  \date 02/07
*/
class PostField
{
 public:
  // @{ \name Constructors and destructors

  //! Constructor
  PostField(Teuchos::RCP<DRT::Discretization> dis,  ///< [in] Discretization
      PostProblem* problem,                         ///< [in] Instance of PostProblem
      std::string field_name,                       ///< [in] Name of the field
      const int numnd,                              ///< [in] Number of nodes
      const int numele                              ///< [in] Number of elements
  );

  //@}

  // @{ \name Access methods

  //! returns the discretization of this field
  Teuchos::RCP<DRT::Discretization> discretization() { return dis_; }
  Teuchos::RCP<const DRT::Discretization> discretization() const { return dis_; }

  //! returns the number of output processors
  int num_output_procs() const { return num_output_procs_; }

  //! set the number of output processors
  void set_num_output_procs(const int num) { num_output_procs_ = num; }

  //! returns the number of global Dof-Ids
  int global_id_num() const;

  //! returns the PostProblem instance
  PostProblem* problem() { return problem_; }

  //! returns the name of the field
  std::string name() const { return field_name_; }

  //! returns the type of this field
  // FIELDTYP type() const { return type_; }

  //! returns the number of nodes
  int num_nodes() const { return numnd_; }

  //! set the number of nodes
  void set_num_nodes(int nodenumber) { numnd_ = nodenumber; }

  //! returns the number of elements
  int num_elements() const { return numele_; }

  int field_pos() const { return problem_->field_pos(this); }

  //@}

 private:
  //! Discretization of underlying field
  Teuchos::RCP<DRT::Discretization> dis_;

  //! PostProblem of underlying field
  PostProblem* problem_;

  //! Number of processors used for output
  int num_output_procs_;

  //! Name of field to be processed
  std::string field_name_;

  //! Number of nodes
  int numnd_;

  //! Number of Elements
  int numele_;
};


/*!
  \brief Representation of the results of one time step.

  Most filters will loop all time steps from the control file.

  \author m.kue
  \date 02/07
*/
class PostResult
{
 public:
  //@{ \name Constructors and destructors
  PostResult(PostField* field);
  ~PostResult();

  //@}

  //@{ \name Reader methods

  /*! \brief Get result times for a given fieldname
   *
   *  \param fieldname [in] Name of field to be queried
   *
   *  \returns vector with timesteps when the solution was written
   */
  std::vector<double> get_result_times(const std::string& fieldname);

  /*! \brief Get result times for a given fieldname and specific result
   *
   *  \param fieldname [in] Name of field to be queried
   *  \param groupname [in] ??
   *
   *  \returns vector with timesteps when the specific solution was written
   */
  std::vector<double> get_result_times(const std::string& fieldname, const std::string& groupname);

  /*! \brief Get result times and steps for a given fieldname
   *
   *  Fills vector with times and steps when the solution was written
   *
   *  \param fieldname [in] Name of field to be queried
   *  \param times [out] Time instances where results are available
   *  \param steps [out] Time step indices where results are available
   */
  void get_result_timesandsteps(
      const std::string& fieldname, std::vector<double>& times, std::vector<int>& steps);

  /*! \brief Goes to the next result block in the slice
   *
   *  Automatically opens new result files.
   *
   *  \returns 1 if a new result block has been found, otherwise returns 0
   */
  int next_result();

  /*! \brief Goes to the next result block in the slice containing the specific result vector >name<
   *
   *  Automatically opens new result files.
   *
   *  \param name [in] Name of result to be queried
   *
   *  \returns 1 if a new result block has been found, otherwise returns 0
   */
  int next_result(const std::string& name);

  /*! \brief Reads the data of the result vector 'name' from the current result block
   *
   *  \param name [in] result vector name inside the control file
   *
   *  \returns vector with results read
   */
  Teuchos::RCP<Epetra_Vector> read_result(const std::string name);

  /*! \brief Reads the data of the result vector 'name' from the current result block
   *
   * \param name [in] result vector name inside the control file
   * \param elemap [out] element map
   *
   * \returns vector with results read
   */
  Teuchos::RCP<std::map<int, Teuchos::RCP<CORE::LINALG::SerialDenseMatrix>>>
  read_result_serialdensematrix(const std::string name);

  /*! \brief Reads the data of the result vector 'name' from the current result block
   *
   *  \param name [in] result vector name inside the control file
   *
   *  \returns vector with results read
   */
  Teuchos::RCP<Epetra_MultiVector> read_multi_result(const std::string name);

  //@}

  // @{ \name Access methods

  //! returns the field in which this result exists
  PostField* field() { return field_; }

  //! returns the MAP all the result groups are read from
  MAP* group() { return group_; }

  //! returns time of this result
  double time() const;

  //! returns step number of this result
  int step() const;

  //@}

 private:
  //! closes all the currently open result files
  void close_result_files();

  /*! \brief Opens result files
   *
   *  The name is taken from the "result_file" entry in the block
   * 'field_info'
   *
   * \param field_info [in] result block map for control file
   */
  void open_result_files(MAP* field_info);

  //! Tell whether a given result group belongs to this result.
  int match_field_result(MAP* result_group) const;

  PostField* field_;
  int pos_;
  MAP* group_;
  IO::HDFReader file_;
};

FOUR_C_NAMESPACE_CLOSE

#endif

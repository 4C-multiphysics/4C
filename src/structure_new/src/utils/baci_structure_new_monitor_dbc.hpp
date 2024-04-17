/*----------------------------------------------------------------------------*/
/*! \file

\brief Monitor tagged Dirichlet boundary conditions

\level 3


*/
/*----------------------------------------------------------------------------*/

#ifndef FOUR_C_STRUCTURE_NEW_MONITOR_DBC_HPP
#define FOUR_C_STRUCTURE_NEW_MONITOR_DBC_HPP

#include "baci_config.hpp"

#include "baci_lib_condition.hpp"
#include "baci_linalg_fixedsizematrix.hpp"

#include <Epetra_Map.h>

FOUR_C_NAMESPACE_OPEN

namespace IO
{
  class DiscretizationWriter;
}
namespace DRT
{
  class Discretization;
}
namespace STR
{
  class Dbc;
  namespace TIMINT
  {
    class BaseDataGlobalState;
    class BaseDataIO;
  }  // namespace TIMINT

  /** \brief Monitor Dirichlet boundary conditions
   *
   *  This class can be used to monitor e.g. the reaction forces and the area
   *  change of a tagged Dirichlet condition during a simulation. To tag a Dirichlet
   *  condition just add the corresponding TAG, e.g. \"monitor_reaction\"
   *
   *  E 1 - NUMDOF 3 ONOFF 1 0 0 VAL 0.0 0.0 0.0 FUNCT 0 0 0 TAG monitor_reaction
   *
   *  If the TAG can be found for any Dirichlet condition the reaction force as
   *  well as the reference and current area will be stored in a text file
   *  located at
   *
   *  <OUTPUT_PATH>/<OUTPUT_FILE_NAME>_monitor_dbc/<ID>_monitor_dbc.data
   *
   *  If no tag is found nothing is happening.
   *
   *  \author hiermeier \date 01/18 */
  class MonitorDbc
  {
    const static unsigned DIM = 3;

    /// constants for the FILE output
    const static unsigned OF_WIDTH = 24;

    /// constants for the SCREEN output
    const static unsigned OS_WIDTH = 14;

   public:
    MonitorDbc() = default;

    /// initialize class members
    void Init(const Teuchos::RCP<STR::TIMINT::BaseDataIO>& io_ptr, DRT::Discretization& discret,
        STR::TIMINT::BaseDataGlobalState& gstate, STR::Dbc& dbc);

    /// setup new class members
    void Setup();

    /// monitor the tensile test results and write them to a text file
    void Execute(IO::DiscretizationWriter& writer);

   private:
    int GetUniqueId(int tagged_id, DRT::Condition::GeometryType gtype) const;

    void CreateReactionForceCondition(
        const DRT::Condition& tagged_cond, DRT::Discretization& discret) const;

    void GetTaggedCondition(std::vector<const DRT::Condition*>& tagged_conds,
        const std::string& cond_name, const std::string& tag_name,
        const DRT::Discretization& discret) const;

    void CreateReactionMaps(const DRT::Discretization& discret, const DRT::Condition& rcond,
        Teuchos::RCP<Epetra_Map>* react_maps) const;

    void ReadResultsPriorRestartStepAndWriteToFile(
        const std::vector<std::string>& full_restart_filepaths, int restart_step) const;

    void GetArea(double area_ref[], const DRT::Condition* rcond) const;

    double GetReactionForce(
        CORE::LINALG::Matrix<3, 1>& rforce_xyz, const Teuchos::RCP<Epetra_Map>* react_maps) const;

    double GetReactionMoment(CORE::LINALG::Matrix<3, 1>& rmoment_xyz,
        const Teuchos::RCP<Epetra_Map>* react_maps, const DRT::Condition* rcond) const;

    std::vector<std::string> CreateFilePaths(
        const std::vector<Teuchos::RCP<DRT::Condition>>& rconds, const std::string& full_dirpath,
        const std::string& filename_only_prefix, const std::string& file_type) const;

    void ClearFilesAndWriteHeader(const std::vector<Teuchos::RCP<DRT::Condition>>& rconds,
        std::vector<std::string>& full_filepaths, bool write_condition_header);

    void WriteConditionHeader(
        std::ostream& os, const int col_width, const DRT::Condition* cond = nullptr) const;

    void WriteColumnHeader(std::ostream& os, const int col_width) const;

    void WriteResultsToFile(const std::string& full_filepath,
        const CORE::LINALG::Matrix<DIM, 1>& rforce, const CORE::LINALG::Matrix<DIM, 1>& rmoment,
        const double& area_ref, const double& area_curr) const;

    void WriteResultsToScreen(const Teuchos::RCP<DRT::Condition>& rcond_ptr,
        const CORE::LINALG::Matrix<DIM, 1>& rforce, const CORE::LINALG::Matrix<DIM, 1>& rmoment,
        const double& area_ref, const double& area_curr) const;

    void WriteResults(std::ostream& os, const int col_width, const int precision,
        const unsigned step, const double time, const CORE::LINALG::Matrix<DIM, 1>& rforce,
        const CORE::LINALG::Matrix<DIM, 1>& rmoment, const double& area_ref,
        const double& area_cur) const;

    inline const Epetra_Comm& Comm() const;

    inline void ThrowIfNotInit() const { dsassert(isinit_, "Call Init() first!"); }

    inline void ThrowIfNotSetup() const { dsassert(issetup_, "Call Setup() first!"); }

   private:
    DRT::Discretization* discret_ptr_ = nullptr;
    STR::TIMINT::BaseDataGlobalState* gstate_ptr_ = nullptr;
    STR::Dbc* dbc_ptr_ = nullptr;

    std::vector<std::string> full_filepaths_ = std::vector<std::string>();

    /// extract the dofs of the reaction forces which shall be monitored
    std::map<int, std::vector<Teuchos::RCP<Epetra_Map>>> react_maps_;
    unsigned of_precision_ = -1;
    unsigned os_precision_ = -1;

    bool isempty_ = true;
    bool isinit_ = false;
    bool issetup_ = false;
  };

}  // namespace STR

FOUR_C_NAMESPACE_CLOSE

#endif

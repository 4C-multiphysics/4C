/*----------------------------------------------------------------------*/
/*! \file
\brief  Coupling Communication Manager automatically creates all required coupling object to
transform matrixes, vectors, ...

\level 2


*----------------------------------------------------------------------*/

#include "4C_config.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Epetra_Comm.h>

#define COUP_MANAGER_DEBUG_OUT

#ifndef FOUR_C_FSI_XFEM_COUPLING_COMM_MANAGER_HPP
#define FOUR_C_FSI_XFEM_COUPLING_COMM_MANAGER_HPP

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class MultiMapExtractor;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Coupling::Adapter
{
  class Coupling;
  class CouplingConverter;
}  // namespace Coupling::Adapter

namespace XFEM
{
  class CouplingCommManager
  {
   public:
    enum TransferType  // partial is on the interfacee //full is the whole discretisation //global
                       // is for all discretizations
    {
      full_to_full,
      full_to_partial,
      partial_to_full,
      partial_to_partial,
      partial_to_global,
      full_to_global,
    };

    enum MatrixTransferType
    {
      row,
      col,
      row_and_col
    };

    //! constructor
    explicit CouplingCommManager(std::map<int, Teuchos::RCP<const Core::FE::Discretization>> dis,
        std::string cond_name, int startdim = 0, int enddim = 3);

    //! constructor
    explicit CouplingCommManager(Teuchos::RCP<const Core::FE::Discretization> dis0,
        std::string cond_name, int startdim = 0, int enddim = 3);

    //! constructor
    explicit CouplingCommManager(Teuchos::RCP<const Core::FE::Discretization> dis0,
        Teuchos::RCP<const Core::FE::Discretization> dis1, std::string cond_name, int startdim = 0,
        int enddim = 3);

    //! virtual destructor to support polymorph destruction
    virtual ~CouplingCommManager() = default;

    //! Insert a Vector A into vector B (choose type of transfer, add or scaling) - Version vor
    //! RCP<const Core::LinAlg::Vector> vecA
    void insert_vector(const int idxA, Teuchos::RCP<const Core::LinAlg::Vector> vecA,
        const int idxB, Teuchos::RCP<Core::LinAlg::Vector> vecB,
        const CouplingCommManager::TransferType ttype, bool add = false, double scale = 1.0);

    //! Insert a Vector A into vector B (choose type of transfer, add or scaling) - Version vor
    //! RCP<Core::LinAlg::Vector> vecA
    void insert_vector(const int idxA, Teuchos::RCP<Core::LinAlg::Vector> vecA, const int idxB,
        Teuchos::RCP<Core::LinAlg::Vector> vecB, const CouplingCommManager::TransferType ttype,
        bool add = false, double scale = 1.0)
    {
      insert_vector(idxA, Teuchos::rcp_static_cast<const Core::LinAlg::Vector>(vecA), idxB, vecB,
          ttype, add, scale);
    }

    //! Insert a Matrix A (from discretization A) into Matrix B (from discretization B) (choose type
    //! of transfer, add or scaling)
    bool insert_matrix(
        int transform_id,  // Unique Id to be set for this transformation object (to be save use
                           // different one, for different matrix transformation)
        int idxA, const Core::LinAlg::SparseMatrix& matA, int idxB,
        Core::LinAlg::SparseMatrix& matB, const CouplingCommManager::MatrixTransferType mttype,
        double scale = 1.0, bool exactmatch = true, bool addmatrix = false);

    Teuchos::RCP<Core::LinAlg::MultiMapExtractor> get_map_extractor(int idx);

   protected:
    Teuchos::RCP<Coupling::Adapter::CouplingConverter> get_coupling_converter(int idxA, int idxB);

    Teuchos::RCP<Coupling::Adapter::Coupling> get_coupling(int idxA, int idxB);

    Teuchos::RCP<Core::LinAlg::MultiMapExtractor> get_full_map_extractor()
    {
      return fullextractor_;
    }

    Teuchos::RCP<Coupling::Adapter::MatrixLogicalSplitAndTransform> get_transform(int transform_id);

    void debug_out(
        std::string str1, std::string str2 = "", std::string str3 = "", std::string str4 = "");

   private:
    void setup(std::map<int, Teuchos::RCP<const Core::FE::Discretization>> dis);

    void setup_multi_map_extractors(
        std::map<int, Teuchos::RCP<const Core::FE::Discretization>> dis);

    void setup_full_map_extractors(std::map<int, Teuchos::RCP<const Core::FE::Discretization>> dis);

    void setup_couplings(std::map<int, Teuchos::RCP<const Core::FE::Discretization>> dis);

    void setup_full_couplings(std::map<int, Teuchos::RCP<const Core::FE::Discretization>> dis);

    void setup_full_extractor(std::map<int, Teuchos::RCP<const Core::FE::Discretization>> dis);

    std::string cond_name_;
    int startdim_;
    int enddim_;

    // All MultiMapExtractors
    std::map<int, Teuchos::RCP<Core::LinAlg::MultiMapExtractor>> mme_;

    // Couling Objects will just be initizalized in case we have more discretizations!
    std::map<std::pair<int, int>, Teuchos::RCP<Coupling::Adapter::Coupling>> coup_;

    // Transformation Objects will just be initizalized in case we use matrix transformations!
    std::map<int, Teuchos::RCP<Coupling::Adapter::MatrixLogicalSplitAndTransform>> transform_;

    Teuchos::RCP<Core::LinAlg::MultiMapExtractor> fullextractor_;
  };
}  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif

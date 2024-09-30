/*----------------------------------------------------------------------*/
/*! \file

\brief A class providing coupling capabilities based on non-linear
       mortar methods

\level 1


*----------------------------------------------------------------------*/

#ifndef FOUR_C_ADAPTER_COUPLING_NONLIN_MORTAR_HPP
#define FOUR_C_ADAPTER_COUPLING_NONLIN_MORTAR_HPP

/*---------------------------------------------------------------------*
 | headers                                                 farah 10/14 |
 *---------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_coupling_adapter_mortar.hpp"
#include "4C_fem_condition.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_Comm.h>
#include <Epetra_Map.h>
#include <Teuchos_ParameterListAcceptorDefaultBase.hpp>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------*
 | forward declarations                                    farah 10/14 |
 *---------------------------------------------------------------------*/
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Core::Nodes
{
  class Node;
}

namespace CONTACT
{
  class Interface;
}

namespace Core::LinAlg
{
  class SparseMatrix;
}

namespace Adapter
{
  class CouplingNonLinMortar : public Coupling::Adapter::CouplingMortar
  {
   public:
    /**
     * Construct nonlinear coupling with basic parameters. The remaining information is passed in
     * setup().
     */
    CouplingNonLinMortar(int spatial_dimension, Teuchos::ParameterList mortar_coupling_params,
        Teuchos::ParameterList contact_dynamic_params,
        Core::FE::ShapeFunctionType shape_function_type);

    /*!
    \brief initialize routine

    */
    virtual void setup(Teuchos::RCP<Core::FE::Discretization> masterdis,
        Teuchos::RCP<Core::FE::Discretization> slavedis, std::vector<int> coupleddof,
        const std::string& couplingcond);

    virtual void setup_spring_dashpot(Teuchos::RCP<Core::FE::Discretization> masterdis,
        Teuchos::RCP<Core::FE::Discretization> slavedis,
        Teuchos::RCP<Core::Conditions::Condition> spring, const int coupling_id,
        const Epetra_Comm& comm);

    virtual void integrate_lin_d(const std::string& statename,
        const Teuchos::RCP<Core::LinAlg::Vector> vec,
        const Teuchos::RCP<Core::LinAlg::Vector> veclm);

    virtual void integrate_lin_dm(const std::string& statename,
        const Teuchos::RCP<Core::LinAlg::Vector> vec,
        const Teuchos::RCP<Core::LinAlg::Vector> veclm);

    virtual void integrate_all(const std::string& statename,
        const Teuchos::RCP<Core::LinAlg::Vector> vec,
        const Teuchos::RCP<Core::LinAlg::Vector> veclm);

    virtual void evaluate_sliding(const std::string& statename,
        const Teuchos::RCP<Core::LinAlg::Vector> vec,
        const Teuchos::RCP<Core::LinAlg::Vector> veclm);

    virtual void print_interface(std::ostream& os);

    virtual Teuchos::RCP<Core::LinAlg::SparseMatrix> d_lin_matrix()
    {
      if (DLin_ == Teuchos::null) FOUR_C_THROW("ERROR: DLin Matrix is null pointer!");
      return DLin_;
    };

    virtual Teuchos::RCP<Core::LinAlg::SparseMatrix> m_lin_matrix()
    {
      if (MLin_ == Teuchos::null) FOUR_C_THROW("ERROR: MLin Matrix is null pointer!");
      return MLin_;
    };

    virtual Teuchos::RCP<Core::LinAlg::SparseMatrix> h_matrix()
    {
      if (H_ == Teuchos::null) FOUR_C_THROW("ERROR: H Matrix is null pointer!");
      return H_;
    };

    virtual Teuchos::RCP<Core::LinAlg::SparseMatrix> t_matrix()
    {
      if (T_ == Teuchos::null) FOUR_C_THROW("ERROR: T Matrix is null pointer!");
      return T_;
    };

    virtual Teuchos::RCP<Core::LinAlg::SparseMatrix> n_matrix()
    {
      if (N_ == Teuchos::null) FOUR_C_THROW("ERROR: N Matrix is null pointer!");
      return N_;
    };

    // create projection operator Dinv*M
    void create_p() override;

    virtual Teuchos::RCP<Core::LinAlg::Vector> gap()
    {
      if (gap_ == Teuchos::null) FOUR_C_THROW("ERROR: gap vector is null pointer!");
      return gap_;
    };

    /// the mortar interface itself
    Teuchos::RCP<CONTACT::Interface> interface() const { return interface_; }

   protected:
    /*!
    \brief Read Mortar Condition

    */
    virtual void read_mortar_condition(Teuchos::RCP<Core::FE::Discretization> masterdis,
        Teuchos::RCP<Core::FE::Discretization> slavedis, std::vector<int> coupleddof,
        const std::string& couplingcond, Teuchos::ParameterList& input,
        std::map<int, Core::Nodes::Node*>& mastergnodes,
        std::map<int, Core::Nodes::Node*>& slavegnodes,
        std::map<int, Teuchos::RCP<Core::Elements::Element>>& masterelements,
        std::map<int, Teuchos::RCP<Core::Elements::Element>>& slaveelements);

    /*!
    \brief Add Mortar Nodes

    */
    virtual void add_mortar_nodes(Teuchos::RCP<Core::FE::Discretization> masterdis,
        Teuchos::RCP<Core::FE::Discretization> slavedis, std::vector<int> coupleddof,
        Teuchos::ParameterList& input, std::map<int, Core::Nodes::Node*>& mastergnodes,
        std::map<int, Core::Nodes::Node*>& slavegnodes,
        std::map<int, Teuchos::RCP<Core::Elements::Element>>& masterelements,
        std::map<int, Teuchos::RCP<Core::Elements::Element>>& slaveelements,
        Teuchos::RCP<CONTACT::Interface>& interface, int numcoupleddof);

    /*!
    \brief Add Mortar Elements

    */
    virtual void add_mortar_elements(Teuchos::RCP<Core::FE::Discretization> masterdis,
        Teuchos::RCP<Core::FE::Discretization> slavedis, Teuchos::ParameterList& input,
        std::map<int, Teuchos::RCP<Core::Elements::Element>>& masterelements,
        std::map<int, Teuchos::RCP<Core::Elements::Element>>& slaveelements,
        Teuchos::RCP<CONTACT::Interface>& interface, int numcoupleddof);

    /*!
    \brief complete interface, store as internal variable
           store maps as internal variable and do parallel redist.

    */
    virtual void complete_interface(Teuchos::RCP<Core::FE::Discretization> masterdis,
        Teuchos::RCP<CONTACT::Interface>& interface);

    /*!
    \brief initialize matrices (interla variables)

    */
    virtual void init_matrices();

    /*!
    \brief create strategy object if required

    */
    virtual void create_strategy(Teuchos::RCP<Core::FE::Discretization> masterdis,
        Teuchos::RCP<Core::FE::Discretization> slavedis, Teuchos::ParameterList& input,
        int numcoupleddof);

    /*!
    \brief transform back to initial parallel distribution

    */
    virtual void matrix_row_col_transform();

    /// check setup call
    const bool& is_setup() const { return issetup_; };

    /// check init and setup call
    void check_setup() const override
    {
      if (!is_setup()) FOUR_C_THROW("ERROR: Call setup() first!");
    }

   protected:
    bool issetup_;                    ///< check for setup
    Teuchos::RCP<Epetra_Comm> comm_;  ///< communicator
    int myrank_;                      ///< my proc id

    Teuchos::RCP<Epetra_Map> slavenoderowmap_;  ///< map of slave row nodes (after parallel redist.)
    Teuchos::RCP<Epetra_Map>
        pslavenoderowmap_;                  ///< map of slave row nodes (before parallel redist.)
    Teuchos::RCP<Epetra_Map> smdofrowmap_;  ///< map of sm merged row dofs (after parallel redist.)
    Teuchos::RCP<Epetra_Map>
        psmdofrowmap_;  ///< map of sm merged row dofs (before parallel redist.)

    Teuchos::RCP<Core::LinAlg::SparseMatrix> DLin_;  ///< linearization of D matrix
    Teuchos::RCP<Core::LinAlg::SparseMatrix> MLin_;  ///< linearization of M matrix

    Teuchos::RCP<Core::LinAlg::SparseMatrix>
        H_;  ///< Matrix containing the tangent derivatives with respect to slave dofs
    Teuchos::RCP<Core::LinAlg::SparseMatrix>
        T_;  ///< Matrix containing the tangent vectors of the slave nodes
    Teuchos::RCP<Core::LinAlg::SparseMatrix>
        N_;                                   ///< Matrix containing the (weighted) gap derivatives
                                              ///< with respect to master and slave dofs
    Teuchos::RCP<Core::LinAlg::Vector> gap_;  ///< gap vector

    Teuchos::RCP<CONTACT::Interface> interface_;  ///< interface
  };
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_DBC_HPP
#define FOUR_C_STRUCTURE_NEW_DBC_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_abstract_prepostoperator.hpp"

#include <Teuchos_RCPDecl.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class SparseOperator;
  class SparseMatrix;
  class MapExtractor;
  class Map;
}  // namespace Core::LinAlg

namespace Core::Conditions
{
  class LocsysManager;
}

namespace Solid
{
  namespace TimeInt
  {
    class Base;
    class BaseDataGlobalState;
  }  // namespace TimeInt

  /*! \brief Object to handle Dirichlet boundary conditions for solid dynamics
   *
   *  This class provides capabilities to
   *  - handle Dirichlet boundary conditions
   *  - rotate the coordinate system (referred to as 'locSys')
   */
  class Dbc
  {
   public:
    //! Constructor
    Dbc();

    //! Destructor
    virtual ~Dbc() = default;

    //! Initialize class variables
    virtual void init(const std::shared_ptr<Core::FE::Discretization>& discret,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& freact,
        const std::shared_ptr<const Solid::TimeInt::Base>& timint_ptr);

    //! Setup class variables
    virtual void setup();

    /*! \brief Apply the DBC to system of equations
     *
     *  \note Stay in the local coordinate system and do not rotate back (if locSys is defined).*/
    void apply_dirichlet_to_local_system(
        Core::LinAlg::SparseOperator& A, Core::LinAlg::Vector<double>& b) const;

    /*! \brief Apply the DBC to a vector
     *
     *  \note Stay in the global coordinate system (Rotation: global-->local-->global).*/
    void apply_dirichlet_to_vector(Core::LinAlg::Vector<double>& vec) const;

    /*! \brief Apply the DBC to the rhs vector and calculate and save the reaction forces
     *
     *  \note Stay in the global coordinate system (Rotation: global-->local-->global).*/
    void apply_dirichlet_to_rhs(Core::LinAlg::Vector<double>& b) const;

    //! Update the locsys manager
    void update_loc_sys_manager();

    //! Calculate the dirichlet increment of the current (time) step
    std::shared_ptr<Core::LinAlg::Vector<double>> get_dirichlet_increment();

    /*! \brief Evaluate and apply the DBC
     *
     * \note Stay in the global coordinate system (Rotation: global-->local-->global).*/
    virtual void apply_dirichlet_bc(const double& time,
        std::shared_ptr<Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<Core::LinAlg::Vector<double>> vel,
        std::shared_ptr<Core::LinAlg::Vector<double>> acc, bool recreatemap) const;

    /*! \brief Insert non-dbc dof values of source vector into the non-dbc dofs of target vector
     *
     *  \param[in] source_ptr Source vector with values to be inserted
     *  \param[in/out] target_ptr Target vector where values shall be inserted into
     */
    void insert_vector_in_non_dbc_dofs(
        std::shared_ptr<const Core::LinAlg::Vector<double>> source_ptr,
        std::shared_ptr<Core::LinAlg::Vector<double>> target_ptr) const;

    //! @name Access functions
    //!@{

    //! Get the Dirichlet Boundary Condition map extractor
    std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() const;

    //! Get a pointer to the local system manager
    std::shared_ptr<Core::Conditions::LocsysManager> loc_sys_manager_ptr();

    //! Get the zeros vector
    const Core::LinAlg::Vector<double>& get_zeros() const;
    std::shared_ptr<const Core::LinAlg::Vector<double>> get_zeros_ptr() const;

    //!@}

    //! Allows to expand dbc map with provided maptoadd
    void add_dirich_dofs(const std::shared_ptr<const Core::LinAlg::Map> maptoadd);

    //! Allows to contract dbc map with provided maptoremove
    void remove_dirich_dofs(const std::shared_ptr<const Core::LinAlg::Map> maptoremove);

    /*! \brief Rotate the system matrix from a global to a local coordinate system
     *
     *  \pre #locsysman_ has to be defined.
     *
     *  \note Works only for Core::LinAlg::SparseMatrices.
     **/
    bool rotate_global_to_local(Core::LinAlg::SparseOperator& A) const;

    /*! \brief Rotate the rhs vector from the global to the local coordinate system
     *
     *  \pre #locsysman_ has to be defined.
     *
     *  \param[in] v Vector to be rotated
     */
    bool rotate_global_to_local(Core::LinAlg::Vector<double>& v) const;

    /*! \brief Rotate the rhs vector from the global to the local coordinate system
     *
     *  \pre #locsysman_ has to be defined.
     *
     *  \param[in] v Vector to be rotated
     *  \param[in] offset ??
     */
    bool rotate_global_to_local(Core::LinAlg::Vector<double>& v, bool offset) const;

    /*! \brief Rotate a vector from the local to the global coordinate system
     *
     *  \pre #locsysman_ has to be defined.
     *
     *  \param[in] v Vector to be rotated
     */
    bool rotate_local_to_global(Core::LinAlg::Vector<double>& v) const;

    /*! \brief Rotate a vector from the local to the global coordinate system
     *
     *  \pre #locsysman_ has to be defined.
     *
     *  \param[in] v Vector to be rotated
     *  \param[in] offset ??
     */
    bool rotate_local_to_global(Core::LinAlg::Vector<double>& v, bool offset) const;

   protected:
    //! Returns the initialization status
    const bool& is_init() const { return isinit_; };

    //! Returns the setup status
    const bool& is_setup() const { return issetup_; };

    //! Checks the initialization status
    void check_init() const;

    //! Checks the initialization and setup status
    void check_init_setup() const;

    //! Get discretization pointer
    std::shared_ptr<Core::FE::Discretization> discret_ptr();
    std::shared_ptr<const Core::FE::Discretization> discret_ptr() const;

    //! Access the reaction force
    Core::LinAlg::Vector<double>& freact() const;

    //! Get the locsys transformation matrix
    std::shared_ptr<const Core::LinAlg::SparseMatrix> get_loc_sys_trafo() const;

    //! Get the global state
    const Solid::TimeInt::BaseDataGlobalState& g_state() const;

    //! Has #locsysman_ be defined?
    bool is_loc_sys() const { return islocsys_; };

    /*! \brief Extract the reaction forces
     *
     *  \param b ??
     */
    void extract_freact(Core::LinAlg::Vector<double>& b) const;

    /*! Apply the DBC to the right hand side in the local coordinate system and
     *  do not rotate it back to the global coordinate system. */
    void apply_dirichlet_to_local_rhs(Core::LinAlg::Vector<double>& b) const;

    /*! \brief Apply the DBC to the Jacobian in the local coordinate system
     *
     *  \note This does not rotate the result back to the global coordinate system.
     *
     *  \param[in/out] A Jacobian matrix
     */
    void apply_dirichlet_to_local_jacobian(Core::LinAlg::SparseOperator& A) const;

   protected:
    //! Flag indicating the initialization status.
    bool isinit_;

    //! Flag indicating the setup status.
    bool issetup_;

    //! Flag indicating if a #locsysman_ was defined.
    bool islocsys_;

    //! discretization pointer
    std::shared_ptr<Core::FE::Discretization> discret_ptr_;

    //! pointer to the overlying time integrator (read-only)
    std::shared_ptr<const Solid::TimeInt::Base> timint_ptr_;

    //! Pointer to the local coordinate system manager
    std::shared_ptr<Core::Conditions::LocsysManager> locsysman_ptr_;

    //! Some vector with system size and filled with zeros.
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_ptr_;

    //! Dirichlet boundary condition map extractor.
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmap_ptr_;

   private:
    //! Reaction force
    Core::LinAlg::Vector<double>* freact_ptr_;

  };  // namespace Solid
}  // namespace Solid

namespace NOX
{
  namespace Nln
  {
    namespace LinSystem
    {
      namespace PrePostOp
      {
        /*! \brief PrePostOperator class to modify the linear system before the linear system is
         * going to be solved.
         *
         * We use this pre/post operator to apply the DBC on the linear system of equations before
         * the linear system is going to be solved. This gives us the opportunity to rotate the
         * matrix only once, if locSys is defined and to apply possible modifications to the linear
         * system at different places without the need to re-apply the DBC (see PTC for an example).
         *
         * */
        class Dbc : public NOX::Nln::Abstract::PrePostOperator
        {
         public:
          //! constructor
          Dbc(const Teuchos::RCP<const Solid::Dbc>& dbc_ptr);

          //! \brief Apply the DBC and rotate the system of equations if necessary (derived)
          void run_pre_apply_jacobian_inverse(::NOX::Abstract::Vector& rhs,
              Core::LinAlg::SparseOperator& jac, const NOX::Nln::LinearSystem& linsys) override;

         private:
          //! pointer to the underlying class, which provides the whole functionality
          Teuchos::RCP<const Solid::Dbc> dbc_ptr_;

        };  // class Dbc (pre/post operator)
      }  // namespace PrePostOp
    }  // namespace LinSystem
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif

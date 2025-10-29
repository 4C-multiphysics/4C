// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_WALL_DATASTATE_HPP
#define FOUR_C_PARTICLE_WALL_DATASTATE_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief wall data state container
   *
   */
  class WallDataState final
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] params particle simulation parameter list
     */
    explicit WallDataState(const Teuchos::ParameterList& params);

    /*!
     * \brief init wall data state container
     *
     *
     * \param[in] walldiscretization wall discretization
     */
    void init(const std::shared_ptr<Core::FE::Discretization> walldiscretization);

    /*!
     * \brief setup wall data state container
     *
     */
    void setup();

    /*!
     * \brief check for correct maps
     *
     */
    void check_for_correct_maps();

    /*!
     * \brief update maps of state vectors
     *
     */
    void update_maps_of_state_vectors();

    //! @name get states (read only access)
    //! @{

    //! get wall displacements (row map based)
    inline std::shared_ptr<const Core::LinAlg::Vector<double>> get_disp_row() const
    {
      return disp_row_;
    };

    //! get wall displacements (column map based)
    inline std::shared_ptr<const Core::LinAlg::Vector<double>> get_disp_col() const
    {
      return disp_col_;
    };

    //! get wall displacements (row map based) after last transfer
    inline std::shared_ptr<const Core::LinAlg::Vector<double>> get_disp_row_last_transfer() const
    {
      return disp_row_last_transfer_;
    };

    //! get wall velocities (column map based)
    inline std::shared_ptr<const Core::LinAlg::Vector<double>> get_vel_col() const
    {
      return vel_col_;
    };

    //! get wall accelerations (column map based)
    inline std::shared_ptr<const Core::LinAlg::Vector<double>> get_acc_col() const
    {
      return acc_col_;
    };

    //! get wall forces (column map based)
    inline std::shared_ptr<const Core::LinAlg::Vector<double>> get_force_col() const
    {
      return force_col_;
    };

    //! @}

    //! @name get states (read and write access)
    //! @{

    //! get wall displacements (row map based)
    inline std::shared_ptr<Core::LinAlg::Vector<double>> get_disp_row() { return disp_row_; };
    inline std::shared_ptr<Core::LinAlg::Vector<double>>& get_ref_disp_row() { return disp_row_; };

    //! get wall displacements (column map based)
    inline std::shared_ptr<Core::LinAlg::Vector<double>> get_disp_col() { return disp_col_; };
    inline std::shared_ptr<Core::LinAlg::Vector<double>>& get_ref_disp_col() { return disp_col_; };

    //! get wall displacements (row map based) after last transfer
    inline std::shared_ptr<Core::LinAlg::Vector<double>> get_disp_row_last_transfer()
    {
      return disp_row_last_transfer_;
    };

    //! get wall velocities (column map based)
    inline std::shared_ptr<Core::LinAlg::Vector<double>> get_vel_col() { return vel_col_; };

    //! get wall accelerations (column map based)
    inline std::shared_ptr<Core::LinAlg::Vector<double>> get_acc_col() { return acc_col_; };

    //! get wall forces (column map based)
    inline std::shared_ptr<Core::LinAlg::Vector<double>> get_force_col() { return force_col_; };

    //! @}

   private:
    //! particle simulation parameter list
    const Teuchos::ParameterList& params_;

    //! wall discretization
    std::shared_ptr<Core::FE::Discretization> walldiscretization_;

    //! current dof row map
    std::shared_ptr<Core::LinAlg::Map> curr_dof_row_map_;

    //! @name stored states
    //! @{

    //! wall displacements (row map based)
    std::shared_ptr<Core::LinAlg::Vector<double>> disp_row_;

    //! wall displacements (column map based)
    std::shared_ptr<Core::LinAlg::Vector<double>> disp_col_;

    //! wall displacements (row map based) after last transfer
    std::shared_ptr<Core::LinAlg::Vector<double>> disp_row_last_transfer_;

    //! wall velocities (column map based)
    std::shared_ptr<Core::LinAlg::Vector<double>> vel_col_;

    //! wall accelerations (column map based)
    std::shared_ptr<Core::LinAlg::Vector<double>> acc_col_;

    //! wall forces (column map based)
    std::shared_ptr<Core::LinAlg::Vector<double>> force_col_;

    //! @}
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif

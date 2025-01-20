// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SO3_MATERIAL_HPP
#define FOUR_C_MAT_SO3_MATERIAL_HPP

#include "4C_config.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"

#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class So3Material : public Core::Mat::Material
  {
   public:
    int unique_par_object_id() const override = 0;

    void pack(Core::Communication::PackBuffer& data) const override = 0;

    void unpack(Core::Communication::UnpackBuffer& buffer) override = 0;

    //! @name Evaluation methods
    /*!
     * @brief Evaluate the material law, i.e. the stress tensor and the constitutive tensor
     *
     * @param[in] defgrad  Deformation gradient
     * @param[in] glstrain Green-Lagrange strain
     * @param[in] params   Container for additional information
     * @param[out] stress  2nd Piola-Kirchhoff stresses
     * @param[out] cmat    Constitutive matrix
     * @param[in] gp       Current Gauss point
     * @param[in] eleGID   Global element ID
     */
    virtual void evaluate(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
        Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat, int gp,
        int eleGID) = 0;

    /*!
     * @brief Evaluate the nonlinear mass matrix
     *
     * @param[in] defgrad       Deformation gradient
     * @param[in] glstrain      Green-Lagrange strain
     * @param[in] params        Container for additional information
     * @param[out] linmass_disp Linear mass displacement
     * @param[out] linmass_vel  Linear mass velocity
     * @param[in] gp            Current Gauss point
     * @param[in] eleGID        Global element ID
     */
    virtual void evaluate_non_lin_mass(const Core::LinAlg::Matrix<3, 3>* defgrd,
        const Core::LinAlg::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
        Core::LinAlg::Matrix<6, 1>* linmass_disp, Core::LinAlg::Matrix<6, 1>* linmass_vel, int gp,
        int eleGID)
    {
      FOUR_C_THROW("Material of type %s does not support evaluation of nonlinear mass matrix",
          Core::Materials::to_string(this->material_type()).data());
    }

    /*!
     * @brief Evaluate the strain energy function (for hyperelastic materials only)
     *
     * @param[in] glstrain Green-Lagrange strain
     * @param[in, out] psi Strain energy function
     * @param[in] gp       Current Gauss point
     * @param[in] eleGID   Global element ID
     */
    virtual void strain_energy(
        const Core::LinAlg::Matrix<6, 1>& glstrain, double& psi, int gp, int eleGID) const
    {
      FOUR_C_THROW("Material of type %s does not support calculation of strain energy",
          Core::Materials::to_string(this->material_type()).data());
    }

    /*!
     * @brief Evaluate the Cauchy stress contracted with normal and direction vector and
     * its linearizations with given deformation gradient.
     *
     * Cauchy stress is evaluated within this function call. If requested, the required
     * linearizations are calculated. A potential thermal dependency is handled if the temperature
     * is handed in.
     *
     * @param[in] defgrd              Deformation gradient (\f[\mathbf{F}\f])
     * @param[in] n                   Normal vector (\f[\mathbf{n}\f])
     * @param[in] dir                 Direction vector (\f[\mathbf{v}\f]),
     *                                can be either normal or tangential vector
     * @param[out] cauchy_n_dir       Cauchy stress tensor contracted using the vectors n and dir
     *                                (\f[ \mathbf{\sigma} \cdot \mathbf{n} \cdot \mathbf{v} \f])
     * @param[out] d_cauchyndir_dn    Derivative of cauchy_n_dir w.r.t. vector n
     *                                (\f[ \frac{ \mathrm{d} \mathbf{\sigma} \cdot \mathbf{n}
     *                                \cdot \mathbf{v}}{\mathrm{d} \mathbf{n}} \f])
     * @param[out] d_cauchyndir_ddir  Derivative of cauchy_n_dir w.r.t. direction vector v
     *                                (\f[ \frac{ \mathrm{d} \mathbf{\sigma} \cdot \mathbf{n}
     *                                \cdot \mathbf{v}}{\mathrm{d} \mathbf{v}} \f])
     * @param[out] d_cauchyndir_dF    Derivative of cauchy_n_dir w.r.t. deformation gradient
     *                                (\f[ \frac{ \mathrm{d} \mathbf{\sigma} \cdot \mathbf{n}
     *                                \cdot \mathbf{v}}{\mathrm{d} \mathbf{F}} \f])
     * @param[out] d2_cauchyndir_dF2  Second derivative of cauchy_n_dir w.r.t. deformation gradient
     *                                (\f[ \frac{\mathrm{d}^2 \mathbf{\sigma} \cdot \mathbf{n} \cdot
     *                                \mathbf{v}}{\mathrm{d} \mathbf{F} \mathrm{d} \mathbf{F}} \f])
     * @param[out] d2_cauchyndir_dF_dn   Second derivative of cauchy_n_dir w.r.t. deformation
     *                                   gradient and normal vector
     *                                   (\f[ \frac{\mathrm{d}^2 \mathbf{\sigma} \cdot \mathbf{n}
     *                                   \cdot \mathbf{v}}{\mathrm{d} \mathbf{F} \mathrm{d}
     *                                   \mathbf{n}} \f])
     * @param[out] d2_cauchyndir_dF_ddir Second derivative of cauchy_n_dir w.r.t. deformation
     *                                   gradient and direction vector
     *                                   (\f[ \frac{\mathrm{d}^2 \mathbf{\sigma} \cdot \mathbf{n}
     *                                   \cdot \mathbf{v}}{\mathrm{d} \mathbf{F} \mathrm{d}
     *                                   \mathbf{v} } \f])
     * @param[in] gp                Current Gauss point
     * @param[in] eleGID            Global element ID
     * @param[in] concentration     Concentration
     * @param[in] temp              Temperature
     * @param[out] d_cauchyndir_dT  Derivative of cauchy_n_dir w.r.t. temperature (\f[ \frac{
     *                              \mathrm{d} \mathbf{\sigma} \cdot \mathbf{n} \cdot \mathbf{v}}{
     *                              \mathrm{d} T} \f])
     * @param[out] d2_cauchyndir_dF_dT   Second derivative of cauchy_n_dir w.r.t. deformation
     *                                   gradient and temperature (\f[ \frac{\mathrm{d}^2
     *                                   \mathbf{\sigma} \cdot \mathbf{n} \cdot
     *                                   \mathbf{v}}{\mathrm{d} \mathbf{F} \mathrm{d} T } \f])
     */
    virtual void evaluate_cauchy_n_dir_and_derivatives(const Core::LinAlg::Matrix<3, 3>& defgrd,
        const Core::LinAlg::Matrix<3, 1>& n, const Core::LinAlg::Matrix<3, 1>& dir,
        double& cauchy_n_dir, Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dn,
        Core::LinAlg::Matrix<3, 1>* d_cauchyndir_ddir, Core::LinAlg::Matrix<9, 1>* d_cauchyndir_dF,
        Core::LinAlg::Matrix<9, 9>* d2_cauchyndir_dF2,
        Core::LinAlg::Matrix<9, 3>* d2_cauchyndir_dF_dn,
        Core::LinAlg::Matrix<9, 3>* d2_cauchyndir_dF_ddir, int gp, int eleGID,
        const double* concentration, const double* temp, double* d_cauchyndir_dT,
        Core::LinAlg::Matrix<9, 1>* d2_cauchyndir_dF_dT)
    {
      FOUR_C_THROW("evaluate_cauchy_n_dir_and_derivatives not implemented for material of type %s",
          Core::Materials::to_string(this->material_type()).data());
    }

    /*!
     * @brief Evaluate the derivative of the deformation gradient w.r.t. degree of freedom x
     *
     * @param[in] defgrd        Deformation gradient
     * @param[in] concentration Concentration at Gauss point
     * @param[out] d_F_dx       Derivative of deformation gradient w.r.t. degree of freedom x
     */
    virtual void evaluate_linearization_od(const Core::LinAlg::Matrix<3, 3>& defgrd,
        double concentration, Core::LinAlg::Matrix<9, 1>* d_F_dx)
    {
      FOUR_C_THROW("evaluate_linearization_od not implemented for material of type %s",
          Core::Materials::to_string(this->material_type()).data());
    }
    //@}

    /*!
     * @brief Return whether material includes a varying material density
     */
    virtual bool varying_density() const { return false; }


    //! @name Handling of Gauss point data
    /*!
     * @brief Check if element kinematics and material kinematics are compatible
     */
    virtual void valid_kinematics(Inpar::Solid::KinemType kinem) = 0;

    /*!
     * @brief Set up for materials with GP data (e.g., history variables)
     *
     * @param[in] numgp   Current Gauss point
     * @param[in] container Input parameter container
     */
    virtual void setup(int numgp, const Core::IO::InputParameterContainer& container) {}

    /*!
     * @brief Post setup routine which will be called after all elements were read and set up
     *
     * This method will be called after the input phase to setup the material with
     * input data that has not yet been read during the setup(int,Input::LineDefinition*) call.
     *
     * @param[in] params Container for additional information passed from the element
     * @param[in] eleGID Global element ID
     */
    virtual void post_setup(Teuchos::ParameterList& params, const int eleGID) {}

    /*!
     * @brief Update of GP data (e.g., history variables)
     */
    virtual void update() {}

    /*!
     * @brief Indicator, whether the extended update call is used
     *
     * Return true, if the material needs the update(defgrd, gp, params, eleGID) call
     */
    virtual bool uses_extended_update() { return false; }

    /*!
     * @brief Update of GP data (e.g., history variables)
     *
     * This method is currently only called from specific element types. If you need the additional
     * functionality compared to update() with any other element than the adapted ones, you need to
     * implement it yourself. Currently only HEX8 and HEX8FBAR elements are supported
     *
     * Materials that use this method need to return true in UsesExtendedUpdate()
     *
     * @param[in] defgrd Deformation gradient
     * @param[in] gp     Gauss point
     * @param[in] params Container for additional information
     * @param[in] eleGID Global element ID
     */
    virtual void update(Core::LinAlg::Matrix<3, 3> const& defgrd, int const gp,
        Teuchos::ParameterList& params, int const eleGID)
    {
    }

    /*!
     * @brief Reset time step (for time adaptivity)
     */
    virtual void reset_step() {}

    /*!
     * @brief Store internal history variables to be eventually reset at some point
     *
     * @param[in] timestep Timestep
     */
    virtual void store_history(int timestep) {}

    /*!
     * @brief Set history variables from time point given as input
     *
     * @param[in] timestep Timestep
     */
    virtual void set_history(int timestep) {}
    //@}


    //! @name Visualization methods
    //@{
    /*!
     * @brief Return names of visualization data
     *
     * @param[in] names Names of the data to export
     */
    virtual void vis_names(std::map<std::string, int>& names) const {}

    /*!
     * @brief Return visualization data
     *
     * @param[in] name  Name of the data to export
     * @param[in] data  Data to export
     * @param[in] numgp Gauss point
     */
    virtual bool vis_data(const std::string& name, std::vector<double>& data, int numgp) const
    {
      return false;
    }

    /*!
     * @brief Return visualization data
     *
     * @param[in] name  Name of the data to export
     * @param[in] data  Data to export
     * @param[in] numgp Gauss point
     * @param[in] eleId Element ID
     */
    virtual bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleId) const
    {
      return false;
    }

    /*!
     * @brief Register names of the internal data that should be saved during runtime output
     *
     * @param[out] name_and_size Unordered map of names of the data with the respective vector size
     */
    virtual void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const
    {
    }

    /*!
     * @brief Evaluate internal data for every Gauss point saved for output during runtime
     * output
     *
     * @param[in] name  Name of the data to export
     * @param[out] data NUMGPxNUMDATA Matrix holding the data
     *
     * @return true if data is set by the material, otherwise false
     */
    virtual bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
    {
      return false;
    }
    //@}


    //! @name Query methods
    /*!
     * @brief Return whether the material requires the deformation gradient for its evaluation
     */
    virtual bool needs_defgrd() const { return false; }
    //@}
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

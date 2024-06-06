/*----------------------------------------------------------------------*/
/*! \file

\brief This file holds the definition of a mixture constituent interface

\level 3


*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_MIXTURE_CONSTITUENT_HPP
#define FOUR_C_MIXTURE_CONSTITUENT_HPP

#include "4C_config.hpp"

#include "4C_comm_pack_buffer.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_material_parameter_base.hpp"

#include <memory>
#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Mat
{
  class Anisotropy;
}

/*!
 * \brief The mixture namespace holds all mixture specific classes.
 *
 * The idea behind mixtures is that multiple materials share the same
 * deformation and the stress response is a mass fraction weighted
 * average of the stresses of each constituent.
 */
namespace MIXTURE
{
  // forward declaration
  class MixtureConstituent;
  class MixtureRule;

  namespace PAR
  {
    class MixtureConstituent : public Core::Mat::PAR::Parameter
    {
     public:
      explicit MixtureConstituent(const Teuchos::RCP<Core::Mat::PAR::Material>& matdata);

      /// create material instance of matching type with my parameters
      virtual std::unique_ptr<MIXTURE::MixtureConstituent> CreateConstituent(int id) = 0;

      /// create material instance of matching type with my parameters
      Teuchos::RCP<Core::Mat::Material> create_material() final;

      static MIXTURE::PAR::MixtureConstituent* Factory(int matnum);
    };
  }  // namespace PAR

  /*!
   * \brief This is the base class of a constituent in a mixture defining the interface to the
   * holder class
   *
   * This abstract class defines the interface that a constituents needs to implement. It has to be
   * paired with Mat::Mixture and MIXTURE::MixtureRule.
   *
   * Example input lines:
   * MAT 1 MAT_Mixture NUMCONST 2 MATIDSCONST 11 12 MASSFRAC 0.4 0.6 MATIDMIXTURERULE 10
   *  DENS 0.1
   * MAT 10 MIX_Rule_Simple
   * MAT 11 MIX_Constituent_ElastHyper NUMMAT 1 MATIDS 101 MAT 12
   * MIX_Constituent_ElastHyper NUMMAT 1 MATIDS 102
   * MAT 101 ELAST_CoupLogNeoHooke MODE YN C1 2.5e4 C2 0.27
   * MAT 102 ELAST_CoupAnisoExpo K1 1.666666666666e4 K2 10.0 GAMMA 0.0 K1COMP
   *  0.833333333333e4 K2COMP 10.0 ADAPT_ANGLE No INIT 0 STR_TENS_ID 1000
   * MAT 1000 ELAST_StructuralTensor STRATEGY Standard
   *
   */
  class MixtureConstituent
  {
   public:
    MixtureConstituent(MIXTURE::PAR::MixtureConstituent* params, int id);

    virtual ~MixtureConstituent() = default;

    /// Returns the id of the constituent
    int Id() const { return id_; }

    /*!
     * \brief Pack data into a char vector from this class
     *
     * The vector data contains all information to rebuild the exact copy of an instance of a class
     * on a different processor. The first entry in data hast to be an integer which is the unique
     * parobject id defined at the top of the file and delivered by UniqueParObjectId().
     *
     * @param data (in/put) : vector storing all data to be packed into this instance.
     */
    virtual void PackConstituent(Core::Communication::PackBuffer& data) const;

    /*!
     * \brief Unpack data from a char vector into this class to be called from a derived class
     *
     * The vector data contains all information to rebuild the exact copy of an instance of a class
     * on a different processor. The first entry in data hast to be an integer which is the unique
     * parobject id defined at the top of the file and delivered by UniqueParObjectId().
     *
     * @param position (in/out) : current position to unpack data
     * @param data (in) : vector storing all data to be unpacked into this instance.
     */
    virtual void UnpackConstituent(
        std::vector<char>::size_type& position, const std::vector<char>& data);

    /// material type
    virtual Core::Materials::MaterialType MaterialType() const = 0;

    /*!
     * \brief Register anisotropy extensions of all sub-materials of the constituent
     *
     * \param anisotropy Reference to the global anisotropy manager
     */
    virtual void register_anisotropy_extensions(Mat::Anisotropy& anisotropy)
    {
      // do nothing in the default case
    }

    /*!
     * Initialize the constituent with the parameters of the input line
     *
     * @param numgp (in) Number of Gauss-points
     * @param params (in/out) Parameter list for exchange of parameters
     */
    virtual void ReadElement(int numgp, Input::LineDefinition* linedef);

    /*!
     * Returns whether the constituent is already set up
     * @return true if the constituent is already set up, otherwise false
     */
    bool is_setup() { return is_setup_; }

    /*!
     * \brief Setup the constituent
     *
     * This method is called once for Gauss point at the beginning of the simulation.
     * The constituent should setup all internal variables and materials.
     *
     * @param params Container for additional information
     * @param eleGID Global element id
     */
    virtual void Setup(Teuchos::ParameterList& params, int eleGID);

    /*!
     * \brief Update of the internal variables
     *
     * This method is called once per Gauss point between each time step to update the internal
     * variables. (Not needed for simple constituents)
     *
     * @param defgrd Deformation gradient of the previous timestep
     * @param params Container for additional information
     * @param gp Gauss point
     * @param eleGID Global element identifier
     */
    virtual void Update(Core::LinAlg::Matrix<3, 3> const& defgrd, Teuchos::ParameterList& params,
        const int gp, const int eleGID)
    {
    }

    /*!
     * @brief Update of the internal variables used for mixture rules evaluating the elastic part of
     * the deformation. This method must be explicitly called by the mixture rule and will not be
     * invoked automatically!
     *
     * @param F Deformation gradient of the previous timestep
     * @param iFext External deformation gradient of the previous timestep
     * @param params Container for additional information
     * @param dt Time increment from the previous timestep
     * @param gp Gauss-point
     * @param eleGID Global element id
     */
    virtual void UpdateElasticPart(const Core::LinAlg::Matrix<3, 3>& F,
        const Core::LinAlg::Matrix<3, 3>& iFext, Teuchos::ParameterList& params, const double dt,
        const int gp, const int eleGID)
    {
      // do nothing
    }

    /*!
     *
     * @brief Method that is executed before the first evaluate call, once for each Gauss point
     *
     * @param params (in) : Container for additional information
     * @param gp (in) : Gauss point
     * @param eleGID (in) : Global element id
     */
    virtual void pre_evaluate(
        MixtureRule& mixtureRule, Teuchos::ParameterList& params, int gp, int eleGID)
    {
      // do nothing in the default case
    }

    /*!
     * \brief Returns the scalar indicating the growth scale from the reference configuration
     *
     * \return double
     */
    [[nodiscard]] virtual double GetGrowthScalar(int gp) const { return 1.0; }


    /*!
     * @brief evaluates the derivative of the growth scalar w.r.t. Cauchy-Green deformation tensor
     *
     * @note This matrix is usually just a zero-matrix. It is non-zero if the growth scalar changes
     * with the deformation.
     *
     * @param gp (in) : Gauss point id
     * @param eleGID (in) : Global element id
     * @return Core::LinAlg::Matrix<1, 6> Derivative of the growth scalar w.r.t. Cauchy-Green
     * deformation tensor
     */
    [[nodiscard]] virtual Core::LinAlg::Matrix<1, 6> GetDGrowthScalarDC(int gp, int eleGID) const
    {
      const Core::LinAlg::Matrix<1, 6> dGrowthScalarDC(true);
      return dGrowthScalarDC;
    };

    /*!
     * @brief Evaluates the stress and material linearization of the constituents with an
     * inelastic part of the deformation
     *
     * The total deformation is #F, which is split into two parts:
     *
     * $\boldsymbol{F} = \boldsymbol{F}_e \cdot \boldsymbol{F}_in$
     *
     * Only elastic part $\boldsymbol{F}_e$ causes stresses. The inelastic part is only needed
     * for the linearization.
     *
     * @note S_stress and the linearization are specific quantities. They have to be multiplied with
     * the density of the constituent to obtain the real stress oder linearization.
     *
     * @param F Total deformation gradient
     * @param iF_in inverse inelastic part of the deformation
     * @param params Container for additional information
     * @param S_stress 2nd specific Piola-Kirchhoff stress in stress-like Voigt notation
     * @param cmat specific linearization of the material tensor in Voigt notation
     * @param gp Gauss-point
     * @param eleGID Global element id
     */
    virtual void EvaluateElasticPart(const Core::LinAlg::Matrix<3, 3>& F,
        const Core::LinAlg::Matrix<3, 3>& iF_in, Teuchos::ParameterList& params,
        Core::LinAlg::Matrix<6, 1>& S_stress, Core::LinAlg::Matrix<6, 6>& cmat, int gp, int eleGID);

    /*!
     * Evaluates the constituents. Needs to compute the stress contribution of the constituent out
     * of the displacements. Will be called for each Gauss point
     *
     * @note S_stress and the linearization are specific quantities. They have to be multiplied with
     * the density of the constituent to obtain the real stress oder linearization.
     *
     * @param F (in) : Deformation gradient
     * @param E_strain (in) : Green-Lagrange strain in strain-like Voigt notation
     * @param params Container for additional information
     * @param S_stress (out) : 2nd specific Piola Kirchhoff stress tensor in stress like
     * Voigt-notation
     * @param cmat (out) : specific constitutive tensor in Voigt notation
     * @param gp (in) : Number of Gauss point
     * @param eleGID (in) : Global element id
     */
    virtual void Evaluate(const Core::LinAlg::Matrix<3, 3>& F,
        const Core::LinAlg::Matrix<6, 1>& E_strain, Teuchos::ParameterList& params,
        Core::LinAlg::Matrix<6, 1>& S_stress, Core::LinAlg::Matrix<6, 6>& cmat, int gp,
        int eleGID) = 0;

    /// Returns the refenrence mass density. Needs to be implemented by the deriving class.

    /*!
     * \brief Register names of the internal data that should be saved during runtime output
     *
     * \param name_and_size [out] : unordered map of names of the data with the respective vector
     * size
     */
    virtual void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const
    {
      // do nothing for simple constituents
    }

    /*!
     * \brief Evaluate internal data for every Gauss point saved for output during runtime output
     *
     * \param name [in] : Name of the data to export
     * \param data [out] : NUMGPxNUMDATA Matrix holding the data
     *
     * \return true if data is set by the material, otherwise false
     */
    virtual bool EvaluateOutputData(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
    {
      return false;
    }



   protected:
    /*!
     * Get number of Gauss points used
     *
     * @return Number of Gauss points
     */
    int num_gp() const { return numgp_; }

   private:
    ///! Number of Gauss points
    int numgp_;

    ///! Indicator, whether the constituent has already read the element
    bool has_read_element_;

    ///! Indicator, whether the constituent is already set up
    bool is_setup_;

    ///! Id of the constituent
    const int id_;
  };

}  // namespace MIXTURE

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SCATRA_REACTION_HPP
#define FOUR_C_MAT_SCATRA_REACTION_HPP



#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  // forward declaration
  class MatListReactions;

  namespace PAR
  {
    namespace REACTIONCOUPLING
    {
      class ReactionInterface;
    }

    enum ReactionCoupling
    {
      reac_coup_none,                   ///< no coupling, initialization value
      reac_coup_simple_multiplicative,  ///< coupling of type A*B
      reac_coup_power_multiplicative,   ///< coupling of type A*B
      reac_coup_constant,               //< no coupling, constant increase/decrease
      reac_coup_michaelis_menten,       // coupling of type (B/(const+B))*A
      reac_coup_byfunction,             // coupling defined by function from input file
    };

    /*!
    \brief This file contains the base material for reactive scalars

    This class encapsulates the reaction kinematics, defined by the reaction_coupling variable.
    The key methods are the CalcReaBodyForce...() methods. There, all
    reaction terms and derivatives are evaluated.
    Note: The CalcReaBodyForce...() terms are of arbitrary form, but keep in mind that if you are
    unlucky, (if the term is not of the form K(c) \cdot c_i ) you might get stability issues.

    Detailed description of the corresponding equations and examples:


    <h3>Homogeneous Scatra Couplings</h3>

    Note that for the implementation of the simple_multiplicative scatra coupling:
    assume the following reaction: 1*A + 2*B  --> 3*C with reaction coefficient 4.0

    If we assume the reaction is depending on the product of all
    reactants (this corresponds to couplingtype "simple_multiplicative"),
    the corresponding equations are: \partial_t A = -(4*1*B)*A  (negative since reactant)
                                     \partial_t B = -(4*2*A)*B  (negative since reactant)
                                     \partial_t C = + 4*3*A*B   (positive since product)

    This equation is achieved in 4C via the MAT_scatra_reaction material:
    ----------------------------------------------------------MATERIALS
    MAT 1 MAT_matlist_reactions LOCAL No NUMMAT 3 MATIDS 2 4 5 NUMREAC 1 REACIDS 3 END //collect
    Concentrations MAT 2 MAT_scatra DIFFUSIVITY 0.0 MAT 4 MAT_scatra DIFFUSIVITY 0.0 MAT 5
    MAT_scatra DIFFUSIVITY 0.0 MAT 3 MAT_scatra_reaction NUMSCAL 3 STOICH -1 -2 3 REACCOEFF 4.0
    COUPLING simple_multiplicative ROLE 1 1 0

    Implementation of the reaction term hence f(c)=(-(4*1*B)*A;-(4*2*A)*B;4*3*A*B) and corresponding
    derivatives.


    <h3>Michaelis-Menten like Reaction Kinetics</h3>

    Note that for the implementation of the michaelis-menten scatra coupling:
    reactant A promotes C and reactant B influences only until certain limit

    MAT 1 MAT_matlist_reactions LOCAL No NUMMAT 3 MATIDS 2 4 5 NUMREAC 1 REACIDS 3 END //collect
    Concentrations MAT 2 MAT_scatra DIFFUSIVITY 0.0 MAT 4 MAT_scatra DIFFUSIVITY 0.0 MAT 5
    MAT_scatra DIFFUSIVITY 0.0 MAT 3 MAT_scatra_reaction NUMSCAL 3 STOICH -1 -1 1 REACCOEFF 5.0
    COUPLING michaelis_menten ROLE -1 3 0

    The corresponding equations are
                \partial_t A = -5*A*(B/(3+B))
                \partial_t B = -5*A*(B/(3+B))
                \partial_t C =  5*A*(B/(3+B))

    Thereby ROLE does describe of how to build the reaction term (negative value -1.3: simply
    multiply by A, positive value 3.2: multiply by B/(B+3.2) ) and STOICH does describe on which
    scalar the reaction term should be applied. Here another example:

    MAT 3 MAT_scatra_reaction NUMSCAL 4 STOICH +2 -1 0 0  REACCOEFF 2.1 COUPLING michaelis_menten
    ROLE -1 0 1.2 3.0

    The corresponding equations are
               \partial_t A =  2*2.1*A*C/(C+1.2)*D/(D+3.0)
               \partial_t B = -1*2.1*A*C/(C+1.2)*D/(D+3.0)
               \partial_t C =  0
               \partial_t D =  0


    <h3>Constant Kinetics</h3>

    Note for the implementation of the constant scatra coupling:
    Product A is constantly produced

    MAT 1 MAT_matlist_reactions LOCAL No NUMMAT 1 MATIDS 2 NUMREAC 1 REACIDS 3 END //collect
    Concentrations MAT 2 MAT_scatra DIFFUSIVITY 0.0 MAT 3 MAT_scatra_reaction NUMSCAL 1 STOICH 2
    REACCOEFF 5.0 COUPLING constant ROLE 1

    The corresponding equation is:
                \partial_t A = 5*2

    Implementation is of form: \partial_t c_i + K_i(c)*c_i = f_i(c), were f_i(c) is supposed not to
    depend linearly on c_i hence we have to calculate and set K(c)=(0) and f(c)=(5*2) and zero
    derivatives.


    <h3>Power Law Kinetics</h3>

    Note for the implementation of the power law:
    assume the following reaction: 1*A + 2*B  --> 3*C with reaction coefficient 4.0

    If we assume the reaction is depending on the product of all reactants via a power law
    (this corresponds to couplingtype "power_multiplicative"),

    The corresponding equations are: \partial_t A = -(4*1*B^2)*A^3  (negative since reactant)
                                     \partial_t B = -(4*2*A^3)*B^2  (negative since reactant)
                                     \partial_t C = + 4*3*A^3*B^2   (positive since product)

    This equation is in 4C achieved by the MAT_scatra_reaction material:
    ----------------------------------------------------------MATERIALS
    MAT 1 MAT_matlist_reactions LOCAL No NUMMAT 3 MATIDS 2 4 5 NUMREAC 1 REACIDS 3 END //collect
    Concentrations MAT 2 MAT_scatra DIFFUSIVITY 0.0 MAT 4 MAT_scatra DIFFUSIVITY 0.0 MAT 5
    MAT_scatra DIFFUSIVITY 0.0 MAT 3 MAT_scatra_reaction NUMSCAL 3 STOICH -1 -2 3 REACCOEFF 4.0
    COUPLING power_multiplicative ROLE 3 2 0


    <h3>Kinetics Defined by Function</h3>

    Note for the implementation of the reaction-by-function:
    assume the following reaction: 1*A + 2*B  --> 3*C with reaction coefficient 4.0

    The reaction is defined by a function from the input file
    (this corresponds to couplingtype "by_function").

    It can reproduce all other implemented reaction types (and more). Also, linearizations are
    computed automatically However, keep in mind, that it might be little slower ...

    For instance if you want to model mechaelis-menten type kinematics of the form

               \partial_t A =  2*2.1*A*C/(C+1.2)*D/(D+3.0)
               \partial_t B = -1*2.1*A*C/(C+1.2)*D/(D+3.0)
               \partial_t C =  0
               \partial_t D =  0

    This equation is in 4C achieved by the MAT_scatra_reaction material combined with the reaction
    by function feature as:
    ----------------------------------------------------------MATERIALS
    MAT 1 MAT_matlist_reactions LOCAL No NUMMAT 3 MATIDS 2 4 5 NUMREAC 1 REACIDS 3 END //collect
    Concentrations MAT 2 MAT_scatra DIFFUSIVITY 0.0 MAT 4 MAT_scatra DIFFUSIVITY 0.0 MAT 5
    MAT_scatra DIFFUSIVITY 0.0 MAT 3 MAT_scatra_reaction NUMSCAL 4 STOICH +2 -1 0 0  REACCOEFF 2.1
    COUPLING by_function ROLE 1 1 0 0
    -------------------------------------------------------------FUNCT1
    COMPONENT 0 VARFUNCTION phi1*phi3/(phi3+1.2)*phi4/(phi4+3.0)

    Note: You have to use the VARFUNCTION function in order to be able to read the 'phix' variables.
          The function is defined within the role list


    <h3>REACSTART Feature</h3>

    Note for the implementation of the reacstart:
    Assume concentration A is reproducing with reaction coefficient 1.0 and additionally if the
    concentration of B exceeds some threshold 2.0 another reaction starts A->3*C with reaction
    coefficient 4.0.

    The corresponding equations could be:
                \partial_t A = -(-1.0)*A - 4.0*A*(B - 2.0)_{+} (first term positive, since
    equivalent as reactant with negative reaction coefficient) \partial_t B = 0 \partial_t c
    = 3.0*4.0*A*(B - 2.0)_{+}   (positive since product)

    This equation is achieved in 4C via the boundary condition:
    MAT 1 MAT_matlist_reactions LOCAL No NUMMAT 3 MATIDS 2 2 2 NUMREAC 2 REACIDS 3 4 END //collect
    Concentrations MAT 2 MAT_scatra DIFFUSIVITY 0.0 MAT 3 MAT_scatra_reaction NUMSCAL 3 STOICH -1 0
    0 REACCOEFF -1.0 COUPLING simple_multiplicative ROLE -1 0 0 MAT 4 MAT_scatra_reaction NUMSCAL 3
    STOICH -1 0 3 REACCOEFF 4.0 COUPLING simple_multiplicative ROLE -1 -1 0 REACSTART 0.0 2.0 0.0

    <h3>DISTRFUNCT Feature</h3>

    spatially varying reaction coefficient

    The spatial distribution is defined by a function from the input file.
    <CODE> DISTRFUNCT </CODE> defines the corresponding function ID.
    Its default value is <CODE> DISTRFUNCT=0 </CODE>, which implies a spatially constant reaction
    coefficient.

    Example:<BR>
    Assume uncoupled harmonic oscillators \f$A(x,t)\f$ with spatially linearly varying frequency
    \f$w(x) \in [0.5\pi, 2\pi]\f$ on the spatial domain \f$x \in [-4,4]\f$.<BR> The following PDEs
    describe the oscillators:<BR>
               \f{eqnarray*}{ \partial_t A &=& B\\
                              \partial_t B &=& -w(x)^2 A\\
                               w(x) &=& 0.5\pi+1.5\pi\frac{x+4}{8}
                \f}

    This equation is achieved in 4C via the MAT_scatra_reaction material:

    <CODE>
            MAT 1 MAT_matlist_reactions LOCAL No NUMMAT 2  MATIDS 2 2 NUMREAC 2 REACIDS 4 5 END
    //collect Concentrations<BR> MAT 2 MAT_scatra DIFFUSIVITY 0.0<BR> MAT 4 MAT_scatra_reaction
    NUMSCAL 2 STOICH  1 0 REACCOEFF  1.0 DISTRFUNCT 0 COUPLING simple_multiplicative ROLE 0 1<BR>
            MAT 5 MAT_scatra_reaction NUMSCAL 2 STOICH  0 1 REACCOEFF -1.0 DISTRFUNCT 2 COUPLING
    simple_multiplicative ROLE 1 0<BR>
            -------------------------------------------------------------FUNCT2<BR>
            FUNCT 2 COMPONENT 0 EXPR 0 0 0 FUNCTION (0.5*pi+(x+4)*1.5*pi/8)^2
    </CODE>

    Note: In <CODE> MAT 4 </CODE>, the value of <CODE>DISTRFUNCT=0</CODE> implies a spatially
    constant reaction coefficient. Since zero is the default value, it is, however, not necessary to
    define explicitly define a reaction coefficient as spatially constant by stating
    <CODE>DISTRFUNCT 0</CODE>. Stating <CODE>DISTRFUNCT 0</CODE> may be omitted.

    */

    /*----------------------------------------------------------------------*/
    /// parameters for scalar transport material
    class ScatraReactionMat : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      ScatraReactionMat(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// returns the enum of the current coupling type
      Mat::PAR::ReactionCoupling set_coupling_type(const Core::Mat::PAR::Parameter::Data& matdata);

      /// Initialize
      void initialize();

      /// number of scalars in this reaction
      const int numscal_;

      /// the list of material IDs
      const std::vector<int> stoich_;

      /// reaction coefficient
      const double reaccoeff_;

      /// ID of function describing spatial distribution of reaction coefficient
      const int distrfunctreaccoeffid_;

      /// type of coupling
      const Mat::PAR::ReactionCoupling coupling_;

      /// specifies scalar type in reaction
      const std::vector<double> couprole_;

      /// parameter to define start of reaction
      const std::vector<double> reacstart_;

      /// flag if there is a spatial distribution of reaction coefficient
      const bool isdistrfunctreaccoeff_;

      /// flag if there is a reacstart value
      bool isreacstart_;

      /// flag if initialization has been done
      bool isinit_;

      /// implementation of reaction coupling
      std::shared_ptr<REACTIONCOUPLING::ReactionInterface> reaction_;

    };  // class ScatraReactionMat

  }  // namespace PAR

  class ScatraReactionMatType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ScatraReactionMatType"; }

    static ScatraReactionMatType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ScatraReactionMatType instance_;
  };  // class ScatraReactionMatType

  /*----------------------------------------------------------------------*/
  /// wrapper for scalar transport material
  class ScatraReactionMat : public Core::Mat::Material
  {
    friend class Mat::MatListReactions;

   public:
    /// construct empty material object
    ScatraReactionMat();

    /// construct the material object given material parameters
    explicit ScatraReactionMat(Mat::PAR::ScatraReactionMat* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return ScatraReactionMatType::instance().unique_par_object_id();
    }

    /*!
      \brief Pack this class so it can be communicated

      Resizes the vector data and stores all information of a class in it.
      The first information to be stored in data has to be the
      unique parobject id delivered by unique_par_object_id() which will then
      identify the exact class on the receiving processor.

      \param data (in/out): char vector to store class information
    */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
      \brief Unpack data from a char vector into this class

      The vector data contains all information to rebuild the
      exact copy of an instance of a class on a different processor.
      The first entry in data has to be an integer which is the unique
      parobject id defined at the top of this file and delivered by
      unique_par_object_id().

      \param data (in) : vector storing all data to be unpacked into this
      instance.
    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    /// initialize
    virtual void initialize() { params_->initialize(); }

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_scatra_reaction;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ScatraReactionMat>(*this);
    }

    /// return number of scalars for this reaction
    int num_scal() const { return params_->numscal_; }

    double calc_perm_influence(const int k,  //!< current scalar id
        const std::vector<double>& phinp,    //!< scalar values at t_(n+1)
        const double time,                   //!< current time
        const double* gpcoord,               //!< Gauss-point coordinates
        const double scale                   //!< scaling factor for reference concentrations
    ) const;

    void calc_perm_influence_deriv(const int k,  //!< current scalar id
        std::vector<double>& derivs,             //!< vector with derivatives (to be filled)
        const std::vector<double>& phinp,        //!< scalar values at t_(n+1)
        const double time,                       //!< current time
        const double* gpcoord,                   //!< Gauss-point coordinates
        const double scale                       //!< scaling factor for reference concentrations
    ) const;

   protected:
    /// return ID of function for spatial distribution of reaction coefficient
    int dis_funct_reac_coeff_id() const { return params_->distrfunctreaccoeffid_; }

    //! Return reaction coefficient at Gauss-point
    /*!
         Depending on whether the reaction coefficient is defined as spatially varying
         or not (see Mat::PAR::ScatraReactionMat::isdistrfunctreaccoeff_ ), the reaction coefficient
       defined by <CODE> REACCOEFF </CODE> (see Mat::PAR::ScatraReactionMat::reaccoeff_ ) is either
       updated by evaluating the distribution function <CODE> DISTRFUNCT </CODE> (see
       Mat::PAR::ScatraReactionMat::distrfunctreaccoeffid_ ) at the Gauss-point coordinates, or left
       as is.

         \return reaction coefficient at Gauss-point (double)
       */
    virtual double reac_coeff(const std::vector<std::pair<std::string, double>>&
            constants  //!< vector containing values which are independent of the scalars
    ) const;

    /// return stoichometrie
    const std::vector<int>* stoich() const { return &params_->stoich_; }

    /// return type of coupling
    Mat::PAR::ReactionCoupling coupling() const { return params_->coupling_; }

    /// return role in coupling
    const std::vector<double>* couprole() const { return &params_->couprole_; }

    /// delayed reaction start coefficient
    const std::vector<double>* reac_start() const { return &params_->reacstart_; }

    /// return flag if there is a reacstart value
    bool is_reac_start() const { return params_->isreacstart_; }

    /// return flag if there is a spatial distribution function for the reaction coefficient
    bool get_is_distr_funct_reac_coeff() const { return params_->isdistrfunctreaccoeff_; }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// calculate advanced reaction terms
    double calc_rea_body_force_term(const int k,  //!< current scalar id
        const std::vector<double>& phinp,         //!< scalar values at t_(n+1)
        const std::vector<std::pair<std::string, double>>&
            constants,  //!< vector containing values which are independent of the scalars
        const double
            scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
    ) const;

    /// calculate advanced reaction term derivatives
    void calc_rea_body_force_deriv_matrix(const int k,  //!< current scalar id
        std::vector<double>& derivs,                    //!< vector with derivatives (to be filled)
        const std::vector<double>& phinp,               //!< scalar values at t_(n+1)
        const std::vector<std::pair<std::string, double>>&
            constants,  //!< vector containing values which are independent of the scalars
        const double
            scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
    ) const;

    /// calculate advanced reaction term derivatives after additional variables of the specified
    /// function
    void calc_rea_body_force_deriv_matrix_add_variables(const int k,  //!< current scalar id
        std::vector<double>& derivs,  //!< vector with derivatives (to be filled)
        const std::vector<std::pair<std::string, double>>& variables,  //!< variables
        const std::vector<std::pair<std::string, double>>&
            constants,    //!< constants (including the scalar values phinp)
        double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
    ) const;

    /// add variables to the by-function reaction
    void add_additional_variables(const int k,                        //!< current scalar id
        const std::vector<std::pair<std::string, double>>& variables  //!< variables
    ) const;

   protected:
    /// helper for calculating advanced reaction terms
    double calc_rea_body_force_term(int k,  //!< current scalar id
        const std::vector<double>& phinp,   //!< scalar values at t_(n+1)
        const std::vector<std::pair<std::string, double>>&
            constants,      //!< vector containing values which are independent of the scalars
        double scale_reac,  //!< scaling factor for reaction term (= reaction coefficient *
                            //!< stoichometry)
        double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
    ) const;

    /// helper for calculating advanced reaction term derivatives
    void calc_rea_body_force_deriv(int k,  //!< current scalar id
        std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
        const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
        const std::vector<std::pair<std::string, double>>&
            constants,      //!< vector containing values which are independent of the scalars
        double scale_reac,  //!< scaling factor for reaction term (= reaction coefficient *
                            //!< stoichometry)
        double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
    ) const;

    /// helper for calculating advanced reaction term derivatives
    void calc_rea_body_force_deriv_add_variables(int k,  //!< current scalar id
        std::vector<double>& derivs,                     //!< vector with derivatives (to be filled)
        const std::vector<std::pair<std::string, double>>& variables,  //!< variables
        const std::vector<std::pair<std::string, double>>&
            constants,      //!< constants (including the scalar values phinp)
        double scale_reac,  //!< scaling factor for reaction term (= reaction coefficient *
                            //!< stoichometry)
        double scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
    ) const;

   private:
    /// my material parameters
    Mat::PAR::ScatraReactionMat* params_;
  };  // class ScatraReactionMat

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

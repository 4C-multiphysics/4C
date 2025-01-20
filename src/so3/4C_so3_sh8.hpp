// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SO3_SH8_HPP
#define FOUR_C_SO3_SH8_HPP

#include "4C_config.hpp"

#include "4C_linalg_serialdensematrix.hpp"
#include "4C_so3_hex8.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
struct _SOH8_DATA;

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    class SoSh8Type : public SoHex8Type
    {
     public:
      std::string name() const override { return "SoSh8Type"; }

      static SoSh8Type& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      int initialize(Core::FE::Discretization& dis) override;

      void nodal_block_information(
          Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static SoSh8Type instance_;

      std::string get_element_type_string() const { return "SOLIDSH8"; }
    };

    /*!
    \brief A C++ 8-node Solid-Shell element inherited from so_hex8

    The Solid-Shell element technology is based on the work of
    (1) Vu-Quoc, Tan: "Optimal solid shells for non-linear analyses
                       of multilayer composites", CMAME 2003
    (2) Klinkel, Gruttmann, Wagner: "A robust non-linear solid shell element
                                     based on a mixed variational fromulation"

    Refer also to the Semesterarbeit of Alexander Popp, 2006

    */
    class SoSh8 : public SoHex8
    {
     public:
      //! @name Friends
      friend class SoSh8Type;
      friend class Soh8Surface;
      friend class Soh8Line;

      //@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner : elements owning processor
      */
      SoSh8(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      SoSh8(const SoSh8& old);

      /*!
      \brief Deep copy this instance of Solid3 and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int unique_par_object_id() const override
      {
        return SoSh8Type::instance().unique_par_object_id();
      }

      /*!
      \brief Pack this class so it can be communicated

      \ref pack and \ref unpack are used to communicate this element

      */
      void pack(Core::Communication::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref pack and \ref unpack are used to communicate this element

      */
      void unpack(Core::Communication::UnpackBuffer& buffer) override;

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      SoSh8Type& element_type() const override { return SoSh8Type::instance(); }

      //@}

      //! @name Input and Creation

      /*!
      \brief Read input for this element
      */
      bool read_element(const std::string& eletype, const std::string& distype,
          const Core::IO::InputParameterContainer& container) override;



      //@}

      //! @name Evaluation

      /*!
      \brief Evaluate an element

      Evaluate so_sh8 element stiffness, mass, internal forces, etc.

      \param params (in/out): ParameterList for communication between control routine
                              and elements
      \param discretization : pointer to discretization for de-assembly
      \param lm (in)        : location matrix for de-assembly
      \param elemat1 (out)  : (stiffness-)matrix to be filled by element. If nullptr on input,
                              the controlling method does not expect the element to fill
                              this matrix.
      \param elemat2 (out)  : (mass-)matrix to be filled by element. If nullptr on input,
                              the controlling method does not expect the element to fill
                              this matrix.
      \param elevec1 (out)  : (internal force-)vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \param elevec2 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \param elevec3 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \return 0 if successful, negative otherwise
      */
      int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

      //! definition of shell-thickness direction
      enum ThicknessDirection
      {
        undefined,  ///< no clear direction identified
        globx,      ///< global x
        globy,      ///< global y
        globz,      ///< global z
        autoj,      ///< find automatically by Jacobian
        author,     ///< automatically set to x
        autos,      ///< automatically set to y
        autot,      ///< automatically set to z
        enfor,      ///< read-in r-direction is rearranged to t-dir
        enfos,      ///< read-in s-direction is rearranged to t-dir
        enfot,      ///< read-in t-direction stays t-dir
        none        ///< no rearrangement
      };

      enum ANSType
      {
        anssosh8,
        ansnone
      };

     private:
      // don't want = operator
      SoSh8& operator=(const SoSh8& old);

     protected:
      static constexpr int num_sp = 8;  ///< number of ANS sampling points, here 8
      static constexpr int num_ans =
          3;  ///< number of modified ANS strains (E_rt,E_st,E_tt), here 3
      //! shell-thickness direction
      ThicknessDirection thickdir_;

      ANSType anstype_;

      //! in case of changed "thin" direction this is true
      bool nodes_rearranged_;

      //! vector in thickness direction for compatibility with sosh8
      std::vector<double> thickvec_;

      //! Compute stiffness and mass matrix
      void sosh8_nlnstiffmass(std::vector<int>& lm,  ///< location matrix
          std::vector<double>& disp,                 ///< current displacements
          std::vector<double>& residual,             ///< current residual displ
          Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8>*
              stiffmatrix,                                             ///< element stiffness matrix
          Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8>* massmatrix,  ///< element mass matrix
          Core::LinAlg::Matrix<NUMDOF_SOH8, 1>* force,      ///< element internal force vector
          Core::LinAlg::Matrix<NUMDOF_SOH8, 1>* force_str,  // element structural force vector
          Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D>* elestress,  ///< stresses at GP
          Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D>* elestrain,  ///< strains at GP
          Teuchos::ParameterList& params,            ///< algorithmic parameters e.g. time
          const Inpar::Solid::StressType iostress,   ///< stress output option
          const Inpar::Solid::StrainType iostrain);  ///< strain output option

      //! Evaluate all ANS related data at the ANS sampling points
      void sosh8_anssetup(
          const Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8>& xrefe,  ///< material element coords
          const Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8>& xcurr,  ///< current element coords
          std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>>**
              deriv_sp,  ///< derivs eval. at all sampling points
          std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>>&
              jac_sps,  ///< jac at all sampling points
          std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>>&
              jac_cur_sps,  ///< current jac at all sampling points
          Core::LinAlg::Matrix<num_ans * num_sp, NUMDOF_SOH8>& B_ans_loc) const;  ///< modified B

      //! Evaluate transformation matrix T (parameter->material) at gp
      void sosh8_evaluate_t(
          const Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>& jac,  ///< actual jacobian
          Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& TinvT);  ///< T^{-T}

      //! Return true Cauchy-stress at gausspoint
      void sosh8_cauchy(Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D>* elestress,
          const int gp, const Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>& defgrd,
          const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& glstrain,
          const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& stress);

      //! Find "thin"=thickness direction
      ThicknessDirection sosh8_findthickdir();

      //! Find aspect ratio of the element
      double sosh8_calcaspectratio();

      //! Calculate the STC matrix
      virtual void do_calc_stc_matrix(Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8>& elemat1,
          const Inpar::Solid::StcScale stc_scaling, const int stc_layer, std::vector<int>& lm,
          Core::FE::Discretization& discretization, bool calcinverse);

      //! Find parametric co-ordinate which directs in enforced thickness direction
      ThicknessDirection sosh8_enfthickdir(Core::LinAlg::Matrix<NUMDIM_SOH8, 1>&
              thickdirglo  ///< global direction of enforced thickness direction
      );

      //! return thickness direction
      std::vector<double> get_thickvec() { return thickvec_; };

      //! Debug gmsh-plot to check thickness direction
      void sosh8_gmshplotlabeledelement(const int LabelIds[NUMNOD_SOH8]);

      std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> sosh8_derivs_sdc();

      /** \brief Evaluate the reference and current jacobian as well as the
       *  respective determinants */
      bool sosh8_evaluatejacobians(const unsigned gp,
          const std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>>& derivs,
          const Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8>& xrefe,
          const Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8>& xcurr,
          Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>& jac_ref, double& detJ_ref,
          Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>& jac_curr, double& detJ_curr) const;

      /** \brief evaluate the jacobian and the determinant for the given GP */
      void sosh8_evaluatejacobian(const unsigned gp,
          const std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>>& derivs,
          const Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8>& x,
          Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>& jac, double& detJ) const;

      /** \brief Get the local B-operator */
      void sosh8_get_bop_loc(const unsigned gp,
          const std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>>& derivs,
          const Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>& jac_curr, const double* r,
          const double* s, const Core::LinAlg::Matrix<num_ans * num_sp, NUMDOF_SOH8>& B_ans_loc,
          Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, NUMDOF_SOH8>& bop_loc) const;

      /** \brief Get the local green lagrange strain */
      void sosh8_get_glstrain_loc(const unsigned gp,
          const Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>& jac_curr,
          const Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>& jac,
          const std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>>& jac_sps,
          const std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>>& jac_cur_sps,
          const double* r, const double* s,
          Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& lstrain) const;

      void sosh8_get_deformationgradient(const unsigned gp,
          const std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>>& derivs,
          const Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8>& xcurr,
          const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& glstrain,
          Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>& defgrd) const;

      double sosh8_calc_energy(const std::vector<double>& disp, Teuchos::ParameterList& params);

      double sosh8_third_invariant(
          const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& glstrain) const;

     private:
      std::string get_element_type_string() const { return "SOLIDSH8"; }
    };  // class So_sh8



  }  // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif

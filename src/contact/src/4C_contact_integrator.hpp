// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_INTEGRATOR_HPP
#define FOUR_C_CONTACT_INTEGRATOR_HPP

#include "4C_config.hpp"

#include "4C_inpar_wear.hpp"
#include "4C_mortar_integrator.hpp"
#include "4C_utils_pairedvector.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SerialDenseVector;
}
namespace Mortar
{
  class ParamsInterface;
}

namespace CONTACT
{
  // forward declaration
  class ParamsInterface;
  /*!
   \brief A class to perform Gaussian integration and assembly of Mortar
   matrices on the overlap of two Mortar::Elements (1 Slave, 1 Master)
   in 1D (which is equivalent to a 2D coupling problem) and in 2D
   (which is equivalent to a 3D coupling problem).

   This is a derived class from Mortar::Integrator which does
   the contact-specific stuff for 3d mortar coupling.

   */
  class Integrator
  {
   public:
    /*!
     \brief Constructor  with shape function specification

     Constructs an instance of this class using a specific type of shape functions.<br>
     Note that this is \b not a collective call as overlaps are
     integrated in parallel by individual processes.<br>
     Note also that this constructor relies heavily on the
     Core::FE::IntegrationPoints structs to get Gauss points
     and corresponding weights.

     */
    Integrator(Teuchos::ParameterList& params, Core::FE::CellType eletype, MPI_Comm comm);

    /*!
     \brief Destructor

     */
    virtual ~Integrator() = default;

    //! don't want = operator
    Integrator operator=(const Integrator& old) = delete;
    //! don't want copy constructor
    Integrator(const Integrator& old) = delete;

    //! get specified integration type
    inline Mortar::IntType integration_type() const { return integrationtype_; }

    MPI_Comm get_comm() const { return Comm_; }

    //! @name 2D and 3D integration methods

    /*!
     \brief check for boundary segmentation in 2D

     */
    bool boundary_segm_check_2d(Mortar::Element& sele, std::vector<Mortar::Element*> meles);

    /*!
     \brief check for boundary segmentation in 2D

     */
    bool boundary_segm_check_3d(Mortar::Element& sele, std::vector<Mortar::Element*> meles);


    /*!
     \brief Build all integrals and linearizations without segmentation -- 2D
     (i.e. M, g, LinM, Ling and possibly D, LinD)

     */
    virtual void integrate_deriv_ele_2d(Mortar::Element& sele, std::vector<Mortar::Element*> meles,
        bool* boundary_ele, const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr);
    virtual void integrate_deriv_ele_2d(Mortar::Element& sele, std::vector<Mortar::Element*> meles,
        bool* boundary_ele, const std::shared_ptr<CONTACT::ParamsInterface>& cparams_ptr);

    /*!
     \brief integrate D matrix without lin...

     */
    void integrate_d(Mortar::Element& sele, MPI_Comm comm, bool lin = false);

    /*!
     \brief Build all integrals and linearizations on a 1D slave /
     master overlap (i.e. M, g, LinM, Ling and possibly D, LinD and
     wear)

     */
    virtual void integrate_deriv_segment_2d(Mortar::Element& sele, double& sxia, double& sxib,
        Mortar::Element& mele, double& mxia, double& mxib, MPI_Comm comm,
        const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr);
    virtual void integrate_deriv_segment_2d(Mortar::Element& sele, double& sxia, double& sxib,
        Mortar::Element& mele, double& mxia, double& mxib, MPI_Comm comm,
        const std::shared_ptr<CONTACT::ParamsInterface>& cparams_ptr);

    /*!
     \brief Build all integrals and linearizations without segmentation -- 3D
     (i.e. M, g, LinM, Ling and possibly D, LinD)

     */
    virtual void integrate_deriv_ele_3d(Mortar::Element& sele, std::vector<Mortar::Element*> meles,
        bool* boundary_ele, bool* proj_, MPI_Comm comm,
        const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr);
    virtual void integrate_deriv_ele_3d(Mortar::Element& sele, std::vector<Mortar::Element*> meles,
        bool* boundary_ele, bool* proj_, MPI_Comm comm,
        const std::shared_ptr<CONTACT::ParamsInterface>& cparams_ptr);

    /*!
     \brief Build all integrals and linearizations on a 2D slave /
     master integration cell (i.e. M, g, LinM, Ling and possibly D, LinD)
     for the auxiliary plane coupling case

     */
    virtual void integrate_deriv_cell_3d_aux_plane(Mortar::Element& sele, Mortar::Element& mele,
        std::shared_ptr<Mortar::IntCell> cell, double* auxn, MPI_Comm comm,
        const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr);
    virtual void integrate_deriv_cell_3d_aux_plane(Mortar::Element& sele, Mortar::Element& mele,
        std::shared_ptr<Mortar::IntCell> cell, double* auxn, MPI_Comm comm,
        const std::shared_ptr<CONTACT::ParamsInterface>& cparams_ptr);

    /*!
     \brief Build all integrals and linearizations on a 2D slave /
     master integration cell (i.e. M, g, LinM, Ling) for
     the auxiliary plane coupling case with quadratic interpolation

     */
    void integrate_deriv_cell_3d_aux_plane_quad(Mortar::Element& sele, Mortar::Element& mele,
        Mortar::IntElement& sintele, Mortar::IntElement& mintele,
        std::shared_ptr<Mortar::IntCell> cell, double* auxn);

    /*!
     \brief ....

     */
    void integrate_deriv_cell_3d_aux_plane_lts(Mortar::Element& sele, Mortar::Element& lsele,
        Mortar::Element& mele, std::shared_ptr<Mortar::IntCell> cell, double* auxn, MPI_Comm comm);

    /*!
     \brief ....

     */
    void integrate_deriv_cell_3d_aux_plane_stl(Mortar::Element& mele, Mortar::Element& lele,
        Mortar::Element& sele, std::shared_ptr<Mortar::IntCell> cell, double* auxn, MPI_Comm comm);

    /*!
     \brief Compute penalty scaling factor kappa on slave element

     */
    void integrate_kappa_penalty(
        Mortar::Element& sele, double* sxia, double* sxib, Core::LinAlg::SerialDenseVector& gseg);


    /*!
     \brief Compute penalty scaling factor kappa on slave element for LTS algorithm
            for last converged configuration

     */
    void integrate_kappa_penalty_lts(Mortar::Element& sele);

    /*!
     \brief Compute penalty scaling factor kappa on slave integration element
     (special version for the 3D quadratic case)

     */
    void integrate_kappa_penalty(Mortar::Element& sele, Mortar::IntElement& sintele, double* sxia,
        double* sxib, Core::LinAlg::SerialDenseVector& gseg);

    //@}

    //! @name 2D and 3D linearization methods

    /*!
     \brief Compute directional derivative of segment end coordinates
     Xi on a 1D slave / master overlap

     */
    void deriv_xi_a_b_2d(const Mortar::Element& sele, double sxia, double sxib,
        const Mortar::Element& mele, double mxia, double mxib,
        std::vector<Core::Gen::Pairedvector<int, double>>& derivxi, bool startslave, bool endslave,
        int linsize) const;

    /*!
     \brief Compute directional derivative of master Gauss point
     coordinates XiGP on a 1D slave / master overlap

     */
    void deriv_xi_gp_2d(const Mortar::Element& sele, const Mortar::Element& mele, double sxigp,
        double mxigp, const Core::Gen::Pairedvector<int, double>& derivsxi,
        Core::Gen::Pairedvector<int, double>& derivmxi, int linsize) const;

    /*!
     \brief Compute directional derivative of master Gauss point
     coordinates XiGP on a 2D slave / master integration cell

     */
    void deriv_xi_gp_3d(const Mortar::Element& sele, const Mortar::Element& mele,
        const double* sxigp, const double* mxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& derivsxi,
        std::vector<Core::Gen::Pairedvector<int, double>>& derivmxi, const double alpha) const;

    /*!
     \brief Compute directional derivative of slave / master Gauss point
     coordinates XiGP on a 2D slave / master integration cell
     (This is the AuxPlane version, thus master and slave are projected)

     */
    void deriv_xi_gp_3d_aux_plane(const Mortar::Element& ele, const double* xigp,
        const double* auxn, std::vector<Core::Gen::Pairedvector<int, double>>& derivxi,
        double alpha, std::vector<Core::Gen::Pairedvector<int, double>>& derivauxn,
        Core::Gen::Pairedvector<int, Core::LinAlg::Matrix<3, 1>>& derivgp) const;

    /*!
     \brief Assemble g~ contribution of current overlap into slave nodes

     */
    bool assemble_g(MPI_Comm comm, Mortar::Element& sele, Core::LinAlg::SerialDenseVector& gseg);

    /*!
     \brief Assemble g~ contribution of current overlap into slave nodes
     (special version for 3D quadratic mortar with piecewise linear LM interpolation)

     */
    bool assemble_g(
        MPI_Comm comm, Mortar::IntElement& sintele, Core::LinAlg::SerialDenseVector& gseg);

    // GP calls
    /*!
     \brief Return number of Gauss points for this instance

     */
    int n_gp() const { return ngp_; }

    /*!
     \brief Return coordinates of a specific GP in 1D/2D CElement

     */
    double coordinate(int gp, int dir) const { return coords_(gp, dir); }

    /*!
     \brief Return weight of a specific GP in 1D/2D CElement

     */
    double weight(int gp) const { return weights_[gp]; }

    /*!
     \brief Get problem dimension

     Note that only 2D and 3D are possible here as this refers to the global
     problem dimension. On integration level this corresponds to 1D integration
     (dim_==2) and 2D integration (dim_==3) on the interface!

     */
    int n_dim() const { return dim_; };

   protected:
    /*!
     \brief Initialize Gauss rule (points, weights) for this Mortar::Integrator

     */
    void initialize_gp(Core::FE::CellType eletype);

    /*!
     * @brief Perform integration at Gauss point for 3D problems.
     * This is where the distinction between methods should be, i.e. mortar, augmented, gpts,...
     *
     * @param[in] sele     current mortar slave element
     * @param[in] mele     current mortar master element
     * @param[in] sval     slave side shape function evaluated at current Gauss point
     * @param[in] lmval    Lagrangian multiplier shape function evaluated at current Gauss point
     * @param[in] mval     master side shape function evaluated at current Gauss point
     * @param[in] sderiv   slave side shape function derivative at current Gauss point
     * @param[in] mderiv   master side shape function derivative at current Gauss point
     * @param[in] lmderiv  Lagrangian multiplier shape function derivative evaluated at current
     *                     Gauss point
     * @param[in] dualmap  directional derivative of dual shape functions
     * @param[in] wgt      Gauss point weight
     * @param[in] jac           Jacobian determinant of integration cell
     * @param[in] derivjac      directional derivative of cell Jacobian
     * @param[in] normal        integration cell normal
     * @param[in] dnmap_unit    directional derivative of integration cell normal
     * @param[in] gap           gap
     * @param[in] deriv_gap     directional derivative of gap
     * @param[in] sxi       slave side Gauss point coordinates
     * @param[in] mxi       master side Gauss point coordinates
     * @param[in] derivsxi  directional derivative of slave side Gauss point coordinates
     * @param[in] derivmxi  directional derivative of master side Gauss point coordinates
     */
    virtual void integrate_gp_3d(Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& mderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, double& wgt,
        double& jac, Core::Gen::Pairedvector<int, double>& derivjac, double* normal,
        std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double& gap,
        Core::Gen::Pairedvector<int, double>& deriv_gap, double* sxi, double* mxi,
        std::vector<Core::Gen::Pairedvector<int, double>>& derivsxi,
        std::vector<Core::Gen::Pairedvector<int, double>>& derivmxi);

    /*!
     * @brief Perform integration at Gauss point for 2D problems.
     * This is where the distinction between methods should be, i.e. mortar, augmented, gpts,...
     *
     * @param[in] sele     mortar slave element
     * @param[in] mele     mortar master element
     * @param[in] sval     slave side shape function evaluated at current Gauss point
     * @param[in] lmval    Lagrangian multiplier shape function evaluated at current Gauss point
     * @param[in] mval     master side shape function evaluated at current Gauss point
     * @param[in] sderiv   slave side shape function derivative at current Gauss point
     * @param[in] mderiv   master side shape function derivative at current Gauss point
     * @param[in] lmderiv  Lagrangian multiplier shape function derivative evaluated at current
     *                     Gauss point
     * @param[in] dualmap  directional derivative of dual shape functions
     * @param[in] wgt      Gauss point weight
     * @param[in] jac           Jacobian determinant of integration cell
     * @param[in] derivjac      directional derivative of cell Jacobian
     * @param[in] normal        integration cell normal
     * @param[in] dnmap_unit    directional derivative of integration cell normal
     * @param[in] gap           gap
     * @param[in] deriv_gap     directional derivative of gap
     * @param[in] sxi       slave side Gauss point coordinates
     * @param[in] mxi       master side Gauss point coordinates
     * @param[in] derivsxi  directional derivative of slave side Gauss point coordinates
     * @param[in] derivmxi  directional derivative of master side Gauss point coordinates
     */
    virtual void integrate_gp_2d(Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& mderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, double& wgt,
        double& jac, Core::Gen::Pairedvector<int, double>& derivjac, double* normal,
        std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double& gap,
        Core::Gen::Pairedvector<int, double>& deriv_gap, double* sxi, double* mxi,
        std::vector<Core::Gen::Pairedvector<int, double>>& derivsxi,
        std::vector<Core::Gen::Pairedvector<int, double>>& derivmxi);

    /*!
     \brief evaluate D2-matrix entries at GP

     */
    void inline gp_d2(Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& lm2val, Core::LinAlg::SerialDenseVector& m2val,
        double& jac, double& wgt, MPI_Comm comm);

    /*!
     \brief evaluate D/M-matrix entries at GP

     */
    void gp_dm(Mortar::Element& sele, Mortar::Element& mele, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& mval, double& jac,
        double& wgt, bool& bound);

    /*!
     \brief evaluate D/M-matrix entries at GP (3D quadratic)

     */
    void inline gp_3d_dm_quad(Mortar::Element& sele, Mortar::Element& mele,
        Mortar::IntElement& sintele, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseVector& lmintval, Core::LinAlg::SerialDenseVector& sval,
        Core::LinAlg::SerialDenseVector& mval, const double& jac, double& wgt, const int& nrow,
        const int& nintrow, const int& ncol, const int& ndof, bool& bound);

    /*!
     \brief lin D/M-matrix entries at GP for bound case

     */
    void inline gp_2d_dm_lin_bound(Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& mval,
        Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& mderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
        double& jac, double& wgt, const Core::Gen::Pairedvector<int, double>& derivjac,
        std::vector<Core::Gen::Pairedvector<int, double>>& derivsxi,
        std::vector<Core::Gen::Pairedvector<int, double>>& derivmxi,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);

    /*!
     \brief lin D/M-matrix entries at GP for bound case

     */
    void inline gp_2d_dm_lin(int& iter, bool& bound, bool& linlm, Mortar::Element& sele,
        Mortar::Element& mele, Core::LinAlg::SerialDenseVector& sval,
        Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseMatrix& sderiv, Core::LinAlg::SerialDenseMatrix& mderiv,
        Core::LinAlg::SerialDenseMatrix& lmderiv, double& jac, double& wgt,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dmxigp,
        const Core::Gen::Pairedvector<int, double>& derivjac,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);

    /*!
     \brief lin D/M-matrix entries at GP for elebased integration

     */
    void inline gp_2d_dm_ele_lin(int& iter, bool& bound, Mortar::Element& sele,
        Mortar::Element& mele, Core::LinAlg::SerialDenseVector& sval,
        Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseMatrix& mderiv, double& dxdsxi, double& wgt,
        const Core::Gen::Pairedvector<int, double>& dmxigp,
        const Core::Gen::Pairedvector<int, double>& derivjac,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);

    /*!
     \brief lin D/M-matrix entries at GP

     */
    void gp_3d_dm_lin(Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& mval,
        Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& mderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
        double& wgt, double& jac, std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        std::vector<Core::Gen::Pairedvector<int, double>>& dmxigp,
        Core::Gen::Pairedvector<int, double>& jacintcellmap,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);

    /*!
     \brief lin D/M-matrix entries at GP for bound case

     */
    void inline gp_3d_dm_lin_bound(Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& mval,
        Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& lmderiv, Core::LinAlg::SerialDenseMatrix& mderiv,
        double& jac, double& wgt, const Core::Gen::Pairedvector<int, double>& derivjac,
        std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        std::vector<Core::Gen::Pairedvector<int, double>>& dmxigp,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);

    /*!
     \brief lin D/M-matrix entries at GP for bound case (3D quad)

     */
    void inline gp_3d_dm_quad_lin(bool& duallin, Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& svalmod,
        Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseMatrix& sderiv, Core::LinAlg::SerialDenseMatrix& mderiv,
        Core::LinAlg::SerialDenseMatrix& lmderiv, double& wgt, double& jac,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dpsxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dpmxigp,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap,
        bool dualquad3d);

    void inline gp_3d_dm_quad_pwlin_lin(int& iter, Mortar::Element& sele, Mortar::Element& sintele,
        Mortar::Element& mele, Core::LinAlg::SerialDenseVector& sval,
        Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseVector& lmintval,
        Core::LinAlg::SerialDenseMatrix& sderiv, Core::LinAlg::SerialDenseMatrix& mderiv,
        Core::LinAlg::SerialDenseMatrix& lmintderiv, double& wgt, double& jac,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dpsxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dpmxigp,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap);

    /*!
     \brief evaluate weighted Gap entries at GP

     */
    void gp_3d_w_gap(Mortar::Element& sele, Core::LinAlg::SerialDenseVector& sval,
        Core::LinAlg::SerialDenseVector& lmval, double* gap, double& jac, double& wgt,
        bool quadratic, int nintrow = 0);

    /*!
     \brief evaluate weighted Gap entries at GP

     */
    void inline gp_2d_w_gap(Mortar::Element& sele, Core::LinAlg::SerialDenseVector& sval,
        Core::LinAlg::SerialDenseVector& lmval, double* gap, double& jac, double& wgt);

    /*!
     \brief evaluate geometrical gap at GP
     */
    void gap_3d(Mortar::Element& sele, Mortar::Element& mele, Core::LinAlg::SerialDenseVector& sval,
        Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& mderiv, double* gap, double* gpn,
        std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        std::vector<Core::Gen::Pairedvector<int, double>>& dmxigp,
        Core::Gen::Pairedvector<int, double>& dgapgp,
        std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit);


    /*!
     \brief evaluate geometrical gap at GP
     */
    void gap_2d(Mortar::Element& sele, Mortar::Element& mele, Core::LinAlg::SerialDenseVector& sval,
        Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& mderiv, double* gap, double* gpn,
        std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        std::vector<Core::Gen::Pairedvector<int, double>>& dmxigp,
        Core::Gen::Pairedvector<int, double>& dgapgp,
        std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit);

    void inline gp_2d_g_lin(int& iter, Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& mval,
        Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& lmderiv, double& gap, double* gpn, double& jac,
        double& wgt, Core::Gen::Pairedvector<int, double>& dgapgp,
        Core::Gen::Pairedvector<int, double>& jacintcellmap,
        std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);

    /*!
     \brief evaluate weighted Gap entries at GP (quad-pwlin)

     */
    void inline gp_3d_g_quad_pwlin(Mortar::Element& sele, Mortar::IntElement& sintele,
        Mortar::Element& mele, Core::LinAlg::SerialDenseVector& sval,
        Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseVector& lmintval,
        Core::LinAlg::SerialDenseMatrix& scoord, Core::LinAlg::SerialDenseMatrix& mcoord,
        Core::LinAlg::SerialDenseMatrix& sderiv, Core::LinAlg::SerialDenseMatrix& mderiv,
        double* gap, double* gpn, double* lengthn, double& jac, double& wgt,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dmxigp,
        Core::Gen::Pairedvector<int, double>& dgapgp,
        std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit);

    /*!
     \brief evaluate weighted Gap entries at GP

     */
    void gp_g_lin(int& iter, Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& mval,
        Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& lmderiv, double& gap, double* gpn, double& jac,
        double& wgt, Core::Gen::Pairedvector<int, double>& dgapgp,
        Core::Gen::Pairedvector<int, double>& jacintcellmap,
        std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);

    /*!
     \brief evaluate weighted Gap entries at GP (quad)

     */
    void inline gp_3d_g_quad_lin(int& iter, Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& svalmod,
        Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& lmderiv, double& gap, double* gpn, double& jac,
        double& wgt, bool& duallin, const Core::Gen::Pairedvector<int, double>& dgapgp,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dpsxigp,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap,
        bool dualquad3d);

    /*!
     \brief evaluate weighted Gap entries at GP (quad)

     */
    void inline gp_3d_g_quad_pwlin_lin(int& iter, Mortar::IntElement& sintele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmintval,
        Core::LinAlg::SerialDenseMatrix& sderiv, Core::LinAlg::SerialDenseMatrix& lmintderiv,
        double& gap, double* gpn, double& jac, double& wgt,
        const Core::Gen::Pairedvector<int, double>& dgapgp,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp);

    /*!
     \brief evaluate and lin slipincr at GP

     */
    void inline gp_2d_slip_incr(Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& mval,
        Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& mderiv, double& jac, double& wgt, double* jumpvalv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dmxigp,
        Core::Gen::Pairedvector<int, double>& dslipgp, int& linsize);

    /*!
     \brief evaluate and lin slipincr at GP

     */
    void inline gp_3d_slip_incr(Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& mval,
        Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& mderiv, double& jac, double& wgt, double* jumpvalv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dmxigp,
        std::vector<Core::Gen::Pairedvector<int, double>>& dslipgp);

    /*!
     \brief evaluate and lin slipincr at GP at node

     */
    void inline gp_2d_slip_incr_lin(int& iter, Mortar::Element& sele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseMatrix& sderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
        double& jac, double& wgt, double* jumpvalv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const Core::Gen::Pairedvector<int, double>& dslipgp,
        const Core::Gen::Pairedvector<int, double>& derivjac,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);

    void inline gp_3d_slip_incr_lin(int& iter, Mortar::Element& sele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseMatrix& sderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
        double& jac, double& wgt, double* jumpvalv,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dslipgp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);
    /*!
     \brief evaluate  T and E matrix

     */
    void inline gp_te(Mortar::Element& sele, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseVector& sval, double& jac, double& wgt, double* jumpval);

    /*!
     \brief evaluate  T and E matrix

     */
    void inline gp_te_master(Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseVector& lm2val,
        Core::LinAlg::SerialDenseVector& mval, double& jac, double& wgt, double* jumpval,
        MPI_Comm comm);

    /*!
     \brief evaluate Lin T and E matrix

     */
    void inline gp_2d_te_lin(int& iter, Mortar::Element& sele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseMatrix& sderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
        double& jac, double& wgt, double* jumpval,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const Core::Gen::Pairedvector<int, double>& derivjac,
        const Core::Gen::Pairedvector<int, double>& dsliptmatrixgp,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);

    /*!
     \brief evaluate Lin T and E matrix

     */
    void inline gp_2d_te_master_lin(int& iter,  // like k
        Mortar::Element& sele, Mortar::Element& mele, Core::LinAlg::SerialDenseVector& sval,
        Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseMatrix& mderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
        double& dsxideta, double& dxdsxi, double& dxdsxidsxi, double& wgt, double* jumpval,
        const Core::Gen::Pairedvector<int, double>& dsxigp,
        const Core::Gen::Pairedvector<int, double>& dmxigp,
        const Core::Gen::Pairedvector<int, double>& derivjac,
        const Core::Gen::Pairedvector<int, double>& dsliptmatrixgp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& ximaps,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap,
        MPI_Comm comm);

    /*!
     \brief evaluate Lin T and E matrix

     */
    void inline gp_3d_te_lin(int& iter, Mortar::Element& sele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseMatrix& sderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
        double& jac, double& wgt, double* jumpval,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap,
        const Core::Gen::Pairedvector<int, double>& dsliptmatrixgp,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);

    /*!
     \brief evaluate Lin T and E matrix (Master)

     */
    void inline gp_3d_te_master_lin(int& iter, Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& mval,
        Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseVector& lm2val,
        Core::LinAlg::SerialDenseMatrix& sderiv, Core::LinAlg::SerialDenseMatrix& mderiv,
        Core::LinAlg::SerialDenseMatrix& lmderiv, Core::LinAlg::SerialDenseMatrix& lm2deriv,
        double& jac, double& wgt, double* jumpval,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dmxigp,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap,
        const Core::Gen::Pairedvector<int, double>& dsliptmatrixgp,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dual2map,
        MPI_Comm comm);

    /*!
     \brief evaluate wear + lin at GP

     */
    void inline gp_2d_wear(Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& mderiv,
        Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseMatrix& lmderiv,
        Core::LinAlg::SerialDenseMatrix& lagmult, double* gpn, double& jac, double& wgt,
        double* jumpval, double* wearval, Core::Gen::Pairedvector<int, double>& dsliptmatrixgp,
        Core::Gen::Pairedvector<int, double>& dweargp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dmxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);

    /*!
     \brief evaluate wear + lin at GP

     */
    void inline gp_3d_wear(Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& mderiv,
        Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseMatrix& lmderiv,
        Core::LinAlg::SerialDenseMatrix& lagmult, double* gpn, double& jac, double& wgt,
        double* jumpval, double* wearval, Core::Gen::Pairedvector<int, double>& dsliptmatrixgp,
        Core::Gen::Pairedvector<int, double>& dweargp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dmxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);

    /*!
     \brief lin weighted wear at GP

     */
    void inline gp_2d_wear_lin(int& iter, Mortar::Element& sele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseMatrix& sderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
        double& jac, double* gpn, double& wgt, double& wearval, double* jumpval,
        const Core::Gen::Pairedvector<int, double>& dweargp,
        const Core::Gen::Pairedvector<int, double>& derivjac,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);

    /*!
     \brief lin weighted wear at GP

     */
    void inline gp_3d_wear_lin(int& iter, Mortar::Element& sele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseMatrix& sderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
        double& jac, double* gpn, double& wgt, double& wearval, double* jumpval,
        const Core::Gen::Pairedvector<int, double>& dweargp,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);


    /*!
    \brief evaluate scalar normal coupling condition for poro no penetration entries at GP
    (poro-contact)

    */
    void inline gp_ncoup_deriv(Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& mval,
        Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& mderiv, double* ncoup, double* gpn, double& jac,
        double& wgt, double* gpcoord,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dmxigp,
        std::map<int, double>& dncoupgp, std::map<int, double>& dvelncoupgp,
        std::map<int, double>& dpresncoupgp,
        std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, bool quadratic,
        int nintrow = 0);

    /*!
    \brief evaluate weighted normal coupling entries at GP

    */
    void inline gp_ncoup_lin(int& iter, Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& mval,
        Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& lmderiv, double& ncoup, double* gpn, double& jac,
        double& wgt, const std::map<int, double>& dncoupgp,
        const std::map<int, double>& dvelncoupgp, const std::map<int, double>& dpresncoupgp,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dmxigp,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap);

    /*!
    \brief Calculate Determinate of the Deformation Gradient at GP

    */
    double det_deformation_gradient(
        Mortar::Element& sele, double& wgt, double* gpcoord, std::map<int, double>& JLin);

    /*!
    \brief Templated Calculate Determinate of the Deformation Gradient at GP

    */
    template <Core::FE::CellType parentdistype, int dim>
    double t_det_deformation_gradient(
        Mortar::Element& sele, double& wgt, double* gpcoord, std::map<int, double>& JLin);

    /*!
     \brief Return the Wear shape fcn type (wear weighting...)

     */
    Inpar::Wear::WearShape wear_shape_fcn() { return wearshapefcn_; }

    /*!
     \brief Return type of wear surface definition

     */
    Inpar::Wear::WearSide wear_side() { return wearside_; }

    /*!
     \brief Return type of wear algorithm

     */
    Inpar::Wear::WearType wear_type() { return weartype_; }

    /*!
     \brief Return the LM shape fcn type

     */
    Mortar::ShapeFcn shape_fcn() { return shapefcn_; }

    /*!
     \brief Return the LM interpolation / testing type for quadratic FE

     */
    Mortar::LagMultQuad lag_mult_quad() { return lagmultquad_; }
    //@}

    //! containing contact input parameters
    Teuchos::ParameterList& imortar_;
    //! communicator
    MPI_Comm Comm_;

    //! number of Gauss points
    int ngp_;
    //! Gauss point coordinates
    Core::LinAlg::SerialDenseMatrix coords_;
    //! Gauss point weights
    std::vector<double> weights_;
    //! dimension of problem (2D or 3D)
    int dim_;

    // inputs from parameter list
    //! lm shape function type
    Mortar::ShapeFcn shapefcn_;
    //! type of lm interpolation for quadr. FE
    Mortar::LagMultQuad lagmultquad_;
    //! gp-wise evaluated slip increment
    bool gpslip_;
    //! contact algorithm
    Mortar::AlgorithmType algo_;
    //! solution stratety
    CONTACT::SolvingStrategy stype_;
    //! flag for closest point normal -> change in linsize
    bool cppnormal_;

    // wear inputs from parameter list
    //! type of wear law
    Inpar::Wear::WearLaw wearlaw_;
    //! flag for implicit wear algorithm
    bool wearimpl_;
    //! definition of wear surface
    Inpar::Wear::WearSide wearside_;
    //! definition of contact wear algorithm
    Inpar::Wear::WearType weartype_;
    //! type of wear shape function
    Inpar::Wear::WearShape wearshapefcn_;
    //! flag for steady state wear
    bool sswear_;
    //! wear coefficient
    double wearcoeff_;
    //! wear coefficient master
    double wearcoeffm_;
    //! fixed slip for steady state wear
    double ssslip_;

    //! flag for non-smooth contact
    bool nonsmooth_;
    //! flag is true if (self) contact surface is non-smooth
    const bool nonsmoothselfcontactsurface_;

   private:
    //! integration type from the parameter-list
    Mortar::IntType integrationtype_;
  };  // class Integrator
}  // namespace CONTACT


FOUR_C_NAMESPACE_CLOSE

#endif

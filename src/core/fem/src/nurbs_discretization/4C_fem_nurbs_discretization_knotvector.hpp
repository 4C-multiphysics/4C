// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_NURBS_DISCRETIZATION_KNOTVECTOR_HPP
#define FOUR_C_FEM_NURBS_DISCRETIZATION_KNOTVECTOR_HPP

#include "4C_config.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class InputParameterContainer;
}

namespace Core::FE
{
  namespace Nurbs
  {
    class KnotvectorObjectType : public Core::Communication::ParObjectType
    {
     public:
      std::string name() const override { return "KnotvectorObjectType"; }

      static KnotvectorObjectType& instance() { return instance_; };

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

     private:
      static KnotvectorObjectType instance_;
    };

    /*!
    \brief A class to manage a nurbs knotvector (isogeometric analysis)

    The class contains the data structures + some service functions, e.g.
    - do we have interpolation?
    - is it an open knot vector or periodic knot?
    - some access methods
    - ...

    ParObject is implemented to be able to write the knotvector to disc for I/O.

    */
    class Knotvector : public Core::Communication::ParObject
    {
     public:
      //! Knotvector types
      enum class KnotvectorType
      {
        Undefined,
        Interpolated,
        Periodic,
      };

      /*!
      \brief standard constructor

      \param  dim (in) Spatial dimension of the knotspan
      \param  npatches (in) Number of patches
       */
      Knotvector(const int dim, const int npatches);

      /*!
      \brief empty constructor

       */
      Knotvector();


      //! @name Access methods for shapefunction evaluation
      //! @{

      /*!
      \brief convert an element gid to its corresponding triple knot index

      Given a global element id, this method will return the base indices of the elements 'local'
      knotspans

      \param gid (i) Global element id
      \param npatch (o) The number/ID of the patch this element belongs to.
      \param loc_cart_id (o) patch local cartesian base indices for knotvectors, elements and
             control points

      \note This method heavily relies on the cartesian structure of the knotspan. We assume a
      element/control point numbering as follows:

      \verbatim

            patch local cartesian element index:

               (num_u,num_v,num_w)

      gid = patchoffset+num_u+num_v*nele                 (2d)

      gid = patchoffset+num_u+num_v*nele+num_w*nele*mele (3d)

      \endverbatim

      \note Here, all elements in a patch are assumed to be numbered consecutively and patchoffset
      is the number of all elements from previous patches.

      Example: Element gid=7, linear element, 2D, npatch 1

                     domain size: nele=6,mele=2

                     The numbers in the picture are control
         point gids

      \verbatim

             v ^
               |
                  +----------------+
              24  | 25    26    27 |  28     29    30    31
                  |                |
              16  | 17    18    19 |  20     21    22    23
                  |                |
               8  |  9    10    11 |  12     13    14    15
                  +----------------+
               0     1     2     3     4      5     6     7  ->
                                                              u
         will return (1,1)

      \endverbatim

      The numbering of the control points is assumed to be accordingly.
      */
      void convert_ele_gid_to_knot_ids(
          const int gid, int& npatch, std::vector<int>& loc_cart_id) const;


      /*!
      \brief convert an element local id + patch number to its corresponding gid

      \param  npatch (i)
              The number of the patch this element
              belongs to.

      \param  loc_cart_id (i)
              patch local cartesian base indices
              for knotvectors, elements and control points

      \return gid
              given the base indices of the elements 'local' knotspans and the id
              of the local knotspan, i.e. the patch number, a global element id
              will be returned


      \note This method heavily relies on the
            cartesian structure of the knotspan.
            We assume a element/control point numbering
      as follows:

      \verbatim

            patch local cartesian element index:

               (num_u,num_v,num_w)

      gid = patchoffset+num_u+num_v*nele                 (2d)

      gid = patchoffset+num_u+num_v*nele+num_w*nele*mele (3d)

      \endverbatim

      \note This is the inverse method to convert_ele_gid_to_knot_ids. See its documentation for
      further details.

      \sa convert_ele_gid_to_knot_ids()
      */
      int convert_ele_knot_ids_to_gid(const int& npatch, const std::vector<int>& loc_cart_id);


      /*!
      \brief get element knot vectors to a given element id

      This method will be called before any shapefunction evaluation

      \param eleknots (o) the element local knotvector to the given a global element id
      \param gid (i) given a global element id

      \note  This method heavily relies on the
             cartesian structure of the knotspan
             and the consecutive order of elements/control
             points in the patches.

       Example: Element gid=7, linear element, 2D
                      The numbers in the picture are control point gids (see above)

     \return bool zero_size
             If the integration elements size in the knotspace
             is zero, this value will be returned true (and
             otherwise false)
             Zero sized elements are simply to be skipped during
             integration.

        \verbatim

        knots_[0]

        ||----|----|----|----|----|----|----||


        knots_[1]

        ||----|----|----||

        base index (1,1) will allow to access the local knotspan

        eleknots[0]

              |----|----|----|----|


        eleknots[1]

         |----|----|----||


        \endverbatim

      */
      bool get_ele_knots(std::vector<Core::LinAlg::SerialDenseVector>& eleknots, int gid) const;



      /*!
      \brief extract element surface's knot vectors out of the
             knot vector of the parent element. On the fly, get
             orientation of normal vector.

      \param eleknots  (o) knot vector of parent element
      \param surfknots (o) extracted knot vector of surface element
      \param normalfac (o) orientation of normal vector
      \param pgid      (i) global id of parent element
      \param surfaceid (i) element local id of surface

      \return bool zero_size

      If the integration elements size in the knotspace
      is zero, this value will be returned true (and
      otherwise false)
      Zero sized elements are simply to be skipped during
      integration
      */
      bool get_boundary_ele_and_parent_knots(std::vector<Core::LinAlg::SerialDenseVector>& eleknots,
          std::vector<Core::LinAlg::SerialDenseVector>& surfknots, double& normalfac, int pgid,
          const int surfaceid) const;


      //! @}

      //! @name Insert methods

      /*!
      \brief set knots for a patch in one direction

      \param direction (in) the direction to which the knotvector corresponds
      \param npatch (in) the number of the patch this knotvector belongs to
      \param degree (in) the degree of the bspline polynomial belonging to this knot vector
      \param numknots (in) the number knots that will be added here
      \param knotvectortype (in) specifies whether we add a periodic or interpolating knot vector
      \param directions_knots (in) the knotvector to be inserted
      */
      void set_knots(const int& direction, const int& npatch, const int& degree,
          const int& numknots, KnotvectorType knotvectortype,
          const std::vector<double>& directions_knots);

      //! @}

      //! @name Checks

      /*!
      \brief finish

      counting the number of knots added, doing consistency checks for size and periodicity

      Calculate offset arrays for patches for access methods.

      An unfinished knotvector can not be accessed!

      \param smallest_gid_in_dis (i) Global ID of the first element to be stored in the offset array
      */
      void finish_knots(const int smallest_gid_in_dis);

      //! @}

      //! @name Pack/Unpack for io --- implementation of ParObject's virtual classes
      //! @{

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined in the
      ParObject header file
      */
      int unique_par_object_id() const override
      {
        return KnotvectorObjectType::instance().unique_par_object_id();
      }

      /*!
      \brief Pack this class so it can be communicated

      \ref pack and \ref unpack are used to communicate this class

      */
      void pack(Core::Communication::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref pack and \ref unpack are used to communicate this class

      */
      void unpack(Core::Communication::UnpackBuffer& buffer) override;

      //! @}

      //! @name variable access methods

      /*!
      \brief Return the degree of the nurbs patch

      \param npatch (in) The number of the patch
      */
      virtual std::vector<int> return_degree(const int npatch)
      {
        if (!filled_)
        {
          FOUR_C_THROW("can't access data. knotvector not completed\n");
        }
        return (degree_[npatch]);
      };

      /*!
      \brief Return the knot sizes of the nurbs patch

      \param npatch (i) The number of the patch
      */
      virtual std::vector<int> return_n_x_m_x_l(const int npatch)
      {
        if (!filled_)
        {
          FOUR_C_THROW("can't access data. knotvector not completed\n");
        }
        return (n_x_m_x_l_[npatch]);
      };


      /*!
      \brief Return number of zero sized elements in each direction of knotspan of this patch

      \return vector number of zero sized elements in knotspan of this patch
      */
      virtual std::vector<int> return_n_zerosize_ele(const int npatch);


      /*!
      \brief Return the global id of the next nonzero sized element in the knotspan

      This method is required for postprocessing only, where
      we visualize dummy elements as a replacement for zero-sized
      elements (to maintain the connectivity and the cartesian
      knot-span structure)

      \return gid of the next nonzero sized element in knotspan
      */
      virtual int return_next_nonzero_ele_gid(const int zero_ele_gid);


      /*!
      \brief Return the element sizes of the nurbs patch

      \param npatch (i) The number of the patch
      */
      virtual std::vector<int> return_nele_x_mele_x_lele(const int npatch)
      {
        if (!filled_)
        {
          FOUR_C_THROW("can't access data. knotvector not completed\n");
        }
        return (nele_x_mele_x_lele_[npatch]);
      };


      /*!
      \brief Return the element offsets of nurbs patches

      \return offsets array
      */
      virtual std::vector<int> return_offsets()
      {
        if (!filled_)
        {
          FOUR_C_THROW("can't access data. knotvector not completed\n");
        }
        return (offsets_);
      };

      /*!
      \brief Return the id of the patch containing the actual global node

      \param gid (i) global node id

      \return npatch  --- The number of the patch
      */
      int return_patch_id(const int gid) const
      {
        // gid is at least in patch 0 (or higher)
        int npatch = 0;

        for (int np = 1; np < npatches_; ++np)
        {
          // if this is true, gid is in this patch (or higher)
          if (gid >= offsets_[np])
          {
            npatch++;
          }
          else
          {
            break;
          }
        }
        return (npatch);
      };

      /*!
      \brief Return the number of patches

      \return number of patches
      */
      virtual int return_np()
      {
        if (!filled_)
        {
          FOUR_C_THROW("can't access data. knotvector not completed\n");
        }
        return (npatches_);
      };

      //! @}

      /*!
      \brief Print knot vector to given output stream

      @param[in] os Output stream to be used for printing
      */
      void print(std::ostream& os) const;

      /**
       * Return the InputSpec containing the parameters needed to create a Knotvector.
       */
      [[nodiscard]] static Core::IO::InputSpec spec();

      /**
       * Create a Knotvector from the given input @p data. The data is expected to match the spec().
       */
      [[nodiscard]] static Knotvector from_input(const Core::IO::InputParameterContainer& data);

     private:
      //! Spatial dimension
      int dim_;

      //! number of patches
      int npatches_;

      //! indicates that knots are ready for access
      bool filled_;

      /*! @name
       *
       * We use nested std::vectors to represent the data:
       * - The outer-most std::vector refers to the patches
       * - The second outer-most std::vector refers to the directions n,m,l
       * - The inner data structure holds the actual values
       */
      //! @{

      //! degree of bspline-polynomials defined on this knotvector
      std::vector<std::vector<int>> degree_;

      //! number of knots in each direction
      std::vector<std::vector<int>> n_x_m_x_l_;

      //! number of elements in each direction
      std::vector<std::vector<int>> nele_x_mele_x_lele_;

      //! are the component closed or open knotvectors?
      std::vector<std::vector<KnotvectorType>> interpolation_;

      //! number of elements in each direction
      std::vector<int> offsets_;

      //! the actual values
      std::vector<std::vector<std::vector<double>>> knot_values_;

      //! @}
    };

  }  // namespace Nurbs

}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif

/*!----------------------------------------------------------------------
\file mortar_manager_base.cpp
\brief Abstract base class to control all mortar coupling

<pre>
-------------------------------------------------------------------------
                        BACI Contact library
            Copyright (2008) Technical University of Munich

Under terms of contract T004.008.000 there is a non-exclusive license for use
of this work by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library is proprietary software. It must not be published, distributed,
copied or altered in any form or any media without written permission
of the copyright holder. It may be used under terms and conditions of the
above mentioned license by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library contains and makes use of software copyrighted by Sandia Corporation
and distributed under LGPL licence. Licensing does not apply to this or any
other third party software used here.

Questions? Contact Dr. Michael W. Gee (gee@lnm.mw.tum.de)
                   or
                   Prof. Dr. Wolfgang A. Wall (wall@lnm.mw.tum.de)

http://www.lnm.mw.tum.de

-------------------------------------------------------------------------
</pre>

<pre>
Maintainer: Alexander Popp
            popp@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15238
</pre>

*----------------------------------------------------------------------*/

#include "Epetra_SerialComm.h"
#include "mortar_manager_base.H"

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 01/10|
 *----------------------------------------------------------------------*/
MORTAR::ManagerBase::ManagerBase()
{
  //**********************************************************************
  // empty constructor (this is an abstract base class)
  //**********************************************************************
  // Setup of the mortar contact library is done by a derived class. This
  // derived class is specific to the FEM code into which the mortar contact
  // library is meant to be integrated. For BACI this is realized via the
  // CONTACT::ContactManager class! There the following actions are performed:
  //**********************************************************************
  // 1) get problem dimension (2D or 3D)
  // 2) read and check contact input parameters
  // 3) read and check contact boundary conditions
  // 4) build contact interfaces
  //**********************************************************************
  // A similar process also applies to mortar meshtying libraries. Again
  // a specific derived class is needed. For BACI this is realized via the
  // CONTACT::MeshtyingManager class!
  //**********************************************************************

  // create a simple serial communicator
  comm_ = Teuchos::rcp(new Epetra_SerialComm());

  return;
}


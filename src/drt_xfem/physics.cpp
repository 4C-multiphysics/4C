/*!
\file physics.cpp

\brief contains information about physical fields

<pre>
Maintainer: Axel Gerstenberger
            gerstenberger@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15236
</pre>
*/
#ifdef CCADISCRET

#include <iostream>
#include "physics.H"
#include "../drt_lib/drt_dserror.H"

// return a string representation for each enum value in XFEM::PHYSICS::Field
std::string XFEM::PHYSICS::physVarToString(const XFEM::PHYSICS::Field var)
{
    std::string text;
    switch (var){
        case Dispx:            text = "Dispx"; break;
        case Dispy:            text = "Dispy"; break;
        case Dispz:            text = "Dispz"; break;
        case Velx:             text = "Velx "; break;
        case Vely:             text = "Vely "; break;
        case Velz:             text = "Velz "; break;
        case Temp:             text = "Temp "; break;
        case Pres:             text = "Pres "; break;
        case DiscPres:         text = "DiscPres "; break;
        case LMPLambdax:       text = "LMPLambdax"; break;
        case LMPLambday:       text = "LMPLambday"; break;
        case LMPLambdaz:       text = "LMPLambdaz"; break;
        case Tauxx:            text = "Tauxx"; break;
        case Tauxy:            text = "Tauxy"; break;
        case Tauxz:            text = "Tauxz"; break;
        case Tauyx:            text = "Tauyx"; break;
        case Tauyy:            text = "Tauyy"; break;
        case Tauyz:            text = "Tauyz"; break;
        case Tauzx:            text = "Tauzx"; break;
        case Tauzy:            text = "Tauzy"; break;
        case Tauzz:            text = "Tauzz"; break;
        default:
          std::cout << var << std::endl;
          dserror("no string defined for Field");
    };
    return text;
}

#endif  // #ifdef CCADISCRET

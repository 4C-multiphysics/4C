/*----------------------------------------------------------------------*/
/*! \file

\brief contains base class for a generic output filter (ensight and vtk are derived from this
 class)


\level 0
*/


#include "post_common_writer_base.H"
#include "post_common.H"


PostWriterBase::PostWriterBase(PostField* field, const std::string& filename)
    : field_(field),
      filename_(filename),
      myrank_(field->problem()->comm()->MyPID()),
      numproc_(field->problem()->comm()->NumProc())
{
}

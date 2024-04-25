/*---------------------------------------------------------------------------*/
/*! \file

\brief functionality to read particles from file

\level 3


*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                    sfuchs 04/2018 |
 *---------------------------------------------------------------------------*/
#include "4C_particle_engine_particlereader.hpp"

#include "4C_io_pstream.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_object.hpp"
#include "4C_particle_engine_typedefs.hpp"

#include <Teuchos_Time.hpp>

#include <utility>

FOUR_C_NAMESPACE_OPEN


/*---------------------------------------------------------------------------*
 | constructor                                                sfuchs 03/2018 |
 *---------------------------------------------------------------------------*/
INPUT::ParticleReader::ParticleReader(const INPUT::DatFileReader& reader, std::string sectionname)
    : reader_(reader), comm_(reader.Comm()), sectionname_(std::move(sectionname))
{
  // empty constructor
}

/*---------------------------------------------------------------------------*
 | do the actual reading of particles                         sfuchs 03/2018 |
 *---------------------------------------------------------------------------*/
void INPUT::ParticleReader::Read(std::vector<PARTICLEENGINE::ParticleObjShrdPtr>& particles)
{
  const int myrank = comm_->MyPID();
  const int numproc = comm_->NumProc();
  std::string inputfile_name = reader_.MyInputfileName();

  const int numparticles = reader_.ExcludedSectionLength(sectionname_);

  // proceed only if particles are given in .dat-file
  if (numparticles > 0)
  {
    Teuchos::Time time("", true);

    if (!myrank && !reader_.MyOutputFlag()) IO::cout << "Read and create particles\n" << IO::flush;

    // read in the particles block-wise:
    // EITHER one block per processor so that the number of blocks is numproc
    // OR number of blocks is numparticles if less particles than procs are read in
    // determine a rough blocksize
    int nblock = std::min(numproc, numparticles);
    int bsize = std::max(numparticles / nblock, 1);

    // an upper limit for bsize
    int maxblocksize = 200000;

    if (bsize > maxblocksize)
    {
      // without an additional increase of nblock by 1 the last block size
      // could reach a maximum value of (2*maxblocksize)-1, potentially
      // violating the intended upper limit!
      nblock = 1 + static_cast<int>(numparticles / maxblocksize);
      bsize = maxblocksize;
    }

    // open input file at the right position
    // note that stream is valid on proc 0 only!
    std::ifstream file;
    if (!myrank)
    {
      file.open(inputfile_name.c_str());
      file.seekg(reader_.ExcludedSectionPosition(sectionname_));
    }

    std::string line;
    bool endofsection = false;

    if (!myrank && !reader_.MyOutputFlag())
    {
      printf("numparticle %d nblock %d bsize %d\n", numparticles, nblock, bsize);
      fflush(stdout);
    }

    // note that the last block is special....
    for (int block = 0; block < nblock; ++block)
    {
      double t1 = time.totalElapsedTime(true);

      if (!myrank and !endofsection)
      {
        int bcount = 0;

        while (getline(file, line))
        {
          if (line.find("--") == 0)
          {
            endofsection = true;
            break;
          }
          else
          {
            std::istringstream linestream;
            linestream.str(line);

            PARTICLEENGINE::TypeEnum particletype;
            PARTICLEENGINE::ParticleStates particlestates;

            std::string typelabel;
            std::string type;

            std::string poslabel;
            std::vector<double> pos(3);

            // read in particle type and position
            linestream >> typelabel >> type >> poslabel >> pos[0] >> pos[1] >> pos[2];

            if (typelabel != "TYPE") FOUR_C_THROW("expected particle type label 'TYPE'!");

            if (poslabel != "POS") FOUR_C_THROW("expected particle position label 'POS'!");

            // get enum of particle type
            particletype = PARTICLEENGINE::EnumFromTypeName(type);

            // allocate memory to hold particle position state
            particlestates.resize(PARTICLEENGINE::Position + 1);

            // set position state
            particlestates[PARTICLEENGINE::Position] = pos;

            // optional particle states
            {
              std::string statelabel;
              PARTICLEENGINE::StateEnum particlestate;
              std::vector<double> state;

              while (linestream >> statelabel)
              {
                // optional particle radius
                if (statelabel == "RAD")
                {
                  particlestate = PARTICLEENGINE::Radius;
                  state.resize(1);
                  linestream >> state[0];
                }
                // optional rigid body color
                else if (statelabel == "RIGIDCOLOR")
                {
                  particlestate = PARTICLEENGINE::RigidBodyColor;
                  state.resize(1);
                  linestream >> state[0];
                }
                else
                  FOUR_C_THROW(
                      "optional particle state with label '%s' unknown!", statelabel.c_str());

                if (not linestream)
                  FOUR_C_THROW("expecting values of state '%s' if label '%s' is set!",
                      PARTICLEENGINE::EnumToStateName(particlestate).c_str(), statelabel.c_str());

                // allocate memory to hold optional particle state
                if (static_cast<int>(particlestates.size()) < (particlestate + 1))
                  particlestates.resize(particlestate + 1);

                // set optional particle state
                particlestates[particlestate] = state;
              }
            }

            // construct and store read in particle object
            particles.emplace_back(
                std::make_shared<PARTICLEENGINE::ParticleObject>(particletype, -1, particlestates));

            ++bcount;
            if (block != nblock - 1)  // last block takes all the rest
              if (bcount == bsize)    // block is full
              {
                break;
              }
          }
        }
      }

      double t2 = time.totalElapsedTime(true);
      if (!myrank && !reader_.MyOutputFlag())
      {
        printf("reading %10.5e secs\n", t2 - t1);
        fflush(stdout);
      }
    }

    if (!myrank && !reader_.MyOutputFlag())
      printf("in............................................. %10.5e secs\n",
          time.totalElapsedTime(true));
  }
}

FOUR_C_NAMESPACE_CLOSE

/*

  Simple 2D active Q-Tensor with conserved binary parameter

  To do:

    * print test statistics for verbose = 2
    * implement the boundary layer + bdry conditions
    * go to 3d

*/

#include <omp.h>
#include "header.hpp"
#include "options.hpp"
#include "write.hpp"
#include "tools.hpp"
#include "models.hpp"
#include "error_msg.hpp"

#include <chrono>

#ifdef DEBUG
#include <fenv.h>
#endif

using namespace std;

// =============================================================================
// Variables

/** Program options
 * @{ */

/** verbosity level
 *
 * 0: no output
 * 1: normal output
 * 2: extended output (default)
 * */
unsigned verbose = 2;
/** ptr to model */
ModelPtr model;
/** model name */
string model_name;
/** compress output? (we use zip) */
bool compress, compress_full;
/** name of the run */
string runname;
/** Output dir (or tmp dir before moving files to the archive) */
string output_dir;
/** write any output? */
bool no_write = false;
/** skip runtime warnings? */
bool no_warning = false;
/** are the runtime warnings fatal? (i.e. they do stop the simulation) */
bool stop_at_warning = false;
/** padding for output */
unsigned pad;

/** @} */


/** Simulation parameters shared by all models
 * @{ */

/** size of the system */
unsigned LX, LY;
/** total number of nodes */
unsigned N;
/** number of threads */
unsigned nthreads;
/** Total number of time steps */
unsigned nsteps;
/** Time interval between data outputs */
unsigned ninfo;
/** Time at which to start the output */
unsigned nstart;
/** number of subdivisions for a time step */
unsigned nsubsteps;
/** effective time step */
double time_step;
/** Boundary conditions flag */
unsigned BC;

/** @} */

/** Some other variables
 * @{ */

/** Total time spent writing output (in sec) */
chrono::duration<double> write_duration;

/** pure eyecandy */
string title = R"(
      __  __
     |  \/  |__ _ ______
     | |\/| / _` (_-<_-<
     |_|  |_\__,_/__/__/

     Many active systems
     simulations, 2016-7
)";
// fix vim highlighting:"

/** @} */

// =============================================================================

/** Init multi-threading
  *
  * Somewhow omp_get_num_threads() is not working properly with gcc... so we
  * need to use this trick to get the standard number of threads. Or I am too
  * dumb to use omp...
  * */
void SetThreads()
{
  // if nthreads is 1 we use the default number of threads from OpenMP

  nthreads = (unsigned) omp_get_max_threads();
  if(verbose) cout << nthreads << " active threads" << endl;    
  
  /*
  if(nthreads == 1)
  {
    // count the number of OpenMP threads
    unsigned count = 0;
    #pragma omp parallel
    {
      #pragma omp atomic
      ++count;
    }
    nthreads = count;
  }
  */
}

/** This is the main algorithm */
void Algorithm()
{
  for(unsigned t=0; t<nsteps; ++t)
  {
    if(t%ninfo==0)
    {
      // write current frame
      if(!no_write and t>=nstart)
      {
        const auto start = chrono::steady_clock::now();

        try
        {
          WriteFrame(t);
        }
        catch(...) {
          cerr << "error" << endl;
          throw;
        }

        write_duration += chrono::steady_clock::now() - start;
      }

      // some verbose
      if(verbose>1) cout << '\n';
      if(verbose)
        cout << "timesteps t = " << setw(pad) << setfill(' ') << right
                                   << t << " to "
                                   << setw(pad) << setfill(' ') << right
                                   << t+ninfo << endl;
      if(verbose>1)   cout << string(width, '-') << endl;
    }

    // do the computation
    for(unsigned s=0; s<nsubsteps; ++s)
      model->Step();

    // runtime warnings/checks
    if(t%ninfo==0 and !no_warning)
    {
      try
      {
        if(verbose>1)
          model->RuntimeStats();

        model->RuntimeChecks();
      }
      catch(error_msg e)
      {
        throw;
      }
      catch(warning_msg e)
      {
        if(stop_at_warning)
          throw;
        else
        {
          if(verbose) cerr << "warning: " << e.what() << endl;
        }
      }
    }
  }

  // finally write final frame
  if(!no_write and nsteps>=nstart) WriteFrame(nsteps);
}


/** Program entry */
int main(int argc, char **argv)
{
  // if in debug mode, catch all arithmetic exceptions
#ifdef DEBUG
  //feenableexcept(FE_INVALID | FE_OVERFLOW);
#endif

  // print that beautiful title
  cout << title << endl;

  try
  {
    // ========================================
    // Setup

    if(argc<2) throw error_msg("no argument provided. Type -h for help.");
    // declare models
    DeclareModels();
    // parse program options (may throw) and create model pointer
    ParseProgramOptions(argc, argv);
    // Process options, i.e. get real params values from input
    ProcessProgramOptions();
    // check that we have a run name
    if(runname.empty())
      throw error_msg("please specify a file path for this run.");
    // print simulation parameters
    PrintParameters();

    // ========================================
    // Initialization
    if(verbose) cout << endl << "Initialization" << endl << string(width, '=')
                     << endl;

    // warning and flags
    // no output
    if(no_write and verbose) cout << "warning: output is not enabled." << endl;

    // model init
    if(verbose) cout << "model initialization ..." << flush;
    try {
      model->Initialize();
    } catch(...) {
      if(verbose) cout << " error" << endl;
      throw;
    }
    if(verbose) cout << " done" << endl;

    // parameters init
    if(verbose) cout << "system initialisation ..." << flush;
    try {
      model->Configure();
    } catch(...) {
      if(verbose) cout << " error" << endl;
      throw;
    }
    if(verbose) cout << " done" << endl;

    // write params to file
    if(!no_write)
    {
      if(verbose) cout << "create output directory " << " ...";
      try {
        CreateOutputDir();
      } catch(...) {
        if(verbose) cout << " error" << endl;
        throw;
      }
      if(verbose) cout << " done" << endl;

      ClearOutput();

      if(verbose and compress_full)
        cout << "create output file " << runname << ".zip ...";
      if(verbose and not compress_full)
        cout << "write parameters ...";

      try {
        WriteParams();
      } catch(...) {
        if(verbose) cout << " error" << endl;
        throw;
      }
      if(verbose) cout << " done" << endl;
    }

    // multi-threading

      if(verbose) cout << "multi-threading ... " << flush;
      SetThreads();
      if(verbose) cout << nthreads << " active threads" << endl;


    // preparation
    if(verbose)   cout << "preparation ... " << flush;
    model->Pre();
    if(verbose) cout << " done" << endl;

    // Run banner
    if(verbose) cout << endl << "Run" << endl << string(width, '=') << "\n\n";

    // print some stats
    model->PreRunStats();
      // record starting time
  const auto start = chrono::steady_clock::now();
  // run the thing
  Algorithm();
  // record end time
  const auto duration = chrono::steady_clock::now() - start;

  if(verbose) cout << "post-processing ... " << flush;
  model->Post();
  if(verbose) cout << "done" << endl;

  if(verbose)
  {
    cout << endl << "Statistics" << endl << string(width, '=') << endl;
    cout << "Total run time :                    "
         << chrono::duration_cast<chrono::milliseconds>(duration).count()
            /1000. << " s" << endl;
    cout << "Total time spent writing output :   "
         << chrono::duration_cast<chrono::milliseconds>(write_duration).count()
            /1000. << " s" << endl;
  }
  }
  // custom small messages
  catch(const error_msg& e) {
    cerr << argv[0] << ": error: " << e.what() << endl;
    return 1;
  }
  // bad alloc (possibly from memory())
  catch(const bad_alloc& ba) {
    cerr << argv[0] << ": error initializing memory: " << ba.what() << endl;
    return 1;
  }
  // all the rest (mainly from boost)
  catch(const exception& e) {
    cerr << argv[0] << ": " << e.what() << endl;
    return 1;
  }



  return 0;
}

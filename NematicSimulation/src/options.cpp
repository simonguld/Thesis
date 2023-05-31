#include "header.hpp"
#include "models.hpp"
#include "options.hpp"
#include "tools.hpp"
#include "error_msg.hpp"
#include "random.hpp"

using namespace std;
namespace opt = boost::program_options;

// =============================================================================
// Options

// declare variables externally
// parameters
extern unsigned verbose, nthreads, pad, ninfo, nsteps, LX, LY,
                nsubsteps, nstart;
extern double time_step;
extern bool no_write, pure_nematic, compress, compress_full, no_warning,
            stop_at_warning;
extern string runname, init_config, output_dir;
extern unsigned BC;
// model
extern ModelPtr model;
extern string model_name;

/** the variables map used to collect options */
opt::variables_map vm;
/** name of the inpute file */
string inputname = "";
/** angle in degrees (input variable only) */
double angle_deg;
/** Switches */
bool list_models, force_delete;
/** The random number seed */
unsigned seed;

void ParseProgramOptions(int ac, char **av)
{
  // options allowed only in the command line
  opt::options_description generic("Generic options");
  generic.add_options()
    ("help,h",
     "produce help message")
    ("verbose,v", opt::value<unsigned>(&verbose)->implicit_value(2),
     "verbosity level (0=none, 1=little, 2=normal, 3=debug)")
    ("input,i", opt::value<string>(&inputname),
     "input file")
    ("force-delete,f", opt::bool_switch(&force_delete),
     "force deletion of existing output file")
    ("threads,t",
     opt::value<unsigned>(&nthreads)->default_value(0)->implicit_value(1),
     "number of threads (0=no multithreading, 1=OpenMP default, "
     ">1=your favorite number)")
    ("list,l", opt::bool_switch(&list_models),
     "list available models")
    ("compress,c", opt::bool_switch(&compress),
     "compress individual files using zip")
    ("compress-full", opt::bool_switch(&compress_full),
     "compress full output using zip (might be slow)")
    ("no-write", opt::bool_switch(&no_write),
     "disable file output (for testing purposes)");

  // options allowed both in the command line and config file
  opt::options_description config("Program options");
  config.add_options()
    ("output,o", opt::value<string>(&runname),
     "output name (if compression is on, then .zip is added automatically)")
    ("model,m", opt::value<string>(&model_name),
     "model name (type --list for a list)")
    ("LX", opt::value<unsigned>(&LX),
     "# of nodes in the x direction")
    ("LY", opt::value<unsigned>(&LY),
     "# of nodes in the y direction")
    ("nsteps", opt::value<unsigned>(&nsteps),
     "iterate this many steps in total")
    ("nsubsteps", opt::value<unsigned>(&nsubsteps)->default_value(1u),
     "subdivision of a single time step "
     "(effective time step is 1/nsubsteps)")
    ("ninfo", opt::value<unsigned>(&ninfo),
     "save frame every so many steps")
    ("seed", opt::value<unsigned>(&seed),
     "set seed for random number generation (random if unset)")
    ("no-warning", opt::bool_switch(&no_warning),
     "disable model specific runtime warnings")
    ("stop-at-warning", opt::bool_switch(&stop_at_warning),
     "runtime warnings interrupt the algorithm")
    ("nstart", opt::value<unsigned>(&nstart)->default_value(0u),
     "time at which to start the output")
    ("bc", opt::value<unsigned>(&BC)->default_value(0u),
     "boundary conditions flag (0=pbc, any other value is defined by the model)");

  // command line options
  opt::options_description cmdline_options;
  cmdline_options.add(generic).add(config);
  opt::options_description config_file_options;
  config_file_options.add(config);

  // first unnamed argument is the input file
  opt::positional_options_description p;
  p.add("input", 1);

  // reintialize vm in case we run this function twice
  vm = opt::variables_map();

  // parse first the cmd line to get model name (no throw)
  opt::store(
    opt::command_line_parser(ac, av)
    .options(cmdline_options)
    .positional(p)
    .allow_unregistered()
    .run(), vm);
  opt::notify(vm);

  // print help msg and exit
  if(vm.count("help") and model_name.empty() and inputname.empty())
  {
    cout << cmdline_options << endl;
    cout << endl << "You can see model-specific options by specifying a model "
                    "using '-m' or '--model'" << endl;
    exit(0);
  }

  // parse input file (values are not erased, such that cmd line args
  // are 'stronger') (no throw)
  if(not inputname.empty())
  {
    std::fstream file(inputname.c_str(), std::fstream::in);
    if(!file.good()) throw error_msg("can not open runcard file ", inputname);
    opt::store(opt::parse_config_file(file, config_file_options, true), vm);
    opt::notify(vm);
  }

  // print help msg and exit
  if(vm.count("help") and model_name.empty())
  {
    cout << cmdline_options << endl;
    exit(0);
  }

  // list the models and exit
  if(list_models)
  {
    cout << "Available models" << endl << string(width, '=') << endl;
    ListModels();
    cout << endl;
    exit(0);
  }

  // init random numbers (again)
  if(vm.count("seed")) set_seed(seed);

  // we need a model
  if(model_name=="")
    throw error_msg("please specify model. "
                    "Type --list or -l for a list of available models.");
  // construct model (may throw)
  model = NewModel(model_name, LX, LY, BC);
  // add model options
  auto model_options = model->GetOptions();
  for(const auto& o : model_options)
  {
    cmdline_options.add(o);
    config_file_options.add(o);
  }

  // parse the cmd line a second time with model options (throw)
  opt::store(
    opt::command_line_parser(ac, av)
    .options(cmdline_options)
    .positional(p)
    .run(), vm);
  opt::notify(vm);

  // print help msg and exit
  if(vm.count("help"))
  {
    cout << cmdline_options << endl;
    exit(0);
  }

  // parse input file (values are not erased, such that cmd line args
  // are 'stronger')
  if(inputname.empty())
    throw error_msg("please provide an input file / type -h for help.");
  else
  {
    std::fstream file(inputname.c_str(), std::fstream::in);
    if(!file.good()) throw error_msg("can not open runcard file ", inputname);
    opt::store(opt::parse_config_file(file, config_file_options), vm);
    opt::notify(vm);
  }

  // fix compression mode: if we compress the full archive we do not compress
  // individual files.
  if(compress_full) compress=false;

  // Set default value for runname (depends on compression)
  if(vm.count("output")==0)
  {
    if(compress_full) runname = "output";
    else runname = "./";
  }
}

void ProcessProgramOptions()
{
  // compute the correct padding
  pad = inline_str(nsteps).length();

  // compute effective time step
  time_step = 1./nsubsteps;

  // set nstart to the next correct frame (round above)
  if(nstart%ninfo) nstart = (1u+nstart/ninfo)*ninfo;
}

/** Print variables from variables_map
  *
  * from: https://gist.github.com/gesquive/8673796
  */
void print_vm(const opt::variables_map& vm, unsigned padding)
{
  for (opt::variables_map::const_iterator it = vm.begin(); it != vm.end(); ++it)
  {
    // pass if defaulted or empty
    if (vm[it->first].defaulted() || it->second.defaulted()) continue;
    if (((boost::any)it->second.value()).empty()) continue;

    std::cout << std::left << std::setw(floor(padding/2)) << it->first;

    /*if (((boost::any)it->second.value()).empty()) {
      std::cout << "(empty)";
    }
    if (vm[it->first].defaulted() || it->second.defaulted()) {
      std::cout << "(default)";
    }*/

    std::cout << std::right << std::setw(ceil(padding/2));

    bool is_char;
    try {
      boost::any_cast<const char*>(it->second.value());
      is_char = true;
    } catch (const boost::bad_any_cast &) {
      is_char = false;
    }
    bool is_str;
    try {
      boost::any_cast<std::string>(it->second.value());
      is_str = true;
    } catch (const boost::bad_any_cast &) {
      is_str = false;
    }

    if (((boost::any)it->second.value()).type() == typeid(int)) {
      std::cout << vm[it->first].as<int>() << std::endl;
    } else if (((boost::any)it->second.value()).type() == typeid(unsigned)) {
      std::cout << vm[it->first].as<unsigned>() << std::endl;
    } else if (((boost::any)it->second.value()).type() == typeid(size_t)) {
      std::cout << vm[it->first].as<size_t>() << std::endl;
    } else if (((boost::any)it->second.value()).type() == typeid(bool)) {
      std::cout << (vm[it->first].as<bool>() ? "true" : "false") << std::endl;
    } else if (((boost::any)it->second.value()).type() == typeid(double)) {
      std::cout << vm[it->first].as<double>() << std::endl;
    } else if (((boost::any)it->second.value()).type()
               == typeid(vector<double>)) {
      std::cout << vec2str(vm[it->first].as<vector<double>>()) << std::endl;
    } else if (((boost::any)it->second.value()).type()
               == typeid(vector<unsigned>)) {
      std::cout << vec2str(vm[it->first].as<vector<unsigned>>()) << std::endl;
    } else if (is_char) {
      std::cout << vm[it->first].as<const char *>() << std::endl;
    } else if (is_str) {
      std::string temp = vm[it->first].as<std::string>();
      if (temp.size()) {
        std::cout << temp << std::endl;
      } else {
        std::cout << "true" << std::endl;
      }
    } else { // Assumes that the only remainder is vector<string>
      try {
        auto vect = vm[it->first].as<std::vector<std::string> >();
        unsigned int i = 0;
        for (auto oit=vect.begin();
            oit != vect.end(); oit++, ++i) {
          std::cout << "\r> " << it->first
                    << "[" << i << "]=" << (*oit) << std::endl;
        }
      } catch (const boost::bad_any_cast &) {
        std::cout << "UnknownType("
                  << ((boost::any)it->second.value()).type().name() << ")" << std::endl;
      }
    }
  }
}

void PrintParameters()
{
  // print the simulation parameters
  if(verbose)
  {
    cout << "Run parameters" << endl;
    cout << string(width, '=') << endl;
    print_vm(vm, width);
  }
}

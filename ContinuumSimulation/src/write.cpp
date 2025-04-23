#include "header.hpp"
#include "write.hpp"
#include "models.hpp"
#include "error_msg.hpp"
#include "random.hpp"
#include "tools.hpp"

using namespace std;

// =============================================================================
// Serialization

// parameters
extern unsigned LX, LY, BC;
extern unsigned N;
extern unsigned ninfo;
extern unsigned nstart;
extern unsigned nsteps;
extern unsigned nsubsteps;
extern bool compress, compress_full;
extern string runname, output_dir;
extern bool force_delete;
// the model
extern ModelPtr model;
extern string model_name;
extern vector<string> ext_str;

void WriteFrame(unsigned t)
{
  // construct output name
  const string oname = inline_str(output_dir, "frame", t, ".json");

  // write
  {
    stringstream buffer;
    {
      oarchive ar(buffer, "frame", 1);
      // serialize
      model.ptr->serialize_frame(ar);

      if(ar.bad_value()) throw error_msg("nan found while writing file.");
    }
    // dump to file
    std::ofstream ofs(oname.c_str(), ios::out);
    ofs << buffer.rdbuf();
  }

  // compress
  if(compress)
  {
    const int ret = system(inline_str("zip -jm ", oname, ".zip ", oname,
                                      " > /dev/null 2>&1").c_str());
    if(ret!=0) throw error_msg("zip non-zero return value ", ret, ".");
  }
  if(compress_full)
  {
    const int ret = system(inline_str("zip -jm ", runname, ".zip ", oname,
                                      " > /dev/null 2>&1").c_str());
    if(ret!=0) throw error_msg("zip non-zero return value ", ret, ".");
  }
}

void WriteParams()
{
  // a name that makes sense
  const string oname = inline_str(output_dir, "parameters.json");

  // write
  {
    stringstream buffer;
    {
      // serialize
      oarchive ar(buffer, "parameters", 1);
      // ...program parameters...
      ar & auto_name(LX)
         & auto_name(LY)
         & auto_name(BC)
         & auto_name(nsteps)
         & auto_name(nsubsteps)
         & auto_name(ninfo)
         & auto_name(nstart)
         & auto_name(model_name);
      // ...and model parameters
      model.ptr->serialize_params(ar);

      if(ar.bad_value()) throw error_msg("nan found while writing file.");
    }
    // dump to file
    std::ofstream ofs(oname.c_str(), ios::out);
    ofs << buffer.rdbuf();
  }

  // compress
  if(compress)
  {
    const int ret = system(inline_str("zip -jm ", oname, ".zip ", oname,
                                      " > /dev/null 2>&1").c_str());
    if(ret!=0) throw error_msg("zip non-zero return value ", ret, ".");
  }
  if(compress_full)
  {
    const int ret = system(inline_str("zip -jm ", runname, ".zip ", oname,
                                      " > /dev/null 2>&1").c_str());
    if(ret!=0) throw error_msg("zip non-zero return value ", ret, ".");
  }
}

void ClearOutput()
{
  if(compress_full)
  {
    // file name of the output file
    const string fname = runname + ".zip";

    {
      // try open it
      ifstream infile(fname);
      // does not exist we are fine
      if(not infile.good()) return;
    }

    if(not force_delete)
    {
      // ask
      char answ = 0;
      cout << "remove output file '" << fname << "'? ";
      cin >> answ;

      if(answ != 'y' and answ != 'Y')
        throw error_msg("output file '", fname,
                        "' already exist, please provide a different name.");
    }

    // delete
    const int ret = system(inline_str("rm -f ", fname).c_str());
    if(ret) throw error_msg("rm returned non-zero value ", ret, ".");
  }
  else
  {
    // extension of single files
    string ext = compress ? ".json.zip" : ".json";

    // check that parameters.json does not exist in the output dir and if it
    // does ask for deletion (this is not completely fool proof, but ok...)
    {
      ifstream infile(output_dir + "parameters" + ext);
      if(not infile.good()) return;
    }

    if(not force_delete)
    {
      // ask
      char answ = 0;
      cout << "remove output files in directory '" << output_dir << "'? ";
      cin >> answ;

      if(answ != 'y' and answ != 'Y')
        throw error_msg("output files already exist in directory '",
                        output_dir, "'.");
    }

    // delete all output files
    int ret;
    ret = system(inline_str("rm -f ", output_dir, "/parameters.json").c_str());
    if(ret) throw error_msg("rm returned non-zero value ", ret, ".");
    ret = system(inline_str("rm -f ", output_dir, "/parameters.json.zip").c_str());
    if(ret) throw error_msg("rm returned non-zero value ", ret, ".");
    ret = system(inline_str("rm -f ", output_dir, "/frame*.json").c_str());
    if(ret) throw error_msg("rm returned non-zero value ", ret, ".");
    ret = system(inline_str("rm -f ", output_dir, "/frame*.json.zip").c_str());
    if(ret) throw error_msg("rm returned non-zero value ", ret, ".");
  }
}

void CreateOutputDir()
{
  // if full compression is on: we need to create a random tmp directory
  if(compress_full)
  {
    // use hash of runname string plus salt
    hash<string> hash_fn;
    unsigned dir_name = hash_fn(inline_str(runname, randu()));
    output_dir = inline_str("/tmp/", dir_name, "/");
  }
  // if full compression is off: just dump files where they belong
  else
    // note that runname can not be empty from options.cpp
    output_dir = runname + ( runname.back()=='/' ? "" : "/" );


  // create output dir if needed
  const int ret = system(inline_str("mkdir -p ", output_dir).c_str());
  if(ret)
    throw error_msg("can not create output directory, mkdir returned ", ret, ".");
}

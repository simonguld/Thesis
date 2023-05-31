#include "header.hpp"
#include "models/minimal.hpp"

using namespace std;
namespace opt = boost::program_options;

Minimal::Minimal(unsigned LX, unsigned LY, unsigned BC)
  : Model(LX, LY, BC, GridType::Periodic)
{}

void Minimal::Initialize()
{
  // initialize variables
}

void Minimal::Configure()
{
  // initial configuration
}

option_list Minimal::GetOptions()
{
  opt::options_description options;
  // add model specific options here
  return { options };
}

void Minimal::Step()
{
  // single time step
}

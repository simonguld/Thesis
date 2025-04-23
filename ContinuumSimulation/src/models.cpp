#include "header.hpp"
#include "models.hpp"
#include "tools.hpp"
#include "lb.hpp"
#include "error_msg.hpp"

using namespace std;
namespace opt = boost::program_options;

// Model declaration
#include "declare_models.cpp"

// =============================================================================
// Initialization functions

Model::Model(unsigned LX, unsigned LY, unsigned BC, GridType Type)
  : Grid(LX, LY, Type), BC(BC)
{}

// =============================================================================
// Models managment

/** Factory and description (for storage) */
struct FactoryAndDesc {
  string  desc;
  Factory fact;
};

/** The list of all available models */
static map<string, FactoryAndDesc> model_list;

void _add_model(string name, string desc, Factory fact)
{
  // check that name is not registered
  if(model_list.find(name)!=model_list.end())
    throw error_msg("model '", name, "' already registered.");
  // register
  model_list[name] = { desc, fact };
}

ModelPtr NewModel(string name, unsigned LX, unsigned LY, unsigned BC)
{
  // if name matches we construct using factory
  for(auto& i : model_list)
    if(i.first==name) return i.second.fact(LX, LY, BC);
  // we did not find anything
  throw error_msg("model '", name,
      "' can not be found. Type --list for a list of available models.");
}

void ListModels()
{
  for(auto& i : model_list)
  {
    cout << '\n' << i.first << ":" << '\n'
         << string(i.first.length()+1, '-') << '\n';

    // split description into words
    vector<string> words = split(i.second.desc);

    // break lines
    size_t p = 0;
    for(const auto& w : words)
    {
      if(p+w.size()>width)
      {
        cout << '\n';
        p = 0;
      }

      cout << w << " ";
      p += w.size() + 1;
    }
    cout << '\n';
  }
}

#include "header.hpp"
#include "serialization.hpp"

using namespace std;

/** The number of spaces in indendation */
static const unsigned padding = 2;

oarchive::oarchive(std::ostream& stream_, string id, unsigned version)
  : stream(stream_)
{
  if(!stream.good()) throw bad_stream();
  // write initial brace
  open_group();
  // add description and version
  add("id", id);
  add("version", version);
  // open the data group
  add_key("data");
  open_group();
}

oarchive::~oarchive()
{
  // close the 'data' group
  close_group();
  // write final brace
  if(stream.good()) stream << endl << '}';
}

void oarchive::indent(unsigned n)
{
  level += n;
}

void oarchive::unindent(unsigned n)
{
  level -= n;
}

void oarchive::open_group(const std::string& obrace)
{
  // open brace
  stream << obrace;
  // indent one level
  indent();
  // next element is first
  first = true;
}

void oarchive::close_group(const string& cbrace)
{
  // unindent
  unindent();
  // new line
  stream << endl << string(padding*level, ' ');
  // close brace
  stream << cbrace;
  // not the first in the list
  first = false;
}

void oarchive::add_key(const string& key)
{
  new_line();
  stream << "\"" << key << "\" : ";
}

void oarchive::new_line()
{
  // add comma
  if(!first) stream << ',';
  else first = false;
  // new line
  stream << endl;
  // indent
  stream << string(padding*level, ' ');
}

template<>
void oarchive::add_element<const char*>(const char* const& t)
{
  std::stringstream ss;
  ss << "\"" << t << "\"";
  stream << ss.str();
}

template<>
void oarchive::add_element<std::string>(const std::string& t)
{
  add_element(t.c_str());
}

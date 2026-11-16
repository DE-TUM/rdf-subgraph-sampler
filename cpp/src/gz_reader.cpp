#include "bgpgen/gz_reader.hpp"

#include "bgpgen/types.hpp"

#ifdef BGPGEN_HAVE_ZLIB
#include <zlib.h>
#endif

namespace bgpgen {

bool is_gz_path(const std::string& path) {
  return path.size() > 3 && path.compare(path.size() - 3, 3, ".gz") == 0;
}

#ifdef BGPGEN_HAVE_ZLIB

std::vector<char> read_gz_file(const std::string& path) {
  gzFile gz = gzopen(path.c_str(), "rb");
  if (!gz) throw BgpError("cannot open '" + path + "'");
  gzbuffer(gz, 1u << 20);

  std::vector<char> out;
  out.resize(64u << 20);
  size_t used = 0;
  while (true) {
    if (used == out.size()) out.resize(out.size() * 2);
    int n = gzread(gz, out.data() + used, static_cast<unsigned>(out.size() - used));
    if (n < 0) {
      int errnum = 0;
      std::string msg = gzerror(gz, &errnum);
      gzclose(gz);
      throw BgpError("error reading '" + path + "': " + msg);
    }
    if (n == 0) break;
    used += static_cast<size_t>(n);
  }
  gzclose(gz);
  out.resize(used);
  return out;
}

#else

std::vector<char> read_gz_file(const std::string& path) {
  throw BgpError("'" + path + "' is gzipped but bgpgen was built without zlib");
}

#endif

} // namespace bgpgen

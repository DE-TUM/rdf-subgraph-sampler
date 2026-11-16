#pragma once

#include <string>
#include <vector>

namespace bgpgen {

/// True if `path` ends in ".gz".
bool is_gz_path(const std::string& path);

/// Decompresses a .gz file fully into memory. Throws BgpError if zlib support
/// was not compiled in or the file is unreadable/corrupt.
std::vector<char> read_gz_file(const std::string& path);

} // namespace bgpgen

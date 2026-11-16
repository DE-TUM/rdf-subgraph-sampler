#pragma once

#include "bgpgen/term_table.hpp"
#include "bgpgen/types.hpp"

#include <string>
#include <vector>

namespace bgpgen {

/// Strict N-Triples parser — the bulk data path.
///
/// Subject/predicate IRIs are interned bracket-less; literal objects are
/// interned as their RAW token bytes (quotes, escapes, @lang / ^^<dt>
/// suffix), so output round-trips the input verbatim. Blank nodes are scoped
/// by `document_id`. Throws BgpError with line information on syntax errors.
///
/// Buffers larger than `chunk_bytes` are split at newlines and parsed on
/// `threads` threads (0 = all hardware threads) into chunk-local term tables
/// that are folded into `terms` afterwards. Chunk boundaries depend only on
/// the input bytes, so term ids — and everything downstream — are identical
/// for any thread count.
std::vector<Triple> parse_nt_buffer(const char* data, size_t size, TermTable& terms,
                                    uint32_t document_id, unsigned threads = 0,
                                    size_t chunk_bytes = 4u << 20);

/// Memory-maps `path` and parses it with parse_nt_buffer.
std::vector<Triple> load_nt_file(const std::string& path, TermTable& terms, uint32_t document_id,
                                 unsigned threads = 0, size_t chunk_bytes = 4u << 20);

} // namespace bgpgen

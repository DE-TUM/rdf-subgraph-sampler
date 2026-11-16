#pragma once

#include "bgpgen/types.hpp"

#include <algorithm>
#include <span>
#include <vector>

namespace bgpgen {

/// All adjacency lists are sorted unique TermId arrays; set operations are
/// merges over these spans.
using IdSpan = std::span<const TermId>;

inline bool contains(IdSpan sorted, TermId id) {
  return std::binary_search(sorted.begin(), sorted.end(), id);
}

inline void sort_unique(std::vector<TermId>& ids) {
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
}

inline void merge_union(IdSpan a, IdSpan b, std::vector<TermId>& out) {
  out.clear();
  out.reserve(a.size() + b.size());
  std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
}

inline void merge_intersect(IdSpan a, IdSpan b, std::vector<TermId>& out) {
  out.clear();
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
}

} // namespace bgpgen

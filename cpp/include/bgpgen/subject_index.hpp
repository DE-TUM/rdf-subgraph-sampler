#pragma once

#include "bgpgen/graph.hpp"
#include "bgpgen/types.hpp"

#include <span>
#include <vector>

namespace bgpgen {

/// Node-major view of the data: for every subject, its full out-edge row as
/// (PredIndex, neighbor) pairs, grouped contiguously (predicates ascending,
/// neighbors ascending within a predicate). Star sampling and the DFS
/// matcher draw directly from these rows; out-degree is O(1).
///
/// Built with reverse=true it indexes objects instead: row_of(o) lists
/// (PredIndex, subject) pairs — the matcher's reverse adjacency.
class SubjectIndex {
public:
  struct Entry {
    PredIndex pred;
    TermId obj; // the neighbor: object (forward) or subject (reverse)
  };

  /// `bnode_base` = first non-IRI term id. Subjects are sorted, and the id
  /// space is segmented, so IRI subjects — the only star roots, matching the
  /// Python sampler — form a prefix of the subject list.
  void build(const TripleStore& store, TermId bnode_base, bool reverse = false);

  size_t subject_count() const { return subjects_.size(); }
  size_t iri_subject_count() const { return iri_subjects_; }
  TermId subject_at(size_t pos) const { return subjects_[pos]; }
  uint32_t degree(size_t pos) const { return offsets_[pos + 1] - offsets_[pos]; }
  std::span<const Entry> row(size_t pos) const {
    return {entries_.data() + offsets_[pos], offsets_[pos + 1] - offsets_[pos]};
  }

  /// Row by subject id; empty if the subject has no out-edges. One array
  /// lookup: the DFS matcher calls this once per search step.
  std::span<const Entry> row_of(TermId s) const {
    if (s >= pos_of_.size() || pos_of_[s] == kNoRow) return {};
    return row(pos_of_[s]);
  }
  uint32_t out_degree(TermId s) const { return static_cast<uint32_t>(row_of(s).size()); }

  /// True if `s` has at least `need` different IRI neighbours (ids below
  /// `bnode_base`). Stops at the need-th one; `need` is at most 255.
  bool has_iri_neighbours(TermId s, uint32_t need, TermId bnode_base) const {
    if (need == 0) return true;
    std::span<const Entry> r = row_of(s);
    if (r.size() < need) return false;
    TermId seen[256];
    uint32_t n = 0;
    for (const Entry& e : r) {
      if (e.obj >= bnode_base) continue;
      uint32_t i = 0;
      while (i < n && seen[i] != e.obj) ++i;
      if (i < n) continue;
      seen[n++] = e.obj;
      if (n >= need) return true;
    }
    return false;
  }

private:
  static constexpr uint32_t kNoRow = UINT32_MAX;
  std::vector<TermId> subjects_;  // sorted unique
  std::vector<uint32_t> pos_of_;  // term id -> position in subjects_, or kNoRow
  std::vector<uint32_t> offsets_; // subject_count() + 1
  std::vector<Entry> entries_;
  size_t iri_subjects_ = 0;
};

/// True if `v` has at least `out` different IRI out-neighbours, `in` different
/// IRI in-neighbours and `all` different IRI neighbours counting both. A graph
/// node that takes a template node needs this (see SlotDegrees).
inline bool enough_neighbours(const SubjectIndex& fwd, const SubjectIndex& rev, TermId v,
                              uint32_t out, uint32_t in, uint32_t all, TermId bnode_base) {
  if (!fwd.has_iri_neighbours(v, out, bnode_base) || !rev.has_iri_neighbours(v, in, bnode_base))
    return false;
  if (all <= out || all <= in) return true;
  TermId seen[256];
  uint32_t n = 0;
  for (const SubjectIndex* idx : {&fwd, &rev})
    for (const SubjectIndex::Entry& e : idx->row_of(v)) {
      if (e.obj >= bnode_base) continue;
      uint32_t i = 0;
      while (i < n && seen[i] != e.obj) ++i;
      if (i < n) continue;
      seen[n++] = e.obj;
      if (n >= all) return true;
    }
  return false;
}

} // namespace bgpgen

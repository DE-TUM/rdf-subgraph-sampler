#pragma once

#include "bgpgen/sorted_ids.hpp"
#include "bgpgen/types.hpp"

#include <vector>

namespace bgpgen {

/// Read-only, predicate-partitioned index over the data graph.
///
/// Each predicate gets two CSR halves: forward (sorted subjects, each with a
/// sorted span of objects) and reverse (sorted objects with sorted subjects).
/// Every adjacency query is a binary search plus a contiguous span — no
/// hashing, no per-triple objects.
class TripleStore {
public:
  TripleStore() = default;

  /// Consumes a (frozen-id) triple vector. Duplicate triples are removed
  /// (RDF set semantics — matches what a SPARQL endpoint counts over).
  /// Builds in parallel (0 = all hardware threads); the result is identical
  /// for any thread count.
  void build(std::vector<Triple> triples, unsigned threads = 0);

  size_t triple_count() const { return triple_count_; }
  size_t predicate_count() const { return pred_terms_.size(); }

  /// Predicates present in the data, as term ids in ascending order.
  IdSpan predicate_terms() const { return pred_terms_; }
  TermId pred_term(PredIndex p) const { return pred_terms_[p]; }

  /// kNoPred if the predicate does not occur in the data.
  PredIndex pred_index(TermId predicate) const;

  IdSpan objects_of(TermId s, PredIndex p) const;
  IdSpan subjects_of(TermId o, PredIndex p) const;

  bool has(TermId s, PredIndex p, TermId o) const { return contains(objects_of(s, p), o); }

  /// All distinct subjects (resp. objects) of a predicate, sorted.
  IdSpan subjects_with(PredIndex p) const { return preds_[p].subjects; }
  IdSpan objects_with(PredIndex p) const { return preds_[p].objects; }

  /// Number of (deduplicated) triples with predicate p.
  size_t triples_with(PredIndex p) const { return preds_[p].fwd_objects.size(); }

  /// Calls fn(subject, objects_span) for every subject of p, ascending.
  template <class Fn>
  void for_each_subject(PredIndex p, Fn&& fn) const {
    const PredicateCsr& csr = preds_[p];
    for (size_t i = 0; i < csr.subjects.size(); ++i)
      fn(csr.subjects[i], IdSpan{csr.fwd_objects.data() + csr.fwd_offsets[i],
                                 csr.fwd_offsets[i + 1] - csr.fwd_offsets[i]});
  }

  /// Calls fn(object, subjects_span) for every object of p, ascending.
  template <class Fn>
  void for_each_object(PredIndex p, Fn&& fn) const {
    const PredicateCsr& csr = preds_[p];
    for (size_t i = 0; i < csr.objects.size(); ++i)
      fn(csr.objects[i], IdSpan{csr.rev_subjects.data() + csr.rev_offsets[i],
                                csr.rev_offsets[i + 1] - csr.rev_offsets[i]});
  }

private:
  struct PredicateCsr {
    std::vector<TermId> subjects;        // sorted unique
    std::vector<uint32_t> fwd_offsets;   // subjects.size() + 1
    std::vector<TermId> fwd_objects;     // sorted within each subject's span
    std::vector<TermId> objects;         // sorted unique
    std::vector<uint32_t> rev_offsets;   // objects.size() + 1
    std::vector<TermId> rev_subjects;    // sorted within each object's span
  };

  std::vector<TermId> pred_terms_; // sorted; position == PredIndex
  std::vector<PredicateCsr> preds_;
  size_t triple_count_ = 0;
};

} // namespace bgpgen

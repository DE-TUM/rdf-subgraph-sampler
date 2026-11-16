#include "bgpgen/subject_index.hpp"

#include <algorithm>

namespace bgpgen {

void SubjectIndex::build(const TripleStore& store, TermId bnode_base, bool reverse) {
  size_t n_preds = store.predicate_count();
  auto keys_with = [&](PredIndex p) {
    return reverse ? store.objects_with(p) : store.subjects_with(p);
  };
  auto for_each = [&](PredIndex p, auto&& fn) {
    if (reverse)
      store.for_each_object(p, fn);
    else
      store.for_each_subject(p, fn);
  };

  // Union of all per-predicate key lists (each already sorted unique).
  for (PredIndex p = 0; p < n_preds; ++p) {
    std::vector<TermId> next;
    merge_union(subjects_, keys_with(p), next);
    subjects_ = std::move(next);
  }
  iri_subjects_ = static_cast<size_t>(
      std::lower_bound(subjects_.begin(), subjects_.end(), bnode_base) - subjects_.begin());

  auto pos_of = [&](TermId s) {
    return static_cast<size_t>(std::lower_bound(subjects_.begin(), subjects_.end(), s) -
                               subjects_.begin());
  };

  // Count degrees, scan to offsets, then scatter. Predicates are visited in
  // ascending PredIndex order, so each row ends up (pred asc, neighbor asc).
  std::vector<uint32_t> degree(subjects_.size(), 0);
  for (PredIndex p = 0; p < n_preds; ++p)
    for_each(p, [&](TermId s, IdSpan others) {
      degree[pos_of(s)] += static_cast<uint32_t>(others.size());
    });

  offsets_.assign(subjects_.size() + 1, 0);
  for (size_t i = 0; i < subjects_.size(); ++i) offsets_[i + 1] = offsets_[i] + degree[i];
  entries_.resize(offsets_.back());

  std::vector<uint32_t> cursor(offsets_.begin(), offsets_.end() - 1);
  for (PredIndex p = 0; p < n_preds; ++p)
    for_each(p, [&](TermId s, IdSpan others) {
      uint32_t& cur = cursor[pos_of(s)];
      for (TermId o : others) entries_[cur++] = {p, o};
    });

  pos_of_.assign(subjects_.empty() ? 0 : static_cast<size_t>(subjects_.back()) + 1, kNoRow);
  for (size_t i = 0; i < subjects_.size(); ++i) pos_of_[subjects_[i]] = static_cast<uint32_t>(i);
}

} // namespace bgpgen

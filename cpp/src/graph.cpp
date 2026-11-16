#include "bgpgen/graph.hpp"

#include "bgpgen/thread_pool.hpp"

#include <algorithm>
#include <tuple>

namespace bgpgen {

// Instead of one big (p,s,o) sort, triples are bucketed by predicate with a
// two-pass count/scatter (per-chunk cursors — disjoint writes, no atomics),
// then each predicate's bucket is sorted and turned into its CSR halves
// independently. Both loops parallelize; the result is identical for any
// thread count because bucket contents are fully sorted afterwards.
void TripleStore::build(std::vector<Triple> triples, unsigned threads) {
  if (triples.empty()) return;
  constexpr size_t kGrain = 64u << 10;
  size_t chunk_count = (triples.size() + kGrain - 1) / kGrain;

  // 1. Distinct predicates, ascending — position becomes PredIndex.
  std::vector<std::vector<TermId>> chunk_preds(chunk_count);
  parallel_for(chunk_count, 1, threads, [&](size_t b, size_t e, size_t) {
    for (size_t c = b; c < e; ++c) {
      std::vector<TermId>& preds = chunk_preds[c];
      size_t hi = std::min((c + 1) * kGrain, triples.size());
      for (size_t i = c * kGrain; i < hi; ++i)
        if (preds.empty() || preds.back() != triples[i].p) preds.push_back(triples[i].p);
      sort_unique(preds);
    }
  });
  for (const std::vector<TermId>& preds : chunk_preds) {
    std::vector<TermId> merged;
    merge_union(pred_terms_, preds, merged);
    pred_terms_ = std::move(merged);
  }
  size_t n_preds = pred_terms_.size();
  auto pred_of = [&](TermId p) {
    return static_cast<size_t>(std::lower_bound(pred_terms_.begin(), pred_terms_.end(), p) -
                               pred_terms_.begin());
  };

  // 2. Count each chunk's contribution per bucket.
  std::vector<uint32_t> counts(chunk_count * n_preds, 0);
  parallel_for(chunk_count, 1, threads, [&](size_t b, size_t e, size_t) {
    for (size_t c = b; c < e; ++c) {
      uint32_t* row = counts.data() + c * n_preds;
      size_t hi = std::min((c + 1) * kGrain, triples.size());
      for (size_t i = c * kGrain; i < hi; ++i) ++row[pred_of(triples[i].p)];
    }
  });

  // 3. Exclusive scan (predicate-major, chunk-minor) -> per-chunk cursors.
  std::vector<size_t> bucket_begin(n_preds + 1, 0);
  std::vector<size_t> starts(chunk_count * n_preds);
  size_t cursor = 0;
  for (size_t p = 0; p < n_preds; ++p) {
    bucket_begin[p] = cursor;
    for (size_t c = 0; c < chunk_count; ++c) {
      starts[c * n_preds + p] = cursor;
      cursor += counts[c * n_preds + p];
    }
  }
  bucket_begin[n_preds] = cursor;

  // 4. Scatter into buckets.
  std::vector<Triple> buckets(triples.size());
  parallel_for(chunk_count, 1, threads, [&](size_t b, size_t e, size_t) {
    std::vector<size_t> cur;
    for (size_t c = b; c < e; ++c) {
      cur.assign(starts.begin() + static_cast<ptrdiff_t>(c * n_preds),
                 starts.begin() + static_cast<ptrdiff_t>((c + 1) * n_preds));
      size_t hi = std::min((c + 1) * kGrain, triples.size());
      for (size_t i = c * kGrain; i < hi; ++i) buckets[cur[pred_of(triples[i].p)]++] = triples[i];
    }
  });
  triples = {}; // the scatter copy supersedes the input

  // 5. Sort each bucket and emit its CSR halves, one predicate at a time.
  preds_.resize(n_preds);
  std::vector<size_t> kept(n_preds);
  parallel_for(n_preds, 1, threads, [&](size_t b, size_t e, size_t) {
    for (size_t p = b; p < e; ++p) {
      Triple* lo = buckets.data() + bucket_begin[p];
      Triple* hi = buckets.data() + bucket_begin[p + 1];
      std::sort(lo, hi, [](const Triple& a, const Triple& b) {
        return std::tie(a.s, a.o) < std::tie(b.s, b.o);
      });
      hi = std::unique(lo, hi,
                       [](const Triple& a, const Triple& b) { return a.s == b.s && a.o == b.o; });
      kept[p] = static_cast<size_t>(hi - lo);

      PredicateCsr& csr = preds_[p];
      for (Triple* t = lo; t != hi; ++t) {
        if (csr.subjects.empty() || csr.subjects.back() != t->s) {
          csr.subjects.push_back(t->s);
          csr.fwd_offsets.push_back(static_cast<uint32_t>(csr.fwd_objects.size()));
        }
        csr.fwd_objects.push_back(t->o);
      }
      csr.fwd_offsets.push_back(static_cast<uint32_t>(csr.fwd_objects.size()));

      std::sort(lo, hi, [](const Triple& a, const Triple& b) {
        return std::tie(a.o, a.s) < std::tie(b.o, b.s);
      });
      for (Triple* t = lo; t != hi; ++t) {
        if (csr.objects.empty() || csr.objects.back() != t->o) {
          csr.objects.push_back(t->o);
          csr.rev_offsets.push_back(static_cast<uint32_t>(csr.rev_subjects.size()));
        }
        csr.rev_subjects.push_back(t->s);
      }
      csr.rev_offsets.push_back(static_cast<uint32_t>(csr.rev_subjects.size()));
    }
  });
  for (size_t k : kept) triple_count_ += k;
}

PredIndex TripleStore::pred_index(TermId predicate) const {
  auto it = std::lower_bound(pred_terms_.begin(), pred_terms_.end(), predicate);
  if (it == pred_terms_.end() || *it != predicate) return kNoPred;
  return static_cast<PredIndex>(it - pred_terms_.begin());
}

IdSpan TripleStore::objects_of(TermId s, PredIndex p) const {
  if (p == kNoPred) return {};
  const PredicateCsr& csr = preds_[p];
  auto it = std::lower_bound(csr.subjects.begin(), csr.subjects.end(), s);
  if (it == csr.subjects.end() || *it != s) return {};
  size_t pos = static_cast<size_t>(it - csr.subjects.begin());
  return {csr.fwd_objects.data() + csr.fwd_offsets[pos],
          csr.fwd_offsets[pos + 1] - csr.fwd_offsets[pos]};
}

IdSpan TripleStore::subjects_of(TermId o, PredIndex p) const {
  if (p == kNoPred) return {};
  const PredicateCsr& csr = preds_[p];
  auto it = std::lower_bound(csr.objects.begin(), csr.objects.end(), o);
  if (it == csr.objects.end() || *it != o) return {};
  size_t pos = static_cast<size_t>(it - csr.objects.begin());
  return {csr.rev_subjects.data() + csr.rev_offsets[pos],
          csr.rev_offsets[pos + 1] - csr.rev_offsets[pos]};
}

} // namespace bgpgen

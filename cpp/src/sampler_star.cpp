#include "bgpgen/sampler_star.hpp"

#include "bgpgen/intern_map.hpp"

#include <algorithm>

namespace bgpgen {

namespace {

constexpr size_t kMaxReshuffles = 300;
constexpr size_t kMinNewPerPass = 5;

/// Fills `out` with `k` distinct values from [0, n), in random pick order
/// (rejection sampling — k is tiny, so the linear duplicate scan is cheap).
void sample_distinct(uint32_t n, uint32_t k, std::mt19937_64& rng, std::vector<uint32_t>& out) {
  out.clear();
  while (out.size() < k) {
    uint32_t r = static_cast<uint32_t>(rng() % n);
    if (std::find(out.begin(), out.end(), r) == out.end()) out.push_back(r);
  }
}

} // namespace

std::vector<StarRecord> sample_stars(const TripleStore& store, const SubjectIndex& index,
                                     const TermTable& terms, const StarOptions& opt, Dedup& dedup,
                                     std::mt19937_64& rng, SampleReport* report) {
  std::vector<StarRecord> out;
  SampleReport local;
  SampleReport& rep = report ? *report : local;

  // Candidates: IRI subjects with out-degree >= size.
  std::vector<uint32_t> stars;
  for (size_t i = 0; i < index.iri_subject_count(); ++i)
    if (index.degree(i) >= opt.size) stars.push_back(static_cast<uint32_t>(i));
  if (stars.empty()) {
    rep.stop_reason = "no subjects with degree >= size";
    return out;
  }
  const size_t max_consecutive = std::max<size_t>(500, stars.size() / 10);

  std::shuffle(stars.begin(), stars.end(), rng);
  size_t star_i = 0;
  size_t pass_start_count = 0;
  size_t consecutive_dups = 0;

  std::vector<uint32_t> picks, inst;
  std::vector<std::string> hash_parts;
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  while (out.size() < opt.num) {
    if (star_i >= stars.size()) {
      size_t new_in_pass = out.size() - pass_start_count;
      if (new_in_pass == 0) {
        rep.stop_reason = "dataset exhausted (no new queries in a full pass)";
        break;
      }
      if (new_in_pass < kMinNewPerPass) {
        rep.stop_reason = "diminishing returns (<" + std::to_string(kMinNewPerPass) +
                          " new queries in a pass)";
        break;
      }
      if (rep.reshuffles >= kMaxReshuffles) {
        rep.stop_reason = "max reshuffles reached";
        break;
      }
      std::shuffle(stars.begin(), stars.end(), rng);
      star_i = 0;
      pass_start_count = out.size();
      ++rep.reshuffles;
      consecutive_dups = 0;
    }

    size_t pos = stars[star_i++];
    std::span<const SubjectIndex::Entry> row = index.row(pos);
    uint32_t deg = static_cast<uint32_t>(row.size());
    uint32_t k = opt.size;

    // Select k distinct (pred, obj) pairs, then choose how many / which
    // object positions get the constant object.
    sample_distinct(deg, k, rng, picks);
    uint32_t n_inst =
        std::uniform_int_distribution<uint32_t>(opt.min_objects, opt.max_objects)(rng);
    if (n_inst > 0)
      sample_distinct(k, n_inst, rng, inst);
    else
      inst.clear();
    auto obj_bound = [&](uint32_t i) {
      return std::find(inst.begin(), inst.end(), i) != inst.end();
    };

    StarRecord rec;
    Query& q = rec.query;
    q.var_names.push_back("?s"); // var 0
    q.patterns.reserve(k);

    for (uint32_t i = 0; i < k; ++i) {
      const SubjectIndex::Entry& e = row[picks[i]];
      TermId pred_term = store.pred_term(e.pred);
      rec.meta.predicates_used.push_back(pred_term);

      QTerm p;
      if (unit(rng) <= opt.p_predicate) {
        p = pred_term;
        q.x.push_back(pred_term);
      } else {
        p = qvar(static_cast<uint32_t>(q.var_names.size()));
        q.var_names.push_back("?p" + std::to_string(i));
      }

      QTerm o;
      if (obj_bound(i)) {
        o = e.obj;
        rec.meta.instantiated_objects.push_back(e.obj);
      } else {
        o = qvar(static_cast<uint32_t>(q.var_names.size()));
        q.var_names.push_back("?o" + std::to_string(i));
      }
      q.patterns.push_back({qvar(0), p, o});
    }

    if (!dedup.insert(q, terms)) {
      ++rep.duplicates_skipped;
      if (++consecutive_dups >= max_consecutive) {
        rep.stop_reason = "too many consecutive duplicates (" +
                          std::to_string(consecutive_dups) + ")";
        break;
      }
      continue;
    }
    consecutive_dups = 0;

    // Python hash_query_pattern: bracket-less "pred=obj" / "pred=?" parts
    // over the selected pairs (independent of predicate binding), sorted,
    // joined with "|". (Python hashes with md5; we use hash_bytes.)
    hash_parts.clear();
    for (uint32_t i = 0; i < k; ++i) {
      const SubjectIndex::Entry& e = row[picks[i]];
      std::string part(terms.lexical(store.pred_term(e.pred)));
      part += '=';
      if (obj_bound(i))
        part += term_token(terms, e.obj);
      else
        part += '?';
      hash_parts.push_back(std::move(part));
    }
    std::sort(hash_parts.begin(), hash_parts.end());
    std::string joined;
    for (size_t i = 0; i < hash_parts.size(); ++i) {
      if (i) joined += '|';
      joined += hash_parts[i];
    }
    rec.meta.query_hash = hash_bytes(joined);

    rec.meta.source_subject = index.subject_at(pos);
    rec.meta.star_size = deg;
    rec.meta.object_instantiation_pattern = inst;
    rec.meta.n_objects_instantiated = n_inst;
    out.push_back(std::move(rec));
  }
  return out;
}

} // namespace bgpgen

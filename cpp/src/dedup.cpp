#include "bgpgen/dedup.hpp"

#include "bgpgen/intern_map.hpp"

#include <algorithm>
#include <array>
#include <cstring>

namespace bgpgen {

namespace {

// --- WL canonicalization ------------------------------------------------
//
// Terms live in a tagged u64 space: constants are their TermId, variables
// are kCVar | dense-var-index. Signatures are flat u64 vectors compared
// lexicographically — a different total order than Python's string tuples,
// but any consistent order yields the same equivalence classes.

constexpr uint64_t kCVar = 1ull << 63;

using CTriple = std::array<uint64_t, 3>;
using Sig = std::vector<uint64_t>;

/// Maps signature objects to dense color ranks (Python's normalize()).
std::vector<uint32_t> normalize(const std::vector<Sig>& sigs) {
  std::vector<const Sig*> sorted;
  sorted.reserve(sigs.size());
  for (const Sig& s : sigs) sorted.push_back(&s);
  std::sort(sorted.begin(), sorted.end(), [](const Sig* a, const Sig* b) { return *a < *b; });
  sorted.erase(std::unique(sorted.begin(), sorted.end(),
                           [](const Sig* a, const Sig* b) { return *a == *b; }),
               sorted.end());
  std::vector<uint32_t> colors(sigs.size());
  for (size_t v = 0; v < sigs.size(); ++v) {
    auto it = std::lower_bound(sorted.begin(), sorted.end(), &sigs[v],
                               [](const Sig* a, const Sig* b) { return *a < *b; });
    colors[v] = static_cast<uint32_t>(it - sorted.begin());
  }
  return colors;
}

class Canonicalizer {
public:
  explicit Canonicalizer(const std::vector<TriplePattern>& patterns) {
    std::vector<uint32_t> var_ids;
    auto enc = [&](QTerm t) -> uint64_t {
      if (!q_is_var(t)) return t;
      uint32_t id = q_var_id(t);
      for (size_t i = 0; i < var_ids.size(); ++i)
        if (var_ids[i] == id) return kCVar | i;
      var_ids.push_back(id);
      return kCVar | (var_ids.size() - 1);
    };
    for (const TriplePattern& tp : patterns)
      triples_.push_back({enc(tp.s), enc(tp.p), enc(tp.o)});
    n_vars_ = var_ids.size();
  }

  std::string key() {
    std::vector<CTriple> canon;
    if (n_vars_ == 0) {
      canon = triples_;
      std::sort(canon.begin(), canon.end());
    } else {
      canon = search(refine(initial_colors()));
    }
    std::string out;
    out.resize(canon.size() * 24);
    std::memcpy(out.data(), canon.data(), out.size());
    return out;
  }

private:
  static bool is_var(uint64_t t) { return (t & kCVar) != 0; }
  static size_t var_index(uint64_t t) { return static_cast<size_t>(t & ~kCVar); }

  std::vector<uint32_t> initial_colors() const {
    std::vector<Sig> sigs(n_vars_);
    for (size_t v = 0; v < n_vars_; ++v) {
      std::vector<Sig> occs;
      for (const CTriple& tri : triples_) {
        for (int pos = 0; pos < 3; ++pos) {
          if (tri[pos] != (kCVar | v)) continue;
          Sig occ{static_cast<uint64_t>(pos)};
          // Constant context: (pos2, term) for non-var other positions.
          std::vector<std::pair<uint64_t, uint64_t>> cctx;
          Sig vctx;
          for (int p2 = 0; p2 < 3; ++p2) {
            if (p2 == pos) continue;
            if (is_var(tri[p2]))
              vctx.push_back(static_cast<uint64_t>(p2));
            else
              cctx.emplace_back(static_cast<uint64_t>(p2), tri[p2]);
          }
          std::sort(cctx.begin(), cctx.end());
          occ.push_back(cctx.size());
          for (auto& [p2, t] : cctx) {
            occ.push_back(p2);
            occ.push_back(t);
          }
          occ.push_back(vctx.size());
          occ.insert(occ.end(), vctx.begin(), vctx.end());
          occs.push_back(std::move(occ));
        }
      }
      std::sort(occs.begin(), occs.end());
      for (const Sig& o : occs) {
        sigs[v].push_back(o.size());
        sigs[v].insert(sigs[v].end(), o.begin(), o.end());
      }
    }
    return normalize(sigs);
  }

  std::vector<uint32_t> refine(std::vector<uint32_t> colors) const {
    for (size_t round = 0; round < n_vars_; ++round) {
      std::vector<Sig> sigs(n_vars_);
      for (size_t v = 0; v < n_vars_; ++v) {
        std::vector<Sig> nbr;
        for (const CTriple& tri : triples_) {
          for (int pos = 0; pos < 3; ++pos) {
            if (tri[pos] != (kCVar | v)) continue;
            // (p2, kind, value) per other position; kind 1 = var (by color).
            std::vector<std::array<uint64_t, 3>> parts;
            for (int p2 = 0; p2 < 3; ++p2) {
              if (p2 == pos) continue;
              if (is_var(tri[p2]))
                parts.push_back({static_cast<uint64_t>(p2), 1, colors[var_index(tri[p2])]});
              else
                parts.push_back({static_cast<uint64_t>(p2), 0, tri[p2]});
            }
            std::sort(parts.begin(), parts.end());
            Sig occ{static_cast<uint64_t>(pos)};
            for (auto& p : parts) occ.insert(occ.end(), p.begin(), p.end());
            nbr.push_back(std::move(occ));
          }
        }
        std::sort(nbr.begin(), nbr.end());
        sigs[v].push_back(colors[v]);
        for (const Sig& o : nbr) {
          sigs[v].push_back(o.size());
          sigs[v].insert(sigs[v].end(), o.begin(), o.end());
        }
      }
      std::vector<uint32_t> next = normalize(sigs);
      if (next == colors) break;
      colors = std::move(next);
    }
    return colors;
  }

  std::vector<CTriple> search(const std::vector<uint32_t>& colors) const {
    // Group variables by color, ascending.
    uint32_t max_color = 0;
    for (uint32_t c : colors) max_color = std::max(max_color, c);
    std::vector<std::vector<size_t>> groups(max_color + 1);
    for (size_t v = 0; v < n_vars_; ++v) groups[colors[v]].push_back(v);

    // All singleton: assign canonical numbers by first appearance over the
    // abstract-sorted triple order.
    bool all_single = true;
    for (const auto& g : groups)
      if (g.size() > 1) all_single = false;

    if (all_single) {
      std::vector<CTriple> abstract = triples_;
      for (CTriple& tri : abstract)
        for (uint64_t& t : tri)
          if (is_var(t)) t = kCVar | colors[var_index(t)];
      std::vector<size_t> order(triples_.size());
      for (size_t i = 0; i < order.size(); ++i) order[i] = i;
      std::sort(order.begin(), order.end(),
                [&](size_t a, size_t b) { return abstract[a] < abstract[b]; });

      std::vector<uint64_t> mapping(n_vars_, kInvalidTerm);
      uint64_t counter = 0;
      for (size_t idx : order)
        for (uint64_t t : triples_[idx])
          if (is_var(t) && mapping[var_index(t)] == kInvalidTerm)
            mapping[var_index(t)] = counter++;

      std::vector<CTriple> renamed = triples_;
      for (CTriple& tri : renamed)
        for (uint64_t& t : tri)
          if (is_var(t)) t = kCVar | mapping[var_index(t)];
      std::sort(renamed.begin(), renamed.end());
      return renamed;
    }

    // Individualize each member of the smallest ambiguous group; keep the
    // minimal result.
    const std::vector<size_t>* target = nullptr;
    for (const auto& g : groups)
      if (g.size() > 1 && (!target || g.size() < target->size())) target = &g;

    std::vector<CTriple> best;
    bool have_best = false;
    for (size_t v : *target) {
      std::vector<uint32_t> nc(colors);
      nc[v] = max_color + 1;
      std::vector<CTriple> result = search(refine(std::move(nc)));
      if (!have_best || result < best) {
        best = std::move(result);
        have_best = true;
      }
    }
    return best;
  }

  std::vector<CTriple> triples_;
  size_t n_vars_ = 0;
};

} // namespace

std::string canonical_bgp_key(const std::vector<TriplePattern>& patterns) {
  return Canonicalizer(patterns).key();
}

void Dedup::set_template(const CompiledTemplate& tmpl) {
  if (method_ != DedupMethod::Symmetry) return;
  tmpl_ = &tmpl;
  symmetries_ = template_symmetries(tmpl, kMaxSymmetries);
}

bool Dedup::symmetry_key(const Query& q, std::string& key) {
  if (symmetries_.perms.empty() || q.patterns.size() != tmpl_->triples.size()) return false;
  constexpr uint32_t kVar = UINT32_MAX, kUnset = UINT32_MAX - 1;
  const size_t m = q.patterns.size();

  // Label of every slot (constant or "variable") and of every edge. A slot
  // must carry the same label in all its triples.
  slot_label_.assign(tmpl_->n_slots, kUnset);
  labels_.resize(m);
  auto node = [&](uint8_t slot, QTerm t) {
    uint32_t label = q_is_var(t) ? kVar : t;
    if (slot_label_[slot] == kUnset) slot_label_[slot] = label;
    return slot_label_[slot] == label;
  };
  for (size_t i = 0; i < m; ++i) {
    const TemplateTriple& tt = tmpl_->triples[i];
    const TriplePattern& tp = q.patterns[i];
    if (!node(tt.s_slot, tp.s) || !node(tt.o_slot, tp.o)) return false;
    labels_[i] = {slot_label_[tt.s_slot], q_is_var(tp.p) ? kVar : tp.p, slot_label_[tt.o_slot]};
  }

  // A symmetry only reorders the edge labels; orders are compared by the
  // labels' ranks. First sort the labels inside every group of
  // interchangeable leaves, then keep the smallest order among the remaining
  // symmetries.
  rank_.assign(m, 0);
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < m; ++j) rank_[i] += labels_[j] < labels_[i];
  order_.resize(m);
  for (size_t j = 0; j < m; ++j) order_[j] = static_cast<uint8_t>(j);
  for (const std::vector<uint8_t>& group : symmetries_.twin_edges) {
    // Insertion sort of the group's edges by rank, written back to its positions.
    for (size_t a = 1; a < group.size(); ++a)
      for (size_t b = a; b > 0 && rank_[order_[group[b]]] < rank_[order_[group[b - 1]]]; --b)
        std::swap(order_[group[b]], order_[group[b - 1]]);
  }
  const std::vector<uint8_t>* best = &symmetries_.perms[0];
  for (size_t k = 1; k < symmetries_.perms.size(); ++k) {
    const std::vector<uint8_t>& perm = symmetries_.perms[k];
    for (size_t j = 0; j < m; ++j) {
      uint8_t r = rank_[order_[perm[j]]], rb = rank_[order_[(*best)[j]]];
      if (r == rb) continue;
      if (r < rb) best = &perm;
      break;
    }
  }

  key.assign(1, 'S');
  for (size_t j = 0; j < m; ++j)
    key.append(reinterpret_cast<const char*>(labels_[order_[(*best)[j]]].data()),
               3 * sizeof(uint32_t));
  return true;
}

bool Dedup::insert(const Query& q, const TermTable& terms) {
  if (method_ == DedupMethod::Symmetry) {
    std::string key;
    if (!symmetry_key(q, key)) key = "W" + canonical_bgp_key(q.patterns);
    return canon_.insert(std::move(key)).second;
  }
  if (method_ == DedupMethod::Hash) return hashes_.insert(hash_key(q, terms)).second;
  return canon_.insert(canonical_bgp_key(q.patterns)).second;
}

uint64_t Dedup::hash_key(const Query& q, const TermTable& terms) {
  // Faithful port of _hash_dedup_key: one "pred=obj" part per pattern,
  // tokens as they appear in the output triples (predicates keep their
  // brackets / variable names), variable objects collapse to "?". Sorted,
  // joined with "|", hashed. Subjects are ignored.
  parts_.clear();
  for (const TriplePattern& tp : q.patterns) {
    std::string part = q_token(q, terms, tp.p);
    part += '=';
    if (q_is_var(tp.o))
      part += '?';
    else
      part += term_token(terms, tp.o);
    parts_.push_back(std::move(part));
  }
  std::sort(parts_.begin(), parts_.end());
  std::string joined;
  for (size_t i = 0; i < parts_.size(); ++i) {
    if (i) joined += '|';
    joined += parts_[i];
  }
  return hash_bytes(joined);
}

void Dedup::key(const Query& q, const TermTable& terms, std::string& out) {
  if (method_ == DedupMethod::Symmetry) {
    if (!symmetry_key(q, out)) out = "W" + canonical_bgp_key(q.patterns);
  } else if (method_ == DedupMethod::Hash) {
    uint64_t h = hash_key(q, terms);
    out.assign(reinterpret_cast<const char*>(&h), sizeof h);
  } else {
    out = canonical_bgp_key(q.patterns);
  }
}

namespace {
uint64_t hash_of_key(const std::string& key) {
  uint64_t h = 0;
  std::memcpy(&h, key.data(), sizeof h);
  return h;
}
} // namespace

bool Dedup::contains(const std::string& key) const {
  if (method_ == DedupMethod::Hash) return hashes_.count(hash_of_key(key)) != 0;
  return canon_.count(key) != 0;
}

bool Dedup::insert_key(std::string key) {
  if (method_ == DedupMethod::Hash) return hashes_.insert(hash_of_key(key)).second;
  return canon_.insert(std::move(key)).second;
}

} // namespace bgpgen

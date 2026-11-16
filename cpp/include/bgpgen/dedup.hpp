#pragma once

#include "bgpgen/query.hpp"
#include "bgpgen/template.hpp"

#include <array>

#include <string>
#include <unordered_set>
#include <vector>

namespace bgpgen {

enum class DedupMethod { Hash, Wl, Symmetry };

/// Canonical byte string of a BGP under variable renaming: two pattern
/// lists get the same key iff they are isomorphic. Port of dedup.py's
/// canonicalize_bgp (WL-1 color refinement + individualize-and-refine),
/// operating on integer terms — constants order by TermId instead of token
/// text, which changes the canonical representative but not the equivalence
/// classes, which is all dedup needs.
std::string canonical_bgp_key(const std::vector<TriplePattern>& patterns);

/// Duplicate-query filter, port of samplers/dedup.py.
///
/// "hash": multiset of "pred_token=obj_token_or_?" parts (subject ignored) —
/// fast, correct for stars. "wl": WL-1 canonical form — correct for any
/// shape; the canonical byte string itself is the set key, so there is no
/// collision risk. "symmetry": the same classes as "wl" for queries that all
/// come from one template (set_template), without the WL search: two such
/// queries are isomorphic exactly when a symmetry of the template maps one
/// onto the other, so the key is the smallest image of the query under the
/// template's symmetries (interchangeable leaves are sorted, the remaining
/// symmetries are tried one by one). Queries for which this does not hold (a template
/// node bound in one triple and a variable in another), templates with
/// parallel edges or too many symmetries, and a Dedup without a template get
/// the WL key; the two kinds of keys never describe isomorphic queries.
class Dedup {
public:
  explicit Dedup(DedupMethod method) : method_(method) {}

  /// True if the query is new (and remembers it).
  bool insert(const Query& q, const TermTable& terms);

  /// "symmetry" only: the template of every query passed to insert() from
  /// now on. One template per Dedup; it must outlive the Dedup.
  void set_template(const CompiledTemplate& tmpl);

  size_t size() const { return method_ == DedupMethod::Hash ? hashes_.size() : canon_.size(); }

  /// Templates with more symmetries than this, not counting the reordering of
  /// interchangeable leaves, use the WL key.
  static constexpr size_t kMaxSymmetries = 720;

private:
  DedupMethod method_;
  std::unordered_set<uint64_t> hashes_;
  std::unordered_set<std::string> canon_;
  std::vector<std::string> parts_; // scratch

  bool symmetry_key(const Query& q, std::string& key);
  const CompiledTemplate* tmpl_ = nullptr;
  TemplateSymmetries symmetries_;                // no perms = use WL
  std::vector<uint8_t> order_;                   // scratch: edge at each position
  std::vector<uint32_t> slot_label_;             // scratch
  std::vector<std::array<uint32_t, 3>> labels_;  // scratch, per edge
  std::vector<uint8_t> rank_;                    // scratch, per edge
};

} // namespace bgpgen

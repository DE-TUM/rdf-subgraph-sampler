#pragma once

#include "bgpgen/intern_map.hpp"
#include "bgpgen/types.hpp"

#include <cassert>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace bgpgen {

/// Dictionary encoder for RDF terms.
///
/// While loading, terms are interned with provisional ids that carry their
/// kind in the top two bits. freeze() renumbers everything into the final
/// segmented id space (IRIs, then blank nodes, then literals) and rewrites
/// the caller's triple buffers — pure arithmetic, no lookup tables. IRI ids
/// are identical before and after freeze.
///
/// IRIs are stored bracket-less so the same IRI unifies across subject,
/// predicate, and object positions. Literals are stored as the RAW N-Triples
/// object token (quotes, escapes, @lang / ^^<dt> suffix included) so output
/// round-trips the input bytes verbatim — exactly what the Python sampler
/// emits.
class TermTable {
public:
  /// `arena_chunk_bytes` sizes the string arena's allocation unit; the
  /// parallel loader uses small chunk-local tables, the shared table the
  /// default.
  explicit TermTable(size_t arena_chunk_bytes = 16u << 20);
  TermTable(const TermTable&) = delete;
  TermTable& operator=(const TermTable&) = delete;

  // --- Building (before freeze) -------------------------------------------

  /// `iri` is the bracket-less IRI text.
  TermId intern_iri(std::string_view iri);

  /// Blank node labels are scoped to their source document: the same label
  /// from different documents yields distinct terms.
  TermId intern_bnode(std::string_view label, uint32_t document);

  /// `raw_token` is the full N-Triples literal token, byte-for-byte.
  TermId intern_literal(std::string_view raw_token);

  /// Index remap produced by absorb(): provisional ids of the absorbed table
  /// to provisional ids of the absorbing one.
  struct MergeRemap {
    std::vector<TermId> iris, bnodes, literals;
    TermId map(TermId provisional) const;
  };

  /// Merges every term of `src` (also pre-freeze) into this table. The
  /// parallel N-Triples loader folds chunk-local tables into the shared one
  /// with this, then rewrites the chunk's triples via the returned remap.
  MergeRemap absorb(const TermTable& src);

  /// Renumbers provisional ids into the final segmented space and rewrites
  /// the given triple vectors in place. Call exactly once, after all loads.
  void freeze(std::span<std::vector<Triple>* const> triple_sets);

  // --- Frozen access --------------------------------------------------------

  size_t size() const { return iris_.count() + bnodes_.count() + literals_.count(); }
  size_t iri_count() const { return iris_.count(); }
  TermId bnode_base() const { return bnode_base_; }
  TermId literal_base() const { return literal_base_; }

  TermKind kind(TermId id) const {
    assert(frozen_);
    return id < bnode_base_ ? TermKind::Iri
                            : (id < literal_base_ ? TermKind::BNode : TermKind::Literal);
  }
  bool is_iri(TermId id) const { return id < bnode_base_; }
  bool is_bnode(TermId id) const { return id >= bnode_base_ && id < literal_base_; }
  bool is_literal(TermId id) const {
    assert(frozen_);
    return id >= literal_base_;
  }

  /// IRI text (bracket-less), blank node label, or raw literal token.
  std::string_view lexical(TermId id) const;

  /// Lookup without interning; kInvalidTerm if absent. Valid before and after
  /// freeze (IRI ids are stable).
  TermId find_iri(std::string_view iri) const;

private:
  // One arena-backed string space per term kind. The intern map stores only
  // (hash, index); KeyOf resolves an index back to its key bytes, so a Space
  // must never move once constructed (TermTable is non-movable).
  struct Space;
  struct KeyOf {
    const Space* space;
    std::string_view operator()(uint32_t index) const;
  };
  struct Space {
    Space() : map(KeyOf{this}) {}
    std::vector<std::unique_ptr<char[]>> chunks;
    size_t arena_bytes = 16u << 20;
    size_t chunk_used = 0;
    size_t chunk_size = 0;
    std::vector<const char*> ptrs;     // start of each entry's key bytes
    std::vector<uint32_t> key_lens;    // full key length (lexical + suffix)
    InternMap<KeyOf> map;

    uint32_t count() const { return static_cast<uint32_t>(ptrs.size()); }
    uint32_t intern(std::string_view key, bool& inserted);
  };

  Space iris_;
  Space bnodes_;
  Space literals_;

  std::string key_scratch_;

  bool frozen_ = false;
  TermId bnode_base_ = 0;
  TermId literal_base_ = 0;
};

} // namespace bgpgen

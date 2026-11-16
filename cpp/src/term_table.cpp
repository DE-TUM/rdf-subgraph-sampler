#include "bgpgen/term_table.hpp"

#include <cstring>

namespace bgpgen {

namespace {

constexpr uint32_t kKindShift = 30;
constexpr uint32_t kIndexMask = (1u << kKindShift) - 1;
constexpr uint32_t kBNodeTag = 1u << kKindShift;
constexpr uint32_t kLiteralTag = 2u << kKindShift;

} // namespace

// --- Space ------------------------------------------------------------------

std::string_view TermTable::KeyOf::operator()(uint32_t index) const {
  return {space->ptrs[index], space->key_lens[index]};
}

uint32_t TermTable::Space::intern(std::string_view key, bool& inserted) {
  uint64_t h = hash_bytes(key);
  uint32_t found = map.find(key, h);
  if (found != InternMap<KeyOf>::kNotFound) {
    inserted = false;
    return found;
  }
  if (chunk_used + key.size() > chunk_size) {
    chunk_size = std::max(arena_bytes, key.size());
    chunks.push_back(std::make_unique<char[]>(chunk_size));
    chunk_used = 0;
  }
  char* dst = chunks.back().get() + chunk_used;
  std::memcpy(dst, key.data(), key.size());
  chunk_used += key.size();
  ptrs.push_back(dst);
  key_lens.push_back(static_cast<uint32_t>(key.size()));
  uint32_t index = count() - 1;
  if (index >= kIndexMask) throw BgpError("term table overflow (more than 2^30 terms of a kind)");
  map.insert(h, index);
  inserted = true;
  return index;
}

// --- TermTable ----------------------------------------------------------------

TermTable::TermTable(size_t arena_chunk_bytes) {
  iris_.arena_bytes = bnodes_.arena_bytes = literals_.arena_bytes = arena_chunk_bytes;
}

TermId TermTable::intern_iri(std::string_view iri) {
  assert(!frozen_);
  bool inserted;
  return iris_.intern(iri, inserted);
}

TermId TermTable::intern_bnode(std::string_view label, uint32_t document) {
  assert(!frozen_);
  key_scratch_.assign(label);
  key_scratch_.append(reinterpret_cast<const char*>(&document), 4);
  bool inserted;
  return bnodes_.intern(key_scratch_, inserted) | kBNodeTag;
}

TermId TermTable::intern_literal(std::string_view raw_token) {
  assert(!frozen_);
  bool inserted;
  return literals_.intern(raw_token, inserted) | kLiteralTag;
}

TermId TermTable::MergeRemap::map(TermId provisional) const {
  uint32_t index = provisional & kIndexMask;
  switch (provisional >> kKindShift) {
  case 0: return iris[index];
  case 1: return bnodes[index];
  default: return literals[index];
  }
}

TermTable::MergeRemap TermTable::absorb(const TermTable& src) {
  assert(!frozen_ && !src.frozen_);
  MergeRemap remap;
  bool inserted;

  remap.iris.resize(src.iris_.count());
  for (uint32_t i = 0; i < src.iris_.count(); ++i)
    remap.iris[i] = iris_.intern(KeyOf{&src.iris_}(i), inserted);

  // Blank node keys already embed the document id; reuse them verbatim.
  remap.bnodes.resize(src.bnodes_.count());
  for (uint32_t i = 0; i < src.bnodes_.count(); ++i)
    remap.bnodes[i] = bnodes_.intern(KeyOf{&src.bnodes_}(i), inserted) | kBNodeTag;

  // Literal keys are raw token bytes — no table-local ids inside.
  remap.literals.resize(src.literals_.count());
  for (uint32_t i = 0; i < src.literals_.count(); ++i)
    remap.literals[i] = literals_.intern(KeyOf{&src.literals_}(i), inserted) | kLiteralTag;
  return remap;
}

void TermTable::freeze(std::span<std::vector<Triple>* const> triple_sets) {
  assert(!frozen_);
  bnode_base_ = iris_.count();
  literal_base_ = bnode_base_ + bnodes_.count();
  auto remap = [&](TermId t) -> TermId {
    switch (t >> kKindShift) {
    case 0:
      return t;
    case 1:
      return (t & kIndexMask) + bnode_base_;
    default:
      return (t & kIndexMask) + literal_base_;
    }
  };
  for (std::vector<Triple>* triples : triple_sets)
    for (Triple& t : *triples) t = {remap(t.s), remap(t.p), remap(t.o)};
  frozen_ = true;
}

std::string_view TermTable::lexical(TermId id) const {
  assert(frozen_);
  if (id < bnode_base_) return {iris_.ptrs[id], iris_.key_lens[id]};
  if (id < literal_base_) {
    uint32_t i = id - bnode_base_;
    return {bnodes_.ptrs[i], bnodes_.key_lens[i] - 4};
  }
  uint32_t i = id - literal_base_;
  return {literals_.ptrs[i], literals_.key_lens[i]};
}

TermId TermTable::find_iri(std::string_view iri) const {
  uint32_t found = iris_.map.find(iri, hash_bytes(iri));
  return found == InternMap<KeyOf>::kNotFound ? kInvalidTerm : found;
}

} // namespace bgpgen

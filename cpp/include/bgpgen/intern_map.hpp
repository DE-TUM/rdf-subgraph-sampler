#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

namespace bgpgen {

/// A fast 64-bit string hash (xor-multiply over 8-byte chunks).
inline uint64_t hash_bytes(std::string_view s) {
  constexpr uint64_t kMul = 0x9E3779B97F4A7C15ull; // golden-ratio multiplier
  uint64_t h = 0x2545F4914F6CDD1Dull ^ (s.size() * kMul);
  const char* p = s.data();
  size_t n = s.size();
  while (n >= 8) {
    uint64_t chunk;
    std::memcpy(&chunk, p, 8);
    h = (h ^ chunk) * kMul;
    h ^= h >> 29;
    p += 8;
    n -= 8;
  }
  uint64_t tail = 0;
  std::memcpy(&tail, p, n);
  h = (h ^ tail) * kMul;
  h ^= h >> 32;
  return h;
}

/// Open-addressing hash map from string keys to dense u32 indices.
///
/// Keys are not stored here: the owner keeps them in an arena and supplies a
/// `GetKey` functor mapping an index back to its string_view. Stored hashes
/// make most probe comparisons a single integer compare.
template <class GetKey>
class InternMap {
public:
  static constexpr uint32_t kNotFound = 0xFFFFFFFFu;

  explicit InternMap(GetKey get_key) : get_key_(get_key) { rehash(1024); }

  uint32_t find(std::string_view key, uint64_t hash) const {
    size_t slot = hash & mask_;
    while (true) {
      const Entry& e = entries_[slot];
      if (e.index == kNotFound) return kNotFound;
      if (e.hash == hash && get_key_(e.index) == key) return e.index;
      slot = (slot + 1) & mask_;
    }
  }

  void insert(uint64_t hash, uint32_t index) {
    if ((size_ + 1) * 10 > entries_.size() * 7) rehash(entries_.size() * 2);
    insert_no_grow(hash, index);
    ++size_;
  }

  size_t size() const { return size_; }

private:
  struct Entry {
    uint64_t hash = 0;
    uint32_t index = kNotFound;
  };

  void insert_no_grow(uint64_t hash, uint32_t index) {
    size_t slot = hash & mask_;
    while (entries_[slot].index != kNotFound) slot = (slot + 1) & mask_;
    entries_[slot] = {hash, index};
  }

  void rehash(size_t capacity) {
    std::vector<Entry> old = std::move(entries_);
    entries_.assign(capacity, Entry{});
    mask_ = capacity - 1;
    for (const Entry& e : old)
      if (e.index != kNotFound) insert_no_grow(e.hash, e.index);
  }

  GetKey get_key_;
  std::vector<Entry> entries_;
  size_t mask_ = 0;
  size_t size_ = 0;
};

} // namespace bgpgen

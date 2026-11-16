#include "bgpgen/nt_parser.hpp"

#include "bgpgen/thread_pool.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cctype>
#include <cstring>
#include <memory>

namespace bgpgen {

namespace {

struct MappedFile {
  const char* data = nullptr;
  size_t size = 0;

  explicit MappedFile(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw BgpError("cannot open '" + path + "'");
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw BgpError("cannot stat '" + path + "'");
    }
    size = static_cast<size_t>(st.st_size);
    if (size > 0) {
      void* p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
      if (p == MAP_FAILED) {
        ::close(fd);
        throw BgpError("cannot mmap '" + path + "'");
      }
      data = static_cast<const char*>(p);
    }
    ::close(fd);
  }
  ~MappedFile() {
    if (data) ::munmap(const_cast<char*>(data), size);
  }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
};

void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

/// Parses one buffer (or one newline-aligned chunk of it); not reentrant.
/// Tokens without escapes are interned directly from the buffer bytes (zero
/// copy); escaped IRIs go through a scratch buffer. Literals are always
/// interned verbatim.
class NtParser {
public:
  /// `file_begin`, when given, points at the start of the whole buffer so
  /// that error messages can report absolute line numbers from a mid-buffer
  /// chunk; the prefix is only scanned on the error path.
  NtParser(const char* begin, const char* end, TermTable& terms, uint32_t document,
           const char* file_begin = nullptr)
      : p_(begin), end_(end), chunk_begin_(begin), file_begin_(file_begin ? file_begin : begin),
        terms_(terms), document_(document) {}

  std::vector<Triple> run() {
    std::vector<Triple> triples;
    while (skip_to_token()) {
      TermId s = parse_subject();
      require_space();
      TermId p = parse_predicate();
      require_space();
      TermId o = parse_object();
      skip_ws();
      expect('.');
      triples.push_back({s, p, o});
    }
    return triples;
  }

private:
  [[noreturn]] void fail(const std::string& what) const {
    size_t line = line_;
    for (const char* p = file_begin_; p < chunk_begin_; ++p) line += (*p == '\n');
    throw BgpError("N-Triples line " + std::to_string(line) + ": " + what);
  }

  /// Advances past whitespace, newlines, and comments; false at end of input.
  bool skip_to_token() {
    while (p_ != end_) {
      char c = *p_;
      if (c == '\n') {
        ++line_;
        ++p_;
      } else if (c == ' ' || c == '\t' || c == '\r') {
        ++p_;
      } else if (c == '#') {
        while (p_ != end_ && *p_ != '\n') ++p_;
      } else {
        return true;
      }
    }
    return false;
  }

  void skip_ws() {
    while (p_ != end_ && (*p_ == ' ' || *p_ == '\t')) ++p_;
  }

  void require_space() {
    if (p_ == end_ || (*p_ != ' ' && *p_ != '\t')) fail("expected whitespace");
    skip_ws();
  }

  void expect(char c) {
    if (p_ == end_ || *p_ != c) fail(std::string("expected '") + c + "'");
    ++p_;
  }

  TermId parse_subject() {
    if (p_ == end_) fail("unexpected end of input");
    if (*p_ == '<') return parse_iri();
    if (*p_ == '_') return parse_bnode();
    fail("subject must be an IRI or blank node");
  }

  TermId parse_predicate() {
    if (p_ == end_ || *p_ != '<') fail("predicate must be an IRI");
    return parse_iri();
  }

  TermId parse_object() {
    if (p_ == end_) fail("unexpected end of input");
    if (*p_ == '<') return parse_iri();
    if (*p_ == '_') return parse_bnode();
    if (*p_ == '"') return parse_literal();
    fail("object must be an IRI, blank node, or literal");
  }

  TermId parse_iri() {
    ++p_; // consume '<'
    const char* start = p_;
    bool escaped = false;
    while (p_ != end_ && *p_ != '>') {
      if (*p_ == '\\') escaped = true;
      if (*p_ == '\n') fail("newline inside IRI");
      ++p_;
    }
    if (p_ == end_) fail("unterminated IRI");
    std::string_view raw(start, static_cast<size_t>(p_ - start));
    ++p_; // consume '>'
    return terms_.intern_iri(escaped ? unescape(raw) : raw);
  }

  TermId parse_bnode() {
    ++p_;
    if (p_ == end_ || *p_ != ':') fail("expected '_:'");
    ++p_;
    const char* start = p_;
    while (p_ != end_ && *p_ != ' ' && *p_ != '\t' && *p_ != '\n' && *p_ != '\r') ++p_;
    // Labels may contain '.' but not end with one, so trailing dots belong to
    // the statement terminator (e.g. "_:b." at the end of a line).
    while (p_ > start && p_[-1] == '.') --p_;
    if (p_ == start) fail("empty blank node label");
    return terms_.intern_bnode({start, static_cast<size_t>(p_ - start)}, document_);
  }

  /// Interns the ENTIRE literal token (quotes, escapes, @lang / ^^<dt>
  /// suffix) verbatim — sampler output must round-trip the input bytes.
  TermId parse_literal() {
    const char* start = p_;
    ++p_; // consume '"'
    while (p_ != end_ && *p_ != '"') {
      if (*p_ == '\\') {
        ++p_;
        if (p_ == end_) break;
      }
      if (*p_ == '\n') fail("newline inside literal (must be escaped)");
      ++p_;
    }
    if (p_ == end_) fail("unterminated literal");
    ++p_; // consume closing '"'

    if (p_ != end_ && *p_ == '@') { // language tag
      ++p_;
      const char* lang_start = p_;
      while (p_ != end_ && (std::isalnum(static_cast<unsigned char>(*p_)) || *p_ == '-')) ++p_;
      if (p_ == lang_start) fail("empty language tag");
    } else if (p_ + 1 < end_ && p_[0] == '^' && p_[1] == '^') { // datatype
      p_ += 2;
      if (p_ == end_ || *p_ != '<') fail("expected datatype IRI");
      ++p_;
      while (p_ != end_ && *p_ != '>') {
        if (*p_ == '\n') fail("newline inside IRI");
        ++p_;
      }
      if (p_ == end_) fail("unterminated datatype IRI");
      ++p_; // consume '>'
    }
    return terms_.intern_literal({start, static_cast<size_t>(p_ - start)});
  }

  std::string_view unescape(std::string_view raw) {
    scratch_.clear();
    for (size_t i = 0; i < raw.size(); ++i) {
      if (raw[i] != '\\') {
        scratch_ += raw[i];
        continue;
      }
      if (++i == raw.size()) fail("dangling escape");
      switch (raw[i]) {
      case 't': scratch_ += '\t'; break;
      case 'b': scratch_ += '\b'; break;
      case 'n': scratch_ += '\n'; break;
      case 'r': scratch_ += '\r'; break;
      case 'f': scratch_ += '\f'; break;
      case '"': scratch_ += '"'; break;
      case '\'': scratch_ += '\''; break;
      case '\\': scratch_ += '\\'; break;
      case 'u': append_utf8(scratch_, parse_hex(raw, i, 4)); break;
      case 'U': append_utf8(scratch_, parse_hex(raw, i, 8)); break;
      default: fail("invalid escape sequence");
      }
    }
    return scratch_;
  }

  uint32_t parse_hex(std::string_view raw, size_t& i, int digits) {
    if (i + static_cast<size_t>(digits) >= raw.size()) fail("truncated unicode escape");
    uint32_t cp = 0;
    for (int d = 0; d < digits; ++d) {
      char c = raw[++i];
      uint32_t nibble = (c >= '0' && c <= '9')   ? static_cast<uint32_t>(c - '0')
                        : (c >= 'a' && c <= 'f') ? static_cast<uint32_t>(c - 'a' + 10)
                        : (c >= 'A' && c <= 'F') ? static_cast<uint32_t>(c - 'A' + 10)
                                                 : 0xFFFFFFFFu;
      if (nibble == 0xFFFFFFFFu) fail("invalid unicode escape");
      cp = (cp << 4) | nibble;
    }
    return cp;
  }

  const char* p_;
  const char* end_;
  const char* chunk_begin_;
  const char* file_begin_;
  TermTable& terms_;
  uint32_t document_;
  std::string scratch_;
  size_t line_ = 1;
};

} // namespace

std::vector<Triple> parse_nt_buffer(const char* data, size_t size, TermTable& terms,
                                    uint32_t document_id, unsigned threads, size_t chunk_bytes) {
  const char* end = data + size;
  if (size <= chunk_bytes)
    return NtParser(data, end, terms, document_id).run();

  // Chunk boundaries are a pure function of the buffer bytes (split at the
  // first newline past each chunk_bytes mark), never of the thread count.
  std::vector<const char*> bounds{data};
  while (bounds.back() + chunk_bytes < end) {
    const char* probe = bounds.back() + chunk_bytes;
    const char* nl =
        static_cast<const char*>(std::memchr(probe, '\n', static_cast<size_t>(end - probe)));
    if (!nl || nl + 1 >= end) break;
    bounds.push_back(nl + 1);
  }
  bounds.push_back(end);
  size_t n = bounds.size() - 1;

  // Parse every chunk into its own small-arena table, in parallel.
  struct Chunk {
    std::unique_ptr<TermTable> terms;
    std::vector<Triple> triples;
  };
  std::vector<Chunk> chunks(n);
  parallel_for(n, 1, threads, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) {
      chunks[i].terms = std::make_unique<TermTable>(size_t{256} << 10);
      chunks[i].triples =
          NtParser(bounds[i], bounds[i + 1], *chunks[i].terms, document_id, data).run();
    }
  });

  // Fold the chunk tables into the shared one, in chunk order (serial: this
  // touches only distinct terms, a small fraction of the parse work).
  std::vector<TermTable::MergeRemap> remaps(n);
  for (size_t i = 0; i < n; ++i) remaps[i] = terms.absorb(*chunks[i].terms);

  // Rewrite chunk-local ids and concatenate, in parallel.
  std::vector<size_t> offsets(n + 1, 0);
  for (size_t i = 0; i < n; ++i) offsets[i + 1] = offsets[i] + chunks[i].triples.size();
  std::vector<Triple> triples(offsets[n]);
  parallel_for(n, 1, threads, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) {
      const TermTable::MergeRemap& remap = remaps[i];
      Triple* out = triples.data() + offsets[i];
      for (const Triple& t : chunks[i].triples) *out++ = {remap.map(t.s), remap.map(t.p),
                                                          remap.map(t.o)};
    }
  });
  return triples;
}

std::vector<Triple> load_nt_file(const std::string& path, TermTable& terms, uint32_t document_id,
                                 unsigned threads, size_t chunk_bytes) {
  MappedFile file(path);
  return parse_nt_buffer(file.data, file.size, terms, document_id, threads, chunk_bytes);
}

} // namespace bgpgen

#pragma once

#include <cstdint>
#include <cstdio>
#include <ostream>
#include <string_view>
#include <vector>

namespace bgpgen {

/// Minimal streaming JSON writer matching Python's json.dump(..., indent=2)
/// output: 2-space indent, ": " key separator, "," item separator, and
/// ensure_ascii escaping (non-ASCII as \uXXXX, astral codepoints as
/// surrogate pairs) — so files diff cleanly against the Python tool's.
class JsonWriter {
public:
  explicit JsonWriter(std::ostream& out) : out_(out) {}

  void begin_array() { open('['); }
  void end_array() { close(']'); }
  void begin_object() { open('{'); }
  void end_object() { close('}'); }

  void key(std::string_view k) {
    item_prefix();
    write_string(k);
    out_ << ": ";
    pending_value_ = true;
  }

  void string(std::string_view s) {
    value_prefix();
    write_string(s);
  }
  void number(int64_t v) {
    value_prefix();
    out_ << v;
  }
  void number(uint64_t v) {
    value_prefix();
    out_ << v;
  }
  void number(uint32_t v) { number(static_cast<uint64_t>(v)); }
  void number(int v) { number(static_cast<int64_t>(v)); }

private:
  void indent(size_t depth) {
    out_ << '\n';
    for (size_t i = 0; i < depth * 2; ++i) out_ << ' ';
  }

  /// Newline + indent before an array item or object key; comma if not first.
  void item_prefix() {
    if (stack_.empty()) return;
    if (stack_.back() > 0) out_ << ',';
    ++stack_.back();
    indent(stack_.size());
  }

  void value_prefix() {
    if (pending_value_) {
      pending_value_ = false;
      return;
    }
    item_prefix();
  }

  void open(char c) {
    value_prefix();
    out_ << c;
    stack_.push_back(0);
  }

  void close(char c) {
    size_t items = stack_.back();
    stack_.pop_back();
    if (items > 0) indent(stack_.size());
    out_ << c;
  }

  void write_string(std::string_view s) {
    out_ << '"';
    size_t i = 0;
    while (i < s.size()) {
      unsigned char c = static_cast<unsigned char>(s[i]);
      if (c == '"') {
        out_ << "\\\"";
        ++i;
      } else if (c == '\\') {
        out_ << "\\\\";
        ++i;
      } else if (c == '\n') {
        out_ << "\\n";
        ++i;
      } else if (c == '\t') {
        out_ << "\\t";
        ++i;
      } else if (c == '\r') {
        out_ << "\\r";
        ++i;
      } else if (c == '\b') {
        out_ << "\\b";
        ++i;
      } else if (c == '\f') {
        out_ << "\\f";
        ++i;
      } else if (c >= 0x20 && c < 0x7F) {
        out_ << static_cast<char>(c);
        ++i;
      } else if (c < 0x20) {
        escape(c);
        ++i;
      } else {
        // Decode one UTF-8 sequence; on malformed input pass the byte through
        // as U+00XX (matches nothing Python would produce, but never garbles).
        uint32_t cp = c;
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) {
          cp = c & 0x1Fu;
          len = 2;
        } else if ((c & 0xF0) == 0xE0) {
          cp = c & 0x0Fu;
          len = 3;
        } else if ((c & 0xF8) == 0xF0) {
          cp = c & 0x07u;
          len = 4;
        }
        if (i + len <= s.size() && len > 1) {
          bool ok = true;
          for (size_t j = 1; j < len; ++j) {
            unsigned char cc = static_cast<unsigned char>(s[i + j]);
            if ((cc & 0xC0) != 0x80) {
              ok = false;
              break;
            }
            cp = (cp << 6) | (cc & 0x3Fu);
          }
          if (!ok) {
            cp = c;
            len = 1;
          }
        } else {
          cp = c;
          len = 1;
        }
        if (cp >= 0x10000) {
          uint32_t v = cp - 0x10000;
          escape(0xD800 + (v >> 10));
          escape(0xDC00 + (v & 0x3FF));
        } else {
          escape(cp);
        }
        i += len;
      }
    }
    out_ << '"';
  }

  void escape(uint32_t cp) {
    char buf[8];
    std::snprintf(buf, sizeof buf, "\\u%04x", cp);
    out_ << buf;
  }

  std::ostream& out_;
  std::vector<size_t> stack_; // item count per open container
  bool pending_value_ = false;
};

} // namespace bgpgen

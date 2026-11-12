# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0
#
# Portable SHA-256, sourced by the fetch scripts: `sha256sum` (GNU coreutils)
# or `shasum -a 256` (macOS, Perl). Prints the bare hex digest of stdin or $1.
sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$@" | cut -d' ' -f1
  else echo "need sha256sum or shasum" >&2; return 1; fi
}
# verify <file> <expected-hex> [keep]: on mismatch deletes the file so a rerun
# refetches it, unless a third argument says the file is not ours to delete
verify() {
  local got; got=$(sha256 "$1")
  if [[ "$got" != "$2" ]]; then
    echo "checksum mismatch for $1: $got != $2" >&2
    [[ -n "${3:-}" ]] || rm -f "$1"
    return 1
  fi
  echo "$1: OK"
}

#!/usr/bin/env bash
set -euo pipefail
root=${1:?Usage: build_mockturtle.sh MOCKTURTLE_ROOT OUTPUT_BINARY}
out=${2:?output required}
script=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
includes=("-I$root/include")
for path in "$root"/lib/*; do
  [[ -d "$path" ]] && includes+=("-I$path" "-I$path/include")
done
c++ -std=c++17 -O2 -DNDEBUG -DFMT_HEADER_ONLY "${includes[@]}" "$script/mockturtle_driver.cpp" -o "$out"

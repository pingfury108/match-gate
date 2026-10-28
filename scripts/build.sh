#!/usr/bin/env bash
# 在 pi400 上构建 release 版（llama.cpp 是 C++，首次编译 30-50 分钟）。
set -euo pipefail
cd "$(dirname "$0")/.."

# bindgen 需要 libclang；Debian bullseye 装在 /usr/lib/llvm-11/lib。
if [ -z "${LIBCLANG_PATH:-}" ] && [ -d /usr/lib/llvm-11/lib ]; then
    export LIBCLANG_PATH=/usr/lib/llvm-11/lib
fi

# pi400 只有 4 核 3.7GB 内存，并行度别开太高。
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-3}"

exec cargo build --release "$@"

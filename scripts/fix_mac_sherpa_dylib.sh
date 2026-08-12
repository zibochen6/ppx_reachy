#!/usr/bin/env bash
# 修复 macOS 上 sherpa-onnx wheel 缺失 libonnxruntime dylib 的问题。
#
# 现象: `import sherpa_onnx` 报
#   Library not loaded: @rpath/libonnxruntime.1.27.0.dylib
# 原因: mac wheel 的 _sherpa_onnx.so 动态链接 onnxruntime 的共享库,
#       但 wheel 里没有携带它; onnxruntime pip 包里有, 只是路径不对。
# 修复: 把 onnxruntime 的 dylib 软链进 sherpa_onnx/lib/ (rpath 会找到)。
#
# 用法: bash scripts/fix_mac_sherpa_dylib.sh [venv 目录, 默认 .venv]
set -euo pipefail

VENV="${1:-.venv}"
SITE=$(ls -d "$VENV"/lib/python*/site-packages 2>/dev/null | head -1)
if [[ -z "$SITE" ]]; then
  echo "找不到 site-packages: $VENV" >&2
  exit 1
fi

SP_LIB="$SITE/sherpa_onnx/lib"
ORT_CAPI="$SITE/onnxruntime/capi"

if [[ ! -d "$SP_LIB" ]]; then
  echo "sherpa_onnx 未安装: $SP_LIB" >&2
  exit 1
fi
if [[ ! -d "$ORT_CAPI" ]]; then
  echo "onnxruntime 未安装: $ORT_CAPI" >&2
  exit 1
fi

fixed=0
for dylib in "$ORT_CAPI"/libonnxruntime.*.dylib; do
  [[ -e "$dylib" ]] || continue
  name=$(basename "$dylib")
  if [[ ! -e "$SP_LIB/$name" ]]; then
    ln -s "$dylib" "$SP_LIB/$name"
    echo "链接 $name"
    fixed=1
  fi
done

# onnxruntime >= 1.27 的主 dylib 还依赖 @rpath/libonnxruntime.1.dylib
if [[ ! -e "$SP_LIB/libonnxruntime.1.dylib" ]]; then
  main=$(ls "$ORT_CAPI"/libonnxruntime.*.dylib 2>/dev/null | head -1 || true)
  if [[ -n "$main" ]]; then
    ln -s "$main" "$SP_LIB/libonnxruntime.1.dylib"
    echo "链接 libonnxruntime.1.dylib"
    fixed=1
  fi
fi

if [[ "$fixed" -eq 0 ]]; then
  echo "没有需要修复的链接 (已修复过?)"
else
  echo "✅ 已修复。验证: $VENV/bin/python -c 'import sherpa_onnx'"
fi

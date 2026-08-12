#!/usr/bin/env python3
"""Download the sherpa-onnx KWS model and generate wake-word keywords.

Usage:
    uv run python scripts/download_kws_model.py
    uv run python scripts/download_kws_model.py --keywords 皮皮虾,小柴
    uv run python scripts/download_kws_model.py --model-dir models/kws

Downloads the Chinese keyword-spotting model
(sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01, ~33 MB tarball,
resumable HTTPS download) and prepares the model directory consumed by
``WakeWordDetector``:

    encoder.onnx / decoder.onnx / joiner.onnx   (int8 variants, symlinked)
    tokens.txt / keywords.txt                    (ppinyin tokens per wake word)
    test_wavs/                                   (model self-test audio)

The wake words are converted to pinyin tokens with the exact same logic as
sherpa-onnx's ``text2token --tokens-type ppinyin`` (pypinyin + tone-split),
so no CLI/sentencepiece dependency is needed.  All tokens are validated
against ``tokens.txt`` before writing ``keywords.txt``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tarfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("download_kws_model")

MODEL_NAME = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
DEFAULT_URL = (
    f"https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    f"{MODEL_NAME}.tar.bz2"
)
PREFERRED_ENCODER = "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
PREFERRED_DECODER = "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
PREFERRED_JOINER = "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"


def to_ppinyin(word: str) -> list[str]:
    """Convert one wake word to ppinyin tokens (sherpa text2token logic)."""
    try:
        from pypinyin import pinyin
        from pypinyin.contrib.tone_convert import to_finals_tone, to_initials
    except ImportError:
        logger.error("需要 pypinyin 包: 先运行 `uv pip install pypinyin`")
        raise

    tokens: list[str] = []
    for syllable in pinyin(word):
        x = syllable[0]
        initial = to_initials(x, strict=False)
        final = to_finals_tone(x, strict=False)
        if initial == "" and final == "":
            tokens.append(x)
        else:
            if initial:
                tokens.append(initial)
            if final:
                tokens.append(final)
    return tokens


def generate_keywords(words: list[str], tokens_file: Path) -> Path:
    """Write model_dir/keywords.txt; raise if any token is unknown."""
    known = set()
    with tokens_file.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                known.add(parts[0])

    out = tokens_file.parent / "keywords.txt"
    lines: list[str] = []
    for word in words:
        tokens = to_ppinyin(word)
        unknown = [t for t in tokens if t not in known]
        if unknown:
            logger.error(
                "唤醒词 %r 的 token %s 不在 tokens.txt 中(拼音分词不匹配)",
                word, unknown,
            )
            raise SystemExit(1)
        lines.append(" ".join(tokens) + f" @{word}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("keywords.txt: %d 个唤醒词 -> %s", len(lines), out)
    return out


def download_resumable(url: str, dest: Path, expected_bytes: int) -> None:
    """Download with byte-range resume; safe to re-run after interruption."""
    import httpx

    headers = {"User-Agent": "Mozilla/5.0"}
    with httpx.Client(
        timeout=httpx.Timeout(60.0, connect=15.0),
        headers=headers,
        follow_redirects=True,  # GitHub releases 302 到 CDN
    ) as client:
        existing = dest.stat().st_size if dest.is_file() else 0
        if existing >= expected_bytes:
            logger.info("已存在完整文件，跳过下载: %s", dest)
            return
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            logger.info("断点续传: %s (已有 %d / %d 字节)", dest, existing, expected_bytes)
        try:
            with client.stream("GET", url, headers=headers) as resp:
                resp.raise_for_status()
                mode = "ab" if existing > 0 else "wb"
                with dest.open(mode) as f:
                    for chunk in resp.iter_bytes(1 << 16):
                        f.write(chunk)
        except Exception:
            logger.warning("下载中断(可重新运行以续传): %s", dest)
            raise
        actual = dest.stat().st_size
        if actual < expected_bytes:
            raise RuntimeError(f"下载不完整: {actual} / {expected_bytes} 字节")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并准备 sherpa-onnx KWS 唤醒模型")
    parser.add_argument("--keywords", default="皮皮虾",
                        help="唤醒词,逗号分隔 (默认: 皮皮虾)")
    parser.add_argument("--model-dir", default="models/kws",
                        help="模型输出目录 (默认: models/kws)")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="模型 tar.bz2 下载地址 (默认: GitHub release;"
                             " 国内可换 ModelScope 镜像)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tarball = model_dir / f"{MODEL_NAME}.tar.bz2"

    # 1. Download (resumable)
    import httpx

    try:
        head = httpx.head(
            args.url, follow_redirects=True,
            timeout=httpx.Timeout(30.0, connect=15.0),
        )
        expected = int(head.headers.get("content-length") or 0)
    except Exception:
        expected = 0
        logger.warning("无法获取文件大小,将直接下载")
    if not expected:
        logger.warning("GitHub 下载失败或不可达;可换 --url 镜像。")
        # 仍然尝试下载(依赖 Range 的服务器必须返回 content-length)
    download_resumable(args.url, tarball, expected)

    # 2. Extract
    extract_dir = model_dir / MODEL_NAME
    if not (extract_dir / PREFERRED_ENCODER).is_file():
        logger.info("解压模型...")
        with tarfile.open(tarball, "r:bz2") as tf:
            tf.extractall(model_dir, filter="data")
    src = extract_dir
    for name in (PREFERRED_ENCODER, PREFERRED_DECODER, PREFERRED_JOINER, "tokens.txt"):
        if not (src / name).is_file():
            raise RuntimeError(f"模型包缺少文件: {name}")
    if not (src / "tokens.txt").is_file():
        raise RuntimeError("模型包缺少 tokens.txt")

    # 3. Stable names (symlink to the chosen variants)
    mapping = {
        "encoder.onnx": PREFERRED_ENCODER,
        "decoder.onnx": PREFERRED_DECODER,
        "joiner.onnx": PREFERRED_JOINER,
    }
    for stable, actual in mapping.items():
        link = model_dir / stable
        if link.is_symlink() or link.is_file():
            link.unlink()
        link.symlink_to(Path(MODEL_NAME) / actual)
    shutil.copyfile(src / "tokens.txt", model_dir / "tokens.txt")

    # 4. Self-check audio (model's own test wavs + keywords)
    test_wavs = model_dir / "test_wavs"
    if (src / "test_wavs").is_dir():
        if test_wavs.exists():
            shutil.rmtree(test_wavs)
        shutil.copytree(src / "test_wavs", test_wavs)
    else:
        logger.warning("模型包无 test_wavs,跳过自检音频")

    # 5. Wake words -> keywords.txt
    words = [w.strip() for w in args.keywords.split(",") if w.strip()]
    if not words:
        raise SystemExit("--keywords 不能为空")
    generate_keywords(words, model_dir / "tokens.txt")

    logger.info(
        "✅ KWS 模型就绪: %s (唤醒词: %s)",
        model_dir, "、".join(words),
    )
    logger.info("下一步: uv run chaihuo-reachy test 检查 KWS 自检")


if __name__ == "__main__":
    main()

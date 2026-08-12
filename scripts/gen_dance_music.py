"""Synthesize simple backing tracks for each dance style (pure numpy)."""
import numpy as np
import wave
from pathlib import Path

SR = 44100
DUR = 32.0


def env(n, decay):
    t = np.arange(n) / SR
    return np.exp(-t / decay)


def kick(sr):
    n = int(0.12 * sr)
    t = np.arange(n) / sr
    return (np.sin(2 * np.pi * (55 + 120 * np.exp(-t / 0.02)) * t) * env(n, 0.04)).astype(np.float32)


def snare(sr):
    n = int(0.09 * sr)
    noise = np.random.default_rng(7).uniform(-1, 1, n).astype(np.float32)
    return (noise * env(n, 0.018) * 0.7).astype(np.float32)


def hat(sr, tone=0.3):
    n = int(0.03 * sr)
    noise = np.random.default_rng(3).uniform(-1, 1, n).astype(np.float32)
    return (noise * env(n, 0.006) * tone).astype(np.float32)


def bass_note(sr, freq, dur):
    n = int(dur * sr)
    t = np.arange(n) / sr
    return (np.sin(2 * np.pi * freq * t) * env(n, dur * 0.6) * 0.25).astype(np.float32)


def build_track(bpm, pattern, bass_freqs, swing=False):
    beat = 60.0 / bpm
    total_beats = int(DUR / beat) + 2
    length = int((total_beats + 1) * beat * SR)
    track = np.zeros(length, dtype=np.float32)
    k, s, h = kick(SR), snare(SR), hat(SR)
    for b in range(total_beats):
        start = int(b * beat * SR)
        steps = pattern[b % len(pattern)]
        for i, hit in enumerate(steps):
            off = int(i * (beat / len(steps)) * SR)
            pos = start + off
            if hit in (1, 3) and pos + len(k) < length:
                track[pos:pos + len(k)] += k
            if hit in (2, 3) and pos + len(s) < length:
                track[pos:pos + len(s)] += s
            if hit and pos + len(h) < length:
                track[pos:pos + len(h)] += h * 0.35
        # bass on each beat (or swung)
        f = bass_freqs[b % len(bass_freqs)]
        note = bass_note(SR, f, beat * 0.9)
        pos = start
        if pos + len(note) < length:
            track[pos:pos + len(note)] += note
    peak = np.abs(track).max() or 1.0
    return (track / peak * 0.85 * 32767).astype(np.int16)


def write(name, pcm):
    path = Path("music") / f"{name}.wav"
    path.parent.mkdir(exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    print("wrote", path, pcm.shape[0] / SR, "s")


# happy: 120 BPM, four-on-the-floor kick + offbeat hats
write("happy", build_track(120, [(1, 0, 0, 0, 1, 0, 0, 0),
                                  (1, 0, 1, 0, 1, 0, 1, 0)], [220, 220, 262, 220]))
# swing: 100 BPM, kick on 1&3, snare on 2&4, swung feel
write("swing", build_track(100, [(1, 0, 2, 0, 1, 0, 2, 0),
                                  (1, 0, 2, 0, 1, 0, 2, 2)], [196, 196, 220, 196]))
# robot: 90 BPM, stiff on-beat kick + snare, no swing
write("robot", build_track(90, [(1, 0, 0, 0, 2, 0, 0, 0)], [110, 110, 110, 147]))
# random: 115 BPM mash
write("random", build_track(115, [(1, 2, 0, 0, 1, 2, 0, 0),
                                   (1, 2, 1, 0, 1, 2, 1, 2)], [165, 165, 196, 165]))

"""Generate the two original FreeHoops beats (procedural, no samples, no third-party
material, so they are free to use on YouTube/TikTok). Writes WAV; encode with ffmpeg."""
import numpy as np, sys, wave
SR = 48000
rng = np.random.default_rng(7)

def env(n, a=0.002, d=0.2):
    t = np.arange(n) / SR
    e = np.minimum(1, t / a) * np.exp(-t / d)
    return e

def kick(n):
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t / 0.04)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * env(n, 0.001, 0.18) * 1.0

def snare(n):
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    noise = np.convolve(noise, [1, -0.6], "same")
    tone = np.sin(2 * np.pi * 190 * t) * env(n, 0.001, 0.05)
    return (noise * env(n, 0.001, 0.12) * 0.45 + tone * 0.5)

def hat(n, open_=False):
    noise = rng.standard_normal(n)
    noise = noise - np.convolve(noise, np.ones(4) / 4, "same")
    return noise * env(n, 0.0005, 0.12 if open_ else 0.03) * 0.25

def clap(n):
    out = np.zeros(n)
    for k, off in enumerate((0, 0.01, 0.02)):
        i = int(off * SR)
        m = n - i
        out[i:] += rng.standard_normal(m) * env(m, 0.0005, 0.03 if k < 2 else 0.15)
    return out * 0.3

def note_freq(m):
    return 440.0 * 2 ** ((m - 69) / 12)

def bass(n, m):
    t = np.arange(n) / SR
    f = note_freq(m)
    s = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) + 0.15 * np.sign(np.sin(2 * np.pi * f * t))
    return s * env(n, 0.005, 0.35) * 0.45

def stab(n, chord):
    t = np.arange(n) / SR
    s = np.zeros(n)
    for m in chord:
        f = note_freq(m)
        for det in (-0.004, 0.004):
            ph = 2 * np.pi * f * (1 + det) * t
            s += (2 * ((ph / (2 * np.pi)) % 1) - 1)  # saw
    s = np.convolve(s, np.ones(8) / 8, "same")
    return s * env(n, 0.003, 0.22) * 0.07

def pad(n, chord):
    t = np.arange(n) / SR
    s = np.zeros(n)
    for m in chord:
        f = note_freq(m)
        s += np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2.003 * t)
    a = np.minimum(1, t / 0.4) * np.minimum(1, (t[-1] - t) / 0.3 + 1e-9)
    return s * a * 0.035

def place(buf, x, pos):
    end = min(len(buf), pos + len(x))
    buf[pos:end] += x[: end - pos]

def track(bpm, bars, prog, bass_pat, swing=0.0, style="hiphop"):
    beat = 60 / bpm
    step = beat / 4
    n = int(bars * 4 * beat * SR)
    L = np.zeros(n); R = np.zeros(n)
    for bar in range(bars):
        chord, root = prog[bar % len(prog)]
        b0 = bar * 4 * beat
        intro = bar < 2
        for s16 in range(16):
            t = b0 + s16 * step + (swing * step if s16 % 2 else 0)
            p = int(t * SR)
            if style == "hiphop":
                kick_on = s16 in (0, 7, 10) or (s16 == 14 and bar % 2)
                snare_on = s16 in (4, 12)
            else:
                kick_on = s16 % 4 == 0
                snare_on = s16 in (4, 12)
            if kick_on and not (intro and s16 != 0):
                k = kick(int(0.4 * SR)); place(L, k, p); place(R, k, p)
            if snare_on and not intro:
                sn = snare(int(0.3 * SR)) + clap(int(0.3 * SR)) * 0.6
                place(L, sn * 0.9, p); place(R, sn, p)
            if s16 % 2 == 0 or style != "hiphop":
                h = hat(int(0.2 * SR), open_=(s16 == 14 and style != "hiphop"))
                place(L, h * (0.8 if s16 % 4 else 1.0), p); place(R, h * 0.7, p)
            m = bass_pat[s16]
            if m is not None and not intro:
                bs = bass(int(step * 2.2 * SR), root + m)
                place(L, bs, p); place(R, bs, p)
            if s16 in (0, 6, 10) and style != "hiphop" and not intro:
                st = stab(int(step * 2 * SR), chord); place(L, st, p); place(R, st * 0.9, p)
        pd = pad(int(4 * beat * SR), chord)
        place(L, pd * 1.1, int(b0 * SR)); place(R, pd * 0.9, int(b0 * SR))
        if style == "hiphop" and not intro:
            for s16 in (2, 10):
                st = stab(int(step * 3 * SR), chord)
                p = int((b0 + s16 * step) * SR)
                place(L, st * 0.8, p); place(R, st, p)
    mix = np.stack([L, R], 1)
    mix = np.tanh(mix * 1.4) / np.tanh(1.4)
    mix /= max(1e-6, np.abs(mix).max()) / 0.89
    return mix

def write(path, mix):
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())

# Beat 1: laid-back boom-bap, 92 BPM, A minor
prog1 = [([57, 60, 64], 33), ([53, 57, 60], 29), ([48, 52, 55], 36), ([55, 59, 62], 31)]
bp1 = [0, None, None, 0, None, None, 0, None, 12, None, 7, None, None, None, 10, None]
write(sys.argv[1], track(92, 48, prog1, bp1, swing=0.18, style="hiphop"))
# Beat 2: upbeat four-on-the-floor, 118 BPM, D minor
prog2 = [([62, 65, 69], 38), ([58, 62, 65], 34), ([60, 64, 67], 36), ([57, 61, 64], 33)]
bp2 = [0, None, 12, None, 0, None, 12, None, 0, None, 12, None, 0, 7, 12, None]
write(sys.argv[2], track(118, 64, prog2, bp2, style="house"))

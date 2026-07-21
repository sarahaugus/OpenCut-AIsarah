"""Video assembly service — FFmpeg-based montaż engine.

Accepts a timeline config with clips, effects, transitions, audio overlay,
and text overlays, then builds and executes an FFmpeg command to produce
the final video. Output goes to /tmp/ and is served as a download URL.
"""

import asyncio
import json
import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEMP_DIR = "/tmp/opencut_assembly"


def _wrap_text(text: str, max_chars: int = 22) -> list[str]:
    """Word-wrap text for video overlay."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + len(w) + 1 <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [text]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}

# ── xfade transition name mapping ──────────────────────────────────────
# Maps OpenCut frontend transition types to FFmpeg xfade transition names.
XFADE_MAP: dict[str, str] = {
    "cross-dissolve": "fade",
    "dip-black": "fadeblack",
    "fade-white": "fadewhite",
    "slide-left": "slideleft",
    "slide-right": "slideright",
    "wipe-left": "wipeleft",
    "wipe-right": "wiperight",
    "push": "smoothleft",
    "zoom": "zoomin",
    "iris-wipe": "circleopen",
    "clock-wipe": "clock",
    "morph": "dissolve",
    "dissolve-zoom": "dissolve",
    "band-slide": "hslide",
    "checkerboard": "fade",
    "glitch": "fade",
    "film-burn": "fade",
    "page-peel": "fade",
    "spin": "fade",
    "cube-spin": "fade",
}

# ── FFmpeg effect filter builders ──────────────────────────────────────

def _build_video_filters(effects: list[dict]) -> list[str]:
    """Build list of FFmpeg filter strings from effect definitions."""
    filters: list[str] = []
    for ef in effects:
        t = ef.get("type", "")
        p = ef.get("params", {})
        if t == "blur":
            intensity = p.get("intensity", 15)
            sigma = max(intensity / 5, 0.001)
            filters.append(f"gblur=sigma={sigma}")
        elif t == "color-adjust":
            parts = []
            b = p.get("brightness")
            c = p.get("contrast")
            s = p.get("saturation")
            if b is not None:
                parts.append(f"brightness={b}")
            if c is not None:
                parts.append(f"contrast={c}")
            if s is not None:
                parts.append(f"saturation={s}")
            if parts:
                filters.append("eq=" + ":".join(parts))
            temp = p.get("temperature")
            if temp is not None:
                kelvin = 6500 + temp * 500
                filters.append(f"colortemperature=temperature={kelvin}")
        elif t == "sharpen":
            intensity = p.get("intensity", 0.5)
            filters.append(f"unsharp=lx=3:ly=3:la={intensity}")
        elif t == "chromatic-aberration":
            intensity = p.get("intensity", 5)
            filters.append(f"chromashift=cbh={intensity}:cbv={intensity}")
        elif t == "vignette":
            intensity = p.get("intensity", 0.5)
            filters.append(f"vignette=PI/4.5:max_eval=4:aspect=1/1:dither=0:angle={intensity}")
        elif t == "film-grain":
            intensity = p.get("intensity", 15)
            size = p.get("size", 1)
            filters.append(f"noise=alls={intensity}:allf=t+u,smartblur=lr={size}:ls={size}")
        elif t == "motion-blur":
            samples = p.get("samples", 3)
            filters.append(f"tmix=frames={samples}")
        elif t == "posterize":
            levels = p.get("levels", 4)
            filters.append(f"posterize=levels={levels}")
        elif t == "rgb-split":
            offset = p.get("offset", 5)
            filters.append(f"rgbashift=rh={offset}:gh=0:bh={-offset}:rv={offset}:gv=0:bv={-offset}")
        elif t == "lens-distortion":
            distortion = p.get("distortion", 0.1)
            fov = p.get("fov", 1.0)
            filters.append(f"lenscorrection=cx=0.5:cy=0.5:k1={distortion}:k2=0")
        elif t == "chroma-key":
            color = p.get("keyColor", "#00b140")
            tolerance = p.get("tolerance", 0.35)
            softness = p.get("softness", 0.08)
            colorname = color.lstrip("#")
            # colorkey uses 0-1 range for tolerance/softness
            filters.append(f"colorkey=color=0x{colorname}:similarity={1 - tolerance}:blend={softness}")
        elif t == "duotone":
            c1 = p.get("color1", "#ff0000")
            c2 = p.get("color2", "#0000ff")
            c1 = c1.lstrip("#")
            c2 = c2.lstrip("#")
            r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
            r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
            filters.append(
                f"colorchannelmixer=rr=0:rg=0:rb=0:"
                f"gr=0:gg=0:gb=0:"
                f"br=0:bg=0:bb=0,"
                f"lutrgb=r={r1/255}*X+{r2/255}*(65535-X)/65535:"
                f"g={g1/255}*X+{g2/255}*(65535-X)/65535:"
                f"b={b1/255}*X+{b2/255}*(65535-X)/65535"
            )
        elif t == "glitch":
            intensity = p.get("intensity", 0.3)
            # Approximate glitch: random noise + color shifting
            filters.append(f"noise=alls=15:allf=t+u,chromashift=cbh={intensity * 20}")
    return filters


# ── Filter preset lookup ───────────────────────────────────────────────

FILTER_PRESETS: dict[str, dict[str, float]] = {
    "grayscale": {"brightness": 0, "contrast": 1, "saturation": 0, "temperature": 0},
    "sepia": {"brightness": 0.05, "contrast": 0.95, "saturation": 0.3, "temperature": 0.6},
    "vintage": {"brightness": -0.05, "contrast": 1.15, "saturation": 0.6, "temperature": 0.3},
    "warm": {"brightness": 0.03, "contrast": 1.05, "saturation": 1.1, "temperature": 0.5},
    "cool": {"brightness": 0, "contrast": 1.05, "saturation": 0.9, "temperature": -0.5},
    "vivid": {"brightness": 0.02, "contrast": 1.3, "saturation": 1.8, "temperature": 0},
    "muted": {"brightness": 0.05, "contrast": 0.85, "saturation": 0.5, "temperature": 0},
    "dramatic": {"brightness": -0.08, "contrast": 1.5, "saturation": 0.7, "temperature": -0.2},
    "high-key": {"brightness": 0.2, "contrast": 0.8, "saturation": 0.8, "temperature": 0.1},
    "low-key": {"brightness": -0.15, "contrast": 1.4, "saturation": 0.6, "temperature": -0.1},
    "noir": {"brightness": -0.1, "contrast": 1.6, "saturation": 0, "temperature": 0},
    "golden-hour": {"brightness": 0.05, "contrast": 1.1, "saturation": 1.2, "temperature": 0.7},
    "sunset-glow": {"brightness": 0.08, "contrast": 1.15, "saturation": 1.3, "temperature": 0.8},
    "moonlight": {"brightness": -0.1, "contrast": 1.2, "saturation": 0.3, "temperature": -0.7},
    "cyberpunk": {"brightness": -0.05, "contrast": 1.6, "saturation": 1.5, "temperature": -0.6},
    "film-noir-bw": {"brightness": -0.15, "contrast": 1.8, "saturation": 0, "temperature": 0},
    "dreamy": {"brightness": 0.15, "contrast": 0.75, "saturation": 0.7, "temperature": 0.2},
    "retro-vhs": {"brightness": 0.05, "contrast": 1.3, "saturation": 1.4, "temperature": 0.4},
    "cinematic-teal-orange": {"brightness": -0.03, "contrast": 1.25, "saturation": 1.1, "temperature": 0.35},
    "bleach-bypass": {"brightness": -0.05, "contrast": 1.7, "saturation": 0.35, "temperature": 0.1},
    "cross-process": {"brightness": 0.1, "contrast": 1.2, "saturation": 1.6, "temperature": -0.3},
    "faded-film": {"brightness": 0.12, "contrast": 0.9, "saturation": 0.55, "temperature": 0.15},
}


def _build_audio_effect_filters(audio_effects: list[dict]) -> list[str]:
    """Build FFmpeg audio filter strings."""
    filters: list[str] = []
    for ef in audio_effects:
        t = ef.get("type", "")
        p = ef.get("params", {})
        if t == "eq":
            low = p.get("lowGain", 0)
            mid = p.get("midGain", 0)
            high = p.get("highGain", 0)
            low_f = p.get("lowFreq", 320)
            high_f = p.get("highFreq", 3200)
            filters.append(f"equalizer=f={low_f}:t=h:width=1:g={low}")
            filters.append(f"equalizer=f=1000:width=1:g={mid}")
            filters.append(f"equalizer=f={high_f}:t=h:width=1:g={high}")
        elif t == "compressor":
            thresh = p.get("threshold", -24)
            ratio = p.get("ratio", 4)
            attack = p.get("attack", 10)
            release = p.get("release", 100)
            makeup = p.get("makeupGain", 0)
            filters.append(f"compand=attacks={attack}:decays={release}:points=-80/-80|-{abs(thresh)}/-{abs(thresh)}|0/0:gain={makeup}:volume=1")
        elif t == "noise-gate":
            thresh = p.get("threshold", -40)
            attack = p.get("attack", 5)
            release = p.get("release", 50)
            filters.append(f"agate=threshold={thresh}:attack={attack}:release={release}:ratio=100")
        elif t == "reverb":
            room = p.get("roomSize", 50) / 100
            damping = p.get("damping", 50) / 100
            wet = p.get("wetLevel", 30) / 100
            delay = 20 + room * 60
            decay = 0.3 + room * 0.6
            filters.append(f"aecho=0.8:0.88:{delay}:{decay},volume={wet}")
        elif t == "de-esser":
            freq = p.get("frequency", 6000)
            thresh = p.get("threshold", -30)
            reduction = p.get("reduction", 10)
            filters.append(f"equalizer=f={freq}:t=q:width=100:g=-{reduction},compand=attacks=0.1:decays=10:points=-80/-80|-{abs(thresh)}/-{abs(thresh)}|0/0")
        elif t == "limiter":
            ceiling = p.get("ceiling", -1)
            release = p.get("release", 50)
            filters.append(f"alimiter=limit={ceiling}dB:attack=0.1:release={release}")
    return filters


# ── Assembly runner ────────────────────────────────────────────────────

class AssemblyService:
    """Build and run FFmpeg commands for video assembly."""

    async def assemble(
        self,
        config: dict[str, Any],
        file_map: dict[int, str],  # file_index -> saved temp path
    ) -> str:
        """Run the full assembly pipeline. Returns path to output video."""
        os.makedirs(TEMP_DIR, exist_ok=True)
        output_name = config.get("output_name") or uuid.uuid4().hex[:12]
        output_path = os.path.join(TEMP_DIR, f"{output_name}.mp4")

        timeline = config.get("timeline", config)
        clips_config = timeline.get("clips", [])
        audio_overlay = timeline.get("audio_overlay")
        texts = timeline.get("texts", [])
        output_cfg = timeline.get("output", {})

        if not clips_config:
            raise ValueError("No clips defined in timeline")

        width = output_cfg.get("resolution", "1920x1080").split("x")[0]
        height = output_cfg.get("resolution", "1920x1080").split("x")[1]
        fps = output_cfg.get("fps", 30)

        # ── Step 0: Calculate auto clip duration ──────────────────

        auto_clip_duration: float | None = None
        output_duration = output_cfg.get("duration")
        if output_duration and len(clips_config) > 0:
            auto_clip_duration = output_duration / len(clips_config)
        elif audio_overlay:
            ao_list = audio_overlay if isinstance(audio_overlay, list) else [audio_overlay]
            total_audio_dur = 0.0
            for ao in ao_list:
                if ao.get("file_index") is not None:
                    ap = file_map.get(ao["file_index"])
                    if ap and os.path.exists(ap):
                        total_audio_dur += self._get_media_duration(ap)
            if total_audio_dur > 0 and len(clips_config) > 0:
                auto_clip_duration = total_audio_dur / len(clips_config)

        # ── Calculate clip positions on the timeline ──────────────

        clip_positions: list[float] = []
        current_pos = 0.0
        for c in clips_config:
            clip_positions.append(current_pos)
            current_pos += self._get_clip_duration(c)

        # ── Step 1: Process each clip ─────────────────────────────

        clip_inputs: list[str] = []      # ffmpeg -i arguments
        clip_vstreams: list[str] = []    # labeled video streams
        clip_astreams: list[str] = []    # labeled audio streams
        filter_chains: list[str] = []    # full filtergraph
        input_idx = 0
        stream_idx = 0

        for i, clip_cfg in enumerate(clips_config):
            fi = clip_cfg.get("file_index", i)
            path = file_map.get(fi)
            if not path or not os.path.exists(path):
                raise ValueError(f"File index {fi} not found or uploaded")

            trim = clip_cfg.get("trim", {})
            trim_start = trim.get("start", 0)
            trim_end = trim.get("end")
            speed = clip_cfg.get("speed", 1.0)
            volume = clip_cfg.get("volume", 1.0)
            effects = clip_cfg.get("effects", [])
            filter_preset = clip_cfg.get("filter_preset")
            transition = clip_cfg.get("transition_from_previous")
            is_image = self._is_image(path)

            # Input — images use loop, video uses -ss
            if is_image:
                clip_inputs.extend(["-loop", "1", "-framerate", str(fps)])
                dur = trim_end - trim_start if trim_end is not None else auto_clip_duration
                if dur is not None and dur > 0:
                    clip_inputs.extend(["-t", str(dur)])
                clip_inputs.extend(["-i", path])
                # Set trim.end so _get_clip_duration returns correct value
                if dur is not None and trim_end is None:
                    clip_cfg.setdefault("trim", {})["end"] = dur
            else:
                clip_inputs.extend(["-ss", str(trim_start)])
                if trim_end is not None:
                    clip_inputs.extend(["-t", str(trim_end - trim_start)])
                clip_inputs.extend(["-i", path])

            # Video filter chain
            vf_parts: list[str] = []

            # Effects
            all_vfx = list(effects)
            if filter_preset and filter_preset in FILTER_PRESETS:
                preset_params = FILTER_PRESETS[filter_preset]
                all_vfx.append({"type": "color-adjust", "params": dict(preset_params)})
            vf_parts.extend(_build_video_filters(all_vfx))

            # Speed
            if speed != 1.0:
                vf_parts.append(f"setpts={1/speed}*PTS")

            # Ensure constant frame rate (required by xfade transition)
            vf_parts.append(f"fps={fps}")

            vlabel = f"v{stream_idx}"

            # Transition handling
            if transition and i > 0:
                xtype = XFADE_MAP.get(transition.get("type", "cross-dissolve"), "fade")
                xdur = transition.get("duration", 0.5)

                prev_label = f"v{stream_idx - 1}"
                prev_dur = self._get_clip_duration(clips_config[i - 1])
                overlap_start = prev_dur - xdur

                # Use overlay with fade for transition
                # Instead of simple xfade, we overlay the new clip on the previous
                vf_str = ",".join(vf_parts)
                clip_vstreams.append(f"[{input_idx}:v]{vf_str}[{vlabel}]")
            else:
                vf_str = ",".join(vf_parts)
                clip_vstreams.append(f"[{input_idx}:v]{vf_str}[{vlabel}]")

            # Audio stream — images generate silence, videos use input audio
            alabel = f"a{stream_idx}"
            af_parts: list[str] = []
            audio_effects = clip_cfg.get("audio_effects", [])
            af_parts.extend(_build_audio_effect_filters(audio_effects))
            if volume != 1.0:
                af_parts.append(f"volume={volume}")
            if speed != 1.0:
                af_parts.append(f"atempo={max(0.5, min(2.0, speed))}")

            if is_image:
                dur = self._get_clip_duration(clip_cfg)
                if af_parts:
                    af_str = ",".join(af_parts)
                    filter_chains.append(f"aevalsrc=0:d={dur}:s=44100,{af_str}[{alabel}]")
                else:
                    filter_chains.append(f"aevalsrc=0:d={dur}:s=44100[{alabel}]")
            elif af_parts:
                af_str = ",".join(af_parts)
                clip_astreams.append(f"[{input_idx}:a]{af_str}[{alabel}]")
            else:
                clip_astreams.append(f"[{input_idx}:a]anull[{alabel}]")

            stream_idx += 1
            input_idx += 1

        # ── Step 2: Build concat / transition graph ─────────────────

        # New approach: use concat filter with all clips sequentially
        # For transitions between clips, we use xfade on the overlay

        # Simple approach: just concat all clips
        all_vlabels = "".join(f"[v{i}]" for i in range(stream_idx))
        all_alabels = "".join(f"[a{i}]" for i in range(stream_idx))

        filter_parts: list[str] = []

        # Video concat
        video_concat_inputs = "".join(f"[v{i}]" for i in range(stream_idx))
        audio_concat_inputs = "".join(f"[a{i}]" for i in range(stream_idx))

        # Build the filter graph — add per-clip defs then build concat/transition
        filter_chains.extend(clip_vstreams)
        filter_chains.extend(clip_astreams)

        # Process clips with transitions
        has_transitions = any(
            c.get("transition_from_previous") for c in clips_config[1:]
        )

        if has_transitions and stream_idx > 1:
            # Build transition overlay chain using v0 as starting point
            filter_chains.append(f"[a0]adelay=50[a_t0]")

            # Cumulative duration of the composite output (for correct xfade offset)
            composite_dur = self._get_clip_duration(clips_config[0])
            for i in range(1, stream_idx):
                trans = clips_config[i].get("transition_from_previous", {})
                xtype = XFADE_MAP.get(trans.get("type", "cross-dissolve"), "fade")
                xdur = trans.get("duration", 0.5)
                prev_v = f"v_t{i-1}" if i > 1 else "v0"
                prev_a = f"a_t{i-1}" if i > 1 else "a_t0"

                if xtype in XFADE_MAP.values():
                    filter_chains.append(
                        f"[{prev_v}][v{i}]xfade=transition={xtype}:duration={xdur}:offset={composite_dur - xdur}[v_t{i}]"
                    )
                else:
                    filter_chains.append(f"[{prev_v}][v{i}]concat=n=2:v=1:a=0[v_t{i}]")

                # Concat audio
                filter_chains.append(f"[{prev_a}][a{i}]concat=n=2:v=0:a=1[a_t{i}]")

                composite_dur = composite_dur + self._get_clip_duration(clips_config[i]) - xdur

            final_v = f"v_t{stream_idx - 1}"
            final_a = f"a_t{stream_idx - 1}"
        else:
            filter_chains.append(f"{video_concat_inputs}concat=n={stream_idx}:v=1:a=0[final_v]")
            filter_chains.append(f"{audio_concat_inputs}concat=n={stream_idx}:v=0:a=1[final_a]")
            final_v = "final_v"
            final_a = "final_a"

        # ── Step 3: Text overlays ──────────────────────────────────

        text_files: list[str] = []
        if texts:
            # Write text lines to temp files for reliable UTF-8 handling
            dt_parts: list[str] = []
            for t in texts:
                raw_text = t.get("text", "")
                clip_idx = t.get("clip_index")
                if clip_idx is not None and clip_idx < len(clip_positions):
                    start = clip_positions[clip_idx]
                    duration = self._get_clip_duration(clips_config[clip_idx])
                else:
                    start = t.get("start", 0)
                    duration = t.get("duration", 2)
                position = t.get("position", "center")
                font_size = t.get("fontSize", 36)
                color = t.get("color", "white")

                lines = _wrap_text(raw_text)
                line_count = len(lines)
                line_h = font_size * 1.3
                block_h = line_count * line_h

                x_map = {"center": "(w-text_w)/2", "left": "20", "right": "w-text_w-20"}
                x = x_map.get(position, "(w-text_w)/2")

                for li, line in enumerate(lines):
                    y_expr = {
                        "center": f"(h-{block_h})/2 + {li}*{line_h}",
                        "top": f"40 + {li}*{line_h}",
                        "bottom": f"h-40-{block_h} + {li}*{line_h}",
                    }.get(position, f"(h-{block_h})/2 + {li}*{line_h}")

                    # Write text to file for reliable UTF-8
                    tf = os.path.join(TEMP_DIR, f"txt_{uuid.uuid4().hex[:8]}.txt")
                    with open(tf, "w", encoding="utf-8") as f:
                        f.write(line)
                    text_files.append(tf)

                    dt_parts.append(
                        f"drawtext=textfile={tf}:fontsize={font_size}:fontcolor={color}:"
                        f"x={x}:y={y_expr}:"
                        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
                        f"enable='between(t,{start},{start + duration})'"
                    )

            if dt_parts:
                dt_str = ",".join(dt_parts)
                filter_chains.append(f"[{final_v}]{dt_str}[final_wtext]")
                final_v = "final_wtext"



        # ── Step 4: Audio overlay (background music) ───────────────

        total_dur = sum(
            self._get_clip_duration(c) for c in clips_config
        )

        if audio_overlay:
            ao_list = audio_overlay if isinstance(audio_overlay, list) else [audio_overlay]
            bgm_indices: list[int] = []
            total_audio_dur = 0.0

            for ao_idx, ao in enumerate(ao_list):
                if ao.get("file_index") is None:
                    continue
                audio_path = file_map.get(ao["file_index"])
                if not audio_path or not os.path.exists(audio_path):
                    continue

                clip_inputs.extend(["-i", audio_path])
                aov_volume = ao.get("volume", 1.0)
                aov_fade_in = ao.get("fade_in", 0)
                aov_fade_out = ao.get("fade_out", 0)

                aov_filters: list[str] = [f"volume={aov_volume}"]
                if aov_fade_in > 0:
                    aov_filters.append(f"afade=t=in:d={aov_fade_in}")
                if aov_fade_out > 0:
                    aov_filters.append(f"afade=t=out:st={total_dur - aov_fade_out}:d={aov_fade_out}")

                ov_effects = ao.get("effects", [])
                aov_filters.extend(_build_audio_effect_filters(ov_effects))

                aov_filter_str = ",".join(aov_filters)
                bgm_label = f"bgm{ao_idx}"
                filter_chains.append(f"[{input_idx}:a]{aov_filter_str}[{bgm_label}]")
                bgm_indices.append(ao_idx)
                total_audio_dur += self._get_media_duration(audio_path)
                input_idx += 1

            if bgm_indices:
                if len(bgm_indices) == 1:
                    filter_chains.append(
                        f"[{final_a}][bgm0]amix=inputs=2:duration=longest:dropout_transition=2[final_amix]"
                    )
                else:
                    bgm_labels = "".join(f"[bgm{i}]" for i in bgm_indices)
                    filter_chains.append(
                        f"{bgm_labels}concat=n={len(bgm_indices)}:v=0:a=1[bgm_concat]"
                    )
                    filter_chains.append(
                        f"[{final_a}][bgm_concat]amix=inputs=2:duration=longest:dropout_transition=2[final_amix]"
                    )
                final_a = "final_amix"

                # Extend video if audio overlay is longer than the composition
                if total_audio_dur > total_dur + 0.05:
                    extend = total_audio_dur - total_dur
                    filter_chains.append(
                        f"[{final_v}]tpad=stop_mode=clone:stop_duration={extend}[final_v_ext]"
                    )
                    final_v = "final_v_ext"

        # ── Step 5: Scale to output resolution ─────────────────────

        filter_chains.append(
            f"[{final_v}]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[outv]"
        )

        # ── Build final filter_complex ──────────────────────────────

        filter_complex = ";".join(filter_chains)

        cmd = [
            "ffmpeg", "-y",
            *clip_inputs,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", f"[{final_a}]",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]

        logger.info("Assembly command: %s", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error = stderr.decode("utf-8", errors="replace")[-20000:]
            logger.error("Assembly FFmpeg failed: %s", error)
            raise RuntimeError(f"FFmpeg assembly failed: {error}")

        if not os.path.exists(output_path):
            raise RuntimeError("Assembly completed but output file not found")

        logger.info("Assembly complete: %s", output_path)
        return output_path

    @staticmethod
    def _get_media_duration(filepath: str) -> float:
        """Get duration of a media file in seconds via ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                    filepath,
                ],
                capture_output=True, text=True, timeout=30,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    @staticmethod
    def _is_image(filepath: str) -> bool:
        """Check if a file is an image by extension."""
        return Path(filepath).suffix.lower() in IMAGE_EXTENSIONS

    def _get_clip_duration(self, clip_cfg: dict) -> float:
        """Get the duration of a clip after trim and speed adjustments."""
        trim = clip_cfg.get("trim", {})
        start = trim.get("start", 0)
        end = trim.get("end")
        speed = clip_cfg.get("speed", 1.0)
        raw_dur = (end - start) if end else 10.0
        return raw_dur / max(0.1, speed)

    @staticmethod
    def cleanup_old_files(max_age_seconds: int = 3600) -> None:
        """Remove assembly temp files older than max_age."""
        import time
        now = time.time()
        if not os.path.isdir(TEMP_DIR):
            return
        for fname in os.listdir(TEMP_DIR):
            fpath = os.path.join(TEMP_DIR, fname)
            if (fname.startswith("assembly_") and fname.endswith(".mp4")) or (fname.startswith("txt_") and fname.endswith(".txt")):
                age = now - os.path.getmtime(fpath)
                if age > max_age_seconds:
                    try:
                        os.remove(fpath)
                        logger.info("Cleaned old assembly: %s", fpath)
                    except OSError:
                        pass


assembly_service = AssemblyService()

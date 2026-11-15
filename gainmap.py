"""Korekce stínění objektivu (DNG opcode GainMap) pro DNG z telefonů Pixel.

Pixel ukládá korekci vinětace a barevného stínění do DNG tagu OpcodeList2
jako čtyři mapy GainMap (jedna pro každý kanál Bayerovy masky). LibRaw (a tím
i rawpy) je sám neaplikuje, takže bez tohoto kroku mají fotky tmavé a
barevně posunuté rohy.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import cv2
import numpy as np

OPCODE_LIST2 = 51009
GAIN_MAP_ID = 9


@dataclass
class GainMap:
    top: int
    left: int
    bottom: int
    right: int
    plane: int
    planes: int
    row_pitch: int
    col_pitch: int
    spacing_v: float
    spacing_h: float
    origin_v: float
    origin_h: float
    data: np.ndarray  # (points_v, points_h, map_planes) float32


def parse_opcode_list(buf: bytes) -> list[GainMap]:
    """Rozparsuje binární OpcodeList (big-endian) a vrátí všechny GainMap."""
    maps: list[GainMap] = []
    if not buf or len(buf) < 4:
        return maps
    (count,) = struct.unpack_from(">I", buf, 0)
    pos = 4
    for _ in range(count):
        if pos + 16 > len(buf):
            break
        opcode_id, _ver, _flags, size = struct.unpack_from(">IIII", buf, pos)
        pos += 16
        body = buf[pos:pos + size]
        pos += size
        if opcode_id != GAIN_MAP_ID or len(body) < 76:
            continue
        (top, left, bottom, right, plane, planes, row_pitch, col_pitch,
         pts_v, pts_h) = struct.unpack_from(">10I", body, 0)
        spacing_v, spacing_h, origin_v, origin_h = struct.unpack_from(">4d", body, 40)
        (map_planes,) = struct.unpack_from(">I", body, 72)
        n = pts_v * pts_h * map_planes
        if n == 0 or len(body) < 76 + 4 * n:
            continue
        data = np.frombuffer(body, dtype=">f4", count=n, offset=76)
        data = data.astype(np.float32).reshape(pts_v, pts_h, map_planes)
        maps.append(GainMap(top, left, bottom, right, plane, planes,
                            max(1, row_pitch), max(1, col_pitch),
                            spacing_v, spacing_h, origin_v, origin_h, data))
    return maps


def read_gain_maps(path: str) -> list[GainMap]:
    """Najde OpcodeList2 v DNG (i v pod-IFD) a vrátí mapy. Při chybě []. """
    try:
        import tifffile
    except ImportError:
        return []

    def walk(pages):
        for page in pages:
            tag = page.tags.get(OPCODE_LIST2)
            if tag is not None:
                value = tag.value
                if not isinstance(value, (bytes, bytearray)):
                    value = np.asarray(value, dtype=np.uint8).tobytes()
                found = parse_opcode_list(bytes(value))
                if found:
                    return found
            sub = getattr(page, "pages", None)
            if sub:
                found = walk(sub)
                if found:
                    return found
        return []

    try:
        with open(path, "rb") as fh, tifffile.TiffFile(fh) as tif:
            return walk(tif.pages)
    except Exception:
        return []


def apply_gain_maps(raw, maps: list[GainMap]) -> int:
    """Aplikuje mapy přímo na Bayerova data v objektu rawpy (na místě).

    Vrací počet použitých map.
    """
    img = raw.raw_image_visible
    if img.ndim != 2:
        return 0
    colors = raw.raw_colors_visible
    black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
    white = float(raw.white_level)
    height, width = img.shape
    applied = 0

    for gm in maps:
        if gm.plane != 0:
            continue
        bottom = min(gm.bottom, height)
        right = min(gm.right, width)
        if gm.top >= bottom or gm.left >= right:
            continue
        rows = np.arange(gm.top, bottom, gm.row_pitch, dtype=np.float32)
        cols = np.arange(gm.left, right, gm.col_pitch, dtype=np.float32)
        pts_v, pts_h = gm.data.shape[:2]
        # souřadnice pixelu v relativních jednotkách (0..1) -> index do mapy
        yi = (rows / max(1, height - 1) - gm.origin_v) / (gm.spacing_v or 1.0)
        xi = (cols / max(1, width - 1) - gm.origin_h) / (gm.spacing_h or 1.0)
        yi = np.clip(yi, 0, pts_v - 1).astype(np.float32)
        xi = np.clip(xi, 0, pts_h - 1).astype(np.float32)
        map_x = np.ascontiguousarray(np.broadcast_to(xi[None, :], (len(yi), len(xi))))
        map_y = np.ascontiguousarray(np.broadcast_to(yi[:, None], (len(yi), len(xi))))
        gain = cv2.remap(np.ascontiguousarray(gm.data[:, :, 0]), map_x, map_y,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        sl = (slice(gm.top, bottom, gm.row_pitch), slice(gm.left, right, gm.col_pitch))
        bl = black[colors[sl]]
        sub = (img[sl].astype(np.float32) - bl) * gain + bl
        np.clip(sub, 0, white, out=sub)
        img[sl] = (sub + 0.5).astype(img.dtype)
        applied += 1
    return applied

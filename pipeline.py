"""Zpracování RAW: načtení DNG, odšumění, tóny a barvy, doostření."""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
import rawpy

import gainmap


@dataclass
class Params:
    # tóny a barvy
    exposure: float = 0.0      # EV
    temperature: float = 0.0   # -100 (studená) .. 100 (teplá)
    tint: float = 0.0          # -100 (zelená) .. 100 (purpurová)
    highlights: float = 0.0    # -100 .. 100
    shadows: float = 0.0       # -100 .. 100
    contrast: float = 0.0      # -100 .. 100
    vibrance: float = 0.0      # -100 .. 100
    saturation: float = 0.0    # -100 .. 100
    # odšumění
    denoise_luma: float = 0.0    # 0 .. 100
    denoise_chroma: float = 0.0  # 0 .. 100
    smooth: float = 0.0          # 0 .. 100 (zjemnění oblohy mezi hvězdami)
    # detaily
    clarity: float = 0.0           # -100 .. 100
    sharpen_amount: float = 0.0    # 0 .. 100
    sharpen_radius: float = 1.0    # px (při 100 %)
    sharpen_masking: float = 20.0  # 0 .. 100 (ochrana hladkých ploch)
    # noční obloha
    bg_removal: float = 0.0        # 0 .. 100 (vyrovnání vinětace a světelného znečištění)
    black_point: float = 0.0       # 0 .. 100 (% úrovně oblohy, které se odečte)
    structure: float = 0.0         # 0 .. 100 (zvýraznění velkých struktur – Mléčné dráhy)
    stars: float = 100.0           # 0 .. 100 (jas hvězd vůči záři oblohy; 100 = beze změny)


@dataclass
class Background:
    """Hladký model jasu pozadí oblohy (vinětace + světelné znečištění)."""
    coeffs: np.ndarray   # (10, 3) koeficienty pro R, G, B
    aspect: float        # šířka / výška snímku
    ref: float           # úroveň oblohy po vyrovnání (0..1, lineárně)
    floor: np.ndarray    # (3,) spodní mez modelu – omezuje zesílení v rozích
    env: np.ndarray | None = None   # (h, w, 3) jemná oprava: zbytkové přesvětlení pozadí


# ---------------------------------------------------------------- načtení

_ILLUMINANT_K = {1: 5500, 2: 4150, 3: 2856, 4: 5500, 17: 2856, 18: 4874, 19: 6774,
                 20: 5503, 21: 6504, 22: 7504, 23: 5003, 24: 3200}


def dng_white_balance(path: str):
    """Vyvážení bílé z DNG tagu AsShotWhiteXY (takto ho ukládá Pixel).

    LibRaw tento tag neumí, a bez něj by použil denní světlo. Vrací
    násobitele [R, G, B, G] nebo None.
    """
    try:
        import tifffile

        def rat(tag):
            a = np.asarray(tag.value, dtype=np.float64).reshape(-1, 2)
            return a[:, 0] / a[:, 1]

        with open(path, "rb") as fh, tifffile.TiffFile(fh) as tif:
            tags = tif.pages[0].tags
            if 50729 not in tags or 50721 not in tags:
                return None
            x, y = rat(tags[50729])
            cm = rat(tags[50721]).reshape(3, 3)
            cc = rat(tags[50723]).reshape(3, 3) if 50723 in tags else np.eye(3)
            if 50722 in tags and 50778 in tags and 50779 in tags:
                n = (x - 0.3320) / (0.1858 - y)                      # McCamy: xy -> teplota
                temp = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33
                t1 = _ILLUMINANT_K.get(int(tags[50778].value), 2856)
                t2 = _ILLUMINANT_K.get(int(tags[50779].value), 6504)
                g = 1.0 if t1 == t2 else float(np.clip((1 / temp - 1 / t2) / (1 / t1 - 1 / t2), 0, 1))
                cm = g * cm + (1 - g) * rat(tags[50722]).reshape(3, 3)
                if 50724 in tags:
                    cc = g * cc + (1 - g) * rat(tags[50724]).reshape(3, 3)
            neutral = cc @ cm @ np.array([x / y, 1.0, (1 - x - y) / y])
            if np.any(neutral <= 0):
                return None
            m = neutral[1] / neutral
            return [float(m[0]), 1.0, float(m[2]), 1.0]
    except Exception:
        return None

def load_raw(path: str, lens_correction: bool = True) -> tuple[np.ndarray, dict]:
    """Vyvolá RAW do lineárního 16bit RGB (sRGB primárky, bez gamy).

    Vrací (obraz HxWx3 uint16, info).
    """
    info = {"gain_maps": 0}
    maps = gainmap.read_gain_maps(path) if lens_correction else []
    # souborový objekt místo cesty: funguje i s diakritikou v cestě na Windows
    with open(path, "rb") as fh, rawpy.imread(fh) as raw:
        if maps:
            try:
                info["gain_maps"] = gainmap.apply_gain_maps(raw, maps)
            except Exception as exc:  # korekce je volitelná, vyvolání pokračuje
                info["gain_map_error"] = str(exc)
        wb = {"use_camera_wb": True}
        cam = raw.camera_whitebalance
        if not cam or cam[0] <= 0 or cam[2] <= 0:      # LibRaw WB z DNG nepřečetl
            user = dng_white_balance(path)
            if user:
                wb = {"user_wb": user}
                info["wb"] = user
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            **wb,
            output_color=rawpy.ColorSpace.sRGB,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
    info["size"] = (rgb.shape[1], rgb.shape[0])
    return rgb, info


# ---------------------------------------------------------------- pomocné

def _srgb_encode(x: np.ndarray) -> np.ndarray:
    """sRGB gama. Záporné hodnoty (šum pod černým bodem) kóduje zrcadlově,
    aby se šum kolem černé zprůměroval až při odšumění a nezesvětloval oblohu."""
    a = np.minimum(np.abs(x), 1)
    enc = np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(a, 1 / 2.4) - 0.055)
    return (np.sign(x) * enc).astype(np.float32)


def _shoulder(x: np.ndarray, knee: float = 0.75) -> np.ndarray:
    """Plynulé stlačení přepalů místo tvrdého ořezu."""
    over = x > knee
    if over.any():
        span = 1.0 - knee
        x = x.copy()
        x[over] = knee + span * np.tanh((x[over] - knee) / span)
    return x


def _nlm_strength(slider: float) -> float:
    """Posuvník 0..100 -> síla odšumění (v jednotkách jasu 0..1), progresivně."""
    return 0.2 * (slider / 100.0) ** 1.5


def _nlm_slider(sigma: float, factor: float) -> float:
    """Změřený šum -> hodnota posuvníku (inverze k _nlm_strength)."""
    return 100.0 * (max(sigma, 0.0) * factor / 0.2) ** (2 / 3)


def _guided(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter (He et al.) – vyhlazení respektující hrany v `guide`."""
    k = (2 * r + 1, 2 * r + 1)
    mean_i = cv2.blur(guide, k)
    mean_p = cv2.blur(src, k)
    cov = cv2.blur(guide * src, k) - mean_i * mean_p
    var = cv2.blur(guide * guide, k) - mean_i * mean_i
    a = cov / (var + eps)
    b = mean_p - a * mean_i
    return cv2.blur(a, k) * guide + cv2.blur(b, k)


# ---------------------------------------------------------------- pozadí oblohy

def _basis(u: np.ndarray, v: np.ndarray, aspect: float) -> np.ndarray:
    """Radiální členy (vinětace objektivu) + mírný lineární gradient."""
    d = np.sqrt(aspect * aspect + 1.0)
    x = (u - 0.5) * 2 * aspect / d
    y = (v - 0.5) * 2 / d
    r2 = x * x + y * y
    r = np.sqrt(r2)
    e1 = np.maximum(r - 0.72, 0) ** 2      # okraje a rohy: vinětace se tam láme
    e2 = np.maximum(r - 0.86, 0) ** 2
    return np.stack([np.ones_like(x), r2, r2 ** 2, r2 ** 3, x, y, x * r2, y * r2, e1, e2], axis=-1)


def fit_background(img: np.ndarray, cell: int = 40) -> Background:
    """Odhadne z celého snímku hladký průběh jasu pozadí.

    Snímek se rozdělí na buňky, z každé se vezme nízký percentil (ignoruje
    hvězdy) a proloží se hladká plocha. Buňky jasnější než okolí (Mléčná
    dráha, popředí) se z prokládání postupně vyřazují, takže model popisuje
    jen pozadí a skutečná struktura oblohy zůstane zachována.
    """
    hgt, wid = img.shape[:2]
    aspect = wid / hgt
    gy, gx = hgt // cell, wid // cell
    if gy < 6 or gx < 6:
        return Background(np.zeros((10, 3)), aspect, 0.0, np.zeros(3))
    cells = img[:gy * cell, :gx * cell].reshape(gy, cell, gx, cell, 3)
    cells = cells.transpose(0, 2, 1, 3, 4).reshape(gy, gx, cell * cell, 3)
    low = np.percentile(cells, 30, axis=2).astype(np.float64) / 65535.0
    v, u = np.meshgrid((np.arange(gy) + 0.5) * cell / hgt,
                       (np.arange(gx) + 0.5) * cell / wid, indexing="ij")
    a = _basis(u.ravel(), v.ravel(), aspect)
    val = low.reshape(-1, 3)
    lum = val.sum(axis=1)
    keep = np.ones(len(lum), bool)
    for _ in range(5):
        coef, *_ = np.linalg.lstsq(a[keep], lum[keep], rcond=None)
        model = np.maximum(a @ coef, 1e-6)
        resid = (lum - model) / model
        med = np.median(resid[keep])
        sd = 1.4826 * np.median(np.abs(resid[keep] - med)) + 1e-9
        new = (resid < med + 1.2 * sd) & (resid > med - 3.0 * sd)
        if new.sum() < 0.35 * len(lum):
            break
        keep = new
    coeffs, *_ = np.linalg.lstsq(a[keep], val[keep], rcond=None)
    surf = a @ coeffs
    floor = 0.4 * surf.max(axis=0)
    bg = Background(coeffs, aspect, 1.0, floor)

    # Druhá fáze: zbytkové nerovnosti pozadí (záře u okrajů, světlo z okolí).
    # Z vyrovnaného snímku bez hvězd se vezme "spodní obálka" ve velkém okně –
    # to, co je všude kolem stejně světlé, je pozadí; užší útvary (Mléčná
    # dráha s tmavými pásy) nad obálku vyčnívají a zůstanou.
    flat = img.astype(np.float32) / 65535.0 / eval_background(bg, (0, 0, 1, 1), img.shape)
    f = 160.0 / max(hgt, wid)
    low_res = []
    for c in range(3):
        ch = cv2.medianBlur(np.ascontiguousarray(flat[:, :, c]), 5)
        low_res.append(cv2.resize(cv2.GaussianBlur(ch, (0, 0), 4), None, fx=f, fy=f,
                                  interpolation=cv2.INTER_AREA))
    low_res = np.stack(low_res, axis=-1)
    k = max(5, int(0.30 * min(low_res.shape[:2])) | 1)
    # "otevření" (eroze + dilatace): odstraní světlé útvary užší než okno, ale
    # tmavé popředí (střecha, stromy) neroztáhne do okolní oblohy
    env = cv2.morphologyEx(low_res, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                           borderType=cv2.BORDER_REPLICATE)
    env = cv2.GaussianBlur(env, (0, 0), k / 4.0, borderType=cv2.BORDER_REPLICATE)
    base = np.median(env, axis=(0, 1))
    bg.env = np.clip(env / base, 0.85, 1.6).astype(np.float32)

    # Neutralizace: nejtmavší obloha vyjde ve všech kanálech stejně (šedá).
    flat = img.astype(np.float32) / 65535.0 / eval_background(bg, (0, 0, 1, 1), img.shape)
    m = max(4, min(hgt, wid) // 12)
    dark = np.empty(3)
    for c in range(3):
        ch = cv2.GaussianBlur(cv2.medianBlur(np.ascontiguousarray(flat[:, :, c]), 5), (0, 0), 6)
        dark[c] = np.percentile(ch[m:-m, m:-m], 10)
    bg.coeffs = coeffs * dark
    bg.floor = floor * dark
    bg.ref = float(np.median(np.maximum(a @ bg.coeffs, bg.floor), axis=0).min())
    return bg


def eval_background(bg: Background, region, shape) -> np.ndarray:
    """Vyhodnotí model pro výřez `region` = (u0, v0, u1, v1) v rozsahu 0..1."""
    u0, v0, u1, v1 = region
    n = 64
    v, u = np.meshgrid(np.linspace(v0, v1, n), np.linspace(u0, u1, n), indexing="ij")
    coarse = np.maximum(_basis(u, v, bg.aspect) @ bg.coeffs, bg.floor).astype(np.float32)
    if bg.env is not None:
        eh, ew = bg.env.shape[:2]
        mx = (u * ew - 0.5).astype(np.float32)
        my = (v * eh - 0.5).astype(np.float32)
        coarse = coarse * cv2.remap(bg.env, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(coarse, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------- oddělení hvězd

def _split_stars(img: np.ndarray, p: Params, scale: float) -> np.ndarray:
    """Rozdělí lineární snímek na hladkou záři oblohy a vrstvu hvězd.

    Záře (Mléčná dráha) se získá odfiltrováním hvězd mediánem a vyhlazením,
    takže v ní není zrno. Hvězdy jsou to, co nad záři vyčnívá; z té vrstvy se
    odstraní šum (vše pod prahem) a hvězdy se ztlumí, aby po silném
    zesvětlení oblohy zůstaly drobné a nepřepálené. Nic se nedokresluje –
    obě vrstvy pocházejí ze snímku a zase se sečtou.
    """
    f = min(1.0, 0.25 / scale)                       # pracovní rozlišení: čtvrtina plného
    work = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
    unit = scale * f / 0.25                          # 1 px ve čtvrtinovém rozlišení
    sig = (2.0 + p.smooth / 100.0 * 10.0) * unit
    glow = np.empty_like(work)
    for c in range(3):
        ch = cv2.medianBlur(np.ascontiguousarray(work[:, :, c]), 5 if unit > 0.6 else 3)
        ch = cv2.medianBlur(ch, 5 if unit > 0.6 else 3)
        glow[:, :, c] = cv2.GaussianBlur(ch, (0, 0), max(0.5, sig))
    # barva záře se mění jen pozvolna -> barevné rozdíly vyhladit víc než jas
    mean = glow.mean(axis=2)
    for c in range(3):
        glow[:, :, c] = mean + cv2.GaussianBlur(glow[:, :, c] - mean, (0, 0), max(1.0, 3 * sig))
    if f < 1:
        glow = cv2.resize(glow, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    stars = img - glow
    s_sig = 0.9 * scale
    if s_sig >= 0.4:
        stars = cv2.GaussianBlur(stars, (0, 0), s_sig)        # potlačí šum pixelů, hvězdy zůstanou
    lum = stars.mean(axis=2)
    noise = 1.4826 * float(np.median(np.abs(lum - np.median(lum)))) + 1e-9
    thr = 2.2 + 2.0 * p.smooth / 100.0
    m = np.clip((lum - thr * noise) / (2.0 * noise), 0, 1)
    m = m * m * (3 - 2 * m)
    gain = np.float32(2.0 ** (-(1 - p.stars / 100.0) * 4.0))
    return np.maximum(glow, 0) + stars * (m * gain)[:, :, None]


# ---------------------------------------------------------------- hlavní zpracování

def process(lin: np.ndarray, p: Params, scale: float = 1.0,
            bg: Background | None = None, region=(0.0, 0.0, 1.0, 1.0)) -> np.ndarray:
    """Lineární uint16 RGB -> hotový sRGB uint16 RGB.

    `scale` = poměr velikosti náhledu vůči plnému rozlišení; poloměry filtrů se
    podle něj přepočítají, aby zmenšený náhled odpovídal exportu.
    """
    img = lin.astype(np.float32) / 65535.0

    # vyrovnání pozadí oblohy: vydělení hladkým modelem jasu (jako flat-field),
    # obloha pak má všude stejnou, neutrální úroveň bg.ref
    has_bg = bg is not None and bg.ref > 0
    if p.bg_removal > 0 and has_bg:
        surf = eval_background(bg, region, img.shape)
        gain = np.float32(bg.ref) / surf
        if p.bg_removal < 100:
            gain = np.power(gain, np.float32(p.bg_removal / 100.0))
        img *= gain
        del surf, gain

    # černý bod: odečte zadaný podíl úrovně oblohy (před expozicí)
    if p.black_point > 0 and has_bg:
        # Barevný šum se musí vyhladit ještě v lineárních datech: po silném
        # roztažení by šumovější kanály (R, B) vyšly po gamě tmavší a obloha
        # by zezelenala. Vyhlazují se jen rozdíly R-G a B-G, jas zůstává.
        if p.denoise_chroma > 0 and not (p.smooth > 0):
            sig = p.denoise_chroma / 100.0 * 8.0 * scale
            if sig >= 0.4:
                g = img[:, :, 1]
                # hvězdy (výrazně nad úrovní oblohy) se vynechají, aby si
                # zachovaly barvu a nevznikaly kolem nich barevné lemy
                star = cv2.dilate((img.max(axis=2) > 2.0 * bg.ref).astype(np.uint8),
                                  np.ones((3, 3), np.uint8)).astype(bool)
                w = (~star).astype(np.float32)
                wsum = np.maximum(cv2.GaussianBlur(w, (0, 0), sig), 1e-3)
                for c in (0, 2):
                    d = cv2.GaussianBlur((img[:, :, c] - g) * w, (0, 0), sig) / wsum
                    img[:, :, c] = np.where(star, img[:, :, c], g + d)
        img -= np.float32(p.black_point / 100.0 * bg.ref)

    # noční obloha se zjemněním: hladká záře + drobné hvězdy
    split = p.smooth > 0 and p.black_point > 0 and has_bg
    if split:
        img = _split_stars(img, p, scale)

    # vyvážení bílé + expozice (lineární prostor)
    t, g = p.temperature / 100.0, p.tint / 100.0
    gains = np.array([1 + 0.30 * t, 1 - 0.20 * g, 1 - 0.30 * t], np.float32)
    img *= gains * np.float32(2.0 ** p.exposure)
    img = _srgb_encode(_shoulder(img))

    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    del img
    y = np.ascontiguousarray(ycc[:, :, 0])
    cr = ycc[:, :, 1] - 0.5
    cb = ycc[:, :, 2] - 0.5

    # --- odšumění jasu (non-local means na 16 bitech)
    if p.denoise_luma > 0 and not split:
        h = _nlm_strength(p.denoise_luma) * 65535.0 / 1.2
        off = np.float32(0.2)                      # rezerva pro záporné hodnoty
        y16 = (np.clip((y + off) / (1 + off), 0, 1) * 65535.0 + 0.5).astype(np.uint16)
        y16 = cv2.fastNlMeansDenoising(
            y16, h=[float(h)], templateWindowSize=5, searchWindowSize=15,
            normType=cv2.NORM_L1)
        y = y16.astype(np.float32) / 65535.0 * (1 + off) - off

    # --- odšumění barvy (guided filter vedený jasem, hrany zůstanou)
    if p.denoise_chroma > 0:
        r = max(1, int(round((2 + p.denoise_chroma * 0.10) * scale)))
        eps = 1e-4 + (p.denoise_chroma / 100.0) ** 2 * 4e-3
        cr = _guided(y, cr, r, eps)
        cb = _guided(y, cb, r, eps)

    # --- zjemnění: vyhladí plochy mezi hvězdami, hvězdy zůstanou ostré.
    # Hvězda = pixel výrazně jasnější než své okolí (vůči změřenému šumu).
    if p.smooth > 0 and not split:
        sig = p.smooth / 100.0 * 12.0 * scale
        if sig >= 0.4:
            # Hvězdy se hledají vždy ve stejném (náhledovém) rozlišení, aby
            # export vypadal stejně jako náhled.
            f = min(1.0, 0.35 / scale)
            yw = cv2.resize(y, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else y
            local = yw - cv2.GaussianBlur(yw, (0, 0), max(1.5, 6.0 * scale * f))
            noise = 1.4826 * float(np.median(np.abs(local - np.median(local)))) + 1e-6
            # čím silnější zjemnění, tím jasnější musí hvězda být, aby zůstala
            thr = 3.0 + 6.0 * p.smooth / 100.0
            star = (local > thr * noise).astype(np.float32)
            star = cv2.dilate(star, np.ones((3, 3), np.uint8))
            keep = np.clip(cv2.GaussianBlur(star, (0, 0), 0.8) * 1.5, 0, 1)   # měkký okraj
            if f < 1:
                keep = cv2.resize(keep, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
            w = 1.0 - keep

            def fill(ch, sg):
                return cv2.GaussianBlur(ch * w, (0, 0), sg) / np.maximum(cv2.GaussianBlur(w, (0, 0), sg), 1e-3)

            y = keep * y + w * fill(y, sig)
            cr = keep * cr + w * fill(cr, 2 * sig)
            cb = keep * cb + w * fill(cb, 2 * sig)

    y = np.maximum(y, 0)

    # --- stíny / světla (lokálně podle rozmazaného jasu)
    if p.shadows or p.highlights:
        sigma = max(2.0, 0.02 * max(y.shape) )
        small = cv2.resize(y, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), max(0.8, sigma * 0.125))
        mask = cv2.resize(small, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask, 0, 1)
        ev = (1.5 * p.shadows / 100.0) * (1 - mask) ** 2 \
            + (1.0 * p.highlights / 100.0) * mask ** 2
        gain = np.exp2(ev).astype(np.float32)
        y = y * gain
        cr = cr * gain
        cb = cb * gain

    # --- kontrast (S-křivka kolem středu)
    if p.contrast:
        k = 2.0 ** (p.contrast / 100.0 * 0.8)
        yc = np.clip(y, 0, 1)
        y = np.where(yc < 0.5, 0.5 * (2 * yc) ** k, 1 - 0.5 * (2 * (1 - yc)) ** k)
        y = y.astype(np.float32)

    # --- struktura oblohy: zesílí jen velké útvary (mlhoviny, Mléčná dráha).
    # Hvězdy se nejdřív odfiltrují, takže se nezvětšují a šum se nezesiluje.
    if p.structure > 0:
        f = min(1.0, 0.25 / scale) * scale          # počítáme na čtvrtinovém rozlišení
        ys = cv2.resize(y, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else y
        k = max(3, int(round(14 * f)) | 1)
        ys = cv2.morphologyEx(ys, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        fine = cv2.GaussianBlur(ys, (0, 0), max(0.8, 10 * f))
        coarse = cv2.GaussianBlur(fine, (0, 0), max(2.0, 220 * f))
        band = cv2.resize(fine - coarse, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_CUBIC)
        y = y + np.float32(p.structure / 100.0 * 3.0) * band

    # --- zřetelnost (lokální kontrast ve středních tónech)
    if p.clarity:
        sigma = max(1.0, 30.0 * scale)
        base = cv2.GaussianBlur(y, (0, 0), sigma)
        mid = 1 - np.abs(2 * np.clip(y, 0, 1) - 1)
        y = y + (p.clarity / 100.0 * 0.6) * (y - base) * mid

    # --- doostření (unsharp mask jen na jasu, s maskou hladkých ploch)
    sigma = p.sharpen_radius * scale
    if p.sharpen_amount > 0 and sigma >= 0.3:
        detail = y - cv2.GaussianBlur(y, (0, 0), sigma)
        thr = p.sharpen_masking / 100.0 * 0.03
        if thr > 0:
            edge = cv2.GaussianBlur(np.abs(detail), (0, 0), max(1.0, 2 * sigma))
            detail *= np.clip(edge / thr, 0, 1)
        y = y + (p.sharpen_amount / 100.0 * 2.0) * detail

    # --- noční obloha: v téměř černém pozadí je barva jen šum -> potlačit
    if p.black_point > 0 and has_bg:
        fade = np.clip(y / 0.22, 0, 1) ** 1.5
        cr = cr * fade
        cb = cb * fade

    # --- sytost / živost
    if p.saturation or p.vibrance:
        mag = np.sqrt(cr * cr + cb * cb)
        f = (1 + p.saturation / 100.0) * (1 + p.vibrance / 100.0 * (1 - np.clip(mag / 0.25, 0, 1)))
        f = np.maximum(f, 0).astype(np.float32)
        cr = cr * f
        cb = cb * f

    ycc[:, :, 0] = y
    ycc[:, :, 1] = cr + 0.5
    ycc[:, :, 2] = cb + 0.5
    rgb = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
    np.clip(rgb, 0, 1, out=rgb)
    return (rgb * 65535.0 + 0.5).astype(np.uint16)


# ---------------------------------------------------------------- automatika

def auto_exposure(lin: np.ndarray) -> float:
    """Odhad expozice: střední jas na ~18 % šedé, bez zbytečných přepalů."""
    step = max(1, max(lin.shape[:2]) // 800)
    s = lin[::step, ::step].astype(np.float32) / 65535.0
    lum = 0.2126 * s[:, :, 0] + 0.7152 * s[:, :, 1] + 0.0722 * s[:, :, 2]
    med = float(np.median(lum))
    hi = float(np.percentile(lum, 99.5))
    ev = np.log2(0.18 / max(med, 1e-4))
    ev = min(ev, np.log2(1.6 / max(hi, 1e-4)))  # světla smí lehce do "ramene"
    return float(np.clip(round(ev * 20) / 20, -2.0, 5.0))


def estimate_noise(lin: np.ndarray, ev: float) -> float:
    """Odhad šumu (směrodatná odchylka jasu po gamě) ze středového výřezu."""
    hgt, wid = lin.shape[:2]
    n = min(1024, hgt, wid)
    y0, x0 = (hgt - n) // 2, (wid - n) // 2
    crop = lin[y0:y0 + n, x0:x0 + n].astype(np.float32) / 65535.0 * (2.0 ** ev)
    y = cv2.cvtColor(_srgb_encode(_shoulder(crop)), cv2.COLOR_RGB2GRAY)
    resid = y - cv2.medianBlur(y, 3)
    return float(1.4826 * np.median(np.abs(resid - np.median(resid))))


def auto_params(lin: np.ndarray) -> Params:
    ev = auto_exposure(lin)
    sigma = estimate_noise(lin, ev)
    luma = float(np.clip(round(_nlm_slider(sigma - 0.001, 1.25)), 0, 60))
    chroma = float(np.clip(luma * 1.3 + 15, 15, 90))
    return Params(
        exposure=ev, highlights=-25, shadows=25, contrast=12, vibrance=18,
        denoise_luma=luma, denoise_chroma=chroma,
        clarity=10, sharpen_amount=45, sharpen_radius=1.0, sharpen_masking=25,
    )


def auto_astro_params(lin: np.ndarray, small: np.ndarray, bg: Background) -> Params:
    """Nastavení pro noční oblohu / Mléčnou dráhu podle změřeného snímku."""
    if bg is None or bg.ref <= 0:
        return auto_params(lin)
    # vyrovnaný jas bez hvězd: jak moc vystupuje Mléčná dráha nad pozadí?
    img = small.astype(np.float32) / 65535.0
    img *= bg.ref / eval_background(bg, (0, 0, 1, 1), img.shape)
    lum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lum = cv2.GaussianBlur(cv2.medianBlur(lum, 5), (0, 0), 6)
    m = max(4, min(lum.shape) // 12)
    core = lum[m:-m, m:-m]
    lo, hi = np.percentile(core, [3, 99.7])
    span = max(float(hi - lo), 0.02 * bg.ref)
    black = max(float(lo) - 0.05 * span, 0.0)           # pozadí zůstane těsně nad černou
    bp = float(np.clip(round(black / bg.ref * 100), 0, 99))
    black = bp / 100 * bg.ref
    ev = float(np.clip(round(np.log2(0.17 / max(hi - black, 1e-5)) * 20) / 20, -2.0, 8.0))
    p = Params(exposure=ev, bg_removal=100, black_point=bp, structure=30, contrast=10,
               clarity=10, vibrance=0, saturation=35, sharpen_amount=0)
    # šum změříme na výřezu už s tímto nastavením (silné roztažení ho zvýrazní)
    hgt, wid = lin.shape[:2]
    n = min(700, hgt, wid)
    y0, x0 = (hgt - n) // 2, (wid - n) // 2
    region = (x0 / wid, y0 / hgt, (x0 + n) / wid, (y0 + n) / hgt)
    probe = process(lin[y0:y0 + n, x0:x0 + n], replace(p, clarity=0, structure=0), 1.0, bg, region)
    y = cv2.cvtColor(probe, cv2.COLOR_RGB2GRAY).astype(np.float32) / 65535.0
    resid = y - cv2.medianBlur(y, 3)
    sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    p.denoise_luma = float(np.clip(round(_nlm_slider(sigma, 2.2)), 20, 90))
    p.denoise_chroma = 100.0
    p.sharpen_amount, p.sharpen_masking = 20, 60
    p.smooth = 60
    p.stars = 55
    return p


# ---------------------------------------------------------------- uložení

def save_image(path: str, rgb16: np.ndarray, jpeg_quality: int = 95) -> None:
    """Uloží podle přípony: .jpg (8 bit), .png / .tif (16 bit)."""
    ext = "." + path.rsplit(".", 1)[-1].lower()
    bgr = cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR)
    if ext in (".jpg", ".jpeg"):
        bgr = ((bgr.astype(np.uint32) + 128) // 257).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    elif ext in (".tif", ".tiff", ".png"):
        ok, buf = cv2.imencode(ext, bgr)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    if not ok:
        raise IOError("Image encoding failed")
    with open(path, "wb") as fh:  # imencode + open zvládne diakritiku v cestě
        fh.write(buf.tobytes())

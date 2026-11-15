# Pixel RAW Enhancer

A desktop app (Python + PySide6) for enhancing RAW/DNG photos from Google Pixel phones:
noise reduction, tone and color, sharpness and detail. It is tuned especially for night sky
shots (stars, Milky Way).

## Installation and running

Requires Python 3.10+ (tested with 3.11).

### Windows (PowerShell)

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python app.py                   # or: python app.py photo.dng

You can also run the app without activating the venv: `.\.venv\Scripts\python.exe app.py`

- If activation fails with an *execution policy* error, run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
- Create the venv with the `py` launcher, not `python`. If an MSYS2 Python (`C:\msys64\...`)
  is on your `PATH`, `python` starts that one instead. It has no pip, and prebuilt packages
  (PySide6, rawpy, OpenCV) can't be installed into it.

### macOS / Linux

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python app.py

## Quick adjustments (simple controls)

After opening an image, click **Night sky** (stars, Milky Way) or **Regular photo** at the top
of the panel. The app sets everything up automatically and you just fine-tune the result
with the sliders:

- **Boost stars**: brighter and more numerous stars (to the left: smaller and dimmer)
- **Boost Milky Way**: stronger glow and dark lanes
- **Darker sky**: blacker background, more contrast
- **Smoother sky**: less grain (fewer faint stars remain)
- **More vivid colors**, **Warmer tone**

All of them start at zero. The *Reset quick adjustments to zero* button sets them back.

Below them are the detailed sliders, and the two are linked:

- moving a quick adjustment immediately updates the detailed sliders (e.g. *Boost stars*
  moves *Star brightness* and *Smoothing*),
- a manual change to a detailed slider is kept, and the quick adjustments build on top of it.

The detailed settings can be hidden with the checkbox above them.

## Other controls

- **Open** (Ctrl+O) or drag a file into the window. **Auto** is applied after loading.
- **Auto** (Ctrl+U): exposure from the histogram, noise reduction strength from the measured noise.
- **Auto night sky** (Ctrl+Shift+U): for stars and the Milky Way. Nothing is painted in,
  only brightness and color are adjusted:
  - *Background flattening*: divides the image by a smooth model of the sky brightness
    (lens vignetting and light pollution), so the sky is evenly dark and neutral everywhere.
  - *Black point*: what percentage of the sky brightness is subtracted (more = darker background).
  - *Sky structure*: enhances only large features (the Milky Way) without enlarging stars.
  - Fine-tune with *Exposure* (Milky Way brightness) and *Luminance noise* (grain).
- **Reset** (Ctrl+R): base exposure only, no adjustments.
- **Before / After** (B): compare with the unedited image.
- **100% crop** (Z): full-resolution preview. Click in the image to choose the spot.
  Judge noise and sharpness here, because the downscaled preview distorts them.
- **Exact preview** (P): a preview of the whole image computed from full resolution. It
  matches the export exactly (the fast preview only approximates noise and smoothing), but
  each change takes a few seconds.
- **Smoothing** (slider in Noise Reduction): for night sky shots, splits the image into
  a smooth glow (Milky Way) and a star layer. The glow is smoothed so it has no grain, and only
  stars that clearly stand out above the noise are kept. Higher value = smoother glow and fewer
  faint stars.
- **Star brightness** (Night Sky section): how bright the stars are relative to the sky glow.
  Lower value = smaller, non-clipped stars; 100 = the same brightening as the sky.
- **Crop** (sliders at the bottom of the panel): cuts the given percentage from each side,
  e.g. a roof or a bright edge. Applies to both the preview and the export.
- **Export** (Ctrl+E): JPEG, or 16-bit TIFF/PNG at full resolution.

## Files

- `app.py`: GUI
- `pipeline.py`: image processing (`load_raw`, `process`, `auto_params`, `save_image`)
- `gainmap.py`: lens shading correction from the DNG OpcodeList2 tag (GainMap),
  which LibRaw doesn't apply on its own

White balance for Pixel DNGs is read from the AsShotWhiteXY tag (LibRaw doesn't support it).

## Known limitations

- Export doesn't keep EXIF data, and the output is always sRGB.
- Luminance noise reduction (non-local means) is slow on 50 MP images (tens of seconds).
- Exporting a 50 MP image needs several GB of RAM.

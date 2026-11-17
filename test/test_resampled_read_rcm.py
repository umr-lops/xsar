"""End-to-end tests of the resampled read of RCM products (`RcmDataset.load_digital_number`).

Test product: RCM1 ScanSAR 50 m (SC50MB) 2020-02-14, 19 127 lines x 18 042 samples
of 20 m, HH and HV. At 1000 m, a cell is 50 x 50 pixels: 382 x 360 cells.
The corners of the image are outside the swath (zeros).

The cells are read in the order of the tiff files (before the flips done by xsar),
and compared with the pixels of the tiff files, recomputed with numpy."""

import glob
import logging

import numpy as np
import pytest
import rasterio
from rasterio.windows import Window

import xsar

PRODUCT = "RCM1_OK1050603_PK1050605_1_SC50MB_20200214_115905_HH_HV_Z010"
CHUNKS = {"line": 5000, "sample": 5000}
CELL = 50  # pixels per cell at 1000 m

logging.getLogger("xsar").setLevel(logging.WARNING)


@pytest.fixture(scope="module")
def product():
    return xsar.get_test_file(PRODUCT)


@pytest.fixture(scope="module")
def rcm(product):
    return xsar.RcmDataset(product, resolution="1000m")


@pytest.fixture(scope="module")
def at_1000m(rcm):
    return rcm.load_digital_number(resolution="1000m", chunks=dict(CHUNKS))


def pixels_of_cell(product, pol, i, j):
    """The 50 x 50 full resolution pixels of cell (i, j)."""
    path = glob.glob(f"{product}/imagery/*_{pol}.tif")[0]
    with rasterio.open(path) as src:
        return src.read(1, window=Window(CELL * j, CELL * i, CELL, CELL))


def rms_of_valid_pixels(pixels):
    valid = pixels[pixels > 0].astype(np.float64)
    return np.sqrt(np.mean(valid**2)) if valid.size else 0


def test_grid_at_1000m(at_1000m):
    """19 127 x 18 042 pixels of 20 m at 1000 m: 382 x 360 whole cells of 50 x 50 pixels.

    Cell coordinates are the cell centers: 24.5 for the first cell, and
    381 x 50 + 24.5 = 19 074.5 (line), 359 x 50 + 24.5 = 17 974.5 (sample) for the last.
    The 27 last lines and 42 last samples (less than one cell) are not used."""
    ds = at_1000m

    assert ds.sizes["line"] == 382
    assert ds.sizes["sample"] == 360
    assert ds.line[0] == 24.5
    assert ds.line[-1] == 19074.5
    assert ds.sample[0] == 24.5
    assert ds.sample[-1] == 17974.5
    assert ds.digital_number.dtype == np.float32
    assert ds.valid_pixel_fraction.dtype == np.float32
    assert ds.valid_pixel_fraction.dims == ("pol", "line", "sample")
    assert ds.valid_pixel_fraction.attrs["units"] == "1"


def test_four_cells_checked_by_hand(at_1000m):
    """Four cells of the scene, with their values (rms of the valid pixels):

    * corner cell (0, 0), outside the swath: 0, valid_pixel_fraction 0;
    * last cell (381, 359), outside the swath too: 0, valid_pixel_fraction 0;
    * cell in the middle of the scene (191, 180), full: HH 4791.75, HV 1494.51;
    * cell (379, 226) on the swath border: 1270 valid pixels out of 2500,
      valid_pixel_fraction 0.508, HH 16921.15, HV 6852.39."""
    ds = at_1000m
    hh = ds.digital_number.sel(pol="HH").values
    hv = ds.digital_number.sel(pol="HV").values
    fraction = ds.valid_pixel_fraction.sel(pol="HH").values

    assert hh[0, 0] == 0
    assert fraction[0, 0] == 0
    assert hh[381, 359] == 0
    assert fraction[381, 359] == 0
    np.testing.assert_allclose(hh[191, 180], 4791.75, atol=0.01)
    np.testing.assert_allclose(hv[191, 180], 1494.51, atol=0.01)
    np.testing.assert_allclose(fraction[191, 180], 1)
    np.testing.assert_allclose(hh[379, 226], 16921.15, atol=0.01)
    np.testing.assert_allclose(hv[379, 226], 6852.39, atol=0.01)
    np.testing.assert_allclose(fraction[379, 226], 0.508, atol=1e-6)


def test_any_cell_is_the_rms_of_its_own_pixels(at_1000m, product):
    """Same check on 20 cells straddling the swath border and 20 cells taken at random:
    each cell is the rms of the non-zero pixels it covers, and valid_pixel_fraction
    the fraction of these pixels (the same in HH and HV)."""
    ds = at_1000m
    fraction = ds.valid_pixel_fraction.sel(pol="HH").values
    n, m = fraction.shape
    rng = np.random.default_rng(0)
    border = np.argwhere((fraction > 0) & (fraction < 1))
    cells = [tuple(c) for c in border[rng.choice(len(border), 20, replace=False)]]
    cells += [(rng.integers(n), rng.integers(m)) for _ in range(20)]

    np.testing.assert_array_equal(ds.valid_pixel_fraction.sel(pol="HV"), fraction)
    for pol in ["HH", "HV"]:
        dn = ds.digital_number.sel(pol=pol).values
        for i, j in cells:
            pixels = pixels_of_cell(product, pol, i, j)
            np.testing.assert_allclose(
                dn[i, j],
                rms_of_valid_pixels(pixels),
                rtol=1e-5,
                err_msg=f"{pol} cell {(i, j)}",
            )
            np.testing.assert_allclose(
                fraction[i, j],
                np.mean(pixels > 0),
                atol=1e-6,
                err_msg=f"{pol} cell {(i, j)}",
            )


def test_valid_pixel_fraction_is_in_the_xsar_dataset(rcm):
    ds = rcm.dataset

    assert ds.valid_pixel_fraction.dims == ("pol", "line", "sample")
    assert ds.valid_pixel_fraction.shape == ds.digital_number.shape

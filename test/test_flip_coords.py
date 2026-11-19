"""RS2 and RCM images flipped by xsar: each cell must be at its coordinate.

xsar flips the lines (ascending pass) or the samples (left-looking ascending or
right-looking descending pass) so that the line increases with the azimuth time and
the sample with the range. At reduced resolution, the pixels left over (less than
one cell) are at the end of the file, so at the start of a flipped axis.

For cells at both ends of the flipped axis, the cell value is compared with the rms of
the non-zero pixels found at its coordinate in the full resolution image."""

import glob
import logging

import numpy as np
import pytest
import rasterio
from rasterio.windows import Window

import xsar

logging.getLogger("xsar").setLevel(logging.WARNING)

PRODUCTS = {
    # right-looking, descending: samples flipped
    "rcm_samples_flipped": (
        xsar.RcmDataset,
        "RCM1_OK1050603_PK1050605_1_SC50MB_20200214_115905_HH_HV_Z010",
        "imagery/*_{pol}.tif",
    ),
    # right-looking, ascending: lines flipped
    "rcm_lines_flipped": (
        xsar.RcmDataset,
        "RCM1_OK2460179_PK2462841_1_SCLND_20230301_072431_VV_VH_GRD",
        "imagery/*_{pol}.tif",
    ),
    # right-looking, descending: samples flipped
    "rs2_samples_flipped": (
        xsar.RadarSat2Dataset,
        "RS2_OK135107_PK1187782_DK1151894_SCWA_20220407_182127_VV_VH_SGF",
        "imagery_{pol}.tif",
    ),
}


def fixed_rs2_reader():
    from xradarsat2 import radarSat2_xarray_reader

    return hasattr(radarSat2_xarray_reader, "read_cells")


def file_start(coord, r, flipped, last):
    """First file pixel of the cell centered on `coord` (xsar convention)."""
    start = int(round(coord - (r - 1) / 2))
    return last - (start + int(r) - 1) if flipped else start


def rms_of_valid_pixels(pixels):
    valid = pixels[pixels > 0].astype(np.float64)
    return np.sqrt(np.mean(valid**2)) if valid.size else 0


@pytest.mark.parametrize("case", list(PRODUCTS))
def test_cells_are_at_their_coordinates(case):
    cls, name, pattern = PRODUCTS[case]
    if cls is xsar.RadarSat2Dataset and not fixed_rs2_reader():
        pytest.skip("needs xradarsat2 with the fixed resampled read")
    path = xsar.get_test_file(name)
    sar = cls(path, resolution="1000m")
    ds = sar.dataset
    pol = str(ds.pol.values[0])
    dn = ds.digital_number.sel(pol=pol).values
    tiff = glob.glob(f"{path}/{pattern.format(pol=pol)}")[0]
    with rasterio.open(tiff) as src:
        height, width = src.shape
    flipped = {"line": sar.sar_meta.lines_flipped, "sample": sar.sar_meta.samples_flipped}
    last = {"line": height - 1, "sample": width - 1}
    r = {d: float(ds[d].values[1] - ds[d].values[0]) for d in ("line", "sample")}
    assert flipped["line"] or flipped["sample"]

    axis = "sample" if flipped["sample"] else "line"
    other = "line" if axis == "sample" else "sample"
    n = ds.sizes[axis]
    k = ds.sizes[other] // 2
    cells = list(range(5, 15)) + list(range(n - 15, n - 5))
    with rasterio.open(tiff) as src:
        for q in cells:
            index = {axis: q, other: k}
            start = {
                d: file_start(ds[d].values[index[d]], r[d], flipped[d], last[d])
                for d in ("line", "sample")
            }
            pixels = src.read(
                1,
                window=Window(start["sample"], start["line"], int(r["sample"]), int(r["line"])),
            )
            np.testing.assert_allclose(
                dn[index["line"], index["sample"]],
                rms_of_valid_pixels(pixels),
                rtol=1e-5,
                err_msg=f"{case}: cell {index}",
            )


@pytest.mark.parametrize("case", ["rcm_samples_flipped", "rcm_lines_flipped"])
def test_full_resolution_coordinates_unchanged(case):
    """At full resolution, the flipped coordinates are still 0, 1, 2, ..."""
    cls, name, _ = PRODUCTS[case]
    ds = cls(xsar.get_test_file(name)).dataset

    np.testing.assert_array_equal(ds.line, np.arange(ds.sizes["line"]))
    np.testing.assert_array_equal(ds.sample, np.arange(ds.sizes["sample"]))

"""
Sentinel-1 SAFE read from S3, on a local fake S3 server (moto): no network, no credentials.

The test SAFE is uploaded to the fake bucket, opened with an ``s3://`` URL and
``storage_options``, and compared with the same SAFE opened locally.
"""

import os
import pickle
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

import xsar

# remote measurement files need xarray-safe-s1 with safe_s1.remote
pytest.importorskip("safe_s1.remote")
pytest.importorskip("moto.server")
pytest.importorskip("s3fs")
boto3 = pytest.importorskip("boto3")

PRODUCT = "S1B_IW_GRDH_1SDV_20181013T062322_20181013T062347_013130_018428_Z010.SAFE"
BUCKET = "eodata"
KEYS = {"key": "testing", "secret": "testing"}
REGION = "us-east-1"


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def endpoint():
    """
    URL of a fake S3 server (moto), in its own process.

    Not a thread of the test process: `rasterio.open` keeps the GIL while GDAL
    makes its HTTP requests, which would block a server thread forever.
    """
    saved = {k: os.environ.get(k) for k in ("NO_PROXY", "no_proxy")}
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "moto.server", "-H", "127.0.0.1", "-p", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("the moto server did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.terminate()
        server.wait(timeout=30)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def local_safe():
    return xsar.get_test_file(PRODUCT)


@pytest.fixture(scope="module")
def s3_url(endpoint, local_safe):
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=KEYS["key"],
        aws_secret_access_key=KEYS["secret"],
        region_name=REGION,
    )
    client.create_bucket(Bucket=BUCKET)
    safe = os.path.basename(local_safe.rstrip("/"))
    for root, _, files in os.walk(local_safe):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), local_safe)
            client.upload_file(os.path.join(root, f), BUCKET, f"S1/{safe}/{rel}")
    return f"s3://{BUCKET}/S1/{safe}"


@pytest.fixture(scope="module")
def storage_options(endpoint):
    return dict(KEYS, client_kwargs={"endpoint_url": endpoint, "region_name": REGION})


def test_open_dataset_from_s3(local_safe, s3_url, storage_options):
    local = xsar.open_dataset(local_safe, resolution="1000m")
    remote = xsar.open_dataset(s3_url, resolution="1000m", storage_options=storage_options)
    assert set(remote.data_vars) == set(local.data_vars)
    for var in ("digital_number", "sigma0", "valid_pixel_fraction"):
        np.testing.assert_array_equal(remote[var].values, local[var].values)
    np.testing.assert_array_equal(remote.line.values, local.line.values)
    np.testing.assert_array_equal(remote.sample.values, local.sample.values)


def test_meta_keeps_storage_options(s3_url, storage_options):
    meta = xsar.Sentinel1Meta(s3_url, storage_options=storage_options)
    assert meta.path == s3_url
    # pickle (dask) and from_dict (actors) re-open the remote SAFE with its options
    for copy in (
        pickle.loads(pickle.dumps(meta)),
        xsar.Sentinel1Meta.from_dict(meta.dict),
    ):
        assert copy.storage_options == storage_options
        assert copy.name == meta.name


def test_storage_options_with_meta_object(s3_url, storage_options):
    meta = xsar.Sentinel1Meta(s3_url, storage_options=storage_options)
    with pytest.raises(ValueError, match="storage_options"):
        xsar.Sentinel1Dataset(meta, storage_options=storage_options)

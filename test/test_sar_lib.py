import unittest
import src.sarlib as sr


class BasicTest(unittest.TestCase):
    def test_nrcs(self):
        sar = sr.SentinelReader('toto')
        slicing = slice(60, 200, 10)
        nrcs = sar.sarlib.nrcs.loc(slicing)
        nrcs_result = []
        self.assertEqual(nrcs, nrcs_result)

    def test_nesz(self):
        sar = sr.SentinelReader('toto')
        slicing = slice(170, 200, 2)
        nesz = sar.sarlib.nesz.loc(slicing)
        nesz_result = []
        self.assertEqual(nesz, nesz_result)

    def test_lat(self):
        sar = sr.SentinelReader('toto')
        slicing = slice(60, 200, 10)
        lat = sar.sarlib.lat.loc(slicing)
        lat_result = []
        self.assertEqual(lat, lat_result)

    def test_lon(self):
        sar = sr.SentinelReader('toto')
        slicing = slice(60, 200, 10)
        lon = sar.sarlib.lon.loc(slicing)
        lon_result = []
        self.assertEqual(lon, lon_result)

    def test_sigma0(self):
        sar = sr.SentinelReader('toto')
        slicing = slice(60, 200, 10)
        sigma0 = sar.sarlib.sigma0.loc(slicing)
        sigma0_result = []
        self.assertEqual(sigma0, sigma0_result)

    def test_digital_number(self):
        sar = sr.SentinelReader('toto')
        slicing = slice(60, 200, 10)
        digital_number = sar.sarlib.digital_number.loc(slicing)
        digital_number_result = []
        self.assertEqual(digital_number, digital_number_result)


if __name__ == '__main__':
    unittest.main()

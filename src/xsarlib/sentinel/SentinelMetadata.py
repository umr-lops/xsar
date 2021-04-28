# author : alevieux
# -*- coding: utf-8 -*-
# date : $Date$
# usage : 

import datetime
import logging
import os


class SentinelMetadata:
    def __init__(self, annotation_file=None, calibration_file=None, noise_calibration_file=None, raster_objects=None,
                 sigma0_lut=None, gamma0_lut=None, noise_lut=None):
        self.annotation_file = annotation_file
        self.calibration_file = calibration_file
        self.noise_calibration_file = noise_calibration_file
        self.raster_objects = raster_objects
        self.sigma0_lut = sigma0_lut
        self.gamma0_lut = gamma0_lut
        self.noise_lut = noise_lut

    def getFiles(self):
        return zip(self.raster_objects, self.annotation_file, self.calibration_file, self.noise_calibration_file)

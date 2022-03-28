from lxml import objectify
import jmespath
import logging
from collections.abc import Iterable
import os
import re
import yaml
from .utils import get_glob

logger = logging.getLogger('xsar.xml_parser')
logger.addHandler(logging.NullHandler())


# TODO: no variable caching is not while  https://github.com/dask/distributed/issues/5610 is not solved
class XmlParser:
    """
    Parameters
    ----------
    xpath_mappings: dict
        first level key is xml file type
        second level key is variable name to be created
        mappings may be 'xpath', or 'tuple(func,xpath)', or 'dict'
            - xpath is an lxml xpath
            - func is a decoder function fed by xpath
            - dict is a nested dict with same structure, to create more hierarchy levels.
    compounds_vars: dict
        compounds variables are variables composed of several variables taken from xpath_mappings
        the key is the variable name, and the value is a tuple or a dict.
            if dict: (key, jpath), where key is the
                sub variable name to create, and jpath is a jmespath in xpath_mappings.
            if tuple: ( func, iterable ), where func(iterable) will be called to convert the iterable to another object.
                iterable values are jpath. if iterable is a tuple, func(*iterable) will be called.
    namespaces: dict
        xml namespaces, passed to lxml.xpath.
        namespaces are mutualised between all handled xml files.
    """

    def __init__(self, xpath_mappings={}, compounds_vars={}, namespaces={}):
        self._namespaces = namespaces
        self._xpath_mappings = xpath_mappings
        self._compounds_vars = compounds_vars

    def __del__(self):
        logger.debug('__del__ XmlParser')

    def getroot(self, xml_file):
        """return xml root object from xml_file. (also update self._namespaces with fetched ones)"""
        xml_root = objectify.parse(xml_file).getroot()
        self._namespaces.update(xml_root.nsmap)
        return xml_root

    def xpath(self, xml_file, path):
        """
        get path from xml_file. this is a simple wrapper for `objectify.parse(xml_file).getroot().xpath(path)`
        """

        xml_root = self.getroot(xml_file)
        result = [getattr(e, 'pyval', e) for e in xml_root.xpath(path, namespaces=self._namespaces)]
        return result

    def get_var(self, xml_file, jpath, describe=False):
        """
        get simple variable in xml_file.

        Parameters
        ----------
        xml_file: str
            xml filename
        jpath: str
            jmespath string reaching xpath in xpath_mappings
        describe: bool
            If True, describe the variable (ie return xpath used)

        Returns
        -------
        object
            xpath list, or decoded object, if a conversion function was specified in xpath_mappings
        """

        func = None
        xpath = jmespath.search(jpath, self._xpath_mappings)
        if xpath is None:
            raise KeyError('jmespath "%s" not found in xpath_mappings' % jpath)

        if isinstance(xpath, tuple) and callable(xpath[0]):
            func, xpath = xpath

        if describe:
            return xpath

        if not isinstance(xpath, str):
            raise NotImplementedError('Non leaf xpath of type "%s" instead of str' % type(xpath).__name__)

        result = self.xpath(xml_file, xpath)
        if func is not None:
            result = func(result)

        return result

    def get_compound_var(self, xml_file, var_name, describe=False):
        """

        Parameters
        ----------
        var_name: str

            key in self._compounds_vars

        xml_file: str

            xml_file to use.

        describe: bool

            If True, only returns a string describing the variable (file, xpath, etc...)



        Returns
        -------
        object

        """

        if describe:
            # keep only informative parts in filename
            # sub SAFE path
            minifile = re.sub('.*SAFE/', '', xml_file)
            minifile = re.sub(r'-.*\.xml', '.xml', minifile)

        var_object = self._compounds_vars[var_name]

        func = None
        if isinstance(var_object, dict) and 'func' in var_object and callable(var_object['func']):
            func = var_object['func']
            if isinstance(var_object['args'], tuple):
                args = var_object['args']
            else:
                raise ValueError('args must be a tuple when func is called')
        else:
            args = var_object

        result = None
        if isinstance(args, dict):
            result = {}
            for key, path in args.items():
                result[key] = self.get_var(xml_file, path, describe=describe)
        elif isinstance(args, Iterable):
            result = [self.get_var(xml_file, p, describe=describe) for p in args]

        if isinstance(args, tuple):
            result = tuple(result)

        if func is not None and not describe:
            # apply converter
            result = func(*result)

        if describe:
            if isinstance(result, dict):
                result = result.values()
            description = yaml.safe_dump({var_name: {minifile: result}})
            return description
        else:
            return result

    def __del__(self):
        logger.debug('__del__ XmlParser')

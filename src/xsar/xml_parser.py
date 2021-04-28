import os
from lxml import objectify
import jmespath
import logging
from collections.abc import Iterable

logger = logging.getLogger('xsar.xml_parser')
logger.addHandler(logging.NullHandler())


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
        self._xml_roots = {}
        self._xpath_cache = {}
        self._var_cache = {}
        self._compounds_vars_cache = {}
        self._namespaces = namespaces
        self._xpath_mappings = xpath_mappings
        self._compounds_vars = compounds_vars

    def xpath(self, xml_file, path):
        """
        get path from xml_file, with object caching.

        Parameters
        ----------
        xml_file: str
            xml filename
        path: str
            xpath expression

        Returns
        -------
        list
            same list as lxml.xpath

        """
        if xml_file not in self._xml_roots:
            xml_root = objectify.parse(xml_file).getroot()
            self._namespaces.update(xml_root.nsmap)
            self._xml_roots[xml_file] = xml_root

        if xml_file not in self._xpath_cache:
            self._xpath_cache[xml_file] = {}

        if path not in self._xpath_cache[xml_file]:
            logger.debug("xpath no cache hit for '%s' on file %s" % (path, os.path.basename(xml_file)))
            xml_root = self._xml_roots[xml_file]
            result = xml_root.xpath(path, namespaces=self._namespaces)
            self._xpath_cache[xml_file][path] = [getattr(e, 'pyval', e) for e in result]
        else:
            logger.debug("xpath cache hit for '%s' on file %s" % (path, os.path.basename(xml_file)))

        return self._xpath_cache[xml_file][path]

    def get_var(self, xml_file, jpath):
        """
        get simple variable in xml_file.

        Parameters
        ----------
        xml_file: str
            xml filename
        jpath: str
            jmespath string reaching xpath in xpath_mappings

        Returns
        -------
        object
            xpath list, or decoded object, if a conversion function was specified in xpath_mappings
        """

        if (xml_file, jpath) in self._var_cache:
            logger.debug("get_var cache hit for jpath '%s' on file %s" % (jpath, os.path.basename(xml_file)))
            return self._var_cache[(xml_file, jpath)]

        logger.debug("get_var no cache hit for jpath '%s' on file %s" % (jpath, os.path.basename(xml_file)))
        func = None
        xpath = jmespath.search(jpath, self._xpath_mappings)
        if xpath is None:
            raise KeyError('jmespath "%s" not found in xpath_mappings' % jpath)

        if isinstance(xpath, tuple) and callable(xpath[0]):
            func, xpath = xpath

        if not isinstance(xpath, str):
            raise NotImplementedError('Non leaf xpath of type "%s" instead of str' % type(xpath).__name__)

        result = self.xpath(xml_file, xpath)
        if func is not None:
            result = func(result)

        self._var_cache[(xml_file, jpath)] = result

        return result

    def get_compound_var(self, xml_file, var_name):
        """

        Parameters
        ----------
        var_name: str
            key in self._compounds_vars
        xml_file: str
            xml_file to use.

        Returns
        -------
        object

        """

        if (xml_file, var_name) in self._compounds_vars_cache:
            logger.debug("get_compound_var cache hit for '%s' on file %s" % (var_name, os.path.basename(xml_file)))
            return self._compounds_vars_cache[(xml_file, var_name)]

        var_object = self._compounds_vars[var_name]
        logger.debug("get_compound_var no cache hit for '%s' on file %s" % (var_name, os.path.basename(xml_file)))

        func = None
        if isinstance(var_object,dict) and 'func' in var_object and callable(var_object['func']):
            func = var_object['func']
            if isinstance(var_object['args'],tuple):
                args = var_object['args']
            else:
                raise ValueError('args must be a tuple when func is called')
        else:
            args = var_object

        result = None
        if isinstance(args, dict):
            result = {}
            for key, path in args.items():
                result[key] = self.get_var(xml_file, path)
        elif isinstance(args, Iterable):
            result = [self.get_var(xml_file, p) for p in args]

        if isinstance(args, tuple):
            result = tuple(result)

        if func is not None:
            # apply converter
            result = func(*result)

        # store result in cache for subsequent call
        self._compounds_vars_cache[(xml_file, var_name)] = result
        return result

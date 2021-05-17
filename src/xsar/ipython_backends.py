def repr_mimebundle_SentinelMeta(self, include=None, exclude=None):
    """html output for notebook"""
    try:
        import cartopy
        import geoviews as gv
        import geoviews.feature as gf
        import jinja2
        import geopandas as gpd
    except ModuleNotFoundError as e:
        return {'text/html': str(self)}
    gv.extension('bokeh', logo=False)

    template = jinja2.Template(
        """
        <div align="left">
            <h5>{{ intro }}</h5>
            <table style="width:100%">
                <thead>
                    <tr>
                        <th colspan="2">{{ short_name }}</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>
                            <table>
                                {% for key, value in properties.items() %}
                                 <tr>
                                     <th> {{ key }} </th>
                                     <td> {{ value }} </td>
                                 </tr>
                                {% endfor %}
                            </table>
                        </td>
                        <td>{{ location }}</td>
                    </tr>
                </tbody>
            </table>

        </div>

        """
    )

    crs = cartopy.crs.PlateCarree()

    world = gv.operation.resample_geometry(gf.land.geoms('10m')).opts(color='khaki', projection=crs, alpha=0.5)

    center = self.footprint.centroid
    xlim = (center.x - 20, center.x + 20)
    ylim = (center.y - 20, center.y + 20)

    if self.multidataset and \
            len(self.subdatasets) == len(
        self._footprints):  # checks len because SAFEs like IW_SLC has only one footprint for 3 subdatasets
        dsid = [s.split(':')[2] for s in self.subdatasets]
        footprint = self._footprints
    else:
        dsid = [self.dsid]
        footprint = [self.footprint]

    footprints_df = gpd.GeoDataFrame(
        {
            'dsid': dsid,
            'geometry': footprint
        }
    )

    footprint = gv.Polygons(footprints_df).opts(projection=crs, xlim=xlim, ylim=ylim, alpha=0.5, tools=['hover'])

    location = (world * footprint).opts(width=400, height=400, title='Map')

    data, metadata = location._repr_mimebundle_(include=include, exclude=exclude)

    properties = self.to_dict()
    properties['orbit_pass'] = self.orbit_pass
    if self.pixel_atrack_m is not None:
        properties['pixel size'] = "%.1f * %.1f meters (atrack * xtrack)" % (
            self.pixel_atrack_m, self.pixel_xtrack_m)
    properties['coverage'] = self.coverage
    properties['start_date'] = self.start_date
    properties['stop_date'] = self.stop_date
    if len(self.subdatasets) > 0:
        properties['subdatasets'] = "list of %d subdatasets" % len(self.subdatasets)
    properties = {k: v for k, v in properties.items() if v is not None}

    if self.multidataset:
        intro = "Multi (%d) dataset" % len(self.subdatasets)
    else:
        intro = "Single dataset"
        properties['dsid'] = self.dsid

    if 'text/html' in data:
        data['text/html'] = template.render(
            intro=intro,
            short_name=self.short_name,
            properties=properties,
            location=data['text/html']
        )

    return data, metadata


def repr_mimebundle_SentinelDataset(self, include=None, exclude=None):
    try:
        import jinja2
        import holoviews as hv
        from shapely.geometry import Polygon
    except ModuleNotFoundError as e:
        return {'text/html': str(self)}
    hv.extension('bokeh', logo=False)

    template = jinja2.Template(
        """
        <div align="left">
            <h5>{{ intro }}</h5>
            <table style="width:100%">
                <thead>
                    <tr>
                        <th colspan="2">{{ short_name }}</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>
                            <table>
                                {% for key, value in properties.items() %}
                                 <tr>
                                     <th> {{ key }} </th>
                                     <td> {{ value }} </td>
                                 </tr>
                                {% endfor %}
                            </table>
                        </td>
                        <td>{{ location }}</td>
                    </tr>
                </tbody>
            </table>

        </div>

        """
    )

    grid = (hv.Path(Polygon(self._bbox_coords_ori)).opts(color='blue') * hv.Polygons(
        Polygon(self._bbox_coords)).opts(color='blue', fill_color='cyan')).opts(xlabel='atrack', ylabel='xtrack')

    data, metadata = grid._repr_mimebundle_(include=include, exclude=exclude)

    properties = {}
    if self.pixel_atrack_m is not None:
        properties['pixel size'] = "%.1f * %.1f meters (atrack * xtrack)" % (
            self.pixel_atrack_m, self.pixel_xtrack_m)
    properties['coverage'] = self.coverage
    properties = {k: v for k, v in properties.items() if v is not None}

    if self.sliced:
        intro = "dataset slice"
    else:
        intro = "full dataset coverage"

    if 'text/html' in data:
        data['text/html'] = template.render(
            intro=intro,
            short_name=self.s1meta.short_name,
            properties=properties,
            location=data['text/html']
        )

    return data, metadata


repr_mimebundle_wrapper = {
    'Sentinel1Meta': repr_mimebundle_SentinelMeta,
    'Sentinel1Dataset': repr_mimebundle_SentinelDataset
}


def repr_mimebundle(obj, include=None, exclude=None):
    return repr_mimebundle_wrapper[type(obj).__name__](obj, include=include, exclude=exclude)
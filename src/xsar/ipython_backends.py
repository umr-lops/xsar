try:
    # will fall back to repr if some modules are missing

    # make sure we are running from a notebook
    # if test fail, nothing will be imported, and that will save lot of importtime
    assert get_ipython() is not None
    import cartopy
    import holoviews as hv
    import geoviews as gv
    import geoviews.feature as gf
    import jinja2
    import geopandas as gpd
    import holoviews.ipython.display_hooks as display_hooks
    from shapely.geometry import Polygon
except (ModuleNotFoundError, AssertionError, NameError):
    pass


def repr_mimebundle_Sentinel1Meta(self, include=None, exclude=None):
    """html output for notebook"""

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

    world = gv.operation.resample_geometry(gf.land.geoms("10m")).opts(
        color="khaki", projection=crs, alpha=0.5
    )

    center = self.footprint.centroid
    xlim = (center.x - 20, center.x + 20)
    ylim = (center.y - 20, center.y + 20)

    if self.multidataset and len(self.subdatasets) == len(
        self._footprints
    ):  # checks len because SAFEs like IW_SLC has only one footprint for 3 subdatasets
        dsid = [s.split(":")[2] for s in self.subdatasets]
        footprint = self._footprints
    else:
        dsid = [self.dsid]
        footprint = [self.footprint]

    footprints_df = gpd.GeoDataFrame(
        {"dsid": dsid, "geometry": footprint}, crs="EPSG:4326"
    )

    opts = {"bokeh": dict(tools=["hover"])}

    footprint = (
        gv.Polygons(footprints_df, label="footprint")
        .opts(projection=crs, xlim=xlim, ylim=ylim, alpha=0.5)
        .opts(
            **(opts.get(hv.Store.current_backend) or {}),
            backend=hv.Store.current_backend
        )
    )

    orbit = (
        gv.Points(self.orbit["geometry"].to_crs("EPSG:4326"), label="orbit")
        .opts(projection=crs, xlim=xlim, ylim=ylim, alpha=0.5)
        .opts(
            **(opts.get(hv.Store.current_backend) or {}),
            backend=hv.Store.current_backend
        )
    )

    opts = {"bokeh": dict(width=400, height=400), "matplotlib": dict(fig_inches=5)}
    location = (
        (world * footprint * orbit)
        .opts(title="Map")
        .opts(
            **(opts.get(hv.Store.current_backend) or {}),
            backend=hv.Store.current_backend
        )
    )
    data, metadata = display_hooks.render(location)

    properties = self.to_dict()
    properties["orbit_pass"] = self.orbit_pass
    if self.pixel_line_m is not None:
        properties["pixel size"] = "%.1f * %.1f meters (line * sample)" % (
            self.pixel_line_m,
            self.pixel_sample_m,
        )
    properties["coverage"] = self.coverage
    properties["start_date"] = self.start_date
    properties["stop_date"] = self.stop_date
    if len(self.subdatasets) > 0:
        properties["subdatasets"] = "list of %d subdatasets" % len(self.subdatasets)
    properties = {k: v for k, v in properties.items() if v is not None}

    if self.multidataset:
        intro = "Multi (%d) dataset" % len(self.subdatasets)
    else:
        intro = "Single dataset"
        properties["dsid"] = self.dsid

    if "text/html" in data:
        data["text/html"] = template.render(
            intro=intro,
            short_name=self.short_name,
            properties=properties,
            location=data["text/html"],
        )

    return data, metadata


def repr_mimebundle_Sentinel1Dataset(self, include=None, exclude=None):

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

    opts = {"bokeh": dict(fill_color="cyan")}
    grid = (
        hv.Path(Polygon(self._bbox_coords_ori)).opts(color="blue")
        * hv.Polygons(Polygon(self._bbox_coords))
        .opts(color="blue")
        .opts(
            **(opts.get(hv.Store.current_backend) or {}),
            backend=hv.Store.current_backend
        )
    ).opts(xlabel="line", ylabel="sample")

    data, metadata = display_hooks.render(grid)
    properties = {}
    if self.pixel_line_m is not None:
        properties["pixel size"] = "%.1f * %.1f meters (line * sample)" % (
            self.pixel_line_m,
            self.pixel_sample_m,
        )
    properties["coverage"] = self.coverage
    properties = {k: v for k, v in properties.items() if v is not None}

    if self.sliced:
        intro = "dataset slice"
    else:
        intro = "full dataset coverage"

    if "text/html" in data:
        data["text/html"] = template.render(
            intro=intro,
            short_name=self.sar_meta.short_name,
            properties=properties,
            location=data["text/html"],
        )

    return data, metadata


repr_mimebundle_wrapper = {
    "Sentinel1Meta": repr_mimebundle_Sentinel1Meta,
    "Sentinel1Dataset": repr_mimebundle_Sentinel1Dataset,
}


def repr_mimebundle(obj, include=None, exclude=None):
    try:
        return repr_mimebundle_wrapper[type(obj).__name__](
            obj, include=include, exclude=exclude
        )
    except Exception:
        return {"text": repr(obj)}

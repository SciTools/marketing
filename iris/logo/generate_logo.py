"""
Module to generate the Iris logo in every required format.
Uses `xml.ElementTree` for SVG file editing.
"""

from argparse import ArgumentParser
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom
from zipfile import ZipFile

from cartopy import crs as ccrs
from cartopy.feature import LAND
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.use("agg")

# Main dimensions and proportions. The logo's SVG elements can be configured
# within their respective private functions.

# How much bigger than the globe the iris clip should be.
CLIP_GLOBE_RATIO = 1.28
# Pixel size of the largest dimension of the main logo.
# (Proportions dictated by those of the clip).
LOGO_SIZE = 1024
# Pixel height of the banner (the height of the globe within the banner).
BANNER_HEIGHT = 256
# How much smaller than the globe the banner text should be.
TEXT_GLOBE_RATIO = 0.6


# XML ElementTree setup.
NAMESPACES = {"svg": "http://www.w3.org/2000/svg"}
ET.register_namespace("", NAMESPACES["svg"])


class _SvgNamedElement(ET.Element):
    def __init__(self, id, tag, is_def=False, attrib={}, **extra):
        """
        `ET.Element` with extra properties to help construct SVG.
        id = mandatory `id` string in `attrib` dict, referencable as a class attribute.
        is_def = attrib denoting whether to store in the SVG `defs` section.
        """
        super().__init__(tag, attrib, **extra)
        self.id = id
        self.attrib["id"] = self.id
        self.is_def = is_def


def _svg_clip():
    """Generate the clip for masking the entire logo contents."""
    # SVG string representing bezier curves, drawn in a GUI then size-converted
    # for this file.
    path_string = (
        "M 66.582031,73.613283 C 62.196206,73.820182 51.16069,"
        "82.105643 33.433594,80.496096 18.711759,79.15939 "
        "10.669958,73.392913 8.4375,74.619143 7.1228015,75.508541 "
        "6.7582985,76.436912 6.703125,78.451174 6.64868,80.465549 "
        "5.5985568,94.091611 12.535156,107.18359 c 8.745259,"
        "16.50554 21.06813,20.14551 26.152344,20.24414 12.671346,"
        "0.24601 24.681745,-21.45054 27.345703,-30.62304 3.143758,"
        "-10.82453 4.457007,-22.654297 1.335938,-23.119141 "
        "-0.233298,-0.06335 -0.494721,-0.08606 -0.78711,-0.07227 z "
        "m 7.119141,-5.857422 c -0.859687,0.02602 -1.458448,"
        "0.280361 -1.722656,0.806641 -2.123362,3.215735 3.515029,"
        "16.843803 -3.970704,34.189448 -5.828231,13.50478 "
        "-13.830895,19.32496 -13.347656,21.81446 0.444722,1.5177 "
        "1.220596,2.14788 3.13086,2.82226 1.910471,0.67381 "
        "14.623496,5.87722 29.292968,3.36524 18.494166,-3.16726 "
        "25.783696,-13.69084 27.449216,-18.4668 C 118.68435,"
        "100.38386 101.63623,82.323612 93.683594,76.970705 "
        "86.058314,71.837998 77.426483,67.643118 73.701172,"
        "67.755861 Z M 114.02539,33.224611 C 103.06524,33.255401 "
        "88.961605,40.28151 83.277344,44.67969 74.333356,51.599967 "
        "66.27534,60.401955 68.525391,62.601564 c 2.420462,"
        "3.001097 17.201948,1.879418 31.484379,14.316407 11.11971,"
        "9.683149 14.21486,19.049139 16.74609,19.361328 1.58953,"
        "0.0486 2.43239,-0.488445 3.66797,-2.085938 1.23506,"
        "-1.597905 10.14187,-12.009723 12.27148,-26.654297 "
        "2.68478,-18.462878 -5.13068,-28.607634 -9.18554,"
        "-31.658203 -2.52647,-1.900653 -5.831,-2.666512 -9.48438,"
        "-2.65625 z M 39.621094,14.64258 C 39.094212,14.665 "
        "38.496575,14.789793 37.767578,15.003908 35.823484,"
        "15.574873 22.460486,18.793044 12.078125,29.396486 "
        "-1.0113962,42.764595 -0.68566506,55.540029 0.79101563,"
        "60.376955 4.4713185,72.432363 28.943765,77.081596 "
        "38.542969,76.765627 49.870882,76.392777 61.593892,"
        "73.978953 61.074219,70.884768 60.890477,67.042613 "
        "48.270811,59.312854 44.070312,40.906252 40.799857,"
        "26.575361 43.83381,17.190581 41.970703,15.458986 "
        "41.184932,14.853981 40.49923,14.605209 39.621094,14.64258 "
        "Z M 67.228516,0.08984563 C 60.427428,0.11193533 "
        "55.565192,2.1689455 53.21875,3.7949238 42.82192,10.999553 "
        "45.934544,35.571547 49.203125,44.54883 c 3.857276,"
        "10.594054 9.790034,20.931896 12.589844,19.484375 "
        "3.619302,-1.360857 7.113072,-15.680732 23.425781,"
        "-25.339844 12.700702,-7.520306 22.61812,-7.553206 "
        "23.69922,-9.849609 0.53758,-1.487663 0.28371,-2.45185 "
        "-0.86328,-4.113281 C 106.90759,23.069051 99.699016,"
        "11.431303 86.345703,4.89258 78.980393,1.2860119 72.51825,"
        "0.07266475 67.228516,0.08984563 Z"
    )
    original_size_xy = np.array([133.334, 131.521])
    scaling = LOGO_SIZE / max(original_size_xy)
    clip = _SvgNamedElement(id="iris_clip", tag="clipPath", is_def=True)
    clip.append(
        ET.Element(
            "path", attrib={"d": path_string, "transform": f"scale({scaling})"}
        )
    )
    return clip, original_size_xy


def _svg_background():
    """Generate the background rectangle for the logo."""
    gradient = _SvgNamedElement(
        id="background_gradient",
        tag="linearGradient",
        is_def=True,
        attrib={"y1": "0%", "y2": "100%",},
    )
    gradient.extend(
        [
            ET.Element(
                "stop", attrib={"offset": "0", "stop-color": "#13385d",},
            ),
            ET.Element(
                "stop", attrib={"offset": "0.43", "stop-color": "#0b3849",},
            ),
            ET.Element(
                "stop", attrib={"offset": "1", "stop-color": "#272b2c",},
            ),
        ]
    )
    background = _SvgNamedElement(
        id="background",
        tag="rect",
        attrib={
            "height": "100%",
            "width": "100%",
            "fill": f"url(#{gradient.id})",
        },
    )
    return [background, gradient]


def _svg_sea():
    """Generate the circle representing the globe's sea in the logo."""
    # Not using Cartopy for sea since it doesn't actually render curves/circles.
    gradient = _SvgNamedElement(
        id="sea_gradient", tag="radialGradient", is_def=True
    )
    gradient.extend(
        [
            ET.Element(
                "stop", attrib={"offset": "0", "stop-color": "#20b0ea"},
            ),
            ET.Element(
                "stop", attrib={"offset": "1", "stop-color": "#156475",},
            ),
        ]
    )
    sea = _SvgNamedElement(
        id="sea",
        tag="circle",
        attrib={
            "cx": "50%",
            "cy": "50%",
            "r": f"{50.5 / CLIP_GLOBE_RATIO}%",
            "fill": f"url(#{gradient.id})",
        },
    )
    return [sea, gradient]


def _svg_glow():
    """Generate the coloured glow to go behind the globe in the logo."""
    gradient = _SvgNamedElement(
        id="glow_gradient",
        tag="radialGradient",
        is_def=True,
        attrib={
            "gradientTransform": "scale(1.15, 1.35), translate(-0.1, -0.3)"
        },
    )
    gradient.extend(
        [
            ET.Element(
                "stop",
                attrib={
                    "offset": "0",
                    "stop-color": "#0aaea7",
                    "stop-opacity": "0.85882354",
                },
            ),
            ET.Element(
                "stop",
                attrib={
                    "offset": "0.67322218",
                    "stop-color": "#18685d",
                    "stop-opacity": "0.74117649",
                },
            ),
            ET.Element(
                "stop", attrib={"offset": "1", "stop-color": "#b6df34",},
            ),
        ]
    )
    blur = _SvgNamedElement(id="glow_blur", tag="filter", is_def=True)
    blur.append(ET.Element("feGaussianBlur", attrib={"stdDeviation": "14"}))
    glow = _SvgNamedElement(
        id="glow",
        tag="circle",
        attrib={
            "cx": "50%",
            "cy": "50%",
            "r": f"{52 / CLIP_GLOBE_RATIO}%",
            "fill": f"url(#{gradient.id})",
            "filter": f"url(#{blur.id})",
            "stroke": "#ffffff",
            "stroke-width": "2",
            "stroke-opacity": "0.797414",
        },
    )
    return [glow, gradient, blur]


def _svg_land(logo_size_xy, logo_centre_xy, rotate=False):
    """
    Generate the circle representing the globe's land in the logo, clipped by
    appropriate coastline shapes (using Matplotlib and Cartopy).
    """
    # Set plotting size/proportions.
    mpl_points_per_inch = 72
    plot_inches = logo_size_xy / mpl_points_per_inch
    plot_padding = (1 - (1 / CLIP_GLOBE_RATIO)) / 2

    # Create land with simplified coastlines.
    simple_geometries = [
        geometry.simplify(0.8, True) for geometry in LAND.geometries()
    ]
    LAND.geometries = lambda: iter(simple_geometries)

    # Variable that will store the sequence of land-shaped SVG clips for
    # each longitude.
    land_clips = []

    # Create a sequence of longitude values.
    central_longitude = -30
    central_latitude = 22.9
    perspective_tilt = -4.99
    rotation_frames = 180 if rotate else 1
    rotation_longitudes = np.linspace(
        start=central_longitude + 360,
        stop=central_longitude,
        num=rotation_frames,
        endpoint=False,
    )
    # Normalise to -180..+180
    rotation_longitudes = (rotation_longitudes + 360.0 + 180.0) % 360.0 - 180.0

    transform_string = (
        f"rotate({perspective_tilt} {logo_centre_xy[0]} "
        f"{logo_centre_xy[1]})"
    )

    land_clip_id = "land_clip"
    for lon in rotation_longitudes:
        # Use Matplotlib and Cartopy to generate land-shaped SVG clips for each longitude.
        projection_rotated = ccrs.Orthographic(
            central_longitude=lon, central_latitude=central_latitude
        )

        # Use constants set earlier to achieve desired dimensions.
        plt.figure(0, figsize=plot_inches)
        ax = plt.subplot(projection=projection_rotated)
        plt.subplots_adjust(
            left=plot_padding,
            bottom=plot_padding,
            right=1 - plot_padding,
            top=1 - plot_padding,
        )
        ax.add_feature(LAND)

        # Save as SVG and extract the resultant code.
        svg_bytes = BytesIO()
        plt.savefig(svg_bytes, format="svg")
        svg_mpl = ET.fromstring(svg_bytes.getvalue())

        # Find land paths and convert to clip paths.
        land_clip = _SvgNamedElement(
            id=land_clip_id,
            tag="clipPath",
            is_def=True,
            attrib={"transform": transform_string},
        )
        mpl_land = svg_mpl.find(".//svg:g[@id='figure_1']", NAMESPACES)
        land_paths = mpl_land.find(
            ".//svg:g[@id='PathCollection_1']", NAMESPACES
        )
        land_clip.extend(list(land_paths))
        for path in land_clip:
            # Remove all other attribute items.
            path.attrib = {"d": path.attrib["d"], "stroke-linejoin": "round"}
        land_clips.append(land_clip)

    gradient = _SvgNamedElement(
        id="land_gradient", tag="radialGradient", is_def=True
    )
    gradient.extend(
        [
            ET.Element(
                "stop", attrib={"offset": "0", "stop-color": "#d5e488"}
            ),
            ET.Element(
                "stop", attrib={"offset": "1", "stop-color": "#aec928"}
            ),
        ]
    )
    land = _SvgNamedElement(
        id="land",
        tag="circle",
        attrib={
            "cx": "50%",
            "cy": "50%",
            "r": f"{50 / CLIP_GLOBE_RATIO}%",
            "fill": f"url(#{gradient.id})",
            "clip-path": f"url(#{land_clip_id})",
        },
    )
    return [land, gradient], land_clips


def _svg_logo(iris_clip, land_clip, other_elements, logo_size_xy):
    """Assemble sub-elements into SVG for the logo."""
    # Group contents into a logo subgroup (so text can be stored separately).
    logo_group = ET.Element("svg", attrib={"id": "logo_group"})
    logo_group.attrib["viewBox"] = f"0 0 {logo_size_xy[0]} {logo_size_xy[1]}"

    # The elements that will just be referenced by artwork elements.
    defs_element = ET.Element("defs")
    defs_element.extend([iris_clip, land_clip])
    # The elements that are displayed (not just referenced).
    # All artwork is clipped by the Iris shape.
    artwork_element = ET.Element(
        "g", attrib={"clip-path": f"url(#{iris_clip.id})"}
    )
    for element in other_elements:
        assert isinstance(element, _SvgNamedElement)
        if element.is_def:
            target_parent = defs_element
        else:
            target_parent = artwork_element
        target_parent.append(element)
    for parent_element in (defs_element, artwork_element):
        logo_group.append(parent_element)

    root = ET.Element("svg")
    for ix, dim in enumerate(("width", "height")):
        root.attrib[dim] = str(logo_size_xy[ix])
    root.append(logo_group)

    return root


def _svg_banner(logo_svg, size_xy, text):
    """Make the banner SVG based on an input logo SVG."""
    banner_height = size_xy[1]
    text_size = banner_height * TEXT_GLOBE_RATIO
    text_x = banner_height + 8
    # Manual y centring since SVG dominant-baseline not widely supported.
    text_y = banner_height - (banner_height - text_size) / 2
    text_y *= 0.975  # Slight offset

    text_element = ET.Element(
        "text",
        attrib={
            "x": str(text_x),
            "y": str(text_y),
            "font-size": f"{text_size}pt",
            "font-family": "georgia",
        },
    )
    text_element.text = text

    root = deepcopy(logo_svg)
    for dimension, name in enumerate(("width", "height")):
        root.attrib[name] = str(size_xy[dimension])

    # Left-align the logo.
    banner_logo_group = root.find("svg", NAMESPACES)
    banner_logo_group.attrib["preserveAspectRatio"] = "xMinYMin meet"

    root.append(text_element)

    return root


def _write_svg_file(filename, svg_root, write_dir=None, zip_archive=None):
    """Format the svg then write the svg to a file in write_dir, or
    optionally to an open ZipFile."""
    input_string = ET.tostring(svg_root)
    pretty_xml = minidom.parseString(input_string).toprettyxml()
    # Remove extra empty lines from Matplotlib.
    pretty_xml = "\n".join(
        [line for line in pretty_xml.split("\n") if line.strip()]
    )

    if isinstance(zip_archive, ZipFile):
        # Add to zip file if zip_archive provided.
        zip_archive.writestr(filename, pretty_xml)
        return zip_archive.filename
    elif Path(write_dir).is_dir():
        # Otherwise write to file normally.
        write_path = write_dir.joinpath(filename)
        with open(write_path, "w") as f:
            f.write(pretty_xml)
        return write_path
    else:
        raise ValueError("No valid write_dir or zip_archive provided.")


def generate_logo(
    filename_prefix="iris",
    write_dir=Path.cwd(),
    banner_text="Iris",
    banner_width=588,
    rotate=False,
):
    """Generate the Iris logo and accompanying banner with configurable text.

    Images written in SVG format using `xml.ElementTree`.

    Parameters
    ----------
    filename_prefix : str
        How each created logo file name should start.
    write_dir : str or pathlib.Path
        The directory in which to create the logo files.
    banner_text : str
        The text to include in the banner logo.
    banner_width : int
        Pixel width of the banner logo - must be manually adjusted to fit
        `banner_text`.
    rotate : bool
        Whether to also generate ZIP file of rotated-longitude logos.
        NOTE: takes approx 1min to generate this.

    Returns
    -------
    set of pathlib.Path
        Paths of the created logo files.

    """

    print("LOGO GENERATION START ...")

    write_dir = Path(write_dir)
    # Pixel dimensions of text banner.
    banner_size_xy = [banner_width, BANNER_HEIGHT]

    ###########################################################################
    # Assemble SVG elements for logo.

    # Get SVG and info for the logo's clip.
    iris_clip, clip_size_xy_original = _svg_clip()

    # Use clip proportions to dictate logo proportions and dimensions.
    # clip_size_xy_original = clip["original_size_xy"]
    logo_proportions_xy = clip_size_xy_original / max(clip_size_xy_original)
    logo_size_xy = logo_proportions_xy * LOGO_SIZE
    logo_centre_xy = logo_size_xy / 2

    # Get SVG objects for land, including multiple clips if rotate=True.
    svg_land, land_clips = _svg_land(logo_size_xy, logo_centre_xy, rotate)

    # Make a list of the SVG elements that don't need explicit naming in
    # _svg_logo(). Ordering is important.
    svg_elements = []
    svg_elements.extend(_svg_background())
    svg_elements.extend(_svg_glow())
    svg_elements.extend(_svg_sea())
    svg_elements.extend(svg_land)

    # Create SVG's for each rotation.
    logo_list = []
    banner_list = []
    for clip in land_clips:
        logo_list.append(
            _svg_logo(iris_clip, clip, svg_elements, logo_size_xy)
        )
        banner_list.append(
            _svg_banner(logo_list[-1], banner_size_xy, banner_text)
        )

    ###########################################################################
    # Write files.

    written_path_set = set()

    write_dict = {
        "logo": logo_list,
        "logo-title": banner_list,
    }
    for suffix, svg_list in write_dict.items():
        # Write the main files.
        filename = f"{filename_prefix}-{suffix}.svg"
        written_path_set.add(
            _write_svg_file(filename, svg_list[-1], write_dir=write_dir)
        )

        # Zip archive containing components for manual creation of rotating logo.
        zip_path = write_dir.joinpath(f"{filename_prefix}-{suffix}_rotate.zip")
        if len(svg_list) > 1:
            with ZipFile(zip_path, "w") as rotate_zip:
                for ix, svg_rotated in enumerate(svg_list):
                    written_path_set.add(
                        _write_svg_file(
                            f"{suffix}_rotate{ix:03d}",
                            svg_rotated,
                            zip_archive=rotate_zip,
                        )
                    )

                readme_str = (
                    "Several tools are available to stitch these "
                    "images into a rotating GIF.\n\nE.g. "
                    "http://blog.gregzaal.com/2015/08/06/making-an"
                    "-optimized-gif-in-gimp/"
                )
                rotate_zip.writestr("_README.txt", readme_str)

    print("LOGO GENERATION COMPLETE")

    return written_path_set


def main():
    description = "Command line access for the generate_logo() " \
                  "function - docstring above. Input the parameters " \
                  "as optional arguments (--my_arg value). The return value(" \
                  "s) will be printed."
    docstring = str(generate_logo.__doc__)
    parser = ArgumentParser(description=description,
                            usage=docstring)
    _, args = parser.parse_known_args()

    keys = args[::2]
    values = args[1::2]
    assert all([key[:2] == "--" for key in keys])
    keys = [key[2:] for key in keys]
    kwargs = dict(zip(keys, values))

    print(generate_logo(**kwargs))


if __name__ == "__main__":
    main()

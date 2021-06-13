"""
Module to generate the Iris logo in every required format.
Uses `xml.ElementTree` for SVG file editing.
"""

from argparse import ArgumentParser
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union
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
    def __init__(
        self,
        xml_id: str,
        tag: str,
        is_def: bool = False,
        attrib: Dict[str, str] = None,
        **extra: str,
    ):
        """
        `ET.Element` with extra properties to help construct SVG.

        xml_id = mandatory `id` string in `attrib` dict, referencable as a class attribute.
        is_def = attrib denoting whether to store in the SVG `defs` section.

        """
        if attrib is None:
            attrib = dict()
        super().__init__(tag, attrib, **extra)
        self.xml_id = xml_id
        self.attrib["id"] = self.xml_id
        self.is_def = is_def


def _matrix_transform_string(
    scaling_xy: np.ndarray,
    centre_xy: np.ndarray,
    translation_xy: np.ndarray,
) -> str:
    """
    Common script for generating SVG matrix transformation string. Note this
    doesn't [currently] allow for skew arguments.
    """
    recentre_xy = centre_xy - (scaling_xy * centre_xy)
    translation_xy += recentre_xy
    matrix_sequence = [
        scaling_xy[0],
        0,
        0,
        scaling_xy[1],
        translation_xy[0],
        translation_xy[1],
    ]
    matrix_string = " ".join([str(i) for i in matrix_sequence])
    return f"matrix({matrix_string})"


def _svg_clip() -> Tuple[_SvgNamedElement, np.ndarray]:
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
    size_xy = np.array([133.334, 131.521])
    visual_centre_xy = np.array([66.149, 67.952])

    scaling = LOGO_SIZE / max(size_xy)
    size_offset_xy = (size_xy - max(size_xy)) / 2
    centre_offset_xy = visual_centre_xy - (size_xy / 2)
    offset_xy = (size_offset_xy + centre_offset_xy) * scaling

    clip = _SvgNamedElement(xml_id="iris_clip", tag="clipPath", is_def=True)
    clip.append(
        ET.Element("path", attrib={"d": path_string, "transform": f"scale({scaling})"})
    )
    return clip, offset_xy


def _svg_background() -> List[_SvgNamedElement]:
    """Generate the background rectangle for the logo."""
    gradient = _SvgNamedElement(
        xml_id="background_gradient",
        tag="linearGradient",
        is_def=True,
        attrib={
            "y1": "0%",
            "y2": "100%",
        },
    )
    gradient.extend(
        [
            ET.Element(
                "stop",
                attrib={
                    "offset": "0",
                    "stop-color": "#13385d",
                },
            ),
            ET.Element(
                "stop",
                attrib={
                    "offset": "0.43",
                    "stop-color": "#0b3849",
                },
            ),
            ET.Element(
                "stop",
                attrib={
                    "offset": "1",
                    "stop-color": "#272b2c",
                },
            ),
        ]
    )
    background = _SvgNamedElement(
        xml_id="background",
        tag="rect",
        attrib={
            "height": "100%",
            "width": "100%",
            "fill": f"url(#{gradient.xml_id})",
        },
    )
    return [background, gradient]


def _svg_sea() -> List[_SvgNamedElement]:
    """Generate the circle representing the globe's sea in the logo."""
    # Not using Cartopy for sea since it doesn't actually render curves/circles.
    gradient = _SvgNamedElement(
        xml_id="sea_gradient", tag="radialGradient", is_def=True
    )
    gradient.extend(
        [
            ET.Element(
                "stop",
                attrib={"offset": "0", "stop-color": "#20b0ea"},
            ),
            ET.Element(
                "stop",
                attrib={
                    "offset": "1",
                    "stop-color": "#156475",
                },
            ),
        ]
    )
    sea = _SvgNamedElement(
        xml_id="sea",
        tag="circle",
        attrib={
            "cx": "50%",
            "cy": "50%",
            "r": f"{50.5 / CLIP_GLOBE_RATIO}%",
            "fill": f"url(#{gradient.xml_id})",
        },
    )
    return [sea, gradient]


def _svg_glow() -> List[_SvgNamedElement]:
    """Generate the coloured glow to go behind the globe in the logo."""
    gradient_scale_xy = np.array([1.15, 1.35])
    gradient_centre_xy = np.array([0.5, 0.5])
    gradient_translation_xy = np.array([0, -0.25])
    matrix_string = _matrix_transform_string(
        gradient_scale_xy, gradient_centre_xy, gradient_translation_xy
    )
    gradient = _SvgNamedElement(
        xml_id="glow_gradient",
        tag="radialGradient",
        is_def=True,
        attrib={"gradientTransform": matrix_string},
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
                "stop",
                attrib={
                    "offset": "1",
                    "stop-color": "#b6df34",
                },
            ),
        ]
    )
    blur = _SvgNamedElement(xml_id="glow_blur", tag="filter", is_def=True)
    blur.append(ET.Element("feGaussianBlur", attrib={"stdDeviation": "14"}))
    glow = _SvgNamedElement(
        xml_id="glow",
        tag="circle",
        attrib={
            "cx": "50%",
            "cy": "50%",
            "r": f"{52 / CLIP_GLOBE_RATIO}%",
            "fill": f"url(#{gradient.xml_id})",
            "filter": f"url(#{blur.xml_id})",
            "stroke": "#ffffff",
            "stroke-width": "2",
            "stroke-opacity": "0.797414",
        },
    )
    return [glow, gradient, blur]


def _svg_land(
    rotate=False,
) -> Tuple[List[_SvgNamedElement], List[_SvgNamedElement]]:
    """
    Generate the circle representing the globe's land in the logo, clipped by
    appropriate coastline shapes (using Matplotlib and Cartopy).
    """
    # Set plotting size/proportions.
    mpl_points_per_inch = 72
    plot_inches = LOGO_SIZE / mpl_points_per_inch
    plot_padding = (1 - (1 / CLIP_GLOBE_RATIO)) / 2

    # Create land with simplified coastlines.
    simple_geometries = [geometry.simplify(0.8, True) for geometry in LAND.geometries()]
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

    for lon in rotation_longitudes:
        # Use Matplotlib and Cartopy to generate land-shaped SVG clips for each longitude.
        projection_rotated = ccrs.Orthographic(
            central_longitude=lon, central_latitude=central_latitude
        )

        # Use constants set earlier to achieve desired dimensions.
        plt.figure(0, figsize=[plot_inches] * 2)
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
            xml_id=f"land_clip_{lon}",
            tag="clipPath",
            is_def=True,
        )
        mpl_land = svg_mpl.find(".//svg:g[@id='figure_1']", NAMESPACES)
        land_paths = mpl_land.find(".//svg:g[@id='PathCollection_1']", NAMESPACES)
        land_clip.extend(list(land_paths))
        for path in land_clip:
            # Remove all other attribute items.
            path.attrib = {"d": path.attrib["d"], "stroke-linejoin": "round"}
        land_clips.append(land_clip)
    plt.close()

    gradient = _SvgNamedElement(
        xml_id="land_gradient", tag="radialGradient", is_def=True
    )
    gradient.extend(
        [
            ET.Element("stop", attrib={"offset": "0", "stop-color": "#d5e488"}),
            ET.Element("stop", attrib={"offset": "1", "stop-color": "#aec928"}),
        ]
    )
    logo_centre = LOGO_SIZE / 2
    land = _SvgNamedElement(
        xml_id="land",
        tag="circle",
        attrib={
            "cx": "50%",
            "cy": "50%",
            "r": f"{50 / CLIP_GLOBE_RATIO}%",
            "fill": f"url(#{gradient.xml_id})",
            "clip-path": f"url(#{land_clips[0].xml_id})",
            "transform": f"rotate({perspective_tilt} {logo_centre} {logo_centre})",
        },
    )
    if rotate:
        animation_values = ";".join([f"url(#{clip.xml_id})" for clip in land_clips])
        frames = len(land_clips)
        duration = frames / 30
        land.append(
            ET.Element(
                "animate",
                attrib={
                    "attributeName": "clip-path",
                    "values": animation_values,
                    "begin": "0s",
                    "repeatCount": "indefinite",
                    "dur": f"{duration}s",
                },
            )
        )
    return [land, gradient], land_clips


def _svg_logos(
    iris_clip: _SvgNamedElement,
    other_elements: Iterable[_SvgNamedElement],
    offset_xy: np.ndarray,
    banner_size_xy: np.ndarray,
    banner_text: str,
    banner_version: str = None,
) -> Tuple[_SvgNamedElement, _SvgNamedElement]:
    """Assemble sub-elements into SVG's for the logo and banner."""
    # Make the logo SVG first.
    logo_root = _SvgNamedElement(
        "logo", "svg", attrib={"viewBox": f"0 0 {LOGO_SIZE} {LOGO_SIZE}"}
    )

    # The elements that will just be referenced by artwork elements.
    defs_element = _SvgNamedElement("defs", "defs")
    defs_element.append(iris_clip)
    # The elements that are displayed (not just referenced).
    # All artwork is clipped by the Iris shape.
    artwork_element = _SvgNamedElement(
        "artwork", "g", attrib={"clip-path": f"url(#{iris_clip.xml_id})"}
    )
    for element in other_elements:
        assert isinstance(element, _SvgNamedElement)
        if element.is_def:
            target_parent = defs_element
        else:
            target_parent = artwork_element
        target_parent.append(element)
    logo_root.extend((defs_element, artwork_element))

    # Shrink and translate contents - aligning the offset centre with the image
    # dimensional centre.
    offset_scaling_xy = (LOGO_SIZE - abs(offset_xy * 3)) / LOGO_SIZE
    # Lock aspect ratio.
    offset_scaling_xy = np.repeat(min(offset_scaling_xy), 2)
    logo_centre_xy = np.repeat(LOGO_SIZE / 2, 2)
    matrix_string = _matrix_transform_string(
        offset_scaling_xy, logo_centre_xy, offset_xy * -1
    )
    artwork_element.attrib["transform"] = matrix_string

    # Take a copy for including in the banner, BEFORE adding final properties.
    banner_logo = deepcopy(logo_root)

    logo_root.attrib.update(dict.fromkeys(("width", "height"), str(LOGO_SIZE)))
    logo_desc = ET.Element("desc")
    logo_desc.text = (
        "Logo for the SciTools Iris project - https://github.com/SciTools/iris/"
    )
    logo_root.insert(0, logo_desc)

    ############################################################################
    # Make the banner SVG, incorporating the logo SVG.
    banner_root = _SvgNamedElement("banner", "svg")
    for dimension, name in enumerate(("width", "height")):
        banner_root.attrib[name] = str(banner_size_xy[dimension])

    banner_desc = ET.Element("desc")
    banner_desc.text = (
        "Banner logo for the SciTools Iris project - https://github.com/SciTools/iris/"
    )
    banner_root.insert(0, banner_desc)

    # Left-align the logo.
    banner_logo.attrib["preserveAspectRatio"] = "xMinYMin meet"
    banner_root.append(banner_logo)

    # Text element(s).
    banner_height = banner_size_xy[1]
    text_size = banner_height * TEXT_GLOBE_RATIO
    text_x = banner_size_xy[0] - 16
    # Manual y centring since SVG dominant-baseline not widely supported.
    text_y = banner_height - (banner_height - text_size) / 2
    text_y *= 0.975  # Slight offset
    text_common_attrib = {
        "x": str(text_x),
        "font-family": "georgia",
        "text-anchor": "end",
    }

    text_element = _SvgNamedElement(
        "text",
        "text",
        attrib=dict(
            {
                "y": str(text_y),
                "font-size": f"{text_size}pt",
            },
            **text_common_attrib,
        ),
    )
    text_element.text = banner_text
    banner_root.append(text_element)

    if banner_version is not None:
        version_size = text_size / 6
        version_y = text_y + version_size + 16
        version_element = _SvgNamedElement(
            "version",
            "text",
            attrib=dict(
                {"y": str(version_y), "font-size": f"{version_size}pt"},
                **text_common_attrib,
            ),
        )
        version_element.text = banner_version
        banner_root.append(version_element)

    ############################################################################

    return logo_root, banner_root


def _write_svg_file(
    filename: str,
    svg_root: _SvgNamedElement,
    write_dir: Union[Path, str] = None,
    zip_archive: ZipFile = None,
) -> Path:
    """Format the svg then write the svg to a file in write_dir, or
    optionally to an open ZipFile."""
    # Add a credit comment at top of SVG.
    comment = (
        f"Created by https://github.com/SciTools/marketing/iris/logo/generate_logo.py"
    )
    svg_root.insert(0, ET.Comment(comment))

    input_string = ET.tostring(svg_root)
    pretty_xml = minidom.parseString(input_string).toprettyxml()
    # Remove extra empty lines from Matplotlib.
    pretty_xml = "\n".join([line for line in pretty_xml.split("\n") if line.strip()])

    if isinstance(zip_archive, ZipFile):
        # Add to zip file if zip_archive provided.
        zip_archive.writestr(filename, pretty_xml)
        result = Path(zip_archive.filename)
    elif Path(write_dir).is_dir():
        # Otherwise write to file normally.
        write_path = write_dir.joinpath(filename)
        with open(write_path, "w") as f:
            f.write(pretty_xml)
        result = write_path
    else:
        raise ValueError("No valid write_dir or zip_archive provided.")

    return result


def generate_logo(
    filename_prefix: str = "iris",
    write_dir: Union[str, Path] = Path.cwd(),
    banner_text: str = "Iris",
    banner_width: int = 588,
    banner_version: str = None,
    rotate: bool = False,
) -> Dict[str, Path]:
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
    banner_version : str, optional
        A version string to include in the banner logo. Default is None.
    rotate : bool
        Whether to also generate rotating earth logos.
        NOTE: takes approx 1min to generate this. Also, animated SVG files are
              known to cause high web-browser CPU demand.

    Returns
    -------
    dict of pathlib.Path
        Paths of the created logo files.

    """

    print("LOGO GENERATION START ...")

    write_dir = Path(write_dir)
    banner_width = int(banner_width)
    rotate = bool(rotate)

    # Pixel dimensions of text banner.
    banner_size_xy = [banner_width, BANNER_HEIGHT]

    # Get SVG and info for the logo's clip.
    # clip_offset_xy: used to align clip visual centre with image dimensional centre.
    iris_clip, clip_offset_xy = _svg_clip()

    # Make a list of the SVG elements that don't need explicit naming in
    # _svg_logo(). Ordering is important.
    svg_elements = [*_svg_background(), *_svg_glow(), *_svg_sea()]

    logo_kwargs = {
        "iris_clip": iris_clip,
        "offset_xy": clip_offset_xy,
        "banner_size_xy": banner_size_xy,
        "banner_text": banner_text,
        "banner_version": banner_version,
    }

    logo_names = ("logo", "logo-title")
    written_paths = {}

    # Create the main logos.
    # Specialised SVG objects for the land and the coastlines clip.
    svg_land, land_clip = _svg_land()
    logos_static = _svg_logos(
        other_elements=svg_elements + svg_land + land_clip, **logo_kwargs
    )
    for logo_type_ix, suffix in enumerate(logo_names):
        filename = f"{filename_prefix}-{suffix}.svg"
        logo_write = logos_static[logo_type_ix]
        logo_path = _write_svg_file(filename, logo_write, write_dir=write_dir)
        written_paths[suffix] = logo_path

    ###########################################################################
    if rotate:
        logo_names_rotate = [suffix + "_rotate" for suffix in logo_names]
        svg_land_rotating, land_clip_rotations = _svg_land(rotate=True)

        # Logos animated to rotate.
        logos_rotating = _svg_logos(
            other_elements=svg_elements + svg_land_rotating + land_clip_rotations,
            **logo_kwargs,
        )
        for logo_type_ix, suffix in enumerate(logo_names_rotate):
            filename = f"{filename_prefix}-{suffix}.svg"
            logo_write = logos_rotating[logo_type_ix]
            logo_path = _write_svg_file(filename, logo_write, write_dir=write_dir)
            written_paths[f"{suffix}_svg"] = logo_path

        # A series of fixed logos for each rotation longitude.
        fixed_clip_name = "land_clip"
        svg_land[0].attrib["clip-path"] = f"url(#{fixed_clip_name})"
        for clip in land_clip_rotations:
            clip.attrib["id"] = fixed_clip_name
        logos_rotated = [
            _svg_logos(other_elements=svg_elements + svg_land + [clip], **logo_kwargs)
            for clip in land_clip_rotations
        ]
        for logo_type_ix, suffix in enumerate(logo_names_rotate):
            # Insert fixed rotation logos into a ZIP file.
            zip_path = write_dir.joinpath(f"{filename_prefix}-{suffix}.zip")
            with ZipFile(zip_path, "w") as rotate_zip:
                for ix, logo in enumerate(logos_rotated):
                    logo_write = logo[logo_type_ix]
                    _write_svg_file(
                        f"{suffix}{ix:03d}",
                        logo_write,
                        zip_archive=rotate_zip,
                    )

                readme_str = (
                    "Several tools are available to stitch these "
                    "images into a rotating GIF.\n\nE.g. "
                    "http://blog.gregzaal.com/2015/08/06/making-an"
                    "-optimized-gif-in-gimp/"
                )
                rotate_zip.writestr("_README.txt", readme_str)

            written_paths[f"{suffix}_zip"] = zip_path

    ###########################################################################

    print("LOGO GENERATION COMPLETE")

    return written_paths


def main():
    description = (
        "Command line access for the generate_logo() "
        "function - docstring above. Input the parameters "
        "as optional arguments (--my_arg value). The return value("
        "s) will be printed."
    )
    docstring = str(generate_logo.__doc__)
    parser = ArgumentParser(description=description, usage=docstring)
    _, args = parser.parse_known_args()

    keys = args[::2]
    values = args[1::2]
    assert all([key[:2] == "--" for key in keys])
    keys = [key[2:] for key in keys]
    kwargs = dict(zip(keys, values))

    print(generate_logo(**kwargs))


if __name__ == "__main__":
    main()

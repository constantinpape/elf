import zipfile
from xml.dom import minidom
from typing import List


def _parse_attributes(xml_elem, parse_input) -> List:
    """Parse xml input.

    Args:
        xml_elem: An XML parsing element containing an "attributes" member.
        parse_input: [["attribute_name", python_type_name],
                      ["52", int], ["1.234", float], ["neurite", str], ...]

    Returns:
        list of python-typed values - [52, 1.234, "neurite", ...]
    """
    parse_output = []
    attributes = xml_elem.attributes
    for x in parse_input:
        try:
            if x[1] == int:
                # ensure float strings can be parsed too
                parse_output.append(int(float(attributes[x[0]].value)))
            else:
                parse_output.append(x[1](attributes[x[0]].value))
        except (KeyError, ValueError):
            parse_output.append(None)
    return parse_output


# Construct annotation. Annotations are trees (called things inside the nml files).
def read_coords_from_nml(nml):
    """@private
    """
    annotation_elems = nml.getElementsByTagName("thing")
    skeleton_coordinates = {}

    # TODO parse the skeleton id and use as key instead of linear index
    for skel_id, annotation_elem in enumerate(annotation_elems):
        node_elems = annotation_elem.getElementsByTagName("node")
        coords = []
        for node_elem in node_elems:
            x, y, z = _parse_attributes(node_elem, [["x", float], ["y", float], ["z", float]])
            # TODO is this stored in physical coordinates?
            # need to transform appropriately
            coords.append([z, y, x])
        skeleton_coordinates[skel_id] = coords
    return skeleton_coordinates


def read_edges_from_nml(nml):
    """@private
    """
    annotation_elems = nml.getElementsByTagName("thing")
    skeleton_edges = {}

    # TODO parse the skeleton id and use as key instead of linear index
    for skel_id, annotation_elem in enumerate(annotation_elems):

        edge_elems = annotation_elem.getElementsByTagName("edge")
        edges = []

        for edge_elem in edge_elems:
            src_id, trgt_id = _parse_attributes(edge_elem, [["source", int], ["target", int]])
            edges.append([src_id, trgt_id])

        skeleton_edges[skel_id] = edges

    return skeleton_edges


# TODO read and return tree structure, comments etc.
def parse_nml(nml_str):
    """@private
    """
    # TODO figure this out
    # read the pixel size
    # try:
    #     param = nml_str.getElementsByTagName("parameters")[0].getElementsByTagName("scale")[0]
    #     file_scaling = parse_attributes(param,
    #                                     [["x", float], ["y", float], ["z", float]])
    # except IndexError:
    #     # file_scaling = [1, 1, 1]
    #     pass
    coord_dict = read_coords_from_nml(nml_str)
    edge_dict = read_edges_from_nml(nml_str)
    return coord_dict, edge_dict


# TODO return additional annotations etc
# TODO figure out scaling
def read_nml(input_path: str):
    """Read skeleton stored in nml or nmx format.

    The nml format is used by Knossos. For details on the nml format see:
    https://github.com/knossos-project/knossos_utils/blob/master/knossos_utils/skeleton.py

    Args:
        input_path: Path to the nml/nmx file.

    Returns:
        The parsed skeleton.
    """
    # from knossos zip
    if input_path.endswith("k.zip"):
        zipper = zipfile.ZipFile(input_path)
        if "annotation.xml" not in zipper.namelist():
            raise Exception("k.zip file does not contain annotation.xml")
        xml_string = zipper.read("annotation.xml")
        nml = minidom.parseString(xml_string)
        out = parse_nml(nml)

    # from nmx (pyKnossos)
    elif input_path.endswith("nmx"):

        out = {}
        with zipfile.ZipFile(input_path, "r") as zf:
            for ff in zf.namelist():
                if not ff.endswith(".nml"):
                    continue
                nml = minidom.parseString(zf.read(ff))
                out[ff] = parse_nml(nml)

    # from nml
    else:
        nml = minidom.parse(input_path)
        out = parse_nml(nml)

    return out


# def write_nml():
#     pass

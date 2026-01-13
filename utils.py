import pandas as pd
import numpy as np
from pathlib import Path
from girder_client import GirderClient
import shutil

# from dsa_helpers import imread, imwrite
# from dsa_helpers.girder_utils import get_thumbnail, get_item_large_image_metadata


def replace_dots_in_keys(d, sep=" "):
    if isinstance(d, dict):
        return {k.replace(".", sep): replace_dots_in_keys(v, sep) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_dots_in_keys(i, sep) for i in d]
    else:
        return d


def format_ann_docs_for_paperjs(
    geojson_ann_docs: list[dict], return_as_input_to_paper: bool = False
) -> list[dict]:
    """Format DSA annotations in the geojson format into a format that paperdragon
    can read and show as polygons.

    Args:
        geojson_ann_docs (list[dict]): A list of geojson annotation documents.
        return_as_input_to_paper (bool): Whether to return the formatted documents as
            input to paper. Defaults to False.

    Returns:
        list[dict]: A list of formatted annotation documents for paperjs. Note that
            it will return an empty list if the documents could not be formatted.

    """
    # NOTE: hard-coded to only do the first doc.
    formatted_docs = []

    # Loop through each annotation document.
    for ann_doc in geojson_ann_docs:
        # Grab the features of this document, contains the polygon info.
        features_to_include = []

        for feature in ann_doc["features"]:
            geometry = feature["geometry"]
            if geometry["type"] == "Polygon":
                properties = feature["properties"]
                properties["rescale"] = {"strokeWidth": properties["lineWidth"]}
                properties["strokeColor"] = properties["lineColor"]
                properties["source"] = "dsa"

                # Paperjs only supports multipolygons.
                geometry["type"] = "MultiPolygon"

                coordinates = geometry["coordinates"]

                adjusted_coordinates = [
                    [[[int(coord) for coord in point[:2]] for point in coordinates[0]]]
                ]

                geometry["coordinates"] = adjusted_coordinates

                features_to_include.append(feature)

        if len(features_to_include):
            formatted_docs.append(features_to_include)

    if return_as_input_to_paper:
        return get_input_to_paper_dict(formatted_docs)

    return formatted_docs


def get_input_to_paper_dict(formatted_docs: list[dict] | None = None) -> dict:
    """Get the input to paper dictionary.

    Args:
        formatted_docs (list[dict]): A list of formatted annotation documents for paperjs.

    Returns:
        dict: The input to paper dictionary.

    """
    input_to_paper = [{"type": "clearItems"}]

    if formatted_docs is not None and len(formatted_docs):
        for doc in formatted_docs:
            input_to_paper.append(
                {
                    "type": "drawItems",
                    "itemList": doc,
                }
            )

    return {"actions": input_to_paper}


def format_dsa_annotations(geojson_ann_docs: list[dict]) -> list[dict]:
    """Format DSA annotations in the geojson format into a format that paperdragon
    can read and show as polygons.

    Args:
        geojson_ann_docs (list[dict]): A list of geojson annotation documents.

    Returns:
        list[dict]: A list of formatted annotation documents for paperjs. Note that
            it will return an empty list if the documents could not be formatted.

    """
    # NOTE: hard-coded to only do the first doc.
    formatted_docs = []

    # Loop through each annotation document.
    for ann_doc in geojson_ann_docs:
        # Grab the features of this document, contains the polygon info.
        features_to_include = []

        for feature in ann_doc["features"]:
            geometry = feature["geometry"]
            if geometry["type"] == "Polygon":
                properties = feature["properties"]
                properties["rescale"] = {"strokeWidth": properties["lineWidth"]}
                properties["strokeColor"] = properties["lineColor"]
                properties["source"] = "dsa"

                # Paperjs only supports multipolygons.
                geometry["type"] = "MultiPolygon"

                coordinates = geometry["coordinates"]

                adjusted_coordinates = [
                    [[[int(coord) for coord in point[:2]] for point in coordinates[0]]]
                ]

                geometry["coordinates"] = adjusted_coordinates

                features_to_include.append(feature)

        if len(features_to_include):
            formatted_docs.append(features_to_include)

    return formatted_docs

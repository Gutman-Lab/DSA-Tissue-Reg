import dash_paperdragon
import base64
from PIL import Image
import io

from dash_paperdragon import utils
from dash_paperdragon.utils import (
    getId,
    CHANNEL_COLORS,
    annotationToGeoJson,
    get_box_instructions,
    generate_random_boxes,
    classes,
    colors,
    generate_dsaStyle_string,
)

from dash import (
    Dash,
    callback,
    html,
    Input,
    Output,
    dcc,
    State,
    ALL,
    MATCH,
    callback_context,
    no_update,
)
import dash_bootstrap_components as dbc
import json, random
import dash_ag_grid
from pprint import pprint
import dashPaperDragonHelpers as hlprs
import requests
import sampleTileSources as sts
import re
import girder_client  # In theory not necessary, could use requests
from sampleTileSources import tileSources

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=[
        "https://unpkg.com/react@17/umd/react.production.min.js",
        "https://unpkg.com/react-dom@17/umd/react-dom.production.min.js",
        "/assets/react-color-bundle.js",  # Load this first
        "/assets/dashAgGridFunctions.js",  # Then load your grid functions
    ],
)


# possible bindings for actions:
# keyDown
# keyUp
# mouseEnter
# mouseLeave
# click ?

# supported actions:
# cycleProp (property)
# cyclePropReverse (property)
# deleteItem
# newItem
# editItem
# dashCallback (callback)


def create_base64_rectangle():
    # Create a small rectangle image (50x50 pixels)
    img = Image.new("RGB", (50, 50), color="red")

    # Convert PIL image to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_str}", 50, 50


# hsi_color = rgb_to_hsi(rgb_color['r'], rgb_color['g'], rgb_color['b'])

demo_inputToPaper = {"actions": [{"type": "drawItems", "itemList": sts.sampleShapes}]}


tileSourceDict = {x["label"]: x["tileSources"] for x in tileSources}

tileSourceDictTwo = {x["label"]: x for x in tileSources}


config = {
    "eventBindings": [
        {"event": "keyDown", "key": "c", "action": "cycleProp", "property": "class"},
        {
            "event": "keyDown",
            "key": "x",
            "action": "cyclePropReverse",
            "property": "class",
        },
        {"event": "keyDown", "key": "d", "action": "deleteItem"},
        {"event": "keyDown", "key": "n", "action": "newItem", "tool": "rectangle"},
        {"event": "keyDown", "key": "e", "action": "editItem", "tool": "rectangle"},
        {"event": "keyDown", "key": "l", "action": "grabColor"},
        {"event": "mouseEnter", "action": "dashCallback", "callback": "mouseEnter"},
        {"event": "mouseLeave", "action": "dashCallback", "callback": "mouseLeave"},
    ],
    "callbacks": [
        {"eventName": "item-created", "callback": "createItem"},
        {"eventName": "property-changed", "callback": "propertyChanged"},
        {"eventName": "item-deleted", "callback": "itemDeleted"},
        {"eventName": "item-edited", "callback": "itemEdited"},
    ],
    "properties": {"class": classes},
    "defaultStyle": {
        "fillColor": colors[0],
        "strokeColor": colors[0],
        "rescale": {
            "strokeWidth": 1,
        },
        "fillOpacity": 0.2,
    },
    "styles": {
        "class": {
            k: {"fillColor": c, "strokeColor": c} for (k, c) in zip(classes, colors)
        }
    },
}


imgSrc_control_table = dash_ag_grid.AgGrid(
    id="imgSrc_table",
    rowData=[],
    columnDefs=[
        {
            "field": "idx",
            "width": 75,
        },  ## This is important to track as this is what OSD uses internally
        {
            "field": "scaleFactor",
            "headerName": "Scale Factor",
            "width": 100,
            "editable": True,
        },
        {
            "field": "palette",
            "headerName": "Color",
            "cellRenderer": "colorCellRenderer",
            "width": 100,
            "editable": True,
        },
        {"field": "colorMeThis", "cellEditor": "colorPickerEditor", "editable": True},
        {
            "field": "isVisible",
            "cellRenderer": "agCheckboxCellRenderer",
            "cellEditor": "agCheckboxCellEditor",
            "editable": True,
            # "headerName": "<span>&#x1F441;</span>",
            "width": 100,
        },
        {
            "field": "flipped",
            "headerName": "Flip",
            "cellRenderer": "agCheckboxCellRenderer",
            "cellEditor": "agCheckboxCellEditor",
            "editable": True,
            "width": 100,
        },
        {"field": "_id", "header": "Item ID"},
        {
            "field": "opacity",
            "header": "Opacity",
            "width": 100,
            "cellEditor": "agNumberCellEditor",
            "cellEditorParams": {
                "min": 0,
                "max": 1,
                "precision": 2,
                "step": 0.01,
                "showStepperButtons": True,
            },
            "editable": True,
        },
        {"field": "xOffset", "header": "X Offset", "editable": True},
        {"field": "yOffset", "header": "Y Offset", "editable": True},
        # {"field": "xOffsetPixels", "header": "X Offset", "editable": True},
        # {"field": "yOffsetPixels", "header": "Y Offset", "editable": True},
        {"field": "rotation", "header": "Rotation", "editable": True},
        {"field": "width", "header": "Width", "width": 80},
        {"field": "height", "header": "Height", "width": 80},
        {"field": "sizeX"},
        {"field": "sizeY"},
    ],
    defaultColDef={
        "resizable": True,
        "sortable": True,
        "filter": True,
        "columnSize": "autoSize",
        "maxWidth": 120,
    },
    dashGridOptions={"rowHeight": 25},
    style={"height": "200px"},
)


geoJsonShapeColumns = [
    {"field": "userdata.objId", "headerName": "objId", "width": 80, "sortable": True},
    {"field": "userdata.class", "width": 80},
    {"field": "type", "headerName": "Type", "width": 120},
    {"field": "properties.fillOpacity", "headerName": "fill %", "width": 80},
    {"field": "properties.strokeColor"},
    {"field": "properties.fillColor", "width": 100},
    {"field": "rotation"},
]


annotationTableCols = [
    {"field": "annotation.name"},
    {"field": "_version"},
    {"field": "updated"},
    {"field": "_id"},
]

dsaAnnotation_table = dash_ag_grid.AgGrid(
    id="annotationTable",
    columnDefs=annotationTableCols,
    columnSizeOptions={"defaultMaxWidth": 200},
    defaultColDef={
        "resizable": True,
        "sortable": True,
        "defaultMaxWidth": 150,
    },
    dashGridOptions={
        "pagination": True,
        "paginationAutoPageSize": True,
        "rowSelection": "single",
    },
    style={"height": "300px"},
)


def get_default_tilesource_props():
    """Define default properties for any tile source"""
    return {
        "x": 0,
        "y": 0,
        "opacity": 1,
        "rotation": 0,
        "flipped": False,
        "scaleFactor": 1,
        "mm_x": 0.00025,  # Default for ISIC images
        "isVisible": True,
    }


def normalize_tile_source(r):
    """Normalize a single tile source configuration"""
    try:
        defaults = get_default_tilesource_props()

        # Build the tile source URL
        imgTileSource = f"{r['apiUrl']}/item/{r['_id']}/tiles/dzi.dzi"
        if r.get("palette"):
            imgTileSource += "?style=" + generate_dsaStyle_string(
                r["palette"][1], r.get("opacity", defaults["opacity"])
            )

        # Ensure safe division
        sizeX = float(r.get("sizeX", 1)) or 1  # Use 1 if sizeX is 0 or None

        return {
            "tileSource": imgTileSource,
            "x": float(r.get("xOffset", defaults["x"])) / sizeX,
            "y": float(r.get("yOffset", defaults["y"])) / sizeX,
            "opacity": float(r.get("opacity", defaults["opacity"])),
            "rotation": float(r.get("rotation", defaults["rotation"])),
            "flipped": bool(r.get("flipped", defaults["flipped"])),
            "scaleFactor": float(r.get("scaleFactor", defaults["scaleFactor"])),
        }
    except Exception as e:
        print(f"Error in normalize_tile_source for {r.get('_id', 'NO_ID')}: {e}")
        return None


def create_tile_source_props(r, index=None):
    """Create tile source properties for viewer updates"""
    try:
        defaults = get_default_tilesource_props()
        sizeX = float(r.get("sizeX", 1)) or 1  # Use 1 if sizeX is 0 or None

        imgTileSource = f"{r['apiUrl']}/item/{r['_id']}/tiles/dzi.dzi"
        props = {
            "tileSource": imgTileSource,
            "opacity": (
                float(r.get("opacity", defaults["opacity"]))
                if r.get("isVisible", defaults["isVisible"])
                else 0
            ),
            "x": float(r.get("xOffset", defaults["x"])) / sizeX,
            "y": float(r.get("yOffset", defaults["y"])) / sizeX,
            "rotation": float(r.get("rotation", defaults["rotation"])),
            "flipped": bool(r.get("flipped", defaults["flipped"])),
            "scaleFactor": float(r.get("scaleFactor", defaults["scaleFactor"])),
            "tileSource": imgTileSource,
        }

        if index is not None:
            props["index"] = index

        return props
    except Exception as e:
        print(f"Error in create_tile_source_props for {r.get('_id', 'NO_ID')}: {e}")
        return None


## Create element
osdElement = dash_paperdragon.DashPaperdragon(
    id="osdViewerComponent",
    config=config,
    zoomLevel=0,
    viewportBounds={"x": 0, "y": 0, "width": 0, "height": 0},
    curMousePosition={"x": 0, "y": 0},
    inputToPaper=None,  # demo_inputToPaper,  ## If I am doing this.. I also need to put it in the shape table
    outputFromPaper=None,
    viewerWidth=800,
    pixelColor={"r": 0, "g": 0, "b": 0},
)


@callback(
    Output("pixelColor_disp", "children"), Input("osdViewerComponent", "pixelColor")
)
def update_pixel_color(pixelColor):
    print(pixelColor, "was received")
    if pixelColor:
        color_string = f"RGB: {pixelColor['r']}, {pixelColor['g']}, {pixelColor['b']}"

        rgbColor = f"rgb({pixelColor['r']}, {pixelColor['g']}, {pixelColor['b']})"
        color_box_style = {
            "display": "inline-block",
            "width": "20px",
            "height": "20px",
            "backgroundColor": rgbColor,
            "marginLeft": "10px",
            "border": "1px solid #000",
        }
        return html.Div([html.Span(color_string), html.Div(style=color_box_style)])


## Make HTML layout
coordinate_display = html.Div(
    [
        dbc.Row([dbc.Col(html.H2("Zoom and Mouse Position"), className="mb-2")]),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Zoom Level", className="card-title"),
                                    html.Div(
                                        id="zoomLevel_disp", className="card-text"
                                    ),
                                ]
                            )
                        ],
                        className="mb-1",
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Viewport Bounds", className="card-title"),
                                    html.Div(
                                        id="viewportBounds_disp", className="card-text"
                                    ),
                                ]
                            )
                        ],
                        className="mb-1",
                    ),
                    width=8,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H6("Mouse Position", className="card-title"),
                                    html.Div(id="mousePos_disp", className="card-text"),
                                ]
                            )
                        ],
                        className="img-control-grid",
                        style={"height": "3.5rem"},
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H6("Pixel Color", className="card-title"),
                                    html.Div(
                                        id="pixelColor_disp", className="card-text"
                                    ),
                                ]
                            )
                        ],
                        className="mb-1 img-control-card",
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H6(
                                        "Highlighted Object", className="card-title"
                                    ),
                                    html.Div(
                                        id="curObject_disp", className="card-text"
                                    ),
                                ]
                            )
                        ],
                        className="mb-1 img-control-card",
                    ),
                    width=8,
                ),
            ]
        ),
        dbc.Row(
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H5(
                                "Tile Source Info", className="card-title text-center"
                            ),
                            # html.Div(id="osdTileProperties", className="card-text"),
                            # html.Div(id="imgScrControls_data", className="card-text"),
                            imgSrc_control_table,
                        ]
                    )
                ],
                className="img-control-card",
            )
        ),
        dbc.Row(
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H5("Shape Table", className="card-title text-center"),
                            dash_ag_grid.AgGrid(
                                id="shapeDataTable",
                                columnDefs=geoJsonShapeColumns,
                                columnSizeOptions={"defaultMaxWidth": 200},
                                # columnSize="sizeToFit",
                                rowData=sts.sampleShapes,  ## Only if setting sample shapes above..
                                defaultColDef={
                                    "resizable": True,
                                    "sortable": True,
                                    "defaultMaxWidth": 150,
                                },
                                style={"height": "300px"},
                            ),
                        ],
                        style={
                            "margin": "0px",
                            "padding": "0px",
                            "marginBottom": "10px",
                            # "height": "200px",
                        },
                    )
                ],
                className="img-control-card",
                style={"height": "300px"},
            )
        ),
        dbc.Row(
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H5(
                                "Annotation Table", className="card-title text-center"
                            ),
                            dsaAnnotation_table,
                        ],
                        style={
                            "margin": "0px",
                            "padding": "0px",
                            "marginBottom": "10px",
                        },
                    )
                ],
                className="img-control-card",
                style={"height": "100px"},
            )
        ),
    ],
    className=" g-0",
    # style={"display": "flex", "flex-direction": "row"},
)

# Add this near your other UI components
add_source_button = dbc.Button(
    "Add Tile Source",
    id="add_source_button",
    color="primary",
    className="me-2",
    n_clicks=0,
)


imageSelect_dropdown = html.Div(
    [
        html.Label(
            "Select an image",
            className="text-center mb-3",
            style={"marginTop": "5px", "marginRight": "5px"},
        ),  # "margin-bottom": "5px
        dbc.Select(
            id="imageSelect",
            options=[x["label"] for x in tileSources],
            value=tileSources[0]["label"],  ## Starting with imageStack example
            className="mb-4 d-inline",
            style={"width": "300px", "marginLleft": "10px", "marginTop": "1px"},
        ),
        dbc.Button(
            "Make Random Rects",
            id="make_random_button",
            className="m-1 d-inline",
            style={"marginLeft": "10px", "height": "40px"},  # , "marginTop": "15px"},
        ),
        add_source_button,
        dbc.Switch(
            id="clearItems-toggle",
            label="Clear Items",
            value=False,
            className="mt-2 d-inline",
        ),
    ],
    style={"display": "flex", "flexDirection": "row", "align": "center"},
)


app.layout = dbc.Container(
    [
        dcc.Store(id="osdShapeData_store", data=[]),
        dbc.Row(dbc.Col(html.H1("Dash Paperdragon", className="text-center"))),
        dbc.Row(
            [
                dbc.Col([imageSelect_dropdown, osdElement], width=8),
                dbc.Col(
                    coordinate_display,
                    width=4,
                ),
            ],
        ),
    ],
    fluid=True,
)
## End of layout

### OutputFromPaper needs to be cleared as well once the message/state has been acknowledged


def find_index_by_objId(data, target_objId):
    for index, item in enumerate(data):
        if item.get("userdata", {}).get("objId", None) == target_objId:
            return index
    return -1  # return -1 if no match is found


def standardize_tilesource_properties(imgSet_data):
    ## The imgSet_data should have  the following format
    #   {
    #     "label": "TCGA-BF-A1Q0-01A-02-TSB",
    #     "_id": "5b9f10a8e62914002e956509",
    #     "apiUrl": "https://api.digitalslidearchive.org/api/v1/",
    #     "tileSources": [{"tileSourceId": "5b9f10a8e62914002e956509"}],
    # },

    tileSourceList = imgSet_data["tileSources"]
    ### This function will take a complicated tile source Object and make sure it has all the necessary properties for the table
    ## This will return a flattened list, where each image has all the necessary properties

    apiUrl = imgSet_data["apiUrl"]

    imgSources = []

    for idx, ts in enumerate(tileSourceList):
        ## in this separate case... the tileSourz
        if isinstance(ts, str):
            ts = {
                "tileSourceId": ts,
                "x": 0,
                "y": 0,
                "opacity": 1,
                "flipped": False,
                "apiUrl": apiUrl,
            }

        if "sizeX" not in ts:
            try:
                ## This will break if the girder Client requires login.. didn't want this issue while debugging
                gc = girder_client.GirderClient(apiUrl=apiUrl)
                tileInfo = gc.get(f"item/{ts['tileSourceId']}/tiles")
                ts.update(tileInfo)
            except Exception as e:
                print(f"Error getting tile info: {e}")
                continue

        # Use a default value of 0.00025 for mm_x if it's not present
        mm_x = ts.get("mm_x", 0.00025)
        ##sizeX needss to be defined for any DSA tilesource.. so not sure GET makes sense.. may be better to fail
        sizeX = ts["sizeX"]  ###
        sizeY = ts["sizeY"]

        ## Patch in the palette if it exists..
        if not mm_x:
            mm_x = 0.00025
        ## I create a new img_dict and make sure I have all the properties defined
        img_dict = {
            "isVisible": True,
            "flipped": ts.get("flipped", False),
            "opacity": ts.get("opacity", 1),
            "_id": ts["tileSourceId"],
            "apiUrl": apiUrl,
            "rotation": ts.get("rotation", 0),
            "width": sizeX * mm_x,  # Safe multiplication
            "height": sizeY * mm_x,
            "xOffset": ts.get("x", 0),
            "yOffset": ts.get("y", 0),
            "idx": idx,
            "palette": ts.get("palette", None),
            "sizeX": sizeX,
            "sizeY": sizeY,
            "scaleFactor": ts.get("scaleFactor", 1),
        }
        imgSources.append(img_dict)

    return imgSources


### This initial function will take a tile source and make sure it has all the necessary properties


## THIS CALLBACK ONLY DEALS WITH THE UPDATING THE IMG SOURCE TABLE.. CHANGING THE SELECTED IMAGE GROUP DOESN"T DO ANTTHINH YET


def add_random_base64_tilesource():
    base64_img, width, height = create_base64_rectangle()

    # Create a complete tile source object with all required properties
    new_tilesource = {
        "isVisible": True,
        "flipped": False,
        "opacity": 0.5,
        "_id": f"base64_{random.randint(1000, 9999)}",  # Generate random ID
        "apiUrl": None,
        "rotation": 0,
        "width": width,
        "height": height,
        "xOffset": 0,
        "yOffset": 0,
        "idx": 0,  # This will be adjusted when added to existing sources
        "palette": None,
        "sizeX": width,
        "sizeY": height,
        "scaleFactor": 1,
        "tileSource": {
            "type": "image",
            "url": base64_img,
            "width": width,
            "height": height,
        },
    }

    paper_input = {
        "actions": [
            {
                "type": "addTileSource",
                "tileSource": {
                    "tileSource": new_tilesource["tileSource"],
                    "x": new_tilesource["xOffset"],
                    "y": new_tilesource["yOffset"],
                    "opacity": new_tilesource["opacity"],
                    "rotation": new_tilesource["rotation"],
                    "flipped": new_tilesource["flipped"],
                    "scaleFactor": new_tilesource["scaleFactor"],
                },
            }
        ]
    }

    return paper_input, new_tilesource


@callback(
    Output("osdViewerComponent", "tileSourceProps"),
    Output("osdViewerComponent", "tileSources"),
    Output("osdViewerComponent", "inputToPaper"),
    Output("imgSrc_table", "rowData"),
    [
        Input("imgSrc_table", "cellValueChanged"),
        Input("imgSrc_table", "rowData"),
        Input("imageSelect", "value"),
        Input("add_source_button", "n_clicks"),
    ],
    State("osdViewerComponent", "tileSources"),
    prevent_initial_call=False,
)
def update_tilesource_props(
    cellChanged, img_data, dataSetSelect, add_source_clicks, currentTileSources
):
    print(cellChanged, "is the cellChanged")
    print(img_data, "is the img_data")
    print(dataSetSelect, "is the dataSetSelect")
    print(add_source_clicks, "is the add_source_clicks")
    print(currentTileSources, "is the currentTileSources")

    ctx = callback_context

    triggered_prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    print(f"Triggered by: {triggered_prop_id}")

    ### First case to deal with is the initial load, or the imageSelect dropdown change
    if triggered_prop_id == "imageSelect" or not currentTileSources:
        print("Now setting image select...")
        imgSet = tileSourceDictTwo[dataSetSelect]
        imgSources = standardize_tilesource_properties(imgSet)

        normalized_tilesources = [
            normalize_tile_source(r) for r in imgSources if r is not None
        ]

        print(imgSources, "is the imgSources")
        return no_update, normalized_tilesources, no_update, imgSources

    if triggered_prop_id == "add_source_button" and add_source_clicks:
        print("Adding new tile source")

        paper_input, new_tilesource = add_random_base64_tilesource()
        # Find the maximum index in current img_data
        max_idx = -1
        if img_data:
            max_idx = max(row.get("idx", -1) for row in img_data)
        new_tilesource["idx"] = max_idx + 1

        img_data.append(new_tilesource)
        return no_update, no_update, paper_input, img_data

    # Handle table cell changes
    if triggered_prop_id == "imgSrc_table" and len(img_data):
        tilesource_props = [
            create_tile_source_props(r, i)
            for i, r in enumerate(img_data)
            if r is not None
        ]
        if not tilesource_props:
            print("No valid tile source props found")
            return no_update, no_update, no_update
        return tilesource_props, no_update, no_update, no_update

    else:
        print("No triggered prop id found")
        return no_update, currentTileSources, no_update, no_update


## NEED TO CLEAR THE MESSAGE ONCE THE EVENT FIRES...
@callback(
    Output("osdViewerComponent", "inputToPaper", allow_duplicate=True),
    Output("osdShapeData_store", "data"),
    Output("osdViewerComponent", "outputFromPaper"),
    Input("osdViewerComponent", "outputFromPaper"),
    Input("make_random_button", "n_clicks"),
    State("osdViewerComponent", "viewportBounds"),
    State("osdShapeData_store", "data"),
    State("clearItems-toggle", "value"),
    Input("annotationTable", "selectedRows"),
    prevent_initial_call=True,
)
def handleOutputFromPaper(
    paperOutput,
    make_random_boxesClicked,
    viewPortBounds,
    currentShapeData,
    clearItems,
    selectedAnnotation,
):
    ### Need to determine which input triggered the callback

    ctx = callback_context

    if not ctx.triggered:
        return no_update, no_update, {}

    try:
        triggered_prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    except:
        print("Something odd about the context... need better error handling..")
        print(ctx.triggered)
        return no_update, no_update, {}

    ## if the osdViewerComponent is the trigger then we need to process the outputFromPaper

    ## Process the annotation table selection and pull the annotation and then push
    ## it to the paperdragon, also need to convert the DSA format
    if triggered_prop_id == "annotationTable":
        shapesToAdd = annotationToGeoJson(selectedAnnotation[0])

        inputToPaper = {"actions": []}

        # If I don't clear items, I also need to update the shapesToAdd
        if clearItems:
            inputToPaper["actions"].append({"type": "clearItems"})
        inputToPaper["actions"].append({"type": "drawItems", "itemList": shapesToAdd})
        # print(shapesToAdd)
        return inputToPaper, shapesToAdd, {}

    if triggered_prop_id == "osdViewerComponent":
        osdEventType = paperOutput.get("data", {}).get("callback", None)
        if not osdEventType:
            osdEventType = paperOutput.get("callback", None)

        if osdEventType in ["mouseLeave", "mouseEnter"]:
            return no_update, no_update, {}
        elif osdEventType == "createItem":
            # print(paperOutput["data"])

            ## NEED TO MAKE SURE THE OBJECT ID IS SET
            si = get_box_instructions(
                paperOutput["data"]["point"]["x"],
                paperOutput["data"]["point"]["y"],
                paperOutput["data"]["size"]["width"],
                paperOutput["data"]["size"]["height"],
                colors[0],
                {"class": classes[0], "objId": getId()},
            )
            currentShapeData.append(si)

            return createItem(paperOutput["data"]), currentShapeData, {}
        elif osdEventType == "propertyChanged":
            ### Handle property change.. probably class change but could be color or other thing in the future
            # print(paperOutput["data"])
            # print(changedProp, "is the changedProp")
            ## TO DO--- THIS IS NOT CONSISTENTLY FIRING ON EVERY CHANGE..

            changedProp = paperOutput.get("data", {}).get("property", "")
            if changedProp == "class":
                newClass = paperOutput.get("data", {}).get("item", {}).get("class", "")
                objectId = (
                    paperOutput.get("data", {}).get("item", {}).get("objectId", "")
                )
                for r in currentShapeData:
                    if r["userdata"]["objectId"] == objectId:
                        r["userdata"]["class"] = newClass
                        print("Changed object class to", newClass)
                        break
                return no_update, currentShapeData, {}
        elif osdEventType == "itemDeleted":
            # print(paperOutput["data"]["item"])
            itemId = paperOutput["data"]["item"][1]["data"]["userdata"]["objectId"]
            print("Item Deleted", itemId)
            currentShapeData = [
                x for x in currentShapeData if x["userdata"]["objectId"] != itemId
            ]
            return no_update, currentShapeData, {}
            ### TO DO-- CLARIFY FROM TOM WHAT THE DELETEITEM callback should return in the react component

            ## Note the class is changing, but that also changes the color... will need to think about how to keep all this stuff in sync
        elif osdEventType == "itemEdited":
            print("ITEM WAS EDITED")
            print(paperOutput)
            ### NEED TO UPDATE THE TABLE WITH THE OBJECT THAT WAS UPDATED...
            ## WILL ADD A ROTATION PREOPRETY FOR NOW..
            try:
                print(currentShapeData[0])
            except:
                print(
                    "Trying to print currentShapeData[0], but it's throwing an error..."
                )
                print(currentShapeData, "is what I was trying to iterate on...")

            editedObjId = paperOutput["data"]["userdata"].get("objId", None)
            ## NEED TO DEAL WITH CASE IF objID is not set on an edited item?  This maybe shouldn't happen though.. TBD...
            editedObjIdx = find_index_by_objId(currentShapeData, editedObjId)
            if editedObjIdx == -1:
                print("Could not find object with objId", editedObjId)
                return no_update, currentShapeData, {}
            else:
                print("Found object at index", editedObjIdx)

                currentShapeData[editedObjIdx]["rotation"] = "IWASROTATED"

            return no_update, currentShapeData, {}

        else:
            print("Unhandled osdEventType", osdEventType)
            print(paperOutput, "is the paperOutput")

            return no_update, no_update, {}
    # {'callback': 'propertyChanged', 'data': {'item': {'class': 'e', 'objectId': 1}, 'property': 'class'}} is the paperOutput

    elif triggered_prop_id == "make_random_button":
        ### Clear Items.. or not..

        shapesToAdd = generate_random_boxes(3, viewPortBounds)
        inputToPaper = {"actions": []}
        # print(shapesToAdd)
        # If I don't clear items, I also need to update the shapesToAdd
        if clearItems:
            inputToPaper["actions"].append({"type": "clearItems"})
            inputToPaper["actions"].append(
                {"type": "drawItems", "itemList": shapesToAdd}
            )

            return inputToPaper, shapesToAdd, {}

        else:
            ## Add new items to paper, and also update the local array store
            print("Adding new items to paper ... keeping the old")
            inputToPaper["actions"].append(
                {"type": "drawItems", "itemList": shapesToAdd}
            )
            currentShapeData = currentShapeData + shapesToAdd
            return inputToPaper, currentShapeData, {}

    else:
        print(triggered_prop_id, "was the triggered prop")

    return no_update, no_update, {}


@callback(
    Output("annotationTable", "rowData"),
    Input("imageSelect", "value"),
)
def populate_dsa_annotation_table(imageSelect):
    imgTileSources = tileSourceDictTwo[imageSelect]["tileSources"]

    # Get the first tile source and its API URL
    if len(imgTileSources) > 0:
        firstSource = imgTileSources[0]
        apiUrl = tileSourceDictTwo[imageSelect]["apiUrl"]
        itemId = firstSource["tileSourceId"]

        annotationUrl = f"{apiUrl}/annotation?itemId={itemId}"

        r = requests.get(annotationUrl)
        annotationData = r.json()

        for a in annotationData:
            a["apiUrl"] = apiUrl

        return annotationData

    return []


### This will pull the annotations from the DSA for the given item, I am going to focus on
### the first item tilesources


## This updates the mouse tracker
@callback(
    Output("mousePos_disp", "children"), Input("osdViewerComponent", "curMousePosition")
)
def update_mouseCoords(curMousePosition):
    return (
        f'{int(curMousePosition["x"])},{int(curMousePosition["y"])}'
        if curMousePosition["x"] is not None
        else ""
    )


## Update the zoom state
@callback(
    Output("zoomLevel_disp", "children"), Input("osdViewerComponent", "zoomLevel")
)
def updateZoomLevel(currentZoom):
    return "{:.3f}".format(currentZoom)


## Update the viewportBounds display
@callback(
    Output("viewportBounds_disp", "children"),
    Input("osdViewerComponent", "viewportBounds"),
)
def update_viewportBounds(viewPortBounds):
    vp = viewPortBounds
    return (
        f'x:{int(vp["x"])} y:{int(vp["y"])} w:{int(vp["width"])} h:{int(vp["height"])}'
    )


# osdShapeData
@callback(Output("shapeDataTable", "rowData"), Input("osdShapeData_store", "data"))
def updateShapeDataTable(shapeData):
    ### The structure of the data stats with array of actions
    ## Then need to parse the actions to get the itemList
    if not shapeData:
        return []

    flattened_data = []
    for shp in shapeData:
        # flattened_data.append(hlprs.convertPaperInstructions_toTableForm(shp))
        flattened_data.append(shp)
    return flattened_data


@callback(
    Output("curObject_disp", "children"), Input("osdViewerComponent", "curShapeObject")
)
def update_curShapeObject(curShapeObject):

    print(curShapeObject, "is current shape detected..")
    if curShapeObject:
        return json.dumps(curShapeObject.get("properties", {}).get("userdata", {}))
    else:
        return no_update


def createItem(data):
    # cprint("createItem", data)
    x = get_box_instructions(
        data["point"]["x"],
        data["point"]["y"],
        data["size"]["width"],
        data["size"]["height"],
        colors[0],
        {"class": classes[0], "objId": getId()},
    )
    out = {"actions": [{"type": "drawItems", "itemList": [x]}]}
    return out


# this is if you want to trigger deleting and item from the python side
def deleteItem(id):
    output = {"actions": [{"type": "deleteItem", "id": id}]}
    return output


# this listens to a deletion event triggered from the client side
def itemDeleted(data):
    print("itemDeleted", data)
    return None


# this listens to an edited event triggered from the client side
def itemEdited(data):
    print("itemEdited", data)
    return None


# this listens to a property changed event triggered from the client side
def propertyChanged(data):
    print("propertyChanged", data)
    return None


def mouseLeave(args):
    print("Mouse Leave", args)
    return None


def mouseEnter(args):
    print(args)
    return None


if __name__ == "__main__":
    app.run_server(debug=True)


# # ### Detect changes in xOffset, yOffset, and opacity
# @callback(
#     Output("imgScrControls_data", "children"),
#     Output("osdViewerComponent", "tileSourceProps"),
#     Input({"type": "x", "index": ALL}, "value"),
#     Input({"type": "y", "index": ALL}, "value"),
#     Input({"type": "opacity", "index": ALL}, "value"),
#     Input({"type": "rotation", "index": ALL}, "value"),
# )
# def process_tileSource_changes(x, y, opacity, rotation):
#     ctx = callback_context
#     if not ctx.triggered:
#         return no_update

#     # Transform the complex array to the specified format
#     transformed_array = []
#     indexes = set()

#     complex_array = ctx.inputs_list

#     # First, gather all unique indexes to create a template for the dictionaries
#     for group in complex_array:
#         for item in group:
#             indexes.add(item["id"]["index"])

#     # Initialize dictionaries for each index
#     for index in indexes:
#         transformed_array.append({"index": index})

#     # Populate the dictionaries with values from the complex array
#     for group in complex_array:
#         for item in group:
#             index = item["id"]["index"]
#             type_ = item["id"]["type"]
#             value = item["value"]
#             # Find the dictionary with the matching index and update it with the new value
#             for dict_ in transformed_array:
#                 if dict_["index"] == index:
#                     dict_[type_] = value

#     ## This controls whether the text version for transformed array is displayed on the screen
#     # print(transformed_array)
#     print(transformed_array)
#     return no_update, transformed_array

### This populates the image selection table when the imageSelect dropwdown is changed

# @callback(
#     Output("annotationTable", "rowData"),
#     Input("imageSelect", "value"),
# )
# def populate_dsa_annotation_table(imageSelect):
#     imgTileSources = tileSourceDict[imageSelect]
#   "tileSrcIdx": idx,
#                 # "compositeOperation": "screen",
# Valid values are 'source-over', 'source-atop', 'source-in', 'source-out', 'destination-over', 'destination-atop', 'destination-in', 'destination-out', 'lighter', 'difference', 'copy', 'xor', etc. For complete list of modes, please
## ADD FLIPPED OPTION!!!

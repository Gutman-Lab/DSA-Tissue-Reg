## This should accept a folderId and display all of the  associated images in both a table or dataview
from dash import html, Input, Output, State, callback, dcc, no_update
from settings import gc, memory, DSA_BASE_URL, token_info
import dash_bootstrap_components as dbc
import dash_ag_grid
import dash_paperdragon
import json
from pprint import pprint


sampleCaseFolder = "641bfd45867536bb7a236ae1"

caseList = [
    {"label": "E20-11", "value": "641bfd45867536bb7a236ae1"},
    {"label": "E20-106", "value": "641bfdd9867536bb7a236c3d"},
]


caseImage_Viewer = dash_paperdragon.DashPaperdragon(
    id="currentSlide_osdViewer",
    viewerHeight=200,
    viewportBounds={"x": 0, "y": 0, "width": 0, "height": 0},
)


caseSelect_dropdown = html.Div(
    [
        html.Label("Select an case", className="text-center"),
        dbc.Select(
            id="caseSelect",
            options=caseList,
            value=caseList[0]["value"],
            className="mb-4 d-inline",
            style={"width": "300px", "marginLleft": "10px", "marginTop": "1px"},
        ),
    ]
)


adrcCaseTable = dash_ag_grid.AgGrid(
    id="currentCaseTable",
    columnDefs=[
        {"headerName": "Name", "field": "name"},
        {"headerName": "Size", "field": "size"},
        {"headerName": "Case ID", "field": "meta.npSchema.caseID"},
        {"headerName": "Region Name", "field": "meta.npSchema.regionName"},
        {"headerName": "Stain", "field": "meta.npSchema.stainID"},
        {"headerName": "Block ID", "field": "meta.npSchema.blockID"},
        {"headerName": "Annotation Count", "field": "annotationCount"},
    ],
    defaultColDef={
        "flex": 1,
        "minWidth": 100,
        "filterParams": {"debounceMs": 2500},
        "floatingFilter": True,
        "sortable": True,
        "resizable": True,
    },
    dashGridOptions={
        "pagination": True,
        "paginationAutoPageSize": True,
        "rowSelection": "single",
    },
    className="ag-theme-alpine compact",
    style={"height": "300px"},
)


annotationTable = dash_ag_grid.AgGrid(
    id="currentCaseAnnotationTable",
    columnDefs=[
        {"headerName": "annotationId", "field": "_id"},
        {"headerName": "Name", "field": "annotation.name"},
        {"headerName": "groups", "field": "groups"},
    ],
    defaultColDef={
        "flex": 1,
        "minWidth": 100,
        "filterParams": {"debounceMs": 2500},
        # "floatingFilter": True,
        "sortable": True,
        "resizable": True,
    },
    dashGridOptions={
        "pagination": True,
        "paginationAutoPageSize": True,
        "rowSelection": "multiple",
    },
    className="ag-theme-alpine compact",
    style={"height": "300px"},
)


annotation_control = html.Div(id="annotation_control", children=[annotationTable])
caseViewStats = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Card(
                    [
                        dbc.CardHeader("Total Slides", className="text-center"),
                        dbc.CardBody(id="totalSlides", className="text-center"),
                    ]
                )
            ],
            width=1,
            className="m-2",
        ),
        dbc.Col(caseImage_Viewer, width=7),
        dbc.Col(annotation_control, width=3),
    ]
)


caseViewer_layout = dbc.Container(
    [
        caseSelect_dropdown,
        dcc.Store(id="caseRootFolderId_store", data=sampleCaseFolder),
        dcc.Store(id="caseSlideSet_store", data=[]),
        dbc.Row([adrcCaseTable]),
        caseViewStats,
    ],
    fluid=True,
)


### Populate caseSlideSet based on the currently slided caseID
@callback(Output("caseSlideSet_store", "data"), [Input("caseSelect", "value")])
@memory.cache  ## Remove this when you start doing any updates..
def populate_caseSlideSet(caseFolderId):
    ## May want to add schema validation here in the future
    slideList = list(gc.listItem(caseFolderId))

    annotationCountStr = (
        f"annotation/counts?items={','.join([x['_id'] for x in slideList])}"
    )
    annotationCounts = gc.get(annotationCountStr)
    for sl in slideList:
        print(sl)
        sl["annotationCount"] = annotationCounts[sl["_id"]]
    return slideList


## Generate a datable of the slides from the case
@callback(Output("currentCaseTable", "rowData"), [Input("caseSlideSet_store", "data")])
def update_caseTable(slideList):

    return slideList


## Update/generate stats based on the current case Data

@callback(Output("totalSlides", "children"), [Input("caseSlideSet_store", "data")])
def update_caseStats(slideList):
    return len(slideList)


@callback(
    Output("currentSlide_osdViewer", "tileSources"),
    Input("currentCaseTable", "selectedRows"),
)
def updateCaseOsdViewer(selectedRows):
    if selectedRows:
        # print(selectedRows)
        slideId = selectedRows[0]["_id"]
        return [
            {
                "tileSource": f"{DSA_BASE_URL}/item/{slideId}/tiles/dzi.dzi?token={token_info['_id']}",
                "width": 100000,
            }
        ]
    return no_update


## update annotations available for current image
@callback(
    Output("currentCaseAnnotationTable", "rowData"),
    Input("currentCaseTable", "selectedRows"),
)
def updateCaseOsdViewer(selectedRows):
    if selectedRows:
        # print(selectedRows)
        slideId = selectedRows[0]["_id"]
        itemAnnotations = gc.get(
            f"annotation?itemId={slideId}&token={token_info['_id']}"
        )

        return itemAnnotations
    return []


## update annotations available for current image
@callback(
    Output("currentSlide_osdViewer", "inputToPaper"),
    Input("currentCaseAnnotationTable", "selectedRows"),
)
def displayImageAnnotatoins(selectedRows):
    if selectedRows:
        # print(selectedRows)

        annotationId = selectedRows[0]["_id"]
        annotationData = gc.get(
            f"annotation/{annotationId}/geojson?token={token_info['_id']}"
        )
        # print(annotationData)
        paperJS_annotations = geojson_to_paperjs(annotationData)
        if not paperJS_annotations:
            return no_update
        # pprint(paperJS_annotations)
        # slideId = selectedRows[0]["_id"]
        # itemAnnotations = gc.get(
        #     f"annotation?itemId={slideId}&token={token_info['_id']}"
        # )

        return {
            "actions": [
                {"type": "clearItems"},
                {"type": "drawItems", "itemList": paperJS_annotations},
            ]
        }

    return {"actions": [{"type": "clearItems"}]}


def geojson_to_paperjs(geojson):
    # Assuming the GeoJSON object is a Polygon
    try:
        coordinates = geojson["features"][0]["geometry"]["coordinates"][0]
        # ]  # Get the coordinates of the Polygon
        # # print(coordinates)

        # # Convert the coordinates to a Paper.js path
        paperjs_path = {
            "paperType": "Path",
            "args": [
                {
                    "fillColor": "red",
                    "fillOpacity": 0.3,
                    "rescale": {"strokeWidth": 1},
                    "segments": [{"x": int(x), "y": int(y)} for x, y, z in coordinates],
                    "closed": True,
                    "strokeColor": "red",
                    # Add other properties as needed...
                }
            ],
            "userdata": {"class": "a", "objectId": 40},
        }

        # pprint(paperjs_path)
    except:
        print("Geojson didn't parse?")
        return None

    return [paperjs_path]


# def generate_paperjs_polygon(shapeInfo):
#     jsPolygon = [
#         {
#             "paperType": "Path",
#             "args": [
#                 {
#                     "fillColor": "red",
#                     "strokeColor": "red",
#                     "rescale": {"strokeWidth": 1},
#                     "fillOpacity": 0.2,
#                     "segments": [
#                         {"x": 7849, "y": 19637},
#                         {"x": 8823, "y": 20637},
#                         {"x": 7849, "y": 21637},
#                     ],
#                     "closed": True,
#                 }
#             ],
#             "userdata": {"class": "a", "objectId": 40},
#         },
#         # Other shapes...
#     ]

#     return jsPolygon


## Get associated annotations for selected case..


## Populate the viewer base


# ## Populate caseSelect dropdown with case names
# @callback(
#     Output("caseSelect", "options"),
#     [Input("caseRootFolderId", "data")],
# )
# def populate_caseSelect(caseRootFolderId):
#     if caseRootFolderId:
#         # Get the list of case names
#         caseList = gc.listItem(caseRootFolderId)
#         return [{"label": x["name"], "value": x["_id"]} for x in caseList]


# @callback(Output("caseSelect", "options"), [Input("caseSelect", "value")])

## https://megabrain.neurology.emory.edu/api/v1/annotation/counts
# ion/counts?items=1%2C2%2C3%2C4%2C5' \

# def showSelectedImage(selectedRow, expData):
#     Output("dendraFixed_osdViewer", "tileSources"),
#     Input("dendraExperimentTable", "selectedRows"),
#     State("dendraExperimentTable", "rowData"),
# )
# def showSelectedImage(selectedRow, expData):
#     if selectedRow:
#         imageId = selectedRow[0]["_id"]
#         print(selectedRow)
#         exp_df = pd.DataFrame(expData)
#         regData = getImageRegPair(exp_df, imageId)
#         print(regData)

#         tileSrcSpec = [
#             f"http://glasslab.neurology.emory.edu:8080/api/v1/item/{regData['fixedImage_id']}/tiles/dzi.dzi",
#             f"http://glasslab.neurology.emory.edu:8080/api/v1/item/{regData['movingImage_id']}/tiles/dzi.dzi",
#         ]
#         return tileSrcSpec
#     # , [
#     #         {"opacity": 0.5, "x": 1, "y": 1},
#     #         {"opacity": 0.2, "x": 0, "y": 0},
#     #     ]
#     else:
#         return no_update
#     # geojson[
#     #     "properties"
#     # ],  # Use the properties of the GeoJSON object as userdata
# }
# pprint(paperjs_path)
# jsPolygon = [
#     {
#         "paperType": "Path",
#         "args": [
#             {
#                 "fillColor": "red",
#                 "strokeColor": "red",
#                 "rescale": {"strokeWidth": 1},
#                 "fillOpacity": 0.3,
#                 "segments": [
#                     {"x": 7849, "y": 19637},
#                     {"x": 8823, "y": 28637},
#                     {"x": 7849, "y": 21637},
#                 ],
#                 "closed": True,
#             }
#         ],
#         "userdata": {"class": "a", "objectId": 40},
#     },
#     # Other shapes...
# ]

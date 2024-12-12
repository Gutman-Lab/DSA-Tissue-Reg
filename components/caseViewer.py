## This should accept a folderId and display all of the  associated images in both a table or dataview
from dash import html, Input, Output, State, callback, dcc, no_update
from settings import gc, memory, DSA_BASE_URL, token_info
import dash_bootstrap_components as dbc
import dash_ag_grid
import dash_paperdragon
from pprint import pprint


sampleCaseFolder = "641bfd45867536bb7a236ae1"

## Good exaples to start with..
caseList = [
    {"label": "E20-11", "value": "641bfd45867536bb7a236ae1"},
    {"label": "E20-106", "value": "641bfdd9867536bb7a236c3d"},
]


caseImage_Viewer = dash_paperdragon.DashPaperdragon(
    id="currentSlide_osdViewer",
    viewerHeight=200,
    viewportBounds={"x": 0, "y": 0, "width": 0, "height": 0},
)


filter_controls = dbc.Row(
    [
        dbc.Col(
            html.Div(
                [
                    html.Label(
                        "Select a case ",
                        className="text-center mr-2",
                        style={"marginRight": "15px", "marginTop": "1px"},
                    ),
                    dbc.Select(
                        id="caseSelect",
                        options=caseList,
                        value=caseList[0]["value"],
                        className="d-inline",
                        style={
                            "width": "200px",
                            "marginLeft": "15px",
                            "marginTop": "1px",
                        },
                    ),
                ],
                className="d-flex align-items-center",
            ),
            width="auto",
        ),
        dbc.Col(
            html.Div(
                [
                    html.Label(
                        "Filter by Block ID: ",
                        className="text-center mr-2",
                        style={"marginRight": "15px"},
                    ),
                    dbc.Select(
                        id="blockID_filter_select",
                        options=[{"label": "All Blocks", "value": "all"}],
                        value="all",
                        style={"width": "200px", "marginLeft": "15px"},
                    ),
                ],
                className="d-flex align-items-center",
            ),
            width="auto",
        ),
        dbc.Col(
            dbc.Checkbox(
                id="show_annotated_only",
                label="Show only annotated slides",
                value=False,
                className="ml-3",
            ),
            width="auto",
        ),
    ],
    className="mb-3 align-items-center",
)


adrcCaseTable = dash_ag_grid.AgGrid(
    id="currentCaseTable",
    columnDefs=[
        {
            "headerName": "Case ID",
            "field": "meta.npSchema.caseID",
            "hide": True,
        },  ## Not needed given filters
        {"headerName": "Region Name", "field": "meta.npSchema.regionName"},
        {"headerName": "Stain", "field": "meta.npSchema.stainID"},
        {
            "headerName": "Block ID",
            "field": "meta.npSchema.blockID",
            "filter": "agSetColumnFilter",
            "filterParams": {"values": [], "suppressSorting": True},
        },
        {"headerName": "Annotation Count", "field": "annotationCount"},
        {"headerName": "Name", "field": "name"},
        {"headerName": "Size", "field": "size", "hide": True},
    ],
    defaultColDef={
        "flex": 1,
        "minWidth": 80,
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


caseSlideCount = dbc.Card(
    [
        dbc.CardHeader("Total Slides", className="text-center"),
        dbc.CardBody(id="totalSlides", className="text-center"),
    ]
)


annotation_control = html.Div(id="annotation_control", children=[annotationTable])


# Add new thumbnail card component
def create_thumbnail_card(item):
    return dbc.Card(
        [
            dbc.CardHeader(
                item.get("meta", {})
                .get("npSchema", {})
                .get("stainID", "Unknown Stain"),
                className="text-center",
            ),
            dbc.CardImg(
                src=f"{DSA_BASE_URL}/item/{item['_id']}/tiles/thumbnail?token={token_info['_id']}",
                top=True,
                style={"height": "150px", "objectFit": "contain"},
            ),
        ],
        className="mb-3",
        style={"width": "200px"},
    )


# Add thumbnail grid container
thumbnail_grid = html.Div(id="thumbnail-grid", className="d-flex flex-wrap gap-3 mt-3")

# Update the layout to include the thumbnail grid
caseViewer_layout = dbc.Container(
    [
        filter_controls,
        dcc.Store(id="caseRootFolderId_store", data=sampleCaseFolder),
        dcc.Store(id="caseSlideSet_store", data=[]),
        dbc.Row(
            [
                dbc.Col(adrcCaseTable, width=6),
                dbc.Col(annotationTable, width=6),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(caseImage_Viewer, width=6),
                dbc.Col(thumbnail_grid, width=6),
            ]
        ),
    ],
    fluid=True,
)


# Add callback to populate the filter options
@callback(
    Output("blockID_filter_select", "options"),
    [Input("caseSlideSet_store", "data"), Input("show_annotated_only", "value")],
)
def update_blockID_filter_options(slideList, show_annotated_only):
    if not slideList:
        return [{"label": "All Blocks", "value": "all"}]

    # First filter slides by annotation count if needed
    if show_annotated_only:
        slideList = [
            slide for slide in slideList if slide.get("annotationCount", 0) > 0
        ]

    # Extract unique blockIDs from the filtered data
    unique_blockIDs = list(
        set(
            slide.get("meta", {}).get("npSchema", {}).get("blockID", "")
            for slide in slideList
        )
    )
    unique_blockIDs = [bid for bid in unique_blockIDs if bid]  # Remove empty values

    # Create options list with "All Blocks" as first option
    options = [{"label": "All Blocks", "value": "all"}]
    options.extend([{"label": bid, "value": bid} for bid in sorted(unique_blockIDs)])

    return options


# Add callback to apply the external filter
@callback(
    Output("currentCaseTable", "dashGridOptions"),
    Input("blockID_filter_select", "value"),
    prevent_initial_call=True,
)
def update_external_filter(selected_block):
    if selected_block == "all":
        return {
            "isExternalFilterPresent": {"function": "false"},
            "doesExternalFilterPass": {"function": "true"},
        }
    else:
        filter_function = f"params.data.meta.npSchema.blockID === '{selected_block}'"
        return {
            "isExternalFilterPresent": {"function": "true"},
            "doesExternalFilterPass": {"function": filter_function},
        }


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
@callback(
    [Output("currentCaseTable", "rowData"), Output("currentCaseTable", "selectedRows")],
    [Input("caseSlideSet_store", "data")],
)
def update_caseTable(slideList):
    if not slideList:
        return [], []

    # Return both the rowData and the first row as selected
    return slideList, [slideList[0]] if slideList else []


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
    [
        Output("currentCaseAnnotationTable", "rowData"),
        Output("currentCaseAnnotationTable", "selectedRows"),
    ],
    Input("currentCaseTable", "selectedRows"),
)
def updateCaseAnnotations(selectedRows):
    if selectedRows:
        slideId = selectedRows[0]["_id"]
        itemAnnotations = gc.get(
            f"annotation?itemId={slideId}&token={token_info['_id']}"
        )

        # Return both the rowData and the first row as selected
        return itemAnnotations, [itemAnnotations[0]] if itemAnnotations else []

    return [], []


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
            f"annotation/{annotationId}/geojson"  ## /geojson?token={token_info['_id']}"
        )
        # from utils import format_ann_docs_for_paperjs
        # from dsahelpers.dash import format_ann_docs_for_paperjs
        from dsa_helpers.dash.dash_paperdragon_utils import (
            get_input_to_paper_dict,
            format_ann_docs_for_paperjs,
        )

        paperJS_annotations = format_ann_docs_for_paperjs([annotationData])
        if not paperJS_annotations:
            return no_update

        return get_input_to_paper_dict(paperJS_annotations)

    return {"actions": [{"type": "clearItems"}]}


@callback(Output("currentCaseTable", "columnDefs"), Input("caseSlideSet_store", "data"))
def update_blockID_filter(slideList):
    if not slideList:
        return no_update

    # Extract unique blockIDs from the data
    unique_blockIDs = list(
        set(
            slide.get("meta", {}).get("npSchema", {}).get("blockID", "")
            for slide in slideList
        )
    )
    unique_blockIDs = [bid for bid in unique_blockIDs if bid]  # Remove empty values

    # Update the columnDefs with new filter values
    columnDefs = [
        {"headerName": "Name", "field": "name"},
        {"headerName": "Size", "field": "size"},
        {"headerName": "Case ID", "field": "meta.npSchema.caseID"},
        {"headerName": "Region Name", "field": "meta.npSchema.regionName"},
        {"headerName": "Stain", "field": "meta.npSchema.stainID"},
        {
            "headerName": "Block ID",
            "field": "meta.npSchema.blockID",
            "filter": "agSetColumnFilter",
            "filterParams": {"values": unique_blockIDs, "suppressSorting": True},
        },
        {"headerName": "Annotation Count", "field": "annotationCount"},
    ]

    return columnDefs


# Add callback to update thumbnails based on case and block selection
@callback(
    Output("thumbnail-grid", "children"),
    [Input("caseSlideSet_store", "data"), Input("blockID_filter_select", "value")],
)
def update_thumbnails(slideList, selected_block):
    if not slideList or selected_block == "all":
        return []  # Return empty when no block is selected or "all" is selected

    # Filter slides by block ID
    filtered_slides = [
        slide
        for slide in slideList
        if slide.get("meta", {}).get("npSchema", {}).get("blockID") == selected_block
    ]

    # Create thumbnail cards for each slide
    thumbnail_cards = [create_thumbnail_card(slide) for slide in filtered_slides]

    return thumbnail_cards

from dash import html, dcc
import dash_bootstrap_components as dbc

import dash_paperdragon

osdConfig = config = {
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
}


# Add viewer components for fixed (HE) and moving images
fixed_image_viewer = dash_paperdragon.DashPaperdragon(
    id="fixed-image-viewer",
    viewerHeight=400,
    viewerWidth=400,
    viewportBounds={"x": 0, "y": 0, "width": 0, "height": 0},
    config=osdConfig,
)

moving_image_viewer = dash_paperdragon.DashPaperdragon(
    id="moving-image-viewer",
    viewerHeight=400,
    viewerWidth=400,
    viewportBounds={"x": 0, "y": 0, "width": 0, "height": 0},
    config=osdConfig,
)


merged_image_viewer = dash_paperdragon.DashPaperdragon(
    id="merged-image-viewer",
    viewerHeight=400,
    viewerWidth=400,
    viewportBounds={"x": 0, "y": 0, "width": 0, "height": 0},
    config=osdConfig,
)


osdViewer_layout = dcc.Loading(
    id="viewers-loading",
    type="circle",
    children=[
        dbc.Row(
            [
                # Fixed viewer column
                dbc.Col(
                    [
                        html.Div(
                            id="fixed-image-info",
                            className="image-info mb-2",
                        ),
                        html.Div(
                            className="viewer-container",
                            children=[
                                html.Div(
                                    "FIXED",
                                    className="viewer-label",
                                ),
                                fixed_image_viewer,
                            ],
                        ),
                    ],
                    width=4,
                ),
                # Moving viewer column
                dbc.Col(
                    [
                        html.Div(
                            id="moving-image-info",
                            className="image-info mb-2",
                        ),
                        html.Div(
                            className="viewer-container",
                            children=[
                                html.Div(
                                    "MOVING",
                                    className="viewer-label",
                                ),
                                moving_image_viewer,
                            ],
                        ),
                    ],
                    width=4,
                ),
                # Merged viewer column
                dbc.Col(
                    [
                        html.Div(
                            id="merged-image-info",
                            className="image-info mb-2",
                        ),
                        html.Div(
                            className="viewer-container",
                            children=[
                                html.Div(
                                    "MERGED",
                                    className="viewer-label",
                                ),
                                merged_image_viewer,
                            ],
                        ),
                    ],
                    width=4,
                ),
            ],
            className="g-2",
        ),
    ],
)

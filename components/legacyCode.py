# @callback(
#     Output("caseSetViewer", "children"),
#     Input("selectedRegionData", "data"),
# )
# def update_SelectedRegionSetViewer(slideList):
#     # print(slideList)
#     if slideList:
#         return html.Div(children=[len(slideList), [x["name"] for x in slideList]])
#     else:
#         return html.Div(children=["No slides selected"])

#     # Get slides
#     fixed_slide_id, moving_slide = get_slides_for_registration(
#         slideList, selected_block
#     )
# #     print(he_slide, "-----HE-----", moving_slide, "HE and moving slide...")
#     if not selected_slide_id:
#         selected_slide_id = moving_slide["_id"]

#     available_stains = get_slide_stain_info(filtered_slides)

#     # Create thumbnail cards for each slide
#     thumbnail_cards = [
#         create_thumbnail_card(slide, selected=(slide.get("_id") == selected_slide_id))
#         for slide in filtered_slides
#     ]

#     if "HE" in available_stains:
#         fixed_slide_id = available_stains["HE"]

#     # he_tiles_info = gc.get(f"item/{fixed_slide_id}/tiles")
#     # # print(he_tiles_info)

#     # if selected_slide_id:
#     #     moving_tiles_info = gc.get(f"item/{selected_slide_id}/tiles")

#     return thumbnail_cards, fixed_slide_id, selected_slide_id


# ## The selected region contains all the cases from the same Block ID... this update should trigger when the blockID is changed


###


# dcc.Store(id="selectedRegionData", data=None),

# # Define merged image controls with the modal button
# merged_image_controls = dbc.Row(
#     [
#         dbc.Col(
#             [
#                 html.Label("Opacity:", className="mb-1 small"),
#                 dcc.Slider(
#                     id="moving-image-opacity",
#                     min=0,
#                     max=1,
#                     step=0.1,
#                     value=0.5,
#                     marks=None,
#                     tooltip={"placement": "bottom", "always_visible": True},
#                     className="mt-1 narrow-slider",
#                 ),
#             ],
#             width="auto",
#             style={"width": "150px"},
#         ),
#         dbc.Col(
#             [
#                 html.Label("Step:", className="mb-1 small"),
#                 dbc.Select(
#                     id="offset-step-size",
#                     options=[
#                         {"label": "±1", "value": 1},
#                         {"label": "±10", "value": 10},
#                         {"label": "±100", "value": 100},
#                     ],
#                     value=1,
#                     size="sm",
#                 ),
#             ],
#             width="auto",
#             style={"width": "80px"},
#         ),
#         dbc.Col(
#             [
#                 html.Label("X:", className="mb-1 small"),
#                 dcc.Input(
#                     id="moving-image-x-offset",
#                     type="number",
#                     value=0,
#                     step="any",
#                     className="form-control form-control-sm",
#                     style={"width": "80px"},
#                     inputMode="numeric",
#                     pattern="[0-9]*",
#                 ),
#             ],
#             width="auto",
#         ),
#         dbc.Col(
#             [
#                 html.Label("Y:", className="mb-1 small"),
#                 dcc.Input(
#                     id="moving-image-y-offset",
#                     type="number",
#                     value=0,
#                     step="any",
#                     className="form-control form-control-sm",
#                     style={"width": "80px"},
#                     inputMode="numeric",
#                     pattern="[0-9]*",
#                 ),
#             ],
#             width="auto",
#         ),
#         dbc.Col(
#             [
#                 html.Label("Rot:", className="mb-1 small"),
#                 dcc.Input(
#                     id="moving-image-rotation",
#                     type="number",
#                     value=0,
#                     className="form-control form-control-sm",
#                     style={"width": "80px"},
#                     inputMode="numeric",
#                     pattern="[0-9]*",
#                 ),
#             ],
#             width="auto",
#         ),
#         # Add the modal toggle button
#         dbc.Col(
#             [
#                 html.Label(
#                     "\u00A0", className="mb-1 small d-block"
#                 ),  # Non-breaking space for alignment
#                 dbc.Button(
#                     "Points",
#                     id="open-registration-points-modal",
#                     color="secondary",
#                     size="sm",
#                     className="mt-1",
#                 ),
#             ],
#             width="auto",
#         ),
#     ],
#     className="g-2 align-items-end",
# )


# thumbnail_size_select = [
#     html.Label("Width:", className="mb-1 small"),
#     dbc.Select(
#         id="thumbnail-width-selector",
#         options=[
#             {
#                 "label": "256px",
#                 "value": 256,
#             },
#             {
#                 "label": "512px",
#                 "value": 512,
#             },
#             {
#                 "label": "1024px",
#                 "value": 1024,
#             },
#             {
#                 "label": "2048px",
#                 "value": 2048,
#             },
#         ],
#         value=1024,
#         size="sm",
#     ),
# ]


# # Basic layout for registration controls
# registrationControls_layout = html.Div(
#     [
#         # Add the store component
#         data_stores,
#         # Your existing layout components
#         dbc.Container(
#             [
#                 dbc.Row(
#                     [
#                         dbc.Col(
#                             [
#                                 # Static controls section
#                                 dbc.Row(
#                                     [
#                                         dbc.Col(
#                                             [
#                                                 html.Label(
#                                                     "Select Case:",
#                                                     className="mb-1 small",
#                                                 ),
#                                                 dbc.Select(
#                                                     id="registration_caseSelect",
#                                                     options=caseList,
#                                                     value=caseList[0]["value"],
#                                                     size="sm",
#                                                 ),
#                                             ],
#                                             width="auto",
#                                         ),
#                                         dbc.Col(
#                                             [
#                                                 html.Label(
#                                                     "Block ID:", className="mb-1 small"
#                                                 ),
#                                                 dbc.Select(
#                                                     id="registration_blockID_filter_select",
#                                                     options=[{}],
#                                                     size="sm",
#                                                 ),
#                                             ],
#                                             width="auto",
#                                         ),
#                                         dbc.Col(
#                                             dbc.Checkbox(
#                                                 id="registration_show_annotated_only",
#                                                 label="Only annotated",
#                                                 value=False,
#                                                 className="ml-3",
#                                             ),
#                                             width="auto",
#                                         ),
#                                         # Feature Detection Method - made narrower
#                                         dbc.Col(
#                                             [
#                                                 html.Label(
#                                                     "Method:", className="mb-1 small"
#                                                 ),
#                                                 dbc.Select(
#                                                     id="feature-detection-method",
#                                                     options=[
#                                                         {
#                                                             "label": "ORB (Fast)",
#                                                             "value": "orb",
#                                                         },
#                                                         {
#                                                             "label": "SIFT (Accurate)",
#                                                             "value": "sift",
#                                                         },
#                                                         {
#                                                             "label": "AKAZE",
#                                                             "value": "akaze",
#                                                         },
#                                                         {
#                                                             "label": "Mutual Information",
#                                                             "value": "intensity",
#                                                         },
#                                                     ],
#                                                     value="akaze",
#                                                     size="sm",
#                                                 ),
#                                             ],
#                                             width="auto",
#                                             style={
#                                                 "width": "160px"
#                                             },  # Specify exact width
#                                         ),
#                                         # Number of Points - made narrower
#                                         dbc.Col(
#                                             [
#                                                 html.Label(
#                                                     "Points:", className="mb-1 small"
#                                                 ),
#                                                 dbc.Select(
#                                                     id="num-points-selector",
#                                                     options=[
#                                                         {"label": "6 pts", "value": 6},
#                                                         {"label": "8 pts", "value": 8},
#                                                         {
#                                                             "label": "12 pts",
#                                                             "value": 12,
#                                                         },
#                                                         {
#                                                             "label": "16 pts",
#                                                             "value": 16,
#                                                         },
#                                                     ],
#                                                     value=8,
#                                                     size="sm",
#                                                 ),
#                                             ],
#                                             width="auto",
#                                             style={
#                                                 "width": "90px"
#                                             },  # Specify exact width
#                                         ),
#                                         # Thumbnail Width
#                                         dbc.Col(
#                                             thumbnail_size_select,
#                                             width="auto",
#                                             style={
#                                                 "width": "110px"
#                                             },  # Specify exact width
#                                         ),
#                                     ],
#                                     className="mb-2 g-2 align-items-end",
#                                 ),
#                                 # Thumbnails and merged controls in same row
#                                 dbc.Row(
#                                     [
#                                         # Thumbnail grid with horizontal scroll
#                                         dbc.Col(
#                                             dcc.Loading(
#                                                 id="registration-loading",
#                                                 type="circle",
#                                                 children=[
#                                                     html.Div(
#                                                         id="registration-thumbnail-grid",
#                                                         className="d-flex flex-nowrap gap-2",
#                                                         style={
#                                                             "overflowX": "auto",
#                                                             "whiteSpace": "nowrap",
#                                                             "paddingBottom": "10px",
#                                                             "maxHeight": "200px",
#                                                         },
#                                                     ),
#                                                 ],
#                                             ),
#                                             width=7,
#                                             style={
#                                                 "minWidth": 0,
#                                             },
#                                         ),
#                                         # Merged image controls
#                                         dbc.Col(
#                                             merged_image_controls,
#                                             width=5,
#                                             className="align-self-center",
#                                         ),
#                                     ],
#                                     className="mb-3",
#                                 ),
#                                 # Viewers section
#                                 osdViewer_layout,
#                             ],
#                             className="mb-2",
#                         ),
#                     ],
#                 ),
#                 # Add the modal here
#                 registration_points_modal,  # This is the key addition
#             ],
#             fluid=True,
#             className="px-2",
#         ),
#     ]
# )


# @callback(
#     Output("fixed-image-info", "children"),
#     Output("fixed-image-viewer", "tileSources"),
#     Input("fixed_image_id", "data"),
# )
# def update_fixedImageData(fixed_image_id):
#     ## This updates the metadata, and also the fixed image viewer tile source
#     ## Trying to tie these callbacks to a single variable change to make it more manageable
#     if fixed_image_id:

#         fixed_tile_info = gc.get(f"item/{fixed_image_id}/tiles")
#         return (
#             [
#                 html.Div(
#                     f"Size: {fixed_tile_info.get('sizeX', 'N/A')}×{fixed_tile_info.get('sizeY', 'N/A')}  |  "
#                     f"Resolution: {fixed_tile_info.get('mm_x', 'N/A')}  |  "
#                     f"Magnification: {fixed_tile_info.get('magnification', 'N/A')}"
#                 )
#             ],
#             [
#                 {
#                     "tileSource": f"{DSA_BASE_URL}/item/{fixed_image_id}/tiles/dzi.dzi?token={token_info['_id']}",
#                     "width": fixed_tile_info.get("width", 100000),
#                 }
#             ],
#         )


# # ## This will actually update the tileSourceprops / location for the moving image
# # @callback(

# #         Output("moving-image-x-offset", "value"),
# #         Output("moving-image-y-offset", "value"),
# #         Output("moving-image-rotation", "value"),
# #         Input()


# # Callback to populate thumbnails using the shared caseSlideSet_store
# @callback(
#     Output("registration-thumbnail-grid", "children"),
#     Output("fixed_image_id", "data"),
#     Output("moving_image_id", "data"),
#     [
#         Input("registration_caseSlideSet_store", "data"),
#         Input("registration_blockID_filter_select", "value"),
#         Input("selected-moving-slide", "data"),
#     ],
# )
# def update_registration_thumbnails(slideList, selected_block, selected_slide_id):
#     """Single callback to handle all thumbnail grid updates"""
#     if not slideList or not selected_block:
#         return [], {}, {}, no_update

#     """Extract slide ID and stain ID from slides if available"""

#     def get_slide_stain_info(slides):
#         stain_slideId_dict = {}

#         for slide in slides:
#             stain_id = slide.get("meta", {}).get("npSchema", {}).get("stainID")
#             if stain_id:  # Only return if stainID exists
#                 stain_slideId_dict[stain_id] = slide["_id"]

#         return stain_slideId_dict

#     # Filter slides by block ID
#     filtered_slides = [
#         slide
#         for slide in slideList
#         if slide.get("meta", {}).get("npSchema", {}).get("blockID") == selected_block
#     ]

#     # from pprint import pprint

#     # ### Get the stain and ID for the selected set

#     # Get slides
#     fixed_slide_id, moving_slide = get_slides_for_registration(
#         slideList, selected_block
#     )
#     print(he_slide, "-----HE-----", moving_slide, "HE and moving slide...")
#     if not selected_slide_id:
#         selected_slide_id = moving_slide["_id"]

#     available_stains = get_slide_stain_info(filtered_slides)

#     # Create thumbnail cards for each slide
#     thumbnail_cards = [
#         create_thumbnail_card(slide, selected=(slide.get("_id") == selected_slide_id))
#         for slide in filtered_slides
#     ]

#     if "HE" in available_stains:
#         fixed_slide_id = available_stains["HE"]

#     # he_tiles_info = gc.get(f"item/{fixed_slide_id}/tiles")
#     # # print(he_tiles_info)

#     # if selected_slide_id:
#     #     moving_tiles_info = gc.get(f"item/{selected_slide_id}/tiles")

#     return thumbnail_cards, fixed_slide_id, selected_slide_id


# ## CREATE A CALLBACK THAT ACTUALLY MOVES THE MOVING IMAGE TILE SOURCE..


# # ## THIS IS A NIGHTMARE CALLBACK...
# # # Simplified callbacks that use the imported functions
# # @callback(
# #     [
# #         # Output("fixed-image-viewer", "tileSources"),
# #         # Output("moving-image-viewer", "tileSources"),
# #         Output("moving-image-x-offset", "value"),
# #         Output("moving-image-y-offset", "value"),
# #         Output("moving-image-rotation", "value"),
# #         Output("moving-image-info", "children"),
# #         Output("moving-image-metadata", "data"),
# #         Output("fixedImage_fiducial_points", "data"),
# #         Output("movingImage_fiducial_points", "data"),
# #     ],
# #     [
# #         Input("registration_caseSlideSet_store", "data"),
# #         Input("registration_blockID_filter_select", "value"),
# #         Input("feature-detection-method", "value"),
# #         Input("num-points-selector", "value"),
# #         Input("thumbnail-width-selector", "value"),
# #     ],
# #     background=True,
# #     background_callback_manager=background_callback_manager,
# #     allow_duplicates=True,
# # )
# # def setup_registration_images(
# #     slideList,
# #     selected_block,
# #     detection_method,
# #     num_points,
# #     thumbnail_width,
# # ):
# #     # print(f"selected_block: {selected_block}")

# #     # Get slides
# #     he_slide, moving_slide = get_slides_for_registration(slideList, selected_block)
# #     if not he_slide or not moving_slide:
# #         empty_return = 0, 0, 0, [], {}, [], []
# #         return empty_return

# #     # Get image dimensions for both slides
# #     try:
# #         he_tiles_info = gc.get(f"item/{he_slide['_id']}/tiles")
# #         moving_tiles_info = gc.get(f"item/{moving_slide['_id']}/tiles")

# #         fixed_bounds = {
# #             "width": he_tiles_info.get("sizeX", 10000),
# #             "height": he_tiles_info.get("sizeY", 10000),
# #         }
# #         moving_bounds = {
# #             "width": moving_tiles_info.get("sizeX", 10000),
# #             "height": moving_tiles_info.get("sizeY", 10000),
# #         }

# #         # Create tile sources with correct dimensions
# #         fixed_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{he_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": fixed_bounds["width"],
# #             }
# #         ]
# #         moving_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{moving_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": moving_bounds["width"],
# #             }
# #         ]

# #         try:
# #             # Get thumbnails with specified width
# #             fixed_thumb = get_thumbnail_image(he_slide["_id"], width=thumbnail_width)
# #             moving_thumb = get_thumbnail_image(
# #                 moving_slide["_id"], width=thumbnail_width
# #             )

# #             if fixed_thumb is None or moving_thumb is None:
# #                 print("Failed to retrieve thumbnails")
# #                 return (
# #                     # fixed_tile_source,
# #                     # moving_tile_source,
# #                     {"actions": []},
# #                     0,  # Default X offset
# #                     0,  # Default Y offset
# #                     0,  # Default rotation
# #                     [],
# #                     {},
# #                     [],
# #                     [],
# #                     None,
# #                     None,
# #                 )

# #             # Normalize image sizes for feature detection
# #             norm_fixed, norm_moving, (fixed_scale, moving_scale) = (
# #                 normalize_image_sizes(fixed_thumb, moving_thumb)
# #             )

# #             # print(
# #             #     f"Normalized shapes - fixed: {norm_fixed.shape}, moving: {norm_moving.shape}"
# #             # )
# #             # print(f"Scale factors - fixed: {fixed_scale}, moving: {moving_scale}")

# #             # Generate registration points on normalized images
# #             fixed_points, moving_points, debug_info = create_registration_points(
# #                 norm_fixed,
# #                 norm_moving,
# #                 detection_method,
# #                 num_points,
# #                 fixed_bounds,
# #             )

# #             if fixed_points and moving_points:
# #                 # Print original points
# #                 # if fixed_scale != moving_scale:
# #                 # print("Original fixed points:", fixed_points)
# #                 # print("Original moving points:", moving_points)

# #                 # Scale fixed points back to original fixed image size
# #                 fixed_points = [
# #                     (x * fixed_scale["x"], y * fixed_scale["y"])
# #                     for x, y in fixed_points
# #                 ]
# #                 moving_points = [
# #                     (x * moving_scale["x"], y * moving_scale["y"])
# #                     for x, y in moving_points
# #                 ]

# #                 # print("After scaling to original size:")
# #                 # print("Fixed points:", fixed_points)
# #                 # print("Moving points:", moving_points)

# #                 # Scale to full image coordinates
# #                 fixed_points = scale_points_to_full_size(
# #                     fixed_points, fixed_thumb.shape, fixed_bounds
# #                 )
# #                 moving_points = scale_points_to_full_size(
# #                     moving_points, moving_thumb.shape, moving_bounds
# #                 )

# #                 # print("After scaling to full size:")
# #                 # print("Fixed points:", fixed_points)
# #                 # print("Moving points:", moving_points)

# #                 # Convert points to numpy arrays for transformation calculation
# #                 fixed_np = np.float32(fixed_points)
# #                 moving_np = np.float32(moving_points)

# #                 # Calculate transformation matrix
# #                 transform_matrix = cv2.estimateAffinePartial2D(moving_np, fixed_np)[0]

# #                 # Extract transformation parameters
# #                 scale = np.sqrt(
# #                     transform_matrix[0, 0] ** 2 + transform_matrix[0, 1] ** 2
# #                 )
# #                 rotation = np.degrees(
# #                     np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
# #                 )
# #                 x_offset = transform_matrix[0, 2]
# #                 y_offset = transform_matrix[1, 2]

# #                 # Generate colors and create GeoJSON features
# #                 colors = generate_distinct_colors(len(fixed_points))
# #                 fixed_items = create_geojson_features(fixed_points, colors, "fixed")
# #                 moving_items = create_geojson_features(
# #                     moving_points, colors, "moving", layer_idx=1
# #                 )

# #                 # print("=== Successful Feature Matching Return ===")
# #                 # print("Fixed Items:", fixed_items)
# #                 # print("Number of items:", len(fixed_items))
# #                 # print(
# #                 #     "First item example:", fixed_items[0] if fixed_items else "No items"
# #                 # )
# #                 # print(
# #                 #     "Paper Input Structure:",
# #                 #     {"actions": [{"type": "drawItems", "itemList": fixed_items}]},
# #                 # )
# #                 # print("=====================================")
# #                 # print(fixed_points, moving_points, "fixed_points, moving_points")
# #                 return (
# #                     # fixed_tile_source,
# #                     # {"actions": [{"type": "drawItems", "itemList": fixed_items}]},
# #                     # moving_tile_source,
# #                     x_offset,  # X offset value
# #                     y_offset,  # Y offset value
# #                     rotation,  # Rotation value
# #                     [
# #                         html.Div(
# #                             f"Size: {moving_tiles_info.get('sizeX', 'N/A')}×{moving_tiles_info.get('sizeY', 'N/A')}  |  "
# #                             f"Resolution: {he_tiles_info.get('mm_x', 'N/A')}  |  "
# #                             f"Magnification: {moving_tiles_info.get('magnification', 'N/A')}"
# #                         )
# #                     ],
# #                     {
# #                         "sizeX": moving_tiles_info.get("sizeX", 1.0),
# #                         "sizeY": moving_tiles_info.get("sizeY", 1.0),
# #                         f"Resolution: {he_tiles_info.get('mm_x', 'N/A')}  |  "
# #                         "magnification": moving_tiles_info.get("magnification", "N/A"),
# #                     },
# #                     fixed_points,
# #                     moving_points,
# #                 )

# #         except Exception as e:
# #             print(f"Error in feature matching: {str(e)}")
# #             import traceback

# #             traceback.print_exc()

# #         # Return defaults if registration fails
# #         return (
# #             # moving_tile_source,
# #             {"actions": []},
# #             0,  # Default X offset
# #             0,  # Default Y offset
# #             0,  # Default rotation
# #             [],
# #             {},
# #             [],
# #             [],
# #         )

# #     except Exception as e:
# #         print(f"Error getting tiles info: {str(e)}")
# #         fixed_bounds = {"width": 10000, "height": 10000}
# #         moving_bounds = fixed_bounds.copy()

# #         # Create tile sources with correct dimensions
# #         fixed_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{he_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": fixed_bounds["width"],
# #             }
# #         ]
# #         moving_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{moving_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": moving_bounds["width"],
# #             }
# #         ]

# #         # Return defaults if registration fails
# #         return (
# #             # fixed_tile_source,
# #             # moving_tile_source,
# #             {"actions": []},
# #             0,  # Default X offset
# #             0,  # Default Y offset
# #             0,  # Default rotation
# #             [],
# #             {},
# #             [],
# #             [],
# #             None,
# #             None,
# #         )


# ## The fixed and moving points are stored in separate stores and can be accessed independent
# ## of all the other operations.. this code simply draws the unadultered points on the original merged or fixed
# ## image space.
# ## MAY BE BETTER TO MERGE THESE T OMAKE SURE COLORS ARE ALWAYS CONSISTENT..


# @callback(
#     [
#         Output("merged-image-viewer", "tileSources"),
#         Output("merged-image-viewer", "inputToPaper"),
#         Output("merged-image-info", "children"),
#     ],
#     [
#         Input("fixed-image-viewer", "tileSources"),
#         Input("moving-image-viewer", "tileSources"),
#         Input("fixed-image-viewer", "inputToPaper"),
#         Input("moving-image-viewer", "inputToPaper"),
#         Input("moving-image-opacity", "value"),
#     ],
# )
# def update_merged_viewer(fixed_tiles, moving_tiles, fixed_paper, moving_paper, opacity):
#     """Combine fixed and moving images in the merged viewer"""
#     if not fixed_tiles or not moving_tiles:
#         return [], {}, ["No tile information available"]

#     try:
#         merged_sources = [
#             fixed_tiles[0],  # Fixed image as base layer
#             {
#                 **moving_tiles[0],  # Moving image with opacity
#                 "opacity": opacity,  # Use the slider value
#                 "compositeOperation": "source-over",  # This controls how images are blended
#             },
#         ]

#         # Combine the paper inputs (registration points) from both viewers
#         merged_paper = {"actions": []}
#         if fixed_paper and "actions" in fixed_paper:
#             merged_paper["actions"].extend(fixed_paper["actions"])
#         if moving_paper and "actions" in moving_paper:
#             merged_paper["actions"].extend(moving_paper["actions"])

#         ### This should contain the merged points for both the fixed and moving images
#         # print(merged_paper, "merged_paper items")

#         # Get info for the merged viewer
#         merged_info = [
#             html.Div(
#                 "Merged View (Moving image opacity: 0.5)",
#                 style={
#                     "whiteSpace": "nowrap",
#                     "overflow": "hidden",
#                     "textOverflow": "ellipsis",
#                 },
#             )
#         ]

#         return merged_sources, merged_paper, merged_info

#     except Exception as e:
#         print(f"Error in merged viewer: {str(e)}")
#         return [], {}, ["Error creating merged view"]


# @callback(
#     Output("merged-image-viewer", "tileSourceProps"),
#     [
#         Input("moving-image-opacity", "value"),
#         Input("moving-image-x-offset", "value"),
#         Input("moving-image-y-offset", "value"),
#         Input("moving-image-rotation", "value"),
#         Input("moving-image-metadata", "data"),
#     ],
# )
# def update_transform(opacity, x_offset, y_offset, rotation, metadata):
#     """Update the moving image transformation based on control values"""
#     try:
#         opacity = float(opacity) if opacity is not None else 1.0
#         x_offset = float(x_offset) if x_offset is not None else 0
#         y_offset = float(y_offset) if y_offset is not None else 0
#         rotation = float(rotation) if rotation is not None else 0

#         # Get scale factor from stored metadata
#         scale_factor = metadata.get("sizeX", 1.0)

#         props = [
#             {
#                 "opacity": opacity,
#                 "x": x_offset,
#                 "y": y_offset,
#                 "rotation": rotation,
#                 "flipped": False,
#                 "scaleFactor": scale_factor,
#                 "index": 0,
#             }
#         ]

#         # print(
#         #     f"Updating transform - Opacity: {opacity}, Scale: {scale_factor}, Rotation: {rotation}, X: {x_offset}, Y: {y_offset}"
#         # )
#         return props

#     except Exception as e:
#         print(f"Error in transform update: {str(e)}")
#         return [
#             {
#                 "opacity": 1.0,
#                 "x": 0,
#                 "y": 0,
#                 "rotation": 0,
#                 "flipped": False,
#                 "scaleFactor": 1.0,
#                 "index": 0,
#             }
#         ]


# clientside_callback(
#     """
#     function(stepSize) {
#         // Update the step attribute of x and y offset inputs
#         const xOffset = document.getElementById('moving-image-x-offset');
#         const yOffset = document.getElementById('moving-image-y-offset');
#         if (xOffset && yOffset) {
#             xOffset.step = stepSize;
#             yOffset.step = stepSize;
#         }
#         return [null, null  ];
#     }
#     """,
#     [Output("moving-image-x-offset", "step"), Output("moving-image-y-offset", "step")],
#     [Input("offset-step-size", "value")],
# )


# @callback(
#     [
#         Output("selected-moving-slide", "data"),
#         Output("registration-loading", "children", allow_duplicate=True),
#     ],
#     Input({"type": "thumbnail-card", "index": ALL, "stain": ALL}, "n_clicks"),
#     State({"type": "thumbnail-card", "index": ALL, "stain": ALL}, "id"),
#     State("selected-moving-slide", "data"),
#     prevent_initial_call=True,
# )
# def handle_thumbnail_click(n_clicks, ids, selected_moving_slide):
#     """Handle clicks on thumbnail card headers"""
#     ## This will also select the moving-slide if the value is not already set...

#     if not ctx.triggered_id:
#         return no_update, no_update

#     # Get the clicked card's ID and stain
#     clicked_id = ctx.triggered_id["index"]
#     clicked_stain = ctx.triggered_id["stain"]

#     def get_first_non_HE_slide(slides_list):
#         """Return the first slide index that isn't HE"""
#         for slide in slides_list:
#             if slide["stain"] != "HE":
#                 return slide["index"]
#         return None  # Return None if no non-HE slides found

#     if not selected_moving_slide:
#         selected_moving_slide = get_first_non_HE_slide(ids)

#     # Only update selection if it's not an HE slide
#     if clicked_stain.upper() != "HE":
#         print(f"Selected moving slide: {clicked_id}")
#         return clicked_id, no_update

#     return no_update, selected_moving_slide


# # # Add callbacks to control the modal and populate its content
# # @callback(
# #     Output("registration-points-modal", "is_open"),
# #     [
# #         Input("open-registration-points-modal", "n_clicks"),
# #         Input("close-registration-points-modal", "n_clicks"),
# #     ],
# #     State("registration-points-modal", "is_open"),
# # )
# # def toggle_modal(n1, n2, is_open):
# #     if n1 or n2:
# #         return not is_open
# #     return is_open


# def format_coord(value):
#     """Helper function to format coordinates"""
#     return f"{value:0.2f}" if isinstance(value, (int, float)) else "N/A"


# def compute_difference(moving_val, fixed_val):
#     """Compute difference between moving and fixed coordinates"""
#     try:
#         if moving_val is not None and fixed_val is not None:
#             diff = moving_val - fixed_val
#             return f"{diff:0.2f}"
#         return "N/A"
#     except:
#         return "N/A"

# # # Add callback to store optimal transform
# # @callback(
# #     Output("optimal-transform-store", "data"),
# #     [
# #         Input("fixed-image-viewer", "inputToPaper"),
# #         Input("moving-image-viewer", "inputToPaper"),
# #     ],
# # )
# # def update_optimal_transform(fixed_points, moving_points):
# #     if not fixed_points or not moving_points:
# #         return None

# #     try:
# #         # Get the points from the viewers
# #         fixed_items = fixed_points.get("actions", [{}])[0].get("itemList", [])
# #         moving_items = moving_points.get("actions", [{}])[0].get("itemList", [])

# #         # Calculate the transform
# #         calc_x, calc_y, calc_rotation = calculate_optimal_transform(
# #             fixed_items, moving_items
# #         )
# #         if calc_x is None:
# #             return None

# #         return {"x": calc_x, "y": calc_y, "rotation": calc_rotation}
# #     except Exception as e:
# #         print(f"Error updating optimal transform: {str(e)}")
# #         return None


# # @callback(
# #     [
# #         Output("moving-image-x-offset", "value", allow_duplicate=True),
# #         Output("moving-image-y-offset", "value", allow_duplicate=True),
# #         Output("moving-image-rotation", "value", allow_duplicate=True),
# #     ],
# #     Input("toggle-transform-button", "n_clicks"),
# #     [
# #         State("moving-image-x-offset", "value"),
# #         State("moving-image-y-offset", "value"),
# #         State("moving-image-rotation", "value"),
# #         State("optimal-transform-store", "data"),
# #     ],
# #     prevent_initial_call=True,
# # )
# # def toggle_transform(
# #     n_clicks, current_x, current_y, current_rotation, optimal_transform
# # ):
# #     if not n_clicks or optimal_transform is None:
# #         return dash.no_update, dash.no_update, dash.no_update

# #     # On odd clicks, switch to optimal transform
# #     if n_clicks % 2 == 1:
# #         print("Switching to optimal transform:", optimal_transform)
# #         return (
# #             optimal_transform["x"],
# #             optimal_transform["y"],
# #             optimal_transform["rotation"],
# #         )
# #     # On even clicks, switch back to previous manual transform
# #     else:
# #         print("Switching to manual transform:", current_x, current_y, current_rotation)
# #         return current_x, current_y, current_rotation

# #     # Get slides
# #     he_slide, moving_slide = get_slides_for_registration(slideList, selected_block)
# #     if not he_slide or not moving_slide:
# #         empty_return = 0, 0, 0, [], {}, [], []
# #         return empty_return


# #         fixed_bounds = {
# #             "width": he_tiles_info.get("sizeX", 10000),
# #             "height": he_tiles_info.get("sizeY", 10000),
# #         }
# #         moving_bounds = {
# #             "width": moving_tiles_info.get("sizeX", 10000),
# #             "height": moving_tiles_info.get("sizeY", 10000),
# #         }

# #         # Create tile sources with correct dimensions
# #         fixed_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{he_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": fixed_bounds["width"],
# #             }
# #         ]
# #         moving_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{moving_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": moving_bounds["width"],
# #             }
# #         ]

# #         try:
# #             # Get thumbnails with specified width
# #             fixed_thumb = get_thumbnail_image(he_slide["_id"], width=thumbnail_width)
# #             moving_thumb = get_thumbnail_image(
# #                 moving_slide["_id"], width=thumbnail_width
# #             )

# #             if fixed_thumb is None or moving_thumb is None:
# #                 print("Failed to retrieve thumbnails")
# #                 return (
# #                     # fixed_tile_source,
# #                     # moving_tile_source,
# #                     {"actions": []},
# #                     0,  # Default X offset
# #                     0,  # Default Y offset
# #                     0,  # Default rotation
# #                     [],
# #                     {},
# #                     [],
# #                     [],
# #                     None,
# #                     None,
# #                 )

# #             # Normalize image sizes for feature detection
# #             norm_fixed, norm_moving, (fixed_scale, moving_scale) = (
# #                 normalize_image_sizes(fixed_thumb, moving_thumb)
# #             )

# #             # print(
# #             #     f"Normalized shapes - fixed: {norm_fixed.shape}, moving: {norm_moving.shape}"
# #             # )
# #             # print(f"Scale factors - fixed: {fixed_scale}, moving: {moving_scale}")

# #             # Generate registration points on normalized images
# #             fixed_points, moving_points, debug_info = create_registration_points(
# #                 norm_fixed,
# #                 norm_moving,
# #                 detection_method,
# #                 num_points,
# #                 fixed_bounds,
# #             )

# #             if fixed_points and moving_points:
# #                 # Print original points
# #                 # if fixed_scale != moving_scale:
# #                 # print("Original fixed points:", fixed_points)
# #                 # print("Original moving points:", moving_points)

# #                 # Scale fixed points back to original fixed image size
# #                 fixed_points = [
# #                     (x * fixed_scale["x"], y * fixed_scale["y"])
# #                     for x, y in fixed_points
# #                 ]
# #                 moving_points = [
# #                     (x * moving_scale["x"], y * moving_scale["y"])
# #                     for x, y in moving_points
# #                 ]

# #                 # print("After scaling to original size:")
# #                 # print("Fixed points:", fixed_points)
# #                 # print("Moving points:", moving_points)

# #                 # Scale to full image coordinates
# #                 fixed_points = scale_points_to_full_size(
# #                     fixed_points, fixed_thumb.shape, fixed_bounds
# #                 )
# #                 moving_points = scale_points_to_full_size(
# #                     moving_points, moving_thumb.shape, moving_bounds
# #                 )

# #                 # print("After scaling to full size:")
# #                 # print("Fixed points:", fixed_points)
# #                 # print("Moving points:", moving_points)

# #                 # Convert points to numpy arrays for transformation calculation
# #                 fixed_np = np.float32(fixed_points)
# #                 moving_np = np.float32(moving_points)

# #                 # Calculate transformation matrix
# #                 transform_matrix = cv2.estimateAffinePartial2D(moving_np, fixed_np)[0]

# #                 # Extract transformation parameters
# #                 scale = np.sqrt(
# #                     transform_matrix[0, 0] ** 2 + transform_matrix[0, 1] ** 2
# #                 )
# #                 rotation = np.degrees(
# #                     np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
# #                 )
# #                 x_offset = transform_matrix[0, 2]
# #                 y_offset = transform_matrix[1, 2]

# #                 # Generate colors and create GeoJSON features
# #                 colors = generate_distinct_colors(len(fixed_points))
# #                 fixed_items = create_geojson_features(fixed_points, colors, "fixed")
# #                 moving_items = create_geojson_features(
# #                     moving_points, colors, "moving", layer_idx=1
# #                 )

# #                 # print("=== Successful Feature Matching Return ===")
# #                 # print("Fixed Items:", fixed_items)
# #                 # print("Number of items:", len(fixed_items))
# #                 # print(
# #                 #     "First item example:", fixed_items[0] if fixed_items else "No items"
# #                 # )
# #                 # print(
# #                 #     "Paper Input Structure:",
# #                 #     {"actions": [{"type": "drawItems", "itemList": fixed_items}]},
# #                 # )
# #                 # print("=====================================")
# #                 # print(fixed_points, moving_points, "fixed_points, moving_points")
# #                 return (
# #                     # fixed_tile_source,
# #                     # {"actions": [{"type": "drawItems", "itemList": fixed_items}]},
# #                     # moving_tile_source,
# #                     x_offset,  # X offset value
# #                     y_offset,  # Y offset value
# #                     rotation,  # Rotation value
# #                     [
# #                         html.Div(
# #                             f"Size: {moving_tiles_info.get('sizeX', 'N/A')}×{moving_tiles_info.get('sizeY', 'N/A')}  |  "
# #                             f"Resolution: {he_tiles_info.get('mm_x', 'N/A')}  |  "
# #                             f"Magnification: {moving_tiles_info.get('magnification', 'N/A')}"
# #                         )
# #                     ],
# #                     {
# #                         "sizeX": moving_tiles_info.get("sizeX", 1.0),
# #                         "sizeY": moving_tiles_info.get("sizeY", 1.0),
# #                         f"Resolution: {he_tiles_info.get('mm_x', 'N/A')}  |  "
# #                         "magnification": moving_tiles_info.get("magnification", "N/A"),
# #                     },
# #                     fixed_points,
# #                     moving_points,
# #                 )

# #         except Exception as e:
# #             print(f"Error in feature matching: {str(e)}")
# #             import traceback

# #             traceback.print_exc()

# #         # Return defaults if registration fails
# #         return (
# #             # moving_tile_source,
# #             {"actions": []},
# #             0,  # Default X offset
# #             0,  # Default Y offset
# #             0,  # Default rotation
# #             [],
# #             {},
# #             [],
# #             [],
# #         )

# #     except Exception as e:
# #         print(f"Error getting tiles info: {str(e)}")
# #         fixed_bounds = {"width": 10000, "height": 10000}
# #         moving_bounds = fixed_bounds.copy()

# #         # Create tile sources with correct dimensions
# #         fixed_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{he_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": fixed_bounds["width"],
# #             }
# #         ]
# #         moving_tile_source = [
# #             {
# #                 "tileSource": f"{DSA_BASE_URL}/item/{moving_slide['_id']}/tiles/dzi.dzi?token={token_info['_id']}",
# #                 "width": moving_bounds["width"],
# #             }
# #         ]

# #         # Return defaults if registration fails
# #         return (
# #             # fixed_tile_source,
# #             # moving_tile_source,
# #             {"actions": []},
# #             0,  # Default X offset
# #             0,  # Default Y offset
# #             0,  # Default rotation
# #             [],
# #             {},
# #             [],
# #             [],
# #             None,
# #             None,
# #         )

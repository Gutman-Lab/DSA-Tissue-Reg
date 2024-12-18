# # Add modal for thumbnail debugging
# thumbnail_debug_modal = dbc.Modal(
#     [
#         dbc.ModalHeader(dbc.ModalTitle("Registration Debug Information")),
#         dbc.ModalBody(
#             [
#                 dbc.Tabs(
#                     [
#                         dbc.Tab(
#                             [html.Div(id="modal-fixed-debug-content")],
#                             label="Fixed Image",
#                         ),
#                         dbc.Tab(
#                             [html.Div(id="modal-moving-debug-content")],
#                             label="Moving Image",
#                         ),
#                     ]
#                 )
#             ]
#         ),
#         dbc.ModalFooter(
#             dbc.Button(
#                 "Close", id="close-thumbnail-modal", className="ms-auto", n_clicks=0
#             )
#         ),
#     ],
#     id="thumbnail-debug-modal",
#     size="lg",
# )

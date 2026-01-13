# Dash paperdragon configuration.
COLORS = ["red", "orange", "yellow", "green", "blue", "purple"]
CLASSES = ["a", "b", "c", "d", "e", "f"]
DASH_PAPERDRAGON_CONFIG = {
    "eventBindings": [
        {
            "event": "keyDown",
            "key": "c",
            "action": "cycleProp",
            "property": "class",
        },
        {
            "event": "keyDown",
            "key": "x",
            "action": "cyclePropReverse",
            "property": "class",
        },
        {"event": "keyDown", "key": "d", "action": "deleteItem"},
        {
            "event": "keyDown",
            "key": "n",
            "action": "newItem",
            "tool": "rectangle",
        },
        {
            "event": "keyDown",
            "key": "e",
            "action": "editItem",
            "tool": "rectangle",
        },
        {
            "event": "mouseEnter",
            "action": "dashCallback",
            "callback": "mouseEnter",
        },
        {
            "event": "mouseLeave",
            "action": "dashCallback",
            "callback": "mouseLeave",
        },
    ],
    "callbacks": [
        {"eventName": "item-created", "callback": "createItem"},
        {"eventName": "property-changed", "callback": "propertyChanged"},
        {"eventName": "item-deleted", "callback": "itemDeleted"},
        {"eventName": "item-edited", "callback": "itemEdited"},
    ],
    "properties": {"class": CLASSES[0]},
    "defaultStyle": {
        "fillColor": COLORS[0],
        "strokeColor": COLORS[0],
        "rescale": {
            "strokeWidth": 1,
        },
        "fillOpacity": 0.2,
    },
    "styles": {
        "class": {
            k: {"fillColor": c, "strokeColor": c}
            for (k, c) in zip(CLASSES, COLORS)
        }
    },
}

DSA_COLOR_MAP = {
    "White Matter": "rgb(0,0,255)",
    "Gray Matter": "rgb(0,128,0)",
    "Superficial": "rgb(255,255,0)",
    "Leptomeninges": "rgb(0,0,0)",
    "Exclude": "rgb(255,0,0)",
}

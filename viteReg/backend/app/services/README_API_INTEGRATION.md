# Image Matching WebUI API Integration Notes

## API Endpoint Discovery

The actual API endpoints for image-matching-webui may differ from the template implementation. To find the correct endpoints:

1. **Check the source code**: https://github.com/Vincentqyw/image-matching-webui/tree/main/imcui/api
2. **Inspect the Gradio app**: Gradio apps expose API endpoints automatically
3. **Check the running service**: Visit `http://localhost:7860/docs` if FastAPI docs are available

## Common API Patterns

### Gradio API
If the webui uses Gradio, it typically exposes:
- `/api/predict` - Main prediction endpoint
- `/api/predict/` - Alternative endpoint format

### Custom FastAPI
If there's a custom API server, check `imcui/api/server.py` for:
- Route definitions
- Request/response models
- Endpoint paths

## Testing the API

You can test the API structure by:

```python
import requests

# Try common endpoints
endpoints = [
    "http://localhost:7860/api/match",
    "http://localhost:7860/api/predict",
    "http://localhost:7860/api/v1/match",
]

for endpoint in endpoints:
    try:
        response = requests.get(endpoint)
        print(f"{endpoint}: {response.status_code}")
        if response.status_code == 200:
            print(f"  Response: {response.json()}")
    except Exception as e:
        print(f"{endpoint}: Error - {e}")
```

## Updating the Integration

Once you've identified the correct API structure, update:

1. `call_image_matching_api()` in `image_matching_webui_helper.py`
   - Update `api_endpoint` variable
   - Adjust request payload format
   - Update response parsing

2. Request format may need:
   - File uploads instead of base64
   - Different parameter names
   - Different response structure

## Alternative: Using Gradio Client

If the webui uses Gradio, you can use the Gradio Python client:

```python
from gradio_client import Client

client = Client("http://localhost:7860")
result = client.predict(
    image0_path,
    image1_path,
    extractor,
    matcher,
    api_name="/match"  # Check actual API name
)
```

This is more robust than raw HTTP calls.

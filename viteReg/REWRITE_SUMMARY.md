# DSA Tissue Registration Application - Rewrite Summary

## Application Overview

**Purpose**: A web application for registering serial tissue sections from the Digital Slide Archive (DSA), enabling users to:
- Pull image sets from DSA
- Register serial sections (align images from different stains/sections)
- Translate annotation documents from one registered image to another
- View and manage stored registrations

**Current Stack**: 
- Backend: Python (Flask/Dash)
- Frontend: Dash (Python-based reactive framework)
- Image Processing: OpenCV, SimpleITK, scikit-image
- Deep Learning: PyTorch, Transformers (for LightGlue/UNI models)
- Storage: DSA (Girder-based), diskcache for caching

**Target Stack**:
- Backend: Python (FastAPI/Flask REST API)
- Frontend: React
- Image Processing: Same (backend)
- Deep Learning: Same (backend)

---

## Core Functionality

### 1. Case & Slide Management
- **Case Selection**: Users select cases from a predefined list
- **Slide Filtering**: Filter slides by:
  - Block ID
  - Annotation status (show only annotated slides)
  - Stain type (HE, IHC, etc.)
- **Slide Metadata**: Display case ID, region name, stain ID, block ID, annotation count

### 2. Image Registration
- **Fixed/Moving Image Selection**: 
  - Fixed image: Typically HE (Hematoxylin & Eosin) stained slide
  - Moving image: IHC (Immunohistochemistry) or other stained slide
- **Registration Methods**:
  - **SIFT** (Scale-Invariant Feature Transform)
  - **ORB** (Oriented FAST and Rotated BRIEF)
  - **AKAZE** (Accelerated-KAZE)
  - **Intensity-based** matching
  - **LightGlue** (Deep learning-based feature matching)
  - **SimpleITK** (Intensity-based registration with mutual information)
- **Fiducial Points**: 
  - Automatic detection of matching points
  - Manual point placement/adjustment
  - Point visualization on images
- **Transform Calculation**: 
  - Affine transformation (rotation, scale, translation)
  - Similarity transform
  - Parameters: rotation angle, scale factor, x/y offsets

### 3. Image Viewing
- **Multi-viewer Layout**:
  - Fixed image viewer
  - Moving image viewer  
  - Merged/composite viewer (overlay with opacity control)
- **Image Viewers**: Uses OpenSeadragon (via dash_paperdragon) for:
  - Deep zoom (DZI tile sources)
  - Pan and zoom
  - Point visualization
  - Transform application
- **Thumbnail Grid**: Display all slides in selected case/block with:
  - Thumbnail images
  - Stain type indicators
  - Selection highlighting

### 4. Registration Storage & Management
- **Stored Registrations View**: 
  - List of previously registered image pairs
  - Source and target image thumbnails
  - Registered result preview
  - Blend slider for viewing registered images

### 5. Annotation Translation
- **Annotation Transfer**: Translate annotations from fixed to moving image using calculated transform
- **Geojson Support**: Handle annotation formats compatible with DSA

---

## Architecture & Components

### Current Component Structure

#### Main Application Files
- `oldApp.py` / `app.py`: Main Dash application entry point
- `settings.py`: Configuration, authentication, shared state
  - DSA API client initialization
  - Token management
  - Caching setup (diskcache, joblib Memory)
  - Background callback manager

#### Frontend Components (Dash)
- `components/registrationControls.py`: Main registration interface
  - Case/slide selection controls
  - Registration parameter controls
  - Fiducial point grids
  - Image viewer layout
- `components/showStoredReg.py`: Stored registrations viewer
- `components/caseSelectionView.py`: Case selection interface
- `components/caseViewer.py`: Case browsing interface
- `components/lightGlueView.py`: LightGlue registration interface
- `components/osdRegViewers.py`: OpenSeadragon viewer components

#### Backend Utilities
- `utils/registration_utils.py`: Core registration algorithms
  - Feature detection (SIFT, ORB, AKAZE)
  - Point matching and filtering
  - Geometric validation
  - Image normalization
  - Transform calculation
- `utils/carlosReg_utils.py`: SimpleITK-based registration
- `components/carlos_reg_utils.py`: Alternative registration implementation

### Data Flow

1. **User selects case** → Fetch slides from DSA folder
2. **User filters by block ID** → Filter slide list
3. **User selects fixed/moving images** → Load image metadata and thumbnails
4. **Registration triggered** → 
   - Fetch thumbnail images
   - Normalize image sizes
   - Detect feature points
   - Match points
   - Calculate transform
   - Store fiducial points
5. **Transform applied** → Update merged viewer with transform parameters
6. **User adjusts parameters** → Real-time update of merged viewer
7. **Save registration** → Store transform and metadata (if implemented)

---

## Key Features & User Interactions

### Registration Workflow
1. Select case from dropdown
2. Filter by block ID (optional)
3. Toggle "Only annotated" filter
4. Select registration method (SIFT/ORB)
5. Select image ROI size (256/384/512/1024)
6. System auto-detects fixed (HE) and moving (IHC) slides
7. User can click thumbnails to change moving image
8. Registration runs automatically when images are selected
9. Fiducial points displayed in grids and on images
10. Transform parameters shown (rotation, scale, offset)
11. User can manually adjust parameters via inputs
12. Merged viewer shows result with opacity control

### Image Viewer Features
- Deep zoom (DZI tiles from DSA)
- Pan and zoom controls
- Fiducial point visualization (colored circles)
- Transform application (rotation, scale, translation)
- Opacity control for overlay
- Composite operation modes

### Data Display
- AG Grid tables for:
  - Slide metadata
  - Fiducial points (fixed and moving)
  - Stored registrations
- Filtering and sorting in grids
- Row selection for image selection

---

## Dependencies & Integrations

### External Services
- **DSA (Digital Slide Archive)**: 
  - Girder-based API
  - Base URL: Configurable via `DSA_BASE_URL` env var
  - Authentication: API key via `DSAKEY` env var
  - Endpoints used:
    - `GET /item?folderId={id}`: List items in folder
    - `GET /item/{id}/tiles`: Get tile metadata
    - `GET /item/{id}/tiles/dzi.dzi`: Deep zoom image source
    - `GET /item/{id}/tiles/thumbnail`: Thumbnail images
    - `GET /annotation/counts?items={ids}`: Annotation counts
    - `GET /annotation?itemId={id}`: Get annotations
    - `GET /annotation/{id}/geojson`: Get annotation GeoJSON

### Python Dependencies (Key)
- **Web Framework**: Dash 2.9.1, Flask 2.2.3
- **Image Processing**: 
  - opencv-python 4.10.0.84
  - SimpleITK
  - scikit-image
  - Pillow 11.0.0
- **Deep Learning**:
  - torch 2.5.1
  - torchvision 0.20.1
  - transformers 4.47.0
- **Data Handling**:
  - numpy 1.24.2
  - pandas 1.5.3
- **DSA Integration**:
  - girder-client 3.1.20
- **Caching**:
  - diskcache 5.6.1
  - joblib 1.4.2
- **UI Components**:
  - dash-bootstrap-components 1.4.1
  - dash_ag_grid 31.3.0
  - dash_paperdragon 0.1.11 (OpenSeadragon wrapper)

### Frontend Dependencies (Current)
- Dash components (server-rendered)
- Bootstrap CSS
- OpenSeadragon (via dash_paperdragon)

---

## Registration Algorithms

### 1. Feature-Based Methods (OpenCV)

#### SIFT
- Scale-invariant feature detection
- Good for images with scale differences
- Config: 500 features

#### ORB
- Fast, rotation-invariant
- Good for real-time applications
- Config: 10000 features, 8 levels

#### AKAZE
- Multi-scale feature detection
- Better edge detection
- Config: 6 octaves, 6 layers per octave

#### Workflow:
1. Detect keypoints in both images
2. Compute descriptors
3. Match descriptors (BFMatcher or FLANN)
4. Filter matches (ratio test, RANSAC)
5. Extract matched point pairs
6. Calculate affine transform from points

### 2. Intensity-Based Methods (SimpleITK)

#### Similarity Transform Registration
- Uses mutual information metric
- Gradient descent optimizer
- Multi-resolution approach
- Parameters: scale, rotation, translation

#### Workflow:
1. Convert images to grayscale
2. Initialize transform (center-based)
3. Set up registration method (mutual information)
4. Multi-resolution registration (4→2→1 shrink factors)
5. Execute registration
6. Extract transform parameters

### 3. Deep Learning Methods

#### LightGlue
- State-of-the-art feature matching
- Uses SuperPoint for feature detection
- Better matching accuracy
- Requires model download

#### UNI Model
- Vision transformer for feature extraction
- Mentioned in code but may not be fully implemented

---

## State Management (Current)

### Dash Store Components
- `registration_caseRootFolderId_store`: Current case folder ID
- `registration_caseSlideSet_store`: List of slides in current case
- `selectedRegionData`: Filtered slide list by block ID
- `fixed_image_id`: Selected fixed image ID
- `moving_image_id`: Selected moving image ID
- `fixedImage_fiducial_points`: Detected/manual points for fixed image
- `movingImage_fiducial_points`: Detected/manual points for moving image
- `optimal-transform-store`: Calculated transform parameters
- `moving-image-info_store`: Moving image metadata
- `registration_caseId`: Case ID
- `registration_blockId`: Block ID

### Caching
- **diskcache**: For background callbacks and general caching
- **joblib Memory**: For function result caching (registration results, image processing)

---

## API Endpoints Needed (Backend)

### Case & Slide Management
- `GET /api/cases`: List available cases
- `GET /api/cases/{caseId}/slides`: Get slides for a case
- `GET /api/slides/{slideId}`: Get slide metadata
- `GET /api/slides/{slideId}/thumbnail`: Get thumbnail image
- `GET /api/slides/{slideId}/annotations`: Get annotations for slide
- `GET /api/slides/{slideId}/annotations/count`: Get annotation count

### Image Registration
- `POST /api/register`: Perform registration
  - Input: `fixed_image_id`, `moving_image_id`, `method`, `roi_size`
  - Output: `fixed_points`, `moving_points`, `transform_params`
- `POST /api/register/transform`: Calculate transform from points
  - Input: `fixed_points[]`, `moving_points[]`
  - Output: `rotation`, `scale`, `offset_x`, `offset_y`
- `GET /api/register/stored`: List stored registrations
- `POST /api/register/save`: Save registration result
- `GET /api/register/{id}`: Get stored registration

### Image Viewing
- `GET /api/images/{imageId}/tiles`: Get tile metadata
- `GET /api/images/{imageId}/dzi`: Get DZI tile source URL
- `GET /api/images/{imageId}/thumbnail`: Get thumbnail (with caching)

### Annotations
- `POST /api/annotations/translate`: Translate annotations using transform
  - Input: `source_annotations`, `transform_params`
  - Output: `translated_annotations`

---

## UI Components to Recreate (React)

### Layout Components
1. **Main Container**: Tab-based layout
   - Stored Registrations tab
   - Registration tab
   - Case Selection tab (optional)

2. **Case Selection Panel**:
   - Dropdown for case selection
   - Block ID filter dropdown
   - "Only annotated" checkbox
   - Image ROI size selector
   - Registration method selector

3. **Thumbnail Grid**:
   - Responsive grid of slide thumbnails
   - Stain type indicators (color-coded headers)
   - Selection highlighting
   - Click to select moving image

4. **Image Viewers**:
   - Fixed image viewer (OpenSeadragon)
   - Moving image viewer (OpenSeadragon)
   - Merged/composite viewer (OpenSeadragon with overlay)
   - Image metadata display (size, resolution, magnification)

5. **Registration Controls**:
   - Fiducial points grid (AG Grid or similar)
   - Transform parameter inputs (rotation, scale, offset x/y)
   - Parameter display (calculated values)
   - Manual adjustment controls

6. **Stored Registrations View**:
   - Table/grid of stored registrations
   - Source/target thumbnails
   - Registered result preview
   - Blend slider

### Data Tables
- Slide metadata table (filterable, sortable)
- Fiducial points tables (editable?)
- Stored registrations table

---

## Recommendations for React Rewrite

### Backend Architecture
1. **FastAPI** recommended over Flask for:
   - Automatic API documentation
   - Type hints and validation
   - Async support for long-running operations
   - Better performance

2. **RESTful API Design**:
   - Separate endpoints for each resource
   - Use proper HTTP methods (GET, POST, PUT, DELETE)
   - Return JSON consistently
   - Include pagination for large lists

3. **Background Jobs**:
   - Use Celery or similar for long-running registration tasks
   - WebSocket or SSE for progress updates
   - Job status endpoints

4. **Caching Strategy**:
   - Keep diskcache for expensive operations
   - Add Redis for shared state (if multi-user)
   - Cache thumbnail images
   - Cache registration results

5. **Error Handling**:
   - Consistent error response format
   - Proper HTTP status codes
   - Detailed error messages for debugging

### Frontend Architecture
1. **State Management**:
   - **Redux Toolkit** or **Zustand** for global state
   - **React Query** or **SWR** for server state/caching
   - Local state for UI-only concerns

2. **Component Structure**:
   ```
   src/
   ├── components/
   │   ├── CaseSelection/
   │   ├── ImageViewer/
   │   ├── RegistrationControls/
   │   ├── ThumbnailGrid/
   │   └── StoredRegistrations/
   ├── hooks/
   │   ├── useRegistration.ts
   │   ├── useDSA.ts
   │   └── useImageLoader.ts
   ├── services/
   │   ├── api.ts
   │   └── dsa.ts
   ├── store/
   │   └── slices/
   └── types/
   ```

3. **Image Viewing**:
   - Use **OpenSeadragon** directly (not via wrapper)
   - React wrapper: `openseadragon-react` or custom hook
   - Handle tile loading and caching
   - Implement point visualization overlay

4. **UI Framework**:
   - **Material-UI** or **Ant Design** for components
   - **AG Grid React** for data tables (same as current)
   - **React Bootstrap** if keeping Bootstrap theme

5. **Real-time Updates**:
   - WebSocket for registration progress
   - Optimistic updates for UI responsiveness
   - Loading states and skeletons

### Key Considerations

1. **Performance**:
   - Lazy load image viewers
   - Virtualize long lists
   - Debounce parameter adjustments
   - Cache API responses

2. **User Experience**:
   - Show loading states during registration
   - Progress indicators for long operations
   - Error boundaries for graceful failures
   - Undo/redo for parameter adjustments

3. **Code Organization**:
   - Separate concerns (UI, business logic, API)
   - Reusable hooks for common operations
   - TypeScript for type safety
   - Unit tests for critical functions

4. **Migration Strategy**:
   - Build API first, test with Postman/curl
   - Build React frontend incrementally
   - Keep old app running during transition
   - Migrate one feature at a time

---

## Configuration

### Environment Variables
- `DSA_BASE_URL`: DSA API base URL
- `DSAKEY`: DSA API key for authentication
- `CACHE_DIR`: Directory for diskcache (default: `.npCacheDir`)

### Configuration Files
- `config.yaml`: Legacy config (may not be used)
- `.env`: Environment variables (not in repo)

### Default Values
- Sample case folder: `641bfd45867536bb7a236ae1`
- Default registration method: SIFT
- Default ROI size: 256
- Default thumbnail width: 1024

---

## Known Issues & Technical Debt

1. **Multiple App Files**: `oldApp.py`, `app.py`, `new_app.py` - consolidate
2. **Duplicate Registration Code**: Multiple implementations in different files
3. **Hard-coded Values**: Case IDs, folder IDs in code
4. **Error Handling**: Inconsistent across components
5. **Type Safety**: No type hints in many functions
6. **Testing**: No visible test suite
7. **Documentation**: Minimal inline documentation

---

## File Structure Reference

```
DSA-Tissue-Reg/
├── oldApp.py                 # Main entry point (current)
├── app.py                    # Alternative entry point
├── settings.py               # Configuration & auth
├── config.yaml               # Legacy config
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container definition
├── runTissueReg.sh          # Build/run script
├── components/               # Dash components
│   ├── registrationControls.py
│   ├── showStoredReg.py
│   ├── caseSelectionView.py
│   ├── caseViewer.py
│   ├── lightGlueView.py
│   └── osdRegViewers.py
├── utils/                    # Backend utilities
│   ├── registration_utils.py
│   └── carlosReg_utils.py
└── cache/                    # Disk cache directory
```

---

## Next Steps for Rewrite

1. **Phase 1: API Development**
   - Set up FastAPI project
   - Implement DSA client wrapper
   - Create registration service layer
   - Build REST endpoints
   - Add API documentation

2. **Phase 2: Core React App**
   - Set up React + TypeScript project
   - Implement routing
   - Set up state management
   - Create API client
   - Build basic layout

3. **Phase 3: Feature Migration**
   - Case selection
   - Image viewing
   - Registration workflow
   - Stored registrations

4. **Phase 4: Polish**
   - Error handling
   - Loading states
   - Performance optimization
   - Testing
   - Documentation

---

## Additional Notes

- The app uses **dash_paperdragon** which wraps OpenSeadragon for Dash
- Registration can be computationally expensive - consider background jobs
- Some registration methods require model downloads (LightGlue)
- The app caches registration results to avoid recomputation
- Annotation translation is mentioned but may not be fully implemented
- The merged viewer supports opacity and composite operations for overlay visualization


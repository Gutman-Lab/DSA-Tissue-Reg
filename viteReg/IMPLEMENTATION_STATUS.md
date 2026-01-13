# Implementation Status

This document tracks the progress of implementing functionality from the Python version into the React/FastAPI stack.

## ✅ Completed

### Backend Infrastructure
- [x] FastAPI application structure
- [x] DSA client service (`app/services/dsa_client.py`)
  - Girder client wrapper
  - Authentication with API key
  - Methods for listing items, getting metadata, thumbnails, DZI URLs
  - Annotation count fetching

### API Endpoints
- [x] `GET /api/cases` - List available cases
- [x] `GET /api/cases/{case_id}/slides` - Get slides for a case with filtering
  - Supports filtering by `block_id`
  - Supports filtering by `only_annotated`
  - Returns annotation counts
- [x] `GET /api/slides/{slide_id}` - Get slide metadata
- [x] `GET /api/images/{image_id}/tiles` - Get tile metadata
- [x] `GET /api/images/{image_id}/dzi` - Get DZI URL
- [x] `GET /api/images/{image_id}/thumbnail` - Get thumbnail URL

## 🚧 In Progress

### Backend
- [ ] Registration endpoints (`/api/register`)
  - [ ] `POST /api/register` - Perform registration
  - [ ] `POST /api/register/transform` - Calculate transform from points
  - [ ] `GET /api/register/stored` - List stored registrations
  - [ ] `POST /api/register/save` - Save registration

### Frontend
- [ ] Case selection component
- [ ] Thumbnail grid component
- [ ] Image viewers (using bdsa-react-components SlideViewer)
- [ ] Registration controls

## 📋 Next Steps (Priority Order)

### Phase 1: Core Data Flow (Current)
1. ✅ DSA client service
2. ✅ Case/slide endpoints
3. ✅ Image metadata endpoints
4. ⏭️ **Next**: Registration service layer
5. ⏭️ Registration endpoints

### Phase 2: Frontend Basics
1. Case selection dropdown
2. Thumbnail grid with filtering
3. Basic layout with tabs
4. API client integration

### Phase 3: Image Viewing
1. Integrate SlideViewer from bdsa-react-components
2. Fixed/moving image viewers
3. Merged/composite viewer
4. Fiducial point visualization

### Phase 4: Registration Workflow
1. Registration method selection
2. Automatic registration trigger
3. Fiducial points display
4. Transform parameter controls
5. Manual parameter adjustment

### Phase 5: Advanced Features
1. Stored registrations view
2. Annotation translation
3. Background job processing
4. Progress indicators

## 📝 Implementation Notes

### DSA Client Service
The `DSAClient` class wraps `girder_client` and provides:
- Authentication on initialization
- Helper methods for common DSA operations
- Error handling and logging
- Singleton pattern via `get_dsa_client()`

### Case/Slide Data Structure
Based on the Python version, slides have:
- `_id`: Item ID from DSA
- `name`: Slide name
- `meta.npSchema.blockID`: Block identifier
- `meta.npSchema.stainID`: Stain type (HE, IHC, etc.)
- `annotationCount`: Number of annotations (fetched separately)

### Registration Algorithms Available
From `REWRITE_SUMMARY.md` and code analysis:
1. **SIFT** - Scale-invariant feature detection (500 features)
2. **ORB** - Fast, rotation-invariant (10000 features, 8 levels)
3. **AKAZE** - Multi-scale feature detection (6 octaves)
4. **SimpleITK** - Intensity-based with mutual information
5. **LightGlue** - Deep learning-based (requires model download)

### Key Python Functions to Port
From `utils/registration_utils.py`:
- `get_thumbnail_image(item_id, width)` - Get thumbnail as numpy array
- `normalize_image_sizes(fixed_img, moving_img)` - Resize images to same size
- `create_registration_points(fixed_img, moving_img, method, num_points)` - Main registration function
- `calculate_optimal_transform(fixed_points, moving_points)` - Calculate affine transform

## 🔍 Reference Files

### Python Implementation
- `components/registrationControls.py` - Main registration UI and logic
- `utils/registration_utils.py` - Core registration algorithms
- `utils/carlosReg_utils.py` - SimpleITK-based registration
- `settings.py` - DSA client initialization

### Documentation
- `REWRITE_SUMMARY.md` - Comprehensive feature documentation
- `.cursorrules` - bdsa-react-components integration guide

## 🎯 Success Criteria

The implementation is complete when:
1. Users can select a case and view slides
2. Users can select fixed/moving images
3. Registration runs automatically and shows fiducial points
4. Users can adjust transform parameters manually
5. Merged viewer shows registered result with opacity control
6. Stored registrations can be viewed and managed


/**
 * Registration API service
 */
import { apiClient } from './api'

export interface RegistrationRequest {
  fixed_image_id: string
  moving_image_id: string
  method?: 'sift' | 'orb' | 'akaze' | 'simpleitk' | 'tps'
  roi_size?: number
}

export interface RegistrationResponse {
  job_id: string
  status: string
  message: string
}

export interface RegistrationResult {
  job_id: string
  status: string
  fixed_image_id: string
  moving_image_id: string
  transform_matrix: number[][]
  rotation_degrees: number
  offset_x: number
  offset_y: number
  scale: number
  mutual_information: number
  normalized_cross_correlation?: number  // NCC - better for different stains
  structural_similarity?: number  // SSIM - structural similarity
  dice_coefficient?: number  // Dice - best for different stains (mask overlap)
  success: boolean
  error?: string
  created_at?: string
  completed_at?: string
  method?: string  // Registration method: 'simpleitk' (rigid/affine) or 'affine_tps' (affine+TPS hybrid, uses LightGlue)
  num_matches?: number  // Number of feature matches (LightGlue/TPS only)
  num_inliers?: number  // Number of inlier matches (LightGlue/TPS only)
}

export async function startRigidRegistration(
  fixedImageId: string,
  movingImageId: string
): Promise<RegistrationResponse> {
  return apiClient.post<RegistrationResponse>('/registration/rigid', {
    fixed_image_id: fixedImageId,
    moving_image_id: movingImageId,
    method: 'simpleitk',
  })
}

export async function getRegistrationStatus(
  jobId: string
): Promise<RegistrationResult> {
  return apiClient.get<RegistrationResult>(`/registration/job/${jobId}`)
}

export async function autoRegisterCase(
  caseId: string,
  blockId?: string,
  method?: 'simpleitk' | 'affine_tps'
): Promise<RegistrationResponse[]> {
  const params = new URLSearchParams()
  if (blockId) {
    params.append('block_id', blockId)
  }
  if (method) {
    params.append('method', method)
  }
  const query = params.toString()
  return apiClient.post<RegistrationResponse[]>(
    `/registration/auto-register/${caseId}${query ? `?${query}` : ''}`
  )
}

export async function clearCache(): Promise<{ success: boolean; message: string }> {
  return apiClient.post<{ success: boolean; message: string }>(
    '/registration/clear-cache'
  )
}

export async function loadStoredRegistrations(
  caseId: string,
  method?: 'simpleitk' | 'affine_tps'
): Promise<RegistrationResult[]> {
  const params = new URLSearchParams()
  if (method) {
    params.append('method', method)
  }
  const query = params.toString()
  return apiClient.get<RegistrationResult[]>(
    `/registration/stored-registrations/${caseId}${query ? `?${query}` : ''}`
  )
}

export interface ParameterExplorationParams {
  extractor_type?: string
  max_keypoints?: number
  detection_threshold?: number
  nms_window_size?: number
  n_layers?: number
  depth_confidence?: number
  width_confidence?: number
  filter_threshold?: number
  affine_ransac_thresh_px?: number
  affine_max_iters?: number
  max_matches_for_tps?: number
  tps_min_inliers?: number
  thumbnail_width?: number
}

export async function exploreParameters(
  fixedId: string,
  movingId: string,
  params: ParameterExplorationParams
): Promise<RegistrationResult & { parameters: ParameterExplorationParams }> {
  const queryParams = new URLSearchParams({
    fixed_id: fixedId,
    moving_id: movingId,
    ...Object.fromEntries(
      Object.entries(params).filter(([_, v]) => v !== undefined).map(([k, v]) => [k, String(v)])
    )
  })
  return apiClient.get<RegistrationResult & { parameters: ParameterExplorationParams }>(
    `/registration/explore-parameters?${queryParams.toString()}`
  )
}


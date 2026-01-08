/**
 * Type definitions for the application
 */

export interface Slide {
  id: string
  name: string
  case_id: string
  block_id?: string | null
  stain_type?: string | null
  annotation_count: number
  meta?: {
    npSchema?: {
      blockID?: string
      stainID?: string
      regionName?: string
    }
    // Backward compatibility: npReg (defaults to simpleitk)
    npReg?: {
      srcImage?: string
      method?: string
      xOffset?: number
      yOffset?: number
      scale?: number
      rotation?: number
      regImageSize?: number
      preRotate?: string
      mutual_information?: number
      dice_coefficient?: number
      normalized_cross_correlation?: number
      structural_similarity?: number
      relative_rotation?: number
      num_matches?: number
      num_inliers?: number
    }
    // Method-specific registrations: npReg_simpleitk, npReg_lightglue, etc.
    npReg_simpleitk?: {
      srcImage?: string
      method?: string
      xOffset?: number
      yOffset?: number
      scale?: number
      rotation?: number
      regImageSize?: number
      preRotate?: string
      mutual_information?: number
      dice_coefficient?: number
      normalized_cross_correlation?: number
      structural_similarity?: number
      relative_rotation?: number
    }
    npReg_lightglue?: {
      srcImage?: string
      method?: string
      xOffset?: number
      yOffset?: number
      scale?: number
      rotation?: number
      regImageSize?: number
      preRotate?: string
      mutual_information?: number
      dice_coefficient?: number
      normalized_cross_correlation?: number
      structural_similarity?: number
      relative_rotation?: number
      num_matches?: number
      num_inliers?: number
    }
  }
}

export interface Point {
  x: number
  y: number
}

export interface TransformParams {
  rotation: number
  scale: number
  offset_x: number
  offset_y: number
}

export interface RegistrationRequest {
  fixed_image_id: string
  moving_image_id: string
  method: 'sift' | 'orb' | 'akaze' | 'lightglue' | 'simpleitk'
  roi_size: number
}

export interface RegistrationResponse {
  fixed_points: Point[]
  moving_points: Point[]
  transform_params: TransformParams
  debug_info: Record<string, unknown>
}


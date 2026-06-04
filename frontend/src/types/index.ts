/** Mask2Former-Cutout shared types */

export interface CutoutRequest {
  target_classes?: string[]
  confidence_threshold: number
  return_format: 'png' | 'alpha' | 'composite'
}

export interface CutoutResponse {
  status: string
  message: string
  mask_url: string
  mask_base64: string
  processing_time_ms: number
  detected_classes: string[]
}

export interface HealthResponse {
  status: string
  model_loaded: boolean
  gpu_available: boolean
  gpu_name: string
  version: string
}

export type CutoutMode = 'person' | 'car' | 'all'

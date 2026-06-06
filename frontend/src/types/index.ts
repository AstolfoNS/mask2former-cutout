/** Mask2Former-Cutout shared types */

export interface CutoutRequest {
  target_classes?: string
  score_threshold: number
  return_mask: boolean
  return_overlay: boolean
  return_cutout: boolean
  model_id?: string
}

export interface TimingInfo {
  preprocess_ms: number
  inference_ms: number
  postprocess_ms: number
  total_ms: number
}

export interface CutoutResponse {
  job_id: string
  status: string
  classes: string[]
  model_id: string
  model_label: string
  files: {
    original_url?: string
    cutout_url?: string
    mask_url?: string
    mask_person_url?: string
    mask_car_url?: string
    overlay_url?: string
  }
  timing: TimingInfo
}

export interface ResultQueryResponse {
  job_id: string
  status: string
  files: Record<string, string>
}

export interface HealthResponse {
  status: string
  device: string
  model_loaded: boolean
  gpu_name: string
  version: string
  active_model_id: string
}

export interface ModelInfo {
  id: string
  label: string
  path: string
  active: boolean
}

export interface ModelListResponse {
  models: ModelInfo[]
}

export type CutoutMode = 'person' | 'car' | 'both'

export type ViewMode = 'original' | 'mask' | 'overlay' | 'cutout'

export interface HistoryEntry {
  job_id: string
  timestamp: number
  filename: string
  classes: string[]
  model_id: string
  model_label: string
  files: CutoutResponse['files']
  thumbnail_url?: string
}

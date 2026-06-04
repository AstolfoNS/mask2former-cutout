/**
 * Composable for interacting with the backend cutout API.
 */
import { ref } from 'vue'
import axios from 'axios'
import type { CutoutResponse, HealthResponse, CutoutMode } from '@/types'

const API_BASE = '/api/v1'

export function useCutoutApi() {
  const isProcessing = ref(false)
  const lastResult = ref<CutoutResponse | null>(null)
  const error = ref<string | null>(null)

  async function checkHealth(): Promise<HealthResponse | null> {
    try {
      const { data } = await axios.get<HealthResponse>(`${API_BASE}/health`)
      return data
    } catch {
      return null
    }
  }

  async function runCutout(
    file: File,
    mode: CutoutMode = 'all',
    threshold: number = 0.5,
    format: string = 'png',
  ): Promise<CutoutResponse | null> {
    isProcessing.value = true
    error.value = null
    lastResult.value = null

    const formData = new FormData()
    formData.append('file', file)
    if (mode !== 'all') {
      formData.append('target_classes', mode)
    }
    formData.append('confidence_threshold', String(threshold))
    formData.append('return_format', format)

    try {
      const { data } = await axios.post<CutoutResponse>(
        `${API_BASE}/cutout`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } },
      )
      lastResult.value = data
      return data
    } catch (e: any) {
      const msg = e.response?.data?.detail || e.message || 'Unknown error'
      error.value = msg
      return null
    } finally {
      isProcessing.value = false
    }
  }

  function clearResult() {
    lastResult.value = null
    error.value = null
  }

  return {
    isProcessing,
    lastResult,
    error,
    checkHealth,
    runCutout,
    clearResult,
  }
}

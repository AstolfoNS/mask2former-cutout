/**
 * Composable for interacting with the backend cutout API.
 */
import { ref } from "vue"
import axios from "axios"
import type { CutoutResponse, HealthResponse, CutoutMode, HistoryEntry } from "@/types"

const API_BASE = "/api"

function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem("cutout-history")
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveHistory(entries: HistoryEntry[]) {
  const trimmed = entries.slice(0, 20)
  localStorage.setItem("cutout-history", JSON.stringify(trimmed))
}

const isProcessing = ref(false)
const lastResult = ref<CutoutResponse | null>(null)
const error = ref<string | null>(null)
const history = ref<HistoryEntry[]>(loadHistory())

export function useCutoutApi() {
  async function checkHealth(): Promise<HealthResponse | null> {
    try {
      const { data } = await axios.get<HealthResponse>(API_BASE + "/health")
      return data
    } catch {
      return null
    }
  }

  async function runCutout(
    file: File,
    mode: CutoutMode = "both",
    threshold: number = 0.5,
  ): Promise<CutoutResponse | null> {
    isProcessing.value = true
    error.value = null
    lastResult.value = null

    const formData = new FormData()
    formData.append("file", file)

    if (mode === "person") {
      formData.append("target_classes", "person")
    } else if (mode === "car") {
      formData.append("target_classes", "car")
    }

    formData.append("score_threshold", String(threshold))
    formData.append("return_mask", "true")
    formData.append("return_overlay", "true")
    formData.append("return_cutout", "true")

    try {
      const { data } = await axios.post<CutoutResponse>(
        API_BASE + "/segment",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } },
      )
      lastResult.value = data

      const entry: HistoryEntry = {
        job_id: data.job_id,
        timestamp: Date.now(),
        filename: file.name,
        classes: data.classes,
        files: data.files,
        thumbnail_url: data.files.overlay_url || data.files.cutout_url,
      }
      history.value.unshift(entry)
      saveHistory(history.value)

      return data
    } catch (e: any) {
      const msg = e.response?.data?.detail || e.message || "Unknown error"
      error.value = msg
      return null
    } finally {
      isProcessing.value = false
    }
  }

  async function queryResult(jobId: string): Promise<CutoutResponse | null> {
    try {
      const { data } = await axios.get<CutoutResponse>(
        API_BASE + "/results/" + jobId,
      )
      lastResult.value = data
      return data
    } catch {
      return null
    }
  }

  function clearResult() {
    lastResult.value = null
    error.value = null
  }

  function clearHistory() {
    history.value = []
    localStorage.removeItem("cutout-history")
  }

  return {
    isProcessing,
    lastResult,
    error,
    history,
    checkHealth,
    runCutout,
    queryResult,
    clearResult,
    clearHistory,
  }
}

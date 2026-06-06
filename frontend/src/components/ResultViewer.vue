<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import {
  NCard,
  NEmpty,
  NTag,
  NSpace,
  NTabs,
  NTabPane,
  NButton,
  NProgress,
  NText,
  NStatistic,
} from 'naive-ui'
import type { CutoutResponse, ViewMode } from '@/types'

const props = defineProps<{
  result: CutoutResponse | null
  originalFile: File | null
  isProcessing: boolean
}>()

const activeView = ref<ViewMode>('overlay')
const originalUrl = ref<string>('')

watch(
  () => props.originalFile,
  (file) => {
    if (file) {
      originalUrl.value = URL.createObjectURL(file)
    }
  },
)

const hasResult = computed(() => props.result !== null && props.result.status === 'success')

const currentImageUrl = computed(() => {
  if (!props.result) return ''
  const files = props.result.files
  switch (activeView.value) {
    case 'original':
      return files.original_url || originalUrl.value
    case 'mask':
      return files.mask_url || ''
    case 'overlay':
      return files.overlay_url || ''
    case 'cutout':
      return files.cutout_url || ''
    default:
      return ''
  }
})

const classColors: Record<string, string> = {
  person: '#52c41a',
  car: '#1890ff',
}

function downloadFile(url: string, filename: string) {
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

function downloadCurrent() {
  if (!props.result || !currentImageUrl.value) return
  const ext = activeView.value === 'cutout' ? 'cutout.png' : `${activeView.value}.png`
  const filename = `${props.result.job_id}_${ext}`
  downloadFile(currentImageUrl.value, filename)
}

function colorFor(cls: string): string {
  return classColors[cls] || '#999'
}
</script>

<template>
  <NCard title="Result Preview" class="w-full">
    <!-- Loading state -->
    <div v-if="isProcessing" class="py-8">
      <NProgress
        type="line"
        :percentage="100"
        :indicator-placement="'inside'"
        processing
      />
      <NText class="text-sm text-gray-400 mt-3 block text-center">
        Running inference on GPU...
      </NText>
    </div>

    <!-- Result view -->
    <div v-else-if="hasResult && result" class="space-y-4">
      <!-- Detected classes & timing -->
      <div class="flex items-center justify-between flex-wrap gap-2">
        <NSpace>
          <NText class="text-sm font-medium">Detected:</NText>
          <NTag
            v-for="cls in result.classes"
            :key="cls"
            :color="{ color: colorFor(cls), textColor: '#fff' }"
            size="small"
          >
            {{ cls }}
          </NTag>
          <NTag v-if="result.classes.length === 0" type="default" size="small">
            none
          </NTag>
        </NSpace>
        <NText class="text-xs text-gray-400">
          {{ result.timing.total_ms.toFixed(0) }} ms
          (inference: {{ result.timing.inference_ms.toFixed(0) }} ms)
        </NText>
      </div>

      <!-- View tabs -->
      <NTabs
        v-model:value="activeView"
        type="segment"
        size="small"
        animated
      >
        <NTabPane name="original" tab="Original" />
        <NTabPane name="overlay" tab="Overlay" />
        <NTabPane name="mask" tab="Mask" />
        <NTabPane name="cutout" tab="Cutout" />
      </NTabs>

      <!-- Image display -->
      <div
        class="w-full bg-repeat rounded overflow-hidden border"
        style="min-height: 256px; background-image: url('data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 20 20%22><rect width=%2210%22 height=%2210%22 fill=%22%23eee%22/><rect x=%2210%22 y=%2210%22 width=%2210%22 height=%2210%22 fill=%22%23eee%22/><rect x=%2210%22 width=%2210%22 height=%2210%22 fill=%22%23ddd%22/><rect y=%2210%22 width=%2210%22 height=%2210%22 fill=%22%23ddd%22/></svg>');"
      >
        <img
          v-if="currentImageUrl"
          :src="currentImageUrl"
          class="w-full object-contain max-h-80"
          :alt="activeView"
        />
        <div v-else class="flex items-center justify-center h-64 text-gray-400 text-sm">
          No preview available for this view.
        </div>
      </div>

      <!-- Download buttons -->
      <NSpace>
        <NButton
          size="small"
          type="primary"
          @click="downloadCurrent"
        >
          Download {{ activeView === 'cutout' ? 'Cutout (PNG)' : activeView }}
        </NButton>
        <NButton
          v-if="result.files.cutout_url"
          size="small"
          @click="downloadFile(result.files.cutout_url, `${result.job_id}_cutout.png`)"
        >
          Download Cutout
        </NButton>
        <NButton
          v-if="result.files.overlay_url"
          size="small"
          @click="downloadFile(result.files.overlay_url, `${result.job_id}_overlay.png`)"
        >
          Download Overlay
        </NButton>
        <NButton
          v-if="result.files.mask_url"
          size="small"
          @click="downloadFile(result.files.mask_url, `${result.job_id}_mask.png`)"
        >
          Download Mask
        </NButton>
      </NSpace>

      <!-- Timing details -->
      <NStatistic label="Total Time" :value="`${result.timing.total_ms.toFixed(0)} ms`">
        <template #suffix>
          <NText class="text-xs text-gray-400">
            pre: {{ result.timing.preprocess_ms.toFixed(0) }}ms
            | inf: {{ result.timing.inference_ms.toFixed(0) }}ms
            | post: {{ result.timing.postprocess_ms.toFixed(0) }}ms
          </NText>
        </template>
      </NStatistic>
    </div>

    <!-- No result state -->
    <NEmpty
      v-else
      description="No result yet. Upload an image and click Run Cutout."
    />
  </NCard>
</template>

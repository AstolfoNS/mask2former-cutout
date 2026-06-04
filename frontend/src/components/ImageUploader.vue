<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  NUpload,
  NButton,
  NSelect,
  NSlider,
  NSpace,
  NCard,
  NText,
  NUploadDragger,
  NP,
  useMessage,
} from 'naive-ui'
import type { UploadFileInfo } from 'naive-ui'
import { useCutoutApi } from '@/composables/useCutoutApi'
import type { CutoutMode } from '@/types'

const props = defineProps<{
  isProcessing: boolean
  currentMode: CutoutMode
}>()

const emit = defineEmits<{
  'image-selected': [file: File]
  'cutout-result': [result: { maskBase64: string; detectedClasses: string[]; processingTimeMs: number }]
  'processing-state': [processing: boolean]
  'mode-change': [mode: CutoutMode]
  'service-status': [status: string]
}>()

const message = useMessage()
const { checkHealth, runCutout } = useCutoutApi()

const selectedFile = ref<File | null>(null)
const threshold = ref(0.5)
const returnFormat = ref<'png' | 'alpha' | 'composite'>('png')
const previewUrl = ref<string>('')

const modeOptions = [
  { label: 'All (Person + Car)', value: 'all' as CutoutMode },
  { label: 'Person Only', value: 'person' as CutoutMode },
  { label: 'Car Only', value: 'car' as CutoutMode },
]

const formatOptions = [
  { label: 'Mask (PNG)', value: 'png' },
  { label: 'Alpha Channel (RGBA)', value: 'alpha' },
  { label: 'Composite Overlay', value: 'composite' },
]

onMounted(async () => {
  const health = await checkHealth()
  if (health) {
    emit('service-status', health.status)
  }
})

function onFileChange(options: { file: UploadFileInfo; fileList: UploadFileInfo[] }) {
  const file = options.file.file
  if (file) {
    selectedFile.value = file
    previewUrl.value = URL.createObjectURL(file)
    emit('image-selected', file)
  }
}

async function handleCutout() {
  if (!selectedFile.value) {
    message.warning('Please select an image first.')
    return
  }

  emit('processing-state', true)
  const result = await runCutout(
    selectedFile.value,
    props.currentMode,
    threshold.value,
    returnFormat.value,
  )
  emit('processing-state', false)

  if (result) {
    emit('cutout-result', {
      maskBase64: result.mask_base64,
      detectedClasses: result.detected_classes,
      processingTimeMs: result.processing_time_ms,
    })
    message.success(`Cutout completed in ${result.processing_time_ms.toFixed(0)} ms`)
  } else {
    message.error('Cutout failed. Check console for details.')
  }
}
</script>

<template>
  <NCard title="Upload Image" class="w-full">
    <NUpload
      accept="image/*"
      :max="1"
      :show-file-list="false"
      @change="onFileChange"
    >
      <NUploadDragger class="cursor-pointer">
        <div class="py-8 text-center">
          <NP class="text-gray-500">
            Click or drag an image to this area
          </NP>
          <NP class="text-xs text-gray-400 mt-1">
            Supports JPG, PNG, WEBP
          </NP>
        </div>
      </NUploadDragger>
    </NUpload>

    <div v-if="previewUrl" class="mt-4">
      <img
        :src="previewUrl"
        alt="Preview"
        class="w-full h-48 object-contain rounded border bg-gray-50"
      />
    </div>

    <div class="mt-4 space-y-3">
      <div>
        <NText class="text-sm font-medium">Target Mode</NText>
        <NSelect
          :value="currentMode"
          :options="modeOptions"
          size="small"
          class="mt-1"
          @update:value="(v: CutoutMode) => emit('mode-change', v)"
        />
      </div>

      <div>
        <NText class="text-sm font-medium">
          Confidence Threshold: {{ threshold.toFixed(2) }}
        </NText>
        <NSlider
          v-model:value="threshold"
          :min="0.1"
          :max="0.95"
          :step="0.05"
          class="mt-1"
        />
      </div>

      <div>
        <NText class="text-sm font-medium">Output Format</NText>
        <NSelect
          v-model:value="returnFormat"
          :options="formatOptions"
          size="small"
          class="mt-1"
        />
      </div>
    </div>

    <NSpace justify="end" class="mt-4">
      <NButton
        type="primary"
        :loading="isProcessing"
        :disabled="!selectedFile"
        @click="handleCutout"
      >
        {{ isProcessing ? 'Processing...' : 'Run Cutout' }}
      </NButton>
    </NSpace>
  </NCard>
</template>

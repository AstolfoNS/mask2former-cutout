<script setup lang="ts">
import { computed, ref, onBeforeUnmount, onMounted } from 'vue'
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
import type { CutoutMode, CutoutResponse, ModelInfo } from '@/types'

const props = defineProps<{
  isProcessing: boolean
  currentMode: CutoutMode
}>()

const emit = defineEmits<{
  'image-selected': [file: File]
  'cutout-result': [result: CutoutResponse]
  'processing-state': [processing: boolean]
  'mode-change': [mode: CutoutMode]
  'service-status': [status: string, gpuName: string]
}>()

const message = useMessage()
const { checkHealth, listModels, runCutout } = useCutoutApi()

const uploadFileList = ref<UploadFileInfo[]>([])
const selectedFile = ref<File | null>(null)
const threshold = ref(0.5)
const previewUrl = ref<string>('')
const selectedModelId = ref<string | null>(null)
const models = ref<ModelInfo[]>([])
const modelOptions = ref<{ label: string; value: string }[]>([])
const modelLoading = ref(false)

const selectedModel = computed(() => (
  models.value.find((model) => model.id === selectedModelId.value) || null
))

const modeOptions = [
  { label: 'Both (Person + Car)', value: 'both' as CutoutMode },
  { label: 'Person Only', value: 'person' as CutoutMode },
  { label: 'Car Only', value: 'car' as CutoutMode },
]

onMounted(async () => {
  const [health] = await Promise.all([
    checkHealth(),
    loadModels(),
  ])
  if (health) {
    emit('service-status', health.status, health.gpu_name)
  }
})

async function loadModels() {
  modelLoading.value = true
  try {
    const response = await listModels()
    models.value = response?.models || []
    modelOptions.value = models.value.map((model) => ({
      label: buildModelOptionLabel(model),
      value: model.id,
    }))
    selectedModelId.value = models.value.find((model) => model.active)?.id
      || models.value[0]?.id
      || null
  } finally {
    modelLoading.value = false
  }
}

function buildModelOptionLabel(model: ModelInfo): string {
  const name = formatModelName(model)
  const activeSuffix = model.active ? ' (active)' : ''
  if (model.id === 'default') {
    return `Default: ${name}${activeSuffix}`
  }
  return `${name}${activeSuffix}`
}

function formatModelName(model: ModelInfo): string {
  const normalizedPath = model.path.replace(/\\/g, '/')
  const parts = normalizedPath.split('/').filter(Boolean)
  const bestModelIndex = parts.lastIndexOf('best_model')

  if (bestModelIndex > 0) {
    return `${parts[bestModelIndex - 1]}/best_model`
  }
  if (parts.length >= 2) {
    return parts.slice(-2).join('/')
  }
  return model.label || model.id
}

function onFileChange(options: { file: UploadFileInfo; fileList: UploadFileInfo[] }) {
  const file = options.file.file
  if (file) {
    uploadFileList.value = [options.file]
    selectedFile.value = file
    if (previewUrl.value) {
      URL.revokeObjectURL(previewUrl.value)
    }
    previewUrl.value = URL.createObjectURL(file)
    emit('image-selected', file)
  }
}

onBeforeUnmount(() => {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
  }
})

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
    selectedModelId.value,
  )
  emit('processing-state', false)

  if (result) {
    emit('cutout-result', result)
    message.success(
      `Cutout completed in ${result.timing.total_ms.toFixed(0)} ms`
    )
  } else {
    message.error('Cutout failed. Please try again.')
  }
}
</script>

<template>
  <NCard title="Upload Image" class="w-full">
    <NUpload
      v-model:file-list="uploadFileList"
      accept="image/*"
      :default-upload="false"
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

    <!-- Preview -->
    <div v-if="previewUrl" class="mt-4">
      <img
        :src="previewUrl"
        alt="Preview"
        class="w-full h-48 object-contain rounded border bg-gray-50"
      />
    </div>

    <!-- Parameters -->
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
        <NText class="text-sm font-medium">Inference Model</NText>
        <NSelect
          v-model:value="selectedModelId"
          :options="modelOptions"
          :loading="modelLoading"
          :disabled="modelOptions.length === 0"
          size="small"
          class="mt-1"
        />
        <NText
          v-if="selectedModel"
          class="block mt-1 text-xs text-gray-500 break-all"
        >
          {{ selectedModel.id === 'default' ? 'Default model' : 'Selected model' }}:
          {{ selectedModel.path }}
        </NText>
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
    </div>

    <!-- Submit -->
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

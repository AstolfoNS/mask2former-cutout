<script setup lang="ts">
import { ref } from 'vue'
import {
  NConfigProvider,
  NMessageProvider,
  NLayout,
  NLayoutHeader,
  NLayoutContent,
  NLayoutFooter,
  NH1,
  NP,
  NTag,
  darkTheme,
} from 'naive-ui'
import ImageUploader from './components/ImageUploader.vue'
import MaskCanvas from './components/MaskCanvas.vue'
import ResultPanel from './components/ResultPanel.vue'
import type { CutoutMode } from './types'

const currentFile = ref<File | null>(null)
const currentMode = ref<CutoutMode>('all')
const maskBase64 = ref<string>('')
const detectedClasses = ref<string[]>([])
const processingTime = ref<number>(0)
const isProcessing = ref(false)
const serviceStatus = ref<string>('checking...')

function onImageSelected(file: File) {
  currentFile.value = file
  maskBase64.value = ''
  detectedClasses.value = []
}

function onCutoutResult(result: { maskBase64: string; detectedClasses: string[]; processingTimeMs: number }) {
  maskBase64.value = result.maskBase64
  detectedClasses.value = result.detectedClasses
  processingTime.value = result.processingTimeMs
}

function onProcessingState(processing: boolean) {
  isProcessing.value = processing
}

function onModeChange(mode: CutoutMode) {
  currentMode.value = mode
}
</script>

<template>
  <NConfigProvider>
    <NMessageProvider>
      <NLayout class="min-h-screen">
        <NLayoutHeader bordered class="px-6 py-3">
          <div class="flex items-center justify-between max-w-6xl mx-auto">
            <div class="flex items-center gap-3">
              <NH1 class="!text-xl !font-bold !m-0">Mask2Former Cutout</NH1>
              <NTag :type="serviceStatus === 'ok' ? 'success' : 'warning'" size="small">
                {{ serviceStatus }}
              </NTag>
            </div>
            <NP class="!m-0 text-sm text-gray-500">
              Person &amp; Car Matting
            </NP>
          </div>
        </NLayoutHeader>

        <NLayoutContent class="max-w-6xl mx-auto w-full p-6">
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="space-y-4">
              <ImageUploader
                :is-processing="isProcessing"
                :current-mode="currentMode"
                @image-selected="onImageSelected"
                @cutout-result="onCutoutResult"
                @processing-state="onProcessingState"
                @mode-change="onModeChange"
                @service-status="(s: string) => serviceStatus = s"
              />
            </div>

            <div class="space-y-4">
              <MaskCanvas
                :mask-base64="maskBase64"
                :original-file="currentFile"
              />
              <ResultPanel
                :detected-classes="detectedClasses"
                :processing-time="processingTime"
                :is-processing="isProcessing"
              />
            </div>
          </div>
        </NLayoutContent>

        <NLayoutFooter bordered class="text-center py-3 text-sm text-gray-400">
          Mask2Former-Cutout v0.1.0 | Built with Vue 3 + FastAPI
        </NLayoutFooter>
      </NLayout>
    </NMessageProvider>
  </NConfigProvider>
</template>

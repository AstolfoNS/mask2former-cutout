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
} from 'naive-ui'
import ImageUploader from './components/ImageUploader.vue'
import ResultViewer from './components/ResultViewer.vue'
import HistoryList from './components/HistoryList.vue'
import type { CutoutMode, CutoutResponse } from './types'

const currentFile = ref<File | null>(null)
const currentMode = ref<CutoutMode>('both')
const lastResult = ref<CutoutResponse | null>(null)
const isProcessing = ref(false)
const serviceStatus = ref<string>('checking...')
const gpuInfo = ref<string>('')

function onImageSelected(file: File) {
  currentFile.value = file
  lastResult.value = null
}

function onCutoutResult(result: CutoutResponse) {
  lastResult.value = result
}

function onProcessingState(processing: boolean) {
  isProcessing.value = processing
}

function onModeChange(mode: CutoutMode) {
  currentMode.value = mode
}

function onServiceStatus(status: string, gpuName: string) {
  serviceStatus.value = status
  gpuInfo.value = gpuName
}

function onSelectHistory(result: CutoutResponse) {
  lastResult.value = result
  currentFile.value = null
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
                {{ serviceStatus === 'ok' ? gpuInfo : serviceStatus }}
              </NTag>
            </div>
            <NP class="!m-0 text-sm text-gray-500">
              Person &amp; Car Matting
            </NP>
          </div>
        </NLayoutHeader>

        <NLayoutContent class="max-w-6xl mx-auto w-full p-6">
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Left column: Upload + Parameters -->
            <div class="space-y-4">
              <ImageUploader
                :is-processing="isProcessing"
                :current-mode="currentMode"
                @image-selected="onImageSelected"
                @cutout-result="onCutoutResult"
                @processing-state="onProcessingState"
                @mode-change="onModeChange"
                @service-status="onServiceStatus"
              />
            </div>

            <!-- Right column: Result + History -->
            <div class="space-y-4">
              <ResultViewer
                :result="lastResult"
                :original-file="currentFile"
                :is-processing="isProcessing"
              />
              <HistoryList
                @select-history="onSelectHistory"
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

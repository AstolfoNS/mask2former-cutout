<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { NCard, NEmpty, NP, NSpace, NSwitch } from 'naive-ui'

const props = defineProps<{
  maskBase64: string
  originalFile: File | null
}>()

const showOverlay = ref(true)
const opacity = ref(0.5)

const originalUrl = ref<string>('')

watch(
  () => props.originalFile,
  (file) => {
    if (file) {
      originalUrl.value = URL.createObjectURL(file)
    }
  },
)

const maskUrl = computed(() => {
  if (!props.maskBase64) return ''
  return `data:image/png;base64,${props.maskBase64}`
})

const hasResult = computed(() => !!props.maskBase64)
</script>

<template>
  <NCard title="Result Preview" class="w-full">
    <div v-if="hasResult" class="space-y-3">
      <NSpace align="center">
        <NSwitch v-model:value="showOverlay" size="small" />
        <NP class="!m-0 text-sm">Show overlay</NP>
        <NP v-if="showOverlay" class="!m-0 text-xs text-gray-400">
          Opacity: {{ (opacity * 100).toFixed(0) }}%
        </NP>
      </NSpace>

      <div class="relative w-full bg-gray-100 rounded overflow-hidden" style="min-height: 256px;">
        <!-- Original image -->
        <img
          v-if="originalUrl"
          :src="originalUrl"
          class="w-full object-contain"
          alt="Original"
        />
        <!-- Mask overlay -->
        <img
          v-if="showOverlay && maskUrl"
          :src="maskUrl"
          :style="{ opacity: opacity, position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain' }"
          alt="Mask overlay"
        />
      </div>

      <!-- Bare mask -->
      <div class="bg-gray-100 rounded overflow-hidden">
        <NP class="text-xs text-gray-400 px-2 pt-2">Mask Only</NP>
        <img
          v-if="maskUrl"
          :src="maskUrl"
          class="w-full object-contain"
          alt="Mask"
        />
      </div>
    </div>

    <NEmpty v-else description="No result yet. Upload an image and click Run Cutout." />
  </NCard>
</template>

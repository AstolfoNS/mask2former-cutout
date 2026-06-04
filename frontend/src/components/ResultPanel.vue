<script setup lang="ts">
import { computed } from 'vue'
import { NCard, NTag, NStatistic, NSpace, NProgress, NText } from 'naive-ui'

const props = defineProps<{
  detectedClasses: string[]
  processingTime: number
  isProcessing: boolean
}>()

const classColors: Record<string, string> = {
  person: '#52c41a',
  car: '#1890ff',
}

function colorFor(cls: string): string {
  return classColors[cls] || '#999'
}
</script>

<template>
  <NCard title="Detection Result" class="w-full">
    <div v-if="isProcessing" class="py-4">
      <NProgress
        type="line"
        :percentage="100"
        :indicator-placement="'inside'"
        processing
      />
      <NText class="text-sm text-gray-400 mt-2 block text-center">
        Running inference...
      </NText>
    </div>

    <div v-else-if="detectedClasses.length > 0">
      <NStatistic label="Processing Time" :value="`${processingTime.toFixed(0)} ms`" />
      <div class="mt-3">
        <NText class="text-sm font-medium">Detected Classes:</NText>
        <NSpace class="mt-1">
          <NTag
            v-for="cls in detectedClasses"
            :key="cls"
            :color="{ color: colorFor(cls), textColor: '#fff' }"
            size="medium"
          >
            {{ cls }}
          </NTag>
        </NSpace>
      </div>
    </div>

    <div v-else class="py-4 text-center text-gray-400">
      No result available.
    </div>
  </NCard>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import {
  NCard,
  NEmpty,
  NList,
  NListItem,
  NButton,
  NTag,
  NText,
  NPopconfirm,
} from 'naive-ui'
import { useCutoutApi } from '@/composables/useCutoutApi'
import type { CutoutResponse, HistoryEntry } from '@/types'

const emit = defineEmits<{
  'select-history': [result: CutoutResponse]
}>()

const { history, clearHistory } = useCutoutApi()

function formatTime(ts: number): string {
  const d = new Date(ts)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

function onSelect(entry: HistoryEntry) {
  // Convert HistoryEntry to CutoutResponse shape
  const result: CutoutResponse = {
    job_id: entry.job_id,
    status: 'success',
    classes: entry.classes,
    model_id: entry.model_id || 'default',
    model_label: entry.model_label || entry.model_id || 'default',
    files: entry.files,
    timing: {
      preprocess_ms: 0,
      inference_ms: 0,
      postprocess_ms: 0,
      total_ms: 0,
    },
  }
  emit('select-history', result)
}

const classColors: Record<string, string> = {
  person: '#52c41a',
  car: '#1890ff',
}

function colorFor(cls: string): string {
  return classColors[cls] || '#999'
}
</script>

<template>
  <NCard title="History" class="w-full">
    <div v-if="history.length > 0">
      <div class="flex justify-end mb-2">
        <NPopconfirm @positive-click="clearHistory">
          <template #trigger>
            <NButton size="tiny" text type="error">Clear All</NButton>
          </template>
          Clear all history?
        </NPopconfirm>
      </div>

      <div class="overflow-y-auto" style="max-height: 360px">
        <NList clickable>
          <NListItem
            v-for="entry in history"
            :key="entry.job_id"
            @click="onSelect(entry)"
          >
            <template #prefix>
              <img
                v-if="entry.thumbnail_url"
                :src="entry.thumbnail_url"
                class="w-12 h-12 object-cover rounded border"
                alt="thumb"
              />
            </template>
            <div class="flex flex-col gap-1">
              <NText class="text-sm font-medium truncate">
                {{ entry.filename }}
              </NText>
              <div class="flex items-center gap-2">
                <NTag
                  v-for="cls in entry.classes"
                  :key="cls"
                  :color="{ color: colorFor(cls), textColor: '#fff' }"
                  size="tiny"
                >
                  {{ cls }}
                </NTag>
                <NText class="text-xs text-gray-400">
                  {{ entry.model_label || entry.model_id || 'default' }} | {{ formatTime(entry.timestamp) }}
                </NText>
              </div>
            </div>
          </NListItem>
        </NList>
      </div>
    </div>

    <NEmpty v-else description="No history yet." class="py-4" />
  </NCard>
</template>

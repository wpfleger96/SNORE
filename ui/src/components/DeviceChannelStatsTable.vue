<template>
    <section v-if="rows.length" class="section-card">
        <h2>Device Channel Statistics</h2>
        <div v-if="isMobile" class="card-list">
            <div v-for="row in rows" :key="row.glossaryKey" class="data-card">
                <div class="data-card-header">
                    <span>
                        {{ GLOSSARY[row.glossaryKey].label }}
                        <InfoHint :glossary-key="row.glossaryKey" />
                    </span>
                </div>
                <div v-for="(stat, i) in STATISTICS" :key="stat.key" class="data-card-row">
                    <span class="data-card-label">{{ stat.label }}</span>
                    <span class="data-card-value">{{ row.cells[i] }}</span>
                </div>
            </div>
        </div>
        <Table v-else>
            <TableHeader>
                <TableRow>
                    <TableHead>Channel</TableHead>
                    <TableHead v-for="stat in STATISTICS" :key="stat.key">
                        {{ stat.label }}
                    </TableHead>
                </TableRow>
            </TableHeader>
            <TableBody>
                <TableRow v-for="row in rows" :key="row.glossaryKey">
                    <TableCell class="whitespace-normal">
                        {{ GLOSSARY[row.glossaryKey].label }}
                        <InfoHint :glossary-key="row.glossaryKey" />
                    </TableCell>
                    <TableCell v-for="(stat, i) in STATISTICS" :key="stat.key">
                        {{ row.cells[i] }}
                    </TableCell>
                </TableRow>
            </TableBody>
        </Table>
    </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import InfoHint from '@/components/InfoHint.vue'
import { useIsMobile } from '@/composables/useIsMobile'
import { GLOSSARY } from '@/utils/glossary'
import type { DeviceChannelStats } from '@/types'

const props = defineProps<{ stats?: DeviceChannelStats | null }>()

const { isMobile } = useIsMobile()

// Each channel's fields are `<field>_<key>` for every entry of STATISTICS.
const CHANNELS = [
    { glossaryKey: 'fl_device', field: 'device_flg', decimals: 2 },
    { glossaryKey: 'snore_device', field: 'snore', decimals: 2 },
    { glossaryKey: 'resp_rate_device', field: 'respiratory_rate', decimals: 1 },
    { glossaryKey: 'tidal_volume_device', field: 'tidal_volume', decimals: 0 },
    { glossaryKey: 'mv_device', field: 'minute_ventilation', decimals: 1 },
    { glossaryKey: 'ti_device', field: 'ti', decimals: 2 },
    { glossaryKey: 'ie_ratio_device', field: 'ie_ratio', decimals: 1 },
    { glossaryKey: 'pulse_device', field: 'pulse', decimals: 0 },
] as const
const STATISTICS = [
    { key: 'min', label: 'Min' },
    { key: 'median', label: 'Median' },
    { key: '95th', label: '95th' },
    { key: '99_5th', label: '99.5th' },
    { key: 'max', label: 'Max' },
] as const

const rows = computed(() => {
    const s = props.stats
    if (!s) return []
    return CHANNELS.map(({ glossaryKey, field, decimals }) => ({
        glossaryKey,
        values: STATISTICS.map(({ key }) => s[`${field}_${key}` as const]),
        decimals,
    }))
        .filter((row) => row.values.some((value) => value != null))
        .map(({ glossaryKey, values, decimals }) => ({
            glossaryKey,
            cells: values.map((value) => value?.toFixed(decimals) ?? '---'),
        }))
})
</script>

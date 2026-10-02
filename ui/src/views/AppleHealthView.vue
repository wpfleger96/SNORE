<template>
    <div v-if="loading" class="loading-state">
        <Loader2 class="inline h-4 w-4 animate-spin" /> Loading sleep data...
    </div>

    <ErrorState v-else-if="error" :message="error" :retry="reload" />

    <div v-else-if="data" class="apple-health-view">
        <h1 class="page-title">Apple Health</h1>

        <div class="stats-grid mb-6">
            <StatCard
                label="Avg Sleep"
                :value="avgTotalSleep"
                unit="hr"
                :decimals="1"
                glossary-key="total_sleep"
            />
            <StatCard
                label="Avg Efficiency"
                :value="avgEfficiency"
                unit="%"
                :decimals="1"
                glossary-key="sleep_efficiency"
            />
            <StatCard
                label="Avg Deep"
                :value="avgDeep"
                unit="hr"
                :decimals="1"
                glossary-key="deep_sleep"
            />
            <StatCard
                label="Avg REM"
                :value="avgRem"
                unit="hr"
                :decimals="1"
                glossary-key="rem_sleep"
            />
        </div>
        <p class="stats-caption">Averages over {{ data.items.length }} displayed nights</p>

        <div v-if="data.items.length === 0" class="empty-state">
            <p class="empty-message">No Apple Health sleep data found.</p>
            <RouterLink to="/import" class="cta-link">
                Import your Apple Health export to get started
            </RouterLink>
        </div>

        <template v-else>
            <div v-if="isMobile" class="card-list">
                <RouterLink
                    v-for="night in data.items"
                    :key="night.night_date"
                    class="data-card"
                    :to="`/apple-health/${night.night_date}`"
                >
                    <div class="data-card-header">{{ formatDateFull(night.night_date) }}</div>
                    <div class="data-card-row">
                        <span class="data-card-label">Total Sleep (hr)</span>
                        <span class="data-card-value">{{
                            fmtHours(night.total_sleep_seconds)
                        }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">Efficiency (%)</span>
                        <span class="data-card-value">{{
                            fmtPct(night.sleep_efficiency_pct)
                        }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">Core (hr)</span>
                        <span class="data-card-value">{{ fmtHours(night.core_seconds) }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">Deep (hr)</span>
                        <span class="data-card-value">{{ fmtHours(night.deep_seconds) }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">REM (hr)</span>
                        <span class="data-card-value">{{ fmtHours(night.rem_seconds) }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">Source</span>
                        <span class="data-card-value">{{ night.preferred_source ?? '---' }}</span>
                    </div>
                </RouterLink>
            </div>
            <Table v-else>
                <TableHeader>
                    <TableRow>
                        <TableHead>Date</TableHead>
                        <TableHead class="text-right">Total Sleep</TableHead>
                        <TableHead class="text-right">Efficiency</TableHead>
                        <TableHead class="text-right">Core</TableHead>
                        <TableHead class="text-right">Deep</TableHead>
                        <TableHead class="text-right">REM</TableHead>
                        <TableHead>Source</TableHead>
                    </TableRow>
                </TableHeader>
                <TableBody>
                    <TableRow
                        v-for="night in data.items"
                        :key="night.night_date"
                        class="night-row"
                        @click="router.push(`/apple-health/${night.night_date}`)"
                    >
                        <TableCell>
                            <RouterLink
                                :to="`/apple-health/${night.night_date}`"
                                class="night-link"
                            >
                                {{ formatDateFull(night.night_date) }}
                            </RouterLink>
                        </TableCell>
                        <TableCell class="text-right tabular-nums">
                            {{ fmtHours(night.total_sleep_seconds) }}
                        </TableCell>
                        <TableCell class="text-right tabular-nums">
                            {{ fmtPct(night.sleep_efficiency_pct) }}
                        </TableCell>
                        <TableCell class="text-right tabular-nums">
                            {{ fmtHours(night.core_seconds) }}
                        </TableCell>
                        <TableCell class="text-right tabular-nums">
                            {{ fmtHours(night.deep_seconds) }}
                        </TableCell>
                        <TableCell class="text-right tabular-nums">
                            {{ fmtHours(night.rem_seconds) }}
                        </TableCell>
                        <TableCell class="text-muted-foreground text-sm">
                            {{ night.preferred_source ?? '---' }}
                        </TableCell>
                    </TableRow>
                </TableBody>
            </Table>

            <PaginationBar
                :offset="offset"
                :page-size="PAGE_SIZE"
                :total="data.total"
                @page="onPage"
            />
        </template>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { Loader2 } from '@lucide/vue'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import StatCard from '@/components/StatCard.vue'
import PaginationBar from '@/components/PaginationBar.vue'
import ErrorState from '@/components/ErrorState.vue'
import { useApiLoad } from '@/composables/useApiLoad'
import { useIsMobile } from '@/composables/useIsMobile'
import { getHealthNights } from '@/api/health'
import { avg, formatDateFull } from '@/utils/formatting'

const router = useRouter()
const { isMobile } = useIsMobile()

const PAGE_SIZE = 30
const offset = ref(0)

const { data, loading, error, reload } = useApiLoad(() =>
    getHealthNights({ limit: PAGE_SIZE, offset: offset.value }),
)

function onPage(newOffset: number): void {
    offset.value = newOffset
    void reload()
}

function fmtHours(seconds: number | null | undefined): string {
    return seconds != null ? (seconds / 3600).toFixed(1) : '---'
}

function fmtPct(pct: number | null | undefined): string {
    return pct != null ? pct.toFixed(1) : '---'
}

const avgTotalSleep = computed(() => {
    const vals = data.value?.items.map((n) => n.total_sleep_seconds)
    if (!vals) return null
    const a = avg(vals)
    return a != null ? a / 3600 : null
})

const avgEfficiency = computed(() => {
    const vals = data.value?.items.map((n) => n.sleep_efficiency_pct)
    return vals ? avg(vals) : null
})

const avgDeep = computed(() => {
    const vals = data.value?.items.map((n) => n.deep_seconds)
    if (!vals) return null
    const a = avg(vals)
    return a != null ? a / 3600 : null
})

const avgRem = computed(() => {
    const vals = data.value?.items.map((n) => n.rem_seconds)
    if (!vals) return null
    const a = avg(vals)
    return a != null ? a / 3600 : null
})
</script>

<style scoped>
.apple-health-view {
    max-width: 1000px;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 0.75rem;
}

.mb-6 {
    margin-bottom: 1.5rem;
}

.night-row {
    cursor: pointer;
}

.night-row:hover {
    background: var(--color-accent);
}

.night-link {
    color: var(--color-primary);
    text-decoration: none;
}

.night-link:hover {
    text-decoration: underline;
}

.tabular-nums {
    font-variant-numeric: tabular-nums;
}

.empty-state {
    padding: 3rem 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    text-align: center;
}

.stats-caption {
    font-size: 0.75rem;
    color: var(--color-muted-foreground);
    margin-bottom: 1.5rem;
}

.empty-message {
    color: var(--color-muted-foreground);
}

.cta-link {
    color: var(--color-primary);
    text-decoration: none;
    font-weight: 500;
}

.cta-link:hover {
    text-decoration: underline;
}
</style>

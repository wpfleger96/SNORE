<template>
    <div v-if="loading" class="loading-state">
        <Loader2 class="inline h-4 w-4 animate-spin" /> Loading events...
    </div>

    <ErrorState v-else-if="error" :message="error" :retry="reload" />

    <div v-else class="event-explorer">
        <RouterLink :to="{ name: 'session-detail', params: { id: sessionId } }" class="back-link">
            <ArrowLeft class="inline h-4 w-4" /> Back to Session
        </RouterLink>

        <h1 class="page-title">Events — Session #{{ sessionId }}</h1>

        <!-- Summary -->
        <div class="summary-row">
            <StatCard label="Total Events" :value="filteredEvents.length" :decimals="0" />
            <!-- Computed here from device-scored events and session length. -->
            <StatCard
                label="Events/Hour"
                :value="eventsPerHour"
                :decimals="1"
                :provenance="glossaryProvenance('events_per_hour')"
            />
            <StatCard label="Types" :value="uniqueTypes.length" :decimals="0" />
        </div>

        <!-- Event Match (if analysis exists) -->
        <div v-if="matchResult" class="section-card">
            <h2>Device-scored vs SNORE-detected</h2>
            <div class="match-grid">
                <StatCard
                    label="Device-scored"
                    :value="matchResult.machine_count"
                    :decimals="0"
                    glossary-key="machine_events"
                />
                <StatCard
                    label="SNORE-detected"
                    :value="matchResult.programmatic_count"
                    :decimals="0"
                    glossary-key="programmatic_events"
                    field="programmatic_count"
                />
                <StatCard
                    label="Matched"
                    :value="matchResult.matched"
                    :decimals="0"
                    field="matched"
                />
                <StatCard
                    label="False Positives"
                    :value="matchResult.false_positives"
                    :decimals="0"
                    glossary-key="false_positives"
                    field="false_positives"
                />
                <StatCard
                    label="False Negatives"
                    :value="matchResult.false_negatives"
                    :decimals="0"
                    glossary-key="false_negatives"
                    field="false_negatives"
                />
                <StatCard
                    label="Sensitivity"
                    :value="sensitivity"
                    unit="%"
                    :decimals="1"
                    glossary-key="sensitivity"
                    :provenance="glossaryProvenance('sensitivity')"
                />
            </div>
        </div>

        <h2 class="events-heading">Device-scored events</h2>

        <!-- Filters -->
        <div class="filter-bar">
            <span class="filter-label text-muted-foreground">Filter by type:</span>
            <div class="type-chips">
                <button
                    v-for="t in uniqueTypes"
                    :key="t"
                    class="inline-flex cursor-pointer items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors"
                    :class="
                        activeTypes.has(t)
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted/50 text-muted-foreground opacity-50'
                    "
                    @click="toggleType(t)"
                >
                    {{ t }}
                </button>
            </div>
            <InfoHint label="Event Types">
                <EventTypeLegend />
            </InfoHint>
        </div>

        <!-- Event List (mobile cards) -->
        <div v-if="isMobile" class="card-list">
            <div v-for="(row, i) in paginatedEvents" :key="i" class="data-card">
                <div class="data-card-header">
                    <span
                        class="event-badge"
                        :style="{ background: EVENT_COLORS[row.event_type] ?? '#ccc' }"
                    >
                        {{ row.event_type }}
                    </span>
                    <button
                        type="button"
                        class="time-link text-primary"
                        @click="jumpToWaveform(row.offset_seconds)"
                    >
                        {{ formatTimeOffset(row.offset_seconds) }}
                    </button>
                </div>
                <div class="data-card-row">
                    <span class="data-card-label">Duration</span>
                    <span class="data-card-value">{{ row.duration_seconds.toFixed(1) }}s</span>
                </div>
                <div class="data-card-row">
                    <span class="data-card-label"
                        >SpO₂ Drop <InfoHint glossary-key="spo2_drop"
                    /></span>
                    <span class="data-card-value">{{
                        row.spo2_drop != null ? row.spo2_drop.toFixed(1) + '%' : emDash
                    }}</span>
                </div>
                <div class="data-card-row">
                    <span class="data-card-label">Peak FL <InfoHint glossary-key="peak_fl" /></span>
                    <span class="data-card-value">{{
                        row.peak_flow_limitation != null
                            ? row.peak_flow_limitation.toFixed(2)
                            : emDash
                    }}</span>
                </div>
            </div>
        </div>

        <!-- Event List (desktop table) -->
        <Table v-else>
            <TableHeader>
                <TableRow>
                    <TableHead style="width: 80px">Type</TableHead>
                    <TableHead>Time</TableHead>
                    <TableHead style="width: 100px">Duration</TableHead>
                    <TableHead style="width: 100px" class="whitespace-nowrap"
                        >SpO₂ Drop <InfoHint glossary-key="spo2_drop"
                    /></TableHead>
                    <TableHead style="width: 90px" class="whitespace-nowrap"
                        >Peak FL <InfoHint glossary-key="peak_fl"
                    /></TableHead>
                </TableRow>
            </TableHeader>
            <TableBody>
                <TableRow v-for="(row, i) in paginatedEvents" :key="i" class="odd:bg-muted/50">
                    <TableCell>
                        <span
                            class="event-badge"
                            :style="{ background: EVENT_COLORS[row.event_type] ?? '#ccc' }"
                        >
                            {{ row.event_type }}
                        </span>
                    </TableCell>
                    <TableCell>
                        <button
                            class="time-link text-primary"
                            @click="jumpToWaveform(row.offset_seconds)"
                        >
                            {{ formatTimeOffset(row.offset_seconds) }}
                        </button>
                    </TableCell>
                    <TableCell>{{ row.duration_seconds.toFixed(1) }}s</TableCell>
                    <TableCell>{{
                        row.spo2_drop != null ? row.spo2_drop.toFixed(1) + '%' : emDash
                    }}</TableCell>
                    <TableCell>{{
                        row.peak_flow_limitation != null
                            ? row.peak_flow_limitation.toFixed(2)
                            : emDash
                    }}</TableCell>
                </TableRow>
            </TableBody>
        </Table>

        <PaginationBar
            :offset="currentPage * pageSize"
            :page-size="pageSize"
            :total="filteredEvents.length"
            @page="(newOffset) => (currentPage = Math.floor(newOffset / pageSize))"
        />
    </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import { Loader2, ArrowLeft } from '@lucide/vue'
import StatCard from '@/components/StatCard.vue'
import PaginationBar from '@/components/PaginationBar.vue'
import InfoHint from '@/components/InfoHint.vue'
import EventTypeLegend from '@/components/EventTypeLegend.vue'
import { getSessionEvents, getEventMatch } from '@/api/events'
import { getSession } from '@/api/sessions'
import { useApiLoad } from '@/composables/useApiLoad'
import { useIsMobile } from '@/composables/useIsMobile'
import { formatTimeOffset } from '@/utils/formatting'
import { glossaryProvenance } from '@/utils/provenance'
import { EVENT_COLORS } from '@/types'
import type { EventItem, EventMatchResult } from '@/types'
import ErrorState from '@/components/ErrorState.vue'

const emDash = '\u2014' // em-dash used for null display
const props = defineProps<{ sessionId: number }>()
const router = useRouter()
const { isMobile } = useIsMobile()

const activeTypes = ref<Set<string>>(new Set())
const currentPage = ref(0)
const pageSize = 50

const { data, loading, error, reload } = useApiLoad(async () => {
    const [events, session] = await Promise.all([
        getSessionEvents(props.sessionId),
        getSession(props.sessionId, false),
    ])
    let match: EventMatchResult | null = null
    try {
        match = await getEventMatch(props.sessionId)
    } catch {
        // No analysis — match panel hidden
    }
    return { events, duration: session.duration_hours, match }
}, 'Failed to load events')

const allEvents = computed<EventItem[]>(() => data.value?.events ?? [])
const matchResult = computed(() => data.value?.match ?? null)
const sessionDuration = computed(() => data.value?.duration ?? 0)

const uniqueTypes = computed(() => [...new Set(allEvents.value.map((e) => e.event_type))].sort())

const filteredEvents = computed(() => {
    if (activeTypes.value.size === 0) return allEvents.value
    return allEvents.value.filter((e) => activeTypes.value.has(e.event_type))
})

const paginatedEvents = computed(() => {
    const start = currentPage.value * pageSize
    return filteredEvents.value.slice(start, start + pageSize)
})

const eventsPerHour = computed(() => {
    if (!sessionDuration.value) return null
    return filteredEvents.value.length / sessionDuration.value
})

const sensitivity = computed(() => {
    if (!matchResult.value || matchResult.value.machine_count === 0) return null
    return (matchResult.value.matched / matchResult.value.machine_count) * 100
})

watch(activeTypes, () => {
    currentPage.value = 0
})

function toggleType(type: string): void {
    const s = new Set(activeTypes.value)
    if (s.has(type)) s.delete(type)
    else s.add(type)
    activeTypes.value = s
}

function jumpToWaveform(offsetSec: number): void {
    void router.push({
        name: 'session-detail',
        params: { id: props.sessionId },
        query: { t: String(Math.floor(offsetSec)) },
    })
}
</script>

<style scoped>
.event-explorer {
    max-width: 1100px;
}

.summary-row {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}

/* Matches the shared .section-card h2 heading style. */
.events-heading {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

.match-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 0.5rem;
}

.type-chips {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
}

.event-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--color-foreground);
}

.time-link {
    cursor: pointer;
    text-decoration: none;
    background: none;
    border: none;
    padding: 0;
    font: inherit;
    color: inherit;
}
.time-link:hover {
    text-decoration: underline;
}

@media (max-width: 767.98px) {
    .time-link {
        min-height: var(--tap-target);
    }
}
</style>

<template>
    <div v-if="loading" class="loading-state">
        <Loader2 class="inline h-4 w-4 animate-spin" /> Loading analysis...
    </div>

    <div v-else-if="error" class="error-state">
        <AlertTriangle class="inline h-4 w-4" /> {{ error }}
    </div>

    <div v-else-if="noAnalysis" class="no-analysis">
        <RouterLink :to="{ name: 'session-detail', params: { id: sessionId } }" class="back-link">
            <ArrowLeft class="inline h-4 w-4" /> Back to Session
        </RouterLink>
        <div class="empty-card border border-border bg-card">
            <BarChart3 class="empty-icon text-muted-foreground" />
            <p class="text-muted-foreground">No analysis results for this session.</p>
            <Button :disabled="!canWrite || running" @click="handleRunAnalysis">
                <Loader2 v-if="running" class="h-4 w-4 animate-spin" />
                <Play v-else class="h-4 w-4" />
                {{ canWrite ? 'Run Analysis' : 'Read-only mode' }}
            </Button>
        </div>
    </div>

    <div v-else-if="analysis" class="analysis-view">
        <RouterLink :to="{ name: 'session-detail', params: { id: sessionId } }" class="back-link">
            <ArrowLeft class="inline h-4 w-4" /> Back to Session
        </RouterLink>

        <h1 class="page-title">Analysis — Session #{{ sessionId }}</h1>

        <ExperimentalBanner
            class="mb-5"
            title="Experimental analysis."
            body="Everything SNORE detects on this page (events, indices, flow-limitation classes, breathing patterns) is its own heuristic scored from the flow waveform, not a clinically validated measurement. Device-scored events are the reference."
        />

        <!-- Summary -->
        <div class="summary-row">
            <StatCard
                label="Duration"
                :value="analysis.session_duration_hours"
                unit="hrs"
                :decimals="1"
                glossary-key="session_duration_hours"
                :provenance="MARKS.sessionDuration"
            />
            <StatCard
                label="Total Breaths"
                :value="analysis.total_breaths"
                :decimals="0"
                glossary-key="total_breaths"
                :provenance="MARKS.totalBreaths"
            />
            <StatCard
                label="Device-scored Events"
                :value="analysis.machine_events?.length ?? 0"
                :decimals="0"
                glossary-key="machine_events"
            />
            <StatCard
                v-if="analysis.pulse_change_count != null"
                label="Pulse Changes"
                :value="analysis.pulse_change_count"
                :decimals="0"
                glossary-key="pulse_change_count"
                :provenance="MARKS.pulseChanges"
            />
        </div>

        <!-- Mode Comparison Table -->
        <div class="section-card">
            <h2>Mode Comparison</h2>
            <div v-if="isMobile" class="card-list">
                <div v-for="(row, i) in modeRows" :key="i" class="data-card">
                    <div class="data-card-header">{{ row.mode }}</div>
                    <div class="data-card-row">
                        <span class="data-card-label"
                            >AHI <ProvenanceMark :provenance="MARKS.modeAhi" />
                            <InfoHint glossary-key="ahi"
                        /></span>
                        <span class="data-card-value"
                            ><strong>{{ row.ahi.toFixed(1) }}</strong></span
                        >
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label"
                            >RDI <ProvenanceMark :provenance="MARKS.rdi" />
                            <InfoHint glossary-key="rdi"
                        /></span>
                        <span class="data-card-value">{{ row.rdi.toFixed(1) }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label"
                            >Apneas <ProvenanceMark :provenance="MARKS.snoreDetected" />
                            <InfoHint glossary-key="apneas"
                        /></span>
                        <span class="data-card-value">{{ row.apneas }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label"
                            >Hypopneas <ProvenanceMark :provenance="MARKS.snoreDetected" />
                            <InfoHint glossary-key="hypopneas"
                        /></span>
                        <span class="data-card-value">{{ row.hypopneas }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label"
                            >RERAs <ProvenanceMark :provenance="MARKS.snoreDetected" />
                            <InfoHint glossary-key="reras"
                        /></span>
                        <span class="data-card-value">{{ row.reras }}</span>
                    </div>
                </div>
            </div>
            <Table v-else>
                <TableHeader>
                    <TableRow>
                        <TableHead>Mode</TableHead>
                        <TableHead class="whitespace-nowrap" style="width: 80px">
                            AHI <ProvenanceMark :provenance="MARKS.modeAhi" />
                            <InfoHint glossary-key="ahi" />
                        </TableHead>
                        <TableHead class="whitespace-nowrap" style="width: 80px">
                            RDI <ProvenanceMark :provenance="MARKS.rdi" />
                            <InfoHint glossary-key="rdi" />
                        </TableHead>
                        <TableHead class="whitespace-nowrap" style="width: 80px">
                            Apneas <ProvenanceMark :provenance="MARKS.snoreDetected" />
                            <InfoHint glossary-key="apneas" />
                        </TableHead>
                        <TableHead class="whitespace-nowrap" style="width: 100px">
                            Hypopneas <ProvenanceMark :provenance="MARKS.snoreDetected" />
                            <InfoHint glossary-key="hypopneas" />
                        </TableHead>
                        <TableHead class="whitespace-nowrap" style="width: 80px">
                            RERAs <ProvenanceMark :provenance="MARKS.snoreDetected" />
                            <InfoHint glossary-key="reras" />
                        </TableHead>
                    </TableRow>
                </TableHeader>
                <TableBody>
                    <TableRow v-for="(row, i) in modeRows" :key="i" class="odd:bg-muted/50">
                        <TableCell>{{ row.mode }}</TableCell>
                        <TableCell
                            ><strong>{{ row.ahi.toFixed(1) }}</strong></TableCell
                        >
                        <TableCell>{{ row.rdi.toFixed(1) }}</TableCell>
                        <TableCell>{{ row.apneas }}</TableCell>
                        <TableCell>{{ row.hypopneas }}</TableCell>
                        <TableCell>{{ row.reras }}</TableCell>
                    </TableRow>
                </TableBody>
            </Table>
        </div>

        <!-- Per-mode Events -->
        <div class="section-card">
            <h2>
                SNORE-detected Events by Mode <ProvenanceMark :provenance="MARKS.snoreDetected" />
            </h2>
            <ToggleGroup
                :model-value="selectedMode"
                type="single"
                variant="outline"
                class="mode-selector"
                @update:model-value="
                    (v) => {
                        if (v) selectedMode = v as string
                    }
                "
            >
                <ToggleGroupItem v-for="mode in modeOptions" :key="mode" :value="mode">
                    {{ mode }}
                </ToggleGroupItem>
            </ToggleGroup>

            <div v-if="selectedModeResult">
                <div v-if="isMobile" class="card-list">
                    <div v-for="(row, i) in paginatedEvents" :key="i" class="data-card">
                        <div class="data-card-header">
                            <span
                                class="event-badge"
                                :style="{ background: EVENT_COLORS[row.type] ?? '#ddd' }"
                            >
                                {{ row.type }}
                            </span>
                            <span class="mobile-card-time">{{ formatTimeOffset(row.start) }}</span>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label"
                                >Duration <ProvenanceMark :provenance="MARKS.snoreEventDuration"
                            /></span>
                            <span class="data-card-value">{{ row.duration.toFixed(1) }}s</span>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label"
                                >Flow Red. <ProvenanceMark :provenance="MARKS.flowReduction" />
                                <InfoHint glossary-key="flow_reduction"
                            /></span>
                            <span class="data-card-value"
                                >{{ (row.flowReduction * 100).toFixed(0) }}%</span
                            >
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label"
                                >Confidence <ProvenanceMark :provenance="MARKS.confidence" />
                                <InfoHint glossary-key="confidence"
                            /></span>
                            <span class="data-card-value"
                                >{{ (row.confidence * 100).toFixed(0) }}%</span
                            >
                        </div>
                    </div>
                </div>
                <Table v-else>
                    <TableHeader>
                        <TableRow>
                            <TableHead style="width: 80px">Type</TableHead>
                            <TableHead>Start Time</TableHead>
                            <TableHead class="whitespace-nowrap" style="width: 90px">
                                Duration <ProvenanceMark :provenance="MARKS.snoreEventDuration" />
                            </TableHead>
                            <TableHead class="whitespace-nowrap" style="width: 100px">
                                Flow Red. <ProvenanceMark :provenance="MARKS.flowReduction" />
                                <InfoHint glossary-key="flow_reduction" />
                            </TableHead>
                            <TableHead class="whitespace-nowrap" style="width: 100px">
                                Confidence <ProvenanceMark :provenance="MARKS.confidence" />
                                <InfoHint glossary-key="confidence" />
                            </TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        <TableRow
                            v-for="(row, i) in paginatedEvents"
                            :key="i"
                            class="odd:bg-muted/50"
                        >
                            <TableCell>
                                <span
                                    class="event-badge"
                                    :style="{ background: EVENT_COLORS[row.type] ?? '#ddd' }"
                                >
                                    {{ row.type }}
                                </span>
                            </TableCell>
                            <TableCell>{{ formatTimeOffset(row.start) }}</TableCell>
                            <TableCell>{{ row.duration.toFixed(1) }}s</TableCell>
                            <TableCell>{{ (row.flowReduction * 100).toFixed(0) }}%</TableCell>
                            <TableCell>{{ (row.confidence * 100).toFixed(0) }}%</TableCell>
                        </TableRow>
                    </TableBody>
                </Table>

                <div v-if="totalEventPages > 1" class="flex items-center justify-between px-2 py-4">
                    <span class="text-sm text-muted-foreground">
                        Page {{ eventsPage + 1 }} of {{ totalEventPages }}
                    </span>
                    <div class="flex gap-2">
                        <Button
                            variant="outline"
                            size="sm"
                            :disabled="eventsPage === 0"
                            @click="eventsPage--"
                        >
                            Previous
                        </Button>
                        <Button
                            variant="outline"
                            size="sm"
                            :disabled="eventsPage >= totalEventPages - 1"
                            @click="eventsPage++"
                        >
                            Next
                        </Button>
                    </div>
                </div>
            </div>
        </div>

        <!-- CSR / Periodic Breathing -->
        <div v-if="analysis.csr_detection || analysis.periodic_breathing" class="section-card">
            <h2>Breathing Patterns</h2>
            <div v-if="analysis.csr_episodes?.length" class="pattern-info">
                <strong>CSR Episodes:</strong>
                <ProvenanceMark :provenance="MARKS.snoreDetected" />
                <InfoHint glossary-key="csr" />
                {{ analysis.csr_episodes.length }}
            </div>
            <div v-if="analysis.periodic_breathing_episodes?.length" class="pattern-info">
                <strong>Periodic Breathing Episodes:</strong>
                <ProvenanceMark :provenance="MARKS.snoreDetected" />
                <InfoHint glossary-key="periodic_breathing" />
                {{ analysis.periodic_breathing_episodes.length }}
            </div>
        </div>

        <!-- Flow Limitation Analysis -->
        <div v-if="flowAnalysis" class="section-card">
            <h2>Flow Limitation</h2>
            <div class="summary-row" style="margin-bottom: 1rem">
                <StatCard
                    label="Flow Limitation Index"
                    :value="flowAnalysis!.flow_limitation_index * 100"
                    unit="%"
                    :decimals="1"
                    glossary-key="flow_limitation_index"
                    :provenance="MARKS.flowLimitationIndex"
                />
                <StatCard
                    label="Total Breaths"
                    :value="flowAnalysis!.total_breaths"
                    :decimals="0"
                    glossary-key="total_breaths"
                    :provenance="MARKS.totalBreaths"
                />
                <StatCard
                    label="Avg Confidence"
                    :value="flowAnalysis!.average_confidence * 100"
                    unit="%"
                    :decimals="1"
                    glossary-key="avg_confidence"
                    :provenance="MARKS.avgConfidence"
                />
            </div>
            <div v-if="isMobile" class="card-list">
                <div v-for="cls in flDistributionRows" :key="cls.classNum" class="data-card">
                    <div class="data-card-header">
                        <span class="flex items-center gap-2">
                            <FlowClassPopover :fl-class="cls" />
                            <span>Class {{ cls.classNum }}: {{ cls.name }}</span>
                        </span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">Severity</span>
                        <span class="data-card-value">{{ cls.severity }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label"
                            >Count <ProvenanceMark :provenance="MARKS.snoreDetected"
                        /></span>
                        <span class="data-card-value">{{ cls.count }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label"
                            >% Breaths <ProvenanceMark :provenance="MARKS.snoreDetected"
                        /></span>
                        <span class="data-card-value">{{ cls.pct.toFixed(1) }}%</span>
                    </div>
                </div>
            </div>
            <Table v-else>
                <TableHeader>
                    <TableRow>
                        <TableHead style="width: 60px">Shape</TableHead>
                        <TableHead style="width: 60px">Class</TableHead>
                        <TableHead>Name</TableHead>
                        <TableHead style="width: 80px">Severity</TableHead>
                        <TableHead class="whitespace-nowrap" style="width: 80px">
                            Count <ProvenanceMark :provenance="MARKS.snoreDetected" />
                        </TableHead>
                        <TableHead class="whitespace-nowrap" style="width: 100px">
                            % Breaths <ProvenanceMark :provenance="MARKS.snoreDetected" />
                        </TableHead>
                    </TableRow>
                </TableHeader>
                <TableBody>
                    <TableRow
                        v-for="cls in flDistributionRows"
                        :key="cls.classNum"
                        class="odd:bg-muted/50"
                    >
                        <TableCell>
                            <FlowClassPopover :fl-class="cls" />
                        </TableCell>
                        <TableCell>{{ cls.classNum }}</TableCell>
                        <TableCell>{{ cls.name }}</TableCell>
                        <TableCell>{{ cls.severity }}</TableCell>
                        <TableCell>{{ cls.count }}</TableCell>
                        <TableCell>{{ cls.pct.toFixed(1) }}%</TableCell>
                    </TableRow>
                </TableBody>
            </Table>

            <!-- FL class legend -->
            <Collapsible v-model:open="flLegendOpen" class="mt-4">
                <CollapsibleTrigger as-child>
                    <button
                        type="button"
                        class="flex w-full items-center justify-between px-1 py-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
                    >
                        What do these classes mean?
                        <ChevronDown
                            class="h-4 w-4 transition-transform"
                            :class="{ 'rotate-180': flLegendOpen }"
                        />
                    </button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                    <div class="mt-2 border border-border rounded-md overflow-hidden">
                        <div
                            v-for="[num, info] in flLegendEntries"
                            :key="num"
                            class="flex items-start gap-3 px-3 py-2 text-sm border-t border-border first:border-t-0"
                        >
                            <div class="shrink-0 text-muted-foreground">
                                <FlowClassGlyph :class-num="Number(num)" size="lg" />
                            </div>
                            <div class="flex-1 min-w-0">
                                <div class="flex items-center gap-2 mb-0.5">
                                    <span class="font-medium"
                                        >Class {{ num }}: {{ info.name }}</span
                                    >
                                    <SeverityBadge :severity="info.severity" />
                                </div>
                                <p class="text-muted-foreground text-xs">{{ info.description }}</p>
                            </div>
                        </div>
                    </div>
                </CollapsibleContent>
            </Collapsible>
        </div>

        <!-- Event Comparison -->
        <div v-if="comparison" class="section-card">
            <h2>Event Comparison</h2>
            <div class="summary-row" style="margin-bottom: 1rem">
                <StatCard
                    label="Device-scored Events"
                    :value="comparison.machine_event_count"
                    :decimals="0"
                    glossary-key="machine_events"
                />
                <StatCard
                    label="SNORE-detected Events"
                    :value="comparison.programmatic_event_count"
                    :decimals="0"
                    glossary-key="programmatic_events"
                    :provenance="MARKS.programmaticEvents"
                />
                <StatCard
                    label="False Negatives"
                    :value="comparison.false_negatives?.length ?? 0"
                    :decimals="0"
                    glossary-key="false_negatives"
                    :provenance="MARKS.falseNegatives"
                />
                <StatCard
                    label="False Positives"
                    :value="
                        (comparison.false_positives_apnea?.length ?? 0) +
                        (comparison.false_positives_hypopnea?.length ?? 0)
                    "
                    :decimals="0"
                    glossary-key="false_positives"
                    :provenance="MARKS.falsePositives"
                />
            </div>

            <div v-if="comparison.false_negatives?.length" class="compare-table-section">
                <h3>False Negatives: device-scored events SNORE missed</h3>
                <div v-if="isMobile" class="card-list">
                    <div
                        v-for="(e, i) in comparison.false_negatives"
                        :key="'fn-' + i"
                        class="data-card"
                    >
                        <div class="data-card-header">
                            <span
                                class="event-badge"
                                :style="{ background: EVENT_COLORS[e.event_type] ?? '#ddd' }"
                                >{{ e.event_type }}</span
                            >
                            <RouterLink
                                :to="{
                                    name: 'session-detail',
                                    params: { id: sessionId },
                                    query: { t: e.start_time },
                                }"
                                class="mobile-card-time text-primary hover:underline"
                            >
                                {{ formatTimeOffset(e.start_time) }}
                            </RouterLink>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label">Duration</span>
                            <span class="data-card-value"
                                >{{ e.duration.toFixed(1) }}s
                                <ProvenanceMark :provenance="comparisonDurationProvenance(e)"
                            /></span>
                        </div>
                    </div>
                </div>
                <Table v-else>
                    <TableHeader>
                        <TableRow>
                            <TableHead style="width: 80px">Type</TableHead>
                            <TableHead>Time</TableHead>
                            <TableHead style="width: 90px">Duration</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        <TableRow
                            v-for="(e, i) in comparison.false_negatives"
                            :key="'fn-' + i"
                            class="odd:bg-muted/50"
                        >
                            <TableCell>
                                <span
                                    class="event-badge"
                                    :style="{ background: EVENT_COLORS[e.event_type] ?? '#ddd' }"
                                    >{{ e.event_type }}</span
                                >
                            </TableCell>
                            <TableCell>
                                <RouterLink
                                    :to="{
                                        name: 'session-detail',
                                        params: { id: sessionId },
                                        query: { t: e.start_time },
                                    }"
                                    class="text-primary hover:underline"
                                >
                                    {{ formatTimeOffset(e.start_time) }}
                                </RouterLink>
                            </TableCell>
                            <TableCell
                                >{{ e.duration.toFixed(1) }}s
                                <ProvenanceMark :provenance="comparisonDurationProvenance(e)"
                            /></TableCell>
                        </TableRow>
                    </TableBody>
                </Table>
            </div>

            <div
                v-if="
                    comparison.false_positives_apnea?.length ||
                    comparison.false_positives_hypopnea?.length
                "
                class="compare-table-section"
            >
                <h3>
                    False Positives: SNORE-detected events the device did not score
                    <ProvenanceMark :provenance="MARKS.snoreDetected" />
                </h3>
                <div v-if="isMobile" class="card-list">
                    <div v-for="(e, i) in allFalsePositives" :key="'fp-' + i" class="data-card">
                        <div class="data-card-header">
                            <span
                                class="event-badge"
                                :style="{ background: EVENT_COLORS[e.event_type] ?? '#ddd' }"
                                >{{ e.event_type }}</span
                            >
                            <RouterLink
                                :to="{
                                    name: 'session-detail',
                                    params: { id: sessionId },
                                    query: { t: e.start_time },
                                }"
                                class="mobile-card-time text-primary hover:underline"
                            >
                                {{ formatTimeOffset(e.start_time) }}
                            </RouterLink>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label">Duration</span>
                            <span class="data-card-value"
                                >{{ e.duration.toFixed(1) }}s
                                <ProvenanceMark :provenance="comparisonDurationProvenance(e)"
                            /></span>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label"
                                >Confidence <ProvenanceMark :provenance="MARKS.confidence"
                            /></span>
                            <span class="data-card-value">{{
                                e.confidence != null ? (e.confidence * 100).toFixed(0) + '%' : '---'
                            }}</span>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label"
                                >Flow Red. <ProvenanceMark :provenance="MARKS.flowReduction"
                            /></span>
                            <span class="data-card-value">{{
                                e.flow_reduction != null
                                    ? (e.flow_reduction * 100).toFixed(0) + '%'
                                    : '---'
                            }}</span>
                        </div>
                    </div>
                </div>
                <Table v-else>
                    <TableHeader>
                        <TableRow>
                            <TableHead style="width: 80px">Type</TableHead>
                            <TableHead>Time</TableHead>
                            <TableHead style="width: 90px">Duration</TableHead>
                            <TableHead class="whitespace-nowrap" style="width: 100px">
                                Confidence <ProvenanceMark :provenance="MARKS.confidence" />
                            </TableHead>
                            <TableHead class="whitespace-nowrap" style="width: 100px">
                                Flow Red. <ProvenanceMark :provenance="MARKS.flowReduction" />
                            </TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        <TableRow
                            v-for="(e, i) in allFalsePositives"
                            :key="'fp-' + i"
                            class="odd:bg-muted/50"
                        >
                            <TableCell>
                                <span
                                    class="event-badge"
                                    :style="{ background: EVENT_COLORS[e.event_type] ?? '#ddd' }"
                                    >{{ e.event_type }}</span
                                >
                            </TableCell>
                            <TableCell>
                                <RouterLink
                                    :to="{
                                        name: 'session-detail',
                                        params: { id: sessionId },
                                        query: { t: e.start_time },
                                    }"
                                    class="text-primary hover:underline"
                                >
                                    {{ formatTimeOffset(e.start_time) }}
                                </RouterLink>
                            </TableCell>
                            <TableCell
                                >{{ e.duration.toFixed(1) }}s
                                <ProvenanceMark :provenance="comparisonDurationProvenance(e)"
                            /></TableCell>
                            <TableCell>{{
                                e.confidence != null ? (e.confidence * 100).toFixed(0) + '%' : '---'
                            }}</TableCell>
                            <TableCell>{{
                                e.flow_reduction != null
                                    ? (e.flow_reduction * 100).toFixed(0) + '%'
                                    : '---'
                            }}</TableCell>
                        </TableRow>
                    </TableBody>
                </Table>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import { Button } from '@/components/ui/button'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Loader2, AlertTriangle, ArrowLeft, BarChart3, Play, ChevronDown } from '@lucide/vue'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import StatCard from '@/components/StatCard.vue'
import InfoHint from '@/components/InfoHint.vue'
import FlowClassGlyph from '@/components/FlowClassGlyph.vue'
import FlowClassPopover from '@/components/FlowClassPopover.vue'
import SeverityBadge from '@/components/SeverityBadge.vue'
import ExperimentalBanner from '@/components/ExperimentalBanner.vue'
import ProvenanceMark from '@/components/ProvenanceMark.vue'
import { getAnalysis, runAnalysis } from '@/api/analysis'
import { getWaveformCompare } from '@/api/waveforms'
import { useAuth } from '@/composables/useAuth'
import { useIsMobile } from '@/composables/useIsMobile'
import { formatTimeOffset } from '@/utils/formatting'
import { EVENT_COLORS } from '@/types'
import { FLOW_LIMITATION_CLASSES } from '@/utils/flowLimitation'
import { provenanceFor, type Provenance } from '@/utils/provenance'
import type { AnalysisResult, EventComparisonDetail, EventComparisonResult } from '@/types'

interface FlowAnalysis {
    total_breaths: number
    class_distribution: Record<string, number>
    flow_limitation_index: number
    average_confidence: number
}

// Tiers of the metrics this page labels. Mode results, their event lists, FL class
// counts, and CSR/PB episodes are all SNORE detections; the bare names `apneas`,
// `hypopneas`, `reras` map to device counts elsewhere, so those use the literal tier.
const MARKS = {
    snoreDetected: 'experimental' as Provenance,
    sessionDuration: provenanceFor('session_duration_hours'),
    totalBreaths: provenanceFor('total_breaths'),
    pulseChanges: provenanceFor('pulse_change_count'),
    modeAhi: provenanceFor('ahi', { schema: 'ModeResult' }),
    rdi: provenanceFor('rdi'),
    snoreEventDuration: provenanceFor('duration', { schema: 'ApneaEvent' }),
    flowReduction: provenanceFor('flow_reduction'),
    confidence: provenanceFor('confidence'),
    flowLimitationIndex: provenanceFor('flow_limitation_index'),
    avgConfidence: provenanceFor('avg_confidence'),
    programmaticEvents: provenanceFor('programmatic_event_count'),
    falseNegatives: provenanceFor('false_negatives'),
    falsePositives: provenanceFor('false_positives'),
}

function comparisonDurationProvenance(e: EventComparisonDetail): Provenance {
    return provenanceFor('duration', { source: e.source, schema: 'EventComparisonDetail' })
}

const { canWrite } = useAuth()
const { isMobile } = useIsMobile()

const props = defineProps<{ sessionId: number }>()

const loading = ref(true)
const error = ref<string | null>(null)
const noAnalysis = ref(false)
const running = ref(false)
const analysis = ref<AnalysisResult | null>(null)
const comparison = ref<EventComparisonResult | null>(null)
const selectedMode = ref('')
const eventsPage = ref(0)
const eventsPageSize = 25
const flLegendOpen = ref(false)

const modeOptions = computed(() => Object.keys(analysis.value?.mode_results ?? {}))

const selectedModeResult = computed(() => {
    if (!analysis.value || !selectedMode.value) return null
    return analysis.value.mode_results[selectedMode.value] ?? null
})

interface ModeRow {
    mode: string
    ahi: number
    rdi: number
    apneas: number
    hypopneas: number
    reras: number
}

const modeRows = computed<ModeRow[]>(() => {
    if (!analysis.value) return []
    return Object.entries(analysis.value.mode_results ?? {}).map(([name, r]) => ({
        mode: name,
        ahi: r.ahi,
        rdi: r.rdi,
        apneas: r.apneas.length,
        hypopneas: r.hypopneas.length,
        reras: r.reras?.length ?? 0,
    }))
})

interface EventRow {
    type: string
    start: number
    duration: number
    flowReduction: number
    confidence: number
}

const modeEvents = computed<EventRow[]>(() => {
    const r = selectedModeResult.value
    if (!r) return []
    const events: EventRow[] = []
    for (const a of r.apneas ?? []) {
        events.push({
            type: a.event_type,
            start: a.start_time,
            duration: a.duration,
            flowReduction: a.flow_reduction,
            confidence: a.confidence,
        })
    }
    for (const h of r.hypopneas ?? []) {
        events.push({
            type: 'H',
            start: h.start_time,
            duration: h.duration,
            flowReduction: h.flow_reduction,
            confidence: h.confidence,
        })
    }
    for (const re of r.reras ?? []) {
        events.push({
            type: 'RE',
            start: re.start_time,
            duration: re.duration,
            flowReduction: 0,
            confidence: re.confidence,
        })
    }
    events.sort((a, b) => a.start - b.start)
    return events
})

const paginatedEvents = computed(() => {
    const start = eventsPage.value * eventsPageSize
    return modeEvents.value.slice(start, start + eventsPageSize)
})

const totalEventPages = computed(() => Math.ceil(modeEvents.value.length / eventsPageSize))

interface FlDistRow {
    classNum: number
    name: string
    severity: string
    count: number
    pct: number
}

const flowAnalysis = computed<FlowAnalysis | null>(() => {
    const fa = analysis.value?.flow_analysis as FlowAnalysis | null | undefined
    return fa ?? null
})

const flDistributionRows = computed<FlDistRow[]>(() => {
    const fa = flowAnalysis.value
    if (!fa) return []
    const total = fa.total_breaths || 1
    return Object.entries(fa.class_distribution)
        .map(([k, count]) => {
            const classNum = parseInt(k)
            const info = FLOW_LIMITATION_CLASSES[classNum]
            return {
                classNum,
                name: info?.name ?? `Class ${classNum}`,
                severity: info?.severity ?? '—',
                count,
                pct: (count / total) * 100,
            }
        })
        .sort((a, b) => a.classNum - b.classNum)
})

const flLegendEntries = computed(() =>
    Object.entries(FLOW_LIMITATION_CLASSES).sort(([a], [b]) => Number(a) - Number(b)),
)

const allFalsePositives = computed(() =>
    [
        ...(comparison.value?.false_positives_apnea ?? []),
        ...(comparison.value?.false_positives_hypopnea ?? []),
    ].sort((a, b) => a.start_time - b.start_time),
)

watch(selectedMode, () => {
    eventsPage.value = 0
})

async function handleRunAnalysis(): Promise<void> {
    running.value = true
    try {
        analysis.value = await runAnalysis(props.sessionId)
        noAnalysis.value = false
        if (modeOptions.value.length > 0) {
            selectedMode.value = modeOptions.value[0]
        }
    } catch (err: unknown) {
        error.value = err instanceof Error ? err.message : 'Failed to run analysis'
    } finally {
        running.value = false
    }
}

// useApiLoad skipped — 404 routes to noAnalysis state, not generic error
onMounted(async () => {
    try {
        analysis.value = await getAnalysis(props.sessionId)
        if (modeOptions.value.length > 0) {
            selectedMode.value = modeOptions.value[0]
        }
    } catch (err: unknown) {
        const status = (err as { response?: { status?: number } }).response?.status
        if (status === 404) {
            noAnalysis.value = true
        } else {
            error.value = err instanceof Error ? err.message : 'Failed to load analysis'
        }
    } finally {
        loading.value = false
    }

    try {
        comparison.value = await getWaveformCompare(props.sessionId)
    } catch {
        // Comparison data not available — section won't render
    }
})
</script>

<style scoped>
.analysis-view,
.no-analysis {
    max-width: 1200px;
}

.summary-row {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}

.mode-selector {
    margin-bottom: 0.75rem;
}

.empty-card {
    text-align: center;
    padding: 3rem;
    border-radius: 8px;
}

.empty-icon {
    width: 2.5rem;
    height: 2.5rem;
    margin: 0 auto 1rem;
    display: block;
}

.empty-card p {
    margin-bottom: 1rem;
}

.event-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.pattern-info {
    font-size: 0.9rem;
    padding: 0.4rem 0;
}

.compare-table-section {
    margin-top: 1rem;
}

.compare-table-section h3 {
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* Supplement to the shared .data-card-header: times render lighter and smaller
   than the bold header text; space-between already pushes them right. */
.mobile-card-time {
    font-weight: 400;
    font-size: 0.875rem;
}
</style>

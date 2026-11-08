<template>
    <div v-if="loading" class="loading-state">
        <Loader2 class="inline h-4 w-4 animate-spin" /> Loading analysis...
    </div>

    <div v-else-if="error" class="error-state">
        <AlertTriangle class="inline h-4 w-4" /> {{ error }}
    </div>

    <div v-else-if="!shown" class="no-analysis">
        <RouterLink :to="{ name: 'session-detail', params: { id: sessionId } }" class="back-link">
            <ArrowLeft class="inline h-4 w-4" /> Back to Session
        </RouterLink>
        <div class="empty-card border border-border bg-card">
            <BarChart3 class="empty-icon text-muted-foreground" />
            <p class="text-muted-foreground">No analysis results for this session.</p>
            <Button :disabled="!canWrite" @click="openRunDialog">
                <Play class="h-4 w-4" />
                {{ canWrite ? 'Run Analysis' : 'Read-only mode' }}
            </Button>
        </div>
    </div>

    <div v-else class="analysis-view">
        <RouterLink :to="{ name: 'session-detail', params: { id: sessionId } }" class="back-link">
            <ArrowLeft class="inline h-4 w-4" /> Back to Session
        </RouterLink>

        <div class="page-header">
            <h1 class="page-title">Analysis — Session #{{ sessionId }}</h1>
            <Button v-if="canWrite" variant="outline" size="sm" @click="openRunDialog">
                <Play class="mr-2 h-4 w-4" />
                Run Analysis
            </Button>
        </div>

        <ExperimentalBanner
            class="mb-5"
            title="Experimental analysis."
            body="Everything SNORE detects on this page (events, indices, flow-limitation classes, breathing patterns) is its own heuristic, not a clinically validated measurement. Device-scored events are the reference."
        />

        <div v-if="preview" role="status" class="preview-notice">
            <p>
                <span class="font-medium">{{
                    showingPreview ? 'Showing an unsaved preview.' : 'Showing saved results.'
                }}</span>
                The preview ({{ previewModes }}) was not saved and is lost when you leave this
                page.<template v-if="!analysis"> This session has no saved analysis.</template>
            </p>
            <div class="preview-actions">
                <ToggleGroup
                    v-if="analysis"
                    type="single"
                    variant="outline"
                    size="sm"
                    aria-label="Results shown"
                    :model-value="showingPreview ? 'preview' : 'saved'"
                    @update:model-value="
                        (v) => {
                            if (v) previewSelected = v === 'preview'
                        }
                    "
                >
                    <ToggleGroupItem value="saved">Saved results</ToggleGroupItem>
                    <ToggleGroupItem value="preview">Unsaved preview</ToggleGroupItem>
                </ToggleGroup>
                <Button variant="ghost" size="sm" @click="preview = null">Discard preview</Button>
            </div>
        </div>

        <!-- Summary -->
        <div class="summary-row">
            <StatCard
                label="Duration"
                :value="shown.session_duration_hours"
                unit="hrs"
                :decimals="1"
                glossary-key="session_duration_hours"
            />
            <StatCard
                label="Total Breaths"
                :value="shown.total_breaths"
                :decimals="0"
                glossary-key="total_breaths"
            />
            <StatCard
                label="Device-scored Events"
                :value="shown.machine_events?.length ?? 0"
                :decimals="0"
                glossary-key="machine_events"
            />
            <StatCard
                v-if="shown.pulse_change_count != null"
                label="Pulse Changes"
                :value="shown.pulse_change_count"
                :decimals="0"
                glossary-key="pulse_change_count"
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
                            >AHI <InfoHint glossary-key="mode_ahi"
                        /></span>
                        <span class="data-card-value"
                            ><strong>{{ row.ahi.toFixed(1) }}</strong></span
                        >
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">RDI <InfoHint glossary-key="rdi" /></span>
                        <span class="data-card-value">{{ row.rdi.toFixed(1) }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label"
                            >Apneas <InfoHint glossary-key="apneas"
                        /></span>
                        <span class="data-card-value">{{ row.apneas }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label"
                            >Hypopneas <InfoHint glossary-key="hypopneas"
                        /></span>
                        <span class="data-card-value">{{ row.hypopneas }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">RERAs <InfoHint glossary-key="reras" /></span>
                        <span class="data-card-value">{{ row.reras }}</span>
                    </div>
                </div>
            </div>
            <Table v-else>
                <TableHeader>
                    <TableRow>
                        <TableHead>Mode</TableHead>
                        <TableHead class="whitespace-nowrap" style="width: 80px">
                            AHI <InfoHint glossary-key="mode_ahi" />
                        </TableHead>
                        <TableHead class="whitespace-nowrap" style="width: 80px">
                            RDI <InfoHint glossary-key="rdi" />
                        </TableHead>
                        <TableHead class="whitespace-nowrap" style="width: 80px">
                            Apneas <InfoHint glossary-key="apneas" />
                        </TableHead>
                        <TableHead class="whitespace-nowrap" style="width: 100px">
                            Hypopneas <InfoHint glossary-key="hypopneas" />
                        </TableHead>
                        <TableHead class="whitespace-nowrap" style="width: 80px">
                            RERAs <InfoHint glossary-key="reras" />
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
            <h2>Events by Mode</h2>
            <ToggleGroup
                :model-value="activeMode"
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
                            <span class="data-card-label">Duration</span>
                            <span class="data-card-value">{{ row.duration.toFixed(1) }}s</span>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label"
                                >Flow Red. <InfoHint glossary-key="flow_reduction"
                            /></span>
                            <span class="data-card-value"
                                >{{ (row.flowReduction * 100).toFixed(0) }}%</span
                            >
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label"
                                >Confidence <InfoHint glossary-key="confidence"
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
                            <TableHead style="width: 90px">Duration</TableHead>
                            <TableHead class="whitespace-nowrap" style="width: 100px">
                                Flow Red. <InfoHint glossary-key="flow_reduction" />
                            </TableHead>
                            <TableHead class="whitespace-nowrap" style="width: 100px">
                                Confidence <InfoHint glossary-key="confidence" />
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

                <PaginationBar v-bind="eventsBar" @page="setEventsOffset" />
            </div>
        </div>

        <!-- CSR / Periodic Breathing -->
        <div v-if="shown.csr_detection || shown.periodic_breathing" class="section-card">
            <h2>Breathing Patterns</h2>
            <div v-if="shown.csr_episodes?.length" class="pattern-info">
                <strong>CSR Episodes:</strong>
                <InfoHint glossary-key="csr" />
                {{ shown.csr_episodes.length }}
            </div>
            <div v-if="shown.periodic_breathing_episodes?.length" class="pattern-info">
                <strong>Periodic Breathing Episodes:</strong>
                <InfoHint glossary-key="periodic_breathing" />
                {{ shown.periodic_breathing_episodes.length }}
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
                />
                <StatCard
                    label="Total Breaths"
                    :value="flowAnalysis!.total_breaths"
                    :decimals="0"
                    glossary-key="total_breaths"
                />
                <StatCard
                    label="Avg Confidence"
                    :value="flowAnalysis!.average_confidence * 100"
                    unit="%"
                    :decimals="1"
                    glossary-key="avg_confidence"
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
                        <span class="data-card-label">Count</span>
                        <span class="data-card-value">{{ cls.count }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">% Breaths</span>
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
                        <TableHead style="width: 80px">Count</TableHead>
                        <TableHead style="width: 80px">% Breaths</TableHead>
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

        <!-- Event Comparison: the compare endpoint reads the saved analysis, so a preview has none -->
        <div v-if="showingPreview" class="section-card">
            <h2>Event Comparison</h2>
            <p class="text-sm text-muted-foreground">
                Event comparison is available for saved results only.
            </p>
        </div>
        <div v-else-if="comparisonLoading && !comparison" class="loading-state">
            <Loader2 class="inline h-4 w-4 animate-spin" /> Loading event comparison...
        </div>
        <ErrorState
            v-else-if="comparisonError"
            :message="comparisonError"
            :retry="loadComparison"
        />
        <div v-else-if="comparison" class="section-card">
            <h2>Event Comparison</h2>
            <p class="mb-3 text-sm text-muted-foreground">Detection mode: {{ comparison.mode }}</p>
            <div class="summary-row" style="margin-bottom: 1rem">
                <StatCard
                    label="Device-scored Events"
                    :value="comparison.machine_event_count"
                    :decimals="0"
                    glossary-key="machine_events"
                    >Apneas and hypopneas, plus RERAs when the device scored any</StatCard
                >
                <StatCard
                    label="SNORE-detected Events"
                    :value="comparison.programmatic_event_count"
                    :decimals="0"
                    glossary-key="programmatic_events"
                />
                <StatCard label="Matched" :value="comparison.matched_events" :decimals="0" />
                <StatCard
                    label="Sensitivity"
                    :value="sensitivity"
                    unit="%"
                    :decimals="1"
                    glossary-key="sensitivity"
                />
                <StatCard
                    label="False Negatives"
                    :value="comparison.false_negatives?.length ?? 0"
                    :decimals="0"
                    glossary-key="false_negatives"
                />
                <StatCard
                    label="False Positives"
                    :value="
                        (comparison.false_positives_apnea?.length ?? 0) +
                        (comparison.false_positives_hypopnea?.length ?? 0) +
                        (comparison.false_positives_rera?.length ?? 0)
                    "
                    :decimals="0"
                    glossary-key="false_positives"
                />
            </div>

            <div v-if="comparison.false_negatives?.length" class="compare-table-section">
                <h3>False Negatives: Device-scored Events SNORE Missed</h3>
                <div v-if="isMobile" class="card-list">
                    <div v-for="(e, i) in pagedFalseNegatives" :key="'fn-' + i" class="data-card">
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
                            <span class="data-card-value">{{ formatDuration(e.duration) }}</span>
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
                            v-for="(e, i) in pagedFalseNegatives"
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
                            <TableCell>{{ formatDuration(e.duration) }}</TableCell>
                        </TableRow>
                    </TableBody>
                </Table>
                <PaginationBar v-bind="falseNegativesBar" @page="setFalseNegativesOffset" />
            </div>

            <div
                v-if="
                    comparison.false_positives_apnea?.length ||
                    comparison.false_positives_hypopnea?.length ||
                    comparison.false_positives_rera?.length
                "
                class="compare-table-section"
            >
                <h3>False Positives: SNORE-detected Events the Device Did Not Score</h3>
                <div v-if="isMobile" class="card-list">
                    <div v-for="(e, i) in pagedFalsePositives" :key="'fp-' + i" class="data-card">
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
                            <span class="data-card-value">{{ e.duration.toFixed(1) }}s</span>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label">Confidence</span>
                            <span class="data-card-value">{{
                                e.confidence != null ? (e.confidence * 100).toFixed(0) + '%' : '---'
                            }}</span>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label">Flow Red.</span>
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
                            <TableHead style="width: 100px">Confidence</TableHead>
                            <TableHead style="width: 100px">Flow Red.</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        <TableRow
                            v-for="(e, i) in pagedFalsePositives"
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
                            <TableCell>{{ e.duration.toFixed(1) }}s</TableCell>
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
                <PaginationBar v-bind="falsePositivesBar" @page="setFalsePositivesOffset" />
            </div>
        </div>
    </div>

    <AlertDialog :open="runDialogOpen" @update:open="runDialogOpen = $event">
        <AlertDialogContent class="max-w-[450px]">
            <AlertDialogHeader>
                <AlertDialogTitle>Run Analysis</AlertDialogTitle>
                <AlertDialogDescription as-template>
                    <div class="space-y-4">
                        <AnalysisModePicker
                            v-model:modes="runModes"
                            v-model:primary-mode="runPrimaryMode"
                        />
                        <div>
                            <Toggle v-model="saveResults" variant="outline"> Save results </Toggle>
                            <p class="mt-1 text-xs text-muted-foreground">
                                {{
                                    saveResults
                                        ? "The results become this session's saved analysis."
                                        : 'The results are shown as a preview and not saved.'
                                }}
                            </p>
                        </div>
                        <p v-if="runError" class="text-sm text-destructive">
                            <AlertTriangle class="inline h-4 w-4" /> {{ runError }}
                        </p>
                    </div>
                </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
                <AlertDialogCancel :disabled="running">Cancel</AlertDialogCancel>
                <Button :disabled="running || !runModes.length" @click="handleRunAnalysis">
                    <Loader2 v-if="running" class="mr-2 h-4 w-4 animate-spin" />
                    <Play v-else class="mr-2 h-4 w-4" />
                    Run
                </Button>
            </AlertDialogFooter>
        </AlertDialogContent>
    </AlertDialog>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { isAxiosError } from 'axios'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import { Button } from '@/components/ui/button'
import { Toggle } from '@/components/ui/toggle'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import {
    AlertDialog,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Loader2, AlertTriangle, ArrowLeft, BarChart3, Play, ChevronDown } from '@lucide/vue'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import StatCard from '@/components/StatCard.vue'
import InfoHint from '@/components/InfoHint.vue'
import ExperimentalBanner from '@/components/ExperimentalBanner.vue'
import FlowClassGlyph from '@/components/FlowClassGlyph.vue'
import FlowClassPopover from '@/components/FlowClassPopover.vue'
import SeverityBadge from '@/components/SeverityBadge.vue'
import PaginationBar from '@/components/PaginationBar.vue'
import ErrorState from '@/components/ErrorState.vue'
import AnalysisModePicker from '@/components/AnalysisModePicker.vue'
import { getAnalysis, runAnalysis } from '@/api/analysis'
import type { AnalysisMode } from '@/api/analysis'
import { getWaveformCompare } from '@/api/waveforms'
import { useAuth } from '@/composables/useAuth'
import { useApiLoad } from '@/composables/useApiLoad'
import { useIsMobile } from '@/composables/useIsMobile'
import { usePagedRows } from '@/composables/usePagedRows'
import { formatDuration, formatTimeOffset } from '@/utils/formatting'
import { EVENT_COLORS } from '@/types'
import { FLOW_LIMITATION_CLASSES } from '@/utils/flowLimitation'
import type { AnalysisResult, EventComparisonResult, SessionFlowAnalysis } from '@/types'

const { canWrite } = useAuth()
const { isMobile } = useIsMobile()

const props = defineProps<{ sessionId: number }>()

const loading = ref(true)
const error = ref<string | null>(null)
const analysis = ref<AnalysisResult | null>(null)
// The user's pick in the Events by Mode toggle; activeMode is what is shown.
const selectedMode = ref('')
const flLegendOpen = ref(false)

const runDialogOpen = ref(false)
const runModes = ref<AnalysisMode[]>(['aasm'])
const runPrimaryMode = ref<AnalysisMode>('aasm')
const saveResults = ref(true)
const running = ref(false)
const runError = ref<string | null>(null)

// An unsaved run's results. They sit beside the saved analysis rather than
// replacing it, so the saved results and their cached comparisons stay intact.
const preview = ref<AnalysisResult | null>(null)
const previewSelected = ref(false)
const showingPreview = computed(
    () => preview.value !== null && (previewSelected.value || analysis.value === null),
)
const shown = computed(() => (showingPreview.value ? preview.value : analysis.value))
const previewModes = computed(() => Object.keys(preview.value?.mode_results ?? {}).join(', '))

// Saved results per mode; cleared when the session changes or a saved run replaces them.
// A null entry is a mode the server has no comparison for (404).
const comparisonCache = new Map<string, EventComparisonResult | null>()

const modeOptions = computed(() => Object.keys(shown.value?.mode_results ?? {}))

// The picked mode when the shown results have it, else their first mode.
const activeMode = computed(() =>
    modeOptions.value.includes(selectedMode.value)
        ? selectedMode.value
        : (modeOptions.value[0] ?? ''),
)

const {
    data: comparisonData,
    loading: comparisonLoading,
    error: comparisonError,
    reload: loadComparison,
} = useApiLoad(async () => {
    const mode = activeMode.value
    if (!mode || showingPreview.value) return null
    const cached = comparisonCache.get(mode)
    if (cached !== undefined) return cached
    try {
        const result = await getWaveformCompare(props.sessionId, { mode })
        comparisonCache.set(mode, result)
        return result
    } catch (err: unknown) {
        // A 404 means no comparison for this mode: hide the card. Anything else is a failure.
        if (isAxiosError(err) && err.response?.status === 404) {
            comparisonCache.set(mode, null)
            return null
        }
        throw err
    }
}, 'Failed to load event comparison')

// Data for another mode stays hidden until its own load lands.
const comparison = computed(() =>
    comparisonData.value?.mode === activeMode.value ? comparisonData.value : null,
)

// Mount, running an analysis, the mode toggle and leaving a preview all land here.
watch([analysis, activeMode, showingPreview], () => void loadComparison())
watch(
    () => props.sessionId,
    () => {
        comparisonCache.clear()
        preview.value = null
    },
)

const sensitivity = computed(() => {
    if (!comparison.value?.machine_event_count) return null
    return (comparison.value.matched_events / comparison.value.machine_event_count) * 100
})

const selectedModeResult = computed(() => {
    if (!shown.value || !activeMode.value) return null
    return shown.value.mode_results[activeMode.value] ?? null
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
    if (!shown.value) return []
    return Object.entries(shown.value.mode_results ?? {}).map(([name, r]) => ({
        mode: name,
        ahi: r.ahi,
        rdi: r.rdi,
        apneas: r.apnea_count,
        hypopneas: r.hypopnea_count,
        reras: r.rera_count,
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

const {
    pageRows: paginatedEvents,
    barProps: eventsBar,
    setOffset: setEventsOffset,
} = usePagedRows(modeEvents)

interface FlDistRow {
    classNum: number
    name: string
    severity: string
    count: number
    pct: number
}

const flowAnalysis = computed<SessionFlowAnalysis | null>(() => shown.value?.flow_analysis ?? null)

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
        ...(comparison.value?.false_positives_rera ?? []),
    ].sort((a, b) => a.start_time - b.start_time),
)

const sortedFalseNegatives = computed(() =>
    [...(comparison.value?.false_negatives ?? [])].sort((a, b) => a.start_time - b.start_time),
)

const {
    pageRows: pagedFalsePositives,
    barProps: falsePositivesBar,
    setOffset: setFalsePositivesOffset,
} = usePagedRows(allFalsePositives)

const {
    pageRows: pagedFalseNegatives,
    barProps: falseNegativesBar,
    setOffset: setFalseNegativesOffset,
} = usePagedRows(sortedFalseNegatives)

function openRunDialog(): void {
    runError.value = null
    runDialogOpen.value = true
}

async function handleRunAnalysis(): Promise<void> {
    const store = saveResults.value
    running.value = true
    runError.value = null
    try {
        const result = await runAnalysis(props.sessionId, {
            modes: runModes.value,
            primary_mode: runPrimaryMode.value,
            store_results: store,
        })
        if (store) {
            comparisonCache.clear()
            analysis.value = result
            preview.value = null
        } else {
            preview.value = result
            previewSelected.value = true
        }
        runDialogOpen.value = false
    } catch (err: unknown) {
        runError.value = err instanceof Error ? err.message : 'Failed to run analysis'
    } finally {
        running.value = false
    }
}

// useApiLoad skipped — 404 routes to the empty state, not generic error
onMounted(async () => {
    try {
        analysis.value = await getAnalysis(props.sessionId)
    } catch (err: unknown) {
        const status = (err as { response?: { status?: number } }).response?.status
        if (status !== 404) {
            error.value = err instanceof Error ? err.message : 'Failed to load analysis'
        }
    } finally {
        loading.value = false
    }
})
</script>

<style scoped>
.analysis-view,
.no-analysis {
    max-width: 1200px;
}

.page-header {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.5rem 1rem;
}

.preview-notice {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    padding: 0.75rem 1rem;
    border: 1px dashed var(--color-primary);
    border-radius: 8px;
    font-size: 0.875rem;
}

.preview-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
}

@media (max-width: 767.98px) {
    .page-header :deep(button),
    .preview-actions :deep(button) {
        min-height: var(--tap-target);
    }
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

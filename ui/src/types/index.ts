// API types are re-exported from the OpenAPI-generated definitions in
// ./generated.ts (regenerate with `just ui-generate-types`). Only types
// without a backend response model are hand-written below.
import type { components } from './generated'

type Schemas = components['schemas']

// Auth
export type AuthStatusResponse = Schemas['AuthStatusResponse']
export type UserInfo = Schemas['UserInfo']
export type ProfileInfo = Schemas['ProfileInfo']

export type ProfileResponse = Schemas['ProfileResponse']

// Pagination — generic over the concrete instantiations FastAPI generates
// (PaginatedResponse_SessionListItem_ etc.), so the envelope shape cannot drift.
export type PaginatedResponse<T> = Omit<Schemas['PaginatedResponse_SessionListItem_'], 'items'> & {
    items: T[]
}

// Sessions
export type SessionListItem = Schemas['SessionListItem']
export type SessionStatistics = Schemas['SessionStatistics']
export type SessionSetting = Schemas['SessionSetting']
export type SessionDetail = Schemas['SessionDetail']

// Waveforms
export type WaveformInfo = Schemas['WaveformInfo']
export type WaveformDataResponse = Schemas['WaveformDataResponse']

// Events
export type EventItem = Schemas['EventItem']
export type EventMatchResult = Schemas['EventMatchResult']

// Devices
export type DeviceInfo = Schemas['DeviceInfo']
export type DeviceDetail = Schemas['DeviceDetail']
export type DeviceUsageSummary = Schemas['DeviceUsageSummary']
export type SettingsChange = Schemas['SettingsChange']
export type SettingChangeEntry = Schemas['SettingChangeEntry']

// Delete preview
export type DeletePreview = Schemas['DeletePreview']

// Stats
export type TherapySummary = Schemas['TherapySummary']
export type EventTypeCount = Schemas['EventTypeCount']
export type PeriodStatistics = Schemas['PeriodStatistics']
export type DataRange = Schemas['DataRange']

// Days
export type DayListItem = Schemas['DayListItem']
export type DayDetail = Schemas['DayDetail']
export type DateListResponse = Schemas['DateListResponse']

// Equipment
export type MaskLogEntryResponse = Schemas['MaskLogEntryResponse']
export type MaskEpochResponse = Schemas['MaskEpochResponse']

// RX
export type RxPeriodResponse = Schemas['RxPeriodResponse']
export type RxComparisonResponse = Schemas['RxComparisonResponse']
export type RxSettingChange = Schemas['RxSettingChange']
export type RxChangesResponse = Schemas['RxChangesResponse']
export type RxAllResponse = Schemas['RxAllResponse']

// Analysis
export type AnalysisListItem = Schemas['AnalysisListItem']
export type AnalysisSessionDetail = Schemas['AnalysisSessionDetail']
export type AnalysisDeletePreview = Schemas['AnalysisDeletePreview']
export type AnalysisResult = Schemas['AnalysisResult']
export type AnalysisJobStatus = Schemas['AnalysisJobStatus']
export type AnalysisJobsListResponse = Schemas['AnalysisJobsListResponse']
export type AnalysisEvent = Schemas['AnalysisEvent']
export type ModeResult = Schemas['ModeResult']
export type ApneaEvent = Schemas['ApneaEvent']
export type HypopneaEvent = Schemas['HypopneaEvent']
export type RERAEvent = Schemas['RERAEvent']

// Apple Health
export type HealthNightSummaryRead = Schemas['HealthNightSummaryRead']
export type HealthNightDetailRead = Schemas['HealthNightDetailRead']
export type HealthSampleRead = Schemas['HealthSampleRead']
export type HealthImportResultSummary = Schemas['HealthImportResultSummary']

// Import
export type ImportSource = Schemas['ImportSource']
export type PipelineJobStatus = Schemas['PipelineJobStatus']
export type PipelineJobsListResponse = Schemas['PipelineJobsListResponse']
export type LinkedAnalysisSummary = Schemas['LinkedAnalysisSummary']
export type ImportResultSummary = Schemas['ImportResultSummary']
export type ImportSourceResultSummary = Schemas['ImportSourceResultSummary']

// ImportSourceResult and ImportResult are returned as SSE event data (not HTTP
// response models), so they are absent from the OpenAPI spec and hand-written here.
export interface ImportSourceResult {
    source: ImportSource
    imported: number
    skipped: number
    failed: number
    warnings: string[]
}

export interface ImportResult {
    total_imported: number
    total_skipped: number
    total_failed: number
    sources: ImportSourceResult[]
    warnings: string[]
}

// Database
export type DatabaseStatsPublic = Schemas['DatabaseStatsPublic']
export type VacuumResult = Schemas['VacuumResult']
export type ResetResult = Schemas['ResetResult']
export type DeleteDataResult = Schemas['DeleteDataResult']

// Validation — events (apnea/hypopnea)
export type ValidationReport = Schemas['ValidationReport']
export type AggregateMetrics = Schemas['AggregateMetrics']
export type SessionValidation = Schemas['SessionValidation']

// Validation — FL signal
export type FlValidationReport = Schemas['FlValidationReport']
export type FlAggregateMetrics = Schemas['FlAggregateMetrics']
export type FlSessionValidation = Schemas['FlSessionValidation']

// Validation — breath trends
export type BreathTrendsValidationReport = Schemas['BreathTrendsValidationReport']
export type BreathTrendsAggregateMetrics = Schemas['BreathTrendsAggregateMetrics']
export type BreathTrendsSessionValidation = Schemas['BreathTrendsSessionValidation']
export type ChannelComparison = Schemas['ChannelComparison']
export type ChannelAggregateMetrics = Schemas['ChannelAggregateMetrics']

// Validation — RERA proxy vs machine RE
export type ReraValidationReport = Schemas['ReraValidationReport']
export type ReraAggregateMetrics = Schemas['ReraAggregateMetrics']
export type ReraSessionValidation = Schemas['ReraSessionValidation']

// Validation — Apple Health cross-source correlation
export type AppleCrossValidationReport = Schemas['AppleCrossValidationReport']
export type AppleCrossAggregate = Schemas['AppleCrossAggregate']
export type AppleCrossNightRecord = Schemas['AppleCrossNightRecord']
export type PairCorrelation = Schemas['PairCorrelation']

// Persisted validation runs
export type ValidationRunStatus = Schemas['ValidationRunStatus']
export type ValidationRunDetail = Schemas['ValidationRunDetail']
export type ValidationRunsListResponse = Schemas['ValidationRunsListResponse']
export type ValidationRunRequest = Schemas['ValidationRunRequest']
// ValidatorType is a bare Literal on the backend, so openapi-typescript inlines
// it rather than emitting a named schema — derive it from the request field.
export type ValidatorType = ValidationRunRequest['validator_type']

// Waveform compare
export type EventComparisonResult = Schemas['EventComparisonResult']
export type EventComparisonDetail = Schemas['EventComparisonDetail']

// Bulk delete
export type BulkDeletePreviewRequest = Schemas['BulkDeletePreviewRequest']

// ---------------------------------------------------------------------------
// UI-only types below. TrendData and RecordsData are hand-written views of
// /stats/trends and /stats/records (generated: TrendsResponse, RecordsResponse).
// WaveformType, WAVEFORM_LABELS, EVENT_COLORS are pure UI constants.
// ---------------------------------------------------------------------------

// Waveform type and display labels
export type WaveformType =
    | 'flow'
    | 'pressure'
    | 'therapy_pressure'
    | 'epap'
    | 'leak'
    | 'mv'
    | 'rr'
    | 'tv'
    | 'spo2'
    | 'pulse'
    | 'fl'
    | 'snore'
    | 'ie_ratio'
    | 'ti'
    | 'pressure_hr'
    | 'trigger_cycle'

export const WAVEFORM_LABELS: Record<string, string> = {
    flow: 'Flow Rate',
    pressure: 'Pressure',
    therapy_pressure: 'Therapy Pressure',
    epap: 'EPAP',
    leak: 'Leak Rate',
    mv: 'Minute Ventilation',
    rr: 'Respiratory Rate',
    tv: 'Tidal Volume',
    spo2: 'SpO₂',
    pulse: 'Pulse Rate',
    // Device-reported channels — named to distinguish from SNORE's computed metrics
    fl: 'Flow Limitation (device)',
    snore: 'Snore (device)',
    ie_ratio: 'I:E Ratio',
    ti: 'Inspiratory Time',
    pressure_hr: 'Mask Pressure (25 Hz)',
    trigger_cycle: 'Trigger/Cycle (raw codes)',
}

export const WAVEFORM_GLOSSARY_MAP: Record<WaveformType, string> = {
    flow: 'flow',
    pressure: 'pressure',
    therapy_pressure: 'therapy_pressure',
    epap: 'epap',
    leak: 'leak',
    mv: 'mv',
    rr: 'resp_rate',
    tv: 'tidal_volume',
    spo2: 'spo2',
    pulse: 'pulse',
    fl: 'fl_device',
    snore: 'snore_device',
    ie_ratio: 'ie_ratio_waveform',
    ti: 'ti_waveform',
    pressure_hr: 'pressure_hr_waveform',
    trigger_cycle: 'trigger_cycle',
}

// Stats trend/records payloads (untyped on the backend)
export interface TrendData {
    ahi: [string, number | null][]
    usage: [string, number | null][]
    spo2: [string, number | null][]
    leak: [string, number | null][]
    pressure?: [string, number | null][]
    epap?: [string, number | null][]
    rr?: [string, number | null][]
    pulse?: [string, number | null][]
    mv?: [string, number | null][]
    oai?: [string, number | null][]
    cai?: [string, number | null][]
    hi?: [string, number | null][]
    rera?: [string, number | null][]
    total_sleep_hours?: [string, number | null][]
    sleep_efficiency?: [string, number | null][]
}

export interface RecordsData {
    [metric: string]: { best: [string, number][]; worst: [string, number][] }
}

// Event colors for waveform overlay
export const EVENT_COLORS: Record<string, string> = {
    OA: 'rgba(220, 38, 38, 0.25)', // red — Obstructive Apnea
    CA: 'rgba(37, 99, 235, 0.25)', // blue — Central Apnea
    MA: 'rgba(168, 85, 247, 0.25)', // purple — Mixed Apnea
    H: 'rgba(234, 179, 8, 0.25)', // yellow — Hypopnea
    RE: 'rgba(34, 197, 94, 0.25)', // green — RERA
    FL: 'rgba(249, 115, 22, 0.2)', // orange — Flow Limitation
}

// Solid variants for event marker lines/labels drawn on the chart canvas;
// EVENT_COLORS stays translucent for CSS band/badge backgrounds.
export const EVENT_SOLID_COLORS: Record<string, string> = {
    OA: 'rgba(220, 38, 38, 0.85)',
    CA: 'rgba(37, 99, 235, 0.85)',
    MA: 'rgba(168, 85, 247, 0.85)',
    H: 'rgba(234, 179, 8, 0.85)',
    RE: 'rgba(34, 197, 94, 0.85)',
    FL: 'rgba(249, 115, 22, 0.85)',
}

// About / build provenance (excluded from OpenAPI schema)
export interface AboutInfo {
    version: string
    git_sha: string
    build_time: string
    uptime_seconds: number
    auth_mode: string
    python_version: string
    sqlite_version: string
    update_pending: boolean
    update_pending_since: string | null
}

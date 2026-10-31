import { describe, expect, it } from 'vitest'
import * as analysisApi from '@/api/analysis'
import * as validationApi from '@/api/validation'
import * as validationRunsApi from '@/api/validationRuns'

const ALL_WRAPPERS = [
    ...Object.keys(analysisApi),
    ...Object.keys(validationApi),
    ...Object.keys(validationRunsApi),
    'getDayExperimental',
    'getExperimentalStats',
    'getWaveformCompare',
]
// Cancelling a post-import job reads no metrics, and ImportView needs it.
const EXPERIMENTAL_WRAPPERS = ALL_WRAPPERS.filter((name) => name !== 'cancelAnalysisJob')

const ALLOWED_SOURCES = [
    /^\.\.\/views\/(AnalysisView|AnalysisManagementView|ValidationView|DayExperimentalView)\.vue$/,
    /^\.\.\/components\/validation\//,
    /^\.\.\/composables\/useValidationRuns\.ts$/,
]

const SOURCES = import.meta.glob<string>(
    ['../views/*.vue', '../components/**/*.vue', '../composables/*.ts'],
    { query: '?raw', import: 'default', eager: true },
)

function findOffenders(): string[] {
    return Object.entries(SOURCES)
        .filter(([file]) => !ALLOWED_SOURCES.some((allowed) => allowed.test(file)))
        .flatMap(([file, source]) =>
            EXPERIMENTAL_WRAPPERS.filter((name) => new RegExp(`\\b${name}\\b`).test(source)).map(
                (name) => `${file}: ${name}`,
            ),
        )
}

describe('experimental boundary', () => {
    it('test_wrapper_and_source_lists_are_populated', () => {
        expect(EXPERIMENTAL_WRAPPERS).toContain('getAnalysis')
        expect(EXPERIMENTAL_WRAPPERS).not.toContain('cancelAnalysisJob')
        expect(Object.keys(SOURCES)).toContain('../views/DashboardView.vue')
    })

    it('test_default_ui_references_no_experimental_wrappers', () => {
        expect(findOffenders()).toEqual([])
    })
})

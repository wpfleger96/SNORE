import { beforeAll, describe, expect, it } from 'vitest'
import api from '@/api/client'
import experimentalOperations from './experimentalOperations.json'

// Cancelling a post-import job reads no metrics, and ImportView needs it.
const METRIC_FREE_WRAPPERS = ['cancelAnalysisJob']

const ALLOWED_SOURCES = [
    /^\.\.\/views\/(AnalysisView|AnalysisManagementView|ValidationView|DayExperimentalView)\.vue$/,
    /^\.\.\/components\/TherapyAssessmentCard\.vue$/,
    /^\.\.\/components\/validation\//,
    /^\.\.\/composables\/useValidationRuns\.ts$/,
]

// Allowlisted modules that read experimental data: default pages must not import them.
const EXPERIMENTAL_MODULES = /\buseValidationRuns\b|\/validation\/\w+\.vue/

const SOURCES = import.meta.glob<string>(
    ['../**/*.{vue,ts}', '!../api/**', '!../tests/**', '!../types/**'],
    { query: '?raw', import: 'default', eager: true },
)

const API_MODULES = import.meta.glob<Record<string, unknown>>(
    ['../api/*.ts', '!../api/client.ts'],
    { eager: true },
)

interface WrapperCall {
    name: string
    operation: string
}

/** Calls every exported API function with a fake transport to see which operation it hits. */
async function callEveryWrapper(): Promise<WrapperCall[]> {
    const calls: WrapperCall[] = []
    let name = ''
    api.defaults.adapter = async (config) => {
        calls.push({ name, operation: `${config.method?.toUpperCase()} ${config.url}` })
        return { data: null, status: 200, statusText: 'OK', headers: {}, config }
    }
    for (const module of Object.values(API_MODULES)) {
        for (const [exportName, wrapper] of Object.entries(module)) {
            if (typeof wrapper !== 'function') continue
            name = exportName
            // A helper that rejects these placeholders makes no request, so records nothing.
            await Promise.resolve()
                .then(() => wrapper('{}', '{}', '{}'))
                .catch(() => undefined)
        }
    }
    return calls
}

function findOffenders(sources: Record<string, string>, wrappers: string[]): string[] {
    return Object.entries(sources)
        .filter(([file]) => !ALLOWED_SOURCES.some((allowed) => allowed.test(file)))
        .flatMap(([file, source]) => [
            ...wrappers
                .filter((name) => new RegExp(`\\b${name}\\b`).test(source))
                .map((name) => `${file}: ${name}`),
            ...(source.match(EXPERIMENTAL_MODULES) ?? []).map((match) => `${file}: ${match}`),
        ])
}

describe('experimental boundary', () => {
    const operations = experimentalOperations.map((operation) =>
        operation.replace('/api/v1', '').replace(/\{\w+\}/g, '{}'),
    )
    let calls: WrapperCall[]
    let experimentalWrappers: string[]

    beforeAll(async () => {
        calls = await callEveryWrapper()
        experimentalWrappers = calls
            .filter(({ operation }) => operations.includes(operation))
            .map(({ name }) => name)
            .filter((name) => !METRIC_FREE_WRAPPERS.includes(name))
    })

    it('test_every_experimental_operation_has_a_wrapper', () => {
        const called = new Set(calls.map(({ operation }) => operation))
        expect(operations.filter((operation) => !called.has(operation))).toEqual([])
    })

    it('test_wrapper_and_source_lists_are_populated', () => {
        expect(experimentalWrappers).toEqual(
            expect.arrayContaining(['getAnalysis', 'getDayExperimental', 'getWaveformCompare']),
        )
        expect(experimentalWrappers).not.toContain('cancelAnalysisJob')
        expect(experimentalWrappers).not.toContain('getDay')
        expect(Object.keys(SOURCES)).toEqual(
            expect.arrayContaining([
                '../App.vue',
                '../router/index.ts',
                '../utils/formatting.ts',
                '../views/DashboardView.vue',
            ]),
        )
        expect(Object.keys(SOURCES)).not.toContain('../api/analysis.ts')
    })

    it('test_default_ui_references_no_experimental_wrappers', () => {
        expect(findOffenders(SOURCES, experimentalWrappers)).toEqual([])
    })

    it.each([
        "import { useValidationRuns } from '@/composables/useValidationRuns'",
        "import FlValidationPanel from '@/components/validation/FlValidationPanel.vue'",
        "import FlValidationPanel from './validation/FlValidationPanel.vue'",
    ])('test_default_page_importing_experimental_module_is_flagged: %s', (source) => {
        const offenders = findOffenders({ '../views/DashboardView.vue': source }, [])
        expect(offenders).toHaveLength(1)
    })
})

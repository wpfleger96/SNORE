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

// The router lazy-loads every view and holds route paths, not API calls.
const ROUTER = '../router/index.ts'

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

/** `../views/Foo.vue` -> `views/Foo`, the form an import specifier resolves to. */
function moduleId(file: string): string {
    return file.replace(/^\.\.\//, '').replace(/\.(vue|ts)$/, '')
}

/** Resolves an import specifier written in `file` to a `src`-relative module id. */
function resolveImport(file: string, specifier: string): string {
    const base = specifier.startsWith('@/')
        ? specifier.slice(2)
        : `${moduleId(file).split('/').slice(0, -1).join('/')}/${specifier}`
    const parts: string[] = []
    for (const part of base.split('/')) {
        if (part === '..') parts.pop()
        else if (part !== '.' && part !== '') parts.push(part)
    }
    return parts.join('/').replace(/\.(vue|ts)$/, '')
}

/** Modules that read experimental data: every allowlisted source except the views' router. */
function experimentalModuleIds(sources: Record<string, string>): Set<string> {
    return new Set(
        Object.keys(sources)
            .filter((file) => ALLOWED_SOURCES.some((allowed) => allowed.test(file)))
            .map(moduleId),
    )
}

/** A raw client call (`client.get('/path')`) to an experimental API path; route links are not calls. */
function pathPattern(operation: string): RegExp {
    const path = operation
        .split(' ')[1]!
        .replace('/api/v1', '')
        .replace(/[.*+?^$()|[\]\\]/g, '\\$&')
    const segment = '(?:[^/\'"`]+|\\$\\{[^}]*\\})'
    return new RegExp(
        `\\.(?:get|post|put|patch|delete)\\s*(?:<[^>]*>)?\\(\\s*['"\`]${path.replace(/\{\w+\}/g, segment)}(?:\\?[^'"\`]*)?['"\`]`,
    )
}

function findOffenders(
    sources: Record<string, string>,
    wrappers: string[],
    modules: Set<string>,
    paths: RegExp[],
): string[] {
    const specifier = /(?:\bfrom|\bimport)\s*\(?\s*['"]([^'"]+)['"]/g
    return Object.entries(sources)
        .filter(([file]) => !ALLOWED_SOURCES.some((allowed) => allowed.test(file)))
        .flatMap(([file, source]) => [
            ...wrappers
                .filter((name) => new RegExp(`\\b${name}\\b`).test(source))
                .map((name) => `${file}: ${name}`),
            ...(file === ROUTER
                ? []
                : [
                      ...[...source.matchAll(specifier)]
                          .map((match) => match[1]!)
                          .filter((spec) => modules.has(resolveImport(file, spec)))
                          .map((spec) => `${file}: imports ${spec}`),
                      ...paths
                          .filter((pattern) => pattern.test(source))
                          .map((pattern) => `${file}: raw call ${pattern.source}`),
                  ]),
        ])
}

describe('experimental boundary', () => {
    const operations = experimentalOperations.map((operation) =>
        operation.replace('/api/v1', '').replace(/\{\w+\}/g, '{}'),
    )
    const paths = experimentalOperations.map(pathPattern)
    const modules = experimentalModuleIds(SOURCES)
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
        expect(findOffenders(SOURCES, experimentalWrappers, modules, paths)).toEqual([])
    })

    it('test_experimental_module_set_covers_views_card_and_validation_modules', () => {
        expect([...modules]).toEqual(
            expect.arrayContaining([
                'views/AnalysisView',
                'components/TherapyAssessmentCard',
                'composables/useValidationRuns',
            ]),
        )
        expect([...modules]).not.toContain('router/index')
    })

    it.each([
        [
            '../views/DashboardView.vue',
            "import { useValidationRuns } from '@/composables/useValidationRuns'",
        ],
        [
            '../views/DashboardView.vue',
            "import Panel from '@/components/validation/FlValidationPanel.vue'",
        ],
        ['../components/StatCard.vue', "import Panel from './validation/FlValidationPanel.vue'"],
        [
            '../views/DashboardView.vue',
            "import TherapyAssessmentCard from '@/components/TherapyAssessmentCard.vue'",
        ],
        ['../views/DashboardView.vue', "import AnalysisView from '@/views/AnalysisView.vue'"],
        ['../components/StatCard.vue', "import AnalysisView from '../views/AnalysisView.vue'"],
        ['../views/DashboardView.vue', "const View = () => import('@/views/AnalysisView.vue')"],
    ])('test_default_page_importing_experimental_module_is_flagged: %s %s', (file, source) => {
        expect(findOffenders({ [file]: source }, [], modules, paths)).toHaveLength(1)
    })

    it.each([
        "client.get('/stats/experimental')",
        'client.get(`/sessions/${id}/analysis`)',
        "client.post('/validate/fl', body)",
        "client.get('/days/2026-01-01/experimental')",
    ])('test_default_page_calling_experimental_path_is_flagged: %s', (source) => {
        const offenders = findOffenders({ '../views/StatsView.vue': source }, [], modules, paths)
        expect(offenders.length).toBeGreaterThan(0)
    })

    it('test_router_lazy_loading_experimental_views_is_allowed', () => {
        const source = "component: () => import('@/views/AnalysisView.vue'), path: '/analysis'"
        expect(findOffenders({ [ROUTER]: source }, [], modules, paths)).toEqual([])
    })
})

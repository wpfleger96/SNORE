import { describe, expect, it, vi, beforeEach } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { ref, defineComponent } from 'vue'
import type { PipelineJobStatus } from '@/types'

vi.mock('@/composables/useAuth')
vi.mock('@/api/importJobs', () => ({
    ACTIVE_PIPELINE_STAGES: new Set([
        'uploading',
        'queued',
        'importing',
        'analysis_queued',
        'analyzing',
    ]),
    getImportJobs: vi.fn(),
    cancelImport: vi.fn(),
}))
vi.mock('@/api/analysis', () => ({
    cancelAnalysisJob: vi.fn(),
}))
vi.mock('@/api/import', async (importActual) => {
    const actual = await importActual<typeof import('@/api/import')>()
    return {
        ...actual,
        importFiles: vi.fn(),
        precheckFiles: vi.fn(),
        importHealthFile: vi.fn(),
    }
})
vi.mock('@/utils/formatting', () => ({
    formatBytes: (n: number) => `${n}B`,
}))
vi.mock('@/components/ui/button', () => ({
    Button: { template: '<button v-bind="$attrs"><slot /></button>' },
}))

// Stub the archive card (tested in ArchiveRescanCard.test.ts) with a buttonless
// element so tests that click the view's last button still reach Import.
vi.mock('@/components/ArchiveRescanCard.vue', () => ({
    default: defineComponent({
        name: 'ArchiveRescanCard',
        props: ['profileId'],
        emits: ['started'],
        template: `<div class="archive-stub" @click="$emit('started')" />`,
    }),
}))

// Stub ImportJobsPanel to emit cancel with jobs[0] on button click.
vi.mock('@/components/ImportJobsPanel.vue', () => ({
    default: defineComponent({
        name: 'ImportJobsPanel',
        props: ['jobs'],
        emits: ['cancel'],
        template: `<div class="jobs-panel"><button v-if="jobs.length" class="cancel-trigger" @click="$emit('cancel', jobs[0])">Cancel</button></div>`,
    }),
}))

import { makeAuthMock } from './helpers/mockUseAuth'
import { useAuth } from '@/composables/useAuth'
import { getImportJobs, cancelImport } from '@/api/importJobs'
import { cancelAnalysisJob } from '@/api/analysis'
import { importFiles, precheckFiles, importHealthFile } from '@/api/import'
import ImportView from '@/views/ImportView.vue'

function makeJob(overrides: Partial<PipelineJobStatus> = {}): PipelineJobStatus {
    return {
        job_id: 'job-1',
        job_type: 'upload',
        state: 'active',
        stage: 'queued',
        file_count: 3,
        created_at: '2024-01-01T00:00:00+00:00',
        started_at: null,
        finished_at: null,
        progress_message: null,
        sessions_imported: null,
        import_result: null,
        error_message: null,
        analysis_job_id: null,
        analysis_queued: null,
        linked_analysis: null,
        ...overrides,
    }
}

async function mountWithJob(job: PipelineJobStatus) {
    vi.mocked(getImportJobs).mockResolvedValue({ jobs: [job] })
    const wrapper = mount(ImportView)
    await flushPromises()
    return wrapper
}

describe('ImportView cancel handler', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        vi.mocked(useAuth).mockReturnValue(
            makeAuthMock({
                profiles: ref([{ id: 1, name: 'Test' }]),
                activeProfileId: ref(1),
            }) as never,
        )
        vi.mocked(getImportJobs).mockResolvedValue({ jobs: [] })
        vi.mocked(cancelImport).mockResolvedValue(undefined)
        vi.mocked(cancelAnalysisJob).mockResolvedValue(undefined)
    })

    it('test_analyzing_stage_calls_cancelAnalysisJob_with_analysis_job_id', async () => {
        const job = makeJob({
            stage: 'analyzing',
            analysis_job_id: 'aj-123',
        })
        const wrapper = await mountWithJob(job)

        await wrapper.find('.cancel-trigger').trigger('click')
        await flushPromises()

        expect(cancelAnalysisJob).toHaveBeenCalledWith('aj-123')
        expect(cancelImport).not.toHaveBeenCalled()
    })

    it('test_analysis_queued_stage_calls_cancelAnalysisJob_with_analysis_job_id', async () => {
        const job = makeJob({
            stage: 'analysis_queued',
            analysis_job_id: 'aj-queued',
        })
        const wrapper = await mountWithJob(job)

        await wrapper.find('.cancel-trigger').trigger('click')
        await flushPromises()

        expect(cancelAnalysisJob).toHaveBeenCalledWith('aj-queued')
        expect(cancelImport).not.toHaveBeenCalled()
    })

    it('test_importing_stage_calls_cancelImport', async () => {
        const job = makeJob({ job_id: 'import-1', stage: 'importing' })
        const wrapper = await mountWithJob(job)

        await wrapper.find('.cancel-trigger').trigger('click')
        await flushPromises()

        expect(cancelImport).toHaveBeenCalledWith('import-1')
        expect(cancelAnalysisJob).not.toHaveBeenCalled()
    })

    it('test_analyzing_stage_with_null_analysis_job_id_falls_back_to_cancelImport', async () => {
        const job = makeJob({
            job_id: 'import-2',
            stage: 'analyzing',
            analysis_job_id: null,
        })
        const wrapper = await mountWithJob(job)

        await wrapper.find('.cancel-trigger').trigger('click')
        await flushPromises()

        expect(cancelImport).toHaveBeenCalledWith('import-2')
        expect(cancelAnalysisJob).not.toHaveBeenCalled()
    })
})

// ---------------------------------------------------------------------------
// Helpers for precheck tests
// ---------------------------------------------------------------------------

function makeFile(name: string, size: number, relativePath?: string): File {
    const file = new File([new Uint8Array(size)], name, { type: 'application/octet-stream' })
    if (relativePath) {
        Object.defineProperty(file, 'webkitRelativePath', {
            value: relativePath,
            configurable: true,
        })
    }
    return file
}

async function triggerFileSelection(
    wrapper: ReturnType<typeof mount>,
    files: File[],
): Promise<void> {
    const input = wrapper.find('input[type="file"]').element as HTMLInputElement
    const fileListLike = { length: files.length, item: (i: number) => files[i] ?? null }
    files.forEach((f, i) => {
        ;(fileListLike as Record<string | number, unknown>)[i] = f
    })
    Object.defineProperty(fileListLike, Symbol.iterator, {
        value: () => files[Symbol.iterator](),
    })
    Object.defineProperty(input, 'files', { value: fileListLike, configurable: true })
    await wrapper.find('input[type="file"]').trigger('change')
    await flushPromises()
}

function makeDeferred<T>(): { promise: Promise<T>; resolve: (v: T) => void } {
    let resolve!: (v: T) => void
    const promise = new Promise<T>((res) => {
        resolve = res
    })
    return { promise, resolve }
}

// Two session nights so that when BRP.edf is marked skippable, BRP2.edf
// keeps uploadCount > 0 — enabling the Import button and allowing handleImport
// to reach the importFiles call rather than short-circuiting via resetUpload().
const resMedFiles = [
    makeFile('STR.edf', 200, 'MySD/STR.edf'),
    makeFile('BRP.edf', 1000, 'MySD/DATALOG/20240101_010000_BRP.edf'),
    makeFile('BRP2.edf', 800, 'MySD/DATALOG/20240102_010000_BRP2.edf'),
    makeFile('Identification.json', 100, 'MySD/Identification.json'),
]

const nonResMedFiles = [makeFile('random.txt', 50, 'MyFolder/random.txt')]

describe('ImportView precheck', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        vi.mocked(useAuth).mockReturnValue(
            makeAuthMock({
                profiles: ref([{ id: 1, name: 'Test' }]),
                activeProfileId: ref(1),
            }) as never,
        )
        vi.mocked(getImportJobs).mockResolvedValue({ jobs: [] })
        vi.mocked(importFiles).mockResolvedValue({ job_id: 'job-1' } as never)
        vi.mocked(precheckFiles).mockResolvedValue(new Set())
    })

    it('precheck fires on selection for ResMed-structured entries', async () => {
        const wrapper = mount(ImportView)
        await flushPromises()

        await triggerFileSelection(wrapper, resMedFiles)

        expect(precheckFiles).toHaveBeenCalledOnce()
    })

    it('precheck does NOT fire for non-ResMed selection', async () => {
        const wrapper = mount(ImportView)
        await flushPromises()

        await triggerFileSelection(wrapper, nonResMedFiles)

        expect(precheckFiles).not.toHaveBeenCalled()
    })

    it('skippable non-anchor entries are excluded from importFiles call', async () => {
        // precheck marks the data file as skippable, not the anchor
        vi.mocked(precheckFiles).mockResolvedValue(
            new Set(['MySD/DATALOG/20240101_010000_BRP.edf']),
        )

        const wrapper = mount(ImportView)
        await flushPromises()
        await triggerFileSelection(wrapper, resMedFiles)

        await wrapper.findAll('button').at(-1)!.trigger('click')
        await flushPromises()

        const sentEntries = vi.mocked(importFiles).mock.calls[0][0]
        const sentPaths = sentEntries.map((e: { path: string }) => e.path)
        expect(sentPaths).not.toContain('MySD/DATALOG/20240101_010000_BRP.edf')
    })

    it('anchor files are always retained even when marked skippable', async () => {
        // server wrongly marks STR.edf skippable — client must still send it
        vi.mocked(precheckFiles).mockResolvedValue(
            new Set(['MySD/STR.edf', 'MySD/DATALOG/20240101_010000_BRP.edf']),
        )

        const wrapper = mount(ImportView)
        await flushPromises()
        await triggerFileSelection(wrapper, resMedFiles)

        await wrapper.findAll('button').at(-1)!.trigger('click')
        await flushPromises()

        const sentPaths = vi
            .mocked(importFiles)
            .mock.calls[0][0].map((e: { path: string }) => e.path)
        expect(sentPaths).toContain('MySD/STR.edf')
        expect(sentPaths).toContain('MySD/Identification.json')
        expect(sentPaths).not.toContain('MySD/DATALOG/20240101_010000_BRP.edf')
    })

    it('fail-open — precheck error causes importFiles to receive all entries', async () => {
        vi.mocked(precheckFiles).mockResolvedValue(new Set())

        const wrapper = mount(ImportView)
        await flushPromises()
        await triggerFileSelection(wrapper, resMedFiles)

        await wrapper.findAll('button').at(-1)!.trigger('click')
        await flushPromises()

        const sentPaths = vi
            .mocked(importFiles)
            .mock.calls[0][0].map((e: { path: string }) => e.path)
        expect(sentPaths).toHaveLength(resMedFiles.length)
    })

    it('forceUploadAll sends all entries despite non-empty skippable set', async () => {
        vi.mocked(precheckFiles).mockResolvedValue(
            new Set(['MySD/DATALOG/20240101_010000_BRP.edf']),
        )

        const wrapper = mount(ImportView)
        await flushPromises()
        await triggerFileSelection(wrapper, resMedFiles)

        // Check the "Upload all files" checkbox
        const checkbox = wrapper.find('input[type="checkbox"]')
        await checkbox.setValue(true)

        await wrapper.findAll('button').at(-1)!.trigger('click')
        await flushPromises()

        const sentPaths = vi
            .mocked(importFiles)
            .mock.calls[0][0].map((e: { path: string }) => e.path)
        expect(sentPaths).toHaveLength(resMedFiles.length)
        expect(sentPaths).toContain('MySD/DATALOG/20240101_010000_BRP.edf')
    })

    it('stale precheck result is discarded on reselection', async () => {
        const deferred1 = makeDeferred<Set<string>>()
        const deferred2 = makeDeferred<Set<string>>()
        vi.mocked(precheckFiles)
            .mockReturnValueOnce(deferred1.promise)
            .mockReturnValueOnce(deferred2.promise)

        const wrapper = mount(ImportView)
        await flushPromises()

        // First selection (precheck pending, not yet resolved)
        const input = wrapper.find('input[type="file"]').element as HTMLInputElement
        const makeFileList = (files: File[]) => {
            const fl = { length: files.length, item: (i: number) => files[i] ?? null }
            files.forEach((f, i) => {
                ;(fl as Record<string | number, unknown>)[i] = f
            })
            Object.defineProperty(fl, Symbol.iterator, { value: () => files[Symbol.iterator]() })
            return fl
        }
        Object.defineProperty(input, 'files', {
            value: makeFileList(resMedFiles),
            configurable: true,
        })
        await wrapper.find('input[type="file"]').trigger('change')
        // Do NOT flush — deferred1 is still pending

        // Second selection before first resolves
        Object.defineProperty(input, 'files', {
            value: makeFileList([
                makeFile('STR.edf', 50, 'NewSD/STR.edf'),
                makeFile('BRP2.edf', 200, 'NewSD/DATALOG/20240102_BRP.edf'),
            ]),
            configurable: true,
        })
        await wrapper.find('input[type="file"]').trigger('change')

        // Resolve first precheck with a non-empty set (should be stale)
        deferred1.resolve(new Set(['MySD/DATALOG/20240101_010000_BRP.edf']))
        await flushPromises()

        // Resolve second precheck with empty set
        deferred2.resolve(new Set())
        await flushPromises()

        // Click Import — importFiles should receive all entries from second selection
        await wrapper.findAll('button').at(-1)!.trigger('click')
        await flushPromises()

        const sentPaths = vi
            .mocked(importFiles)
            .mock.calls[0][0].map((e: { path: string }) => e.path)
        // Stale result from first precheck must not filter second selection's entries
        expect(sentPaths).not.toContain('MySD/DATALOG/20240101_010000_BRP.edf')
    })

    it('anchor retention hard case — Identification.json in skippable set is still uploaded', async () => {
        // Server wrongly marks both anchors skippable — client must retain them both.
        vi.mocked(precheckFiles).mockResolvedValue(
            new Set([
                'MySD/STR.edf',
                'MySD/Identification.json',
                'MySD/DATALOG/20240101_010000_BRP.edf',
            ]),
        )

        const wrapper = mount(ImportView)
        await flushPromises()
        await triggerFileSelection(wrapper, resMedFiles)

        await wrapper.findAll('button').at(-1)!.trigger('click')
        await flushPromises()

        const sentPaths = vi
            .mocked(importFiles)
            .mock.calls[0][0].map((e: { path: string }) => e.path)
        expect(sentPaths).toContain('MySD/STR.edf')
        expect(sentPaths).toContain('MySD/Identification.json')
        expect(sentPaths).not.toContain('MySD/DATALOG/20240101_010000_BRP.edf')
    })

    it('profile switch after precheck uses new profile skip set, not stale one', async () => {
        vi.mocked(useAuth).mockReturnValue(
            makeAuthMock({
                profiles: ref([
                    { id: 1, name: 'Profile A' },
                    { id: 2, name: 'Profile B' },
                ]),
                activeProfileId: ref(1),
            }) as never,
        )
        // Profile 1 marks BRP.edf skippable; profile 2 has nothing skippable.
        vi.mocked(precheckFiles)
            .mockResolvedValueOnce(new Set(['MySD/DATALOG/20240101_010000_BRP.edf']))
            .mockResolvedValueOnce(new Set())

        const wrapper = mount(ImportView)
        await flushPromises()
        await triggerFileSelection(wrapper, resMedFiles)
        // Profile 1 precheck has resolved with a skippable entry.

        // Switch to profile 2 — triggers re-precheck which clears the stale set.
        await wrapper.find('select').setValue(2)
        await flushPromises()
        // Profile 2 precheck has resolved with an empty set.

        // Import should send all files (profile 2 has no skippable entries).
        await wrapper.findAll('button').at(-1)!.trigger('click')
        await flushPromises()

        const sentPaths = vi
            .mocked(importFiles)
            .mock.calls[0][0].map((e: { path: string }) => e.path)
        expect(sentPaths).toHaveLength(resMedFiles.length)
    })

    it('double-click on Import triggers importFiles exactly once', async () => {
        vi.mocked(precheckFiles).mockResolvedValue(new Set())
        // Keep importFiles pending so the uploading phase stays active during the test.
        vi.mocked(importFiles).mockReturnValue(new Promise(() => {}))

        const wrapper = mount(ImportView)
        await flushPromises()
        await triggerFileSelection(wrapper, resMedFiles)

        const btn = wrapper.findAll('button').at(-1)!
        // Two rapid clicks — second must be rejected by the re-entrancy guard.
        void btn.trigger('click')
        void btn.trigger('click')
        await flushPromises()

        expect(importFiles).toHaveBeenCalledTimes(1)
    })

    it('profile switch in error phase re-triggers precheck', async () => {
        vi.mocked(useAuth).mockReturnValue(
            makeAuthMock({
                profiles: ref([
                    { id: 1, name: 'Profile A' },
                    { id: 2, name: 'Profile B' },
                ]),
                activeProfileId: ref(1),
            }) as never,
        )
        vi.mocked(precheckFiles).mockResolvedValue(new Set())
        vi.mocked(importFiles).mockRejectedValue(new Error('Upload failed'))

        const wrapper = mount(ImportView)
        await flushPromises()
        await triggerFileSelection(wrapper, resMedFiles)

        // Trigger a failing import to land in error phase.
        await wrapper.findAll('button').at(-1)!.trigger('click')
        await flushPromises()
        expect(wrapper.find('.error-text').exists()).toBe(true)

        const precheckCallsBefore = vi.mocked(precheckFiles).mock.calls.length

        // Switch profile while in error phase — must re-run precheck.
        await wrapper.find('select').setValue(2)
        await flushPromises()

        expect(vi.mocked(precheckFiles).mock.calls.length).toBe(precheckCallsBefore + 1)
    })

    it('skip summary text renders during uploading when files are skipped', async () => {
        vi.mocked(precheckFiles).mockResolvedValue(
            new Set(['MySD/DATALOG/20240101_010000_BRP.edf']),
        )
        // Keep importFiles pending so uploading phase stays visible
        vi.mocked(importFiles).mockReturnValue(new Promise(() => {}))

        const wrapper = mount(ImportView)
        await flushPromises()
        await triggerFileSelection(wrapper, resMedFiles)

        await wrapper.findAll('button').at(-1)!.trigger('click')
        await flushPromises()

        expect(wrapper.find('.skip-summary').exists()).toBe(true)
        expect(wrapper.find('.skip-summary').text()).toContain('Skipped 1 files already on server')
    })
})

// ---------------------------------------------------------------------------
// Apple Health tab tests
// ---------------------------------------------------------------------------

describe('ImportView Apple Health tab', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        vi.mocked(useAuth).mockReturnValue(
            makeAuthMock({
                profiles: ref([{ id: 1, name: 'Test' }]),
                activeProfileId: ref(1),
            }) as never,
        )
        vi.mocked(getImportJobs).mockResolvedValue({ jobs: [] })
        vi.mocked(importFiles).mockResolvedValue({ job_id: 'cpap-job' } as never)
        vi.mocked(importHealthFile).mockResolvedValue({ job_id: 'health-job' } as never)
        vi.mocked(precheckFiles).mockResolvedValue(new Set())
    })

    it('test_default_tab_is_cpap', async () => {
        const wrapper = mount(ImportView)
        await flushPromises()

        expect((wrapper.vm as unknown as { activeTab: string }).activeTab).toBe('cpap')
    })

    it('test_switching_to_health_tab_hides_cpap_pane', async () => {
        const wrapper = mount(ImportView)
        await flushPromises()

        // Switch to health tab by directly setting the reactive state.
        ;(wrapper.vm as unknown as { activeTab: string }).activeTab = 'health'
        await flushPromises()

        // The CPAP file input lives inside the v-show="activeTab === 'cpap'" div.
        // When health tab is active that div gets display:none.
        const cpapFileInput = wrapper.find('input[type="file"][multiple]')
        expect(cpapFileInput.element.parentElement!.style.display).toBe('none')

        // The health file input's parent div should not be hidden.
        const healthFileInput = wrapper.find('input[type="file"][accept=".zip"]')
        expect(healthFileInput.element.parentElement!.style.display).not.toBe('none')
    })

    it('test_health_import_calls_importHealthFile_with_file_and_selected_profile_id', async () => {
        const wrapper = mount(ImportView)
        await flushPromises()

        // Switch to health tab and set a non-default profile.
        const vm = wrapper.vm as unknown as { activeTab: string; selectedProfileId: number | null }
        vm.activeTab = 'health'
        vm.selectedProfileId = 3
        await flushPromises()

        // Select a zip file via the health file input.
        const zipFile = new File([new Uint8Array(512)], 'export.zip', { type: 'application/zip' })
        const healthInput = wrapper.find('input[type="file"][accept=".zip"]')
        Object.defineProperty(healthInput.element, 'files', {
            value: { length: 1, 0: zipFile, item: (i: number) => (i === 0 ? zipFile : null) },
            configurable: true,
        })
        await healthInput.trigger('change')
        await flushPromises()

        // Click the Import button (last button in the DOM — health pane selected phase).
        await wrapper.findAll('button').at(-1)!.trigger('click')
        await flushPromises()

        expect(importHealthFile).toHaveBeenCalledOnce()
        expect(importHealthFile).toHaveBeenCalledWith(zipFile, 3, expect.any(Function))
        // CPAP importFiles must not be called from the health tab path.
        expect(importFiles).not.toHaveBeenCalled()
    })

    it('test_health_tab_import_does_not_invoke_cpap_importFiles', async () => {
        const wrapper = mount(ImportView)
        await flushPromises()

        ;(wrapper.vm as unknown as { activeTab: string }).activeTab = 'health'
        await flushPromises()

        const zipFile = new File([new Uint8Array(100)], 'export.zip', { type: 'application/zip' })
        const healthInput = wrapper.find('input[type="file"][accept=".zip"]')
        Object.defineProperty(healthInput.element, 'files', {
            value: { length: 1, 0: zipFile, item: (i: number) => (i === 0 ? zipFile : null) },
            configurable: true,
        })
        await healthInput.trigger('change')
        await flushPromises()

        await wrapper.findAll('button').at(-1)!.trigger('click')
        await flushPromises()

        expect(importFiles).not.toHaveBeenCalled()
        expect(importHealthFile).toHaveBeenCalledOnce()
    })
})

// ---------------------------------------------------------------------------
// Archive re-import
// ---------------------------------------------------------------------------

// All session files skippable — uploadCount === 0 state.
const allSkippedFiles = [
    makeFile('STR.edf', 200, 'MySD/STR.edf'),
    makeFile('BRP.edf', 1000, 'MySD/DATALOG/20240101_010000_BRP.edf'),
    makeFile('BRP2.edf', 800, 'MySD/DATALOG/20240102_010000_BRP2.edf'),
    makeFile('Identification.json', 100, 'MySD/Identification.json'),
]
const allSkippedPaths = new Set([
    'MySD/DATALOG/20240101_010000_BRP.edf',
    'MySD/DATALOG/20240102_010000_BRP2.edf',
])

describe('ImportView archive re-import', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        vi.mocked(useAuth).mockReturnValue(
            makeAuthMock({
                profiles: ref([
                    { id: 1, name: 'Test' },
                    { id: 2, name: 'Other' },
                ]),
                activeProfileId: ref(1),
            }) as never,
        )
        vi.mocked(getImportJobs).mockResolvedValue({ jobs: [] })
        vi.mocked(precheckFiles).mockResolvedValue(allSkippedPaths)
    })

    it('test_archive_card_shown_without_selecting_a_folder', async () => {
        const wrapper = mount(ImportView)
        await flushPromises()

        const card = wrapper.findComponent({ name: 'ArchiveRescanCard' })
        expect(card.exists()).toBe(true)
        expect(card.isVisible()).toBe(true)
        expect(card.props('profileId')).toBe(1)
    })

    it('test_archive_card_targets_the_selected_profile', async () => {
        const wrapper = mount(ImportView)
        await flushPromises()

        await wrapper.find('#import-profile').setValue('2')

        expect(wrapper.findComponent({ name: 'ArchiveRescanCard' }).props('profileId')).toBe(2)
    })

    it('test_archive_card_hidden_on_health_tab', async () => {
        const wrapper = mount(ImportView)
        await flushPromises()

        ;(wrapper.vm as unknown as { activeTab: string }).activeTab = 'health'
        await flushPromises()

        expect(wrapper.findComponent({ name: 'ArchiveRescanCard' }).isVisible()).toBe(false)
    })

    it('test_started_rescan_refreshes_jobs', async () => {
        const wrapper = mount(ImportView)
        await flushPromises()

        await wrapper.find('.archive-stub').trigger('click')
        await flushPromises()

        // Once at mount, once after the rescan started.
        expect(getImportJobs).toHaveBeenCalledTimes(2)
    })

    it('test_started_rescan_returns_selected_folder_to_drop_zone', async () => {
        const wrapper = mount(ImportView)
        await flushPromises()
        await triggerFileSelection(wrapper, allSkippedFiles)
        expect(wrapper.text()).toContain('MySD')

        await wrapper.find('.archive-stub').trigger('click')
        await flushPromises()

        expect(wrapper.text()).not.toContain('MySD')
        expect(wrapper.findAll('button').some((b) => b.text() === 'Change folder')).toBe(false)
    })

    it('test_all_files_on_server_points_to_archive_card_and_disables_import', async () => {
        const wrapper = mount(ImportView)
        await flushPromises()
        await triggerFileSelection(wrapper, allSkippedFiles)

        expect(wrapper.text()).toContain('Use Re-import from server archive below')
        const importBtn = wrapper.findAll('button').find((b) => b.text() === 'Import')!
        expect(importBtn.attributes('disabled')).toBeDefined()
    })
})

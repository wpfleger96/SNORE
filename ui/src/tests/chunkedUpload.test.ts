import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/api/client', () => ({
    default: { post: vi.fn() },
    apiPost: () => vi.fn(),
    apiGet: () => vi.fn(),
    apiPatch: () => vi.fn(),
    apiDelete: () => vi.fn(),
    apiGetOrNull: () => vi.fn(),
    createApiEndpoint: () => vi.fn(),
}))

import api from '@/api/client'
import {
    importFiles,
    precheckFiles,
    buildChunks,
    type FileEntry,
    type ChunkedImportProgress,
} from '@/api/import'

function makeEntry(path: string, sizeBytes: number): FileEntry {
    return { file: new File([new Uint8Array(sizeBytes)], path.split('/').pop()!), path }
}

function formDataFiles(formData: FormData): string[] {
    return (formData.getAll('files') as File[]).map((f) => f.name)
}

describe('buildChunks', () => {
    const strEdf = makeEntry('STR.edf', 1_000_000)
    const idJson = makeEntry('Identification.json', 500)

    it('keeps all files in one chunk when under limit', () => {
        const data = [
            makeEntry('DATALOG/2024/20240101_010000_BRP.edf', 5_000_000),
            makeEntry('DATALOG/2024/20240102_010000_BRP.edf', 5_000_000),
        ]
        const chunks = buildChunks(data, [strEdf, idJson], 20_000_000)
        expect(chunks).toHaveLength(1)
        expect(chunks[0]).toHaveLength(4)
    })

    it('splits into multiple chunks when exceeding limit', () => {
        const data = [
            makeEntry('DATALOG/2024/20240101_010000_BRP.edf', 40_000_000),
            makeEntry('DATALOG/2024/20240102_010000_BRP.edf', 40_000_000),
        ]
        const chunks = buildChunks(data, [strEdf], 45_000_000)
        expect(chunks).toHaveLength(2)
    })

    it('duplicates anchor files in every chunk', () => {
        const data = [
            makeEntry('DATALOG/2024/20240101_010000_BRP.edf', 40_000_000),
            makeEntry('DATALOG/2024/20240102_010000_BRP.edf', 40_000_000),
        ]
        const chunks = buildChunks(data, [strEdf], 45_000_000)
        for (const chunk of chunks) {
            const paths = chunk.map((e) => e.path)
            expect(paths).toContain('STR.edf')
        }
    })

    it('keeps files for the same night together', () => {
        const data = [
            makeEntry('DATALOG/2024/20240101_010000_BRP.edf', 10_000_000),
            makeEntry('DATALOG/2024/20240101_010000_PLD.edf', 10_000_000),
            makeEntry('DATALOG/2024/20240101_010000_EVE.edf', 5_000_000),
            makeEntry('DATALOG/2024/20240102_010000_BRP.edf', 10_000_000),
        ]
        const chunks = buildChunks(data, [strEdf], 30_000_000)
        const chunk1Paths = chunks[0].map((e) => e.path)
        expect(chunk1Paths.filter((p) => p.includes('20240101'))).toHaveLength(3)
    })

    it('puts undated files in final batch', () => {
        const data = [
            makeEntry('DATALOG/2024/20240101_010000_BRP.edf', 40_000_000),
            makeEntry('some_other_file.txt', 100),
        ]
        const chunks = buildChunks(data, [strEdf], 45_000_000)
        const lastChunk = chunks[chunks.length - 1]
        const paths = lastChunk.map((e) => e.path)
        expect(paths).toContain('some_other_file.txt')
    })

    it('handles single oversized night as its own batch', () => {
        const data = [
            makeEntry('DATALOG/2024/20240101_010000_BRP.edf', 80_000_000),
            makeEntry('DATALOG/2024/20240102_010000_BRP.edf', 5_000_000),
        ]
        const chunks = buildChunks(data, [strEdf], 45_000_000)
        expect(chunks.length).toBeGreaterThanOrEqual(2)
    })

    it('returns one chunk with only anchors when no data files', () => {
        const chunks = buildChunks([], [strEdf], 45_000_000)
        expect(chunks).toHaveLength(1)
        expect(chunks[0]).toHaveLength(1)
        expect(chunks[0][0].path).toBe('STR.edf')
    })
})

describe('importFiles', () => {
    beforeEach(() => {
        vi.resetAllMocks()
        vi.mocked(api.post).mockResolvedValue({ data: { job_id: 'job-1' } })
    })

    it('sends single request when total size is under limit', async () => {
        const entries = [makeEntry('DATALOG/2024/20240101_010000_BRP.edf', 1000)]
        await importFiles(entries)
        expect(api.post).toHaveBeenCalledTimes(1)
    })

    it('sends multiple requests when total size exceeds limit', async () => {
        const entries = [
            makeEntry('STR.edf', 1_000_000),
            makeEntry('DATALOG/2024/20240101_010000_BRP.edf', 50_000_000),
            makeEntry('DATALOG/2024/20240102_010000_BRP.edf', 50_000_000),
        ]
        await importFiles(entries)
        expect(vi.mocked(api.post).mock.calls.length).toBeGreaterThan(1)
    })

    it('includes anchor files in every chunk request', async () => {
        const entries = [
            makeEntry('STR.edf', 1_000_000),
            makeEntry('Identification.json', 500),
            makeEntry('DATALOG/2024/20240101_010000_BRP.edf', 50_000_000),
            makeEntry('DATALOG/2024/20240102_010000_BRP.edf', 50_000_000),
        ]
        await importFiles(entries)
        const calls = vi.mocked(api.post).mock.calls
        for (const call of calls) {
            const fd = call[1] as FormData
            const fileNames = formDataFiles(fd)
            expect(fileNames).toContain('STR.edf')
            expect(fileNames).toContain('Identification.json')
        }
    })

    it('forwards profile_id to every chunk', async () => {
        const entries = [
            makeEntry('STR.edf', 1_000_000),
            makeEntry('DATALOG/2024/20240101_010000_BRP.edf', 50_000_000),
            makeEntry('DATALOG/2024/20240102_010000_BRP.edf', 50_000_000),
        ]
        await importFiles(entries, undefined, 42)
        const calls = vi.mocked(api.post).mock.calls
        for (const call of calls) {
            const fd = call[1] as FormData
            expect(fd.get('profile_id')).toBe('42')
        }
    })

    it('sends batch_id from first response on subsequent chunks', async () => {
        const entries = [
            makeEntry('STR.edf', 1_000_000),
            makeEntry('DATALOG/2024/20240101_010000_BRP.edf', 50_000_000),
            makeEntry('DATALOG/2024/20240102_010000_BRP.edf', 50_000_000),
        ]
        await importFiles(entries)
        const calls = vi.mocked(api.post).mock.calls
        // First call has no batch_id
        expect((calls[0][1] as FormData).get('batch_id')).toBeNull()
        // Subsequent calls carry the job_id from the first response
        for (let i = 1; i < calls.length; i++) {
            expect((calls[i][1] as FormData).get('batch_id')).toBe('job-1')
        }
    })

    it('sends batch_final=true only on the last chunk', async () => {
        const entries = [
            makeEntry('STR.edf', 1_000_000),
            makeEntry('DATALOG/2024/20240101_010000_BRP.edf', 50_000_000),
            makeEntry('DATALOG/2024/20240102_010000_BRP.edf', 50_000_000),
        ]
        await importFiles(entries)
        const calls = vi.mocked(api.post).mock.calls
        for (let i = 0; i < calls.length; i++) {
            const fd = calls[i][1] as FormData
            const isLast = i === calls.length - 1
            expect(fd.get('batch_final')).toBe(String(isLast))
        }
    })

    it('reports correct batch index and total in progress events', async () => {
        const entries = [
            makeEntry('STR.edf', 1_000_000),
            makeEntry('DATALOG/2024/20240101_010000_BRP.edf', 50_000_000),
            makeEntry('DATALOG/2024/20240102_010000_BRP.edf', 50_000_000),
        ]

        const progressEvents: ChunkedImportProgress[] = []
        vi.mocked(api.post).mockImplementation((_url, _data, config) => {
            if (config?.onUploadProgress) {
                config.onUploadProgress({ loaded: 100, total: 100, bytes: 100 } as never)
            }
            return Promise.resolve({ data: { job_id: 'job-1' } })
        })

        await importFiles(entries, (p) => progressEvents.push({ ...p }))
        const batches = new Set(progressEvents.map((p) => p.batchIndex))
        expect(batches.size).toBeGreaterThan(1)
        expect(progressEvents.every((p) => p.totalBatches === batches.size)).toBe(true)
    })

    it('includes batch number in error message on failure', async () => {
        const entries = [
            makeEntry('STR.edf', 1_000_000),
            makeEntry('DATALOG/2024/20240101_010000_BRP.edf', 50_000_000),
            makeEntry('DATALOG/2024/20240102_010000_BRP.edf', 50_000_000),
        ]
        vi.mocked(api.post)
            .mockResolvedValueOnce({ data: { job_id: 'job-1' } })
            .mockRejectedValueOnce(new Error('Request failed with status code 413'))

        await expect(importFiles(entries)).rejects.toThrow(/Batch \d+\/\d+ failed:/)
    })

    it('does not send batch fields on single-request uploads', async () => {
        const entries = [makeEntry('small.edf', 1000)]
        await importFiles(entries)
        const fd = vi.mocked(api.post).mock.calls[0][1] as FormData
        expect(fd.get('batch_id')).toBeNull()
        expect(fd.get('batch_final')).toBeNull()
    })
})

describe('precheckFiles', () => {
    beforeEach(() => {
        vi.resetAllMocks()
    })

    it('returns Set of skippable paths on success', async () => {
        vi.mocked(api.post).mockResolvedValue({
            data: { skippable: ['DATALOG/a/b.edf', 'DATALOG/c/d.edf'] },
        })
        const entries = [makeEntry('DATALOG/a/b.edf', 100), makeEntry('DATALOG/c/d.edf', 200)]
        const result = await precheckFiles(entries)
        expect(result).toEqual(new Set(['DATALOG/a/b.edf', 'DATALOG/c/d.edf']))
    })

    it('returns empty Set without throwing on rejection (exercises catch branch)', async () => {
        // api.post itself rejects — verifies the catch branch, not just an empty response.
        vi.mocked(api.post).mockRejectedValue(new Error('network error'))
        const entries = [makeEntry('DATALOG/a/b.edf', 100)]
        const result = await precheckFiles(entries)
        expect(result).toEqual(new Set())
        expect(api.post).toHaveBeenCalled()
    })

    it('sends profile_id when provided', async () => {
        vi.mocked(api.post).mockResolvedValue({ data: { skippable: [] } })
        const entries = [makeEntry('DATALOG/a/b.edf', 100)]
        await precheckFiles(entries, 42)
        const [, body] = vi.mocked(api.post).mock.calls[0]
        expect(body).toMatchObject({ profile_id: 42 })
    })

    it('omits profile_id when undefined', async () => {
        vi.mocked(api.post).mockResolvedValue({ data: { skippable: [] } })
        const entries = [makeEntry('DATALOG/a/b.edf', 100)]
        await precheckFiles(entries)
        const [, body] = vi.mocked(api.post).mock.calls[0]
        expect(body).not.toHaveProperty('profile_id')
    })

    it('excludes anchor files and non-DATALOG paths from the request body', async () => {
        vi.mocked(api.post).mockResolvedValue({ data: { skippable: [] } })
        const entries = [
            makeEntry('STR.edf', 100),
            makeEntry('Identification.json', 100),
            makeEntry('some_file.txt', 100),
            makeEntry('DATALOG/20240101_010000_BRP.edf', 100),
        ]
        await precheckFiles(entries)
        const [, body] = vi.mocked(api.post).mock.calls[0]
        const paths = (body as { files: { path: string }[] }).files.map((f) => f.path)
        expect(paths).not.toContain('STR.edf')
        expect(paths).not.toContain('Identification.json')
        expect(paths).not.toContain('some_file.txt')
        expect(paths).toContain('DATALOG/20240101_010000_BRP.edf')
    })

    it('returns empty Set without calling api.post when no DATALOG candidates', async () => {
        const entries = [makeEntry('STR.edf', 100), makeEntry('some_file.txt', 100)]
        const result = await precheckFiles(entries)
        expect(result).toEqual(new Set())
        expect(api.post).not.toHaveBeenCalled()
    })
})

import api, { apiPost } from './client'
import type { RescanRequest } from '@/types'
import type { AxiosProgressEvent } from 'axios'

export interface FileEntry {
    file: File
    path: string
}

export interface ChunkedImportProgress {
    loaded: number
    total: number
    batchIndex: number
    totalBatches: number
}

interface JobResponse {
    job_id: string
}

const CHUNK_SIZE_LIMIT: number = (() => {
    const env = import.meta.env.VITE_CHUNK_SIZE_BYTES
    if (typeof env === 'string' && /^\d+$/.test(env)) return parseInt(env, 10)
    return 90 * 1024 * 1024
})()

const ANCHOR_FILE_RE = /^(STR\.edf|Identification\.(json|tgt))$/i
const DATALOG_RE = /(^|\/)DATALOG\//i

export function isAnchorFile(entry: FileEntry): boolean {
    const name = entry.path.split('/').pop() ?? ''
    return ANCHOR_FILE_RE.test(name)
}

export function isImportableFile(entry: FileEntry): boolean {
    if (isAnchorFile(entry)) return true
    if (!DATALOG_RE.test(entry.path)) return false
    const basename = entry.path.split('/').pop() ?? ''
    return /\.edf$/i.test(basename) && !basename.startsWith('.')
}

interface PrecheckFileEntry {
    path: string
    size: number
}

interface PrecheckRequest {
    files: PrecheckFileEntry[]
    profile_id?: number
}

interface PrecheckResponse {
    skippable: string[]
}

export async function precheckFiles(
    entries: FileEntry[],
    profileId?: number,
): Promise<Set<string>> {
    // Only send non-anchor DATALOG files; anchors are never skippable and
    // other paths are irrelevant to the server's dedupe check.
    const candidates = entries.filter((e) => !isAnchorFile(e) && DATALOG_RE.test(e.path))
    if (candidates.length === 0) return new Set()
    try {
        const body: PrecheckRequest = {
            files: candidates.map((e) => ({ path: e.path, size: e.file.size })),
            ...(profileId !== undefined ? { profile_id: profileId } : {}),
        }
        const { data } = await api.post<PrecheckResponse>('/import/precheck', body, {
            timeout: 5000,
        })
        return new Set(data.skippable)
    } catch (err) {
        console.warn('[precheck] failed, proceeding with full upload', err)
        return new Set()
    }
}

function extractNightDate(path: string): string | null {
    const match = path.match(/(\d{8})_\d{6}_\w+\.edf$/i)
    return match ? match[1] : null
}

function groupByNight(entries: FileEntry[]): {
    byNight: Map<string, FileEntry[]>
    undated: FileEntry[]
} {
    const byNight = new Map<string, FileEntry[]>()
    const undated: FileEntry[] = []
    for (const entry of entries) {
        const night = extractNightDate(entry.path)
        if (night !== null) {
            if (!byNight.has(night)) byNight.set(night, [])
            byNight.get(night)!.push(entry)
        } else {
            undated.push(entry)
        }
    }
    return { byNight, undated }
}

export function buildChunks(
    nonAnchorEntries: FileEntry[],
    anchorEntries: FileEntry[],
    chunkSizeLimit: number = CHUNK_SIZE_LIMIT,
): FileEntry[][] {
    const anchorSize = anchorEntries.reduce((s, e) => s + e.file.size, 0)
    const { byNight, undated } = groupByNight(nonAnchorEntries)
    const sortedNights = [...byNight.entries()].sort(([a], [b]) => a.localeCompare(b))

    const chunks: FileEntry[][] = []
    let batchEntries: FileEntry[] = []
    let batchDataSize = 0

    const flushIfNeeded = (nextSize: number) => {
        if (batchDataSize > 0 && anchorSize + batchDataSize + nextSize > chunkSizeLimit) {
            chunks.push([...anchorEntries, ...batchEntries])
            batchEntries = []
            batchDataSize = 0
        }
    }

    for (const [, nightFiles] of sortedNights) {
        const nightSize = nightFiles.reduce((s, e) => s + e.file.size, 0)
        flushIfNeeded(nightSize)
        batchEntries.push(...nightFiles)
        batchDataSize += nightSize
    }

    for (const entry of undated) {
        flushIfNeeded(entry.file.size)
        batchEntries.push(entry)
        batchDataSize += entry.file.size
    }

    if (batchEntries.length > 0 || chunks.length === 0) {
        chunks.push([...anchorEntries, ...batchEntries])
    }

    return chunks
}

async function uploadSingleChunk(
    entries: FileEntry[],
    profileId: number | undefined,
    onProgress?: (event: AxiosProgressEvent) => void,
    batchId?: string,
    batchFinal?: boolean,
): Promise<JobResponse> {
    const formData = new FormData()
    for (const entry of entries) {
        formData.append('files', entry.file, entry.path)
    }
    if (profileId !== undefined) formData.append('profile_id', String(profileId))
    if (batchId !== undefined) formData.append('batch_id', batchId)
    if (batchFinal !== undefined) formData.append('batch_final', String(batchFinal))
    const { data } = await api.post<JobResponse>('/import/', formData, {
        onUploadProgress: onProgress,
    })
    return data
}

export async function importFiles(
    entries: FileEntry[],
    onProgress?: (progress: ChunkedImportProgress) => void,
    profileId?: number,
): Promise<JobResponse> {
    const totalSize = entries.reduce((s, e) => s + e.file.size, 0)

    if (totalSize <= CHUNK_SIZE_LIMIT) {
        return uploadSingleChunk(
            entries,
            profileId,
            onProgress
                ? (event) =>
                      onProgress({
                          loaded: event.loaded,
                          total: event.total || totalSize,
                          batchIndex: 1,
                          totalBatches: 1,
                      })
                : undefined,
        )
    }

    const anchorEntries = entries.filter(isAnchorFile)
    const nonAnchorEntries = entries.filter((e) => !isAnchorFile(e))
    const chunks = buildChunks(nonAnchorEntries, anchorEntries)
    const totalBatches = chunks.length

    const totalSendBytes = chunks.reduce(
        (s, chunk) => s + chunk.reduce((cs, e) => cs + e.file.size, 0),
        0,
    )

    let completedSendBytes = 0
    let batchId: string | undefined
    let lastJobResponse: JobResponse | null = null

    for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i]
        const batchIndex = i + 1
        const isLast = batchIndex === totalBatches
        const chunkBytes = chunk.reduce((s, e) => s + e.file.size, 0)

        try {
            lastJobResponse = await uploadSingleChunk(
                chunk,
                profileId,
                onProgress
                    ? (event) => {
                          const sent =
                              event.total && event.total > 0
                                  ? (event.loaded / event.total) * chunkBytes
                                  : 0
                          onProgress({
                              loaded: Math.round(completedSendBytes + sent),
                              total: totalSendBytes,
                              batchIndex,
                              totalBatches,
                          })
                      }
                    : undefined,
                batchId,
                isLast,
            )
            if (batchId === undefined) batchId = lastJobResponse.job_id
            completedSendBytes += chunkBytes
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : 'Upload failed'
            throw new Error(`Batch ${batchIndex}/${totalBatches} failed: ${msg}`, { cause: err })
        }
    }

    return lastJobResponse!
}

export const triggerRescan = apiPost<JobResponse, [body: RescanRequest]>(
    '/import/rescan',
    (body) => ({ data: body }),
)

/**
 * Upload a single Apple Health export file (export.xml or similar).
 * Uses a plain multipart POST — no chunking, no precheck.
 */
export async function importHealthFile(
    file: File,
    profileId?: number | null,
    onUploadProgress?: (fraction: number) => void,
): Promise<{ job_id: string }> {
    const formData = new FormData()
    formData.append('files', file)
    formData.append('import_type', 'health')
    if (profileId != null) formData.append('profile_id', String(profileId))
    const { data } = await api.post<{ job_id: string }>('/import/', formData, {
        onUploadProgress: onUploadProgress
            ? (event: AxiosProgressEvent) => {
                  const fraction = event.total && event.total > 0 ? event.loaded / event.total : 0
                  onUploadProgress(fraction)
              }
            : undefined,
    })
    return data
}

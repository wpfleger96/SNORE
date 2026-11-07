import api from './client'

// Raw api.get required — apiGet wrapper doesn't support responseType: 'blob'

export interface ExportParams {
    from_date?: string
    to_date?: string
    device?: string
}

export interface CsvExportParams extends ExportParams {
    include_waveforms?: boolean
}

export interface RawExportParams extends ExportParams {
    trim_str?: boolean
}

export async function exportCsv(params?: CsvExportParams): Promise<Blob> {
    const { data } = await api.get<Blob>('/export/csv', { params, responseType: 'blob' })
    return data
}

export async function exportJson(params?: ExportParams): Promise<Blob> {
    const { data } = await api.get<Blob>('/export/json', { params, responseType: 'blob' })
    return data
}

export async function exportRaw(params?: RawExportParams): Promise<Blob> {
    const { data } = await api.get<Blob>('/export/raw', { params, responseType: 'blob' })
    return data
}

export function downloadBlob(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    setTimeout(() => URL.revokeObjectURL(url), 1000)
}

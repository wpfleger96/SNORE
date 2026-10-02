import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { ref } from 'vue'

vi.mock('vue-router', () => ({
    useRoute: () => ({ query: {} }),
    useRouter: () => ({ push: vi.fn() }),
}))
vi.mock('@/api/sessions', () => ({ getSession: vi.fn() }))
vi.mock('@/api/events', () => ({ getSessionEvents: vi.fn() }))
vi.mock('@/api/days', () => ({ getDay: vi.fn(), getDates: vi.fn() }))
vi.mock('@/api/stats', () => ({ getDataRange: vi.fn() }))
vi.mock('@/api/health', () => ({ getHealthNight: vi.fn() }))
vi.mock('@/composables/useWaveformWindow', () => ({
    useWaveformWindow: () => ({
        data: ref(null),
        loading: ref(false),
        error: ref(null),
        loadWindow: vi.fn(),
        reset: vi.fn(),
    }),
}))
// Stub InfoHint so stat-card labels render as plain text (no Popover).
vi.mock('@/components/InfoHint.vue', () => ({
    default: { template: '<span class="info-hint-stub" />' },
}))

import { getSession } from '@/api/sessions'
import { getDates } from '@/api/days'
import { getDataRange } from '@/api/stats'
import SessionDetailView from '@/views/SessionDetailView.vue'

const SESSION = {
    id: 1470,
    therapy_day: '2026-04-06',
    start_time: '2026-04-06T23:42:24',
    duration_hours: 9.59,
    device_manufacturer: 'ResMed',
    device_model: 'AirSense11AutoSet',
    therapy_mode: 'APAP',
    event_count: 4,
    has_event_data: false,
    waveform_types: [],
    settings: [],
    statistics: { ahi: 0.52, oai: 0.1, cai: 0.2, hi: 0.22 },
}

function markTiers(el: Element): string[] {
    return [...el.querySelectorAll('button.provenance-mark')].map(
        (b) => b.getAttribute('aria-label')?.split(':')[0] ?? '',
    )
}

async function mountView() {
    const wrapper = mount(SessionDetailView, {
        props: { sessionId: 1470 },
        global: {
            stubs: {
                RouterLink: { template: '<a><slot /></a>' },
                WaveformToolbar: true,
                WaveformChart: true,
                MultiWaveformView: true,
            },
        },
    })
    await flushPromises()
    return wrapper
}

describe('SessionDetailView provenance', () => {
    beforeEach(() => {
        vi.mocked(getSession).mockResolvedValue(SESSION as never)
        vi.mocked(getDates).mockRejectedValue(new Error('offline'))
        vi.mocked(getDataRange).mockRejectedValue(new Error('offline'))
    })

    it('test_header_marks_session_ahi_derived_and_duration_unmarked', async () => {
        const wrapper = await mountView()
        const meta = wrapper.findAll('.session-meta > span')
        const ahi = meta.find((s) => s.text().startsWith('AHI'))
        const duration = meta.find((s) => s.text().includes('hours'))
        expect(ahi && markTiers(ahi.element)).toEqual(['Derived'])
        expect(duration && markTiers(duration.element)).toEqual([])
    })

    it('test_respiratory_index_cards_use_session_statistics_tiers', async () => {
        const wrapper = await mountView()
        const tiers = Object.fromEntries(
            wrapper
                .findAll('.stat-card')
                .map((c) => [c.find('.stat-label').text(), markTiers(c.element)]),
        )
        for (const label of ['AHI', 'OAI', 'CAI', 'HI']) {
            expect(tiers[label], label).toEqual(['Derived'])
        }
    })
})

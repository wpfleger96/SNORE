import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import { computed, ref } from 'vue'
import type { SessionDetail } from '@/types'

vi.mock('vue-router', () => ({
    useRoute: () => ({ query: {} }),
    useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/api/sessions', () => ({ getSession: vi.fn() }))
vi.mock('@/api/events', () => ({ getSessionEvents: vi.fn() }))
vi.mock('@/api/days', () => ({ getDay: vi.fn() }))
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

vi.mock('@/composables/useAvailableDates', () => ({
    useAvailableDates: () => ({
        load: vi.fn(),
        loaded: ref(false),
        minValue: computed(() => undefined),
        maxValue: computed(() => undefined),
        isDateDisabled: () => false,
        sortedDates: computed(() => []),
    }),
    strToCalendarDate: () => undefined,
    calendarDateToStr: () => '',
    adjacentDates: () => ({ prev: null, next: null }),
}))

vi.mock('@/components/WaveformChart.vue', () => ({ default: { template: '<div />' } }))
vi.mock('@/components/WaveformToolbar.vue', () => ({ default: { template: '<div />' } }))
vi.mock('@/components/MultiWaveformView.vue', () => ({ default: { template: '<div />' } }))

// Stub InfoHint so stat-card labels render as plain text (no Popover/teleport).
vi.mock('@/components/InfoHint.vue', () => ({
    default: { template: '<span class="info-hint-stub" />' },
}))

import { getSession } from '@/api/sessions'
import SessionDetailView from '@/views/SessionDetailView.vue'

// Only the fields SessionDetailView reads are populated.
function makeSession(
    statistics: Record<string, unknown> | null,
    overrides: Record<string, unknown> = {},
): SessionDetail {
    return {
        id: 7,
        therapy_day: '2026-05-03',
        start_time: '2026-05-03T22:00:00Z',
        duration_hours: 8,
        event_count: 0,
        waveform_types: [],
        has_event_data: false,
        statistics,
        ...overrides,
    } as unknown as SessionDetail
}

async function mountSession(
    statistics: Record<string, unknown> | null,
    overrides: Record<string, unknown> = {},
): Promise<VueWrapper> {
    vi.mocked(getSession).mockResolvedValue(makeSession(statistics, overrides))
    const wrapper = mount(SessionDetailView, {
        props: { sessionId: 7 },
        global: { stubs: { RouterLink: { template: '<a><slot /></a>' } } },
    })
    await flushPromises()
    return wrapper
}

function findCard(wrapper: VueWrapper, label: string) {
    return wrapper.findAll('.stat-card').find((c) => c.find('.stat-label').text() === label)
}

const OPTIONAL_COUNT_CARDS: [string, string][] = [
    ['Unclassified Apneas', 'unclassified_apneas'],
    ['Mixed Apneas', 'mixed_apneas'],
    ['RERAs', 'reras'],
    ['Flow Limitations', 'flow_limitations'],
]

describe('SessionDetailView respiratory event cards', () => {
    beforeEach(() => {
        vi.clearAllMocks()
    })

    it.each(OPTIONAL_COUNT_CARDS)('test_null_hides_%s_card', async (label, field) => {
        const wrapper = await mountSession({ [field]: null })

        expect(findCard(wrapper, label)).toBeUndefined()
    })

    it.each(OPTIONAL_COUNT_CARDS)('test_zero_shows_%s_card_as_zero', async (label, field) => {
        const wrapper = await mountSession({ [field]: 0 })

        expect(findCard(wrapper, label)?.find('.stat-value').text()).toBe('0')
    })

    it('test_null_rei_hides_card_and_value_shows_it', async () => {
        const hidden = await mountSession({ rei: null })
        expect(findCard(hidden, 'REI')).toBeUndefined()

        const shown = await mountSession({ rei: 2.34 })
        expect(findCard(shown, 'REI')?.find('.stat-value').text()).toBe('2.3')
    })

    it('test_unclassified_apneas_card_follows_central_apneas', async () => {
        const wrapper = await mountSession({ central_apneas: 1, unclassified_apneas: 2 })

        const labels = wrapper.findAll('.stat-label').map((l) => l.text())
        expect(labels[labels.indexOf('Central Apneas') + 1]).toBe('Unclassified Apneas')
    })
})

describe('SessionDetailView AHI sources', () => {
    beforeEach(() => {
        vi.clearAllMocks()
    })

    it('test_device_ahi_card_sits_next_to_the_labelled_recount', async () => {
        const wrapper = await mountSession({ ahi: 2.64, ahi_device: 2.1 })

        const labels = wrapper.findAll('.stat-label').map((l) => l.text())
        expect(labels[labels.indexOf('AHI') + 1]).toBe('Device-reported AHI')
        expect(findCard(wrapper, 'Device-reported AHI')?.find('.stat-value').text()).toBe('2.1')
        expect(findCard(wrapper, 'Device-reported AHI')?.find('.stat-footnote').text()).toBe(
            'Whole therapy day',
        )
        expect(findCard(wrapper, 'AHI')?.find('.stat-footnote').text()).toBe('SNORE recount')
    })

    it('test_null_device_ahi_hides_the_card_and_keeps_the_recount', async () => {
        const wrapper = await mountSession({ ahi: 2.64, ahi_device: null })

        expect(findCard(wrapper, 'Device-reported AHI')).toBeUndefined()
        expect(findCard(wrapper, 'AHI')?.find('.stat-value').text()).toBe('2.6')
    })

    it('test_zero_device_ahi_is_shown', async () => {
        const wrapper = await mountSession({ ahi_device: 0 })

        expect(findCard(wrapper, 'Device-reported AHI')?.find('.stat-value').text()).toBe('0.0')
    })

    it('test_missing_recount_carries_no_source_note', async () => {
        const wrapper = await mountSession({ ahi: null })

        expect(findCard(wrapper, 'AHI')?.find('.stat-footnote').exists()).toBe(false)
    })
})

describe('SessionDetailView device channel and mask-on sections', () => {
    const CHANNEL_STATS = { device_flg_median: 0, device_flg_95th: 0.2, device_flg_max: 0.7 }
    const SPANS = [
        {
            session_id: 7,
            start: '2026-05-03T22:00:00',
            end: '2026-05-04T06:00:00',
            duration_seconds: 28800,
        },
    ]

    beforeEach(() => {
        vi.clearAllMocks()
    })

    function sectionHeadings(wrapper: VueWrapper): string[] {
        return wrapper.findAll('section h2').map((h) => h.text())
    }

    it('test_both_sections_render_from_the_session_payload', async () => {
        const wrapper = await mountSession(
            {},
            { channel_stats: CHANNEL_STATS, mask_on_spans: SPANS },
        )

        expect(sectionHeadings(wrapper)).toEqual(['Flow Limitation & Snore', 'Mask-On Spans'])
    })

    it('test_sections_do_not_depend_on_session_statistics', async () => {
        const wrapper = await mountSession(null, {
            channel_stats: CHANNEL_STATS,
            mask_on_spans: SPANS,
        })

        expect(sectionHeadings(wrapper)).toEqual(['Flow Limitation & Snore', 'Mask-On Spans'])
    })

    it('test_null_fields_hide_both_sections', async () => {
        const wrapper = await mountSession({}, { channel_stats: null, mask_on_spans: null })

        expect(sectionHeadings(wrapper)).toEqual([])
    })
})

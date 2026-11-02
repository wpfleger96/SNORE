import { afterEach, describe, expect, it } from 'vitest'
import { mount, type VueWrapper } from '@vue/test-utils'
import { nextTick } from 'vue'
import type { MaskOnSpan } from '@/types'
import { setMediaMatches } from './matchMedia'
import MaskOnSpansTable from '@/components/MaskOnSpansTable.vue'

// Timezone-naive device wall-clock times, exactly as the API serializes them.
const SPANS: MaskOnSpan[] = [
    {
        session_id: 7,
        start: '2026-04-06T23:42:24',
        end: '2026-04-07T03:10:00',
        duration_seconds: 12456,
    },
    {
        session_id: 7,
        start: '2026-04-07T03:14:30',
        end: '2026-04-07T03:45:00',
        duration_seconds: 1830,
    },
]

function mountSpans(spans: MaskOnSpan[] | null | undefined): VueWrapper {
    return mount(MaskOnSpansTable, { props: { spans } })
}

describe('MaskOnSpansTable', () => {
    afterEach(() => {
        setMediaMatches(false)
    })

    it('test_one_row_per_span_in_api_order_with_clock_durations', () => {
        const rows = mountSpans(SPANS).findAll('tbody tr')

        expect(rows).toHaveLength(2)
        expect(rows[0].findAll('td')[0].text()).toContain('Apr 6, 2026')
        expect(rows[0].findAll('td')[2].text()).toBe('3:27:36')
        expect(rows[1].findAll('td')[2].text()).toBe('0:30:30')
    })

    it('test_wall_clock_times_render_as_the_device_recorded_them', () => {
        const cells = mountSpans(SPANS).findAll('tbody tr')[0].findAll('td')

        expect(cells[0].text()).toContain('11:42:24')
        expect(cells[0].text()).toContain('PM')
        expect(cells[1].text()).toContain('Apr 7, 2026')
        expect(cells[1].text()).toContain('03:10:00')
        expect(cells[1].text()).toContain('AM')
    })

    it('test_short_span_shows_distinct_start_and_end_seconds', () => {
        const cells = mountSpans([
            {
                session_id: 1,
                start: '2026-04-07T03:14:00',
                end: '2026-04-07T03:14:30',
                duration_seconds: 30,
            },
        ])
            .findAll('tbody tr')[0]
            .findAll('td')

        expect(cells[0].text()).toContain('03:14:00')
        expect(cells[1].text()).toContain('03:14:30')
        expect(cells[2].text()).toBe('0:00:30')
    })

    it.each([
        ['null', null],
        ['undefined', undefined],
        ['empty', []],
    ])('test_section_hidden_when_spans_are_%s', (_, spans) => {
        expect(mountSpans(spans).find('section').exists()).toBe(false)
    })

    it('test_mobile_renders_shared_cards_instead_of_a_table', async () => {
        setMediaMatches(true)
        const wrapper = mountSpans(SPANS)
        await nextTick()

        expect(wrapper.find('table').exists()).toBe(false)
        expect(wrapper.find('.card-list').exists()).toBe(true)
        const cards = wrapper.findAll('.data-card')
        expect(cards).toHaveLength(2)
        const secondCardRows = cards[1].findAll('.data-card-row')
        expect(secondCardRows.map((r) => r.find('.data-card-label').text())).toEqual([
            'Start',
            'End',
            'Duration',
        ])
        expect(secondCardRows[0].find('.data-card-value').text()).toContain('03:14:30')
        expect(secondCardRows[2].find('.data-card-value').text()).toBe('0:30:30')
    })
})

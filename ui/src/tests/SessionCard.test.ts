import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import SessionCard from '@/components/SessionCard.vue'
import type { SessionListItem } from '@/types'

const SESSION = {
    id: 1470,
    start_time: '2026-04-06T23:42:24',
    therapy_day: '2026-04-06',
    duration_hours: 9.59,
    ahi: 0.52,
    enabled: true,
    manufacturer: 'ResMed',
    model: 'AirSense11AutoSet',
    serial_number: '23456789012',
} as SessionListItem

function markTiers(el: Element): string[] {
    return [...el.querySelectorAll('button.provenance-mark')].map(
        (b) => b.getAttribute('aria-label')?.split(':')[0] ?? '',
    )
}

function mountCard() {
    return mount(SessionCard, {
        props: { session: SESSION, selected: false, canWrite: true },
        global: { stubs: { RouterLink: { template: '<a><slot /></a>' } } },
    })
}

function row(wrapper: ReturnType<typeof mountCard>, label: string) {
    const found = wrapper
        .findAll('.data-card-row')
        .find((r) => r.find('.data-card-label').text() === label)
    if (!found) throw new Error(`no row labelled ${label}`)
    return found
}

describe('SessionCard provenance', () => {
    it('test_ahi_label_carries_session_list_derived_mark', () => {
        const wrapper = mountCard()
        expect(markTiers(row(wrapper, 'AHI').element)).toEqual(['Derived'])
    })

    it('test_duration_label_is_device_reported_and_unmarked', () => {
        const wrapper = mountCard()
        expect(markTiers(row(wrapper, 'Duration').element)).toEqual([])
    })

    it('test_marks_are_not_inside_the_date_link', () => {
        const wrapper = mountCard()
        expect(wrapper.element.querySelectorAll('a .provenance-mark')).toHaveLength(0)
    })
})

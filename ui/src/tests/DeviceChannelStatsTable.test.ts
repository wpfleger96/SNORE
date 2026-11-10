import { describe, expect, it, vi } from 'vitest'
import { mount, type VueWrapper } from '@vue/test-utils'
import type { DeviceChannelStats } from '@/types'

vi.mock('@/components/InfoHint.vue', () => ({
    default: { template: '<span class="info-hint-stub" />' },
}))

import DeviceChannelStatsTable from '@/components/DeviceChannelStatsTable.vue'

const FULL_STATS: DeviceChannelStats = {
    device_flg_median: 0.05,
    device_flg_95th: 0.214,
    device_flg_max: 0.68,
    snore_median: 0.1,
    snore_95th: 1.5,
    snore_max: 4.25,
    respiratory_rate_median: 12.0,
    respiratory_rate_95th: 14.44,
    respiratory_rate_max: 17.2,
    tidal_volume_median: 520.4,
    tidal_volume_95th: 700,
    tidal_volume_max: 1039.6,
    minute_ventilation_median: 6.5,
    minute_ventilation_95th: 8.36,
    minute_ventilation_max: 11.2,
    ti_median: 1.624,
    ti_95th: 2,
    ti_max: 2,
    ie_ratio_median: 50,
    ie_ratio_95th: 64.04,
    ie_ratio_max: 120,
    pulse_median: 58.4,
    pulse_95th: 71.6,
    pulse_max: 88,
}

function mountStats(stats: DeviceChannelStats | null | undefined): VueWrapper {
    return mount(DeviceChannelStatsTable, { props: { stats } })
}

function bodyRows(wrapper: VueWrapper): string[][] {
    return wrapper.findAll('tbody tr').map((row) => row.findAll('td').map((cell) => cell.text()))
}

describe('DeviceChannelStatsTable', () => {
    it('test_heading_names_all_device_channels', () => {
        expect(mountStats(FULL_STATS).find('h2').text()).toBe('Device Channel Statistics')
    })

    it('test_every_channel_renders_median_95th_max_at_its_own_precision', () => {
        expect(bodyRows(mountStats(FULL_STATS))).toEqual([
            ['Flow Limitation (device)', '0.05', '0.21', '0.68'],
            ['Snore (device)', '0.10', '1.50', '4.25'],
            ['Respiratory Rate (br/min)', '12.0', '14.4', '17.2'],
            ['Tidal Volume (mL)', '520', '700', '1040'],
            ['Minute Ventilation (L/min)', '6.5', '8.4', '11.2'],
            ['Inspiratory Time (s)', '1.62', '2.00', '2.00'],
            ['I:E Ratio (%)', '50.0', '64.0', '120.0'],
            ['Pulse (bpm)', '58', '72', '88'],
        ])
    })

    it('test_no_oximeter_hides_only_the_pulse_row', () => {
        const wrapper = mountStats({
            ...FULL_STATS,
            pulse_median: null,
            pulse_95th: null,
            pulse_max: null,
        })

        const labels = bodyRows(wrapper).map((row) => row[0])
        expect(labels).toHaveLength(7)
        expect(labels).not.toContain('Pulse (bpm)')
    })

    it('test_zero_values_render_as_real_measurements', () => {
        const wrapper = mountStats({
            ...FULL_STATS,
            device_flg_median: 0,
            device_flg_95th: 0,
            device_flg_max: 0,
        })

        expect(bodyRows(wrapper)[0]).toEqual(['Flow Limitation (device)', '0.00', '0.00', '0.00'])
    })

    it('test_missing_channel_hides_only_its_row', () => {
        const wrapper = mountStats({
            device_flg_median: null,
            device_flg_95th: null,
            device_flg_max: null,
            snore_median: 0,
            snore_95th: 1.5,
            snore_max: 4.25,
        })

        expect(bodyRows(wrapper)).toEqual([['Snore (device)', '0.00', '1.50', '4.25']])
    })

    it('test_partly_null_channel_keeps_its_row_with_a_placeholder', () => {
        const wrapper = mountStats({ snore_median: null, snore_95th: 1.5, snore_max: null })

        expect(bodyRows(wrapper)).toEqual([['Snore (device)', '---', '1.50', '---']])
    })

    it.each([
        ['null', null],
        ['undefined', undefined],
        ['all channels null', { device_flg_median: null, snore_max: null }],
    ])('test_section_hidden_when_stats_are_%s', (_, stats) => {
        expect(mountStats(stats).find('section').exists()).toBe(false)
    })
})

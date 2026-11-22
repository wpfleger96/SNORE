import { afterEach, describe, expect, it, vi } from 'vitest'
import { mount, type VueWrapper } from '@vue/test-utils'
import type { DeviceChannelStats } from '@/types'
import { setMediaMatches } from './matchMedia'

vi.mock('@/components/InfoHint.vue', () => ({
    default: { template: '<span class="info-hint-stub" />' },
}))

import DeviceChannelStatsTable from '@/components/DeviceChannelStatsTable.vue'

const FULL_STATS: DeviceChannelStats = {
    device_flg_min: 0,
    device_flg_median: 0.05,
    device_flg_95th: 0.214,
    device_flg_99_5th: 0.5,
    device_flg_max: 0.68,
    snore_min: 0,
    snore_median: 0.1,
    snore_95th: 1.5,
    snore_99_5th: 3,
    snore_max: 4.25,
    respiratory_rate_min: 8.0,
    respiratory_rate_median: 12.0,
    respiratory_rate_95th: 14.44,
    respiratory_rate_99_5th: 16.5,
    respiratory_rate_max: 17.2,
    tidal_volume_min: 300.4,
    tidal_volume_median: 520.4,
    tidal_volume_95th: 700,
    tidal_volume_99_5th: 950,
    tidal_volume_max: 1039.6,
    minute_ventilation_min: 3.2,
    minute_ventilation_median: 6.5,
    minute_ventilation_95th: 8.36,
    minute_ventilation_99_5th: 10.04,
    minute_ventilation_max: 11.2,
    ti_min: 0.8,
    ti_median: 1.624,
    ti_95th: 2,
    ti_99_5th: 2,
    ti_max: 2,
    ie_ratio_min: 30,
    ie_ratio_median: 50,
    ie_ratio_95th: 64.04,
    ie_ratio_99_5th: 110,
    ie_ratio_max: 120,
    pulse_min: 48,
    pulse_median: 58.4,
    pulse_95th: 71.6,
    pulse_99_5th: 84,
    pulse_max: 88,
}

function mountStats(stats: DeviceChannelStats | null | undefined): VueWrapper {
    return mount(DeviceChannelStatsTable, { props: { stats } })
}

function bodyRows(wrapper: VueWrapper): string[][] {
    return wrapper.findAll('tbody tr').map((row) => row.findAll('td').map((cell) => cell.text()))
}

describe('DeviceChannelStatsTable', () => {
    afterEach(() => setMediaMatches(false))

    it('test_headers_name_each_statistic_in_column_order', () => {
        const headers = mountStats(FULL_STATS)
            .findAll('thead th')
            .map((cell) => cell.text())
        expect(headers).toEqual(['Channel', 'Min', 'Median', '95th', '99.5th', 'Max'])
    })

    it('test_mobile_renders_one_card_per_channel_with_labelled_statistics', () => {
        setMediaMatches(true)
        const wrapper = mountStats({ snore_median: 0.1, snore_max: 4.25 })

        expect(wrapper.find('table').exists()).toBe(false)
        const cards = wrapper.findAll('.data-card')
        expect(cards).toHaveLength(1)
        expect(cards[0]!.find('.data-card-header').text()).toBe('Snore (device)')
        expect(
            cards[0]!
                .findAll('.data-card-row')
                .map((row) => [
                    row.find('.data-card-label').text(),
                    row.find('.data-card-value').text(),
                ]),
        ).toEqual([
            ['Min', '---'],
            ['Median', '0.10'],
            ['95th', '---'],
            ['99.5th', '---'],
            ['Max', '4.25'],
        ])
    })

    it('test_heading_names_all_device_channels', () => {
        expect(mountStats(FULL_STATS).find('h2').text()).toBe('Device Channel Statistics')
    })

    it('test_every_channel_renders_every_statistic_at_its_own_precision', () => {
        expect(bodyRows(mountStats(FULL_STATS))).toEqual([
            ['Flow Limitation (device)', '0.00', '0.05', '0.21', '0.50', '0.68'],
            ['Snore (device)', '0.00', '0.10', '1.50', '3.00', '4.25'],
            ['Respiratory Rate (br/min)', '8.0', '12.0', '14.4', '16.5', '17.2'],
            ['Tidal Volume (mL)', '300', '520', '700', '950', '1040'],
            ['Minute Ventilation (L/min)', '3.2', '6.5', '8.4', '10.0', '11.2'],
            ['Inspiratory Time (s)', '0.80', '1.62', '2.00', '2.00', '2.00'],
            ['I:E Ratio (%)', '30.0', '50.0', '64.0', '110.0', '120.0'],
            ['Pulse (bpm)', '48', '58', '72', '84', '88'],
        ])
    })

    it('test_no_oximeter_hides_only_the_pulse_row', () => {
        const wrapper = mountStats({
            ...FULL_STATS,
            pulse_min: null,
            pulse_median: null,
            pulse_95th: null,
            pulse_99_5th: null,
            pulse_max: null,
        })

        const labels = bodyRows(wrapper).map((row) => row[0])
        expect(labels).toHaveLength(7)
        expect(labels).not.toContain('Pulse (bpm)')
    })

    it('test_zero_values_render_as_real_measurements', () => {
        const wrapper = mountStats({
            ...FULL_STATS,
            device_flg_min: 0,
            device_flg_median: 0,
            device_flg_95th: 0,
            device_flg_99_5th: 0,
            device_flg_max: 0,
        })

        expect(bodyRows(wrapper)[0]).toEqual([
            'Flow Limitation (device)',
            '0.00',
            '0.00',
            '0.00',
            '0.00',
            '0.00',
        ])
    })

    it('test_missing_channel_hides_only_its_row', () => {
        const wrapper = mountStats({
            device_flg_min: null,
            device_flg_median: null,
            device_flg_95th: null,
            device_flg_99_5th: null,
            device_flg_max: null,
            snore_min: 0,
            snore_median: 0,
            snore_95th: 1.5,
            snore_99_5th: 3,
            snore_max: 4.25,
        })

        expect(bodyRows(wrapper)).toEqual([
            ['Snore (device)', '0.00', '0.00', '1.50', '3.00', '4.25'],
        ])
    })

    it('test_partly_null_channel_keeps_its_row_with_a_placeholder', () => {
        const wrapper = mountStats({ snore_median: null, snore_95th: 1.5, snore_max: null })

        expect(bodyRows(wrapper)).toEqual([['Snore (device)', '---', '---', '1.50', '---', '---']])
    })

    it.each([
        ['null', null],
        ['undefined', undefined],
        ['all channels null', { device_flg_median: null, snore_max: null }],
    ])('test_section_hidden_when_stats_are_%s', (_, stats) => {
        expect(mountStats(stats).find('section').exists()).toBe(false)
    })
})

import { describe, expect, it } from 'vitest'
import { deviceFilterOptions } from '@/utils/deviceOptions'

const device = (model: string, serial_number: string, manufacturer = 'ResMed') => ({
    manufacturer,
    model,
    serial_number,
})

describe('deviceFilterOptions', () => {
    it('test_unique_model_label_omits_serial_and_value_is_serial', () => {
        expect(deviceFilterOptions([device('AirSense 11', 'SN1')])).toEqual([
            { label: 'ResMed AirSense 11', value: 'SN1' },
        ])
    })

    it('test_shared_model_labels_append_serial', () => {
        const options = deviceFilterOptions([
            device('AirSense 11', 'SN1'),
            device('AirSense 10', 'SN2'),
            device('AirSense 11', 'SN3'),
        ])

        expect(options).toEqual([
            { label: 'ResMed AirSense 11 (SN1)', value: 'SN1' },
            { label: 'ResMed AirSense 10', value: 'SN2' },
            { label: 'ResMed AirSense 11 (SN3)', value: 'SN3' },
        ])
    })

    it('test_same_model_different_manufacturer_is_not_ambiguous', () => {
        const options = deviceFilterOptions([device('X1', 'SN1', 'A'), device('X1', 'SN2', 'B')])

        expect(options.map((o) => o.label)).toEqual(['A X1', 'B X1'])
    })

    it('test_no_devices_returns_no_options', () => {
        expect(deviceFilterOptions([])).toEqual([])
    })
})

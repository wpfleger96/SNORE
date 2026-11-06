import type { DeviceInfo } from '@/types'

export interface DeviceOption {
    label: string
    value: string
}

/**
 * Device filter options. The value is the serial number, which is what the
 * API's `device` param matches; the label is manufacturer + model, with the
 * serial appended only when another listed device has the same name. Devices
 * with an empty serial are left out: the API cannot filter by it, and an empty
 * value is not a valid select item.
 */
export function deviceFilterOptions(
    allDevices: Pick<DeviceInfo, 'manufacturer' | 'model' | 'serial_number'>[],
): DeviceOption[] {
    const devices = allDevices.filter((d) => d.serial_number)
    const name = (d: (typeof devices)[number]): string => `${d.manufacturer} ${d.model}`
    const counts = new Map<string, number>()
    for (const d of devices) counts.set(name(d), (counts.get(name(d)) ?? 0) + 1)
    return devices.map((d) => ({
        label: counts.get(name(d))! > 1 ? `${name(d)} (${d.serial_number})` : name(d),
        value: d.serial_number,
    }))
}

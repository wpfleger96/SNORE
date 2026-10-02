import { flushPromises, type VueWrapper } from '@vue/test-utils'

// Helpers for driving the real reka-ui Select. The wrapper must be mounted with
// `attachTo: document.body` (content is portalled). The jsdom stubs Select
// needs are installed in `tests/setup.ts`.

/** Open the Select at `trigger` and return its rendered options. */
export async function openSelect(
    wrapper: VueWrapper,
    trigger = '[role="combobox"]',
): Promise<HTMLElement[]> {
    wrapper
        .get(trigger)
        .element.dispatchEvent(
            new PointerEvent('pointerdown', { bubbles: true, button: 0, pointerType: 'mouse' }),
        )
    await flushPromises()
    return Array.from(document.body.querySelectorAll<HTMLElement>('[role="option"]'))
}

/** Label of the option the open Select marks as selected, or null when none is. */
export function selectedOptionLabel(options: HTMLElement[]): string | null {
    const selected = options.find((el) => el.getAttribute('aria-selected') === 'true')
    return selected?.textContent?.trim() ?? null
}

/** Open the Select at `trigger` and choose the option labelled `label`. */
export async function chooseSelectOption(
    wrapper: VueWrapper,
    label: string,
    trigger = '[role="combobox"]',
): Promise<void> {
    const option = (await openSelect(wrapper, trigger)).find(
        (el) => el.textContent?.trim() === label,
    )
    if (!option) throw new Error(`Select option "${label}" not rendered`)
    option.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerType: 'mouse' }))
    await flushPromises()
}

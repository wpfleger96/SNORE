import { flushPromises } from '@vue/test-utils'

// Helpers for driving AnalysisModePicker by its visible labels. `root` is the
// mounted wrapper's element, or document.body when the picker is in a dialog.

type PickerGroup = 'Modes' | 'Primary mode'

/** The toggle group under the `group` label, or null when it is not rendered. */
export function pickerGroup(root: ParentNode, group: PickerGroup): HTMLElement | null {
    const label = [...root.querySelectorAll('span')].find((s) => s.textContent?.trim() === group)
    return label ? root.querySelector<HTMLElement>(`[aria-labelledby="${label.id}"]`) : null
}

/** Labels of the pressed items in the `group` toggle group. */
export function pressedModes(root: ParentNode, group: PickerGroup): string[] {
    return [...(pickerGroup(root, group)?.querySelectorAll('button[data-state="on"]') ?? [])].map(
        (b) => b.textContent!.trim(),
    )
}

/** Click the item labelled `mode` in the `group` toggle group. */
export async function clickMode(root: ParentNode, group: PickerGroup, mode: string): Promise<void> {
    const button = [...(pickerGroup(root, group)?.querySelectorAll('button') ?? [])].find(
        (b) => b.textContent?.trim() === mode,
    )
    if (!button) throw new Error(`${group} item "${mode}" not rendered`)
    button.click()
    await flushPromises()
}

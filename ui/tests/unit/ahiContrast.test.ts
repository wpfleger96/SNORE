import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { AHI_COLOR_SCALE } from '@/utils/ahiScale'

// Read from disk: Vitest stubs CSS imports, `?raw` included, to an empty string.
const CSS = readFileSync(
    resolve(fileURLToPath(import.meta.url), '../../../src/assets/tailwind.css'),
    'utf8',
)

const THEMES = ['light', 'dark'] as const
type Theme = (typeof THEMES)[number]
const BACKGROUNDS = ['--card', '--muted']
const TIERS = AHI_COLOR_SCALE.map((entry) => entry.tier)
const WCAG_AA_TEXT = 4.5

/** Custom properties declared in the top-level `selector { ... }` block of tailwind.css. */
function declarations(selector: ':root' | '.dark'): Map<string, string> {
    const css = CSS.replace(/\/\*[\s\S]*?\*\//g, '')
    const escaped = selector.replace('.', '\\.')
    const block = new RegExp(`^${escaped}\\s*\\{([^}]*)\\}`, 'm').exec(css)
    if (!block) throw new Error(`tailwind.css has no top-level ${selector} block`)
    return new Map(
        [...block[1]!.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)].map((m) => [m[1]!, m[2]!.trim()]),
    )
}

const ROOT = declarations(':root')
const THEME_VARS: Record<Theme, Map<string, string>> = {
    light: ROOT,
    dark: new Map([...ROOT, ...declarations('.dark')]),
}

/** WCAG 2.x relative luminance of `--name` in `theme`, which must be an `oklch()` colour. */
function luminance(theme: Theme, name: string): number {
    const value = THEME_VARS[theme].get(name)
    const match =
        value &&
        /^oklch\(\s*([\d.]+)(%?)\s+([\d.]+)\s+([\d.]+)(?:deg)?\s*(?:\/[^)]*)?\)$/.exec(value)
    if (!match) {
        throw new Error(
            `${theme} ${name} is not a parsable oklch() colour: ${value ?? 'undeclared'}`,
        )
    }
    const L = Number(match[1]) / (match[2] ? 100 : 1)
    const hue = (Number(match[4]) * Math.PI) / 180
    const [a, b] = [Number(match[3]) * Math.cos(hue), Number(match[3]) * Math.sin(hue)]
    // OKLab -> LMS -> linear sRGB (Ottosson, https://bottosson.github.io/posts/oklab/).
    const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
    const rgb = [
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
    ].map((c) => Math.min(1, Math.max(0, c)))
    return 0.2126 * rgb[0]! + 0.7152 * rgb[1]! + 0.0722 * rgb[2]!
}

function contrast(theme: Theme, fg: string, bg: string): number {
    const [hi, lo] = [luminance(theme, fg), luminance(theme, bg)].sort((x, y) => y - x)
    return (hi! + 0.05) / (lo! + 0.05)
}

describe('AHI tier text contrast', () => {
    const cases = THEMES.flatMap((theme) =>
        TIERS.flatMap((tier) => BACKGROUNDS.map((bg) => ({ theme, tier, bg }))),
    )

    it.each(cases)('$theme --ahi-$tier on $bg meets WCAG AA', ({ theme, tier, bg }) => {
        expect(contrast(theme, `--ahi-${tier}`, bg)).toBeGreaterThanOrEqual(WCAG_AA_TEXT)
    })
})

describe('AHI legend scale', () => {
    it.each(AHI_COLOR_SCALE)('$tier swatch uses the shared fill token', ({ tier, color }) => {
        expect(color).toBe(`var(--ahi-${tier}-fill)`)
        expect(ROOT.has(`--ahi-${tier}-fill`), `:root declares --ahi-${tier}-fill`).toBe(true)
    })
})

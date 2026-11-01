import { computed, ref, watch } from 'vue'
import type { Ref } from 'vue'

/** One page of `rows` for `PaginationBar`, returning to the first page whenever `rows` changes. */
export function usePagedRows<T>(rows: Ref<T[]>, pageSize = 25) {
    const page = ref(0)
    const pageRows = computed(() =>
        rows.value.slice(page.value * pageSize, (page.value + 1) * pageSize),
    )
    const barProps = computed(() => ({
        offset: page.value * pageSize,
        pageSize,
        total: rows.value.length,
    }))
    function setOffset(offset: number): void {
        page.value = Math.floor(offset / pageSize)
    }
    watch(rows, () => {
        page.value = 0
    })
    return { pageRows, barProps, setOffset }
}

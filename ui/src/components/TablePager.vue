<template>
    <div v-if="totalPages > 1" class="flex items-center justify-between px-2 py-4">
        <span class="text-sm text-muted-foreground"> Page {{ page + 1 }} of {{ totalPages }} </span>
        <div class="flex gap-2">
            <Button variant="outline" size="sm" :disabled="page === 0" @click="page--">
                Previous
            </Button>
            <Button variant="outline" size="sm" :disabled="page >= totalPages - 1" @click="page++">
                Next
            </Button>
        </div>
    </div>
</template>

<script setup lang="ts">
import { Button } from '@/components/ui/button'

defineProps<{ totalPages: number }>()
const page = defineModel<number>('page', { required: true })
</script>

<template>
  <section class="stage">
    <div class="stage-head">
      <span class="stage-num">3</span>
      <span class="stage-title">Prompt</span>
    </div>
    <div class="stage-body">
      <div v-if="exp.promptMarkdown.value" class="prompt-block">
        <pre>{{ exp.promptMarkdown.value }}</pre>
      </div>
      <p v-else class="stage-empty">注入后显示 Prompt</p>
    </div>
  </section>
</template>

<script setup>
import { inject } from "vue";
const exp = inject("exp");
</script>

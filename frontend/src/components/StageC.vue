<template>
  <section>
    <h2>Stage C · 送入 LLM 的 Prompt</h2>
    <div v-if="exp.promptMarkdown.value" class="prompt-block">
      <pre>{{ exp.promptMarkdown.value }}</pre>
    </div>
    <p v-else class="stage-summary">注入后显示 System / User 提示词</p>
  </section>
</template>

<script setup>
import { inject } from "vue";
const exp = inject("exp");
</script>

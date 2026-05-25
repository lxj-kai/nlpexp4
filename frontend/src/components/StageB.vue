<template>
  <section>
    <h2>Stage B · 噪音注入后文档</h2>
    <div v-if="exp.injectSummary.value" class="inject-md" v-html="exp.renderMd(exp.injectSummary.value)" />
    <div v-if="exp.injectedHtml.value" v-html="exp.injectedHtml.value" class="doc-stage" />
    <p v-else class="stage-summary">点击「仅注入」或「运行完整流水线」</p>
  </section>
</template>

<script setup>
import { inject } from "vue";
const exp = inject("exp");
</script>

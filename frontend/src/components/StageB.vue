<template>
  <section class="stage">
    <div class="stage-head">
      <span class="stage-num">2</span>
      <span class="stage-title">噪音注入</span>
    </div>
    <div class="stage-body">
      <div v-if="exp.injectSummary.value" class="inject-md" v-html="exp.renderMd(exp.injectSummary.value)" />
      <div v-if="exp.injectedHtml.value" v-html="exp.injectedHtml.value" class="doc-stage" />
      <p v-else class="stage-empty">点击「预览」或「运行」</p>
    </div>
  </section>
</template>

<script setup>
import { inject } from "vue";
const exp = inject("exp");
</script>

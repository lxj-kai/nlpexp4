<template>
  <section>
    <h2>Stage A · 原始检索池</h2>
    <div class="field-row">
      <div class="field">
        <label>问题</label>
        <div class="readonly-box">{{ exp.query.value || "—" }}</div>
      </div>
      <div class="field">
        <label>标准答案</label>
        <div class="readonly-box">{{ exp.gold.value || "—" }}</div>
      </div>
    </div>
    <div v-if="exp.retrievalHtml.value" v-html="exp.retrievalHtml.value" class="doc-stage" />
    <p v-else class="stage-summary">选择样本后自动加载检索池</p>
  </section>
</template>

<script setup>
import { inject } from "vue";
const exp = inject("exp");
</script>

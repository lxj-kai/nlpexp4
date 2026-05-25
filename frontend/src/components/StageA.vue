<template>
  <section class="stage">
    <div class="stage-head">
      <span class="stage-num">1</span>
      <span class="stage-title">检索池</span>
    </div>
    <div class="stage-body">
      <div class="qa-row">
        <div class="qa-item">
          <label>Q 问题</label>
          <div class="val">{{ exp.query.value || "—" }}</div>
        </div>
        <div class="qa-item">
          <label>A 标准答案</label>
          <div class="val">{{ exp.gold.value || "—" }}</div>
        </div>
      </div>
      <div v-if="exp.retrievalHtml.value" v-html="exp.retrievalHtml.value" class="doc-stage" />
      <p v-else class="stage-empty">选择样本后加载</p>
    </div>
  </section>
</template>

<script setup>
import { inject } from "vue";
const exp = inject("exp");
</script>

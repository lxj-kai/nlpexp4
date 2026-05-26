<template>
  <section class="stage">
    <div class="stage-head">
      <span class="stage-num">4</span>
      <span class="stage-title">答案与评测</span>
    </div>
    <div class="stage-body">
      <template v-if="exp.runResult.value">
        <div class="answer-block">
          <strong>模型预测</strong>
          <blockquote>{{ exp.runResult.value.prediction }}</blockquote>
        </div>

        <div class="metrics-grid">
          <div class="metric-card highlight">
            <div class="label">F1</div>
            <div class="value">{{ exp.fmt(exp.runResult.value.metrics.token_f1) }}</div>
          </div>
          <div class="metric-card">
            <div class="label">EM</div>
            <div class="value">{{ exp.fmt(exp.runResult.value.metrics.em) }}</div>
          </div>
          <div class="metric-card">
            <div class="label">Contains</div>
            <div class="value">{{ exp.fmt(exp.runResult.value.metrics.contains) }}</div>
          </div>
          <div class="metric-card">
            <div class="label">ROUGE-L</div>
            <div class="value">{{ exp.fmt(exp.runResult.value.metrics.rouge_l) }}</div>
          </div>
          <div class="metric-card highlight">
            <div class="label">ISR</div>
            <div class="value">{{ exp.fmt(exp.runResult.value.metrics.isr) }}</div>
          </div>
          <div class="metric-card highlight">
            <div class="label">NAR</div>
            <div class="value">{{ exp.fmt(exp.runResult.value.metrics.nar) }}</div>
          </div>
        </div>

        <div style="text-align:center">
          <span class="verdict-badge" :class="exp.runResult.value.metrics.verdict">
            {{ exp.verdictLabel(exp.runResult.value.metrics.verdict) }}
          </span>
        </div>

        <div class="meta-line" v-if="exp.runResult.value.meta">
          {{ exp.runResult.value.meta.method }} ·
          {{ exp.runResult.value.meta.prompt_tokens }}+{{ exp.runResult.value.meta.completion_tokens }} tokens ·
          {{ exp.runResult.value.meta.latency }}s
          <template v-if="exp.runResult.value.meta.cached"> · cached</template>
        </div>
      </template>
      <p v-else class="stage-empty">运行后显示结果</p>
    </div>
  </section>
</template>

<script setup>
import { inject } from "vue";
const exp = inject("exp");
</script>

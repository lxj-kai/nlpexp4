<template>
  <section>
    <h2>Stage D · 模型回答与评测</h2>
    <template v-if="exp.runResult.value">
      <div class="answer-block">
        <strong>模型预测</strong>
        <blockquote>{{ exp.runResult.value.prediction }}</blockquote>
      </div>
      <table class="metrics-table">
        <thead>
          <tr><th>指标</th><th>值</th></tr>
        </thead>
        <tbody>
          <tr><td>EM</td><td>{{ exp.fmt(exp.runResult.value.metrics.em) }}</td></tr>
          <tr><td>Contains</td><td>{{ exp.fmt(exp.runResult.value.metrics.contains) }}</td></tr>
          <tr><td>Token-F1</td><td>{{ exp.fmt(exp.runResult.value.metrics.token_f1) }}</td></tr>
          <tr><td>ROUGE-L</td><td>{{ exp.fmt(exp.runResult.value.metrics.rouge_l) }}</td></tr>
          <tr><td>ISR</td><td>{{ exp.fmt(exp.runResult.value.metrics.isr) }}</td></tr>
          <tr><td>NAR</td><td>{{ exp.fmt(exp.runResult.value.metrics.nar) }}</td></tr>
          <tr>
            <td>判定</td>
            <td>
              <span class="verdict" :class="exp.runResult.value.metrics.verdict">
                {{ exp.verdictLabel(exp.runResult.value.metrics.verdict) }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
      <p class="stage-summary">
        method={{ exp.runResult.value.meta?.method }} ·
        prompt_tokens={{ exp.runResult.value.meta?.prompt_tokens }} ·
        completion_tokens={{ exp.runResult.value.meta?.completion_tokens }} ·
        latency={{ exp.runResult.value.meta?.latency }}s ·
        cached={{ exp.runResult.value.meta?.cached }}
      </p>
    </template>
    <p v-else class="stage-summary">运行完整流水线后显示</p>
  </section>
</template>

<script setup>
import { inject } from "vue";
const exp = inject("exp");
</script>

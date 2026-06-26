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
          <blockquote>{{ exp.runResult.value.prediction?.trim() || "（空）" }}</blockquote>
        </div>

        <div class="metrics-grid">
          <div class="metric-card highlight">
            <div class="label">Judge 分数</div>
            <div class="value">{{ exp.fmt(exp.runResult.value.metrics.judge_score) }}</div>
            <div class="sub" v-if="exp.runResult.value.meta?.judge_model">
              {{ exp.runResult.value.meta.judge_model }}
            </div>
          </div>
          <div class="metric-card highlight">
            <div class="label">Judge 正确</div>
            <div class="value">
              {{ exp.runResult.value.metrics.judge_correct === 1 ? "是" : exp.runResult.value.metrics.judge_correct === 0 ? "否" : "—" }}
            </div>
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
          生成 {{ exp.runResult.value.meta.generation_model || "—" }}
          · 审查 {{ exp.runResult.value.meta.judge_model || exp.judgeModel.value || "—" }}
          · {{ exp.runResult.value.meta.method }}
          · {{ exp.runResult.value.meta.prompt_tokens }}+{{ exp.runResult.value.meta.completion_tokens }} tokens
          · {{ exp.runResult.value.meta.latency?.toFixed?.(2) ?? exp.runResult.value.meta.latency }}s
          <template v-if="exp.runResult.value.meta.cached"> · cached</template>
        </div>
      </template>
      <p v-else class="stage-empty">运行后显示结果（审查由 DeepSeek 完成）</p>
    </div>
  </section>
</template>

<script setup>
import { inject } from "vue";
const exp = inject("exp");
</script>

<style scoped>
.metric-card .sub {
  font-size: 0.7rem;
  color: #64748b;
  margin-top: 0.25rem;
  word-break: break-all;
}
</style>

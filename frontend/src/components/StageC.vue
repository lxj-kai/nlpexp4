<template>
  <section class="stage">
    <div class="stage-head">
      <span class="stage-num">3</span>
      <span class="stage-title">方法流程与 Prompt</span>
    </div>
    <div class="stage-body">
      <div class="method-flow">
        <div class="method-flow-head">
          <div>
            <div class="method-name">{{ exp.methodFlow.value.label }}</div>
            <div class="method-summary">{{ exp.methodFlow.value.summary }}</div>
          </div>
          <div class="method-tags">
            <span class="method-badge" :class="exp.methodFlow.value.badge">
              {{ exp.methodFlow.value.badge }}
            </span>
            <span class="method-cost">{{ exp.methodFlow.value.cost }}</span>
          </div>
        </div>
        <div class="flow-steps">
          <template v-for="(step, idx) in exp.methodFlow.value.steps" :key="step">
            <div class="flow-step">
              <span class="flow-index">{{ idx + 1 }}</span>
              <span>{{ step }}</span>
            </div>
            <span v-if="idx < exp.methodFlow.value.steps.length - 1" class="flow-arrow">→</span>
          </template>
        </div>
      </div>
      <div v-if="visibleCalls.length" class="prompt-call-list">
        <article
          v-for="(call, idx) in visibleCalls"
          :key="`${call.title}-${idx}`"
          class="prompt-call"
        >
          <div class="prompt-call-head">
            <span>{{ call.title || `Call ${idx + 1}` }}</span>
            <em>{{ call.output ? "提示词 / 输出" : "提示词" }}</em>
          </div>
          <div class="prompt-block">
            <details class="prompt-part" :open="idx === 0">
              <summary>提示词</summary>
              <pre>{{ call.prompt_markdown }}</pre>
            </details>
            <details v-if="call.output" class="prompt-part" :open="idx === 0">
              <summary>输出</summary>
              <pre class="call-output">{{ call.output }}</pre>
            </details>
          </div>
        </article>
      </div>
      <div v-else-if="exp.promptMarkdown.value" class="prompt-block">
        <pre>{{ exp.promptMarkdown.value }}</pre>
      </div>
      <p v-else class="stage-empty">注入后显示 Prompt</p>
    </div>
  </section>
</template>

<script setup>
import { computed } from "vue";
import { inject } from "vue";
const exp = inject("exp");

const visibleCalls = computed(() => exp.promptCalls.value || []);
</script>

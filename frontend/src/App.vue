<template>
  <div class="app" :class="{ loading: exp.busy.value }">
    <header class="app-header">
      <h1>面向 RAG 噪音的鲁棒性推理实验平台</h1>
      <div class="tab-bar">
        <button :class="{ active: tab === 'experiment' }" @click="tab = 'experiment'">实验平台</button>
        <button :class="{ active: tab === 'benchmark' }" @click="tab = 'benchmark'">数据引擎</button>
      </div>
    </header>

    <template v-if="tab === 'experiment'">
      <ParamPanel />
      <div class="pipeline">
        <StageA />
        <div class="arrow">→</div>
        <StageB />
        <div class="arrow">→</div>
        <StageC />
        <div class="arrow">→</div>
        <StageD />
      </div>
    </template>

    <BenchmarkEngine v-else />
  </div>
</template>

<script setup>
import { provide, ref } from "vue";
import { useExperiment } from "./composables/useExperiment.js";
import ParamPanel from "./components/ParamPanel.vue";
import StageA from "./components/StageA.vue";
import StageB from "./components/StageB.vue";
import StageC from "./components/StageC.vue";
import StageD from "./components/StageD.vue";
import BenchmarkEngine from "./components/BenchmarkEngine.vue";

const exp = useExperiment();
provide("exp", exp);

const tab = ref("experiment");
</script>

<style>
.tab-bar {
  display: flex;
  gap: 4px;
  margin-left: auto;
}

.tab-bar button {
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 600;
  padding: 4px 14px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: #fff;
  cursor: pointer;
  color: var(--ink-muted);
  transition: all 0.15s;
}

.tab-bar button.active {
  background: var(--blue);
  color: #fff;
  border-color: var(--blue);
}

.tab-bar button:hover:not(.active) {
  background: var(--surface);
}
</style>

<template>
  <div class="app" :class="{ loading: exp.busy.value }">
    <header class="app-header">
      <h1>面向 RAG 噪音的鲁棒性推理实验平台</h1>
      <span class="lead">RGB · 噪音注入 · 矫正方法 · 鲁棒性评估</span>
    </header>

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
  </div>
</template>

<script setup>
import { provide } from "vue";
import { useExperiment } from "./composables/useExperiment.js";
import ParamPanel from "./components/ParamPanel.vue";
import StageA from "./components/StageA.vue";
import StageB from "./components/StageB.vue";
import StageC from "./components/StageC.vue";
import StageD from "./components/StageD.vue";

const exp = useExperiment();
provide("exp", exp);
</script>

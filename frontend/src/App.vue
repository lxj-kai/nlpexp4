<template>
  <div class="app" :class="{ loading: exp.busy.value }">
    <header class="app-header">
      <h1>面向 RAG 检索噪音的鲁棒性推理实验平台</h1>
      <p class="lead">
        RGB 数据集 · 噪音注入（比例 / 类型 / 位置）· Naive RAG 与多种矫正方法 · EM / F1 / ROUGE-L / ISR / NAR
      </p>
    </header>

    <div class="layout">
      <ParamPanel />
      <main class="main">
        <StageA />
        <StageB />
        <StageC />
        <StageD />
      </main>
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

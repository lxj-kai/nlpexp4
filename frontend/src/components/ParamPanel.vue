<template>
  <div class="toolbar">
    <div class="field">
      <label>语言</label>
      <select :value="exp.language.value" @change="exp.language.value = $event.target.value; exp.onLangSubsetChange()">
        <option v-for="l in exp.languages.value" :key="l" :value="l">{{ l === "zh" ? "中文" : "English" }}</option>
      </select>
    </div>

    <div class="field">
      <label>子集</label>
      <select :value="exp.subset.value" @change="exp.subset.value = $event.target.value; exp.onLangSubsetChange()">
        <option v-for="s in exp.subsets.value" :key="s" :value="s">{{ s }}</option>
      </select>
    </div>

    <div class="field">
      <label>样本</label>
      <select :value="exp.sampleId.value" @change="exp.sampleId.value = +$event.target.value; exp.loadSample()">
        <option v-for="item in exp.samples.value" :key="item.id" :value="item.id">{{ item.label }}</option>
      </select>
    </div>

    <div class="sep" />

    <div class="field">
      <label>噪音</label>
      <input type="range" min="0" max="1" step="0.05"
             :value="exp.noiseRatio.value"
             @input="exp.noiseRatio.value = +$event.target.value" />
      <span class="ratio-val">{{ exp.noiseRatio.value.toFixed(2) }}</span>
    </div>

    <div class="field">
      <label>类型</label>
      <select :value="exp.noiseType.value" @change="exp.noiseType.value = $event.target.value">
        <option v-for="t in exp.noiseTypes.value" :key="t" :value="t">{{ t }}</option>
      </select>
    </div>

    <div class="field">
      <label>位置</label>
      <select :value="exp.noisePosition.value" @change="exp.noisePosition.value = $event.target.value">
        <option v-for="p in exp.noisePositions.value" :key="p" :value="p">{{ p }}</option>
      </select>
    </div>

    <div class="sep" />

    <div class="field">
      <label>方法</label>
      <select :value="exp.method.value" @change="exp.method.value = $event.target.value">
        <option v-for="m in exp.methods.value" :key="m" :value="m">{{ m }}</option>
      </select>
    </div>

    <button type="button" class="btn" :disabled="exp.busy.value" @click="exp.doInject()">预览</button>
    <button type="button" class="btn btn-primary" :disabled="exp.busy.value" @click="exp.doRun()">运行</button>

    <span v-if="exp.error.value" class="alert alert-error">{{ exp.error.value }}</span>
    <span v-if="exp.warn.value" class="alert alert-warn">{{ exp.warn.value }}</span>
  </div>
</template>

<script setup>
import { inject } from "vue";
const exp = inject("exp");
</script>

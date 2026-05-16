<template>
  <div class="app" :class="{ loading: busy }">
    <header class="app-header">
      <h1>面向 RAG 检索噪音的鲁棒性推理实验平台</h1>
      <p class="lead">
        RGB 数据集 · 噪音注入（比例 / 类型 / 位置）· Naive RAG 与多种矫正方法 · EM / F1 / ROUGE-L / ISR / NAR
      </p>
    </header>

    <div class="layout">
      <aside class="panel">
        <h2>实验参数</h2>

        <div class="field">
          <label>语言</label>
          <div class="radio-group">
            <label v-for="l in languages" :key="l">
              <input v-model="language" type="radio" :value="l" @change="onLangSubsetChange" />
              {{ l === "zh" ? "中文" : "English" }}
            </label>
          </div>
        </div>

        <div class="field">
          <label>子集</label>
          <select v-model="subset" @change="onLangSubsetChange">
            <option v-for="s in subsets" :key="s" :value="s">{{ s }}</option>
          </select>
        </div>

        <div class="field">
          <label>样本</label>
          <select v-model="sampleId" @change="loadSample">
            <option v-for="item in samples" :key="item.id" :value="item.id">
              {{ item.label }}
            </option>
          </select>
        </div>

        <div class="field">
          <label>噪音比例 {{ noiseRatio.toFixed(2) }}</label>
          <input v-model.number="noiseRatio" type="range" min="0" max="1" step="0.05" />
        </div>

        <div class="field">
          <label>噪音类型</label>
          <select v-model="noiseType">
            <option v-for="t in noiseTypes" :key="t" :value="t">{{ t }}</option>
          </select>
        </div>

        <div class="field">
          <label>噪音位置</label>
          <select v-model="noisePosition">
            <option v-for="p in noisePositions" :key="p" :value="p">{{ p }}</option>
          </select>
        </div>

        <div class="field">
          <label>矫正方法</label>
          <select v-model="method">
            <option v-for="m in methods" :key="m" :value="m">{{ m }}</option>
          </select>
        </div>

        <div class="btn-row">
          <button type="button" class="btn btn-secondary" :disabled="busy" @click="doInject">
            仅注入（预览 Stage B/C）
          </button>
          <button type="button" class="btn btn-primary" :disabled="busy" @click="doRun">
            运行完整流水线
          </button>
        </div>

        <p v-if="error" class="alert alert-error">{{ error }}</p>
        <p v-if="warn" class="alert alert-warn">{{ warn }}</p>
      </aside>

      <main class="main">
        <section>
          <h2>Stage A · 原始检索池</h2>
          <div class="field-row">
            <div class="field">
              <label>问题</label>
              <div class="readonly-box">{{ query || "—" }}</div>
            </div>
            <div class="field">
              <label>标准答案</label>
              <div class="readonly-box">{{ gold || "—" }}</div>
            </div>
          </div>
          <div v-if="retrievalHtml" v-html="retrievalHtml" class="doc-stage" />
          <p v-else class="stage-summary">选择样本后自动加载检索池</p>
        </section>

        <section>
          <h2>Stage B · 噪音注入后文档</h2>
          <div v-if="injectSummary" class="inject-md" v-html="renderMd(injectSummary)" />
          <div v-if="injectedHtml" v-html="injectedHtml" class="doc-stage" />
          <p v-else class="stage-summary">点击「仅注入」或「运行完整流水线」</p>
        </section>

        <section>
          <h2>Stage C · 送入 LLM 的 Prompt</h2>
          <div v-if="promptMarkdown" class="prompt-block">
            <pre>{{ promptMarkdown }}</pre>
          </div>
          <p v-else class="stage-summary">注入后显示 System / User 提示词</p>
        </section>

        <section>
          <h2>Stage D · 模型回答与评测</h2>
          <template v-if="runResult">
            <div class="answer-block">
              <strong>模型预测</strong>
              <blockquote>{{ runResult.prediction }}</blockquote>
            </div>
            <table class="metrics-table">
              <thead>
                <tr>
                  <th>指标</th>
                  <th>值</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>EM</td>
                  <td>{{ fmt(runResult.metrics.em) }}</td>
                </tr>
                <tr>
                  <td>Contains</td>
                  <td>{{ fmt(runResult.metrics.contains) }}</td>
                </tr>
                <tr>
                  <td>Token-F1</td>
                  <td>{{ fmt(runResult.metrics.token_f1) }}</td>
                </tr>
                <tr>
                  <td>ROUGE-L</td>
                  <td>{{ fmt(runResult.metrics.rouge_l) }}</td>
                </tr>
                <tr>
                  <td>ISR</td>
                  <td>{{ fmt(runResult.metrics.isr) }}</td>
                </tr>
                <tr>
                  <td>NAR</td>
                  <td>{{ fmt(runResult.metrics.nar) }}</td>
                </tr>
                <tr>
                  <td>判定</td>
                  <td>
                    <span class="verdict" :class="runResult.metrics.verdict">
                      {{ verdictLabel(runResult.metrics.verdict) }}
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
            <p class="stage-summary">
              method={{ runResult.meta?.method }} · prompt_tokens={{ runResult.meta?.prompt_tokens }} ·
              completion_tokens={{ runResult.meta?.completion_tokens }} · latency={{ runResult.meta?.latency }}s ·
              cached={{ runResult.meta?.cached }}
            </p>
          </template>
          <p v-else class="stage-summary">运行完整流水线后显示</p>
        </section>
      </main>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from "vue";
import {
  fetchConfig,
  fetchSamples,
  fetchSample,
  postInject,
  postRun,
} from "./api/client.js";

const languages = ref(["zh", "en"]);
const subsets = ref(["main", "refine", "fact", "int"]);
const noiseTypes = ref(["semantic", "counterfactual", "mixed"]);
const noisePositions = ref(["front", "back", "interleave", "surround"]);
const methods = ref(["naive"]);

const language = ref("zh");
const subset = ref("main");
const sampleId = ref(0);
const samples = ref([]);
const noiseRatio = ref(0.5);
const noiseType = ref("semantic");
const noisePosition = ref("interleave");
const method = ref("naive");

const query = ref("");
const gold = ref("");
const retrievalHtml = ref("");
const injectSummary = ref("");
const injectedHtml = ref("");
const promptMarkdown = ref("");
const runResult = ref(null);

const busy = ref(false);
const error = ref("");
const warn = ref("");

function payload() {
  return {
    language: language.value,
    subset: subset.value,
    sample_id: sampleId.value,
    noise_ratio: noiseRatio.value,
    noise_type: noiseType.value,
    noise_position: noisePosition.value,
  };
}

function fmt(n) {
  return typeof n === "number" ? n.toFixed(3) : "—";
}

function verdictLabel(v) {
  const map = {
    correct: "正确",
    partial: "部分正确",
    wrong: "错误",
    noise_biased: "噪音主导",
  };
  return map[v] || v;
}

function renderMd(md) {
  return md
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}

async function onLangSubsetChange() {
  error.value = "";
  try {
    const data = await fetchSamples(language.value, subset.value);
    samples.value = data.items || [];
    if (samples.value.length) {
      sampleId.value = samples.value[0].id;
      await loadSample();
    }
  } catch (e) {
    error.value = e.message || String(e);
  }
}

async function loadSample() {
  error.value = "";
  runResult.value = null;
  injectSummary.value = "";
  injectedHtml.value = "";
  promptMarkdown.value = "";
  try {
    const data = await fetchSample(sampleId.value, language.value, subset.value);
    query.value = data.query;
    gold.value = data.gold;
    retrievalHtml.value = data.retrieval_html;
  } catch (e) {
    error.value = e.message || String(e);
  }
}

async function doInject() {
  busy.value = true;
  error.value = "";
  warn.value = "正在注入（不调用 LLM）…";
  try {
    const data = await postInject(payload());
    injectSummary.value = data.summary;
    injectedHtml.value = data.injected_html;
    promptMarkdown.value = data.prompt_markdown;
    warn.value = "";
  } catch (e) {
    error.value = e.response?.data?.detail || e.message || String(e);
    warn.value = "";
  } finally {
    busy.value = false;
  }
}

async function doRun() {
  busy.value = true;
  error.value = "";
  warn.value = "正在调用 LLM，请稍候…";
  try {
    const data = await postRun({ ...payload(), method: method.value });
    injectSummary.value = data.inject_summary;
    injectedHtml.value = data.injected_html;
    promptMarkdown.value = data.prompt_markdown;
    runResult.value = data;
    warn.value = "";
  } catch (e) {
    const detail = e.response?.data?.detail;
    error.value = typeof detail === "string" ? detail : JSON.stringify(detail) || e.message;
    warn.value = "";
  } finally {
    busy.value = false;
  }
}

onMounted(async () => {
  try {
    const cfg = await fetchConfig();
    if (cfg.noise_types) noiseTypes.value = cfg.noise_types;
    if (cfg.noise_positions) noisePositions.value = cfg.noise_positions;
    if (cfg.methods) methods.value = cfg.methods;
    if (cfg.subsets) subsets.value = cfg.subsets;
    if (cfg.languages) languages.value = cfg.languages;
  } catch {
    /* use defaults */
  }
  await onLangSubsetChange();
});
</script>

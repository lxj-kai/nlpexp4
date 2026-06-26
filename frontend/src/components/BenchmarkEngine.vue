<template>
  <div class="bench-engine">
    <div class="bench-top">
      <div class="bench-title">
        <h2>数据构建引擎</h2>
        <span class="bench-desc">
          {{ currentDataset.label }} · {{ currentDataset.language.toUpperCase() }} · {{ dataset?.total ?? 0 }} 条样本
        </span>
      </div>
      <div class="bench-toolbar">
        <select v-model="datasetKey" class="dataset-select">
          <option v-for="item in datasetOptions" :key="item.key" :value="item.key">
            {{ item.label }}
          </option>
        </select>
        <select v-model="operationFilter" :disabled="!operations.length">
          <option value="all">全部类型</option>
          <option v-for="op in operations" :key="op" :value="op">{{ opLabel(op) }}</option>
        </select>
        <button class="btn" :disabled="loading" @click="loadDataset">
          {{ loading ? '加载中...' : '刷新数据' }}
        </button>
        <button class="btn ghost" :disabled="!filteredRecords.length" @click="pickRandom">随机样本</button>
      </div>
    </div>

    <div class="stats-row">
      <div class="stat-card dataset-stat">
        <label>数据集</label>
        <strong>{{ dataset?.dataset || currentDataset.key }}</strong>
      </div>
      <div class="stat-card">
        <label>语言</label>
        <strong>{{ currentDataset.language.toUpperCase() }}</strong>
      </div>
      <div class="stat-card">
        <label>样本数</label>
        <strong>{{ dataset?.total ?? 0 }}</strong>
      </div>
      <div class="stat-card">
        <label>Positive</label>
        <strong>{{ countText(dataset?.summary?.positive_docs) }}</strong>
      </div>
      <div class="stat-card">
        <label>Negative</label>
        <strong>{{ countText(dataset?.summary?.negative_docs) }}</strong>
      </div>
      <div class="stat-card">
        <label>Counterfactual</label>
        <strong>{{ countText(dataset?.summary?.positive_wrong_docs) }}</strong>
      </div>
      <div class="stat-card">
        <label>ID 范围</label>
        <strong>{{ listText(dataset?.summary?.id_range, ' - ') }}</strong>
      </div>
    </div>

    <div class="ops-row" v-if="dataset && operations.length">
      <button
        v-for="(count, op) in dataset.summary.operations"
        :key="op"
        class="op-chip"
        :class="{ active: operationFilter === op }"
        @click="operationFilter = operationFilter === op ? 'all' : op"
      >
        <span>{{ opLabel(op) }}</span>
        <b>{{ count }}</b>
      </button>
    </div>

    <div v-if="error" class="gen-error">{{ error }}</div>

    <div v-if="selected" class="workspace">
      <aside class="sample-list">
        <div class="list-head">
          <span>样本</span>
          <b>{{ filteredRecords.length }}</b>
        </div>
        <button
          v-for="record in filteredRecords"
          :key="record.id"
          class="sample-item"
          :class="{ active: selected.id === record.id }"
          @click="selectedId = record.id"
        >
          <span>#{{ record.id }}</span>
          <small>{{ sampleSubtitle(record) }}</small>
        </button>
      </aside>

      <main class="sample-detail">
        <section class="qa-panel">
          <div class="meta-line">
            <span v-for="(chip, i) in selectedChips" :key="`${chip}-${i}`">{{ chip }}</span>
          </div>
          <div class="question">{{ selected.query }}</div>
          <div class="answer">答案：{{ answerText(selected.answer) }}</div>
          <div v-if="selected.fakeanswer" class="fake-answer">干扰答案：{{ selected.fakeanswer }}</div>
        </section>

        <section v-if="supportPath.length" class="support-panel">
          <div class="panel-head">
            <h3>推理路径</h3>
            <span>{{ selected.mobilemem_meta?.required_positive_docs || selected.positive.length }} 个目标事件</span>
          </div>
          <div class="support-grid">
            <div v-for="(item, i) in supportPath" :key="i" class="support-item">
              <b>{{ item.event_id }}</b>
              <span>{{ item.event_name }}</span>
              <em>{{ item.metric_value }}</em>
            </div>
          </div>
        </section>

        <section v-else class="support-panel">
          <div class="panel-head">
            <h3>字段结构</h3>
            <span>{{ opLabel(recordOperation(selected)) }}</span>
          </div>
          <div class="field-grid">
            <div v-for="item in docStats" :key="item.label" class="field-item">
              <label>{{ item.label }}</label>
              <b>{{ item.value }}</b>
            </div>
          </div>
        </section>

        <section class="doc-grid" :class="{ three: selected.positive_wrong.length }">
          <div class="doc-col">
            <div class="doc-head pos-c">Positive ({{ selected.positive.length }})</div>
            <div class="doc-scroll">
              <div v-for="(doc, i) in selected.positive" :key="'p' + i" class="doc-block pos">
                <b>P{{ i + 1 }}</b>
                <pre>{{ doc }}</pre>
              </div>
            </div>
          </div>
          <div class="doc-col">
            <div class="doc-head neg-c">Negative ({{ selected.negative.length }})</div>
            <div class="doc-scroll">
              <div v-for="(doc, i) in selected.negative.slice(0, 20)" :key="'n' + i" class="doc-block neg">
                <b>N{{ i + 1 }}</b>
                <pre>{{ doc }}</pre>
              </div>
              <div v-if="selected.negative.length > 20" class="more-docs">还有 {{ selected.negative.length - 20 }} 条噪声证据</div>
            </div>
          </div>
          <div v-if="selected.positive_wrong.length" class="doc-col">
            <div class="doc-head cf-c">Counterfactual ({{ selected.positive_wrong.length }})</div>
            <div class="doc-scroll">
              <div v-for="(doc, i) in selected.positive_wrong" :key="'cf' + i" class="doc-block cf">
                <b>C{{ i + 1 }}</b>
                <pre>{{ doc }}</pre>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>

    <div v-else-if="!loading" class="empty-state">暂无可展示样本</div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { fetchBenchmarkDataset, fetchConfig } from "../api/client.js";

const NOISER_SUBSETS = new Set([
  "2wikimqa",
  "2wikiimqa",
  "bamboogle",
  "hotpotqa",
  "nq",
  "priorqa",
  "rgb_nb",
  "strategyqa",
  "tempqa",
]);

const FALLBACK_SUBSETS = [
  "main",
  "refine",
  "fact",
  "int",
  "mobilemem_shopping_graph_hard_120",
  "mobilemem_shopping_graph_noncalc_hard_120",
  "2wikimqa",
  "bamboogle",
  "priorqa",
  "rgb_nb",
  "strategyqa",
  "tempqa",
];

const DATASET_META = {
  main: { label: "RGB 基础集", language: "zh" },
  refine: { label: "RGB Refine", language: "zh" },
  fact: { label: "RGB 反事实", language: "zh" },
  int: { label: "RGB 信息整合", language: "zh" },
  mobilemem_shopping_graph_hard_120: { label: "MobileMem 购物Graph Hard·120", language: "zh" },
  mobilemem_shopping_graph_noncalc_hard_120: { label: "MobileMem 购物Graph 非计算·120", language: "zh" },
  "2wikimqa": { label: "2WikiMQA", language: "en" },
  "2wikiimqa": { label: "2WikiMQA", language: "en" },
  bamboogle: { label: "Bamboogle", language: "en" },
  hotpotqa: { label: "HotpotQA", language: "en" },
  nq: { label: "NQ", language: "en" },
  priorqa: { label: "PriorQA", language: "en" },
  rgb_nb: { label: "RGB-NoiserBench", language: "en" },
  strategyqa: { label: "StrategyQA", language: "en" },
  tempqa: { label: "TempQA", language: "en" },
};

const dataset = ref(null);
const datasetOptions = ref(FALLBACK_SUBSETS.map(toDatasetOption));
const loading = ref(false);
const loadingOptions = ref(false);
const error = ref("");
const selectedId = ref(null);
const operationFilter = ref("all");
const datasetKey = ref("main");

const records = computed(() => dataset.value?.records || []);
const operations = computed(() => Object.keys(dataset.value?.summary?.operations || {}));
const currentDataset = computed(() => (
  datasetOptions.value.find(item => item.key === datasetKey.value) || toDatasetOption(datasetKey.value)
));
const filteredRecords = computed(() => {
  if (operationFilter.value === "all") return records.value;
  return records.value.filter(record => recordOperation(record) === operationFilter.value);
});
const selected = computed(() => (
  filteredRecords.value.find(record => record.id === selectedId.value) || filteredRecords.value[0] || null
));
const supportPath = computed(() => selected.value?.mobilemem_meta?.support_path || []);
const selectedChips = computed(() => {
  if (!selected.value) return [];
  const record = selected.value;
  const chips = [`#${record.id}`, opLabel(recordOperation(record))];
  const scope = recordScope(record);
  const metric = record.mobilemem_meta?.metric_name || record.benchmark_meta?.metric_name;
  if (scope) chips.push(scope);
  if (metric) chips.push(metric);
  return chips;
});
const docStats = computed(() => {
  if (!selected.value) return [];
  return [
    { label: "答案数", value: selected.value.answer.length },
    { label: "Positive", value: selected.value.positive.length },
    { label: "Negative", value: selected.value.negative.length },
    { label: "Counterfactual", value: selected.value.positive_wrong.length },
  ];
});

watch(filteredRecords, list => {
  if (!list.some(record => record.id === selectedId.value)) {
    selectedId.value = list[0]?.id || null;
  }
});

watch(datasetKey, () => {
  operationFilter.value = "all";
  if (!loadingOptions.value) loadDataset();
});

function toDatasetOption(key) {
  const meta = DATASET_META[key] || {};
  return {
    key,
    label: meta.label || key,
    language: meta.language || (NOISER_SUBSETS.has(key) ? "en" : "zh"),
  };
}

function recordOperation(record) {
  return record?.benchmark_meta?.operation || record?.mobilemem_meta?.operation || "qa";
}

function recordScope(record) {
  if (!record) return "";
  return record.benchmark_meta?.scope_desc || record.mobilemem_meta?.scope_desc || docCountSummary(record);
}

function docCountSummary(record) {
  const parts = [
    `P${record.positive?.length || 0}`,
    `N${record.negative?.length || 0}`,
  ];
  if (record.positive_wrong?.length) parts.push(`CF${record.positive_wrong.length}`);
  return parts.join(" / ");
}

function sampleSubtitle(record) {
  return `${opLabel(recordOperation(record))} · ${recordScope(record)}`;
}

function opLabel(op) {
  return ({
    qa: "普通问答",
    refine: "精炼问答",
    counterfactual: "反事实",
    integration: "信息整合",
    noiserbench: "NoiserBench",
    sum: "求和",
    max_entity: "最大值",
    min_entity: "最小值",
    range: "范围",
    earliest_order_product: "最早下单商品",
    latest_order_shop: "最晚下单店铺",
    latest_event_product: "最晚发生商品",
    event_to_shop: "指定事件找店铺",
    shop_to_product: "指定店铺找商品",
    rating_event_name: "指定评分找事件",
  })[op] || op || "未知";
}

function listText(value, sep = " / ") {
  return Array.isArray(value) && value.length ? value.join(sep) : "-";
}

function countText(value) {
  if (!Array.isArray(value) || !value.length) return "-";
  if (value.length === 1) return String(value[0]);
  const min = value[0];
  const max = value[value.length - 1];
  return `${min}-${max} / ${value.length}种`;
}

function answerText(answer) {
  if (Array.isArray(answer)) return answer.length ? answer.join(" / ") : "-";
  if (answer === undefined || answer === null || answer === "") return "-";
  return String(answer);
}

function pickRandom() {
  const list = filteredRecords.value;
  if (!list.length) return;
  selectedId.value = list[Math.floor(Math.random() * list.length)]?.id || null;
}

async function loadDatasetOptions() {
  loadingOptions.value = true;
  try {
    const cfg = await fetchConfig();
    const subsets = Array.isArray(cfg.subsets) && cfg.subsets.length ? cfg.subsets : FALLBACK_SUBSETS;
    datasetOptions.value = subsets.map(toDatasetOption);
    if (!datasetOptions.value.some(item => item.key === datasetKey.value)) {
      datasetKey.value = datasetOptions.value[0]?.key || "main";
    }
  } finally {
    loadingOptions.value = false;
  }
}

async function loadDataset() {
  loading.value = true;
  error.value = "";
  try {
    const data = await fetchBenchmarkDataset(currentDataset.value.language, currentDataset.value.key);
    dataset.value = data;
    selectedId.value = data.records?.[0]?.id || null;
    operationFilter.value = "all";
  } catch (e) {
    error.value = e.response?.data?.detail || e.message || "数据集加载失败";
    dataset.value = null;
    selectedId.value = null;
  } finally {
    loading.value = false;
  }
}

onMounted(async () => {
  try {
    await loadDatasetOptions();
  } catch {
    datasetOptions.value = FALLBACK_SUBSETS.map(toDatasetOption);
  }
  await loadDataset();
});
</script>

<style scoped>
.bench-engine { height: 100%; overflow-y: auto; padding: 12px 16px; color: var(--ink); }
.bench-top { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 10px; }
.bench-title h2 { font-size: 16px; margin: 0; color: var(--ink); }
.bench-desc { font-size: 11px; color: var(--ink-muted); }
.bench-toolbar { display: flex; align-items: center; justify-content: flex-end; flex-wrap: wrap; gap: 6px; }
.bench-toolbar select { height: 28px; padding: 0 8px; border: 1px solid var(--border); border-radius: 4px; background: #fff; font-size: 12px; }
.bench-toolbar select:disabled { color: var(--ink-muted); background: #f8fafc; }
.dataset-select { min-width: 230px; max-width: 320px; }
.btn { height: 28px; padding: 0 12px; background: var(--blue); color: #fff; border: none; border-radius: 4px; font-size: 12px; font-weight: 600; cursor: pointer; }
.btn.ghost { background: #fff; color: var(--blue); border: 1px solid #bfdbfe; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }

.stats-row { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 8px; margin-bottom: 8px; }
.dataset-stat { grid-column: span 2; }
.stat-card { background: #fff; border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; min-width: 0; }
.stat-card label { display: block; font-size: 10px; color: var(--ink-muted); margin-bottom: 3px; }
.stat-card strong { display: block; font-size: 13px; color: var(--ink); overflow-wrap: anywhere; }

.ops-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.op-chip { display: flex; align-items: center; gap: 6px; height: 26px; padding: 0 10px; border: 1px solid var(--border); border-radius: 999px; background: #fff; color: var(--ink); font-size: 12px; cursor: pointer; }
.op-chip b { color: var(--blue); }
.op-chip.active { border-color: var(--blue); background: var(--blue-bg); }

.workspace { display: grid; grid-template-columns: 240px minmax(0, 1fr); gap: 10px; min-height: 0; }
.sample-list { background: #fff; border: 1px solid var(--border); border-radius: 6px; padding: 8px; max-height: calc(100vh - 230px); overflow-y: auto; }
.list-head { display: flex; justify-content: space-between; align-items: center; font-size: 12px; font-weight: 700; margin-bottom: 6px; }
.sample-item { width: 100%; text-align: left; padding: 7px 8px; margin-bottom: 4px; border: 1px solid transparent; border-radius: 5px; background: #f8fafc; cursor: pointer; }
.sample-item span { display: block; font-size: 12px; font-weight: 700; color: var(--ink); }
.sample-item small { display: block; font-size: 10px; color: var(--ink-muted); margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sample-item.active { border-color: var(--blue); background: var(--blue-bg); }

.sample-detail { min-width: 0; }
.qa-panel, .support-panel { background: #fff; border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; margin-bottom: 8px; }
.meta-line { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.meta-line span { font-size: 10px; padding: 2px 6px; border-radius: 999px; background: #f1f5f9; color: var(--ink-muted); }
.question { font-size: 13px; line-height: 1.65; color: var(--ink); }
.answer { margin-top: 8px; padding: 6px 8px; border-left: 3px solid #15803d; background: #f0fdf4; color: #15803d; font-size: 13px; font-weight: 700; }
.fake-answer { margin-top: 6px; padding: 6px 8px; border-left: 3px solid #b45309; background: #fff7ed; color: #92400e; font-size: 12px; font-weight: 700; }

.panel-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 8px; }
.panel-head h3 { margin: 0; font-size: 13px; }
.panel-head span { font-size: 11px; color: var(--ink-muted); text-align: right; }
.support-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; }
.support-item { background: #f8fafc; border: 1px solid var(--border); border-radius: 5px; padding: 7px; min-width: 0; }
.support-item b { display: block; font-size: 11px; color: var(--blue); }
.support-item span { display: block; font-size: 11px; line-height: 1.35; margin: 4px 0; }
.support-item em { font-style: normal; color: #15803d; font-size: 12px; font-weight: 700; }
.field-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; }
.field-item { background: #f8fafc; border: 1px solid var(--border); border-radius: 5px; padding: 7px; }
.field-item label { display: block; font-size: 10px; color: var(--ink-muted); margin-bottom: 3px; }
.field-item b { font-size: 13px; color: var(--ink); }

.doc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.doc-grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.doc-col { min-width: 0; background: #fff; border: 1px solid var(--border); border-radius: 6px; padding: 8px; }
.doc-head { font-size: 11px; font-weight: 700; margin-bottom: 6px; }
.pos-c { color: #15803d; }
.neg-c { color: #64748b; }
.cf-c { color: #b45309; }
.doc-scroll { max-height: 420px; overflow-y: auto; }
.doc-block { position: relative; padding: 7px 8px 7px 30px; margin-bottom: 5px; border-radius: 5px; }
.doc-block b { position: absolute; left: 7px; top: 8px; font-size: 9px; }
.doc-block pre { margin: 0; white-space: pre-wrap; font-family: inherit; font-size: 10px; line-height: 1.45; color: var(--ink); }
.doc-block.pos { background: #f0fdf4; border-left: 2px solid #15803d; }
.doc-block.neg { background: #f8fafc; border-left: 2px solid #94a3b8; }
.doc-block.cf { background: #fff7ed; border-left: 2px solid #b45309; }
.more-docs { text-align: center; padding: 8px; font-size: 11px; color: var(--ink-muted); }
.gen-error { padding: 6px 10px; background: #fef2f2; color: #991b1b; border-radius: 4px; font-size: 12px; margin-bottom: 8px; }
.empty-state { padding: 24px; text-align: center; color: var(--ink-muted); font-size: 12px; }

@media (max-width: 1100px) {
  .stats-row { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .dataset-stat { grid-column: span 1; }
  .doc-grid.three { grid-template-columns: 1fr; }
}

@media (max-width: 900px) {
  .bench-top, .bench-toolbar { align-items: stretch; flex-direction: column; }
  .dataset-select { min-width: 0; max-width: none; }
  .stats-row, .workspace, .doc-grid, .support-grid, .field-grid { grid-template-columns: 1fr; }
  .sample-list { max-height: 240px; }
}
</style>

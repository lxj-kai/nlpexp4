import { computed, onMounted, ref, watch } from "vue";
import {
  fetchConfig,
  fetchSample,
  fetchSamples,
  postInject,
  postRun,
} from "../api/client.js";

/** 与 backend dataset_registry 同步；/config 失败时作 fallback */
const FALLBACK_DATASETS = [
  {
    id: "rgb",
    label: "RGB（基准 · 新闻/百科）",
    languages: ["zh", "en"],
    subsets: ["main", "refine", "fact", "int"],
    default_language: "zh",
    default_subset: "main",
  },
  {
    id: "2wiki",
    label: "2WikiMultihopQA（英文多跳 · hard neg）",
    languages: ["en"],
    subsets: ["main", "fact"],
    default_language: "en",
    default_subset: "main",
  },
  {
    id: "cmedqa",
    label: "CmedqaRetrieval（中文医学）",
    languages: ["zh"],
    subsets: ["main", "fact"],
    default_language: "zh",
    default_subset: "main",
  },
  {
    id: "miriad",
    label: "MIRIAD-5.8M（英文医学 · 大规模）",
    languages: ["en"],
    subsets: ["main", "fact"],
    default_language: "en",
    default_subset: "main",
  },
  {
    id: "bright",
    label: "BRIGHT（英文 · hard neg · 长文推理）",
    languages: ["en"],
    subsets: ["main", "fact"],
    default_language: "en",
    default_subset: "main",
  },
  {
    id: "multihop_rag",
    label: "MultiHop-RAG（英文 · evidence 明确 · 新闻整合）",
    languages: ["en"],
    subsets: ["main", "fact"],
    default_language: "en",
    default_subset: "main",
  },
  {
    id: "tempo",
    label: "TEMPO（英文 · 论坛长文 · 多域）",
    languages: ["en"],
    subsets: ["main", "fact"],
    default_language: "en",
    default_subset: "main",
  },
  {
    id: "noiser_bench",
    label: "NoiserBench（ACL'25 · 7类噪音 RAG）",
    languages: ["en"],
    subsets: ["hotpotqa", "rgb_nb", "bamboogle", "strategyqa", "tempqa", "priorqa", "nq", "2wikimqa"],
    default_language: "en",
    default_subset: "hotpotqa",
  },
];

export function useExperiment() {
  const datasets = ref([...FALLBACK_DATASETS]);
  const languages = ref(["zh", "en"]);
  const subsets = ref(["main", "refine", "fact", "int"]);
  const noiseTypesAll = ref(["semantic", "counterfactual", "mixed"]);
  const noisePositions = ref(["front", "back", "interleave", "surround"]);
  const methods = ref(["naive"]);

  const generationModel = ref("");
  const judgeModel = ref("");
  const judgeApiBase = ref("");

  const dataset = ref("rgb");
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

  const datasetLabel = computed(() => {
    const spec = datasets.value.find((d) => d.id === dataset.value);
    return spec?.label || dataset.value;
  });

  /** fact 子集才有 positive_wrong，才支持 counterfactual / mixed */
  const noiseTypes = computed(() => {
    if (subset.value === "fact") {
      return noiseTypesAll.value;
    }
    return noiseTypesAll.value.filter((t) => t === "semantic");
  });

  function currentSpec() {
    return datasets.value.find((d) => d.id === dataset.value) || null;
  }

  function applyDatasetSpec(spec) {
    if (!spec) return;
    languages.value = spec.languages || languages.value;
    subsets.value = spec.subsets || subsets.value;
    if (!languages.value.includes(language.value)) {
      language.value = spec.default_language || languages.value[0];
    }
    if (!subsets.value.includes(subset.value)) {
      subset.value = spec.default_subset || subsets.value[0];
    }
  }

  watch(noiseTypes, (types) => {
    if (!types.includes(noiseType.value)) {
      noiseType.value = "semantic";
    }
  });

  function payload() {
    return {
      dataset: dataset.value,
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

  function fmtPct(n) {
    return typeof n === "number" ? `${(n * 100).toFixed(1)}%` : "—";
  }

  function verdictLabel(v) {
    return (
      { correct: "正确", partial: "部分正确", wrong: "错误", noise_biased: "噪音主导" }[v] || v
    );
  }

  function renderMd(md) {
    return md
      .replace(/^### (.+)$/gm, "<h3>$1</h3>")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br>");
  }

  async function reloadSamples() {
    error.value = "";
    try {
      const data = await fetchSamples(dataset.value, language.value, subset.value);
      samples.value = data.items || [];
      if (samples.value.length) {
        sampleId.value = samples.value[0].id;
        await loadSample();
      } else {
        query.value = "";
        gold.value = "";
        retrievalHtml.value = "";
        error.value = `数据集 ${datasetLabel.value} 无样本，请先运行 prepare 脚本`;
      }
    } catch (e) {
      error.value = e.response?.data?.detail || e.message || String(e);
    }
  }

  async function onDatasetChange() {
    applyDatasetSpec(currentSpec());
    runResult.value = null;
    await reloadSamples();
  }

  async function onLangSubsetChange() {
    runResult.value = null;
    await reloadSamples();
  }

  async function loadSample() {
    error.value = "";
    runResult.value = null;
    injectSummary.value = "";
    injectedHtml.value = "";
    promptMarkdown.value = "";
    try {
      const data = await fetchSample(
        sampleId.value,
        dataset.value,
        language.value,
        subset.value,
      );
      query.value = data.query;
      gold.value = data.gold;
      retrievalHtml.value = data.retrieval_html;
    } catch (e) {
      error.value = e.response?.data?.detail || e.message || String(e);
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
    warn.value = "正在调用 LM Studio 问答 + DeepSeek 审查，请稍候…";
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
      if (cfg.datasets?.length) datasets.value = cfg.datasets;
      if (cfg.default_dataset) dataset.value = cfg.default_dataset;
      if (cfg.noise_types) noiseTypesAll.value = cfg.noise_types;
      if (cfg.noise_positions) noisePositions.value = cfg.noise_positions;
      if (cfg.methods) methods.value = cfg.methods;
      if (cfg.generation_model) generationModel.value = cfg.generation_model;
      if (cfg.judge_model) judgeModel.value = cfg.judge_model;
      if (cfg.judge_api_base) judgeApiBase.value = cfg.judge_api_base;
      applyDatasetSpec(currentSpec());
    } catch (e) {
      warn.value = `配置加载失败，使用本地数据集列表（${e.message || e}）`;
      applyDatasetSpec(currentSpec());
    }
    await reloadSamples();
  });

  return {
    datasets,
    languages,
    subsets,
    noiseTypes,
    noisePositions,
    methods,
    generationModel,
    judgeModel,
    judgeApiBase,
    datasetLabel,
    dataset,
    language,
    subset,
    sampleId,
    samples,
    noiseRatio,
    noiseType,
    noisePosition,
    method,
    query,
    gold,
    retrievalHtml,
    injectSummary,
    injectedHtml,
    promptMarkdown,
    runResult,
    busy,
    error,
    warn,
    fmt,
    fmtPct,
    verdictLabel,
    renderMd,
    onDatasetChange,
    onLangSubsetChange,
    loadSample,
    doInject,
    doRun,
  };
}

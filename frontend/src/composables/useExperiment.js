import { computed, onMounted, ref } from "vue";
import {
  fetchConfig,
  fetchSample,
  fetchSamples,
  postInject,
  postRun,
} from "../api/client.js";

const METHOD_FLOW = {
  naive: {
    label: "Naive RAG",
    badge: "baseline",
    cost: "1 call",
    summary: "直接拼接检索上下文并生成答案。",
    steps: ["固定检索上下文", "拼接全部文档", "单次生成答案"],
  },
  prompt: {
    label: "Prompt-aware",
    badge: "ours",
    cost: "1 call",
    summary: "只修改 system prompt，引导模型质疑噪音文档。",
    steps: ["固定检索上下文", "加入噪音警惕指令", "单次生成答案"],
  },
  iterative: {
    label: "CRAG-style Filter",
    badge: "paper-style",
    cost: "n docs + 1 call",
    summary: "生成前先给检索文档打 high/mid/low 分，只把通过的文档送去回答。",
    steps: ["检索文档打分", "丢弃 low 文档", "用保留文档生成"],
  },
  selfrag: {
    label: "Self-RAG-style Gate",
    badge: "paper-style",
    cost: "n docs + 2~3 calls",
    summary: "先做相关性门控，生成后再检查答案是否被证据支撑，失败才重生成。",
    steps: ["相关性门控", "生成候选答案", "答案支撑验证", "失败则重生成"],
  },
  confidence: {
    label: "CoT-Evidence",
    badge: "ours",
    cost: "1 call",
    summary: "显式拆解信息需求并做证据匹配，再输出最终答案。",
    steps: ["拆解信息需求", "逐篇匹配证据", "综合支持信息", "标签抽取答案"],
  },
  voting: {
    label: "Evidence Voting",
    badge: "ours",
    cost: "3~4 calls",
    summary: "多个证据视角独立作答，再多数投票或聚合。",
    steps: ["三种 persona 独立生成", "检测多数一致", "必要时 LLM 聚合", "输出最终答案"],
  },
  adaptive: {
    label: "Adaptive Router",
    badge: "ours",
    cost: "2~5 calls",
    summary: "先检测上下文风险，再路由到不同矫正方法。",
    steps: ["检测噪音类型", "选择矫正路径", "执行目标方法", "记录路由原因"],
  },
  iterative_sc: {
    label: "Iterative Self-Check",
    badge: "ours",
    cost: "3~7 calls",
    summary: "生成后自检一致性，失败则修订，最多迭代三轮。",
    steps: ["初次生成", "答案一致性自检", "定位问题", "重读文档修订"],
  },
};

const METHOD_ORDER = [
  "naive",
  "prompt",
  "iterative",
  "selfrag",
  "confidence",
  "voting",
  "adaptive",
  "iterative_sc",
];

const SUBSET_LABELS = {
  "2wikimqa": "2WikiMQA",
  "2wikiimqa": "2WikiMQA",
  "hotpotqa": "HotpotQA",
  "bamboogle": "Bamboogle",
  "nq": "NQ",
  "priorqa": "PriorQA",
  "rgb_nb": "RGB-NoiserBench",
  "strategyqa": "StrategyQA",
  "tempqa": "TempQA",
  "mobilemem_shopping_graph_hard_120": "MobileMem 购物Graph Hard·120",
  "mobilemem_shopping_graph_noncalc_hard_120": "MobileMem 购物Graph 非计算·120",
};

const NOISE_POSITION_LABELS = {
  front: "前置插入",
  back: "后置插入",
  interleave: "交错插入",
  surround: "前后包围",
};

function sortMethods(values) {
  const rank = new Map(METHOD_ORDER.map((name, idx) => [name, idx]));
  return [...values].sort((a, b) => {
    const ra = rank.has(a) ? rank.get(a) : 999;
    const rb = rank.has(b) ? rank.get(b) : 999;
    return ra === rb ? a.localeCompare(b) : ra - rb;
  });
}

function methodInfo(name) {
  return {
    id: name,
    ...(METHOD_FLOW[name] || {
      label: name,
      badge: "custom",
      cost: "unknown",
      summary: "当前方法未登记流程元信息。",
      steps: ["接收固定上下文", "执行后端方法", "返回答案"],
    }),
  };
}

export function useExperiment() {
  const mobileMemSubsets = new Set([
    "mobilemem_shopping_graph_hard_120",
    "mobilemem_shopping_graph_noncalc_hard_120",
  ]);
  const noiserSubsets = new Set([
    "2wikimqa",
    "2wikiimqa",
    "hotpotqa",
    "bamboogle",
    "nq",
    "priorqa",
    "rgb_nb",
    "strategyqa",
    "tempqa",
  ]);
  const languages = ref(["zh", "en"]);
  const subsets = ref(["main", "refine", "fact", "int", ...mobileMemSubsets]);
  const noiseTypes = ref(["semantic", "counterfactual", "mixed"]);
  const noisePositions = ref(["front", "back", "interleave", "surround"]);
  const methods = ref([...METHOD_ORDER]);

  const language = ref("zh");
  const subset = ref("main");
  const sampleId = ref(0);
  const samples = ref([]);
  const noiseRatio = ref(0.70);
  const noiseType = ref("semantic");
  const noisePosition = ref("interleave");
  const method = ref("naive");
  const methodFlow = computed(() => (
    methodInfo(method.value)
  ));
  const methodFlowList = computed(() => sortMethods(methods.value).map(methodInfo));

  const query = ref("");
  const gold = ref("");
  const retrievalHtml = ref("");
  const injectSummary = ref("");
  const injectedHtml = ref("");
  const promptMarkdown = ref("");
  const promptCalls = ref([]);
  const runResult = ref(null);

  const busy = ref(false);
  const error = ref("");
  const warn = ref("");

  function payload() {
    return {
      language: language.value,
      subset: subset.value,
      sample_id: sampleId.value,
      method: method.value,
      noise_ratio: noiseRatio.value,
      noise_type: noiseType.value,
      noise_position: noisePosition.value,
    };
  }

  function fmt(n) {
    return typeof n === "number" ? n.toFixed(3) : "—";
  }

  function verdictLabel(v) {
    return (
      { correct: "正确", partial: "部分正确", wrong: "错误", noise_biased: "噪音主导" }[v] || v
    );
  }

  function methodLabel(name) {
    return METHOD_FLOW[name]?.label || name;
  }

  function subsetLabel(name) {
    return SUBSET_LABELS[name] || name;
  }

  function noisePositionLabel(name) {
    return NOISE_POSITION_LABELS[name] || name;
  }

  function renderMd(md) {
    return md
      .replace(/^### (.+)$/gm, "<h3>$1</h3>")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br>");
  }

  async function onLangSubsetChange() {
    error.value = "";
    if (mobileMemSubsets.has(subset.value)) {
      noiseType.value = "semantic";
    }
    if (noiserSubsets.has(subset.value)) {
      language.value = "en";
    }
    if (
      subset.value === "mobilemem_shopping_graph_hard_120" ||
      subset.value === "mobilemem_shopping_graph_noncalc_hard_120"
    ) {
      noiseRatio.value = 0.70;
      noisePosition.value = "interleave";
    }
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
    promptCalls.value = [];
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
    if (mobileMemSubsets.has(subset.value)) {
      noiseType.value = "semantic";
    }
    busy.value = true;
    error.value = "";
    warn.value = "正在注入（不调用 LLM）…";
    try {
      const data = await postInject(payload());
      injectSummary.value = data.summary;
      injectedHtml.value = data.injected_html;
      promptMarkdown.value = data.prompt_markdown;
      promptCalls.value = data.prompt_calls || [];
      warn.value = "";
    } catch (e) {
      error.value = e.response?.data?.detail || e.message || String(e);
      warn.value = "";
    } finally {
      busy.value = false;
    }
  }

  async function changeMethod(nextMethod) {
    if (method.value === nextMethod) return;
    const hadPreview = Boolean(promptMarkdown.value || injectedHtml.value || injectSummary.value);
    method.value = nextMethod;
      runResult.value = null;

    if (hadPreview) {
      await doInject();
      return;
    }

    promptMarkdown.value = "";
    promptCalls.value = [];
  }

  async function doRun() {
    if (mobileMemSubsets.has(subset.value)) {
      noiseType.value = "semantic";
    }
    busy.value = true;
    error.value = "";
    warn.value = "正在调用 LLM，请稍候…";
    try {
      const data = await postRun({ ...payload(), method: method.value });
      injectSummary.value = data.inject_summary;
      injectedHtml.value = data.injected_html;
      promptMarkdown.value = data.prompt_markdown;
      promptCalls.value = data.prompt_calls || [];
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
      if (cfg.methods) methods.value = sortMethods(cfg.methods);
      if (cfg.subsets) subsets.value = cfg.subsets;
      if (cfg.languages) languages.value = cfg.languages;
    } catch {
      /* use defaults */
    }
    await onLangSubsetChange();
  });

  return {
    languages, subsets, noiseTypes, noisePositions, methods,
    language, subset, sampleId, samples,
    noiseRatio, noiseType, noisePosition, method,
    methodFlow, methodFlowList,
    query, gold, retrievalHtml,
    injectSummary, injectedHtml, promptMarkdown, promptCalls, runResult,
    busy, error, warn,
    fmt, verdictLabel, renderMd, methodLabel, subsetLabel, noisePositionLabel,
    onLangSubsetChange, loadSample, doInject, doRun, changeMethod,
  };
}

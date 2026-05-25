import { onMounted, ref } from "vue";
import {
  fetchConfig,
  fetchSample,
  fetchSamples,
  postInject,
  postRun,
} from "../api/client.js";

export function useExperiment() {
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

  return {
    languages, subsets, noiseTypes, noisePositions, methods,
    language, subset, sampleId, samples,
    noiseRatio, noiseType, noisePosition, method,
    query, gold, retrievalHtml,
    injectSummary, injectedHtml, promptMarkdown, runResult,
    busy, error, warn,
    fmt, verdictLabel, renderMd,
    onLangSubsetChange, loadSample, doInject, doRun,
  };
}

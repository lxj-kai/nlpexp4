<template>
  <div class="bench-engine">
    <!-- Top bar: title + input + button -->
    <div class="bench-top">
      <div class="bench-title">
        <h2>数据构建引擎</h2>
        <span class="bench-desc">爬取真实文档 + LLM 生成 → RAG 鲁棒性测试数据</span>
      </div>
      <div class="gen-bar">
        <input v-model="keyword" placeholder="输入关键词，如：量子力学" @keyup.enter="generate" />
        <button class="btn-gen" :disabled="generating" @click="generate">
          {{ generating ? '生成中...' : '一键生成' }}
        </button>
      </div>
    </div>

    <!-- Pipeline steps (compact) -->
    <div class="bench-pipeline">
      <div v-for="s in pipelineSteps" :key="s.num" class="pipe-step-wrap">
        <div class="pipe-step" :class="getStepClass(s.num)">
          <div class="step-badge">{{ s.num }}</div>
          <div>
            <div class="step-title">{{ s.title }}</div>
            <div class="step-desc">{{ s.desc }}</div>
          </div>
        </div>
        <div class="pipe-arrow" v-if="s.num < 4">→</div>
      </div>
    </div>

    <!-- Step-by-step output (compact) -->
    <div class="gen-output" v-if="steps.length">
      <div v-for="s in steps" :key="s.step" class="step-result" :class="s.status">
        <div class="step-head">
          <span class="step-icon">
            <template v-if="s.status === 'done'">&#10003;</template>
            <template v-else-if="s.status === 'running'">&#9679;</template>
            <template v-else>&#10007;</template>
          </span>
          <strong>{{ s.title }}</strong>
          <span class="step-summary" v-if="s.data">
            <template v-if="s.step === 1"> — 找到 {{ s.data.found }} 篇文章</template>
            <template v-else-if="s.step === 2"> — 获取 {{ s.data.fetched }} 篇文档</template>
            <template v-else-if="s.step === 4"> — {{ s.data.n_positive }}P · {{ s.data.n_negative }}N · {{ s.data.n_wrong }}CF</template>
          </span>
        </div>
        <div class="step-data" v-if="s.data">
          <template v-if="s.step === 1">
            <span v-for="t in s.data.titles?.slice(0, 8)" :key="t" class="wiki-tag">{{ t }}</span>
            <span v-if="s.data.titles?.length > 8" class="wiki-tag more">+{{ s.data.titles.length - 8 }}</span>
          </template>
          <template v-else-if="s.step === 2 && s.data?.docs">
            <div class="doc-preview-scroll">
              <div v-for="d in s.data.docs" :key="d.title" class="doc-preview-item">
                <b>{{ d.title }}</b>: {{ d.text }}
              </div>
            </div>
          </template>
          <template v-else-if="s.step === 3">
            <div class="gen-qa-inline">
              <span><b>Q:</b> {{ s.data.question }}</span>
              <span><b>A:</b> {{ s.data.answer }}</span>
            </div>
          </template>
        </div>
      </div>
    </div>

    <!-- Generated Entry (compact grid) -->
    <div class="gen-entry" v-if="entry">
      <div class="entry-qa">
        <div class="qa-item"><label>问题</label><div class="qa-val">{{ entry.query }}</div></div>
        <div class="qa-item"><label class="ok">正确答案</label><div class="qa-val ok">{{ entry.answer }}</div></div>
      </div>

      <div class="doc-grid">
        <div class="doc-col">
          <div class="doc-head pos-c">Positive ({{ entry.positive.length }})</div>
          <div class="doc-scroll">
            <div v-for="(p, i) in entry.positive" :key="'p'+i" class="doc-block pos">
              <b>P{{ i }}</b> {{ p.slice(0, 150) }}…
            </div>
          </div>
        </div>
        <div class="doc-col">
          <div class="doc-head neg-c">Negative ({{ entry.negative.length }})</div>
          <div class="doc-scroll">
            <div v-for="(n, i) in entry.negative" :key="'n'+i" class="doc-block neg">
              <b>N{{ i }}</b> {{ n.slice(0, 120) }}…
            </div>
          </div>
        </div>
        <div class="doc-col" v-if="entry.positive_wrong.length">
          <div class="doc-head cf-c">反事实 ({{ entry.positive_wrong.length }})</div>
          <div class="doc-scroll">
            <div v-for="(w, i) in entry.positive_wrong" :key="'w'+i" class="doc-block cf">
              <b>CF{{ i }}</b> {{ w.slice(0, 150) }}…
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-if="error" class="gen-error">{{ error }}</div>
  </div>
</template>

<script setup>
import { ref } from "vue";

const pipelineSteps = [
  { num: 1, title: "搜索维基", desc: "关键词 → Wikipedia API" },
  { num: 2, title: "获取文档", desc: "提取正文内容" },
  { num: 3, title: "LLM 生成", desc: "Q/A + 反事实" },
  { num: 4, title: "组装输出", desc: "RGB 兼容格式" },
];

const keyword = ref("量子力学");
const generating = ref(false);
const steps = ref([]);
const entry = ref(null);
const error = ref("");
const currentStep = ref(0);

function getStepClass(num) {
  if (!generating.value && steps.value.length === 0) return "";
  const s = steps.value.find(x => x.step === num);
  if (!s) return num <= currentStep.value ? "" : "pending";
  return s.status;
}

async function generate() {
  generating.value = true;
  steps.value = [];
  entry.value = null;
  error.value = "";
  currentStep.value = 0;

  try {
    const resp = await fetch("/api/benchmark/generate_one", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keyword: keyword.value }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const evt = JSON.parse(line.slice(6));

          if (evt.step === 0 && evt.status === "failed") {
            error.value = evt.data?.error || "生成失败";
            continue;
          }

          if (evt.step === 5 && evt.data?.entry) {
            entry.value = evt.data.entry;
            continue;
          }

          currentStep.value = evt.step;
          const existing = steps.value.findIndex(s => s.step === evt.step);
          if (existing >= 0) {
            steps.value[existing] = evt;
          } else {
            steps.value.push(evt);
          }
        } catch {}
      }
    }
  } catch (e) {
    error.value = e.message;
  } finally {
    generating.value = false;
  }
}
</script>

<style scoped>
.bench-engine { height: 100%; overflow-y: auto; padding: 12px 16px; }

.bench-top { display: flex; align-items: center; gap: 16px; margin-bottom: 8px; }
.bench-title { flex-shrink: 0; }
.bench-title h2 { font-size: 16px; margin: 0; color: var(--ink); }
.bench-desc { font-size: 11px; color: var(--ink-muted); }
.gen-bar { display: flex; align-items: center; gap: 6px; flex: 1; }
.gen-bar input { flex: 1; padding: 5px 8px; border: 1px solid var(--border); border-radius: 4px; font-size: 12px; }
.btn-gen { padding: 5px 14px; background: var(--blue); color: #fff; border: none; border-radius: 4px; font-size: 12px; font-weight: 600; cursor: pointer; white-space: nowrap; }
.btn-gen:disabled { opacity: 0.5; cursor: not-allowed; }

.bench-pipeline {
  display: flex; align-items: center; gap: 3px;
  padding: 6px 10px; background: var(--blue-bg);
  border: 1px solid #c7d2fe; border-radius: 6px; margin-bottom: 8px;
}
.pipe-step-wrap { display: flex; align-items: center; gap: 3px; }
.pipe-step { display: flex; align-items: center; gap: 6px; padding: 3px 6px; border-radius: 5px; transition: all 0.3s; }
.pipe-step.done { background: #dcfce7; }
.pipe-step.running { background: #fef3c7; animation: pulse 1s infinite; }
.pipe-step.failed { background: #fee2e2; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }
.step-badge { width: 18px; height: 18px; border-radius: 50%; background: var(--blue); color: #fff; font-size: 10px; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.step-title { font-size: 11px; font-weight: 700; color: var(--ink); }
.step-desc { font-size: 9px; color: var(--ink-muted); }
.pipe-arrow { color: var(--blue); font-size: 14px; font-weight: 700; }

/* Steps output */
.gen-output { margin-bottom: 8px; }
.step-result { padding: 5px 10px; margin: 3px 0; border-radius: 5px; border: 1px solid var(--border); background: #fff; }
.step-result.done { border-left: 3px solid #15803d; }
.step-result.running { border-left: 3px solid #d97706; background: #fffbeb; }
.step-result.failed { border-left: 3px solid #b91c1c; background: #fef2f2; }
.step-head { display: flex; align-items: center; gap: 5px; font-size: 12px; }
.step-icon { font-size: 13px; }
.step-result.done .step-icon { color: #15803d; }
.step-result.running .step-icon { color: #d97706; }
.step-result.failed .step-icon { color: #b91c1c; }
.step-summary { font-weight: 400; color: var(--ink-muted); font-size: 11px; }
.step-data { font-size: 11px; margin-top: 3px; color: var(--ink-muted); }
.wiki-tag { background: #e2e8f0; padding: 1px 5px; border-radius: 3px; margin: 0 1px; font-size: 10px; display: inline-block; }
.wiki-tag.more { background: #cbd5e1; color: #475569; }
.gen-qa-inline { display: flex; flex-direction: column; gap: 1px; font-size: 12px; }
.doc-preview-scroll { max-height: 100px; overflow-y: auto; }
.doc-preview-item { padding: 2px 6px; margin: 1px 0; background: #f8fafc; border-radius: 3px; font-size: 10px; line-height: 1.4; border-left: 2px solid #94a3b8; }

/* Result entry */
.gen-entry { background: #fff; border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; }
.entry-qa { margin-bottom: 8px; }
.qa-item { margin-bottom: 4px; }
.qa-item label { font-size: 10px; font-weight: 600; color: var(--accent); display: block; }
.qa-val { padding: 3px 6px; border: 1px solid var(--border); border-radius: 3px; font-size: 12px; }
.qa-val.ok { border-left: 3px solid #15803d; color: #15803d; }
.qa-val.bad { border-left: 3px solid #b91c1c; color: #b91c1c; }
.ok { color: #15803d; }
.bad { color: #b91c1c; }
.qa-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }

/* Doc grid */
.doc-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.doc-head { font-size: 11px; font-weight: 700; margin-bottom: 3px; }
.pos-c { color: #15803d; }
.neg-c { color: #64748b; }
.cf-c { color: #b91c1c; }
.doc-block { font-size: 10px; line-height: 1.4; padding: 4px 6px; margin: 2px 0; border-radius: 3px; }
.doc-block.pos { background: #f0fdf4; border-left: 2px solid #15803d; }
.doc-block.neg { background: #f8fafc; border-left: 2px solid #94a3b8; }
.doc-block.cf { background: #fef2f2; border-left: 2px solid #b91c1c; }
.doc-block b { font-size: 9px; margin-right: 2px; }
.doc-scroll { max-height: 180px; overflow-y: auto; }

.gen-error { padding: 6px 10px; background: #fef2f2; color: #991b1b; border-radius: 4px; font-size: 12px; margin-top: 6px; }
</style>

<template>
  <div class="method-preview">
    <div class="preview-top">
      <div class="preview-title">
        <h2>当前方法案例预览</h2>
        <span>RGB 基础集 zh/main #0 · {{ demoCase.query }}</span>
      </div>
      <div class="preview-method-select">
        <label>当前方法</label>
        <select :value="activeMethod" @change="selectMethod($event.target.value)">
          <option v-for="flow in flows" :key="flow.id" :value="flow.id">
            {{ flow.label }}
          </option>
        </select>
      </div>
    </div>

    <section class="case-panel">
      <div class="case-question">
        <label>问题</label>
        <strong>{{ demoCase.query }}</strong>
      </div>
      <div class="case-answer">
        <label>标准答案</label>
        <strong>{{ demoCase.answer }}</strong>
      </div>
      <div class="case-note">
        <label>检索上下文</label>
        <span>示例中混入直接证据、相似事故、同地点不同事故和同类灾害噪音。</span>
      </div>
    </section>

    <section class="docs-panel">
      <article v-for="doc in demoDocs" :key="doc.id" class="demo-doc" :class="doc.kind">
        <div class="doc-id">{{ doc.id }}</div>
        <div>
          <strong>{{ doc.label }}</strong>
          <p>{{ doc.text }}</p>
        </div>
      </article>
    </section>

    <section class="demo-panel">
      <div class="demo-head">
        <div>
          <h3>{{ activeFlow?.label || activeMethod }}</h3>
          <span>只演示当前选中方法在该样例上的实验处理流程。</span>
        </div>
        <div class="method-tags" v-if="activeFlow">
          <span class="method-badge" :class="activeFlow.badge">{{ badgeLabel(activeFlow.badge) }}</span>
          <span class="method-cost">{{ activeFlow.cost }}</span>
        </div>
      </div>

      <div class="demo-events">
        <article v-for="(event, idx) in currentDemo.events" :key="event.title" class="demo-event">
          <div class="event-index">{{ idx + 1 }}</div>
          <div class="event-body">
            <h4>{{ event.title }}</h4>
            <p>{{ event.detail }}</p>
            <div v-if="event.docs?.length" class="event-docs">
              <span
                v-for="doc in docsFor(event.docs)"
                :key="event.title + doc.id"
                class="event-doc"
                :class="doc.kind"
              >
                {{ doc.id }} · {{ doc.label }}
              </span>
            </div>
            <pre v-if="event.output">{{ event.output }}</pre>
          </div>
        </article>
      </div>

      <div class="final-answer">
        <label>最终输出</label>
        <strong>{{ currentDemo.final }}</strong>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, inject, ref, watch } from "vue";

const exp = inject("exp");
const activeMethod = ref(exp.method.value || "naive");

const demoCase = {
  query: "印控克什米尔地区寺庙踩踏事故死亡人数",
  answer: "12人",
};

const demoDocs = [
  {
    id: "P1",
    kind: "positive",
    label: "直接证据",
    text: "海外网报道：位于印控查谟和克什米尔地区的瓦希诺德维寺发生踩踏事件，目前已造成12人死亡、14人受伤。",
  },
  {
    id: "P2",
    kind: "positive",
    label: "复述证据",
    text: "今年1月，印控克什米尔地区一座印度教寺庙发生踩踏事故，造成12人死亡。",
  },
  {
    id: "P3",
    kind: "positive",
    label: "摘要证据",
    text: "Jan 1, 2022：瓦希诺德维寺发生踩踏事件，12人死亡、14人受伤。",
  },
  {
    id: "N1",
    kind: "negative",
    label: "同国相似事故",
    text: "印度南部板球比赛售票现场发生踩踏，至少20人受伤；拉贾斯坦邦寺庙入口踩踏造成3人死亡。",
  },
  {
    id: "N2",
    kind: "negative",
    label: "同类重大事故",
    text: "韩国梨泰院万圣节踩踏事故造成154人死亡。",
  },
  {
    id: "N3",
    kind: "negative",
    label: "同地点不同事故",
    text: "印控克什米尔地区发生巴士坠谷交通事故，已造成10人死亡。",
  },
  {
    id: "N4",
    kind: "negative",
    label: "同类灾害噪音",
    text: "莫尔维吊桥垮塌事故导致135人死亡。",
  },
];

const methodDemos = {
  naive: {
    events: [
      {
        title: "取出这个 case 的检索池",
        detail: "样本里先有问题、标准答案、支撑证据和噪音证据；这个例子中，P1/P2/P3 是目标事故证据，N2/N3/N4 是相似但不该计入的干扰数字。",
        docs: ["P1", "P2", "P3", "N2", "N3", "N4"],
      },
      {
        title: "按噪音设置混成一份上下文",
        detail: "Naive RAG 不额外挑证据，也不先排除噪音；支撑证据和噪音证据会一起出现在同一个上下文里。",
        docs: ["P1", "P2", "P3", "N2", "N3", "N4"],
        output: "上下文 = 目标事故证据 + 相似事故噪音 + 其它灾害数字",
      },
      {
        title: "一次性生成并记录结果",
        detail: "系统把问题和整份上下文一次性交给 Naive RAG，拿到输出后直接作为本方法答案，再和标准答案计算指标。",
        output: "方法输出：12人\n标准答案：12人",
      },
    ],
    final: "12人",
  },
  prompt: {
    events: [
      {
        title: "不换检索池，只换回答规则",
        detail: "它和 Naive 使用同一份混合上下文，也只调用一次；区别是把普通问答规则换成“带噪音质疑”的回答规则。",
        docs: ["P1", "P2", "P3", "N1", "N2", "N3"],
      },
      {
        title: "要求先判断文档是否真有用",
        detail: "有用文档必须能支撑这个问题的答案；只像问题、但事件不对或指标不对的文档，要当作噪音跳过。",
        docs: ["N2", "N3", "N4"],
      },
      {
        title: "只基于有用文档输出答案",
        detail: "在这个 case 里，P1/P2/P3 才支撑寺庙踩踏死亡人数；N2/N3/N4 虽然也有数字，但不作为答案来源。",
        output: "可用依据：P1, P2, P3\n方法输出：12人",
      },
    ],
    final: "12人",
  },
  iterative: {
    events: [
      {
        title: "生成前给每篇文档打分",
        detail: "这一路先做检索质量打分：high 表示可直接支撑答案，mid 表示相关但不够直接，low 表示该在生成前剔除。",
        docs: ["P1", "P2", "P3", "N2", "N3", "N4"],
      },
      {
        title: "过滤 low 后再组织上下文",
        detail: "CRAG-style 的关键动作发生在生成之前：先缩小上下文，再让回答阶段只看保留下来的文档。",
        docs: ["P1", "P2", "P3"],
      },
      {
        title: "一次生成最终答案",
        detail: "生成之后不再额外检查答案是否被证据支持；它比较的是“先过滤上下文”能不能减少相似数字干扰。",
        output: "过滤结果：P1/P2/P3 保留，N2/N3/N4 丢弃或降权\n最终生成：12人",
      },
    ],
    final: "12人",
  },
  selfrag: {
    events: [
      {
        title: "先做相关性门控",
        detail: "第一步不是给 high/mid/low 分，而是判断文档是否值得进入回答阶段：RELEVANT 或 IRRELEVANT。",
        docs: ["P1", "P2", "P3"],
      },
      {
        title: "基于相关文档生成候选答案",
        detail: "这一步会产出一个候选答案，但还不直接结束，因为 Self-RAG-style 还要回头检查这个答案是否真的有支撑。",
        output: "候选输出：12人",
      },
      {
        title: "生成后检查答案支撑性",
        detail: "它和 CRAG-style 的主要区别在这里：答案出来以后还要判 SUPPORTED / PARTIAL / UNSUPPORTED；若不支撑，再缩小文档重生成。",
        docs: ["P1"],
        output: "支撑检查：SUPPORTED\n重生成：不需要",
      },
    ],
    final: "12人",
  },
  confidence: {
    events: [
      {
        title: "套用固定的证据核对模板",
        detail: "不是为每个问题单独设计一条思路，而是所有样本都走同一套模板：先把“要找什么”和“限制条件是什么”列出来。",
        output: "本例槽位：目标事件=瓦希诺德维寺踩踏事故\n本例槽位：目标指标=死亡人数",
      },
      {
        title: "按同一套规则检查文档",
        detail: "每篇文档都拿去和这些槽位对齐：同时满足目标事件和目标指标的才算支持，只相似但不完整的放回噪音侧。",
        docs: ["P1", "P3", "N2", "N3", "N4"],
      },
      {
        title: "汇总支持文档并输出答案",
        detail: "最后只根据被判为支持的文档给出答案，并把用到的支持证据记录下来，方便解释和评测。",
        output: "支持证据：P1, P3\n方法输出：12人",
      },
    ],
    final: "12人",
  },
  voting: {
    events: [
      {
        title: "从多个角度各跑一次",
        detail: "同一个 case 会被拆成几个互补视角处理，例如直接事实、事件匹配、摘要验证。",
        docs: ["P1", "P2", "P3"],
      },
      {
        title: "收集多个候选答案",
        detail: "如果不同视角给出一致结果，就直接采用多数；如果分歧明显，再做一次聚合。",
        output: "候选 A：12人\n候选 B：12人\n候选 C：12人\n多数：12人",
      },
      {
        title: "输出投票后的答案",
        detail: "这个 case 里多个视角都落在 12人，因此最终输出保持一致。",
      },
    ],
    final: "12人",
  },
  adaptive: {
    events: [
      {
        title: "先判断这个 case 的风险类型",
        detail: "上下文里有多个相似灾害数字，但目标证据仍然存在，因此它更像语义噪音问题。",
        docs: ["P1", "N2", "N3", "N4"],
      },
      {
        title: "选择合适的处理策略",
        detail: "系统根据风险判断，把这个 case 交给更强调证据匹配的路径，而不是直接按 Naive 处理。",
        output: "选择路径：证据匹配\n原因：相似数字较多",
      },
      {
        title: "按选中的策略输出答案",
        detail: "最终仍然记录使用了哪条路径，方便后面分析这种路由有没有带来收益。",
        docs: ["P1", "P3"],
      },
    ],
    final: "12人",
  },
  iterative_sc: {
    events: [
      {
        title: "先拿到一个初稿答案",
        detail: "第一轮先按当前上下文得到初稿，不急着把它作为最终结果。",
        docs: ["P1"],
        output: "draft = 12人",
      },
      {
        title: "拿初稿回看证据和噪音",
        detail: "检查初稿是否被目标证据支撑，以及噪音数字是否其实来自另一个事件。",
        docs: ["N2", "N3", "N4"],
      },
      {
        title: "必要时修订，否则保留",
        detail: "这个 case 中初稿能被 P1/P3 支持，所以不再改写答案。",
        output: "check = supported\nrevision = not needed",
      },
    ],
    final: "12人",
  },
};

const flows = computed(() => exp.methodFlowList.value);
const activeFlow = computed(() => (
  flows.value.find(flow => flow.id === activeMethod.value) || flows.value[0] || null
));
const currentDemo = computed(() => methodDemos[activeMethod.value] || methodDemos.naive);

watch(() => exp.method.value, value => {
  if (value && value !== activeMethod.value) activeMethod.value = value;
});

watch(flows, list => {
  if (list.length && !list.some(flow => flow.id === activeMethod.value)) {
    activeMethod.value = list[0].id;
  }
}, { immediate: true });

function selectMethod(id) {
  activeMethod.value = id;
  exp.method.value = id;
}

function docsFor(ids = []) {
  return ids.map(id => demoDocs.find(doc => doc.id === id) || { id, kind: "neutral", label: "文档" });
}

function badgeLabel(value) {
  return ({
    baseline: "基线",
    ours: "本文方法",
    "paper-style": "论文范式",
    custom: "自定义",
  })[value] || value;
}
</script>

<style scoped>
.method-preview {
  height: 100%;
  overflow-y: auto;
  padding: 12px 16px;
  color: var(--ink);
}

.preview-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 10px;
}

.preview-title h2 {
  margin: 0;
  font-size: 16px;
}

.preview-title span {
  font-size: 11px;
  color: var(--ink-muted);
}

.preview-method-select {
  display: flex;
  align-items: center;
  gap: 6px;
}

.preview-method-select label {
  font-size: 11px;
  font-weight: 700;
  color: var(--accent);
  white-space: nowrap;
}

.preview-method-select select {
  height: 28px;
  min-width: 190px;
  padding: 0 8px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: #fff;
  font-size: 12px;
}

.case-panel {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) 140px minmax(0, 1fr);
  gap: 8px;
  margin-bottom: 8px;
}

.case-question,
.case-answer,
.case-note {
  min-width: 0;
  padding: 10px 12px;
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 6px;
}

.case-panel label,
.final-answer label {
  display: block;
  margin-bottom: 4px;
  color: var(--ink-muted);
  font-size: 10px;
  font-weight: 700;
}

.case-question strong,
.case-answer strong {
  color: var(--ink);
  font-size: 14px;
}

.case-note span {
  display: block;
  color: var(--ink-body);
  font-size: 12px;
  line-height: 1.45;
}

.docs-panel {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 8px;
  margin-bottom: 10px;
}

.demo-doc {
  display: grid;
  grid-template-columns: 36px minmax(0, 1fr);
  gap: 8px;
  min-height: 112px;
  padding: 8px 10px;
  background: #fff;
  border: 1px solid var(--border);
  border-left-width: 3px;
  border-radius: 6px;
}

.demo-doc.positive { border-left-color: #15803d; }
.demo-doc.negative { border-left-color: #94a3b8; }

.doc-id {
  width: 28px;
  height: 22px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 11px;
  background: #f1f5f9;
  color: var(--accent);
}

.demo-doc.positive .doc-id {
  color: #166534;
  background: #dcfce7;
}

.demo-doc strong {
  display: block;
  margin-bottom: 4px;
  color: var(--ink);
  font-size: 12px;
}

.demo-doc p {
  margin: 0;
  color: var(--ink-body);
  font-size: 11px;
  line-height: 1.45;
}

.demo-panel {
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}

.demo-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
}

.demo-head h3 {
  margin: 0;
  color: var(--ink);
  font-size: 14px;
}

.demo-head span {
  color: var(--ink-muted);
  font-size: 11px;
}

.method-tags {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 3px;
  flex-shrink: 0;
}

.demo-events {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 0;
}

.demo-event {
  display: grid;
  grid-template-columns: 32px minmax(0, 1fr);
  gap: 8px;
  min-height: 150px;
  padding: 12px;
  border-right: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}

.event-index {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: var(--blue);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 800;
}

.event-body h4 {
  margin: 0 0 5px;
  color: var(--ink);
  font-size: 12px;
}

.event-body p {
  margin: 0;
  color: var(--ink-body);
  font-size: 11px;
  line-height: 1.5;
}

.event-docs {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 8px;
}

.event-doc {
  padding: 2px 6px;
  border-radius: 999px;
  font-size: 10px;
  font-weight: 700;
  background: #f1f5f9;
  color: var(--ink-muted);
}

.event-doc.positive {
  background: #dcfce7;
  color: #166534;
}

.event-doc.negative {
  background: #e2e8f0;
  color: #475569;
}

.event-body pre {
  margin: 8px 0 0;
  padding: 8px;
  white-space: pre-wrap;
  border-radius: 4px;
  background: #f8fafc;
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: 10px;
  line-height: 1.45;
}

.final-answer {
  padding: 10px 12px;
  border-top: 1px solid var(--border);
  background: #f0fdf4;
}

.final-answer strong {
  color: #166534;
  font-size: 18px;
}

@media (max-width: 900px) {
  .preview-top,
  .demo-head {
    align-items: stretch;
    flex-direction: column;
  }

  .preview-method-select {
    align-items: stretch;
    flex-direction: column;
  }

  .preview-method-select select {
    width: 100%;
  }

  .case-panel {
    grid-template-columns: 1fr;
  }
}
</style>

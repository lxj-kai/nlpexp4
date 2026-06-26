<template>
  <div class="data-builder">
    <div class="builder-top">
      <div class="builder-title">
        <h2>数据构建</h2>
        <span>从 MobileMem 原始生活记忆到 RGB 兼容 RAG 测试样本</span>
      </div>
      <div class="dataset-switch" role="tablist" aria-label="MobileMem dataset builders">
        <button
          v-for="item in builders"
          :key="item.key"
          :class="{ active: activeKey === item.key }"
          @click="selectBuilder(item.key)"
        >
          {{ item.label }}
        </button>
      </div>
    </div>

    <div class="builder-summary">
      <div class="summary-main">
        <span class="summary-badge">{{ activeBuilder.badge }}</span>
        <strong>{{ activeBuilder.name }}</strong>
        <span>{{ activeBuilder.goal }}</span>
      </div>
      <div class="summary-stats">
        <span v-for="stat in activeBuilder.stats" :key="stat.label">
          <b>{{ stat.value }}</b>{{ stat.label }}
        </span>
      </div>
    </div>

    <div class="builder-layout">
      <aside class="step-rail">
        <button
          v-for="(step, idx) in activeBuilder.steps"
          :key="step.title"
          class="rail-step"
          :class="{ active: activeStep === idx }"
          @click="activeStep = idx"
        >
          <span class="rail-num">{{ idx + 1 }}</span>
          <span>
            <strong>{{ step.title }}</strong>
            <small>{{ step.short }}</small>
          </span>
        </button>
      </aside>

      <section class="step-detail">
        <div class="step-detail-head">
          <div>
            <span class="step-kicker">Step {{ activeStep + 1 }}</span>
            <h3>{{ currentStep.title }}</h3>
          </div>
          <span class="step-tag">{{ currentStep.tag }}</span>
        </div>

        <div class="detail-grid">
          <div class="detail-panel">
            <div class="panel-label">输入</div>
            <div v-if="currentStep.input.title" class="panel-title">{{ currentStep.input.title }}</div>
            <p v-for="line in currentStep.input.lines" :key="line">{{ line }}</p>
            <div v-if="currentStep.input.fields?.length" class="field-list">
              <div v-for="field in currentStep.input.fields" :key="field.name" class="field-row">
                <span>{{ field.name }}</span>
                <strong>{{ field.value }}</strong>
              </div>
            </div>
          </div>

          <div class="detail-panel">
            <div class="panel-label">处理</div>
            <div v-if="currentStep.process.title" class="panel-title">{{ currentStep.process.title }}</div>
            <p v-for="line in currentStep.process.lines" :key="line">{{ line }}</p>
            <div v-if="currentStep.process.formula" class="formula">
              {{ currentStep.process.formula }}
            </div>
          </div>

          <div class="detail-panel">
            <div class="panel-label">输出</div>
            <div v-if="currentStep.output.title" class="panel-title">{{ currentStep.output.title }}</div>
            <p v-for="line in currentStep.output.lines" :key="line">{{ line }}</p>
            <pre v-if="currentStep.output.code">{{ currentStep.output.code }}</pre>
          </div>
        </div>

        <div v-if="currentStep.records?.length" class="record-table">
          <div class="record-head">
            <span v-for="col in currentStep.columns" :key="col">{{ col }}</span>
          </div>
          <div v-for="row in currentStep.records" :key="row.join('-')" class="record-row">
            <span v-for="(cell, idx) in row" :key="idx">{{ cell }}</span>
          </div>
        </div>
      </section>
    </div>

    <div class="final-sample">
      <div class="sample-block question">
        <label>Query</label>
        <p>{{ activeBuilder.sample.query }}</p>
      </div>
      <div class="sample-block answer">
        <label>Answer</label>
        <p>{{ activeBuilder.sample.answer }}</p>
      </div>
      <div class="sample-block">
        <label>Positive</label>
        <p>{{ activeBuilder.sample.positive }}</p>
      </div>
      <div class="sample-block">
        <label>Negative</label>
        <p>{{ activeBuilder.sample.negative }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from "vue";

const builders = [
  {
    key: "reasoning",
    label: "Reasoning",
    badge: "派生答案",
    name: "MobileMem-Reasoning",
    goal: "先用事件描述定位同一条私人记忆，再从结构化记录中计算出不直接出现在上下文里的答案。",
    stats: [
      { label: "positive", value: "2" },
      { label: "negative", value: "8+" },
      { label: "hardness", value: "anchor+artifact" },
    ],
    steps: [
      {
        title: "打开原始记忆",
        short: "事件锚点 + 关联记录",
        tag: "stage4 / stage7",
        input: {
          title: "同一 event_id 下的两类材料",
          lines: [
            "年度事件提供自然语言场景，用来定位目标记忆。",
            "朋友圈、读书、音乐、转账等截图提供结构化字段。",
          ],
          fields: [
            { name: "event_name", value: "年中生日周末咖啡馆里整理拍摄作品集" },
            { name: "artifact_type", value: "friend" },
          ],
        },
        process: {
          title: "合并同一事件的证据",
          lines: [
            "保留事件描述作为锚点文档。",
            "把同一事件的朋友圈记录转成 artifact 文档。",
          ],
        },
        output: {
          title: "候选正例文档",
          lines: ["P0 是事件描述；P1 是同一事件的朋友圈结构化记录。"],
        },
      },
      {
        title: "抽取可计算字段",
        short: "结构化字段解析",
        tag: "parse_artifact",
        input: {
          title: "朋友圈字段",
          lines: ["likes 与 comments 都是列表字段，不能直接照抄一个原文答案。"],
          fields: [
            { name: "likes", value: "廖晨瑜、梅思岚、温子瑜" },
            { name: "comments", value: "廖晨瑜、温子瑜" },
          ],
        },
        process: {
          title: "构造派生答案",
          lines: ["计算互动总次数，并检查答案没有直接出现在上下文里。"],
          formula: "3 likes + 2 comments = 5次",
        },
        output: {
          title: "reasoning candidate",
          lines: ["reasoning_type=interaction_count；answer_field=likes_plus_comments。"],
        },
      },
      {
        title: "生成问题约束",
        short: "锚点定位 + 计算目标",
        tag: "question builder",
        input: {
          title: "事件定位线索",
          lines: [
            "使用事件描述里的片段，而不是直接复制事件标题。",
            "问题要求先定位事件，再查看同一事件的朋友圈记录。",
          ],
        },
        process: {
          title: "把任务写成两跳问题",
          lines: ["第一跳定位目标事件；第二跳读取 artifact 并计算点赞与评论总数。"],
        },
        output: {
          title: "问题草稿",
          lines: [
            "在私人记忆中，先根据“生日过完的那个周末...”定位对应事件；再查看同一事件的朋友圈记录并进行计算，朋友圈里的点赞和评论加起来一共有多少次互动？",
          ],
        },
      },
      {
        title: "筛选真实噪音",
        short: "相近但不支持",
        tag: "negative mining",
        input: {
          title: "其它 MobileMem 记忆",
          lines: ["只借用真实生成的其它事件文档，不制造与目标答案冲突的反事实文档。"],
        },
        process: {
          title: "相似度与支持性过滤",
          lines: [
            "优先选朋友圈、生日、照片整理等主题相近文档。",
            "排除会支持 5次 这个答案的文档。",
          ],
        },
        output: {
          title: "negative pool",
          lines: ["示例噪音：雨夜下班朋友圈、除夕晒年夜饭、春季城市咖啡小聚等。"],
        },
      },
      {
        title: "导出 RGB 样本",
        short: "统一字段格式",
        tag: "RGB schema",
        input: {
          title: "正例、负例和派生答案",
          lines: ["把构建结果落到 query / answer / positive / negative / positive_wrong。"],
        },
        process: {
          title: "质量约束",
          lines: ["requires_anchor_doc=true；requires_artifact_doc=true；answer_not_verbatim_in_context=true。"],
        },
        output: {
          title: "最终 JSON",
          lines: [],
          code: `{
  "id": 320001,
  "answer": ["5次"],
  "positive": 2,
  "negative": 8,
  "mobilemem_meta.reasoning_type": "interaction_count"
}`,
        },
      },
    ],
    sample: {
      query: "先定位“生日过完的那个周末...”对应事件，再查看同一事件的朋友圈记录并计算点赞和评论总互动数。",
      answer: "5次",
      positive: "事件描述 + 同一事件朋友圈记录",
      negative: "真实但不支持答案的其它朋友圈/生活事件",
    },
  },
  {
    key: "graph",
    label: "Graph",
    badge: "跨事件聚合",
    name: "MobileMem-Graph Hard",
    goal: "筛选多条满足同一条件的目标记录，再跨文档聚合或比较数值，单篇文档无法直接回答。",
    stats: [
      { label: "positive", value: "4" },
      { label: "negative", value: "10" },
      { label: "operation", value: "sum" },
    ],
    steps: [
      {
        title: "建立记录图谱",
        short: "事件节点 + artifact 节点",
        tag: "artifact graph",
        input: {
          title: "MobileMem 全年记录",
          lines: [
            "每个事件保留 event_id、event_time、parent_event 与 artifact 类型。",
            "购物、转账、车票、音乐、视频等结构化记录都挂到事件节点上。",
          ],
        },
        process: {
          title: "标准化可比较字段",
          lines: ["把实际价格、转账金额、票价等解析成可聚合的数值。"],
        },
        output: {
          title: "候选 artifact 表",
          lines: ["每条记录拥有 record_type、event_time、metric_value。"],
        },
      },
      {
        title: "生成目标条件",
        short: "类型 + 时间窗",
        tag: "conditioned_window",
        input: {
          title: "条件模板",
          lines: ["记录类型是购物截图；发生时间在闭区间 2025-01-24 21:00 至 2025-03-24 20:10。"],
        },
        process: {
          title: "扫描图谱并筛选",
          lines: ["只有 artifact 类型和事件发生时间同时满足条件才进入目标集合。"],
        },
        output: {
          title: "目标记录数",
          lines: ["共筛出 4 条目标购物记录。"],
        },
        columns: ["事件ID", "发生时间", "商品", "实际价格"],
        records: [
          ["8", "2025-01-24 21:00", "宠物智能摄像头远程可视版", "239元"],
          ["13", "2025-01-29 15:00", "春节年货零食大礼包", "238元"],
          ["22", "2025-03-15 14:30", "商务休闲修身长袖衬衫两件装", "398元"],
          ["23", "2025-03-24 20:10", "高铁往返车票深圳至长沙代购服务", "24元"],
        ],
      },
      {
        title: "执行聚合计算",
        short: "多文档求和",
        tag: "metric=sum",
        input: {
          title: "四条目标记录的 metric_value",
          lines: ["每条 positive 只提供一个局部数值，不能只读其中一条。"],
        },
        process: {
          title: "跨事件聚合",
          lines: ["把四条购物截图的实际价格全部相加。"],
          formula: "239 + 238 + 398 + 24 = 899元",
        },
        output: {
          title: "派生答案",
          lines: ["answer_is_derived=true；single_doc_answerable=false。"],
        },
      },
      {
        title: "加入迷惑文档",
        short: "同类型非目标",
        tag: "real negatives",
        input: {
          title: "同为购物截图的其它记录",
          lines: ["选择同类型、同月份或主题接近的真实记录，让噪音足够像目标记录。"],
        },
        process: {
          title: "排除目标条件外文档",
          lines: [
            "时间窗外的购物记录不能进入 positive。",
            "价格更高或主题更像的记录也不能计入答案。",
          ],
        },
        output: {
          title: "negative 示例",
          lines: ["春节前妻子礼物 3299元、清明后高铁票 486.5元、元旦双肩包 198元。"],
        },
      },
      {
        title: "导出图谱样本",
        short: "多正例约束",
        tag: "RGB schema",
        input: {
          title: "目标集合与噪音集合",
          lines: ["positive 是 4 条目标记录；negative 是 10 条真实非目标记录。"],
        },
        process: {
          title: "写入可审计元数据",
          lines: ["记录 target_condition、support_path、operation、metric 等字段，方便回放构建过程。"],
        },
        output: {
          title: "最终 JSON",
          lines: [],
          code: `{
  "id": 340001,
  "answer": ["899元"],
  "positive": 4,
  "negative": 10,
  "mobilemem_meta.operation": "sum",
  "mobilemem_meta.required_positive_docs": 4
}`,
        },
      },
    ],
    sample: {
      query: "筛选 2025-01-24 21:00 至 2025-03-24 20:10 的 4 条购物截图，并合计实际价格。",
      answer: "899元",
      positive: "4 条目标购物截图，每条给出一个实际价格",
      negative: "同类型但不满足时间窗或条件的真实购物记录",
    },
  },
];

const activeKey = ref("reasoning");
const activeStep = ref(0);

const activeBuilder = computed(() => builders.find((item) => item.key === activeKey.value) || builders[0]);
const currentStep = computed(() => activeBuilder.value.steps[activeStep.value] || activeBuilder.value.steps[0]);

function selectBuilder(key) {
  activeKey.value = key;
  activeStep.value = 0;
}
</script>

<style scoped>
.data-builder {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 12px 16px;
}

.builder-top {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 8px;
}

.builder-title {
  flex: 1;
  min-width: 220px;
}

.builder-title h2 {
  font-size: 16px;
  margin: 0;
  color: var(--ink);
}

.builder-title span {
  font-size: 11px;
  color: var(--ink-muted);
}

.dataset-switch {
  display: flex;
  gap: 4px;
  flex-shrink: 0;
}

.dataset-switch button {
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 700;
  padding: 5px 12px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: #fff;
  color: var(--ink-muted);
  cursor: pointer;
}

.dataset-switch button.active {
  background: var(--blue);
  border-color: var(--blue);
  color: #fff;
}

.builder-summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 10px;
  border: 1px solid #c7d2fe;
  border-radius: 6px;
  background: var(--blue-bg);
  margin-bottom: 8px;
}

.summary-main {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  font-size: 12px;
  color: var(--ink-body);
}

.summary-main strong {
  color: var(--ink);
  white-space: nowrap;
}

.summary-main span:last-child {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.summary-badge {
  flex-shrink: 0;
  padding: 2px 6px;
  border-radius: 3px;
  background: #fff;
  border: 1px solid #bfdbfe;
  color: #1d4ed8;
  font-size: 10px;
  font-weight: 800;
}

.summary-stats {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.summary-stats span {
  padding: 2px 6px;
  border-radius: 3px;
  background: #fff;
  border: 1px solid #dbe3ef;
  font-size: 10px;
  color: var(--ink-muted);
  white-space: nowrap;
}

.summary-stats b {
  color: var(--ink);
  margin-right: 3px;
}

.builder-layout {
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr);
  gap: 8px;
  min-height: 390px;
}

.step-rail {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.rail-step {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 8px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: #fff;
  color: var(--ink-body);
  cursor: pointer;
  text-align: left;
}

.rail-step.active {
  border-color: var(--blue);
  background: #eff6ff;
}

.rail-num {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: #e2e8f0;
  color: #334155;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 800;
  flex-shrink: 0;
}

.rail-step.active .rail-num {
  background: var(--blue);
  color: #fff;
}

.rail-step strong {
  display: block;
  font-size: 12px;
  line-height: 1.2;
}

.rail-step small {
  display: block;
  margin-top: 2px;
  color: var(--ink-muted);
  font-size: 10px;
  line-height: 1.25;
}

.step-detail {
  min-width: 0;
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}

.step-detail-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 9px 12px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
}

.step-kicker {
  display: block;
  font-size: 10px;
  font-weight: 800;
  color: var(--blue);
  text-transform: uppercase;
}

.step-detail h3 {
  margin: 1px 0 0;
  font-size: 15px;
  color: var(--ink);
}

.step-tag {
  padding: 3px 7px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: #fff;
  font-size: 10px;
  font-weight: 800;
  color: var(--ink-muted);
  white-space: nowrap;
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  padding: 10px;
}

.detail-panel {
  min-height: 190px;
  padding: 9px 10px;
  border: 1px solid #dbe3ef;
  border-radius: 6px;
  background: #fff;
}

.panel-label {
  font-size: 10px;
  font-weight: 800;
  color: var(--blue);
  margin-bottom: 4px;
}

.panel-title {
  font-size: 12px;
  font-weight: 800;
  color: var(--ink);
  margin-bottom: 5px;
}

.detail-panel p {
  font-size: 11px;
  line-height: 1.55;
  color: var(--ink-body);
  margin: 0 0 5px;
}

.field-list {
  margin-top: 7px;
  border-top: 1px solid #e2e8f0;
  padding-top: 5px;
}

.field-row {
  display: grid;
  grid-template-columns: 78px minmax(0, 1fr);
  gap: 6px;
  padding: 3px 0;
  font-size: 10px;
  line-height: 1.35;
}

.field-row span {
  color: var(--ink-muted);
  font-family: var(--font-mono);
}

.field-row strong {
  color: var(--ink);
  font-weight: 700;
  word-break: break-word;
}

.formula {
  margin-top: 9px;
  padding: 8px;
  border-radius: 5px;
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  color: #166534;
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 800;
  line-height: 1.35;
}

.detail-panel pre {
  max-height: 120px;
  overflow: auto;
  margin: 4px 0 0;
  padding: 7px;
  border-radius: 4px;
  border: 1px solid #dbe3ef;
  background: #f8fafc;
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: 10px;
  line-height: 1.45;
  white-space: pre-wrap;
}

.record-table {
  margin: 0 10px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}

.record-head,
.record-row {
  display: grid;
  grid-template-columns: 70px 140px minmax(0, 1fr) 90px;
  gap: 0;
}

.record-head span {
  padding: 6px 8px;
  background: #f1f5f9;
  border-right: 1px solid var(--border);
  font-size: 10px;
  font-weight: 800;
  color: var(--accent);
}

.record-row span {
  padding: 6px 8px;
  border-top: 1px solid #e2e8f0;
  border-right: 1px solid #e2e8f0;
  font-size: 11px;
  color: var(--ink-body);
  line-height: 1.35;
  overflow-wrap: anywhere;
}

.record-head span:last-child,
.record-row span:last-child {
  border-right: 0;
}

.final-sample {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(120px, 0.6fr) minmax(0, 1fr) minmax(0, 1fr);
  gap: 8px;
  margin-top: 8px;
}

.sample-block {
  min-height: 74px;
  padding: 8px 9px;
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 6px;
}

.sample-block label {
  display: block;
  margin-bottom: 4px;
  font-size: 10px;
  font-weight: 800;
  color: var(--accent);
}

.sample-block p {
  font-size: 11px;
  line-height: 1.45;
  color: var(--ink-body);
}

.sample-block.answer {
  border-left: 3px solid var(--positive);
}

.sample-block.answer p {
  color: var(--positive);
  font-size: 15px;
  font-weight: 800;
}

.sample-block.question {
  border-left: 3px solid var(--blue);
}

@media (max-width: 900px) {
  .builder-top,
  .builder-summary {
    align-items: stretch;
    flex-direction: column;
  }

  .builder-layout {
    grid-template-columns: 1fr;
  }

  .step-rail {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .detail-grid,
  .final-sample {
    grid-template-columns: 1fr;
  }

  .summary-main,
  .summary-main span:last-child {
    white-space: normal;
  }
}
</style>

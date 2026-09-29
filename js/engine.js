/* 检索引擎的浏览器实现 —— 与 Python 版保持算法一致。
 *
 * 为什么能搬过来: BM25、实体链接、图遍历都是确定性算法，数据量也小
 * (1092 个 chunk / 227 实体 / 477 条边)，浏览器毫秒级跑完。
 * 唯一没搬的是 dense 向量检索 —— 那需要额外加载 10MB 的 ONNX WASM，
 * 对演示页面不划算。所以静态版的语义类问题会弱于完整版，页面上有注明。
 */

/* ---------- 分词: CJK 字符 bigram + ASCII 整词 + 数字 ---------- */
const RE_CJK = /[一-鿿]+/g;
const RE_ASCII = /[A-Za-z]+/g;
const RE_NUM = /\d+(?:\.\d+)?/g;

export function tokenize(text) {
  const out = [];
  let m;
  RE_CJK.lastIndex = 0;
  while ((m = RE_CJK.exec(text))) {
    const s = m[0];
    if (s.length === 1) out.push(s);
    else for (let i = 0; i < s.length - 1; i++) out.push(s.slice(i, i + 2));
  }
  RE_ASCII.lastIndex = 0;
  while ((m = RE_ASCII.exec(text))) out.push(m[0].toLowerCase());
  RE_NUM.lastIndex = 0;
  while ((m = RE_NUM.exec(text))) out.push(m[0]);
  return out;
}

/* ---------- BM25 ---------- */
export class BM25 {
  constructor(k1 = 1.5, b = 0.75) { this.k1 = k1; this.b = b; }

  index(items) {                       // items: [{id, text}]
    this.ids = []; this.tf = []; this.len = [];
    this.df = new Map(); this.post = new Map();
    for (const { id, text } of items) {
      const toks = tokenize(text);
      const tf = new Map();
      for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
      const i = this.ids.length;
      this.ids.push(id); this.tf.push(tf); this.len.push(toks.length);
      for (const t of tf.keys()) {
        this.df.set(t, (this.df.get(t) || 0) + 1);
        if (!this.post.has(t)) this.post.set(t, []);
        this.post.get(t).push(i);
      }
    }
    this.n = this.ids.length;
    this.avg = this.len.reduce((a, x) => a + x, 0) / (this.n || 1);
    return this;
  }

  search(query, topK = 10) {
    const q = new Map();
    for (const t of tokenize(query)) q.set(t, (q.get(t) || 0) + 1);
    const scores = new Map();
    for (const term of q.keys()) {
      const df = this.df.get(term) || 0;
      if (!df) continue;
      const idf = Math.max(Math.log(1 + (this.n - df + 0.5) / (df + 0.5)), 1e-6);
      for (const i of this.post.get(term)) {
        const f = this.tf[i].get(term);
        const denom = f + this.k1 * (1 - this.b + this.b * this.len[i] / this.avg);
        scores.set(i, (scores.get(i) || 0) + idf * f * (this.k1 + 1) / denom);
      }
    }
    return [...scores.entries()].sort((a, b) => b[1] - a[1]).slice(0, topK)
      .map(([i, s]) => [this.ids[i], s]);
  }
}

/* ---------- 知识图谱 ---------- */
const PUNCT = "（）()。，,、．. 　\t\r\n·:：“”\"'-—_/\\";
const NOISE = ["有限公司", "股份有限公司", "ltd", "limited", "pte", "inc",
               "corp", "corporation", "co", "company"];

export function normalizeName(s) {
  s = (s || "").trim().toLowerCase();
  for (const c of PUNCT) s = s.split(c).join("");
  for (const w of NOISE) s = s.split(w).join("");
  return s;
}

export class Graph {
  constructor(raw) {
    this.ent = new Map();
    for (const e of raw.entities) this.ent.set(e.id, e);
    this.out = new Map(); this.in = new Map();
    this.edges = raw.edges;
    for (const e of raw.edges) {
      if (!this.out.has(e.s)) this.out.set(e.s, []);
      if (!this.in.has(e.o)) this.in.set(e.o, []);
      this.out.get(e.s).push(e); this.in.get(e.o).push(e);
    }
    // 实体表面形式，长的优先 —— 保证最长匹配
    this.surface = [];
    for (const e of raw.entities)
      for (const f of [e.n, ...(e.a || [])])
        if (f && f.length >= 2) this.surface.push([f, e.id]);
    this.surface.sort((a, b) => b[0].length - a[0].length);
  }

  name(id) {
    if (id.startsWith("lit:")) return id.slice(4);
    const e = this.ent.get(id);
    return e ? e.n : id;
  }
  type(id) {
    if (id.startsWith("lit:")) return "Literal";
    const e = this.ent.get(id);
    return e ? e.t : "Unknown";
  }
  risk(id) { const e = this.ent.get(id); return e ? e.risk : null; }
  degree(id) {
    return (this.out.get(id) || []).length + (this.in.get(id) || []).length;
  }

  /* 字面量不得作为遍历中转 —— 两只基金因金额数字相同就"有关联"是假阳性 */
  steps(id) {
    if (id.startsWith("lit:")) return [];
    const r = [];
    for (const e of this.out.get(id) || []) r.push({ e, dir: 1, to: e.o, from: e.s });
    for (const e of this.in.get(id) || []) r.push({ e, dir: -1, to: e.s, from: e.o });
    return r;
  }

  link(text) {
    const found = new Map();
    const used = new Array(text.length).fill(false);
    for (const [form, id] of this.surface) {
      let at = text.indexOf(form);
      while (at !== -1) {
        let free = true;
        for (let i = at; i < at + form.length; i++) if (used[i]) { free = false; break; }
        if (free) {
          for (let i = at; i < at + form.length; i++) used[i] = true;
          if (!found.has(id)) found.set(id, form);
          break;
        }
        at = text.indexOf(form, at + 1);
      }
    }
    return [...found.entries()];
  }

  /* 枚举到目标的多条简单路径，不穿透枢纽节点 */
  pathsTo(start, pred, maxHops = 4, maxPaths = 12, hubDegree = 12) {
    const out = [];
    const stack = [[start, [], new Set([start])]];
    while (stack.length && out.length < maxPaths) {
      const [node, path, seen] = stack.pop();
      if (path.length >= maxHops) continue;
      if (path.length && hubDegree != null && this.degree(node) > hubDegree) continue;
      for (const st of this.steps(node)) {
        if (seen.has(st.to)) continue;
        const np = [...path, st];
        if (pred(st.to)) { out.push(np); if (out.length >= maxPaths) break; }
        else stack.push([st.to, np, new Set([...seen, st.to])]);
      }
    }
    out.sort((a, b) => a.length - b.length);
    return out;
  }

  pathDict(path) {
    const nodes = [], seen = new Set();
    const push = (id) => {
      if (seen.has(id)) return;
      seen.add(id);
      nodes.push({ id, name: this.name(id), type: this.type(id), risk: this.risk(id) });
    };
    if (!path.length) return { nodes: [], edges: [] };
    push(path[0].from);
    const edges = [];
    for (const st of path) {
      push(st.to);
      edges.push({ from: st.from, to: st.to, rel: st.e.r, docs: st.e.d || [] });
    }
    return { nodes, edges };
  }
}

/* ---------- 意图路由 + 图查询模板 ---------- */
const RISK_HINTS = ["高风险", "风险", "制裁", "可疑", "关联方", "穿透", "关联"];
const SHARED_HINTS = ["共同董事", "共同", "同时担任", "同时是"];
const COUNT_HINTS = ["几只", "几家", "多少", "数量", "列出", "哪些"];
const REL_HINTS = ["管理人", "托管", "注册在", "母公司", "隶属", "股东", "董事",
                   "关联", "穿透", "旗下"];

export function route(question, linked) {
  if (!linked.length) return { primary: "lexical", reason: "未链接到任何图实体" };
  if (SHARED_HINTS.some(h => question.includes(h)))
    return { primary: "graph", reason: "共同董事类: 证据跨多家主体, 无词面重叠" };
  if (RISK_HINTS.some(h => question.includes(h)))
    return { primary: "graph", reason: "风险穿透类: 需沿关系链多跳" };
  if (COUNT_HINTS.some(h => question.includes(h)) && REL_HINTS.some(h => question.includes(h)))
    return { primary: "graph", reason: "聚合类: top-k 范式无法覆盖全部证据" };
  if (REL_HINTS.some(h => question.includes(h)))
    return { primary: "graph", reason: "关系类: 答案需跨文档拼接" };
  return { primary: "lexical", reason: "单跳属性类: 词面检索已足够" };
}

export function graphSearch(g, question, docToChunks, topK) {
  const linked = g.link(question);
  const r = route(question, linked);
  if (!linked.length) return { hits: [], graphs: [], linked, route: r, intent: "" };

  const seeds = linked.map(([id]) => id);
  const hits = new Map();          // doc -> {hop, path}
  const graphs = [];
  let intent = "neighborhood";

  const addPath = (path) => {
    const gd = g.pathDict(path);
    if (graphs.length < 4) graphs.push(gd);
    for (const st of path)
      for (const d of st.e.d || [])
        if (!hits.has(d)) hits.set(d, path.length);
  };

  if (SHARED_HINTS.some(h => question.includes(h))) {
    intent = "shared_director";
    for (const seed of seeds)
      for (const e1 of g.in.get(seed) || []) {
        if (e1.r !== "DIRECTOR_OF") continue;
        for (const e2 of g.out.get(e1.s) || []) {
          if (e2.r !== "DIRECTOR_OF" || e2.o === seed) continue;
          addPath([{ e: e1, dir: -1, from: seed, to: e1.s },
                   { e: e2, dir: 1, from: e1.s, to: e2.o }]);
        }
      }
  } else if (RISK_HINTS.some(h => question.includes(h))) {
    intent = "risk_path";
    const risky = (id) => id === "lit:高风险" || g.risk(id) === "高风险";
    for (const seed of seeds)
      for (const path of g.pathsTo(seed, risky)) {
        addPath(path);
        // 目标辖区的"高风险"这一事实来自监管名单，证据要带上
        const tail = path[path.length - 1].to;
        for (const e of g.out.get(tail) || [])
          if (e.r === "RISK_LEVEL")
            for (const d of e.d || []) if (!hits.has(d)) hits.set(d, path.length);
      }
  }

  if (!hits.size) {              // 兜底: 邻域扩散
    // 与 Python 版保持同一套标签 —— 两版的 intent 字符串要能逐题对上,
    // 否则没法做跨语言一致性校验。
    intent = intent === "neighborhood" ? "neighborhood" : "neighborhood(fallback)";
    let frontier = seeds.map(s => [s, []]);
    const seen = new Set(seeds);
    for (let hop = 1; hop <= 3 && frontier.length; hop++) {
      const next = [];
      for (const [node, path] of frontier)
        for (const st of g.steps(node)) {
          const np = [...path, st];
          for (const d of st.e.d || []) if (!hits.has(d)) hits.set(d, hop);
          if (!seen.has(st.to)) { seen.add(st.to); next.push([st.to, np]); }
          if (graphs.length < 3 && hop <= 2) graphs.push(g.pathDict(np));
        }
      frontier = next;
    }
  }

  // 按文档轮转发放预算，保证与 BM25 在同一量纲上比较
  const docs = [...hits.entries()].sort((a, b) => a[1] - b[1] || (a[0] < b[0] ? -1 : 1));
  const out = [];
  const maxR = Math.max(0, ...docs.map(([d]) => (docToChunks.get(d) || []).length));
  for (let r0 = 0; r0 < maxR && out.length < topK; r0++)
    for (const [doc, hop] of docs) {
      const cs = docToChunks.get(doc) || [];
      if (r0 < cs.length) { out.push([cs[r0], 1 / hop - r0 * 1e-4]); if (out.length >= topK) break; }
    }
  return { hits: out, graphs: graphs.slice(0, 3), linked, route: r, intent };
}

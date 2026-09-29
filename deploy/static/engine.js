/* 检索引擎的浏览器实现 —— 与 Python 版保持算法一致。
 *
 * 为什么能搬过来: BM25、实体链接、图遍历都是确定性算法，数据量也小
 * (两百来个 chunk / 几十个实体 / 不到两百条边)，浏览器毫秒级跑完。
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
const PUNCT = "（）()。，,、．. 　\t\r\n·:：“”\"'-—_/\\《》「」！!？?…";

export function normalizeName(s) {
  s = (s || "").trim().toLowerCase();
  for (const c of PUNCT) s = s.split(c).join("");
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
    // 实体表面形式，长的优先 —— 保证最长匹配("灰二太太狼"不被"灰太狼"吃掉)
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
  /* 只数连向实体的边 —— 字面量边(口头禅、原型……)不让普通角色变成枢纽 */
  entityDegree(id) {
    return (this.out.get(id) || []).filter(e => !e.o.startsWith("lit:")).length
         + (this.in.get(id) || []).length;
  }
  isHub(id, hubDegree) { return hubDegree != null && this.entityDegree(id) > hubDegree; }

  /* 字面量不得作为遍历中转 —— 两个角色因变身时限相同就"有关系"是假阳性 */
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

  /* src -> dst 的简单路径，按层 BFS(短的在前)，不穿透枢纽 —— 与 Python find_paths 一致 */
  findPaths(src, dst, maxHops = 4, limit = 12, hubDegree = 12) {
    const out = [];
    let frontier = [[src, [], new Set([src])]];
    for (let h = 0; h < maxHops && frontier.length; h++) {
      const next = [];
      for (const [node, path, seen] of frontier) {
        if (path.length && this.isHub(node, hubDegree)) continue;
        for (const st of this.steps(node)) {
          if (seen.has(st.to)) continue;
          const np = [...path, st];
          if (st.to === dst) { out.push(np); if (out.length >= limit) return out; }
          else next.push([st.to, np, new Set([...seen, st.to])]);
        }
      }
      frontier = next;
    }
    return out;
  }

  pathDict(path) {
    const nodes = [], seen = new Set();
    const push = (id) => {
      if (seen.has(id)) return;
      seen.add(id);
      nodes.push({ id, name: this.name(id), type: this.type(id) });
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
const HUB_DEGREE = 12;
const PATH_HINTS = ["什么关系", "有关系", "有什么联系", "什么联系", "有联系", "关联",
                    "有什么关系", "是什么人", "的什么人"];
const COUNT_HINTS = ["哪些", "哪几", "几个", "几只", "几台", "多少", "列出", "都有谁", "有谁"];
const REL_HINTS = ["爸爸", "妈妈", "父亲", "母亲", "爷爷", "奶奶", "外公", "外婆",
                   "老婆", "妻子", "老公", "丈夫", "儿子", "女儿", "表哥", "表妹", "表姐",
                   "表姨", "侄子", "二叔", "岳父", "搭档", "大哥", "老大", "首领", "成员",
                   "喜欢的", "好朋友", "校长", "村长", "封印", "住着", "住在", "学生",
                   "阵营", "三人组", "解开"];
/* 问题里的关系词 -> 图里的关系类型 —— 与 Python 版 REL_WORDS 逐项一致 */
const REL_WORDS = {
  "爸爸": ["PARENT_OF"], "妈妈": ["PARENT_OF"], "父亲": ["PARENT_OF"],
  "母亲": ["PARENT_OF"], "儿子": ["PARENT_OF"], "女儿": ["PARENT_OF"],
  "爷爷": ["PARENT_OF", "GRANDPARENT_OF"], "奶奶": ["PARENT_OF", "GRANDPARENT_OF"],
  "外公": ["PARENT_OF", "GRANDPARENT_OF"], "外婆": ["PARENT_OF", "GRANDPARENT_OF"],
  "代孙": ["DESCENDANT_OF", "PARENT_OF"],
  "老婆": ["SPOUSE_OF"], "妻子": ["SPOUSE_OF"], "老公": ["SPOUSE_OF"],
  "丈夫": ["SPOUSE_OF"], "岳父": ["SPOUSE_OF", "PARENT_OF"],
  "岳母": ["SPOUSE_OF", "PARENT_OF"],
  "表哥": ["COUSIN_OF"], "表妹": ["COUSIN_OF"], "表姐": ["COUSIN_OF"],
  "表弟": ["COUSIN_OF"], "表姨": ["COUSIN_OF", "PARENT_OF"],
  "二叔": ["UNCLE_OF"], "叔叔": ["UNCLE_OF"], "侄子": ["UNCLE_OF"],
  "搭档": ["PARTNER_OF"],
  "大哥": ["LEADER_OF", "MEMBER_OF"], "老大": ["LEADER_OF", "MEMBER_OF"],
  "首领": ["LEADER_OF"], "成员": ["MEMBER_OF"], "组成": ["MEMBER_OF"],
  "喜欢": ["LIKES"], "好朋友": ["FRIEND_OF"],
  "校长": ["HEAD_OF"], "村长": ["HEAD_OF"], "当家": ["HEAD_OF"],
  "上学": ["STUDENT_OF"], "学生": ["STUDENT_OF"],
  "住着": ["LIVES_IN"], "住在": ["LIVES_IN"],
  "坐落": ["LOCATED_IN"], "位于": ["LOCATED_IN"],
  "封印": ["SEALED_BY", "RELEASED_BY"],
  "许愿": ["WISHED_ON"],
  "原型": ["PROTOTYPE"], "武器": ["WEAPON"], "兵器": ["WEAPON"],
  "爱吃": ["FAVORITE_FOOD"], "口头禅": ["CATCHPHRASE"],
  "职务": ["ROLE"], "身份": ["ROLE"], "做什么": ["ROLE"], "干什么": ["ROLE"],
  "必杀技": ["SPECIAL_MOVE"], "编号": ["UNIT_NO"], "号机": ["UNIT_NO"],
  "变换": ["TRANSFORM_TIME"], "变身": ["TRANSFORM_TIME"],
  "巨人": ["GIANT_FORM"], "阵营": ["FACTION"], "生日": ["BIRTHDAY"],
};

export function route(question, linked) {
  if (!linked.length) return { primary: "lexical", reason: "未链接到任何图实体" };
  if (linked.length >= 2 && PATH_HINTS.some(h => question.includes(h)))
    return { primary: "graph", reason: "关系路径类: 两端实体之间的中间环节无词面重叠" };
  if (COUNT_HINTS.some(h => question.includes(h)) && REL_HINTS.some(h => question.includes(h)))
    return { primary: "graph", reason: "聚合类: top-k 范式无法覆盖全部证据" };
  if (REL_HINTS.some(h => question.includes(h)))
    return { primary: "graph", reason: "关系类: 答案需跨文档拼接" };
  return { primary: "lexical", reason: "单跳属性类: 词面检索已足够" };
}

/* chunks: 可选的 Map(chunkId -> {r: 原文})。给了就在文档内优先发写着这条关系的那一块 */
export function graphSearch(g, question, docToChunks, topK, chunks = null) {
  const linked = g.link(question);
  const r = route(question, linked);
  if (!linked.length) return { hits: [], graphs: [], linked, route: r, intent: "" };

  const seeds = linked.map(([id]) => id);
  const hits = new Map();          // doc -> {hop, focus:Set}
  const graphs = [];
  let intent = "neighborhood";

  const forms = (id) => {
    if (id.startsWith("lit:")) return [id.slice(4)];
    if (g.isHub(id, HUB_DEGREE)) return [];
    const e = g.ent.get(id);
    return e ? [e.n, ...(e.a || [])] : [];
  };
  const add = (doc, e, hop) => {
    if (!doc) return;
    if (!hits.has(doc)) hits.set(doc, { hop, focus: new Set() });
    for (const f of [...forms(e.s), ...forms(e.o)]) hits.get(doc).focus.add(f);
  };

  if (seeds.length >= 2 && PATH_HINTS.some(h => question.includes(h))) {
    intent = "relation_path";
    for (let i = 0; i < seeds.length; i++)
      for (let j = i + 1; j < seeds.length; j++)
        for (const path of g.findPaths(seeds[i], seeds[j], 4, 12, HUB_DEGREE)) {
          if (graphs.length < 4) graphs.push(g.pathDict(path));
          for (const st of path) for (const d of st.e.d || []) add(d, st.e, path.length);
        }
  } else {
    const rels = new Set();
    for (const [w, rs] of Object.entries(REL_WORDS)) if (question.includes(w)) rs.forEach(x => rels.add(x));
    if (rels.size) {
      intent = "relation_chain";
      let frontier = seeds.map(s => [s, []]);
      const seen = new Set(seeds), reached = [];
      for (let hop = 1; hop < 4 && frontier.length; hop++) {
        const next = [];
        for (const [node, path] of frontier)
          for (const st of g.steps(node)) {
            if (!rels.has(st.e.r)) continue;
            const np = [...path, st];
            for (const d of st.e.d || []) add(d, st.e, hop);
            if (graphs.length < 3) graphs.push(g.pathDict(np));
            if (!seen.has(st.to) && !g.isHub(st.to, HUB_DEGREE)) {
              seen.add(st.to); next.push([st.to, np]); reached.push([st.to, hop]);
            }
          }
        frontier = next;
      }
      // 终点实体再补上它自己的档案 —— 性格、爱好只写在档案里, 不在任何一条边上
      for (const [node, hop] of reached)
        for (const e of g.out.get(node) || [])
          if (e.r === "APPEARS_IN") for (const d of e.d || []) if (!hits.has(d)) add(d, e, hop + 1);
    }
  }

  if (!hits.size) {              // 兜底: 邻域扩散
    // 与 Python 版保持同一套标签 —— 两版的 intent 字符串要能逐题对上,
    // 否则没法做跨语言一致性校验。
    intent = intent === "neighborhood" ? "neighborhood" : "neighborhood(fallback)";
    let frontier = seeds.map(s => [s, []]);
    const seen = new Set(seeds);
    for (let hop = 1; hop <= 4 && frontier.length; hop++) {
      const next = [];
      for (const [node, path] of frontier)
        for (const st of g.steps(node)) {
          const np = [...path, st];
          for (const d of st.e.d || []) add(d, st.e, hits.has(d) ? hits.get(d).hop : hop);
          // 枢纽(节目节点)可以作为终点，不能再往外扩
          if (!seen.has(st.to) && !g.isHub(st.to, HUB_DEGREE)) {
            seen.add(st.to); next.push([st.to, np]);
          }
          if (graphs.length < 3 && hop <= 2) graphs.push(g.pathDict(np));
        }
      frontier = next;
    }
  }

  // 文档内的块顺序: 提到 focus 名称多的块优先(不数档案主人自己的名字, 先抹掉节目名)
  const subjectOf = g.subjectOf || (g.subjectOf = (() => {
    const m = new Map();
    for (const e of g.edges) if (e.r === "APPEARS_IN") for (const d of e.d || []) if (!m.has(d)) m.set(d, e.s);
    return m;
  })());
  const hubNames = g.hubNames || (g.hubNames = [...g.ent.values()]
    .filter(e => g.isHub(e.id, HUB_DEGREE)).flatMap(e => [e.n, ...(e.a || [])])
    .sort((a, b) => b.length - a.length));
  const order = (doc, focus) => {
    const cs = docToChunks.get(doc) || [];
    if (!chunks || !focus.size) return cs;
    const own = new Set(subjectOf.has(doc) ? forms(subjectOf.get(doc)) : []);
    const fs = [...focus].filter(f => f && !own.has(f));
    const score = (cid) => {
      let t = (chunks.get(cid) || {}).r || "";
      for (const n of hubNames) t = t.split(n).join("");
      return fs.reduce((a, f) => a + (t.includes(f) ? 1 : 0), 0);
    };
    return cs.map((c, i) => [c, score(c), i]).sort((a, b) => b[1] - a[1] || a[2] - b[2]).map(x => x[0]);
  };

  // 按文档轮转发放预算，保证与 BM25 在同一量纲上比较
  const docs = [...hits.entries()].sort((a, b) => a[1].hop - b[1].hop || (a[0] < b[0] ? -1 : 1))
    .map(([d, h]) => [d, h.hop, order(d, h.focus)]);
  const out = [];
  const maxR = Math.max(0, ...docs.map(([, , cs]) => cs.length));
  for (let r0 = 0; r0 < maxR && out.length < topK; r0++)
    for (const [doc, hop, cs] of docs) {
      if (r0 < cs.length) { out.push([cs[r0], 1 / hop - r0 * 1e-4]); if (out.length >= topK) break; }
    }
  return { hits: out, graphs: graphs.slice(0, 3), linked, route: r, intent };
}

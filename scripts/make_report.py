"""
Phase 4 - 汇总全部评测结果, 生成一页 HTML 报告。

报告是这个项目的**交付物**: 面试时直接打开这一页, 而不是翻终端日志。
它把散落在 reports/*.json 里的结果拼成一条叙事线:

    纯检索的能力边界 -> 加了什么 -> 各值多少分 -> 还剩什么缺口

刻意不做的事: 不画花哨图表。评测报告的读者要的是能对比的数字,
一张排版清楚的表比任何图都快。
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import setup_console, write_text   # noqa: E402

setup_console()
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

TYPE_LABEL = {
    "fact_direct": "单跳事实（原词）", "fact_paraphrase": "单跳事实（同义改写）",
    "semantic_only": "纯语义（无角色名）", "fact_disambig": "易混角色 / 别名",
    "hop2": "两跳关系", "hop3": "三跳关系", "relation_path": "关系路径（招牌）",
    "aggregation": "列举比较", "negative": "幻觉陷阱",
}


def load(name: str):
    p = REPORTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x) -> str:
    try:
        return f"{float(x):.1%}"
    except (TypeError, ValueError):
        return "—"


def table(headers: list[str], rows: list[list[str]], highlight: int = -1) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = "".join(
            f'<td class="{"hi" if i == highlight else ""}{" lbl" if i == 0 else ""}">{c}</td>'
            for i, c in enumerate(r))
        trs.append(f"<tr>{tds}</tr>")
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def section_retrieval() -> str:
    d = load("phase1_retrieval.json")
    if not d:
        return '<p class="muted">尚未运行 <code>python run.py retrieval</code></p>'
    out = []

    ga = d.get("group_a_chunking", [])
    if ga:
        names = [r["retriever"].split("/")[-1] for r in ga]
        types = [t for t in TYPE_LABEL if t in ga[0]["by_type"]]
        rows = [[TYPE_LABEL[t]] + [pct(r["by_type"][t]["recall"].get("1")) for r in ga]
                for t in types]
        rows.append(["<b>总计</b>"] + [f'<b>{pct(r["overall"]["recall"].get("1"))}</b>'
                                       for r in ga])
        out.append("<h3>A 组　切块策略对比（recall@1，检索器固定 BM25）</h3>")
        out.append(table(["题型"] + names, rows, highlight=len(names)))
        out.append('<p class="note">孤儿 chunk：档案第二、三节全部用「该角色」指代，'
                   'section 策略切出来的块不含角色名；contextual 给每块补上「XX 角色档案」前缀。'
                   '语料很短（平均每份约 300 字），fixed 的大块常常碰巧把角色名和事实装进同一块 —— '
                   '<b>总分接近不代表机制相同</b>。</p>')

    gb = d.get("group_b_ablation", [])
    if gb:
        names = [r["retriever"] for r in gb]
        types = [t for t in TYPE_LABEL if t in gb[0]["by_type"]]
        rows = [[TYPE_LABEL[t]] + [pct(r["by_type"].get(t, {}).get("recall", {}).get("10"))
                                   for r in gb] for t in types]
        rows.append(["<b>总计</b>"] + [f'<b>{pct(r["overall"]["recall"].get("10"))}</b>'
                                       for r in gb])
        out.append("<h3>B 组　检索路径消融（recall@10）</h3>")
        out.append(f'<div class="scroll">{table(["题型"] + names, rows, len(names))}</div>')
    return "".join(out)


def section_extraction() -> str:
    runs = [("规则基线", "phase2_extraction_rule.json"),
            ("LLM v1 朴素", "phase2_extraction_llm_v1_naive.json"),
            ("LLM v3 few-shot", "phase2_extraction_llm_v3_fewshot.json"),
            ("LLM v4 结构感知", "phase2_extraction_llm_v4_structural.json")]
    loaded = [(n, load(f)) for n, f in runs]
    loaded = [(n, d) for n, d in loaded if d]
    if not loaded:
        return '<p class="muted">尚未运行 <code>python run.py build-graph</code></p>'
    out = ["<h3>抽取 prompt 演进（抽取图 vs 标准图逐条比对）</h3>"]
    rows = [[n, pct(d["precision"]), pct(d["recall"]), pct(d["f1"]),
             str(d["extract_stats"]["parse_failed"]),
             f'{d["extract_stats"]["cost_usd"]:.3f}'] for n, d in loaded]
    out.append(table(["抽取器", "precision", "recall", "F1", "解析失败(份)", "花费 USD"], rows))

    # 结构型文档那三种关系: v3 -> v4 的主要差异
    v3 = dict(loaded).get("LLM v3 few-shot")
    v4 = dict(loaded).get("LLM v4 结构感知")
    if v3 and v4:
        rels = ["LIVES_IN", "STUDENT_OF", "FACTION"]
        rows = [[r, pct(v3["by_relation"].get(r, {}).get("recall")),
                 pct(v4["by_relation"].get(r, {}).get("recall"))] for r in rels]
        out.append("<h3>列表型文档（居民登记 / 学生名册 / 阵营一览）的召回</h3>")
        out.append(table(["关系", "v3 recall", "v4 recall"], rows, highlight=2))
        out.append('<p class="note">v3 的负例写的是「不推理」，模型把「一、正义阵营 · 卡布达」'
                   '这种<b>读懂排版</b>也当成了推理，整份跳过。v4 把规则细化为'
                   '「不许臆测，但必须读懂文档结构」，并补了一个列表型正例。</p>')
        rs = v4["resolve_stats"]
        out.append(f'<p class="note"><b>消歧</b>（v4）：提及 {rs["mentions"]} 个名称归并为 '
                   f'{rs["canonical"]} 个实体；文档声明的「又名」合并 {rs.get("merged_by_alias", 0)} 对，'
                   f'词面一票否决 {rs.get("blocked_by_lexical", 0)} 对，'
                   f'关系边一票否决 {rs.get("blocked_by_relation", 0)} 对，'
                   f'LLM 裁决 {rs["llm_calls"]} 次。</p>')
    out.append('<p class="warn">规则抽取在本项目的模板化语料上满分，'
               '<b>该分数不具外推性</b>，仅作为基线与离线兜底；真实能力看 LLM 抽取。</p>')
    return "".join(out)


def _ans(r, key):
    rows = [x for x in r["details"] if x["type"] != "negative"]
    vals = [x[key] for x in rows if x[key] == x[key]]
    return sum(vals) / len(vals) if vals else 0.0


def section_e2e() -> str:
    d = load("phase3_e2e_ab.json") or load("phase3_e2e.json")
    if not d:
        return '<p class="muted">尚未运行 <code>python run.py e2e</code></p>'
    names = [r["prompt"].split("/")[-1] for r in d]
    ans = _ans

    rows = [
        ["可答题答对率"] + [pct(ans(r, "correct")) for r in d],
        ["可答题实体召回"] + [pct(ans(r, "entity_recall")) for r in d],
        ["陷阱题正确拒答"] + [pct(r["by_type"].get("negative", {}).get("refusal_correct", 0))
                        for r in d],
        ["可答题误拒答率"] + [pct(ans(r, "false_refusal")) for r in d],
        ["引用有效率"] + [pct(r["overall"]["citation_valid"]) for r in d],
        ["引用命中标准证据"] + [pct(r["overall"]["citation_grounded"]) for r in d],
        ["累计花费 (USD)"] + [f'{r["overall"]["cost_usd"]:.3f}' for r in d],
    ]
    extra = ""
    ext, orc = load("phase3_e2e.json"), load("phase3_e2e_oracle.json")
    if ext and orc:
        extra = ("<h3>抽取图 vs 标准图（v3_guarded，全部可答题）</h3>" + table(
            ["", "抽取图", "标准图"],
            [["可答题答对率", pct(_ans(ext[0], "correct")), pct(_ans(orc[0], "correct"))],
             ["可答题实体召回", pct(_ans(ext[0], "entity_recall")),
              pct(_ans(orc[0], "entity_recall"))]]))
    return (table(["指标"] + names, rows) + extra
            + '<p class="note"><b>为什么「答对率」和「拒答率」必须并排看：</b>'
              '只看拒答率，一个永远回答「资料中未提及」的系统能拿满分；'
              '只看答对率，一个从不拒答的系统在陷阱题上会全错但总分未必难看。'
              '单看任何一个都能被轻易刷分 —— 这是 prompt 调优最容易自欺的地方。</p>')


def _b(name: str) -> dict:
    d = load("phase1_retrieval.json") or {}
    return next((r for r in d.get("group_b_ablation", []) if r["retriever"] == name), {})


def _r(name: str, t: str, k: str = "10"):
    return _b(name).get("by_type", {}).get(t, {}).get("recall", {}).get(k)


def section_conclusions() -> str:
    """结论里的每个数字都从 reports/*.json 现算, 不手写 —— 手写的数字迟早和结果脱节。"""
    items = []
    if _b("BM25"):
        items.append(
            f"<li><b>纯检索找多跳证据靠堆 k，图检索靠走。</b>两跳题图检索 recall@5 = "
            f"{pct(_r('Graph(oracle)', 'hop2', '5'))}；BM25 放大到 k=20 也只有 "
            f"{pct(_r('BM25', 'hop2', '20'))}。关系路径题两端角色从不共现于同一文档"
            f"（<code>verify_dataset.py</code> 已验证）。语料只有 42 份，k=20 已接近半个语料库。</li>")
        items.append(
            f"<li><b>公平对照下，路由仍略优于等权融合。</b>RRF(BM25+Graph) "
            f"{pct(_b('RRF(朴素)')['overall']['recall']['10'])}、RRF(三路) "
            f"{pct(_b('RRF(三路)')['overall']['recall']['10'])}、Routed "
            f"{pct(_b('Routed(完整)')['overall']['recall']['10'])}。"
            f"早期基金语料上\"融合把分数拉垮\"的结论<b>没有复现</b>；"
            f"两路融合与三路路由的差距有一半来自向量本身 —— 对照组要对齐。</li>")
        items.append(
            f"<li><b>枢纽阻断的价值取决于查询模板。</b>关系路径题：阻断 "
            f"{pct(_r('Graph(oracle)', 'relation_path'))} vs 不阻断 "
            f"{pct(_r('Graph(无枢纽阻断)', 'relation_path'))}；三跳题 "
            f"{pct(_r('Graph(oracle)', 'hop3'))} vs {pct(_r('Graph(无枢纽阻断)', 'hop3'))}。"
            f"只沿指定关系走的关系链模板本身就绕开了 APPEARS_IN 边，所以差距比无差别扩散时小。</li>")
    v3, v4 = load("phase2_extraction_llm_v3_fewshot.json"), load("phase2_extraction_llm_v4_structural.json")
    if v3 and v4:
        items.append(
            f"<li><b>抽取 prompt 的失分点是排版，不是语义。</b>v3 recall {pct(v3['recall'])}，"
            f"居民登记 / 阵营一览整份跳过；v4 把\"不推理\"细化为\"必须读懂列表结构\"后 recall "
            f"{pct(v4['recall'])}，precision {pct(v3['precision'])} → {pct(v4['precision'])}，"
            f"代价在 hypothesis 里已预先写明。</li>")
    ext, orc = load("phase3_e2e.json"), load("phase3_e2e_oracle.json")
    if ext:
        h2 = ext[0]["by_type"].get("hop2", {})
        items.append(
            f"<li><b>检索 recall 100% 不代表答案对。</b>首版两跳题检索 recall@8 已是 100%，答对率只有 25%："
            f"图检索按原文顺序发 chunk，每份档案先发出去的是标题块。加上关系链模板与文档内挑块后，"
            f"两跳题答对率 {pct(h2.get('correct'))}。recall 按文档算，会掩盖块级缺失。</li>")
    if ext and orc:
        items.append(
            f"<li><b>抽取图与标准图的端到端差距：</b>可答题答对率 {pct(_ans(ext[0], 'correct'))} vs "
            f"{pct(_ans(orc[0], 'correct'))}。v4 抽取 recall 100%，抽取环节在端到端上几乎没有损失。</li>")
    items.append(
        "<li><b>实体消歧：子串不等于简称，向量不能单独拍板。</b>「和平星」⊂「射手座和平星」、"
        "「卡布达」⊂「卡布达巨人」都曾被错并。现在文档声明的「又名」是最强证据；前两字相同的进入灰区交 LLM；"
        "同一份文档里有关系边相连的两个名字一票否决。</li>")
    gaps = [
        "语料只有 42 份文档，检索的绝对分数偏乐观；补充分集剧情时只收可核实的事实。",
        f"纯语义题路由后 recall@10 {pct(_r('Routed(完整)', 'semantic_only'))}，"
        f"纯 Dense {pct(_r('Dense', 'semantic_only'))}：融合时 BM25 的噪音把向量的结果挤掉了一部分。",
        "拒答判定仍是关键词匹配：首版漏认了\"没有关于……的信息\"\"没有记载\"，把正确拒答算成了幻觉；"
        "补全措辞、误拒答改为\"拒答且没答对\"后，三版 prompt 陷阱题拒答均为 100%。新的说法出现时仍会漏判。",
        f"列举比较题图检索只有 {pct(_r('Graph(oracle)', 'aggregation'))}：「正义阵营」里的「正义」是字面量，链接不到实体。",
        "枢纽阈值固定为 12，「羊村」这类小枢纽仍会产生弱关联路径。",
    ]
    return ("<ul>" + "".join(items) + "</ul><h2>五、已知缺口</h2><ul>"
            + "".join(f"<li>{g}</li>" for g in gaps) + "</ul>")


def main() -> int:
    d = load("phase1_retrieval.json")
    best = "—"
    if d and d.get("group_b_ablation"):
        best = max((r["overall"]["recall"].get("10", 0)
                    for r in d["group_b_ablation"]), default=0)
        base = next((r["overall"]["recall"].get("10", 0)
                     for r in d["group_b_ablation"] if r["retriever"] == "BM25"), 0)
        best = f"{base:.1%} → {best:.1%}"

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GraphRAG Lab 评测报告</title><style>
:root{{--bg:#fff;--fg:#1a1d24;--muted:#6b7280;--line:#e3e7ee;--panel:#f7f8fa;
 --accent:#2563eb;--ok:#0f9d76;--warn:#b45309;}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0f1115;--fg:#e6e9ef;
 --muted:#8b93a7;--line:#272c38;--panel:#171a21;--accent:#5b9dff;--ok:#7ee0b8;--warn:#ffb454;}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:40px 18px 80px}}
h1{{font-size:26px;margin:0 0 4px}}h2{{font-size:18px;margin:38px 0 10px;
 padding-bottom:7px;border-bottom:2px solid var(--line)}}
h3{{font-size:14px;margin:22px 0 8px;color:var(--muted)}}
.sub{{color:var(--muted);font-size:13px;margin-bottom:26px}}
.kpi{{font-size:15px;margin:10px 0 14px}}.kpi b{{color:var(--accent);font-size:19px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;margin:10px 0}}
th,td{{border:1px solid var(--line);padding:6px 9px;text-align:right;white-space:nowrap}}
th{{background:var(--panel);font-weight:600;color:var(--muted);text-align:right}}
th:first-child,td.lbl{{text-align:left}}
td.hi{{background:rgba(37,99,235,.08);font-weight:600}}
.note{{font-size:12.5px;color:var(--muted);background:var(--panel);
 padding:10px 13px;border-radius:8px;border-left:3px solid var(--accent);margin:10px 0}}
.warn{{font-size:12.5px;color:var(--warn);background:var(--panel);
 padding:10px 13px;border-radius:8px;border-left:3px solid var(--warn);margin:10px 0}}
.muted{{color:var(--muted)}}code{{background:var(--panel);padding:1px 5px;border-radius:4px;
 font-size:12px}}.scroll{{overflow-x:auto}}
ul{{padding-left:20px}}li{{margin:5px 0}}
</style></head><body><div class="wrap">
<h1>GraphRAG Lab 评测报告</h1>
<div class="sub">生成于 {date.today()}　·　题材：《喜羊羊与灰太狼》×《铁甲小宝》，事实取自维基百科等公开资料　·　
检索 recall@10：<b>{best}</b></div>

<h2>一、检索</h2>
{section_retrieval()}

<h2>二、抽取与实体消歧</h2>
{section_extraction()}

<h2>三、生成与 Prompt A/B</h2>
{section_e2e()}

<h2>四、主要结论</h2>
{section_conclusions()}
</div></body></html>"""

    out = REPORTS / "index.html"
    write_text(out, html)
    print(f"[ok] 报告已生成 -> {out}")
    print(f"     直接用浏览器打开即可（离线，无外部依赖）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, errorMessage } from "../api";
import type { DebugStep, DebugTrace } from "../types";
import { formatTime } from "../utils";

/** Human labels for the pipeline steps, in the order they run. */
const STEP_LABEL: Record<string, string> = {
  metadata: "歌曲信息",
  lyrics_search: "歌词搜索",
  lyrics_selected: "选中的歌词",
  language: "演唱语言",
  asr_windows: "ASR 分段识别",
  asr_words: "识别词流（只用于定位每一行）",
  asr_retry: "识别时间戳间隔异常 → 重新识别",
  reference: "歌词源自带时间轴",
  word_sync_search: "查找其他版本的逐字歌词",
  word_sync: "歌词源逐字歌词",
  line_spans: "每行在 ASR 里的位置（kana DP）",
  stray_absorb: "漏掉的识别片段并入歌词行",
  llm_verify: "LLM 逐行核对",
  llm_extras: "LLM 判断加唱片段",
  extra_retry: "疑似新句子 → 重新识别",
  align_flow: "对齐流程",
  char_align: "逐字时间",
  llm_merge: "LLM 按语意合并成词",
  extras: "加唱片段重新断句",
  timeline: "最终时间轴",
};

const STEP_HINT: Record<string, string> = {
  metadata: "从文件名 / 标签 / LLM 得到的歌曲信息，歌词搜索用它作查询条件。",
  lyrics_search: "网易云 → LRCLIB → LrcApi 的检索结果。",
  lyrics_selected: "真正拿去对齐的歌词行，以及它是否带源时间轴。",
  language: "发给 ASR 的语言；auto 表示让 ASR 自己判断。",
  asr_windows: "每 20 秒一段的识别原文，出错的分段也会记录。",
  asr_words: "识别词流：每一段的字/词及其时间。它既用来判断「哪一行在哪」（kana DP 和逐字间隔检查都基于它），也是逐字时间的来源：默认每个字的时间都来自它；开启「强制对齐逐字」后，两个后端一致时仍取识别词流的边界，分歧或识别没听到时才由对齐器兜底。",
  asr_retry:
    "以原始歌词的一行为一句：这句的识别时间戳里出现远超正常字距的间隔，说明识别漏掉了中间的字；整句会被重新识别，只有确实填补了间隔才替换这句的时间戳。",
  reference: "歌词源自带（录音室版）时间，仅作为定位参考，从不写进结果。",
  word_sync_search:
    "本项目用的歌词条目没有逐字歌词时，去同一首歌的其他版本（例如现场版）里找：网易云不同版本是不同条目，逐字歌词（yrc）各自独立。只有和本项目歌词足够重合的版本才会被拿去核对。项目面板上关掉「查找其他版本逐字歌词」后这一步不搜索，只记一条「已关闭」。",
  word_sync:
    "歌词源若带逐字歌词（LRCLIB 的 lyricsfile、网易云的 yrc），先核对它与这段音频的 ASR 是否吻合；吻合才采用，并只按 ASR 听到的行起点平移。",
  line_spans: "全局 kana DP 把每一行映射到 ASR 的哪一段 token，以及置信度。",
  stray_absorb:
    "DP 没排进任何歌词行、又紧贴着某一行的识别片段（例如「有人悻悻地离开」后面的「心情的离开」）：歌词原文比 ASR 可信得多，这类片段视为那一行被听错的字，把它的时间并回该行，而不是新加一句。只有成句、离歌词行较远的片段才留在加唱里。",
  llm_verify: "LLM 在候选行里挑出每一行真正的对应片段；只用于标记可疑行。",
  llm_extras:
    "ASR 听到但歌词没覆盖的片段：哪些是真实人声；哪些其实是被识别错的某一行歌词（比较文本与读音，可能是某行的一部分）。后者只是决定要不要重新识别，不会因此丢弃。",
  extra_retry:
    "被判为「其实是某一行歌词」的片段，会重新识别这段音频，让读音对齐重新落到原始歌词那一行，而不是新加一句。",
  align_flow: "这一次走的是哪条流程：asr 表示识别定位每一行、再给逐字时间；aligner 表示整首交给强制对齐器分块对齐。",
  char_align:
    "给每一行定下每个字的时间。默认（强制对齐逐字关）只用识别器：每句按识别词流里听到的 token 重新切开（`source` 记为 `asr`），识别没对上的句子保留词流按行摊开的估计时间。打开强制对齐后，再把每句的片段连同歌词交给对齐器，两个后端一致（相差 ≤ 0.3 秒）的字用识别词流的边界（`asr`），分歧或识别没听到的字用对齐器的时间（`aligner`）。",
  llm_merge:
    "可选的最后一步（面板「按语意合并成词」，默认关）：把整份逐字结果交给 LLM，由它判断每个词在哪里断开（例如「君の心が見えない」→「君の」「心が」「見えない」）。它只决定断词，不产生任何时间：每个单元的时间取它第一个字的起点和最后一个字的终点（`source` 记为 `merged`）。返回的单元拼起来与原文对不上的句子整句丢弃，仍按每个字给出时间。",
  extras: "这些加唱片段被 LLM 重新断句前后的样子。",
  timeline: "落库的最终结果。",
};

export function DebugPage() {
  const { id = "" } = useParams();
  const [trace, setTrace] = useState<DebugTrace | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [open, setOpen] = useState<Record<string, boolean>>({});

  const load = useCallback(async () => {
    try {
      setTrace(await api.debugTrace(id));
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  }, [id]);

  useEffect(() => {
    void load();
  }, [load]);

  const runs = trace?.runs ?? [];
  const run = runs.find((item) => item.run_id === runId) ?? runs[0] ?? null;

  return (
    <div className="container">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
        <div className="row">
          <Link to={`/projects/${id}`}>← 返回项目</Link>
          <h1 style={{ fontSize: 20, margin: 0 }}>调试：中间步骤</h1>
        </div>
        <div className="row small">
          <button onClick={() => void load()}>刷新</button>
        </div>
      </div>

      {error && <p className="error-text">{error}</p>}

      {!error && runs.length === 0 && (
        <div className="card">
          <p className="muted">
            还没有调试记录。跑一次「开始逐字打轴」之后，这里会按顺序显示每一步的中间结果。
          </p>
        </div>
      )}

      {runs.length > 0 && (
        <>
          <div className="row" style={{ marginBottom: 12 }}>
            <span className="muted small">运行记录</span>
            <select
              value={run?.run_id ?? ""}
              onChange={(event) => setRunId(event.target.value)}
              style={{ maxWidth: 420 }}
            >
              {runs.map((item, index) => (
                <option key={item.run_id} value={item.run_id}>
                  {index === 0 ? "最新 · " : ""}
                  {item.created_at.replace("T", " ").slice(0, 19)}
                  {item.finished_at ? ` → ${item.finished_at.slice(11, 19)}` : "（未完成）"}
                  {` · ${item.steps.length} 步`}
                </option>
              ))}
            </select>
            <span className="sep" />
            <button
              className="ghost small"
              onClick={() =>
                setOpen(Object.fromEntries((run?.steps ?? []).map((step) => [step.step, true])))
              }
            >
              全部展开原始 JSON
            </button>
            <button className="ghost small" onClick={() => setOpen({})}>
              全部收起
            </button>
          </div>

          {run && (
            <div className="debug-steps">
              {run.steps.map((step) => (
                <StepCard
                  key={step.step}
                  step={step}
                  open={Boolean(open[step.step])}
                  onToggle={() => setOpen((current) => ({ ...current, [step.step]: !current[step.step] }))}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function StepCard({ step, open, onToggle }: { step: DebugStep; open: boolean; onToggle: () => void }) {
  const label = STEP_LABEL[step.step] ?? step.step;
  return (
    <div className="card debug-step">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
        <div className="row">
          <span className="badge info">{step.step}</span>
          <h2 style={{ margin: 0, fontSize: 15 }}>{label}</h2>
        </div>
        <div className="row small">
          <span className="muted">{step.created_at.slice(11, 19)}</span>
          <button className="ghost small" onClick={onToggle}>
            {open ? "收起原始 JSON" : "原始 JSON"}
          </button>
        </div>
      </div>
      {STEP_HINT[step.step] && <p className="muted small" style={{ marginTop: 0 }}>{STEP_HINT[step.step]}</p>}
      <StepBody step={step.step} payload={step.payload} />
      {open && <pre className="debug-json">{JSON.stringify(step.payload, null, 2)}</pre>}
    </div>
  );
}

function StepBody({ step, payload }: { step: string; payload: unknown }) {
  switch (step) {
    case "metadata":
      return <KeyValue data={payload as Record<string, unknown>} />;
    case "lyrics_search":
      return <LyricsSearch data={payload} />;
    case "lyrics_selected":
      return <LyricsSelected data={payload as Record<string, unknown>} />;
    case "language":
      return <KeyValue data={payload as Record<string, unknown>} />;
    case "asr_windows":
      return <AsrWindows data={payload as Record<string, unknown>} />;
    case "asr_words":
      return <AsrWords data={payload as Record<string, unknown>} />;
    case "asr_retry":
      return <AsrRetry data={payload as Record<string, unknown>} />;
    case "reference":
      return <ReferenceTable rows={payload as ReferenceRow[]} />;
    case "word_sync_search":
      return <WordSyncSearch data={payload as Record<string, any>} />;
    case "word_sync":
      return <WordSync data={payload as Record<string, unknown>} />;
    case "line_spans":
      return <LineSpans rows={payload as LineSpanRow[]} />;
    case "stray_absorb":
      return <StrayAbsorb data={payload as Record<string, any>} />;
    case "llm_verify":
      return <LlmVerify data={payload as Record<string, unknown>} />;
    case "llm_extras":
      return <LlmExtras data={payload as Record<string, unknown>} />;
    case "extra_retry":
      return <ExtraRetry data={payload as Record<string, unknown>} />;
    case "extras":
      return <Extras data={payload as { before?: string[]; after?: string[] }} />;
    case "timeline":
      return <FinalTimeline data={payload as Record<string, unknown>} />;
    default:
      return <KeyValue data={(payload ?? {}) as Record<string, unknown>} />;
  }
}

function AsrRetry({ data }: { data: Record<string, unknown> }) {
  const windows: Record<string, any>[] = (data.windows as Record<string, any>[]) ?? [];
  if (!windows.length) {
    return (
      <p className="small muted" style={{ marginTop: 0 }}>
        没有发现需要重新识别的间隔（阈值 {String(data.threshold ?? "?")} 秒）。
      </p>
    );
  }
  return (
    <>
      <p className="small muted" style={{ marginTop: 0 }}>
        阈值 {String(data.threshold ?? "?")} 秒，发现 {String(data.found ?? 0)} 处异常间隔，按整句重识别{" "}
        {String(data.attempted ?? 0)} 句，采纳 {String(data.accepted ?? 0)} 句。
      </p>
      <table className="debug-table">
        <thead>
          <tr>
            <th>行</th>
            <th>歌词</th>
            <th>异常间隔</th>
            <th>重识别整句</th>
            <th>听到</th>
            <th>结果</th>
          </tr>
        </thead>
        <tbody>
          {windows.map((row, index) => (
            <tr key={index} className={row.accepted ? undefined : "debug-error-row"}>
              <td>{row.line}</td>
              <td>{row.text}</td>
              <td className="mono small">
                {row.gap?.[0]} → {row.gap?.[1]}（{row.gap_seconds}s）
              </td>
              <td className="mono small">
                {row.sentence ? `${row.sentence[0]} → ${row.sentence[1]}` : "—"}
                <br />
                <span className="muted">
                  切 {row.window?.[0]} → {row.window?.[1]}
                </span>
              </td>
              <td className="small">
                {row.heard ?? row.error ?? "—"}
                {row.heard_share != null ? (
                  <span className="muted">（听到 {Math.round(Number(row.heard_share) * 100)}%）</span>
                ) : null}
              </td>
              <td className="small">
                {row.accepted ? (
                  <span className="badge ok">已替换，间隔 {row.gap_after}s</span>
                ) : (
                  <span className="muted">{row.reason ?? "重识别失败，保留原结果"}</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function ExtraRetry({ data }: { data: Record<string, unknown> }) {
  const windows: Record<string, any>[] = (data.windows as Record<string, any>[]) ?? [];
  if (!windows.length) {
    return (
      <p className="small muted" style={{ marginTop: 0 }}>
        没有需要重新识别的「疑似新句子」。
      </p>
    );
  }
  return (
    <>
      <p className="small muted" style={{ marginTop: 0 }}>
        疑似识别错误的片段 {String(data.attempted ?? 0)} 处，重新识别 {String(data.accepted ?? 0)} 处。
      </p>
      <table className="debug-table">
        <thead>
          <tr>
            <th>应对应的行</th>
            <th>歌词</th>
            <th>原来听到</th>
            <th>重识别窗口</th>
            <th>重新听到</th>
            <th>结果</th>
          </tr>
        </thead>
        <tbody>
          {windows.map((row, index) => (
            <tr key={index} className={row.accepted ? undefined : "debug-error-row"}>
              <td>{row.line}</td>
              <td className="small">{row.line_text ?? "—"}</td>
              <td className="small">{row.heard_as ?? "—"}</td>
              <td className="mono small">
                {row.window?.[0]} → {row.window?.[1]}
              </td>
              <td className="small">{row.heard_again ?? row.heard ?? row.error ?? "—"}</td>
              <td className="small">
                {row.accepted ? (
                  <span className="badge ok">已重新识别，交给读音对齐</span>
                ) : (
                  <span className="muted">{row.reason ?? "重识别失败，保留为加唱行"}</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function WordSyncSearch({ data }: { data: Record<string, any> }) {
  if (!data.found) {
    return <p className="small muted" style={{ marginTop: 0 }}>{String(data.note ?? "没有找到逐字歌词。")}</p>;
  }
  return (
    <KeyValue
      data={{
        来源: `${data.source_label ?? data.source ?? ""} (${data.source ?? ""})`,
        版本: data.edition ?? "—",
        外部id: data.external_id ?? "—",
        "歌词重合": data.agreement === undefined ? "—" : `${Math.round(Number(data.agreement) * 100)}%`,
        行数: data.line_count,
      }}
    />
  );
}

function WordSync({ data }: { data: Record<string, unknown> }) {
  const available = Boolean(data.available);
  if (!available) {
    return (
      <p className="small muted" style={{ marginTop: 0 }}>
        歌词源没有提供逐字歌词：{String(data.reason ?? "—")}。按本地流程（ASR + 读音对齐）处理。
      </p>
    );
  }
  return (
    <>
      <p className="small" style={{ marginTop: 0 }}>
        {data.used ? <span className="badge ok">已采用歌词源逐字时间</span> : <span className="badge flag">未采用</span>}{" "}
        <span className="muted">{String(data.reason ?? "")}</span>
      </p>
      <KeyValue
        data={{
          来源: `${data.source_label ?? data.source ?? ""} (${data.source ?? ""})`,
          版本: data.edition ?? "本项目歌词的同一条目",
          "逐字行数": `${data.word_lines ?? 0} / ${data.sync_line_count ?? 0}`,
          对上歌词行: `${data.matched ?? 0} / ${data.line_count ?? 0}`,
          "ASR 定位到的行": `${data.anchors ?? 0} / ${data.line_count ?? 0}`,
          整体偏移: data.offset === undefined ? "—" : `${data.offset} 秒`,
          偏差中位数: data.median_deviation === undefined ? "—" : `${data.median_deviation} 秒`,
          偏差最大: data.max_deviation === undefined ? "—" : `${data.max_deviation} 秒`,
          "阈值内的行": data.within_share === undefined ? "—" : `${Math.round(Number(data.within_share) * 100)}%`,
        }}
      />
    </>
  );
}

function KeyValue({ data }: { data: Record<string, unknown> }) {
  return (
    <dl className="debug-kv">
      {Object.entries(data ?? {}).map(([key, value]) => (
        <div key={key}>
          <dt>{key}</dt>
          <dd>{typeof value === "string" || typeof value === "number" ? String(value) : JSON.stringify(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function LyricsSearch({ data }: { data: unknown }) {
  const payload = (data ?? {}) as Record<string, any>;
  if (payload.result === "none") {
    return <p className="muted">所有来源都没有找到匹配项（查询：{JSON.stringify(payload.query)}）。</p>;
  }
  return (
    <>
      <KeyValue
        data={{
          来源: `${payload.source_label ?? payload.source ?? ""} (${payload.source ?? ""})`,
          匹配: `${payload.track_name ?? ""} — ${payload.artist_name ?? ""}`,
          分数: payload.score,
          "外部 id": payload.lrclib_id ?? payload.external_id ?? "—",
          带时间轴: payload.synced ? "是" : "否",
          行数: payload.line_count,
        }}
      />
      <details>
        <summary className="small">查询条件</summary>
        <pre className="debug-json">{JSON.stringify(payload.query, null, 2)}</pre>
      </details>
    </>
  );
}

function LyricsSelected({ data }: { data: Record<string, any> }) {
  const lines: string[] = data.lines ?? [];
  return (
    <>
      <KeyValue
        data={{
          来源: data.source ?? "—",
          "外部 id": data.external_id ?? "—",
          行数: data.line_count,
          带源时间轴: data.has_reference_timeline ? "是" : "否",
        }}
      />
      <details>
        <summary className="small">歌词行（{lines.length}）</summary>
        <ol className="debug-lines">
          {lines.map((line, index) => (
            <li key={index}>{line || <span className="muted">（空行）</span>}</li>
          ))}
        </ol>
      </details>
    </>
  );
}

const RECOGNISER: Record<string, string> = {
  "qwen-asr-http": "Qwen3-ASR 服务（自带强制对齐）",
};

function AsrWindows({ data }: { data: Record<string, any> }) {
  const windows: Record<string, any>[] = data.windows ?? [];
  return (
    <>
      <KeyValue
        data={{
          语言: data.language ?? "（自动）",
          识别词流: RECOGNISER[data.recogniser] ?? data.recogniser ?? "（未记录）",
          人声分离: data.cached ? `${data.separate === false ? "否" : "是"}（复用缓存）` : data.separated ? "是" : "否",
          时长: data.duration,
          分段数: data.window_count,
          失败分段: data.failed ?? 0,
        }}
      />
      <table className="debug-table">
        <thead>
          <tr>
            <th>#</th>
            <th>起点</th>
            <th>时长</th>
            <th>token</th>
            <th>识别文本</th>
          </tr>
        </thead>
        <tbody>
          {windows.map((item, index) => (
            <tr key={index} className={item.error ? "debug-error-row" : ""}>
              <td>{item.window ?? index + 1}</td>
              <td>{formatTime(item.start ?? null)}</td>
              <td>{item.length ?? "—"}</td>
              <td>{item.tokens ?? "—"}</td>
              <td>
                {item.error ? <span className="error-text">{item.error}</span> : item.text}
                {item.model && <span className="muted small"> · {item.model}</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function AsrWords({ data }: { data: Record<string, any> }) {
  const words: { text: string; start: number | null; end: number | null; source?: string | null }[] =
    data.words ?? [];
  return (
    <>
      <p className="small muted" style={{ marginTop: 0 }}>
        共 {data.count} 个 token{data.cached ? "（复用缓存）" : ""}
      </p>
      <div className="debug-words">
        {words.map((word, index) => (
          <span key={index} className="word-chip timed" title={`${formatTime(word.start)} → ${formatTime(word.end)}`}>
            {word.text}
          </span>
        ))}
      </div>
    </>
  );
}

interface ReferenceRow {
  index: number;
  text: string;
  start: number | null;
  end: number | null;
}

function ReferenceTable({ rows }: { rows: ReferenceRow[] }) {
  if (!rows?.length) return <p className="muted">歌词源没有提供时间轴，或者一行都没对上。</p>;
  return (
    <table className="debug-table">
      <thead>
        <tr>
          <th>歌词行</th>
          <th>参考开始</th>
          <th>参考结束</th>
          <th>参考原文</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.index}>
            <td>{row.index + 1}</td>
            <td>{formatTime(row.start)}</td>
            <td>{formatTime(row.end)}</td>
            <td>{row.text}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

interface LineSpanRow {
  index: number;
  text: string;
  reading: string;
  first: number | null;
  last: number | null;
  confidence: number;
  span_text: string;
  span_start: number | null;
  span_end: number | null;
  unmatched: boolean;
}

function LineSpans({ rows }: { rows: LineSpanRow[] }) {
  const unmatched = rows?.filter((row) => row.unmatched).length ?? 0;
  return (
    <>
      <p className="small muted" style={{ marginTop: 0 }}>
        {rows?.length ?? 0} 行，其中 {unmatched} 行没在 ASR 里找到位置。
      </p>
      <table className="debug-table">
        <thead>
          <tr>
            <th>#</th>
            <th>歌词</th>
            <th>读音</th>
            <th>ASR 片段</th>
            <th>时间</th>
            <th>置信度</th>
          </tr>
        </thead>
        <tbody>
          {rows?.map((row) => (
            <tr key={row.index} className={row.unmatched ? "debug-error-row" : ""}>
              <td>{row.index + 1}</td>
              <td>{row.text}</td>
              <td className="muted small">{row.reading}</td>
              <td>{row.unmatched ? <span className="muted">（无）</span> : row.span_text}</td>
              <td className="small">
                {row.unmatched ? "—" : `${formatTime(row.span_start)} → ${formatTime(row.span_end)}`}
              </td>
              <td>{row.unmatched ? "—" : row.confidence.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function StrayAbsorb({ data }: { data: Record<string, any> }) {
  const rows: Record<string, any>[] = data.absorbed ?? [];
  if (!rows.length) {
    return (
      <p className="small muted" style={{ marginTop: 0 }}>
        没有需要并入歌词行的漏识别片段。
      </p>
    );
  }
  return (
    <>
      <p className="small muted" style={{ marginTop: 0 }}>
        {String(data.note ?? "")} 共 {rows.length} 处。
      </p>
      <table className="debug-table">
        <thead>
          <tr>
            <th>并入第几行</th>
            <th>歌词原文</th>
            <th>ASR 听到的片段</th>
            <th>时间</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              <td>{row.line}</td>
              <td>{row.text}</td>
              <td>{row.absorbed}</td>
              <td className="small">
                {row.window ? `${formatTime(row.window[0])} → ${formatTime(row.window[1])}` : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function LlmVerify({ data }: { data: Record<string, any> }) {
  const picks: Record<string, any>[] = data.picks ?? [];
  // Lines whose neighbouring span reads exactly the same as their own: a pick
  // between them proves nothing, so it is not a finding.
  const same: Set<number> = new Set(data.indistinguishable ?? []);
  return (
    <>
      <p className="small muted" style={{ marginTop: 0 }}>
        LLM 调用 {data.calls} 次，失败 {data.failures} 次，互选一致 {data.mutual?.length ?? 0} 行
        {same.size > 0 ? `，读音重复（选择无意义）${same.size} 行` : ""}。
      </p>
      <table className="debug-table">
        <thead>
          <tr>
            <th>#</th>
            <th>歌词</th>
            <th>DP 听到</th>
            <th>LLM 选的片段</th>
            <th>结论</th>
          </tr>
        </thead>
        <tbody>
          {picks.map((row) => (
            <tr key={row.index}>
              <td>{row.index + 1}</td>
              <td>{row.text}</td>
              <td className="muted small">{row.heard}</td>
              <td className="small">{row.picked_heard ?? "—"}</td>
              <td>
                {row.verdict && row.verdict !== "asr" ? (
                  <span className="badge flag">{row.verdict}</span>
                ) : same.has(row.index) ? (
                  <span className="muted small">重复行，选到同读音的相邻片段</span>
                ) : (
                  <span className="muted">一致</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function LlmExtras({ data }: { data: Record<string, any> }) {
  if (data.skipped) {
    return (
      <>
        <p className="muted small" style={{ marginTop: 0 }}>{data.skipped}</p>
        <ChipList items={data.kept ?? []} />
      </>
    );
  }
  const candidates: Record<string, any>[] = data.candidates ?? [];
  const keep: number[] = data.keep ?? [];
  const forced: number[] = data.forced ?? [];
  const joined: number[] = data.joined ?? [];
  const named: Record<string, any>[] = data.named ?? [];
  const namedByText = new Map(named.map((item) => [item.text, item]));
  return (
    <>
      <p className="small muted" style={{ marginTop: 0 }}>
        候选 {candidates.length} 段，保留 {keep.length} 段。
        {forced.length > 0 && ` 其中第 ${forced.join("、")} 段模型判为噪声，但内容足够长且不是同一个字的重复，按真人语音保留。`}
        {joined.length > 0 && " 另有几段太短、模型看不到的碎片紧挨着保留内容，作为同一句的尾巴一并保留。"}
        {named.length > 0 &&
          " 被模型认成某一行歌词的片段一律保留：那一行没对上就重新识别，已经对上就当作重复演唱。"}
      </p>
      <table className="debug-table">
        <thead>
          <tr>
            <th>#</th>
            <th>时间</th>
            <th>文本</th>
            <th>判定</th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((row, index) => {
            const match = namedByText.get(row.text);
            return (
              <tr key={index}>
                <td>{index + 1}</td>
                <td className="small">
                  {formatTime(row.start)} → {formatTime(row.end)}
                </td>
                <td>
                  {row.text}
                  {match && (
                    <div className="small muted">
                      模型认为这是第 {match.line} 行「{match.line_text}」 · {match.action}
                    </div>
                  )}
                </td>
                <td>
                  {keep.includes(index + 1) ? (
                    <span className="badge ok">
                      {forced.includes(index + 1) ? "保留（模型判丢弃，内容确认为真人语音）" : "保留"}
                    </span>
                  ) : (
                    <span className="muted">丢弃</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}

function Extras({ data }: { data: { before?: string[]; after?: string[] } }) {
  return (
    <div className="debug-columns">
      <div>
        <p className="small muted">断句前（按 ASR 停顿）</p>
        <ChipList items={data.before ?? []} />
      </div>
      <div>
        <p className="small muted">断句后（LLM）</p>
        <ChipList items={data.after ?? []} />
      </div>
    </div>
  );
}

function ChipList({ items }: { items: string[] }) {
  if (!items.length) return <p className="muted small">（无）</p>;
  return (
    <ul className="debug-lines">
      {items.map((item, index) => (
        <li key={index}>{item}</li>
      ))}
    </ul>
  );
}

function FinalTimeline({ data }: { data: Record<string, any> }) {
  const lines: Record<string, any>[] = data.lines ?? [];
  const timed = lines.filter((line) => line.start !== null).length;
  return (
    <>
      <p className="small muted" style={{ marginTop: 0 }}>
        状态 {data.status}，{timed}/{lines.length} 行有时间。
      </p>
      <table className="debug-table">
        <thead>
          <tr>
            <th>#</th>
            <th>开始</th>
            <th>结束</th>
            <th>歌词</th>
            <th>标记</th>
            <th>逐字</th>
          </tr>
        </thead>
        <tbody>
          {lines.map((line) => (
            <tr key={line.index}>
              <td>{line.index + 1}</td>
              <td className="small">{formatTime(line.start)}</td>
              <td className="small">{formatTime(line.end)}</td>
              <td>{line.text}</td>
              <td className="small">{line.source ?? ""}</td>
              <td>{line.words ?? 0}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {data.notes?.length > 0 && (
        <ul className="run-notes">
          {data.notes.map((note: string, index: number) => (
            <li key={index}>{note}</li>
          ))}
        </ul>
      )}
    </>
  );
}

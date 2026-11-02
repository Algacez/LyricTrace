import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, errorMessage } from "../api";
import type { Project, Task } from "../types";
import { STATUS_LABEL } from "../utils";
import { AlignPanel } from "./AlignPanel";
import { TimelineEditor } from "./TimelineEditor";

export function ProjectPage() {
  const { id = "" } = useParams();
  const [project, setProject] = useState<Project | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<{ message: string; kind: "info" | "error" } | null>(null);
  // Bumped when a background run finishes, so the editor adopts the new timeline.
  const [timelineKey, setTimelineKey] = useState(0);
  const toastTimer = useRef<number | null>(null);
  const settledTask = useRef<string | null>(null);

  const notify = useCallback((message: string, kind: "info" | "error" = "info") => {
    setToast({ message, kind });
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), 4000);
  }, []);

  const load = useCallback(async () => {
    try {
      const data = await api.getProject(id);
      setProject(data);
      setError(null);
      return data;
    } catch (err) {
      setError(errorMessage(err));
      return null;
    }
  }, [id]);

  const loadTasks = useCallback(async () => {
    try {
      setTasks(await api.tasks(id));
    } catch {
      /* ignore */
    }
  }, [id]);

  useEffect(() => {
    void load();
    void loadTasks();
  }, [load, loadTasks]);

  const latest = tasks[0] ?? null;
  const busy = latest?.status === "queued" || latest?.status === "running";

  useEffect(() => {
    if (!busy) return;
    const timer = window.setInterval(() => {
      void load();
      void loadTasks();
    }, 1200);
    return () => window.clearInterval(timer);
  }, [busy, load, loadTasks]);

  // When a run settles, pull the fresh timeline into the editor exactly once.
  useEffect(() => {
    if (!latest || (latest.status !== "succeeded" && latest.status !== "failed")) return;
    if (settledTask.current === latest.id) return;
    settledTask.current = latest.id;
    void load().then(() => setTimelineKey((value) => value + 1));
  }, [latest, load]);

  const run = useCallback(
    async (resetCache: boolean) => {
      try {
        await api.align(id, resetCache);
        await loadTasks();
        await load();
      } catch (err) {
        notify(errorMessage(err), "error");
      }
    },
    [id, load, loadTasks, notify],
  );

  if (error) {
    return (
      <div className="container">
        <p className="error-text">{error}</p>
        <Link to="/">← 返回项目列表</Link>
      </div>
    );
  }

  if (!project) return <div className="container muted">加载中…</div>;

  return (
    <div className="container">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
        <div className="row">
          <Link to="/">← 项目列表</Link>
          <h1 style={{ fontSize: 20, margin: 0 }}>{project.title || project.original_filename}</h1>
          <span className={`badge ${project.status === "failed" ? "err" : project.status === "ready" ? "ok" : "info"}`}>
            {STATUS_LABEL[project.timeline_status]}
          </span>
        </div>
        <div className="row">
          <Link to={`/projects/${project.id}/debug`}>
            <button className="ghost" title="查看这次对齐的每一步中间结果">
              调试
            </button>
          </Link>
          {(["json", "lrc", "ass"] as const).map((format) => (
            <a key={format} href={api.exportUrl(project.id, format)} download>
              <button disabled={!project.timeline || project.timeline.lines.length === 0}>
                导出 {format.toUpperCase()}
              </button>
            </a>
          ))}
        </div>
      </div>

      <div className="editor-grid">
        <TimelineEditor
          key={`${project.id}-${timelineKey}`}
          project={project}
          onProjectChange={setProject}
        />
        <AlignPanel
          project={project}
          task={latest}
          busy={busy}
          onProjectChange={setProject}
          onRun={run}
        />
      </div>

      {toast && <div className={`toast ${toast.kind === "error" ? "error" : ""}`}>{toast.message}</div>}
    </div>
  );
}

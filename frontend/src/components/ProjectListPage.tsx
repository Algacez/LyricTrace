import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, errorMessage } from "../api";
import type { ProjectSummary } from "../types";
import { STATUS_LABEL, formatTime } from "../utils";

/** How many project cards a page holds. Every list in the app pages. */
const PAGE_SIZE = 12;

export function ProjectListPage() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  // Which project cards are ticked for a batch delete.
  const [checked, setChecked] = useState<string[]>([]);
  const navigate = useNavigate();

  const load = useCallback(async () => {
    try {
      const data = await api.listProjects({
        q: query || undefined,
        limit: PAGE_SIZE,
        offset,
      });
      setProjects(data.items);
      setTotal(data.total);
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, [query, offset]);

  useEffect(() => {
    void load();
  }, [load]);

  // A new search starts at the first page again.
  useEffect(() => {
    setOffset(0);
  }, [query]);

  // Poll while any project is still being processed.
  useEffect(() => {
    const pending = projects.some((p) => p.status === "queued" || p.status === "processing");
    if (!pending) return;
    const timer = window.setInterval(() => void load(), 1500);
    return () => window.clearInterval(timer);
  }, [projects, load]);

  const upload = async (files: FileList | File[]) => {
    const file = Array.from(files)[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const project = await api.upload(file);
      navigate(`/projects/${project.id}`);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setUploading(false);
    }
  };

  const remove = async (id: string) => {
    if (!window.confirm("删除该项目及其音频文件？此操作不可撤销。")) return;
    try {
      await api.deleteProject(id);
      setProjects((current) => current.filter((item) => item.id !== id));
      setTotal((current) => Math.max(current - 1, 0));
      setChecked((current) => current.filter((item) => item !== id));
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const removeChecked = async () => {
    if (checked.length === 0) return;
    if (!window.confirm(`删除选中的 ${checked.length} 个项目？音频文件和工作目录都会删掉，此操作不可撤销。`)) return;
    try {
      await api.deleteProjects(checked);
      setChecked([]);
      await load();
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const page = Math.floor(offset / PAGE_SIZE) + 1;
  const allChecked = projects.length > 0 && checked.length === projects.length;

  const toggle = (id: string) =>
    setChecked((current) => (current.includes(id) ? current.filter((item) => item !== id) : [...current, id]));

  return (
    <div className="container">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 16 }}>
        <h1 style={{ fontSize: 20, margin: 0 }}>LyricTrace</h1>
        <input
          type="search"
          placeholder="搜索歌名 / 歌手 / 文件名"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          style={{ maxWidth: 280 }}
        />
      </div>
      {total > PAGE_SIZE && (
        <div className="row small muted" style={{ gap: 8, marginBottom: 12 }}>
          <span>
            第 {page} / {pages} 页 · 共 {total} 个项目
          </span>
          <button disabled={page <= 1} onClick={() => setOffset(Math.max(offset - PAGE_SIZE, 0))}>
            上一页
          </button>
          <button
            disabled={page >= pages}
            onClick={() => setOffset(Math.min(offset + PAGE_SIZE, (pages - 1) * PAGE_SIZE))}
          >
            下一页
          </button>
        </div>
      )}

      {/* Batch delete: tick the cards, or tick the whole page, and everything
          each one owns goes with it (DB rows, audio, work/). */}
      {!loading && projects.length > 0 && (
        <div className="row small" style={{ gap: 10, marginBottom: 12 }}>
          <label className="row small muted" style={{ gap: 4 }}>
            <input
              type="checkbox"
              checked={allChecked}
              onChange={(event) => setChecked(event.target.checked ? projects.map((item) => item.id) : [])}
            />
            全选本页
          </label>
          <button className="danger ghost" disabled={checked.length === 0} onClick={() => void removeChecked()}>
            删除选中{checked.length > 0 ? `（${checked.length}）` : ""}
          </button>
        </div>
      )}

      <div
        className={`upload-zone ${dragging ? "dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          void upload(event.dataTransfer.files);
        }}
        onClick={() => document.getElementById("file-input")?.click()}
      >
        {uploading ? (
          <span>正在上传…</span>
        ) : (
          <>
            <div style={{ fontSize: 16, marginBottom: 4 }}>拖入音频文件，或点击选择</div>
            <div className="small">支持 mp3 / flac / wav / m4a / aac / ogg / opus / webm</div>
          </>
        )}
        <input
          id="file-input"
          type="file"
          accept="audio/*,.flac,.m4a,.opus,.webm"
          hidden
          onChange={(event) => {
            if (event.target.files) void upload(event.target.files);
            event.target.value = "";
          }}
        />
      </div>

      {error && <p className="error-text">{error}</p>}

      {loading ? (
        <p className="muted">加载中…</p>
      ) : projects.length === 0 ? (
        <div className="empty-state">还没有项目。上传一个音频文件，粘贴歌词后即可自动对齐。</div>
      ) : (
        <div className="project-grid">
          {projects.map((project) => (
            <div key={project.id} className={`project-card ${checked.includes(project.id) ? "selected" : ""}`}>
              <div className="head">
                <input
                  type="checkbox"
                  className="pick"
                  checked={checked.includes(project.id)}
                  onChange={() => toggle(project.id)}
                  title="选中后批量删除"
                />
                <Link to={`/projects/${project.id}`} className="title">
                  {project.title || project.metadata.title || "（未命名）"}
                </Link>
              </div>
              <div className="muted small">
                {project.artist || "未知歌手"} · {formatTime(project.duration, false)}
              </div>
              <div className="row small" style={{ gap: 6 }}>
                <span className={`badge ${project.timeline_status === "complete" ? "ok" : "warn"}`}>
                  {STATUS_LABEL[project.timeline_status]}
                </span>
                {(project.status === "queued" || project.status === "processing") && (
                  <span className="badge info">处理中</span>
                )}
              </div>
              {(project.status === "queued" || project.status === "processing") && (
                <div className="progress">
                  <div style={{ width: `${Math.round(project.progress * 100)}%` }} />
                </div>
              )}
              {project.error && <div className="error-text small">{project.error}</div>}
              <div className="actions">
                <Link to={`/projects/${project.id}`}>
                  <button className="primary">打开</button>
                </Link>
                <button className="danger ghost" onClick={() => void remove(project.id)}>
                  删除
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

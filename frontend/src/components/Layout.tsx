import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { applyTheme, getTheme, onThemeChange, type Theme } from "../theme";

const THEME_LABEL: Record<Theme, string> = { light: "日间模式", dark: "夜间模式" };

export function Layout() {
  const [theme, setTheme] = useState<Theme>(getTheme);

  useEffect(() => onThemeChange(setTheme), []);

  // The toggle advertises where it takes you, not where you are.
  const target: Theme = theme === "light" ? "dark" : "light";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          Lyric<span>Trace</span>
        </div>
        <nav>
          <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
            项目
          </NavLink>
          <NavLink to="/cache" className={({ isActive }) => (isActive ? "active" : "")}>
            API 缓存
          </NavLink>
          <a href="/docs" target="_blank" rel="noreferrer">
            API 文档
          </a>
        </nav>
        <div className="spacer" />
        <span className="muted small">逐字歌词同步</span>
        <button
          className="theme-toggle"
          type="button"
          onClick={() => applyTheme(target)}
          title={`切换到${THEME_LABEL[target]}`}
          aria-label={`切换到${THEME_LABEL[target]}`}
        >
          {target === "dark" ? <MoonIcon /> : <SunIcon />}
          <span>{THEME_LABEL[target]}</span>
        </button>
      </header>
      <Outlet />
    </div>
  );
}

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <circle cx="12" cy="12" r="4.2" />
      <path d="M12 2.5v2.2M12 19.3v2.2M4.2 4.2l1.6 1.6M18.2 18.2l1.6 1.6M2.5 12h2.2M19.3 12h2.2M4.2 19.8l1.6-1.6M18.2 5.8l1.6-1.6" strokeLinecap="round" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path
        d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

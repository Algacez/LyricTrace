import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { Layout } from "./components/Layout";
import { ApiCachePage } from "./components/ApiCachePage";
import { DebugPage } from "./components/DebugPage";
import { ProjectListPage } from "./components/ProjectListPage";
import { ProjectPage } from "./components/ProjectPage";
import { applyTheme, getTheme } from "./theme";
import "./styles.css";

applyTheme(getTheme());

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <ProjectListPage /> },
      { path: "projects/:id", element: <ProjectPage /> },
      { path: "projects/:id/debug", element: <DebugPage /> },
      { path: "cache", element: <ApiCachePage /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);

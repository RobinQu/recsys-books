import { execFile } from "node:child_process";
import { createServer } from "node:http";

const projectRoot = "/home/coder/project";
const allowedTargets = Object.freeze({
  data: `${projectRoot}/recsys_lab/data.py`,
  classic: `${projectRoot}/recsys_lab/experiments.py`,
  industrial: `${projectRoot}/recsys_lab/industrial_experiments.py`,
  notebook_generator: `${projectRoot}/scripts/generate_notebooks.py`,
});

function json(response, status, body) {
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });
  response.end(JSON.stringify(body));
}

createServer((request, response) => {
  const url = new URL(request.url ?? "/", "http://127.0.0.1:8081");
  if (url.pathname === "/healthz") {
    json(response, 200, { status: "ok" });
    return;
  }
  if (url.pathname !== "/open") {
    json(response, 404, { detail: "not found" });
    return;
  }

  const target = url.searchParams.get("target") ?? "";
  const file = allowedTargets[target];
  if (!file) {
    json(response, 400, { detail: "unknown source target" });
    return;
  }

  // execFile receives a fixed executable and a whitelist-derived argument. No
  // shell is involved, so a request cannot turn the helper into command input.
  execFile("code-server", ["--reuse-window", file], { timeout: 8000 }, (error) => {
    if (error) {
      json(response, 503, { detail: "code-server is not ready", error: error.message });
      return;
    }
    json(response, 200, { status: "opened", file });
  });
}).listen(8081, "0.0.0.0");

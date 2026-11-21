/*
 * Frontend configuration (M11.1).
 *
 * This is the ONLY place the API location is set. To point the UI at a
 * different backend, edit API_BASE_URL here; nothing else needs to change.
 *
 * The backend's CORS allow-list (src/api/main.py) must include the origin
 * this frontend is served from. The default allow-list covers:
 *   http://127.0.0.1:5173  and  http://localhost:5173
 * i.e. the origins used by:
 *   python -m http.server 5173 --directory frontend
 */
window.CAMPUS_VERIFY_CONFIG = {
  API_BASE_URL: "http://127.0.0.1:8000",
  // First NLI request can be slow (model load / CPU inference), and URL
  // mode also fetches a remote page, so the timeout is deliberately generous.
  REQUEST_TIMEOUT_MS: 120000,
};

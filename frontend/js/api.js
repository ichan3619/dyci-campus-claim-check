/*
 * Thin HTTP client for the Campus Verify API (M11.1).
 *
 * Responsibility: make the three backend calls and turn failures into a
 * small, predictable ApiError. It contains NO verification logic, NO
 * verdict logic and NO evidence interpretation -- the backend response is
 * returned exactly as received.
 *
 * Calls made (and the only ones):
 *   GET  /health
 *   POST /verify/text   { claim }
 *   POST /verify/url    { claim, url }
 */
(function (root) {
  "use strict";

  class ApiError extends Error {
    /**
     * kind:
     *   "network"    - the API could not be reached at all
     *   "timeout"    - no response within REQUEST_TIMEOUT_MS
     *   "validation" - HTTP 422, request body rejected by the API
     *   "rejected"   - HTTP 400, request refused by policy/validation
     *                  (e.g. SSRF guard on /verify/url)
     *   "upstream"   - HTTP 502, the submitted URL could not be fetched or
     *                  produced no usable content. NOT a verdict.
     *   "rate_limited" - HTTP 429, too many requests from this browser; the
     *                  message says when to try again
     *   "busy"       - HTTP 503, every checking slot is in use; try again shortly
     *   "server"     - HTTP 5xx other than 502 and 503
     *   (HTTP 401 / 411 / 413 are reported as "rejected")
     *   "http"       - any other non-2xx status
     */
    constructor(kind, message, status) {
      super(message);
      this.name = "ApiError";
      this.kind = kind;
      this.status = status === undefined ? null : status;
    }
  }

  function config() {
    return root.CAMPUS_VERIFY_CONFIG || {};
  }

  function baseUrl() {
    return String(config().API_BASE_URL || "").replace(/\/+$/, "");
  }

  /**
   * FastAPI returns `detail` as a string for HTTPException (400/502) but as
   * an ARRAY of objects for request-validation failures (422). Normalize
   * both to one readable string.
   */
  function detailToMessage(detail, fallback) {
    if (typeof detail === "string" && detail.trim()) return detail;
    if (Array.isArray(detail) && detail.length) {
      const parts = detail.map(function (item) {
        if (!item || typeof item !== "object") return String(item);
        const loc = Array.isArray(item.loc)
          ? item.loc.filter(function (p) { return p !== "body"; }).join(".")
          : "";
        const msg = item.msg || "invalid value";
        return loc ? loc + ": " + msg : msg;
      });
      return parts.join("; ");
    }
    return fallback;
  }

  function kindForStatus(status) {
    if (status === 422 || status === 415) return "validation";
    if (status === 400 || status === 401 || status === 411 || status === 413) return "rejected";
    if (status === 429) return "rate_limited";
    if (status === 503) return "busy";
    if (status === 502) return "upstream";
    if (status >= 500) return "server";
    return "http";
  }

  async function request(path, options) {
    const controller = new AbortController();
    const timeoutMs = Number(config().REQUEST_TIMEOUT_MS) || 120000;
    const timer = setTimeout(function () { controller.abort(); }, timeoutMs);

    try {
      let response;
      try {
        response = await fetch(baseUrl() + path, Object.assign({}, options, { signal: controller.signal }));
      } catch (err) {
        if (err && err.name === "AbortError") {
          throw new ApiError("timeout", "The API did not respond within " + Math.round(timeoutMs / 1000) + " seconds.");
        }
        throw new ApiError("network", "Could not reach the API at " + baseUrl() + ".");
      }

      let body = null;
      try {
        body = await response.json();
      } catch (_) {
        body = null; // non-JSON error body; handled below
      }

      if (!response.ok) {
        const fallback = "The API returned HTTP " + response.status + ".";
        const error = new ApiError(
          kindForStatus(response.status),
          detailToMessage(body && body.detail, fallback),
          response.status
        );
        // A 422 whose detail is a list is the framework's field-by-field report (technical); a string detail was
        // written by this API for people (e.g. the image upload messages).
        error.fieldReport = Array.isArray(body && body.detail);
        throw error;
      }
      if (body === null) {
        throw new ApiError("http", "The API returned a response that is not valid JSON.", response.status);
      }
      return body;
    } finally {
      clearTimeout(timer);
    }
  }

  // A file from the API (the dashboard's CSV): {text, filename}. Errors are read like any other API error.
  async function download(path) {
    const response = await fetch(baseUrl() + path, { method: "GET" });
    if (!response.ok) {
      let detail = null;
      try { detail = (await response.json()).detail; } catch (_) { detail = null; }
      throw new ApiError(kindForStatus(response.status),
        detailToMessage(detail, "The API returned HTTP " + response.status + "."), response.status);
    }
    const match = /filename="([^"]+)"/.exec(response.headers.get("Content-Disposition") || "");
    return { text: await response.text(), filename: match ? match[1] : "report.csv" };
  }

  function postJson(path, payload) {
    return request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  root.CampusVerifyApi = {
    ApiError: ApiError,
    detailToMessage: detailToMessage,
    getHealth: function () { return request("/health", { method: "GET" }); },
    // Dashboard counts (CAMPUS_VERIFY_STATS): outcomes per day, never message text. Last 7 days by default.
    getStats: function (days) { return request("/stats?days=" + encodeURIComponent(days || 7), { method: "GET" }); },
    // The same counts as a spreadsheet (one row per day and kind of check).
    getStatsCsv: function (days) { return download("/stats.csv?days=" + encodeURIComponent(days || 7)); },
    verifyText: function (claim) { return postJson("/verify/text", { claim: claim }); },
    // Sign in with Google, DYCI accounts only (CAMPUS_VERIFY_AUTH=google). The session is an HttpOnly cookie.
    getAuthConfig: function () { return request("/auth/config", { method: "GET" }); },
    getMe: function () { return request("/auth/me", { method: "GET" }); },
    signInGoogle: function (credential) { return postJson("/auth/google", { credential: credential }); },
    signOut: function () { return postJson("/auth/logout", {}); },
    setHistory: function (on, deleteSaved) {
      return postJson("/account/history", { history_enabled: !!on, delete_saved: !!deleteSaved });
    },
    // "My checks" (CAMPUS_VERIFY_HISTORY): the signed-in student's own saved checks; nobody else can read them.
    listChecks: function () { return request("/account/checks", { method: "GET" }); },
    deleteCheck: function (id) { return postJson("/account/checks/" + encodeURIComponent(id) + "/delete", {}); },
    clearChecks: function () { return postJson("/account/checks/clear", {}); },
    deleteAccount: function (credential, confirm) {
      return postJson("/account/delete", { credential: credential, confirm: confirm });
    },
    setTourSeen: function () { return postJson("/account/tour-seen", {}); },
    listUsers: function () { return request("/admin/users", { method: "GET" }); },
    // Sources (CAMPUS_VERIFY_SOURCES): Facebook pages, administrators only. A token is sent once and never comes back.
    listSources: function () { return request("/admin/sources", { method: "GET" }); },
    // Fix B (CAMPUS_VERIFY_FETCH_STATUS): the live test-page fetch that feeds the checker; 404 when off.
    fetchStatus: function () { return request("/admin/fetch-status", { method: "GET" }); },
    addSource: function (pageId, tier, token) {
      return postJson("/admin/sources", { page_id: pageId, tier: tier, token: token });
    },
    updateSource: function (key, changes) { return postJson("/admin/sources/" + encodeURIComponent(key), changes); },
    replaceSourceToken: function (key, token) {
      return postJson("/admin/sources/" + encodeURIComponent(key) + "/token", { token: token });
    },
    recordCutoff: function (day) { return postJson("/admin/sources/cutoff", { day: day }); },
    sourceAction: function (key, action) {      // check | fetch | remove
      return postJson("/admin/sources/" + encodeURIComponent(key) + "/" + action, {});
    },
    setUserActive: function (id, active) { return postJson("/admin/users/" + encodeURIComponent(id) + "/active", { active: !!active }); },
    verifyUrl: function (claim, url) { return postJson("/verify/url", { claim: claim, url: url }); },
    // Image Check (Phase 1): the selected file object itself is the raw request body (no form encoding, no reading in the page).
    verifyImage: function (file) {
      return request("/verify/image", { method: "POST", headers: { "Content-Type": file.type }, body: file });
    },
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = root.CampusVerifyApi;
  }
})(typeof window !== "undefined" ? window : globalThis);

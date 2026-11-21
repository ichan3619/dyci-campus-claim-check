/*
 * Campus claim check UI (M11.1).
 *
 * Responsibility: collect input, call the API (via api.js), and DISPLAY the
 * response. There is no verification, retrieval, classification or verdict
 * logic here -- every verdict, score and piece of evidence shown is exactly
 * what the backend returned.
 *
 * Security note: all backend-supplied text (claim, evidence text, URLs,
 * IDs) is inserted with textContent / setAttribute only, never parsed as HTML.
 * Evidence from /verify/url is arbitrary third-party page content.
 */
(function () {
  "use strict";

  var Api = window.CampusVerifyApi;

  // ---- DOM helpers -------------------------------------------------------

  function $(id) { return document.getElementById(id); }

  function el(tag, opts, children) {
    var node = document.createElement(tag);
    opts = opts || {};
    if (opts.className) node.className = opts.className;
    if (opts.text !== undefined) node.textContent = opts.text;
    if (opts.attrs) {
      Object.keys(opts.attrs).forEach(function (k) { node.setAttribute(k, opts.attrs[k]); });
    }
    (children || []).forEach(function (c) { if (c) node.appendChild(c); });
    return node;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  // ---- presentation-only lookups -----------------------------------------

  // Wording for the three canonical verdicts. The canonical token itself is
  // always displayed unchanged next to this wording.
  var VERDICTS = {
    SUPPORTED: {
      cls: "supported", mark: "\u2713",
      headline: "Yes \u2014 a DYCI post backs this",
      note: "An official DYCI post agrees with your message."
    },
    CONTRADICTED: {
      cls: "contradicted", mark: "\u2715",
      headline: "No \u2014 DYCI\u2019s post says otherwise",
      note: "An official DYCI post says something different. Read what it says below before you share anything."
    },
    INSUFFICIENT_EVIDENCE: {
      cls: "insufficient", mark: "?",
      headline: "Not confirmed yet",
      note: "No DYCI post settles this yet. That does not mean it is false: read the closest post below, and wait for DYCI to announce it before passing it on."
    }
  };

  // Readable form of a backend evidence_status value WITHOUT changing its
  // meaning: underscores to spaces, sentence case. The raw value is shown too.
  function readableStatus(value) {
    var s = String(value).replace(/_/g, " ").toLowerCase();
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  function fmtScore(n) {
    return typeof n === "number" && isFinite(n) ? n.toFixed(3) : String(n);
  }

  function isHttpUrl(value) {
    return /^https?:\/\//i.test(String(value || ""));
  }

  // Written for students (pre-deployment review L4): no API, server or framework terms.
  var ERROR_TITLES = {
    network: "Can\u2019t reach the checker",
    timeout: "The check took too long",
    validation: "Please check what you entered",
    rejected: "This can\u2019t be checked",
    upstream: "Couldn\u2019t open that link",
    rate_limited: "Too many checks in a short time",
    busy: "The checker is busy",
    server: "Something went wrong on our side",
    http: "Something went wrong"
  };

  var ERROR_HINTS = {
    network: "No verdict was produced. The checker may be starting up or offline. Please try again in a minute.",
    timeout: "No verdict was produced. The checker may be starting up. Please try again.",
    validation: "No verdict was produced.",
    rejected: "No verdict was produced.",
    upstream: "No verdict was produced. This is a problem opening or reading the page, not a result about your claim. Try another link, or leave the link empty.",
    rate_limited: "No verdict was produced. Wait a moment and check again; the limit keeps the checker available for everyone.",
    busy: "No verdict was produced. Many people are checking at the same moment. Try again in about half a minute.",
    server: "No verdict was produced. Please try again later.",
    http: "No verdict was produced. Please try again later."
  };

  // The server's own wording is shown when it was written for people (too short, too many requests, the image upload
  // messages). Technical text (the framework's field report, the link guard's and fetcher's internals, network errors)
  // is replaced by a plain sentence and kept under "Technical details".
  var PLAIN_MESSAGES = {
    network: "The checker isn\u2019t responding right now.",
    timeout: "The checker didn\u2019t answer in time.",
    validation: "Messages can be up to 1,000 characters, and a link must start with http:// or https://.",
    upstream: "That page couldn\u2019t be opened or read.",
    server: "The check couldn\u2019t be completed.",
    http: "The check couldn\u2019t be completed."
  };

  function plainMessage(err) {
    var kind = err && err.kind ? err.kind : "http";
    var raw = err && err.message ? err.message : "";
    if (kind === "validation" && !err.fieldReport && raw) return raw;
    if (kind === "rejected" && /^Refusing to fetch/.test(raw)) {
      return "This link can\u2019t be used. Only public web pages can be checked.";
    }
    if (kind === "rejected" && /too large/i.test(raw)) return "That is too long to check. Please shorten it.";
    if (kind === "rejected" || kind === "rate_limited" || kind === "busy") return raw || PLAIN_MESSAGES.http;
    return PLAIN_MESSAGES[kind] || raw || "Something went wrong.";
  }

  // ---- state helpers -------------------------------------------------------

  var form = $("verify-form");
  var claimInput = $("claim");
  var urlInput = $("url");
  var submitBtn = $("submit-btn");
  var resetBtn = $("reset-btn");
  var claimError = $("claim-error");
  var loading = $("loading");
  var errorBox = $("error");
  var resultBox = $("result");

  function hideStatus() {
    loading.hidden = true;
    errorBox.hidden = true;
  }

  // While a check runs, the status line walks through what is happening (the last step stays until the answer).
  var STEPS = {
    text: ["Reading your message\u2026", "Searching DYCI\u2019s posts\u2026", "Comparing with the closest post\u2026"],
    url: ["Opening the page\u2026", "Reading the post\u2026", "Comparing it with your message\u2026"],
    picture: ["Reading the picture\u2026", "Checking each statement\u2026", "Matching the photo with DYCI\u2019s posts\u2026"]
  };
  // A check usually answers within about 10 seconds. Past that, other checks are most likely ahead in line (the server
  // runs CAMPUS_VERIFY_MAX_CONCURRENT at once), so the line says so instead of looking stuck.
  var WAIT_NOTE_MS = 10000;
  var WAIT_NOTE = "Many people are checking right now. This can take up to about 40 seconds; please keep this page open.";
  var stepTimer = null;
  var waitTimer = null;

  function showSteps(steps) {
    clearInterval(stepTimer);
    clearTimeout(waitTimer);
    var i = 0;
    loading.textContent = steps[0];
    loading.hidden = false;
    stepTimer = setInterval(function () {
      if (i < steps.length - 1) { i += 1; loading.textContent = steps[i]; }
    }, 1500);
    waitTimer = setTimeout(function () {
      clearInterval(stepTimer);
      loading.textContent = WAIT_NOTE;
    }, WAIT_NOTE_MS);
  }

  function stopSteps() {
    clearInterval(stepTimer);
    clearTimeout(waitTimer);
    loading.hidden = true;
  }

  function setBusy(isBusy, steps) {
    submitBtn.disabled = isBusy;
    submitBtn.textContent = isBusy ? "Checking\u2026" : "Check claim";
    if (isBusy) { showSteps(steps); } else { stopSteps(); }
  }

  function showError(err) {
    var kind = err && err.kind ? err.kind : "http";
    $("error-title").textContent = ERROR_TITLES[kind] || ERROR_TITLES.http;
    var message = plainMessage(err), raw = err && err.message ? err.message : "";
    $("error-message").textContent = message;
    $("error-hint").textContent = ERROR_HINTS[kind] || "";
    var tech = $("error-tech");
    if (tech) {
      tech.hidden = !raw || raw === message;
      $("error-tech-text").textContent = raw;
    }
    errorBox.hidden = false;
    errorBox.focus();
  }

  // ---- rendering -------------------------------------------------------------

  function metaList(pairs) {
    var dl = el("dl", { className: "meta" });
    pairs.forEach(function (p) {
      dl.appendChild(el("dt", { text: p[0] }));
      var dd = el("dd");
      if (p[2] === "link" && isHttpUrl(p[1])) {
        dd.appendChild(el("a", {
          text: p[1],
          attrs: { href: p[1], target: "_blank", rel: "noopener noreferrer" }
        }));
      } else if (p[2] === "code") {
        dd.appendChild(el("code", { text: p[1] }));
      } else {
        dd.textContent = p[1] === "" || p[1] === null || p[1] === undefined ? "(none returned)" : String(p[1]);
      }
      dl.appendChild(dd);
    });
    return dl;
  }

  // "2026-10-26T16:22:24+0000" or "2026-10-26" -> "26 October 2026"; anything else is shown as it is.
  function readableDate(value) {
    if (!value) return "";
    var m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(value));
    if (!m) return String(value);
    var d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
    return isNaN(d) ? String(value) : d.toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });
  }

  // The one sentence that settled the answer, first: "DYCI's post (20 September 2026) says: "...""
  function settlingSentence(data) {
    var h = data.holiday_check;
    if (h && h.applied) {
      return el("p", { className: "settles" }, [el("strong", { text: "The national holiday list says: " }),
        document.createTextNode(h.statement)]);
    }
    if ((data.verdict !== "SUPPORTED" && data.verdict !== "CONTRADICTED") || !data.selected_passage || !data.used_evidence) {
      return null;
    }
    var when = readableDate(data.used_evidence.original_date || data.used_evidence.published_date);
    var text = String(data.selected_passage.text || "").replace(/\s+/g, " ").trim();
    if (!text) return null;
    if (text.length > 280) { text = text.slice(0, 277) + "\u2026"; }
    return el("p", { className: "settles" }, [
      el("strong", { text: "DYCI\u2019s post" + (when ? " (" + when + ")" : "") + " says: " }),
      el("q", { text: text })]);
  }

  function renderVerdict(data) {
    var meta = VERDICTS[data.verdict];
    var known = !!meta;
    if (!known) {
      meta = {
        cls: "unknown", mark: "?",
        headline: "Unrecognized result",
        note: "The checker returned a result this page does not recognize. It is shown unchanged."
      };
    }
    var box = el("div", { className: "verdict " + meta.cls });
    box.appendChild(el("div", { className: "verdict-mark", text: meta.mark, attrs: { "aria-hidden": "true" } }));
    var body = el("div");
    body.appendChild(el("h2", { text: meta.headline, attrs: { id: "result-title" } }));
    body.appendChild(el("p", { text: meta.note }));
    var says = settlingSentence(data);
    if (says) { body.appendChild(says); }
    body.appendChild(el("p", { className: "verdict-code", text: "Result code: " + String(data.verdict) }));
    if (data.relevance_check && data.relevance_check.withheld) {
      body.appendChild(el("p", { text: "The closest post may not address this claim, so no verdict is given." }));
    }
    if (data.other_school_check && data.other_school_check.withheld) {
      body.appendChild(el("p", { text: "The post used does not mention " + data.other_school_check.missing_in_post.join(", ") + ", so no verdict is given." }));
    }
    if (data.wrong_post_check && data.wrong_post_check.withheld) {
      body.appendChild(el("p", { text: "The closest post is about something else (it does not mention " + data.wrong_post_check.message_dates + "), so no verdict is given." }));
    }
    if (data.tense_guard_check && data.tense_guard_check.withheld) {
      body.appendChild(el("p", { text: "The post only says this will happen; it does not say it has happened, so no verdict is given." }));
    }
    if (data.class_day_check && data.class_day_check.withheld) {
      body.appendChild(el("p", { text: "The closest DYCI post is about a different day, so it cannot settle classes on the day you asked about; no verdict is given." }));
    }
    if (data.same_day_shown) {
      body.appendChild(el("p", { text: "DYCI posted an advisory dated the day you asked about. It is shown first below; check it, because it may change what earlier posts said." }));
    }
    var parts = data.reason_parts;
    if (parts && parts.main_verdict === "SUPPORTED") {
      body.appendChild(el("p", { text: "✓ " + parts.main_claim }));
      var reasonLine = {
        backed: "✓ Reason: " + parts.reason,
        not_stated: "? Reason “" + parts.reason + "”: the DYCI post does not state this reason.",
        different: "✗ Reason “" + parts.reason + "”: the DYCI post gives a different reason."
      }[parts.reason_label] || "";
      if (reasonLine) body.appendChild(el("p", { text: reasonLine }));
    }
    var mainPoint = data.main_point;   // rule M v2 (CAMPUS_VERIFY_MAIN_POINT); never CONTRADICTED
    if (mainPoint && mainPoint.outcome === "main_not_supported") {
      body.appendChild(el("p", { text: "The main point was checked on its own: “" + mainPoint.main + "” could not be confirmed, so the message is not confirmed." }));
    }
    if (data.header_date_check) {
      body.appendChild(el("p", { text: "The advisory is dated " + data.header_date_check.header_date + " and says this happens today or immediately, so its date counts as the day it is about." }));
    }
    if (data.same_day_lookup) {
      body.appendChild(el("p", { text: "Other DYCI advisories dated " + data.same_day_lookup.day_name + " are shown first below. Check them; a later advisory may change what earlier posts said." }));
    }
    var sameDay = data.same_day_advisory || data.same_day_judge;
    if (sameDay && sameDay.action === "supported") {
      body.appendChild(el("p", { text: "DYCI posted an advisory dated the day you asked about, and it backs your message. It is shown first below." }));
    } else if (sameDay && sameDay.action === "disagree_withheld") {
      body.appendChild(el("p", { text: "DYCI's advisories dated that day disagree about your message, so no verdict is given. Check the latest announcement; they are shown first below." }));
    } else if (sameDay && sameDay.action === "stale_withheld") {
      body.appendChild(el("p", { text: "DYCI posted an advisory dated the day you asked about, so the earlier post may be out of date; no verdict is given. The advisory is shown first below." }));
    } else if (sameDay) {
      body.appendChild(el("p", { text: "DYCI posted an advisory dated the day you asked about. It is shown first below; check it, because it may change what earlier posts said." }));
    }
    var near = data.relative_resolution && data.relative_resolution.guard;
    if (near && near.withheld) {
      body.appendChild(el("p", { text: "The closest post is about a different time, so it cannot settle a message about that day; no verdict is given." }));
    }
    if (data.relative_date_check && data.relative_date_check.withheld) {
      body.appendChild(el("p", { text: "Your message is about \u201c" + data.relative_date_check.words[0] + "\u201d, and the closest post is not from today or yesterday, so no verdict is given." }));
    }
    box.appendChild(body);
    return box;
  }

  // ---- M12.3: how the backend router read the input ---------------------------
  // The router (backend) decides whether the input is a claim to check
  // (CHECKABLE_CLAIM) or a request for information (INFO_SEEKING). An
  // INFO_SEEKING response carries no verdict; this page only displays that.

  function isInfoSeeking(data) {
    return !!(data.route && data.route.status === "INFO_SEEKING");
  }

  // Owner policy: verdicts for English claims only for now. A Tagalog/Taglish claim comes back with
  // verdict_withheld "language": the closest announcements, no truth verdict.
  function isEvidenceOnly(data) {
    return data.verdict_withheld === "language";
  }

  function renderEvidenceOnlyHeader() {
    var box = el("div", { className: "verdict info" });
    box.appendChild(el("div", { className: "verdict-mark", text: "i", attrs: { "aria-hidden": "true" } }));
    var body = el("div");
    body.appendChild(el("h2", { text: "Here\u2019s what DYCI posted about this", attrs: { id: "result-title" } }));
    body.appendChild(el("p", {
      text: "The closest official DYCI posts are below \u2014 read them to see what they say. For Tagalog and Taglish " +
            "messages we show the posts without a yes/no answer, because automatic checks in these languages are not " +
            "reliable enough yet."
    }));
    body.appendChild(el("p", { className: "verdict-code", text: "Result code: evidence only (Tagalog/Taglish)" }));
    box.appendChild(body);
    return box;
  }

  function routeLabel(route) {
    if (!route) return "(router off: every input is checked as a claim)";
    var caveats = route.caveats || [];
    var notes = [];
    if (caveats.indexOf("fragment") !== -1) notes.push("short fragment");
    if (caveats.indexOf("tagalog_or_taglish") !== -1) notes.push("Tagalog/Taglish");
    var name = route.status === "INFO_SEEKING" ? "Request for information" : "Claim to check";
    return name + " (" + route.status + (notes.length ? "; " + notes.join(", ") : "") + ")";
  }

  function renderInformationHeader(data) {
    var box = el("div", { className: "verdict info" });
    box.appendChild(el("div", { className: "verdict-mark", text: "i", attrs: { "aria-hidden": "true" } }));
    var body = el("div");
    body.appendChild(el("h2", { text: "This looks like a question, not a claim", attrs: { id: "result-title" } }));
    body.appendChild(el("p", { className: "verdict-code", text: "No verdict (INFO_SEEKING)" }));
    body.appendChild(el("p", {
      text: data.route.guidance || "No truth verdict is given for a request for information."
    }));
    box.appendChild(body);
    return box;
  }

  function passageQuote(passage) {
    var quote = el("blockquote", { className: "evidence-text" });
    if (passage.context_before_text) {
      quote.appendChild(el("span", { className: "passage-context", text: passage.context_before_text + " " }));
    }
    quote.appendChild(document.createTextNode(passage.text));
    return quote;
  }

  // The most relevant announcement, shown right under every result so the reader can check it. When nothing passed the
  // relevance check, the top retrieved post is still shown, labelled as a low match that may not be related.
  var CLOSEST_NOTE = {
    claim: "This is the passage the classifier compared with the claim. Being selected does not mean the claim is true.",
    question: "The passage that best matches your question. Read it yourself: no claim was checked, so nothing here is marked true or false.",
    evidence: "The passage that best matches your message. Read it yourself: no verdict was given, so nothing here is marked true or false."
  };

  // Display-only date notice (date_notice in the response): the post does not contain a date the message names.
  function dateNoteFor(data, doc) {
    var n = data.date_notice && doc && data.date_notice.posts && data.date_notice.posts[doc.document_id];
    return n ? el("p", { className: "note date-note", text: n.text }) : null;
  }

  // Display only (results/round2/wrong_post_decision.md): a contradiction from a post that does not mention the day the
  // message names may be about a different day. The verdict is unchanged; the reader is told why it may not apply.
  function contradictionDayNote(data) {
    var doc = data.used_evidence;
    var n = data.verdict === "CONTRADICTED" && data.date_notice && doc && data.date_notice.posts && data.date_notice.posts[doc.document_id];
    if (!n) return null;
    return el("p", { className: "note date-note", text: "The post this is based on does not mention " + n.missing.join(" or ") +
      ". If your message is about a different day, this post may not apply to it." });
  }

  // Display only (post_age_notice; owner 2026-10-25): the verdict rests on an old or undated post while the message is
  // about now. The verdict is unchanged; the reader is warned it may be outdated or about a different event.
  function oldPostNote(data) {
    var n = data.post_age_notice;
    return n && n.text ? el("p", { className: "note date-note old-post", text: n.text }) : null;
  }

  // Display only (hearsay_notice, CAMPUS_VERIFY_HEARSAY_NOTE): the message reads as a prediction or hearsay. Shown under
  // the verdict or header in every mode; it never changes the verdict.
  // CAMPUS_VERIFY_HOLIDAYS: the warning when the national holiday list settled the answer, else the holiday note.
  function holidayNote(data) {
    var h = data.holiday_check;
    if (h && h.applied) {
      return el("p", { className: "note date-note resolved",
                       text: "Holiday calendar: " + h.statement + " " + h.warning });
    }
    var n = data.holiday_notice;
    return n && n.text ? el("p", { className: "note date-note", text: n.text }) : null;
  }

  function hearsayNote(data) {
    var n = data.hearsay_notice;
    return n && n.text ? el("p", { className: "note date-note hearsay", text: n.text }) : null;
  }

  // Display only (relative_resolution, CAMPUS_VERIFY_RESOLVE_RELATIVE): which calendar day "tomorrow" etc. were read as.
  function resolutionNote(data) {
    var r = data.relative_resolution;
    if (!r || !r.words || !r.words.length) return null;
    var parts = r.words.map(function (w, i) { return "“" + w + "” was read as " + r.read_as[i]; });
    return el("p", { className: "note date-note resolved", text: parts.join("; ") + " (Philippine time)." });
  }

  // CAMPUS_VERIFY_TAGALOG_PHRASES: the English sentence a Tagalog/Taglish message was checked as. Display only.
  function tagalogNote(data) {
    var m = data.tagalog_translation;
    if (m && m.accepted && m.translation) {      // CAMPUS_VERIFY_TAGALOG_TRANSLATE: local machine translation
      return el("p", { className: "note date-note resolved",
                       text: "Read as (machine translation): “" + m.translation + "”. Tagalog checks are less tested than English." });
    }
    var t = data.tagalog_reading;
    if (!t || !t.reading) return null;
    return el("p", { className: "note date-note resolved",
                     text: "Read as: “" + t.reading + "” (fixed Tagalog phrase list; Tagalog support is limited)." });
  }

  // Display-only relative-date notice (relative_date_notice): the message says "tomorrow"/"bukas"; show the post's date.
  function relativeNoteFor(data, doc) {
    var n = data.relative_date_notice && doc && data.relative_date_notice.posts && data.relative_date_notice.posts[doc.document_id];
    return n ? el("p", { className: "note date-note", text: n.text }) : null;
  }

  function renderClosest(data, kind) {
    var panel = el("div", { className: "panel closest" });
    var doc = data.used_evidence;
    var top = (data.retrieved_evidence || [])[0];
    panel.appendChild(el("h3", { text: doc ? "Closest announcement" : "Closest post (low match)" }));
    if (doc) {
      panel.appendChild(el("p", { className: "note", text: CLOSEST_NOTE[kind] }));
      var docNote = dateNoteFor(data, doc);
      if (docNote) { panel.appendChild(docNote); }
      var docRel = relativeNoteFor(data, doc);
      if (docRel) { panel.appendChild(docRel); }
      panel.appendChild(data.selected_passage ? passageQuote(data.selected_passage)
                                               : el("blockquote", { className: "evidence-text", text: doc.text }));
      panel.appendChild(evidenceMeta(doc));
      return panel;
    }
    if (!top) {
      panel.appendChild(el("p", { text: "No campus announcement was found." }));
      return panel;
    }
    panel.appendChild(el("p", {
      className: "scope-note",
      text: "No campus announcement matches this. This checker answers claims and questions about DYCI announcements, for example \u201cMay pasok ba sa Sept 28?\u201d or \u201cThe midterms are on September 28 to 30.\u201d"
    }));
    panel.appendChild(el("p", {
      className: "note",
      text: "Low match, may not be related: no announcement was similar enough to check against, so this is only the nearest post. It was not used for any verdict."
    }));
    var topNote = dateNoteFor(data, top);
    if (topNote) { panel.appendChild(topNote); }
    var topRel = relativeNoteFor(data, top);
    if (topRel) { panel.appendChild(topRel); }
    panel.appendChild(el("blockquote", { className: "evidence-text", text: top.text }));
    panel.appendChild(evidenceMeta(top));
    return panel;
  }

  function renderSummaryPanel(data, mode) {
    var info = isInfoSeeking(data) || isEvidenceOnly(data);
    var panel = el("div", { className: "panel" });
    panel.appendChild(el("h3", { text: info ? "How this was handled" : "How this was checked" }));
    panel.appendChild(el("p", {
      text: mode === "url"
        ? "Searched the submitted URL only. That page was used for this request and is not stored."
        : "Searched the campus announcements."
    }));
    var statusDd = readableStatus(data.evidence_status);
    var rows = [["Read as", routeLabel(data.route)]];
    if (isEvidenceOnly(data)) {
      rows.push(["Verdict", "Withheld: verdicts are given for English claims only for now"]);
    }
    if (data.claim_rewrite && data.claim_rewrite.changed) {
      rows.push(["Checked as", data.claim_rewrite.rewritten]);
    }
    if (data.other_school_check && data.other_school_check.withheld) {
      rows.push(["Other-school check", "The verdict was withheld because the post used does not mention the school named in the message."]);
    }
    if (data.tense_guard_check && data.tense_guard_check.withheld) {
      rows.push(["Tense guard", "The verdict was withheld because the post only says the event will happen."]);
    }
    if (data.same_day_advisory) {
      rows.push(["Same-day advisory", "An advisory dated the day in your message was also checked (" + data.same_day_advisory.action.replace(/_/g, " ") + ")."]);
    }
    if (data.wrong_post_check && data.wrong_post_check.withheld) {
      rows.push(["Wrong-post check", "The verdict was withheld because the post used is probably about something else (it does not mention the date in your message)."]);
    }
    if (data.relative_date_check && data.relative_date_check.withheld) {
      rows.push(["Relative date check", "The verdict was withheld because the post used is not from today or yesterday."]);
    }
    if (data.relevance_check && data.relevance_check.withheld) {
      rows.push(["Relevance check", "The verdict was withheld because the closest post may not address this claim."]);
    }
    if (data.date_check) {
      rows.push(["Date check", "The classifier's label was withheld because the evidence does not back the date in the claim (" + data.date_check + ")."]);
    }
    rows.push(
      ["Evidence status", statusDd + " (" + data.evidence_status + ")"],
      ["Explanation", data.explanation],
      ["Data status", data.data_status]
    );
    panel.appendChild(metaList(rows));
    return panel;
  }

  function renderModelScore(cr) {
    var panel = el("div", { className: "panel" });
    panel.appendChild(el("h3", { text: "Model score" }));
    panel.appendChild(el("p", {
      className: "note",
      text: "Raw classifier output for the evidence used, one score per label. It reflects how the model rates the claim against that one passage. It is not the probability that the claim is true."
    }));

    var labels = Object.keys(cr.label_scores || {}).sort(function (a, b) {
      return cr.label_scores[b] - cr.label_scores[a];
    });
    var table = el("table");
    var head = el("tr", {}, [
      el("th", { text: "Label", attrs: { scope: "col" } }),
      el("th", { className: "num", text: "Model score", attrs: { scope: "col" } })
    ]);
    table.appendChild(el("thead", {}, [head]));
    var tbody = el("tbody");
    labels.forEach(function (label) {
      var isPredicted = label === cr.label;
      var tr = el("tr", { className: isPredicted ? "predicted" : "" }, [
        el("td", { text: label + (isPredicted ? " (predicted)" : "") }),
        el("td", { className: "num", text: fmtScore(cr.label_scores[label]) })
      ]);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    panel.appendChild(table);
    if (cr.model) {
      panel.appendChild(el("p", { className: "note", text: "Judged by: " + cr.model }));
    }
    return panel;
  }

  // Readable date from the API's value ("2026-09-27T04:26:04+0000" or "2026-01-10"). Only the date part is used, so
  // no time-zone conversion can move it to another day; anything unrecognised is shown unchanged.
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  function fmtDate(value) {
    var m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(value || ""));
    return m ? Number(m[3]) + " " + MONTHS[Number(m[2]) - 1] + " " + m[1] : String(value);
  }

  // How the system acquired each document (API field `provenance`). A test-Page copy is never presented as DYCI's own
  // Page: the system has no access to it.
  var PROVENANCE = {
    test_page_mirror: {
      label: "Copy of a DYCI post, re-posted by the project team to its own test Facebook Page (not DYCI’s Page)",
      tag: "Test Page copy", dateLabel: "Re-posted on test Page"
    },
    manual_entry: { label: "DYCI post, entered by hand by the project team", tag: "Entered by hand", dateLabel: "Posted by DYCI" },
    submitted_page: { label: "The page you submitted, used for this check only", tag: "Your link", dateLabel: "Page date" },
    dyci_website: { label: "DYCI's website (dyci.edu.ph)", tag: "DYCI website", dateLabel: "Posted by DYCI" },
    org_page: { label: "Post by a DYCI student organisation, not an official DYCI announcement", tag: "Student organisation", dateLabel: "Posted" },
    fictional: { label: "Fictional development example, not a real campus notice", tag: "Fictional example", dateLabel: "Dated" }
  };
  var NOT_RECORDED = "Not recorded";

  function evidenceMeta(doc) {
    var kind = kindOf(doc);
    var rows = [["Document ID", doc.document_id, "code"]];
    if (kind) {
      rows.push(["Source", kind.label]);
    }
    if (doc.published_date) {
      rows.push([kind ? kind.dateLabel : "Posted", fmtDate(doc.published_date)]);
    }
    if (doc.provenance === "test_page_mirror") {
      rows.push(doc.original_url ? ["Original DYCI post", doc.original_url, "link"] : ["Original DYCI post", NOT_RECORDED]);
      if (doc.original_date) {
        rows.push(["Original date", fmtDate(doc.original_date)]);
      }
      rows.push(["Copy on the test Page", doc.source_url, "link"]);
    } else if (doc.provenance === "manual_entry") {
      rows.push(doc.source_url ? ["Original DYCI post", doc.source_url, "link"] : ["Original DYCI post", NOT_RECORDED]);
    } else {
      if (doc.original_url) {
        rows.push(["Original announcement", doc.original_url, "link"]);
      }
      if (doc.original_date) {
        rows.push(["Original date", fmtDate(doc.original_date)]);
      }
      rows.push(["Source URL", doc.source_url, "link"]);
    }
    var box = el("div");
    box.appendChild(metaList(rows.slice(1)));                 // what a reader needs: source, dates, links
    var tech = metaList([rows[0], ["Source type", doc.source_type, "code"], ["Similarity", fmtScore(doc.score)]]);
    tech.className = "meta tech";                              // identifiers and the retrieval score, de-emphasised
    box.appendChild(tech);
    return box;
  }
  function kindOf(doc) {
    return PROVENANCE[doc.provenance] || (doc.synthetic ? PROVENANCE.fictional : null);
  }

  function dateLabelOf(doc) {
    return kindOf(doc) ? kindOf(doc).dateLabel : "Posted";
  }


  function renderRetrieved(list, usedDoc, usedTag, data) {
    var panel = el("div", { className: "panel" });
    panel.appendChild(el("h3", { text: "Retrieved evidence" }));
    panel.appendChild(el("p", {
      className: "note",
      text: "Similarity is how closely a passage matches the claim\u2019s wording during retrieval. It is retrieval similarity, not the probability that the claim is true."
    }));
    if (!list || !list.length) {
      panel.appendChild(el("p", { text: "Retrieval returned no passages for this claim." }));
      return panel;
    }
    var ul = el("ul", { className: "evidence-list" });
    list.forEach(function (doc) {
      var li = el("li");
      var head = el("div", { className: "item-head" }, [
        el("strong", { text: doc.document_id }),
        doc.published_date ? el("span", { text: dateLabelOf(doc) + " " + fmtDate(doc.published_date) }) : null,
        el("span", { text: "Similarity " + fmtScore(doc.score) }),
        kindOf(doc) ? el("span", { className: "tag example", text: kindOf(doc).tag }) : null,
        usedDoc && usedDoc.document_id === doc.document_id
          ? el("span", { className: "tag", text: usedTag }) : null
      ]);
      li.appendChild(head);
      var itemNote = dateNoteFor(data || {}, doc);
      if (itemNote) { li.appendChild(itemNote); }
      var itemRel = relativeNoteFor(data || {}, doc);
      if (itemRel) { li.appendChild(itemRel); }
      var details = el("details");
      details.appendChild(el("summary", { text: "Show passage and source" }));
      details.appendChild(evidenceMeta(doc));
      details.appendChild(el("blockquote", { className: "evidence-text", text: doc.text }));
      li.appendChild(details);
      ul.appendChild(li);
    });
    panel.appendChild(ul);
    return panel;
  }

  function renderResult(data, mode) {
    var info = isInfoSeeking(data);
    clear(resultBox);
    var evidenceOnly = isEvidenceOnly(data);
    resultBox.appendChild(el("p", { className: "claim-echo" }, [
      document.createTextNode(info ? "You asked: " : evidenceOnly ? "Message: " : "Claim checked: "),
      el("q", { text: data.claim })
    ]));
    if (data.scope_notice && data.scope_notice.text) {
      resultBox.appendChild(el("p", { className: "note date-note scope", text: data.scope_notice.text }));
    }
    if (data.unit_check) {
      // Rule N (CAMPUS_VERIFY_UNIT_CHECK): the message names another office, level, grade or venue than the post.
      resultBox.appendChild(el("p", { className: "note date-note scope", text: "The DYCI post is about " +
        (data.unit_check.post_units || []).join(" / ") + ", not " + data.unit_check.message_unit + ", so this is not confirmed." }));
    }
    if (data.class_wording) {
      // Rule C (CAMPUS_VERIFY_CLASS_WORDING): the clipped class wording was checked in the standard form.
      resultBox.appendChild(el("p", { className: "note date-note scope", text: "Checked as: “" +
        data.class_wording.rewritten + "”" }));
    }
    if (data.org_source && data.org_source.school_wide) {
      // Rule O-SW (CAMPUS_VERIFY_ORG_SCHOOL_WIDE): a school-wide message confirmed by a department page.
      resultBox.appendChild(el("p", { className: "note date-note scope", text: data.org_source.note +
        (data.org_source.post_date ? " Posted " + data.org_source.post_date + "." : "") +
        " DYCI's own posts did not settle it." }));
    } else if (data.org_source) {
      // Rule O (CAMPUS_VERIFY_ORG_PAGES): DYCI's posts gave no answer; the organisation's own post decided.
      resultBox.appendChild(el("p", { className: "note date-note scope", text: (data.org_source.label ?
        (data.verdict === "SUPPORTED" ? "Supported" : "Contradicted") + ": " + data.org_source.label + ". From " : "This answer comes from a post by ") +
        data.org_source.name + (data.org_source.post_date ? " (" + data.org_source.post_date + ")" : "") +
        ", not an official DYCI announcement. DYCI's own posts did not settle it." }));
    } else if (data.org_related) {
      resultBox.appendChild(el("p", { className: "note date-note scope", text: "Related post by " + data.org_related.name +
        (data.org_related.post_date ? " (" + data.org_related.post_date + ")" : "") + ": “" +
        data.org_related.text.slice(0, 160) + "…” It did not decide this answer." }));
    }
    if (data.value_withhold) {
      // Rule V (CAMPUS_VERIFY_VALUE_WITHHOLD): the message gives another time, duration, percentage or place.
      resultBox.appendChild(el("p", { className: "note date-note scope", text: "The DYCI post gives " +
        (data.value_withhold.post_values || []).join(" / ") + ", not " + data.value_withhold.message_value +
        ", so this is not confirmed." }));
    }
    if (data.link_source) {
      // Rule U (CAMPUS_VERIFY_LINK_SOURCE): the link is not DYCI's; the answer above comes from DYCI's own posts.
      var ls = data.link_source;
      resultBox.appendChild(el("p", { className: "note date-note scope", text: "This link is from " + (ls.host || "another site") +
        ", which is not a DYCI source. The answer above comes from DYCI's own posts. " +
        (ls.read ? "The linked page itself " + ls.page_says + " your message." : "The linked page could not be read.") }));
    }
    var more = el("details", { className: "more" });
    more.appendChild(el("summary", { text: "Details: how this was checked and every post retrieved" }));
    if (info) {
      resultBox.appendChild(renderInformationHeader(data));
      resultBox.appendChild(renderClosest(data, "question"));
      more.appendChild(renderSummaryPanel(data, mode));
      more.appendChild(renderRetrieved(data.retrieved_evidence, data.used_evidence, "Shown above", data));
    } else if (evidenceOnly) {
      resultBox.appendChild(renderEvidenceOnlyHeader());
      resultBox.appendChild(renderClosest(data, "evidence"));
      more.appendChild(renderSummaryPanel(data, mode));
      more.appendChild(renderRetrieved(data.retrieved_evidence, data.used_evidence, "Shown above", data));
    } else {
      resultBox.appendChild(renderVerdict(data));
      var oldPost = oldPostNote(data);
      if (oldPost) { resultBox.appendChild(oldPost); }
      var differentDay = contradictionDayNote(data);
      if (differentDay) { resultBox.appendChild(differentDay); }
      resultBox.appendChild(renderClosest(data, "claim"));
      more.appendChild(renderSummaryPanel(data, mode));
      if (data.classification_result) {
        more.appendChild(renderModelScore(data.classification_result));
      }
      more.appendChild(renderRetrieved(data.retrieved_evidence, data.used_evidence, "Used by the classifier", data));
    }
    var holiday = holidayNote(data);
    if (holiday) { resultBox.insertBefore(holiday, resultBox.firstChild ? resultBox.firstChild.nextSibling : null); }
    var hearsay = hearsayNote(data);
    if (hearsay) { resultBox.insertBefore(hearsay, resultBox.firstChild ? resultBox.firstChild.nextSibling : null); }
    var resolved = resolutionNote(data);
    if (resolved) { resultBox.insertBefore(resolved, resultBox.firstChild ? resultBox.firstChild.nextSibling : null); }
    var tagalog = tagalogNote(data);
    if (tagalog) { resultBox.insertBefore(tagalog, resultBox.firstChild ? resultBox.firstChild.nextSibling : null); }
    if (uiExtras && !info) {
      var steps = nextSteps(data);
      if (steps) { resultBox.appendChild(steps); }
      resultBox.appendChild(shareBar(data));
    }
    resultBox.appendChild(more);
    resultBox.hidden = false;
    resultBox.focus();
  }

  // ---- CAMPUS_VERIFY_UI_EXTRAS (display only; /health says when it is on) ----------------------------------------
  // Tap-to-try examples, a "what to do next" panel when nothing is confirmed, and a copy/share summary for group chats.
  // None of these reads or changes a verdict: they only reword what the answer already says.
  var uiExtras = false;
  var EXAMPLES = ["may pasok ba bukas?", "The midterm exams are on October 5 to 7.", "Is Nov 1 a holiday?",
                  "I heard there are no classes tomorrow."];

  function addExamples() {
    var help = $("claim-help");
    if (!help || $("examples")) return;
    var box = el("div", { className: "examples", attrs: { id: "examples" } });
    box.appendChild(el("span", { className: "help", text: "Try: " }));
    EXAMPLES.forEach(function (text) {
      var b = el("button", { className: "chip", text: text, attrs: { type: "button" } });
      b.addEventListener("click", function () {
        var claim = $("claim");
        claim.value = text;
        claim.focus();
      });
      box.appendChild(b);
    });
    help.parentNode.insertBefore(box, help.nextSibling);
  }

  var SHARE_WORDS = {
    SUPPORTED: "Supported by an official DYCI post",
    CONTRADICTED: "Contradicted by an official DYCI post",
    INSUFFICIENT_EVIDENCE: "Not confirmed: no DYCI post settles this"
  };

  function shareSummary(data) {
    var h = data.holiday_check && data.holiday_check.applied ? data.holiday_check : null;
    var head = data.verdict ? SHARE_WORDS[data.verdict] || String(data.verdict) : "No verdict: the closest DYCI posts are shown";
    if (h) { head = head.replace("an official DYCI post", "the national holiday list"); }
    var lines = ["DYCI Campus Check: " + head, "Message: “" + data.claim + "”"];
    var post = data.used_evidence || (data.retrieved_evidence && data.retrieved_evidence[0]);
    if (h) {
      lines.push("Holiday list: " + h.statement);
    } else if (post) {
      var when = post.original_date || post.published_date;
      var passage = data.selected_passage && data.selected_passage.text ? data.selected_passage.text : post.text;
      passage = String(passage).replace(/\s+/g, " ").trim();
      if (passage.length > 220) { passage = passage.slice(0, 217) + "…"; }
      lines.push((data.used_evidence ? "Post" : "Closest post") + (when ? " (" + when + ")" : "") + ": “" + passage + "”");
      var link = isHttpUrl(post.original_url) ? post.original_url : isHttpUrl(post.source_url) ? post.source_url : null;
      if (link) { lines.push(link); }
    }
    lines.push("Not an official DYCI statement. Always check DYCI’s own page.");
    return lines.join("\n");
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) { return navigator.clipboard.writeText(text); }
    return new Promise(function (resolve, reject) {
      var t = document.createElement("textarea");
      t.value = text;
      t.setAttribute("readonly", "");
      t.style.position = "fixed";
      t.style.opacity = "0";
      document.body.appendChild(t);
      t.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
      document.body.removeChild(t);
      if (ok) { resolve(); } else { reject(new Error("copy failed")); }
    });
  }

  function shareBar(data) {
    var text = shareSummary(data);
    var bar = el("div", { className: "actions share-bar" });
    var status = el("span", { className: "help", attrs: { role: "status" } });
    var copy = el("button", { className: "secondary", text: "Copy result", attrs: { type: "button" } });
    copy.addEventListener("click", function () {
      copyText(text).then(function () { status.textContent = "Copied. Paste it in your group chat."; },
                          function () { status.textContent = "Could not copy. Select the text above instead."; });
    });
    bar.appendChild(copy);
    if (navigator.share) {
      var share = el("button", { className: "secondary", text: "Share", attrs: { type: "button" } });
      share.addEventListener("click", function () { navigator.share({ text: text }).catch(function () {}); });
      bar.appendChild(share);
    }
    bar.appendChild(status);
    return bar;
  }

  function nextSteps(data) {
    if (data.verdict && data.verdict !== "INSUFFICIENT_EVIDENCE") return null;
    var box = el("div", { className: "note next-steps" });
    box.appendChild(el("p", { className: "next-title", text: "What to do next" }));
    var list = el("ul");
    [data.verdict ? "No DYCI post confirms or denies this yet. Read the closest posts below: one may still answer it."
                  : "Read the closest DYCI posts below: they are the official source for this.",
     "Check DYCI’s official Facebook page for a newer announcement.",
     "Until DYCI posts about it, treat it as unconfirmed and don’t forward it as fact."].forEach(function (t) {
      list.appendChild(el("li", { text: t }));
    });
    box.appendChild(list);
    return box;
  }

  // ---- events ------------------------------------------------------------------

  form.addEventListener("submit", function (event) {
    event.preventDefault();

    var claim = claimInput.value.trim();
    var url = urlInput.value.trim();

    claimError.hidden = true;
    claimInput.removeAttribute("aria-invalid");
    if (!claim) {
      claimError.textContent = "Enter a claim to check.";
      claimError.hidden = false;
      claimInput.setAttribute("aria-invalid", "true");
      claimInput.focus();
      return;
    }
    // Input guard (same rule as the API): fewer than 3 letters cannot be a claim or question.
    if ((claim.match(/\p{L}/gu) || []).length < 3) {
      claimError.textContent = "Please type a claim or question about a DYCI announcement, for example \u201cWalang pasok sa Sept 28?\u201d or \u201cThe midterms are on September 28 to 30.\u201d";
      claimError.hidden = false;
      claimInput.setAttribute("aria-invalid", "true");
      claimInput.focus();
      return;
    }

    if (activeMode === "url" && !url) {
      claimError.textContent = "Paste the link to check this message against.";
      claimError.hidden = false;
      urlInput.focus();
      return;
    }
    if (activeMode !== "url") { url = ""; }                 // the Text tab never sends a link
    var mode = url ? "url" : "text";
    hideStatus();
    resultBox.hidden = true;
    setBusy(true, STEPS[mode]);

    var call = mode === "url" ? Api.verifyUrl(claim, url) : Api.verifyText(claim);
    call.then(function (data) {
      setBusy(false);
      renderResult(data, mode);
    }).catch(function (err) {
      setBusy(false);
      showError(err);
    });
  });

  // ---- Image Check (Phase 1) ---------------------------------------------------------------------------------------
  var imageForm = $("image-form");
  var imageInput = $("image-file");
  var imagePreview = $("image-preview");
  var imageError = $("image-error");
  var imageSubmit = $("image-submit");
  var IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];
  var IMAGE_MAX_BYTES = 5 * 1024 * 1024;
  var previewUrl = null;

  function imageProblem(file) {
    if (!file) return "Choose a screenshot or photo first.";
    if (IMAGE_TYPES.indexOf(file.type) < 0) return "Use a JPEG, PNG or WebP image.";
    if (file.size > IMAGE_MAX_BYTES) return "The image is larger than 5 MB.";
    return null;
  }

  imageInput.addEventListener("change", function () {
    imageError.hidden = true;
    if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
    var file = imageInput.files && imageInput.files[0];
    if (file && !imageProblem(file)) {
      previewUrl = URL.createObjectURL(file);            // local preview only; nothing is read into the page
      imagePreview.src = previewUrl;
      imagePreview.hidden = false;
    } else {
      imagePreview.hidden = true;
      imagePreview.removeAttribute("src");
    }
  });

  // Part B: sentence-level results. The headline describes the checked statements, never the whole picture.
  var SENTENCE_HEADLINES = {
    conflict: { cls: "contradicted", mark: "!" },
    possible_conflict: { cls: "info", mark: "?" },
    partly_confirmed: { cls: "info", mark: "~" },
    all_match: { cls: "supported", mark: "✓" },
    not_confirmed: { cls: "insufficient", mark: "–" }
  };

  function statementOutcome(s) {
    if (s.status === "not_checked_short") return "Not checked (too short to check on its own)";
    if (s.status === "not_checked_limit") return "Not checked (limit of 12 statements)";
    if (s.status === "uncertain") return "Couldn’t check reliably";
    if (s.outcome === "match") return "Matches an announcement";
    if (s.outcome === "contradiction") return "Conflicts with an announcement";
    var tc = s.text_check || {};
    if (tc.verdict_withheld === "language") return "No verdict (Tagalog/Taglish: closest posts only)";
    if (tc.route && tc.route.status === "INFO_SEEKING") return "No verdict (reads as a question)";
    return "Not confirmed";
  }

  function renderSentenceResult(data) {
    var sum = data.sentence_summary;
    var meta = SENTENCE_HEADLINES[sum.kind] || { cls: "unknown", mark: "?" };
    clear(resultBox);
    var box = el("div", { className: "verdict " + meta.cls });
    box.appendChild(el("div", { className: "verdict-mark", text: meta.mark, attrs: { "aria-hidden": "true" } }));
    var body = el("div");
    body.appendChild(el("h2", { text: sum.label, attrs: { id: "result-title" } }));
    body.appendChild(el("p", { text: sum.checked + " statement" + (sum.checked === 1 ? "" : "s") + " checked: " +
      sum.matches + " match, " + sum.contradictions + " conflict, " + sum.inconclusive + " not confirmed." }));
    if (sum.kind === "possible_conflict") {
      body.appendChild(el("p", { text: "No other statement in this picture matches the post it conflicts with, so that post may not be about the same announcement." }));
    }
    if (sum.kind === "all_match" || sum.kind === "partly_confirmed") {
      body.appendChild(el("p", { text: "Only the checked statements were compared; parts that were not checked are not confirmed. “Not confirmed” does not mean the claim is false." }));
    }
    box.appendChild(body);
    resultBox.appendChild(box);
    var list = el("ol", { className: "statement-list" });
    (data.sentence_checks || []).forEach(function (s) {
      var item = el("li", { className: "statement " + (s.outcome || s.status) });
      item.appendChild(el("q", { text: s.text }));
      item.appendChild(el("p", { className: "statement-outcome", text: statementOutcome(s) }));
      var used = s.text_check && s.text_check.used_evidence;
      if (used && (s.outcome === "match" || s.outcome === "contradiction")) {
        var post = el("p", { className: "note" }, [document.createTextNode("Post: " + used.document_id + " ")]);
        if (isHttpUrl(used.source_url)) {
          post.appendChild(el("a", { text: "open", attrs: { href: used.source_url, target: "_blank", rel: "noopener noreferrer" } }));
        }
        item.appendChild(post);
      }
      list.appendChild(item);
    });
    resultBox.appendChild(list);
    var skipped = sum.not_checked_short + sum.not_checked_limit;
    if (skipped) {
      resultBox.appendChild(el("p", { className: "note", text: skipped + " part" + (skipped === 1 ? " was" : "s were") + " not checked (" +
        sum.not_checked_short + " too short, " + sum.not_checked_limit + " over the limit of 12)." }));
    }
    resultBox.hidden = false;
  }

  function renderImageResult(data) {
    if (data.check_basis === "sentences" && data.sentence_summary) {
      renderSentenceResult(data);
    } else if (data.text_check) {
      renderResult(data.text_check, "text");
    } else {
      clear(resultBox);
      resultBox.hidden = false;
    }
    if (data.check_basis === "whole_text_fallback") {
      resultBox.insertBefore(el("p", { className: "note uncertain", text: "Checked as one piece of text (no separate statements found)." }), resultBox.firstChild);
    }
    var panel = el("div", { className: "panel ocr-panel" });
    panel.appendChild(el("h3", { text: "Text read from the picture" }));
    panel.appendChild(el("blockquote", { className: "evidence-text", text: data.ocr && data.ocr.text ? data.ocr.text : "(no readable text)" }));
    panel.appendChild(el("p", { className: data.status === "uncertain" ? "note uncertain" : "note", text: data.note }));
    // Source rule: a screenshot from another institution's page is not checked; show the closest DYCI posts, or say
    // that the source page could not be read.
    var src = data.source_check;
    if (src && src.status === "no_header" && src.note) {
      panel.appendChild(el("p", { className: "note", text: src.note }));
    }
    if (src && src.status === "other_page" && src.closest && src.closest.length) {
      panel.appendChild(el("p", { className: "note", text: "Closest DYCI announcements:" }));
      var list = el("ul", { className: "closest-list" });
      src.closest.forEach(function (d) {
        list.appendChild(el("li", { text: (d.text || "").slice(0, 200) + ((d.text || "").length > 200 ? "\u2026" : "") }));
      });
      panel.appendChild(list);
    }
    if (data.poster_cleanup) {
      // Rule R (CAMPUS_VERIFY_POSTER_CLEANUP): display only; the checked text below already reflects it.
      panel.appendChild(el("p", { className: "note", text: "Read as a poster: the seal, school name, label and footer were set aside" +
        (data.poster_cleanup.poster_date ? " (poster dated " + data.poster_cleanup.poster_date + ")" : "") + "." }));
    }
    if (data.ocr && data.ocr.normalization && data.ocr.normalization.length) {
      panel.appendChild(el("p", { className: "note", text: "Small reading slips were tidied before checking (extra spaces, or a time like 1.00 PM read as 1:00 PM). Checked text: " + data.ocr.checked_text }));
    }
    if (data.ocr && data.ocr.truncated) {
      panel.appendChild(el("p", { className: "note", text: "Only the first part of the text was checked." }));
    }
    resultBox.insertBefore(panel, resultBox.firstChild);
    if (data.assessment) { renderLevelCard(data); }
    if (data.provenance) { resultBox.appendChild(renderProvenance(data.provenance)); }
    resultBox.focus();
  }

  // Provenance facts (Phase 2B P1): what the file says about itself. Display only; it never sets a verdict or level.
  function renderProvenance(p) {
    var box = el("div", { className: "panel provenance-panel" });
    box.appendChild(el("h3", { text: "File details (Content Credentials and camera)" }));
    box.appendChild(el("p", { text: p.summary }));
    box.appendChild(el("p", { className: "note", text: p.note }));
    return box;
  }

  // Legitimacy level (results/round2/legitimacy_levels_spec.md): one card on top; everything else folded under Details.
  var LEVEL_TONE = { 5: "supported", 4: "supported", 3: "insufficient", 2: "sketchy", 1: "contradicted" };

  function safeLink(url, text) {
    if (!url || !/^https:\/\/(www\.)?facebook\.com\//.test(url)) return null;
    return el("a", { text: text, attrs: { href: url, target: "_blank", rel: "noopener noreferrer" } });
  }

  // A half-dial with five segments, the first `level` of them lit. It shows a position on the fixed 1-5 scale of
  // results/round2/legitimacy_levels_spec.md, not a probability.
  function levelGauge(level) {
    var NS = "http://www.w3.org/2000/svg", R = 40, CX = 50, CY = 50;
    var svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 100 60");
    svg.setAttribute("class", "gauge");
    svg.setAttribute("aria-hidden", "true");
    for (var i = 0; i < 5; i++) {
      var a0 = Math.PI * (1 - i / 5) - 0.04, a1 = Math.PI * (1 - (i + 1) / 5) + 0.04;
      var p = document.createElementNS(NS, "path");
      p.setAttribute("d", "M " + (CX + R * Math.cos(a0)).toFixed(2) + " " + (CY - R * Math.sin(a0)).toFixed(2) +
        " A " + R + " " + R + " 0 0 1 " + (CX + R * Math.cos(a1)).toFixed(2) + " " + (CY - R * Math.sin(a1)).toFixed(2));
      p.setAttribute("class", i < level ? "seg on" : "seg");
      svg.appendChild(p);
    }
    var t = document.createElementNS(NS, "text");
    t.setAttribute("x", "50");
    t.setAttribute("y", "52");
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("class", "gauge-value");
    t.textContent = String(level);
    svg.appendChild(t);
    return svg;
  }

  function renderLevelCard(data) {
    var a = data.assessment;
    var details = el("details", { className: "more" }, [el("summary", { text: "Details: the text read from the picture and each statement" })]);
    while (resultBox.firstChild) { details.appendChild(resultBox.firstChild); }
    var card = el("div", { className: "verdict level-card " + (LEVEL_TONE[a.level] || "insufficient") });
    card.appendChild(levelGauge(a.level));
    var body = el("div");
    body.appendChild(el("p", { className: "verdict-code", text: "Level " + a.level + " of 5" }));
    body.appendChild(el("h2", { text: a.name }));
    if (a.reasons && a.reasons.length) {
      body.appendChild(el("ul", { className: "level-reasons" }, a.reasons.map(function (r) { return el("li", { text: r }); })));
    }
    var tl = data.text_line;   // Option A (CAMPUS_VERIFY_PICTURE_TEXT_LINE): the typed-text verdict; display only
    if (tl && tl.line) {
      var tlLink = safeLink(tl.post_url, "open");
      body.appendChild(el("p", { className: "text-line" }, tlLink ? [document.createTextNode(tl.line + " "), tlLink] : [document.createTextNode(tl.line)]));
      if (tl.differs) { body.appendChild(el("p", { className: "note", text: "The level above comes from checking each statement separately; see Details." })); }
    }
    var found = data.image_signals && data.image_signals.photo_match;
    var photoLink = found ? safeLink(found.post_url, "Open the DYCI post the photo appears in") : null;
    if (photoLink) { body.appendChild(el("p", {}, [photoLink])); }
    card.appendChild(body);
    var box = [card];
    if (a.level === 3) {
      var unsure = el("div", { className: "panel level-unsure" });
      if (a.why_not_sure) { unsure.appendChild(el("p", { className: "note uncertain", text: "Why it can't confirm: " + a.why_not_sure })); }
      if (a.closest && a.closest.length) {
        unsure.appendChild(el("h3", { text: "Closest DYCI announcements (closest, not a match)" }));
        unsure.appendChild(el("ul", {}, a.closest.map(function (c) {
          var link = safeLink(c.url, "open");
          return el("li", {}, [el("span", { text: (c.date ? c.date + ": " : "") + c.text + " " }), link]);
        })));
      }
      if (a.check) { unsure.appendChild(el("p", { className: "note", text: a.check })); }
      box.push(unsure);
    }
    var how = el("details", { className: "more" }, [el("summary", { text: "How this works" }),
      el("p", { className: "note", text: a.explain })]);
    if (a.photo_note) { how.appendChild(el("p", { className: "note", text: a.photo_note })); }
    box.push(how, details);
    box.forEach(function (n) { resultBox.appendChild(n); });
  }

  imageForm.addEventListener("submit", function (event) {
    event.preventDefault();
    var file = imageInput.files && imageInput.files[0];
    var problem = imageProblem(file);
    imageError.hidden = true;
    if (problem) {
      imageError.textContent = problem;
      imageError.hidden = false;
      return;
    }
    hideStatus();
    resultBox.hidden = true;
    imageSubmit.disabled = true;
    showSteps(STEPS.picture);
    Api.verifyImage(file).then(function (data) {
      imageSubmit.disabled = false;
      stopSteps();
      renderImageResult(data);
    }).catch(function (err) {
      imageSubmit.disabled = false;
      stopSteps();
      showError(err);
    });
  });

  resetBtn.addEventListener("click", function () {
    form.reset();
    claimError.hidden = true;
    claimInput.removeAttribute("aria-invalid");
    hideStatus();
    resultBox.hidden = true;
    clear(resultBox);
    claimInput.focus();
  });

  // ---- layout: Text | URL (| picture) tabs, sidebar views, mobile menu ------------------------------------------------
  // Presentation only: the tabs choose which form is shown; every check still goes through the same API calls.
  var activeMode = "text";
  var tabList = $("mode-tabs");
  var urlField = $("url-field");
  var verifyForm = $("verify-form");
  var pictureSection = $("image-check");

  function selectMode(mode) {
    activeMode = mode;
    Array.prototype.forEach.call(tabList.querySelectorAll("[role=tab]"), function (t) {
      var on = t.getAttribute("data-mode") === mode;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
    });
    verifyForm.hidden = mode === "picture";
    if (pictureSection) { pictureSection.hidden = mode !== "picture" || !pictureTabAdded; }
    urlField.hidden = mode !== "url";
    if (mode !== "url") { urlInput.value = ""; }
    claimError.hidden = true;
  }

  tabList.addEventListener("click", function (event) {
    var tab = event.target.closest ? event.target.closest("[role=tab]") : null;
    if (tab) { selectMode(tab.getAttribute("data-mode")); }
  });
  tabList.addEventListener("keydown", function (event) {           // arrow keys move between tabs
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    var tabs = Array.prototype.slice.call(tabList.querySelectorAll("[role=tab]"));
    var i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    var next = tabs[(i + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    next.focus();
    selectMode(next.getAttribute("data-mode"));
  });

  var pictureTabAdded = false;
  function addPictureTab() {                                        // only when /health says the picture check is on
    if (pictureTabAdded) return;
    pictureTabAdded = true;
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    var rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    [["x", "3"], ["y", "5"], ["width", "18"], ["height", "14"], ["rx", "2"]].forEach(function (a) { rect.setAttribute(a[0], a[1]); });
    var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", "m3 16 5-5 4 4 3-3 6 6");
    svg.appendChild(rect);
    svg.appendChild(path);
    var tab = el("button", { className: "tab", attrs: { type: "button", role: "tab", id: "tab-picture",
      "aria-selected": "false", "data-mode": "picture", tabindex: "-1" } }, [svg, document.createTextNode("Image")]);
    tabList.insertBefore(tab, $("tab-url"));
    selectMode(activeMode);
  }

  var VIEW_TITLES = { check: "New check", dashboard: "Dashboard", guide: "New check", account: "Account",
                      users: "Users", sources: "Sources", signin: "Sign in" };
  var VIEWS = ["check", "dashboard", "account", "users", "sources", "signin"];
  var sidebar = $("sidebar"), scrim = $("scrim"), menuBtn = $("menu-btn");

  function setMenu(open) {
    sidebar.classList.toggle("open", open);
    scrim.hidden = !open;
    menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
  }
  menuBtn.addEventListener("click", function () { setMenu(!sidebar.classList.contains("open")); });
  scrim.addEventListener("click", function () { setMenu(false); });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && sidebar.classList.contains("open")) { setMenu(false); menuBtn.focus(); }
  });

  function showView(view) {
    if (typeof auth !== "undefined" && auth.mustSignIn && view !== "signin") { view = "signin"; }   // AUTH_REQUIRED
    var shown = view === "guide" ? "check" : view;
    document.body.classList.toggle("landing-mode", shown === "signin");
    VIEWS.forEach(function (v) { var s = $("view-" + v); if (s) { s.hidden = v !== shown; } });
    $("view-title").textContent = VIEW_TITLES[view] || "New check";
    Array.prototype.forEach.call(sidebar.querySelectorAll(".nav-item"), function (b) {
      var on = b.getAttribute("data-view") === view;
      b.classList.toggle("active", on);
      if (on) { b.setAttribute("aria-current", "page"); } else { b.removeAttribute("aria-current"); }
    });
    setMenu(false);
    if (view === "dashboard") { loadDashboard(); }
    if (view === "users") { loadUsers(); }
    if (view === "sources") { loadSources(); }
    if (view === "dashboard" && typeof sourcesState !== "undefined" && sourcesState.on) { loadSourcesAlert(); }
    if (view === "account" && typeof myChecks !== "undefined") { loadChecks(); }
    if (view === "guide") { $("guide").focus(); }
  }
  sidebar.addEventListener("click", function (event) {
    var item = event.target.closest ? event.target.closest(".nav-item[data-view]") : null;   // "Take the tour" has none
    if (item) { showView(item.getAttribute("data-view")); }
  });

  // ---- dashboard (GET /stats): counts only; "Contradicted" is shown under its own name, never as "fake" ----------------
  var OUTCOME_LABELS = {
    supported: "Supported", contradicted: "Contradicted", not_confirmed: "Not confirmed",
    withheld_language: "Evidence only (Tagalog/Taglish)", question: "Questions", could_not_read: "Could not read",
    level_5: "Level 5", level_4: "Level 4", level_3: "Level 3", level_2: "Level 2", level_1: "Level 1",
    no_text: "No readable text", other_page: "Another school's page"
  };
  var KIND_LABELS = { text: "Typed messages", link: "Links", picture: "Pictures" };

  function svgEl(tag, attrs) {
    var n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }

  function tile(label, value, sub, dark) {
    return el("div", { className: "tile" + (dark ? " dark" : "") }, [
      el("p", { className: "tile-label", text: label }),
      el("p", { className: "tile-value", text: value }),
      sub ? el("p", { className: "tile-sub", text: sub }) : null]);
  }

  function dailyChart(daily) {
    var W = 900, H = 240, pad = { l: 34, r: 10, t: 12, b: 28 };
    var max = 1;
    daily.forEach(function (d) { max = Math.max(max, d.text + d.link + d.picture); });
    var step = Math.max(1, Math.ceil(max / 4));
    var top = step * 4, bw = (W - pad.l - pad.r) / daily.length;
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, class: "chart", role: "img",
      "aria-label": "Verifications per day for the last " + daily.length + " days, split into typed messages, links and pictures" });
    for (var g = 0; g <= 4; g++) {
      var y = H - pad.b - (H - pad.t - pad.b) * g / 4;
      svg.appendChild(svgEl("line", { x1: pad.l, x2: W - pad.r, y1: y, y2: y, class: "grid-line" }));
      var t = svgEl("text", { x: pad.l - 6, y: y + 4, class: "axis", "text-anchor": "end" });
      t.textContent = String(step * g);
      svg.appendChild(t);
    }
    daily.forEach(function (d, i) {
      var x = pad.l + i * bw + bw * 0.2, w = bw * 0.6, base = H - pad.b;
      ["text", "link", "picture"].forEach(function (kind) {
        if (!d[kind]) return;
        var h = (H - pad.t - pad.b) * d[kind] / top;
        var r = svgEl("rect", { x: x, y: base - h, width: w, height: h, class: "bar " + kind });
        var tip = svgEl("title");
        tip.textContent = d.date + ": " + d[kind] + " " + KIND_LABELS[kind].toLowerCase();
        r.appendChild(tip);
        svg.appendChild(r);
        base -= h;
      });
      var lbl = svgEl("text", { x: x + w / 2, y: H - 9, class: "axis", "text-anchor": "middle" });
      var every = Math.ceil(daily.length / 10);                                 // at most ten date labels
      if (daily.length <= 7 || i % every === 0) {
        lbl.textContent = new Date(d.date + "T00:00:00").toLocaleDateString(undefined,
          daily.length <= 7 ? { weekday: "short" } : { month: "short", day: "numeric" });
        svg.appendChild(lbl);
      }
    });
    return svg;
  }

  function barList(title, rows, total) {
    var box = el("div", { className: "panel-card" }, [el("h2", { className: "panel-title", text: title })]);
    var list = el("ul", { className: "bar-list" });
    rows.forEach(function (r) {
      var fill = el("span", { className: "bar-fill " + (r[2] || "") });
      fill.style.transform = "scaleX(" + (total ? r[1] / total : 0) + ")";     // counts, drawn to scale; no percentages
      list.appendChild(el("li", {}, [
        el("span", { className: "bar-name", text: r[0] }),
        el("span", { className: "bar-track" }, [fill]),
        el("span", { className: "bar-count", text: total ? r[1] + " of " + total : String(r[1]) })]));
    });
    box.appendChild(list);
    return box;
  }

  function renderDashboard(s) {
    var body = $("dash-body");
    clear(body);
    var typed = s.outcomes.text, links = s.outcomes.link, pics = s.outcomes.picture;
    var sum = function (key) { return (typed[key] || 0) + (links[key] || 0); };
    var matched = s.matched.text + s.matched.link + s.matched.picture;
    var answered = s.total - sum("could_not_read") - (pics.could_not_read || 0);
    var tiles = el("div", { className: "tiles" }, [
      tile("Total verifications", String(s.total), "last " + s.days + " days" +
        (s.all_time ? " \u00b7 " + s.all_time.total + " since " + s.all_time.since : "")),
      tile("Supported", String(sum("supported")), "an announcement agrees"),
      tile("Contradicted", String(sum("contradicted")), "an announcement conflicts"),
      tile("Not confirmed", String(sum("not_confirmed")), "not settled, not the same as false"),
      tile("Pictures checked", String(s.checks.picture), "screenshots and photos"),
      tile("Coverage", answered ? matched + " of " + answered : "–",
        "checks where a DYCI post matched", true)]);
    body.appendChild(tiles);

    var trend = el("div", { className: "panel-card wide" }, [
      el("h2", { className: "panel-title", text: "Verifications in the last " + s.days + " days" }), dailyChart(s.daily),
      el("ul", { className: "chart-legend" }, ["text", "link", "picture"].map(function (k) {
        return el("li", {}, [el("span", { className: "swatch " + k, attrs: { "aria-hidden": "true" } }),
          document.createTextNode(KIND_LABELS[k])]);
      }))]);
    body.appendChild(trend);

    var grid = el("div", { className: "dash-grid" });
    grid.appendChild(barList("Typed messages and links by outcome",
      ["supported", "contradicted", "not_confirmed", "withheld_language", "question", "could_not_read"].map(function (k) {
        return [OUTCOME_LABELS[k], sum(k), k];
      }), s.checks.text + s.checks.link));
    grid.appendChild(barList("Pictures by level",
      ["level_5", "level_4", "level_3", "level_2", "level_1", "no_text", "other_page", "could_not_read"].map(function (k) {
        return [OUTCOME_LABELS[k], pics[k] || 0, "lvl-" + k];
      }), s.checks.picture));
    grid.appendChild(barList("Input type",
      ["text", "link", "picture"].map(function (k) { return [KIND_LABELS[k], s.checks[k], "kind-" + k]; }), s.total));
    body.appendChild(grid);
    body.appendChild(reportTable(s, sum));
    body.appendChild(el("p", { className: "help", text: s.note + " “Not confirmed” does not mean false, and " +
      "Tagalog/Taglish messages get evidence only, so these counts are not a count of fake news." }));
  }

  // Printed report only: the totals as a plain table, readable in black and white.
  function reportTable(s, sum) {
    var rows = [["Total verifications", s.total], ["Typed messages", s.checks.text], ["Links", s.checks.link],
      ["Pictures", s.checks.picture]];
    ["supported", "contradicted", "not_confirmed", "withheld_language", "question", "could_not_read"].forEach(function (k) {
      rows.push(["Messages and links: " + OUTCOME_LABELS[k], sum(k)]);
    });
    rows.push(["Checks where a DYCI post matched", s.matched.text + s.matched.link + s.matched.picture]);
    var tbody = el("tbody", {}, rows.map(function (r) {
      return el("tr", {}, [el("th", { text: r[0], attrs: { scope: "row" } }), el("td", { text: String(r[1]) })]);
    }));
    return el("div", { className: "print-only report-table" }, [el("h2", { className: "panel-title", text: "Totals" }),
      el("table", {}, [tbody])]);
  }

  function reportHeader(s) {
    var last = s.daily.length ? s.daily[s.daily.length - 1].date : "";
    var first = s.daily.length ? s.daily[0].date : "";
    $("report-period").textContent = "Period: last " + s.days + " days (" + first + " to " + last + ")";
    reportStamp();
  }

  function reportStamp() {                          // the printed time is the time of printing
    $("report-made").textContent = "Generated " + new Date().toLocaleString() +
      (auth.user ? " by " + (auth.user.name || auth.user.email) + " (" + auth.user.email + ")" : "");
  }

  function loadDashboard() {
    var status = $("dash-status");
    status.textContent = "Loading the counts…";
    status.hidden = false;
    Api.getStats(Number($("dash-range").value) || 7).then(function (s) {
      status.hidden = true;
      renderDashboard(s);
      reportHeader(s);
    }).catch(function () {
      status.textContent = "The counts couldn’t be loaded right now.";
    });
  }
  $("dash-refresh").addEventListener("click", loadDashboard);
  $("dash-print").addEventListener("click", function () {
    reportStamp();
    window.print();
  });
  $("dash-csv").addEventListener("click", function () {
    var btn = this, status = $("dash-csv-status");
    btn.disabled = true;
    status.textContent = "Preparing the spreadsheet…";
    Api.getStatsCsv(Number($("dash-range").value) || 7).then(function (f) {
      var url = URL.createObjectURL(new Blob([f.text], { type: "text/csv;charset=utf-8" }));
      var a = el("a", { attrs: { href: url, download: f.filename } });
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
      status.textContent = "Downloaded " + f.filename + ".";
    }, function (err) {
      status.textContent = err && err.message ? err.message : "The spreadsheet could not be made right now.";
    }).then(function () { btn.disabled = false; });
  });
  $("dash-range").addEventListener("change", loadDashboard);

  // ---- Sign in with Google, DYCI accounts only (CAMPUS_VERIFY_AUTH=google; /health says when it is on) -------------
  // Google's own button (Google Identity Services) gives the page a signed token; the server checks it and keeps the
  // session in an HttpOnly cookie. Nothing here decides who may sign in: the server does.
  var auth = { on: false, required: false, mustSignIn: false, user: null, gisLoaded: false, config: null, stats: false };

  function loadGoogleScript(done) {
    if (auth.gisLoaded) { done(); return; }
    var s = document.createElement("script");
    s.src = "https://accounts.google.com/gsi/client";
    s.async = true;
    s.onload = function () { auth.gisLoaded = true; done(); };
    s.onerror = function () { showSignInError("Google sign-in could not be loaded. Check your connection and reload the page."); };
    document.head.appendChild(s);
  }

  function showSignInError(text) {
    var e = $("signin-error");
    e.textContent = text;
    e.hidden = !text;
  }

  function renderGoogleButtons() {
    if (!window.google || !google.accounts || !auth.config) return;
    google.accounts.id.initialize({ client_id: auth.config.client_id, callback: onGoogleCredential,
                                    hd: auth.config.domain, ux_mode: "popup", auto_select: false });
    [$("google-btn"), $("signin-top")].forEach(function (box) {
      if (!box) return;
      clear(box);
      google.accounts.id.renderButton(box, { theme: "outline", size: box.id === "google-btn" ? "large" : "medium",
                                             text: "signin_with", shape: "pill" });
    });
  }

  function onGoogleCredential(response) {
    showSignInError("");
    Api.signInGoogle(response.credential).then(function (r) { signedIn(r.user); },
      function (err) { showSignInError(err && err.message ? err.message : "Sign-in failed. Please try again."); });
  }

  function signedIn(user) {
    auth.user = user;
    auth.mustSignIn = false;
    $("account-chip").hidden = false;
    $("account-name").textContent = user.name || user.email;
    $("signin-top").hidden = true;
    $("nav-account").hidden = false;
    $("nav-users").hidden = !user.is_admin;
    $("nav-sources").hidden = !(sourcesState.on && user.is_admin);
    if (sourcesState.on && user.is_admin) { loadSourcesAlert(); }
    $("nav-dashboard").hidden = !(auth.stats && user.is_admin);              // with sign-in on, admins only
    $("acct-name").textContent = user.name || user.email;
    $("acct-email").textContent = user.email;
    $("acct-avatar").textContent = initials(user.name || user.email);
    $("acct-role").textContent = user.is_admin ? "Administrator" : "Student";
    $("acct-role").classList.toggle("admin", !!user.is_admin);
    $("acct-since").textContent = user.created_at ? "Member since " + new Date(user.created_at).toLocaleDateString(
      undefined, { year: "numeric", month: "long", day: "numeric" }) : "";
    $("history-toggle").checked = !!user.history_enabled;
    showView("check");
    if (typeof tour !== "undefined") { offerTour(); }
  }

  function signedOut() {
    auth.user = null;
    myChecks.checks = [];                                // the next account on this page never sees these
    clear($("checks-list"));
    auth.mustSignIn = auth.required;
    $("account-chip").hidden = true;
    $("nav-account").hidden = true;
    $("nav-users").hidden = true;
    $("nav-sources").hidden = true;
    $("nav-dashboard").hidden = true;
    $("signin-top").hidden = auth.required;          // optional sign-in: a small button in the top bar
    loadGoogleScript(renderGoogleButtons);
    showView(auth.required ? "signin" : "check");
    if (typeof tour !== "undefined") { offerTour(); }
  }

  function initAuth(info) {
    auth.on = true;
    auth.required = !!info.required;
    auth.stats = !$("nav-dashboard").hidden;                                   // /health turned the dashboard on
    $("nav-dashboard").hidden = true;
    $("signin-domain").textContent = "@" + info.domain;
    Api.getAuthConfig().then(function (cfg) {
      auth.config = cfg;
      Api.getMe().then(function (r) { signedIn(r.user); }, function () { signedOut(); });
    }, function () { showSignInError("Sign-in is not available right now."); });
  }

  function doSignOut() {
    Api.signOut().then(signedOut, signedOut);
    if (window.google && google.accounts) { google.accounts.id.disableAutoSelect(); }
  }
  $("signout-btn").addEventListener("click", doSignOut);
  $("signout-btn-2").addEventListener("click", doSignOut);

  function initials(name) {
    var parts = String(name).replace(/@.*/, "").split(/[\s._-]+/).filter(Boolean);
    return ((parts[0] || "?").charAt(0) + (parts.length > 1 ? parts[parts.length - 1].charAt(0) : "")).toUpperCase();
  }

  // Deleting is verified twice: typing DELETE, then a fresh "Sign in with Google" by the same account (the server
  // checks the account and that the confirmation is under 5 minutes old).
  var deleteDialog = $("delete-dialog");

  function deleteError(text) {
    var e = $("delete-error");
    e.textContent = text || "";
    e.hidden = !text;
  }

  function closeDeleteDialog() {
    if (deleteDialog.open) { deleteDialog.close(); }
    if (window.google && google.accounts && auth.config) {          // back to the normal sign-in callback
      google.accounts.id.initialize({ client_id: auth.config.client_id, callback: onGoogleCredential,
                                      hd: auth.config.domain, ux_mode: "popup", auto_select: false });
    }
  }

  function onDeleteCredential(response) {
    deleteError("");
    Api.deleteAccount(response.credential, $("delete-confirm").value).then(function () {
      closeDeleteDialog();
      $("delete-status").textContent = "";
      signedOut();
    }, function (err) { deleteError(err && err.message ? err.message : "Could not delete the account. Please try again."); });
  }

  $("delete-account-btn").addEventListener("click", function () {
    $("delete-confirm").value = "";
    $("delete-google-btn").hidden = true;
    $("delete-google-hint").hidden = false;
    deleteError("");
    if (deleteDialog.showModal) { deleteDialog.showModal(); } else { deleteDialog.setAttribute("open", ""); }
    $("delete-confirm").focus();
  });
  $("delete-cancel").addEventListener("click", closeDeleteDialog);
  deleteDialog.addEventListener("cancel", function () { closeDeleteDialog(); });
  $("delete-confirm").addEventListener("input", function () {
    var ready = this.value.trim().toUpperCase() === "DELETE";
    $("delete-google-hint").hidden = ready;
    $("delete-google-btn").hidden = !ready;
    if (!ready) return;
    loadGoogleScript(function () {
      if (!window.google || !google.accounts || !auth.config) return;
      google.accounts.id.initialize({ client_id: auth.config.client_id, callback: onDeleteCredential,
                                      hd: auth.config.domain, ux_mode: "popup", auto_select: false });
      var box = $("delete-google-btn");
      clear(box);
      google.accounts.id.renderButton(box, { theme: "filled_black", size: "large", text: "continue_with", shape: "pill" });
    });
  });

  $("history-toggle").addEventListener("change", function () {
    var box = this, status = $("history-status");
    var saved = myChecks.checks.length, deleteSaved = false;
    if (!box.checked && saved > 0) {               // turning history off: the student decides about what is saved
      deleteSaved = window.confirm("History will be turned off. Also delete your " + saved + " saved check" +
        (saved === 1 ? "" : "s") + "?\n\nOK: delete them now.\nCancel: keep them (they are removed after 90 days).");
    }
    status.textContent = "Saving…";
    Api.setHistory(box.checked, deleteSaved).then(function (r) {
      auth.user = r.user;
      status.textContent = r.user.history_enabled ? "History is on. New checks are saved under My checks." :
        (deleteSaved ? "History is off and your saved checks were deleted." : "History is off. New checks are not saved.");
      if (myChecks.on) { loadChecks(); }
    }, function () { box.checked = !box.checked; status.textContent = "Could not save. Please try again."; });
  });

  // ---- My checks (GET /account/checks; CAMPUS_VERIFY_HISTORY): the student's own entries, shown as they were then --
  var myChecks = { on: false, checks: [] };
  var CHECK_DOTS = { supported: "supported", contradicted: "contradicted", not_confirmed: "insufficient",
                     could_not_read: "insufficient" };
  var KIND_NAMES = { text: "Message", link: "Link", picture: "Picture" };

  function checkItem(c) {
    var label = (typeof OUTCOME_LABELS !== "undefined" && OUTCOME_LABELS[c.outcome]) || c.outcome;
    var head = el("div", { className: "check-head" }, [
      el("span", { className: "check-answer" }, [el("span", { className: "dot " + (CHECK_DOTS[c.outcome] || "info"),
        attrs: { "aria-hidden": "true" } }), document.createTextNode(label)]),
      el("span", { className: "check-when", text: (KIND_NAMES[c.kind] || c.kind) + " \u00b7 " + fmtWhen(c.at) })]);
    var body = [head];
    if (c.message) { body.push(el("p", { className: "check-message", text: c.message })); }
    if (c.kind === "picture") { body.push(el("p", { className: "help", text: "Picture check (the picture is not kept)." })); }
    if (c.url) { body.push(el("p", { className: "help check-url", text: "Link: " + c.url })); }
    if (c.post_excerpt) {
      body.push(el("p", { className: "help", text: "DYCI post" + (c.post_date ? " (" + c.post_date + ")" : "") + ": " +
        c.post_excerpt }));
    }
    var actions = el("div", { className: "check-actions" });
    if (c.message && c.kind !== "picture") {
      var again = el("button", { className: "secondary small", text: "Check again", attrs: { type: "button" } });
      again.addEventListener("click", function () { checkAgain(c); });
      actions.appendChild(again);
    }
    var del = el("button", { className: "secondary small", text: "Delete", attrs: { type: "button",
      "aria-label": "Delete this saved check" } });
    del.addEventListener("click", function () {
      del.disabled = true;
      Api.deleteCheck(c.id).then(loadChecks, function () { del.disabled = false; $("checks-status").textContent = "Could not delete. Please try again."; });
    });
    actions.appendChild(del);
    body.push(actions);
    return el("li", { className: "check-item" }, body);
  }

  function loadChecks() {
    if (!myChecks.on || !auth.user) return;
    var status = $("checks-status"), list = $("checks-list");
    status.textContent = "Loading…";
    Api.listChecks().then(function (r) {
      myChecks.checks = r.checks;
      clear(list);
      r.checks.forEach(function (c) { list.appendChild(checkItem(c)); });
      $("checks-clear").hidden = r.checks.length === 0;
      status.textContent = r.checks.length ? r.checks.length + " saved (newest " + r.keep_last + " kept, " + r.keep_days +
        " days each)." : (r.history_enabled ? "Nothing saved yet. Your next checks will appear here." :
        "Turn on \u201cKeep a history of my checks\u201d above to save your checks here.");
    }, function () { status.textContent = "Could not load your saved checks."; });
  }

  function checkAgain(c) {                          // today's answer: the message goes through the check again
    showView("check");
    selectMode(c.kind === "link" && c.url ? "url" : "text");
    claimInput.value = c.message;
    if (c.kind === "link" && c.url) { urlInput.value = c.url; }
    if (form.requestSubmit) { form.requestSubmit(); } else { form.dispatchEvent(new Event("submit", { cancelable: true })); }
  }

  $("checks-clear").addEventListener("click", function () {
    if (!window.confirm("Delete all your saved checks? This cannot be undone.")) return;
    Api.clearChecks().then(loadChecks, function () { $("checks-status").textContent = "Could not delete. Please try again."; });
  });

  function initHistory() {
    myChecks.on = true;
    $("checks-card").hidden = false;
  }

  // ---- Users (administrators only; the server refuses everyone else) ------------------------------------------------
  function fmtWhen(iso) {
    if (!iso) return "";
    var d = new Date(iso);
    return isNaN(d) ? iso : d.toLocaleString();
  }

  function loadUsers() {
    var status = $("users-status"), body = $("users-body");
    status.textContent = "Loading…";
    Api.listUsers().then(function (r) {
      status.textContent = r.users.length + " account" + (r.users.length === 1 ? "" : "s");
      clear(body);
      var table = el("table");
      var head = el("tr");
      ["Name", "Email", "Last sign-in", "Role", "History", "Status", ""].forEach(function (h) { head.appendChild(el("th", { text: h })); });
      table.appendChild(el("thead", {}, [head]));
      var rows = el("tbody");
      r.users.forEach(function (u) {
        var tr = el("tr");
        tr.appendChild(el("td", { text: u.name }));
        tr.appendChild(el("td", { text: u.email }));
        tr.appendChild(el("td", { text: fmtWhen(u.last_login_at) }));
        tr.appendChild(el("td", { text: u.is_admin ? "Administrator" : "Student" }));
        tr.appendChild(el("td", { text: u.history_enabled ? "On" : "Off" }));
        tr.appendChild(el("td", { text: u.active ? "Active" : "Disabled" }));
        var cell = el("td");
        if (!(auth.user && u.email === auth.user.email)) {
          var b = el("button", { className: "secondary small", text: u.active ? "Disable" : "Enable", attrs: { type: "button" } });
          b.addEventListener("click", function () {
            b.disabled = true;
            Api.setUserActive(u.id, !u.active).then(loadUsers, function (err) {
              b.disabled = false;
              status.textContent = err && err.message ? err.message : "Could not change the account.";
            });
          });
          cell.appendChild(b);
        }
        tr.appendChild(cell);
        rows.appendChild(tr);
      });
      table.appendChild(rows);
      body.appendChild(table);
    }, function (err) { status.textContent = err && err.message ? err.message : "Could not load the accounts."; });
  }
  $("users-refresh").addEventListener("click", loadUsers);

  // ---- Sources (administrators; CAMPUS_VERIFY_SOURCES): Facebook pages and their token status, never a token ------------
  var sourcesState = { on: false, data: null };
  var TOKEN_STATUS = {
    ok: ["Working", "supported"], expiring: ["Expiring soon", "insufficient"], invalid: ["Token not working", "contradicted"],
    no_access: ["Cannot read page", "contradicted"], rate_limited: ["Meta busy, retry later", "insufficient"],
    unreachable: ["Meta unreachable", "insufficient"], unchecked: ["Not checked yet", "info"]
  };
  var TIER_NAMES = { official: "Official", recognized: "Recognized", unverified: "Unverified" };

  function fetchSummary(f) {
    if (!f) return "Not fetched yet";
    if (f.status !== "ok") return f.error || f.reason || f.status;
    var parts = [(f.added || 0) + " new"];
    if (f.edited) parts.push(f.edited + " edited");
    if (f.retired) parts.push(f.retired + " removed on Facebook");
    if (f.reserved_for_final_evaluation) parts.push(f.reserved_for_final_evaluation + " kept for the final evaluation");
    return parts.join(", ");
  }

  function attentionText(n) {
    return n + " source" + (n === 1 ? " needs" : "s need") + " attention: a token stopped working or expires soon.";
  }

  function loadSourcesAlert() {
    Api.listSources().then(function (r) {
      var n = r.needs_attention.length, badge = $("nav-sources-badge"), box = $("dash-sources-alert");
      badge.hidden = !n;
      badge.textContent = n ? String(n) : "";
      clear(box);
      box.hidden = !n;
      if (n) {
        var open = el("button", { className: "secondary small", text: "Open Sources", attrs: { type: "button" } });
        open.addEventListener("click", function () { showView("sources"); });
        box.appendChild(el("p", { text: attentionText(n) }));
        box.appendChild(open);
      }
    }, function () { /* the dashboard still works without it */ });
  }

  // Fetching is locked until the final-evaluation cutoff is recorded, once (docs/plans/past_and_new_posts_plan.md).
  function cutoffForm() {
    var today = new Date(), pad = function (n) { return (n < 10 ? "0" : "") + n; };
    var input = el("input", { attrs: { type: "date", id: "cutoff-day", max: today.getFullYear() + "-" +
      pad(today.getMonth() + 1) + "-" + pad(today.getDate()) } });
    input.value = input.getAttribute("max");
    var button = el("button", { className: "primary small", text: "Record cutoff", attrs: { type: "button" } });
    var note = el("p", { className: "help", attrs: { role: "status" } });
    button.addEventListener("click", function () {
      if (!input.value) { note.textContent = "Choose a date."; return; }
      if (!window.confirm("Record " + input.value + " as the final-evaluation cutoff?\n\nThis is done once and " +
        "cannot be changed. The checker as it is right now is saved as the version the final evaluation will " +
        "run, and fetching is unlocked.")) return;
      button.disabled = true;
      note.textContent = "Recording\u2026";
      Api.recordCutoff(input.value).then(function () { loadSources(); }, function (err) {
        button.disabled = false;
        note.textContent = err && err.message ? err.message : "That did not work. Please try again.";
      });
    });
    return el("div", { className: "cutoff-form" }, [
      el("p", { text: "Fetching is locked until the final-evaluation cutoff is recorded. Choose the day the page " +
        "token arrived. This is done once: the checker as it is now is saved as the version the final evaluation " +
        "runs, and fetching is unlocked." }),
      el("label", { text: "Cutoff date", attrs: { for: "cutoff-day" } }), input, button, note]);
  }

  function sourceAction(s, action, button, status) {
    button.disabled = true;
    status.textContent = { check: "Asking Meta…", fetch: "Fetching posts…", remove: "Removing…" }[action];
    Api.sourceAction(s.key, action).then(function (r) {
      if (action === "fetch") {
        var res = r.result[s.key];
        status.textContent = s.name + ": " + fetchSummary(res) + ".";
      } else {
        status.textContent = "";
      }
      loadSources();
    }, function (err) {
      button.disabled = false;
      status.textContent = err && err.message ? err.message : "That did not work. Please try again.";
    });
  }

  function sourceRow(s, status) {
    var st = TOKEN_STATUS[s.token_status] || [s.token_status, "info"];
    var tokenInfo = "…" + (s.token_hint || "????") + " \u00b7 saved " + fmtWhen(s.token_set_at) +
      (s.token_expires_at ? " \u00b7 " + (s.token_expires_at === "never" ? "does not expire" :
        "expires " + fmtWhen(s.token_expires_at)) : "");
    var actions = el("div", { className: "row-actions" });
    function action(label, fn, cls) {
      var b = el("button", { className: (cls || "secondary") + " small", text: label, attrs: { type: "button" } });
      b.addEventListener("click", function () { fn(b); });
      actions.appendChild(b);
      return b;
    }
    action("Check", function (b) { sourceAction(s, "check", b, status); });
    if (s.include) { action("Fetch now", function (b) { sourceAction(s, "fetch", b, status); }); }
    action("Replace token", function () { openTokenDialog(s); });
    action(s.include ? "Turn off" : "Turn on", function (b) {
      b.disabled = true;
      Api.updateSource(s.key, { include: !s.include }).then(loadSources, function (err) {
        b.disabled = false;
        status.textContent = err && err.message ? err.message : "Could not change the page.";
      });
    });
    action("Remove", function (b) {
      if (!window.confirm("Remove " + s.name + " as a source? Its token is deleted. Posts already collected stay " +
        "in the archive.")) return;
      sourceAction(s, "remove", b, status);
    }, "secondary danger");
    return el("tr", { className: s.include ? "" : "muted-row" }, [
      el("td", {}, [el("strong", { text: s.name }), el("div", { className: "help", text: "Page ID " + s.page_id +
        (s.include ? "" : " \u00b7 turned off") })]),
      el("td", { text: TIER_NAMES[s.tier] || s.tier }),
      el("td", {}, [el("span", { className: "check-answer" }, [el("span", { className: "dot " + st[1],
        attrs: { "aria-hidden": "true" } }), document.createTextNode(st[0])]),
        el("div", { className: "help", text: tokenInfo }),
        s.token_status !== "ok" && s.token_detail ? el("div", { className: "help", text: s.token_detail }) : null]),
      el("td", {}, [el("div", { text: fetchSummary(s.last_fetch) }),
        el("div", { className: "help", text: (s.last_fetch_at ? fmtWhen(s.last_fetch_at) + " \u00b7 " : "") +
          s.archived_posts + " post" + (s.archived_posts === 1 ? "" : "s") + " collected" }),
        pastPosts(s)]),
      el("td", {}, [actions])]);
  }

  // Past-posts fetching (CAMPUS_VERIFY_SOURCES_BACKFILL_MINUTES): how far back this page has been read.
  function pastPosts(s) {
    if (!s.backfill_at) return null;
    var back = s.backfill_until ? new Date(Number(s.backfill_until) * 1e3).toISOString().slice(0, 10) : null;
    var state = s.backfill_done ? "finished" : (s.backfill_note || "going").indexOf("paused") === 0 ? "paused" : "going";
    return el("div", { className: "help", text: "Past posts: " + state + (back ? ", back to " + back : "") + " \u00b7 " +
      s.backfill_added + " added" + (s.backfill_note && s.backfill_note !== "going" ? " (" + s.backfill_note + ")" : "") });
  }

  // Fix B (CAMPUS_VERIFY_FETCH_STATUS): the live fetch that feeds the checker, with plain warnings. Hidden when off.
  function loadFetchStatus(box) {
    if (!Api.fetchStatus) return;
    Api.fetchStatus().then(function (s) {
      var a = s.auto_fetch, sv = s.served || {};
      box.appendChild(el("h3", { text: "Live fetch (test page → checker)" }));
      box.appendChild(el("p", { className: "help", text: (a.enabled ? "Every " + a.interval_minutes + " min" : "Off") +
        " · last run " + (a.last_run || "not yet") + " · last success " + (a.last_success || "not since the checker started") +
        " · serving " + (sv.posts == null ? "?" : sv.posts) + " posts" + (sv.newest_post ? ", newest " + sv.newest_post : "") }));
      (s.warnings || []).forEach(function (w) { box.appendChild(el("p", { className: "note", text: "⚠ " + w })); });
      if (!(s.warnings || []).length) box.appendChild(el("p", { className: "help", text: "✓ No problems seen." }));
      var rows = (s.recent || []).slice(0, 8).map(function (e) {
        var what = e.ok ? (e.added || 0) + " new" + (e.edited ? ", " + e.edited + " edited" : "") +
          (e.retired ? ", " + e.retired + " removed" : "") + (e.skipped_no_text ? ", " + e.skipped_no_text + " skipped (no text)" : "")
          : "failed: " + (e.error || "");
        return el("li", { text: e.at + " — " + what });
      });
      if (rows.length) box.appendChild(el("ul", { className: "help" }, rows));
      box.appendChild(el("p", { className: "help", text: s.note }));
    }, function () { box.hidden = true; });
  }

  function loadSources() {
    var status = $("sources-status"), body = $("sources-body");
    status.textContent = "Loading…";
    Api.listSources().then(function (r) {
      sourcesState.data = r;
      status.textContent = "";
      $("sources-summary").textContent = r.sources.length + " page" + (r.sources.length === 1 ? "" : "s") +
        " \u00b7 " + (r.fetch_minutes ? "fetched every " + r.fetch_minutes + " min" : "fetched only with Fetch now") +
        (r.backfill_minutes ? " \u00b7 past posts back to " + r.backfill_from : "") +
        " \u00b7 tokens encrypted (" + (r.encryption === "windows-dpapi" ? "Windows" : "key") + ")";
      var alert = $("sources-alert");
      clear(alert);
      var notes = [];
      if (r.needs_attention.length) { notes.push(attentionText(r.needs_attention.length)); }
      notes.forEach(function (t) { alert.appendChild(el("p", { text: t })); });
      if (!r.cutoff) { alert.appendChild(cutoffForm()); }
      alert.hidden = !notes.length && !!r.cutoff;
      clear(body);
      if (!r.sources.length) {
        body.appendChild(el("p", { className: "help", text: "No pages yet. Add one below." }));
      } else {
        var head = el("tr");
        ["Page", "Trust level", "Token", "Last fetch", ""].forEach(function (h) { head.appendChild(el("th", { text: h })); });
        body.appendChild(el("table", { className: "sources-table" }, [el("thead", {}, [head]),
          el("tbody", {}, r.sources.map(function (s) { return sourceRow(s, status); }))]));
      }
      body.appendChild(el("p", { className: "help", text: "Collected posts go to the archive. The checker starts " +
        "answering from them once the served collection is built from the archive (scale plan S3)." +
        "" }));
      var live = el("div", { className: "live-fetch" });
      body.appendChild(live);
      loadFetchStatus(live);
      if (r.frozen) {
        body.appendChild(el("p", { className: "help", text: "Final-evaluation cutoff: " + r.frozen.cutoff_date +
          (r.frozen.frozen_tag ? ". The checker as it was then is kept as version " + r.frozen.frozen_tag +
            (r.frozen.frozen_commit ? " (" + r.frozen.frozen_commit.slice(0, 7) + ")" : "") +
            "; the final evaluation runs that version." : ".") }));
      }
      var log = $("sources-log");
      clear(log);
      r.activity.forEach(function (a) {
        log.appendChild(el("li", {}, [el("span", { className: "check-when", text: fmtWhen(a.at) + " \u00b7 " + a.by }),
          el("div", { text: a.action + " \u00b7 " + a.source + (a.detail ? " \u00b7 " + a.detail : "") })]));
      });
      if (!r.activity.length) { log.appendChild(el("li", { className: "help", text: "Nothing yet." })); }
      var badge = $("nav-sources-badge");
      badge.hidden = !r.needs_attention.length;
      badge.textContent = r.needs_attention.length ? String(r.needs_attention.length) : "";
    }, function (err) { status.textContent = err && err.message ? err.message : "Could not load the sources."; });
  }
  $("sources-refresh").addEventListener("click", loadSources);

  $("add-source-form").addEventListener("submit", function (event) {
    event.preventDefault();
    var status = $("add-source-status"), tokenBox = $("source-token");
    var button = this.querySelector("button[type=submit]");
    button.disabled = true;
    status.textContent = "Asking Meta…";
    Api.addSource($("source-page-id").value.trim(), $("source-tier").value, tokenBox.value.trim()).then(function (r) {
      tokenBox.value = "";
      $("source-page-id").value = "";
      status.textContent = "Added " + r.source.name + ".";
      loadSources();
    }, function (err) {
      status.textContent = err && err.message ? err.message : "Could not add the page.";
    }).then(function () { button.disabled = false; });
  });

  var tokenDialog = $("token-dialog"), tokenFor = null;
  function openTokenDialog(s) {
    tokenFor = s;
    $("token-for").textContent = s.name + " (page ID " + s.page_id + "). The old token is replaced only if Meta " +
      "accepts the new one.";
    $("token-input").value = "";
    $("token-error").hidden = true;
    if (tokenDialog.showModal) { tokenDialog.showModal(); } else { tokenDialog.setAttribute("open", ""); }
    $("token-input").focus();
  }
  function closeTokenDialog() {
    $("token-input").value = "";
    if (tokenDialog.open) { tokenDialog.close(); }
  }
  $("token-cancel").addEventListener("click", closeTokenDialog);
  tokenDialog.addEventListener("cancel", function () { $("token-input").value = ""; });
  $("token-form").addEventListener("submit", function (event) {
    event.preventDefault();
    var error = $("token-error"), button = this.querySelector("button[type=submit]");
    button.disabled = true;
    error.hidden = true;
    Api.replaceSourceToken(tokenFor.key, $("token-input").value.trim()).then(function () {
      closeTokenDialog();
      loadSources();
    }, function (err) {
      error.textContent = err && err.message ? err.message : "Could not replace the token.";
      error.hidden = false;
    }).then(function () { button.disabled = false; });
  });

  // ---- guided tour (CAMPUS_VERIFY_TOUR; display only) ----------------------------------------------------------------
  // A short first-visit tour: the page dims, one part is lit, a card explains it. Next / Back / Skip, Esc to close,
  // arrow keys to move. Signed-in accounts see it once by themselves (remembered in the account); visitors who are
  // not signed in get a small banner instead, because nothing is stored in the browser.
  var tour = { on: false, i: 0, steps: [], opener: null, bannerClosed: false };

  function exampleVerdicts() {
    var rows = [["supported", "Supported", "A DYCI post agrees with the message."],
                ["contradicted", "Contradicted", "A DYCI post says something different."],
                ["insufficient", "Not confirmed", "No DYCI post settles it yet. This does not mean it is false."]];
    var list = el("ul", { className: "tour-example legend" });
    rows.forEach(function (r) {
      list.appendChild(el("li", {}, [el("span", { className: "dot " + r[0], attrs: { "aria-hidden": "true" } }),
        el("div", {}, [el("h3", { text: r[1] }), el("p", { text: r[2] })])]));
    });
    return list;
  }

  // The five picture levels, as named in src/image/levels.py and explained in results/round2/legitimacy_levels_spec.md.
  function exampleLevels() {
    var rows = [[5, "Matches DYCI", "Every statement read from the picture matches a DYCI post, and none conflicts."],
                [4, "Mostly matches", "Some statements match a DYCI post and none conflicts; the rest couldn\u2019t be checked."],
                [3, "Can\u2019t confirm", "Nothing clearly matches or conflicts. Often the text was hard to read, or DYCI " +
                    "hasn\u2019t posted about it. This does not mean it is fake."],
                [2, "Sketchy", "It looks official, but something in it doesn\u2019t fit DYCI\u2019s posts. Check " +
                    "DYCI\u2019s page before sharing."],
                [1, "Conflicts with DYCI", "A DYCI post clearly says something different."]];
    var list = el("ul", { className: "tour-example legend" });
    rows.forEach(function (r) {
      list.appendChild(el("li", {}, [el("span", { className: "dot " + LEVEL_TONE[r[0]], attrs: { "aria-hidden": "true" } }),
        el("div", {}, [el("h3", { text: "Level " + r[0] + ": " + r[1] }), el("p", { text: r[2] })])]));
    });
    return list;
  }

  function exampleEvidence() {
    return el("div", { className: "tour-example post-example" }, [
      el("p", { className: "tour-tag", text: "Example" }),
      el("p", { className: "post-meta", text: "DYCI Advisory · 20 August 2026 · Open the original post" }),
      el("q", { text: "There will be NO CLASSES AT ALL LEVELS tomorrow, August 21, 2026 …" }),
      el("p", { className: "note date-note", text: "Notes like this one explain dates, holidays or rumours. Read them." })]);
  }

  function tourSteps() {
    var small = window.matchMedia && window.matchMedia("(max-width: 860px)").matches;
    return [
      { title: "Check before you share",
        text: "Heard something about classes, events or announcements? This checker compares it with DYCI’s own " +
              "official posts and tells you what they say." },
      { target: "mode-tabs", title: "Three ways to check",
        text: "Type what you heard, upload a screenshot or photo of an announcement, or paste a link to a post." },
      { target: "claim", title: "Type it the way you heard it",
        text: "English, Tagalog or Taglish all work, for example “walang pasok bukas?” or “The midterms " +
              "are on October 5.” One message at a time works best." },
      { target: "submit-btn", title: "Press Check claim",
        text: "The answer takes a few seconds. Nothing you type is stored." },
      { title: "What the answer means", body: exampleVerdicts,
        text: "Every answer is one of these three." },
      { title: "Always read the evidence", body: exampleEvidence,
        text: "Under the answer you see the official post it came from, with its date and a link. Open it to be sure." },
      pictureTabAdded ? { title: "Pictures get a level from 1 to 5", body: exampleLevels,
        text: "A screenshot or photo is summed up as one level, from what its text and photo match. The level is " +
              "a summary of those checks, not a percentage." } : null,
      { target: small ? "menu-btn" : "guide", title: "Help is always here",
        text: small ? "Open the menu for your account, About (with the Terms of Use), and this tour again."
                    : "This card explains the answers any time. The menu has About, your account, and “Take the tour”." }
    ].filter(Boolean);
  }

  function placeTour() {
    var step = tour.steps[tour.i], spot = $("tour-spot"), card = $("tour-card");
    var target = step.target ? $(step.target) : null;
    if (target && target.offsetParent === null) { target = null; }            // hidden on this screen: centre the card
    card.classList.toggle("centered", !target);
    spot.hidden = !target;
    if (!target) { card.style.top = ""; card.style.left = ""; return; }
    var r = target.getBoundingClientRect(), pad = 8;
    spot.style.top = (r.top - pad) + "px";
    spot.style.left = (r.left - pad) + "px";
    spot.style.width = (r.width + pad * 2) + "px";
    spot.style.height = (r.height + pad * 2) + "px";
    if (window.matchMedia && window.matchMedia("(max-width: 860px)").matches) {    // phones: a bottom panel
      card.style.top = ""; card.style.left = "";
      return;
    }
    var cw = card.offsetWidth, ch = card.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
    var top = r.bottom + 16 + ch <= vh ? r.bottom + 16 : Math.max(16, r.top - ch - 16);
    var left = Math.min(Math.max(16, r.left), vw - cw - 16);
    card.style.top = top + "px";
    card.style.left = left + "px";
  }

  function showTourStep(i) {
    tour.i = Math.max(0, Math.min(i, tour.steps.length - 1));
    var step = tour.steps[tour.i];
    $("tour-step").textContent = (tour.i + 1) + " of " + tour.steps.length;
    $("tour-title").textContent = step.title;
    var body = $("tour-body");
    clear(body);
    body.appendChild(el("p", { text: step.text }));
    if (step.body) { body.appendChild(step.body()); }
    var dots = $("tour-dots");
    clear(dots);
    tour.steps.forEach(function (_, k) { dots.appendChild(el("span", { className: k === tour.i ? "on" : "" })); });
    $("tour-back").hidden = tour.i === 0;
    $("tour-next").textContent = tour.i === tour.steps.length - 1 ? "Finish" : "Next";
    var target = step.target ? $(step.target) : null;
    if (target && target.scrollIntoView) { target.scrollIntoView({ block: "center", behavior: "auto" }); }
    placeTour();
    $("tour-card").focus();
  }

  function startTour() {
    if (typeof auth !== "undefined" && auth.mustSignIn) return;               // sign-in first when it is required
    showView("check");
    setMenu(false);
    tour.opener = document.activeElement;
    tour.steps = tourSteps();
    $("tour").hidden = false;
    document.body.classList.add("touring");
    showTourStep(0);
  }

  function endTour() {
    $("tour").hidden = true;
    document.body.classList.remove("touring");
    $("tour-banner").hidden = true;
    tour.bannerClosed = true;
    if (typeof auth !== "undefined" && auth.user && !auth.user.tour_seen) {
      auth.user.tour_seen = true;
      Api.setTourSeen().catch(function () {});
    }
    if (tour.opener && tour.opener.focus) { tour.opener.focus(); }
  }

  $("tour-next").addEventListener("click", function () {
    if (tour.i >= tour.steps.length - 1) { endTour(); } else { showTourStep(tour.i + 1); }
  });
  $("tour-back").addEventListener("click", function () { showTourStep(tour.i - 1); });
  $("tour-skip").addEventListener("click", endTour);
  $("tour-start").addEventListener("click", startTour);
  $("tour-banner-start").addEventListener("click", startTour);
  $("tour-banner-close").addEventListener("click", function () { $("tour-banner").hidden = true; tour.bannerClosed = true; });
  document.addEventListener("keydown", function (event) {
    if ($("tour").hidden) return;
    if (event.key === "Escape") { event.preventDefault(); endTour(); }
    else if (event.key === "ArrowRight") { $("tour-next").click(); }
    else if (event.key === "ArrowLeft" && tour.i > 0) { showTourStep(tour.i - 1); }
    else if (event.key === "Tab") {                                            // keep the keyboard inside the card
      var f = Array.prototype.filter.call($("tour-card").querySelectorAll("button"), function (b) { return !b.hidden; });
      if (!f.length) return;
      if (event.shiftKey && document.activeElement === f[0]) { event.preventDefault(); f[f.length - 1].focus(); }
      else if (!event.shiftKey && document.activeElement === f[f.length - 1]) { event.preventDefault(); f[0].focus(); }
    }
  });
  window.addEventListener("resize", function () { if (!$("tour").hidden) placeTour(); });

  function initTour() {
    tour.on = true;
    $("nav-tour").hidden = false;
    offerTour();
  }

  // Signed in and never finished: start by itself. Not signed in: the banner (nothing is remembered in the browser).
  function offerTour() {
    if (!tour.on || !$("tour").hidden) return;
    if (typeof auth !== "undefined" && auth.user) {
      $("tour-banner").hidden = true;
      if (!auth.user.tour_seen) { startTour(); }
    } else if (!(typeof auth !== "undefined" && auth.mustSignIn)) {
      $("tour-banner").hidden = tour.bannerClosed;
    }
  }

  // ---- startup: GET /health -----------------------------------------------------
  // Retried for up to a minute: a host that sleeps when unused can take that long to start. The status line is
  // optional (the page works without it), and its wording is for students, not developers.

  var STATUS = {
    starting: "Starting the checker… this can take up to a minute.",
    ready: "Checker ready.",
    down: "The checker can't be reached right now. Please try again in a few minutes."
  };
  var STATUS_RETRY_MS = 5000, STATUS_GIVE_UP_MS = 60000, STATUS_SLOW_MS = 3000;
  var statusEl = $("api-status");

  function setStatus(text, state) {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.classList.toggle("pending", state === "pending");
    statusEl.classList.toggle("bad", state === "bad");
  }

  var startedAt = Date.now();
  var slowTimer = setTimeout(function () { setStatus(STATUS.starting, "pending"); }, STATUS_SLOW_MS);
  function checkHealth() {
    Api.getHealth().then(function (h) {
      clearTimeout(slowTimer);
      var imageCheck = $("image-check");
      if (h.image_check === true && imageCheck) { imageCheck.hidden = false; }   // Image Check only when the API enables it
      if (h.image_check === true) { addPictureTab(); }
      if (h.image_check === true && $("guide-levels")) { $("guide-levels").hidden = false; }   // the five picture levels
      if (h.stats === true) { $("nav-dashboard").hidden = false; }              // Dashboard only when counting is on
      if (h.stats === true && window.location.hash === "#dashboard") { showView("dashboard"); }   // deep link
      if (h.ui_extras === true) { uiExtras = true; addExamples(); }             // CAMPUS_VERIFY_UI_EXTRAS
      if (h.auth && h.auth.provider === "google") { initAuth(h.auth); }         // CAMPUS_VERIFY_AUTH=google
      if (h.tour === true) { initTour(); }                                       // CAMPUS_VERIFY_TOUR
      if (h.history === true) { initHistory(); }                                 // CAMPUS_VERIFY_HISTORY
      if (h.sources === true) { sourcesState.on = true; }                        // CAMPUS_VERIFY_SOURCES (admins)
      setStatus(STATUS.ready, "ok");
    }).catch(function () {
      clearTimeout(slowTimer);
      if (Date.now() - startedAt < STATUS_GIVE_UP_MS) {
        setStatus(STATUS.starting, "pending");
        setTimeout(checkHealth, STATUS_RETRY_MS);
      } else {
        setStatus(STATUS.down, "bad");
      }
    });
  }
  checkHealth();
})();

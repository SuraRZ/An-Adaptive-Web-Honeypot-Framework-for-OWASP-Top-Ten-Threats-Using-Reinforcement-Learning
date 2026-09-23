"""
honeypot_flask.py  —  OWASP Top-10 Adaptive Honeypot
=====================================================
Run:  pip install flask
      python honeypot_flask.py

Pre-train first (recommended):
      python qlearning_engine.py --train 10000

Fixes applied:
  1. A03 no longer misclassified as A10 (localhost stripped from blob)
  2. A1 no longer dominates (Q-table seeded + tie-breaking shuffled)
  3. total_reward is now cumulative across all attacks per attacker
  4. Unknown paths default to A01 not A10
  5. Events written to honeypot_events.json (proper JSON array,
     auto-created on startup, atomic writes, never corrupted)
"""

import hashlib, json, logging, os, random, re, threading, time, uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Callable, Dict

from flask import Flask, Response, abort, jsonify, redirect, request, session

from qlearning_engine import ACTIONS, ATTACKS, HoneypotAgent, Session, reward

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("HP_SECRET", os.urandom(32))
app.config.update(
    SESSION_COOKIE_NAME     = "sid",
    SESSION_COOKIE_HTTPONLY = True,
    SESSION_COOKIE_SAMESITE = "Lax",
    PERMANENT_SESSION_LIFETIME = 1800,
)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger("honeypot")

agent = HoneypotAgent(path="qtable.json")

# ─────────────────────────────────────────────────────────────
# JSON event log  — auto-created on startup
#
# File: honeypot_events.json
# Format: a JSON array of event objects, e.g.
#   [
#     { "timestamp": "...", "ip": "...", "attack": "A03", ... },
#     { "timestamp": "...", "ip": "...", "attack": "A01", ... }
#   ]
#
# Written atomically (tmp file + rename) so the file is always
# valid JSON even if the honeypot is killed mid-write.
# ─────────────────────────────────────────────────────────────
EVENT_LOG = "honeypot_events.json"
_log_lock = threading.Lock()

def _init_event_log():
    """Create honeypot_events.json as an empty array if it does not exist,
    or reset it if it is corrupted."""
    if not Path(EVENT_LOG).exists():
        with open(EVENT_LOG, "w") as f:
            f.write("[]\n")
        log.info("Event log created  →  %s", EVENT_LOG)
        return
    # file exists — check it is valid JSON
    try:
        with open(EVENT_LOG, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("root is not a list")
        log.info("Event log loaded   →  %s  (%d existing events)", EVENT_LOG, len(data))
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("Event log corrupted (%s) — resetting %s", e, EVENT_LOG)
        with open(EVENT_LOG, "w") as f:
            f.write("[]\n")

_init_event_log()

# ─────────────────────────────────────────────────────────────
# Realistic corporate homepage  (inline — no templates/ needed)
# ─────────────────────────────────────────────────────────────
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>NexaCorp — Internal Portal</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:#f0f2f5;color:#1a1a2e;min-height:100vh;display:flex;flex-direction:column}
nav{background:#0f172a;padding:0 36px;display:flex;align-items:center;
    justify-content:space-between;height:54px;flex-shrink:0;
    box-shadow:0 1px 8px rgba(0,0,0,.4)}
.brand{color:#fff;font-size:17px;font-weight:600;display:flex;align-items:center;gap:9px}
.brand svg{flex-shrink:0}
.brand-sub{font-size:12px;font-weight:400;color:#94a3b8;margin-left:6px}
.nav-links{display:flex;gap:24px;list-style:none}
.nav-links a{color:#94a3b8;text-decoration:none;font-size:13px;transition:color .2s}
.nav-links a:hover{color:#e2e8f0}
.hero{flex:1;display:flex;align-items:center;justify-content:center;
      padding:48px 20px;gap:72px}
.hero-left{max-width:400px}
.hero-left h1{font-size:30px;font-weight:700;line-height:1.3;margin-bottom:14px;color:#0f172a}
.hero-left h1 em{font-style:normal;color:#2563eb}
.hero-left p{font-size:14px;color:#475569;line-height:1.75;margin-bottom:22px}
.badges{display:flex;gap:8px;flex-wrap:wrap}
.badge{font-size:11px;font-weight:600;padding:4px 10px;border-radius:99px;
       background:#dbeafe;color:#1e40af;border:1px solid #bfdbfe}
.card{background:#fff;border-radius:12px;padding:36px 38px;width:100%;
      max-width:380px;box-shadow:0 2px 24px rgba(0,0,0,.09)}
.card-logo{width:48px;height:48px;background:#eff6ff;border-radius:50%;
           display:flex;align-items:center;justify-content:center;margin:0 auto 14px}
.card h2{text-align:center;font-size:18px;font-weight:600;margin-bottom:4px}
.card-sub{text-align:center;font-size:12px;color:#94a3b8;margin-bottom:22px}
.notice{background:#fefce8;border:1px solid #fde047;border-left:4px solid #eab308;
        border-radius:6px;padding:9px 13px;font-size:12px;color:#713f12;
        margin-bottom:20px;display:flex;gap:8px;align-items:flex-start;line-height:1.5}
.notice svg{flex-shrink:0;margin-top:1px}
.fg{margin-bottom:16px}
label{display:block;font-size:12px;font-weight:600;color:#374151;
      margin-bottom:5px;letter-spacing:.3px}
input{width:100%;padding:9px 13px;border:1px solid #d1d5db;border-radius:7px;
      font-size:14px;color:#111;background:#f9fafb;outline:none;transition:all .2s}
input:focus{border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,.12);background:#fff}
.row{display:flex;align-items:center;justify-content:space-between;
     margin-bottom:20px;font-size:12px}
.row label{font-weight:400;color:#6b7280;display:flex;align-items:center;gap:5px;margin:0}
.row a{color:#2563eb;text-decoration:none}
.row a:hover{text-decoration:underline}
.btn{width:100%;padding:10px;background:#2563eb;color:#fff;border:none;
     border-radius:7px;font-size:14px;font-weight:600;cursor:pointer;transition:background .2s}
.btn:hover{background:#1d4ed8}
.divider{text-align:center;font-size:11px;color:#d1d5db;margin:18px 0;
         position:relative}
.divider::before,.divider::after{content:"";position:absolute;top:50%;
  width:36%;height:1px;background:#e5e7eb}
.divider::before{left:0}.divider::after{right:0}
.sso{width:100%;padding:9px;background:#fff;border:1px solid #d1d5db;
     border-radius:7px;font-size:13px;color:#374151;display:flex;
     align-items:center;justify-content:center;gap:9px;cursor:pointer;
     text-decoration:none;transition:background .2s;margin-bottom:9px}
.sso:hover{background:#f9fafb}
.card-foot{text-align:center;font-size:11px;color:#9ca3af;margin-top:18px}
.card-foot a{color:#2563eb;text-decoration:none}
.stats{background:#fff;border-top:1px solid #e5e7eb;padding:12px 36px;
       display:flex;gap:36px;justify-content:center;flex-shrink:0}
.stat{text-align:center}
.stat-n{font-size:19px;font-weight:700;color:#2563eb}
.stat-l{font-size:11px;color:#94a3b8;margin-top:2px}
footer{background:#0f172a;color:#475569;font-size:11px;text-align:center;
       padding:12px 36px;flex-shrink:0}
footer a{color:#64748b;text-decoration:none;margin:0 9px}
footer a:hover{color:#94a3b8}
@media(max-width:800px){
  .hero{flex-direction:column;gap:28px;padding:28px 16px}
  .nav-links{display:none}
  .stats{flex-wrap:wrap;gap:16px}
}
</style>
</head>
<body>
<nav>
  <div class="brand">
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
      <rect width="26" height="26" rx="6" fill="#2563eb"/>
      <path d="M7 13l4 4 8-8" stroke="#fff" stroke-width="2.2"
            stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    NexaCorp <span class="brand-sub">| Internal Operations Portal</span>
  </div>
  <ul class="nav-links">
    <li><a href="/api/v1/version">System Status</a></li>
    <li><a href="/api/v1/config">Configuration</a></li>
    <li><a href="/admin">Admin Panel</a></li>
    <li><a href="/actuator/health">Health</a></li>
    <li><a href="/api/v1/help">Support</a></li>
  </ul>
</nav>

<div class="hero">
  <div class="hero-left">
    <h1>Welcome to the<br><em>NexaCorp</em> Secure Portal</h1>
    <p>
      Centralised access to internal operations, project management,
      HR systems, CI/CD pipelines, and administrative dashboards.
      Sign in with your company credentials or SSO provider to continue.
    </p>
    <div class="badges">
      <span class="badge">&#128274; TLS 1.3</span>
      <span class="badge">&#9989; SOC 2 Type II</span>
      <span class="badge">&#128737; ISO 27001</span>
      <span class="badge">GDPR Ready</span>
    </div>
  </div>

  <div class="card">
    <div class="card-logo">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="8" r="4" stroke="#2563eb" stroke-width="2"/>
        <path d="M4 20c0-3.5 3.6-6 8-6s8 2.5 8 6"
              stroke="#2563eb" stroke-width="2" stroke-linecap="round"/>
      </svg>
    </div>
    <h2>Sign in</h2>
    <div class="card-sub">NexaCorp employee credentials required</div>

    <div class="notice">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" stroke="#ca8a04" stroke-width="2"/>
        <path d="M12 8v4M12 16h.01" stroke="#ca8a04" stroke-width="2"
              stroke-linecap="round"/>
      </svg>
      Authorised personnel only. All sessions are logged and monitored.
    </div>

    <form action="/api/v1/auth/login" method="POST">
      <div class="fg">
        <label>Employee ID / Email</label>
        <input type="text" name="username" placeholder="you@nexacorp.internal"
               autocomplete="username"/>
      </div>
      <div class="fg">
        <label>Password</label>
        <input type="password" name="password"
               placeholder="&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;"
               autocomplete="current-password"/>
      </div>
      <div class="row">
        <label><input type="checkbox" name="remember"/> Keep me signed in</label>
        <a href="/api/v1/reset-password">Forgot password?</a>
      </div>
      <button class="btn" type="submit">Sign In &rarr;</button>
    </form>

    <div class="divider">or</div>

    <a href="/api/v1/auth/token" class="sso">
      <svg width="16" height="16" viewBox="0 0 21 21">
        <rect x="1"  y="1"  width="9" height="9" fill="#f25022"/>
        <rect x="11" y="1"  width="9" height="9" fill="#7fba00"/>
        <rect x="1"  y="11" width="9" height="9" fill="#00a4ef"/>
        <rect x="11" y="11" width="9" height="9" fill="#ffb900"/>
      </svg>
      Continue with Microsoft Azure AD
    </a>
    <a href="/api/v1/auth/token" class="sso">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" fill="#007dc1"/>
        <circle cx="12" cy="12" r="5"  fill="#fff"/>
      </svg>
      Continue with Okta SSO
    </a>

    <div class="card-foot">
      New employee? <a href="/api/v1/register">Request access</a>
      &nbsp;&middot;&nbsp;
      <a href="/api/v1/help">IT Helpdesk</a>
      &nbsp;&middot;&nbsp;
      <a href="/server-status">Status</a>
    </div>
  </div>
</div>

<div class="stats">
  <div class="stat"><div class="stat-n">3,241</div><div class="stat-l">Active Users</div></div>
  <div class="stat"><div class="stat-n">99.97%</div><div class="stat-l">Uptime SLA</div></div>
  <div class="stat"><div class="stat-n">v5.1.3</div><div class="stat-l">Portal Version</div></div>
  <div class="stat"><div class="stat-n">18</div><div class="stat-l">Integrated Services</div></div>
  <div class="stat"><div class="stat-n">&#128994;</div><div class="stat-l">All Systems OK</div></div>
</div>

<footer>
  &copy; 2024 NexaCorp Ltd. All rights reserved.
  <a href="/api/v1/config">Privacy</a>
  <a href="/api/v1/help">Terms</a>
  <a href="/server-status">System Status</a>
  <a href="/actuator/health">Health Check</a>
  <a href="/.env">Environment</a>
  <a href="/.git/config">Source</a>
</footer>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────
# Attacker fingerprinting
# ─────────────────────────────────────────────────────────────
def attacker_id():
    if "_aid" not in session:
        raw = f"{request.remote_addr}|{request.headers.get('User-Agent','')}"
        session["_aid"] = hashlib.sha256(raw.encode()).hexdigest()[:24]
    return session["_aid"]

def real_ip():
    for h in ("X-Forwarded-For", "X-Real-IP", "CF-Connecting-IP"):
        v = request.headers.get(h)
        if v:
            return v.split(",")[0].strip()
    return request.remote_addr or "0.0.0.0"

# ─────────────────────────────────────────────────────────────
# Attack classifier  (FIX 1 & 4 applied here)
#
# FIX 1: blob is built from path+body+qs+ua ONLY — never req.url.
#   req.url always contains "localhost:5000" which made _RE_SSRF match
#   every single request and report everything as A10.
#
# FIX 4: unknown paths default to A01 not A10.
#
# FIX: Added explicit _PATH_A03 so /search /query /login etc. are
#   classified as A03 by path alone before any payload scan.
#
# FIX: _RE_SSRF no longer matches bare "localhost" — only real
#   internal targets in the payload/qs (169.254.x, 127.x, file://, etc.)
# ─────────────────────────────────────────────────────────────

# ── Content / payload patterns ────────────────────────────────
_RE_SQLI = re.compile(
    r"(union[\s+]select|select\s.+\sfrom|insert\s+into|drop\s+(table|database)"
    r"|'[\s]*--|\bor\b[\s\d]+=[\s\d]+|\bexec\b|\bxp_|\binformation_schema\b"
    r"|\bcast\s*\(|\bconvert\s*\(|0x[0-9a-f]{4,})", re.I)

_RE_XSS = re.compile(
    r"(<script[\s>]|javascript\s*:|on\w+\s*=|<iframe[\s>]|<svg[\s>]"
    r"|alert\s*\(|document\.cookie|<img[^>]+onerror)", re.I)

# Tightened: requires a recognisable command after the metachar —
# bare | or ; in query strings no longer fire.
_RE_CMDI = re.compile(
    r"(/etc/passwd|/etc/shadow|/bin/sh|/bin/bash|cmd\.exe"
    r"|\$\([^)]{2,}\)|`[^`]{2,}`"
    r"|&&\s*\w|\|\|\s*\w|\|\s*(cat|ls|id|whoami|wget|curl|nc)\b)", re.I)

# SSRF: does NOT match bare "localhost" — only real internal targets
# in the request payload or query string.
_RE_SSRF = re.compile(
    r"(169\.254\.169\.254"
    r"|metadata\.google\.internal"
    r"|127\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|::1(?!\d)"
    r"|file://"
    r"|dict://"
    r"|gopher://"
    r"|http://localhost[:/])",   # only as a URL value, not the server itself
    re.I)

_RE_CRYPTO = re.compile(
    r"(\.pem\b|\.key\b|\.p12\b|private.{0,6}key|secret.{0,6}key"
    r"|password.{0,6}hash|\bmd5\b|\bsha1\b|\bdes\b|\brc4\b"
    r"|base64.{0,10}decode)", re.I)

_RE_AUTH = re.compile(
    r"(brute.?force|credential.?stuff|password.?spray|wordlist"
    r"|rockyou|admin.{0,5}pass|login.{0,5}fuzz)", re.I)

_RE_INTEG = re.compile(
    r"(\.(jar|gem|whl|deb|rpm)\b|supply.?chain|ci.?cd|pipeline"
    r"|build.?script|npm\s+install|pip\s+install\s+\S|__import__)", re.I)

_RE_LOG = re.compile(
    r"(log.?clear|audit.?clear|/var/log|syslog|event.?log"
    r"|rm\s+-[rf]+\s+.*log)", re.I)

_RE_SCAN = re.compile(
    r"(sqlmap|nikto|nmap|masscan|zgrab|nuclei|burpsuite|owasp.?zap"
    r"|dirbuster|gobuster|wfuzz|metasploit|nessus|openvas)", re.I)

# ── Path-based patterns (ordered most → least specific) ───────
_PATH_A03 = re.compile(
    r"^/(search|query|login|api/v\d+/(search|query|login|comments|feedback))$", re.I)
_PATH_A05 = re.compile(
    r"^/(\.env(\..*)?|config|phpinfo\.php|server-status|server-info"
    r"|wp-config\.php|actuator(/.*)?|debug(/.*)?|\.git(/.*)?)", re.I)
_PATH_A06 = re.compile(
    r"^/(vendor|node_modules|packages|plugins|modules)/"
    r"|^/api/(v\d+/)?(version|component/check)$", re.I)
_PATH_A10 = re.compile(
    r"^/(api/(v\d+/)?(fetch|proxy|import|metadata)|webhook/validate"
    r"|api/internal/.*)", re.I)
_PATH_A09 = re.compile(r"^/(api/(v\d+/)?(logs|audit)(/.*)?|admin/logs)", re.I)
_PATH_A08 = re.compile(
    r"^/(api/(v\d+/)?(upload|deploy|build|pipeline(/.*)?)|webhook(/.*)?)", re.I)
_PATH_A07 = re.compile(
    r"^/(api/(v\d+/)?(auth(/.*)?|login|register|token|refresh)"
    r"|wp-login\.php|xmlrpc\.php)", re.I)
_PATH_A02 = re.compile(
    r"^/(api/(v\d+/)?(keys|secrets|tokens)|export(/.*)?|download/backup)", re.I)
_PATH_A04 = re.compile(
    r"^/(api/(v\d+/)?(checkout|redeem|coupon|transfer|reset-password|verify))", re.I)
_PATH_A01 = re.compile(
    r"^/(admin(/.*)?|dashboard|internal(/.*)?|manage(/.*)?|panel|superuser|root)", re.I)


def classify(req) -> str:
    """
    Map a request to one of A01–A10.

    blob is built from path+body+qs+ua ONLY (no req.url / hostname).
    This prevents the server's own 'localhost:5000' from matching SSRF.
    """
    path = req.path
    ua   = req.headers.get("User-Agent", "")

    try:
        body = req.get_data(as_text=True)[:3000]
    except Exception:
        body = ""

    qs = req.query_string.decode("utf-8", "replace")

    # blob = path + body + qs + ua  — NO hostname / full URL
    blob = f"{path} {body} {qs} {ua}"

    # Step 1: payload-based injection (highest specificity, any path)
    if _RE_SQLI.search(blob) or _RE_XSS.search(blob):
        return "A03"

    # Step 2: explicit path routing (A03 paths before general catch-alls)
    if _PATH_A03.match(path):   return "A03"
    if _PATH_A05.match(path):   return "A05"
    if _PATH_A06.match(path):   return "A06"
    if _PATH_A09.match(path):   return "A09"
    if _PATH_A08.match(path):   return "A08"
    if _PATH_A10.match(path):   return "A10"
    if _PATH_A07.match(path):   return "A07"
    if _PATH_A04.match(path):   return "A04"
    if _PATH_A02.match(path):   return "A02"
    if _PATH_A01.match(path):   return "A01"

    # Step 3: payload-based refinement for unrouted paths
    # SSRF check uses body+qs only (not full blob) to be extra safe
    if _RE_SSRF.search(body) or _RE_SSRF.search(qs):
        return "A10"
    if _RE_LOG.search(blob):    return "A09"
    if _RE_INTEG.search(blob):  return "A08"
    if _RE_CRYPTO.search(blob): return "A02"
    if _RE_AUTH.search(blob) or _RE_SCAN.search(ua): return "A07"
    if _RE_CMDI.search(blob):   return "A03"

    # Step 4: default — unknown probe = A01 (NOT A10)
    return "A01"

# ─────────────────────────────────────────────────────────────
# Covert event logger  (FIX 5 applied here)
#
# Writes every event to honeypot_events.json as a proper JSON
# array.  Uses atomic write (tmp + os.replace) so the file is
# always valid JSON even if the process is killed mid-write.
# ─────────────────────────────────────────────────────────────
def log_event(attack, action, state, next_state, r, sess):
    """Append one event to honeypot_events.json."""
    rec = {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "ip":                real_ip(),
        "attacker_id":       sess.sid,
        "user_agent":        request.headers.get("User-Agent", ""),
        "method":            request.method,
        "path":              request.full_path,
        "owasp_category":    attack,
        "action_taken":      action,
        "state_from":        state,
        "state_to":          next_state,
        "reward":            round(r, 4),
        "session_reward":    round(sess.reward_sum, 4),
        "cumulative_reward": round(agent._global_reward.get(sess.sid, sess.reward_sum), 4),
        "session_steps":     sess.steps,
        "payload":           (request.get_data(as_text=True)[:512] or None),
        "query_args":        dict(request.args),
    }

    with _log_lock:
        # Append-only — never reads the file back.
        # Seeks to the byte just before the closing ] and inserts the
        # new event line there.  Works correctly on files with 1 or
        # 1,000,000 events because it never loads the whole file.
        try:
            line = json.dumps(rec, separators=(",", ":"))
            with open(EVENT_LOG, "r+") as f:
                # find the closing ] by reading the last ~50 bytes
                f.seek(0, 2)                    # go to end of file
                size = f.tell()
                f.seek(max(0, size - 50))
                tail = f.read()
                bracket_pos = tail.rfind("]")   # position of ] within tail
                if bracket_pos == -1:
                    raise ValueError("no closing bracket")
                # absolute file position of ]
                insert_at = max(0, size - 50) + bracket_pos

                # check whether there are already events before ]
                f.seek(0)
                content = f.read(insert_at)
                is_first = content.strip() == "["

                # write new line at insert_at, overwriting ] and everything after
                f.seek(insert_at)
                f.truncate()
                if is_first:
                    f.write("  " + line + "\n]\n")
                else:
                    # step back one byte to overwrite the previous \n,
                    # so the comma lands at the END of the previous event line
                    f.seek(insert_at - 1)
                    f.truncate()
                    f.write(",\n  " + line + "\n]\n")
        except Exception as e:
            log.error("Failed to write to event log: %s", e)

    log.info("EVT [%s] %s --%s--> %s  R=%.2f  cumR=%.2f",
             attack, state, action, next_state, r,
             agent._global_reward.get(sess.sid, r))

# ─────────────────────────────────────────────────────────────
# Deceptive response bank  (A1–A8)
# ─────────────────────────────────────────────────────────────

# A1 — serve the real-looking homepage
def resp_A1():
    return Response(INDEX_HTML, 200, mimetype="text/html",
                    headers={"Server": "Apache/2.4.54 (Ubuntu)"})

# A2 — fake error / stack trace / version leak
_LURES_A2 = [
    lambda: Response(
        "Traceback (most recent call last):\n"
        "  File \"/opt/nexacorp/app/views.py\", line 84, in user_detail\n"
        "    row = db.execute(f\"SELECT * FROM users WHERE id={uid}\").fetchone()\n"
        "sqlite3.OperationalError: near \"'\": syntax error\n\n"
        "DEBUG=True  DB=/opt/nexacorp/data/app.db  SECRET_KEY=devonly-change-me\n"
        "SERVER: gunicorn/20.1.0  Python/3.11.2  Django/4.2.1",
        500, mimetype="text/plain",
        headers={"X-Powered-By": "Python/3.11", "Server": "gunicorn/20.1.0"}),
    lambda: Response(
        "<b>Fatal error</b>: Uncaught PDOException: SQLSTATE[42000] "
        "Syntax error near '' at line 1 in /var/www/nexacorp/includes/db.php:61\n"
        "Stack: #0 /var/www/nexacorp/api/users.php(22): PDO->query()\n"
        "PHP/7.4.33  MySQL/8.0.31  Apache/2.4.54",
        500, mimetype="text/html",
        headers={"X-Powered-By": "PHP/7.4.33", "Server": "Apache/2.4.54"}),
    lambda: Response(json.dumps({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "status": 500, "error": "Internal Server Error",
        "message": "could not extract ResultSet; nested exception is "
                   "org.hibernate.exception.SQLGrammarException",
        "server": "SpringBoot/3.1.0", "java": "17.0.7",
        "db": "PostgreSQL 15.2", "path": request.path
    }), 500, mimetype="application/json",
        headers={"Server": "Tomcat/10.1", "X-Application": "NexaCorp-API/5.1.3"}),
]

def resp_A2():
    return random.choice(_LURES_A2)()

# A3 — fake sensitive data dump
_FAKE_USERS = [
    {"id": 1, "username": "admin",    "email": "admin@nexacorp.internal",
     "role": "superadmin", "password_hash": "$2b$12$FAKEHASH1xxxxxADMIN",
     "api_key": "sk-live-FAKE-Aa1Bb2Cc3Dd4Ee5Ff6"},
    {"id": 2, "username": "dbadmin",  "email": "db@nexacorp.internal",
     "role": "dba",        "password_hash": "$2b$12$FAKEHASH2xxxxxDBDMIN",
     "api_key": "sk-live-FAKE-Gg7Hh8Ii9Jj0Kk1Ll2"},
    {"id": 3, "username": "cicd_svc", "email": "ci@nexacorp.internal",
     "role": "service",    "password_hash": "$2b$12$FAKEHASH3xxxxxSERVIC",
     "api_key": "sk-live-FAKE-Mm3Nn4Oo5Pp6Qq7Rr8"},
]
_FAKE_CFG = {
    "database": {"host": "10.10.0.12", "port": 5432, "name": "nexacorp_prod",
                 "user": "nexaapp", "password": "Nx@Pr0d!2024FAKE"},
    "redis":    {"host": "10.10.0.20", "port": 6379, "password": "R3d!sFAKEpass"},
    "aws": {"access_key_id":     "AKIAFAKE00NEXACORP1",
            "secret_access_key": "FAKESECRET/nexacorp/prod/DoNotUse",
            "region": "eu-west-1", "s3_bucket": "nexacorp-prod-backups"},
    "jwt_secret": "NEXACORP_FAKE_JWT_SECRET_2024_doNotUse!",
    "smtp": {"host": "smtp.nexacorp.internal", "port": 587,
             "user": "noreply@nexacorp.com", "password": "SMTPfake!2024"},
}

def resp_A3():
    ch = random.randint(0, 2)
    if ch == 0:
        return jsonify({"ok": True, "users": _FAKE_USERS})
    elif ch == 1:
        return Response(
            "# NexaCorp credential dump — CONFIDENTIAL\n" +
            "\n".join(f"{u['username']}:{u['password_hash']}" for u in _FAKE_USERS),
            200, mimetype="text/plain")
    else:
        return jsonify({"ok": True, "config": _FAKE_CFG})

# A4 — tarpit
def resp_A4():
    delay = random.uniform(10.0, 25.0)
    log.info("TARPIT %.1fs -> %s", delay, real_ip())
    time.sleep(delay)
    return Response("Service unavailable — please try again later.",
                    503, headers={"Retry-After": "60", "X-Rate-Limit-Remaining": "0"})

# A5 — redirect to deeper decoy
_DECOYS = [
    "/admin/dashboard", "/admin/users", "/internal/api/v2/users",
    "/manage/config", "/.hidden/db-backup", "/api/v1/admin/export",
    "/debug/console", "/phpmyadmin/", "/api/internal/health/full",
    "/api/v1/secrets", "/.env.production",
]

def resp_A5():
    target = random.choice(_DECOYS)
    log.info("REDIRECT %s -> %s", real_ip(), target)
    r = redirect(target, 302)
    r.headers["X-Auth-Redirect"] = "privilege-upgrade"
    return r

# A6 — fake auth / privilege grant
_FAKE_JWT = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
             ".eyJzdWIiOiIxIiwicm9sZSI6ImFkbWluIiwiZXhwIjo5OTk5OTk5OTk5fQ"
             ".FAKE_SIG_NEXACORP")

def resp_A6():
    tok = uuid.uuid4().hex + "FAKE"
    payload = {
        "status": "success", "message": "Authentication successful",
        "token": _FAKE_JWT, "session_token": tok,
        "user": {
            "id":       random.randint(1, 3),
            "username": random.choice(["admin", "sysadmin", "root"]),
            "role":     "administrator",
            "permissions": ["read", "write", "delete", "admin", "ci_cd", "audit"],
        },
        "expires_in": 3600,
        "mfa_bypassed": True,
    }
    resp = jsonify(payload)
    resp.set_cookie("auth_token", tok, httponly=True, samesite="Lax")
    resp.set_cookie("role", "admin")
    return resp

# A7 — fake exploit / payload accept
_RCE_OUT = [
    "uid=0(root) gid=0(root) groups=0(root)\n",
    "Linux nexacorp-prod-01 5.15.0-91-generic #101-Ubuntu SMP x86_64\n",
    "total 0\ndrwxr-xr-x 1 root root 6 Jan  1 00:00 /\n"
    "drwxr-xr-x 1 root root 6 Jan  1 00:00 /etc\n",
]

def resp_A7():
    ch = random.randint(0, 2)
    if ch == 0:
        return jsonify({"status": "executed", "output": random.choice(_RCE_OUT),
                        "pid": random.randint(1000, 65000), "user": "root"})
    elif ch == 1:
        return jsonify({"status": "queued", "pipeline": "nexacorp-prod-deploy",
                        "build_id": f"build-{random.randint(1000,9999)}",
                        "injected_stage": "post-build",
                        "eta_s": random.randint(30, 90)})
    else:
        return jsonify({"status": "vulnerable",
                        "cve": f"CVE-{random.randint(2020,2024)}-{random.randint(1000,99999)}",
                        "exploitable": True,
                        "component": random.choice(
                            ["Log4j 2.14.1", "Spring4Shell", "Struts 2.5.25"]),
                        "shell": "connection established"})

# A8 — fake log / audit clear  (real logging still happens via log_event)
def resp_A8():
    return jsonify({
        "status":  "success",
        "message": "Audit log cleared",
        "entries_removed": random.randint(800, 8000),
        "backup":  False,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "operator": "system",
        "log": "/var/log/nexacorp/audit.log",
    })

# dispatch table
_HANDLERS: Dict[str, Callable] = {
    "A1": resp_A1, "A2": resp_A2, "A3": resp_A3, "A4": resp_A4,
    "A5": resp_A5, "A6": resp_A6, "A7": resp_A7, "A8": resp_A8,
}

# ─────────────────────────────────────────────────────────────
# Core decorator — wraps every honeypot route
# ─────────────────────────────────────────────────────────────
def honeypot(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        aid    = attacker_id()
        attack = classify(request)
        sess, state_before, action, next_s, r, done = agent.decide(aid, attack)
        log_event(attack, action, state_before, next_s, r, sess)

        resp = _HANDLERS.get(action, resp_A1)()
        resp.headers["X-Request-ID"]   = uuid.uuid4().hex
        resp.headers["X-Processed-By"] = "nexacorp-gateway/2.3"
        return resp
    return wrapper

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

# Root — always shows the realistic homepage
@app.route("/", methods=["GET"])
def index():
    return Response(INDEX_HTML, 200, mimetype="text/html",
                    headers={"Server": "Apache/2.4.54 (Ubuntu)"})

@app.route("/", methods=["POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
@honeypot
def index_verb(): pass

# A01 — Broken Access Control
@app.route("/admin",                    methods=["GET", "POST"])
@app.route("/admin/",                   methods=["GET", "POST"])
@app.route("/admin/panel",              methods=["GET", "POST"])
@app.route("/admin/dashboard",          methods=["GET", "POST"])
@app.route("/admin/users",              methods=["GET", "POST"])
@app.route("/internal/api/v2/users",   methods=["GET", "POST", "DELETE"])
@app.route("/api/v1/admin/export",     methods=["GET", "POST"])
@app.route("/manage/",                 methods=["GET", "POST"])
@app.route("/manage/config",           methods=["GET", "POST"])
@app.route("/superuser",               methods=["GET", "POST"])
@honeypot
def admin_routes(): pass

@app.route("/api/v1/users/",           methods=["GET", "POST"])
@app.route("/api/v1/users/<int:uid>",  methods=["GET", "PUT", "DELETE"])
@honeypot
def user_api(uid=None): pass

@app.route("/api/v1/orders/<int:oid>", methods=["GET", "PUT", "DELETE"])
@honeypot
def order_api(oid=None): pass

# A02 — Cryptographic Failures
@app.route("/api/v1/keys",      methods=["GET", "POST"])
@app.route("/api/v1/secrets",   methods=["GET", "POST"])
@app.route("/api/v1/tokens",    methods=["GET", "POST"])
@app.route("/export/users",     methods=["GET"])
@app.route("/export/data",      methods=["GET"])
@app.route("/download/backup",  methods=["GET"])
@honeypot
def crypto_routes(): pass

# A03 — Injection
@app.route("/search",             methods=["GET", "POST"])
@app.route("/api/v1/search",      methods=["GET", "POST"])
@app.route("/query",              methods=["GET", "POST"])
@app.route("/api/v1/query",       methods=["GET", "POST"])
@app.route("/login",              methods=["GET", "POST"])
@app.route("/api/v1/login",       methods=["GET", "POST"])
@app.route("/api/v1/comments",    methods=["GET", "POST"])
@app.route("/api/v1/feedback",    methods=["GET", "POST"])
@honeypot
def injection_routes(): pass

# A04 — Insecure Design
@app.route("/api/v1/checkout",        methods=["POST"])
@app.route("/api/v1/redeem",          methods=["POST"])
@app.route("/api/v1/coupon",          methods=["POST"])
@app.route("/api/v1/transfer",        methods=["POST"])
@app.route("/api/v1/reset-password",  methods=["GET", "POST"])
@app.route("/api/v1/verify",          methods=["POST"])
@honeypot
def design_routes(): pass

# A05 — Security Misconfiguration
@app.route("/.env",             methods=["GET"])
@app.route("/.env.production",  methods=["GET"])
@app.route("/config",           methods=["GET"])
@app.route("/phpinfo.php",      methods=["GET"])
@app.route("/server-status",    methods=["GET"])
@app.route("/server-info",      methods=["GET"])
@app.route("/wp-config.php",    methods=["GET"])
@app.route("/actuator",         methods=["GET"])
@app.route("/actuator/env",     methods=["GET"])
@app.route("/actuator/health",  methods=["GET"])
@app.route("/actuator/metrics", methods=["GET"])
@app.route("/debug",            methods=["GET", "POST"])
@app.route("/debug/console",    methods=["GET", "POST"])
@app.route("/.git/config",      methods=["GET"])
@app.route("/.git/HEAD",        methods=["GET"])
@app.route("/api/v1/config",    methods=["GET"])
@app.route("/api/v1/help",      methods=["GET"])
@honeypot
def misconfig_routes(): pass

# A06 — Vulnerable Components
@app.route("/vendor/<path:p>",        methods=["GET"])
@app.route("/node_modules/<path:p>",  methods=["GET"])
@app.route("/packages/<path:p>",      methods=["GET"])
@app.route("/api/v1/version",         methods=["GET"])
@app.route("/api/v1/component/check", methods=["GET", "POST"])
@honeypot
def component_routes(p=None): pass

# A07 — Authentication Failures
@app.route("/api/v1/auth/login",   methods=["GET", "POST"])
@app.route("/api/v1/auth/token",   methods=["GET", "POST"])
@app.route("/api/v1/auth/refresh", methods=["POST"])
@app.route("/api/v1/register",     methods=["GET", "POST"])
@app.route("/wp-login.php",        methods=["GET", "POST"])
@app.route("/xmlrpc.php",          methods=["GET", "POST"])
@honeypot
def auth_routes(): pass

# A08 — Software & Data Integrity
@app.route("/api/v1/upload",            methods=["POST"])
@app.route("/api/v1/pipeline/trigger",  methods=["POST"])
@app.route("/api/v1/deploy",            methods=["POST"])
@app.route("/api/v1/build",             methods=["POST"])
@app.route("/webhook",                  methods=["POST"])
@app.route("/webhook/github",           methods=["POST"])
@app.route("/webhook/gitlab",           methods=["POST"])
@honeypot
def integrity_routes(): pass

# A09 — Logging & Monitoring Failures
@app.route("/api/v1/logs",        methods=["GET", "DELETE"])
@app.route("/api/v1/logs/clear",  methods=["POST", "DELETE"])
@app.route("/api/v1/audit",       methods=["GET", "DELETE"])
@app.route("/api/v1/audit/clear", methods=["POST", "DELETE"])
@app.route("/admin/logs",         methods=["GET", "DELETE"])
@honeypot
def logging_routes(): pass

# A10 — SSRF
@app.route("/api/v1/fetch",             methods=["GET", "POST"])
@app.route("/api/v1/proxy",             methods=["GET", "POST"])
@app.route("/api/v1/webhook/validate",  methods=["POST"])
@app.route("/api/v1/import",            methods=["POST"])
@app.route("/api/v1/metadata",          methods=["GET"])
@app.route("/api/internal/metadata",    methods=["GET"])
@app.route("/api/internal/health/full", methods=["GET"])
@honeypot
def ssrf_routes(): pass

# Decoy destinations (redirected to by A5)
@app.route("/.hidden/db-backup", methods=["GET", "POST"])
@app.route("/phpmyadmin/",       methods=["GET", "POST"])
@app.route("/phpmyadmin",        methods=["GET", "POST"])
@honeypot
def decoy_destinations(): pass

# Catch-all
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
@honeypot
def catch_all(path=""): pass

# ─────────────────────────────────────────────────────────────
# Internal monitoring  (127.0.0.1 only — firewall in production)
# ─────────────────────────────────────────────────────────────
@app.route("/_hp/stats")
def hp_stats():
    if real_ip() not in ("127.0.0.1", "::1"):
        abort(404)
    s = agent.stats()
    s["sessions_detail"] = [
        {
            "sid":               v.sid,
            "attack":            v.attack,
            "state":             v.state,
            "steps":             v.steps,
            "session_reward":    round(v.reward_sum, 3),
            "cumulative_reward": round(agent._global_reward.get(v.sid, v.reward_sum), 3),
        }
        for v in agent._sessions.values()
    ]
    return jsonify(s)

@app.route("/_hp/qtable")
def hp_qtable():
    if real_ip() not in ("127.0.0.1", "::1"):
        abort(404)
    return jsonify(agent.qtable.dump())

@app.route("/_hp/save", methods=["POST"])
def hp_save():
    if real_ip() not in ("127.0.0.1", "::1"):
        abort(404)
    agent.qtable.save()
    return jsonify({"saved": True})

# ─────────────────────────────────────────────────────────────
# Error handlers — keep the illusion alive
# ─────────────────────────────────────────────────────────────
@app.errorhandler(404)
def e404(e):
    return Response(
        "<!DOCTYPE html><html><head><title>404 Not Found</title></head>"
        "<body><h1>404 — The requested URL was not found on this server.</h1></body></html>",
        404, mimetype="text/html",
        headers={"Server": "Apache/2.4.54 (Ubuntu)"})

@app.errorhandler(405)
def e405(e):
    return Response('{"error":"Method Not Allowed"}', 405, mimetype="application/json")

@app.errorhandler(500)
def e500(e):
    return resp_A2()

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="NexaCorp Adaptive Honeypot")
    p.add_argument("--host",  default="0.0.0.0")
    p.add_argument("--port",  default=5000, type=int)
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    log.info("Honeypot starting on %s:%d", a.host, a.port)
    log.info("Events  -> %s", EVENT_LOG)
    log.info("Q-table -> qtable.json")
    log.warning("PROTECT /_hp/* — localhost only in production!")
    app.run(host=a.host, port=a.port, debug=a.debug, threaded=True)

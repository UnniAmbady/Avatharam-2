# Avatharam-2
# ver-0.0
# Avatharam-2.2 — UI Revamp (Streamlit app)
# Cosmetic/layout update + sanity-wired to the previously working flow.
# - ☰ Trigram toggles side panel; Start/Stop moved there (Start label only).
# - Fixed avatar (June_HR_public). No selector and no extra text.
# - On app start, show the static avatar preview image in the main panel.
#   When a session is started, the viewer replaces that image in-place.
# - Below the viewer area (same spot as before), mic_recorder renders with
#   labels changed to Speak / Stop (behavior unchanged).
# - Replace text edit box with st.chat_input (2-line look).
# - Debug log kept in a text_area at the bottom, like the original.

import json
import os
import time
from pathlib import Path
from typing import Optional, List

import requests
import streamlit as st
import streamlit.components.v1 as components

# ---------------- Page ----------------
st.set_page_config(page_title="Avatharam-2", layout="centered")
st.text("by Krish Ambady")

# --------------- CSS (2-line chat input + small polish) ---------------
st.markdown(
    """
<style>
  .block-container { padding-top:.6rem; padding-bottom:1rem; }
  iframe { border:none; border-radius:16px; }
  .rowbtn .stButton>button { height:40px; font-size:.95rem; border-radius:12px; }
  /* chat_input ~2 lines */
  div.stChatInput textarea {
      min-height: 3.4em !important;
      max-height: 3.8em !important;
  }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Fixed Avatar ----------------
FIXED_AVATAR = {
    "avatar_id": "June_HR_public",
    "default_voice": "68dedac41a9f46a6a4271a95c733823c",
    "normal_preview": "https://files2.heygen.ai/avatar/v3/74447a27859a456c955e01f21ef18216_45620/preview_talk_1.webp",
    "pose_name": "June HR",
    "status": "ACTIVE",
}

# ---------------- Secrets ----------------
SECRETS = st.secrets if "secrets" in dir(st) else {}
HEYGEN_API_KEY = (
    SECRETS.get("HeyGen", {}).get("heygen_api_key")
    or SECRETS.get("heygen", {}).get("heygen_api_key")
    or os.getenv("HEYGEN_API_KEY")
)
OPENAI_API_KEY = (
    SECRETS.get("openai", {}).get("secret_key")
    or os.getenv("OPENAI_API_KEY")
)
if not HEYGEN_API_KEY:
    st.error("Missing HeyGen API key in .streamlit/secrets.toml")
    st.stop()
if not OPENAI_API_KEY:
    st.error("Missing OpenAI API key in .streamlit/secrets.toml")
    st.stop()

# --------------- HeyGen Endpoints --------------
BASE = "https://api.heygen.com/v1"
API_STREAM_NEW = f"{BASE}/streaming.new"
API_CREATE_TOKEN = f"{BASE}/streaming.create_token"
API_STREAM_TASK = f"{BASE}/streaming.task"
API_STREAM_STOP = f"{BASE}/streaming.stop"

HEADERS_XAPI = {
    "accept": "application/json",
    "x-api-key": HEYGEN_API_KEY,
    "Content-Type": "application/json",
}


def _headers_bearer(tok: str):
    return {
        "accept": "application/json",
        "Authorization": f"Bearer {tok}",
        "Content-Type": "application/json",
    }

# --------- Debug buffer ----------
ss = st.session_state
ss.setdefault("debug_buf", [])
ss.setdefault("session_id", None)
ss.setdefault("session_token", None)
ss.setdefault("offer_sdp", None)
ss.setdefault("rtc_config", None)
ss.setdefault("last_text", "")
ss.setdefault("last_reply", "")
ss.setdefault("show_sidebar", False)
ss.setdefault("Name", "Friend")


def debug(msg: str):
    ss.debug_buf.append(str(msg))
    if len(ss.debug_buf) > 1000:
        ss.debug_buf[:] = ss.debug_buf[-1000:]

# ------------- HTTP helpers --------------

import requests


def _post_xapi(url, payload=None):
    r = requests.post(url, headers=HEADERS_XAPI, data=json.dumps(payload or {}), timeout=60)
    raw = r.text
    try:
        body = r.json()
    except Exception:
        body = {"_raw": raw}
    debug(f"[POST x-api] {url} -> {r.status_code}")
    if r.status_code >= 400:
        debug(raw)
        r.raise_for_status()
    return r.status_code, body, raw


def _post_bearer(url, token, payload=None):
    r = requests.post(url, headers=_headers_bearer(token), data=json.dumps(payload or {}), timeout=60)
    raw = r.text
    try:
        body = r.json()
    except Exception:
        body = {"_raw": raw}
    debug(f"[POST bearer] {url} -> {r.status_code}")
    if r.status_code >= 400:
        debug(raw)
        r.raise_for_status()
    return r.status_code, body, raw


# ------------- Session helpers -------------


def new_session(avatar_id: str, voice_id: Optional[str] = None):
    payload = {"avatar_id": avatar_id}
    if voice_id:
        payload["voice_id"] = voice_id
    _, body, _ = _post_xapi(API_STREAM_NEW, payload)
    data = body.get("data") or {}
    sid = data.get("session_id")
    offer_sdp = (data.get("offer") or data.get("sdp") or {}).get("sdp")
    ice2 = data.get("ice_servers2")
    ice1 = data.get("ice_servers")
    if isinstance(ice2, list) and ice2:
        rtc_config = {"iceServers": ice2}
    elif isinstance(ice1, list) and ice1:
        rtc_config = {"iceServers": ice1}
    else:
        rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    if not sid or not offer_sdp:
        raise RuntimeError(f"Missing session_id or offer in response: {body}")
    return {"session_id": sid, "offer_sdp": offer_sdp, "rtc_config": rtc_config}


def create_session_token(session_id: str) -> str:
    _, body, _ = _post_xapi(API_CREATE_TOKEN, {"session_id": session_id})
    tok = (body.get("data") or {}).get("token") or (body.get("data") or {}).get("access_token")
    if not tok:
        raise RuntimeError(f"Missing token in response: {body}")
    return tok


def send_text_to_avatar(session_id: str, session_token: str, text: str):
    debug(f"[avatar] speak {len(text)} chars")
    _post_bearer(
        API_STREAM_TASK,
        session_token,
        {
            "session_id": session_id,
            "task_type": "repeat",
            "task_mode": "sync",
            "text": text,
        },
    )


def stop_session(session_id: Optional[str], session_token: Optional[str]):
    if not (session_id and session_token):
        return
    try:
        _post_bearer(API_STREAM_STOP, session_token, {"session_id": session_id})
    except Exception as e:
        debug(f"[stop_session] {e}")


# ---------------- Header: Trigram ----------------
cols = st.columns([1, 12, 1])
with cols[0]:
    if st.button("☰", help="Open menu (Trigram for Heaven)"):
        ss.show_sidebar = not ss.show_sidebar
        debug(f"[ui] sidebar -> {ss.show_sidebar}")

# ---------------- Sidebar (Start/Stop moved) ----------------
if ss.show_sidebar:
    with st.sidebar:
        st.markdown("### Text")
        if st.button("Start", key="start_btn"):
            # Same code path as working version (was Start / Restart)
            if ss.session_id and ss.session_token:
                stop_session(ss.session_id, ss.session_token)
                time.sleep(0.2)
            debug("Step 1: streaming.new")
            payload = new_session(FIXED_AVATAR["avatar_id"], FIXED_AVATAR.get("default_voice"))
            sid, offer_sdp, rtc_config = payload["session_id"], payload["offer_sdp"], payload["rtc_config"]
            debug("Step 2: streaming.create_token")
            tok = create_session_token(sid)
            debug("Step 3: sleep 1.0s before viewer")
            time.sleep(1.0)
            ss.session_id, ss.session_token = sid, tok
            ss.offer_sdp, ss.rtc_config = offer_sdp, rtc_config
            debug(f"[ready] session_id={sid[:8]}…")
        if st.button("Stop", key="stop_btn"):
            stop_session(ss.session_id, ss.session_token)
            ss.session_id = None
            ss.session_token = None
            ss.offer_sdp = None
            ss.rtc_config = None
            debug("[stopped] session cleared")

# ---------------- Main viewer area ----------------
viewer_path = Path(__file__).parent / "viewer.html"
viewer_loaded = ss.session_id and ss.session_token and ss.offer_sdp

def _image_compat(url: str, caption: str = ""):
    """Show image with compatibility across Streamlit versions.
    - Prefer use_container_width
    - Fallback to deprecated use_column_width
    - Final fallback: no sizing keyword
    """
    try:
        st.image(url, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(url, caption=caption, use_column_width=True)
        except TypeError:
            st.image(url, caption=caption)

if viewer_loaded and viewer_path.exists():
    html = (
        viewer_path.read_text(encoding="utf-8")
        .replace("__SESSION_TOKEN__", ss.session_token)
        .replace("__AVATAR_NAME__", FIXED_AVATAR["pose_name"])
        .replace("__SESSION_ID__", ss.session_id)
        .replace("__OFFER_SDP__", json.dumps(ss.offer_sdp)[1:-1])
        .replace("__RTC_CONFIG__", json.dumps(ss.rtc_config or {}))
    )
    components.html(html, height=340, scrolling=False)
else:
    # Show the static preview image until a session is live
    _image_compat(
        FIXED_AVATAR["normal_preview"],
        caption=f"{FIXED_AVATAR['pose_name']} ({FIXED_AVATAR['avatar_id']})",
    )

# =================== Voice Recorder (mic_recorder) ===================
# Keep location & behavior the same as the working version; only labels changed.
try:
    from streamlit_mic_recorder import mic_recorder
    _HAS_MIC = True
except Exception:
    mic_recorder = None  # type: ignore
    _HAS_MIC = False

wav_bytes: Optional[bytes] = None
if not _HAS_MIC:
    st.warning("`streamlit-mic-recorder` is not installed.")
else:
    audio = mic_recorder(
        start_prompt="Speak",
        stop_prompt="Stop",
        just_once=False,
        use_container_width=False,
        key="mic_recorder",
        format="wav",
    )
    if isinstance(audio, dict) and "bytes" in audio:
        wav_bytes = audio["bytes"]
        debug(f"[mic] received {len(wav_bytes)} bytes")
    elif isinstance(audio, (bytes, bytearray)):
        wav_bytes = bytes(audio)
        debug(f"[mic] received {len(wav_bytes)} bytes (raw)")
    else:
        debug("[mic] waiting for recording…")

# ---- Audio playback (ABOVE transcript), plus simple fallback transcript like original
if wav_bytes:
    st.audio(wav_bytes, format="audio/wav", autoplay=False)
    # Fallback stub text so pipeline still works without Whisper
    ss.last_text = "Thanks! (audio captured)"
    debug(f"[voice→text] {ss.last_text}")

# ---- Prompt input (chat_input, 2-line look)
placeholder = f"{ss.Name}, You need to Press the 'Speak' button and post your Question and once you complet your sentence press [Stop]. This will help to edit the sentence before we send it to Chat GPT."
user_msg = st.chat_input(placeholder)
if user_msg:
    ss.last_text = (user_msg or "").strip()
    debug(f"[user] {ss.last_text}")

# ============ Actions (Test-1 and ChatGPT) ============
# Keep same action buttons/behavior as before; only viewer/session controls moved.
col1, col2 = st.columns(2, gap="small")
with col1:
    if st.button("Test-1", use_container_width=True):
        if not (ss.session_id and ss.session_token and ss.offer_sdp):
            st.warning("Start a session first.")
        else:
            send_text_to_avatar(ss.session_id, ss.session_token, "Hello. Welcome to the test demonstration.")
with col2:
    if st.button("ChatGPT", use_container_width=True):
        if not (ss.session_id and ss.session_token and ss.offer_sdp):
            st.warning("Start a session first.")
        else:
            # Minimal OpenAI call using requests like original
            OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
            OPENAI_HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-5-nano",
                "messages": [
                    {"role": "system", "content": "You are a clear, concise assistant."},
                    {"role": "user", "content": ss.last_text or ""},
                ],
                "temperature": 0.6,
                "max_tokens": 600,
            }
            try:
                r = requests.post(OPENAI_CHAT_URL, headers=OPENAI_HEADERS, data=json.dumps(payload), timeout=60)
                body = r.json()
                reply = (body.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                ss.last_reply = reply
                debug(f"[openai] status {r.status_code}")
                # Speak back via avatar (simple, no chunking)
                if reply:
                    send_text_to_avatar(ss.session_id, ss.session_token, reply)
            except Exception as e:
                st.error("ChatGPT call failed. See Debug for details.")
                debug(f"[openai error] {repr(e)}")

# -------------- LLM Reply (read-only) --------------
if ss.get("last_reply"):
    st.subheader("ChatGPT Reply (read-only)")
    st.text_area("", value=ss.last_reply, height=160, label_visibility="collapsed")

# -------------- Debug box --------------
st.text_area("Debug", value="\n".join(ss.debug_buf), height=220, disabled=True)


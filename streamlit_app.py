# Avatharam-2
# ver-0.0
# Interface to ChatGPT (Send transcript → get reply → avatar reads it)
# HeyGen avatar + mic_recorder + (optional) faster-whisper
# Changes in this version:
# 1) Button label renamed from "Test-2 (Send transcript)" to "ChatGPT".
# 2) Button now sends the transcript to OpenAI, gets a response, then feeds
#    that response to the HeyGen avatar to read out (chunked if long).

import json
import os
import time
import tempfile
from pathlib import Path
from typing import Optional, List

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components

# --- Optional STT backend ---
try:
    from faster_whisper import WhisperModel  # local inference
    _HAS_FWHISPER = True
except Exception:
    WhisperModel = None  # type: ignore
    _HAS_FWHISPER = False

try:
    from streamlit_mic_recorder import mic_recorder
    _HAS_MIC = True
except Exception:
    mic_recorder = None  # type: ignore
    _HAS_MIC = False

# ---------------- Page ----------------
st.set_page_config(page_title="Avatharam-2", layout="centered")
st.text("by Krish Ambady")
st.title("Avatharam-2")

st.markdown(
    """
<style>
  .block-container { padding-top:.6rem; padding-bottom:1rem; }
  iframe { border:none; border-radius:16px; }
  .rowbtn .stButton>button { height:40px; font-size:.95rem; border-radius:12px; }
  .hint { font-size:.9rem; opacity:.75; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------- Secrets ---------------

def _get(s: dict, *keys, default=None):
    cur = s
    try:
        for k in keys:
            cur = cur[k]
        return cur
    except Exception:
        return default

SECRETS = st.secrets if "secrets" in dir(st) else {}
HEYGEN_API_KEY = (
    _get(SECRETS, "HeyGen", "heygen_api_key")
    or _get(SECRETS, "heygen", "heygen_api_key")
    or os.getenv("HEYGEN_API_KEY")
)
OPENAI_API_KEY = (
    _get(SECRETS, "openai", "secret_key")
    or os.getenv("OPENAI_API_KEY")
)

if not HEYGEN_API_KEY:
    st.error(
        "Missing HeyGen API key in `.streamlit/secrets.toml`.\n\n[HeyGen]\nheygen_api_key = \"…\""
    )
    st.stop()

if not OPENAI_API_KEY:
    st.error(
        "Missing OpenAI API key in `.streamlit/secrets.toml`.\n\n[openai]\nsecret_key = \"sk-…\""
    )
    st.stop()

# --------------- HeyGen Endpoints --------------
BASE = "https://api.heygen.com/v1"
API_LIST_AVATARS = f"{BASE}/streaming/avatar.list"  # GET (x-api-key)
API_STREAM_NEW = f"{BASE}/streaming.new"  # POST (x-api-key)
API_CREATE_TOKEN = f"{BASE}/streaming.create_token"  # POST (x-api-key)
API_STREAM_TASK = f"{BASE}/streaming.task"  # POST (Bearer)
API_STREAM_STOP = f"{BASE}/streaming.stop"  # POST (Bearer)

HEADERS_XAPI = {
    "accept": "application/json",
    "x-api-key": HEYGEN_API_KEY,
    "Content-Type": "application/json",
}


def headers_bearer(tok: str):
    return {
        "accept": "application/json",
        "Authorization": f"Bearer {tok}",
        "Content-Type": "application/json",
    }


# --------- Debug buffer ----------
ss = st.session_state
ss.setdefault("debug_buf", [])


def debug(msg: str):
    ss.debug_buf.append(str(msg))
    if len(ss.debug_buf) > 1000:
        ss.debug_buf[:] = ss.debug_buf[-1000:]


# ------------- HTTP helpers --------------

def _get(url, params=None):
    r = requests.get(url, headers=HEADERS_XAPI, params=params, timeout=45)
    raw = r.text
    try:
        body = r.json()
    except Exception:
        body = {"_raw": raw}
    debug(f"[GET] {url} -> {r.status_code}")
    if r.status_code >= 400:
        debug(raw)
        r.raise_for_status()
    return r.status_code, body, raw


def _post_xapi(url, payload=None):
    r = requests.post(
        url, headers=HEADERS_XAPI, data=json.dumps(payload or {}), timeout=60
    )
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
    r = requests.post(
        url, headers=headers_bearer(token), data=json.dumps(payload or {}), timeout=60
    )
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


# --------- Avatars (ACTIVE only) ---------
@st.cache_data(ttl=300)
def fetch_interactive_avatars():
    _, body, _ = _get(API_LIST_AVATARS)
    items = []
    for a in (body.get("data") or []):
        if isinstance(a, dict) and a.get("status") == "ACTIVE":
            items.append(
                {
                    "label": a.get("pose_name") or a.get("avatar_id"),
                    "avatar_id": a.get("avatar_id"),
                    "default_voice": a.get("default_voice"),
                }
            )
    seen, out = set(), []
    for it in items:
        aid = it.get("avatar_id")
        if aid and aid not in seen:
            seen.add(aid)
            out.append(it)
    return out


avatars = fetch_interactive_avatars()
if not avatars:
    st.error("No ACTIVE interactive avatars returned by HeyGen.")
    st.stop()

# Default to Alessandra if present
default_idx = 0
for i, a in enumerate(avatars):
    if a["avatar_id"] == "Alessandra_CasualLook_public":
        default_idx = i
        break

choice = st.selectbox(
    "Choose an avatar", [a["label"] for a in avatars], index=default_idx
)
selected = next(a for a in avatars if a["label"] == choice)


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
    tok = (body.get("data") or {}).get("token") or (body.get("data") or {}).get(
        "access_token"
    )
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


def stop_session(session_id: str, session_token: str):
    try:
        _post_bearer(API_STREAM_STOP, session_token, {"session_id": session_id})
    except Exception as e:
        debug(f"[stop_session] {e}")


# --------- OpenAI (ChatGPT) helpers ---------
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}


def chatgpt_reply(user_text: str, system: str = "You are a clear, concise assistant.") -> str:
    payload = {
        "model": "gpt-40-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.6,
        "max_tokens": 600,
    }
    r = requests.post(OPENAI_CHAT_URL, headers=OPENAI_HEADERS, data=json.dumps(payload), timeout=60)
    raw = r.text
    try:
        body = r.json()
    except Exception:
        debug(f"[openai] non-json: {raw[:240]}")
        r.raise_for_status()
        raise
    debug(f"[openai] status {r.status_code}")
    if r.status_code >= 400:
        debug(raw)
        r.raise_for_status()
    try:
        return (body["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        debug(f"[openai] unexpected response: {body}")
        raise RuntimeError("OpenAI response parsing failed")


def _chunk_text(text: str, limit: int = 400) -> List[str]:
    """Split text into <limit> char chunks at sentence boundaries where possible."""
    t = " ".join(text.split())
    if len(t) <= limit:
        return [t]
    out: List[str] = []
    cur = []
    cur_len = 0
    for token in t.split(" "):
        if cur_len + 1 + len(token) > limit:
            out.append(" ".join(cur).strip())
            cur = [token]
            cur_len = len(token)
        else:
            cur.append(token)
            cur_len += (len(token) + (1 if cur_len > 0 else 0))
    if cur:
        out.append(" ".join(cur).strip())
    return out


# ---------- Streamlit state ----------
ss.setdefault("session_id", None)
ss.setdefault("session_token", None)
ss.setdefault("offer_sdp", None)
ss.setdefault("rtc_config", None)
ss.setdefault("last_text", "")
ss.setdefault("last_reply", "")

# -------------- Controls row --------------
st.write("")
c1, c2 = st.columns(2)
with c1:
    if st.button("Start / Restart", use_container_width=True):
        if ss.session_id and ss.session_token:
            stop_session(ss.session_id, ss.session_token)
            time.sleep(0.2)
        debug("Step 1: streaming.new")
        payload = new_session(selected["avatar_id"], selected.get("default_voice"))
        sid, offer_sdp, rtc_config = (
            payload["session_id"],
            payload["offer_sdp"],
            payload["rtc_config"],
        )
        debug("Step 2: streaming.create_token")
        tok = create_session_token(sid)
        debug("Step 3: sleep 1.0s before viewer")
        time.sleep(1.0)
        ss.session_id, ss.session_token = sid, tok
        ss.offer_sdp, ss.rtc_config = offer_sdp, rtc_config
        debug(f"[ready] session_id={sid[:8]}…")
with c2:
    if st.button("Stop", type="secondary", use_container_width=True):
        if ss.session_id and ss.session_token:
            stop_session(ss.session_id, ss.session_token)
        ss.session_id = None
        ss.session_token = None
        ss.offer_sdp = None
        ss.rtc_config = None
        debug("[stopped] session cleared")

# ----------- Viewer embed -----------
viewer_path = Path(__file__).parent / "viewer.html"
if not viewer_path.exists():
    st.warning("viewer.html not found next to streamlit_app.py.")
else:
    if ss.session_id and ss.session_token and ss.offer_sdp:
        html = (
            viewer_path.read_text(encoding="utf-8")
            .replace("__SESSION_TOKEN__", ss.session_token)
            .replace("__AVATAR_NAME__", selected["label"])
            .replace("__SESSION_ID__", ss.session_id)
            .replace("__OFFER_SDP__", json.dumps(ss.offer_sdp)[1:-1])  # raw newlines
            .replace("__RTC_CONFIG__", json.dumps(ss.rtc_config or {}))
        )
        components.html(html, height=340, scrolling=False)
    else:
        st.info("Click **Start / Restart** to open a session and load the viewer.")

# =================== Voice Recorder (mic_recorder) ===================

wav_bytes: Optional[bytes] = None
if not _HAS_MIC:
    st.warning("`streamlit-mic-recorder` is not installed.")
else:
    # Use a STABLE key so state isn’t lost on rerun.
    audio = mic_recorder(
        start_prompt="Start",
        stop_prompt="Stop",
        just_once=False,
        use_container_width=False,
        key="mic_recorder",
        format="wav",
    )

    if audio is None:
        debug("[mic] waiting for recording…")
    else:
        # mic_recorder returns dict with .bytes after Stop
        if isinstance(audio, dict) and "bytes" in audio:
            wav_bytes = audio["bytes"]
            debug(f"[mic] received {len(wav_bytes)} bytes")
        elif isinstance(audio, (bytes, bytearray)):
            wav_bytes = bytes(audio)
            debug(f"[mic] received {len(wav_bytes)} bytes (raw)")
        else:
            debug(f"[mic] unexpected payload: {type(audio)}")

# ---- Audio playback (ABOVE transcript)
if wav_bytes:
    st.audio(wav_bytes, format="audio/wav", autoplay=False)

    # Transcribe (fast-path with faster-whisper, else stub)
    text = ""
    if _HAS_FWHISPER:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp.flush()
                tmp_path = tmp.name
            with st.spinner("Transcribing…"):
                model = WhisperModel("base", compute_type="int8")
                segments, info = model.transcribe(
                    tmp_path, language="en", vad_filter=True
                )
                text = " ".join([seg.text for seg in segments]).strip()
        except Exception as e:
            debug(f"[whisper] {e}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    if not text:
        # Fallback stub text (lets the pipeline work even without Whisper)
        import wave, io as _io

        try:
            with wave.open(_io.BytesIO(wav_bytes), "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                secs = frames / float(rate or 16000)
            text = f"I heard you for about {secs:.1f} seconds."
        except Exception:
            text = "Thanks! (audio captured)"

    ss.last_text = text
    debug(f"[voice→text] {text if text else '(empty)'}")

# ---- Transcript box (editable)
st.subheader("Transcript")
ss.last_text = st.text_area(" ", value=ss.last_text, height=140, label_visibility="collapsed")

# ============ Actions ============
st.write("")
col1, col2 = st.columns(2, gap="small")
with col1:
    if st.button("Test-1", use_container_width=True):
        if not (ss.session_id and ss.session_token and ss.offer_sdp):
            st.warning("Start a session first.")
        else:
            send_text_to_avatar(
                ss.session_id,
                ss.session_token,
                "Hello. Welcome to the test demonstration.",
            )
with col2:
    if st.button("ChatGPT", use_container_width=True):
        if not (ss.session_id and ss.session_token and ss.offer_sdp):
            st.warning("Start a session first.")
        else:
            user_text = (ss.last_text or "").strip()
            if not user_text:
                st.warning("Transcript is empty.")
            else:
                try:
                    with st.spinner("Asking ChatGPT…"):
                        reply = chatgpt_reply(user_text)
                    ss.last_reply = reply
                    # Speak back via avatar (chunk if long)
                    chunks = _chunk_text(reply, limit=380)
                    for i, ck in enumerate(chunks, 1):
                        send_text_to_avatar(ss.session_id, ss.session_token, ck)
                        # small pacing gap between chunks
                        time.sleep(0.2)
                    st.success("ChatGPT reply sent to avatar.")
                except Exception as e:
                    st.error("ChatGPT call failed. See Debug for details.")
                    debug(f"[openai error] {repr(e)}")

# -------------- LLM Reply (read-only) --------------
if ss.get("last_reply"):
    st.subheader("ChatGPT Reply (read-only)")
    st.text_area("", value=ss.last_reply, height=160, label_visibility="collapsed")

# -------------- Debug box --------------
st.text_area("Debug", value="\n".join(ss.debug_buf), height=220, disabled=True)

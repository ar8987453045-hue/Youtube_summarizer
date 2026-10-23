import json
import re
import urllib.parse
import urllib.request
from urllib.parse import urlparse, parse_qs

import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

# Must be the first Streamlit command in the file.
st.set_page_config(page_title="YouTube Summarizer", page_icon="🎬")

# ======================================================================
# SETTINGS (change these to customize the app)
# ======================================================================

# The AI models the user can choose from (name shown -> Hugging Face model name)
MODELS = {
    "Fast (smallest, quickest)": "sshleifer/distilbart-cnn-6-6",
    "Balanced (recommended)": "sshleifer/distilbart-cnn-12-6",
    "Best quality (large, slow)": "facebook/bart-large-cnn",
}

# Roughly how many words the FINAL summary should have for each length
TARGET_WORDS = {"Short": 80, "Medium": 180, "Long": 350}

CHUNK_WORDS = 350  # the AI reads the transcript in pieces of this many words


# ======================================================================
# HELPER 1: get the video ID from a YouTube link
# ======================================================================
def get_video_id(url):
    url = url.strip()
    parsed = urlparse(url)
    host = parsed.hostname

    # Short links: https://youtu.be/VIDEO_ID
    if host == "youtu.be":
        return parsed.path[1:] or None

    if host in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        # Normal links: https://www.youtube.com/watch?v=VIDEO_ID
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        # Shorts / embed / live links: /shorts/ID, /embed/ID, /live/ID
        parts = parsed.path.split("/")
        if len(parts) > 2 and parts[1] in ("shorts", "embed", "live"):
            return parts[2]
    return None


# ======================================================================
# HELPER 2: get the video title (no API key needed)
# ======================================================================
@st.cache_data(show_spinner=False)
def get_video_info(video_id):
    """Returns (title, channel). If anything fails, returns (None, None)."""
    try:
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        oembed = ("https://www.youtube.com/oembed?format=json&url="
                  + urllib.parse.quote(watch_url, safe=""))
        with urllib.request.urlopen(oembed, timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("title"), data.get("author_name")
    except Exception:
        return None, None


# ======================================================================
# HELPER 3: get the transcript (English, or translated to English)
# ======================================================================
@st.cache_data(show_spinner=False)
def get_transcript(video_id):
    api = YouTubeTranscriptApi()
    available = list(api.list(video_id))  # every caption track of the video
    if not available:
        raise ValueError("This video has no captions.")

    # Prefer English captions; human-made ones before auto-generated ones.
    english = [t for t in available if t.language_code.lower().startswith("en")]
    if english:
        english.sort(key=lambda t: t.is_generated)  # False (human) sorts first
        chosen = english[0]
    else:
        chosen = available[0]  # e.g. Hindi

    original_language = chosen.language
    translated = False

    # Not English? Ask YouTube to translate the captions into English,
    # because our AI summarizer only understands English.
    if not chosen.language_code.lower().startswith("en"):
        if chosen.is_translatable:
            chosen = chosen.translate("en")
            translated = True
        else:
            raise ValueError(
                f"The captions are in {original_language} and cannot be translated."
            )

    text = " ".join(snippet.text for snippet in chosen.fetch())
    text = re.sub(r"\s+", " ", text).strip()  # tidy up line breaks and spaces
    return {"text": text, "language": original_language, "translated": translated}


# ======================================================================
# HELPER 4: load the AI model (only once per model)
# ======================================================================
@st.cache_resource(show_spinner=False)
def load_summarizer(model_name):
    return pipeline("summarization", model=model_name)


# ======================================================================
# HELPER 5: summarize long text (the "better quality" method)
# ======================================================================
def split_into_chunks(text, size=CHUNK_WORDS):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def summarize_piece(summarizer, piece, max_tokens):
    """Summarize one chunk. Keeps the limits sensible for short chunks."""
    n_words = len(piece.split())
    max_len = max(20, min(max_tokens, int(n_words * 0.7)))
    min_len = max(10, min(30, max_len // 2))
    result = summarizer(piece, max_length=max_len, min_length=min_len,
                        do_sample=False, truncation=True)
    return result[0]["summary_text"].strip()


def summarize(text, summarizer, length_name, progress_bar):
    """
    Two-stage summarizing:
      1. Summarize every chunk of the transcript.
      2. If the joined result is still longer than we want, summarize
         the summaries again (up to 3 rounds) until it is short enough.
    """
    target = TARGET_WORDS[length_name]
    chunks = split_into_chunks(text)

    # Stage 1: summarize each chunk
    first_tokens = {"Short": 60, "Medium": 110, "Long": 170}[length_name]
    pieces = []
    for i, chunk in enumerate(chunks):
        pieces.append(summarize_piece(summarizer, chunk, first_tokens))
        progress_bar.progress((i + 1) / len(chunks) * 0.9)
    combined = " ".join(pieces)

    # Stage 2: shrink until it is about the right length
    for _ in range(3):
        if len(combined.split()) <= target * 1.3:
            break
        chunks2 = split_into_chunks(combined)
        tokens_each = int(target * 1.4 / len(chunks2))
        tokens_each = max(30, min(200, tokens_each))
        combined = " ".join(summarize_piece(summarizer, c, tokens_each) for c in chunks2)
        if len(chunks2) == 1:
            break

    progress_bar.progress(1.0)
    return combined


# ======================================================================
# HELPER 6: turn the summary into bullet points
# ======================================================================
def to_bullets(summary):
    sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
    bullets, seen = [], set()
    for s in sentences:
        s = s.strip()
        key = s.lower()
        if len(s) > 3 and key not in seen:  # skip empty and repeated sentences
            seen.add(key)
            bullets.append(s)
    return bullets


# ======================================================================
# HELPER 7: friendly error messages
# ======================================================================
def friendly_error(e):
    name = type(e).__name__
    if name == "TranscriptsDisabled":
        return "This video has captions turned off, so it cannot be summarized."
    if name == "NoTranscriptFound":
        return "No usable captions were found for this video."
    if name in ("VideoUnavailable", "VideoUnplayable"):
        return "This video is unavailable (private, deleted or region-blocked)."
    if name in ("RequestBlocked", "IpBlocked"):
        return ("YouTube is blocking requests from this server. This often happens "
                "on cloud hosting. The app should still work when run on your own computer.")
    return f"Something went wrong: {e}"


# ======================================================================
# THE WEB PAGE
# ======================================================================
st.title("🎬 YouTube Video Summarizer")
st.write("By ~Ashish ranjan.........")
st.write("Paste a YouTube link and get a short summary. Works with English "
         "videos and with videos in other languages (Hindi, etc.), which are "
         "translated to English first.")

# ----- Settings in the sidebar -----
with st.sidebar:
    st.header("Settings")
    length_name = st.select_slider("Summary length", options=list(TARGET_WORDS),
                                   value="Medium")
    model_label = st.selectbox("AI model", list(MODELS), index=1)
    st.caption("The first time you use a model, it is downloaded once "
               "(this can take a few minutes). 'Best quality' needs a lot of "
               "memory and may not work on free cloud hosting.")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

# ----- When the button is clicked: do the work and SAVE the result -----
# We save the result in st.session_state. Streamlit re-runs this whole file
# whenever you touch any widget (like the download button), and the saved
# result is what keeps the summary on screen.
if st.button("Summarize", type="primary"):
    video_id = get_video_id(url)

    if not video_id:
        st.error("That doesn't look like a valid YouTube link.")
    else:
        try:
            with st.spinner("Getting transcript..."):
                info = get_transcript(video_id)
            title, channel = get_video_info(video_id)

            if len(info["text"].split()) < 30:
                raise ValueError("The transcript is too short to summarize.")

            with st.spinner("Loading the AI model (first time = download)..."):
                summarizer = load_summarizer(MODELS[model_label])

            st.write("Summarizing...")
            bar = st.progress(0.0)
            summary = summarize(info["text"], summarizer, length_name, bar)
            bar.empty()

            st.session_state["result"] = {
                "video_id": video_id,
                "url": url.strip(),
                "title": title,
                "channel": channel,
                "summary": summary,
                "bullets": to_bullets(summary),
                "transcript": info["text"],
                "language": info["language"],
                "translated": info["translated"],
                "length": length_name,
            }
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(friendly_error(e))

# ----- Show the saved result -----
result = st.session_state.get("result")
if result:
    st.divider()

    # Thumbnail (YouTube serves it from a simple address) + title
    st.image(f"https://img.youtube.com/vi/{result['video_id']}/hqdefault.jpg", width=320)
    st.subheader(result["title"] or "Video summary")
    if result["channel"]:
        st.caption(f"by {result['channel']}")

    if result["translated"]:
        st.info(f"The captions were in {result['language']}. They were translated "
                "to English by YouTube, then summarized. The summary is in English.")

    view = st.radio("Show as", ["Bullet points", "Paragraph"], horizontal=True)
    st.subheader(f"Summary ({result['length']})")
    if view == "Bullet points":
        for b in result["bullets"]:
            st.markdown(f"- {b}")
    else:
        st.write(result["summary"])

    # Text file the user can download
    download_text = (
        f"{result['title'] or 'YouTube video'}\n"
        f"{result['url']}\n\n"
        "SUMMARY\n"
        + "\n".join(f"- {b}" for b in result["bullets"])
        + "\n"
    )
    st.download_button("⬇️ Download summary (.txt)", data=download_text,
                       file_name="summary.txt", mime="text/plain")

    with st.expander("Show full transcript"):
        st.write(result["transcript"])

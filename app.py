import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline


# ---------- Helper 1: get the video ID from a YouTube link ----------
def get_video_id(url):
    parsed = urlparse(url)
    # Short links: https://youtu.be/VIDEO_ID
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    # Normal links: https://www.youtube.com/watch?v=VIDEO_ID
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        # Shorts links: https://www.youtube.com/shorts/VIDEO_ID
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]
    return None


# ---------- Helper 2: get the transcript text ----------
def get_transcript(video_id):
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id)  # downloads the captions
    return " ".join(snippet.text for snippet in fetched)


# ---------- Helper 3: load the AI model (only once) ----------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


# ---------- Helper 4: summarize long text ----------
def summarize(text, summarizer):
    # The model can only read ~700 words at a time,
    # so we cut the transcript into chunks of 400 words.
    words = text.split()
    chunks = [" ".join(words[i:i + 400]) for i in range(0, len(words), 400)]

    summaries = []
    for chunk in chunks:
        result = summarizer(chunk, max_length=120, min_length=30, do_sample=False)
        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)


# ---------- The web page ----------
st.title("🎬 YouTube Video Summarizer")
st.write("By~ashish ranjan.....")
st.write("Paste a YouTube link and get a short summary.")

url = st.text_input("YouTube URL")

if st.button("Summarize"):
    video_id = get_video_id(url)

    if not video_id:
        st.error("That doesn't look like a valid YouTube link.")
    else:
        try:
            with st.spinner("Getting transcript..."):
                transcript = get_transcript(video_id)

            with st.spinner("Summarizing (first run downloads the model, please wait)..."):
                summarizer = load_summarizer()
                summary = summarize(transcript, summarizer)

            st.subheader("Summary")
            st.write(summary)

            with st.expander("Show full transcript"):
                st.write(transcript)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

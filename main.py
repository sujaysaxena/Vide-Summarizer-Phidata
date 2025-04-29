import os
import streamlit as st
import time
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv
from phi.agent import Agent 
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
from pathlib import Path
import mimetypes

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Streamlit Page Config
st.set_page_config(
    page_title="Video Summarizer Agent",
    page_icon="📽️",
    layout="wide"
)

st.title("🎥 Video Summarizer Agent")
st.header("This app is powered by Google Gemini + DuckDuckGo Search")

# Initialize the AI Agent
def initialize_agent():
    return Agent(
        name="Video Summarizer Agent",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True
    )

multiagent_agent = initialize_agent()

# Upload video
video_file = st.file_uploader(
    "Upload a video file here", 
    type=['mp4', 'mov', 'avi'], 
    help="Upload a video for AI Analysis"
)

if video_file:
    # Save video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix="mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    # Display the uploaded video
    st.video(video_path, format="video/mp4", start_time=0)

    # User query
    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content. The AI Agent will analyze and gather additional contextual information if needed.",
        help="Provide specific questions or insights you want from the video"
    )

    # Analyze button
    if st.button("🔍 Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("⚠️ Please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("⏳ Processing video and gathering insights..."):
                    # Upload and poll for completion
                    mime_type, _ = mimetypes.guess_type(video_path)
                    processed_video = upload_file(video_path, mime_type=mime_type or "video/mp4")
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    # Generate the prompt
                    analysis_prompt = f"""
                    Analyze the uploaded video for content and context.
                    Respond to the following query using video insights and supplementary web research:
                    {user_query}

                    Provide a detailed, user-friendly, and actionable response.
                    """

                    # Run the AI agent
                    response = multiagent_agent.run(analysis_prompt, videos=[processed_video])

                # Display results
                st.subheader("🧠 AI Analysis Result")
                st.markdown(response.content)

                #Provide a download or copy option
                st.download_button(
                    label="💾 Download Summary",
                    data=response.content,
                    file_name="video_summary.txt",
                    mime="text/plain"
                )

            except Exception as error:
                st.error(f"❌ An error occurred during analysis: {error}")
            finally:
                # Clean up temporary file
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("📂 Please upload a video file to begin analysis.")

# Custom text area height styling
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

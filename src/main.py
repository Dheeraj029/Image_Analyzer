import streamlit as st
import os
import json
import requests  # Needed to download image from URL
from dotenv import load_dotenv
from analyzer import HybridImageAnalyzer, make_decision

# Load environment variables
load_dotenv()

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Hybrid AI Analyzer", page_icon="👁️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .css-card {
        background-color: #262730; padding: 25px; border-radius: 10px;
        border: 1px solid #41444C; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px;
    }
    .ai-title {
        color: #A6E3E9; font-size: 24px; font-weight: 700; margin-bottom: 10px;
        border-bottom: 1px solid #41444C; padding-bottom: 10px;
    }
    .ai-summary { font-size: 16px; line-height: 1.6; color: #E0E0E0; }
    .metric-box {
        background-color: #1E1E24; border: 1px solid #41444C; border-radius: 8px;
        padding: 15px; text-align: center; height: 100px;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
    }
    .metric-label { font-size: 12px; color: #A0A0A0; text-transform: uppercase; margin-bottom: 5px; }
    .metric-value { font-size: 20px; font-weight: 700; color: #FFFFFF; }
    .text-green { color: #00FF7F !important; }
    .text-yellow { color: #FFD700 !important; }
    .text-red { color: #FF4B4B !important; }
    .stProgress > div > div > div > div { background-color: #00CC96; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTION: Download Image safely
# -----------------------------------------------------------------------------
def load_image_from_url(url):
    """
    Downloads an image from a URL and converts it to bytes.
    Returns: (image_bytes, error_message)
    """
    try:
        # User-Agent header mimics a browser to avoid 403 Forbidden errors
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Check if the content is actually an image
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            return None, "The URL did not return an image. It might be a webpage."
            
        return response.content, None
    except requests.exceptions.MissingSchema:
        return None, "Invalid URL. Please include http:// or https://"
    except Exception as e:
        return None, f"Could not download image: {str(e)}"

# -----------------------------------------------------------------------------
# 3. SETUP & SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    vision_key = os.getenv("AZURE_VISION_KEY")
    openai_key = os.getenv("AZURE_OPENAI_KEY")
    
    status_v = "✅ Connected" if vision_key else "❌ Missing"
    status_ai = "✅ Connected" if openai_key else "❌ Missing"
    
    st.write(f"**Vision API:** {status_v}")
    st.write(f"**OpenAI GPT:** {status_ai}")
    st.divider()
    st.info("Ensure `.env` file is present.")

# -----------------------------------------------------------------------------
# 4. MAIN APP LOGIC
# -----------------------------------------------------------------------------
st.title("👁️ Hybrid AI Image Analyzer")
st.markdown("### Computer Vision v3.2 + Azure OpenAI GPT-4")

# INPUT TABS
input_mode = st.radio("Select Input Mode:", ["Upload Image", "Image URL"], horizontal=True, label_visibility="collapsed")

final_image_bytes = None
display_image = None

# --- HANDLE INPUTS ---
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        final_image_bytes = uploaded_file.getvalue()
        display_image = uploaded_file

else:
    url_input = st.text_input("Paste Image URL:", placeholder="https://example.com/image.jpg")
    if url_input:
        with st.spinner("Downloading image..."):
            # Download image locally first to avoid Azure 400 errors
            img_data, error = load_image_from_url(url_input.strip())
            
            if error:
                st.error(f"❌ {error}")
            else:
                final_image_bytes = img_data
                display_image = img_data  # Streamlit can display raw bytes

# --- ANALYZE LOGIC ---
if final_image_bytes and display_image:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(display_image, caption="Source Image", use_container_width=True)
        analyze_btn = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

    if analyze_btn:
        with col2:
            if not vision_key or not openai_key:
                st.error("Missing API Keys in .env file.")
            else:
                with st.spinner("🤖 Vision API analyzing..."):
                    analyzer = HybridImageAnalyzer(
                        os.getenv("AZURE_VISION_ENDPOINT"),
                        os.getenv("AZURE_VISION_KEY"),
                        os.getenv("AZURE_OPENAI_ENDPOINT"),
                        os.getenv("AZURE_OPENAI_KEY"),
                        os.getenv("AZURE_OPENAI_DEPLOYMENT")
                    )
                    
                    # SEND BYTES ALWAYS (Fixes 400 Bad Request for URLs)
                    vision_result = analyzer.analyze_visual_features(final_image_bytes)
                
                if "error" in vision_result:
                    st.error(f"Vision API Error: {vision_result['error']}")
                else:
                    # Get AI Title & Summary
                    with st.spinner("🧠 GPT-4 generating title and summary..."):
                        ai_response = analyzer.generate_human_summary(vision_result)
                        ai_title = ai_response['title']
                        ai_summary = ai_response['summary']

                    # --- UI DISPLAY ---
                    st.markdown(f"""
                        <div class="css-card">
                            <div class="ai-title">✨ {ai_title}</div>
                            <div class="ai-summary">{ai_summary}</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Prepare Metrics
                    try:
                        confidence = vision_result["description"]["captions"][0]["confidence"]
                        decision = make_decision(confidence)
                    except:
                        confidence = 0.0
                        decision = "Uncertain"

                    decision_color_class = "text-red"
                    if "High" in decision: decision_color_class = "text-green"
                    elif "Moderate" in decision: decision_color_class = "text-yellow"

                    # Metrics Row
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-label">Confidence</div>
                                <div class="metric-value">{confidence:.1%}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-label">Decision</div>
                                <div class="metric-value {decision_color_class}" style="font-size: 18px;">
                                    {decision}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    with m3:
                        obj_count = len(vision_result.get("objects", []))
                        st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-label">Objects</div>
                                <div class="metric-value">{obj_count}</div>
                            </div>
                        """, unsafe_allow_html=True)

                    # Tabs
                    st.write("")
                    tab_obj, tab_tags, tab_json = st.tabs(["📦 Objects", "🏷️ Tags", "⚙️ Raw Data"])
                    
                    with tab_obj:
                        objects = vision_result.get("objects", [])
                        if objects:
                            for obj in objects:
                                st.write(f"**{obj['object'].title()}**")
                                st.progress(obj['confidence'])
                        else:
                            st.info("No specific objects detected.")

                    with tab_tags:
                        tags = [t['name'] for t in vision_result.get("tags", []) if t['confidence'] > 0.5]
                        if tags:
                            st.write(", ".join([f"`{t}`" for t in tags]))
                        else:
                            st.write("No high-confidence tags.")
                        
                    with tab_json:
                        st.json(vision_result)
                        
                        # DOWNLOAD BUTTON
                        st.markdown("---")
                        json_string = json.dumps(vision_result, indent=4)
                        st.download_button(
                            label="📥 Download JSON Result",
                            data=json_string,
                            file_name="analysis_result.json",
                            mime="application/json"
                        )

                    if confidence < 0.5:
                        st.warning("⚠️ System is uncertain. Please verify manually.")

elif input_mode == "Image URL" and not final_image_bytes:
    # Helper text when empty
    st.info("👆 Paste a valid image URL to start (e.g., ends in .jpg or .png)")
else:
    st.info("👆 Upload an image to start.")
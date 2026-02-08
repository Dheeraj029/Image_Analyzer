import streamlit as st
import os
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
    
    /* Standard Card */
    .css-card {
        background-color: #262730; 
        padding: 25px; 
        border-radius: 10px;
        border: 1px solid #41444C; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    
    /* AI Generated Title Style */
    .ai-title {
        color: #A6E3E9;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 10px;
        border-bottom: 1px solid #41444C;
        padding-bottom: 10px;
    }
    
    /* Summary Text */
    .ai-summary {
        font-size: 16px;
        line-height: 1.6;
        color: #E0E0E0;
    }
    
    /* Metric Box Styles */
    .metric-box {
        background-color: #1E1E24;
        border: 1px solid #41444C;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-label { font-size: 12px; color: #A0A0A0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .metric-value { font-size: 20px; font-weight: 700; color: #FFFFFF; }
    
    /* Status Colors */
    .text-green { color: #00FF7F !important; }
    .text-yellow { color: #FFD700 !important; }
    .text-red { color: #FF4B4B !important; }
    
    .stProgress > div > div > div > div { background-color: #00CC96; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SETUP & SIDEBAR
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
# 3. MAIN APP LOGIC
# -----------------------------------------------------------------------------
st.title("👁️ Hybrid AI Image Analyzer")
st.markdown("### Computer Vision v3.2 + Azure OpenAI GPT-4")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption="Source Image", use_container_width=True)
        analyze_btn = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

    if analyze_btn:
        with col2:
            if not vision_key or not openai_key:
                st.error("Missing API Keys in .env file.")
            else:
                with st.spinner("🤖 Vision API analyzing pixels..."):
                    analyzer = HybridImageAnalyzer(
                        os.getenv("AZURE_VISION_ENDPOINT"),
                        os.getenv("AZURE_VISION_KEY"),
                        os.getenv("AZURE_OPENAI_ENDPOINT"),
                        os.getenv("AZURE_OPENAI_KEY"),
                        os.getenv("AZURE_OPENAI_DEPLOYMENT")
                    )
                    
                    # 1. Get Technical Vision Data
                    image_bytes = uploaded_file.getvalue()
                    vision_result = analyzer.analyze_visual_features(image_bytes)
                
                if "error" in vision_result:
                    st.error(f"Vision API Error: {vision_result['error']}")
                else:
                    # 2. Get AI Title & Summary
                    with st.spinner("🧠 GPT-4 generating title and summary..."):
                        ai_response = analyzer.generate_human_summary(vision_result)
                        ai_title = ai_response['title']
                        ai_summary = ai_response['summary']

                    # --- UI DISPLAY ---
                    
                    # AI Summary Card with Title
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

                    if confidence < 0.5:
                        st.warning("⚠️ System is uncertain. Please verify manually.")

else:
    st.info("👆 Upload an image to start.")
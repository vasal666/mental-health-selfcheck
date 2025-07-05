import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import random
import time

# Initialize session state variables
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'scores' not in st.session_state:
    st.session_state.scores = []
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False
if 'total_score' not in st.session_state:
    st.session_state.total_score = 0
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = {'label': 'NEUTRAL', 'score': 0.5}
if 'journal_input' not in st.session_state:
    st.session_state.journal_input = ""

st.set_page_config(
    page_title="MindWell - Mental Health Self-Check", 
    layout="wide",
    page_icon="🧠"
)

# Load Sentiment Model
@st.cache_resource
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_sentiment_model()

# Modern glass-style UI with enhancements
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
    }
    .stButton>button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(37, 117, 252, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #5a0db9 0%, #1c68f0 100%);
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(37, 117, 252, 0.4);
    }
    .secondary-btn>button {
        background: rgba(255, 255, 255, 0.3) !important;
        color: #2c3e50 !important;
        border: 1px solid rgba(37, 117, 252, 0.3) !important;
    }
    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.4);
        padding: 1rem;
        border-radius: 15px;
        transition: background-color 0.3s ease;
    }
    .stRadio > div:hover {
        background-color: rgba(255, 255, 255, 0.6);
    }
    .progress-container {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.4);
        border-radius: 10px;
        margin: 1.5rem 0;
        overflow: hidden;
    }
    .progress-bar {
        height: 12px;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .resource-card {
        background: rgba(255, 255, 255, 0.5);
        border-radius: 15px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .resource-card:hover {
        background: rgba(255, 255, 255, 0.7);
        transform: translateX(5px);
    }
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #555;
        font-size: 0.9rem;
        margin-top: 2rem;
        border-top: 1px solid rgba(0, 0, 0, 0.1);
    }
    .quote-card {
        background: linear-gradient(135deg, #6a11cb20 0%, #2575fc20 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        text-align: center;
        font-style: italic;
        font-size: 1.1rem;
    }
    .risk-card {
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    .analysis-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .analysis-card {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }

    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .progress-container {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.4);
        border-radius: 10px;
        margin: 0.5rem 0;
        height: 8px;
        overflow: hidden;
    }

    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 10px;
    }

    .mood-indicator {
        position: relative;
        height: 40px;
        margin: 1rem 0;
    }

    .mood-scale {
        display: flex;
        justify-content: space-between;
        padding: 0 1rem;
        font-size: 1.5rem;
    }

    .mood-arrow {
        position: absolute;
        top: -10px;
        width: 0; 
        height: 0; 
        border-left: 8px solid transparent;
        border-right: 8px solid transparent;
        border-top: 12px solid #2575fc;
        transform: translateX(-50%);
    }

    .confidence-meter {
        height: 8px;
        background: rgba(0, 0, 0, 0.1);
        border-radius: 4px;
        margin: 1rem 0;
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #2575fc, #6a11cb);
        border-radius: 4px;
    }
    }
    @media (max-width: 768px) {
        .analysis-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
""", unsafe_allow_html=True)

questions = [
    "I feel nervous or anxious frequently.",
    "I struggle to concentrate or stay focused.",
    "I feel sad, down, or hopeless most of the time.",
    "I have trouble sleeping or oversleep often.",
    "I feel tired or have low energy nearly every day.",
    "I avoid social interactions or isolate myself.",
    "I feel worthless or excessively guilty.",
    "I find little interest or pleasure in things I used to enjoy.",
    "I feel overwhelmed and unable to cope."
]

options = {
    0: "Not at all",
    1: "Several days",
    2: "More than half the days",
    3: "Nearly every day"
}

LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    "NEGATIVE": "Negative",
    "NEUTRAL": "Neutral",
    "POSITIVE": "Positive"
}
# Remove premature use of label_map and sentiment here (will be used in analysis section)
st.markdown("""
<div class="glass-card">
    <div style="text-align:center;">
        <h1 style="color:#2c3e50; margin-bottom:0.5rem;">🧠 MindWell - Mental Health Self-Check</h1>
        <p style="color:#555; font-size:1.1rem;">Reflect, express, and receive personalized guidance through an intelligent self-assessment</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar with resources
with st.sidebar:
    st.markdown(f"""
    <div style="padding:1.5rem; border-radius:20px; background:rgba(255,255,255,0.4);">
        <h3 style="color:#2c3e50;">📚 Mental Health Resources</h3>
        <div class="resource-card">
            <strong>National Suicide Prevention Lifeline</strong>
            <p>Call 988 or 1-800-273-8255</p>
        </div>
        <div class="resource-card">
            <strong>Crisis Text Line</strong>
            <p>Text HOME to 741741</p>
        </div>
        <div class="resource-card">
            <strong>BetterHelp Online Therapy</strong>
            <p>www.betterhelp.com</p>
        </div>
        <div class="resource-card">
            <strong>Headspace Meditation</strong>
            <p>www.headspace.com</p>
        </div>
        <div style="margin-top:2rem;">
            <h3 style="color:#2c3e50;">📊 Your Progress</h3>
            <p>Questions completed: {min(st.session_state.step, 9) if not st.session_state.show_result else 9}/9</p>
            <div class="progress-container" style="margin: 0.5rem 0;">
                <div class="progress-bar" style="width: {min(st.session_state.step/9, 1)*100 if not st.session_state.show_result else 100}%;"></div>
            </div>
            <p style="font-size:0.95rem; color:#555;">{('Assessment complete!' if st.session_state.show_result else 'Keep going!')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Progress bar
progress = st.session_state.step / len(questions) if len(questions) > 0 else 0
st.markdown(f"""
<div class="progress-container">
    <div class="progress-bar" style="width: {progress * 100}%"></div>
</div>
""", unsafe_allow_html=True)

# Main content
if not st.session_state.show_result:
    if st.session_state.step < len(questions):
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color:#2c3e50;">Question {st.session_state.step + 1}/9</h3>
            <h4>{questions[st.session_state.step]}</h4>
        """, unsafe_allow_html=True)

        val = st.radio("Select an option:", list(options.values()), key=f"q_{st.session_state.step}")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Next Question", key=f"btn_{st.session_state.step}"):
                val_num = list(options.keys())[list(options.values()).index(val)]
                st.session_state.scores.append(val_num)
                st.session_state.step += 1
                st.rerun()
        with col2:
            if st.session_state.step > 0:
                if st.button("Previous Question", key=f"prev_{st.session_state.step}"):
                    st.session_state.step -= 1
                    if st.session_state.scores:
                        st.session_state.scores.pop()
                    st.rerun()

        st.markdown("""</div>""", unsafe_allow_html=True)
    else:
        st.session_state.total_score = sum(st.session_state.scores)
        st.session_state.show_result = True
        st.rerun()
else:
    if not st.session_state.show_analysis:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#2c3e50;">2️⃣ How are you feeling today?</h3>
            <p>Express your thoughts and emotions freely. This helps us understand your current state better.</p>
        """, unsafe_allow_html=True)

        journal_input = st.text_area("Write about your current mood, thoughts, or anything on your mind:", 
                                   height=200,
                                   placeholder="I've been feeling...",
                                   value=st.session_state.journal_input)

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("Analyze My Mental Health"):
                if journal_input.strip():
                    st.session_state.journal_input = journal_input
                    with st.spinner("Analyzing your emotions..."):
                        time.sleep(1.5)
                        sentiment = sentiment_pipeline(journal_input)[0]
                        st.session_state.sentiment = sentiment
                        st.session_state.show_analysis = True
                        st.rerun()
                else:
                    st.warning("Please share your thoughts before analyzing")
        with col2:
            if st.button("Edit Questions", key="edit_questions"):
                st.session_state.step = 0
                st.session_state.scores = []
                st.session_state.show_result = False
                st.session_state.show_analysis = False
                st.rerun()
        with col3:
            if st.button("Start Over", key="start_over"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        st.markdown("""</div>""", unsafe_allow_html=True)
    else:
        # Display analysis results
        sentiment = st.session_state.sentiment
        journal_input = st.session_state.journal_input
        
        # Map label to human-readable format (fix variable name and logic)
        label_display = LABEL_MAP.get(str(sentiment.get('label', '')).upper(), str(sentiment.get('label', '')))
        label_class = label_display.lower()

        # Fix confidence score logic (should be between 0 and 1)
        try:
            confidence_score = float(sentiment.get('score', 0.5))
        except Exception:
            confidence_score = 0.5
        confidence_score = min(max(confidence_score, 0.0), 1.0)

        # Mood arrow position (fix logic)
        if label_class == 'negative':
            mood_arrow_left = 16
        elif label_class == 'neutral':
            mood_arrow_left = 50
        else:
            mood_arrow_left = 84

        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color:#2c3e50;">🧠 Your Mental Health Analysis</h3>
            <div class="analysis-grid">
                <!-- Score Card -->
                <div class="analysis-card">
                    <div class="metric-icon">📊</div>
                    <h4>Questionnaire Score</h4>
                    <h2 style="color:#2575fc; margin:0.5rem 0;">{st.session_state.total_score}/27</h2>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {min(st.session_state.total_score/27*100, 100)}%;"></div>
                    </div>
                    <p class="metric-description">
                        {'Low risk' if st.session_state.total_score <= 9 else 'Moderate risk' if st.session_state.total_score <= 18 else 'High risk'} level
                    </p>
                </div>
                <!-- Mood Card -->
                <div class="analysis-card">
                    <div class="metric-icon">{'😔' if label_class == 'negative' else '😐' if label_class == 'neutral' else '😊'}</div>
                    <h4>Current Mood</h4>
                    <div class="sentiment-label {label_class}">
                        {label_display}
                        <span class="sentiment-emoji">→</span>
                    </div>
                    <div class="mood-indicator">
                        <div class="mood-scale">
                            <span>😔</span>
                            <span>😐</span>
                            <span>😊</span>
                        </div>
                        <div class="mood-arrow" style="left:{mood_arrow_left}%;"></div>
                    </div>
                </div>
                <!-- Confidence Card -->
                <div class="analysis-card">
                    <div class="metric-icon">🔍</div>
                    <h4>Confidence</h4>
                    <h2 style="color:#2575fc; margin:0.5rem 0;">{confidence_score*100:.0f}%</h2>
                    <div class="confidence-meter">
                        <div class="confidence-fill" style="width:{min(confidence_score*100, 100)}%;"></div>
                    </div>
                    <p class="metric-description">
                        {'Low' if confidence_score < 0.5 else 'Moderate' if confidence_score < 0.8 else 'High'} confidence
                    </p>
                </div>
            </div>
</div>
""", unsafe_allow_html=True)
        # Calculate risk level
        ques_level = st.session_state.total_score / 9  # 0-3 scale
        sent_level = 1.5 if label_display == "Negative" else 0.5 if label_display == "Neutral" else 0
        final_score = ques_level + sent_level
        
        # Display risk assessment
        if final_score <= 2:
            risk_html = """
            <div class="risk-card" style="background:rgba(76, 175, 80, 0.15);">
                <h3 style="color:#2c3e50;">🟢 Low Risk Assessment</h3>
                <p>You're showing strong mental wellness! Keep maintaining these healthy patterns.</p>
            </div>
            """
        elif final_score <= 4:
            risk_html = """
            <div class="risk-card" style="background:rgba(255, 193, 7, 0.15);">
                <h3 style="color:#2c3e50;">🟡 Moderate Risk Assessment</h3>
                <p>You're showing some signs of stress. These suggestions can help you maintain balance.</p>
            </div>
            """
        else:
            risk_html = """
            <div class="risk-card" style="background:rgba(244, 67, 54, 0.15);">
                <h3 style="color:#2c3e50;">🔴 High Risk Assessment</h3>
                <p>Your responses suggest you may be struggling. Please consider these resources.</p>
            </div>
            """
        st.markdown(risk_html, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("<h4 style='color:#2c3e50;'>🧘 Personalized Support Plan</h4>", unsafe_allow_html=True)
        
        # Recommendations based on risk level
        if final_score <= 2:
            st.markdown("""
            <div class="glass-card" style="background:rgba(76, 175, 80, 0.1);">
                <h5>✅ Wellness Maintenance Tips</h5>
                <ul>
                    <li>Practice daily gratitude journaling</li>
                    <li>Get 10-15 minutes of morning sunlight</li>
                    <li>Take nature walks 3 times a week</li>
                    <li>Digital detox for 1 hour daily</li>
                    <li>Try mindfulness meditation</li>
                </ul>
                <div class="quote-card">
                    "The greatest weapon against stress is our ability to choose one thought over another."
                    <div style="margin-top:0.5rem; font-weight:bold;">- William James</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif final_score <= 4:
            st.markdown("""
            <div class="glass-card" style="background:rgba(255, 193, 7, 0.1);">
                <h5>🛠️ Stress Management Strategies</h5>
                <ul>
                    <li>Practice box breathing (4-4-4-4 technique)</li>
                    <li>Limit social media to 30 min/day</li>
                    <li>Journaling prompt: "What's weighing on my mind?"</li>
                    <li>Replace one caffeine drink with herbal tea</li>
                    <li>Establish a consistent sleep schedule</li>
                </ul>
                <div style="text-align:center; margin:1.5rem 0;">
                    <iframe width="100%" height="250" src="https://www.youtube.com/embed/inpok4MKVLM" 
                        title="Breathing Exercise" frameborder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen style="border-radius:12px;">
                    </iframe>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="background:rgba(244, 67, 54, 0.1);">
                <h5>⚠️ Support & Recovery Plan</h5>
                <ul>
                    <li>Reach out to a mental health professional</li>
                    <li>Practice the 5-4-3-2-1 grounding technique</li>
                    <li>Connect with a trusted friend or family member</li>
                    <li>Prioritize 7-9 hours of quality sleep</li>
                    <li>Read "The Happiness Trap" by Russ Harris</li>
                </ul>
                <div style="display:flex; justify-content:space-around; margin-top:1.5rem; flex-wrap:wrap;">
                    <div style="text-align:center; margin:0.5rem;">
                        <img src="https://images.unsplash.com/photo-1506126613408-eca07ce68773?auto=format&fit=crop&w=200" 
                            style="width:100px; height:100px; border-radius:50%; object-fit:cover; border:3px solid #6a11cb;">
                        <p><strong>Talk Therapy</strong></p>
                    </div>
                    <div style="text-align:center; margin:0.5rem;">
                        <img src="https://images.unsplash.com/photo-1591744539986-9e4b1d5d1a1f?auto=format&fit=crop&w=200" 
                            style="width:100px; height:100px; border-radius:50%; object-fit:cover; border:3px solid #6a11cb;">
                        <p><strong>Support Groups</strong></p>
                    </div>
                    <div style="text-align:center; margin:0.5rem;">
                        <img src="https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?auto=format&fit=crop&w=200" 
                            style="width:100px; height:100px; border-radius:50%; object-fit:cover; border:3px solid #6a11cb;">
                        <p><strong>Mindfulness</strong></p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional resources
        st.markdown("---")
        st.markdown("<h4 style='color:#2c3e50;'>📚 Additional Resources</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:1rem;">
            <div class="resource-card">
                <strong>National Alliance on Mental Illness (NAMI)</strong>
                <p>Helpline: 1-800-950-NAMI (6264)</p>
            </div>
            <div class="resource-card">
                <strong>Talkspace Online Therapy</strong>
                <p>www.talkspace.com</p>
            </div>
            <div class="resource-card">
                <strong>Calm Meditation App</strong>
                <p>www.calm.com</p>
            </div>
            <div class="resource-card">
                <strong>7 Cups Online Therapy</strong>
                <p>www.7cups.com</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons at the bottom
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 Retake Assessment", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("✏️ Edit Journal", use_container_width=True):
                st.session_state.show_analysis = False
                st.rerun()
        with col3:
            if st.button("📥 Download Summary", use_container_width=True):
                st.success("Summary download started! (This is a demo feature)")
        
        st.markdown("""</div>""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>MindWell Self-Check is for informational purposes only. It is not a substitute for professional medical advice.</p>
    <p>If you're in crisis, please contact the National Suicide Prevention Lifeline at 988 or 1-800-273-TALK (8255).</p>
</div>
""", unsafe_allow_html=True)
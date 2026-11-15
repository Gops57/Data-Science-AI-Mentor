import streamlit as st
import streamlit.components.v1 as components
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime, timezone, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:  # very old Python
    ZoneInfo = None


# ============================================================
# PAGE CONFIGURATION  (must be the first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="Data Science & AI Mentor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MODEL / PATHS
# ============================================================

MODEL_PATH = "/content/merged_model"

LOTTIE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "welcome_animation.json"
)


# ============================================================
# HELPERS
# ============================================================

def clean_html(text):
    """
    Remove leading indentation and blank lines from an HTML / CSS string.

    Streamlit's st.markdown() uses Markdown rules: any line indented by
    4+ spaces (or a blank line inside an HTML block) turns the rest into a
    CODE BLOCK. That is what made raw HTML / CSS appear on the dashboard.
    Flattening the string avoids this completely.
    """
    return "\n".join(
        line.strip() for line in text.splitlines() if line.strip()
    )


def render_html(text):
    st.markdown(clean_html(text), unsafe_allow_html=True)


# ============================================================
# INDIA STANDARD TIME
# ============================================================

IST_FALLBACK = timezone(timedelta(hours=5, minutes=30))


def get_ist_time():
    """Current time in IST. Falls back to a fixed UTC+5:30 offset if the
    system has no timezone database (common on Windows / slim images)."""
    try:
        tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else IST_FALLBACK
    except Exception:
        tz = IST_FALLBACK
    return datetime.now(tz).strftime("%I:%M %p")


def get_ist_datetime():
    """Date + time in IST, used in the printed header."""
    try:
        tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else IST_FALLBACK
    except Exception:
        tz = IST_FALLBACK
    return datetime.now(tz).strftime("%d %b %Y, %I:%M %p IST")


# ============================================================
# SUBJECTS
# ============================================================

SUBJECTS = [
    "Python",
    "SQL",
    "Statistics",
    "Machine Learning",
    "Deep Learning",
    "Data Visualization",
    "Data Wrangling"
]


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Hello! I'm your Data Science & AI Mentor.**\n\n"
                "I'm here to help you learn Data Science concepts "
                "through clear explanations, examples and practical "
                "guidance.\n\n"
                "Choose a subject and ask your first question."
            ),
            "time": get_ist_time(),
            "subject": "Welcome"
        }
    ]

DEFAULTS = {
    "subject": "Machine Learning",
    "theme": "Dark",
    "font_size": 16,
    "response_style": "Balanced",
    "response_length": "Medium",
    "show_time": True,
    "show_subject": True,
}

for _key, _value in DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value


# ============================================================
# THEME COLOURS
# (Widgets below use key= so their values are already updated
#  in session_state BEFORE this code runs -> theme and font size
#  change instantly instead of one click late.)
# ============================================================

if st.session_state.theme == "Dark":
    PALETTE = {
        "BG": "#0B0D12",
        "SIDEBAR": "#10131A",
        "CARD": "#151922",
        "CARD2": "#1D222D",
        "INPUT": "#0F1219",
        "TEXT": "#F1F4FA",
        "MUTED": "#9AA4B5",
        "BORDER": "#2A303D",
        "SCHEME": "dark",
    }
else:
    PALETTE = {
        "BG": "#F4F6FA",
        "SIDEBAR": "#FFFFFF",
        "CARD": "#FFFFFF",
        "CARD2": "#EEF1F6",
        "INPUT": "#FFFFFF",
        "TEXT": "#172033",
        "MUTED": "#58657A",
        "BORDER": "#D5DCE8",
        "SCHEME": "light",
    }

PALETTE["ACCENT"] = "#4F46E5"
PALETTE["FONT"] = str(int(st.session_state.font_size))
PALETTE["FONT_SMALL"] = str(max(int(st.session_state.font_size) - 3, 11))

BG = PALETTE["BG"]
TEXT = PALETTE["TEXT"]
MUTED = PALETTE["MUTED"]


# ============================================================
# CSS  (plain string + token replacement -> no { } escaping bugs)
# ============================================================

CSS_TEMPLATE = """
<style>

:root {
    color-scheme: %%SCHEME%%;
    --mentor-bg: %%BG%%;
    --mentor-sidebar: %%SIDEBAR%%;
    --mentor-card: %%CARD%%;
    --mentor-card2: %%CARD2%%;
    --mentor-input: %%INPUT%%;
    --mentor-text: %%TEXT%%;
    --mentor-muted: %%MUTED%%;
    --mentor-border: %%BORDER%%;
    --mentor-accent: %%ACCENT%%;
}

/* ---------- APP BACKGROUND ---------- */

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background-color: %%BG%% !important;
    color: %%TEXT%% !important;
}

[data-testid="stHeader"] button,
[data-testid="stHeader"] svg,
[data-testid="stToolbar"] * {
    color: %%TEXT%% !important;
}

.block-container {
    max-width: 1350px;
    padding-top: 2.5rem;
    padding-bottom: 2rem;
}

/* ---------- GENERAL TEXT ---------- */

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
.stMarkdown p, .stMarkdown li,
[data-testid="stWidgetLabel"] p,
label, label p, label span {
    color: %%TEXT%% !important;
}

hr {
    border-color: %%BORDER%% !important;
    background-color: %%BORDER%% !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
    color: %%MUTED%% !important;
}

/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background-color: %%SIDEBAR%% !important;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid %%BORDER%%;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: %%TEXT%% !important;
}

.sidebar-description {
    color: %%MUTED%% !important;
    font-size: 14px;
    font-weight: 500;
    line-height: 1.6;
}

/* ---------- ALERT (Quick Question box) ---------- */

[data-testid="stAlert"],
[data-testid="stAlertContainer"] {
    background-color: %%CARD2%% !important;
    border: 1px solid %%BORDER%% !important;
    border-radius: 10px !important;
}

[data-testid="stAlert"] * {
    color: %%TEXT%% !important;
}

/* ---------- HERO ---------- */

.hero {
    background: linear-gradient(135deg, #3730A3, #4F46E5, #6D28D9);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 20px;
}

.hero-title {
    color: #FFFFFF !important;
    font-size: 34px;
    font-weight: 800;
}

.hero-subtitle {
    color: #E2E4F5 !important;
    font-size: 15px;
    margin-top: 5px;
}

/* ---------- INFO CARDS ---------- */

.info-card {
    background-color: %%CARD%%;
    border: 1px solid %%BORDER%%;
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    color: %%TEXT%%;
    font-size: 13px;
}

.info-card strong { color: %%TEXT%%; }

/* ---------- WELCOME ---------- */

.welcome-container {
    text-align: center;
    padding: 8px 10px 18px 10px;
}

.welcome-title {
    color: %%TEXT%%;
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 6px;
}

.welcome-description {
    color: %%MUTED%%;
    font-size: 15px;
    line-height: 1.6;
    max-width: 720px;
    margin: 0 auto 18px auto;
}

.subject-title {
    color: %%TEXT%%;
    font-size: 22px;
    font-weight: 750;
    margin-top: 12px;
    margin-bottom: 14px;
}

.mentor-status {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid %%BORDER%%;
    background: %%CARD%%;
    color: %%MUTED%%;
    font-size: 12px;
    margin-bottom: 12px;
}

.current-subject {
    text-align: center;
    color: %%MUTED%%;
    font-size: 13px;
    margin: 12px 0 18px 0;
}

.current-subject strong { color: %%TEXT%%; }

/* ---------- CHAT ---------- */

[data-testid="stChatMessage"] {
    background-color: %%CARD%% !important;
    border: 1px solid %%BORDER%%;
    border-radius: 16px;
    margin-bottom: 12px;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td,
[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 {
    color: %%TEXT%% !important;
    line-height: 1.65 !important;
}

/* FONT SIZE (driven by the Text Size slider) */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td,
[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em {
    font-size: %%FONT%%px !important;
}

[data-testid="stChatMessage"] pre,
[data-testid="stChatMessage"] code {
    font-size: calc(%%FONT%%px - 1px) !important;
    background-color: %%CARD2%% !important;
    color: %%TEXT%% !important;
}

[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p,
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] {
    font-size: %%FONT_SMALL%%px !important;
    color: %%MUTED%% !important;
}

/* ---------- CHAT INPUT ---------- */

[data-testid="stChatInput"],
[data-testid="stChatInput"] > div {
    background-color: %%CARD%% !important;
    border-color: %%BORDER%% !important;
    border-radius: 15px !important;
}

[data-testid="stChatInput"] textarea {
    background-color: %%INPUT%% !important;
    color: %%TEXT%% !important;
    -webkit-text-fill-color: %%TEXT%% !important;
    font-size: %%FONT%%px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: %%MUTED%% !important;
    -webkit-text-fill-color: %%MUTED%% !important;
}

/* ---------- SELECT BOX + DROPDOWN LIST ---------- */

div[data-baseweb="select"] > div {
    background-color: %%INPUT%% !important;
    border: 1px solid %%BORDER%% !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="select"] svg {
    color: %%TEXT%% !important;
    fill: %%TEXT%% !important;
}

div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] [role="option"] {
    background-color: %%CARD%% !important;
    color: %%TEXT%% !important;
}

div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [aria-selected="true"] {
    background-color: %%CARD2%% !important;
}

/* ---------- SLIDER ---------- */

[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    color: %%TEXT%% !important;
}

/* ---------- EXPANDER ---------- */

[data-testid="stExpander"] {
    background-color: %%CARD%% !important;
    border: 1px solid %%BORDER%% !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {
    background-color: %%CARD%% !important;
    color: %%TEXT%% !important;
}

[data-testid="stExpander"] summary *,
[data-testid="stExpander"] p,
[data-testid="stExpander"] label,
[data-testid="stExpander"] span {
    color: %%TEXT%% !important;
}

/* ---------- BUTTONS ---------- */

.stButton > button {
    background-color: %%CARD2%% !important;
    color: %%TEXT%% !important;
    border: 1px solid %%BORDER%% !important;
    border-radius: 10px !important;
    font-weight: 600;
}

.stButton > button * { color: inherit !important; }

.stButton > button:hover,
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background-color: %%ACCENT%% !important;
    border-color: %%ACCENT%% !important;
    color: #FFFFFF !important;
}

/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    color: %%MUTED%%;
    font-size: 12px;
    padding-top: 25px;
}

/* ---------- PRINT-ONLY HEADER (hidden on screen) ---------- */

.print-header { display: none; }

/* ============================================================
   PRINT STYLES  (used by the Print button)
   Prints only the conversation, on white paper, full length.
   ============================================================ */

@media print {

    @page { size: A4; margin: 14mm; }

    :root { color-scheme: light !important; }

    html, body {
        background: #FFFFFF !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }

    html, body, #root, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    section.main, .main, .block-container {
        height: auto !important;
        min-height: 0 !important;
        overflow: visible !important;
        position: static !important;
        background: #FFFFFF !important;
        color: #172033 !important;
        max-width: 100% !important;
        padding: 0 !important;
    }

    /* hide everything except the conversation */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stBottom"],
    [data-testid="stChatInput"],
    [data-testid="stHorizontalBlock"],
    [data-testid="stElementContainer"]:has(.hero),
    [data-testid="stElementContainer"]:has(.subject-title),
    [data-testid="stElementContainer"]:has(.current-subject),
    [data-testid="stElementContainer"]:has(.footer),
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"],
    iframe {
        display: none !important;
    }

    /* printed title block */
    .print-header {
        display: block !important;
        color: #172033 !important;
        border-bottom: 1px solid #D5DCE8;
        padding-bottom: 8px;
        margin-bottom: 14px;
    }

    .print-header .ph-title { font-size: 20px; font-weight: 700; }
    .print-header .ph-meta  { font-size: 12px; color: #58657A !important; margin-top: 2px; }

    /* messages: white paper, dark text, never split across pages */
    [data-testid="stChatMessage"] {
        background: #FFFFFF !important;
        border: 1px solid #D5DCE8 !important;
        break-inside: avoid;
        page-break-inside: avoid;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }

    [data-testid="stChatMessage"] *,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong {
        color: #172033 !important;
    }

    [data-testid="stChatMessage"] pre,
    [data-testid="stChatMessage"] code {
        background: #EEF1F6 !important;
        color: #172033 !important;
        white-space: pre-wrap !important;
        word-break: break-word;
    }

    [data-testid="stChatMessage"] [data-testid="stCaptionContainer"],
    [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] * {
        color: #58657A !important;
    }
}

@media (max-width: 768px) {
    .hero-title { font-size: 22px; }
    .hero-subtitle { font-size: 13px; }
}

</style>
"""

_css = CSS_TEMPLATE
for _name, _val in PALETTE.items():
    _css = _css.replace(f"%%{_name}%%", _val)

render_html(_css)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model


tokenizer, model = load_model()


# ============================================================
# RESPONSE SETTINGS
# ============================================================

length_map = {
    "Short": 150,
    "Medium": 250,
    "Detailed": 400
}

style_map = {
    "Simple":
        "Use simple language and easy examples.",
    "Balanced":
        "Give clear, accurate and student-friendly explanations.",
    "Detailed":
        "Give detailed explanations with concepts, examples and important points."
}


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(subject, question):

    messages = [
        {
            "role": "system",
            "content": (
                "I'm the reasoning engine behind a Data Science "
                "and AI learning assistant.\n\n"
                "Answer the student's question directly.\n"
                "Do not introduce yourself.\n"
                "Do not mention your identity or role.\n"
                "Do not repeat these instructions.\n"
                "Do not include system instructions in the answer.\n"
                "Do not begin with unnecessary greetings.\n"
                "Focus directly on the student's question.\n"
                "Give accurate and student-friendly answers.\n\n"
                + style_map[st.session_state.response_style]
            )
        },
        {
            "role": "user",
            "content": f"Subject: {subject}\n\nQuestion: {question}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=length_map[st.session_state.response_length],
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode only newly generated tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


# ============================================================
# CHAT MESSAGE RENDERING (used for history AND live messages)
# ============================================================

def render_message(message):

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        metadata = []

        if st.session_state.show_subject and message.get("subject"):
            metadata.append(f"📚 {message['subject']}")

        if st.session_state.show_time and message.get("time"):
            metadata.append(f"🕐 {message['time']}")

        if metadata:
            st.caption(" • ".join(metadata))


def welcome_message(text):
    return [
        {
            "role": "assistant",
            "content": text,
            "time": get_ist_time(),
            "subject": "Welcome"
        }
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 AI Mentor")

    render_html(
        """
        <div class="sidebar-description">
        Your personalized Data Science learning assistant
        </div>
        """
    )

    st.divider()

    # QUICK QUESTION

    st.markdown("### 💡 Quick Question")

    quick_questions = {
        "Python": "What are Python decorators?",
        "SQL": "What is an INNER JOIN?",
        "Statistics": "What is standard deviation?",
        "Machine Learning": "What is overfitting?",
        "Deep Learning": "What is backpropagation?",
        "Data Visualization": "When should I use a bar chart?",
        "Data Wrangling": "What is data cleaning?"
    }

    st.info(quick_questions[st.session_state.subject])

    st.divider()

    # CLEAR CHAT

    if st.button("🗑️ Clear Conversation", use_container_width=True):

        st.session_state.messages = welcome_message(
            "👋 **Welcome back!**\n\n"
            "Choose a subject and ask your next Data Science question."
        )
        st.rerun()

    st.divider()

    # CUSTOMIZATION  (widgets use key= -> instant updates)

    with st.expander("⚙️ Customize Mentor"):

        st.markdown("#### 🎨 Appearance")

        st.radio(
            "Theme",
            ["Dark", "Light"],
            key="theme",
            horizontal=True
        )

        st.slider(
            "🔤 Text Size",
            min_value=14,
            max_value=20,
            step=1,
            key="font_size"
        )

        st.divider()

        st.markdown("#### 🧠 Response Preferences")

        st.selectbox(
            "Response Style",
            ["Simple", "Balanced", "Detailed"],
            key="response_style"
        )

        st.selectbox(
            "Response Length",
            ["Short", "Medium", "Detailed"],
            key="response_length"
        )

        st.divider()

        st.markdown("#### 👁️ Display")

        st.checkbox("Show timestamps", key="show_time")

        st.checkbox("Show subject", key="show_subject")

    st.divider()

    # ABOUT

    with st.expander("ℹ️ About"):

        st.markdown(
            "**Data Science & AI Mentor**\n\n"
            "Domain-specific learning assistant for Data Science students.\n\n"
            "**Model:** Phi-3 Mini\n\n"
            "**Fine-tuning:** QLoRA\n\n"
            "**Subjects:** 7\n\n"
            "**Interface:** Streamlit"
        )


# ============================================================
# MAIN WELCOME / DASHBOARD
# ============================================================

welcome_left, welcome_right = st.columns(
    [2.2, 1],
    vertical_alignment="center"
)

with welcome_left:

    render_html(
        """
        <div class="welcome-container">
            <div class="mentor-status">
                🎓 Data Science Learning Assistant
            </div>
            <div class="welcome-title">
                Welcome to Data Science &amp; AI Mentor
            </div>
            <div class="welcome-description">
                Learn Data Science through clear explanations,
                practical examples, and focused guidance across
                seven core subjects.
            </div>
        </div>
        """
    )

with welcome_right:

    lottie_shown = False

    if os.path.exists(LOTTIE_PATH):

        try:
            with open(LOTTIE_PATH, "r", encoding="utf-8") as f:
                lottie_json = json.dumps(json.load(f)).replace("</", "<\\/")

            components.html(
                """
                <div style="display:flex;justify-content:center;
                            align-items:center;height:210px;overflow:hidden;">
                    <lottie-player id="mentor-anim" background="transparent"
                        speed="1" loop autoplay
                        style="width:180px;height:200px;"></lottie-player>
                </div>
                <script src="https://unpkg.com/@lottiefiles/lottie-player@2.0.8/dist/lottie-player.js"></script>
                <script>
                    const animationData = %s;
                    customElements.whenDefined("lottie-player").then(() => {
                        document.getElementById("mentor-anim").load(animationData);
                    });
                </script>
                """ % lottie_json,
                height=220,
                scrolling=False
            )
            lottie_shown = True

        except Exception:
            lottie_shown = False

    if not lottie_shown:
        render_html(
            '<div style="text-align:center;font-size:72px;">🎓</div>'
        )


# HERO

render_html(
    """
    <div class="hero">
        <div class="hero-title">Your Intelligent Learning Companion</div>
        <div class="hero-subtitle">
            Explore concepts, understand difficult topics,
            and strengthen your Data Science knowledge.
        </div>
    </div>
    """
)

# INFORMATION CARDS

c1, c2, c3 = st.columns(3)

with c1:
    render_html(
        """
        <div class="info-card">
            📚 <strong>7</strong><br>Learning Subjects
        </div>
        """
    )

with c2:
    render_html(
        """
        <div class="info-card">
            🧠 <strong>AI Mentor</strong><br>Domain-Focused Learning
        </div>
        """
    )

with c3:
    render_html(
        """
        <div class="info-card">
            ⚡ <strong>QLoRA</strong><br>Fine-Tuned Model
        </div>
        """
    )

st.write("")

# SUBJECT SELECTION

render_html(
    '<div class="subject-title">📚 Choose Your Subject</div>'
)

subject_rows = [SUBJECTS[:4], SUBJECTS[4:]]

for row_index, row in enumerate(subject_rows):

    columns = st.columns(len(row))

    for column, subject in zip(columns, row):

        with column:

            is_selected = st.session_state.subject == subject

            if st.button(
                f"✓ {subject}" if is_selected else subject,
                key=f"batch1_subject_{row_index}_{subject}",
                type="primary" if is_selected else "secondary",
                use_container_width=True
            ):
                st.session_state.subject = subject
                st.rerun()

# ACTIVE SUBJECT INDICATOR  (was a plain string -> printed {MUTED} as text)

render_html(
    f"""
    <div class="current-subject">
        Currently selected:
        <strong>{st.session_state.subject}</strong>
    </div>
    """
)


# ============================================================
# TOOLBAR
# ============================================================

b1, b2, _empty = st.columns([1, 1, 6])

with b1:

    if st.button("🖨️ Print"):
        # invisible iframe that asks the parent page to open the print dialog
        components.html(
            "<script>window.parent.print();</script>",
            height=0
        )

with b2:

    if st.button("🔄 New Chat"):

        st.session_state.messages = welcome_message(
            "👋 **New conversation started.**\n\n"
            "Choose a subject and ask your question."
        )
        st.rerun()


# ============================================================
# PRINT HEADER (invisible on screen, shown only when printing)
# ============================================================

render_html(
    f"""
    <div class="print-header">
        <div class="ph-title">Data Science &amp; AI Mentor</div>
        <div class="ph-meta">
            Conversation • Subject: {st.session_state.subject}
            • Printed {get_ist_datetime()}
        </div>
    </div>
    """
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    render_message(message)


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(f"Ask about {st.session_state.subject}...")


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    subject_now = st.session_state.subject

    user_message = {
        "role": "user",
        "content": question,
        "time": get_ist_time(),
        "subject": subject_now
    }

    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Preparing your answer..."):
            try:
                answer = generate_response(subject_now, question)
            except Exception:
                answer = (
                    "I couldn't generate a response "
                    "right now. Please try again."
                )

    # Timestamp is taken AFTER generation, so the reply shows when it
    # was actually produced (previously it copied the question's time).
    assistant_message = {
        "role": "assistant",
        "content": answer,
        "time": get_ist_time(),
        "subject": subject_now
    }

    st.session_state.messages.append(assistant_message)

    # Re-run so the new reply is drawn by the same code as the history
    # (same style, subject and timestamp line, no duplicate bubbles).
    st.rerun()


# ============================================================
# FOOTER
# ============================================================

render_html(
    '<div class="footer">'
    'Data Science &amp; AI Mentor • Phi-3 Mini • QLoRA • Streamlit'
    '</div>'
)

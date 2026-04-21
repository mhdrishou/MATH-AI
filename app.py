import streamlit as st
from mistralai import Mistral
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytesseract
import cv2
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="MathGPT AI", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {background: linear-gradient(135deg,#0f172a,#1e293b); color:white;}
.glass {background:rgba(255,255,255,0.05);padding:20px;border-radius:15px;backdrop-filter:blur(10px);}
.stButton>button {background:linear-gradient(90deg,#6366f1,#22c55e);color:white;border-radius:10px;}
.user {background:#2563eb;padding:10px;border-radius:10px;margin:5px 0;}
.bot {background:#374151;padding:10px;border-radius:10px;margin:5px 0;}
</style>
""", unsafe_allow_html=True)

# ---------------- NAV ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------------- LANDING ----------------
if st.session_state.page == "home":
    st.title("🧠 MathGPT AI")
    st.subheader("Solve • Scan • Understand")

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.write("📷 Scan math from images\n🧠 AI explanations\n📈 Graphs\n🎯 Topic detection")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Start"):
        st.session_state.page = "chat"

    st.markdown("---")
    st.write("👨‍💻 MUHAMMED RISHAN 10-E")
    st.write("👨‍💻 EHAN AL AMISH 10-E")

# ---------------- CHAT ----------------
elif st.session_state.page == "chat":

    if st.button("⬅ Back"):
        st.session_state.page = "home"

    st.title("🤖 MathGPT AI Solver")

    # API
    api_key = st.secrets["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)

    # Memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat
    for msg in st.session_state.messages:
        style = "user" if msg["role"]=="user" else "bot"
        st.markdown(f"<div class='{style}'>{msg['content']}</div>", unsafe_allow_html=True)

    # ---------------- IMAGE SCANNER ----------------
    st.subheader("📷 Scan Math Problem")

    uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

    def preprocess(image):
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,11,2)
        return thresh

    def clean_text(text):
        text = text.replace("×","*")
        text = text.replace("÷","/")
        text = text.replace("^","**")
        text = text.replace("–","-")
        return text.strip()

    def ai_fix(text):
        response = client.chat.complete(
            model="mistral-small",
            messages=[
                {"role":"system","content":"Fix OCR math into a valid math expression."},
                {"role":"user","content":text}
            ]
        )
        return response.choices[0].message.content

    if uploaded:
        img = Image.open(uploaded)
        st.image(img)

        processed = preprocess(img)

        raw_text = pytesseract.image_to_string(processed)
        cleaned = clean_text(raw_text)
        fixed = ai_fix(cleaned)

        st.write("📝 Detected:", fixed)

        prompt = fixed
    else:
        prompt = st.chat_input("Ask math...")

    # ---------------- FUNCTIONS ----------------
    x = sp.symbols('x')

    def try_sympy(q):
        try:
            expr = sp.sympify(q)
            return sp.solve(expr, x)
        except:
            return None

    def plot(expr):
        try:
            f = sp.lambdify(x, expr, "numpy")
            xv = np.linspace(-10,10,100)
            yv = f(xv)
            fig, ax = plt.subplots()
            ax.plot(xv,yv)
            ax.grid()
            st.pyplot(fig)
        except:
            pass

    def detect_topic(q):
        response = client.chat.complete(
            model="mistral-small",
            messages=[
                {"role":"system","content":"Classify: Algebra, Trigonometry, Calculus, Geometry, Arithmetic. One word."},
                {"role":"user","content":q}
            ]
        )
        return response.choices[0].message.content

    # ---------------- RESPONSE ----------------
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        st.markdown(f"<div class='user'>{prompt}</div>", unsafe_allow_html=True)

        topic = detect_topic(prompt)

        sym = try_sympy(prompt)

        if sym:
            answer = f"🎯 Topic: {topic}\n\n✅ Answer: {sym}"
        else:
            res = client.chat.complete(
                model="mistral-small",
                messages=[
                    {"role":"system","content":"Solve step by step clearly."}
                ] + st.session_state.messages
            )
            answer = f"🎯 Topic: {topic}\n\n" + res.choices[0].message.content

        st.markdown(f"<div class='bot'>{answer}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role":"assistant","content":answer})

        # graph
        try:
            expr = sp.sympify(prompt)
            st.subheader("📈 Graph")
            plot(expr)
        except:
            pass

import json
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
CHATLOG = ROOT / "reports" / "chatlog.jsonl"
TASKS   = ROOT / "reports" / "tasks.json"

st.set_page_config(page_title="Agent Console", layout="wide")
st.title("🧠 Agent Live Console")

left, right = st.columns([2,1])

with left:
    st.subheader("Chat")
    if CHATLOG.exists():
        lines = CHATLOG.read_text(encoding="utf-8").splitlines()
        for line in lines[-200:]:
            try:
                m = json.loads(line)
                role = m.get("role","assistant")
                txt  = m.get("content","")
                # blue-ish for assistant, green-ish for you
                bg = "#0e639c22" if role == "assistant" else "#6a995522"
                st.markdown(
                    f"<div style='padding:8px;border-radius:10px;margin:6px 0;background:{bg}'>"
                    f"<b>{role}</b><br/>{txt}</div>",
                    unsafe_allow_html=True
                )
            except Exception:
                pass
    else:
        st.info("no chat yet — talk to the agent to populate this")

with right:
    st.subheader("Tasks")
    data = {}
    if TASKS.exists():
        try:
            data = json.loads(TASKS.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    for t in data.values():
        st.write(f"**{t.get('name','?')}** — {t.get('note','')}")
        try:
            st.progress(int(float(t.get('pct', 0))))
        except Exception:
            st.progress(0)

st.caption("Click the button to refresh the view.")
if st.button("Refresh"):
    st.experimental_set_query_params(refresh="1")

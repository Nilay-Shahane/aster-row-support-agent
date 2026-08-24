import streamlit as st, uuid, html
from langchain_core.messages import HumanMessage
from graph import graph

st.set_page_config(page_title="Aster & Row Support", page_icon="🪶", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""<style>.app-header{display:flex;align-items:center;gap:.5rem;margin-bottom:.25rem}.app-header h1{margin:0;font-size:1.4rem;font-weight:600}.app-subtitle{color:#6b7280;font-size:1.1rem;margin-bottom:1rem}.badge{display:inline-block;padding:.12rem .5rem;border-radius:5px;font-size:.9rem;font-weight:500;margin:0 .3rem .3rem 0}.badge-route{background:#f3f4f6;color:#374151}.badge-conf-high{background:#ecfdf5;color:#047857}.badge-conf-med{background:#fffbeb;color:#b45309}.badge-conf-low{background:#fef2f2;color:#b91c1c}.badge-handoff{background:#fee2e2;color:#991b1b}.sources-box{background:#f9fafb;padding:.5rem .8rem;margin-top:.5rem;font-size:1rem;border-radius:6px;color:#4b5563}</style>
<div class="app-header"><span style="font-size:1.6rem;">🪶</span><h1>Aster & Row Support</h1></div><div class="app-subtitle">Returns, shipping, and order assistance.</div>""", unsafe_allow_html=True)

def reset_session(): st.session_state.update(thread_id=str(uuid.uuid4()), chat_history=[], graph_state={"messages": []})
if "thread_id" not in st.session_state: reset_session()

def get_conf(c):
    if c is None: return "", ""
    try:
        s = float(c)
        return ("badge-conf-high", f"High ({s:.2f})") if s >= 0.75 else ("badge-conf-med", f"Medium ({s:.2f})") if s >= 0.4 else ("badge-conf-low", f"Low ({s:.2f})")
    except (TypeError, ValueError):
        return {"grounded": ("badge-conf-high", "Grounded"), "conflicting": ("badge-conf-med", "Conflicting"), "insufficient": ("badge-conf-low", "Insufficient")}.get(c, ("badge-conf-med", str(c)))

def render_meta(msg):
    chips = []
    if msg.get("route"): chips.append(f'<span class="badge badge-route">Route: {html.escape(str(msg["route"]))}</span>')
    cls, lbl = get_conf(msg.get("confidence"))
    if cls: chips.append(f'<span class="badge {cls}">Confidence: {lbl}</span>')
    if msg.get("handoff"): chips.append('<span class="badge badge-handoff">⚑ Escalate to Human</span>')
    if msg.get("injection_detected"): chips.append('<span class="badge badge-handoff">⚠ Prompt Injection Flagged</span>')
    if chips: st.markdown(" ".join(chips), unsafe_allow_html=True)
    if msg.get("sources"): st.markdown(f'<div class="sources-box">📚 <b>Sources:</b> {", ".join(map(html.escape, map(str, msg["sources"])))}</div>', unsafe_allow_html=True)

def get_trace(m): return {"route": m.get("route"), "confidence": m.get("confidence"), "handoff_triggered": m.get("handoff"), "sources_used": m.get("sources"), "injection_detected": m.get("injection_detected"), "order_id": m.get("order_id"), "order_context_fetched": bool(m.get("order_context")), "tool_calls": m.get("tool_calls")}

with st.sidebar:
    st.button("🗑️ New Conversation", on_click=reset_session, use_container_width=True)
    st.caption(f"Session: `{st.session_state.thread_id[:8]}`"); st.divider()
    st.markdown("**Try asking**\n\n📦 Order status  \n✈️ Shipping policy  \n🔄 Follow-up questions")

if not st.session_state.chat_history: st.info("**Welcome to Aster & Row Support**\n\nTry asking:\n- Where is my order?\n- What is the return policy?\n- Can I change my shipping address?")

for m in st.session_state.chat_history:
    with st.chat_message(m["role"], avatar="🪶" if m["role"] == "assistant" else "👤"):
        st.write(m["content"])
        if m["role"] == "assistant":
            render_meta(m)
            with st.expander("⚙️ View Turn Trace"): st.json(get_trace(m))

if ui := st.chat_input("Ask about your order or policy..."):
    st.session_state.chat_history.append({"role": "user", "content": ui})
    with st.chat_message("user", avatar="👤"): st.write(ui)
    
    st.session_state.graph_state["messages"].append(HumanMessage(content=ui))
    st.session_state.graph_state["query"] = ui

    with st.chat_message("assistant", avatar="🪶"), st.spinner("Generating..."):
        try:
            res = graph.invoke(st.session_state.graph_state, config={"configurable": {"thread_id": st.session_state.thread_id}})
            st.session_state.graph_state, ans = res, res.get("answer", "I couldn't generate an answer.")
            st.write(ans)
            
            md = {"role": "assistant", "content": ans, "sources": res.get("sources_used", []), "handoff": res.get("handoff", False), "route": res.get("route"), "confidence": res.get("confidence"), "injection_detected": res.get("injection_detected", False), "order_id": res.get("order_id"), "order_context": res.get("order_context"), "tool_calls": res.get("tool_calls", [])}
            render_meta(md)
            with st.expander("⚙️ View Turn Trace"): st.json(get_trace(md))
            st.session_state.chat_history.append(md)
        except Exception as e:
            st.error("I encountered a system error. Please try again.")
            with st.expander("Error Details"): st.write(str(e))
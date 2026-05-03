"""
app.py — Streamlit UI for the Intelligent Reading Comprehension System

Screens:
  1. Article Input
  2. Question & Answer Quiz View
  3. Hint Panel
  4. Developer / Analytics Dashboard
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title='RC Quiz System',
    page_icon='📚',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    .main .block-container { padding-top: 1.5rem; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
    }

    /* ── Cards ── */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(102,126,234,0.3);
        margin-bottom: 16px;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p { font-size: 0.9rem; margin: 4px 0 0; opacity: 0.9; }

    .info-card {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 16px;
        padding: 20px 24px;
        color: white;
        box-shadow: 0 4px 20px rgba(17,153,142,0.3);
        margin: 12px 0;
    }

    .error-card {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        border-radius: 16px;
        padding: 20px 24px;
        color: white;
        box-shadow: 0 4px 20px rgba(235,51,73,0.3);
        margin: 12px 0;
    }

    .hint-card {
        background: #fffbea;
        border-left: 4px solid #f6b93b;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .hint-card-2 {
        background: #fff3e0;
        border-left: 4px solid #e67e22;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .hint-card-3 {
        background: #fce4ec;
        border-left: 4px solid #e74c3c;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    /* ── Page Header ── */
    .page-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 28px 36px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px rgba(102,126,234,0.25);
    }
    .page-header h1 { margin: 0; font-size: 1.8rem; }
    .page-header p { margin: 6px 0 0; opacity: 0.9; font-size: 1rem; }

    /* ── Question Box ── */
    .question-box {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        border-radius: 16px;
        padding: 24px 28px;
        color: white;
        font-size: 1.1rem;
        margin: 16px 0;
        box-shadow: 0 4px 20px rgba(52,152,219,0.25);
    }

    /* ── Passage Box ── */
    .passage-box {
        background: #f0f4ff;
        border: 1px solid #d0d8f0;
        border-radius: 12px;
        padding: 20px 24px;
        font-size: 0.95rem;
        line-height: 1.7;
        max-height: 350px;
        overflow-y: auto;
        margin-bottom: 16px;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }

    /* ── Status Badge ── */
    .status-badge {
        display: inline-block;
        background: #27ae60;
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-badge-warn {
        display: inline-block;
        background: #e67e22;
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Engine (cached) ─────────────────────────────────────
@st.cache_resource
def load_engine():
    from src.inference import InferenceEngine
    return InferenceEngine()

engine = load_engine()

# ── Session State Init ────────────────────────────────────────
if 'current_sample' not in st.session_state:
    st.session_state.current_sample = None
if 'user_answer' not in st.session_state:
    st.session_state.user_answer = None
if 'answer_checked' not in st.session_state:
    st.session_state.answer_checked = False
if 'hints_revealed' not in st.session_state:
    st.session_state.hints_revealed = 0
if 'session_results' not in st.session_state:
    st.session_state.session_results = []
if 'latencies' not in st.session_state:
    st.session_state.latencies = []
if 'show_answer' not in st.session_state:
    st.session_state.show_answer = False

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 4px;">
        <div style="font-size: 3rem;">📚</div>
        <h2 style="margin:4px 0 0; font-size:1.3rem; letter-spacing:0.5px;">RC Quiz System</h2>
        <p style="font-size:0.78rem; opacity:0.7; margin-top:2px;">Intelligent Reading Comprehension<br>& Quiz Generation</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio(
        'Navigate',
        ['📝 Article Input', '❓ Quiz View', '💡 Hints', '📊 Dashboard'],
        index=0
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    if engine.models_loaded:
        st.markdown('<div style="text-align:center;"><span class="status-badge">✓ Models Loaded</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center;"><span class="status-badge-warn">✗ Models Not Loaded</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding-top:24px; opacity:0.5; font-size:0.75rem;">
        AI Lab Project — Spring 2026<br>
        FAST NUCES Islamabad
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SCREEN 1 — Article Input
# ══════════════════════════════════════════════════════════════

if page == '📝 Article Input':
    st.markdown("""
    <div class="page-header">
        <h1>📝 Article Input</h1>
        <p>Paste a reading passage or load a random sample from the RACE dataset to get started.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col1:
        article_text = st.text_area(
            'Reading Passage',
            value=st.session_state.current_sample['article'] if st.session_state.current_sample else '',
            height=300,
            placeholder='Paste your reading passage here...'
        )

    with col2:
        st.markdown('#### Quick Actions')
        st.markdown("")
        if st.button('🎲 Load Random Sample', use_container_width=True, type='primary'):
            sample = engine.get_random_sample()
            if sample:
                st.session_state.current_sample = sample
                st.session_state.answer_checked = False
                st.session_state.hints_revealed = 0
                st.session_state.show_answer = False
                st.rerun()
            else:
                st.error('Could not load sample. Check dataset files.')

        if st.session_state.current_sample:
            st.markdown(f"""
            <div class="info-card">
                <strong>Sample Loaded</strong> ✓<br>
                <span style="font-size:0.85rem; opacity:0.7;">Ready to generate quiz</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    if st.button('🚀 Submit & Generate Quiz', use_container_width=True, type='primary'):
        if not article_text.strip():
            st.error('Please enter a reading passage or load a random sample.')
        else:
            with st.spinner('Running Model A & Model B inference...'):
                if st.session_state.current_sample and article_text == st.session_state.current_sample['article']:
                    sample = st.session_state.current_sample
                else:
                    result = engine.full_inference(article_text)
                    sample = {
                        'article': article_text,
                        'question': result['question'],
                        'options': result.get('generated_options', result['options']),
                        'correct_answer': result['predicted_answer'],
                        'correct_text': result['options'].get(result['predicted_answer'], ''),
                    }

                st.session_state.current_sample = sample
                st.session_state.answer_checked = False
                st.session_state.hints_revealed = 0
                st.session_state.show_answer = False
                st.markdown('<div class="success-card"><strong>Quiz generated!</strong> Navigate to <strong>Quiz View</strong> in the sidebar.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SCREEN 2 — Quiz View
# ══════════════════════════════════════════════════════════════

elif page == '❓ Quiz View':
    st.markdown("""
    <div class="page-header">
        <h1>❓ Question & Answer Quiz</h1>
        <p>Read the passage, pick your answer, and see how the ML model evaluates your choice.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.current_sample is None:
        st.warning('No article loaded. Go to **📝 Article Input** first and load a sample.')
    else:
        sample = st.session_state.current_sample

        with st.expander('📖 Reading Passage (click to expand)', expanded=False):
            st.markdown(f'<div class="passage-box">{sample["article"]}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="question-box"><strong>Question:</strong> {sample["question"]}</div>', unsafe_allow_html=True)
        st.markdown("")

        options = sample['options']
        option_labels = list(options.keys())

        selected = st.radio(
            'Choose your answer:',
            option_labels,
            format_func=lambda x: f"**{x}:** {options[x]}",
            index=None
        )

        col1, col2 = st.columns(2)

        with col1:
            check_clicked = st.button('✅ Check Answer', use_container_width=True, type='primary',
                                       disabled=selected is None)

        with col2:
            if st.button('🔄 New Question', use_container_width=True):
                new_sample = engine.get_random_sample()
                if new_sample:
                    st.session_state.current_sample = new_sample
                    st.session_state.answer_checked = False
                    st.session_state.hints_revealed = 0
                    st.session_state.show_answer = False
                    st.rerun()

        if check_clicked and selected:
            st.session_state.user_answer = selected
            st.session_state.answer_checked = True

            start = time.time()
            prob, feat, latency = engine.verify_answer(
                sample['article'], sample['question'], options[selected]
            )
            total_lat = time.time() - start

            is_correct = (selected == sample['correct_answer'])

            st.session_state.session_results.append({
                'question': sample['question'][:50],
                'selected': selected,
                'correct': sample['correct_answer'],
                'is_correct': is_correct,
                'confidence': prob,
                'latency': total_lat
            })
            st.session_state.latencies.append(total_lat)

        if st.session_state.answer_checked and st.session_state.user_answer:
            st.markdown("")
            is_correct = (st.session_state.user_answer == sample['correct_answer'])

            if is_correct:
                st.markdown(f"""
                <div class="success-card">
                    <strong>🎉 Correct!</strong><br>
                    The answer is <strong>{sample["correct_answer"]}</strong>: {options[sample["correct_answer"]]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-card">
                    <strong>❌ Incorrect.</strong><br>
                    You selected <strong>{st.session_state.user_answer}</strong>, but the correct answer is
                    <strong>{sample["correct_answer"]}</strong>: {options[sample["correct_answer"]]}
                </div>
                """, unsafe_allow_html=True)

            with st.expander('🔍 Model A — Option Confidence Scores'):
                ranked, rank_lat = engine.rank_all_options(
                    sample['article'], sample['question'], options
                )
                conf_data = []
                for label, text, prob in ranked:
                    marker = ' ✅' if label == sample['correct_answer'] else ''
                    conf_data.append({
                        'Option': f'{label}{marker}',
                        'Text': text[:60],
                        'Confidence': f'{prob:.4f}'
                    })
                st.table(pd.DataFrame(conf_data))


# ══════════════════════════════════════════════════════════════
# SCREEN 3 — Hint Panel
# ══════════════════════════════════════════════════════════════

elif page == '💡 Hints':
    st.markdown("""
    <div class="page-header">
        <h1>💡 Hint Panel</h1>
        <p>Reveal hints one at a time to help you find the correct answer. Hints get more specific as you go.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.current_sample is None:
        st.warning('No article loaded. Go to **📝 Article Input** first and load a sample.')
    else:
        sample = st.session_state.current_sample

        st.markdown(f'<div class="question-box"><strong>Question:</strong> {sample["question"]}</div>', unsafe_allow_html=True)
        st.markdown("")

        hints, h_lat = engine.generate_hints(sample['article'], sample['question'], n_hints=3)

        hint_labels = ['🟢 Hint 1 — General Clue', '🟡 Hint 2 — More Specific', '🔴 Hint 3 — Near-Explicit']
        hint_classes = ['hint-card', 'hint-card-2', 'hint-card-3']

        for i in range(3):
            if i < st.session_state.hints_revealed:
                hint_text = hints[i] if i < len(hints) else 'No additional hint available.'
                st.markdown(f'<div class="{hint_classes[i]}"><strong>{hint_labels[i]}</strong><br>{hint_text}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#f5f5f5; border-radius:12px; padding:16px 20px; margin-bottom:12px; text-align:center; color:#999;">
                    🔒 <strong>{hint_labels[i]}</strong> — Click below to reveal
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")
        col1, col2 = st.columns(2)

        with col1:
            if st.session_state.hints_revealed < 3:
                if st.button(f'🔓 Reveal Hint {st.session_state.hints_revealed + 1}',
                             use_container_width=True, type='primary'):
                    st.session_state.hints_revealed += 1
                    st.rerun()
            else:
                st.markdown('<div class="success-card" style="text-align:center;"><strong>All hints revealed!</strong></div>', unsafe_allow_html=True)

        with col2:
            if st.session_state.hints_revealed >= 3:
                if st.button('👁️ Reveal Answer', use_container_width=True, type='secondary'):
                    st.session_state.show_answer = True

        if st.session_state.show_answer:
            st.markdown("")
            correct = sample['correct_answer']
            st.markdown(f"""
            <div class="success-card" style="text-align:center; font-size:1.1rem;">
                🎯 <strong>The correct answer is {correct}:</strong> {sample["options"][correct]}
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SCREEN 4 — Developer / Analytics Dashboard
# ══════════════════════════════════════════════════════════════

elif page == '📊 Dashboard':
    st.markdown("""
    <div class="page-header">
        <h1>📊 Developer / Analytics Dashboard</h1>
        <p>View model performance metrics, session analytics, latency tracking, and export data.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        '🤖 Model A Performance', '🧩 Model B Performance',
        '📈 Session Analytics', '💾 Export Data'
    ])

    # ── Tab 1: Model A ──
    with tab1:
        st.subheader('Model A — Answer Verification Performance')

        try:
            comparison = pd.read_csv(os.path.join(
                os.path.dirname(__file__), '..', 'data', 'processed', 'model_a_comparison.csv'
            ))

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(comparison, x='name', y=['accuracy', 'f1', 'precision', 'recall'],
                             title='Model A — Metric Comparison',
                             barmode='group',
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(comparison, use_container_width=True)

            # Metrics summary
            best = comparison.loc[comparison['f1'].idxmax()]
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric('Best Model', best['name'][:20])
            mcol2.metric('Best Accuracy', f"{best['accuracy']:.4f}")
            mcol3.metric('Best F1', f"{best['f1']:.4f}")
            mcol4.metric('Train Time', f"{best.get('train_time', 0):.2f}s")

        except FileNotFoundError:
            st.warning('No Model A comparison data found. Run model_a_train.py first.')

    # ── Tab 2: Model B ──
    with tab2:
        st.subheader('Model B — Distractor & Hint Performance')

        st.markdown('#### Distractor Generation')
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="metric-card">
                <h2>100%</h2>
                <p>Distractor Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h2>0.558</h2>
                <p>Ranker F1 Score</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h2>0.534</h2>
                <p>Ranker Precision</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('#### Hint Generation')
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color:#1a1a2e;">
                <h2>57.7%</h2>
                <p>Hint Precision@K</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color:#1a1a2e;">
                <h2>99.9%</h2>
                <p>Hint Scorer Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%); color:#1a1a2e;">
                <h2>-10.41</h2>
                <p>R² Score</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 3: Session Analytics ──
    with tab3:
        st.subheader('Session Analytics')

        if st.session_state.session_results:
            results_df = pd.DataFrame(st.session_state.session_results)

            col1, col2, col3 = st.columns(3)
            total = len(results_df)
            correct = results_df['is_correct'].sum()

            col1.metric('Questions Answered', total)
            col2.metric('Correct Answers', f'{correct}/{total}')
            col3.metric('Accuracy', f'{correct/total*100:.1f}%' if total > 0 else 'N/A')

            if st.session_state.latencies:
                avg_lat = np.mean(st.session_state.latencies)
                st.metric('Avg Inference Latency', f'{avg_lat:.3f}s')

                fig = px.line(y=st.session_state.latencies,
                              title='Inference Latency per Request',
                              labels={'x': 'Request #', 'y': 'Latency (s)'})
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(results_df, use_container_width=True)
        else:
            st.info('No quiz attempts yet. Go to **Quiz View** to start answering questions!')

    # ── Tab 4: Export ──
    with tab4:
        st.subheader('Export Session Data')

        if st.session_state.session_results:
            results_df = pd.DataFrame(st.session_state.session_results)
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                '📥 Download Session Results as CSV',
                csv,
                'session_results.csv',
                'text/csv',
                use_container_width=True
            )
        else:
            st.info('No data to export yet.')

        st.divider()

        if st.button('🗑️ Clear Session Data', use_container_width=True):
            st.session_state.session_results = []
            st.session_state.latencies = []
            st.rerun()

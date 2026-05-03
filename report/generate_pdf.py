"""
generate_pdf.py — Generates the final report as a styled HTML file, then opens it for PDF conversion.
"""

import os
import webbrowser

REPORT_DIR = os.path.dirname(__file__)
HTML_PATH = os.path.join(REPORT_DIR, 'final_report.html')


def generate_html():
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Final Report — Intelligent Reading Comprehension System</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700;800&display=swap');

  :root {
    --primary: #667eea;
    --primary-dark: #5a67d8;
    --accent: #764ba2;
    --dark: #1a1a2e;
    --text: #2d3748;
    --light: #f7fafc;
    --border: #e2e8f0;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Inter', sans-serif;
    color: var(--text);
    line-height: 1.8;
    font-size: 11pt;
  }

  @page { margin: 2cm; size: A4; }

  /* ── COVER PAGE ── */
  .cover-page {
    width: 100%;
    height: 100vh;
    min-height: 1100px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 70%, #533483 100%);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    color: white;
    position: relative;
    overflow: hidden;
    page-break-after: always;
  }

  .cover-page::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 70%, rgba(102,126,234,0.15) 0%, transparent 50%),
                radial-gradient(circle at 70% 30%, rgba(118,75,162,0.12) 0%, transparent 50%);
    animation: none;
  }

  .cover-logo {
    font-size: 5rem;
    margin-bottom: 16px;
    z-index: 1;
  }

  .cover-university {
    font-size: 1rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 40px;
    z-index: 1;
  }

  .cover-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.2;
    max-width: 700px;
    z-index: 1;
    margin-bottom: 12px;
  }

  .cover-subtitle {
    font-size: 1.15rem;
    font-weight: 300;
    opacity: 0.85;
    max-width: 500px;
    z-index: 1;
    margin-bottom: 60px;
  }

  .cover-divider {
    width: 80px;
    height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    border-radius: 2px;
    margin: 24px auto;
    z-index: 1;
  }

  .cover-info-box {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 32px 48px;
    z-index: 1;
    min-width: 420px;
  }

  .cover-info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
  }
  .cover-info-row:last-child { border-bottom: none; }

  .cover-info-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    opacity: 0.6;
  }
  .cover-info-value {
    font-weight: 600;
    font-size: 0.95rem;
  }

  .cover-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    padding: 6px 20px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1px;
    margin-top: 24px;
    z-index: 1;
  }

  .cover-footer {
    position: absolute;
    bottom: 40px;
    font-size: 0.75rem;
    opacity: 0.4;
    z-index: 1;
  }

  /* ── CONTENT PAGES ── */
  .content { padding: 0 20px; max-width: 800px; margin: 0 auto; }

  h1 {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: var(--dark);
    margin: 40px 0 16px;
    padding-bottom: 8px;
    border-bottom: 3px solid var(--primary);
    page-break-after: avoid;
  }

  h2 {
    font-size: 1.2rem;
    color: var(--primary-dark);
    margin: 28px 0 12px;
    page-break-after: avoid;
  }

  h3 {
    font-size: 1rem;
    color: var(--accent);
    margin: 20px 0 8px;
  }

  p { margin: 8px 0 12px; text-align: justify; }

  ul, ol { margin: 8px 0 12px 24px; }
  li { margin-bottom: 4px; }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0 24px;
    font-size: 0.9rem;
  }
  th {
    background: linear-gradient(135deg, var(--primary), var(--accent));
    color: white;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
  }
  td {
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
  }
  tr:nth-child(even) { background: #f8f9ff; }
  tr:hover { background: #eef1ff; }

  code {
    background: #edf2f7;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85rem;
    color: #e53e3e;
  }

  .highlight-box {
    background: linear-gradient(135deg, #f0f4ff, #e8ecff);
    border-left: 4px solid var(--primary);
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 16px 0;
  }

  .page-break { page-break-before: always; }

  .toc { margin: 24px 0; }
  .toc a { color: var(--primary-dark); text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
  .toc li { margin: 6px 0; }
</style>
</head>
<body>

<!-- ════════════════════════════════════════════ -->
<!-- COVER PAGE -->
<!-- ════════════════════════════════════════════ -->
<div class="cover-page">
  <div class="cover-logo">📚</div>
  <div class="cover-university">National University of Computer & Emerging Sciences</div>
  <div class="cover-university" style="margin-bottom:20px; opacity:0.5; letter-spacing:3px; font-size:0.85rem;">FAST School of Computing — Islamabad Campus</div>

  <div class="cover-divider"></div>

  <div class="cover-title">Intelligent Reading Comprehension &amp; Quiz Generation System</div>
  <div class="cover-subtitle">Using Machine Learning &amp; Neural Networks on the RACE Dataset</div>

  <div class="cover-divider"></div>

  <div class="cover-info-box">
    <div class="cover-info-row">
      <span class="cover-info-label">Course</span>
      <span class="cover-info-value">Artificial Intelligence Lab (AL2002)</span>
    </div>
    <div class="cover-info-row">
      <span class="cover-info-label">Section</span>
      <span class="cover-info-value">CS-C</span>
    </div>
    <div class="cover-info-row">
      <span class="cover-info-label">Semester</span>
      <span class="cover-info-value">Spring 2026</span>
    </div>
    <div class="cover-info-row">
      <span class="cover-info-label">Team Members</span>
      <span class="cover-info-value">
        Muhammad Waqas — 23i-0569<br>
        Komail Raza — 23i-0717
      </span>
    </div>
    <div class="cover-info-row">
      <span class="cover-info-label">Submission Date</span>
      <span class="cover-info-value">May 11, 2026</span>
    </div>
  </div>

  <div class="cover-badge">AI LAB PROJECT — SPRING 2026</div>

  <div class="cover-footer">FAST NUCES Islamabad &bull; Department of Computer Science &bull; 2026</div>
</div>

<!-- ════════════════════════════════════════════ -->
<!-- TABLE OF CONTENTS -->
<!-- ════════════════════════════════════════════ -->
<div class="content">
<h1>Table of Contents</h1>
<ol class="toc" style="font-size:1rem; line-height:2;">
  <li><a href="#abstract">Abstract</a></li>
  <li><a href="#introduction">Introduction &amp; Motivation</a></li>
  <li><a href="#related">Related Work</a></li>
  <li><a href="#dataset">Dataset Analysis</a></li>
  <li><a href="#modela">Model A: Design, Training, Results</a></li>
  <li><a href="#modelb">Model B: Design, Training, Results</a></li>
  <li><a href="#ui">User Interface Description</a></li>
  <li><a href="#evaluation">Evaluation &amp; Discussion</a></li>
  <li><a href="#limitations">Limitations &amp; Future Work</a></li>
  <li><a href="#conclusion">Conclusion</a></li>
  <li><a href="#references">References</a></li>
</ol>

<!-- ════════════════════════════════════════════ -->
<!-- 1. ABSTRACT -->
<!-- ════════════════════════════════════════════ -->
<div class="page-break"></div>
<h1 id="abstract">1. Abstract</h1>
<div class="highlight-box">
<p>We present an intelligent reading comprehension and quiz generation system built on the RACE (ReAding Comprehension from Examinations) dataset. The system combines classical machine learning models for answer verification, question generation, distractor creation, and hint generation into an interactive Streamlit-based web application. Model A handles answer verification using Logistic Regression, SVM, Naive Bayes, Random Forest, XGBoost, and ensemble methods, alongside unsupervised approaches (K-Means, GMM) and semi-supervised learning (Label Propagation). Model B generates plausible distractor options using frequency-based candidate extraction with Random Forest ranking and produces graduated hints via sentence-level cosine similarity scoring. The system processes approximately 87,000 training samples from Chinese middle and high school English exams. Our best supervised model achieves a Macro F1 of 0.45 on the binary verification task, while the Random Forest difficulty estimator achieves 83% accuracy. The distractor generator maintains 100% accuracy in excluding the correct answer, and the hint system achieves 57.7% Precision@K. The complete system is deployed as a four-screen Streamlit application featuring article input, interactive quiz view with color-coded feedback, graduated hint panels, and a developer analytics dashboard.</p>
</div>

<!-- ════════════════════════════════════════════ -->
<!-- 2. INTRODUCTION -->
<!-- ════════════════════════════════════════════ -->
<h1 id="introduction">2. Introduction &amp; Motivation</h1>
<p>Reading comprehension is a fundamental skill in education, and automated assessment tools can significantly reduce instructor workload while providing students with immediate, interactive feedback. Traditional assessment systems rely on manually crafted questions and answer keys, which are time-consuming to produce and difficult to scale.</p>

<p>The goal of this project is to build an end-to-end system that:</p>
<ol>
  <li><strong>Verifies answers</strong> to reading comprehension questions using classical ML classifiers.</li>
  <li><strong>Generates questions</strong> from reading passages using template-based approaches ranked by ML models.</li>
  <li><strong>Creates plausible distractors</strong> (wrong answer options) that are semantically related but incorrect.</li>
  <li><strong>Provides graduated hints</strong> that progressively reveal information to guide learners.</li>
  <li><strong>Presents everything</strong> in an accessible, interactive Streamlit web interface.</li>
</ol>

<p>The RACE dataset provides an ideal testbed: it contains ~100,000 multiple-choice questions across two difficulty levels (middle and high school), generated by human English teachers, ensuring high quality and diversity.</p>

<h2>Why Classical ML?</h2>
<p>While large language models dominate NLP leaderboards, classical ML approaches remain valuable for interpretability, efficiency (no GPU needed), and educational value in understanding foundational ML concepts like TF-IDF, cosine similarity, and ensemble methods.</p>

<!-- ════════════════════════════════════════════ -->
<!-- 3. RELATED WORK -->
<!-- ════════════════════════════════════════════ -->
<div class="page-break"></div>
<h1 id="related">3. Related Work</h1>
<p><strong>1. Lai et al. (2017)</strong> — <em>"RACE: Large-scale ReAding Comprehension Dataset From Examinations."</em> EMNLP 2017. Introduced the RACE dataset with ~28,000 passages and ~100,000 questions.</p>
<p><strong>2. Rajpurkar et al. (2016)</strong> — <em>"SQuAD: 100,000+ Questions for Machine Comprehension of Text."</em> EMNLP 2016. Introduced the extractive QA paradigm with feature engineering insights.</p>
<p><strong>3. Welbl et al. (2017)</strong> — <em>"Crowdsourcing Multiple Choice Science Questions."</em> NeurIPS Workshop. Explored automated distractor generation using knowledge bases and word embeddings.</p>
<p><strong>4. Gao et al. (2019)</strong> — <em>"Generating Distractors for Reading Comprehension Questions from Real Examinations."</em> AAAI 2019. Proposed neural distractor generation using candidate-extraction-and-ranking paradigm.</p>
<p><strong>5. Mitkov et al. (2006)</strong> — <em>"A Computer-Aided Environment for Generating Multiple-Choice Test Items."</em> Natural Language Engineering. Pioneered template-based question generation with NLP.</p>
<p><strong>6. Pan et al. (2020)</strong> — <em>"Unsupervised Multi-hop Question Answering by Question Generation."</em> NeurIPS 2020. Demonstrated joint optimization of question generation and answering.</p>

<!-- ════════════════════════════════════════════ -->
<!-- 4. DATASET ANALYSIS -->
<!-- ════════════════════════════════════════════ -->
<h1 id="dataset">4. Dataset Analysis</h1>
<h2>4.1 Overview</h2>
<table>
  <tr><th>Split</th><th>Samples</th><th>Unique Articles</th></tr>
  <tr><td>Train</td><td>87,866</td><td>~25,000</td></tr>
  <tr><td>Validation</td><td>4,887</td><td>~1,400</td></tr>
  <tr><td>Test</td><td>4,934</td><td>~1,400</td></tr>
</table>

<h2>4.2 Answer Distribution</h2>
<p>The answer labels show mild imbalance: A (21.8%), B (25.9%), C (27.2%), D (25.2%). Class balance ratio: 1.25.</p>

<h2>4.3 Key Findings</h2>
<ul>
  <li>Average article length: ~250 words</li>
  <li>High school articles are significantly longer than middle school</li>
  <li>Fill-in-the-blank is the most common question type</li>
  <li>High school comprises ~70% of the dataset</li>
  <li>Option lengths are balanced across correct and incorrect answers</li>
</ul>

<!-- ════════════════════════════════════════════ -->
<!-- 5. MODEL A -->
<!-- ════════════════════════════════════════════ -->
<div class="page-break"></div>
<h1 id="modela">5. Model A: Design, Training, Results</h1>

<h2>5.1 Feature Engineering</h2>
<p>Each sample is expanded into 4 binary instances (one per option). Features include TF-IDF and One-Hot cosine similarities between article/question/option pairs, plus handcrafted lexical features (word counts, overlap ratios).</p>

<table>
  <tr><th>Feature</th><th>Description</th></tr>
  <tr><td><code>tfidf_cos_art_opt</code></td><td>TF-IDF cosine similarity: article ↔ option</td></tr>
  <tr><td><code>tfidf_cos_q_opt</code></td><td>TF-IDF cosine similarity: question ↔ option</td></tr>
  <tr><td><code>oh_cos_art_opt</code></td><td>One-Hot cosine similarity: article ↔ option</td></tr>
  <tr><td><code>overlap_art_opt</code></td><td>Word overlap ratio: article ↔ option</td></tr>
  <tr><td><code>overlap_q_opt</code></td><td>Word overlap ratio: question ↔ option</td></tr>
  <tr><td><code>article_word_count</code></td><td>Number of words in article</td></tr>
</table>

<h2>5.2 Supervised Models</h2>
<table>
  <tr><th>Model</th><th>Accuracy</th><th>Macro F1</th><th>Precision</th><th>Recall</th><th>Time</th></tr>
  <tr><td>Logistic Regression</td><td>0.750</td><td>0.429</td><td>0.375</td><td>0.500</td><td>0.01s</td></tr>
  <tr><td>SVM (LinearSVC)</td><td>0.750</td><td>0.429</td><td>0.375</td><td>0.500</td><td>0.02s</td></tr>
  <tr><td>XGBoost</td><td>0.745</td><td>0.445</td><td>0.538</td><td>0.503</td><td>0.23s</td></tr>
  <tr><td>Naive Bayes</td><td>0.647</td><td>0.329</td><td>0.404</td><td>0.317</td><td>0.01s</td></tr>
  <tr><td><strong>Random Forest (Difficulty)</strong></td><td><strong>0.830</strong></td><td><strong>0.787</strong></td><td><strong>0.800</strong></td><td><strong>0.777</strong></td><td>0.83s</td></tr>
</table>

<h2>5.3 Unsupervised &amp; Semi-Supervised</h2>
<p><strong>K-Means</strong> and <strong>GMM</strong> clustering on question-answer features. <strong>Label Propagation</strong> with 10% labeled data achieved Macro F1 of 0.463.</p>

<h2>5.4 Ensemble Methods</h2>
<p>Soft Voting (LR + RF + XGBoost) and Stacking ensembles were implemented and evaluated.</p>

<!-- ════════════════════════════════════════════ -->
<!-- 6. MODEL B -->
<!-- ════════════════════════════════════════════ -->
<div class="page-break"></div>
<h1 id="modelb">6. Model B: Design, Training, Results</h1>

<h2>6.1 Distractor Generation</h2>
<p>Pipeline: (1) Extract candidate phrases from article, (2) Compute features (character match, length ratio, passage frequency, cosine similarity), (3) Rank with Random Forest, (4) Apply diversity penalty.</p>

<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Distractor Ranker F1</td><td>0.558</td></tr>
  <tr><td><strong>Distractor Accuracy</strong></td><td><strong>100%</strong></td></tr>
  <tr><td>Top Feature: len_ratio</td><td>0.464 importance</td></tr>
</table>

<h2>6.2 Hint Generation</h2>
<p>Sentences scored by keyword overlap (40%), answer overlap (40%), and TF-IDF cosine similarity (20%). Three graduated hints: general → specific → near-explicit.</p>

<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Hint Scorer Accuracy</td><td>99.9%</td></tr>
  <tr><td>Hint Precision@K</td><td>57.7%</td></tr>
</table>

<!-- ════════════════════════════════════════════ -->
<!-- 7. USER INTERFACE -->
<!-- ════════════════════════════════════════════ -->
<h1 id="ui">7. User Interface Description</h1>
<p>The Streamlit application consists of four screens:</p>

<table>
  <tr><th>Screen</th><th>Description</th></tr>
  <tr><td><strong>1. Article Input</strong></td><td>Text area for custom passages, "Load Random Sample" button, Submit trigger</td></tr>
  <tr><td><strong>2. Quiz View</strong></td><td>Question display, 4 radio options, Check Answer with green/red feedback, confidence scores</td></tr>
  <tr><td><strong>3. Hint Panel</strong></td><td>Three graduated hints revealed one at a time, Reveal Answer after all hints</td></tr>
  <tr><td><strong>4. Dashboard</strong></td><td>Model metrics, bar charts, session analytics, latency tracking, CSV export</td></tr>
</table>

<p>The UI features a dark gradient sidebar, styled gradient cards, loading spinners, error handling, and accessible color contrast ratios.</p>

<!-- ════════════════════════════════════════════ -->
<!-- 8. EVALUATION & DISCUSSION -->
<!-- ════════════════════════════════════════════ -->
<div class="page-break"></div>
<h1 id="evaluation">8. Evaluation &amp; Discussion</h1>

<h2>Key Findings</h2>
<ol>
  <li><strong>Answer verification is hard</strong> — the 75% majority baseline is difficult to beat with lexical features alone. XGBoost shows the most promise (F1: 0.445).</li>
  <li><strong>Difficulty estimation works well</strong> — article length and sentence count are strong signals (83% accuracy).</li>
  <li><strong>Distractor generation is effective</strong> — 100% accuracy in excluding correct answers with diversity filtering.</li>
  <li><strong>Hints are moderately useful</strong> — 57.7% contain answer keywords.</li>
</ol>

<h2>Ethical Considerations</h2>
<ul>
  <li><strong>Bias:</strong> RACE passages come from Chinese school exams; cultural and linguistic bias may limit generalization.</li>
  <li><strong>Accessibility:</strong> UI designed with sufficient color contrast and readable fonts.</li>
  <li><strong>Academic integrity:</strong> Generated questions should not be used in real exams without human review.</li>
  <li><strong>Transparency:</strong> UI clearly indicates AI-generated content.</li>
</ul>

<!-- ════════════════════════════════════════════ -->
<!-- 9. LIMITATIONS & FUTURE WORK -->
<!-- ════════════════════════════════════════════ -->
<h1 id="limitations">9. Limitations &amp; Future Work</h1>

<h2>Limitations</h2>
<ul>
  <li>Lexical features insufficient for complex reasoning questions requiring inference.</li>
  <li>Template-based question generation produces generic questions.</li>
  <li>Binary verification framing doesn't capture inter-option relationships.</li>
  <li>Models trained on 5K-row subset for speed; full-data training would improve scores.</li>
</ul>

<h2>Future Work</h2>
<ul>
  <li>Transformer-based features (BERT/RoBERTa embeddings) for classical ML models.</li>
  <li>Neural question generation (T5, GPT-2 fine-tuning).</li>
  <li>Multi-class formulation (4-class direct prediction).</li>
  <li>Active learning with Label Propagation.</li>
  <li>Adaptive difficulty based on user performance.</li>
</ul>

<!-- ════════════════════════════════════════════ -->
<!-- 10. CONCLUSION -->
<!-- ════════════════════════════════════════════ -->
<h1 id="conclusion">10. Conclusion</h1>
<p>We built a complete reading comprehension quiz system combining classical ML for answer verification, distractor generation, and hint scoring with an interactive Streamlit UI. The system demonstrates that while classical approaches have limitations on complex reasoning tasks, they provide fast, interpretable, and resource-efficient solutions suitable for educational applications. The modular architecture supports easy extension with more advanced models in the future.</p>

<!-- ════════════════════════════════════════════ -->
<!-- 11. REFERENCES -->
<!-- ════════════════════════════════════════════ -->
<h1 id="references">11. References</h1>
<ol>
  <li>Lai, G., Xie, Q., Liu, H., Yang, Y., &amp; Hovy, E. (2017). <em>RACE: Large-scale ReAding Comprehension Dataset From Examinations.</em> EMNLP 2017.</li>
  <li>Rajpurkar, P., Zhang, J., Lopyrev, K., &amp; Liang, P. (2016). <em>SQuAD: 100,000+ Questions for Machine Comprehension of Text.</em> EMNLP 2016.</li>
  <li>Welbl, J., Liu, N. F., &amp; Gardner, M. (2017). <em>Crowdsourcing Multiple Choice Science Questions.</em> NeurIPS Workshop.</li>
  <li>Gao, Y., Bing, L., Chen, W., Lyu, M. R., &amp; King, I. (2019). <em>Generating Distractors for Reading Comprehension Questions from Real Examinations.</em> AAAI 2019.</li>
  <li>Mitkov, R., Ha, L. A., &amp; Karamanis, N. (2006). <em>A Computer-Aided Environment for Generating Multiple-Choice Test Items.</em> Natural Language Engineering.</li>
  <li>Pan, L., Chen, W., Xiong, W., Kan, M. Y., &amp; Wang, W. Y. (2020). <em>Unsupervised Multi-hop Question Answering by Question Generation.</em> NeurIPS 2020.</li>
</ol>

</div><!-- /content -->
</body>
</html>"""

    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"HTML report generated: {HTML_PATH}")
    return HTML_PATH


def convert_to_pdf():
    html_path = generate_html()
    pdf_path = os.path.join(REPORT_DIR, 'final_report.pdf')

    # Try weasyprint first
    try:
        from weasyprint import HTML
        HTML(filename=html_path).write_pdf(pdf_path)
        print(f"PDF generated with weasyprint: {pdf_path}")
        return pdf_path
    except ImportError:
        pass

    # Try pdfkit
    try:
        import pdfkit
        pdfkit.from_file(html_path, pdf_path)
        print(f"PDF generated with pdfkit: {pdf_path}")
        return pdf_path
    except (ImportError, Exception):
        pass

    # Fallback: open in browser for manual print
    print("No PDF library found. Opening HTML in browser — use Ctrl+P to save as PDF.")
    webbrowser.open(f'file:///{os.path.abspath(html_path)}')
    return html_path


if __name__ == '__main__':
    convert_to_pdf()

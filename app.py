import streamlit as st
import re
import math
import requests
import json
import time
from datetime import datetime

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="SpamGuard AI - 垃圾邮件卫士",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 自定义 CSS (模拟 React/Tailwind UI) ---
st.markdown("""
<style>
    /* 全局背景色 */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* 隐藏默认页眉 */
    header {visibility: hidden;}
    
    /* 自定义页眉 */
    .custom-header {
        background-color: white;
        border-bottom: 1px solid #e2e8f0;
        padding: 1rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        z-index: 999;
        margin-bottom: 2rem;
    }
    
    .header-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .logo-icon {
        background-color: #2563eb;
        padding: 0.5rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: bold;
    }
    
    .header-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }

    /* 卡片样式 */
    .card {
        background-color: white;
        border-radius: 1rem;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    /* 结果卡片 - 垃圾邮件 */
    .result-card-spam {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 1rem;
        padding: 1.5rem;
    }
    
    /* 结果卡片 - 正常邮件 */
    .result-card-ham {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 1rem;
        padding: 1.5rem;
    }

    /* 按钮样式 */
    .stButton>button {
        width: 100%;
        border-radius: 0.75rem;
        padding: 0.75rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background-color: white;
    }
    
    /* 历史记录项 */
    .history-item {
        background-color: #f1f5f9;
        padding: 0.75rem;
        border-radius: 0.75rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    
    .history-item:hover {
        background-color: white;
        border-color: #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>

<div class="custom-header">
    <div class="header-logo">
        <div class="logo-icon">🛡️</div>
        <h1 class="header-title">SpamGuard AI 垃圾邮件卫士</h1>
    </div>
    <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; background: #f1f5f9; padding: 0.25rem 0.5rem; border-radius: 0.25rem;">
        高级安全防护
    </div>
</div>
""", unsafe_allow_html=True)

# --- 3. 朴素贝叶斯分类器逻辑 ---
class NaiveBayesClassifier:
    def __init__(self, training_data):
        self.spam_counts = {}
        self.ham_counts = {}
        self.spam_total = 0
        self.ham_total = 0
        self.vocabulary = set()
        self.train(training_data)

    def tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def train(self, data):
        for item in data:
            tokens = self.tokenize(item['text'])
            target_counts = self.spam_counts if item['isSpam'] else self.ham_counts
            if item['isSpam']:
                self.spam_total += 1
            else:
                self.ham_total += 1
            for token in set(tokens):
                target_counts[token] = target_counts.get(token, 0) + 1
                self.vocabulary.add(token)

    def classify(self, text):
        tokens = self.tokenize(text)
        spam_prob = math.log(self.spam_total / (self.spam_total + self.ham_total))
        ham_prob = math.log(self.ham_total / (self.spam_total + self.ham_total))
        matched_features = []
        for word in self.vocabulary:
            p_w_spam = (self.spam_counts.get(word, 0) + 1) / (self.spam_total + 2)
            p_w_ham = (self.ham_counts.get(word, 0) + 1) / (self.ham_total + 2)
            if word in tokens:
                spam_prob += math.log(p_w_spam)
                ham_prob += math.log(p_w_ham)
                score = math.log(p_w_spam) - math.log(p_w_ham)
                if abs(score) > 0.5:
                    matched_features.append((word, score))
        is_spam = spam_prob > ham_prob
        confidence = 1 / (1 + math.exp(min(max(ham_prob - spam_prob, -10), 10)))
        if not is_spam: confidence = 1 - confidence
        return {
            "isSpam": is_spam,
            "confidence": confidence,
            "matchedFeatures": [f[0] for f in sorted(matched_features, key=lambda x: abs(x[1]), reverse=True)[:12]]
        }

TRAINING_DATA = [
    {"text": "恭喜你中奖了，点击领取奖金", "isSpam": True},
    {"text": "您的账户存在异常，请立即修改密码", "isSpam": True},
    {"text": "特价优惠，最后一天，不容错过", "isSpam": True},
    {"text": "明天下午三点开会，请准时参加", "isSpam": False},
    {"text": "关于下周项目的进度报告，请查收", "isSpam": False},
    {"text": "你好，好久不见，最近怎么样？", "isSpam": False},
]
classifier = NaiveBayesClassifier(TRAINING_DATA)

# --- 4. 状态管理 ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

# --- 5. 侧边栏 (历史记录) ---
with st.sidebar:
    st.markdown("### 🕒 历史记录")
    if not st.session_state.history:
        st.info("暂无分析历史")
    else:
        if st.button("清空全部"):
            st.session_state.history = []
            st.rerun()
        
        for i, item in enumerate(reversed(st.session_state.history)):
            color = "#ef4444" if item['isSpam'] else "#10b981"
            label = "垃圾邮件" if item['isSpam'] else "正常邮件"
            st.markdown(f"""
            <div class="history-item">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-size: 10px; font-weight: 700; color: white; background: {color}; padding: 2px 6px; border-radius: 4px;">{label}</span>
                    <span style="font-size: 10px; color: #94a3b8;">{item['time']}</span>
                </div>
                <div style="font-size: 12px; font-weight: 600; color: #1e293b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{item['subject'] or '(无主题)'}</div>
            </div>
            """, unsafe_allow_html=True)

# --- 6. 主内容区 ---
col_main, col_info = st.columns([2, 1])

with col_main:
    # 输入区域
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
        <span style="font-size: 1.25rem;">📧</span>
        <h2 style="font-size: 1.125rem; font-weight: 600; margin: 0;">分析电子邮件</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        subject = st.text_input("邮件主题", placeholder="例如：恭喜您获得100万大奖...")
        body = st.text_area("邮件内容", placeholder="在此粘贴完整的邮件正文...", height=250)
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            mode = st.radio("分析模式", ["AI 智能模式", "传统模式"], horizontal=True, label_visibility="collapsed")
        
        if st.button("开始分析", type="primary"):
            if not subject and not body:
                st.error("请输入邮件主题或内容。")
            else:
                with st.spinner("正在分析中..."):
                    if mode == "传统模式":
                        res = classifier.classify(f"{subject} {body}")
                        reasoning = f"### 传统模式分析结果\n基于朴素贝叶斯算法分析。关键特征词：{', '.join(res['matchedFeatures'])}"
                        st.session_state.current_result = {
                            "isSpam": res['isSpam'],
                            "confidence": res['confidence'],
                            "reasoning": reasoning,
                            "category": "传统过滤 (机器学习)",
                            "subject": subject,
                            "body": body
                        }
                    else:
                        api_key = st.secrets.get("KIMI_API_KEY")
                        if not api_key:
                            st.error("未配置 KIMI_API_KEY。请在 Streamlit Secrets 中设置。")
                        else:
                            try:
                                response = requests.post(
                                    "https://api.moonshot.cn/v1/chat/completions",
                                    headers={"Authorization": f"Bearer {api_key}"},
                                    json={
                                        "model": "moonshot-v1-8k",
                                        "messages": [
                                            {"role": "system", "content": "你是一个专业的垃圾邮件分析专家。请返回 JSON 格式结果。"},
                                            {"role": "user", "content": f"分析邮件：主题:{subject}, 正文:{body}。返回 isSpam(bool), confidence(float), reasoning(str), category(str)"}
                                        ],
                                        "temperature": 0.3
                                    }
                                )
                                data = response.json()
                                content = data['choices'][0]['message']['content']
                                json_str = re.search(r'\{.*\}', content, re.DOTALL).group()
                                ai_result = json.loads(json_str)
                                st.session_state.current_result = {
                                    **ai_result,
                                    "subject": subject,
                                    "body": body
                                }
                            except Exception as e:
                                st.error(f"AI 分析失败: {str(e)}")
                
                if st.session_state.current_result:
                    st.session_state.history.append({
                        "subject": subject,
                        "isSpam": st.session_state.current_result['isSpam'],
                        "time": datetime.now().strftime("%H:%M")
                    })
                    st.session_state.chat_messages = [] # 重置聊天

    # 结果展示
    if st.session_state.current_result:
        res = st.session_state.current_result
        card_class = "result-card-spam" if res['isSpam'] else "result-card-ham"
        title = "疑似垃圾邮件" if res['isSpam'] else "疑似合法邮件"
        icon = "🚨" if res['isSpam'] else "✅"
        color = "#991b1b" if res['isSpam'] else "#065f46"
        
        st.markdown(f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="font-size: 2rem;">{icon}</div>
                    <div>
                        <h3 style="margin: 0; color: {color}; font-size: 1.25rem; font-weight: 700;">{title}</h3>
                        <p style="margin: 0; color: {color}; font-size: 0.875rem; font-weight: 500;">类别：{res['category']}</p>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.5rem; font-weight: 900; color: {color};">{res['confidence']:.0%}</div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: {color}; opacity: 0.7; text-transform: uppercase;">置信度</div>
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.5); padding: 1rem; border-radius: 0.75rem; border: 1px solid rgba(0,0,0,0.05);">
                <h4 style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 0.5rem;">分析推理</h4>
                <div style="font-size: 0.875rem; color: #334155;">
        """, unsafe_allow_html=True)
        st.markdown(res['reasoning'])
        st.markdown("</div></div></div>", unsafe_allow_html=True)
        
        # AI 聊天
        st.markdown("---")
        st.markdown("### 💬 咨询 AI 专家")
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if prompt := st.chat_input("询问关于此邮件的任何问题..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    try:
                        api_key = st.secrets.get("KIMI_API_KEY")
                        context = f"主题: {res['subject']}\n正文: {res['body']}\n分析结果: {title} ({res['category']})"
                        response = requests.post(
                            "https://api.moonshot.cn/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}"},
                            json={
                                "model": "moonshot-v1-8k",
                                "messages": [
                                    {"role": "system", "content": f"你是一位网络安全专家。上下文：{context}"},
                                    *st.session_state.chat_messages
                                ],
                                "temperature": 0.7
                            }
                        )
                        ans = response.json()['choices'][0]['message']['content']
                        st.write(ans)
                        st.session_state.chat_messages.append({"role": "assistant", "content": ans})
                    except:
                        st.error("聊天服务暂时不可用。")

with col_info:
    st.markdown("""
    <div class="card" style="background-color: #2563eb; color: white; border: none;">
        <h3 style="color: white; font-size: 1.125rem; font-weight: 700; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
            <span>🛡️</span> 工作原理
        </h3>
        <div style="font-size: 0.875rem; line-height: 1.6; color: #dbeafe;">
            <p><strong>AI 智能模式</strong><br>使用先进的大语言模型分析语言模式、元数据线索和复杂的网络钓鱼策略。</p>
            <hr style="border-color: rgba(255,255,255,0.2); margin: 1rem 0;">
            <p><strong>传统模式 (机器学习)</strong><br>基于朴素贝叶斯算法，通过学习样本的词频分布进行快速匹配，不依赖外部服务。</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3 style="font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem;">📊 系统状态</h3>
        <p style="font-size: 0.875rem; color: #64748b;">
            • 引擎版本: v1.0.2<br>
            • AI 模型: Kimi-8k<br>
            • 运行环境: Streamlit Cloud
        </p>
    </div>
    """, unsafe_allow_html=True)

import streamlit as st
import re
import math
import requests
import json

# --- 1. 朴素贝叶斯分类器逻辑 (从 TS 移植) ---
class NaiveBayesClassifier:
    def __init__(self, training_data):
        self.spam_counts = {}
        self.ham_counts = {}
        self.spam_total = 0
        self.ham_total = 0
        self.vocabulary = set()
        self.train(training_data)

    def tokenize(self, text):
        # 简单的分词逻辑
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
            # 计算单词在垃圾邮件中的概率 (拉普拉斯平滑)
            p_w_spam = (self.spam_counts.get(word, 0) + 1) / (self.spam_total + 2)
            p_w_ham = (self.ham_counts.get(word, 0) + 1) / (self.ham_total + 2)
            
            if word in tokens:
                spam_prob += math.log(p_w_spam)
                ham_prob += math.log(p_w_ham)
                # 计算特征贡献度
                score = math.log(p_w_spam) - math.log(p_w_ham)
                if abs(score) > 0.5:
                    matched_features.append((word, score))

        is_spam = spam_prob > ham_prob
        # 简单的置信度计算
        confidence = 1 / (1 + math.exp(min(max(ham_prob - spam_prob, -10), 10)))
        if not is_spam:
            confidence = 1 - confidence
            
        return {
            "isSpam": is_spam,
            "confidence": confidence,
            "matchedFeatures": [f[0] for f in sorted(matched_features, key=lambda x: abs(x[1]), reverse=True)[:12]]
        }

# 默认训练数据
TRAINING_DATA = [
    {"text": "恭喜你中奖了，点击领取奖金", "isSpam": True},
    {"text": "您的账户存在异常，请立即修改密码", "isSpam": True},
    {"text": "特价优惠，最后一天，不容错过", "isSpam": True},
    {"text": "明天下午三点开会，请准时参加", "isSpam": False},
    {"text": "关于下周项目的进度报告，请查收", "isSpam": False},
    {"text": "你好，好久不见，最近怎么样？", "isSpam": False},
]

classifier = NaiveBayesClassifier(TRAINING_DATA)

# --- 2. Streamlit 界面设置 ---
st.set_page_config(page_title="SpamGuard AI - 垃圾邮件检测", page_icon="🛡️", layout="wide")

st.title("🛡️ SpamGuard AI 邮件检测系统")
st.markdown("使用机器学习和 AI 技术保护您的收件箱。")

col1, col2 = st.columns([2, 1])

with col1:
    subject = st.text_input("邮件主题", placeholder="请输入邮件主题...")
    body = st.text_area("邮件正文", placeholder="请输入邮件正文内容...", height=300)
    
    mode = st.radio("选择分析模式", ["传统模式 (朴素贝叶斯)", "AI 模式 (Kimi AI)"], horizontal=True)

    if st.button("开始分析", type="primary"):
        if not body:
            st.warning("请输入邮件内容后再进行分析。")
        else:
            full_text = f"{subject} {body}"
            
            if mode == "传统模式 (朴素贝叶斯)":
                with st.spinner("正在进行概率分析..."):
                    result = classifier.classify(full_text)
                    
                st.subheader("分析结果")
                if result['isSpam']:
                    st.error(f"🚨 疑似垃圾邮件 (置信度: {result['confidence']:.2%})")
                else:
                    st.success(f"✅ 疑似正常邮件 (置信度: {result['confidence']:.2%})")
                
                st.info(f"**关键特征词**: {', '.join(result['matchedFeatures'])}")
                st.markdown("---")
                st.write("注：传统模式基于本地词频统计，不消耗 API 额度。")
                
            else:
                # AI 模式
                api_key = st.secrets.get("KIMI_API_KEY") or st.sidebar.text_input("请输入 Kimi API Key", type="password")
                
                if not api_key:
                    st.error("请在侧边栏配置 API Key 或在 Streamlit Secrets 中设置。")
                else:
                    with st.spinner("AI 正在深度分析中..."):
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
                            # 提取 JSON
                            json_str = re.search(r'\{.*\}', content, re.DOTALL).group()
                            ai_result = json.loads(json_str)
                            
                            st.subheader("AI 深度分析结果")
                            if ai_result['isSpam']:
                                st.error(f"🚨 垃圾邮件 - 分类: {ai_result['category']}")
                            else:
                                st.success(f"✅ 正常邮件 - 分类: {ai_result['category']}")
                            
                            st.write(f"**置信度**: {ai_result['confidence']:.2%}")
                            st.markdown(f"**分析理由**:\n{ai_result['reasoning']}")
                            
                        except Exception as e:
                            st.error(f"AI 分析失败: {str(e)}")

with col2:
    st.subheader("📊 系统说明")
    st.write("""
    - **传统模式**: 使用朴素贝叶斯算法，在本地运行，速度极快。
    - **AI 模式**: 使用 Kimi 大模型进行语义理解，能够识别复杂的钓鱼和诈骗手段。
    """)
    
    st.subheader("📝 最近记录")
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    for h in st.session_state.history[-5:]:
        st.text(f"{h['time']} - {h['label']}")

    st.markdown("---")
    st.caption("SpamGuard AI v1.0 | 基于 Streamlit 构建")

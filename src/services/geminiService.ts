import { NaiveBayesClassifier, DEFAULT_TRAINING_DATA } from './spamClassifier';

// Initialize the classifier with default training data
const classifier = new NaiveBayesClassifier(DEFAULT_TRAINING_DATA);

export interface SpamAnalysisResult {
  isSpam: boolean;
  confidence: number;
  reasoning: string;
  category: string;
}

export async function analyzeEmail(subject: string, body: string): Promise<SpamAnalysisResult> {
  const apiKey = import.meta.env.VITE_KIMI_API_KEY;

  if (!apiKey) {
    throw new Error("未配置 VITE_KIMI_API_KEY。请在环境变量中设置。");
  }

  const prompt = `分析以下电子邮件并确定它是垃圾邮件还是合法邮件（非垃圾邮件）。请使用中文回答。
    
    主题: ${subject}
    正文: ${body}
    
    请以 JSON 格式提供分析结果，包含以下字段：
    - isSpam: 布尔值 (true 表示垃圾邮件)
    - confidence: 数字 (0 到 1)
    - reasoning: 字符串 (详细的 Markdown 格式解释，使用中文)
    - category: 字符串 (例如：网络钓鱼、推广、社交、交易、个人等，使用中文)`;

  const response = await fetch("https://api.moonshot.cn/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "moonshot-v1-8k",
      messages: [
        { role: "system", content: "你是一个专业的垃圾邮件分析专家。你必须始终返回有效的 JSON 格式数据。" },
        { role: "user", content: prompt }
      ],
      temperature: 0.3,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error?.message || "AI 分析请求失败");
  }

  const data = await response.json();
  const content = data.choices[0].message.content;
  
  // Extract JSON from content
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  const jsonStr = jsonMatch ? jsonMatch[0] : content;
  
  return JSON.parse(jsonStr);
}

export async function traditionalAnalyzeEmail(subject: string, body: string): Promise<SpamAnalysisResult> {
  const result = classifier.classify(subject, body);

  let reasoning = `### 传统模式分析结果 (朴素贝叶斯分类器)\n\n`;
  reasoning += `基于机器学习算法（朴素贝叶斯）对邮件文本进行概率分析：\n\n`;
  
  if (result.matchedFeatures.length > 0) {
    reasoning += `#### 关键特征词分析 (Top 12)\n`;
    reasoning += `这些词汇对最终分类结果影响最大（正数表示偏向垃圾邮件，负数表示偏向正常邮件）：\n\n`;
    result.matchedFeatures.forEach(feature => {
      reasoning += `- \`${feature}\`\n`;
    });
  }
  
  reasoning += `\n\n**分析结论**: 该模型通过学习大量垃圾邮件和正常邮件的词频分布，计算出当前邮件为 **${result.isSpam ? '垃圾邮件' : '正常邮件'}** 的概率。`;
  reasoning += `\n\n**置信度**: ${Math.round(result.confidence * 100)}%`;

  return {
    isSpam: result.isSpam,
    confidence: result.confidence,
    reasoning,
    category: result.isSpam ? '传统过滤 (机器学习 - 疑似垃圾)' : '传统过滤 (机器学习 - 疑似合法)'
  };
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export async function sendMessageToAI(messages: ChatMessage[], emailContext: string): Promise<string> {
  const apiKey = import.meta.env.VITE_KIMI_API_KEY;

  if (!apiKey) {
    throw new Error("未配置 VITE_KIMI_API_KEY。请在环境变量中设置。");
  }

  const response = await fetch("https://api.moonshot.cn/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "moonshot-v1-8k",
      messages: [
        { 
          role: "system", 
          content: `你是一位专门从事电子邮件安全的网络安全专家。
          用户正在询问关于他们刚刚分析的一封特定电子邮件的问题。
          电子邮件上下文：
          ${emailContext}
          
          请提供有帮助、清晰且可操作的建议。如果邮件有危险，请强调谨慎。请始终使用中文回答。` 
        },
        ...messages
      ],
      temperature: 0.7,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error?.message || "AI 聊天请求失败");
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

import React, { useState, useEffect, useRef } from 'react';
import { Shield, ShieldAlert, ShieldCheck, History, Trash2, Mail, Send, Loader2, AlertCircle, MessageSquare, X, ChevronDown, ChevronUp, Cpu, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import ReactMarkdown from 'react-markdown';
import { analyzeEmail, traditionalAnalyzeEmail, SpamAnalysisResult, sendMessageToAI } from './services/geminiService';
import { DEFAULT_TRAINING_DATA } from './services/spamClassifier';

interface AnalysisHistory extends SpamAnalysisResult {
  id: string;
  subject: string;
  body: string;
  timestamp: number;
  mode: 'ai' | 'traditional';
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
}

declare global {
  interface Window {
    aistudio: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }
}

export default function App() {
  const [subject, setSubject] = useState('');
  const [body, setBody] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<SpamAnalysisResult | null>(null);
  const [history, setHistory] = useState<AnalysisHistory[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'ai' | 'traditional'>('ai');

  // Chat State
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Load history from localStorage
  useEffect(() => {
    const savedHistory = localStorage.getItem('spamguard_history');
    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);
        // Ensure every item has a unique ID to prevent duplicate key warnings
        const validatedHistory = Array.isArray(parsed) ? parsed.map((item: any) => ({
          ...item,
          id: item.id || crypto.randomUUID()
        })) : [];
        setHistory(validatedHistory);
      } catch (e) {
        console.error('Failed to load history', e);
      }
    }
  }, []);

  // Save history to localStorage
  useEffect(() => {
    localStorage.setItem('spamguard_history', JSON.stringify(history));
  }, [history]);

  // Scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!subject.trim() && !body.trim()) return;

    setIsAnalyzing(true);
    setError(null);
    setResult(null);
    setChatMessages([]);
    setIsChatOpen(false);

    try {
      const analysis = mode === 'ai' 
        ? await analyzeEmail(subject, body)
        : await traditionalAnalyzeEmail(subject, body);
        
      setResult(analysis);
      
      const newEntry: AnalysisHistory = {
        ...analysis,
        id: crypto.randomUUID(),
        subject,
        body,
        timestamp: Date.now(),
        mode,
      };
      
      setHistory(prev => [newEntry, ...prev].slice(0, 20));
    } catch (err) {
      setError('分析邮件失败。请重试。');
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || isChatLoading) return;

    const userMessage = chatInput;
    setChatInput('');
    const newUserMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', text: userMessage };
    setChatMessages(prev => [...prev, newUserMessage]);
    setIsChatLoading(true);

    try {
      const context = `主题: ${subject}\n正文: ${body}\n分析结果: ${result?.isSpam ? '垃圾邮件' : '合法邮件'} (${result?.category})`;
      
      // Prepare history for the API
      const historyForAI = chatMessages.map(msg => ({
        role: msg.role,
        content: msg.text
      }));
      
      // Add current user message
      historyForAI.push({ role: 'user', content: userMessage });

      const responseText = await sendMessageToAI(historyForAI, context);
      setChatMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'assistant', text: responseText }]);
    } catch (err) {
      console.error("Chat failed", err);
      setChatMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'assistant', text: "抱歉，我遇到了一个错误。请确保已在设置中配置 KIMI_API_KEY。" }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('spamguard_history');
  };

  const deleteHistoryItem = (id: string) => {
    setHistory(prev => prev.filter(item => item.id !== id));
  };

  const loadFromHistory = (item: AnalysisHistory) => {
    setSubject(item.subject);
    setBody(item.body);
    setResult(item);
    setMode(item.mode || 'ai');
    setChatMessages([]);
    setIsChatOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-[#1e293b] font-sans selection:bg-blue-100">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-blue-600 p-1.5 rounded-lg">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900">SpamGuard AI 垃圾邮件卫士</h1>
          </div>
          <div className="flex items-center gap-4">
            <div className="hidden sm:block text-xs font-medium text-slate-500 uppercase tracking-widest bg-slate-100 px-2 py-1 rounded">
              高级安全防护
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-3 gap-8 pb-24">
        {/* Input Section */}
        <div className="lg:col-span-2 space-y-6">
            <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <Mail className="w-5 h-5 text-blue-600" />
                分析电子邮件
              </h2>
              
              {/* Mode Selector */}
              <div className="flex bg-slate-100 p-1 rounded-xl border border-slate-200">
                <button
                  onClick={() => setMode('ai')}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    mode === 'ai' 
                      ? 'bg-white text-blue-600 shadow-sm' 
                      : 'text-slate-500 hover:text-slate-700'
                  }`}
                >
                  <Cpu className="w-3.5 h-3.5" />
                  AI 智能模式
                </button>
                <button
                  onClick={() => setMode('traditional')}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    mode === 'traditional' 
                      ? 'bg-white text-blue-600 shadow-sm' 
                      : 'text-slate-500 hover:text-slate-700'
                  }`}
                >
                  <Zap className="w-3.5 h-3.5" />
                  传统模式
                </button>
              </div>
            </div>

            <form onSubmit={handleAnalyze} className="space-y-4">
              <div>
                <label htmlFor="subject" className="block text-sm font-medium text-slate-700 mb-1">
                  邮件主题
                </label>
                <input
                  id="subject"
                  type="text"
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  placeholder="例如：您好，恭喜您获得100万大奖，请点击链接领取：http://fake-link.com"
                  className="w-full px-4 py-2 rounded-xl border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all outline-none"
                />
              </div>
              <div>
                <label htmlFor="body" className="block text-sm font-medium text-slate-700 mb-1">
                  邮件内容
                </label>
                <textarea
                  id="body"
                  rows={8}
                  value={body}
                  onChange={(e) => setBody(e.target.value)}
                  placeholder="在此粘贴完整的邮件正文..."
                  className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all outline-none resize-none"
                />
              </div>
              <button
                type="submit"
                disabled={isAnalyzing || (!subject.trim() && !body.trim())}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white font-semibold py-3 rounded-xl transition-all flex items-center justify-center gap-2 shadow-lg shadow-blue-200"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    AI 正在分析中...
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    {mode === 'ai' ? '开始 AI 分析' : '开始传统分析'}
                  </>
                )}
              </button>
            </form>
          </section>

          {/* Result Section */}
          <AnimatePresence mode="wait">
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl flex items-start gap-3"
              >
                <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                <p className="text-sm">{error}</p>
              </motion.div>
            )}

            {result && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                <section
                  className={`rounded-2xl border p-6 shadow-sm ${
                    result.isSpam 
                      ? 'bg-red-50 border-red-200' 
                      : 'bg-emerald-50 border-emerald-200'
                  }`}
                >
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                      {result.isSpam ? (
                        <div className="bg-red-600 p-2 rounded-full">
                          <ShieldAlert className="w-6 h-6 text-white" />
                        </div>
                      ) : (
                        <div className="bg-emerald-600 p-2 rounded-full">
                          <ShieldCheck className="w-6 h-6 text-white" />
                        </div>
                      )}
                      <div>
                        <h3 className={`text-xl font-bold ${result.isSpam ? 'text-red-900' : 'text-emerald-900'}`}>
                          {result.isSpam ? '疑似垃圾邮件' : '疑似合法邮件'}
                        </h3>
                        <p className={`text-sm font-medium ${result.isSpam ? 'text-red-700' : 'text-emerald-700'}`}>
                          类别：{result.category}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-2xl font-black ${result.isSpam ? 'text-red-900' : 'text-emerald-900'}`}>
                        {Math.round(result.confidence * 100)}%
                      </div>
                      <div className={`text-xs font-bold uppercase tracking-wider ${result.isSpam ? 'text-red-600' : 'text-emerald-600'}`}>
                        置信度
                      </div>
                    </div>
                  </div>

                  <div className="prose prose-sm max-w-none prose-slate">
                    <div className={`p-4 rounded-xl bg-white/50 border ${result.isSpam ? 'border-red-100' : 'border-emerald-100'}`}>
                      <h4 className="text-sm font-bold uppercase tracking-wider text-slate-500 mb-2">AI 分析推理</h4>
                      <ReactMarkdown>{result.reasoning}</ReactMarkdown>
                    </div>
                  </div>

                  {/* Chat Affordance */}
                  <div className="mt-6 pt-6 border-t border-slate-200/50 flex justify-end">
                    <button
                      onClick={() => setIsChatOpen(true)}
                      className="flex items-center gap-2 text-sm font-semibold text-slate-600 hover:text-slate-900 transition-colors"
                    >
                      <MessageSquare className="w-4 h-4" />
                      咨询 AI 专家
                    </button>
                  </div>
                </section>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Sidebar / History */}
        <div className="space-y-6">
          <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 overflow-hidden flex flex-col max-h-[calc(100vh-8rem)]">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <History className="w-5 h-5 text-slate-400" />
                历史记录
              </h2>
              {history.length > 0 && (
                <button
                  onClick={clearHistory}
                  className="text-xs text-slate-400 hover:text-red-500 transition-colors font-medium"
                >
                  清空全部
                </button>
              )}
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 pr-2 -mr-2">
              {history.length === 0 ? (
                <div className="text-center py-12">
                  <div className="bg-slate-50 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                    <History className="w-6 h-6 text-slate-300" />
                  </div>
                  <p className="text-sm text-slate-400">暂无分析历史</p>
                </div>
              ) : (
                history.map((item) => (
                  <div
                    key={item.id}
                    onClick={() => loadFromHistory(item)}
                    className="group relative bg-slate-50 hover:bg-white hover:shadow-md border border-transparent hover:border-slate-200 p-3 rounded-xl cursor-pointer transition-all"
                  >
                    <div className="flex items-start justify-between gap-2 mb-1">
                      <span className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${
                        item.isSpam ? 'bg-red-100 text-red-700' : 'bg-emerald-100 text-emerald-700'
                      }`}>
                        {item.isSpam ? '垃圾邮件' : '正常邮件'}
                      </span>
                      <span className="text-[10px] font-medium text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded">
                        {item.mode === 'ai' ? 'AI' : '传统'}
                      </span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteHistoryItem(item.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 text-slate-400 hover:text-red-500 transition-all"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                    <h4 className="text-sm font-semibold text-slate-900 truncate">
                      {item.subject || '(无主题)'}
                    </h4>
                    <p className="text-xs text-slate-500 line-clamp-1 mt-0.5">
                      {item.body}
                    </p>
                    <div className="mt-2 text-[10px] text-slate-400 font-medium">
                      {new Date(item.timestamp).toLocaleDateString()} • {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>

          {/* Info Card */}
          <section className="bg-blue-600 rounded-2xl p-6 text-white shadow-lg shadow-blue-200">
            <h3 className="font-bold mb-2 flex items-center gap-2">
              <ShieldCheck className="w-5 h-5" />
              工作原理
            </h3>
            <div className="space-y-4 text-sm text-blue-100 leading-relaxed">
              <div>
                <p className="font-bold text-white mb-1">AI 智能模式</p>
                <p>使用先进的大语言模型分析语言模式、元数据线索和复杂的网络钓鱼策略。</p>
              </div>
              <div className="pt-2 border-t border-blue-500/50">
                <p className="font-bold text-white mb-1">传统模式 (机器学习)</p>
                <p>基于朴素贝叶斯算法，通过学习 {DEFAULT_TRAINING_DATA.length} 个样本（垃圾邮件和正常邮件）的词频分布进行快速匹配，不依赖外部 AI 服务。</p>
              </div>
            </div>
          </section>
        </div>
      </main>

      {/* Chat Interface Overlay */}
      <AnimatePresence>
        {isChatOpen && (
          <motion.div
            initial={{ opacity: 0, y: 100 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 100 }}
            className="fixed bottom-0 right-0 left-0 sm:left-auto sm:right-8 sm:bottom-8 sm:w-96 z-50 px-4 pb-4 sm:px-0 sm:pb-0"
          >
            <div className="bg-white rounded-2xl shadow-2xl border border-slate-200 flex flex-col h-[500px]">
              {/* Chat Header */}
              <div className="p-4 border-b border-slate-100 flex items-center justify-between bg-slate-50 rounded-t-2xl">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                  <h3 className="font-bold text-sm">AI 安全专家</h3>
                </div>
                <button 
                  onClick={() => setIsChatOpen(false)}
                  className="p-1 hover:bg-slate-200 rounded-lg transition-colors"
                >
                  <X className="w-4 h-4 text-slate-500" />
                </button>
              </div>

              {/* Chat Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {chatMessages.length === 0 && (
                  <div className="text-center py-8">
                    <p className="text-xs text-slate-400">询问关于此邮件分析的任何问题...</p>
                  </div>
                )}
                {chatMessages.map((msg) => (
                  <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[85%] p-3 rounded-2xl text-sm ${
                      msg.role === 'user' 
                        ? 'bg-blue-600 text-white rounded-br-none' 
                        : 'bg-slate-100 text-slate-800 rounded-bl-none'
                    }`}>
                      <div className="prose prose-sm prose-invert max-w-none">
                        <ReactMarkdown>
                          {msg.text}
                        </ReactMarkdown>
                      </div>
                    </div>
                  </div>
                ))}
                {isChatLoading && (
                  <div className="flex justify-start">
                    <div className="bg-slate-100 p-3 rounded-2xl rounded-bl-none">
                      <Loader2 className="w-4 h-4 animate-spin text-slate-400" />
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Chat Input */}
              <form onSubmit={handleSendMessage} className="p-4 border-t border-slate-100">
                <div className="relative">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="输入您的问题..."
                    className="w-full pl-4 pr-12 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                  />
                  <button
                    type="submit"
                    disabled={!chatInput.trim() || isChatLoading}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 bg-blue-600 text-white rounded-lg disabled:bg-slate-300 transition-colors"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </div>
              </form>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Floating Chat Button (when closed) */}
      {!isChatOpen && result && (
        <motion.button
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          onClick={() => setIsChatOpen(true)}
          className="fixed bottom-8 right-8 w-14 h-14 bg-blue-600 text-white rounded-full shadow-xl flex items-center justify-center hover:bg-blue-700 transition-all z-40 group"
        >
          <MessageSquare className="w-6 h-6" />
          <span className="absolute right-full mr-4 bg-slate-900 text-white text-xs py-1.5 px-3 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
            咨询 AI 专家
          </span>
        </motion.button>
      )}
    </div>
  );
}

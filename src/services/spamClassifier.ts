import spamExamples from '../data/spam_examples.json';
import hamExamples from '../data/ham_examples.json';

/**
 * A simple Naive Bayes Classifier for Spam Detection.
 * It learns from a training set of spam and ham (normal) emails.
 */

export interface TrainingExample {
  subject?: string;
  body: string;
  isSpam: boolean;
}

export class NaiveBayesClassifier {
  private spamWordCounts: Map<string, number> = new Map();
  private hamWordCounts: Map<string, number> = new Map();
  private spamTotalWords = 0;
  private hamTotalWords = 0;
  private spamDocCount = 0;
  private hamDocCount = 0;
  private vocabulary: Set<string> = new Set();
  private featureScores: Map<string, number> = new Map(); // Mutual Information scores

  constructor(examples: TrainingExample[] = []) {
    if (examples.length > 0) {
      this.train(examples);
    }
  }

  /**
   * Tokenizes text into words, removing punctuation and converting to lowercase.
   * Also extracts synthetic features like URLs, currency, etc.
   */
  private tokenize(text: string, isSubject = false): string[] {
    const tokens: string[] = [];
    const stopWords = new Set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '个', '为', '以', '于', '他', '她', '它', '们', '来', '用', '中', '大', '小', '多', '少', '对', '能', '如', '地', '得', '而', '及', '与', '或', '但', '其', '此', '之', '由', '被', '把', '让', '使', '给', '等', '及', '并', '且', '又', '再', '还', '也', '即', '却', '虽', '然', '虽', '然', '但', '是', '因', '为', '所', '以', '如', '果', '那', '么', '如', '此', '这', '样', '那', '样', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'if', 'then', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'as', 'of']);
    
    // Synthetic Features
    if (/[!?]{2,}/.test(text)) tokens.push('[EXCESSIVE_PUNCTUATION]');
    if (/[¥$€£]/.test(text)) tokens.push('[HAS_CURRENCY]');
    if (/https?:\/\/[^\s]+/.test(text)) tokens.push('[HAS_URL]');
    if (/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/.test(text)) tokens.push('[HAS_EMAIL]');
    
    // Suspicious keywords (Spam indicators)
    const spamKeywords = [
      '免费', '中奖', '赢取', '大奖', '发票', '代开', '避税', '兼职', '刷单', '赚钱', '暴利', '秘密', '紧急', '立即', '最后机会', '限时', '优惠', '折扣', '特价', '退订', 'unsubscribe', 'free', 'win', 'prize', 'winner', 'urgent', 'immediately', 'limited', 'offer', 'discount', 'invoice', 'tax', 'profit', 'secret',
      '救助', '方案', '专项', '申请', '详情', '点击', '了解', '回复', '拒收', '注意', '普查', '健康', '皮肤', '白癜风', '银屑病', '牛皮癣', '医疗', '专家', '名额', '领取'
    ];
    for (const keyword of spamKeywords) {
      if (text.toLowerCase().includes(keyword)) {
        tokens.push(`[KEYWORD_${keyword.toUpperCase()}]`);
      }
    }

    // Behavioral patterns
    if (/回复\s*[“"']?\s*\d\s*[”"']?/.test(text)) tokens.push('[REPLY_WITH_NUMBER]');
    if (/回复\s*[“"']?\s*[a-zA-Z]\s*[”"']?/.test(text)) tokens.push('[REPLY_WITH_LETTER]');
    if (/拒收.*回复/.test(text)) tokens.push('[UNSUBSCRIBE_PATTERN]');
    if (/申请.*方案/.test(text)) tokens.push('[APPLICATION_PATTERN]');

    // Handle "spaced out" words (e.g., "f r e e")
    const deSpaced = text.replace(/([a-z])\s(?=[a-z]\s)/gi, '$1').replace(/([a-z])\s([a-z])($|\s)/gi, '$1$2');
    
    // Normalize: lowercase and handle numbers
    const normalizedText = deSpaced.toLowerCase()
      .replace(/\d{7,}/g, ' [LONG_NUMBER] ')
      .replace(/\d+/g, ' [NUMBER] ');

    const rawTokens = normalizedText.split(/[\s,.;:!?()\[\]{}'"]+/);
    
    for (const token of rawTokens) {
      if (!token || token.length < 1) continue;
      
      const prefix = isSubject ? 'sub:' : '';
      
      // If it's Chinese characters
      if (/[\u4e00-\u9fa5]/.test(token)) {
        const chars = token.split('').filter(c => /[\u4e00-\u9fa5]/.test(c));
        
        // Unigrams
        for (const char of chars) {
          if (!stopWords.has(char)) {
            tokens.push(prefix + char);
          }
        }
        
        // Bigrams
        for (let i = 0; i < chars.length - 1; i++) {
          tokens.push(prefix + chars[i] + chars[i+1]);
        }

        // Trigrams
        for (let i = 0; i < chars.length - 2; i++) {
          tokens.push(prefix + chars[i] + chars[i+1] + chars[i+2]);
        }
      } else {
        // English/Other
        if (!stopWords.has(token) && token.length > 1) {
          tokens.push(prefix + token);
          
          // Check for ALL CAPS (English only)
          if (token === token.toUpperCase() && /[a-z]/i.test(token)) {
            tokens.push('[ALL_CAPS]');
          }
        }
      }
    }
    return tokens;
  }

  /**
   * Trains the classifier with a set of examples.
   * Uses Mutual Information for feature selection.
   */
  public train(examples: TrainingExample[]) {
    // 1. Initial counts
    const wordDocCounts: Map<string, { spam: number; ham: number }> = new Map();
    
    for (const example of examples) {
      const subjectTokens = example.subject ? this.tokenize(example.subject, true) : [];
      const bodyTokens = this.tokenize(example.body, false);
      const uniqueTokens = new Set([...subjectTokens, ...bodyTokens]);
      
      if (example.isSpam) {
        this.spamDocCount++;
        for (const token of uniqueTokens) {
          const counts = wordDocCounts.get(token) || { spam: 0, ham: 0 };
          counts.spam++;
          wordDocCounts.set(token, counts);
        }
      } else {
        this.hamDocCount++;
        for (const token of uniqueTokens) {
          const counts = wordDocCounts.get(token) || { spam: 0, ham: 0 };
          counts.ham++;
          wordDocCounts.set(token, counts);
        }
      }
    }

    // 2. Feature Selection using Mutual Information
    const totalDocs = this.spamDocCount + this.hamDocCount;
    const pSpam = this.spamDocCount / totalDocs;
    const pHam = this.hamDocCount / totalDocs;
    
    const miScores: { token: string; score: number }[] = [];
    
    for (const [token, counts] of wordDocCounts.entries()) {
      // P(W=1)
      const pW1 = (counts.spam + counts.ham) / totalDocs;
      const pW0 = 1 - pW1;
      
      // Mutual Information calculation
      // MI = sum P(W,C) * log(P(W,C) / (P(W)*P(C)))
      let mi = 0;
      
      // P(W=1, C=Spam)
      const pW1S = counts.spam / totalDocs;
      if (pW1S > 0) mi += pW1S * Math.log(pW1S / (pW1 * pSpam));
      
      // P(W=1, C=Ham)
      const pW1H = counts.ham / totalDocs;
      if (pW1H > 0) mi += pW1H * Math.log(pW1H / (pW1 * pHam));
      
      // P(W=0, C=Spam)
      const pW0S = (this.spamDocCount - counts.spam) / totalDocs;
      if (pW0S > 0) mi += pW0S * Math.log(pW0S / (pW0 * pSpam));
      
      // P(W=0, C=Ham)
      const pW0H = (this.hamDocCount - counts.ham) / totalDocs;
      if (pW0H > 0) mi += pW0H * Math.log(pW0H / (pW0 * pHam));
      
      miScores.push({ token, score: mi });
    }

    // Keep top 5000 most informative features (increased from 3000)
    // Also filter out features that appear in fewer than 2 documents to avoid overfitting
    // BUT always keep synthetic features (starting with '[')
    const selectedFeatures = new Set(
      miScores
        .filter(f => f.token.startsWith('[') || (wordDocCounts.get(f.token)?.spam || 0) + (wordDocCounts.get(f.token)?.ham || 0) >= 2)
        .sort((a, b) => b.score - a.score)
        .slice(0, 5000)
        .map(f => f.token)
    );

    // 3. Final training with selected features
    for (const example of examples) {
      const subjectTokens = example.subject ? this.tokenize(example.subject, true) : [];
      const bodyTokens = this.tokenize(example.body, false);
      const allTokens = [...subjectTokens, ...bodyTokens];
      
      if (example.isSpam) {
        for (const token of allTokens) {
          if (selectedFeatures.has(token)) {
            this.spamWordCounts.set(token, (this.spamWordCounts.get(token) || 0) + 1);
            this.spamTotalWords++;
            this.vocabulary.add(token);
          }
        }
      } else {
        for (const token of allTokens) {
          if (selectedFeatures.has(token)) {
            this.hamWordCounts.set(token, (this.hamWordCounts.get(token) || 0) + 1);
            this.hamTotalWords++;
            this.vocabulary.add(token);
          }
        }
      }
    }
  }

  /**
   * Classifies a text and returns the probability of it being spam.
   */
  public classify(subject: string, body: string): { isSpam: boolean; confidence: number; matchedFeatures: string[] } {
    const subjectTokens = this.tokenize(subject, true);
    const bodyTokens = this.tokenize(body, false);
    const tokens = [...subjectTokens, ...bodyTokens];
    
    const totalDocs = this.spamDocCount + this.hamDocCount;
    if (totalDocs === 0) return { isSpam: false, confidence: 0.5, matchedFeatures: [] };

    // Log-probabilities to avoid underflow
    let spamLogProb = Math.log(this.spamDocCount / totalDocs);
    let hamLogProb = Math.log(this.hamDocCount / totalDocs);

    const vocabSize = this.vocabulary.size;
    const features: { token: string; score: number }[] = [];
    const alpha = 0.1; // Lidstone smoothing (often better than Laplace +1)

    for (const token of tokens) {
      if (!this.vocabulary.has(token)) continue;

      // Lidstone smoothing
      const spamWordProb = ((this.spamWordCounts.get(token) || 0) + alpha) / (this.spamTotalWords + alpha * vocabSize);
      const hamWordProb = ((this.hamWordCounts.get(token) || 0) + alpha) / (this.hamTotalWords + alpha * vocabSize);

      // Weighting: Subject tokens and special features have more impact
      let weight = 1.0;
      if (token.startsWith('sub:')) weight = 2.5;
      if (token.startsWith('[KEYWORD_')) weight = 3.0; // High weight for manually defined keywords
      if (token.startsWith('[REPLY_')) weight = 5.0; // Very high weight for behavioral patterns
      if (token === '[UNSUBSCRIBE_PATTERN]') weight = 5.0;
      if (token === '[APPLICATION_PATTERN]') weight = 3.0;
      
      let spamLog = Math.log(spamWordProb);
      let hamLog = Math.log(hamWordProb);

      // Manual boost for strong indicators if they are not in training data or have low score
      if (token.startsWith('[REPLY_') || token === '[UNSUBSCRIBE_PATTERN]') {
        spamLog += 2.0; // Boost spam probability
      }

      spamLogProb += spamLog * weight;
      hamLogProb += hamLog * weight;
      
      features.push({ token, score: (spamLog - hamLog) * weight });
    }

    // Convert log-probabilities back to probability
    const maxLog = Math.max(spamLogProb, hamLogProb);
    const spamProb = Math.exp(spamLogProb - maxLog) / (Math.exp(spamLogProb - maxLog) + Math.exp(hamLogProb - maxLog));

    // Sort features by absolute influence
    const sortedFeatures = features
      .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
      .filter((f, index, self) => self.findIndex(t => t.token === f.token) === index)
      .slice(0, 15)
      .map(f => {
        let label = f.token;
        if (label.startsWith('sub:')) label = `[主题] ${label.substring(4)}`;
        if (label.startsWith('[KEYWORD_')) label = `[关键词] ${label.substring(9, label.length - 1)}`;
        if (label === '[REPLY_WITH_NUMBER]') label = '[模式] 回复数字';
        if (label === '[REPLY_WITH_LETTER]') label = '[模式] 回复字母';
        if (label === '[UNSUBSCRIBE_PATTERN]') label = '[模式] 拒收退订';
        if (label === '[APPLICATION_PATTERN]') label = '[模式] 申请方案';
        return `${label} (${f.score > 0 ? '+' : ''}${f.score.toFixed(2)})`;
      });

    // Use a slightly more conservative threshold for spam (0.65 instead of 0.6)
    return {
      isSpam: spamProb > 0.65,
      confidence: Math.max(spamProb, 1 - spamProb),
      matchedFeatures: sortedFeatures
    };
  }
}

// Construct the default training set from external JSON files
export const DEFAULT_TRAINING_DATA: TrainingExample[] = [
  ...spamExamples.map(text => ({ body: text, isSpam: true })),
  ...hamExamples.map(text => ({ body: text, isSpam: false }))
];

import React, { useState, useEffect, useRef, useMemo } from 'react';
import axios from 'axios';
import {
  MessageSquare, Database, FileText, Send, Trash2,
  Upload, ShieldCheck, Plus, History, ChevronRight,
  RefreshCw
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// --- API ---
const API_BASE = 'http://localhost:8001/api';

// --- Types ---
interface Message {
  role: 'user' | 'ai';
  content: string;
}

interface CaseState {
  phase: string;
  intent: string;
  case_summary: string;
  risk_alerts?: string[];
  missing_slots?: string[];
  slots?: Record<string, any>;
  [key: string]: any;
}

interface Session {
  id: string;
  title: string;
  messages: Message[];
  state: CaseState | null;
  lastUpdated: number;
}

interface Chunk {
  id: string;
  content: string;
  metadata: Record<string, any>;
}

type UploadDocType = 'case' | 'law' | 'interpretation';

interface UploadTask {
  id: string;
  fileName: string;
  docType: UploadDocType;
  status: 'waiting' | 'uploading' | 'processing' | 'done' | 'error';
  progress: number;
  errorMessage?: string;
}

const DOC_TYPE_LABEL: Record<UploadDocType, string> = {
  case: '裁判文书',
  law: '法律条款',
  interpretation: '司法解释',
};

const App: React.FC = () => {
  // --- Global Navigation ---
  const [view, setView] = useState<'chat' | 'docs' | 'db'>('chat');
  const [dbTab, setDbTab] = useState<UploadDocType>('case');

  // --- Session Management ---
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  // --- Upload Queue & Control ---
  const [uploadQueue, setUploadQueue] = useState<UploadTask[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [uploadType, setUploadType] = useState<UploadDocType>('case');
  const isPausedRef = useRef(false);

  // --- UI States ---
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingChunks, setLoadingChunks] = useState(false);
  const [chunks, setChunks] = useState<{ law_articles: Chunk[], court_cases: Chunk[] }>({ law_articles: [], court_cases: [] });
  const [expandedCase, setExpandedCase] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);

  // --- 1. Persistence Logic ---
  useEffect(() => {
    const saved = localStorage.getItem('law_judge_sessions');
    if (saved) {
      const parsed = JSON.parse(saved);
      setSessions(parsed);
      if (parsed.length > 0) setActiveSessionId(parsed[0].id);
    } else {
      createNewSession();
    }
  }, []);

  useEffect(() => {
    if (sessions.length > 0) {
      localStorage.setItem('law_judge_sessions', JSON.stringify(sessions));
    }
  }, [sessions]);

  // --- 2. Scrolling Logic ---
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSessionId, loading, sessions]);

  useEffect(() => {
    if (view === 'db' || view === 'chat') fetchChunks();
  }, [view]);

  // --- Session Helpers ---
  const createNewSession = () => {
    const newSession: Session = {
      id: Date.now().toString(),
      title: `新案件 ${new Date().toLocaleTimeString()}`,
      messages: [{ role: 'ai', content: '您好，我是司法智能助手，很高兴能协助您进行民商事法律实务及模拟质证建议。' }],
      state: null,
      lastUpdated: Date.now()
    };
    setSessions(prev => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
    setView('chat');
  };

  const deleteSession = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!window.confirm('确定要删除此会话吗？')) return;
    const filtered = sessions.filter(s => s.id !== id);
    setSessions(filtered);
    if (activeSessionId === id) {
      setActiveSessionId(filtered.length > 0 ? filtered[0].id : null);
    }
  };

  const currentSession = sessions.find(s => s.id === activeSessionId);

  // --- Handlers ---
  const fetchChunks = async () => {
    setLoadingChunks(true);
    try {
      const chunksRes = await axios.get(`${API_BASE}/chunks`);

      setChunks({
        law_articles: (chunksRes.data.law_articles || []).filter((c: any) => c && c.metadata),
        court_cases: (chunksRes.data.court_cases || []).filter((c: any) => c && c.metadata)
      });
    } catch (e) {
      console.error('Fetch chunks failed:', e);
    } finally {
      setTimeout(() => setLoadingChunks(false), 500);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || !activeSessionId || loading) return;

    const userMsg = input.trim();
    const sessionId = activeSessionId;

    setSessions(prev => prev.map(s => s.id === sessionId ? {
      ...s,
      messages: [...s.messages, { role: 'user', content: userMsg }],
      lastUpdated: Date.now()
    } : s));

    setInput('');
    setLoading(true);

    try {
      const prevState = currentSession?.state;
      const res = await axios.post(`${API_BASE}/chat`, {
        message: userMsg,
        state_override: prevState
      });

      setSessions(prev => prev.map(s => s.id === sessionId ? {
        ...s,
        messages: res.data.messages,
        state: res.data.state,
        lastUpdated: Date.now()
      } : s));
    } catch (e) {
      setSessions(prev => prev.map(s => s.id === sessionId ? {
        ...s,
        messages: [...s.messages, { role: 'ai', content: '⚠️ 后端响应失败。请检查后台服务状态。' }],
        lastUpdated: Date.now()
      } : s));
    } finally {
      setLoading(false);
    }
  };

  const handleBatchUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const newTasks: UploadTask[] = Array.from(files).map(f => ({
      id: Math.random().toString(36).substring(2, 9),
      fileName: f.name,
      docType: uploadType,
      status: 'waiting',
      progress: 0
    }));

    setUploadQueue(prev => [...prev, ...newTasks]);

    for (const file of Array.from(files)) {
      while (isPausedRef.current) await new Promise(r => setTimeout(r, 500));
      await processFileUpload(file, uploadType);
    }
    e.target.value = '';
  };

  const processFileUpload = async (file: File, type: UploadDocType) => {
    setUploadQueue(prev => prev.map(t => t.fileName === file.name ? { ...t, status: 'uploading' } : t));
    const formData = new FormData();
    formData.append('file', file);
    try {
      await axios.post(`${API_BASE}/upload`, formData, {
        params: { doc_type: type },
        onUploadProgress: (p) => {
          const percent = Math.min(Math.round((p.loaded * 100) / (p.total || 100)), 85);
          setUploadQueue(q => q.map(t => t.fileName === file.name ? { ...t, progress: percent } : t));
        }
      });
      setUploadQueue(q => q.map(t => t.fileName === file.name ? { ...t, status: 'processing' } : t));
    } catch (e) {
      setUploadQueue(q => q.map(t => t.fileName === file.name ? { ...t, status: 'error', errorMessage: '上传失败' } : t));
    }
  };

  // --- Task Polling ---
  useEffect(() => {
    const hasActive = uploadQueue.some(t => t.status === 'uploading' || t.status === 'processing');
    if (!hasActive) return;

    const poll = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE}/ingestion/tasks`);
        const backend = res.data;
        setUploadQueue(prev => prev.map(t => {
          const b = backend[t.fileName];
          if (!b) return t;
          if (b.status === 'completed') return { ...t, status: 'done', progress: 100 };
          if (b.status === 'failed') return { ...t, status: 'error', progress: 100, errorMessage: b.error };
          if (b.status === 'processing' || b.status === 'pending') return { ...t, status: 'processing', progress: Math.max(t.progress, 90) };
          return t;
        }));
      } catch (e) { console.error(e); }
    }, 2000);
    return () => clearInterval(poll);
  }, [uploadQueue]);

  const handleDeleteChunk = async (type: 'case' | 'law', id: string) => {
    if (!window.confirm('确认删除此切块？')) return;
    try {
      await axios.delete(`${API_BASE}/chunks/${type}/${id}`);
      fetchChunks();
    } catch (e) { alert('删除失败'); }
  };

  const handleDeleteFile = async (type: string, name: string) => {
    if (!window.confirm(`确认物理删除整份【${name}】及其所有关联索引？`)) return;
    try {
      await axios.delete(`${API_BASE}/files/${type}/${encodeURIComponent(name)}`);
      fetchChunks();
    } catch (e) { alert('删除文档失败，请检查索引库状态。'); }
  };

  // --- Grouping logic for Explorer ---
  const groupedCases = useMemo(() => {
    const map = new Map<string, Chunk[]>();
    chunks.court_cases.forEach(c => {
      const name = c.metadata?.case_name || c.metadata?.case_id || '未命名文书';
      if (!map.has(name)) map.set(name, []);
      map.get(name)!.push(c);
    });
    return Array.from(map.entries());
  }, [chunks.court_cases]);

  const groupedLaw = useMemo(() => {
    // 自动分离法律与解释
    const targetSubtype = dbTab === 'law' ? 'law' : dbTab;
    const source = chunks.law_articles.filter(c => (c.metadata?.doc_subtype || 'law') === targetSubtype);
    const map = new Map<string, Chunk[]>();
    source.forEach(c => {
      const name = c.metadata?.law_name || c.metadata?.document_name || '未命名文档';
      if (!map.has(name)) map.set(name, []);
      map.get(name)!.push(c);
    });
    return Array.from(map.entries());
  }, [chunks.law_articles, dbTab]);

  return (
    <div className="app-container">
      {/* 动画 Blob 背景 */}
      <div className="aurora-container">
        <div className="aurora-blob blob-1"></div>
        <div className="aurora-blob blob-2"></div>
        <div className="aurora-blob blob-3"></div>
      </div>

      <nav className="sidebar glass-panel">
        <div className="logo" style={{ fontSize: '1.8rem', marginBottom: '20px' }}>⚖️</div>
        <button onClick={() => setView('chat')} className={`nav-btn ${view === 'chat' ? 'active' : ''}`}><MessageSquare size={24} /><span>庭审</span></button>
        <button onClick={() => setView('docs')} className={`nav-btn ${view === 'docs' ? 'active' : ''}`}><FileText size={24} /><span>文档</span></button>
        <button onClick={() => setView('db')} className={`nav-btn ${view === 'db' ? 'active' : ''}`}><Database size={24} /><span>知识</span></button>
      </nav>

      <main className="main-content">
        <AnimatePresence mode="wait">
          {view === 'chat' && (
            <motion.div key="chat" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} className="chat-layout">
              <aside className="session-sidebar glass-panel">
                <button className="new-chat-btn" onClick={createNewSession}><Plus size={18} /> 案情咨询</button>
                <div className="session-list">
                  {sessions.map(s => (
                    <div
                      key={s.id}
                      className={`session-item ${activeSessionId === s.id ? 'active' : ''}`}
                      onClick={() => setActiveSessionId(s.id)}
                    >
                      <History size={14} style={{ opacity: 0.6 }} />
                      <span className="session-title">{s.title}</span>
                      <Trash2 size={14} className="delete-icon" onClick={(e) => deleteSession(e, s.id)} />
                    </div>
                  ))}
                </div>
              </aside>

              <section className="chat-main-area">
                <div className="chat-container glass-panel">
                  <div className="msg-list">
                    {currentSession ? currentSession.messages.map((m, i) => (
                      <div key={i} className={`msg-bubble ${m.role === 'user' ? 'msg-user' : 'msg-ai'}`}>
                        <div className="msg-content" style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
                      </div>
                    )) : <div className="empty-state">尚未开启会话</div>}
                    {loading && <div className="msg-bubble msg-ai typing"><span className="dot"></span><span className="dot"></span><span className="dot"></span></div>}
                    <div ref={chatEndRef} />
                  </div>
                  <div className="chat-input-area">
                    <textarea
                      value={input}
                      onChange={e => setInput(e.target.value)}
                      onKeyDown={e => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleSend();
                        }
                      }}
                      className="chat-input"
                      placeholder="描述案情细节，或提供证据线索..."
                      rows={Math.min(input.split('\n').length, 5)}
                      disabled={loading || !activeSessionId}
                    />
                    <button onClick={handleSend} disabled={loading || !activeSessionId} className="send-btn"><Send size={20} /></button>
                  </div>
                </div>
              </section>

              <aside className="info-panel glass-panel">
                <div className="info-header"><ShieldCheck size={20} color="var(--secondary)" /> 案件大脑状态</div>
                <div className="state-info">
                  <div className="state-item"><span className="label">当前审判阶段</span><span className={`badge-phase ${currentSession?.state?.phase ? 'active' : ''}`}>{currentSession?.state?.phase || 'Reception'}</span></div>
                  <div className="state-item"><span className="label">识别意图</span><span className="value">{currentSession?.state?.intent || '分析中'}</span></div>
                </div>
                {currentSession?.state?.risk_alerts && currentSession.state.risk_alerts.length > 0 && (
                  <div className="risk-radar-section">
                    <h4 className="risk-title">🚨 实时风险预警</h4>
                    {currentSession.state.risk_alerts.map((r, idx) => <div key={idx} className="risk-item-card">{r}</div>)}
                  </div>
                )}
                {currentSession?.state?.missing_slots && currentSession.state.missing_slots.length > 0 && (
                  <div className="missing-slots-section">
                    <h4 className="slots-title">📝 待查明事实要素</h4>
                    <div className="slots-grid">{currentSession.state.missing_slots.map(s => <span key={s} className="slot-badge-missing">{s}</span>)}</div>
                  </div>
                )}
                {currentSession?.state?.case_summary && (
                  <div className="summary-section">
                    <h4 className="summary-title">案情摘要看板</h4>
                    <div className="summary-box">{currentSession.state.case_summary}</div>
                  </div>
                )}
              </aside>
            </motion.div>
          )}

          {view === 'docs' && (
            <motion.section key="docs" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} className="glass-panel docs-view">
              <h2>📂 法律文库数据中心</h2>
              <div className="upload-tabs">
                <button onClick={() => setUploadType('case')} className={uploadType === 'case' ? 'active' : ''}>🏛️ 裁判文书</button>
                <button onClick={() => setUploadType('law')} className={uploadType === 'law' ? 'active' : ''}>📜 法律条款</button>
                <button onClick={() => setUploadType('interpretation')} className={uploadType === 'interpretation' ? 'active' : ''}>⚖️ 司法解释</button>
              </div>
              <div className="upload-zone" onClick={() => document.getElementById('fileInput')?.click()}>
                <div className="upload-icon-wrapper"><Upload size={48} /></div>
                <p>点击或拖拽上传 {DOC_TYPE_LABEL[uploadType]} 格式资料</p>
                <input id="fileInput" type="file" multiple style={{ display: 'none' }} onChange={handleBatchUpload} />
              </div>

              {uploadQueue.length > 0 && (
                <div className="upload-progress-container">
                  <div className="queue-header"><h4>入库任务队列列表</h4> <button onClick={() => setUploadQueue([])} className="clear-queue-btn">清空</button></div>
                  <div className="progress-list">
                    {uploadQueue.map(t => (
                      <div key={t.id} className={`progress-card ${t.status}`}>
                        <div className="progress-info"><span className="file-name">[{DOC_TYPE_LABEL[t.docType]}] {t.fileName}</span> <span className={`status-badge ${t.status}`}>{t.status === 'done' ? '✅ 已入库' : t.status === 'processing' ? '⚙️ 正在解析' : t.status}</span></div>
                        <div className="progress-bar-bg"><motion.div className="progress-bar-fill" animate={{ width: `${t.progress}%` }} /></div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.section>
          )}

          {view === 'db' && (
            <motion.section key="db" initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.98 }} className="glass-panel db-view">
              <div className="db-header">
                <h2>🔍 SeekDB 知识库资源扫描</h2>
                <div className="tab-group">
                  <button onClick={() => setDbTab('case')} className={dbTab === 'case' ? 'active' : ''}>案例库</button>
                  <button onClick={() => setDbTab('law')} className={dbTab === 'law' ? 'active' : ''}>法条库</button>
                  <button onClick={() => setDbTab('interpretation')} className={dbTab === 'interpretation' ? 'active' : ''}>解释库</button>
                </div>
                <button onClick={fetchChunks} className="refresh-btn" disabled={loadingChunks}><RefreshCw size={16} className={loadingChunks ? 'spinning' : ''} /></button>
              </div>
              <div className="db-content-scroll">
                {(dbTab === 'case' ? groupedCases : groupedLaw).length === 0 && <div className="empty-db">暂无库内数据，请前往文档中心上传。</div>}
                {(dbTab === 'case' ? groupedCases : groupedLaw).map(([name, list]) => (
                  <div key={name} className="case-group-panel">
                    <div className="case-header" onClick={() => setExpandedCase(expandedCase === name ? null : name)}>
                      <span className="case-title">{dbTab === 'case' ? '📄' : (dbTab === 'law' ? '📜' : '⚖️')} {name}</span>
                      <div className="case-meta"><span>{list.length} 切块</span> <button className="delete-file-btn" onClick={(e) => { e.stopPropagation(); handleDeleteFile(dbTab, name); }}><Trash2 size={16} /></button> <ChevronRight size={16} style={{ transition: '0.3s', transform: expandedCase === name ? 'rotate(90deg)' : 'none' }} /></div>
                    </div>
                    {expandedCase === name && (
                      <div className="chunk-list">
                        {list.map(c => (
                          <div key={c.id} className="chunk-item">
                            <div className="chunk-top">
                              <span className="tag-type">{dbTab === 'case' ? (c.metadata?.logic_type || 'case') : (c.metadata?.article_num || '全文段')}</span>
                              <button onClick={() => handleDeleteChunk(dbTab === 'case' ? 'case' : 'law', c.id)} className="delete-btn"><Trash2 size={16} /></button>
                            </div>
                            <div className="chunk-body" style={{ whiteSpace: 'pre-wrap' }}>{c.content}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </motion.section>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
};

export default App;

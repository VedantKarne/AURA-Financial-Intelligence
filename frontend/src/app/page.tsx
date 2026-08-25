'use client'; //It is a special instruction understood by Next.js.
//"Everything in this file should run inside the user's browser."

import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function Home() {
  const [activeTab, setActiveTab] = useState('chat');
  const [mounted, setMounted] = useState(false);
  const [systemTime, setSystemTime] = useState('');

  useEffect(() => {
    setMounted(true);
    // Dynamic system clock for terminal aesthetic
    const updateTime = () => {
      const now = new Date();
      setSystemTime(now.toUTCString().replace('GMT', 'UTC'));
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  if (!mounted) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#0A0F1E' }}>
        <div className="pulse" style={{ color: '#00D4FF', fontSize: '1.2rem', fontFamily: 'monospace' }}>
          SECURE PROTOCOL INITIALIZING...
        </div>
      </div>
    );
  }

  return (
    <div className="container" style={{ padding: '2rem 1.5rem', maxWidth: '1440px', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Premium Header */}
      <header className="glass glass-card" style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1.5rem 2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {/* Futuristic SVG Logo */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '40px', height: '40px', background: 'rgba(0, 245, 160, 0.05)', borderRadius: '10px', border: '1px solid rgba(0, 245, 160, 0.2)' }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L2 22h20L12 2z" stroke="url(#logoGradient)" strokeWidth="2" strokeLinejoin="round" />
              <path d="M12 7l6 11H6l6-11z" fill="url(#logoGradient)" fillOpacity="0.2" />
              <circle cx="12" cy="13" r="2" fill="#8B5CF6" />
              <defs>
                <linearGradient id="logoGradient" x1="2" y1="22" x2="22" y2="2" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#00F5A0" />
                  <stop offset="1" stopColor="#10B981" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: '1.4rem', fontFamily: 'var(--font-display)', fontWeight: 800, letterSpacing: '-0.02em', background: 'linear-gradient(135deg, #FFF 60%, #CBD5E1 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              AURA INTELLIGENCE
            </h1>
            <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span className="dot-status"></span> AI-Engine Connected
            </p>
          </div>
        </div>

        {/* Tab Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
          <div className="tab-container">
            <button
              className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              Intelligence Chat
            </button>
            <button
              className={`tab-btn ${activeTab === 'kpi' ? 'active' : ''}`}
              onClick={() => setActiveTab('kpi')}
            >
              KPI Analytics
            </button>
            <button
              className={`tab-btn ${activeTab === 'report' ? 'active' : ''}`}
              onClick={() => setActiveTab('report')}
            >
              Briefing Generator
            </button>
          </div>

          {/* Terminal System Time */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            <span>SYS_TIME: {systemTime}</span>
            <span style={{ color: 'var(--primary-accent)' }}>SECURE_NET_OK</span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {activeTab === 'chat' && <ChatInterface />}
        {activeTab === 'kpi' && <KpiDashboard />}
        {activeTab === 'report' && <ReportGenerator />}
      </main>
    </div>
  );
}

function ChatInterface() {
  const [query, setQuery] = useState('');
  const [kValue, setKValue] = useState(6);
  const [messages, setMessages] = useState<{
    role: string;
    content: string;
    sources?: { citation: string; snippet: string }[];
  }[]>([]);
  const [loading, setLoading] = useState(false);
  const [threadId, setThreadId] = useState<string>('');
  const [history, setHistory] = useState<string[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messageEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Set thread ID
    const randomId = Math.random().toString(36).substring(2, 15);
    setThreadId(randomId);

    // Fetch history from local storage
    const saved = localStorage.getItem('aura_query_history');
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error(e);
      }
    }
  }, []);

  useEffect(() => {
    // Scroll to bottom on new messages
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (customQuery?: string) => {
    const activeQuery = customQuery || query;
    if (!activeQuery.trim()) return;

    const userMsg = { role: 'user', content: activeQuery, sources: [] };
    setMessages(prev => [...prev, userMsg]);
    setQuery('');
    setLoading(true);

    // Save query to history
    setHistory(prev => {
      const updated = [activeQuery, ...prev.filter(item => item !== activeQuery)].slice(0, 8);
      localStorage.setItem('aura_query_history', JSON.stringify(updated));
      return updated;
    });

    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg.content, k: kValue, thread_id: threadId })
      });
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.message, sources: data.sources || [] }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error connecting to API.', sources: [] }]);
    }
    setLoading(false);
  };

  const handleSuggestionClick = (suggestionText: string) => {
    setQuery(suggestionText);
    textareaRef.current?.focus();
  };

  const clearChat = () => {
    setMessages([]);
    const randomId = Math.random().toString(36).substring(2, 15);
    setThreadId(randomId);
  };

  const deleteHistoryItem = (e: React.MouseEvent, item: string) => {
    e.stopPropagation();
    setHistory(prev => {
      const updated = prev.filter(q => q !== item);
      localStorage.setItem('aura_query_history', JSON.stringify(updated));
      return updated;
    });
  };

  // Helper to parse source citations out of markdown assistant messages
  const parseCitations = (text: string) => {
    const markers = ["### Source Documents:", "Source Documents:", "Sources:"];
    let index = -1;
    let foundMarker = "";

    for (const m of markers) {
      index = text.indexOf(m);
      if (index !== -1) {
        foundMarker = m;
        break;
      }
    }

    if (index === -1) {
      return { mainText: text, citations: [] };
    }

    const mainText = text.substring(0, index).trim();
    const citationsSection = text.substring(index + foundMarker.length).trim();
    const citations = citationsSection
      .split('\n')
      .map(line => line.replace(/^-\s*/, '').trim())
      .filter(line => line.length > 0);

    return { mainText, citations };
  };

  // Preprocess raw markdown text to convert citations [Company | Quarter | Year | Section]
  // into custom markdown links that can be caught by ReactMarkdown custom components
  const preprocessMarkdown = (text: string, sources: { citation: string, snippet: string }[]) => {
    if (!sources || sources.length === 0) return text;

    // Matches Company | Quarter | Year | Section patterns inside brackets
    const citationRegex = /\[\s*([A-Za-z0-9\s]+)\s*\|\s*(Q\d)\s*\|\s*(\d{4})\s*\|\s*([A-Za-z0-9\s]+)\s*\]/g;

    const citationIndices: Record<string, number> = {};
    let counter = 0;

    return text.replace(citationRegex, (match, co, q, yr, sec) => {
      const normalizedKey = `${co.trim()} | ${q.trim()} | ${yr.trim()} | ${sec.trim()}`;

      const sourceExists = sources.some(s => {
        const sKey = s.citation.replace(/\s+/g, '').toLowerCase();
        const nKey = normalizedKey.replace(/\s+/g, '').toLowerCase();
        return sKey.includes(nKey) || nKey.includes(sKey);
      });
      if (!sourceExists) return match;

      const matchingSource = sources.find(s => {
        const sKey = s.citation.replace(/\s+/g, '').toLowerCase();
        const nKey = normalizedKey.replace(/\s+/g, '').toLowerCase();
        return sKey.includes(nKey) || nKey.includes(sKey);
      });

      const keyToUse = matchingSource ? matchingSource.citation : normalizedKey;

      if (!citationIndices[keyToUse]) {
        counter++;
        citationIndices[keyToUse] = counter;
      }
      const idx = citationIndices[keyToUse];
      return `[${idx}](citation:${encodeURIComponent(keyToUse)})`;
    });
  };

  function CitationBubble({ citation, snippet, index }: { citation: string; snippet?: string; index: React.ReactNode }) {
    return (
      <span className="inline-citation-bubble-container">
        <span className="inline-citation-bubble">{index}</span>
        {snippet && (
          <span className="citation-tooltip-panel">
            <span className="citation-tooltip-header">{citation}</span>
            <span className="citation-tooltip-body">{snippet}</span>
          </span>
        )}
      </span>
    );
  }

  const promptSuggestions = [
    "Summarize Q3 guidance for Apple",
    "Microsoft Cloud revenue performance",
    "Compare CapEx trends in Nvidia vs MSFT",
    "Evaluate Apple Q3 headwinds & risks"
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: '1.5rem', flex: 1, minHeight: 'calc(100vh - 12rem)' }}>

      {/* Chat Sidebar Controls */}
      <div className="glass glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem', height: 'fit-content', padding: '1.5rem' }}>
        <div>
          <h4 style={{ margin: '0 0 1rem 0', fontSize: '0.85rem', color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.12em', textAlign: 'center', fontWeight: 700 }}>
            Intelligence Tuning
          </h4>
          <div style={{ height: '1px', background: 'linear-gradient(90deg, transparent, rgba(0,245,160,0.3), transparent)', marginBottom: '1rem' }} />
          <div style={{ paddingBottom: '1.25rem', borderBottom: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
              Select number of top references to generate response from:
            </span>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              <span>Response Richness</span>
              <strong style={{ color: 'var(--primary-accent)' }}>{kValue} references</strong>
            </div>
            <input
              type="range"
              min="1"
              max="30"
              value={kValue}
              onChange={(e) => setKValue(parseInt(e.target.value))}
            />
            <small style={{ display: 'block', fontSize: '0.7rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
              Higher values broaden document context coverage; lower values yield faster, more focused insights.
            </small>
          </div>
        </div>
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
            <h4 style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.12em', fontWeight: 700 }}>
              Query History
            </h4>
            {history.length > 0 && (
              <button
                onClick={() => { setHistory([]); localStorage.removeItem('aura_query_history'); }}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', fontSize: '0.75rem', cursor: 'pointer', hover: { color: 'var(--danger)' } } as any}
              >
                Clear All
              </button>
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '200px', overflowY: 'auto' }}>
            {history.length === 0 ? (
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>No search history</span>
            ) : (
              history.map((h, i) => (
                <div
                  key={i}
                  onClick={() => handleSend(h)}
                  className="suggestion-chip"
                  style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '7px 10px', borderRadius: 'var(--radius-sm)', overflow: 'hidden', textOverflow: 'ellipsis' }}
                >
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '85%' }}>{h}</span>
                  <svg
                    width="12"
                    height="12"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    style={{ cursor: 'pointer', opacity: 0.6 }}
                    onClick={(e) => deleteHistoryItem(e, h)}
                  >
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </div>
              ))
            )}
          </div>
        </div>

        <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1.25rem', marginTop: 'auto' }}>
          <button className="btn btn-danger" onClick={clearChat} style={{ width: '100%', fontSize: '0.82rem', padding: '0.6rem 1rem' }}>
            Reset Conversation
          </button>
        </div>
      </div>

      {/* Main Chat Feed Area */}
      <div className="glass glass-card" style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: '600px', padding: '1.5rem 1.5rem 1rem 1.5rem' }}>

        {/* Messages Feed */}
        <div style={{ flex: 1, overflowY: 'auto', marginBottom: '1rem', paddingRight: '0.5rem' }}>
          {messages.length === 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '350px', textAlign: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '64px', height: '64px', borderRadius: '50%', background: 'rgba(139, 92, 246, 0.08)', border: '1px solid rgba(139, 92, 246, 0.15)', marginBottom: '1.5rem' }}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" strokeWidth="1.5">
                  <path d="M9.813 15.904L9 21l5.545-3.327A10.5 10.5 0 1012 3c-5.799 0-10.5 4.7-10.5 10.5 0 2.062.593 3.987 1.616 5.617l.006.01L9.813 15.9z" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M9.75 9.75h4.5m-4.5 3h4.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <h3 style={{ margin: '0 0 0.5rem 0', fontFamily: 'var(--font-display)', fontWeight: 700 }}>
                What financial intelligence do you require?
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', maxWidth: '450px', margin: 0 }}>
                Ask qualitative or quantitative questions across companies. The AI Engine will execute hybrid RAG across transcripts and databases.
              </p>
            </div>
          ) : (
            messages.map((m, i) => {
              const isUser = m.role === 'user';
              const { mainText, citations } = isUser ? { mainText: m.content, citations: [] } : parseCitations(m.content);

              return (
                <div key={i} className={`chat-message ${isUser ? 'chat-user' : 'chat-ai'}`}>
                  <div style={{ display: 'flex', gap: '0.75rem', maxWidth: '85%', flexDirection: isUser ? 'row-reverse' : 'row' }}>

                    {/* Avatar Badge */}
                    <div style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '50%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.75rem',
                      fontWeight: 'bold',
                      flexShrink: 0,
                      background: isUser ? 'rgba(0, 245, 160, 0.08)' : 'rgba(139, 92, 246, 0.08)',
                      border: `1px solid ${isUser ? 'rgba(0, 245, 160, 0.25)' : 'rgba(139, 92, 246, 0.25)'}`,
                      color: isUser ? 'var(--primary-accent)' : 'var(--ai-accent)'
                    }}>
                      {isUser ? 'U' : 'AI'}
                    </div>

                    <div className="chat-bubble">
                      {isUser ? (
                        <div style={{ whiteSpace: 'pre-wrap' }}>{mainText}</div>
                      ) : (
                        <div className="report-markdown">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              a: ({ href, children }) => {
                                if (href && href.startsWith('citation:')) {
                                  const citationKey = decodeURIComponent(href.replace('citation:', ''));
                                  const source = m.sources?.find(s => s.citation.trim() === citationKey.trim());
                                  return (
                                    <CitationBubble
                                      citation={citationKey}
                                      snippet={source?.snippet}
                                      index={children}
                                    />
                                  );
                                }
                                return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
                              }
                            }}
                          >
                            {preprocessMarkdown(mainText, m.sources || [])}
                          </ReactMarkdown>
                        </div>
                      )}

                      {/* Display Citations Beautifully if present */}
                      {!isUser && citations.length > 0 && (
                        <div className="citations-wrapper">
                          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px' }}>
                            Reference Sources
                          </div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            {citations.map((cite, cIdx) => (
                              <div key={cIdx} className="citation-card">
                                <span className="citation-badge">Ref</span>
                                <span style={{ fontFamily: 'monospace' }}>{cite}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })
          )}
          {loading && (
            <div className="chat-message chat-ai">
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                <div style={{ width: '32px', height: '32px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)', color: 'var(--ai-accent)', fontSize: '0.75rem', fontWeight: 'bold' }}>
                  AI
                </div>
                <div className="chat-bubble pulse" style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--bg-card)', border: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="pulse">
                    <circle cx="12" cy="12" r="10" strokeDasharray="30 10" />
                  </svg>
                  Thinking and retrieving documents...
                </div>
              </div>
            </div>
          )}
          <div ref={messageEndRef} />
        </div>

        {/* Floating suggestion prompts */}
        {messages.length === 0 && (
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '1rem', justifyContent: 'center' }}>
            {promptSuggestions.map((suggestion, index) => (
              <button
                key={index}
                className="suggestion-chip"
                onClick={() => handleSuggestionClick(suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}

        {/* AI Query Experience centerpiece box */}
        <div className="query-box-container">
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '8px' }}>
            {/* Sparkles AI Icon — animated */}
            <div style={{ padding: '10px', color: 'var(--ai-accent)', display: 'flex', alignItems: 'center', position: 'relative' }}>
              {/* Outer pulse ring — visible when typing */}
              {query.trim().length > 0 && (
                <span style={{
                  position: 'absolute',
                  inset: '4px',
                  borderRadius: '50%',
                  border: '1.5px solid rgba(139, 92, 246, 0.4)',
                  animation: 'ping-ring 1.2s ease-out infinite',
                  pointerEvents: 'none',
                }} />
              )}
              <svg
                width="20" height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{
                  animation: query.trim().length > 0
                    ? 'spin-sun 3s linear infinite'
                    : 'spin-sun 8s linear infinite',
                  filter: query.trim().length > 0
                    ? 'drop-shadow(0 0 6px rgba(139, 92, 246, 0.8))'
                    : 'drop-shadow(0 0 3px rgba(139, 92, 246, 0.4))',
                  transition: 'filter 0.3s ease',
                }}
              >
                <path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707m0-12.728l.707.707m11.314 11.314l.707-.707M12 8a4 4 0 100 8 4 4 0 000-8z" />
              </svg>
            </div>

            <textarea
              ref={textareaRef}
              className="query-textarea"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Ask about company risks, earnings trends, strategic initiatives, market outlook, or compare multiple companies..."
            />

            <button
              className="btn btn-primary"
              onClick={() => handleSend()}
              disabled={loading || !query.trim()}
              style={{ padding: '8px 16px', borderRadius: '12px', height: '42px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}

function KpiDashboard() {
  const [kpis, setKpis] = useState<any[]>([]);
  const [company, setCompany] = useState('Apple');
  const [loading, setLoading] = useState(false);

  const fetchKpis = async () => {
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:8000/api/kpis?company=${company}`);
      const data = await res.json();
      setKpis(data.kpis || []);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchKpis();
  }, [company]);

  return (
    <div className="glass glass-card" style={{ padding: '2rem' }}>

      {/* KPI Controls Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2.5rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.8rem', fontFamily: 'var(--font-display)', fontWeight: 700 }}>
            Structured KPI Stream
          </h2>
          <p style={{ margin: '4px 0 0 0', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
            Reported quantitative variables extracted directly from filings databases.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Filer:</label>
          <select
            className="input-field"
            style={{ width: '160px', padding: '10px 14px', background: 'rgba(21, 31, 52, 0.65)' }}
            value={company}
            onChange={(e) => setCompany(e.target.value)}
          >
            <option value="Apple">Apple Inc. (AAPL)</option>
            <option value="Microsoft">Microsoft Corp. (MSFT)</option>
            <option value="Nvidia">Nvidia Corp. (NVDA)</option>
          </select>

          <button
            className="btn btn-outline"
            onClick={fetchKpis}
            disabled={loading}
            style={{ padding: '10px 18px', height: '42px' }}
          >
            Reload Stream
          </button>
        </div>
      </div>

      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '6rem 0' }}>
          <div className="pulse" style={{ display: 'flex', gap: '8px', alignItems: 'center', color: 'var(--primary-accent)', fontFamily: 'monospace' }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="pulse">
              <circle cx="12" cy="12" r="10" strokeDasharray="30 10" />
            </svg>
            EXECUTING QUANT DATA RETRIEVAL...
          </div>
        </div>
      ) : kpis.length === 0 ? (
        <div style={{ padding: '4rem', textAlign: 'center', color: 'var(--text-muted)', border: '1px dashed var(--border-color)', borderRadius: 'var(--radius-lg)' }}>
          No reported financial parameters stored in the database for {company}.
        </div>
      ) : (
        <div className="kpi-grid">
          {kpis.map((kpi, i) => (
            <div key={i} className="kpi-card">

              {/* Card Header */}
              <div className="kpi-header">
                <span style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-display)' }}>
                  {kpi.period}
                </span>
                <span className="dot-status" style={{ background: 'var(--secondary-accent)', boxShadow: '0 0 8px var(--secondary-accent)' }}></span>
              </div>

              {/* Financial metrics rows */}
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <div className="kpi-row">
                  <span className="kpi-label">Revenue:</span>
                  <span className="kpi-val">${kpi.revenue_b}B</span>
                </div>

                {kpi.net_income_b !== undefined && (
                  <div className="kpi-row">
                    <span className="kpi-label">Net Income:</span>
                    <span className="kpi-val">${kpi.net_income_b}B</span>
                  </div>
                )}

                <div className="kpi-row">
                  <span className="kpi-label">Diluted EPS:</span>
                  <span className="kpi-val">${kpi.eps}</span>
                </div>

                <div className="kpi-row">
                  <span className="kpi-label">Gross Margin:</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span className="kpi-val">{kpi.gross_margin_pct}%</span>
                    <div className="gauge-track">
                      <div className="gauge-fill" style={{ width: `${Math.min(100, Math.max(0, kpi.gross_margin_pct))}%` }}></div>
                    </div>
                  </div>
                </div>

                <div className="kpi-row">
                  <span className="kpi-label">YoY Growth:</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <strong style={{ color: kpi.revenue_growth_yoy_pct > 0 ? 'var(--success)' : 'var(--danger)', fontSize: '16px' }}>
                      {kpi.revenue_growth_yoy_pct > 0 ? '+' : ''}{kpi.revenue_growth_yoy_pct}%
                    </strong>
                    {kpi.revenue_growth_yoy_pct > 0 ? (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--success)" strokeWidth="2.5">
                        <path d="M12 19.5v-15m0 0L5.25 11.25M12 4.5l6.75 6.75" />
                      </svg>
                    ) : (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--danger)" strokeWidth="2.5">
                        <path d="M12 4.5v15m0 0l-6.75-6.75M12 19.5l6.75-6.75" />
                      </svg>
                    )}
                  </div>
                </div>

                {kpi.guidance_revenue_low_b && kpi.guidance_revenue_high_b ? (
                  <div className="kpi-row" style={{ marginTop: '6px', paddingTop: '10px', borderTop: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <span className="kpi-label">Revenue Guidance:</span>
                    <span className="kpi-val" style={{ color: 'var(--primary-accent)' }}>
                      ${kpi.guidance_revenue_low_b}B - ${kpi.guidance_revenue_high_b}B
                    </span>
                  </div>
                ) : null}
              </div>

            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ReportGenerator() {
  const [company, setCompany] = useState('Apple');
  const [year, setYear] = useState('2024');
  const [quarter, setQuarter] = useState('Q3');
  const [report, setReport] = useState('');
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(0);

  // Simulated stepper animation while API processes report
  useEffect(() => {
    let interval: any;
    if (loading) {
      setStep(0);
      interval = setInterval(() => {
        setStep(s => (s < 3 ? s + 1 : s));
      }, 3500); // 3.5s per step, total duration ~14s typical for complex RAG synthesis
    } else {
      setStep(0);
    }
    return () => clearInterval(interval);
  }, [loading]);

  const generateReport = async () => {
    setLoading(true);
    setReport('');
    try {
      const res = await fetch('http://localhost:8000/api/generate-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company, year: parseInt(year), quarter })
      });
      const data = await res.json();
      setReport(data.report);
    } catch (err) {
      setReport("Failed to generate report brief.");
    }
    setLoading(false);
  };

  return (
    <div className="report-layout">

      {/* Parameters Panel */}
      <div className="glass glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem', padding: '1.75rem' }}>
        <div style={{ textAlign: 'center' }}>
          <h3 style={{ margin: '0 0 0.4rem 0', fontSize: '1.2rem', fontFamily: 'var(--font-display)', fontWeight: 700 }}>
            Brief Parameters
          </h3>
          <div style={{ height: '1px', background: 'linear-gradient(90deg, transparent, rgba(0,245,160,0.3), transparent)', margin: '0.75rem 0 0.25rem' }} />
          <p style={{ margin: '0.4rem 0 0 0', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Select briefing targets for synthesis.
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.4rem' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '8px' }}>Company</label>
            <select className="input-field" value={company} onChange={e => setCompany(e.target.value)}>
              <option value="Apple">Apple Inc. (AAPL)</option>
              <option value="Microsoft">Microsoft Corp. (MSFT)</option>
              <option value="Nvidia">Nvidia Corp. (NVDA)</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '8px' }}>Calendar Year</label>
            <select className="input-field" value={year} onChange={e => setYear(e.target.value)}>
              <option value="2024">2024</option>
              <option value="2023">2023</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '8px' }}>Target Quarter</label>
            <select className="input-field" value={quarter} onChange={e => setQuarter(e.target.value)}>
              <option value="Q1">Q1 (Q1 Transcript)</option>
              <option value="Q2">Q2 (Q2 Transcript)</option>
              <option value="Q3">Q3 (Q3 Transcript)</option>
              <option value="Q4">Q4 (Q4 Transcript)</option>
            </select>
          </div>

          <button
            className="btn btn-primary"
            onClick={generateReport}
            disabled={loading}
            style={{ marginTop: '0.4rem', width: '100%', padding: '0.75rem 1rem', fontSize: '0.88rem', fontWeight: 600 }}
          >
            {loading ? 'Compiling brief...' : 'Generate Intelligence Brief'}
          </button>
        </div>

        {/* Real-time synthesis log stepper */}
        {loading && (
          <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1.2rem', marginTop: '0.5rem' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Synthesis Pipeline Logs
            </span>

            <div className="stepper-container">
              <div className={`stepper-item ${step >= 0 ? 'active' : ''} ${step > 0 ? 'done' : ''}`}>
                <span className="stepper-dot"></span>
                <span>Initializing deep compiler agents</span>
              </div>
              <div className={`stepper-item ${step >= 1 ? 'active' : ''} ${step > 1 ? 'done' : ''}`}>
                <span className="stepper-dot"></span>
                <span>Extracting quarterly KPI metrics</span>
              </div>
              <div className={`stepper-item ${step >= 2 ? 'active' : ''} ${step > 2 ? 'done' : ''}`}>
                <span className="stepper-dot"></span>
                <span>Analyzing transcripts via vector store</span>
              </div>
              <div className={`stepper-item ${step >= 3 ? 'active' : ''} ${step > 3 ? 'done' : ''}`}>
                <span className="stepper-dot"></span>
                <span>Synthesizing investment brief report</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Report Viewport */}
      <div className="glass glass-card" style={{ minHeight: '65vh', padding: '2rem' }}>
        {loading && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '400px' }}>
            <div className="pulse" style={{ color: 'var(--primary-accent)', fontSize: '0.9rem', fontFamily: 'monospace', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="pulse">
                <circle cx="12" cy="12" r="10" strokeDasharray="30 10" />
              </svg>
              AI AGENT PIPELINE RUNNING - GATHERING CORPUS DATA...
            </div>
          </div>
        )}

        {!loading && !report && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '400px', textAlign: 'center' }}>
            <div style={{ width: '48px', height: '48px', borderRadius: '50%', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', marginBottom: '1.2rem' }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <h4 style={{ margin: '0 0 0.25rem 0', fontWeight: 600 }}>No intelligence report generated yet</h4>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', maxWidth: '380px', margin: 0 }}>
              Adjust parameters in the left panel and click the generate button to compile a structured investment briefing brief.
            </p>
          </div>
        )}

        {!loading && report && (
          <div className="report-markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
          </div>
        )}
      </div>

    </div>
  );
}

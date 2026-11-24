import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import "./App.css";

const API_BASE = `${import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"}/api`;

const SESSION_STORAGE_KEY = "medassist-session-id";
const SESSION_ID = sessionStorage.getItem(SESSION_STORAGE_KEY) || `session_${Date.now()}`;
sessionStorage.setItem(SESSION_STORAGE_KEY, SESSION_ID);

// ── Icon helpers ──────────────────────────────────────────────────────
const STEP_ICONS = {
  orchestrator: "🧠",
  thought:      "💭",
  action:       "🔧",
  observation:  "👁️",
  critique:     "✅",
  answer:       "📋",
};

const QUALITY_METRIC_LABELS = {
  task_clarity: "Clear task",
  topic_specificity: "Specific topic",
  context_detail: "Useful context",
  output_guidance: "Requested format",
  citation_coverage: "Claims with citations",
  retrieved_source_match: "Citations match retrieved sources",
  query_term_relevance: "Topic-word overlap (heuristic)",
  answer_detail: "Answer detail and structure",
};

function App() {
  const [question,       setQuestion]       = useState("");
  const [messages,       setMessages]       = useState([]);
  const [documents,      setDocuments]      = useState([]);
  const [systemStatus,   setSystemStatus]   = useState(null);
  const [quotaNow, setQuotaNow] = useState(0);
  const [isGenerating,   setIsGenerating]   = useState(false);
  const [isUploading,    setIsUploading]    = useState(false);
  const [uploadStatus,   setUploadStatus]   = useState("");
  const [isDragging,     setIsDragging]     = useState(false);
  const [traceOpen,      setTraceOpen]      = useState(false);
  const [activeTrace,    setActiveTrace]    = useState([]);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [analyticsData, setAnalyticsData] = useState(null);
  const [analyticsError, setAnalyticsError] = useState("");
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem("medassist-theme") !== "light");

  const messagesEndRef = useRef(null);
  const fileInputRef   = useRef(null);
  const messageIdRef = useRef(0);
  const quotaSecondsLeft = systemStatus?.llm_rate_limited
    ? Math.max(0, Math.ceil(((systemStatus.quotaExpiresAt || 0) - quotaNow) / 1000))
    : 0;

  // ── API calls ─────────────────────────────────────────────────────
  const fetchStatus = useCallback(() => {
    axios.get(`${API_BASE}/status`)
      .then(res => {
        const status = res.data;
        const retrySeconds = status.llm_retry_after_seconds || 0;
        const now = Date.now();
        setQuotaNow(now);
        setSystemStatus({
          ...status,
          quotaExpiresAt: now + retrySeconds * 1000,
        });
      })
      .catch(() => setSystemStatus(null));
  }, []);

  const fetchDocuments = useCallback(() => {
    axios.get(`${API_BASE}/documents`)
      .then(res => setDocuments(res.data.files || res.data || []))
      .catch(e => console.error("Error fetching documents:", e));
  }, []);

  useEffect(() => {
    fetchStatus();
    fetchDocuments();
  }, [fetchStatus, fetchDocuments]);

  useEffect(() => {
    if (!systemStatus?.llm_rate_limited) return undefined;

    const timer = window.setInterval(() => {
      setQuotaNow(Date.now());
    }, 1000);
    return () => window.clearInterval(timer);
  }, [systemStatus?.llm_rate_limited]);

  useEffect(() => {
    if (systemStatus?.llm_rate_limited && quotaSecondsLeft === 0) {
      fetchStatus();
    }
  }, [fetchStatus, quotaSecondsLeft, systemStatus?.llm_rate_limited]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isGenerating]);

  useEffect(() => {
    localStorage.setItem("medassist-theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  // ── Main chat handler ─────────────────────────────────────────────
  const askQuestion = async (prefilledText) => {
    const query = prefilledText || question;
    if (!query.trim() || isGenerating) return;
    if (!prefilledText) setQuestion("");

    const messageId = ++messageIdRef.current;
    const createdAt = new Date();
    const userMsg = {
      id:        `${messageId}-user`,
      role:      "user",
      text:      query,
      timestamp: createdAt,
    };
    const assistantMsgId = `${messageId}-assistant`;
    const assistantMsg = {
      id:         assistantMsgId,
      role:       "assistant",
      text:       "",
      sources:    [],
      trace:      [],
      intent:     "",
      quality:    "",
      qualityMetrics: null,
      timestamp:  createdAt,
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setIsGenerating(true);
    setActiveTrace([]);
    setTraceOpen(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ question: query, session_id: SESSION_ID }),
      });

      if (!response.ok) {
        throw new Error(`Chat request failed (${response.status})`);
      }
      if (!response.body) throw new Error("No response body");

      const reader  = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) continue;

          try {
            const event = JSON.parse(trimmed.substring(6));

            if (event.type === "trace") {
              // Agent reasoning step
              setActiveTrace(prev => [...prev, event.data]);
              setMessages(prev => prev.map(m =>
                m.id === assistantMsgId
                  ? { ...m, trace: [...(m.trace || []), event.data] }
                  : m
              ));
            }

            if (event.type === "token") {
              // Stream answer text
              setMessages(prev => prev.map(m =>
                m.id === assistantMsgId
                  ? { ...m, text: m.text + event.data }
                  : m
              ));
            }

            if (event.type === "done") {
              // Final metadata
              setMessages(prev => prev.map(m =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      sources: event.data.sources || [],
                      intent:  event.data.intent  || "",
                      quality: event.data.quality || "",
                      qualityMetrics: event.data.quality_metrics || null,
                    }
                  : m
              ));
            }

            if (event.type === "error") {
              // Error from backend
              const errorMessage = String(event.data || "The AI request failed.");
              setMessages(prev => prev.map(m =>
                m.id === assistantMsgId
                  ? { ...m, text: `⚠️ ${errorMessage}` }
                  : m
              ));
              if (/Groq.*(?:quota|rate limit)/i.test(errorMessage)) {
                fetchStatus();
              }
            }
          } catch (parseErr) {
            console.error("SSE parse error:", parseErr);
          }
        }
      }

    } catch (err) {
      console.error("Chat error:", err);
      setMessages(prev => prev.map(m =>
        m.id === assistantMsgId
          ? { ...m, text: "❌ Error connecting to the AI backend. Please ensure the server is running." }
          : m
      ));
    } finally {
      setIsGenerating(false);
    }
  };

  // ── Upload handler ────────────────────────────────────────────────
  const handleUploadFile = async (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadStatus("Only PDF files are allowed.");
      return;
    }
    setIsUploading(true);
    setUploadStatus("Uploading & indexing...");

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await axios.post(`${API_BASE}/upload`, form);
      setUploadStatus(res.data.message || "Uploaded successfully!");
      fetchDocuments();
      fetchStatus();
      setTimeout(() => setUploadStatus(""), 5000);
    } catch (e) {
      setUploadStatus(e.response?.data?.detail || "Upload failed. Check that the backend is available.");
      setTimeout(() => setUploadStatus(""), 5000);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteDocument = async (filename) => {
    if (!window.confirm(`Remove "${filename}" from the knowledge base?`)) return;
    try {
      await axios.delete(`${API_BASE}/documents/${encodeURIComponent(filename)}`);
      fetchDocuments();
      fetchStatus();
    } catch (e) {
      alert(e.response?.data?.detail || "Failed to delete document.");
    }
  };

  const handleClearHistory = async () => {
    if (!window.confirm("Clear conversation memory for this session?")) return;
    try {
      await axios.delete(`${API_BASE}/chat/history/${SESSION_ID}`);
      setMessages([]);
      setActiveTrace([]);
    } catch {
      alert("Failed to clear history.");
    }
  };

  const handleAnalytics = async () => {
    if (showAnalytics) {
      setShowAnalytics(false);
      return;
    }

    setShowAnalytics(true);
    setAnalyticsError("");
    try {
      const res = await axios.get(`${API_BASE}/analytics/stats`);
      setAnalyticsData(res.data);
    } catch (e) {
      setAnalyticsError(e.response?.data?.detail || "Could not load analytics.");
    }
  };

  const exportCitations = async (sources) => {
    if (!sources?.length) return;
    const params = new URLSearchParams();
    sources.forEach(source => params.append("sources", source));
    try {
      const response = await axios.get(
        `${API_BASE}/citations/${encodeURIComponent(SESSION_ID)}?${params.toString()}`,
        { responseType: "blob" },
      );
      const url = URL.createObjectURL(response.data);
      const link = document.createElement("a");
      link.href = url;
      link.download = "medassist-citations.bib";
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      alert(e.response?.data?.detail || "Could not export citations.");
    }
  };

  // ── Drag & Drop ───────────────────────────────────────────────────
  const handleDragOver  = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = ()  => setIsDragging(false);
  const handleDrop      = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) handleUploadFile(e.dataTransfer.files[0]);
  };

  // ── Markdown-like renderer for AI responses ───────────────────────
  const renderAnswer = (text) => {
    if (!text) return null;

    // Split into sections by ## headers
    const parts = text.split(/^##\s+/m).filter(Boolean);

    if (parts.length <= 1) {
      // No sections — just render paragraphs
      return (
        <div className="ai-body">
          {text.split("\n\n").map((p, i) => (
            <p key={i} className="ai-para">{p}</p>
          ))}
        </div>
      );
    }

    return (
      <div className="ai-body">
        {parts.map((section, i) => {
          const nl = section.indexOf("\n");
          const heading = nl !== -1 ? section.substring(0, nl).trim() : section.trim();
          const body    = nl !== -1 ? section.substring(nl + 1).trim() : "";

          const contentBlocks = [];
          let bulletItems = [];
          const flushBullets = () => {
            if (bulletItems.length) {
              contentBlocks.push(
                <ul key={`list-${contentBlocks.length}`} className="ai-bullets">
                  {bulletItems.map((item, index) => <li key={index}>{item}</li>)}
                </ul>,
              );
              bulletItems = [];
            }
          };
          body.split("\n").forEach(line => {
            const bullet = line.match(/^\s*[•*-]\s+(.+)/);
            if (bullet) {
              bulletItems.push(bullet[1].trim());
              return;
            }
            flushBullets();
            if (line.trim()) {
              contentBlocks.push(
                <p key={`paragraph-${contentBlocks.length}`} className="ai-para">{line.trim()}</p>,
              );
            }
          });
          flushBullets();

          return (
            <div key={i} className="ai-section">
              {heading && <div className="ai-section-title">## {heading}</div>}
              {contentBlocks}
            </div>
          );
        })}
      </div>
    );
  };

  // ── Sample questions ──────────────────────────────────────────────
  const samples = [
    "What are the symptoms and risk factors of Type 2 Diabetes?",
    "I have chest pain, shortness of breath, and fatigue. What could this be?",
    "How is asthma diagnosed and managed according to clinical guidelines?",
    "Summarize the key findings in the Cardiovascular diseases document.",
  ];

  // ── Render ────────────────────────────────────────────────────────
  return (
    <div className={`app-layout${darkMode ? "" : " light-theme"}`}>

      {/* ── Sidebar ── */}
      <div className="sidebar">
        <div className="sidebar-header">
          <span className="logo-icon">🩺</span>
          <h1 className="app-title">MedAssist-AI</h1>
          <div className="app-subtitle">Agentic Medical RAG System</div>
        </div>

        <div className="sidebar-body">

          {/* System Status */}
          <section className="sidebar-section">
            <h2 className="section-heading">System Status</h2>
            <div className="status-card">
              {systemStatus ? (
                <>
                  <div className="status-row">
                    <span className={`status-dot ${systemStatus.llm_configured && !systemStatus.llm_rate_limited ? "online" : "offline"}`} />
                    <span className="status-label">LLM:</span>
                    <span className="status-value">
                      {systemStatus.llm_rate_limited
                        ? "Provider quota reached"
                        : systemStatus.llm_configured
                          ? systemStatus.llm_model
                          : "API key not configured"}
                    </span>
                  </div>
                  {systemStatus.llm_rate_limited && (
                    <div className="status-row">
                      <span className="status-label">Retry in:</span>
                      <span className="status-value">
                        {quotaSecondsLeft > 0
                          ? `${Math.floor(quotaSecondsLeft / 60)}:${String(quotaSecondsLeft % 60).padStart(2, "0")}`
                          : "checking provider"}
                      </span>
                    </div>
                  )}
                  <div className="status-row">
                    <span className="status-label">Retrieval:</span>
                    <span className="status-value">Hybrid BM25 + FAISS</span>
                  </div>
                  <div className="status-row">
                    <span className="status-label">Chunks:</span>
                    <span className="status-value">{systemStatus.total_chunks}</span>
                  </div>
                  <div className="status-row">
                    <span className="status-label">Agents:</span>
                    <span className="status-value">{systemStatus.agents?.length ?? 0}</span>
                  </div>
                  <div className="status-row">
                    <span className="status-label">Tools:</span>
                    <span className="status-value">{systemStatus.tools?.join(", ")}</span>
                  </div>
                  <details className="status-capabilities">
                    <summary>Agent and tool details</summary>
                    <div className="status-capability-group">
                      <strong>Orchestration</strong>
                      <span>
                        {systemStatus.orchestrator?.name}: {systemStatus.orchestrator?.routing}
                      </span>
                    </div>
                    {systemStatus.agent_details?.map(agent => (
                      <div className="status-capability-group" key={agent.name}>
                        <strong>{agent.name}</strong>
                        <span>{agent.type} · up to {agent.max_iterations} tool steps</span>
                      </div>
                    ))}
                    <div className="status-capability-group">
                      <strong>{systemStatus.quality_scorer?.name}</strong>
                      <span>{systemStatus.quality_scorer?.type}</span>
                    </div>
                    {systemStatus.tool_details?.map(tool => (
                      <div className="status-capability-group" key={tool.name}>
                        <strong>{tool.name}</strong>
                        <span>{tool.description}</span>
                      </div>
                    ))}
                  </details>
                </>
              ) : (
                <div className="status-row">
                  <span className="status-dot offline" />
                  <span className="status-value">Backend offline</span>
                </div>
              )}
            </div>
          </section>

          {/* Document Library */}
          <section className="sidebar-section">
            <h2 className="section-heading">Document Library</h2>
            <div className="doc-library">
              {documents.length === 0 ? (
                <div className="empty-state">No documents indexed.</div>
              ) : (
                documents.map((doc, i) => (
                  <div key={i} className="doc-item">
                    <div className="doc-info">
                      <span className="doc-name" title={doc.name}>{doc.name}</span>
                      <div className="doc-meta">
                        <span className="doc-tag">{doc.category}</span>
                        <span className="doc-chunks">{doc.chunks} chunks</span>
                      </div>
                    </div>
                    <button
                      className="doc-delete-btn"
                      onClick={() => handleDeleteDocument(doc.name)}
                      title="Remove"
                    >🗑️</button>
                  </div>
                ))
              )}
            </div>
          </section>

          {/* Upload */}
          <section className="sidebar-section">
            <h2 className="section-heading">Add Document</h2>
            <div
              className={`upload-zone${isDragging ? " dragging" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="upload-icon">📤</div>
              <div className="upload-text">Drag & drop a medical PDF</div>
              <div className="upload-sub">or click to browse</div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                style={{ display: "none" }}
                onChange={e => handleUploadFile(e.target.files?.[0])}
              />
            </div>
            {uploadStatus && (
              <div className="upload-status">
                {isUploading && <span className="spinner" />}
                {uploadStatus}
              </div>
            )}
          </section>

        </div>
      </div>

      {/* ── Main Panel ── */}
      <div className="main-panel">

        {/* Chat Header */}
        <div className="chat-header">
          <div>
            <h2>Medical Agent Console</h2>
            <p>Multi-agent RAG with ReAct reasoning, memory, and transparent quality metrics</p>
          </div>
          <div className="header-actions">
            <button className="btn-ghost" onClick={handleAnalytics}>
              📊 Analytics
            </button>
            <button className="btn-ghost" onClick={() => setDarkMode(v => !v)}>
              {darkMode ? "☀️ Light mode" : "🌙 Dark mode"}
            </button>
            <button className="btn-ghost" onClick={handleClearHistory}>
              🗑 Clear Memory
            </button>
            <button
              className={`btn-ghost${traceOpen ? " active" : ""}`}
              onClick={() => setTraceOpen(v => !v)}
            >
              🧠 Agent Trace
            </button>
          </div>
        </div>

        <div className="content-area">

          {showAnalytics && (
            <section className="analytics-panel" aria-live="polite">
              <div className="trace-header">
                📊 Usage Analytics
                <button className="trace-close" onClick={() => setShowAnalytics(false)}>✕</button>
              </div>
              {analyticsError ? (
                <p className="analytics-empty">{analyticsError}</p>
              ) : analyticsData ? (
                <div className="analytics-content">
                  <div className="analytics-stat">
                    <strong>{analyticsData.total_queries}</strong>
                    <span>Total questions</span>
                  </div>
                  <div className="analytics-stat">
                    <strong>{analyticsData.avg_token_count ?? "—"}</strong>
                    <span>Estimated answer tokens</span>
                  </div>
                  <div className="analytics-stat">
                    <strong>{analyticsData.avg_answer_score ?? "—"}</strong>
                    <span>Average citation/relevance signal / 100 (heuristic)</span>
                  </div>
                  <div className="analytics-stat">
                    <strong>{analyticsData.avg_prompt_score ?? "—"}</strong>
                    <span>Average prompt-clarity score / 100</span>
                  </div>
                  <h3>Top intents</h3>
                  {analyticsData.top_intents?.length ? analyticsData.top_intents.map(([intent, count]) => (
                    <div className="analytics-intent" key={intent}>
                      <span>{intent.replaceAll("_", " ")}</span>
                      <strong>{count}</strong>
                    </div>
                  )) : <p className="analytics-empty">No classified queries yet.</p>}
                  <h3>Evidence-score distribution</h3>
                  <div className="analytics-intent">
                    <span>High</span><strong>{analyticsData.quality_breakdown?.high ?? 0}</strong>
                  </div>
                  <div className="analytics-intent">
                    <span>Moderate</span><strong>{analyticsData.quality_breakdown?.moderate ?? 0}</strong>
                  </div>
                  <div className="analytics-intent">
                    <span>Low</span><strong>{analyticsData.quality_breakdown?.low ?? 0}</strong>
                  </div>
                </div>
              ) : <p className="analytics-empty">Loading analytics…</p>}
            </section>
          )}

          {/* Agent Trace Panel */}
          {traceOpen && (
            <div className="trace-panel">
              <div className="trace-header">
                🧠 Agent Reasoning Trace
                <button className="trace-close" onClick={() => setTraceOpen(false)}>✕</button>
              </div>
              <div className="trace-body">
                {activeTrace.length === 0 ? (
                  <div className="trace-empty">
                    Agent steps will appear here as the system reasons through your question.
                  </div>
                ) : (
                  activeTrace.map((step, i) => (
                    <div key={i} className={`trace-step trace-${step.type}`}>
                      <span className="trace-icon">{STEP_ICONS[step.type] || "•"}</span>
                      <div className="trace-content">
                        <span className="trace-type">{step.type.toUpperCase()}</span>
                        <div className="trace-text">{step.content}</div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* Chat Messages */}
          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="welcome-screen">
                <span className="welcome-icon">🏥</span>
                <h3>Welcome to MedAssist-AI</h3>
                <p>
                  An agentic AI system that reasons step-by-step through medical documents
                  using ReAct agents, hybrid retrieval, and transparent prompt and evidence scoring.
                </p>
                <div className="sample-questions">
                  {samples.map((sq, i) => (
                    <button key={i} className="sample-btn" onClick={() => askQuestion(sq)}>
                      {sq}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map(msg => (
                <div key={msg.id} className={`msg-row ${msg.role}`}>
                  <div className="msg-avatar">
                    {msg.role === "user" ? "👤" : "🩺"}
                  </div>
                  <div className="msg-bubble">
                    {msg.role === "user" ? (
                      <p className="user-text">{msg.text}</p>
                    ) : (
                      <>
                        {msg.text ? renderAnswer(msg.text) : (
                          <div className="skeleton">
                            <div className="skel-line long" />
                            <div className="skel-line medium" />
                            <div className="skel-line short" />
                          </div>
                        )}

                        {/* Quality Badge */}
                        {msg.quality && (
                          <div>
                            <div className={`quality-badge quality-${msg.quality.toLowerCase()}`}>
                              📊 Citation/relevance signal: {msg.qualityMetrics?.answer_quality?.score ?? "—"}/100 · {msg.quality}
                              {msg.intent && <span> · Intent: {msg.intent.replace("_", " ")}</span>}
                            </div>
                            <p className="quality-signal-warning">
                              Heuristic only—not a fact-check or answer-accuracy score.
                            </p>
                          </div>
                        )}
                        {msg.qualityMetrics && (
                          <details className="quality-details">
                            <summary>
                              Why this score? · Prompt clarity {msg.qualityMetrics.prompt_quality.score}/100
                            </summary>
                            <div className="quality-detail-grid">
                              {[["Prompt clarity", msg.qualityMetrics.prompt_quality],
                                ["Citation/relevance signals (heuristic)", msg.qualityMetrics.answer_quality]].map(([title, result]) => (
                                <section className="quality-detail-card" key={title}>
                                  <h4>{title}: {result.score}/100 ({result.level})</h4>
                                  {Object.entries(result.metrics).map(([key, value]) => (
                                    <div className="quality-metric" key={key}>
                                      <div className="quality-metric-label">
                                        <span>{QUALITY_METRIC_LABELS[key] || key.replaceAll("_", " ")}</span>
                                        <span>{value}%{result.weights?.[key] ? ` · weight ${result.weights[key]}%` : ""}</span>
                                      </div>
                                      <div
                                        className="quality-meter"
                                        role="meter"
                                        aria-label={QUALITY_METRIC_LABELS[key] || key}
                                        aria-valuemin={0}
                                        aria-valuemax={100}
                                        aria-valuenow={value}
                                      >
                                        <span style={{ width: `${value}%` }} />
                                      </div>
                                    </div>
                                  ))}
                                  {result.suggestions?.length > 0 && (
                                    <ul className="quality-suggestions">
                                      {result.suggestions.map((suggestion, index) => (
                                        <li key={index}>{suggestion}</li>
                                      ))}
                                    </ul>
                                  )}
                                  <p className="quality-method-note">{result.note}</p>
                                </section>
                              ))}
                            </div>
                            <p className="quality-thresholds">
                              Prompt clarity: High ≥75, Moderate ≥45, Low &lt;45.
                              Citation/relevance signals: High ≥80, Moderate ≥60, Low &lt;60.
                              Prompt clarity is scored separately. These thresholds do not test whether source passages support claims or verify answer accuracy.
                            </p>
                          </details>
                        )}

                        {/* Sources */}
                        {msg.sources?.length > 0 && (
                          <div className="sources-box">
                            <div className="sources-heading">
                              <div className="sources-title">📚 Sources:</div>
                              <button className="source-export-btn" onClick={() => exportCitations(msg.sources)}>
                                Export BibTeX
                              </button>
                            </div>
                            <div className="sources-list">
                              {msg.sources.map((s, i) => (
                                <span key={i} className="source-chip">{s}</span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Show trace for this message */}
                        {msg.trace?.length > 0 && (
                          <button
                            className="show-trace-btn"
                            onClick={() => { setActiveTrace(msg.trace); setTraceOpen(true); }}
                          >
                            🧠 View Agent Reasoning ({msg.trace.length} steps)
                          </button>
                        )}
                      </>
                    )}
                    <div className="msg-meta">
                      {new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </div>
                  </div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

        </div>

        {/* Input Area */}
        <div className="input-area">
          <form onSubmit={e => { e.preventDefault(); askQuestion(); }}>
            <div className="input-wrapper">
              <input
                className="chat-input"
                type="text"
                placeholder="Ask a medical question or describe symptoms..."
                value={question}
                onChange={e => setQuestion(e.target.value)}
                disabled={isGenerating}
              />
              <button
                className="send-btn"
                type="submit"
                disabled={isGenerating || !question.trim()}
              >
                {isGenerating ? <span className="spinner sm" /> : "➔"}
              </button>
            </div>
          </form>
          <div className="disclaimer">
            ⚠️ For research assistance only. Not a substitute for professional medical advice.
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;

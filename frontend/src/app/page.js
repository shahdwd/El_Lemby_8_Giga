"use client";

import { useState, useEffect, useRef } from "react";

export default function Home() {
  const [language, setLanguage] = useState("ar"); // "ar" | "en"
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [sessions, setSessions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedDoc, setUploadedDoc] = useState(null); // { name, size }
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [expandedCitations, setExpandedCitations] = useState({}); // { msgIdx: bool }
  const messagesEndRef = useRef(null);

  // Localization keys
  const t = {
    ar: {
      title: "قانوني — مساعد القانون المصري",
      subtitle: "مستشار ذكي معزز ببحث متجهي ورسم بياني ونظام تدقيق الاقتباسات الفوري.",
      disclaimer: "تنبيه: هذا المساعد يوفر معلومات قانونية استرشادية فقط بناءً على النصوص المسترجعة، ولا يعتبر استشارة قانونية رسمية.",
      placeholder: "اكتب سؤالك القانوني هنا (مثال: ما هي عقوبة السرقة؟)...",
      send: "إرسال",
      newChat: "محادثة جديدة",
      confidence: "مستوى الثقة",
      citations: "المراجع والاقتباسات القانونية",
      graphPath: "مسار التحليل (Neo4j Graph)",
      verificationOutcome: "فحص Grounding",
      emptyStateTitle: "كيف يمكنني مساعدتك اليوم؟",
      emptyStateSubtitle: "اسأل عن أي مادة في القانون المدني، العقوبات، أو القوانين المصرية الأخرى.",
      uploadTitle: "تحميل وثيقة قانونية (Live Ingestion)",
      uploadSubtitle: "اسحب ملف PDF أو انقر لتحميله للتحليل والتلخيص",
      high: "عالي",
      medium: "متوسط",
      low: "منخفض",
      passed: "موثق بالكامل",
      patched: "تم تصحيح المراجع",
      retried: "أعيد توليده بموثوقية",
      fallback: "مسترجع فقط (تعذر الدمج الموثوق)",
      session: "جلسة",
      delete: "حذف",
      noCitations: "لا توجد اقتباسات مستخدمة في الرد.",
      confidenceTooltip: "يتم احتساب ثقة النظام بناءً على درجة تشابه البحث المتجهي وعدد تطابقات الرسم البياني ونتائج التحقق من الاقتباسات.",
    },
    en: {
      title: "Qanony — Egyptian Law Assistant",
      subtitle: "Intelligent assistant enhanced with Vector-Graph RAG and deterministic citation safety checks.",
      disclaimer: "Disclaimer: This assistant provides general informational guidance based on retrieved texts. It does not constitute formal legal advice.",
      placeholder: "Type your legal query here (e.g., What is the penalty for theft?)...",
      send: "Send",
      newChat: "New Chat",
      confidence: "Confidence Level",
      citations: "Legal References & Citations",
      graphPath: "Explainability Path (Neo4j Graph)",
      verificationOutcome: "Grounding Verification",
      emptyStateTitle: "How can I help you today?",
      emptyStateSubtitle: "Ask about any article in the Civil Code, Penal Code, or other Egyptian legislations.",
      uploadTitle: "Upload Legal Document (Live Ingestion)",
      uploadSubtitle: "Drag & drop PDF or click to upload for summary & explainability",
      high: "High",
      medium: "Medium",
      low: "Low",
      passed: "Fully Grounded",
      patched: "Citations Patched",
      retried: "Regenerated Securely",
      fallback: "Raw Retrieval Fallback",
      session: "Session",
      delete: "Delete",
      noCitations: "No citations cited in this response.",
      confidenceTooltip: "System confidence is calculated dynamically using vector similarity scores, graph overlap, and citation verification outcomes.",
    }
  };

  const activeTranslation = t[language];

  // API Backend Base URL
  const API_BASE = "http://localhost:8000";

  // Initialize new session
  const startNewChat = () => {
    setMessages([]);
    setUploadedDoc(null);
    const newSessionId = `session-${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);

    // Add to local session history
    setSessions(prev => {
      if (prev.includes(newSessionId)) return prev;
      return [newSessionId, ...prev];
    });
  };

  useEffect(() => {
    startNewChat();
  }, []);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Handle Form Submit
  const handleSend = async (e) => {
    if (e) e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue;
    setInputValue("");
    setIsLoading(true);

    // Append User message
    const msgId = messages.length;
    setMessages(prev => [...prev, { role: "user", content: userMessage }]);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId,
          language: language,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to communicate with API server");
      }

      const data = await response.json();

      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          citations: data.citations || [],
          graphPath: data.graph_path || [],
          confidence: data.confidence || "medium",
          outcome: data.citation_check_outcome || "passed",
        }
      ]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: language === "ar"
            ? "عذراً، فشل الاتصال بخادم الخدمة. يرجى التأكد من تشغيل خادم FastAPI على المنفذ 8000."
            : "Sorry, failed to connect to API server. Please ensure the FastAPI server is running on port 8000.",
          error: true
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const deleteSession = (sessId, e) => {
    e.stopPropagation();
    setSessions(prev => prev.filter(s => s !== sessId));
    if (sessionId === sessId) {
      startNewChat();
    }
  };

  const toggleCitation = (idx) => {
    setExpandedCitations(prev => ({
      ...prev,
      [idx]: !prev[idx]
    }));
  };

  const triggerUpload = () => {
    // Mock live ingestion file upload
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".pdf,.txt,.md";
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        setUploadedDoc({ name: file.name, size: (file.size / 1024).toFixed(1) });
        // Add fake system message for upload demo
        setMessages(prev => [
          ...prev,
          {
            role: "assistant",
            system: true,
            content: language === "ar"
              ? `تم تحميل الملف "${file.name}" بنجاح وجاري إدراجه وتلخيصه كمرجع إضافي للمحادثة.`
              : `File "${file.name}" uploaded successfully and pinned as context for this session.`
          }
        ]);
      }
    };
    input.click();
  };

  return (
    <div className={`app-container ${language === "ar" ? "rtl" : "ltr"}`}>
      {/* Sidebar (Session management & Upload) */}
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="sidebar-header">
          <button className="btn btn-primary new-chat-btn" onClick={startNewChat}>
            <span>➕</span>
            {activeTranslation.newChat}
          </button>
        </div>

        <div className="sidebar-content">
          {/* Active Document Upload box */}
          <div style={{ marginBottom: "20px" }}>
            <h3 style={{ fontSize: "0.8rem", marginBottom: "8px", color: "var(--text-secondary)" }}>
              {activeTranslation.uploadTitle}
            </h3>
            {uploadedDoc ? (
              <div className="uploaded-doc-badge">
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "80%" }}>
                  📄 {uploadedDoc.name} ({uploadedDoc.size} KB)
                </span>
                <button
                  onClick={() => setUploadedDoc(null)}
                  style={{ background: "transparent", border: "none", color: "var(--error)", cursor: "pointer" }}
                >
                  ✕
                </button>
              </div>
            ) : (
              <div className="upload-box" onClick={triggerUpload}>
                <div style={{ fontSize: "1.5rem" }}>📤</div>
                <div className="upload-box-text">{activeTranslation.uploadSubtitle}</div>
              </div>
            )}
          </div>

          <h3 style={{ fontSize: "0.8rem", marginBottom: "8px", color: "var(--text-secondary)" }}>
            {language === "ar" ? "الجلسات النشطة" : "Active Sessions"}
          </h3>
          {sessions.map((sess) => (
            <div
              key={sess}
              className={`session-item ${sess === sessionId ? "active" : ""}`}
              onClick={() => {
                setSessionId(sess);
                setMessages([]);
              }}
            >
              <span className="session-title">
                💬 {activeTranslation.session} ({sess.replace("session-", "")})
              </span>
              <button className="session-delete" onClick={(e) => deleteSession(sess, e)}>
                🗑️
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* Main chat window */}
      <main className="chat-main">
        {/* Header */}
        <header className="app-header">
          <div className="brand">
            <div className="brand-logo">⚖️</div>
            <div>
              <h1 className="brand-title">{activeTranslation.title}</h1>
              <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                {activeTranslation.subtitle}
              </p>
            </div>
          </div>

          <div className="header-actions">
            {/* Language toggle switch */}
            <div
              className="toggle-lang"
              onClick={() => setLanguage(language === "ar" ? "en" : "ar")}
            >
              <div className={`toggle-lang-item ${language === "ar" ? "active" : ""}`}>العربية</div>
              <div className={`toggle-lang-item ${language === "en" ? "active" : ""}`}>English</div>
            </div>
          </div>
        </header>

        {/* Legal Disclaimer */}
        <div className="disclaimer-banner">
          <span>⚠️</span>
          <span>{activeTranslation.disclaimer}</span>
        </div>

        {/* Conversation Area */}
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">⚖️</div>
              <h2 className="empty-state-title">{activeTranslation.emptyStateTitle}</h2>
              <p className="empty-state-subtitle">{activeTranslation.emptyStateSubtitle}</p>
            </div>
          ) : (
            messages.map((msg, index) => {
              if (msg.system) {
                return (
                  <div key={index} style={{ textAlign: "center", margin: "10px 0", color: "var(--accent-primary)", fontSize: "0.85rem" }}>
                    ℹ️ {msg.content}
                  </div>
                );
              }
              return (
                <div key={index} className={`message-wrapper ${msg.role}`}>
                  <div className="message-meta" style={{ marginBottom: "6px", display: "block" }}>
                    <strong>{msg.role === "user" ? (language === "ar" ? "أنت" : "You") : (language === "ar" ? "المساعد القانوني" : "Legal Assistant")}</strong>
                  </div>
                  <div className="message-bubble">
                    <div style={{ whiteSpace: "pre-wrap" }}>{msg.content}</div>

                    {/* Assistant metadata indicators */}
                    {msg.role === "assistant" && !msg.error && (
                      <div style={{ marginTop: "16px", display: "flex", flexDirection: "column", gap: "12px" }}>

                        {/* Badges line */}
                        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                          {/* Confidence level badge */}
                          <span
                            className={`badge badge-confidence-${msg.confidence}`}
                            title={activeTranslation.confidenceTooltip}
                          >
                            🛡️ {activeTranslation.confidence}: {activeTranslation[msg.confidence]}
                          </span>

                          {/* Grounding outcome badge */}
                          <span
                            className={`badge badge-outcome-${msg.outcome}`}
                          >
                            ✅ {activeTranslation.verificationOutcome}: {activeTranslation[msg.outcome]}
                          </span>
                        </div>

                        {/* Explainability path (Neo4j graph nodes) */}
                        {msg.graphPath && msg.graphPath.length > 0 && (
                          <div className="graph-path-box">
                            <div className="graph-path-title">
                              <span>🔗</span>
                              {activeTranslation.graphPath}
                            </div>
                            <div className="graph-nodes-container">
                              {msg.graphPath.map((node, nIdx) => (
                                <div key={nIdx} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                  <span className={`graph-node ${node.label.toLowerCase()}`}>
                                    {node.label === "Law" ? "📜" : "📌"} {node.name}
                                  </span>
                                  {nIdx < msg.graphPath.length - 1 && (
                                    <span className="graph-arrow">
                                      {language === "ar" ? "◀" : "▶"}
                                    </span>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Collapsible citations list */}
                        <div className="citations-box">
                          <div
                            className="citations-header"
                            onClick={() => toggleCitation(index)}
                          >
                            <span>📚 {activeTranslation.citations}</span>
                            <span>{expandedCitations[index] ? "▲" : "▼"}</span>
                          </div>

                          {expandedCitations[index] && (
                            <div className="citations-list">
                              {msg.citations.length === 0 ? (
                                <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
                                  {activeTranslation.noCitations}
                                </div>
                              ) : (
                                msg.citations.map((cit, cIdx) => (
                                  <div key={cIdx} className="citation-item">
                                    <div className="citation-title">
                                      📜 {cit.law_name} — المادة {cit.article_id}
                                    </div>
                                    {cit.text_snippet && (
                                      <p className="citation-snippet">
                                        &ldquo;{cit.text_snippet}&rdquo;
                                      </p>
                                    )}
                                  </div>
                                ))
                              )}
                            </div>
                          )}
                        </div>

                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}
          {isLoading && (
            <div className="message-wrapper assistant">
              <div className="message-bubble" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontSize: "1.2rem", animation: "spin 1s linear infinite" }}>⏳</span>
                <span>{language === "ar" ? "جاري التفكير والتحقق من القوانين..." : "Thinking & verifying laws..."}</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input box */}
        <div className="input-panel">
          <form onSubmit={handleSend}>
            <div className="input-container">
              <textarea
                className="chat-textarea"
                rows="1"
                placeholder={activeTranslation.placeholder}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
              />
              <button
                type="submit"
                className="btn btn-primary"
                disabled={isLoading}
              >
                {activeTranslation.send}
              </button>
            </div>
          </form>
        </div>
      </main>

      <style jsx global>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

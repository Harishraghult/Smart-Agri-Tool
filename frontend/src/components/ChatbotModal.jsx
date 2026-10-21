import React, { useState, useRef, useEffect } from 'react';
import { errorMessage } from '../api';
import { Send, Bot, User, Sparkles } from 'lucide-react';

const SUGGESTIONS = [
  'How to treat Tomato Early Blight?',
  'What is the ideal NPK ratio for Rice?',
  'How to control aphids and spider mites organically?',
  'What weather conditions trigger Late Blight?'
];

// Render **bold** / *italic* markers from the bot's markdown-style replies
function renderInline(text) {
  return text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g).map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith('*') && part.endsWith('*') && part.length > 2) return <em key={i}>{part.slice(1, -1)}</em>;
    return part;
  });
}

export default function ChatbotModal() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I am your **AgriPulse AI Assistant**. Ask me anything about crop diseases, pest remedies, fertilizer schedules, or weather protection!'
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (textToSend) => {
    const userMsg = (textToSend || input).trim();
    if (!userMsg || loading) return;
    setInput('');

    setMessages((prev) => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);

    try {
      const response = await fetch('/api/v1/chat/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: 'user_session_1',
          message: userMsg,
          history: messages
        })
      });
      if (!response.ok) throw new Error(`Backend error: ${response.status} ${response.statusText}`);
      const data = await response.json();
      setMessages((prev) => [...prev, { role: 'assistant', content: data.bot_response }]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [...prev, { role: 'assistant', content: `⚠️ ${errorMessage(err)}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel" style={{ height: '700px', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{ padding: '20px', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'color-mix(in srgb, var(--accent-emerald) 20%, transparent)', color: 'var(--accent-emerald)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Bot size={22} />
        </div>
        <div>
          <h3 style={{ fontSize: '1.1rem', color: 'var(--text-primary)' }}>AI Agricultural Advisory Chatbot</h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Powered by Agricultural Pathology RAG Engine</p>
        </div>
      </div>

      {/* Suggestion Chips */}
      <div style={{ padding: '10px 20px', background: 'var(--surface-inset)', borderBottom: '1px solid var(--border-color)', display: 'flex', gap: '8px', overflowX: 'auto' }}>
        <span style={{ fontSize: '0.75rem', color: 'var(--accent-emerald)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px', shrink: 0 }}>
          <Sparkles size={14} /> Quick Questions:
        </span>
        {SUGGESTIONS.map((sug, i) => (
          <button
            key={i}
            className="btn-secondary"
            style={{ fontSize: '0.75rem', padding: '4px 10px', borderRadius: '999px', shrink: 0, whiteSpace: 'nowrap' }}
            onClick={() => handleSend(sug)}
          >
            {sug}
          </button>
        ))}
      </div>

      {/* Messages */}
      <div style={{ flex: 1, padding: '20px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              display: 'flex',
              gap: '12px',
              alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '80%'
            }}
          >
            {msg.role === 'assistant' && (
              <div style={{ width: '30px', height: '30px', borderRadius: '50%', background: 'var(--accent-emerald)', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', shrink: 0 }}>
                <Bot size={16} />
              </div>
            )}

            <div
              style={{
                background: msg.role === 'user' ? 'linear-gradient(135deg, var(--accent-emerald), var(--accent-emerald-dark))' : 'var(--surface-subtle-2)',
                border: msg.role === 'assistant' ? '1px solid var(--border-color)' : 'none',
                color: msg.role === 'user' ? '#ffffff' : 'var(--text-primary)',
                padding: '14px 18px',
                borderRadius: '16px',
                fontSize: '0.9rem',
                lineHeight: '1.5',
                whiteSpace: 'pre-line'
              }}
            >
              {renderInline(msg.content)}
            </div>

            {msg.role === 'user' && (
              <div style={{ width: '30px', height: '30px', borderRadius: '50%', background: 'var(--accent-blue)', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', shrink: 0 }}>
                <User size={16} />
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div style={{ alignSelf: 'flex-start', color: 'var(--text-secondary)', fontSize: '0.85rem', fontStyle: 'italic', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Sparkles size={16} className="pulse-dot" /> Advisory Assistant is formulating answer...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div style={{ padding: '16px 20px', borderTop: '1px solid var(--border-color)', display: 'flex', gap: '12px' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask a question about crop disease, pest control, or fertilizers..."
          className="form-input"
          style={{ flex: 1 }}
        />
        <button className="btn-primary" onClick={() => handleSend()} disabled={loading}>
          <Send size={18} /> Send
        </button>
      </div>
    </div>
  );
}

"use client";

import { useEffect, useState, useRef, use } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Square, Plus, FileText, X, AlertCircle } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

export default function ChatPage({ params }: { params: Promise<{ id: string }> }) {
  const resolvedParams = use(params);
  const chatId = resolvedParams.id;
  const [chat, setChat] = useState<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
  const [messages, setMessages] = useState<any[]>([]); // eslint-disable-line @typescript-eslint/no-explicit-any
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [agentState, setAgentState] = useState<string | null>(null);
  const [attachedDoc, setAttachedDoc] = useState<{ filename: string; docId?: string } | null>(null);
  const [uploadingDoc, setUploadingDoc] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isSendingRef = useRef(false);
  const evtSourceRef = useRef<EventSource | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  useEffect(() => {
    return () => {
      if (evtSourceRef.current) {
        evtSourceRef.current.close();
        evtSourceRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const loadChat = async () => {
      try {
        const data = await api.getChat(chatId);
        setChat(data);
        setMessages(data.messages || []);
      } catch (e: unknown) {
        if (e instanceof Error && 'status' in e && (e as {status?: number}).status === 404) router.push("/");
      } finally {
        setLoading(false);
      }
    };
    loadChat();
  }, [chatId, router]);

  const loadChatManual = async () => {
    try {
      const data = await api.getChat(chatId);
      setChat(data);
      setMessages(data.messages || []);
    } catch {
      // ignoring error for manual refresh
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF files (.pdf) are supported for RAG.");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    setUploadingDoc(true);
    setUploadError(null);

    try {
      const res = await api.uploadChatDocument(chatId, file);
      setAttachedDoc({
        filename: res.filename || file.name,
        docId: res.doc_id,
      });
      setUploadError(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to upload and index PDF for RAG";
      setUploadError(msg);
    } finally {
      setUploadingDoc(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleSend = async () => {
    if ((!input.trim() && !attachedDoc) || isSendingRef.current || uploadingDoc) return;
    const content = input.trim() || (attachedDoc ? `Please summarize and analyze this attached document: ${attachedDoc.filename}` : "");
    const docToAttach = attachedDoc;

    setInput("");
    setAttachedDoc(null);
    setUploadError(null);
    setSending(true);
    isSendingRef.current = true;
    
    try {
      const attachedDocs = docToAttach ? [docToAttach.filename] : [];
      const resp = await api.sendMessage(chatId, content, attachedDocs);
      await loadChatManual(); // load user message immediately
      setAgentState("Thinking...");
      
      if (resp && resp.job_id) {
        // connect to SSE
        if (evtSourceRef.current) {
          evtSourceRef.current.close();
        }
        const evtSource = new EventSource(`${process.env.NEXT_PUBLIC_API_URL || ""}/api/v1/jobs/${resp.job_id}/events/stream`, {
          withCredentials: true
        });
        evtSourceRef.current = evtSource;
        
        const updateState = (e: MessageEvent) => {
          try {
            const parsed = JSON.parse(e.data);
            if (e.type === "PLAN_CREATED" || e.type === "PLANNING") setAgentState("Planning...");
            else if (e.type === "TOOL_CALL" || e.type === "ACTION") {
              const payload = parsed.payload;
              if (payload && (payload.task_type === "rag_search" || (payload.output && payload.output.toLowerCase().includes("rag")))) {
                setAgentState("Retrieving from uploaded documents (RAG)...");
              } else if (payload && payload.output && payload.output.includes("search")) {
                setAgentState("Searching the web...");
              } else {
                setAgentState("Calling tools...");
              }
            } else if (e.type === "GENERATING") setAgentState("Generating answer...");
            else if (e.type === "JOB_STARTED") setAgentState("Agent started...");
          } catch {}
        };
        
        evtSource.onmessage = updateState;
        
        const eventTypes = ["PLAN_CREATED", "PLANNING", "TOOL_CALL", "ACTION", "GENERATING", "JOB_STARTED"];
        eventTypes.forEach(type => evtSource.addEventListener(type, updateState));
        
        let pollTimer: NodeJS.Timeout | null = null;

        const cleanup = async () => {
          if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
          }
          if (evtSourceRef.current) {
            evtSourceRef.current.close();
            evtSourceRef.current = null;
          }
          await loadChatManual();
          setSending(false);
          setAgentState(null);
          isSendingRef.current = false;
        };

        // Fallback polling: every 1.5s while sending
        pollTimer = setInterval(async () => {
          try {
            const data = await api.getChat(chatId);
            if (data.messages && data.messages.length > 0) {
              const lastMsg = data.messages[data.messages.length - 1];
              if (lastMsg.role === "assistant" && lastMsg.job_id === resp.job_id) {
                setChat(data);
                setMessages(data.messages);
                await cleanup();
              }
            }
          } catch {}
        }, 1500);

        evtSource.addEventListener("JOB_COMPLETED", async (e: MessageEvent) => {
          try {
            const parsed = JSON.parse(e.data);
            if (parsed.payload && parsed.payload.output) {
              setMessages(prev => {
                if (prev.some(m => m.content === parsed.payload.output)) return prev;
                return [
                  ...prev,
                  {
                    message_id: "optimistic-" + Date.now(),
                    chat_id: chatId,
                    role: "assistant",
                    content: parsed.payload.output,
                    job_id: resp.job_id,
                    created_at: new Date().toISOString()
                  }
                ];
              });
            }
          } catch (err) {}
          await cleanup();
        });

        evtSource.addEventListener("JOB_FAILED", async (e: MessageEvent) => {
          try {
            const parsed = JSON.parse(e.data);
            const errorMsg = parsed.payload?.error || "Agent failed to generate a response. Please try again.";
            setMessages(prev => [
              ...prev,
              {
                message_id: "optimistic-err-" + Date.now(),
                chat_id: chatId,
                role: "assistant",
                content: "Error: " + errorMsg,
                job_id: resp.job_id,
                created_at: new Date().toISOString()
              }
            ]);
          } catch (err) {}
          await cleanup();
        });

        evtSource.addEventListener("JOB_CANCELLED", async () => {
          await cleanup();
        });

        evtSource.onerror = async () => {
          await cleanup();
        };
        
        return; // don't set sending false yet
      }
    } catch {
      alert("Failed to send message");
    }
    
    setSending(false);
    setAgentState(null);
    isSendingRef.current = false;
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (loading) return <div className="flex-1 flex items-center justify-center">Loading...</div>;
  if (!chat) return null;

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-black relative">
      {/* No top header bar — "Research Agent" label and Share/More removed */}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 md:px-12 pt-8 pb-8 flex justify-center">
        <div className="w-full max-w-3xl flex flex-col">
          
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">{chat.title || "New Chat"}</h1>
            <div className="text-sm text-gray-500 mb-4">
              Updated {new Date(chat.updated_at).toLocaleDateString()} · {messages.length} messages
            </div>
            {messages.length === 0 && (
              <div className="mt-8 border border-gray-800 rounded-2xl p-8 text-center text-gray-400 bg-[#0A0A0A]">
                No messages yet. Say hello to start.
              </div>
            )}
          </div>

          <div className="space-y-6">
            {messages.map(msg => (
              <div key={msg.message_id} className={`flex gap-4 ${msg.role === "assistant" ? "flex-row-reverse" : ""}`}>
                {msg.role === "user" && (
                  <div className="flex-shrink-0 mt-1">
                    <div className="w-8 h-8 rounded-full bg-[#1A1A1A] border border-gray-700 flex items-center justify-center text-sm font-bold text-white">
                      U
                    </div>
                  </div>
                )}
                <div className={`flex-1 min-w-0 ${msg.role === "assistant" ? "text-right" : "text-left"}`}>
                  <div className="text-sm font-medium mb-1 text-gray-400 capitalize">{msg.role}</div>
                  {(() => {
                    const match = msg.role === "user" ? msg.content.match(/^\[Attached Document:\s*(.*?)\]\n\n([\s\S]*)$/) : null;
                    const attachedDocName = match ? match[1] : null;
                    const displayContent = match ? match[2] : msg.content;
                    return (
                      <div 
                        className={`leading-relaxed inline-block text-left max-w-[90%] overflow-hidden ${
                          msg.role === "assistant" 
                            ? "text-gray-200" 
                            : "bg-blue-600 text-white px-5 py-4 rounded-3xl shadow-sm rounded-tl-sm"
                        }`}
                      >
                        {attachedDocName && (
                          <div className="flex items-center gap-1.5 mb-2.5 px-3 py-1.5 bg-black/25 border border-white/20 rounded-xl text-xs text-blue-100 w-fit">
                            <FileText className="w-3.5 h-3.5 text-blue-200 flex-shrink-0" />
                            <span className="font-medium truncate max-w-[220px]">{attachedDocName}</span>
                            <span className="bg-white/20 text-white text-[9px] px-1.5 py-0.5 rounded uppercase font-bold tracking-wider ml-1">
                              RAG
                            </span>
                          </div>
                        )}
                        <ReactMarkdown 
                          remarkPlugins={[remarkGfm, remarkMath]}
                          rehypePlugins={[rehypeKatex]}
                          components={{
                            table: ({node, ...props}) => <div className="overflow-x-auto"><table className="border-collapse border border-gray-700 my-4 w-full text-sm" {...props} /></div>,
                            th: ({node, ...props}) => <th className="border border-gray-600 bg-black/30 px-4 py-2 text-left" {...props} />,
                            td: ({node, ...props}) => <td className="border border-gray-700 px-4 py-2" {...props} />,
                            a: ({node, ...props}) => <a className="text-blue-400 hover:underline font-medium" target="_blank" rel="noopener noreferrer" {...props} />,
                            p: ({node, ...props}) => <p className="mb-3 last:mb-0 whitespace-pre-wrap" {...props} />,
                            ul: ({node, ...props}) => <ul className="list-disc pl-6 mb-3" {...props} />,
                            ol: ({node, ...props}) => <ol className="list-decimal pl-6 mb-3" {...props} />,
                            li: ({node, ...props}) => <li className="mb-1" {...props} />,
                            strong: ({node, ...props}) => <strong className="font-bold text-gray-100" {...props} />,
                            h1: ({node, ...props}) => <h1 className="text-2xl font-bold mb-3 mt-4 text-white" {...props} />,
                            h2: ({node, ...props}) => <h2 className="text-xl font-bold mb-3 mt-4 text-white" {...props} />,
                            h3: ({node, ...props}) => <h3 className="text-lg font-bold mb-2 mt-3 text-white" {...props} />,
                          }}
                        >
                          {displayContent.replace(/\\\[/g, "$$$$").replace(/\\\]/g, "$$$$").replace(/\\\(/g, "$$").replace(/\\\)/g, "$$")}
                        </ReactMarkdown>
                      </div>
                    );
                  })()}
                </div>
              </div>
            ))}

            {/* Agent status indicator (shown while agent is working) */}
            {agentState && (
              <div className="flex gap-4">
                <div className="flex-1 min-w-0 flex items-center h-8 text-left">
                  <div className="text-sm font-medium text-purple-400 animate-pulse">{agentState}</div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} className="h-4" />
          </div>
        </div>
      </div>

      {/* Input area */}
      <div className="flex-shrink-0 p-4 flex justify-center bg-black">
        <div className="w-full max-w-3xl relative">
          {/* Upload error message if any */}
          {uploadError && (
            <div className="mb-2 px-3 py-2 bg-red-950/80 border border-red-800/80 rounded-2xl text-xs text-red-200 flex items-center justify-between shadow-md">
              <div className="flex items-center gap-1.5">
                <AlertCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
                <span>{uploadError}</span>
              </div>
              <button onClick={() => setUploadError(null)} className="text-red-400 hover:text-white ml-2">
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          <div className="bg-[#1A1A1A] border border-gray-800 rounded-3xl p-3 flex flex-col shadow-lg">
            {/* Attached PDF indicator chip */}
            {attachedDoc && (
              <div className="flex items-center gap-2 mb-2 px-3 py-1.5 bg-blue-950/60 border border-blue-800/70 rounded-2xl text-xs text-blue-200 w-fit animate-fade-in">
                <FileText className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" />
                <span className="font-medium truncate max-w-[220px]">{attachedDoc.filename}</span>
                <span className="bg-blue-600/40 border border-blue-500/50 text-blue-300 text-[10px] px-1.5 py-0.5 rounded font-semibold uppercase tracking-wider">
                  RAG Auto-Enabled
                </span>
                <button
                  type="button"
                  onClick={() => setAttachedDoc(null)}
                  className="hover:text-white ml-1 text-gray-400"
                  title="Remove attachment"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            {/* Uploading indicator */}
            {uploadingDoc && (
              <div className="flex items-center gap-2 mb-2 px-3 py-1.5 bg-gray-900 border border-gray-700 rounded-2xl text-xs text-gray-300 w-fit animate-pulse">
                <div className="w-3.5 h-3.5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin flex-shrink-0" />
                <span>Indexing PDF for RAG...</span>
              </div>
            )}

            <div className="flex items-end w-full">
              {/* + Sign Button to upload PDF for RAG */}
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploadingDoc || sending}
                className="w-9 h-9 flex items-center justify-center text-gray-400 hover:text-white hover:bg-gray-800 rounded-full transition disabled:opacity-40 flex-shrink-0 mr-1.5 mb-0.5"
                title="Upload PDF for RAG (+)"
              >
                <Plus className="h-5 w-5" />
              </button>

              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                className="hidden"
                onChange={handleFileUpload}
              />

              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={attachedDoc ? `Ask a question about ${attachedDoc.filename}...` : "How can I help you today?"}
                className="flex-1 bg-transparent text-white placeholder-gray-500 outline-none resize-none max-h-48 py-2 px-2 overflow-y-auto min-h-[44px]"
                rows={1}
                style={{ minHeight: "44px" }}
              />

              <div className="flex items-center flex-shrink-0 mb-1 ml-2">
                {/* Send / Stop button */}
                <button 
                  onClick={handleSend}
                  disabled={!sending && (!input.trim() && !attachedDoc)}
                  className="w-9 h-9 flex items-center justify-center bg-white text-black rounded-full hover:bg-gray-200 transition disabled:opacity-40"
                  title={sending ? "Agent is answering..." : "Send"}
                >
                  {sending ? (
                    <Square className="h-4 w-4 fill-black" />
                  ) : (
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="currentColor"
                      className="h-5 w-5"
                    >
                      <path d="M12 4l8 8h-6v8h-4v-8H4l8-8z" />
                    </svg>
                  )}
                </button>
              </div>
            </div>
          </div>
          <div className="text-center text-xs text-gray-500 mt-3 mb-1">
            AI can make mistakes. Check important info.
          </div>
        </div>
      </div>
    </div>
  );
}
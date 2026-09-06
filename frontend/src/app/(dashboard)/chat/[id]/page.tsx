"use client";

import { useEffect, useState, useRef, use } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Send, Mic, Plus, Bot, Share, MoreHorizontal } from "lucide-react";
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

  const handleSend = async () => {
    if (!input.trim() || isSendingRef.current) return;
    const content = input.trim();
    setInput("");
    setSending(true);
    isSendingRef.current = true;
    
    try {
      const resp = await api.sendMessage(chatId, content);
      await loadChatManual(); // load user message immediately
      setAgentState("Receiving request...");
      
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
              if (payload && payload.output && payload.output.includes("search")) {
                setAgentState("Searching/retrieving...");
              } else {
                setAgentState("Calling tools...");
              }
            } else if (e.type === "GENERATING") setAgentState("Generating answer...");
            else if (e.type === "JOB_STARTED") setAgentState("Agent started...");
          } catch {}
        };
        
        evtSource.onmessage = updateState;
        
        // The backend uses 'event: EVENT_TYPE' which translates to named events in EventSource
        // We should add generic listeners for the event types backend emits.
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

        // Fallback polling: every 1.5s while sending, poll chat messages in case SSE dropped or finished instantly
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
      <div className="h-14 flex items-center justify-between px-4 flex-shrink-0">
        <div className="flex items-center space-x-2">
          <div className="bg-[#1A1A1A] border border-gray-800 rounded-full px-3 py-1.5 flex items-center space-x-2 text-sm cursor-pointer hover:bg-gray-800 transition">
            <Bot className="h-4 w-4 text-purple-400" />
            <span className="font-medium">Research Agent</span>
          </div>
        </div>
        <div className="flex items-center space-x-2 text-gray-400">
          <button className="p-2 hover:text-white transition"><Share className="h-4 w-4" /></button>
          <button className="p-2 hover:text-white transition"><MoreHorizontal className="h-4 w-4" /></button>
        </div>
      </div>

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
                  <div 
                    className={`leading-relaxed inline-block text-left max-w-[90%] overflow-hidden ${
                      msg.role === "assistant" 
                        ? "text-gray-200" 
                        : "bg-blue-600 text-white px-5 py-4 rounded-3xl shadow-sm rounded-tl-sm"
                    }`}
                  >
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
                      {msg.content.replace(/\\\[/g, "$$$$").replace(/\\\]/g, "$$$$").replace(/\\\(/g, "$$").replace(/\\\)/g, "$$")}
                    </ReactMarkdown>
                  </div>
                </div>
              </div>
            ))}
            {agentState && (
              <div className="flex gap-4 flex-row-reverse">
                <div className="flex-1 min-w-0 flex items-center h-8 justify-end text-right">
                  <div className="text-sm font-medium text-purple-400 animate-pulse">{agentState}</div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} className="h-4" />
          </div>
        </div>
      </div>

      <div className="flex-shrink-0 p-4 flex justify-center bg-black">
        <div className="w-full max-w-3xl relative">
          <div className="bg-[#1A1A1A] border border-gray-800 rounded-3xl p-3 flex items-end shadow-lg">
            <button className="p-2 text-gray-400 hover:text-white transition flex-shrink-0 mb-1">
              <Plus className="h-5 w-5" />
            </button>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="How can I help you today?"
              className="flex-1 bg-transparent text-white placeholder-gray-500 outline-none resize-none max-h-48 py-2 px-2 overflow-y-auto min-h-[44px]"
              rows={1}
              style={{ minHeight: "44px" }}
            />
            <div className="flex items-center space-x-2 flex-shrink-0 mb-1">
              <button className="p-2 text-gray-400 hover:text-white transition">
                <Mic className="h-5 w-5" />
              </button>
              <button 
                onClick={handleSend}
                disabled={!input.trim() || sending}
                className="p-2 bg-white text-black rounded-full hover:bg-gray-200 transition disabled:opacity-50">
                <Send className="h-5 w-5" />
              </button>
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
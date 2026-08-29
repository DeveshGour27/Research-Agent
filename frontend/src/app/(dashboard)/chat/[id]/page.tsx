"use client";

import { useEffect, useState, useRef, use } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Send, Mic, Plus, Bot, Share, MoreHorizontal } from "lucide-react";

export default function ChatPage({ params }: { params: Promise<{ id: string }> }) {
  const resolvedParams = use(params);
  const chatId = resolvedParams.id;
  const [chat, setChat] = useState<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
  const [messages, setMessages] = useState<any[]>([]); // eslint-disable-line @typescript-eslint/no-explicit-any
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  useEffect(() => {
    const loadChat = async () => {
      try {
        const data = await api.getChat(chatId);
        setChat(data);
        setMessages(data.messages || []);
      } catch (e: unknown) {
        if (e instanceof Error && 'status' in e && e.status === 404) router.push("/");
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
    if (!input.trim() || sending) return;
    const content = input.trim();
    setInput("");
    setSending(true);
    
    try {
      await api.sendMessage(chatId, content);
      await loadChatManual();
    } catch {
      alert("Failed to send message");
    } finally {
      setSending(false);
    }
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

      <div className="flex-1 overflow-y-auto px-4 md:px-12 pt-8 pb-32 flex justify-center">
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
              <div key={msg.message_id} className="flex gap-4">
                <div className="flex-shrink-0 mt-1">
                  {msg.role === "user" ? (
                    <div className="w-8 h-8 rounded-full bg-[#1A1A1A] border border-gray-700 flex items-center justify-center text-sm font-bold text-white">
                      U
                    </div>
                  ) : (
                    <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center text-black">
                      <Bot className="h-5 w-5" />
                    </div>
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium mb-1 text-gray-400 capitalize">{msg.role}</div>
                  <div className="text-gray-200 whitespace-pre-wrap leading-relaxed">{msg.content}</div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} className="h-4" />
          </div>
        </div>
      </div>

      <div className="flex-shrink-0 p-4 flex justify-center bg-gradient-to-t from-black via-black to-transparent absolute bottom-0 left-0 right-0">
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
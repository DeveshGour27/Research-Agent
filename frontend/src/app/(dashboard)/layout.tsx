
"use client";

import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { api } from "@/lib/api";
import Link from "next/link";
import { Plus, Search, MessageSquare, Settings, Sun, LogOut } from "lucide-react";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [chats, setChats] = useState<any[]>([]); // eslint-disable-line @typescript-eslint/no-explicit-any
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const pathname = usePathname();

  const [user, setUser] = useState<{email?: string} | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [chatsData, userData] = await Promise.all([
          api.getChats(),
          api.getMe()
        ]);
        setChats(chatsData);
        setUser(userData);
      } catch (err: unknown) {
        if (err instanceof Error && 'status' in err && (err.status === 401 || err.status === 403)) {
          router.push("/login");
        }
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [pathname, router]);

  const handleLogout = async () => {
    try {
      await api.logout();
      router.push("/login");
    } catch (e) {}
  };

  const handleNewChat = async () => {
    try {
      const chat = await api.createChat();
      router.push(`/chat/${chat.chat_id}`);
    } catch (e) {
      alert("Failed to create chat");
    }
  };

  if (loading) return <div className="min-h-screen bg-black flex items-center justify-center text-white">Loading...</div>;

  return (
    <div className="flex h-screen bg-black text-white">
      {/* Sidebar */}
      <div className="w-[260px] bg-[#0A0A0A] border-r border-gray-800 flex flex-col flex-shrink-0">
        <div className="p-4 flex items-center space-x-3">
          <div className="w-8 h-8 rounded-full bg-white text-black flex items-center justify-center font-bold text-sm">
            AI
          </div>
          <span className="font-semibold flex-1">AI Assistant</span>
        </div>
        
        <div className="px-4 pb-4">
          <div className="relative">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-500" />
            <input 
              type="text" 
              placeholder="Search..." 
              className="w-full bg-[#1A1A1A] text-sm text-gray-300 rounded-full pl-9 pr-4 py-2 outline-none border border-transparent focus:border-gray-700 transition"
            />
          </div>
        </div>

        <div className="px-4 mb-4">
          <button 
            onClick={handleNewChat}
            className="w-full bg-white text-black flex items-center justify-center space-x-2 py-2.5 rounded-full font-medium hover:bg-gray-200 transition"
          >
            <Plus className="h-4 w-4" />
            <span>Start New Chat</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3">
          <div className="text-xs font-medium text-gray-500 px-2 py-2 mb-1 flex items-center uppercase tracking-wider">
            Recent
          </div>
          <div className="space-y-1">
            {chats.map(chat => {
              const isActive = pathname === `/chat/${chat.chat_id}`;
              return (
                <Link key={chat.chat_id} href={`/chat/${chat.chat_id}`} className={`block p-3 rounded-xl transition ${isActive ? "bg-[#1A1A1A]" : "hover:bg-[#1A1A1A]"}`}>
                  <div className="text-sm font-medium text-gray-200 truncate">{chat.title || "New Chat"}</div>
                  <div className="text-xs text-gray-500 mt-1 flex justify-between">
                    <span>{new Date(chat.updated_at).toLocaleDateString()}</span>
                  </div>
                </Link>
              );
            })}
            {chats.length === 0 && (
              <div className="text-sm text-gray-500 px-2 py-4">No recent chats</div>
            )}
          </div>
        </div>

        <div className="p-4 border-t border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <button className="flex items-center space-x-2 text-sm text-gray-400 hover:text-white transition">
              <Settings className="h-4 w-4" />
              <span>Settings</span>
            </button>
            <button className="flex items-center space-x-2 text-sm text-gray-400 hover:text-white transition bg-[#1A1A1A] px-3 py-1.5 rounded-full">
              <Sun className="h-4 w-4" />
              <span>Light</span>
            </button>
          </div>
          
          <div className="flex items-center justify-between group cursor-pointer">
            <div className="flex items-center space-x-3 truncate">
              <div className="w-8 h-8 rounded-full bg-[#1A1A1A] border border-gray-700 flex items-center justify-center text-sm font-bold flex-shrink-0">
                {user?.email?.[0]?.toUpperCase() || "U"}
              </div>
              <div className="truncate">
                <div className="text-sm font-medium truncate" title={user?.email}>{user?.email || "User"}</div>
                <div className="text-xs text-gray-500">Member</div>
              </div>
            </div>
            <button onClick={handleLogout} className="text-gray-500 hover:text-white transition opacity-0 group-hover:opacity-100">
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {children}
      </div>
    </div>
  );
}


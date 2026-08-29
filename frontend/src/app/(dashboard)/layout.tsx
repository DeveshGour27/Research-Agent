"use client";

import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { api } from "@/lib/api";
import Link from "next/link";
import { Plus, Search, Settings, Sun, LogOut, X } from "lucide-react";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [chats, setChats] = useState<any[]>([]); // eslint-disable-line @typescript-eslint/no-explicit-any
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const pathname = usePathname();

  const [user, setUser] = useState<{email?: string, username?: string} | null>(null);
  
  // Settings modal state
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [settingsMsg, setSettingsMsg] = useState({ text: "", isError: false });

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
        if (err instanceof Error && 'status' in err) {
          const status = (err as {status?: number}).status;
          if (status === 401 || status === 403) {
            router.push("/login");
          }
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
    } catch (e: unknown) {
      console.error(e);
    }
  };

  const handleNewChat = async () => {
    try {
      const chat = await api.createChat();
      router.push(`/chat/${chat.chat_id}`);
    } catch (e: unknown) {
      alert("Failed to create chat");
      console.error(e);
    }
  };
  
  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setSettingsMsg({ text: "Updating...", isError: false });
    try {
      await api.changePassword({ old_password: oldPassword, new_password: newPassword });
      setSettingsMsg({ text: "Password changed successfully!", isError: false });
      setOldPassword("");
      setNewPassword("");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setSettingsMsg({ text: msg || "Failed to change password", isError: true });
    }
  };
  
  const handleDeleteAccount = async () => {
    if (!confirm("Are you sure you want to permanently delete your account? This action cannot be undone.")) return;
    try {
      await api.deleteAccount();
      router.push("/login");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setSettingsMsg({ text: msg || "Failed to delete account", isError: true });
    }
  };

  if (loading) return <div className="min-h-screen bg-black flex items-center justify-center text-white">Loading...</div>;

  return (
    <div className="flex h-screen bg-black text-white">
      {/* Settings Modal */}
      {isSettingsOpen && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center">
          <div className="bg-[#0f0f0f] border border-gray-800 rounded-xl w-full max-w-md p-6 relative">
            <button onClick={() => setIsSettingsOpen(false)} className="absolute top-4 right-4 text-gray-500 hover:text-white">
              <X className="w-5 h-5" />
            </button>
            <h2 className="text-xl font-bold mb-6">Settings</h2>
            
            {settingsMsg.text && (
              <div className={`p-3 rounded mb-4 text-sm ${settingsMsg.isError ? 'bg-red-900/50 text-red-200' : 'bg-green-900/50 text-green-200'}`}>
                {settingsMsg.text}
              </div>
            )}
            
            <form onSubmit={handleChangePassword} className="space-y-4 mb-8">
              <h3 className="text-sm font-medium text-gray-400">Change Password</h3>
              <div>
                <input 
                  type="password" 
                  placeholder="Old Password" 
                  required
                  value={oldPassword}
                  onChange={e => setOldPassword(e.target.value)}
                  className="w-full p-2 bg-black border border-gray-700 rounded text-white focus:border-white outline-none text-sm"
                />
              </div>
              <div>
                <input 
                  type="password" 
                  placeholder="New Password (min 8 characters)" 
                  required
                  minLength={8}
                  value={newPassword}
                  onChange={e => setNewPassword(e.target.value)}
                  className="w-full p-2 bg-black border border-gray-700 rounded text-white focus:border-white outline-none text-sm"
                />
              </div>
              <button type="submit" className="bg-white text-black text-sm font-medium py-2 px-4 rounded hover:bg-gray-200 transition">
                Update Password
              </button>
            </form>
            
            <div className="border-t border-gray-800 pt-6">
              <h3 className="text-sm font-medium text-red-400 mb-2">Danger Zone</h3>
              <button onClick={handleDeleteAccount} className="w-full text-left p-3 border border-red-900/50 bg-red-900/10 text-red-400 rounded hover:bg-red-900/20 transition text-sm">
                Delete Account...
              </button>
            </div>
          </div>
        </div>
      )}

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
            <button onClick={() => setIsSettingsOpen(true)} className="flex items-center space-x-2 text-sm text-gray-400 hover:text-white transition">
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
                {user?.username?.[0]?.toUpperCase() || user?.email?.[0]?.toUpperCase() || "U"}
              </div>
              <div className="truncate">
                <div className="text-sm font-medium truncate" title={user?.email}>{user?.username || user?.email || "User"}</div>
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

"use client";
import { useState, Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

function ResetPasswordContent() {
  const searchParams = useSearchParams();
  const token = searchParams.get("token") || "";
  const email = searchParams.get("email") || "";
  
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [msg, setMsg] = useState({ text: "", isError: false });
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      setMsg({ text: "Passwords do not match", isError: true });
      return;
    }
    
    setLoading(true);
    try {
      const res = await fetch("/api/v1/auth/reset-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, token, password })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to reset");
      setMsg({ text: "Password reset successful!", isError: false });
      setSuccess(true);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setMsg({ text: message || "Failed to reset password", isError: true });
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-black">
        <div className="w-full max-w-md p-8 bg-[#0f0f0f] rounded-xl border border-gray-800 text-center">
          <h1 className="text-xl font-bold mb-4 text-white">Password Reset!</h1>
          <Link href="/login" className="text-white hover:underline bg-white/10 px-4 py-2 rounded">Go to Login</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-black">
      <div className="w-full max-w-md p-8 bg-[#0f0f0f] rounded-xl border border-gray-800">
        <h1 className="text-2xl font-bold mb-6 text-center text-white">Set New Password</h1>
        {msg.text && <div className={`p-3 rounded mb-4 text-sm ${msg.isError ? 'bg-red-900/50 text-red-200' : 'bg-green-900/50 text-green-200'}`}>{msg.text}</div>}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">New Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
              className="w-full p-2 bg-black border border-gray-700 rounded text-white focus:border-white outline-none"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Confirm New Password</label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              minLength={8}
              className="w-full p-2 bg-black border border-gray-700 rounded text-white focus:border-white outline-none"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-white text-black font-medium py-2 rounded hover:bg-gray-200 transition disabled:opacity-50"
          >
            {loading ? "Resetting..." : "Reset Password"}
          </button>
        </form>
      </div>
    </div>
  );
}

export default function ResetPassword() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-black"></div>}>
      <ResetPasswordContent />
    </Suspense>
  );
}

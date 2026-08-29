
"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import Link from "next/link";

export default function Signup() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await api.signup({ email, password });
      setSuccess(true);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message || "Failed to sign up");
      } else {
        setError("Failed to sign up");
      }
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-black">
        <div className="w-full max-w-md p-8 bg-[#0f0f0f] rounded-xl border border-gray-800 text-center">
          <h1 className="text-xl font-bold mb-4 text-white">Check Your Email</h1>
          <p className="text-gray-400 mb-6">We have sent a verification link to {email}.</p>
          <Link href="/login" className="text-white hover:underline">Return to login</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-black">
      <div className="w-full max-w-md p-8 bg-[#0f0f0f] rounded-xl border border-gray-800">
        <h1 className="text-2xl font-bold mb-6 text-center text-white">Sign Up</h1>
        {error && <div className="bg-red-900/50 text-red-200 p-3 rounded mb-4 text-sm">{error}</div>}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full p-2 bg-black border border-gray-700 rounded text-white focus:border-white outline-none"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
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
            {loading ? "Signing up..." : "Sign Up"}
          </button>
        </form>
        <p className="mt-4 text-center text-sm text-gray-500">
          Already have an account? <Link href="/login" className="text-white hover:underline">Log in</Link>
        </p>
      </div>
    </div>
  );
}


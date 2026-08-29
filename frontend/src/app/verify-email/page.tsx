
"use client";
import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import Link from "next/link";
import { Suspense } from "react";

function VerifyEmailContent() {
  const searchParams = useSearchParams();
  const token = searchParams.get("token");
  const email = searchParams.get("email");
  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    if (!token || !email) {
      setTimeout(() => {
        setStatus("error");
        setErrorMsg("Missing verification token or email.");
      }, 0);
      return;
    }

    const verify = async () => {
      try {
        await api.verify({ token, email });
        setStatus("success");
      } catch (err: unknown) {
        setStatus("error");
        if (err instanceof Error) {
          setErrorMsg(err.message || "Invalid or expired token.");
        } else {
          setErrorMsg("Invalid or expired token.");
        }
      }
    };
    verify();
  }, [token, email]);

  return (
    <div className="flex items-center justify-center min-h-screen bg-black">
      <div className="w-full max-w-md p-8 bg-[#0f0f0f] rounded-xl border border-gray-800 text-center">
        {status === "loading" && <p className="text-gray-400">Verifying email...</p>}
        {status === "success" && (
          <>
            <h1 className="text-xl font-bold mb-4 text-green-400">Email Verified!</h1>
            <p className="text-gray-400 mb-6">Your account is now ready.</p>
            <Link href="/login" className="bg-white text-black px-4 py-2 rounded hover:bg-gray-200 inline-block font-medium">Go to Login</Link>
          </>
        )}
        {status === "error" && (
          <>
            <h1 className="text-xl font-bold mb-4 text-red-500">Verification Failed</h1>
            <p className="text-gray-400 mb-6">{errorMsg}</p>
            <Link href="/signup" className="text-white hover:underline">Return to sign up</Link>
          </>
        )}
      </div>
    </div>
  );
}

export default function VerifyEmail() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-black" />}>
      <VerifyEmailContent />
    </Suspense>
  );
}


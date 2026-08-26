import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const BACKEND_URL = process.env.BACKEND_API_URL || "http://localhost:8000";
const BACKEND_KEY = process.env.BACKEND_API_KEY;

async function proxyRequest(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  if (!BACKEND_KEY) {
    return new Response(
      JSON.stringify({ error: "Server configuration error: BACKEND_API_KEY is not set." }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }

  const { path } = await params;
  const pathString = path.join("/");
  const searchParams = req.nextUrl.searchParams;
  const targetUrl = new URL(`${BACKEND_URL}/api/v1/${pathString}`);
  targetUrl.search = searchParams.toString();

  const headers = new Headers(req.headers);
  
  // Strip hop-by-hop and host headers
  headers.delete("host");
  headers.delete("connection");
  headers.delete("content-length");
  headers.delete("transfer-encoding");
  
  // CRITICAL: Prevent browser from overriding the server credential
  headers.set("X-API-Key", BACKEND_KEY);

  try {
    const fetchOptions: RequestInit & { duplex?: "half" } = {
      method: req.method,
      headers,
    };

    if (req.method !== "GET" && req.method !== "HEAD") {
      fetchOptions.body = req.body;
      fetchOptions.duplex = "half";
    }

    const backendResponse = await fetch(targetUrl.toString(), fetchOptions);

    const responseHeaders = new Headers(backendResponse.headers);
    responseHeaders.delete("content-encoding");

    return new Response(backendResponse.body, {
      status: backendResponse.status,
      statusText: backendResponse.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    console.error("BFF Proxy Error:", error);
    return new Response(
      JSON.stringify({ error: "Internal Server Error: Unable to reach backend." }),
      { status: 502, headers: { "Content-Type": "application/json" } }
    );
  }
}

export const GET = proxyRequest;
export const POST = proxyRequest;
export const PUT = proxyRequest;
export const DELETE = proxyRequest;
export const PATCH = proxyRequest;
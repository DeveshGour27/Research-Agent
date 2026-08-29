const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";

export class APIError extends Error {
  constructor(public status: number, public message: string, public code?: string) {
    super(message);
    this.name = "APIError";
  }
}

async function fetchAPI(endpoint: string, options: RequestInit = {}) {
  const url = `${BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    credentials: "include", // essential for cookies
  });

  if (response.status === 204) {
    return null;
  }

  let data;
  try {
    data = await response.json();
  } catch {
    if (!response.ok) {
      throw new APIError(response.status, "An unexpected error occurred");
    }
    return null;
  }

  if (!response.ok) {
    const errorMsg = data.detail || (data.error && data.error.message) || "API Error";
    const errorCode = data.error && data.error.code;
    throw new APIError(response.status, errorMsg, errorCode);
  }

  return data;
}

export const api = {
  login: (data: Record<string, unknown>) => fetchAPI("/api/v1/auth/login", { method: "POST", body: JSON.stringify(data) }),
  signup: (data: Record<string, unknown>) => fetchAPI("/api/v1/auth/signup", { method: "POST", body: JSON.stringify(data) }),
  verify: (data: Record<string, unknown>) => fetchAPI("/api/v1/auth/verify", { method: "POST", body: JSON.stringify(data) }),
  logout: () => fetchAPI("/api/v1/auth/logout", { method: "POST" }),
  getMe: () => fetchAPI("/api/v1/auth/me"),
  getGoogleAuthUrl: () => fetchAPI("/api/v1/auth/google/login"),
  changePassword: (data: Record<string, unknown>) => fetchAPI("/api/v1/auth/change-password", { method: "POST", body: JSON.stringify(data) }),
  deleteAccount: () => fetchAPI("/api/v1/auth/delete-account", { method: "POST" }),
  
  getChats: () => fetchAPI("/api/v1/chats"),
  createChat: () => fetchAPI("/api/v1/chats", { method: "POST" }),
  getChat: (chatId: string) => fetchAPI(`/api/v1/chats/${chatId}`),
  deleteChat: (chatId: string) => fetchAPI(`/api/v1/chats/${chatId}`, { method: "DELETE" }),
  
  sendMessage: (chatId: string, content: string) => 
    fetchAPI(`/api/v1/chats/${chatId}/messages`, { method: "POST", body: JSON.stringify({ content }) }),
};


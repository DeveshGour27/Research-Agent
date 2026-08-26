export class ApiError extends Error {
  status: number;
  code?: string;
  details?: unknown;

  constructor(status: number, message: string, code?: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

interface RequestOptions extends RequestInit {
  params?: Record<string, string>;
}

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "";

export async function fetchClient<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const { params, headers, ...customConfig } = options;

  let url = `${API_BASE_URL}${endpoint}`;
  if (params) {
    const searchParams = new URLSearchParams(params);
    url += `?${searchParams.toString()}`;
  }

  const config: RequestInit = {
    ...customConfig,
    headers: {
      "Content-Type": "application/json",
      ...headers,
    },
  };

  try {
    const response = await fetch(url, config);

    if (!response.ok) {
      let errorMessage = "An error occurred while communicating with the server.";
      let errorCode = "UNKNOWN_ERROR";
      let errorDetails = undefined;

      try {
        const errorData = await response.json();
        if (errorData?.error) {
          errorMessage = errorData.error.message || errorMessage;
          errorCode = errorData.error.code || errorCode;
          errorDetails = errorData.error.request_id;
        } else if (errorData?.detail) {
          errorMessage = Array.isArray(errorData.detail) 
            ? errorData.detail.map((errItem: { msg: string }) => errItem.msg).join(", ")
            : errorData.detail;
        }
      } catch {
        // Response was not JSON
        errorMessage = response.statusText || errorMessage;
      }

      throw new ApiError(response.status, errorMessage, errorCode, errorDetails);
    }

    // Some endpoints might return 204 No Content
    if (response.status === 204) {
      return {} as T;
    }

    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    // Network errors or parsing errors
    throw new ApiError(
      0,
      error instanceof Error ? error.message : "Network error. Unable to reach the service.",
      "NETWORK_ERROR"
    );
  }
}

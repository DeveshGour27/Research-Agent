import { API_BASE_URL } from "./client";

export type ConnectionState =
  | "DISCONNECTED"
  | "CONNECTING"
  | "CONNECTED"
  | "RECONNECTING"
  | "COMPLETED"
  | "FAILED"
  | "CLOSED";

export interface ResearchEvent {
  event_id: string;
  job_id: string;
  event_type: string;
  timestamp: string;
  sequence: number;
  source: string;
  payload: Record<string, unknown> & {
    request_type?: string;
    component_name?: string;
    status?: string;
    goal?: string;
    error?: string;
    tool_name?: string;
  };
}

interface SSEControllerOptions {
  jobId: string;
  onEvent: (event: ResearchEvent) => void;
  onStateChange: (state: ConnectionState) => void;
  maxRetries?: number;
}

export class ResearchSSEController {
  private jobId: string;
  private onEvent: (event: ResearchEvent) => void;
  private onStateChange: (state: ConnectionState) => void;
  private maxRetries: number;

  private state: ConnectionState = "DISCONNECTED";
  private abortController: AbortController | null = null;
  private retryCount = 0;
  private lastEventId: string | null = null;
  private isIntentionalClose = false;

  constructor(options: SSEControllerOptions) {
    this.jobId = options.jobId;
    this.onEvent = options.onEvent;
    this.onStateChange = options.onStateChange;
    this.maxRetries = options.maxRetries ?? 5;
  }

  private setState(newState: ConnectionState) {
    if (this.state === newState) return;
    this.state = newState;
    this.onStateChange(newState);
  }

  public connect() {
    if (this.state === "CONNECTED" || this.state === "CONNECTING" || this.state === "RECONNECTING") {
      return; // Prevent duplicate connections
    }
    
    this.isIntentionalClose = false;
    this.setState(this.retryCount === 0 ? "CONNECTING" : "RECONNECTING");
    this.startFetch();
  }

  public disconnect() {
    this.isIntentionalClose = true;
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
    this.setState("CLOSED");
  }

  private async startFetch() {
    this.abortController = new AbortController();
    
    try {
      const headers: Record<string, string> = {
        "Accept": "text/event-stream",
      };
      
      if (this.lastEventId) {
        headers["Last-Event-ID"] = this.lastEventId;
      }

      const response = await fetch(`${API_BASE_URL}/api/v1/jobs/${this.jobId}/events/stream`, {
        method: "GET",
        headers,
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      if (!response.body) {
        throw new Error("ReadableStream not supported in this browser.");
      }

      this.setState("CONNECTED");
      this.retryCount = 0; // Reset retries on successful connection

      await this.readStream(response.body.getReader());
    } catch (err: unknown) {
      const error = err as Error;
      if (this.isIntentionalClose || error.name === "AbortError") {
        return; // Normal cleanup
      }

      if (this.retryCount < this.maxRetries) {
        this.retryCount++;
        this.setState("DISCONNECTED");
        const backoffMs = Math.min(1000 * Math.pow(2, this.retryCount), 10000);
        setTimeout(() => this.connect(), backoffMs);
      } else {
        this.setState("FAILED");
      }
    }
  }

  private async readStream(reader: ReadableStreamDefaultReader<Uint8Array>) {
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        // SSE boundaries are blank lines (\n\n or \r\n\r\n)
        let boundaryIndex;
        while ((boundaryIndex = buffer.indexOf("\n\n")) !== -1 || (boundaryIndex = buffer.indexOf("\r\n\r\n")) !== -1) {
          const isCrLf = buffer.startsWith("\r\n\r\n", boundaryIndex);
          const blockLen = boundaryIndex + (isCrLf ? 4 : 2);
          
          const eventBlock = buffer.slice(0, boundaryIndex);
          buffer = buffer.slice(blockLen);

          this.parseEventBlock(eventBlock);
        }
      }
    } finally {
      reader.releaseLock();
    }
    
    // If stream ended cleanly without intentional close, it means backend closed it.
    // In our backend contract, it closes when terminal state is reached.
    if (!this.isIntentionalClose && this.state !== "FAILED") {
       this.setState("COMPLETED");
    }
  }

  private parseEventBlock(block: string) {
    const lines = block.split(/\r?\n/);
    let dataBuffer = "";

    for (const line of lines) {
      if (line.startsWith(":")) continue; // Comment

      const colonIndex = line.indexOf(":");
      if (colonIndex === -1) continue;

      const field = line.slice(0, colonIndex).trim();
      const value = line.slice(colonIndex + 1).trim();

      if (field === "id") {
        this.lastEventId = value;
      } else if (field === "data") {
        dataBuffer += dataBuffer ? "\n" + value : value;
      }
    }

    if (dataBuffer) {
      try {
        const payload: ResearchEvent = JSON.parse(dataBuffer);
        this.onEvent(payload);
        
        // Terminal states check
        if (payload.event_type === "JOB_COMPLETED" || payload.event_type === "JOB_FAILED" || payload.event_type === "JOB_CANCELLED") {
            this.isIntentionalClose = true;
            if (this.abortController) {
                this.abortController.abort();
                this.abortController = null;
            }
            this.setState("COMPLETED");
        }
      } catch (err) {
        console.error("Failed to parse SSE JSON data:", err, dataBuffer);
      }
    }
  }
}

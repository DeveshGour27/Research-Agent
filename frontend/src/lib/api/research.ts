import { fetchClient } from "./client";

export interface CreateResearchRequest {
  goal: string;
  chat_id?: string | null;
}

export interface CreateResearchResponse {
  job_id: string;
  status: string;
  created_at: string;
}

/**
 * Submit a new research investigation.
 * POST /api/v1/research
 */
export async function createResearchJob(data: CreateResearchRequest): Promise<CreateResearchResponse> {
  return fetchClient<CreateResearchResponse>("/api/v1/research", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export interface ResearchJobStatusResponse {
  job_id: string;
  status: string;
  result: string | null;
  error: string | null;
  execution_time_seconds: number | null;
}

/**
 * Get job status.
 * GET /api/v1/research/jobs/{job_id}
 */
export async function getResearchJobStatus(jobId: string): Promise<ResearchJobStatusResponse> {
  return fetchClient<ResearchJobStatusResponse>(`/api/v1/research/jobs/${jobId}`, {
    method: "GET",
  });
}

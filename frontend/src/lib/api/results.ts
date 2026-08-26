import { fetchClient } from "./client";

export interface ResearchResultResponse {
  job_id: string;
  status: string;
  result: string | null;
  error: string | null;
  completed_at: string | null;
}

/**
 * Retrieve the final result of a completed research job.
 * GET /api/v1/research/{job_id}/result
 */
export async function getResearchResult(jobId: string): Promise<ResearchResultResponse> {
  return fetchClient<ResearchResultResponse>(`/api/v1/research/${jobId}/result`, {
    method: "GET",
  });
}

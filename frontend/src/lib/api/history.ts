import { fetchClient } from "./client";

export interface ResearchJobHistoryItem {
  job_id: string;
  goal: string;
  status: string;
  created_at: string;
  completed_at: string | null;
}

export interface ResearchJobHistoryResponse {
  items: ResearchJobHistoryItem[];
  total: number;
  limit: number;
  offset: number;
}

/**
 * Retrieve paginated job history for the authenticated user.
 * GET /api/v1/research/jobs
 */
export async function getResearchHistory(limit: number = 20, offset: number = 0): Promise<ResearchJobHistoryResponse> {
  return fetchClient<ResearchJobHistoryResponse>(`/api/v1/research/jobs?limit=${limit}&offset=${offset}`, {
    method: "GET",
  });
}

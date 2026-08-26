import { fetchClient } from "./client";

export interface HITLRequestResponse {
  request_id: string;
  request_type: string;
  component_name: string;
  status: string;
  created_at: string;
  expires_at: string | null;
  decided_at: string | null;
  decided_by: string | null;
  decision_reason: string | null;
}

export interface HITLRequestsListResponse {
  job_id: string;
  hitl_requests: HITLRequestResponse[];
}

/**
 * Get all HITL requests for a job.
 * GET /api/v1/research/{job_id}/hitl
 */
export async function getHitlRequests(jobId: string): Promise<HITLRequestsListResponse> {
  return fetchClient<HITLRequestsListResponse>(`/api/v1/research/${jobId}/hitl`, {
    method: "GET",
  });
}

/**
 * Approve a HITL request.
 * POST /api/v1/research/{job_id}/hitl/{request_id}/approve
 */
export async function approveHitlRequest(jobId: string, requestId: string): Promise<{ status: string; message: string }> {
  return fetchClient<{ status: string; message: string }>(`/api/v1/research/${jobId}/hitl/${requestId}/approve`, {
    method: "POST",
  });
}

/**
 * Reject a HITL request.
 * POST /api/v1/research/{job_id}/hitl/{request_id}/reject
 */
export async function rejectHitlRequest(jobId: string, requestId: string): Promise<{ status: string; message: string }> {
  return fetchClient<{ status: string; message: string }>(`/api/v1/research/${jobId}/hitl/${requestId}/reject`, {
    method: "POST",
  });
}
